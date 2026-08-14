#!/usr/bin/env python3
"""
test_dispatcher.py - the CrossTrade signal dispatcher and incubator bridge.

Location:  ~/src/trading/tests/test_dispatcher.py

Run:  python tests/test_dispatcher.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure. No network is touched: `live.dispatcher._urlopen` is
swapped for a fake opener, and the log fixtures are written to a temp dir.

The central claims
------------------
1. A malformed order never leaves the process. An unknown action, a fractional
   or non-positive quantity, and a price-bearing order type all raise at
   `format_crosstrade_payload` rather than being forwarded to a broker.
2. A failed send is RECORDED, not raised. A 500 and a socket timeout both come
   back as `ok=False` result dicts carrying latency and a reason, because the
   dispatcher's job is to leave a record of what it attempted.
3. Slippage is measured in ticks against `backtest.specs`, never assumed. A
   symbol with no spec is excluded and reported - it does not quietly get a
   default tick size, which would turn a 1-tick fill into a fake outlier.
4. An unmeasurable slippage is None, not 0.0. Zero reads as perfect execution.
"""

from __future__ import annotations

import io
import json
import socket
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.specs import get_spec  # noqa: E402
from live import dispatcher  # noqa: E402
from live.dispatcher import (  # noqa: E402
    evaluate_incubator_sync, format_crosstrade_payload, load_config,
    send_execution_signal,
)

FAILURES: list[str] = []
LIVE_URL = "https://api.crosstrade.io/webhook/abc123"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# fake transport
# --------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    """Records the request it was handed, then returns or raises what it was told."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.request = None
        self.timeout = None

    def __call__(self, req, timeout=None):
        self.request = req
        self.timeout = timeout
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def with_opener(outcome):
    """Install a fake opener and return it. Callers restore() in a finally block
    so a failed assertion cannot leave the stdlib opener swapped out."""
    opener = _Opener(outcome)
    dispatcher._urlopen = opener
    return opener


def restore(original):
    dispatcher._urlopen = original


# --------------------------------------------------------------------------
# 1. payload formatting
# --------------------------------------------------------------------------

def test_payload_shape():
    print("\nPAYLOAD STRUCTURE")
    p = format_crosstrade_payload("es", "buy", 2, account_id="Sim101")
    check("exact schema and key order",
          list(p.keys()) == ["account", "action", "symbol", "orderType", "quantity"],
          str(list(p.keys())))
    check("values upper-cased, quantity int",
          p == {"account": "Sim101", "action": "BUY", "symbol": "ES",
                "orderType": "MARKET", "quantity": 2}, json.dumps(p))
    check("quantity is a real int, not a str",
          isinstance(p["quantity"], int) and not isinstance(p["quantity"], bool))
    check("payload is JSON-serialisable", json.loads(json.dumps(p)) == p)

    p2 = format_crosstrade_payload("  nq  ", "SeLl", "3", order_type="market",
                                   account_id="Sim101")
    check("whitespace stripped, mixed case normalised, numeric string coerced",
          p2["symbol"] == "NQ" and p2["action"] == "SELL" and p2["quantity"] == 3,
          json.dumps(p2))

    check("account defaults to live/config.json",
          format_crosstrade_payload("ES", "BUY", 1)["account"]
          == load_config()["account_id"])

    check("FLATTEN is accepted",
          format_crosstrade_payload("ES", "flatten", 1, account_id="X")["action"]
          == "FLATTEN")


def test_payload_rejections():
    print("\nPAYLOAD VALIDATION (nothing malformed reaches the broker)")
    bad = [
        ("unknown action", ("ES", "LONG", 1), {}),
        ("empty action", ("ES", "", 1), {}),
        ("empty symbol", ("   ", "BUY", 1), {}),
        ("zero quantity", ("ES", "BUY", 0), {}),
        ("negative quantity", ("ES", "BUY", -1), {}),
        ("fractional quantity", ("ES", "BUY", 1.5), {}),
        ("bool quantity", ("ES", "BUY", True), {}),
        ("non-numeric quantity", ("ES", "BUY", "two"), {}),
        ("LIMIT without a price", ("ES", "BUY", 1), {"order_type": "LIMIT"}),
        ("STOP without a price", ("ES", "BUY", 1), {"order_type": "STOP"}),
    ]
    for name, args, kw in bad:
        kw = dict(kw, account_id="Sim101")
        try:
            format_crosstrade_payload(*args, **kw)
            check(f"rejects {name}", False, "no exception raised")
        except ValueError:
            check(f"rejects {name}", True)
        except Exception as exc:
            check(f"rejects {name}", False, f"raised {type(exc).__name__}")


# --------------------------------------------------------------------------
# 2. dispatch
# --------------------------------------------------------------------------

def test_send_success():
    print("\nDISPATCH - 200")
    original = dispatcher._urlopen
    payload = format_crosstrade_payload("ES", "BUY", 1, account_id="Sim101")
    opener = with_opener(_FakeResponse(200, b'{"status":"accepted"}'))
    try:
        res = send_execution_signal(payload, webhook_url=LIVE_URL, timeout_seconds=2.0)
    finally:
        restore(original)

    check("ok on 200", res["ok"] is True and res["http_status"] == 200, str(res["error"]))
    check("body captured", res["response_body"] == '{"status":"accepted"}')
    check("no error on success", res["error"] is None)
    check("latency recorded", isinstance(res["latency_ms"], float) and res["latency_ms"] >= 0)
    check("timestamp is UTC ISO-8601",
          res["timestamp"].endswith("+00:00"), res["timestamp"])
    check("POSTed with JSON content-type",
          opener.request.get_method() == "POST"
          and opener.request.headers.get("Content-type") == "application/json")
    check("body on the wire is the payload",
          json.loads(opener.request.data.decode()) == payload)
    check("timeout passed to the transport", opener.timeout == 2.0, str(opener.timeout))


def test_send_http_error():
    print("\nDISPATCH - 500")
    original = dispatcher._urlopen
    err = urllib.error.HTTPError(LIVE_URL, 500, "Internal Server Error", {},
                                 io.BytesIO(b"upstream exploded"))
    with_opener(err)
    try:
        res = send_execution_signal({"account": "Sim101", "action": "BUY"},
                                    webhook_url=LIVE_URL, timeout_seconds=2.0)
    finally:
        restore(original)

    check("does not raise on 500", True)
    check("ok is False", res["ok"] is False)
    check("status recorded", res["http_status"] == 500, str(res["http_status"]))
    check("error explains the failure", "500" in (res["error"] or ""), str(res["error"]))
    check("error body captured", res["response_body"] == "upstream exploded")
    check("attempted payload retained", res["payload"]["action"] == "BUY")

    # A 3xx/4xx must not be mistaken for success either.
    err4 = urllib.error.HTTPError(LIVE_URL, 403, "Forbidden", {}, io.BytesIO(b"nope"))
    with_opener(err4)
    try:
        res4 = send_execution_signal({"a": 1}, webhook_url=LIVE_URL, timeout_seconds=2.0)
    finally:
        restore(original)
    check("403 is not ok", res4["ok"] is False and res4["http_status"] == 403)


def test_send_timeout():
    print("\nDISPATCH - timeout and network failure")
    original = dispatcher._urlopen

    for label, exc in (
        ("socket.timeout", socket.timeout("timed out")),
        ("TimeoutError", TimeoutError("timed out")),
        ("URLError wrapping a timeout", urllib.error.URLError(socket.timeout("timed out"))),
    ):
        with_opener(exc)
        try:
            res = send_execution_signal({"a": 1}, webhook_url=LIVE_URL, timeout_seconds=2.0)
        finally:
            restore(original)
        check(f"{label} -> ok=False, no status",
              res["ok"] is False and res["http_status"] is None)
        check(f"{label} -> reported as a timeout",
              "timeout" in (res["error"] or "").lower(), str(res["error"]))
        check(f"{label} -> latency still recorded", res["latency_ms"] >= 0)

    with_opener(urllib.error.URLError(ConnectionRefusedError("refused")))
    try:
        res = send_execution_signal({"a": 1}, webhook_url=LIVE_URL, timeout_seconds=2.0)
    finally:
        restore(original)
    check("connection refused -> network error, not a timeout",
          res["ok"] is False and "network error" in (res["error"] or ""), str(res["error"]))


def test_placeholder_is_not_dispatchable():
    print("\nDISPATCH - guards")
    original = dispatcher._urlopen
    opener = with_opener(_FakeResponse(200, b"ok"))
    try:
        res = send_execution_signal({"a": 1},
                                    webhook_url="https://api.crosstrade.io/webhook/placeholder",
                                    timeout_seconds=2.0)
    finally:
        restore(original)
    check("placeholder URL is refused", res["ok"] is False and "placeholder" in res["error"])
    check("nothing was sent", opener.request is None)

    opener = with_opener(_FakeResponse(200, b"ok"))
    try:
        res = send_execution_signal({}, webhook_url=LIVE_URL, timeout_seconds=2.0)
    finally:
        restore(original)
    check("empty payload is refused", res["ok"] is False and opener.request is None)

    cfg = load_config()
    check("shipped config still carries the required keys and a 2.0s timeout",
          float(cfg["timeout_seconds"]) == 2.0 and cfg["account_id"] == "Sim101"
          and cfg["environment"] == "incubator_sim")


# --------------------------------------------------------------------------
# 3. incubator log parsing
# --------------------------------------------------------------------------

CSV_FIXTURE = """\
timestamp,symbol,action,quantity,signal_price,realized_price,status
2026-08-11T14:30:00Z,ES,BUY,1,5000.00,5000.25,Filled
2026-08-11T15:00:00Z,ES,SELL,1,5010.00,5009.50,Filled
2026-08-11T15:30:00Z,ES,BUY,1,5005.00,,Rejected
2026-08-12T14:30:00Z,NQ,BUY,1,17000.00,17001.00,Filled
2026-08-12T15:30:00Z,ZZZ,BUY,1,100.00,100.50,Filled
"""


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def test_log_parsing():
    print("\nINCUBATOR SYNC - CSV fixture")
    es_tick = get_spec("ES").tick_size          # 0.25
    nq_tick = get_spec("NQ").tick_size          # 0.25

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        res = evaluate_incubator_sync(_write(tmp, "fills.csv", CSV_FIXTURE))

    # 5 signals, 4 with a fill; ZZZ has no spec so it is counted but not measured.
    expected_slips = [0.25 / es_tick, 0.50 / es_tick, 1.00 / nq_tick]
    expected_mean = round(sum(expected_slips) / len(expected_slips), 4)

    check("every row is a signal", res["n_signals"] == 5, str(res["n_signals"]))
    check("rejected row is not a fill", res["n_filled"] == 4, str(res["n_filled"]))
    check("fill success rate", abs(res["fill_success_rate"] - 0.8) < 1e-9,
          str(res["fill_success_rate"]))
    check("mean slippage in ticks", res["mean_slippage_ticks"] == expected_mean,
          f"{res['mean_slippage_ticks']} vs {expected_mean}")
    check("max slippage in ticks", res["max_slippage_ticks"] == 4.0,
          str(res["max_slippage_ticks"]))
    check("only measurable fills counted", res["n_slippage_measured"] == 3,
          str(res["n_slippage_measured"]))
    check("symbol with no spec is excluded, not defaulted",
          res["unknown_symbols"] == ["ZZZ"] and "ZZZ" not in
          {s for s in res["per_symbol"] if res["per_symbol"][s]["n_slippage_measured"]},
          str(res["unknown_symbols"]))
    check("the exclusion is surfaced as a warning",
          any("ZZZ" in w for w in res["warnings"]), str(res["warnings"]))
    check("per-symbol breakdown",
          res["per_symbol"]["ES"]["n_signals"] == 3
          and res["per_symbol"]["ES"]["n_filled"] == 2
          and res["per_symbol"]["ES"]["mean_slippage_ticks"] == 1.5,
          json.dumps(res["per_symbol"]["ES"]))
    check("tick size came from backtest.specs",
          res["per_symbol"]["ES"]["tick_size"] == es_tick)
    check("no backtest comparison unless asked", res["backtest_sync"] is None)


def test_log_edge_cases():
    print("\nINCUBATOR SYNC - edge cases")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Nothing measurable must NOT report perfect execution.
        p = _write(tmp, "none.csv",
                   "symbol,signal_price,realized_price,status\n"
                   "ES,5000.00,,Rejected\nES,5001.00,,Rejected\n")
        res = evaluate_incubator_sync(p)
        check("no fills -> slippage is None, never 0.0",
              res["mean_slippage_ticks"] is None and res["fill_success_rate"] == 0.0,
              str(res["mean_slippage_ticks"]))

        # Alternative column spellings from a different NT8 export template.
        p = _write(tmp, "alt.csv",
                   "Instrument,Intended Price,Fill Price\nES,5000.00,5000.50\n")
        res = evaluate_incubator_sync(p)
        check("alias columns and a missing status column are handled",
              res["n_filled"] == 1 and res["mean_slippage_ticks"] == 2.0,
              json.dumps({k: res[k] for k in ("n_filled", "mean_slippage_ticks")}))
        check("the status assumption is disclosed",
              any("status" in w for w in res["warnings"]), str(res["warnings"]))

        # JSON export.
        p = _write(tmp, "fills.json", json.dumps([
            {"symbol": "ES", "signal_price": 5000.0, "realized_price": 5000.25,
             "status": "Filled"},
            {"symbol": "ES", "signal_price": 5000.0, "realized_price": 5000.0,
             "status": "Filled"},
        ]))
        res = evaluate_incubator_sync(p)
        check("JSON logs parse to the same numbers",
              res["n_signals"] == 2 and res["mean_slippage_ticks"] == 0.5,
              str(res["mean_slippage_ticks"]))

        # Backtest reconciliation is a count gap, and is honest about it.
        res = evaluate_incubator_sync(p, backtest_trades=[{"symbol": "ES"}] * 5)
        sync = res["backtest_sync"]
        check("count reconciliation reports the gap",
              sync["live_fills"] == 2 and sync["backtest_trades"] == 5
              and sync["delta"] == -3, json.dumps(sync["by_symbol"]))

        try:
            evaluate_incubator_sync(tmp / "does_not_exist.csv")
            check("missing log raises", False, "no exception")
        except FileNotFoundError:
            check("missing log raises FileNotFoundError", True)


if __name__ == "__main__":
    test_payload_shape()
    test_payload_rejections()
    test_send_success()
    test_send_http_error()
    test_send_timeout()
    test_placeholder_is_not_dispatchable()
    test_log_parsing()
    test_log_edge_cases()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
