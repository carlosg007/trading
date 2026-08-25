"""
live.dispatcher - the Windows incubator bridge and CrossTrade signal dispatcher.

Location:  ~/src/trading/live/dispatcher.py

This is the only module in the repo that sends an order anywhere. Everything
above it (agents, strategies, backtest) produces numbers; this produces
side effects on an account, so it is deliberately small, deliberately strict,
and refuses to guess.

Three jobs
----------
1. `format_crosstrade_payload` - build the JSON CrossTrade expects. Validated:
   an unrecognised action or a non-positive quantity raises rather than being
   forwarded. A typo'd side reaching a broker webhook is not recoverable by a
   later check.
2. `send_execution_signal` - POST it with a hard timeout, and RETURN the
   outcome rather than raising. A dispatcher that throws on a 500 loses the
   record of what it attempted; the record is the point. Every result carries
   `ok`, `http_status`, `latency_ms`, and either `response_body` or `error`.
3. `evaluate_incubator_sync` - read NT8's exported fill logs and measure what
   the simulation assumed against what the platform actually did: realised
   slippage in TICKS (not dollars, not points) and the fill success rate.

Why slippage is reported in ticks
---------------------------------
`backtest.engine` charges slippage as a per-bar fraction of price built from
tick SIZE (see `_cost_arrays`), with a default of 1 tick each way. Ticks are
therefore the unit in which the live number is directly comparable to the
backtest assumption. Dollars are not - they fold in the multiplier, so ES and
MES would report different slippage for identical execution quality.

The placeholder URL is not dispatchable
---------------------------------------
`live/config.json` ships with a placeholder webhook. `send_execution_signal`
detects it and returns an error result WITHOUT sending, so a half-configured
install fails loudly at the dispatcher instead of silently POSTing signals into
a 404 and logging what looks like a transport problem.

Not implemented on purpose
--------------------------
Price-bearing order types (LIMIT, STOP) are rejected: this signature carries no
price field, and defaulting a limit order to the market is how a risk-managed
entry becomes an unbounded one. Add the price to the signature first.
"""

from __future__ import annotations

import csv
import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
INCUBATOR_LOG_DIR = Path("/mnt/backtest/artifacts/incubator_logs")

# A webhook URL containing any of these is a template, not a destination.
_PLACEHOLDER_MARKERS = ("placeholder", "example.com", "<", "changeme")

# Actions CrossTrade accepts from a strategy signal. FLATTEN closes whatever is
# open on the account; it takes a quantity for schema symmetry only.
VALID_ACTIONS = frozenset({"BUY", "SELL", "FLATTEN"})

# MARKET only, until this module can carry a price. See the module docstring.
VALID_ORDER_TYPES = frozenset({"MARKET"})


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Read the connection profile. Missing keys are an error, not a default -
    a dispatcher that invents an account id would trade the wrong account.
    """
    p = Path(path) if path is not None else CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"No dispatcher config at {p}")
    with open(p) as fh:
        cfg = json.load(fh)

    required = ("crosstrade_webhook_url", "account_id", "environment", "timeout_seconds")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"{p} is missing required key(s): {', '.join(missing)}")
    return cfg


def is_placeholder_url(url: str) -> bool:
    low = str(url).lower()
    return any(m in low for m in _PLACEHOLDER_MARKERS)


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------

def format_crosstrade_payload(
    symbol: str,
    action: str,
    quantity: int,
    order_type: str = "MARKET",
    account_id: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build the CrossTrade webhook payload.

        {"account": ..., "action": "BUY", "symbol": "ES",
         "orderType": "MARKET", "quantity": 1}

    `account_id` defaults to the one in `live/config.json`. Symbol, action and
    order type are upper-cased; quantity is coerced to int and must be > 0.

    Raises ValueError on an unknown action, an unsupported order type, a blank
    symbol, or a quantity that is not a positive whole number.
    """
    sym = str(symbol).strip().upper()
    if not sym:
        raise ValueError("symbol is empty")

    act = str(action).strip().upper()
    if act not in VALID_ACTIONS:
        raise ValueError(
            f"action {action!r} is not one of {sorted(VALID_ACTIONS)}. "
            f"Refusing to forward an unrecognised side to a broker."
        )

    ot = str(order_type).strip().upper()
    if ot not in VALID_ORDER_TYPES:
        raise ValueError(
            f"order_type {order_type!r} is not supported. This signature carries "
            f"no price, so only {sorted(VALID_ORDER_TYPES)} can be dispatched safely."
        )

    # bool is an int subclass; True would silently become quantity 1.
    if isinstance(quantity, bool):
        raise ValueError("quantity must be a number, not a bool")
    qty_f = float(quantity)
    if qty_f != int(qty_f):
        raise ValueError(f"quantity {quantity!r} is not a whole number of contracts")
    qty = int(qty_f)
    if qty <= 0:
        raise ValueError(f"quantity must be positive, got {quantity!r}")

    if account_id is None:
        account_id = load_config(config_path)["account_id"]

    return {
        "account": str(account_id),
        "action": act,
        "symbol": sym,
        "orderType": ot,
        "quantity": qty,
    }


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

# Aliased so tests can substitute a fake opener without patching the stdlib
# module globally. Resolved from module globals at call time.
_urlopen = urllib.request.urlopen


def _result(
    *, ok: bool, url: str, payload: dict, started: float,
    http_status: int | None = None, response_body: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "ok": ok,
        "http_status": http_status,
        "response_body": response_body,
        "error": error,
        "url": url,
        "payload": payload,
    }


def send_execution_signal(
    payload: dict,
    webhook_url: str | None = None,
    timeout_seconds: float | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    POST `payload` to the CrossTrade webhook with a hard timeout (2.0s from
    `live/config.json` unless overridden).

    Never raises on a transport or HTTP failure - returns a result dict:

        timestamp      ISO-8601 UTC, when the attempt finished
        latency_ms     wall clock around the request, measured either way
        ok             True only on a 2xx
        http_status    int, or None if nothing was sent / no response arrived
        response_body  decoded body on a response (success or error)
        error          human-readable reason on failure, else None
        url, payload   what was attempted

    A placeholder webhook URL is refused before the socket is opened.
    """
    if webhook_url is None or timeout_seconds is None:
        cfg = load_config(config_path)
        if webhook_url is None:
            webhook_url = cfg["crosstrade_webhook_url"]
        if timeout_seconds is None:
            timeout_seconds = float(cfg["timeout_seconds"])

    started = time.perf_counter()

    if not isinstance(payload, dict) or not payload:
        return _result(ok=False, url=str(webhook_url), payload=payload, started=started,
                       error="payload must be a non-empty dict")

    if is_placeholder_url(webhook_url):
        return _result(
            ok=False, url=webhook_url, payload=payload, started=started,
            error=(f"refusing to dispatch to placeholder webhook URL {webhook_url!r}; "
                   f"set crosstrade_webhook_url in {CONFIG_PATH}"),
        )

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )

    try:
        with _urlopen(req, timeout=float(timeout_seconds)) as resp:
            status = int(getattr(resp, "status", None) or resp.getcode())
            text = resp.read().decode("utf-8", errors="replace")
        return _result(ok=200 <= status < 300, url=webhook_url, payload=payload,
                       started=started, http_status=status, response_body=text,
                       error=None if 200 <= status < 300 else f"HTTP {status}")

    except urllib.error.HTTPError as exc:            # 4xx / 5xx
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            text = None
        return _result(ok=False, url=webhook_url, payload=payload, started=started,
                       http_status=int(exc.code), response_body=text,
                       error=f"HTTP {exc.code}: {exc.reason}")

    except (socket.timeout, TimeoutError) as exc:
        return _result(ok=False, url=webhook_url, payload=payload, started=started,
                       error=f"timeout after {timeout_seconds}s: {exc or 'timed out'}")

    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return _result(ok=False, url=webhook_url, payload=payload, started=started,
                           error=f"timeout after {timeout_seconds}s: {reason}")
        return _result(ok=False, url=webhook_url, payload=payload, started=started,
                       error=f"network error: {reason}")

    except Exception as exc:                          # never let a send kill the loop
        return _result(ok=False, url=webhook_url, payload=payload, started=started,
                       error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# incubator reconciliation
# --------------------------------------------------------------------------

# NT8 exports are not standardised across templates, so accept the common
# spellings rather than silently reading zero rows.
_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol":         ("symbol", "instrument", "ticker", "contract"),
    "signal_price":   ("signal_price", "signalprice", "intended_price", "expected_price",
                       "order_price", "requested_price"),
    "realized_price": ("realized_price", "realised_price", "fill_price", "fillprice",
                       "avg_fill_price", "price", "executed_price"),
    "status":         ("status", "state", "order_state", "fill_status"),
    "quantity":       ("quantity", "qty", "size", "filled_quantity"),
    "action":         ("action", "side", "direction", "market_position"),
    "ts":             ("ts", "timestamp", "time", "fill_time", "datetime"),
}

_FILLED = {"filled", "fill", "executed", "complete", "completed", "ok", "true", "1"}
_UNFILLED = {"rejected", "cancelled", "canceled", "expired", "error", "failed",
             "unfilled", "false", "0", "working", "pending"}


def _norm_row(row: dict) -> dict[str, Any]:
    """Map one raw log row onto the canonical field names."""
    lowered = {str(k).strip().lower().replace(" ", "_"): v for k, v in row.items()}
    out: dict[str, Any] = {}
    for canon, names in _ALIASES.items():
        for n in names:
            if n in lowered and lowered[n] not in (None, ""):
                out[canon] = lowered[n]
                break
    return out


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _read_log(path: Path) -> list[dict]:
    """Read a fill log. `.json` is a list of objects (or {"fills": [...]}); anything
    else is parsed as delimited text."""
    if path.suffix.lower() == ".json":
        with open(path) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for key in ("fills", "executions", "rows", "data"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"{path} does not contain a list of fill records")
        return [r for r in data if isinstance(r, dict)]

    with open(path, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        return list(csv.DictReader(fh, dialect=dialect))


def _tick_size(symbol: str) -> float | None:
    """Tick size from `backtest.specs`, or None if the symbol is unknown there."""
    try:
        from backtest.specs import get_spec
    except Exception:                                  # pragma: no cover - import guard
        return None
    try:
        return float(get_spec(symbol).tick_size)
    except KeyError:
        return None


def _is_filled(row: dict) -> bool | None:
    """True / False from an explicit status; None when the log does not say."""
    raw = row.get("status")
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _FILLED:
        return True
    if s in _UNFILLED:
        return False
    if s.startswith("partial"):
        return True
    return None


def evaluate_incubator_sync(
    log_file_path: str | Path,
    backtest_trades: list | None = None,
) -> dict[str, Any]:
    """
    Compare what NT8 actually did against what the strategy asked for.

    `log_file_path` may be absolute, or a bare filename resolved inside
    /mnt/backtest/artifacts/incubator_logs/. CSV (or any delimited text) and
    JSON are both accepted; column names are matched case-insensitively against
    the aliases above.

    Per fill:  slippage_ticks = |realized_price - signal_price| / tick_size,
    with tick_size taken from `backtest.specs` - never assumed. A row whose
    symbol has no spec is counted as a signal but excluded from the slippage
    average and listed under `unknown_symbols`, because a wrong tick size turns
    a clean fill into a fake 100-tick outlier.

    Returned:
        n_signals, n_filled, fill_success_rate
        mean_slippage_ticks, median_slippage_ticks, max_slippage_ticks   (None
            if nothing was measurable - never 0.0, which would read as perfect)
        n_slippage_measured, per_symbol, unknown_symbols, warnings
        backtest_sync   {...} when `backtest_trades` is given, else None

    `backtest_trades` is an optional list of dicts (or a DataFrame's
    `to_dict("records")`) carrying at least `symbol`; it is used only for a
    count reconciliation and, where both sides expose a price, a mean divergence
    in ticks. It is NOT a validation gate - the Phase 3 holdout is.
    """
    path = Path(log_file_path)
    if not path.is_absolute() and not path.exists():
        path = INCUBATOR_LOG_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"No incubator fill log at {path}")

    raw_rows = _read_log(path)
    warnings: list[str] = []
    unknown: set[str] = set()

    n_signals = 0
    n_filled = 0
    n_status_unknown = 0
    slips: list[float] = []
    per_symbol: dict[str, dict[str, Any]] = {}

    for i, raw in enumerate(raw_rows):
        row = _norm_row(raw)
        sym = str(row.get("symbol", "")).strip().upper()
        if not sym:
            warnings.append(f"row {i}: no symbol column, skipped")
            continue
        n_signals += 1

        bucket = per_symbol.setdefault(
            sym, {"n_signals": 0, "n_filled": 0, "slips": [], "tick_size": None})
        bucket["n_signals"] += 1

        realized = _as_float(row.get("realized_price"))
        signal = _as_float(row.get("signal_price"))
        status = _is_filled(row)

        # No status column: a row is a fill iff it carries a realised price.
        filled = status if status is not None else (realized is not None)
        if status is None:
            n_status_unknown += 1
        if filled:
            n_filled += 1
            bucket["n_filled"] += 1

        if not filled or realized is None or signal is None:
            continue

        tick = _tick_size(sym)
        bucket["tick_size"] = tick
        if tick is None or tick <= 0:
            unknown.add(sym)
            continue
        slip = abs(realized - signal) / tick
        slips.append(slip)
        bucket["slips"].append(slip)

    if n_status_unknown:
        warnings.append(
            f"{n_status_unknown} row(s) had no recognised status column; "
            f"treated a row as filled iff it carried a realized price")
    if unknown:
        warnings.append(
            f"no contract spec for {', '.join(sorted(unknown))} - excluded from "
            f"slippage (add them to backtest/specs.py)")

    def _summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"mean_slippage_ticks": None, "median_slippage_ticks": None,
                    "max_slippage_ticks": None}
        return {"mean_slippage_ticks": round(sum(values) / len(values), 4),
                "median_slippage_ticks": round(median(values), 4),
                "max_slippage_ticks": round(max(values), 4)}

    symbols_out = {}
    for sym, b in sorted(per_symbol.items()):
        s = _summary(b["slips"])
        s.update({
            "n_signals": b["n_signals"],
            "n_filled": b["n_filled"],
            "fill_success_rate": (round(b["n_filled"] / b["n_signals"], 6)
                                  if b["n_signals"] else None),
            "n_slippage_measured": len(b["slips"]),
            "tick_size": b["tick_size"],
        })
        symbols_out[sym] = s

    out: dict[str, Any] = {
        "log_file": str(path),
        "n_signals": n_signals,
        "n_filled": n_filled,
        "fill_success_rate": round(n_filled / n_signals, 6) if n_signals else None,
        "n_slippage_measured": len(slips),
        "per_symbol": symbols_out,
        "unknown_symbols": sorted(unknown),
        "warnings": warnings,
        "backtest_sync": _reconcile(per_symbol, backtest_trades),
    }
    out.update(_summary(slips))
    return out


def _reconcile(per_symbol: dict, backtest_trades: Iterable | None) -> dict | None:
    """
    Count reconciliation against the backtest's trade list. Deliberately shallow:
    it reports the count gap per symbol, nothing more. Matching individual live
    fills to backtest trades needs a shared order id, which NT8's export does not
    carry, and a timestamp-nearest match would invent agreement.
    """
    if backtest_trades is None:
        return None

    records = list(backtest_trades)
    bt_counts: dict[str, int] = {}
    unusable = 0
    for t in records:
        if hasattr(t, "get"):
            sym = t.get("symbol") or t.get("Symbol")
        else:
            sym = getattr(t, "symbol", None)
        if not sym:
            unusable += 1
            continue
        sym = str(sym).strip().upper()
        bt_counts[sym] = bt_counts.get(sym, 0) + 1

    live_counts = {s: b["n_filled"] for s, b in per_symbol.items()}
    symbols = sorted(set(live_counts) | set(bt_counts))
    by_symbol = {
        s: {"live_fills": live_counts.get(s, 0),
            "backtest_trades": bt_counts.get(s, 0),
            "delta": live_counts.get(s, 0) - bt_counts.get(s, 0)}
        for s in symbols
    }
    return {
        "backtest_trades": len(records),
        "live_fills": sum(live_counts.values()),
        "delta": sum(live_counts.values()) - len(records),
        "by_symbol": by_symbol,
        "unusable_backtest_rows": unusable,
        "note": ("count reconciliation only - no per-trade matching, NT8's export "
                 "carries no shared order id"),
    }


# --------------------------------------------------------------------------
# the NT8 log-reading primitives, shared
# --------------------------------------------------------------------------
# `portfolio/incubator_recorder.py` turns the SAME rows into ledger trades, so
# these four stopped being private the moment it existed. Public names rather
# than a second consumer reaching for the underscored ones, because the alias
# table and the fill test now have two callers and an edit to either changes
# both: a spelling this module stops recognising is a fill the recorder stops
# recording, and the ledger would simply be short a trade with nothing saying
# so.
#
# One reader, one alias table, one definition of "this row was filled". A
# recorder with its own copy would drift, and the drift shows up as a promotion
# decision taken on a trade list that disagrees with the sync report printed
# beside it.
read_fill_log = _read_log
normalize_fill_row = _norm_row
parse_log_number = _as_float
fill_status = _is_filled
FILL_ALIASES = _ALIASES
