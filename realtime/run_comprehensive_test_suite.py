#!/usr/bin/env python3
"""
realtime.run_comprehensive_test_suite - BUY then CLOSE, every instrument on
every account, to find out which pairs the broker will actually accept.

    python3 realtime/run_comprehensive_test_suite.py              # plan only
    python3 realtime/run_comprehensive_test_suite.py --send       # place them
    python3 realtime/run_comprehensive_test_suite.py --send \\
        --accounts SimIncubator1 --instruments MES,MNQ,MGC

WHAT THIS COSTS, BEFORE ANYTHING ELSE
=====================================
The full matrix is 29 instruments x 4 accounts x 2 orders = 232 REAL ORDERS,
and at a two-second settle between each BUY and its CLOSE it runs for about
five minutes of continuous order flow. They are simulated accounts, so no
money moves - but three things are still true and none of them is obvious:

  * THE FILL RECORD IS EVIDENCE. `evaluate_incubator_sync` reconciles NT8
    fills and `scripts/incubator_tracker.py` decides graduation to the prop
    track on forward paper trades. 232 test fills land in that record beside
    the real ones. `SimProp1` and `SimProp2` are `prop_eval` accounts - the
    ones graduation is measured on - which is why `--accounts` exists and why
    the default set is the two incubator accounts only.
  * THE LIVE LOOP IS PROBABLY RUNNING, AND IT CANNOT SEE THESE POSITIONS.
    `PositionBook` records only what that process opened; a position this
    script leaves behind is one the loop will decline to flatten, correctly,
    forever. Worse, on an instrument a live basket holds - MNQ, MES, MGC, 6E,
    6J today - the loop's own order nets against a test position at the broker
    while neither system knows about the other's. `--check-overlap` is on by
    default and refuses those pairs unless `--allow-overlap` is passed.
  * A BUY THAT SUCCEEDS WHILE ITS CLOSE FAILS LEAVES AN OPEN POSITION. Over
    116 pairs that is not a hypothetical. Every BUY is followed by a CLOSE in
    a `finally`, the outcome of both is recorded, and any pair that ends
    OPEN is listed by name at the top of the scorecard rather than counted
    into a total nobody reads.

NOTHING IS REIMPLEMENTED. Instruments resolve through
`realtime.contract_resolver`, commands are built by
`realtime.crosstrade_formatter`, and the wire is
`live.dispatcher.send_execution_signal` - the same three the live loop uses.
A harness that formatted its own payload would prove the broker accepts THE
HARNESS, which is not the question.

A CONTRACT PAST ITS ROLL DATE IS SKIPPED, NOT FAILED. The resolver refuses it
and that refusal is correct; counting it as a connectivity failure would put a
red number against a broker that was never asked. The rates roll on
2026-08-31, so a run that crosses midnight UTC will skip ZB/ZN/ZF/ZT/ZW from
that moment - which is the table working, not the suite breaking.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.contract_resolver import (  # noqa: E402
    ContractResolverError,
    resolve_contract,
)
from realtime.crosstrade_formatter import (  # noqa: E402
    CrossTradeFormatError,
    format_crosstrade_command,
    format_flatten_command,
    redact,
)
from realtime.live_dispatcher import (  # noqa: E402
    _host_only,
    _url_path,
    resolve_credentials,
)

DEFAULT_REPORT = REPO_ROOT / "logs" / "test_run_comprehensive.json"
PORTFOLIOS = REPO_ROOT / "config" / "portfolios.json"

#: The instrument families, in the order a reader scans them.
GROUPS: dict[str, list[str]] = {
    "FX": ["6A", "6B", "6C", "6E", "6J", "6S"],
    "Equity Indices": ["ES", "MES", "NQ", "MNQ", "RTY", "YM"],
    "Commodities & Metals": ["GC", "MGC", "SI", "PL", "HO", "NG", "RB", "LE"],
    "Rates & Ags": ["ZB", "ZN", "ZF", "ZT", "ZC", "ZS", "ZW"],
    "Crypto": ["BTC", "ETH"],
}
ALL_INSTRUMENTS = [s for group in GROUPS.values() for s in group]

#: The incubator pair only. `SimProp1`/`SimProp2` are `prop_eval` accounts and
#: graduation to the prop track is decided on their forward paper record, so
#: they are opt-in rather than default - `--accounts` takes them.
DEFAULT_ACCOUNTS = ["SimIncubator1", "SimIncubator2"]
ALL_ACCOUNTS = ["SimIncubator1", "SimIncubator2", "SimProp1", "SimProp2"]

SETTLE_SECONDS = 2.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def live_basket_symbols() -> set[str]:
    """Every instrument an ACTIVE portfolio can trade right now.

    Read from the routing table rather than hardcoded: the baskets moved twice
    this week and a stale copy here would mean the overlap guard protected the
    wrong instruments.
    """
    try:
        cfg = json.loads(PORTFOLIOS.read_text())
    except (OSError, ValueError):
        return set()
    held: set[str] = set()
    for block in (cfg.get("portfolios") or {}).values():
        if isinstance(block, dict) and block.get("active_strategies"):
            held.update((block.get("basket") or {}).get("assets") or [])
    return held


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("BUY then CLOSE every instrument on every account and "
                     "report which pairs the broker accepted. Places nothing "
                     "without --send."))
    p.add_argument("--send", action="store_true",
                   help="ACTUALLY PLACE ORDERS. Without it this prints the "
                        "plan and exits.")
    p.add_argument("--accounts", default=",".join(DEFAULT_ACCOUNTS),
                   help=(f"comma-separated (default: "
                         f"{','.join(DEFAULT_ACCOUNTS)}). The prop accounts "
                         f"are opt-in: {','.join(ALL_ACCOUNTS)}"))
    p.add_argument("--instruments", default=",".join(ALL_INSTRUMENTS),
                   help="comma-separated roots (default: all 29)")
    p.add_argument("--qty", type=int, default=1, help="contracts (default: 1)")
    p.add_argument("--settle", type=float, default=SETTLE_SECONDS,
                   help=f"seconds between BUY and CLOSE (default: "
                        f"{SETTLE_SECONDS})")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="per-request seconds (default: 5.0)")
    p.add_argument("--allow-overlap", action="store_true",
                   help="test instruments a LIVE basket also trades. Off by "
                        "default: the running loop nets against these at the "
                        "broker and cannot see the test position.")
    p.add_argument("--report", default=str(DEFAULT_REPORT),
                   help=f"JSON report path (default: {DEFAULT_REPORT})")
    return p


def _scrub(text: str, url: str) -> str:
    out = redact(str(text))
    path = _url_path(url)
    return out.replace(path, "/<redacted>") if path else out


def _one_order(sender, url: str, key: str, account: str, instrument: str,
               action: str, qty: int, timeout: float) -> dict:
    """Format and send ONE order. Never raises - the outcome is the return."""
    rec = {"account": account, "instrument": instrument, "action": action,
           "qty": None if action == "CLOSE" else qty,
           "ok": False, "http_status": None, "elapsed_ms": None,
           "error": None, "response": None}
    try:
        if action == "CLOSE":
            command = format_flatten_command(account=account,
                                             instrument=instrument, key=key)
        else:
            command = format_crosstrade_command(
                account=account, instrument=instrument, action=action,
                qty=qty, order_type="MARKET", key=key)
    except (CrossTradeFormatError, ContractResolverError) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec

    started = time.perf_counter()
    result = sender(command, webhook_url=url, timeout_seconds=timeout)
    rec["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    rec["ok"] = bool(result.get("ok"))
    rec["http_status"] = result.get("http_status")
    rec["error"] = result.get("error")
    body = result.get("response_body")
    if body:
        rec["response"] = _scrub(body, url)[:300]
    return rec


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accounts = [a.strip() for a in args.accounts.split(",") if a.strip()]
    roots = [s.strip().upper() for s in args.instruments.split(",")
             if s.strip()]

    try:
        url, key, cred_source = resolve_credentials()
    except Exception as exc:                                   # noqa: BLE001
        print(f"FAILED  credentials: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    if not url:
        print("FAILED  CROSSTRADE_WEBHOOK_URL is not set in .env",
              file=sys.stderr)
        return 2

    overlap = live_basket_symbols()
    W = 86
    print("=" * W)
    print("COMPREHENSIVE CONNECTIVITY SUITE")
    print("=" * W)
    print(f"  endpoint    : {_host_only(url)}  (path withheld)")
    print(f"  creds from  : {cred_source or 'unknown'}")
    print(f"  accounts    : {', '.join(accounts)}")
    print(f"  instruments : {len(roots)}")
    print(f"  settle      : {args.settle}s between BUY and CLOSE")
    if overlap:
        print(f"  LIVE baskets: {', '.join(sorted(overlap))}"
              + ("  (testing them anyway: --allow-overlap)"
                 if args.allow_overlap else "  (SKIPPED - see --allow-overlap)"))
    print("-" * W)

    # ---- resolve and plan, before a single order --------------------------
    plan: list[tuple[str, str, str]] = []       # (account, root, instrument)
    skipped: list[dict] = []
    for account in accounts:
        for root in roots:
            if root in overlap and not args.allow_overlap:
                skipped.append({"account": account, "root": root,
                                "reason": "a LIVE basket trades this; the "
                                          "running loop would net against it"})
                continue
            try:
                instrument = resolve_contract(root)
            except ContractResolverError as exc:
                skipped.append({"account": account, "root": root,
                                "reason": f"{type(exc).__name__}: {exc}"})
                continue
            plan.append((account, root, instrument))

    print(f"  {len(plan)} pair(s) to test, {len(skipped)} skipped")
    if skipped:
        by_reason: dict[str, list[str]] = {}
        for s in skipped:
            head = s["reason"].split(".")[0][:60]
            by_reason.setdefault(head, []).append(f"{s['account']}/{s['root']}")
        for reason, pairs in by_reason.items():
            print(f"    SKIP {len(pairs):3}  {reason}")
    print("-" * W)

    if not args.send:
        print("  NOT SENT.  Nothing left this process.")
        print(f"  This plan is {len(plan) * 2} real orders across "
              f"{len(accounts)} account(s).")
        print("  Re-run with --send to place them.")
        print("=" * W)
        return 0

    from live.dispatcher import send_execution_signal

    results: list[dict] = []
    left_open: list[dict] = []
    print(f"  {'ACCOUNT':<15}{'INSTRUMENT':<14}{'BUY':<22}{'CLOSE':<22}")
    print("-" * W)
    for account, root, instrument in plan:
        buy = _one_order(send_execution_signal, url, key, account, instrument,
                         "BUY", args.qty, args.timeout)
        close = None
        try:
            if buy["ok"]:
                time.sleep(args.settle)
        finally:
            # ALWAYS attempted when the BUY went through, including on a
            # KeyboardInterrupt during the settle - an interrupted run that
            # left positions open would be the worst outcome this script has.
            if buy["ok"]:
                close = _one_order(send_execution_signal, url, key, account,
                                   instrument, "CLOSE", args.qty,
                                   args.timeout)
        pair = {"account": account, "root": root, "instrument": instrument,
                "buy": buy, "close": close}
        results.append(pair)
        if buy["ok"] and not (close and close["ok"]):
            left_open.append(pair)

        def cell(r):
            if r is None:
                return "not attempted"
            head = "OK " if r["ok"] else "FAIL"
            return f"{head} {r['http_status'] or '-'} {r['elapsed_ms'] or 0:.0f}ms"
        print(f"  {account:<15}{instrument:<14}{cell(buy):<22}"
              f"{cell(close):<22}")

    # ---- the scorecard ----------------------------------------------------
    total = len(results) * 2
    ok = sum(1 for r in results
             for leg in (r["buy"], r["close"]) if leg and leg["ok"])
    failed = sum(1 for r in results
                 for leg in (r["buy"], r["close"]) if leg and not leg["ok"])
    not_attempted = sum(1 for r in results if r["close"] is None)

    print("=" * W)
    if left_open:
        # FIRST, and by name. A total nobody reads is not a warning.
        print(f"  !! {len(left_open)} POSITION(S) MAY STILL BE OPEN - the BUY "
              f"succeeded and the CLOSE did not:")
        for p in left_open:
            print(f"       {p['account']}  {p['instrument']}")
        print("     Close them in NinjaTrader. The live loop will NOT: its "
              "PositionBook records only what that process opened.")
        print("-" * W)
    print(f"  Total tested : {total} orders across {len(results)} pair(s)")
    print(f"  Succeeded    : {ok}")
    print(f"  Failed       : {failed}")
    if not_attempted:
        print(f"  CLOSE skipped: {not_attempted} (the BUY had already failed)")
    print(f"  Skipped      : {len(skipped)} pair(s) never sent")
    print("=" * W)

    report = {
        "generated_utc": _now(),
        "endpoint_host": _host_only(url),
        "accounts": accounts,
        "settle_seconds": args.settle,
        "qty": args.qty,
        "totals": {"orders": total, "succeeded": ok, "failed": failed,
                   "close_not_attempted": not_attempted,
                   "pairs_skipped": len(skipped),
                   "pairs_left_open": len(left_open)},
        "left_open": [{"account": p["account"], "instrument": p["instrument"]}
                      for p in left_open],
        "skipped": skipped,
        "results": results,
    }
    dest = Path(args.report)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"  report -> {dest}")

    # Non-zero if anything failed OR anything is open. An open position is
    # the outcome that needs a human, so it must not exit 0.
    return 1 if (failed or left_open) else 0


if __name__ == "__main__":
    raise SystemExit(main())
