#!/usr/bin/env python3
"""
The watchdog: is the loop still seeing the market?

    python3 scripts/watchdog.py --tf 1h
    python3 scripts/watchdog.py --tf 1h --arm-kill-switch     # escalate

THE FAILURE IT EXISTS FOR
=========================
A loop that is UP and reading a dead feed prints "no signal" on every line and
looks exactly like a quiet Tuesday. `systemctl status` says active, the log is
growing, nothing has crashed, and no trade has been possible for six hours.
Process supervision cannot see that; only the DATA can.

So this checks the facts that reveal it:

    regime age      how long since the daemon published a reading, and how old
                    the BAR behind that reading is
    feed age        how old the newest bar the configured feed can return is,
                    against the width of one bar
    kill switch     armed or clear, and reported either way
    engine state    what the loop has sent today, and any unverified claims

WHAT IT WILL NOT DO
===================
**It never places, cancels or flattens an order.** Escalation is an alert, and
- only with `--arm-kill-switch` - arming the switch, which blocks NEW entries
and touches nothing that is open. Flattening is a human decision made with the
runbook open: an automated flatten on a shared account closes whatever is
there, including positions placed by hand, and it fires exactly when the
information is worst.

EXIT CODES, because a timer's only voice is its status
    0  healthy
    1  DEGRADED - something is stale or missing; alert sent
    2  could not run the check at all (config unreadable, bad arguments)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mdlib.env import load_env                                     # noqa: E402

load_env()

from realtime.feed import FeedError, resolve_feed, tf_delta        # noqa: E402
from realtime.lifecycle import EngineState                        # noqa: E402
from realtime.regime_reader import (RegimeStateError,             # noqa: E402
                                    get_all_regimes)
from realtime.risk_firewall import (arm_kill_switch,              # noqa: E402
                                    kill_switch_engaged)

HEALTHY, DEGRADED, BROKEN = 0, 1, 2


def _age_s(stamp) -> float | None:
    if stamp in (None, ""):
        return None
    try:
        when = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds()


def check_regime(max_write_age_s: float, tf: str,
                 max_bar_multiple: float) -> list[dict]:
    """
    The published quadrant, on TWO clocks with two different limits.

    They answer different questions and must not share a threshold. The WRITE
    age is the daemon's own liveness - it republishes on a timer, so anything
    past a couple of intervals means the daemon is not running. The BAR age is
    the market's, and a bar is stamped at its OPEN: an hourly bar is up to an
    hour old the moment it closes, so a flat thirty-minute limit would fire on
    every healthy cycle and teach an operator to ignore the one line that
    matters. The bar limit is therefore a MULTIPLE of the bar width.
    """
    width = tf_delta(tf).total_seconds()
    max_bar_age_s = width * float(max_bar_multiple) + width
    try:
        readings = get_all_regimes()
    except RegimeStateError as exc:
        return [{"check": "regime", "ok": False, "detail": str(exc)}]
    if not readings:
        return [{"check": "regime", "ok": False,
                 "detail": "the state file publishes no symbols"}]

    out = []
    for symbol, record in sorted(readings.items()):
        bar_age = record.get("bar_age_seconds")
        write_age = record.get("age_seconds")
        bar_ok = bar_age is not None and bar_age <= max_bar_age_s
        write_ok = write_age is not None and write_age <= max_write_age_s
        out.append({
            "check": "regime", "symbol": symbol, "ok": bar_ok and write_ok,
            "quadrant": record.get("quadrant"),
            "bar_age_s": None if bar_age is None else round(bar_age),
            "write_age_s": None if write_age is None else round(write_age),
            "detail": (
                f"{symbol} {record.get('quadrant')} — "
                f"bar {(bar_age or 0) / 60:.0f} min "
                f"({'ok' if bar_ok else f'>{max_bar_age_s / 60:.0f} min'}), "
                f"daemon wrote {(write_age or 0) / 60:.0f} min ago "
                f"({'ok' if write_ok else f'>{max_write_age_s / 60:.0f} min'})")})
    return out


def check_feed(tf: str, feed_mode: str, max_multiple: float) -> list[dict]:
    """
    Can the feed still produce a bar, and is the newest one recent?

    Measured from the bar's CLOSE and against the WIDTH of a bar, because a bar
    is stamped at its open: an hourly bar is always an hour "old" by that
    reading, and a threshold drawn on it would fire on every healthy cycle.
    """
    try:
        feed = resolve_feed(feed_mode)
    except FeedError as exc:
        return [{"check": "feed", "ok": False, "detail": str(exc)}]

    try:
        bars, _ = feed.closed_bars(["MNQ"], tf, 5)
    except Exception as exc:                                       # noqa: BLE001
        return [{"check": "feed", "ok": False,
                 "detail": f"{feed.describe()}: {type(exc).__name__}: {exc}"}]

    if not bars:
        return [{"check": "feed", "ok": False,
                 "detail": f"{feed.describe()} returned no bars at all"}]

    width = tf_delta(tf).total_seconds()
    out = []
    for symbol, frame in sorted(bars.items()):
        newest = frame["ts"].iloc[-1]
        since_close = (datetime.now(timezone.utc)
                       - newest.to_pydatetime()).total_seconds() - width
        limit = width * float(max_multiple)
        out.append({
            "check": "feed", "symbol": symbol, "ok": since_close <= limit,
            "newest_bar": str(newest), "since_close_s": round(since_close),
            "detail": (f"{feed.describe()}: newest {symbol} {tf} bar {newest} "
                       f"closed {since_close / 60:.1f} min ago "
                       f"(limit {limit / 60:.0f} min)")})
    return out


def check_state(state_path: str | None) -> list[dict]:
    try:
        state = EngineState(state_path)
    except RuntimeError as exc:
        return [{"check": "state", "ok": False, "detail": str(exc)}]
    claims = [c for c in state.open_claims() if not c.get("confirmed")]
    return [{
        "check": "state", "ok": not claims,
        "orders_today": state.orders_sent_today(),
        "realised_pnl": state.realised_pnl_today(),
        "detail": (f"{state.orders_sent_today()} orders today, realised "
                   f"{state.realised_pnl_today():,.2f}"
                   + (f", {len(claims)} UNVERIFIED claim(s) from a previous "
                      f"run — reconcile in NinjaTrader" if claims else ""))}]


def alert(message: str, webhook: str | None = None) -> bool:
    """Post to Discord, or say why not. Never raises: losing the alert must not
    lose the check that produced it."""
    try:
        from mdlib.env import discord_webhook                      # noqa: PLC0415
        url = webhook or discord_webhook()
        if not url:
            return False
        from backtest.discord_reporter import build_payload, post_embed  # noqa: PLC0415
        embed = {"title": "🐕 Trading watchdog — DEGRADED",
                 "description": f"```\n{message[:3800]}\n```",
                 "color": 0xE67E22}
        result = post_embed(build_payload([embed]), url)
        return bool(getattr(result, "ok", result))
    except Exception as exc:                                       # noqa: BLE001
        print(f"[watchdog] alert failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Feed and process health checks.")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--feed", default="auto",
                    choices=("auto", "nt8", "live", "lake"))
    ap.add_argument("--max-regime-write-age-sec", type=float, default=1800.0,
                    help="how long since the DAEMON last published before it "
                         "counts as not running. Not the bar's age — see "
                         "--max-bar-age-multiple")
    ap.add_argument("--max-bar-age-multiple", type=float, default=2.0,
                    help="how many bar widths past a bar's close counts as "
                         "stale")
    ap.add_argument("--state-path", default=None)
    ap.add_argument("--arm-kill-switch", action="store_true",
                    help="on DEGRADED, arm the kill switch. Blocks new entries "
                         "and touches nothing that is open")
    ap.add_argument("--json", dest="json_out", metavar="PATH")
    ap.add_argument("--quiet", action="store_true",
                    help="do not post an alert")
    args = ap.parse_args(argv)

    try:
        results = (check_regime(args.max_regime_write_age_sec, args.tf,
                                args.max_bar_age_multiple)
                   + check_feed(args.tf, args.feed, args.max_bar_age_multiple)
                   + check_state(args.state_path))
    except Exception as exc:                                       # noqa: BLE001
        print(f"[watchdog] COULD NOT RUN: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return BROKEN

    engaged, why = kill_switch_engaged()
    results.append({"check": "kill_switch", "ok": True,
                    "engaged": engaged,
                    "detail": why or "clear"})

    failed = [r for r in results if not r["ok"]]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for record in results:
        mark = "ok  " if record["ok"] else "FAIL"
        print(f"[{stamp}] {mark} {record['check']:<12} {record['detail']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"at": stamp, "results": results}, indent=2) + "\n",
            encoding="utf-8")

    if not failed:
        return HEALTHY

    summary = "\n".join(f"{r['check']}: {r['detail']}" for r in failed)
    if args.arm_kill_switch:
        path = arm_kill_switch(f"watchdog: {failed[0]['check']} degraded")
        summary += f"\n\nKILL SWITCH ARMED at {path} — new entries blocked. " \
                   f"Open positions were NOT touched."
    if not args.quiet:
        alert(summary)
    return DEGRADED


if __name__ == "__main__":
    raise SystemExit(main())
