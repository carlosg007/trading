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
from datetime import datetime, timedelta, timezone
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
from realtime.market_calendar import (EXCHANGE_TZ,             # noqa: E402
                                      market_status)
from realtime.risk_firewall import (arm_kill_switch,              # noqa: E402
                                    kill_switch_engaged)

HEALTHY, DEGRADED, BROKEN = 0, 1, 2

#: CME futures trade nearly around the clock, and the gaps are the point here:
#: a feed that returns nothing at 02:00 on a Saturday is a CLOSED MARKET, not a
#: dead feed, and a watchdog that cannot tell them apart pages somebody every
#: two minutes for 49 hours.
#:
#: THE SCHEDULE MOVED OUT OF THIS FILE, and only the schedule.
#: `realtime/market_calendar.py` now owns the four numbers and the two
#: comparators; `market_status` above is IMPORTED from there and is the same
#: function `master_live.py` stands itself down on. Two processes that decide
#: opposite things about a Sunday evening would each log correctly while one of
#: them was wrong, and nothing downstream could see it — which is exactly the
#: failure a second copy of these constants produces. `EXCHANGE_TZ` is
#: re-exported because callers and tests reach for `wd.EXCHANGE_TZ`.
#:
#: Nothing else moved. The SUPPRESSION policy below — which checks a closed
#: market silences, and which it must not — is a watchdog decision and stays a
#: watchdog decision.

#: How long a persistent DEGRADED state waits before it says so again. The
#: first alert is immediate; this only governs the REMINDERS, so a fault that
#: nobody has fixed is still audible without being a stream.
ALERT_REMINDER_HOURS = 4.0

#: Where the last verdict is remembered. The watchdog is a ONESHOT under a
#: 2-minute timer, not a resident daemon, so nothing survives in memory between
#: checks - debouncing that lived in a variable would reset 720 times a day and
#: alert on every one of them.
ALERT_STATE_FILE = REPO_ROOT / "data" / "watchdog_alert_state.json"

#: The checks whose failure is a STALENESS claim, and therefore meaningless
#: while the market is shut. `state` and `kill_switch` are deliberately absent:
#: an unverified claim or an armed switch is just as real on a Saturday, and
#: suppressing those would use the weekend to hide a fault that has nothing to
#: do with the weekend.
STALENESS_CHECKS = ("regime", "feed")


def read_alert_state(path: Path | None = None) -> dict:
    """
    The previous verdict, or an empty dict.

    UNREADABLE STATE IS TREATED AS NO STATE, which makes the next DEGRADED look
    like a fresh transition and alert. That is the safe direction: the failure
    is one duplicate alert, where the opposite - inventing a "we already told
    them" - is a fault that never gets announced at all.
    """
    try:
        return json.loads((path or ALERT_STATE_FILE).read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def write_alert_state(state: dict, path: Path | None = None) -> None:
    """Remember this verdict. A failed write must not fail the check."""
    dest = path or ALERT_STATE_FILE
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[watchdog] could not persist alert state: {exc}",
              file=sys.stderr)


def decide_alert(previous: dict, degraded: bool, now: datetime,
                 reminder_hours: float = ALERT_REMINDER_HOURS
                 ) -> tuple[str | None, dict]:
    """
    `(kind, next_state)` where kind is 'transition', 'reminder', 'recovery'
    or None.

    The whole point is that a STATE is not an EVENT. The old code posted on
    every cycle a fault was present, so one stale feed produced an alert every
    two minutes until somebody fixed it - 30 an hour, all of them the same
    sentence. What an operator needs to be told is that something CHANGED, and
    then periodically that it still has not.
    """
    was_degraded = bool(previous.get("degraded"))
    last_at = previous.get("last_alert_at")
    state = {"degraded": degraded,
             "last_alert_at": last_at,
             "updated_at": now.isoformat(timespec="seconds")}

    if degraded and not was_degraded:
        state["last_alert_at"] = now.isoformat(timespec="seconds")
        return "transition", state
    if not degraded and was_degraded:
        state["last_alert_at"] = now.isoformat(timespec="seconds")
        return "recovery", state
    if degraded and was_degraded:
        age = _age_of(last_at, now)
        # No recorded time for the last alert means it cannot be shown to be
        # recent, so it reminds. Staying quiet on an unknown would let a
        # truncated state file silence a live fault indefinitely.
        if age is None or age >= reminder_hours * 3600.0:
            state["last_alert_at"] = now.isoformat(timespec="seconds")
            return "reminder", state
    return None, state


def _age_of(stamp, now: datetime) -> float | None:
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (now - when).total_seconds()



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


def alert(message: str, webhook: str | None = None,
          kind: str = "transition") -> bool:
    """Post to Discord, or say why not. Never raises: losing the alert must not
    lose the check that produced it.

    `kind` picks the heading. A RECOVERY is a different event from a fault and
    must not arrive wearing the same red title - an operator scanning a channel
    reads the colour before the words, and "resolved" in amber reads as another
    page.
    """
    try:
        from mdlib.env import discord_webhook                      # noqa: PLC0415
        url = webhook or discord_webhook()
        if not url:
            return False
        from backtest.discord_reporter import build_payload, post_embed  # noqa: PLC0415
        title, color = {
            "recovery": ("🐕 Trading watchdog — RECOVERED", 0x2ECC71),
            "reminder": ("🐕 Trading watchdog — STILL DEGRADED", 0xE67E22),
        }.get(kind, ("🐕 Trading watchdog — DEGRADED", 0xE67E22))
        embed = {"title": title,
                 "description": f"```\n{message[:3800]}\n```",
                 "color": color}
        # (webhook, payload) — that ORDER, and build_payload takes ONE embed,
        # not a list. Reversed, `requests.post` was handed the payload dict as
        # its URL and raised AttributeError inside the except below, so every
        # DEGRADED verdict was found, printed, and silently never delivered.
        result = post_embed(url, build_payload(embed))
        # post_embed returns a DICT. `getattr(result, "ok", result)` found no
        # attribute and fell back to the dict itself, which is truthy for every
        # non-empty dict — a failed post reported success.
        if not result.get("ok"):
            print(f"[watchdog] alert not delivered: "
                  f"HTTP {result.get('http_status')} {result.get('error')}",
                  file=sys.stderr)
            return False
        return True
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
    ap.add_argument("--reminder-hours", type=float,
                    default=ALERT_REMINDER_HOURS,
                    help=(f"how long a persistent DEGRADED state waits before "
                          f"saying so again (default {ALERT_REMINDER_HOURS})"))
    ap.add_argument("--alert-state-path", default=None,
                    help="where the last verdict is remembered")
    ap.add_argument("--ignore-market-hours", action="store_true",
                    help=("check staleness even while the market is shut. For "
                          "diagnosing the feed out of hours"))
    ap.add_argument("--now", default=None,
                    help="ISO timestamp to evaluate the schedule at (testing)")
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

    now = datetime.now(timezone.utc)
    if args.now:
        now = datetime.fromisoformat(args.now)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

    # MARKET HOURS, applied to the staleness checks ONLY. A feed returning
    # nothing at 02:00 on a Saturday is a closed exchange, and a bar two days
    # old at 09:00 on a Sunday is the Friday close doing exactly what it should
    # - neither is a fault, and both used to page every two minutes for the
    # whole 49-hour weekend.
    #
    # `state` and `kill_switch` are NOT suppressed. An unverified claim or an
    # armed switch is just as true on a Saturday, and using the weekend to
    # silence those would hide a fault that has nothing to do with the weekend.
    is_open, market_reason = market_status(now)
    if not is_open and not args.ignore_market_hours:
        for record in results:
            if record["check"] in STALENESS_CHECKS and not record["ok"]:
                record["ok"] = True
                record["market_closed"] = True
                record["detail"] = f"MARKET_CLOSED ({market_reason}) — {record['detail']}"

    failed = [r for r in results if not r["ok"]]
    stamp = now.isoformat(timespec="seconds")
    for record in results:
        mark = "ok  " if record["ok"] else "FAIL"
        print(f"[{stamp}] {mark} {record['check']:<12} {record['detail']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"at": stamp, "results": results}, indent=2) + "\n",
            encoding="utf-8")

    if not is_open and not args.ignore_market_hours:
        print(f"[{stamp}] ok   market       {market_reason}")

    state_path = Path(args.alert_state_path) if args.alert_state_path else None
    kind, next_state = decide_alert(read_alert_state(state_path), bool(failed),
                                    now, args.reminder_hours)

    if not failed:
        # The recovery notice is the reason this runs on the healthy path too.
        # A channel that is told about every fault and never about a fix leaves
        # an operator to infer the fix from silence, which is the same signal
        # as a watchdog that has stopped running.
        if kind == "recovery" and not args.quiet:
            alert(f"All checks passed at {stamp}.", kind="recovery")
        write_alert_state(next_state, state_path)
        return HEALTHY

    summary = "\n".join(f"{r['check']}: {r['detail']}" for r in failed)
    if args.arm_kill_switch:
        path = arm_kill_switch(f"watchdog: {failed[0]['check']} degraded")
        summary += f"\n\nKILL SWITCH ARMED at {path} — new entries blocked. " \
                   f"Open positions were NOT touched."
    if kind and not args.quiet:
        if kind == "reminder":
            summary += (f"\n\nStill degraded. Reminders are throttled to one "
                        f"every {args.reminder_hours:g}h.")
        alert(summary, kind=kind)
    write_alert_state(next_state, state_path)
    # The EXIT CODE still reports the fault on every cycle. Only the Discord
    # post is debounced: the timer's status is how `systemctl` and anything
    # scraping it see the fault, and silencing that would hide it from
    # everything, not just from the channel.
    return DEGRADED


if __name__ == "__main__":
    raise SystemExit(main())
