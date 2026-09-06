#!/usr/bin/env python3
"""
tools/tool_contract_guard.py - which contracts roll, and how long is left.

Location:  ~/src/trading/tools/tool_contract_guard.py

    python3 tools/tool_contract_guard.py
    python3 tools/tool_contract_guard.py --markdown
    python3 tools/tool_contract_guard.py --contracts /tmp/probe.json --today 2026-09-08

The Watchdog & Ops agent's read of `config/contracts.json`. Every symbol's
`roll_date` is the LAST session its `active_contract` should be traded;
`realtime/contract_resolver.py` refuses a symbol past that date. This counts
down to it and raises the two alerts the architecture asks for - 5 days out
and 3 days out - plus the two states that matter more than either.

THE TWO STATES THAT ARE NOT COUNTDOWNS
--------------------------------------
**OVERDUE.** A `roll_date` already in the past does not mean the roll happened.
It means the resolver is now refusing that symbol and every order for it fails
at dispatch. This is reported as CRITICAL and separately from the countdown,
because "you have -6 days" is not a warning an operator parses at a glance.

**A STALE FILE.** `contracts.json` carries its own `valid_until`. Past it, the
whole document is out of date and each individual countdown inside it is being
computed from figures nobody has refreshed - a symbol reading "12 days left"
may have rolled twice. The file-level verdict is reported FIRST and never
folded into the per-symbol table, because a fresh-looking row inside a stale
file is the one reading that would send an operator away reassured.

WHY THE COUNTDOWN IS IN CALENDAR DAYS
------------------------------------
`roll_date` is a calendar date and the thresholds the architecture specifies
are "5-day" and "3-day". Counting sessions instead would make the alert fire
on a different day than the number in the config implies, and an alert whose
arithmetic disagrees with the file it is reading is worse than no alert. The
session count is REPORTED alongside when `realtime/market_calendar.py` can be
imported, as context - never as the threshold.

The reference day is the CME SESSION date, not the UTC date: after 18:00 ET
the session has already rolled to tomorrow, and a guard using the UTC date
would report one day more than the desk has.

WHAT THIS DOES NOT DO
---------------------
**It does not check the contracts exist on the feed.** `6E DEC26` being named
here is not evidence NinjaTrader carries it. `nt8-check` answers that.

**It does not roll anything.** It reads; the roll is an edit to
`config/contracts.json` that a human makes.

Exit codes:  0 nothing due · 1 alerts present · 2 could not run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACTS = REPO_ROOT / "config" / "contracts.json"

#: The two countdown thresholds the architecture specifies, in calendar days.
ALERT_DAYS_URGENT = 3
ALERT_DAYS_NOTICE = 5

STATUS_OVERDUE = "OVERDUE"
STATUS_TODAY = "ROLL TODAY"
STATUS_URGENT = "URGENT"
STATUS_NOTICE = "NOTICE"
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"

#: Anything in here is a finding an operator must see.
ALERT_STATUSES = (STATUS_OVERDUE, STATUS_TODAY, STATUS_URGENT,
                  STATUS_NOTICE, STATUS_ERROR)

SEVERITY = {
    STATUS_OVERDUE: 0,
    STATUS_ERROR: 1,
    STATUS_TODAY: 2,
    STATUS_URGENT: 3,
    STATUS_NOTICE: 4,
    STATUS_OK: 5,
}


class ContractGuardError(RuntimeError):
    """The contracts file could not be read as a contracts file."""


def session_today(now: datetime | None = None) -> date:
    """
    The CME session date to count from.

    Uses `realtime.market_calendar.session_date_of` when it imports, so this
    agrees with every other session attribution in the repo. The fallback is
    the ET calendar date, which differs only in the 18:00-24:00 ET window -
    and the fallback is REPORTED by `provenance()` rather than assumed
    equivalent.
    """
    moment = now or datetime.now(timezone.utc)
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from realtime.market_calendar import session_date_of  # noqa: PLC0415
        return session_date_of(moment)
    except Exception:                                           # noqa: BLE001
        try:
            from zoneinfo import ZoneInfo                       # noqa: PLC0415
            return moment.astimezone(ZoneInfo("America/New_York")).date()
        except Exception:                                       # noqa: BLE001
            return moment.date()


def provenance() -> str:
    """Which session-date source is in force, named rather than assumed."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from realtime.market_calendar import session_date_of  # noqa: F401,PLC0415
        return "realtime.market_calendar.session_date_of"
    except Exception:                                           # noqa: BLE001
        return "ET calendar date (market_calendar did not import)"


def load_contracts(path: Path) -> dict[str, Any]:
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractGuardError(f"{path}: no such file") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractGuardError(f"{path}: unreadable ({exc})") from None
    if not isinstance(blob, dict) or not isinstance(blob.get("contracts"), dict):
        raise ContractGuardError(
            f"{path}: no 'contracts' object — this is not a contracts file. "
            f"Refusing to report 'no rolls due' from a document that could "
            f"not be parsed.")
    return blob


def _parse_date(text: Any) -> date | None:
    if not isinstance(text, str):
        return None
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError:
        return None


def sessions_remaining(start: date, end: date) -> int | None:
    """
    CME sessions from `start` up to and including `end`, or None if unknown.

    Context only. The alert threshold is the calendar-day count; see the
    module docstring.
    """
    if end < start:
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from realtime.market_calendar import (                  # noqa: PLC0415
            load_holidays, WEEKEND, HOLIDAY)
        holidays = load_holidays()
    except Exception:                                           # noqa: BLE001
        return None

    from datetime import timedelta                              # noqa: PLC0415
    count, cursor = 0, start
    while cursor <= end:
        if cursor.weekday() < 5 and cursor not in holidays:
            count += 1
        cursor += timedelta(days=1)
    _ = (WEEKEND, HOLIDAY)
    return count


def evaluate_file(blob: dict[str, Any], today: date) -> dict[str, Any]:
    """The file-level verdict: is this document still in date?"""
    raw = blob.get("valid_until")
    verdict: dict[str, Any] = {"valid_until": raw, "stale": False, "note": ""}
    if raw is None:
        verdict["note"] = ("the file declares no 'valid_until'; its freshness "
                           "is unknown, not confirmed")
        return verdict
    stamp = _parse_date(raw)
    if stamp is None:
        verdict["note"] = f"'valid_until' is {raw!r}, which is not a date"
        verdict["stale"] = True
        return verdict
    verdict["stale"] = stamp < today
    verdict["note"] = (
        f"the contracts file expired on {stamp} ({(today - stamp).days} days "
        f"ago); every countdown below is computed from figures nobody has "
        f"refreshed"
        if verdict["stale"] else
        f"valid until {stamp} ({(stamp - today).days} days left)")
    return verdict


def evaluate_symbol(symbol: str, entry: Any, today: date) -> dict[str, Any]:
    """One row. A row is produced for every symbol, including broken ones."""
    row: dict[str, Any] = {
        "symbol": symbol, "active_contract": None, "next_contract": None,
        "roll_date": None, "days_remaining": None, "sessions_remaining": None,
        "status": STATUS_ERROR, "note": "",
    }
    if not isinstance(entry, dict):
        row["note"] = (f"the entry is a {type(entry).__name__}, not an object")
        return row

    row["active_contract"] = entry.get("active_contract")
    row["next_contract"] = entry.get("next_contract")

    roll = _parse_date(entry.get("roll_date"))
    if roll is None:
        row["note"] = (f"roll_date is {entry.get('roll_date')!r}, which is not "
                       f"a date — this symbol has no countdown, and an absent "
                       f"one is not a distant one")
        return row

    row["roll_date"] = roll.isoformat()
    remaining = (roll - today).days
    row["days_remaining"] = remaining
    row["sessions_remaining"] = sessions_remaining(today, roll)

    if not row["next_contract"]:
        row["status"] = STATUS_ERROR
        row["note"] = ("no next_contract — there is nothing to roll INTO, so "
                       "the roll cannot be actioned from this file")
        return row

    if remaining < 0:
        row["status"] = STATUS_OVERDUE
        row["note"] = (f"roll date passed {abs(remaining)} day(s) ago; "
                       f"contract_resolver refuses {symbol} now and every "
                       f"order for it fails at dispatch")
    elif remaining == 0:
        row["status"] = STATUS_TODAY
        row["note"] = (f"last tradable session for {row['active_contract']} — "
                       f"roll to {row['next_contract']} before the next open")
    elif remaining <= ALERT_DAYS_URGENT:
        row["status"] = STATUS_URGENT
        row["note"] = (f"{remaining} day(s) to roll into "
                       f"{row['next_contract']}")
    elif remaining <= ALERT_DAYS_NOTICE:
        row["status"] = STATUS_NOTICE
        row["note"] = (f"{remaining} day(s) to roll into "
                       f"{row['next_contract']}")
    else:
        row["status"] = STATUS_OK
        row["note"] = f"{remaining} day(s) out"
    return row


def evaluate(blob: dict[str, Any], today: date) -> dict[str, Any]:
    rows = [evaluate_symbol(symbol, entry, today)
            for symbol, entry in blob.get("contracts", {}).items()]
    rows.sort(key=lambda r: (SEVERITY.get(r["status"], 9),
                             r["days_remaining"]
                             if r["days_remaining"] is not None else 9999,
                             r["symbol"]))
    alerts = [r for r in rows if r["status"] in ALERT_STATUSES]
    file_verdict = evaluate_file(blob, today)
    return {
        "today": today.isoformat(),
        "session_date_source": provenance(),
        "file": file_verdict,
        "rows": rows,
        "alerts": alerts,
        "alerting": bool(alerts) or file_verdict["stale"],
    }


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------

def format_text(report: dict[str, Any], show_all: bool) -> str:
    file_verdict = report["file"]
    lines = ["CONTRACT GUARD — config/contracts.json",
             f"  session date  {report['today']}  "
             f"({report['session_date_source']})",
             f"  file          {'STALE — ' if file_verdict['stale'] else ''}"
             f"{file_verdict['note']}",
             f"  symbols       {len(report['rows'])}  "
             f"({len(report['alerts'])} alerting)"]

    shown = report["rows"] if show_all else report["alerts"]
    if not shown:
        lines.append("")
        lines.append("  No roll inside the 5-day window and nothing overdue.")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"  {'SYMBOL':<7} {'STATUS':<11} {'DAYS':>5} {'SESS':>5}  "
                 f"{'ACTIVE':<12} {'NEXT':<12} NOTE")
    for row in shown:
        days = "—" if row["days_remaining"] is None else str(row["days_remaining"])
        sess = "—" if row["sessions_remaining"] is None else str(row["sessions_remaining"])
        lines.append(f"  {row['symbol']:<7} {row['status']:<11} {days:>5} "
                     f"{sess:>5}  {str(row['active_contract'] or '—'):<12} "
                     f"{str(row['next_contract'] or '—'):<12} {row['note']}")
    return "\n".join(lines)


def format_markdown(report: dict[str, Any]) -> str:
    """A card for the shared `#system-health` / `#roll-alerts` topic."""
    file_verdict = report["file"]
    out = ["*Contract guard — CME rolls*",
           f"Session date `{report['today']}`"]
    if file_verdict["stale"]:
        out.append(f"⛔️ *STALE FILE* — {file_verdict['note']}")

    if not report["alerts"]:
        out.append("")
        out.append("No roll inside the 5-day window and nothing overdue.")
        return "\n".join(out)

    out.append("")
    for row in report["alerts"]:
        icon = {STATUS_OVERDUE: "⛔️", STATUS_ERROR: "⛔️", STATUS_TODAY: "🔴",
                STATUS_URGENT: "🟠", STATUS_NOTICE: "🟡"}.get(row["status"], "•")
        days = "—" if row["days_remaining"] is None else row["days_remaining"]
        out.append(f"{icon} *{row['symbol']}* · {row['status']} · "
                   f"{days} day(s)  \n"
                   f"`{row['active_contract'] or '—'}` → "
                   f"`{row['next_contract'] or '—'}`  \n"
                   f"{row['note']}")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count down to each CME contract roll and alert at "
                    "5 and 3 days.")
    parser.add_argument("--contracts", default=str(DEFAULT_CONTRACTS),
                        help=f"contracts file (default {DEFAULT_CONTRACTS})")
    parser.add_argument("--today", default=None, metavar="YYYY-MM-DD",
                        help="override the session date (testing)")
    parser.add_argument("--all", action="store_true",
                        help="show every symbol, not just the alerting ones")
    parser.add_argument("--markdown", action="store_true",
                        help="emit a Markdown card for #roll-alerts")
    parser.add_argument("--json", action="store_true",
                        help="emit the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.today:
        today = _parse_date(args.today)
        if today is None:
            print(f"--today {args.today!r} is not a date", file=sys.stderr)
            return 2
    else:
        today = session_today()

    try:
        blob = load_contracts(Path(args.contracts))
    except ContractGuardError as exc:
        print(f"contract guard could not run: {exc}", file=sys.stderr)
        return 2

    report = evaluate(blob, today)

    if args.json:
        print(json.dumps(report, indent=2))
    elif args.markdown:
        print(format_markdown(report))
    else:
        print(format_text(report, args.all))

    return 1 if report["alerting"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
