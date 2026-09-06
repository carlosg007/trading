#!/usr/bin/env python3
"""
realtime/market_calendar.py - when CME futures are actually trading.

    from realtime.market_calendar import market_status, next_open

    is_open, why = market_status()          # (True, "open (Tue 09:31 ET)")
    reopen        = next_open()             # aware UTC instant

ONE DEFINITION, AND THAT IS THE ENTIRE POINT OF THE FILE
========================================================
The weekly schedule below was written in `scripts/watchdog.py` and lived only
there, which was fine while the watchdog was the only thing that asked. It is
now asked by two processes that must agree: the watchdog decides whether a
stale feed is a fault or a closed exchange, and `master_live.py` decides
whether to keep a process resident. A second copy of these four numbers would
be invisible when it drifted - the watchdog would suppress its alerts on a
schedule the loop was still trading, or the loop would stand down on a Sunday
evening the watchdog considered open, and every log line on both sides would
read correctly. `scripts.watchdog.market_status` is now an alias for the
function here.

THE SCHEDULE
============
    Sunday    18:00 ET   open
    Mon-Thu   17:00 ET   daily maintenance halt, one hour
    Mon-Thu   18:00 ET   open
    Friday    17:00 ET   weekend close, until Sunday 18:00 ET

**Exchange local time, never a fixed UTC offset.** 17:00 ET is 21:00 UTC in
summer and 22:00 UTC in winter; an offset hard-coded from either one is an hour
wrong for half the year, and the half it is wrong for contains the boundary
that stands a live loop down.

**The boundaries are half-open at the close and closed at the open** - 17:00:00
exactly is SHUT, 18:00:00 exactly is TRADING - so the windows meet with no
minute belonging to both and none belonging to neither. That is a `>=` and a
`<`, written once, here. The repository has already paid for one comparator
that was restated in prose and implemented differently (`mdlib/regimes.py`'s
ADX `>` against a spec that says `>=`); this one is pinned by
`tests/test_market_calendar.py` at the exact second on both sides.

FOUR PHASES, NOT A BOOLEAN
==========================
`market_status` answers the yes/no question the watchdog asks. `session_phase`
answers the one an operator asks at 17:30 on a Tuesday, which is *why*:

    OPEN         trading
    MAINTENANCE  the daily one-hour halt. Reopens in under an hour.
    WEEKEND      Friday 17:00 ET to Sunday 18:00 ET.
    HOLIDAY      an exchange holiday, from the calendar file below.

The distinction matters to anything that decides how long to wait. A process
that exits into a MAINTENANCE break comes back in under an hour; one that exits
into a WEEKEND is gone for 49.

EXCHANGE HOLIDAYS ARE READ, NOT GUESSED
=======================================
There is no arithmetic rule for the CME holiday calendar - it is published, it
follows US federal holidays only approximately, and it mixes full closures with
early closes at times that change year to year. `backtest/event_calendar.py`
already carries the lesson: a generated date that lands in the right week and
the wrong day is worse than no filter at all, because the report says the run
was filtered.

So holidays come from a FILE and from nowhere else:

    $BT_CME_HOLIDAYS, else /mnt/backtest/reference/calendar/cme_holidays.csv

    session_date,status,close_et,note
    2026-11-26,closed,,Thanksgiving
    2026-11-27,early,13:00,day after Thanksgiving
    2026-12-25,closed,,Christmas
    2026-12-24,early,13:00,Christmas Eve

**With no file, no holiday is modelled and the market reads OPEN on one.** That
asymmetry is deliberate and is the safe direction: a holiday this module does
not know about costs one idle day of a process that finds no new bars and does
nothing, while a holiday it invents stands a live loop down on a trading day.
`holiday_provenance()` states which of the two you have, and `describe()` puts
it in the startup banner rather than leaving it to be assumed.

A file that EXISTS and cannot be parsed raises `MarketCalendarError` instead of
degrading to "no holidays". An operator who wrote the file meant to have
holidays, and silently not having them is the failure they would never see.

**Keyed on the CME SESSION date**, the 18:00 ET roll that
`backtest.event_calendar.session_date` owns for the backtests - so `2026-12-25,
closed` shuts the window from Dec 24 18:00 ET to Dec 25 17:00 ET, which is the
session that is actually closed. Pairing it with `2026-12-24,early,13:00`
expresses the real CME Christmas schedule exactly: shut from Dec 24 13:00 ET
until Dec 25 18:00 ET. A calendar-date key would have closed the wrong twenty
three hours and looked right.

THE FILE IS READ ONCE PER PROCESS
=================================
It lives on the NFS mount, which is mounted `hard` - a read that blocks blocks
forever rather than returning an error - so a per-cycle read would put an NFS
stall directly in the path of a live dispatch loop. It is cached on first use
and re-read only through `load_holidays(refresh=True)`. Since `master_live.py`
now exits at every close and starts again at every open, the cache is at most
one session old by construction.

    python3 realtime/market_calendar.py
    python3 realtime/market_calendar.py --now 2026-12-25T20:00:00Z
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

#: The zone the schedule is DEFINED in. Must equal
#: `backtest.event_calendar.ET`; `tests/test_market_calendar.py` asserts it
#: rather than trusting two string literals to stay spelled the same.
ET_NAME = "America/New_York"
EXCHANGE_TZ = ZoneInfo(ET_NAME)

#: 17:00 ET shuts, 18:00 ET opens. Both are Friday's weekend boundary and the
#: Monday-Thursday maintenance boundary; they are the same two numbers and are
#: named once each rather than four times.
DAILY_CLOSE_HOUR = 17
DAILY_OPEN_HOUR = 18

#: Monday is 0, matching `datetime.weekday()`.
FRIDAY, SATURDAY, SUNDAY = 4, 5, 6

OPEN = "OPEN"
MAINTENANCE = "MAINTENANCE"
WEEKEND = "WEEKEND"
HOLIDAY = "HOLIDAY"

HOLIDAY_ENV = "BT_CME_HOLIDAYS"
DEFAULT_HOLIDAY_FILE = Path("/mnt/backtest/reference/calendar/cme_holidays.csv")

#: How far `next_open`/`next_close` will look before giving up. The longest
#: ordinary closure is 49 hours; nine days covers a holiday calendar that
#: brackets a weekend and still refuses to scan forever if somebody writes a
#: file that closes every session.
SEARCH_HORIZON_DAYS = 9


class MarketCalendarError(RuntimeError):
    """
    The calendar cannot answer the question it was asked.

    Raised rather than returning a default. "No holidays" and "a holiday file
    that did not parse" are different facts, and only one of them is safe to
    trade on.
    """


@dataclass(frozen=True)
class Holiday:
    """
    One non-standard session, keyed on the CME session date.

    `full` closes the whole session (the previous evening's 18:00 ET open
    through 17:00 ET). `close_et` instead ends it early, on the session date's
    own calendar day - which is what an early close actually is.
    """

    session_date: date
    full: bool
    close_et: time | None
    note: str = ""

    def describe(self) -> str:
        if self.full:
            what = "closed"
        else:
            what = f"early close {self.close_et:%H:%M} ET"
        return (f"{self.session_date:%Y-%m-%d} {what}"
                + (f" ({self.note})" if self.note else ""))


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------
def _as_utc(now: datetime | None) -> datetime:
    """
    Anything a caller holds an instant in -> an aware UTC datetime.

    A NAIVE datetime is treated as UTC, matching
    `backtest.event_calendar._as_utc_index` and the lake's guarantee that every
    timestamp in this repository is UTC. Localizing it to the machine's zone
    instead would move every boundary by the operator's offset, silently, and
    the machine's zone is not a fact about the exchange.
    """
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def session_date_of(moment: datetime) -> date:
    """
    The CME session date `moment` belongs to.

    The same 18:00 ET roll `backtest.event_calendar.session_date` applies to a
    whole index, for one instant and without pandas: at or after 18:00 ET the
    bar belongs to the NEXT calendar day's session. A Sunday 23:00 ET print is
    Monday's session, and a holiday table keyed any other way closes the wrong
    day.
    """
    local = _as_utc(moment).astimezone(EXCHANGE_TZ)
    if local.hour >= DAILY_OPEN_HOUR:
        return (local + timedelta(days=1)).date()
    return local.date()


# --------------------------------------------------------------------------
# the holiday file
# --------------------------------------------------------------------------
_CACHE: dict[str, tuple[dict[date, Holiday], str]] = {}


def holiday_path(path: str | Path | None = None) -> Path:
    """Where holidays are read from: the argument, $BT_CME_HOLIDAYS, the default."""
    if path:
        return Path(path)
    return Path(os.environ.get(HOLIDAY_ENV) or DEFAULT_HOLIDAY_FILE)


def _parse_close(text: str, lineno: int) -> time:
    try:
        hh, mm = text.strip().split(":")
        return time(int(hh), int(mm))
    except Exception as exc:                                       # noqa: BLE001
        raise MarketCalendarError(
            f"line {lineno}: close_et {text!r} is not HH:MM ({exc})") from exc


def _parse(text: str, source: str) -> dict[date, Holiday]:
    out: dict[date, Holiday] = {}
    for lineno, row in enumerate(csv.reader(text.splitlines()), start=1):
        if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
            continue
        first = row[0].strip().lower()
        if first in ("session_date", "date"):        # a header, not a holiday
            continue
        cells = [c.strip() for c in row] + ["", "", ""]
        try:
            when = date.fromisoformat(cells[0])
        except ValueError as exc:
            raise MarketCalendarError(
                f"{source} line {lineno}: {cells[0]!r} is not an ISO date "
                f"({exc})") from exc
        status = cells[1].lower() or "closed"
        if status in ("closed", "full", "holiday"):
            entry = Holiday(when, True, None, cells[3])
        elif status in ("early", "early_close", "half"):
            if not cells[2]:
                raise MarketCalendarError(
                    f"{source} line {lineno}: status 'early' needs a close_et "
                    f"time; an early close with no time is not a schedule")
            entry = Holiday(when, False, _parse_close(cells[2], lineno),
                            cells[3])
        else:
            raise MarketCalendarError(
                f"{source} line {lineno}: status {cells[1]!r} is neither "
                f"'closed' nor 'early'")
        if when in out:
            raise MarketCalendarError(
                f"{source} line {lineno}: {when} appears twice; a session has "
                f"one schedule")
        out[when] = entry
    return out


def load_holidays(path: str | Path | None = None,
                  refresh: bool = False) -> dict[date, Holiday]:
    """
    `{session_date: Holiday}`, cached per path for the life of the process.

    A MISSING file yields `{}` and is not an error - see the module docstring:
    no calendar means no holiday is modelled, which is the safe direction. A
    file that exists and does not parse RAISES, because an operator who wrote
    one meant to have holidays and silently not having them is the failure they
    would never see.
    """
    target = holiday_path(path)
    key = str(target)
    if refresh:
        _CACHE.pop(key, None)
    if key in _CACHE:
        return _CACHE[key][0]

    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        table, source = {}, (f"no holiday file at {target}; exchange holidays "
                             f"are NOT modelled")
    except OSError as exc:
        raise MarketCalendarError(
            f"holiday file {target} exists but cannot be read: {exc}") from exc
    else:
        table = _parse(text, str(target))
        source = f"{len(table)} session(s) from {target}"
    _CACHE[key] = (table, source)
    return table


def holiday_provenance(path: str | Path | None = None) -> str:
    """One line naming where holidays came from, or saying there are none."""
    load_holidays(path)
    return _CACHE[str(holiday_path(path))][1]


# --------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------
def session_phase(now: datetime | None = None,
                  holidays: dict[date, Holiday] | None = None
                  ) -> tuple[str, str]:
    """
    `(phase, reason)` - one of OPEN / MAINTENANCE / WEEKEND / HOLIDAY.

    The weekly schedule is checked first and the holiday table only overlays a
    window the ordinary week already calls open. A holiday entry for a Saturday
    therefore changes nothing rather than fighting the weekend rule, and a
    malformed calendar cannot open a market that is shut.
    """
    now = _as_utc(now)
    local = now.astimezone(EXCHANGE_TZ)
    dow, hour = local.weekday(), local.hour

    if dow == FRIDAY and hour >= DAILY_CLOSE_HOUR:
        return WEEKEND, "weekend — closed Friday 17:00 ET"
    if dow == SATURDAY:
        return WEEKEND, "weekend — Saturday"
    if dow == SUNDAY and hour < DAILY_OPEN_HOUR:
        return WEEKEND, "weekend — reopens Sunday 18:00 ET"
    if dow <= 3 and DAILY_CLOSE_HOUR <= hour < DAILY_OPEN_HOUR:
        return MAINTENANCE, "daily maintenance break 17:00-18:00 ET"

    table = load_holidays() if holidays is None else holidays
    entry = table.get(session_date_of(local))
    if entry is not None:
        if entry.full:
            return HOLIDAY, f"exchange holiday — {entry.describe()}"
        # An early close ends the session on its OWN calendar day. The evening
        # half of the same session - the previous day after 18:00 ET - is
        # ordinary trading and must not be swept up by it.
        if (entry.close_et is not None
                and local.date() == entry.session_date
                and local.time() >= entry.close_et):
            return HOLIDAY, f"exchange holiday — {entry.describe()}"
    return OPEN, f"open ({local:%a %H:%M} ET)"


def market_status(now: datetime | None = None,
                  holidays: dict[date, Holiday] | None = None
                  ) -> tuple[bool, str]:
    """
    `(is_open, reason)` for CME futures, in exchange local time.

    The watchdog's original signature, preserved exactly:
    `scripts.watchdog.market_status` is this function.
    """
    phase, reason = session_phase(now, holidays)
    return phase == OPEN, reason


def _scan(now: datetime, want_open: bool,
          holidays: dict[date, Holiday] | None) -> datetime:
    """
    The earliest instant at or after `now` whose openness is `want_open`.

    A minute-by-minute walk rather than closed-form arithmetic. Every boundary
    in the weekly schedule is on the hour and every early close is on the
    minute, so a minute step lands on all of them exactly; and the alternative -
    enumerating candidate boundaries and then reasoning about which of a
    holiday, a weekend and a maintenance break wins - is where the version of
    this function that returns a plausible wrong answer lives. At most
    ~13,000 cheap iterations, and it is called at a process's start and end,
    not inside a cycle.
    """
    if holidays is None:
        holidays = load_holidays()
    probe = now.replace(second=0, microsecond=0)
    limit = probe + timedelta(days=SEARCH_HORIZON_DAYS)
    while probe <= limit:
        if ((session_phase(probe, holidays)[0] == OPEN) is want_open):
            # `probe` may be the floor of `now`; they share a minute and
            # therefore a phase, so the answer is `now` itself.
            return now if probe <= now else probe
        probe += timedelta(minutes=1)
    raise MarketCalendarError(
        f"no {'open' if want_open else 'closed'} minute within "
        f"{SEARCH_HORIZON_DAYS} days of {now:%Y-%m-%d %H:%M}Z — check the "
        f"holiday calendar ({holiday_provenance()})")


def next_open(now: datetime | None = None,
              holidays: dict[date, Holiday] | None = None) -> datetime:
    """
    The next instant the market is open, as aware UTC. `now` if it is open now.

    Returning `now` rather than the next *transition* is what a caller waiting
    for the open wants: `sleep((next_open(now) - now).total_seconds())` is
    correct in both states and needs no branch.
    """
    return _scan(_as_utc(now), True, holidays)


def next_close(now: datetime | None = None,
               holidays: dict[date, Holiday] | None = None) -> datetime:
    """The next instant the market is closed, as aware UTC. `now` if shut now."""
    return _scan(_as_utc(now), False, holidays)


def seconds_until_open(now: datetime | None = None,
                       holidays: dict[date, Holiday] | None = None) -> float:
    """`0.0` while the market is open, otherwise the wait in seconds."""
    now = _as_utc(now)
    return max(0.0, (next_open(now, holidays) - now).total_seconds())


def describe(now: datetime | None = None,
             holidays: dict[date, Holiday] | None = None) -> str:
    """
    The startup-banner line: the phase, why, the next boundary, the provenance.

    The provenance is on the banner rather than in a docstring because "no
    holiday file" and "a holiday file with twelve dates" produce identical
    behaviour on every ordinary Tuesday and different behaviour on exactly the
    days somebody would want to check afterwards.
    """
    now = _as_utc(now)
    phase, reason = session_phase(now, holidays)
    local = now.astimezone(EXCHANGE_TZ)
    if phase == OPEN:
        shut = next_close(now, holidays)
        boundary = (f"closes {shut.astimezone(EXCHANGE_TZ):%a %H:%M} ET "
                    f"(in {(shut - now).total_seconds() / 3600:.1f}h)")
    else:
        back = next_open(now, holidays)
        boundary = (f"reopens {back.astimezone(EXCHANGE_TZ):%a %H:%M} ET "
                    f"(in {(back - now).total_seconds() / 3600:.1f}h)")
    return (f"[market] {phase}: {reason}; {boundary}. "
            f"Now {local:%Y-%m-%d %H:%M:%S} ET / {now:%H:%M:%S} UTC. "
            f"Holidays: {holiday_provenance()}.")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="market_calendar.py",
        description="Is the CME open? Reads nothing but the clock and the "
                    "holiday file; places no orders.")
    ap.add_argument("--now", default=None,
                    help="an ISO instant to evaluate instead of the clock, "
                         "e.g. 2026-12-25T20:00:00Z. Naive input is UTC")
    ap.add_argument("--holidays", default=None,
                    help=f"holiday CSV (default ${HOLIDAY_ENV}, else "
                         f"{DEFAULT_HOLIDAY_FILE})")
    args = ap.parse_args(argv)

    when = None
    if args.now:
        when = datetime.fromisoformat(args.now.replace("Z", "+00:00"))
    try:
        table = load_holidays(args.holidays)
    except MarketCalendarError as exc:
        print(f"[market] REFUSING TO ANSWER: {exc}")
        return 2
    print(describe(when, table))
    for entry in sorted(table.values(), key=lambda h: h.session_date)[:10]:
        print(f"           {entry.describe()}")
    return 0 if market_status(when, table)[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
