#!/usr/bin/env python3
"""
The CME schedule gate, and every way it could be right about the wrong hour.

Location:  ~/src/trading/tests/test_market_calendar.py

    .venv/bin/pytest tests/test_market_calendar.py

ASSERT-BASED, so `tests/conftest.py` collects it normally and every case
reports through pytest. There is no `check()` helper here on purpose: read
conftest before adding one, because a collector-style suite reports GREEN while
its checks fail. Inner helpers are named `_*` rather than `test_*` for the
other half of that trap - pytest calls any module-level `test_*` whose only
argument is defaulted, and one that did real work would run outside the
fixtures that were supposed to contain it.

READS NO BARS AND OPENS NO SOCKET. Every case is an instant and an expected
verdict, worked out from the published CME schedule above the code.

WHAT IS PINNED, AND WHY EACH ONE IS A WAY THE GATE COULD MISLEAD
===============================================================
  * **The boundaries, at the exact second.** 17:00:00 is SHUT and 18:00:00 is
    TRADING. An off-by-one at either end is invisible in every log - the loop
    stands down a minute early, or trades a minute into a halt where there are
    no bars anyway - and the repository has already paid once for a comparator
    that was `>` in the code and `>=` in the prose (`mdlib/regimes.py`'s ADX).
  * **Daylight saving, on both sides of it.** The schedule is defined in ET.
    17:00 ET is 22:00 UTC in January and 21:00 UTC in July, so a gate built on
    a fixed offset is right for half the year - and the half it is wrong for
    still produces a plausible weekend, one hour displaced.
  * **WEEKEND and MAINTENANCE stay distinct.** Both are "closed" and they cost
    different things: one is a 60-minute halt, the other is 49 hours of
    unwatched gap risk. A supervisor that could not tell them apart would
    treat them the same, and the close report's loud line is drawn on it.
  * **A naive datetime is UTC**, matching the lake and
    `backtest.event_calendar`. Read as machine-local it would move every
    boundary by the operator's offset, silently.
  * **Holidays are keyed on the SESSION date**, the 18:00 ET roll. A
    calendar-date key closes the wrong twenty-three hours and looks right.
  * **A holiday file that does not parse RAISES.** "No holidays" and "a
    holiday file nobody could read" must never produce the same behaviour.
  * **There is ONE definition.** `scripts.watchdog.market_status` is this
    module's function, not a copy. Two processes disagreeing about a Sunday
    evening would each log correctly while one of them was wrong.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from realtime import market_calendar as mc                         # noqa: E402

ET = ZoneInfo("America/New_York")

#: Every case below passes this explicitly so no test ever reads the operator's
#: real holiday file off the NFS mount - a suite whose verdict changed when
#: somebody dropped a CSV on a server would be worse than no suite.
NO_HOLIDAYS: dict = {}


def _et(text: str) -> datetime:
    """An ET wall-clock string as an aware instant, DST resolved by the zone."""
    return datetime.fromisoformat(text).replace(tzinfo=ET)


# --------------------------------------------------------------------------
# the weekly schedule
# --------------------------------------------------------------------------
# 2026-08-28 is a Friday, 08-29 a Saturday, 08-30 a Sunday, 08-31 a Monday.
@pytest.mark.parametrize("when,is_open,why", [
    ("2026-08-28 16:59", True,  "Friday, one minute before the weekend close"),
    ("2026-08-28 17:00", False, "the weekend starts AT 17:00, not after it"),
    ("2026-08-28 17:01", False, "Friday evening is the weekend"),
    ("2026-08-29 00:00", False, "Saturday, midnight"),
    ("2026-08-29 12:00", False, "Saturday, midday"),
    ("2026-08-29 23:59", False, "Saturday, all of it"),
    ("2026-08-30 17:59", False, "Sunday, one minute before the reopen"),
    ("2026-08-30 18:00", True,  "the reopen is inclusive"),
    ("2026-08-30 18:01", True,  "Sunday evening IS a trading session"),
    ("2026-08-31 16:59", True,  "Monday, before the halt"),
    ("2026-08-31 17:30", False, "Monday maintenance break"),
    ("2026-08-31 18:00", True,  "maintenance ends AT 18:00"),
    ("2026-09-02 02:00", True,  "Wednesday overnight IS a trading session"),
    ("2026-09-03 17:30", False, "Thursday maintenance break"),
])
def test_the_published_cme_schedule(when, is_open, why):
    assert mc.market_status(_et(when), NO_HOLIDAYS)[0] is is_open, why


def test_the_fixtures_are_the_weekdays_they_claim_to_be():
    """A schedule test on the wrong weekday passes for the wrong reason."""
    assert _et("2026-08-28 12:00").weekday() == 4, "expected a Friday"
    assert _et("2026-08-29 12:00").weekday() == 5, "expected a Saturday"
    assert _et("2026-08-30 12:00").weekday() == 6, "expected a Sunday"
    assert _et("2026-08-31 12:00").weekday() == 0, "expected a Monday"


@pytest.mark.parametrize("when,phase", [
    ("2026-08-28 17:00", mc.WEEKEND),
    ("2026-08-29 12:00", mc.WEEKEND),
    ("2026-08-30 12:00", mc.WEEKEND),
    ("2026-08-31 17:30", mc.MAINTENANCE),
    ("2026-09-01 17:00", mc.MAINTENANCE),
    ("2026-09-03 17:59", mc.MAINTENANCE),
    ("2026-09-01 10:00", mc.OPEN),
])
def test_a_60_minute_halt_and_a_49_hour_weekend_are_not_the_same_closure(
        when, phase):
    """Both read 'closed'. They cost different things, so they are named."""
    assert mc.session_phase(_et(when), NO_HOLIDAYS)[0] == phase


def test_the_boundaries_meet_with_no_minute_in_both_and_none_in_neither():
    """17:00:00 shut, 17:59:59 shut, 18:00:00 open — to the second."""
    assert mc.market_status(_et("2026-08-31 16:59:59"), NO_HOLIDAYS)[0] is True
    assert mc.market_status(_et("2026-08-31 17:00:00"), NO_HOLIDAYS)[0] is False
    assert mc.market_status(_et("2026-08-31 17:59:59"), NO_HOLIDAYS)[0] is False
    assert mc.market_status(_et("2026-08-31 18:00:00"), NO_HOLIDAYS)[0] is True


# --------------------------------------------------------------------------
# daylight saving
# --------------------------------------------------------------------------
def _first_friday(year: int, month: int) -> date:
    day = date(year, month, 1)
    return day + timedelta(days=(4 - day.weekday()) % 7)


def test_the_same_wall_clock_closes_the_week_in_january_and_in_july():
    """
    The schedule is ET. A gate built on a fixed UTC offset passes half the
    year and is an hour wrong for the other half - and the wrong half still
    produces a weekend that looks entirely reasonable in the log.
    """
    winter = datetime.combine(_first_friday(2026, 1), time(17, 0), tzinfo=ET)
    summer = datetime.combine(_first_friday(2026, 7), time(17, 0), tzinfo=ET)
    assert winter.weekday() == summer.weekday() == 4

    assert mc.market_status(winter, NO_HOLIDAYS)[0] is False
    assert mc.market_status(summer, NO_HOLIDAYS)[0] is False
    assert mc.market_status(winter - timedelta(minutes=1), NO_HOLIDAYS)[0] is True
    assert mc.market_status(summer - timedelta(minutes=1), NO_HOLIDAYS)[0] is True

    # And the offsets really do differ, so the two cases above are not the
    # same instant wearing two labels.
    assert winter.astimezone(timezone.utc).hour == 22, "EST is UTC-5"
    assert summer.astimezone(timezone.utc).hour == 21, "EDT is UTC-4"


def test_the_gate_is_driven_from_utc_instants_not_wall_clock_strings():
    """The loop holds UTC. Both spellings of one instant must agree."""
    for utc_hour, is_open in ((21, True), (22, False)):     # a January Friday
        stamp = datetime(2026, 1, 2, utc_hour, 0, tzinfo=timezone.utc)
        assert stamp.weekday() == 4
        assert mc.market_status(stamp, NO_HOLIDAYS)[0] is is_open


def test_a_naive_datetime_is_utc_not_the_machines_zone():
    naive = datetime(2026, 8, 29, 12, 0)                    # a Saturday
    aware = naive.replace(tzinfo=timezone.utc)
    assert mc.market_status(naive, NO_HOLIDAYS) == mc.market_status(
        aware, NO_HOLIDAYS)


# --------------------------------------------------------------------------
# the next boundary
# --------------------------------------------------------------------------
def test_next_open_crosses_the_whole_weekend():
    friday_close = _et("2026-08-28 17:00")
    assert mc.next_open(friday_close, NO_HOLIDAYS) == _et("2026-08-30 18:00")
    assert mc.seconds_until_open(friday_close, NO_HOLIDAYS) == 49 * 3600


def test_next_open_crosses_only_the_maintenance_halt():
    monday_halt = _et("2026-08-31 17:00")
    assert mc.next_open(monday_halt, NO_HOLIDAYS) == _et("2026-08-31 18:00")
    assert mc.seconds_until_open(monday_halt, NO_HOLIDAYS) == 3600


def test_next_open_is_now_when_the_market_is_already_open():
    """`sleep(next_open(now) - now)` must be correct in BOTH states."""
    trading = _et("2026-09-01 10:17:33")
    assert mc.next_open(trading, NO_HOLIDAYS) == trading
    assert mc.seconds_until_open(trading, NO_HOLIDAYS) == 0.0


def test_next_close_finds_the_end_of_the_session():
    assert mc.next_close(_et("2026-08-31 09:00"),
                         NO_HOLIDAYS) == _et("2026-08-31 17:00")
    shut = _et("2026-08-29 12:00")
    assert mc.next_close(shut, NO_HOLIDAYS) == shut


def test_the_reopen_is_returned_in_utc_however_it_was_asked():
    answer = mc.next_open(_et("2026-08-29 12:00"), NO_HOLIDAYS)
    assert answer.tzinfo is not None
    assert answer.astimezone(ET).hour == 18


# --------------------------------------------------------------------------
# session dates
# --------------------------------------------------------------------------
def test_the_18_00_roll_is_the_backtests_roll():
    """
    `session_date_of` must agree with `backtest.event_calendar.session_date`.

    Two spellings of the CME session roll would put the live holiday gate on
    one day and every backtest attribution on another, and both would look
    right. Checked across a whole week at hourly resolution rather than on the
    two obvious cases.
    """
    import pandas as pd
    from backtest.event_calendar import session_date

    stamps = pd.date_range("2026-08-28", periods=24 * 7, freq="h", tz="UTC")
    expected = session_date(stamps)
    for stamp, want in zip(stamps, expected):
        assert mc.session_date_of(stamp.to_pydatetime()) == want.date(), stamp


def test_a_sunday_evening_print_belongs_to_mondays_session():
    assert mc.session_date_of(_et("2026-08-30 23:00")) == date(2026, 8, 31)
    assert mc.session_date_of(_et("2026-08-31 09:00")) == date(2026, 8, 31)
    assert mc.session_date_of(_et("2026-08-31 17:30")) == date(2026, 8, 31)
    assert mc.session_date_of(_et("2026-08-31 18:00")) == date(2026, 9, 1)


# --------------------------------------------------------------------------
# exchange holidays
# --------------------------------------------------------------------------
#: The real CME Thanksgiving 2026 shape: Thursday's session closed outright,
#: Friday's session ending early at 13:00 ET. 2026-11-26 is a Thursday.
THANKSGIVING_CSV = """session_date,status,close_et,note
# a comment, and a blank line, both ignored

2026-11-26,closed,,Thanksgiving
2026-11-27,early,13:00,day after Thanksgiving
"""


@pytest.fixture()
def thanksgiving(tmp_path):
    path = tmp_path / "cme_holidays.csv"
    path.write_text(THANKSGIVING_CSV, encoding="utf-8")
    return mc.load_holidays(path, refresh=True)


def test_the_holiday_fixture_is_the_weekday_it_claims_to_be():
    assert date(2026, 11, 26).weekday() == 3, "expected a Thursday"


@pytest.mark.parametrize("when,phase,why", [
    ("2026-11-25 16:00", mc.OPEN,
     "Wednesday afternoon is an ordinary session"),
    ("2026-11-25 19:00", mc.HOLIDAY,
     "Wednesday 19:00 ET is already THURSDAY's session, and it is closed"),
    ("2026-11-26 10:00", mc.HOLIDAY, "Thanksgiving Thursday"),
    ("2026-11-26 16:59", mc.HOLIDAY, "still Thursday's closed session"),
    ("2026-11-26 17:30", mc.MAINTENANCE,
     "the ordinary halt wins; both are shut and the reason is the halt"),
    ("2026-11-26 18:00", mc.OPEN,
     "Thursday 18:00 ET opens FRIDAY's session, which trades"),
    ("2026-11-27 12:59", mc.OPEN, "Friday, one minute before the early close"),
    ("2026-11-27 13:00", mc.HOLIDAY, "the early close is inclusive"),
    ("2026-11-27 17:30", mc.WEEKEND, "and then the ordinary weekend"),
])
def test_a_holiday_closes_the_session_not_the_calendar_day(
        thanksgiving, when, phase, why):
    assert mc.session_phase(_et(when), thanksgiving)[0] == phase, why


def test_an_early_close_does_not_swallow_its_own_evening_open(thanksgiving):
    """
    The evening HALF of a session is the previous calendar day after 18:00 ET.

    Friday's session is `2026-11-27` and starts Thursday 19:00 ET. An early
    close keyed on the session date alone, with no check that the instant is
    on the session date's own calendar day, would shut Thursday evening too -
    an extra five hours nobody asked for, on a session that was trading.
    """
    assert mc.session_date_of(_et("2026-11-26 19:00")) == date(2026, 11, 27)
    assert mc.session_phase(_et("2026-11-26 19:00"), thanksgiving)[0] == mc.OPEN


def test_next_open_steps_over_a_closed_session(thanksgiving):
    assert mc.next_open(_et("2026-11-26 10:00"),
                        thanksgiving) == _et("2026-11-26 18:00")
    # After the Friday early close the next open is the ordinary Sunday one.
    assert mc.next_open(_et("2026-11-27 13:00"),
                        thanksgiving) == _et("2026-11-29 18:00")


def test_a_holiday_cannot_open_a_market_the_week_already_closed(tmp_path):
    """The weekly schedule is checked first; the table only overlays it."""
    path = tmp_path / "h.csv"
    path.write_text("2026-08-29,closed,,a Saturday entry\n", encoding="utf-8")
    table = mc.load_holidays(path, refresh=True)
    assert mc.session_phase(_et("2026-08-29 12:00"), table)[0] == mc.WEEKEND


# --------------------------------------------------------------------------
# the holiday file itself
# --------------------------------------------------------------------------
def test_a_missing_file_models_no_holiday_and_says_so(tmp_path):
    """
    The safe direction, and it is asymmetric on purpose.

    A holiday nobody told us about costs one idle day of a process that finds
    no bars. An invented holiday costs a trading day. So the absence of a file
    is not an error - but it IS stated, because "no file" and "twelve dates"
    behave identically on every ordinary Tuesday.
    """
    missing = tmp_path / "nope.csv"
    assert mc.load_holidays(missing, refresh=True) == {}
    assert "NOT modelled" in mc.holiday_provenance(missing)


def test_a_present_file_names_itself_in_the_provenance(tmp_path):
    path = tmp_path / "cme.csv"
    path.write_text(THANKSGIVING_CSV, encoding="utf-8")
    mc.load_holidays(path, refresh=True)
    assert "2 session(s)" in mc.holiday_provenance(path)
    assert str(path) in mc.holiday_provenance(path)


@pytest.mark.parametrize("body,fragment", [
    ("not-a-date,closed,,\n", "is not an ISO date"),
    ("2026-11-26,shut,,\n", "neither 'closed' nor 'early'"),
    ("2026-11-27,early,,\n", "needs a close_et"),
    ("2026-11-27,early,noon,\n", "is not HH:MM"),
    ("2026-11-26,closed,,\n2026-11-26,early,13:00,\n", "appears twice"),
])
def test_a_file_that_does_not_parse_raises_rather_than_degrading(
        tmp_path, body, fragment):
    """
    An operator who wrote the file meant to have holidays.

    Falling back to an empty table would trade a closed session with every log
    line reading correctly - the failure they would never see. `master_live`
    turns this into REFUSING TO START.
    """
    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(mc.MarketCalendarError) as exc:
        mc.load_holidays(path, refresh=True)
    assert fragment in str(exc.value)


def test_the_file_is_read_once_per_process(tmp_path):
    """
    It lives on a HARD-mounted NFS share, where a blocked read blocks forever.

    A per-cycle read would put that stall directly in a live dispatch loop.
    """
    path = tmp_path / "cme.csv"
    path.write_text("2026-11-26,closed,,Thanksgiving\n", encoding="utf-8")
    first = mc.load_holidays(path, refresh=True)
    path.write_text("2026-12-25,closed,,Christmas\n", encoding="utf-8")
    assert mc.load_holidays(path) == first, "the cache must hold"
    assert mc.load_holidays(path, refresh=True) != first, "refresh must not"


# --------------------------------------------------------------------------
# one definition
# --------------------------------------------------------------------------
def test_the_watchdog_and_the_loop_share_one_schedule():
    """
    Not "agree on". ARE. `scripts.watchdog.market_status` is this function.

    A copy would be invisible when it drifted: the watchdog would suppress its
    alerts on a schedule the loop was still trading, or the loop would stand
    down on a Sunday evening the watchdog called open, and every log line on
    both sides would read correctly.
    """
    import scripts.watchdog as wd
    assert wd.market_status is mc.market_status
    assert wd.EXCHANGE_TZ is mc.EXCHANGE_TZ


def test_the_zone_is_the_backtests_zone():
    from backtest.event_calendar import ET as BACKTEST_ET
    assert mc.ET_NAME == BACKTEST_ET


# --------------------------------------------------------------------------
# the wiring into master_live and its unit file
# --------------------------------------------------------------------------
UNIT = REPO / "deploy" / "systemd" / "trading-master-live.service"
TIMER = REPO / "deploy" / "systemd" / "trading-master-live.timer"


def test_the_gate_is_off_unless_something_asks_for_it():
    """
    A dry run started by hand at 17:30 must keep printing, not vanish.

    The unit file is where the supervised process asks for the opposite.
    """
    import master_live
    args = master_live.build_parser().parse_args([])
    assert args.halt_when_closed is False
    assert args.closed_exit_code == master_live.EXIT_MARKET_CLOSED == 3


def test_every_documented_flag_still_parses():
    """The gate must not have cost an existing command its spelling."""
    import master_live
    args = master_live.build_parser().parse_args([
        "--live", "--tf", "1h", "--feed", "auto", "--interval-sec", "60",
        "--session-cutoff-utc", "20:55", "--max-session-loss-usd", "1000",
        "--max-regime-write-age-sec", "900", "--halt-when-closed"])
    assert args.live and args.tf == "1h" and args.feed == "auto"
    assert args.interval_sec == 60 and args.session_cutoff_utc == "20:55"
    assert args.max_session_loss_usd == 1000
    assert args.max_regime_write_age_sec == 900
    assert args.halt_when_closed is True


def test_the_flag_and_the_restart_policy_are_one_decision():
    """
    TWO FILES, ONE KEY - and here the two keys are in the same file.

    `--halt-when-closed` makes the loop exit 3 at every close.
    `RestartPreventExitStatus=` is the only thing that makes systemd honour
    it. With the flag and without the 3, `Restart=always` brings the process
    straight back up: it would exit at 17:00, restart at 17:00:10, and spend
    the weekend loading a dispatcher every ten seconds - strictly worse than
    the resident loop this replaced, and every log line would read correctly.
    """
    unit = UNIT.read_text(encoding="utf-8")
    assert "--halt-when-closed" in unit
    prevent = [ln for ln in unit.splitlines()
               if ln.startswith("RestartPreventExitStatus=")]
    assert len(prevent) == 1, prevent
    codes = prevent[0].split("=", 1)[1].split()
    import master_live
    assert str(master_live.EXIT_MARKET_CLOSED) in codes
    assert "2" in codes, "REFUSING TO START must still not be retried"


def test_the_timer_never_starts_the_loop_into_the_weekend():
    """
    Every open window begins 18:00 ET on a Sunday, Monday, Tuesday, Wednesday
    or Thursday. A Friday or Saturday firing would start a process that stands
    itself down immediately - harmless, and a log full of harmless noise is
    where a real failure hides.
    """
    timer = TIMER.read_text(encoding="utf-8")
    line = next(ln for ln in timer.splitlines()
                if ln.startswith("OnCalendar="))
    assert "Fri" not in line and "Sat" not in line
    for day in ("Sun", "Mon", "Tue", "Wed", "Thu"):
        assert day in line, day
    # THE ZONE IS PART OF THE EXPRESSION. 17:55 ET is 21:55 UTC in summer and
    # 22:55 in winter; written in UTC this fires an hour off for half the year.
    assert mc.ET_NAME in line
    assert "17:55" in line
    assert "Persistent=false" in timer, "a catch-up would fire into a closure"


# --------------------------------------------------------------------------
# the close itself
# --------------------------------------------------------------------------
class _Book:
    def __init__(self, held):
        self._held = held

    def open_positions(self):
        return list(self._held)


class _Dispatcher:
    """
    A stub that RAISES on everything except the one read the close is allowed.

    The close must not be able to send, flatten, or evaluate. Stubbing the
    permitted call and booby-trapping the rest is what makes that a test
    rather than a promise in a docstring - `send_execution_signal`,
    `process_bar_cycle` and `dispatch_exits` all land on `__getattr__`.
    """

    def __init__(self, held=()):
        self.positions = _Book(held)

    def __getattr__(self, name):
        raise AssertionError(
            f"the market-hours close reached dispatcher.{name}; it may read "
            f"the position book and nothing else")


class _State:
    path = "data/engine_state.json"

    def __init__(self, claims=()):
        self._claims = list(claims)

    def open_claims(self):
        return list(self._claims)


def test_the_close_reports_and_never_trades():
    import master_live
    held = [{"portfolio_id": "SimIncubator2", "symbol": "MES",
             "direction": "LONG", "quantity": 2}]
    text = master_live.market_close_report(
        _Dispatcher(held), _State(), mc.WEEKEND, "weekend — closed Friday "
                                                 "17:00 ET")
    assert "SimIncubator2/MES LONG x2" in text
    assert "NOTHING IS BEING FLATTENED" in text
    assert "49 hours" in text, "a weekend close must say what it is carrying"


def test_a_maintenance_close_is_not_dressed_as_a_weekend():
    """
    Carrying inventory through a 60-minute halt is ordinary. Through 49 hours
    it is a decision. A close report that shouted equally at both would train
    an operator to skip the line that matters.
    """
    import master_live
    held = [{"portfolio_id": "SimIncubator2", "symbol": "MES",
             "direction": "LONG", "quantity": 2}]
    text = master_live.market_close_report(
        _Dispatcher(held), _State(), mc.MAINTENANCE,
        "daily maintenance break 17:00-18:00 ET")
    assert "NOTHING IS BEING FLATTENED" in text
    assert "49 hours" not in text


def test_a_flat_book_and_an_unverified_claim_are_reported_apart():
    """
    A CLAIM IS NOT A POSITION. `EngineState` records what this process SENT;
    the book records what it believes it HOLDS. They can disagree, and a
    disagreement at the close is the one worth looking at.
    """
    import master_live
    text = master_live.market_close_report(
        _Dispatcher(), _State([{"key": "k", "action": "BUY", "quantity": 1}]),
        mc.WEEKEND, "weekend")
    assert "FLAT" in text
    assert "1 UNVERIFIED claim" in text


def test_releasing_caches_never_raises_and_reports_a_measurement():
    """
    It runs in a shutdown path, ahead of nothing but the exit - but the report
    it prints is the only running measurement of the growth this gate bounds,
    so it has to produce one even when a cache it names has moved.
    """
    import master_live
    line = master_live.release_caches()
    assert "RSS" in line and "gc collected" in line
    assert master_live.rss_mb() > 0.0
