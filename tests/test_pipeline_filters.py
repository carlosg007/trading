"""
tests/test_pipeline_filters.py - the entry filters and the five-stage pipeline.

Location: ~/src/trading/tests/test_pipeline_filters.py

Covers what was added on 2026-08-17:

    backtest/event_calendar.py   the news and day-of-week ENTRY filters
    backtest/report.py           day_of_week_breakdown / losing_weekdays
    backtest/engine.py           the config fields and the run_backtest hook
    backtest/pipeline.py         the stage-to-stage handoff contract
    backtest/{baseline,scan,audit_gates,verify_full,promote}.py  the stages

The checks that matter most, and why they are here rather than assumed:

  * **The fill bar, not the signal bar.** The engine fills at the next bar's
    open, so a mask covering only the bars inside a news window still lets one
    entry per event fill inside it. That leak is a single trade per event -
    invisible in any aggregate - so it is pinned directly.
  * **Exits are never suppressed.** A filter that blocked exits would hold a
    position through the release it was built to avoid. Checked on a real
    engine run, not on the mask.
  * **The session date, not the UTC date.** Sunday 18:00 ET is Monday's
    session. Checked in both EST and EDT, and on daily bars as well as
    intraday, because the same one-line rule has to be right for all four.
  * **An empty calendar raises.** A news filter that blocks nothing and a
    calendar covering none of the span are identical downstream.

Runs without the lake and without a network. `mdlib.lake.iter_bars` is stubbed
where an engine run is needed; every other fixture is synthetic.

    python tests/test_pipeline_filters.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import backtest.engine as engine                                  # noqa: E402
from backtest.event_calendar import (CalendarError, EVENT_KINDS,   # noqa: E402
                                     MacroEvent, PUBLISHED, RULE,
                                     _widen_to_fill_bar,
                                     apply_entry_filters,
                                     entry_block_mask, filter_config_kwargs,
                                     is_news_blocked, load_event_calendar,
                                     parse_exclude_days, provenance_of,
                                     read_calendar_file, schedule_rule_events,
                                     session_date, session_weekday,
                                     weekday_blocked)
from backtest.engine import BacktestConfig, run_backtest           # noqa: E402
from backtest.pipeline import (BEST_PARAMS_FILE, pipeline_dir,     # noqa: E402
                               read_stage, stage_banner, write_stage)
from backtest.report import (day_of_week_breakdown,                # noqa: E402
                             format_day_of_week, format_dual_scorecard,
                             losing_weekdays)

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
    return bool(ok)


def raises(fn, exc=Exception) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:
        return True, f"{type(e).__name__}: {e}"
    except Exception as e:                                        # noqa: BLE001
        return False, f"raised the wrong type: {type(e).__name__}: {e}"
    return False, "did not raise"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def bars_15m(start="2024-02-01", days=12, symbol="ES",
             open_hour=14, open_minute=30, per_day=26) -> pd.DataFrame:
    """
    15m bars over one session a day, one symbol, UTC.

    `open_hour` defaults to the RTH open. The news checks pass 12 instead: an
    08:30 ET release is 13:30 UTC in winter, which is BEFORE the RTH open, so a
    fixture that starts at 14:30 has no bar for the window to land on and a
    working news filter would suppress nothing on it.
    """
    rows, px = [], 4800.0
    rng = np.random.default_rng(11)
    for d in pd.bdate_range(start, periods=days, tz="UTC"):
        for k in range(per_day):
            ts = (d + pd.Timedelta(hours=open_hour, minutes=open_minute)
                  + pd.Timedelta(minutes=15 * k))
            px *= 1 + rng.normal(0, 0.0006)
            rows.append((ts, symbol, px, px * 1.001, px * 0.999, px, 900))
    return pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low",
                                       "close", "volume"])


def alternating_signals(bars: pd.DataFrame):
    """Enter on every 4th bar, exit two bars later. Deterministic, dense."""
    n = len(bars)
    e = pd.Series(np.arange(n) % 4 == 0, index=bars.index)
    x = pd.Series(np.arange(n) % 4 == 2, index=bars.index)
    return e, x


def stub_iter_bars(frames: dict[str, pd.DataFrame]):
    """Replace `mdlib.lake.iter_bars` for one engine run."""
    import mdlib.lake as lake

    def _iter(symbols, tf, start=None, end=None, **kw):
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        for s in syms:
            yield s, frames[s].reset_index(drop=True)

    original = lake.iter_bars
    lake.iter_bars = _iter
    return original


def restore_iter_bars(original) -> None:
    import mdlib.lake as lake
    lake.iter_bars = original


# --------------------------------------------------------------------------
# 1. Session dates - the CME convention, not the UTC one
# --------------------------------------------------------------------------
def test_session_dates() -> None:
    print("\n1. Session dates: Sunday 18:00 ET opens Monday")

    # Sunday 2024-02-04 23:00 UTC = 18:00 EST, the week's open.
    idx = pd.DatetimeIndex(["2024-02-04T22:59:00Z", "2024-02-04T23:00:00Z",
                            "2024-02-05T14:30:00Z"])
    got = [str(d.date()) for d in session_date(idx)]
    check("Sunday 17:59 ET is still Sunday's session",
          got[0] == "2024-02-04", got[0])
    check("Sunday 18:00 ET rolls into Monday's session",
          got[1] == "2024-02-05", got[1])
    check("Monday morning is Monday", got[2] == "2024-02-05", got[2])

    # Daily bars are stamped at UTC midnight; in ET that is the PREVIOUS
    # evening past 18:00, so the same one-line rule has to put them back.
    est = pd.DatetimeIndex(["2024-01-08T00:00:00Z"])      # a Monday, EST
    edt = pd.DatetimeIndex(["2024-06-03T00:00:00Z"])      # a Monday, EDT
    check("a 1d bar stamped UTC midnight lands on its own weekday in EST",
          int(session_weekday(est)[0]) == 0, str(session_date(est)[0].date()))
    check("...and in EDT, where the UTC offset is an hour different",
          int(session_weekday(edt)[0]) == 0, str(session_date(edt)[0].date()))

    wd = session_weekday(pd.DatetimeIndex(
        ["2024-02-05T14:30:00Z", "2024-02-09T14:30:00Z"]))
    check("weekday integers are Monday=0 .. Friday=4",
          list(wd) == [0, 4], str(list(wd)))


# --------------------------------------------------------------------------
# 2. The rule-generated calendar
# --------------------------------------------------------------------------
def test_rule_calendar() -> None:
    print("\n2. The rule-generated calendar, and what it admits about itself")

    ev = schedule_rule_events("2024-01-01", "2024-12-31")
    kinds = {k: [e for e in ev if e.kind == k] for k in EVENT_KINDS}
    check("every kind is generated", all(kinds.values()),
          {k: len(v) for k, v in kinds.items()})
    check("every rule-generated event is stamped RULE",
          provenance_of(ev) == RULE, provenance_of(ev))
    check("FOMC generates eight meetings a year",
          len([e for e in kinds["FOMC"]
               if e.ts_utc.year == 2024]) == 8,
          str(len([e for e in kinds["FOMC"] if e.ts_utc.year == 2024])))

    # NFP is the one rule that is the actual published convention.
    nfp = {e.ts_utc.tz_convert("America/New_York") for e in kinds["NFP"]
           if e.ts_utc.year == 2024}
    check("NFP is always a Friday", all(t.dayofweek == 4 for t in nfp))
    check("NFP is always in the first seven days of a month",
          all(t.day <= 7 for t in nfp))
    check("NFP is always 08:30 ET, in both EST and EDT",
          all((t.hour, t.minute) == (8, 30) for t in nfp))

    # The DST check that a fixed UTC offset would fail.
    winter = [e for e in kinds["NFP"] if e.ts_utc.month == 1][0]
    summer = [e for e in kinds["NFP"] if e.ts_utc.month == 7][0]
    check("08:30 ET is 13:30 UTC in winter", winter.ts_utc.hour == 13,
          str(winter.ts_utc))
    check("and 12:30 UTC in summer - not a fixed offset",
          summer.ts_utc.hour == 12, str(summer.ts_utc))

    ok, msg = raises(lambda: schedule_rule_events("2024-01-01", "2024-06-01",
                                                  kinds=["GDP"]), CalendarError)
    check("an unknown event kind raises rather than being ignored", ok, msg)


def test_published_calendar(tmp: Path) -> None:
    print("\n3. A published calendar file outranks the rules")

    csv = tmp / "events.csv"
    csv.write_text("event,ts_utc,source\n"
                   "FOMC,2024-01-31T19:00:00Z,federalreserve.gov\n"
                   "CPI,2024-01-11 08:30:00,bls.gov\n", encoding="utf-8")
    ev = read_calendar_file(csv)
    check("both rows are read", len(ev) == 2, str(len(ev)))
    check("and stamped PUBLISHED, never RULE",
          provenance_of(ev) == PUBLISHED, provenance_of(ev))
    # The naive row is ET local, so 08:30 EST is 13:30 UTC.
    cpi = [e for e in ev if e.kind == "CPI"][0]
    check("a zone-less timestamp is read as ET local, not as UTC",
          cpi.ts_utc == pd.Timestamp("2024-01-11T13:30:00Z"), str(cpi.ts_utc))
    fomc = [e for e in ev if e.kind == "FOMC"][0]
    check("an explicit Z is taken at face value",
          fomc.ts_utc == pd.Timestamp("2024-01-31T19:00:00Z"), str(fomc.ts_utc))

    loaded = load_event_calendar("2024-01-01", "2024-02-01", path=csv)
    check("load_event_calendar prefers the file over the rules",
          provenance_of(loaded) == PUBLISHED, provenance_of(loaded))

    bad = tmp / "bad.csv"
    bad.write_text("event,ts_utc\nGDP,2024-01-11T13:30:00Z\n", encoding="utf-8")
    ok, msg = raises(lambda: read_calendar_file(bad), CalendarError)
    check("an unknown event in the file raises rather than being skipped",
          ok, msg)

    ok, msg = raises(lambda: load_event_calendar(
        "2024-01-01", "2024-02-01", path=tmp / "absent.csv", allow_rule=False),
        CalendarError)
    check("allow_rule=False refuses to fall back to approximate dates",
          ok, msg)


# --------------------------------------------------------------------------
# 4. The news mask
# --------------------------------------------------------------------------
def test_news_mask() -> None:
    print("\n4. The news window: boundaries, merging, and the empty case")

    ev = [MacroEvent("CPI", pd.Timestamp("2024-02-13T13:30:00Z"), PUBLISHED)]
    idx = pd.date_range("2024-02-13T12:00:00Z", "2024-02-13T15:00:00Z",
                        freq="15min")
    m = is_news_blocked(idx, 30, events=ev)
    blocked = [str(t.time()) for t, b in zip(idx, m) if b]
    check("a 30-minute window blocks exactly [13:00, 14:00]",
          blocked == ["13:00:00", "13:15:00", "13:30:00", "13:45:00",
                      "14:00:00"], str(blocked))

    # The bar one tick outside must be clear - an off-by-one here silently
    # widens or narrows every window in a 16-year run.
    edge = pd.DatetimeIndex(["2024-02-13T12:59:00Z", "2024-02-13T13:00:00Z",
                             "2024-02-13T14:00:00Z", "2024-02-13T14:01:00Z"])
    m2 = is_news_blocked(edge, 30, events=ev)
    check("the boundary is inclusive and one minute outside it is not",
          list(m2) == [False, True, True, False], str(list(m2)))

    # Two events inside one window of each other must not leave a hole.
    pair = [MacroEvent("CPI", pd.Timestamp("2024-02-13T13:30:00Z"), PUBLISHED),
            MacroEvent("PPI", pd.Timestamp("2024-02-13T14:15:00Z"), PUBLISHED)]
    span = pd.date_range("2024-02-13T13:00:00Z", "2024-02-13T14:45:00Z",
                         freq="15min")
    m3 = is_news_blocked(span, 30, events=pair)
    check("overlapping windows merge with no gap between them",
          bool(m3.all()), str(list(m3)))

    # searchsorted vs the obvious O(n*m) form, on a case small enough to brute
    # force. The fast path is the only one that can run on 5.6M bars.
    many = [MacroEvent("NFP", pd.Timestamp(f"2024-02-{d:02d}T13:30:00Z"),
                       PUBLISHED) for d in (2, 9, 13, 14, 20)]
    grid = pd.date_range("2024-02-01T00:00:00Z", "2024-02-25T00:00:00Z",
                         freq="17min")
    fast = is_news_blocked(grid, 45, events=many)
    w = pd.Timedelta(minutes=45)
    slow = np.array([any(e.ts_utc - w <= t <= e.ts_utc + w for e in many)
                     for t in grid])
    check("the searchsorted path agrees with brute force everywhere",
          bool((fast == slow).all()),
          f"{int((fast != slow).sum())} disagreements")

    ok, msg = raises(lambda: is_news_blocked(
        pd.date_range("2024-02-13T13:00:00Z", periods=4, freq="15min"), 30,
        events=[MacroEvent("CPI", pd.Timestamp("2019-01-01T13:30:00Z"),
                           PUBLISHED)]), CalendarError)
    check("a calendar covering none of the span RAISES rather than "
          "returning all-clear", ok, msg[:70])

    quiet = is_news_blocked(
        pd.date_range("2024-02-13T13:00:00Z", periods=4, freq="15min"), 30,
        events=[MacroEvent("CPI", pd.Timestamp("2019-01-01T13:30:00Z"),
                           PUBLISHED)], strict=False)
    check("...and strict=False is the deliberate way to accept that",
          not quiet.any())


def test_fill_bar_widening() -> None:
    print("\n5. The fill bar, not the signal bar")

    raw = np.array([False, False, True, True, False, False])
    wide = _widen_to_fill_bar(raw)
    check("the bar BEFORE the window is blocked too - its fill lands inside",
          list(wide) == [False, True, True, True, False, False],
          str(list(wide)))
    check("the widening never runs off the end of the array",
          len(wide) == len(raw) and _widen_to_fill_bar(
              np.array([], dtype=bool)).size == 0)
    check("a bar mask with nothing set widens to nothing",
          not _widen_to_fill_bar(np.zeros(5, dtype=bool)).any())

    # End to end: the last bar before a window must lose its entry.
    ev = [MacroEvent("CPI", pd.Timestamp("2024-02-13T13:30:00Z"), PUBLISHED)]
    idx = pd.date_range("2024-02-13T12:30:00Z", "2024-02-13T14:30:00Z",
                        freq="15min")
    mask, info = entry_block_mask(idx, news_filter=True, events=ev)
    bar_only = is_news_blocked(idx, 30, events=ev)
    check("entry_block_mask blocks one more bar than the raw window",
          int(mask.sum()) == int(bar_only.sum()) + 1,
          f"{int(mask.sum())} vs {int(bar_only.sum())}")
    check("and reports the calendar's provenance with the count",
          info["news_provenance"] == PUBLISHED
          and info["news_events_in_span"] == 1, str(info["news_provenance"]))


# --------------------------------------------------------------------------
# 6. The weekday mask
# --------------------------------------------------------------------------
def test_weekday_mask() -> None:
    print("\n6. The day-of-week entry filter")

    idx = pd.date_range("2024-02-05T14:30:00Z", "2024-02-10T00:00:00Z",
                        freq="6h")
    m = weekday_blocked(idx, [0, 4])
    days = {int(d) for d, b in zip(session_weekday(idx), m) if b}
    check("only the named weekdays are blocked", days == {0, 4}, str(days))
    check("an empty exclude list blocks nothing",
          not weekday_blocked(idx, []).any())
    check("None blocks nothing", not weekday_blocked(idx, None).any())

    ok, msg = raises(lambda: weekday_blocked(idx, [7]), CalendarError)
    check("a weekday outside 0-6 raises rather than matching nothing", ok, msg)

    check("parse_exclude_days parses and de-duplicates",
          parse_exclude_days("4,0,4") == (0, 4),
          str(parse_exclude_days("4,0,4")))
    check("an empty string is None, not an empty tuple",
          parse_exclude_days("") is None and parse_exclude_days(None) is None)
    ok, msg = raises(lambda: parse_exclude_days("Mon"), CalendarError)
    check("a non-integer weekday raises", ok, msg)
    ok, msg = raises(lambda: parse_exclude_days("9"), CalendarError)
    check("an out-of-range weekday raises", ok, msg)


def test_apply_entry_filters() -> None:
    print("\n7. apply_entry_filters suppresses entries on BOTH sides")

    idx = pd.date_range("2024-02-05T14:30:00Z", periods=200, freq="1h")
    longs = pd.Series(True, index=range(len(idx)))
    shorts = pd.Series(True, index=range(len(idx)))
    e, se, info = apply_entry_filters(idx, longs, shorts, exclude_days=[4])

    dow = session_weekday(idx)
    check("long entries are removed on the excluded weekday",
          not bool(e.to_numpy()[dow == 4].any()))
    check("short entries are removed too - a filter is not long-only",
          not bool(se.to_numpy()[dow == 4].any()))
    check("entries on other weekdays survive", bool(e.to_numpy()[dow == 1].all()))
    check("both sides are counted in the report",
          info["long_entries_suppressed"] > 0
          and info["short_entries_suppressed"] == info["long_entries_suppressed"],
          f"{info['long_entries_suppressed']}/{info['short_entries_suppressed']}")
    check("the total is the sum of the two sides",
          info["entries_suppressed"] == info["long_entries_suppressed"]
          + info["short_entries_suppressed"])
    check("a Series in gives a Series out, index preserved",
          isinstance(e, pd.Series) and list(e.index) == list(longs.index))
    check("short_entries=None comes back None, not an empty mask",
          apply_entry_filters(idx, longs, None, exclude_days=[4])[1] is None)

    # The API cannot be handed exits at all - that is the point.
    import inspect
    sig = set(inspect.signature(apply_entry_filters).parameters)
    check("the signature accepts no exit mask, so exits cannot be suppressed",
          "exits" not in sig and "short_exits" not in sig, str(sorted(sig)))


# --------------------------------------------------------------------------
# 8. The engine hook
# --------------------------------------------------------------------------
def test_engine_hook() -> None:
    print("\n8. The engine applies the filters, and only to entries")

    bars = bars_15m(days=15)
    original = stub_iter_bars({"ES": bars})
    try:
        plain = run_backtest("ES", "15m", alternating_signals,
                             cfg=BacktestConfig(chunk_size=0))
        filtered = run_backtest(
            "ES", "15m", alternating_signals,
            cfg=BacktestConfig(chunk_size=0, exclude_days=(4,)))
    finally:
        restore_iter_bars(original)

    check("an unfiltered run records no entry_filters block at all",
          "entry_filters" not in plain.stats or not plain.stats["entry_filters"])
    check("a filtered run records what it did",
          bool(filtered.stats.get("entry_filters")),
          str(sorted(filtered.stats.get("entry_filters", {}))))
    check("the filter removed trades", len(filtered.trades) < len(plain.trades),
          f"{len(plain.trades)} → {len(filtered.trades)}")

    # No trade may be ENTERED on the excluded weekday. Trades may still EXIT
    # there - suppressing an exit would hold a position through the very
    # session the filter excludes.
    ent_dow = session_weekday(filtered.trades["entry_time"])
    check("no trade enters on the excluded weekday",
          not bool((ent_dow == 4).any()), str(sorted(set(ent_dow.tolist()))))

    exit_dow = session_weekday(filtered.trades["exit_time"])
    check("but trades are still allowed to EXIT on it - exits are never "
          "suppressed", bool((exit_dow == 4).any()),
          str(sorted(set(exit_dow.tolist()))))

    check("the recorded suppression count is non-zero and matches the config",
          filtered.stats["entry_filters"]["exclude_days"] == [4]
          and filtered.stats["entry_filters"]["entries_suppressed"] > 0,
          str(filtered.stats["entry_filters"]["entries_suppressed"]))

    # Filters off must change nothing about an existing backtest.
    original = stub_iter_bars({"ES": bars})
    try:
        again = run_backtest("ES", "15m", alternating_signals,
                             cfg=BacktestConfig(chunk_size=0))
    finally:
        restore_iter_bars(original)
    check("with both filters off the result is unchanged, trade for trade",
          len(again.trades) == len(plain.trades)
          and float(again.trades["pnl"].sum()) == float(plain.trades["pnl"].sum()))


def test_engine_news_hook() -> None:
    print("\n9. The engine's news filter, end to end")

    # 12:00 → 20:00 UTC, so the 13:30 UTC (08:30 ET) releases fall inside the
    # session rather than before its first bar.
    bars = bars_15m(start="2024-02-01", days=20, open_hour=12, per_day=32)
    original = stub_iter_bars({"ES": bars})
    try:
        plain = run_backtest("ES", "15m", alternating_signals,
                             cfg=BacktestConfig(chunk_size=0))
        news = run_backtest("ES", "15m", alternating_signals,
                            cfg=BacktestConfig(chunk_size=0, news_filter=True,
                                               news_window_minutes=45))
    finally:
        restore_iter_bars(original)

    info = news.stats.get("entry_filters", {})
    check("the run records the calendar's provenance",
          info.get("news_provenance") in (RULE, PUBLISHED),
          str(info.get("news_provenance")))
    check("the rule-generated fallback is labelled RULE, not passed off as "
          "published", info.get("news_provenance") == RULE,
          str(info.get("news_provenance")))
    check("the window is recorded with the run",
          info.get("news_window_minutes") == 45.0)
    check("entries were suppressed", info.get("entries_suppressed", 0) > 0,
          str(info.get("entries_suppressed")))
    check("and the trade count fell", len(news.trades) < len(plain.trades),
          f"{len(plain.trades)} → {len(news.trades)}")

    # No entry may fill inside a blocked window. This is the check the fill-bar
    # widening exists for.
    events = load_event_calendar(bars["ts"].iloc[0], bars["ts"].iloc[-1])
    blocked = is_news_blocked(bars["ts"], 45, events=events)
    blocked_times = set(pd.DatetimeIndex(bars["ts"])[blocked])
    hits = [t for t in pd.DatetimeIndex(news.trades["entry_time"])
            if t in blocked_times]
    check("no trade ENTERS inside a blocked window", not hits,
          f"{len(hits)} leaked")


# --------------------------------------------------------------------------
# 10. Day-of-week attribution
# --------------------------------------------------------------------------
def test_dow_breakdown() -> None:
    print("\n10. Day-of-week attribution, against hand-computed answers")

    # Three Monday trades (+100, -40, +10) and two Wednesday trades (-50, -50).
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime([
            "2024-02-05T15:00:00Z", "2024-02-12T15:00:00Z",
            "2024-02-19T15:00:00Z", "2024-02-07T15:00:00Z",
            "2024-02-14T15:00:00Z"], utc=True),
        "exit_time": pd.to_datetime([
            "2024-02-05T16:00:00Z", "2024-02-12T16:00:00Z",
            "2024-02-19T16:00:00Z", "2024-02-07T16:00:00Z",
            "2024-02-14T16:00:00Z"], utc=True),
        "pnl": [100.0, -40.0, 10.0, -50.0, -50.0],
    })
    b = day_of_week_breakdown(trades)
    mon = b[b["weekday"] == 0].iloc[0]
    wed = b[b["weekday"] == 2].iloc[0]

    check("Monday's trade count is 3", int(mon["trades"]) == 3)
    check("Monday's net P&L is +70", float(mon["net_pnl"]) == 70.0,
          str(mon["net_pnl"]))
    check("Monday's win rate is 2/3", abs(float(mon["win_rate"]) - 2 / 3) < 1e-9,
          str(mon["win_rate"]))
    check("Monday's profit factor is 110/40 = 2.75",
          abs(float(mon["profit_factor"]) - 2.75) < 1e-9,
          str(mon["profit_factor"]))
    check("Wednesday's net P&L is -100", float(wed["net_pnl"]) == -100.0)
    check("Wednesday's win rate is 0", float(wed["win_rate"]) == 0.0)
    check("a day with no wins has profit factor 0.00 - a real number",
          float(wed["profit_factor"]) == 0.0, str(wed["profit_factor"]))

    # The other end: no LOSSES makes the ratio undefined. NaN, never inf - an
    # infinity in a table read by eye is a division by zero wearing a disguise.
    all_wins = day_of_week_breakdown(pd.DataFrame({
        "entry_time": pd.to_datetime(["2024-02-06T15:00:00Z"], utc=True),
        "exit_time": pd.to_datetime(["2024-02-06T16:00:00Z"], utc=True),
        "pnl": [40.0]}))
    tue_row = all_wins[all_wins["weekday"] == 1].iloc[0]
    check("a day with no losses has profit factor NaN, never inf",
          pd.isna(tue_row["profit_factor"]), str(tue_row["profit_factor"]))

    tue = b[b["weekday"] == 1].iloc[0]
    check("a weekday with no trades is present with zeros, not absent",
          int(tue["trades"]) == 0 and float(tue["net_pnl"]) == 0.0)
    check("Mon-Fri are all present", len(b) == 5, str(len(b)))
    check("weekend rows are absent when nothing traded then",
          set(b["weekday"]) == {0, 1, 2, 3, 4})

    # Share of P&L is signed by the DAY, against the absolute total, so a
    # losing day in a losing run still reads negative.
    check("the losing day's share of P&L is negative",
          float(wed["pct_of_net_pnl"]) < 0, str(wed["pct_of_net_pnl"]))
    check("and the winning day's is positive",
          float(mon["pct_of_net_pnl"]) > 0, str(mon["pct_of_net_pnl"]))

    check("empty trades give an empty frame, not a crash",
          day_of_week_breakdown(pd.DataFrame()).empty)
    check("a frame with no pnl column gives an empty frame",
          day_of_week_breakdown(pd.DataFrame({"entry_time": [1]})).empty)
    check("format_day_of_week renders every present row",
          all(d in format_day_of_week(b) for d in ("Mon", "Wed", "Fri")))

    check("losing_weekdays honours the trade floor - 2 trades is not evidence",
          losing_weekdays(b, min_trades=20) == [], str(losing_weekdays(b, 20)))
    check("...and names the day once the floor is met",
          losing_weekdays(b, min_trades=2) == [2],
          str(losing_weekdays(b, 2)))

    # Attribution is on the ENTRY, and on the session date. A Sunday-evening
    # entry belongs to Monday.
    overnight = pd.DataFrame({
        "entry_time": pd.to_datetime(["2024-02-04T23:30:00Z"], utc=True),
        "exit_time": pd.to_datetime(["2024-02-06T15:00:00Z"], utc=True),
        "pnl": [25.0]})
    ob = day_of_week_breakdown(overnight)
    check("a Sunday 18:30 ET entry is attributed to MONDAY's session",
          int(ob[ob["trades"] > 0].iloc[0]["weekday"]) == 0,
          str(ob[ob["trades"] > 0].iloc[0]["day"]))


def test_scorecard_gate_suppression() -> None:
    print("\n11. Stage 1's scorecard drops the gate table")

    m = {"sharpe": 1.2, "profit_factor": 1.4, "trade_count": 300,
         "max_drawdown_pct": -8.0, "win_rate": 0.55, "total_pnl": 1000.0}
    with_gates = format_dual_scorecard(m, None, show_gates=True)
    without = format_dual_scorecard(m, None, show_gates=False)
    check("the default still prints the gates",
          "ACCEPTANCE GATES" in with_gates)
    check("show_gates=False removes the gate table entirely",
          "ACCEPTANCE GATES" not in without and "NOT EVALUATED" not in without)
    check("but keeps the metrics and the verdict",
          "Sharpe" in without and "VERDICT" in without)


# --------------------------------------------------------------------------
# 12. The stage handoff contract
# --------------------------------------------------------------------------
def test_pipeline_handoff(tmp: Path) -> None:
    print("\n12. The stage-to-stage handoff refuses the wrong file")

    d = pipeline_dir("demo", tmp / "pipe", create=True)
    p = write_stage(d / "surviving_assets.json", 1, "demo",
                    {"surviving": ["NQ", "ES"]})
    check("write_stage creates the file", p.exists())
    blob = read_stage(p, 1, "demo")
    check("the payload round-trips", blob["surviving"] == ["NQ", "ES"])
    check("provenance is attached: stage, strategy and a UTC stamp",
          blob["stage"] == 1 and blob["strategy"] == "demo"
          and blob["generated_utc"].endswith("+00:00"))
    check("no .tmp file is left behind",
          not list(d.glob("*.tmp")), str(list(d.glob("*.tmp"))))

    ok, msg = raises(lambda: read_stage(p, 2, "demo"), ValueError)
    check("reading it as another stage's output raises", ok, msg)
    ok, msg = raises(lambda: read_stage(p, 1, "other_strategy"), ValueError)
    check("reading another strategy's handoff raises", ok, msg)
    ok, msg = raises(lambda: read_stage(d / "absent.json", 2),
                     FileNotFoundError)
    check("a missing file names the stage that should have written it",
          ok and "stage 2" in msg, msg)

    check("stage_banner names the stage it belongs to",
          "STAGE 3/5" in stage_banner(3, "demo"))


# --------------------------------------------------------------------------
# 13. The stages themselves
# --------------------------------------------------------------------------
STAGES = [
    ("backtest/baseline.py", "Stage 1/5"),
    ("backtest/scan.py", "Stage 2/5"),
    ("backtest/audit_gates.py", "Stage 3/5"),
    ("backtest/verify_full.py", "Stage 4/5"),
    ("backtest/promote.py", "Promote"),
]


def test_stage_clis() -> None:
    print("\n13. Every stage runs as a script and documents itself")

    for rel, marker in STAGES:
        r = subprocess.run([sys.executable, str(REPO / rel), "--help"],
                           capture_output=True, text=True, cwd=REPO, timeout=180)
        check(f"{rel} --help exits 0", r.returncode == 0,
              (r.stderr or "")[-200:])
        if r.returncode == 0:
            check(f"{rel} says which stage it is",
                  marker.lower() in r.stdout.lower(), r.stdout[:80])

    # The filter flags exist on every stage AND on the batch runner, spelled
    # the same way - they come from one `add_filter_args`.
    for rel in [s[0] for s in STAGES[:4]] + ["backtest/run.py"]:
        r = subprocess.run([sys.executable, str(REPO / rel), "--help"],
                           capture_output=True, text=True, cwd=REPO, timeout=180)
        check(f"{rel} exposes --news-filter and --exclude-days",
              "--news-filter" in r.stdout and "--exclude-days" in r.stdout)


def _profile(**quadrants) -> dict:
    """A RegimeProfiler-shaped profile from `regime="pf,trades"` pairs."""
    from backtest.profiler import REGIMES
    breakdown, total = {}, 0
    for key, (pf, n) in quadrants.items():
        regime = {"hvt": REGIMES[0], "hvr": REGIMES[1],
                  "lvt": REGIMES[2], "lvr": REGIMES[3]}[key]
        breakdown[regime] = {"trade_count": n, "profit_factor": pf,
                             "win_rate": 55.0, "net_pnl": 100.0 * n}
        total += n
    return {"regime_breakdown": breakdown, "trades_profiled": total,
            "trades_unplaced": 0}


def test_baseline_screen() -> None:
    print("\n14. Stage 1's REGIME firewall")
    from backtest.baseline import (best_quadrant, kill_switch_regimes, screen)
    from backtest.profiler import REGIMES

    keep, why, best = screen({"A": _profile(hvt=(1.28, 80)), "B": None})
    check("a quadrant at 1.28 over 80 trades survives", keep, why)
    check("...and the winning quadrant is named, with its own numbers",
          best["regime"] == REGIMES[0] and best["profit_factor"] == 1.28
          and best["trade_count"] == 80 and best["version"] == "A", str(best))

    # The bar was lowered 1.15 -> 1.00 on 2026-08-20 by operator instruction.
    # 1.14 now SURVIVES; what must still be dropped is a quadrant that did not
    # break even.
    drop, why, best = screen({"A": _profile(hvt=(0.94, 400)), "B": None})
    check("0.94 is below the 1.00 bar and is dropped", not drop and best is None,
          why)
    check("...and the reason names the best quadrant and how far short it fell",
          "0.94" in why and REGIMES[0] in why, why)
    edge, why, _ = screen({"A": _profile(hvt=(1.00, 50)), "B": None})
    check("exactly 1.00 over exactly 50 trades is on the boundary and survives",
          edge, why)
    loosened, why, _ = screen({"A": _profile(hvt=(1.14, 400)), "B": None})
    check("1.14 now survives, where the 1.15 bar dropped it", loosened, why)

    thin, why, _ = screen({"A": _profile(hvt=(2.40, 49)), "B": None})
    check("a 2.40 PF over 49 trades in that quadrant is NOT an environment",
          not thin and "49" in why and "sample floor of 50" in why, why)
    # The floor scales past the flat 50 once the run is large enough.
    corner, why, _ = screen({"A": _profile(hvt=(2.40, 60), lvr=(0.9, 4940)),
                             "B": None})
    check("...nor is 60 trades out of 5,000 - the 10% share binds where the "
          "flat 50 would not", not corner and "500" in why, why)

    # Both bars bind on the SAME quadrant - the whole point of the screen.
    split, why, _ = screen({"A": _profile(hvt=(1.90, 11), lvr=(0.90, 400)),
                            "B": None})
    check("the best PF and the largest trade count in DIFFERENT quadrants "
          "do not combine to a pass", not split, why)

    # Blended profit factor is irrelevant now: this configuration loses money
    # overall and still survives on one environment. That is the intended
    # loosening. The bar sat at 1.15 for that reason until 2026-08-20.
    mixed, why, best = screen({"A": _profile(hvt=(1.60, 90), hvr=(0.55, 300),
                                             lvr=(0.70, 250)), "B": None})
    check("a configuration that loses overall survives on ONE good quadrant",
          mixed and best["regime"] == REGIMES[0], why)

    none, why, _ = screen({"A": _profile(), "B": None})
    check("no trades in any regime is dropped with its OWN reason",
          not none and "never fired" in why, why)
    absent, why, _ = screen({"A": None, "B": None})
    check("no profile at all is dropped and says so",
          not absent and "no regime profile" in why, why)

    # Version B can carry a configuration Version A failed.
    keep, why, best = screen({"A": _profile(hvt=(0.87, 400)),
                              "B": _profile(lvt=(1.31, 60))})
    check("Version B can carry a configuration Version A failed",
          keep and best["version"] == "B" and best["regime"] == REGIMES[2], why)
    check("...and with Version B absent, Version A alone decides",
          screen({"A": _profile(hvt=(0.87, 400)), "B": None})[0] is False)

    check("the bars are adjustable per run",
          screen({"A": _profile(hvt=(1.20, 80)), "B": None},
                 min_profit_factor=1.50)[0] is False
          and screen({"A": _profile(hvt=(1.20, 80)), "B": None},
                     min_trades=200)[0] is False)

    # Equal factors, unequal alpha: the engine wins, not the corner.
    tied = best_quadrant(_profile(hvt=(1.40, 90), lvr=(1.40, 900)))
    check("equal profit factors resolve on alpha contribution, and the "
          "larger book wins",
          tied["regime"] == REGIMES[3] and tied["trade_count"] == 900,
          str(tied))
    check("a profile with nothing clearing returns None, not a best-effort pick",
          best_quadrant(_profile(hvt=(0.90, 900))) is None)

    # The kill switch is derived, and never invented.
    ks = kill_switch_regimes(REGIMES[0])
    check("the kill switch is the other THREE quadrants, in regime order",
          ks == list(REGIMES[1:]), str(ks))
    check("no optimal regime yields an EMPTY kill switch, never all four - "
          "'trade nowhere' has to come from a decision",
          kill_switch_regimes(None) == []
          and kill_switch_regimes("Some Other Regime") == [])


def test_baseline_report(tmp: Path) -> None:
    print("\n14b. Stage 1's markdown report and its pair handoff")
    from backtest.baseline import (_evaluated_line, _human_bars, _row,
                                   build_markdown_report, stage2_command,
                                   write_markdown_report)

    trades = pd.DataFrame({
        "entry_time": pd.to_datetime(
            ["2020-01-06 15:00", "2020-01-07 15:00", "2020-01-08 15:00",
             "2020-01-10 15:00"], utc=True),
        "pnl": [120.0, -80.0, 45.0, -200.0]})
    dow = day_of_week_breakdown(trades)
    ma = {"ok": True, "trade_count": 4, "profit_factor": 0.59, "sharpe": -0.31,
          "sortino": -0.40, "calmar": -0.20, "win_rate": 0.5,
          "max_drawdown_pct": -8.2, "total_return_pct": -0.12,
          "annualized_return_pct": -0.4, "total_pnl": -115.0,
          "gross_pnl": 40.0, "total_costs": 155.0, "n_days": 20,
          "trades": trades,
          "entry_filters": {"news_filter": True, "news_window_minutes": 30,
                            "news_kinds": ["NFP"], "news_provenance": RULE,
                            "news_events_in_span": 214,
                            "entries_suppressed": 37,
                            "long_entries_before": 400,
                            "short_entries_before": 0}}
    mb = dict(ma, profit_factor=1.31, trade_count=90, sharpe=0.44,
              total_pnl=900.0)

    dropped = _row("NQ", "1m", ma, None, False, "best profit factor 0.87", dow,
                   3_200_000, 12.4)
    kept = _row("NQ", "5m", ma, mb, True, "Version B profit factor 1.31", dow,
                661_000, 8.1)

    check("bar counts read the way an operator says them",
          (_human_bars(3_200_000), _human_bars(661_000)) == ("3.2M", "661k"),
          f"{_human_bars(3_200_000)} / {_human_bars(661_000)}")

    line = _evaluated_line("[1/8]", dropped)
    check("a skipped Version B reports NOT RUN on the progress line, not 0.00",
          "NOT RUN" in line and "0.00" not in line and "[DROPPED]" in line,
          line)
    line = _evaluated_line("[2/8]", kept)
    check("both profit factors and the verdict are on one line",
          "0.59 PF" in line and "1.31 PF" in line and "[SURVIVES]" in line,
          line)

    md = build_markdown_report(
        "demo", {"Strategy": "demo"}, [dropped, kept],
        [{"symbol": "CL", "timeframe": "15m", "error": "ValueError: no bars"}])
    for want in ("Sharpe", "Sortino", "Profit factor", "Win rate",
                 "Max drawdown", "Net return", "Trades", "Friction costs",
                 "SURVIVES", "DROPPED", "NQ · 5m", "NQ · 1m"):
        check(f"the report carries {want!r}", want in md)
    check("...the day-of-week attribution", "| Mon |" in md and "| Fri |" in md)
    check("...the macro filter audit, events scanned and entries cut",
          "214 events" in md and "entries cut" in md)
    check("...and the configuration that errored, rather than dropping it",
          "ValueError: no bars" in md)
    check("Version B's column says NOT RUN where it did not run - never n/a",
          "NOT RUN" in md.split("### NQ · 5m")[0])
    check("a screen that evaluated nothing still renders a report",
          "Nothing was evaluated"
          in build_markdown_report("demo", {"Strategy": "demo"}, [], []))

    d = tmp / "stage1_md"
    p = write_markdown_report(d / "stage1_baseline_report.md", md)
    check("the report is written atomically, leaving no .tmp behind",
          p.exists() and not p.with_suffix(".md.tmp").exists())

    # The Stage 2 command targets the surviving PAIRS, not the whole grid.
    pairs = [{"symbol": "NQ", "tf": "5m"}, {"symbol": "NQ", "tf": "15m"},
             {"symbol": "GC", "tf": "15m"}]
    cmd = "\n".join(stage2_command("demo", pairs, "2013-01-01", "2022-12-31"))
    check("stage 2 is handed only the surviving symbols",
          "--symbols GC,NQ" in cmd, cmd)
    check("...only the surviving timeframes, ascending",
          "--tf 5m,15m" in cmd, cmd)
    check("...and the timeframes that failed are omitted entirely",
          "1m," not in cmd and "30m" not in cmd, cmd)
    check("a ragged survivor set is flagged as a SUPERSET of what survived",
          "SUPERSET" in cmd)
    check("a complete grid needs no such warning",
          "SUPERSET" not in "\n".join(stage2_command(
              "demo", pairs + [{"symbol": "GC", "tf": "5m"}], None, None)))
    check("nothing surviving prints no command to run",
          "scan.py" not in "\n".join(stage2_command("demo", [], None, None)))


def test_window_guard() -> None:
    print("\n15. Stage 3 refuses a holdout it has already seen")
    from backtest.audit_gates import WindowOverlapError, check_windows

    check_windows("2013-01-01", "2022-12-31", "2023-01-01", "2026-01-01")
    check("a clean split is accepted", True)

    ok, msg = raises(lambda: check_windows("2013-01-01", "2023-06-30",
                                           "2023-01-01", "2026-01-01"),
                     WindowOverlapError)
    check("an in-sample window running into the holdout is REFUSED", ok, msg[:70])
    ok, _ = raises(lambda: check_windows("2013-01-01", "2023-01-01",
                                         "2023-01-01", "2026-01-01"),
                   WindowOverlapError)
    check("...including one that merely touches it", ok)
    ok, msg = raises(lambda: check_windows("2013-01-01", None,
                                           "2023-01-01", "2026-01-01"),
                     WindowOverlapError)
    check("an open-ended in-sample window is refused - it eats the holdout",
          ok, msg[:60])
    ok, _ = raises(lambda: check_windows("2013-01-01", "2022-12-31",
                                         "2026-01-01", "2023-01-01"),
                   WindowOverlapError)
    check("an inverted holdout window is refused", ok)


def test_stage2_to_stage3_handoff(tmp: Path) -> None:
    print("\n16. Stage 2's winner reaches Stage 3 intact")
    from backtest.audit_gates import load_params
    from backtest.scan import write_best_params

    d = tmp / "s23"
    d.mkdir(parents=True, exist_ok=True)
    scan = {
        "symbol": "NQ", "combinations": 12, "evaluated": 9, "rejected": [{}],
        "selection": "GATE 1 PASS",
        "winner": {"params": {"ema_period": 30},
                   "metrics": {"sharpe": 1.4, "trade_count": 300,
                               "trades": pd.DataFrame()},
                   "gate1": {"status": "PASS"}, "sharpe": 1.4},
    }
    written = write_best_params(scan, "demo", "NQ", "15m", "2013-01-01",
                                "2022-12-31", {"atr_mult": 2.0}, d)
    names = {w.name for w in written}
    check("a single-timeframe sweep writes the plain name stage 3 falls back "
          "to", BEST_PARAMS_FILE.format(symbol="NQ") in names, str(names))
    check("and the timeframe-suffixed name beside it",
          BEST_PARAMS_FILE.format(symbol="NQ_15m") in names, str(names))
    p = d / BEST_PARAMS_FILE.format(symbol="NQ")

    blob = json.loads(p.read_text())
    check("params is the FULL effective set, not only the swept axes",
          blob["params"] == {"atr_mult": 2.0, "ema_period": 30},
          str(blob["params"]))
    check("variants_tested travels with it",
          blob["variants_tested"] == 9, str(blob["variants_tested"]))
    check("the in-sample metrics are scalars only - no DataFrame leaked in",
          "trades" not in (blob["in_sample"] or {}),
          str(sorted((blob["in_sample"] or {}))))

    params, prov = load_params("demo", "NQ", d, {}, use_defaults=False)
    check("Stage 3 reads back exactly what Stage 2 wrote",
          params == {"atr_mult": 2.0, "ema_period": 30}, str(params))
    check("and records where the parameters came from",
          "stage 2 winner" in prov["params_source"], prov["params_source"])
    check("with the variant count, so the audit can state N",
          prov["variants_tested"] == 9)

    over, prov2 = load_params("demo", "NQ", d, {"ema_period": 50}, False)
    check("--param overrides the stage 2 winner",
          over["ema_period"] == 50 and "--param" in prov2["params_source"],
          prov2["params_source"])

    ok, msg = raises(lambda: load_params("demo", "CL", d, {}, False),
                     FileNotFoundError)
    check("a missing best_params file is an ERROR, not a silent fallback to "
          "the defaults", ok and "stage 2" in msg, msg[:80])
    dflt, prov3 = load_params("demo", "CL", d, {"x": 1}, use_defaults=True)
    check("--defaults is the deliberate way to certify the module's defaults",
          dflt == {"x": 1} and prov3["variants_tested"] is None)

    ok, msg = raises(lambda: load_params("other", "NQ", d, {}, False),
                     ValueError)
    check("another strategy's best_params is refused", ok, msg[:70])


def test_cost_drag() -> None:
    print("\n17. Stage 4's cost drag, against hand-computed answers")
    from backtest.verify_full import cost_drag, format_cost_drag

    trades = pd.DataFrame({
        "entry_time": pd.to_datetime(["2020-03-02", "2020-04-02",
                                      "2021-03-02"], utc=True),
        "exit_time": pd.to_datetime(["2020-03-03", "2020-04-03",
                                     "2021-03-03"], utc=True),
        "gross_pnl": [500.0, -200.0, 300.0],
        "costs": [20.0, 20.0, 20.0],
        "pnl": [480.0, -220.0, 280.0],
    })
    d = cost_drag(trades, None, None, BacktestConfig())
    check("gross P&L is the sum of gross", d["gross_pnl"] == 600.0)
    check("total costs is the sum of costs", d["total_costs"] == 60.0)
    check("net P&L is the sum of net", d["net_pnl"] == 540.0)
    check("cost per trade is 60/3", d["cost_per_trade"] == 20.0)
    check("cost share is 60/600 = 10%", abs(d["cost_share_pct"] - 10.0) < 1e-9,
          str(d["cost_share_pct"]))
    check("the per-year split has both years",
          [r["year"] for r in d["by_year"]] == [2020, 2021],
          str([r["year"] for r in d["by_year"]]))
    check("2020 gross is 500 - 200 = 300",
          d["by_year"][0]["gross_pnl"] == 300.0)
    check("the split is unavailable without a config or a spec",
          d["split_available"] is False)

    losing = trades.assign(gross_pnl=[-500.0, -200.0, -300.0])
    dl = cost_drag(losing, None, None, BacktestConfig())
    check("cost share is None - never 0% - when there is no gross profit",
          dl["cost_share_pct"] is None, str(dl["cost_share_pct"]))
    check("format_cost_drag prints n/a rather than a fabricated percentage",
          "n/a" in format_cost_drag(dl))
    check("an empty trade list gives zeros and no crash",
          cost_drag(pd.DataFrame(), None, None, BacktestConfig())["trades"] == 0)


def test_promote_certification(tmp: Path) -> None:
    print("\n18. Stage 5 verifies Stage 3's certification")
    from backtest.promote import load_gate_certification, promote

    # Any live module serves: this case promotes a FILE and asserts nothing
    # about what it computes. `sma_crossover.py` filled the role until it
    # was deleted.
    source = REPO / "strategies" / "experimental" / "ema_crossover_20260821.py"
    if not source.exists():
        check(f"{source.name} is present to promote", False, str(source))
        return

    def audit_file(name: str, status: str, stage: int = 3,
                   version: str = "A") -> Path:
        p = tmp / name
        write_stage(p, stage, "sma_crossover", {
            "symbol": "NQ",
            "in_sample": {"start": "2013-01-01", "end": "2022-12-31"},
            "holdout": {"start": "2023-01-01", "end": "2026-01-01"},
            "variants_tested": 27,
            "versions": {version: {"gate_audit": {
                "status": status, "passed": status == "PASS",
                "gates": {"gate1": {"status": status, "checks": []},
                          "gate2": {"status": status, "checks": []},
                          "gate3": {"status": status, "checks": []}}}}},
        })
        return p

    passing = audit_file("gate_audit_NQ.json", "PASS")
    audit, prov = load_gate_certification(passing, "A")
    check("a PASS certification is read", audit["status"] == "PASS")
    check("its provenance carries the file's SHA-256",
          len(prov["audit_sha256"]) == 64)
    check("and the holdout window it was certified over",
          prov["holdout"]["start"] == "2023-01-01")

    out = promote("cert_ok", "A", source, audit_path=passing, commit=False,
                  incubator=tmp / "inc")
    meta = out["meta"]
    check("promotion on a PASS certification succeeds",
          meta["gate_audit_status"] == "PASS")
    check("meta.json records which audit certified it",
          isinstance(meta["certification"], dict)
          and meta["certification"]["audit_symbol"] == "NQ")
    check("and that the gates were not overridden",
          meta["gates_overridden"] is False)

    failing = audit_file("gate_audit_ES.json", "FAIL")
    ok, msg = raises(lambda: promote("cert_bad", "A", source,
                                     audit_path=failing, commit=False,
                                     incubator=tmp / "inc"), SystemExit)
    check("a FAIL certification refuses promotion", ok, msg[:70])
    check("and the refusal names the file it read", "gate_audit_ES" in msg, msg[:90])

    forced = promote("cert_forced", "A", source, audit_path=failing,
                     force=True, commit=False, incubator=tmp / "inc")
    check("--force promotes anyway", forced["meta"]["gate_audit_status"] == "FAIL")
    check("and the override is on the record",
          forced["meta"]["gates_overridden"] is True)

    nev = audit_file("gate_audit_CL.json", "NOT EVALUATED")
    ok, _ = raises(lambda: promote("cert_ne", "A", source, audit_path=nev,
                                   commit=False, incubator=tmp / "inc"),
                   SystemExit)
    check("NOT EVALUATED is refused too - it is not a pass", ok)

    wrong_stage = audit_file("gate_audit_stage1.json", "PASS", stage=1)
    ok, msg = raises(lambda: load_gate_certification(wrong_stage, "A"),
                     ValueError)
    check("a stage 1 file cannot be used as a certification", ok, msg[:70])

    ok, msg = raises(lambda: load_gate_certification(passing, "B"), ValueError)
    check("asking for a version the audit does not carry raises", ok, msg[:80])
    ok, _ = raises(lambda: load_gate_certification(tmp / "nope.json", "A"),
                   FileNotFoundError)
    check("a missing audit file raises", ok)

    # The old workflow keeps working: no audit file, promotion allowed, and
    # meta.json says plainly that nothing certified it.
    plain = promote("no_cert", "A", source, commit=False, incubator=tmp / "inc")
    check("promoting with no certification still works (the bt-run workflow)",
          plain["meta"]["certification"] == "NOT CERTIFIED")
    ok, msg = raises(lambda: promote("needs_cert", "A", source, commit=False,
                                     require_certification=True,
                                     incubator=tmp / "inc"), SystemExit)
    check("--require-certification turns that into a refusal", ok, msg[:70])


# --------------------------------------------------------------------------
# 20. The confluence strategy and multi-timeframe scanning
# --------------------------------------------------------------------------
# The seven cases that stood here tested `ema_trend_filter`'s OWN
# specification — its session VWAP, its RSI gate, its four confluence toggles,
# their warm-up and dead-axis behaviour, their causality, and its declared
# 1,296-cell grid. That module was deleted, so those cases have no subject:
# they could not be re-pointed at a live strategy without rewriting every
# assertion, because what they asserted was that module's own numbers.
#
# The coverage is NOT lost. `t3_braid_scalp_20260823` is the live module of the
# same shape — three confluence layers plus a news filter — and
# `tests/test_t3_braid_scalp_20260823.py` holds it to the same standard in 39
# cases: each layer subtracting on its own, the toggles mirrored across the two
# sides, causality by truncation AND by perturbation, the declared grid, and
# the warm-up a toggled-off layer must not pay for.
# `tests/test_double_rsi_macd_scalp_20260823.py` does the same for the other.
#
# `globex_bars` went with them; it had no other caller.


def test_multi_timeframe_cli() -> None:
    print("\n27. Multi-timeframe scanning")
    from backtest.run import parse_timeframes

    check("a comma-separated list parses in the order given",
          parse_timeframes("1m,5m,15m,30m", None) == ["1m", "5m", "15m", "30m"])
    check("one timeframe is still a list, so callers have one code path",
          parse_timeframes("15m", None) == ["15m"])
    check("no --tf falls back to the module's declared timeframe",
          parse_timeframes(None, "15m") == ["15m"])
    check("duplicates collapse, order preserved",
          parse_timeframes("30m, 5m ,30m", None) == ["30m", "5m"])

    ok, msg = raises(lambda: parse_timeframes("7m", None), ValueError)
    check("a timeframe the lake cannot serve is refused up front", ok, msg[:60])
    check("every derived timeframe the lake declares is accepted",
          parse_timeframes("1m,5m,15m,30m,1h,2h,4h,1d,1w", None) ==
          ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"])


def test_multi_timeframe_handoff(tmp: Path) -> None:
    print("\n28. Multi-timeframe handoff to stage 3")
    from backtest.audit_gates import discover_symbols, load_params
    from backtest.scan import write_best_params

    d = tmp / "mtf"
    d.mkdir(parents=True, exist_ok=True)

    def scan_for(tf, ema):
        return {"symbol": "NQ", "combinations": 864, "evaluated": 864,
                "rejected": [], "selection": "GATE 1 PASS",
                "winner": {"params": {"trend_period": ema},
                           "metrics": {"sharpe": 1.1},
                           "gate1": {"status": "PASS"}, "sharpe": 1.1}}

    # A multi-timeframe sweep: per-tf files, and NO unsuffixed file.
    for tf, ema in (("5m", 400), ("15m", 200)):
        written = write_best_params(scan_for(tf, ema), "demo", "NQ", tf,
                                    "2013-01-01", "2022-12-31", {}, d,
                                    timeframes=["5m", "15m"],
                                    variants_all_timeframes=1728)
        check(f"{tf}: one file written, suffixed with the timeframe",
              len(written) == 1 and written[0].name == "best_params_NQ_5m.json"
              if tf == "5m" else written[0].name == "best_params_NQ_15m.json",
              written[0].name)

    check("a multi-timeframe sweep writes NO unsuffixed best_params",
          not (d / "best_params_NQ.json").exists(),
          "stage 3 must be told which timeframe it is certifying")

    blob = json.loads((d / "best_params_NQ_15m.json").read_text())
    check("each file records the full cross-timeframe search size",
          blob["variants_tested"] == 864
          and blob["variants_tested_all_timeframes"] == 1728,
          f"{blob['variants_tested']} / {blob['variants_tested_all_timeframes']}")

    # Stage 3 must pick up the timeframe it is running on, not the other one.
    p5, prov5 = load_params("demo", "NQ", d, {}, False, tf="5m")
    p15, _ = load_params("demo", "NQ", d, {}, False, tf="15m")
    check("stage 3 at 5m reads the 5m winner",
          p5["trend_period"] == 400, str(p5))
    check("stage 3 at 15m reads the 15m winner",
          p15["trend_period"] == 200, str(p15))
    check("and names the file it used", "NQ_5m" in prov5["params_source"],
          prov5["params_source"])

    check("symbol discovery strips the timeframe suffix, not just the prefix",
          discover_symbols(d, "15m") == ["NQ"], str(discover_symbols(d, "15m")))
    check("...and finds nothing when asked for a timeframe nobody swept",
          discover_symbols(d, "1h") == [], str(discover_symbols(d, "1h")))

    # A single-timeframe sweep writes both, and the unsuffixed one is the
    # fallback for the pre-multi-timeframe handoff.
    solo = tmp / "solo"
    solo.mkdir(parents=True, exist_ok=True)
    written = write_best_params(scan_for("15m", 200), "demo", "NQ", "15m",
                                "2013-01-01", "2022-12-31", {}, solo,
                                timeframes=["15m"])
    check("a single-timeframe sweep writes both the suffixed and plain names",
          len(written) == 2
          and {w.name for w in written} == {"best_params_NQ_15m.json",
                                            "best_params_NQ.json"},
          str([w.name for w in written]))

    # The fallback must not certify one timeframe's winner on another's bars.
    ok, msg = raises(lambda: load_params("demo", "NQ", solo, {}, False,
                                         tf="5m"), ValueError)
    check("certifying 5m bars with a 15m winner is REFUSED", ok, msg[:70])


def test_filter_config_kwargs() -> None:
    print("\n19. The CLI flags map onto the config the engine reads")
    import argparse

    from backtest.event_calendar import add_filter_args

    p = add_filter_args(argparse.ArgumentParser())
    args = p.parse_args(["--news-filter", "--news-window", "45",
                         "--news-kinds", "cpi,nfp", "--exclude-days", "0,4"])
    kw = filter_config_kwargs(args)
    check("the flags parse into BacktestConfig fields",
          kw == {"news_filter": True, "news_window_minutes": 45.0,
                 "news_kinds": ("CPI", "NFP"), "exclude_days": (0, 4)},
          str(kw))
    check("every key is a real BacktestConfig field",
          set(kw) <= set(BacktestConfig.__dataclass_fields__),
          str(set(kw) - set(BacktestConfig.__dataclass_fields__)))
    check("BacktestConfig accepts them", BacktestConfig(**kw).news_filter)

    off = filter_config_kwargs(p.parse_args([]))
    check("with no flags the filters are off and exclude_days is None",
          off["news_filter"] is False and off["exclude_days"] is None)

    bad = p.parse_args(["--news-kinds", "GDP"])
    ok, msg = raises(lambda: filter_config_kwargs(bad), CalendarError)
    check("an unknown event kind raises at parse time, not mid-run", ok, msg)


# --------------------------------------------------------------------------
# 29. The Drop Unprofitable Days contract - stage 1 decides, 2 and 3 inherit
# --------------------------------------------------------------------------
def dow_trades(per_day: dict[int, float], n_weeks: int = 12,
               start: str = "2020-01-06") -> pd.DataFrame:
    """
    A trade log with a KNOWN P&L per weekday. `per_day` maps Mon=0..Fri=4.

    Two trades a day, split around the target, so gross win and gross loss are
    both non-zero and the profit factor is a real number rather than NaN. Entry
    stamps are 15:00 UTC - 10:00 ET, inside the session whose date is the
    calendar date - so the session-date roll is not what this fixture tests;
    `test_session_dates` covers the roll itself.
    """
    rows = []
    for w in range(n_weeks):
        monday = pd.Timestamp(start, tz="UTC") + pd.Timedelta(weeks=w)
        for d, pnl in per_day.items():
            ts = monday + pd.Timedelta(days=d, hours=15)
            rows += [(ts, pnl + 10.0), (ts + pd.Timedelta(minutes=5), -10.0)]
    return pd.DataFrame(rows, columns=["entry_time", "pnl"])


def test_unprofitable_weekdays() -> None:
    print("\n29. unprofitable_weekdays - every session below PF 1.00")
    from backtest.report import exclude_days_basis, unprofitable_weekdays

    # Mon and Tue below 1.00, Wed/Thu/Fri above it.
    dow = day_of_week_breakdown(dow_trades({0: -100.0, 1: -20.0, 2: 40.0,
                                            3: 40.0, 4: 40.0}))
    bad = unprofitable_weekdays(dow, min_trades=10)
    check("EVERY weekday under the bar is returned, not just the worst",
          [d["weekday"] for d in bad] == [0, 1],
          str([(d["day_name"], d["profit_factor"]) for d in bad]))
    check("...in weekday order, with the long name for the artifact",
          [d["day_name"] for d in bad] == ["Monday", "Tuesday"])
    check("...and the metrics that condemned each one",
          all({"trades", "net_pnl", "profit_factor"} <= set(d) for d in bad))
    check("the rule is stated in words for the record",
          "profit factor < 1.00" in exclude_days_basis(20)
          and "ENTRY session" in exclude_days_basis(20), exclude_days_basis(20))

    # 0, 1 and 5 days are the same code path and the same shape.
    check("ZERO days below the bar is an empty list, never None",
          unprofitable_weekdays(
              day_of_week_breakdown(dow_trades({0: 40.0, 1: 40.0, 2: 40.0,
                                                3: 40.0, 4: 40.0})),
              min_trades=10) == [])
    one = unprofitable_weekdays(
        day_of_week_breakdown(dow_trades({0: -100.0, 1: 40.0, 2: 40.0,
                                          3: 40.0, 4: 40.0})), min_trades=10)
    check("ONE day below the bar is a one-element list",
          [d["weekday"] for d in one] == [0])
    allbad = unprofitable_weekdays(
        day_of_week_breakdown(dow_trades({0: -50.0, 1: -50.0, 2: -50.0,
                                          3: -50.0, 4: -50.0})), min_trades=10)
    check("ALL FIVE below the bar excludes all five - no cap, no index error",
          [d["weekday"] for d in allbad] == [0, 1, 2, 3, 4])

    check("the trade floor is a floor, not a formality",
          unprofitable_weekdays(dow, min_trades=1000) == [])
    check("an empty breakdown is [], not a crash",
          unprofitable_weekdays(day_of_week_breakdown(None)) == []
          and unprofitable_weekdays(pd.DataFrame()) == [])

    # A day with no losing trades has an UNDEFINED profit factor. It is the
    # opposite of a day to exclude, and `NaN < 1.00` being False must not be
    # the only thing standing between it and the blacklist.
    clean = day_of_week_breakdown(pd.DataFrame({
        "entry_time": pd.date_range("2020-01-06 15:00", periods=40, freq="7D",
                                    tz="UTC"),
        "pnl": [25.0] * 40}))
    check("a weekday that never lost has an undefined PF and is NEVER dropped",
          pd.isna(clean.loc[clean["weekday"] == 0, "profit_factor"].iloc[0])
          and unprofitable_weekdays(clean, min_trades=10) == [])

    # A weekend row only exists when something traded then, and it is a bug
    # worth seeing rather than a session to prune.
    sat = pd.concat([dow_trades({0: 40.0, 1: 40.0, 2: 40.0, 3: 40.0, 4: 40.0}),
                     pd.DataFrame({
                         "entry_time": pd.date_range("2020-01-11 15:00",
                                                     periods=40, freq="7D",
                                                     tz="UTC"),
                         "pnl": [-500.0] * 40})], ignore_index=True)
    b = day_of_week_breakdown(sat)
    check("a losing SATURDAY is present in the table but never blacklisted",
          5 in set(b["weekday"]) and unprofitable_weekdays(b, 10) == [],
          str(sorted(set(b["weekday"]))))

    # Determinism: the same breakdown must give the same answer every time.
    picks = {tuple(d["weekday"] for d in unprofitable_weekdays(dow, 10))
             for _ in range(5)}
    check("the answer is deterministic", picks == {(0, 1)}, str(picks))


def test_stage1_regime_row() -> None:
    print("\n29b. Stage 1 writes optimal_regime, regime_pf and the kill switch")
    from backtest.baseline import (_regime_cell, _row, build_markdown_report,
                                   screen)
    from backtest.profiler import REGIMES

    dow = day_of_week_breakdown(dow_trades({0: -100.0, 1: -20.0, 2: 40.0,
                                            3: 40.0, 4: 40.0}))
    m = {"ok": True, "trade_count": 420, "profit_factor": 0.96, "sharpe": 0.1,
         "win_rate": 0.52, "max_drawdown_pct": -9.0, "total_pnl": -200.0,
         "gross_pnl": 5000.0, "total_costs": 900.0, "n_days": 300}

    profiles = {"A": _profile(hvt=(1.28, 120), hvr=(0.80, 150),
                              lvt=(1.05, 90), lvr=(0.62, 60)), "B": None}
    ok, why, best = screen(profiles)
    kept = _row("NQ", "15m", m, None, ok, why, dow, 1000, 1.0,
                profiles=profiles, best=best)

    check("the optimal regime is captured by name",
          kept["optimal_regime"] == REGIMES[0], str(kept["optimal_regime"]))
    check("...with the profit factor that cleared the bar",
          kept["regime_pf"] == 1.28 and kept["regime_trade_count"] == 120)
    check("...and the kill switch is the other three quadrants",
          kept["kill_switch_regimes"] == list(REGIMES[1:]),
          str(kept["kill_switch_regimes"]))
    check("the blended profit factor is BELOW 1.00 and the pair still "
          "survives - the screen is now the quadrant, not the average",
          kept["survived"] and kept["profit_factor_a"] == 0.96)

    thin = {"A": _profile(hvt=(1.90, 12)), "B": None}
    ok2, why2, best2 = screen(thin)
    dropped = _row("CL", "15m", m, None, ok2, why2, dow, 1000, 1.0,
                   profiles=thin, best=best2)
    check("a DROPPED configuration names no optimal regime",
          not dropped["survived"] and dropped["optimal_regime"] is None
          and dropped["regime_pf"] is None)
    check("...and derives NO kill switch - a stand-down list from a quadrant "
          "that failed the bar is an instruction nothing certified",
          dropped["kill_switch_regimes"] == [])
    check("the two states stay distinguishable in the summary column",
          (_regime_cell(kept), _regime_cell(dropped))
          == (f"{REGIMES[0]} (1.28)", "none"),
          f"{_regime_cell(kept)} / {_regime_cell(dropped)}")

    md = build_markdown_report("demo", {"Strategy": "demo"},
                               [kept, dropped], [])
    check("the report prints the FOUR-QUADRANT matrix per asset",
          all(r in md for r in REGIMES))
    check("...for every asset EVALUATED, drops included - the matrix of a "
          "failure is how you see whether it failed on edge or on sample size",
          "12 trades < 30" in md, md[md.index("### CL"):][:1500])
    check("...marking the quadrant that cleared both bars",
          "**CLEARS**" in md)
    check("...a quadrant the strategy never traded in is a ROW, not an "
          "omission", "no trades in this regime" in md)
    check("...the optimal regime and the kill switch, spelled out",
          "Optimal regime" in md and "Kill switch" in md
          and "do NOT trade in" in md)
    check("...and the circularity, in the same block",
          "IN-SAMPLE" in md and "Stage 3" in md and "best of four" in md)
    check("the summary table carries an Optimal regime column",
          "Optimal regime" in md and f"| {REGIMES[0]} (1.28) |" in md)

    check("the day-of-week table survives as DESCRIPTIVE only",
          "| Mon |" in md and "descriptive" in md)
    for gone in ("Dropped days", "exclude_days=[", "EXCLUDED",
                 "--no-drop-losing-days"):
        check(f"...and the calendar pruning is gone: no {gone!r}",
              gone not in md)


def test_stage1_survivors_leaderboard() -> None:
    print("\n29c. STAGE 1 SURVIVORS LEADERBOARD")
    from backtest.baseline import _row, screen, survivors_leaderboard
    from backtest.profiler import REGIMES

    dow = day_of_week_breakdown(dow_trades({0: -100.0, 1: 40.0, 2: 40.0,
                                            3: 40.0, 4: 40.0}))
    base = {"ok": True, "trade_count": 120, "win_rate": 0.55,
            "max_drawdown_pct": -6.0, "total_pnl": 900.0, "gross_pnl": 2000.0,
            "total_costs": 300.0, "n_days": 300, "sharpe": 0.8}

    def row(sym, tf, pf_a, profiles):
        ok, why, best = screen(profiles)
        return _row(sym, tf, {**base, "profit_factor": pf_a},
                    ({**base, "profit_factor": 1.51} if profiles["B"] else None),
                    ok, why, dow, 1, 1.0, profiles=profiles, best=best)

    # NQ survives on B's quadrant (1.72); ES on A's (1.20); CL clears nothing.
    # B's quadrant has to out-SCORE A's, not merely out-factor it: the two
    # versions are ranked on alpha contribution since 2026-08-21, so a sharp
    # 65-trade corner no longer beats a 200-trade book at 1.02.
    nq = row("NQ", "15m", 1.10, {"A": _profile(hvt=(1.02, 200)),
                                 "B": _profile(lvt=(1.72, 200))})
    es = row("ES", "5m", 1.08, {"A": _profile(hvr=(1.20, 310)), "B": None})
    cl = row("CL", "5m", 0.70, {"A": _profile(lvr=(0.70, 400)), "B": None})

    out = survivors_leaderboard([es, cl, nq])
    check("the leaderboard is titled as specified",
          "STAGE 1 SURVIVORS LEADERBOARD" in out)
    for col in ("SYMBOL", "TF", "PF (A)", "PF (B)", "OPTIMAL REGIME",
                "REGIME PF", "REGIME TRADES", "VER"):
        check(f"...and carries the {col!r} column", col in out)
    body = [ln for ln in out.splitlines()
            if ln.strip().startswith(("NQ", "ES", "CL"))]
    check("only SURVIVORS are listed - a dropped contract is not a survivor",
          len(body) == 2 and not any(ln.strip().startswith("CL")
                                     for ln in body), str(body))
    check("sorted by the REGIME profit factor the screen decided on, "
          "descending - NOT by either version's blended factor",
          body[0].strip().startswith("NQ"), str(body))
    check("the winning quadrant is named on the row", REGIMES[2] in body[0])
    check("...with the version that produced it, so a quadrant carried by a "
          "classifier is not read as one the rules found",
          "VB" in body[0] and "VA" in body[1], str(body))
    check("a Version B that never ran reads NOT RUN, never 0.00 or a dash",
          "NOT RUN" in body[1] and "0.00" not in body[1], body[1])
    check("the kill switch is NOT a column - it is always the other three, "
          "and spelling it out wraps the table",
          "KILL SWITCH" not in out)
    check("a stage where nothing cleared the firewall still prints the table "
          "and says so",
          "STAGE 1 SURVIVORS LEADERBOARD" in survivors_leaderboard([cl])
          and "no configuration cleared the regime firewall"
          in survivors_leaderboard([cl]))


def test_stage2_winners_leaderboard() -> None:
    print("\n29d. STAGE 2 OPTIMIZED WINNERS LEADERBOARD")
    from backtest.scan import winners_leaderboard

    rows = [
        {"symbol": "ES", "timeframe": "5m", "sharpe": 0.71,
         "profit_factor": 1.12, "max_drawdown_pct": -9.4,
         "winner": {"fast": 5, "slow": 20}, "exclude_days": [],
         "exclude_days_named": [], "selection": "x"},
        {"symbol": "NQ", "timeframe": "15m", "sharpe": 1.84,
         "profit_factor": 1.55, "max_drawdown_pct": -6.1,
         "winner": {"fast": 4, "slow": 30}, "exclude_days": [0, 4],
         "exclude_days_named": ["Mon", "Fri"], "selection": "x"},
        {"symbol": "GC", "timeframe": "15m", "sharpe": float("nan"),
         "profit_factor": None, "max_drawdown_pct": None, "winner": None,
         "exclude_days": [], "exclude_days_named": [], "selection": "x"},
    ]
    out = winners_leaderboard(rows)
    check("the leaderboard is titled as specified",
          "STAGE 2 OPTIMIZED WINNERS LEADERBOARD" in out)
    for col in ("SYMBOL", "TF", "IS PF", "IS SHARPE", "MAX DD",
                "WINNING PARAMS", "EXCLUDED DAYS"):
        check(f"...and carries the {col!r} column", col in out)
    body = [ln for ln in out.splitlines()
            if ln.strip().startswith(("NQ", "ES", "GC"))]
    check("sorted by in-sample Sharpe descending - the metric it selected on",
          body[0].strip().startswith("NQ") and body[1].strip().startswith("ES"),
          str(body))
    check("a contract with no measurable Sharpe sorts LAST, never disappears",
          body[-1].strip().startswith("GC") and "n/a" in body[-1], body[-1])
    check("the winning parameters are named in full, not summarised",
          "fast=4, slow=30" in body[0], body[0])
    check("the excluded days are on the row, so two rows fitted to different "
          "weeks are not read as comparable",
          "Mon, Fri" in body[0] and "none" in body[1], str(body[:2]))
    check("no sweep completed still prints the table",
          "no sweep completed" in winners_leaderboard([]))


def test_stage3_certification_leaderboard() -> None:
    print("\n29e. STAGE 3 GATE CERTIFICATION LEADERBOARD")
    from backtest.audit_gates import GATE_R, certification_leaderboard
    from backtest.report import FAIL, NOT_EVALUATED, PASS

    # Under the Regime-Switching Incubator Charter the verdict is Gate R -
    # the holdout profit factor and trade count INSIDE the designated
    # quadrant. Gates 1-3 are still on the row and still individually, but
    # they are advisory: NQ/A below certifies with a FAILING Gate 1, which is
    # the whole point of the change and would have been impossible before it.
    results = [
        {"symbol": "ES", "timeframe": "15m",
         "path": Path("gate_audit_ES_15m.json"),
         "status": {"A": FAIL}, "passed": {"A": False},
         "target_quadrant": "Q2",
         "gates": {"A": {"gate1": FAIL, "gate2": NOT_EVALUATED,
                         "gate3": NOT_EVALUATED, GATE_R: FAIL}},
         "regime_measured": {"A": {"profit_factor": 0.61,
                                   "trade_count": 400}},
         "exclude_days": []},
        {"symbol": "NQ", "timeframe": "15m",
         "path": Path("gate_audit_NQ_15m.json"),
         "status": {"A": PASS, "B": FAIL}, "passed": {"A": True, "B": False},
         "target_quadrant": "Q1",
         "gates": {"A": {"gate1": FAIL, "gate2": PASS, "gate3": PASS,
                         GATE_R: PASS},
                   "B": {"gate1": PASS, "gate2": FAIL,
                         "gate3": NOT_EVALUATED, GATE_R: FAIL}},
         "regime_measured": {"A": {"profit_factor": 1.42, "trade_count": 88},
                             "B": {"profit_factor": 0.90, "trade_count": 40}},
         "exclude_days": [0, 4]},
    ]
    out = certification_leaderboard(results)
    check("the leaderboard is titled as specified",
          "STAGE 3 GATE CERTIFICATION LEADERBOARD" in out)
    for col in ("SYMBOL", "TF", "QUAD", "GATE R (OOS REGIME)", "OOS PF",
                "OOS N", "GATE 1 (IS)", "GATE 2 (WFO/MC)", "GATE 3 (OOS)",
                "FINAL STATUS"):
        check(f"...and carries the {col!r} column", col in out)
    body = [ln for ln in out.splitlines()
            if ln.strip().startswith(("NQ", "ES"))]
    check("one row per (contract, VERSION) - A and B are certified separately",
          len(body) == 3, str(body))
    check("what passed sorts first", "CERTIFIED" in body[0]
          and "NOT CERTIFIED" not in body[0], body[0])

    # The charter's central claim, on the table: the verdict follows Gate R,
    # and an aggregate gate cannot veto it. Before 2026-08-21 this row would
    # have read NOT CERTIFIED on the strength of the Gate 1 FAIL beside it.
    check("a CERTIFIED row can carry a FAILING advisory Gate 1 - nothing is "
          "pruned on a blended-sample metric",
          "CERTIFIED" in body[0] and "FAIL" in body[0], body[0])
    check("the quadrant Gate R was measured in is on the row - an OOS profit "
          "factor with no quadrant beside it is a blended number under a "
          "regime-gated verdict", "Q1" in body[0], body[0])
    check("...with the quadrant's own profit factor and trade count",
          "1.42" in body[0] and "88" in body[0], body[0])
    check("a gate that was never run reads NOT EVAL, never PASS and never FAIL",
          "NOT EVAL" in out and sum("NOT EVAL" in ln for ln in body) == 2,
          str(body))
    check("a FAILING Gate R is NOT CERTIFIED however the advisory gates read",
          all("NOT CERTIFIED" in ln for ln in body[1:]), str(body[1:]))
    check("the certified week is on the row - the gates were run on it",
          "Mon, Fri" in body[0], body[0])
    check("no completed audit still prints the table",
          "nothing was certified" in certification_leaderboard([]))


def test_stage1_to_stage2_exclude_days(tmp: Path) -> None:
    print("\n29f. The handoff: per-pair exclusions, and who overrides whom")
    from backtest.pipeline import SURVIVORS_FILE, stage1_exclude_days
    from backtest.scan import resolve_exclude_days

    blob = {
        "surviving_pairs": [
            {"symbol": "NQ", "tf": "5m", "dropped_days": ["Monday", "Friday"],
             "exclude_days": [0, 4]},
            {"symbol": "NQ", "tf": "15m", "dropped_days": ["Friday"],
             "exclude_days": [4]},
            {"symbol": "GC", "tf": "15m", "dropped_days": [],
             "exclude_days": []},
        ]}
    m = stage1_exclude_days(blob)
    check("the exclusion is keyed per (symbol, timeframe), not globally",
          m == {("NQ", "5m"): (0, 4), ("NQ", "15m"): (4,)}, str(m))
    check("MULTIPLE days survive the handoff as a list, not collapsed to one",
          m[("NQ", "5m")] == (0, 4))
    check("a pair with nothing to exclude is ABSENT, not mapped to ()",
          ("GC", "15m") not in m)
    check("a handoff written before the contract existed yields no exclusions",
          stage1_exclude_days({"surviving_pairs": [{"symbol": "NQ",
                                                    "tf": "5m"}]}) == {}
          and stage1_exclude_days(None) == {} and stage1_exclude_days({}) == {})

    # Precedence, the part an operator has to be able to predict.
    check("with no CLI flag, Stage 1's decision applies automatically",
          resolve_exclude_days("NQ", "5m", None, m)[0] == (0, 4))
    check("...per pair, so NQ·15m gets Friday alone",
          resolve_exclude_days("NQ", "15m", None, m)[0] == (4,))
    days, why = resolve_exclude_days("NQ", "5m", (2, 3), m)
    check("an explicit --exclude-days OVERRIDES the artifact outright",
          days == (2, 3) and "CLI" in why, f"{days} · {why}")
    check("--ignore-stage1-exclude-days sweeps the whole week",
          resolve_exclude_days("NQ", "5m", None, m,
                               ignore_stage1=True)[0] is None)
    check("...but never disables an explicit --exclude-days",
          resolve_exclude_days("NQ", "5m", (1,), m,
                               ignore_stage1=True)[0] == (1,))
    check("a pair Stage 1 excluded nothing for sweeps unfiltered",
          resolve_exclude_days("GC", "15m", None, m) == (None, "none"))
    check("every answer carries its provenance, never a bare list",
          all(isinstance(resolve_exclude_days(*a)[1], str) and
              resolve_exclude_days(*a)[1]
              for a in (("NQ", "5m", None, m), ("GC", "15m", None, m),
                        ("NQ", "5m", (1,), m))))

    d = tmp / "dwd"
    write_stage(d / SURVIVORS_FILE, 1, "demo", blob)
    back = read_stage(d / SURVIVORS_FILE, 1, "demo")
    check("the mapping survives the JSON round trip",
          stage1_exclude_days(back) == m, str(stage1_exclude_days(back)))


def test_stage2_to_stage3_filter_inheritance(tmp: Path) -> None:
    print("\n29g. Stage 3 certifies the week Stage 2 optimised on")
    from backtest.audit_gates import _resolve_filters, load_params

    d = tmp / "dwd3"
    payload = {
        "symbol": "NQ", "timeframe": "15m", "params": {"trend_period": 200},
        "variants_tested": 12, "selection": "x",
        "entry_filters": {"exclude_days": [0, 4],
                          "exclude_days_named": ["Mon", "Fri"],
                          "exclude_days_source": "stage 1 Drop Losing Days"},
    }
    write_stage(d / BEST_PARAMS_FILE.format(symbol="NQ_15m"), 2, "demo",
                payload)
    _params, prov = load_params("demo", "NQ", d, {}, False, tf="15m")
    check("Stage 2's exclusion reaches Stage 3 on the winner",
          (prov.get("entry_filters") or {}).get("exclude_days") == [0, 4],
          str(prov.get("entry_filters")))

    base = {"news_filter": False, "news_window_minutes": 30.0,
            "news_kinds": None, "exclude_days": None}
    kw, why = _resolve_filters(dict(base), prov)
    check("...and is APPLIED, so the certification is not of a different "
          "strategy", kw["exclude_days"] == (0, 4) and "inherited" in why, why)
    kw, why = _resolve_filters({**base, "exclude_days": (3,)}, prov)
    check("an explicit --exclude-days on Stage 3 still wins",
          kw["exclude_days"] == (3,) and "CLI" in why, why)
    check("the news filter is NOT inherited - a rule-generated calendar must "
          "not creep into a gate verdict",
          _resolve_filters(dict(base), {"entry_filters": {
              "exclude_days": [], "news_filter": True}})[0]["news_filter"]
          is False)
    kw, why = _resolve_filters(dict(base), {"entry_filters": None})
    check("a best_params written before the contract inherits nothing",
          kw["exclude_days"] is None and why == "none", why)


def test_drop_losing_days_clis() -> None:
    print("\n29h. Stage 1's calendar flags are gone; the filters remain")
    out = {}
    for rel in ("backtest/baseline.py", "backtest/scan.py",
                "backtest/audit_gates.py"):
        r = subprocess.run([sys.executable, str(REPO / rel), "--help"],
                           capture_output=True, text=True, cwd=REPO, timeout=180)
        out[rel] = r.stdout if r.returncode == 0 else ""
        check(f"{rel} --help still exits 0", r.returncode == 0,
              (r.stderr or "")[-160:])
    # Stage 1's calendar pruning is GONE - the regime quadrant replaced it.
    # Its flags must be gone with it: a --no-drop-losing-days that parses and
    # changes nothing is worse than one that errors.
    for flag in ("--no-drop-losing-days", "--dow-min-pf", "--dow-min-trades"):
        check(f"Stage 1 no longer offers {flag}",
              flag not in out["backtest/baseline.py"])
    check("Stage 1 documents the REGIME bars in its place",
          "--min-profit-factor" in out["backtest/baseline.py"]
          and "--min-trades" in out["backtest/baseline.py"]
          and "quadrant" in out["backtest/baseline.py"].lower())
    check("Stage 2 still accepts a handoff that carries exclude_days - the "
          "inheritance path is unchanged, Stage 1 simply writes none",
          "--ignore-stage1-exclude-days" in out["backtest/scan.py"])
    check("Stage 2 documents --ignore-stage1-exclude-days",
          "--ignore-stage1-exclude-days" in out["backtest/scan.py"])
    check("all three still expose the manual --exclude-days that overrides it",
          all("--exclude-days" in v for v in out.values()))


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 29. The Version A / Version B handoff, Stage 1 -> Stage 2 -> Stage 3
# --------------------------------------------------------------------------
# Stage 1 screens BOTH versions and a pair survives on either, recording which
# one carried it. Until 2026-08-25 that answer travelled as far as the Stage 2
# summary row and then stopped: `--ml` was a single global flag an operator
# typed, so a pair that only Version B cleared was certified as Version A
# unless somebody remembered to type it. The failure is silent in every
# output - the Stage 3 summary is complete, every gate is filled in, and the
# version column simply reads `A`.
def test_version_b_survives_the_handoff(tmp: Path) -> None:
    print("\n29. A Version B survivor is evaluated as Version B")

    from backtest.pipeline import stage1_pairs
    from backtest.audit_gates import resolve_version_b
    import argparse

    survivors = {"surviving_pairs": [
        {"symbol": "NQ", "tf": "30m", "version": "B", "status": "PROMOTED",
         "optimal_regime": "High Volatility / Ranging", "quadrant": "Q2",
         "regime_pf": 1.4, "regime_trade_count": 61},
        {"symbol": "NQ", "tf": "1h", "version": "B", "status": "PROMOTED",
         "optimal_regime": "High Volatility / Ranging", "quadrant": "Q2",
         "regime_pf": 1.2, "regime_trade_count": 44},
        {"symbol": "ES", "tf": "15m", "version": "A", "status": "PROMOTED",
         "optimal_regime": "Low Volatility / Trending", "quadrant": "Q3",
         "regime_pf": 1.1, "regime_trade_count": 90},
    ]}

    pairs = stage1_pairs(survivors)
    by_pair = {(p["symbol"], p["tf"]): p for p in pairs}
    check("Stage 1's handoff carries the version per pair",
          [p["version"] for p in pairs] == ["B", "B", "A"],
          str([p["version"] for p in pairs]))

    # -- Stage 2: the confirmation is requested for B and not for A ---------
    from backtest.scan import _resolve_ml_confirmation
    args = argparse.Namespace(ml=False, no_stage1_ml=False, ml_threshold=0.5)
    scan = {"winner": {"params": {"ema_period": 20}}}
    for sym, tf, want in (("NQ", "30m", True), ("NQ", "1h", True),
                          ("ES", "15m", False)):
        scope = {"version": by_pair[(sym, tf)]["version"]}
        # `bars=None` short-circuits before any engine work; what is under
        # test is WHICH pairs are asked for, not the backtest itself.
        record = _resolve_ml_confirmation(args, scope, scan, "x.py", sym, tf,
                                          None, None, {}, "demo")
        asked = "no in-memory bars" in str(record.get("reason", ""))
        check(f"Stage 2 {'requests' if want else 'does not request'} the ML "
              f"confirmation for {sym} {tf} (stage1_version="
              f"{scope['version']})",
              asked is want, record.get("reason", ""))

    # -- Stage 3: Version B is certified for a B survivor, with no --ml -----
    args = argparse.Namespace(ml=False, no_stage1_ml=False)
    for sym, tf, want in (("NQ", "30m", True), ("NQ", "1h", True),
                          ("ES", "15m", False)):
        target = {"symbol": sym, "timeframe": tf,
                  "stage1_version": by_pair[(sym, tf)]["version"]}
        got, why = resolve_version_b(target, args)
        check(f"Stage 3 {'certifies' if want else 'does not certify'} Version "
              f"B for {sym} {tf} with no --ml typed", got is want, why)

    # A global --ml is a SUPERSET and can never turn a B survivor into an A
    # certification - the direction that matters.
    args_ml = argparse.Namespace(ml=True, no_stage1_ml=False)
    check("--ml still certifies Version B for the B survivors",
          all(resolve_version_b({"stage1_version": "B"}, args_ml)[0]
              for _ in range(1)))
    check("--ml certifies Version B for the A survivors too, which is a "
          "superset and not the bug",
          resolve_version_b({"stage1_version": "A"}, args_ml)[0] is True)

    # The override exists so a B survivor whose classifier cannot be rebuilt
    # stays auditable, and it RECORDS that it fired.
    args_off = argparse.Namespace(ml=False, no_stage1_ml=True)
    got, why = resolve_version_b({"stage1_version": "B"}, args_off)
    check("--no-stage1-ml overrides the handoff and says so",
          got is False and "no-stage1-ml" in why, why)

    # A pair with no recorded version is Version A and says so, rather than
    # defaulting to an hours-long classifier run nobody asked for.
    got, why = resolve_version_b({}, argparse.Namespace(ml=False,
                                                        no_stage1_ml=False))
    check("a pair with no recorded version certifies Version A only",
          got is False and "no stage1_version" in why, why)

    # -- the orchestrator's independent second reading ---------------------
    from backtest.run_pipeline import check_version_b_certified
    from backtest.pipeline import pipeline_dir, write_stage, STAGE3_SUMMARY_FILE

    d = pipeline_dir("demo_vb", tmp / "vb", create=True)
    write_stage(d / STAGE3_SUMMARY_FILE, 3, "demo_vb", {"results": [
        {"symbol": "NQ", "timeframe": "30m", "version": "A"},
        {"symbol": "NQ", "timeframe": "30m", "version": "B"},
        {"symbol": "NQ", "timeframe": "1h", "version": "A"},
    ]})
    gaps = check_version_b_certified("demo_vb", [("NQ", "30m"), ("NQ", "1h")],
                                     str(tmp / "vb"))
    check("the orchestrator passes the pair that WAS certified on B",
          not any("30m" in g for g in gaps), str(gaps))
    check("...and names the pair that was audited as Version A only",
          len(gaps) == 1 and "1h" in gaps[0], str(gaps))


# --------------------------------------------------------------------------
# 30. Stage 2's two minimum robustness bars
# --------------------------------------------------------------------------
def test_stage2_refuses_a_spike_and_a_ruinous_cell() -> None:
    print("\n30. Stage 2 exports no spike and no ruined account")

    from backtest.scan import (FRAGILE_RUIN, FRAGILE_SPIKE,
                               SELECTED_PRUNED_FRAGILE, _select_best_row,
                               fragility_of)
    from backtest.pipeline import RUIN_MIN_DRAWDOWN_PCT
    from backtest.report import PASS as GATE_PASS

    def cell(sharpe, *, spike=False, dd=-20.0, gate1=GATE_PASS, ema=20):
        return {"ema_period": ema, "sharpe": sharpe, "gate1": gate1,
                "max_drawdown_pct": dd, "is_spike": spike,
                "plateau_score": sharpe}

    # A spike with the best Sharpe on the grid, beside a stable shelf.
    rows = [cell(3.10, spike=True, ema=20), cell(1.40, ema=21),
            cell(1.35, ema=22)]
    best, selection = _select_best_row(rows)
    check("the spike is NOT exported when a stable plateau exists",
          best is not None and best["ema_period"] == 21,
          f"picked ema_period={None if best is None else best['ema_period']}")
    check("...and the winner is the stable cell, not merely a lower-ranked "
          "spike", best is not None and not best["is_spike"])

    # A ruinous cell with the best Sharpe. -100% is not a deeper loss; it is
    # arithmetic that has stopped describing an account.
    rows = [cell(2.90, dd=RUIN_MIN_DRAWDOWN_PCT, ema=20), cell(1.10, ema=21)]
    best, _ = _select_best_row(rows)
    check("a cell that drew down to the ruin boundary is not exported",
          best is not None and best["ema_period"] == 21,
          f"picked ema_period={None if best is None else best['ema_period']}")
    rows = [cell(2.90, dd=-140.0, ema=20), cell(1.10, ema=21)]
    best, _ = _select_best_row(rows)
    check("...and neither is one past it", best["ema_period"] == 21)

    # The bars are applied BEFORE Gate 1 is preferred. A ruinous spike that is
    # the grid's only Gate 1 pass must not be exported under a "GATE 1 PASS"
    # heading - the most persuasive label this stage can print on its least
    # defensible row.
    rows = [cell(2.5, spike=True, dd=-160.0, gate1=GATE_PASS, ema=20),
            cell(0.9, gate1="FAIL", ema=21)]
    best, selection = _select_best_row(rows)
    check("a ruinous spike does not win on being the only Gate 1 pass",
          best is not None and best["ema_period"] == 21, str(selection))

    # Nothing eligible at all -> PRUNED_FRAGILE, and NO winner.
    rows = [cell(2.5, spike=True, ema=20), cell(2.4, dd=-101.0, ema=21)]
    best, selection = _select_best_row(rows)
    check("a grid with no eligible cell yields no winner",
          best is None, str(best))
    check("...and is labelled PRUNED_FRAGILE rather than NO SHARPE",
          selection == SELECTED_PRUNED_FRAGILE, selection)

    # The reasons are distinguishable, and ruin outranks the spike label.
    check("an isolated spike is reported as a spike",
          fragility_of(cell(1.0, spike=True)) == FRAGILE_SPIKE)
    check("a ruined account is reported as ruinous, even when it is also a "
          "spike",
          fragility_of(cell(1.0, spike=True, dd=-120.0)) == FRAGILE_RUIN)
    check("a healthy cell is not fragile", fragility_of(cell(1.0)) is None)

    # The counterfactual is NOT pruned - it answers what the old rule would
    # have picked, which is a different question.
    rows = [cell(3.10, spike=True, ema=20), cell(1.40, ema=21)]
    old_best, _ = _select_best_row(rows, rank="sharpe", prune_fragile=False)
    check("the pre-charter counterfactual still names the spike, so the "
          "comparison stays honest", old_best["ema_period"] == 20)

    # Stage 3 admits a pair only when the Stage 2 status reads OPTIMIZED.
    from backtest.audit_gates import stage2_targets
    from backtest.scan import STAGE2_PRUNED_FRAGILE
    targets = stage2_targets({"results": [
        {"symbol": "NQ", "timeframe": "30m", "status": STAGE2_PRUNED_FRAGILE},
        {"symbol": "ES", "timeframe": "30m", "status": "OPTIMIZED"},
    ]}, "30m")
    certifiable = {t["symbol"]: t["certifiable"] for t in targets}
    check("a PRUNED_FRAGILE pair is not certifiable by Stage 3",
          certifiable == {"NQ": False, "ES": True}, str(certifiable))

    # ...and the REASON must say the sweep ran. A fragility prune carries no
    # `error` text, because it is not an error, so the bare fallback used to
    # report Stage 2's one deliberate anti-overfitting verdict as "stage 2
    # recorded no parameters" - which reads as a broken run and sends an
    # operator hunting a crash that never happened.
    from backtest.audit_gates import stage2_skip_reason
    fragile = stage2_skip_reason(STAGE2_PRUNED_FRAGILE)
    check("a fragility prune says the sweep RAN and locked no winner",
          "locked no winner" in fragile and "not a failed run" in fragile,
          fragile)
    check("...and never borrows the wording of a sweep that produced nothing",
          "recorded no parameters" not in fragile, fragile)
    broke = stage2_skip_reason("ERROR")
    check("a status that is neither still reports no parameters, and NAMES "
          "itself so the two cannot be read as one",
          "recorded no parameters" in broke and "ERROR" in broke, broke)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pipefilt_") as td:
        tmp = Path(td)
        test_session_dates()
        test_rule_calendar()
        test_published_calendar(tmp)
        test_news_mask()
        test_fill_bar_widening()
        test_weekday_mask()
        test_apply_entry_filters()
        test_engine_hook()
        test_engine_news_hook()
        test_dow_breakdown()
        test_scorecard_gate_suppression()
        test_pipeline_handoff(tmp)
        test_stage_clis()
        test_baseline_screen()
        test_baseline_report(tmp)
        test_window_guard()
        test_stage2_to_stage3_handoff(tmp)
        test_cost_drag()
        test_promote_certification(tmp)
        test_filter_config_kwargs()
        test_multi_timeframe_cli()
        test_multi_timeframe_handoff(tmp)
        test_unprofitable_weekdays()
        test_stage1_regime_row()
        test_stage1_survivors_leaderboard()
        test_stage2_winners_leaderboard()
        test_stage3_certification_leaderboard()
        test_stage1_to_stage2_exclude_days(tmp)
        test_stage2_to_stage3_filter_inheritance(tmp)
        test_drop_losing_days_clis()
        test_version_b_survives_the_handoff(tmp)
        test_stage2_refuses_a_spike_and_a_ruinous_cell()

    print("\n" + "=" * 60)
    if _failures:
        print(f"  {len(_failures)} CHECK(S) FAILED")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
