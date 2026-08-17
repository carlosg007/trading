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


def test_baseline_screen() -> None:
    print("\n14. Stage 1's survival screen")
    from backtest.baseline import screen

    keep, why = screen({"ok": True, "trade_count": 200, "profit_factor": 1.05})
    check("profit factor 1.05 survives", keep, why)
    drop, why = screen({"ok": True, "trade_count": 200, "profit_factor": 0.99})
    check("profit factor 0.99 is dropped", not drop, why)
    edge, why = screen({"ok": True, "trade_count": 200, "profit_factor": 1.00})
    check("exactly 1.00 is on the boundary and survives", edge, why)
    none, why = screen({"ok": True, "trade_count": 0})
    check("no trades is dropped with its OWN reason, not as a 0.00 PF",
          not none and "never fired" in why, why)
    fail, why = screen({"ok": False, "error": "boom"})
    check("a failed run is dropped and says so", not fail and "boom" in why, why)
    check("the bar is adjustable per run",
          screen({"ok": True, "trade_count": 200, "profit_factor": 1.10},
                 min_profit_factor=1.25)[0] is False)


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

    source = REPO / "strategies" / "experimental" / "sma_crossover.py"
    if not source.exists():
        check("sma_crossover.py is present to promote", False, str(source))
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
def _load_etf():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_etf_test", REPO / "strategies" / "experimental" / "ema_trend_filter.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def globex_bars(days: int = 60, seed: int = 3) -> pd.DataFrame:
    """15m bars covering the whole 24h Globex day, so sessions actually roll."""
    rows, px = [], 5000.0
    rng = np.random.default_rng(seed)
    for d in pd.bdate_range("2024-01-02", periods=days, tz="UTC"):
        for k in range(96):
            ts = d + pd.Timedelta(minutes=15 * k)
            px *= 1 + rng.normal(0.00004, 0.0009)
            rows.append((ts, "NQ", px, px * 1.0012, px * 0.9988, px, 500 + k))
    return pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low",
                                       "close", "volume"])


def test_session_vwap() -> None:
    print("\n20. Session VWAP: resets at the CME open, and is causal")
    m = _load_etf()
    bars = globex_bars()

    ordinals = m._session_ordinal(bars["ts"])
    check("the VWAP session rule is the SAME rule --exclude-days uses",
          bool((ordinals == np.asarray(session_date(bars["ts"]),
                                       dtype="datetime64[D]").astype("int64")
                ).all()))
    # The bug this check exists for: dividing .asi8 by nanoseconds-per-day on a
    # microsecond-resolution index collapses every date onto one ordinal, the
    # VWAP never resets, and the series still looks like a plausible price line.
    n_sessions = len(set(ordinals.tolist()))
    check("sixty days of bars produce many sessions, not one",
          n_sessions > 50, f"{n_sessions} sessions")

    vwap = m._session_vwap(bars)
    first = pd.Series(ordinals).ne(pd.Series(ordinals).shift()).to_numpy()
    typical = ((bars["high"] + bars["low"] + bars["close"]) / 3).to_numpy()
    check("the first bar of each session has VWAP == its own typical price",
          bool(np.allclose(vwap.to_numpy()[first], typical[first])),
          f"{int(first.sum())} resets")

    contained = all(
        vwap[ordinals == sess].min() >= bars["low"][ordinals == sess].min() - 1e-9
        and vwap[ordinals == sess].max() <= bars["high"][ordinals == sess].max() + 1e-9
        for sess in set(ordinals.tolist()))
    check("VWAP never leaves its own session's price range - no volume leaks "
          "across the reset", contained)

    # Causality, checked directly: recomputing on a truncated frame must give
    # the same value at the last bar. A session-total (rather than cumulative)
    # VWAP would fail this at every bar but the last of each session.
    cut = 4000
    check("VWAP at bar i is unchanged when every later bar is deleted",
          abs(float(m._session_vwap(bars.iloc[:cut].reset_index(drop=True)).iloc[-1])
              - float(vwap.iloc[cut - 1])) < 1e-9)

    zero_vol = bars.copy()
    zero_vol.loc[:, "volume"] = 0.0
    check("a session with no traded volume has NaN VWAP, never 0.0",
          bool(m._session_vwap(zero_vol).isna().all()))


def test_rsi() -> None:
    print("\n21. RSI(14): Wilder, and its undefined ends")
    m = _load_etf()

    rising = pd.Series(np.arange(1, 40, dtype=float))
    r = m._rsi(rising, 14)
    check("fourteen straight up bars give RSI 100, not inf or NaN",
          float(r.iloc[-1]) == 100.0, str(r.iloc[-1]))
    falling = pd.Series(np.arange(40, 1, -1, dtype=float))
    check("straight down gives RSI 0", float(m._rsi(falling, 14).iloc[-1]) == 0.0)
    flat = pd.Series(np.full(40, 100.0))
    check("a dead-flat window is RSI 50 by convention, not 0/0",
          float(m._rsi(flat, 14).iloc[-1]) == 50.0)
    check("warm-up stays NaN rather than being painted 50",
          bool(m._rsi(rising, 14).iloc[:14].isna().all()))

    mixed = pd.Series(np.cumsum(np.random.default_rng(5).normal(0, 1, 200)) + 100)
    rm = m._rsi(mixed, 14).dropna()
    check("RSI stays inside [0, 100]", bool((rm >= 0).all() and (rm <= 100).all()))


def _candidate_mask(m, bars: pd.DataFrame, **p) -> np.ndarray:
    """
    The raw CANDIDATE triggers for one toggle combination, rebuilt from the
    module's own series.

    Rebuilt rather than read off `signal_fn`, because `signal_fn` returns the
    POSITION WALK's entries — one at a time, a trigger arriving while a
    position is open being ignored — and the subtractive property being checked
    here is a property of the candidates, not of the walk. See "A filter
    subtracts candidates, not trades" in the module docstring.
    """
    s_ = m._series(bars, p["fast_period"], p["slow_period"], p["trend_period"])
    ready = s_["fast"].notna() & s_["slow"].notna() & s_["atr"].notna()
    if p["use_trend"]:
        ready &= s_["trend"].notna()
    if p["use_vwap"]:
        ready &= s_["vwap"].notna()
    if p["use_rsi"]:
        ready &= s_["rsi"].notna()
    if p["use_volatility"]:
        ready &= s_["atr_ma"].notna()

    above = (s_["fast"] > s_["slow"]) & ready
    below = (s_["fast"] < s_["slow"]) & ready
    prev = ready.shift(1, fill_value=False)
    cross_up = above & ~above.shift(1, fill_value=False) & prev
    cross_down = below & ~below.shift(1, fill_value=False) & prev

    window, _flat = m._session_masks(bars["ts"])
    close = bars["close"]
    yes = pd.Series(True, index=bars.index)
    long_c = (cross_up
              & (close > s_["trend"] if p["use_trend"] else yes)
              & (close > s_["vwap"] if p["use_vwap"] else yes)
              & (s_["rsi"] > m.RSI_MIDLINE if p["use_rsi"] else yes)
              & (s_["atr"] > s_["atr_ma"] if p["use_volatility"] else yes)
              & ready).to_numpy() & window
    short_c = (cross_down
               & (close < s_["trend"] if p["use_trend"] else yes)
               & (close < s_["vwap"] if p["use_vwap"] else yes)
               & (s_["rsi"] < m.RSI_MIDLINE if p["use_rsi"] else yes)
               & (s_["atr"] > s_["atr_ma"] if p["use_volatility"] else yes)
               & ready).to_numpy() & window
    return long_c | short_c


def test_filter_toggles() -> None:
    print("\n22. Filter toggles: each one subtracts CANDIDATES")
    import itertools

    m = _load_etf()
    bars = globex_bars(days=90, seed=11)
    TOGGLES = ("use_trend", "use_vwap", "use_rsi", "use_volatility")
    ALL_OFF = dict.fromkeys(TOGGLES, False)
    BASE = {"fast_period": 9, "slow_period": 21, "trend_period": 200}

    def realised(**over):
        e, x, se, sx = m.signal_fn(bars, **{**BASE, **ALL_OFF, **over})
        return e.to_numpy() | se.to_numpy()

    def candidates(**over):
        return _candidate_mask(m, bars, **{**BASE, **ALL_OFF, **over})

    bare = candidates()
    check("with every toggle off the candidates are the bare crossover",
          bare.sum() > 0, f"{bare.sum()} candidate triggers")

    # THE invariant: turning any toggle on can only remove candidates. Checked
    # for all sixteen combinations against every combination one toggle poorer,
    # not just against the bare case.
    holds, broke = True, ""
    for combo in itertools.product([False, True], repeat=4):
        on = dict(zip(TOGGLES, combo))
        c_on = candidates(**on)
        for j, name in enumerate(TOGGLES):
            if not combo[j]:
                continue
            fewer = dict(on, **{name: False})
            if (c_on & ~candidates(**fewer)).any():
                holds, broke = False, f"{on} vs {fewer}"
    check("adding any filter to any combination only ever removes candidates",
          holds, broke)

    # How hard each one bites is a property of THIS fixture, not of the
    # strategy - on a random-walk fixture RSI sits near 50 at most crossings
    # and removes nothing. What is asserted is only that none of them ADDS a
    # candidate; the counts are printed because a filter that removes nothing
    # on real bars is one to drop rather than keep as decoration.
    for name in TOGGLES:
        n_on = int(candidates(**{name: True}).sum())
        check(f"{name} never adds a candidate "
              f"({n_on} of {int(bare.sum())} survive on this fixture)",
              n_on <= int(bare.sum()), f"{n_on} vs {int(bare.sum())}")

    # The realised trade list is NOT nested, and that is the documented
    # behaviour rather than a defect: declining an early trigger leaves the
    # strategy flat for a later one it would have been holding through. This is
    # pinned so nobody "fixes" the walk into nesting them, and so the module's
    # correction note keeps a test behind it.
    r_vwap, r_bare = realised(use_vwap=True), realised()
    extra = int((r_vwap & ~r_bare).sum())
    check("a filtered run can take entries the unfiltered run never did",
          extra > 0, f"{extra} entries exist only in the filtered run")
    check("...while its candidates remain a strict subset",
          not bool((candidates(use_vwap=True) & ~bare).any()))
    check("so trade COUNT is not the test of whether a filter binds",
          True, f"realised {int(r_vwap.sum())} vs {int(r_bare.sum())}")

    # Every realised entry must be one of its own configuration's candidates.
    for combo in itertools.product([False, True], repeat=4):
        on = dict(zip(TOGGLES, combo))
        if (realised(**on) & ~candidates(**on)).any():
            check(f"realised entries are always candidates ({on})", False)
            break
    else:
        check("every realised entry is a candidate of its own configuration",
              True)

    # Each active filter must actually hold at every entry it allowed.
    s_ = m._series(bars, 9, 21, 200)
    e, _x, se, _sx = m.signal_fn(bars, **{**BASE, "use_trend": True,
                                          "use_vwap": True, "use_rsi": True,
                                          "use_volatility": True})
    cl, vw, rsi = (bars["close"].to_numpy(), s_["vwap"].to_numpy(),
                   s_["rsi"].to_numpy())
    tr, atr, ama = (s_["trend"].to_numpy(), s_["atr"].to_numpy(),
                    s_["atr_ma"].to_numpy())
    longs, shorts = np.flatnonzero(e.to_numpy()), np.flatnonzero(se.to_numpy())
    check("with all four on, every long entry satisfies all four conditions",
          all(cl[i] > tr[i] and cl[i] > vw[i] and rsi[i] > 50.0
              and atr[i] > ama[i] for i in longs), f"{len(longs)} longs")
    check("and every short entry satisfies their mirrors",
          all(cl[i] < tr[i] and cl[i] < vw[i] and rsi[i] < 50.0
              and atr[i] > ama[i] for i in shorts), f"{len(shorts)} shorts")


def test_toggle_warmup_and_dead_axis() -> None:
    print("\n23. A disabled filter costs nothing, including its warm-up")
    m = _load_etf()
    bars = globex_bars(days=90, seed=11)
    BASE = {"fast_period": 9, "slow_period": 21}
    OFF = dict(use_trend=False, use_vwap=False, use_rsi=False,
               use_volatility=False)

    def first_entry(**over):
        e, _x, se, _sx = m.signal_fn(bars, **{**BASE, **OFF, **over})
        idx = np.flatnonzero(e.to_numpy() | se.to_numpy())
        return int(idx[0]) if len(idx) else None

    on = first_entry(use_trend=True, trend_period=400)
    off = first_entry(use_trend=False, trend_period=400)
    check("the trend filter delays the first entry past its EMA warm-up",
          on is not None and off is not None and on > off, f"{off} → {on}")
    check("...and switching it off does NOT inherit that 400-bar wait",
          off < 400, f"first entry at bar {off}")

    # `trend_period` is a dead axis with the filter off - the grid's declared
    # cell count exceeds the number of distinct strategies because of it.
    def sig(**over):
        e, _x, se, _sx = m.signal_fn(bars, **{**BASE, **OFF, **over})
        return e.to_numpy() | se.to_numpy()

    check("with use_trend=False, trend_period 200 and 400 are identical",
          bool((sig(use_trend=False, trend_period=200)
                == sig(use_trend=False, trend_period=400)).all()))
    check("with use_trend=True they differ, so the axis is live when used",
          bool((sig(use_trend=True, trend_period=200)
                != sig(use_trend=True, trend_period=400)).any()))

    # RSI/volatility warm-ups are shorter than the trend's, so this checks the
    # per-toggle assembly rather than one blanket condition.
    check("switching only the volatility filter on delays the start too",
          first_entry(use_volatility=True) >= off)


def test_toggle_causality() -> None:
    """No toggle combination lets a signal depend on a later bar."""
    print("\n24. Causality holds for every toggle combination")
    import itertools

    m = _load_etf()
    bars = globex_bars(days=60, seed=21)
    CUT = 4000

    # Deleting every bar after CUT must not change any signal before it. This
    # catches lookahead the AST validator cannot see - a resample, a centred
    # window, a groupby-transform over a whole session - because it tests the
    # property directly rather than the syntax that usually causes it.
    #
    # The truncated frame's FINAL bar is excluded, and only that one.
    # `_session_masks` marks the last bar of a frame as a session-flatten bar
    # because it has no successor, which is correct - a position is never left
    # open past the end of the data - and is a fact about the frame boundary
    # rather than about the signals.
    bad = []
    for combo in itertools.product([False, True], repeat=4):
        kw = dict(zip(("use_trend", "use_vwap", "use_rsi", "use_volatility"),
                      combo))
        full = [a.to_numpy()[:CUT - 1] for a in m.signal_fn(bars, **kw)]
        trunc = [a.to_numpy()[:CUT - 1] for a in
                 m.signal_fn(bars.iloc[:CUT].reset_index(drop=True), **kw)]
        if not all(bool((f == t).all()) for f, t in zip(full, trunc)):
            bad.append(kw)
    check("all 16 toggle combinations: truncating the future changes no "
          "earlier signal", not bad, str(bad[:2]))

    # And the same for the indicator series the tear sheet draws, since a
    # non-causal line under a causal signal is its own kind of wrong.
    for name in ("fast", "slow", "trend", "vwap", "rsi", "atr", "atr_ma"):
        a = m._series(bars, 9, 21, 200)[name].to_numpy()[:CUT]
        b = m._series(bars.iloc[:CUT].reset_index(drop=True), 9, 21,
                      200)[name].to_numpy()
        check(f"{name} is unchanged when every later bar is deleted",
              bool(np.allclose(a, b, equal_nan=True)))


def test_toggle_validation() -> None:
    print("\n25. Toggles are booleans, not truthy values")
    m = _load_etf()
    for bad in ("false", "False", 0, 1, 1.0, None, []):
        ok, _msg = raises(lambda b=bad: m.make_signal_fn(use_trend=b),
                          ValueError)
        check(f"use_trend={bad!r} is refused rather than coerced", ok)
    for name in ("use_vwap", "use_rsi", "use_volatility"):
        ok, msg = raises(lambda n=name: m.make_signal_fn(**{n: "false"}),
                         ValueError)
        check(f"{name} is checked too, and the error names it",
              ok and name in msg, msg[:60])
    check("np.bool_ is accepted - the grid round-trips through numpy",
          m.make_signal_fn(use_trend=np.bool_(True)) is not None)


def test_strategy_declarations() -> None:
    print("\n26. The module's four declarations, after the refactor")
    from agents.tier3_workers import load_strategy
    from backtest.scan import expand_grid

    m = _load_etf()
    path = REPO / "strategies" / "experimental" / "ema_trend_filter.py"
    bars = globex_bars(days=40)

    check("DEFAULT_PARAMS matches the specification block",
          m.DEFAULT_PARAMS == {"fast_period": 9, "slow_period": 21,
                               "trend_period": 200, "use_trend": True,
                               "use_vwap": True, "use_rsi": False,
                               "use_volatility": False, "sl_atr_mult": 1.5,
                               "tp_atr_mult": 2.0, "trailing": False},
          str(m.DEFAULT_PARAMS))
    combos = expand_grid(m.PARAM_GRID)
    check("PARAM_GRID is the declared 1,296 cells", len(combos) == 1296,
          str(len(combos)))
    check("the grid's axes are the specified ones",
          m.PARAM_GRID["trend_period"] == [200, 400]
          and m.PARAM_GRID["use_trend"] == [True, False]
          and m.PARAM_GRID["use_vwap"] == [True, False]
          and m.PARAM_GRID["use_rsi"] == [False]
          and m.PARAM_GRID["use_volatility"] == [False]
          and m.PARAM_GRID["sl_atr_mult"] == [1.0, 1.5, 2.0]
          and m.PARAM_GRID["tp_atr_mult"] == [1.5, 2.0, 3.0]
          and m.PARAM_GRID["trailing"] == [False, True])
    # `trend_period` is dead wherever use_trend is False, so the declared cell
    # count overstates the number of distinct strategies. Both numbers are
    # pinned: the docstring quotes them, and a grid edit that changes one
    # without the other should fail here rather than in a leaderboard.
    distinct = {tuple(sorted((k, v) for k, v in c.items()
                             if not (k == "trend_period" and not c["use_trend"])))
                for c in combos}
    check("1,296 declared cells are 972 distinct signal configurations",
          len(distinct) == 972, str(len(distinct)))
    check("...so variants_tested OVERSTATES the search, the safe direction",
          len(combos) > len(distinct))

    fn, info = load_strategy(path, {})
    check("load_strategy binds the new defaults",
          info["bound_params"]["trend_period"] == 200)
    out = fn(bars)
    check("signal_fn returns the four-mask form", isinstance(out, tuple)
          and len(out) == 4)
    check("every mask is boolean and bars-length",
          all(len(s_) == len(bars) and s_.dtype == bool for s_ in out))

    ind = m.indicators(bars)
    check("indicators draws the session VWAP when use_vwap is on",
          "Session VWAP" in ind)
    # A line the entries did not respect makes the trades that cross it look
    # like bugs. Omitted, not greyed out.
    off = m.indicators(bars, use_trend=False, use_vwap=False)
    check("no Trend EMA line when use_trend is off",
          not any("Trend EMA" in k for k in off), str(list(off)))
    check("no VWAP line when use_vwap is off",
          "Session VWAP" not in off, str(list(off)))
    check("the trigger EMAs and the stop are always drawn",
          any("Fast EMA" in k for k in off) and any("Slow EMA" in k for k in off)
          and any("Stop" in k for k in off), str(list(off)))
    check("every indicator series is the full length of the frame",
          all(len(v) == len(bars) for v in ind.values()),
          str({k: len(v) for k, v in ind.items()}))
    check("RSI is NOT drawn - it is a 0-100 oscillator on a price axis",
          not any("RSI" in k for k in ind))

    logic = m.LOGIC
    check("LOGIC names VWAP and RSI in the entry sentence",
          "VWAP" in logic["entry"] and "RSI" in logic["entry"])
    check("LOGIC carries a slot for every toggle, so the card states which "
          "filters ran",
          all(f"{{{t}}}" in logic["entry"]
              for t in ("use_trend", "use_vwap", "use_rsi", "use_volatility")))
    check("LOGIC's slots are all bindable parameters",
          all(f"{{{k}}}" not in logic["entry"] + logic["exit"]
              or k in m.DEFAULT_PARAMS
              for k in ("fast_period", "slow_period", "trend_period",
                        "sl_atr_mult", "tp_atr_mult", "trailing")))
    # The card must render with the run's own values substituted in.
    _fn2, info2 = load_strategy(path, {"trend_period": 400})
    check("the strategy card states the bound parameters, not the defaults",
          "400" in info2["logic"]["entry"], info2["logic"]["entry"][:60])


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
        test_window_guard()
        test_stage2_to_stage3_handoff(tmp)
        test_cost_drag()
        test_promote_certification(tmp)
        test_filter_config_kwargs()
        test_session_vwap()
        test_rsi()
        test_filter_toggles()
        test_toggle_warmup_and_dead_axis()
        test_toggle_causality()
        test_toggle_validation()
        test_strategy_declarations()
        test_multi_timeframe_cli()
        test_multi_timeframe_handoff(tmp)

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
