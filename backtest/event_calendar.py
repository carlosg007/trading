"""
backtest/event_calendar.py - causal time-of-event filters on ENTRIES.

Location: ~/src/trading/backtest/event_calendar.py

Two filters live here, and they are the same shape: given the bar timestamps,
produce a boolean mask of the bars an entry may NOT be taken on. Neither of
them looks at a price, so neither can be the source of an edge - they can only
remove trades.

    is_news_blocked(timestamps, window_minutes=30)  -> mask
    weekday_blocked(timestamps, exclude_days=[0, 4]) -> mask
    apply_entry_filters(...)                         -> filtered masks + stats

Three rules the engine depends on, all of them silent when wrong
----------------------------------------------------------------
1. **Entries only, never exits.** Suppressing an exit would hold a position
   through an FOMC statement, which is the opposite of what a news filter is
   for. `apply_entry_filters` takes the two entry masks and returns two entry
   masks; it is not given the exits at all, so the mistake cannot be made here.

2. **A signal is judged on the bar it FILLS, not the bar it fires.** The engine
   fills at the NEXT bar's open, so blocking bar `i` alone still lets a signal
   on `i-1` fill inside the window. Every mask this module hands the engine is
   therefore widened by one bar backwards (`_widen_to_fill_bar`): a signal is
   blocked if its own bar or its fill bar is blocked. Without that widening a
   30-minute FOMC window leaks exactly one entry per event, which is small
   enough to never look wrong and is the entire population the filter exists
   to remove.

3. **The blocking is causal, and that needs saying plainly.** Blocking the
   half hour BEFORE a release is not lookahead: US macro release schedules are
   published a year ahead, so "08:30 ET on the first Friday is payrolls" is
   ex-ante public information. What is NOT knowable in advance is the OUTCOME,
   and nothing here reads one - no print, no surprise index, no realized move.
   A filter keyed on the outcome would be lookahead of the purest kind, and the
   line between the two is the whole reason this module only ever sees
   timestamps.

Where the event dates come from, and why that matters more than the code
------------------------------------------------------------------------
The mechanism above is exact and tested. The DATES are the weak link, and the
module reports which it used on every call rather than letting the two be
confused:

    PUBLISHED   read from an operator-supplied calendar file (see
                `load_event_calendar`). Actual release timestamps.
    RULE        generated here from a release-schedule rule. NFP's rule is the
                real BLS convention (first Friday, 08:30 ET) and is right
                nearly always. CPI, PPI and FOMC have NO closed-form rule -
                their anchors below reproduce the right WEEK and frequently
                the wrong DAY.

A 30-minute window on the wrong day is worse than no filter: it removes a
random half hour and leaves the event itself untouched, while the report says
the run was news-filtered. So `RULE` provenance is carried into
`stats["entry_filters"]`, onto the scorecard and into every gate audit that
sees it, and `is_news_blocked` refuses to run at all on a span its calendar
does not cover rather than returning an all-False mask that reads as "no events
in this period".

Before a news-filtered run backs a promotion decision, drop a real calendar in:

    $BT_MACRO_CALENDAR, else /mnt/backtest/reference/macro/us_macro_events.csv

    event,ts_utc,source
    FOMC,2024-01-31T19:00:00Z,federalreserve.gov
    CPI,2024-01-11T13:30:00Z,bls.gov
    NFP,2024-02-02T13:30:00Z,bls.gov

`event` is one of FOMC/CPI/NFP/PPI, `ts_utc` is the release instant (an ET
local time is accepted too, see `load_event_calendar`), `source` is free text.

    python3 backtest/event_calendar.py --start 2015-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# The zone every US macro release is scheduled in. Never a fixed UTC offset:
# 08:30 ET is 13:30 UTC in winter and 12:30 UTC in summer, and a hard-coded
# offset is right for one half of the year and an hour wrong for the other.
ET = "America/New_York"

EVENT_KINDS: tuple[str, ...] = ("FOMC", "CPI", "NFP", "PPI")

# Provenance tokens. These travel with every mask this module produces.
PUBLISHED = "PUBLISHED"
RULE = "RULE"
MIXED = "MIXED"

CALENDAR_ENV = "BT_MACRO_CALENDAR"
DEFAULT_CALENDAR = Path("/mnt/backtest/reference/macro/us_macro_events.csv")

# Scheduled release time, local to New York.
RELEASE_TIME_ET: dict[str, time] = {
    "CPI": time(8, 30),
    "PPI": time(8, 30),
    "NFP": time(8, 30),
    # The statement, not the start of the meeting. The two-day meeting's second
    # day is what moves the market.
    "FOMC": time(14, 0),
}

# Day-of-month anchors for the rule-generated calendar. Read the module
# docstring before trusting any of these: only NFP has an actual published
# convention behind it.
#
#   CPI  BLS releases mid-month, typically the 10th-13th. Anchor 12.
#   PPI  usually a day or two after CPI in recent years. Anchor 14.
#   Both are rolled FORWARD off a weekend, never backward: a release that would
#   fall on a Saturday goes to the following Monday.
_ANCHOR_DAY = {"CPI": 12, "PPI": 14}

# FOMC meets eight times a year on a schedule the Fed publishes years ahead and
# which follows no arithmetic rule at all. These anchors put a meeting in the
# right month and, snapped to the nearest Wednesday, usually the right week.
_FOMC_ANCHORS = {1: 29, 3: 19, 4: 30, 6: 15, 7: 30, 9: 18, 11: 5, 12: 15}

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class CalendarError(RuntimeError):
    """
    The calendar cannot answer the question it was asked.

    Raised rather than returning an empty mask. An all-False block mask and a
    calendar that covers none of the requested span are indistinguishable
    downstream, and the second one silently turns `--news-filter` into a flag
    that prints on the report and does nothing.
    """


@dataclass(frozen=True)
class MacroEvent:
    """One scheduled release."""

    kind: str
    ts_utc: pd.Timestamp
    provenance: str
    source: str = ""


# --------------------------------------------------------------------------
# Timestamp handling
# --------------------------------------------------------------------------
def _as_utc_index(timestamps) -> pd.DatetimeIndex:
    """
    Anything the engine holds bar timestamps in -> a tz-aware UTC index.

    Naive input is treated as UTC rather than as local time. Every timestamp in
    this repo's lake is UTC (`mdlib.lake` guarantees it), and localizing a naive
    index to the machine's zone would shift a whole backtest by the operator's
    offset with nothing raising.
    """
    if isinstance(timestamps, pd.DataFrame):
        if "ts" not in timestamps.columns:
            raise CalendarError(
                "a bars frame was passed with no 'ts' column; "
                f"got {list(timestamps.columns)}")
        timestamps = timestamps["ts"]

    idx = pd.DatetimeIndex(pd.Series(timestamps).values
                           if isinstance(timestamps, pd.Series)
                           else timestamps)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def session_date(timestamps) -> pd.DatetimeIndex:
    """
    The CME SESSION date each bar belongs to, as a tz-naive daily index.

    A futures week does not start at midnight. CME opens Sunday 18:00 ET and
    each session runs to 17:00 ET the next day, so a bar stamped Sunday 23:00
    ET belongs to Monday's session and a "Friday" strategy that keyed off the
    calendar date would be reading two different things on either side of
    18:00.

    The rule is one line - any bar at or after 18:00 ET rolls to the next
    calendar day - and it is correct for daily bars as well as intraday ones,
    which is not obvious and is worth the check in the test suite. A 1d bar is
    stamped at UTC midnight; in ET that is 19:00 or 20:00 on the PREVIOUS
    calendar day, which is past 18:00, so the roll puts it back onto its own
    date in both EST and EDT.
    """
    et = _as_utc_index(timestamps).tz_convert(ET)
    rolled = et.normalize() + pd.to_timedelta((et.hour >= 18).astype("int64"),
                                              unit="D")
    return pd.DatetimeIndex(rolled.tz_localize(None))


def session_weekday(timestamps) -> np.ndarray:
    """Monday=0 ... Sunday=6, on the session date rather than the UTC date."""
    return session_date(timestamps).dayofweek.to_numpy()


# --------------------------------------------------------------------------
# The event calendar
# --------------------------------------------------------------------------
def calendar_path() -> Path:
    """Where a published calendar would be read from, if one exists."""
    return Path(os.environ.get(CALENDAR_ENV) or DEFAULT_CALENDAR)


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _roll_to_weekday(d: date) -> date:
    """Saturday and Sunday roll FORWARD to Monday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _nearest_wednesday(d: date) -> date:
    """The Wednesday closest to `d`, ties going forward."""
    offset = (2 - d.weekday()) % 7          # days forward to the next Wednesday
    forward, backward = offset, offset - 7
    return d + timedelta(days=forward if abs(forward) <= abs(backward)
                         else backward)


def _et_instant(d: date, kind: str) -> pd.Timestamp:
    """A local ET release date+time as a UTC instant, DST handled by the zone."""
    naive = pd.Timestamp(datetime.combine(d, RELEASE_TIME_ET[kind]))
    return naive.tz_localize(ET).tz_convert("UTC")


def schedule_rule_events(start, end,
                         kinds: Sequence[str] | None = None) -> list[MacroEvent]:
    """
    Generate the release calendar from schedule rules. APPROXIMATE - see below.

    NFP is the only kind here whose rule is the real convention: the BLS
    Employment Situation is released on the first Friday of the month at 08:30
    ET, and the exceptions are rare enough to name (a first Friday falling on
    the 1st occasionally pushes to the second Friday, and a federal holiday
    moves it).

    CPI, PPI and FOMC have no closed-form schedule. Their anchors put an event
    in the right month and usually the right week, and frequently on the wrong
    DAY - which for a 30-minute window means blocking a half hour where nothing
    happened while leaving the release itself tradeable. Every event returned
    here is stamped `provenance=RULE` so that fact travels with the mask instead
    of being rediscovered by whoever reads the equity curve.
    """
    kinds = tuple(kinds or EVENT_KINDS)
    unknown = set(kinds) - set(EVENT_KINDS)
    if unknown:
        raise CalendarError(f"unknown event kind(s): {sorted(unknown)}; "
                            f"known kinds are {list(EVENT_KINDS)}")

    lo = _utc_stamp(start).tz_localize(None)
    hi = _utc_stamp(end).tz_localize(None)

    out: list[MacroEvent] = []
    # A month either side of the span, so an event whose window overlaps the
    # first or last bar is not lost to the month boundary.
    cursor = date(lo.year, lo.month, 1) - timedelta(days=31)
    stop = date(hi.year, hi.month, 1) + timedelta(days=62)
    while cursor <= stop:
        y, m = cursor.year, cursor.month
        if "NFP" in kinds:
            out.append(MacroEvent("NFP", _et_instant(_first_friday(y, m), "NFP"),
                                  RULE, "rule: first Friday 08:30 ET"))
        for kind in ("CPI", "PPI"):
            if kind in kinds:
                d = _roll_to_weekday(date(y, m, _ANCHOR_DAY[kind]))
                out.append(MacroEvent(kind, _et_instant(d, kind), RULE,
                                      f"rule: ~day {_ANCHOR_DAY[kind]} 08:30 ET"))
        if "FOMC" in kinds and m in _FOMC_ANCHORS:
            d = _nearest_wednesday(date(y, m, _FOMC_ANCHORS[m]))
            out.append(MacroEvent("FOMC", _et_instant(d, "FOMC"), RULE,
                                  "rule: anchored Wednesday 14:00 ET"))
        cursor = date(y + (m == 12), (m % 12) + 1, 1)

    out.sort(key=lambda e: e.ts_utc)
    return out


def read_calendar_file(path: str | Path) -> list[MacroEvent]:
    """
    Read a published calendar. Every row is `PUBLISHED`.

    Columns: `event`, `ts_utc` (or `ts`/`timestamp`), optional `source`. A
    timestamp with no zone is read as ET local time rather than UTC, because a
    hand-maintained release calendar is written in the zone the release is
    announced in; `ts_utc` values carrying an explicit offset or a trailing Z
    are taken at face value.

    A malformed row raises. Skipping it would quietly shrink the calendar, and
    a filter that blocks fewer events than it claims to is the failure mode
    this whole module is arranged against.
    """
    path = Path(path)
    if not path.exists():
        raise CalendarError(f"no calendar file at {path}")

    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    kind_col = cols.get("event") or cols.get("kind")
    ts_col = cols.get("ts_utc") or cols.get("ts") or cols.get("timestamp")
    if kind_col is None or ts_col is None:
        raise CalendarError(
            f"{path} needs an 'event' column and a 'ts_utc' column; "
            f"found {list(df.columns)}")

    # Parsed one row at a time rather than as a column. A hand-maintained
    # calendar routinely mixes `2024-01-31T19:00:00Z` with `2024-01-11 08:30`,
    # and pandas refuses a column of mixed offsets outright - `utc=True` would
    # accept it by reading the naive rows as UTC, which is five hours wrong for
    # every ET release. A few hundred rows costs nothing to loop over.
    stamps = []
    for i, value in enumerate(df[ts_col]):
        try:
            ts = pd.Timestamp(value)
        except Exception as e:                                    # noqa: BLE001
            raise CalendarError(
                f"{path} row {i + 2}: unparseable timestamp {value!r}") from e
        if pd.isna(ts):
            raise CalendarError(f"{path} row {i + 2}: unparseable timestamp")
        stamps.append(ts.tz_localize(ET).tz_convert("UTC") if ts.tz is None
                      else ts.tz_convert("UTC"))

    src_col = cols.get("source")
    events: list[MacroEvent] = []
    for i, (kind, ts) in enumerate(zip(df[kind_col], stamps)):
        k = str(kind).strip().upper()
        if k not in EVENT_KINDS:
            raise CalendarError(
                f"{path} row {i + 2}: unknown event {kind!r}; "
                f"known kinds are {list(EVENT_KINDS)}")
        if pd.isna(ts):
            raise CalendarError(f"{path} row {i + 2}: unparseable timestamp")
        events.append(MacroEvent(k, pd.Timestamp(ts), PUBLISHED,
                                 str(df[src_col].iloc[i]) if src_col else str(path)))
    events.sort(key=lambda e: e.ts_utc)
    return events


def load_event_calendar(start=None, end=None,
                        kinds: Sequence[str] | None = None,
                        path: str | Path | None = None,
                        allow_rule: bool = True) -> list[MacroEvent]:
    """
    The calendar to filter against: the published file if there is one, else
    the rule-generated schedule.

    `allow_rule=False` makes a missing file an error instead of a silent
    downgrade to approximate dates - which is what a run backing a promotion
    decision should pass.
    """
    kinds = tuple(kinds or EVENT_KINDS)
    p = Path(path) if path is not None else calendar_path()
    if p.exists():
        events = [e for e in read_calendar_file(p) if e.kind in kinds]
        if events:
            return _clip(events, start, end)
        if not allow_rule:
            raise CalendarError(
                f"{p} carries no {list(kinds)} events, and allow_rule=False.")

    if not allow_rule:
        raise CalendarError(
            f"no published macro calendar at {p} and allow_rule=False. Write "
            f"one (event,ts_utc,source) or pass allow_rule=True to fall back "
            f"to APPROXIMATE rule-generated dates.")
    if start is None or end is None:
        raise CalendarError(
            "the rule-generated calendar needs a start and an end; there is no "
            "such thing as every event that ever happened.")
    return _clip(schedule_rule_events(start, end, kinds), start, end)


def _utc_stamp(value) -> pd.Timestamp | None:
    """A scalar of any of the shapes callers pass, as a UTC Timestamp."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _clip(events: list[MacroEvent], start, end) -> list[MacroEvent]:
    if start is None and end is None:
        return events
    lo, hi = _utc_stamp(start), _utc_stamp(end)
    return [e for e in events
            if (lo is None or e.ts_utc >= lo - pd.Timedelta(days=1))
            and (hi is None or e.ts_utc <= hi + pd.Timedelta(days=1))]


def provenance_of(events: Iterable[MacroEvent]) -> str:
    """PUBLISHED, RULE, or MIXED - never a guess about which."""
    seen = {e.provenance for e in events}
    if not seen:
        return "EMPTY"
    return seen.pop() if len(seen) == 1 else MIXED


# --------------------------------------------------------------------------
# The masks
# --------------------------------------------------------------------------
def _merged_windows(events: Sequence[MacroEvent],
                    window: pd.Timedelta) -> tuple[np.ndarray, np.ndarray]:
    """
    Event instants -> merged [start, end] interval arrays, in ns since epoch.

    Merging matters for the lookup below: `is_news_blocked` finds the LAST
    interval starting at or before a bar and tests that one interval's end,
    which is only sufficient when no interval is contained in another. CPI and
    PPI 30 minutes apart would otherwise leave a hole between them.
    """
    if not events:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

    stamps = pd.DatetimeIndex([e.ts_utc for e in events]).sort_values()
    starts = (stamps - window).asi8
    ends = (stamps + window).asi8

    m_starts, m_ends = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= m_ends[-1]:
            m_ends[-1] = max(m_ends[-1], e)
        else:
            m_starts.append(s)
            m_ends.append(e)
    return np.asarray(m_starts, dtype="int64"), np.asarray(m_ends, dtype="int64")


def is_news_blocked(timestamps,
                    window_minutes: float = 30,
                    kinds: Sequence[str] | None = None,
                    events: Sequence[MacroEvent] | None = None,
                    path: str | Path | None = None,
                    allow_rule: bool = True,
                    strict: bool = True) -> np.ndarray:
    """
    True for every bar inside +/- `window_minutes` of a scheduled US macro event.

    This is a mask over BARS. It is not the mask the engine applies to signals -
    see `entry_block_mask`, which widens it by one bar to account for next-bar-
    open fills.

    Parameters
    ----------
    timestamps
        Bar timestamps: a Series, a DatetimeIndex, or a bars frame with a `ts`
        column. Naive values are read as UTC.
    window_minutes
        Half-width. 30 blocks the half hour either side of the release.
    kinds
        Which events to block. Default: FOMC, CPI, NFP, PPI.
    events
        A pre-loaded calendar, to avoid re-reading it per symbol.
    strict
        Raise when the calendar has no event inside the span of `timestamps`.
        A run over 2015-2022 with an empty calendar would otherwise report a
        news-filtered backtest in which nothing was ever filtered.

    The cost is O(n log m), not O(n*m): the intervals are merged and searched
    with `searchsorted`. A 16-year 1-minute symbol is 5.6M bars against ~700
    events, and the broadcast form of this comparison would allocate 4e9
    booleans.
    """
    idx = _as_utc_index(timestamps)
    n = len(idx)
    if n == 0:
        return np.zeros(0, dtype=bool)

    if events is None:
        events = load_event_calendar(idx[0], idx[-1], kinds=kinds, path=path,
                                     allow_rule=allow_rule)
    elif kinds is not None:
        events = [e for e in events if e.kind in set(kinds)]

    window = pd.Timedelta(minutes=float(window_minutes))
    if window < pd.Timedelta(0):
        raise CalendarError(f"window_minutes must be >= 0, got {window_minutes}")

    span_lo, span_hi = idx[0] - window, idx[-1] + window
    in_span = [e for e in events if span_lo <= e.ts_utc <= span_hi]
    if not in_span:
        if strict:
            raise CalendarError(
                f"the macro calendar holds no {list(kinds or EVENT_KINDS)} "
                f"event between {idx[0]} and {idx[-1]}. Refusing to return an "
                f"all-clear mask: a news filter that blocks nothing is "
                f"indistinguishable on the report from one that had nothing to "
                f"block. Supply a calendar covering this span "
                f"(${CALENDAR_ENV} / {DEFAULT_CALENDAR}), or pass strict=False "
                f"to accept that this period genuinely holds no events.")
        return np.zeros(n, dtype=bool)

    starts, ends = _merged_windows(in_span, window)
    bar_ns = idx.asi8
    # The last interval starting at or before each bar. Intervals are merged,
    # so if that one does not contain the bar, none does.
    pos = np.searchsorted(starts, bar_ns, side="right") - 1
    blocked = np.zeros(n, dtype=bool)
    hit = pos >= 0
    blocked[hit] = bar_ns[hit] <= ends[pos[hit]]
    return blocked


def weekday_blocked(timestamps,
                    exclude_days: Iterable[int] | None) -> np.ndarray:
    """
    True for every bar whose SESSION weekday is in `exclude_days`.

    Monday=0 through Sunday=6, matching `pandas.Timestamp.dayofweek` and
    `datetime.weekday()`. `[0, 4]` excludes Mondays and Fridays.

    Attribution is by session date, not UTC date - see `session_date`. On 15m
    bars the two disagree for every bar between 18:00 and 24:00 ET, which is
    the entire Globex evening: a "no Friday trades" rule keyed on the UTC date
    would keep trading through Thursday evening's Friday session and stop at
    Friday 18:00 ET when the Monday session had already begun.
    """
    idx = _as_utc_index(timestamps)
    days = set(int(d) for d in (exclude_days or ()))
    if not days:
        return np.zeros(len(idx), dtype=bool)
    bad = days - set(range(7))
    if bad:
        raise CalendarError(
            f"exclude_days takes weekday integers 0-6 (Mon-Sun); got {sorted(bad)}")
    return np.isin(session_weekday(idx), sorted(days))


def _widen_to_fill_bar(blocked: np.ndarray) -> np.ndarray:
    """
    A bar mask -> the signal mask that keeps the FILL out of the blocked window.

    The engine fills at the next bar's open, so a signal on bar `i` becomes a
    position on bar `i+1`. Blocking only the bars inside the window therefore
    lets exactly one entry per event through - the one signalled on the bar
    before the window opens - which is both the least visible outcome and
    precisely the trade the filter exists to stop.

    Widening backwards is causal: it needs the scheduled event times and the
    bar timestamps, both known before the bar trades. Nothing here reads a
    price.
    """
    if blocked.size == 0:
        return blocked
    out = blocked.copy()
    out[:-1] |= blocked[1:]
    return out


def entry_block_mask(timestamps,
                     news_filter: bool = False,
                     news_window_minutes: float = 30,
                     news_kinds: Sequence[str] | None = None,
                     exclude_days: Iterable[int] | None = None,
                     events: Sequence[MacroEvent] | None = None,
                     calendar_path_: str | Path | None = None,
                     allow_rule: bool = True,
                     strict: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """
    The combined entry-suppression mask, plus what it did.

    Returns `(mask, info)`. `info` is what gets recorded on the run, and it
    names the calendar's provenance: a report that says "news filter: on"
    without saying whether the dates were published or rule-generated invites
    the reading that a rule-generated run blocked the actual releases.
    """
    idx = _as_utc_index(timestamps)
    n = len(idx)
    mask = np.zeros(n, dtype=bool)
    info: dict[str, Any] = {
        "news_filter": bool(news_filter),
        "exclude_days": sorted(int(d) for d in (exclude_days or ())),
        "bars": int(n),
    }

    if news_filter:
        if events is None:
            events = load_event_calendar(
                idx[0] if n else None, idx[-1] if n else None,
                kinds=news_kinds, path=calendar_path_, allow_rule=allow_rule)
        news = is_news_blocked(idx, news_window_minutes, kinds=news_kinds,
                               events=events, strict=strict)
        widened = _widen_to_fill_bar(news)
        mask |= widened
        used = [e for e in events
                if n and idx[0] <= e.ts_utc <= idx[-1]]
        info.update({
            "news_window_minutes": float(news_window_minutes),
            "news_kinds": list(news_kinds or EVENT_KINDS),
            "news_events_in_span": len(used),
            "news_provenance": provenance_of(used),
            "news_bars_blocked": int(news.sum()),
            "news_signal_bars_blocked": int(widened.sum()),
        })

    if info["exclude_days"]:
        dow = _widen_to_fill_bar(weekday_blocked(idx, info["exclude_days"]))
        mask |= dow
        info["dow_signal_bars_blocked"] = int(dow.sum())
        info["exclude_days_named"] = [WEEKDAY_NAMES[d] for d in info["exclude_days"]]

    info["signal_bars_blocked"] = int(mask.sum())
    return mask, info


def apply_entry_filters(timestamps,
                        entries,
                        short_entries=None,
                        **kwargs) -> tuple[Any, Any, dict[str, Any]]:
    """
    Suppress blocked ENTRIES on both sides. Exits are not accepted, by design.

    Returns `(entries, short_entries, info)` with the same types it was handed,
    so a strategy module can call this on its own masks and the engine can call
    it on the unpacked ones. `info` additionally reports how many entries each
    side actually lost, which is the number worth printing: a mask covering 4%
    of bars can remove 40% of entries or none of them, and only the second
    number says whether the filter changed the strategy.
    """
    mask, info = entry_block_mask(timestamps, **kwargs)

    def _suppress(side, label: str):
        if side is None:
            return None
        arr = np.asarray(pd.Series(side).fillna(False).astype(bool).values)
        before = int(arr.sum())
        kept = arr & ~mask
        info[f"{label}_before"] = before
        info[f"{label}_suppressed"] = before - int(kept.sum())
        if isinstance(side, pd.Series):
            return pd.Series(kept, index=side.index, name=side.name)
        return kept

    e = _suppress(entries, "long_entries")
    se = _suppress(short_entries, "short_entries")
    info["entries_suppressed"] = (info.get("long_entries_suppressed", 0)
                                  + info.get("short_entries_suppressed", 0))
    return e, se, info


def describe_filters(info: dict[str, Any] | None) -> str:
    """One console line per filter, or a line saying neither ran."""
    if not info:
        return "  entry filters : none"
    lines = []
    if info.get("news_filter"):
        prov = info.get("news_provenance", "?")
        warn = ("  [!] APPROXIMATE DATES - see backtest/event_calendar.py"
                if prov in (RULE, MIXED) else "")
        lines.append(
            f"  news filter   : ON  +/-{info.get('news_window_minutes', 0):g}m "
            f"around {', '.join(info.get('news_kinds', []))}  "
            f"({info.get('news_events_in_span', 0)} events, {prov}){warn}")
    if info.get("exclude_days"):
        lines.append(
            f"  day filter    : excluding "
            f"{', '.join(info.get('exclude_days_named', []))} sessions")
    if not lines:
        return "  entry filters : none"
    # Only after `apply_entry_filters` has seen actual signals. The bar mask on
    # its own cannot say how many entries it removed, and printing a 0 there
    # would read as a filter that changed nothing.
    if "entries_suppressed" in info:
        offered = (info.get("long_entries_before", 0)
                   + info.get("short_entries_before", 0))
        lines.append(f"  entries cut   : {info['entries_suppressed']:,} "
                     f"of {offered:,}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The CLI surface, defined once
# --------------------------------------------------------------------------
def add_filter_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Attach `--news-filter`, `--news-window`, `--news-kinds` and `--exclude-days`.

    Defined here rather than in each of the five stages so the flags cannot
    drift apart. A `--exclude-days` that means Monday=0 in Stage 1 and Sunday=0
    in Stage 3 would move every trade by a day between two runs that both claim
    to have excluded Mondays.
    """
    g = parser.add_argument_group("entry filters (backtest/event_calendar.py)")
    g.add_argument("--news-filter", action="store_true",
                   help="Block entries within --news-window of a scheduled US "
                        "macro release. Falls back to APPROXIMATE dates unless "
                        f"a calendar exists at ${CALENDAR_ENV} / "
                        f"{DEFAULT_CALENDAR}")
    g.add_argument("--news-window", type=float, default=30.0, metavar="MIN",
                   help="Half-width of the block in minutes (default 30)")
    g.add_argument("--news-kinds", default=",".join(EVENT_KINDS),
                   help=f"Events to block (default {','.join(EVENT_KINDS)})")
    g.add_argument("--exclude-days", default=None, metavar="D,D",
                   help="Suppress entries on these SESSION weekdays. "
                        "Monday=0 .. Sunday=6, so '0,4' is Mon and Fri.")
    return parser


def parse_exclude_days(text: str | None) -> tuple[int, ...] | None:
    """`"0,4"` -> `(0, 4)`. A malformed value raises rather than being ignored."""
    if text is None or not str(text).strip():
        return None
    days = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            d = int(part)
        except ValueError as e:
            raise CalendarError(
                f"--exclude-days takes weekday integers 0-6 (Mon-Sun); "
                f"got {part!r}") from e
        if not 0 <= d <= 6:
            raise CalendarError(
                f"--exclude-days takes weekday integers 0-6 (Mon-Sun); got {d}")
        days.append(d)
    return tuple(sorted(set(days))) or None


def filter_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Parsed flags -> the `BacktestConfig` fields the engine reads."""
    kinds = tuple(k.strip().upper()
                  for k in str(getattr(args, "news_kinds", "") or "").split(",")
                  if k.strip())
    unknown = set(kinds) - set(EVENT_KINDS)
    if unknown:
        raise CalendarError(f"unknown event kind(s): {sorted(unknown)}; "
                            f"known kinds are {list(EVENT_KINDS)}")
    return {
        "news_filter": bool(getattr(args, "news_filter", False)),
        "news_window_minutes": float(getattr(args, "news_window", 30.0)),
        "news_kinds": kinds or None,
        "exclude_days": parse_exclude_days(getattr(args, "exclude_days", None)),
    }


# --------------------------------------------------------------------------
# CLI - what calendar would a run actually use?
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Inspect the macro calendar a news-filtered run would use.")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--kinds", default=",".join(EVENT_KINDS))
    p.add_argument("--path", default=None, help="Calendar CSV to read")
    p.add_argument("--no-rule", action="store_true",
                   help="Fail instead of falling back to approximate dates")
    args = p.parse_args(argv)

    kinds = tuple(k.strip().upper() for k in args.kinds.split(",") if k.strip())
    try:
        events = load_event_calendar(args.start, args.end, kinds=kinds,
                                     path=args.path,
                                     allow_rule=not args.no_rule)
    except CalendarError as e:
        print(f"CalendarError: {e}", file=sys.stderr)
        return 1

    prov = provenance_of(events)
    src = Path(args.path) if args.path else calendar_path()
    print(f"\nmacro calendar  {args.start} → {args.end}")
    print(f"  file        : {src}{'' if src.exists() else '  (absent)'}")
    print(f"  provenance  : {prov}")
    print(f"  events      : {len(events):,}")
    counts = pd.Series([e.kind for e in events]).value_counts() if events \
        else pd.Series(dtype=int)
    for kind in kinds:
        print(f"    {kind:<5} {int(counts.get(kind, 0)):>5}")

    if prov in (RULE, MIXED):
        print("\n  [!] RULE-generated dates are APPROXIMATE. NFP's first-Friday")
        print("      rule is the real convention; CPI, PPI and FOMC anchors put")
        print("      an event in the right week and often the wrong day, which")
        print("      for a 30-minute window means blocking the wrong half hour.")
        print(f"      Write a published calendar to {DEFAULT_CALENDAR}")
        print(f"      (or ${CALENDAR_ENV}) before this filter backs a promotion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
