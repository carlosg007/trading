#!/usr/bin/env python3
"""
test_live_feed.py — the bar feed seam: that a forming bar never reaches a
strategy, that live bars are shaped and aggregated exactly like lake bars, and
that asking for a live feed that is not configured refuses rather than quietly
handing back history.

Location:  ~/src/trading/tests/test_live_feed.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_live_feed.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_live_feed.py

EVERY CASE FAILS THROUGH `assert`. Nothing here touches the network: the live
path is exercised against a fake client that returns bars this file wrote, so
the cases run on a plane and still test the thing that matters.

WHAT THIS COVERS, and why each case is here rather than assumed:

  * **THE FORMING BAR IS THE WHOLE POINT.** A bar is stamped when it OPENS, so
    the 14:00 bar on an hourly feed is unfinished until 15:00. Acting on it is
    lookahead — a decision made with information from inside the interval
    being traded — and it is the failure that makes a live curve diverge from
    its backtest for reasons nobody can find. The boundary is tested from both
    sides, to the second.
  * **MORE THAN ONE BAR CAN BE UNFINISHED.** After a gap or a reconnect a feed
    can hand back several open intervals; dropping only the last would leave a
    half-formed bar in place looking complete.
  * **LIVE BARS ARE BUILT LIKE LAKE BARS.** The aggregation is the lake's own
    resampler, so a live 15m bar and a backtested 15m bar are the same object.
    A second implementation would differ in the last decimal, which is enough
    to move a bar across an indicator threshold.
  * **A SHORT FRAME RAISES.** A strategy handed bars with no `volume` does not
    fail — it computes a different signal, quietly.
  * **`--feed live` REFUSES when nothing is configured.** Falling back would
    hand an operator 18-day-old bars in a session they believe is live, and a
    stale tape reads exactly like a market with no signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.feed import (BAR_COLUMNS,                            # noqa: E402
                           FeedError,
                           LakeFeed,
                           LiveFeed,
                           drop_forming_bar,
                           normalize_bars,
                           resolve_feed,
                           source_tf,
                           tf_delta,
                           to_timeframe)

# One trading morning of 1-minute bars, deterministic and hand-checkable: the
# close walks up by 1.0 a minute from 100.0, so any aggregation's OHLC can be
# read straight off the arithmetic.
START = pd.Timestamp("2026-08-25 14:00:00", tz="UTC")


def minute_frame(n: int = 120, start: pd.Timestamp = START,
                 symbol: str = "NQ") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = pd.Series(range(n), dtype="float64") + 100.0
    return pd.DataFrame({
        "ts": ts,
        "symbol": symbol,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": pd.Series(range(n), dtype="int64") + 1,
    })


class FakeClient:
    """A bar client with no network: it returns what the case wrote."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, int]] = []

    def minute_bars(self, symbol: str, minutes: int) -> pd.DataFrame:
        self.calls.append((symbol, minutes))
        frame = self.frames.get(str(symbol).upper())
        return pd.DataFrame() if frame is None else frame.tail(minutes)


# --------------------------------------------------------------------------
# 1. the forming bar
# --------------------------------------------------------------------------

def test_a_bar_is_closed_only_once_its_interval_has_elapsed() -> None:
    """
    The boundary, from both sides, to the second.

    A bar stamped 14:00 on an hourly feed covers 14:00–15:00. At 14:59:59 it is
    still filling: its close is the current price and its high and low are
    whatever has printed so far. At 15:00:00 exactly it is finished. Trading
    the first is lookahead; refusing the second would throw away the newest
    fact available.
    """
    bars = pd.DataFrame({"ts": pd.to_datetime(
        ["2026-08-25 13:00", "2026-08-25 14:00"], utc=True)})

    still_open, dropped = drop_forming_bar(
        bars, "1h", now=pd.Timestamp("2026-08-25 14:59:59", tz="UTC"))
    assert list(still_open["ts"].astype(str)) == ["2026-08-25 13:00:00+00:00"]
    assert dropped == pd.Timestamp("2026-08-25 14:00:00", tz="UTC")

    closed, none_dropped = drop_forming_bar(
        bars, "1h", now=pd.Timestamp("2026-08-25 15:00:00", tz="UTC"))
    assert len(closed) == 2
    assert none_dropped is None


def test_every_unfinished_bar_is_dropped_not_only_the_last() -> None:
    """
    After a gap or a reconnect a feed can return several open intervals.
    Dropping only the tail leaves the one before it in place, looking complete
    — and it is the bar the strategy would then act on.
    """
    bars = pd.DataFrame({"ts": pd.to_datetime(
        ["2026-08-25 12:00", "2026-08-25 13:00",
         "2026-08-25 14:00", "2026-08-25 15:00"], utc=True)})
    closed, dropped = drop_forming_bar(
        bars, "1h", now=pd.Timestamp("2026-08-25 14:30", tz="UTC"))

    assert list(closed["ts"].astype(str)) == ["2026-08-25 12:00:00+00:00",
                                              "2026-08-25 13:00:00+00:00"]
    assert dropped == pd.Timestamp("2026-08-25 15:00:00", tz="UTC")


def test_the_rule_scales_with_the_timeframe() -> None:
    """The same bar is closed at 15m and forming at 1h. The width comes from
    the lake's own table, so a timeframe this repository cannot build is one
    the feed refuses rather than invents a width for."""
    bars = pd.DataFrame({"ts": pd.to_datetime(["2026-08-25 14:00"], utc=True)})
    now = pd.Timestamp("2026-08-25 14:20", tz="UTC")

    assert len(drop_forming_bar(bars, "15m", now=now)[0]) == 1
    assert len(drop_forming_bar(bars, "1h", now=now)[0]) == 0

    assert tf_delta("15m") == pd.Timedelta("15min")
    assert tf_delta("1h") == pd.Timedelta("1h")
    assert source_tf("15m") == "1m"
    with pytest.raises(FeedError, match="unknown timeframe"):
        tf_delta("7m")


def test_a_naive_timestamp_is_read_as_utc_not_local() -> None:
    """A feed that stamped bars without a zone would otherwise shift the
    boundary by the operator's offset — and be correct in London and wrong in
    New York, on the same code."""
    bars = pd.DataFrame({"ts": pd.to_datetime(["2026-08-25 14:00"], utc=True)})
    closed, _ = drop_forming_bar(bars, "1h",
                                 now=pd.Timestamp("2026-08-25 15:00"))
    assert len(closed) == 1


# --------------------------------------------------------------------------
# 2. the frame contract
# --------------------------------------------------------------------------

def test_a_vendor_frame_becomes_the_canonical_one() -> None:
    """`ts_event` and mixed case in, the lake's own columns out — so nothing
    downstream can tell which feed produced a frame."""
    vendor = pd.DataFrame({
        "ts_event": ["2026-08-25T14:00:00Z", "2026-08-25T14:01:00Z"],
        "Open": [100.0, 101.0], "High": [102.0, 103.0],
        "Low": [99.0, 100.0], "Close": [101.0, 102.0],
        "Volume": [10, 20], "rtype": [34, 34]})
    out = normalize_bars(vendor, "NQ")

    assert list(out.columns) == list(BAR_COLUMNS)
    assert str(out["ts"].dt.tz) == "UTC"
    assert out["symbol"].tolist() == ["NQ", "NQ"]
    assert out["close"].dtype == "float64"


def test_a_frame_missing_a_column_raises_rather_than_arriving_short() -> None:
    """A strategy handed bars with no `volume` does not fail; it computes a
    different signal and reports it as though nothing were wrong."""
    with pytest.raises(FeedError, match="volume"):
        normalize_bars(pd.DataFrame({
            "ts": ["2026-08-25T14:00:00Z"], "open": [1.0], "high": [2.0],
            "low": [0.5], "close": [1.5]}), "NQ")


def test_duplicate_timestamps_collapse_to_the_last_one() -> None:
    """A top-up that overlaps what is already held must not double a bar —
    a repeated 14:00 would be counted twice by every indicator."""
    frame = pd.concat([minute_frame(3), minute_frame(3)], ignore_index=True)
    out = normalize_bars(frame, "NQ")
    assert len(out) == 3
    assert out["ts"].is_monotonic_increasing


# --------------------------------------------------------------------------
# 3. aggregation
# --------------------------------------------------------------------------

def test_minutes_aggregate_the_way_the_lake_aggregates() -> None:
    """
    Left-labelled, left-closed, through `mdlib.lake`'s own resampler.

    The 14:00 15-minute bar opens on 14:00's open and closes on 14:14's close.
    Getting this wrong by one bar is invisible: every frame is still populated
    and every indicator still returns a number.
    """
    minutes = minute_frame(60)
    out = to_timeframe(minutes, "15m")

    assert len(out) == 4
    assert out["ts"].iloc[0] == pd.Timestamp("2026-08-25 14:00", tz="UTC")
    assert out["open"].iloc[0] == minutes["open"].iloc[0]
    assert out["close"].iloc[0] == minutes["close"].iloc[14]
    assert out["high"].iloc[0] == minutes["high"].iloc[:15].max()
    assert out["low"].iloc[0] == minutes["low"].iloc[:15].min()
    assert out["volume"].iloc[0] == minutes["volume"].iloc[:15].sum()


def test_a_native_timeframe_is_not_resampled() -> None:
    minutes = minute_frame(10)
    assert to_timeframe(minutes, "1m").equals(minutes.reset_index(drop=True))


# --------------------------------------------------------------------------
# 4. the live feed end to end
# --------------------------------------------------------------------------

def test_the_live_feed_returns_closed_bars_only(monkeypatch) -> None:
    """
    The whole path: vendor frame -> normalize -> aggregate -> drop the forming
    bar. 100 minutes from 14:00 reaches 15:39, so at 15:44 the 15:30 bar is
    still filling and the newest CLOSED 15m bar is 15:15 — the case fails if
    the clock is not the one it pinned, because at any later time 15:30
    closes.
    """
    client = FakeClient({"NQ": minute_frame(100)})
    feed = LiveFeed(client, label="fake")

    import realtime.feed as feed_module
    monkeypatch.setattr(
        feed_module, "datetime",
        type("D", (), {"now": staticmethod(
            lambda tz=None: pd.Timestamp("2026-08-25 15:44", tz="UTC"))}))

    bars, sources = feed.closed_bars(["MNQ"], "15m", lookback_bars=10)

    assert sources == {"MNQ": "NQ"}, "the micro's bars come from its parent"
    frame = bars["MNQ"]
    assert frame["ts"].iloc[-1] == pd.Timestamp("2026-08-25 15:15", tz="UTC")
    assert list(frame.columns) == list(BAR_COLUMNS)
    # The order is for the micro; only the tape is the parent's.
    assert set(frame["symbol"]) == {"MNQ"}
    assert feed.last_forming["MNQ"] == pd.Timestamp("2026-08-25 15:30",
                                                    tz="UTC")


def test_one_vendor_call_serves_every_micro_of_one_parent(monkeypatch) -> None:
    """MNQ and NQ are one tape. Fetching it twice would double the bill and
    could return two different frames across a publish boundary."""
    client = FakeClient({"NQ": minute_frame(100)})
    feed = LiveFeed(client, label="fake")
    import realtime.feed as feed_module
    monkeypatch.setattr(
        feed_module, "datetime",
        type("D", (), {"now": staticmethod(
            lambda tz=None: pd.Timestamp("2026-08-25 15:45", tz="UTC"))}))

    feed.closed_bars(["MNQ", "NQ"], "15m", lookback_bars=10)
    assert [c[0] for c in client.calls] == ["NQ"]


def test_a_lagging_feed_holds_back_a_half_published_bar(monkeypatch) -> None:
    """
    THE HAZARD THE CLOCK ALONE MISSES.

    The clock says 16:05 and the vendor has published only to 15:50. The 15:45
    quarter-hour is over by the clock, but the feed holds five minutes of it —
    aggregate that and the bar has the wrong high, the wrong low, the wrong
    close and a fifth of the volume, and it reads as a quiet quarter of an
    hour. Cutting at the feed's HORIZON produces fewer bars instead of wrong
    ones.
    """
    client = FakeClient({"NQ": minute_frame(110)})     # 14:00 -> 15:49
    client.horizon = lambda: pd.Timestamp("2026-08-25 15:50", tz="UTC")
    feed = LiveFeed(client, label="lagging")

    import realtime.feed as feed_module
    monkeypatch.setattr(
        feed_module, "datetime",
        type("D", (), {"now": staticmethod(
            lambda tz=None: pd.Timestamp("2026-08-25 16:05", tz="UTC"))}))

    bars, _ = feed.closed_bars(["NQ"], "15m", lookback_bars=10)
    assert bars["NQ"]["ts"].iloc[-1] == pd.Timestamp("2026-08-25 15:30",
                                                     tz="UTC")


def test_a_symbol_the_feed_cannot_answer_for_is_absent_not_empty() -> None:
    """"No bars for this symbol" and "this symbol has no signal" are different
    statements, and the loop reports them differently."""
    feed = LiveFeed(FakeClient({"NQ": minute_frame(100)}), label="fake")
    bars, sources = feed.closed_bars(["MNQ", "MGC"], "15m", lookback_bars=5)
    assert "MGC" not in bars and "MGC" not in sources


def test_a_client_without_the_contract_is_refused_at_construction() -> None:
    with pytest.raises(FeedError, match="minute_bars"):
        LiveFeed(object(), label="not a client")


# --------------------------------------------------------------------------
# 5. choosing a feed
# --------------------------------------------------------------------------

def test_asking_for_live_without_one_refuses_rather_than_falling_back(
        monkeypatch) -> None:
    """
    The fallback is the dangerous direction. An operator who typed `--feed
    live` and silently got 18-day-old bars would read a rehearsal as a live
    session — and a stale tape produces no signals, which looks exactly like a
    quiet market.
    """
    import realtime.feed as feed_module
    monkeypatch.setattr(feed_module, "live_feed_available",
                        lambda: (False, "no API key"))

    with pytest.raises(FeedError, match="Refusing to fall back"):
        resolve_feed("live")
    assert isinstance(resolve_feed("auto"), LakeFeed)
    assert isinstance(resolve_feed("lake"), LakeFeed)


def test_an_unknown_feed_mode_is_refused() -> None:
    with pytest.raises(FeedError, match="--feed must be one of"):
        resolve_feed("streaming")


def test_the_lake_feed_still_answers_and_says_what_it_is() -> None:
    """The fallback has to keep working: it is what every rehearsal runs on,
    and it must describe itself as historical so nobody reads it as live."""
    feed = LakeFeed()
    assert "historical" in feed.describe()
    bars, sources = feed.closed_bars(["MNQ"], "1h", lookback_bars=50)
    assert sources["MNQ"] == "NQ"
    frame = bars["MNQ"]
    assert len(frame) == 50
    assert list(frame.columns)[:7] == list(BAR_COLUMNS)
    assert set(frame["symbol"]) == {"MNQ"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
