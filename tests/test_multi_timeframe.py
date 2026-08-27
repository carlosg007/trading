#!/usr/bin/env python3
"""
tests/test_multi_timeframe.py - bar-width resolution across the live loop.

Location: ~/src/trading/tests/test_multi_timeframe.py

    .venv/bin/python3 -m pytest tests/test_multi_timeframe.py -v

ASSERT-BASED so `tests/conftest.py` collects it case by case. Helpers are
named `_...`, because pytest collects any module-level `test_*` it can call —
including one whose only argument is defaulted — and `test_regime_profiler.py`
was bitten by exactly that.

THE DEFECT THIS FILE GUARDS
---------------------------
Until 2026-08-27 `master_live.py` read ONE timeframe and handed those bars to
every strategy in the roster, while `StrategyHandle` recorded only the
certified SYMBOL. A strategy swept, plateau-selected and Gate-R certified on
3m bars was therefore evaluated on 1h bars: real signals, correct log lines,
and a certification describing a different tape. It was caught during pre-live
arming, before any order went out.

The dispatch half is tested in `tests/test_live_dispatcher.py`. This file
covers the two pieces underneath it — measuring a frame's width, and the
resampling that produces the frames in the first place.

NO NEW RESAMPLER WAS WRITTEN, and that is the point of the integrity cases
here. `realtime.feed.to_timeframe` delegates to `mdlib.lake._resample`, the
implementation every backtest and every certification was computed with. A
second aggregation in the live path would be free to disagree with it
somewhere in the last decimal, which is the drift these tests exist to refuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

from realtime.feed import infer_timeframe, tf_delta, to_timeframe  # noqa: E402


# ==========================================================================
# fixtures
# ==========================================================================

def _minutes(n: int = 720, start: str = "2026-08-20 00:00") -> pd.DataFrame:
    """
    Deterministic 1-minute bars. No randomness: a resampling test whose input
    moves is one that will fail for a reason nobody can reproduce.
    """
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    step = np.arange(n, dtype=float)
    close = 15000.0 + step
    return pd.DataFrame({
        "ts": ts,
        # `mdlib.lake._resample` carries the symbol through, so the frame it
        # is handed must have one. Its absence is what a live frame never has.
        "symbol": "MNQ",
        "open": close - 0.5,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": np.full(n, 10.0),
    })


def _frame(freq: str, n: int = 60, gap_after: int | None = None) -> pd.DataFrame:
    ts = list(pd.date_range("2026-08-20", periods=n, freq=freq, tz="UTC"))
    if gap_after is not None:
        ts = ts[:gap_after] + [t + pd.Timedelta("6h") for t in ts[gap_after:]]
    c = 15000.0 + np.arange(len(ts), dtype=float)
    return pd.DataFrame({"ts": ts, "open": c, "high": c + 1, "low": c - 1,
                         "close": c, "volume": 10.0})


# ==========================================================================
# measuring a frame's width
# ==========================================================================

def test_every_ladder_timeframe_is_recognised():
    for freq, token in (("1min", "1m"), ("2min", "2m"), ("3min", "3m"),
                        ("5min", "5m"), ("15min", "15m"), ("30min", "30m"),
                        ("1h", "1h")):
        assert infer_timeframe(_frame(freq)) == token, freq


def test_the_mode_is_used_so_a_session_gap_does_not_move_the_answer():
    """
    THE MODE, NOT THE MEAN OR THE MEDIAN. A session break is a six-hour hole
    between two adjacent rows and a holiday is longer; both drag an average,
    and on a short frame both can drag a median. Every bar INSIDE a session
    sits one width from its neighbour, so the most common spacing is the bar
    width by construction.
    """
    hourly_with_break = _frame("1h", n=30, gap_after=15)
    assert infer_timeframe(hourly_with_break) == "1h"

    # A frame that is mostly gap still answers on its modal spacing.
    assert infer_timeframe(_frame("3min", n=40, gap_after=8)) == "3m"


def test_an_unmeasurable_frame_returns_none_rather_than_the_nearest_token():
    """
    Guessing defeats the purpose. An unrecognised width has to reach the
    caller as "unknown" so the guard can allow it through as unmeasured,
    rather than as a confident wrong token that would refuse a valid strategy.
    """
    assert infer_timeframe(_frame("7min")) is None, "7m is not buildable here"
    assert infer_timeframe(_frame("1h", n=1)) is None, "one row has no spacing"
    assert infer_timeframe(pd.DataFrame()) is None
    assert infer_timeframe(None) is None


def test_the_default_is_honoured_when_nothing_can_be_measured():
    assert infer_timeframe(pd.DataFrame(), default="1h") == "1h"
    assert infer_timeframe(_frame("1h"), default="3m") == "1h", \
        "a measurable frame beats the default"


def test_it_reads_an_index_as_well_as_a_ts_column():
    frame = _frame("15min").set_index("ts")
    assert infer_timeframe(frame) == "15m"


def test_infer_is_the_inverse_of_tf_delta():
    """
    The two must agree or the guard compares a token against a width nobody
    else uses. `1w` is excluded: its rule is the anchored offset `W-MON`
    rather than a Timedelta, so `tf_delta` raises on it — a pre-existing
    quirk, and not a width the live loop trades.
    """
    for token in ("1m", "2m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"):
        width = tf_delta(token)
        ts = pd.date_range("2026-08-20", periods=40, freq=width, tz="UTC")
        c = 15000.0 + np.arange(40, dtype=float)
        frame = pd.DataFrame({"ts": ts, "open": c, "high": c, "low": c,
                              "close": c, "volume": 1.0})
        assert infer_timeframe(frame) == token, token


# ==========================================================================
# SPEC TEST 3: resampled bar integrity
# ==========================================================================

def test_resampled_bar_integrity():
    """
    OHLCV aggregation, volume preservation and timestamp alignment, checked
    against the 1-minute bars the higher width was built from.

    Open=first, High=max, Low=min, Close=last, Volume=sum — asserted per
    bucket against the source rows rather than against a second computation,
    so this cannot pass by reimplementing the bug.
    """
    minutes = _minutes(n=720)                       # 12 hours of 1m bars

    for token, size in (("3m", 3), ("5m", 5), ("15m", 15), ("30m", 30),
                        ("1h", 60)):
        out = to_timeframe(minutes, token)
        assert len(out) == 720 // size, f"{token}: {len(out)} bars"

        # No NaN reaches a strategy: a NaN close is a signal computed on a
        # price that did not trade.
        assert out[["open", "high", "low", "close"]].isna().sum().sum() == 0

        # Volume is CONSERVED. A resampler that dropped or double-counted it
        # would mis-size every position drawn on a volume feature.
        assert out["volume"].sum() == minutes["volume"].sum(), token

        # Per-bucket aggregation against the source rows.
        for i in (0, 1, len(out) - 1):
            src = minutes.iloc[i * size:(i + 1) * size]
            row = out.iloc[i]
            assert row["open"] == src["open"].iloc[0], f"{token}[{i}] open"
            assert row["high"] == src["high"].max(), f"{token}[{i}] high"
            assert row["low"] == src["low"].min(), f"{token}[{i}] low"
            assert row["close"] == src["close"].iloc[-1], f"{token}[{i}] close"
            assert row["volume"] == src["volume"].sum(), f"{token}[{i}] volume"


def test_resampled_bars_are_stamped_at_the_open_and_land_on_the_boundary():
    """
    `ts` is the bar's OPEN throughout this repository, and an hourly bar must
    sit on the hour. A resampler stamping the close would shift every bar one
    width forward and every signal with it.
    """
    minutes = _minutes(n=720)
    for token, minute_offsets in (("1h", {0}), ("30m", {0, 30}),
                                  ("15m", {0, 15, 30, 45})):
        out = to_timeframe(minutes, token)
        ts = pd.to_datetime(out["ts"], utc=True)
        assert str(ts.dt.tz) == "UTC", token
        assert set(ts.dt.minute.unique()) <= minute_offsets, token
        assert ts.dt.second.eq(0).all(), token
        # The first bucket opens with the first minute it contains.
        assert ts.iloc[0] == pd.Timestamp(minutes["ts"].iloc[0])


def test_resampling_introduces_no_lookahead():
    """
    Each bar may only see minutes inside it. Truncating the source must leave
    every completed earlier bar byte-identical — if a later minute leaked into
    an earlier bar, removing it would change that bar.
    """
    minutes = _minutes(n=720)
    full = to_timeframe(minutes, "1h")
    truncated = to_timeframe(minutes.iloc[:360], "1h")

    assert len(truncated) == 6
    for col in ("ts", "open", "high", "low", "close", "volume"):
        pd.testing.assert_series_equal(
            full[col].iloc[:6].reset_index(drop=True),
            truncated[col].reset_index(drop=True),
            check_names=False,
            obj=f"{col}: an earlier bar changed when later minutes were removed")


def test_a_native_timeframe_passes_straight_through():
    minutes = _minutes(n=60)
    out = to_timeframe(minutes, "1m")
    assert len(out) == len(minutes)
    assert out["close"].tolist() == minutes["close"].tolist()


def test_an_empty_frame_resamples_to_an_empty_frame():
    assert to_timeframe(pd.DataFrame(), "1h").empty


def test_the_repository_stamps_bars_at_the_OPEN_not_the_close():
    """
    `_resample` uses `label="left", closed="left"`. Pinned because it is the
    convention every certification was computed under: stamping at the close
    would shift every bar one width forward, and every signal drawn on it.
    """
    import inspect                                            # noqa: PLC0415
    from mdlib import lake                                    # noqa: PLC0415
    src = inspect.getsource(lake._resample)
    assert 'label="left"' in src and 'closed="left"' in src, (
        "the lake's resampling convention changed; every timeframe in every "
        "backtest was computed under label/closed = left")


# ==========================================================================
# the bucket resolver
# ==========================================================================

def _handles(*timeframes):
    class _H:
        def __init__(self, tf):
            self.certified_timeframe = tf
            self.strategy_id = f"demo_{tf}"
    return [_H(tf) for tf in timeframes]


def test_the_buckets_are_the_widths_the_roster_names():
    from master_live import required_timeframes                # noqa: PLC0415

    class _D:
        strategies = _handles("1h", "3m", "3m")
    assert required_timeframes(_D(), "15m") == ["3m", "1h"], \
        "deduplicated, and sorted narrowest first"


def test_a_handle_with_no_timeframe_falls_back_to_the_cli_flag():
    """
    The old behaviour, and correct for it: a meta.json written before the key
    existed carries no claim to contradict.
    """
    from master_live import required_timeframes                # noqa: PLC0415

    class _D:
        strategies = _handles(None, "3m")
    assert required_timeframes(_D(), "1h") == ["3m", "1h"]

    class _Empty:
        strategies = []
    assert required_timeframes(_Empty(), "1h") == ["1h"], \
        "an empty roster still reads its one flagged width"
