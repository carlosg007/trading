"""
Tests for `strategies/experimental/dual_ema_slope_scalp_20260831.py`.

ASSERT-BASED, and deliberately so. The collector-style suites in this tree
record failures by appending to a module-level list and only signal through
`sys.exit(1)` in `main()`, which bare pytest never calls - so they watch checks
fail and report green. `tests/conftest.py` routes those. This suite uses plain
asserts and needs no routing.

Every module-level name beginning with `test_` is collected, INCLUDING a helper
whose only argument is defaulted. Helpers here are `_bars` / `_ohlc`, never
`test_*`.

What each section defends:

  * SECTION 1 - the module contract, and the corrected quadrant ids.
  * SECTION 2 - the higher horizon. Nothing resamples; `_htf_span` converts
    "fifty hours" onto the run's own bars. This is where the second real bug
    lived: `_bar_minutes` divided a MICROSECOND-backed index by a nanosecond
    constant, read a 5-minute bar as 0.005 minutes, and turned EMA(600) into
    EMA(600000) - a flat line, invisible in every mask.
  * SECTION 3 - the candle patterns, on bars built to be each shape.
  * SECTION 4 - the grid, both counts, and the toggles in isolation.
  * SECTION 5 - causality and the walk.
"""
from __future__ import annotations

import ast
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.engine import unpack_signals                       # noqa: E402
from backtest.profiler import REGIMES, REGIME_TO_QUADRANT        # noqa: E402
from backtest.promote import RISK_KEYS                           # noqa: E402
from backtest.run import RISK_PARAMS                             # noqa: E402
from strategies.experimental import (                            # noqa: E402
    dual_ema_slope_scalp_20260831 as M)
from strategies.experimental import (                            # noqa: E402
    sma_momentum_crossover_20260818 as SMC)

MODULE_PATH = (REPO / "strategies" / "experimental"
               / "dual_ema_slope_scalp_20260831.py")
BASE = dict(M.DEFAULT_PARAMS)


def _bars(n: int = 4000, freq: str = "5min", seed: int = 7,
          sigma: float = 8.0, drift: float = 900.0) -> pd.DataFrame:
    """A synthetic frame. `ts` is a COLUMN and the index positional, which is
    the shape the engine actually hands a strategy."""
    rng = np.random.default_rng(seed)
    px = 20000 + np.cumsum(rng.normal(0, sigma, n)) + np.linspace(0, drift, n)
    ts = pd.date_range("2021-03-01 13:30", periods=n, freq=freq, tz="UTC")
    o = px + rng.normal(0, 1.5, n)
    return pd.DataFrame({
        "ts": ts, "open": o,
        "high": np.maximum(px, o) + np.abs(rng.normal(0, 6, n)),
        "low": np.minimum(px, o) - np.abs(rng.normal(0, 6, n)),
        "close": px,
        "volume": rng.integers(500, 4000, n).astype(float)})


def _ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """A tiny frame from explicit (open, high, low, close) tuples, padded with
    flat bars so the rolling windows have something to warm up on."""
    pad = [(100.0, 101.0, 99.0, 100.0)] * M.EXPANSION_WINDOW
    allrows = pad + rows
    n = len(allrows)
    return pd.DataFrame({
        "ts": pd.date_range("2021-03-01 14:00", periods=n, freq="5min",
                            tz="UTC"),
        "open": [r[0] for r in allrows], "high": [r[1] for r in allrows],
        "low": [r[2] for r in allrows], "close": [r[3] for r in allrows],
        "volume": [1000.0] * n})


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _bars()


# ---------------------------------------------------------------------------
# SECTION 1 - the module contract
# ---------------------------------------------------------------------------
def test_signal_fn_returns_four_aligned_boolean_masks(bars):
    out = M.signal_fn(bars, **BASE)
    assert isinstance(out, tuple) and len(out) == 4
    for s in out:
        assert isinstance(s, pd.Series)
        assert s.dtype == bool, f"expected bool mask, got {s.dtype}"
        assert s.index.equals(bars.index)
        assert not s.isna().any()


def test_engine_unpacks_the_four_mask_form(bars):
    le, lx, se, sx = unpack_signals(M.signal_fn(bars, **BASE), len(bars),
                                    bars.index)
    assert all(m.dtype == bool for m in (le, lx, se, sx))


def test_both_sides_fire(bars):
    le, _lx, se, _sx = M.signal_fn(bars, **BASE)
    assert int(le.sum()) > 0, "no long entries"
    assert int(se.sum()) > 0, "no short entries"


def test_required_declarations_are_present():
    for name in ("signal_fn", "indicators", "LOGIC", "PARAM_GRID",
                 "make_signal_fn", "ml_features"):
        assert hasattr(M, name), f"the module contract requires {name}"
    for key in ("concept", "entry", "exit"):
        assert key in M.LOGIC and M.LOGIC[key].strip()


def test_metadata_declarations():
    assert M.PORTFOLIO_GROUP == "Equity_Momentum_Basket"
    assert M.STRATEGY_MODULE == "dual_ema_slope_scalp_20260831"
    assert M.TIMEFRAME == "5m"
    assert M.HTF_MINUTES == 60
    assert M.MIN_NORM_ATR == 0.0005


def test_target_quadrants_are_corrected_against_the_profiler():
    """The request said "Q1: Low Vol / Trending" and "Q2: High Vol /
    Trending". `backtest.profiler` says those two regimes are Q3 and Q1 -
    and that Q2 is High Volatility / RANGING, the one environment the
    request's own premise says to filter out. Taking the digits literally
    would aim this at chop.
    """
    for name in M.TARGET_REGIMES:
        assert name in REGIMES, f"{name!r} is not one of {list(REGIMES)}"
    ids = tuple(REGIME_TO_QUADRANT[n] for n in M.TARGET_REGIMES)
    assert ids == M.TARGET_QUADRANTS == ("Q3", "Q1")
    assert "Q2" not in M.TARGET_QUADRANTS, (
        "Q2 is High Volatility / Ranging - the chop this premise excludes")


def test_indicators_are_full_length_and_labelled_with_the_real_span(bars):
    """A line called "1h EMA(50)" drawn over 5-minute candles as a 600-bar
    average would be a chart disagreeing with itself."""
    ind = M.indicators(bars, **BASE)
    assert isinstance(ind, dict) and ind
    for name, s in ind.items():
        assert len(s) == len(bars)
        assert s.index.equals(bars.index)
    assert "HTF macro EMA(600)" in ind, list(ind)


def test_risk_keys_are_declared_and_swept():
    assert RISK_PARAMS == RISK_KEYS == ("sl_atr_mult", "tp_atr_mult",
                                        "trailing")
    for key in RISK_KEYS:
        assert key in M.DEFAULT_PARAMS, (
            f"{key} must be declared or promote._risk_block writes "
            f"'NOT DECLARED' for it")
        assert key in M.PARAM_GRID, f"{key} must be swept"


# ---------------------------------------------------------------------------
# SECTION 2 - the higher horizon
# ---------------------------------------------------------------------------
def test_nothing_resamples():
    """This repository's position, written down in
    `double_rsi_macd_scalp_20260823`: a module that resampled internally
    would be a second aggregation free to disagree with `mdlib.lake` about
    bar boundaries and the Sunday merge, silently."""
    src = MODULE_PATH.read_text()
    assert ".resample(" not in src, "the module resamples internally"


def test_htf_span_scales_fifty_hours_onto_each_rung():
    assert M._htf_span(50, 5) == 600
    assert M._htf_span(50, 15) == 200
    assert M._htf_span(50, 30) == 100
    assert M._htf_span(50, 60) == 50


def test_on_an_hourly_run_the_spans_are_the_requests_own_numbers():
    """The identity case, and the cheapest check that the scaling is right:
    at 1h, "50 bars of a 1h chart" must be EMA(50) exactly."""
    hourly = _bars(n=1200, freq="1h")
    L = M._layers(hourly, 7, 17, 50, 7, 17, 0.15, 1.0, 20, "09:30", "16:00",
                  True, True, True, False)
    assert L["htf_spans"] == (50, 7, 17), L["htf_spans"]


def test_bar_minutes_is_unit_independent():
    """THE REGRESSION THIS PINS. `index.view("int64") / 6e10` assumes a
    NANOSECOND-backed index. Pandas 3.0 indexes can be second-, milli-,
    micro- or nanosecond-backed, and on the microsecond index these fixtures
    produce it returned 0.005 for a 5-minute bar - scaling EMA(600) to
    EMA(600000), a flat line, with no mask changing shape to show it.
    """
    for freq, want in (("5min", 5.0), ("15min", 15.0), ("30min", 30.0),
                       ("1h", 60.0)):
        b = _bars(n=400, freq=freq)
        got = M._bar_minutes(M._bar_timestamps(b))
        assert got == pytest.approx(want), f"{freq}: got {got}, want {want}"


def test_bar_minutes_survives_gaps_and_a_short_frame():
    """The median, not the first difference: a real futures index carries the
    CME break, the Sunday reopen and holiday half-days."""
    b = _bars(n=300, freq="5min")
    ts = b["ts"].tolist()
    gapped = ts[:100] + [t + pd.Timedelta(hours=18) for t in ts[100:]]
    b = b.assign(ts=gapped)
    assert M._bar_minutes(M._bar_timestamps(b)) == pytest.approx(5.0)
    tiny = b.iloc[:2].copy()
    assert not np.isfinite(M._bar_minutes(M._bar_timestamps(tiny)))


def test_an_unknown_bar_size_falls_back_to_the_period_unscaled():
    assert M._htf_span(50, float("nan")) == 50
    assert M._htf_span(50, 0.0) == 50


# ---------------------------------------------------------------------------
# SECTION 3 - the candle patterns
# ---------------------------------------------------------------------------
def test_a_bullish_pin_bar_is_recognised():
    # open 100, tiny body up to 100.5, long lower wick to 96, high 100.6
    cd = M._candles(_ohlc([(100.0, 100.6, 96.0, 100.5)]))
    assert bool(cd["bull_pin"].iloc[-1]), "hammer not detected"
    assert not bool(cd["bear_pin"].iloc[-1])


def test_a_bearish_pin_bar_is_recognised():
    cd = M._candles(_ohlc([(100.0, 104.0, 99.4, 99.5)]))
    assert bool(cd["bear_pin"].iloc[-1]), "shooting star not detected"
    assert not bool(cd["bull_pin"].iloc[-1])


def test_engulfing_needs_the_previous_bar_and_the_opposite_colour():
    # down bar, then an up bar covering it
    cd = M._candles(_ohlc([(100.0, 100.2, 98.8, 99.0),
                           (98.9, 101.5, 98.7, 101.2)]))
    assert bool(cd["bull_engulf"].iloc[-1]), "bullish engulfing not detected"
    # two up bars in a row cannot engulf
    cd2 = M._candles(_ohlc([(99.0, 100.2, 98.8, 100.0),
                            (100.0, 101.5, 99.9, 101.2)]))
    assert not bool(cd2["bull_engulf"].iloc[-1])


def test_an_expansion_bar_needs_a_wide_range_and_a_strong_close():
    # the pad bars have range 2.0, so 1.5x is 3.0; this bar's range is 8.0
    cd = M._candles(_ohlc([(100.0, 108.0, 100.0, 107.5)]))
    assert bool(cd["bull_expand"].iloc[-1]), "expansion bar not detected"
    # same width, but closing mid-range is not an expansion entry
    cd2 = M._candles(_ohlc([(100.0, 108.0, 100.0, 104.0)]))
    assert not bool(cd2["bull_expand"].iloc[-1])


def test_a_zero_range_bar_is_not_a_pattern():
    """Every ratio in `_candles` is 0/0 on a bar with high == low. NaN in a
    boolean context is False in some paths and an error in others, so they are
    masked out explicitly."""
    cd = M._candles(_ohlc([(100.0, 100.0, 100.0, 100.0)]))
    for key in ("bull_pin", "bear_pin", "bull_engulf", "bear_engulf",
                "bull_expand", "bear_expand", "bullish", "bearish"):
        assert not bool(cd[key].iloc[-1]), f"{key} fired on a zero-range bar"


def test_the_expansion_reference_is_shifted():
    """A bar compared against a window that already contains it is comparing
    itself to itself."""
    compact = "".join(MODULE_PATH.read_text().split())
    assert ").mean().shift(1)" in compact


# ---------------------------------------------------------------------------
# SECTION 4 - the grid and the toggles
# ---------------------------------------------------------------------------
def test_active_grid_is_162_cells():
    counts = {k: len(v) for k, v in M.PARAM_GRID.items()}
    cells = int(np.prod(list(counts.values())))
    assert cells == 162, f"3x3x3x3x2 = 162; got {cells} from {counts}"
    for key in M.PARAM_GRID:
        assert key in M.DEFAULT_PARAMS, f"{key} is swept but not declared"


def test_the_requested_grid_is_preserved_and_is_486_not_108():
    full = M.FULL_PARAM_GRID_AS_REQUESTED
    assert int(np.prod([len(v) for v in full.values()])) == 486
    assert "slow_ema" not in M.PARAM_GRID, "slow_ema is the pinned axis"
    assert M.DEFAULT_PARAMS["slow_ema"] == 17
    for key, values in full.items():
        if key in M.PARAM_GRID:
            assert M.PARAM_GRID[key] == values, f"{key} was trimmed too"


def test_every_grid_cell_produces_valid_masks(bars):
    keys = list(M.PARAM_GRID)
    n = 0
    for combo in itertools.product(*(M.PARAM_GRID[k] for k in keys)):
        params = {**BASE, **dict(zip(keys, combo))}
        le, lx, se, sx = M.signal_fn(bars, **params)
        assert le.dtype == bool and se.dtype == bool
        open_long = int(le.sum()) - int(lx.sum())
        open_short = int(se.sum()) - int(sx.sum())
        assert 0 <= open_long <= 1, f"{open_long} unmatched long entries"
        assert 0 <= open_short <= 1, f"{open_short} unmatched short entries"
        assert open_long + open_short <= 1, "long and short open together"
        n += 1
    assert n == 162


def test_validate_refuses_a_crossed_ribbon(bars):
    with pytest.raises(ValueError, match="faster than slow_ema"):
        M.signal_fn(bars, **{**BASE, "fast_ema": 21, "slow_ema": 17})


def test_validate_refuses_incoherent_cells(bars):
    with pytest.raises(ValueError, match="min_slope"):
        M.signal_fn(bars, **{**BASE, "min_slope": -0.1})
    with pytest.raises(ValueError, match="sl_atr_mult"):
        M.signal_fn(bars, **{**BASE, "sl_atr_mult": 0.01})
    with pytest.raises(ValueError, match="nearer than"):
        M.signal_fn(bars, **{**BASE, "sl_atr_mult": 2.0, "tp_atr_mult": 0.5})


def test_tp_none_with_trailing_false_is_allowed(bars):
    """Layer 4 supplies a signal exit - the ribbon closing - so the un-exited
    runner the request warns about cannot occur here."""
    le, lx, se, sx = M.signal_fn(bars, **{**BASE, "tp_atr_mult": None,
                                          "trailing": False})
    assert int(lx.sum()) + int(sx.sum()) > 0, "nothing ever exited"
    assert np.isnan(M._tp_distance(None))


def test_make_signal_fn_rejects_an_unknown_parameter():
    with pytest.raises(ValueError, match="trail_atr_mult"):
        M.make_signal_fn(trail_atr_mult=3.0)


def test_the_slope_gate_is_signed_not_a_magnitude(bars):
    """A single test on `abs(slope)` would let a long fire on a fast leg
    dropping hard. The two sides must not both widen with min_slope."""
    L = M._layers(bars, 7, 17, 50, 7, 17, 0.15, 1.0, 20, "09:30", "16:00",
                  True, True, True, False)
    slope = L["slope"]
    up = (slope >= 0.15).fillna(False)
    dn = (slope <= -0.15).fillna(False)
    assert int(up.sum()) > 0 and int(dn.sum()) > 0
    assert int((up & dn).sum()) == 0, "a bar was both rising and falling"


def test_min_slope_binds_on_candidate_triggers(bars):
    """Compare CANDIDATES, never realised trades: a filter removes candidate
    triggers, and the walk holds one position at a time, so a trade count can
    go UP when a filter is tightened."""
    counts = {}
    for ms in (0.0, 0.15, 1.0):
        L = M._layers(bars, 7, 17, 50, 7, 17, ms, 1.0, 20, "09:30", "16:00",
                      True, True, True, False)
        counts[ms] = int(L["trigger_long"].sum())
    assert counts[0.0] >= counts[0.15] >= counts[1.0], counts
    assert counts[0.0] > counts[1.0], f"min_slope did not bind: {counts}"


def test_alpha_trigger_toggle_falls_back_to_a_cross_not_a_state(bars):
    on = M._layers(bars, 7, 17, 50, 7, 17, 0.15, 1.0, 20, "09:30", "16:00",
                   True, True, True, False)
    off = M._layers(bars, 7, 17, 50, 7, 17, 0.15, 1.0, 20, "09:30", "16:00",
                    True, False, True, False)
    assert not on["trigger_long"].equals(off["trigger_long"])
    # A CROSS is rare; a STATE would be true on roughly half of all bars.
    assert int(off["trigger_long"].sum()) < len(bars) // 10


def test_baseline_and_volume_toggles_bind(bars):
    on = M._layers(bars, 7, 17, 50, 7, 17, 0.15, 1.0, 20, "09:30", "16:00",
                   True, True, True, False)
    no_base = M._layers(bars, 7, 17, 50, 7, 17, 0.15, 1.0, 20, "09:30",
                        "16:00", False, True, True, False)
    no_vol = M._layers(bars, 7, 17, 50, 7, 17, 0.15, 1.0, 20, "09:30",
                       "16:00", True, True, False, False)
    assert bool(no_base["trend_long"].all()), "Layer 1 toggle did nothing"
    assert int(on["trend_long"].sum()) <= int(no_base["trend_long"].sum())
    assert int(on["confirm"].sum()) < int(no_vol["confirm"].sum())


def test_session_filter_restricts_to_the_named_session(bars):
    L = M._layers(bars, 7, 17, 50, 7, 17, 0.15, 1.0, 20, "09:30", "16:00",
                  True, True, True, True)
    et = L["ts_et"]
    minutes = et.hour * 60 + et.minute
    inside = (minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)
    assert (L["session_ok"].to_numpy() == inside).all()
    assert L["session_ok"].sum() < len(bars)


def test_session_uses_a_named_zone_not_a_fixed_offset():
    src = MODULE_PATH.read_text()
    assert "America/New_York" in src and "tz_convert" in src


# ---------------------------------------------------------------------------
# SECTION 5 - causality and the walk
# ---------------------------------------------------------------------------
def test_ml_features_shape_and_finiteness(bars):
    f = M.ml_features(bars, **BASE)
    assert isinstance(f, pd.DataFrame)
    assert len(f) == len(bars), "the caller RAISES on a row-count mismatch"
    assert f.index.equals(bars.index)
    assert np.isfinite(f.to_numpy()).all()


def test_no_negative_shift_anywhere_in_the_module():
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else None
        if name not in ("shift", "pct_change", "diff"):
            continue
        for arg in list(node.args) + [k.value for k in node.keywords]:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                raise AssertionError(
                    f"{name}() with a negative offset reads the future "
                    f"(line {node.lineno})")


def test_truncating_the_future_does_not_change_the_past(bars):
    cut = len(bars) - 400
    full = M.ml_features(bars, **BASE).iloc[:cut]
    part = M.ml_features(bars.iloc[:cut].copy(), **BASE)
    pd.testing.assert_frame_equal(full, part, check_exact=False, atol=1e-9)


def test_truncating_the_future_does_not_change_the_signals(bars):
    cut = len(bars) - 400
    full = M.signal_fn(bars, **BASE)
    part = M.signal_fn(bars.iloc[:cut].copy(), **BASE)
    for a, b, name in zip(full, part, ("le", "lx", "se", "sx")):
        assert (a.to_numpy()[:cut] == b.to_numpy()).all(), (
            f"{name} moved when future bars were removed")


def test_walk_kernel_matches_the_shared_standard():
    rng = np.random.default_rng(3)
    n = 200
    px = 100 + np.cumsum(rng.normal(0, 1.0, n))
    args = (
        (rng.random(n) > 0.93), (rng.random(n) > 0.93),
        (rng.random(n) > 0.95), (rng.random(n) > 0.95),
        px, px + 1.0, px - 1.0, np.full(n, 1.0), np.zeros(n, dtype=bool))
    for sl, tp, trail in ((1.0, 3.0, False), (1.0, np.nan, False),
                          (2.0, 3.0, True), (1.5, np.nan, True)):
        mine = M._walk_loop(*args, sl, tp, trail)
        theirs = SMC._walk(*args, sl, tp, trail)
        for a, b in zip(mine, theirs):
            assert np.array_equal(np.asarray(a), np.asarray(b),
                                  equal_nan=True), (
                f"walk diverged at sl={sl} tp={tp} trailing={trail}")


def test_short_stop_sits_above_the_fill():
    n = 40
    open_ = np.full(n, 100.0)
    short_ok = np.zeros(n, dtype=bool)
    short_ok[0] = True
    _le, _lx, _se, _sx, stop, _tp = M._walk_loop(
        np.zeros(n, dtype=bool), short_ok,
        np.zeros(n, dtype=bool), np.zeros(n, dtype=bool),
        open_, np.full(n, 101.0), np.full(n, 99.0), np.full(n, 1.0),
        np.zeros(n, dtype=bool), 1.0, np.nan, False)
    assert stop[1] > open_[1], "short stop is not above the fill"


def test_degenerate_zero_range_frame_produces_masks_not_nan():
    n = 300
    flat = pd.DataFrame({
        "ts": pd.date_range("2021-01-04 14:30", periods=n, freq="5min",
                            tz="UTC"),
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
        "volume": 0.0}, index=range(n))
    out = M.signal_fn(flat, **BASE)
    assert len(out) == 4
    for s in out:
        assert s.dtype == bool and not s.isna().any()
    assert int(out[0].sum()) == 0, "a zero-ATR frame has no tradable bracket"


def test_ast_audit_objects_to_exactly_one_thing():
    """The validator's allowlist has no `backtest`, and this module imports
    `backtest.event_calendar` for `use_news_filter`. PINNED AT ONE so the
    sanctioned exception cannot be spent on anything else."""
    from agents.tier3_workers import _audit_ast
    objections = _audit_ast(ast.parse(MODULE_PATH.read_text()))
    assert len(objections) == 1, f"expected 1 objection, got {objections}"
    assert "backtest.event_calendar" in objections[0], objections[0]
