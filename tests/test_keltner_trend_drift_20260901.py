"""
Tests for `strategies/experimental/keltner_trend_drift_20260901.py`.

ASSERT-BASED, and deliberately so. The collector-style suites in this tree
record failures by appending to a module-level list and only signal through
`sys.exit(1)` in `main()`, which bare pytest never calls - so they watch checks
fail and report green. `tests/conftest.py` exists to route those. This suite
uses plain asserts and needs no routing.

Every module-level name beginning with `test_` is collected, INCLUDING a helper
whose only argument is defaulted - that trap once ran four sections without
their `$BT_ARTIFACTS` redirect and wrote real JSON onto the NFS mount. Helpers
here are named `_check_*` or `_bars`, never `test_*`.

What each section defends:

  * SECTION 1 - the module contract. `signal_fn` must return the FOUR-mask form;
    a three-tuple would silently lose the short side into a plausible long-only
    equity curve.
  * SECTION 2 - the risk-key fix this module exists to get right. The trailing
    distance MUST live in `sl_atr_mult`, because `promote._risk_block` writes
    only RISK_KEYS and a stop under any other name would be absent from the
    promoted card.
  * SECTION 3 - the toggles, each in isolation. A toggle that does nothing is
    worse than no toggle: the sweep prices it and reports a difference that
    was noise.
  * SECTION 4 - causality. Nothing in `ml_features` may read bar i+1.
  * SECTION 5 - the walk, held to `sma_momentum_crossover_20260818._walk` on
    identical arrays. Copies drift; that parity is the only thing standing
    between the convention and two silently different stops.
"""
from __future__ import annotations

import ast
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
    keltner_trend_drift_20260901 as M)
from strategies.experimental import (                            # noqa: E402
    sma_momentum_crossover_20260818 as SMC)

MODULE_PATH = (REPO / "strategies" / "experimental"
               / "keltner_trend_drift_20260901.py")
BASE = dict(M.DEFAULT_PARAMS)


def _bars(n: int = 900, seed: int = 11, drift: float = 55.0) -> pd.DataFrame:
    """A synthetic 30m frame with a drift and enough noise to cross the band.

    `ts` is a COLUMN and the index is positional, which is the shape the engine
    actually hands a strategy - a time-indexed fixture would exercise the other
    branch of `_bar_timestamps` and hide a break in this one.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2021-03-01 13:30", periods=n, freq="30min", tz="UTC")
    px = 4000 + np.linspace(0, drift, n) + np.cumsum(rng.normal(0, 1.2, n))
    return pd.DataFrame({
        "ts": ts,
        "open": px + rng.normal(0, 0.2, n),
        "high": px + np.abs(rng.normal(0, 1.4, n)),
        "low": px - np.abs(rng.normal(0, 1.4, n)),
        "close": px,
        "volume": rng.integers(400, 3000, n).astype(float)})


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _bars()


# ---------------------------------------------------------------------------
# SECTION 1 - the module contract
# ---------------------------------------------------------------------------
def test_signal_fn_returns_four_aligned_boolean_masks(bars):
    out = M.signal_fn(bars, **BASE)
    assert isinstance(out, tuple) and len(out) == 4, (
        "signal_fn must return (entries, exits, short_entries, short_exits); "
        "a three-tuple loses the short side silently")
    for s in out:
        assert isinstance(s, pd.Series)
        assert s.dtype == bool, f"expected bool mask, got {s.dtype}"
        assert s.index.equals(bars.index)
        assert not s.isna().any()


def test_engine_unpacks_the_four_mask_form(bars):
    le, lx, se, sx = unpack_signals(M.signal_fn(bars, **BASE), len(bars),
                                    bars.index)
    assert all(m.dtype == bool for m in (le, lx, se, sx))
    assert int(le.sum()) + int(se.sum()) > 0, "fixture produced no entries"


def test_required_declarations_are_present():
    for name in ("signal_fn", "indicators", "LOGIC", "PARAM_GRID",
                 "make_signal_fn", "ml_features"):
        assert hasattr(M, name), f"the module contract requires {name}"
    for key in ("concept", "entry", "exit"):
        assert key in M.LOGIC and M.LOGIC[key].strip()


def test_metadata_declarations():
    assert M.PORTFOLIO_GROUP == "Trend_Drift"
    assert M.STRATEGY_MODULE == "keltner_trend_drift_20260901"
    assert M.EMA_SLOW == 200
    assert M.ATR_PERIOD == 14


def test_target_quadrant_agrees_with_the_profiler():
    """Q3 is Low-Vol/Trending in `backtest.profiler`, and nowhere else decides.

    Pinned against the profiler's own map rather than a literal, because a
    module and a daemon disagreeing about what Q3 means is invisible
    downstream: the strategy is stood down in the environment it was certified
    for and turned loose in the one it never traded, with every log line
    reading correctly.
    """
    for name in M.TARGET_REGIMES:
        assert name in REGIMES, f"{name!r} is not one of {list(REGIMES)}"
    ids = tuple(REGIME_TO_QUADRANT[n] for n in M.TARGET_REGIMES)
    assert ids == M.TARGET_QUADRANTS == ("Q3",)


def test_indicators_are_full_length_and_named(bars):
    ind = M.indicators(bars, **BASE)
    assert isinstance(ind, dict) and ind
    for name, s in ind.items():
        assert len(s) == len(bars), f"{name} is not full length"
        assert s.index.equals(bars.index)


# ---------------------------------------------------------------------------
# SECTION 2 - the risk-parameter fix
# ---------------------------------------------------------------------------
def test_risk_keys_are_exactly_the_three_the_pipeline_serializes():
    assert RISK_PARAMS == RISK_KEYS == ("sl_atr_mult", "tp_atr_mult",
                                        "trailing")
    for key in RISK_KEYS:
        assert key in M.DEFAULT_PARAMS, (
            f"{key} must be declared or promote._risk_block writes "
            f"'NOT DECLARED' for it")


def test_the_trailing_distance_is_swept_as_sl_atr_mult():
    """The whole point of this module's risk block.

    `_walk_loop` has ONE distance and `trailing` only selects its anchor, so a
    separate trail multiplier has nowhere to go - and would not reach the
    promoted card even if it did.
    """
    assert M.PARAM_GRID["sl_atr_mult"] == [2.0, 3.0, 4.0]
    assert M.PARAM_GRID["trailing"] == [True]
    assert M.DEFAULT_PARAMS["trailing"] is True
    assert M.DEFAULT_PARAMS["tp_atr_mult"] is None


def test_no_trail_atr_mult_anywhere_in_the_module():
    src = MODULE_PATH.read_text()
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    assert "trail_atr_mult" not in names, (
        "trail_atr_mult must not be a parameter of this module - the engine "
        "has no second stop distance and promote.py would not serialize it")


def test_make_signal_fn_rejects_an_unknown_parameter():
    with pytest.raises(ValueError, match="trail_atr_mult"):
        M.make_signal_fn(trail_atr_mult=3.0)


def test_param_grid_is_54_cells_and_every_axis_is_present():
    counts = {k: len(v) for k, v in M.PARAM_GRID.items()}
    assert counts == {"ema_fast": 2, "keltner_len": 3, "keltner_mult": 3,
                      "sl_atr_mult": 3, "trailing": 1}
    cells = int(np.prod(list(counts.values())))
    assert cells == 54, (
        f"the request labelled this 108; 2x3x3x3 = {cells}. `trailing` is "
        f"pinned and multiplies nothing")
    for key in M.PARAM_GRID:
        assert key in M.DEFAULT_PARAMS, f"{key} is swept but not declared"


def test_grid_values_match_the_request():
    assert M.PARAM_GRID["ema_fast"] == [30, 50]
    assert M.PARAM_GRID["keltner_len"] == [20, 30, 40]
    assert M.PARAM_GRID["keltner_mult"] == [1.5, 2.0, 2.5]


def test_validate_refuses_incoherent_cells(bars):
    with pytest.raises(ValueError, match="ema_fast"):
        M.signal_fn(bars, **{**BASE, "ema_fast": 200})
    with pytest.raises(ValueError, match="sl_atr_mult"):
        M.signal_fn(bars, **{**BASE, "sl_atr_mult": 0.01})
    with pytest.raises(ValueError, match="keltner_mult"):
        M.signal_fn(bars, **{**BASE, "keltner_mult": 0.0})


def test_every_grid_cell_produces_valid_masks(bars):
    """All 54 cells walk without raising and return well-formed masks."""
    import itertools
    keys = list(M.PARAM_GRID)
    n = 0
    for combo in itertools.product(*(M.PARAM_GRID[k] for k in keys)):
        params = {**BASE, **dict(zip(keys, combo))}
        le, lx, se, sx = M.signal_fn(bars, **params)
        assert le.dtype == bool and se.dtype == bool
        # At most ONE side may be open at the end of the frame: an entry on the
        # final bars has no later bar to exit on, and the walk holds one
        # position at a time. More than one unmatched entry means the machine
        # lost track of a position.
        open_long = int(le.sum()) - int(lx.sum())
        open_short = int(se.sum()) - int(sx.sum())
        assert 0 <= open_long <= 1, f"{open_long} unmatched long entries"
        assert 0 <= open_short <= 1, f"{open_short} unmatched short entries"
        assert open_long + open_short <= 1, "long and short open together"
        n += 1
    assert n == 54


# ---------------------------------------------------------------------------
# SECTION 3 - the toggles, in isolation
# ---------------------------------------------------------------------------
def test_baseline_filter_binds_on_candidate_triggers(bars):
    """Compare CANDIDATES, never realised trades.

    A filter can only remove candidate triggers; the walk holds one position at
    a time, so declining an early trigger can leave the strategy flat for a
    later one it would otherwise have been holding through - and the trade
    count can go UP when a filter is enabled.
    """
    # A DOWNTRENDING frame, deliberately. On a rising one every upper-band
    # cross already satisfies `close > ema_fast > ema_slow`, so the filter is
    # non-binding and the test would pass on a module that ignored the toggle
    # entirely. Bounces against a falling ribbon are the population it exists
    # to remove.
    falling = _bars(seed=5, drift=-90.0)
    on = M._layers(falling, 50, 20, 2.0, "09:30", "16:00", (0, 1, 2, 3, 4),
                   True, True, False, False)
    off = M._layers(falling, 50, 20, 2.0, "09:30", "16:00", (0, 1, 2, 3, 4),
                    False, True, False, False)
    cand_on = int((on["trend_long"] & on["trigger_long"]).sum())
    cand_off = int((off["trend_long"] & off["trigger_long"]).sum())
    assert cand_off > 0, "the fixture produced no upper-band crosses at all"
    assert cand_on < cand_off, "use_baseline_filter removed no candidates"


def test_alpha_trigger_toggle_changes_the_trigger(bars):
    on = M._layers(bars, 50, 20, 2.0, "09:30", "16:00", (0, 1, 2, 3, 4),
                   True, True, False, False)
    off = M._layers(bars, 50, 20, 2.0, "09:30", "16:00", (0, 1, 2, 3, 4),
                    True, False, False, False)
    assert not on["trigger_long"].equals(off["trigger_long"])


def test_time_filter_restricts_to_the_named_session(bars):
    L = M._layers(bars, 50, 20, 2.0, "09:30", "16:00", (0, 1, 2, 3, 4),
                  True, True, True, False)
    et = L["ts_et"]
    minutes = et.hour * 60 + et.minute
    inside = (minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)
    assert (L["gate"].to_numpy() == inside).all()
    assert L["gate"].sum() < len(bars), "the session gate admitted every bar"


def test_session_uses_a_named_zone_not_a_fixed_offset():
    """A literal UTC offset is an hour wrong for more than half of any sample.

    Checked on the SOURCE rather than on behaviour, because the failure is
    silent: a fixed offset simply selects a different four hours and every log
    line still reads correctly.
    """
    src = MODULE_PATH.read_text()
    assert "America/New_York" in src
    assert "tz_convert" in src


def test_day_filter_restricts_to_allowed_days(bars):
    L = M._layers(bars, 50, 20, 2.0, "09:30", "16:00", (0, 1),
                  True, True, False, True)
    dows = set(pd.Series(L["ts_et"].dayofweek)[L["gate"].to_numpy()].unique())
    assert dows <= {0, 1}, f"day filter admitted {dows}"


def test_restrictive_filters_default_off():
    """Gate R needs 30 holdout trades inside ONE quadrant, and every
    conjunctive filter is drawn from that budget."""
    assert M.DEFAULT_PARAMS["use_time_filter"] is False
    assert M.DEFAULT_PARAMS["use_day_filter"] is False
    assert M.DEFAULT_PARAMS["use_news_filter"] is False


# ---------------------------------------------------------------------------
# SECTION 4 - causality
# ---------------------------------------------------------------------------
def test_ml_features_shape_and_finiteness(bars):
    f = M.ml_features(bars, **BASE)
    assert isinstance(f, pd.DataFrame)
    assert len(f) == len(bars), "the caller RAISES on a row-count mismatch"
    assert f.index.equals(bars.index)
    assert np.isfinite(f.to_numpy()).all(), "non-finite value in the matrix"


def test_ml_features_never_reads_a_future_bar():
    """No negative shift and no negative-period pct_change in the module."""
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
    """The decisive causality check: recompute on a prefix and require the
    overlapping rows to be identical."""
    cut = len(bars) - 120
    full = M.ml_features(bars, **BASE).iloc[:cut]
    part = M.ml_features(bars.iloc[:cut].copy(), **BASE)
    pd.testing.assert_frame_equal(full, part, check_exact=False, atol=1e-9)


def test_truncating_the_future_does_not_change_the_signals(bars):
    cut = len(bars) - 120
    full = M.signal_fn(bars, **BASE)[0].to_numpy()[:cut]
    part = M.signal_fn(bars.iloc[:cut].copy(), **BASE)[0].to_numpy()
    assert (full == part).all(), "an entry moved when future bars were removed"


# ---------------------------------------------------------------------------
# SECTION 5 - the walk
# ---------------------------------------------------------------------------
def _walk_fixture() -> dict:
    n = 40
    open_ = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    atr = np.full(n, 1.0)
    long_ok = np.zeros(n, dtype=bool)
    long_ok[0] = True
    short_ok = np.zeros(n, dtype=bool)
    sig = np.zeros(n, dtype=bool)
    flat = np.zeros(n, dtype=bool)
    return dict(long_ok=long_ok, short_ok=short_ok, sig=sig, open_=open_,
                high=high, low=low, atr=atr, flat=flat)


def test_walk_kernel_matches_the_shared_standard():
    """Copies drift. This is the only thing standing between the convention
    and two silently different stops."""
    fx = _walk_fixture()
    rng = np.random.default_rng(3)
    n = 200
    px = 100 + np.cumsum(rng.normal(0, 1.0, n))
    args = (
        (rng.random(n) > 0.93), (rng.random(n) > 0.93),
        (rng.random(n) > 0.95), (rng.random(n) > 0.95),
        px, px + 1.0, px - 1.0, np.full(n, 1.0), np.zeros(n, dtype=bool))
    for sl, tp, trail in ((2.0, np.nan, False), (2.0, np.nan, True),
                          (1.5, 3.0, False), (1.5, 3.0, True)):
        mine = M._walk_loop(*args, sl, tp, trail)
        theirs = SMC._walk(*args, sl, tp, trail)
        for a, b in zip(mine, theirs):
            assert np.array_equal(np.asarray(a), np.asarray(b),
                                  equal_nan=True), (
                f"walk diverged from sma_momentum_crossover at "
                f"sl={sl} tp={tp} trailing={trail}")


def test_trailing_stop_anchors_on_the_fill_bar_not_the_signal_bar():
    """The high-water mark starts at the FILL. Anchoring it to the signal bar
    puts the stop a bar early and every level after it is wrong."""
    fx = _walk_fixture()
    n = len(fx["open_"])
    high = fx["high"].copy()
    high[1] = 500.0                       # a spike on the SIGNAL bar's fill
    _le, exits, _se, _sx, stop, _tp = M._walk_loop(
        fx["long_ok"], fx["short_ok"], fx["sig"], fx["sig"],
        fx["open_"], high, fx["low"], fx["atr"], fx["flat"],
        2.0, np.nan, True)
    live = np.flatnonzero(np.isfinite(stop))
    assert live[0] == 1, "the stop must first be live on the FILL bar"
    assert stop[1] == pytest.approx(500.0 - 2.0), (
        "the mark must include the fill bar's own high")


def test_short_stop_sits_above_the_fill():
    """A short stop placed below the fill is breached by the fill bar itself."""
    fx = _walk_fixture()
    short_ok = np.zeros(len(fx["open_"]), dtype=bool)
    short_ok[0] = True
    _le, _lx, _se, _sx, stop, _tp = M._walk_loop(
        np.zeros(len(fx["open_"]), dtype=bool), short_ok,
        fx["sig"], fx["sig"], fx["open_"], fx["high"], fx["low"],
        fx["atr"], fx["flat"], 2.0, np.nan, False)
    assert stop[1] > fx["open_"][1], "short stop is not above the fill"


def test_no_target_is_modelled_as_nan_not_a_reachable_level():
    assert np.isnan(M._tp_distance(None))
    assert M._tp_distance(3.0) == 3.0


def test_degenerate_zero_range_frame_produces_masks_not_nan():
    n = 60
    flat = pd.DataFrame({
        "ts": pd.date_range("2021-01-04 14:30", periods=n, freq="30min",
                            tz="UTC"),
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
        "volume": 0.0}, index=range(n))
    out = M.signal_fn(flat, **BASE)
    assert len(out) == 4
    for s in out:
        assert s.dtype == bool and not s.isna().any()
    assert int(out[0].sum()) == 0, "a zero-ATR frame has no tradable bracket"


def test_ast_audit_objects_to_exactly_one_thing():
    """The AST validator's allowlist has no `backtest` in it, and this module
    imports `backtest.event_calendar` for `use_news_filter`.

    That single objection is the sanctioned exception - the same one
    `t3_braid_scalp_20260823` carries. PINNED AT EXACTLY ONE ENTRY so the
    exception cannot be spent on anything else: a second objection appearing
    here means new reach was added under cover of an allowance granted for the
    news calendar.
    """
    from agents.tier3_workers import _audit_ast
    objections = _audit_ast(ast.parse(MODULE_PATH.read_text()))
    assert len(objections) == 1, f"expected 1 objection, got {objections}"
    assert "backtest.event_calendar" in objections[0], objections[0]
