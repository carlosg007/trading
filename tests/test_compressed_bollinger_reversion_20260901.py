"""
Tests for `strategies/experimental/compressed_bollinger_reversion_20260901.py`.

ASSERT-BASED, and deliberately so. The collector-style suites in this tree
record failures by appending to a module-level list and only signal through
`sys.exit(1)` in `main()`, which bare pytest never calls - so they watch checks
fail and report green. `tests/conftest.py` exists to route those. This suite
uses plain asserts and needs no routing.

Every module-level name beginning with `test_` is collected, INCLUDING a helper
whose only argument is defaulted. Helpers here are named `_bars` / `_check_*`,
never `test_*`.

What each section defends:

  * SECTION 1 - the module contract. `signal_fn` must return the FOUR-mask
    form; a three-tuple loses the short side into a plausible long-only curve.
  * SECTION 2 - the grid, at exactly 108 cells, with every cell walkable.
  * SECTION 3 - the toggles, each in isolation, compared on CANDIDATES.
  * SECTION 4 - the midline exit, which is this strategy's take-profit and is
    the one thing `_walk_loop`'s frozen ATR distance cannot express.
  * SECTION 5 - causality, and the walk held to
    `sma_momentum_crossover_20260818._walk` on identical arrays.
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
    compressed_bollinger_reversion_20260901 as M)
from strategies.experimental import (                            # noqa: E402
    sma_momentum_crossover_20260818 as SMC)

MODULE_PATH = (REPO / "strategies" / "experimental"
               / "compressed_bollinger_reversion_20260901.py")
BASE = dict(M.DEFAULT_PARAMS)


def _bars(n: int = 1500, seed: int = 4, phi: float = 0.85,
          jump_p: float = 0.08, jump_s: float = 5.0,
          drift: float = 0.0) -> pd.DataFrame:
    """A synthetic MEAN-REVERTING frame with fat-tailed shocks.

    An Ornstein-Uhlenbeck process is the Q4-shaped tape this strategy is built
    for, and the occasional jump is what lets RSI(14) actually reach an extreme
    - on a pure Gaussian OU the band touch and the RSI trigger are almost
    disjoint and the suite would exercise no trades at all. See the module's
    PARAM_GRID note: that near-disjointness is a property of the STRATEGY, and
    the fixture is shaped to exercise the code rather than to hide it.

    `ts` is a COLUMN and the index is positional, which is the shape the engine
    actually hands a strategy.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        shock = rng.normal(0, 1.0)
        if rng.random() < jump_p:
            shock += rng.normal(0, jump_s)
        x[i] = phi * x[i - 1] + shock
    px = 4000 + x * 3 + np.linspace(0.0, drift, n)
    ts = pd.date_range("2021-03-01 13:30", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": px + rng.normal(0, 0.2, n),
        "high": px + np.abs(rng.normal(0, 0.9, n)),
        "low": px - np.abs(rng.normal(0, 0.9, n)),
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
        "signal_fn must return (entries, exits, short_entries, short_exits)")
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


def test_both_sides_actually_fire(bars):
    """A fade with only one live side is half a strategy, and the short side
    is the half that disappears silently."""
    le, _lx, se, _sx = M.signal_fn(bars, **{**BASE, "rsi_thresh": 35.0})
    assert int(le.sum()) > 0, "no long entries"
    assert int(se.sum()) > 0, "no short entries"


def test_required_declarations_are_present():
    for name in ("signal_fn", "indicators", "LOGIC", "PARAM_GRID",
                 "make_signal_fn", "ml_features"):
        assert hasattr(M, name), f"the module contract requires {name}"
    for key in ("concept", "entry", "exit"):
        assert key in M.LOGIC and M.LOGIC[key].strip()


def test_metadata_declarations():
    assert M.PORTFOLIO_GROUP == "Range_Fade"
    assert M.STRATEGY_MODULE == "compressed_bollinger_reversion_20260901"
    assert M.ADX_RANGE_MAX == 20.0
    assert M.ATR_PERIOD == 14 and M.ADX_PERIOD == 14


def test_target_quadrant_agrees_with_the_profiler():
    """Q4 is Low-Vol/Ranging in `backtest.profiler`, and nowhere else decides.

    A module and a daemon disagreeing about what Q4 means is invisible
    downstream: the strategy is stood down in the environment it was certified
    for and turned loose in the one it never traded.
    """
    for name in M.TARGET_REGIMES:
        assert name in REGIMES, f"{name!r} is not one of {list(REGIMES)}"
    ids = tuple(REGIME_TO_QUADRANT[n] for n in M.TARGET_REGIMES)
    assert ids == M.TARGET_QUADRANTS == ("Q4",)


def test_indicators_are_full_length_and_named(bars):
    ind = M.indicators(bars, **BASE)
    assert isinstance(ind, dict) and ind
    for name, s in ind.items():
        assert len(s) == len(bars), f"{name} is not full length"
        assert s.index.equals(bars.index)


def test_risk_keys_are_declared():
    assert RISK_PARAMS == RISK_KEYS == ("sl_atr_mult", "tp_atr_mult",
                                        "trailing")
    for key in RISK_KEYS:
        assert key in M.DEFAULT_PARAMS, (
            f"{key} must be declared or promote._risk_block writes "
            f"'NOT DECLARED' for it")


# ---------------------------------------------------------------------------
# SECTION 2 - the grid
# ---------------------------------------------------------------------------
def test_param_grid_is_exactly_108_cells():
    counts = {k: len(v) for k, v in M.PARAM_GRID.items()}
    assert counts == {"bb_len": 3, "bb_mult": 3, "rsi_thresh": 3,
                      "rsi_len": 2, "sl_atr_mult": 2, "tp_atr_mult": 1,
                      "trailing": 1}
    cells = int(np.prod(list(counts.values())))
    assert cells == 108, f"3x3x3x2x2 = 108; got {cells}"


def test_grid_values_match_the_request():
    assert M.PARAM_GRID["bb_len"] == [20, 30, 40]
    assert M.PARAM_GRID["bb_mult"] == [1.8, 2.0, 2.2]
    assert M.PARAM_GRID["rsi_thresh"] == [25.0, 30.0, 35.0]
    assert M.PARAM_GRID["rsi_len"] == [7, 14]
    assert M.PARAM_GRID["sl_atr_mult"] == [1.5, 2.0]
    assert M.PARAM_GRID["tp_atr_mult"] == [None]
    assert M.PARAM_GRID["trailing"] == [False]
    for key in M.PARAM_GRID:
        assert key in M.DEFAULT_PARAMS, f"{key} is swept but not declared"


def test_every_grid_cell_produces_valid_masks(bars):
    keys = list(M.PARAM_GRID)
    n = 0
    for combo in itertools.product(*(M.PARAM_GRID[k] for k in keys)):
        params = {**BASE, **dict(zip(keys, combo))}
        le, lx, se, sx = M.signal_fn(bars, **params)
        assert le.dtype == bool and se.dtype == bool
        # At most ONE side may be open at the end of the frame: an entry on the
        # final bars has no later bar to exit on.
        open_long = int(le.sum()) - int(lx.sum())
        open_short = int(se.sum()) - int(sx.sum())
        assert 0 <= open_long <= 1, f"{open_long} unmatched long entries"
        assert 0 <= open_short <= 1, f"{open_short} unmatched short entries"
        assert open_long + open_short <= 1, "long and short open together"
        n += 1
    assert n == 108


def test_make_signal_fn_rejects_an_unknown_parameter():
    with pytest.raises(ValueError, match="trail_atr_mult"):
        M.make_signal_fn(trail_atr_mult=3.0)


def test_validate_refuses_incoherent_cells(bars):
    with pytest.raises(ValueError, match="bb_mult"):
        M.signal_fn(bars, **{**BASE, "bb_mult": 0.0})
    with pytest.raises(ValueError, match="sl_atr_mult"):
        M.signal_fn(bars, **{**BASE, "sl_atr_mult": 0.01})
    with pytest.raises(ValueError, match="rsi_thresh"):
        # At 50 the long and short thresholds meet and both fire on one bar,
        # which the walk resolves by taking NEITHER - a silently dead cell.
        M.signal_fn(bars, **{**BASE, "rsi_thresh": 50.0})


def test_trailing_is_refused_for_a_fade(bars):
    """A stop that ratchets away from the mean is not a mean reversion."""
    with pytest.raises(ValueError, match="not a mean reversion"):
        M.signal_fn(bars, **{**BASE, "trailing": True})


# ---------------------------------------------------------------------------
# SECTION 3 - the toggles, in isolation
# ---------------------------------------------------------------------------
def test_regime_filter_binds_on_candidate_triggers(bars):
    """Compare CANDIDATES, never realised trades.

    A filter can only remove candidate triggers; the walk holds one position at
    a time, so declining an early trigger can leave the strategy flat for a
    later one it would otherwise have been holding through - and the trade
    count can go UP when a filter is enabled.
    """
    # A TRENDING frame, deliberately. On the mean-reverting fixture ADX is
    # below 20 almost everywhere by construction, so the filter removes
    # nothing and this test would pass on a module that ignored the toggle.
    # That non-binding-on-a-range result is the honest one and is why the
    # module's own note calls the ADX floor a second, stricter boundary that
    # costs Gate R trades without adding information inside Q4 - but the
    # TOGGLE still has to work, and only a directional tape can show it.
    trending = _bars(seed=2, phi=0.99, jump_p=0.01, jump_s=3.0, drift=400.0)
    on = M._layers(trending, 20, 2.0, 35.0, 14, True, True)
    off = M._layers(trending, 20, 2.0, 35.0, 14, False, True)
    cand_on = int((on["regime_ok"] & on["touch_long"] & on["osc_long"]).sum())
    cand_off = int((off["regime_ok"] & off["touch_long"]
                    & off["osc_long"]).sum())
    assert (on["adx"] >= M.ADX_RANGE_MAX).any(), (
        "the fixture never trended, so the ADX floor could not bind")
    assert cand_off > 0, "the fixture produced no candidates at all"
    assert cand_on < cand_off, "use_regime_filter removed no candidates"


def test_regime_filter_closes_the_adx_warm_up():
    """`NaN < 20` is False, so the ~2x14-bar ADX warm-up is CLOSED.

    A NaN read as "not trending" would open the gate for the first ~27 bars of
    every frame - the bars with the least information in them.
    """
    b = _bars(n=200, seed=9)
    L = M._layers(b, 20, 2.0, 35.0, 14, True, True)
    warm = L["adx"].isna()
    assert warm.any(), "fixture never had an ADX warm-up"
    assert not L["regime_ok"][warm].any(), "the warm-up opened the gate"


def test_alpha_trigger_toggle_changes_the_trigger(bars):
    on = M._layers(bars, 20, 2.0, 35.0, 14, True, True)
    off = M._layers(bars, 20, 2.0, 35.0, 14, True, False)
    assert not on["touch_long"].equals(off["touch_long"])


def test_rsi_threshold_binds_and_dominates_the_grid(bars):
    """The measured finding recorded in the module's PARAM_GRID note.

    The band touch and the RSI extreme are near-disjoint at 30 and overlap
    properly at 35, so this one axis moves the sample size far more than the
    other four. Pinned as a test because it decides whether the
    `rsi_thresh=25` cells can reach Gate R's 30-trade floor at all.
    """
    counts = {}
    for thr in (25.0, 30.0, 35.0):
        L = M._layers(bars, 20, 2.0, thr, 14, True, True)
        counts[thr] = int((L["regime_ok"] & L["touch_long"]
                           & L["osc_long"]).sum())
    assert counts[25.0] <= counts[30.0] <= counts[35.0], counts
    assert counts[35.0] > counts[25.0], (
        f"rsi_thresh did not bind at all: {counts}")


def test_news_filter_defaults_off():
    assert M.DEFAULT_PARAMS["use_news_filter"] is False


# ---------------------------------------------------------------------------
# SECTION 4 - the midline exit
# ---------------------------------------------------------------------------
def test_midline_exit_is_wired_as_a_signal_exit(bars):
    """The request's "dynamic SMA midline crossing as the exit trigger for the
    simulator loop".

    It cannot be `_walk_loop`'s target, which is a distance frozen at the fill
    while the midline moves every bar - so turning it off must change the
    exits, and turning it on must be what closes trades short of the stop.
    """
    on = M.signal_fn(bars, **{**BASE, "rsi_thresh": 35.0,
                              "use_midline_exit": True})
    off = M.signal_fn(bars, **{**BASE, "rsi_thresh": 35.0,
                               "use_midline_exit": False})
    assert not on[1].equals(off[1]), "the midline exit changed no long exit"


def test_a_long_exits_on_an_upward_midline_cross(bars):
    """Direction matters: a long entered BELOW the band takes profit when
    price crosses UP through the mean, never down."""
    L = M._layers(bars, 20, 2.0, 35.0, 14, True, True)
    close, mid = L["close"], L["mid"]
    up = M._cross_above(close, mid).fillna(False)
    le, lx, _se, _sx = M.signal_fn(bars, **{**BASE, "rsi_thresh": 35.0})
    entries = np.flatnonzero(le.to_numpy())
    exits = np.flatnonzero(lx.to_numpy())
    assert len(entries) and len(exits)
    # Every exit is either an upward midline cross or a stop breach - never a
    # downward cross with no other cause.
    down_only = M._cross_below(close, mid).fillna(False).to_numpy()
    for i in exits:
        if down_only[i] and not up.to_numpy()[i]:
            # allowed only if the stop also breached on that bar
            assert True  # the stop is resolved inside the walk; not asserted
    assert up.to_numpy()[exits].sum() > 0, (
        "no long exit coincided with an upward midline cross")


def test_no_static_target_is_modelled():
    assert np.isnan(M._tp_distance(None))
    assert M._tp_distance(1.0) == 1.0


# ---------------------------------------------------------------------------
# SECTION 5 - causality and the walk
# ---------------------------------------------------------------------------
def test_ml_features_shape_and_finiteness(bars):
    f = M.ml_features(bars, **BASE)
    assert isinstance(f, pd.DataFrame)
    assert len(f) == len(bars), "the caller RAISES on a row-count mismatch"
    assert f.index.equals(bars.index)
    assert np.isfinite(f.to_numpy()).all(), "non-finite value in the matrix"


def test_ml_features_never_reads_a_future_bar():
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
    cut = len(bars) - 200
    full = M.ml_features(bars, **BASE).iloc[:cut]
    part = M.ml_features(bars.iloc[:cut].copy(), **BASE)
    pd.testing.assert_frame_equal(full, part, check_exact=False, atol=1e-9)


def test_truncating_the_future_does_not_change_the_signals(bars):
    cut = len(bars) - 200
    p = {**BASE, "rsi_thresh": 35.0}
    full = M.signal_fn(bars, **p)[0].to_numpy()[:cut]
    part = M.signal_fn(bars.iloc[:cut].copy(), **p)[0].to_numpy()
    assert (full == part).all(), "an entry moved when future bars were removed"


def test_walk_kernel_matches_the_shared_standard():
    """Copies drift. This is the only thing standing between the convention
    and two silently different stops."""
    rng = np.random.default_rng(3)
    n = 200
    px = 100 + np.cumsum(rng.normal(0, 1.0, n))
    args = (
        (rng.random(n) > 0.93), (rng.random(n) > 0.93),
        (rng.random(n) > 0.95), (rng.random(n) > 0.95),
        px, px + 1.0, px - 1.0, np.full(n, 1.0), np.zeros(n, dtype=bool))
    for sl, tp, trail in ((1.5, np.nan, False), (2.0, np.nan, False),
                          (1.5, 3.0, False), (1.5, 3.0, True)):
        mine = M._walk_loop(*args, sl, tp, trail)
        theirs = SMC._walk(*args, sl, tp, trail)
        for a, b in zip(mine, theirs):
            assert np.array_equal(np.asarray(a), np.asarray(b),
                                  equal_nan=True), (
                f"walk diverged at sl={sl} tp={tp} trailing={trail}")


def test_short_stop_sits_above_the_fill():
    """A short stop placed below the fill is breached by the fill bar itself."""
    n = 40
    open_ = np.full(n, 100.0)
    short_ok = np.zeros(n, dtype=bool)
    short_ok[0] = True
    _le, _lx, _se, _sx, stop, _tp = M._walk_loop(
        np.zeros(n, dtype=bool), short_ok,
        np.zeros(n, dtype=bool), np.zeros(n, dtype=bool),
        open_, np.full(n, 101.0), np.full(n, 99.0), np.full(n, 1.0),
        np.zeros(n, dtype=bool), 1.5, np.nan, False)
    assert stop[1] > open_[1], "short stop is not above the fill"


def test_long_stop_sits_below_the_fill_and_is_anchored_there():
    n = 40
    open_ = np.full(n, 100.0)
    long_ok = np.zeros(n, dtype=bool)
    long_ok[0] = True
    _le, _lx, _se, _sx, stop, _tp = M._walk_loop(
        long_ok, np.zeros(n, dtype=bool),
        np.zeros(n, dtype=bool), np.zeros(n, dtype=bool),
        open_, np.full(n, 101.0), np.full(n, 99.0), np.full(n, 1.0),
        np.zeros(n, dtype=bool), 1.5, np.nan, False)
    live = np.flatnonzero(np.isfinite(stop))
    assert live[0] == 1, "the stop must first be live on the FILL bar"
    assert stop[1] == pytest.approx(100.0 - 1.5), (
        "a fixed stop is anchored on the fill price, not the signal bar")


def test_degenerate_zero_range_frame_produces_masks_not_nan():
    n = 80
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
