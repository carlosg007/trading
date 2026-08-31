"""
Tests for `strategies/experimental/intrinsic_alpha_engine_20260831.py`.

ASSERT-BASED, and deliberately so. The collector-style suites in this tree
record failures by appending to a module-level list and only signal through
`sys.exit(1)` in `main()`, which bare pytest never calls - so they watch checks
fail and report green. `tests/conftest.py` routes those. This suite uses plain
asserts and needs no routing.

Every module-level name beginning with `test_` is collected, INCLUDING a helper
whose only argument is defaulted. Helpers here are `_bars` / `_zigzag`, never
`test_*`.

What each section defends:

  * SECTION 1 - the module contract. Four masks; a three-tuple would lose the
    short side into a plausible long-only curve.
  * SECTION 2 - the intrinsic-time machine, against a zig-zag whose turns are
    known by construction. This is where the real logic is, and where the
    first implementation was WRONG: with the direction still unknown a single
    running extreme follows price in both directions, always equals the close,
    and no threshold can ever be breached - the machine emitted zero events on
    a 10% zig-zag under a 2% threshold and every mask came back empty.
  * SECTION 3 - the grid, both counts, and the toggles in isolation.
  * SECTION 4 - causality. Nothing may read bar i+1, and the volatility
    reference the threshold is scaled by must be SHIFTED.
  * SECTION 5 - the walk, held to `sma_momentum_crossover_20260818._walk`.
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
    intrinsic_alpha_engine_20260831 as M)
from strategies.experimental import (                            # noqa: E402
    sma_momentum_crossover_20260818 as SMC)

MODULE_PATH = (REPO / "strategies" / "experimental"
               / "intrinsic_alpha_engine_20260831.py")
BASE = dict(M.DEFAULT_PARAMS)


def _bars(n: int = 3000, seed: int = 11, sigma: float = 12.0,
          drift: float = 0.0) -> pd.DataFrame:
    """A synthetic 15m frame. `ts` is a COLUMN and the index positional,
    which is the shape the engine actually hands a strategy."""
    rng = np.random.default_rng(seed)
    px = (20000 + np.cumsum(rng.normal(0, sigma, n))
          + np.linspace(0.0, drift, n))
    ts = pd.date_range("2021-03-01 13:30", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": px + rng.normal(0, 2, n),
        "high": px + np.abs(rng.normal(0, 9, n)),
        "low": px - np.abs(rng.normal(0, 9, n)),
        "close": px,
        "volume": rng.integers(500, 4000, n).astype(float)})


def _zigzag() -> np.ndarray:
    """Four legs of exactly 10%, so every turn is known by construction."""
    return np.concatenate([np.linspace(100, 110, 40),
                           np.linspace(110, 100, 40),
                           np.linspace(100, 112, 40),
                           np.linspace(112, 103, 40)])


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
    """A reversal engine with one live side is half a strategy, and the short
    side is the half that disappears silently."""
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
    assert M.PORTFOLIO_GROUP == "Index_Intrinsic_Basket"
    assert M.STRATEGY_MODULE == "intrinsic_alpha_engine_20260831"
    assert M.ADX_EXHAUSTION_MAX == 30.0
    assert M.EMA_MACRO == 200
    assert M.MIN_NORM_ATR == 0.0005


def test_target_quadrants_are_corrected_against_the_profiler():
    """The request said "Q1: Low Vol / Trending" and "Q3: Low Vol / Ranging".

    `backtest.profiler` is the only authority and says Q3 and Q4 for those two
    regimes. The REGIMES are kept, the ids move - pinned here against the
    profiler's own map rather than a literal, because a module and a daemon
    disagreeing about what a quadrant means is invisible downstream.
    """
    for name in M.TARGET_REGIMES:
        assert name in REGIMES, f"{name!r} is not one of {list(REGIMES)}"
    ids = tuple(REGIME_TO_QUADRANT[n] for n in M.TARGET_REGIMES)
    assert ids == M.TARGET_QUADRANTS == ("Q3", "Q4")
    assert "Q1" not in M.TARGET_QUADRANTS, (
        "Q1 is High Volatility / Trending, which is not this premise")


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
        assert key in M.PARAM_GRID, f"{key} must be swept"


# ---------------------------------------------------------------------------
# SECTION 2 - the intrinsic-time machine
# ---------------------------------------------------------------------------
def test_directional_changes_are_found_on_a_known_zigzag():
    """Four 10% legs under a 2% threshold must produce four events.

    THE REGRESSION THIS PINS: the first implementation tracked ONE running
    extreme while the direction was still unknown. `mode >= 0` and
    `mode <= 0` are both true at mode 0, so the extreme followed price both
    ways, always equalled the close, and `close <= ext * (1 - thr)` could
    never fire. It returned zero events here and zero trades everywhere.
    """
    close = _zigzag()
    thr = np.full(len(close), 0.02)
    up, dn, os_u, _ext = M._dc_loop(close, thr)
    assert int(up.sum()) == 2, f"expected 2 upward DCs, got {int(up.sum())}"
    assert int(dn.sum()) == 2, f"expected 2 downward DCs, got {int(dn.sum())}"
    # First leg rises from 100; the 2% breach is at 102, ~bar 8 of 40.
    first_up = int(np.flatnonzero(up)[0])
    assert 6 <= first_up <= 10, first_up
    # The turns alternate: up, down, up, down.
    order = sorted([(i, "up") for i in np.flatnonzero(up)]
                   + [(i, "dn") for i in np.flatnonzero(dn)])
    assert [d for _i, d in order] == ["up", "dn", "up", "dn"], order


def test_the_first_run_has_no_overshoot():
    """It has no previous DC point to measure from, so the overshoot is
    UNDEFINED rather than zero. NaN keeps the warm-up out of the signal;
    a zero would be an overshoot the machine never observed."""
    close = _zigzag()
    thr = np.full(len(close), 0.02)
    up, dn, os_u, _ext = M._dc_loop(close, thr)
    events = sorted(np.flatnonzero(up).tolist() + np.flatnonzero(dn).tolist())
    assert np.isnan(os_u[events[0]]), "the first DC reported an overshoot"
    assert np.isfinite(os_u[events[1]]), "the second DC reported none"
    assert int(np.isfinite(os_u).sum()) == len(events) - 1


def test_the_overshoot_is_measured_from_the_previous_dc_not_the_extreme():
    """The scaling law <omega> ~ delta describes the run PAST the last
    confirmed reversal. Measured from the extreme it would be identically one
    threshold on every event and `os_mult` would sweep nothing."""
    close = _zigzag()
    thr = np.full(len(close), 0.02)
    up, dn, os_u, _ext = M._dc_loop(close, thr)
    finite = os_u[np.isfinite(os_u)]
    assert (finite > 1.5).all(), (
        f"overshoots collapsed toward one threshold: {finite}")
    # Leg two runs 102.05 -> 110 before reversing: 7.79% / 2% ~ 3.9.
    assert 3.0 < finite[0] < 5.0, finite[0]


def test_a_bar_with_no_threshold_is_never_an_event():
    close = _zigzag()
    thr = np.full(len(close), np.nan)
    up, dn, os_u, _ext = M._dc_loop(close, thr)
    assert not up.any() and not dn.any()
    assert not np.isfinite(os_u).any()


def test_a_wider_threshold_never_finds_more_events():
    """Monotone by construction, checked on both shapes.

    On the ZIG-ZAG the count is identical at every threshold below the leg
    size - each leg is monotonic, so there is no finer structure for a tighter
    threshold to find, and 1% and 5% both return exactly the four turns. The
    strict inequality only exists on a series with noise in it, which is why
    the random walk is here too rather than the zig-zag alone.
    """
    close = _zigzag()
    counts = [int(u.sum()) + int(d.sum())
              for u, d, _o, _e in
              (M._dc_loop(close, np.full(len(close), t))
               for t in (0.01, 0.02, 0.05))]
    assert counts[0] >= counts[1] >= counts[2], counts

    noisy = _bars(n=2000, seed=5)["close"].to_numpy(dtype="float64")
    ncounts = [int(u.sum()) + int(d.sum())
               for u, d, _o, _e in
               (M._dc_loop(noisy, np.full(len(noisy), t))
                for t in (0.001, 0.005, 0.02))]
    assert ncounts[0] >= ncounts[1] >= ncounts[2], ncounts
    assert ncounts[0] > ncounts[2], f"threshold did not bind: {ncounts}"


def test_os_mult_binds_on_candidate_triggers(bars):
    """Compare CANDIDATES, never realised trades.

    A filter can only remove candidate triggers; the walk holds one position
    at a time, so declining an early trigger can leave the strategy flat for a
    later one it would otherwise have been holding through - and the trade
    count can go UP when a filter is tightened.
    """
    counts = {}
    for om in (1.0, 2.0, 4.0):
        L = M._layers(bars, 0.003, om, 1.0, 20, 20, "09:30", "16:00",
                      True, True, True, False)
        counts[om] = int(L["trigger_long"].sum())
    assert counts[1.0] >= counts[2.0] >= counts[4.0], counts
    assert counts[1.0] > counts[4.0], f"os_mult did not bind: {counts}"


# ---------------------------------------------------------------------------
# SECTION 3 - the grid and the toggles
# ---------------------------------------------------------------------------
def test_active_grid_is_162_cells():
    counts = {k: len(v) for k, v in M.PARAM_GRID.items()}
    cells = int(np.prod(list(counts.values())))
    assert cells == 162, f"3x3x3x3x2 = 162; got {cells} from {counts}"
    for key in M.PARAM_GRID:
        assert key in M.DEFAULT_PARAMS, f"{key} is swept but not declared"


def test_the_requested_grid_is_preserved_and_is_486_not_108():
    """The request labelled its grid 108. 3x3x3x3x3x2 = 486, and nothing in
    it is pinned - so the label was wrong, not the axes."""
    full = M.FULL_PARAM_GRID_AS_REQUESTED
    assert int(np.prod([len(v) for v in full.values()])) == 486
    for key, values in full.items():
        assert key in M.DEFAULT_PARAMS
        if key in M.PARAM_GRID:
            assert M.PARAM_GRID[key] == values, (
                f"{key} was trimmed; only vol_mult should be")
    assert "vol_mult" not in M.PARAM_GRID, "vol_mult is the pinned axis"
    assert M.DEFAULT_PARAMS["vol_mult"] == 1.0


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


def test_tp_none_with_trailing_false_is_allowed(bars):
    """This module HAS signal exits - the opposite DC and the baseline
    reversion - so the un-exited runner the request warns about cannot occur.
    Refusing the pair would drop 54 of 162 cells on a hazard this strategy
    does not have."""
    le, lx, se, sx = M.signal_fn(bars, **{**BASE, "tp_atr_mult": None,
                                          "trailing": False})
    assert int(lx.sum()) + int(sx.sum()) > 0, "nothing ever exited"
    assert np.isnan(M._tp_distance(None))
    assert M._tp_distance(3.0) == 3.0


def test_validate_refuses_incoherent_cells(bars):
    with pytest.raises(ValueError, match="delta_pct"):
        M.signal_fn(bars, **{**BASE, "delta_pct": 0.0})
    with pytest.raises(ValueError, match="os_mult"):
        M.signal_fn(bars, **{**BASE, "os_mult": -1.0})
    with pytest.raises(ValueError, match="sl_atr_mult"):
        M.signal_fn(bars, **{**BASE, "sl_atr_mult": 0.01})
    with pytest.raises(ValueError, match="nearer than"):
        M.signal_fn(bars, **{**BASE, "sl_atr_mult": 2.0, "tp_atr_mult": 0.5})


def test_make_signal_fn_rejects_an_unknown_parameter():
    with pytest.raises(ValueError, match="trail_atr_mult"):
        M.make_signal_fn(trail_atr_mult=3.0)


def test_alpha_trigger_toggle_isolates_the_scaling_law_claim(bars):
    on = M._layers(bars, 0.003, 2.0, 1.0, 20, 20, "09:30", "16:00",
                   True, True, True, False)
    off = M._layers(bars, 0.003, 2.0, 1.0, 20, 20, "09:30", "16:00",
                    True, False, True, False)
    assert int(on["trigger_long"].sum()) < int(off["trigger_long"].sum()), (
        "the overshoot requirement removed no candidates")
    # With the trigger off it is a bare DC reversal - still an EVENT, so the
    # toggle must not turn every bar into an entry.
    assert int(off["trigger_long"].sum()) < len(bars) // 10


def test_volume_filter_binds_on_candidates(bars):
    on = M._layers(bars, 0.003, 2.0, 1.0, 20, 20, "09:30", "16:00",
                   True, True, True, False)
    off = M._layers(bars, 0.003, 2.0, 1.0, 20, 20, "09:30", "16:00",
                    True, True, False, False)
    assert int(on["confirm"].sum()) < int(off["confirm"].sum())


def test_session_filter_restricts_to_the_named_session(bars):
    L = M._layers(bars, 0.003, 2.0, 1.0, 20, 20, "09:30", "16:00",
                  True, True, True, True)
    et = L["ts_et"]
    minutes = et.hour * 60 + et.minute
    inside = (minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)
    assert (L["session_ok"].to_numpy() == inside).all()
    assert L["session_ok"].sum() < len(bars)


def test_session_uses_a_named_zone_not_a_fixed_offset():
    src = MODULE_PATH.read_text()
    assert "America/New_York" in src and "tz_convert" in src


def test_baseline_filter_toggle_is_wired_even_though_it_barely_binds(bars):
    """The request's Layer 1 is an OR, so it passes on most bars for BOTH
    sides. It is implemented as specified rather than quietly tightened to an
    AND - but the toggle still has to change something, or it is decoration.
    """
    calm = M._layers(bars, 0.003, 2.0, 1.0, 20, 20, "09:30", "16:00",
                     True, True, True, False)
    off = M._layers(bars, 0.003, 2.0, 1.0, 20, 20, "09:30", "16:00",
                    False, True, True, False)
    assert bool(off["trend_long"].all()), "the toggle did not disable Layer 1"
    assert int(calm["trend_long"].sum()) <= int(off["trend_long"].sum())


# ---------------------------------------------------------------------------
# SECTION 4 - causality
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


def test_the_volatility_reference_is_shifted(bars):
    """The threshold is scaled by a trailing median of normalised ATR. A
    median whose window ENDED at bar i would let that bar help set the
    threshold it is then tested against."""
    compact = "".join(MODULE_PATH.read_text().split())
    assert ".median().shift(1)" in compact, (
        "the volatility reference is not a SHIFTED trailing median")
    L = M._layers(bars, 0.003, 2.0, 1.0, 20, 20, "09:30", "16:00",
                  True, True, True, False)
    thr = L["threshold"]
    # The reference window must be entirely warm-up at the start.
    assert thr.iloc[:M.VOL_REF_WINDOW].isna().all(), (
        "a threshold existed before its reference window filled")


def test_truncating_the_future_does_not_change_the_past(bars):
    cut = len(bars) - 300
    full = M.ml_features(bars, **BASE).iloc[:cut]
    part = M.ml_features(bars.iloc[:cut].copy(), **BASE)
    pd.testing.assert_frame_equal(full, part, check_exact=False, atol=1e-9)


def test_truncating_the_future_does_not_change_the_signals(bars):
    """The decisive causality check for a STATE MACHINE: it walks forward, so
    a prefix must reproduce the prefix of the full run exactly."""
    cut = len(bars) - 300
    full = M.signal_fn(bars, **BASE)
    part = M.signal_fn(bars.iloc[:cut].copy(), **BASE)
    for a, b, name in zip(full, part, ("le", "lx", "se", "sx")):
        assert (a.to_numpy()[:cut] == b.to_numpy()).all(), (
            f"{name} moved when future bars were removed")


# ---------------------------------------------------------------------------
# SECTION 5 - the walk
# ---------------------------------------------------------------------------
def test_walk_kernel_matches_the_shared_standard():
    rng = np.random.default_rng(3)
    n = 200
    px = 100 + np.cumsum(rng.normal(0, 1.0, n))
    args = (
        (rng.random(n) > 0.93), (rng.random(n) > 0.93),
        (rng.random(n) > 0.95), (rng.random(n) > 0.95),
        px, px + 1.0, px - 1.0, np.full(n, 1.0), np.zeros(n, dtype=bool))
    for sl, tp, trail in ((1.5, 3.0, False), (1.5, np.nan, False),
                          (2.0, 3.0, True), (1.0, np.nan, True)):
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
        np.zeros(n, dtype=bool), 1.5, np.nan, False)
    assert stop[1] > open_[1], "short stop is not above the fill"


def test_degenerate_zero_range_frame_produces_masks_not_nan():
    n = 300
    flat = pd.DataFrame({
        "ts": pd.date_range("2021-01-04 14:30", periods=n, freq="15min",
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
    `backtest.event_calendar` for `use_news_filter`. That single objection is
    the sanctioned exception - the same one `t3_braid_scalp_20260823` carries.
    PINNED AT ONE so it cannot be spent on anything else."""
    from agents.tier3_workers import _audit_ast
    objections = _audit_ast(ast.parse(MODULE_PATH.read_text()))
    assert len(objections) == 1, f"expected 1 objection, got {objections}"
    assert "backtest.event_calendar" in objections[0], objections[0]
