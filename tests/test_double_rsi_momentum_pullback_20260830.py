"""
`strategies/experimental/double_rsi_momentum_pullback_20260830.py`.

Nothing here reads the lake. The bars are synthetic, so a failure is a fact
about the module rather than about what the market did in 2019.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = (REPO_ROOT / "strategies" / "experimental"
               / "double_rsi_momentum_pullback_20260830.py")


def _load():
    spec = importlib.util.spec_from_file_location("dr_pullback", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _bars(n: int = 1400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.6, n))
    return pd.DataFrame(
        {"open": close + rng.normal(0, 0.1, n),
         "high": close + np.abs(rng.normal(0, 0.4, n)),
         "low": close - np.abs(rng.normal(0, 0.4, n)),
         "close": close,
         "volume": rng.integers(500, 5000, n).astype(float)},
        index=pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"))


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

def test_all_four_declarations_are_present():
    """Each closes a specific way a result goes wrong silently: no signal_fn
    and the engine cannot call the module; no indicators and the inspector
    recomputes the series where it is free to disagree; no LOGIC and the tear
    sheet says "not declared"; no PARAM_GRID and --scan sweeps nothing."""
    for name in ("signal_fn", "indicators", "LOGIC", "PARAM_GRID"):
        assert hasattr(M, name), name
    assert set(M.LOGIC) >= {"concept", "entry", "exit"}


def test_returns_the_four_mask_form():
    """A three-tuple or a bare Series raises in unpack_signals rather than
    silently losing the short side into a plausible long-only curve."""
    bars = _bars()
    out = M.signal_fn(bars)
    assert len(out) == 4
    for s in out:
        assert s.dtype == bool and s.index.equals(bars.index)


def test_the_walk_holds_one_position_at_a_time():
    bars = _bars()
    le, lx, se, sx = M.signal_fn(bars)
    assert int(le.sum()) - int(lx.sum()) in (0, 1)
    assert int(se.sum()) - int(sx.sum()) in (0, 1)
    assert not (le & se).any(), "a bar cannot open both sides"


# --------------------------------------------------------------------------
# Causality
# --------------------------------------------------------------------------

def test_signals_do_not_move_when_the_future_is_removed():
    """
    THE test. Every comparison at bar i must read only bars <= i, so truncating
    the frame cannot change a signal that already fired. A lookahead in the
    rolling pullback window or the volume mean would show up here and nowhere
    else - the equity curve it produces looks excellent.
    """
    bars = _bars(seed=11)
    cut = 900
    full = M.signal_fn(bars)
    trunc = M.signal_fn(bars.iloc[:cut])
    for f, t in zip(full, trunc):
        assert (f.iloc[:cut].to_numpy() == t.to_numpy()).all()


def test_the_pullback_window_excludes_the_current_bar():
    """`.shift(1)` on the rolling max is what stops a bar that both dips below
    50 and crosses satisfying the pullback with itself - which would make the
    window parameter do nothing."""
    src = MODULE_PATH.read_text()
    assert ".shift(1)" in src
    body = src[src.index("def _layers"):src.index("def signal_fn")]
    assert "dipped.shift(1)" in body or "dipped = dipped.shift(1)" in body


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("params,why", [
    ({"rsi_fast_len": 21, "rsi_slow_len": 21}, "fast must be strictly faster"),
    ({"rsi_fast_len": 28, "rsi_slow_len": 14}, "inverted lengths"),
    ({"sl_atr_mult": 0.1}, "a stop inside the noise is hit by the fill bar"),
    ({"sl_atr_mult": 2.0, "tp_atr_mult": 0.5}, "reward:risk below the floor"),
    ({"rsi_exit_long": 40.0}, "a long exit below 50 is not an extreme"),
    ({"rsi_exit_short": 60.0}, "a short exit above 50 is not an extreme"),
    ({"pullback_window": 0}, "a window of zero bars is not a pullback"),
    ({"volume_mult": 0.0}, "a multiplier of zero admits every bar"),
])
def test_parameter_sets_that_do_not_describe_this_strategy_are_refused(
        params, why):
    with pytest.raises(ValueError):
        M.signal_fn(_bars(600), **params)


def test_unknown_parameters_raise_rather_than_being_ignored():
    """A stale grid key silently dropped would sweep the DEFAULT and report it
    under the swept name."""
    with pytest.raises(ValueError):
        M.make_signal_fn(bogus_param=1)


def test_no_target_is_unreachable_not_zero():
    """tp_atr_mult=None means no target. 0.0 would place it at the fill and
    close every trade on its own entry bar."""
    assert M._tp_distance(None) == float("inf")
    assert M._tp_distance(3.0) == 3.0
    le, _, se, _ = M.signal_fn(_bars(), tp_atr_mult=None)
    assert int(le.sum()) + int(se.sum()) > 0, "still trades without a target"


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

def test_a_toggle_that_is_off_means_no_opinion_not_the_opposite():
    bars = _bars()
    on = M.signal_fn(bars, use_volume_filter=True)
    off = M.signal_fn(bars, use_volume_filter=False)
    assert int(off[0].sum()) + int(off[2].sum()) >= \
        int(on[0].sum()) + int(on[2].sum()), (
        "removing a confirmation cannot REDUCE the candidate set")


def test_missing_volume_refuses_entries_rather_than_passing_them():
    """A frame with no volume column is a MISSING filter, not a satisfied one.
    Refusing is the direction that cannot invent trades the confirmation never
    cleared."""
    bars = _bars().drop(columns=["volume"])
    le, _, se, _ = M.signal_fn(bars, use_volume_filter=True)
    assert int(le.sum()) == 0 and int(se.sum()) == 0


def test_indicators_come_from_the_same_computation_as_the_signals():
    bars = _bars()
    ind = M.indicators(bars)
    assert all(len(v) == len(bars) for v in ind.values())
    assert any("RSI(5)" in k for k in ind) and any("RSI(21)" in k for k in ind)


def test_ml_features_are_finite_ratios_one_row_per_bar():
    """Shape is checked by the filter and a mismatch raises. Ratios rather than
    levels, or the classifier learns the symbol instead of the setup."""
    bars = _bars()
    f = M.ml_features(bars)
    assert len(f) == len(bars) and not f.empty
    assert np.isfinite(f.to_numpy()).all()
    assert "volume_ratio" in f.columns and "rsi_spread" in f.columns


# --------------------------------------------------------------------------
# Declarations that carry a claim
# --------------------------------------------------------------------------

def test_target_quadrants_match_this_repositorys_numbering():
    """
    `mdlib/regimes.py` is the only authority: Q1 High-Vol/Trending, Q2
    High-Vol/Ranging, Q3 Low-Vol/Trending, Q4 Low-Vol/Ranging. The request
    named "Q1 (Low Vol/Trending)" and "Q2 (High Vol/Trending)"; both ids were
    wrong and are corrected in the module. A quadrant id that disagrees with
    the daemon is invisible downstream - the strategy is stood down in the
    environment it was certified for.
    """
    from mdlib.regimes import QUADRANT_LABELS             # noqa: PLC0415
    # Built from the labels rather than restated, so this check moves with
    # mdlib rather than needing to be kept in step with it by hand.
    by_name = {label: f"Q{i}" for i, label in QUADRANT_LABELS.items() if i}
    for regime, quadrant in zip(M.TARGET_REGIMES, M.TARGET_QUADRANTS):
        assert by_name[regime] == quadrant, (regime, quadrant, by_name)
    # And the request's own numbering is the thing that was wrong.
    assert by_name["Low Volatility / Trending"] == "Q3"
    assert by_name["High Volatility / Trending"] == "Q1"


def test_the_grid_is_counted_honestly():
    """The count is the point: the reported Sharpe is the maximum of this many
    draws from one sample, and that maximum climbs with N."""
    cells = 1
    for values in M.PARAM_GRID.values():
        cells *= len(values)
    assert cells == 162, "trimmed from the request's 1,458 on 2026-08-30"
    assert "162" in MODULE_PATH.read_text(), (
        "the module must state its own cell count")

    # The original is kept for provenance, and it must remain a RECORD rather
    # than drift into something the module never swept.
    full = 1
    for values in M.FULL_PARAM_GRID_AS_REQUESTED.values():
        full *= len(values)
    assert full == 1458
    assert "1,458" in MODULE_PATH.read_text()

    for key in M.PARAM_GRID:
        assert key in M.DEFAULT_PARAMS, f"{key} is swept but not a parameter"

    # THE PINNED PARAMETERS MUST BE THE DEFAULTS. A grid that pins a value
    # while the default differs sweeps around one setting and reports another:
    # every cell would run at volume_mult 1.05 while the provenance comment
    # says 1.1 was chosen.
    pinned = set(M.FULL_PARAM_GRID_AS_REQUESTED) - set(M.PARAM_GRID)
    assert pinned == {"pullback_window", "volume_mult"}
    assert M.DEFAULT_PARAMS["pullback_window"] == 3
    assert M.DEFAULT_PARAMS["volume_mult"] == 1.1
    for key in pinned:
        assert M.DEFAULT_PARAMS[key] in M.FULL_PARAM_GRID_AS_REQUESTED[key], (
            f"{key} is pinned to a value the original grid never contained")


def test_the_module_passes_the_ast_security_audit():
    """No file, network or OS access; no eval/exec/__import__/open."""
    from agents.tier3_workers import _audit_ast            # noqa: PLC0415
    assert _audit_ast(ast.parse(MODULE_PATH.read_text())) == []


def test_every_grid_cell_this_module_accepts_is_walkable():
    """A sample of the grid, run end to end. A cell that raises anywhere other
    than _validate is a bug, not a refusal."""
    keys = list(M.PARAM_GRID)
    combos = list(itertools.product(*M.PARAM_GRID.values()))
    bars = _bars(400, seed=3)
    checked = 0
    # A stride PROPORTIONAL to the grid, not a constant. A fixed 97 was sized
    # for the original 1,458 cells and silently sampled two of the trimmed
    # 162 - the assertion below caught it, which is the only reason this is a
    # comment rather than a coverage hole.
    stride = max(1, len(combos) // 12)
    for combo in combos[::stride]:             # a spread across the space
        params = dict(zip(keys, combo))
        try:
            out = M.signal_fn(bars, **params)
        except ValueError:
            continue                           # a declared refusal
        assert len(out) == 4
        checked += 1
    assert checked > 5, "the sample must actually exercise cells"
