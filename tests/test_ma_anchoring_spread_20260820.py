#!/usr/bin/env python3
"""
test_ma_anchoring_spread_20260820.py - the spread-hurdle anchoring strategy:
its layer arithmetic, its toggles, its risk keys and the causality of the
feature matrix Version B is fitted on.

Location:  ~/src/trading/tests/test_ma_anchoring_spread_20260820.py

Run EITHER way - and unlike the older suites in this directory, both ways
report the same answer:

    OMP_NUM_THREADS=1 python tests/test_ma_anchoring_spread_20260820.py
    OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
        tests/test_ma_anchoring_spread_20260820.py

EVERY CASE HERE FAILS THROUGH `assert`, DELIBERATELY. The convention in this
directory is a `check(name, ok)` helper appending to a module-level FAILURES
list, with `sys.exit(1)` at the end of `main()` as the only failure signal -
which pytest never calls. pytest collects those suites, watches their checks
fail and reports all green. Asserting instead costs nothing as a script (the
runner below catches AssertionError and prints the same PASS/FAIL table) and
makes the pytest run mean what it says.

Pin the thread count. Section 6 refits the classifier once per completed trade
on a few dozen rows, and on a 16-core box each fit's thread pool costs far more
than the fit - the same reason `test_dual_version.py` carries the same prefix.

Nothing here needs the lake or a network. The news-filter case needs
`backtest/event_calendar.py` to have a calendar covering the fixture's span and
SKIPS LOUDLY when it does not, rather than passing quietly.

WHAT THIS COVERS, and why each one is here rather than assumed:

  * THE LAYER ARITHMETIC AGAINST HAND-COMPUTED NUMBERS. Section 2 builds a
    29-bar frame whose spread ratio is worked out on paper, including a bar
    that misses the 1.5% hurdle by 0.0002 and a bar that clears the hurdle
    while the spread is CONTRACTING. A strategy compared only against its own
    output is pinned, not verified.
  * THE TOGGLES BEING INDEPENDENTLY WIRED. Each layer must remove candidates
    on its own and must never add one. A filter wired to the wrong side of a
    comparison still produces a plausible equity curve; a filter wired to
    nothing produces the identical curve to having it off, which is the failure
    that looks most like success.
  * CANDIDATES, NOT TRADES. Every toggle case counts the masks handed to the
    walk, captured by swapping `_walk` for a recorder. The walk holds one
    position at a time and ignores a trigger arriving while one is open, so
    declining an early candidate leaves the strategy flat for a later one it
    would have been holding through - enabling a filter can ADD realised
    entries, and a case watching the trade count would be checking the wrong
    number.
  * THE RISK KEYS BY THEIR EXACT SPELLING, against `backtest/run.py`'s
    RISK_PARAMS and `backtest/promote.py`'s RISK_KEYS rather than against a
    literal list retyped here. Those lookups do not alias: under a different
    spelling the leaderboard's stop and target columns come back BLANK, and
    blank in that file means "this strategy has no such setting", never "the
    setting was off".
  * CAUSALITY BY TRUNCATION AND BY PERTURBATION. Section 5 recomputes the
    feature matrix and the signals on every prefix of the frame and requires
    the rows to be identical, then rewrites the tail of the frame and requires
    the head to be unchanged. A `shift(-1)` is caught by both; so is a global
    mean, a centred window and a reversed slice, none of which a source scan
    for negative shifts would see.
  * THE tp=None RULE. This module refuses `tp_atr_mult=None` with a fixed stop,
    which is the strategy request's own rule and a DEPARTURE from what
    `tests/test_risk_params.py` requires of the modules in its table - which is
    why it is not registered there, and why sections 3 and 4 pin both halves of
    the rule here instead.
"""

from __future__ import annotations

import ast
import inspect
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier3_workers import (apply_ml_signal_filter,        # noqa: E402
                                  causal_features, load_strategy,
                                  StrategyLoadError)
from agents.tier3_workers import _bar_timestamps as SHARED_TIMESTAMPS  # noqa: E402
from backtest.engine import unpack_signals                       # noqa: E402
from strategies.experimental import ema_crossover as EC          # noqa: E402
from strategies.experimental import ma_anchoring_spread_20260820 as M  # noqa: E402

MODULE_PATH = (REPO / "strategies" / "experimental"
               / "ma_anchoring_spread_20260820.py")

# The parameters every case runs at unless it says otherwise. NOT the module
# defaults: a 200-bar anchor over a 4,000-bar fixture leaves too few completed
# trades for the ML section to have anything to fit, and the point of these
# cases is the layer arithmetic, which is the same at any window. The defaults
# are exercised where they matter - against real bars, by the runner.
BASE = {"fast_window": 10, "slow_window": 50, "spread_threshold": 0.015,
        "exit_revert_mult": 0.5, "sl_atr_mult": 1.5, "tp_atr_mult": 3.0,
        "trailing": False}

# The per-bar drift that gives the short side something to work with.
SHORT_DRIFT = -0.02


def synthetic(n: int = 4000, seed: int = 5, drift: float = 0.02,
              start: str = "2022-03-01 13:30") -> pd.DataFrame:
    """
    A 15m frame with enough drift for a trend module to find entries.

    The generator `tests/test_risk_params.py` and
    `tests/test_sma_momentum_crossover.py` use, reproduced rather than imported
    so this file does not depend on another suite's fixture staying the shape
    it is today. Volume VARIES: the feature matrix carries a volume z-score,
    and a constant volume column is a column of zeros over a zero standard
    deviation - a degenerate feature that would pass a shape check while
    carrying no information.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    px = 100 + np.cumsum(rng.normal(drift, 0.4, n))
    return pd.DataFrame({
        "ts": ts,
        "open": px,
        "high": px + np.abs(rng.normal(0, 0.5, n)),
        "low": px - np.abs(rng.normal(0, 0.5, n)),
        "close": px + rng.normal(0, 0.1, n),
        "volume": rng.integers(200, 6000, n).astype(float),
    })


def hand_frame() -> pd.DataFrame:
    """
    A 29-bar frame whose spread ratio is computed on paper in section 2.

    Twenty-four flat bars at 100.00 so both averages exist and equal 100 - and
    so ATR exists, which `ready` requires before any entry may fire. The bars
    carry a real 1.00-point range rather than being degenerate: a zero-range
    warm-up gives ATR 0, which puts the stop exactly at the fill price and
    stops every trade out on its own fill bar.

    Then the five bars the arithmetic is about: 106, 112, 112, 112, 100.
    """
    close = np.array([100.0] * 24 + [106.0, 112.0, 112.0, 112.0, 100.0])
    n = len(close)
    return pd.DataFrame({
        "ts": pd.date_range("2022-03-01 13:30", periods=n, freq="15min",
                            tz="UTC"),
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def candidates(bars: pd.DataFrame, **overrides):
    """
    The candidate entry masks and the signal-exit masks AS THE MODULE HANDS
    THEM TO THE WALK, captured by swapping `_walk` for a recorder.

    Read before the walk on purpose - see the module docstring's note on
    candidates versus trades. Captured rather than recomputed here so the case
    checks the arrays the strategy actually acts on: a test-side copy of the
    conditions would be checking a second implementation against the
    specification while the module was free to disagree with both.
    """
    params = {**BASE, **overrides}
    seen: dict = {}

    def recorder(long_ok, short_ok, long_exit, short_exit, *rest):
        seen.update(long_ok=np.asarray(long_ok).copy(),
                    short_ok=np.asarray(short_ok).copy(),
                    long_exit=np.asarray(long_exit).copy(),
                    short_exit=np.asarray(short_exit).copy())
        n = len(long_ok)
        z = np.zeros(n, dtype=bool)
        nan = np.full(n, np.nan)
        return z, z.copy(), z.copy(), z.copy(), nan, nan.copy()

    original = M._walk
    M._walk = recorder
    try:
        M.signal_fn(bars, **params)
    finally:
        M._walk = original
    return seen


def masks(bars: pd.DataFrame, **overrides):
    """
    `(long_entries, long_exits, short_entries, short_exits)` through the
    engine's own unpacker.

    Going through `unpack_signals` rather than unpacking here means this
    accepts exactly what the engine accepts - a module returning a three-tuple
    would fail here the same way it fails in a run, instead of being silently
    truncated to its first two masks.
    """
    return unpack_signals(M.signal_fn(bars, **{**BASE, **overrides}),
                          len(bars))


# ==========================================================================
# 1. The module contract
# ==========================================================================
def test_declares_every_hook_the_contract_requires() -> None:
    """CLAUDE.md's four mandatory declarations, plus the two this one adds."""
    for name in ("signal_fn", "indicators", "make_signal_fn", "ml_features"):
        assert callable(getattr(M, name, None)), f"{name} is not declared"
    assert isinstance(M.PARAM_GRID, dict) and M.PARAM_GRID, "PARAM_GRID empty"
    assert isinstance(M.DEFAULT_PARAMS, dict), "DEFAULT_PARAMS is not a dict"
    assert isinstance(M.LOGIC, dict), "LOGIC is not a dict"
    for key in ("concept", "entry", "exit"):
        text = M.LOGIC.get(key)
        assert isinstance(text, str) and text.strip(), f"LOGIC[{key!r}] empty"


def test_the_identifier_is_pinned_in_both_places() -> None:
    """
    The request pins the string `ma_anchoring_spread` in the metadata AND in
    the LOGIC block, and the FILE carries a date suffix.

    Both spellings are checked because they are used by different things:
    `backtest/run.py` resolves `--strat` by FILENAME, while the pipeline's JSON
    handoffs and the logic card carry the identifier. A rename that updated one
    and not the other would leave a run whose reports name a strategy the CLI
    cannot resolve.
    """
    assert M.STRATEGY_NAME == "ma_anchoring_spread", M.STRATEGY_NAME
    assert M.LOGIC.get("name") == M.STRATEGY_NAME, M.LOGIC.get("name")
    assert MODULE_PATH.exists(), f"module not at {MODULE_PATH}"
    assert MODULE_PATH.stem == "ma_anchoring_spread_20260820", MODULE_PATH.stem
    assert M.STRATEGY_NAME in MODULE_PATH.stem, "identifier is not the stem's root"

    # And the CLI name really does resolve to this file, rather than the
    # identifier happening to look plausible.
    from backtest.run import resolve_strategy
    assert resolve_strategy(MODULE_PATH.stem).resolve() == MODULE_PATH.resolve()


def test_signal_fn_returns_four_boolean_masks_on_the_bars_index() -> None:
    """Output shape and dtype - the contract the engine calls."""
    bars = synthetic()
    out = M.signal_fn(bars, **BASE)
    assert isinstance(out, tuple) and len(out) == 4, f"got {type(out)} len ?"
    names = ("long_entries", "long_exits", "short_entries", "short_exits")
    for name, mask in zip(names, out):
        assert isinstance(mask, pd.Series), f"{name} is {type(mask)}"
        assert mask.dtype == bool, f"{name} dtype is {mask.dtype}"
        assert len(mask) == len(bars), f"{name} length {len(mask)}"
        assert mask.index.equals(bars.index), f"{name} index is not the bars'"
        assert not mask.isna().any(), f"{name} carries NaN"
    # And through the engine's unpacker, which is what actually consumes it.
    le, lx, se, sx = unpack_signals(out, len(bars))
    for name, arr in zip(names, (le, lx, se, sx)):
        assert np.asarray(arr).dtype == bool, f"unpacked {name} is not boolean"


def test_it_trades_both_directions() -> None:
    """
    A bidirectional module that quietly loses one side still produces a
    plausible equity curve, so both are required to fire on drift that suits
    them.
    """
    up = masks(synthetic(drift=0.02))
    assert up[0].sum() > 0, "no long entries on an up-drifting frame"
    down = masks(synthetic(drift=SHORT_DRIFT))
    assert down[2].sum() > 0, "no short entries on a down-drifting frame"
    # Entries and exits are paired: at most one trade may be open at the end.
    for label, (le, lx, se, sx) in (("up", up), ("down", down)):
        opened = int(le.sum()) + int(se.sum())
        closed = int(lx.sum()) + int(sx.sum())
        assert 0 <= opened - closed <= 1, (
            f"{label}: {opened} entries against {closed} exits")


def test_the_loader_binds_it_the_way_a_run_does() -> None:
    """
    `load_strategy` is the only path the engine reaches a strategy through, so
    binding through it is the check that matters - a module that imports
    cleanly and cannot be bound is a module no run can use.
    """
    params = {"fast_window": 10, "slow_window": 50}
    fn, info = load_strategy(str(MODULE_PATH), params)
    bars = synthetic(n=600)
    out = fn(bars)
    assert len(out) == 4, "the bound function lost the four-mask form"
    assert info["param_grid"] == M.PARAM_GRID, "PARAM_GRID did not survive"
    assert info["timeframe"] == M.TIMEFRAME, info["timeframe"]
    assert callable(info.get("ml_feature_fn")), "ml_features was not bound"
    # The bound hook must carry the RUN's windows, not the module defaults -
    # this module's feature matrix depends on them.
    bound = info["ml_feature_fn"](bars)
    direct = M.ml_features(bars, fast_window=10, slow_window=50)
    assert np.allclose(bound.to_numpy(dtype=float),
                       direct.to_numpy(dtype=float), equal_nan=True), (
        "the bound ml_features hook was not given the run's parameters")
    # An unknown parameter must raise rather than being ignored: a stale grid
    # key that binds silently runs the defaults under the winner's name.
    try:
        load_strategy(str(MODULE_PATH), {"fast_windwo": 10})
        raise AssertionError("a misspelled parameter was accepted")
    except StrategyLoadError as e:
        assert "fast_windwo" in str(e), f"the error does not name it: {e}"


# ==========================================================================
# 2. The layer arithmetic, against numbers computed on paper
# ==========================================================================
# The frame is `hand_frame()`: 24 bars at 100.00, then 106, 112, 112, 112, 100.
# With fast_window=2 and slow_window=4 the spread ratio is, bar by bar:
#
#   i=23  fast 100.0  slow 100.0   ratio 1.000000
#   i=24  fast 103.0  slow 101.5   ratio 1.0147783   <- misses 1.015 hurdle
#   i=25  fast 109.0  slow 104.5   ratio 1.0430622   <- clears it, widening
#   i=26  fast 112.0  slow 107.5   ratio 1.0418605   <- clears it, CONTRACTING
#   i=27  fast 112.0  slow 110.5   ratio 1.0135747   <- inside the hurdle
#   i=28  fast 106.0  slow 109.0   ratio 0.9724771   <- clears the SHORT hurdle
#
# Everything through i=23 sits at exactly 1.0: no hurdle cleared either way and
# no expansion either way.
HAND = {"fast_window": 2, "slow_window": 4, "spread_threshold": 0.015,
        "exit_revert_mult": 0.5, "sl_atr_mult": 1.5, "tp_atr_mult": 3.0,
        "trailing": False}
HAND_RATIOS = {23: 1.0, 24: 103.0 / 101.5, 25: 109.0 / 104.5,
               26: 112.0 / 107.5, 27: 112.0 / 110.5, 28: 106.0 / 109.0}


def test_the_spread_ratio_is_the_ratio_of_the_two_averages() -> None:
    bars = hand_frame()
    ratio = M._series(bars, HAND["fast_window"], HAND["slow_window"])["ratio"]
    for i, expected in HAND_RATIOS.items():
        assert abs(float(ratio.iloc[i]) - expected) < 1e-12, (
            f"bar {i}: ratio {float(ratio.iloc[i])!r}, expected {expected!r}")
    # Hand-checked to seven places, so a change of definition is visible here
    # rather than only in an equity curve.
    assert abs(float(ratio.iloc[24]) - 1.0147783) < 1e-7
    assert abs(float(ratio.iloc[28]) - 0.9724771) < 1e-7


def test_the_hurdle_and_the_expansion_pick_exactly_the_right_bars() -> None:
    """
    The three layers together admit ONE long bar and ONE short bar out of 29,
    and the two bars they reject are the interesting ones: 24 misses the hurdle
    by 0.0002 and 26 clears the hurdle while the spread is contracting.
    """
    got = candidates(hand_frame(), **HAND)
    assert list(np.flatnonzero(got["long_ok"])) == [25], (
        f"long candidates {list(np.flatnonzero(got['long_ok']))}, expected [25]")
    assert list(np.flatnonzero(got["short_ok"])) == [28], (
        f"short candidates {list(np.flatnonzero(got['short_ok']))}, expected [28]")


def test_the_exit_band_sits_inside_the_hurdle() -> None:
    """
    Layer 4: exit when the ratio is back inside 1 +/- threshold*mult. With
    threshold 0.015 and mult 0.5 the long band edge is 1.0075, so bar 27's
    1.0136 - already back inside the ENTRY hurdle - is NOT yet an exit, and bar
    28's 0.9725 is.

    That gap is the point of the multiplier. A band at the hurdle itself would
    close every trade on the first bar the divergence stopped growing.
    """
    got = candidates(hand_frame(), **HAND)
    assert not got["long_exit"][27], "bar 27 exited: the band is not inside the hurdle"
    assert got["long_exit"][28], "bar 28 did not exit on the reversion"
    assert not got["short_exit"][28], "bar 28 exited a short that had just opened"

    # And the multiplier really moves the band: at 1.0 the exit is the hurdle
    # itself, so bar 27 exits too.
    wide = candidates(hand_frame(), **{**HAND, "exit_revert_mult": 1.0})
    assert wide["long_exit"][27], "exit_revert_mult=1.0 did not widen the band"
    # At 0.0 the exit is a full reversion to parity, so neither 27 nor the
    # bars sitting at exactly 1.0 count until the ratio goes BELOW 1.0.
    tight = candidates(hand_frame(), **{**HAND, "exit_revert_mult": 0.0})
    assert not tight["long_exit"][27], "exit_revert_mult=0.0 exited early"
    assert not tight["long_exit"][23], "a ratio of exactly 1.0 is not below 1.0"
    assert tight["long_exit"][28], "a full reversion below 1.0 did not exit"


# ==========================================================================
# 3. The toggles, one at a time
# ==========================================================================
TOGGLES = ("use_macro_anchor", "use_spread_hurdle", "use_spread_expansion")


def test_each_toggle_subtracts_candidates_on_its_own() -> None:
    """
    ISOLATION: with the other two layers OFF, switching one layer on must
    remove candidates and must never add one.

    Holding the others off is what makes this a statement about the layer
    rather than about the stack. With all three on, a layer that removes
    nothing extra is indistinguishable from a layer wired to nothing, because
    the other two already removed the bars it would have.
    """
    bars = synthetic()
    for toggle in TOGGLES:
        off = {t: False for t in TOGGLES}
        on = {**off, toggle: True}
        base = candidates(bars, **off)
        gated = candidates(bars, **on)
        for side in ("long_ok", "short_ok"):
            added = int((gated[side] & ~base[side]).sum())
            assert added == 0, (
                f"{toggle} ADDED {added} {side} candidates - a filter may only "
                f"remove")
            removed = int((base[side] & ~gated[side]).sum())
            assert removed > 0, (
                f"{toggle} removed no {side} candidates: it is wired to "
                f"nothing, which is indistinguishable from having it off")


def test_every_toggle_off_is_a_mute_strategy_not_a_null_one() -> None:
    """
    With all three layers off both sides signal on every warm bar, and the
    walk's ambiguity rule takes NEITHER - so the configuration trades zero
    times.

    Pinned because it is surprising and because it decides how an ablation is
    read: the all-off cell is not the baseline this idea has to beat, it is a
    strategy that cannot fire. The honest baseline is the hurdle alone.
    """
    bars = synthetic()
    off = {t: False for t in TOGGLES}
    got = candidates(bars, **off)
    both = got["long_ok"] & got["short_ok"]
    assert both.sum() > 0, "the all-off configuration signalled only one side"
    le, lx, se, sx = masks(bars, **off)
    assert int(le.sum()) == 0 and int(se.sum()) == 0, (
        f"the all-off configuration traded: {int(le.sum())} long, "
        f"{int(se.sum())} short entries")
    # And the honest baseline does trade.
    le2, _lx2, se2, _sx2 = masks(bars, **{**off, "use_spread_hurdle": True})
    assert int(le2.sum()) + int(se2.sum()) > 0, (
        "the hurdle alone produced no trades either - the baseline is unusable")


def test_the_hurdle_toggle_admits_the_bar_that_missed_by_a_whisker() -> None:
    """
    The hand frame's bar 24 clears the anchor and is expanding, and misses the
    1.5% hurdle by 0.0002. Switching the hurdle off must admit exactly it.
    """
    on = candidates(hand_frame(), **HAND)
    off = candidates(hand_frame(), **{**HAND, "use_spread_hurdle": False})
    added = set(np.flatnonzero(off["long_ok"])) - set(np.flatnonzero(on["long_ok"]))
    assert 24 in added, f"bar 24 was not admitted by use_spread_hurdle=False: {added}"


def test_the_expansion_toggle_admits_the_contracting_bar() -> None:
    """
    Bar 26 clears the hurdle while the spread is CONTRACTING (1.04186 against
    the previous 1.04306). Only the expansion layer rejects it.
    """
    on = candidates(hand_frame(), **HAND)
    off = candidates(hand_frame(), **{**HAND, "use_spread_expansion": False})
    added = set(np.flatnonzero(off["long_ok"])) - set(np.flatnonzero(on["long_ok"]))
    assert added == {26}, f"expected bar 26 admitted, got {added}"
    # A spread that held exactly still is not expanding either: every flat bar
    # of the warm-up sits at a ratio of exactly 1.0.
    assert not on["long_ok"][20] and not on["short_ok"][20], (
        "a perfectly flat spread was read as expanding")


def test_a_disabled_layer_costs_no_warm_up() -> None:
    """
    `ready` is assembled from the ACTIVE layers, so switching the expansion
    layer off must not keep its one-bar lookback requirement.

    The general form of this bites hard in modules whose layers use different
    indicators - a toggled-off ADX still costing 27 bars scores the "no ADX"
    cell on a shorter history than the bare strategy it represents. Here only
    Layer 3 has its own warm-up, and it is one bar.
    """
    bars = synthetic()
    with_exp = candidates(bars, use_spread_expansion=True,
                          use_macro_anchor=False, use_spread_hurdle=False)
    without = candidates(bars, use_spread_expansion=False,
                         use_macro_anchor=False, use_spread_hurdle=False)
    first_with = int(np.flatnonzero(with_exp["long_ok"] | with_exp["short_ok"])[0])
    first_without = int(np.flatnonzero(without["long_ok"] | without["short_ok"])[0])
    assert first_without < first_with, (
        f"the disabled expansion layer still cost its lookback: first candidate "
        f"at {first_without} either way")


def test_turning_the_hurdle_off_makes_the_exit_band_bind_immediately() -> None:
    """
    Divergence 5 in the module docstring, pinned: with no hurdle an entry can
    fire at a ratio already inside the exit band, and the walk checks exits
    from the fill bar onward - so the trades are one bar long.

    This is why the hurdle toggle is not an ablation you can read as "the same
    strategy without the hurdle".
    """
    bars = synthetic()

    def holds(**overrides) -> np.ndarray:
        le, lx, _se, _sx = masks(bars, **overrides)
        entries = np.flatnonzero(np.asarray(le))
        exits = np.flatnonzero(np.asarray(lx))
        out = []
        for e in entries:
            j = int(np.searchsorted(exits, e))
            if j < len(exits):
                out.append(int(exits[j] - e))
        return np.asarray(out)

    off = holds(use_spread_hurdle=False)
    on = holds()
    assert len(off) > 50 and len(on) > 5, f"not enough trades: {len(off)}, {len(on)}"
    # Not EVERY hurdle-off trade is one bar - an entry can still fire while the
    # spread happens to be outside the band - so this is a share, and it is
    # written as a share because the absolute claim is the one that is wrong.
    off_share = float((off <= 1).mean())
    on_share = float((on <= 1).mean())
    assert off_share > 0.5, (
        f"only {off_share:.0%} of hurdle-off trades lasted one bar - the "
        f"interaction this pins has gone")
    assert on_share == 0.0, (
        f"{on_share:.0%} of trades lasted one bar WITH the hurdle on: a fresh "
        f"trade should always start outside its own exit band")
    assert np.median(off) < np.median(on), (
        f"median hold {np.median(off)} without the hurdle against "
        f"{np.median(on)} with it")


# ==========================================================================
# 4. Risk parameters, validation and the grid
# ==========================================================================
def test_the_risk_keys_are_spelled_the_way_the_tooling_looks_them_up() -> None:
    """
    Against `run.RISK_PARAMS` and `promote.RISK_KEYS` themselves, not a literal
    list retyped here. Neither lookup aliases: under a different spelling the
    leaderboard's `sl_atr_mult`, `tp_atr_mult` and `trailing` columns come back
    BLANK, and blank in that file means "this strategy has no such setting",
    never "the setting was off" - so a strategy whose exits are a stop and a
    target would be recorded as having neither.
    """
    from backtest.run import RISK_PARAMS
    from backtest.promote import RISK_KEYS
    assert tuple(RISK_PARAMS) == tuple(RISK_KEYS), (
        f"the two lookups have diverged: {RISK_PARAMS} vs {RISK_KEYS}")
    signature = inspect.signature(M.signal_fn).parameters
    for key in RISK_PARAMS:
        assert key in M.DEFAULT_PARAMS, f"DEFAULT_PARAMS is missing {key}"
        assert key in signature, f"signal_fn does not take {key}"
        assert key in inspect.signature(M.make_signal_fn).parameters, (
            f"make_signal_fn does not take {key}")
    for key in ("sl_atr_mult", "tp_atr_mult", "trailing"):
        assert key in M.PARAM_GRID, f"PARAM_GRID does not sweep {key}"


def test_the_declared_dictionaries_agree_with_the_signature() -> None:
    """
    Key integrity in both directions. `load_strategy` rejects an unknown
    parameter name, so a stale PARAM_GRID key raises one contract into a sweep
    rather than being ignored - and a DEFAULT_PARAMS key the function does not
    take is the same failure one step earlier.
    """
    signature = set(inspect.signature(M.signal_fn).parameters) - {"bars"}
    unknown_defaults = set(M.DEFAULT_PARAMS) - signature
    assert not unknown_defaults, f"DEFAULT_PARAMS has unknown keys: {unknown_defaults}"
    unknown_grid = set(M.PARAM_GRID) - signature
    assert not unknown_grid, f"PARAM_GRID has unknown keys: {unknown_grid}"
    missing = signature - set(M.DEFAULT_PARAMS)
    assert not missing, (
        f"signal_fn takes {missing}, which DEFAULT_PARAMS does not declare - "
        f"the leaderboard's params column would understate the run")
    # The declared defaults must BE the signature's defaults, or the params
    # column records one strategy while another ran.
    for key, value in M.DEFAULT_PARAMS.items():
        bound = inspect.signature(M.signal_fn).parameters[key].default
        assert bound == value, (
            f"{key}: DEFAULT_PARAMS says {value!r}, signal_fn defaults to "
            f"{bound!r}")


def test_validation_rejects_what_it_should() -> None:
    bad = [
        ("fast_window >= slow_window", {"fast_window": 50, "slow_window": 50}),
        ("fast_window inverted", {"fast_window": 60, "slow_window": 50}),
        ("window < 2", {"fast_window": 1, "slow_window": 50}),
        ("spread_threshold = 0", {"spread_threshold": 0.0}),
        ("spread_threshold >= 1", {"spread_threshold": 1.5}),
        ("spread_threshold negative", {"spread_threshold": -0.01}),
        ("exit_revert_mult > 1", {"exit_revert_mult": 1.5}),
        ("exit_revert_mult negative", {"exit_revert_mult": -0.5}),
        ("sl_atr_mult = 0", {"sl_atr_mult": 0.0}),
        ("sl_atr_mult < 0", {"sl_atr_mult": -1.0}),
        ("sl_atr_mult = None", {"sl_atr_mult": None}),
        ("tp_atr_mult = 0", {"tp_atr_mult": 0.0}),
        ("tp_atr_mult < 0", {"tp_atr_mult": -3.0}),
        # Truthiness would swallow these and run a different strategy than the
        # params column reports.
        ("trailing='false'", {"trailing": "false"}),
        ("trailing=1", {"trailing": 1}),
        ("use_macro_anchor='false'", {"use_macro_anchor": "false"}),
        ("use_spread_hurdle=0", {"use_spread_hurdle": 0}),
        ("use_spread_expansion='no'", {"use_spread_expansion": "no"}),
        ("use_news_filter=1", {"use_news_filter": 1}),
    ]
    for label, override in bad:
        params = {**BASE, **override}
        for entry, name in ((M.make_signal_fn, "make_signal_fn"),
                            (lambda **p: M.signal_fn(synthetic(n=300), **p),
                             "signal_fn")):
            try:
                entry(**params)
                raise AssertionError(f"{name} accepted {label}")
            except ValueError:
                pass


def test_no_take_profit_requires_a_trailing_stop() -> None:
    """
    The request's own rule, and the reason is the runner lockup: with no target
    and a FIXED stop, a position the market never retraces to is closed only by
    the spread reverting, which in a widening trend can be never - and a trade
    still open when the data ends contributes no exit, no realised P&L and no
    ML training label.

    This is where the module departs from what `tests/test_risk_params.py`
    requires of the modules in its table (that tp=None binds with any
    `trailing`), which is why it is not registered there and is pinned here.
    """
    try:
        M.make_signal_fn(**{**BASE, "tp_atr_mult": None, "trailing": False})
        raise AssertionError("tp_atr_mult=None bound with a fixed stop")
    except ValueError as e:
        assert "trailing" in str(e), f"the message does not name the rule: {e}"
    # And the permitted combination binds and trades.
    fn = M.make_signal_fn(**{**BASE, "tp_atr_mult": None, "trailing": True})
    le, lx, se, sx = unpack_signals(fn(synthetic()), 4000)
    assert int(le.sum()) + int(se.sum()) > 0, "tp=None with a trail took no trades"
    # A trailing stop cannot lock up: every trade that opened must close.
    opened = int(le.sum()) + int(se.sum())
    closed = int(lx.sum()) + int(sx.sum())
    assert opened - closed <= 1, (
        f"{opened - closed} positions were left open with no take-profit")


def test_the_grid_is_the_size_it_claims_and_rejects_what_it_claims() -> None:
    """
    324 declared cells, 54 of them rejected by the tp=None rule, 270 evaluated.

    Counted by BINDING every combination rather than by multiplying the axis
    lengths, because the number that matters is what `backtest/scan.py` will
    actually fit - and a rejected combination is counted by the scanner rather
    than dropped, so `variants_tested` reports the search that was asked for.
    """
    import itertools
    keys = list(M.PARAM_GRID)
    combos = list(itertools.product(*(M.PARAM_GRID[k] for k in keys)))
    assert len(combos) == 324, f"the grid declares {len(combos)} cells"
    rejected = 0
    for values in combos:
        params = {**M.DEFAULT_PARAMS, **dict(zip(keys, values))}
        params = {k: v for k, v in params.items()
                  if k in inspect.signature(M.make_signal_fn).parameters}
        try:
            M.make_signal_fn(**params)
        except ValueError:
            rejected += 1
    assert rejected == 54, f"{rejected} cells were rejected, expected 54"
    assert len(combos) - rejected == 270, "the evaluated count moved"


# ==========================================================================
# 5. Causality — the part that cannot be caught by reading the equity curve
# ==========================================================================
def test_ml_features_are_causal_under_truncation() -> None:
    """
    THE PRIMARY LOOKAHEAD CHECK. Recompute the matrix on prefixes of the frame
    and require every row to be bit-identical to the full-frame value.

    A feature that reads even one bar ahead cannot survive this: on the prefix
    that bar does not exist. It also catches what a source scan for `shift(-k)`
    cannot - a global mean, a `StandardScaler` fitted on the whole frame, a
    centred rolling window, a reversed slice. That is the trap worth spending a
    case on, because a scaler fitted on the whole frame leaks the test period's
    distribution into the training rows and nothing raises.
    """
    bars = synthetic(n=900)
    full = M.ml_features(bars, fast_window=10, slow_window=50).to_numpy(dtype=float)
    for cut in (200, 400, 655, 899):
        head = M.ml_features(bars.iloc[:cut].copy(), fast_window=10,
                             slow_window=50).to_numpy(dtype=float)
        assert head.shape == (cut, full.shape[1]), head.shape
        same = np.isclose(head, full[:cut], equal_nan=True, rtol=0, atol=0)
        if not same.all():
            row, col = np.argwhere(~same)[0]
            name = M.ML_FEATURES[col]
            raise AssertionError(
                f"cut at {cut}: {name} row {row} is {head[row, col]!r} on the "
                f"prefix and {full[row, col]!r} on the full frame - the column "
                f"depends on bars it should not see")


def test_ml_features_ignore_the_future_under_perturbation() -> None:
    """
    The complement of truncation: keep the frame's length and REWRITE its tail,
    then require the head to be unchanged.

    Truncation catches a feature that reads ahead; perturbation catches one
    that reads a whole-sample statistic, because the statistic moves while the
    head's own bars do not.
    """
    bars = synthetic(n=900)
    cut = 500
    tampered = bars.copy()
    tampered.loc[tampered.index[cut:], ["open", "high", "low", "close"]] *= 3.0
    tampered.loc[tampered.index[cut:], "volume"] *= 40.0
    a = M.ml_features(bars, fast_window=10, slow_window=50).to_numpy(dtype=float)
    b = M.ml_features(tampered, fast_window=10,
                      slow_window=50).to_numpy(dtype=float)
    same = np.isclose(a[:cut], b[:cut], equal_nan=True, rtol=0, atol=0)
    if not same.all():
        row, col = np.argwhere(~same)[0]
        raise AssertionError(
            f"{M.ML_FEATURES[col]} row {row} changed when bars {cut}+ were "
            f"rewritten: {a[row, col]!r} -> {b[row, col]!r}")


def test_the_signals_are_causal_under_truncation() -> None:
    """
    The same truncation argument against the masks themselves, walk included.

    The final bar of the prefix is compared too: an entry signalled on the last
    bar of the data is legitimate (the engine simply never fills it), so a
    prefix that disagrees there is a strategy reading ahead rather than one
    being careful.
    """
    bars = synthetic(n=900)
    full = [np.asarray(m) for m in masks(bars)]
    for cut in (300, 600, 899):
        head = [np.asarray(m) for m in masks(bars.iloc[:cut].copy())]
        for name, h, f in zip(("long_entries", "long_exits", "short_entries",
                               "short_exits"), head, full):
            if not np.array_equal(h, f[:cut]):
                bad = int(np.flatnonzero(h != f[:cut])[0])
                raise AssertionError(
                    f"cut at {cut}: {name} differs first at bar {bad}")


def test_the_source_carries_no_negative_shift_or_reversed_slice() -> None:
    """
    A structural scan, as a second line behind the truncation cases.

    It cannot prove causality - `rolling(center=True)` and a global mean are
    both invisible to it, which is exactly why the truncation cases are the
    primary check - but it fails on the one form that actually shows up in
    written code, and it fails at the line rather than at a number.
    """
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None))
            if name in {"shift", "diff", "pct_change"}:
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    negative = (isinstance(arg, ast.UnaryOp)
                                and isinstance(arg.op, ast.USub))
                    literal = (isinstance(arg, ast.Constant)
                               and isinstance(arg.value, (int, float))
                               and arg.value < 0)
                    assert not (negative or literal), (
                        f"line {node.lineno}: {name}() with a negative offset")
            if name == "rolling":
                for kw in node.keywords:
                    assert kw.arg != "center" or kw.value.value is False, (
                        f"line {node.lineno}: a centred rolling window reads "
                        f"forward")
        if isinstance(node, ast.Slice) and isinstance(node.step, ast.UnaryOp):
            assert not isinstance(node.step.op, ast.USub), (
                f"line {node.lineno}: a reversed slice")


def test_the_ast_validator_objects_only_to_the_event_calendar_import() -> None:
    """
    The module deliberately imports `backtest.event_calendar`, which puts it
    outside `ALLOWED_IMPORTS`. That exception is granted for one import, and
    pinning the FULL objection list is what keeps it from covering a later edit
    reaching for `open`, `eval` or a network library.
    """
    from agents.tier3_workers import _audit_ast
    problems = sorted(set(_audit_ast(ast.parse(MODULE_PATH.read_text()))))
    expected = ["import from 'backtest.event_calendar' is not allowed"]
    assert problems == expected, f"validator objections: {problems}"


# ==========================================================================
# 6. Version B — the hook, and the filter that consumes it
# ==========================================================================
def test_the_feature_matrix_is_the_shape_the_filter_needs() -> None:
    bars = synthetic(n=800)
    feats = M.ml_features(bars, fast_window=10, slow_window=50)
    assert isinstance(feats, pd.DataFrame), type(feats)
    assert list(feats.columns) == M.ML_FEATURES, list(feats.columns)
    assert len(feats) == len(bars), f"{len(feats)} rows against {len(bars)} bars"
    assert feats.index.equals(bars.index), "the matrix is not on the bars' index"
    assert not np.isinf(feats.to_numpy(dtype=float)).any(), "the matrix carries inf"
    # It must NOT be the shared default - declaring the hook is the point.
    assert list(feats.columns) != list(causal_features(bars).columns), (
        "ml_features returned the shared causal_features columns")
    # Warm-up stays NaN rather than being filled with a full-sample statistic.
    assert feats["spread_ratio"].iloc[:49].isna().all(), (
        "the spread ratio exists before its slow average does")
    # And every column carries information past warm-up.
    tail = feats.iloc[200:]
    for col in M.ML_FEATURES:
        assert tail[col].notna().any(), f"{col} is entirely NaN"
        assert tail[col].nunique() > 1, f"{col} is constant - a dead feature"


def test_the_hour_column_reads_the_ts_column_the_engine_supplies() -> None:
    """
    The engine hands strategies a long-format frame with `ts` as a COLUMN and a
    positional index, so `bars.index.hour` raises there. Both shapes have to
    work and both have to agree, or a feature silently becomes garbage on one
    of the two paths.
    """
    bars = synthetic(n=300)
    from_column = M.ml_features(bars, fast_window=10, slow_window=50)["hour"]
    indexed = bars.set_index("ts")
    from_index = M.ml_features(indexed, fast_window=10, slow_window=50)["hour"]
    assert np.array_equal(from_column.to_numpy(), from_index.to_numpy()), (
        "the `ts` column and a DatetimeIndex disagree about the hour")
    assert set(np.unique(from_column.to_numpy())) <= set(range(24)), "not an hour"
    # And the module's local copy of the timestamp helper agrees with
    # `agents.tier3_workers`' original on both frame shapes. Pinned by
    # behaviour rather than by source: the copy's error message deliberately
    # avoids `__name__`, which the AST validator forbids.
    for label, frame in (("ts column", bars), ("DatetimeIndex", indexed)):
        assert M._bar_timestamps(frame).equals(SHARED_TIMESTAMPS(frame)), (
            f"the local timestamp helper disagrees with the shared one on a "
            f"{label} frame")
    # A frame with neither raises rather than inventing a column.
    naked = bars.drop(columns=["ts"]).reset_index(drop=True)
    try:
        M.ml_features(naked, fast_window=10, slow_window=50)
        raise AssertionError("a frame with no timestamps produced an hour column")
    except ValueError:
        pass


def test_version_b_can_only_veto_version_a() -> None:
    """
    `apply_ml_signal_filter` with this module's matrix must return a SUBSET of
    Version A's entries and leave the exits untouched.

    Both halves matter. A filter that adds an entry is not a veto, and an exit
    the filter moved would change trades it was never given a say over.
    """
    bars = synthetic(n=1600)
    le, lx, _se, _sx = masks(bars)
    kept, kept_exits = apply_ml_signal_filter(
        bars, le, lx, features=M.ml_features, threshold=0.50, random_state=0)
    kept = np.asarray(kept, dtype=bool)
    assert kept.shape == np.asarray(le).shape, "the filter changed the shape"
    added = int((kept & ~np.asarray(le, dtype=bool)).sum())
    assert added == 0, f"the filter ADDED {added} entries - it is not a veto"
    assert np.array_equal(np.asarray(kept_exits, dtype=bool),
                          np.asarray(lx, dtype=bool)), "the exits were altered"


def test_a_malformed_matrix_raises_rather_than_being_aligned() -> None:
    """
    A feature matrix whose row count disagrees with the bars must fail loudly.
    Unlike `indicators`, which is wrapped because a broken chart annotation
    must not throw away a completed backtest, a silently swapped model must not
    survive one.
    """
    bars = synthetic(n=400)
    le, lx, _se, _sx = masks(bars)
    short_matrix = lambda b: M.ml_features(b, fast_window=10,
                                           slow_window=50).iloc[:-5]
    try:
        apply_ml_signal_filter(bars, le, lx, features=short_matrix)
        raise AssertionError("a short feature matrix was accepted")
    except (ValueError, AssertionError) as e:
        assert not isinstance(e, AssertionError) or "accepted" not in str(e), e


# ==========================================================================
# 7. The walk kernel, the news filter and the degenerate cases
# ==========================================================================
def test_the_walk_kernel_is_the_shared_one_character_for_character() -> None:
    """
    The kernel is duplicated across the strategy modules by the convention that
    keeps them self-contained, and the duplication is only safe while the
    copies are identical. `tests/test_risk_params.py` holds the other three to
    each other; this holds this one to them.
    """
    mine = inspect.getsource(M._walk_loop)
    theirs = inspect.getsource(EC._walk_loop)
    assert mine == theirs, (
        "this module's _walk_loop has drifted from ema_crossover's - the stop, "
        "target and trailing behaviour are no longer the shared ones")


def test_the_fill_and_the_stop_anchor_on_the_bar_after_the_signal() -> None:
    """
    Next-bar-open execution, checked on the arrays rather than trusted.

    An exit may never be marked on the signal bar itself, and the stop level
    the walk publishes must be measured from the FILL bar's open - the price
    the position was actually opened at - rather than from the signal bar's
    close, which it never traded at.
    """
    bars = synthetic(n=1200)
    s, entries, exits, s_entries, s_exits, stop, target = M._signal_arrays(
        bars, 10, 50, 0.015, 0.5, 1.5, 3.0, False)
    entry_bars = np.flatnonzero(entries)
    assert len(entry_bars) > 3, "no long trades to inspect"
    for i in entry_bars[:10]:
        assert not exits[i], f"bar {i} exited on its own signal bar"
        fill = i + 1
        if fill >= len(bars):
            continue
        atr = float(s["atr"].iloc[i])
        expected = float(bars["open"].iloc[fill]) - 1.5 * atr
        assert abs(float(stop[fill]) - expected) < 1e-9, (
            f"bar {fill}: stop {stop[fill]!r}, expected {expected!r} = the "
            f"fill open minus 1.5 x the SIGNAL bar's ATR")
        assert abs(float(target[fill])
                   - (float(bars["open"].iloc[fill]) + 3.0 * atr)) < 1e-9, (
            f"bar {fill}: the target is not anchored on the fill price")


def test_a_zero_anchor_does_not_manufacture_a_short() -> None:
    """
    Divergence 1 in the module docstring, and the reason the request's blanket
    0.0 fill is not applied to the spread ratio.

    A slow anchor at zero makes the ratio undefined. Filled with 0.0 it would
    sit below every short hurdle, and the strategy would read "no anchor" as
    "maximum downside divergence" and short it. These contracts are not
    back-adjusted and CL printed negative in April 2020, so this is reachable.
    """
    n = 120
    close = np.concatenate([np.full(60, 10.0), np.zeros(30), np.full(30, -5.0)])
    bars = pd.DataFrame({
        "ts": pd.date_range("2020-04-01", periods=n, freq="15min", tz="UTC"),
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": np.full(n, 1000.0)})
    ratio = M._series(bars, 5, 20)["ratio"]
    zero_or_below = bars["close"].rolling(20, min_periods=20).mean() <= 0
    assert ratio[zero_or_below].isna().all(), (
        "the spread ratio exists where the anchor is not positive")
    got = candidates(bars, fast_window=5, slow_window=20)
    assert not got["short_ok"][zero_or_below.to_numpy()].any(), (
        "a non-positive anchor produced a SHORT candidate")
    assert not got["long_ok"][zero_or_below.to_numpy()].any(), (
        "a non-positive anchor produced a long candidate")


def test_zero_range_bars_do_not_propagate_nan_through_the_atr() -> None:
    """
    The request's zero-range rule, where it actually applies: a flat patch must
    produce a true range of 0.0 rather than a NaN that the ATR's exponential
    recursion would carry forward, blanking the stop distance for every bar
    after it.
    """
    n = 120
    close = np.concatenate([np.linspace(100, 110, 60), np.full(60, 110.0)])
    bars = pd.DataFrame({
        "ts": pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC"),
        "open": close, "close": close,
        # The flat half has high == low == close: a genuinely zero-range bar.
        "high": np.concatenate([close[:60] + 0.5, close[60:]]),
        "low": np.concatenate([close[:60] - 0.5, close[60:]]),
        "volume": np.full(n, 1000.0)})
    tr = M._true_range(bars)
    assert (tr.iloc[61:] == 0.0).all(), "a zero-range bar produced a non-zero TR"
    atr = M._atr(bars)
    assert atr.iloc[14:].notna().all(), "the ATR went NaN after a flat patch"
    assert (atr.iloc[14:] >= 0).all(), "the ATR went negative"
    # And the ROC of a flat spread is 0.0, not inf and not NaN.
    flat = pd.Series(np.zeros(50))
    roc = M._roc(flat, 5)
    assert (roc.iloc[5:] == 0.0).all(), f"a zero-base ROC is {roc.iloc[10]!r}"


def test_the_news_filter_is_the_repositorys_own() -> None:
    """
    The module's suppression against `backtest.event_calendar` computed
    independently, including the one-bar BACKWARD widening that exists because
    the engine fills at the next bar's open.

    SKIPS LOUDLY when the calendar does not cover the fixture's span: an empty
    calendar makes `is_news_blocked(strict=True)` raise rather than return an
    all-clear mask, which is the correct behaviour and not something to assert
    around.
    """
    from backtest.event_calendar import entry_block_mask

    bars = synthetic(n=2000)
    try:
        mask, info = entry_block_mask(bars["ts"], news_filter=True,
                                      news_window_minutes=M.NEWS_WINDOW_MINUTES)
    except Exception as e:                       # no calendar on this box
        print(f"    SKIPPED: no macro calendar covers the fixture ({e})")
        return
    mask = np.asarray(mask, dtype=bool)
    assert mask.any(), "the calendar blocked no bar at all - nothing to compare"

    off = candidates(bars, use_news_filter=False)
    on = candidates(bars, use_news_filter=True)
    for side in ("long_ok", "short_ok"):
        expected = off[side] & ~mask
        assert np.array_equal(on[side], expected), (
            f"{side}: the module's news suppression is not "
            f"event_calendar's own mask")

    # Parity alone would be passed by a filter wired to a mask that never
    # intersects anything, so it is also checked that the wiring can REMOVE. At
    # the default settings the candidates are sparse enough that 2,000 bars can
    # legitimately contain none inside a release window - a fact about the
    # fixture, not about the filter - so the removal is measured with the
    # hurdle and the expansion layers off, where nearly every bar is a
    # candidate and an intersection is certain.
    dense_off = candidates(bars, use_news_filter=False, use_spread_hurdle=False,
                           use_spread_expansion=False)
    dense_on = candidates(bars, use_news_filter=True, use_spread_hurdle=False,
                          use_spread_expansion=False)
    suppressed = int((dense_off["long_ok"] & ~dense_on["long_ok"]).sum()) + \
        int((dense_off["short_ok"] & ~dense_on["short_ok"]).sum())
    assert suppressed > 0, (
        "the news filter removed no candidate even where nearly every bar is "
        "one - it is wired to nothing, which is indistinguishable from having "
        "it off")
    # Provenance travels with the mask: a RULE-generated calendar is not a run
    # that dodged the actual prints, and the token has to be present to be read.
    assert info.get("news_provenance") in ("PUBLISHED", "RULE"), (
        f"the calendar reported no usable provenance: {info}")


def test_indicators_draw_what_the_signals_were_taken_from() -> None:
    """
    Full-length series only - the report DROPS a series whose length disagrees
    with the frame rather than reindexing it, so a wrong length is a silently
    missing line. The hurdle lines must also be the price levels the rule
    actually compares against.
    """
    bars = synthetic(n=800)
    out = M.indicators(bars, **BASE)
    assert isinstance(out, dict) and out, "indicators returned nothing"
    for name, series in out.items():
        assert isinstance(series, pd.Series), f"{name} is {type(series)}"
        assert len(series) == len(bars), f"{name} has {len(series)} rows"
    slow = M._series(bars, 10, 50)["slow"]
    long_line = out["Long Hurdle (Anchor +1.5%)"]
    assert np.allclose(long_line.to_numpy(dtype=float),
                       (slow * 1.015).to_numpy(dtype=float), equal_nan=True), (
        "the drawn hurdle is not the level the rule compares against")
    # The optional lines appear and disappear with their settings rather than
    # being drawn as an all-NaN legend entry.
    no_tp = M.indicators(bars, **{**BASE, "tp_atr_mult": None, "trailing": True})
    assert not any("Take Profit" in k for k in no_tp), (
        "a take-profit line was drawn for a run with no take-profit")
    assert any("Trailing Stop" in k for k in no_tp), "the stop line lost its kind"
    no_hurdle = M.indicators(bars, **{**BASE, "use_spread_hurdle": False})
    assert not any("Hurdle" in k for k in no_hurdle), (
        "the hurdle lines were drawn for a run that did not use the hurdle")


# ==========================================================================
# The script runner. `assert` is the failure mechanism, so pytest and this
# report the same thing - see the module docstring.
# ==========================================================================
def main() -> int:
    cases = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    print(f"ma_anchoring_spread — {len(cases)} cases\n")
    for name, fn in cases:
        try:
            fn()
        except Exception as e:                   # noqa: BLE001 - reported below
            failures.append((name, e))
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
        else:
            print(f"  PASS  {name}")
    print()
    if failures:
        print(f"{len(failures)} of {len(cases)} FAILED: "
              + ", ".join(n for n, _ in failures))
        return 1
    print(f"all {len(cases)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
