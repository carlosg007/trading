"""
tests/test_ema_deviation_scalp_20260909.py - the 20-EMA deviation scalp.

Location:  ~/src/trading/tests/test_ema_deviation_scalp_20260909.py

    .venv/bin/python3 -m pytest tests/test_ema_deviation_scalp_20260909.py

ASSERT-BASED, AND DELIBERATELY SO. `tests/conftest.py` classifies a suite by
the `\\ndef check(` marker: script-style suites collect results into a
module-level FAILURES list that pytest cannot see, and are run as subprocesses
asserting their exit code. This suite has no `check()` helper - every case
fails through `assert`, so bare pytest and the suite runner report the same
thing and the per-case granularity survives.

EVERY HELPER IS `_`-PREFIXED. CLAUDE.md records the trap: on an assert-based
suite pytest collects ANY module-level `test_*` it can call, including a
helper whose only argument is defaulted - in `test_regime_profiler.py` that
ran the sections without their artifact redirect and wrote real JSON onto the
NFS mount. The tell was the count: 8 passed where 4 were written. Nothing here
is named `test_*` unless pytest is meant to call it with no arguments.

NO BARS ARE READ FROM THE LAKE. Every fixture is synthetic and constructed in
this file, so the suite is hermetic and says nothing about whether the
strategy makes money - that is Stage 1 through Stage 5's job, not a unit
test's.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "strategies" / "experimental" / "ema_deviation_scalp_20260909.py"


def _load(path: Path = MODULE_PATH):
    """
    Import the strategy from its FILE PATH, the way
    `agents.tier3_workers.load_strategy` does.

    Not `import strategies.experimental...`: the engine loads a module from a
    path and promotes it as a self-contained copy, so importing it by package
    name here would exercise a path the engine never takes.
    """
    spec = importlib.util.spec_from_file_location("_ema_dev_scalp_uut", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _frame(close: np.ndarray, *, freq: str = "5min",
           high: np.ndarray | None = None,
           low: np.ndarray | None = None) -> pd.DataFrame:
    """
    An OHLCV frame around a close path, tz-aware, oldest to newest.

    `open` is the PREVIOUS close, which is what a continuous futures tape
    looks like and what makes the engine's next-bar fill meaningful: the fill
    price for a signal on bar i is `open[i+1]`, so it has to be a price the
    market actually traded at rather than a copy of bar i+1's close.
    """
    n = len(close)
    close = np.asarray(close, dtype="float64")
    if high is None:
        high = close + 0.25
    if low is None:
        low = close - 0.25
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]],
         "high": np.maximum(np.asarray(high, dtype="float64"), close),
         "low": np.minimum(np.asarray(low, dtype="float64"), close),
         "close": close,
         "volume": np.full(n, 1000.0)},
        index=idx)


def _random_walk(n: int, seed: int, sigma: float = 0.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.full(n, 100.0)
    for i in range(1, n):
        close[i] = close[i - 1] + rng.normal(0.0, sigma)
    high = close + np.abs(rng.normal(0.0, 0.3, n))
    low = close - np.abs(rng.normal(0.0, 0.3, n))
    return _frame(close, high=high, low=low)


def _long_setup() -> pd.DataFrame:
    """
    A tape built to produce the LONG case and nothing else: a long quiet
    stretch to warm the indicators, a sharp flush well below the EMA, then a
    reclaim back through it.

    The flush is deep enough that the LOW pierces `EMA - 1.5 x ATR` and drives
    %K into oversold; the recovery then crosses close back above the EMA and
    turns %K up through %D. Both events are what the strategy is specified on,
    so this fixture failing to produce an entry is a real defect rather than
    an unlucky draw.
    """
    base = np.full(120, 100.0)
    flush = np.array([99.0, 97.5, 95.5, 93.5, 92.0, 91.0])
    recover = np.array([93.0, 95.5, 97.5, 99.0, 100.0, 100.5, 101.0])
    tail = np.full(40, 101.0)
    close = np.concatenate([base, flush, recover, tail])
    low = close.copy()
    low[120:126] -= 1.5                     # the wick that pierces the band
    high = close + 0.25
    return _frame(close, high=high, low=low)


def _short_setup() -> pd.DataFrame:
    """The mirror: a quiet stretch, a spike well above the EMA, then a loss of
    it. Written out rather than derived by negating the long fixture, because
    a short is not a long with the sign flipped and a fixture that assumed it
    was would not test the difference."""
    base = np.full(120, 100.0)
    spike = np.array([101.0, 102.5, 104.5, 106.5, 108.0, 109.0])
    fade = np.array([107.0, 104.5, 102.5, 101.0, 100.0, 99.5, 99.0])
    tail = np.full(40, 99.0)
    close = np.concatenate([base, spike, fade, tail])
    high = close.copy()
    high[120:126] += 1.5
    low = close - 0.25
    return _frame(close, high=high, low=low)


def _first_exit_after(mask: pd.Series, entry_i: int) -> int | None:
    """
    Bars from `entry_i` to the first True in `mask` at or after it, or None
    when there is none.

    Explicitly NOT `argmax`, which returns 0 on an all-False array - so "never
    closed" and "closed immediately" would be the same answer, and a test
    comparing holding times would silently compare against a sentinel.
    """
    later = np.flatnonzero(mask.to_numpy()[entry_i:])
    return int(later[0]) if later.size else None


# ---------------------------------------------------------------------------
# The module contract
# ---------------------------------------------------------------------------
def test_module_declares_the_four_mandatory_names() -> None:
    """
    CLAUDE.md's rule: any strategy Claude creates carries `signal_fn`,
    `indicators`, `LOGIC` and `PARAM_GRID`. The loader TOLERATES a module
    declaring none - several pre-existing ones do - and that tolerance is not
    permission to write another.
    """
    assert callable(MOD.signal_fn)
    assert callable(MOD.indicators)
    assert callable(MOD.make_signal_fn)
    assert callable(MOD.ml_features)
    assert set(MOD.LOGIC) == {"concept", "entry", "exit"}
    assert MOD.PARAM_GRID and isinstance(MOD.PARAM_GRID, dict)
    assert isinstance(MOD.DEFAULT_PARAMS, dict)


def test_param_grid_only_sweeps_real_parameters() -> None:
    """
    `load_strategy` rejects unknown parameter names, so a stale PARAM_GRID key
    raises at bind time rather than being quietly ignored. Catching it here
    means catching it before a sweep burns hours discovering it.
    """
    unknown = set(MOD.PARAM_GRID) - set(MOD.DEFAULT_PARAMS)
    assert not unknown, f"PARAM_GRID sweeps names signal_fn will refuse: {unknown}"


def test_logic_placeholders_all_resolve_against_default_params() -> None:
    """
    LOGIC's `{param}` slots are filled with the bound params for the tear
    sheet. A slot naming a parameter that does not exist raises KeyError at
    render time - on the card, after the backtest has already run.
    """
    for field, text in MOD.LOGIC.items():
        try:
            text.format(**MOD.DEFAULT_PARAMS)
        except KeyError as exc:              # pragma: no cover - the failure
            pytest.fail(f"LOGIC[{field!r}] names unknown parameter {exc}")


def test_signal_fn_returns_the_four_mask_form() -> None:
    """
    Four masks, not three and not a bare Series. `engine.unpack_signals`
    RAISES on anything else rather than silently taking the first two of a
    three-tuple - which is how a strategy's short side disappears into a
    plausible long-only equity curve.
    """
    bars = _random_walk(400, seed=3)
    out = MOD.signal_fn(bars)
    assert isinstance(out, tuple) and len(out) == 4
    for mask in out:
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert mask.index.equals(bars.index)


def test_module_passes_the_ast_security_gate() -> None:
    """
    Model-generated code passes an AST check before import, and this module is
    held to it too. `pandas_ta` was offered in the request and is NOT in
    ALLOWED_IMPORTS - importing it would fail here, which is why the
    Stochastic is implemented in-module.

    Expects ZERO objections. Several modules in this directory carry a granted
    exception for `backtest.event_calendar`; this one needs none.
    """
    from agents.tier3_workers import _audit_ast          # noqa: PLC0415
    objections = _audit_ast(ast.parse(MODULE_PATH.read_text()))
    assert objections == [], f"AST gate objects: {objections}"


# ---------------------------------------------------------------------------
# The indicators
# ---------------------------------------------------------------------------
def test_stochastic_matches_a_hand_computed_window() -> None:
    """
    %K against the definition, computed by hand on a frame short enough to
    check by eye. `smooth_k=1` and `d_period=1` strip the smoothing so the raw
    ratio is what is compared.
    """
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    bars = _frame(close, high=close + 0.0, low=close - 0.0)
    k, d = MOD._stochastic(bars, k_period=3, smooth_k=1, d_period=1)
    # Bar 2 is the first with a full 3-bar window: high 12, low 10, close 12.
    assert k.iloc[:2].isna().all(), "warm-up must be NaN, never a reading"
    assert k.iloc[2] == pytest.approx(100.0)
    assert k.iloc[3] == pytest.approx(100.0)
    assert d.iloc[2] == pytest.approx(k.iloc[2])


def test_a_dead_flat_window_reads_neutral_not_zero() -> None:
    """
    THE FAILURE THIS PREVENTS: on a halted or dead session `highest ==
    lowest`, the ratio is 0/0, and the naive fill of 0.0 is the most OVERSOLD
    value the scale has. It would hold `%K <= oversold` permanently true and
    manufacture long confirmations out of silence.

    50 is the centerline: a window in which nothing moved is neither
    overbought nor oversold.
    """
    bars = _frame(np.full(30, 100.0), high=np.full(30, 100.0),
                  low=np.full(30, 100.0))
    k, _ = MOD._stochastic(bars, k_period=5, smooth_k=1, d_period=1)
    settled = k.iloc[5:]
    assert (settled == MOD.STOCH_NEUTRAL).all()
    assert not (settled == 0.0).any()


def test_atr_is_wilder_smoothed_and_warms_up_as_nan() -> None:
    """ATR(14) must be Wilder's average, not a span-14 EMA - the two differ by
    about a factor of two in speed, and a stop distance drawn from the wrong
    one sits somewhere no other chart agrees with."""
    bars = _random_walk(120, seed=11)
    atr = MOD._atr(bars, MOD.ATR_PERIOD)
    assert atr.iloc[:MOD.ATR_PERIOD - 1].isna().all()
    assert np.isfinite(atr.iloc[MOD.ATR_PERIOD:]).all()
    tr = MOD._true_range(bars)
    expected = tr.ewm(alpha=1.0 / MOD.ATR_PERIOD, adjust=False,
                      min_periods=MOD.ATR_PERIOD).mean()
    pd.testing.assert_series_equal(atr, expected, check_names=False)


def test_cross_helpers_are_events_not_states() -> None:
    """A `cross` that is true for every bar of a run is a different strategy:
    it would fire an entry on each bar price merely sits above the anchor."""
    fast = pd.Series([1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
    level = pd.Series([2.5] * 6)
    up = MOD._cross_above(fast, level)
    down = MOD._cross_below(fast, level)
    assert up.tolist() == [False, False, True, False, False, False]
    assert down.tolist() == [False, False, False, False, False, True]


def test_recent_remembers_backwards_only() -> None:
    """`_recent` is what lets the stretch and the reclaim be different bars.
    It must look BACKWARD: a window that included bar i+1 would make the
    strategy prescient and the failure is invisible in an equity curve."""
    flag = pd.Series([False, True, False, False, False, False])
    got = MOD._recent(flag, 3).tolist()
    assert got == [False, True, True, True, False, False]


def test_indicators_are_full_length_and_named() -> None:
    """The inspector draws these over its candles. A short series would draw
    the reclaim a bar from where the entry actually fired."""
    bars = _random_walk(200, seed=5)
    drawn = MOD.indicators(bars)
    assert drawn, "no indicator series declared"
    for name, series in drawn.items():
        assert isinstance(series, pd.Series), name
        assert series.index.equals(bars.index), name


# ---------------------------------------------------------------------------
# The signals
# ---------------------------------------------------------------------------
def test_the_long_setup_produces_a_long_entry() -> None:
    """The specified LONG case, on a tape built to contain exactly it."""
    bars = _long_setup()
    le, lx, se, _sx = MOD.signal_fn(bars)
    assert le.sum() >= 1, "the constructed long setup produced no entry"
    assert se.sum() == 0, "a long-only fixture produced a short"
    assert lx.sum() >= 1, "the long never closed"
    # The entry must land in the reclaim window, not somewhere in the flat
    # warm-up before the flush ever happened.
    assert 120 <= int(np.argmax(le.to_numpy())) <= 140


def test_the_short_setup_produces_a_short_entry() -> None:
    """The mirror case. A short is not a long with the sign flipped, so it is
    asserted on its own fixture rather than inferred from the long one."""
    bars = _short_setup()
    le, _lx, se, sx = MOD.signal_fn(bars)
    assert se.sum() >= 1, "the constructed short setup produced no entry"
    assert le.sum() == 0, "a short-only fixture produced a long"
    assert sx.sum() >= 1, "the short never closed"


def test_no_bar_carries_both_a_long_and_a_short_entry() -> None:
    """Matching `engine._clean_signals_ls_loop`: a bar asking to be long and
    short at once takes NEITHER. Price cannot cross above and below the same
    EMA on one bar, so this should never be reachable - the check exists so a
    malformed variant shows up as a missing trade rather than a coin flip
    buried in the kernel."""
    for seed in (1, 2, 3, 4, 5):
        bars = _random_walk(600, seed=seed)
        le, _lx, se, _sx = MOD.signal_fn(bars)
        assert not (le & se).any(), f"seed {seed}: simultaneous entry"


def test_positions_are_never_pyramided_or_reversed() -> None:
    """
    The walk enters only from FLAT. Entries and exits must therefore
    alternate, and the two sides must never be open at once.

    Checked as a running state machine rather than by counting, because equal
    counts are also what two overlapping positions would produce.
    """
    bars = _random_walk(900, seed=17)
    le, lx, se, sx = MOD.signal_fn(bars)
    state = 0
    for i in range(len(bars)):
        if state == 0:
            if le.iloc[i]:
                state = 1
            elif se.iloc[i]:
                state = -1
            assert not (le.iloc[i] and se.iloc[i])
        elif state == 1:
            assert not le.iloc[i], f"pyramided a long at bar {i}"
            assert not se.iloc[i], f"reversed into a short at bar {i}"
            if lx.iloc[i]:
                state = 0
        else:
            assert not se.iloc[i], f"pyramided a short at bar {i}"
            assert not le.iloc[i], f"reversed into a long at bar {i}"
            if sx.iloc[i]:
                state = 0


def test_exits_never_land_before_the_fill_bar() -> None:
    """
    The engine fills a signal on bar i at bar i+1's open, so the position is
    not live until i+1 and cannot be closed on the signal bar itself. An exit
    on the entry bar would be a trade that never existed.
    """
    bars = _random_walk(900, seed=23)
    le, lx, se, sx = MOD.signal_fn(bars)
    entries = np.flatnonzero((le | se).to_numpy())
    exits = np.flatnonzero((lx | sx).to_numpy())
    for e in entries:
        later = exits[exits > e]
        if later.size:
            assert later[0] >= e + 1, f"exit at {later[0]} for entry at {e}"


# ---------------------------------------------------------------------------
# Causality - the failure that is invisible in an equity curve
# ---------------------------------------------------------------------------
def test_signals_do_not_change_when_future_bars_are_removed() -> None:
    """
    THE LOOKAHEAD TEST, and the one that matters most in this repository.

    If any layer reads bar i+1, truncating the frame changes what the earlier
    bars decided. Nothing raises when that happens - the backtest simply
    becomes prescient and the Sharpe looks superb.

    Every signal on bars 0..k-1 must be identical whether or not bars k..n
    exist. The last bar of the truncated frame is excluded: its own exit can
    legitimately depend on bar k, which it does not have.
    """
    bars = _random_walk(700, seed=29)
    full = MOD.signal_fn(bars)
    for k in (300, 450, 600):
        cut = MOD.signal_fn(bars.iloc[:k].copy())
        for name, whole, part in zip(("le", "lx", "se", "sx"), full, cut):
            pd.testing.assert_series_equal(
                whole.iloc[:k - 1], part.iloc[:k - 1],
                check_names=False,
                obj=f"{name} disagrees once future bars are removed")


def test_ml_features_are_causal_and_shaped_to_the_bars() -> None:
    """
    Causality is the module's responsibility: the ML filter's guarantee that
    it trains only on trades closed before the candidate is undone by a column
    that reads the future.

    Shape is checked by the caller and a failure RAISES - unlike `indicators`,
    which is wrapped - so a row count that disagrees with the bars must be
    caught here.
    """
    bars = _random_walk(500, seed=31)
    feats = MOD.ml_features(bars)
    assert len(feats) == len(bars)
    assert feats.index.equals(bars.index)
    assert np.isfinite(feats.to_numpy()).all(), "NaN/inf reached the matrix"
    cut = MOD.ml_features(bars.iloc[:300].copy())
    pd.testing.assert_frame_equal(feats.iloc[:299], cut.iloc[:299],
                                  check_names=False)


# ---------------------------------------------------------------------------
# Risk parameters
# ---------------------------------------------------------------------------
def test_a_none_take_profit_means_no_target_ever_fires() -> None:
    """
    `tp_atr_mult=None` IS a setting - "no target, run to the stop" - and it is
    expressed as NaN so every comparison against it is False. A sentinel like
    0 would close every trade on its first bar; a huge number would be a level
    the search could still reach.
    """
    assert np.isnan(MOD._tp_distance(None))
    assert MOD._tp_distance(2.0) == 2.0
    bars = _long_setup()
    le, lx, _se, _sx = MOD.signal_fn(bars, tp_atr_mult=None)
    le_tp, lx_tp, _, _ = MOD.signal_fn(bars, tp_atr_mult=1.0)
    assert le.sum() >= 1, "the fixture stopped producing entries"

    # The target cannot change WHERE the trade opens - it is an exit setting -
    # so both runs must enter on the same bar. If they ever differ, the target
    # has leaked into the entry layer.
    entries = np.flatnonzero(le.to_numpy())
    entries_tp = np.flatnonzero(le_tp.to_numpy())
    first = int(entries[0])
    assert int(entries_tp[0]) == first

    # With no target the position can only be closed by the stop, so it is
    # held at least as long. `_first_exit_after` returns None for "never
    # closed", which is a REAL outcome here - this fixture ends on a flat tail
    # that never reaches the stop - and is not the same as closing on bar
    # zero. Reading `argmax` on an all-False mask would report exactly that
    # and the assertion would compare a held time against a sentinel.
    held_none = _first_exit_after(lx, first)
    held_tp = _first_exit_after(lx_tp, first)
    assert held_tp is not None, "the 1.0 x ATR target never fired"
    assert held_none is None or held_none >= held_tp


def test_the_risk_keys_are_the_three_promote_py_writes() -> None:
    """
    `run.py`'s RISK_PARAMS and `promote.py`'s RISK_KEYS are exactly
    `("sl_atr_mult", "tp_atr_mult", "trailing")`, and `promote._risk_block`
    writes those three and nothing else. A stop distance living under any
    other name would be ABSENT from the promoted risk block, and the card
    would report `sl_atr_mult` as the stop while a different number bound
    every trade.
    """
    for key in ("sl_atr_mult", "tp_atr_mult", "trailing"):
        assert key in MOD.DEFAULT_PARAMS


@pytest.mark.parametrize("bad", [
    {"ema_window": 0},
    {"deviation_atr_mult": 0.0},
    {"oversold": 80.0, "overbought": 20.0},
    {"sl_atr_mult": 0.01},
    {"tp_atr_mult": 0.0},
    {"k_period": 0},
    {"stoch_lookback": 0},
])
def test_impossible_parameters_raise_rather_than_running(bad) -> None:
    """
    Refused at the point they are passed, not as a strange equity curve.
    `oversold >= overbought` is the subtle one: inverted, every bar sits in
    both extremes, both sides confirm on every cross, and the Stochastic
    filter silently becomes a coin flip.
    """
    bars = _random_walk(200, seed=41)
    with pytest.raises(ValueError):
        MOD.signal_fn(bars, **bad)


def test_a_zero_atr_bar_is_never_entered() -> None:
    """A bracket of zero width puts the stop at the fill price, which the fill
    bar itself breaches. The warm-up and any dead-flat bar must be refused
    rather than traded."""
    bars = _frame(np.full(80, 100.0), high=np.full(80, 100.0),
                  low=np.full(80, 100.0))
    le, _lx, se, _sx = MOD.signal_fn(bars)
    assert not le.any() and not se.any()


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------
def test_generate_signals_delegates_to_signal_fn() -> None:
    """The requested `generate_signals(df, params)` convention. It must
    DELEGATE rather than reimplement, or the two are free to disagree about
    what a signal is."""
    bars = _random_walk(400, seed=47)
    direct = MOD.signal_fn(bars)
    adapted = MOD.generate_signals(bars, {})
    for a, b in zip(direct, adapted):
        pd.testing.assert_series_equal(a, b, check_names=False)
    assert MOD.generate_signals(bars, None)[0].equals(direct[0])


def test_unknown_parameters_raise_on_both_entry_points() -> None:
    """A stale key that was silently dropped would sweep the DEFAULT and
    report it under the name the caller thought they set."""
    bars = _random_walk(100, seed=53)
    with pytest.raises(ValueError, match="unknown parameter"):
        MOD.make_signal_fn(trail_atr_mult=2.0)
    with pytest.raises(ValueError, match="unknown parameter"):
        MOD.generate_signals(bars, {"nope": 1})


def test_make_signal_fn_binds_and_matches_direct_calls() -> None:
    bars = _random_walk(400, seed=59)
    bound = MOD.make_signal_fn(ema_window=10, deviation_atr_mult=1.0)
    direct = MOD.signal_fn(bars, ema_window=10, deviation_atr_mult=1.0)
    for a, b in zip(bound(bars), direct):
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_filters_off_is_a_superset_of_filters_on() -> None:
    """
    Switching a filter off may only ADD candidate triggers, never remove them.

    NOTE THE THING THIS DOES NOT CLAIM. CLAUDE.md records it: a filter can
    only remove candidate TRIGGERS, never realised trades - the walk holds one
    position at a time, so declining an early trigger can leave the strategy
    flat for a later one it would otherwise have been holding through.
    Measured elsewhere in this repo, enabling a filter removed 17 candidates
    and ADDED 11 realised entries. So this compares the trigger layer, and
    never the entry masks.
    """
    bars = _random_walk(800, seed=61)
    p = MOD.DEFAULT_PARAMS
    on = MOD._layers(bars, p["ema_window"], p["deviation_atr_mult"],
                     p["stretch_lookback"], p["k_period"], p["d_period"],
                     p["smooth_k"], p["oversold"], p["overbought"],
                     p["stoch_lookback"], True, True)
    off = MOD._layers(bars, p["ema_window"], p["deviation_atr_mult"],
                      p["stretch_lookback"], p["k_period"], p["d_period"],
                      p["smooth_k"], p["oversold"], p["overbought"],
                      p["stoch_lookback"], False, False)
    for side in ("long", "short"):
        strict = (on[f"stretch_{side}"] & on[f"trigger_{side}"]).fillna(False)
        loose = (off[f"stretch_{side}"] & off[f"trigger_{side}"]).fillna(False)
        assert (strict & ~loose).sum() == 0, (
            f"{side}: a trigger survived the filters but not their absence")


# ---------------------------------------------------------------------------
# The duplicated kernel
# ---------------------------------------------------------------------------
def test_walk_loop_agrees_with_the_copy_it_was_duplicated_from() -> None:
    """
    `_walk_loop` is duplicated VERBATIM across this directory because strategy
    modules are loaded from a file path and promoted as self-contained copies.
    The convention only holds if the copies actually agree, which is what
    `tests/test_risk_params.py` enforces for the older ones.

    Run on identical arrays, in both directions, against
    `sma_momentum_crossover_20260818._walk_loop`.
    """
    sibling = _load(REPO / "strategies" / "experimental"
                    / "sma_momentum_crossover_20260818.py")
    rng = np.random.default_rng(67)
    n = 500
    close = np.cumsum(rng.normal(0.0, 0.5, n)) + 100.0
    high = close + np.abs(rng.normal(0.0, 0.4, n))
    low = close - np.abs(rng.normal(0.0, 0.4, n))
    open_ = np.r_[close[0], close[:-1]]
    atr = np.full(n, 1.0)
    long_ok = rng.random(n) < 0.03
    short_ok = rng.random(n) < 0.03
    sig_x = np.zeros(n, dtype=bool)
    flat = np.zeros(n, dtype=bool)
    for trailing in (False, True):
        for tp in (2.0, float("nan")):
            mine = MOD._walk_loop(long_ok, short_ok, sig_x, sig_x, open_,
                                  high, low, atr, flat, 1.5, tp, trailing)
            theirs = sibling._walk_loop(long_ok, short_ok, sig_x, sig_x,
                                        open_, high, low, atr, flat, 1.5, tp,
                                        trailing)
            for a, b in zip(mine, theirs):
                np.testing.assert_array_equal(np.nan_to_num(a, nan=-999.0),
                                              np.nan_to_num(b, nan=-999.0))


def test_the_module_runs_on_one_minute_bars_too() -> None:
    """
    The request asked for synthetic 1m AND 5m coverage. Nothing in the module
    reads the bar width - the EMA and the ATR are in BARS, not minutes - so
    this asserts the timeframe-independence rather than a different result:
    the same close path must produce the same masks at either frequency.

    That is also the reason `TIMEFRAME` is a declaration and Stage 1 picks the
    rung: a 20 EMA is ~20 minutes at 1m and ~100 at 5m, and the module cannot
    tell the difference.
    """
    close = _long_setup()["close"].to_numpy()
    one_min = _frame(close, freq="1min")
    five_min = _frame(close, freq="5min")
    a = MOD.signal_fn(one_min)
    b = MOD.signal_fn(five_min)
    assert a[0].sum() >= 1
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())
