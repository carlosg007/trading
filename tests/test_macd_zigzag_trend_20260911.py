"""
tests/test_macd_zigzag_trend_20260911.py - MACD + ZigZag + EMA continuation.

Location:  ~/src/trading/tests/test_macd_zigzag_trend_20260911.py

    .venv/bin/python3 -m pytest tests/test_macd_zigzag_trend_20260911.py

ASSERT-BASED, DELIBERATELY. `tests/conftest.py` classifies a suite by the
`\\ndef check(` marker: script-style suites are run as subprocesses asserting
an exit code. This suite has no `check()` helper, so bare pytest and the suite
runner report the same thing and per-case granularity survives.

NO BARS ARE READ FROM THE LAKE. Every fixture is synthetic, so this suite says
nothing about whether the strategy makes money - that is Stage 1 through 5's
job, not a unit test's.

TWO CASES CARRY MOST OF THE WEIGHT.

`test_the_zigzag_confirms_late_and_never_at_the_extreme` is the one the whole
module turns on. A ZigZag is a repainter by default: the familiar form marks a
swing high at the bar holding the highest price of a CENTRED window, which is
not known until the right half has printed. Every other case here passes for
both forms; only this one and the truncation case can tell them apart.

`test_the_walk_matches_the_shared_one_when_the_step_is_zero` holds this
module's `_walk_loop` - the only copy in the directory that is not verbatim -
to identical output against the shared implementation at `trail_step_mult=0`.
The convention exists so two copies cannot disagree about the fill timeline;
proving non-divergence is worth more than copying and hoping.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = (REPO / "strategies" / "experimental"
               / "macd_zigzag_trend_20260911.py")
SHARED_WALK_PATH = (REPO / "strategies" / "experimental"
                    / "semafor_ha_momentum_20260910.py")


def _load(path: Path, name: str):
    """Import from the FILE PATH, the way `tier3_workers.load_strategy` does -
    not by package name, which is a path the engine never takes."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load(MODULE_PATH, "_macdzz_uut")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _frame(close, *, high=None, low=None, freq: str = "15min") -> pd.DataFrame:
    """
    An OHLCV frame around a close path, tz-aware, oldest to newest.

    `open` is the PREVIOUS close, which is what makes the engine's next-bar
    fill meaningful: the fill for a signal on bar i is `open[i+1]`, so it has
    to be a price the market actually traded at.
    """
    close = np.asarray(close, dtype="float64")
    n = len(close)
    if high is None:
        high = close + 0.5
    if low is None:
        low = close - 0.5
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]],
         "high": np.maximum(np.asarray(high, dtype="float64"), close),
         "low": np.minimum(np.asarray(low, dtype="float64"), close),
         "close": close,
         "volume": np.full(n, 1000.0)},
        index=idx)


def _impulse_pullback(legs: int = 60, seed: int = 7) -> pd.DataFrame:
    """
    Alternating impulse and retrace - the tape this strategy is specified for,
    and the only shape that produces ZigZag pivots at all. A pure trend never
    retraces enough to confirm one; a pure range never makes a higher high
    over a higher low.
    """
    rng = np.random.default_rng(seed)
    out, lvl = [], 100.0
    for _ in range(legs):
        out.append(lvl + np.cumsum(rng.normal(0.25, 0.5, 70)))
        lvl = out[-1][-1]
        out.append(lvl + np.cumsum(rng.normal(-0.30, 0.5, 40)))
        lvl = out[-1][-1]
    return _frame(np.concatenate(out))


def _flat(n: int = 400, level: float = 100.0) -> pd.DataFrame:
    """A dead tape: no range, so ATR is 0 and nothing is tradable."""
    return _frame(np.full(n, level), high=np.full(n, level),
                  low=np.full(n, level))


def _masks(bars, **params):
    return MOD.signal_fn(bars, **params)


# ---------------------------------------------------------------------------
# The module contract
# ---------------------------------------------------------------------------
def test_module_declares_the_four_mandatory_names() -> None:
    """CLAUDE.md's rule: any strategy Claude creates carries `signal_fn`,
    `indicators`, `LOGIC` and `PARAM_GRID`. The loader TOLERATES a module
    declaring none - several pre-existing ones do - and that tolerance is not
    permission to write another."""
    assert callable(MOD.signal_fn)
    assert callable(MOD.indicators)
    assert callable(MOD.make_signal_fn)
    assert callable(MOD.ml_features)
    assert set(MOD.LOGIC) == {"concept", "entry", "exit"}
    assert MOD.PARAM_GRID and isinstance(MOD.PARAM_GRID, dict)
    assert isinstance(MOD.DEFAULT_PARAMS, dict)


def test_declares_the_matrix_the_ladder_and_the_quadrants() -> None:
    """TIMEFRAME stays SINGULAR and stays declared: `tier3_workers` falls back
    to DEFAULT_TIMEFRAME "1d", so a module declaring only the plural would
    silently run an intraday system on DAILY bars."""
    assert MOD.TIMEFRAME == "15m"
    assert MOD.TIMEFRAMES == ["5m", "15m", "30m", "1h"]
    assert len(MOD.SYMBOLS) == len(set(MOD.SYMBOLS)) == 19
    assert MOD.TARGET_QUADRANTS == ("Q1", "Q3")
    assert set(MOD.TARGET_QUADRANTS) <= {"Q1", "Q2", "Q3", "Q4"}


def test_param_grid_only_sweeps_real_parameters() -> None:
    """`load_strategy` rejects unknown parameter names, so a stale PARAM_GRID
    key raises at bind time."""
    assert not set(MOD.PARAM_GRID) - set(MOD.DEFAULT_PARAMS)
    assert int(np.prod([len(v) for v in MOD.PARAM_GRID.values()])) == 36


def test_the_take_profit_carries_the_requested_reward_risk() -> None:
    """The request asks for 2x or 3x the stop distance. The ratio lives in the
    relationship between the two ATR multiples rather than in a third
    parameter that could disagree with them."""
    assert MOD._reward_risk(2.0, 4.0) == 2.0
    assert MOD._reward_risk(2.0, 6.0) == 3.0
    assert MOD._reward_risk(MOD.DEFAULT_PARAMS["sl_atr_mult"],
                            MOD.DEFAULT_PARAMS["tp_atr_mult"]) == 2.0
    # Both swept targets are 2R and 3R against the swept stops.
    for sl in MOD.PARAM_GRID["sl_atr_mult"]:
        for tp in MOD.PARAM_GRID["tp_atr_mult"]:
            assert MOD._reward_risk(sl, tp) > 1.0, (sl, tp)


def test_logic_placeholders_all_resolve() -> None:
    """A LOGIC slot naming a parameter that does not exist raises KeyError at
    render time - on the card, after the backtest has run."""
    for field, text in MOD.LOGIC.items():
        try:
            text.format(**MOD.DEFAULT_PARAMS)
        except KeyError as exc:              # pragma: no cover
            pytest.fail(f"LOGIC[{field!r}] names unknown parameter {exc}")


def test_module_passes_the_ast_security_gate() -> None:
    """Model-generated code passes an AST check before import and this module
    is held to it too. Expects ZERO objections."""
    from agents.tier3_workers import _audit_ast          # noqa: PLC0415
    assert _audit_ast(ast.parse(MODULE_PATH.read_text())) == []


def test_the_risk_keys_are_the_three_promote_py_writes() -> None:
    """`backtest/promote.py` lifts exactly these three into the promoted
    package's `risk` block. A module spelling one differently gets a risk
    block that describes a strategy nobody ran."""
    for key in ("sl_atr_mult", "tp_atr_mult", "trailing"):
        assert key in MOD.DEFAULT_PARAMS, key


# ---------------------------------------------------------------------------
# The ZigZag - the whole causality question
# ---------------------------------------------------------------------------
def test_the_zigzag_confirms_late_and_never_at_the_extreme() -> None:
    """
    THE CASE THIS MODULE TURNS ON.

    A swing high is confirmed on the bar price has retraced `dev * ATR` from
    the extreme - NOT on the bar of the extreme. Read at the extreme it is
    lookahead, and a backtest on that form sells the exact top of every leg.

    Built as a single clean peak so the extreme's index is known, then
    asserted three ways: nothing is confirmed at or before the peak, the
    confirmation lands strictly after it, and the level recorded is the peak
    itself.
    """
    up = np.linspace(100.0, 140.0, 120)
    down = np.linspace(140.0, 100.0, 120)
    bars = _frame(np.concatenate([up, down[1:]]))
    peak = 119
    atr = MOD._atr(bars, MOD.ATR_PERIOD).to_numpy(dtype="float64")
    lh, ph, ll, pl, conf = MOD._zigzag_loop(
        bars["high"].to_numpy(dtype="float64"),
        bars["low"].to_numpy(dtype="float64"), atr, 3.0)

    fired = np.flatnonzero(conf == 1)
    assert fired.size, "no swing high confirmed in a clean peak"
    first = int(fired[0])
    assert first > peak, (
        f"a swing high was confirmed at bar {first}, at or before the peak at "
        f"{peak} - that is the centred, repainting form")
    # The LEVEL is the peak; only the KNOWLEDGE of it is late.
    assert lh[first] == pytest.approx(float(bars["high"].iloc[peak]), rel=1e-9)
    # And nothing knew about it beforehand.
    assert not np.isfinite(lh[:first]).any()


def test_the_confirmed_level_is_carried_forward_and_never_backward() -> None:
    """
    The as-of arrays hold what was known BY each bar. A level must appear on
    its confirmation bar and persist, and must never appear on an earlier one
    - a backward fill would hand every bar of the leg a pivot the market had
    not made yet.
    """
    bars = _impulse_pullback(legs=8)
    atr = MOD._atr(bars, MOD.ATR_PERIOD).to_numpy(dtype="float64")
    lh, ph, ll, pl, conf = MOD._zigzag_loop(
        bars["high"].to_numpy(dtype="float64"),
        bars["low"].to_numpy(dtype="float64"), atr, 3.0)
    changes = np.flatnonzero(np.r_[False, lh[1:] != lh[:-1]]
                             & np.isfinite(lh))
    for c in changes:
        assert conf[c] == 1, (
            f"last_high changed at bar {c} without a confirmation there")
    # Step function: between confirmations the level is held, not interpolated.
    for a, b in zip(changes, changes[1:]):
        seg = lh[a:b]
        assert np.all(seg == seg[0]), "the level moved between confirmations"


def test_the_structure_test_needs_both_a_higher_high_and_a_higher_low() -> None:
    """
    A higher high on a LOWER low is an expanding range, not a trend, and the
    request names the pair. Asserted on the series rather than by construction
    so a rule that quietly dropped one half is visible.
    """
    bars = _impulse_pullback(legs=20)
    L = MOD._layers(bars, 200, 3.0, 12, 26, 9, True)
    hh = L["last_high"] > L["prev_high"]
    hl = L["last_low"] > L["prev_low"]
    assert (L["structure_bull"] & ~(hh & hl)).sum() == 0
    assert (L["structure_bull"] ^ (hh & hl)).sum() == 0
    ll = L["last_low"] < L["prev_low"]
    lh = L["last_high"] < L["prev_high"]
    assert (L["structure_bear"] ^ (ll & lh)).sum() == 0
    # The two are mutually exclusive: a structure cannot be both.
    assert (L["structure_bull"] & L["structure_bear"]).sum() == 0


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------
def test_the_macd_is_the_standard_definition() -> None:
    """
    line = EMA(fast) - EMA(slow); signal = EMA(line); hist = line - signal.

    The signal is an EMA of the LINE, not of price - a signal computed from
    price is a third moving average that crosses on different bars, and this
    module's momentum gate is a comparison between the two.
    """
    bars = _impulse_pullback(legs=6)
    close = bars["close"]
    M = MOD._macd(close, 12, 26, 9)
    assert np.allclose(M["macd"].dropna(),
                       (MOD._ema(close, 12) - MOD._ema(close, 26)).dropna())
    assert np.allclose(M["signal"].dropna(),
                       MOD._ema(M["macd"], 9).dropna())
    assert np.allclose(M["hist"].dropna(),
                       (M["macd"] - M["signal"]).dropna())
    wrong = MOD._ema(close, 9)
    both = pd.concat([M["signal"], wrong], axis=1).dropna()
    assert not np.allclose(both.iloc[:, 0], both.iloc[:, 1]), (
        "a signal taken from price is indistinguishable here; this proves "
        "nothing")


def test_the_histogram_must_be_expanding_not_merely_positive() -> None:
    """`require_expansion` is the difference between 'momentum is present' and
    'momentum is accelerating', and the request asks for the second."""
    bars = _impulse_pullback(legs=20)
    strict = MOD._layers(bars, 200, 3.0, 12, 26, 9, True)["momo_long"]
    loose = MOD._layers(bars, 200, 3.0, 12, 26, 9, False)["momo_long"]
    assert loose.sum() > strict.sum(), (int(strict.sum()), int(loose.sum()))
    assert (strict & ~loose).sum() == 0, "expansion admitted a bar the loose "\
                                         "form vetoes"


def test_the_entry_is_the_edge_of_the_setup_not_every_bar_of_it() -> None:
    """The request states four CONDITIONS. Taken as a state they hold for a
    run of bars and would re-fire on each; the entry is the bar the set FIRST
    holds."""
    bars = _impulse_pullback(legs=20)
    L = MOD._layers(bars, 200, 3.0, 12, 26, 9, True)
    assert L["trigger_long"].sum() < L["setup_long"].sum()
    assert (L["trigger_long"] & ~L["setup_long"]).sum() == 0
    # A trigger never fires on a bar whose predecessor was already a setup.
    assert (L["trigger_long"] & L["setup_long"].shift(1, fill_value=False)
            ).sum() == 0


# ---------------------------------------------------------------------------
# The walk, and the one parameter it grew
# ---------------------------------------------------------------------------
def test_the_walk_matches_the_shared_one_when_the_step_is_zero() -> None:
    """
    THE CASE THAT KEEPS THE COPY HONEST.

    Every other module carries `_walk_loop` verbatim so two copies cannot
    disagree about the fill timeline. This one adds `trail_step_mult`, so the
    convention is preserved by PROOF instead: at a step of zero it must
    reproduce the shared implementation exactly, on the same arrays, in both
    directions.
    """
    shared = _load(SHARED_WALK_PATH, "_shared_walk")
    rng = np.random.default_rng(11)
    n = 900
    close = 100.0 + np.cumsum(rng.normal(0, 0.6, n))
    high = close + np.abs(rng.normal(0, 0.4, n))
    low = close - np.abs(rng.normal(0, 0.4, n))
    open_ = np.r_[close[0], close[:-1]]
    atr = np.full(n, 1.25)
    le_ok = rng.random(n) < 0.02
    se_ok = rng.random(n) < 0.02
    lx = rng.random(n) < 0.01
    sx = rng.random(n) < 0.01
    flat = np.zeros(n, dtype=bool)

    for trailing in (False, True):
        for tp in (float("nan"), 4.0):
            mine = MOD._walk_loop(le_ok, se_ok, lx, sx, open_, high, low, atr,
                                  flat, 2.0, tp, trailing, 0.0)
            theirs = shared._walk_loop(le_ok, se_ok, lx, sx, open_, high, low,
                                       atr, flat, 2.0, tp, trailing)
            for a, b in zip(mine, theirs):
                assert np.array_equal(a, b, equal_nan=True), (
                    f"diverged at trailing={trailing} tp={tp}")


def test_the_trailing_step_holds_the_stop_still_between_increments() -> None:
    """
    A stepped trail is NOT a continuous trail rounded off. The stop stays
    where it is until the position has made a whole step of new profit, then
    jumps by exactly one step.

    Built on a clean monotonic advance so the levels are checkable by eye.
    """
    n = 300
    close = np.linspace(100.0, 160.0, n)
    bars_high = close + 0.1
    bars_low = close - 0.1
    open_ = np.r_[close[0], close[:-1]]
    atr = np.full(n, 1.0)
    le = np.zeros(n, dtype=bool); le[5] = True
    none = np.zeros(n, dtype=bool)

    stepped = MOD._walk_loop(le, none, none, none, open_, bars_high, bars_low,
                             atr, none, 2.0, float("nan"), True, 1.0)[4]
    smooth = MOD._walk_loop(le, none, none, none, open_, bars_high, bars_low,
                            atr, none, 2.0, float("nan"), True, 0.0)[4]
    live = np.isfinite(stepped)
    moves_stepped = int(np.sum(np.diff(stepped[live]) > 0))
    moves_smooth = int(np.sum(np.diff(smooth[live]) > 0))

    # THE EXACT CLAIM, not a hand-picked ratio. The tape advances ~60 points
    # while the position is live and the step is 1.0 x ATR(1.0), so the stop
    # must move ~60 times - once per whole step earned - against a continuous
    # trail that moves on nearly every one of the ~295 live bars. An
    # assertion like `< smooth/5` would have been a number chosen to pass;
    # this one fails if the step stops meaning "one increment of profit".
    advance = float(smooth[live][-1] - smooth[live][0])
    assert abs(moves_stepped - round(advance / 1.0)) <= 2, (
        f"{moves_stepped} steps over an advance of {advance:.1f} at step 1.0")
    assert moves_stepped < moves_smooth / 4, (moves_stepped, moves_smooth)
    # Never looser than the continuous trail, and never below the initial stop.
    assert np.all(stepped[live] <= smooth[live] + 1e-9)
    assert np.all(np.diff(stepped[live]) >= -1e-9), "the stop moved backwards"


def test_a_step_larger_than_the_move_never_trails_at_all() -> None:
    """The degenerate end of the same rule: a step the position never earns
    leaves the stop at its initial level, which is a fixed stop."""
    n = 200
    close = np.linspace(100.0, 103.0, n)
    open_ = np.r_[close[0], close[:-1]]
    atr = np.full(n, 1.0)
    le = np.zeros(n, dtype=bool); le[5] = True
    none = np.zeros(n, dtype=bool)
    levels = MOD._walk_loop(le, none, none, none, open_, close + 0.1,
                            close - 0.1, atr, none, 2.0, float("nan"),
                            True, 50.0)[4]
    live = levels[np.isfinite(levels)]
    assert np.allclose(live, live[0]), "a step nobody earned still moved"


# ---------------------------------------------------------------------------
# The contract's shape
# ---------------------------------------------------------------------------
def test_signal_fn_returns_the_four_mask_form() -> None:
    """A three-tuple or a bare Series RAISES in `engine.unpack_signals` rather
    than silently losing the short side into a plausible long-only curve."""
    bars = _impulse_pullback()
    out = _masks(bars)
    assert isinstance(out, tuple) and len(out) == 4
    for mask in out:
        assert isinstance(mask, pd.Series) and mask.dtype == bool
        assert mask.index.equals(bars.index)


def test_both_sides_trade_and_pair_up() -> None:
    """A bidirectional module whose short side never fires is a long-only
    strategy with a plausible curve and a mask nobody checked."""
    bars = _impulse_pullback()
    le, lx, se, sx = _masks(bars)
    assert le.sum() > 0, "the long side never fired"
    assert se.sum() > 0, "the short side never fired"
    # At most one position may still be open when the tape ends, on one side.
    lo, so = int(le.sum()) - int(lx.sum()), int(se.sum()) - int(sx.sum())
    assert lo in (0, 1) and so in (0, 1) and lo + so <= 1


def test_positions_are_never_pyramided_or_reversed() -> None:
    """The walk enters only from flat, so a short signal arriving while long
    is ignored and the long must exit first."""
    bars = _impulse_pullback()
    le, lx, se, sx = _masks(bars)
    assert not (le & se).any(), "a bar opened both sides"
    state = 0
    for i in range(len(bars)):
        if state == 0 and (le.iloc[i] or se.iloc[i]):
            state = 1 if le.iloc[i] else -1
            continue
        if state == 1:
            assert not se.iloc[i], f"short opened while long at bar {i}"
            if lx.iloc[i]:
                state = 0
        elif state == -1:
            assert not le.iloc[i], f"long opened while short at bar {i}"
            if sx.iloc[i]:
                state = 0


def test_exits_never_land_before_the_fill_bar() -> None:
    """The fill for a signal on bar i is `open[i+1]`, so the position is not
    live until i+1 and cannot be closed on i."""
    bars = _impulse_pullback()
    le, lx, se, sx = _masks(bars)
    entries = np.flatnonzero((le | se).to_numpy())
    exits = np.flatnonzero((lx | sx).to_numpy())
    for e in entries:
        after = exits[exits > e]
        if after.size:
            assert after[0] >= e + 1, f"exit at {after[0]} for entry at {e}"


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------
def test_signals_never_change_when_future_bars_are_removed() -> None:
    """
    Truncation is the only test that can see a forward-reading window from
    outside. Every rolling window, the ZigZag state machine and the risk model
    must end at the bar they describe.
    """
    bars = _impulse_pullback()
    full = _masks(bars)
    for cut in (1500, 3000, 5000):
        truncated = _masks(bars.iloc[:cut])
        for name, a, b in zip(("long_entries", "long_exits",
                               "short_entries", "short_exits"),
                              full, truncated):
            assert (a.iloc[:cut].to_numpy() == b.to_numpy()).all(), (
                f"{name} changed in the first {cut} bars when the future was "
                f"removed - the module repaints")


def test_no_negative_shift_appears_anywhere_in_the_module() -> None:
    """A `.shift(-n)` reads FORWARD, and one anywhere in a signal path is a
    repainting module that passes every value-based test."""
    for node in ast.walk(ast.parse(MODULE_PATH.read_text())):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "shift"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                pytest.fail(f"negative .shift() at line {node.lineno}")


def test_no_frame_wide_statistic_reaches_the_risk_model() -> None:
    """
    The stop, the target and the trailing step are all frozen at the SIGNAL
    bar from that bar's ATR. Sizing any of them from a statistic over the
    whole frame - a median ATR, say - would make the risk model depend on bars
    that have not printed. That is the same lookahead the ZigZag is written to
    avoid, and it is no better for arriving through the risk model; it was in
    the first draft of this module and the truncation case is what caught it.
    """
    # Targeted at WHOLE-ARRAY reducers only. `ewm(...).mean()` and
    # `rolling(...).mean()` are causal by construction and appear all over
    # this module; banning the substring `.mean()` would have caught those and
    # said nothing about the actual fault.
    src = MODULE_PATH.read_text()
    for banned in ("np.nanmedian", "np.nanmean", "np.median(", "np.mean(",
                   "np.nanpercentile", "np.percentile("):
        assert banned not in src, f"{banned} reaches the risk model"
    # And the walk takes the step as a MULTIPLE, sized from the signal bar's
    # own ATR beside the stop - not as a pre-computed distance handed in from
    # a frame-wide statistic, which is the shape the first draft had.
    assert "trail_step_mult" in src
    assert "trail_step = trail_step_mult * atr[i]" in src


def test_ml_features_are_causal_and_shaped_to_the_bars() -> None:
    """Shape is checked by the caller and a failure RAISES - unlike
    `indicators`, which is wrapped."""
    bars = _impulse_pullback()
    full = MOD.ml_features(bars)
    assert len(full) == len(bars) and full.index.equals(bars.index)
    assert np.isfinite(full.to_numpy()).all(), "a non-finite feature"
    cut = 3000
    assert np.allclose(full.iloc[:cut].to_numpy(),
                       MOD.ml_features(bars.iloc[:cut]).to_numpy()), (
        "a feature column changed when future bars were removed")


def test_indicators_are_full_length_and_named() -> None:
    """Drawn from the same `_layers` call `signal_fn` uses, so the chart
    cannot draw a crossover a bar from where the entry happened."""
    bars = _impulse_pullback(legs=8)
    drawn = MOD.indicators(bars)
    for name, series in drawn.items():
        assert isinstance(series, pd.Series) and len(series) == len(bars), name
        assert series.index.equals(bars.index), name
    assert any("MACD" in n for n in drawn)
    assert any("ZigZag" in n for n in drawn)
    assert any("EMA" in n for n in drawn)


# ---------------------------------------------------------------------------
# Parameter hygiene
# ---------------------------------------------------------------------------
def test_a_dead_flat_tape_produces_no_trades() -> None:
    """ATR is 0 on a tape that never moved, so no bracket has width and the
    ZigZag threshold is zero - which would otherwise make every bar a pivot."""
    le, lx, se, sx = _masks(_flat())
    assert not le.any() and not se.any() and not lx.any() and not sx.any()


@pytest.mark.parametrize("bad", [
    {"ema_period": 1},
    {"macd_fast": 26, "macd_slow": 12},
    {"macd_fast": 1},
    {"macd_signal": 0},
    {"zigzag_atr_mult": 0.0},
    {"zigzag_atr_mult": -1.0},
    {"sl_atr_mult": 0.01},
    {"tp_atr_mult": 0.0},
    {"tp_atr_mult": 2.0},          # at the stop: reward:risk 1.0
    {"tp_atr_mult": 1.0},          # inside the stop
    {"trailing": "yes"},
    {"trail_step_atr": -0.5},
])
def test_impossible_parameters_raise_rather_than_running(bad) -> None:
    """Refused where they are passed, rather than as a strange equity curve
    somebody has to explain later."""
    with pytest.raises(ValueError):
        MOD.signal_fn(_impulse_pullback(legs=4), **bad)


def test_unknown_parameters_raise_on_both_entry_points() -> None:
    """A stale grid key silently dropped would sweep the DEFAULT and report it
    under the swept name."""
    with pytest.raises(ValueError):
        MOD.make_signal_fn(no_such_param=1)
    with pytest.raises(ValueError):
        MOD.generate_signals(_impulse_pullback(legs=4), {"no_such_param": 1})


def test_generate_signals_delegates_to_signal_fn() -> None:
    """It delegates rather than reimplementing, so the two can never disagree
    about what a signal is."""
    bars = _impulse_pullback(legs=8)
    for a, b in zip(MOD.signal_fn(bars, **MOD.DEFAULT_PARAMS),
                    MOD.generate_signals(bars)):
        assert a.equals(b)


def test_make_signal_fn_binds_and_matches_direct_calls() -> None:
    """The parameterised form `load_strategy` prefers."""
    bars = _impulse_pullback(legs=8)
    bound = MOD.make_signal_fn(ema_period=50, zigzag_atr_mult=2.0)
    direct = MOD.signal_fn(bars, ema_period=50, zigzag_atr_mult=2.0)
    for a, b in zip(bound(bars), direct):
        assert a.equals(b)
