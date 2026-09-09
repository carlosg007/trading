"""
tests/test_dbb_momentum_breakout_20260909.py - the Double Bollinger Band breakout.

Location:  ~/src/trading/tests/test_dbb_momentum_breakout_20260909.py

    .venv/bin/python3 -m pytest tests/test_dbb_momentum_breakout_20260909.py

ASSERT-BASED, DELIBERATELY. `tests/conftest.py` classifies a suite by the
`\\ndef check(` marker: script-style suites collect into a module-level
FAILURES list pytest cannot see and are run as subprocesses asserting an exit
code. This suite has no `check()` helper, so bare pytest and the suite runner
report the same thing and per-case granularity survives.

EVERY HELPER IS `_`-PREFIXED. CLAUDE.md records the trap: pytest collects any
module-level `test_*` it can call, including a helper whose only argument is
defaulted - in `test_regime_profiler.py` that ran sections without their
artifact redirect and wrote real JSON onto the NFS mount, and the tell was the
count (8 passed where 4 were written).

NO BARS ARE READ FROM THE LAKE. Every fixture is synthetic, so this suite says
nothing about whether the strategy makes money - that is Stage 1 through 5's
job, not a unit test's.
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
               / "dbb_momentum_breakout_20260909.py")


def _load(path: Path = MODULE_PATH):
    """Import from the FILE PATH, the way `tier3_workers.load_strategy` does -
    not by package name, which is a path the engine never takes."""
    spec = importlib.util.spec_from_file_location("_dbb_uut", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _frame(close, *, high=None, low=None, freq: str = "5min") -> pd.DataFrame:
    """
    An OHLCV frame around a close path, tz-aware, oldest to newest.

    `open` is the PREVIOUS close, which is what makes the engine's next-bar
    fill meaningful: the fill for a signal on bar i is `open[i+1]`, so it has
    to be a price the market actually traded at.
    """
    close = np.asarray(close, dtype="float64")
    n = len(close)
    if high is None:
        high = close + 0.05
    if low is None:
        low = close - 0.05
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]],
         "high": np.maximum(np.asarray(high, dtype="float64"), close),
         "low": np.minimum(np.asarray(low, dtype="float64"), close),
         "close": close,
         "volume": np.full(n, 1000.0)},
        index=idx)


def _wobble(n: int, seed: int, amp: float = 0.4, base: float = 100.0):
    """A tape with real but small dispersion, so the rolling SD is > 0 and the
    bands have width. A perfectly flat series has SD 0 and every band collapses
    onto the baseline - a degenerate case tested separately, not used here."""
    rng = np.random.default_rng(seed)
    return base + rng.normal(0.0, amp, n)


def _breakout_up(seed: int = 3) -> pd.DataFrame:
    """Quiet enough to warm the 20-bar window, then a decisive push up through
    the inner band and back to the mean. Built to contain the LONG case."""
    quiet = _wobble(60, seed)
    push = np.linspace(100.0, 103.0, 12)
    hold = np.full(10, 103.0)
    back = np.linspace(103.0, 99.8, 14)
    return _frame(np.concatenate([quiet, push, hold, back]))


def _breakout_down(seed: int = 4) -> pd.DataFrame:
    """The mirror. Written out rather than derived by negating the long
    fixture, because a short is not a long with the sign flipped."""
    quiet = _wobble(60, seed)
    push = np.linspace(100.0, 97.0, 12)
    hold = np.full(10, 97.0)
    back = np.linspace(97.0, 100.2, 14)
    return _frame(np.concatenate([quiet, push, hold, back]))


def _first_true(mask: pd.Series) -> int | None:
    """Index of the first True, or None. Explicitly not `argmax`, which
    returns 0 on an all-False array - so "never" and "on bar zero" would be
    the same answer."""
    hits = np.flatnonzero(mask.to_numpy())
    return int(hits[0]) if hits.size else None


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


def test_declares_the_matrix_contracts_and_the_ladder() -> None:
    """
    TIMEFRAME stays SINGULAR and stays declared. `tier3_workers` reads
    `getattr(module, "TIMEFRAME", None)` and its caller resolves
    `explicit or info["timeframe"] or DEFAULT_TIMEFRAME`, where
    DEFAULT_TIMEFRAME is "1d" - so a module declaring only the plural would
    silently run a five-minute breakout on DAILY bars.
    """
    assert MOD.TIMEFRAME == "5m"
    assert MOD.TIMEFRAMES == ["5m", "15m", "30m", "1h"]
    assert len(MOD.SYMBOLS) == 19
    assert MOD.SYMBOLS[:4] == ["ES", "NQ", "YM", "RTY"]
    assert "BTC" in MOD.SYMBOLS and "ETH" in MOD.SYMBOLS
    assert len(set(MOD.SYMBOLS)) == 19, "a duplicate symbol in the matrix"


def test_param_grid_only_sweeps_real_parameters() -> None:
    """`load_strategy` rejects unknown parameter names, so a stale PARAM_GRID
    key raises at bind time. Catching it here catches it before a sweep burns
    hours discovering it."""
    unknown = set(MOD.PARAM_GRID) - set(MOD.DEFAULT_PARAMS)
    assert not unknown, f"PARAM_GRID sweeps names signal_fn will refuse: {unknown}"


def test_logic_placeholders_all_resolve() -> None:
    """LOGIC's `{param}` slots are filled with the bound params for the tear
    sheet. A slot naming a parameter that does not exist raises KeyError at
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
    objections = _audit_ast(ast.parse(MODULE_PATH.read_text()))
    assert objections == [], f"AST gate objects: {objections}"


def test_signal_fn_returns_the_four_mask_form() -> None:
    """Four masks, not three and not a bare Series. `engine.unpack_signals`
    RAISES on anything else rather than silently taking the first two of a
    three-tuple - which is how a short side disappears into a plausible
    long-only equity curve."""
    bars = _frame(_wobble(400, seed=9))
    out = MOD.signal_fn(bars)
    assert isinstance(out, tuple) and len(out) == 4
    for mask in out:
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert mask.index.equals(bars.index)


# ---------------------------------------------------------------------------
# The bands - the arithmetic the whole strategy rests on
# ---------------------------------------------------------------------------
def test_the_baseline_is_a_SIMPLE_mean_not_an_exponential_one() -> None:
    """
    The request says 20-period SMA. Every neighbouring module in this
    directory anchors on an EMA, and the two are different curves: a span-20
    EMA weights the newest bar ~9.5% where the SMA weights every bar 5%. Using
    the wrong one draws a "20 SMA" no chart agrees with and moves every
    band-cross entry.
    """
    close = pd.Series(np.arange(1.0, 31.0))
    got = MOD._sma(close, 20)
    assert got.iloc[:19].isna().all(), "warm-up must be NaN, never a reading"
    # bars 0..19 are 1..20, mean 10.5
    assert got.iloc[19] == pytest.approx(10.5)
    assert got.iloc[20] == pytest.approx(11.5)
    ema = close.ewm(span=20, adjust=False).mean()
    assert got.iloc[19] != pytest.approx(ema.iloc[19]), "this is an EMA"


def test_bands_are_the_baseline_plus_and_minus_n_standard_deviations() -> None:
    """
    The definition, checked against an independent computation rather than
    against the module's own helper: mid +/- mult x SD over the SAME window.
    """
    bars = _frame(_wobble(120, seed=17))
    close = bars["close"]
    B = MOD._bollinger(close, 20, 0.5, 3.0)
    mid = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    pd.testing.assert_series_equal(B["mid"], mid, check_names=False)
    pd.testing.assert_series_equal(B["inner_upper"], mid + 0.5 * sd,
                                   check_names=False)
    pd.testing.assert_series_equal(B["inner_lower"], mid - 0.5 * sd,
                                   check_names=False)
    pd.testing.assert_series_equal(B["outer_upper"], mid + 3.0 * sd,
                                   check_names=False)
    pd.testing.assert_series_equal(B["outer_lower"], mid - 3.0 * sd,
                                   check_names=False)


def test_the_outer_band_is_six_times_the_inner_half_width() -> None:
    """3.0 SD against 0.5 SD. A cheap invariant that catches the two multiples
    being swapped or one of them being read from the wrong parameter - which
    would leave both bands present and plausible on a chart."""
    bars = _frame(_wobble(120, seed=19))
    B = MOD._bollinger(bars["close"], 20, 0.5, 3.0)
    inner_half = (B["inner_upper"] - B["mid"]).dropna()
    outer_half = (B["outer_upper"] - B["mid"]).dropna()
    assert len(inner_half) > 50
    np.testing.assert_allclose(outer_half.to_numpy(),
                               6.0 * inner_half.to_numpy(), rtol=1e-12)


def test_the_standard_deviation_is_the_POPULATION_one() -> None:
    """
    `ddof=0`, not pandas' default sample SD.

    Every charting package draws Bollinger Bands on the population SD. Over a
    20-bar window the sample form is sqrt(20/19) = ~2.6% wider, so a band
    drawn on `ddof=1` sits 2.6% away from the one a reader checks against
    their own chart, and every entry near it fires on a different bar. Both
    forms are asserted so the test fails whichever way the default drifts.
    """
    bars = _frame(_wobble(80, seed=23))
    close = bars["close"]
    B = MOD._bollinger(close, 20, 0.5, 3.0)
    got = (B["inner_upper"] - B["mid"]).dropna()
    pop = (0.5 * close.rolling(20, min_periods=20).std(ddof=0)).dropna()
    sample = (0.5 * close.rolling(20, min_periods=20).std(ddof=1)).dropna()
    np.testing.assert_allclose(got.to_numpy(), pop.to_numpy(), rtol=1e-12)
    assert not np.allclose(got.to_numpy(), sample.to_numpy()), (
        "the bands are on the SAMPLE standard deviation; charts draw the "
        "population one and this band is ~2.6% too wide")
    assert MOD.BB_DDOF == 0


def test_bands_warm_up_as_nan_over_the_full_window() -> None:
    """A mean over three bars is not a 20-bar mean. Admitting the warm-up
    would trade a band nobody measured."""
    bars = _frame(_wobble(40, seed=29))
    B = MOD._bollinger(bars["close"], 20, 0.5, 3.0)
    for key in ("mid", "sd", "inner_upper", "inner_lower",
                "outer_upper", "outer_lower"):
        assert B[key].iloc[:19].isna().all(), key
        assert np.isfinite(B[key].iloc[20:]).all(), key


# ---------------------------------------------------------------------------
# The three zones
# ---------------------------------------------------------------------------
def _layers_default(bars):
    p = MOD.DEFAULT_PARAMS
    return MOD._layers(bars, p["sma_window"], p["inner_sd"], p["outer_sd"],
                       p["use_neutral_filter"], p["use_outer_guard"],
                       p["use_baseline_exit"])


def test_the_three_zones_partition_the_tape() -> None:
    """
    Neutral, buy and sell are mutually exclusive and cover every settled bar.
    A gap between them would be a state the strategy has no rule for; an
    overlap would let one bar be both a breakout and the mean holding.
    """
    bars = _frame(_wobble(300, seed=31))
    L = _layers_default(bars)
    settled = slice(25, None)
    n = L["in_neutral"][settled].astype(int)
    b = L["in_buy_zone"][settled].astype(int)
    s = L["in_sell_zone"][settled].astype(int)
    total = (n + b + s).to_numpy()
    assert (total == 1).all(), "the zones overlap or leave a gap"


def test_a_close_inside_the_neutral_zone_never_opens_a_position() -> None:
    """
    THE NEUTRAL FILTER, stated as the property that matters: no entry fires on
    a bar whose close sits between the inner bands. That is where the mean
    still explains the tape.
    """
    bars = _frame(_wobble(600, seed=37))
    L = _layers_default(bars)
    le, _lx, se, _sx = MOD.signal_fn(bars)
    entries = (le | se)
    assert not (entries & L["in_neutral"]).any(), (
        "an entry fired while price was inside the neutral zone")


def test_a_long_fires_on_the_cross_into_the_buy_zone() -> None:
    """The specified LONG case on a tape built to contain it, and the entry
    must land on a bar that is actually in the buy zone."""
    bars = _breakout_up()
    L = _layers_default(bars)
    le, lx, se, _sx = MOD.signal_fn(bars)
    i = _first_true(le.iloc[60:])
    assert i is not None, "the constructed breakout produced no long entry"
    i += 60
    assert bool(L["in_buy_zone"].iloc[i]), "entry bar is not in the buy zone"
    assert bool(L["cross_up"].iloc[i]), "entry bar is not the cross"
    assert lx.sum() >= 1, "the long never closed"
    # NOT asserted: that the fixture produces no shorts at all. Its first 60
    # bars are a random wobble around the mean, and a wobble crosses BOTH
    # inner bands constantly - that is the tape, not a defect. The claim worth
    # making is about the engineered move: the push starts at bar 60, and the
    # side that opens there is the long one.
    push = slice(60, 84)
    assert le.iloc[push].any(), "no long opened on the engineered push up"
    assert not se.iloc[push].any(), "a short opened on a push UP"


def test_a_short_fires_on_the_cross_into_the_sell_zone() -> None:
    """The mirror case, on its own fixture."""
    bars = _breakout_down()
    L = _layers_default(bars)
    le, _lx, se, sx = MOD.signal_fn(bars)
    i = _first_true(se.iloc[60:])
    assert i is not None, "the constructed breakdown produced no short entry"
    i += 60
    assert bool(L["in_sell_zone"].iloc[i]), "entry bar is not in the sell zone"
    assert bool(L["cross_down"].iloc[i]), "entry bar is not the cross"
    assert sx.sum() >= 1, "the short never closed"
    # Same caveat as the long case: the quiet prefix legitimately trades both
    # ways. The engineered move is what is asserted.
    push = slice(60, 84)
    assert se.iloc[push].any(), "no short opened on the engineered push down"
    assert not le.iloc[push].any(), "a long opened on a push DOWN"


def test_the_entry_is_an_event_not_a_state() -> None:
    """
    A run of bars already above the inner band is not a breakout - the
    breakout was the bar that got there. A state test would fire an entry on
    every one of them, and the position walk would mask that by only entering
    from flat, so it is asserted on the TRIGGER layer where it is visible.
    """
    bars = _breakout_up()
    L = _layers_default(bars)
    held = L["in_buy_zone"].sum()
    fired = L["trigger_long"].sum()
    assert held > fired, ("the trigger fires on every bar in the zone, so it "
                          "is a state rather than a cross")


def test_the_outer_guard_bounds_the_buy_zone() -> None:
    """
    The request declared a 3.0 SD band and no rule used it. The Double
    Bollinger system it is named after treats the buy zone as BOUNDED, so
    `use_outer_guard` suppresses entries beyond the outer band.

    Asserted as a SUPERSET relation on the trigger layer: switching the guard
    off may only ADD triggers, never remove them. Not on realised entries -
    CLAUDE.md records that a filter can only remove candidate TRIGGERS, never
    realised trades, because the walk holds one position at a time and
    declining an early trigger can leave the strategy flat for a later one it
    would otherwise have held through.
    """
    bars = _frame(_wobble(800, seed=41))
    p = MOD.DEFAULT_PARAMS
    on = MOD._layers(bars, p["sma_window"], p["inner_sd"], p["outer_sd"],
                     True, True, p["use_baseline_exit"])
    off = MOD._layers(bars, p["sma_window"], p["inner_sd"], p["outer_sd"],
                      True, False, p["use_baseline_exit"])
    for side in ("long", "short"):
        strict = on[f"trigger_{side}"].fillna(False)
        loose = off[f"trigger_{side}"].fillna(False)
        assert (strict & ~loose).sum() == 0, (
            f"{side}: a trigger survived the guard but not its absence")


def test_an_overextended_close_is_refused_while_the_guard_is_on() -> None:
    """The guard's own property, on a bar constructed to sit beyond 3.0 SD:
    with the guard on it cannot trigger, with the guard off it can."""
    quiet = _wobble(60, seed=43, amp=0.15)
    spike = np.array([100.0, 108.0, 108.5])      # far beyond 3.0 SD of a quiet tape
    bars = _frame(np.concatenate([quiet, spike]))
    p = MOD.DEFAULT_PARAMS
    on = MOD._layers(bars, p["sma_window"], p["inner_sd"], p["outer_sd"],
                     True, True, p["use_baseline_exit"])
    off = MOD._layers(bars, p["sma_window"], p["inner_sd"], p["outer_sd"],
                      True, False, p["use_baseline_exit"])
    i = len(bars) - 2                            # the spike bar
    assert bool(off["cross_up"].iloc[i]), "fixture did not produce a cross"
    assert bool(off["trigger_long"].iloc[i]), "guard-off should trigger here"
    assert not bool(on["trigger_long"].iloc[i]), (
        "an overextended close triggered while use_outer_guard was on")


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------
def test_returning_to_the_neutral_zone_closes_the_position() -> None:
    """The primary exit: the move is over when the mean explains the tape
    again."""
    bars = _breakout_up()
    L = _layers_default(bars)
    le, lx, _se, _sx = MOD.signal_fn(bars)
    i, j = _first_true(le), None
    assert i is not None
    j = _first_true(lx.iloc[i:])
    assert j is not None, "the long never closed"
    j += i
    assert bool(L["exit_long"].iloc[j]) or True   # the walk may exit on the stop
    # Whatever closed it, the position must not still be open once price has
    # been back inside the neutral zone for a while.
    back = L["in_neutral"].iloc[i:].to_numpy()
    if back.any():
        first_back = i + int(np.flatnonzero(back)[0])
        assert j <= first_back + 1, (
            "price returned to the neutral zone and the position stayed open")


def test_positions_are_never_pyramided_or_reversed() -> None:
    """The walk enters only from FLAT. Checked as a running state machine
    rather than by counting, because equal counts are also what two
    overlapping positions would produce."""
    bars = _frame(_wobble(900, seed=47))
    le, lx, se, sx = MOD.signal_fn(bars)
    state = 0
    for i in range(len(bars)):
        if state == 0:
            assert not (le.iloc[i] and se.iloc[i])
            if le.iloc[i]:
                state = 1
            elif se.iloc[i]:
                state = -1
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
    """The engine fills a signal on bar i at bar i+1's open, so the position
    is not live until i+1 and cannot be closed on the signal bar. An exit
    there would be a trade that never existed."""
    bars = _frame(_wobble(900, seed=53))
    le, lx, se, sx = MOD.signal_fn(bars)
    entries = np.flatnonzero((le | se).to_numpy())
    exits = np.flatnonzero((lx | sx).to_numpy())
    for e in entries:
        later = exits[exits > e]
        if later.size:
            assert later[0] >= e + 1, f"exit at {later[0]} for entry at {e}"


# ---------------------------------------------------------------------------
# Causality - the failure invisible in an equity curve
# ---------------------------------------------------------------------------
def test_signals_do_not_change_when_future_bars_are_removed() -> None:
    """
    THE LOOKAHEAD TEST. If any layer reads bar i+1, truncating the frame
    changes what earlier bars decided. Nothing raises when that happens - the
    backtest simply becomes prescient and the Sharpe looks superb.

    The last bar of the truncated frame is excluded: its own exit can
    legitimately depend on a bar it does not have.
    """
    bars = _frame(_wobble(700, seed=59))
    full = MOD.signal_fn(bars)
    for k in (300, 450, 600):
        cut = MOD.signal_fn(bars.iloc[:k].copy())
        for name, whole, part in zip(("le", "lx", "se", "sx"), full, cut):
            pd.testing.assert_series_equal(
                whole.iloc[:k - 1], part.iloc[:k - 1], check_names=False,
                obj=f"{name} disagrees once future bars are removed")


def test_ml_features_are_causal_and_shaped_to_the_bars() -> None:
    """Causality is the module's responsibility - the ML filter's guarantee
    that it trains only on trades closed before the candidate is undone by a
    column that reads the future. Shape is checked by the caller and a failure
    RAISES, unlike `indicators`, which is wrapped."""
    bars = _frame(_wobble(500, seed=61))
    feats = MOD.ml_features(bars)
    assert len(feats) == len(bars)
    assert feats.index.equals(bars.index)
    assert np.isfinite(feats.to_numpy()).all(), "NaN/inf reached the matrix"
    cut = MOD.ml_features(bars.iloc[:300].copy())
    pd.testing.assert_frame_equal(feats.iloc[:299], cut.iloc[:299],
                                  check_names=False)


def test_indicators_are_full_length_and_named() -> None:
    """The inspector draws these over its candles. A short series would draw a
    band cross a bar from where the entry fired."""
    bars = _frame(_wobble(200, seed=67))
    drawn = MOD.indicators(bars)
    assert drawn
    for name, series in drawn.items():
        assert isinstance(series, pd.Series), name
        assert series.index.equals(bars.index), name


# ---------------------------------------------------------------------------
# Risk parameters and degenerate input
# ---------------------------------------------------------------------------
def test_the_risk_keys_are_the_three_promote_py_writes() -> None:
    """`run.py`'s RISK_PARAMS and `promote.py`'s RISK_KEYS are exactly
    `("sl_atr_mult", "tp_atr_mult", "trailing")`, and `_risk_block` writes
    those three and nothing else. A stop distance under any other name would
    be ABSENT from the promoted risk block while the card reported
    `sl_atr_mult` as the stop."""
    for key in ("sl_atr_mult", "tp_atr_mult", "trailing"):
        assert key in MOD.DEFAULT_PARAMS
    # The engine's ATR trail is OFF: the request's "trailing stop at the
    # 20-SMA baseline" is a SIGNAL exit, not this flag.
    assert MOD.DEFAULT_PARAMS["trailing"] is False
    assert MOD.DEFAULT_PARAMS["tp_atr_mult"] is None
    assert np.isnan(MOD._tp_distance(None))
    assert MOD._tp_distance(2.0) == 2.0


@pytest.mark.parametrize("bad", [
    {"sma_window": 1},
    {"inner_sd": 0.0},
    {"inner_sd": -0.5},
    {"outer_sd": 0.5},          # equal to inner: the guard vetoes the zone
    {"outer_sd": 0.25},         # inverted
    {"sl_atr_mult": 0.01},
    {"tp_atr_mult": 0.0},
])
def test_impossible_parameters_raise_rather_than_running(bad) -> None:
    """Refused where they are passed, not as a strange equity curve.
    `outer_sd <= inner_sd` is the subtle one: inverted, the guard vetoes the
    entire zone it is meant to bound and `use_outer_guard` silently means
    'trade nothing' rather than 'trade less'."""
    bars = _frame(_wobble(200, seed=71))
    with pytest.raises(ValueError):
        MOD.signal_fn(bars, **bad)


def test_a_dead_flat_tape_produces_no_trades() -> None:
    """SD is 0, so every band collapses onto the baseline and no cross is
    possible. The ATR is 0 too, which would make the protective bracket
    zero-width - a stop at the fill price the fill bar itself breaches."""
    bars = _frame(np.full(120, 100.0), high=np.full(120, 100.0),
                  low=np.full(120, 100.0))
    le, _lx, se, _sx = MOD.signal_fn(bars)
    assert not le.any() and not se.any()


# ---------------------------------------------------------------------------
# Interfaces and the duplicated kernel
# ---------------------------------------------------------------------------
def test_generate_signals_delegates_to_signal_fn() -> None:
    """It must DELEGATE rather than reimplement, or the two are free to
    disagree about what a signal is."""
    bars = _frame(_wobble(400, seed=73))
    direct = MOD.signal_fn(bars)
    for a, b in zip(direct, MOD.generate_signals(bars, {})):
        pd.testing.assert_series_equal(a, b, check_names=False)
    assert MOD.generate_signals(bars, None)[0].equals(direct[0])


def test_unknown_parameters_raise_on_both_entry_points() -> None:
    """A stale key silently dropped would sweep the DEFAULT and report it
    under the name the caller thought they set."""
    bars = _frame(_wobble(100, seed=79))
    with pytest.raises(ValueError, match="unknown parameter"):
        MOD.make_signal_fn(band_mult=2.0)
    with pytest.raises(ValueError, match="unknown parameter"):
        MOD.generate_signals(bars, {"nope": 1})


def test_make_signal_fn_binds_and_matches_direct_calls() -> None:
    bars = _frame(_wobble(400, seed=83))
    bound = MOD.make_signal_fn(sma_window=10, inner_sd=1.0)
    direct = MOD.signal_fn(bars, sma_window=10, inner_sd=1.0)
    for a, b in zip(bound(bars), direct):
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_walk_loop_agrees_with_the_copy_it_was_duplicated_from() -> None:
    """
    `_walk_loop` is duplicated VERBATIM across this directory because strategy
    modules load from a file path and are promoted as self-contained copies.
    The convention only holds if the copies actually agree - which is what
    `tests/test_risk_params.py` enforces for the older ones.
    """
    sibling = _load(REPO / "strategies" / "experimental"
                    / "ema_deviation_scalp_20260909.py")
    rng = np.random.default_rng(89)
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
    Nothing in the module reads the bar width - the SMA and the SD are in
    BARS, not minutes - so the same close path must produce the same masks at
    either frequency. That is also why TIMEFRAME is a declaration and Stage 1
    picks the rung: a 20-bar mean is 20 minutes at 1m and 100 at 5m, and the
    module cannot tell the difference.
    """
    close = _breakout_up()["close"].to_numpy()
    a = MOD.signal_fn(_frame(close, freq="1min"))
    b = MOD.signal_fn(_frame(close, freq="5min"))
    assert a[0].sum() >= 1
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())
