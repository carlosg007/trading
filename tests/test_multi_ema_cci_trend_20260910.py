"""
tests/test_multi_ema_cci_trend_20260910.py - the six-EMA ribbon + CCI trigger.

Location:  ~/src/trading/tests/test_multi_ema_cci_trend_20260910.py

    .venv/bin/python3 -m pytest tests/test_multi_ema_cci_trend_20260910.py

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

TWO CASES CARRY MOST OF THE WEIGHT. `test_the_cci_denominator_is_the_mean
_absolute_deviation` pins the indicator against its definition rather than
against itself: CCI computed on a standard deviation reads about 20% smaller,
crosses zero on the same bars, and makes the +/-100 levels this module uses as
a target mean something no chart agrees with. And
`test_signals_never_change_when_future_bars_are_removed` is the only case that
can see a forward-reading window from the outside.
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
               / "multi_ema_cci_trend_20260910.py")


def _load(path: Path = MODULE_PATH):
    """Import from the FILE PATH, the way `tier3_workers.load_strategy` does -
    not by package name, which is a path the engine never takes."""
    spec = importlib.util.spec_from_file_location("_ribbon_uut", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load()


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
        high = close + 0.6
    if low is None:
        low = close - 0.6
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]],
         "high": np.maximum(np.asarray(high, dtype="float64"), close),
         "low": np.minimum(np.asarray(low, dtype="float64"), close),
         "close": close,
         "volume": np.full(n, 1000.0)},
        index=idx)


def _impulse_pullback(legs: int = 30, seed: int = 5) -> pd.DataFrame:
    """
    Alternating impulse and retrace - the tape this strategy is specified for.

    A pure trend never pulls back, so CCI never recrosses zero; a pure range
    never aligns the ribbon. Neither exercises the module.
    """
    rng = np.random.default_rng(seed)
    out, lvl = [], 100.0
    for _ in range(legs):
        out.append(lvl + np.cumsum(rng.normal(0.22, 0.55, 60)))
        lvl = out[-1][-1]
        out.append(lvl + np.cumsum(rng.normal(-0.28, 0.55, 35)))
        lvl = out[-1][-1]
    return _frame(np.concatenate(out))


def _flat(n: int = 300, level: float = 100.0) -> pd.DataFrame:
    """A dead tape: no range, so ATR is 0 and nothing is tradable."""
    return _frame(np.full(n, level), high=np.full(n, level),
                  low=np.full(n, level))


def _masks(bars, **params):
    return MOD.signal_fn(bars, **params)


def _L(bars, **over):
    """`_layers` on the defaults, with overrides - one place for the long
    positional signature."""
    p = {**MOD.DEFAULT_PARAMS, **over}
    return MOD._layers(bars, int(p["ribbon_step"]), int(p["ribbon_count"]),
                       int(p["cci_period"]), float(p["cci_trigger"]),
                       float(p["cci_extreme"]),
                       bool(p["require_strict_stack"]),
                       bool(p["use_ribbon_exit"]),
                       bool(p["use_cci_extreme_exit"]))


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
    """
    TIMEFRAME stays SINGULAR and stays declared. `tier3_workers` reads
    `getattr(module, "TIMEFRAME", None)` and its caller resolves
    `explicit or info["timeframe"] or DEFAULT_TIMEFRAME`, where
    DEFAULT_TIMEFRAME is "1d" - so a module declaring only the plural would
    silently run an intraday ribbon on DAILY bars.
    """
    assert MOD.TIMEFRAME == "15m"
    assert MOD.TIMEFRAMES == ["5m", "15m", "30m", "1h"]
    assert len(MOD.SYMBOLS) == len(set(MOD.SYMBOLS)) == 19
    assert MOD.TARGET_QUADRANTS == ("Q1", "Q3")
    # The quadrant ids are `mdlib/regimes.py`'s and nothing else's. 0 is
    # UNDEFINED - indicator warm-up - and is not a quadrant.
    assert set(MOD.TARGET_QUADRANTS) <= {"Q1", "Q2", "Q3", "Q4"}


def test_the_ribbon_is_the_requested_six_emas() -> None:
    """4/8/12/16/20/24, derived from one sweepable spacing rather than listed
    - but the DEFAULT must be exactly what was asked for."""
    p = MOD.DEFAULT_PARAMS
    assert MOD._ribbon_periods(p["ribbon_step"], p["ribbon_count"]) == [
        4, 8, 12, 16, 20, 24]
    assert p["ribbon_count"] == 6
    # Every swept spacing still yields six evenly spaced means.
    for step in MOD.PARAM_GRID["ribbon_step"]:
        periods = MOD._ribbon_periods(step, p["ribbon_count"])
        assert len(periods) == 6
        assert periods == sorted(periods)
        assert len(set(periods)) == 6, f"step {step} repeats a period"


def test_param_grid_only_sweeps_real_parameters() -> None:
    """`load_strategy` rejects unknown parameter names, so a stale PARAM_GRID
    key raises at bind time. Catching it here catches it before a sweep burns
    hours discovering it."""
    unknown = set(MOD.PARAM_GRID) - set(MOD.DEFAULT_PARAMS)
    assert not unknown, f"PARAM_GRID sweeps names signal_fn will refuse: {unknown}"


def test_the_search_is_small_enough_to_report_honestly() -> None:
    """`variants_tested` travels with every result so a Sharpe can be read
    against the size of the search that produced it."""
    combos = int(np.prod([len(v) for v in MOD.PARAM_GRID.values()]))
    assert combos == 36, combos


def test_risk_variants_name_only_real_parameters() -> None:
    """The request's "Version A/B" are risk CONFIGURATIONS, not this
    repository's A/B - see the module docstring."""
    assert set(MOD.RISK_VARIANTS) == {"A_ribbon", "B_trailing"}
    for name, variant in MOD.RISK_VARIANTS.items():
        unknown = set(variant) - set(MOD.DEFAULT_PARAMS)
        assert not unknown, f"{name} names {unknown}"
    for key, value in MOD.RISK_VARIANTS["A_ribbon"].items():
        assert MOD.DEFAULT_PARAMS[key] == value, key
    b = MOD.RISK_VARIANTS["B_trailing"]
    assert b["trailing"] is True and b["sl_atr_mult"] == 2.5
    assert b["use_cci_extreme_exit"] is True
    # NEITHER variant models an ATR target. The CCI extreme is the request's
    # take-profit and it is a SIGNAL, because the engine has no target orders.
    assert all(v["tp_atr_mult"] is None for v in MOD.RISK_VARIANTS.values())


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


def test_the_risk_keys_are_the_three_promote_py_writes() -> None:
    """`backtest/promote.py` lifts exactly these three into the promoted
    package's `risk` block. A module spelling one of them differently gets a
    risk block that describes a strategy nobody ran."""
    for key in ("sl_atr_mult", "tp_atr_mult", "trailing"):
        assert key in MOD.DEFAULT_PARAMS, key


# ---------------------------------------------------------------------------
# The indicator, against its definition
# ---------------------------------------------------------------------------
def test_the_cci_denominator_is_the_mean_absolute_deviation() -> None:
    """
    CCI is `(TP - SMA(TP)) / (0.015 * MAD)`, and MAD is the mean ABSOLUTE
    DEVIATION - not the standard deviation.

    Pinned against an independent `rolling.apply` of the definition rather
    than against the module's own helper, so the fast path cannot drift from
    what it claims to compute. The SD form is checked to be genuinely
    different: it crosses zero on the same bars, so no entry test would ever
    notice, but it reads about 20% smaller and the +/-100 levels this module
    uses as Version B's target would then mean something no chart agrees with.
    """
    bars = _impulse_pullback(legs=6)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    sma = tp.rolling(14).mean()
    mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    reference = (tp - sma) / (MOD.CCI_CONSTANT * mad)

    ours = MOD._cci(bars, 14)
    assert np.allclose(ours.dropna().to_numpy(),
                       reference.dropna().to_numpy(), atol=1e-9)

    wrong = (tp - sma) / (MOD.CCI_CONSTANT * tp.rolling(14).std())
    both = pd.concat([ours, wrong], axis=1).dropna()
    assert not np.allclose(both.iloc[:, 0], both.iloc[:, 1]), (
        "the SD form is indistinguishable here; this case proves nothing")
    assert MOD.CCI_CONSTANT == 0.015


def test_the_mean_absolute_deviation_window_ends_at_its_own_bar() -> None:
    """Window `i` spans `i - period + 1 .. i`. A window centred on i, or one
    that reaches forward, would make every CCI reading a repainter."""
    values = pd.Series(np.arange(20, dtype="float64"))
    mad = MOD._mean_abs_dev(values, 5)
    assert mad.iloc[:4].isna().all(), "a value before the window is full"
    # On 0..19 every full window is five consecutive integers, whose mean
    # absolute deviation about their own mean is 1.2 exactly.
    assert np.allclose(mad.iloc[4:].to_numpy(), 1.2)


def test_indicators_draw_every_ribbon_member() -> None:
    """The ALIGNMENT is the signal, and a chart showing two of six means
    cannot show a reader why a bar was aligned or was not."""
    bars = _impulse_pullback(legs=4)
    drawn = MOD.indicators(bars)
    for period in (4, 8, 12, 16, 20, 24):
        assert f"EMA({period})" in drawn, period
    assert any(n.startswith("CCI(") for n in drawn)
    for name, series in drawn.items():
        assert isinstance(series, pd.Series) and len(series) == len(bars), name
        assert series.index.equals(bars.index), name


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
def test_the_strict_stack_is_the_pairwise_chain() -> None:
    """
    EMA_4 > EMA_8 > ... > EMA_24, checked as the chain the request names.

    On a long clean uptrend the ribbon stacks; on the mirror it inverts. Both
    are asserted because a short is not a long with the sign flipped, and a
    stack test written as `~stack_long` would be true on every tangled bar.
    """
    up = _frame(np.linspace(100.0, 160.0, 400))
    L = _L(up)
    assert L["stack_long"].iloc[-50:].all(), "a clean uptrend did not stack"
    assert not L["stack_short"].iloc[-50:].any()

    down = _frame(np.linspace(160.0, 100.0, 400))
    L = _L(down)
    assert L["stack_short"].iloc[-50:].all(), "a clean downtrend did not stack"
    assert not L["stack_long"].iloc[-50:].any()


def test_the_relaxed_alignment_admits_more_than_the_strict_one() -> None:
    """The request offers the two with an "or", so the relaxed form is a
    SUPERSET: every strictly stacked bar is aligned, and some others are
    too. `require_strict_stack` is in PARAM_GRID because the request did not
    say which binds."""
    bars = _impulse_pullback(legs=12)
    strict = _L(bars, require_strict_stack=True)["align_long"]
    either = _L(bars, require_strict_stack=False)["align_long"]
    assert (strict & ~either).sum() == 0, "a strictly stacked bar was dropped"
    assert either.sum() > strict.sum(), (int(strict.sum()), int(either.sum()))


def test_the_trigger_is_an_event_not_a_state() -> None:
    """CCI above zero is true for every bar of a leg; the event is the bar it
    got there. A state test would fire an entry on every one of them."""
    bars = _impulse_pullback(legs=12)
    L = _L(bars)
    above = (L["cci"] > 0.0).sum()
    assert L["cci_up"].sum() < above / 3, (
        f"{int(L['cci_up'].sum())} crosses against {int(above)} bars above "
        f"zero - the trigger is behaving like a state")


def test_the_ribbon_condition_actually_gates_the_trigger() -> None:
    """
    The ribbon is the CONDITION and the CCI cross is the EVENT. If every cross
    became a trigger the ribbon would be decoration.

    COMPARED ON THE TRIGGERS, NEVER ON THE TRADE COUNT. CLAUDE.md records why:
    the walk holds one position at a time and ignores a trigger arriving while
    one is open, so declining an early trigger can leave the strategy flat for
    a later one it would have been holding through - and a filter that removed
    candidates can ADD realised entries.
    """
    bars = _impulse_pullback(legs=12)
    L = _L(bars)
    assert L["trigger_long"].sum() < L["cci_up"].sum(), (
        "every zero-line cross became a trigger; the ribbon gates nothing")
    assert (L["trigger_long"] & ~L["cci_up"]).sum() == 0
    assert (L["trigger_long"] & ~L["align_long"]).sum() == 0


def test_the_ribbon_exit_is_a_signal_and_can_be_turned_off() -> None:
    """The request's stop "anchored just below EMA_24". The engine has no stop
    ORDERS, so an average-anchored stop can only be an exit mask - a close
    back through the slowest EMA."""
    bars = _impulse_pullback(legs=12)
    without = _L(bars, use_ribbon_exit=False)["exit_long"]
    with_exit = _L(bars, use_ribbon_exit=True)["exit_long"]
    assert with_exit.sum() > without.sum()
    assert (without & ~with_exit).sum() == 0
    assert MOD.DEFAULT_PARAMS["use_ribbon_exit"] is True


def test_the_cci_extreme_exit_is_off_in_the_baseline() -> None:
    """Version B's target. Off in A_ribbon, which is DEFAULT_PARAMS."""
    bars = _impulse_pullback(legs=12)
    without = _L(bars, use_cci_extreme_exit=False)["exit_long"]
    with_exit = _L(bars, use_cci_extreme_exit=True)["exit_long"]
    assert with_exit.sum() > without.sum()
    assert (without & ~with_exit).sum() == 0
    assert MOD.DEFAULT_PARAMS["use_cci_extreme_exit"] is False


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
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert mask.index.equals(bars.index)


def test_both_sides_actually_trade() -> None:
    """A bidirectional module whose short side never fires is a long-only
    strategy with a plausible curve and a mask nobody checked."""
    bars = _impulse_pullback()
    le, lx, se, sx = _masks(bars)
    assert le.sum() > 0, "the long side never fired"
    assert se.sum() > 0, "the short side never fired"
    # At most one position may still be open when the tape ends - the walk
    # cannot close a position the data stops before - and on at most one side,
    # because it holds one position at a time across both.
    long_open = int(le.sum()) - int(lx.sum())
    short_open = int(se.sum()) - int(sx.sum())
    assert long_open in (0, 1) and short_open in (0, 1)
    assert long_open + short_open <= 1


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
    The only case that can see a forward-reading window from the outside.

    Every rolling window in this module must end at the bar it describes -
    including the sliding-window MAD, which is where a centred window would be
    easiest to write by accident.
    """
    bars = _impulse_pullback()
    full = _masks(bars)
    for cut in (600, 1500, 3000):
        truncated = _masks(bars.iloc[:cut])
        for name, a, b in zip(("long_entries", "long_exits",
                               "short_entries", "short_exits"),
                              full, truncated):
            assert (a.iloc[:cut].to_numpy() == b.to_numpy()).all(), (
                f"{name} changed in the first {cut} bars when the future was "
                f"removed - the module repaints")


def test_no_negative_shift_appears_anywhere_in_the_module() -> None:
    """A `.shift(-n)` reads FORWARD, and one anywhere in a signal path is a
    repainting module that passes every value-based test. Checked on the
    source because behaviour only catches what a fixture exercises."""
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "shift"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                pytest.fail(f"negative .shift() at line {node.lineno} reads "
                            f"forward")


def test_ml_features_are_causal_and_shaped_to_the_bars() -> None:
    """Shape is checked by the caller and a failure RAISES - unlike
    `indicators`, which is wrapped. Causality is this module's
    responsibility."""
    bars = _impulse_pullback()
    full = MOD.ml_features(bars)
    assert len(full) == len(bars)
    assert full.index.equals(bars.index)
    assert np.isfinite(full.to_numpy()).all(), "a non-finite feature"
    # One normalised gap per consecutive pair - five for a six-mean ribbon.
    gaps = [c for c in full.columns if c.startswith("gap_")]
    assert len(gaps) == 5, gaps
    cut = 1500
    truncated = MOD.ml_features(bars.iloc[:cut])
    assert np.allclose(full.iloc[:cut].to_numpy(), truncated.to_numpy()), (
        "a feature column changed when future bars were removed")


# ---------------------------------------------------------------------------
# Parameter hygiene
# ---------------------------------------------------------------------------
def test_a_dead_flat_tape_produces_no_trades() -> None:
    """ATR is 0 on a tape that never moved, and the CCI denominator is 0 too.
    A zero ATR is a MEASUREMENT, not a gap, and entering there would place a
    bracket of zero width around the fill."""
    le, lx, se, sx = _masks(_flat())
    assert not le.any() and not se.any()
    assert not lx.any() and not sx.any()


@pytest.mark.parametrize("bad", [
    {"ribbon_step": 0},
    {"ribbon_count": 1},
    {"cci_period": 1},
    {"cci_extreme": 0.0},
    {"cci_trigger": 150.0},
    {"sl_atr_mult": 0.01},
    {"tp_atr_mult": 0.0},
    {"tp_atr_mult": -1.0},
    {"trailing": "yes"},
])
def test_impossible_parameters_raise_rather_than_running(bad) -> None:
    """Refused where they are passed, rather than as a strange equity curve
    somebody has to explain later."""
    with pytest.raises(ValueError):
        MOD.signal_fn(_impulse_pullback(legs=3), **bad)


def test_unknown_parameters_raise_on_both_entry_points() -> None:
    """A stale grid key silently dropped would sweep the DEFAULT and report it
    under the swept name."""
    bars = _impulse_pullback(legs=3)
    with pytest.raises(ValueError):
        MOD.make_signal_fn(no_such_param=1)
    with pytest.raises(ValueError):
        MOD.generate_signals(bars, {"no_such_param": 1})


def test_generate_signals_delegates_to_signal_fn() -> None:
    """It delegates rather than reimplementing, so the two can never disagree
    about what a signal is."""
    bars = _impulse_pullback(legs=6)
    direct = MOD.signal_fn(bars, **MOD.DEFAULT_PARAMS)
    adapted = MOD.generate_signals(bars)
    for a, b in zip(direct, adapted):
        assert a.equals(b)


def test_make_signal_fn_binds_and_matches_direct_calls() -> None:
    """The parameterised form `load_strategy` prefers."""
    bars = _impulse_pullback(legs=6)
    bound = MOD.make_signal_fn(ribbon_step=6, cci_period=20)
    direct = MOD.signal_fn(bars, ribbon_step=6, cci_period=20)
    for a, b in zip(bound(bars), direct):
        assert a.equals(b)


def test_both_risk_variants_run_and_differ() -> None:
    """The request's two configurations. They must both produce trades and
    must not be the same strategy under two names."""
    bars = _impulse_pullback()
    a = _masks(bars, **MOD.RISK_VARIANTS["A_ribbon"])
    b = _masks(bars, **MOD.RISK_VARIANTS["B_trailing"])
    assert a[0].sum() > 0 and a[2].sum() > 0, "A_ribbon traded nothing"
    assert b[0].sum() > 0 and b[2].sum() > 0, "B_trailing traded nothing"
    assert not all(x.equals(y) for x, y in zip(a, b)), (
        "the two risk variants produce identical masks")
