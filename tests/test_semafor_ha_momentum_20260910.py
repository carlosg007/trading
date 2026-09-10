"""
tests/test_semafor_ha_momentum_20260910.py - HA Smoothed + Semafor + RSI.

Location:  ~/src/trading/tests/test_semafor_ha_momentum_20260910.py

    .venv/bin/python3 -m pytest tests/test_semafor_ha_momentum_20260910.py

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

THE CASE THIS SUITE EXISTS FOR is `test_signals_never_change_when_future_bars
_are_removed`. A Semafor is defined on a CENTRED window and is therefore a
repainter by default: read at the pivot bar it knows the next `swing_lookback`
bars. The module confirms the pivot late instead, and truncation is the only
test that can tell the two apart from the outside - the centred version passes
every other case in this file.
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
               / "semafor_ha_momentum_20260910.py")


def _load(path: Path = MODULE_PATH):
    """Import from the FILE PATH, the way `tier3_workers.load_strategy` does -
    not by package name, which is a path the engine never takes."""
    spec = importlib.util.spec_from_file_location("_semafor_uut", path)
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


def _impulse_pullback(legs: int = 40, seed: int = 5) -> pd.DataFrame:
    """
    Alternating impulse and retrace - the tape this strategy is specified for.

    A pure trend contains no swing lows to confirm and a pure range contains no
    macro lean, so neither exercises the module. Long enough that the triple
    coincidence of pivot, momentum gate and EMA cross occurs at all: it is a
    SELECTIVE setup, and on a few hundred bars its absence says nothing.
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
    silently run an intraday system on DAILY bars.
    """
    assert MOD.TIMEFRAME == "15m"
    assert MOD.TIMEFRAMES == ["5m", "15m", "30m", "1h"]
    assert len(MOD.SYMBOLS) == len(set(MOD.SYMBOLS)) == 19
    assert MOD.TARGET_QUADRANTS == ("Q1", "Q3")
    # The quadrant ids are `mdlib/regimes.py`'s and nothing else's. 0 is
    # UNDEFINED - indicator warm-up - and is not a quadrant.
    assert set(MOD.TARGET_QUADRANTS) <= {"Q1", "Q2", "Q3", "Q4"}


def test_param_grid_only_sweeps_real_parameters() -> None:
    """`load_strategy` rejects unknown parameter names, so a stale PARAM_GRID
    key raises at bind time. Catching it here catches it before a sweep burns
    hours discovering it."""
    unknown = set(MOD.PARAM_GRID) - set(MOD.DEFAULT_PARAMS)
    assert not unknown, f"PARAM_GRID sweeps names signal_fn will refuse: {unknown}"


def test_the_search_is_small_enough_to_report_honestly() -> None:
    """`variants_tested` travels with every result so a Sharpe can be read
    against the size of the search that produced it. A 400-cell grid over the
    same bars is a machine for manufacturing an in-sample Sharpe."""
    combos = int(np.prod([len(v) for v in MOD.PARAM_GRID.values()]))
    assert combos == 36, combos


def test_risk_variants_name_only_real_parameters() -> None:
    """The request's "Version A/B" are risk CONFIGURATIONS, not this
    repository's A/B - see the module docstring. They are still bound through
    `signal_fn`, so a stale key would raise at bind time."""
    assert set(MOD.RISK_VARIANTS) == {"A_fixed", "B_trailing"}
    for name, variant in MOD.RISK_VARIANTS.items():
        unknown = set(variant) - set(MOD.DEFAULT_PARAMS)
        assert not unknown, f"{name} names {unknown}"
    # A_fixed IS the default: the package this module ships as.
    for key, value in MOD.RISK_VARIANTS["A_fixed"].items():
        assert MOD.DEFAULT_PARAMS[key] == value, key
    b = MOD.RISK_VARIANTS["B_trailing"]
    assert b["trailing"] is True and b["sl_atr_mult"] == 2.5
    assert b["use_rsi_zone_exit"] is True
    # NEITHER variant models an ATR target. The RSI zone is the request's
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
    # Every position that opened also closed, EXCEPT one that may still be
    # open when the tape ends - the walk cannot close a position the data
    # stops before. At most one, and on at most one side, because the walk
    # holds one position at a time across both.
    long_open = int(le.sum()) - int(lx.sum())
    short_open = int(se.sum()) - int(sx.sum())
    assert long_open in (0, 1), (int(le.sum()), int(lx.sum()))
    assert short_open in (0, 1), (int(se.sum()), int(sx.sum()))
    assert long_open + short_open <= 1, (
        "two positions open at the end - the walk held both sides at once")


def test_positions_are_never_pyramided_or_reversed() -> None:
    """The walk enters only from flat, so a short signal arriving while long
    is ignored and the long must exit first. A bar carrying both is taken by
    neither side, matching `engine._clean_signals_ls_loop`."""
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
# Causality - the reason this module is not a repainter
# ---------------------------------------------------------------------------
def test_signals_never_change_when_future_bars_are_removed() -> None:
    """
    THE CASE THIS SUITE EXISTS FOR.

    A Semafor is defined on a CENTRED window: a swing low at bar p is the
    lowest of the `swing_lookback` bars on either side of it, which is not
    known until `swing_lookback` more bars have printed. Read at bar p that is
    lookahead, and a backtest built on it buys the exact bottom of every leg.

    Truncation is the only test that can tell the centred form from the
    confirmed one from the outside: every other case in this file passes for
    both. If any signal in the first N bars changes when the bars after N are
    removed, the module read a bar that had not printed.
    """
    bars = _impulse_pullback()
    full = _masks(bars)
    for cut in (400, 900, 1500):
        truncated = _masks(bars.iloc[:cut])
        for name, a, b in zip(("long_entries", "long_exits",
                               "short_entries", "short_exits"),
                              full, truncated):
            assert (a.iloc[:cut].to_numpy() == b.to_numpy()).all(), (
                f"{name} changed in the first {cut} bars when the future was "
                f"removed - the module repaints")


def test_the_pivot_is_confirmed_late_not_read_centred() -> None:
    """
    The pivot is LOCATED `swing_lookback` bars back from the bar that confirms
    it. A V-bottom puts the low at a known index; the mask must be True
    exactly `swing_lookback` bars after it, never on it.
    """
    lookback = 5
    down = np.linspace(110.0, 100.0, 30)
    up = np.linspace(100.0, 112.0, 30)
    bars = _frame(np.concatenate([down, up[1:]]))
    low_bar = 29                              # the V's bottom
    swing_low, _ = MOD._swing_points(bars, lookback)
    fired = np.flatnonzero(swing_low.to_numpy())
    assert fired.size, "no swing low found in a V-bottom"
    assert low_bar not in fired, (
        "the pivot fired ON the low bar - that is the centred, repainting form")
    assert low_bar + lookback in fired, (
        f"expected confirmation at {low_bar + lookback}, got {fired.tolist()}")


def test_no_negative_shift_appears_anywhere_in_the_module() -> None:
    """
    A `.shift(-n)` reads FORWARD, and one of them anywhere in a signal path is
    a repainting module that passes every value-based test.

    Checked on the source rather than on behaviour because behaviour can only
    catch the ones a fixture happens to exercise.
    """
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
    """
    Shape is checked by the caller and a failure RAISES - unlike `indicators`,
    which is wrapped. Causality is this module's responsibility: the filter's
    guarantee that it trains only on trades closed before the candidate is
    undone by a column that reads the future.
    """
    bars = _impulse_pullback()
    full = MOD.ml_features(bars)
    assert len(full) == len(bars)
    assert full.index.equals(bars.index)
    assert np.isfinite(full.to_numpy()).all(), "a non-finite feature"
    cut = 900
    truncated = MOD.ml_features(bars.iloc[:cut])
    assert np.allclose(full.iloc[:cut].to_numpy(), truncated.to_numpy()), (
        "a feature column changed when future bars were removed")


# ---------------------------------------------------------------------------
# The indicators themselves
# ---------------------------------------------------------------------------
def test_the_rsi_is_wilders_not_a_simple_mean() -> None:
    """
    Wilder-smoothed gains and losses, not a simple mean of them. The simple
    form is a different oscillator that crosses 30 on different bars, and this
    module's entire momentum gate is a threshold cross.
    """
    rng = np.random.default_rng(2)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, 400)))
    ours = MOD._rsi(close, 14)
    delta = close.diff()
    simple_gain = delta.clip(lower=0).rolling(14).mean()
    simple_loss = (-delta).clip(lower=0).rolling(14).mean()
    simple = 100.0 - 100.0 / (1.0 + simple_gain / simple_loss)
    tail = slice(50, None)
    assert not np.allclose(ours[tail].to_numpy(), simple[tail].to_numpy()), (
        "the RSI matches the SIMPLE-mean form; it is not Wilder's")
    assert ours.dropna().between(0.0, 100.0).all()


def test_the_heiken_ashi_open_is_the_documented_recursion() -> None:
    """
    `HA_open[i] = (HA_open[i-1] + HA_close[i-1]) / 2`, computed as the EWM it
    is identical to. Checked against an explicit loop so the vectorised form
    cannot drift from the definition it claims to implement.

    Compared from the warm-up onward: the loop seeds at the textbook
    `(open+close)/2` and the EWM at the first valid HA close, and the two
    converge by a factor of two per bar - which is the seed difference the
    module docstring records, not a difference in the rule.
    """
    bars = _impulse_pullback(legs=6)
    HA = MOD._heiken_ashi_smoothed(bars, 20, 1)   # smooth=1 isolates the open
    ha_close = HA["ha_close"].to_numpy()
    ha_open = HA["ha_open"].to_numpy()
    valid = np.flatnonzero(np.isfinite(ha_close))
    start = int(valid[0])
    loop = np.full(len(bars), np.nan)
    loop[start] = ha_close[start]
    for i in range(start + 1, len(bars)):
        loop[i] = (loop[i - 1] + ha_close[i - 1]) / 2.0
    tail = slice(start + 40, None)
    assert np.allclose(ha_open[tail], loop[tail], atol=1e-9), (
        "the HA open is not the documented recursion")


def test_indicators_are_full_length_and_named() -> None:
    """Drawn over the inspector's candles from the same `_layers` call
    `signal_fn` uses, so the chart cannot draw a cross a bar from where the
    entry actually happened."""
    bars = _impulse_pullback(legs=6)
    drawn = MOD.indicators(bars)
    assert drawn, "no indicator series declared"
    for name, series in drawn.items():
        assert isinstance(series, pd.Series), name
        assert len(series) == len(bars), name
        assert series.index.equals(bars.index), name
    assert any("RSI" in n for n in drawn)
    assert any("HA" in n for n in drawn)


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
def test_the_trigger_is_an_event_not_a_state() -> None:
    """A run of bars already above the fast EMA is not a fresh trigger - the
    trigger was the bar that got there - and a state test would fire an entry
    on every one of them."""
    bars = _impulse_pullback(legs=8)
    L = MOD._layers(bars, 20, 5, 12, 9, 14, 30.0, 70.0, 10, True, True, False)
    above = (L["close"] > L["ema_fast"]).sum()
    assert L["cross_up"].sum() < above / 3, (
        f"{int(L['cross_up'].sum())} crosses against {int(above)} bars above "
        f"the EMA - the trigger is behaving like a state")


def test_the_momentum_gate_requires_the_rsi_cross_after_the_low() -> None:
    """
    The ordering, not merely a shared window. Both events inside the last
    `momentum_window` bars also admits a cross that PRECEDED the low it is
    supposed to confirm, which is not the hypothesis.

    Checked by widening the window: a gate that only tested co-occurrence
    would be insensitive to the pivot's own age, so opening the window must
    admit strictly more bars and never fewer.
    """
    bars = _impulse_pullback(legs=10)
    narrow = MOD._layers(bars, 20, 5, 12, 9, 14, 30.0, 70.0, 3,
                         True, True, False)["gate_long"]
    wide = MOD._layers(bars, 20, 5, 12, 9, 14, 30.0, 70.0, 30,
                       True, True, False)["gate_long"]
    assert wide.sum() > narrow.sum(), (
        int(narrow.sum()), int(wide.sum()))
    # Widening only ever ADDS bars: the narrow gate is a subset.
    assert (narrow & ~wide).sum() == 0, "a narrow-window bar the wide one drops"


def test_the_macro_filter_binds_and_can_be_turned_off() -> None:
    """`use_macro_trend` is in PARAM_GRID because the request declared the
    ribbon without saying what it costs. It has to actually bind, or the sweep
    is comparing a setting against itself.

    COMPARED ON THE TRIGGERS, NEVER ON THE TRADE COUNT. CLAUDE.md records why:
    the walk holds one position at a time and ignores a trigger arriving while
    one is open, so declining an early trigger can leave the strategy flat for
    a later one it would have been holding through - and a filter that removed
    candidates can ADD realised entries.
    """
    bars = _impulse_pullback()
    on = MOD._layers(bars, 20, 5, 12, 9, 14, 30.0, 70.0, 10, True, True,
                     False)["trigger_long"]
    off = MOD._layers(bars, 20, 5, 12, 9, 14, 30.0, 70.0, 10, False, True,
                      False)["trigger_long"]
    assert off.sum() > on.sum(), (int(on.sum()), int(off.sum()))
    assert (on & ~off).sum() == 0, "the filter admitted a candidate it vetoes"


def test_the_rsi_zone_exit_is_a_signal_and_only_when_asked() -> None:
    """The request's "RSI opposite zone take-profit". The engine has no target
    ORDERS, so it can only be an exit mask - and it is off in the default
    (A_fixed) configuration."""
    bars = _impulse_pullback(legs=10)
    without = MOD._layers(bars, 20, 5, 12, 9, 14, 30.0, 70.0, 10, True, False,
                          False)["exit_long"]
    with_zone = MOD._layers(bars, 20, 5, 12, 9, 14, 30.0, 70.0, 10, True,
                            False, True)["exit_long"]
    assert with_zone.sum() > without.sum()
    assert (without & ~with_zone).sum() == 0
    assert MOD.DEFAULT_PARAMS["use_rsi_zone_exit"] is False


def test_a_dead_flat_tape_produces_no_trades() -> None:
    """ATR is 0 on a tape that never moved. A zero ATR is a MEASUREMENT, not a
    gap, and entering there would place a bracket of zero width around the
    fill."""
    le, lx, se, sx = _masks(_flat())
    assert not le.any() and not se.any()
    assert not lx.any() and not sx.any()


# ---------------------------------------------------------------------------
# Parameter hygiene
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"ha_period": 1},
    {"fast_ema": 0},
    {"rsi_period": 1},
    {"swing_lookback": 0},
    {"momentum_window": 0},
    {"rsi_oversold": 70.0, "rsi_overbought": 30.0},
    {"rsi_oversold": 0.0},
    {"sl_atr_mult": 0.01},
    {"tp_atr_mult": 0.0},
    {"tp_atr_mult": -1.0},
    {"trailing": "yes"},
])
def test_impossible_parameters_raise_rather_than_running(bad) -> None:
    """Refused where they are passed, rather than as a strange equity curve
    somebody has to explain later."""
    with pytest.raises(ValueError):
        MOD.signal_fn(_impulse_pullback(legs=4), **bad)


def test_unknown_parameters_raise_on_both_entry_points() -> None:
    """A stale grid key silently dropped would sweep the DEFAULT and report it
    under the swept name."""
    bars = _impulse_pullback(legs=4)
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
    bound = MOD.make_signal_fn(swing_lookback=8, momentum_window=15)
    direct = MOD.signal_fn(bars, swing_lookback=8, momentum_window=15)
    for a, b in zip(bound(bars), direct):
        assert a.equals(b)


def test_both_risk_variants_run_and_differ() -> None:
    """The request's two configurations. They must both produce trades and
    must not be the same strategy under two names."""
    bars = _impulse_pullback()
    a = _masks(bars, **MOD.RISK_VARIANTS["A_fixed"])
    b = _masks(bars, **MOD.RISK_VARIANTS["B_trailing"])
    assert a[0].sum() > 0 and a[2].sum() > 0, "A_fixed traded nothing"
    assert b[0].sum() > 0 and b[2].sum() > 0, "B_trailing traded nothing"
    assert not all(x.equals(y) for x, y in zip(a, b)), (
        "the two risk variants produce identical masks")
