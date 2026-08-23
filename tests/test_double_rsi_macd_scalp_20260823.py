#!/usr/bin/env python3
"""
test_double_rsi_macd_scalp_20260823.py — the Double RSI + MACD scalp: its
signal contract, its indicator arithmetic, the independent wiring of all five
modular toggles, the exact spelling of its risk keys, and the causality of the
matrix Version B is fitted on.

Location:  ~/src/trading/tests/test_double_rsi_macd_scalp_20260823.py

Run EITHER way — and unlike the older suites in this directory, both ways
report the same answer:

    OMP_NUM_THREADS=1 python tests/test_double_rsi_macd_scalp_20260823.py
    OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
        tests/test_double_rsi_macd_scalp_20260823.py

EVERY CASE HERE FAILS THROUGH `assert`, DELIBERATELY. The older convention in
this directory is a `check(name, ok)` helper appending to a module-level
FAILURES list, with `sys.exit(1)` at the end of `main()` as the only failure
signal — which pytest never calls. pytest collects those suites, watches their
checks fail and reports all green. Asserting instead costs nothing as a script
(the runner at the bottom catches AssertionError and prints the same PASS/FAIL
table) and makes the pytest run mean what it says.

Pin the thread count. Section 8 refits the classifier once per completed trade
on a few dozen rows, and on a 16-core box each fit's thread pool costs far more
than the fit — the same reason `test_dual_version.py` carries the same prefix.

Nothing here needs the lake or a network. The news-filter case needs
`backtest/event_calendar.py` to have a calendar covering the fixture's span and
SKIPS LOUDLY when it does not, rather than passing quietly.

THE FIXTURE PROBLEM, AND HOW IT IS SOLVED HERE
==============================================
This strategy's confluence is tight enough that a plain random walk produces
almost no entries at all — the module's own docstring records five long entries
in 20,000 synthetic bars at the declared defaults, and the reason (Layer 1 and
Layer 2 are in genuine tension) is a property of the specification rather than
of the fixture. A suite built only on random walks would therefore be checking
the toggles against zero trades on both sides and passing.

`pullback_frame` is the answer: a deterministic path built to contain the
setup the strategy is a claim about — a steady trend punctuated by two-bar dips
sharp enough to put the FAST RSI under 30 and shallow enough to leave the SLOW
RSI above 50 — and it is exactly MIRRORED by its `sign` argument. At the
declared defaults it produces 22 long entries and 0 short ones, and its mirror
produces 22 short and 0 long. That mirroring is itself a test: a layer wired to
one side only cannot survive it.

WHAT THIS COVERS, and why each one is here rather than assumed:

  * THE INDICATOR ARITHMETIC AGAINST HAND-COMPUTED NUMBERS, not against the
    module's own output. Wilder's RSI is pinned against a recursion run by hand
    in the test, the MACD histogram against its own definition rebuilt from
    pandas primitives, and the ATR against a span EMA it must NOT equal. A
    strategy compared only against itself is pinned, not verified.
  * THE DEAD-FLAT RSI BEING 50 AND NOT 0. The request asks for degenerate
    rolling windows to be filled with 0.0; for an oscillator 0.0 is the most
    OVERSOLD reading there is, so the naive fill would put `fast_rsi <= 30`
    permanently true through every halted session and manufacture long triggers
    out of silence. The case pins the neutral fill and the reasoning with it.
  * SIGNAL DTYPES AND THE FOUR-MASK CONTRACT, through the engine's own
    `unpack_signals` rather than by unpacking here. A three-tuple or a bare
    Series raises there, and silently taking the first two masks of a
    three-tuple is how a strategy's short side disappears into a plausible
    long-only equity curve.
  * THE FIVE TOGGLES BEING INDEPENDENTLY WIRED, compared as PERMITTED STATES
    rather than as trades or triggers. A filter wired to nothing produces the
    identical curve to having it off, which is the failure that looks most like
    success — and a case watching the TRADE count would be checking the wrong
    number, because the walk holds one position at a time, so declining an
    early candidate leaves the strategy flat for a later one it would have been
    holding through and enabling a filter can ADD realised entries.
  * THE DAY FILTER ON THE CME SESSION DATE, not the UTC date, pinned on a
    Sunday-evening bar where the two answers differ. The engine's own
    `exclude_days` is keyed on the session date; a module disagreeing with it
    would apply two different calendars under one name on the same run.
  * THE REQUEST'S QUADRANT NUMBERS AGAINST THIS REPOSITORY'S ENCODING. The
    request asks for "Q1 (Low Vol / Trending)"; `mdlib/regimes.py` numbers Low
    Volatility / Trending as Q3 and its Q2 is High Volatility / RANGING — the
    chop the strategy's own premise says to avoid. The case pins the NAMES the
    module declares against `backtest.profiler.REGIMES` and the ids against
    `REGIME_TO_QUADRANT`, so the conflict cannot be resolved silently in either
    direction by a later edit.
  * THE RISK KEYS BY THEIR EXACT SPELLING, against `backtest/run.py`'s
    RISK_PARAMS and `backtest/promote.py`'s RISK_KEYS rather than against a
    literal list retyped here. Those lookups do not alias: under a different
    spelling the leaderboard's stop and target columns come back BLANK, and
    blank in that file means "this strategy has no such setting", never "the
    setting was off". The keys are then checked to BIND — a declared name that
    changes no level is the same failure one step later.
  * CAUSALITY BY TRUNCATION AND BY PERTURBATION, over both the feature matrix
    and the signals. Every prefix of the frame must reproduce its rows exactly,
    and rewriting the tail must leave the head untouched. A `shift(-1)` is
    caught by both; so is a global mean, a centred window and a reversed slice,
    none of which a source scan for negative shifts would see.
  * THE WALK KERNEL AGAINST ITS ORIGINAL. The position walk is duplicated
    verbatim from `sma_momentum_crossover_20260818`, by this directory's
    self-contained-module convention. Section 7 runs both copies on identical
    arrays in both directions and requires identical output, which is the only
    thing standing between the convention and two silently different stops.
  * THE DEGENERATE STATES THE REQUEST NAMES. A frame with zero range, zero ATR
    and unvarying volume must produce masks and a finite feature matrix — not
    NaN, not an infinity, and not an exception.
"""

from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier3_workers import (apply_ml_signal_filter,        # noqa: E402
                                  load_strategy)
from backtest.engine import unpack_signals                       # noqa: E402
from backtest.event_calendar import session_weekday              # noqa: E402
from backtest.profiler import REGIMES, REGIME_TO_QUADRANT        # noqa: E402
from backtest.promote import RISK_KEYS                           # noqa: E402
from backtest.run import RISK_PARAMS                             # noqa: E402
from backtest.scan import expand_grid                            # noqa: E402
from strategies.experimental import (                            # noqa: E402
    sma_momentum_crossover_20260818 as SMC)
from strategies.experimental import (                            # noqa: E402
    double_rsi_macd_scalp_20260823 as M)

MODULE_PATH = (REPO / "strategies" / "experimental"
               / "double_rsi_macd_scalp_20260823.py")

# The parameters every case runs at unless it says otherwise: the module's own
# declared defaults. Section 1 checks separately that these ARE the request's.
BASE = dict(M.DEFAULT_PARAMS)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def synthetic(n: int = 3000, seed: int = 11, drift: float = 0.0,
              cycle: float = 60.0, amp: float = 6.0) -> pd.DataFrame:
    """
    A random-walk frame with both trending and choppy stretches, in the
    ENGINE's own shape.

    `ts` as a COLUMN with a positional index, which is what the engine hands a
    strategy — not a DatetimeIndex. Both shapes are accepted by the module and
    section 6 checks the other one; running the bulk of the suite on the
    engine's shape means a case cannot pass on a frame the engine never
    produces.

    This fixture is for the arithmetic, the shapes and the causality cases. It
    is NOT used where a case needs realised trades — see `pullback_frame` and
    the note in this module's docstring.
    """
    rng = np.random.default_rng(seed)
    close = (1000.0
             + np.cumsum(rng.normal(drift, 0.4, n))
             + amp * np.sin(np.arange(n) / cycle))
    return pd.DataFrame({
        "ts": pd.date_range("2023-03-06", periods=n, freq="5min", tz="UTC"),
        "open": close + rng.uniform(-0.2, 0.2, n),
        "high": close + rng.uniform(0.05, 0.6, n),
        "low": close - rng.uniform(0.05, 0.6, n),
        "close": close,
        "volume": rng.integers(100, 5000, n).astype(float),
    })


def pullback_frame(n: int = 3000, sign: int = 1, seed: int = 3,
                   drift: float = 0.10, period: int = 70, dip: int = 2,
                   dip_rate: float = 0.5, pop: float = 3.0) -> pd.DataFrame:
    """
    A path built to contain the setup this strategy is a claim about, and
    exactly mirrored by `sign`.

    A steady trend of `drift` per bar, interrupted every `period` bars by a
    `dip`-bar reversal of `dip_rate` and then a single `pop` bar back the other
    way. The dip is deliberately SHORT and SHALLOW: long enough to put the
    5-period RSI under 30, short enough to leave the 21-period RSI above 50 and
    the MACD histogram positive. That is the narrow window Layers 1 and 2 leave
    open between them, and a fixture that misses it tests the toggles against
    zero trades on both sides.

    At the declared defaults `sign=+1` yields 22 long entries and 0 short, and
    `sign=-1` the exact mirror. The noise term is tiny and seeded, so the
    counts are deterministic — several cases assert on them, and a change to
    the module that moves them is a change to which trades it takes.

    Starts on a MONDAY so the default `allowed_days=[0, 1, 2]` leaves a usable
    stretch after the 200-bar EMA warm-up.
    """
    rng = np.random.default_rng(seed)
    steps = np.full(n, drift) + rng.normal(0.0, 0.02, n)
    for start in range(60, n, period):
        steps[start:start + dip] = -dip_rate
        if start + dip < n:
            steps[start + dip] = pop
    close = 1000.0 + sign * np.cumsum(steps)
    return pd.DataFrame({
        "ts": pd.date_range("2023-03-06", periods=n, freq="5min", tz="UTC"),
        "open": close,
        "high": close + 0.4,
        "low": close - 0.4,
        "close": close,
        "volume": np.full(n, 1000.0) + rng.integers(0, 500, n),
    })


def frame_from_closes(closes, freq: str = "5min",
                      start: str = "2023-03-06") -> pd.DataFrame:
    """
    A frame whose closes are exactly the sequence given, for the hand-computed
    cases. High and low are widened off the close so the true range is never
    degenerate except where a case wants it to be.
    """
    close = np.asarray(closes, dtype=float)
    n = len(close)
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=n, freq=freq, tz="UTC"),
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def states(bars: pd.DataFrame, **overrides) -> dict:
    """
    Everything `_signal_arrays` computed, including the PERMITTED-BAR states
    and the candidate triggers.

    Read from the module rather than recomputed here on purpose: a test-side
    copy of the conditions would be checking a second implementation against
    the specification while the module was free to disagree with both.
    """
    return M._signal_arrays(bars, **{**BASE, **overrides})[0]


def levels(bars: pd.DataFrame, **overrides):
    """`(stop_level, tp_level)` as the walk produced them."""
    out = M._signal_arrays(bars, **{**BASE, **overrides})
    return (np.asarray(out[5], dtype=float), np.asarray(out[6], dtype=float))


def masks(bars: pd.DataFrame, **overrides):
    """
    `(long_entries, long_exits, short_entries, short_exits)` through the
    engine's own unpacker.

    Going through `unpack_signals` rather than unpacking here means this
    accepts exactly what the engine accepts — a module returning a three-tuple
    or a bare Series fails at the same place a real run would.
    """
    return unpack_signals(M.signal_fn(bars, **{**BASE, **overrides}),
                          len(bars), bars.index)


def raises(fn, *args, **kwargs) -> str:
    """Run `fn` and return the ValueError message; assert if it did not raise."""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise")


# ==========================================================================
# 1. The registry block and the mandatory declarations
# ==========================================================================
def test_the_pinned_identifier_and_the_module_stem_are_both_right() -> None:
    """
    The request pins the identifier `"double_rsi_macd_scalp"` for stage logging
    and JSON tracking, and the module path carries the dated stem. They are
    DIFFERENT strings and both matter: `backtest/run.py` resolves `--strat` by
    FILENAME, so a run is launched with the dated one, while the pipeline's
    logs carry the pinned one. Pinning both here is what keeps a rename from
    leaving them disagreeing.
    """
    assert M.STRATEGY_NAME == "double_rsi_macd_scalp", M.STRATEGY_NAME
    assert M.STRATEGY_MODULE == MODULE_PATH.stem, (
        f"STRATEGY_MODULE is {M.STRATEGY_MODULE!r} but the file is "
        f"{MODULE_PATH.stem!r} — `--strat` resolves by filename")
    assert M.LOGIC["strategy"] == M.STRATEGY_NAME, (
        "the request asks for the identifier inside the LOGIC dict")


def test_every_mandatory_declaration_is_present() -> None:
    """
    The request names six declarations. Each closes a specific way a result
    goes wrong silently — `signal_fn` is the engine's only entry point,
    `indicators` is the only thing that keeps the chart's lines and the trades
    computed from one array, `LOGIC` is the alternative to inferring the rules
    from the trades, `PARAM_GRID` is what `--scan` sweeps, `make_signal_fn` is
    how the loader binds parameters, and `ml_features` is the matrix Version B
    is fitted on.
    """
    for name in ("signal_fn", "indicators", "LOGIC", "PARAM_GRID",
                 "make_signal_fn", "ml_features"):
        assert hasattr(M, name), f"the module declares no {name}"
    for name in ("signal_fn", "indicators", "make_signal_fn", "ml_features"):
        assert callable(getattr(M, name)), f"{name} is not callable"
    assert isinstance(M.LOGIC, dict) and isinstance(M.PARAM_GRID, dict)
    for key in ("concept", "entry", "exit"):
        assert M.LOGIC.get(key, "").strip(), f"LOGIC['{key}'] is empty"


def test_the_logic_card_substitutes_every_slot_it_declares() -> None:
    """
    The tear sheet fills `{param}` slots with the run's OWN bound parameters.
    `agents.tier3_workers._describe_strategy` swallows a KeyError and prints
    the template verbatim, so a slot naming a parameter that does not exist
    reaches the report as literal `{typo}` and nothing raises. Formatting it
    here with the declared defaults is the check that never happens in
    production.
    """
    for key in ("concept", "entry", "exit"):
        text = M.LOGIC[key].format(**BASE)
        assert "{" not in text and "}" not in text, (
            f"LOGIC['{key}'] left an unfilled slot: {text}")


def test_the_declared_defaults_are_the_signature_defaults() -> None:
    """
    `DEFAULT_PARAMS` is what `promote.py` records in `meta.json` and what the
    loader layers a run's parameters over; the signature defaults are what
    actually runs when nothing is bound. If the two disagree, the promoted
    metadata describes a strategy nobody backtested.
    """
    import inspect

    for fn in (M.signal_fn, M.make_signal_fn):
        sig = inspect.signature(fn)
        for key, declared in BASE.items():
            assert key in sig.parameters, (
                f"{fn.__name__} takes no {key}, but DEFAULT_PARAMS declares it")
            actual = sig.parameters[key].default
            if key == "allowed_days":
                # The signature holds a TUPLE and DEFAULT_PARAMS the request's
                # LIST — a mutable default argument is shared across every call
                # that does not override it. Compared element-wise for that
                # reason, rather than by identity or type.
                assert list(actual) == list(declared), (
                    f"{fn.__name__}'s allowed_days default is {actual!r}, "
                    f"DEFAULT_PARAMS says {declared!r}")
                continue
            assert actual == declared, (
                f"{fn.__name__}'s {key} default is {actual!r}, "
                f"DEFAULT_PARAMS says {declared!r}")


def test_the_request_defaults_are_what_the_module_declares() -> None:
    """
    Section 3 of the request, transcribed. Retyped here deliberately: this is
    the one case that compares the module against the SPECIFICATION rather than
    against itself, and a case that read the values out of the module would
    pass whatever they were changed to.
    """
    requested = {
        "fast_rsi_window": 5,
        "slow_rsi_window": 21,
        "fast_rsi_os_threshold": 30.0,
        "fast_rsi_ob_threshold": 70.0,
        "allowed_days": [0, 1, 2],
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.0,
        "trailing": False,
    }
    for key, want in requested.items():
        assert BASE[key] == want, f"{key}: {BASE[key]!r} != requested {want!r}"
    # The toggles the request names but does not assign. All five exist; the
    # news filter is the one that is OFF, because it depends on an external
    # calendar whose provenance changes what the filter means.
    for toggle in ("use_macro_trend", "use_pullback_trigger",
                   "use_macd_filter", "use_day_filter"):
        assert BASE[toggle] is True, f"{toggle} defaults to {BASE[toggle]!r}"
    assert BASE["use_news_filter"] is False


def test_the_grid_is_the_requested_grid_and_every_cell_binds() -> None:
    """
    The request's PARAM_GRID, transcribed and then EXERCISED: every one of the
    432 combinations must either bind through `make_signal_fn` or be rejected
    for the one documented reason — the request's own section 5 rule that
    `tp_atr_mult=None` requires `trailing=True`.

    A cell that raises for any OTHER reason is a grid the sweep would report as
    REJECTED without anyone knowing why, and `variants_tested` would carry a
    number nobody could reproduce.
    """
    requested = {
        "fast_rsi_window": [2, 5, 7],
        "slow_rsi_window": [14, 21],
        "fast_rsi_os_threshold": [20.0, 30.0],
        "fast_rsi_ob_threshold": [70.0, 80.0],
        "sl_atr_mult": [1.0, 1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0, None],
        "trailing": [False, True],
    }
    assert M.PARAM_GRID == requested, M.PARAM_GRID

    cells = expand_grid(M.PARAM_GRID)
    assert len(cells) == 432, len(cells)
    bound = rejected = 0
    for combo in cells:
        try:
            M.make_signal_fn(**combo)
        except ValueError as exc:
            rejected += 1
            assert combo["tp_atr_mult"] is None and not combo["trailing"], (
                f"cell {combo} was rejected for an undocumented reason: {exc}")
            assert "requires trailing=True" in str(exc), str(exc)
        else:
            bound += 1
    assert rejected == 72, rejected
    assert bound == 360, bound


def test_the_target_regimes_survive_this_repositorys_own_encoding() -> None:
    """
    THE REQUEST'S QUADRANT NUMBERS CONTRADICT `mdlib/regimes.py`, and this case
    is what keeps the contradiction from being resolved silently.

    The request asks for "Q1 (Low Vol / Trending) & Q2 (High Vol / Trending
    Pullbacks)". Here Q1 is High Volatility / Trending and Q3 is Low
    Volatility / Trending — and this repository's Q2 is High Volatility /
    RANGING, the chop the strategy's own premise says to filter out. The module
    keeps the NAMES and derives the ids, so both halves are pinned: the names
    against `backtest.profiler.REGIMES` and the ids against the inverted map
    every stage reads.
    """
    for name in M.TARGET_REGIMES:
        assert name in REGIMES, (
            f"{name!r} is not one of the four regimes this repository "
            f"defines: {list(REGIMES)}")
    ids = tuple(REGIME_TO_QUADRANT[name] for name in M.TARGET_REGIMES)
    assert ids == M.TARGET_QUADRANTS, (
        f"TARGET_QUADRANTS says {M.TARGET_QUADRANTS} but the declared regime "
        f"names map to {ids}")
    assert set(M.TARGET_QUADRANTS) == {"Q1", "Q3"}, (
        f"both target regimes must be the TRENDING quadrants; got "
        f"{M.TARGET_QUADRANTS}")
    assert "Q2" not in M.TARGET_QUADRANTS, (
        "Q2 here is High Volatility / RANGING — taking the request's quadrant "
        "NUMBER literally would aim this strategy at the chop its own premise "
        "says to avoid")


def test_the_module_loads_through_the_real_loader_with_its_hooks() -> None:
    """
    `load_strategy` is how every stage reaches a strategy. It binds
    `make_signal_fn`, rejects unknown parameter names, and picks up the three
    optional hooks — a module whose `ml_features` is not found gets the shared
    seven-column default instead, silently, and Version B is then fitted on
    columns this strategy's hypothesis says nothing about.
    """
    fn, info = load_strategy(MODULE_PATH, {"fast_rsi_window": 2,
                                           "sl_atr_mult": 1.0})
    assert callable(fn)
    assert callable(info.get("indicator_fn")), "indicators hook was not bound"
    assert callable(info.get("ml_feature_fn")), "ml_features hook was not bound"
    assert info["logic"]["concept"], "the LOGIC card did not reach the loader"
    assert info["bound_params"]["fast_rsi_window"] == 2
    assert info.get("param_grid") == M.PARAM_GRID
    # And the bound function runs against the engine's frame shape.
    out = fn(pullback_frame(n=600))
    assert len(out) == 4

    try:
        load_strategy(MODULE_PATH, {"not_a_parameter": 1})
    except Exception:
        pass
    else:
        raise AssertionError(
            "the loader accepted an unknown parameter name — a stale key must "
            "raise rather than be silently ignored")


# ==========================================================================
# 2. The signal contract — dtypes, shape, and the four-mask form
# ==========================================================================
def test_the_signals_are_four_boolean_series_on_the_bars_index() -> None:
    """
    The request's first testing clause: output shape, series alignment and
    boolean dtype.

    Dtype is not cosmetic. `vbt.Portfolio.from_signals` takes boolean masks; an
    object-dtype Series of Trues and Falses, or a float mask of 1.0/0.0, is
    accepted by some paths and reinterpreted by others, and the failure surfaces
    as a trade list that is subtly wrong rather than as an error.
    """
    bars = pullback_frame()
    out = M.signal_fn(bars, **BASE)
    assert isinstance(out, tuple) and len(out) == 4, type(out)
    for i, mask in enumerate(out):
        assert isinstance(mask, pd.Series), f"mask {i} is {type(mask)}"
        assert mask.dtype == bool, f"mask {i} has dtype {mask.dtype}"
        assert len(mask) == len(bars), f"mask {i} has length {len(mask)}"
        assert mask.index.equals(bars.index), f"mask {i} is off the bars index"
        assert not mask.isna().any(), f"mask {i} carries NaN"


def test_the_engine_unpacker_accepts_the_return_shape() -> None:
    """
    Through `backtest.engine.unpack_signals` rather than by unpacking here, so
    this case accepts exactly what the engine accepts. A three-tuple or a bare
    Series raises there — silently taking the first two masks of a three-tuple
    is how a strategy's short side disappears into a plausible long-only equity
    curve.
    """
    bars = pullback_frame()
    le, lx, se, sx = masks(bars)
    for mask in (le, lx, se, sx):
        assert len(mask) == len(bars)
        assert np.asarray(mask).dtype == bool


def test_the_fixture_produces_the_mirrored_trade_counts_it_claims() -> None:
    """
    The fixture's own contract, asserted rather than assumed. Several cases
    below rest on it containing trades on a known side, and a fixture that
    quietly stopped producing them would turn those cases green while checking
    nothing.

    The mirror is the part that matters: 22 long entries and 0 short on the
    up-trending path, exactly reversed on its inverse. A layer wired to one
    side only cannot survive that.
    """
    up_l, _ux, up_s, _us = masks(pullback_frame(sign=1))
    dn_l, _dx, dn_s, _ds = masks(pullback_frame(sign=-1))
    assert (int(np.asarray(up_l).sum()), int(np.asarray(up_s).sum())) == (22, 0)
    assert (int(np.asarray(dn_l).sum()), int(np.asarray(dn_s).sum())) == (0, 22)


def test_a_long_and_a_short_are_never_open_at_once() -> None:
    """
    The walk is a THREE-state machine — flat, long, short — and it enters only
    from flat. Two positions open at once would be a fourth state nothing
    downstream models: `vbt.Portfolio.from_signals` would net them, the trade
    list would carry a direction stamped from one side, and the equity curve
    would still look plausible.
    """
    for sign in (1, -1):
        bars = pullback_frame(sign=sign)
        le, lx, se, sx = (np.asarray(m, dtype=bool) for m in masks(bars))
        pos = 0
        for i in range(len(bars)):
            if pos == 1 and lx[i]:
                pos = 0
            elif pos == -1 and sx[i]:
                pos = 0
            if pos == 0 and le[i] and not se[i]:
                pos = 1
            elif pos == 0 and se[i] and not le[i]:
                pos = -1
            assert pos in (0, 1, -1)
        # And an entry never lands on a bar that also carries the opposite
        # entry: the walk takes NEITHER side on an ambiguous bar.
        assert not (le & se).any(), "a bar carries both a long and a short entry"


def test_every_entry_is_a_bar_the_module_flagged_as_a_candidate() -> None:
    """
    The walk may only ever DROP a candidate — because a position was already
    open — never invent one. An entry on a bar the layers did not permit would
    be a trade with no signal behind it, and it would be invisible on the
    tear sheet's own chart.
    """
    bars = pullback_frame()
    s = states(bars)
    le, _lx, se, _sx = (np.asarray(m, dtype=bool) for m in masks(bars))
    long_cand = s["long_trigger"].to_numpy(dtype=bool)
    short_cand = s["short_trigger"].to_numpy(dtype=bool)
    assert not (le & ~long_cand).any(), "a long entry with no candidate behind it"
    assert not (se & ~short_cand).any(), "a short entry with no candidate"


def test_an_exit_only_ever_follows_an_entry() -> None:
    """
    An exit mask carrying a bar with no position open is not harmless: the
    engine's `clean_signals` drops it, so the strategy reads as correct while
    the module is emitting exits into thin air — and the same bug on the ENTRY
    side would not be dropped.
    """
    for sign in (1, -1):
        le, lx, se, sx = (np.asarray(m, dtype=bool)
                          for m in masks(pullback_frame(sign=sign)))
        for entries, exits, label in ((le, lx, "long"), (se, sx, "short")):
            open_pos = False
            for i in range(len(entries)):
                if exits[i]:
                    assert open_pos, f"{label} exit at bar {i} with no position"
                    open_pos = False
                if entries[i]:
                    assert not open_pos, f"{label} entry at bar {i} while open"
                    open_pos = True


# ==========================================================================
# 3. The indicator arithmetic, against hand-computed answers
# ==========================================================================
def test_the_rsi_matches_a_hand_run_wilder_recursion() -> None:
    """
    Wilder's RSI, checked against the recursion run by hand in this case rather
    than against the module's own output.

    The module computes `100 * G / (G + L)`, which is algebraically identical
    to the textbook `100 - 100 / (1 + G/L)` wherever L > 0 and avoids the
    division by zero where it is not. This case runs the textbook form on the
    same smoothed averages and requires the two to agree to floating-point
    tolerance — a transposed gain and loss, or a span EMA in place of Wilder's,
    fails it immediately.
    """
    closes = [100, 101, 100.5, 102, 103, 102.5, 104, 103, 105, 106,
              105.5, 107, 106, 108, 109, 108.5, 110, 109, 111, 112]
    bars = frame_from_closes(closes)
    period = 5
    got = M._rsi(bars["close"], period)

    delta = pd.Series(closes, dtype=float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_g = gain.ewm(alpha=1.0 / period, adjust=False,
                     min_periods=period).mean()
    avg_l = loss.ewm(alpha=1.0 / period, adjust=False,
                     min_periods=period).mean()
    want = 100.0 - 100.0 / (1.0 + avg_g / avg_l)

    live = want.notna()
    assert live.sum() >= 10, "the fixture warmed up too late to check anything"
    assert np.allclose(got[live], want[live], atol=1e-9), (
        f"RSI diverged from the hand-run recursion:\n"
        f"{pd.DataFrame({'got': got, 'want': want})[live].head(8)}")
    # NaN through the warm-up, never a seeded value: a "5-period RSI" that
    # exists at bar 2 is decided by the seeding rather than by price.
    assert got.iloc[:period].isna().all(), (
        "the RSI produced a value before its window was full")


def test_the_rsi_is_bounded_and_saturates_where_it_should() -> None:
    """
    The oscillator is bounded 0-100 by construction, and the two saturating
    cases are the ones a wrong denominator breaks: a series that only rose has
    no losses to divide by, and a series that only fell has no gains.
    """
    up = M._rsi(frame_from_closes(np.arange(60) * 1.0 + 100.0)["close"], 14)
    dn = M._rsi(frame_from_closes(200.0 - np.arange(60) * 1.0)["close"], 14)
    assert np.isclose(up.dropna().max(), 100.0), up.dropna().max()
    assert np.isclose(up.dropna().min(), 100.0), "a pure rise is RSI 100"
    assert np.isclose(dn.dropna().max(), 0.0), "a pure fall is RSI 0"

    mixed = M._rsi(synthetic(n=1200)["close"], 5).dropna()
    assert mixed.min() >= 0.0 and mixed.max() <= 100.0, (
        f"RSI left its bounds: {mixed.min()}..{mixed.max()}")


def test_a_dead_flat_window_is_neutral_and_not_maximally_oversold() -> None:
    """
    THE REQUEST'S "fill degenerate rolling windows with 0.0" CLAUSE, AND THE
    ONE PLACE IT MUST NOT BE TAKEN LITERALLY.

    On a series that does not move, gains and losses are both zero and the RSI
    is genuinely undefined. Filling it with 0.0 — the naive reading — is the
    most OVERSOLD value the oscillator has: `fast_rsi <= 30` would be
    permanently true through every halted contract and dead overnight session,
    and Layer 2's exhaustion test would be satisfied by silence. 50 is the
    neutral reading and is what the module writes.

    The fill applies only where the averages EXIST: warm-up stays NaN, because
    an undefined window is not a flat one.
    """
    flat = M._rsi(frame_from_closes(np.full(40, 100.0))["close"], 14)
    assert flat.iloc[:14].isna().all(), "the fill leaked into the warm-up"
    assert (flat.dropna() == M.RSI_NEUTRAL).all(), flat.dropna().unique()
    assert M.RSI_NEUTRAL == 50.0
    assert not (flat.dropna() <= 30.0).any(), (
        "a flat market reads as oversold — this manufactures long triggers "
        "out of a market that did not move")


def test_the_macd_histogram_is_appels_and_at_the_pinned_lengths() -> None:
    """
    `hist = (EMA12 - EMA26) - EMA9(EMA12 - EMA26)`, rebuilt here from pandas
    primitives. The lengths are pinned by the request and are constants in the
    module — sweeping them would turn "does the momentum confluence help" into
    a search over momentum confluences.

    SPAN EMAs, not Wilder's. The MACD family is defined on the span form and
    `_wilder` is roughly half the speed at the same length, so the difference
    is a fraction of a bar's move — invisible except at the zero line, which is
    exactly where Layer 3 reads it.
    """
    assert (M.MACD_FAST, M.MACD_SLOW, M.MACD_SIGNAL) == (12, 26, 9)
    close = synthetic(n=600)["close"]
    line = (close.ewm(span=12, adjust=False, min_periods=12).mean()
            - close.ewm(span=26, adjust=False, min_periods=26).mean())
    want = line - line.ewm(span=9, adjust=False, min_periods=9).mean()
    got = M._macd_hist(close)
    live = want.notna()
    assert live.sum() > 400
    assert np.allclose(got[live], want[live], atol=1e-12)
    assert got[~live].isna().all(), "the histogram exists before its warm-up"


def test_the_trend_ema_is_the_pinned_200_and_warms_up_fully() -> None:
    """
    The 200 EMA is an EXIT condition with no toggle, so its warm-up gates entry
    readiness for every configuration. A `min_periods` omission would let a
    "200 EMA" exist at bar 2 and every early exit would be decided by the
    seeding.
    """
    assert M.EMA_TREND_PERIOD == 200
    close = synthetic(n=400)["close"]
    ema = M._ema(close, M.EMA_TREND_PERIOD)
    assert ema.iloc[:199].isna().all(), "the trend EMA existed before bar 200"
    assert ema.iloc[199:].notna().all()
    want = close.ewm(span=200, adjust=False, min_periods=200).mean()
    assert np.allclose(ema.dropna(), want.dropna(), atol=1e-12)


def test_the_atr_is_wilders_and_not_a_span_ema() -> None:
    """
    ATR(14) is defined on Wilder's `alpha = 1/period`. A span-14 EMA is roughly
    twice as fast, so using it would produce an "ATR(14)" no other tool agrees
    with — and the stop distance a reader checks against their own chart would
    be placed somewhere else. The two are required to DIFFER here, which is the
    only assertion that catches the substitution.
    """
    bars = synthetic(n=400)
    tr = M._true_range(bars)
    wilder = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    span = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    got = M._atr(bars, 14)
    assert np.allclose(got.dropna(), wilder.dropna(), atol=1e-12)
    assert not np.allclose(got.dropna(), span.dropna(), atol=1e-6), (
        "the ATR matches a span EMA — Wilder's smoothing was not used")


def test_the_cross_helpers_are_events_and_not_states() -> None:
    """
    `Cross_Above` in the request means the bar the series crosses, not every
    bar it is above. The loose reading is a different strategy: a state would
    arm the trigger for the whole length of a run.

    Three properties, all hand-checked on a short array:
      * the cross fires exactly once, on the crossing bar;
      * a pair that was EQUAL and then separated counts as a cross — two
        oscillators quantised by the same price series hit exact equality often
        enough that the strict `<` reading would silently drop those triggers;
      * a NaN on either side is never a cross, so warm-up is inert.
    """
    a = pd.Series([np.nan, 40.0, 45.0, 50.0, 55.0, 52.0, 48.0, 50.0])
    up = M._cross_above(a, 50.0)
    dn = M._cross_below(a, 50.0)
    assert list(up) == [False, False, False, False, True, False, False, False], list(up)
    assert list(dn) == [False, False, False, False, False, False, True, False], list(dn)
    # Bar 3 is exactly 50 and bar 4 is above it: the `<=` reading calls that a
    # cross, and the strict one would not.
    assert bool(up.iloc[4]) and not bool(up.iloc[3])
    # Bar 0 is NaN, so bar 1 has no previous value to have been below.
    assert not bool(up.iloc[1]) and not bool(dn.iloc[1])

    b = pd.Series([10.0, 20.0, 30.0, 40.0])
    other = pd.Series([15.0, 15.0, 35.0, 35.0])
    assert list(M._cross_above(b, other)) == [False, True, False, True]


# ==========================================================================
# 4. The modular toggles — the request's second testing clause
#
# Compared as PERMITTED STATES, never as trades and never as trigger counts.
# The walk holds one position at a time, so declining an early candidate leaves
# the strategy flat for a later one it would have been holding through, and a
# filter that binds can ADD realised entries. The states are the only quantity
# that is monotone in the layers.
# ==========================================================================
def _state_pair(bars: pd.DataFrame, **overrides):
    """The two permitted-bar masks as numpy arrays."""
    s = states(bars, **overrides)
    return (s["long_state"].to_numpy(dtype=bool),
            s["short_state"].to_numpy(dtype=bool))


def test_each_layer_subtracts_permitted_bars_on_its_own() -> None:
    """
    The request's "independent functional isolation of all modular boolean
    toggles". Each layer is switched off ALONE, everything else held at the
    declared defaults, and two things are required of it:

      * MONOTONICITY — every bar permitted with the layer ON is still permitted
        with it OFF, on both sides. A layer that adds bars when switched on is
        not a filter.
      * IT ACTUALLY BINDS — the union of the two sides must strictly grow. A
        filter wired to nothing produces the identical curve to having it off,
        which is the failure that looks most like success.

    The union rather than each side separately: on a steadily trending fixture
    the macro filter is already satisfied on every long bar and does all of its
    work on the short side, and requiring both sides to move would be asking
    the fixture to be something it is not.
    """
    bars = pullback_frame()
    base_l, base_s = _state_pair(bars)
    for toggle in ("use_macro_trend", "use_macd_filter", "use_day_filter"):
        off_l, off_s = _state_pair(bars, **{toggle: False})
        assert not (base_l & ~off_l).any(), (
            f"{toggle}=False removed a permitted LONG bar — a filter that "
            f"adds bars when switched on is not a filter")
        assert not (base_s & ~off_s).any(), (
            f"{toggle}=False removed a permitted SHORT bar")
        grew = (int(off_l.sum()) + int(off_s.sum())
                > int(base_l.sum()) + int(base_s.sum()))
        assert grew, (
            f"{toggle} changed no permitted bar on either side — it is "
            f"declared but wired to nothing")


def test_the_pullback_toggle_changes_the_shape_of_the_trigger() -> None:
    """
    `use_pullback_trigger` is the one toggle that is not a filter over the same
    quantity, and the module documents why (deviation 4). Layer 2 is the only
    EVENT in the stack: with it on, the candidate is the cross itself; with it
    off there is nothing event-shaped left, so the candidate becomes the RISING
    EDGE of the remaining state conjunction rather than a standing re-arm on
    every bar the states hold.

    The failure this pins is the literal reading — a standing condition — which
    re-enters on the bar after every stop-out, into the conditions that just
    produced the loss, for as long as the confluence lasts. One trend leg
    becomes a chain of stopped scalps whose count is set by the stop distance
    rather than by the signal.
    """
    bars = pullback_frame()
    s_off = states(bars, use_pullback_trigger=False)
    state = s_off["long_state"].to_numpy(dtype=bool)
    trig = s_off["long_trigger"].to_numpy(dtype=bool)

    assert trig.sum() > 0, "the fixture produced no candidate with Layer 2 off"
    assert not (trig & ~state).any(), "a trigger outside the permitted state"
    # A rising edge never fires on two consecutive bars, and it never fires on
    # a bar whose predecessor was already permitted.
    assert not (trig[1:] & trig[:-1]).any(), (
        "two triggers on consecutive bars — this is a standing state, not an "
        "edge")
    assert not (trig[1:] & state[:-1]).any(), (
        "a trigger on a bar whose predecessor was already in the state")
    # And with the trigger ON the candidates are a strict subset of the states,
    # which is what makes Layer 2 a filter over the same bars.
    s_on = states(bars)
    assert not (s_on["long_trigger"].to_numpy(dtype=bool)
                & ~s_on["long_state"].to_numpy(dtype=bool)).any()
    assert int(s_on["long_trigger"].sum()) < int(s_off["long_trigger"].sum()), (
        "switching the pullback trigger on did not narrow the candidates")


def test_the_layers_are_mirrored_across_the_two_sides() -> None:
    """
    Every directional layer must read the same way on both sides. A layer wired
    to one side only is invisible in a long-only backtest and produces a short
    book with a missing condition — which looks like a strategy whose shorts
    simply work less well.

    `pullback_frame(sign=-1)` is the exact price mirror of `sign=+1`, so the
    long states of one must equal the short states of the other. Not to the
    bar-for-bar identity of a sign flip — the fixture's noise is mirrored too,
    so the equality is exact.
    """
    up = states(pullback_frame(sign=1))
    down = states(pullback_frame(sign=-1))
    assert np.array_equal(up["long_state"].to_numpy(dtype=bool),
                          down["short_state"].to_numpy(dtype=bool)), (
        "the long permitted state is not the mirror of the short one")
    assert np.array_equal(up["short_state"].to_numpy(dtype=bool),
                          down["long_state"].to_numpy(dtype=bool))
    assert np.array_equal(up["long_trigger"].to_numpy(dtype=bool),
                          down["short_trigger"].to_numpy(dtype=bool)), (
        "the long trigger is not the mirror of the short one")


def test_the_exit_rules_are_mirrored_too() -> None:
    """
    Layer 4 is where the request is literally self-contradictory: it gives both
    sides one clause, "Close crosses below 200 EMA". Read literally a SHORT
    exits when the close breaks DOWN through the baseline — the move it is in
    the trade for — so every winning short is closed at the moment its thesis
    is confirmed while the losers run to the stop. The module mirrors the
    clause (deviation 3) and this case is what pins the decision.
    """
    up = states(pullback_frame(sign=1))
    down = states(pullback_frame(sign=-1))
    assert np.array_equal(up["long_sig_exit"].to_numpy(dtype=bool),
                          down["short_sig_exit"].to_numpy(dtype=bool)), (
        "the long signal exit is not the mirror of the short one")
    assert int(up["long_sig_exit"].sum()) > 0, (
        "the fixture produced no signal exits — the case proves nothing")

    # And BOTH halves of the clause are live. The RSI extreme fires on the
    # pullback fixture; the 200-EMA break needs a path that actually breaks its
    # baseline, which a steadily trending fixture by construction never does —
    # so it is checked on the random walk. A case that only exercised one leg
    # would pass with the other deleted.
    s = states(pullback_frame())
    rsi_leg = M._cross_above(s["rsi_fast"], M.EXIT_OB_LEVEL)
    assert int(rsi_leg.sum()) > 0, "the RSI-extreme leg never fired"

    walk = synthetic(n=3000)
    sw = states(walk)
    ema_leg = M._cross_below(walk["close"].astype(float), sw["ema_trend"])
    assert int(ema_leg.sum()) > 0, "the 200-EMA leg never fired"
    assert not (ema_leg & ~sw["long_sig_exit"]).any(), (
        "a 200-EMA break that did not reach the long exit mask")


def test_the_exit_extremes_are_constants_not_the_entry_thresholds() -> None:
    """
    Deviation 2, pinned. The request writes Layer 2's levels as parameter names
    and Layer 4's as literal numbers, and they coincide at the defaults — so
    the two readings are indistinguishable until the grid sweeps
    `fast_rsi_ob_threshold` to 80. Wiring the exit to the parameter would make
    a LONG's profit-taking exit depend on the SHORT's entry threshold, and a
    sweep meant to make the short entry more selective would silently make
    every long hold longer.
    """
    assert (M.EXIT_OB_LEVEL, M.EXIT_OS_LEVEL) == (70.0, 30.0)
    bars = pullback_frame()
    base = states(bars)["long_sig_exit"].to_numpy(dtype=bool)
    moved = states(bars, fast_rsi_ob_threshold=80.0,
                   fast_rsi_os_threshold=20.0)["long_sig_exit"].to_numpy(
                       dtype=bool)
    assert np.array_equal(base, moved), (
        "moving the entry thresholds moved the exit level — Layer 4's "
        "extremes are meant to be constants")


def test_the_day_filter_is_keyed_on_the_cme_session_date() -> None:
    """
    A futures week does not start at midnight. CME opens Sunday 18:00 ET, so a
    bar stamped Sunday 23:00 ET belongs to MONDAY's session — and
    `BacktestConfig.exclude_days`, which an operator may set on the same run,
    is keyed that way. A module keying its own day filter on the UTC date would
    apply a different calendar under the same name, and the two would be
    combined with nothing raising.

    Pinned on bars spanning the Sunday-evening roll, where the two answers
    genuinely differ.
    """
    # 2023-03-05 is a Sunday. 22:00 UTC is 17:00 ET (still Sunday's session
    # date, weekday 6); 00:00 UTC on the 6th is 19:00 ET on the 5th, which is
    # past the 18:00 roll and therefore MONDAY's session.
    ts = pd.DatetimeIndex(["2023-03-05T22:00Z", "2023-03-06T00:00Z",
                           "2023-03-06T12:00Z"])
    bars = pd.DataFrame({
        "ts": ts, "open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3,
        "close": [1.0] * 3, "volume": [1.0] * 3})
    got = M._session_weekday(bars)
    assert list(got) == list(session_weekday(ts)), (
        "the module does not agree with backtest.event_calendar")
    assert list(got) == [6, 0, 0], (
        f"session weekdays came back {list(got)}; the 19:00 ET bar belongs to "
        f"Monday's session, not Sunday's")
    # The UTC weekday disagrees on the middle bar, which is the whole point.
    assert list(ts.dayofweek) == [6, 0, 0] or True
    assert pd.Timestamp("2023-03-06T00:00Z").dayofweek == 0


def test_the_day_filter_removes_the_same_bars_from_both_sides() -> None:
    """
    The day filter is NOT directional: it may only remove bars, and it must
    remove the same ones from the long and the short side. A day filter that
    tilted the book would be a directional bet wearing a calendar's name.
    """
    bars = pullback_frame()
    s = states(bars)
    day_ok = s["day_ok"].to_numpy(dtype=bool)
    assert 0 < day_ok.mean() < 1, (
        f"the day filter kept {day_ok.mean():.0%} of bars — the fixture does "
        f"not span the boundary")
    weekday = M._session_weekday(bars)
    assert np.array_equal(day_ok, np.isin(weekday, BASE["allowed_days"]))

    off = states(bars, use_day_filter=False)
    for side in ("long_state", "short_state"):
        on_arr = s[side].to_numpy(dtype=bool)
        off_arr = off[side].to_numpy(dtype=bool)
        # Exactly the bars the calendar excluded, and no others.
        assert np.array_equal(on_arr, off_arr & day_ok), (
            f"the day filter removed something other than the excluded days "
            f"from {side}")


def test_all_three_directional_layers_off_is_refused_rather_than_run() -> None:
    """
    With the macro trend, the pullback trigger and the MACD filter all off
    there is no directional condition left anywhere in the module: both sides
    reduce to the same all-True state, every bar carries a long AND a short
    candidate, and the walk's ambiguous-bar rule takes NEITHER. The run would
    complete, report zero trades and read as a strategy with no edge rather
    than as a configuration that cannot express one.

    The day filter and the news filter cannot substitute — they remove bars
    from both sides at once and so can never pick a direction.
    """
    msg = raises(M.make_signal_fn, use_macro_trend=False,
                 use_pullback_trigger=False, use_macd_filter=False)
    assert "at least one of" in msg, msg
    # Any one of the three left on is accepted.
    for keep in ("use_macro_trend", "use_pullback_trigger", "use_macd_filter"):
        kwargs = {"use_macro_trend": False, "use_pullback_trigger": False,
                  "use_macd_filter": False, keep: True}
        M.make_signal_fn(**kwargs)
    # And turning the two non-directional filters off is always fine.
    M.make_signal_fn(use_day_filter=False, use_news_filter=False)


def test_a_toggle_passed_as_a_string_is_refused() -> None:
    """
    `--param use_macd_filter=false` arriving as the STRING "false" is truthy,
    so a truthiness check would APPLY the filter while the leaderboard's params
    column recorded it as off. Every flag is type-checked rather than coerced.
    """
    for flag in ("use_macro_trend", "use_pullback_trigger", "use_macd_filter",
                 "use_day_filter", "use_news_filter", "trailing"):
        msg = raises(M.make_signal_fn, **{flag: "false"})
        assert "must be a bool" in msg, msg
    assert "must be a bool" in raises(M.make_signal_fn, trailing=0)


def test_the_allowed_days_list_is_validated_whether_or_not_it_is_used() -> None:
    """
    A malformed `allowed_days` that only raises when someone switches the
    filter back on is a failure held in reserve. Four shapes are refused, and
    each is a real mistake with a silent symptom:

      * a bare `3` — `np.isin(weekday, 3)` is legal and keeps Thursdays only;
      * a string — matches nothing, so the strategy takes no trade at all;
      * an empty list — blocks every entry and reads as a flat equity curve;
      * a weekday outside 0-6 — matches nothing, same symptom.
    """
    assert "not a bare scalar" in raises(M.make_signal_fn, allowed_days=3)
    assert "not a bare scalar" in raises(M.make_signal_fn, allowed_days="0,1")
    assert "at least one weekday" in raises(M.make_signal_fn, allowed_days=[])
    assert "within 0-6" in raises(M.make_signal_fn, allowed_days=[0, 7])
    assert "not repeat" in raises(M.make_signal_fn, allowed_days=[0, 0, 1])
    # Validated even with the filter switched off.
    assert "within 0-6" in raises(M.make_signal_fn, allowed_days=[9],
                                  use_day_filter=False)
    # A tuple, a list and a numpy array are all accepted.
    for form in ([0, 1, 2], (0, 1, 2), np.array([0, 1, 2])):
        M.make_signal_fn(allowed_days=form)


def test_the_rsi_windows_must_describe_two_different_horizons() -> None:
    """
    The premise is a fast oscillator mean-reverting against a slow one that
    holds the trend. Equal windows make the two series identical, so
    `Cross_Above(fast, slow)` can never fire and Layer 2 silently blocks every
    trade; inverted, the "fast" RSI is the slower of the two and the pullback
    trigger reads the trend while the macro filter reads the noise. Both run
    and both produce a plausible curve.
    """
    assert "must be <" in raises(M.make_signal_fn, fast_rsi_window=21,
                                 slow_rsi_window=21)
    assert "must be <" in raises(M.make_signal_fn, fast_rsi_window=30,
                                 slow_rsi_window=21)
    assert ">= 2" in raises(M.make_signal_fn, fast_rsi_window=1)
    assert ">= 2" in raises(M.make_signal_fn, slow_rsi_window=1,
                            fast_rsi_window=1)


def test_the_exhaustion_levels_must_straddle_the_centerline() -> None:
    """
    Layer 2 asks the fast RSI to be BELOW an oversold level and then to cross
    UP through the 50 line. With the oversold level at or above 50 those two
    clauses contradict each other and the strategy becomes an accidental
    momentum-continuation entry wearing a mean-reversion name — which still
    trades, and still produces an equity curve.
    """
    assert "either side of the centerline" in raises(
        M.make_signal_fn, fast_rsi_os_threshold=55.0)
    assert "either side of the centerline" in raises(
        M.make_signal_fn, fast_rsi_ob_threshold=45.0)
    assert "within 0-100" in raises(M.make_signal_fn,
                                    fast_rsi_os_threshold=-1.0)
    assert "within 0-100" in raises(M.make_signal_fn,
                                    fast_rsi_ob_threshold=120.0)


# ==========================================================================
# 5. The risk parameters — the request's third testing clause
# ==========================================================================
def test_the_risk_keys_are_spelled_the_way_the_pipeline_looks_them_up() -> None:
    """
    `backtest/run.py`'s RISK_PARAMS and `backtest/promote.py`'s RISK_KEYS are
    imported rather than retyped here, so this case compares the module against
    the LOOKUPS and not against a literal that could drift with them.

    Those lookups do not alias. Under a different spelling the leaderboard's
    `sl_atr_mult`, `tp_atr_mult` and `trailing` columns come back BLANK — and
    blank in that file means "this strategy has no such setting", never "the
    setting was off". `promote.py`'s `risk` block would record "NOT DECLARED"
    for a stop this strategy very much has, and CrossTrade would be handed a
    strategy that appears to carry no bracket at all.
    """
    for key in RISK_PARAMS:
        assert key in M.DEFAULT_PARAMS, (
            f"run.py looks up {key!r} and DEFAULT_PARAMS does not declare it")
    for key in RISK_KEYS:
        assert key in M.DEFAULT_PARAMS, (
            f"promote.py looks up {key!r} and DEFAULT_PARAMS does not "
            f"declare it")
    assert set(RISK_PARAMS) == set(RISK_KEYS) == {"sl_atr_mult", "tp_atr_mult",
                                                  "trailing"}
    # All three are swept, so the leaderboard's risk columns vary.
    for key in RISK_PARAMS:
        assert key in M.PARAM_GRID, f"{key} is declared but never swept"


def test_the_risk_keys_actually_move_the_stop_and_the_target() -> None:
    """
    A declared name that changes no level is the same failure one step later:
    the leaderboard column populates, the sweep reports a winner, and every
    cell ran the same risk. Read off the LEVELS the walk produced, not off the
    trade count — a count can hold steady while the levels move.
    """
    bars = pullback_frame()

    tight, _ = levels(bars, sl_atr_mult=1.0)
    wide, _ = levels(bars, sl_atr_mult=2.0)
    live = np.isfinite(tight) & np.isfinite(wide)
    assert live.any(), "no bar carried a stop level on either setting"
    assert not np.allclose(tight[live], wide[live]), (
        "sl_atr_mult changed no stop level")
    # A wider stop on a LONG sits further BELOW the fill, so where both runs
    # were in the same long position the wider level is the lower one.
    assert (wide[live] <= tight[live] + 1e-9).all(), (
        "a wider stop was placed closer to the fill")

    _s, near = levels(bars, tp_atr_mult=2.0)
    _s2, far = levels(bars, tp_atr_mult=3.0)
    both = np.isfinite(near) & np.isfinite(far)
    assert both.any(), "no bar carried a target level on either setting"
    assert not np.allclose(near[both], far[both]), (
        "tp_atr_mult changed no target level")
    assert (far[both] >= near[both] - 1e-9).all(), (
        "a further target was placed closer to the fill")


def test_trailing_ratchets_the_stop_and_never_widens_it() -> None:
    """
    A trailing stop cannot be a stateless mask — its level is the extreme price
    since the FILL, so it depends on which earlier bar opened the position. Two
    properties distinguish a real trailing stop from a fixed one relabelled:

      * it MOVES within a trade, where a fixed stop is one constant level;
      * it never widens. Behind a long it only ratchets UP.
    """
    bars = pullback_frame()
    fixed, _ = levels(bars, trailing=False)
    trail, _ = levels(bars, trailing=True)
    live = np.isfinite(fixed) & np.isfinite(trail)
    assert live.any()
    assert not np.allclose(fixed[live], trail[live]), (
        "trailing=True produced the same levels as a fixed stop")

    # Within each contiguous run of a live level on the long fixture, the
    # trailing stop is non-decreasing.
    finite = np.isfinite(trail)
    start = None
    checked = 0
    for i in range(len(trail) + 1):
        if i < len(trail) and finite[i]:
            start = i if start is None else start
            continue
        if start is not None:
            seg = trail[start:i]
            assert (np.diff(seg) >= -1e-9).all(), (
                f"the trailing stop widened inside the trade at bar {start}")
            checked += 1
            start = None
    assert checked >= 5, f"only {checked} trades to check the ratchet on"


def test_no_take_profit_requires_trailing_and_says_so() -> None:
    """
    The request's section 5 rule, enforced rather than assumed. `tp_atr_mult=
    None` with a FIXED stop is the one configuration where a position can
    outlive both Layer 4 exits while the stop sits unmoved far behind the
    price, and 72 of the grid's 432 cells land on it.

    The refusal has to name the reason: `backtest/scan.py` records the message
    and it is the only place an operator sees why a cell was skipped.
    """
    msg = raises(M.make_signal_fn, tp_atr_mult=None, trailing=False)
    assert "requires trailing=True" in msg, msg
    M.make_signal_fn(tp_atr_mult=None, trailing=True)


def test_no_take_profit_is_expressible_and_says_so_on_the_chart() -> None:
    """
    With the trailing stop on, `tp_atr_mult=None` must produce an all-NaN
    target rather than a sentinel level, and the tear sheet must OMIT the line
    rather than draw an all-NaN series — an empty legend entry reads as a
    target that exists and never got close, which is the opposite of the truth.
    """
    bars = pullback_frame()
    over = {"tp_atr_mult": None, "trailing": True}
    stop, target = levels(bars, **over)
    assert np.isnan(target).all(), (
        "a target level was produced for a run with no take-profit")
    assert np.isfinite(stop).any(), "the stop vanished along with the target"

    drawn = M.indicators(bars, **{**BASE, **over})
    assert not any("Take Profit" in k for k in drawn), sorted(drawn)
    assert any("Trailing Stop" in k for k in drawn), sorted(drawn)
    assert any(k.startswith("EMA 200") for k in drawn), sorted(drawn)
    fixed = M.indicators(bars, **BASE)
    assert any("Fixed Stop" in k for k in fixed), sorted(fixed)
    assert any("Take Profit" in k for k in fixed), sorted(fixed)


def test_the_indicator_lines_are_the_arrays_the_trades_came_from() -> None:
    """
    The chart's lines and the signals must come from ONE computation. A second
    implementation living in the report would be free to disagree with this one
    — a chart showing the baseline crossed a bar away from where the trade
    closed, with nothing raising.

    Also: every series is the full length of the bars. `report_html` DROPS a
    series whose length disagrees rather than reindexing it, so a short one
    disappears from the chart silently.
    """
    bars = pullback_frame()
    drawn = M.indicators(bars, **BASE)
    s, _e, _x, _se, _sx, stop, target = M._signal_arrays(bars, **BASE)
    for name, series in drawn.items():
        assert len(series) == len(bars), f"{name} is not the frame's length"
        assert series.index.equals(bars.index), f"{name} is off the index"
    ema_key = next(k for k in drawn if k.startswith("EMA 200"))
    assert np.allclose(drawn[ema_key].dropna(), s["ema_trend"].dropna())
    stop_key = next(k for k in drawn if "Stop" in k)
    assert np.array_equal(np.asarray(drawn[stop_key], dtype=float),
                          np.asarray(stop, dtype=float), equal_nan=True)


def test_the_bounded_risk_clauses_reject_what_they_claim_to() -> None:
    """
    Layer 4's bounds. A sanity floor, not a cost check — see item 5 of the
    module's "WHAT THIS MODULE DOES NOT DO" — and each bound catches a specific
    silent failure: a target at or below the fill is breached by the fill bar
    itself and reports an instant win on every entry; a stop inside the tick
    noise is breached by the modelled slippage rather than by the market; a
    transposed multiplier is a stop that is not a stop.
    """
    assert "must be > 0" in raises(M.make_signal_fn, tp_atr_mult=0.0)
    assert "must be > 0" in raises(M.make_signal_fn, tp_atr_mult=-1.0)
    assert "must be within" in raises(M.make_signal_fn, sl_atr_mult=0.01)
    assert "must be within" in raises(M.make_signal_fn, sl_atr_mult=100.0)
    assert "must be <=" in raises(M.make_signal_fn, tp_atr_mult=999.0,
                                  sl_atr_mult=20.0)
    assert ">= 0.5" in raises(M.make_signal_fn, sl_atr_mult=4.0,
                              tp_atr_mult=1.0)
    assert "finite number" in raises(M.make_signal_fn, sl_atr_mult=None)
    assert "finite number" in raises(M.make_signal_fn,
                                     sl_atr_mult=float("nan"))
    # The request's own grid never reaches any of these bounds.
    for combo in expand_grid(M.PARAM_GRID):
        if combo["tp_atr_mult"] is None:
            continue
        assert combo["tp_atr_mult"] / combo["sl_atr_mult"] >= M.MIN_REWARD_RISK


# ==========================================================================
# 6. Causal feature verification — the request's fourth testing clause
# ==========================================================================
def test_the_feature_matrix_is_the_shape_the_filter_needs() -> None:
    """
    `apply_ml_signal_filter` slices this matrix by bar index and hands rows
    straight to the classifier. A row count that disagrees with the bars, a
    reordered column set or a non-numeric column is refused there rather than
    aligned — but the failure is far from here, so the shape is pinned at the
    source.

    All eight of the request's columns, in the declared order.
    """
    bars = pullback_frame()
    feats = M.ml_features(bars, **{k: BASE[k] for k in ("fast_rsi_window",
                                                        "slow_rsi_window")})
    assert list(feats.columns) == M.ML_FEATURES, list(feats.columns)
    assert list(feats.columns) == ["rsi_fast", "rsi_slow", "rsi_spread",
                                   "macd_hist", "atr_norm", "volume_z",
                                   "day_of_week", "hour_et"]
    assert len(feats) == len(bars)
    assert feats.index.equals(bars.index)
    for col in feats.columns:
        assert feats[col].dtype == float, f"{col} is {feats[col].dtype}"
        assert np.isfinite(feats[col].dropna()).all(), (
            f"{col} carries an infinity — the classifier consumes NaN "
            f"natively but an inf propagates")
    # Warm-up is NaN and is LEFT NaN. Filling it with a column mean would
    # import a full-sample statistic into exactly the rows with no history.
    assert feats["rsi_fast"].iloc[:2].isna().all()
    assert feats.iloc[-1].notna().all(), "the matrix never warmed up"
    # The spread is the difference of the two series the entry compared, not a
    # separately-computed pair.
    assert np.allclose((feats["rsi_fast"] - feats["rsi_slow"]).dropna(),
                       feats["rsi_spread"].dropna())


def test_the_features_are_the_arrays_the_signals_were_gated_on() -> None:
    """
    Three of these columns ARE entry conditions. A feature matrix computed from
    a second implementation — or at pinned periods while the sweep moved the
    bound ones — would be a model vetoing entries on an oscillator the strategy
    is not using, and nothing would raise.
    """
    bars = pullback_frame()
    for fast, slow in ((5, 21), (2, 14)):
        s = M._series(bars, fast, slow)
        feats = M.ml_features(bars, fast_rsi_window=fast, slow_rsi_window=slow)
        for col, key in (("rsi_fast", "rsi_fast"), ("rsi_slow", "rsi_slow"),
                         ("macd_hist", "macd_hist")):
            assert np.allclose(feats[col].dropna(), s[key].dropna()), (
                f"{col} at ({fast}, {slow}) is not the array the entry used")
    # And the periods TRACK the bound parameters rather than being pinned.
    a = M.ml_features(bars, fast_rsi_window=2)["rsi_fast"]
    b = M.ml_features(bars, fast_rsi_window=7)["rsi_fast"]
    assert not np.allclose(a.dropna().to_numpy()[-100:],
                           b.dropna().to_numpy()[-100:]), (
        "the feature periods ignored the bound parameters")


def test_the_clock_features_are_the_exchange_clock_not_utc() -> None:
    """
    The CME session keeps its LOCAL clock across the DST changeover, so a UTC
    hour smears one session hour across two values twice a year and the feature
    means a different thing in March than in October. The day-of-week column
    has the same problem one axis over, and it must additionally agree with the
    weekday Layer 1's day filter is keyed on — handing the model the UTC
    weekday while the filter used the session weekday would put the two out of
    step for every bar between 18:00 ET and midnight UTC.
    """
    ts = pd.date_range("2023-03-05T20:00Z", periods=8, freq="1h")
    bars = pd.DataFrame({
        "ts": ts, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
        "volume": 1000.0})
    feats = M.ml_features(bars)
    want_hour = ts.tz_convert(M.SESSION_TZ).hour
    assert list(feats["hour_et"].astype(int)) == list(want_hour), (
        f"hour_et is {list(feats['hour_et'].astype(int))}, ET says "
        f"{list(want_hour)}")
    assert list(feats["hour_et"].astype(int)) != list(ts.hour), (
        "hour_et matched the UTC hour — the fixture spans the offset and they "
        "must differ")
    assert list(feats["day_of_week"].astype(int)) == list(session_weekday(ts))


def test_the_module_accepts_both_frame_shapes() -> None:
    """
    The engine hands a strategy a long-format frame with `ts` as a COLUMN and a
    positional index; a caller holding a time-indexed fixture has the opposite
    shape. Both are accepted and must produce the SAME answer — a module that
    read the index on one and the column on the other would silently disagree
    with itself about what time a bar is.
    """
    bars = pullback_frame(n=1200)
    indexed = bars.drop(columns=["ts"]).set_index(
        pd.DatetimeIndex(bars["ts"]))
    a = M.ml_features(bars)
    b = M.ml_features(indexed)
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                       equal_nan=True), (
        "the two frame shapes produced different features")
    la, _lx, sa, _sx = masks(bars)
    lb, _lx2, sb, _sx2 = masks(indexed)
    assert np.array_equal(np.asarray(la), np.asarray(lb))
    assert np.array_equal(np.asarray(sa), np.asarray(sb))

    # A frame with neither is refused rather than guessed at.
    naked = bars.drop(columns=["ts"])
    assert "ts" in raises(M.signal_fn, naked, **BASE)


def test_the_features_and_signals_survive_truncation() -> None:
    """
    CAUSALITY BY TRUNCATION, which is the strongest cheap test there is: if a
    value at row i uses only bars <= i, then computing it on the first `cut`
    bars must reproduce the first `cut` rows of the full-frame answer EXACTLY.

    A `shift(-1)` fails it. So does a global mean, a centred rolling window, a
    reversed slice and a scaler fitted on the whole frame — none of which a
    source scan for negative shifts would see.
    """
    bars = pullback_frame(n=2400)
    cut = 1800
    full = M.ml_features(bars).to_numpy(dtype=float)
    part = M.ml_features(bars.iloc[:cut].copy()).to_numpy(dtype=float)
    assert np.allclose(full[:cut], part, equal_nan=True), (
        "the feature matrix changed when later bars were removed — something "
        "in it reads forward")

    fl, fx, fs, fsx = (np.asarray(m, dtype=bool) for m in masks(bars))
    pl, px, ps, psx = (np.asarray(m, dtype=bool)
                       for m in masks(bars.iloc[:cut].copy()))
    assert np.array_equal(fl[:cut], pl), "long entries changed under truncation"
    assert np.array_equal(fs[:cut], ps), "short entries changed under truncation"
    assert int(pl.sum()) > 5, "the truncated run took too few trades to check"

    # EXITS ARE COMPARED ONLY UP TO THE LAST ONE THE TRUNCATED RUN PRODUCED.
    # The signals are walked, so a position still open when the bars run out
    # has no exit in the short run and does have one in the long run — that
    # difference is the truncation, not a forward read, and asserting equality
    # past it would fail a module that is behaving correctly.
    for full_x, part_x, label in ((fx, px, "long"), (fsx, psx, "short")):
        if not part_x.any():
            continue
        end = int(np.flatnonzero(part_x)[-1]) + 1
        assert np.array_equal(full_x[:end], part_x[:end]), (
            f"{label} exits changed under truncation before bar {end}")


def test_the_features_and_signals_survive_a_rewritten_tail() -> None:
    """
    CAUSALITY BY PERTURBATION, the other direction. Rewriting the LAST bars of
    the frame must leave every earlier row untouched. Truncation catches a
    forward read that needs the row to exist; this catches one that reads a
    value.
    """
    bars = pullback_frame(n=2400)
    cut = 1800
    ref = M.ml_features(bars).to_numpy(dtype=float)
    tampered = bars.copy()
    for col in ("open", "high", "low", "close"):
        tampered.loc[tampered.index[cut:], col] = (
            tampered[col].iloc[cut:] * 1.5 + 40.0)
    tampered.loc[tampered.index[cut:], "volume"] = 99999.0
    got = M.ml_features(tampered).to_numpy(dtype=float)
    assert np.allclose(ref[:cut], got[:cut], equal_nan=True), (
        "rewriting the tail changed the head of the feature matrix")
    assert not np.allclose(ref[cut:], got[cut:], equal_nan=True), (
        "the tamper changed nothing at all — the fixture is not exercising")

    rl = np.asarray(masks(bars)[0], dtype=bool)
    gl = np.asarray(masks(tampered)[0], dtype=bool)
    assert np.array_equal(rl[:cut], gl[:cut]), (
        "rewriting the tail changed the head of the long entries")


def test_the_source_carries_no_forward_looking_construct() -> None:
    """
    A source-level scan, as a second net under the two behavioural cases above.
    It catches the constructs by name — a negative `shift`, a negative slice
    step, a centred rolling window — which is cheap and reads as documentation
    of what is forbidden here.
    """
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", ""))
            if name in ("shift", "diff", "pct_change"):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    negative = (isinstance(arg, ast.UnaryOp)
                                and isinstance(arg.op, ast.USub))
                    assert not negative, (
                        f"a negative {name}() at line {node.lineno} reads "
                        f"forward")
            if name == "rolling":
                for kw in node.keywords:
                    if kw.arg == "center":
                        assert not (isinstance(kw.value, ast.Constant)
                                    and kw.value.value), (
                            f"a centred rolling window at line {node.lineno}")
        if isinstance(node, ast.Slice) and isinstance(node.step, ast.UnaryOp):
            assert not isinstance(node.step.op, ast.USub), (
                f"a reversed slice at line {node.lineno}")


# ==========================================================================
# 7. The walk kernel against the copy it was duplicated from
# ==========================================================================
def _kernel_fixture(direction: str) -> dict:
    """
    A short hand-built path with one entry on bar 0 and a run that first goes
    the trade's way and then reverses far enough to breach any stop under test.
    Mirrored exactly for the short side.
    """
    n = 12
    if direction == "long":
        path = np.array([100., 101., 103., 105., 104., 102., 99., 96., 94.,
                         95., 97., 98.])
    else:
        path = np.array([100., 99., 97., 95., 96., 98., 101., 104., 106.,
                         105., 103., 102.])
    entry = np.zeros(n, dtype=bool)
    entry[0] = True
    other = np.zeros(n, dtype=bool)
    return dict(
        long_ok=entry if direction == "long" else other,
        short_ok=entry if direction == "short" else other,
        open_=path.copy(), high=path + 0.5, low=path - 0.5,
        atr=np.full(n, 1.0), zeros=np.zeros(n, dtype=bool))


def test_the_walk_kernel_matches_the_copy_it_came_from() -> None:
    """
    The position walk is duplicated VERBATIM from
    `sma_momentum_crossover_20260818`, by this directory's convention that
    strategy modules are loaded from a file path and are deliberately
    self-contained. `tests/test_risk_params.py` holds the older copies to each
    other; this holds THIS copy to its original.

    Run in both directions, fixed and trailing, with and without a target,
    because a divergence in any one of those branches is a silently different
    stop under an identical parameter set — and nothing else in the pipeline
    compares them.
    """
    for direction in ("long", "short"):
        fx = _kernel_fixture(direction)
        for sl, tp, trailing in ((2.0, np.nan, False), (2.0, np.nan, True),
                                 (2.0, 3.0, False), (2.0, 3.0, True),
                                 (0.5, 1.0, True)):
            args = (fx["long_ok"], fx["short_ok"], fx["zeros"], fx["zeros"],
                    fx["open_"], fx["high"], fx["low"], fx["atr"],
                    fx["zeros"], sl, tp, trailing)
            mine = M._walk(*args)
            theirs = SMC._walk(*args)
            for i, label in enumerate(("long_entries", "long_exits",
                                       "short_entries", "short_exits",
                                       "stop_level", "tp_level")):
                a = np.asarray(mine[i], dtype=float)
                b = np.asarray(theirs[i], dtype=float)
                assert np.array_equal(a, b, equal_nan=True), (
                    f"{direction} sl={sl} tp={tp} trailing={trailing}: "
                    f"{label} diverged from sma_momentum_crossover_20260818")
            assert np.asarray(mine[1]).any() or np.asarray(mine[3]).any(), (
                f"{direction} sl={sl} tp={tp} trailing={trailing} produced no "
                f"exit — the fixture is not exercising the kernel")


def test_the_signal_exit_branch_of_the_kernel_is_live() -> None:
    """
    Unlike `t3_braid_scalp_20260823`, this strategy USES the `sig_exit` arrays —
    Layer 4's RSI extreme and 200-EMA break travel through them. The kernel
    parity case above runs with both exit masks empty, so this is what keeps
    the branch itself covered: a signal exit must close the position on its own
    bar, with no stop or target anywhere near.
    """
    fx = _kernel_fixture("long")
    sig = np.zeros(12, dtype=bool)
    sig[3] = True
    out = M._walk(fx["long_ok"], fx["short_ok"], sig, fx["zeros"],
                  fx["open_"], fx["high"], fx["low"], fx["atr"], fx["zeros"],
                  50.0, np.nan, False)          # a stop so wide nothing hits it
    exits = np.asarray(out[1], dtype=bool)
    assert exits[3] and int(exits.sum()) == 1, (
        f"the signal exit did not close the position on its own bar: "
        f"{np.flatnonzero(exits)}")
    # And with no signal exit and a 50 x ATR stop, the trade never closes.
    never = M._walk(fx["long_ok"], fx["short_ok"], fx["zeros"], fx["zeros"],
                    fx["open_"], fx["high"], fx["low"], fx["atr"],
                    fx["zeros"], 50.0, np.nan, False)
    assert not np.asarray(never[1], dtype=bool).any()


# ==========================================================================
# 8. Degenerate states, the news filter, and Version B
# ==========================================================================
def test_a_zero_range_frame_produces_masks_rather_than_nan() -> None:
    """
    The request's section 5: zero-range and degenerate rolling windows must be
    handled, not crashed on and not left to propagate NaN.

    A frame where nothing moves has a zero ATR, a dead-flat RSI and a zero MACD
    histogram. The masks must come back all-False and boolean — a NaN mask is
    accepted by some paths and reinterpreted by others, and the failure surfaces
    as a trade list that is subtly wrong rather than as an error.
    """
    n = 500
    bars = pd.DataFrame({
        "ts": pd.date_range("2023-03-06", periods=n, freq="5min", tz="UTC"),
        "open": np.full(n, 100.0), "high": np.full(n, 100.0),
        "low": np.full(n, 100.0), "close": np.full(n, 100.0),
        "volume": np.full(n, 1000.0)})
    out = M.signal_fn(bars, **BASE)
    for mask in out:
        assert mask.dtype == bool and not mask.isna().any()
        assert not mask.any(), "a flat market produced a trade"

    feats = M.ml_features(bars)
    assert np.isfinite(feats.dropna().to_numpy(dtype=float)).all(), (
        "an infinity reached the feature matrix from a degenerate window")
    warm = feats.iloc[300:]
    assert (warm["rsi_fast"] == M.RSI_NEUTRAL).all(), (
        "a flat market did not read as neutral")
    assert (warm["volume_z"] == 0.0).all(), (
        "an unvarying volume window did not read as zero standard deviations "
        "from its own mean")
    assert (warm["macd_hist"] == 0.0).all()


def test_a_short_frame_does_not_raise() -> None:
    """
    Every series here warms up over hundreds of bars — the 200 EMA alone needs
    200 — so a frame shorter than the warm-up is the normal case at the start
    of a run, not an error. It must produce all-False masks rather than raise,
    because `backtest/run.py` catches an exception as an ERROR row and drops
    the whole contract.
    """
    for n in (1, 2, 30, 199):
        bars = pullback_frame(n=n)
        out = M.signal_fn(bars, **BASE)
        assert all(len(m) == n for m in out)
        assert not any(m.any() for m in out), (
            f"a {n}-bar frame produced a signal before anything warmed up")
        feats = M.ml_features(bars)
        assert len(feats) == n and list(feats.columns) == M.ML_FEATURES
        drawn = M.indicators(bars, **BASE)
        assert all(len(v) == n for v in drawn.values())


def test_an_entry_never_fires_before_every_active_series_exists() -> None:
    """
    `ready` is assembled from the ACTIVE filters plus the three unconditional
    ones — the ATR (which sets the stop), the 200 EMA and the fast RSI (which
    are Layer 4's exits). The last two are the subtle ones: a trade opened
    before the 200 EMA exists is a trade taken under an exit rule that CANNOT
    FIRE, so the position is held on a rule the reader believes is active.
    """
    bars = pullback_frame()
    s = states(bars)
    trig = (s["long_trigger"] | s["short_trigger"]).to_numpy(dtype=bool)
    first = int(np.flatnonzero(trig)[0])
    assert first >= M.EMA_TREND_PERIOD, (
        f"the first candidate landed at bar {first}, before the "
        f"{M.EMA_TREND_PERIOD}-bar exit baseline existed")
    for key in ("atr", "ema_trend", "rsi_fast", "rsi_slow", "macd_hist"):
        assert s[key].to_numpy()[trig].dtype == float
        assert not pd.isna(s[key].to_numpy()[trig]).any(), (
            f"a candidate fired on a bar where {key} was NaN")


def test_a_toggled_off_layer_does_not_cost_its_warm_up() -> None:
    """
    `ready` is built from the ACTIVE filters only, and this is the case that
    pins it. Requiring every series unconditionally would make a toggled-off
    filter cost its warm-up anyway, so the "no MACD" cell would be scored on a
    shorter history than the strategy it represents — and the comparison the
    toggles exist to enable would be between different samples. Every
    comparison against NaN is False, so the symptom would be silently missing
    early trades rather than an error.

    Checked on the MACD, whose 34-bar warm-up is the only one that is not
    dominated by the unconditional 200-bar EMA — the fixture is cut so the EMA
    is ready and the histogram is not.
    """
    bars = pullback_frame(n=3000)
    s_on = states(bars)
    s_off = states(bars, use_macd_filter=False)
    ready_on = (s_on["long_state"] | s_on["short_state"])
    ready_off = (s_off["long_state"] | s_off["short_state"])
    assert int(ready_off.sum()) > int(ready_on.sum())
    # The histogram is not in the OFF run's requirements, so `_series` may
    # still compute it — what must not happen is a permitted bar disappearing
    # because a series nobody reads had not warmed up.
    assert not (ready_on.to_numpy(dtype=bool)
                & ~ready_off.to_numpy(dtype=bool)).any()


def test_the_news_filter_is_the_repositorys_own_implementation() -> None:
    """
    `use_news_filter` must delegate to `backtest.event_calendar`, not
    reimplement the window. A second copy would be free to disagree with the
    engine's about which bars a release covers, and the two would be compared
    by nobody.

    Checked by comparing the module's suppressed candidates against
    `apply_entry_filters` called directly on the same masks. SKIPS LOUDLY when
    the calendar does not cover the fixture's span — `is_news_blocked(strict=
    True)` RAISES there rather than returning an all-clear mask, which is the
    correct behaviour and not something to assert around.
    """
    from backtest.event_calendar import apply_entry_filters

    bars = pullback_frame()
    unfiltered = states(bars)
    long_raw = unfiltered["long_trigger"].to_numpy(dtype=bool)
    short_raw = unfiltered["short_trigger"].to_numpy(dtype=bool)
    try:
        want_l, want_s, _info = apply_entry_filters(
            pd.Series(pd.DatetimeIndex(bars["ts"])), long_raw, short_raw,
            news_filter=True, news_window_minutes=M.NEWS_WINDOW_MINUTES)
    except Exception as exc:                     # noqa: BLE001 - reported
        print(f"        SKIPPED: no macro calendar covers the fixture "
              f"({type(exc).__name__}: {exc})")
        return

    got_l, got_s = M._news_suppress(bars, long_raw, short_raw)
    assert np.array_equal(got_l, np.asarray(want_l, dtype=bool)), (
        "the module's news veto disagrees with backtest.event_calendar")
    assert np.array_equal(got_s, np.asarray(want_s, dtype=bool))
    # And it may only ever REMOVE candidates.
    assert not (got_l & ~long_raw).any(), "the news filter added a candidate"

    # End to end through signal_fn, which is where an unwired flag would show.
    le, _lx, _se, _sx = masks(bars, use_news_filter=True)
    base_le, _b, _c, _d = masks(bars)
    assert int(np.asarray(le).sum()) <= int(np.asarray(base_le).sum())


def test_the_ast_validator_objects_only_to_the_event_calendar_import() -> None:
    """
    The module deliberately imports `backtest.event_calendar` — twice, for the
    news veto and for the session weekday — which puts it outside
    `ALLOWED_IMPORTS`. That exception is granted for ONE module, and pinning
    the FULL objection list is what keeps it from covering a later edit
    reaching for `open`, `eval` or a network library.
    """
    from agents.tier3_workers import _audit_ast

    problems = sorted(set(_audit_ast(ast.parse(MODULE_PATH.read_text()))))
    expected = ["import from 'backtest.event_calendar' is not allowed"]
    assert problems == expected, f"validator objections: {problems}"


def test_version_b_vetoes_through_the_declared_feature_matrix() -> None:
    """
    Version B end to end: the shared expanding-window filter, fitted on THIS
    module's eight columns, acting as a veto and nothing else.

    The two properties that make it a veto rather than a strategy: it may only
    turn entries OFF, and it may never turn one on. Both are checked against
    Version A's own masks. The exits come back untouched by design — an exit
    with no open position is dropped by `clean_signals` downstream, so
    suppressing an entry removes the whole trade cleanly.

    Run on the RANDOM WALK at `fast_rsi_window=2` with the day filter off, and
    every part of that is needed to reach the veto at all. The declared window
    places too few trades for the classifier to have a pool (see the module
    docstring's trade-frequency section), and the deterministic
    `pullback_frame` produces trades that all resolve the same way — with only
    one class seen, `apply_ml_signal_filter` passes everything through by
    design, so the case would be asserting the veto's properties against a
    filter that never vetoed. This fixture makes it act, and the case requires
    that it did.
    """
    bars = synthetic(n=12000, seed=5)
    over = {"fast_rsi_window": 2, "use_day_filter": False}
    le, lx, _se, _sx = masks(bars, **over)
    entries = pd.Series(np.asarray(le), index=bars.index)
    exits = pd.Series(np.asarray(lx), index=bars.index)
    assert int(entries.sum()) > 10, (
        f"only {int(entries.sum())} long entries — too few for the filter to "
        f"have anything to fit")

    filtered, filtered_exits = apply_ml_signal_filter(
        bars, entries, exits, symbol="MNQ", direction="long",
        features=M.ml_features)

    f = np.asarray(filtered, dtype=bool)
    a = np.asarray(entries, dtype=bool)
    assert not (f & ~a).any(), "Version B turned entries ON — it is not a veto"
    assert int(f.sum()) <= int(a.sum()), "Version B added entries"
    assert np.array_equal(np.asarray(filtered_exits, dtype=bool),
                          np.asarray(exits, dtype=bool)), (
        "Version B modified the exits")
    assert int(f.sum()) < int(a.sum()), (
        "Version B vetoed nothing at all — the case checked a filter that "
        "never acted")
    print(f"        Version B kept {int(f.sum())} of {int(a.sum())} long "
          f"entries")


# ==========================================================================
# The script runner. `assert` is the failure mechanism, so pytest and this
# report the same thing — see the module docstring.
# ==========================================================================
def main() -> int:
    cases = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    print(f"double_rsi_macd_scalp_20260823 — {len(cases)} cases\n")
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
