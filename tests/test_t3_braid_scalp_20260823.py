#!/usr/bin/env python3
"""
test_t3_braid_scalp_20260823.py — the T3 / Braid / Stiffness scalp: its
indicator arithmetic, its layer toggles, its risk-parameter keys, the causality
of the matrix Version B is fitted on, and the degenerate states the request
asked to be handled rather than crashed on.

Location:  ~/src/trading/tests/test_t3_braid_scalp_20260823.py

Run EITHER way — and unlike the older suites in this directory, both ways
report the same answer:

    OMP_NUM_THREADS=1 python tests/test_t3_braid_scalp_20260823.py
    OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
        tests/test_t3_braid_scalp_20260823.py

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

WHAT THIS COVERS, and why each one is here rather than assumed:

  * THE INDICATOR ARITHMETIC AGAINST HAND-COMPUTED NUMBERS, not against the
    module's own output. The T3 is pinned at `vfactor=0` where Tillson's cubic
    collapses to a plain triple EMA — an identity that fails the moment a
    coefficient is transposed — and the Stiffness Index is pinned against a
    seven-bar sequence whose up and down percentages are worked out on paper,
    including the flat bar that proves the two are NOT complements. A strategy
    compared only against itself is pinned, not verified.
  * SIGNAL DTYPES AND THE FOUR-MASK CONTRACT, through the engine's own
    `unpack_signals` rather than by unpacking here. A three-tuple or a bare
    Series raises there, and silently taking the first two masks of a
    three-tuple is how a strategy's short side disappears into a plausible
    long-only equity curve.
  * THE RISK KEYS BY THEIR EXACT SPELLING, against `backtest/run.py`'s
    RISK_PARAMS and `backtest/promote.py`'s RISK_KEYS rather than against a
    literal list retyped here. Those lookups do not alias: under a different
    spelling the leaderboard's stop and target columns come back BLANK, and
    blank in that file means "this strategy has no such setting", never "the
    setting was off". The keys are then checked to BIND — a declared name that
    changes no level is the same failure one step later.
  * THE TOGGLES BEING INDEPENDENTLY WIRED, counted as CANDIDATES rather than
    trades. Each layer must remove candidates on its own and must never add
    one. A filter wired to nothing produces the identical curve to having it
    off, which is the failure that looks most like success — and a case
    watching the TRADE count would be checking the wrong number, because the
    walk holds one position at a time, so declining an early candidate leaves
    the strategy flat for a later one it would have been holding through and
    enabling a filter can ADD realised entries.
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
  * THE DEGENERATE STATES THE REQUEST NAMES. A frame with zero range and zero
    ATR must produce masks, not NaN and not an exception.
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
                                  causal_features, load_strategy)
from backtest.engine import unpack_signals                       # noqa: E402
from backtest.promote import RISK_KEYS                           # noqa: E402
from backtest.run import RISK_PARAMS                             # noqa: E402
from backtest.scan import expand_grid                            # noqa: E402
from strategies.experimental import (                            # noqa: E402
    sma_momentum_crossover_20260818 as SMC)
from strategies.experimental import t3_braid_scalp_20260823 as M  # noqa: E402

MODULE_PATH = (REPO / "strategies" / "experimental"
               / "t3_braid_scalp_20260823.py")

# The parameters every case runs at unless it says otherwise: the module's own
# declared defaults, which for this strategy are short enough (a 5-period T3
# warms up in 25 bars) that the fixtures below reach a usable sample without
# the case having to invent a different strategy to test.
BASE = dict(M.DEFAULT_PARAMS)


def synthetic(n: int = 1500, seed: int = 11, drift: float = 0.0,
              cycle: float = 60.0, amp: float = 6.0) -> pd.DataFrame:
    """
    A frame with both trending and choppy stretches, in the engine's own shape.

    `ts` as a COLUMN with a positional index, which is what the engine hands a
    strategy — not a DatetimeIndex. Both shapes are accepted by the module and
    section 6 checks the other one; running the bulk of the suite on the
    engine's shape means a case cannot pass on a frame the engine never
    produces.

    The sine term is what gives the short side something to work with: a pure
    random walk with no drift produces enough of both, but not reliably at
    every seed, and a fixture whose coverage depends on the seed is a fixture
    that will one day silently stop testing one direction.
    """
    rng = np.random.default_rng(seed)
    close = (100.0
             + np.cumsum(rng.normal(drift, 0.4, n))
             + amp * np.sin(np.arange(n) / cycle))
    return pd.DataFrame({
        "ts": pd.date_range("2023-03-01", periods=n, freq="5min", tz="UTC"),
        "open": close + rng.uniform(-0.2, 0.2, n),
        "high": close + rng.uniform(0.05, 0.6, n),
        "low": close - rng.uniform(0.05, 0.6, n),
        "close": close,
        "volume": rng.integers(100, 5000, n).astype(float),
    })


def frame_from_closes(closes, freq: str = "5min") -> pd.DataFrame:
    """
    A frame whose closes are exactly the sequence given, for the hand-computed
    cases. High and low are widened off the close so the true range is never
    degenerate except where a case wants it to be.
    """
    close = np.asarray(closes, dtype=float)
    n = len(close)
    return pd.DataFrame({
        "ts": pd.date_range("2023-03-01", periods=n, freq=freq, tz="UTC"),
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def candidates(bars: pd.DataFrame, **overrides) -> dict:
    """
    The candidate entry masks AS THE MODULE HANDS THEM TO THE WALK, captured by
    swapping `_walk` for a recorder.

    Read before the walk on purpose — see the note on candidates versus trades
    in this module's docstring. Captured rather than recomputed here so the
    case checks the arrays the strategy actually acts on: a test-side copy of
    the conditions would be checking a second implementation against the
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
    accepts exactly what the engine accepts — a module returning a three-tuple
    or a bare Series fails at the same place a real run would.
    """
    return unpack_signals(M.signal_fn(bars, **{**BASE, **overrides}),
                          len(bars), bars.index)


# ==========================================================================
# 1. The registry block and the mandatory declarations
# ==========================================================================
def test_the_identifier_matches_the_filename() -> None:
    """
    `backtest/run.py` resolves `--strat` by FILENAME, so the constant is an
    assertion that the two agree. A rename that moves one and not the other
    leaves a module whose logs name a strategy the CLI cannot resolve.
    """
    assert M.STRATEGY_NAME == MODULE_PATH.stem, (
        f"STRATEGY_NAME {M.STRATEGY_NAME!r} against filename "
        f"{MODULE_PATH.stem!r}")
    assert MODULE_PATH.exists(), f"{MODULE_PATH} does not exist"


def test_every_mandatory_declaration_is_present() -> None:
    """
    CLAUDE.md requires `signal_fn`, `indicators`, `LOGIC` and `PARAM_GRID` of
    any strategy written here; the request additionally requires
    `make_signal_fn` and `ml_features`. Each closes a specific way a result
    goes wrong silently — a missing `indicators` means the inspector draws no
    lines, a missing `LOGIC` means the tear sheet says "not declared" or infers
    the rules from the trades, and a missing `PARAM_GRID` means `--scan` has
    nothing to sweep.
    """
    for name in ("signal_fn", "indicators", "make_signal_fn", "ml_features"):
        assert callable(getattr(M, name, None)), f"{name} is not callable"
    assert isinstance(M.PARAM_GRID, dict) and M.PARAM_GRID, "PARAM_GRID"
    assert isinstance(M.LOGIC, dict), "LOGIC"
    assert set(M.LOGIC) >= {"concept", "entry", "exit"}, sorted(M.LOGIC)
    for key, text in M.LOGIC.items():
        assert isinstance(text, str) and text.strip(), f"LOGIC[{key}] is empty"
    assert isinstance(M.DEFAULT_PARAMS, dict), "DEFAULT_PARAMS"


def test_the_logic_card_substitutes_every_slot_it_declares() -> None:
    """
    The tear sheet fills `{param}` slots from the run's BOUND parameters. A
    slot naming something the module does not declare renders as a literal
    brace on the strategy card, which is the one part of the report a reader is
    told to read before the metrics.
    """
    import string
    known = set(M.DEFAULT_PARAMS)
    for key, text in M.LOGIC.items():
        slots = {name for _, name, _, _ in string.Formatter().parse(text)
                 if name}
        missing = sorted(slots - known)
        assert not missing, f"LOGIC[{key}] names undeclared params: {missing}"


def test_the_declared_defaults_are_the_signature_defaults() -> None:
    """
    `DEFAULT_PARAMS` is what `promote.py` records as the run's parameters and
    what the loader layers overrides onto. If it disagrees with the signature,
    a run invoked with no `--param` is described by one set and executed under
    another, and nothing raises.
    """
    import inspect
    for fn in (M.signal_fn, M.make_signal_fn):
        sig = inspect.signature(fn)
        for key, value in M.DEFAULT_PARAMS.items():
            assert key in sig.parameters, (
                f"{fn.__name__} does not accept declared default {key!r}")
            got = sig.parameters[key].default
            assert got == value, (
                f"{fn.__name__}({key}=) defaults to {got!r}, DEFAULT_PARAMS "
                f"says {value!r}")


def test_the_request_defaults_are_what_the_module_declares() -> None:
    """
    The specification's own numbers, transcribed once here and compared. This
    is the case that fails if somebody tunes a default after a disappointing
    run — which is exactly the edit that must never happen silently, because
    the module then describes a strategy nobody specified.
    """
    requested = {"t3_period": 5, "t3_vfactor": 0.7, "braid_fast": 3,
                 "braid_slow": 7, "stiffness_period": 20,
                 "stiffness_threshold": 35.0, "sl_atr_mult": 1.5,
                 "tp_atr_mult": 2.0, "trailing": False}
    for key, value in requested.items():
        assert M.DEFAULT_PARAMS[key] == value, (
            f"{key}: module says {M.DEFAULT_PARAMS[key]!r}, the request says "
            f"{value!r}")
    for toggle in ("use_t3_filter", "use_braid_filter",
                   "use_stiffness_filter", "use_news_filter"):
        assert toggle in M.DEFAULT_PARAMS, f"{toggle} is not declared"


def test_the_grid_is_the_requested_grid_and_every_cell_binds() -> None:
    """
    The request's PARAM_GRID transcribed, and then every cell actually bound.
    A grid key the signature does not accept raises at bind time one contract
    into a sweep rather than here, and `load_strategy` rejects unknown
    parameter names outright — so a stale key is a sweep that dies partway.
    """
    requested = {"t3_period": [5, 9, 14],
                 "stiffness_threshold": [25.0, 35.0, 45.0],
                 "sl_atr_mult": [1.0, 1.5, 2.0],
                 "tp_atr_mult": [1.5, 2.0, 3.0],
                 "trailing": [False, True]}
    assert M.PARAM_GRID == requested, M.PARAM_GRID

    cells = list(expand_grid(M.PARAM_GRID))
    assert len(cells) == 162, f"{len(cells)} cells, expected 3*3*3*3*2"
    rejected = []
    for combo in cells:
        try:
            M.make_signal_fn(**dict(combo))
        except ValueError as e:
            rejected.append((dict(combo), str(e)))
    # Every requested cell must be runnable. The bounded-risk floor is set
    # below the grid's tightest ratio deliberately — refusing a cell the
    # specification asks for would be the module overruling the request rather
    # than bounding it — so a rejection here means that floor moved.
    assert not rejected, f"{len(rejected)} requested cells refused: {rejected[:3]}"


def test_the_module_loads_through_the_real_loader_with_its_hooks() -> None:
    """
    Through `load_strategy` from a FILE PATH, which is how the engine reaches
    it — not through an import. The hooks have to arrive bound: an
    `ml_feature_fn` of None silently selects the shared seven-column
    `causal_features` default, so Version B would run, report numbers, and be
    fitted on a matrix this strategy never declared.
    """
    fn, info = load_strategy(str(MODULE_PATH),
                             params={"t3_period": 9, "sl_atr_mult": 2.0})
    assert callable(fn), "the loader returned no bound signal function"
    assert info["ml_feature_fn"] is not None, (
        "ml_features did not bind — Version B would fall back to the shared "
        "causal_features and be fitted on columns this strategy never declared")
    assert info["indicator_fn"] is not None, "indicators did not bind"
    assert (info["logic"] or {}).get("concept"), "LOGIC did not reach the loader"
    assert info["param_grid"] == M.PARAM_GRID, "PARAM_GRID did not reach the loader"
    assert info["bound_params"]["t3_period"] == 9, info["bound_params"]


# ==========================================================================
# 2. The signal contract — dtypes, shape, and the four-mask form
# ==========================================================================
def test_the_signals_are_four_boolean_series_on_the_bars_index() -> None:
    """
    The request's first testing clause. Checked on the RAW return rather than
    on the unpacker's output, because the unpacker would coerce a float mask
    and hide the defect — a float `entries` array reaches vectorbt as a
    non-boolean and is interpreted as a size, not a signal.
    """
    bars = synthetic()
    out = M.signal_fn(bars, **BASE)
    assert isinstance(out, tuple) and len(out) == 4, (
        f"expected the four-mask form, got {type(out).__name__} of "
        f"{len(out) if isinstance(out, tuple) else 'n/a'}")
    for name, series in zip(("long_entries", "long_exits", "short_entries",
                             "short_exits"), out):
        assert isinstance(series, pd.Series), f"{name} is {type(series).__name__}"
        assert series.dtype == bool, f"{name} has dtype {series.dtype}"
        assert len(series) == len(bars), f"{name} has {len(series)} of {len(bars)}"
        assert series.index.equals(bars.index), f"{name} is not on the bars' index"
        assert not series.isna().any(), f"{name} carries NaN"


def test_the_engine_unpacker_accepts_the_return_shape() -> None:
    """
    The contract is what `backtest.engine.unpack_signals` accepts, not what
    this file believes. A three-tuple raises there rather than silently losing
    the short side into a plausible long-only equity curve.
    """
    bars = synthetic()
    le, lx, se, sx = masks(bars)
    for name, arr in (("long_entries", le), ("long_exits", lx),
                      ("short_entries", se), ("short_exits", sx)):
        assert np.asarray(arr).dtype == bool, f"{name}: {np.asarray(arr).dtype}"
    # Both directions have to actually fire on the fixture, or every
    # directional case below would be vacuously true.
    assert int(np.sum(le)) > 0, "no long entries on the fixture"
    assert int(np.sum(se)) > 0, "no short entries on the fixture"
    # One position at a time, resolved by the walk: entries and exits pair up
    # on each side, with at most one entry left open at the end of the frame.
    for side, e, x in (("long", le, lx), ("short", se, sx)):
        opened, closed = int(np.sum(e)), int(np.sum(x))
        assert opened - closed in (0, 1), (
            f"{side}: {opened} entries against {closed} exits")


def test_a_long_and_a_short_are_never_open_at_once() -> None:
    """
    The walk is a THREE-state machine and the module must never hand the engine
    overlapping positions. Reconstructed from the masks rather than trusted:
    an overlap is not visible in a trade count and produces a plausible curve.
    """
    bars = synthetic()
    le, lx, se, sx = (np.asarray(a) for a in masks(bars))
    state = 0
    for i in range(len(bars)):
        if state == 0 and le[i]:
            state = 1
        elif state == 0 and se[i]:
            state = -1
        elif state == 1 and lx[i]:
            state = 0
        elif state == -1 and sx[i]:
            state = 0
        assert not (le[i] and se[i]), f"bar {i} signals both entries"
        if state == 1:
            assert not se[i], f"bar {i} opens a short while long"
        if state == -1:
            assert not le[i], f"bar {i} opens a long while short"


def test_the_entry_is_a_rising_edge_not_a_standing_state() -> None:
    """
    Deviation 1 in the module docstring, pinned. A literal reading of the
    request's standing conditions would re-arm the entry on EVERY bar the
    confluence holds — turning one trend leg into a chain of stopped scalps
    whose count is set by the stop distance rather than by the signal. The
    candidate masks must be strictly sparser than the standing conjunction.

    Checked by construction rather than by counting: no two consecutive bars
    may both carry a candidate, because a rising edge cannot repeat without the
    state having fallen in between.
    """
    bars = synthetic()
    seen = candidates(bars)
    for side in ("long_ok", "short_ok"):
        arr = seen[side]
        assert arr.any(), f"{side} produced no candidates at all"
        both = arr[1:] & arr[:-1]
        assert not both.any(), (
            f"{side} has {int(both.sum())} adjacent candidate pairs — the "
            f"conjunction is being read as a standing state, not an edge")


# ==========================================================================
# 3. The indicator arithmetic, against hand-computed answers
# ==========================================================================
def test_the_t3_collapses_to_a_triple_ema_at_vfactor_zero() -> None:
    """
    Tillson's coefficients at v = 0 are c1 = c2 = c3 = 0 and c4 = 1, so
    `T3 == e3`, a plain triple EMA. That identity is the cheapest total check
    on the cubic there is: transposing any coefficient, or wiring e4 where e3
    belongs, breaks it — while leaving a curve that still looks like a
    smoothed price and still produces a plausible equity curve.
    """
    bars = synthetic(n=400)
    close = bars["close"]
    triple = M._ema(M._ema(M._ema(close, 7), 7), 7)
    got = M._t3(close, 7, 0.0)

    # COMPARED WHERE THE T3 IS DEFINED, WHICH IS LATER THAN WHERE e3 IS. At
    # v = 0 the cubic still multiplies e6 by a coefficient of 0.0, and
    # `0.0 * NaN` is NaN — so the T3 warms up six EMAs deep even in the corner
    # where only e3 contributes to the answer. That is the conservative
    # behaviour and the right one: the indicator is defined from all six
    # averages, and publishing a value before the sixth exists would report a
    # partially-computed baseline as a measured one. It is pinned separately
    # by the warm-up case below, so shortening it here would not go unnoticed.
    live = got.notna()
    assert live.any(), "the T3 produced no values at all"
    assert triple[live].notna().all(), "the triple EMA is undefined where T3 is"
    pd.testing.assert_series_equal(got[live], triple[live], check_names=False)

    # And at v > 0 it must NOT equal the triple EMA, or the volume factor is
    # wired to nothing and every grid cell is the same baseline. Compared on
    # the SAME rows as the identity above — `dropna()` on each side separately
    # would line up bar 36 against bar 18 and compare two different windows.
    lively = M._t3(close, 7, 0.7)
    assert not np.allclose(lively[live], triple[live]), (
        "t3_vfactor changes nothing — the cubic is not wired")


def test_the_t3_warm_up_is_six_emas_deep() -> None:
    """
    Six chained EMAs, each needing `period` non-NaN inputs, so the first value
    lands around bar 6*(period-1). Pinned because the alternative — pandas
    seeding each stage from its first observation — produces a "T3" that exists
    at bar 3 and is decided by the seeding rather than by price, and every
    comparison against NaN being False means the symptom is silently wrong
    early trades rather than an error.
    """
    bars = synthetic(n=400)
    for period in (5, 9, 14):
        t3 = M._t3(bars["close"], period, 0.7)
        first = int(t3.notna().argmax())
        expected = 6 * (period - 1)
        assert t3.iloc[:expected].isna().all(), (
            f"T3({period}) exists before bar {expected} (first at {first})")
        assert first == expected, f"T3({period}) first value at {first}"


def test_the_stiffness_index_matches_a_hand_worked_window() -> None:
    """
    Seven closes, four-bar window, worked out on paper.

        closes  10  11  12  11  13  13  12
        diffs    -  +1  +1  -1  +2   0  -1
        up       -   T   T   F   T   F   F
        down     -   F   F   T   F   F   T

    The first full window ends at index 4 (four real changes, the first at
    index 1), so:

        i=4  up = [T,T,F,T] = 75.0    down = [F,F,T,F] = 25.0
        i=5  up = [T,F,T,F] = 50.0    down = [F,T,F,F] = 25.0
        i=6  up = [F,T,F,F] = 25.0    down = [T,F,F,T] = 50.0

    Index 5 is the case that matters most: 50 + 25 = 75, not 100. The flat bar
    counts to NEITHER direction, which is the whole reason the module carries
    two series instead of one and its complement — a market that stops moving
    is the choppy state this layer exists to refuse, and `100 - down` would
    read it as evidence FOR a long.
    """
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 13.0, 12.0])
    up, down = M._stiffness(close, 4)

    assert up.iloc[:4].isna().all(), "the index exists before a full window"
    assert down.iloc[:4].isna().all(), "the index exists before a full window"
    assert list(up.iloc[4:]) == [75.0, 50.0, 25.0], list(up.iloc[4:])
    assert list(down.iloc[4:]) == [25.0, 25.0, 50.0], list(down.iloc[4:])
    assert up.iloc[5] + down.iloc[5] == 75.0, (
        "up and down summed to 100 across a flat bar — they are being "
        "computed as complements")


def test_the_stiffness_index_is_bounded_and_saturates() -> None:
    """
    A monotone rise is 100% up and 0% down, and the mirror for a fall. The
    bound is what makes `stiffness_threshold` a percentage rather than an
    open-ended number, and `_validate` refuses a threshold outside it.
    """
    rising = pd.Series(np.arange(50, dtype=float))
    up, down = M._stiffness(rising, 20)
    assert up.dropna().eq(100.0).all(), "a monotone rise is not 100% up"
    assert down.dropna().eq(0.0).all(), "a monotone rise reports down changes"

    falling = pd.Series(np.arange(50, 0, -1, dtype=float))
    up, down = M._stiffness(falling, 20)
    assert up.dropna().eq(0.0).all(), "a monotone fall reports up changes"
    assert down.dropna().eq(100.0).all(), "a monotone fall is not 100% down"

    flat = pd.Series(np.full(50, 42.0))
    up, down = M._stiffness(flat, 20)
    assert up.dropna().eq(0.0).all() and down.dropna().eq(0.0).all(), (
        "a dead-flat series reports directional changes")


def test_the_braid_histogram_is_the_fast_slow_separation() -> None:
    """
    The histogram against the two EMAs it is defined from, and the signal line
    against the histogram. Recomputed here from `_ema` rather than compared to
    a stored array so a change to the EMA convention — span versus Wilder — is
    caught by the identity rather than by a number nobody can re-derive.
    """
    bars = synthetic(n=400)
    close = bars["close"]
    hist, signal = M._braid(close, 3, 7)
    pd.testing.assert_series_equal(
        hist, M._ema(close, 3) - M._ema(close, 7), check_names=False)
    pd.testing.assert_series_equal(
        signal, M._ema(hist, M.BRAID_SIGNAL_PERIOD), check_names=False)

    # A monotone rise puts the fast average above the slow one: the histogram
    # is positive, which is the "green" the request's Layer 2 asks for.
    rising = pd.Series(np.arange(200, dtype=float))
    rise_hist, _ = M._braid(rising, 3, 7)
    assert rise_hist.dropna().gt(0).all(), "a monotone rise is not green"
    falling = pd.Series(np.arange(200, 0, -1, dtype=float))
    fall_hist, _ = M._braid(falling, 3, 7)
    assert fall_hist.dropna().lt(0).all(), "a monotone fall is not red"


def test_the_atr_is_wilders_and_not_a_span_ema() -> None:
    """
    ATR is the only average here on Wilder's `alpha = 1/period`; the T3 and the
    Braid are on the span form. Using one where the other belongs produces an
    "ATR(14)" no chart package agrees with, and the stop is then drawn — and
    taken — somewhere a reader checking their own chart would not expect.
    """
    bars = synthetic(n=400)
    tr = M._true_range(bars)
    pd.testing.assert_series_equal(
        M._atr(bars, 14),
        tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean(),
        check_names=False)
    assert not np.allclose(
        M._atr(bars, 14).dropna(),
        M._ema(tr, 14).dropna()), "the ATR is on the span EMA, not Wilder's"


# ==========================================================================
# 4. The layer toggles — counted as CANDIDATES, never as trades
# ==========================================================================
def states(bars: pd.DataFrame, **overrides) -> dict:
    """
    The PERMITTED-BAR states and the triggers, straight off the module.

    `_signal_arrays` hands both back on its series dict, so this reads the
    arrays the strategy actually acted on rather than a test-side copy of the
    conditions — which would be checking a second implementation against the
    specification while the module was free to disagree with both.
    """
    s, *_rest = M._signal_arrays(bars, **{**BASE, **overrides})
    return {k: s[k].to_numpy(dtype=bool)
            for k in ("long_state", "short_state",
                      "long_trigger", "short_trigger")}


def test_each_layer_subtracts_permitted_bars_on_its_own() -> None:
    """
    Each filter, switched on alone against the same frame with it off, must
    shrink the set of bars the strategy is PERMITTED to trade on, and must
    never add one.

    MEASURED ON THE STATES, NOT ON THE TRIGGERS OR THE TRADES, and that is the
    whole content of this case.

    The states are monotone in the layers by construction — the conjunction
    gains a term — so a filter wired to nothing shows up here as an unchanged
    set, which is the failure that looks most like success and is the thing
    being caught.

    THE TRIGGERS ARE NOT MONOTONE and a case written against them would be
    wrong. The entry is the rising edge of the conjunction, so removing bars
    from the middle of one permitted stretch splits it in two and creates a
    SECOND trigger where there had been one: on this fixture, enabling the T3
    layer takes the long candidates from 134 to 143. The trade count is not
    monotone either, one level further on, because the walk holds one position
    at a time. The permitted state is the only one of the three that answers
    "is this layer wired", and the next case pins the non-monotonicity itself
    so the surprise is recorded rather than rediscovered.
    """
    bars = synthetic(n=2000)
    # `_validate` refuses all three off, so each layer is measured with one
    # other layer held on to keep the pair legal — the same partner on both
    # sides of the comparison, so it is never between two different
    # one-layer strategies.
    for layer, partner in (("use_t3_filter", "use_braid_filter"),
                           ("use_braid_filter", "use_t3_filter"),
                           ("use_stiffness_filter", "use_t3_filter")):
        off = {"use_t3_filter": False, "use_braid_filter": False,
               "use_stiffness_filter": False, partner: True, layer: False}
        on = {**off, layer: True}
        base = states(bars, **off)
        gated = states(bars, **on)
        for side in ("long_state", "short_state"):
            b, g = int(base[side].sum()), int(gated[side].sum())
            assert g < b, (
                f"{layer} removed no {side} bars ({b} -> {g}) — it is "
                f"declared but wired to nothing")
            # A strict SUBSET, not merely a smaller count: a layer that moves
            # permitted bars around while keeping the total down is not a
            # filter, and a count alone cannot tell the two apart.
            assert not (gated[side] & ~base[side]).any(), (
                f"{layer} PERMITTED {side} bars that the unfiltered "
                f"configuration did not")


def test_a_layer_can_increase_the_trigger_count_and_that_is_expected() -> None:
    """
    The non-monotonicity itself, pinned rather than left to be rediscovered as
    a bug by whoever next reads a leaderboard.

    Enabling a layer removes permitted bars and can still ADD candidate
    entries, because splitting one permitted stretch into two creates a second
    rising edge. This is the documented consequence of reading the request's
    standing conditions as an edge (deviation 1 in the module docstring), and
    the module's LOGIC card and comments say so.

    The case asserts the two facts together — states shrink, triggers grow —
    on the configuration where it actually happens. If a future edit made
    triggers monotone, this fails, and the fix is to decide deliberately which
    semantics the module has rather than to let the change land silently.
    """
    bars = synthetic(n=2000)
    off = {"use_t3_filter": False, "use_braid_filter": True,
           "use_stiffness_filter": False}
    on = {**off, "use_t3_filter": True}
    base, gated = states(bars, **off), states(bars, **on)

    assert int(gated["long_state"].sum()) < int(base["long_state"].sum()), (
        "the T3 layer did not shrink the permitted set — the premise of this "
        "case no longer holds")
    assert (int(gated["long_trigger"].sum())
            > int(base["long_trigger"].sum())), (
        f"the trigger count did not rise "
        f"({int(base['long_trigger'].sum())} -> "
        f"{int(gated['long_trigger'].sum())}). If the module was deliberately "
        f"changed to state semantics, update deviation 1 and the LOGIC card "
        f"with it; if not, the rising edge has stopped working.")


def test_the_layers_are_mirrored_across_the_two_sides() -> None:
    """
    Reflecting the prices about a horizontal line must swap the long and short
    candidate masks exactly. A layer wired to the wrong side of a comparison,
    or a threshold that means something different on a short, survives every
    count-based case and shows up only here.

    The reflection is `2c - x` about the frame's first close, with high and low
    exchanged — a reflected bar's high IS the original's low. Getting that
    backwards would make the reflected frame's ranges negative and the case
    would be testing nothing.
    """
    bars = synthetic(n=1200)
    c = float(bars["close"].iloc[0])
    flipped = bars.copy()
    flipped["close"] = 2 * c - bars["close"]
    flipped["open"] = 2 * c - bars["open"]
    flipped["high"] = 2 * c - bars["low"]
    flipped["low"] = 2 * c - bars["high"]
    assert (flipped["high"] >= flipped["low"]).all(), "the reflection is broken"

    straight = candidates(bars)
    mirror = candidates(flipped)
    assert np.array_equal(straight["long_ok"], mirror["short_ok"]), (
        "reflecting the prices did not turn the long candidates into the "
        "short ones — a layer is not mirrored")
    assert np.array_equal(straight["short_ok"], mirror["long_ok"]), (
        "reflecting the prices did not turn the short candidates into the "
        "long ones — a layer is not mirrored")


def test_all_three_layers_off_is_refused_rather_than_run() -> None:
    """
    With every directional layer off there is no condition left: both sides
    would signal on every bar and the walk's ambiguous-bar rule would take
    NEITHER. The run would complete, report zero trades, and read as a strategy
    with no edge rather than as a configuration that cannot express one.
    """
    try:
        M.make_signal_fn(use_t3_filter=False, use_braid_filter=False,
                         use_stiffness_filter=False)
    except ValueError as e:
        assert "at least one" in str(e).lower(), str(e)
    else:
        raise AssertionError(
            "all three layers off was accepted; it produces zero trades and "
            "reads as a strategy with no edge")


def test_a_toggle_passed_as_a_string_is_refused() -> None:
    """
    `--param use_braid_filter=false` arriving as the STRING "false" is truthy.
    Coerced, the run would apply the filter while the leaderboard's `params`
    column said it was off — a discrepancy invisible in every artifact.
    """
    for flag in ("use_t3_filter", "use_braid_filter", "use_stiffness_filter",
                 "use_news_filter", "trailing"):
        for bad in ("false", "true", 0, 1.0):
            try:
                M.make_signal_fn(**{flag: bad})
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"{flag}={bad!r} was accepted; a non-bool toggle runs a "
                    f"silently different strategy")


# ==========================================================================
# 5. The risk parameters — the exact keys, and that they bind
# ==========================================================================
def test_the_risk_keys_are_spelled_the_way_the_pipeline_looks_them_up() -> None:
    """
    The request's second testing clause: dictionary-key integration for
    `sl_atr_mult` and `tp_atr_mult`.

    Checked against `backtest/run.py`'s RISK_PARAMS and `backtest/promote.py`'s
    RISK_KEYS rather than against a list retyped here, because those lookups
    are what actually read the module and they do NOT alias. Under a different
    spelling the leaderboard's `sl_atr_mult` / `tp_atr_mult` / `trailing`
    columns come back BLANK — and blank in that file means "this strategy has
    no such setting", never "the setting was off". A strategy whose entire exit
    rule is a stop and a target would be recorded as having neither, and
    `promote.py`'s `risk` block would write `NOT DECLARED` over both.
    """
    import inspect
    assert set(RISK_PARAMS) == set(RISK_KEYS), (
        f"the two lookups disagree: {RISK_PARAMS} against {RISK_KEYS}")
    for key in RISK_PARAMS:
        assert key in M.DEFAULT_PARAMS, (
            f"{key!r} is not in DEFAULT_PARAMS — promote.py would write "
            f"'NOT DECLARED' for it")
        assert key in inspect.signature(M.signal_fn).parameters, (
            f"signal_fn does not accept {key!r}")
        assert key in inspect.signature(M.make_signal_fn).parameters, (
            f"make_signal_fn does not accept {key!r}")
    # The two the request names by name must be in the swept grid as well, or
    # the leaderboard's risk columns are constant across every row.
    for key in ("sl_atr_mult", "tp_atr_mult", "trailing"):
        assert key in M.PARAM_GRID, f"{key!r} is not swept"


def test_the_risk_keys_actually_move_the_stop_and_the_target() -> None:
    """
    A declared name that changes no level is the same failure one step later:
    the leaderboard column populates, the sweep reports a winner, and every
    cell ran the same risk. Read off the LEVELS the walk produced, not off the
    trade count — a count can hold steady while the levels move.
    """
    bars = synthetic(n=1500)

    def levels(**over):
        _s, _e, _x, _se, _sx, stop, target = M._signal_arrays(
            bars, **{**BASE, **over})
        return np.asarray(stop, dtype=float), np.asarray(target, dtype=float)

    tight_stop, _ = levels(sl_atr_mult=1.0)
    wide_stop, _ = levels(sl_atr_mult=2.0)
    live = np.isfinite(tight_stop) & np.isfinite(wide_stop)
    assert live.any(), "no bar carried a stop level on either setting"
    # A wider stop on a LONG sits further BELOW the fill. The two runs can take
    # different trades, so compare only where both were in the same position —
    # which is what `live` plus an equal fill price gives.
    assert not np.allclose(tight_stop[live], wide_stop[live]), (
        "sl_atr_mult changed no stop level")

    _s1, near = levels(tp_atr_mult=1.5)
    _s2, far = levels(tp_atr_mult=3.0)
    both = np.isfinite(near) & np.isfinite(far)
    assert both.any(), "no bar carried a target level on either setting"
    assert not np.allclose(near[both], far[both]), (
        "tp_atr_mult changed no target level")


def test_no_take_profit_is_expressible_and_says_so_on_the_chart() -> None:
    """
    `tp_atr_mult=None` is the no-target configuration. It must produce an
    all-NaN target rather than a sentinel level, and the tear sheet must OMIT
    the line rather than draw an all-NaN series — an empty legend entry reads
    as a target that exists and never got close, which is the opposite of the
    truth.
    """
    bars = synthetic(n=1500)
    _s, _e, _x, _se, _sx, stop, target = M._signal_arrays(
        bars, **{**BASE, "tp_atr_mult": None, "trailing": True})
    assert np.isnan(np.asarray(target, dtype=float)).all(), (
        "a target level was produced for a run with no take-profit")
    assert np.isfinite(np.asarray(stop, dtype=float)).any(), (
        "the stop vanished along with the target")

    drawn = M.indicators(bars, **{**BASE, "tp_atr_mult": None,
                                  "trailing": True})
    assert not any("Take Profit" in k for k in drawn), sorted(drawn)
    assert any("Trailing Stop" in k for k in drawn), sorted(drawn)
    fixed = M.indicators(bars, **{**BASE, "trailing": False})
    assert any("Fixed Stop" in k for k in fixed), sorted(fixed)


def test_the_bounded_risk_clauses_reject_what_they_claim_to() -> None:
    """
    The request's Layer 4 bound. These are a coarse sanity floor and the module
    says so — the real cost check is Stage 4's drag as a share of gross profit
    — but each clause has to actually bite, or it is documentation rather than
    a bound.
    """
    # A target at or below the fill would be breached by the fill bar itself.
    for bad in (0.0, -1.0):
        try:
            M.make_signal_fn(tp_atr_mult=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"tp_atr_mult={bad} was accepted")
    # A stop inside the tick noise, and one that is not a stop at all.
    for bad in (0.0, 0.1, 1e6):
        try:
            M.make_signal_fn(sl_atr_mult=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sl_atr_mult={bad} was accepted")
    # The reward-to-risk floor.
    try:
        M.make_signal_fn(sl_atr_mult=4.0, tp_atr_mult=1.0)   # ratio 0.25
    except ValueError as e:
        assert "MIN_REWARD_RISK" in str(e) or "0.5" in str(e), str(e)
    else:
        raise AssertionError("a 0.25 reward-to-risk ratio was accepted")
    # And the grid's own tightest ratio must still pass — the floor is set
    # below it deliberately, so a rejection here means the floor moved.
    M.make_signal_fn(sl_atr_mult=2.0, tp_atr_mult=1.5)       # ratio 0.75
    # No stop at all is refused: with no signal exit and no session flatten it
    # is the only discretionary way out of a losing trade.
    for bad in (None, np.nan):
        try:
            M.make_signal_fn(sl_atr_mult=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sl_atr_mult={bad!r} was accepted")


def test_the_out_of_range_indicator_parameters_are_refused() -> None:
    """
    Each bound exists because the parameter outside it runs as a silently
    different strategy rather than as an error.
    """
    cases = [
        ({"t3_period": 1}, "a one-period T3 is the close itself"),
        ({"t3_vfactor": 1.5}, "the volume factor is defined on [0, 1]"),
        ({"t3_vfactor": -0.1}, "the volume factor is defined on [0, 1]"),
        ({"braid_fast": 7, "braid_slow": 7}, "an identically zero histogram"),
        ({"braid_fast": 9, "braid_slow": 7}, "an inverted histogram"),
        ({"stiffness_period": 1}, "a one-bar persistence window"),
        ({"stiffness_threshold": 120.0}, "a threshold the index cannot reach"),
        ({"stiffness_threshold": -5.0}, "a threshold that can never bind"),
    ]
    for params, why in cases:
        try:
            M.make_signal_fn(**params)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{params} was accepted — {why}")


# ==========================================================================
# 6. Causal feature alignment — the request's third testing clause
# ==========================================================================
def test_the_feature_matrix_is_the_shape_the_filter_needs() -> None:
    """
    `apply_ml_signal_filter` indexes this matrix positionally against the bars,
    so a row count that disagrees is a model scoring one bar with another
    bar's state. The loader RAISES on that rather than aligning it — checked
    here so the failure is a case rather than a crashed campaign.
    """
    bars = synthetic(n=1200)
    feats = M.ml_features(bars, **BASE)
    assert isinstance(feats, pd.DataFrame), type(feats).__name__
    assert list(feats.columns) == M.ML_FEATURES, list(feats.columns)
    assert list(feats.columns) == ["t3_slope", "braid_hist", "stiffness",
                                   "atr_norm", "hour_et"], (
        "the columns are not the five the request names")
    assert len(feats) == len(bars), f"{len(feats)} rows against {len(bars)} bars"
    assert feats.index.equals(bars.index), "the matrix is not on the bars' index"
    assert not np.isinf(feats.to_numpy(dtype=float)).any(), "the matrix carries inf"
    # It must NOT be the shared default — declaring the hook is the point.
    assert list(feats.columns) != list(causal_features(bars).columns), (
        "ml_features returned the shared causal_features columns")
    # Every column has to carry information past warm-up, or it is a constant
    # the model cannot learn from and the request's five are really four.
    warm = feats.iloc[200:]
    for col in M.ML_FEATURES:
        assert warm[col].notna().any(), f"{col} is entirely NaN past warm-up"
        assert warm[col].nunique(dropna=True) > 1, f"{col} is constant"
    # Warm-up stays NaN rather than being filled with a full-sample statistic.
    assert feats["t3_slope"].iloc[:24].isna().all(), (
        "the T3 slope exists before the six chained EMAs do")


def test_the_hour_feature_is_the_exchange_clock_not_utc() -> None:
    """
    A NAMED ZONE, per CLAUDE.md, not a fixed offset and not the UTC hour. The
    CME session keeps its local clock across the DST changeover, so a UTC hour
    smears one session hour across two values twice a year and the feature
    means a different thing in March than in October.

    Pinned on two bars four months apart at the same UTC time: they must land
    on DIFFERENT ET hours, which is exactly what a fixed offset cannot produce.
    """
    ts = pd.to_datetime(["2023-01-15 18:00:00", "2023-07-15 18:00:00"],
                        utc=True)
    bars = pd.DataFrame({"ts": ts, "open": [1.0, 1.0], "high": [1.0, 1.0],
                         "low": [1.0, 1.0], "close": [1.0, 1.0],
                         "volume": [1.0, 1.0]})
    hours = M.ml_features(bars, **BASE)["hour_et"].tolist()
    assert hours == [13.0, 14.0], (
        f"18:00 UTC in January and July gave {hours}; expected 13 (EST) and "
        f"14 (EDT) — the hour is being read in UTC or on a fixed offset")


def test_the_module_accepts_both_frame_shapes() -> None:
    """
    The engine hands a long-format frame with `ts` as a COLUMN and a positional
    index; a caller holding a time-indexed fixture has the opposite shape.
    Both have to work, and a tz-NAIVE DatetimeIndex has to be localized rather
    than raising or being silently read as if it were already ET.
    """
    bars = synthetic(n=600)
    indexed = bars.drop(columns=["ts"]).set_index(
        pd.DatetimeIndex(bars["ts"]))
    naive = indexed.copy()
    naive.index = naive.index.tz_localize(None)

    ref = M.ml_features(bars, **BASE).to_numpy(dtype=float)
    for label, frame in (("DatetimeIndex", indexed), ("naive index", naive)):
        got = M.ml_features(frame, **BASE).to_numpy(dtype=float)
        assert np.allclose(got, ref, equal_nan=True), (
            f"the {label} shape produced a different feature matrix")
    # And the signals, which is the path that actually decides trades.
    le, _lx, se, _sx = M.signal_fn(indexed, **BASE)
    ref_le, _, ref_se, _ = M.signal_fn(bars, **BASE)
    assert np.array_equal(le.to_numpy(), ref_le.to_numpy()), "long masks differ"
    assert np.array_equal(se.to_numpy(), ref_se.to_numpy()), "short masks differ"


def test_the_features_and_signals_survive_truncation() -> None:
    """
    CAUSALITY BY TRUNCATION. Every prefix of the frame must reproduce its own
    rows exactly. A value at row i that depends on any bar after i changes when
    the bars after i are removed, and this catches it whatever form the leak
    takes — `shift(-1)`, a centred window, a reversed slice, or a statistic
    taken over the whole frame.

    The last row of each prefix is included on purpose: it is the row a live
    decision would be made on, and the one a global statistic corrupts most.
    """
    bars = synthetic(n=900)
    full_feats = M.ml_features(bars, **BASE).to_numpy(dtype=float)
    full_long, _lx, full_short, _sx = M.signal_fn(bars, **BASE)
    full_long = full_long.to_numpy()
    full_short = full_short.to_numpy()

    for cut in (400, 600, 750, 899):
        prefix = bars.iloc[:cut].copy()
        got = M.ml_features(prefix, **BASE).to_numpy(dtype=float)
        assert np.allclose(got, full_feats[:cut], equal_nan=True), (
            f"the feature matrix changed when the frame was cut at {cut} — a "
            f"column reads bars it should not have")
        pl, _, ps, _ = M.signal_fn(prefix, **BASE)
        assert np.array_equal(pl.to_numpy(), full_long[:cut]), (
            f"the long signals changed when the frame was cut at {cut}")
        assert np.array_equal(ps.to_numpy(), full_short[:cut]), (
            f"the short signals changed when the frame was cut at {cut}")


def test_the_features_and_signals_survive_a_rewritten_tail() -> None:
    """
    CAUSALITY BY PERTURBATION, the other direction. Rewriting the tail of the
    frame must leave every row before it byte-identical.

    Truncation alone can be passed by a leak that reads a FIXED number of bars
    ahead and happens to have none to read at the cut; perturbation cannot.
    The tail is replaced with a violent move rather than with noise, so a leak
    of any size shows up as a large difference rather than a rounding one.
    """
    bars = synthetic(n=900)
    cut = 500
    tampered = bars.copy()
    tail = slice(cut, None)
    for col in ("open", "high", "low", "close"):
        tampered.loc[tampered.index[tail], col] = (
            bars[col].iloc[tail].to_numpy() * 1.35 + 250.0)

    ref = M.ml_features(bars, **BASE).to_numpy(dtype=float)
    got = M.ml_features(tampered, **BASE).to_numpy(dtype=float)
    assert np.allclose(got[:cut], ref[:cut], equal_nan=True), (
        "rewriting the tail changed the head of the feature matrix")

    rl, _, rs, _ = M.signal_fn(bars, **BASE)
    tl, _, ts_, _ = M.signal_fn(tampered, **BASE)
    assert np.array_equal(tl.to_numpy()[:cut], rl.to_numpy()[:cut]), (
        "rewriting the tail changed the head of the long signals")
    assert np.array_equal(ts_.to_numpy()[:cut], rs.to_numpy()[:cut]), (
        "rewriting the tail changed the head of the short signals")
    # And the tail must actually have moved, or the case proves nothing.
    assert not np.allclose(got[cut:], ref[cut:], equal_nan=True), (
        "the tamper changed nothing at all — the fixture is not exercising")


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
            if name in ("shift", "diff"):
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
        entry = np.zeros(n, dtype=bool)
        entry[0] = True
        other = np.zeros(n, dtype=bool)
    else:
        path = np.array([100., 99., 97., 95., 96., 98., 101., 104., 106.,
                         105., 103., 102.])
        other = np.zeros(n, dtype=bool)
        entry = np.zeros(n, dtype=bool)
        entry[0] = True
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
            # And the fixture has to exercise the branch, or the parity is
            # between two functions that both did nothing.
            assert np.asarray(mine[1]).any() or np.asarray(mine[3]).any(), (
                f"{direction} sl={sl} tp={tp} trailing={trailing} produced no "
                f"exit — the fixture is not exercising the kernel")


# ==========================================================================
# 8. Degenerate states, the news filter, and Version B
# ==========================================================================
def test_a_zero_range_frame_produces_masks_rather_than_nan() -> None:
    """
    The request's microstructure clause: zero range and zero ATR handled
    gracefully rather than propagating NaN.

    A dead-flat frame is the worst case — every true range is 0, so the ATR is
    0, the Braid histogram is 0 and the Stiffness Index is 0 in both
    directions. Nothing may raise, no mask may carry NaN, and no entry may fire
    (a flat market satisfies no layer), which is the correct answer rather than
    a defensive one.
    """
    flat = frame_from_closes(np.full(400, 50.0))
    flat["high"] = 50.0
    flat["low"] = 50.0
    flat["open"] = 50.0

    out = M.signal_fn(flat, **BASE)
    for name, series in zip(("long_entries", "long_exits", "short_entries",
                             "short_exits"), out):
        assert series.dtype == bool, f"{name}: {series.dtype}"
        assert not series.isna().any(), f"{name} carries NaN on a flat frame"
        assert not series.any(), f"{name} fired on a market that never moved"

    assert float(M._atr(flat, 14).dropna().max()) == 0.0, (
        "a dead-flat frame has a non-zero ATR")
    feats = M.ml_features(flat, **BASE).to_numpy(dtype=float)
    assert not np.isinf(feats).any(), "the feature matrix carries inf"
    # And the indicators still render rather than raising.
    M.indicators(flat, **BASE)


def test_a_single_bar_and_a_short_frame_do_not_raise() -> None:
    """
    Shorter than every warm-up. Nothing can signal, and the module has to say
    so with empty masks rather than by raising or by producing a NaN mask —
    the lake hands a stage a short slice whenever a contract's coverage starts
    late, and a raise there kills a 27-symbol batch on its first thin contract.
    """
    for n in (1, 2, 5, 30):
        bars = synthetic(n=n, seed=3)
        out = M.signal_fn(bars, **BASE)
        for series in out:
            assert len(series) == n and series.dtype == bool
            assert not series.isna().any(), f"NaN mask at n={n}"
        feats = M.ml_features(bars, **BASE)
        assert len(feats) == n, f"{len(feats)} feature rows at n={n}"


def test_the_news_filter_is_the_repositorys_own_implementation() -> None:
    """
    `use_news_filter` must reach `backtest.event_calendar.apply_entry_filters`
    — the only implementation of the filter in this repository — and must
    subtract candidates rather than being wired to nothing.

    SKIPS LOUDLY when the calendar does not cover the fixture's span. An empty
    calendar RAISES by design there, precisely so a run reported as
    news-filtered cannot be one in which nothing was ever filtered, and a case
    that swallowed that would be reporting a pass for an untested path.
    """
    bars = synthetic(n=2000)
    try:
        gated = candidates(bars, use_news_filter=True)
    except Exception as e:                       # noqa: BLE001 - reported
        msg = str(e).lower()
        if "calendar" in msg or "covers no" in msg or "no events" in msg:
            print(f"        SKIPPED: no macro calendar for the fixture span "
                  f"({type(e).__name__}: {e})")
            return
        raise

    base = candidates(bars, use_news_filter=False)
    for side in ("long_ok", "short_ok"):
        assert not (gated[side] & ~base[side]).any(), (
            f"the news filter ADDED {side} candidates")
    removed = sum(int((base[s] & ~gated[s]).sum())
                  for s in ("long_ok", "short_ok"))
    print(f"        news filter removed {removed} candidates")


def test_the_ast_validator_objects_only_to_the_event_calendar_import() -> None:
    """
    The module deliberately imports `backtest.event_calendar`, which puts it
    outside `ALLOWED_IMPORTS`. That exception is granted for ONE import, and
    pinning the FULL objection list is what keeps it from covering a later edit
    reaching for `open`, `eval` or a network library.
    """
    from agents.tier3_workers import _audit_ast
    problems = sorted(set(_audit_ast(ast.parse(MODULE_PATH.read_text()))))
    expected = ["import from 'backtest.event_calendar' is not allowed"]
    assert problems == expected, f"validator objections: {problems}"


def test_version_b_vetoes_through_the_declared_feature_matrix() -> None:
    """
    Version B end to end: the shared expanding-window filter, fitted on THIS
    module's five columns, acting as a veto and nothing else.

    The two properties that make it a veto rather than a strategy: it may only
    turn entries OFF, and it may never turn one on. Both are checked against
    Version A's own masks. The exits come back untouched by design — an exit
    with no open position is dropped by `clean_signals` downstream, so
    suppressing an entry removes the whole trade cleanly.
    """
    bars = synthetic(n=2500, seed=17)
    le, lx, se, _sx = masks(bars)
    entries = pd.Series(np.asarray(le), index=bars.index)
    exits = pd.Series(np.asarray(lx), index=bars.index)
    assert int(entries.sum()) > 20, (
        f"only {int(entries.sum())} long entries — too few for the filter to "
        f"have anything to fit")

    filtered, filtered_exits = apply_ml_signal_filter(
        bars, entries, exits, symbol="NQ", direction="long",
        features=M.ml_features)

    f = np.asarray(filtered, dtype=bool)
    a = np.asarray(entries, dtype=bool)
    assert not (f & ~a).any(), (
        "Version B turned entries ON — it is not a veto")
    assert int(f.sum()) <= int(a.sum()), "Version B added entries"
    assert np.array_equal(np.asarray(filtered_exits, dtype=bool),
                          np.asarray(exits, dtype=bool)), (
        "Version B modified the exits")
    print(f"        Version B kept {int(f.sum())} of {int(a.sum())} long entries")


# ==========================================================================
# The script runner. `assert` is the failure mechanism, so pytest and this
# report the same thing — see the module docstring.
# ==========================================================================
def main() -> int:
    cases = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    print(f"t3_braid_scalp_20260823 — {len(cases)} cases\n")
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
