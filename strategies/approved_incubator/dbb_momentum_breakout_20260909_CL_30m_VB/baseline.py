"""
dbb_momentum_breakout_20260909.py - the Double Bollinger Band breakout.

Location:  ~/src/trading/strategies/experimental/dbb_momentum_breakout_20260909.py

Two Bollinger envelopes around one 20-period SMA. The INNER pair at 0.5 SD
splits the tape into three states - a neutral middle where nothing is traded,
a buy zone above it and a sell zone below - and the OUTER pair at 3.0 SD marks
where a move has gone far enough to be exhaustion rather than momentum. An
entry is the bar price LEAVES the neutral zone; the position is held while the
zone holds and closed when price comes back.

================================================================================
STRATEGY SPECIFICATION & BACKTEST REQUEST  (filled from the 2026-09-09 request)
================================================================================
1. STRATEGY METADATA
   Strategy Name:       dbb_momentum_breakout
   Strategy Archetype:  Momentum / breakout
   Primary Timeframe:   5m  (TIMEFRAME; the ladder is TIMEFRAMES)
   Target Assets:       the 19-contract matrix (SYMBOLS)

2. CORE CONCEPT & HYPOTHESIS
   A move that clears half a standard deviation from its own 20-bar mean has
   stopped being noise around that mean and started being a direction. The
   inefficiency harvested is the tail of that transition: participants who
   must react to a move already underway. It persists because the reaction is
   structural - stops, hedges and mandates - rather than a view. The neutral
   zone is where the mean still explains the tape and nothing is traded; the
   outer band is where the move has already paid whoever was going to be paid.

3. INDICATORS & PARAMETER GRID
   SMA({sma_window}) - the baseline. A SIMPLE mean, per the request; the
       neighbouring modules' `_ema` is a different, faster average and would
       draw a different baseline.
   Bollinger inner: SMA +/- {inner_sd} x rolling SD({sma_window})
   Bollinger outer: SMA +/- {outer_sd} x rolling SD({sma_window})
   ATR(14), Wilder - the protective stop only, never a band.

4. ENTRY & EXIT EXECUTION RULES
   Long:   Close crosses ABOVE the +{inner_sd} SD band (enters the buy zone).
   Short:  Close crosses BELOW the -{inner_sd} SD band (enters the sell zone).
   Filter: no entry while price sits inside the neutral zone, and - see the
           note below - no entry beyond the outer band.
   Exit:   price returns to the neutral zone, or closes back through the
           SMA baseline; whichever comes first. A protective ATR stop at
           {sl_atr_mult} x ATR(14) sits underneath both.
   Session Rules:   none modelled - NOT SPECIFIED in the request.
   Execution Fill:  next-bar open, engine-applied slippage and commission.

5. BACKTEST EXECUTION CONTROLS
   Repository defaults: in-sample 2013-01-01..2022-12-31, holdout 2023-01-01
   onward and untouched during optimisation, Version B evaluated against A.

THREE PLACES THE REQUEST LEFT A DECISION, MADE HERE AND FLAGGED
---------------------------------------------------------------
**The outer 3.0 SD band was declared but no rule used it.** Written literally
it would be drawn on the chart and touch nothing. The Double Bollinger system
it is named after defines the buy zone as BOUNDED - between the inner and
outer bands - and treats price beyond the outer band as overextended, so
`use_outer_guard` suppresses entries there and is ON by default. It is a real
change to the signal, so it is a parameter and it is in PARAM_GRID: the sweep
answers whether it binds rather than the question being settled here.

**"Suppress new entries when price is between -0.5 and +0.5 SD" is implied by
the entry itself.** A close that has just crossed the +0.5 band is by
definition outside the neutral zone, so the filter cannot veto a cross. It is
implemented anyway, as an explicit gate, because it is what makes the entry
STATE-checked as well as event-checked - and because a later variant that
entered on a state rather than a cross would need it and would otherwise
silently lack it. `use_neutral_filter` is declared so this is visible rather
than dead code nobody can account for.

**"Trailing stop at the 20-SMA baseline" is not the engine's `trailing`.**
That flag means a trailing ATR stop inside the position walk. The baseline
exit here is a SIGNAL exit - price closing back through the SMA - so
`trailing` is False and the SMA exit is carried in the signal masks. Naming
them the same thing is exactly how a stop distance ends up meaning two things,
so both are spelled out: `sl_atr_mult` is the protective ATR stop and nothing
else, and it is what `promote.py` writes into the risk block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

#: The bar the DEFAULTS are set for, and the one the engine actually uses.
#: SINGULAR AND DECLARED: `agents.tier3_workers` reads
#: `getattr(module, "TIMEFRAME", None)` and its caller resolves
#: `explicit or info["timeframe"] or DEFAULT_TIMEFRAME`, where
#: DEFAULT_TIMEFRAME is "1d". Dropping this in favour of the plural below
#: would not raise - it would silently run a five-minute breakout on DAILY
#: bars. Nothing in this repository reads a `TIMEFRAMES`.
TIMEFRAME = "5m"

#: The ladder Stage 1 is meant to screen across. DESCRIPTIVE ONLY - no loader
#: or pipeline stage reads this name; a timeframe is selected per run with
#: `--timeframe`. Written down so the intended ladder is recoverable from the
#: module rather than from whoever typed the last command.
#:
#: 1m is absent deliberately: `intraday_start_year` exists because pre-2013
#: 1-minute data is sparse for ten symbols, and a 20-bar mean on those bars
#: describes the gaps as much as the tape.
TIMEFRAMES = ["5m", "15m", "30m", "1h"]

#: The 19-contract matrix. All nineteen reconcile against the Databento
#: definitions - `python -m backtest.specs` reports UNVERIFIED for M2K and MYM
#: alone, and neither is here.
#:
#: A DECLARATION OF WHERE THE PREMISE MIGHT HOLD, NOT A CLAIM THAT IT DOES.
#: Cross-sectional by default: each contract is its own simulation on its own
#: multiplier, tick size and commission, and symbols are never blended into
#: one equity curve. Two carried caveats from CLAUDE.md, neither caught by the
#: spec reconciler because it checks the CONTRACT rather than the bars: SI
#: definitions stop at 2016 and CL at 2025-12.
SYMBOLS = ["ES", "NQ", "YM", "RTY", "CL", "NG", "RB", "HO", "GC", "SI", "PL",
           "6E", "6B", "6J", "6S", "ZB", "ZN", "BTC", "ETH"]

STRATEGY_NAME = "DBB_MOMENTUM_BREAKOUT"
STRATEGY_MODULE = "dbb_momentum_breakout_20260909"

PORTFOLIO_GROUP = "Momentum_Breakout"
CORRELATION_PROFILE = (
    "Spans every group in the matrix, so a basket drawn from it is only as "
    "diversified as the contracts chosen. The index names are one factor at "
    "four capitalisations and the energy names are another; routing sizes "
    "against that, not against the symbol count.")

#: The regime the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and nothing here reads these. Quadrant ids
#: are `mdlib/regimes.py`'s: Q1 High-Vol/Trending, Q2 High-Vol/Ranging,
#: Q3 Low-Vol/Trending, Q4 Low-Vol/Ranging, 0 UNDEFINED (warm-up, not a
#: quadrant). A breakout nominates the TRENDING pair - in a ranging regime a
#: band exit is the mean reasserting itself, which is what Gate R settles.
TARGET_REGIMES = ("High Volatility / Trending", "Low Volatility / Trending")
TARGET_QUADRANTS = ("Q1", "Q3")

#: Wilder's ATR period, for the protective stop only. Fixed at 14 rather than
#: exposed: it never enters a band, so sweeping it would move only the stop
#: and confound the band study it sits under.
ATR_PERIOD = 14

#: Population standard deviation, `ddof=0`, and this is not a detail. Every
#: charting package draws Bollinger Bands on the population SD; pandas'
#: `.std()` defaults to the SAMPLE SD (`ddof=1`), which over a 20-bar window
#: is about 2.6% wider. A band drawn 2.6% wider than the one a reader checks
#: against their own chart is a different band, and every entry near it fires
#: on a different bar.
BB_DDOF = 0

#: A stop nearer than this to the fill sits inside the bar's own noise and the
#: run measures the cost model rather than the breakout.
MIN_STOP_ATR_MULT = 0.25


DEFAULT_PARAMS = {
    "sma_window": 20,
    "inner_sd": 0.5,
    "outer_sd": 3.0,
    "use_neutral_filter": True,
    "use_outer_guard": True,
    "use_baseline_exit": True,
    "sl_atr_mult": 2.0,
    "tp_atr_mult": None,
    "trailing": False,
}

#: The search space `backtest/run.py --scan` sweeps.
#:
#:     3 x 3 x 2 x 2 = 36 combinations
#:
#: Coarse on purpose. `variants_tested` travels with every result precisely so
#: a reader can see how large the search was that produced a Sharpe.
#:
#: `use_outer_guard` is swept because the request declared the outer band
#: without saying what it does - the sweep answers whether bounding the buy
#: zone earns its place instead of that being decided by whoever wrote the
#: module.
#:
#: `tp_atr_mult` is pinned None and `trailing` False by the premise: a
#: breakout is exited when the move ends, which is what the baseline and
#: neutral-zone exits detect. A fixed target would cap the move the strategy
#: exists to hold, and the engine's ATR trail would compete with the SMA that
#: is already doing that job. Both are still declared, so one out-of-band run
#: can ask the question the grid does not.
PARAM_GRID = {
    "sma_window": [10, 20, 50],
    "inner_sd": [0.5, 1.0, 1.5],
    "sl_atr_mult": [1.5, 2.0],
    "use_outer_guard": [True, False],
}

LOGIC = {
    "concept": (
        "Two Bollinger envelopes around one SMA({sma_window}). A move that "
        "clears {inner_sd} SD from its own 20-bar mean has stopped being "
        "noise around that mean and started being a direction; the "
        "participants who must react to it - stops, hedges, mandates - are "
        "the edge. The neutral zone inside +/-{inner_sd} SD is where the mean "
        "still explains the tape and nothing is traded. Trades are attributed "
        "by their ENTRY bar, never the exit, so a breakout opened in one "
        "quadrant that runs into another still scores where it opened."),
    "entry": (
        "LONG when Close crosses ABOVE SMA({sma_window}) + {inner_sd} x SD, "
        "entering the buy zone. SHORT when Close crosses BELOW "
        "SMA({sma_window}) - {inner_sd} x SD. With use_outer_guard the zone "
        "is BOUNDED at {outer_sd} SD and a close beyond it is treated as "
        "overextended rather than as a signal."),
    "exit": (
        "Price returning to the neutral zone (back inside +/-{inner_sd} SD), "
        "or Close crossing back through the SMA({sma_window}) baseline with "
        "use_baseline_exit - whichever the bar reaches first. A protective "
        "stop at {sl_atr_mult} x ATR(14) sits underneath both. No take-profit "
        "is modelled (tp_atr_mult={tp_atr_mult}) and the engine's ATR trail "
        "is off (trailing={trailing}): the SMA is the trailing exit here, and "
        "it is a SIGNAL rather than a bracket."),
}


# ---------------------------------------------------------------------------
# Indicator kernels
#
# `_wilder`, `_true_range`, `_atr`, `_as_series`, `_cross_above`,
# `_cross_below` and `_walk_loop` below are DUPLICATED VERBATIM
# from `ema_deviation_scalp_20260909.py`, which shares them with the rest of
# this directory. Strategy modules are loaded from a FILE PATH by
# `agents.tier3_workers.load_strategy` and promoted as a self-contained copy,
# so a shared import would resolve against whatever happens to sit beside the
# module at load time - and a promoted package must reproduce the file that
# was certified, byte for byte. `tests/test_risk_params.py` holds the copies
# to identical output; do not "improve" one alone.
# ---------------------------------------------------------------------------
def _sma(series: pd.Series, period: int) -> pd.Series:
    """
    A SIMPLE moving average - the request's baseline, and NOT `_ema`.

    Every neighbouring module in this directory anchors on an exponential
    mean. A span-20 EMA and a 20-bar SMA are different curves: the EMA weights
    the newest bar about 9.5% and the SMA weights every bar 5%, so the EMA
    turns sooner and the bands drawn around it sit somewhere else. Using the
    wrong one would produce a "20 SMA" no chart agrees with, and every
    band-cross entry would fire on a different bar.

    NaN until `period` bars exist: `min_periods=period`. A mean over three
    bars is not a 20-bar mean, and admitting it would trade the warm-up.
    """
    return series.astype("float64").rolling(int(period),
                                            min_periods=int(period)).mean()


def _bollinger(close: pd.Series, period: int, inner_sd: float,
               outer_sd: float) -> dict[str, pd.Series]:
    """
    The two envelopes around one baseline: `(mid, inner_upper, inner_lower,
    outer_upper, outer_lower, sd)`.

    ONE `sd` SERIES FEEDS BOTH PAIRS. Computing them separately would let a
    future edit change the window on one and not the other, and the two
    envelopes would then describe different dispersions of the same tape -
    invisible on a chart, because both would still look like bands.

    `ddof=BB_DDOF` (0, the population SD) rather than pandas' default sample
    SD. See BB_DDOF: over a 20-bar window the sample form is ~2.6% wider, and
    a band 2.6% from where a reader's chart draws it fires on different bars.

    CAUSAL: `.rolling()` looks strictly backwards.
    """
    close = close.astype("float64")
    mid = _sma(close, period)
    sd = close.rolling(int(period), min_periods=int(period)).std(ddof=BB_DDOF)
    return {
        "mid": mid, "sd": sd,
        "inner_upper": mid + float(inner_sd) * sd,
        "inner_lower": mid - float(inner_sd) * sd,
        "outer_upper": mid + float(outer_sd) * sd,
        "outer_lower": mid - float(outer_sd) * sd,
    }


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing - the average the ATR is actually defined on.

    `alpha = 1/period` with `adjust=False` is Wilder's recursion exactly. A
    span-`period` EMA (`_ema` above) is a different, roughly twice as fast,
    average, and using it here would produce an "ATR(14)" that no other tool
    agrees with - so a reader checking a stop distance against their own chart
    would see it somewhere else.
    """
    return series.ewm(alpha=1.0 / period, adjust=False,
                      min_periods=period).mean()


def _true_range(bars: pd.DataFrame) -> pd.Series:
    """
    True range per bar.

    `.shift(1)` looks one bar BACKWARD, which is the correct direction: the
    true range at bar i uses bar i-1's close and nothing later.
    """
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev = close.shift(1)
    return pd.concat([(high - low).abs(),
                      (high - prev).abs(),
                      (low - prev).abs()], axis=1).max(axis=1)


def _atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    True range, Wilder-smoothed. NaN until `period` bars exist.

    A ZERO ATR IS A MEASUREMENT, NOT A GAP, and it is left as the 0.0 it is.
    A window in which nothing moved - a halted contract, a dead overnight
    hour - has a true ATR of zero. `signal_fn` refuses to enter on such a bar
    rather than placing a bracket of zero width around the fill.
    """
    return _wilder(_true_range(bars), period)


def _as_series(value, index) -> pd.Series:
    """A scalar level as a Series on `index`; a Series is returned as-is."""
    if isinstance(value, pd.Series):
        return value
    return pd.Series(float(value), index=index)


def _cross_above(fast: pd.Series, level) -> pd.Series:
    """
    True on the bar `fast` crosses UP through `level` - a series or a scalar.

        fast[i] > level[i]  AND  fast[i-1] <= level[i-1]

    A STATE (`fast > level`) would be true for every bar of a run; this is the
    EVENT, true once. `<=` on the previous bar rather than `<` so a pair that
    was exactly equal and then separated counts as a cross - with two
    oscillators quantised by the same price series, exact equality is not the
    measure-zero event it is for two continuous curves.

    NaN-SAFE BY CONSTRUCTION: every comparison against NaN is False, so a
    warm-up bar is never a cross. `.shift(1)` looks one bar BACKWARD.
    """
    other = _as_series(level, fast.index)
    return (fast > other) & (fast.shift(1) <= other.shift(1))


def _cross_below(fast: pd.Series, level) -> pd.Series:
    """
    True on the bar `fast` crosses DOWN through `level`. The exact mirror of
    `_cross_above`, written out rather than expressed as its negation:
    `~_cross_above` is true on every bar that merely fails to be a cross up,
    which is almost all of them.
    """
    other = _as_series(level, fast.index)
    return (fast < other) & (fast.shift(1) >= other.shift(1))


def _stochastic(bars: pd.DataFrame, k_period: int, smooth_k: int,
                d_period: int) -> tuple[pd.Series, pd.Series]:
    """
    The Stochastic oscillator, returned as `(%K, %D)`.

        raw   = 100 * (close - lowest_low(k_period))
                    / (highest_high(k_period) - lowest_low(k_period))
        %K    = SMA(raw, smooth_k)          <- the "slow" smoothing
        %D    = SMA(%K,  d_period)

    This is the SLOW Stochastic when `smooth_k > 1` and the fast one when it
    is 1, which is why `smooth_k` is a parameter rather than a constant: the
    request named "%K=8, %D=3, smooth=3" in one place and `k_period: 14,
    d_period: 3, smooth_k: 3` in another, and both are reachable from here.

    THE DEAD-FLAT WINDOW READS 50, NOT 0. When `highest == lowest` the ratio
    is 0/0 and the oscillator is undefined; `STOCH_NEUTRAL` resolves it to the
    centerline, because a window in which nothing moved is neither overbought
    nor oversold. The naive 0.0 is the most oversold value the scale has and
    would hold `%K <= oversold` true through every halted session, so a dead
    tape would manufacture long confirmations. The fill is applied ONLY where
    the rolling extremes exist, so warm-up stays NaN and is never a signal.

    CAUSAL: `.rolling()` and `.shift(1)` look strictly backwards. Nothing here
    reads bar i+1.
    """
    high, low, close = bars["high"], bars["low"], bars["close"]
    kp, sk, dp = int(k_period), int(smooth_k), int(d_period)
    lowest = low.rolling(kp, min_periods=kp).min()
    highest = high.rolling(kp, min_periods=kp).max()
    span = highest - lowest
    # `.where(span > 0)` makes the flat window NaN rather than 0/0; the
    # explicit fill then puts the neutral 50 in only where the extremes are
    # real. A `span == 0` row inside the warm-up stays NaN either way.
    raw = 100.0 * (close.astype("float64") - lowest) / span.where(span > 0)
    raw = raw.where(span.isna() | (span > 0), STOCH_NEUTRAL)
    percent_k = raw.rolling(sk, min_periods=sk).mean()
    percent_d = percent_k.rolling(dp, min_periods=dp).mean()
    return percent_k, percent_d


# --------------------------------------------------------------------------
# The position walk
# --------------------------------------------------------------------------
def _walk_loop(long_entry_ok: np.ndarray,
               short_entry_ok: np.ndarray,
               long_sig_exit: np.ndarray,
               short_sig_exit: np.ndarray,
               open_: np.ndarray,
               high: np.ndarray,
               low: np.ndarray,
               atr: np.ndarray,
               flat_bar: np.ndarray,
               sl_mult: float,
               tp_mult: float,
               trailing: bool) -> tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    """
    THREE-state machine over the bars: flat, long, or short - each side under
    its own stop, target and signal exit.

    DUPLICATED VERBATIM from `keltner_trend_drift_20260901.py`, which shares
    it with `sma_momentum_crossover_20260818.py`, `t3_braid_scalp_20260823.py`
    and `ema_crossover_20260821.py`, by the same convention that duplicates
    `_wilder` and `_atr` across this directory. `tests/test_risk_params.py`
    runs the copies on identical arrays in both directions and requires
    identical output - do not "improve" one alone.

    The timeline matches the engine's. An entry signal on bar i is filled at
    bar i+1's open, so the position is live from bar i+1, the fill price is
    `open_[i + 1]`, and the extreme-price mark starts there - NOT on the
    signal bar. Both distances are frozen at `mult * ATR` as measured on the
    SIGNAL bar and never re-measured as volatility changes.

    The two sides, written out rather than folded into a sign flip, because a
    reader has to be able to check them against the specification by eye:

        LONG    stop     fill - dist    trailing: (highest high since fill) - dist
                target   fill + dist
                exits    low  <= stop   or  high >= target
        SHORT   stop     fill + dist    trailing: (lowest low   since fill) + dist
                target   fill - dist
                exits    high >= stop   or  low  <= target

    A short stop placed BELOW the fill would be breached by the fill bar
    itself; this is the reason the sides are not a sign flip.

    `tp_mult` is NaN when no take-profit is modelled, rather than a sentinel
    like 0 or a huge number. NaN propagates into `target` and every comparison
    against it is False, so the target simply never fires on either side.

    A bar carrying BOTH a long and a short entry signal while flat takes
    NEITHER, matching `backtest.engine._clean_signals_ls_loop`. A well-formed
    strategy cannot produce it - price cannot cross above and below the same
    EMA on one bar - so the branch exists to make a malformed one visible as a
    missing trade rather than a plausible one-sided curve.

    A position is never reversed directly. The walk enters only from flat, so
    a short signal arriving while long is ignored; the long must exit first.

    Exits are checked from the fill bar onward, never on the signal bar.
    """
    n = long_entry_ok.shape[0]
    long_entries = np.zeros(n, dtype=np.bool_)
    long_exits = np.zeros(n, dtype=np.bool_)
    short_entries = np.zeros(n, dtype=np.bool_)
    short_exits = np.zeros(n, dtype=np.bool_)
    stop_level = np.full(n, np.nan)
    tp_level = np.full(n, np.nan)

    state = 0                            # 0 flat, 1 long, -1 short
    fill_i = 0
    stop_dist = 0.0
    tp_dist = np.nan
    entry_px = np.nan
    hw = 0.0                             # highest high since the fill (long)
    lw = 0.0                             # lowest low since the fill (short)

    for i in range(n):
        if state == 0:
            go_long = long_entry_ok[i]
            go_short = short_entry_ok[i]
            if go_long and go_short:
                continue                 # ambiguous bar: take neither side
            if go_long or go_short:
                fill_i = i + 1
                stop_dist = sl_mult * atr[i]
                # NaN in, NaN out: no take-profit stays no take-profit.
                tp_dist = tp_mult * atr[i]
                entry_px = np.nan
                hw = -np.inf
                lw = np.inf
                if go_long:
                    long_entries[i] = True
                    state = 1
                else:
                    short_entries[i] = True
                    state = -1
            continue

        if i < fill_i:
            continue

        if i == fill_i:
            # The engine's own fill: the open of the bar AFTER the signal. Not
            # the signal bar's close, which the position never traded at.
            entry_px = open_[i]

        if state == 1:
            if trailing:
                if high[i] > hw:
                    hw = high[i]
                level = hw - stop_dist
            else:
                level = entry_px - stop_dist

            target = entry_px + tp_dist   # NaN when no target is modelled
            stop_level[i] = level
            tp_level[i] = target

            hit_stop = low[i] <= level
            # False whenever `target` is NaN, which is how "no take-profit" is
            # expressed. Every comparison against NaN is False in both numba
            # and numpy, so the interpreted fallback cannot disagree with the
            # compiled loop about it.
            hit_tp = high[i] >= target

            # A bar that breaches both produces one exit on this bar either
            # way, and the engine fills it at the next bar's open regardless -
            # so the intrabar race between the stop and the target is not
            # resolved here because it cannot change the result.
            if hit_stop or hit_tp or long_sig_exit[i] or flat_bar[i]:
                long_exits[i] = True
                state = 0
        else:
            if trailing:
                if low[i] < lw:
                    lw = low[i]
                level = lw + stop_dist
            else:
                level = entry_px + stop_dist

            target = entry_px - tp_dist   # NaN when no target is modelled
            stop_level[i] = level
            tp_level[i] = target

            hit_stop = high[i] >= level
            hit_tp = low[i] <= target     # False when target is NaN

            if hit_stop or hit_tp or short_sig_exit[i] or flat_bar[i]:
                short_exits[i] = True
                state = 0

    return (long_entries, long_exits, short_entries, short_exits,
            stop_level, tp_level)


try:                                    # pragma: no cover - env dependent
    from numba import njit

    # No `cache=True`. `load_strategy` imports this module from a file path,
    # so it is not importable by name. Compiling still works and numba writes
    # the cache; it is the LOAD in a later process that fails, with
    # `ModuleNotFoundError: No module named '<dynamic>'`, which is why the
    # first run after a cache wipe looks clean and the second one dies.
    _walk = njit(nogil=True)(_walk_loop)
except ImportError:                     # pragma: no cover - env dependent
    # Same function, interpreted. `backtest.engine.clean_signals` degrades the
    # same way, and the fallback has to exist because a missing compiler must
    # not change which trades a strategy takes.
    _walk = _walk_loop


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _validate(sma_window: int, inner_sd: float, outer_sd: float,
              sl_atr_mult: float, tp_atr_mult: float | None,
              trailing: bool) -> None:
    """
    Refuse a parameter set that cannot mean what it says, where it is passed
    rather than as a strange equity curve. Each has a specific failure behind
    it:

    * `sma_window < 2` makes the rolling SD undefined or zero-width, and every
      band collapses onto the baseline - so every bar is simultaneously a
      cross of both.
    * `inner_sd <= 0` erases the neutral zone: the buy zone starts AT the mean
      and the strategy fires on every bar that drifts to the wrong side.
    * `outer_sd <= inner_sd` inverts the buy zone - the overextension guard
      would then veto the entire zone it is supposed to bound, and
      `use_outer_guard` would silently mean "trade nothing" rather than
      "trade less".
    * A stop inside MIN_STOP_ATR_MULT sits inside the bar's own noise.
    * `tp_atr_mult` may be None - that IS a setting, meaning "no target" - but
      a non-positive number would put the target at or behind the fill and
      close every trade on its first bar.
    """
    if int(sma_window) < 2:
        raise ValueError(
            f"sma_window must be >= 2; got {sma_window!r}. Below that the "
            f"rolling standard deviation has no width and every band "
            f"collapses onto the baseline.")
    if float(inner_sd) <= 0.0:
        raise ValueError(
            f"inner_sd must be > 0; got {inner_sd!r}. At zero there is no "
            f"neutral zone and the buy zone starts at the mean itself.")
    if float(outer_sd) <= float(inner_sd):
        raise ValueError(
            f"outer_sd must be > inner_sd; got outer_sd={outer_sd!r} and "
            f"inner_sd={inner_sd!r}. Inverted, the overextension guard vetoes "
            f"the whole buy zone and use_outer_guard means 'trade nothing'.")
    if float(sl_atr_mult) < MIN_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be >= {MIN_STOP_ATR_MULT}; got "
            f"{sl_atr_mult!r}. A tighter stop sits inside the bar's own noise "
            f"and the run measures the cost model.")
    if tp_atr_mult is not None and float(tp_atr_mult) <= 0.0:
        raise ValueError(
            f"tp_atr_mult must be > 0 or None (no target); got "
            f"{tp_atr_mult!r}. At or below zero the target sits at or behind "
            f"the fill and every trade closes on its first bar.")
    if not isinstance(trailing, (bool, np.bool_)):
        raise ValueError(f"trailing must be a bool; got {trailing!r}")


def _tp_distance(tp_atr_mult: float | None) -> float:
    """
    The take-profit multiple as the walk wants it: NaN for "no target".

    NaN rather than 0 or a large sentinel, because every comparison against
    NaN is False in both the compiled and the interpreted loop - so the target
    never fires and the drawn line is a visible gap.
    """
    if tp_atr_mult is None:
        return float("nan")
    return float(tp_atr_mult)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
def _layers(bars: pd.DataFrame, sma_window: int, inner_sd: float,
            outer_sd: float, use_neutral_filter: bool, use_outer_guard: bool,
            use_baseline_exit: bool) -> dict[str, pd.Series]:
    """
    Every series both `signal_fn` and `indicators` read, computed ONCE.

    One call site for the maths is the point: a second implementation in the
    report would be free to disagree with this one and draw a band cross a bar
    from where the trade fired, with nothing raising.

    The toggles live here rather than in `signal_fn` so a disabled filter is a
    Series of True with the underlying series still computed and still drawn -
    a reader can see what the filter WOULD have said on a run where it was off.
    """
    close = bars["close"].astype("float64")
    B = _bollinger(close, sma_window, inner_sd, outer_sd)
    atr = _atr(bars, ATR_PERIOD)

    # THE THREE STATES. `neutral` is the middle band where the mean still
    # explains the tape; the two zones are outside it. Written as explicit
    # comparisons rather than as `~neutral` split by sign, so a reader can
    # check each against the specification by eye.
    in_neutral = (close <= B["inner_upper"]) & (close >= B["inner_lower"])
    in_buy_zone = close > B["inner_upper"]
    in_sell_zone = close < B["inner_lower"]

    # THE ENTRY IS A CROSS, NOT A STATE. A run of bars already above the inner
    # band is not a breakout - the breakout was the bar that got there - and a
    # state test would fire an entry on every one of them.
    cross_up = _cross_above(close, B["inner_upper"])
    cross_down = _cross_below(close, B["inner_lower"])

    # The neutral filter. Redundant against a CROSS by construction - a bar
    # that just crossed the inner band is outside the zone by definition - and
    # implemented anyway so the entry is state-checked as well as
    # event-checked. See the module docstring.
    not_neutral = (~in_neutral) if use_neutral_filter else pd.Series(
        True, index=bars.index)

    # The overextension guard: the buy zone is BOUNDED by the outer band. A
    # close beyond 3.0 SD has already paid whoever was going to be paid.
    if use_outer_guard:
        guard_long = close <= B["outer_upper"]
        guard_short = close >= B["outer_lower"]
    else:
        guard_long = pd.Series(True, index=bars.index)
        guard_short = pd.Series(True, index=bars.index)

    trigger_long = cross_up & not_neutral & guard_long
    trigger_short = cross_down & not_neutral & guard_short

    # THE EXITS. Returning to the neutral zone is the tighter of the two and
    # fires first by construction: the zone is +/-inner_sd around the mean, so
    # price is back inside it BEFORE it reaches the baseline. The baseline
    # cross is kept as the second, looser condition because a gap can carry
    # price through the zone in one bar without ever closing inside it.
    exit_long = in_neutral.copy()
    exit_short = in_neutral.copy()
    if use_baseline_exit:
        exit_long = exit_long | _cross_below(close, B["mid"])
        exit_short = exit_short | _cross_above(close, B["mid"])

    return {
        "close": close, "mid": B["mid"], "sd": B["sd"], "atr": atr,
        "inner_upper": B["inner_upper"], "inner_lower": B["inner_lower"],
        "outer_upper": B["outer_upper"], "outer_lower": B["outer_lower"],
        "in_neutral": in_neutral, "in_buy_zone": in_buy_zone,
        "in_sell_zone": in_sell_zone,
        "cross_up": cross_up, "cross_down": cross_down,
        "trigger_long": trigger_long, "trigger_short": trigger_short,
        "exit_long": exit_long, "exit_short": exit_short,
        "sd_from_mid": (close - B["mid"]) / B["sd"].replace(0.0, np.nan),
    }


# --------------------------------------------------------------------------
# The contract functions
# --------------------------------------------------------------------------
def signal_fn(bars: pd.DataFrame,
              sma_window: int = 20,
              inner_sd: float = 0.5,
              outer_sd: float = 3.0,
              use_neutral_filter: bool = True,
              use_outer_guard: bool = True,
              use_baseline_exit: bool = True,
              sl_atr_mult: float = 2.0,
              tp_atr_mult: float | None = None,
              trailing: bool = False) -> tuple[pd.Series, pd.Series,
                                               pd.Series, pd.Series]:
    """
    Returns the FOUR-MASK form: (long_entries, long_exits, short_entries,
    short_exits), all boolean Series on `bars.index`.

    A three-tuple or a bare Series RAISES in `engine.unpack_signals` rather
    than silently losing the short side into a plausible long-only curve.

    THE SHORT SIDE IS NOT THE LONG SIDE WITH A SIGN FLIP. Inside `_walk_loop`
    the short stop sits ABOVE the fill; placed below it, the fill bar itself
    would breach it. The masks are written out per side for the same reason.

    `trailing` is the engine's ATR trail and is OFF. The request's "trailing
    stop at the 20-SMA baseline" is a SIGNAL exit and travels in the exit
    masks; `sl_atr_mult` is the protective bracket underneath it and nothing
    else.
    """
    _validate(sma_window, inner_sd, outer_sd, sl_atr_mult, tp_atr_mult,
              trailing)

    L = _layers(bars, int(sma_window), float(inner_sd), float(outer_sd),
                bool(use_neutral_filter), bool(use_outer_guard),
                bool(use_baseline_exit))

    long_ok = L["trigger_long"].fillna(False).to_numpy()
    short_ok = L["trigger_short"].fillna(False).to_numpy()
    long_sig_exit = L["exit_long"].fillna(False).to_numpy()
    short_sig_exit = L["exit_short"].fillna(False).to_numpy()

    atr = L["atr"].to_numpy(dtype="float64")
    # A bar with no ATR yet is not tradable: the protective bracket has no
    # width, and admitting it would place a stop at the fill price.
    warm = ~np.isfinite(atr) | (atr <= 0.0)
    # The bands warm up too, and on a longer window than the ATR: a NaN band
    # makes every cross False anyway, but the SD is also NaN through the
    # warm-up and an entry there would be sized against a dispersion nobody
    # measured.
    warm = warm | ~np.isfinite(L["sd"].to_numpy(dtype="float64"))
    long_ok = long_ok & ~warm
    short_ok = short_ok & ~warm

    flat_bar = np.zeros(len(bars), dtype=bool)

    le, lx, se, sx, _, _ = _walk(
        long_ok, short_ok, long_sig_exit, short_sig_exit,
        bars["open"].to_numpy(dtype="float64"),
        bars["high"].to_numpy(dtype="float64"),
        bars["low"].to_numpy(dtype="float64"),
        np.nan_to_num(atr, nan=0.0), flat_bar, float(sl_atr_mult),
        _tp_distance(tp_atr_mult), bool(trailing))

    idx = bars.index
    return (pd.Series(le, index=idx), pd.Series(lx, index=idx),
            pd.Series(se, index=idx), pd.Series(sx, index=idx))


def generate_signals(df: pd.DataFrame, params: dict | None = None
                     ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    The `generate_signals(df, params)` calling convention, as a thin adapter.

    NOT THE ENGINE'S ENTRY POINT. `agents.tier3_workers.load_strategy` binds
    `make_signal_fn(**params)` or `signal_fn(bars, **params)` and knows
    nothing about this name. It delegates rather than reimplementing, so the
    two can never disagree about what a signal is.
    """
    bound = {**DEFAULT_PARAMS, **(params or {})}
    unknown = set(bound) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    return signal_fn(df, **bound)


def indicators(bars: pd.DataFrame, **params) -> dict[str, pd.Series]:
    """
    Full-length series drawn over the trade inspector's candles, from the same
    `_layers` call `signal_fn` uses - so the chart cannot draw a band cross a
    bar from where the entry actually happened.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["sma_window"]), float(p["inner_sd"]),
                float(p["outer_sd"]), bool(p["use_neutral_filter"]),
                bool(p["use_outer_guard"]), bool(p["use_baseline_exit"]))
    inner, outer = float(p["inner_sd"]), float(p["outer_sd"])
    return {
        f"SMA({int(p['sma_window'])})": L["mid"],
        f"+{inner} SD": L["inner_upper"],
        f"-{inner} SD": L["inner_lower"],
        f"+{outer} SD": L["outer_upper"],
        f"-{outer} SD": L["outer_lower"],
        f"ATR({ATR_PERIOD})": L["atr"],
    }


def ml_features(bars: pd.DataFrame, **params) -> pd.DataFrame:
    """
    The matrix Version B's classifier is fitted on: one row per bar, in order.

    CAUSALITY IS THIS MODULE'S RESPONSIBILITY. Every column is built from
    `_layers`, which reads only bars <= i, and nothing is scaled against the
    whole frame - a scaler fitted end to end leaks the test period's
    distribution into the training rows without tripping any shift-based
    audit.

    SHAPE IS CHECKED BY THE CALLER AND A FAILURE RAISES - unlike `indicators`,
    which is wrapped, because a broken chart annotation must not throw away a
    completed backtest but a silently swapped feature matrix must not survive
    one.

    The columns are the state the RULES read, as RATIOS rather than levels. A
    band width of 12.5 means nothing across contracts, and a classifier fitted
    on levels learns the symbol rather than the setup.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["sma_window"]), float(p["inner_sd"]),
                float(p["outer_sd"]), bool(p["use_neutral_filter"]),
                bool(p["use_outer_guard"]), bool(p["use_baseline_exit"]))
    close = L["close"]
    atr = L["atr"].replace(0.0, np.nan)
    sd = L["sd"].replace(0.0, np.nan)
    volume = (bars["volume"].astype("float64") if "volume" in bars.columns
              else pd.Series(np.nan, index=bars.index))
    vol_sma = volume.rolling(20, min_periods=20).mean().shift(1)
    vol_std = volume.rolling(20, min_periods=20).std().shift(1)
    out = pd.DataFrame({
        "sd_from_mid": L["sd_from_mid"],
        "band_width_norm": (L["inner_upper"] - L["inner_lower"]) / close,
        "outer_width_norm": (L["outer_upper"] - L["outer_lower"]) / close,
        "sd_over_atr": sd / atr,
        "norm_atr": atr / close,
        "dist_inner_upper_sd": (close - L["inner_upper"]) / sd,
        "dist_inner_lower_sd": (close - L["inner_lower"]) / sd,
        "dist_outer_upper_sd": (close - L["outer_upper"]) / sd,
        "roc_3": close.pct_change(3),
        "roc_10": close.pct_change(10),
        "range_atr": (bars["high"] - bars["low"]) / atr,
        "volume_z": (volume - vol_sma) / vol_std.replace(0.0, np.nan),
    }, index=bars.index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_signal_fn(**params):
    """The parameterised form `load_strategy` prefers. Binds `params` once and
    returns the one-argument callable the engine walks."""
    bound = {**DEFAULT_PARAMS, **(params or {})}
    unknown = set(bound) - set(DEFAULT_PARAMS)
    if unknown:
        # Raise rather than ignore. A stale grid key silently dropped would
        # sweep the DEFAULT and report it under the swept name.
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")

    def _fn(bars: pd.DataFrame):
        return signal_fn(bars, **bound)
    return _fn
