"""
macd_zigzag_trend_20260911.py - MACD momentum inside a confirmed ZigZag trend.

Location:  ~/src/trading/strategies/experimental/macd_zigzag_trend_20260911.py

Three layers, each answering a different question, and an entry needs all of
them. An EMA says which way the tape leans. A ZigZag says whether the market
STRUCTURE agrees - higher highs over higher lows, or the mirror. MACD says
momentum is not merely positive but EXPANDING. The structure test is what
makes this more than a moving-average filter on a MACD cross: it requires the
last two swing highs and the last two swing lows to be in order, which a
single indicator cannot see.

================================================================================
STRATEGY SPECIFICATION & BACKTEST REQUEST  (filled from the 2026-09-11 request)
================================================================================
1. STRATEGY METADATA
   Strategy Name:       macd_zigzag_trend
   Strategy Archetype:  Trend continuation / momentum expansion
   Primary Timeframe:   15m  (TIMEFRAME; the ladder is TIMEFRAMES)
   Target Assets:       the 19-contract matrix (SYMBOLS)

2. CORE CONCEPT & HYPOTHESIS
   A trend that is still being fed shows it in three places at once: price on
   the right side of its own mean, structure making progress, and momentum
   accelerating rather than merely present. The inefficiency harvested is the
   continuation leg - flow that has to follow a move already confirmed by
   structure. It persists because the confirmation is late by construction:
   a swing point is only a swing point once price has retraced from it, and
   everybody trading the level is trading it after that.

3. INDICATORS & PARAMETER GRID
   EMA({ema_period})  - the baseline. Price on the wrong side of it vetoes.
   ZigZag, {zigzag_atr_mult} x ATR(14) reversal threshold - swing highs and
       lows, CONFIRMED rather than centred. See the causality note below.
   MACD({macd_fast}, {macd_slow}, {macd_signal}) - line, signal, histogram.
   ATR(14), Wilder - the stop, the target and the ZigZag threshold.

4. ENTRY & EXIT EXECUTION RULES
   Long:   Close > EMA({ema_period}); structure bullish (the last two
           confirmed highs rising AND the last two confirmed lows rising);
           MACD line above signal; histogram > 0 AND greater than the previous
           bar.
   Short:  the mirror, written out rather than negated.
   Exit:   ATR stop at {sl_atr_mult} x ATR(14), target at {tp_atr_mult} x
           ATR(14), and the opposite trigger. Trailing with a STEP - see
           below.
   Session Rules:   none modelled - NOT SPECIFIED in the request.
   Execution Fill:  next-bar open, engine-applied slippage and commission.

5. BACKTEST EXECUTION CONTROLS
   Repository defaults: in-sample 2013-01-01..2022-12-31, holdout 2023-01-01
   onward and untouched during optimisation, Version B evaluated against A.

THE ZIGZAG IS THE WHOLE RISK, AND IT IS NOT VECTORISED
------------------------------------------------------
**A ZigZag cannot be computed by a vectorised window without reading the
future, and the request's own "strictly lookahead-free" requirement is what
rules that form out.** The familiar implementation marks a swing high at the
bar holding the highest price of a window centred on it - which is not known
until the right half of that window has printed. Read at the pivot bar it is
lookahead, and a backtest built on it sells the exact top of every leg and
reports a curve nothing can reproduce.

So the pivot here is CONFIRMED BY RETRACEMENT, which is the definition that
is causal: an extreme becomes a swing high on the bar price has fallen
{zigzag_atr_mult} x ATR from it, and it is only from that bar onward that any
rule may know about it. `_zigzag_loop` is a state machine over the bars for
the same reason `_walk_loop` is - the answer at bar i depends on the path, not
on a window - and it is compiled with numba exactly like the position walk. It
reads `high[i]`, `low[i]` and nothing later, and NO NEGATIVE SHIFT APPEARS
ANYWHERE IN THIS MODULE.

The cost is real and is the point: the entry is late by the whole retracement,
exactly as it would be live. `test_signals_never_change_when_future_bars_are
_removed` is what holds it.

THE TRAILING STEP, AND WHY THE SHARED WALK GREW ONE PARAMETER
-------------------------------------------------------------
Every other module in this directory carries `_walk_loop` duplicated verbatim,
and that convention exists so two copies cannot disagree about the fill
timeline. This module's copy adds ONE parameter, `trail_step`, because the
request asks for a trailing stop that moves only after a defined increment of
profit and the shared walk ratchets on every new high.

It is a STRICT SUPERSET: at `trail_step=0.0` the ratchet is continuous and the
function reproduces the shared walk exactly. That is not an assertion in a
comment - `test_the_walk_matches_the_shared_one_when_the_step_is_zero` runs
this module's walk and `semafor_ha_momentum_20260910._walk_loop` on identical
arrays, both directions, and requires identical output. Proving non-divergence
is worth more than copying and hoping.

WHAT THE ENGINE CANNOT DO, STATED SO NO DRAWDOWN IS MISREAD
------------------------------------------------------------
There are no stop or target ORDERS in this repository. `from_signals` is
driven by boolean masks and fills at the NEXT BAR'S OPEN, so a stop is
detected on the bar that breaches it and filled one bar later, not at the stop
price. Every drawdown figure this module produces is a next-bar-open drawdown
and is worse than a resting-order backtest would show. The take-profit is the
same: `tp_atr_mult` is a DETECTION level, not a limit order, so a target
tagged intrabar and gone by the close is not a fill.
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
#: would not raise - it would silently run an intraday system on DAILY bars.
TIMEFRAME = "15m"

#: The ladder Stage 1 is meant to screen across. DESCRIPTIVE ONLY - no loader
#: or pipeline stage reads this name; a timeframe is selected per run with
#: `--timeframe`.
#:
#: 1m and 2m are absent deliberately: `intraday_start_year` exists because
#: pre-2013 1-minute data is sparse for ten symbols, and a 26-bar EMA on those
#: bars describes the gaps as much as the tape.
TIMEFRAMES = ["5m", "15m", "30m", "1h"]

#: The 19-contract matrix. `python -m backtest.specs` reports UNVERIFIED for
#: M2K and MYM alone, and neither is here.
#:
#: A DECLARATION OF WHERE THE PREMISE MIGHT HOLD, NOT A CLAIM THAT IT DOES.
#: Cross-sectional by default: each contract is its own simulation on its own
#: multiplier, tick size and commission, and symbols are never blended into
#: one equity curve. Two carried caveats from CLAUDE.md, neither caught by the
#: spec reconciler because it checks the CONTRACT rather than the bars: SI
#: definitions stop at 2016 and CL at 2025-12.
SYMBOLS = ["ES", "NQ", "YM", "RTY", "CL", "NG", "RB", "HO", "GC", "SI", "PL",
           "6E", "6B", "6J", "6S", "ZB", "ZN", "BTC", "ETH"]

STRATEGY_NAME = "MACD_ZIGZAG_TREND"
STRATEGY_MODULE = "macd_zigzag_trend_20260911"

PORTFOLIO_GROUP = "Trend_Continuation"
CORRELATION_PROFILE = (
    "Spans every group in the matrix, so a basket drawn from it is only as "
    "diversified as the contracts chosen. The index names are one factor at "
    "four capitalisations and the energy names are another; routing sizes "
    "against that, not against the symbol count.")

#: The regime the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and nothing here reads these. Quadrant ids
#: are `mdlib/regimes.py`'s: Q1 High-Vol/Trending, Q2 High-Vol/Ranging,
#: Q3 Low-Vol/Trending, Q4 Low-Vol/Ranging, 0 UNDEFINED (warm-up, not a
#: quadrant). A structure-confirmed continuation nominates the TRENDING pair -
#: in a range the ZigZag alternates without making progress and the structure
#: test is measuring noise, which is what Gate R settles rather than this.
TARGET_REGIMES = ("High Volatility / Trending", "Low Volatility / Trending")
TARGET_QUADRANTS = ("Q1", "Q3")

#: Wilder's ATR period. Used by the stop, the target AND the ZigZag threshold,
#: so it is fixed rather than exposed: sweeping it would move three things at
#: once and no result could be attributed to any of them.
ATR_PERIOD = 14

#: A stop nearer than this to the fill sits inside the bar's own noise and the
#: run measures the cost model rather than the strategy.
MIN_STOP_ATR_MULT = 0.25


DEFAULT_PARAMS = {
    "ema_period": 200,
    "zigzag_atr_mult": 3.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "require_expansion": True,
    "sl_atr_mult": 2.0,
    # 2R AGAINST THE 2.0 STOP. The request asks for "2x or 3x the stop loss
    # distance", and the engine's walk takes the target as its own ATR
    # multiple - so the reward:risk ratio lives in the relationship between
    # these two numbers rather than in a third parameter that could disagree
    # with them. `_reward_risk()` reports it and `_validate` refuses a target
    # inside the stop. 3R is `tp_atr_mult=6.0`, which PARAM_GRID sweeps.
    "tp_atr_mult": 4.0,
    "trailing": True,
    # The trailing STEP, in ATR. The stop moves only once price has made this
    # much new profit beyond the level the current stop was computed from, so
    # it ratchets in increments rather than on every tick of a new high. 0.0
    # is a continuous trail and reproduces the shared walk exactly.
    "trail_step_atr": 0.5,
}

#: The search space `backtest/run.py --scan` sweeps.
#:
#:     3 x 3 x 2 x 2 = 36 combinations
#:
#: Coarse on purpose. `variants_tested` travels with every result precisely so
#: a reader can see how large the search was that produced a Sharpe; a 400-cell
#: grid over the same bars is a machine for manufacturing an in-sample Sharpe.
#:
#: `zigzag_atr_mult` is swept because it IS the strategy's definition of a
#: swing - too small and every wiggle is structure, too large and the
#: confirmation arrives after the leg is over. `tp_atr_mult` carries the
#: request's "2x or 3x" as the two values 4.0 and 6.0 against the 2.0 stop.
#:
#: The MACD periods are PINNED at the request's 12/26/9 and not swept. They
#: are the definition of the indicator; a swept MACD is a different oscillator
#: at every value rather than the same one measured more carefully.
PARAM_GRID = {
    "ema_period": [50, 100, 200],
    "zigzag_atr_mult": [2.0, 3.0, 4.0],
    "sl_atr_mult": [1.5, 2.0],
    "tp_atr_mult": [4.0, 6.0],
}

LOGIC = {
    "concept": (
        "A trend still being fed shows it in three places at once: price on "
        "the right side of EMA({ema_period}), market STRUCTURE making "
        "progress, and momentum expanding rather than merely present. The "
        "ZigZag supplies the structure - the last two confirmed swing highs "
        "and the last two confirmed swing lows, each confirmed only once "
        "price retraced {zigzag_atr_mult} x ATR from it, so nothing here "
        "knows a pivot before the market did. Trades are attributed by their "
        "ENTRY bar, never the exit, so a move opened in one quadrant that "
        "runs into another still scores where it opened."),
    "entry": (
        "LONG when Close > EMA({ema_period}), the structure is bullish "
        "(higher confirmed high AND higher confirmed low), "
        "MACD({macd_fast},{macd_slow},{macd_signal}) is above its signal, and "
        "the histogram is positive and larger than the previous bar's. SHORT "
        "on the mirror: below the EMA, lower confirmed low AND lower "
        "confirmed high, MACD below signal, histogram negative and more "
        "negative than the previous bar's. The entry fires on the bar the "
        "whole set FIRST holds, not on every bar it continues to hold."),
    "exit": (
        "A protective stop at {sl_atr_mult} x ATR(14) and a target at "
        "{tp_atr_mult} x ATR(14) - a reward:risk of "
        "tp_atr_mult/sl_atr_mult - both DETECTED on the bar they are breached "
        "and filled at the next bar's open, because the engine has no stop or "
        "target orders. The opposite trigger also closes the position. With "
        "trailing={trailing} the stop follows the high-water mark in steps of "
        "{trail_step_atr} x ATR, moving only after that much new profit rather "
        "than on every new high."),
}


# ---------------------------------------------------------------------------
# Indicator kernels
#
# `_ema`, `_wilder`, `_true_range`, `_atr`, `_as_series`, `_cross_above` and
# `_cross_below` are DUPLICATED VERBATIM from
# `multi_ema_cci_trend_20260910.py`, which shares them with the rest of this
# directory. Strategy modules are loaded from a FILE PATH by
# `agents.tier3_workers.load_strategy` and promoted as a self-contained copy,
# so a shared import would resolve against whatever happens to sit beside the
# module at load time - and a promoted package must reproduce the file that
# was certified, byte for byte. Do not "improve" one alone.
#
# `_walk_loop` is that same copy PLUS ONE PARAMETER - see the module
# docstring, and the test that holds it to identical output at trail_step=0.
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    A span-`period` exponential mean, NaN until `period` bars exist.

    `min_periods=period` rather than pandas' default of 1: without it the
    first bar's EMA is that bar's own close, which is not an average of
    anything, and every early bar is trivially on one side of it.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


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
    rather than placing a bracket of zero width around the fill, and the
    ZigZag cannot confirm a pivot on a threshold of zero either.
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

    A STATE (`fast > level`) would be true for every bar of a run; this is the
    EVENT, true once. NaN-SAFE BY CONSTRUCTION: every comparison against NaN
    is False, so a warm-up bar is never a cross. `.shift(1)` looks one bar
    BACKWARD.
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


def _macd(close: pd.Series, fast: int, slow: int, signal: int
          ) -> dict[str, pd.Series]:
    """
    MACD line, signal and histogram, on the standard EMA definition.

        line   = EMA(fast) - EMA(slow)
        signal = EMA(line, signal)
        hist   = line - signal

    The signal EMA is taken over the LINE, not over price - a signal computed
    from price is a third moving average that crosses on different bars, and
    this module's entire momentum gate is a comparison between the two.
    """
    line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(line, signal)
    return {"macd": line, "signal": sig, "hist": line - sig}


def _zigzag_loop(high: np.ndarray, low: np.ndarray, atr: np.ndarray,
                 dev_mult: float) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray,
                                           np.ndarray]:
    """
    Confirmed ZigZag pivots, as four AS-OF arrays plus a confirmation marker.

    Returns `(last_high, prev_high, last_low, prev_low, confirmed)`, each
    aligned to the bars. At bar i they hold the pivots KNOWN BY BAR i and
    nothing later, which is the entire contract of this function.

    THE PIVOT IS CONFIRMED BY RETRACEMENT, NOT BY A CENTRED WINDOW. A swing
    high is an extreme that price has since fallen `dev_mult * ATR` away from;
    the bar it becomes a swing high is the bar of that retracement, not the
    bar of the extreme. Those are different bars and the difference is the
    whole causality question: the centred form knows at the extreme, which is
    lookahead, and a backtest on it sells the exact top of every leg.

    A STATE MACHINE RATHER THAN A WINDOW, and necessarily so - the answer at
    bar i depends on the path taken to get there, not on a fixed span of bars,
    so there is no vectorised form that is also causal. The same reason
    `_walk_loop` is a loop, and it is compiled the same way.

    The threshold is in ATR rather than percent so one parameter means the
    same thing on ES at 5000 and on 6J at 0.0068. A bar with no ATR yet, or a
    zero ATR, can confirm nothing: the comparison against NaN is False and a
    zero threshold would make every bar a pivot.

    Ties and the seed. Direction starts UNKNOWN and is taken from the first
    bar that has an ATR, comparing that bar's own range halves; the first
    pivot is therefore whichever side retraces first, and the arrays stay NaN
    until two of each have been confirmed. A structure rule reading NaN gets
    False, which is the correct warm-up answer.
    """
    n = high.shape[0]
    last_high = np.full(n, np.nan)
    prev_high = np.full(n, np.nan)
    last_low = np.full(n, np.nan)
    prev_low = np.full(n, np.nan)
    confirmed = np.zeros(n, dtype=np.int8)

    direction = 0                 # 0 unknown, 1 tracking a high, -1 a low
    ext = np.nan                  # the running extreme of the current leg
    lh = np.nan
    ph = np.nan
    ll = np.nan
    pl = np.nan

    for i in range(n):
        a = atr[i]
        ok = np.isfinite(a) and a > 0.0
        if ok:
            thresh = dev_mult * a
            if direction == 0:
                # Seed on the first usable bar. Whichever way it retraces
                # first decides, so this costs at most one leg of lateness and
                # never reads forward.
                direction = 1
                ext = high[i]
            elif direction == 1:
                if high[i] > ext:
                    ext = high[i]
                elif ext - low[i] >= thresh:
                    ph = lh
                    lh = ext
                    confirmed[i] = 1
                    direction = -1
                    ext = low[i]
            else:
                if low[i] < ext:
                    ext = low[i]
                elif high[i] - ext >= thresh:
                    pl = ll
                    ll = ext
                    confirmed[i] = -1
                    direction = 1
                    ext = high[i]

        # WRITTEN AT THE END OF THE BAR, so bar i carries the confirmation
        # made ON bar i and every earlier one, and nothing after.
        last_high[i] = lh
        prev_high[i] = ph
        last_low[i] = ll
        prev_low[i] = pl

    return last_high, prev_high, last_low, prev_low, confirmed


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
               trailing: bool,
               trail_step_mult: float) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]:
    """
    THREE-state machine over the bars: flat, long, or short - each side under
    its own stop, target and signal exit.

    The shared walk from `multi_ema_cci_trend_20260910.py` PLUS `trail_step`,
    and a strict superset of it: at `trail_step=0.0` the ratchet is continuous
    and the output is identical, which a test holds rather than a comment.

    The timeline matches the engine's. An entry signal on bar i is filled at
    bar i+1's open, so the position is live from bar i+1, the fill price is
    `open_[i + 1]`, and the extreme-price mark starts there - NOT on the
    signal bar. Both distances are frozen at `mult * ATR` as measured on the
    SIGNAL bar and never re-measured as volatility changes.

    The two sides, written out rather than folded into a sign flip, because a
    reader has to be able to check them against the specification by eye:

        LONG    stop     fill - dist    trailing: (stepped high-water) - dist
                target   fill + dist
                exits    low  <= stop   or  high >= target
        SHORT   stop     fill + dist    trailing: (stepped low-water)  + dist
                target   fill - dist
                exits    high >= stop   or  low  <= target

    A short stop placed BELOW the fill would be breached by the fill bar
    itself; this is the reason the sides are not a sign flip.

    THE STEPPED TRAIL. `anchor` is the price the current stop was computed
    from. It advances only in whole `trail_step` increments, so the stop stays
    put until the position has made that much new profit and then jumps by
    exactly one step - which is what "move the stop only after an increment of
    profit" means, and is NOT the same as a continuous trail rounded off. With
    `trail_step_mult <= 0` the anchor tracks the high-water mark on every bar and
    the behaviour collapses to the shared walk.

    A bar carrying BOTH a long and a short entry signal while flat takes
    NEITHER, matching `backtest.engine._clean_signals_ls_loop`. A well-formed
    strategy cannot produce it - the structure cannot be bullish and bearish
    on one bar - so the branch exists to make a malformed one visible as a
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
    trail_step = 0.0
    tp_dist = np.nan
    entry_px = np.nan
    hw = 0.0                             # highest high since the fill (long)
    lw = 0.0                             # lowest low since the fill (short)
    anchor = np.nan                      # what the current stop is measured from

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
                # THE STEP IS FROZEN AT THE SIGNAL BAR, like the two
                # distances above. Sizing it from anything computed over the
                # whole frame - a median ATR, say - would make the step a
                # property of bars that have not printed, which is the
                # lookahead this module's ZigZag is written to avoid; it would
                # be no better for arriving through the risk model.
                trail_step = trail_step_mult * atr[i]
                entry_px = np.nan
                hw = -np.inf
                lw = np.inf
                anchor = np.nan
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
            anchor = entry_px

        if state == 1:
            if trailing:
                if high[i] > hw:
                    hw = high[i]
                if trail_step > 0.0:
                    # Whole steps only. A 1.7-step move advances the anchor by
                    # one step, not by 1.7 - the stop is meant to sit still
                    # between increments.
                    while hw - anchor >= trail_step:
                        anchor = anchor + trail_step
                else:
                    anchor = hw
                level = anchor - stop_dist
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
                if trail_step > 0.0:
                    while anchor - lw >= trail_step:
                        anchor = anchor - trail_step
                else:
                    anchor = lw
                level = anchor + stop_dist
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
    _zigzag = njit(nogil=True)(_zigzag_loop)
except ImportError:                     # pragma: no cover - env dependent
    # The same functions, interpreted. `backtest.engine.clean_signals`
    # degrades the same way, and the fallback has to exist because a missing
    # compiler must not change which trades a strategy takes.
    _walk = _walk_loop
    _zigzag = _zigzag_loop


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _reward_risk(sl_atr_mult: float, tp_atr_mult: float | None) -> float:
    """The target's distance as a multiple of the stop's. NaN with no target."""
    if tp_atr_mult is None:
        return float("nan")
    return float(tp_atr_mult) / float(sl_atr_mult)


def _validate(ema_period: int, zigzag_atr_mult: float, macd_fast: int,
              macd_slow: int, macd_signal: int, sl_atr_mult: float,
              tp_atr_mult: float | None, trailing: bool,
              trail_step_atr: float) -> None:
    """
    Refuse a parameter set that cannot mean what it says, where it is passed
    rather than as a strange equity curve. Each has a specific failure behind
    it:

    * A period below 2 makes its average the series itself.
    * `macd_fast >= macd_slow` inverts the oscillator: the "fast" mean lags
      the "slow" one, the histogram changes sign, and every rule reading it
      means the opposite of what it says while still computing.
    * `zigzag_atr_mult <= 0` confirms a pivot on every bar, so the structure
      test is reading noise at maximum frequency rather than swings.
    * A stop inside MIN_STOP_ATR_MULT sits inside the bar's own noise.
    * A target INSIDE the stop is a reward:risk below 1 that the walk will
      almost always resolve as a win, which reads as a high hit rate on a
      strategy that loses money. The request asks for 2x or 3x; anything at or
      under 1x is refused as a typo rather than honoured.
    * A negative `trail_step_atr` would walk the stop the wrong way.
    """
    for name, value in (("ema_period", ema_period),
                        ("macd_fast", macd_fast), ("macd_slow", macd_slow),
                        ("macd_signal", macd_signal)):
        if int(value) < 2:
            raise ValueError(
                f"{name} must be >= 2; got {value!r}. Below that the average "
                f"is the series itself and the indicator collapses onto "
                f"price.")
    if int(macd_fast) >= int(macd_slow):
        raise ValueError(
            f"macd_fast must be < macd_slow; got {macd_fast!r} and "
            f"{macd_slow!r}. Inverted, the histogram changes sign and every "
            f"rule reading it means the opposite of what it says.")
    if float(zigzag_atr_mult) <= 0.0:
        raise ValueError(
            f"zigzag_atr_mult must be > 0; got {zigzag_atr_mult!r}. At or "
            f"below zero every bar confirms a pivot and the structure test "
            f"reads noise rather than swings.")
    if float(sl_atr_mult) < MIN_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be >= {MIN_STOP_ATR_MULT}; got "
            f"{sl_atr_mult!r}. A tighter stop sits inside the bar's own noise "
            f"and the run measures the cost model.")
    if tp_atr_mult is not None:
        if float(tp_atr_mult) <= 0.0:
            raise ValueError(
                f"tp_atr_mult must be > 0 or None (no target); got "
                f"{tp_atr_mult!r}.")
        if _reward_risk(sl_atr_mult, tp_atr_mult) <= 1.0:
            raise ValueError(
                f"tp_atr_mult {tp_atr_mult!r} is at or inside the stop "
                f"{sl_atr_mult!r} (reward:risk "
                f"{_reward_risk(sl_atr_mult, tp_atr_mult):.2f}). The request "
                f"specifies 2x or 3x the stop distance; a target inside the "
                f"stop resolves as a win almost every time and reads as a "
                f"high hit rate on a strategy that loses money.")
    if not isinstance(trailing, (bool, np.bool_)):
        raise ValueError(f"trailing must be a bool; got {trailing!r}")
    if float(trail_step_atr) < 0.0:
        raise ValueError(
            f"trail_step_atr must be >= 0; got {trail_step_atr!r}. A negative "
            f"step would walk the stop away from the position.")


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
def _layers(bars: pd.DataFrame, ema_period: int, zigzag_atr_mult: float,
            macd_fast: int, macd_slow: int, macd_signal: int,
            require_expansion: bool) -> dict[str, pd.Series]:
    """
    Every series both `signal_fn` and `indicators` read, computed ONCE.

    One call site for the maths is the point: a second implementation in the
    report would be free to disagree with this one and draw a crossover a bar
    from where the trade fired, with nothing raising.

    The toggle lives here rather than in `signal_fn` so a disabled filter is a
    Series of True with the underlying series still computed and still drawn -
    a reader can see what the filter WOULD have said on a run where it was off.
    """
    close = bars["close"].astype("float64")
    idx = bars.index
    atr = _atr(bars, ATR_PERIOD)
    ema = _ema(close, int(ema_period))
    M = _macd(close, int(macd_fast), int(macd_slow), int(macd_signal))

    lh, ph, ll, pl, conf = _zigzag(
        bars["high"].to_numpy(dtype="float64"),
        bars["low"].to_numpy(dtype="float64"),
        atr.to_numpy(dtype="float64"),
        float(zigzag_atr_mult))
    last_high = pd.Series(lh, index=idx)
    prev_high = pd.Series(ph, index=idx)
    last_low = pd.Series(ll, index=idx)
    prev_low = pd.Series(pl, index=idx)

    # MARKET STRUCTURE, from CONFIRMED pivots only. Higher high AND higher low
    # for a bull structure - both, because a higher high on a lower low is an
    # expanding range rather than a trend, and it is the pair that the request
    # names. NaN before two of each have been confirmed, and every comparison
    # against NaN is False, which is the correct warm-up answer.
    structure_bull = (last_high > prev_high) & (last_low > prev_low)
    structure_bear = (last_low < prev_low) & (last_high < prev_high)

    above_ema = close > ema
    below_ema = close < ema

    hist = M["hist"]
    # EXPANDING, not merely present. `.diff()` looks one bar BACKWARD.
    hist_rising = hist.diff() > 0.0
    hist_falling = hist.diff() < 0.0
    if require_expansion:
        momo_long = (M["macd"] > M["signal"]) & (hist > 0.0) & hist_rising
        momo_short = (M["macd"] < M["signal"]) & (hist < 0.0) & hist_falling
    else:
        momo_long = (M["macd"] > M["signal"]) & (hist > 0.0)
        momo_short = (M["macd"] < M["signal"]) & (hist < 0.0)

    setup_long = above_ema & structure_bull & momo_long
    setup_short = below_ema & structure_bear & momo_short

    # THE ENTRY IS THE BAR THE SETUP FIRST HOLDS, not every bar it continues
    # to hold. The request states four CONDITIONS; taken as a state they are
    # true for a run of bars and would re-fire on each one. `.shift(1)` looks
    # one bar BACKWARD, and a NaN warm-up bar shifts in as False, so the first
    # true bar of the series counts as an edge.
    trigger_long = setup_long & ~setup_long.shift(1, fill_value=False)
    trigger_short = setup_short & ~setup_short.shift(1, fill_value=False)

    return {
        "close": close, "ema": ema, "atr": atr,
        "macd": M["macd"], "signal": M["signal"], "hist": hist,
        "last_high": last_high, "prev_high": prev_high,
        "last_low": last_low, "prev_low": prev_low,
        "pivot_confirmed": pd.Series(conf, index=idx),
        "structure_bull": structure_bull, "structure_bear": structure_bear,
        "above_ema": above_ema, "below_ema": below_ema,
        "momo_long": momo_long, "momo_short": momo_short,
        "setup_long": setup_long, "setup_short": setup_short,
        "trigger_long": trigger_long, "trigger_short": trigger_short,
        "exit_long": trigger_short, "exit_short": trigger_long,
        "dist_ema_atr": (close - ema) / atr.replace(0.0, np.nan),
        "hist_atr": hist / atr.replace(0.0, np.nan),
    }


# --------------------------------------------------------------------------
# The contract functions
# --------------------------------------------------------------------------
def signal_fn(bars: pd.DataFrame,
              ema_period: int = 200,
              zigzag_atr_mult: float = 3.0,
              macd_fast: int = 12,
              macd_slow: int = 26,
              macd_signal: int = 9,
              require_expansion: bool = True,
              sl_atr_mult: float = 2.0,
              tp_atr_mult: float | None = 4.0,
              trailing: bool = True,
              trail_step_atr: float = 0.5) -> tuple[pd.Series, pd.Series,
                                                    pd.Series, pd.Series]:
    """
    Returns the FOUR-MASK form: (long_entries, long_exits, short_entries,
    short_exits), all boolean Series on `bars.index`.

    A three-tuple or a bare Series RAISES in `engine.unpack_signals` rather
    than silently losing the short side into a plausible long-only curve.

    THE SHORT SIDE IS NOT THE LONG SIDE WITH A SIGN FLIP. Inside `_walk_loop`
    the short stop sits ABOVE the fill; placed below it, the fill bar itself
    would breach it. The masks are written out per side for the same reason.

    NOTHING HERE READS A BAR LATER THAN i. The ZigZag confirms a pivot only on
    the bar price has retraced from it, and no negative shift appears anywhere
    in this module.
    """
    _validate(ema_period, zigzag_atr_mult, macd_fast, macd_slow, macd_signal,
              sl_atr_mult, tp_atr_mult, trailing, trail_step_atr)

    L = _layers(bars, int(ema_period), float(zigzag_atr_mult), int(macd_fast),
                int(macd_slow), int(macd_signal), bool(require_expansion))

    long_ok = L["trigger_long"].fillna(False).to_numpy()
    short_ok = L["trigger_short"].fillna(False).to_numpy()
    long_sig_exit = L["exit_long"].fillna(False).to_numpy()
    short_sig_exit = L["exit_short"].fillna(False).to_numpy()

    atr = L["atr"].to_numpy(dtype="float64")
    # A bar with no ATR yet is not tradable: the protective bracket has no
    # width, and admitting it would place a stop at the fill price.
    warm = ~np.isfinite(atr) | (atr <= 0.0)
    # The EMA and the MACD warm up on their own windows, and the structure
    # test needs four confirmed pivots. A NaN in any of them already makes the
    # trigger False; masking explicitly keeps that true however they are swept.
    warm = warm | ~np.isfinite(L["ema"].to_numpy(dtype="float64"))
    warm = warm | ~np.isfinite(L["hist"].to_numpy(dtype="float64"))
    long_ok = long_ok & ~warm
    short_ok = short_ok & ~warm

    flat_bar = np.zeros(len(bars), dtype=bool)

    le, lx, se, sx, _, _ = _walk(
        long_ok, short_ok, long_sig_exit, short_sig_exit,
        bars["open"].to_numpy(dtype="float64"),
        bars["high"].to_numpy(dtype="float64"),
        bars["low"].to_numpy(dtype="float64"),
        np.nan_to_num(atr, nan=0.0), flat_bar, float(sl_atr_mult),
        _tp_distance(tp_atr_mult), bool(trailing), float(trail_step_atr))

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
    `_layers` call `signal_fn` uses - so the chart cannot draw a crossover a
    bar from where the entry actually happened.

    The two confirmed pivot levels are drawn as step lines rather than as
    markers, which is what they are: the level a rule read at that bar.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["ema_period"]), float(p["zigzag_atr_mult"]),
                int(p["macd_fast"]), int(p["macd_slow"]),
                int(p["macd_signal"]), bool(p["require_expansion"]))
    return {
        f"EMA({int(p['ema_period'])})": L["ema"],
        f"MACD({int(p['macd_fast'])},{int(p['macd_slow'])})": L["macd"],
        f"Signal({int(p['macd_signal'])})": L["signal"],
        "MACD hist": L["hist"],
        "ZigZag last high": L["last_high"],
        "ZigZag last low": L["last_low"],
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
    MACD histogram of 12.5 means nothing across contracts, and a classifier
    fitted on levels learns the symbol rather than the setup. The swing
    distances are in ATR for the same reason.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["ema_period"]), float(p["zigzag_atr_mult"]),
                int(p["macd_fast"]), int(p["macd_slow"]),
                int(p["macd_signal"]), bool(p["require_expansion"]))
    close = L["close"]
    atr = L["atr"].replace(0.0, np.nan)
    volume = (bars["volume"].astype("float64") if "volume" in bars.columns
              else pd.Series(np.nan, index=bars.index))
    vol_sma = volume.rolling(20, min_periods=20).mean().shift(1)
    vol_std = volume.rolling(20, min_periods=20).std().shift(1)
    out = pd.DataFrame({
        "dist_ema_atr": L["dist_ema_atr"],
        "hist_atr": L["hist_atr"],
        "hist_slope_atr": L["hist"].diff() / atr,
        "macd_minus_signal_atr": (L["macd"] - L["signal"]) / atr,
        "swing_high_dist_atr": (close - L["last_high"]) / atr,
        "swing_low_dist_atr": (close - L["last_low"]) / atr,
        "swing_range_atr": (L["last_high"] - L["last_low"]) / atr,
        "higher_high": (L["last_high"] - L["prev_high"]) / atr,
        "higher_low": (L["last_low"] - L["prev_low"]) / atr,
        "structure_bull": L["structure_bull"].astype("float64"),
        "structure_bear": L["structure_bear"].astype("float64"),
        "norm_atr": atr / close,
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
