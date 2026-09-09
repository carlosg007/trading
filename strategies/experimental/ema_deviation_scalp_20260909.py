"""
ema_deviation_scalp_20260909.py - the Trader DNA 20-EMA deviation scalp.

Location:  ~/src/trading/strategies/experimental/ema_deviation_scalp_20260909.py

A mean-reversion scalp: price stretches away from a 20 EMA by a multiple of
ATR, then RECLAIMS the average, with a Stochastic cross out of an extreme as
the confirmation that the reclaim has momentum behind it rather than being one
bar of noise inside a trend that is still going the other way.

================================================================================
STRATEGY SPECIFICATION & BACKTEST REQUEST  (filled from the 2026-09-09 request)
================================================================================
1. STRATEGY METADATA
   Strategy Name:       ema_deviation_scalp
   Strategy Archetype:  Mean-Reversion (scalp)
   Primary Timeframe:   5m                      <- ASSUMED, see TIMEFRAME
   Target Assets:       NQ, ES, RTY, YM         <- ASSUMED, see SYMBOLS

2. CORE CONCEPT & HYPOTHESIS
   Intraday order flow overshoots. A burst of one-sided activity pushes price
   a measurable distance from its own short-term average, and the liquidity
   that was consumed to do it is replaced at a price nearer the average. The
   inefficiency harvested is that overshoot; it persists because the flow that
   causes it is usually not information, and the participants replacing the
   liquidity are compensated for taking the other side. The EMA is the anchor,
   the ATR multiple is what makes "far" comparable across contracts and
   volatility regimes, and the Stochastic cross is the evidence that the
   reversion has actually begun rather than being predicted.

3. INDICATORS & PARAMETER GRID
   EMA({ema_window}) on close - the anchor.
   ATR(14), Wilder - the deviation yardstick and the bracket width.
   Stochastic(%K {k_period}, smooth {smooth_k}, %D {d_period}) - confirmation.
   See DEFAULT_PARAMS and PARAM_GRID below.

4. ENTRY & EXIT EXECUTION RULES
   Long:   Low < EMA - {deviation_atr_mult} x ATR within the last
           {stretch_lookback} bars, AND Close crosses ABOVE the EMA, AND %K
           crosses above %D having been <= {oversold}. The reclaim and the
           Stochastic cross need only land within {stoch_lookback} bars of
           each other - the entry fires on whichever completes the pair. See
           `_layers`: demanding them on the SAME bar produced two triggers
           where the parts fired 19 and 13 times.
   Short:  the mirror - High > EMA + {deviation_atr_mult} x ATR, Close crosses
           BELOW the EMA, %K crosses below %D having been >= {overbought}.
   TP:     {tp_atr_mult} x ATR from the fill.
   SL:     {sl_atr_mult} x ATR from the fill. trailing={trailing}.
   Session Rules:   none modelled - NOT SPECIFIED in the request. Every bar the
                    lake serves is tradable here. A scalp is the archetype most
                    exposed to the overnight session, so this is the first
                    thing to revisit if the Stage 1 profile looks strange.
   Execution Fill:  next-bar open, with contract slippage and commission
                    applied by the engine.

5. BACKTEST EXECUTION CONTROLS
   In-Sample / Holdout / Version B: the repository defaults - in-sample
   2013-01-01..2022-12-31, holdout 2023-01-01 onward and untouched during
   optimisation, Version B evaluated against Version A out of sample.

WHAT IS ASSUMED HERE RATHER THAN SPECIFIED, so it is visible before a number
is read: the timeframe, the symbol basket and the absence of a session window.
The request fixed the formula and the brackets and said nothing about any of
the three, and each gets decided whether or not anyone writes it down.

TWO NOTES ON THE REQUEST AS WRITTEN
-----------------------------------
`pandas_ta` was offered as an import. It is NOT in
`agents.tier3_workers.ALLOWED_IMPORTS` (`__future__`, `numpy`, `pandas`,
`math`, `vectorbtpro`, `numba`), so the AST gate would refuse this module for
importing it. The Stochastic is implemented below instead, which is also the
convention in this directory - modules are loaded from a FILE PATH and
promoted as a self-contained copy, so every kernel is duplicated rather than
shared.

`generate_signals(df, params)` was the requested entry point. The engine's
contract is `signal_fn(bars, **params)` / `make_signal_fn(**params)`, and a
module exposing only `generate_signals` cannot be called by
`agents.tier3_workers.load_strategy` at all. Both are provided: `signal_fn` is
the contract, `generate_signals` is a thin adapter over it for the requested
calling convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

#: The bar this was designed on. ASSUMED - the request named no timeframe.
#: Stage 1 screens the ladder and the winning (tf, params) pair is the claim,
#: never the parameters alone. A 20 EMA is ~100 minutes at 5m and ~10 hours at
#: 30m, so the anchor means a different thing on every rung.
TIMEFRAME = "5m"

#: ASSUMED - the request named no assets. Index futures, which is where the
#: overshoot-and-reclaim premise has the most participants replacing
#: liquidity. All four are verified in `backtest/specs.py`.
SYMBOLS = ["NQ", "ES", "RTY", "YM"]

STRATEGY_NAME = "EMA_DEVIATION_SCALP"
STRATEGY_MODULE = "ema_deviation_scalp_20260909"

#: Portfolio routing metadata. Descriptive only - `config/portfolios.json` is
#: the authority on which account trades what.
PORTFOLIO_GROUP = "Mean_Reversion_Scalp"
CORRELATION_PROFILE = (
    "Index-correlated. NQ, ES, RTY and YM are one factor at four "
    "capitalisations, so a basket of them concentrates rather than "
    "diversifies - a fact for portfolio routing to size against.")

#: The regime the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and nothing in this module reads these.
#: Quadrant ids are `mdlib/regimes.py`'s and nothing else's: Q1 High-Vol/
#: Trending, Q2 High-Vol/Ranging, Q3 Low-Vol/Trending, Q4 Low-Vol/Ranging,
#: 0 UNDEFINED (warm-up, not a quadrant). A reversion scalp nominates the
#: RANGING pair; a stretch away from the mean in a trending regime is often
#: the trend rather than an overshoot, which is exactly what Gate R settles.
TARGET_REGIMES = ("High Volatility / Ranging", "Low Volatility / Ranging")
TARGET_QUADRANTS = ("Q2", "Q4")

#: Wilder's ATR period. Fixed at 14 rather than exposed: it sets BOTH the
#: deviation yardstick and the bracket width, so a swept ATR period would move
#: the entry distance and the stop distance together and no cell of the grid
#: could separate which one paid.
ATR_PERIOD = 14

#: What a Stochastic reads when its own window is dead flat. `highest ==
#: lowest` makes the ratio 0/0 and the oscillator genuinely undefined; 50 is
#: the centerline and says "neither extreme". The naive 0.0 is the most
#: OVERSOLD value the indicator has - it would hold `%K <= oversold`
#: permanently true through every halted or dead session and manufacture long
#: confirmations out of silence.
STOCH_NEUTRAL = 50.0

#: A stop nearer than this to the fill is refused: at a quarter of an ATR the
#: bracket sits inside the bar's own noise and the strategy measures the cost
#: model rather than the reversion.
MIN_STOP_ATR_MULT = 0.25


DEFAULT_PARAMS = {
    "ema_window": 20,
    "deviation_atr_mult": 1.5,
    "stretch_lookback": 5,
    "k_period": 14,
    "d_period": 3,
    "smooth_k": 3,
    "oversold": 20.0,
    "overbought": 80.0,
    "stoch_lookback": 3,
    "use_stretch_filter": True,
    "use_stoch_filter": True,
    "use_stoch_exit": False,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 1.0,
    "trailing": False,
}

#: The search space `backtest/run.py --scan` sweeps.
#:
#:     3 x 3 x 3 x 2 = 54 combinations
#:
#: Kept coarse deliberately. Nine parameters are exposed above and sweeping
#: all of them over intraday bars is a machine for manufacturing an in-sample
#: Sharpe; `variants_tested` travels with every result precisely so a reader
#: can see how big the search was that produced it.
#:
#: `k_period` carries 8 as well as 14 because the request named BOTH - "%K=8"
#: in the prose and `k_period: 14` in the parameter list. 14 is the default
#: and 8 is in the grid, so the sweep answers which the tape prefers instead
#: of the question being settled by whichever line was read first.
#:
#: `sl_atr_mult` is the STOP DISTANCE and `trailing` is pinned False by the
#: premise: a scalp taking {tp_atr_mult} x ATR out of a reversion has a
#: target, and a trailing stop on a position that is meant to be closed at the
#: mean would give back the move it was opened for. `run.py`'s RISK_PARAMS and
#: `promote.py`'s RISK_KEYS are exactly `("sl_atr_mult", "tp_atr_mult",
#: "trailing")`, so a stop distance under any other name would be absent from
#: the promoted risk block entirely.
PARAM_GRID = {
    "ema_window": [10, 20, 30],
    "deviation_atr_mult": [1.0, 1.5, 2.0],
    "k_period": [8, 14],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "trailing": [False],
}

LOGIC = {
    "concept": (
        "Intraday order flow overshoots. A burst of one-sided activity pushes "
        "price a measurable distance from EMA({ema_window}), and the "
        "liquidity consumed to do it is replaced nearer the average. The ATR "
        "multiple is what makes 'far' comparable across contracts and "
        "volatility regimes; the Stochastic cross is evidence the reversion "
        "has BEGUN rather than a prediction that it will. Trades are "
        "attributed by their ENTRY bar, never the exit, so a scalp opened in "
        "one quadrant that closes in another still scores where it opened."),
    "entry": (
        "LONG when Low dipped below EMA({ema_window}) - {deviation_atr_mult} "
        "x ATR(14) within the last {stretch_lookback} bars, Close crosses "
        "ABOVE the EMA, and Stochastic %K crosses above %D having been <= "
        "{oversold}. The two crosses need only arrive within "
        "{stoch_lookback} bars of each other; the entry fires on the later. "
        "SHORT mirrors it: "
        "High above EMA + {deviation_atr_mult} x ATR, Close crossing BELOW "
        "the EMA, %K crossing below %D from >= {overbought}."),
    "exit": (
        "The ATR bracket, whichever side the bar reaches first: a target at "
        "{tp_atr_mult} x ATR(14) and a stop at {sl_atr_mult} x ATR(14) from "
        "the fill (trailing={trailing}). With use_stoch_exit the position "
        "also closes on the opposite Stochastic cross."),
}


# ---------------------------------------------------------------------------
# Indicator kernels
#
# DUPLICATED VERBATIM from `keltner_trend_drift_20260901.py`, which shares
# them with the rest of this directory. Strategy modules are loaded from a
# FILE PATH by `agents.tier3_workers.load_strategy` and promoted as a
# self-contained copy, so a shared import would resolve against whatever
# happens to sit beside the module at load time - and a promoted package must
# reproduce the file that was certified, byte for byte, not whatever a helper
# module has become since.
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    A span-`period` exponential moving average.

    `adjust=False` is the recursive form every charting package draws, and the
    one the 20-EMA anchor is specified in. It is NOT Wilder's smoothing - see
    `_wilder`, which is roughly half this speed and is what the ATR below is
    defined on. Using either where the other belongs produces a line no other
    tool agrees with.
    """
    return series.astype("float64").ewm(span=int(period), adjust=False).mean()


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


def _recent(flag: pd.Series, window: int) -> pd.Series:
    """
    True where `flag` was True on ANY of the last `window` bars, the current
    bar included.

    The stretch and the reclaim are two different bars by construction - price
    cannot be a full ATR multiple below the average and closing back above it
    at the same instant - so the stretch has to be remembered for a few bars
    or the strategy can never fire. `window` is how long the overshoot stays
    relevant; past it, the reclaim is just a crossing.

    CAUSAL and INCLUSIVE OF THE CURRENT BAR: `.rolling(window)` spans bars
    i-window+1 .. i. `min_periods=1` so the first bars of the frame answer on
    what exists rather than being NaN, which `fillna(False)` would then read
    as "no stretch" anyway - the difference is only that this way a fixture
    shorter than `window` still behaves.
    """
    return (flag.fillna(False).astype("float64")
            .rolling(int(window), min_periods=1).max() > 0.0)


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
def _validate(ema_window: int, deviation_atr_mult: float,
              stretch_lookback: int, k_period: int, d_period: int,
              smooth_k: int, oversold: float, overbought: float,
              stoch_lookback: int, sl_atr_mult: float,
              tp_atr_mult: float | None, trailing: bool) -> None:
    """
    Refuse a parameter set that cannot mean what it says, at the point it is
    passed rather than as a strange equity curve.

    Each of these has a specific failure behind it:

    * A non-positive period makes `.rolling()` raise from inside pandas, three
      frames from the caller who typed it.
    * `deviation_atr_mult <= 0` means "stretched" is satisfied by every bar
      that is merely on the wrong side of the average, which is half of them -
      the strategy stops being a deviation trade and nothing in the output
      says so.
    * `oversold >= overbought` inverts the confirmation: every bar is
      simultaneously in both extremes, so both sides confirm on every cross
      and the Stochastic filter silently becomes a coin flip.
    * A stop inside `MIN_STOP_ATR_MULT` sits inside the bar's own noise, and
      the backtest measures the cost model rather than the reversion.
    * `tp_atr_mult` may be None - that IS a setting, meaning "no target, run
      to the stop" - but a non-positive number is not: it would put the target
      at or behind the fill, closing every trade on its first bar.
    """
    for name, value in (("ema_window", ema_window),
                        ("stretch_lookback", stretch_lookback),
                        ("k_period", k_period), ("d_period", d_period),
                        ("smooth_k", smooth_k),
                        ("stoch_lookback", stoch_lookback)):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1; got {value!r}")
    if float(deviation_atr_mult) <= 0.0:
        raise ValueError(
            f"deviation_atr_mult must be > 0; got {deviation_atr_mult!r}. At "
            f"zero every bar on the wrong side of the EMA counts as "
            f"stretched and this stops being a deviation strategy.")
    if not 0.0 <= float(oversold) < float(overbought) <= 100.0:
        raise ValueError(
            f"need 0 <= oversold < overbought <= 100; got "
            f"oversold={oversold!r}, overbought={overbought!r}. Inverted, "
            f"every bar sits in both extremes and both sides confirm on "
            f"every cross.")
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
    never fires and the drawn line is a visible gap. A target at 10,000 x ATR
    would be a level the search could still, in principle, reach.
    """
    if tp_atr_mult is None:
        return float("nan")
    return float(tp_atr_mult)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
def _layers(bars: pd.DataFrame, ema_window: int, deviation_atr_mult: float,
            stretch_lookback: int, k_period: int, d_period: int,
            smooth_k: int, oversold: float, overbought: float,
            stoch_lookback: int, use_stretch_filter: bool,
            use_stoch_filter: bool) -> dict[str, pd.Series]:
    """
    Every series both `signal_fn` and `indicators` read, computed ONCE.

    One call site for the maths is the point: a second implementation living
    in the report would be free to disagree with this one and draw a reclaim a
    bar from where the trade actually fired, with nothing raising.

    The toggles are here rather than in `signal_fn` so that a disabled filter
    is a Series of True with the underlying series still computed and still
    drawn on the chart - a reader can see what the filter WOULD have said on
    the run where it was switched off.
    """
    close = bars["close"].astype("float64")
    ema = _ema(close, ema_window)
    atr = _atr(bars, ATR_PERIOD)
    percent_k, percent_d = _stochastic(bars, k_period, smooth_k, d_period)

    lower_band = ema - float(deviation_atr_mult) * atr
    upper_band = ema + float(deviation_atr_mult) * atr

    # THE STRETCH IS THE WICK, NOT THE CLOSE. The overshoot this harvests is
    # where price actually traded, and a bar can spike a full ATR multiple
    # through the band and close back inside it - that bar IS the event, and
    # reading `close` here would miss exactly the sharpest ones.
    stretched_below = bars["low"].astype("float64") < lower_band
    stretched_above = bars["high"].astype("float64") > upper_band

    stretch_long = (_recent(stretched_below, stretch_lookback)
                    if use_stretch_filter
                    else pd.Series(True, index=bars.index))
    stretch_short = (_recent(stretched_above, stretch_lookback)
                     if use_stretch_filter
                     else pd.Series(True, index=bars.index))

    # THE RECLAIM: a cross, not a state. Price has to actually come back
    # through the anchor on this bar; a run of bars already above it is not a
    # reclaim and would fire an entry on every one of them.
    reclaim_long = _cross_above(close, ema)
    reclaim_short = _cross_below(close, ema)

    # THE CONFIRMATION: %K crossing %D, having been in the extreme recently.
    # `_recent` on the extreme rather than a test on the cross bar itself,
    # because %K has usually already left the zone on the bar it crosses - a
    # strict `%K[i] <= oversold` would reject nearly every real signal.
    k_cross_up = _cross_above(percent_k, percent_d)
    k_cross_down = _cross_below(percent_k, percent_d)
    was_oversold = _recent(percent_k <= float(oversold), stoch_lookback)
    was_overbought = _recent(percent_k >= float(overbought), stoch_lookback)

    confirm_long = ((k_cross_up & was_oversold) if use_stoch_filter
                    else pd.Series(True, index=bars.index))
    confirm_short = ((k_cross_down & was_overbought) if use_stoch_filter
                     else pd.Series(True, index=bars.index))

    # THE TRIGGER FIRES ON THE LATER OF THE TWO EVENTS, not on their exact
    # coincidence. The reclaim and the Stochastic cross are independent
    # crossings of two different series and they land on the same bar only by
    # luck: on a representative 5m fixture the reclaim fired 19 times, the
    # confirmation 13, and they coincided TWICE. A strategy demanding
    # simultaneity is one that almost never trades, and its backtest would
    # report an honest-looking Sharpe over a handful of trades that means
    # nothing.
    #
    # So each condition may arrive within `stoch_lookback` bars of the other,
    # and the trigger is the bar on which the pair becomes COMPLETE -
    # whichever of the two that is. Written as a union of the two orderings
    # rather than as `recent(a) & recent(b)`, because the latter is a STATE
    # that stays true for the whole overlap and would re-trigger on every bar
    # of it.
    #
    # It remains an EVENT and it remains causal: `_recent` looks strictly
    # backwards, so a confirmation arriving BEFORE the reclaim is remembered
    # and one arriving AFTER it fires on its own bar. Nothing reads forward to
    # ask whether a cross is coming.
    reclaim_recent_long = _recent(reclaim_long, stoch_lookback)
    reclaim_recent_short = _recent(reclaim_short, stoch_lookback)
    confirm_recent_long = _recent(confirm_long, stoch_lookback)
    confirm_recent_short = _recent(confirm_short, stoch_lookback)

    trigger_long = ((reclaim_long & confirm_recent_long)
                    | (confirm_long & reclaim_recent_long))
    trigger_short = ((reclaim_short & confirm_recent_short)
                     | (confirm_short & reclaim_recent_short))

    return {
        "close": close, "ema": ema, "atr": atr,
        "lower_band": lower_band, "upper_band": upper_band,
        "percent_k": percent_k, "percent_d": percent_d,
        "stretched_below": stretched_below, "stretched_above": stretched_above,
        "stretch_long": stretch_long, "stretch_short": stretch_short,
        "reclaim_long": reclaim_long, "reclaim_short": reclaim_short,
        "confirm_long": confirm_long, "confirm_short": confirm_short,
        "trigger_long": trigger_long, "trigger_short": trigger_short,
        "k_cross_up": k_cross_up, "k_cross_down": k_cross_down,
        "dev_atr": (close - ema) / atr.replace(0.0, np.nan),
    }


# --------------------------------------------------------------------------
# The contract functions
# --------------------------------------------------------------------------
def signal_fn(bars: pd.DataFrame,
              ema_window: int = 20,
              deviation_atr_mult: float = 1.5,
              stretch_lookback: int = 5,
              k_period: int = 14,
              d_period: int = 3,
              smooth_k: int = 3,
              oversold: float = 20.0,
              overbought: float = 80.0,
              stoch_lookback: int = 3,
              use_stretch_filter: bool = True,
              use_stoch_filter: bool = True,
              use_stoch_exit: bool = False,
              sl_atr_mult: float = 1.5,
              tp_atr_mult: float | None = 1.0,
              trailing: bool = False) -> tuple[pd.Series, pd.Series,
                                               pd.Series, pd.Series]:
    """
    Returns the FOUR-MASK form: (long_entries, long_exits, short_entries,
    short_exits), all boolean Series on `bars.index`.

    A three-tuple or a bare Series RAISES in `engine.unpack_signals` rather
    than silently losing the short side into a plausible long-only curve.

    THE SHORT SIDE IS NOT THE LONG SIDE WITH A SIGN FLIP. Inside `_walk_loop`
    the short stop sits ABOVE the fill; a short stop placed below it would be
    breached by the fill bar itself. The masks are written out per side for
    the same reason - a reader has to be able to check them against the
    specification by eye.

    The entry is a bar-level EVENT (the reclaim cross and the Stochastic
    cross) inside standing conditions (the remembered stretch), so a long
    stretch of bars above the EMA produces a signal only where the cross
    fires. The walk enters only when FLAT: a second trigger while a position
    is open is ignored rather than pyramided, and an opposite trigger is
    ignored rather than reversing.
    """
    _validate(ema_window, deviation_atr_mult, stretch_lookback, k_period,
              d_period, smooth_k, oversold, overbought, stoch_lookback,
              sl_atr_mult, tp_atr_mult, trailing)

    L = _layers(bars, int(ema_window), float(deviation_atr_mult),
                int(stretch_lookback), int(k_period), int(d_period),
                int(smooth_k), float(oversold), float(overbought),
                int(stoch_lookback), bool(use_stretch_filter),
                bool(use_stoch_filter))

    long_ok = (L["stretch_long"] & L["trigger_long"]).fillna(False).to_numpy()
    short_ok = (L["stretch_short"]
                & L["trigger_short"]).fillna(False).to_numpy()

    if use_stoch_exit:
        # The opposite cross closes the position early. OFF by default: the
        # premise is a bracketed scalp, and an oscillator exit competing with
        # a 1.0 x ATR target changes what is being measured.
        long_sig_exit = L["k_cross_down"].fillna(False).to_numpy()
        short_sig_exit = L["k_cross_up"].fillna(False).to_numpy()
    else:
        long_sig_exit = np.zeros(len(bars), dtype=bool)
        short_sig_exit = np.zeros(len(bars), dtype=bool)

    atr = L["atr"].to_numpy(dtype="float64")
    # A bar with no ATR yet is not tradable: the bracket has no width. This is
    # the warm-up, and admitting it would place a stop at the fill price.
    warm = ~np.isfinite(atr) | (atr <= 0.0)
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
    The requested `generate_signals(df, params)` calling convention, as a thin
    adapter over `signal_fn`.

    THIS IS NOT THE ENGINE'S ENTRY POINT. `agents.tier3_workers.load_strategy`
    binds `make_signal_fn(**params)` or `signal_fn(bars, **params)` and knows
    nothing about this name, so a module exposing only this function could not
    be backtested at all. It exists because the request asked for it, and it
    delegates rather than reimplementing so the two can never disagree about
    what a signal is.

    Unknown keys RAISE here exactly as they do in `make_signal_fn`: a stale
    key that was silently dropped would run the DEFAULT and report it under
    the name the caller thought they set.
    """
    bound = {**DEFAULT_PARAMS, **(params or {})}
    unknown = set(bound) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    return signal_fn(df, **bound)


def indicators(bars: pd.DataFrame, **params) -> dict[str, pd.Series]:
    """
    Full-length series drawn over the trade inspector's candles, from the same
    `_layers` call `signal_fn` uses - so the chart cannot draw a reclaim a bar
    from where the entry actually happened.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["ema_window"]), float(p["deviation_atr_mult"]),
                int(p["stretch_lookback"]), int(p["k_period"]),
                int(p["d_period"]), int(p["smooth_k"]), float(p["oversold"]),
                float(p["overbought"]), int(p["stoch_lookback"]),
                bool(p["use_stretch_filter"]), bool(p["use_stoch_filter"]))
    dev = float(p["deviation_atr_mult"])
    return {
        f"EMA({int(p['ema_window'])})": L["ema"],
        f"EMA - {dev}xATR": L["lower_band"],
        f"EMA + {dev}xATR": L["upper_band"],
        f"ATR({ATR_PERIOD})": L["atr"],
        f"Stoch %K({int(p['k_period'])},{int(p['smooth_k'])})": L["percent_k"],
        f"Stoch %D({int(p['d_period'])})": L["percent_d"],
    }


def ml_features(bars: pd.DataFrame, **params) -> pd.DataFrame:
    """
    The matrix Version B's classifier is fitted on: one row per bar, in order.

    CAUSALITY IS THIS MODULE'S RESPONSIBILITY. Every column is built from
    `_layers`, which reads only bars <= i, and nothing here is scaled against
    the whole frame - a scaler fitted end to end leaks the test period's
    distribution into the training rows without tripping any shift-based
    audit. `HistGradientBoostingClassifier` needs no scaler, which is why none
    is fitted.

    SHAPE IS CHECKED BY THE CALLER AND A FAILURE RAISES - unlike `indicators`,
    which is wrapped, because a broken chart annotation must not throw away a
    completed backtest but a silently swapped feature matrix must not survive
    one.

    The columns are the state the RULES read, expressed as RATIOS rather than
    levels. A deviation of 12.5 points means nothing across contracts, and a
    classifier fitted on levels learns the symbol rather than the setup.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["ema_window"]), float(p["deviation_atr_mult"]),
                int(p["stretch_lookback"]), int(p["k_period"]),
                int(p["d_period"]), int(p["smooth_k"]), float(p["oversold"]),
                float(p["overbought"]), int(p["stoch_lookback"]),
                bool(p["use_stretch_filter"]), bool(p["use_stoch_filter"]))
    close = L["close"]
    atr = L["atr"].replace(0.0, np.nan)
    volume = (bars["volume"].astype("float64") if "volume" in bars.columns
              else pd.Series(np.nan, index=bars.index))
    vol_sma = volume.rolling(20, min_periods=20).mean().shift(1)
    vol_std = volume.rolling(20, min_periods=20).std().shift(1)
    out = pd.DataFrame({
        "dev_atr": L["dev_atr"],
        "norm_atr": atr / close,
        "percent_k": L["percent_k"],
        "percent_d": L["percent_d"],
        "k_minus_d": L["percent_k"] - L["percent_d"],
        "dist_lower_atr": (close - L["lower_band"]) / atr,
        "dist_upper_atr": (close - L["upper_band"]) / atr,
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
        # Raise rather than ignore. A stale grid key that was silently dropped
        # would sweep the DEFAULT and report it under the swept name.
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")

    def _fn(bars: pd.DataFrame):
        return signal_fn(bars, **bound)
    return _fn
