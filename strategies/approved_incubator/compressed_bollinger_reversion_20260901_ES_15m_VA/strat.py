"""
Compressed Bollinger Reversion — a Q4 range fade in three layers: an ADX floor
refuses a directional tape, a close pushing outside a compressed Bollinger band
is the entry, and the position is taken off at the central SMA rather than at a
fixed ATR distance.

Location:
    ~/src/trading/strategies/experimental/compressed_bollinger_reversion_20260901.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding — do not edit this block to match it.

    1. FILE
       strategies/experimental/compressed_bollinger_reversion_20260901.py

    2. CORE LOGIC & MATH
       Regime Filter: ADX(14) < 20 (confirming absence of directional trend).
       Long Entry:  Close crosses below Lower Bollinger Band
           (SMA(bb_len) - bb_mult * std) AND RSI(rsi_len) < rsi_thresh.
       Short Entry: Close crosses above Upper Bollinger Band
           (SMA(bb_len) + bb_mult * std) AND RSI(rsi_len) > (100 - rsi_thresh).
       Exit Logic:  Position closes when Close crosses the central SMA(bb_len)
           midline (dynamic take-profit) OR hits a hard stop loss at
           sl_atr_mult * ATR(14).

    3. PARAMETER GRID (108 cells)
       bb_len (20, 30, 40); bb_mult (1.8, 2.0, 2.2); rsi_thresh (25, 30, 35);
       rsi_len (7, 14); sl_atr_mult (1.5, 2.0);
       trailing = False (fixed); tp_atr_mult = None (fixed)

    4. METADATA & FRAMEWORK HOOKS
       TARGET_QUADRANTS = ("Q4",);  PORTFOLIO_GROUP = "Range_Fade"
       The signal array must handle the dynamic SMA midline crossing as the
       exit trigger for the simulator loop.

    5. TESTING
       tests/test_compressed_bollinger_reversion_20260901.py

THE GRID IS 108 CELLS, AND THAT IS THE LABEL AND THE ARITHMETIC AGREEING
========================================================================
`3 x 3 x 3 x 2 x 2 = 108`; `trailing` and `tp_atr_mult` are pinned at one
value each and multiply nothing. Read it against the ladder before quoting it:
`--tf 5m,15m,30m,1h` is 108 fits PER timeframe PER contract, so 432 per symbol
and 1,728 across the four declared assets. `variants_tested` carries whichever
count actually ran onto every artifact, so the Sharpe this grid reports can be
read against the search that produced it.

ONE LINE OF THE REQUEST IS ANSWERED DIFFERENTLY
===============================================
**There is no `resolve_strategy()` to implement here, and `baseline.py` is not
an API.** `resolve_strategy` lives in `backtest/run.py` and maps a strategy
NAME to a module PATH — it is the runner's loader, not something a strategy
declares; a copy in this file would be dead code that shadows nothing.
`baseline.py` is not an interface either: `promote.py` produces it with
`shutil.copyfile(source, baseline_py)`, so it is a verbatim copy of THIS file
taken at promotion. The way to "implement the baseline API" is to get this
module's own contract right — `signal_fn`, `indicators`, `LOGIC`,
`PARAM_GRID`, `make_signal_fn`, `ml_features` — and the promoted baseline is
then correct because it IS this module.

WHAT THIS MODULE CANNOT MODEL, SAID OUT LOUD
============================================
**There are no stop or target ORDERS anywhere in this engine.**
`vbt.Portfolio.from_signals` is driven by boolean masks and fills them at the
NEXT bar's open. The hard stop below is therefore detected on the bar whose low
breaches it and filled at the following open — *not at the stop price*.

That gap matters more here than in a trend strategy, and the arithmetic says
why. This strategy's gross target is the distance from the band to the midline,
`bb_mult * sigma` — and sigma is small BY CONSTRUCTION in a low-volatility
quadrant, because that is what Q4 means. Against it stand mandatory costs of
one tick of slippage each way plus commission: on ES that is $29.58 a round
turn, or 0.59 index points; on NQ $14.58, or 0.73 points. **Before sweeping
this grid, measure the median `bb_mult * sigma` over Q4-designated bars per
(symbol, timeframe) and divide it by that round trip.** A pair whose median
target does not clear its costs by a comfortable multiple is not measuring an
edge, it is measuring the cost model — which is the same arithmetic that
removed 1m/2m/3m from the default ladder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

#: The bar this was designed on. Stage 1 screens the ladder and the winning
#: (tf, params) pair is the claim - never the parameters alone.
TIMEFRAME = "30m"

#: The request's index range-fade basket, spelled as contracts this lake serves
#: and `backtest/specs.py` has verified.
SYMBOLS = ["ES", "NQ"]

STRATEGY_NAME = "COMPRESSED_BOLLINGER_REVERSION"
STRATEGY_MODULE = "compressed_bollinger_reversion_20260901"

#: Portfolio routing metadata. Descriptive only - `config/portfolios.json` is
#: the authority on which account trades what, and no basket named here exists
#: until somebody adds it there.
PORTFOLIO_GROUP = "Range_Fade"
CORRELATION_PROFILE = (
    "Uncorrelated / diversifier against the index trend book. It fades the "
    "chop that a trend follower pays for, so its losing environment is the "
    "other book's winning one - a fact for portfolio routing to size on, not "
    "a claim that either leg hedges the other.")

#: The regime the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and nothing in this module reads these.
#:
#: The request's "Q4 (Low-Vol / Ranging)" agrees with `mdlib/regimes.py`,
#: which is the only authority: Q1 High-Vol/Trending, Q2 High-Vol/Ranging,
#: Q3 Low-Vol/Trending, Q4 Low-Vol/Ranging, 0 UNDEFINED (warm-up, not a
#: quadrant). The ids are pinned against `backtest.profiler` in the test suite
#: so a future edit cannot drift them.
TARGET_REGIMES = ("Low Volatility / Ranging",)
TARGET_QUADRANTS = ("Q4",)

ATR_PERIOD = 14
ADX_PERIOD = 14

#: The request's directional floor. Note what it is NOT: the regime system
#: calls a bar Trending on `ADX > 25` - strictly greater, so a bar at exactly
#: 25.00000 is Ranging - and Q4 is Low-Vol AND not trending. This module's
#: 20 is therefore a SECOND, STRICTER boundary sitting inside a strategy whose
#: verdict is measured on the first: bars with ADX in (20, 25] are Q4 by the
#: certification's definition and refused here.
#:
#: That is a real cost, not a redundancy, because Gate R certifies on >= 30
#: holdout trades inside Q4 and this filter spends from that budget. It is
#: kept because the request names it, and it is behind `use_regime_filter` so
#: Stage 2 can price what it costs.
ADX_RANGE_MAX = 20.0

#: An RSI whose gain and loss both average zero is 0/0 - undefined rather than
#: oversold. Neutral is the honest reading, and it keeps a flat stretch of bars
#: from registering as an extreme.
RSI_NEUTRAL = 50.0

#: A stop nearer than this to the fill sits inside the bar's own noise.
MIN_STOP_ATR_MULT = 0.25

DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_mult": 2.0,
    "rsi_thresh": 30.0,
    "rsi_len": 14,
    "use_regime_filter": True,
    "use_alpha_trigger": True,
    "use_midline_exit": True,
    "use_news_filter": False,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": None,
    "trailing": False,
}

#: The search space `backtest/run.py --scan` and `backtest/scan.py` sweep.
#:
#:     3 x 3 x 3 x 2 x 2 = 108 combinations
#:
#: `tp_atr_mult` is pinned None and `trailing` pinned False, and both are the
#: premise rather than an omission. The take-profit is the MIDLINE, which is a
#: moving level and cannot be expressed as an ATR distance from the fill - it
#: is carried as a signal-exit mask instead (see `signal_fn`). A trailing stop
#: would follow price away from the mean, which is the opposite of a fade.
#: Both are still declared so a single out-of-band run can ask the question the
#: grid does not: `--param tp_atr_mult=1.0`.
#: MEASURED BEFORE THE FIRST BACKTEST, on synthetic mean-reverting frames:
#: `rsi_thresh` DOMINATES this grid's sample size and the other four axes do
#: not come close. The band touch and the RSI extreme are nearly disjoint at
#: 30 and only overlap properly at 35 -
#:
#:     rsi_thresh   band touches   RSI extremes   BOTH   trades
#:             30             29             11      6       13
#:             35             29             26     11       25
#:
#: because a 2-sigma push in a COMPRESSED band is small in absolute terms and
#: does not generate the directional momentum RSI(14) needs to reach 30. The
#: two conditions measure the same move at different speeds.
#:
#: The consequence is a Gate R problem, not a tuning preference: certification
#: needs >= 30 holdout trades inside Q4, and the `rsi_thresh=25` cells may not
#: produce them at all on a real tape. Read `variants_tested` alongside the
#: per-cell trade counts before believing any cell at 25, and expect the 35
#: column to be where the sample lives.
PARAM_GRID = {
    "bb_len": [20, 30, 40],
    "bb_mult": [1.8, 2.0, 2.2],
    "rsi_thresh": [25.0, 30.0, 35.0],
    "rsi_len": [7, 14],
    "sl_atr_mult": [1.5, 2.0],
    "tp_atr_mult": [None],
    "trailing": [False],
}

LOGIC = {
    "concept": (
        "A fade of the outer band in a market with no direction to fight. "
        "ADX(14) below {adx_range_max} says the tape is not trending; a close "
        "pushing outside SMA({bb_len}) +/- {bb_mult} x sigma with "
        "RSI({rsi_len}) at an extreme is the entry; and the position is taken "
        "off at the central SMA rather than at a fixed distance, because in a "
        "compressed regime the mean is the only level the move is actually "
        "reverting TO."),
    "entry": (
        "LONG when ADX(14) < {adx_range_max}, Close crosses BELOW "
        "SMA({bb_len}) - {bb_mult} x sigma and RSI({rsi_len}) < "
        "{rsi_thresh}. SHORT mirrors it: Close crosses ABOVE "
        "SMA({bb_len}) + {bb_mult} x sigma and RSI({rsi_len}) > "
        "100 - {rsi_thresh}."),
    "exit": (
        "Close crossing the SMA({bb_len}) midline - UP for a long, DOWN for a "
        "short - or the hard ATR stop at {sl_atr_mult} x ATR(14), whichever "
        "the bar reaches first. No static take-profit is modelled "
        "(tp_atr_mult={tp_atr_mult}); the midline IS the target."),
}


# ---------------------------------------------------------------------------
# Indicator kernels
#
# DUPLICATED VERBATIM from `double_rsi_momentum_pullback_20260830.py` (_wilder,
# _rsi, _true_range, _atr, _as_series, _cross_above, _cross_below, _walk_loop),
# `sma_momentum_crossover_20260818.py` (_adx) and `t3_braid_scalp_20260823.py`
# (_bar_timestamps). Strategy modules are loaded from a FILE PATH by
# `agents.tier3_workers.load_strategy` and promoted as a self-contained copy,
# so a shared import would resolve against whatever happens to sit beside the
# module at load time - and a promoted package must reproduce the file that was
# certified, byte for byte, not whatever a helper module has become since.
#
# `_adx` is taken from `sma_momentum_crossover_20260818` DELIBERATELY rather
# than rewritten: that copy is validated against TA-Lib by its own suite, and
# its docstring already works through what a four-point warm-up disagreement
# does near a threshold of 20 - which is exactly this module's threshold.
# ---------------------------------------------------------------------------
def _wilder(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing — the average the RSI and the ATR are actually defined
    on.

    `alpha = 1/period` with `adjust=False` is Wilder's recursion exactly. A
    span-`period` EMA (`_ema` above) is a different, roughly twice as fast,
    average, and using it here would produce an "RSI(14)" and an "ATR(14)" that
    no other tool agrees with — so a reader checking a stop distance or an
    oversold reading against their own chart would see it somewhere else.

    THE RECURSION IS SEEDED BY THE EWM, NOT BY WILDER'S SMA. Wilder's original
    formulation seeds the first average with a simple mean of the first
    `period` values and recurses from there; `ewm(adjust=False)` starts the
    recursion at the first observation. The two converge within a few dozen
    bars and differ only through the warm-up, and this form is what
    `sma_momentum_crossover_20260818` and `t3_braid_scalp_20260823` already use
    for their ATRs — one convention across the directory is worth more than a
    closer match to one vendor's seeding, and the test suite pins THIS form
    against a hand-run recursion rather than against a chart.
    """
    return series.ewm(alpha=1.0 / period, adjust=False,
                      min_periods=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI, computed as `100 * avg_gain / (avg_gain + avg_loss)`.

    That expression is ALGEBRAICALLY IDENTICAL to the textbook
    `100 - 100 / (1 + avg_gain/avg_loss)` wherever the denominator is
    non-zero — expand it and the two agree exactly — and it is written this way
    because the textbook form divides by `avg_loss`, which is zero on any
    window with no down close. A market that only rose has an RSI of 100, not
    an infinity or a NaN, and this form produces the 100 directly instead of
    relying on the division's overflow and a fix-up afterwards.

    THE DEAD-FLAT WINDOW IS 50, NOT 0. When gains and losses are both zero the
    ratio is 0/0 and the oscillator is genuinely undefined. `RSI_NEUTRAL`
    resolves it to 50, the centerline, because a window in which nothing moved
    is neither overbought nor oversold — and because 0.0, which a naive
    "fill degenerate windows with 0.0" reading would give, is the most oversold
    value the indicator has: it would put `fast_rsi <= 30` permanently true
    through every halted or dead session and manufacture long triggers out of
    silence. The fill is applied ONLY where the underlying averages exist, so
    warm-up stays NaN.

    CAUSAL: `.diff()` and Wilder's recursion look strictly backwards.
    """
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    total = avg_gain + avg_loss
    # `.where(total > 0)` makes the flat window NaN rather than 0/0, and the
    # explicit fill then puts the neutral 50 in only where both averages are
    # real. A `total == 0` row inside the warm-up stays NaN either way, since
    # `total` is NaN there.
    rsi = 100.0 * avg_gain / total.where(total > 0)
    return rsi.where(total.isna() | (total > 0), RSI_NEUTRAL)


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
    A window in which nothing moved at all — a halted contract, a dead
    overnight hour — has a true ATR of zero. The request's "fill degenerate
    windows with 0.0" is satisfied downstream rather than here:
    `_signal_arrays` passes `np.nan_to_num(atr, nan=0.0)` into the walk, so
    warm-up NaN becomes a zero stop DISTANCE on bars where `ready` already
    forbids an entry, and a genuinely zero ATR on a live bar produces a stop at
    the fill price that the fill bar itself breaches. Both are visible as an
    immediate exit rather than as a NaN level nothing ever breaches, which is
    what a NaN stop distance would silently become.
    """
    return _wilder(_true_range(bars), period)


def _as_series(value, index) -> pd.Series:
    """A scalar level as a Series on `index`; a Series is returned as-is."""
    if isinstance(value, pd.Series):
        return value
    return pd.Series(float(value), index=index)


def _cross_above(fast: pd.Series, level) -> pd.Series:
    """
    True on the bar `fast` crosses UP through `level` — a series or a scalar.

    The definition, written out because "cross" is used loosely elsewhere and
    the loose reading is a different strategy:

        fast[i] > level[i]  AND  fast[i-1] <= level[i-1]

    A STATE (`fast > level`) would be true for every bar of a run; this is the
    EVENT, true once. `<=` on the previous bar rather than `<` so a pair that
    was exactly equal and then separated counts as a cross — with two
    oscillators quantised by the same price series, exact equality is not the
    measure-zero event it is for two continuous curves.

    NaN-SAFE BY CONSTRUCTION: every comparison against NaN is False, so a warm-
    up bar is never a cross. `.shift(1)` looks one bar BACKWARD — nothing here
    reads bar i+1.
    """
    other = _as_series(level, fast.index)
    return (fast > other) & (fast.shift(1) <= other.shift(1))


def _cross_below(fast: pd.Series, level) -> pd.Series:
    """
    True on the bar `fast` crosses DOWN through `level`. The exact mirror of
    `_cross_above`, written out rather than expressed as its negation: `~cross_
    above` is true on every bar that merely fails to be a cross up, which is
    almost all of them.
    """
    other = _as_series(level, fast.index)
    return (fast < other) & (fast.shift(1) >= other.shift(1))




def _rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI, computed as `100 * avg_gain / (avg_gain + avg_loss)`.

    That expression is ALGEBRAICALLY IDENTICAL to the textbook
    `100 - 100 / (1 + avg_gain/avg_loss)` wherever the denominator is
    non-zero — expand it and the two agree exactly — and it is written this way
    because the textbook form divides by `avg_loss`, which is zero on any
    window with no down close. A market that only rose has an RSI of 100, not
    an infinity or a NaN, and this form produces the 100 directly instead of
    relying on the division's overflow and a fix-up afterwards.

    THE DEAD-FLAT WINDOW IS 50, NOT 0. When gains and losses are both zero the
    ratio is 0/0 and the oscillator is genuinely undefined. `RSI_NEUTRAL`
    resolves it to 50, the centerline, because a window in which nothing moved
    is neither overbought nor oversold — and because 0.0, which a naive
    "fill degenerate windows with 0.0" reading would give, is the most oversold
    value the indicator has: it would put `fast_rsi <= 30` permanently true
    through every halted or dead session and manufacture long triggers out of
    silence. The fill is applied ONLY where the underlying averages exist, so
    warm-up stays NaN.

    CAUSAL: `.diff()` and Wilder's recursion look strictly backwards.
    """
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    total = avg_gain + avg_loss
    # `.where(total > 0)` makes the flat window NaN rather than 0/0, and the
    # explicit fill then puts the neutral 50 in only where both averages are
    # real. A `total == 0` row inside the warm-up stays NaN either way, since
    # `total` is NaN there.
    rsi = 100.0 * avg_gain / total.where(total > 0)
    return rsi.where(total.isna() | (total > 0), RSI_NEUTRAL)



def _adx(bars: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """
    Wilder's ADX. NaN until roughly `2 * period` bars exist.

    The construction, written out because "ADX" names several different
    calculations in the wild and only one of them is Wilder's:

        +DM  = high - prev_high, when that exceeds both (prev_low - low) and 0
        -DM  = prev_low - low,   when that exceeds both (high - prev_high) and 0
        +DI  = 100 * wilder(+DM) / wilder(TR)
        -DI  = 100 * wilder(-DM) / wilder(TR)
        DX   = 100 * |+DI - -DI| / (+DI + -DI)
        ADX  = wilder(DX)

    DIRECTION-AGNOSTIC BY CONSTRUCTION. `DX` takes the absolute difference of
    the two directional indicators, so a strong downtrend and an equally strong
    uptrend produce the same ADX. That is why one threshold can govern both
    sides of this strategy without smuggling in a directional bias — and it is
    also why ADX alone cannot pick a side, which is the macro baseline's job.

    CAUSAL. Both `.shift(1)` calls look one bar BACKWARD, `_wilder` is an
    exponential recursion over past values only, and nothing here reads a bar
    later than i. The double smoothing is why the warm-up is ~2 x period: the
    directional movement is smoothed once and the resulting DX is smoothed
    again, so ADX(14) does not exist until about bar 27.

    VALIDATED AGAINST TA-LIB, and the disagreement is worth knowing about.
    `tests/test_sma_momentum_crossover.py` checks this against `talib.ADX` on
    random walks: by ~300 bars in the two agree to within 1e-6 ADX points and
    by ~400 to within 1e-9, converging rather than ever becoming bit-identical.
    Before that they differ by as much as 4 points, because TA-Lib seeds
    Wilder's recursion with a simple
    sum over the first `period` values while `_wilder`'s `ewm(adjust=False)`
    seeds from the first value. The gap decays exponentially and neither
    seeding is wrong, but near a threshold of 20 four points decides the gate —
    so in the first hundred-odd bars after ADX comes into existence this filter
    can open or close on a bar where a chart package would not agree. That is a
    warm-up artefact, not a lookahead: both forms read only past bars.

    TWO DEGENERATE STATES ARE MEASUREMENTS, NOT GAPS, and both are filled
    explicitly. `+DI + -DI == 0` is a window that produced no directional
    movement either way; a smoothed true range of 0 is a window in which
    nothing moved at all. Both mean "no trend", which is DX 0. Left as the 0/0
    NaN the division produces, either would propagate through the second
    smoothing and blank out ADX for the next `period` bars — silently closing
    the entry gate long after the flat patch ended. TA-Lib returns 0.0 for the
    zero-range case and this agrees with it. The fills are conditioned on the
    inputs existing, so warm-up stays NaN.
    """
    high, low = bars["high"], bars["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = _wilder(_true_range(bars), period)
    # A zero ATR means no range at all over the window; the DIs are then
    # undefined rather than infinite, and are left NaN for the DX fill below.
    safe_atr = atr.where(atr > 0)
    plus_di = 100.0 * _wilder(plus_dm, period) / safe_atr
    minus_di = 100.0 * _wilder(minus_dm, period) / safe_atr

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum > 0)
    # Both DIs exist and neither side moved: that is DX 0 ("no trend"), which
    # is a measurement, not a gap.
    dx = dx.mask(plus_di.notna() & minus_di.notna() & (di_sum == 0), 0.0)
    # And a window with no range at all, where the DIs are undefined rather
    # than zero. `atr.notna()` keeps warm-up out of this: it is the smoothed
    # range EXISTING and being zero that means nothing moved.
    dx = dx.mask(atr.notna() & (atr <= 0), 0.0)
    return _wilder(dx, period)


def _bar_timestamps(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """
    The bar timestamps as a tz-AWARE UTC index, from the `ts` column or a
    DatetimeIndex.

    The engine hands strategies a long-format frame with `ts` as a COLUMN and a
    positional index, so `bars.index.hour` raises there. A caller holding a
    time-indexed fixture has the opposite shape. Accepting both keeps this
    usable from either without silently producing garbage from one.

    Always tz-aware, never naive: a naive index is localized to UTC, which is
    what the lake stores. Pandas 3.0 raises on comparing or converting a naive
    index against an aware one, so an unlocalized frame would fail at the
    `tz_convert` in `ml_features` rather than silently being read as ET.
    """
    if "ts" in bars.columns:
        return pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    if isinstance(bars.index, pd.DatetimeIndex):
        idx = bars.index
        return idx if idx.tz is not None else idx.tz_localize("UTC")
    # `type(bars.index)` rather than its `__name__`: the AST validator flags
    # every dunder access, and this module's exception from `ALLOWED_IMPORTS`
    # is granted for the `backtest.event_calendar` import and nothing else —
    # the test suite pins that objection list at exactly one entry, so a
    # cosmetic `__name__` here would spend the exception on a error message.
    raise ValueError(
        "bars needs a `ts` column or a DatetimeIndex; got an index of type "
        f"{type(bars.index)} and columns {list(bars.columns)}")



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
    THREE-state machine over the bars: flat, long, or short — each side under
    its own stop, target and signal exit.

    This kernel is DUPLICATED VERBATIM from
    `sma_momentum_crossover_20260818.py`, which shares it with
    `t3_braid_scalp_20260823.py`, `ema_crossover_20260821.py` and
    `ema_trend_filter.py`, by the same convention that duplicates `_wilder` and
    `_atr` across this directory: strategy modules are loaded from a file path
    and are deliberately self-contained. `tests/test_risk_params.py` runs the
    older copies on identical arrays in both directions and requires identical
    output, and `tests/test_double_rsi_macd_scalp_20260823.py` holds THIS copy
    to the same standard against `sma_momentum_crossover_20260818._walk` — do
    not "improve" one alone.

    UNLIKE `t3_braid_scalp_20260823`, THIS STRATEGY ACTUALLY USES THE
    `sig_exit` ARRAYS. They carry Layer 4's RSI-extreme and 200-EMA crosses, so
    a position here can be closed by the signal as well as by the bracket, and
    the branch that module leaves permanently False is load-bearing here.

    A trailing stop cannot be a stateless mask. Its level is the extreme price
    since ENTRY offset by a fixed distance, so bar i's exit condition depends on
    which earlier bar opened the position — which depends on every entry before
    it. The fixed stop and the take-profit are anchored on the FILL PRICE, which
    is `open` on the bar after the signal, so they are path-dependent for the
    same reason. This walks the bars once and resolves entries, stops, targets,
    signal exits and the session flatten together rather than splitting them
    across layers that could disagree.

    The timeline matches the engine's. An entry signal on bar i is filled at
    bar i+1's open, so the position is live from bar i+1, the fill price is
    `open_[i + 1]`, and the extreme-price mark starts there — NOT on the signal
    bar. Both distances are frozen at `mult * ATR` as measured on the SIGNAL bar
    and never re-measured as volatility changes.

    The two sides, written out rather than folded into a sign flip, because a
    reader has to be able to check them against the specification by eye:

        LONG    stop     fill - dist    trailing: (highest high since fill) - dist
                target   fill + dist
                exits    low  <= stop   or  high >= target
        SHORT   stop     fill + dist    trailing: (lowest low   since fill) + dist
                target   fill - dist
                exits    high >= stop   or  low  <= target

    The trailing stop never widens on either side: it ratchets UP behind a long
    and DOWN in front of a short, tracking the best price the position has seen.

    Both sides also exit on their own `sig_exit` mask and on `flat_bar`.

    `sl_mult` and `tp_mult` are shared by the two sides. That is a modelling
    choice, not an oversight: a strategy whose short stop is a different width
    from its long stop is two strategies sharing a name, and the sweep could not
    tell which side a winning cell belongs to.

    `tp_mult` is NaN when no take-profit is modelled, rather than a sentinel like
    0 or a huge number. NaN propagates into `target` and every comparison against
    it is False, so the target simply never fires on either side and the drawn
    line is a gap — a take-profit at 10,000 x ATR would be a line a reader could
    see and a level the search could still, in principle, reach.

    A bar carrying BOTH a long and a short entry signal while flat takes
    NEITHER, matching `backtest.engine._clean_signals_ls_loop`. The strategy has
    asked to be long and short at once and choosing a side here would bury a coin
    flip inside the kernel. A well-formed strategy cannot produce it — the slow
    RSI cannot be both above and below 50 — so the branch exists to make a
    malformed one visible as a missing trade rather than a plausible one-sided
    curve.

    A position is never reversed directly. The walk enters only from flat, so a
    short signal arriving while long is ignored; the long must exit first, on its
    stop, its target, its signal exit or the bell.

    Exits are checked from the fill bar onward, never on the signal bar itself.

    Returns `(long_entries, long_exits, short_entries, short_exits, stop_level,
    tp_level)`. Both levels are live for every bar a position is open, on
    whichever side it is open, and NaN everywhere else — they are what the tear
    sheet draws, so the lines a reader sees breached are the arrays the exits
    were taken from rather than a second reconstruction of them.
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
            # expressed. Every comparison against NaN is False in both numba and
            # numpy, so the interpreted fallback cannot disagree with the
            # compiled loop about it.
            hit_tp = high[i] >= target

            # A bar that breaches both produces one exit on this bar either way,
            # and the engine fills it at the next bar's open regardless — so the
            # intrabar race between the stop and the target is not resolved here
            # because it cannot change the result. On those bars the modelled
            # fill is neither the stop price nor the target price.
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

    # No `cache=True`. `load_strategy` imports this module from a file path, so
    # it is not importable by name. Compiling still works and numba writes the
    # cache; it is the LOAD in a later process that fails, with
    # `ModuleNotFoundError: No module named '<dynamic>'`, which is why the
    # first run after a cache wipe looks clean and the second one dies.
    # `backtest/engine.py` keeps its cache because it is imported normally.
    _walk = njit(nogil=True)(_walk_loop)
except ImportError:                     # pragma: no cover - env dependent
    # Same function, interpreted. `backtest.engine.clean_signals` degrades the
    # same way, and the fallback has to exist because a missing compiler must
    # not change which trades a strategy takes.
    _walk = _walk_loop


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _validate(bb_len: int, bb_mult: float, rsi_thresh: float, rsi_len: int,
              sl_atr_mult: float, tp_atr_mult: float | None,
              trailing: bool) -> None:
    """
    Refuse a parameter set this module cannot describe honestly.

    A REJECTED cell is still counted in `variants_tested` - `scan.py` counts
    the combination it evaluated, not the ones that survived.
    """
    for name, value, floor in (("bb_len", bb_len, 2), ("rsi_len", rsi_len, 2)):
        if not isinstance(value, (int, np.integer)) or int(value) < floor:
            raise ValueError(
                f"{name} must be an integer >= {floor}; got {value!r}")
    if not np.isfinite(bb_mult) or float(bb_mult) <= 0.0:
        raise ValueError(f"bb_mult must be > 0; got {bb_mult!r}")
    if not np.isfinite(rsi_thresh) or not 0.0 < float(rsi_thresh) < 50.0:
        # Strictly below 50 because the SHORT side reads `100 - rsi_thresh`.
        # At 50 the two thresholds meet and both sides fire on the same bar,
        # which the walk resolves by taking NEITHER - a silently dead cell.
        raise ValueError(
            f"rsi_thresh must be in (0, 50) so the long and short thresholds "
            f"stay apart; got {rsi_thresh!r}")
    if not np.isfinite(sl_atr_mult) or float(sl_atr_mult) < MIN_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be >= {MIN_STOP_ATR_MULT}; got {sl_atr_mult!r}")
    if tp_atr_mult is not None:
        if not np.isfinite(tp_atr_mult) or float(tp_atr_mult) <= 0.0:
            raise ValueError(
                f"tp_atr_mult must be > 0 or None; got {tp_atr_mult!r}")
    if not isinstance(trailing, (bool, np.bool_)):
        raise ValueError(f"trailing must be a bool; got {trailing!r}")
    if trailing and tp_atr_mult is None:
        # A fade whose stop follows price away from the mean is not this
        # strategy. Refused rather than silently allowed, because the midline
        # exit would still fire and the result would look almost right.
        raise ValueError(
            "trailing=True with no take-profit is not a mean reversion: the "
            "stop would ratchet away from the mean the trade is fading TO")


def _tp_distance(tp_atr_mult: float | None) -> float:
    """
    `None` means NO static target, expressed as NaN.

    NaN rather than 0.0, which would place the target at the fill and close
    every trade on its own entry bar - and NaN rather than a huge number,
    which is a level the search could in principle reach. Every comparison
    against NaN is False in both the compiled and the interpreted walk.

    THE REAL TARGET IS THE MIDLINE, and it is not expressible here: it moves
    every bar, while this distance is frozen at the fill. It travels as a
    signal-exit mask instead - see `signal_fn`.
    """
    return float("nan") if tp_atr_mult is None else float(tp_atr_mult)


def _news_suppress(bars: pd.DataFrame,
                   long_ok: np.ndarray,
                   short_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop candidate triggers whose FILL bar lands inside a macro-release window.

    Delegates to `backtest.event_calendar.apply_entry_filters`, the only
    implementation of this filter in the repository. The mask is widened one
    bar BACKWARDS there, because the engine fills at the next bar's open and a
    signal is judged on the bar it FILLS; and an empty calendar RAISES rather
    than returning an all-clear, so a run reported as news-filtered cannot be
    one in which nothing was ever filtered.
    """
    from backtest.event_calendar import apply_entry_filters
    return apply_entry_filters(bars, long_ok, short_ok)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
def _layers(bars: pd.DataFrame, bb_len: int, bb_mult: float,
            rsi_thresh: float, rsi_len: int, use_regime_filter: bool,
            use_alpha_trigger: bool) -> dict:
    """
    Every series the signals, the chart and the ML matrix are built from,
    computed ONCE and shared - so the inspector cannot draw a band touch a bar
    from where the entry fired and the classifier cannot be trained on a
    slightly different strategy.

    EVERY COLUMN READS ONLY BARS <= i. The crossings use `.shift(1)`, which
    looks one bar BACKWARD; the rolling windows are trailing; `_wilder` is a
    causal recursion. There is no centred window and no `.shift(-1)` anywhere
    in this module.
    """
    close = bars["close"].astype("float64")
    n = int(bb_len)
    mid = close.rolling(n, min_periods=n).mean()
    # ddof=0: the POPULATION standard deviation of the window, which is what
    # every charting package draws a Bollinger band from. ddof=1 would widen
    # the band by sqrt(n/(n-1)) - about 2.6% at n=20 - and put the touch on a
    # different bar from the one a reader checking against their chart sees.
    sigma = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + float(bb_mult) * sigma
    lower = mid - float(bb_mult) * sigma

    rsi = _rsi(close, int(rsi_len))
    adx = _adx(bars, ADX_PERIOD)
    atr = _atr(bars, ATR_PERIOD)

    if use_regime_filter:
        # `< ADX_RANGE_MAX`, and NaN is False, so the ADX warm-up (~2 x period)
        # is closed rather than open. A NaN read as "not trending" would open
        # the gate for the first ~27 bars of every frame.
        regime_ok = (adx < ADX_RANGE_MAX).fillna(False)
    else:
        regime_ok = pd.Series(True, index=bars.index)

    if use_alpha_trigger:
        touch_long = _cross_below(close, lower)
        touch_short = _cross_above(close, upper)
    else:
        # Without the band trigger the strategy is the oscillator alone. It is
        # expressed as the RSI crossing its own threshold - an EVENT - because
        # the STATE (`rsi < thresh`) is true for every bar of a run and would
        # make the toggle turn every oversold bar into an entry.
        touch_long = _cross_below(rsi, float(rsi_thresh))
        touch_short = _cross_above(rsi, 100.0 - float(rsi_thresh))

    osc_long = (rsi < float(rsi_thresh)).fillna(False)
    osc_short = (rsi > 100.0 - float(rsi_thresh)).fillna(False)

    return {
        "close": close, "mid": mid, "sigma": sigma,
        "upper": upper, "lower": lower,
        "rsi": rsi, "adx": adx, "atr": atr,
        "norm_atr": (atr / close.replace(0.0, np.nan)),
        "regime_ok": regime_ok,
        "touch_long": touch_long.fillna(False),
        "touch_short": touch_short.fillna(False),
        "osc_long": osc_long, "osc_short": osc_short,
        "ts": _bar_timestamps(bars),
    }


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def signal_fn(bars: pd.DataFrame,
              bb_len: int = 20,
              bb_mult: float = 2.0,
              rsi_thresh: float = 30.0,
              rsi_len: int = 14,
              use_regime_filter: bool = True,
              use_alpha_trigger: bool = True,
              use_midline_exit: bool = True,
              use_news_filter: bool = False,
              sl_atr_mult: float = 1.5,
              tp_atr_mult: float | None = None,
              trailing: bool = False) -> tuple[pd.Series, pd.Series,
                                               pd.Series, pd.Series]:
    """
    Returns the FOUR-MASK form: (long_entries, long_exits, short_entries,
    short_exits), all boolean Series on `bars.index`.

    A three-tuple or a bare Series RAISES in `engine.unpack_signals` rather
    than silently losing the short side into a plausible long-only curve.

    THE MIDLINE IS THE TAKE-PROFIT, AND IT TRAVELS AS A SIGNAL-EXIT MASK.
    This is the request's "dynamic SMA midline crossing as the exit trigger
    for the simulator loop", and it cannot be expressed any other way here:
    `_walk_loop`'s target is a distance frozen at the fill, while the midline
    moves every bar. So the walk is handed `long_sig_exit` / `short_sig_exit`
    and resolves them against the hard stop in the same pass.

    The two sides are written out rather than folded into a sign flip:

        LONG  entered BELOW the lower band, exits when Close crosses UP
              through the midline.  Stop sits BELOW the fill.
        SHORT entered ABOVE the upper band, exits when Close crosses DOWN
              through the midline.  Stop sits ABOVE the fill.

    A short stop placed below the fill would be breached by the fill bar
    itself, which is why `_walk_loop` keeps the sides apart rather than
    negating one into the other.

    The entry is a bar-level EVENT inside standing conditions, so a long
    stretch of bars outside the band produces a signal only where the cross
    fires. The walk enters only when FLAT: a second trigger while a position is
    open is ignored rather than pyramided, and an opposite trigger is ignored
    rather than reversing.
    """
    _validate(bb_len, bb_mult, rsi_thresh, rsi_len, sl_atr_mult, tp_atr_mult,
              trailing)

    L = _layers(bars, int(bb_len), float(bb_mult), float(rsi_thresh),
                int(rsi_len), bool(use_regime_filter), bool(use_alpha_trigger))

    long_ok = (L["regime_ok"] & L["touch_long"] & L["osc_long"]).to_numpy()
    short_ok = (L["regime_ok"] & L["touch_short"] & L["osc_short"]).to_numpy()

    close = L["close"]
    if use_midline_exit:
        long_sig_exit = _cross_above(close, L["mid"]).fillna(False).to_numpy()
        short_sig_exit = _cross_below(close, L["mid"]).fillna(False).to_numpy()
    else:
        # With the midline exit off the hard stop is the ONLY exit, which is a
        # different strategy and is why this toggle exists: it prices what the
        # dynamic target is worth against letting the stop decide.
        long_sig_exit = np.zeros(len(bars), dtype=bool)
        short_sig_exit = np.zeros(len(bars), dtype=bool)

    atr = L["atr"].to_numpy(dtype="float64")
    # A bar with no ATR yet is not tradable: the stop has no width. This is the
    # warm-up, and admitting it would place the stop at the fill.
    warm = ~np.isfinite(atr) | (atr <= 0.0)
    long_ok = long_ok & ~warm
    short_ok = short_ok & ~warm

    if use_news_filter:
        long_ok, short_ok = _news_suppress(bars, long_ok, short_ok)

    flat_bar = np.zeros(len(bars), dtype=bool)

    le, lx, se, sx, _, _ = _walk_loop(
        long_ok, short_ok, long_sig_exit, short_sig_exit,
        bars["open"].to_numpy(dtype="float64"),
        bars["high"].to_numpy(dtype="float64"),
        bars["low"].to_numpy(dtype="float64"),
        atr, flat_bar, float(sl_atr_mult),
        _tp_distance(tp_atr_mult), bool(trailing))

    idx = bars.index
    return (pd.Series(le, index=idx), pd.Series(lx, index=idx),
            pd.Series(se, index=idx), pd.Series(sx, index=idx))


def indicators(bars: pd.DataFrame, **params) -> dict[str, pd.Series]:
    """
    Full-length series drawn over the trade inspector's candles, from the same
    `_layers` call `signal_fn` uses - so the chart cannot draw a band touch a
    bar from where the entry actually happened.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["bb_len"]), float(p["bb_mult"]),
                float(p["rsi_thresh"]), int(p["rsi_len"]),
                bool(p["use_regime_filter"]), bool(p["use_alpha_trigger"]))
    return {
        f"BB mid SMA({int(p['bb_len'])})": L["mid"],
        f"BB upper (+{float(p['bb_mult'])}s)": L["upper"],
        f"BB lower (-{float(p['bb_mult'])}s)": L["lower"],
        f"RSI({int(p['rsi_len'])})": L["rsi"],
        f"ADX({ADX_PERIOD})": L["adx"],
        "ADX range max": pd.Series(ADX_RANGE_MAX, index=bars.index),
        f"ATR({ATR_PERIOD})": L["atr"],
    }


def ml_features(bars: pd.DataFrame, **params) -> pd.DataFrame:
    """
    The matrix Version B's classifier is fitted on: one row per bar, in order.

    CAUSALITY IS THIS MODULE'S RESPONSIBILITY. Every column is built from
    `_layers`, which reads only bars <= i, and nothing here is scaled against
    the whole frame - a scaler fitted end to end leaks the test period's
    distribution into the training rows without tripping any shift-based audit.
    `HistGradientBoostingClassifier` needs no scaler, which is why none is
    fitted.

    SHAPE IS CHECKED BY THE CALLER AND A FAILURE RAISES - unlike `indicators`,
    which is wrapped, because a broken chart annotation must not throw away a
    completed backtest but a silently swapped feature matrix must not survive
    one.

    `band_width_norm` is the column this strategy lives or dies on: it is the
    gross target expressed as a fraction of price, and the classifier's most
    useful job here is learning which touches are too narrow to clear their
    own costs.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["bb_len"]), float(p["bb_mult"]),
                float(p["rsi_thresh"]), int(p["rsi_len"]),
                bool(p["use_regime_filter"]), bool(p["use_alpha_trigger"]))
    close = L["close"]
    atr = L["atr"].replace(0.0, np.nan)
    sigma = L["sigma"].replace(0.0, np.nan)
    volume = (bars["volume"].astype("float64") if "volume" in bars.columns
              else pd.Series(np.nan, index=bars.index))
    vol_sma = volume.rolling(20, min_periods=20).mean().shift(1)
    vol_std = volume.rolling(20, min_periods=20).std().shift(1)
    ts_et = L["ts"].tz_convert("America/New_York")
    out = pd.DataFrame({
        "norm_atr": L["norm_atr"],
        # The target, normalised. `bb_mult * sigma / close` is the distance
        # from the band to the midline as a fraction of price - the number the
        # round-trip cost has to be read against.
        "band_width_norm": (float(p["bb_mult"]) * sigma) / close,
        "dist_from_mid_sigma": (close - L["mid"]) / sigma,
        "rsi": L["rsi"],
        "adx": L["adx"],
        "roc_5": close.pct_change(5),
        "roc_15": close.pct_change(15),
        "close_over_atr": close.diff() / atr,
        "volume_z": (volume - vol_sma) / vol_std.replace(0.0, np.nan),
        "hour_et": pd.Series(ts_et.hour, index=bars.index, dtype="float64"),
        "dow_et": pd.Series(ts_et.dayofweek, index=bars.index,
                            dtype="float64"),
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
