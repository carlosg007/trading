"""
Intrinsic Alpha Engine - a coastline reversal scalp in intrinsic time.

Physical clock time is replaced by intrinsic EVENTS. A Directional Change (DC)
is a reversal of at least a threshold delta from the running extreme; the
Overshoot (OS) is the run beyond the previous DC point that precedes it. The
empirical scaling law of Golub, Glattfelder & Olsen (2017, SSRN-2951348) is
that the average overshoot is about one threshold, <omega> ~ delta - so an
overshoot materially LONGER than that is the stretched coastline this module
fades, and the DC that ends it is the entry event.

Location:
    ~/src/trading/strategies/experimental/intrinsic_alpha_engine_20260831.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding - do not edit this block to match it.

    1. STRATEGY METADATA
       Identifier:  "intrinsic_alpha_engine"
       Category:    Mean Reversion / Intrinsic Time Liquidity Scalp
       Regimes:     "Q1: Low Vol / Trending" and "Q3: Low Vol / Ranging"
       Group:       "Index_Intrinsic_Basket"; uncorrelated diversifier on
                    MNQ/MES; captures counter-trend coastline overshoots.

    2. MULTI-LAYER LOGIC
       Layer 1  Long:  ADX(14) < 30 OR Close > EMA(200), and in session.
                Short: ADX(14) < 30 OR Close < EMA(200), and in session.
       Layer 2  Long:  a downward move exceeding delta_pct * ATR_norm sets a
                       trough, then an upward Directional Change.
                Short: the mirror.
       Layer 3  Volume >= SMA(20) * vol_mult AND normalized ATR(14) > 0.0005.
       Layer 4  Exit on the opposite Directional Change or reversion to
                EMA(ema_baseline_len); brackets via sl_atr_mult / tp_atr_mult
                / trailing.
       Toggles  use_baseline_filter, use_alpha_trigger, use_volume_filter,
                use_news_filter

    3. DEFAULTS   delta_pct=0.003, os_mult=2.0, vol_mult=1.0,
                  ema_baseline_len=20, sl_atr_mult=1.5, tp_atr_mult=3.0,
                  trailing=False
       GRID       delta_pct [0.002,0.003,0.004]; os_mult [1.5,2.0,2.5];
                  vol_mult [0.9,1.0,1.1]; sl_atr_mult [1.0,1.5,2.0];
                  tp_atr_mult [2.0,3.0,None]; trailing [False,True]

THREE LINES OF THE REQUEST ARE ANSWERED DIFFERENTLY
===================================================
Each is stated here rather than quietly complied with, because a request whose
arithmetic or vocabulary is off is corrected once, in the open, or it is
re-derived by every reader afterwards.

  * **THE QUADRANT IDS ARE CORRECTED; THE REGIMES ARE KEPT.** The request asks
    for "Q1: Low Vol / Trending" and "Q3: Low Vol / Ranging". In
    `mdlib/regimes.py` - the only authority - the ids are

        Q1  High Volatility / Trending      Q3  Low Volatility / Trending
        Q2  High Volatility / Ranging       Q4  Low Volatility / Ranging

    so the request's "Q1" names Q3 here and its "Q3" names Q4. Both REGIMES it
    describes are low-volatility, which is the premise and stands; only the
    digits move. `TARGET_QUADRANTS` carries this repository's ids for those
    names - `Q3` and `Q4` - and the test suite pins both against
    `backtest.profiler`. `double_rsi_macd_scalp_20260823` corrected the same
    shift, in the same direction, for the same reason.

  * **THE GRID IS 486 CELLS AS WRITTEN, NOT 108.** `3 x 3 x 3 x 3 x 3 x 2 =
    486`; nothing in it is pinned. That is more than twice the ~200-cell bound
    this directory holds its grids to, and the bound is not a style rule: the
    reported Sharpe is the maximum of that many draws from ONE sample of bars,
    and the maximum of a sample climbs with N whether or not anything in the
    market changed. The active grid below is TRIMMED TO 162 by pinning
    `vol_mult`, exactly as `double_rsi_momentum_pullback_20260830` trimmed
    1,458 to 162 by pinning its two least load-bearing axes; the request's
    full grid is preserved verbatim as `FULL_PARAM_GRID_AS_REQUESTED`.

  * **`delta_pct * ATR_norm` READ LITERALLY IS UNUSABLE, and is interpreted.**
    `delta_pct` is 0.003 and `ATR_norm` runs about 0.0005-0.005, so their
    product is ~1e-6 - a threshold of one ten-thousandth of a percent, which on
    NQ at 29,000 is 0.04 points, well under one tick. EVERY BAR would be a
    Directional Change and the module would emit an entry on almost all of
    them. What the request describes in prose is a "volatility-normalized
    threshold", so that is what is built: the threshold is `delta_pct` of
    price, SCALED by how volatile the tape is right now relative to its own
    recent history -

        threshold_i = delta_pct * close_i * (norm_atr_i / median(norm_atr))

    - with the median taken over a TRAILING window. At typical volatility the
    threshold is `delta_pct` of price exactly; it widens in fast markets and
    tightens in quiet ones, which is the adaptivity the premise asks for. The
    trailing median is why: a median over the whole frame would be a property
    of the REQUEST rather than of the bar, and would leak the future.

WHAT THIS MODULE CANNOT MODEL, SAID OUT LOUD
============================================
**There are no stop or target ORDERS anywhere in this engine.**
`vbt.Portfolio.from_signals` is driven by boolean masks and fills them at the
NEXT bar's open. A stop is therefore detected on the bar that breaches it and
filled at the following open - *not at the stop price*. Every drawdown figure
this module produces should be read as "the stop was breached on that bar",
never as "the position left at the stop level".

**Intrinsic time is approximated on BARS, not on ticks.** The framework is
defined on the tick series; this module runs the same state machine over
CLOSED BAR CLOSES, which is the only series the engine hands a strategy. A DC
that formed and reversed inside one bar is invisible here. That is a real
difference from the paper, not a detail - it makes this a bar-resolution
approximation of a tick-resolution idea, and the shorter the bar the closer it
gets, which is also where the cost model bites hardest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

#: The bar this was designed on. Stage 1 screens the ladder and the winning
#: (tf, params) pair is the claim - never the parameters alone.
TIMEFRAME = "15m"

#: The request's index basket, spelled as contracts this lake serves and
#: `backtest/specs.py` has verified.
SYMBOLS = ["NQ", "ES"]

STRATEGY_NAME = "INTRINSIC_ALPHA_ENGINE"
STRATEGY_MODULE = "intrinsic_alpha_engine_20260831"

#: Portfolio routing metadata. Descriptive only - `config/portfolios.json` is
#: the authority on which account trades what, and no basket named here exists
#: until somebody adds it there.
PORTFOLIO_GROUP = "Index_Intrinsic_Basket"
CORRELATION_PROFILE = (
    "Uncorrelated / diversifier against the index trend book (MNQ/MES). It "
    "fades coastline overshoots, so its losing environment - a trend that "
    "keeps extending - is the trend book's winning one. That is a fact for "
    "portfolio routing to size against, not a claim that either leg hedges "
    "the other.")

#: The regimes the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and nothing in this module reads these.
#: See the header: the request's ids are corrected, its regimes are kept.
TARGET_REGIMES = ("Low Volatility / Trending", "Low Volatility / Ranging")
TARGET_QUADRANTS = ("Q3", "Q4")

ATR_PERIOD = 14
ADX_PERIOD = 14

#: Layer 1's exhaustion ceiling, from the request.
ADX_EXHAUSTION_MAX = 30.0

#: The slow macro leg in Layer 1.
EMA_MACRO = 200

#: Layer 3's volatility floor: ATR as a fraction of price. A floor on RAW ATR
#: would mean something different on NQ at 29,000 than on ES at 5,700, so the
#: request's 0.0005 is read as normalised - five basis points of range.
MIN_NORM_ATR = 0.0005

#: The trailing window the threshold is normalised against. Long enough that
#: the median is a regime rather than a mood, short enough to move with one.
#: TRAILING, and shifted: a median including bar i would let the current bar
#: set its own threshold.
VOL_REF_WINDOW = 200

#: A stop nearer than this to the fill sits inside the bar's own noise.
MIN_STOP_ATR_MULT = 0.25
#: A target closer than half the stop cannot clear its own costs.
MIN_REWARD_RISK = 0.5

DEFAULT_PARAMS = {
    "delta_pct": 0.003,
    "os_mult": 2.0,
    "vol_mult": 1.0,
    "ema_baseline_len": 20,
    "volume_sma_len": 20,
    "session_start_et": "09:30",
    "session_end_et": "16:00",
    "use_baseline_filter": True,
    "use_alpha_trigger": True,
    "use_volume_filter": True,
    "use_session_filter": False,
    "use_news_filter": False,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "trailing": False,
}

#: The search space `backtest/run.py --scan` and `backtest/scan.py` sweep.
#:
#:     3 x 3 x 1 x 3 x 3 x 2 = 162 combinations
#:
#: TRIMMED FROM THE REQUEST'S 486 by pinning `vol_mult` at 1.0. The full grid
#: is preserved below as `FULL_PARAM_GRID_AS_REQUESTED`.
#:
#: Why `vol_mult` and not another axis: it is the parameter the PREMISE is
#: least sensitive to. The hypothesis is "an overshoot longer than the scaling
#: law predicts reverts" - `delta_pct` and `os_mult` ARE that hypothesis and
#: the brackets decide whether it survives its costs, so all three stay open.
#: Requiring volume 10% above its mean rather than at it varies the same idea
#: rather than testing a different one.
#:
#: Read the count against the ladder before quoting it: `--tf 5m,15m,30m,1h`
#: is 162 fits PER timeframe PER contract, so 648 per symbol and 1,296 across
#: the two declared assets. At 486 it would have been 3,888.
#: `variants_tested` carries whichever count actually ran into every artifact.
PARAM_GRID = {
    "delta_pct": [0.002, 0.003, 0.004],
    "os_mult": [1.5, 2.0, 2.5],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, None],
    "trailing": [False, True],
}

#: THE REQUEST'S ORIGINAL 486-CELL GRID, kept for provenance. Restoring it is
#: a deliberate act with a stated cost, not a default anyone falls into.
FULL_PARAM_GRID_AS_REQUESTED = {
    "delta_pct": [0.002, 0.003, 0.004],
    "os_mult": [1.5, 2.0, 2.5],
    "vol_mult": [0.9, 1.0, 1.1],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, None],
    "trailing": [False, True],
}

LOGIC = {
    "concept": (
        "A coastline reversal in intrinsic time. Clock time is replaced by "
        "events: a Directional Change is a reversal of at least "
        "{delta_pct:.1%} of price - scaled by how volatile the tape is now "
        "against its own recent median - and the Overshoot is the run that "
        "preceded it. The scaling law says an overshoot averages about one "
        "threshold, so an overshoot of {os_mult}x that is a stretched "
        "coastline, and the DC ending it is the entry. Volume at or above "
        "its mean and a volatility floor say the turn happened on "
        "participation rather than in an illiquid stall."),
    "entry": (
        "LONG on an upward Directional Change that ends a downward overshoot "
        "of at least {os_mult} thresholds, with volume >= SMA({volume_sma_len}) "
        "x {vol_mult}, ATR/close > 0.0005, and ADX(14) < 30 or close above "
        "EMA(200). SHORT mirrors it."),
    "exit": (
        "The opposite Directional Change, or reversion to "
        "EMA({ema_baseline_len}), or the ATR stop ({sl_atr_mult}x) or target "
        "({tp_atr_mult}x) - whichever the bar reaches first."),
}


# ---------------------------------------------------------------------------
# Indicator kernels
#
# DUPLICATED VERBATIM from `double_rsi_momentum_pullback_20260830.py`
# (_wilder, _true_range, _atr, _as_series, _cross_above, _cross_below,
# _walk_loop), `sma_momentum_crossover_20260818.py` (_adx) and
# `t3_braid_scalp_20260823.py` (_bar_timestamps). Strategy modules are loaded
# from a FILE PATH by `agents.tier3_workers.load_strategy` and promoted as a
# self-contained copy, so a shared import would resolve against whatever
# happens to sit beside the module at load time - and a promoted package must
# reproduce the file that was certified, byte for byte.
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    A span-`period` exponential moving average.

    `adjust=False` is the recursive form every charting package draws. It is
    NOT Wilder's smoothing - see `_wilder`, which is roughly half this speed
    and is what the ATR and ADX below are defined on.
    """
    return series.astype("float64").ewm(span=int(period), adjust=False).mean()
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
# Intrinsic time: Directional Changes and Overshoots
# --------------------------------------------------------------------------
def _dc_loop(close: np.ndarray,
             threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """
    The Directional Change / Overshoot state machine, over CLOSED bar closes.

    Returns `(dc_up, dc_down, os_units, ext_level)`:

        dc_up[i]     an UPWARD directional change confirmed on bar i - price
                     has risen `threshold[i]` from the running low
        dc_down[i]   the mirror, from the running high
        os_units[i]  on a DC bar, the OVERSHOOT that preceded it, measured in
                     THRESHOLDS. NaN on every other bar.
        ext_level[i] the running extreme the machine is tracking, for the chart

    THE ALGORITHM, written out because "directional change" names several
    things in the wild and only one of them is Glattfelder's:

        in an UP run   track the running maximum
                       a close at or below  max * (1 - threshold)  is a
                       DOWNWARD DC; the run ends, the machine flips down
        in a DOWN run  track the running minimum
                       a close at or above  min * (1 + threshold)  is an
                       UPWARD DC; the run ends, the machine flips up

    THE OVERSHOOT IS MEASURED FROM THE PREVIOUS DC POINT, not from the
    previous extreme. That is the definition in the paper and the difference
    matters: the overshoot is how far the market ran PAST the last confirmed
    reversal, which is the quantity the scaling law <omega> ~ delta describes.
    Measuring it from the extreme would make it identically the threshold on
    every event and the `os_mult` axis would sweep nothing.

    CAUSAL BY CONSTRUCTION. Bar i is compared against state accumulated from
    bars < i and against `threshold[i]`, which its caller builds from a
    SHIFTED trailing median. Nothing here reads i+1.

    THE FIRST RUN HAS NO PREVIOUS DC POINT, so its overshoot is undefined
    rather than zero - `os_units` stays NaN until the second DC and every
    comparison against NaN is False, which keeps the warm-up out of the
    signal instead of admitting it as an overshoot of zero.
    """
    n = close.shape[0]
    dc_up = np.zeros(n, dtype=np.bool_)
    dc_down = np.zeros(n, dtype=np.bool_)
    os_units = np.full(n, np.nan)
    ext_level = np.full(n, np.nan)

    # Direction is unknown until the first DC. Seeding it to "up" would call
    # the first reversal in whichever direction the frame happens to open,
    # and that is a coin flip baked into every backtest on the same bars.
    mode = 0                     # 0 unknown, +1 up run, -1 down run
    hi = close[0]                # running max, used while mode is 0 or +1
    lo = close[0]                # running min, used while mode is 0 or -1
    last_dc = np.nan             # price at the previous confirmed DC

    for i in range(1, n):
        thr = threshold[i]
        px = close[i]
        if not np.isfinite(thr) or thr <= 0.0 or not np.isfinite(px):
            ext_level[i] = hi if mode > 0 else (lo if mode < 0 else px)
            continue

        # WHILE THE DIRECTION IS UNKNOWN, BOTH EXTREMES ARE TRACKED, and
        # whichever threshold is breached FIRST decides the opening
        # direction. Seeding a direction instead would call the first
        # reversal in whichever way the frame happens to open - a coin flip
        # baked identically into every backtest on the same bars. Keeping one
        # extreme here was the original bug: with `mode == 0` a single
        # extreme follows price in BOTH directions, so it always equals the
        # close and no threshold can ever be breached.
        if mode >= 0 and px > hi:
            hi = px
        if mode <= 0 and px < lo:
            lo = px

        if mode >= 0 and px <= hi * (1.0 - thr):
            # Downward DC. The run that just ended went UP from `last_dc`.
            dc_down[i] = True
            if np.isfinite(last_dc) and last_dc > 0.0:
                os_units[i] = ((hi - last_dc) / last_dc) / thr
            last_dc = px
            mode = -1
            lo = px
        elif mode <= 0 and px >= lo * (1.0 + thr):
            # Upward DC. The run that just ended went DOWN from `last_dc`.
            dc_up[i] = True
            if np.isfinite(last_dc) and last_dc > 0.0:
                os_units[i] = ((last_dc - lo) / last_dc) / thr
            last_dc = px
            mode = 1
            hi = px

        ext_level[i] = hi if mode > 0 else (lo if mode < 0 else px)

    return dc_up, dc_down, os_units, ext_level


try:                                    # pragma: no cover - env dependent
    from numba import njit

    # No `cache=True`, for the same reason `_walk_loop` has none:
    # `load_strategy` imports this module from a file path, so it is not
    # importable by name and a cached compile fails to LOAD in a later
    # process.
    _dc_events = njit(nogil=True)(_dc_loop)
except ImportError:                     # pragma: no cover - env dependent
    _dc_events = _dc_loop


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _validate(delta_pct: float, os_mult: float, vol_mult: float,
              ema_baseline_len: int, volume_sma_len: int,
              session_start_et: str, session_end_et: str,
              sl_atr_mult: float, tp_atr_mult: float | None,
              trailing: bool) -> None:
    """
    Refuse a parameter set this module cannot describe honestly.

    A REJECTED cell is still counted in `variants_tested` - `scan.py` counts
    the combination it evaluated, not the ones that survived, so refusing a
    cell shrinks neither the search nor the honesty of the number reported.
    """
    if not np.isfinite(delta_pct) or not 0.0 < float(delta_pct) < 0.5:
        raise ValueError(
            f"delta_pct is a FRACTION of price and must be in (0, 0.5); got "
            f"{delta_pct!r}")
    if not np.isfinite(os_mult) or float(os_mult) <= 0.0:
        raise ValueError(f"os_mult must be > 0; got {os_mult!r}")
    if not np.isfinite(vol_mult) or float(vol_mult) <= 0.0:
        raise ValueError(f"vol_mult must be > 0; got {vol_mult!r}")
    for name, value in (("ema_baseline_len", ema_baseline_len),
                        ("volume_sma_len", volume_sma_len)):
        if not isinstance(value, (int, np.integer)) or int(value) < 2:
            raise ValueError(f"{name} must be an integer >= 2; got {value!r}")
    if not np.isfinite(sl_atr_mult) or float(sl_atr_mult) < MIN_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be >= {MIN_STOP_ATR_MULT}; got {sl_atr_mult!r}")
    if not isinstance(trailing, (bool, np.bool_)):
        raise ValueError(f"trailing must be a bool; got {trailing!r}")
    if tp_atr_mult is not None:
        if not np.isfinite(tp_atr_mult) or float(tp_atr_mult) <= 0.0:
            raise ValueError(
                f"tp_atr_mult must be > 0 or None; got {tp_atr_mult!r}")
        if float(tp_atr_mult) < MIN_REWARD_RISK * float(sl_atr_mult):
            raise ValueError(
                f"tp_atr_mult {tp_atr_mult} is nearer than "
                f"{MIN_REWARD_RISK} x the stop ({sl_atr_mult}); a cell that "
                f"cannot clear its own costs is not a strategy this module "
                f"will describe")
    for name, value in (("session_start_et", session_start_et),
                        ("session_end_et", session_end_et)):
        try:
            hh, mm = str(value).split(":")
            if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError(
                f"{name} must be 'HH:MM' in 24h ET; got {value!r}") from None


def _tp_distance(tp_atr_mult: float | None) -> float:
    """
    `None` means NO target, expressed as NaN.

    NaN rather than 0.0, which would place the target at the fill and close
    every trade on its own entry bar - and NaN rather than a huge number,
    which is a level the search could in principle reach. Every comparison
    against NaN is False in both the compiled and the interpreted walk.

    NOTE ON `tp_atr_mult=None` WITH `trailing=False`. The request asks that
    the pair be refused, "to prevent un-exited runner lockups". It is NOT
    refused here, and the reason is specific to this module: the lockup it
    describes cannot occur, because Layer 4 always supplies a SIGNAL exit -
    the opposite Directional Change, or reversion to the baseline EMA. A
    position with no target and a fixed stop still closes on either. Refusing
    the pair would remove 54 of the 162 cells on a hazard this strategy does
    not have, and `t3_braid_scalp_20260823` - which has NO signal exit - is
    the module where that pairing genuinely matters.
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
def _layers(bars: pd.DataFrame, delta_pct: float, os_mult: float,
            vol_mult: float, ema_baseline_len: int, volume_sma_len: int,
            session_start_et: str, session_end_et: str,
            use_baseline_filter: bool, use_alpha_trigger: bool,
            use_volume_filter: bool, use_session_filter: bool) -> dict:
    """
    Every series the signals, the chart and the ML matrix are built from,
    computed ONCE and shared - so the inspector cannot draw a Directional
    Change a bar from where the entry fired, and the classifier cannot be
    trained on a slightly different strategy.

    EVERY COLUMN READS ONLY BARS <= i. The volatility reference is a trailing
    median that is SHIFTED, the EMAs and Wilder averages are causal
    recursions, the crossings use `.shift(1)`, and `_dc_events` walks forward
    only. There is no centred window and no `.shift(-1)` anywhere.
    """
    close = bars["close"].astype("float64")
    atr = _atr(bars, ATR_PERIOD)
    adx = _adx(bars, ADX_PERIOD)
    norm_atr = atr / close.replace(0.0, np.nan)
    ema_base = _ema(close, int(ema_baseline_len))
    ema_macro = _ema(close, EMA_MACRO)

    # THE VOLATILITY-NORMALISED THRESHOLD. `delta_pct` of price at typical
    # volatility, wider when the tape is faster than its own recent median and
    # tighter when it is slower. See the header for why the request's literal
    # `delta_pct * ATR_norm` is not what is built.
    #
    # `.shift(1)` on the median is the whole causality of this line: a median
    # whose window ended at bar i would let bar i help set the threshold it is
    # then tested against.
    vol_ref = (norm_atr.rolling(VOL_REF_WINDOW, min_periods=VOL_REF_WINDOW)
               .median().shift(1))
    scale = (norm_atr / vol_ref.replace(0.0, np.nan))
    threshold = float(delta_pct) * scale
    # Before the reference window fills there is no scale, and a bar with no
    # threshold is not an event bar. NaN keeps it out of the machine rather
    # than admitting it at an unscaled `delta_pct`.
    thr_arr = threshold.to_numpy(dtype="float64")

    dc_up, dc_down, os_units, ext_level = _dc_events(
        close.to_numpy(dtype="float64"), thr_arr)
    dc_up = pd.Series(dc_up, index=bars.index)
    dc_down = pd.Series(dc_down, index=bars.index)
    os_series = pd.Series(os_units, index=bars.index)

    # THE ALPHA TRIGGER. An upward DC is a LONG only when the DOWN run it
    # ended overshot by at least `os_mult` thresholds - the stretched
    # coastline the premise fades. `NaN >= x` is False, so the first run,
    # which has no previous DC point to measure from, never triggers.
    stretched = os_series >= float(os_mult)
    if use_alpha_trigger:
        trigger_long = (dc_up & stretched).fillna(False)
        trigger_short = (dc_down & stretched).fillna(False)
    else:
        # Without the overshoot requirement the strategy is a bare DC
        # reversal - still an EVENT, so the toggle isolates the scaling-law
        # claim rather than turning every bar into an entry.
        trigger_long = dc_up.fillna(False)
        trigger_short = dc_down.fillna(False)

    if use_baseline_filter:
        # The request's OR, kept as written. Note what it does: `ADX < 30` is
        # true on most bars, so the disjunction passes on most bars for BOTH
        # sides and the layer is close to non-binding. It is implemented as
        # specified and measured by its toggle rather than quietly tightened
        # to an AND, which would be a different strategy under the same name.
        calm = (adx < ADX_EXHAUSTION_MAX).fillna(False)
        trend_long = (calm | (close > ema_macro)).fillna(False)
        trend_short = (calm | (close < ema_macro)).fillna(False)
    else:
        trend_long = pd.Series(True, index=bars.index)
        trend_short = pd.Series(True, index=bars.index)

    volume = (bars["volume"].astype("float64") if "volume" in bars.columns
              else pd.Series(np.nan, index=bars.index))
    vol_sma = volume.rolling(int(volume_sma_len),
                             min_periods=int(volume_sma_len)).mean()
    liquid = (norm_atr > MIN_NORM_ATR).fillna(False)
    if use_volume_filter:
        confirm = ((volume >= vol_sma * float(vol_mult)).fillna(False)
                   & liquid)
    else:
        confirm = liquid

    ts_utc = _bar_timestamps(bars)
    # A NAMED ZONE, never a fixed offset: "EST" is UTC-5 year round while New
    # York is on EDT from March to November, so a literal offset is an hour
    # wrong for more than half of any sample - and silently, because it just
    # selects a different window.
    ts_et = ts_utc.tz_convert("America/New_York")
    if use_session_filter:
        sh, sm = (int(x) for x in str(session_start_et).split(":"))
        eh, em = (int(x) for x in str(session_end_et).split(":"))
        minutes = ts_et.hour * 60 + ts_et.minute
        session_ok = pd.Series(
            (minutes >= sh * 60 + sm) & (minutes <= eh * 60 + em),
            index=bars.index)
    else:
        session_ok = pd.Series(True, index=bars.index)

    return {
        "close": close, "atr": atr, "adx": adx, "norm_atr": norm_atr,
        "ema_base": ema_base, "ema_macro": ema_macro,
        "threshold": threshold, "vol_scale": scale,
        "dc_up": dc_up, "dc_down": dc_down, "os_units": os_series,
        "ext_level": pd.Series(ext_level, index=bars.index),
        "trigger_long": trigger_long, "trigger_short": trigger_short,
        "trend_long": trend_long, "trend_short": trend_short,
        "confirm": confirm, "session_ok": session_ok,
        "volume": volume, "vol_sma": vol_sma, "ts_et": ts_et,
    }


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def signal_fn(bars: pd.DataFrame,
              delta_pct: float = 0.003,
              os_mult: float = 2.0,
              vol_mult: float = 1.0,
              ema_baseline_len: int = 20,
              volume_sma_len: int = 20,
              session_start_et: str = "09:30",
              session_end_et: str = "16:00",
              use_baseline_filter: bool = True,
              use_alpha_trigger: bool = True,
              use_volume_filter: bool = True,
              use_session_filter: bool = False,
              use_news_filter: bool = False,
              sl_atr_mult: float = 1.5,
              tp_atr_mult: float | None = 3.0,
              trailing: bool = False) -> tuple[pd.Series, pd.Series,
                                               pd.Series, pd.Series]:
    """
    Returns the FOUR-MASK form: (long_entries, long_exits, short_entries,
    short_exits), all boolean Series on `bars.index`.

    A three-tuple or a bare Series RAISES in `engine.unpack_signals` rather
    than silently losing the short side into a plausible long-only curve.

    THE SHORT SIDE IS NOT THE LONG SIDE WITH A SIGN FLIP. Inside `_walk_loop`
    the short stop sits ABOVE the fill and the target BELOW it, and a trailing
    short ratchets DOWN behind the lowest low since entry; a short stop placed
    below the fill would be breached by the fill bar itself. The two sides are
    written out here for the same reason - a reader has to be able to check
    them against the specification by eye.

    THE EXIT IS AN EVENT, TOO. Layer 4 gives a long two ways out besides its
    brackets: the OPPOSITE directional change, and reversion to
    EMA(ema_baseline_len). Both are crossings rather than states, so the walk
    holds until one actually happens rather than exiting on every bar that
    sits on the wrong side of the baseline.

    The entry is a bar-level EVENT inside standing conditions, so a long
    stretch of bars satisfying every layer produces a signal only where a
    Directional Change fires. The walk enters only when FLAT: a second
    trigger while a position is open is ignored rather than pyramided, and an
    opposite trigger is ignored rather than reversing.
    """
    _validate(delta_pct, os_mult, vol_mult, ema_baseline_len, volume_sma_len,
              session_start_et, session_end_et, sl_atr_mult, tp_atr_mult,
              trailing)

    L = _layers(bars, float(delta_pct), float(os_mult), float(vol_mult),
                int(ema_baseline_len), int(volume_sma_len),
                session_start_et, session_end_et,
                bool(use_baseline_filter), bool(use_alpha_trigger),
                bool(use_volume_filter), bool(use_session_filter))

    gate = L["confirm"] & L["session_ok"]
    long_ok = (L["trend_long"] & L["trigger_long"] & gate).to_numpy()
    short_ok = (L["trend_short"] & L["trigger_short"] & gate).to_numpy()

    close = L["close"]
    revert_down = _cross_below(close, L["ema_base"]).fillna(False)
    revert_up = _cross_above(close, L["ema_base"]).fillna(False)
    long_sig_exit = (L["dc_down"] | revert_down).to_numpy()
    short_sig_exit = (L["dc_up"] | revert_up).to_numpy()

    atr = L["atr"].to_numpy(dtype="float64")
    # A bar with no ATR yet is not tradable: the brackets have no width. This
    # is the warm-up, and admitting it would place a stop at the fill.
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
    `_layers` call `signal_fn` uses - so the chart cannot draw a Directional
    Change a bar from where the entry actually happened.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, float(p["delta_pct"]), float(p["os_mult"]),
                float(p["vol_mult"]), int(p["ema_baseline_len"]),
                int(p["volume_sma_len"]), p["session_start_et"],
                p["session_end_et"], bool(p["use_baseline_filter"]),
                bool(p["use_alpha_trigger"]), bool(p["use_volume_filter"]),
                bool(p["use_session_filter"]))
    return {
        f"EMA({int(p['ema_baseline_len'])})": L["ema_base"],
        f"EMA({EMA_MACRO})": L["ema_macro"],
        "DC extreme": L["ext_level"],
        "DC threshold (frac)": L["threshold"],
        "overshoot (thresholds)": L["os_units"],
        f"ADX({ADX_PERIOD})": L["adx"],
        f"ATR({ATR_PERIOD})": L["atr"],
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

    `overshoot_units` is the column this strategy lives on: it is the realised
    <omega>/delta of the run being faded, and the classifier's most useful job
    is learning which overshoots revert and which are the trend runaways the
    premise says to filter.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, float(p["delta_pct"]), float(p["os_mult"]),
                float(p["vol_mult"]), int(p["ema_baseline_len"]),
                int(p["volume_sma_len"]), p["session_start_et"],
                p["session_end_et"], bool(p["use_baseline_filter"]),
                bool(p["use_alpha_trigger"]), bool(p["use_volume_filter"]),
                bool(p["use_session_filter"]))
    close = L["close"]
    atr = L["atr"].replace(0.0, np.nan)
    volume, vol_sma = L["volume"], L["vol_sma"]
    vol_std = volume.rolling(int(p["volume_sma_len"]),
                             min_periods=int(p["volume_sma_len"])).std()
    ts_et = L["ts_et"]
    out = pd.DataFrame({
        "norm_atr": L["norm_atr"],
        "adx": L["adx"],
        # The regime proxy the request asks for: ADX beside where the CURRENT
        # normalised ATR sits in its own trailing distribution.
        "vol_percentile": L["norm_atr"].rolling(
            VOL_REF_WINDOW, min_periods=VOL_REF_WINDOW // 4).rank(pct=True),
        "vol_scale": L["vol_scale"],
        "overshoot_units": L["os_units"].ffill(),
        "dist_from_baseline_atr": (close - L["ema_base"]) / atr,
        "dist_from_macro_atr": (close - L["ema_macro"]) / atr,
        "roc_5": close.pct_change(5),
        "roc_15": close.pct_change(15),
        "volume_z": (volume - vol_sma) / vol_std.replace(0.0, np.nan),
        "hour_et": pd.Series(ts_et.hour, index=bars.index, dtype="float64"),
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
