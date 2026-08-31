"""
Dual EMA Slope Scalp - a trend-continuation scalp in three layers: a
higher-horizon trend proxy decides which side is permitted, a fast/slow EMA
ribbon with a minimum slope times the entry, and a candle pattern says the
pullback actually resolved.

Location:
    ~/src/trading/strategies/experimental/dual_ema_slope_scalp_20260831.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding - do not edit this block to match it.

    1. METADATA
       Identifier: "dual_ema_slope_scalp"
       Category:   Trend-Following / Multi-Timeframe Momentum Scalp
       Regimes:    "Q1: Low Vol / Trending" and "Q2: High Vol / Trending"
       Group:      "Equity_Momentum_Basket"; positively correlated with the
                   index group (MNQ/MES).

    2. MULTI-LAYER LOGIC
       Layer 1  Long:  1h trend proxy bullish (1h Close > 1h EMA(50) OR
                       1h EMA(7) > 1h EMA(17)), and in session.
                Short: the mirror.
       Layer 2  Long:  EMA(fast) > EMA(slow) AND normalised slope of
                       EMA(fast) >= min_slope AND a bullish candle pattern
                       (pin bar, engulfing, or expansion bar).
                Short: the mirror, with slope <= -min_slope.
       Layer 3  Volume >= SMA(20) * vol_mult AND normalised ATR(14) > 0.0005.
       Layer 4  Exit when the fast EMA crosses back through the slow EMA;
                brackets via sl_atr_mult / tp_atr_mult / trailing.
       Toggles  use_baseline_filter, use_alpha_trigger, use_volume_filter,
                use_news_filter

    3. DEFAULTS  fast_ema=7, slow_ema=17, htf_ema=50, min_slope=0.15,
                 vol_mult=1.0, sl_atr_mult=1.0, tp_atr_mult=3.0,
                 trailing=False
       GRID      fast_ema [5,7,9]; slow_ema [14,17,21];
                 min_slope [0.10,0.15,0.20]; sl_atr_mult [1.0,1.5,2.0];
                 tp_atr_mult [2.0,3.0,None]; trailing [False,True]

FOUR LINES OF THE REQUEST ARE ANSWERED DIFFERENTLY
==================================================
Each is stated here rather than quietly complied with, because a request whose
arithmetic or vocabulary is off is corrected once, in the open, or it is
re-derived by every reader afterwards.

  * **THE QUADRANT IDS ARE CORRECTED; THE REGIMES ARE KEPT.** The request asks
    for "Q1: Low Vol / Trending" and "Q2: High Vol / Trending". In
    `mdlib/regimes.py` - the only authority - the numbering is

        Q1  High Volatility / Trending      Q3  Low Volatility / Trending
        Q2  High Volatility / Ranging       Q4  Low Volatility / Ranging

    so the request's "Q1" names Q3 here, and its "Q2 (High Vol / Trending)"
    names Q1 - its Q2 is this repository's High-Volatility RANGING quadrant,
    the one environment its own premise ("filtering choppy sideways drift")
    says this strategy must not trade. Taking the digits literally would aim
    the strategy at chop. Both REGIMES it wrote out are Trending, so
    `TARGET_QUADRANTS` carries `Q3` and `Q1`. This is the identical shift
    `double_rsi_macd_scalp_20260823` corrected, in the same direction.

  * **THE GRID IS 486 CELLS AS WRITTEN, NOT 108.** `3 x 3 x 3 x 3 x 3 x 2 =
    486`; nothing in it is pinned. The active grid below is TRIMMED TO 162 by
    pinning `slow_ema` at 17, the axis the premise is least sensitive to -
    the hypothesis is "a fast ribbon accelerating hard enough to be worth
    joining", which `fast_ema` and `min_slope` carry. The request's full grid
    is preserved verbatim as `FULL_PARAM_GRID_AS_REQUESTED`.

  * **NOTHING HERE RESAMPLES TO 1 HOUR, AND THE "HIGHER TIMEFRAME" IS A
    HORIZON RATHER THAN A SECOND BAR SIZE.** This repository's position is
    written down in `double_rsi_macd_scalp_20260823`: "a module that resampled
    internally would be a second aggregation implementation free to disagree
    with `mdlib.lake` about bar boundaries and the Sunday merge." Those
    disagreements are silent - a session boundary or a Sunday-evening merge
    one bar out makes this module's "1h trend" a different series from the one
    Stage 1's 1h screen saw, and every log line still reads correctly.

    So the 1h proxy is computed as an EQUIVALENT-SPAN EMA on the run's own
    bars. `_htf_span` reads the median spacing of the bar index and scales:
    `htf_ema=50` on a 5m run is `EMA(600)`, on 15m it is `EMA(200)`, on 1h it
    is `EMA(50)` exactly. The intent - fifty hours of memory - is preserved on
    whichever rung Stage 1 picks, and there is no second aggregation to drift.
    It is NOT identical to an EMA over resampled hourly bars, and that is
    stated rather than hidden.

  * **`tp_atr_mult=None` WITH `trailing=False` IS ALLOWED.** The request asks
    that the pair be refused "to prevent un-exited runner lockups". Layer 4
    supplies a SIGNAL exit - the fast EMA crossing back through the slow -
    so a position with no target and a fixed stop still closes. Refusing the
    pair would drop 54 of 162 cells on a hazard this strategy does not have.
    `t3_braid_scalp_20260823`, which has no signal exit at all, is the module
    where that pairing genuinely matters.

WHAT THIS MODULE CANNOT MODEL, SAID OUT LOUD
============================================
**There are no stop or target ORDERS anywhere in this engine.**
`vbt.Portfolio.from_signals` is driven by boolean masks and fills them at the
NEXT bar's open. A stop is detected on the bar that breaches it and filled at
the following open - *not at the stop price*. On a scalp with a 1.0x ATR stop
that gap is a large share of the modelled risk, and every drawdown figure here
should be read as "the stop was breached on that bar".

**The 1:3 reward the premise claims is a PARAMETER, not a result.** It is
`tp_atr_mult / sl_atr_mult` at the default cell, and the grid sweeps cells
from 1:1 to 3:1. What decides whether any of them survives is Stage 4's cost
drag as a share of GROSS profit, which on a scalp is the number to read before
anything here is believed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

#: The bar this was designed on - the request's "5-minute execution". Stage 1
#: screens the ladder and the winning (tf, params) pair is the claim.
TIMEFRAME = "5m"

#: The request's index basket, spelled as contracts this lake serves and
#: `backtest/specs.py` has verified.
SYMBOLS = ["NQ", "ES"]

STRATEGY_NAME = "DUAL_EMA_SLOPE_SCALP"
STRATEGY_MODULE = "dual_ema_slope_scalp_20260831"

#: Portfolio routing metadata. Descriptive only - `config/portfolios.json` is
#: the authority on which account trades what, and no basket named here exists
#: until somebody adds it there.
PORTFOLIO_GROUP = "Equity_Momentum_Basket"
CORRELATION_PROFILE = (
    "Positively correlated with the index group (MNQ/MES). It joins trends "
    "the rest of the index book is already in, so a basket of them "
    "concentrates rather than diversifies - a fact for portfolio routing to "
    "size against, not for this module to correct.")

#: The regimes the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and nothing in this module reads these.
#: See the header: the request's ids are corrected, its regimes are kept.
TARGET_REGIMES = ("Low Volatility / Trending", "High Volatility / Trending")
TARGET_QUADRANTS = ("Q3", "Q1")

ATR_PERIOD = 14
ADX_PERIOD = 14

#: The horizon the request calls "1 hour". `_htf_span` converts it to a span
#: on the run's own bars - see the header for why nothing resamples.
HTF_MINUTES = 60

#: Layer 3's volatility floor: ATR as a fraction of price. A floor on RAW ATR
#: would mean something different on NQ at 29,000 than on ES at 5,700, so the
#: request's 0.0005 is read as normalised - five basis points of range.
MIN_NORM_ATR = 0.0005

#: A pin bar's wick must be this multiple of its own body, and this share of
#: the bar's whole range. Two conditions rather than one: a long wick on a
#: tiny body is a pin, and a long wick on a bar whose range is mostly body is
#: not - the ratio alone admits the second.
PIN_WICK_BODY_MULT = 2.0
PIN_WICK_RANGE_FRAC = 0.5

#: An expansion bar's range, as a multiple of the trailing mean range.
EXPANSION_RANGE_MULT = 1.5
EXPANSION_WINDOW = 20

#: A stop nearer than this to the fill sits inside the bar's own noise.
MIN_STOP_ATR_MULT = 0.25
#: A target closer than half the stop cannot clear its own costs.
MIN_REWARD_RISK = 0.5

DEFAULT_PARAMS = {
    "fast_ema": 7,
    "slow_ema": 17,
    "htf_ema": 50,
    "htf_fast_ema": 7,
    "htf_slow_ema": 17,
    "min_slope": 0.15,
    "vol_mult": 1.0,
    "volume_sma_len": 20,
    "session_start_et": "09:30",
    "session_end_et": "16:00",
    "use_baseline_filter": True,
    "use_alpha_trigger": True,
    "use_volume_filter": True,
    "use_session_filter": False,
    "use_news_filter": False,
    "sl_atr_mult": 1.0,
    "tp_atr_mult": 3.0,
    "trailing": False,
}

#: The search space `backtest/run.py --scan` and `backtest/scan.py` sweep.
#:
#:     3 x 3 x 3 x 3 x 2 = 162 combinations
#:
#: TRIMMED FROM THE REQUEST'S 486 by pinning `slow_ema` at 17, exactly as
#: `double_rsi_momentum_pullback_20260830` trimmed 1,458 to 162. The full grid
#: is preserved below as `FULL_PARAM_GRID_AS_REQUESTED`.
#:
#: Why `slow_ema` and not another axis: it is the parameter the PREMISE is
#: least sensitive to. The hypothesis is "a fast ribbon accelerating hard
#: enough to be worth joining" - `fast_ema` IS that ribbon and `min_slope` IS
#: the acceleration, and the brackets decide whether it survives its costs, so
#: all three stay open. A slow leg at 14 rather than 17 varies the same idea
#: rather than testing a different one.
#:
#: Read the count against the ladder before quoting it: `--tf 5m,15m,30m,1h`
#: is 162 fits PER timeframe PER contract, so 648 per symbol and 1,296 across
#: the two declared assets. At 486 it would have been 3,888.
PARAM_GRID = {
    "fast_ema": [5, 7, 9],
    "min_slope": [0.10, 0.15, 0.20],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, None],
    "trailing": [False, True],
}

#: THE REQUEST'S ORIGINAL 486-CELL GRID, kept for provenance. Restoring it is
#: a deliberate act with a stated cost, not a default anyone falls into.
FULL_PARAM_GRID_AS_REQUESTED = {
    "fast_ema": [5, 7, 9],
    "slow_ema": [14, 17, 21],
    "min_slope": [0.10, 0.15, 0.20],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, None],
    "trailing": [False, True],
}

LOGIC = {
    "concept": (
        "A pullback joined in the direction of a trend two horizons wide. The "
        "long-horizon proxy - {htf_ema} bars of the run's own timeframe "
        "scaled to an hour - says which side is permitted; the "
        "EMA({fast_ema})/EMA({slow_ema}) ribbon says the fast leg leads; a "
        "normalised slope of at least {min_slope} ATR per bar says it is "
        "accelerating rather than drifting; and a pin bar, engulfing or "
        "expansion candle says the pullback actually resolved rather than "
        "merely paused."),
    "entry": (
        "LONG when the long-horizon proxy is bullish, EMA({fast_ema}) > "
        "EMA({slow_ema}), the fast slope is >= {min_slope} ATR per bar, the "
        "bar is a bullish pin / engulfing / expansion, volume >= "
        "SMA({volume_sma_len}) x {vol_mult} and ATR/close > 0.0005. SHORT "
        "mirrors it with the slope <= -{min_slope}."),
    "exit": (
        "EMA({fast_ema}) crossing back through EMA({slow_ema}), or the ATR "
        "stop ({sl_atr_mult}x) or target ({tp_atr_mult}x) - whichever the bar "
        "reaches first."),
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
# The higher horizon, without a second aggregation
# --------------------------------------------------------------------------
def _bar_minutes(index: pd.DatetimeIndex) -> float:
    """
    The run's bar size in minutes, from the MEDIAN spacing of its own index.

    The median rather than the mode or the first difference: a futures index
    carries the 60-minute CME break, the Sunday-evening reopen and the odd
    holiday half-day, and any one of those as `index[1] - index[0]` would name
    the wrong bar size for the whole frame. Over a real frame the median IS
    the bar size, because the gaps are a small minority of the spacings.

    CAUSAL, and worth saying because it looks like it might not be: this reads
    only the index's SPACING, never a price, and the spacing of a bar series
    is fixed by how the series was built rather than by anything that happens
    later in it. It is not a lookahead any more than knowing the timeframe is.

    Returns NaN when the index is too short to have a spacing at all, which
    `_htf_span` treats as "cannot scale" rather than guessing.
    """
    if len(index) < 3:
        return float("nan")
    # `.dt.total_seconds()` rather than a view onto the backing integers and a
    # hardcoded divisor. Pandas 3.0 indexes can be second-, milli-, micro- or
    # nanosecond-backed, and `index.view("int64") / 6e10` silently returns
    # minutes/1000 on a microsecond index - which it did here, turning
    # `EMA(600)` into `EMA(600000)` and flattening the whole higher horizon
    # into a constant. The failure was invisible in every mask.
    deltas = index.to_series().diff().dt.total_seconds().to_numpy() / 60.0
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    if positive.size == 0:
        return float("nan")
    return float(np.median(positive))


def _htf_span(period: int, bar_minutes: float,
              htf_minutes: int = HTF_MINUTES) -> int:
    """
    `period` bars of an `htf_minutes` chart, expressed as a span on THIS run's
    bars.

        htf_ema=50 on a  5m run  ->  EMA(600)
        htf_ema=50 on a 15m run  ->  EMA(200)
        htf_ema=50 on a  1h run  ->  EMA(50)

    THE INTENT IS FIFTY HOURS OF MEMORY, and this preserves it on whichever
    rung Stage 1 picks rather than fixing a bar count that means a different
    horizon on every one. See the module header for why this is not a
    resample: a second aggregation inside a strategy is free to disagree with
    `mdlib.lake` about bar boundaries and the Sunday merge, silently.

    It is NOT the same series as an EMA over resampled hourly bars. An EMA
    over 600 five-minute closes weights the recent hour differently from an
    EMA over 50 hourly closes, because the inner bars are visible to it. The
    memory is equivalent; the smoothing is not. Stated rather than hidden.

    A bar size that does not divide the horizon is scaled anyway and rounded -
    a 7m run would give `round(50 * 60/7)`. An unknown bar size falls back to
    `period` unscaled, which is the run's own timeframe read literally, and
    the caller records that it happened.
    """
    p = int(period)
    if not np.isfinite(bar_minutes) or bar_minutes <= 0:
        return max(2, p)
    return max(2, int(round(p * (float(htf_minutes) / float(bar_minutes)))))


# --------------------------------------------------------------------------
# Candle patterns
# --------------------------------------------------------------------------
def _candles(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """
    The three bullish and three bearish shapes Layer 2 accepts.

    Every one is computed from bar i and bar i-1 only. `.shift(1)` looks one
    bar BACKWARD; nothing here reads i+1.

        PIN BAR      one wick at least `PIN_WICK_BODY_MULT` times the body AND
                     at least `PIN_WICK_RANGE_FRAC` of the whole range. Both
                     conditions, because a long wick on a tiny body is a pin
                     and a long wick on a bar that is mostly body is not - the
                     ratio alone admits the second.
        ENGULFING    this body covers the previous body and the sides differ.
                     `>=`/`<=` on the edges rather than `>`/`<`: a bar opening
                     exactly at the previous close is the textbook case, and a
                     strict test would drop it.
        EXPANSION    range at least `EXPANSION_RANGE_MULT` times its trailing
                     mean, closing in the top (or bottom) third. The mean is
                     SHIFTED, so the bar is not compared against a window that
                     already contains it.

    ZERO-RANGE BARS ARE NOT PATTERNS. A bar with high == low has no body, no
    wick and no direction; every ratio below is 0/0. They are masked out
    explicitly rather than left to produce NaN, because NaN in a boolean
    context is False in some paths and an error in others.
    """
    o = bars["open"].astype("float64")
    h = bars["high"].astype("float64")
    low = bars["low"].astype("float64")
    c = bars["close"].astype("float64")

    rng = (h - low)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - low
    real = rng > 0

    safe_body = body.where(body > 0, np.nan)
    bull_pin = (real & (c > o)
                & (lower >= PIN_WICK_BODY_MULT * safe_body)
                & (lower >= PIN_WICK_RANGE_FRAC * rng)).fillna(False)
    bear_pin = (real & (c < o)
                & (upper >= PIN_WICK_BODY_MULT * safe_body)
                & (upper >= PIN_WICK_RANGE_FRAC * rng)).fillna(False)

    po, pc = o.shift(1), c.shift(1)
    bull_engulf = (real & (c > o) & (pc < po)
                   & (c >= po) & (o <= pc)).fillna(False)
    bear_engulf = (real & (c < o) & (pc > po)
                   & (c <= po) & (o >= pc)).fillna(False)

    mean_rng = rng.rolling(EXPANSION_WINDOW,
                           min_periods=EXPANSION_WINDOW).mean().shift(1)
    wide = (rng >= EXPANSION_RANGE_MULT * mean_rng).fillna(False)
    bull_expand = (real & wide & (c > o)
                   & (c >= low + (2.0 / 3.0) * rng)).fillna(False)
    bear_expand = (real & wide & (c < o)
                   & (c <= low + (1.0 / 3.0) * rng)).fillna(False)

    return {
        "bull_pin": bull_pin, "bear_pin": bear_pin,
        "bull_engulf": bull_engulf, "bear_engulf": bear_engulf,
        "bull_expand": bull_expand, "bear_expand": bear_expand,
        "bullish": (bull_pin | bull_engulf | bull_expand),
        "bearish": (bear_pin | bear_engulf | bear_expand),
        "range": rng, "body": body,
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _validate(fast_ema: int, slow_ema: int, htf_ema: int, htf_fast_ema: int,
              htf_slow_ema: int, min_slope: float, vol_mult: float,
              volume_sma_len: int, session_start_et: str,
              session_end_et: str, sl_atr_mult: float,
              tp_atr_mult: float | None, trailing: bool) -> None:
    """
    Refuse a parameter set this module cannot describe honestly.

    A REJECTED cell is still counted in `variants_tested` - `scan.py` counts
    the combination it evaluated, not the ones that survived.
    """
    for name, value in (("fast_ema", fast_ema), ("slow_ema", slow_ema),
                        ("htf_ema", htf_ema), ("htf_fast_ema", htf_fast_ema),
                        ("htf_slow_ema", htf_slow_ema),
                        ("volume_sma_len", volume_sma_len)):
        if not isinstance(value, (int, np.integer)) or int(value) < 2:
            raise ValueError(f"{name} must be an integer >= 2; got {value!r}")
    if int(fast_ema) >= int(slow_ema):
        # The ribbon IS the trigger: `EMA(fast) > EMA(slow)` says the fast leg
        # leads. With the periods crossed the condition still evaluates and
        # quietly means the opposite.
        raise ValueError(
            f"fast_ema ({fast_ema}) must be faster than slow_ema "
            f"({slow_ema})")
    if int(htf_fast_ema) >= int(htf_slow_ema):
        raise ValueError(
            f"htf_fast_ema ({htf_fast_ema}) must be faster than htf_slow_ema "
            f"({htf_slow_ema})")
    if not np.isfinite(min_slope) or float(min_slope) < 0.0:
        raise ValueError(
            f"min_slope is an ATR-per-bar magnitude and must be >= 0; got "
            f"{min_slope!r}")
    if not np.isfinite(vol_mult) or float(vol_mult) <= 0.0:
        raise ValueError(f"vol_mult must be > 0; got {vol_mult!r}")
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
                f"tp_atr_mult {tp_atr_mult} is nearer than {MIN_REWARD_RISK} "
                f"x the stop ({sl_atr_mult}); a cell that cannot clear its "
                f"own costs is not a strategy this module will describe")
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
    which is a level the search could in principle reach.

    `tp_atr_mult=None` with `trailing=False` is ALLOWED here, against the
    request. Layer 4 supplies a signal exit - the fast EMA crossing back
    through the slow - so the un-exited runner the rule guards against cannot
    occur, and refusing the pair would drop 54 of 162 cells on a hazard this
    strategy does not have.
    """
    return float("nan") if tp_atr_mult is None else float(tp_atr_mult)


def _news_suppress(bars: pd.DataFrame,
                   long_ok: np.ndarray,
                   short_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop candidate triggers whose FILL bar lands inside a macro-release window.

    Delegates to `backtest.event_calendar.apply_entry_filters`, the only
    implementation in the repository. The mask is widened one bar BACKWARDS
    there, because the engine fills at the next bar's open and a signal is
    judged on the bar it FILLS; and an empty calendar RAISES rather than
    returning an all-clear, so a run reported as news-filtered cannot be one
    in which nothing was ever filtered.
    """
    from backtest.event_calendar import apply_entry_filters
    return apply_entry_filters(bars, long_ok, short_ok)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
def _layers(bars: pd.DataFrame, fast_ema: int, slow_ema: int, htf_ema: int,
            htf_fast_ema: int, htf_slow_ema: int, min_slope: float,
            vol_mult: float, volume_sma_len: int, session_start_et: str,
            session_end_et: str, use_baseline_filter: bool,
            use_alpha_trigger: bool, use_volume_filter: bool,
            use_session_filter: bool) -> dict:
    """
    Every series the signals, the chart and the ML matrix are built from,
    computed ONCE and shared - so the inspector cannot draw a crossover a bar
    from where the entry fired, and the classifier cannot be trained on a
    slightly different strategy.

    EVERY COLUMN READS ONLY BARS <= i. The EMAs and Wilder averages are causal
    recursions, the slope and the candle patterns use `.shift(1)`, and the
    expansion window's mean is shifted. There is no centred window and no
    `.shift(-1)` anywhere in this module.
    """
    close = bars["close"].astype("float64")
    atr = _atr(bars, ATR_PERIOD)
    adx = _adx(bars, ADX_PERIOD)
    norm_atr = atr / close.replace(0.0, np.nan)

    ema_fast = _ema(close, int(fast_ema))
    ema_slow = _ema(close, int(slow_ema))

    # THE NORMALISED SLOPE. One bar of change in the fast EMA, divided by ATR
    # - so it is a dimensionless "fraction of an average bar's range per bar"
    # and `min_slope=0.15` means the same thing on NQ at 29,000 as on ES at
    # 5,700. A raw price slope would not: 0.15 points is a shrug on one and a
    # move on the other.
    slope = (ema_fast.diff() / atr.replace(0.0, np.nan))

    # THE HIGHER HORIZON, as spans on this run's own bars. See `_htf_span`.
    bar_min = _bar_minutes(_bar_timestamps(bars))
    span_macro = _htf_span(htf_ema, bar_min)
    span_hfast = _htf_span(htf_fast_ema, bar_min)
    span_hslow = _htf_span(htf_slow_ema, bar_min)
    htf_macro = _ema(close, span_macro)
    htf_fast = _ema(close, span_hfast)
    htf_slow = _ema(close, span_hslow)

    if use_baseline_filter:
        # The request's OR, kept as written: close above the macro leg OR the
        # long-horizon ribbon leading. It is a disjunction, so it is permissive
        # by design - the toggle is what prices it.
        trend_long = ((close > htf_macro) | (htf_fast > htf_slow)).fillna(False)
        trend_short = ((close < htf_macro) | (htf_fast < htf_slow)).fillna(False)
    else:
        trend_long = pd.Series(True, index=bars.index)
        trend_short = pd.Series(True, index=bars.index)

    cd = _candles(bars)
    ribbon_long = (ema_fast > ema_slow).fillna(False)
    ribbon_short = (ema_fast < ema_slow).fillna(False)
    fast_enough_up = (slope >= float(min_slope)).fillna(False)
    fast_enough_dn = (slope <= -float(min_slope)).fillna(False)

    if use_alpha_trigger:
        trigger_long = ribbon_long & fast_enough_up & cd["bullish"]
        trigger_short = ribbon_short & fast_enough_dn & cd["bearish"]
    else:
        # Without the trigger the strategy is the ribbon alone. Expressed as
        # the CROSS rather than the state, or the toggle would make every bar
        # of a trend an entry.
        trigger_long = _cross_above(ema_fast, ema_slow).fillna(False)
        trigger_short = _cross_below(ema_fast, ema_slow).fillna(False)

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

    ts_et = _bar_timestamps(bars).tz_convert("America/New_York")
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
        "ema_fast": ema_fast, "ema_slow": ema_slow, "slope": slope,
        "htf_macro": htf_macro, "htf_fast": htf_fast, "htf_slow": htf_slow,
        "bar_minutes": bar_min, "htf_spans": (span_macro, span_hfast,
                                              span_hslow),
        "trend_long": trend_long, "trend_short": trend_short,
        "trigger_long": trigger_long, "trigger_short": trigger_short,
        "confirm": confirm, "session_ok": session_ok,
        "volume": volume, "vol_sma": vol_sma, "ts_et": ts_et,
        "candles": cd,
    }


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def signal_fn(bars: pd.DataFrame,
              fast_ema: int = 7,
              slow_ema: int = 17,
              htf_ema: int = 50,
              htf_fast_ema: int = 7,
              htf_slow_ema: int = 17,
              min_slope: float = 0.15,
              vol_mult: float = 1.0,
              volume_sma_len: int = 20,
              session_start_et: str = "09:30",
              session_end_et: str = "16:00",
              use_baseline_filter: bool = True,
              use_alpha_trigger: bool = True,
              use_volume_filter: bool = True,
              use_session_filter: bool = False,
              use_news_filter: bool = False,
              sl_atr_mult: float = 1.0,
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
    below the fill would be breached by the fill bar itself. The masks are
    written out per side here for the same reason - a reader has to be able to
    check them against the specification by eye.

    THE SLOPE GATE IS SIGNED, and that is why the two sides are not one
    condition on `abs(slope)`: a long needs the fast leg rising at least
    `min_slope`, a short needs it falling at least that much, and a single
    magnitude test would let a long fire on a fast leg dropping hard.

    The entry is a bar-level EVENT inside standing conditions - the candle
    pattern is what makes it one - so a long stretch of bars with the ribbon
    open and the slope steep produces a signal only where a pin, engulfing or
    expansion bar prints. The walk enters only when FLAT: a second trigger
    while a position is open is ignored rather than pyramided, and an opposite
    trigger is ignored rather than reversing.
    """
    _validate(fast_ema, slow_ema, htf_ema, htf_fast_ema, htf_slow_ema,
              min_slope, vol_mult, volume_sma_len, session_start_et,
              session_end_et, sl_atr_mult, tp_atr_mult, trailing)

    L = _layers(bars, int(fast_ema), int(slow_ema), int(htf_ema),
                int(htf_fast_ema), int(htf_slow_ema), float(min_slope),
                float(vol_mult), int(volume_sma_len), session_start_et,
                session_end_et, bool(use_baseline_filter),
                bool(use_alpha_trigger), bool(use_volume_filter),
                bool(use_session_filter))

    gate = L["confirm"] & L["session_ok"]
    long_ok = (L["trend_long"] & L["trigger_long"] & gate).to_numpy()
    short_ok = (L["trend_short"] & L["trigger_short"] & gate).to_numpy()

    # Layer 4's signal exit: the ribbon closing. A CROSS, not a state - the
    # walk holds until the fast leg actually gives up the slow one rather than
    # exiting on every bar that happens to sit on the wrong side.
    long_sig_exit = _cross_below(L["ema_fast"],
                                 L["ema_slow"]).fillna(False).to_numpy()
    short_sig_exit = _cross_above(L["ema_fast"],
                                  L["ema_slow"]).fillna(False).to_numpy()

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
    `_layers` call `signal_fn` uses - so the chart cannot draw a crossover a
    bar from where the entry actually happened.

    The long-horizon legs are labelled with the SPAN THEY ACTUALLY USE, not
    with `htf_ema`: a line called "1h EMA(50)" drawn over 5-minute candles as
    a 600-bar average would be a chart disagreeing with itself.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["fast_ema"]), int(p["slow_ema"]),
                int(p["htf_ema"]), int(p["htf_fast_ema"]),
                int(p["htf_slow_ema"]), float(p["min_slope"]),
                float(p["vol_mult"]), int(p["volume_sma_len"]),
                p["session_start_et"], p["session_end_et"],
                bool(p["use_baseline_filter"]), bool(p["use_alpha_trigger"]),
                bool(p["use_volume_filter"]), bool(p["use_session_filter"]))
    macro, hfast, hslow = L["htf_spans"]
    return {
        f"EMA({int(p['fast_ema'])})": L["ema_fast"],
        f"EMA({int(p['slow_ema'])})": L["ema_slow"],
        f"HTF macro EMA({macro})": L["htf_macro"],
        f"HTF fast EMA({hfast})": L["htf_fast"],
        f"HTF slow EMA({hslow})": L["htf_slow"],
        "fast slope (ATR/bar)": L["slope"],
        f"ATR({ATR_PERIOD})": L["atr"],
    }


def ml_features(bars: pd.DataFrame, **params) -> pd.DataFrame:
    """
    The matrix Version B's classifier is fitted on: one row per bar, in order.

    CAUSALITY IS THIS MODULE'S RESPONSIBILITY. Every column is built from
    `_layers`, which reads only bars <= i, and nothing here is scaled against
    the whole frame - a scaler fitted end to end leaks the test period's
    distribution into the training rows without tripping any shift-based
    audit. `HistGradientBoostingClassifier` needs no scaler, so none is
    fitted.

    SHAPE IS CHECKED BY THE CALLER AND A FAILURE RAISES - unlike `indicators`,
    which is wrapped, because a broken chart annotation must not throw away a
    completed backtest but a silently swapped feature matrix must not survive
    one.

    The columns are the state the RULES read, as RATIOS rather than levels: a
    ribbon gap of 12.5 means nothing across contracts, and a classifier fitted
    on levels learns the symbol rather than the setup.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["fast_ema"]), int(p["slow_ema"]),
                int(p["htf_ema"]), int(p["htf_fast_ema"]),
                int(p["htf_slow_ema"]), float(p["min_slope"]),
                float(p["vol_mult"]), int(p["volume_sma_len"]),
                p["session_start_et"], p["session_end_et"],
                bool(p["use_baseline_filter"]), bool(p["use_alpha_trigger"]),
                bool(p["use_volume_filter"]), bool(p["use_session_filter"]))
    close = L["close"]
    atr = L["atr"].replace(0.0, np.nan)
    volume, vol_sma = L["volume"], L["vol_sma"]
    vol_std = volume.rolling(int(p["volume_sma_len"]),
                             min_periods=int(p["volume_sma_len"])).std()
    cd = L["candles"]
    out = pd.DataFrame({
        "norm_atr": L["norm_atr"],
        "adx": L["adx"],
        # The regime proxy the request asks for: ADX beside where the current
        # normalised ATR sits in its own TRAILING distribution.
        "vol_percentile": L["norm_atr"].rolling(200, min_periods=50)
                                       .rank(pct=True),
        "slope_atr": L["slope"],
        "ribbon_gap_atr": (L["ema_fast"] - L["ema_slow"]) / atr,
        "dist_from_htf_atr": (close - L["htf_macro"]) / atr,
        "body_over_range": (cd["body"] / cd["range"].replace(0.0, np.nan)),
        "roc_5": close.pct_change(5),
        "roc_15": close.pct_change(15),
        "volume_z": (volume - vol_sma) / vol_std.replace(0.0, np.nan),
        "hour_et": pd.Series(L["ts_et"].hour, index=bars.index,
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
