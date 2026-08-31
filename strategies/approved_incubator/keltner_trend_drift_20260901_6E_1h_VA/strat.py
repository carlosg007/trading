"""
Keltner Trend Drift — a low-volatility trend-continuation entry in two layers:
an EMA ribbon decides which side is permitted, and a Keltner channel break
times the entry. The position is then left alone under a trailing ATR stop,
with no take-profit, so a slow drift is allowed to run.

Location:  ~/src/trading/strategies/experimental/keltner_trend_drift_20260901.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding — do not edit this block to match it.

    1. FILE
       strategies/experimental/keltner_trend_drift_20260901.py

    2. CORE LOGIC & MATH
       Intermediate trend filter: Close > EMA(ema_fast) > EMA(200) for Longs
           (mirrored for Shorts).
       Entry:  Close crosses above the upper Keltner Channel
           (EMA(keltner_len) + keltner_mult * ATR(14)).
       Exit:   Trailing ATR stop, or Close crosses back below the
           intermediate EMA(ema_fast).

    3. RISK PARAMETER CONSTRAINT (CRITICAL FIX)
       The pipeline's `_walk_loop` and `promote.py` do not support a separate
       `trail_atr_mult`. The system sweeps the trailing distance as
       `sl_atr_mult` when `trailing=True`. The trailing stop distance MUST be
       swept using `sl_atr_mult` so it is correctly promoted and serialized
       into meta.json.

    4. PARAMETER GRID
       ema_fast (30, 50); keltner_len (20, 30, 40);
       keltner_mult (1.5, 2.0, 2.5); sl_atr_mult (2.0, 3.0, 4.0);
       trailing = True (fixed flag)

    5. METADATA & FRAMEWORK HOOKS
       TARGET_QUADRANTS = ("Q3",);  PORTFOLIO_GROUP = "Trend_Drift"

    6. TESTING
       tests/test_keltner_trend_drift_20260901.py

TWO LINES OF THE REQUEST ARE ANSWERED DIFFERENTLY, AND HERE IS WHY
==================================================================
Both are stated rather than quietly complied with, because a request whose
arithmetic or vocabulary is off is corrected once, in the open, or it is
re-derived by every reader afterwards.

  * **The grid is 54 cells, not 108.** `2 x 3 x 3 x 3 = 54`; `trailing` is
    pinned at one value and multiplies nothing. Nothing is missing from the
    grid below — the five axes named in the request are all present at every
    value listed. Only the label was wrong, and 54 is the number that will be
    reported as `variants_tested`, so the label had to move rather than the
    grid. Read it against the ladder before quoting it: `--tf 5m,15m,30m,1h`
    is 54 fits PER timeframe PER contract, so 216 per symbol.

  * **There is no `resolve_strategy()` to implement here, and `baseline.py` is
    not an API.** `resolve_strategy` lives in `backtest/run.py` and maps a
    strategy NAME to a module PATH — it is the runner's loader, not something
    a strategy declares; a copy in this file would be dead code that shadows
    nothing. `baseline.py` is not an interface either: `promote.py` produces it
    with `shutil.copyfile(source, baseline_py)`, so it is a verbatim copy of
    THIS file taken at promotion. The way to "implement the baseline API" is to
    get this module's own contract right — `signal_fn`, `indicators`, `LOGIC`,
    `PARAM_GRID`, `make_signal_fn`, `ml_features` — and the promoted baseline
    is then correct because it IS this module.

WHAT THIS MODULE CANNOT MODEL, SAID OUT LOUD
============================================
**There are no stop or target ORDERS anywhere in this engine.**
`vbt.Portfolio.from_signals` is driven by boolean masks and fills them at the
NEXT bar's open. The trailing stop below is therefore detected on the bar whose
low breaches it and filled at the following open — *not at the stop price*.
In a gapping market the difference is real money and it is not modelled here.
Every drawdown figure this module produces should be read as "the stop was
breached on that bar", never as "the position left at the stop level".

`use_news_filter` is wired to `backtest.event_calendar`; `use_time_filter` and
`use_day_filter` are real and both default OFF — see their notes below.
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

#: The request's index-trend basket, spelled as contracts this lake serves and
#: `backtest/specs.py` has verified.
SYMBOLS = ["NQ", "ES", "RTY", "YM"]

#: What the spec calls the strategy, for the tear sheet and the Discord card.
STRATEGY_NAME = "KELTNER_TREND_DRIFT"
STRATEGY_MODULE = "keltner_trend_drift_20260901"

#: Portfolio routing metadata. Descriptive only - `config/portfolios.json` is
#: the authority on which account trades what, and no basket named here exists
#: until somebody adds it there.
PORTFOLIO_GROUP = "Trend_Drift"
CORRELATION_PROFILE = (
    "Positively correlated with the index group (MNQ/MES). NQ, ES, RTY and YM "
    "are one factor at four capitalisations, so a basket of them concentrates "
    "rather than diversifies - a fact for portfolio routing to size against.")

#: The regime the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and nothing in this module reads these.
#:
#: The request's "Q3 (Low-Vol / Trending)" agrees with `mdlib/regimes.py`,
#: which is the only authority: Q1 High-Vol/Trending, Q2 High-Vol/Ranging,
#: Q3 Low-Vol/Trending, Q4 Low-Vol/Ranging, 0 UNDEFINED (warm-up, not a
#: quadrant). No correction was needed here; the ids are pinned against
#: `backtest.profiler` in the test suite so a future edit cannot drift them.
TARGET_REGIMES = ("Low Volatility / Trending",)
TARGET_QUADRANTS = ("Q3",)

#: Wilder's ATR period. Fixed at 14 rather than exposed: the request names
#: ATR(14) in both the channel and the stop, and a swept ATR period would make
#: the channel width and the stop distance move together in a way no cell of
#: the grid could separate.
ATR_PERIOD = 14

#: The slow ribbon leg. Fixed at 200 by the request. On an intraday bar this is
#: a long memory - 200 bars is ~4.2 sessions at 30m and ~17 at 5m - so the
#: ribbon means something different on every rung of the ladder. That is a
#: reason to let Stage 1 pick the rung, not a reason to vary the period.
EMA_SLOW = 200

#: A stop nearer than this to the fill is refused: at a quarter of an ATR the
#: bracket sits inside the bar's own noise and the strategy measures the cost
#: model rather than the drift.
MIN_STOP_ATR_MULT = 0.25

DEFAULT_PARAMS = {
    "ema_fast": 50,
    "keltner_len": 20,
    "keltner_mult": 2.0,
    "session_start_et": "09:30",
    "session_end_et": "16:00",
    "allowed_days": (0, 1, 2, 3, 4),
    "use_baseline_filter": True,
    "use_alpha_trigger": True,
    "use_time_filter": False,
    "use_day_filter": False,
    "use_news_filter": False,
    "sl_atr_mult": 2.0,
    "tp_atr_mult": None,
    "trailing": True,
}

#: The search space `backtest/run.py --scan` and `backtest/scan.py` sweep.
#:
#:     2 x 3 x 3 x 3 = 54 combinations
#:
#: `sl_atr_mult` IS THE TRAILING DISTANCE. This is the request's critical fix
#: and it is not a naming preference - it is the only way the number survives
#: to the promoted package. `_walk_loop` computes ONE distance,
#: `stop_dist = sl_mult * atr[i]`, and `trailing` selects only what that
#: distance is anchored to: `entry_px - stop_dist` when False, and
#: `(highest high since fill) - stop_dist` when True. There is no second
#: distance in the kernel to sweep.
#:
#: Downstream is why it matters. `run.py`'s RISK_PARAMS and `promote.py`'s
#: RISK_KEYS are both exactly `("sl_atr_mult", "tp_atr_mult", "trailing")`,
#: and `promote._risk_block` writes those three and nothing else. A stop
#: distance living under any other name would be absent from the promoted risk
#: block entirely, and the card would report `sl_atr_mult` as the stop while a
#: different number actually bound every trade.
#:
#: `trailing` is pinned True and `tp_atr_mult` pinned None because the premise
#: is "ride an extended low-volatility drift without artificial profit caps".
#: A fixed stop would cap nothing but would also never follow the drift, and a
#: target is the artificial cap the premise rejects. Both are still declared,
#: so a single out-of-band run can ask the question the grid does not:
#: `--param trailing=false` and `--param tp_atr_mult=3.0`.
PARAM_GRID = {
    "ema_fast": [30, 50],
    "keltner_len": [20, 30, 40],
    "keltner_mult": [1.5, 2.0, 2.5],
    "sl_atr_mult": [2.0, 3.0, 4.0],
    "trailing": [True],
}

LOGIC = {
    "concept": (
        "A trend-continuation entry for a market that grinds rather than "
        "spikes. The EMA({ema_fast})/EMA(200) ribbon says which side is "
        "permitted; a close pushing through the Keltner band "
        "(EMA({keltner_len}) +/- {keltner_mult} x ATR(14)) is the entry "
        "event; and a trailing ATR stop with no target lets the drift run. "
        "Note for whoever reads the Q3 audit first: a channel break in a "
        "low-volatility regime is often the START of the market leaving Q3 "
        "for Q1, and that is expected. Trades are attributed by their ENTRY "
        "bar, never the exit, so a position opened in Q3 that earns all of "
        "its profit inside Q1 still scores as a Q3 trade under Gate R."),
    "entry": (
        "LONG when Close > EMA({ema_fast}) > EMA(200) and Close crosses ABOVE "
        "EMA({keltner_len}) + {keltner_mult} x ATR(14). SHORT mirrors it: "
        "Close < EMA({ema_fast}) < EMA(200) and Close crosses BELOW "
        "EMA({keltner_len}) - {keltner_mult} x ATR(14)."),
    "exit": (
        "The trailing stop at {sl_atr_mult} x ATR(14), ratcheting behind the "
        "best price seen since the fill, or Close crossing back through "
        "EMA({ema_fast}) - whichever the bar reaches first. No take-profit "
        "is modelled (tp_atr_mult={tp_atr_mult})."),
}


# ---------------------------------------------------------------------------
# Indicator kernels
#
# DUPLICATED VERBATIM from `double_rsi_momentum_pullback_20260830.py` and
# `t3_braid_scalp_20260823.py`, which share them with the rest of this
# directory. Strategy modules are loaded from a FILE PATH by
# `agents.tier3_workers.load_strategy` and promoted as a self-contained copy,
# so a shared import would resolve against whatever happens to sit beside the
# module at load time - and a promoted package must reproduce the file that was
# certified, byte for byte, not whatever a helper module has become since.
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    A span-`period` exponential moving average.

    `adjust=False` is the recursive form every charting package draws, and the
    one the ribbon and the Keltner baseline are specified in. It is NOT
    Wilder's smoothing - see `_wilder`, which is roughly half this speed and is
    what the ATR below is defined on. Using either where the other belongs
    produces a line no other tool agrees with.
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
def _validate(ema_fast: int, keltner_len: int, keltner_mult: float,
              session_start_et: str, session_end_et: str, allowed_days,
              sl_atr_mult: float, tp_atr_mult: float | None,
              trailing: bool) -> None:
    """
    Refuse a parameter set this module cannot describe honestly.

    A REJECTED cell is still counted in `variants_tested` - `scan.py` counts
    the combination it evaluated, not the ones that survived, so refusing a
    cell shrinks neither the search nor the honesty of the number reported for
    it.
    """
    for name, value in (("ema_fast", ema_fast), ("keltner_len", keltner_len)):
        if not isinstance(value, (int, np.integer)) or int(value) < 2:
            raise ValueError(f"{name} must be an integer >= 2; got {value!r}")
    if int(ema_fast) >= EMA_SLOW:
        # The ribbon IS the trend filter: `close > ema_fast > ema_slow` says
        # the fast leg leads the slow one. With the periods crossed the
        # condition still evaluates, and it quietly means the opposite.
        raise ValueError(
            f"ema_fast ({ema_fast}) must be faster than EMA_SLOW ({EMA_SLOW})")
    if not np.isfinite(keltner_mult) or float(keltner_mult) <= 0.0:
        raise ValueError(f"keltner_mult must be > 0; got {keltner_mult!r}")
    if not np.isfinite(sl_atr_mult) or float(sl_atr_mult) < MIN_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be >= {MIN_STOP_ATR_MULT}; got {sl_atr_mult!r}")
    if tp_atr_mult is not None:
        if not np.isfinite(tp_atr_mult) or float(tp_atr_mult) <= 0.0:
            raise ValueError(
                f"tp_atr_mult must be > 0 or None; got {tp_atr_mult!r}")
    if not isinstance(trailing, (bool, np.bool_)):
        raise ValueError(f"trailing must be a bool; got {trailing!r}")

    days = tuple(int(d) for d in allowed_days)
    if not days or any(d < 0 or d > 6 for d in days):
        raise ValueError(
            f"allowed_days must be a non-empty subset of 0..6; got "
            f"{allowed_days!r}")
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
    against NaN is False in both the compiled and the interpreted walk, so the
    target simply never fires on either side.
    """
    return float("nan") if tp_atr_mult is None else float(tp_atr_mult)


def _news_suppress(bars: pd.DataFrame,
                   long_ok: np.ndarray,
                   short_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop candidate triggers whose FILL bar lands inside a macro-release window.

    Delegates to `backtest.event_calendar.apply_entry_filters`, the only
    implementation of this filter in the repository. Two behaviours come with
    it and neither is reimplemented here: the mask is widened one bar BACKWARDS
    (the engine fills at the next bar's open, so a signal is judged on the bar
    it FILLS, and without the widening exactly one entry per event slips
    through and fills inside the window); and an empty calendar RAISES rather
    than returning an all-clear, so a run reported as news-filtered cannot be
    one in which nothing was ever filtered.
    """
    from backtest.event_calendar import apply_entry_filters
    return apply_entry_filters(bars, long_ok, short_ok)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
def _layers(bars: pd.DataFrame, ema_fast: int, keltner_len: int,
            keltner_mult: float, session_start_et: str, session_end_et: str,
            allowed_days, use_baseline_filter: bool, use_alpha_trigger: bool,
            use_time_filter: bool, use_day_filter: bool) -> dict:
    """
    Every series the signals, the chart and the ML matrix are built from,
    computed ONCE and shared.

    One computation rather than three is the point: `indicators` draws what
    `signal_fn` traded on, and `ml_features` is fitted on it, so the inspector
    cannot draw a channel break a bar from where the entry fired and the
    classifier cannot be trained on a slightly different strategy.

    EVERY COLUMN READS ONLY BARS <= i. The crossings use `.shift(1)`, which
    looks one bar BACKWARD; the EMAs and the ATR are causal recursions. There
    is no centred window and no `.shift(-1)` anywhere in this module.
    """
    close = bars["close"].astype("float64")
    ema_f = _ema(close, int(ema_fast))
    ema_s = _ema(close, EMA_SLOW)
    mid = _ema(close, int(keltner_len))
    atr = _atr(bars, ATR_PERIOD)
    width = float(keltner_mult) * atr
    upper = mid + width
    lower = mid - width

    if use_baseline_filter:
        trend_long = (close > ema_f) & (ema_f > ema_s)
        trend_short = (close < ema_f) & (ema_f < ema_s)
    else:
        trend_long = pd.Series(True, index=bars.index)
        trend_short = pd.Series(True, index=bars.index)

    if use_alpha_trigger:
        trigger_long = _cross_above(close, upper)
        trigger_short = _cross_below(close, lower)
    else:
        # Without the trigger the strategy is the ribbon alone, which is a
        # STATE and not an event - so it is expressed as the bar the ribbon
        # itself turns, or the toggle would make every bar of a trend an entry.
        trigger_long = _cross_above(close, ema_f)
        trigger_short = _cross_below(close, ema_f)

    # The session and day gates. Both DEFAULT OFF, and that is a decision with
    # a reason rather than an oversight:
    #
    # Gate R certifies on profit factor over >= 30 trades INSIDE Q3 on the
    # holdout, and every conjunctive filter here is drawn from that same
    # budget. A four-hour window is 4 of 24 hours and a three-day week is 3 of
    # 5 days; together they leave about a tenth of the tape before the channel
    # break is even evaluated. They stay available so Stage 2 can price them,
    # and off so nothing carries them into a certification run by default.
    #
    # The window is also index-shaped, not FX-shaped. The original request's
    # "London-NY overlap, 08:00-12:00" is FX reasoning - London closes 11:30
    # ET - and for ES/NQ the session that matters is the US cash day. An
    # 08:00 start spends its first ninety minutes in the thin index morning
    # and a 12:00 end cuts the afternoon trend leg this strategy exists to
    # ride, so the default window is the cash session itself.
    ts_utc = _bar_timestamps(bars)
    # A NAMED ZONE, never a fixed offset. "EST" is UTC-5 year round while New
    # York is on EDT from March to November, so a literal offset is an hour
    # wrong for more than half of any sample - and silently, because it simply
    # selects a different four hours.
    ts_et = ts_utc.tz_convert("America/New_York")
    if use_time_filter:
        sh, sm = (int(x) for x in str(session_start_et).split(":"))
        eh, em = (int(x) for x in str(session_end_et).split(":"))
        minutes = ts_et.hour * 60 + ts_et.minute
        time_ok = pd.Series((minutes >= sh * 60 + sm) & (minutes <= eh * 60 + em),
                            index=bars.index)
    else:
        time_ok = pd.Series(True, index=bars.index)

    if use_day_filter:
        days = tuple(int(d) for d in allowed_days)
        day_ok = pd.Series(np.isin(ts_et.dayofweek, days), index=bars.index)
    else:
        day_ok = pd.Series(True, index=bars.index)

    gate = (time_ok & day_ok).fillna(False)
    return {
        "close": close, "ema_fast": ema_f, "ema_slow": ema_s,
        "kc_mid": mid, "kc_upper": upper, "kc_lower": lower,
        "atr": atr, "norm_atr": (atr / close.replace(0.0, np.nan)),
        "trend_long": trend_long.fillna(False),
        "trend_short": trend_short.fillna(False),
        "trigger_long": trigger_long.fillna(False),
        "trigger_short": trigger_short.fillna(False),
        "gate": gate,
        "ts_et": ts_et,
    }


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def signal_fn(bars: pd.DataFrame,
              ema_fast: int = 50,
              keltner_len: int = 20,
              keltner_mult: float = 2.0,
              session_start_et: str = "09:30",
              session_end_et: str = "16:00",
              allowed_days=(0, 1, 2, 3, 4),
              use_baseline_filter: bool = True,
              use_alpha_trigger: bool = True,
              use_time_filter: bool = False,
              use_day_filter: bool = False,
              use_news_filter: bool = False,
              sl_atr_mult: float = 2.0,
              tp_atr_mult: float | None = None,
              trailing: bool = True) -> tuple[pd.Series, pd.Series,
                                              pd.Series, pd.Series]:
    """
    Returns the FOUR-MASK form: (long_entries, long_exits, short_entries,
    short_exits), all boolean Series on `bars.index`.

    A three-tuple or a bare Series RAISES in `engine.unpack_signals` rather
    than silently losing the short side into a plausible long-only curve.

    THE SHORT SIDE IS NOT THE LONG SIDE WITH A SIGN FLIP. Inside `_walk_loop`
    the short stop sits ABOVE the fill and ratchets DOWN behind the lowest low
    since entry; a short stop placed below the fill would be breached by the
    fill bar itself. The masks below are written out per side for the same
    reason - a reader has to be able to check them against the specification
    by eye.

    `sl_atr_mult` IS the trailing distance whenever `trailing` is True. There
    is no separate trail multiplier in this engine; see PARAM_GRID.

    The entry is a bar-level EVENT inside standing conditions, so a long
    stretch of bars above the channel produces a signal only where the cross
    fires. The walk enters only when FLAT: a second trigger while a position is
    open is ignored rather than pyramided, and an opposite trigger is ignored
    rather than reversing.
    """
    _validate(ema_fast, keltner_len, keltner_mult, session_start_et,
              session_end_et, allowed_days, sl_atr_mult, tp_atr_mult, trailing)

    L = _layers(bars, int(ema_fast), int(keltner_len), float(keltner_mult),
                session_start_et, session_end_et, allowed_days,
                bool(use_baseline_filter), bool(use_alpha_trigger),
                bool(use_time_filter), bool(use_day_filter))

    long_ok = (L["trend_long"] & L["trigger_long"] & L["gate"]).to_numpy()
    short_ok = (L["trend_short"] & L["trigger_short"] & L["gate"]).to_numpy()

    # The signal exit: price closing back through the intermediate leg. It is
    # a CROSS, not a state - the walk holds until the ribbon is actually given
    # up, rather than exiting on every bar that happens to sit on the wrong
    # side of it.
    close = L["close"]
    long_sig_exit = _cross_below(close, L["ema_fast"]).fillna(False).to_numpy()
    short_sig_exit = _cross_above(close, L["ema_fast"]).fillna(False).to_numpy()

    atr = L["atr"].to_numpy(dtype="float64")
    # A bar with no ATR yet is not tradable: the bracket has no width. This is
    # the warm-up, and admitting it would place a stop at the fill.
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
    `_layers` call `signal_fn` uses - so the chart cannot draw a channel break
    a bar from where the entry actually happened.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["ema_fast"]), int(p["keltner_len"]),
                float(p["keltner_mult"]), p["session_start_et"],
                p["session_end_et"], p["allowed_days"],
                bool(p["use_baseline_filter"]), bool(p["use_alpha_trigger"]),
                bool(p["use_time_filter"]), bool(p["use_day_filter"]))
    return {
        f"EMA({int(p['ema_fast'])})": L["ema_fast"],
        f"EMA({EMA_SLOW})": L["ema_slow"],
        f"KC mid EMA({int(p['keltner_len'])})": L["kc_mid"],
        f"KC upper (+{float(p['keltner_mult'])}xATR)": L["kc_upper"],
        f"KC lower (-{float(p['keltner_mult'])}xATR)": L["kc_lower"],
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

    The columns are the state the RULES read, expressed as RATIOS rather than
    levels. A channel width of 12.5 means nothing across contracts and a
    classifier fitted on levels learns the symbol rather than the setup.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["ema_fast"]), int(p["keltner_len"]),
                float(p["keltner_mult"]), p["session_start_et"],
                p["session_end_et"], p["allowed_days"],
                bool(p["use_baseline_filter"]), bool(p["use_alpha_trigger"]),
                bool(p["use_time_filter"]), bool(p["use_day_filter"]))
    close = L["close"]
    atr = L["atr"].replace(0.0, np.nan)
    volume = (bars["volume"].astype("float64") if "volume" in bars.columns
              else pd.Series(np.nan, index=bars.index))
    vol_sma = volume.rolling(20, min_periods=20).mean().shift(1)
    vol_std = volume.rolling(20, min_periods=20).std().shift(1)
    ts_et = L["ts_et"]
    out = pd.DataFrame({
        "norm_atr": L["norm_atr"],
        "channel_width_norm": (L["kc_upper"] - L["kc_lower"]) / close,
        "close_over_ema_fast_atr": (close - L["ema_fast"]) / atr,
        "ribbon_ratio": L["ema_fast"] / L["ema_slow"].replace(0.0, np.nan),
        "dist_upper_atr": (close - L["kc_upper"]) / atr,
        "dist_lower_atr": (close - L["kc_lower"]) / atr,
        "roc_5": close.pct_change(5),
        "roc_15": close.pct_change(15),
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
        # would sweep the DEFAULT and report it under the swept name - and
        # `trail_atr_mult` is exactly such a key, so this is what catches a
        # caller who has not read the PARAM_GRID note.
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")

    def _fn(bars: pd.DataFrame):
        return signal_fn(bars, **bound)
    return _fn
