"""
energy_intraday_nonlinear_ar_20260902 - nonlinear intraday momentum in NYMEX
energy, restricted to bull-regime open-outcry hours.

THE PREMISE
===========
Wang & Yang (2009) report that high-frequency energy futures - heating oil and
natural gas in particular - show nonlinear directional predictability that
exists ONLY in bull market conditions, and that a functional-coefficient
autoregressive form earns a positive Mean Forecast Trading Return over the
NYMEX open-outcry session. The trade expressed here is that reading: a macro
uptrend gate (Close > EMA(200)), a trending-market gate (ADX(14) > threshold),
an autoregressive momentum trigger (this bar's return positive) confirmed by a
functional-coefficient proxy `Ut = Close - SMA(L) > 0`, taken only inside
09:00-14:30 America/New_York.

Why it might persist: the open-outcry window is when physical hedgers transact,
and their flow is one-directional and price-insensitive over hours rather than
seconds. The asymmetry - predictability in bull states and not bear states - is
the part most likely to be a 2009 artefact, and Gate 3 on the 2023+ holdout is
what settles that. It is stated here so a reader can see what would falsify it.

FOUR PLACES THE REQUEST DID NOT MATCH THE ENGINE
================================================
Stated here rather than approximated silently, per the strategy-request rules.

1. **THE REQUEST'S QUADRANT ID IS WRONG AND IS CORRECTED HERE.** It asks for
   "Q2: High Vol / Trending". In `mdlib/regimes.py` - the only authority, and
   the numbering every cached quadrant, Stage 1 designation and Gate R verdict
   is drawn on - Q2 is High-Volatility/**RANGING** and High-Vol/Trending is
   **Q1**. The label and the digit cannot both be honoured. The LABEL is what
   this module follows, because the request's own Layer 1 requires ADX(14) > 25,
   which is the trending condition by definition: a strategy that only fires
   above the ADX threshold cannot be certified in a ranging quadrant. So
   `TARGET_QUADRANTS = ("Q1",)`.

   This is not a cosmetic fix. A strategy registered under the wrong quadrant id
   is stood down in the environment it was certified for and turned loose in the
   one it never traded, and every log line reads correctly while it happens -
   `config/portfolios.json` schema 1.0.0 shipped exactly that failure once.

2. **"BULL MARKETS ONLY" IS NOT A QUADRANT AXIS.** The four quadrants are
   volatility x trend and carry no direction - there is no bull or bear
   quadrant to certify into. The bull restriction is real and is enforced where
   it belongs, in Layer 1: `Close > EMA(200)` plus a long-only mask. Nothing in
   `TARGET_QUADRANTS` expresses it and nothing should.

3. **THERE ARE NO STOP OR TARGET ORDERS.** `from_signals` is driven by boolean
   masks and fills at the next bar's OPEN. A stop is detected on the bar that
   breaches it and filled one bar later, NOT at the stop price. Every drawdown
   figure this module produces means "the stop was breached on that bar", never
   "the position left at the stop level". In a gapping market - and energy gaps
   on inventory prints - the difference is real money and is not modelled.

4. **THE 14:30 FLATTEN EXITS AT THE NEXT BAR'S OPEN, i.e. 15:00 ET.** Same
   cause. The flatten is a signal like any other, so a position open into 14:30
   is signalled out on that bar and filled at 15:00's open. It is a genuine
   half-hour of exposure past the stated flatten and it is not a bug in the
   module; `--flat-by-close` has the same property. A reader comparing this to
   a broker's flat-at-14:30 will see the difference and it is accounted for
   here.

SESSION TIMES GO THROUGH A NAMED ZONE
=====================================
`America/New_York`, never a fixed offset. "EST" is UTC-5 year round while New
York is on EDT from March to November, so a literal offset is an hour wrong for
more than half of any sample - and silently, because it simply selects a
different five and a half hours. `--flat-by-close` uses a fixed
`session_close_utc` and is therefore REDUNDANT with, and slightly wrong
against, this module's own flatten; do not pass it.

Bars are stamped when they OPEN (the lake resamples `label="left"`), so the
14:00 bar on a 30m feed covers 14:00-14:30 and the entry window is the
half-open interval [09:00, 14:30).

`use_news_filter` is wired to `backtest.event_calendar`. Energy is the group
where that matters most - the EIA petroleum status report is 10:30 ET
Wednesday, inside this window every week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

#: The bar this was designed on. The premise is explicitly a 30-minute result,
#: so this is the rung the claim is about rather than one Stage 1 picked.
TIMEFRAME = "30m"

#: The request's energy basket, spelled as contracts this lake serves. All
#: three reconcile clean against Databento definitions in
#: `backtest.specs.verify_specs()` - checked 2026-09-02, no problem rows.
SYMBOLS = ["HO", "NG", "MCL"]

#: Pinned EXACTLY as the request spells it, for bt-stage1 logging and JSON
#: pipeline tracking. It also appears in LOGIC["strategy_id"].
STRATEGY_ID = "energy_intraday_nonlinear_ar"
STRATEGY_NAME = "ENERGY_INTRADAY_NONLINEAR_AR"
STRATEGY_MODULE = "energy_intraday_nonlinear_ar_20260902"

#: Portfolio routing metadata. DESCRIPTIVE ONLY - `config/portfolios.json` is
#: the authority on which account trades what, and no basket named here exists
#: until somebody adds it there.
PORTFOLIO_GROUP = "Energy_Momentum_Basket"
CORRELATION_PROFILE = (
    "Diversifier against the index and metals books: HO, NG and MCL are driven "
    "by physical supply, weather and inventory rather than by rates or "
    "earnings. Note the basket is NOT internally uncorrelated - HO and MCL are "
    "both petroleum-complex and move together on crude; NG is the genuinely "
    "independent leg. Routing should size HO and MCL as closer to one bet than "
    "three.")

#: The regime the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and NOTHING IN THIS MODULE READS THESE.
#:
#: The request said "Q2: High Vol / Trending". Q2 is High-Vol/RANGING; the
#: label it gave names Q1. See point 1 of the module docstring for why the
#: label wins. The ids are pinned against `backtest.profiler` in the test suite
#: so a future edit cannot drift them back.
TARGET_REGIMES = ("High Volatility / Trending",)
TARGET_QUADRANTS = ("Q1",)

#: Wilder's periods. Fixed rather than exposed: the request names ATR(14) and
#: ADX(14), and a swept ATR period would make the stop distance and the
#: volatility floor move together in a way no cell of the grid could separate.
ATR_PERIOD = 14
ADX_PERIOD = 14

#: The macro bull filter. Fixed at 200 by the request. At 30m this is ~7.7
#: NYMEX sessions of memory.
EMA_MACRO = 200

#: The Layer 3 volatility floor, as ATR(14)/Close. The request's number.
#: It is a DEAD-TAPE FLOOR, not a regime gate: at 30m, MCL near $70 runs a
#: normalised ATR around 0.003 and NG around 0.006, so 0.0005 removes halted
#: and holiday bars and essentially nothing else. Calibrated against the lake
#: before it is trusted as more than that.
MIN_NORM_ATR = 0.0005

#: How wide a macro release blacks out, when `use_news_filter` is on. Shared
#: with `sma_momentum_crossover_20260818`.
NEWS_WINDOW_MINUTES = 30.0

#: A stop nearer than this to the fill is refused: at a quarter of an ATR the
#: bracket sits inside the bar's own noise and the strategy measures the cost
#: model rather than the momentum.
MIN_STOP_ATR_MULT = 0.25

DEFAULT_PARAMS = {
    "sma_L_len": 20,
    "adx_thresh": 25.0,
    "vol_sma_len": 20,
    "session_start_et": "09:00",
    "session_end_et": "14:30",
    "use_baseline_filter": True,
    "use_alpha_trigger": True,
    "use_volume_filter": True,
    "use_news_filter": False,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "trailing": True,
}

#: The search space `backtest/run.py --scan` and `backtest/scan.py` sweep.
#:
#:     3 x 3 x 3 x 3 x 2 = 162 combinations
#:
#: THAT IS A LARGE GRID AND THE NUMBER IS THE REQUEST'S, NOT A CHOICE MADE
#: HERE. 162 cells over one instrument's 30m history is enough search that the
#: best in-sample Sharpe it returns should be read as the maximum of 162 draws
#: rather than as a measurement. `variants_tested` carries the count onto every
#: result so that reading is available; Gate 3 on the untouched 2023+ holdout
#: is what actually settles it. If the in-sample winner does not survive the
#: holdout, suspect the grid size first.
#:
#: 27 CELLS ARE REFUSED BY `_validate` AND STILL COUNTED. `tp_atr_mult=None`
#: with `trailing=False` is the un-exited runner the request asks to prevent,
#: so those cells raise. `scan.py` counts the combination it EVALUATED, not the
#: ones that survived, so refusing a cell shrinks neither the search nor the
#: honesty of the number reported for it.
#:
#: `sl_atr_mult` IS THE TRAILING DISTANCE when `trailing` is True. `_walk_loop`
#: computes ONE distance, `stop_dist = sl_mult * atr[i]`, and `trailing`
#: selects only what it is anchored to: `entry_px - stop_dist` when False and
#: `(highest high since fill) - stop_dist` when True. There is no second
#: distance in the kernel to sweep, and a stop distance living under any other
#: name would be absent from the promoted risk block entirely - `run.py`'s
#: RISK_PARAMS and `promote.py`'s RISK_KEYS are both exactly
#: `("sl_atr_mult", "tp_atr_mult", "trailing")`.
PARAM_GRID = {
    "sma_L_len": [10, 20, 30],
    "adx_thresh": [20.0, 25.0, 30.0],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, None],
    "trailing": [False, True],
}

LOGIC = {
    # Pinned for bt-stage1 logging and JSON pipeline tracking.
    "strategy_id": STRATEGY_ID,
    "concept": (
        "Wang & Yang (2009): 30-minute energy futures show nonlinear "
        "directional predictability ONLY in bull states. Long-only by "
        "construction. A macro uptrend (Close > EMA(200)) and a trending tape "
        "(ADX(14) > {adx_thresh}) say the state is right; an autoregressive "
        "momentum trigger (this bar's return positive) with a "
        "functional-coefficient proxy Ut = Close - SMA({sma_L_len}) > 0 is the "
        "entry event; and the trade is confined to the NYMEX open-outcry "
        "window 09:00-14:30 ET, where the hedging flow the premise rests on "
        "actually transacts. The asymmetry is the fragile part of the premise "
        "and the 2023+ holdout is what settles it."),
    "entry": (
        "LONG when Close > EMA(200) and ADX(14) > {adx_thresh} and the bar "
        "stamp is inside [09:00, 14:30) ET, AND Close > Close[1] and "
        "Ut = Close - SMA({sma_L_len}) > 0, AND Volume > "
        "SMA(Volume, {vol_sma_len}) and ATR(14)/Close > 0.0005. "
        "SHORT: never - the premise is bull-state only."),
    "exit": (
        "Close crossing back BELOW SMA({sma_L_len}), the bracket at "
        "{sl_atr_mult} x ATR(14) (trailing={trailing}) or "
        "{tp_atr_mult} x ATR(14) as a target, or the 14:30 ET session "
        "flatten - whichever the bar reaches first. Every exit is filled at "
        "the NEXT bar's open, so the flatten leaves at 15:00 ET."),
}


# ---------------------------------------------------------------------------
# Indicator kernels
#
# DUPLICATED VERBATIM from `sma_momentum_crossover_20260818.py` and
# `keltner_trend_drift_20260901.py`, which share them with the rest of this
# directory. Strategy modules are loaded from a FILE PATH by
# `agents.tier3_workers.load_strategy` and promoted as a self-contained copy,
# so a shared import would resolve against whatever happens to sit beside the
# module at load time - and a promoted package must reproduce the file that was
# certified, byte for byte, not whatever a helper module has become since.
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    A span-`period` exponential moving average.

    `adjust=False` is the recursive form every charting package draws. It is
    NOT Wilder's smoothing - see `_wilder`, which is roughly half this speed
    and is what the ATR and ADX are defined on. Using either where the other
    belongs produces a line no other tool agrees with.
    """
    return series.astype("float64").ewm(span=int(period), adjust=False).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing - the average the ATR and the ADX are actually defined
    on.

    `alpha = 1/period` with `adjust=False` is Wilder's recursion exactly. A
    span-`period` EMA is a different, roughly twice as fast, average.

    THE RECURSION IS SEEDED BY THE EWM, NOT BY WILDER'S SMA. Wilder seeds the
    first average with a simple mean of the first `period` values;
    `ewm(adjust=False)` starts at the first observation. The two converge
    within a few dozen bars and this form is what the rest of this directory
    already uses - one convention across the directory is worth more than a
    closer match to one vendor's seeding.
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

    A ZERO ATR IS A MEASUREMENT, NOT A GAP, and is left as the 0.0 it is. A
    window in which nothing moved - a halted contract, a dead overnight hour -
    has a true ATR of zero. The request's "fill degenerate windows with 0.0" is
    satisfied at the walk boundary rather than here: `signal_fn` passes
    `np.nan_to_num(atr, nan=0.0)` into the kernel, so warm-up NaN becomes a
    zero stop DISTANCE on bars where the readiness mask already forbids an
    entry. Both are visible as an immediate exit rather than as a NaN level
    nothing ever breaches.
    """
    return _wilder(_true_range(bars), period)


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
    uptrend produce the same ADX. That is precisely why this strategy needs the
    EMA(200) baseline as well: ADX alone cannot tell a bull state from a bear
    one, and the premise is bull-only.

    CAUSAL. Both `.diff()` calls look one bar BACKWARD, `_wilder` is an
    exponential recursion over past values only, and nothing here reads a bar
    later than i. The double smoothing is why the warm-up is ~2 x period.

    TWO DEGENERATE STATES ARE MEASUREMENTS, NOT GAPS, and both are filled
    explicitly. `+DI + -DI == 0` is a window that produced no directional
    movement either way; a smoothed true range of 0 is a window in which
    nothing moved at all. Both mean "no trend", which is DX 0. Left as the 0/0
    NaN the division produces, either would propagate through the second
    smoothing and blank the ADX for the next `period` bars - silently closing
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
    dx = dx.mask(plus_di.notna() & minus_di.notna() & (di_sum == 0), 0.0)
    dx = dx.mask(atr.notna() & (atr <= 0), 0.0)
    return _wilder(dx, period)


def _as_series(value, index) -> pd.Series:
    """A scalar level as a Series on `index`; a Series is returned as-is."""
    if isinstance(value, pd.Series):
        return value
    return pd.Series(float(value), index=index)


def _cross_below(fast: pd.Series, level) -> pd.Series:
    """
    True on the bar `fast` crosses DOWN through `level` - a series or a scalar.

    The definition, written out because "cross" is used loosely elsewhere and
    the loose reading is a different strategy:

        fast[i] < level[i]  AND  fast[i-1] >= level[i-1]

    A STATE (`fast < level`) would be true for every bar of a run; this is the
    EVENT, true once. It matters for the exit: the request says "Close crosses
    below the SMA(L) baseline", and reading that as a state would exit on the
    first bar of any position opened while already below - which cannot happen
    here, since Layer 2 requires Ut > 0 - and would make the exit fire again on
    every subsequent bar rather than once.

    `>=` on the previous bar rather than `>` so a pair that was exactly equal
    and then separated counts as a cross.

    NaN-SAFE BY CONSTRUCTION: every comparison against NaN is False, so a
    warm-up bar is never a cross. `.shift(1)` looks one bar BACKWARD.
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
    `tz_convert` below rather than silently being read as ET.
    """
    if "ts" in bars.columns:
        return pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    if isinstance(bars.index, pd.DatetimeIndex):
        idx = bars.index
        return idx if idx.tz is not None else idx.tz_localize("UTC")
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
    THREE-state machine over the bars: flat, long, or short - each side under
    its own stop, target and signal exit.

    This kernel is DUPLICATED VERBATIM from
    `sma_momentum_crossover_20260818.py`, which shares it with
    `t3_braid_scalp_20260823.py`, `keltner_trend_drift_20260901.py`,
    `ema_crossover_20260821.py` and `ema_trend_filter.py`, by the same
    convention that duplicates `_wilder` and `_atr` across this directory.
    `tests/test_risk_params.py` runs the copies on identical arrays in both
    directions and requires identical output - do not "improve" one alone.

    THIS STRATEGY IS LONG-ONLY, so `short_entry_ok` is all-False and the short
    branch below never runs. The branch is kept rather than deleted for two
    reasons: the kernel has to stay byte-identical to its siblings for the
    shared test above, and a future bidirectional variant that deleted it would
    have to reimplement the short bracket - which is the exact place a mirrored
    rule is silently wrong (the short stop sits ABOVE the fill; one placed
    below is breached by the fill bar itself).

    A trailing stop cannot be a stateless mask. Its level is the extreme price
    since ENTRY offset by a fixed distance, so bar i's exit condition depends on
    which earlier bar opened the position - which depends on every entry before
    it. The fixed stop and the take-profit are anchored on the FILL PRICE, which
    is `open` on the bar after the signal, so they are path-dependent for the
    same reason. This walks the bars once and resolves entries, stops, targets,
    signal exits and the session flatten together rather than splitting them
    across layers that could disagree.

    The timeline matches the engine's. An entry signal on bar i is filled at
    bar i+1's open, so the position is live from bar i+1, the fill price is
    `open_[i + 1]`, and the extreme-price mark starts there - NOT on the signal
    bar. Both distances are frozen at `mult * ATR` as measured on the SIGNAL bar
    and never re-measured as volatility changes.

        LONG    stop     fill - dist    trailing: (highest high since fill) - dist
                target   fill + dist
                exits    low  <= stop   or  high >= target
        SHORT   stop     fill + dist    trailing: (lowest low   since fill) + dist
                target   fill - dist
                exits    high >= stop   or  low  <= target

    The trailing stop never widens on either side: it ratchets UP behind a long
    and DOWN in front of a short, tracking the best price the position has seen.

    Both sides also exit on their own `sig_exit` mask and on `flat_bar`.

    `tp_mult` is NaN when no take-profit is modelled, rather than a sentinel
    like 0 or a huge number. NaN propagates into `target` and every comparison
    against it is False, so the target simply never fires - a take-profit at
    10,000 x ATR would be a level the search could still, in principle, reach.

    A bar carrying BOTH a long and a short entry signal while flat takes
    NEITHER, matching `backtest.engine._clean_signals_ls_loop`.

    A position is never reversed directly. The walk enters only from flat.

    Exits are checked from the fill bar onward, never on the signal bar itself.

    Returns `(long_entries, long_exits, short_entries, short_exits, stop_level,
    tp_level)`. Both levels are live for every bar a position is open and NaN
    everywhere else - they are what the tear sheet draws, so the lines a reader
    sees breached are the arrays the exits were taken from rather than a second
    reconstruction of them.
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
            # resolved here because it cannot change the result. On those bars
            # the modelled fill is neither the stop price nor the target price.
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
    _walk = njit(nogil=True)(_walk_loop)
except ImportError:                     # pragma: no cover - env dependent
    # Same function, interpreted. `backtest.engine.clean_signals` degrades the
    # same way, and the fallback has to exist because a missing compiler must
    # not change which trades a strategy takes.
    _walk = _walk_loop


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _validate(sma_L_len: int, adx_thresh: float, vol_sma_len: int,
              session_start_et: str, session_end_et: str,
              sl_atr_mult: float, tp_atr_mult: float | None,
              trailing: bool) -> None:
    """
    Refuse a parameter set this module cannot describe honestly.

    A REJECTED cell is still counted in `variants_tested` - `scan.py` counts
    the combination it evaluated, not the ones that survived, so refusing a
    cell shrinks neither the search nor the honesty of the number reported.
    """
    for name, value in (("sma_L_len", sma_L_len),
                        ("vol_sma_len", vol_sma_len)):
        if not isinstance(value, (int, np.integer)) or int(value) < 2:
            raise ValueError(f"{name} must be an integer >= 2; got {value!r}")
    if not np.isfinite(adx_thresh) or not 0.0 <= float(adx_thresh) <= 100.0:
        raise ValueError(
            f"adx_thresh must be within ADX's own 0..100 range; got "
            f"{adx_thresh!r}. A threshold outside it is not a strict filter, "
            f"it is a gate that never opens or never closes.")
    if not np.isfinite(sl_atr_mult) or float(sl_atr_mult) < MIN_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be >= {MIN_STOP_ATR_MULT}; got {sl_atr_mult!r}")
    if tp_atr_mult is not None:
        if not np.isfinite(tp_atr_mult) or float(tp_atr_mult) <= 0.0:
            raise ValueError(
                f"tp_atr_mult must be > 0 or None; got {tp_atr_mult!r}")
    if not isinstance(trailing, (bool, np.bool_)):
        raise ValueError(f"trailing must be a bool; got {trailing!r}")

    # THE REQUEST'S RUNNER GUARD. `tp_atr_mult=None` with `trailing=False`
    # leaves a position with only a FIXED stop below the fill and no upside
    # exit of any kind from the bracket - the un-exited runner.
    #
    # Worth being exact about what it actually costs here, because this module
    # has two other exits the guard does not depend on: the SMA(L) cross and
    # the unconditional 14:30 flatten, either of which closes the position.
    # So the lockup this refuses is not reachable through `signal_fn`'s own
    # defaults. It is refused anyway, and not as ceremony: the flatten is the
    # only exit that cannot be switched off, and a caller who reaches for
    # `use_alpha_trigger=False` on an unswept cell is one edit away from a
    # configuration where a fixed stop is the sole bracket on a position the
    # premise expects to run. The guard costs 27 of the grid's 162 cells and
    # they are counted; see PARAM_GRID.
    if tp_atr_mult is None and not bool(trailing):
        raise ValueError(
            "tp_atr_mult=None requires trailing=True. With no target and a "
            "stop that never follows the price, the bracket has no upside "
            "exit at all and a winning position is held on the session "
            "flatten alone. Set trailing=True, or give tp_atr_mult a value.")

    for name, value in (("session_start_et", session_start_et),
                        ("session_end_et", session_end_et)):
        try:
            hh, mm = str(value).split(":")
            if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError(
                f"{name} must be 'HH:MM' in 24h ET; got {value!r}") from None
    if _minutes(session_start_et) >= _minutes(session_end_et):
        raise ValueError(
            f"session_start_et ({session_start_et}) must be before "
            f"session_end_et ({session_end_et}); an empty or inverted window "
            f"produces no entries at all, which reads as a market that never "
            f"set up.")


def _minutes(hhmm: str) -> int:
    """`'09:30'` -> 570. Minutes past midnight in whatever zone the caller
    means; this module always means America/New_York."""
    hh, mm = (int(x) for x in str(hhmm).split(":"))
    return hh * 60 + mm


def _tp_distance(tp_atr_mult: float | None) -> float:
    """
    `None` means NO target, expressed as NaN.

    NaN rather than 0.0, which would place the target at the fill and close
    every trade on its own entry bar - and NaN rather than a huge number, which
    is a level the search could in principle reach. Every comparison against
    NaN is False in both the compiled and the interpreted walk, so the target
    simply never fires on either side.
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

    THE RETURN IS A THREE-TUPLE AND IS UNPACKED AS ONE. `apply_entry_filters`
    returns `(entries, short_entries, info)`. Four modules in this directory -
    `keltner_trend_drift_20260901`, `compressed_bollinger_reversion_20260901`,
    `dual_ema_slope_scalp_20260831` and `intrinsic_alpha_engine_20260831` -
    `return apply_entry_filters(...)` straight into a two-name unpack at the
    call site, so `use_news_filter=True` raises `ValueError: too many values to
    unpack` in every one of them. This follows the older, working form in
    `sma_momentum_crossover_20260818` instead. Verified 2026-09-02.

    Imported inside the function rather than at module scope, so a strategy
    module stays importable on a box where the calendar file is absent and the
    news filter costs nothing at all - including its import - for the default
    runs that leave it off.
    """
    from backtest.event_calendar import apply_entry_filters

    if "ts" not in bars.columns:
        raise ValueError(
            "use_news_filter=True needs a `ts` column to place each bar "
            "against the release calendar; this frame has none. The engine "
            "always supplies one - a frame without it is a fixture, and "
            "filtering it against a calendar would be filtering nothing.")

    kept_long, kept_short, _info = apply_entry_filters(
        bars["ts"], long_ok, short_ok,
        news_filter=True,
        news_window_minutes=NEWS_WINDOW_MINUTES)
    return (np.asarray(kept_long, dtype=bool),
            np.asarray(kept_short, dtype=bool))


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
def _layers(bars: pd.DataFrame, sma_L_len: int, adx_thresh: float,
            vol_sma_len: int, session_start_et: str, session_end_et: str,
            use_baseline_filter: bool, use_alpha_trigger: bool,
            use_volume_filter: bool) -> dict:
    """
    Every series the signals, the chart and the ML matrix are built from,
    computed ONCE and shared.

    One computation rather than three is the point: `indicators` draws what
    `signal_fn` traded on, and `ml_features` is fitted on it, so the inspector
    cannot draw a cross a bar from where the entry fired and the classifier
    cannot be trained on a slightly different strategy.

    EVERY COLUMN READS ONLY BARS <= i. `.diff()` and `.shift(1)` look one bar
    BACKWARD; the EMAs, the ATR and the ADX are causal recursions; every
    rolling window is trailing and none is centred. There is no `.shift(-1)`
    anywhere in this module.
    """
    close = bars["close"].astype("float64")
    ema_macro = _ema(close, EMA_MACRO)
    sma_l = close.rolling(int(sma_L_len), min_periods=int(sma_L_len)).mean()
    atr = _atr(bars, ATR_PERIOD)
    adx = _adx(bars, ADX_PERIOD)

    # A zero close would be a corrupt bar, not a quiet one; NaN keeps it out of
    # the ratio rather than producing an infinity that compares True against
    # the floor.
    norm_atr = atr / close.replace(0.0, np.nan)

    # THE FUNCTIONAL-COEFFICIENT PROXY. `Ut = Close - SMA(L)`, the request's
    # spelling, in PRICE units. It is deliberately not normalised here: the
    # rules only ever test its SIGN, and `ml_features` divides it by the ATR
    # for the classifier, which is where a cross-contract scale matters.
    ut = close - sma_l

    # THE AUTOREGRESSIVE TRIGGER. `Close > Close[1]` - one lag, backwards.
    ar1_up = close > close.shift(1)

    volume = (bars["volume"].astype("float64") if "volume" in bars.columns
              else pd.Series(np.nan, index=bars.index))
    vol_sma = volume.rolling(int(vol_sma_len),
                             min_periods=int(vol_sma_len)).mean()

    ts_utc = _bar_timestamps(bars)
    # A NAMED ZONE, never a fixed offset. "EST" is UTC-5 year round while New
    # York is on EDT from March to November, so a literal offset is an hour
    # wrong for more than half of any sample - and silently, because it simply
    # selects a different five and a half hours of the day.
    ts_et = ts_utc.tz_convert("America/New_York")
    minutes = pd.Series(ts_et.hour * 60 + ts_et.minute, index=bars.index)
    start_m, end_m = _minutes(session_start_et), _minutes(session_end_et)

    # HALF-OPEN [start, end). Bars are stamped when they OPEN, so on a 30m feed
    # the 14:00 bar covers 14:00-14:30 and is the last one wholly inside the
    # outcry session; the 14:30 bar is the flatten bar and is not an entry bar.
    # An inclusive end would make the flatten bar both.
    in_session = (minutes >= start_m) & (minutes < end_m)
    # The flatten. UNCONDITIONAL - it is Layer 4's session exit and is not
    # behind `use_baseline_filter`, because a toggle that could leave a
    # position open overnight is not a filter, it is a different strategy.
    flat_bar = minutes >= end_m

    if use_baseline_filter:
        # THE BULL GATE AND THE TREND GATE TOGETHER. ADX is direction-agnostic,
        # so it cannot supply the "bull" half on its own; EMA(200) is what does.
        baseline_long = (close > ema_macro) & (adx > float(adx_thresh)) \
            & in_session
    else:
        baseline_long = pd.Series(True, index=bars.index)

    if use_alpha_trigger:
        trigger_long = ar1_up & (ut > 0.0)
    else:
        # Without the trigger the strategy is the baseline alone, which is a
        # STATE and not an event - so it is expressed as the bar the price
        # regains SMA(L), or the toggle would make every bar of an uptrend an
        # entry and the comparison against Version A would be meaningless.
        trigger_long = (close > sma_l) & (close.shift(1) <= sma_l.shift(1))

    if use_volume_filter:
        confirm_long = (volume > vol_sma) & (norm_atr > MIN_NORM_ATR)
    else:
        confirm_long = pd.Series(True, index=bars.index)

    return {
        "close": close,
        "ema_macro": ema_macro,
        "sma_l": sma_l,
        "atr": atr,
        "adx": adx,
        "norm_atr": norm_atr,
        "ut": ut,
        "ar1_up": ar1_up.fillna(False),
        "volume": volume,
        "vol_sma": vol_sma,
        "in_session": in_session,
        "flat_bar": flat_bar,
        "baseline_long": baseline_long.fillna(False),
        "trigger_long": trigger_long.fillna(False),
        "confirm_long": confirm_long.fillna(False),
        "ts_et": ts_et,
    }


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def signal_fn(bars: pd.DataFrame,
              sma_L_len: int = 20,
              adx_thresh: float = 25.0,
              vol_sma_len: int = 20,
              session_start_et: str = "09:00",
              session_end_et: str = "14:30",
              use_baseline_filter: bool = True,
              use_alpha_trigger: bool = True,
              use_volume_filter: bool = True,
              use_news_filter: bool = False,
              sl_atr_mult: float = 1.5,
              tp_atr_mult: float | None = 3.0,
              trailing: bool = True) -> tuple[pd.Series, pd.Series]:
    """
    Returns the TWO-MASK long-only form: `(entries, exits)`, both boolean
    Series on `bars.index`.

    Two masks rather than four is the correct shape for a long-only strategy
    and is not a legacy path - `engine.unpack_signals` accepts either, and
    `ema_crossover_20260821` keeps this one exercised by a real module. A
    three-tuple or a bare Series RAISES rather than silently losing a side.

    THE SHORT SIDE IS `False` BY CONSTRUCTION, NOT BY OVERSIGHT. The premise is
    that the predictability exists only in bull states; a mirrored short would
    be trading the half of the paper that reports no edge, and it would do so
    under this module's name and this module's certification.

    `sl_atr_mult` IS the trailing distance whenever `trailing` is True. There
    is no separate trail multiplier in this engine; see PARAM_GRID.

    The entry is a bar-level EVENT inside standing conditions, so a long
    stretch of bars above SMA(L) produces a signal only where the AR(1) trigger
    fires. The walk enters only when FLAT: a second trigger while a position is
    open is ignored rather than pyramided.
    """
    _validate(sma_L_len, adx_thresh, vol_sma_len, session_start_et,
              session_end_et, sl_atr_mult, tp_atr_mult, trailing)

    L = _layers(bars, int(sma_L_len), float(adx_thresh), int(vol_sma_len),
                session_start_et, session_end_et,
                bool(use_baseline_filter), bool(use_alpha_trigger),
                bool(use_volume_filter))

    long_ok = (L["baseline_long"] & L["trigger_long"]
               & L["confirm_long"]).to_numpy()
    # LONG ONLY. The array is all-False and is passed anyway, so the shared
    # kernel keeps its signature and its short branch stays the one the rest of
    # the directory is tested against.
    short_ok = np.zeros(len(bars), dtype=bool)

    # Layer 4's signal exit: the close crossing back BELOW SMA(L). A cross, not
    # a state - see `_cross_below`.
    long_sig_exit = _cross_below(L["close"], L["sma_l"]).fillna(False).to_numpy()
    short_sig_exit = np.zeros(len(bars), dtype=bool)

    atr = L["atr"].to_numpy(dtype="float64")
    # A bar with no ATR yet is not tradable: the bracket has no width. This is
    # the warm-up, and admitting it would place a stop at the fill.
    warm = ~np.isfinite(atr) | (atr <= 0.0)
    long_ok = long_ok & ~warm

    if use_news_filter:
        long_ok, short_ok = _news_suppress(bars, long_ok, short_ok)

    le, lx, _se, _sx, _stop, _tp = _walk(
        long_ok, short_ok, long_sig_exit, short_sig_exit,
        bars["open"].to_numpy(dtype="float64"),
        bars["high"].to_numpy(dtype="float64"),
        bars["low"].to_numpy(dtype="float64"),
        # THE REQUEST'S "fill degenerate windows with 0.0", applied at the
        # walk boundary. A NaN stop DISTANCE would make every comparison in the
        # kernel False and produce a position nothing ever closed; 0.0 makes
        # the stop sit at the fill, which the fill bar breaches, so a bad bar
        # is a visible immediate exit rather than an invisible runner.
        np.nan_to_num(atr, nan=0.0, posinf=0.0, neginf=0.0),
        L["flat_bar"].to_numpy(dtype=bool),
        float(sl_atr_mult), _tp_distance(tp_atr_mult), bool(trailing))

    idx = bars.index
    return pd.Series(le, index=idx), pd.Series(lx, index=idx)


def indicators(bars: pd.DataFrame, **params) -> dict[str, pd.Series]:
    """
    Full-length series drawn over the trade inspector's candles, from the same
    `_layers` call `signal_fn` uses - so the chart cannot draw a cross a bar
    from where the entry actually happened.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["sma_L_len"]), float(p["adx_thresh"]),
                int(p["vol_sma_len"]), p["session_start_et"],
                p["session_end_et"], bool(p["use_baseline_filter"]),
                bool(p["use_alpha_trigger"]), bool(p["use_volume_filter"]))
    return {
        f"EMA({EMA_MACRO})": L["ema_macro"],
        f"SMA({int(p['sma_L_len'])})": L["sma_l"],
        f"Ut = Close - SMA({int(p['sma_L_len'])})": L["ut"],
        f"ADX({ADX_PERIOD})": L["adx"],
        f"ATR({ATR_PERIOD})": L["atr"],
        f"Volume SMA({int(p['vol_sma_len'])})": L["vol_sma"],
    }


def ml_features(bars: pd.DataFrame, **params) -> pd.DataFrame:
    """
    The matrix Version B's classifier is fitted on: one row per bar, in order.

    The six columns the request names, and nothing else:

        norm_atr        ATR(14) / Close. Normalised because a raw ATR in
                        points would let the model learn "NG" instead of
                        "volatile" - HO quotes in dollars per gallon and MCL
                        in dollars per barrel.
        volume_z        Volume against its own trailing 20-bar mean and
                        standard deviation.
        adx_14          The trend-strength half of the regime proxy.
        vol_pct_rank    The volatility half: where this bar's normalised ATR
                        sits within its own TRAILING 252-bar window, in [0, 1].
                        A ROLLING rank, never a full-sample percentile - a
                        percentile taken over the whole frame is a statistic of
                        the test period leaking into the training rows, which
                        is lookahead no shift-based audit would catch. Together
                        these two are the quadrant proxy: ADX is the trend axis
                        and this is the volatility axis, the same pair
                        `mdlib/regimes.py` classifies on.
        ut_atr          The functional-coefficient spread Ut = Close - SMA(L),
                        divided by ATR so it is comparable across contracts.
        ar1_ret         The one-period lagged return, `close.pct_change(1)` -
                        the autoregressive term the premise is named for. One
                        lag BACKWARD.
        hour_et         Hour of day in America/New_York. The premise is a
                        session-shaped effect, so the classifier is allowed to
                        see where in the session it is.

    CAUSALITY IS THIS MODULE'S RESPONSIBILITY. Every column is built from
    `_layers`, which reads only bars <= i, and nothing here is scaled against
    the whole frame. `HistGradientBoostingClassifier` needs no scaler, which is
    why none is fitted.

    THE ROLLING STATISTICS ARE SHIFTED ONE BAR. `vol_sma`/`vol_std` and the
    percentile window use `.shift(1)`, so bar i's z-score and rank are measured
    against a window ENDING AT i-1. Bar i's own value is still legitimate to
    read - the engine fills at i+1's open - but excluding it keeps the
    normalisation from being partly a statistic of the value it normalises,
    which is what makes a z-score of a spike read as unremarkable.

    SHAPE IS CHECKED BY THE CALLER AND A FAILURE RAISES - unlike `indicators`,
    which is wrapped, because a broken chart annotation must not throw away a
    completed backtest but a silently swapped feature matrix must not survive
    one.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["sma_L_len"]), float(p["adx_thresh"]),
                int(p["vol_sma_len"]), p["session_start_et"],
                p["session_end_et"], bool(p["use_baseline_filter"]),
                bool(p["use_alpha_trigger"]), bool(p["use_volume_filter"]))

    close = L["close"]
    atr = L["atr"].replace(0.0, np.nan)
    volume = L["volume"]
    vol_mean = volume.rolling(20, min_periods=20).mean().shift(1)
    vol_std = volume.rolling(20, min_periods=20).std().shift(1)

    # The trailing volatility percentile. `min_periods` well below the window
    # so the column exists early in the sample rather than being 0.0 for the
    # first year and then switching on - a step change in a feature is
    # something the classifier will happily learn as a date.
    norm_atr = L["norm_atr"]
    vol_rank = norm_atr.shift(1).rolling(252, min_periods=60).rank(pct=True)

    out = pd.DataFrame({
        "norm_atr": norm_atr,
        # The 1e-8 keeps a dead-flat volume window off a divide by zero. It is
        # far below any real standard deviation, so it changes no live value.
        "volume_z": (volume - vol_mean) / (vol_std + 1e-8),
        "adx_14": L["adx"],
        "vol_pct_rank": vol_rank,
        "ut_atr": L["ut"] / atr,
        "ar1_ret": close.pct_change(1),
        "hour_et": pd.Series(L["ts_et"].hour, index=bars.index,
                             dtype="float64"),
    }, index=bars.index)
    # The request's "fill degenerate rolling windows with 0.0 to prevent silent
    # NaN propagation". Applied to the MATRIX rather than to the indicators
    # themselves, so a warm-up ATR stays NaN where `signal_fn` reads it and
    # blocks an entry, while the classifier sees a finite row.
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
