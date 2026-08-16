"""
EMA trend filter — buy the pullback inside an established up-trend.

Location:  ~/src/trading/strategies/experimental/ema_trend_filter.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding — do not edit this block to match it.

    1. STRATEGY METADATA
       Strategy Name:      ema_trend_filter
       Strategy Archetype: Trend-Following (pullback continuation)
       Primary Timeframe:  15m
       Target Assets:      NQ, ES, CL, GC

    2. CORE CONCEPT & HYPOTHESIS
       See "Hypothesis" below. Carried into LOGIC["concept"] near-verbatim.

    3. INDICATORS & PARAMETER GRID
       Indicators:         EMA(trend_period) regime, EMA(pullback_period)
                           pullback reference, ATR(14) for both risk distances.
       Default Parameters: trend_period=200, pullback_period=20,
                           sl_atr_mult=2.0, tp_atr_mult=3.0, trailing=True
       PARAM_GRID:         see below — 108 combinations.

    4. ENTRY & EXIT EXECUTION RULES
       Long Entry:   close > EMA(trend_period)          (regime)
                 AND low  <= EMA(pullback_period)       (the pullback)
                 AND close >  EMA(pullback_period)      (the recovery)
       Short Entry:  Long Only.
       Take Profit:  fill price + tp_atr_mult x ATR(14), or None.
       Stop Loss:    sl_atr_mult x ATR(14), trailing or fixed.
       Session Rules: entry signals 09:30-15:30 America/New_York; flat on the
                      last bar before 16:00 America/New_York.
       Execution Fill: next-bar open with contract-specific slippage and
                      commission.

    5. BACKTEST EXECUTION CONTROLS
       In-Sample Period:   2013-01-01 to 2022-12-31
       Phase 3 Holdout:    YES — 2023-01-01 to 2025-12-31, RESERVED. It does
                           not overlap the in-sample window and must not be
                           looked at until Gate 1 and Gate 2 are settled. Once
                           it has been seen it is no longer a holdout.
       ML Filter (B):      YES, via --ml.
       Run Mode:           Multi-Asset Runner (bt-run).

    The in-sample start respects `intraday_start_year`: pre-2013 1-minute data
    is sparse for ten symbols, and a 15m bar built from sparse minutes behaves
    differently from one built from complete ones.

Hypothesis
----------
A trend is a persistent order-flow imbalance, not a price level. While it
holds, the participants creating it — index rebalancing, systematic trend
books, and the slow accumulation of a large position — are not finished, and
they keep buying. What they do not do is buy at any price: they wait for the
inventory that fast money is trying to unload.

So the tradable moment is not the trend itself but the shakeout inside it. A
dip to the short-horizon average flushes the leveraged latecomers who bought
the extension; when price closes back above that average on the same bar, the
flush is over and the persistent buyer is still there. The long-horizon EMA is
what says the persistent buyer exists at all.

The edge is therefore a liquidity premium collected inside a trend, and it
should persist because the two participant groups have different clocks. It is
fragile in the obvious way: the same touch-and-recover looks identical on the
bar the trend actually ends, and the stop is what pays for those. That is why
the stop is a swept parameter here rather than an afterthought — but see the
warning about sweeping it under "Read this before reading the equity curve".

WHY THIS IS NOT `ema_crossover` WITH ANOTHER MASK
-------------------------------------------------
`ema_crossover` buys the EVENT of a regime beginning; this buys a dip inside a
regime that is already established. They pay for different things — an
option-like premium on the transition versus a liquidity premium on the
shakeout — and they fire on different bars: the crossover has by construction
already happened before this module's regime filter is even true. Their
leaderboards are comparable, and a strong correlation between the two would be
a finding about the market rather than a duplication in the code.

Contract — the one `backtest.engine` and `agents.tier3_workers` both call:

    signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, pd.Series]

`bars` is ONE symbol's OHLCV frame, oldest to newest, lowercase columns, with a
`ts` column in UTC. Returns `(entries, exits)` as boolean Series on bars.index.

Never hand this a multi-symbol frame. `get_bars` sorts by `(ts, symbol)`, so an
EMA over the concatenation blends unrelated contracts and produces signals that
are the right length, the right dtype, and meaningless. The engine calls this
per symbol precisely so that cannot happen.

Risk parameters
---------------
    sl_atr_mult   stop distance, in ATR(14) multiples. Required, > 0.
    tp_atr_mult   take-profit distance, in ATR(14) multiples above the FILL
                  price. `None` means no take-profit is modelled at all — not
                  a take-profit at infinity, and the tear sheet says so.
    trailing      True  → the stop ratchets with the high-water mark since the
                          fill and never widens.
                  False → the stop is fixed at `fill price - sl_atr_mult x ATR`
                          and never moves in either direction.

Both distances are frozen at `ATR(14)` as measured on the SIGNAL bar and never
re-measured.

Read this before reading the equity curve
-----------------------------------------
Six things about this module are NOT what a reader would assume from the rules
above. None of them flatters the result, but a drawdown, a win rate or a trade
count read without knowing them is being read wrong.

1. THE STOP AND THE TARGET ARE FILLED AT THE NEXT BAR'S OPEN, NOT AT THEIR OWN
   PRICES. The engine fills every signal at the next bar's open; there are no
   stop or limit ORDERS anywhere in it. A breach is detected on the bar whose
   low (stop) or high (target) crosses the level and executed at the following
   bar's open — worse than the level on a fast move, better on a snapback. A
   live broker would fill at or near the level. Treat both exit prices here as
   an approximation with a one-bar lag, and do not read the drawdown as what a
   live account would have taken.

2. THE STOP-VERSUS-TARGET RACE INSIDE A BAR IS NOT RESOLVED, AND DOES NOT NEED
   TO BE. When a bar breaches both levels, a real bracket order fills at
   whichever came first and the two prices differ. Here both produce the same
   exit signal on the same bar and the same next-bar-open fill, so the order
   inside the bar changes nothing about the result. That is a consequence of
   the one-bar lag above, not a resolution of the ambiguity: on those bars the
   modelled fill is neither the stop price nor the target price.

3. THE TRAILING STOP IS A HIGH-WATER TRAIL, NOT A FIXED OFFSET FROM ENTRY.
   With `trailing=True` the level is the highest price reached since the
   position went live, minus the frozen distance. The high-water mark starts on
   the FILL bar, not the signal bar — including the signal bar's high would
   credit the position with a price it never held through and set the first
   stop too far away. With `trailing=False` the anchor is the actual fill
   price, `open` on the fill bar, and not the signal bar's close.

4. THE EXITS ARE COMPUTED IN THIS MODULE, NOT BY THE ENGINE. A trailing stop is
   path-dependent, so it cannot be a stateless boolean mask over bars, and once
   the walk exists the fixed stop and the target belong in it too rather than
   being split across layers that could disagree. `_walk` below resolves
   entries, the stop, the target, the regime-loss exit and the session flatten
   together. That is why this module contains session logic at all, which
   strategies otherwise must not.

   `_walk` here is an independent COPY of the one in `ema_crossover.py`, not an
   import. Strategy modules are loaded from a file path and are deliberately
   self-contained — the same reason `_wilder`, `_atr` and `_session_masks` are
   duplicated across every module in this directory. The copies are pinned
   against each other in `tests/test_risk_params.py`, which runs both walks on
   the same arrays and requires identical output; a divergence fails there
   rather than showing up as two strategies that disagree about what a stop is.

5. THE RISK PARAMETERS ARE SWEPT, WHICH IS A KNOWN WAY TO FIT NOISE. `--scan`
   varies `sl_atr_mult`, `tp_atr_mult` and `trailing` alongside the periods, so
   the search can find the risk settings that happened to survive this sample's
   worst few days. If the winning cell's edge lives mostly in the risk
   parameters rather than in the pullback, read that as evidence against the
   strategy rather than for it, and check the neighbouring cells in
   `scan_<SYMBOL>.csv` before believing any of it.

6. `--flat-by-close` IS NOT NEEDED AND SHOULD NOT BE PASSED. This module emits
   its own session exit on the last bar starting before 16:00 ET, converted
   through America/New_York so it is right on both sides of a DST change. The
   engine's flag uses a FIXED UTC time, which is 16:00 ET in summer and 15:00
   ET in winter. Passing it is harmless — `clean_signals` keeps whichever exit
   comes first, and this one always does — but it is redundant and its winter
   boundary is an hour off.

Every calculation is causal. Values at bar i use bars <= i only, warm-up stays
NaN and is never treated as a signal, and the engine then fills at bar i+1's
open — so nothing here can see a price it would not have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME = "15m"
SYMBOLS = ["NQ", "ES", "CL", "GC"]
DEFAULT_PARAMS = {"trend_period": 200, "pullback_period": 20,
                  "sl_atr_mult": 2.0, "tp_atr_mult": 3.0, "trailing": True}

# The search space `backtest/run.py --scan` sweeps, declared here because this
# module is the only place that knows what these parameters mean and what the
# signature will accept. 2 x 3 x 3 x 3 x 2 = 108 combinations, every one of
# them valid (every pullback value is below every trend value), so the scan
# reports 108 evaluated rather than 108 attempted and some rejected.
#
# 108 is at the top of what can be reported honestly. Each combination is a fit
# to the same in-sample bars, and the best of 108 is a higher number than the
# best of 9 even when nothing in the market has changed — `variants_tested` is
# carried onto every report and every leaderboard row for exactly that reason.
# The indicator axes are deliberately coarse BECAUSE the risk axes are swept:
# resolving `trend_period` finely on top of a 3 x 3 x 2 risk grid would be
# several hundred fits to one sample, which is a machine for manufacturing an
# in-sample Sharpe rather than a search.
#
# `tp_atr_mult: None` is a real point in the space, not a placeholder — it is
# the pure let-the-stop-decide configuration, kept in the grid so "the target
# earns its place" is something the sweep answers rather than something assumed.
PARAM_GRID = {
    "trend_period": [100, 200],
    "pullback_period": [10, 20, 34],
    "sl_atr_mult": [1.5, 2.0, 2.5],
    "tp_atr_mult": [2.5, 5.0, None],
    "trailing": [True, False],
}

ATR_PERIOD = 14

# Eastern wall-clock, converted through the zone so both sides of a DST change
# are right. Entry SIGNALS are taken on bars starting inside this window; the
# engine fills the next bar's open, so the latest possible fill is the bar
# starting at 15:30 ET.
ENTRY_OPEN_ET = (9, 30)
ENTRY_CLOSE_ET = (15, 30)
SESSION_FLAT_ET = (16, 0)

# Plain-English description for the tear sheet's strategy card, written for a
# reader deciding whether to trade this — not for whoever maintains the module.
# `{param}` slots are filled with the run's own bound parameters, so the card
# states the periods that actually ran rather than the defaults written here.
# The report never infers any of this from the signal arrays: a description
# guessed from the trades would be a guess printed as a fact.
LOGIC = {
    "concept": "Trend-following on the pullback rather than the breakout. A "
               "long-horizon moving average says a persistent buyer is "
               "present; a dip to the short-horizon average flushes the "
               "leveraged latecomers who bought the extension. When price "
               "closes back above that short average on the same bar the "
               "flush is over and the persistent buyer is still there — so "
               "buy the recovery, not the extension.",
    "entry": "Go Long when all three hold on the same bar: the close is above "
             "the Trend EMA ({trend_period}); the bar's low reaches down to "
             "or through the Pullback EMA ({pullback_period}); and the close "
             "recovers back above that Pullback EMA. Entry signals are only "
             "taken between 09:30 and 15:30 New York time; the fill is the "
             "next bar's open.",
    # Written so that every bound value reads correctly, including
    # `tp_atr_mult=None` and either setting of `trailing`. The card cannot
    # branch — `_describe_strategy` only substitutes `{param}` slots — so the
    # sentence states both arms and names the setting that chose between them.
    "exit": "Exit at whichever comes first. (1) The close falls back below "
            "the Trend EMA ({trend_period}) — the regime the trade was "
            "premised on is gone. (2) A stop {sl_atr_mult} x ATR 14 away — "
            "with trailing={trailing}, True means it trails the highest price "
            "reached since the fill and never widens, False means it sits "
            "fixed that far below the fill price. (3) A take-profit "
            "{tp_atr_mult} x ATR 14 above the fill price, where None means NO "
            "take-profit is modelled at all. (4) The last bar before 16:00 "
            "New York time. Every exit fills at the NEXT bar's open, so "
            "neither the stop nor the target is a fill at its own price — "
            "both carry a one-bar lag, and on a bar that breaches both the "
            "modelled fill is neither level.",
}


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def _wilder(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing — the average ATR is actually defined on.

    `alpha = 1/period` with `adjust=False` is Wilder's recursion exactly. A
    span-`period` EMA is a different (roughly twice as fast) average, and using
    it would produce an "ATR(14)" that no other tool agrees with.
    """
    return series.ewm(alpha=1.0 / period, adjust=False,
                      min_periods=period).mean()


def _atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """True range, Wilder-smoothed. NaN until `period` bars exist."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev = close.shift(1)
    # `.shift(1)` looks one bar BACKWARD, which is the correct direction: the
    # true range at bar i uses bar i-1's close and nothing later.
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    return _wilder(tr, period)


def _ema(close: pd.Series, period: int) -> pd.Series:
    """
    Span-`period` EMA, NaN through warm-up.

    `min_periods=period` is the point: without it pandas seeds the average from
    the first bar, so a "200-period EMA" exists at bar 2 and every regime call
    at the start of a symbol's history is decided by the seeding rather than by
    price.
    """
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def _session_masks(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    `(entry_window, session_flat_bar)` in New York wall-clock time.

    Converted through the zone rather than by a fixed UTC offset, so 09:30 ET
    is 09:30 ET in both January and July. A fixed offset is an hour wrong for
    roughly half the year, which silently moves the entry window across the
    open on every bar in that half.

    `session_flat_bar` is the LAST bar that starts before 16:00 ET on its own
    Eastern date, found by comparing each bar to the next rather than by
    matching a wall-clock string. That makes it timeframe-agnostic — it is the
    15:45 bar on 15m and the 15:55 bar on 5m — and it survives a missing bar at
    the close, where a string match would find nothing and never flatten.
    """
    et = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(
        "America/New_York")
    minutes = et.hour * 60 + et.minute

    open_m = ENTRY_OPEN_ET[0] * 60 + ENTRY_OPEN_ET[1]
    close_m = ENTRY_CLOSE_ET[0] * 60 + ENTRY_CLOSE_ET[1]
    flat_m = SESSION_FLAT_ET[0] * 60 + SESSION_FLAT_ET[1]

    entry_window = (minutes >= open_m) & (minutes <= close_m)

    before_flat = minutes < flat_m
    day = et.normalize()
    # The last bar before the flatten time is one that is before it while the
    # next bar either is not, or belongs to a different Eastern day.
    nxt_before = np.r_[before_flat[1:], False]
    same_day = np.r_[day[1:] == day[:-1], False]
    session_flat = before_flat & ~(nxt_before & same_day)

    return np.asarray(entry_window), np.asarray(session_flat)


# --------------------------------------------------------------------------
# The position walk
# --------------------------------------------------------------------------
def _walk_loop(entry_ok: np.ndarray,
               sig_exit: np.ndarray,
               open_: np.ndarray,
               high: np.ndarray,
               low: np.ndarray,
               atr: np.ndarray,
               flat_bar: np.ndarray,
               sl_mult: float,
               tp_mult: float,
               trailing: bool) -> tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    """
    Two-state machine over the bars: flat, or long under a stop and a target.

    Character-for-character the same machine as `ema_crossover._walk_loop`,
    with `sig_exit` in place of `cross_down` — there it is the downward
    crossover, here it is the loss of the regime. Both are "the reason the
    trade existed has gone away". `tests/test_risk_params.py` runs the two
    functions on identical arrays and requires identical output, so the copies
    cannot drift apart silently.

    A trailing stop cannot be a stateless mask. Its level is the highest price
    since ENTRY minus a fixed distance, so bar i's exit condition depends on
    which earlier bar opened the position — which depends on every entry before
    it. The fixed stop and the take-profit are anchored on the FILL PRICE,
    which is `open` on the bar after the signal, so they are path-dependent for
    the same reason.

    The timeline matches the engine's. An entry signal on bar i is filled at
    bar i+1's open, so the position is live from bar i+1, the fill price is
    `open_[i + 1]`, and the high-water mark starts there — NOT on the signal
    bar. Both distances are frozen at `mult * ATR` as measured on the SIGNAL
    bar and never re-measured as volatility changes.

    `tp_mult` is NaN when no take-profit is modelled, rather than a sentinel
    like 0 or a huge number. NaN propagates into `target` and every comparison
    against it is False, so the target simply never fires and the drawn line is
    a gap — a take-profit at 10,000 x ATR would be a line a reader could see
    and a level the search could still, in principle, reach.

    `trailing` selects the stop anchor and nothing else:
        True   level = (highest high since the fill) - dist,  ratcheting
        False  level = (fill price) - dist,                   constant

    Exits are checked from the fill bar onward, never on the signal bar itself.

    Returns `(entries, exits, stop_level, tp_level)`. Both levels are live for
    every bar the position is open and NaN everywhere else — they are what the
    tear sheet draws, so the lines a reader sees breached are the arrays the
    exits were taken from rather than a second reconstruction of them.
    """
    n = entry_ok.shape[0]
    entries = np.zeros(n, dtype=np.bool_)
    exits = np.zeros(n, dtype=np.bool_)
    stop_level = np.full(n, np.nan)
    tp_level = np.full(n, np.nan)

    pos = False
    fill_i = 0
    stop_dist = 0.0
    tp_dist = np.nan
    entry_px = np.nan
    hw = 0.0

    for i in range(n):
        if not pos:
            if entry_ok[i]:
                entries[i] = True
                pos = True
                fill_i = i + 1
                stop_dist = sl_mult * atr[i]
                # NaN in, NaN out: no take-profit stays no take-profit.
                tp_dist = tp_mult * atr[i]
                entry_px = np.nan
                hw = -np.inf
            continue

        if i < fill_i:
            continue

        if i == fill_i:
            # The engine's own fill: the open of the bar AFTER the signal. Not
            # the signal bar's close, which the position never traded at.
            entry_px = open_[i]

        if trailing:
            if high[i] > hw:
                hw = high[i]
            level = hw - stop_dist
        else:
            level = entry_px - stop_dist

        target = entry_px + tp_dist          # NaN when no target is modelled
        stop_level[i] = level
        tp_level[i] = target

        hit_stop = low[i] <= level
        # False whenever `target` is NaN, which is how "no take-profit" is
        # expressed. Every comparison against NaN is False in both numba and
        # numpy, so the interpreted fallback cannot disagree with the compiled
        # loop about it.
        hit_tp = high[i] >= target

        # A bar that breaches both produces one exit on this bar either way,
        # and the engine fills it at the next bar's open regardless — so the
        # intrabar race between the stop and the target is not resolved here
        # because it cannot change the result. On those bars the modelled fill
        # is neither the stop price nor the target price.
        if hit_stop or hit_tp or sig_exit[i] or flat_bar[i]:
            exits[i] = True
            pos = False

    return entries, exits, stop_level, tp_level


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
# Signals
# --------------------------------------------------------------------------
def _validate(trend_period: int, pullback_period: int, sl_atr_mult: float,
              tp_atr_mult: float | None, trailing: bool) -> None:
    """
    Reject parameter sets that do not describe this strategy.

    Every rejection here is recorded by `backtest.scan` as REJECTED and counted
    in the grid, so a combination the strategy refuses shrinks neither the
    reported search nor the honesty of `variants_tested`.
    """
    if trend_period < 2 or pullback_period < 2:
        raise ValueError(
            f"periods must be >= 2; got trend={trend_period}, "
            f"pullback={pullback_period}")
    if pullback_period >= trend_period:
        # Not a stylistic objection. With the two periods equal or inverted the
        # regime filter and the pullback reference are the same line, so "close
        # above the trend EMA" and "close above the pullback EMA" become the
        # same condition and the setup degenerates into a single-EMA touch.
        # That is a different strategy, so it fails here rather than producing
        # a plausible curve under this module's name.
        raise ValueError(
            f"pullback_period must be < trend_period; got {pullback_period} "
            f">= {trend_period}")
    if sl_atr_mult is None or sl_atr_mult <= 0:
        # The stop is not optional. A position with no stop and no target exits
        # only on the regime loss or the bell, which is a different strategy.
        raise ValueError(f"sl_atr_mult must be > 0; got {sl_atr_mult!r}")
    if tp_atr_mult is not None and tp_atr_mult <= 0:
        # None is the no-take-profit configuration and is accepted. A
        # non-positive number is not: a target at or below the fill price would
        # be breached by the fill bar itself.
        raise ValueError(
            f"tp_atr_mult must be > 0, or None for no take-profit; got "
            f"{tp_atr_mult!r}")
    if not isinstance(trailing, (bool, np.bool_)):
        # Truthiness would silently accept "false" (a non-empty string, so
        # True) and 0.0, and the stop would trail or not trail for reasons
        # invisible in the leaderboard's params column.
        raise ValueError(f"trailing must be a bool; got {trailing!r}")


def _series(bars: pd.DataFrame, trend_period: int,
            pullback_period: int) -> dict:
    """The shared calculation behind both `signal_fn` and `indicators`."""
    close = bars["close"]
    return {
        "trend": _ema(close, trend_period),
        "pullback": _ema(close, pullback_period),
        "atr": _atr(bars, ATR_PERIOD),
    }


def _tp_distance_mult(tp_atr_mult: float | None) -> float:
    """
    `tp_atr_mult` as the float the compiled walk takes: NaN means no target.

    numba specialises on argument types, so passing `None` on some calls and a
    float on others would compile two versions of `_walk` and make the "no
    take-profit" path a differently-typed one. NaN keeps a single signature and
    disables the target arithmetically — every comparison against it is False.
    """
    return np.nan if tp_atr_mult is None else float(tp_atr_mult)


def _signal_arrays(bars: pd.DataFrame, trend_period: int, pullback_period: int,
                   sl_atr_mult: float, tp_atr_mult: float | None,
                   trailing: bool) -> tuple:
    """
    Everything the walk produces, from one place.

    `signal_fn` and `indicators` both go through here, so the stop line the
    tear sheet draws is the array the exit was taken from and the pullback
    touch a reader sees is the array the entry came from. Two call sites
    computing this separately would be free to drift apart with nothing raising.
    """
    s = _series(bars, trend_period, pullback_period)
    entry_window, flat_bar = _session_masks(bars["ts"])

    close = bars["close"]
    low = bars["low"]

    # Warm-up is NaN in both EMAs and in ATR, and every comparison against NaN
    # is False, so no signal can fire before all three exist. The explicit
    # notna guard makes that a stated requirement rather than a property of NaN
    # comparison that a later edit could quietly lose — and it matters more on
    # the EXIT than on the entry, because `close < trend` is the negation of a
    # comparison and a bare `~(close > trend)` would be True through warm-up.
    ready = s["trend"].notna() & s["pullback"].notna() & s["atr"].notna()

    in_regime = (close > s["trend"]) & ready
    touched = low <= s["pullback"]          # the pullback reached the average
    recovered = close > s["pullback"]       # and closed back above it

    entry_ok = (in_regime & touched & recovered).to_numpy(dtype=bool)
    entry_ok = entry_ok & np.asarray(entry_window)

    # The regime-loss exit. Stated as `close < trend` and guarded by `ready`
    # rather than written as `~in_regime`, so a warm-up NaN cannot read as a
    # lost regime. A position cannot be open during warm-up anyway; the guard
    # is here so that stays true after the next edit rather than by luck.
    sig_exit = ((close < s["trend"]) & ready).to_numpy(dtype=bool)

    entries, exits, stop, target = _walk(
        entry_ok,
        sig_exit,
        bars["open"].to_numpy(dtype=float),
        bars["high"].to_numpy(dtype=float),
        bars["low"].to_numpy(dtype=float),
        # ATR is non-NaN wherever an entry can fire (`ready` guarantees it);
        # 0.0 elsewhere makes an impossible entry's stop distance zero rather
        # than NaN, which would propagate into a level nothing ever breaches.
        np.nan_to_num(s["atr"].to_numpy(dtype=float), nan=0.0),
        flat_bar,
        float(sl_atr_mult),
        _tp_distance_mult(tp_atr_mult),
        bool(trailing),
    )
    return s, entries, exits, stop, target


def signal_fn(bars: pd.DataFrame,
              trend_period: int = 200,
              pullback_period: int = 20,
              sl_atr_mult: float = 2.0,
              tp_atr_mult: float | None = 3.0,
              trailing: bool = True) -> tuple[pd.Series, pd.Series]:
    """
    Buy the pullback inside the up-trend; exit on the stop, the target, the
    lost regime, or the bell.

    The entry is a bar-level EVENT — a touch and a recovery resolved within one
    bar — rather than a state, so it does not need an edge detector the way a
    crossover state would. The walk enters only when flat, so a run of
    consecutive touch-and-recover bars still opens one position.

    Every comparison at bar i uses only bars <= i. The engine then fills at bar
    i+1's open, so nothing here can see a price it would not have had.
    """
    _validate(trend_period, pullback_period, sl_atr_mult, tp_atr_mult, trailing)

    _s, entries, exits, _stop, _target = _signal_arrays(
        bars, trend_period, pullback_period, sl_atr_mult, tp_atr_mult, trailing)

    return (pd.Series(entries, index=bars.index),
            pd.Series(exits, index=bars.index))


def indicators(bars: pd.DataFrame,
               trend_period: int = 200,
               pullback_period: int = 20,
               sl_atr_mult: float = 2.0,
               tp_atr_mult: float | None = 3.0,
               trailing: bool = True) -> dict[str, pd.Series]:
    """
    The price-scale series, for the tear sheet to draw over the candles.

    Computed HERE, the same way and from the same columns `signal_fn` reads, so
    the touch a reader sees is the array the entry was taken from. A second
    implementation living in the report would be free to disagree with this one
    — a chart showing a pullback one bar away from where the trade fired, with
    nothing raising.

    The stop and the target come from the same `_walk` the signals come from,
    through the same `_signal_arrays`, so they cannot disagree either. Both are
    NaN while flat and the report renders that as a gap, which is the honest
    drawing: there is no stop level when there is no position.

    The take-profit line is OMITTED ENTIRELY when `tp_atr_mult` is None. An
    all-NaN series would render as an empty legend entry, which reads as a
    target that exists and never got close — the opposite of the truth.

    ATR itself is deliberately NOT returned. The inspector draws these on the
    price axis, and a volatility series measured in points sits along the
    bottom of a 20,000-point contract telling a reader nothing. It reaches the
    reader through the stop and target lines, which is where it matters.

    Warm-up stays NaN rather than drawing the averages flat through the first
    bars.
    """
    _validate(trend_period, pullback_period, sl_atr_mult, tp_atr_mult, trailing)

    s, _entries, _exits, stop, target = _signal_arrays(
        bars, trend_period, pullback_period, sl_atr_mult, tp_atr_mult, trailing)

    kind = "Trailing" if trailing else "Fixed"
    out = {
        f"Trend EMA ({trend_period})": s["trend"],
        f"Pullback EMA ({pullback_period})": s["pullback"],
        f"{kind} Stop ({sl_atr_mult}xATR{ATR_PERIOD})":
            pd.Series(stop, index=bars.index),
    }
    if tp_atr_mult is not None:
        out[f"Take Profit ({tp_atr_mult}xATR{ATR_PERIOD})"] = pd.Series(
            target, index=bars.index)
    return out


def make_signal_fn(trend_period: int = 200,
                   pullback_period: int = 20,
                   sl_atr_mult: float = 2.0,
                   tp_atr_mult: float | None = 3.0,
                   trailing: bool = True):
    """
    Bind parameters for `agents.tier3_workers.load_strategy`.

    The loader prefers this factory when params are supplied; the engine itself
    only ever calls the bound `signal_fn(bars)`. Validation runs HERE as well
    as inside `signal_fn`, so `--scan` records a bad combination as REJECTED at
    bind time rather than discovering it one symbol into the sweep.
    """
    _validate(trend_period, pullback_period, sl_atr_mult, tp_atr_mult, trailing)

    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return signal_fn(bars, trend_period=trend_period,
                         pullback_period=pullback_period,
                         sl_atr_mult=sl_atr_mult,
                         tp_atr_mult=tp_atr_mult,
                         trailing=trailing)

    return _bound
