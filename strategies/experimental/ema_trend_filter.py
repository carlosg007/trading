"""
EMA trend filter — trade the fast/slow crossover only in the anchor trend's
direction, and only when volatility is expanding.

Location:  ~/src/trading/strategies/experimental/ema_trend_filter.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding — do not edit this block to match it.

    1. STRATEGY METADATA
       Strategy Name:      ema_trend_filter
       Strategy Archetype: Trend-Following (filtered crossover)
       Primary Timeframe:  15m
       Target Assets:      NQ, ES, CL, GC

    2. CORE CONCEPT & HYPOTHESIS
       See "Hypothesis" below. Carried into LOGIC["concept"] near-verbatim.

    3. INDICATORS & PARAMETER GRID
       Indicators:         EMA(fast_period), EMA(slow_period), the anchor
                           EMA(trend_period), ATR(14), and SMA(ATR(14), 20).
       Default Parameters: fast_period=9, slow_period=21, trend_period=800,
                           sl_atr_mult=2.0, tp_atr_mult=3.0, trailing=True
       PARAM_GRID:         see below — 162 combinations. The target axis
                           carries no `None` point, which is a departure from
                           the convention the other risk-swept modules follow;
                           the reasoning and what it costs are recorded at
                           PARAM_GRID.

    4. ENTRY & EXIT EXECUTION RULES
       Long Entry:   close > EMA(trend_period)            (anchor regime)
                 AND EMA(fast) crosses ABOVE EMA(slow)     (the trigger)
                 AND ATR(14) > SMA(ATR(14), 20)            (expanding vol)
                 AND the bar starts between 09:30 and 15:30 America/New_York.
       Short Entry:  LONG ONLY. See "The short side is specified out" below.
       Take Profit:  fill price + tp_atr_mult x ATR(14), or None.
       Stop Loss:    sl_atr_mult x ATR(14), trailing or fixed.
       Session Rules: entry signals 09:30-15:30 America/New_York; the exit
                      signal fires on the last bar that STARTS before 16:00
                      America/New_York — the 15:45 bar on 15m — so the fill
                      lands on the 16:00 open and nothing is held past the
                      bell.
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
A moving-average crossover is a cheap detector of a change in short-horizon
order flow, and on its own it is close to worthless: most crossings are noise
around a flat mean, and the strategy pays a full round turn for each of them.
Two conditions separate the crossings worth paying for from the rest.

The first is direction. A crossover that agrees with the day-scale trend is a
continuation of an imbalance that already exists — someone is still working an
order — while one that fights it is a countertrend guess. The anchor EMA is
what decides which of the two a given crossing is.

The second is participation. A crossover on contracting volatility is the
market rotating inside a range with nobody pressing; the same crossing on
expanding volatility means real size arrived. ATR against its own 20-period
average is the cheapest available proxy for that, and it is the filter that
removes the mid-session chop where a crossover system bleeds.

The edge, if it exists, is therefore a selection effect rather than a
prediction: the same trigger, taken only when the trend and the volume of
participation both agree with it. It is fragile in the obvious way — the trend
and the volatility filter both lag, so the setup is at its most convincing
exactly at a blow-off top. The stop is what pays for those, which is why it is
a swept parameter here rather than an afterthought; see the warning about
sweeping it under "Read this before reading the equity curve".

The anchor EMA is a proxy, and it is not the 1H 200 EMA
-------------------------------------------------------
The specification calls for the 200-period trend on 1-hour bars. This module
computes EMA(800) on the 15-minute frame instead, DELIBERATELY, and the two are
not the same series.

They cover the same wall-clock lookback — 800 x 15m == 200 x 1H == a little
over eight trading days — but an EMA is a weighted recursion, not a window, and
the 15m version updates four times as often with a smoothing constant four
times smaller. Their values differ, most visibly right after a sharp move,
where the 15m proxy turns sooner.

It is the proxy and not the real thing because the alternative is worse. The
engine hands `signal_fn` ONE frame at ONE timeframe; building a genuine 1H
series here would mean resampling inside the strategy, and a resample carries
its own lookahead trap — a 1H bar stamped at 09:00 is not complete until 10:00,
so joining it back onto the 15m frame without shifting it forward hands the
09:15 bar a close from its own future. That bug is invisible in the equity
curve and it inflates every metric. A single-frame proxy cannot have it: every
value in EMA(800) at bar i is built from closes at or before bar i.

So read `trend_period` as "the anchor trend, expressed in 15m bars", not as a
1H 200 EMA. If the 1H series is genuinely wanted rather than approximated, it
belongs in the reader as a second timeframe with an explicit one-bar shift, not
in here.

The short side is specified out
-------------------------------
The request this module was written from described a symmetric long/short
system. It is LONG ONLY, and that is a limit of the engine rather than a view
about the market.

`backtest/engine.py` drives `vbt.Portfolio.from_signals` with
`direction="longonly"` hardcoded, and `_assemble_result` stamps every trade row
`direction="long"`. The strategy contract is two boolean masks, `(entries,
exits)`, with no channel that could mark one as a short. A short setup emitted
into `entries` is BOUGHT, so its P&L arrives with the sign inverted and nothing
raises — a plausible equity curve that is exactly backwards.

The mirrored rules are stated in the specification block above so the intent
survives; they are not implemented, and no part of this module computes them.
Adding shorts is a change to the engine's signal contract, not to this file.

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
Seven things about this module are NOT what a reader would assume from the
rules above. None of them flatters the result, but a drawdown, a win rate or a
trade count read without knowing them is being read wrong.

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

4. THERE IS NO EXIT ON THE TREND OR THE CROSSOVER REVERSING. This is the one
   that most often surprises a reader, because the entry has three conditions
   and none of them is also an exit. A position is held through the close
   falling back below the anchor EMA, through EMA(fast) crossing back below
   EMA(slow), and through volatility contracting again. It leaves on the stop,
   the target, or the bell, and on nothing else.

   That is a deliberate reading of the specification, which enumerates the
   exits and lists none of those three. It is defensible on its own terms —
   the session flatten bounds every position to a single day, so a trade
   cannot quietly become a long-term hold — but it means the stop is doing ALL
   of the work of getting out of a losing trade. Read the drawdown with that
   in mind, and read a change in `sl_atr_mult` as a change to the only
   discretionary exit the strategy has.

   `_walk` keeps its `sig_exit` argument regardless, and this module passes an
   all-False array into it. The slot stays so the kernel remains identical to
   `ema_crossover`'s, which `tests/test_risk_params.py` pins; wiring a
   crossunder into it is a one-line change if the specification ever gains
   that exit.

5. THE EXITS ARE COMPUTED IN THIS MODULE, NOT BY THE ENGINE. A trailing stop is
   path-dependent, so it cannot be a stateless boolean mask over bars, and once
   the walk exists the fixed stop and the target belong in it too rather than
   being split across layers that could disagree. `_walk` below resolves
   entries, the stop, the target and the session flatten together. That is why
   this module contains session logic at all, which strategies otherwise must
   not.

   `_walk` here is an independent COPY of the one in `ema_crossover.py`, not an
   import. Strategy modules are loaded from a file path and are deliberately
   self-contained — the same reason `_wilder`, `_atr` and `_session_masks` are
   duplicated across every module in this directory. The copies are pinned
   against each other in `tests/test_risk_params.py`, which runs both walks on
   the same arrays and requires identical output; a divergence fails there
   rather than showing up as two strategies that disagree about what a stop is.

6. THE RISK PARAMETERS ARE SWEPT, WHICH IS A KNOWN WAY TO FIT NOISE. `--scan`
   varies `sl_atr_mult`, `tp_atr_mult` and `trailing` alongside the two
   crossover periods — 162 cells, of which 18 are risk settings for each pair
   of periods — so the search can find the risk settings that survived this
   sample's worst few days. Given point 4 — the stop is the only discretionary
   exit — this module is MORE exposed to that than a strategy with a signal
   exit, because the risk parameters are not refining an exit rule, they ARE
   the exit rule. If the winning cell's edge lives mostly in `sl_atr_mult`
   rather than in the crossover, read that as evidence against the strategy
   rather than for it, and check the neighbouring cells in `scan_<SYMBOL>.csv`
   before believing any of it.

7. `--flat-by-close` IS NOT NEEDED AND SHOULD NOT BE PASSED. This module emits
   its own session exit on the last bar starting before 16:00 ET, converted
   through America/New_York so it is right on both sides of a DST change. The
   engine's flag uses a FIXED UTC time, which is 16:00 ET in winter and 15:00
   ET in summer. Passing it is harmless — `clean_signals` keeps whichever exit
   comes first, and this one always does — but it is redundant and its summer
   boundary is an hour off.

   The same DST point applies to reading the specification: 16:00 ET is 21:00
   UTC under EST and 20:00 UTC under EDT. No fixed UTC hour is the close, which
   is why the conversion goes through the named zone.

Every calculation is causal. Values at bar i use bars <= i only, warm-up stays
NaN and is never treated as a signal, and the engine then fills at bar i+1's
open — so nothing here can see a price it would not have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME = "15m"
SYMBOLS = ["NQ", "ES", "CL", "GC"]
DEFAULT_PARAMS = {"fast_period": 9, "slow_period": 21, "trend_period": 800,
                  "sl_atr_mult": 2.0, "tp_atr_mult": 3.0, "trailing": True}

# The search space `backtest/run.py --scan` sweeps, declared here because this
# module is the only place that knows what these parameters mean and what the
# signature will accept. 3 x 3 x 1 x 3 x 3 x 2 = 162 combinations, every one of
# them valid (every fast value is below every slow value, and every slow value
# is below 800), so the scan reports 162 evaluated rather than 162 attempted
# and some rejected.
#
# 162 is inside, but near, the bound at which a search can still be reported
# honestly. Each combination is a fit to the same in-sample bars, and the best
# of 162 is a higher number than the best of 9 even when nothing in the market
# has changed — `variants_tested` is carried onto every report and every
# leaderboard row for exactly that reason. `tests/test_risk_params.py` caps
# this at 200 cells.
#
# THIS GRID SEARCHES NO "NO TAKE-PROFIT" POINT, AND THAT IS A REAL GAP.
# `ema_crossover` puts `None` in its target axis so that "does the take-profit
# earn its place at all?" is a question the sweep ANSWERS. This grid was
# specified without it, deliberately and after the point was raised, so the
# question is not asked here: every one of the 162 cells exits on a target, and
# the winner is the best target rather than evidence that having one beats
# having none.
#
# What that costs, concretely. Point 4 of the module docstring notes there is
# no signal exit, so the target and the stop are the ENTIRE discretionary exit
# rule. Without a `None` cell the sweep cannot distinguish "the 2.0 x ATR
# target is the edge" from "any target is worse than letting the stop and the
# bell decide, and 2.0 is merely the least bad". If the scan's top cells all
# cluster at the widest target, that is the shape a missing `None` point would
# leave and it should be read as a reason to run one, not as a result.
#
# Adding `None` as a fourth target value is 216 cells, past the 200-cell bound.
# The cheap way to ask the question without growing the grid is a single
# out-of-band run at the winning cell with `--param tp_atr_mult=None`, compared
# against the swept winner. `tests/test_risk_params.py` records the exemption
# per module rather than dropping the check, so this stays a decision on the
# record instead of a convention that quietly eroded.
#
# `trend_period` is a ONE-VALUE axis: it is pinned at 800 rather than searched.
# The anchor length is the strategy's premise, and a sweep that moves it is
# searching over which trend to believe in — a different claim from "this
# trigger works inside this trend". Declaring it here rather than omitting it
# makes the pinned value explicit in the scan output and in the leaderboard's
# `params` column, at no cost to the cell count.
PARAM_GRID = {
    "fast_period": [5, 9, 13],
    "slow_period": [21, 34, 50],
    "trend_period": [800],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, 4.0],
    "trailing": [True, False],
}

ATR_PERIOD = 14

# The volatility gate: ATR(14) against its own 20-period simple average. Fixed
# constants rather than parameters, deliberately — they are the specification's
# definition of "volatility is expanding", and exposing them to `--scan` would
# add two more axes to a grid whose risk parameters are already the part most
# able to fit noise.
ATR_MA_PERIOD = 20

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
    "concept": "Trend-following on a filtered moving-average crossover. The "
               "crossover on its own is mostly noise, so two conditions "
               "decide which crossings are worth paying a round turn for: the "
               "long-horizon anchor average says which way the day-scale "
               "imbalance runs, and ATR above its own 20-period average says "
               "real size is participating rather than the market rotating "
               "inside a range. Take the trigger only when both agree with "
               "it.",
    "entry": "Go Long when all four hold on the same bar: the close is above "
             "the Trend EMA ({trend_period}); the Fast EMA ({fast_period}) "
             "crosses up through the Slow EMA ({slow_period}) on this bar; "
             "ATR 14 is above its own 20-period average, so volatility is "
             "expanding; and the bar starts between 09:30 and 15:30 New York "
             "time. Long only — no short is taken. The fill is the next "
             "bar's open.",
    # Written so that every bound value reads correctly, including
    # `tp_atr_mult=None` and either setting of `trailing`. The card cannot
    # branch — `_describe_strategy` only substitutes `{param}` slots — so the
    # sentence states both arms and names the setting that chose between them.
    "exit": "Exit at whichever comes first. (1) A stop {sl_atr_mult} x ATR 14 "
            "away — with trailing={trailing}, True means it trails the "
            "highest price reached since the fill and never widens, False "
            "means it sits fixed that far below the fill price. (2) A "
            "take-profit {tp_atr_mult} x ATR 14 above the fill price, where "
            "None means NO take-profit is modelled at all. (3) The last bar "
            "starting before 16:00 New York time, so the fill lands on the "
            "16:00 open and no position is carried past the bell. There is NO "
            "exit on the trend or the crossover reversing: the position is "
            "held through both, and the stop is the only discretionary way "
            "out of a losing trade. Every exit fills at the NEXT bar's open, "
            "so neither the stop nor the target is a fill at its own price — "
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
    the first bar, so an "800-period EMA" exists at bar 2 and every regime call
    at the start of a symbol's history is decided by the seeding rather than by
    price.
    """
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple moving average, NaN until `period` values exist.

    `min_periods=period` for the same reason as `_ema`: a partial average over
    three ATR readings is not the 20-period average the volatility gate is
    defined against, and letting it exist would open the gate during warm-up on
    whatever the first few bars happened to do.
    """
    return series.rolling(period, min_periods=period).mean()


def _session_masks(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    `(entry_window, session_flat_bar)` in New York wall-clock time.

    Converted through the zone rather than by a fixed UTC offset, so 09:30 ET
    is 09:30 ET in both January and July. A fixed offset is an hour wrong for
    roughly half the year, which silently moves the entry window across the
    open on every bar in that half.

    `session_flat_bar` is the LAST bar that starts before 16:00 ET on its own
    Eastern date, found by comparing each bar to the next rather than by
    matching a wall-clock string. On 15m that is the 15:45 bar, so the engine's
    next-bar fill lands on the 16:00 open and the position is flat AT the bell
    rather than one bar after it. Comparing neighbours rather than matching a
    string makes it timeframe-agnostic — the 15:55 bar on 5m — and it survives
    a missing bar at the close, where a string match would find nothing and
    never flatten at all.
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

    Character-for-character the same machine as `ema_crossover._walk_loop`.
    `tests/test_risk_params.py` runs the two functions on identical arrays and
    requires identical output, so the copies cannot drift apart silently — do
    not "improve" this one alone.

    `sig_exit` is the signal-driven exit: in `ema_crossover` it is the downward
    crossover. THIS module passes an all-False array, because its specification
    enumerates the exits and lists no signal exit at all — see point 4 of the
    module docstring. The argument stays in the signature so the kernel remains
    the shared one rather than a fork of it.

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
def _validate(fast_period: int, slow_period: int, trend_period: int,
              sl_atr_mult: float, tp_atr_mult: float | None,
              trailing: bool) -> None:
    """
    Reject parameter sets that do not describe this strategy.

    Every rejection here is recorded by `backtest.scan` as REJECTED and counted
    in the grid, so a combination the strategy refuses shrinks neither the
    reported search nor the honesty of `variants_tested`.
    """
    if fast_period < 2 or slow_period < 2 or trend_period < 2:
        raise ValueError(
            f"periods must be >= 2; got fast={fast_period}, "
            f"slow={slow_period}, trend={trend_period}")
    if fast_period >= slow_period:
        # Not a stylistic objection. With the two equal the crossover can never
        # fire, and inverted it fires on the opposite event — a downward cross
        # reported under this module's name as a long trigger. Either way the
        # curve would look plausible and describe a strategy nobody specified.
        raise ValueError(
            f"fast_period must be < slow_period; got {fast_period} >= "
            f"{slow_period}")
    if slow_period >= trend_period:
        # The anchor has to be a longer horizon than the trigger, or the
        # regime filter and the crossover are reading the same movement and
        # "the trend agrees with the cross" becomes close to tautological.
        raise ValueError(
            f"slow_period must be < trend_period; got {slow_period} >= "
            f"{trend_period}")
    if sl_atr_mult is None or sl_atr_mult <= 0:
        # The stop is not optional, and in this module it is load-bearing: with
        # no signal exit, a position with no stop and no target would leave
        # only at the bell.
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


def _series(bars: pd.DataFrame, fast_period: int, slow_period: int,
            trend_period: int) -> dict:
    """The shared calculation behind both `signal_fn` and `indicators`."""
    close = bars["close"]
    atr = _atr(bars, ATR_PERIOD)
    return {
        "fast": _ema(close, fast_period),
        "slow": _ema(close, slow_period),
        "trend": _ema(close, trend_period),
        "atr": atr,
        "atr_ma": _sma(atr, ATR_MA_PERIOD),
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


def _signal_arrays(bars: pd.DataFrame, fast_period: int, slow_period: int,
                   trend_period: int, sl_atr_mult: float,
                   tp_atr_mult: float | None, trailing: bool) -> tuple:
    """
    Everything the walk produces, from one place.

    `signal_fn` and `indicators` both go through here, so the stop line the
    tear sheet draws is the array the exit was taken from and the crossover a
    reader sees is the array the entry came from. Two call sites computing this
    separately would be free to drift apart with nothing raising.
    """
    s = _series(bars, fast_period, slow_period, trend_period)
    entry_window, flat_bar = _session_masks(bars["ts"])

    close = bars["close"]

    # Warm-up is NaN in all three EMAs, in ATR and in the ATR average, and
    # every comparison against NaN is False, so no signal can fire before all
    # five exist. The explicit notna guard makes that a stated requirement
    # rather than a property of NaN comparison that a later edit could quietly
    # lose.
    ready = (s["fast"].notna() & s["slow"].notna() & s["trend"].notna()
             & s["atr"].notna() & s["atr_ma"].notna())

    # The crossover as an EVENT, not a state. `above` is False through warm-up
    # (NaN > NaN is False), so requiring the PREVIOUS bar to be ready as well
    # is what stops the first fully-warm bar from registering as a cross: at
    # that bar `above` may flip from False to True purely because the averages
    # came into existence, which is a fact about the warm-up and not about
    # price. `fill_value=False` keeps bar 0 out for the same reason.
    above = (s["fast"] > s["slow"]) & ready
    cross_up = (above & ~above.shift(1, fill_value=False)
                & ready.shift(1, fill_value=False))

    in_regime = close > s["trend"]              # the anchor trend agrees
    expanding = s["atr"] > s["atr_ma"]          # real size is participating

    entry_ok = (in_regime & cross_up & expanding & ready).to_numpy(dtype=bool)
    entry_ok = entry_ok & np.asarray(entry_window)

    # No signal exit. The specification enumerates the exits and lists only the
    # stop, the target and the session flatten — the position is deliberately
    # held through the trend and the crossover reversing (docstring point 4).
    # The array exists because `_walk` keeps the shared kernel's signature; it
    # is all-False, so the walk's `sig_exit[i]` branch is never taken.
    sig_exit = np.zeros(len(bars), dtype=bool)

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
              fast_period: int = 9,
              slow_period: int = 21,
              trend_period: int = 800,
              sl_atr_mult: float = 2.0,
              tp_atr_mult: float | None = 3.0,
              trailing: bool = True) -> tuple[pd.Series, pd.Series]:
    """
    Take the crossover only with the anchor trend and expanding volatility;
    exit on the stop, the target, or the bell.

    Long only — see the module docstring for why the specification's short side
    is not implemented.

    The entry is a bar-level EVENT: the crossover fires on the bar where
    EMA(fast) first closes above EMA(slow), so a long run of bars with fast
    above slow produces one signal rather than a signal every bar. The walk
    enters only when flat, so a second crossover while a position is open is
    ignored rather than pyramided.

    Every comparison at bar i uses only bars <= i. The engine then fills at bar
    i+1's open, so nothing here can see a price it would not have had.
    """
    _validate(fast_period, slow_period, trend_period, sl_atr_mult, tp_atr_mult,
              trailing)

    _s, entries, exits, _stop, _target = _signal_arrays(
        bars, fast_period, slow_period, trend_period, sl_atr_mult, tp_atr_mult,
        trailing)

    return (pd.Series(entries, index=bars.index),
            pd.Series(exits, index=bars.index))


def indicators(bars: pd.DataFrame,
               fast_period: int = 9,
               slow_period: int = 21,
               trend_period: int = 800,
               sl_atr_mult: float = 2.0,
               tp_atr_mult: float | None = 3.0,
               trailing: bool = True) -> dict[str, pd.Series]:
    """
    The price-scale series, for the tear sheet to draw over the candles.

    Computed HERE, the same way and from the same columns `signal_fn` reads, so
    the crossover a reader sees is the array the entry was taken from. A second
    implementation living in the report would be free to disagree with this one
    — a chart showing a cross one bar away from where the trade fired, with
    nothing raising.

    The stop and the target come from the same `_walk` the signals come from,
    through the same `_signal_arrays`, so they cannot disagree either. Both are
    NaN while flat and the report renders that as a gap, which is the honest
    drawing: there is no stop level when there is no position.

    The take-profit line is OMITTED ENTIRELY when `tp_atr_mult` is None. An
    all-NaN series would render as an empty legend entry, which reads as a
    target that exists and never got close — the opposite of the truth.

    ATR AND ITS AVERAGE ARE DELIBERATELY NOT RETURNED, which means the
    volatility gate — one of the three entry conditions — is the one thing a
    reader cannot see on the chart. That is a real gap, and the alternative is
    worse: the inspector draws these on the PRICE axis, so a volatility series
    measured in points would sit flat along the bottom of a 20,000-point
    contract, unreadable, while rescaling it to fit would draw a line at prices
    nothing ever traded at. ATR reaches the reader through the stop and target
    lines, which is where it changes a decision.

    Warm-up stays NaN rather than drawing the averages flat through the first
    bars.
    """
    _validate(fast_period, slow_period, trend_period, sl_atr_mult, tp_atr_mult,
              trailing)

    s, _entries, _exits, stop, target = _signal_arrays(
        bars, fast_period, slow_period, trend_period, sl_atr_mult, tp_atr_mult,
        trailing)

    kind = "Trailing" if trailing else "Fixed"
    out = {
        f"Fast EMA ({fast_period})": s["fast"],
        f"Slow EMA ({slow_period})": s["slow"],
        f"Trend EMA ({trend_period})": s["trend"],
        f"{kind} Stop ({sl_atr_mult}xATR{ATR_PERIOD})":
            pd.Series(stop, index=bars.index),
    }
    if tp_atr_mult is not None:
        out[f"Take Profit ({tp_atr_mult}xATR{ATR_PERIOD})"] = pd.Series(
            target, index=bars.index)
    return out


def make_signal_fn(fast_period: int = 9,
                   slow_period: int = 21,
                   trend_period: int = 800,
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
    _validate(fast_period, slow_period, trend_period, sl_atr_mult, tp_atr_mult,
              trailing)

    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return signal_fn(bars, fast_period=fast_period,
                         slow_period=slow_period,
                         trend_period=trend_period,
                         sl_atr_mult=sl_atr_mult,
                         tp_atr_mult=tp_atr_mult,
                         trailing=trailing)

    return _bound
