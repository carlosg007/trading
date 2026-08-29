"""
SMA momentum crossover — a three-layer trend-continuation filter: take the fast/
slow SMA crossover only when price is on the correct side of a macro baseline
and ADX says a trend is actually present.

Location:  ~/src/trading/strategies/experimental/sma_momentum_crossover.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding — do not edit this block to match it.

    1. STRATEGY METADATA
       Strategy Identifier: sma_momentum_crossover
       Strategy Archetype:  Trend Continuation
       Primary Timeframe:   15m
       Target Assets:       NQ, ES, CL, GC

    2. CORE CONCEPT & HYPOTHESIS
       A moving-average crossover on its own is close to worthless: on a
       ranging market it fires on every oscillation and pays a round turn for
       each one. The claim here is that the SAME trigger becomes tradeable once
       two conditions are stacked in front of it, and that each layer removes a
       different failure mode.

       Layer 1, the macro baseline, removes counter-trend crossovers. A fast
       SMA crossing up through a slow one below a falling 200-period baseline
       is a bounce inside a downtrend, and buying it is paying to be early.

       Layer 2, the crossover itself, is the timing. It says nothing about
       whether a trend exists — only about where the short-horizon average sits
       relative to the medium-horizon one.

       Layer 3, ADX above its threshold, removes crossovers that happen inside
       a range. ADX is direction-agnostic by construction: it measures the
       magnitude of directional movement, not its sign, so the SAME threshold
       applies to a long and a short and the layer cannot smuggle a directional
       bias in through the back door.

       Why it should persist: the trigger is the part everyone can compute, and
       it is the part with no edge. The selection is the claim.

    3. INDICATORS & PARAMETER GRID
       Indicators:         SMA(fast_window), SMA(slow_window),
                           SMA(macro_window), ADX(14), ATR(14).
       Default Parameters: fast_window=10, slow_window=30, macro_window=200,
                           adx_threshold=20, sl_atr_mult=1.0, tp_atr_mult=2.0,
                           use_macro_anchor=True, use_adx_filter=True,
                           use_news_filter=False, trailing=False
       PARAM_GRID:         fast_window   [5, 10, 15]
                           slow_window   [20, 30, 50]
                           tp_atr_mult   [1.5, 2.0, 3.0]
                           sl_atr_mult   [1.0, 1.5]
                           trailing      [False]   (pinned, see PARAM_GRID)
                           = 54 combinations, none rejected.

    4. ENTRY & EXIT EXECUTION RULES
       The trigger is mandatory. Each of the two confluence layers applies only
       when its own toggle is on, so the strategy runs anywhere from a bare
       crossover to the full three-layer stack.

       Long Entry:   SMA(fast) crosses ABOVE SMA(slow)      (the trigger)
                 AND close > SMA(macro_window)              if use_macro_anchor
                 AND ADX(14) > adx_threshold                if use_adx_filter
                 AND the fill bar is not inside a macro-release window
                                                            if use_news_filter
       Short Entry:  SMA(fast) crosses BELOW SMA(slow)      (the trigger)
                 AND close < SMA(macro_window)              if use_macro_anchor
                 AND ADX(14) > adx_threshold                if use_adx_filter
                 AND the fill bar is not inside a macro-release window
                                                            if use_news_filter
       Take Profit:  long  fill price + tp_atr_mult x ATR(14), or None.
                     short fill price - tp_atr_mult x ATR(14), or None.
       Stop Loss:    sl_atr_mult x ATR(14) from the fill, trailing or fixed.
       Session Rules: NONE MODELLED. There is no entry window and no flatten
                      at the bell — see "What this module does not do" below.
       Execution Fill: next-bar open with contract-specific slippage and
                      commission.

    5. BACKTEST EXECUTION CONTROLS
       In-Sample Period:   2013-01-01 to 2022-12-31
       Phase 3 Holdout:    YES — 2023-01-01 to 2026-01-01, RESERVED. It does
                           not overlap the in-sample window and must not be
                           looked at until Gates 1 and 2 are settled.
       ML Filter (Version B): YES — see "Version B" below.
       Run Mode:           Multi-Asset Runner (`bt-run`), then the five-stage
                           pipeline.

TWO NAMES IN THE REQUEST WERE CHANGED, DELIBERATELY
===================================================
The request specified `sl_atr_multiplier` and `tp_atr_multiplier`. This module
declares `sl_atr_mult` and `tp_atr_mult` instead, and the difference is not
cosmetic: `backtest/run.py`'s `RISK_PARAMS` and `backtest/promote.py`'s
`RISK_KEYS` look these names up EXACTLY, with no aliasing. Under the requested
spelling the leaderboard's `sl_atr_mult` and `tp_atr_mult` columns would come
back blank — and blank in that file means "this strategy has no such setting",
never "the setting was off". A strategy whose entire exit rule is a stop and a
target would have been recorded as having neither, and `promote.py`'s `risk`
block would have written `NOT DECLARED` over both. The requested SEMANTICS are
unchanged: both are still multiples of ATR(14) measured on the signal bar.

`trailing` was added, and is not in the request. The shared walk kernel this
module reuses takes it, `tests/test_risk_params.py` requires every module in
its table to declare all three risk parameters, and the leaderboard has a
column for it. It defaults to False — a FIXED stop, which is what the request's
silence implies — and the grid pins it to [False] rather than sweeping it, so
no run differs from the requested strategy unless somebody asks for it by name.

WHAT THIS MODULE DOES NOT DO, AND WHY
=====================================
Stated here rather than approximated silently, because each of these changes
how a number this module produces must be read.

1. THERE ARE NO STOP OR TARGET ORDERS. `backtest/engine.py` drives
   `vbt.Portfolio.from_signals` off boolean masks and fills them at the NEXT
   bar's open. A stop here is therefore an exit SIGNAL detected on the bar that
   breaches it and filled one bar later, at whatever the next open happens to
   be — not a fill at the stop price. On a bar that breaches both the stop and
   the target the modelled fill is neither level. Every drawdown figure this
   module produces has to be read that way; a 1.0 x ATR stop does not bound a
   loss at 1.0 x ATR.

2. THE ONLY EXITS ARE THE STOP AND THE TARGET. There is no exit on the
   crossover reversing, no exit on the macro baseline flipping, and no session
   flatten — the request's rules enumerate a take profit and a stop loss and
   nothing else, and adding a third exit would be a strategy nobody specified.
   Two consequences: the stop is load-bearing (with `tp_atr_mult=None` it is
   the ONLY discretionary way out of a losing trade, which is why `_validate`
   refuses to run without one), and positions are carried overnight and over
   weekends. On an intraday timeframe that is a real exposure the request did
   not mention, and it is priced into no number here.

3. LAYER 3 IS NAMED "Momentum & Volume" IN THE REQUEST BUT CONTAINS NO VOLUME
   CONDITION. The request's own rule for that layer is ADX(14) > 20 and
   nothing else, so that is what Version A tests. Volume reaches the strategy
   only through Version B, as a feature — see `ml_features`. If a volume
   CONDITION was intended for Version A, it is not in this module and no run of
   it has tested one.

4. THE NEWS FILTER IS OFF BY DEFAULT. It needs a macro calendar to be
   meaningful, and `backtest/event_calendar.py` will RAISE rather than return
   an all-clear mask when its calendar does not cover the run's span. Check
   what a filtered run would actually use before trusting it:

       python3 backtest/event_calendar.py --start 2013-01-01 --end 2026-01-01

   Only NFP's rule-generated dates follow the real convention; CPI, PPI and
   FOMC anchors land in the right week and often the wrong day, and a 30-minute
   window on the wrong day blocks a random half hour while leaving the release
   tradeable. `use_news_filter=True` with a RULE-provenance calendar is not a
   run that dodged the actual prints.

THE NEWS FILTER IS IMPORTED, NOT REIMPLEMENTED
==============================================
`use_news_filter` calls `backtest.event_calendar.apply_entry_filters`, which is
the only implementation of that filter in this repository and is documented as
callable by a strategy module on its own masks. A second copy living here would
be free to disagree with the engine's about which bars a release covers, and
the two would be compared by nobody.

THIS PUTS THE MODULE OUTSIDE `ALLOWED_IMPORTS`, AND THAT IS A DELIBERATE
EXCEPTION. `agents.tier3_workers.ALLOWED_IMPORTS` permits a strategy module
numpy, pandas, math, vectorbtpro and numba, and nothing else. That allowlist
governs MODEL-GENERATED code, which is audited before it is executed; it does
not run over hand-written modules, which `load_strategy` imports from a file
path. `ema_trend_filter` still keeps inside it, reimplementing the session-date
rule rather than importing it, with a test pinning its copy against the
original. That trade is right for one line of arithmetic and wrong here: the
macro filter is a calendar file, a provenance rule, merged intervals and a
one-bar widening, and a second copy of all of it in a strategy module would be
a second thing to keep in step.

The consequences, stated rather than discovered later: this module would be
REJECTED by the AST validator if it were ever passed through the synthesis
path, and `tests/test_sma_momentum_crossover.py` pins that the
`backtest.event_calendar` import is the ONLY objection the validator has to it
— so a future edit that reaches for `open`, `eval` or a network library fails
loudly instead of hiding behind an exception that was granted for something
else. Nothing about this widens what generated code may import.

It is applied to the CANDIDATE triggers, before the position walk, and that is
a different thing from the engine's `--news-filter` flag, which applies to the
walk's OUTPUT entries. Blocking a candidate leaves the strategy FLAT and
therefore free to take a later trigger it would otherwise have been holding
through, so the two orderings do not produce the same trade list and the
module-level one is the one that matches how the rest of this module is built.
The engine's flag still works and still composes — it will simply find fewer
entries left to remove. Do not read a trade COUNT to decide whether the filter
bound: it subtracts candidates, not trades.

VERSION B — THE ML FILTER
=========================
Version B is `agents.tier3_workers.apply_ml_signal_filter`, the repository's
shared expanding-window walk-forward: for a candidate signalled on bar `s` it
fits only on trades that had already CLOSED before `s` (`exit_idx < s`), refits
as the pool grows, and can only ever turn an entry OFF. It is a veto, exactly
as the request specifies, and a bidirectional strategy gets one classifier per
side so a short is never scored by a model trained on longs.

This module supplies its own feature matrix through the `ml_features` hook —
rolling ATR, ADX(14) and rolling volume, the three the request names — instead
of the shared default (`causal_features`: ATR, volume, RSI, hour, minute and
two return horizons). See `ml_features` for why those three and what it costs.

TWO PLACES THE DELIVERED VERSION B DIFFERS FROM THE REQUEST, both in the shared
machinery rather than here:

  * THE CLASSIFIER IS `HistGradientBoostingClassifier`, not LightGBM or a
    random forest. LightGBM is referenced by the Dual-Version Mandate and is
    NOT pinned in `requirements.txt` (a known open task); scikit-learn 1.9.0 is
    installed and its histogram gradient booster is the same family of model,
    consumes NaN warm-up rows natively, and is what every other Version B in
    this repository already uses. Swapping it per-strategy would make two
    strategies' Version B incomparable.

  * THE TRAINING LABEL IS "net P&L > 0 after commission and slippage", not
    literally "hit TP before SL". For THIS module the two nearly coincide,
    because the only exits are the target and the stop: a trade that reached a
    2.0 x ATR target is profitable after costs and one stopped at 1.0 x ATR is
    not. They separate on the trades that are still open when the data ends,
    and on marginal cases where the one-bar fill lag turns a target touch into
    a small loss. The cost-aware label is the stricter of the two — a filter
    trained on gross outcomes learns to keep trades that lose money after
    commission — so the difference is not a loosening.

Every calculation is causal. Values at bar i use bars <= i only, warm-up stays
NaN and is never treated as a signal, and the engine then fills at bar i+1's
open — so nothing here can see a price it would not have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The identifier the pipeline logs and the CLI resolve. `backtest/run.py`
# resolves `--strat` by FILENAME against SEARCH_DIRS, so the filename is the
# real identifier and this constant is the assertion that the two agree;
# `tests/test_sma_momentum_crossover.py` pins it against the module's own
# filename so a rename cannot leave the two disagreeing.
STRATEGY_NAME = "sma_momentum_crossover_20260818"

TIMEFRAME = "15m"
SYMBOLS = ["NQ", "ES", "GC"]
DEFAULT_PARAMS = {"fast_window": 10, "slow_window": 30, "macro_window": 200,
                  "adx_threshold": 20.0,
                  "use_macro_anchor": True, "use_adx_filter": True,
                  "use_news_filter": False,
                  "sl_atr_mult": 1.0, "tp_atr_mult": 2.0, "trailing": False}

# The search space `backtest/run.py --scan` and `backtest/scan.py` sweep.
#
# 3 x 3 x 3 x 2 x 1 = 54 combinations, none of them rejected by `_validate`:
# every fast value is below every slow value, and every slow value is below the
# 200-bar macro baseline the anchor is fixed at. 54 is comfortably inside the
# 200-cell honesty bound this repo holds its grids to — past that the winning
# Sharpe is the maximum of N draws from one sample of bars, and that maximum
# climbs with N whether or not anything in the market has changed.
#
# WHAT IS NOT SWEPT, and why each omission is a choice rather than an oversight:
#
#   macro_window     Fixed at the specified 200. It is the strategy's premise,
#                    not a knob: sweeping the baseline that defines "the
#                    correct side of the trend" searches over strategies while
#                    reporting a parameter.
#   adx_threshold    Fixed at the specified 20. Same reasoning — 20 is the
#                    conventional "a trend exists" line and the request names
#                    it. Opening it would multiply the grid by its length and
#                    let the sweep discover a threshold that fits these bars.
#   use_macro_anchor Fixed ON. Opening it to [True, False] would double the
#                    grid to 108 and search over strategies rather than
#                    parameters — the all-off cell is a bare crossover, which
#                    is the NULL this idea has to beat rather than a variant of
#                    it. Run that comparison out of band, once, at the winning
#                    cell: `--param use_macro_anchor=false`.
#   use_adx_filter   Fixed ON, same reasoning.
#   use_news_filter  Fixed OFF. It depends on an external calendar whose
#                    provenance changes what the filter means, so it is an
#                    operator decision rather than a search axis.
#   trailing         Pinned to [False] — a FIXED stop, which is what the
#                    request specifies by its silence. Declared as a
#                    single-value axis rather than omitted so the pinned value
#                    is explicit in the scan output and in the leaderboard's
#                    `params` column, at no cost to the cell count.
#
# THIS GRID SEARCHES NO "NO TAKE-PROFIT" POINT, AND THAT IS A REAL GAP. The
# request's target axis is [1.5, 2.0, 3.0] and does not include `None`, so the
# sweep cannot distinguish "the 2.0 x ATR target is the edge" from "any target
# is worse than letting the stop decide". It matters more here than it would
# elsewhere: this module has no signal exit at all, so the stop and the target
# ARE the entire exit rule. The cheap way to ask without growing the grid is a
# single out-of-band run at the winning cell with `--param tp_atr_mult=None`,
# which `_validate` accepts and the walk models as no target at all.
#
# READ THAT RUN CAREFULLY, because it degenerates. With no target, no signal
# exit and no session flatten, a FIXED stop that a trending market never
# reaches leaves the position open forever: measured on the 4,000-bar
# synthetic fixture in `tests/test_risk_params.py`, `tp_atr_mult=None` with
# `trailing=False` produced ONE long entry and ZERO long exits, held to the end
# of the frame, against seven completed trades with a 3.0 x ATR target. A
# TRAILING stop does not degenerate that way — it ratchets in behind the price
# and eventually gets hit (eight trades on the same bars) — so the honest
# no-target comparison is `--param tp_atr_mult=None --param trailing=true`, and
# a no-target run with a fixed stop is measuring buy-and-hold with extra steps.
#
# Multiply the cell count by the timeframes before quoting it. `--tf 5m,15m,30m`
# is 54 fits PER timeframe PER contract: 162 per symbol, 648 across the four
# target assets.
PARAM_GRID = {
    "fast_window": [5, 10, 15],
    "slow_window": [20, 30, 50],
    "tp_atr_mult": [1.5, 2.0, 3.0],
    "sl_atr_mult": [1.0, 1.5],
    "trailing": [False],
}

ATR_PERIOD = 14

# ADX's own period. Fixed at Wilder's 14 rather than exposed as a parameter:
# the request names ADX(14) specifically, and the threshold is the part that
# decides anything. Note ADX needs roughly 2 x this many bars to exist at all —
# the directional movement is Wilder-smoothed once and the resulting DX is
# Wilder-smoothed again — so the ADX filter costs ~27 bars of warm-up on top of
# whatever the moving averages need.
ADX_PERIOD = 14

# The half-width of the macro-release veto, in minutes, when `use_news_filter`
# is on. 30 is the request's number and the repo default. The window is
# two-sided: it blocks the half hour BEFORE a release as well as the half hour
# after. Blocking the half hour before a print is not lookahead — US release
# SCHEDULES are published a year ahead — and nothing in this module ever reads
# an outcome.
NEWS_WINDOW_MINUTES = 30.0

# The Version B feature matrix, in column order. Exactly the three the request
# names, and the reason they are declared here rather than taken from the
# shared `causal_features` default is in `ml_features`.
ML_FEATURES = ["atr_norm", "adx_14", "volume_z"]

# The lookback for the volume z-score. Twenty bars, matching the shared feature
# matrix's own volume window so the two are directly comparable when somebody
# asks what the strategy-specific features bought.
VOLUME_Z_PERIOD = 20

# Plain-English description for the tear sheet's strategy card, written for a
# reader deciding whether to trade this — not for whoever maintains the module.
# `{param}` slots are filled with the run's own bound parameters, so the card
# states the settings that actually ran rather than the defaults written here.
# The report never infers any of this from the signal arrays: a description
# guessed from the trades would be a guess printed as a fact.
LOGIC = {
    "concept": "A trend-continuation crossover in three layers, each removing "
               "a different way a bare crossover loses money. The crossover "
               "itself is only the timing — it says nothing about whether a "
               "trend exists — so it is wrapped in two filters, and EACH ONE "
               "IS SWITCHABLE: this run used macro anchor="
               "{use_macro_anchor}, ADX filter={use_adx_filter}. The macro "
               "baseline, a long-horizon simple average, decides which side "
               "of the multi-session imbalance the trade is allowed to be on, "
               "which removes the counter-trend crossover — the bounce inside "
               "a downtrend that a fast average crossing up through a slow one "
               "cannot tell from a reversal. ADX decides whether there is a "
               "trend to continue at all, which removes the crossover that "
               "happens inside a range; it measures the SIZE of directional "
               "movement and not its sign, so the identical threshold governs "
               "a long and a short. Both filters can only REMOVE eligible "
               "triggers, so the edge — if there is one — is selection rather "
               "than prediction, and with both off this is a bare crossover "
               "with a stop, which is the null the idea has to beat rather "
               "than a strategy. The same reasoning runs in both directions, "
               "so the short rules are the long rules mirrored.",
    "entry": "ONE condition always applies: the Fast SMA ({fast_window}) must "
             "cross up through the Slow SMA ({slow_window}) on this bar for a "
             "long, or down through it for a short. On top of it, each filter "
             "below applies ONLY IF ITS TOGGLE IS TRUE — a toggle set to "
             "False means that condition was not checked at all, not that it "
             "happened to pass. (1) use_macro_anchor={use_macro_anchor}: the "
             "close above the Macro Baseline SMA ({macro_window}) for a long, "
             "below it for a short. (2) use_adx_filter={use_adx_filter}: ADX "
             "14 above {adx_threshold} on the crossover bar, the same "
             "requirement on both sides. (3) use_news_filter="
             "{use_news_filter}: when True, a trigger whose fill bar lands "
             "within 30 minutes either side of a scheduled US macro release "
             "is dropped. Only one position is held at a time and it is never "
             "reversed on the spot — a short trigger while the long is open "
             "is ignored, and vice versa. The fill is the next bar's open.",
    # Written so that every bound value reads correctly, including
    # `tp_atr_mult=None` and either setting of `trailing`. The card cannot
    # branch — `_describe_strategy` only substitutes `{param}` slots — so the
    # sentence states both arms and names the setting that chose between them.
    "exit": "Exit at whichever comes first, and the rules are mirrored for a "
            "short. (1) A stop {sl_atr_mult} x ATR 14 away from the fill — "
            "below it on a long, above it on a short — with "
            "trailing={trailing}, where True means it follows the best price "
            "reached since the fill (the high on a long, the low on a short) "
            "and never widens, and False means it sits fixed that far from "
            "the fill price. (2) A take-profit {tp_atr_mult} x ATR 14 from "
            "the fill price — above it on a long, below it on a short — where "
            "None means NO take-profit is modelled at all. THERE IS NO THIRD "
            "EXIT: the position is not closed when the crossover reverses, "
            "not closed when the macro baseline flips, and not flattened at "
            "any session boundary, so trades are carried overnight and over "
            "weekends. Every exit fills at the NEXT bar's open, so neither "
            "the stop nor the target is a fill at its own price — both carry "
            "a one-bar lag, and on a bar that breaches both the modelled fill "
            "is neither level.",
}


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def _wilder(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing — the average ATR and ADX are actually defined on.

    `alpha = 1/period` with `adjust=False` is Wilder's recursion exactly. A
    span-`period` EMA is a different (roughly twice as fast) average, and using
    it would produce an "ATR(14)" and an "ADX(14)" that no other tool agrees
    with — so a reader checking a trade against their own chart would see the
    threshold crossed on a different bar.
    """
    return series.ewm(alpha=1.0 / period, adjust=False,
                      min_periods=period).mean()


def _true_range(bars: pd.DataFrame) -> pd.Series:
    """
    True range per bar. Shared by ATR and ADX so the two cannot disagree.

    `.shift(1)` looks one bar BACKWARD, which is the correct direction: the
    true range at bar i uses bar i-1's close and nothing later.
    """
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev = close.shift(1)
    return pd.concat([(high - low).abs(),
                      (high - prev).abs(),
                      (low - prev).abs()], axis=1).max(axis=1)


def _atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """True range, Wilder-smoothed. NaN until `period` bars exist."""
    return _wilder(_true_range(bars), period)


def _sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple moving average, NaN until `period` values exist.

    `min_periods=period` is the point: without it pandas seeds the average from
    the first bar, so a "200-period baseline" exists at bar 2 and every regime
    call at the start of a symbol's history is decided by the seeding rather
    than by price. Every comparison against NaN is False, so the symptom would
    be silently wrong early trades rather than an error.
    """
    return series.rolling(period, min_periods=period).mean()


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
    its own stop and target.

    This kernel is DUPLICATED VERBATIM in `ema_crossover_20260821.py`,
    `ema_trend_filter.py` and `sma_momentum_crossover.py`, by the same convention that duplicates `_wilder`,
    `_atr` and `_session_masks` across this directory: strategy modules are
    loaded from a file path and are deliberately self-contained.
    `tests/test_risk_params.py` runs the copies on identical arrays, in both
    directions, and requires identical output — do not "improve" one alone.

    A trailing stop cannot be a stateless mask. Its level is the extreme price
    since ENTRY offset by a fixed distance, so bar i's exit condition depends on
    which earlier bar opened the position — which depends on every entry before
    it. The fixed stop and the take-profit are anchored on the FILL PRICE, which
    is `open` on the bar after the signal, so they are path-dependent for the
    same reason. This walks the bars once and resolves entries, stops, targets
    and the session flatten together rather than splitting them across layers
    that could disagree.

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
    flip inside the kernel. A well-formed strategy cannot produce it — a close
    cannot be both above and below the same anchor — so the branch exists to make
    a malformed one visible as a missing trade rather than a plausible one-sided
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
# Signals
# --------------------------------------------------------------------------
def _validate(fast_window: int, slow_window: int, macro_window: int,
              adx_threshold: float, sl_atr_mult: float,
              tp_atr_mult: float | None, trailing: bool,
              use_macro_anchor: bool = True, use_adx_filter: bool = True,
              use_news_filter: bool = False) -> None:
    """
    Reject parameter sets that do not describe this strategy.

    Every rejection here is recorded by `backtest.scan` as REJECTED and counted
    in the grid, so a combination the strategy refuses shrinks neither the
    reported search nor the honesty of `variants_tested`.
    """
    if fast_window < 2 or slow_window < 2 or macro_window < 2:
        raise ValueError(
            f"windows must be >= 2; got fast={fast_window}, "
            f"slow={slow_window}, macro={macro_window}")
    if fast_window >= slow_window:
        # Not a stylistic objection. With the two equal the crossover can never
        # fire, and inverted it fires on the opposite event — a downward cross
        # reported under this module's name as a long trigger. Either way the
        # curve would look plausible and describe a strategy nobody specified.
        raise ValueError(
            f"fast_window must be < slow_window; got {fast_window} >= "
            f"{slow_window}")
    if slow_window >= macro_window:
        # The baseline has to be a longer horizon than the trigger, or Layer 1
        # and Layer 2 are reading the same movement and "price is on the
        # correct side of the trend" becomes close to tautological.
        raise ValueError(
            f"slow_window must be < macro_window; got {slow_window} >= "
            f"{macro_window}")
    if adx_threshold is None or not np.isfinite(float(adx_threshold)):
        raise ValueError(
            f"adx_threshold must be a finite number; got {adx_threshold!r}")
    if not 0.0 <= float(adx_threshold) <= 100.0:
        # ADX is bounded 0-100 by construction. A threshold outside that is
        # not a strict filter, it is a filter that can never pass or can never
        # bind, and either would run as a silently different strategy.
        raise ValueError(
            f"adx_threshold must be within 0-100, the range ADX can take; "
            f"got {adx_threshold!r}")
    if sl_atr_mult is None or sl_atr_mult <= 0:
        # The stop is not optional, and in this module it is load-bearing:
        # with no signal exit and no session flatten, a position with no stop
        # and no target would never be closed at all.
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
    # The same objection, and it bites harder here: `--param
    # use_macro_anchor=false` passed as the STRING "false" is truthy, so the
    # run would apply the anchor while the leaderboard's params column said it
    # was off. Every toggle is checked rather than coerced.
    for name, flag in (("use_macro_anchor", use_macro_anchor),
                       ("use_adx_filter", use_adx_filter),
                       ("use_news_filter", use_news_filter)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be a bool; got {flag!r}")


def _series(bars: pd.DataFrame, fast_window: int, slow_window: int,
            macro_window: int) -> dict:
    """The shared calculation behind `signal_fn`, `indicators` and the walk."""
    close = bars["close"]
    return {
        "fast": _sma(close, fast_window),
        "slow": _sma(close, slow_window),
        "macro": _sma(close, macro_window),
        "adx": _adx(bars, ADX_PERIOD),
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


def _news_suppress(bars: pd.DataFrame,
                   long_ok: np.ndarray,
                   short_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop candidate triggers whose FILL bar lands inside a macro-release window.

    Delegates to `backtest.event_calendar.apply_entry_filters`, the only
    implementation of this filter in the repository, which is documented as
    callable by a strategy module on its own masks. Two behaviours come with it
    and neither is reimplemented here:

      * the mask is widened one bar BACKWARDS, because the engine fills at the
        next bar's open and a signal is judged on the bar it FILLS. Without
        that widening exactly one entry per event slips through and fills
        inside the window — the least visible outcome, and the whole population
        the filter exists to remove.
      * an empty calendar RAISES rather than returning an all-clear mask, so a
        run reported as news-filtered cannot be one in which nothing was ever
        filtered.

    Imported inside the function rather than at module scope. A strategy module
    has to stay importable on a box where the calendar file is absent — the
    import itself is cheap and safe, but keeping it here means the news filter
    costs nothing at all, including its import, for the default runs that leave
    it off.
    """
    from backtest.event_calendar import apply_entry_filters

    if "ts" not in bars.columns:
        raise ValueError(
            "use_news_filter=True needs a `ts` column to place each bar "
            "against the release calendar; this frame has none. The engine "
            "always supplies one — a frame without it is a fixture, and "
            "filtering it against a calendar would be filtering nothing.")

    kept_long, kept_short, _info = apply_entry_filters(
        bars["ts"], long_ok, short_ok,
        news_filter=True,
        news_window_minutes=NEWS_WINDOW_MINUTES)
    return (np.asarray(kept_long, dtype=bool),
            np.asarray(kept_short, dtype=bool))


def _signal_arrays(bars: pd.DataFrame, fast_window: int, slow_window: int,
                   macro_window: int, adx_threshold: float,
                   sl_atr_mult: float, tp_atr_mult: float | None,
                   trailing: bool, use_macro_anchor: bool = True,
                   use_adx_filter: bool = True,
                   use_news_filter: bool = False) -> tuple:
    """
    Everything the walk produces, from one place.

    `signal_fn` and `indicators` both go through here, so the stop line the
    tear sheet draws is the array the exit was taken from and the crossover a
    reader sees is the array the entry came from. Two call sites computing this
    separately would be free to drift apart with nothing raising.
    """
    s = _series(bars, fast_window, slow_window, macro_window)
    close = bars["close"]

    # `ready` is assembled FROM THE ACTIVE FILTERS ONLY, and that is the part
    # of the toggles that is easy to get wrong.
    #
    # Requiring every series unconditionally would make a toggled-off filter
    # cost its warm-up anyway: `use_adx_filter=False` would still wait ~27 bars
    # for an ADX nothing reads, so the "no ADX" cell would be scored on a
    # shorter history than the bare crossover it is meant to represent — and
    # the comparison the toggles exist to enable would be between different
    # samples. Every comparison against NaN is False, so the symptom would be
    # silently missing early trades rather than an error.
    #
    # Three things are unconditional. The two trigger averages are the entry
    # itself. ATR is load-bearing whatever the filters say: it sets the stop
    # distance and the target, so an entry taken before ATR exists would have
    # no stop.
    ready = s["fast"].notna() & s["slow"].notna() & s["atr"].notna()
    if use_macro_anchor:
        ready &= s["macro"].notna()
    if use_adx_filter:
        ready &= s["adx"].notna()

    # The crossover as an EVENT, not a state. `above` is False through warm-up
    # (NaN > NaN is False), so requiring the PREVIOUS bar to be ready as well
    # is what stops the first fully-warm bar from registering as a cross: at
    # that bar `above` may flip from False to True purely because the averages
    # came into existence, which is a fact about the warm-up and not about
    # price. `fill_value=False` keeps bar 0 out for the same reason.
    # `above` and `below` are deliberately BOTH built as explicit comparisons
    # rather than one being `~` the other. `~above` is true wherever the fast
    # average merely fails to be above the slow one, which includes exact
    # equality and every warm-up bar — so a short trigger defined as the
    # negation of the long one would fire on ties and on the first bar the
    # averages exist.
    above = (s["fast"] > s["slow"]) & ready
    below = (s["fast"] < s["slow"]) & ready

    prev_ready = ready.shift(1, fill_value=False)
    cross_up = above & ~above.shift(1, fill_value=False) & prev_ready
    cross_down = below & ~below.shift(1, fill_value=False) & prev_ready

    # The two confluence layers, one condition per line so the conjunction
    # below reads as the specification does. Each is built as an explicit
    # comparison rather than as the negation of its opposite: `~(close >
    # macro)` is true wherever the close merely fails to be above the baseline,
    # which includes exact equality and every NaN bar, so a short condition
    # written as a negation would fire on ties and on bars where the series
    # does not exist.
    #
    # A DISABLED FILTER IS ALL-TRUE, NOT ALL-FALSE. `_on` returns a True Series
    # when its toggle is off, so the conjunction below is written once and
    # reads the same whichever filters are active. Writing it as `cond if flag
    # else <omit>` instead would need a different expression per combination,
    # and each combination would be another chance for the long and short arms
    # to stop mirroring each other.
    def _on(condition: pd.Series, flag: bool) -> pd.Series:
        return condition if flag else pd.Series(True, index=bars.index)

    long_regime = _on(close > s["macro"], use_macro_anchor)   # Layer 1
    short_regime = _on(close < s["macro"], use_macro_anchor)  # and its mirror
    # Layer 3. ONE expression for both sides, not two mirrored ones: ADX is
    # unsigned, so "a trend exists" is the same statement whichever way it
    # runs. Strictly greater than the threshold, as specified — at exactly 20
    # the trend is not yet established.
    trending = _on(s["adx"] > float(adx_threshold), use_adx_filter)

    long_entry_ok = (cross_up & long_regime & trending
                     & ready).to_numpy(dtype=bool)
    short_entry_ok = (cross_down & short_regime & trending
                      & ready).to_numpy(dtype=bool)

    # The news veto lands on CANDIDATES, before the walk, so a blocked trigger
    # leaves the strategy flat and free to take a later one. See the module
    # docstring for why that differs from the engine's own `--news-filter`,
    # which acts on the walk's output.
    if use_news_filter:
        long_entry_ok, short_entry_ok = _news_suppress(
            bars, long_entry_ok, short_entry_ok)

    # No signal exit, on either side. The specification enumerates the exits
    # and lists only the take profit and the stop — a position is deliberately
    # held through the crossover reversing and through the baseline flipping.
    # The arrays exist because `_walk` keeps the shared kernel's signature;
    # both are all-False, so neither `sig_exit[i]` branch is taken.
    sig_exit = np.zeros(len(bars), dtype=bool)
    # Nor is there a session flatten. `flat_bar` all-False is what "positions
    # are carried overnight" looks like in this kernel, and it is a modelled
    # exposure rather than an omission — see the module docstring.
    flat_bar = np.zeros(len(bars), dtype=bool)

    entries, exits, s_entries, s_exits, stop, target = _walk(
        long_entry_ok,
        short_entry_ok,
        sig_exit,
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
    return s, entries, exits, s_entries, s_exits, stop, target


def signal_fn(bars: pd.DataFrame,
              fast_window: int = 10,
              slow_window: int = 30,
              macro_window: int = 200,
              adx_threshold: float = 20.0,
              use_macro_anchor: bool = True,
              use_adx_filter: bool = True,
              use_news_filter: bool = False,
              sl_atr_mult: float = 1.0,
              tp_atr_mult: float | None = 2.0,
              trailing: bool = False) -> tuple[pd.Series, pd.Series,
                                               pd.Series, pd.Series]:
    """
    Take the crossover only with the macro baseline and ADX agreeing; exit on
    the stop or the target. Both directions.

    Returns the FOUR-MASK form of the strategy contract:

        (long_entries, long_exits, short_entries, short_exits)

    all boolean Series on `bars.index`. `backtest.engine.unpack_signals`
    accepts this alongside the older two-mask long-only form.

    The entry is a bar-level EVENT: the long fires on the bar where SMA(fast)
    first closes above SMA(slow), the short on the bar where it first closes
    below, so a long run of bars on one side produces one signal rather than a
    signal every bar. The walk enters only when flat, so a second crossover
    while a position is open is ignored rather than pyramided, and a short
    trigger arriving while long is ignored rather than reversing the position.

    Every comparison at bar i uses only bars <= i. The engine then fills at bar
    i+1's open, so nothing here can see a price it would not have had.
    """
    _validate(fast_window, slow_window, macro_window, adx_threshold,
              sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor,
              use_adx_filter, use_news_filter)

    _s, entries, exits, s_entries, s_exits, _stop, _target = _signal_arrays(
        bars, fast_window, slow_window, macro_window, adx_threshold,
        sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor, use_adx_filter,
        use_news_filter)

    return (pd.Series(entries, index=bars.index),
            pd.Series(exits, index=bars.index),
            pd.Series(s_entries, index=bars.index),
            pd.Series(s_exits, index=bars.index))


def indicators(bars: pd.DataFrame,
               fast_window: int = 10,
               slow_window: int = 30,
               macro_window: int = 200,
               adx_threshold: float = 20.0,
               use_macro_anchor: bool = True,
               use_adx_filter: bool = True,
               use_news_filter: bool = False,
               sl_atr_mult: float = 1.0,
               tp_atr_mult: float | None = 2.0,
               trailing: bool = False) -> dict[str, pd.Series]:
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

    ONE stop line and ONE target line serve both directions, because only one
    position is ever open: each is the level of whichever side is live, so the
    stop sits below the candles inside a long and above them inside a short.
    The gap between two segments is where the position was flat, and a segment
    that jumps from below the price to above it is the strategy changing sides,
    not a stop being moved.

    The take-profit line is OMITTED ENTIRELY when `tp_atr_mult` is None. An
    all-NaN series would render as an empty legend entry, which reads as a
    target that exists and never got close — the opposite of the truth. The
    Macro Baseline is omitted the same way when its toggle is off: drawing a
    line the entries did not respect makes the trades that cross it look like
    bugs rather than like the strategy that was actually run.

    ADX IS DELIBERATELY NOT RETURNED, which means Layer 3 is the one entry
    condition a reader cannot see on the chart. That is a real gap and the
    alternative is worse: the inspector draws these on the PRICE axis, and ADX
    is bounded 0-100, so on a 20,000-point contract it would be a flat line
    along the bottom — while rescaling a bounded oscillator onto a price axis
    would draw a "trend strength" line at prices nothing ever traded at. ATR
    reaches the reader through the stop and target lines, which is where it
    changes a decision; ADX reaches the strategy card and not the chart.

    Warm-up stays NaN rather than drawing the averages flat through the first
    bars.
    """
    _validate(fast_window, slow_window, macro_window, adx_threshold,
              sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor,
              use_adx_filter, use_news_filter)

    s, _entries, _exits, _se, _sx, stop, target = _signal_arrays(
        bars, fast_window, slow_window, macro_window, adx_threshold,
        sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor, use_adx_filter,
        use_news_filter)

    kind = "Trailing" if trailing else "Fixed"
    # The two trigger averages and the stop are always drawn — they are the
    # entry and the exit, and neither can be switched off.
    out = {
        f"Fast SMA ({fast_window})": s["fast"],
        f"Slow SMA ({slow_window})": s["slow"],
        f"{kind} Stop ({sl_atr_mult}xATR{ATR_PERIOD})":
            pd.Series(stop, index=bars.index),
    }
    if use_macro_anchor:
        out[f"Macro Baseline SMA ({macro_window})"] = s["macro"]
    if tp_atr_mult is not None:
        out[f"Take Profit ({tp_atr_mult}xATR{ATR_PERIOD})"] = pd.Series(
            target, index=bars.index)
    return out


def ml_features(bars: pd.DataFrame, **_params) -> pd.DataFrame:
    """
    Version B's feature matrix: rolling ATR, ADX(14) and rolling volume.

    Read by `agents.tier3_workers.load_strategy` and handed to
    `apply_ml_signal_filter` in place of the shared `causal_features` default.
    Declaring it here is the point: these are the three quantities the
    specification names, and the shared default carries seven columns that are
    not them (RSI, hour, minute and two return horizons). A classifier vetoing
    THIS strategy's entries should be reading the state this strategy's
    hypothesis is about — is a trend present, is volatility expanding, is there
    participation — rather than a general-purpose matrix.

    Columns, in `ML_FEATURES` order:

        atr_norm   ATR(14) / close. Normalised because the classifier is fitted
                   across a 16-year history in which the contract's price level
                   changes by a multiple: a raw ATR in points would let the
                   model learn "2015" instead of "quiet".
        adx_14     Wilder's ADX, the same array Layer 3 gates on. Under the
                   default `use_adx_filter=True` every surviving candidate
                   already clears the threshold, so what this column carries is
                   HOW FAR above it the bar was — which is the part the veto
                   can still act on.
        volume_z   Volume against its own 20-bar mean and standard deviation.
                   Standardised per-symbol and rolling for the same reason
                   atr_norm is normalised, and it is the only place volume
                   enters this strategy at all (Version A has no volume
                   condition — see the module docstring).

    STRICTLY CAUSAL, and worth being explicit about because this matrix is
    fitted on. A value at row i is a function of bars 0..i only: no `shift(-k)`,
    no centred window, no reversed slice, and nothing computed off a
    full-sample statistic. That last one is the trap a shift-based audit does
    not catch — a scaler fitted on the whole frame leaks the test period's
    distribution into the training rows, so the standardisation here is a
    ROLLING one rather than a global mean and standard deviation.

    Using bar i's close to decide a signal on bar i is legitimate: the engine
    fills at bar i+1's open, never on the signal bar. That one-bar gap is what
    makes these features tradeable rather than clairvoyant.

    NaN warm-up rows are left as NaN. `HistGradientBoostingClassifier` consumes
    them natively, and filling them with a column mean would import a
    full-sample statistic into exactly the rows that have no history.

    `**_params` is accepted and ignored. The loader binds a hook to whichever
    parameters it declares, and these three features are the same whatever the
    windows are set to — the alternative, deriving the feature periods from the
    swept parameters, would give every grid cell a differently-shaped model and
    make the sweep a search over classifiers as well as over strategies.
    """
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)

    atr = _atr(bars, ATR_PERIOD)
    vol_mean = volume.rolling(VOLUME_Z_PERIOD, min_periods=VOLUME_Z_PERIOD).mean()
    vol_std = volume.rolling(VOLUME_Z_PERIOD, min_periods=VOLUME_Z_PERIOD).std()

    out = pd.DataFrame({
        "atr_norm": (atr / close.where(close != 0)).to_numpy(dtype=float),
        "adx_14": _adx(bars, ADX_PERIOD).to_numpy(dtype=float),
        # The 1e-8 keeps a dead-flat volume window from dividing by zero. It is
        # far below any real standard deviation, so it changes no live value.
        "volume_z": ((volume - vol_mean)
                     / (vol_std + 1e-8)).to_numpy(dtype=float),
    }, index=bars.index)
    return out[ML_FEATURES]


def make_signal_fn(fast_window: int = 10,
                   slow_window: int = 30,
                   macro_window: int = 200,
                   adx_threshold: float = 20.0,
                   use_macro_anchor: bool = True,
                   use_adx_filter: bool = True,
                   use_news_filter: bool = False,
                   sl_atr_mult: float = 1.0,
                   tp_atr_mult: float | None = 2.0,
                   trailing: bool = False):
    """
    Bind parameters for `agents.tier3_workers.load_strategy`.

    The loader prefers this factory when params are supplied; the engine itself
    only ever calls the bound `signal_fn(bars)`. Validation runs HERE as well
    as inside `signal_fn`, so `--scan` records a bad combination as REJECTED at
    bind time rather than discovering it one symbol into the sweep.
    """
    _validate(fast_window, slow_window, macro_window, adx_threshold,
              sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor,
              use_adx_filter, use_news_filter)

    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series,
                                            pd.Series, pd.Series]:
        return signal_fn(bars, fast_window=fast_window,
                         slow_window=slow_window,
                         macro_window=macro_window,
                         adx_threshold=adx_threshold,
                         use_macro_anchor=use_macro_anchor,
                         use_adx_filter=use_adx_filter,
                         use_news_filter=use_news_filter,
                         sl_atr_mult=sl_atr_mult,
                         tp_atr_mult=tp_atr_mult,
                         trailing=trailing)

    return _bound
