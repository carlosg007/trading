"""
T3 Braid Scalp — an intraday trend-continuation scalp in three stacked layers:
a Tillson T3 baseline decides which side is permitted, a Braid histogram times
the entry, and a Stiffness Index refuses to trade a market whose closes are not
persistently moving one way.

Location:  ~/src/trading/strategies/experimental/t3_braid_scalp_20260823.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding — do not edit this block to match it.

    1. STRATEGY METADATA & REGISTRY BLOCK
       Strategy Identifier / Script Name: "t3_braid_scalp_20260823"
       Strategy Category / Archetype:     Intraday Scalp / Trend Continuation
       Target Repository Destination:
           strategies/experimental/t3_braid_scalp_20260823.py

    2. MULTI-LAYER ENTRY & EXIT CONFLUENCE LOGIC
       Layer 1 (Trend Baseline - T3 Filter):
           Compute Tillson T3 Moving Average.
           Long Entry:  Close > T3 (and T3 slope is rising / green).
           Short Entry: Close < T3 (and T3 slope is falling / red).
       Layer 2 (Momentum - Braid Filter):
           Compute Braid Filter histogram (based on fast/slow EMA separation).
           Long Confirmation:  Braid histogram is positive (green) AND
               positioned above its internal moving average line.
           Short Confirmation: Braid histogram is negative (red) AND
               positioned below its internal moving average line.
       Layer 3 (Volatility & Trend Consistency - Stiffness Index):
           Compute Stiffness Index (measuring the proportion of close-to-close
           changes in the direction of the trend over a lookback window).
           Entry Filter: Stiffness Index >= stiffness_threshold (Default: 35)
           to filter out choppy, non-persistent price action.
       Layer 4 (Exits & Risk Management):
           Stop Loss & Take Profit: Driven by ATR Bands with variable
           multipliers (`sl_atr_mult`, `tp_atr_mult`).
           Bounded Risk: Enforce parameter bounds ensuring risk-reward ratios
           remain viable after accounting for exchange execution costs and
           slippage.
       Modular Toggles: use_t3_filter, use_braid_filter, use_stiffness_filter,
           use_news_filter

    3. PARAMETERS & VECTORBT PRO OPTIMIZATION GRID
       Defaults:   t3_period=5, t3_vfactor=0.7, braid_fast=3, braid_slow=7,
                   stiffness_period=20, stiffness_threshold=35.0,
                   sl_atr_mult=1.5, tp_atr_mult=2.0, trailing=False
       PARAM_GRID: t3_period           [5, 9, 14]
                   stiffness_threshold [25.0, 35.0, 45.0]
                   sl_atr_mult         [1.0, 1.5, 2.0]
                   tp_atr_mult         [1.5, 2.0, 3.0]
                   trailing            [False, True]

    4. DUAL-VERSION ARCHITECTURE
       Declarations: signal_fn, indicators, LOGIC, PARAM_GRID, make_signal_fn,
                     ml_features.
       VERSION A: pure rule-based, strict next-bar-open execution
                  (`signal_delay=1`).
       VERSION B: apply_ml_signal_filter(..., features=ml_features), model
                  sklearn.ensemble.HistGradientBoostingClassifier, causal
                  features = T3 slope value, Braid histogram height, Stiffness
                  Index level, normalized rolling ATR, Hour-of-Day; binary
                  label 1 if trade net P&L after costs > 0 else 0; strictly a
                  causal veto on Version A signals.

    5. MICROSTRUCTURE & SAFETY AUDIT
       Strict next-bar-open execution, zero lookahead. Degenerate states (zero
       range / zero ATR) handled with safe defaults, no NaN propagation.
       Pandas 3.0 compatible — no `.fillna(method=...)`, no timezone-unaware
       slicing. `use_news_filter` wired to `backtest.event_calendar`.

    6. UNIT TESTING
       tests/test_t3_braid_scalp_20260823.py — signal dtypes, risk-key
       integration (`sl_atr_mult`, `tp_atr_mult`), causal feature alignment.

FIVE PLACES THE REQUEST WAS UNDER-SPECIFIED, AND WHAT WAS CHOSEN
================================================================
Written down rather than decided silently, because each one changes which
trades this module takes and none of them is recoverable from the equity curve.

1. THE ENTRY IS A RISING EDGE, NOT A STATE. The request phrases every layer as
   a standing condition ("Close > T3"), which read literally arms an entry on
   EVERY bar the confluence holds. The walk below only enters from flat, so a
   literal reading re-enters on the bar immediately after every stop-out, into
   the same conditions that just produced the loss, for as long as the
   confluence lasts — one trend leg becomes a chain of stopped scalps whose
   count is set by the stop distance rather than by the signal. This module
   fires on the bar the FULL confluence first becomes true and not again until
   it has been false at least once: one trade per confluence episode. The
   candidate masks are the rising edge of the conjunction, so a layer switched
   off changes where the edges fall, not merely how many survive.

2. THE BRAID SIGNAL LINE IS PINNED AT 14 AND IS NOT A PARAMETER. The request
   names two Braid lengths (3 and 7) and then requires the histogram to be
   "positioned above its internal moving average line" — a third length it
   does not name. Robert Hill's original Braid Filter carries a third length of
   14, so that is what `BRAID_SIGNAL_PERIOD` is set to. It is a module constant
   rather than a swept or bindable parameter: the request's grid does not
   contain it, and a third smoothing length exposed to `--param` is an axis
   somebody would eventually sweep, which turns "does the momentum layer help"
   into a search over momentum layers.

3. THE STIFFNESS INDEX IS DEFINED PER CANDIDATE SIDE. The request defines it as
   "the proportion of close-to-close changes in the direction of the trend",
   which needs a trend direction before it can be computed. Two readings were
   available: key it to the T3 slope, or key it to the side of the candidate
   being tested. This module uses the CANDIDATE SIDE — the long test reads the
   proportion of UP closes and the short test the proportion of DOWN closes.
   The two readings coincide under the default stack (a long candidate already
   requires T3 rising) and diverge only when `use_t3_filter=False`, where the
   side-keyed reading is the one that still means something: with no baseline
   there is no "trend" for a slope-keyed version to point at, and it would
   quietly become an unfilterable all-True layer. Keying it to the side also
   makes the filter exactly mirrored, so it cannot smuggle a directional bias
   in the way a single unsigned threshold applied to a signed quantity would.

   NOTE THAT UP AND DOWN ARE NOT COMPLEMENTS. An unchanged close counts to
   neither, so `up_pct + down_pct <= 100` and both sides can clear a threshold
   of 35 on the same bar. Layer 1 is what picks the side; Layer 3 only ever
   removes candidates.

4. NO TIMEFRAME AND NO TARGET ASSETS WERE SPECIFIED. `TIMEFRAME` and `SYMBOLS`
   below are DEFAULTS chosen so `bt-run` has something to run — 5m because the
   archetype is a scalp, and the four liquid contracts the other experimental
   modules target. Neither is part of the specification and both are overridden
   by `--tf` and `--symbols`. Do not read them as a claim that this strategy
   was designed for those bars.

5. THE FILENAME IN THE REQUEST'S OPENING LINE WAS A TYPO.
   `t3_braid_scalp_202608123.py` carries nine digits where every other mention
   in the same request — the identifier, the target destination and the test
   filename — reads `20260823`. The eight-digit spelling is what this file uses,
   because `backtest/run.py` resolves `--strat` by FILENAME and the identifier
   and the filename have to agree.

WHAT THIS MODULE DOES NOT DO, AND WHY
=====================================
Stated here rather than approximated silently, because each one changes how a
number this module produces must be read.

1. THERE ARE NO STOP OR TARGET ORDERS. `backtest/engine.py` drives
   `vbt.Portfolio.from_signals` off boolean masks and fills them at the NEXT
   bar's open. A stop here is an exit SIGNAL detected on the bar that breaches
   it and filled one bar later, at whatever the next open happens to be — not a
   fill at the stop price. On a bar breaching both the stop and the target the
   modelled fill is neither level. THIS BITES HARDER ON A SCALP THAN ANYWHERE
   ELSE IN THIS DIRECTORY: at 5m a 1.5 x ATR stop and one bar of slippage are
   the same order of magnitude, so a drawdown figure from this module is not
   bounded by `sl_atr_mult x ATR` and must never be quoted as though it were.

2. THE ONLY EXITS ARE THE STOP AND THE TARGET. There is no exit on the
   confluence breaking, no exit on the T3 slope rolling over, and no session
   flatten — the request's Layer 4 enumerates a stop and a target and nothing
   else. Two consequences: the stop is load-bearing (`_validate` refuses to run
   without one), and positions are carried overnight and over weekends. For an
   INTRADAY SCALP that is a real and unpriced exposure, and it is the single
   largest gap between what this module is called and what it does.

3. LAYER 3 IS NAMED "Volatility & Trend Consistency" BUT CONTAINS NO
   VOLATILITY CONDITION. The request's own rule for that layer is the
   Stiffness Index against a threshold and nothing else, so that is what
   Version A tests. Volatility reaches the strategy through the ATR that sizes
   the stop and the target, and through Version B as a feature — never as an
   entry condition.

4. "BOUNDED RISK" IS A COARSE SANITY FLOOR HERE, NOT A COST-ADJUSTED VIABILITY
   PROOF. The request asks for bounds "ensuring risk-reward ratios remain
   viable after accounting for exchange execution costs and slippage". This
   module cannot do that: costs are per-contract and live in
   `backtest/specs.py`, which a strategy module does not read, and the drag
   depends on the realised trade count, which does not exist until the run
   does. What `_validate` enforces is `MIN_REWARD_RISK`, `MIN_STOP_ATR_MULT`
   and the outer bounds below — enough to reject a target the fill bar would
   breach or a stop inside the tick noise, and not enough to certify anything.
   The real answer is Stage 4's cost drag as a share of GROSS profit
   (`backtest/verify_full.py`), and it is the number to read before this
   strategy is believed.

5. THE NEWS FILTER IS OFF BY DEFAULT. It needs a macro calendar to be
   meaningful, and `backtest/event_calendar.py` RAISES rather than returning an
   all-clear mask when its calendar does not cover the run's span. Check what a
   filtered run would actually use before trusting it:

       python3 backtest/event_calendar.py --start 2013-01-01 --end 2026-01-01

   Only NFP's rule-generated dates follow the real convention; CPI, PPI and
   FOMC anchors land in the right week and often the wrong day, and a 30-minute
   window on the wrong day blocks a random half hour while leaving the release
   tradeable. `use_news_filter=True` against a RULE-provenance calendar is not
   a run that dodged the actual prints.

THE NEWS FILTER IS IMPORTED, NOT REIMPLEMENTED
==============================================
`use_news_filter` calls `backtest.event_calendar.apply_entry_filters`, the only
implementation of that filter in this repository and documented as callable by
a strategy module on its own masks. A second copy here would be free to
disagree with the engine's about which bars a release covers, and the two would
be compared by nobody.

THIS PUTS THE MODULE OUTSIDE `ALLOWED_IMPORTS`, DELIBERATELY, exactly as
`sma_momentum_crossover_20260818` is. `agents.tier3_workers.ALLOWED_IMPORTS`
permits a strategy module numpy, pandas, math, vectorbtpro and numba and
nothing else — but that allowlist governs MODEL-GENERATED code, which is
audited before execution; it does not run over hand-written modules, which
`load_strategy` imports from a file path. The consequence, stated rather than
discovered later: this module would be REJECTED by the AST validator if it were
ever passed through the synthesis path, and the test suite pins that the
`backtest.event_calendar` import is the ONLY objection the validator has to it,
so a future edit reaching for `open`, `eval` or a network library fails loudly
instead of hiding behind an exception granted for something else.

It is applied to the CANDIDATE triggers, before the position walk, which is a
different thing from the engine's `--news-filter` flag acting on the walk's
OUTPUT entries. Blocking a candidate leaves the strategy FLAT and therefore
free to take a later trigger it would otherwise have been holding through, so
the two orderings do not produce the same trade list. The engine's flag still
works and still composes — it will simply find fewer entries left to remove. Do
not read a trade COUNT to decide whether the filter bound: it subtracts
candidates, not trades.

VERSION B — THE ML FILTER
=========================
Version B is `agents.tier3_workers.apply_ml_signal_filter`, the repository's
shared expanding-window walk-forward: for a candidate signalled on bar `s` it
fits only on trades that had already CLOSED before `s` (`exit_idx < s`), refits
as the pool grows, and can only ever turn an entry OFF. It is a veto, exactly
as the request specifies, and a bidirectional strategy gets one classifier per
side so a short is never scored by a model trained on longs.

The request's model — `HistGradientBoostingClassifier` — is what the shared
filter already uses, and the request's label ("1 if trade Net P&L after costs
> 0, else 0") is the shared label verbatim. Nothing about Version B is
strategy-specific here except the feature matrix, which this module supplies
through the `ml_features` hook: the five columns the request names, replacing
the shared seven-column `causal_features` default. See `ml_features`.

Every calculation is causal. Values at bar i use bars <= i only, warm-up stays
NaN and is never treated as a signal, and the engine then fills at bar i+1's
open — so nothing here can see a price it would not have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The identifier the pipeline logs and the CLI resolve. `backtest/run.py`
# resolves `--strat` by FILENAME against SEARCH_DIRS, so the filename is the
# real identifier and this constant is the assertion that the two agree; the
# test suite pins it against the module's own filename so a rename cannot leave
# the two disagreeing.
STRATEGY_NAME = "t3_braid_scalp_20260823"

# NEITHER OF THESE IS IN THE REQUEST — see deviation 4 in the docstring. They
# are defaults so `bt-run` has something to run, not a claim about where the
# edge is, and `--tf` / `--symbols` override both.
TIMEFRAME = "5m"
SYMBOLS = ["NQ", "ES", "GC"]

DEFAULT_PARAMS = {
    "t3_period": 5,
    "t3_vfactor": 0.7,
    "braid_fast": 3,
    "braid_slow": 7,
    "stiffness_period": 20,
    "stiffness_threshold": 35.0,
    "use_t3_filter": True,
    "use_braid_filter": True,
    "use_stiffness_filter": True,
    "use_news_filter": False,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 2.0,
    "trailing": False,
}

# The search space `backtest/run.py --scan` and `backtest/scan.py` sweep. The
# request's grid, transcribed: 3 x 3 x 3 x 3 x 2 = 162 combinations, none of
# them rejected by `_validate` (every (sl, tp) pair clears MIN_REWARD_RISK —
# the tightest is tp=1.5 against sl=2.0, a ratio of 0.75).
#
# 162 is inside the 200-cell honesty bound this repo holds its grids to, but
# only just, and `backtest/run.py` prints the cell count before it sweeps.
# MULTIPLY IT BY THE TIMEFRAMES BEFORE QUOTING IT: `--tf 5m,15m,30m` is 162
# fits PER timeframe PER contract — 486 per symbol, 1,944 across the four
# default assets. The winning Sharpe is the maximum of that many draws from one
# sample of bars, and that maximum climbs with N whether or not anything in the
# market has changed.
#
# WHAT IS NOT SWEPT, and why each omission is a choice rather than an oversight:
#
#   t3_vfactor            Fixed at the specified 0.7. It is the T3's shape —
#                         at 0 the curve degenerates to a triple EMA and at 1
#                         it is maximally responsive — so sweeping it searches
#                         over baselines while reporting a parameter. The
#                         request's grid does not open it.
#   braid_fast/braid_slow Fixed at the specified 3 and 7, same reasoning. The
#                         request's grid does not open them.
#   stiffness_period      Fixed at the specified 20. The THRESHOLD is the part
#                         the request opens, and it is the part that decides
#                         anything: the lookback sets what "persistent" is
#                         measured over, which is the layer's premise.
#   BRAID_SIGNAL_PERIOD   Not a parameter at all — see deviation 2.
#   the three layer       Fixed ON. Opening them to [True, False] would
#   toggles               multiply the grid by eight and search over
#                         STRATEGIES rather than parameters — the all-off cell
#                         is not a variant of this idea, it is the null this
#                         idea has to beat (and `_validate` refuses it
#                         outright; see `_validate`). Run those comparisons out
#                         of band, once, at the winning cell:
#                         `--param use_braid_filter=false`.
#   use_news_filter       Fixed OFF. It depends on an external calendar whose
#                         provenance changes what the filter means, so it is an
#                         operator decision rather than a search axis.
#
# THIS GRID SEARCHES NO "NO TAKE-PROFIT" POINT, AND THAT IS A REAL GAP. The
# request's target axis is [1.5, 2.0, 3.0] and does not include `None`, so the
# sweep cannot distinguish "the 2.0 x ATR target is the edge" from "any target
# is worse than letting the stop decide". It matters here because this module
# has no signal exit at all, so the stop and the target ARE the entire exit
# rule. The cheap way to ask without growing the grid is a single out-of-band
# run at the winning cell with `--param tp_atr_mult=None`, which `_validate`
# accepts and the walk models as no target at all — and pair it with
# `--param trailing=true`, because a FIXED stop with no target and no signal
# exit never closes a position a trending market keeps running.
PARAM_GRID = {
    "t3_period": [5, 9, 14],
    "stiffness_threshold": [25.0, 35.0, 45.0],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [1.5, 2.0, 3.0],
    "trailing": [False, True],
}

# Wilder's ATR period. Fixed at 14 rather than exposed: the request's Layer 4
# says "ATR Bands" without naming a length, 14 is the universal default, and
# the MULTIPLIERS are the part the grid opens.
ATR_PERIOD = 14

# The Braid histogram's own smoothing — its "internal moving average line".
# Pinned, not a parameter; see deviation 2 in the module docstring.
BRAID_SIGNAL_PERIOD = 14

# The half-width of the macro-release veto, in minutes, when `use_news_filter`
# is on. 30 is the repo default. The window is two-sided: it blocks the half
# hour BEFORE a release as well as the half hour after. Blocking the half hour
# before a print is not lookahead — US release SCHEDULES are published a year
# ahead — and nothing in this module ever reads an outcome.
NEWS_WINDOW_MINUTES = 30.0

# The exchange's own zone, for the Hour-of-Day feature. A NAMED ZONE rather
# than a fixed UTC offset, per CLAUDE.md: the CME session keeps its local clock
# across the DST changeover, so a UTC hour smears one session hour across two
# values twice a year and hands the classifier a feature that means a different
# thing in March than it does in October.
SESSION_TZ = "America/New_York"

# Version B's feature matrix, in column order — the five the request names.
ML_FEATURES = ["t3_slope", "braid_hist", "stiffness", "atr_norm", "hour_et"]

# --------------------------------------------------------------------------
# Bounded risk. See "WHAT THIS MODULE DOES NOT DO" item 4: these are a sanity
# floor, not a cost-adjusted viability proof, and they are named constants so
# the bound a rejection cites is the bound a reader can find.
# --------------------------------------------------------------------------
# A target closer than half the stop needs better than a 2-in-3 win rate to
# break even BEFORE costs, and this is a scalp — the round turn is charged on
# every one of those trades. Set at 0.5 rather than at 1.0 because the
# request's own grid contains a 0.75 cell (tp=1.5 against sl=2.0) and refusing
# a cell the specification asks for would be this module overruling the
# request rather than bounding it.
MIN_REWARD_RISK = 0.5
# A stop tighter than a quarter of ATR sits inside the bar noise the ATR is
# measuring; with one tick of slippage each way modelled at the engine, it is
# breached by the spread rather than by the market.
MIN_STOP_ATR_MULT = 0.25
# Outer bounds, to catch a transposed or mistyped multiplier rather than to
# express a view. A 25 x ATR stop is not a stop.
MAX_STOP_ATR_MULT = 25.0
MAX_TARGET_ATR_MULT = 50.0


# Plain-English description for the tear sheet's strategy card, written for
# whoever is deciding whether to trade this — not for whoever maintains the
# module. `{param}` slots are filled with the run's own bound parameters, so
# the card states the settings that actually ran rather than the defaults
# written here. The report never infers any of this from the signal arrays: a
# description guessed from the trades is a guess printed as a fact.
LOGIC = {
    "concept": "An intraday scalp that only ever trades WITH a trend it can "
               "already see, in three layers, each removing a different way a "
               "scalp bleeds. The T3 baseline — a six-fold smoothed average "
               "that turns faster than a plain moving average of the same "
               "length without the lag that makes a slow average useless on a "
               "scalp — decides which side is permitted at all: longs only "
               "above a rising T3, shorts only below a falling one, so a "
               "counter-trend bounce is never bought. The Braid histogram, "
               "the gap between a fast and a slow average of the close, is "
               "the timing: it must be on the right side of zero AND pulling "
               "away from its own smoothing, which is the difference between "
               "momentum that exists and momentum that is fading. The "
               "Stiffness Index is the veto: it counts what proportion of the "
               "last {stiffness_period} closes moved in the direction being "
               "traded, and refuses the trade below {stiffness_threshold} "
               "percent — a market that alternates up and down closes will "
               "satisfy the first two layers repeatedly and stop out on every "
               "one of them. EACH LAYER IS SWITCHABLE and this run used T3 "
               "filter={use_t3_filter}, Braid filter={use_braid_filter}, "
               "Stiffness filter={use_stiffness_filter}. Every layer can only "
               "REMOVE bars from the set the strategy is PERMITTED to "
               "trade on — but not necessarily remove trades, and a layer "
               "switched on can leave MORE entries than it found: the entry "
               "fires where the confluence starts, so breaking one long "
               "permitted stretch into two creates a second start. Judge a "
               "layer by the trade list, never by the count. The short rules "
               "are the long rules mirrored throughout.",
    "entry": "The trade fires on the bar where ALL of the active conditions "
             "below first hold together, and not again until at least one of "
             "them has failed — one entry per episode, not one per bar. Each "
             "condition applies ONLY IF ITS TOGGLE IS TRUE; a toggle set to "
             "False means that condition was not checked at all, not that it "
             "happened to pass. (1) use_t3_filter={use_t3_filter}: for a "
             "long, the close above the Tillson T3 ({t3_period}, vfactor "
             "{t3_vfactor}) and the T3 itself higher than on the previous "
             "bar; for a short, the close below it and the T3 lower. (2) "
             "use_braid_filter={use_braid_filter}: for a long, the Braid "
             "histogram — the fast EMA ({braid_fast}) minus the slow EMA "
             "({braid_slow}) of the close — above zero AND above its own "
             "14-period smoothing; for a short, below zero AND below it. (3) "
             "use_stiffness_filter={use_stiffness_filter}: at least "
             "{stiffness_threshold} percent of the last {stiffness_period} "
             "close-to-close changes moving the way the trade is going — up "
             "for a long, down for a short. (4) use_news_filter="
             "{use_news_filter}: when True, a trigger whose FILL bar lands "
             "within 30 minutes either side of a scheduled US macro release "
             "is dropped. Only one position is held at a time and it is never "
             "reversed on the spot — a short trigger while the long is open "
             "is ignored, and vice versa. The fill is the next bar's open.",
    # Written so that every bound value reads correctly, including
    # `tp_atr_mult=None` and either setting of `trailing`. The card cannot
    # branch — the report only substitutes `{param}` slots — so the sentence
    # states both arms and names the setting that chose between them.
    "exit": "Exit at whichever comes first, and the rules are mirrored for a "
            "short. (1) A stop {sl_atr_mult} x ATR 14 away from the fill "
            "price — below it on a long, above it on a short — with "
            "trailing={trailing}, where True means it follows the best price "
            "reached since the fill (the high on a long, the low on a short) "
            "and never widens, and False means it sits fixed that far from "
            "the fill. (2) A take-profit {tp_atr_mult} x ATR 14 from the fill "
            "price — above it on a long, below it on a short — where None "
            "means NO take-profit is modelled at all. THERE IS NO THIRD EXIT: "
            "the position is not closed when the confluence breaks, when the "
            "T3 slope rolls over, or at the end of the session, so despite "
            "the name this scalp carries positions overnight and over "
            "weekends. Both distances are frozen at the ATR measured on the "
            "SIGNAL bar and are never re-measured. Neither is an ORDER: each "
            "is detected on the bar that breaches it and filled at the NEXT "
            "bar's open, so the realised loss on a stop is not "
            "{sl_atr_mult} x ATR.",
}


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    A conventional EMA, `alpha = 2 / (period + 1)`, NaN until `period` values.

    `min_periods=period` is the point: without it pandas seeds the average from
    the first bar, so a "14-period EMA" exists at bar 2 and every early signal
    is decided by the seeding rather than by price. Every comparison against
    NaN is False, so the symptom would be silently wrong early trades rather
    than an error.

    This is the span form, NOT Wilder's `alpha = 1/period`. Both appear in this
    directory and they are different averages: `_wilder` below is used for ATR,
    where Wilder's is the definition, and this one for the T3 and the Braid,
    where Tillson's and the MACD family's is the span form. Using one where the
    other belongs produces an indicator no chart package agrees with, and the
    disagreement is a fraction of a bar's move — invisible except at a
    threshold.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _t3(close: pd.Series, period: int, vfactor: float) -> pd.Series:
    """
    Tillson's T3: six chained EMAs recombined by a cubic in the volume factor.

    The construction, written out because "T3" names one calculation and it is
    worth being able to check this against a chart by eye:

        e1..e6 = EMA chained six deep, each of span `period`
        c1 = -v^3
        c2 =  3v^2 + 3v^3
        c3 = -6v^2 - 3v - 3v^3
        c4 =  1 + 3v + v^3 + 3v^2
        T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3

    At `v = 0` the coefficients collapse to `T3 = e3`, a plain triple EMA; at
    `v = 1` the recombination is maximally responsive and overshoots. 0.7 is
    Tillson's own default and the request's.

    WARM-UP IS SIX DEEP AND THAT IS NOT A ROUNDING ERROR. Each `_ema` needs
    `period` non-NaN inputs, and each stage's input is the previous stage's
    output, so the first non-NaN T3 lands around bar `6 * (period - 1)` —
    roughly bar 25 at `period=5` and bar 79 at `period=14`. The slope needs one
    more. On a 5m chart that is a couple of sessions of warm-up, and `ready`
    below refuses to signal through it rather than treating the NaN as False.

    CAUSAL: an EMA is a recursion over past values only, and chaining six of
    them cannot reach forward. Nothing here reads a bar later than i.
    """
    v = float(vfactor)
    e1 = _ema(close, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    e4 = _ema(e3, period)
    e5 = _ema(e4, period)
    e6 = _ema(e5, period)

    c1 = -(v ** 3)
    c2 = 3.0 * v ** 2 + 3.0 * v ** 3
    c3 = -6.0 * v ** 2 - 3.0 * v - 3.0 * v ** 3
    c4 = 1.0 + 3.0 * v + v ** 3 + 3.0 * v ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _braid(close: pd.Series, fast: int, slow: int) -> tuple[pd.Series,
                                                            pd.Series]:
    """
    The Braid histogram and its internal moving average line.

    `hist` is the separation of a fast and a slow EMA of the close — positive
    ("green") when the short horizon is above the long one. `signal` is that
    histogram's own EMA at `BRAID_SIGNAL_PERIOD`, which is the line the request
    means by "positioned above its internal moving average line".

    The histogram being ABOVE its own smoothing is a different statement from
    the histogram being positive, and the request requires both: positive says
    the fast average leads, above-signal says the lead is still WIDENING.
    Momentum that is positive and narrowing is a trend being handed back, which
    on a scalp is the population that stops out.

    Returned as a pair from ONE function so the two can never be computed from
    different EMAs by two call sites.

    The signal line's warm-up compounds on the histogram's: `hist` needs `slow`
    bars and `signal` needs `BRAID_SIGNAL_PERIOD` non-NaN values of it.
    """
    hist = _ema(close, fast) - _ema(close, slow)
    return hist, _ema(hist, BRAID_SIGNAL_PERIOD)


def _stiffness(close: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """
    The Stiffness Index, per direction: the percentage of the last `period`
    close-to-close changes that moved UP, and the percentage that moved DOWN.

    Two series rather than one because the layer is keyed to the CANDIDATE
    SIDE — see deviation 3 in the module docstring. The long test reads
    `up_pct`, the short test `down_pct`, and the filter is therefore exactly
    mirrored: it cannot express a directional preference, only a persistence
    requirement.

    THEY ARE NOT COMPLEMENTS. An unchanged close counts to neither, so
    `up_pct + down_pct <= 100`, with equality only on a window containing no
    flat bar. On an illiquid contract or a thin overnight session the flat bars
    are exactly the choppy, untradeable state this layer exists to refuse — so
    counting them as evidence for either direction, which `100 - down_pct`
    would do, would make the filter loosest precisely where it should bind
    hardest.

    THE FIRST DIFFERENCE IS NaN AND IS EXCLUDED RATHER THAN COUNTED AS FLAT.
    `(close.diff() > 0)` alone is False at bar 0 for want of a previous close,
    which a plain rolling mean would read as a genuine non-up bar and would
    depress the first window's reading. Masking to `d.notna()` and letting
    `min_periods` do the work means the index is NaN until `period` real
    changes exist, and every value it does report is over a full window.

    CAUSAL: `.diff()` and `.rolling()` both look strictly backwards.
    """
    d = close.diff()
    up = (d > 0).astype(float).where(d.notna())
    down = (d < 0).astype(float).where(d.notna())
    roll = dict(window=period, min_periods=period)
    return (100.0 * up.rolling(**roll).mean(),
            100.0 * down.rolling(**roll).mean())


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing — the average ATR is actually defined on.

    `alpha = 1/period` with `adjust=False` is Wilder's recursion exactly. A
    span-`period` EMA (`_ema` above) is a different, roughly twice as fast,
    average, and using it here would produce an "ATR(14)" that no other tool
    agrees with — so a reader checking a stop distance against their own chart
    would see it placed somewhere else.
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
    A window in which nothing moved at all — a halted contract, a dead
    overnight hour — has a true ATR of zero, and the request's "handle
    degenerate states (zero range / zero ATR) gracefully" is satisfied
    downstream rather than here: `_signal_arrays` passes
    `np.nan_to_num(atr, nan=0.0)` into the walk, so warm-up NaN becomes a zero
    stop DISTANCE on bars where `ready` already forbids an entry, and a
    genuinely zero ATR on a live bar produces a stop at the fill price that the
    fill bar itself breaches. Both are visible as an immediate exit rather than
    as a NaN level nothing ever breaches, which is what a NaN stop distance
    would silently become.
    """
    return _wilder(_true_range(bars), period)


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
    its own stop and target.

    This kernel is DUPLICATED VERBATIM from
    `sma_momentum_crossover_20260818.py`, which shares it with
    `ema_crossover_20260821.py` and `ema_trend_filter.py`, by the same
    convention that duplicates `_wilder` and `_atr` across this directory:
    strategy modules are loaded from a file path and are deliberately
    self-contained. `tests/test_risk_params.py` runs the older copies on
    identical arrays in both directions and requires identical output, and
    `tests/test_t3_braid_scalp_20260823.py` holds THIS copy to the same
    standard against `sma_momentum_crossover_20260818._walk` — do not
    "improve" one alone.

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
def _validate(t3_period: int, t3_vfactor: float, braid_fast: int,
              braid_slow: int, stiffness_period: int,
              stiffness_threshold: float, sl_atr_mult: float,
              tp_atr_mult: float | None, trailing: bool,
              use_t3_filter: bool = True, use_braid_filter: bool = True,
              use_stiffness_filter: bool = True,
              use_news_filter: bool = False) -> None:
    """
    Reject parameter sets that do not describe this strategy.

    Every rejection here is recorded by `backtest.scan` as REJECTED and counted
    in the grid, so a combination the strategy refuses shrinks neither the
    reported search nor the honesty of `variants_tested`.

    The bounded-risk clauses at the end are the request's Layer 4. Read the
    limits of what they can promise in item 4 of "WHAT THIS MODULE DOES NOT DO"
    before quoting them as a cost check — they are not one.
    """
    if t3_period < 2:
        # A one-period EMA is the close itself, so a "T3(1)" is six copies of
        # the price recombined: the baseline would equal the close, `close >
        # t3` would be a coin flip on floating-point noise, and the layer would
        # pass roughly half the bars while appearing to be on.
        raise ValueError(f"t3_period must be >= 2; got {t3_period}")
    if t3_vfactor is None or not np.isfinite(float(t3_vfactor)):
        raise ValueError(
            f"t3_vfactor must be a finite number; got {t3_vfactor!r}")
    if not 0.0 <= float(t3_vfactor) <= 1.0:
        # Tillson's volume factor is defined on [0, 1] — 0 is a triple EMA and
        # 1 is the maximally responsive recombination. Outside it the cubic
        # coefficients grow without bound and the "average" overshoots the
        # price by multiples, which is not a slower or faster baseline but a
        # different object wearing the same name.
        raise ValueError(
            f"t3_vfactor must be within 0-1, the range Tillson's volume "
            f"factor is defined on; got {t3_vfactor!r}")
    if braid_fast < 1 or braid_slow < 2:
        raise ValueError(
            f"braid periods must be >= 1 (fast) and >= 2 (slow); got "
            f"fast={braid_fast}, slow={braid_slow}")
    if braid_fast >= braid_slow:
        # Not a stylistic objection. With the two equal the histogram is
        # identically zero, so neither `hist > 0` nor `hist < 0` ever holds and
        # the momentum layer silently blocks every trade; inverted, the
        # histogram changes sign and a "positive, green" reading is reported
        # under this module's name for a market whose short horizon is BELOW
        # its long one. Either way the curve looks plausible and describes a
        # strategy nobody specified.
        raise ValueError(
            f"braid_fast must be < braid_slow; got {braid_fast} >= "
            f"{braid_slow}")
    if stiffness_period < 2:
        # With a one-bar window the index takes only 0 or 100, so any threshold
        # between them becomes "did the last close tick up", which is not a
        # persistence measure at all.
        raise ValueError(
            f"stiffness_period must be >= 2; got {stiffness_period}")
    if stiffness_threshold is None or not np.isfinite(
            float(stiffness_threshold)):
        raise ValueError(
            f"stiffness_threshold must be a finite number; got "
            f"{stiffness_threshold!r}")
    if not 0.0 <= float(stiffness_threshold) <= 100.0:
        # The index is a percentage of a window and is bounded 0-100 by
        # construction. A threshold outside that is not a strict filter, it is
        # a filter that can never bind or can never pass, and either would run
        # as a silently different strategy.
        raise ValueError(
            f"stiffness_threshold must be within 0-100, the range the index "
            f"can take; got {stiffness_threshold!r}")

    # Truthiness would silently accept "false" (a non-empty string, so True)
    # and 0.0. It bites hardest on the toggles: `--param use_braid_filter=false`
    # passed as the STRING "false" would apply the filter while the
    # leaderboard's params column said it was off. Every flag is checked rather
    # than coerced.
    for name, flag in (("use_t3_filter", use_t3_filter),
                       ("use_braid_filter", use_braid_filter),
                       ("use_stiffness_filter", use_stiffness_filter),
                       ("use_news_filter", use_news_filter),
                       ("trailing", trailing)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be a bool; got {flag!r}")

    if not (use_t3_filter or use_braid_filter or use_stiffness_filter):
        # With all three off there is no directional condition left anywhere in
        # the module: both the long and the short conjunction are all-True, so
        # every bar carries a long AND a short candidate, and the walk's
        # ambiguous-bar rule takes NEITHER. The run would complete, report zero
        # trades and look like a strategy with no edge rather than like a
        # configuration that cannot express one. Refused here so it reads as
        # REJECTED in the scan output instead. The null this idea has to beat
        # is a single layer plus the stop, not no layers at all.
        raise ValueError(
            "at least one of use_t3_filter, use_braid_filter and "
            "use_stiffness_filter must be True; with all three off the "
            "strategy has no directional condition, both sides trigger on "
            "every bar, and the walk takes neither")

    # ---- Layer 4: bounded risk. A sanity floor, NOT a cost check. ----
    if sl_atr_mult is None or not np.isfinite(float(sl_atr_mult)):
        # The stop is not optional, and in this module it is load-bearing: with
        # no signal exit and no session flatten, a position with no stop and no
        # target would never be closed at all.
        raise ValueError(
            f"sl_atr_mult must be a finite number > 0; got {sl_atr_mult!r}")
    if not MIN_STOP_ATR_MULT <= float(sl_atr_mult) <= MAX_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be within {MIN_STOP_ATR_MULT}-"
            f"{MAX_STOP_ATR_MULT} x ATR{ATR_PERIOD}; got {sl_atr_mult!r}. "
            f"Below the floor the stop sits inside the bar noise the ATR is "
            f"measuring and is breached by the modelled slippage rather than "
            f"by the market.")
    if tp_atr_mult is not None:
        # None is the no-take-profit configuration and is accepted.
        if not np.isfinite(float(tp_atr_mult)) or float(tp_atr_mult) <= 0:
            # A target at or below the fill price would be breached by the fill
            # bar itself, reporting an instant win on every entry.
            raise ValueError(
                f"tp_atr_mult must be > 0, or None for no take-profit; got "
                f"{tp_atr_mult!r}")
        if float(tp_atr_mult) > MAX_TARGET_ATR_MULT:
            raise ValueError(
                f"tp_atr_mult must be <= {MAX_TARGET_ATR_MULT} x "
                f"ATR{ATR_PERIOD}, or None for no take-profit; got "
                f"{tp_atr_mult!r}")
        rr = float(tp_atr_mult) / float(sl_atr_mult)
        if rr < MIN_REWARD_RISK:
            raise ValueError(
                f"tp_atr_mult / sl_atr_mult must be >= {MIN_REWARD_RISK}; got "
                f"{tp_atr_mult!r} / {sl_atr_mult!r} = {rr:.3f}. A target "
                f"closer than half the stop needs better than a 2-in-3 win "
                f"rate to break even BEFORE the round turn is charged, and "
                f"this strategy pays one on every scalp.")


def _series(bars: pd.DataFrame, t3_period: int, t3_vfactor: float,
            braid_fast: int, braid_slow: int,
            stiffness_period: int) -> dict:
    """
    The shared calculation behind `signal_fn`, `indicators` and the walk.

    One place, so the T3 line the tear sheet draws is the array the entry was
    gated on and the stop the report shows is the level the exit was taken
    from. Two call sites computing this separately would be free to drift apart
    with nothing raising.
    """
    close = bars["close"]
    t3 = _t3(close, t3_period, t3_vfactor)
    braid_hist, braid_sig = _braid(close, braid_fast, braid_slow)
    stiff_up, stiff_down = _stiffness(close, stiffness_period)
    return {
        "t3": t3,
        # The slope as a first difference of the T3 itself. `> 0` is "rising /
        # green" and `< 0` is "falling / red"; a flat T3 is NEITHER, so an
        # exactly unchanged baseline permits no trade on either side. Written
        # as two explicit comparisons rather than one and its negation for that
        # reason — `~(slope > 0)` would call a dead-flat baseline "falling".
        "t3_slope": t3.diff(),
        "braid_hist": braid_hist,
        "braid_signal": braid_sig,
        "stiff_up": stiff_up,
        "stiff_down": stiff_down,
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

    if "ts" not in bars.columns and not isinstance(bars.index,
                                                   pd.DatetimeIndex):
        raise ValueError(
            "use_news_filter=True needs a `ts` column or a DatetimeIndex to "
            "place each bar against the release calendar; this frame has "
            "neither. The engine always supplies one — a frame without it is "
            "a fixture, and filtering it against a calendar would be "
            "filtering nothing.")

    kept_long, kept_short, _info = apply_entry_filters(
        pd.Series(_bar_timestamps(bars)), long_ok, short_ok,
        news_filter=True,
        news_window_minutes=NEWS_WINDOW_MINUTES)
    return (np.asarray(kept_long, dtype=bool),
            np.asarray(kept_short, dtype=bool))


def _signal_arrays(bars: pd.DataFrame, t3_period: int, t3_vfactor: float,
                   braid_fast: int, braid_slow: int, stiffness_period: int,
                   stiffness_threshold: float, sl_atr_mult: float,
                   tp_atr_mult: float | None, trailing: bool,
                   use_t3_filter: bool = True, use_braid_filter: bool = True,
                   use_stiffness_filter: bool = True,
                   use_news_filter: bool = False) -> tuple:
    """
    Everything the walk produces, from one place.

    `signal_fn` and `indicators` both go through here, so the T3 line a reader
    sees the price cross is the array the entry came from and the stop line
    they see breached is the array the exit was taken from. Two call sites
    computing this separately would be free to drift apart with nothing
    raising.
    """
    s = _series(bars, t3_period, t3_vfactor, braid_fast, braid_slow,
                stiffness_period)
    close = bars["close"]

    # `ready` is assembled FROM THE ACTIVE FILTERS ONLY, and that is the part
    # of the toggles that is easy to get wrong.
    #
    # Requiring every series unconditionally would make a toggled-off filter
    # cost its warm-up anyway: `use_t3_filter=False` would still wait the six
    # chained EMAs for a baseline nothing reads, so the "no T3" cell would be
    # scored on a shorter history than the strategy it is meant to represent —
    # and the comparison the toggles exist to enable would be between different
    # samples. Every comparison against NaN is False, so the symptom would be
    # silently missing early trades rather than an error.
    #
    # ATR is unconditional whatever the filters say: it sets the stop distance
    # and the target, so an entry taken before ATR exists would have no stop.
    ready = s["atr"].notna()
    if use_t3_filter:
        ready &= s["t3"].notna() & s["t3_slope"].notna()
    if use_braid_filter:
        ready &= s["braid_hist"].notna() & s["braid_signal"].notna()
    if use_stiffness_filter:
        ready &= s["stiff_up"].notna() & s["stiff_down"].notna()

    # A DISABLED FILTER IS ALL-TRUE, NOT ALL-FALSE. `_on` returns a True Series
    # when its toggle is off, so the conjunction below is written once and
    # reads the same whichever filters are active. Writing it as `cond if flag
    # else <omit>` instead would need a different expression per combination,
    # and each combination would be another chance for the long and short arms
    # to stop mirroring each other.
    def _on(condition: pd.Series, flag: bool) -> pd.Series:
        return condition if flag else pd.Series(True, index=bars.index)

    # Layer 1 — the T3 baseline. Both halves of the request's rule: the close
    # on the correct side AND the baseline itself moving that way. Each arm is
    # an explicit comparison rather than the negation of its opposite, because
    # `~(close > t3)` is true wherever the close merely fails to be above the
    # baseline — which includes exact equality and every warm-up bar — so a
    # short arm written as a negation would fire on ties and on bars where the
    # series does not exist.
    long_t3 = _on((close > s["t3"]) & (s["t3_slope"] > 0), use_t3_filter)
    short_t3 = _on((close < s["t3"]) & (s["t3_slope"] < 0), use_t3_filter)

    # Layer 2 — the Braid histogram. Positive AND above its own smoothing; the
    # two are different statements and the request requires both.
    long_braid = _on((s["braid_hist"] > 0)
                     & (s["braid_hist"] > s["braid_signal"]),
                     use_braid_filter)
    short_braid = _on((s["braid_hist"] < 0)
                      & (s["braid_hist"] < s["braid_signal"]),
                      use_braid_filter)

    # Layer 3 — the Stiffness Index, read in the direction of the candidate.
    # `>=` as the request specifies, and the two sides read DIFFERENT series
    # rather than one series and its complement — see `_stiffness`.
    thresh = float(stiffness_threshold)
    long_stiff = _on(s["stiff_up"] >= thresh, use_stiffness_filter)
    short_stiff = _on(s["stiff_down"] >= thresh, use_stiffness_filter)

    long_state = long_t3 & long_braid & long_stiff & ready
    short_state = short_t3 & short_braid & short_stiff & ready

    # THE RISING EDGE OF THE CONFLUENCE, not the confluence itself — deviation
    # 1 in the module docstring. `shift(1, fill_value=False)` looks one bar
    # BACKWARD, so bar i's candidate depends on bar i-1 and nothing later.
    #
    # `prev_ready` is what keeps the first fully-warm bar from registering as
    # an edge: `long_state` is False through warm-up because every comparison
    # against NaN is False, so at the first ready bar it can flip False->True
    # purely because the series came into existence. That is a fact about the
    # warm-up, not about price. `fill_value=False` keeps bar 0 out for the same
    # reason.
    prev_ready = ready.shift(1, fill_value=False)
    long_trigger = (long_state & ~long_state.shift(1, fill_value=False)
                    & prev_ready)
    short_trigger = (short_state & ~short_state.shift(1, fill_value=False)
                     & prev_ready)

    long_entry_ok = long_trigger.to_numpy(dtype=bool)
    short_entry_ok = short_trigger.to_numpy(dtype=bool)

    # The PERMITTED-BAR states and the TRIGGERS both travel back on `s`, and
    # keeping them apart is not bookkeeping — they answer different questions
    # and only one of them behaves the way a reader expects.
    #
    # The states are MONOTONE in the layers: switching a filter on can only
    # shrink the set of bars the strategy is permitted to trade on. The
    # TRIGGERS ARE NOT. The entry is the rising edge of the conjunction, so
    # removing bars from the middle of one permitted stretch splits it into two
    # stretches and creates a SECOND trigger where there had been one — turning
    # a filter on can leave more candidate entries than it found. Measured on
    # the suite's own fixture, enabling the T3 layer took the long candidates
    # from 134 to 143.
    #
    # This is the same trap CLAUDE.md records for `ema_trend_filter` one level
    # deeper: there a filter subtracts CANDIDATES but not necessarily trades,
    # because the walk holds one position at a time; here it does not reliably
    # subtract candidates either. NEVER use a count — of trades OR of triggers
    # — to decide whether a layer is wired. Compare the permitted states, which
    # is what `tests/test_t3_braid_scalp_20260823.py` does.
    s["long_state"] = long_state
    s["short_state"] = short_state
    s["long_trigger"] = long_trigger
    s["short_trigger"] = short_trigger

    # The news veto lands on CANDIDATES, before the walk, so a blocked trigger
    # leaves the strategy flat and free to take a later one. See the module
    # docstring for why that differs from the engine's own `--news-filter`,
    # which acts on the walk's output.
    if use_news_filter:
        long_entry_ok, short_entry_ok = _news_suppress(
            bars, long_entry_ok, short_entry_ok)

    # No signal exit, on either side. The request's Layer 4 enumerates the
    # exits and lists only the stop and the target — a position is deliberately
    # held through the confluence breaking and through the T3 slope rolling
    # over. The arrays exist because `_walk` keeps the shared kernel's
    # signature; both are all-False, so neither `sig_exit[i]` branch is taken.
    sig_exit = np.zeros(len(bars), dtype=bool)
    # Nor is there a session flatten. `flat_bar` all-False is what "positions
    # are carried overnight" looks like in this kernel, and for a module called
    # a scalp it is a modelled exposure rather than an omission — see item 2 of
    # "WHAT THIS MODULE DOES NOT DO".
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
        # This is the request's "fill degenerate states with safe defaults" —
        # see `_atr` for why a genuinely zero ATR is left as zero.
        np.nan_to_num(s["atr"].to_numpy(dtype=float), nan=0.0),
        flat_bar,
        float(sl_atr_mult),
        _tp_distance_mult(tp_atr_mult),
        bool(trailing),
    )
    return s, entries, exits, s_entries, s_exits, stop, target


def signal_fn(bars: pd.DataFrame,
              t3_period: int = 5,
              t3_vfactor: float = 0.7,
              braid_fast: int = 3,
              braid_slow: int = 7,
              stiffness_period: int = 20,
              stiffness_threshold: float = 35.0,
              use_t3_filter: bool = True,
              use_braid_filter: bool = True,
              use_stiffness_filter: bool = True,
              use_news_filter: bool = False,
              sl_atr_mult: float = 1.5,
              tp_atr_mult: float | None = 2.0,
              trailing: bool = False) -> tuple[pd.Series, pd.Series,
                                               pd.Series, pd.Series]:
    """
    Take the scalp only where the T3 baseline, the Braid histogram and the
    Stiffness Index all agree; exit on the stop or the target. Both directions.

    Returns the FOUR-MASK form of the strategy contract:

        (long_entries, long_exits, short_entries, short_exits)

    all boolean Series on `bars.index`. `backtest.engine.unpack_signals`
    accepts this alongside the older two-mask long-only form; returning a
    three-tuple or a bare Series raises there rather than silently losing the
    short side.

    The entry is a bar-level EVENT — the rising edge of the full confluence —
    so a long run of bars satisfying every layer produces ONE signal rather
    than one per bar. The walk enters only when flat, so a second trigger while
    a position is open is ignored rather than pyramided, and a short trigger
    arriving while long is ignored rather than reversing the position.

    Every comparison at bar i uses only bars <= i. The engine then fills at bar
    i+1's open, so nothing here can see a price it would not have had. That is
    the request's `signal_delay=1`: it is the engine's contract, not a
    parameter this module sets, and there is no code path here that could fill
    on the signal bar.
    """
    _validate(t3_period, t3_vfactor, braid_fast, braid_slow, stiffness_period,
              stiffness_threshold, sl_atr_mult, tp_atr_mult, trailing,
              use_t3_filter, use_braid_filter, use_stiffness_filter,
              use_news_filter)

    _s, entries, exits, s_entries, s_exits, _stop, _target = _signal_arrays(
        bars, t3_period, t3_vfactor, braid_fast, braid_slow, stiffness_period,
        stiffness_threshold, sl_atr_mult, tp_atr_mult, trailing,
        use_t3_filter, use_braid_filter, use_stiffness_filter, use_news_filter)

    return (pd.Series(entries, index=bars.index),
            pd.Series(exits, index=bars.index),
            pd.Series(s_entries, index=bars.index),
            pd.Series(s_exits, index=bars.index))


def indicators(bars: pd.DataFrame,
               t3_period: int = 5,
               t3_vfactor: float = 0.7,
               braid_fast: int = 3,
               braid_slow: int = 7,
               stiffness_period: int = 20,
               stiffness_threshold: float = 35.0,
               use_t3_filter: bool = True,
               use_braid_filter: bool = True,
               use_stiffness_filter: bool = True,
               use_news_filter: bool = False,
               sl_atr_mult: float = 1.5,
               tp_atr_mult: float | None = 2.0,
               trailing: bool = False) -> dict[str, pd.Series]:
    """
    The price-scale series, for the tear sheet to draw over the candles.

    Computed HERE, the same way and from the same columns `signal_fn` reads, so
    the T3 cross a reader sees is the array the entry was gated on. A second
    implementation living in the report would be free to disagree with this one
    — a chart showing the baseline crossed a bar away from where the trade
    fired, with nothing raising.

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

    THE BRAID HISTOGRAM AND THE STIFFNESS INDEX ARE DELIBERATELY NOT RETURNED,
    which means Layers 2 and 3 are the entry conditions a reader cannot see on
    the chart. That is a real gap and the alternative is worse: the inspector
    draws these on the PRICE axis. The Braid histogram oscillates around zero
    in price units, so on a 20,000-point contract it would be a flat line along
    the bottom of the panel; the Stiffness Index is bounded 0-100, so it would
    be drawn at prices nothing ever traded at. Rescaling either onto the price
    axis would put a line on the chart at levels that are not levels. Both
    reach the reader through the strategy card and through `ml_features`; the
    T3 and the risk levels are the ones that live on the price axis, and they
    are here.

    The take-profit line is OMITTED ENTIRELY when `tp_atr_mult` is None, and
    the T3 line when its toggle is off. An all-NaN series would render as an
    empty legend entry, which reads as a target that exists and never got close
    — the opposite of the truth; and drawing a baseline the entries did not
    respect makes the trades that cross it look like bugs rather than like the
    strategy that was actually run.

    Warm-up stays NaN rather than drawing the baseline flat through the first
    bars.
    """
    _validate(t3_period, t3_vfactor, braid_fast, braid_slow, stiffness_period,
              stiffness_threshold, sl_atr_mult, tp_atr_mult, trailing,
              use_t3_filter, use_braid_filter, use_stiffness_filter,
              use_news_filter)

    s, _e, _x, _se, _sx, stop, target = _signal_arrays(
        bars, t3_period, t3_vfactor, braid_fast, braid_slow, stiffness_period,
        stiffness_threshold, sl_atr_mult, tp_atr_mult, trailing,
        use_t3_filter, use_braid_filter, use_stiffness_filter, use_news_filter)

    kind = "Trailing" if trailing else "Fixed"
    # The stop is always drawn — it is the exit, and it cannot be switched off.
    out = {
        f"{kind} Stop ({sl_atr_mult}xATR{ATR_PERIOD})":
            pd.Series(stop, index=bars.index),
    }
    if use_t3_filter:
        out[f"Tillson T3 ({t3_period}, v={t3_vfactor})"] = s["t3"]
    if tp_atr_mult is not None:
        out[f"Take Profit ({tp_atr_mult}xATR{ATR_PERIOD})"] = pd.Series(
            target, index=bars.index)
    return out


def ml_features(bars: pd.DataFrame,
                t3_period: int = 5,
                t3_vfactor: float = 0.7,
                braid_fast: int = 3,
                braid_slow: int = 7,
                stiffness_period: int = 20,
                **_ignored) -> pd.DataFrame:
    """
    Version B's feature matrix: the five columns the request names.

    Read by `agents.tier3_workers.load_strategy` and handed to
    `apply_ml_signal_filter` in place of the shared `causal_features` default.
    Declaring it here is the point: the shared default carries seven columns
    (ATR, volume, RSI, hour, minute and two return horizons), four of which
    this strategy's hypothesis says nothing about. A classifier vetoing THESE
    entries should be reading the state THIS strategy is a claim about — is the
    baseline turning, is momentum widening, are the closes persistent, is
    volatility expanding, and when in the session is it.

    Columns, in `ML_FEATURES` order:

        t3_slope    The first difference of the Tillson T3 — the request's "T3
                    slope value". Under the default `use_t3_filter=True` every
                    surviving long candidate already has this above zero, so
                    what the column carries is HOW STEEPLY the baseline was
                    turning, which is the part the veto can still act on.
        braid_hist  The Braid histogram, the same array Layer 2 gates on. Its
                    HEIGHT, as the request asks, not its sign: the sign is
                    already spent as an entry condition.
        stiffness   The UP-direction Stiffness Index, 0-100. One column rather
                    than two because the down index is not its complement and
                    including both would hand the model the flat-bar count as a
                    third implicit feature. `apply_ml_signal_filter` fits one
                    classifier PER SIDE, so the short model reads low values as
                    favourable and the long model reads high ones — each learns
                    its own direction from the same column.
        atr_norm    ATR(14) / close — the request's "normalized rolling ATR".
                    Normalised because the classifier is fitted across a
                    16-year history in which the contract's price level changes
                    by a multiple: a raw ATR in points would let the model
                    learn "2015" instead of "volatile".
        hour_et     Hour of day in America/New_York, 0-23. A NAMED ZONE, not a
                    UTC hour and not a fixed offset: the CME session keeps its
                    local clock across the DST changeover, so a UTC hour smears
                    one session hour across two values twice a year and the
                    feature would mean a different thing in March than in
                    October. This is the one column that is a property of the
                    CLOCK rather than of price, and on an intraday scalp it is
                    the one most likely to carry a real effect — the open and
                    the close are not the same market as 02:00.

    THE FEATURE PERIODS TRACK THE BOUND PARAMETERS, and that is a deliberate
    departure from `sma_momentum_crossover_20260818`, which pins its feature
    periods and ignores `**_params`. Here they cannot be pinned honestly: three
    of these five columns ARE the entry conditions, and a `t3_slope` computed
    at period 5 while the sweep is testing period 14 would be a model vetoing
    entries on a baseline the strategy is not using. The cost is real and worth
    stating: every grid cell that moves `t3_period` gets a differently-shaped
    model, so `--scan` is searching over classifiers as well as over
    strategies, and two cells' Version B results are less comparable than their
    Version A results are. `stiffness_threshold`, `sl_atr_mult`, `tp_atr_mult`
    and the toggles do NOT enter here and are absorbed by `**_ignored` — none
    of them changes what a feature MEASURES, only which candidates survive to
    be scored.

    STRICTLY CAUSAL, and worth being explicit about because this matrix is
    fitted on. A value at row i is a function of bars 0..i only: no
    `shift(-k)`, no centred window, no reversed slice, and nothing computed off
    a full-sample statistic. That last one is the trap a shift-based audit does
    not catch — a scaler fitted on the whole frame leaks the test period's
    distribution into the training rows — which is why nothing here is
    standardised against a global mean.

    Using bar i's close to decide a signal on bar i is legitimate: the engine
    fills at bar i+1's open, never on the signal bar. That one-bar gap is what
    makes these features tradeable rather than clairvoyant.

    NaN warm-up rows are left as NaN. `HistGradientBoostingClassifier` consumes
    them natively, and filling them with a column mean would import a
    full-sample statistic into exactly the rows that have no history.
    """
    s = _series(bars, t3_period, t3_vfactor, braid_fast, braid_slow,
                stiffness_period)
    close = bars["close"].astype(float)
    # `_bar_timestamps` returns a tz-AWARE UTC index, so this conversion is
    # defined. A naive index would raise here under pandas 3.0 rather than
    # being silently read as if it were already ET.
    hour_et = _bar_timestamps(bars).tz_convert(SESSION_TZ).hour

    out = pd.DataFrame({
        "t3_slope": s["t3_slope"].to_numpy(dtype=float),
        "braid_hist": s["braid_hist"].to_numpy(dtype=float),
        "stiffness": s["stiff_up"].to_numpy(dtype=float),
        # `close.where(close != 0)` makes a zero close NaN rather than an
        # infinity: a bar with a zero price is bad data, and the classifier
        # consumes a NaN natively while an inf propagates.
        "atr_norm": (s["atr"] / close.where(close != 0)).to_numpy(dtype=float),
        "hour_et": np.asarray(hour_et, dtype=float),
    }, index=bars.index)
    return out[ML_FEATURES]


def make_signal_fn(t3_period: int = 5,
                   t3_vfactor: float = 0.7,
                   braid_fast: int = 3,
                   braid_slow: int = 7,
                   stiffness_period: int = 20,
                   stiffness_threshold: float = 35.0,
                   use_t3_filter: bool = True,
                   use_braid_filter: bool = True,
                   use_stiffness_filter: bool = True,
                   use_news_filter: bool = False,
                   sl_atr_mult: float = 1.5,
                   tp_atr_mult: float | None = 2.0,
                   trailing: bool = False):
    """
    Bind parameters for `agents.tier3_workers.load_strategy`.

    The loader prefers this factory when params are supplied; the engine itself
    only ever calls the bound `signal_fn(bars)`. Validation runs HERE as well
    as inside `signal_fn`, so `--scan` records a bad combination as REJECTED at
    bind time rather than discovering it one symbol into the sweep.
    """
    _validate(t3_period, t3_vfactor, braid_fast, braid_slow, stiffness_period,
              stiffness_threshold, sl_atr_mult, tp_atr_mult, trailing,
              use_t3_filter, use_braid_filter, use_stiffness_filter,
              use_news_filter)

    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series,
                                            pd.Series, pd.Series]:
        return signal_fn(bars,
                         t3_period=t3_period,
                         t3_vfactor=t3_vfactor,
                         braid_fast=braid_fast,
                         braid_slow=braid_slow,
                         stiffness_period=stiffness_period,
                         stiffness_threshold=stiffness_threshold,
                         use_t3_filter=use_t3_filter,
                         use_braid_filter=use_braid_filter,
                         use_stiffness_filter=use_stiffness_filter,
                         use_news_filter=use_news_filter,
                         sl_atr_mult=sl_atr_mult,
                         tp_atr_mult=tp_atr_mult,
                         trailing=trailing)

    return _bound
