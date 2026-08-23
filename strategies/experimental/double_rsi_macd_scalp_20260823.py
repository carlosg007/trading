"""
Double RSI + MACD Scalp — a trend-continuation scalp that buys short-horizon
exhaustion INSIDE a trend the slow RSI already confirms, times the re-entry on
the fast RSI turning back up through the slow RSI or the 50 centerline, and
requires the MACD histogram to be expanding in the same direction before it
takes the trade.

Location:  ~/src/trading/strategies/experimental/double_rsi_macd_scalp_20260823.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding — do not edit this block to match it.

    1. STRATEGY SPECIFICATION & ARCHITECTURAL CONTRACT
       Strategy Identifier: "double_rsi_macd_scalp" (pinned in the metadata and
           the LOGIC dict for bt-stage1 logging and JSON pipeline tracking).
       Module Path: strategies/experimental/double_rsi_macd_scalp_20260823.py
       Category: Multi-Timeframe Momentum / Mean-Reverting Scalp
       Target Market Regimes: Q1 (Low Vol / Trending) & Q2 (High Vol /
           Trending Pullbacks)
       Portfolio Basket & Asset Correlation Tag:
           Portfolio Group: "Index_Scalp_Momentum_Basket"
           Correlation Profile: Core Index Futures (MNQ, MES); exploits
           short-term liquidity pullbacks aligned with macro trend momentum.
       Economic & Quantitative Premise:
           Combines Larry Connors' short-horizon RSI exhaustion model (Connors
           & Alvarez, 2009) with dual-speed momentum confirmation (Wilder,
           1978; Appel, 1979). In strong directional trends, short-term
           pullbacks (Fast RSI) frequently trigger false counter-trend signals.
           Requiring the Slow RSI to hold the dominant trend side of the 50
           centerline while the Fast RSI mean-reverts from extreme exhaustion
           levels (sub-30/super-70) and re-crosses the Slow RSI — confirmed by
           MACD histogram alignment and institutional day-of-week volume
           expansion — isolates high-expectancy trend re-entries while
           filtering choppy sideways noise.

    2. MULTI-LAYER ENTRY & EXIT CONFLUENCE LOGIC
       Layer 1 (Macro Trend & Day Filter):
           Long:  RSI(slow_window) > 50 AND (DayOfWeek in allowed_days IF
                  use_day_filter ELSE True)
           Short: RSI(slow_window) < 50 AND (DayOfWeek in allowed_days IF
                  use_day_filter ELSE True)
       Layer 2 (Primary Alpha Pullback Trigger):
           Long:  (RSI(fast)[1] <= fast_rsi_os_threshold) AND
                  (Cross_Above(RSI(fast), RSI(slow)) OR
                   Cross_Above(RSI(fast), 50))
           Short: (RSI(fast)[1] >= fast_rsi_ob_threshold) AND
                  (Cross_Below(RSI(fast), RSI(slow)) OR
                   Cross_Below(RSI(fast), 50))
       Layer 3 (MACD Histogram Confluence):
           Long:  MACD_Hist(12, 26, 9) > 0 AND MACD_Hist > MACD_Hist[1]
           Short: MACD_Hist(12, 26, 9) < 0 AND MACD_Hist < MACD_Hist[1]
       Layer 4 (Exits & Rebalancing):
           Primary Signal Exit: Fast RSI crosses over the opposite extreme
               threshold (>= 70 for Longs, <= 30 for Shorts) OR Close crosses
               below the 200 EMA.
           Volatility Risk Brackets: `sl_atr_mult`, `tp_atr_mult`, `trailing`.
       Modular Toggles: use_macro_trend, use_pullback_trigger, use_macd_filter,
           use_day_filter, use_news_filter

    3. PARAMETERS & VECTORBT PRO OPTIMIZATION GRID
       Defaults:   fast_rsi_window=5, slow_rsi_window=21,
                   fast_rsi_os_threshold=30.0, fast_rsi_ob_threshold=70.0,
                   allowed_days=[0, 1, 2], sl_atr_mult=1.5, tp_atr_mult=3.0,
                   trailing=False
       PARAM_GRID: fast_rsi_window        [2, 5, 7]
                   slow_rsi_window        [14, 21]
                   fast_rsi_os_threshold  [20.0, 30.0]
                   fast_rsi_ob_threshold  [70.0, 80.0]
                   sl_atr_mult            [1.0, 1.5, 2.0]
                   tp_atr_mult            [2.0, 3.0, None]
                   trailing               [False, True]

    4. DUAL-VERSION ARCHITECTURE
       Declarations: signal_fn, indicators, LOGIC, PARAM_GRID, make_signal_fn,
                     ml_features.
       VERSION A: pure rule-based, strict next-bar-open execution
                  (`signal_delay=1`).
       VERSION B: apply_ml_signal_filter(..., features=ml_features), model
                  sklearn.ensemble.HistGradientBoostingClassifier, causal
                  features = Fast RSI, Slow RSI, RSI spread (fast - slow),
                  MACD histogram, 14-period normalized ATR (ATR / Close),
                  20-bar volume Z-score, Day of Week, Hour of Day; binary
                  label 1 if trade net P&L after costs > 0 else 0; strictly a
                  causal veto on Version A's candidate entries.

    5. MICROSTRUCTURE & SAFETY REQUIREMENTS
       Strict next-bar-open execution (`signal_delay=1`), zero lookahead.
       Zero-range / degenerate rolling windows filled with 0.0 rather than
       propagating NaN. Pandas 3.0 compatible — no `.fillna(method=...)`, no
       positional datetime slicing. `tp_atr_mult=None` requires
       `trailing=True`, to prevent un-exited runner lockups. `use_news_filter`
       wired to `backtest.event_calendar`. Execution-agnostic: risk sizing and
       firm-level drawdown limits belong to the Portfolio Vol-Sizer and
       CrossTrade NAM.

    6. UNIT TESTING & VERIFICATION
       tests/test_double_rsi_macd_scalp_20260823.py — signal shape, alignment
       and dtypes; independent functional isolation of every modular toggle;
       risk-key integrity (`sl_atr_mult`, `tp_atr_mult`, `trailing`); causal
       feature verification.

EIGHT PLACES THE REQUEST WAS UNDER-SPECIFIED OR SELF-CONTRADICTORY
==================================================================
Written down rather than decided silently, because each one changes which
trades this module takes and none of them is recoverable from the equity curve.

1. THE REQUEST'S QUADRANT NUMBERS CONTRADICT THIS REPOSITORY'S ENCODING, AND
   THE NAMES ARE WHAT WAS KEPT. The request asks for "Q1 (Low Vol / Trending) &
   Q2 (High Vol / Trending Pullbacks)". In `mdlib/regimes.py` — the single
   place the quadrant encoding is written down, and the one Stage 1's
   designation, Gate R and the Discord cards all read — the numbering is:

       Q1  High Volatility / Trending
       Q2  High Volatility / Ranging
       Q3  Low Volatility / Trending
       Q4  Low Volatility / Ranging

   So the request's "Q1" names Q3 here, and its "Q2 (High Vol / Trending)"
   names Q1 — its Q2 is this repository's High-Volatility RANGING quadrant,
   which is the one environment the request's own premise ("filtering choppy
   sideways noise") says this strategy must not trade. Taking the NUMBERS
   literally would aim the strategy at chop. `TARGET_REGIMES` below is
   therefore declared as the two REGIME NAMES the request wrote out, and
   `TARGET_QUADRANTS` carries this repository's ids for those names — `Q3` and
   `Q1`, both Trending. The test suite pins both against `backtest.profiler`,
   so a future edit to either encoding fails loudly instead of quietly aiming
   this module at a different market.

   NEITHER IS AN INSTRUCTION TO ANY STAGE. Stage 1 DISCOVERS the home quadrant
   from the run (`backtest/profiler.py::designate`) and nothing in the pipeline
   reads a declaration made here. These constants are the hypothesis this
   module was written under, recorded so a Stage 1 result can be compared
   against it — a designated quadrant outside `TARGET_QUADRANTS` is a finding
   about the strategy, not a bug in the screen.

2. THE EXIT EXTREMES ARE PINNED AT 70 AND 30 AND ARE NOT THE ENTRY THRESHOLDS.
   The request writes Layer 2's levels as PARAMETER NAMES
   (`fast_rsi_os_threshold`, `fast_rsi_ob_threshold`) and Layer 4's as
   LITERAL NUMBERS ("(>= 70 for Longs, <= 30 for Shorts)"). They coincide at
   the defaults and diverge the moment the grid sweeps
   `fast_rsi_ob_threshold` to 80. Wiring the exit to the parameter would mean
   a LONG's profit-taking exit is governed by the SHORT's entry threshold —
   two different roles sharing one number, so a sweep intended to make the
   short entry more selective would also make every long hold longer, and the
   winning cell could not be attributed to either change. `EXIT_OB_LEVEL` and
   `EXIT_OS_LEVEL` are module constants for that reason.

3. THE 200-EMA EXIT IS MIRRORED FOR SHORTS. The request gives Layer 4 one
   clause for both sides: "OR Close crosses below 200 EMA". Read literally, a
   SHORT exits when the close breaks DOWN through the 200 EMA — which is the
   move a short is in the trade for, so the literal reading closes every
   winning short at the moment its thesis is confirmed and leaves the losing
   ones running to the stop. The clause is mirrored: longs exit on a close
   crossing BELOW the 200 EMA, shorts on a close crossing ABOVE it. This is
   the single largest deviation from the letter of the request in this module.

4. WITH THE PULLBACK TRIGGER OFF, THE ENTRY BECOMES A RISING EDGE. Layers 1
   and 3 are standing STATES ("RSI(slow) > 50", "MACD_Hist > 0"); Layer 2 is
   an EVENT (a cross). With Layer 2 on, the conjunction is already event-
   shaped and fires once per pullback, which is what the strategy describes.
   With `use_pullback_trigger=False` there is no event left anywhere in the
   stack, and a literal reading would re-arm on EVERY bar the two states hold
   — so one trend leg becomes a chain of stopped scalps whose count is set by
   the stop distance rather than by the signal. In that configuration the
   candidate is the RISING EDGE of the remaining conjunction: one trade per
   episode. The toggle therefore changes the SHAPE of the trigger, not merely
   how many survive, and `_signal_arrays` says so at the branch.

5. DAY OF WEEK IS THE CME SESSION DATE, NOT THE UTC DATE. The request says
   "DayOfWeek in allowed_days" without saying whose day. `backtest/engine.py`'s
   own `exclude_days` is keyed on the SESSION date — any bar at or after 18:00
   ET belongs to the next session — so a module keying its day filter on the
   UTC date would disagree with the engine's flag about which bars are Monday,
   and the two would be combined on the same run with nothing raising. This
   module calls `backtest.event_calendar.session_weekday`, the repository's
   only implementation, rather than writing a second one.

   THE DEFAULT KEEPS ONLY MON/TUE/WED. `allowed_days=[0, 1, 2]` is the
   request's default and it removes ~40% of the week's ENTRIES before any
   other layer runs. That is a large prior stated as a default, and it is a
   SELECTION on the in-sample week: the request's premise for it
   ("institutional day-of-week volume expansion") is not tested anywhere in
   this module. Run `--param use_day_filter=false` once at the winning cell
   before believing it.

6. `tp_atr_mult=None` REQUIRES `trailing=True`, AS THE REQUEST DEMANDS, AND
   THE GRID CONTAINS CELLS THAT VIOLATE IT. Section 5 makes the requirement
   explicit; section 3's grid crosses `tp_atr_mult: [2.0, 3.0, None]` with
   `trailing: [False, True]`, so 72 of its 432 cells are (no target, fixed
   stop). `_validate` REJECTS those rather than silently running them, which
   `backtest/scan.py` records as REJECTED and still counts in
   `variants_tested` — the search is reported at its true size. Stated
   plainly: this module has signal exits (Layer 4), so a no-target runner is
   not literally un-exitable the way it would be in a stop-and-target-only
   strategy; the requirement is enforced because the request states it, and
   the honest reading is that it removes the one configuration where a
   position can outlive both its RSI exit and its 200-EMA exit while a fixed
   stop sits unmoved far behind the price.

7. NO TIMEFRAME WAS SPECIFIED, AND THE MICROS ARE UNVERIFIED. `TIMEFRAME` is
   a DEFAULT chosen so `bt-run` has something to run — 5m, because the
   archetype is a scalp — and `--tf` overrides it; the request calls the
   strategy "Multi-Timeframe" in its category line but names no bar size.
   `SYMBOLS` is the request's MNQ/MES basket, and per CLAUDE.md's open-tasks
   list the micro contracts have NO definition data downloaded and remain
   UNVERIFIED in `backtest/specs.py`. A wrong multiplier silently scales every
   P&L figure for that symbol and the backtest still looks plausible. Verify
   the MNQ and MES specs before any number this module produces on them is
   quoted; `MES` (multiplier 5) and `MNQ` (multiplier 2) are what
   `backtest/specs.py` currently holds.

8. THE INDICATOR LENGTHS THE REQUEST FIXES ARE MODULE CONSTANTS, NOT
   PARAMETERS. MACD is pinned at (12, 26, 9) and the trend EMA at 200, exactly
   as written, and neither is in `PARAM_GRID` or the signature. Exposing them
   to `--param` would create axes somebody would eventually sweep, which turns
   "does the momentum confluence help" into a search over momentum
   confluences. `ATR_PERIOD` (14) is the request's own ML feature length and
   is used for the risk brackets too, so one ATR serves both.

THE CONFLUENCE IS VERY TIGHT, AND AT THE DEFAULT WINDOWS IT BARELY TRADES
========================================================================
Measured before the first real backtest, on a synthetic 20,000-bar path (a
Student-t(3) random walk with a slow cycle, so it has fatter tails than a
Gaussian and more of the sharp reversals this strategy is looking for), at the
declared defaults. The numbers are a property of that fixture and not of any
contract — what they establish is the SHAPE of the funnel, which the lake will
not change:

    Layer 2 alone (the pullback trigger)          647 long candidates
    + Layer 1 macro (slow RSI > 50)                68
    + Layer 3 MACD (positive and expanding)        11
    + Layer 1 day filter (Mon/Tue/Wed)              5

Five entries in 20,000 bars is roughly seventy 5-minute sessions per trade.
Gate 1 requires >= 100 trades and >= 30 per backtest year, so at
`fast_rsi_window=5` this strategy is at risk of failing on SAMPLE SIZE rather
than on edge — a distinction that matters because the two are fixed by
completely different work.

WHY THE MACRO FILTER IS THE STEP THAT CUTS DEEPEST (647 -> 68) is worth
understanding before anyone concludes the strategy is broken. Layer 2 requires
the FAST RSI to have been at an extreme on the previous bar. On most paths a
move violent enough to put a 5-period RSI under 30 also drags the 21-period RSI
under 50 — so the pullback that Layer 2 wants is, most of the time, already
deep enough to break the trend Layer 1 requires. The two layers are in genuine
tension, and that tension IS the strategy: it is asking for the rare pullback
that is sharp on the short horizon and invisible on the long one.

THE GRID'S `fast_rsi_window=2` CELL IS THE ONE THAT ACTUALLY TRADES. On the
same fixture it produced 56 long and 47 short entries against 5 and 4 at
period 5, and 1 and 2 at period 7 — an order of magnitude, from one axis. That
is not an accident of the fixture: Connors' original model is RSI(2), and the
two-period oscillator is fast enough to reach an extreme on a pullback shallow
enough for the slow RSI to survive. Expect the sweep to select it, and read
that selection as the grid finding the only cell with a usable sample rather
than as evidence that 2 is the optimal lookback.

WHAT TO DO WITH THIS, in order: run Stage 1 and read the TRADE COUNTS before
the profit factors; if the default cell places too few trades to measure,
that is the finding, and the honest responses are the `fast_rsi_window=2` cell,
a faster timeframe, or relaxing the request's Layer 2 to "the fast RSI was
oversold within the last N bars" rather than strictly on the previous one.
THE LAST OF THOSE IS A CHANGE TO THE SPECIFICATION AND IS NOT MADE HERE — the
request says `RSI(fast_window)[1]`, which is the previous bar, and rewriting a
rule because its trade count is inconvenient is how a specification becomes a
description of whatever produced the best equity curve.

WHAT THIS MODULE DOES NOT DO, AND WHY
=====================================
Stated here rather than approximated silently, because each one changes how a
number this module produces must be read.

1. THERE ARE NO STOP OR TARGET ORDERS. `backtest/engine.py` drives
   `vbt.Portfolio.from_signals` off boolean masks and fills them at the NEXT
   bar's open. A stop here is an exit SIGNAL detected on the bar that breaches
   it and filled one bar later, at whatever the next open happens to be — not
   a fill at the stop price. On a bar breaching both the stop and the target
   the modelled fill is neither level. ON A SCALP THIS IS NOT A ROUNDING
   ERROR: at 5m a 1.5 x ATR stop and one bar of slippage are the same order of
   magnitude, so a drawdown figure from this module is NOT bounded by
   `sl_atr_mult x ATR` and must never be quoted as though it were.

2. THERE IS NO SESSION FLATTEN. The request's Layer 4 enumerates the exits and
   a bell is not among them, so positions are carried overnight and over
   weekends. For a module called a scalp that is a real and unpriced exposure.
   The engine's `--flat-by-close` still composes on top and uses a FIXED UTC
   close, which is right in one half of the year and an hour off in the other.

3. THE DAY FILTER GATES ENTRIES ONLY, NEVER EXITS. A position opened on a
   Wednesday is exited on whatever day its exit condition arrives, including a
   Thursday. Filtering exits would express "hold through the release", which is
   the opposite of what a day filter is for.

4. "MULTI-TIMEFRAME" IS A CATEGORY LABEL, NOT AN IMPLEMENTATION. Nothing here
   reads a second bar size. The request's own rules are all computed on the
   run's timeframe; the "multi-timeframe" character is the fast/slow RSI pair,
   which is two HORIZONS on one series. A module that resampled internally
   would be a second aggregation implementation free to disagree with
   `mdlib.lake` about bar boundaries and the Sunday merge.

5. BOUNDED RISK IS A COARSE SANITY FLOOR, NOT A COST-ADJUSTED VIABILITY PROOF.
   `MIN_REWARD_RISK`, `MIN_STOP_ATR_MULT` and the outer bounds reject a target
   the fill bar would breach or a stop inside the tick noise. They cannot
   prove an edge survives costs: costs are per-contract, live in
   `backtest/specs.py` (which a strategy module does not read) and depend on
   the realised trade count, which does not exist until the run does. The real
   answer is Stage 4's cost drag as a share of GROSS profit
   (`backtest/verify_full.py`), and on a scalp it is the number to read before
   anything here is believed.

6. THE NEWS FILTER IS OFF BY DEFAULT. It needs a macro calendar to be
   meaningful, and `backtest/event_calendar.py` RAISES rather than returning an
   all-clear mask when its calendar does not cover the run's span. Check what a
   filtered run would actually use before trusting it:

       python3 backtest/event_calendar.py --start 2013-01-01 --end 2026-01-01

   Only NFP's rule-generated dates follow the real convention; CPI, PPI and
   FOMC anchors land in the right week and often the wrong day, and a
   30-minute window on the wrong day blocks a random half hour while leaving
   the release tradeable. `use_news_filter=True` against a RULE-provenance
   calendar is not a run that dodged the actual prints.

7. THE "INSTITUTIONAL VOLUME EXPANSION" IN THE PREMISE IS NOT AN ENTRY
   CONDITION ANYWHERE IN VERSION A. The request's Layer 1 implements it as a
   day-of-week list and nothing else — no volume series is read by any
   Version A rule. Volume reaches the strategy only through Version B, as the
   20-bar z-score feature. Read the premise accordingly.

THE NEWS FILTER AND THE SESSION WEEKDAY ARE IMPORTED, NOT REIMPLEMENTED
=======================================================================
Both come from `backtest.event_calendar`: `apply_entry_filters` for the macro
veto and `session_weekday` for the day filter. A second copy of either here
would be free to disagree with the engine's about which bars a release covers
or which session a Sunday-evening bar belongs to, and the two would be compared
by nobody.

THIS PUTS THE MODULE OUTSIDE `ALLOWED_IMPORTS`, DELIBERATELY, exactly as
`t3_braid_scalp_20260823` and `sma_momentum_crossover_20260818` are.
`agents.tier3_workers.ALLOWED_IMPORTS` permits a strategy module numpy, pandas,
math, vectorbtpro and numba and nothing else — but that allowlist governs
MODEL-GENERATED code, which is audited before execution; it does not run over
hand-written modules, which `load_strategy` imports from a file path. The
consequence, stated rather than discovered later: this module would be REJECTED
by the AST validator if it were ever passed through the synthesis path, and the
test suite pins that `backtest.event_calendar` is the ONLY module the validator
objects to, so a future edit reaching for `open`, `eval` or a network library
fails loudly instead of hiding behind an exception granted for something else.

The news veto is applied to the CANDIDATE triggers, before the position walk,
which is a different thing from the engine's `--news-filter` flag acting on the
walk's OUTPUT entries. Blocking a candidate leaves the strategy FLAT and
therefore free to take a later trigger it would otherwise have been holding
through, so the two orderings do not produce the same trade list. The engine's
flag still works and still composes — it will simply find fewer entries left to
remove. Do not read a trade COUNT to decide whether the filter bound: it
subtracts candidates, not trades.

VERSION B — THE ML FILTER
=========================
Version B is `agents.tier3_workers.apply_ml_signal_filter`, the repository's
shared expanding-window walk-forward: for a candidate signalled on bar `s` it
fits only on trades that had already CLOSED before `s` (`exit_idx < s`), refits
as the pool grows, and can only ever turn an entry OFF. It is a veto, exactly
as the request specifies, and a bidirectional strategy gets one classifier per
side so a short is never scored by a model trained on longs.

The request's model — `HistGradientBoostingClassifier` — is what the shared
filter already uses, and the request's label ("1 if Net P&L after costs > 0,
else 0") is the shared label verbatim. Nothing about Version B is
strategy-specific here except the feature matrix, which this module supplies
through the `ml_features` hook: the eight columns the request names, replacing
the shared seven-column `causal_features` default. See `ml_features`.

Every calculation is causal. Values at bar i use bars <= i only, warm-up stays
NaN and is never treated as a signal, and the engine then fills at bar i+1's
open — so nothing here can see a price it would not have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The identifier the request pins, and the one the pipeline logs. It is
# DELIBERATELY not the filename: `backtest/run.py` resolves `--strat` by
# FILENAME against SEARCH_DIRS, so the dated stem below is what the CLI needs
# while this string is what the request asked to see in the metadata and the
# LOGIC dict. Both are pinned by the test suite against the module's own path,
# so a rename cannot leave them disagreeing.
#
#     python3 backtest/run.py --strat double_rsi_macd_scalp_20260823 ...
STRATEGY_NAME = "double_rsi_macd_scalp"
STRATEGY_MODULE = "double_rsi_macd_scalp_20260823"

# The request's portfolio tags. Nothing in this repository reads them — no
# stage groups by basket and no correlation clustering is wired to a module
# constant — so they are a RECORD of the hypothesis, carried with the code so
# a reader knows which book this was written for, not an instruction.
PORTFOLIO_GROUP = "Index_Scalp_Momentum_Basket"
CORRELATION_PROFILE = (
    "Core Index Futures (MNQ, MES); exploits short-term liquidity pullbacks "
    "aligned with macro trend momentum.")

# The regimes this strategy is a claim about, BY NAME, spelled exactly as
# `backtest.profiler.REGIMES` spells them — see deviation 1 for why the
# request's own quadrant NUMBERS were not used. `TARGET_QUADRANTS` is this
# repository's id for each name, in the same order.
#
# Declarative only. Stage 1 discovers the home quadrant from the run and reads
# nothing here; a designated quadrant outside this pair is a finding about the
# strategy rather than a bug in the screen.
TARGET_REGIMES = ("Low Volatility / Trending", "High Volatility / Trending")
TARGET_QUADRANTS = ("Q3", "Q1")

# NEITHER OF THESE IS FIXED BY THE REQUEST'S RULES — see deviation 7. Defaults
# so `bt-run` has something to run, not a claim about where the edge is;
# `--tf` and `--symbols` override both. The micro specs are UNVERIFIED in
# `backtest/specs.py`.
TIMEFRAME = "5m"
SYMBOLS = ["MNQ", "MES"]

DEFAULT_PARAMS = {
    "fast_rsi_window": 5,
    "slow_rsi_window": 21,
    "fast_rsi_os_threshold": 30.0,
    "fast_rsi_ob_threshold": 70.0,
    "allowed_days": [0, 1, 2],
    "use_macro_trend": True,
    "use_pullback_trigger": True,
    "use_macd_filter": True,
    "use_day_filter": True,
    "use_news_filter": False,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "trailing": False,
}

# The search space `backtest/run.py --scan` and `backtest/scan.py` sweep. The
# request's grid, transcribed exactly: 3 x 2 x 2 x 2 x 3 x 3 x 2 = 432
# combinations, of which 72 — every (tp_atr_mult=None, trailing=False) cell —
# are REJECTED by `_validate` under the request's own section 5 rule. 360 are
# fitted.
#
# 432 IS MORE THAN TWICE THE 200-CELL BOUND `backtest/scan.py` WARNS AT, AND
# THAT WARNING IS THE POINT OF PRINTING IT. The winning Sharpe is the maximum
# of 360 draws from one sample of bars, and that maximum climbs with N whether
# or not anything in the market has changed. MULTIPLY IT BY THE TIMEFRAMES
# BEFORE QUOTING IT: `--tf 5m,15m,30m` is 360 fits PER timeframe PER contract.
# `variants_tested` carries the count onto every report and every leaderboard
# row for exactly this reason — a swept Sharpe read without N is not a
# measurement. The grid is the request's and is transcribed rather than
# trimmed; trimming it would be this module overruling the specification, and
# the honest response is to report the size, which the pipeline does.
#
# WHAT IS NOT SWEPT, and why each omission is a choice rather than an oversight:
#
#   MACD (12, 26, 9)      Pinned by the request as literal lengths. See
#                         deviation 8.
#   EMA_TREND_PERIOD      Pinned at 200 by the request, same reasoning.
#   ATR_PERIOD            The request's own ML feature length; the MULTIPLIERS
#                         are the part the grid opens.
#   EXIT_OB/OS_LEVEL      Constants, not parameters — see deviation 2.
#   allowed_days          A SET, not a scalar axis: sweeping it means sweeping
#                         the 31 non-empty subsets of the trading week, which
#                         is a search over calendars. The honest test of the
#                         day filter is the single out-of-band run
#                         `--param use_day_filter=false` at the winning cell.
#   the four layer        Fixed at their defaults. Opening them to
#   toggles               [True, False] would multiply the grid by sixteen and
#                         search over STRATEGIES rather than parameters — the
#                         all-off cell is not a variant of this idea, it is
#                         the null this idea has to beat (and `_validate`
#                         refuses it outright). Run those comparisons out of
#                         band, once, at the winning cell.
#   use_news_filter       Fixed OFF. It depends on an external calendar whose
#                         PROVENANCE changes what the filter means, so it is
#                         an operator decision rather than a search axis.
PARAM_GRID = {
    "fast_rsi_window": [2, 5, 7],
    "slow_rsi_window": [14, 21],
    "fast_rsi_os_threshold": [20.0, 30.0],
    "fast_rsi_ob_threshold": [70.0, 80.0],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, None],
    "trailing": [False, True],
}

# Appel's MACD, pinned at the request's lengths. Span EMAs, not Wilder's —
# the MACD family is defined on the span form and `_wilder` below is roughly
# half the speed at the same length, so using it here would produce a "MACD
# histogram" no chart package agrees with.
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# The request's Layer 4 trend baseline, pinned. A span EMA, for the same
# reason.
EMA_TREND_PERIOD = 200

# The RSI centerline Layer 1 splits on and Layer 2 offers as an alternative
# cross. 50 is the definition, not a tuning knob.
RSI_CENTERLINE = 50.0

# Layer 4's exit extremes, as LITERAL LEVELS rather than as the entry
# thresholds — see deviation 2. A long exits when the fast RSI crosses UP
# through EXIT_OB_LEVEL, a short when it crosses DOWN through EXIT_OS_LEVEL.
EXIT_OB_LEVEL = 70.0
EXIT_OS_LEVEL = 30.0

# Wilder's ATR period, and the request's "14-period normalized ATR" feature
# length. One ATR serves the risk brackets and the feature matrix, so the
# volatility the model reads is the volatility the stop was placed off.
ATR_PERIOD = 14

# The request's "20-bar Volume Z-Score" lookback. Version B only — no Version A
# rule reads volume at all (see "WHAT THIS MODULE DOES NOT DO" item 7).
VOLUME_Z_PERIOD = 20

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

# Version B's feature matrix, in column order — the eight the request names.
ML_FEATURES = ["rsi_fast", "rsi_slow", "rsi_spread", "macd_hist",
               "atr_norm", "volume_z", "day_of_week", "hour_et"]

# --------------------------------------------------------------------------
# Bounded risk. See "WHAT THIS MODULE DOES NOT DO" item 5: these are a sanity
# floor, not a cost-adjusted viability proof, and they are named constants so
# the bound a rejection cites is the bound a reader can find.
# --------------------------------------------------------------------------
# A target closer than half the stop needs better than a 2-in-3 win rate to
# break even BEFORE costs, and this is a scalp — the round turn is charged on
# every one of those trades. The request's own grid never reaches it (its
# tightest cell is tp=2.0 against sl=2.0, a ratio of 1.0), so this bound
# rejects a mistyped multiplier rather than a specified one.
MIN_REWARD_RISK = 0.5
# A stop tighter than a quarter of ATR sits inside the bar noise the ATR is
# measuring; with one tick of slippage each way modelled at the engine, it is
# breached by the spread rather than by the market.
MIN_STOP_ATR_MULT = 0.25
# Outer bounds, to catch a transposed or mistyped multiplier rather than to
# express a view. A 25 x ATR stop is not a stop.
MAX_STOP_ATR_MULT = 25.0
MAX_TARGET_ATR_MULT = 50.0

# The value a dead-flat RSI window takes. NOT 0.0 — see `_rsi`. The request's
# section 5 asks for degenerate rolling windows to be filled with 0.0 rather
# than propagating NaN, and for the volume z-score and the MACD histogram 0.0
# IS the true value of a flat window. For the RSI it is not: 0.0 is maximal
# exhaustion, the most bullish reading the oscillator has, so filling a market
# that did not move with 0.0 would manufacture long triggers out of silence.
# A window with no up moves and no down moves is neutral, and neutral is 50.
RSI_NEUTRAL = 50.0


# Plain-English description for the tear sheet's strategy card, written for
# whoever is deciding whether to trade this — not for whoever maintains the
# module. `{param}` slots are filled with the run's own bound parameters, so
# the card states the settings that actually ran rather than the defaults
# written here. The report never infers any of this from the signal arrays: a
# description guessed from the trades is a guess printed as a fact.
#
# `strategy` is the request's pinned identifier, carried in the LOGIC dict as
# asked. `agents.tier3_workers._describe_strategy` reads only `concept`,
# `entry` and `exit`, so the extra key is inert on the tear sheet and is here
# to be found by anything grepping the pipeline's JSON.
LOGIC = {
    "strategy": STRATEGY_NAME,
    "concept": "A pullback scalp that only ever trades WITH a trend it can "
               "already measure. Two RSIs of different speeds run on the same "
               "price: the slow one ({slow_rsi_window} bars) says which side "
               "the market is on — above 50 is an uptrend, below 50 a "
               "downtrend — and the fast one ({fast_rsi_window} bars) times "
               "the entry. The trade is the moment a shallow pullback INSIDE "
               "that trend gives up: the fast RSI has dipped to an extreme "
               "({fast_rsi_os_threshold} for a long, {fast_rsi_ob_threshold} "
               "for a short), then turns and crosses back up through the slow "
               "RSI or through the 50 line. A third check refuses the trade "
               "unless the MACD histogram is on the matching side of zero AND "
               "getting wider, which is the difference between momentum that "
               "is building and momentum that is being handed back — the "
               "population that stops out. The strategy is flat far more often "
               "than it is in, and by design: it is looking for the handful of "
               "bars a trend pauses and resumes, not for the trend itself.",
    "entry": "Go LONG when the slow RSI is above 50, the fast RSI was at or "
             "below {fast_rsi_os_threshold} on the previous bar and has now "
             "crossed back above either the slow RSI or the 50 line, and the "
             "MACD histogram (12, 26, 9) is positive and larger than it was on "
             "the previous bar. Go SHORT on the exact mirror: slow RSI below "
             "50, fast RSI at or above {fast_rsi_ob_threshold} on the previous "
             "bar and now crossing back below the slow RSI or 50, MACD "
             "histogram negative and getting more negative. Entries are taken "
             "only on Monday, Tuesday and Wednesday sessions when the day "
             "filter is on (currently: {allowed_days}, as weekday numbers with "
             "Monday=0). Every entry fills at the NEXT bar's open.",
    "exit": "A long is closed when the fast RSI crosses up through 70 (the "
            "pullback has become an extension), when the close crosses down "
            "through the 200-bar EMA (the trend it was trading has broken), or "
            "on its risk bracket — a stop {sl_atr_mult} x ATR(14) from the "
            "fill and a target {tp_atr_mult} x ATR(14) away, with the stop "
            "trailing behind the best price the trade has seen when trailing "
            "is on (currently: {trailing}). A short mirrors all of it: the "
            "fast RSI crossing down through 30, the close crossing up through "
            "the 200 EMA, or the same bracket inverted. Every exit fills at "
            "the NEXT bar's open, so a stop is filled at the open after the "
            "bar that breached it and NOT at the stop price.",
}


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    A conventional EMA, `alpha = 2 / (period + 1)`, NaN until `period` values.

    `min_periods=period` is the point: without it pandas seeds the average from
    the first bar, so a "200-period EMA" exists at bar 2 and every early signal
    is decided by the seeding rather than by price. Every comparison against
    NaN is False, so the symptom would be silently wrong early trades rather
    than an error.

    This is the span form, NOT Wilder's `alpha = 1/period`. Both appear in this
    module and they are different averages: `_wilder` below is used for the ATR
    and the RSI, where Wilder's is the definition, and this one for the MACD
    and the trend EMA, where Appel's and the moving-average convention is the
    span form. Using one where the other belongs produces an indicator no chart
    package agrees with, and the disagreement is a fraction of a bar's move —
    invisible except at a threshold.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


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


def _macd_hist(close: pd.Series) -> pd.Series:
    """
    Appel's MACD histogram at the request's pinned (12, 26, 9).

    `hist = (EMA12 - EMA26) - EMA9(EMA12 - EMA26)`. Positive means the fast
    average leads; ABOVE its own signal line means the lead is still widening.
    Layer 3 tests the sign and the first difference, which together are the
    "positive and expanding" the request asks for.

    Warm-up compounds: the MACD line needs `MACD_SLOW` closes and the signal
    line needs `MACD_SIGNAL` non-NaN values of the line, so the histogram is
    NaN for roughly `MACD_SLOW + MACD_SIGNAL - 1` bars.

    CAUSAL: an EMA is a recursion over past values only.
    """
    line = _ema(close, MACD_FAST) - _ema(close, MACD_SLOW)
    return line - _ema(line, MACD_SIGNAL)


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
    # the test suite pins that objection list, so a cosmetic `__name__` here
    # would spend the exception on an error message.
    raise ValueError(
        "bars needs a `ts` column or a DatetimeIndex; got an index of type "
        f"{type(bars.index)} and columns {list(bars.columns)}")


def _session_weekday(bars: pd.DataFrame) -> np.ndarray:
    """
    Monday=0 .. Sunday=6 on the CME SESSION date, not the UTC date.

    Delegates to `backtest.event_calendar.session_weekday`, the repository's
    only implementation and the one `BacktestConfig.exclude_days` is keyed on.
    A second copy here would be free to disagree with the engine's about which
    session a Sunday-evening bar belongs to, and a run combining this module's
    day filter with `--exclude-days` would then be applying two different
    calendars under one name.

    Imported inside the function for the same reason the news filter is: a
    strategy module has to stay importable on a box where the calendar file is
    absent, and keeping the import local means the day filter costs nothing at
    all — including its import — for a run that turns it off.
    """
    from backtest.event_calendar import session_weekday

    return np.asarray(session_weekday(_bar_timestamps(bars)))


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
# Signals
# --------------------------------------------------------------------------
def _validate(fast_rsi_window: int, slow_rsi_window: int,
              fast_rsi_os_threshold: float, fast_rsi_ob_threshold: float,
              allowed_days, sl_atr_mult: float, tp_atr_mult: float | None,
              trailing: bool, use_macro_trend: bool = True,
              use_pullback_trigger: bool = True, use_macd_filter: bool = True,
              use_day_filter: bool = True,
              use_news_filter: bool = False) -> None:
    """
    Reject parameter sets that do not describe this strategy.

    Every rejection here is recorded by `backtest.scan` as REJECTED and counted
    in the grid, so a combination the strategy refuses shrinks neither the
    reported search nor the honesty of `variants_tested`.

    The bounded-risk clauses at the end are the request's Layer 4. Read the
    limits of what they can promise in item 5 of "WHAT THIS MODULE DOES NOT DO"
    before quoting them as a cost check — they are not one.
    """
    if fast_rsi_window < 2:
        # An RSI(1) has a one-bar averaging window, so it is 100 on every up
        # close and 0 on every down close and nothing in between. "Fast RSI
        # dipped below 30 then crossed 50" would degenerate to "the last close
        # was down and this one is up", which is not an exhaustion measure at
        # all — and it would still produce a plausible equity curve.
        raise ValueError(
            f"fast_rsi_window must be >= 2; got {fast_rsi_window}")
    if slow_rsi_window < 2:
        raise ValueError(
            f"slow_rsi_window must be >= 2; got {slow_rsi_window}")
    if fast_rsi_window >= slow_rsi_window:
        # The premise is two DIFFERENT horizons: a fast oscillator mean-
        # reverting against a slow one that holds the trend. Equal windows make
        # the two series identical, so `Cross_Above(fast, slow)` can never fire
        # and Layer 2 silently blocks every trade; inverted, the "fast" RSI is
        # the slower of the two and the pullback trigger reads the trend while
        # the macro filter reads the noise. Either way the run completes and
        # describes a strategy nobody specified.
        raise ValueError(
            f"fast_rsi_window must be < slow_rsi_window; got "
            f"{fast_rsi_window} >= {slow_rsi_window}")

    for name, level in (("fast_rsi_os_threshold", fast_rsi_os_threshold),
                        ("fast_rsi_ob_threshold", fast_rsi_ob_threshold)):
        if level is None or not np.isfinite(float(level)):
            raise ValueError(f"{name} must be a finite number; got {level!r}")
        if not 0.0 <= float(level) <= 100.0:
            # The RSI is bounded 0-100 by construction. A threshold outside
            # that is not a strict filter, it is a filter that can never bind
            # or can never pass, and either runs as a silently different
            # strategy.
            raise ValueError(
                f"{name} must be within 0-100, the range the RSI can take; "
                f"got {level!r}")
    if not float(fast_rsi_os_threshold) < RSI_CENTERLINE < float(
            fast_rsi_ob_threshold):
        # The premise is exhaustion BELOW the centerline for a long and ABOVE
        # it for a short, followed by a re-cross of that same centerline. With
        # the oversold level at or above 50 the long trigger's two clauses
        # contradict each other — the fast RSI has to be under the threshold
        # and then cross UP through a line beneath it — and the strategy
        # becomes an accidental momentum-continuation entry wearing a
        # mean-reversion name.
        raise ValueError(
            f"fast_rsi_os_threshold must be below {RSI_CENTERLINE} and "
            f"fast_rsi_ob_threshold above it — the exhaustion levels are "
            f"defined either side of the centerline the trigger re-crosses; "
            f"got os={fast_rsi_os_threshold!r}, ob={fast_rsi_ob_threshold!r}")

    # Truthiness would silently accept "false" (a non-empty string, so True)
    # and 0.0. It bites hardest on the toggles: `--param use_macd_filter=false`
    # passed as the STRING "false" would apply the filter while the
    # leaderboard's params column said it was off. Every flag is checked rather
    # than coerced.
    for name, flag in (("use_macro_trend", use_macro_trend),
                       ("use_pullback_trigger", use_pullback_trigger),
                       ("use_macd_filter", use_macd_filter),
                       ("use_day_filter", use_day_filter),
                       ("use_news_filter", use_news_filter),
                       ("trailing", trailing)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be a bool; got {flag!r}")

    if not (use_macro_trend or use_pullback_trigger or use_macd_filter):
        # With all three off there is no DIRECTIONAL condition left anywhere in
        # the module: both the long and the short conjunction reduce to the
        # same all-True state, so every bar carries a long AND a short
        # candidate and the walk's ambiguous-bar rule takes NEITHER. The run
        # would complete, report zero trades and look like a strategy with no
        # edge rather than like a configuration that cannot express one.
        # Refused here so it reads as REJECTED in the scan output instead.
        # The day filter and the news filter are not directional and cannot
        # substitute — they only ever remove bars from both sides at once.
        raise ValueError(
            "at least one of use_macro_trend, use_pullback_trigger and "
            "use_macd_filter must be True; with all three off the strategy "
            "has no directional condition, both sides trigger on every bar, "
            "and the walk takes neither")

    # `allowed_days` is validated whether or not the toggle is on. A malformed
    # list that only raises when someone switches the filter back on is a
    # failure held in reserve.
    if isinstance(allowed_days, (str, bytes, int, np.integer)):
        # A bare `3` is the single most likely mistake here and it is not a
        # harmless one: `np.isin(weekday, 3)` is legal and would silently keep
        # Thursdays only. A string is worse — `np.isin` against "0,1,2"
        # matches nothing and the strategy takes no trade at all.
        raise ValueError(
            f"allowed_days must be a sequence of weekday numbers "
            f"(Monday=0 .. Sunday=6), not a bare scalar; got "
            f"{allowed_days!r} — write [3] rather than 3")
    try:
        days = [int(d) for d in allowed_days]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"allowed_days must be a sequence of whole weekday numbers "
            f"(Monday=0 .. Sunday=6); got {allowed_days!r}") from exc
    if not days:
        # An empty list blocks every entry on both sides. That is a strategy
        # that cannot trade, not a configuration of this one, and it would
        # report as a flat equity curve rather than as an error.
        raise ValueError(
            "allowed_days must name at least one weekday; an empty list "
            "blocks every entry and reports as a strategy with no signals")
    if len(set(days)) != len(days):
        raise ValueError(
            f"allowed_days must not repeat a weekday; got {allowed_days!r}")
    if any(d < 0 or d > 6 for d in days):
        raise ValueError(
            f"allowed_days must be within 0-6 (Monday=0 .. Sunday=6); got "
            f"{allowed_days!r}")
    # 5 and 6 are ACCEPTED, not rejected. On a CME contract a session date is
    # never a Saturday and a Sunday evening rolls into Monday, so including
    # them is a no-op there — but the lake also carries instruments that trade
    # through the weekend, and refusing the numbers would make this module
    # unusable on them for the sake of a warning nobody needs.

    # ---- Layer 4: bounded risk. A sanity floor, NOT a cost check. ----
    if sl_atr_mult is None or not np.isfinite(float(sl_atr_mult)):
        # The stop is not optional. Layer 4's signal exits can close a
        # position, but neither of them is guaranteed to arrive, and a trade
        # with no stop and no target held to the end of the lake is a single
        # observation reported as a strategy.
        raise ValueError(
            f"sl_atr_mult must be a finite number > 0; got {sl_atr_mult!r}")
    if not MIN_STOP_ATR_MULT <= float(sl_atr_mult) <= MAX_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be within {MIN_STOP_ATR_MULT}-"
            f"{MAX_STOP_ATR_MULT} x ATR{ATR_PERIOD}; got {sl_atr_mult!r}. "
            f"Below the floor the stop sits inside the bar noise the ATR is "
            f"measuring and is breached by the modelled slippage rather than "
            f"by the market.")
    if tp_atr_mult is None:
        # THE REQUEST'S SECTION 5 RULE, ENFORCED RATHER THAN ASSUMED — see
        # deviation 6. 72 of the 432 grid cells land here and are counted as
        # REJECTED.
        if not trailing:
            raise ValueError(
                "tp_atr_mult=None (no take-profit) requires trailing=True. "
                "With no target and a FIXED stop, a position that outlives "
                "both Layer 4 signal exits is held against a stop that never "
                "moves, and the trade is closed by whichever of the two "
                "arrives first — possibly neither, for the length of the "
                "run. The trailing stop is what makes the no-target "
                "configuration exitable.")
    else:
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


def _series(bars: pd.DataFrame, fast_rsi_window: int,
            slow_rsi_window: int) -> dict:
    """
    The shared calculation behind `signal_fn`, `indicators` and `ml_features`.

    One place, so the 200 EMA the tear sheet draws is the array the exit was
    taken from and the RSI pair the classifier reads is the pair the entry was
    gated on. Two call sites computing this separately would be free to drift
    apart with nothing raising.
    """
    close = bars["close"].astype(float)
    rsi_fast = _rsi(close, fast_rsi_window)
    rsi_slow = _rsi(close, slow_rsi_window)
    return {
        "rsi_fast": rsi_fast,
        "rsi_slow": rsi_slow,
        # The request's Version B feature, computed here rather than in
        # `ml_features` so the spread the model reads is the difference of the
        # two series the entry compared.
        "rsi_spread": rsi_fast - rsi_slow,
        "macd_hist": _macd_hist(close),
        "ema_trend": _ema(close, EMA_TREND_PERIOD),
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

    Imported inside the function rather than at module scope, for the reason
    given on `_session_weekday`.
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


def _signal_arrays(bars: pd.DataFrame, fast_rsi_window: int,
                   slow_rsi_window: int, fast_rsi_os_threshold: float,
                   fast_rsi_ob_threshold: float, allowed_days,
                   sl_atr_mult: float, tp_atr_mult: float | None,
                   trailing: bool, use_macro_trend: bool = True,
                   use_pullback_trigger: bool = True,
                   use_macd_filter: bool = True, use_day_filter: bool = True,
                   use_news_filter: bool = False) -> tuple:
    """
    Everything the walk produces, from one place.

    `signal_fn` and `indicators` both go through here, so the 200 EMA a reader
    sees the close cross is the array the exit came from and the stop line they
    see breached is the array the exit was taken from. Two call sites computing
    this separately would be free to drift apart with nothing raising.
    """
    s = _series(bars, fast_rsi_window, slow_rsi_window)
    close = bars["close"].astype(float)
    index = bars.index

    # `ready` is assembled FROM THE ACTIVE FILTERS ONLY, and that is the part
    # of the toggles that is easy to get wrong.
    #
    # Requiring every series unconditionally would make a toggled-off filter
    # cost its warm-up anyway: `use_macd_filter=False` would still wait 34 bars
    # for a histogram nothing reads, so the "no MACD" cell would be scored on a
    # shorter history than the strategy it is meant to represent — and the
    # comparison the toggles exist to enable would be between different
    # samples. Every comparison against NaN is False, so the symptom would be
    # silently missing early trades rather than an error.
    #
    # THREE SERIES ARE UNCONDITIONAL WHATEVER THE TOGGLES SAY, and each for a
    # reason that is invisible if it is got wrong:
    #
    #   atr         sets the stop distance and the target. An entry taken
    #               before the ATR exists would have no stop.
    #   ema_trend   is an EXIT condition (Layer 4) and has no toggle. A trade
    #               opened before the 200 EMA exists is a trade taken under an
    #               exit rule that cannot fire — the position would be held on
    #               a rule the reader believes is active. It costs 200 bars of
    #               warm-up on every configuration and that is the correct
    #               price.
    #   rsi_fast    is the other half of Layer 4, for the same reason.
    ready = (s["atr"].notna() & s["ema_trend"].notna()
             & s["rsi_fast"].notna())
    if use_macro_trend:
        ready &= s["rsi_slow"].notna()
    if use_pullback_trigger:
        # The trigger reads the fast RSI's PREVIOUS bar and compares it against
        # the slow RSI, so both series and one bar of history are required.
        ready &= (s["rsi_slow"].notna() & s["rsi_fast"].shift(1).notna())
    if use_macd_filter:
        ready &= (s["macd_hist"].notna() & s["macd_hist"].shift(1).notna())

    # A DISABLED FILTER IS ALL-TRUE, NOT ALL-FALSE. `_on` returns a True Series
    # when its toggle is off, so the conjunctions below are written once and
    # read the same whichever filters are active. Writing them as `cond if flag
    # else <omit>` instead would need a different expression per combination,
    # and each combination would be another chance for the long and short arms
    # to stop mirroring each other.
    def _on(condition: pd.Series, flag: bool) -> pd.Series:
        return condition if flag else pd.Series(True, index=index)

    # Layer 1a — the macro trend. Each arm is an explicit comparison rather
    # than the negation of its opposite, because `~(rsi_slow > 50)` is true
    # wherever the slow RSI merely fails to be above 50 — which includes exact
    # equality and every warm-up bar — so a short arm written as a negation
    # would fire on ties and on bars where the series does not exist.
    long_macro = _on(s["rsi_slow"] > RSI_CENTERLINE, use_macro_trend)
    short_macro = _on(s["rsi_slow"] < RSI_CENTERLINE, use_macro_trend)

    # Layer 1b — the day filter. NOT directional: it removes the same bars from
    # both sides, so it can never tilt the strategy long or short. Keyed on the
    # CME SESSION weekday — see deviation 5.
    if use_day_filter:
        weekday = _session_weekday(bars)
        day_ok = pd.Series(np.isin(weekday, [int(d) for d in allowed_days]),
                           index=index)
    else:
        day_ok = pd.Series(True, index=index)

    # Layer 2 — the pullback trigger. `[1]` in the request is the PREVIOUS bar,
    # so `.shift(1)`, which looks one bar BACKWARD. The exhaustion test is on
    # the previous bar and the cross is on this one, which is what makes the
    # pair an event: the fast RSI was at the extreme and has now turned.
    prev_fast = s["rsi_fast"].shift(1)
    long_pull = ((prev_fast <= float(fast_rsi_os_threshold))
                 & (_cross_above(s["rsi_fast"], s["rsi_slow"])
                    | _cross_above(s["rsi_fast"], RSI_CENTERLINE)))
    short_pull = ((prev_fast >= float(fast_rsi_ob_threshold))
                  & (_cross_below(s["rsi_fast"], s["rsi_slow"])
                     | _cross_below(s["rsi_fast"], RSI_CENTERLINE)))

    # Layer 3 — the MACD histogram: on the matching side of zero AND expanding.
    # The two are different statements and the request requires both. Positive
    # says the fast average leads; expanding says the lead is still WIDENING,
    # and momentum that is positive and narrowing is a trend being handed back.
    hist = s["macd_hist"]
    prev_hist = hist.shift(1)
    long_macd = _on((hist > 0) & (hist > prev_hist), use_macd_filter)
    short_macd = _on((hist < 0) & (hist < prev_hist), use_macd_filter)

    # The STATE-shaped layers, which is everything except the pullback trigger.
    long_state = long_macro & long_macd & day_ok & ready
    short_state = short_macro & short_macd & day_ok & ready

    if use_pullback_trigger:
        # Layer 2 is an EVENT, so the conjunction is already event-shaped: it
        # fires on the bar the fast RSI turns and not again until the next
        # pullback. No rising edge is imposed on top — two crosses on
        # consecutive bars (the slow RSI on one, the 50 line on the next) are
        # two genuine triggers, and an edge test would silently drop the
        # second.
        long_trigger = long_state & long_pull
        short_trigger = short_state & short_pull
    else:
        # With the trigger off there is no event left anywhere in the stack —
        # see deviation 4. The candidate becomes the RISING EDGE of the
        # remaining conjunction, so one confluence episode is one trade rather
        # than a re-entry on every bar for as long as it lasts.
        #
        # `prev_ready` is what keeps the first fully-warm bar from registering
        # as an edge: `long_state` is False through warm-up because every
        # comparison against NaN is False, so at the first ready bar it can
        # flip False->True purely because the series came into existence. That
        # is a fact about the warm-up, not about price. `fill_value=False`
        # keeps bar 0 out for the same reason.
        prev_ready = ready.shift(1, fill_value=False)
        long_trigger = (long_state & ~long_state.shift(1, fill_value=False)
                        & prev_ready)
        short_trigger = (short_state & ~short_state.shift(1, fill_value=False)
                         & prev_ready)

    long_entry_ok = long_trigger.to_numpy(dtype=bool)
    short_entry_ok = short_trigger.to_numpy(dtype=bool)

    # Layer 4's SIGNAL EXITS, mirrored — see deviation 3 for why the 200-EMA
    # clause is not shared verbatim between the sides. These are events, not
    # states: a long is closed on the bar the fast RSI crosses UP through 70,
    # not on every bar it happens to be above it. A position opened while the
    # fast RSI was already beyond the level is therefore closed by its bracket
    # or by the EMA cross, which is the honest reading of "crosses over" and is
    # only reachable with the pullback trigger switched off.
    long_sig_exit = (_cross_above(s["rsi_fast"], EXIT_OB_LEVEL)
                     | _cross_below(close, s["ema_trend"]))
    short_sig_exit = (_cross_below(s["rsi_fast"], EXIT_OS_LEVEL)
                      | _cross_above(close, s["ema_trend"]))

    # The PERMITTED-BAR states and the TRIGGERS both travel back on `s`, and
    # keeping them apart is not bookkeeping — they answer different questions
    # and only one of them behaves the way a reader expects.
    #
    # The states are MONOTONE in the layers: switching a filter on can only
    # shrink the set of bars the strategy is permitted to trade on. THE
    # TRIGGERS ARE NOT, in either branch above. With the pullback trigger on,
    # a state layer removes bars from a conjunction whose event half is
    # unchanged, so triggers can only be removed — but with it off the entry is
    # a rising edge, and removing bars from the MIDDLE of one permitted stretch
    # splits it into two stretches and creates a SECOND trigger where there had
    # been one.
    #
    # This is the same trap CLAUDE.md records for `ema_trend_filter` one level
    # deeper: there a filter subtracts CANDIDATES but not necessarily trades,
    # because the walk holds one position at a time. NEVER use a count — of
    # trades OR of triggers — to decide whether a layer is wired. Compare the
    # permitted states, which is what
    # `tests/test_double_rsi_macd_scalp_20260823.py` does.
    s["day_ok"] = day_ok
    s["long_state"] = long_state
    s["short_state"] = short_state
    s["long_pull"] = long_pull
    s["short_pull"] = short_pull
    s["long_trigger"] = long_trigger
    s["short_trigger"] = short_trigger
    s["long_sig_exit"] = long_sig_exit
    s["short_sig_exit"] = short_sig_exit

    # The news veto lands on CANDIDATES, before the walk, so a blocked trigger
    # leaves the strategy flat and free to take a later one. See the module
    # docstring for why that differs from the engine's own `--news-filter`,
    # which acts on the walk's output.
    if use_news_filter:
        long_entry_ok, short_entry_ok = _news_suppress(
            bars, long_entry_ok, short_entry_ok)

    # No session flatten. `flat_bar` all-False is what "positions are carried
    # overnight" looks like in this kernel, and for a module called a scalp it
    # is a modelled exposure rather than an omission — see item 2 of "WHAT THIS
    # MODULE DOES NOT DO".
    flat_bar = np.zeros(len(bars), dtype=bool)

    entries, exits, s_entries, s_exits, stop, target = _walk(
        long_entry_ok,
        short_entry_ok,
        long_sig_exit.to_numpy(dtype=bool),
        short_sig_exit.to_numpy(dtype=bool),
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
              fast_rsi_window: int = 5,
              slow_rsi_window: int = 21,
              fast_rsi_os_threshold: float = 30.0,
              fast_rsi_ob_threshold: float = 70.0,
              allowed_days=(0, 1, 2),
              use_macro_trend: bool = True,
              use_pullback_trigger: bool = True,
              use_macd_filter: bool = True,
              use_day_filter: bool = True,
              use_news_filter: bool = False,
              sl_atr_mult: float = 1.5,
              tp_atr_mult: float | None = 3.0,
              trailing: bool = False) -> tuple[pd.Series, pd.Series,
                                               pd.Series, pd.Series]:
    """
    Take the pullback only where the slow RSI, the fast RSI's turn and the MACD
    histogram all agree; exit on the RSI extreme, the 200-EMA break, the stop or
    the target. Both directions.

    Returns the FOUR-MASK form of the strategy contract:

        (long_entries, long_exits, short_entries, short_exits)

    all boolean Series on `bars.index`. `backtest.engine.unpack_signals`
    accepts this alongside the older two-mask long-only form; returning a
    three-tuple or a bare Series raises there rather than silently losing the
    short side.

    The entry is a bar-level EVENT — the fast RSI's cross, inside the standing
    conditions — so a long stretch of bars satisfying every state produces a
    signal only where the trigger fires. The walk enters only when flat, so a
    second trigger while a position is open is ignored rather than pyramided,
    and a short trigger arriving while long is ignored rather than reversing
    the position.

    Every comparison at bar i uses only bars <= i. The engine then fills at bar
    i+1's open, so nothing here can see a price it would not have had. That is
    the request's `signal_delay=1`: it is the engine's contract, not a
    parameter this module sets, and there is no code path here that could fill
    on the signal bar.

    `allowed_days` defaults to a TUPLE in this signature while `DEFAULT_PARAMS`
    declares the request's list. A mutable default argument is shared across
    every call that does not override it, so a caller mutating it would change
    the strategy for the rest of the process; the tuple cannot be mutated and
    the two are compared element-wise by the test suite.
    """
    _validate(fast_rsi_window, slow_rsi_window, fast_rsi_os_threshold,
              fast_rsi_ob_threshold, allowed_days, sl_atr_mult, tp_atr_mult,
              trailing, use_macro_trend, use_pullback_trigger, use_macd_filter,
              use_day_filter, use_news_filter)

    _s, entries, exits, s_entries, s_exits, _stop, _target = _signal_arrays(
        bars, fast_rsi_window, slow_rsi_window, fast_rsi_os_threshold,
        fast_rsi_ob_threshold, allowed_days, sl_atr_mult, tp_atr_mult,
        trailing, use_macro_trend, use_pullback_trigger, use_macd_filter,
        use_day_filter, use_news_filter)

    return (pd.Series(entries, index=bars.index),
            pd.Series(exits, index=bars.index),
            pd.Series(s_entries, index=bars.index),
            pd.Series(s_exits, index=bars.index))


def indicators(bars: pd.DataFrame,
               fast_rsi_window: int = 5,
               slow_rsi_window: int = 21,
               fast_rsi_os_threshold: float = 30.0,
               fast_rsi_ob_threshold: float = 70.0,
               allowed_days=(0, 1, 2),
               use_macro_trend: bool = True,
               use_pullback_trigger: bool = True,
               use_macd_filter: bool = True,
               use_day_filter: bool = True,
               use_news_filter: bool = False,
               sl_atr_mult: float = 1.5,
               tp_atr_mult: float | None = 3.0,
               trailing: bool = False) -> dict[str, pd.Series]:
    """
    The price-scale series, for the tear sheet to draw over the candles.

    Computed HERE, the same way and from the same columns `signal_fn` reads, so
    the 200-EMA break a reader sees is the array the exit was taken from. A
    second implementation living in the report would be free to disagree with
    this one — a chart showing the baseline crossed a bar away from where the
    trade closed, with nothing raising.

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

    THE TWO RSIs AND THE MACD HISTOGRAM ARE DELIBERATELY NOT RETURNED, which
    means Layers 1, 2 and 3 are the entry conditions a reader cannot see on the
    chart. That is a real gap and the alternative is worse: the inspector draws
    these on the PRICE axis. An RSI is bounded 0-100, so on a 20,000-point
    contract it would be drawn flat along the bottom of the panel at levels
    nothing ever traded at; the MACD histogram oscillates around zero in price
    units and would be a flat line in the same place. Rescaling either onto the
    price axis would put a line on the chart at levels that are not levels.
    Both reach the reader through the strategy card and through `ml_features`;
    the 200 EMA and the risk levels are the ones that live on the price axis,
    and they are here.

    The take-profit line is OMITTED ENTIRELY when `tp_atr_mult` is None. An
    all-NaN series would render as an empty legend entry, which reads as a
    target that exists and never got close — the opposite of the truth. The
    200 EMA is always drawn: it is an EXIT condition and has no toggle.

    Warm-up stays NaN rather than drawing the baseline flat through the first
    bars.
    """
    _validate(fast_rsi_window, slow_rsi_window, fast_rsi_os_threshold,
              fast_rsi_ob_threshold, allowed_days, sl_atr_mult, tp_atr_mult,
              trailing, use_macro_trend, use_pullback_trigger, use_macd_filter,
              use_day_filter, use_news_filter)

    s, _e, _x, _se, _sx, stop, target = _signal_arrays(
        bars, fast_rsi_window, slow_rsi_window, fast_rsi_os_threshold,
        fast_rsi_ob_threshold, allowed_days, sl_atr_mult, tp_atr_mult,
        trailing, use_macro_trend, use_pullback_trigger, use_macd_filter,
        use_day_filter, use_news_filter)

    kind = "Trailing" if trailing else "Fixed"
    out = {
        f"EMA {EMA_TREND_PERIOD} (exit baseline)": s["ema_trend"],
        # The stop is always drawn — it is the exit that cannot be switched off.
        f"{kind} Stop ({sl_atr_mult}xATR{ATR_PERIOD})":
            pd.Series(stop, index=bars.index),
    }
    if tp_atr_mult is not None:
        out[f"Take Profit ({tp_atr_mult}xATR{ATR_PERIOD})"] = pd.Series(
            target, index=bars.index)
    return out


def ml_features(bars: pd.DataFrame,
                fast_rsi_window: int = 5,
                slow_rsi_window: int = 21,
                **_ignored) -> pd.DataFrame:
    """
    Version B's feature matrix: the eight columns the request names.

    Read by `agents.tier3_workers.load_strategy` and handed to
    `apply_ml_signal_filter` in place of the shared `causal_features` default.
    Declaring it here is the point: the shared default carries seven columns
    (ATR, volume, RSI, hour, minute and two return horizons), most of which
    this strategy's hypothesis says nothing about. A classifier vetoing THESE
    entries should be reading the state THIS strategy is a claim about — how
    exhausted the fast oscillator is, how far it sits from the slow one, how
    the histogram is behaving, how volatile and how heavily traded the bar is,
    and when in the week and the session it arrived.

    Columns, in `ML_FEATURES` order:

        rsi_fast     The fast RSI, 0-100. Under the default stack every
                     surviving candidate has already crossed back up (or down)
                     through a line, so what the column carries is HOW FAR the
                     turn has gone, which is the part the veto can still act on.
        rsi_slow     The slow RSI. Its SIDE of 50 is already spent as Layer 1;
                     its DISTANCE from 50 is the strength of the trend the
                     pullback is being bought into.
        rsi_spread   `rsi_fast - rsi_slow`, the request's spread. Not derivable
                     by the model from the two columns as cheaply as it is
                     given: a tree splits on one feature at a time, so a
                     difference it would have to approximate through a
                     staircase of splits is worth handing over directly.
        macd_hist    The histogram, the same array Layer 3 gates on. Its
                     HEIGHT, not its sign — the sign is already an entry
                     condition.
        atr_norm     ATR(14) / close — the request's "14-period normalized
                     ATR". Normalised because the classifier is fitted across a
                     16-year history in which the contract's price level
                     changes by a multiple: a raw ATR in points would let the
                     model learn "2015" instead of "volatile".
        volume_z     The 20-bar volume z-score. THE ONLY PLACE THE REQUEST'S
                     "institutional volume expansion" PREMISE IS ACTUALLY
                     MEASURED — no Version A rule reads volume at all.
        day_of_week  Monday=0 .. Sunday=6 on the CME SESSION date, the same
                     weekday Layer 1's day filter is keyed on. Handing the
                     model the UTC weekday while the filter used the session
                     weekday would put the two out of step for every bar
                     between 18:00 ET and midnight UTC.
        hour_et      Hour of day in America/New_York, 0-23. A NAMED ZONE, not
                     a UTC hour and not a fixed offset: the CME session keeps
                     its local clock across the DST changeover, so a UTC hour
                     smears one session hour across two values twice a year and
                     the feature would mean a different thing in March than in
                     October. On an intraday scalp this is the column most
                     likely to carry a real effect — the open and the close are
                     not the same market as 02:00.

    THE TWO RSI PERIODS TRACK THE BOUND PARAMETERS; NOTHING ELSE DOES. Three of
    these columns ARE the entry conditions, and an `rsi_fast` computed at
    period 5 while the sweep is testing period 2 would be a model vetoing
    entries on an oscillator the strategy is not using. The cost is real and
    worth stating: every grid cell that moves a window gets a differently-
    shaped model, so `--scan` is searching over classifiers as well as over
    strategies, and two cells' Version B results are less comparable than their
    Version A results are. The thresholds, the risk multipliers and the toggles
    do NOT enter here and are absorbed by `**_ignored` — none of them changes
    what a feature MEASURES, only which candidates survive to be scored.

    STRICTLY CAUSAL, and worth being explicit about because this matrix is
    fitted on. A value at row i is a function of bars 0..i only: no
    `shift(-k)`, no centred window, no reversed slice, and nothing computed off
    a full-sample statistic. That last one is the trap a shift-based audit does
    not catch — a scaler fitted on the whole frame leaks the test period's
    distribution into the training rows — which is why nothing here is
    standardised against a global mean. The volume z-score is a ROLLING
    20-bar statistic for exactly that reason.

    Using bar i's close to decide a signal on bar i is legitimate: the engine
    fills at bar i+1's open, never on the signal bar. That one-bar gap is what
    makes these features tradeable rather than clairvoyant.

    NaN warm-up rows are left as NaN. `HistGradientBoostingClassifier` consumes
    them natively, and filling them with a column mean would import a
    full-sample statistic into exactly the rows that have no history.
    """
    s = _series(bars, fast_rsi_window, slow_rsi_window)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)

    vol_mean = volume.rolling(VOLUME_Z_PERIOD,
                              min_periods=VOLUME_Z_PERIOD).mean()
    vol_std = volume.rolling(VOLUME_Z_PERIOD,
                             min_periods=VOLUME_Z_PERIOD).std()
    # THE REQUEST'S "fill degenerate rolling windows with 0.0" CLAUSE, and here
    # 0.0 is the true value: a window whose volume never varied has every bar
    # exactly at its mean, which is zero standard deviations from it. Dividing
    # by the zero would give NaN (0/0) or an infinity, and both would propagate
    # into the model as a missing row or an outlier the trees would split on.
    # `.where(vol_std > 0)` keeps the warm-up NaN — an undefined window is not
    # a flat one.
    volume_z = ((volume - vol_mean) / vol_std.where(vol_std > 0)).where(
        vol_std.isna() | (vol_std > 0), 0.0)

    # `_bar_timestamps` returns a tz-AWARE UTC index, so this conversion is
    # defined. A naive index would raise here under pandas 3.0 rather than
    # being silently read as if it were already ET.
    hour_et = _bar_timestamps(bars).tz_convert(SESSION_TZ).hour

    out = pd.DataFrame({
        "rsi_fast": s["rsi_fast"].to_numpy(dtype=float),
        "rsi_slow": s["rsi_slow"].to_numpy(dtype=float),
        "rsi_spread": s["rsi_spread"].to_numpy(dtype=float),
        "macd_hist": s["macd_hist"].to_numpy(dtype=float),
        # `close.where(close != 0)` makes a zero close NaN rather than an
        # infinity: a bar with a zero price is bad data, and the classifier
        # consumes a NaN natively while an inf propagates.
        "atr_norm": (s["atr"] / close.where(close != 0)).to_numpy(dtype=float),
        "volume_z": volume_z.to_numpy(dtype=float),
        "day_of_week": np.asarray(_session_weekday(bars), dtype=float),
        "hour_et": np.asarray(hour_et, dtype=float),
    }, index=bars.index)
    return out[ML_FEATURES]


def make_signal_fn(fast_rsi_window: int = 5,
                   slow_rsi_window: int = 21,
                   fast_rsi_os_threshold: float = 30.0,
                   fast_rsi_ob_threshold: float = 70.0,
                   allowed_days=(0, 1, 2),
                   use_macro_trend: bool = True,
                   use_pullback_trigger: bool = True,
                   use_macd_filter: bool = True,
                   use_day_filter: bool = True,
                   use_news_filter: bool = False,
                   sl_atr_mult: float = 1.5,
                   tp_atr_mult: float | None = 3.0,
                   trailing: bool = False):
    """
    Bind parameters for `agents.tier3_workers.load_strategy`.

    The loader prefers this factory when params are supplied; the engine itself
    only ever calls the bound `signal_fn(bars)`. Validation runs HERE as well
    as inside `signal_fn`, so `--scan` records a bad combination as REJECTED at
    bind time rather than discovering it one symbol into the sweep — which is
    how the 72 (no-target, fixed-stop) cells of the request's grid are counted.
    """
    _validate(fast_rsi_window, slow_rsi_window, fast_rsi_os_threshold,
              fast_rsi_ob_threshold, allowed_days, sl_atr_mult, tp_atr_mult,
              trailing, use_macro_trend, use_pullback_trigger, use_macd_filter,
              use_day_filter, use_news_filter)

    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series,
                                            pd.Series, pd.Series]:
        return signal_fn(bars,
                         fast_rsi_window=fast_rsi_window,
                         slow_rsi_window=slow_rsi_window,
                         fast_rsi_os_threshold=fast_rsi_os_threshold,
                         fast_rsi_ob_threshold=fast_rsi_ob_threshold,
                         allowed_days=allowed_days,
                         use_macro_trend=use_macro_trend,
                         use_pullback_trigger=use_pullback_trigger,
                         use_macd_filter=use_macd_filter,
                         use_day_filter=use_day_filter,
                         use_news_filter=use_news_filter,
                         sl_atr_mult=sl_atr_mult,
                         tp_atr_mult=tp_atr_mult,
                         trailing=trailing)

    return _bound
