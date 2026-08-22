"""
MA anchoring spread — entries require the fast average to diverge past a
PERCENTAGE hurdle from a slow anchor and to still be widening, rather than
merely to have crossed it.

Location:  ~/src/trading/strategies/experimental/ma_anchoring_spread_20260820.py

THE SPECIFICATION THIS WAS WRITTEN FROM
=======================================
Recorded verbatim, before the first backtest, because a specification written
after the equity curve is not a specification. If a run disagrees with any line
below, the run is the finding — do not edit this block to match it.

    1. STRATEGY SPECIFICATION & ARCHITECTURAL CONTRACT
       Strategy Identifier: ma_anchoring_spread
       Module Path:         strategies/experimental/ma_anchoring_spread_20260820.py
       Category:            Macro Momentum / Low-Frequency Anchoring
       Economic Premise:    Exploits institutional anchoring bias and
                            behavioral underreaction (Avramov, Kaplanski,
                            Subrahmanyam 2018; SSRN 3342508). Instead of binary
                            crossover triggers, entries require the Fast MA to
                            diverge significantly beyond a percentage hurdle
                            from the Slow Anchor MA (Spread Ratio = SMA_fast /
                            SMA_slow), confirming sustained institutional
                            momentum rather than low-volume chop.

    2. MULTI-LAYER ENTRY & EXIT CONFLUENCE LOGIC
       Layer 1 (Anchor Baseline):
           Long:  Close > SMA(slow_window)
           Short: Close < SMA(slow_window)
       Layer 2 (Spread Hurdle Trigger):
           spread_ratio = SMA(fast_window) / SMA(slow_window)
           Long:  spread_ratio >= (1.0 + spread_threshold)
           Short: spread_ratio <= (1.0 - spread_threshold)
       Layer 3 (Spread Expansion / Velocity):
           Long:  spread_ratio > spread_ratio[1]
           Short: spread_ratio < spread_ratio[1]
       Layer 4 (Exits & Rebalancing):
           Primary Exit: spread_ratio mean-reverts back inside
           [1.0 - spread_threshold * exit_revert_mult,
            1.0 + spread_threshold * exit_revert_mult]  OR via ATR brackets.
       Modular Toggles: use_macro_anchor, use_spread_hurdle,
                        use_spread_expansion, use_news_filter

    3. PARAMETERS & GRID
       Risk keys:          sl_atr_mult, tp_atr_mult, trailing
       Default Parameters: fast_window=21, slow_window=200,
                           spread_threshold=0.015, exit_revert_mult=0.5,
                           sl_atr_mult=1.5, tp_atr_mult=3.0, trailing=False
       PARAM_GRID:         fast_window       [10, 21, 30]
                           slow_window       [100, 200]
                           spread_threshold  [0.01, 0.015, 0.025]
                           sl_atr_mult       [1.0, 1.5, 2.0]
                           tp_atr_mult       [2.0, 3.0, None]
                           trailing          [False, True]

    4. DUAL-VERSION ARCHITECTURE
       Declarations: signal_fn, indicators, LOGIC, PARAM_GRID, make_signal_fn,
                     ml_features.
       Version A:    pure rule-based, next-bar-open execution.
       Version B:    apply_ml_signal_filter(..., features=ml_features), with
                     HistGradientBoostingClassifier over rolling spread ratio,
                     5- and 15-bar spread ROC, normalised ATR, rolling volume
                     z-score and hour-of-day. Label = net P&L after costs > 0.
                     The model is a causal VETO on Version A's candidates.

    5. MICROSTRUCTURE & SAFETY
       Next-bar-open execution, zero lookahead. Degenerate zero-range /
       zero-spread cases filled with 0.0 rather than left to propagate NaN
       through rolling windows. Pandas 3.0 compatible. tp_atr_mult=None
       requires trailing=True. use_news_filter wired to
       backtest.event_calendar.

WHERE THE DELIVERED MODULE DIVERGES FROM THE REQUEST
====================================================
Four places. Each is a decision that changes a number, so none of them is left
to be discovered from an equity curve later.

1. THE 0.0 DEGENERATE FILL IS NOT APPLIED TO THE SPREAD RATIO ITSELF, and
   applying it there would manufacture trades. `spread_ratio` is
   `SMA_fast / SMA_slow`, so its degenerate case is a slow anchor of zero —
   which is reachable: continuous futures here are NOT back-adjusted and CL
   printed negative in April 2020, so an anchor can cross zero. Filling that
   division with 0.0 would put the ratio at 0.0, which is BELOW every short
   hurdle `1 - spread_threshold` — the strategy would read "no anchor" as
   "maximum downside divergence" and open a short. It is left NaN, every
   comparison against it is False, and the bar is simply untradeable.
   The 0.0 fill IS applied where zero is the honest measurement rather than a
   missing one: the spread ROC of a flat window and the volume z-score of a
   dead-flat window, both in `ml_features`. See `_spread_ratio` and
   `_roc`.

2. `tp_atr_mult=None` REQUIRES `trailing=True`, so 54 of the grid's 324 cells
   are REJECTED rather than run. That is the request's own rule (section 5),
   and `backtest/scan.py` counts a rejected combination rather than dropping
   it, so `variants_tested` still reports the search that was asked for. The
   consequence worth stating: this module therefore does NOT satisfy the
   assumption `tests/test_risk_params.py` holds every module in its table to —
   that `tp_atr_mult=None` binds with any `trailing` — which is why it is not
   registered there. Its stop, target and trailing machinery is pinned instead
   by `tests/test_ma_anchoring_spread_20260820.py`, including that its copy of
   the walk kernel is character-for-character the shared one.

3. THE GRID IS 324 CELLS, WELL PAST THE 200-CELL BOUND this repo holds its
   grids to, and past `backtest/run.py`'s `SIZE_WARN`, so a sweep prints a
   warning before it starts. 270 are evaluated and 54 rejected. The request
   specifies these six axes and these values, so they are what is delivered —
   but read the winner accordingly: the reported Sharpe is the maximum of 270
   draws from one sample of bars, and that maximum climbs with the cell count
   whether or not anything in the market has changed. `variants_tested`
   travels onto every report, leaderboard row and Stage 3 audit, which is what
   makes 270 reportable rather than merely permitted. Multiply before quoting
   it across timeframes: `--tf 5m,15m,30m` is 810 fits per contract.
   The cheap trim, if somebody wants one, is `sl_atr_mult` — three stop widths
   over a sample this size is the axis paying the least per cell.

4. THE MODULE FILENAME AND THE STRATEGY IDENTIFIER DIFFER, deliberately and by
   the request. `STRATEGY_NAME` is `ma_anchoring_spread` and is what the LOGIC
   block and the pipeline JSON carry; the FILE is
   `ma_anchoring_spread_20260820.py`. `backtest/run.py` resolves `--strat` by
   FILENAME against `SEARCH_DIRS`, so every CLI command names the dated form:

       bt-run --strat ma_anchoring_spread_20260820 --symbols NQ --tf 15m

   `tests/test_ma_anchoring_spread_20260820.py` pins both spellings so a rename
   cannot leave them disagreeing silently.

WHAT THIS MODULE DOES NOT DO, AND WHY
=====================================
1. THERE ARE NO STOP OR TARGET ORDERS. `backtest/engine.py` drives
   `vbt.Portfolio.from_signals` off boolean masks and fills them at the NEXT
   bar's open. A stop here is an exit SIGNAL detected on the bar that breaches
   it and filled one bar later, at whatever the next open happens to be — not a
   fill at the stop price. On a bar that breaches both the stop and the target
   the modelled fill is neither level. A 1.5 x ATR stop does not bound a loss
   at 1.5 x ATR, and every drawdown figure here has to be read that way.

2. THERE IS NO SESSION LOGIC. No entry window, no flatten at the bell, so
   positions are carried overnight and over weekends. The request's rules
   enumerate the spread-reversion exit and the ATR brackets and nothing else;
   a session flatten would be a strategy nobody specified. On a 15m timeframe
   the overnight gap is a real exposure and it is priced into no number here.

3. THE ENTRY IS A STATE, NOT AN EVENT, and this is the substantive difference
   from a crossover module. `spread_ratio >= 1 + threshold` stays true for as
   long as the divergence holds, so the entry condition is satisfied on a RUN
   of bars rather than on one. What makes that tradeable is the walk: it enters
   only from flat, so the run produces ONE entry — on its first bar — and the
   rest are ignored while the position is open. Layer 3 narrows it further, to
   the bars where the divergence is still widening. Two consequences: after an
   ATR stop fires inside a still-diverging trend the strategy RE-ENTERS on the
   next expanding bar, and a `spread_threshold` low enough to be satisfied
   most of the time makes this close to always-in-the-market.

4. EVERY TOGGLE OFF IS A MUTE STRATEGY, NOT A NULL ONE. With
   `use_macro_anchor`, `use_spread_hurdle` and `use_spread_expansion` all
   False, the long and short conditions are both just "the averages exist", so
   EVERY warm bar signals both sides at once — and the walk's ambiguity rule
   takes NEITHER. The all-off cell trades zero times. It is therefore not
   usable as the baseline this idea has to beat; the honest baseline is
   `use_spread_hurdle=True` alone, which is the bare hurdle without the anchor
   or the velocity filter. `PARAM_GRID` sweeps none of the toggles, so this is
   only reachable through an explicit `--param`, and a zero-trade run is
   visible in every report it produces — but it looks like a strategy that
   found nothing rather than a configuration that could not fire.

5. TURNING THE HURDLE OFF CHANGES THE EXIT TOO, and not by omission. The exit
   band is `1 +/- spread_threshold * exit_revert_mult`, which is INSIDE the
   entry hurdle, so with the hurdle ON a fresh trade always starts outside its
   own exit condition. With `use_spread_hurdle=False` it often does not: the
   entry can fire at a ratio already inside the band, and the walk checks exits
   from the fill bar onward, so that trade is closed on the bar it opened on.
   Measured on the 4,000-bar synthetic fixture in
   `tests/test_ma_anchoring_spread_20260820.py`, 81% of the hurdle-off trades
   last a single bar and the median hold is 1 bar, against 0% and 12 bars with
   the hurdle on. `use_spread_hurdle=False` is therefore not "the same strategy
   without the hurdle" — it is mostly a one-bar-trade machine paying two round
   turns for each — and an ablation that reads it as the former is comparing
   against something nobody proposed.

THE NEWS FILTER IS IMPORTED, NOT REIMPLEMENTED
==============================================
`use_news_filter` calls `backtest.event_calendar.apply_entry_filters`, the only
implementation of that filter in this repository and one documented as callable
by a strategy module on its own masks. A second copy living here would be free
to disagree with the engine's about which bars a release covers, and the two
would be compared by nobody.

THIS PUTS THE MODULE OUTSIDE `ALLOWED_IMPORTS`, DELIBERATELY, exactly as
`sma_momentum_crossover` is. `agents.tier3_workers.ALLOWED_IMPORTS` permits a
strategy module numpy, pandas, math, vectorbtpro and numba and nothing else,
but that allowlist governs MODEL-GENERATED code, which is audited before it is
executed; it does not run over hand-written modules, which `load_strategy`
imports from a file path. The consequences, stated rather than discovered
later: this module would be REJECTED by the AST validator if it were ever
passed through the synthesis path, and the test suite pins that the
`backtest.event_calendar` import is the ONLY objection the validator has to it
— so a future edit reaching for `open`, `eval` or a network library fails
loudly instead of hiding behind an exception granted for something else.

The filter is applied to the CANDIDATE triggers, before the position walk, and
that is a different thing from the engine's `--news-filter` flag, which applies
to the walk's OUTPUT entries. Blocking a candidate leaves the strategy FLAT and
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

The classifier is `HistGradientBoostingClassifier`, which is what the request
names and what every other Version B in this repository already uses. The
training label is "net P&L after commission and slippage > 0", which is the
request's own label.

The feature matrix comes from this module's `ml_features` hook rather than the
shared `causal_features` default — see that function for the columns and for
the one thing that makes this hook different from every other one in the tree:
two of its columns depend on the swept parameters.

Every calculation is causal. Values at bar i use bars <= i only, warm-up stays
NaN and is never treated as a signal, and the engine then fills at bar i+1's
open — so nothing here can see a price it would not have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The identifier the pipeline logs and the JSON handoffs carry. It is
# deliberately NOT the module filename: `backtest/run.py` resolves `--strat` by
# filename, so the CLI name is `ma_anchoring_spread_20260820` while this is the
# strategy's own name. Both spellings are pinned by the test suite.
STRATEGY_NAME = "ma_anchoring_spread"

TIMEFRAME = "15m"
SYMBOLS = ["NQ", "ES", "CL", "GC"]

DEFAULT_PARAMS = {
    "fast_window": 21,
    "slow_window": 200,
    "spread_threshold": 0.015,
    "exit_revert_mult": 0.5,
    "use_macro_anchor": True,
    "use_spread_hurdle": True,
    "use_spread_expansion": True,
    "use_news_filter": False,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "trailing": False,
}

# The search space `backtest/run.py --scan` and `backtest/scan.py` sweep.
#
# 3 x 2 x 3 x 3 x 3 x 2 = 324 declared cells. 54 of them pair
# `tp_atr_mult=None` with `trailing=False`, which `_validate` REJECTS (see
# divergence 2 in the module docstring), leaving 270 evaluated. A rejected
# combination is counted by the scanner rather than dropped, so the reported
# search is the one that was asked for rather than the one that happened to
# run.
#
# 324 is past the 200-cell bound this repo holds its grids to and past
# `run.py`'s SIZE_WARN, so a sweep prints a warning before it starts. It is
# what the request specifies and it is what is delivered; the honesty cost is
# recorded at divergence 3 rather than paid silently.
#
# WHAT IS NOT SWEPT:
#
#   exit_revert_mult  Fixed at the specified 0.5. It is the exit rule's shape,
#                     and sweeping it alongside `spread_threshold` searches the
#                     entry and the exit band against each other — two axes
#                     that both scale with the same number, which is a way to
#                     find a pair that fits these bars rather than a rule.
#   use_macro_anchor  Fixed ON. Opening the toggles doubles the grid per
#                     toggle and searches over STRATEGIES while reporting a
#                     parameter. Run the ablation out of band, once, at the
#                     winning cell: `--param use_macro_anchor=false`.
#   use_spread_hurdle Fixed ON, same reasoning — and see divergence 5: with it
#                     off the exit band binds on the fill bar, so that cell is
#                     not a comparable strategy.
#   use_spread_expansion  Fixed ON, same reasoning.
#   use_news_filter   Fixed OFF. It depends on an external calendar whose
#                     provenance changes what the filter means, so it is an
#                     operator decision rather than a search axis.
PARAM_GRID = {
    "fast_window": [10, 21, 30],
    "slow_window": [100, 200],
    "spread_threshold": [0.01, 0.015, 0.025],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, None],
    "trailing": [False, True],
}

ATR_PERIOD = 14

# The half-width of the macro-release veto, in minutes, when `use_news_filter`
# is on. 30 is the repo default. The window is two-sided: it blocks the half
# hour BEFORE a release as well as the half hour after. Blocking the half hour
# before a print is not lookahead — US release SCHEDULES are published a year
# ahead — and nothing in this module ever reads an outcome.
NEWS_WINDOW_MINUTES = 30.0

# The two spread rate-of-change horizons in the Version B matrix, in bars.
SPREAD_ROC_FAST = 5
SPREAD_ROC_SLOW = 15

# The lookback for the volume z-score, matching the shared `causal_features`
# window so the two are directly comparable when somebody asks what the
# strategy-specific matrix bought.
VOLUME_Z_PERIOD = 20

# The Version B feature matrix, in column order. Exactly the six the request
# names.
ML_FEATURES = ["spread_ratio", "spread_roc_5", "spread_roc_15",
               "atr_norm", "volume_z", "hour"]

# Plain-English description for the tear sheet's strategy card, written for a
# reader deciding whether to trade this — not for whoever maintains the module.
# `{param}` slots are filled with the run's own bound parameters, so the card
# states the settings that actually ran rather than the defaults written here.
# The report never infers any of this from the signal arrays: a description
# guessed from the trades would be a guess printed as a fact.
#
# `name` is not read by the report — `_describe_strategy` takes `concept`,
# `entry` and `exit` and ignores everything else — and is carried here because
# the request asks for the identifier to be pinned in this block. The test
# suite pins that it equals `STRATEGY_NAME`.
LOGIC = {
    "name": STRATEGY_NAME,
    "concept": "Institutions move size against an anchor, not against a line "
               "crossing. This strategy (ma_anchoring_spread) therefore "
               "refuses the crossover as a trigger and asks instead HOW FAR "
               "the short-horizon average has pulled away from a slow anchor, "
               "as a percentage of the anchor itself: the Spread Ratio, Fast "
               "SMA ({fast_window}) divided by Slow SMA ({slow_window}). A "
               "crossover fires the instant the two averages touch, which is "
               "the moment of least information and the state a quiet, "
               "low-volume market spends most of its time in. Requiring the "
               "spread to clear a percentage hurdle of {spread_threshold} "
               "says the move was large relative to the anchor, and the "
               "behavioural claim is that a move that large is under-reacted "
               "to rather than fully priced — participants anchor on the "
               "recent range and adjust too slowly. Three layers, EACH "
               "SWITCHABLE, and this run used anchor={use_macro_anchor}, "
               "hurdle={use_spread_hurdle}, expansion="
               "{use_spread_expansion}. The anchor decides which side of the "
               "multi-session imbalance a trade may be on. The hurdle decides "
               "whether the divergence is big enough to be worth paying a "
               "round turn for. The expansion condition decides whether it is "
               "still widening, which is what separates a move under way from "
               "one already collapsing back. Every layer can only REMOVE "
               "eligible bars, so the edge — if there is one — is selection "
               "rather than prediction. The same reasoning runs in both "
               "directions, so the short rules are the long rules mirrored "
               "around 1.0.",
    "entry": "Go long when ALL of the active conditions below hold on the "
             "same bar, and short on their mirror image. Each condition "
             "applies ONLY IF ITS TOGGLE IS TRUE — a toggle set to False "
             "means that condition was not checked at all, not that it "
             "happened to pass. (1) use_macro_anchor={use_macro_anchor}: the "
             "close above the Slow Anchor SMA ({slow_window}) for a long, "
             "below it for a short. (2) use_spread_hurdle="
             "{use_spread_hurdle}: the Spread Ratio at or above "
             "1 + {spread_threshold} for a long, at or below "
             "1 - {spread_threshold} for a short. (3) "
             "use_spread_expansion={use_spread_expansion}: the Spread Ratio "
             "higher than it was on the previous bar for a long, lower for a "
             "short — the divergence still widening rather than closing. (4) "
             "use_news_filter={use_news_filter}: when True, a trigger whose "
             "fill bar lands within 30 minutes either side of a scheduled US "
             "macro release is dropped. This is a STATE, not a crossing: the "
             "conditions stay true across a run of bars, and only the first "
             "of that run becomes a trade because the strategy holds one "
             "position at a time and never pyramids. A signal on the opposite "
             "side while a position is open is ignored rather than reversing "
             "it, and a bar that somehow signals both sides at once takes "
             "neither. The fill is the next bar's open.",
    # Written so that every bound value reads correctly, including
    # `tp_atr_mult=None`. The card cannot branch — `_describe_strategy` only
    # substitutes `{param}` slots — so each sentence states both arms and names
    # the setting that chose between them.
    "exit": "Exit at whichever comes first, and the rules are mirrored for a "
            "short. (1) THE SPREAD REVERTS: the Spread Ratio falls back below "
            "1 + {spread_threshold} x {exit_revert_mult} on a long, or rises "
            "back above 1 - {spread_threshold} x {exit_revert_mult} on a "
            "short — a band strictly INSIDE the entry hurdle, so the trade is "
            "given room to breathe before the reversion counts. This is the "
            "primary exit and it is what makes the strategy flat in quiet "
            "markets. (2) A stop {sl_atr_mult} x ATR 14 away from the fill — "
            "below it on a long, above it on a short — with "
            "trailing={trailing}, where True means it follows the best price "
            "reached since the fill (the high on a long, the low on a short) "
            "and never widens, and False means it sits fixed that far from "
            "the fill price. (3) A take-profit {tp_atr_mult} x ATR 14 from "
            "the fill price — above it on a long, below it on a short — where "
            "None means NO take-profit is modelled at all, which this "
            "strategy only permits alongside a trailing stop. There is no "
            "session flatten, so trades are carried overnight and over "
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
    Wilder's smoothing — the average ATR is actually defined on.

    `alpha = 1/period` with `adjust=False` is Wilder's recursion exactly. A
    span-`period` EMA is a different (roughly twice as fast) average, and using
    it would produce an "ATR(14)" that no other tool agrees with — so a reader
    checking a stop distance against their own chart would get a different
    number.
    """
    return series.ewm(alpha=1.0 / period, adjust=False,
                      min_periods=period).mean()


def _true_range(bars: pd.DataFrame) -> pd.Series:
    """
    True range per bar.

    `.shift(1)` looks one bar BACKWARD, which is the correct direction: the
    true range at bar i uses bar i-1's close and nothing later.

    A zero-range bar produces a true range of 0.0 rather than a NaN — the max
    of three non-negative quantities, all of which are zero when nothing moved
    and the previous close is the current one. That is the request's
    "zero-range degenerate case filled with 0.0", and it matters because a NaN
    here would propagate through the ATR's exponential recursion and blank the
    stop distance for every bar after it.
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
    the first bar, so a "200-period anchor" exists at bar 2 and every anchoring
    call at the start of a symbol's history is decided by the seeding rather
    than by price. Every comparison against NaN is False, so the symptom would
    be silently wrong early trades rather than an error.
    """
    return series.rolling(period, min_periods=period).mean()


def _spread_ratio(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    SMA(fast) / SMA(slow) — the quantity the whole strategy is about.

    A ratio rather than a difference, because the hurdle is a PERCENTAGE of the
    anchor. A 30-point spread is an enormous divergence on GC and noise on NQ,
    so a points hurdle would be a different strategy per contract while the
    leaderboard presented them as one.

    THE DEGENERATE CASE IS LEFT NaN, NOT FILLED WITH 0.0, and this is the one
    place this module declines the request's blanket zero-fill (divergence 1 in
    the module docstring). The denominator is a price average, and the futures
    in this lake are NOT back-adjusted — CL printed negative in April 2020 — so
    a slow anchor at or below zero is reachable rather than hypothetical. At
    exactly zero the ratio is undefined; filled with 0.0 it would sit below
    every short hurdle `1 - spread_threshold`, and the strategy would read "no
    anchor" as "maximum downside divergence" and short it. NaN makes every
    comparison False, so the bar is untradeable and says so.

    A NEGATIVE anchor is excluded for the same reason rather than for a
    numerical one: with `slow < 0` the ratio's sign flips, so "fast average 1.5%
    above the anchor" and "1.5% below it" swap places and the hurdles read
    backwards. Those bars are dropped, not rescued.
    """
    return fast / slow.where(slow > 0)


def _bar_timestamps(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """
    The bar timestamps, from the `ts` column or a DatetimeIndex.

    The engine hands strategies a long-format frame with `ts` as a COLUMN and a
    positional index, so `bars.index.hour` raises there. Accepting both shapes
    keeps `ml_features` usable from the engine path and from a caller holding a
    time-indexed frame, without either one silently producing garbage.

    A duplicate of `agents.tier3_workers._bar_timestamps`, by the same
    convention that duplicates `_wilder` and `_atr` across this directory:
    strategy modules are loaded from a file path and are deliberately
    self-contained. The test suite pins this copy against the original by
    BEHAVIOUR — the same timestamps out of both frame shapes — rather than by
    source, because of the one deliberate difference below.

    The error message names `type(bars.index)` rather than its `__name__`, and
    that is not a style choice: `__name__` is a forbidden dunder access to
    `agents.tier3_workers._audit_ast`, and this module already spends its one
    AST-validator exception on the `backtest.event_calendar` import. A second
    objection would make the pinned list two items long, and a list with slack
    in it stops failing on the edit it exists to catch.
    """
    if "ts" in bars.columns:
        return pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    if isinstance(bars.index, pd.DatetimeIndex):
        idx = bars.index
        return idx if idx.tz is not None else idx.tz_localize("UTC")
    raise ValueError(
        "bars needs a `ts` column or a DatetimeIndex; got an index of type "
        f"{type(bars.index)} and columns {list(bars.columns)}")


def _roc(series: pd.Series, periods: int) -> pd.Series:
    """
    Rate of change over `periods` bars, with the zero-base case filled 0.0.

    `(x[i] - x[i-k]) / |x[i-k]|`. The absolute value in the denominator keeps
    the SIGN of the change the change's own: dividing by a negative base would
    report a widening spread as a shrinking one. It cannot bite on the spread
    ratio, whose base is positive by construction wherever it exists, and it is
    written anyway because this helper is one edit away from being pointed at a
    series that can go negative.

    A base of exactly zero is filled with 0.0 rather than left as the infinity
    the division produces — that is the request's zero-spread rule, and here it
    is the honest reading as well: no base to measure against is no measurable
    change. The fill is conditioned on the base EXISTING, so warm-up stays NaN
    rather than being reported as a flat 0% move.

    `.shift(periods)` looks BACKWARD. Nothing here reads a bar later than i.
    """
    base = series.shift(periods)
    out = (series - base) / base.abs().where(base != 0)
    return out.mask(base.notna() & (base == 0), 0.0)


# --------------------------------------------------------------------------
# The position walk
# --------------------------------------------------------------------------
# The kernel below is COPIED BYTE FOR BYTE from `ema_crossover_20260821.py`, and the
# list of modules its own docstring names predates this one — the copy is
# character-for-character identical and
# `tests/test_ma_anchoring_spread_20260820.py` pins it that way against
# `ema_crossover_20260821._walk_loop`, so editing this copy alone fails loudly. Two
# things it does that this strategy relies on: it anchors the stop, the target
# and the trailing high-water mark on the FILL bar's open rather than the
# signal bar's close, and it enters only from flat — which is what turns this
# module's state-based entry condition into one trade per divergence instead
# of one per bar.
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
def _validate(fast_window: int, slow_window: int, spread_threshold: float,
              exit_revert_mult: float, sl_atr_mult: float,
              tp_atr_mult: float | None, trailing: bool,
              use_macro_anchor: bool = True, use_spread_hurdle: bool = True,
              use_spread_expansion: bool = True,
              use_news_filter: bool = False) -> None:
    """
    Reject parameter sets that do not describe this strategy.

    Every rejection here is recorded by `backtest.scan` as REJECTED and counted
    in the grid, so a combination the strategy refuses shrinks neither the
    reported search nor the honesty of `variants_tested`.
    """
    if fast_window < 2 or slow_window < 2:
        raise ValueError(
            f"windows must be >= 2; got fast={fast_window}, slow={slow_window}")
    if fast_window >= slow_window:
        # Not a stylistic objection. With the two equal the spread ratio is
        # identically 1.0 and no hurdle is ever cleared in either direction —
        # a strategy that cannot trade, reported as one that found nothing.
        # Inverted, the "fast" average is the slower of the two and the ratio
        # measures the anchor's divergence from the trigger, which is the same
        # arithmetic describing a different idea. Either way the equity curve
        # would look plausible.
        raise ValueError(
            f"fast_window must be < slow_window; got {fast_window} >= "
            f"{slow_window}")
    if spread_threshold is None or not np.isfinite(float(spread_threshold)):
        raise ValueError(
            f"spread_threshold must be a finite number; got "
            f"{spread_threshold!r}")
    if not 0.0 < float(spread_threshold) < 1.0:
        # It is a FRACTION of the anchor, not a percentage and not a points
        # distance. At 0 the hurdle is no hurdle at all — every bar on the
        # correct side of 1.0 clears it — and at 1.0 or above the short hurdle
        # `1 - spread_threshold` is at or below zero, which a positive ratio
        # can never reach: the strategy would be silently long-only. Somebody
        # passing 1.5 meaning "1.5 percent" gets an error rather than a
        # one-sided backtest.
        raise ValueError(
            f"spread_threshold must be a fraction strictly within (0, 1) — "
            f"0.015 is 1.5%; got {spread_threshold!r}")
    if exit_revert_mult is None or not np.isfinite(float(exit_revert_mult)):
        raise ValueError(
            f"exit_revert_mult must be a finite number; got "
            f"{exit_revert_mult!r}")
    if not 0.0 <= float(exit_revert_mult) <= 1.0:
        # The exit band has to sit INSIDE the entry hurdle. Above 1.0 the band
        # is wider than the hurdle, so a fresh entry is already inside its own
        # exit condition and the walk — which checks exits from the fill bar
        # onward — closes it one bar later, every time. That is a one-bar-trade
        # machine paying two round turns for each, and it would report as a
        # strategy with a poor win rate rather than as a misconfiguration.
        # Zero is allowed and means "exit only on a full reversion to parity".
        raise ValueError(
            f"exit_revert_mult must be within [0, 1] so the exit band sits "
            f"inside the entry hurdle; got {exit_revert_mult!r}")
    if sl_atr_mult is None or sl_atr_mult <= 0:
        raise ValueError(f"sl_atr_mult must be > 0; got {sl_atr_mult!r}")
    if tp_atr_mult is not None and tp_atr_mult <= 0:
        # None is the no-take-profit configuration and is accepted, subject to
        # the trailing requirement below. A non-positive number is not: a
        # target at or below the fill price would be breached by the fill bar
        # itself.
        raise ValueError(
            f"tp_atr_mult must be > 0, or None for no take-profit; got "
            f"{tp_atr_mult!r}")
    if not isinstance(trailing, (bool, np.bool_)):
        # Truthiness would silently accept "false" (a non-empty string, so
        # True) and 0.0, and the stop would trail or not trail for reasons
        # invisible in the leaderboard's params column.
        raise ValueError(f"trailing must be a bool; got {trailing!r}")
    if tp_atr_mult is None and not bool(trailing):
        # The request's rule, and the reason for it is the runner lockup. With
        # no target and a FIXED stop, a position the market never retraces to
        # is closed only by the spread reverting — and in a trend that keeps
        # widening, that can be thousands of bars, or never. The trade is then
        # carried to the end of the frame, where it is not a completed trade at
        # all: it contributes no exit, no realised P&L and no ML training
        # label, while its unrealised path drags the equity curve. A TRAILING
        # stop cannot lock up that way — it ratchets in behind the price and is
        # eventually hit. `PARAM_GRID` pairs both `tp_atr_mult` values with
        # both `trailing` values, so this rejects 54 of its 324 cells; the
        # scanner counts them.
        raise ValueError(
            "tp_atr_mult=None (no take-profit) requires trailing=True: with a "
            "fixed stop and no target a runner has no bounded exit and is "
            "carried to the end of the data")
    # The same objection as `trailing`, and it bites harder here: `--param
    # use_macro_anchor=false` passed as the STRING "false" is truthy, so the
    # run would apply the anchor while the leaderboard's params column said it
    # was off. Every toggle is checked rather than coerced.
    for name, flag in (("use_macro_anchor", use_macro_anchor),
                       ("use_spread_hurdle", use_spread_hurdle),
                       ("use_spread_expansion", use_spread_expansion),
                       ("use_news_filter", use_news_filter)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be a bool; got {flag!r}")


def _series(bars: pd.DataFrame, fast_window: int,
            slow_window: int) -> dict:
    """
    The shared calculation behind `signal_fn`, `indicators`, `ml_features` and
    the walk.

    One place, so the hurdle line the tear sheet draws is the array the entry
    was taken from and the spread ratio the classifier reads is the one the
    rule read. Separate call sites computing this would be free to drift apart
    with nothing raising.
    """
    close = bars["close"]
    fast = _sma(close, fast_window)
    slow = _sma(close, slow_window)
    return {
        "fast": fast,
        "slow": slow,
        "ratio": _spread_ratio(fast, slow),
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
                   spread_threshold: float, exit_revert_mult: float,
                   sl_atr_mult: float, tp_atr_mult: float | None,
                   trailing: bool, use_macro_anchor: bool = True,
                   use_spread_hurdle: bool = True,
                   use_spread_expansion: bool = True,
                   use_news_filter: bool = False) -> tuple:
    """
    Everything the walk produces, from one place.

    `signal_fn` and `indicators` both go through here, so the stop line the
    tear sheet draws is the array the exit was taken from and the hurdle a
    reader sees cleared is the array the entry came from.
    """
    s = _series(bars, fast_window, slow_window)
    close = bars["close"]
    ratio = s["ratio"]

    # `ready` is assembled FROM THE ACTIVE FILTERS ONLY, so a toggled-off layer
    # costs nothing, including its warm-up. It matters less here than in a
    # module whose layers use different indicators — all three layers read the
    # same two averages, so the only toggle-dependent warm-up is Layer 3's
    # one-bar lookback — but it is assembled the same way so that adding a
    # layer with its own series later cannot quietly lengthen the warm-up of
    # the cells that switch it off.
    #
    # Three things are unconditional. Both averages are the spread ratio, which
    # every layer and the exit rule read. ATR is load-bearing whatever the
    # layers say: it sets the stop distance and the target, so an entry taken
    # before ATR exists would have no stop.
    ready = s["fast"].notna() & s["slow"].notna() & s["atr"].notna()
    # `ratio` is NaN wherever the anchor is not strictly positive, which is a
    # tradeability question rather than a warm-up one — see `_spread_ratio`.
    ready &= ratio.notna()
    if use_spread_expansion:
        ready &= ratio.shift(1).notna()

    # A DISABLED LAYER IS ALL-TRUE, NOT ALL-FALSE. `_on` returns a True Series
    # when its toggle is off, so the conjunction below is written once and
    # reads the same whichever layers are active. Writing it as `cond if flag
    # else <omit>` instead would need a different expression per combination,
    # and each combination would be another chance for the long and short arms
    # to stop mirroring each other.
    def _on(condition: pd.Series, flag: bool) -> pd.Series:
        return condition if flag else pd.Series(True, index=bars.index)

    # Layer 1 — the anchor. Each side is an explicit comparison rather than the
    # negation of its opposite: `~(close > slow)` is true wherever the close
    # merely fails to be above the anchor, which includes exact equality and
    # every warm-up bar, so a short condition written as a negation would fire
    # on ties and on bars where the average does not exist.
    long_anchor = _on(close > s["slow"], use_macro_anchor)
    short_anchor = _on(close < s["slow"], use_macro_anchor)

    # Layer 2 — the percentage hurdle. Inclusive (`>=`, `<=`) exactly as the
    # specification writes it: at the hurdle the divergence has been achieved.
    threshold = float(spread_threshold)
    long_hurdle = _on(ratio >= 1.0 + threshold, use_spread_hurdle)
    short_hurdle = _on(ratio <= 1.0 - threshold, use_spread_hurdle)

    # Layer 3 — expansion. STRICTLY greater and strictly less: a spread that
    # held exactly still is not widening, and `>=` would let a flat ratio — the
    # state a stalled move sits in — pass as momentum on both sides at once.
    prev_ratio = ratio.shift(1)
    long_expanding = _on(ratio > prev_ratio, use_spread_expansion)
    short_expanding = _on(ratio < prev_ratio, use_spread_expansion)

    long_entry_ok = (long_anchor & long_hurdle & long_expanding
                     & ready).to_numpy(dtype=bool)
    short_entry_ok = (short_anchor & short_hurdle & short_expanding
                      & ready).to_numpy(dtype=bool)

    # The news veto lands on CANDIDATES, before the walk, so a blocked trigger
    # leaves the strategy flat and free to take a later one. See the module
    # docstring for why that differs from the engine's own `--news-filter`,
    # which acts on the walk's output.
    if use_news_filter:
        long_entry_ok, short_entry_ok = _news_suppress(
            bars, long_entry_ok, short_entry_ok)

    # Layer 4 — the primary exit. The band is INSIDE the hurdle, so a trade is
    # not closed the moment the spread stops widening; it is closed when the
    # spread has given back `1 - exit_revert_mult` of the divergence that
    # opened it.
    #
    # A NaN ratio makes both comparisons False, so a bar where the anchor is
    # not positive produces no signal exit. That is deliberate: the stop and
    # the target still apply, and closing a position because an indicator went
    # undefined would be an exit at a price nobody decided on.
    band = threshold * float(exit_revert_mult)
    long_sig_exit = (ratio < 1.0 + band).to_numpy(dtype=bool)
    short_sig_exit = (ratio > 1.0 - band).to_numpy(dtype=bool)

    # No session flatten. `flat_bar` all-False is what "positions are carried
    # overnight" looks like in this kernel, and it is a modelled exposure
    # rather than an omission — see the module docstring.
    flat_bar = np.zeros(len(bars), dtype=bool)

    entries, exits, s_entries, s_exits, stop, target = _walk(
        long_entry_ok,
        short_entry_ok,
        long_sig_exit,
        short_sig_exit,
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
              fast_window: int = 21,
              slow_window: int = 200,
              spread_threshold: float = 0.015,
              exit_revert_mult: float = 0.5,
              use_macro_anchor: bool = True,
              use_spread_hurdle: bool = True,
              use_spread_expansion: bool = True,
              use_news_filter: bool = False,
              sl_atr_mult: float = 1.5,
              tp_atr_mult: float | None = 3.0,
              trailing: bool = False) -> tuple[pd.Series, pd.Series,
                                               pd.Series, pd.Series]:
    """
    Trade the spread between a fast average and a slow anchor once it has
    cleared a percentage hurdle and is still widening. Both directions.

    Returns the FOUR-MASK form of the strategy contract:

        (long_entries, long_exits, short_entries, short_exits)

    all boolean Series on `bars.index`. `backtest.engine.unpack_signals`
    accepts this alongside the older two-mask long-only form.

    The entry condition is a STATE rather than an event — it holds across a run
    of bars — and the walk enters only when flat, so that run produces one
    trade on its first bar rather than one per bar. A signal on the opposite
    side while a position is open is ignored rather than reversing it.

    Every comparison at bar i uses only bars <= i. The engine then fills at bar
    i+1's open, so nothing here can see a price it would not have had.
    """
    _validate(fast_window, slow_window, spread_threshold, exit_revert_mult,
              sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor,
              use_spread_hurdle, use_spread_expansion, use_news_filter)

    _s, entries, exits, s_entries, s_exits, _stop, _target = _signal_arrays(
        bars, fast_window, slow_window, spread_threshold, exit_revert_mult,
        sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor,
        use_spread_hurdle, use_spread_expansion, use_news_filter)

    return (pd.Series(entries, index=bars.index),
            pd.Series(exits, index=bars.index),
            pd.Series(s_entries, index=bars.index),
            pd.Series(s_exits, index=bars.index))


def indicators(bars: pd.DataFrame,
               fast_window: int = 21,
               slow_window: int = 200,
               spread_threshold: float = 0.015,
               exit_revert_mult: float = 0.5,
               use_macro_anchor: bool = True,
               use_spread_hurdle: bool = True,
               use_spread_expansion: bool = True,
               use_news_filter: bool = False,
               sl_atr_mult: float = 1.5,
               tp_atr_mult: float | None = 3.0,
               trailing: bool = False) -> dict[str, pd.Series]:
    """
    The price-scale series, for the tear sheet to draw over the candles.

    Computed HERE, the same way and from the same columns `signal_fn` reads, so
    the divergence a reader sees is the array the entry was taken from. A
    second implementation living in the report would be free to disagree with
    this one — a chart showing the hurdle cleared a bar away from where the
    trade fired, with nothing raising.

    THE HURDLE IS DRAWN IN PRICE TERMS, which is the whole reason this hook is
    worth reading. The rule is about a ratio, and a ratio cannot be drawn on a
    price axis — but `slow * (1 + spread_threshold)` is the price the FAST
    AVERAGE has to exceed for a long, so the entry condition becomes "the fast
    line above the amber one" and a reader can check a trade by eye. The short
    hurdle is its mirror. Both are omitted when `use_spread_hurdle` is off:
    drawing a line the entries did not respect makes the trades that ignore it
    look like bugs rather than like the strategy that was actually run.

    THE EXIT BAND IS NOT DRAWN, and that is a real gap. As price lines it is
    two more series (`slow * (1 +/- spread_threshold * exit_revert_mult)`),
    which would make eight lines on a chart whose palette carries six distinct
    colour-and-dash styles — the seventh repeats the first, so two different
    lines would be told apart by nothing. The band sits between the anchor and
    the hurdle lines already drawn, and the logic card states its arithmetic
    with the run's own numbers. If the exit is what somebody is investigating,
    swap it in here for the take-profit line rather than adding it.

    The stop and the target come from the same `_walk` the signals come from,
    through the same `_signal_arrays`, so they cannot disagree either. Both are
    NaN while flat and the report renders that as a gap, which is the honest
    drawing: there is no stop level when there is no position. ONE stop line
    and ONE target line serve both directions, because only one position is
    ever open: a segment that jumps from below the price to above it is the
    strategy changing sides, not a stop being moved.

    The take-profit line is OMITTED ENTIRELY when `tp_atr_mult` is None. An
    all-NaN series would render as an empty legend entry, which reads as a
    target that exists and never got close — the opposite of the truth.

    Warm-up stays NaN rather than drawing the averages flat through the first
    bars.
    """
    _validate(fast_window, slow_window, spread_threshold, exit_revert_mult,
              sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor,
              use_spread_hurdle, use_spread_expansion, use_news_filter)

    s, _entries, _exits, _se, _sx, stop, target = _signal_arrays(
        bars, fast_window, slow_window, spread_threshold, exit_revert_mult,
        sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor,
        use_spread_hurdle, use_spread_expansion, use_news_filter)

    kind = "Trailing" if trailing else "Fixed"
    pct = f"{float(spread_threshold) * 100:g}%"
    out = {
        f"Fast SMA ({fast_window})": s["fast"],
        f"Slow Anchor SMA ({slow_window})": s["slow"],
        f"{kind} Stop ({sl_atr_mult}xATR{ATR_PERIOD})":
            pd.Series(stop, index=bars.index),
    }
    if use_spread_hurdle:
        out[f"Long Hurdle (Anchor +{pct})"] = s["slow"] * (
            1.0 + float(spread_threshold))
        out[f"Short Hurdle (Anchor -{pct})"] = s["slow"] * (
            1.0 - float(spread_threshold))
    if tp_atr_mult is not None:
        out[f"Take Profit ({tp_atr_mult}xATR{ATR_PERIOD})"] = pd.Series(
            target, index=bars.index)
    return out


def ml_features(bars: pd.DataFrame,
                fast_window: int = 21,
                slow_window: int = 200,
                **_params) -> pd.DataFrame:
    """
    Version B's feature matrix: the spread state, its velocity, volatility,
    participation and time of day.

    Read by `agents.tier3_workers.load_strategy` and handed to
    `apply_ml_signal_filter` in place of the shared `causal_features` default.
    Declaring it here is the point: a classifier vetoing THIS strategy's
    entries should be reading the state this strategy's hypothesis is about —
    how far the spread has diverged and how fast — rather than a
    general-purpose matrix built around RSI and return horizons.

    Columns, in `ML_FEATURES` order:

        spread_ratio   SMA(fast) / SMA(slow), the quantity the entry rule
                       thresholds. Under the default `use_spread_hurdle=True`
                       every surviving candidate has already cleared the
                       hurdle, so what this column carries is HOW FAR past it
                       the bar was — which is the part the veto can still act
                       on. Scale-free by construction, so a model fitted across
                       sixteen years of a contract whose price level multiplied
                       reads the same number for the same divergence.
        spread_roc_5   The ratio's 5-bar rate of change: is the divergence
        spread_roc_15  accelerating, and over what horizon. Two horizons rather
                       than one because a spread that widened fast and has
                       started to level off is a different state from one still
                       accelerating, and a single ROC cannot separate them.
        atr_norm       ATR(14) / close. Normalised for the same reason the
                       ratio is scale-free: a raw ATR in points would let the
                       model learn "2015" instead of "quiet".
        volume_z       Volume against its own 20-bar mean and standard
                       deviation. This is the only place volume enters the
                       strategy at all — Version A has no volume condition —
                       and it is what the "low-volume chop" half of the premise
                       is measured on. Standardised on a ROLLING window rather
                       than a global one; see the causality note below.
        hour           Hour of day, UTC, from the `ts` column. An integer-coded
                       hour is a legitimate feature for a tree model, which
                       splits on it rather than treating it as a magnitude, and
                       it is how the shared `causal_features` encodes it too.
                       It is the one column here that is not about the spread,
                       and it is in the request because a 15m strategy's edge
                       is not uniform across the session.

    THIS HOOK DEPENDS ON THE SWEPT PARAMETERS, and it is the only one in this
    tree that does. `spread_ratio` and both ROC columns are functions of
    `fast_window` and `slow_window`, which `--scan` sweeps — so a grid run with
    `--ml` searches over CLASSIFIERS as well as over strategies, and two cells'
    Version B results are not comparable on a shared feature definition. The
    alternative was fixing the feature windows at the defaults, which would
    have the classifier read a spread the rule never looked at. The parameters
    are the lesser evil because they are RECORDED: `metrics["meta"]["params"]`
    carries the windows onto every report and leaderboard row, beside
    `metrics["meta"]["ml_features"]`'s column names, so the matrix a Version B
    was fitted on is reconstructable rather than assumed.

    STRICTLY CAUSAL, and worth being explicit about because this matrix is
    fitted on. A value at row i is a function of bars 0..i only: no `shift(-k)`,
    no centred window, no reversed slice, and nothing computed off a
    full-sample statistic. That last one is the trap a shift-based audit does
    not catch — a scaler fitted on the whole frame leaks the test period's
    distribution into the training rows — which is why `volume_z` standardises
    on a rolling window rather than on the column's own mean.

    Using bar i's close to decide a signal on bar i is legitimate: the engine
    fills at bar i+1's open, never on the signal bar. That one-bar gap is what
    makes these features tradeable rather than clairvoyant.

    NaN warm-up rows are left as NaN. `HistGradientBoostingClassifier` consumes
    them natively, and filling them with a column mean would import a
    full-sample statistic into exactly the rows that have no history. The 0.0
    fills that ARE applied — a zero-base rate of change, a dead-flat volume
    window — are measurements rather than gaps, and both are conditioned on
    their inputs existing so warm-up is not swept into them.
    """
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)

    s = _series(bars, fast_window, slow_window)
    ratio = s["ratio"]

    vol_mean = volume.rolling(VOLUME_Z_PERIOD,
                              min_periods=VOLUME_Z_PERIOD).mean()
    vol_std = volume.rolling(VOLUME_Z_PERIOD,
                             min_periods=VOLUME_Z_PERIOD).std()

    out = pd.DataFrame({
        "spread_ratio": ratio.to_numpy(dtype=float),
        "spread_roc_5": _roc(ratio, SPREAD_ROC_FAST).to_numpy(dtype=float),
        "spread_roc_15": _roc(ratio, SPREAD_ROC_SLOW).to_numpy(dtype=float),
        "atr_norm": (s["atr"] / close.where(close != 0)).to_numpy(dtype=float),
        # The 1e-8 keeps a dead-flat volume window from dividing by zero. It is
        # far below any real standard deviation, so it changes no live value,
        # and a flat window then reports z = 0.0 — which is the honest reading:
        # this bar's volume is exactly its recent average.
        "volume_z": ((volume - vol_mean)
                     / (vol_std + 1e-8)).to_numpy(dtype=float),
        "hour": _bar_timestamps(bars).hour.to_numpy(dtype=float),
    }, index=bars.index)
    return out[ML_FEATURES]


def make_signal_fn(fast_window: int = 21,
                   slow_window: int = 200,
                   spread_threshold: float = 0.015,
                   exit_revert_mult: float = 0.5,
                   use_macro_anchor: bool = True,
                   use_spread_hurdle: bool = True,
                   use_spread_expansion: bool = True,
                   use_news_filter: bool = False,
                   sl_atr_mult: float = 1.5,
                   tp_atr_mult: float | None = 3.0,
                   trailing: bool = False):
    """
    Bind parameters for `agents.tier3_workers.load_strategy`.

    The loader prefers this factory when params are supplied; the engine itself
    only ever calls the bound `signal_fn(bars)`. Validation runs HERE as well
    as inside `signal_fn`, so `--scan` records a bad combination as REJECTED at
    bind time rather than discovering it one symbol into the sweep.
    """
    _validate(fast_window, slow_window, spread_threshold, exit_revert_mult,
              sl_atr_mult, tp_atr_mult, trailing, use_macro_anchor,
              use_spread_hurdle, use_spread_expansion, use_news_filter)

    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series,
                                            pd.Series, pd.Series]:
        return signal_fn(bars, fast_window=fast_window,
                         slow_window=slow_window,
                         spread_threshold=spread_threshold,
                         exit_revert_mult=exit_revert_mult,
                         use_macro_anchor=use_macro_anchor,
                         use_spread_hurdle=use_spread_hurdle,
                         use_spread_expansion=use_spread_expansion,
                         use_news_filter=use_news_filter,
                         sl_atr_mult=sl_atr_mult,
                         tp_atr_mult=tp_atr_mult,
                         trailing=trailing)

    return _bound
