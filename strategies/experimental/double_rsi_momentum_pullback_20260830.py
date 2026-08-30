"""
DOUBLE_RSI_MOMENTUM_PULLBACK — a fast-RSI pullback taken in the direction of a
slow-RSI trend, confirmed by a volume expansion and a volatility floor.

    RSI(21) > 50            the intermediate trend is up  (Layer 1)
    RSI(5) dipped < 50      within `pullback_window` bars (Layer 2)
    RSI(5) crosses ABOVE RSI(21)                          (Layer 2, the event)
    volume > SMA(20) * 1.05 and ATR/close > 0.0005        (Layer 3)
                                                          shorts mirror all four

Exit on the fast RSI reaching its extreme (80 long, 20 short), or on the ATR
stop or target, whichever the bar reaches first.

WHAT THIS MODULE DOES NOT DO
============================
1. **It does not choose its regime.** The request nominates Q1/Q2. Nothing here
   reads or asserts a quadrant: Stage 1 measures all four and DESIGNATES the
   one with the highest alpha score, and that designation is evidence rather
   than intent. A module that filtered to a nominated quadrant would make the
   screen agree with the hypothesis by construction.

   The nominated labels also do not match this repository's numbering.
   `mdlib/regimes.py` is the only authority: Q1 High-Vol/Trending, Q2
   High-Vol/Ranging, Q3 Low-Vol/Trending, Q4 Low-Vol/Ranging. The request's
   "Q1 (Low Vol / Trending)" is Q3 here and its "Q2 (High Vol / Trending)" is
   Q1. Both are trending quadrants, so the ECONOMIC premise survives the
   correction; the numbers do not.

2. **It does not handle costs, fills or sizing.** The masks are events on the
   bar that closed. `backtest.engine` fills at the NEXT bar's open and charges
   commission and a tick of slippage each way. Nothing here can see a price it
   would not have had.

3. **The brackets are not a risk system.** `sl_atr_mult` and `tp_atr_mult`
   place a stop and a target inside the walk, evaluated against the bar's own
   high and low. Intrabar order is unknowable from OHLC, so a bar touching both
   is resolved stop-first — the pessimistic reading, and still an assumption.
   Account-level drawdown, daily loss and prop-firm state belong to CrossTrade,
   not here.

4. **It is not gated to a session window.** The request's Layer 1 names a
   "liquid intraday window" without bounds. Inventing hours would put an
   unmeasured, unswept filter in front of every entry, and the one measured
   fact about filters in this repository is that they change the trade list in
   ways a trade count hides. The volume and volatility floors in Layer 3 are
   what stand in for liquidity here, and both are swept.

THE SHORT SIDE IS NOT THE LONG SIDE MIRRORED
============================================
The stop sits ABOVE a short fill and the target BELOW it; the trailing stop
ratchets DOWN against the low-water mark. `_walk_loop` encodes that. A short
stop placed below the fill would be breached by the fill bar itself.
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

#: The request's "Equity_Momentum_Basket", spelled as contracts this lake
#: serves and `backtest/specs.py` has verified. NQ and ES are the index legs;
#: RTY and YM are the same factor at different capitalisations, which is what
#: makes the basket POSITIVELY correlated rather than a hedge - a fact for
#: portfolio routing to act on, not for this module to correct.
SYMBOLS = ["NQ", "ES", "RTY", "YM"]

#: What the spec calls the strategy, for the tear sheet and the Discord card.
STRATEGY_NAME = "DOUBLE_RSI_MOMENTUM_PULLBACK"
STRATEGY_MODULE = "double_rsi_momentum_pullback_20260830"

#: Portfolio routing metadata. Descriptive only - `config/portfolios.json` is
#: the authority on which account trades what, and no basket named here exists
#: until somebody adds it there.
PORTFOLIO_GROUP = "Equity_Momentum_Basket"
CORRELATION_PROFILE = (
    "Positively correlated with the index group (MNQ/MES). NQ, ES, RTY and YM "
    "are one factor at four capitalisations, so a basket of them concentrates "
    "rather than diversifies - a fact for portfolio routing to size against.")

#: The regimes the premise NOMINATES. Stage 1 designates the real one by
#: measuring all four quadrants, and nothing in this module reads these.
#:
#: The request named "Q1 (Low Vol / Trending)" and "Q2 (High Vol / Trending)".
#: Neither number matches `mdlib/regimes.py`, which is the only authority:
#: Q1 High-Vol/Trending, Q2 High-Vol/Ranging, Q3 Low-Vol/Trending, Q4
#: Low-Vol/Ranging. The two REGIMES the request describes are both trending,
#: so the economic premise stands; the ids are corrected here.
TARGET_REGIMES = ("Low Volatility / Trending", "High Volatility / Trending")
TARGET_QUADRANTS = ("Q3", "Q1")

ATR_PERIOD = 14

#: An RSI whose gain and loss both average zero is 0/0 - undefined rather than
#: oversold. Neutral is the honest reading, and it keeps a flat stretch of bars
#: from registering as an extreme.
RSI_NEUTRAL = 50.0

#: Layer 3's volatility floor: ATR as a fraction of price. A floor on RAW ATR
#: would mean something different on NQ at 20,000 than on RTY at 2,000, so the
#: request's 0.0005 is read as normalised - five basis points of range.
MIN_NORM_ATR = 0.0005

#: `_validate` refuses a target closer than half the stop. A grid cell that
#: cannot clear its own costs is not a strategy this module will describe, and
#: a REJECTED cell is still counted in `variants_tested` - refusing it shrinks
#: neither the search nor the honesty of the number reported for it.
MIN_REWARD_RISK = 0.5
MIN_STOP_ATR_MULT = 0.25

DEFAULT_PARAMS = {
    "rsi_fast_len": 5,
    "rsi_slow_len": 21,
    "pullback_window": 3,
    "volume_sma_len": 20,
    "volume_mult": 1.05,
    "rsi_exit_long": 80.0,
    "rsi_exit_short": 20.0,
    "use_baseline_filter": True,
    "use_alpha_trigger": True,
    "use_volume_filter": True,
    "use_news_filter": False,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "trailing": False,
}

#: The request's grid, transcribed. COUNT IT BEFORE RUNNING IT:
#:
#:     3 x 3 x 3 x 3 x 3 x 3 x 2 = 1,458 combinations
#:
#: That is SEVEN TIMES the ~200-cell bound this repository holds its grids to,
#: and the bound is not a style rule. The reported Sharpe is the maximum of
#: that many draws from one sample of bars, and the maximum of a sample climbs
#: with N whether or not anything in the market has changed.
#:
#: Multiply before quoting it: `--tf 5m,15m,30m,1h` is 1,458 fits PER timeframe
#: PER contract - 5,832 per symbol, 23,328 across the four declared assets.
#:
#: It is transcribed rather than trimmed because the grid is the request's to
#: set and `variants_tested` carries the count into every artifact, so the
#: search is at least reported honestly. A 162-cell version testing the same
#: hypothesis is in the module docstring's companion note; prefer it.
PARAM_GRID = {
    "rsi_fast_len": [3, 5, 7],
    "rsi_slow_len": [14, 21, 28],
    "pullback_window": [2, 3, 5],
    "volume_mult": [1.0, 1.1, 1.25],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, None],
    "trailing": [False, True],
}

LOGIC = {
    "concept": ("A pullback taken in the direction of the intermediate trend. "
                "The slow RSI ({rsi_slow_len}) says which way the tape leans; "
                "the fast RSI ({rsi_fast_len}) dipping through 50 and turning "
                "back through the slow line is the entry event; a volume "
                "expansion and a volatility floor say the turn happened on "
                "participation rather than on a thin print."),
    "entry": ("LONG when RSI({rsi_slow_len}) > 50, RSI({rsi_fast_len}) was "
              "below 50 within the last {pullback_window} bars and now crosses "
              "ABOVE RSI({rsi_slow_len}), with volume > SMA({volume_sma_len}) "
              "x {volume_mult} and ATR/close > 0.0005. SHORT mirrors it."),
    "exit": ("RSI({rsi_fast_len}) >= {rsi_exit_long} for a long or "
             "<= {rsi_exit_short} for a short, or the ATR stop "
             "({sl_atr_mult}x) or target ({tp_atr_mult}x), whichever the bar "
             "reaches first."),
}


# ---------------------------------------------------------------------------
# Indicator kernels
#
# DUPLICATED VERBATIM from `double_rsi_macd_scalp_20260823.py`, which shares
# them with the rest of this directory. Strategy modules are loaded from a FILE
# PATH by `agents.tier3_workers.load_strategy` and promoted as a self-contained
# copy, so a shared import would resolve against whatever happens to sit beside
# the module at load time - and a promoted package must reproduce the file that
# was certified, byte for byte, not whatever a helper module has become since.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# This strategy
# ---------------------------------------------------------------------------

def _validate(rsi_fast_len: int, rsi_slow_len: int, pullback_window: int,
              volume_sma_len: int, volume_mult: float,
              rsi_exit_long: float, rsi_exit_short: float,
              sl_atr_mult: float, tp_atr_mult: float | None,
              trailing: bool) -> None:
    """
    Refuse parameter sets that do not describe this strategy.

    A rejection here is recorded by `backtest.scan` as REJECTED and still
    counted in the grid, so refusing a cell shrinks neither the reported search
    nor the honesty of `variants_tested`.
    """
    if rsi_fast_len < 2:
        raise ValueError(f"rsi_fast_len must be >= 2; got {rsi_fast_len}")
    if rsi_slow_len < 2:
        raise ValueError(f"rsi_slow_len must be >= 2; got {rsi_slow_len}")
    if rsi_fast_len >= rsi_slow_len:
        # The premise is a FAST oscillator pulling back inside a SLOWER trend.
        # Equal or inverted lengths describe a different strategy that would
        # still produce trades and a plausible curve.
        raise ValueError(
            f"rsi_fast_len ({rsi_fast_len}) must be < rsi_slow_len "
            f"({rsi_slow_len}): the pullback is the FAST line returning to a "
            f"slower trend")
    if pullback_window < 1:
        raise ValueError(
            f"pullback_window must be >= 1; got {pullback_window}")
    if volume_sma_len < 2:
        raise ValueError(f"volume_sma_len must be >= 2; got {volume_sma_len}")
    if volume_mult <= 0:
        raise ValueError(f"volume_mult must be > 0; got {volume_mult}")
    if not 50.0 < rsi_exit_long <= 100.0:
        raise ValueError(
            f"rsi_exit_long must be in (50, 100]; got {rsi_exit_long}")
    if not 0.0 <= rsi_exit_short < 50.0:
        raise ValueError(
            f"rsi_exit_short must be in [0, 50); got {rsi_exit_short}")
    if sl_atr_mult < MIN_STOP_ATR_MULT:
        raise ValueError(
            f"sl_atr_mult must be >= {MIN_STOP_ATR_MULT}; got {sl_atr_mult}. "
            f"A stop inside the noise is hit by the fill bar.")
    if tp_atr_mult is not None:
        if tp_atr_mult <= 0:
            raise ValueError(f"tp_atr_mult must be > 0; got {tp_atr_mult}")
        if tp_atr_mult / sl_atr_mult < MIN_REWARD_RISK:
            raise ValueError(
                f"tp_atr_mult / sl_atr_mult = "
                f"{tp_atr_mult / sl_atr_mult:.2f} is below MIN_REWARD_RISK "
                f"{MIN_REWARD_RISK}: a target nearer than half the stop "
                f"cannot clear its own costs often enough to matter")
    if not isinstance(trailing, (bool, np.bool_)):
        raise ValueError(f"trailing must be a bool; got {trailing!r}")


def _layers(bars: pd.DataFrame, rsi_fast_len: int, rsi_slow_len: int,
            pullback_window: int, volume_sma_len: int, volume_mult: float,
            use_baseline_filter: bool, use_alpha_trigger: bool,
            use_volume_filter: bool) -> dict[str, pd.Series]:
    """
    Every layer as its own mask, computed once and shared by `signal_fn`,
    `indicators` and `ml_features`.

    ONE implementation, because a second one living in the report would be free
    to disagree with this one and draw a crossover a bar from where the trade
    actually fired, with nothing raising.

    Every comparison at bar i reads only bars <= i. `.rolling` and `.shift(1)`
    are the whole of the causality argument: the cross compares this bar's
    values against the PREVIOUS bar's, and the pullback looks backwards over a
    closed window.
    """
    close = bars["close"].astype("float64")
    fast = _rsi(close, rsi_fast_len)
    slow = _rsi(close, rsi_slow_len)
    atr = _atr(bars, ATR_PERIOD)

    # Layer 1 - the intermediate trend. A toggle that is OFF means "no opinion"
    # and admits both sides, never "the opposite".
    if use_baseline_filter:
        trend_long = slow > 50.0
        trend_short = slow < 50.0
    else:
        trend_long = pd.Series(True, index=bars.index)
        trend_short = pd.Series(True, index=bars.index)

    # Layer 2 - the event. The fast line dipped through 50 within the window
    # AND now crosses the slow line. `.shift(1)` on the rolling max/min keeps
    # the CURRENT bar out of its own lookback: a bar that both dips and crosses
    # would otherwise satisfy the pullback with itself.
    dipped = (fast < 50.0).rolling(pullback_window, min_periods=1).max()
    popped = (fast > 50.0).rolling(pullback_window, min_periods=1).max()
    dipped = dipped.shift(1).fillna(0).astype(bool)
    popped = popped.shift(1).fillna(0).astype(bool)
    cross_up = _cross_above(fast, slow)
    cross_dn = _cross_below(fast, slow)
    if use_alpha_trigger:
        trigger_long = dipped & cross_up
        trigger_short = popped & cross_dn
    else:
        trigger_long, trigger_short = cross_up, cross_dn

    # Layer 3 - participation and volatility. Volume is compared against a mean
    # that EXCLUDES the current bar, for the same reason: a bar cannot be part
    # of the average it has to beat.
    if "volume" in bars.columns:
        volume = bars["volume"].astype("float64")
        vol_sma = volume.rolling(volume_sma_len, min_periods=volume_sma_len
                                 ).mean().shift(1)
        vol_ok = volume > (vol_sma * float(volume_mult))
    else:
        # A lake read without a volume column is a MISSING FILTER, not a
        # passing one. Refusing every entry is the direction that cannot
        # invent trades the confirmation never cleared.
        vol_ok = pd.Series(False, index=bars.index)
    norm_atr = atr / close.replace(0.0, np.nan)
    vola_ok = norm_atr > MIN_NORM_ATR
    confirm = (vol_ok & vola_ok) if use_volume_filter else pd.Series(
        True, index=bars.index)

    return {"fast": fast, "slow": slow, "atr": atr, "norm_atr": norm_atr,
            "trend_long": trend_long.fillna(False),
            "trend_short": trend_short.fillna(False),
            "trigger_long": trigger_long.fillna(False),
            "trigger_short": trigger_short.fillna(False),
            "confirm": confirm.fillna(False)}


def signal_fn(bars: pd.DataFrame,
              rsi_fast_len: int = 5,
              rsi_slow_len: int = 21,
              pullback_window: int = 3,
              volume_sma_len: int = 20,
              volume_mult: float = 1.05,
              rsi_exit_long: float = 80.0,
              rsi_exit_short: float = 20.0,
              use_baseline_filter: bool = True,
              use_alpha_trigger: bool = True,
              use_volume_filter: bool = True,
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

    The entry is a bar-level EVENT inside standing conditions, so a long
    stretch of bars satisfying every layer produces a signal only where the
    cross fires. The walk enters only when FLAT: a second trigger while a
    position is open is ignored rather than pyramided, and an opposite trigger
    is ignored rather than reversing.

    `use_news_filter` is accepted and does nothing. It is in the request's
    toggle list and there is no news feed in this repository; a parameter that
    silently did nothing under a name implying a filter would be worse, so it
    is declared, defaulted OFF, and said out loud here.
    """
    _validate(rsi_fast_len, rsi_slow_len, pullback_window, volume_sma_len,
              volume_mult, rsi_exit_long, rsi_exit_short, sl_atr_mult,
              tp_atr_mult, trailing)

    L = _layers(bars, rsi_fast_len, rsi_slow_len, pullback_window,
                volume_sma_len, volume_mult, use_baseline_filter,
                use_alpha_trigger, use_volume_filter)

    long_ok = (L["trend_long"] & L["trigger_long"] & L["confirm"]).to_numpy()
    short_ok = (L["trend_short"] & L["trigger_short"] & L["confirm"]).to_numpy()

    fast = L["fast"]
    long_sig_exit = (fast >= float(rsi_exit_long)).fillna(False).to_numpy()
    short_sig_exit = (fast <= float(rsi_exit_short)).fillna(False).to_numpy()

    atr = L["atr"].to_numpy(dtype="float64")
    # A bar with no ATR yet is not tradable: the brackets have no width. This
    # is the warm-up, and admitting it would place a stop at the fill.
    warm = ~np.isfinite(atr) | (atr <= 0.0)
    long_ok = long_ok & ~warm
    short_ok = short_ok & ~warm
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


def _tp_distance(tp_atr_mult: float | None) -> float:
    """`None` means NO target, expressed as an unreachable distance.

    Not 0.0, which would place the target at the fill and close every trade on
    its own entry bar.
    """
    return float("inf") if tp_atr_mult is None else float(tp_atr_mult)


def indicators(bars: pd.DataFrame, **params) -> dict[str, pd.Series]:
    """
    Full-length series drawn over the trade inspector's candles, from the same
    `_layers` call `signal_fn` uses - so the chart cannot draw a crossover a
    bar from where the entry actually happened.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["rsi_fast_len"]), int(p["rsi_slow_len"]),
                int(p["pullback_window"]), int(p["volume_sma_len"]),
                float(p["volume_mult"]), bool(p["use_baseline_filter"]),
                bool(p["use_alpha_trigger"]), bool(p["use_volume_filter"]))
    return {
        f"RSI({int(p['rsi_fast_len'])})": L["fast"],
        f"RSI({int(p['rsi_slow_len'])})": L["slow"],
        "RSI 50": pd.Series(50.0, index=bars.index),
        f"ATR({ATR_PERIOD})": L["atr"],
    }


def ml_features(bars: pd.DataFrame, **params) -> pd.DataFrame:
    """
    The matrix Version B's classifier is fitted on: one row per bar, in order.

    CAUSALITY IS THIS MODULE'S RESPONSIBILITY. Every column is built from
    `_layers`, which reads only bars <= i, and nothing here is scaled against
    the whole frame - a scaler fitted end to end leaks the test period's
    distribution into the training rows without tripping any shift-based audit.

    The columns are the state the RULES read, not new information. The filter's
    job is to learn WHEN this setup pays, not to find a different edge.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    L = _layers(bars, int(p["rsi_fast_len"]), int(p["rsi_slow_len"]),
                int(p["pullback_window"]), int(p["volume_sma_len"]),
                float(p["volume_mult"]), bool(p["use_baseline_filter"]),
                bool(p["use_alpha_trigger"]), bool(p["use_volume_filter"]))
    close = bars["close"].astype("float64")
    volume = (bars["volume"].astype("float64") if "volume" in bars.columns
              else pd.Series(np.nan, index=bars.index))
    vol_sma = volume.rolling(int(p["volume_sma_len"]),
                             min_periods=int(p["volume_sma_len"])
                             ).mean().shift(1)
    out = pd.DataFrame({
        "rsi_fast": L["fast"],
        "rsi_slow": L["slow"],
        "rsi_spread": L["fast"] - L["slow"],
        "norm_atr": L["norm_atr"],
        # Ratios rather than levels: a volume of 12,000 means nothing across
        # contracts, and a classifier fitted on levels learns the symbol.
        "volume_ratio": volume / vol_sma.replace(0.0, np.nan),
        "close_over_atr": (close.diff() / L["atr"].replace(0.0, np.nan)),
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
