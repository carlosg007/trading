"""
Intraday EMA crossover — ride the momentum shift, exit on the stop or the bell.

Location:  ~/src/trading/strategies/experimental/ema_crossover.py

Hypothesis
----------
Intraday order flow arrives in bursts rather than smoothly. When enough of it
lands on one side, short-horizon price leaves the intermediate-term trend
behind, and the participants who have to follow — index arbitrage, systematic
trend books, and discretionary traders who were flat — add to it over the
following bars rather than instantly. A fast EMA overtaking a slow one is a
cheap, lagging measurement of exactly that transition.

The edge is a delayed-reaction premium, not a forecast. It should persist
because the follow-through is structural (participants react on their own
clocks, not on the tick), and it is fragile for the same reason it is cheap:
in a chopping session the two averages cross repeatedly, each crossing costs a
spread and a commission, and the strategy pays out its trend days in whipsaw.
The ATR stop, the optional ATR take-profit and the session flatten bound that.

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
The stop and the take-profit are swept, not fixed:

    sl_atr_mult   stop distance, in ATR(14) multiples. Required, > 0.
    tp_atr_mult   take-profit distance, in ATR(14) multiples above the FILL
                  price. `None` means no take-profit is modelled at all — not
                  a take-profit at infinity, and the tear sheet says so.
    trailing      True  → the stop ratchets with the high-water mark since the
                          fill and never widens.
                  False → the stop is fixed at `fill price - sl_atr_mult x ATR`
                          and never moves in either direction.

Both distances are frozen at `ATR(14)` as measured on the SIGNAL bar and never
re-measured. `sl_atr_mult` replaced the earlier `atr_stop_mult` when the risk
grid was added on 2026-08-16; the name is not accepted as an alias, because
`load_strategy` rejects unknown parameter names and a silently ignored stop
multiplier is exactly the failure that rename is meant to make loud.

Read this before reading the equity curve
-----------------------------------------
Six things about this module are NOT what the specification asks for, or are
narrower than the words in it. None of them flatters the result, but a
drawdown, a win rate or a trade count read without knowing them is being read
wrong.

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
   position went live, minus the frozen distance. It never widens. The
   high-water mark starts on the FILL bar, not the signal bar — including the
   signal bar's high would credit the position with a price it never held
   through and set the first stop too far away. With `trailing=False` the
   anchor is the actual fill price, `open` on the fill bar, and not the signal
   bar's close.

4. THE EXITS ARE COMPUTED IN THIS MODULE, NOT BY THE ENGINE. A trailing stop is
   path-dependent — its level depends on which bar opened the position — so it
   cannot be a stateless boolean mask over bars, and once the walk exists the
   fixed stop and the target belong in it too rather than being split across
   two layers that could disagree. `_walk` below is a two-state machine that
   resolves entries, the stop, the target and the session flatten together,
   mirroring what `clean_signals` will later do with them. That is why this
   module contains session logic at all, which strategies otherwise must not.

5. THE STOP IS SWEPT, WHICH IS A KNOWN WAY TO FIT NOISE. `PARAM_GRID` varies
   `sl_atr_mult`, `tp_atr_mult` and `trailing` alongside the periods, so the
   search can find the risk settings that happened to survive this sample's
   worst few days. If the winning cell's edge lives mostly in the risk
   parameters rather than in the crossover, read that as evidence against the
   strategy rather than for it, and check the neighbouring cells in
   `scan_<SYMBOL>.csv` before believing any of it.

6. `--flat-by-close` IS NOT NEEDED AND SHOULD NOT BE PASSED. This module emits
   its own session exit on the last bar starting before 16:00 ET, converted
   through America/New_York so it is right on both sides of a DST change. The
   engine's flag uses a FIXED UTC time, which is 16:00 ET in summer and 15:00
   ET in winter. Passing it is harmless — `clean_signals` keeps whichever exit
   comes first, and this one always does — but it is redundant and its winter
   boundary is an hour off.

One further consequence of the state machine, which is a modelling choice
rather than a limitation: after a stop or a session exit, re-entry requires a
FRESH crossover. The strategy does not step back into a trend it was stopped
out of, because `fast > slow` is still true there and re-entering on the state
rather than the event would reopen the position on the very next bar.

Every calculation is causal. Values at bar i use bars <= i only, warm-up stays
NaN and is never treated as a signal, and the engine then fills at bar i+1's
open — so nothing here can see a price it would not have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME = "15m"
# The primary target is NQ alone; ES, CL and GC are the multi-asset extension
# the request names, and are reached with `--symbols NQ,ES,CL,GC`. They are not
# the default because each contract is its own independent simulation on its
# own specs, and a leaderboard of four is a different claim from a result on
# one.
SYMBOLS = ["NQ"]
DEFAULT_PARAMS = {"fast_period": 9, "slow_period": 21,
                  "sl_atr_mult": 2.0, "tp_atr_mult": None, "trailing": True}

# The search space `backtest/run.py --scan` sweeps, declared here because this
# module is the only place that knows what these parameters mean and what the
# signature will accept. 2 x 3 x 3 x 3 x 2 = 108 combinations, every one of
# them valid (every fast value is below every slow value), so the scan reports
# 108 evaluated rather than 108 attempted and some rejected.
#
# THE INDICATOR AXES WERE COARSENED WHEN THE RISK AXES WERE ADDED, and that
# trade is the whole point. The pre-risk grid was 4 x 4 x 3 = 48; crossing the
# original four stop values, five target values and two trailing flags onto it
# would have been 640 fits to the same in-sample bars, which is not a search
# whose winner can be reported as a measurement. 108 is at the top of what can.
# `variants_tested` is carried onto every report and every leaderboard row for
# exactly that reason: the best of 108 is a higher number than the best of 9
# even when nothing in the market has changed.
#
# `tp_atr_mult: None` is a real point in the space, not a placeholder — it is
# the no-take-profit configuration this strategy shipped with, kept in the grid
# so "the target earns its place" is something the sweep answers rather than
# something assumed.
PARAM_GRID = {
    "fast_period": [9, 20],
    "slow_period": [21, 50, 89],
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
    "concept": "Intraday trend following. Short-term momentum overtaking the "
               "intermediate-term trend marks the start of a directional move "
               "that other participants join over the following bars — so buy "
               "the crossover, ride the continuation, and give it back only "
               "as far as a volatility-scaled trailing stop allows.",
    "entry": "Go Long when the Fast EMA ({fast_period}) crosses above the "
             "Slow EMA ({slow_period}). Entry signals are only taken between "
             "09:30 and 15:30 New York time; the fill is the next bar's open.",
    # Written so that every bound value reads correctly, including
    # `tp_atr_mult=None` and either setting of `trailing`. The card cannot
    # branch — `_describe_strategy` only substitutes `{param}` slots — so the
    # sentence states both arms and names the setting that chose between them.
    "exit": "Exit at whichever comes first. (1) The Fast EMA ({fast_period}) "
            "crosses back below the Slow EMA ({slow_period}). (2) A stop "
            "{sl_atr_mult} x ATR 14 away — with trailing={trailing}, True "
            "means it trails the highest price reached since the fill and "
            "never widens, False means it sits fixed that far below the fill "
            "price. (3) A take-profit {tp_atr_mult} x ATR 14 above the fill "
            "price, where None means NO take-profit is modelled at all. "
            "(4) The last bar before 16:00 New York time. Every exit fills at "
            "the NEXT bar's open, so neither the stop nor the target is a fill "
            "at its own price — both carry a one-bar lag, and on a bar that "
            "breaches both the modelled fill is neither level.",
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
    the first bar, so a "50-period EMA" exists at bar 2 and the first crossover
    of every symbol's history is manufactured by the seeding rather than by
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

    This kernel is DUPLICATED VERBATIM in `ema_crossover.py`,
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
def _validate(fast_period: int, slow_period: int, sl_atr_mult: float,
              tp_atr_mult: float | None, trailing: bool) -> None:
    """
    Reject parameter sets that do not describe this strategy.

    Every rejection here is recorded by `backtest.scan` as REJECTED and counted
    in the grid, so a combination the strategy refuses shrinks neither the
    reported search nor the honesty of `variants_tested`.
    """
    if fast_period < 2 or slow_period < 2:
        raise ValueError(
            f"periods must be >= 2; got fast={fast_period}, slow={slow_period}"
        )
    if fast_period >= slow_period:
        # Not a stylistic objection. Equal or inverted periods make the
        # crossover fire on noise and the result is not the strategy being
        # described, so it fails here rather than producing a plausible curve.
        raise ValueError(
            f"fast_period must be < slow_period; got {fast_period} >= "
            f"{slow_period}"
        )
    if sl_atr_mult is None or sl_atr_mult <= 0:
        # The stop is not optional. A position with no stop and no target exits
        # only on the crossover or the bell, which is a different strategy.
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


def _series(bars: pd.DataFrame, fast_period: int, slow_period: int) -> dict:
    """The shared calculation behind both `signal_fn` and `indicators`."""
    close = bars["close"]
    return {
        "fast": _ema(close, fast_period),
        "slow": _ema(close, slow_period),
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


def _signal_arrays(bars: pd.DataFrame, fast_period: int, slow_period: int,
                   sl_atr_mult: float, tp_atr_mult: float | None,
                   trailing: bool) -> tuple:
    """
    Everything the walk produces, from one place.

    `signal_fn` and `indicators` both go through here, so the stop line the
    tear sheet draws is the array the exit was taken from and the crossing a
    reader sees is the array the entry came from. Two call sites computing this
    separately would be free to drift apart with nothing raising.
    """
    s = _series(bars, fast_period, slow_period)
    entry_window, flat_bar = _session_masks(bars["ts"])

    # Warm-up is NaN in both EMAs and in ATR, and every comparison against NaN
    # is False, so no signal can fire before all three exist. The explicit
    # notna guard makes that a stated requirement rather than a property of NaN
    # comparison that a later edit could quietly lose.
    ready = s["fast"].notna() & s["slow"].notna() & s["atr"].notna()

    above = s["fast"] > s["slow"]
    # `ready` is applied to the PREVIOUS bar as well, so the first bar on which
    # both averages exist cannot be read as a crossing from a warm-up NaN.
    was_above = (above & ready).shift(1).fillna(False).astype(bool)
    now_above = (above & ready)

    cross_up = (now_above & ~was_above).to_numpy(dtype=bool)
    cross_down = (~now_above & was_above).to_numpy(dtype=bool)

    # LONG ONLY. The shared kernel walks three states, and this module feeds it
    # all-False short masks: nothing in this strategy's specification takes a
    # short, and the kernel is shared so it cannot drift from
    # `ema_trend_filter`'s copy — not because both modules trade both ways.
    # `signal_fn` therefore returns the two-mask form, which the engine's
    # `unpack_signals` accepts unchanged.
    no_shorts = np.zeros(len(bars), dtype=bool)

    entries, exits, s_entries, s_exits, stop, target = _walk(
        cross_up & np.asarray(entry_window),
        no_shorts,
        cross_down,
        no_shorts,
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
    # The short pair is returned rather than dropped so this helper has the same
    # shape as `ema_trend_filter`'s. Both are all-False here, and
    # `tests/test_risk_params.py` asserts that rather than assuming it.
    return s, entries, exits, s_entries, s_exits, stop, target


def signal_fn(bars: pd.DataFrame,
              fast_period: int = 9,
              slow_period: int = 21,
              sl_atr_mult: float = 2.0,
              tp_atr_mult: float | None = None,
              trailing: bool = True) -> tuple[pd.Series, pd.Series]:
    """
    Long the upward crossover; exit on the downward one, the stop, the target,
    or the bell.

    The entry is the crossover EVENT, not the state. `fast > slow` is true on
    every bar of a trend and the engine would read a fresh entry on each one;
    comparing against the previous bar isolates the transition.

    `.shift(1)` looks one bar BACKWARD and is the correct direction: the
    comparison at bar i uses only bars <= i. The engine then fills at bar i+1's
    open, so nothing here can see a price it would not have had.
    """
    _validate(fast_period, slow_period, sl_atr_mult, tp_atr_mult, trailing)

    _s, entries, exits, _se, _sx, _stop, _target = _signal_arrays(
        bars, fast_period, slow_period, sl_atr_mult, tp_atr_mult, trailing)

    # The two-mask (long-only) form of the contract. The engine also accepts
    # four; see `backtest.engine.unpack_signals`.
    return (pd.Series(entries, index=bars.index),
            pd.Series(exits, index=bars.index))


def indicators(bars: pd.DataFrame,
               fast_period: int = 9,
               slow_period: int = 21,
               sl_atr_mult: float = 2.0,
               tp_atr_mult: float | None = None,
               trailing: bool = True) -> dict[str, pd.Series]:
    """
    The price-scale series, for the tear sheet to draw over the candles.

    Computed HERE, the same way and from the same column `signal_fn` reads, so
    the crossing a reader sees is the array the entry was taken from. A second
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

    ATR itself is deliberately NOT returned. The inspector draws these on the
    price axis, and a volatility series measured in points sits along the
    bottom of a 20,000-point contract telling a reader nothing. It reaches the
    reader through the stop and target lines, which is where it matters.

    Warm-up stays NaN rather than drawing the averages flat through the first
    bars.
    """
    _validate(fast_period, slow_period, sl_atr_mult, tp_atr_mult, trailing)

    s, _entries, _exits, _se, _sx, stop, target = _signal_arrays(
        bars, fast_period, slow_period, sl_atr_mult, tp_atr_mult, trailing)

    kind = "Trailing" if trailing else "Fixed"
    out = {
        f"Fast EMA ({fast_period})": s["fast"],
        f"Slow EMA ({slow_period})": s["slow"],
        f"{kind} Stop ({sl_atr_mult}xATR{ATR_PERIOD})":
            pd.Series(stop, index=bars.index),
    }
    if tp_atr_mult is not None:
        out[f"Take Profit ({tp_atr_mult}xATR{ATR_PERIOD})"] = pd.Series(
            target, index=bars.index)
    return out


def make_signal_fn(fast_period: int = 9,
                   slow_period: int = 21,
                   sl_atr_mult: float = 2.0,
                   tp_atr_mult: float | None = None,
                   trailing: bool = True):
    """
    Bind parameters for `agents.tier3_workers.load_strategy`.

    The loader prefers this factory when params are supplied; the engine itself
    only ever calls the bound `signal_fn(bars)`. Validation runs HERE as well
    as inside `signal_fn`, so `--scan` records a bad combination as REJECTED at
    bind time rather than discovering it one symbol into the sweep.
    """
    _validate(fast_period, slow_period, sl_atr_mult, tp_atr_mult, trailing)

    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return signal_fn(bars, fast_period=fast_period,
                         slow_period=slow_period,
                         sl_atr_mult=sl_atr_mult,
                         tp_atr_mult=tp_atr_mult,
                         trailing=trailing)

    return _bound
