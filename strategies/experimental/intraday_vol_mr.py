"""
Intraday volatility-exhaustion mean reversion — fade the band, exit at the mean.

Location:  ~/src/trading/strategies/experimental/intraday_vol_mr.py

Hypothesis
----------
Intraday selling pressure overshoots. A liquidation, a stop cascade or a single
large seller pushes price through a volatility-scaled band faster than
information justifies, and the last leg of that move is exhaustion rather than
news. When price pierces the band and closes back inside it on the same bar,
the sellers who could move it have finished, and price reverts toward the
session mean.

The edge is a liquidity premium, not a forecast: it is paid for standing
between a forced seller and the mean. That is why it should persist — the
sellers are not trying to be right, they are trying to be out — and also why it
is fragile. On a day when the move IS information, the reversion never comes
and the stop pays for it. The stop is therefore not risk management bolted on
afterwards; it is the other half of the trade.

Contract — the one `backtest.engine` and `agents.tier3_workers` both call:

    signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, pd.Series]

`bars` is ONE symbol's OHLCV frame, oldest to newest, lowercase columns, with a
`ts` column in UTC. Returns `(entries, exits)` as boolean Series on bars.index.

Read this before reading the equity curve
-----------------------------------------
Three things about this module are NOT what the specification asks for, because
the engine cannot express them. They are all conservative — none of them
flatters the result — but a drawdown or a win rate read without knowing them is
being read wrong.

1. THE TRAILING STOP IS FILLED AT THE NEXT BAR'S OPEN, NOT AT THE STOP PRICE.
   The engine fills every signal at the next bar's open; there is no intrabar
   stop order anywhere in it. So the stop is detected on the bar whose low
   breaches it and executed at the following bar's open, which on a fast move
   is worse than the stop price and on a snapback is better. A real broker
   would fill near the stop. Treat stop-exit prices here as an approximation
   with a one-bar lag, and do not read the drawdown as what a live account
   would have taken.

2. THE STOP AND THE SESSION EXIT ARE COMPUTED IN THIS MODULE, NOT BY THE
   ENGINE. A trailing stop is path-dependent - its level depends on the highest
   price since entry, which depends on when the entry happened - so it cannot
   be written as a stateless boolean mask over bars. `_walk` below is a
   two-state machine that tracks the position and emits the exits, mirroring
   what `clean_signals` will later do with them. That is why this module
   contains session logic at all, which strategies otherwise must not.

3. `--flat-by-close` IS NOT NEEDED AND SHOULD NOT BE PASSED. This module emits
   its own session exit on the last bar starting before 16:00 ET, converted
   properly through America/New_York so it is right on both sides of a DST
   change. The engine's flag uses a FIXED UTC time (default 20:00), which is
   16:00 ET in summer and 15:00 ET in winter. Passing it is harmless -
   `clean_signals` keeps whichever exit comes first, and this one always does -
   but it is redundant and its winter boundary is an hour off.

Every calculation is causal. Values at bar i use bars <= i only, warm-up stays
NaN and is never treated as a signal, and the engine then fills at bar i+1's
open — so nothing here can see a price it would not have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME = "15m"
SYMBOLS = ["NQ", "ES", "CL", "GC"]
DEFAULT_PARAMS = {"ema_period": 20, "atr_mult": 2.5, "rsi_thresh": 30}

# The search space `backtest/run.py --scan` sweeps. 27 combinations, three
# values per axis, spanning the range either side of the default rather than
# resolving it finely: a grid dense enough to find a local optimum on 15m bars
# is a grid dense enough to manufacture one. Every leaderboard row carries the
# count, because a Sharpe selected from 27 in-sample fits is not the same
# measurement as a Sharpe from the first thing tried.
PARAM_GRID = {
    "ema_period": [15, 20, 30],
    "atr_mult": [2.0, 2.5, 3.0],
    "rsi_thresh": [25, 30, 35],
}

# Fixed, not swept. The stop is the other half of the trade rather than a knob:
# sweeping it alongside the entry would let the search find the distance that
# happened to survive this sample's worst three days, which is the definition
# of fitting the noise. Change it deliberately or not at all.
STOP_ATR_MULT = 1.75

ATR_PERIOD = 14
RSI_PERIOD = 14

# Eastern wall-clock, converted through the zone so both sides of a DST change
# are right. Entry SIGNALS are taken on bars starting inside this window; the
# engine fills the next bar's open, so the latest possible fill is the bar
# starting at 15:30 ET.
ENTRY_OPEN_ET = (9, 30)
ENTRY_CLOSE_ET = (15, 30)
SESSION_FLAT_ET = (16, 0)

LOGIC = {
    "concept": "Intraday mean reversion on volatility exhaustion. When price "
               "spikes below a volatility-scaled band and closes back inside "
               "it while momentum is washed out, the sellers who could move it "
               "have finished — so buy the recovery and take profit when price "
               "returns to the session mean.",
    "entry": "Go Long when the bar's low pierces the Lower Band "
             "(EMA {ema_period} minus {atr_mult} x ATR 14) but its close "
             "recovers back above that band, and RSI 14 is below "
             "{rsi_thresh}. Entry signals are only taken between 09:30 and "
             "15:30 New York time; the fill is the next bar's open.",
    "exit": "Exit at whichever comes first: price closing back at or above "
            "the EMA {ema_period} (the target), a trailing stop 1.75 x ATR 14 "
            "below the highest price reached since entry, or the last bar "
            "before 16:00 New York time. Every exit fills at the NEXT bar's "
            "open, so the stop is an approximation with a one-bar lag and not "
            "a fill at the stop price.",
}


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def _wilder(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing — the average ATR and RSI are actually defined on.

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


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Wilder's RSI. NaN until `period` bars exist.

    A zero average loss makes RS infinite and RSI exactly 100, which is the
    correct answer (nothing has fallen), not a division error — so it is
    produced rather than masked. It can never satisfy `rsi < rsi_thresh`.
    """
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), period)
    loss = _wilder((-delta).clip(lower=0.0), period)
    rs = gain / loss
    return 100.0 - 100.0 / (1.0 + rs)


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
    the close, where a string match would simply find nothing and never flatten.
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
               high: np.ndarray,
               low: np.ndarray,
               close: np.ndarray,
               ema: np.ndarray,
               atr: np.ndarray,
               flat_bar: np.ndarray,
               stop_mult: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-state machine over the bars: flat, or long with a trailing stop.

    A trailing stop cannot be a stateless mask. Its level is the highest price
    since ENTRY minus a fixed distance, so bar i's exit condition depends on
    which earlier bar opened the position — which depends on every entry before
    it. This walks the bars once and resolves both together.

    The timeline matches the engine's. An entry signal on bar i is filled at
    bar i+1's open, so the position is live from bar i+1 and the high-water
    mark starts there, NOT on the signal bar. Tracking it from bar i would
    include a high the position never held through and set the first stop too
    far away.

    The stop distance is frozen at `stop_mult * ATR` as measured on the SIGNAL
    bar and does not re-widen as volatility rises. That is what "fixed ATR
    trailing stop" means: the level ratchets up with price and never down.

    Exits are checked from the fill bar onward, never on the signal bar itself.
    """
    n = entry_ok.shape[0]
    entries = np.zeros(n, dtype=np.bool_)
    exits = np.zeros(n, dtype=np.bool_)

    pos = False
    fill_i = 0
    stop_dist = 0.0
    hw = 0.0

    for i in range(n):
        if not pos:
            if entry_ok[i]:
                entries[i] = True
                pos = True
                fill_i = i + 1
                stop_dist = stop_mult * atr[i]
                hw = -np.inf
            continue

        if i < fill_i:
            continue

        if high[i] > hw:
            hw = high[i]

        hit_stop = low[i] <= hw - stop_dist
        hit_target = close[i] >= ema[i]

        if hit_stop or hit_target or flat_bar[i]:
            exits[i] = True
            pos = False

    return entries, exits


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
def _series(bars: pd.DataFrame, ema_period: int, atr_mult: float) -> dict:
    """The shared calculation behind both `signal_fn` and `indicators`."""
    close = bars["close"]
    ema = close.ewm(span=ema_period, adjust=False,
                    min_periods=ema_period).mean()
    atr = _atr(bars, ATR_PERIOD)
    return {
        "ema": ema,
        "atr": atr,
        "rsi": _rsi(close, RSI_PERIOD),
        "lower_band": ema - atr_mult * atr,
    }


def signal_fn(bars: pd.DataFrame,
              ema_period: int = 20,
              atr_mult: float = 2.5,
              rsi_thresh: float = 30) -> tuple[pd.Series, pd.Series]:
    """
    Long the recovery from a band pierce; exit at the mean, the stop, or the bell.

    The entry is the pierce-and-recover EVENT, not the state of being below the
    band. `low < band` alone would be true on every bar of a slide and the
    engine would read a fresh entry on each one; requiring `close >= band` on
    the same bar is what makes it a rejection rather than a continuation.
    """
    if ema_period < 2:
        raise ValueError(f"ema_period must be >= 2; got {ema_period}")
    if atr_mult <= 0:
        raise ValueError(f"atr_mult must be > 0; got {atr_mult}")
    if not 0 < rsi_thresh < 100:
        raise ValueError(f"rsi_thresh must be in (0, 100); got {rsi_thresh}")

    s = _series(bars, ema_period, atr_mult)
    entry_window, flat_bar = _session_masks(bars["ts"])

    # Warm-up is NaN in every series here, and every comparison against NaN is
    # False, so no signal can fire before all three indicators exist. The
    # explicit notna guard makes that a stated requirement rather than a
    # property of NaN comparison that a later edit could quietly lose.
    ready = (s["ema"].notna() & s["atr"].notna() & s["rsi"].notna()
             & s["lower_band"].notna())

    entry_ok = (
        (bars["low"] < s["lower_band"])          # pierced the band
        & (bars["close"] >= s["lower_band"])     # and closed back inside it
        & (s["rsi"] < rsi_thresh)                # with momentum washed out
        & ready
        & pd.Series(entry_window, index=bars.index)
    ).fillna(False).to_numpy(dtype=bool)

    entries, exits = _walk(
        entry_ok,
        bars["high"].to_numpy(dtype=float),
        bars["low"].to_numpy(dtype=float),
        bars["close"].to_numpy(dtype=float),
        # A NaN EMA would make `close >= ema` False and hold a position open
        # through warm-up. It cannot arise - no entry fires before `ready` -
        # but -inf makes the target unreachable rather than accidentally
        # satisfied if that ever changes.
        np.nan_to_num(s["ema"].to_numpy(dtype=float), nan=np.inf),
        np.nan_to_num(s["atr"].to_numpy(dtype=float), nan=0.0),
        flat_bar,
        float(STOP_ATR_MULT),
    )

    return (pd.Series(entries, index=bars.index),
            pd.Series(exits, index=bars.index))


def indicators(bars: pd.DataFrame,
               ema_period: int = 20,
               atr_mult: float = 2.5,
               rsi_thresh: float = 30) -> dict[str, pd.Series]:
    """
    The two PRICE-SCALE series, for the tear sheet to draw over the candles.

    Computed here, the same way and from the same columns `signal_fn` reads, so
    the band a reader sees pierced is the array the entry was taken from. A
    second implementation living in the report would be free to disagree with
    this one and draw the touch a bar away from where the trade fired.

    RSI is deliberately NOT returned. The inspector draws these on the price
    axis, and a 0-100 oscillator plotted against a 15,000-point contract is a
    flat line along the bottom of the chart that tells a reader nothing and
    crowds the legend. The RSI condition is stated in the logic card instead,
    which is where a reader can actually use it.

    Warm-up stays NaN and the report renders it as a gap, rather than drawing
    the band flat through the first fourteen bars.
    """
    s = _series(bars, ema_period, atr_mult)
    return {
        f"EMA ({ema_period})": s["ema"],
        f"Lower Band (−{atr_mult}×ATR{ATR_PERIOD})": s["lower_band"],
    }


def make_signal_fn(ema_period: int = 20,
                   atr_mult: float = 2.5,
                   rsi_thresh: float = 30):
    """
    Bind parameters for `agents.tier3_workers.load_strategy`.

    The loader prefers this factory when params are supplied; the engine itself
    only ever calls the bound `signal_fn(bars)`.
    """
    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return signal_fn(bars, ema_period=ema_period, atr_mult=atr_mult,
                         rsi_thresh=rsi_thresh)

    return _bound
