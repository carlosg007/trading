"""
SMA crossover — the Version A rule-based baseline.

Deliberately the dullest strategy that can still be wrong in an interesting
way. Its job is to exercise the pipeline end to end (lake read → signals →
costs → tear sheet), and to be the reference every ML-filtered Version B has
to beat out-of-sample. If a clever idea cannot beat this after costs, the
cleverness was noise.

Contract — the one `backtest.engine` and `agents.tier3_workers` both call:

    signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, pd.Series]

`bars` is ONE symbol's OHLCV frame, oldest to newest, lowercase columns, UTC
index. Returns `(entries, exits)` as boolean Series on `bars.index`.

Never hand this a multi-symbol frame. `get_bars` sorts by `(ts, symbol)`, so a
rolling mean over the concatenation averages across unrelated contracts and
produces signals that are the right length, the right dtype, and meaningless.
The engine calls this per symbol precisely so that cannot happen.
"""

from __future__ import annotations

import pandas as pd

TIMEFRAME = "1d"
SYMBOLS = ["NQ"]
DEFAULT_PARAMS = {"fast_window": 10, "slow_window": 30}

# The search space `backtest/run.py --scan` sweeps, declared here because this
# module is the only place that knows what these parameters mean and what the
# signature will accept. Kept coarse and few on purpose: nine combinations over
# 4,000 daily bars is a search whose result can be reported honestly, and a
# 400-cell grid over the same bars is a machine for manufacturing an in-sample
# Sharpe. Every combination here is valid - fast is always below slow - so the
# scan reports nine evaluated rather than nine attempted and four rejected.
PARAM_GRID = {
    "fast_window": [5, 10, 20],
    "slow_window": [30, 50, 100],
}

# Plain-English description for the tear sheet's strategy card, written for a
# reader deciding whether to trade this - not for whoever maintains the module.
# `{param}` slots are filled with the run's own bound parameters, so the card
# states the windows that actually ran rather than the defaults written here.
# The report never infers any of this from the signal arrays: a description
# guessed from the trades would be a guess printed as a fact.
LOGIC = {
    "concept": "Trend following. Buy strength when a short average of price "
               "overtakes a long one, and stand aside when it gives way.",
    "entry": "Go Long when the Fast SMA ({fast_window}) crosses above the "
             "Slow SMA ({slow_window}).",
    "exit": "Exit when the Fast SMA ({fast_window}) crosses back below the "
            "Slow SMA ({slow_window}).",
}


def signal_fn(bars: pd.DataFrame,
              fast_window: int = 10,
              slow_window: int = 30) -> tuple[pd.Series, pd.Series]:
    """
    Long when the fast SMA crosses above the slow SMA; flat when it crosses back.

    Signals are the crossover *event*, not the state. `fast > slow` would be
    True on every bar of a trend, and the engine would read a fresh entry on
    each one. Comparing against the previous bar isolates the transition.

    `.shift(1)` looks one bar BACKWARD and is the correct direction: the
    comparison at bar i uses only bars <= i. The engine then fills at bar i+1's
    open, so nothing here can see a price it would not have had.
    """
    if fast_window < 1 or slow_window < 1:
        raise ValueError(
            f"windows must be >= 1; got fast={fast_window}, slow={slow_window}"
        )
    if fast_window >= slow_window:
        # Not a stylistic objection. Equal or inverted windows make the
        # crossover fire on noise and the result is not the strategy being
        # described, so it fails here rather than producing a plausible curve.
        raise ValueError(
            f"fast_window must be < slow_window; got {fast_window} >= "
            f"{slow_window}"
        )

    close = bars["close"]

    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()

    above = fast > slow
    was_above = above.shift(1)

    # Warm-up is NaN in `was_above`, which is falsy, so no signal fires until
    # both means exist and one full bar of history is behind them.
    entries = above & ~was_above.fillna(False).astype(bool)
    exits = ~above & was_above.fillna(False).astype(bool)

    # min_periods leaves the warm-up NaN rather than seeding from a partial
    # window, so these fillna calls only ever fill the warm-up.
    return (entries.fillna(False).astype(bool),
            exits.fillna(False).astype(bool))


def indicators(bars: pd.DataFrame,
               fast_window: int = 10,
               slow_window: int = 30) -> dict[str, pd.Series]:
    """
    The two means, for the tear sheet to draw over the trade inspector's candles.

    Computed HERE, the same way and from the same column `signal_fn` reads, so
    the line a reader sees cross is the line the entry was taken from. A second
    implementation living in the report would be free to disagree with this one
    - a chart showing a crossover one bar away from where the trade fired, with
    nothing raising.

    Warm-up stays NaN. The report renders it as a gap rather than drawing the
    average flat through the first thirty bars.
    """
    close = bars["close"]
    return {
        f"Fast SMA ({fast_window})":
            close.rolling(fast_window, min_periods=fast_window).mean(),
        f"Slow SMA ({slow_window})":
            close.rolling(slow_window, min_periods=slow_window).mean(),
    }


def make_signal_fn(fast_window: int = 10,
                   slow_window: int = 30):
    """
    Bind parameters for `agents.tier3_workers.load_strategy`.

    The loader prefers this factory when params are supplied; the engine itself
    only ever calls the bound `signal_fn(bars)`.
    """
    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return signal_fn(bars, fast_window=fast_window, slow_window=slow_window)

    return _bound
