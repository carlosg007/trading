"""
ema_trend_filter - Version B (ML-filtered), promoted 2026-08-17.

Baseline signals from `baseline.py` (SHA-256 e178019624e5daa0ec6362a9a9d009e1fef6911f51251a840cfe5ea95260f3ec), with the causal ML
filter applied on top. This is the pipeline, not a new idea: every entry here
is an entry Version A also produced, minus the ones the classifier expected to
lose.

The filter is an expanding-window walk-forward. For a candidate entry on bar
`s` it is fitted only on trades that had already CLOSED before `s`, so no
decision uses an outcome that did not exist when it was made. Refitting is
per completed trade, not per bar.

Promoted from: /home/cgrullon/src/trading/strategies/experimental/ema_trend_filter.py
ML threshold : 0.5 - keep the entry when P(win) >= this.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from agents.tier3_workers import apply_ml_signal_filter
from backtest.engine import BacktestConfig

TIMEFRAME = '15m'
SYMBOLS = ['NQ', 'ES', 'CL', 'GC']
DEFAULT_PARAMS = {'fast_period': 5, 'slow_period': 21, 'trend_period': 200, 'use_trend': True, 'use_vwap': True, 'use_rsi': False, 'use_volatility': False, 'sl_atr_mult': 2.0, 'tp_atr_mult': 1.5, 'trailing': True}
ML_THRESHOLD = 0.5

_BASELINE_PATH = Path(__file__).with_name("baseline.py")


def _baseline():
    """Load the promoted rule-based module sitting next to this file."""
    spec = importlib.util.spec_from_file_location(
        "ema_trend_filter_baseline", _BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load baseline from {_BASELINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def signal_fn(bars: pd.DataFrame, **params):
    """
    Version B: the baseline's signals, with losing entries suppressed.

    `bars` is ONE symbol's frame, oldest to newest - the engine calls this once
    per symbol. The symbol is read from the frame when the lake reader put it
    there and falls back to the promoted SYMBOLS entry, because it decides the
    contract multiplier, tick size and commission the filter's training labels
    are net of. Without it the classifier learns from gross outcomes and keeps
    trades that lose money after costs.

    Returns whichever shape the baseline returns: two masks for a long-only
    strategy, four for a bidirectional one. A bidirectional baseline gets one
    classifier per side, each trained on its own completed trades with its own
    P&L sign - see `apply_ml_signal_filter`. Filtering only the long side here
    would ship a promoted Version B whose shorts never met the filter it is
    named for.
    """
    threshold = params.pop("threshold", ML_THRESHOLD)
    cfg = params.pop("cfg", None) or BacktestConfig()

    merged = dict(DEFAULT_PARAMS)
    merged.update(params)

    out = _baseline().signal_fn(bars, **merged)
    if len(out) == 4:
        entries, exits, s_entries, s_exits = out
    else:
        entries, exits = out
        s_entries = s_exits = None

    symbol = None
    if "symbol" in bars.columns:
        present = pd.unique(pd.Series(bars["symbol"]).dropna())
        if len(present) > 1:
            raise ValueError(
                f"bars carry {len(present)} symbols. Pass one symbol's bars: "
                f"a rolling window over an interleaved frame averages across "
                f"contracts and the result looks fine.")
        if len(present) == 1:
            symbol = str(present[0])
    if symbol is None and SYMBOLS:
        symbol = SYMBOLS[0]

    def _b(s):
        return pd.Series(s).fillna(False).astype(bool)

    entries, exits = apply_ml_signal_filter(
        bars, entries, exits, symbol=symbol, cfg=cfg, threshold=threshold,
        direction="long")
    if s_entries is None:
        return _b(entries), _b(exits)

    s_entries, s_exits = apply_ml_signal_filter(
        bars, _b(s_entries), _b(s_exits), symbol=symbol, cfg=cfg,
        threshold=threshold, direction="short")
    return _b(entries), _b(exits), _b(s_entries), _b(s_exits)


def make_signal_fn(**params):
    """Bind parameters for `agents.tier3_workers.load_strategy`."""
    def _bound(bars: pd.DataFrame):
        return signal_fn(bars, **params)

    return _bound
