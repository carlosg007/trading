"""
agents.tier3_workers - the tier that actually runs things.

Location:  ~/src/trading/agents/tier3_workers.py

SCAFFOLD ONLY. Interfaces are defined here; the logic is not written yet.

What this tier is for
---------------------
Tier 3 does the work Tier 1 asked for and Tier 2 will judge: running
backtests, generating ML-filtered strategy variants, and writing artifacts.
It makes no decisions about whether a result is good. It runs the thing and
reports what happened, including when what happened was a failure.

Calling the engine
------------------
`backtest.engine.run_backtest` takes SYMBOLS AND A STRATEGY, not bars and
precomputed signals:

    run_backtest(symbols, tf, signal_fn, start, end, cfg) -> BacktestResult

The engine reads the bars itself, one symbol at a time, and calls `signal_fn`
with that symbol's bars alone.

This is not a stylistic preference and it should not be worked around. The
engine used to accept `(bars, entries, exits)`, where `bars` came from
`get_bars` - a frame sorted by (ts, symbol), which INTERLEAVES instruments.
A strategy computing `close.rolling(200).mean()` over that frame was averaging
across 27 different contracts. Nothing raised: the signals were the right
length and dtype and the equity curve looked plausible. On the full lake it
produced 608,079 trades where the correct per-symbol signals give 86,035.
The signature was deleted so the mistake cannot be made again.

So a worker must never assemble a multi-symbol frame, compute signals over it,
and feed those in. If a future agent finds itself reaching for `get_bars` to
build inputs for a backtest, that is the bug, not the missing convenience.

Costs and constraints are not the worker's to choose. They come from
`BacktestConfig` and `backtest/specs.py`; a worker that runs without costs to
"see if the strategy works" has produced nothing worth reading.

The Dual-Version Mandate
------------------------
Every strategy is expected to produce two variants, run identically:

    Version A   pure rule-based baseline (e.g. an SMA crossover)
    Version B   the same, plus an ML filter on the signals

B is only adopted if it beats A out of sample without breaching the risk
limits. Running only B, or comparing B against a differently-configured A, is
how an ML result gets adopted on no evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

ARTIFACTS = Path("/mnt/backtest/artifacts")


class SignalFn(Protocol):
    """
    The strategies/ contract.

    Receives ONE symbol's bars, positionally indexed from 0, and returns two
    boolean Series aligned to them.
    """

    def __call__(self, bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]: ...


@dataclass
class WorkOrder:
    """One unit of work handed down from Tier 1."""

    name: str
    symbols: list[str]
    timeframe: str
    start: str | None = None
    end: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    variants_tested: int = 1


@dataclass
class WorkReport:
    """
    What happened. Reports failure as readily as success.

    A worker that swallows an exception and returns an empty result is
    indistinguishable, downstream, from a strategy that simply never traded.
    """

    order: WorkOrder
    ok: bool
    stats: dict[str, Any] = field(default_factory=dict)
    breach: dict[str, Any] = field(default_factory=dict)
    artifact_prefix: str | None = None
    error: str | None = None


def run_variant(order: WorkOrder, signal_fn: SignalFn) -> WorkReport:
    """Run one strategy variant and save its artifacts."""
    raise NotImplementedError("tier3_workers: not implemented yet")


def run_dual_version(order: WorkOrder,
                     version_a: SignalFn,
                     version_b: SignalFn) -> tuple[WorkReport, WorkReport]:
    """
    Run the rule-based baseline and the ML-filtered variant on identical terms.

    Same symbols, same window, same costs, same constraints. The only thing
    that may differ between the two is the signal logic.
    """
    raise NotImplementedError("tier3_workers: not implemented yet")


def generate_ml_filter(order: WorkOrder, base: SignalFn) -> SignalFn:
    """Fit a Version B filter over a Version A signal. Training data must not
    include the out-of-sample window."""
    raise NotImplementedError("tier3_workers: not implemented yet")


def main() -> int:
    raise NotImplementedError("tier3_workers: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
