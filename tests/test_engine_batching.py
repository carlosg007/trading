#!/usr/bin/env python3
"""
test_engine_batching.py - does batching change the answer?

Location:  ~/src/trading/tests/test_engine_batching.py

Run:  python tests/test_engine_batching.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

Why this test exists
--------------------
_simulate now feeds vectorbt one chunk of bars at a time instead of the whole
symbol, so a 110-million-row run does not have to hold the full simulation in
RAM at once. The obvious way to chunk - slice by year - is WRONG: a position
open on 31 December is never closed, the trade is silently dropped, and the
backtest quietly reports better numbers than the strategy earned. A dropped
losing trade flatters the equity curve, which is exactly the failure mode this
repo is built to avoid.

So the chunk boundaries are snapped into the gaps BETWEEN trades. The claim
that buys us is strong: chunking is a pure memory optimisation and the trade
list is bit-identical to the unchunked run. That claim is what is tested here,
and it is tested by comparing against the unchunked engine rather than against
a hand-picked expectation, so it holds for any signal pattern.

What it checks
--------------
1. Chunked == unchunked, trade for trade, over a real 2-year 1-minute slice at
   several chunk sizes, including one coprime with the trade spacing so
   boundaries drift through every phase of the trade cycle.
2. A trade deliberately straddling a chunk boundary survives with the right
   entry, exit and P&L. This is the year-slicing bug, tested directly.
3. run_backtest end-to-end, multi-symbol, chunked vs unchunked: same trades,
   same equity, same stats.
4. Peak RSS actually falls. A batching change that does not reduce memory is
   just added complexity.
"""

from __future__ import annotations

import gc
import resource
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from backtest.engine import (BacktestConfig, _simulate, clean_signals,
                             run_backtest)

FAILURES: list[str] = []

# A 2-year slice, as asked: small enough to run in seconds, real enough that
# the bar spacing has the gaps and session breaks synthetic data never has.
START, END = "2022-01-01", "2024-01-01"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def same_trades(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    """Every column, every row, to floating-point exactness."""
    if len(a) != len(b):
        check(f"{label}: same trade count", False, f"chunked {len(a)} vs whole {len(b)}")
        return
    check(f"{label}: same trade count", True, f"{len(a)} trades")
    if a.empty:
        return

    for col in ("entry_time", "exit_time"):
        eq = bool((pd.to_datetime(a[col], utc=True).to_numpy()
                   == pd.to_datetime(b[col], utc=True).to_numpy()).all())
        check(f"{label}: {col} identical", eq)

    for col in ("entry_price", "exit_price", "gross_pnl", "costs", "pnl"):
        diff = float(np.abs(a[col].to_numpy() - b[col].to_numpy()).max())
        # Chunking must not perturb the arithmetic at all, not merely within
        # a tolerance - the same bars go through the same code either way.
        check(f"{label}: {col} identical", diff == 0.0, f"max diff {diff:.12g}")


# --------------------------------------------------------------------------
def test_chunk_parity_real_bars() -> None:
    """Chunked and unchunked must agree on real 1-minute bars."""
    print(f"\n[1] chunked vs unchunked, real 1m bars {START}..{END}")
    from mdlib.lake import get_bars

    rng = np.random.default_rng(20260813)

    for sym in ["ES", "GC", "ZT"]:
        bars = get_bars([sym], "1m", START, END)
        if bars.empty:
            check(f"{sym}: bars loaded", False, "empty")
            continue
        bars = bars.reset_index(drop=True)
        n = len(bars)
        print(f"    {sym}: {n:,} bars")

        # Trades every few hundred bars, so a chunk boundary lands mid-trade
        # often rather than occasionally.
        entries = pd.Series(rng.random(n) < 0.004)
        exits = pd.Series(rng.random(n) < 0.004)
        entries, exits = clean_signals(entries, exits)

        whole = _simulate(bars, entries, exits, sym,
                          BacktestConfig(contracts=2, chunk_size=0))

        # 997 is coprime with the bar spacing, so boundaries drift through
        # every phase of the trade cycle rather than landing consistently.
        for cs in (997, 10_000, 250_000):
            cfg = BacktestConfig(contracts=2, chunk_size=cs)
            chunked = _simulate(bars, entries, exits, sym, cfg)
            same_trades(chunked, whole, f"{sym} chunk={cs:,}")

        del bars, whole
        gc.collect()


def test_trade_straddling_a_boundary() -> None:
    """
    The year-slicing bug, in miniature.

    One trade, opened before the chunk boundary and closed after it. Naive
    slicing drops it. The boundary snapping must carry it whole.
    """
    print("\n[2] a trade that straddles a chunk boundary")
    n = 1000
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")
    bars = pd.DataFrame({
        "ts": idx, "symbol": "ES",
        "open": np.arange(n, dtype=float) + 100.0,
        "high": np.arange(n, dtype=float) + 100.0,
        "low": np.arange(n, dtype=float) + 100.0,
        "close": np.arange(n, dtype=float) + 100.0,
        "volume": np.full(n, 1000.0),
    })

    # Entry at bar 400 (fills 401), exit at bar 600 (fills 601). A chunk size
    # of 500 puts the boundary at bar 500 - right through the middle.
    entries = pd.Series(np.zeros(n, dtype=bool))
    exits = pd.Series(np.zeros(n, dtype=bool))
    entries.iloc[400] = True
    exits.iloc[600] = True

    whole = _simulate(bars, entries, exits, "ES", BacktestConfig(chunk_size=0))
    chunked = _simulate(bars, entries, exits, "ES", BacktestConfig(chunk_size=500))

    check("unchunked produced the trade", len(whole) == 1, f"got {len(whole)}")
    check("chunked did NOT drop the straddling trade", len(chunked) == 1,
          f"got {len(chunked)} - year-slicing would give 0")
    if len(whole) == 1 and len(chunked) == 1:
        r = chunked.iloc[0]
        check("entry fills at bar 401 open (501.0)", r.entry_price == 501.0,
              f"{r.entry_price}")
        check("exit fills at bar 601 open (701.0)", r.exit_price == 701.0,
              f"{r.exit_price}")
        check("P&L matches the unchunked run",
              r.pnl == whole.iloc[0].pnl, f"{r.pnl} vs {whole.iloc[0].pnl}")


def test_run_backtest_end_to_end() -> None:
    """Multi-symbol, through the public entry point, chunked vs unchunked."""
    print(f"\n[3] run_backtest multi-symbol, 1m {START}..{END}")

    syms = ["ES", "NQ", "GC"]

    def crossover(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = bars["close"]
        fast, slow = close.rolling(200).mean(), close.rolling(800).mean()
        return (((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False),
                ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False))

    def run(chunk_size: int):
        return run_backtest(syms, "1m", crossover, START, END,
                            BacktestConfig(contracts=1, trailing_drawdown_pct=5.0,
                                           chunk_size=chunk_size))

    whole = run(0)
    chunked = run(50_000)

    check("trades produced", len(whole.trades) > 0, f"{len(whole.trades)} trades")
    same_trades(chunked.trades, whole.trades, "run_backtest")

    for k in ("n_trades", "total_return_pct", "sharpe", "max_dd_pct",
              "total_costs", "gross_pnl", "net_pnl"):
        a, b = chunked.stats[k], whole.stats[k]
        check(f"stat {k} identical", a == b or (a != a and b != b), f"{a} vs {b}")

    eq_same = bool((chunked.equity.to_numpy() == whole.equity.to_numpy()).all())
    check("equity curve identical", eq_same)
    check("returns index identical",
          chunked.returns.index.equals(whole.returns.index))
    check("breach verdict identical",
          chunked.breach == whole.breach, str(chunked.breach))

    print("\n" + chunked.summary())


def test_memory_actually_drops() -> None:
    """
    Batching has to pay for its complexity in RAM.

    Peak RSS is a high-water mark for the whole process, so this runs the
    chunked pass FIRST in a clean process - if it ran second, the unchunked
    peak would already be banked and the comparison would be meaningless.
    """
    print("\n[4] peak RSS, chunked vs unchunked (largest single symbol)")
    from mdlib.lake import get_bars

    bars = get_bars(["GC"], "1m", "2016-01-01", END).reset_index(drop=True)
    n = len(bars)
    rng = np.random.default_rng(7)
    entries = pd.Series(rng.random(n) < 0.002)
    exits = pd.Series(rng.random(n) < 0.002)
    entries, exits = clean_signals(entries, exits)
    gc.collect()

    base = rss_gib()
    _simulate(bars, entries, exits, "GC", BacktestConfig(chunk_size=100_000))
    gc.collect()
    chunked_peak = rss_gib() - base
    print(f"    {n:,} bars  |  chunked peak +{chunked_peak:.3f} GiB")

    mid = rss_gib()
    _simulate(bars, entries, exits, "GC", BacktestConfig(chunk_size=0))
    gc.collect()
    whole_peak = rss_gib() - mid
    print(f"    {n:,} bars  |  unchunked peak +{whole_peak:.3f} GiB")

    check("chunking reduced peak memory", chunked_peak < whole_peak,
          f"chunked +{chunked_peak:.3f} GiB vs whole +{whole_peak:.3f} GiB")


if __name__ == "__main__":
    test_memory_actually_drops()      # first: needs a clean high-water mark
    test_chunk_parity_real_bars()
    test_trade_straddling_a_boundary()
    test_run_backtest_end_to_end()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
