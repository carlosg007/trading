#!/usr/bin/env python3
"""
test_streaming_lake.py - does reading one symbol at a time change anything?

Location:  ~/src/trading/tests/test_streaming_lake.py

Run:  python tests/test_streaming_lake.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

What is being proved
--------------------
`get_bars` materialises every symbol, concatenates them and sorts the result
by timestamp. On the full 1-minute lake that peaks around 20 GiB, and most of
it is not the data:

    holding all 27 per-symbol frames        7.1 GiB
    + pd.concat into one frame             12.0 GiB
    + sort_values(["ts", "symbol"])        15.7 GiB

`iter_bars` yields the same bars one symbol at a time, and
`run_backtest_streaming` consumes them that way. Two claims follow and both
are checked here:

1. `iter_bars` returns EXACTLY what `get_bars` returns, once reassembled -
   same rows, same columns, same dtypes, same order. If the streaming reader
   quietly dropped or reordered bars, every downstream number would move.
2. `run_backtest_streaming` produces the same trades, equity and stats as
   `run_backtest`, given the same per-symbol signals.

Point 2 has a trap worth naming. `get_bars` returns a frame interleaved by
timestamp, so a strategy computing `close.rolling(200)` on it is averaging
across 27 different instruments - the windows bleed between symbols and the
signals are nonsense. `run_backtest_streaming` calls the strategy once per
symbol, which cannot do that. So the reference path here deliberately applies
the SAME strategy per symbol before calling `run_backtest`; comparing against
the naive whole-frame version would be comparing against a bug.

Memory is measured rather than asserted in the abstract: peak RSS for both
paths over the same symbols, in separate processes so neither inherits the
other's high-water mark.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from backtest.engine import (BacktestConfig, run_backtest,
                             run_backtest_streaming)
from mdlib.lake import get_bars, iter_bars

FAILURES: list[str] = []

SYMS = ["ES", "NQ", "GC", "ZN"]
START, END = "2022-01-01", "2024-01-01"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def stats_equal(a: dict, b: dict) -> bool:
    """
    Compare two stats dicts treating NaN as equal to NaN.

    `sharpe` is NaN whenever the return series has no variance, so a plain
    `a == b` reports two identical results as different.
    """
    if a.keys() != b.keys():
        return False
    return all(x == y or (isinstance(x, float) and isinstance(y, float)
                          and np.isnan(x) and np.isnan(y))
               for x, y in ((a[k], b[k]) for k in a))


def strategy(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    A 200/800 SMA crossover. The strategies/ contract: bars in, signals out.

    Deliberately uses a long lookback, because a long window is what makes
    cross-symbol bleed visible if it ever happens.
    """
    close = bars["close"]
    fast, slow = close.rolling(200).mean(), close.rolling(800).mean()
    entries = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False)
    exits = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False)
    return entries, exits


def signals_per_symbol(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Apply `strategy` to each symbol of a long frame and reassemble.

    This is what the streaming path does internally, done by hand so
    run_backtest can be given identical inputs. Anything else would compare
    the streaming engine against a strategy that bleeds across symbols.
    """
    entries = pd.Series(False, index=bars.index)
    exits = pd.Series(False, index=bars.index)
    for _, idx in bars.groupby("symbol", sort=True).groups.items():
        g = bars.loc[idx]
        e, x = strategy(g.reset_index(drop=True))
        entries.loc[idx] = e.to_numpy()
        exits.loc[idx] = x.to_numpy()
    return entries, exits


# --------------------------------------------------------------------------
def test_iter_bars_matches_get_bars() -> None:
    """Reassembled iter_bars output must be byte-identical to get_bars."""
    print(f"\n[1] iter_bars vs get_bars, {START}..{END}")

    for tf, kwargs in (("1m", {}),
                       ("1d", {}),
                       ("1h", {}),
                       ("1d", {"exclude_degraded": True, "exclude_rolls": True}),
                       ("1d", {"session_merge": False})):
        label = f"tf={tf} {kwargs or 'defaults'}"

        whole = get_bars(SYMS, tf, START, END, **kwargs)
        parts = list(iter_bars(SYMS, tf, START, END, **kwargs))

        rebuilt = (pd.concat([d for _, d in parts], ignore_index=True)
                     .sort_values(["ts", "symbol"])
                     .reset_index(drop=True))

        check(f"{label}: same shape", whole.shape == rebuilt.shape,
              f"{whole.shape} vs {rebuilt.shape}")
        check(f"{label}: same columns", list(whole.columns) == list(rebuilt.columns))
        check(f"{label}: same dtypes",
              whole.dtypes.equals(rebuilt.dtypes),
              "" if whole.dtypes.equals(rebuilt.dtypes) else
              f"{dict(whole.dtypes)} vs {dict(rebuilt.dtypes)}")
        try:
            pd.testing.assert_frame_equal(whole, rebuilt, check_exact=True)
            same = True
            detail = f"{len(whole):,} rows"
        except AssertionError as exc:                    # noqa: BLE001
            same = False
            detail = str(exc).splitlines()[0]
        check(f"{label}: frames identical", same, detail)

        # One symbol per yield, in the order requested, none empty.
        check(f"{label}: one frame per symbol",
              [s for s, _ in parts] == [s for s in SYMS
                                        if s in set(whole["symbol"])],
              str([s for s, _ in parts]))
        check(f"{label}: each frame is a single symbol",
              all(d["symbol"].nunique() == 1 for _, d in parts))
        check(f"{label}: each frame sorted by ts",
              all(d["ts"].is_monotonic_increasing for _, d in parts))


def test_streaming_backtest_matches() -> None:
    """Streaming vs in-memory, same signals, trade for trade."""
    print(f"\n[2] run_backtest_streaming vs run_backtest, 1m {START}..{END}")

    cfg = lambda: BacktestConfig(contracts=1, trailing_drawdown_pct=5.0)

    bars = get_bars(SYMS, "1m", START, END)
    entries, exits = signals_per_symbol(bars)
    print(f"    {len(bars):,} bars, {int(entries.sum()):,} raw entry signals")
    whole = run_backtest(bars, entries, exits, cfg())
    del bars, entries, exits
    gc.collect()

    streamed = run_backtest_streaming(SYMS, "1m", strategy, START, END, cfg())

    check("trades produced", len(whole.trades) > 0, f"{len(whole.trades)} trades")
    check("same trade count", len(streamed.trades) == len(whole.trades),
          f"streaming {len(streamed.trades)} vs whole {len(whole.trades)}")

    if len(streamed.trades) == len(whole.trades) and not whole.trades.empty:
        for col in ("entry_time", "exit_time", "symbol", "direction"):
            eq = bool((streamed.trades[col].to_numpy()
                       == whole.trades[col].to_numpy()).all())
            check(f"{col} identical", eq)
        for col in ("entry_price", "exit_price", "gross_pnl", "costs", "pnl"):
            diff = float(np.abs(streamed.trades[col].to_numpy()
                                - whole.trades[col].to_numpy()).max())
            check(f"{col} identical", diff == 0.0, f"max diff {diff:.12g}")

    for k in ("n_trades", "total_return_pct", "sharpe", "max_dd_pct",
              "total_costs", "gross_pnl", "net_pnl"):
        a, b = streamed.stats[k], whole.stats[k]
        check(f"stat {k} identical", a == b or (a != a and b != b), f"{a} vs {b}")

    check("equity curve identical",
          bool((streamed.equity.to_numpy() == whole.equity.to_numpy()).all()))
    check("returns index identical",
          streamed.returns.index.equals(whole.returns.index))
    check("breach verdict identical", streamed.breach == whole.breach,
          str(streamed.breach))

    print("\n" + streamed.summary())


def test_symbol_order_does_not_matter() -> None:
    """
    Streaming walks symbols in the order requested; run_backtest walks them
    sorted. The pooled result must not depend on either.
    """
    print("\n[3] result is independent of symbol order")
    cfg = BacktestConfig(contracts=1)

    # 1h, not 1d: a 200/800 SMA needs 800 bars, and two years of daily bars is
    # only ~520, so a daily run here produces no trades at all and would prove
    # nothing about ordering.
    a = run_backtest_streaming(SYMS, "1h", strategy, START, END, cfg)
    b = run_backtest_streaming(list(reversed(SYMS)), "1h", strategy,
                               START, END, cfg)

    check("trades produced", len(a.trades) > 0, f"{len(a.trades)} trades")
    check("same trade count", len(a.trades) == len(b.trades),
          f"{len(a.trades)} vs {len(b.trades)}")
    if len(a.trades) == len(b.trades) and not a.trades.empty:
        same = all(bool((a.trades[c].to_numpy() == b.trades[c].to_numpy()).all())
                   for c in ("entry_time", "exit_time", "symbol", "pnl"))
        check("trade rows in identical order", same)
    check("stats identical", stats_equal(a.stats, b.stats))


def test_flat_by_close_path() -> None:
    """The Portfolio A constraint path also has to agree."""
    print("\n[4] flat_by_close through both paths")
    cfg = lambda: BacktestConfig(contracts=1, flat_by_close=True)

    bars = get_bars(SYMS, "1h", START, END)
    entries, exits = signals_per_symbol(bars)
    whole = run_backtest(bars, entries, exits, cfg())
    del bars, entries, exits
    gc.collect()

    streamed = run_backtest_streaming(SYMS, "1h", strategy, START, END, cfg())

    check("same trade count", len(streamed.trades) == len(whole.trades),
          f"{len(streamed.trades)} vs {len(whole.trades)}")
    check("net P&L identical",
          streamed.stats["net_pnl"] == whole.stats["net_pnl"],
          f"{streamed.stats['net_pnl']} vs {whole.stats['net_pnl']}")


def test_bad_signal_length_is_rejected() -> None:
    """A strategy returning the wrong length must fail loudly, not silently."""
    print("\n[5] a mis-sized signal is rejected")

    def bad(bars):
        e, x = strategy(bars)
        return e.iloc[:-5], x

    try:
        run_backtest_streaming(["ES"], "1d", bad, START, END, BacktestConfig())
        check("raises on length mismatch", False, "no exception")
    except ValueError as exc:
        check("raises on length mismatch", True, str(exc)[:70])


def test_peak_memory() -> None:
    """
    Peak RSS of both paths, each in its own process.

    Separate processes matter: peak RSS is a per-process high-water mark, so
    running both in one would let the first path's peak mask the second's.
    """
    print("\n[6] peak RSS above baseline, streaming vs whole-frame")

    # Two measurement traps, both hit on the first attempt at this test:
    #
    #  - Interpreter and library imports cost 0.357 GiB, and vectorbt's numba
    #    kernels cost ~1.2 GiB more the first time from_signals runs. That is
    #    a fixed cost both paths pay. Measured naively at a small scale it
    #    swamps the data entirely and both paths report an identical peak.
    #    So the JIT is warmed on a 5-bar backtest first, and the baseline is
    #    taken after it.
    #  - ru_maxrss is a high-water mark that cannot be reset, so the baseline
    #    is subtracted from a peak sampled live from /proc/self/statm instead.
    #
    # Full history rather than the 2-year window used elsewhere, so the data
    # is large enough for the difference to be the thing being measured.
    prog = r'''
import gc, sys, threading, time, warnings
sys.path.insert(0, %r)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from backtest.engine import BacktestConfig, run_backtest, run_backtest_streaming
from mdlib.lake import get_bars
sys.path.insert(0, %r)
from test_streaming_lake import strategy, signals_per_symbol

def live():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 1024**3

# Warm the vectorbt JIT so its one-off compilation is inside the baseline.
_w = pd.DataFrame({"ts": pd.date_range("2024-01-02", periods=5, freq="D", tz="UTC"),
                   "symbol": "ES", "open": [1.,2.,3.,4.,5.],
                   "high": [1.,2.,3.,4.,5.], "low": [1.,2.,3.,4.,5.],
                   "close": [1.,2.,3.,4.,5.], "volume": [1.]*5})
run_backtest(_w, pd.Series([True,False,False,False,False]),
             pd.Series([False,False,True,False,False]), BacktestConfig())
del _w
gc.collect()

BASE = live()
PEAK = BASE
def watch():
    global PEAK
    while True:
        PEAK = max(PEAK, live()); time.sleep(0.02)
threading.Thread(target=watch, daemon=True).start()

SYMS, START, END = %r, %r, %r
mode = sys.argv[1]
if mode == "stream":
    r = run_backtest_streaming(SYMS, "1m", strategy, START, END, BacktestConfig())
else:
    b = get_bars(SYMS, "1m", START, END)
    e, x = signals_per_symbol(b)
    r = run_backtest(b, e, x, BacktestConfig())
print(f"{PEAK-BASE:.3f} {BASE:.3f} {r.stats['net_pnl']:.6f} {r.stats['n_trades']}")
''' % (str(REPO), str(REPO / "tests"), SYMS, None, END)

    out = {}
    for mode in ("stream", "whole"):
        r = subprocess.run([sys.executable, "-c", prog, mode],
                           capture_output=True, text=True, cwd=str(REPO))
        if r.returncode != 0:
            check(f"{mode} subprocess ran", False, r.stderr.strip()[-400:])
            return
        d, base, pnl, n = r.stdout.strip().split()
        out[mode] = (float(d), float(pnl), int(n))
        print(f"    {mode:<7} peak +{float(d):6.3f} GiB above a {float(base):.3f} GiB "
              f"baseline   net P&L {float(pnl):,.2f}   {int(n):,} trades")

    check("both paths agree on net P&L", out["stream"][1] == out["whole"][1],
          f"{out['stream'][1]} vs {out['whole'][1]}")
    check("both paths agree on trade count", out["stream"][2] == out["whole"][2],
          f"{out['stream'][2]:,} vs {out['whole'][2]:,}")
    check("streaming peak is lower", out["stream"][0] < out["whole"][0],
          f"+{out['stream'][0]:.3f} vs +{out['whole'][0]:.3f} GiB "
          f"({out['whole'][0]/max(out['stream'][0], 1e-9):.1f}x less)")


if __name__ == "__main__":
    test_iter_bars_matches_get_bars()
    test_streaming_backtest_matches()
    test_symbol_order_does_not_matter()
    test_flat_by_close_path()
    test_bad_signal_length_is_rejected()
    test_peak_memory()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
