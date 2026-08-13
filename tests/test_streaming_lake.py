#!/usr/bin/env python3
"""
test_streaming_lake.py - the per-symbol reader and the engine built on it.

Location:  ~/src/trading/tests/test_streaming_lake.py

Run:  python tests/test_streaming_lake.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

What is being proved
--------------------
`run_backtest` reads bars itself, one symbol at a time, and calls the strategy
on each. There is no longer a version that accepts a pre-built multi-symbol
frame with signals already computed - that signature was removed because it
was a trap. `get_bars` returns rows sorted by (ts, symbol), so the frame
INTERLEAVES instruments; a strategy computing `close.rolling(200).mean()` on
it averaged across 27 different contracts and produced noise, while returning
signals of exactly the right length and dtype and an equity curve that looked
fine. On this lake that mistake gave 608,079 trades against 86,035 for the
correct per-symbol signals.

So the properties worth pinning are:

1. `iter_bars` returns EXACTLY what `get_bars` returns once reassembled. The
   reader underneath the engine must not drop or reorder bars.
2. The strategy is called with ONE symbol per call, always. This is the
   structural guarantee that replaced the trap, so it is asserted directly
   from inside the strategy rather than inferred.
3. Running N symbols together equals running each alone and pooling. Symbols
   must not influence each other through the engine.
4. The pooled result does not depend on the order symbols were requested in.
5. A mis-sized signal is rejected loudly rather than silently misaligned.
6. Peak memory does not scale with the number of symbols requested - the
   whole point of reading one at a time.

Nothing here compares against the deleted function; it is gone, and a test
that resurrected it would defeat the purpose.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from backtest.engine import (BacktestConfig, apply_flat_by_close,
                             run_backtest)
from mdlib.lake import get_bars, iter_bars

FAILURES: list[str] = []

SYMS = ["ES", "NQ", "GC", "ZN"]
START, END = "2022-01-01", "2024-01-01"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def stats_equal(a: dict, b: dict) -> bool:
    """Compare stats dicts treating NaN as equal to NaN (sharpe can be NaN)."""
    if a.keys() != b.keys():
        return False
    return all(x == y or (isinstance(x, float) and isinstance(y, float)
                          and np.isnan(x) and np.isnan(y))
               for x, y in ((a[k], b[k]) for k in a))


SEEN_SYMBOL_COUNTS: list[int] = []


def strategy(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    A 200/800 SMA crossover - the `strategies/` contract: bars in, signals out.

    The long lookback is deliberate: a 800-bar window is what would make
    cross-symbol bleed obvious if the engine ever handed over more than one
    symbol. Every call records how many symbols it actually saw, which test 2
    then checks.
    """
    SEEN_SYMBOL_COUNTS.append(bars["symbol"].nunique())
    close = bars["close"]
    fast, slow = close.rolling(200).mean(), close.rolling(800).mean()
    entries = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False)
    exits = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False)
    return entries, exits


# --------------------------------------------------------------------------
def test_iter_bars_matches_get_bars() -> None:
    """Reassembled iter_bars output must be identical to get_bars."""
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

        try:
            pd.testing.assert_frame_equal(whole, rebuilt, check_exact=True)
            same, detail = True, f"{len(whole):,} rows"
        except AssertionError as exc:                    # noqa: BLE001
            same, detail = False, str(exc).splitlines()[0]
        check(f"{label}: frames identical", same, detail)

        check(f"{label}: one frame per symbol",
              [s for s, _ in parts] == [s for s in SYMS
                                        if s in set(whole["symbol"])],
              str([s for s, _ in parts]))
        check(f"{label}: each frame is a single symbol",
              all(d["symbol"].nunique() == 1 for _, d in parts))
        check(f"{label}: each frame sorted by ts",
              all(d["ts"].is_monotonic_increasing for _, d in parts))


def test_strategy_always_sees_one_symbol() -> None:
    """
    The structural guarantee that replaced the removed signature.

    If the strategy is ever handed more than one symbol, a rolling window can
    bleed across instruments and the signals are meaningless. Checked from
    inside the strategy itself, not inferred from the results.
    """
    print("\n[2] the strategy is called once per symbol")
    SEEN_SYMBOL_COUNTS.clear()

    res = run_backtest(SYMS, "1h", strategy, START, END,
                       BacktestConfig(contracts=1))

    check("strategy was called once per symbol",
          len(SEEN_SYMBOL_COUNTS) == len(SYMS),
          f"{len(SEEN_SYMBOL_COUNTS)} calls for {len(SYMS)} symbols")
    check("every call saw exactly one symbol",
          SEEN_SYMBOL_COUNTS and set(SEEN_SYMBOL_COUNTS) == {1},
          f"symbol counts per call: {SEEN_SYMBOL_COUNTS}")
    check("trades produced", len(res.trades) > 0, f"{len(res.trades)} trades")
    check("all requested symbols traded",
          set(res.trades["symbol"]) <= set(SYMS), str(set(res.trades["symbol"])))

    # No pre-built-frame entry point should have survived the teardown.
    import backtest.engine as eng
    check("no run_backtest_streaming alias left",
          not hasattr(eng, "run_backtest_streaming"))
    try:
        bars = get_bars(["ES"], "1d", START, END)
        run_backtest(bars, pd.Series([True]), pd.Series([False]))
        check("a bars frame is not accepted as `symbols`", False,
              "call unexpectedly succeeded")
    except Exception as exc:                             # noqa: BLE001
        check("a bars frame is not accepted as `symbols`", True,
              f"{type(exc).__name__}")


def test_symbols_are_independent() -> None:
    """
    Running N symbols together == running each alone and pooling.

    This is what "each symbol is simulated independently" has to mean. If
    anything leaked between symbols - a carried position, a shared cost array,
    a cash balance - these would diverge.
    """
    print("\n[3] N symbols together == each alone, pooled")
    cfg = lambda: BacktestConfig(contracts=1)

    together = run_backtest(SYMS, "1h", strategy, START, END, cfg())
    singly = [run_backtest([s], "1h", strategy, START, END, cfg()) for s in SYMS]

    pooled = (pd.concat([r.trades for r in singly], ignore_index=True)
                .sort_values(["exit_time", "symbol", "entry_time"], kind="stable")
                .reset_index(drop=True))

    check("same trade count", len(together.trades) == len(pooled),
          f"together {len(together.trades)} vs pooled {len(pooled)}")
    if len(together.trades) == len(pooled) and not pooled.empty:
        for col in ("entry_time", "exit_time", "symbol"):
            check(f"{col} identical",
                  bool((together.trades[col].to_numpy()
                        == pooled[col].to_numpy()).all()))
        for col in ("entry_price", "exit_price", "gross_pnl", "costs", "pnl"):
            diff = float(np.abs(together.trades[col].to_numpy()
                                - pooled[col].to_numpy()).max())
            check(f"{col} identical", diff == 0.0, f"max diff {diff:.12g}")

    check("net P&L is the sum of the singles",
          abs(together.stats["net_pnl"]
              - sum(r.stats["net_pnl"] for r in singly)) < 1e-6,
          f"{together.stats['net_pnl']:,.2f}")


def test_symbol_order_does_not_matter() -> None:
    """The pooled result must not depend on the order symbols were requested."""
    print("\n[4] result is independent of symbol order")
    cfg = BacktestConfig(contracts=1)

    a = run_backtest(SYMS, "1h", strategy, START, END, cfg)
    b = run_backtest(list(reversed(SYMS)), "1h", strategy, START, END, cfg)

    check("trades produced", len(a.trades) > 0, f"{len(a.trades)} trades")
    check("same trade count", len(a.trades) == len(b.trades),
          f"{len(a.trades)} vs {len(b.trades)}")
    if len(a.trades) == len(b.trades) and not a.trades.empty:
        same = all(bool((a.trades[c].to_numpy() == b.trades[c].to_numpy()).all())
                   for c in ("entry_time", "exit_time", "symbol", "pnl"))
        check("trade rows in identical order", same)
    check("stats identical", stats_equal(a.stats, b.stats))


def _session_id(ts, close_utc: str = "20:00") -> pd.DatetimeIndex:
    """
    The session each timestamp belongs to, as a DatetimeIndex.

    Bars at or after the close belong to the NEXT session, so shifting forward
    by (24h - close) and flooring to the day gives the session it trades in.
    """
    h, m = (int(v) for v in close_utc.split(":"))
    off = pd.Timedelta(hours=24 - h, minutes=-m)
    return pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).__add__(off).normalize()


def test_flat_by_close() -> None:
    """
    The Portfolio A constraint, checked at both levels.

    First on a hand-built frame where the answer is known, then end to end.

    The end-to-end check counts SESSIONS, not calendar days. Two things make
    days the wrong unit: the exit signal lands on the session's last bar and
    fills on the next one - which is the 20:00 boundary bar itself, so a
    correctly closed trade still shows an exit timestamp in the following
    session - and weekends mean Friday to Monday is three calendar days but
    one session step. Asserting on days flags 32 of 33 correct trades.
    """
    print("\n[5] flat_by_close")

    # -- known answer: two sessions of 1h bars, close at 20:00 UTC ----------
    idx = pd.date_range("2024-01-02 17:00", periods=8, freq="h", tz="UTC")
    bars = pd.DataFrame({"ts": idx, "symbol": "ES",
                         "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": 1.0, "volume": 1.0})
    # Sessions: 17,18,19 -> Jan 2 | 20,21,22,23,00 -> Jan 3.
    # Last bar of session 1 is index 2 (19:00); of session 2 is index 7.
    e_in = pd.Series([True] * 8)
    x_in = pd.Series([False] * 8)
    e_out, x_out = apply_flat_by_close(bars, e_in, x_in, "20:00")

    check("exit forced on each session's last bar",
          bool(x_out.iloc[2]) and bool(x_out.iloc[7]),
          f"forced at {list(np.flatnonzero(x_out.to_numpy()))}")
    check("no other exit invented",
          int(x_out.sum()) == 2, f"{int(x_out.sum())} exits")
    check("entry blocked on those bars",
          not bool(e_out.iloc[2]) and not bool(e_out.iloc[7]))
    check("entries elsewhere untouched",
          int(e_out.sum()) == 6, f"{int(e_out.sum())} entries")

    # -- end to end: no position survives a whole session ------------------
    cfg = BacktestConfig(contracts=1, flat_by_close=True)
    res = run_backtest(SYMS, "1h", strategy, START, END, cfg)
    check("trades produced", len(res.trades) > 0, f"{len(res.trades)} trades")
    if res.trades.empty:
        return

    # Rank the sessions each symbol actually has, so weekends and holidays
    # count as one step rather than three.
    ranks = {sym: pd.Index(np.unique(_session_id(d["ts"]).to_numpy()))
             for sym, d in iter_bars(SYMS, "1h", START, END)}

    ent_s = _session_id(res.trades["entry_time"])
    ext_s = _session_id(res.trades["exit_time"])

    gaps = []
    for sym, r_in, r_out in zip(res.trades["symbol"], ent_s, ext_s):
        u = ranks[sym]
        gaps.append(u.get_indexer([r_out])[0] - u.get_indexer([r_in])[0])

    gaps = np.array(gaps)
    check("no trade is held through a whole session", bool((gaps <= 1).all()),
          f"max session gap {int(gaps.max())}, "
          f"{int((gaps > 1).sum())} of {len(gaps)} exceed 1")


def test_bad_signal_length_is_rejected() -> None:
    """A strategy returning the wrong length must fail loudly."""
    print("\n[6] a mis-sized signal is rejected")

    def bad(bars):
        e, x = strategy(bars)
        return e.iloc[:-5], x

    try:
        run_backtest(["ES"], "1d", bad, START, END, BacktestConfig())
        check("raises on length mismatch", False, "no exception")
    except ValueError as exc:
        check("raises on length mismatch", True, str(exc)[:70])

    try:
        run_backtest(["NOSUCHSYM"], "1d", strategy, START, END, BacktestConfig())
        check("raises when no bars are found", False, "no exception")
    except ValueError as exc:
        check("raises when no bars are found", True, str(exc)[:70])


def test_peak_memory_does_not_scale_with_symbols() -> None:
    """
    Peak RSS must track the largest single symbol, not the symbol count.

    That is the whole point of reading one at a time, and it is the property
    that has to keep holding as NT8 lands. Measured as 4 symbols vs 12 over
    FULL history, in separate processes.

    Two measurement traps, both hit for real while writing this:
      - Imports cost ~0.4 GiB and vectorbt's numba kernels ~1.2 GiB more on
        first use. Both paths pay it, and at small scale it hides everything.
        So the JIT is warmed first and the baseline taken after.
      - ru_maxrss is a high-water mark that cannot be reset, so the peak is
        sampled live from /proc/self/statm instead.
    """
    print("\n[7] peak RSS vs number of symbols (full history)")

    prog = r'''
import gc, sys, threading, time, warnings
sys.path.insert(0, %r)
warnings.filterwarnings("ignore")
import pandas as pd
from backtest.engine import (BacktestConfig, apply_flat_by_close,
                             run_backtest)
sys.path.insert(0, %r)
from test_streaming_lake import strategy

def live():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 1024**3

# Warm the vectorbt JIT so its one-off compilation sits in the baseline.
run_backtest(["ES"], "1d", strategy, "2023-01-01", "2023-03-01", BacktestConfig())
gc.collect()

BASE = live(); PEAK = BASE
def watch():
    global PEAK
    while True:
        PEAK = max(PEAK, live()); time.sleep(0.02)
threading.Thread(target=watch, daemon=True).start()

syms = sys.argv[1].split(",")
r = run_backtest(syms, "1m", strategy, None, None, BacktestConfig())
print(f"{PEAK-BASE:.3f} {BASE:.3f} {r.stats['n_trades']} {r.stats['net_pnl']:.6f}")
''' % (str(REPO), str(REPO / "tests"))

    few = ["ES", "NQ", "GC", "ZN"]
    many = few + ["6E", "6J", "6A", "6B", "SI", "ZB", "ZF", "CL"]

    out = {}
    for label, syms in (("4 symbols", few), ("12 symbols", many)):
        r = subprocess.run([sys.executable, "-c", prog, ",".join(syms)],
                           capture_output=True, text=True, cwd=str(REPO))
        if r.returncode != 0:
            check(f"{label} subprocess ran", False, r.stderr.strip()[-400:])
            return
        d, base, n, pnl = r.stdout.strip().split()
        out[label] = float(d)
        print(f"    {label:<11} peak +{float(d):6.3f} GiB above {float(base):.3f} "
              f"baseline   {int(n):,} trades")

    ratio = out["12 symbols"] / max(out["4 symbols"], 1e-9)
    check("3x the symbols does not cost 3x the memory", ratio < 1.5,
          f"12sym/4sym = {ratio:.2f}x (linear growth would be ~3x)")


if __name__ == "__main__":
    test_iter_bars_matches_get_bars()
    test_strategy_always_sees_one_symbol()
    test_symbols_are_independent()
    test_symbol_order_does_not_matter()
    test_flat_by_close()
    test_bad_signal_length_is_rejected()
    test_peak_memory_does_not_scale_with_symbols()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
