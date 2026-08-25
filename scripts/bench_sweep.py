#!/usr/bin/env python3
"""
scripts/bench_sweep.py - measure screening throughput, serial against pooled.

Location: ~/src/trading/scripts/bench_sweep.py

WHAT IT MEASURES, AND WHY IT IS A SEPARATE SCRIPT
=================================================
`backtest/parallel.py` parallelises Stage 1's configuration loop. The claim
that makes is a WALL-CLOCK one and nothing else: the same rows, produced by N
processes instead of one. So the benchmark times the real
`backtest.baseline.run_symbol` over real bars rather than a synthetic stand-in,
because the thing being scaled is bar loading plus simulation plus profiling,
and a fixture would measure none of them in their true proportions.

It is separate from the stage because a benchmark that lived inside
`baseline.py` would be a second code path through the screen, free to diverge
from the one operators actually run.

**Throughput is reported two ways and they answer different questions.**
`configs/sec` is what a screen's ETA is built from. `bars/sec` is the one that
survives a change of window or timeframe - a 1-minute contract is ~60x the bars
of a 1-hour one, so a configs/sec figure from a 1h benchmark says nothing about
a 1m screen, while bars/sec roughly does.

**Speedup is measured, never derived from the worker count.** Eight workers is
not 8x: the lake reads contend on one NFS mount, the box is memory-bound before
it is core-bound, and Version B's classifier fits are already pinned to one
thread each. Reporting `jobs` as though it were the factor would overstate
every result.

THIS READS THE REAL LAKE, so it is an operator command. Claude Code does not
run it - see the tool boundary in CLAUDE.md.

    python3 scripts/bench_sweep.py --strat t3_braid_scalp_20260823 \\
        --symbols ES,NQ,CL,GC --tf 5m,15m,30m,1h \\
        --start 2018-01-01 --end 2022-12-31 --jobs 1,auto

    # The 1-minute row is the one that matters for a real screen, and the
    # slowest to produce. Give it its own run:
    python3 scripts/bench_sweep.py --strat t3_braid_scalp_20260823 \\
        --symbols ES,NQ --tf 1m --start 2021-01-01 --end 2022-12-31 --jobs 1,auto
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Before numpy loads anywhere: the pool gives each worker one thread, and the
# parent must match or the serial arm is measured with a different thread count
# from the pooled one, which is not a comparison.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from mdlib.env import load_env                                    # noqa: E402

load_env()

import argparse                                                   # noqa: E402

from backtest.parallel import (describe_plan, map_units,          # noqa: E402
                               resolve_jobs)


def _hms(s: float) -> str:
    s = int(max(0.0, s))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def count_bars(symbol: str, tf: str, start: str, end: str) -> int:
    """
    Bars one configuration will read. Counted, never estimated from a calendar.

    A session count times a bars-per-session constant is wrong by whatever the
    contract's holidays, half-days and coverage gaps amount to, and it is wrong
    in a way that scales with the window - which is exactly the axis a
    bars/sec figure is meant to be comparable across.
    """
    from backtest.run import load_bars
    return int(len(load_bars(symbol, tf, start, end)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strat", required=True)
    p.add_argument("--symbols", required=True, help="ES,NQ,CL,GC or ALL")
    p.add_argument("--tf", required=True, help="5m,15m,30m,1h or ALL_DAY_TRADING")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--jobs", default="1,auto",
                   help="comma-separated arms to time, e.g. '1,4,auto' "
                        "(default '1,auto' — the before/after pair)")
    p.add_argument("--ml", action="store_true",
                   help="also run Version B in every arm. OFF by default: it "
                        "dominates the timing and is the same work in both "
                        "arms, so it dilutes the ratio being measured.")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    from agents.tier3_workers import load_strategy
    from backtest.baseline import ScreenUnit, screen_unit
    from backtest.run import parse_symbols, parse_timeframes, resolve_strategy

    # Resolved exactly the way Stage 1 resolves it, through the same two
    # functions, so the benchmark cannot be measuring a different module or a
    # different symbol set from the screen it claims to describe.
    path = resolve_strategy(args.strat)
    _fn, info = load_strategy(path, {})
    module = info.get("module") if isinstance(info, dict) else None
    symbols = parse_symbols(args.symbols, getattr(module, "SYMBOLS", None))
    tfs = parse_timeframes(args.tf, getattr(module, "TIMEFRAME", None))
    pairs = [(s, t) for s in symbols for t in tfs]

    print("=" * 78)
    print(f"  THROUGHPUT BENCHMARK · {args.strat}")
    print("=" * 78)
    print(f"  window      : {args.start} -> {args.end}")
    print(f"  symbols     : {len(symbols)} · {', '.join(symbols)}")
    print(f"  timeframes  : {len(tfs)} · {', '.join(tfs)}")
    print(f"  configs     : {len(pairs)}")
    print(f"  Version B   : {'yes' if args.ml else 'no (--ml to include)'}")

    print("\n  counting bars (this reads the lake once per configuration)...")
    bars, unreadable = 0, []
    for sym, tf in pairs:
        try:
            bars += count_bars(sym, tf, args.start, args.end)
        except Exception as e:                                    # noqa: BLE001
            # A contract with no bars in this window is a fact about coverage,
            # not a benchmark failure. It is named and excluded from the
            # denominator rather than counted as zero, which would deflate
            # every bars/sec figure below it.
            unreadable.append(f"{sym}·{tf}: {type(e).__name__}")
    print(f"  total bars  : {bars:,}")
    if unreadable:
        print(f"  UNREADABLE  : {len(unreadable)} configuration(s) — "
              f"{'; '.join(unreadable[:4])}")

    ns = argparse.Namespace(**{**vars(args), "ml": args.ml,
                               "jobs": 1, "quiet": True})
    out_dir = Path(args.out_dir) if args.out_dir else None

    results = []
    for arm in [a.strip() for a in args.jobs.split(",") if a.strip()]:
        jobs, reason = resolve_jobs(arm, len(pairs))
        print("\n" + "-" * 78)
        print(describe_plan(len(pairs), jobs, reason))
        units = [ScreenUnit(s, t, path, {}, ns, {}, f"[{i}/{len(pairs)}]",
                            out_dir)
                 for i, (s, t) in enumerate(pairs, 1)]
        t0 = time.monotonic()
        outcome = map_units(screen_unit, units, jobs=jobs)
        elapsed = time.monotonic() - t0
        ok = len(outcome.results)
        results.append({
            "arm": arm, "jobs": jobs, "elapsed": elapsed, "ok": ok,
            "failed": len(outcome.errors),
            "cps": ok / elapsed if elapsed > 0 else 0.0,
            "bps": bars / elapsed if elapsed > 0 and ok == len(pairs) else None,
        })
        print(f"  -> {_hms(elapsed)}  ·  {ok}/{len(pairs)} completed"
              + (f"  ·  {len(outcome.errors)} error(s)" if outcome.errors else ""))

    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"  {'ARM':>6}  {'JOBS':>4}  {'ELAPSED':>9}  {'CONFIGS/SEC':>12}  "
          f"{'BARS/SEC':>12}  {'SPEEDUP':>8}")
    base = results[0]["elapsed"] if results else 0.0
    for r in results:
        bps = f"{r['bps']:,.0f}" if r["bps"] else "--"
        # Speedup against the FIRST arm, which is the before in before/after.
        # It is a measurement, never `jobs` presented as a factor.
        spd = f"{base / r['elapsed']:.2f}x" if r["elapsed"] > 0 and base else "--"
        print(f"  {r['arm']:>6}  {r['jobs']:>4}  {_hms(r['elapsed']):>9}  "
              f"{r['cps']:>12.3f}  {bps:>12}  {spd:>8}")
    if len(results) > 1:
        print(f"\n  Speedup is measured against arm '{results[0]['arm']}', not "
              f"derived from the worker count. It is bounded well below the\n"
              f"  worker count by NFS read contention and by this box being "
              f"memory-bound before it is core-bound.")
    if any(r["failed"] for r in results):
        print("\n  NOTE  some configurations errored; bars/sec is withheld for "
              "any arm that did not complete every one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
