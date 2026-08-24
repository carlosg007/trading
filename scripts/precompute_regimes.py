#!/usr/bin/env python3
"""
Build the pre-computed regime feature cache.

One pass per (symbol, timeframe): read the bars through `mdlib.lake`, compute
Wilder's ADX(14) and ATR(14) over the WHOLE history, take the median ATR over
the in-sample window alone as the volatility boundary, and write the four
quadrant labels to `<cache>/{SYMBOL}_{TF}_regime.parquet`.

    python3 scripts/precompute_regimes.py --symbols NQ,GC --tf 15m,30m
    python3 scripts/precompute_regimes.py --symbols ALL --tf 15m \
        --is-start 2013-01-01 --is-end 2022-12-31

Reading is done through `mdlib.lake.iter_bars` rather than off the parquet
directly, so the cache is built on exactly the bars a backtest sees - the same
1m aggregation, the same Sunday merge, the same column dtypes. A second reader
here would be free to disagree with the one every stage uses about where a 15m
bar starts, and the join back on would still match every timestamp.

`regimes=False` on that read, necessarily: the cache cannot be built from a
frame that is waiting for the cache.

The threshold is deliberately NOT the median of the full series. See
`mdlib/regimes.py` for why, and for why `regime_quadrant == 0` is a warm-up
sentinel rather than a quadrant.

One contract failing does not end the run - a symbol with no bars at a
timeframe is recorded as an error and the loop moves on, and the process exits
non-zero if anything failed. Half a cache is more useful than none, and a
silent partial build is not.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ, so an operator
# opening a fresh terminal never has to `source .env` first. It runs at import
# time, above the imports below, because modules resolve their BT_* variables
# while being imported and loading the file inside main() would be too late for
# those - and would work here, which is the kind of difference nobody notices
# until one runner silently uses the default path. The rules live in ONE
# module: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from mdlib import lake, regimes  # noqa: E402


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def resolve_symbols(raw: str) -> list[str]:
    if raw.strip().upper() == "ALL":
        return list(lake.available_symbols())
    return parse_list(raw)


def resolve_timeframes(raw: str) -> list[str]:
    """Validate every timeframe up front - a typo should cost a second, not a
    full sweep of the contracts that came before it."""
    tfs = parse_list(raw)
    known = set(lake.NATIVE_TFS) | set(lake.DERIVED)
    bad = [t for t in tfs if t not in known]
    if bad:
        raise SystemExit(f"unknown timeframe(s) {bad}. "
                         f"Known: {sorted(known)}")
    return tfs


def build_one(symbol: str, tf: str, is_start: str, is_end: str,
              root: Path | None, force: bool) -> dict:
    path = regimes.cache_path(symbol, tf, root)
    if path.exists() and not force:
        return {"symbol": symbol, "tf": tf, "status": "SKIPPED",
                "note": "exists; --force to rebuild", "path": str(path)}

    # regimes=False: this IS the builder.
    frames = list(lake.iter_bars(symbol, tf, regimes=False))
    if not frames:
        return {"symbol": symbol, "tf": tf, "status": "ERROR",
                "note": "no bars in the lake for this pair"}
    _, bars = frames[0]

    frame, theta = regimes.build(bars, is_start, is_end)
    written = regimes.write_cache(
        frame, symbol, tf, theta, is_start, is_end,
        # The flags this build read under, recorded so a later join onto bars
        # read under different hygiene can say so rather than matching
        # timestamps and quietly serving an ATR from another series.
        bar_flags={"session_merge": True, "exclude_degraded": False,
                   "exclude_rolls": False, "respect_coverage": False},
        root=root,
    )

    counts = frame["regime_quadrant"].value_counts()
    return {
        "symbol": symbol, "tf": tf, "status": "OK", "path": str(written),
        "rows": len(frame), "theta_vol": theta,
        "first_ts": frame.index[0], "last_ts": frame.index[-1],
        "undefined": int(counts.get(regimes.QUADRANT_UNDEFINED, 0)),
        "q1": int(counts.get(1, 0)), "q2": int(counts.get(2, 0)),
        "q3": int(counts.get(3, 0)), "q4": int(counts.get(4, 0)),
    }


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("REGIME CACHE  ·  quadrant counts")
    print("=" * 100)
    head = (f"{'Symbol':<8} {'TF':<5} {'Status':<8} {'Rows':>10} "
            f"{'theta_vol':>11} {'HV/Trend':>9} {'HV/Range':>9} "
            f"{'LV/Trend':>9} {'LV/Range':>9} {'warm-up':>8}")
    print(head)
    print("-" * len(head))
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['symbol']:<8} {r['tf']:<5} {r['status']:<8} "
                  f"{r.get('note', '')}")
            continue
        print(f"{r['symbol']:<8} {r['tf']:<5} {r['status']:<8} "
              f"{r['rows']:>10,} {r['theta_vol']:>11.4f} "
              f"{r['q1']:>9,} {r['q2']:>9,} {r['q3']:>9,} {r['q4']:>9,} "
              f"{r['undefined']:>8,}")
    print("=" * 100)
    print("regime_quadrant: 1 HV/Trending  2 HV/Ranging  3 LV/Trending  "
          "4 LV/Ranging  0 UNDEFINED (ADX/ATR warm-up)")
    print("theta_vol is the MEDIAN ATR(14) over the in-sample window only, "
          "applied to the whole series.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-compute the ADX/ATR regime quadrant cache.")
    ap.add_argument("--symbols", required=True,
                    help="NQ,GC or ALL")
    ap.add_argument("--tf", required=True,
                    help="15m,30m - one or a comma-separated list")
    ap.add_argument("--is-start", default=regimes.DEFAULT_IS_START,
                    help=f"in-sample window start "
                         f"(default {regimes.DEFAULT_IS_START})")
    ap.add_argument("--is-end", default=regimes.DEFAULT_IS_END,
                    help=f"in-sample window end "
                         f"(default {regimes.DEFAULT_IS_END})")
    ap.add_argument("--out-dir", default=None,
                    help=f"cache directory "
                         f"(default {regimes.cache_dir()}, "
                         f"or $BT_REGIME_CACHE)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild files that already exist")
    args = ap.parse_args(argv)

    symbols = resolve_symbols(args.symbols)
    tfs = resolve_timeframes(args.tf)
    root = Path(args.out_dir) if args.out_dir else None

    if pd.Timestamp(args.is_end) <= pd.Timestamp(args.is_start):
        raise SystemExit(f"in-sample window ends at or before it starts: "
                         f"{args.is_start} .. {args.is_end}")

    print(f"Regime pre-computation")
    print(f"  symbols     : {', '.join(symbols)}")
    print(f"  timeframes  : {', '.join(tfs)}")
    print(f"  in-sample   : {args.is_start} .. {args.is_end}")
    print(f"  cache dir   : {regimes.cache_path('X', 'Y', root).parent}")
    print(f"  ADX({regimes.ADX_LENGTH}) > {regimes.ADX_TREND_THRESHOLD} "
          f"= trending;  ATR({regimes.ATR_LENGTH}) > theta_vol = high vol\n")

    rows: list[dict] = []
    failed = 0
    for sym in symbols:
        for tf in tfs:
            print(f"  [{sym} {tf}] building ...", flush=True)
            try:
                r = build_one(sym, tf, args.is_start, args.is_end, root,
                              args.force)
            except Exception as e:                       # noqa: BLE001
                r = {"symbol": sym, "tf": tf, "status": "ERROR",
                     "note": f"{type(e).__name__}: {e}"}
            if r["status"] == "ERROR":
                failed += 1
                print(f"  [{sym} {tf}] ERROR  {r['note']}", flush=True)
            elif r["status"] == "SKIPPED":
                print(f"  [{sym} {tf}] skipped ({r['note']})", flush=True)
            else:
                print(f"  [{sym} {tf}] {r['rows']:,} bars  "
                      f"theta_vol={r['theta_vol']:.4f}  -> {r['path']}",
                      flush=True)
            rows.append(r)

    print_summary(rows)
    if failed:
        print(f"\n{failed} configuration(s) FAILED.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
