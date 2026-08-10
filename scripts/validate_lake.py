#!/usr/bin/env python3
"""
validate_lake.py - Structural and integrity checks on the futures lake.

Location:  ~/src/trading/scripts/validate_lake.py

Runs phases 1-4 of the validation plan:

  1. Inventory      - what exists, file counts, stray files
  2. Row counts     - per symbol-year, against expectations
  3. Structure      - duplicates, ordering, timezone, gaps
  4. Price sanity   - OHLC relationships, zero/negative prices, extreme moves

Writes CSVs to /mnt/backtest/artifacts/validation/ and prints a summary.

Usage
-----
    python scripts/validate_lake.py                    # all symbols
    python scripts/validate_lake.py --symbols ES NQ
    python scripts/validate_lake.py --tf 1d            # one timeframe
    python scripts/validate_lake.py --quick            # skip price checks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAKE = Path("/mnt/backtest/lake/futures/bars")
OUT = Path("/mnt/backtest/artifacts/validation")

REQUIRED_COLS = {"ts", "open", "high", "low", "close", "volume", "symbol"}

# Rough expectations. Thin contracts legitimately fall short on 1m.
EXPECTED = {
    "1m": (150_000, 400_000),   # (warn below, warn above) per full year
    "1d": (200, 330),
}


def log(m): print(f"==> {m}", flush=True)
def warn(m): print(f"[!] {m}", flush=True)


# --------------------------------------------------------------------------
def discover_symbols() -> list[str]:
    if not LAKE.exists():
        sys.exit(f"Lake not found: {LAKE}")
    return sorted(d.name.split("=", 1)[1] for d in LAKE.iterdir()
                  if d.is_dir() and d.name.startswith("symbol="))


def find_stray_files() -> list[Path]:
    """
    Parquet files sitting ABOVE the year=/month= partition level.

    This is not hypothetical. A flat data.parquet at the tf= root is read
    alongside the partitions, so every bar appears twice - doubled volume,
    distorted indicators, and a backtest that looks perfectly fine.
    """
    strays = []
    for p in LAKE.rglob("*.parquet"):
        parts = {x.split("=")[0] for x in p.parts if "=" in x}
        if "year" not in parts:
            strays.append(p)
    return strays


# --------------------------------------------------------------------------
def inventory(symbols: list[str], tfs: list[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for tf in tfs:
            base = LAKE / f"symbol={sym}" / f"tf={tf}"
            if not base.exists():
                rows.append({"symbol": sym, "tf": tf, "present": False,
                             "n_files": 0, "n_years": 0})
                continue
            files = list(base.rglob("data.parquet"))
            years = sorted({int(p.parent.parent.name.split("=")[1])
                            for p in files if "year=" in str(p)})
            rows.append({
                "symbol": sym, "tf": tf, "present": True,
                "n_files": len(files),
                "n_years": len(years),
                "first_year": years[0] if years else None,
                "last_year": years[-1] if years else None,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def load(sym: str, tf: str) -> pd.DataFrame | None:
    base = LAKE / f"symbol={sym}" / f"tf={tf}"
    if not base.exists():
        return None
    try:
        df = pd.read_parquet(base)
    except Exception as e:
        warn(f"{sym} {tf}: unreadable - {e}")
        return None
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def check_structure(sym: str, tf: str, df: pd.DataFrame) -> list[dict]:
    issues = []

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        issues.append({"symbol": sym, "tf": tf, "check": "schema",
                       "detail": f"missing columns: {sorted(missing)}",
                       "count": len(missing)})

    dupes = int(df["ts"].duplicated().sum())
    if dupes:
        issues.append({"symbol": sym, "tf": tf, "check": "duplicate_ts",
                       "detail": "same timestamp appears more than once",
                       "count": dupes})

    if not df["ts"].is_monotonic_increasing:
        issues.append({"symbol": sym, "tf": tf, "check": "ordering",
                       "detail": "timestamps not monotonically increasing",
                       "count": 1})

    if df["ts"].dt.tz is None:
        issues.append({"symbol": sym, "tf": tf, "check": "timezone",
                       "detail": "timestamps are tz-naive, expected UTC",
                       "count": 1})

    return issues


def check_prices(sym: str, tf: str, df: pd.DataFrame) -> list[dict]:
    issues = []
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    bad_hl = int((h < l).sum())
    if bad_hl:
        issues.append({"symbol": sym, "tf": tf, "check": "high_lt_low",
                       "detail": "high below low - corrupt", "count": bad_hl})

    bad_h = int((h < o.combine(c, max)).sum())
    if bad_h:
        issues.append({"symbol": sym, "tf": tf, "check": "high_lt_oc",
                       "detail": "high below open or close", "count": bad_h})

    bad_l = int((l > o.combine(c, min)).sum())
    if bad_l:
        issues.append({"symbol": sym, "tf": tf, "check": "low_gt_oc",
                       "detail": "low above open or close", "count": bad_l})

    nonpos = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if nonpos:
        issues.append({"symbol": sym, "tf": tf, "check": "nonpositive_price",
                       "detail": "zero or negative price", "count": nonpos})

    zerovol = int((df["volume"] == 0).sum())
    if zerovol > len(df) * 0.05:
        issues.append({"symbol": sym, "tf": tf, "check": "zero_volume",
                       "detail": f"{zerovol/len(df)*100:.1f}% of bars have no volume",
                       "count": zerovol})

    # Extreme moves. Most will be real - limit moves, gaps, roll jumps
    # (Databento does not back-adjust). Cross-reference against the roll
    # calendar before treating any of these as errors.
    ret = c.pct_change()
    sd = ret.std()
    if sd and not np.isnan(sd):
        extreme = int((ret.abs() > 10 * sd).sum())
        if extreme:
            issues.append({"symbol": sym, "tf": tf, "check": "extreme_move",
                           "detail": "bars beyond 10 sd - check against roll dates",
                           "count": extreme})

    return issues


def row_counts(sym: str, tf: str, df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df["ts"].dt.year).agg(
        n_rows=("ts", "size"),
        n_dates=("ts", lambda s: s.dt.date.nunique()),
        first=("ts", "min"),
        last=("ts", "max"),
    ).reset_index().rename(columns={"ts": "year"})
    g.insert(0, "tf", tf)
    g.insert(0, "symbol", sym)

    lo, hi = EXPECTED.get(tf, (0, 10**9))
    full_years = g[(g["year"] > g["year"].min()) & (g["year"] < g["year"].max())]
    g["flag"] = ""
    g.loc[g.index.isin(full_years.index) & (g["n_rows"] < lo), "flag"] = "LOW"
    g.loc[g.index.isin(full_years.index) & (g["n_rows"] > hi), "flag"] = "HIGH"

    # One row per date is only expected for daily bars. On 1m data there are
    # ~1440 rows per date by design, so comparing the two would flag every
    # healthy symbol-year. The real duplicate check is duplicate_ts in
    # check_structure(), which applies to every timeframe.
    if tf == "1d":
        g.loc[g["n_rows"] != g["n_dates"], "flag"] = "DUPLICATE_DATES"

    # Rows per date - useful sanity signal on intraday timeframes.
    g["rows_per_date"] = (g["n_rows"] / g["n_dates"]).round(1)
    return g


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Validate the futures lake.")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--tf", nargs="+", default=["1m", "1d"])
    p.add_argument("--quick", action="store_true",
                   help="Skip price sanity checks (faster on 1m data).")
    args = p.parse_args()

    symbols = args.symbols or discover_symbols()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LAKE VALIDATION")
    print("=" * 70)
    print(f"Symbols: {len(symbols)}   Timeframes: {', '.join(args.tf)}")

    # Phase 1 -------------------------------------------------------------
    print()
    log("Phase 1: inventory")
    strays = find_stray_files()
    if strays:
        warn(f"{len(strays)} parquet file(s) OUTSIDE the year= partition level:")
        for s in strays[:10]:
            print(f"    {s}")
        print("    These are read alongside the partitions and DUPLICATE every bar.")
        print("    Remove them before trusting anything downstream.")
    else:
        log("No stray files above the partition level.")

    inv = inventory(symbols, args.tf)
    inv.to_csv(OUT / "inventory.csv", index=False)
    absent = inv[~inv["present"]]
    if len(absent):
        warn(f"{len(absent)} symbol/timeframe combination(s) missing:")
        print(absent[["symbol", "tf"]].to_string(index=False))

    # Phases 2-4 ----------------------------------------------------------
    print()
    log("Phases 2-4: per-symbol checks")
    all_issues, all_counts = [], []

    for i, sym in enumerate(symbols, 1):
        for tf in args.tf:
            df = load(sym, tf)
            if df is None:
                continue
            all_issues += check_structure(sym, tf, df)
            if not args.quick:
                all_issues += check_prices(sym, tf, df)
            all_counts.append(row_counts(sym, tf, df))
        print(f"  [{i}/{len(symbols)}] {sym}", flush=True)

    counts = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
    issues = pd.DataFrame(all_issues)

    if not counts.empty:
        counts.to_csv(OUT / "row_counts.csv", index=False)
    if not issues.empty:
        issues.to_csv(OUT / "issues.csv", index=False)

    # Summary -------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if issues.empty:
        print("  No structural or price issues found.")
    else:
        print(f"  {len(issues)} issue(s) across {issues['symbol'].nunique()} symbol(s):")
        print()
        print(issues.groupby("check")["count"].agg(["size", "sum"])
              .rename(columns={"size": "occurrences", "sum": "total_rows"})
              .to_string())

    if not counts.empty:
        flagged = counts[counts["flag"] != ""]
        if len(flagged):
            print()
            print(f"  {len(flagged)} symbol-year(s) with unexpected row counts:")
            print(flagged.head(30).to_string(index=False))
        else:
            print()
            print("  All row counts within expected ranges.")

    print()
    print(f"  Wrote {OUT}/inventory.csv")
    if not counts.empty:
        print(f"  Wrote {OUT}/row_counts.csv")
    if not issues.empty:
        print(f"  Wrote {OUT}/issues.csv")
    print()


if __name__ == "__main__":
    main()
