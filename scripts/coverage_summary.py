#!/usr/bin/env python3
"""
coverage_summary.py - Per-symbol coverage reference for the futures lake.

Location:  ~/src/trading/scripts/coverage_summary.py

Produces the table every later decision refers back to: which symbols are
usable, over what period, and from when the intraday data is dense enough to
trust.

The key finding this quantifies
-------------------------------
ES 1-minute coverage climbs from ~337 bars/day in 2010 to ~1,130 from 2016
onward, where it plateaus. Volume ties out exactly against the native daily
bars in every year, so nothing is missing - the early years simply record the
same trading in fewer distinct minutes.

Consequence: daily and swing work can use the full history. Intraday work
should start from each symbol's plateau year, because a 30m or 1h bar built
from sparse minutes behaves differently than one built from dense minutes.

Outputs
-------
    /mnt/backtest/reference/futures/coverage.csv        one row per symbol
    /mnt/backtest/reference/futures/coverage_yearly.csv one row per symbol-year

Usage
-----
    python scripts/coverage_summary.py
    python scripts/coverage_summary.py --symbols ES NQ CL
    python scripts/coverage_summary.py --plateau-tolerance 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAKE = Path("/mnt/backtest/lake/futures/bars")
REF = Path("/mnt/backtest/reference/futures")


def log(m): print(f"==> {m}", flush=True)
def warn(m): print(f"[!] {m}", flush=True)


def discover_symbols() -> list[str]:
    if not LAKE.exists():
        sys.exit(f"Lake not found: {LAKE}")
    return sorted(d.name.split("=", 1)[1] for d in LAKE.iterdir()
                  if d.is_dir() and d.name.startswith("symbol="))


def load_year(sym: str, tf: str, year: int) -> pd.DataFrame | None:
    p = LAKE / f"symbol={sym}" / f"tf={tf}" / f"year={year}"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        warn(f"{sym} {tf} {year}: unreadable - {e}")
        return None
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def years_present(sym: str, tf: str) -> list[int]:
    base = LAKE / f"symbol={sym}" / f"tf={tf}"
    if not base.exists():
        return []
    return sorted(int(d.name.split("=")[1]) for d in base.iterdir()
                  if d.is_dir() and d.name.startswith("year="))


# --------------------------------------------------------------------------
def yearly_profile(sym: str) -> pd.DataFrame:
    """One row per year: density, volume, date range."""
    rows = []
    for year in years_present(sym, "1m"):
        m = load_year(sym, "1m", year)
        if m is None:
            continue
        n_dates = m["ts"].dt.date.nunique()
        d = load_year(sym, "1d", year)

        rows.append({
            "symbol": sym,
            "year": year,
            "rows_1m": len(m),
            "sessions": n_dates,
            "bars_per_day": round(len(m) / n_dates, 1) if n_dates else np.nan,
            "rows_1d": len(d) if d is not None else 0,
            "volume_1m": int(m["volume"].sum()),
            "volume_1d": int(d["volume"].sum()) if d is not None else 0,
            "median_daily_volume": int(d["volume"].median()) if d is not None else 0,
            "first": m["ts"].min().date(),
            "last": m["ts"].max().date(),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Volume reconciliation: does 1m sum match the native daily total?
    # A mismatch means one of the two is incomplete.
    df["volume_ratio"] = np.where(
        df["volume_1d"] > 0,
        (df["volume_1m"] / df["volume_1d"]).round(4),
        np.nan,
    )
    return df


def find_plateau(profile: pd.DataFrame, tolerance: float = 0.90) -> int | None:
    """
    First year from which intraday density stays above `tolerance` of the
    symbol's typical mature level.

    "Mature level" is the median bars/day across the most recent five years,
    which avoids letting one unusual year set the bar.
    """
    if profile.empty or len(profile) < 3:
        return None

    recent = profile.tail(5)
    mature = recent["bars_per_day"].median()
    if not mature or np.isnan(mature):
        return None

    threshold = mature * tolerance
    qualifying = profile[profile["bars_per_day"] >= threshold]
    if qualifying.empty:
        return None

    # Walk back from the end to find the start of the unbroken run.
    years = profile["year"].tolist()
    ok = set(qualifying["year"])
    start = years[-1]
    for y in reversed(years):
        if y in ok:
            start = y
        else:
            break
    return int(start)


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Per-symbol coverage summary.")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--plateau-tolerance", type=float, default=0.90,
                   help="Fraction of mature density a year must reach to count "
                        "as dense. Default 0.90.")
    p.add_argument("--out-dir", default=str(REF))
    args = p.parse_args()

    symbols = args.symbols or discover_symbols()
    log(f"Profiling {len(symbols)} symbol(s)")

    all_years, summary = [], []

    for i, sym in enumerate(symbols, 1):
        prof = yearly_profile(sym)
        if prof.empty:
            warn(f"{sym}: no 1m data")
            continue
        all_years.append(prof)

        plateau = find_plateau(prof, args.plateau_tolerance)
        mature = prof.tail(5)["bars_per_day"].median()

        # Any year where 1m and 1d volume disagree
        bad_ratio = prof[(prof["volume_ratio"].notna()) &
                         (prof["volume_ratio"].sub(1).abs() > 0.001)]

        summary.append({
            "symbol": sym,
            "first_date": prof["first"].min(),
            "last_date": prof["last"].max(),
            "n_years": len(prof),
            "rows_1m": int(prof["rows_1m"].sum()),
            "rows_1d": int(prof["rows_1d"].sum()),
            "sessions": int(prof["sessions"].sum()),
            "mature_bars_per_day": round(mature, 1) if mature else np.nan,
            "intraday_start_year": plateau,
            "median_daily_volume": int(prof.tail(5)["median_daily_volume"].median()),
            "volume_mismatch_years": len(bad_ratio),
        })
        print(f"  [{i}/{len(symbols)}] {sym}", flush=True)

    if not summary:
        sys.exit("Nothing profiled.")

    cov = pd.DataFrame(summary).sort_values("median_daily_volume", ascending=False)
    yearly = pd.concat(all_years, ignore_index=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out_dir / "coverage.csv", index=False)
    yearly.to_csv(out_dir / "coverage_yearly.csv", index=False)

    # Report --------------------------------------------------------------
    print()
    print("=" * 100)
    print("COVERAGE SUMMARY  (sorted by liquidity)")
    print("=" * 100)
    show = cov[["symbol", "first_date", "last_date", "n_years",
                "mature_bars_per_day", "intraday_start_year",
                "median_daily_volume", "volume_mismatch_years"]]
    print(show.to_string(index=False))

    print()
    print("-" * 100)
    print("NOTES")
    print("-" * 100)

    mismatch = cov[cov["volume_mismatch_years"] > 0]
    if len(mismatch):
        print(f"  {len(mismatch)} symbol(s) where 1m and 1d volume disagree - investigate:")
        print(mismatch[["symbol", "volume_mismatch_years"]].to_string(index=False))
    else:
        print("  1m volume reconciles exactly against native daily in every")
        print("  symbol-year. Nothing is missing - early years simply record")
        print("  the same trading in fewer distinct minutes.")

    starts = cov["intraday_start_year"].dropna()
    if len(starts):
        print()
        print(f"  Intraday start years range {int(starts.min())}-{int(starts.max())}, "
              f"median {int(starts.median())}.")
        print("  Use these for 30m/1h strategies. Daily and swing work can use")
        print("  the full history.")
        vc = starts.value_counts().sort_index()
        print()
        for yr, n in vc.items():
            syms = cov[cov["intraday_start_year"] == yr]["symbol"].tolist()
            print(f"    {int(yr)}: {n:>2} symbols  {' '.join(syms)}")

    thin = cov[cov["mature_bars_per_day"] < 600]
    if len(thin):
        print()
        print("  Short-session contracts (under 600 bars/day at maturity).")
        print("  This is normal - they trade limited hours, not sparse data:")
        print(thin[["symbol", "mature_bars_per_day"]].to_string(index=False))

    print()
    print(f"  Wrote {out_dir/'coverage.csv'}")
    print(f"  Wrote {out_dir/'coverage_yearly.csv'}")
    print()


if __name__ == "__main__":
    main()
