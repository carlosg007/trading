#!/usr/bin/env python3
"""
fetch_degraded_days.py - Pull the authoritative data-quality calendar.

Location:  ~/src/trading/data_pull/fetch_degraded_days.py

Why this exists
---------------
Databento flags sessions whose capture had problems - dropped packets,
gateway issues. The download log surfaces these as warnings, but the warnings
truncate after three dates ("..."), so scraping the log undercounts badly.

This calls the dataset condition endpoint instead, which reports the
condition of every day in the dataset. It is a metadata call, so it costs
nothing.

Why it matters more than it looks
---------------------------------
The degraded days cluster on high-volatility sessions - 2020-02-27 and
2020-02-28 (COVID crash), 2024-09-18 (FOMC 50bp cut), quarter-end rolls.
Exchange feeds get stressed exactly when it matters most.

The practical risk: a strategy looks robust through a crisis because the
worst bars are missing. Every stress test must check this file first.

Usage
-----
    export DATABENTO_API_KEY="db-..."
    python data_pull/fetch_degraded_days.py
    python data_pull/fetch_degraded_days.py --start 2010-06-06 --end 2026-08-10
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

try:
    import databento as db
except ImportError:
    sys.exit("databento not installed. Activate the trading venv first.")

DATASET = "GLBX.MDP3"
DATASET_START = "2010-06-06"
OUT_DIR = Path("/mnt/backtest/reference/futures")


def log(m): print(f"==> {m}", flush=True)
def warn(m): print(f"[!] {m}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Databento dataset condition calendar.")
    p.add_argument("--start", default=DATASET_START)
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY is not set.")

    client = db.Historical(key)

    log(f"Fetching condition for {args.dataset}: {args.start} -> {args.end}")
    try:
        cond = client.metadata.get_dataset_condition(
            dataset=args.dataset,
            start_date=args.start,
            end_date=args.end,
        )
    except Exception as e:
        sys.exit(f"Request failed: {e}")

    df = pd.DataFrame(cond)
    if df.empty:
        sys.exit("Empty response - nothing to write.")

    # Normalise column names across databento versions
    cols = {c.lower(): c for c in df.columns}
    date_col = next((cols[c] for c in ("date",) if c in cols), df.columns[0])
    cond_col = next((cols[c] for c in ("condition",) if c in cols), None)

    df = df.rename(columns={date_col: "date"})
    if cond_col and cond_col != "condition":
        df = df.rename(columns={cond_col: "condition"})

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full calendar - every day and its condition
    full = out_dir / "dataset_condition.csv"
    df.to_csv(full, index=False)

    # Just the problem days - this is what backtests check against
    if "condition" in df.columns:
        bad = df[df["condition"].astype(str).str.lower() != "available"]
    else:
        warn("No 'condition' column found; writing the full calendar only.")
        bad = pd.DataFrame()

    degraded = out_dir / "degraded_days.csv"
    bad.to_csv(degraded, index=False)

    # Summary -------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  Days in range        {len(df):>8,}")
    print(f"  Not fully available  {len(bad):>8,}")
    if "condition" in df.columns:
        print()
        print(df["condition"].value_counts().to_string())

    if len(bad):
        bad = bad.copy()
        bad["year"] = pd.to_datetime(bad["date"]).dt.year
        print()
        print("  By year:")
        print(bad["year"].value_counts().sort_index().to_string())

        print()
        print("  Most recent 15:")
        print(bad.tail(15).to_string(index=False))

    print()
    print(f"  Wrote {full}")
    print(f"  Wrote {degraded}")
    print()


if __name__ == "__main__":
    main()
