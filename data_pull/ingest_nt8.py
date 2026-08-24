#!/usr/bin/env python3
"""
ingest_nt8.py - normalise NinjaTrader 8 CSV exports into the lake.

Location:  ~/src/trading/data_pull/ingest_nt8.py

Reads the CSVs written by the HistoricalExporter add-on and writes Parquet in
the same shape as the Databento lake, but under a **separate root** so the two
sources can never be mixed in a single query.

    /mnt/backtest/raw/futures_nt8/ES_1d.csv          <- written by NT8
    /mnt/backtest/lake/futures_nt8/bars/symbol=ES/tf=1d/year=2022/month=01/

Why separate
------------
This data exists to answer one question: does a strategy behave the same when
only the data feed changes? A divergence means the strategy was fitting one
vendor's quirks rather than market structure.

That comparison only means something if the two sources stay distinct. Mixing
them in one tree would eventually produce a backtest half on each, and no way
to tell.

**This is not out-of-sample validation.** NT8 builds its continuous contracts
with its own roll rules, different from Databento's volume roll, so some
divergence is expected and is about contract construction rather than the
strategy. Real out-of-sample comes from holding back the last three years of
Databento data.

Timezone
--------
NT8 writes bar times in the platform's local timezone, with no offset in the
file. This script converts to UTC using --tz, which must match the NT8
machine's timezone. Getting this wrong shifts every bar and quietly ruins the
comparison, so the resulting date range is printed for you to sanity-check.

Usage
-----
    python data_pull/ingest_nt8.py                    # all CSVs found
    python data_pull/ingest_nt8.py --symbols ES NQ
    python data_pull/ingest_nt8.py --tz America/New_York
    python data_pull/ingest_nt8.py --compare ES       # vs Databento
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

import numpy as np
import pandas as pd

RAW = Path("/mnt/backtest/raw/futures_nt8")
LAKE = Path("/mnt/backtest/lake/futures_nt8/bars")
DATABENTO_LAKE = Path("/mnt/backtest/lake/futures/bars")
SCRATCH = Path.home() / "scratch"

DEFAULT_TZ = "America/New_York"


def log(m): print(f"==> {m}", flush=True)
def warn(m): print(f"[!] {m}", flush=True)


# --------------------------------------------------------------------------
def find_exports(raw_dir: Path, symbols: list[str] | None) -> list[Path]:
    if not raw_dir.exists():
        sys.exit(f"{raw_dir} not found. Run the HistoricalExporter add-on in NT8 first.")
    files = sorted(raw_dir.glob("*.csv"))
    if symbols:
        wanted = {s.upper() for s in symbols}
        files = [f for f in files if f.stem.split("_")[0].upper() in wanted]
    return files


def parse_export(path: Path, tz: str) -> tuple[str, str, pd.DataFrame]:
    """
    Returns (symbol, timeframe, dataframe).

    Filename convention from the add-on: SYMBOL_TF.csv, e.g. ES_1d.csv
    """
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"{path.name}: expected SYMBOL_TF.csv")
    symbol, tf = stem.rsplit("_", 1)
    symbol = symbol.upper()

    df = pd.read_csv(path)

    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    # NT8 writes local time with no offset. Localize then convert.
    ts = pd.to_datetime(df["ts"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    df["ts"] = ts.dt.tz_convert("UTC")

    before = len(df)
    df = df.dropna(subset=["ts"])
    if len(df) < before:
        # Ambiguous or nonexistent local times at DST transitions.
        warn(f"{symbol} {tf}: dropped {before - len(df)} row(s) at DST boundaries")

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    df = (df.sort_values("ts")
            .drop_duplicates(subset=["ts"], keep="last")
            .reset_index(drop=True))
    df["symbol"] = symbol

    return symbol, tf, df[["ts", "open", "high", "low", "close", "volume", "symbol"]]


def sanity_check(symbol: str, tf: str, df: pd.DataFrame) -> list[str]:
    """Same checks scripts/validate_lake.py applies to the main lake."""
    issues = []
    if df.empty:
        return [f"{symbol} {tf}: empty"]

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    n = int((h < l).sum())
    if n:
        issues.append(f"{symbol} {tf}: {n} bar(s) with high below low")

    n = int((h < o.combine(c, max)).sum())
    if n:
        issues.append(f"{symbol} {tf}: {n} bar(s) with high below open/close")

    n = int((l > o.combine(c, min)).sum())
    if n:
        issues.append(f"{symbol} {tf}: {n} bar(s) with low above open/close")

    n = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if n:
        issues.append(f"{symbol} {tf}: {n} bar(s) with non-positive price")

    if df["ts"].duplicated().any():
        issues.append(f"{symbol} {tf}: duplicate timestamps")

    return issues


def write_lake(df: pd.DataFrame, symbol: str, tf: str) -> int:
    """
    Partitioned exactly like the main lake: symbol=/tf=/year=/month=.

    Never write above the year= level - a flat file at the tf= root is read
    alongside the partitions and duplicates every bar. That has already
    happened once in this project.
    """
    written = 0
    d = df.copy()
    d["_y"] = d["ts"].dt.year
    d["_m"] = d["ts"].dt.month

    for (y, m), chunk in d.groupby(["_y", "_m"], sort=True):
        dest_dir = LAKE / f"symbol={symbol}" / f"tf={tf}" / f"year={y}" / f"month={m:02d}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Write straight to the destination. These files are small (a few
        # hundred daily bars), so the staging dance the Databento puller uses
        # for large minute files is not worth the cross-device copy - local
        # scratch and the NFS lake are different filesystems, so shutil.move
        # falls back to a full copy anyway.
        chunk.drop(columns=["_y", "_m"]).to_parquet(
            dest_dir / "data.parquet", engine="pyarrow",
            compression="zstd", index=False)
        written += len(chunk)

    return written


# --------------------------------------------------------------------------
def compare_sources(symbol: str, tf: str = "1d") -> None:
    """
    Overlap comparison against the Databento lake.

    Divergence is informative, not automatically a fault:
      - Bar count differences usually mean different session definitions.
      - Price differences at roll dates mean different roll rules. NT8 and
        Databento build continuous contracts differently, so this is expected.
      - Price differences AWAY from rolls are the interesting case, and worth
        investigating before trusting either source.
    """
    a_dir = LAKE / f"symbol={symbol}" / f"tf={tf}"
    b_dir = DATABENTO_LAKE / f"symbol={symbol}" / f"tf={tf}"

    if not a_dir.exists():
        warn(f"No NT8 data for {symbol} {tf}")
        return
    if not b_dir.exists():
        warn(f"No Databento data for {symbol} {tf}")
        return

    a = pd.read_parquet(a_dir).sort_values("ts")
    b = pd.read_parquet(b_dir).sort_values("ts")

    a["d"] = a["ts"].dt.date
    b["d"] = b["ts"].dt.date

    a1 = a.groupby("d").agg(nt8_close=("close", "last"), nt8_vol=("volume", "sum"))
    b1 = b.groupby("d").agg(db_close=("close", "last"), db_vol=("volume", "sum"))
    j = a1.join(b1, how="inner")

    if j.empty:
        warn(f"{symbol}: no overlapping dates")
        return

    j["close_diff_pct"] = (j["nt8_close"] / j["db_close"] - 1) * 100
    j["vol_ratio"] = j["nt8_vol"] / j["db_vol"].replace(0, np.nan)

    print()
    print(f"--- {symbol} {tf}: NT8 vs Databento ---")
    print(f"  NT8 range       {a['d'].min()} to {a['d'].max()}  ({len(a1):,} days)")
    print(f"  Databento range {b['d'].min()} to {b['d'].max()}  ({len(b1):,} days)")
    print(f"  Overlap         {j.index.min()} to {j.index.max()}  ({len(j):,} days)")
    print()
    print(f"  Close differences (%):")
    print(f"    median {j['close_diff_pct'].median():>8.4f}")
    print(f"    mean   {j['close_diff_pct'].mean():>8.4f}")
    print(f"    max abs{j['close_diff_pct'].abs().max():>8.4f}")
    print(f"    >0.5% on {int((j['close_diff_pct'].abs() > 0.5).sum()):,} of {len(j):,} days")
    print(f"  Volume ratio median {j['vol_ratio'].median():.3f}")

    worst = j.reindex(j["close_diff_pct"].abs().sort_values(ascending=False).index).head(10)
    if not worst.empty and worst["close_diff_pct"].abs().iloc[0] > 0.1:
        print()
        print("  Largest divergences - check these against roll dates:")
        print(worst[["nt8_close", "db_close", "close_diff_pct"]].round(4).to_string())


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Ingest NT8 CSV exports into the lake.")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--tz", default=DEFAULT_TZ,
                   help=f"Timezone of the NT8 machine. Default {DEFAULT_TZ}. "
                        "Wrong value shifts every bar.")
    p.add_argument("--raw-dir", default=str(RAW))
    p.add_argument("--compare", nargs="*", default=None,
                   help="After ingest, compare these symbols against Databento. "
                        "No arguments compares everything ingested.")
    args = p.parse_args()

    files = find_exports(Path(args.raw_dir), args.symbols)
    if not files:
        sys.exit(f"No CSVs found in {args.raw_dir}")

    log(f"Found {len(files)} export(s). Timezone: {args.tz}")

    all_issues, done, total_rows = [], [], 0

    for f in files:
        try:
            symbol, tf, df = parse_export(f, args.tz)
        except Exception as e:
            warn(f"{f.name}: {e}")
            continue

        if df.empty:
            warn(f"{f.name}: no usable rows")
            continue

        issues = sanity_check(symbol, tf, df)
        all_issues += issues

        n = write_lake(df, symbol, tf)
        total_rows += n
        done.append(symbol)

        log(f"{symbol} {tf}: {n:,} rows  "
            f"{df['ts'].min().date()} to {df['ts'].max().date()}")

    print()
    print("=" * 64)
    print(f"  Ingested {total_rows:,} rows across {len(done)} symbol(s)")
    print(f"  -> {LAKE}")

    if all_issues:
        print()
        warn(f"{len(all_issues)} issue(s):")
        for i in all_issues[:20]:
            print(f"    {i}")
    else:
        print("  No structural issues.")

    print()
    print("  Sanity-check the date ranges above. If they look shifted by hours,")
    print(f"  --tz is wrong (currently {args.tz}).")

    if args.compare is not None:
        targets = args.compare if args.compare else done
        for s in targets:
            compare_sources(s)

    print()


if __name__ == "__main__":
    main()
