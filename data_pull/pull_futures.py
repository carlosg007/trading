#!/usr/bin/env python3
"""
pull_futures.py - Download CME futures 1-minute bars from Databento into the lake.

Location:  ~/src/trading/data_pull/pull_futures.py

What it does
------------
  1. Resolves continuous symbols (e.g. ES.v.0) and saves the roll calendar
     to /mnt/backtest/reference/futures/roll_calendar_<SYM>.json
  2. Prints the cost of the download BEFORE downloading anything.
  3. Only downloads if you pass --confirm.
  4. Pulls year by year, so a failure mid-run doesn't lose completed years.
  5. Writes raw DBN to /mnt/backtest/raw/futures/ (immutable archive)
     and Parquet to /mnt/backtest/lake/futures/bars/symbol=X/tf=1m/year=Y/month=M/

Notes
-----
  * Continuous contracts use the VOLUME roll rule (.v.0) - follows the
    contract actually trading, not just the nearest expiry.
  * Databento does NOT back-adjust. Prices are raw. There will be a gap at
    each roll. The roll calendar is saved so backtests can exclude those days.
  * Timestamps from Databento are UTC. Stored as UTC. Convert at query time.

Setup
-----
    export DATABENTO_API_KEY="db-xxxxxxxxxxxxxxxxxxxx"
    uv run --with databento --with pandas --with pyarrow python pull_futures.py --help

Usage
-----
    # See what it would cost - downloads nothing
    python pull_futures.py --symbols ES NQ --start 2016-01-01 --end 2026-01-01

    # Actually download
    python pull_futures.py --symbols ES NQ --start 2016-01-01 --end 2026-01-01 --confirm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

try:
    import databento as db
except ImportError:
    sys.exit("databento not installed. Run with:\n"
             "  uv run --with databento --with pandas --with pyarrow python pull_futures.py ...")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
LAKE = Path("/mnt/backtest/lake/futures/bars")
RAW = Path("/mnt/backtest/raw/futures")
REFERENCE = Path("/mnt/backtest/reference/futures")
SCRATCH = Path.home() / "scratch"

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
ROLL_RULE = "v"          # v = volume, n = open interest, c = calendar
RANK = 0                 # 0 = front month

# Earliest date GLBX.MDP3 has any data at all.
DATASET_START = "2010-06-06"

# Which lake tf= folder each bar schema lands in.
SCHEMA_TO_TF = {
    "ohlcv-1m": "1m",
    "ohlcv-1h": "1h",
    "ohlcv-1d": "1d",
}

# Non-bar schemas. These have completely different shapes to OHLCV, so they
# get their own destinations and are stored as-received rather than forced
# into the canonical bar schema.
#
#   statistics  open interest, settlement prices, session stats.
#               Open interest matters at swing horizons - it separates new
#               positioning from position closing. L0, small.
#   definition  contract specs: tick size, multiplier, expiry, contract month.
#               Needed for correct position sizing and cost modelling across
#               a multi-symbol universe. L0, small.
#   bbo-1m      bid/ask sampled once per minute. Gives spread as a per-minute
#               liquidity feature. THIS IS L1 - the Standard plan includes
#               only 1 year, beyond which it is billed per byte and is not
#               cheap. Always check the cost preview before pulling.
NON_BAR_SCHEMAS = {
    "statistics": ("lake/futures/statistics", "yearly"),
    "definition": ("reference/futures/definitions", "flat"),
    "bbo-1m": ("lake/futures/bbo", "monthly"),
}

L1_SCHEMAS = {"bbo-1m"}

ALL_SCHEMAS = list(SCHEMA_TO_TF) + list(NON_BAR_SCHEMAS)


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[!] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
def get_client() -> "db.Historical":
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY is not set.\n"
                 "  export DATABENTO_API_KEY=\"db-xxxxxxxx\"\n"
                 "Add it to ~/.bashrc to make it permanent.")
    return db.Historical(key)


def continuous_symbol(sym: str) -> str:
    """ES -> ES.v.0"""
    return f"{sym}.{ROLL_RULE}.{RANK}"


# --------------------------------------------------------------------------
def save_roll_calendar(client, sym: str, start: str, end: str) -> None:
    """
    Record which physical contract the continuous symbol pointed at, and when.
    Lets a backtest exclude roll days instead of trading the price gap.
    """
    csym = continuous_symbol(sym)
    log(f"Resolving roll calendar for {csym}")
    try:
        res = client.symbology.resolve(
            dataset=DATASET,
            symbols=[csym],
            stype_in="continuous",
            stype_out="instrument_id",
            start_date=start,
            end_date=end,
        )
    except Exception as e:
        warn(f"Could not resolve roll calendar for {csym}: {e}")
        return

    REFERENCE.mkdir(parents=True, exist_ok=True)
    out = REFERENCE / f"roll_calendar_{sym}.json"
    out.write_text(json.dumps(res, indent=2, default=str))

    intervals = res.get("result", {}).get(csym, [])
    log(f"  {len(intervals)} contract intervals -> {out}")


# --------------------------------------------------------------------------
def probe_symbols(client, symbols: list[str], start: str, end: str) -> None:
    """
    Report the real available date range for each symbol, without downloading.

    Uses symbology.resolve, which is a metadata call - it tells you which
    physical contracts the continuous symbol maps to, and when. The first
    interval start is effectively the symbol's first available date.

    Symbols that don't exist on this dataset (e.g. ICE products like KC, SB,
    CT which are not on CME) will come back empty or error - that's the point.
    """
    print()
    print(f"{'SYMBOL':<8} {'FIRST':<12} {'LAST':<12} {'INTERVALS':>9}  STATUS")
    print("-" * 62)

    for sym in symbols:
        csym = continuous_symbol(sym)
        try:
            res = client.symbology.resolve(
                dataset=DATASET,
                symbols=[csym],
                stype_in="continuous",
                stype_out="instrument_id",
                start_date=start,
                end_date=end,
            )
        except Exception as e:
            msg = str(e).split("\n")[0][:28]
            print(f"{sym:<8} {'-':<12} {'-':<12} {'-':>9}  ERROR: {msg}")
            continue

        intervals = res.get("result", {}).get(csym, [])
        if not intervals:
            print(f"{sym:<8} {'-':<12} {'-':<12} {0:>9}  NOT ON {DATASET}")
            continue

        try:
            first = min(i["d0"] for i in intervals)
            last = max(i["d1"] for i in intervals)
        except (KeyError, TypeError):
            first, last = "?", "?"

        print(f"{sym:<8} {first:<12} {last:<12} {len(intervals):>9}  ok")

    print()
    print("Symbols showing NOT ON / ERROR need a different dataset or a")
    print("corrected root symbol. Remove them before downloading.")
    print()


# --------------------------------------------------------------------------
def preview_cost(client, symbols: list[str], start: str, end: str,
                 schemas: list[str] | None = None) -> float:
    """Price the whole request before spending anything."""
    schemas = schemas or [SCHEMA]
    csyms = [continuous_symbol(s) for s in symbols]
    log("Pricing request (no data downloaded yet)")

    total = 0.0
    per_schema = {}
    for schema in schemas:
        try:
            c = client.metadata.get_cost(
                dataset=DATASET,
                symbols=csyms,
                schema=schema,
                start=start,
                end=end,
                stype_in="continuous",
            )
        except Exception as e:
            warn(f"Cost preview failed for {schema}: {e}")
            c = float("nan")
        per_schema[schema] = c
        if c == c:  # not NaN
            total += c

    print()
    print(f"  Dataset : {DATASET}")
    print(f"  Symbols : {len(csyms)} -> {', '.join(csyms)}")
    print(f"  Range   : {start} -> {end}")
    for schema, c in per_schema.items():
        flag = "  <-- L1, billed beyond 1yr" if schema in L1_SCHEMAS else ""
        print(f"  {schema:<12}: ${c:,.2f}{flag}")
    print(f"  {'TOTAL':<12}: ${total:,.2f}")
    print()

    if any(s in L1_SCHEMAS for s in schemas) and total > 0:
        print("  ** This request includes L1 data. The Standard plan covers")
        print("     only 1 year of L1; the rest is billed per byte. Check the")
        print("     figure above against your credit balance before confirming.")
        print()

    return total


# --------------------------------------------------------------------------
def normalize(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """
    Databento ohlcv-1m -> canonical lake schema.

    Canonical columns: ts (UTC), symbol, open, high, low, close, volume
    """
    df = df.reset_index()

    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    df = df.rename(columns={ts_col: "ts"})

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    keep = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{sym}: expected columns missing from Databento response: {missing}\n"
                         f"got: {list(df.columns)}")

    out = df[keep].copy()
    out["symbol"] = sym

    # Sort and drop exact duplicate timestamps (defensive)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return out.reset_index(drop=True)


def write_parquet(df: pd.DataFrame, sym: str, tf: str = "1m") -> int:
    """
    Write to lake/futures/bars/symbol=X/tf=1m/year=YYYY/month=MM/data.parquet

    Staged locally then copied, so an interrupted NFS write can't leave a
    half-written Parquet file that still half-parses.
    """
    written = 0
    df = df.copy()
    df["_year"] = df["ts"].dt.year
    df["_month"] = df["ts"].dt.month

    for (yr, mo), chunk in df.groupby(["_year", "_month"], sort=True):
        dest_dir = LAKE / f"symbol={sym}" / f"tf={tf}" / f"year={yr}" / f"month={mo:02d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "data.parquet"

        SCRATCH.mkdir(parents=True, exist_ok=True)
        tmp = SCRATCH / f"{sym}_{tf}_{yr}{mo:02d}.parquet"

        chunk.drop(columns=["_year", "_month"]).to_parquet(
            tmp, engine="pyarrow", compression="zstd", index=False
        )
        # Path.replace() can't cross filesystems (local -> NFS), so use move()
        shutil.move(str(tmp), str(dest))

        written += len(chunk)

    return written


# --------------------------------------------------------------------------
def write_non_bar(df: pd.DataFrame, sym: str, schema: str, year: int) -> int:
    """
    Write a non-OHLCV schema as received, with only the timestamp normalised.

    These schemas have vendor-specific columns that would be lost by forcing
    them into the bar schema. Store faithfully now, interpret later.
    """
    base, style = NON_BAR_SCHEMAS[schema]
    root = Path("/mnt/backtest") / base

    out = df.reset_index()
    for c in ("ts_event", "ts_recv", "ts_ref"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    if "ts_event" in out.columns:
        out = out.rename(columns={"ts_event": "ts"}).sort_values("ts")
    out["symbol"] = sym

    if style == "flat":
        dest_dir = root / f"symbol={sym}"
        name = f"{year}.parquet"
    elif style == "yearly":
        dest_dir = root / f"symbol={sym}" / f"year={year}"
        name = "data.parquet"
    else:  # monthly
        dest_dir = root / f"symbol={sym}" / f"year={year}"
        name = "data.parquet"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name

    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / f"{sym}_{schema}_{year}.parquet"
    out.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    shutil.move(str(tmp), str(dest))
    return len(out)


def already_have_non_bar(sym: str, schema: str, year: int) -> bool:
    base, style = NON_BAR_SCHEMAS[schema]
    root = Path("/mnt/backtest") / base
    if style == "flat":
        return (root / f"symbol={sym}" / f"{year}.parquet").exists()
    return (root / f"symbol={sym}" / f"year={year}" / "data.parquet").exists()


def already_have(sym: str, year: int, tf: str) -> bool:
    """
    True if this symbol-year-timeframe is already in the lake.

    Lets a long run be restarted after a crash without re-downloading
    completed work. Pass --force to re-download anyway.
    """
    base = LAKE / f"symbol={sym}" / f"tf={tf}" / f"year={year}"
    if not base.exists():
        return False
    return any(base.rglob("data.parquet"))


def pull_symbol_year(client, sym: str, year: int, schema: str = SCHEMA,
                     force: bool = False) -> int:
    """Download one symbol-year for one schema. Returns rows written."""
    csym = continuous_symbol(sym)
    is_bar = schema in SCHEMA_TO_TF
    tf = SCHEMA_TO_TF.get(schema, schema)

    if not force:
        have = already_have(sym, year, tf) if is_bar \
            else already_have_non_bar(sym, schema, year)
        if have:
            log(f"{sym} {year} [{tf}]: already in lake, skipping")
            return 0

    # Clamp to the dataset's true start. Requesting 2010-01-01 when the
    # dataset begins 2010-06-06 returns a 422 and silently loses that year.
    start = max(f"{year}-01-01", DATASET_START)

    # Clamp the end too - requesting past the available end also 422s.
    end = min(f"{year + 1}-01-01", str(date.today()))
    if start >= end:
        return 0

    log(f"{sym} {year} [{tf}]: requesting {schema}")
    try:
        store = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[csym],
            schema=schema,
            start=start,
            end=end,
            stype_in="continuous",
        )
    except Exception as e:
        warn(f"{sym} {year} [{tf}]: request failed: {e}")
        return 0

    # Archive the raw response before touching it
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{sym}_{schema}_{year}.dbn.zst"
    try:
        store.to_file(raw_path)
    except Exception as e:
        warn(f"{sym} {year}: could not archive raw DBN: {e}")

    df = store.to_df()
    if df.empty:
        warn(f"{sym} {year} [{tf}]: no rows returned")
        return 0

    if is_bar:
        n = write_parquet(normalize(df, sym), sym, tf=tf)
    else:
        n = write_non_bar(df, sym, schema, year)

    log(f"{sym} {year} [{tf}]: {n:,} rows written")
    return n


# --------------------------------------------------------------------------
def build_daily(sym: str) -> None:
    """
    Derive daily bars from the stored 1m bars.

    Derived rather than pulled so the daily bars are guaranteed to tie out
    to the minute data a backtest actually trades on.

    Day boundary: UTC date. Change here if you want a session-aligned day.
    """
    src = LAKE / f"symbol={sym}" / "tf=1m"
    files = sorted(src.rglob("data.parquet"))
    if not files:
        warn(f"{sym}: no 1m data found, skipping daily")
        return

    log(f"{sym}: building daily bars from {len(files)} monthly files")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True).sort_values("ts")

    daily = (
        df.set_index("ts")
          .resample("1D")
          .agg(open=("open", "first"),
               high=("high", "max"),
               low=("low", "min"),
               close=("close", "last"),
               volume=("volume", "sum"))
          .dropna(subset=["open"])
          .reset_index()
    )
    daily["symbol"] = sym

    dest_dir = LAKE / f"symbol={sym}" / "tf=1d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "data.parquet"

    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / f"{sym}_1d.parquet"
    daily.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    shutil.move(str(tmp), str(dest))

    log(f"{sym}: {len(daily):,} daily bars -> {dest}")


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Pull CME futures 1m bars from Databento.")
    p.add_argument("--symbols", nargs="+", required=True,
                   help="Root symbols, e.g. ES NQ CL GC")
    p.add_argument("--start", default="2016-01-01", help="YYYY-MM-DD")
    p.add_argument("--end", default=str(date.today()), help="YYYY-MM-DD")
    p.add_argument("--confirm", action="store_true",
                   help="Actually download. Without this, only prints cost.")
    p.add_argument("--probe", action="store_true",
                   help="Report each symbol's real available date range, then exit. "
                        "Downloads nothing. Use this before a big pull.")
    p.add_argument("--max-history", action="store_true",
                   help=f"Ignore --start and pull from {DATASET_START}, the "
                        f"earliest date {DATASET} has any data.")
    p.add_argument("--schemas", nargs="+", default=["ohlcv-1m", "ohlcv-1d"],
                   choices=ALL_SCHEMAS,
                   help="Bar schemas: ohlcv-1m, ohlcv-1h, ohlcv-1d. "
                        "Other: statistics (open interest, settlement), "
                        "definition (contract specs), bbo-1m (L1 bid/ask - "
                        "BILLED, only 1yr included on Standard).")
    p.add_argument("--force", action="store_true",
                   help="Re-download symbol-years already present in the lake.")
    p.add_argument("--skip-daily", action="store_true",
                   help="Don't build derived daily bars from 1m. Ignored if "
                        "ohlcv-1d is in --schemas (native beats derived).")
    args = p.parse_args()

    if args.max_history:
        args.start = DATASET_START

    client = get_client()

    if args.probe:
        probe_symbols(client, args.symbols, args.start, args.end)
        return

    cost = preview_cost(client, args.symbols, args.start, args.end, args.schemas)

    if not args.confirm:
        print("Dry run. Nothing downloaded.")
        print("Re-run with --confirm to download.")
        return

    y0 = int(args.start[:4])
    y1 = int(args.end[:4])

    total = 0
    failures: list[str] = []

    for i, sym in enumerate(args.symbols, 1):
        log(f"[{i}/{len(args.symbols)}] {sym}")
        save_roll_calendar(client, sym, args.start, args.end)
        for schema in args.schemas:
            for year in range(y0, y1 + 1):
                try:
                    n = pull_symbol_year(client, sym, year, schema=schema,
                                         force=args.force)
                    total += n
                except Exception as e:
                    warn(f"{sym} {year} [{schema}]: unhandled error: {e}")
                    failures.append(f"{sym} {year} {schema}")

        # Only derive daily if we didn't pull it natively.
        if "ohlcv-1d" not in args.schemas and not args.skip_daily:
            build_daily(sym)

    print()
    log(f"Done. {total:,} rows written across {len(args.symbols)} symbols "
        f"and {len(args.schemas)} schema(s).")

    if failures:
        print()
        warn(f"{len(failures)} symbol-year(s) failed:")
        for f in failures:
            print(f"    {f}")
        print()
        print("Re-run the same command to retry - completed work is skipped.")
    else:
        log("No failures.")


if __name__ == "__main__":
    main()
