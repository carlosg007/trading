#!/usr/bin/env python3
"""
pull_definitions.py - Pull the `definition` schema (contract specs).

Location:  ~/src/trading/data_pull/pull_definitions.py

Why this exists separately from pull_futures.py
-----------------------------------------------
`pull_futures.py --schemas definition` already pulls this schema, but it
requests one whole calendar year per call. Year-long definition requests
currently fail on this connection:

    Error streaming response: Response ended prematurely

every time, for every symbol, while the same request over a few days returns
instantly. The payload is tiny (a continuous symbol yields ~250 rows a year);
it is the length of the stream that breaks, not the size. `pull_futures.py`
derives its window from the year alone, so the window cannot be narrowed from
the command line.

This script chunks the range into short windows and retries each one with
backoff, which makes the pull survive a flaky stream. Everything else matches
pull_futures.py: same dataset, same continuous symbology, same output layout,
cost preview first, and nothing downloads without --confirm.

Output (identical to pull_futures.py, so verify_specs() reads it unchanged):

    /mnt/backtest/reference/futures/definitions/symbol=<SYM>/<YYYY>.parquet

Writes merge into an existing year file rather than replacing it, so a narrow
"just get the current spec" run can be widened by a later backfill without
losing anything.

Usage
-----
    export DATABENTO_API_KEY="db-..."

    # cost preview only, downloads nothing
    python data_pull/pull_definitions.py --symbols PL ZC 6E

    # current specs: a recent window is enough to verify SPECS
    python data_pull/pull_definitions.py --symbols PL ZC 6E \
        --start 2026-06-01 --confirm

    # full history, to catch mid-history spec changes (slow)
    python data_pull/pull_definitions.py --symbols PL ZC 6E \
        --max-history --confirm

Then reconcile:

    python -m backtest.specs
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
import os
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    import databento as db
except ImportError:
    sys.exit("databento not installed. Activate the trading venv first.")

DATASET = "GLBX.MDP3"
SCHEMA = "definition"
ROLL_RULE = "v"
RANK = 0
DATASET_START = "2010-06-06"

DEFINITIONS = Path("/mnt/backtest/reference/futures/definitions")
RAW = Path("/mnt/backtest/raw/futures")
SCRATCH = Path.home() / "scratch"

CHUNK_DAYS = 30
RETRIES = 4
BACKOFF = 3.0          # seconds, doubled each retry


def log(m: str) -> None:
    print(f"==> {m}", flush=True)


def warn(m: str) -> None:
    print(f"[!] {m}", file=sys.stderr, flush=True)


def continuous_symbol(sym: str) -> str:
    """ES -> ES.v.0"""
    return f"{sym}.{ROLL_RULE}.{RANK}"


def get_client() -> "db.Historical":
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY is not set.\n"
                 '  export DATABENTO_API_KEY="db-xxxxxxxx"')
    return db.Historical(key)


def chunks(start: str, end: str, days: int) -> list[tuple[str, str]]:
    """Split [start, end) into windows of at most `days` days."""
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    out = []
    while s < e:
        nxt = min(s + timedelta(days=days), e)
        out.append((str(s), str(nxt)))
        s = nxt
    return out


def preview_cost(client, symbols: list[str], start: str, end: str) -> float:
    log("Pricing request (no data downloaded yet)")
    csyms = [continuous_symbol(s) for s in symbols]
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET, symbols=csyms, schema=SCHEMA,
            start=start, end=end, stype_in="continuous",
        )
    except Exception as e:
        warn(f"Cost preview failed: {e}")
        cost = float("nan")

    print()
    print(f"  Dataset : {DATASET}")
    print(f"  Schema  : {SCHEMA}")
    print(f"  Symbols : {len(symbols)} -> {', '.join(csyms)}")
    print(f"  Range   : {start} -> {end}")
    print(f"  COST    : ${cost:,.2f}")
    print()
    return cost


def fetch_window(client, sym: str, start: str, end: str) -> pd.DataFrame | None:
    """One window, with retries. None means every attempt failed."""
    csym = continuous_symbol(sym)
    delay = BACKOFF
    for attempt in range(1, RETRIES + 1):
        try:
            store = client.timeseries.get_range(
                dataset=DATASET, symbols=[csym], schema=SCHEMA,
                start=start, end=end, stype_in="continuous",
            )
            # Archive the raw response before touching it, as pull_futures does.
            try:
                RAW.mkdir(parents=True, exist_ok=True)
                store.to_file(RAW / f"{sym}_{SCHEMA}_{start}_{end}.dbn.zst")
            except Exception as e:
                warn(f"{sym} {start}: could not archive raw DBN: {e}")
            return store.to_df()
        except Exception as e:
            if attempt == RETRIES:
                warn(f"{sym} {start}->{end}: failed after {RETRIES} attempts: {e}")
                return None
            warn(f"{sym} {start}->{end}: attempt {attempt} failed ({e}); "
                 f"retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    return None


def write_year(df: pd.DataFrame, sym: str, year: int) -> int:
    """
    Merge `df` into symbol=<SYM>/<year>.parquet.

    Merging rather than overwriting means a narrow run does not destroy a
    previous full-year file.
    """
    out = df.reset_index()
    for c in ("ts_recv", "ts_ref"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    if "ts_event" in out.columns:
        out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True, errors="coerce")
        out = out.rename(columns={"ts_event": "ts"})
    out["symbol"] = sym

    dest_dir = DEFINITIONS / f"symbol={sym}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{year}.parquet"

    if dest.exists():
        try:
            prev = pd.read_parquet(dest)
            out = pd.concat([prev, out], ignore_index=True)
        except Exception as e:
            warn(f"{sym} {year}: existing file unreadable, replacing it: {e}")

    subset = [c for c in ("ts", "instrument_id", "raw_symbol") if c in out.columns]
    if subset:
        out = out.drop_duplicates(subset=subset, keep="last")
    if "ts" in out.columns:
        out = out.sort_values("ts")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / f"{sym}_{SCHEMA}_{year}.parquet"
    out.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    # shutil.move, not Path.replace: scratch is local disk and the lake is on
    # NFS, and os.rename cannot cross filesystems.
    shutil.move(str(tmp), str(dest))
    return len(out)


def pull_symbol(client, sym: str, start: str, end: str,
                chunk_days: int) -> tuple[int, int]:
    """Returns (rows written, windows that failed)."""
    frames: dict[int, list[pd.DataFrame]] = {}
    failed = 0

    for w_start, w_end in chunks(start, end, chunk_days):
        df = fetch_window(client, sym, w_start, w_end)
        if df is None:
            failed += 1
            continue
        if df.empty:
            continue
        frames.setdefault(pd.Timestamp(w_start).year, []).append(df)

    written = 0
    for year, parts in sorted(frames.items()):
        written += write_year(pd.concat(parts, ignore_index=False), sym, year)

    if written:
        log(f"{sym}: {written:,} rows across {len(frames)} year file(s)")
    else:
        warn(f"{sym}: no definition rows written")
    return written, failed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pull the definition schema in retryable chunks.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", default="2026-01-01", help="YYYY-MM-DD")
    p.add_argument("--end", default=str(date.today()), help="YYYY-MM-DD")
    p.add_argument("--max-history", action="store_true",
                   help=f"Ignore --start and pull from {DATASET_START}.")
    p.add_argument("--chunk-days", type=int, default=CHUNK_DAYS,
                   help=f"Window size per request (default {CHUNK_DAYS}). "
                        f"Lower it if streams keep truncating.")
    p.add_argument("--confirm", action="store_true",
                   help="Actually download. Without this, only prints cost.")
    args = p.parse_args()

    if args.max_history:
        args.start = DATASET_START
    if args.start < DATASET_START:
        args.start = DATASET_START

    client = get_client()
    preview_cost(client, args.symbols, args.start, args.end)

    if not args.confirm:
        print("Dry run. Nothing downloaded.")
        print("Re-run with --confirm to download.")
        return

    total = 0
    problems: list[str] = []
    for i, sym in enumerate(args.symbols, 1):
        log(f"[{i}/{len(args.symbols)}] {sym} {args.start} -> {args.end}")
        try:
            n, failed = pull_symbol(client, sym, args.start, args.end,
                                    args.chunk_days)
        except Exception as e:
            warn(f"{sym}: unhandled error: {e}")
            problems.append(f"{sym} (error: {e})")
            continue
        total += n
        if n == 0:
            problems.append(f"{sym} (no rows)")
        elif failed:
            problems.append(f"{sym} ({failed} window(s) failed)")

    print()
    log(f"Done. {total:,} rows across {len(args.symbols)} symbol(s).")
    if problems:
        print()
        warn("Incomplete:")
        for s in problems:
            print(f"    {s}")
        print("Re-run to retry - existing rows are merged, not duplicated.")
    print()
    print("Now reconcile:  python -m backtest.specs")


if __name__ == "__main__":
    main()
