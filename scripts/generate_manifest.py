#!/usr/bin/env python3
"""
generate_manifest.py - Content inventory and drift detection for the data mount.

Location:  ~/src/trading/scripts/generate_manifest.py

Records, for every data file under the scanned roots: relative path, size,
SHA-256, row count, timestamp range, and the file's own generation time. The
result is written to /mnt/backtest/manifest.json, and `--verify` re-scans the
mount and reports anything that no longer matches.

What this is for
----------------
The lake is rebuilt from raw/ by scripts that get edited. A parser change that
silently drops a month, a partial write over NFS, or a half-finished rebuild
all leave a lake that still *reads* fine - correct dtypes, plausible prices,
no exception. The backtest on top of it is then wrong in a way no assertion
catches.

A manifest turns that into a diff. If a file's bytes change, `--verify` says
so and names the file. Drift stops being something you discover from a
suspicious Sharpe ratio three weeks later.

This is an integrity check, not a validity check. A file can match its hash
perfectly and still contain garbage prices - that is what validate_lake.py is
for. Run both.

Memory
------
Row counts and timestamp ranges come from the Parquet footer and row-group
statistics, so a 30k-row file is described without decoding a single value.
Hashes stream in 4 MiB chunks. Peak RSS tracks the largest single column read
in the statistics fallback, not the size of the lake, so a full scan of the
mount runs in well under 200 MB.

Usage
-----
    python scripts/generate_manifest.py                    # scan, write manifest
    python scripts/generate_manifest.py --verify           # check mount vs manifest
    python scripts/generate_manifest.py --verify --quick   # size/mtime only, no hashing
    python scripts/generate_manifest.py --roots lake reference raw
    python scripts/generate_manifest.py --dry-run          # print summary, write nothing
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
import concurrent.futures as cf
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

MOUNT = Path("/mnt/backtest")
MANIFEST = MOUNT / "manifest.json"

# Roots scanned by default: everything a backtest reads. raw/ is write-once
# vendor data and 2 GB of DBN, so it is opt-in via --roots rather than paid
# for on every run.
DEFAULT_ROOTS = ["lake", "reference"]

DATA_SUFFIXES = {".parquet", ".csv", ".json", ".dbn", ".zst"}

# Names that live in the data tree but are not data.
EXCLUDE_NAMES = {"manifest.json", ".DS_Store"}

CHUNK = 4 * 1024 * 1024
MANIFEST_VERSION = 1

# Candidate timestamp columns, in preference order. The lake uses `ts`;
# Databento statistics files also carry ts_recv.
TS_COLUMNS = ("ts", "ts_recv", "ts_event", "timestamp", "date")


def log(m): print(f"==> {m}", flush=True)
def warn(m): print(f"[!] {m}", flush=True)


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------
def discover_files(roots: list[Path]) -> list[Path]:
    """Every data file under the given roots, sorted for a stable manifest."""
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            warn(f"root does not exist, skipping: {root}")
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.name in EXCLUDE_NAMES:
                continue
            if p.suffix.lower() in DATA_SUFFIXES:
                found.append(p)
    return sorted(found)


def sha256_of(path: Path) -> str:
    """Stream the file through the hash so RAM stays flat regardless of size."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _iso(value) -> str | None:
    """Normalise whatever the Parquet statistics hand back to a UTC ISO string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return str(value)


def parquet_facts(path: Path) -> dict:
    """
    Row count and timestamp range, read from the footer wherever possible.

    Row counts are always free - they live in the footer. The timestamp range
    normally comes from row-group statistics, which are also free. Only when a
    writer omitted statistics do we fall back to reading the timestamp column,
    and even then it is one column, never the whole file.
    """
    out = {"rows": None, "ts_min": None, "ts_max": None, "ts_column": None,
           "stats_source": "none"}
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        out["error"] = f"unreadable parquet: {e}"
        return out

    md = pf.metadata
    out["rows"] = md.num_rows

    names = pf.schema_arrow.names
    ts_col = next((c for c in TS_COLUMNS if c in names), None)
    if ts_col is None:
        return out
    out["ts_column"] = ts_col

    # Row-group statistics: min/max without touching the data pages.
    idx = names.index(ts_col)
    lo = hi = None
    have_stats = md.num_row_groups > 0
    for rg in range(md.num_row_groups):
        st = md.row_group(rg).column(idx).statistics
        if st is None or not st.has_min_max:
            have_stats = False
            break
        lo = st.min if lo is None or st.min < lo else lo
        hi = st.max if hi is None or st.max > hi else hi

    if have_stats and lo is not None:
        out.update(ts_min=_iso(lo), ts_max=_iso(hi),
                   stats_source="row_group_statistics")
        return out

    # Fallback: one column only. Empty files legitimately have no range.
    try:
        col = pq.read_table(path, columns=[ts_col])[ts_col]
        if len(col):
            out.update(ts_min=_iso(col[0].as_py()), ts_max=_iso(col[-1].as_py()))
        out["stats_source"] = "column_scan"
    except Exception as e:
        out["error"] = f"ts scan failed: {e}"
    return out


def csv_facts(path: Path) -> dict:
    """Row count for CSV by streaming lines. No date range - too costly to infer."""
    try:
        with open(path, "rb") as fh:
            lines = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(CHUNK), b""))
        return {"rows": max(lines - 1, 0), "ts_min": None, "ts_max": None,
                "ts_column": None, "stats_source": "line_count"}
    except Exception as e:
        return {"rows": None, "ts_min": None, "ts_max": None, "ts_column": None,
                "stats_source": "none", "error": f"csv read failed: {e}"}


def partition_fields(rel: Path) -> dict:
    """
    Pull Hive-style key=value partitions out of the path.

    Deliberately generic: bars partition on symbol/tf/year/month, statistics on
    symbol/year, and a future dataset will invent its own. Parsing whatever is
    there beats hardcoding one layout.
    """
    fields: dict[str, object] = {}
    for part in rel.parts:
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        fields[key] = int(value) if value.isdigit() else value
    return fields


def classify(rel: Path) -> dict:
    """Dataset and source labels derived from position in the tree."""
    parts = rel.parts
    info: dict[str, object] = {"source": None, "dataset": None}
    if not parts:
        return info

    if parts[0] == "lake" and len(parts) > 1:
        family = parts[1]                       # futures | futures_nt8 | ...
        info["source"] = "nt8" if family.endswith("_nt8") else "databento"
        kind = parts[2] if len(parts) > 2 and "=" not in parts[2] else None
        info["dataset"] = f"{family}/{kind}" if kind else family
    elif parts[0] == "raw" and len(parts) > 1:
        info["source"] = "nt8" if parts[1].endswith("_nt8") else "databento"
        info["dataset"] = f"raw/{parts[1]}"
    elif parts[0] == "reference":
        info["source"] = "derived"
        info["dataset"] = "/".join(parts[:2])
    else:
        info["dataset"] = parts[0]
    return info


def describe(path: Path, mount: Path, with_hash: bool = True) -> dict:
    """Build one manifest entry. Safe to call from a worker thread."""
    rel = path.relative_to(mount)
    stat = path.stat()

    entry: dict[str, object] = {
        "path": str(rel),
        "format": path.suffix.lower().lstrip("."),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / 1024**2, 4),
        "size_gb": round(stat.st_size / 1024**3, 6),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    entry.update(classify(rel))
    entry.update(partition_fields(rel))

    if path.suffix.lower() == ".parquet":
        entry.update(parquet_facts(path))
    elif path.suffix.lower() == ".csv":
        entry.update(csv_facts(path))
    else:
        entry.update({"rows": None, "ts_min": None, "ts_max": None,
                      "ts_column": None, "stats_source": "none"})

    entry["sha256"] = sha256_of(path) if with_hash else None
    return entry


def build_entries(files: list[Path], mount: Path, workers: int,
                  with_hash: bool = True) -> list[dict]:
    """
    Describe every file, threaded.

    The work is NFS-bound, not CPU-bound - hashlib and the Parquet reader both
    release the GIL - so threads help and processes would only add overhead.
    """
    entries: list[dict] = []
    total = len(files)
    step = max(total // 20, 1)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(describe, f, mount, with_hash): f for f in files}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            src = futures[fut]
            try:
                entries.append(fut.result())
            except Exception as e:
                warn(f"failed: {src} ({e})")
                entries.append({"path": str(src.relative_to(mount)), "error": str(e)})
            if i % step == 0 or i == total:
                print(f"    {i}/{total} files", flush=True)

    entries.sort(key=lambda e: e["path"])
    return entries


# --------------------------------------------------------------------------
# manifest assembly
# --------------------------------------------------------------------------
def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def summarize(entries: list[dict]) -> dict:
    ok = [e for e in entries if "error" not in e]
    by_dataset: dict[str, dict] = {}
    for e in ok:
        d = by_dataset.setdefault(
            e.get("dataset") or "unknown",
            {"files": 0, "bytes": 0, "rows": 0, "ts_min": None, "ts_max": None},
        )
        d["files"] += 1
        d["bytes"] += e.get("size_bytes") or 0
        d["rows"] += e.get("rows") or 0
        for key, better in (("ts_min", min), ("ts_max", max)):
            v = e.get(key)
            if v is not None:
                d[key] = v if d[key] is None else better(d[key], v)

    for d in by_dataset.values():
        d["size_gb"] = round(d["bytes"] / 1024**3, 4)

    total_bytes = sum(e.get("size_bytes") or 0 for e in ok)
    return {
        "file_count": len(entries),
        "error_count": len(entries) - len(ok),
        "total_bytes": total_bytes,
        "total_size_gb": round(total_bytes / 1024**3, 4),
        "total_rows": sum(e.get("rows") or 0 for e in ok),
        "by_dataset": dict(sorted(by_dataset.items())),
    }


def generate(mount: Path, roots: list[str], workers: int, out: Path,
             dry_run: bool) -> int:
    root_paths = [mount / r for r in roots]
    log(f"scanning {', '.join(str(p) for p in root_paths)}")
    files = discover_files(root_paths)
    log(f"{len(files)} data files found; hashing with {workers} workers")

    entries = build_entries(files, mount, workers)
    summary = summarize(entries)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": "scripts/generate_manifest.py",
            "git_commit": git_commit(),
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
        },
        "mount": str(mount),
        "roots": roots,
        "hash_algorithm": "sha256",
        "summary": summary,
        "files": entries,
    }

    log(f"{summary['file_count']} files, {summary['total_size_gb']} GB, "
        f"{summary['total_rows']:,} rows")
    for name, d in summary["by_dataset"].items():
        span = f"{(d['ts_min'] or '?')[:10]} .. {(d['ts_max'] or '?')[:10]}"
        print(f"    {name:<28} {d['files']:>6} files  {d['size_gb']:>8.3f} GB  "
              f"{d['rows']:>12,} rows  {span}")
    if summary["error_count"]:
        warn(f"{summary['error_count']} files could not be described")

    if dry_run:
        log("dry run - manifest not written")
        return 0

    # Write to a temp file in the same directory, then rename. A manifest
    # truncated by an interrupted write is worse than no manifest.
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out)
    log(f"wrote {out} ({out.stat().st_size / 1024**2:.2f} MB)")
    return 0


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def verify(mount: Path, manifest_path: Path, workers: int, quick: bool) -> int:
    """Compare the mount against a stored manifest. Returns a shell exit code."""
    if not manifest_path.exists():
        warn(f"no manifest at {manifest_path} - run without --verify first")
        return 2

    manifest = json.loads(manifest_path.read_text())
    recorded = {e["path"]: e for e in manifest["files"]}
    roots = manifest.get("roots", DEFAULT_ROOTS)

    log(f"manifest generated {manifest.get('generated_at')} "
        f"({len(recorded)} files, roots: {', '.join(roots)})")
    if quick:
        log("quick mode - comparing size and mtime only, no hashing")

    current_files = discover_files([mount / r for r in roots])
    current_entries = build_entries(current_files, mount, workers,
                                    with_hash=not quick)
    current = {e["path"]: e for e in current_entries}

    missing = sorted(set(recorded) - set(current))
    added = sorted(set(current) - set(recorded))
    modified: list[tuple[str, list[str]]] = []

    for path in sorted(set(recorded) & set(current)):
        was, now = recorded[path], current[path]
        deltas = []
        if was.get("size_bytes") != now.get("size_bytes"):
            deltas.append(f"size {was.get('size_bytes')} -> {now.get('size_bytes')}")
        if not quick and was.get("sha256") != now.get("sha256"):
            deltas.append("sha256 differs")
        if was.get("rows") != now.get("rows"):
            deltas.append(f"rows {was.get('rows')} -> {now.get('rows')}")
        if was.get("ts_min") != now.get("ts_min"):
            deltas.append(f"ts_min {was.get('ts_min')} -> {now.get('ts_min')}")
        if was.get("ts_max") != now.get("ts_max"):
            deltas.append(f"ts_max {was.get('ts_max')} -> {now.get('ts_max')}")
        # In quick mode mtime is the only signal that content moved, so it is
        # promoted to a difference. With hashing on it is noise: a touched file
        # with identical bytes has not drifted.
        if quick and was.get("modified_at") != now.get("modified_at"):
            deltas.append(f"mtime {was.get('modified_at')} -> {now.get('modified_at')}")
        if deltas:
            modified.append((path, deltas))

    print()
    log(f"missing: {len(missing)}   new: {len(added)}   modified: {len(modified)}")

    for label, items in (("MISSING (in manifest, not on disk)", missing),
                         ("NEW (on disk, not in manifest)", added)):
        if items:
            print(f"\n{label}:")
            for p in items[:50]:
                print(f"    {p}")
            if len(items) > 50:
                print(f"    ... and {len(items) - 50} more")

    if modified:
        print("\nMODIFIED:")
        for p, deltas in modified[:50]:
            print(f"    {p}\n        {'; '.join(deltas)}")
        if len(modified) > 50:
            print(f"    ... and {len(modified) - 50} more")

    if missing or added or modified:
        print()
        warn("manifest and mount disagree - investigate before trusting a backtest")
        return 1

    print()
    log("clean - every file matches the manifest")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate or verify the data mount manifest.")
    ap.add_argument("--verify", action="store_true",
                    help="check the mount against the existing manifest")
    ap.add_argument("--quick", action="store_true",
                    help="with --verify: compare size and mtime only, skip hashing")
    ap.add_argument("--mount", type=Path, default=MOUNT,
                    help=f"data mount root (default {MOUNT})")
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS,
                    help=f"subdirectories to scan (default: {' '.join(DEFAULT_ROOTS)})")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest path (default <mount>/manifest.json)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel file readers (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and summarise without writing the manifest")
    args = ap.parse_args()

    if args.quick and not args.verify:
        ap.error("--quick only applies with --verify")

    mount = args.mount
    if not mount.exists():
        warn(f"mount not found: {mount}")
        return 2
    manifest_path = args.manifest or (mount / "manifest.json")

    if args.verify:
        return verify(mount, manifest_path, args.workers, args.quick)
    return generate(mount, args.roots, args.workers, manifest_path, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
