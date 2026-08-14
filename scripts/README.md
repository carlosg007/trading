# scripts/ — Lake Validation and Governance Utilities

## Purpose

Operational tooling over the data mount: integrity, validation, coverage, and
regime labelling. These read the lake and write `reference/` or `artifacts/`;
none of them contacts a vendor API (that is `data_pull/`'s exclusive role) and
none simulates a strategy.

## Modules

| Script | Reads | Writes |
|---|---|---|
| `generate_manifest.py` | `lake/`, `reference/` | `/mnt/backtest/manifest.json` |
| `validate_lake.py` | `lake/futures/bars` | `artifacts/validation/*.csv` |
| `coverage_summary.py` | `lake/futures/bars` | `reference/futures/coverage*.csv` |
| `classify_regime.py` | `lake/futures/bars` | `reference/futures/regimes.{parquet,csv}` |

## Two different questions

`generate_manifest.py` and `validate_lake.py` are complementary and **both**
need to pass. They do not overlap:

- **`generate_manifest.py` asks "have the bytes changed?"** It records a
  SHA-256, row count, and timestamp range per file, and `--verify` reports any
  divergence. It catches silent drift: a partial NFS write, a half-finished
  rebuild, a parser change that quietly dropped a month.
- **`validate_lake.py` asks "is the data sane?"** Duplicate timestamps,
  ordering, timezone, gaps, OHLC relationships, stray files above the partition
  level. It catches data that is stably, reproducibly wrong.

A file can match its hash perfectly and contain garbage prices. A file with
correct prices can still have been silently truncated since the last run.

## `generate_manifest.py`

```bash
python scripts/generate_manifest.py                    # scan and write the manifest
python scripts/generate_manifest.py --verify           # full check, hashes included
python scripts/generate_manifest.py --verify --quick   # size/mtime only, no hashing
python scripts/generate_manifest.py --roots lake reference raw
python scripts/generate_manifest.py --dry-run          # summarise, write nothing
```

Records per file: relative path, size (bytes/MB/GB), SHA-256, row count,
`ts_min`/`ts_max`, file mtime, and the Hive partition fields parsed from the
path. Exit codes: `0` clean, `1` drift detected, `2` no manifest or missing
mount — so `--verify` drops straight into a pipeline gate.

**Memory.** Row counts and timestamp ranges come from the Parquet footer and
row-group statistics, so files are described without decoding data pages;
hashes stream in 4 MiB chunks. A full scan of 11,067 files / 143M rows / 2.3 GB
runs in ~54 s at ~211 MB peak RSS.

**Default roots are `lake` and `reference`** — the inputs a backtest depends
on. `raw/` is 2.1 GB of write-once vendor DBN and `artifacts/` changes on every
run, so both are opt-in via `--roots`. Tracking artifacts by default would
produce constant drift warnings and train everyone to ignore the tool.

Writes via a temp file and atomic rename: a manifest truncated by an
interrupted write is worse than no manifest.

## `validate_lake.py`

```bash
python scripts/validate_lake.py                     # all symbols
python scripts/validate_lake.py --symbols ES NQ --tf 1d
python scripts/validate_lake.py --quick             # skip price checks
```

Phases 1–4: inventory, row counts, structure, price sanity. Writes
`artifacts/validation/{inventory,row_counts,issues}.csv`.

Its stray-file check exists because a flat `data.parquet` at the `tf=` root is
read alongside the partitions by anything that globs the directory, and every
bar it contains appears twice — doubled volume, distorted indicators, and a
backtest that looks perfectly fine.

**It currently reports zero.** All 24 stray files were removed on 2026-08-14:
22 after verifying their timestamps were a strict subset of the partitions, and
the last two (`ZS`, `HO`) once the daily years they uniquely held were
re-pulled from Databento into `raw/` and partitioned properly. See
`/mnt/backtest/lake/futures/README.md`.

## `coverage_summary.py`

```bash
python scripts/coverage_summary.py
```

Rebuilds `reference/futures/coverage.csv` and `coverage_yearly.csv`, including
`intraday_start_year` detection. **Must be re-run after any lake rebuild** —
`mdlib.lake`'s `respect_coverage` flag reads it, and stale coverage means
intraday backtests silently use sparse pre-2013 minutes.

Note `data_pull/coverage_summary.py` is a superseded copy without the newer
`find_intraday_start` detection. Use this one.

## `classify_regime.py`

```bash
python scripts/classify_regime.py --threshold 10.0
```

Labels symbol-years Bull/Bear/Neutral into `reference/futures/regimes.parquet`
and `.csv`. Reads daily bars through `mdlib.lake`, so it gets the Sunday-session
merge — without it, ~51 stub "days" a year distort both the 200-day SMA and the
realized-vol figure it reports.

Generated 2026-08-14: 434 symbol-years, 27 symbols, 2010–2026
(130 Bull / 81 Bear / 223 Neutral). `backtest/report.py`'s regime join resolves
against it.

**Re-run after any lake rebuild** — the labels are derived from daily closes, so
a changed lake silently invalidates them.

## Update Cadence

All on demand. Nothing here is scheduled. The standing sequence after any
ingest or lake rebuild:

```bash
python scripts/coverage_summary.py      # coverage is now stale
python scripts/validate_lake.py         # structure and price sanity
python scripts/generate_manifest.py     # refresh the integrity baseline
```

Then `--verify` before any run whose result you intend to act on.

## Continuous Contract Convention

These tools operate on **unadjusted continuous** series and must not assume
otherwise.

Two consequences that show up here specifically:

- **Price-sanity checks see roll gaps.** `validate_lake.py`'s extreme-move
  check will flag legitimate roll boundaries. A flagged bar that coincides with
  a date in `reference/futures/roll_calendar_<SYM>.json` is expected, not a
  data error.
- **`classify_regime.py` measures returns across full years**, which necessarily
  span rolls. Regime labels computed from an unadjusted series inherit the roll
  gaps; treat them as coarse context, not as a return series.

Full statement: `/mnt/backtest/README.md`.
