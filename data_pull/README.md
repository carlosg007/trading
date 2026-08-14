# data_pull/ — Vendor Ingest

## Purpose

The **only** layer in this codebase permitted to talk to a vendor API. Writes
`/mnt/backtest/raw/` and `/mnt/backtest/lake/`; reads nothing from the research
tiers above it.

The one-way dependency is `agents` → `strategies` → `backtest` → `mdlib` →
lake, with `data_pull` writing the lake from the side. If a module outside this
directory acquires a vendor API call, that boundary is broken and there is no
longer a single place to audit what was downloaded, when, and at what cost.

## Modules

| Module | Role | Writes |
|---|---|---|
| `pull_futures.py` | Download CME bars from Databento | `raw/futures/`, `lake/futures/bars/` |
| `pull_definitions.py` | Download the `definition` schema (contract specs) | `raw/futures/`, `reference/futures/definitions/` |
| `fetch_degraded_days.py` | Refresh the data-quality calendar (free metadata call) | `reference/futures/degraded_days.csv` |
| `ingest_nt8.py` | Normalise NT8 CSV exports | `lake/futures_nt8/` |
| `coverage_summary.py` | **Superseded** — use `scripts/coverage_summary.py` | — |

`coverage_summary.py` here is a stale copy; its own docstring points at
`scripts/`, and only the `scripts/` version has the newer `find_intraday_start`
detection. It should be deleted or the duplication made explicit.

`ingest_nt8.py` is currently **untracked in git**. Commit it — the NT8 lake is
unreproducible without it.

## Source Provenance

Two vendors, deliberately kept in separate trees that can never be queried
together:

- **Databento** (`GLBX.MDP3`) → `raw/futures/`, `lake/futures/`. Exchange feed,
  16 years, the In-Sample research dataset.
- **NinjaTrader 8** → `raw/futures_nt8/`, `lake/futures_nt8/`. Broker-side data,
  transferred by hand from a Windows machine. No API.

See `/mnt/backtest/README.md` for why they never mix.

## Schema & Types

Written to the lake as Parquet/ZSTD:

| Column | Databento | NT8 |
|---|---|---|
| `ts` | `timestamp[ns, tz=UTC]` | `timestamp[us, tz=UTC]` |
| `open`/`high`/`low`/`close` | `double` | `double` |
| `volume` | `uint64` | `int64` |
| `symbol` | `large_string` | `large_string` |

The `ns`/`us` and `uint64`/`int64` divergence is intentional — it makes an
accidental concatenation of the two feeds fail rather than silently upcast.

Partition layout, written by every ingest:

```
lake/<family>/bars/symbol=<SYM>/tf=<1m|1d>/year=<YYYY>/month=<MM>/data.parquet
```

**Never write a Parquet file above the `year=` level.** A flat `data.parquet`
at the `tf=` root is read alongside the partitions by anything that globs the
directory, and every bar appears twice.

## Update Cadence

Nothing here runs on a schedule. Every ingest is an explicit, operator-initiated
command, because downloads are billed.

```bash
export DATABENTO_API_KEY="db-..."

# Cost is printed first. Nothing downloads without --confirm.
python data_pull/pull_futures.py --symbols ES NQ --start 2016-01-01 --end 2026-01-01
python data_pull/pull_futures.py --symbols ES NQ --start 2016-01-01 --end 2026-01-01 --confirm

python data_pull/fetch_degraded_days.py     # free metadata call
python data_pull/ingest_nt8.py --tz <exporting machine timezone>
```

**After every ingest, without exception:**

```bash
python scripts/coverage_summary.py      # coverage.csv is now stale
python scripts/validate_lake.py         # structure and price sanity
python scripts/generate_manifest.py     # refresh the integrity record
```

## Continuous Contract Convention

**This layer is where the splice happens, and therefore where it can go wrong.**

Continuous series are built front-month, spliced at the vendor's roll, and are
**NOT back-adjusted** — prices are left exactly as each contract traded, so a
roll appears as a genuine gap between adjacent bars.

- Roll dates are resolved from Databento's symbology endpoint and written to
  `reference/futures/roll_calendar_<SYM>.json`. They are recorded as data, not
  inferred downstream from price action.
- Back-adjusted series are deliberately never written. Back-adjustment destroys
  the true price level, and tick values, margin, and prop-firm drawdown limits
  are all denominated in true prices.
- `ingest_nt8.py` is the exception: NT8 exports an already-continuous series
  spliced on **its own** roll rules, so no roll decision is available to this
  layer and no NT8 roll calendar exists.

Full statement: `/mnt/backtest/README.md`.
