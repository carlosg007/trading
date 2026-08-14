# mdlib/ — The Lake Reader

## Purpose

The single sanctioned way to read market data. Everything above this layer
(`backtest`, `strategies`, `agents`) goes through `mdlib.lake` and never touches
Parquet directly.

That is not stylistic. The reader is where the Sunday-session merge, partition
pruning, timeframe derivation, and the hygiene filters live. Code that reads the
Parquet files directly gets none of them and produces plausible-looking wrong
answers.

## Public API

```python
from mdlib.lake import iter_bars, get_bars, wide, describe

available_symbols()          # tuple of symbols present in the lake
coverage()                   # reference/futures/coverage.csv as a DataFrame
intraday_start_year(symbol)  # first year with dense 1m data
degraded_days()              # frozenset of vendor-flagged bad sessions
roll_dates(symbol)           # tuple of roll boundary dates
iter_bars(...)               # yields (symbol, df), one symbol at a time
get_bars(...)                # single concatenated frame
wide(df, field="close")      # pivot long -> one column per symbol
describe(symbols)            # summary table
```

Both readers return **long format**: `ts, symbol, open, high, low, close,
volume`, always UTC.

## `iter_bars` vs `get_bars`

**Use `iter_bars` unless you genuinely need a cross-sectional frame.**

`iter_bars` yields one symbol at a time. Peak RAM tracks the largest single
symbol (5.6M rows), not the lake — a full-lake backtest peaks around 2.9 GiB,
against 20.5 GiB for building the frame first.

`get_bars` concatenates and sorts by `(ts, symbol)`, so the result
**interleaves instruments**. A strategy doing `close.rolling(200).mean()` over
that frame averages across 27 unrelated contracts. Nothing raises: the signals
come out the right length and dtype, and the equity curve looks plausible. This
produced 608,079 trades where correct per-symbol signals give 86,035.

That is why `backtest.engine.run_backtest` takes a `signal_fn` and reads the
bars itself. There is deliberately **no** entry point that accepts a pre-built
multi-symbol frame with signals already computed. Do not reintroduce one.

Use `get_bars` for correlation work and `wide()` pivots, where the
cross-section is the point. On the full 1m lake it costs ~15 GiB.

## Source Provenance

Reads `/mnt/backtest/lake/futures/bars` (Databento) and reference data from
`/mnt/backtest/reference/futures/`, the latter via `lru_cache` — a running
process will not see a regenerated `coverage.csv` until it restarts.

**Dual-dataset routing is not implemented.** `get_bars` has no `source`
parameter and the reader points at the Databento tree only. `lake/futures_nt8/`
exists on disk and is currently unreachable through this module; reaching it
means adding the source flag described in `CLAUDE.md`'s open tasks.

## Schema & Types

Returned frames: `ts` (`datetime64[ns, UTC]`), `open`/`high`/`low`/`close`
(`float64`), `volume`, `symbol`. Underlying storage is Parquet/ZSTD — see
`/mnt/backtest/lake/futures/README.md` for on-disk Arrow types.

## Session and Timeframe Semantics

- **Only `1m` and `1d` are stored.** `5m/15m/30m/1h/2h/4h` resample from `1m`;
  `1w` from `1d`. Adding a timeframe is a derivation rule, not a re-pull.
- **Sunday sessions merge into Monday** for `1d`/`1w` (`session_merge=True`).
  CME opens Sunday 18:00 ET, so a naive UTC day boundary manufactures ~51 thin
  stub "days" a year that silently corrupt every lookback window. The merge
  lives here, in the reader — one function to change if it is ever wrong.
- **Partition pruning on `year`** happens before any file is opened; at ~1 ms of
  NFS latency per file this dominates read time. Only `year=*` directories are
  read, so anything written above that level is invisible to this module.

  That property once hid a real gap. Until 2026-08-14, ZS `2020`-`2021` and HO
  `2012` existed only in un-partitioned stray files, and the reader returned a
  gapped series with **no marker** — `iter_bars(["ZS"], "1d", "2019-01-01",
  "2022-12-31")` gave 503 bars with 2019-12-31 and 2022-01-03 adjacent, so a
  `pct_change()` booked a two-year move as one day's return. Both gaps are now
  backfilled from Databento and the series are contiguous:

  ```
  iter_bars(["ZS"], "1d", "2019-01-01", "2022-12-31")
      -> 1008 bars: {2019: 252, 2020: 253, 2021: 252, 2022: 251}
  ```

  The lesson survives the fix: **a missing `year=` directory produces a silent
  hole, not an error.** Check coverage before trusting a span.

## Hygiene Flags

| Flag | Effect |
|---|---|
| `exclude_degraded` | Drop sessions Databento flagged as degraded at source |
| `exclude_rolls` | Drop bars at contract roll boundaries |
| `respect_coverage` | Honour `intraday_start_year` — pre-2013 1m is sparse for ten symbols |

## Update Cadence

Code, not data. The lake it reads changes only on an explicit ingest.

## Continuous Contract Convention

Series returned by this module are **unadjusted continuous** contracts: real
price gaps exist at every roll, because prices are left exactly as each contract
traded.

**`exclude_rolls` is the supported way to handle this.** Use it rather than
reimplementing a roll filter per strategy — the roll dates come from
`reference/futures/roll_calendar_<SYM>.json`, which records the vendor's
symbology resolution rather than inferring rolls from price jumps.

A `close.pct_change()` over an unfiltered continuous series books every roll gap
as P&L. Across 27 symbols and 16 years that is large enough to manufacture an
edge that does not exist.

Full statement: `/mnt/backtest/README.md`.
