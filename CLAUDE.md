# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

An autonomous, AI-driven quantitative research environment and prop-firm
compliance pipeline. Data lives on a hard-mounted NFS at `/mnt/backtest` (not in
the repo); code lives here. The goal is to generate, backtest, stress-test, and
deploy multi-asset portfolios that strictly adhere to proprietary trading firm
constraints.

**The failure mode this project is built to avoid is an overfitted backtest that
looks right and fails in live markets.** Most of the conventions below exist to
combat curve-fitting and silent data corruption.

---

## Working Agreement & AI Boundaries

**Verify before asserting.** Do not report that something works because the code
looks correct. Run it. For anything touching P&L, costs, fills, or session
boundaries, construct a case where the answer is known and check the output
matches.

**Respect the AI Division of Labor.**

- **Claude Code (You):** the local terminal expert. Deep multi-file code editing,
  git version control, fixing syntax/indentation errors, scaffolding Python
  environments.
- **Google AI Agents (The Orchestrators):** macro-orchestration — generating
  strategy hypotheses, triggering backtests, running compliance checks. Do not
  attempt to override the Tier 1 (CIO) or Tier 2 (Supervisor) agents.

**Commit before large changes.** `git status` first. The repo is the only backup
for code.

**Push back on requests that would compromise a result.** Removing a cost model,
bypassing the Out-of-Sample (OOS) holdout test, or relaxing prop-firm drawdown
rules to "see if the strategy works" are things to flag rather than do. Being
useful here means being a skeptic.

---
## Token Efficiency & Context Management

**Be concise.** Prefer executing commands and writing code over generating long, conversational explanations in the terminal. 
**Targeted code edits.** When modifying a file, make surgical edits. Do not output the entire file into the chat unless explicitly requested. 
**Never read raw data files.** Do not attempt to `cat`, `head`, or ingest any `.parquet`, `.csv`, or large log files from `/mnt/backtest` into the context window. Use Python scripts to aggregate or print summaries instead.
**Search before reading.** Use `grep` or `rg` (ripgrep) to find specific functions, classes, or variables across the repository instead of reading multiple whole files into context.
**No unprompted refactoring.** Do not refactor working code or reformat files as a side effect of completing a task. Keep the scope of changes as small as possible.

## Library & Execution Discipline

**The Vectorbt Pro Guardrail:** You are using `vectorbtpro`, which is a private, paid library. Its API differs significantly from the free, open-source `vectorbt`. **Do not hallucinate functions.** If you are unsure of a Vectorbt Pro method, write a temporary script to run `dir()`, `help()`, or inspect the library's docstrings via the terminal before writing the implementation.
**Atomic Git Commits:** Make granular, single-purpose commits. If you fix a data bug, commit it. If you then optimize a loop, commit it separately. If a machine learning strategy generation script breaks the server, we must be able to roll back just that script without losing the data fixes.
**Test-Driven Edits:** When refactoring critical engine components (like `backtest/engine.py`), do not modify the main file immediately. Write a minimal reproducible test script (e.g., `test_vbt_array.py`) to prove your multidimensional array logic works on a small slice of data first.

## Environment & Infrastructure

The environment runs on a dedicated Ubuntu 26.04 LTS Proxmox VM configured for
heavy Vectorbt Pro grid searches (8-16 vCPUs, 32-64GB RAM).

Python 3.13 in `.venv` at the repo root (managed with `uv`). Pinned stack in
`requirements.txt`: `pandas 3.0.5`, `pyarrow 25.0.1`, `duckdb 1.5.5`,
`numpy 2.5.2`, `databento 0.83.0`, `vectorbtpro` (private git+ssh dependency),
`scikit-learn 1.9.0`, `numba`, plus `optuna` and `hyperopt` for search.

```bash
source .venv/bin/activate          # or prefix commands with .venv/bin/python
uv pip install -r requirements.txt
```

A second venv at `courses/.venv` exists for study notebooks and is deliberately
separate. `courses/` is gitignored.

There is no test suite, linter config, or build step. Scripts are run directly.

---

## Commands

Run from the repo root.

```bash
# CIO Command Center dashboard (frontend scaffold; agent backend is mocked)
streamlit run dashboard/app.py

# Data manifest: path, size, SHA-256, row count, ts range per file.
# Answers "have the bytes changed?"; validate_lake.py answers "is it sane?".
# Both must pass. --verify exits 1 on drift, so it gates a pipeline.
python scripts/generate_manifest.py                 # rebuild manifest.json
python scripts/generate_manifest.py --verify        # detect drift or corruption
python scripts/generate_manifest.py --verify --quick  # size/mtime only, no hashing

# Validate the lake (phases 1-4: inventory, row counts, structure, price sanity)
python scripts/validate_lake.py                     # all symbols
python scripts/validate_lake.py --symbols ES NQ --tf 1d
python scripts/validate_lake.py --quick             # skip price checks

# Rebuild the per-symbol coverage reference (writes reference/futures/coverage*.csv)
python scripts/coverage_summary.py

# Label symbol-years Bull/Bear/Neutral (writes reference/futures/regimes.{parquet,csv})
python scripts/classify_regime.py --threshold 10.0

# Refresh the degraded-sessions calendar (metadata call, free)
export DATABENTO_API_KEY="db-..."
python data_pull/fetch_degraded_days.py

# Download bars. Prints cost first; downloads nothing without --confirm.
python data_pull/pull_futures.py --symbols ES NQ --start 2016-01-01 --end 2026-01-01
python data_pull/pull_futures.py --symbols ES NQ --start 2016-01-01 --end 2026-01-01 --confirm

# Analysis report from a saved BacktestResult
python backtest/report.py \
  --returns /mnt/backtest/artifacts/strat1_returns.parquet \
  --trades  /mnt/backtest/artifacts/strat1_trades.parquet \
  --name "Strat 1" --variants-tested 12 --costs-included yes \
  --out /mnt/backtest/artifacts/strat1_report
```

---

## Architecture & Code Layout

One-way dependency: `agents` → `strategies` → `backtest` → `mdlib` → lake.
`data_pull` writes the lake and is the **only** place that talks to a vendor API.

**`agents/`** — *SCAFFOLDED 2026-08-13. Interfaces and constraints are written;
the agent logic is not.* Every function raises `NotImplementedError` rather than
returning a placeholder — a stub that returns an empty result is how a pipeline
starts reporting numbers nobody generated. Google AI Agent orchestration
scripts (SDKs: `google-genai`, `mcp`):

- `tier1_master.py`: CIO agent managing the global optimization goals.
- `tier2_supervisors.py`: Prop-Firm Compliance (Max DD, Daily Loss Limits) and
  OOS Validation supervisors.
- `tier3_workers.py`: Backtest execution wrappers and ML-generation scripts.
- `system_monitor.py`: Automated circuit breaker tracking RAM and runaway
  execution loops.

**`mdlib/lake.py`** — The single reader. Two entry points over the same bars,
both returning **long format** (`ts, symbol, open, high, low, close, volume`,
UTC). `wide(df, field)` pivots when a column per symbol is needed.

- **`iter_bars(symbols, tf, start, end, ...)`** yields `(symbol, df)` one symbol
  at a time. **This is what backtests use** — it never builds the monolith, and
  a rolling window on a single-symbol frame cannot bleed across instruments.
- **`get_bars(...)`** concatenates those same frames and sorts by
  `(ts, symbol)`. Use it only when a single cross-sectional frame is genuinely
  needed (correlation work, `wide()`); on the full 1m lake it costs ~15 GiB,
  and the result interleaves symbols — see the engine note below.
- **Dual-Dataset Logic:** must support switching between `databento` (16-20 years,
  the research dataset — with its final 3 years held back as the OOS split) and
  `nt8` (a thin cross-feed sanity check, **not** an OOS gate — see Data Layout).
  *Not implemented — see Open Tasks. `get_bars` currently has no source
  parameter and the lake has no source partition.*
- **Only `1m` and `1d` are stored.** `5m/15m/30m/1h/2h/4h` derive from 1m, `1w`
  from 1d. Adding a timeframe means a `DERIVED` entry, not a re-pull.
- **Sunday sessions merge into Monday** for `1d`/`1w` (`session_merge=True`).
  CME opens Sunday 18:00 ET, so a UTC day boundary otherwise creates ~51 thin
  stub "days" a year that silently corrupt every lookback window. The merge lives
  in the reader, not the lake — one function to change if it is wrong.
- Hygiene flags: `exclude_degraded`, `exclude_rolls`, `respect_coverage`.
- Reference lookups (`coverage()`, `degraded_days()`, `roll_dates()`) read from
  `/mnt/backtest/reference/futures/` and are `lru_cache`d.

**`backtest/engine.py`** — Runs the simulations.
`run_backtest(symbols, tf, signal_fn, start, end, cfg)` → `BacktestResult(returns,
trades, equity, breach, stats)`.

- **The engine reads the bars; you pass the strategy, not the signals.**
  `signal_fn(bars) -> (entries, exits)` is called once per symbol with that
  symbol's bars alone. There is deliberately no way to hand it a pre-built
  multi-symbol frame with signals already computed. That signature existed
  until 2026-08-13 and was a trap: `get_bars` returns rows sorted by
  `(ts, symbol)`, so the frame **interleaves instruments**, and a strategy
  doing `close.rolling(200).mean()` over it was averaging across 27 different
  contracts. The signals were the right length and dtype, nothing raised, and
  the equity curve looked plausible — it produced 608,079 trades where the
  correct per-symbol signals give 86,035. Do not reintroduce a frame-in
  entry point.
- **Streaming by default:** bars are read one symbol at a time via
  `mdlib.lake.iter_bars`, so peak RAM tracks the largest single symbol (5.6M
  rows), not the lake. A full-lake run peaks at ~2.9 GiB; building the frame
  first cost 20.5 GiB.
- **Vectorized Execution:** `_simulate` drives `vbt.Portfolio.from_signals`.
  Do NOT add pandas/numpy `for` loops for massive grid searches. Costs go
  in as per-bar arrays (`slippage` as a fraction of price, `fees` as a fraction
  of order value) so they broadcast inside the compiled simulation — see
  `_cost_arrays`, and note slippage is built from tick **size**, not tick
  **value**. The old loop is kept as `_simulate_legacy`, the oracle in
  `tests/test_engine_vbt.py`.
- **Batched:** `_simulate` feeds vectorbt `cfg.chunk_size` bars at a time.
  Boundaries are snapped into the gaps between trades, so the trade list is
  identical at any chunk size. Never chunk by calendar year — a position open
  on 31 December is silently dropped, which flatters results.
- **Fills are at the next bar's open, never the signal bar's close.** Acting on
  the bar that produced the signal is lookahead bias.
- **Costs are mandatory:** slippage and commissions applied at this layer.
- **Prop-firm constraints:** breach logic (trailing drawdown, daily loss limit,
  consistency rules) lives here, passed in as a `BacktestConfig` parameter rather
  than a separate codebase.

**`backtest/specs.py`** — contract multiplier, tick size, commission per symbol.
A wrong multiplier silently scales every P&L figure for that symbol and the
backtest still looks plausible. `verify_specs()` reconciles against Databento's
`definition` schema.

**`backtest/report.py`** — standalone CLI over saved parquet; joins regime labels
and records `--variants-tested` so a Sharpe is never read without knowing how
many variants it was selected from.

**`data_pull/`** — vendor downloaders. The only layer that touches a vendor API.

**`scripts/`** — lake validation, coverage, and regime-labelling utilities.

**`compliance_rules/`** — Prop-firm constraint sets as JSON, one per program
(`fundednext_rapid.json`). Each rule carries its unit, its basis, and an
`enforcement` block recording whether the engine actually checks it. Today only
`max_trailing_drawdown` is enforced; daily loss, profit target, and consistency
are declared but unimplemented, so **an empty `BacktestResult.breach` is not
evidence of compliance**. Note `BacktestConfig.daily_loss_limit` is in dollars
and is currently dead code — nothing reads it.

**`dashboard/`** — Streamlit CIO Command Center (`streamlit run
dashboard/app.py`). Frontend scaffold only: ruleset discovery, the strategy
vault, and error handling are real; the agent backend is mocked and labelled as
such in the UI. Pinned at `streamlit==1.59.1` deliberately — the current release
resolves `pyarrow` down to 24.0.0, and 25.0.1 is what the lake reader is pinned
to.

**`strategies/`** — Signal logic only: take bars, return `(entries, exits)`. No
cost handling, no session logic, no data access. `approved_incubator/` stages
strategies under evaluation; see its README for the required `meta.json`.

- **Dual-Version Mandate:** every strategy must output two versions.
  - **Version A:** pure rule-based baseline (e.g. standard SMA crossover).
  - **Version B:** ML-augmented filter (Scikit-learn/LightGBM to filter signals).
    *Note: LightGBM is not currently in `requirements.txt`.*

---

## Data Layout & Hygiene

```text
/mnt/backtest/raw/futures/                     original vendor DBN, write-once
/mnt/backtest/lake/futures/bars/symbol=X/tf={1m,1d}/year=Y/month=M/*.parquet
/mnt/backtest/reference/futures/               coverage.csv, degraded_days.csv, roll_calendar_<SYM>.json
/mnt/backtest/artifacts/                       backtest and validation outputs
```

- **Databento (In-Sample):** 16-20 years of historical Globex data. Used for
  Phase 1 & 2 optimization. Continuous contracts are NOT back-adjusted (real
  price gaps exist on roll dates).
- **NinjaTrader 8 (cross-feed only, NOT out-of-sample):** broker-specific data
  in `lake/futures_nt8/`. Genuine out-of-sample (OOS) validation is achieved by
  holding back the final 3 years of the Databento dataset. NT8 export data
  (which only contains ~420 daily bars) is fundamentally incompatible for
  cross-source validation due to continuous contract roll discrepancies.
- **No flat files above `year=`:** never write a parquet file above the partition
  level, or DuckDB will duplicate bars. `validate_lake.py` checks for this.
- `raw/` is immutable so the lake can be rebuilt after a parser bug without
  re-downloading. Do not chown/chmod the NFS mount (NFSv3, no idmapping — the
  bogus UID is expected; permissions are server-side). Never put a venv on NFS.
- `.gitignore` excludes `*.parquet` and `*.csv`, so data files never enter git.

---

## Research Discipline

**A strategy is only valid if it survives Phase 3.** Phase 3 is the **held-back
final 3 years of the Databento dataset**, untouched during optimization. If a
strategy performs well in-sample but its Sharpe ratio collapses on the holdout,
it is overfitted and must be discarded.

**The NT8 tree is not that gate.** It holds ~420 daily bars per symbol and is
spliced on NinjaTrader's own roll rules, so a divergence against Databento
mostly measures the difference in continuous contract construction rather than
strategy decay. Treat it as a thin cross-feed sanity check, never as OOS
validation.

**Costs in every test, from the first one.** The ranking of variants changes once
commissions and slippage (default 1 tick each way) are applied.

**Cross-sectional by default.** A daily strategy on ES alone over 16 years is
~100-200 trades — too thin to separate skill from luck. Pass a symbol list.

**Intraday work respects `intraday_start_year`.** Pre-2013 1-minute data is
sparse for ten symbols — volume reconciles exactly against daily bars, but a 30m
bar built from sparse minutes behaves differently.

**Dual-Version Comparison.** Machine learning is only adopted if Version B
significantly outperforms Version A out-of-sample without violating prop-firm
risk limits.

---

## Reference docs

`docs/` holds the design record: `PLAN.md` (overall plan and the Portfolio A
intraday / Portfolio B swing split), `METHODOLOGY.md`, `INFRASTRUCTURE.md`
(servers, NFS, CrossTrade/NT8 deployment path), `STRATEGY_DEVELOPMENT.md`
(10-stage gate pipeline), `STRATEGY_FAMILIES.md`, `PORTFOLIO.md` (correlation
clusters), `VALIDATION_PLAN.md` (phases 1-8, and the gate: no strategy work until
phases 1-7 are clean or every exception is documented).

---

## Open Tasks & Discrepancies (For Claude to Fix Opportunistically)

- ~~`verify_specs()` in `backtest/specs.py` needs to be run to reconcile
  contract multipliers.~~ Done 2026-08-13 (`python -m backtest.specs`). All
  multipliers confirmed; ZT tick corrected 1/128 → 1/256. Still open: 17
  symbols (PL, grains, LE, FX, crypto, micros) have no definition data
  downloaded and remain UNVERIFIED — pull their definitions before backtesting
  them.
- ~~`backtest/engine.py` needs its legacy numpy/pandas loop replaced by Vectorbt
  Pro.~~ Done 2026-08-13; the docstring claim is now true. Still open: the pull
  of full-history `definition` data for the 17 symbols added that day (only 2026
  was downloaded, so a mid-history spec change would be invisible in their
  definition files — `TICK_HISTORY` for FX came from the lake price grid
  instead). SI definitions stop at 2016, CL at 2025-12.
- ~~Establish the `nt8` data directory structure within `/mnt/backtest/lake/`~~
  Done — `lake/futures_nt8/bars/symbol=<SYM>/tf=1d/year=/month=/` exists (27
  symbols, 563 files), written by `data_pull/ingest_nt8.py`. Still open: update
  `mdlib/lake.py` to route the data source flag — `get_bars` has no `source`
  parameter, so the NT8 tree is currently unreachable through the reader.
- ~~`classify_regime.py` has never been run, so
  `reference/futures/regimes.parquet` does not exist.~~ Run 2026-08-14 at
  `--threshold 10.0`: 434 symbol-years across 27 symbols, 2010-2026
  (130 Bull / 81 Bear / 223 Neutral). `report.py`'s regime join now resolves.
  Re-run after any lake rebuild — the labels are derived from daily closes.
- ~~Two 1d partition gaps in the lake (`ZS` 2020-2021, `HO` 2012) surviving only
  in un-partitioned stray files.~~ Resolved 2026-08-14. The 3 missing
  symbol-years were re-pulled from Databento ($0.00), restoring
  `raw/futures/{ZS_ohlcv-1d_2020,ZS_ohlcv-1d_2021,HO_ohlcv-1d_2012}.dbn.zst`
  and their lake partitions (505 and 312 rows, matching exactly what was
  missing). All 24 stray flat files are now deleted and the lake has **zero**
  files above the `year=` partition level. `validate_lake.py` reports no
  structural or price issues.
- ~~Build the `agents/` tier structure described above.~~ Scaffolded 2026-08-13
  (`agents/`, `google-genai==2.18.1` + `mcp==2.0.0` installed). Still open: all
  four modules are interface-only and raise `NotImplementedError`.
- `data_pull/coverage_summary.py` is a superseded copy of
  `scripts/coverage_summary.py` (its own docstring points at `scripts/`); the
  version in `scripts/` has the newer `find_intraday_start` detection. Delete the
  stale copy or make the duplication explicit.
- ~~`save_roll_calendar` in `data_pull/pull_futures.py` overwrote
  `roll_calendar_<SYM>.json` with only the pulled date range, so a narrow pull
  silently discarded the rest of a symbol's roll history.~~ Fixed 2026-08-14;
  the resolved range now widens to cover any calendar already on disk. Six
  truncated calendars (HO, ZS, LE, PL, ZC, ZW) were re-resolved over full
  history; all 27 now span their data. **Re-run the calendar-vs-coverage audit
  after any narrow pull.**
- LightGBM is referenced by the Dual-Version Mandate but is not pinned in
  `requirements.txt`.
