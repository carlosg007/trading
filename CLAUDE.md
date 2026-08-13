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
bypassing the Out-of-Sample (OOS) NT8 data test, or relaxing prop-firm drawdown
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

**`agents/`** — *PLANNED, NOT YET BUILT. This directory does not exist.* Google
AI Agent orchestration scripts:

- `tier1_master.py`: CIO agent managing the global optimization goals.
- `tier2_supervisors.py`: Prop-Firm Compliance (Max DD, Daily Loss Limits) and
  OOS Validation supervisors.
- `tier3_workers.py`: Backtest execution wrappers and ML-generation scripts.
- `system_monitor.py`: Automated circuit breaker tracking RAM and runaway
  execution loops.

**`mdlib/lake.py`** — The single reader. Every strategy goes through
`get_bars(symbols, tf, start, end, ...)`, which returns **long format**
(`ts, symbol, open, high, low, close, volume`, UTC). `wide(df, field)` pivots
when a column per symbol is needed.

- **Dual-Dataset Logic:** must support switching between `databento` (16-20 years
  In-Sample training) and `nt8` (10 years Out-of-Sample stress testing).
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
`run_backtest(bars, entries, exits, cfg)` → `BacktestResult(returns, trades,
equity, breach, stats)`.

- **Vectorized Execution:** `_simulate` is a single `vbt.Portfolio.from_signals`
  call. Do NOT add pandas/numpy `for` loops for massive grid searches. Costs go
  in as per-bar arrays (`slippage` as a fraction of price, `fees` as a fraction
  of order value) so they broadcast inside the compiled simulation — see
  `_cost_arrays`, and note slippage is built from tick **size**, not tick
  **value**. The old loop is kept as `_simulate_legacy`, the oracle in
  `tests/test_engine_vbt.py`.
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

**`strategies/`** — Signal logic only: take bars, return `(entries, exits)`. No
cost handling, no session logic, no data access. Currently empty.

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
- **NinjaTrader 8 (Out-of-Sample):** 10 years of broker-specific data. Used
  strictly for Phase 3 stress testing. *Not yet landed in the lake.*
- **No flat files above `year=`:** never write a parquet file above the partition
  level, or DuckDB will duplicate bars. `validate_lake.py` checks for this.
- `raw/` is immutable so the lake can be rebuilt after a parser bug without
  re-downloading. Do not chown/chmod the NFS mount (NFSv3, no idmapping — the
  bogus UID is expected; permissions are server-side). Never put a venv on NFS.
- `.gitignore` excludes `*.parquet` and `*.csv`, so data files never enter git.

---

## Research Discipline

**A strategy is only valid if it survives Phase 3.** If a strategy performs well
on Databento data but its Sharpe ratio collapses on NT8 data, it is overfitted
and must be discarded.

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
- Establish the `nt8` data directory structure within `/mnt/backtest/lake/` and
  update `mdlib/lake.py` to route the data source flag seamlessly.
- Build the `agents/` tier structure described above.
- `data_pull/coverage_summary.py` is a superseded copy of
  `scripts/coverage_summary.py` (its own docstring points at `scripts/`); the
  version in `scripts/` has the newer `find_intraday_start` detection. Delete the
  stale copy or make the duplication explicit.
- `classify_regime.py` has never been run — `reference/futures/regimes.parquet`
  does not exist, so `report.py`'s regime join currently finds nothing.
- LightGBM is referenced by the Dual-Version Mandate but is not pinned in
  `requirements.txt`.
