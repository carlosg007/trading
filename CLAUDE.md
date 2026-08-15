# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

An autonomous, high-performance quantitative research environment focused on
**Pure Alpha Discovery** across continuous futures markets. Data lives on a
hard-mounted NFS at `/mnt/backtest` (not in the repo); code lives here
(`~/src/trading`).

The mission of this server is to discover, optimize, and validate statistically
robust market edges — maximising Sharpe, out-of-sample stability, and
walk-forward efficiency — across a 16-to-20-year continuous Databento futures
data lake.

**The failure mode this project is built to avoid is an overfitted backtest that
looks right and fails in live markets.** Most of the conventions below exist to
combat curve-fitting and silent data corruption.

---

## Architectural Separation of Concerns

The overhaul of 2026-08-15 split alpha discovery from account governance. They
are different problems, they fail differently, and mixing them was producing
strategies tuned to a funding program rather than to a market.

**1. Ubuntu Research Server (`backtest`) — this repo.** Pure mathematical alpha
generation and deterministic statistical stress testing. Research is **NOT**
over-constrained with artificial prop-firm intraday balance math: no 40% daily
profit consistency caps, no moving trailing-drawdown rules, no per-account
sizing. A strategy is judged on edge, not on whether a particular funding
program would have tolerated its equity path.

**2. CrossTrade NAM (Windows execution bridge) — outside this repo.** Real-time
broker-level account governance: contract sizing, daily loss limits, trailing
drawdown, consistency rules, and prop-firm challenge state. These are execution
concerns, enforced against a live account balance, and they belong where the
account actually lives.

**Why the split matters when you are editing code here.** Do not add prop-firm
balance math to the research path, and do not treat its absence as a bug. If a
constraint depends on a live account balance or a funding program's rulebook, it
belongs to CrossTrade, not to `backtest/`.

**What this leaves behind in the tree.** The decoupling is architectural and the
code has not been stripped to match it. Still present, and now *legacy for
research purposes*:

- `BacktestConfig.trailing_drawdown_pct` / `.daily_loss_limit`, and
  `check_trailing_drawdown()` in `backtest/engine.py`. Usable for a descriptive
  read of an equity path; **not** a research gate. `daily_loss_limit` remains
  dead code — nothing reads it.
- `compliance_rules/*.json` and `evaluate_compliance()` in
  `agents/tier2_supervisors.py`. Retained as the declarative rulebook that
  CrossTrade governance is specified from.
- An empty `BacktestResult.breach` was never evidence of compliance and is even
  less so now — only `max_trailing_drawdown` was ever enforced.

Do not delete these as part of an unrelated task; their removal or migration is
its own scoped change.

---

## Working Agreement & AI Boundaries

**Verify before asserting.** Do not report that something works because the code
looks correct. Run it in the terminal. For anything touching P&L, costs, fills,
or session boundaries, construct a test case where the answer is known and
verify the output matches.

**The AI Division of Labor:**

- **Claude Code (you)** — local terminal expert and senior quantitative
  engineer. Deep multi-file editing, git, package management via `uv`, AST
  verification, fixing runtime errors, optimising vectorized Vectorbt Pro /
  Numba loops.
- **Google AI Synthesizer (`gemini-3.1-pro-preview` via `google-genai`)** — **single-shot**
  strategy synthesis from a natural-language research hypothesis. One prompt,
  one module, no agentic loop and no iterative self-correction. It writes
  vectorized signal logic and nothing else: it does not choose symbols, size
  positions, or interpret results. Entry point:
  `agents/tier1_master.synthesize_strategy_code`. Needs `GEMINI_API_KEY`;
  without it a campaign falls back to the placeholder template and says so in
  every event and in the verdict.
- **Deterministic Compute Engine (Python / Numba / Vectorbt Pro / SciPy)** —
  calculates 100% of the mathematical metrics: Sharpe, Sortino, Calmar, profit
  factor, WFO, Monte Carlo bootstrap. **LLMs never do math.** If a number
  reaches a human, a deterministic function produced it.
- **CrossTrade NAM (Windows)** — real-time broker-level account governance,
  contract sizing, and prop-firm challenge rules. See the separation above.

**Model-generated code is validated before it is imported.** Synthesized modules
pass an AST check before execution — no file, network, or OS access, no
`eval`/`exec`/`__import__`/`open`. Never bypass that gate to "just try" a
generated strategy.

**Commit before large changes.** `git status` first. The repo is the only backup
for code.

**Push back on requests that compromise rigor.** Removing transaction costs,
bypassing the 3-year OOS holdout, or introducing lookahead bias are critical
flaws to reject. Being useful here means being a skeptic.

---

## Token Efficiency & Context Management

- **Be concise.** Prefer executing commands and writing code over long
  conversational explanations.
- **Targeted code edits.** Surgical edits only. Do not dump entire unchanged
  files into context.
- **Never read raw data files into context.** Do not `cat`, `head`, or ingest
  `.parquet`, `.csv`, or large logs from `/mnt/backtest`. Use small Python
  diagnostic scripts to inspect schemas, shapes, or sample rows.
- **Search before reading.** Use `grep`/`rg` to locate classes, functions, or
  variables instead of loading whole files.
- **No unprompted refactoring.** Keep changes strictly scoped to the task.

---

## Library & Execution Discipline

**The Vectorbt Pro guardrail.** This project uses `vectorbtpro` (private, paid).
Its API differs significantly from open-source `vectorbt`. **Do not hallucinate
methods.** Run a quick terminal inspection (`dir()`, `help()`, docstrings)
before writing an implementation.

> Note: open-source `vectorbt` was uninstalled on 2026-08-15 and unpinned from
> `requirements.txt`, so `import vectorbt` now fails loudly instead of resolving
> to a different library. Always `import vectorbtpro as vbt`. The footgun is not
> dead: `riskfolio-lib` (pinned, imported nowhere here) declares `vectorbt` as a
> dependency, so a reinstall can pull it back — check `uv pip list | grep -i
> vectorbt` after one. `ALLOWED_IMPORTS` in `agents/tier3_workers.py` also
> rejects it in model-generated strategy code.

**Atomic git commits.** Granular and single-purpose. A data fix and a loop
optimisation are two commits, so a script that breaks the server can be rolled
back without losing the data fix.

**Test-driven edits.** When refactoring critical engine components (e.g.
`backtest/engine.py`), do not modify the main file first. Prove the array logic
on a small slice with a minimal reproducible script, then edit.

**Vectorized array signatures.** Strategy modules must conform to the contract
the engine actually calls — see `strategies/` below. Do not invent a different
one.

---

## Environment & Infrastructure

Dedicated Ubuntu 26.04 LTS Proxmox VM sized for heavy Vectorbt Pro grid searches
(8-16 vCPUs, 32-64 GB RAM).

**Package management is `uv`, exclusively.** Python 3.13.14 in a single isolated
production venv at `~/src/trading/.venv` (uv 0.12.2). Do not use `pip install`
directly, do not create secondary venvs in the repo, and **never put a venv on
the NFS mount**.

```bash
source .venv/bin/activate          # or prefix commands with .venv/bin/python
uv pip install -r requirements.txt
uv pip list
```

Pinned stack in `requirements.txt`: `pandas 3.0.5`, `numpy 2.5.2`,
`pyarrow 25.0.1`, `duckdb 1.5.5`, `numba 0.67.0rc1`, `scipy 1.18.0`,
`databento 0.83.0`, `scikit-learn 1.9.0`, `streamlit 1.59.1`,
`google-genai 2.18.1`, `mcp 2.0.0`, plus `optuna` and `hyperopt` for search, and
`vectorbtpro` as a private `git+ssh` dependency.

`streamlit` is pinned deliberately — the current release resolves `pyarrow` down
to 24.0.0, and 25.0.1 is what the lake reader is pinned to.

There is no linter config or build step. Scripts are run directly.

---

## Commands

Run from the repo root.

```bash
# CIO Command Center dashboard (frontend scaffold; agent backend is mocked)
streamlit run dashboard/app.py

# Tests. No pytest config - each is a script that exits non-zero on failure.
# Eight suites, 468 checks. The first four need neither the lake nor a network.
python tests/test_tier1.py              # intent routing, vault, synthesis errors
python tests/test_tier2.py              # compliance, robustness, lifecycle
python tests/test_tier3_workers.py      # worker tools, metrics, RAM ceiling
python tests/test_dispatcher.py         # CrossTrade payload, transport, fill logs
python tests/test_clean_signals.py      # per-symbol signals vs the interleaved trap
python tests/test_streaming_lake.py     # iter_bars and the streaming engine
python tests/test_engine_batching.py    # chunked == unchunked, trade for trade
python tests/test_engine_vbt.py         # vectorbt P&L == the legacy loop oracle

# Data manifest: path, size, SHA-256, row count, ts range per file.
# Answers "have the bytes changed?"; validate_lake.py answers "is it sane?".
# Both must pass. --verify exits 1 on drift, so it gates a pipeline.
python scripts/generate_manifest.py                   # rebuild manifest.json
python scripts/generate_manifest.py --verify          # detect drift or corruption
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
`live/dispatcher.py` is the only place that sends an order anywhere.

**`agents/`** — *SCAFFOLDED 2026-08-13. Interfaces and constraints are written;
much of the agent logic is not.* Every unimplemented function raises
`NotImplementedError` rather than returning a placeholder — a stub that returns
an empty result is how a pipeline starts reporting numbers nobody generated.

- `tier1_master.py`: CIO agent and the synthesis entry point.
  **Implemented:** `run_campaign` (generator; routes a prompt to vault /
  conversational / research and yields progress), `classify_intent`,
  `synthesize_strategy_code` (single-shot Gemini via `google-genai`),
  `build_client`. Still scaffold: `propose_goals`, `prioritise`, `review`,
  `main`. API key from `GEMINI_API_KEY` / `GOOGLE_API_KEY` /
  `GOOGLE_GENAI_API_KEY`.
- `tier2_supervisors.py`: OOS validation and robustness.
  **Implemented:** `evaluate_robustness` (WFO >= 0.50, MC drawdown within the
  ruleset limit), `evaluate_lifecycle_state` (ACTIVE / PAUSED /
  DECOMMISSIONED), `evaluate_compliance` (now the CrossTrade rulebook
  specification, not a research gate — see the separation of concerns). Still
  scaffold: `PropFirmSupervisor`, `OOSValidationSupervisor`, `review_all`.
- `tier3_workers.py`: backtest execution and quantitative testing.
  **Implemented:** `run_strategy_backtest`, `run_walk_forward_analysis`,
  `run_parameter_sensitivity`, `run_monte_carlo_simulation`,
  `generate_strategy_boilerplate`, `load_strategy`. Still scaffold:
  `run_variant`, `run_dual_version`, `generate_ml_filter`, `main`.
- `system_monitor.py`: circuit breaker tracking RAM and runaway execution loops.

**`mdlib/lake.py`** — The single reader. Two entry points over the same bars,
both returning **long format** (`ts, symbol, open, high, low, close, volume`,
UTC). `wide(df, field)` pivots when a column per symbol is needed.

- **`iter_bars(symbols, tf, start, end, ...)`** yields `(symbol, df)` one symbol
  at a time. **This is what backtests use** — it never builds the monolith, and
  a rolling window on a single-symbol frame cannot bleed across instruments.
- **`get_bars(...)`** concatenates those frames and sorts by `(ts, symbol)`. Use
  it only when a single cross-sectional frame is genuinely needed (correlation
  work, `wide()`); on the full 1m lake it costs ~15 GiB, and the result
  interleaves symbols — see the engine note below.
- **Only `1m` and `1d` are stored.** `5m/15m/30m/1h/2h/4h` derive from 1m, `1w`
  from 1d. Adding a timeframe means a `DERIVED` entry, not a re-pull.
- **Sunday sessions merge into Monday** for `1d`/`1w` (`session_merge=True`).
  CME opens Sunday 18:00 ET, so a UTC day boundary otherwise creates ~51 thin
  stub "days" a year that silently corrupt every lookback window. The merge
  lives in the reader, not the lake — one function to change if it is wrong.
- Hygiene flags: `exclude_degraded`, `exclude_rolls`, `respect_coverage`.
- Reference lookups (`coverage()`, `degraded_days()`, `roll_dates()`) read from
  `/mnt/backtest/reference/futures/` and are `lru_cache`d.
- **Dual-dataset routing is not implemented** — `get_bars` has no `source`
  parameter, so the NT8 tree is unreachable through the reader. See Open Tasks.

**`backtest/engine.py`** — Runs the simulations.
`run_backtest(symbols, tf, signal_fn, start, end, cfg)` → `BacktestResult(returns,
trades, equity, breach, stats)`.

- **The engine reads the bars; you pass the strategy, not the signals.**
  `signal_fn(bars) -> (entries, exits)` is called once per symbol with that
  symbol's bars alone. There is deliberately no way to hand it a pre-built
  multi-symbol frame with signals already computed. That signature existed until
  2026-08-13 and was a trap: `get_bars` returns rows sorted by `(ts, symbol)`,
  so the frame **interleaves instruments**, and a strategy doing
  `close.rolling(200).mean()` over it was averaging across 27 different
  contracts. The signals were the right length and dtype, nothing raised, and
  the equity curve looked plausible — it produced 608,079 trades where the
  correct per-symbol signals give 86,035. Do not reintroduce a frame-in entry
  point.
- **Streaming by default:** bars are read one symbol at a time via
  `mdlib.lake.iter_bars`, so peak RAM tracks the largest single symbol (5.6M
  rows), not the lake. A full-lake run peaks at ~2.9 GiB; building the frame
  first cost 20.5 GiB.
- **Vectorized execution:** `_simulate` drives `vbt.Portfolio.from_signals`. Do
  NOT add pandas/numpy `for` loops for massive grid searches. Costs go in as
  per-bar arrays (`slippage` as a fraction of price, `fees` as a fraction of
  order value) so they broadcast inside the compiled simulation — see
  `_cost_arrays`, and note slippage is built from tick **size**, not tick
  **value**. The old loop is kept as `_simulate_legacy`, the oracle in
  `tests/test_engine_vbt.py`.
- **Batched:** `_simulate` feeds vectorbt `cfg.chunk_size` bars at a time.
  Boundaries are snapped into the gaps between trades, so the trade list is
  identical at any chunk size. Never chunk by calendar year — a position open on
  31 December is silently dropped, which flatters results.
- **Fills are at the next bar's open, never the signal bar's close.** Acting on
  the bar that produced the signal is lookahead bias.
- **Costs are mandatory:** slippage and commissions applied at this layer.
- **Numba:** `clean_signals` compiles its two-state machine via `njit`
  (`cache=True, nogil=True`) and falls back to the interpreted loop when numba
  is absent. `tests/test_clean_signals.py` checks it against a pure-Python
  oracle.

**`backtest/specs.py`** — contract multiplier, tick size, commission per symbol.
A wrong multiplier silently scales every P&L figure for that symbol and the
backtest still looks plausible. `verify_specs()` reconciles against Databento's
`definition` schema.

**`backtest/report.py`** — standalone CLI over saved parquet; joins regime labels
and records `--variants-tested` so a Sharpe is never read without knowing how
many variants it was selected from.

**`data_pull/`** — vendor downloaders. The only layer that touches a vendor API.

**`scripts/`** — lake validation, coverage, and regime-labelling utilities.

**`compliance_rules/`** — prop-firm constraint sets as JSON, one per program
(`fundednext_rapid.json`). Each rule carries its unit, its basis, and an
`enforcement` block. Post-overhaul this is the **specification handed to
CrossTrade NAM**, not a research gate.

**`dashboard/`** — Streamlit CIO Command Center. Frontend scaffold only: ruleset
discovery, the strategy vault, and error handling are real; the agent backend is
mocked and labelled as such in the UI.

**`live/`** — the Windows incubator bridge to CrossTrade NAM. `dispatcher.py` is
the **only** module that sends an order anywhere: `format_crosstrade_payload`
(validated — an unknown action, a non-positive or fractional quantity, and
price-bearing order types all raise rather than being forwarded),
`send_execution_signal` (POST with a hard 2.0s timeout; failures are RETURNED as
`ok=False` result dicts with latency and reason, never raised, so the attempt is
always on the record), and `evaluate_incubator_sync` (parses NT8 fill logs from
`/mnt/backtest/artifacts/incubator_logs/`, reporting realised slippage in
**ticks** — tick size read from `backtest/specs.py`, never assumed — and the fill
rate). Connection profile in `live/config.json`; the shipped webhook URL is a
placeholder and the dispatcher refuses to POST to it.

**`strategies/`** — Signal logic only: take bars, return `(entries, exits)`. No
cost handling, no session logic, no data access. `approved_incubator/` stages
strategies under evaluation; see its README for the required `meta.json`.

The loader (`agents/tier3_workers.load_strategy`) accepts either form:

```python
make_signal_fn(**params) -> signal_fn     # preferred when parameterised
signal_fn(bars) -> (entries, exits)       # when it takes none
```

`bars` is ONE symbol's DataFrame, oldest to newest. Return two boolean Series
aligned to it. A module may declare `TIMEFRAME`, `SYMBOLS`, and `DEFAULT_PARAMS`.

- **Dual-Version Mandate:** every strategy outputs two versions. **Version A** is
  a pure rule-based baseline (e.g. an SMA crossover); **Version B** adds an ML
  filter over the same signals. ML is adopted only if B beats A out-of-sample.
  *Note: LightGBM is not currently in `requirements.txt`.*

---

## Data Layout & Hygiene

```text
/mnt/backtest/raw/futures/                     original vendor DBN, write-once
/mnt/backtest/lake/futures/bars/symbol=X/tf={1m,1d}/year=Y/month=M/*.parquet
/mnt/backtest/reference/futures/               coverage.csv, degraded_days.csv, roll_calendar_<SYM>.json
/mnt/backtest/artifacts/                       backtest and validation outputs
```

- **Databento (in-sample):** 16-20 years of Globex history, used for Phase 1 & 2
  optimization. Continuous contracts are NOT back-adjusted — real price gaps
  exist on roll dates.
- **NinjaTrader 8 (cross-feed only, NOT out-of-sample):** broker data in
  `lake/futures_nt8/`. Genuine OOS validation comes from holding back the final
  3 years of the Databento dataset. NT8 exports hold only ~420 daily bars and
  are spliced on NinjaTrader's own roll rules, so they are fundamentally
  incompatible for cross-source validation.
- **No flat files above `year=`:** never write a parquet file above the
  partition level, or DuckDB will duplicate bars. `validate_lake.py` checks this.
- `raw/` is immutable so the lake can be rebuilt after a parser bug without
  re-downloading. Do not chown/chmod the NFS mount (NFSv3, no idmapping — the
  bogus UID is expected; permissions are server-side).
- `.gitignore` excludes `*.parquet` and `*.csv`, so data files never enter git.

---

## Research Discipline

**A strategy is only valid if it survives Phase 3** — the held-back final 3 years
of the Databento dataset, untouched during optimization. If it performs well
in-sample but its Sharpe collapses on the holdout, it is overfitted and must be
discarded.

**The NT8 tree is not that gate.** A divergence against Databento mostly measures
the difference in continuous-contract construction. Thin cross-feed sanity check,
never OOS validation.

**Costs in every test, from the first one.** Variant rankings change once
commissions and slippage (default 1 tick each way) are applied.

**Cross-sectional by default.** A daily strategy on ES alone over 16 years is
~100-200 trades — too thin to separate skill from luck. Pass a symbol list.

**Intraday work respects `intraday_start_year`.** Pre-2013 1-minute data is
sparse for ten symbols — volume reconciles exactly against daily bars, but a 30m
bar built from sparse minutes behaves differently.

**Report the search.** Carry `variants_tested` into every result. A Sharpe read
without knowing how many variants produced it is not a measurement.

---

## Reference docs

`docs/` holds the design record: `PLAN.md` (overall plan, Portfolio A intraday /
Portfolio B swing split), `METHODOLOGY.md`, `INFRASTRUCTURE.md` (servers, NFS,
CrossTrade/NT8 deployment path), `STRATEGY_DEVELOPMENT.md` (10-stage gate
pipeline), `STRATEGY_FAMILIES.md`, `PORTFOLIO.md` (correlation clusters),
`VALIDATION_PLAN.md` (phases 1-8), `PHASE3_VERIFICATION.md`.

---

## Open Tasks & Discrepancies (For Claude to Fix Opportunistically)

- ~~The Gemini synthesis prompt emits a signature the engine cannot call.~~
  **Fixed 2026-08-15.** `SYNTHESIS_SYSTEM_PROMPT` in `agents/tier1_master.py`
  now specifies `def signal_fn(bars: pd.DataFrame, **params) ->
  tuple[pd.Series, pd.Series]`, which is what the engine
  (`backtest/engine.py`) and the loader (`agents/tier3_workers.load_strategy`)
  actually call. `ENGINE_ADAPTER` no longer reshapes arguments; it binds params
  and forces the return to boolean. **This is the one contract** — a strategy
  module that takes unpacked arrays is now wrong, not merely unconventional.
- `mdlib/lake.py` needs the `source` parameter to route between `databento` and
  `nt8`; the NT8 tree (27 symbols, 563 files, written by
  `data_pull/ingest_nt8.py`) is currently unreachable through the reader.
- 17 symbols (PL, grains, LE, FX, crypto, micros) have no definition data
  downloaded and remain UNVERIFIED in `backtest/specs.py` — pull their
  definitions before backtesting them. Only 2026 was downloaded for the batch
  added 2026-08-13, so a mid-history spec change would be invisible; SI
  definitions stop at 2016, CL at 2025-12.
- `data_pull/coverage_summary.py` is a superseded copy of
  `scripts/coverage_summary.py` (its own docstring points at `scripts/`); the
  `scripts/` version has the newer `find_intraday_start` detection. Delete the
  stale copy or make the duplication explicit.
- **Re-run the calendar-vs-coverage audit after any narrow pull** of
  `data_pull/pull_futures.py`.
- LightGBM is referenced by the Dual-Version Mandate but is not pinned in
  `requirements.txt`.
- `agents/` remains largely scaffold — the unimplemented functions listed above
  raise `NotImplementedError`.
- Prop-firm decoupling is architectural only; the legacy surfaces listed under
  Separation of Concerns still exist in `backtest/` and `agents/tier2`.
