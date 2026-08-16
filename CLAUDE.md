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

Two shell aliases live in `~/.bashrc`. Both name `.venv/bin/python3` explicitly
rather than `python3`, because outside an activated venv `python3` is
`/usr/bin/python3` and has neither pandas nor vectorbtpro:

```bash
bt-run     # = .venv/bin/python3 ~/src/trading/backtest/run.py
bt-status  # = .venv/bin/python3 ~/src/trading/backtest/status.py
```

```bash
# CIO Command Center dashboard (frontend scaffold; agent backend is mocked)
streamlit run dashboard/app.py

# Tests. No pytest config - each is a script that exits non-zero on failure.
# Ten suites. The first five, and test_batch_runner, need neither the lake nor
# a network (test_batch_runner's --symbols checks skip, loudly, without it).
# test_report_gates.py shells out to `node` for the trade inspector's own
# checks and skips them, loudly, when node is absent.
python tests/test_tier1.py              # intent routing, vault, synthesis errors
python tests/test_tier2.py              # compliance, robustness, lifecycle
python tests/test_tier3_workers.py      # worker tools, metrics, RAM ceiling
python tests/test_dispatcher.py         # CrossTrade payload, transport, fill logs
python tests/test_report_gates.py       # acceptance gates, scorecard, HTML, promotion
python tests/test_clean_signals.py      # per-symbol signals vs the interleaved trap
python tests/test_streaming_lake.py     # iter_bars and the streaming engine
python tests/test_engine_batching.py    # chunked == unchunked, trade for trade
python tests/test_engine_vbt.py         # vectorbt P&L == the legacy loop oracle
python tests/test_batch_runner.py       # scan == engine, leaderboard, job tracker

# Dual-version integration on real bars (needs the lake). Pin the thread count:
# the ML filter refits per completed trade on a few dozen rows, and on a
# 16-core box each fit's thread pool costs far more than the fit.
OMP_NUM_THREADS=1 python test_dual_version.py

# The multi-asset batch runner. One INDEPENDENT simulation per contract, on
# that contract's own specs - nothing is ever pooled into a blended portfolio.
# Steps 1-4 of the workflow below, per symbol; it promotes nothing. Pins the
# thread count itself. Symbols default to the module's SYMBOLS, timeframe to
# its TIMEFRAME and then 15m. Aliased to `bt-run`.
python3 backtest/run.py --strat sma_crossover --symbols NQ,ES --tf 1d
python3 backtest/run.py --strat sma_crossover --symbols ALL --tf 15m \
  --start 2018-01-01 --end 2023-12-31 --scan --bg
python3 backtest/run.py --strat sma_crossover --symbols NQ --ml   # also Version B

#   --symbols  one (NQ), a list (NQ,ES,CL), or ALL for all 27 lake symbols
#   --scan     sweep the module's PARAM_GRID per symbol (see the scanner below)
#   --ml       also run Version B. OFF by default: the classifier refits once
#              per completed trade, and 27 symbols of that is hours. A skipped
#              Version B reports as NOT RUN everywhere, never as a zero.
#   --bg       detach and run as a background daemon; follow it with bt-status

# What the running batch is doing right now. Aliased to `bt-status`.
python3 backtest/status.py              # one snapshot
python3 backtest/status.py --watch 5    # redraw until the job leaves RUNNING

# Promote a version into the incubator and commit it (see the workflow below).
# The batch writes one snapshot per symbol, so name the contract the promotion
# decision rests on.
python3 backtest/promote.py --strat sma_crossover --version A \
  --source strategies/experimental/sma_crossover.py \
  --metrics /mnt/backtest/artifacts/sma_crossover_<ts>/dual_metrics_NQ.json

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
many variants it was selected from. Also holds the acceptance gates
(`GATE_THRESHOLDS`, `audit_acceptance_gates`) and the terminal
`print_dual_scorecard` — see the dual-version workflow below.

**`backtest/report_html.py`** — the browser tear sheet.
`generate_html_report(bars, result, metrics, gate_audit, out_path, strat_name,
strat_description, version_label, indicators)` writes one self-contained
dark-themed HTML file: gate badges, alpha metrics, the Plotly equity and
underwater curves, the strategy logic card, the monthly heatmap, a
searchable/sortable trade log, and the click-to-inspect candlestick modal.
`write_dual_reports(dual, bars, ...)` emits both versions plus
`dual_metrics.json`.

- **The strategy logic card is plain English**, written for whoever is deciding
  whether to trade the thing: core concept, entry trigger, exit rule, risk
  management (stop, target, session flatten) and execution (fill price,
  slippage, commission, size). No function names, no array shapes. The concept
  and the entry/exit sentences are DECLARED by the strategy module's `LOGIC`
  block with the run's own parameters filled in and carried through
  `metrics["meta"]["logic"]` — never inferred from the signal arrays, because a
  description guessed from the trades is a guess printed as a fact. A module
  that declares none gets "not declared", plus the first paragraph of its
  docstring.
- **The inspector draws the strategy's own indicator lines** over the candles,
  passed in as `indicators={name: series}` (or a DataFrame), each the full
  length of `bars`. They come from the module's `indicators()` hook so the line
  a reader watches cross is the array the entry was taken from — recomputing
  them in the report would let the line and the signal disagree, with nothing
  raising. A series that is not the frame's length is DROPPED, never reindexed;
  warm-up NaN renders as a gap.

- **Everything is inlined** — Plotly, the bar windows the inspector draws, the
  CSS and the JS. ~5 MB a file, and worth it: a report is evidence, and a CDN
  tag renders an empty rectangle the first time someone opens it offline.
- **The trade log caps at `MAX_TRADE_ROWS`** (2,000) and says so on the page
  when the cap bites. A 535k-row table opens in no browser, and a silently
  truncated log reads as a complete one. The inspector caps with it.
- **Fees and slippage are split** out of the engine's single `costs` figure by
  subtracting the deterministic commission (`_cost_split`). When the remainder
  would be negative the split is abandoned and the page says so, rather than
  printing an impossible column.
- **Chart colors are validated** (see the module docstring). The equity/
  drawdown pair clears CVD separation; the green/red entry-exit markers do not,
  so they carry shape and an IN/OUT label as well. Indicator lines avoid the
  blue/red the candles own (amber, violet, teal, slate) and carry a dash
  pattern and a legend label, so "which one is the fast mean" survives
  greyscale.
- **`tests/inspector_dom_test.js`** runs the page's own JavaScript against a
  stub DOM under node — search, sort, row click, the window slice, the
  indicator overlays, Escape. Markup assertions cannot catch an off-by-one that
  draws the wrong trade, or a line sliced one bar out of step with the candles
  under it.

**`backtest/run.py`** — the multi-asset batch runner. Resolves a strategy by
name, then loops: reads ONE symbol through `iter_bars`, optionally sweeps its
`PARAM_GRID`, runs Version A (and Version B with `--ml`) under identical costs,
prints the scorecard and the gate audit, writes that symbol's tear sheets, and
appends its leaderboard row — then the next symbol. It stops at the four-choice
menu, promotes nothing, and has no code path that could. Gates 2 and 3 are NOT
EVALUATED here; a walk-forward, a bootstrap and the 3-year holdout are separate
runs.

- **Symbols are never blended.** Each contract is its own simulation on its own
  `backtest/specs.py` multiplier, tick size and commission. A pooled frame
  would have to pick one multiplier, and the concatenated frame `get_bars`
  returns interleaves instruments — a `rolling(200)` over it averages 27
  contracts and the equity curve still looks plausible. The runner reads
  through `iter_bars` and asserts the frame it got carries one symbol.
- **Artifacts land in `/mnt/backtest/artifacts/<strat_name>_<timestamp>/`**:
  `report_<SYMBOL>_version_a.html` (and `_version_b` with `--ml`),
  `dual_metrics_<SYMBOL>.json`, `scan_<SYMBOL>.csv` with `--scan`,
  `summary_leaderboard.csv`, `job.json`, and `run.log` with `--bg`.
- **`summary_leaderboard.csv`** carries **one row per symbol**, Version A and
  Version B side by side, in this column order:

  ```text
  timestamp strategy symbol tf params
  sharpe_a pf_a win_rate_a max_dd_a trades_a gate1_a
  sharpe_b gate1_b selected_version html_report
  status error variants_tested scan_selection
  ```

  `timestamp` is the run's stamp, identical on every row and equal to the
  directory's, so leaderboards from several runs concatenate and group cleanly.
  `sharpe_b` and `gate1_b` are **blank** when `--ml` was off, not `NOT
  EVALUATED` — a version that never ran did not reach a gate, and one token for
  both would make a skipped B look like a B whose robustness run is merely
  outstanding. `html_report` names the selected version's tear sheet; its
  sibling is one substitution away in the same directory.

  **`selected_version` records which version led IN-SAMPLE and nothing more.**
  It is not a promotion, not a gate result, and not the Dual-Version Mandate's
  verdict — that needs B to beat A out-of-sample, which no run this script
  performs can establish. Values are `A`, `B`, `A (B not run)` and
  `NONE (no measurable Sharpe)`; a tie leaves A selected, because B has to beat
  A and a tie is not a beat. `promote.py` still refuses a version whose gate
  audit is not PASS, whatever this column says.

  The last four columns sit **after** the declared schema rather than inside
  it, so a reader slicing the first fifteen gets exactly the specified file.
  They are there because dropping them would make the CSV lie rather than
  merely make it shorter: without `status`/`error` a symbol that failed to load
  reads as a strategy that produced nothing, and without `variants_tested` a
  `--scan` Sharpe is the best of an unstated N.

  It is REWRITTEN from scratch after every symbol, sorted by `sharpe_a` with
  errored rows last, so a batch killed at symbol 14 leaves a complete
  leaderboard of 14 rather than a half-written line. Sorted on A rather than on
  the selected version because A is the column every row has — ranking on a
  mixture would put a symbol whose ML filter happened to run above one where it
  did not, which is a fact about the flags rather than about the market.
- **One bad contract does not end the batch.** A missing spec, an empty slice
  of the lake or a strategy that raises on one symbol's data is recorded as an
  `ERROR` row and the loop moves on. The process exits 1 if anything failed.
- **`--ml` is opt-in.** Version B refits its classifier once per completed
  trade; across many symbols that is hours. A skipped Version B is reported as
  NOT RUN in the scorecard, the snapshot and the leaderboard — never as a
  Version B that scored nothing, because a comparison that was not made is not
  one the baseline won.

**`backtest/scan.py`** — the vectorbt-native parameter sweep behind `--scan`. A
strategy declares `PARAM_GRID = {"ema_period": [15, 20, 30], ...}`; every
combination becomes a COLUMN of a single `vbt.Portfolio.from_signals` call, so a
27-cell grid costs roughly one backtest rather than 27. The winner is the
highest Sharpe **among the combinations whose Gate 1 audit is PASS**.

- **The scanner's numbers are the engine's numbers.** Per-column trade lists are
  built from the engine's own `_cost_arrays` and `_assemble_result`, with the
  same one-bar signal shift and next-bar-open fill, so a column's metrics are
  what `_simulate` would have produced for those parameters alone —
  `tests/test_batch_runner.py` asserts it trade for trade. Fees are per column
  because the fee fraction is quoted against the FILL price, and a bar where
  one column enters while another exits has two different fills.
- **Columns are batched** (`MAX_CELLS`) so peak RAM tracks bars × columns-per-
  batch. `_simulate` batches bars for the same reason; this batches columns.
- **A combination the strategy rejects is counted, not dropped.**
  `sma_crossover` raises on `fast >= slow`, and a sweep that silently skipped
  those would report a 9-cell search that tested 6.
- **When nothing clears Gate 1** the highest Sharpe overall is returned with
  `selection` set to `HIGHEST SHARPE · NO COMBINATION CLEARED GATE 1`. That
  string goes into the leaderboard verbatim — the run still has to report
  something, and labelling it is the alternative to inventing a pass.
- **The search is part of the result.** The number of combinations evaluated is
  written to `cfg.variants_tested` and carried onto every report and every
  leaderboard row. A swept Sharpe is selected in-sample on the bars it is
  scored on; read without N it is not a measurement.

**`backtest/status.py`** — the job tracker, both halves. `JobTracker` is what
the runner writes (atomically: temp file, then `os.replace`); `main()` is what
`bt-status` reads. `active_job.json` is a single well-known path
(`$BT_ACTIVE_JOB`, default `/mnt/backtest/artifacts/active_job.json`) holding
job id, strategy, symbol counts, the symbol currently evaluating, start time and
a mini-scorecard per completed symbol. It is a progress indicator, not evidence
— each batch overwrites it, and the copy that stays with the reports is
`job.json` in the run's own directory. A job whose state is RUNNING but whose
PID is gone reads as **STALE**, because a bar frozen at 12/27 looks identical
whether the run is slow or dead.

**`backtest/promote.py`** — promotes one version into
`strategies/approved_incubator/<strat>/` and commits it. See the workflow below.

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

### Every new strategy implements all four

The loader tolerates a module that declares none of the following — several
pre-existing ones do, and the tolerance is what keeps them loadable. It is not
permission to write another. **Any strategy Claude creates carries all four**,
because each one closes a specific way a result goes wrong silently:

```python
def signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, pd.Series]: ...
def indicators(bars: pd.DataFrame, **params) -> dict[str, pd.Series]: ...
LOGIC = {"concept": ..., "entry": ..., "exit": ...}
PARAM_GRID = {"ema_period": [15, 20, 30], "atr_mult": [2.0, 2.5, 3.0]}
```

| Declaration | Without it |
|---|---|
| `signal_fn` | The engine cannot call the module at all. This is **the** contract — a module taking unpacked arrays is wrong, not merely unconventional. |
| `indicators` | The trade inspector draws no lines, and the only way to see why a trade fired is to recompute the series somewhere else — where it is free to disagree with the signals and draw a crossover a bar from where the entry actually happened. |
| `LOGIC` | The tear sheet's strategy card says "not declared". The alternative is inferring the rules from the trades, which is a guess printed as a fact. |
| `PARAM_GRID` | `--scan` has nothing to sweep, so the parameters are whichever ones got typed first and were never compared against anything. |

Write `indicators` in the same module and from the same column `signal_fn`
reads. A second implementation living in the report would be free to disagree
with this one, with nothing raising.

A module may also declare the search space `backtest/run.py --scan` sweeps. It
lives here because this module is the only place that knows what its parameters
mean and what its signature will accept — `load_strategy` rejects unknown
parameter names, so a stale key raises rather than being quietly ignored:

```python
PARAM_GRID = {"ema_period": [15, 20, 30], "atr_mult": [2.0, 2.5, 3.0]}
```

Keep it coarse and small. Nine combinations over 4,000 daily bars is a search
whose result can be reported honestly; a 400-cell grid over the same bars is a
machine for manufacturing an in-sample Sharpe.

Two further declarations are optional, read by the loader, and used **only by
the tear sheet** — neither can change a signal:

```python
LOGIC = {"concept": "...",                                  # plain English
         "entry": "Go Long when the Fast SMA ({fast_window}) crosses above "
                  "the Slow SMA ({slow_window}).",
         "exit":  "Exit when it crosses back below."}       # {param} slots are
                                                            # filled with the
                                                            # bound params
def indicators(bars, **params) -> dict[str, pd.Series]:     # full-length series
    ...                                                     # drawn over the
                                                            # inspector's candles
```

Write the indicator series the same way the signals are computed, in the same
module. A second implementation in the report would be free to disagree with
this one and draw a crossover a bar away from where the trade fired, with
nothing raising.

- **Dual-Version Mandate:** every strategy outputs two versions. **Version A** is
  a pure rule-based baseline (e.g. an SMA crossover); **Version B** adds an ML
  filter over the same signals. ML is adopted only if B beats A out-of-sample.
  *Note: LightGBM is not currently in `requirements.txt`.*

---

## The Strategy Request Template

**Every strategy starts from a filled-in copy of this block. It is mandatory.**
No module gets written from a one-line prompt — "try a mean reversion on ES" does
not say over what period, against what costs, at what timeframe, or what would
count as it working, and every one of those gets decided anyway. Decided
silently, after the fact, by whoever is looking at the equity curve. A
specification written before the backtest is the only version of it that cannot
be adjusted to fit the result.

The sections are ordered the way a strategy is judged: why it should work, what
it trades, the rules, what they cost, and what would settle it.

```markdown
### STRATEGY SPECIFICATION & BACKTEST REQUEST

**1. Hypothesis & Market Rationale**
   - What inefficiency is being harvested, and why it persists
   - Strategy family (see docs/STRATEGY_FAMILIES.md)

**2. Universe & Data**
   - Symbols:              NQ,ES  |  ALL
   - Timeframe:            15m
   - In-sample period:     2010-01-01 -> 2023-08-16
   - Holdout (untouched):  final 3 years

**3. Signal Logic (plain English)**
   - Core concept:
   - Entry trigger:
   - Exit rule:
   - Filters / regime conditions:

**4. Risk & Execution**
   - Stop / target:          none modelled unless stated
   - Session flatten:        --flat-by-close ?
   - Contracts:              1
   - Slippage / commission:  1 tick each way + specs.py

**5. Parameter Space & Acceptance**
   - PARAM_GRID:
   - Variants expected:
   - Gates that must clear:  1 in-sample / 2 robustness / 3 holdout
```

Notes on filling it in, and on what each section is defending against:

- **Section 1** is the part that cannot be recovered later. A strategy with no
  stated reason for existing is indistinguishable from one found by searching
  until something looked good, and the two fail differently in live markets.
- **Section 2** names the holdout **before** the run. Reserving the last three
  years afterwards is not reserving them — by then they have been seen.
- **Section 3** becomes the module's `LOGIC` block close to verbatim. If a rule
  cannot be stated here in plain English, it cannot be stated on the tear
  sheet's strategy card either, and nobody deciding whether to trade it will be
  able to read what it does.
- **Section 4** is where the silent assumptions live. The engine models **no
  stop and no take-profit** unless the strategy's own signals produce them, so
  "none modelled" is the default and a drawdown read on the assumption of an
  unstated stop is being read wrong.
- **Section 5** fixes the size of the search before it runs. `--scan` reports
  `variants_tested`, and a Sharpe read without knowing how many variants
  produced it is not a measurement. Stating the gates up front is what stops
  the acceptance criteria from being revised down to meet the result.

---

## The Dual-Version Workflow

What to do when a backtest finishes:

1. Console scorecard and gate audit — **Version A**
2. Standalone HTML report with the trade inspector — **Version A**
3. Console scorecard and gate audit — **Version B**
4. Standalone HTML report with the trade inspector — **Version B**
5. The four-choice menu

Steps 1-4 are automatic. Step 5 is where a human decides, and nothing is
promoted without them.

**1. The gates.** `run_dual_version_backtest` calls `audit_acceptance_gates`
for A and B and attaches the result as `gate_audit` on each version.

| Gate | Criterion | Threshold |
|---|---|---|
| **1 · In-Sample** | Sharpe | >= 1.20 |
| | Profit factor | >= 1.50 |
| | Trades | >= 200 |
| | Max drawdown | <= 15.0 % |
| **2 · Robustness** | WFO efficiency | >= 0.50 |
| | Monte Carlo 95% max DD | <= 18.0 % |
| **3 · OOS Holdout** | Holdout Sharpe / IS Sharpe | >= 0.85 (<= 15% degradation) |

Gates 2 and 3 need evidence the dual run does not produce — a walk-forward, a
bootstrap, and the held-back final 3 years are separate runs. Pass them in via
`robustness={"A": {...}, "B": {...}}` and `holdout={"A": {...}, "B": {...}}`.
**Without them those gates report `NOT EVALUATED`, which is not a pass.**
`audit["passed"]` is True only when all three gates cleared on real numbers, so
nothing can be promoted on a gate that was never run. Drawdowns are compared on
magnitude — the engine signs them negative, and comparing raw would let -40%
clear a 15% limit.

**2. Print the console scorecard, then emit the HTML report — Version A first,
then Version B.** Take the versions one at a time, in that order: the console
scorecard for A, then A's report, then the same pair for B. A reader who sees
Version B's tear sheet before Version A's has no baseline to judge it against,
and the whole point of the mandate is that B is only interesting relative to A.

`print_dual_scorecard(metrics_a, metrics_b, audit_a, audit_b)` renders both
columns side by side with a `B − A` delta, the gate table, per-criterion detail
for every FAIL, and the verdict. Passing `metrics_b=None` (Version B was not
run) drops the B and `B − A` columns entirely and says so in the verdict — a
column of `n/a` under a header reading "B · ML-filtered" invites the reading
that the filter ran and produced nothing.

`report_version_a.html` and `report_version_b.html`, plus `dual_metrics.json`,
land in `/mnt/backtest/artifacts/<strat_name>_<timestamp>/`. The timestamp is on
the directory, not the filename, so a re-run never overwrites the evidence an
earlier promotion decision was made on. A write failure is recorded in
`result["reports"]["error"]` and printed to stderr — never raised, because a
completed backtest is not thrown away over a busy NFS mount. Pass
`emit_reports=False` to skip.

The batch runner passes `prefix=<SYMBOL>`, so its files are
`report_<SYMBOL>_version_a.html` and `dual_metrics_<SYMBOL>.json` in
`<strat_name>_<timestamp>/`. One directory holds one run and a run covers many
contracts; unprefixed, the second symbol's report would overwrite the first
with nothing raising.

Each report carries the gate badges, the alpha metrics, the equity and
underwater curves, the strategy logic card, the monthly heatmap, a searchable
and sortable trade log, and the **trade inspector**: click any row for a
candlestick of that trade, 20 bars before the entry through 10 after the exit,
entry and exit marked, with the strategy's own indicator lines drawn over the
candles so the crossover or band touch behind the trade is visible next to the
IN and OUT markers. The bar windows and the indicator series are embedded at
build time, so the file keeps working with no lake, no server, and no network.

Read the logic card before the metrics. In plain English it states what this
strategy does — core concept, entry trigger, exit rule — and what the engine
actually did: fills on the next bar's open, the session flatten setting, the
costs charged, and that **no stop-loss or take-profit is modelled at all**. A
drawdown figure read on the assumption of an unstated stop is being read wrong.

**3. Present the menu. Do not choose for them.**

```text
[1] Promote Version A to Incubator     [2] Promote Version B to Incubator
[3] Parameter Sweep / Sensitivity      [4] Keep in Experimental / New Idea
```

**4-5. Promote (or sweep, or leave it).**

```bash
python3 backtest/promote.py --strat sma_crossover --version A \
    --source strategies/experimental/sma_crossover.py \
    --metrics /mnt/backtest/artifacts/sma_crossover_<ts>/dual_metrics_NQ.json
```

Promotion is per strategy, and a batch covers many contracts. Name the symbol
whose evidence the decision rests on — the snapshot records which contract
produced the numbers.

Writes `strategies/approved_incubator/<strat>/` — `strat.py`, `meta.json`
(version, symbol, timeframe, params, locked metrics snapshot, gate statuses,
SHA-256, timestamp), plus `baseline.py` for Version B — and commits that
directory alone.

- **Version A is promoted byte for byte** and its SHA-256 recorded, so the
  promoted file provably *is* the file that was backtested. A strategy cleaned
  up on the way through is a different strategy.
- **Version B is a pipeline, not a file.** `baseline.py` is the verbatim source;
  `strat.py` is a generated wrapper applying `apply_ml_signal_filter` at the
  same threshold the run used.
- **Promotion is refused when the gate audit is not PASS.** `--force` overrides
  and records `gates_overridden: true` in `meta.json`.
- **No `--metrics` means `metrics_status: "NOT RECORDED"`** in `meta.json`, not
  an invented snapshot.
- Being in `approved_incubator/` is a record that a version was chosen, **not
  permission to trade it.**

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
~100-200 trades — too thin to separate skill from luck. Pass a symbol list:
`bt-run --strat <name> --symbols ALL`. That runs 27 independent backtests and
gives you a leaderboard, which is the cross-sectional evidence. It is NOT the
same as merging their trades into one equity curve, and nothing in the runner
does that — an edge has to exist on a contract before a portfolio of them means
anything.

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
