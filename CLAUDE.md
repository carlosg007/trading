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

**STRICT TOOL BOUNDARY: NEVER execute `backtest/run.py` or run full backtests
inside Claude Code. Doing so consumes excessive LLM tokens. Claude Code's role
is strictly to write code, perform AST validation, run fast synthetic unit
tests, commit to git, print the exact terminal command for the human operator,
and STOP.**

The five responsibilities, in full:

1. Writing strategy modules and infrastructure code.
2. Running fast static checks and the AST security validator.
3. Running unit tests on small synthetic fixtures.
4. Committing changes to git.
5. Printing the exact manual CLI command for the operator to run in their
   regular shell — then stopping.

This covers any run against the real lake and any multi-symbol run, in the
foreground or detached. **`--bg` does not satisfy the boundary**: the run still
belongs to the session, its log still streams back into context, and the
operator still is not at the console when the gate audit prints. Step 5 of the
dual-version workflow — the four-choice menu — exists so a human sees the
evidence before anything is promoted. Print the command with the flags,
symbols, timeframe and date window spelled out, and stop there.

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
# Seventeen suites. Everything except test_streaming_lake, test_engine_batching
# and test_engine_vbt runs without the lake or a network (test_batch_runner's
# --symbols checks and test_intraday_vol_mr's real-bar section skip, loudly,
# without it). test_report_gates.py shells out to `node` for the trade
# inspector's own checks and skips them, loudly, when node is absent.
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
python tests/test_intraday_vol_mr.py    # the band-fade walk, against hand answers
python tests/test_risk_params.py        # TP/SL/trailing walk, grid, leaderboard
python tests/test_daily_metrics.py      # the daily-close metric frequency contract
python tests/test_pipeline_filters.py   # entry filters, DOW attribution, the 5 stages
python tests/test_regime_cache.py      # regime quadrants, theta_vol, the lake join
python tests/test_profiler_precomputed.py  # the profiler reads the cache, not its own pass
python tests/test_stage1_charter.py     # the Stage 1 charter + the Discord card
OMP_NUM_THREADS=1 \
  python tests/test_sma_momentum_crossover.py   # ADX vs TA-Lib, layers, ml_features

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

# THE FIVE-STAGE PIPELINE. Each stage prints the next stage's command and
# stops; nothing chains automatically, because the point of the stages is that
# a human reads the evidence between them. Handoff files live in
# <BT_ARTIFACTS>/pipeline/<strategy>/ - see backtest/pipeline.py.
python3 backtest/baseline.py    --strat X --symbols ALL --tf 15m \
    --start 2013-01-01 --end 2022-12-31       # 1: drop PF<1.0 contracts
python3 backtest/scan.py        --strat X --tf 15m \
    --start 2013-01-01 --end 2022-12-31       # 2: sweep the survivors

# Stages 1 and 2 take a comma-separated --tf and evaluate each in turn.
# The lake derives 5m/15m/30m/1h/2h/4h from the 1m parquet, so nothing
# resamples in the stage. Stages 3 and 4 take ONE timeframe.
python3 backtest/baseline.py --strat X --symbols ALL --tf 1m,5m,15m,30m \
    --start 2013-01-01 --end 2022-12-31
python3 backtest/audit_gates.py --strat X --tf 15m \
    --is-start 2013-01-01 --is-end 2022-12-31 \
    --holdout-start 2023-01-01 --holdout-end 2026-01-01   # 3: certify
python3 backtest/verify_full.py --strat X --tf 15m \
    --start 2010-01-01 --end 2026-01-01       # 4: tear sheets + cost drag
python3 backtest/promote.py --strat X --version A --source <module.py> \
    --audit-file /mnt/backtest/artifacts/pipeline/X/gate_audit_NQ.json  # 5

# The entry filters. Available on all five stages AND on bt-run, spelled
# identically because they come from one add_filter_args().
bt-run --strat X --symbols NQ --news-filter --news-window 30
bt-run --strat X --symbols NQ --exclude-days 0,4     # no Mon/Fri ENTRIES

# The Regime-Aware Screening Firewall (replaced the Drop Unprofitable Days
# contract, 2026-08-20). Stage 1 profiles every configuration into four
# volatility/trend quadrants and keeps only those with a quadrant at PF >= 1.00
# over >= 30 trades; each survivor carries version, optimal_regime, quadrant
# (Q1..Q4), that quadrant's metrics and kill_switch_regimes into
# surviving_assets.json. NO weekday is blacklisted any more, so Stage 1 writes
# no exclude_days and Stage 2 inherits none.
# Every stage ends by printing its own leaderboard.
python3 backtest/baseline.py --strat X --symbols ALL --tf 15m \
    --min-profit-factor 1.25 --min-trades 50    # tighten the quadrant bars
# STAGE 1 ONLY: --start/--end default to the charter window 2013-01-01..
# 2022-12-31 (the years after it are the Stage 3 holdout) and Version B runs
# by DEFAULT, because survival is decided on either version. --no-ml declines
# it. Every other stage and bt-run are unchanged: --ml stays opt-in there.
python3 backtest/baseline.py --strat X --symbols ALL --tf 15m --no-ml

# Post Stage 1's leaderboard to $BT_DISCORD_WEBHOOK. Reads the handoff and
# recomputes nothing; --dry-run prints the payload and sends nothing.
python3 backtest/discord_reporter.py --stage 1 --strat X
python3 backtest/discord_reporter.py --stage 1 --strat X \
    --survivors /mnt/backtest/artifacts/pipeline/X/surviving_assets.json
# Stage 2 still HONOURS an exclude_days a handoff carries; nothing writes one.
python3 backtest/scan.py --strat X --tf 15m --ignore-stage1-exclude-days
python3 backtest/scan.py --strat X --tf 15m --exclude-days 3,4   # CLI wins

# What macro calendar a news-filtered run would actually use, and whether its
# dates are published or rule-generated. Reads no bars; costs nothing.
python3 backtest/event_calendar.py --start 2015-01-01 --end 2026-01-01

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

# Build the pre-computed regime cache (ADX14/ATR14/quadrant per symbol+tf).
# Writes /mnt/backtest/lake/regimes/{SYMBOL}_{TF}_regime.parquet ($BT_REGIME_CACHE
# overrides). mdlib.lake picks these up automatically; re-run after a data pull.
python scripts/precompute_regimes.py --symbols NQ,GC --tf 15m,30m
python scripts/precompute_regimes.py --symbols ALL --tf 15m --force

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
- **`regimes=True` (default) left-joins the pre-computed regime cache** onto
  each symbol's frame. Applied inside `iter_bars`, so `get_bars` inherits it
  and the two keep returning the same columns —
  `tests/test_streaming_lake.py` pins that equality. Joined PER SYMBOL, inside
  the loop, because the regime file is keyed by timestamp alone and a join on
  the concatenated frame would hand every contract NQ's ADX with the column
  still fully populated. A cache MISS is not an error: the frame comes back
  with no regime columns at all, rather than a column of zeros that would read
  as a regime that was computed and found absent. Nothing is computed on the
  fly — see `mdlib/regimes.py` for why an improvised threshold is worse than
  no threshold.
- Reference lookups (`coverage()`, `degraded_days()`, `roll_dates()`) read from
  `/mnt/backtest/reference/futures/` and are `lru_cache`d.
- **Dual-dataset routing is not implemented** — `get_bars` has no `source`
  parameter, so the NT8 tree is unreachable through the reader. See Open Tasks.

**`mdlib/regimes.py`** — the pre-computed regime feature cache, and the only
place the quadrant encoding is written down. Wilder's ADX(14) and ATR(14) are a
pure function of the bars, and every stage that profiles a result was
recomputing them over the same series.

- **`regime_quadrant` is `uint8`: 1 High-Vol/Trending, 2 High-Vol/Ranging,
  3 Low-Vol/Trending, 4 Low-Vol/Ranging, and 0 = UNDEFINED.** The order matches
  `backtest.profiler.REGIMES`, so a quadrant integer and a profiler label are
  the same statement about the same bar. **0 is not a quadrant.** ADX and ATR
  are undefined during their 14-bar warm-up, and `NaN > 25.0` is False — the
  naive encoding files every warm-up bar under Low-Vol/Ranging, a populated
  column of a regime nobody measured. Bars outside the cache's span get 0 for
  the same reason. Anything reading the column must treat 0 as "no regime".
- **The volatility boundary is pinned to the IN-SAMPLE window**, default
  2013-01-01..2022-12-31, and then applied to the whole series including the
  holdout years — which are therefore labelled by a boundary that never saw
  them. A median taken over whatever window the caller asked for is a property
  of the REQUEST, not of the contract: the same bar would be labelled one way
  by an in-sample run and another by a holdout run, and Gate 3 would measure
  retention between two strategies whose regime definitions disagree. `theta_vol`
  is stored in the file, because a quadrant read without knowing which window
  drew its boundary is not a measurement.
- **`backtest.profiler.RegimeProfiler` consumes this cache from 2026-08-20.**
  When the bars carry `regime_quadrant` — which they do whenever they came
  through `mdlib.lake` — the profiler labels every bar from the cached column
  and runs no indicator pass of its own. It falls back to the original live
  ADX/ATR pass otherwise, and the two are NOT equivalent: the fallback takes
  the median ATR of whatever frame it was handed, so its boundary moves with
  the requested date range while the cache's does not. Which one ran is
  recorded on every profile artifact as `regime_source`
  (`precomputed_cache` / `recomputed_live`) alongside the threshold that drew
  the quadrants, and Stage 1 collects them per configuration into
  `regime_screen.regime_source`. **Switching a (symbol, tf) from recomputed to
  cached MOVES its quadrant boundary and therefore its Stage 1 verdict** — that
  is the intended effect, not drift, but a before/after is not comparable
  across the change.
- The integer→label map lives in `backtest/profiler.py` as
  `QUADRANT_TO_REGIME`, built FROM `QUADRANT_LABELS` here and **checked against
  `REGIMES` at import**, which raises on disagreement. A transposed map would
  move every trade between quadrants with every total still adding up.
- **Provenance lives inside the parquet**, as schema metadata, not in a sidecar
  that can be separated from what it describes: `theta_vol`, the in-sample
  window, the indicator lengths, and the lake hygiene flags the bars were read
  under. A cache built on bars that included roll days holds an ATR shaped by
  those gaps; joined onto a run that excluded them every timestamp still
  matches, so `attach` compares the flags and says so.
- **Warnings go to stderr, once per (symbol, tf).** `mdlib.lake` is a library
  and several callers parse a child process's stdout as data.

**`backtest/engine.py`** — Runs the simulations.
`run_backtest(symbols, tf, signal_fn, start, end, cfg)` → `BacktestResult(returns,
trades, equity, breach, stats)`.

- **The engine reads the bars; you pass the strategy, not the signals.**
  `signal_fn(bars) -> (entries, exits)` — or, since 2026-08-16,
  `-> (entries, exits, short_entries, short_exits)` — is called once per symbol
  with that symbol's bars alone. There is deliberately no way to hand it a pre-built
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
- **Every risk-adjusted ratio is computed on DAILY CLOSES**, whatever timeframe
  the bars were read at. `_daily_returns` collapses the equity curve to one
  point per session, and `_assemble_result` derives Sharpe, Sortino and Calmar
  from that one series via `backtest.report.daily_metrics` — so a 15m run and a
  1d run of the same strategy are directly comparable, and the sqrt(252) in
  `report.sharpe` is applied at the frequency it actually annualizes. Handing
  those functions an intraday return series does not raise; it just returns a
  number scaled by the wrong root. The sampling frequency, annualization
  factor, risk-free rate (`BacktestConfig.risk_free_rate`, 0 by default) and
  day count travel with the numbers as `stats["basis"]`, and reach the
  scorecard JSON as `metrics_basis`.
  **"Daily" means one point per session date PRESENT IN THE DATA, not
  `.resample("1D").last().ffill()`** — a calendar resample invents weekend and
  holiday rows that carry a return of exactly 0.0, dragging the mean and
  standard deviation toward a 365-day year while the sqrt(252) stays put. On a
  20-session synthetic run that alone moved Sharpe from -15.08 to -11.26.
- **Costs are mandatory:** slippage and commissions applied at this layer.
- **Entry filters, from 2026-08-17.** `BacktestConfig.news_filter`,
  `.news_window_minutes`, `.news_kinds` and `.exclude_days` are applied in
  `run_backtest` right after `unpack_signals` — and, separately, in
  `run_dual_version_backtest`, which drives `_simulate` directly rather than
  going through `run_backtest`. Wiring only the engine's entry point would
  leave the flag silently inert for the batch runner and all five pipeline
  stages, which is every path an operator actually uses. What was removed is
  recorded in `stats["entry_filters"]` and reaches the metrics dict; a filter
  that ran and cut nothing reports zero counts, which is a different statement
  from no filter at all. See `backtest/event_calendar.py`.
- **Long AND short, from 2026-08-16.** A strategy returns two masks or four;
  `unpack_signals` accepts both and fills the short pair with False for the
  two-mask form. Signals are resolved by ONE three-state machine
  (`_clean_signals_ls_loop`: flat / long / short) — `clean_signals` delegates to
  it with empty short masks, so there is no second resolver to disagree with the
  first. Three rules it will not guess at: a bar signalling both sides while
  flat takes NEITHER; an opposite entry while in a position is DROPPED, never a
  reversal (a reversal would leave no flat bar for `_chunk_bounds` to cut on);
  and each closed trade's `direction` is stamped from vectorbt's own record,
  never inferred from the sign of the P&L. The long-only `direction="longonly"`
  call is kept as a separate branch from the four-mask one, so adding shorts
  changed no number in any existing backtest. There is no `direction="both"`:
  vectorbtpro refuses `direction` alongside short signal arrays, and the four
  masks ARE the both-directions mode.
- **Still no stop or target ORDERS, in either direction.** A stop is an exit
  signal detected on the breaching bar and filled at the NEXT bar's open. The
  risk machinery lives in the strategy module (`_walk`), not here.
- **Numba:** `clean_signals` compiles its three-state machine via `njit`
  (`cache=True, nogil=True`) and falls back to the interpreted loop when numba
  is absent. `tests/test_clean_signals.py` checks it against a pure-Python
  two-state oracle (`_clean_signals_loop`, kept for exactly that), which also
  pins the long-only reduction exhaustively at every length up to 10.

**`backtest/specs.py`** — contract multiplier, tick size, commission per symbol.
A wrong multiplier silently scales every P&L figure for that symbol and the
backtest still looks plausible. `verify_specs()` reconciles against Databento's
`definition` schema.

**`backtest/report.py`** — also holds `day_of_week_breakdown` /
`losing_weekdays` / `format_day_of_week`: P&L, win rate and trade count per
weekday, attributed by the **ENTRY** session (not the exit — `exclude_days`
acts on entries, so a table keyed on exits would point at a day whose pruning
removes different trades) and on the **session** date (not the UTC date). Every
weekday Mon–Fri is present even with zero trades, because an absent row reads
as missing data when it means "this never traded on a Friday". `losing_weekdays`
enforces a trade floor and remains a suggestion for a human. The records land in
`summarize_result` as `dow_breakdown` and so reach `dual_metrics.json`.

`unprofitable_weekdays` (added 2026-08-19) is the automated half. **Stage 1
stopped calling it on 2026-08-20**, when the regime firewall replaced the Drop
Unprofitable Days contract; it is retained as a library function and has no
caller in the pipeline. It returns EVERY Mon–Fri
weekday whose profit factor is below 1.00 — `DOW_MIN_PROFIT_FACTOR`, the same
bar the contract screen and Gate 1 use — over at least `min_trades` trades, in
weekday order. Zero, one and five days are one code path and one shape; an empty
list means the whole week cleared the bar, which is a result rather than a
missing value. Three rules that are the point of the function: a weekend row is
never returned, since a Saturday row is a bug worth seeing rather than a session
to prune; a weekday whose profit factor is UNDEFINED (it never lost) is never
dropped, checked explicitly rather than left to `NaN < 1.00` being False by
accident; and `exclude_days_basis` states the rule in words so no artifact
records which days were dropped without recording why.

Standalone CLI over saved parquet; joins regime labels
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
  sl_atr_mult tp_atr_mult trailing
  ```

  `timestamp` is the run's stamp, identical on every row and equal to the
  directory's, so leaderboards from several runs concatenate and group cleanly.
  `params` is the **full effective set** the signals were bound to — the
  module's `DEFAULT_PARAMS` with `--param` and `--scan`'s winner layered over
  them, not just what the CLI was handed.
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

  The last seven columns sit **after** the declared schema rather than inside
  it, so a reader slicing the first fifteen gets exactly the specified file.
  They are there because dropping them would make the CSV lie rather than
  merely make it shorter: without `status`/`error` a symbol that failed to load
  reads as a strategy that produced nothing, and without `variants_tested` a
  `--scan` Sharpe is the best of an unstated N.

  `sl_atr_mult`, `tp_atr_mult` and `trailing` lift the winning **risk**
  settings out of the `params` string into columns that filter and sort, which
  is how "did the take-profit earn its place across contracts" gets answered.
  A **blank** cell means the strategy declares no such parameter; the literal
  word `None` under `tp_atr_mult` means it has one and this run modelled **no
  take-profit at all**. Those are different statements and must not collapse
  into one.

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

**`backtest/scan.py`** — **Stage 2**, and the vectorbt-native sweep behind
`--scan`. One module, two entry points: `scan_symbol` is the library the batch
runner calls, and `main()` is the stage that sweeps the in-sample window per
contract and writes `best_params_<SYMBOL>.json`. The stage adds NOTHING to the
search — same selection rule, same tie-break — because a second sweep
implementation in a CLI would be free to disagree with the one `--scan` uses
and the two would be compared by nobody. With no `--symbols` it sweeps Stage
1's survivors. A
strategy declares `PARAM_GRID = {"ema_period": [15, 20, 30], ...}`; every
combination becomes a COLUMN of a single `vbt.Portfolio.from_signals` call, so a
27-cell grid costs roughly one backtest rather than 27. The winner is the
highest Sharpe **among the combinations whose Gate 1 audit is PASS**.

- **The sweep runs the ENTRY filters, from 2026-08-19, and inherits Stage 1's
  losing days automatically.** `_combo_signals` applies the block mask
  immediately after `unpack_signals` and before `apply_flat_by_close` — exactly
  where `run_backtest` applies it, and before `clean_signals_ls`, because
  suppressing an entry after the three-state machine has run would leave the
  other side's signal resolved against a trade that no longer exists. The mask
  is built ONCE per contract by `_entry_block_mask` (it reads timestamps and
  nothing else), not once per grid cell.
  **Before this the sweep ignored `cfg.news_filter` and `cfg.exclude_days`
  entirely**: the flags parsed, reached the config, printed on the console and
  changed no signal, so every parameter set was selected on the unfiltered week
  and then re-run filtered. `tests/test_batch_runner.py` now pins the filtered
  sweep against an `apply_entry_filters` + `_simulate` oracle trade for trade,
  and separately against the unfiltered sweep — parity alone would be passed by
  a no-op mask.
- **Precedence lives in one function, `resolve_exclude_days`.** An explicit
  `--exclude-days` wins outright and applies to every contract; otherwise Stage
  1's per-pair `exclude_days` applies with no flag;
  `--ignore-stage1-exclude-days` removes that middle step only and never
  disables an explicit `--exclude-days`. Whichever way it resolves, the answer
  and its PROVENANCE are printed and written onto
  `best_params_<SYMBOL>_<TF>.json` — "Monday was excluded" is never recorded
  without who decided it.
- **`scan["entry_filters"]` is `None` when neither filter was configured**, and
  a dict recording what was blocked when one was. A filter that ran and cut
  nothing is a different result from no filter, and the two tables are otherwise
  indistinguishable. A `--reuse-scan` rebuild reports `None` — "not recorded" —
  because the CSV holds metrics, not the mask that produced them.
- **`winners_leaderboard` is the table the stage ends on** — Symbol, Timeframe,
  in-sample PF, in-sample Sharpe, max drawdown, the winning parameters in full,
  and the excluded days, sorted by Sharpe descending because that is the metric
  the sweep selected on. A contract with no measurable Sharpe sorts LAST rather
  than being dropped: "this grid produced no trades" is a finding about the
  space, and an absent row reads as a sweep that never ran. The excluded-days
  column is not decoration — two rows with the same parameters and different
  exclusions were fitted to different weeks, and without it the table presents
  them as comparable.

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
- **Risk parameters sweep like any other.** Nothing in the scanner knows
  whether a parameter is an indicator period or a stop multiplier — the
  strategy module decides that. Two consequences, both silent when wrong:
  - `None` is a legitimate grid VALUE (`tp_atr_mult: [2.5, 5.0, None]` searches
    "no take-profit" as one of its points). pandas turns floats-and-`None` into
    float64-and-NaN, and `NaN == None` is False, so the winner is matched with
    `_same_value`, not `==`. Without it the winning row is never flagged
    `selected` and the CSV shows a sweep with no winner beside a run that used
    one. `scan_symbol` raises if the winner cannot be matched back at all.
  - **Risk axes multiply.** Adding 4 stops × 5 targets × 2 trailing flags to a
    48-cell indicator grid is 1,920 fits to one sample. `run.py` prints the
    cell count before it sweeps and warns past `SIZE_WARN` (200). Nothing is
    refused — a grid somebody deliberately wrote is theirs to run — but the
    operator sees which claim they asked for before it starts.
- **Ties break on the shallower drawdown.** Sharpe is the ranking metric
  because it IS the risk-adjusted return; ranking on raw return would pick
  whichever set took the most risk to get there. Ties are real once risk is
  swept — a target no bar ever reaches and `tp_atr_mult=None` produce the same
  trade list — and between two identical Sharpes the smaller `abs(max
  drawdown)` wins rather than whichever the grid declared first. The rule and
  its tie-break live in ONE function, `_select_best_row`, shared by the sweep
  and by the CSV rebuild below — a second copy would be free to disagree about
  which row won, and the two would be compared by nobody.
- **`--reuse-scan` rebuilds the export without re-fitting the grid**, from the
  `scan_<SYMBOL>.csv` files already in the artifact directory. It reads no bars
  and runs no simulation: it re-derives the winner with `_select_best_row` and
  writes `best_params_<SYMBOL>_<TF>.json`. The grid is the expensive half of
  Stage 2 and the export is the cheap one, and they used to fail together — a
  crash in the export cost six fully-swept configurations (2 contracts × 3
  timeframes × 1,296 cells) that had already written their tables. It is for
  recovering an export, **not** for a rerun: `--start`/`--end`/`--param` are
  recorded from the CLI and never checked against the table, so naming a
  different window there writes a file that misdescribes its own bars.
  - **The `params` column is what it reads, not the per-parameter columns.**
    That column is `str(dict(combo))` and is the only field that survives the
    round trip intact — pandas reads a `tp_atr_mult` column of floats-and-`None`
    back as float64 with NaN, and binding NaN to a strategy is not the run that
    was swept. `parse_param_dict` restores the types from any of the four
    shapes a parameter set is written in here (native dict, that Python repr, a
    JSON object, and the console's `fast_period=13, …` form); it RAISES on
    input it cannot read rather than returning `{}`, because an empty dict
    silently binds the module's defaults while the run is reported under the
    winner's name.
  - **A table that disagrees with the rule is refused.** If the CSV's own
    `selected` flag names a different row than `_select_best_row` picks, the
    rebuild raises instead of writing a `best_params` pointing at one row
    beside a table flagging another.
  - **A rebuilt file says so** — `rebuilt_from` and `rebuilt_note`. Its
    `combinations` and `rejected` counts are FLOORS: combinations the strategy
    rejected were never written to the CSV, so a rebuild cannot know them.
    `variants_tested` is unaffected (it has always been the number evaluated),
    and `in_sample` carries only the metrics the table held — win rate, Calmar
    and the day count are omitted rather than defaulted, because a zero win
    rate beside a profitable profit factor is a number nobody computed.

**`backtest/event_calendar.py`** — the two ENTRY filters, and the only place
either is implemented. Named `event_calendar` rather than `calendar` because
the latter shadows the stdlib module for any script run out of `backtest/` —
`_strptime` imports `calendar`, so `bt-run` died on import with the shorter
name.

- **Entries only, never exits, on both sides.** `apply_entry_filters` is handed
  the entry masks and nothing else, so a blocked exit — holding a position
  through the release the filter exists to dodge — is not expressible.
- **A signal is judged on the bar it FILLS.** The engine fills at the next
  bar's open, so every mask is widened one bar backwards
  (`_widen_to_fill_bar`). Without that, exactly one entry per event slips
  through and fills inside the window: the least visible outcome, and the whole
  population the filter was added to remove.
- **Blocking is causal.** US release SCHEDULES are published a year ahead, so
  blocking the half hour before a print is not lookahead. The OUTCOME is not
  knowable and nothing here reads one — this module only ever sees timestamps.
- **Dates carry provenance, and it is not decoration.** `PUBLISHED` comes from
  an operator-supplied CSV (`$BT_MACRO_CALENDAR`, else
  `/mnt/backtest/reference/macro/us_macro_events.csv`). `RULE` is generated
  here. **Only NFP's rule is the real convention** (first Friday, 08:30 ET);
  CPI, PPI and FOMC anchors land in the right week and often the wrong day,
  and a 30-minute window on the wrong day blocks a random half hour while
  leaving the release tradeable. The token is recorded in
  `stats["entry_filters"]`, printed by `describe_filters`, and carried into
  every metrics dict — a news-filtered result must never be read as one that
  dodged the actual prints without checking it.
- **An empty calendar RAISES.** `is_news_blocked(strict=True)` refuses a span
  its calendar does not cover, because an all-False mask and a missing
  calendar are indistinguishable downstream. `strict=False` is the deliberate
  way to say a period genuinely holds no events.
- **`exclude_days` is keyed on the CME SESSION date**, not the UTC date
  (`session_date`: any bar at or after 18:00 ET rolls to the next day). One
  line, and it has to be right for daily bars as well as intraday ones — a 1d
  bar stamped at UTC midnight is 19:00 or 20:00 ET the previous evening, so
  the roll puts it back on its own date in both EST and EDT.
- O(n log m) via merged intervals and `searchsorted`. The broadcast form on
  5.6M bars × ~700 events would allocate 4e9 booleans.

**`backtest/pipeline.py`** — not a stage; the contract BETWEEN them. Where each
stage writes, what the next reads, and the banner that says which stage a log
came from. `read_stage` refuses a file written by the wrong stage or belonging
to another strategy — certifying one strategy's gates against another's
parameters is a mistake nothing downstream could detect. Handoffs are written
atomically (temp file, `os.replace`) into
`<BT_ARTIFACTS>/pipeline/<strategy>/`, one directory per strategy rather than
per run, because the files are a chain and Stage 3 has to find Stage 2's winner
without being told a timestamp. `stage1_exclude_days` is the one place a per-pair
weekday exclusion is read back out, keyed per `(symbol, timeframe)`; a pair with
nothing to exclude is ABSENT from the mapping rather than mapped to an empty
tuple, and a handoff written before the contract existed simply yields `{}`.
**Since the regime firewall replaced Stage 1's Drop Unprofitable Days contract
(2026-08-20) that is what it always returns from a fresh Stage 1 run** — the
inheritance path in Stage 2 and Stage 3 is unchanged and still honours a handoff
that carries days, but nothing in the pipeline writes one any more. `leaderboard(title, header, rows)` renders the end-of-stage table for all
three stages — one implementation, because these are read as a sequence and a
Symbol column aligned one way in Stage 1 and another in Stage 2 makes two tables
of the same contracts look like tables of different things. Columns size to
their widest CELL, so a long parameter set widens its own column rather than
being silently clipped.

**Multi-timeframe scanning.** Stages 1 and 2 accept `--tf 1m,5m,15m,30m` and
run each timeframe independently; `run.parse_timeframes` validates against
`mdlib.lake`'s native and DERIVED sets up front, so a typo fails in a second
rather than several contracts into a sweep. **Nothing resamples in a stage** —
the reader aggregates the 1m parquet, and a second implementation in a stage
would be free to disagree with it about bar boundaries and the Sunday merge.
Three consequences, all of them selection effects that would otherwise go
unrecorded:

- **A period is not a horizon.** `trend_period=200` is ~3.3 hours on 1m and
  four trading days on 30m. Sweeping timeframes tests different strategies,
  not one strategy at several resolutions; the claim is the `(tf, params)`
  pair.
- **A filter subtracts CANDIDATES, not trades.** `ema_trend_filter`'s
  confluence conditions are boolean toggles, and switching one on can only
  remove eligible triggers. It cannot only remove TRADES: the walk holds one
  position at a time and ignores a trigger arriving while one is open, so
  declining an early trigger leaves the strategy flat for a later one it would
  have been holding through. Measured on a synthetic fixture, enabling
  `use_vwap` removed 17 candidates and ADDED 11 realised entries. Never use a
  trade count to decide whether a filter binds — compare candidates or the
  trade list.
- **Stage 1's `surviving` list is a UNION** when several timeframes ran, and
  the file says so (`surviving_is_union_across_timeframes`). `by_timeframe` is
  where "did it survive at 15m" is answered.
- **Stage 2 writes `best_params_<SYMBOL>_<TF>.json` per timeframe and, on a
  multi-timeframe run, NO unsuffixed file.** Picking the best timeframe off a
  leaderboard is a second selection layer stacked on the parameter sweep, so
  Stage 3 has to be told which timeframe it is certifying rather than
  inheriting whichever was written last. `variants_tested_all_timeframes`
  records cells × timeframes; `variants_tested` alone understates the search by
  that factor. Stages 3 and 4 refuse a comma-separated `--tf` — a gate audit
  certifies one (parameters, timeframe) pair, and a lifecycle run writes one
  tear sheet per contract. Certifying at a timeframe whose winner was selected
  on different bars raises rather than proceeding.

**`backtest/baseline.py`** — **Stage 1**, the **Regime-Aware Screening
Firewall**. Version A (and B with `--ml`) on DEFAULT parameters, one simulation
per **(symbol, timeframe) configuration**, each profiled into four
volatility/trend quadrants and screened on the best quadrant rather than on the
blended sample. Survivors are written to `surviving_assets.json` scoped to the
one environment they cleared.

- Defaults, deliberately unswept: a sweep here would screen on the best of N
  per contract, promoting whichever symbol had the most parameters to hide
  behind. The comparison is meant to be between CONTRACTS.
- **Both versions run, over the charter window, by default (2026-08-21).**
  `--start`/`--end` default to `2013-01-01`..`2022-12-31` — reading past the
  end means screening on the Stage 3 holdout, which selects survivors on the
  bars Gate 3 later measures retention against — and `--ml` is ON here alone,
  because survival is decided on EITHER version and a Version-A-only screen
  cannot separate "neither version carried it" from "only one was asked".
  `--no-ml` declines it and reports NOT RUN everywhere. Which window ran and
  who chose it is written onto the handoff as `in_sample_window`.
- No gate table. Nothing at this stage is entitled to a gate verdict, and three
  lines of NOT EVALUATED under a heading teaches a reader to skip the gate
  table — the one thing they must not do at Stage 3.
- **The console is a progress line per configuration and a table of the
  winners.** Two lines each — `RUNNING` with the bar count, then `EVALUATED`
  with both profit factors and the verdict — because 27 contracts × 4
  timeframes is 108 scorecards, and printed in full the only reliable effect is
  that nobody reads the last one.
- **Everything else goes to `stage1_baseline_report.md`** in the pipeline
  directory: both versions' full metrics (Sharpe, Sortino, PF, win rate, max
  DD, net return, trades, friction costs), the four-quadrant regime matrix per
  version, the (now purely descriptive) day-of-week attribution, the
  entry-filter audit, and the drop reason for every configuration that failed.
  Written on EVERY run — a screen where nothing survived is the run whose
  detail matters most — and rewritten from scratch after each configuration, so
  one killed at 14 of 108 leaves a complete report of 14.
- **Survival is one QUADRANT, on either version** — `optimal_regime_PF >= 1.00`
  over `>= 30` trades in that same quadrant. Full detail in the firewall bullet
  below. The trade floor is far below Gate 1's 100 because this stage decides
  what is worth sweeping, not what is worth trading; it is not zero because a
  profit factor over eleven trades clears any bar by accident often enough to
  matter across a 108-cell screen. Admitting Version B is a deliberate loosening
  of the earlier Version-A-only rule: B's classifier is fitted on these same
  bars, so a Stage 1 survivor is no longer necessarily a contract with an
  unfiltered edge. Which version carried it is recorded in the row's `reason`,
  on the leaderboard, and in the report.
- **The handoff is exact pairs.** `surviving_pairs` is
  `[{"symbol": "NQ", "tf": "5m", "version": "A", "status": "PROMOTED",
  "optimal_regime": ..., "quadrant": "Q1", "regime_pf": ...,
  "regime_trade_count": ..., "regime_win_rate": ..., "regime_net_pnl": ...,
  "kill_switch_regimes": [...]}, ...]` and the printed Stage 2 command names
  only those symbols and timeframes. `surviving` is kept beside it as the
  symbol union, because that is what `scan.py` defaults `--symbols` to.
  `--symbols`/`--tf` are two axes, so ragged survivors can only be expressed as
  their cross product — a SUPERSET, and the stage says so rather than quietly
  handing Stage 2 a contract it just dropped.
- **The Regime-Aware Screening Firewall, from 2026-08-20.** This REPLACED the
  Drop Unprofitable Days contract, which is gone from this stage along with its
  `--no-drop-losing-days`, `--dow-min-pf` and `--dow-min-trades` flags. Both
  versions of every configuration are profiled by
  `backtest.profiler.RegimeProfiler` into four quadrants — ADX(14) > 25 is
  Trending, ATR(14) above the contract's OWN median is High Volatility — and
  each version's breakdown is written as
  `regime_profile_<SYMBOL>_<TF>_version_<a|b>.json` beside the handoff.
  - **Survival is one quadrant, not the blend.** `optimal_regime_PF >= 1.00`
    AND `optimal_regime_trade_count >= 30` **in the same quadrant**, on either
    version. **The bar was 1.15 until 2026-08-20, when it was lowered to 1.00
    by operator instruction.** The reason for 1.15 is not answered by that
    change, it is accepted: the quadrant is the best of four, so a bar cleared
    by a hair is a bar cleared by selection rather than by edge. At 1.00 a
    configuration advances on a quadrant that merely broke even. Stage 1 is
    therefore a wide net feeding the Stage 2 sweep, not a verdict about an
    edge — nothing downstream loosened, Gate 1 still binds PF at 1.00 on the
    BLENDED sample over >= 100 trades. `--min-profit-factor` raises it back,
    and the value used is printed in the banner and written onto the handoff. Both bars bind on ONE quadrant — a 1.90 over eleven
    trades beside a 0.90 over four hundred is a strategy with no environment,
    and pairing the best factor with the largest count would advance exactly
    that.
  - **This is a LOOSER screen than the one it replaced, deliberately.** A
    configuration whose blended profit factor is below 1.00 now survives on the
    strength of a single quadrant. The question changed: not "does this make
    money on every bar" but "is there an environment in which it does" — which
    is the honest question for a strategy governed by a live supervisor able to
    stand it down.
  - **The handoff is scoped, and the scope travels.** Each entry of
    `surviving_pairs` is exactly `{"symbol", "tf", "version", "status",
    "optimal_regime", "quadrant", "regime_pf", "regime_trade_count",
    "regime_win_rate", "regime_net_pnl", "kill_switch_regimes"}`.
    `version` is which twin cleared the quadrant — survival is decided on
    either, so a survivor with no version recorded is a pair nobody can
    reproduce. `quadrant` is the charter's `Q1`..`Q4` id, inverted from
    `mdlib.regimes` rather than spelled out again. A configuration that
    cleared nothing is written to `dropped` with `status: "DROPPED"`, a
    `reason`, an explicitly null `optimal_regime` and an empty kill switch;
    `screen_results` carries EVERY configuration evaluated in one shape, and
    is what the Discord card formats. The kill switch is DERIVED as the
    other three quadrants rather than measured: a quadrant that failed the bar
    and a quadrant the strategy never traded in are the same instruction to a
    supervisor, and reading "no evidence" as "permitted" is what puts a contract
    into the one environment nobody sampled. A configuration that cleared
    nothing gets `None` and `[]` — never a best-effort second place, because a
    stand-down list derived from a quadrant that failed is a live-trading
    instruction nothing certified.
  - **It is an in-sample selection layer, and the handoff says so.** The
    quadrant is picked on the same bars Stage 2 sweeps and Stage 3 certifies, so
    a winning Sharpe is the best of (combinations × this best-of-four pick).
    `regime_screen.selected_in_sample` records it; Gate 3's holdout retention is
    the only evidence it generalised.
  - **Stage 2 sweeps the WHOLE window, not the winning quadrant**
    (`regime_screen.applied_to_stage2: false`). Masking the sweep to a subset
    that was itself chosen as the best of four on those same bars would stack a
    second selection layer under the first. The regime is an instruction for the
    live supervisor, not a mask on the search.
  - **The day-of-week table survives as DESCRIPTIVE only.** It is still written
    per configuration and nothing downstream reads it. Which weekday loses is
    largely a restatement of which regime that weekday falls in, and pruning the
    calendar masked the environment instead of naming it.
  - **`--exclude-days` and `--news-filter` remain**, from the shared
    `add_filter_args`, spelled identically on all five stages and on `bt-run`.
    They are explicit operator instructions applied by the engine, not a pruning
    decision this stage makes.
  - **The report carries the four-quadrant matrix for every asset EVALUATED**,
    drops included, with all four regimes as rows — a quadrant with no trades is
    a row saying so, because an absent row reads as missing data when it means
    "this strategy never traded a low-volatility range". The VERDICT column
    names which bar a quadrant missed, since failing on edge and failing on
    sample size are fixed by different work.
- **`survivors_leaderboard` is the table the stage ends on** — Symbol,
  Timeframe, PF (A), PF (B), the optimal regime, its profit factor, its trade
  count and the version that produced it; survivors only, sorted by the REGIME
  profit factor descending. Sorted on the regime rather than on either blended
  factor because that is what the screen decided on: ranking on the blend would
  put a contract with a broad mediocre edge above one with a sharp edge in a
  single environment, which inverts the question the stage now asks. The kill
  switch is NOT a column — it is always the other three quadrants, and spelling
  it out per row pushes the table past 170 characters into a terminal wrap; it
  is printed in full per survivor in the REGIME FIREWALL block beneath the
  table, and carried in full on every handoff row.

**`backtest/audit_gates.py`** — **Stage 3**, and the only script that produces a
gate verdict anybody may act on. Runs the three separate pieces of evidence —
in-sample metrics, walk-forward + Monte Carlo, and the holdout — and writes
`gate_audit_<SYMBOL>.json`.

- **`check_windows` refuses an in-sample window that runs into the holdout**,
  before any bars are read. It also refuses an omitted `--is-end`, which runs
  to the end of the lake and eats the holdout. This is the one check that can
  invalidate everything else in the file.
- Parameters come from Stage 2's `best_params_<SYMBOL>.json`. A missing file is
  an ERROR, not a silent fall back to the defaults — `--defaults` is how you say
  you meant it. `variants_tested` travels with them onto the audit.
- **`_resolve_filters` inherits Stage 2's `exclude_days`** so the certification
  is of the week the parameters were actually selected on. Certifying the whole
  week when the winner was chosen with Monday masked out scores a strategy
  nobody optimised: Gate 1 disagrees with the sweep's own in-sample metrics, and
  Gate 3 measures retention between two different strategies. An explicit
  `--exclude-days` here still wins, and the source is recorded as
  `entry_filters_source`. **The news filter is deliberately NOT inherited** —
  its dates can be rule-generated and approximate, and an approximate blocking
  window must not creep into a gate verdict unasked.
- **`certification_leaderboard` is the table the stage ends on** — one row per
  (contract, VERSION), with Gate 1, Gate 2 and Gate 3 shown SEPARATELY, then the
  rolled-up `CERTIFIED` / `NOT CERTIFIED`, then the excluded days the gates were
  run on. The three gates are separate columns because they fail for different
  reasons and are fixed by different work, and because a `NOT EVAL` is a run
  that has not been done rather than a statement about the strategy — one
  PASS/FAIL column makes those indistinguishable. `FINAL STATUS` is
  `audit["passed"]`, so a NOT EVALUATED reads as NOT CERTIFIED, which is what
  Stage 5 enforces.
- The walk-forward runs with FIXED parameters unless `--wfo-grid` is passed,
  which is recorded as `wfo_optimized: false`; with nothing selected per fold
  the ratio compares two time periods rather than fitted-versus-unseen.

**`backtest/verify_full.py`** — **Stage 4**. The whole lifecycle in one run per
contract: tear sheets, the full trade log as CSV, and the cost drag. **Not a
certification, and it says so in the JSON** (`is_certification: false`) — this
window contains the Stage 3 holdout, so its metrics are in-sample by
construction and no gate table is printed. Cost drag is reported as a total,
per trade, and as a share of GROSS profit; the third is the one that decides
whether an edge is real, and it is `None` rather than `0%` when there is no
gross profit for costs to be a share of.

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

**`backtest/discord_reporter.py`** — the webhook notifier, and the only place
this repo posts anything to Discord. Two cards over one transport:
`--mode promotion` (the default, `--stage 5`) is the promotion scorecard, whose
values are passed in on the command line; `--stage 1` / `--mode baseline` is
Stage 1's regime-firewall leaderboard, read straight out of
`surviving_assets.json`.

- **It computes nothing and decides nothing.** The Stage 1 card prints the
  `status` Stage 1 recorded rather than re-applying the survival hurdle, so a
  card can never promote a configuration the stage dropped. A reporter that
  re-derived a profit factor would be free to disagree with the stage it is
  announcing, and the two would be compared by nobody.
- **The handoff is read through `pipeline.read_stage`**, so a file written by
  the wrong stage or belonging to another strategy is refused rather than
  posted. A Discord card is exactly the artifact nobody cross-checks.
- **A leaderboard is one fixed-width block in the embed DESCRIPTION**, not one
  field per row: Discord caps an embed at 25 fields and 6000 characters, and a
  full screen is 108 configurations. Rows past `STAGE1_MAX_ROWS` are COUNTED on
  the card — a silently shortened leaderboard reads as a complete one — while
  the Evaluated / Promoted / Dropped totals always describe the whole screen.
- **The QUAD column carries the `Q1`..`Q4` id the handoff recorded**, with a
  legend built FROM the rows. No short spelling of a regime name lives in this
  module: a second one would be free to disagree with `mdlib.regimes`, and a
  card naming the wrong environment is caught only in live trading.
- **The webhook URL is a credential** — never printed, never echoed into a
  failure message, only its host. `$BT_DISCORD_WEBHOOK` supplies it.

**`backtest/promote.py`** — **Stage 5**. Promotes one version into
`strategies/approved_incubator/<strat>/` and commits it. See the workflow below.

`--audit-file <gate_audit_SYMBOL.json>` is the Stage 3 certification and is the
**authoritative** gate verdict: the audit inside `dual_metrics.json` comes from
a single dual-version run, which can only ever evaluate Gate 1 — Gates 2 and 3
need a walk-forward, a bootstrap and the held-back years, which are separate
runs. Promotion is refused unless it says PASS (`--force` overrides and records
it). When both files are supplied the certification wins and any disagreement
is recorded rather than resolved silently. `meta.json` gains a `certification`
block — the audit's path, symbol, windows and SHA-256 — or the literal string
`NOT CERTIFIED`, which is written rather than omitted because an absent key
reads as a field nobody filled in. Certification is **not** required by
default, so the older `bt-run` workflow keeps working unchanged;
`--require-certification` turns its absence into a refusal.

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

**`strategies/`** — Signal logic only: take bars, return signal masks. No
cost handling, no session logic, no data access. `approved_incubator/` stages
strategies under evaluation; see its README for the required `meta.json`.

The loader (`agents/tier3_workers.load_strategy`) accepts either form:

```python
make_signal_fn(**params) -> signal_fn     # preferred when parameterised
signal_fn(bars) -> masks                  # when it takes none
```

`bars` is ONE symbol's DataFrame, oldest to newest. Return boolean Series
aligned to it, in one of the two shapes the engine accepts:

```python
(entries, exits)                                  # long only
(entries, exits, short_entries, short_exits)      # bidirectional
```

Neither is deprecated. A long-only strategy returns two — `ema_crossover` does,
and that keeps the compatibility path exercised by a real module rather than
only by a test. Anything else (a three-tuple, a bare Series) RAISES: silently
taking the first two masks of a three-tuple is how a strategy's short side
disappears into a plausible long-only equity curve. A module may declare
`TIMEFRAME`, `SYMBOLS`, and `DEFAULT_PARAMS`.

**A short is not a long with the sign flipped, and every layer has to know it.**
The places where a mirrored rule is wrong rather than merely unwritten, all of
them silent:

- The stop sits ABOVE the fill and the target BELOW it; the trailing stop
  ratchets DOWN, tracking the low-water mark since the fill. A short stop
  placed below the fill is breached by the fill bar itself.
- `gross_pnl` is `entry - exit` per contract.
- Slippage is charged on the side the order CROSSED, not on whether it opened
  or closed: buys are long entries and short exits, sells are long exits and
  short entries (`_cost_arrays`).
- The ML filter's training label is signed by side
  (`_label_baseline_trades(..., direction=)`). Labelling shorts with the long
  formula teaches the classifier the edge exactly inverted, and Version B comes
  back smooth and backwards. A bidirectional strategy gets one classifier per
  side, each trained only on its own completed trades.

### Every new strategy implements all four

The loader tolerates a module that declares none of the following — several
pre-existing ones do, and the tolerance is what keeps them loadable. It is not
permission to write another. **Any strategy Claude creates carries all four**,
because each one closes a specific way a result goes wrong silently:

```python
def signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, ...]: ...
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

A third optional declaration, added 2026-08-18, is **not** cosmetic — it
changes which entries Version B vetoes:

```python
def ml_features(bars, **params) -> pd.DataFrame:   # one row per bar, in order
    ...                                            # the matrix the classifier
                                                   # is fitted on
```

`load_strategy` binds it as `module_info["ml_feature_fn"]` and
`run_dual_version_backtest` hands it to `apply_ml_signal_filter` as `features=`.
**A module that declares none gets `None`, which selects the shared
`causal_features`** — so every strategy written before the hook existed keeps
the Version B it always had, bit for bit. `backtest/promote.py`'s generated
Version B wrapper binds it through the same `bind_ml_features`, because a
promoted Version B fitted on different columns from the Version B whose metrics
justified promoting it is the failure the sharing exists to prevent.

Three things travel with it. **Causality is the module's responsibility** — the
filter's guarantee that it trains only on trades closed before the candidate is
undone by a column that reads the future, and a scaler fitted on the whole
frame leaks the test period's distribution into the training rows without
tripping any shift-based audit. **Shape is checked and failures raise**: a row
count that disagrees with the bars, an empty matrix, or a hook that throws is
refused rather than being aligned or quietly fallen back to the default —
unlike `indicators`, which is wrapped, because a broken chart annotation must
not throw away a completed backtest and a silently swapped model must not
survive one. And **the columns are recorded** on the run as
`metrics["meta"]["ml_features"]`, `None` when Version B did not run at all:
two strategies filtered on different matrices have Version Bs that are not
comparable, and that fact has to travel with the numbers.
`strategies/experimental/sma_momentum_crossover.py` is the first user.

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

The sections run from what the module must declare through to how it is run and
what would settle it. Sections 2, 3 and 4 map onto the module's `LOGIC` block,
its `PARAM_GRID` and its `signal_fn`; sections 1 and 5 are the run.

```text
### STRATEGY SPECIFICATION & BACKTEST REQUEST

================================================================================
1. STRATEGY METADATA
================================================================================
- Strategy Name:        intraday_vol_mr
- Strategy Archetype:   Mean-Reversion
- Primary Timeframe:    15m
- Target Assets:        Multi: NQ,ES,CL,GC  (or ALL)

================================================================================
2. CORE CONCEPT & HYPOTHESIS (Plain English)
================================================================================
- Concept: <what inefficiency is harvested, and why it persists>

================================================================================
3. INDICATORS & PARAMETER GRID (VectorBT Scan)
================================================================================
- Indicators:            <one line each, with the exact period>
- Default Parameters:    <name = value>
- Parameter Search Grid (`PARAM_GRID`):   <name: [values]>

================================================================================
4. ENTRY & EXIT EXECUTION RULES
================================================================================
- Long Entry:            <condition>
- Short Entry:           <condition, or "Long Only">
- Take Profit:           <condition>
- Stop Loss:             <condition, or "none modelled">
- Session Rules:         <entry window, flatten time, in a named timezone>
- Execution Fill:        Next-bar open with contract-specific slippage & commission

================================================================================
5. BACKTEST EXECUTION CONTROLS
================================================================================
- In-Sample Period:      2015-01-01 to 2022-12-31
- Phase 3 Holdout Check: YES / NO   (must NOT overlap the in-sample period)
- ML Filter Comparison (Version B): YES / NO
- Run Mode:              Multi-Asset Runner (`bt-run`)
```

What each section is defending against, and where it has to be checked against
what the engine can actually do:

- **Section 2** is the part that cannot be recovered later. A strategy with no
  stated reason for existing is indistinguishable from one found by searching
  until something looked good, and the two fail differently in live markets. It
  becomes the module's `LOGIC["concept"]` close to verbatim.
- **Section 3** fixes the size of the search before it runs. `--scan` reports
  `variants_tested`, and a Sharpe read without knowing how many variants
  produced it is not a measurement.
- **Section 4 is where a request most often asks for something the engine does
  not have.** Check every line of it against the engine before writing the
  module, and state the gap in the module's docstring rather than approximating
  it silently:
  - **There are no stop or target ORDERS.** `from_signals` is driven by
    boolean masks — two for a long-only strategy, four for a bidirectional
    one — and fills them at the next bar's open. A stop can only be
    expressed as an exit signal, so it is detected on the bar that breaches it
    and filled one bar later — not at the stop price. Say so on the module, or
    every drawdown figure it produces will be read as something it is not.
  - **A trailing stop cannot be a stateless mask at all.** Its level depends on
    the high since entry, which depends on which bar opened the position. It
    needs a state machine inside the module (see
    `strategies/experimental/intraday_vol_mr.py::_walk`), and that machine must
    start its high-water mark at the FILL bar, not the signal bar.
  - **Session times must be converted through a named zone**, not a fixed UTC
    offset. `--flat-by-close` uses a fixed `session_close_utc`, so it is right
    in one half of the year and an hour off in the other; a module doing its
    own session logic should use `America/New_York` and say that the flag is
    then redundant.
- **Section 5's holdout must not overlap the in-sample period.** This is the
  one line in the template that can invalidate everything above it. Reserving
  the last three years after the run is not reserving them — by then they have
  been seen — and an in-sample window that runs into the holdout has spent it
  before Gate 3 is ever evaluated. Check the two date ranges against each other
  before starting.

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

Recalibrated 2026-08-16 for single-strategy account governance.

| Gate | Criterion | Threshold |
|---|---|---|
| **1 · In-Sample** | Sharpe | *informational — not a pass/fail condition* |
| | Profit factor | >= 1.00 |
| | Trades | >= 100, scaled to >= 30 per backtest year, capped at 200 |
| | Max drawdown | <= 12.0 % |
| **2 · Robustness** | WFO efficiency | >= 0.50 |
| | Monte Carlo 95% max DD | <= 18.0 % |
| **3 · OOS Holdout** | Holdout / IS retention | >= 0.80 (<= 20% degradation) |

**Sharpe no longer fails Gate 1.** It is still computed on daily closes, still
printed on every scorecard and tear sheet, and still the metric Gate 3
measures retention on — it sits on the Gate 1 table as an `INFO` row with no
threshold. A Sharpe floor rejects a strategy that makes money after costs and
never draws past 12% for having a lumpy return path, which is a complaint
about the shape of an equity curve rather than about whether there is an edge.
The row stays visible so a reader who remembers the old 1.20 can see it was
demoted rather than silently dropped; `_roll_up` skips `INFO` criteria, and
anything that filters a gate's `checks` on `status != PASS` must skip them too
(see `scan.py`'s `gate1_shortfalls`).

**The trade count is a floor that scales, and a cap that stops it.** 100 is the
minimum for any slice, however short — an out-of-sample window with 40 trades
cannot separate an edge from a run of luck. Past ~3.3 years the per-year rate
binds instead: 120 trades over 16 years is seven a year, and every ratio
computed from it is noise wearing two decimal places. Past ~6.7 years the
`max_required_trades` ceiling (200, added 2026-08-17) binds and the requirement
stops rising: unbounded, the rate demanded 503 trades of a 16.7-year lake run,
which stops being a significance bar — 200 trades already settles that — and
becomes a selectivity bar that fails a Version B whose ML filter did its job
and stood most of the baseline's trades down. Years are counted in SESSIONS
(`metrics_basis.n_days` / 252, falling back to `n_days`), the same way
`annualized_return_pct` counts them; a metrics dict carrying no day count is
held to the bare 100 and the criterion's note says so.

**Gate 3 measures retention on Sharpe, or on profit factor where that ratio is
undefined.** Demoting Sharpe from Gate 1 means a strategy with a non-positive
in-sample Sharpe now reaches Gate 3, where two negative Sharpes divide to a
healthy-looking positive number. Profit factor is what Gate 1 does bind on and
it is strictly positive, so it is the fallback. The criterion's LABEL names
whichever metric was used and `audit["retention_metric"]` records it — the two
are never reported under one heading. `audit["sharpe_retention"]` stays
Sharpe-only and NaN when undefined; `audit["retention"]` is what Gate 3
scored.

Gates 2 and 3 need evidence the dual run does not produce — a walk-forward, a
bootstrap, and the held-back final 3 years are separate runs. Pass them in via
`robustness={"A": {...}, "B": {...}}` and `holdout={"A": {...}, "B": {...}}`.
**Without them those gates report `NOT EVALUATED`, which is not a pass.**
`audit["passed"]` is True only when all three gates cleared on real numbers, so
nothing can be promoted on a gate that was never run. Drawdowns are compared on
magnitude — the engine signs them negative, and comparing raw would let -40%
clear a 12% limit.

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
(version, symbol, timeframe, params, `params_source`, the `risk` block, locked
metrics snapshot, gate statuses, SHA-256, timestamp), plus `baseline.py` for
Version B — and commits that directory alone.

- **`meta.json` records the RUN's parameters, not the module's.** `params` is
  the module's `DEFAULT_PARAMS` with the metrics snapshot's `meta.params`
  layered over it and any explicit `--params` on top; `params_source` says
  which layer won. That ordering matters as soon as `--scan` sweeps anything:
  the promoted Sharpe came from the winning grid cell, and recording the
  defaults beside it would describe a strategy nobody backtested — a stop
  distance that never ran, sitting next to metrics that assume one that did.
- **The `risk` block repeats the stop, target and trailing flag** under names
  a reader (and the CrossTrade governance layer) can find without knowing what
  a given strategy called its periods. Three states are kept distinct:
  `"NOT DECLARED"` — the strategy has no such parameter; `null` — it has one
  and this run modelled it off, which for `tp_atr_mult` means no take-profit at
  all; or the value.
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
  and forces the return to boolean. A strategy module that takes unpacked
  arrays is wrong, not merely unconventional. **Synthesis stays LONG ONLY** —
  the prompt still asks for the two-mask return, which the engine accepts
  unchanged. Extending it to four masks means teaching the model the short
  side's mirrored risk rules, and a model that emits shorts unprompted is
  strictly worse than one that does not.
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
