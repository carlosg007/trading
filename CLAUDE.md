# CLAUDE.md

Autonomous quantitative research environment for **Pure Alpha Discovery** across
continuous futures. Code lives here; the 16-20 year Databento data lake lives on
a hard-mounted NFS at `/mnt/backtest`, never in the repo.

**The failure mode this project exists to avoid is an overfitted backtest that
looks right and fails in live markets.** Almost every rule below is a defence
against curve-fitting or silent data corruption.

Scoped detail lives in `.claude/rules/` and loads when you open the matching
files — the engine contract, the five pipeline stages, the Discord cards, mdlib
and the lake, realtime/live execution, portfolio routing, strategy modules,
agents. This file is only what applies everywhere.

---

## STRICT TOOL BOUNDARY

**NEVER execute `backtest/run.py`, any pipeline stage, or any full backtest from
inside Claude Code.** It burns enormous token volume and takes the human out of
the loop at exactly the point they must be in it.

Your five responsibilities, in full:

1. Write strategy modules and infrastructure code.
2. Run fast static checks and the AST security validator.
3. Run unit tests on small synthetic fixtures.
4. Commit to git.
5. Print the exact CLI command — flags, symbols, timeframe, date window spelled
   out — for the operator to run in their own shell, then **stop**.

This covers any run against the real lake and any multi-symbol run, foreground
or detached. **`--bg` does not satisfy the boundary**: the run still belongs to
the session, its log still streams into context, and the operator still is not
at the console when the gate audit prints. The four-choice promotion menu exists
so a human sees the evidence before anything is promoted.

## Division of labour

- **Claude Code** — multi-file editing, git, `uv`, AST verification, fixing
  runtime errors, optimising vectorized Vectorbt Pro / Numba loops.
- **Gemini synthesizer** (`agents/tier1_master.synthesize_strategy_code`) —
  single-shot vectorized signal logic from a hypothesis. It does not choose
  symbols, size positions, or interpret results.
- **Deterministic compute** (Python / Numba / Vectorbt Pro / SciPy) — 100% of
  the mathematical metrics. **LLMs never do math.** If a number reaches a human,
  a deterministic function produced it.
- **CrossTrade NAM** (Windows, outside this repo) — real-time account
  governance: contract sizing, daily loss limits, trailing drawdown, prop-firm
  challenge state, enforced against a live account balance.

**Model-generated code passes an AST check before import** — no file, network or
OS access, no `eval`/`exec`/`__import__`/`open`. Never bypass that gate to "just
try" a generated strategy.

**Push back on requests that compromise rigor.** Removing transaction costs,
bypassing the 3-year OOS holdout, or introducing lookahead bias are critical
flaws to reject.

---

## Separation of concerns

Alpha discovery and account governance are different problems that fail
differently. **Do not add prop-firm balance math to the research path, and do
not treat its absence as a bug.** If a constraint depends on a live account
balance or a funding program's rulebook, it belongs to CrossTrade.

Legacy surfaces still in the tree — `BacktestConfig.trailing_drawdown_pct` /
`.daily_loss_limit`, `check_trailing_drawdown()`, `compliance_rules/*.json`,
`evaluate_compliance()` — are retained deliberately as the CrossTrade
specification. `daily_loss_limit` is dead code. An empty `BacktestResult.breach`
was never evidence of compliance. Do not delete these as part of an unrelated
task.

---

## Findings that are not recoverable from the code

- **pytest reports a false green on the collector-style suites.** They record
  failures by appending to a module-level `FAILURES` list via `check(name, ok)`
  and never call `assert`; only `sys.exit(1)` in `main()` signals failure, and
  pytest never calls `main()`. Bare pytest therefore watches checks fail and
  reports all green. `tests/conftest.py` exists **because of this** — it excludes
  those suites from normal collection and runs each as a subprocess asserting its
  exit code, which also fixes cross-suite state interference and 59 collection
  errors. `pytest tests/` is the gate (~16 min); run a single suite as a script
  while iterating. **Never bypass conftest's routing.**
- **On ASSERT-based dual-mode suites, pytest collects any module-level `test_*`
  it can call — including a helper whose only argument is defaulted.** In
  `test_regime_profiler.py` that ran the sections directly, without their
  `$BT_ARTIFACTS` redirect, writing real JSON onto the NFS mount. The tell is the
  test count: 8 passed where 4 were written. Name inner sections `_check_*`.
- **A filter can only remove candidate TRIGGERS, never realised trades.** The
  walk holds one position at a time and ignores a trigger arriving while one is
  open, so declining an early trigger leaves the strategy flat for a later one it
  would have been holding through. Measured: enabling `use_vwap` removed 17
  candidates and **added 11 realised entries**. **Never use a trade count to
  decide whether a filter binds** — compare candidates or the trade list.
- **The live daemon's ADX comparator is `>`, not `>=`.** The written spec says
  `ADX >= 25`; `mdlib/regimes.py` and `backtest/profiler.py` have always used
  `>`, and every cached quadrant, Stage 1 designation and Gate R verdict was
  drawn on it. Matching the prose in the daemon alone would move the boundary
  there and leave the backtests behind. At ADX exactly 25.00000 the daemon says
  Ranging.
- **`theta_vol` is LOADED from the pinned 2013-01-01..2022-12-31 in-sample
  anchor, never computed live.** It is written into each regime parquet and read
  back through `provenance()`. A median taken over the caller's window is a
  property of the REQUEST, not the contract: on a quiet morning every bar reads
  high-volatility and a Q1-certified strategy is handed permission for a market
  it is not in, with a plausible equity curve behind it. **An anchor is per
  (symbol, TIMEFRAME)** — NQ is 7.90 at 15m and 11.33 at 30m, a 43% different
  boundary on the same tape.
- **Regime quadrant numbering is `mdlib/regimes.py`'s and nothing else's**:
  1 High-Vol/Trending, 2 High-Vol/Ranging, 3 Low-Vol/Trending, 4
  Low-Vol/Ranging, **0 = UNDEFINED** (indicator warm-up; `NaN > 25.0` is False,
  so the naive encoding files every warm-up bar under Low-Vol/Ranging). 0 is not
  a quadrant. **`config/portfolios.json` schema 1.0.0 numbered these
  incompatibly — every digit named a different environment** — and 1.1.0
  relabels them to agree; `canonical_quadrant` is the only supported resolver. A
  live daemon and a backtest disagreeing about what Q1 means is invisible
  downstream: the strategy is stood down in the environment it was certified for
  and turned loose in the one it never traded, with every log line reading
  correctly.
- **Trades are attributed by their ENTRY bar / ENTRY session, never the exit or
  a later bar** — regime quadrant, friction, and day-of-week alike. The live loop
  acts on the **last CLOSED bar** while the engine fills at the **next bar's
  open**; that timing difference is a real live-vs-backtest attribution gap, so
  never read a live fill and a backtest fill as the same event.
- **Costs are mandatory in every test, from the first one.** Variant rankings
  change once commissions and slippage (1 tick each way) are applied.
- **A strategy is only valid if it survives the 3-year holdout** (2023-01-01
  onward), untouched during optimization. The NT8 tree is **not** that gate — it
  is a thin cross-feed sanity check spliced on NinjaTrader's own roll rules.
- **Report the search.** Carry `variants_tested` into every result. A Sharpe read
  without knowing how many variants produced it is not a measurement.
- **Cross-sectional by default.** A daily strategy on ES alone over 16 years is
  100-200 trades. Symbols are never blended into one equity curve — each contract
  is its own simulation on its own multiplier, tick size and commission.
- **Intraday work respects `intraday_start_year`**: pre-2013 1-minute data is
  sparse for ten symbols, so a 30m bar built from those minutes behaves
  differently even though volume reconciles against daily bars.

---

## Data and infrastructure

- **NFS is mounted `hard`, not `soft`** — a soft mount returns an I/O error on
  timeout, which silently truncates a read mid-backtest. Do not change it, and
  do not chown/chmod the mount (NFSv3, no idmapping; the bogus UID is expected).
- **The DuckDB catalog stays on local disk** (`~/.local/share/mdlib/`). NFSv3
  with `local_lock=none` sends file locking over NLM, which is not safe for
  DuckDB. The catalog is only an index and is rebuildable.
- **Never write a parquet file above the `year=` partition level** or DuckDB
  will duplicate bars. `scripts/validate_lake.py` checks this.
- `/mnt/backtest/raw/` is immutable so the lake can be rebuilt after a parser bug
  without re-downloading.
- **Never read raw data into context.** Do not `cat`/`head` `.parquet`, `.csv`
  or large logs from `/mnt/backtest`; use small Python diagnostic scripts to
  inspect schemas, shapes or sample rows.

## Environment

- **`uv` exclusively.** Python 3.13.14 in one venv at `~/src/trading/.venv`. No
  `pip install`, no secondary venvs, and **never a venv on the NFS mount**.
- `streamlit` is pinned deliberately — the current release resolves `pyarrow`
  down to 24.0.0, and 25.0.1 is what the lake reader needs.
- **There is no linter config and no build step.** Scripts are run directly.
- **Always `import vectorbtpro as vbt`**, never `vectorbt`. Its API differs
  significantly; inspect with `dir()`/`help()` rather than guessing methods.
  Open-source `vectorbt` is uninstalled so the import fails loudly, but
  `riskfolio-lib` (pinned, imported nowhere) declares it as a dependency, so a
  reinstall can pull it back — check `uv pip list | grep -i vectorbt` after one.

```bash
bt-run     # = .venv/bin/python3 ~/src/trading/backtest/run.py
bt-status  # = .venv/bin/python3 ~/src/trading/backtest/status.py
bt-check   # = .venv/bin/python3 ~/src/trading/backtest/check_progress.py
bt-progress  # the same tool
nt8-check  # = .venv/bin/python3 ~/src/trading/realtime/check_nt8_feed.py
feed-status  # the same tool
signal-check # = .venv/bin/python3 ~/src/trading/realtime/check_live_signals.py
live-signals # the same tool
firewall-check # = .venv/bin/python3 ~/src/trading/realtime/check_trade_firewall.py
trade-gate   # the same tool
crosstrade-check # = .venv/bin/python3 ~/src/trading/realtime/check_crosstrade_connection.py
ct-check     # the same tool
             # (all ten from deploy/shell/trading_helpers.sh)
```

## Working style

- **Atomic git commits**, granular and single-purpose, so a script that breaks
  the server rolls back without losing an unrelated data fix. **Scope every
  commit with an explicit pathspec** (`git commit -- <paths>`) — a pathspec on
  `git add` does not scope the commit, and pre-staged files ride along silently.
- Search with `grep`/`rg` before reading whole files; no unprompted refactoring.

## Known gaps

- `mdlib/lake.py` has no `source` parameter, so the NT8 tree is unreachable
  through the reader.
- Several symbols (PL, grains, LE, FX, crypto, micros) are UNVERIFIED in
  `backtest/specs.py` — pull definitions before backtesting them. SI definitions
  stop at 2016, CL at 2025-12.
- `data_pull/coverage_summary.py` is a superseded copy of the `scripts/` version.
- LightGBM is referenced by the Dual-Version Mandate but is not pinned.
- `agents/` is largely scaffold; unimplemented functions raise
  `NotImplementedError` rather than returning empty results.
