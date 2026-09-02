---
name: backtest-engine
description: "Engine execution contract, cost model, profiler quadrant designation, specs, reports, the batch runner, entry filters and the memory guard."
paths:
  - "backtest/engine.py"
  - "backtest/profiler.py"
  - "backtest/specs.py"
  - "backtest/report.py"
  - "backtest/report_html.py"
  - "backtest/run.py"
  - "backtest/event_calendar.py"
  - "backtest/memory_guard.py"
  - "backtest/status.py"
  - "tests/test_engine_*.py"
  - "tests/test_clean_signals.py"
  - "tests/test_risk_params.py"
  - "tests/test_daily_metrics.py"
---

# The engine and the per-run machinery

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

**`backtest/profiler.py`** — the four-quadrant breakdown, and **TRUE HOME
REGIME DISCOVERY** (2026-08-21): the one place a strategy's home quadrant is
chosen. `designate(breakdown, total_profiled)` is shared by the profiler,
Stage 1's screen (`baseline.best_quadrant`) and, through the handoff, by Gate
R — before this each ranked on profit factor with its own trade floor and its
own tie-break, so a `regime_profile_*.json` and the `surviving_assets.json`
written beside it could name DIFFERENT home quadrants for the same run with
nothing raising.

- **The score is ALPHA CONTRIBUTION, `net P&L × profit factor`**, not the
  per-trade edge. A 1.55 factor over 45 trades and a 1.28 over 4,000 are both
  real and the second is the engine; ranking on the factor scoped survivors to
  the first and sent Gate R to certify a corner of the window. `SCORE_PF_CEILING`
  (10.0) caps the factor **for scoring only** — the profiler's 999 sentinel
  means "gross loss was zero", not a measured factor, and `net × 999` ranks an
  unbeaten 51-trade quadrant two orders of magnitude above a 4,000-trade book.
  A capped row says so (`pf_capped`); the reported `profit_factor` is left as
  measured.
- **Three bars, all on the SAME quadrant**: profit factor >=
  `DESIGNATION_MIN_PROFIT_FACTOR` (1.00), **positive net P&L**, and at least
  `designation_floor(total) = max(DESIGNATION_MIN_TRADES, DESIGNATION_MIN_TRADE_FRACTION × placed)`
  = `max(50, 10%)`. Positive net P&L is not decoration: `net × PF` is monotone
  only above zero, and at a factor of 0.00 — a quadrant with no winning trade —
  the product is exactly 0.0 and would outrank a quadrant that lost $5,000 at
  0.50. Requiring it removes the inversion from the selection path rather than
  patching the formula, and every quadrant is still scored and reported.
- **The score has a MEASURED DIRECTION, and `--score-mode vol_normalized` is
  the alternative.** The engine is fixed-size (`BacktestConfig.contracts = 1`,
  `size_type="amount"`), so per-trade P&L moves with the size of the move.
  Measured 2026-09-02 across 6E/6J/ES/NQ/GC/CL: Q3's mean ATR is **0.28x Q1's**
  (0.18x on NQ) and Q3 holds **0.67x** the bars, so at an EQUAL profit factor a
  Q3 quadrant scores about **0.19x** a Q1 one — it has to reach **PF 1.74** to
  outscore a Q1 running 1.20, before Gate R has looked at anything. That is why
  **20 of 20** instances of the three trend-drift archetypes were designated
  into a high-volatility quadrant, `keltner_trend_drift_20260901` included — a
  module that declares `TARGET_QUADRANTS = ("Q3",)`. `vol_normalized_score`
  divides net P&L by the quadrant's **own average absolute trade**
  (`avg_trade_abs_pnl`, measured from the same trades the profit factor is),
  which expresses the quadrant's expectancy in R-multiples and removes the
  scale. **NOT `theta_vol`** — that is one scalar per (symbol, TIMEFRAME), the
  boundary rather than a per-quadrant statistic, so dividing all four by it
  leaves the ranking exactly as it was. **`alpha` REMAINS THE DEFAULT**: every
  strategy in `config/portfolios.json` was designated under it, and a switched
  default would leave the live registry's quadrants and the rule that produced
  them disagreeing with nothing raising. BOTH scores are computed and reported
  on every row whichever one sorts, the handoff records `score_mode`, and
  `would_designate` names what the other rule would have picked — `None` when
  they agree, so a disagreement is never buried in an always-populated field.
  Neither mode moves a designation BAR; the rule decides the order only.
- **The floor scales.** 50 is meaningless once a run places 5,000 trades — a
  quadrant holding 1% of the sample is a corner of the window, not an
  environment. `total_profiled` is the trades PLACED in a quadrant, never the
  trade list: an unplaced trade (outside the frame, or inside the 14-bar
  warm-up) belongs to no quadrant, so counting it raises every quadrant's bar
  on trades no quadrant could claim.
- **This is NOT Gate R's floor.** `baseline.MIN_REGIME_TRADES` stays 30 and is
  still what `audit_gates` imports. They answer different questions — is there
  enough in-sample evidence to NAME a home regime, versus did the named one
  still trade out of sample — and a holdout is shorter than the window that
  chose the quadrant, so holding it to the designation floor would fail
  configurations for the length of the holdout.
- **`primary` is None when nothing clears**, which is a finding rather than a
  missing value. There is deliberately no fall back to the best of a bad set:
  the quadrant becomes Gate R's certification target and a live supervisor's
  permission to trade, and neither may be derived from a quadrant that lost
  money or was measured over twenty trades.
- **`secondary_regimes` are the other POSITIVE-expectancy quadrants**, each
  carrying why it was not designated — eligible-but-outscored, and profitable
  but below the sample floor, kept distinct because they are fixed by
  different work. They are metadata for the live supervisor and **never a
  second certification target**: two permitted quadrants give Gate R two
  chances at a 1.00 holdout profit factor, which is the best-of-N selection
  the single-quadrant rule exists to prevent.
- **`regime_scores` is the whole scored table**, all four quadrants including
  the losing ones, with `eligible` and a `reason` on each. A ranking that
  dropped them would make "disqualified on sample size" and "never traded" the
  same absent row. It travels onto `surviving_assets.json`, onto
  `best_params_<SYMBOL>_<TF>.json` (top level AND nested under
  `stage1_regime`), and is what Stage 3's REGIME STARVATION diagnostic reads.
- Ties break on the larger trade count, then on the declared regime order —
  the better-evidenced claim rather than whichever the quadrant order put
  first. `--min-trades` and `--min-trade-fraction` on Stage 1 move both bars,
  and the values used are printed in the banner and written onto the handoff.

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
- **`--ml` is opt-in.** Version B refits its classifier as the pool of completed
  trade; across many symbols that is hours. A skipped Version B is reported as
  NOT RUN in the scorecard, the snapshot and the leaderboard — never as a
  Version B that scored nothing, because a comparison that was not made is not
  one the baseline won.

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

**`backtest/memory_guard.py`** — watch system RAM and this process's RSS, and
stop a long run deliberately rather than letting the kernel stop it. Added
2026-08-23. `MemoryGuard.enforce(context)` is called at three places:
`data_loader.iter_temporal_chunks` (every chunk boundary), `scan.py` (every
GRID CELL, which is where the masks are actually allocated), and `baseline.py`
(before each configuration).

- **Four tiers, all `>=` so a threshold written as 90 fires AT 90.** OK below
  75%; WARN collects garbage; THROTTLE collects and sleeps 1s; HALT raises
  `MemorySafetyException`. HALT is also triggered by process RSS alone, which
  is the independent trigger for a single runaway sweep on an otherwise idle
  box.
- **The halt trigger is SYSTEM-WIDE, so somebody else's process can stop your
  sweep.** That is the intended trade — the OOM killer is system-wide too, and
  picks its victim by a score this process does not control — but a halt is NOT
  evidence the run was the problem. Every message carries `rss_gib` beside
  `sys_mem_pct` so the two can be told apart.
- **`max_rss_gib` defaults to 24.0 and is INERT on this VM**, which has 24.9
  GiB: the 90% system threshold fires near 22.4 GiB used machine-wide, long
  before this process alone reaches 24. The default is the specification's and
  is right on a bigger box; `MemoryGuard.for_this_machine()` derives one that
  binds.
- **WARN-level `gc.collect()` is paced** (`gc_interval_s`, 2s) because the
  sweep reaches its guard once per grid cell — 432 a chunk — and a collection
  on a heap of million-row frames costs O(100ms). THROTTLE and HALT are never
  paced. The FIRST warn always collects.
- **It exits 75 (EX_TEMPFAIL), never 137.** 137 is 128 + SIGKILL, what the
  shell reports when the OOM killer has actually killed a process — the
  precise outcome this module prevents. Exiting 137 after halting cleanly would
  tell every log scraper that the thing we avoided is what happened.
- **Both runners RE-RAISE a halt past their `except Exception` handlers.** That
  handler is right for a missing spec or an empty slice of the lake — one bad
  contract must not end a 108-configuration screen — and exactly wrong here:
  swallowing a halt moves to the next configuration, which allocates as much,
  on a machine that is no emptier. Stage 2 flushes
  `stage2_partial_<SYM>_<TF>.json` first (counts and spans, never the trade
  frames — writing hundreds of megabytes when the box is out of memory is how a
  graceful halt becomes an ungraceful one); Stage 1 needs no flush because it
  already rewrites its report after every configuration.
- **It does NOT throttle worker threads**, though the objective asked for it.
  The heavy allocation is inside vectorbt / numba / BLAS, whose pools are sized
  at import; setting `OMP_NUM_THREADS` afterwards does not resize an existing
  pool, so the call would look like throttling and change nothing. The lever
  that does exist is `--chunk-years`: fewer bars per chunk is fewer bytes per
  step.
- **`BT_MEMORY_GUARD=off`** disables every tier, read once per guard at
  construction. A loaded workstation must not fail a test suite over its own
  browser, and an operator running at 95% deliberately should say so in the
  command they typed.

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

## Commands

```bash
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
#   --ml       also run Version B. OFF by default: the classifier refits as
#              the closed-trade pool grows (see the refit cadence below), and
#              27 symbols of that is still long. A skipped
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
```

```bash
# The entry filters. Available on all five stages AND on bt-run, spelled
# identically because they come from one add_filter_args().
bt-run --strat X --symbols NQ --news-filter --news-window 30
bt-run --strat X --symbols NQ --exclude-days 0,4     # no Mon/Fri ENTRIES
```

```bash
# What macro calendar a news-filtered run would actually use, and whether its
# dates are published or rule-generated. Reads no bars; costs nothing.
python3 backtest/event_calendar.py --start 2015-01-01 --end 2026-01-01
```
