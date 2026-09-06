---
name: backtest-pipeline
description: "The five-stage pipeline: Stage 1 regime firewall, Stage 2 sweep charter, Stage 3 Gate R certification, Stage 4 lifecycle, Stage 4.5 the day-of-week gate, Stage 5 promotion, temporal chunking and the dual-version gate tables."
paths:
  - "backtest/baseline.py"
  - "backtest/scan.py"
  - "backtest/audit_gates.py"
  - "backtest/verify_full.py"
  - "backtest/dow_gate.py"
  - "backtest/pipeline.py"
  - "backtest/promote.py"
  - "backtest/run_pipeline.py"
  - "backtest/data_loader.py"
  - "tests/test_stage*.py"
  - "tests/test_pipeline_filters.py"
  - "tests/test_batch_runner.py"
  - "tests/test_temporal_chunking.py"
  - "tests/test_report_gates.py"
  - "tests/test_dow_gate.py"
  - "scripts/check_strategy_days.py"
---

# The five-stage pipeline

**`backtest/scan.py`** — **Stage 2**, and the vectorbt-native sweep behind
`--scan`. One module, two entry points: `scan_symbol` is the library the batch
runner calls, and `main()` is the stage that sweeps the in-sample window per
contract and writes `best_params_<SYMBOL>.json`. The stage adds NOTHING to the
search — same selection rule, same tie-break — because a second sweep
implementation in a CLI would be free to disagree with the one `--scan` uses
and the two would be compared by nobody. A
strategy declares `PARAM_GRID = {"ema_period": [15, 20, 30], ...}`; every
combination becomes a COLUMN of a single `vbt.Portfolio.from_signals` call, so a
27-cell grid costs roughly one backtest rather than 27. The winner is the best
**Sharpe plateau** among the combinations whose Gate 1 audit is PASS.

**The charter binds four things about this stage (2026-08-21), and all four are
enforced in the module rather than left to how the command was typed:**

- **Its input is Stage 1's handoff, as EXACT PAIRS.** With no `--symbols` and
  no `--tf`, `resolve_targets` sweeps the `(symbol, timeframe)` survivors
  `surviving_assets.json` names, via `pipeline.stage1_pairs`. It is not the
  cross product of the two unions: `--symbols` and `--tf` are independent axes,
  the survivors are RAGGED (NQ at 5m and 15m, GC at 15m only), and the product
  sweeps configurations the screen dropped. Naming `--symbols` gets those
  contracts at the timeframes each one survived at; naming `--tf` is an
  explicit override and crosses the axes. Either way a pair Stage 1 did not
  promote is swept **and flagged** — in the banner, in `unscreened_pairs` on
  the summary, and as `in_stage1: false` on its row — because a parameter set
  for an unscreened pair must not reach Stage 3 looking like a screened one.
  Each target carries its Stage 1 scope (version, quadrant, optimal regime,
  kill switch) onto `best_params_<SYMBOL>_<TF>.json` as `stage1_regime`.
  **It is transported, never applied**: the sweep runs the whole window, and
  masking it to a quadrant that was itself chosen as the best of four on these
  same bars would stack a second in-sample selection under the first
  (`regime_applied_to_sweep: false`).
- **The in-sample window is the charter's, and the holdout is not read.**
  `check_in_sample_window` refuses a window reaching `HOLDOUT_START`
  (2023-01-01) — and refuses an omitted `--end`, which runs to the end of the
  lake — **before a bar is loaded**, with no override flag. Stage 1 lets a
  deliberate re-screen through because a screen is a filter; an optimiser is
  not. Whatever it reads has been fitted to, so a Stage 2 run that touched the
  holdout leaves Gate 3 measuring retention on bars the winner was already
  chosen on, and nothing downstream can detect it. A flag that spent the
  holdout would be used, and it can only be spent once.
- **Nothing is pruned.** No contract, timeframe, quadrant or parameter set is
  eliminated here on an aggregate metric, and no prop-firm rule is applied —
  those are CrossTrade's, against a live balance. **Gate 1 is a ranking
  PREFERENCE and never a filter**: a grid where nothing clears it still
  produces a winner, still writes `best_params_<SYMBOL>_<TF>.json` and still
  advances, under a `selection` string that says plainly that nothing cleared
  the gate. Stage 3 is what certifies. The guarantee is written as data, not
  prose: `coverage` on the summary counts targets against optimised
  configurations, and a shortfall is a RUN FAILURE (a sweep that raised), never
  a screening result — the stage says so and exits 1.
- **The winner is a plateau, not a spike.** `plateau_scores` reads the Sharpe
  surface the one vectorbt call produced and scores every cell
  `min(own Sharpe, mean Sharpe of its grid neighbours)`. A neighbour differs on
  exactly ONE axis by one step in `axis_order` (numeric axes ascending with
  `None` — "no take-profit", the limit of an ever-wider target — at the far
  end; non-numeric axes keep their declared order, since there is no distance
  between `True` and `False` to sort by). The `min` is the point and a mean
  would defeat it: averaging a cell with its neighbours scores a spike's
  NEIGHBOUR highly, so the sweep would answer an overfit by promoting the cell
  beside it. A neighbour that never traded counts as 0.0 rather than being
  dropped — the hole IS the evidence that the space around the cell does not
  trade. Nothing about a market changes between a 20-bar mean and a 21-bar one,
  so a Sharpe that does is a property of this sample. `is_spike` (neighbours
  keep under `PLATEAU_SPIKE_RATIO`, 0.5, of the cell's Sharpe) is reported for
  every cell and, since 2026-08-25, is one of **two minimum robustness bars
  that decide eligibility**: an isolated spike and a cell whose in-sample
  drawdown reached `RUIN_MIN_DRAWDOWN_PCT` (-100%, `backtest/pipeline.py`, the
  same boundary Stage 3's ruin guard applies) cannot be selected as the winner,
  and a pair whose entire grid fails them is `PRUNED_FRAGILE` in
  `stage2_summary.json` **with no `best_params` file**, so Stage 3 cannot
  certify it. That is a departure from "Stage 2 prunes nothing", and a narrow
  one: the charter's rule is about aggregate PERFORMANCE — a Sharpe, a profit
  factor, a trade count — and neither bar is a performance judgement. One says
  the cell has no neighbourhood and the other says the account was ruined on
  the bars the parameters were chosen on. The bars are applied BEFORE Gate 1 is
  preferred, so a ruinous spike cannot be exported under a "GATE 1 PASS"
  heading. `fragile_counts` records how much of each grid they removed. A grid
  with
  one value per axis has no neighbours and the rank degenerates to Sharpe,
  which is correct: with nothing adjacent tested there is no evidence either
  way. `--select sharpe` restores the pre-charter single-best-cell rule, and
  the `selection` string names which ran.
- **The VERSION travels, and both later stages act on it.** Stage 1 screens
  Version A (rules) and Version B (the same signals with an ML confirmation
  filter) and a pair survives on EITHER, recording which in
  `surviving_assets.json` as `version`. Until 2026-08-25 that answer travelled
  as far as `stage2_summary.json`'s `stage1_version` column and then stopped:
  `--ml` was a single global flag an operator typed, so a pair only Version B
  cleared was certified as Version A unless somebody remembered. The failure is
  invisible in every output — the Stage 3 summary is complete, every gate is
  filled in, and the version column simply reads `A`. Now:
  `audit_gates.resolve_version_b` resolves it PER PAIR (`--ml` still forces B
  everywhere, a superset that cannot cause the bug; `--no-stage1-ml` is the
  override for a classifier that cannot be rebuilt, and it is recorded on the
  audit as `version_b_source`), and `run_pipeline` reads
  `surviving_assets.json` a second time to print the plan before Stage 2 and
  check after Stage 3 that every B survivor actually got a Version B row.
  **No global `--ml` is passed by the orchestrator** — it would run the filter
  over the Version A survivors too, which is the mirror image of the bug.
- **Stage 2 ranks on Version A and CONFIRMS Version B on the winner.**
  `ml_confirm_winner` re-runs the single winning parameter set with the filter
  on and writes both curves to `best_params` as `ml_confirmation`; a "no" is
  always recorded with its reason, never an absent key. It is not the grid,
  for two reasons: the sweep is one `vbt.Portfolio.from_signals` call over
  every combination as a COLUMN while Version B refits per completed trade and
  has no column form, so a 162-cell grid becomes 162 sequential backtests; and
  ranking the grid on the ML-filtered curve would select parameters against a
  classifier fitted on the same in-sample bars, stacking a second in-sample
  selection under the first — the exact reason the sweep is not masked to the
  Stage 1 quadrant either. `b_beats_a` here is IN-SAMPLE and labelled as such.
  Stage 3 is where Version B is certified, on the holdout.
- **Two files leave the stage beside the per-contract winners**:
  `stage2_summary.json` (through `pipeline.write_stage`, so `read_stage` can
  refuse the wrong strategy's) and `stage2_summary_matrix.csv`. Both carry
  every configuration the stage was ASKED to optimise, errors included with
  `params: "NOT OPTIMIZED"` — a shorter table reads as a complete one, and
  "the sweep raised" is a different statement from "the grid produced no
  measurable Sharpe". `rank_requested` and `rank` are separate fields: a
  `--reuse-scan` rebuild of a table written before the plateau columns existed
  can only be ranked on Sharpe however the sweep was invoked, and the Discord
  card prints the one that was APPLIED.

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
- **Ties break on the shallower drawdown.** Sharpe is the underlying metric
  because it IS the risk-adjusted return; ranking on raw return would pick
  whichever set took the most risk to get there. Ties are real once risk is
  swept — and MORE common under the plateau rank, which floors a whole
  neighbourhood at one number — a target no bar ever reaches and `tp_atr_mult=None` produce the same
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

**`backtest/data_loader.py`** — temporal chunking, added 2026-08-23 for
`--symbols ALL` sweeps that OOM. `iter_temporal_chunks(source, chunk_years=2,
warmup_bars=500, settlement_bars=0)` yields one contract's bars as
chronological calendar blocks, each carrying a warm-up tail from the block
before it. `source` is a DataFrame, a parquet path, a `LakeSource(symbol, tf)`
or `(symbol, tf)`; only the last two reduce the LOAD peak, because a frame the
caller already holds cannot be un-held.

- **The peak it removes is not the bars.** `mdlib.lake.iter_bars` already
  streams one symbol at a time. What kills a Stage 2 sweep is `scan_symbol`'s
  four stacked boolean masks — `4 x n_bars x n_combinations` bytes, resident
  before the first vectorbt call, so `_simulate_columns`' column batching does
  not touch them. 5.6M 1-minute bars against a 432-cell grid is 9.0 GiB; eight
  2-year chunks make it 1.1 GiB. `projected_sweep_bytes` is that arithmetic and
  the Stage 2 banner prints it.
- **THIS IS THE THING CLAUDE.md FORBIDS, AND IT IS PAID FOR RATHER THAN
  IGNORED.** `backtest/engine.py::_chunk_bounds` chunks bars EXACTLY, placing a
  boundary only where the strategy is flat. A data loader cannot: the
  boundaries are fixed before any signal exists. So `scan_symbol_chunked`
  carries a **settlement tail** (a trade opened near the payload end can close
  inside the same simulation), attributes every trade to the ONE chunk whose
  payload contains its ENTRY (so the concatenation is a partition, never a
  pile), and **counts what is still lost**: a trade open when its chunk's frame
  ends is invisible to `vbt.Portfolio.trades` and is reported as
  `chunking.truncated_trades`, never dropped in silence.
  `tests/test_temporal_chunking.py` pins that count against the trades actually
  missing from a contiguous sweep, exactly.
- **A chunked sweep is an APPROXIMATION; the engine's batching is not.** With
  an adequate tail the two agree column for column on the suite's fixture, and
  that is not guaranteed in general — a path-dependent strategy can resolve a
  boundary differently. `--chunk-years` is therefore OFF by default and the
  record travels onto `best_params_<SYMBOL>_<TF>.json` as `chunking`, so Stage
  3 can see that the parameters it is certifying were selected on one.
- **500 warm-up bars is not enough for this repository's own strategies.** A
  windowed indicator is exact once its window is full; a RECURSIVE one only
  decays its seed, by `(1 - alpha)^w`. `required_warmup_bars` computes the
  bars needed, and the tolerance is relative to the SEED ERROR — which is
  price-scale, because `ewm` restarts at the chunk's first value. A span
  EMA(50) warmed 346 bars still lands 9.4e-05 out in price units; a 200-bar
  trend EMA at the specified 500-bar default lands **0.75** out.
  `--chunk-warmup auto` sizes the window from the strategy's declared
  parameters and prints what it chose — and cannot see a length that is a
  module CONSTANT, which `double_rsi_macd_scalp_20260823`'s 200-EMA is.
- **The pad that completes a derived timeframe's final bucket stops at the
  requested `--end`.** 5m..4h are RESAMPLED from 1m and a resample labels each
  bucket at its start, so a read stopping at the last bar's LABEL builds that
  bar from one minute — one bar in 221,685 on NQ 15m, wrong in every column
  but `open`, at the end of the window where the last trade closes. The reader
  widens by one bar to cover it and **never past `--end`**: finishing a bucket
  across a holdout boundary would spend fifteen minutes of the holdout on a
  memory optimisation, and it can only be spent once. At the wall the final
  bucket therefore stays truncated, which is exactly what
  `backtest.run.load_bars` produces for the same window — the two paths see the
  same last bar.
- Boundaries are anchored on a GLOBAL grid of `chunk_years`-year blocks, not on
  the first bar, so two contracts with different histories are cut the same way
  and their chunked sweeps are comparable.

**`backtest/pipeline.py`** — not a stage; the contract BETWEEN them. Where each
stage writes, what the next reads, the banner that says which stage a log came
from, and the two dates every stage has to agree about. `CHARTER_IS_START` /
`CHARTER_IS_END` (2013-01-01..2022-12-31) and `HOLDOUT_START` (2023-01-01) live
here rather than in a stage because Stage 1 screens on that window, Stage 2
optimises on it and refuses to read past it, and Stage 3 measures retention
against the years after it — three copies would be one edit away from a Stage 2
sweep that runs a day into a holdout Gate 3 then scores as unseen.
`stage1_pairs` is the one place Stage 1's survivors are read back as exact
`(symbol, timeframe)` pairs with the regime scope each cleared; it is what
Stage 2 sweeps, and it exists so the ragged survivors are never flattened into
the cross product of two axes. `read_stage` refuses a file written by the wrong stage or belonging
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
that carries days, but nothing in the pipeline writes one any more. `STAGE2_SUMMARY_FILE` / `STAGE2_MATRIX_FILE` name Stage 2's summary matrix in
its two forms — the JSON handoff the Discord card reads, and the CSV a human
does. `STAGE3_SUMMARY_FILE` (`stage3_audit_summary.json`) is Stage 3's
equivalent: one file for the whole certification run, beside the per-contract
`gate_audit_<SYMBOL>_<TF>.json` files that remain the AUTHORITATIVE verdict a
promotion rests on — the summary is the index over them, never a replacement. `leaderboard(title, header, rows)` renders the end-of-stage table for all
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
  winners.** Three lines each — `RUNNING` with the bar count, `TIMING` with the
  seconds in load / simulation / profiling and the classifier's fits per side
  (a slow stage and a hung one are otherwise indistinguishable, which is
  exactly how the refit cadence bug read), then `EVALUATED`
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
- **A MODULE'S `TARGET_QUADRANTS` RESTRICTS THE DESIGNATION, from 2026-09-02.**
  When a module declares one (`keltner_trend_drift_20260901` declares
  `("Q3",)`), Stage 1 designates only from that set: the strategy is judged in
  the environment its premise is about rather than in whichever quadrant scored
  highest. **A configuration that misses the bars there is DROPPED, not
  re-homed.** Before this it was silently re-homed — 20 of 20 instances of the
  three trend-drift archetypes were certified into Q1 or Q2, the Q3-declaring
  module included, because a high-volatility quadrant swings larger dollars
  under a fixed-size engine (see the profiler's score bullet). The declaration
  decides which environment the strategy is JUDGED in and **never whether it
  passed**: every bar still binds on the declared quadrant exactly as on any
  other, including Stage 1's own sample floor of `max(50, 10%)` — which is NOT
  Gate R's holdout floor of 30, and a declaration that also relaxed it would be
  a way to certify on thinner evidence by writing a constant in a module. Both
  spellings are accepted (`("Q3",)` or `("Low Volatility / Trending",)`) and an
  unknown name RAISES rather than reducing to "no declaration", which would
  screen the module on dollar alpha exactly as if it had declared nothing. The
  handoff carries `declared_quadrants`, `designation_restricted` and
  `unrestricted_primary` — the last naming the quadrant that DID qualify and
  was deliberately not substituted, so a policy drop is never mistaken for "no
  quadrant cleared". A module declaring nothing keeps the unrestricted
  best-of-four unchanged. Stage 3 then reads the declared quadrant off the
  handoff natively, with no `--regime` override.
- **Survival is one QUADRANT, on either version** — `optimal_regime_PF >= 1.00`
  over `>= max(50, 10% of placed trades)` trades in that same quadrant, with
  **positive net P&L** there, and among the quadrants that clear, the one
  DESIGNATED is the one with the highest **alpha score** (`net P&L × profit
  factor`). See the True Home Regime Discovery bullet below; the rule lives in
  `backtest/profiler.py::designate` and Stage 1 adds none of its own. Full detail in the firewall bullet
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
gate verdict anybody may act on. Locks the parameters Stage 2 selected, runs
them once over the untouched holdout, and writes `gate_audit_<SYMBOL>_<TF>.json`
per configuration plus `stage3_audit_summary.json` over the run.

**The Regime-Switching Incubator Charter binds this stage from 2026-08-21, and
it changed what a certification IS.** The verdict is no longer the roll-up of
Gates 1, 2 and 3. It is **Gate R**, and all six clauses are enforced in the
module rather than left to how the command was typed:

- **The verdict is Gate R — the edge, out of sample, inside ONE quadrant.**
  `regime_gate` reads the holdout's four-quadrant profile and scores only the
  `optimal_regime` Stage 1 designated: `MIN_REGIME_PROFIT_FACTOR` (1.00) over
  `MIN_REGIME_TRADES` (30) in that quadrant, **imported from
  `backtest.baseline`** so a screen and a certification can never be held to
  different numbers. Both bars bind on the SAME quadrant. Three outcomes are
  kept apart because they are fixed by different work: a FAIL on the count
  (including a quadrant the strategy never traded out of sample, which the note
  says outright rather than reporting as a loss), a FAIL on the factor, and
  **NOT EVALUATED when no quadrant was designated — which is not a pass.** The
  quadrant is an INPUT: re-picking the best of four on the holdout would make
  Gate R a selection made on the bars it exists to be unseen evidence about,
  and almost anything clears 1.00 given four attempts. A quadrant name outside
  `profiler.REGIMES` RAISES — left alone it reads as zero trades, which is a
  broken handoff reported as a strategy that stopped trading.
- **Gates 1, 2 and 3 are computed in full, reported in full, and CANNOT fail a
  certification.** They score the BLENDED sample across every market state, and
  a strategy whose live supervisor stands it down outside its quadrant never
  trades that sample — failing it there prunes on a result nobody will realise.
  `charter_audit` folds Gate R in, sets `status`/`passed` from it alone, and
  keeps the pre-charter roll-up verbatim as `aggregate_status` so the verdict
  reads as MOVED rather than quietly dropped. **A configuration can now be
  CERTIFIED with a failing Gate 1. That is the intended effect.** The audit's
  shape is unchanged, so `promote.load_gate_certification` and Stage 5 need no
  change — what moved is what `status` means.
- **`[REGIME STARVATION]`, from 2026-08-21.** When Gate R fails on the TRADE
  COUNT, `regime_starvation` prints and records
  `[REGIME STARVATION] Quadrant {Qn} ({regime}) had only {n} holdout trades.
  Candidate was dominant in {Qn} ({regime}) in sample.` "Gate R FAIL" reads
  identically whether the edge died or the strategy simply never entered its
  own environment again, and those are fixed by completely different work —
  the second is the common failure of a best-of-four in-sample pick and is
  invisible on the gate table, because a quadrant with one holdout trade
  prints a profit factor of 999 and a PASS on the factor row. It is keyed on
  the COUNT check rather than the overall status for that reason. The dominant
  quadrant is read from the Stage 2 handoff's `regime_scores`, **never
  re-derived from the holdout** — naming a new quadrant off the holdout is
  exactly the best-of-four selection Gate R exists to avoid, and the
  diagnostic must not smuggle one in through a print statement. With no scored
  table on the handoff it says the dominance cannot be stated rather than
  guessing.
- **The target is parsed dynamically.** `target_regime` reads Stage 2's
  top-level `optimal_regime` / `target_quadrant` first, falls back to the
  nested `stage1_regime`, then to the Stage 2 summary row — so a handoff
  written before the designation was lifted to the top level still certifies,
  and one carrying only a `Q1`..`Q4` code resolves through
  `regime_for_quadrant`. When the top-level and nested copies DISAGREE the top
  level wins and the disagreement is recorded in `target_regime_source` rather
  than resolved silently. A name outside `REGIMES` still RAISES.
- **The parameters are LOCKED.** Targets come from `stage2_summary.json` as
  exact `(symbol, timeframe)` pairs via `stage2_targets`, not from a glob of
  `best_params_*.json` — a superseded sweep's winner sits in that directory
  indistinguishable from a current one. `--param` still overrides, because an
  operator correcting the record outranks a file, but it BREAKS THE LOCK and
  says so: `params_locked: false` with every overridden key named. Rows Stage 2
  recorded as ERROR are carried through as skips with a reason, never dropped —
  a Stage 3 input shorter than the Stage 2 output turns "the sweep never ran"
  into "this was certified and failed".
- **`--holdout-end` defaults to the PRESENT**, not to a hardcoded year. A fixed
  end silently stops certifying against the newest bars the moment a year rolls
  over, and the verdict looks identical either way. `check_windows` accepts an
  open-ended holdout for that reason and still refuses an open-ended or
  overlapping IN-SAMPLE window. The window defaults now come from
  `pipeline.CHARTER_IS_START` / `CHARTER_IS_END` / `HOLDOUT_START` rather than
  from three more copies of the same dates.
- **No prop-firm rule reaches a verdict.** `_assert_no_prop_firm_rules` REFUSES
  a config carrying `trailing_drawdown_pct` or `daily_loss_limit` and records
  the absence on the audit. The stage never sets them, so the check reads as
  paranoia; it is written down because what it guards is invisible — a trailing
  drawdown set here cuts the equity curve short and changes nothing else on the
  console.
- **`retention_scores` is clause 5, and it is REPORTED, never scored.** Profit
  factor, Sharpe, max drawdown and win rate, in-sample against holdout. The
  drawdown ratio is inverted (`|in-sample| / |holdout|`) so above 1.00 means
  "held up" on every row — a raw `oos/is` would score a strategy that drew down
  twice as deep at 2.00 and sort it to the top. A ratio is `None`, never 0.0,
  when either side is missing or the denominator is zero.
- **A pass is sealed and staged.** `seal_and_promote` writes
  `strategies/approved_incubator/<strategy>/` through `backtest.promote.promote`
  — reused rather than reimplemented, so Stage 3 and Stage 5 cannot disagree
  about what was promoted — and adds a `seal` block to its `meta.json`: SHA-256
  of the strategy code, of `best_params_<SYMBOL>_<TF>.json` and of
  `gate_audit_<SYMBOL>_<TF>.json`. Three hashes because they can be separated:
  the same code under a different winning cell is a different strategy with the
  same code checksum. A missing artifact reads `"NOT AVAILABLE"` rather than
  being omitted. **Nothing is git-committed** — `commit=False`, always; the
  commit and the four-choice menu stay Stage 5's, in front of a human.
  `--no-promote` declines the staging, and a refusal is RETURNED as an error row
  rather than raised, so a completed certification is not thrown away because
  staging hit a read-only checkout.
- **`stage3_audit_summary.json`** is the CAMPAIGN's handoff, written through
  `pipeline.write_stage` and read by `discord_reporter.py --stage 3`. It
  computes NOTHING — every value is transcribed from an audit this run already
  wrote, so the index can never disagree with the verdicts it indexes. Errors
  and skips are rows with `status: "NOT AUDITED"`, which is deliberately not
  `FAIL`: "the run broke" and "the edge did not generalise" must not share a
  token.
- **It MERGES across timeframes rather than overwriting (2026-08-21).** This
  stage certifies ONE timeframe per invocation, so a campaign at 5m, 15m and
  30m runs it three times into that one file; as a plain overwrite it kept only
  the last, and the Discord card — which reads the summary and not the
  directory — announced one timeframe while the other certifications sat on
  disk as `gate_audit_<SYMBOL>_<TF>.json` and reached nobody. `merge_stage3_rows`
  carries every OTHER timeframe's rows forward verbatim and lets this run
  replace its own, so a re-certification that no longer covers a contract
  cannot leave the earlier verdict standing beside the new ones. `runs` records
  what each invocation covered and `coverage` is summed over them; `timeframes`
  is the campaign and `timeframe` is still the last run, as two separate fields
  rather than one that changes meaning. `audits` is the consolidated index over
  the per-pair files — path, SHA-256 and the verdict inside each — which remain
  AUTHORITATIVE; the list says where they are so nothing has to glob a
  directory where a superseded sweep's audit is indistinguishable from a
  current one. Each row also carries `regime_starvation`, the message or
  `None`, because a Gate R FAIL on the trade count and one on the factor are
  fixed by completely different work.
- **`--rebuild-summary` reconstructs that index from the audits on disk.**
  Stage 2's `--reuse-scan` for Stage 3, and for the same reason: the expensive
  half of the stage is already written and the index is the cheap half. It
  reads NO bars, re-scores NO gate, and takes each timeframe's windows from the
  audits themselves rather than from whatever flags were typed while rebuilding.
  Only the SUFFIXED files are read — the unsuffixed `gate_audit_<SYMBOL>.json`
  duplicates whichever timeframe ran last. It exists to recover the campaigns
  the old overwrite already flattened.
- Audits are written per PAIR (`gate_audit_<SYMBOL>_<TF>.json`) as well as to
  the unsuffixed name Stage 5's documented command uses. Certifying NQ at 30m
  after certifying it at 15m would otherwise replace the 15m verdict with no
  trace; the replacement is now announced and the per-pair file survives it.

- **`check_windows` refuses an in-sample window that runs into the holdout**,
  before any bars are read. It also refuses an omitted `--is-end`, which runs
  to the end of the lake and eats the holdout. This is the one check that can
  invalidate everything else in the file.
- Parameters come from Stage 2's `best_params_<SYMBOL>_<TF>.json`. A missing
  file is an ERROR, not a silent fall back to the defaults — `--defaults` is how
  you say you meant it, and it reports NO lock rather than claiming one.
  `variants_tested` travels with them onto the audit.
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
  (contract, VERSION): the target `QUAD`, `GATE R (OOS REGIME)` with the
  quadrant profit factor and trade count it was measured on, then Gate 1,
  Gate 2 and Gate 3 SEPARATELY, then the rolled-up `CERTIFIED` /
  `NOT CERTIFIED`, then the excluded days the gates were run on. The quadrant
  travels with the profit factor because an OOS PF with no quadrant beside it
  is a blended number under a regime-gated verdict. The three advisory gates
  stay separate columns because they fail for different reasons and are fixed
  by different work, and because a `NOT EVAL` is a run that has not been done
  rather than a statement about the strategy — one PASS/FAIL column makes those
  indistinguishable. `FINAL STATUS` is `audit["passed"]`, now Gate R alone, so
  a NOT EVALUATED reads as NOT CERTIFIED, which is what Stage 5 enforces.
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
gross profit for costs to be a share of. The stage ends by printing the
`discord_reporter.py --stage 4` command with `--artifacts` naming THIS run's
directory: the snapshots that card reads live in a timestamped directory, and
one built from the wrong one would describe a different run.

- **NET FRICTION PER REGIME QUADRANT, from 2026-08-24.** `friction_by_regime`
  breaks trades, gross, costs and net P&L out by the quadrant of each trade's
  ENTRY bar, prints the four-row table beside the cost drag, and writes it into
  `dual_metrics_<SYMBOL>.json` as `friction_by_regime` — into the snapshot
  itself, because that is the file a promotion cites and the Stage 4 card
  reads, and a friction figure living anywhere else is one nobody has beside
  the metrics it qualifies. The blended cost share answers "did this pay too
  much"; a regime-gated strategy trades ONE quadrant, so the certified quadrant
  can hand 80% of its gross to the broker while the blend reads 35% on the
  strength of three quadrants no supervisor will permit. **The labels are READ
  from the frame `RegimeProfiler` classified for this same run**, never
  re-derived — a second pass would be free to disagree with the
  `regime_profile_<SYMBOL>_<TF>.json` written beside it, the same trade counted
  in Q1 by one artifact and Q2 by the other with both tables still summing to
  the same totals. All four quadrants are always rows, including untraded ones,
  and warm-up trades in NO quadrant are counted and named rather than left to a
  table that quietly sums to less than the trade list. `cost_share_pct` follows
  `cost_drag`'s rule exactly: `None`, never `0.0`, where gross was not
  positive.
- **`--slippage-atr-mult`, from 2026-08-24, and OFF by default.** The engine's
  slippage was never a static dollar assumption — `_cost_arrays` charges
  `ticks × that bar's tick size / price`, per symbol and per bar, and picks up
  a contract whose tick changed mid-history. What a constant tick count misses
  is that the spread is not constant through TIME, which flatters exactly the
  high-volatility quadrants this pipeline certifies into. The flag charges
  `M × ATR(N)` per side instead, converted to a per-bar tick count at each
  bar's own tick size and floored at one tick — ATR is NaN through its own
  warm-up, and a NaN reaching the engine makes the fill price NaN and drops the
  trade from the P&L with nothing raising. **It is opt-in because turning it on
  changes every P&L figure**: an ATR-scaled run is not comparable with a
  constant-tick one, and every existing certification here was measured on the
  constant. Which model ran travels on the snapshot as `slippage_model`.
  Implemented as a per-bar `slippage_ticks` COLUMN on the bars, read by
  `_cost_arrays` — on the bars rather than on the config because `_simulate`
  hands that function `bars.iloc[lo:hi]`, so a config array would have to be
  sliced in step with every chunk boundary and a drifted slice would charge
  each bar another bar's slippage, wrong in every trade and invisible in a
  total. A non-finite or negative value in that column RAISES.

**`backtest/dow_gate.py`** — **Stage 4.5**, added 2026-09-06 at operator
instruction. Profiles every configuration Stage 2 left parameters for by
WEEKDAY, names the worst session, runs a counterfactual with that weekday's
entries suppressed, and writes the verdict where the live loop can read it.
Runs on Stage 4's own window (`--start`/`--end` are passed by the orchestrator,
not defaulted) so the two stages describe the same bars.

- **IT PRUNES NOTHING, and the rule is a FUNCTION rather than an absence.**
  `promotion_gate()` returns `promote: True` for every verdict, with the
  criterion as a string on the row. Everything reaching this stage advances to
  Stage 5 whatever the weekday table said. What the stage produces is an
  INSTRUCTION for the live supervisor — the same division Stage 1's regime
  firewall draws, and for the same reason: the weekday is chosen in-sample on
  bars Stage 3 already spent, and pruning on it would drop configurations Gate
  R certified on the strength of a calendar. `selected_in_sample: true` and
  `is_certification: false` are on every file it writes.
- **This RE-INSTATES as a LIVE GATE what Stage 1 demoted to descriptive on
  2026-08-20.** The Drop Unprofitable Days contract was removed because which
  weekday loses is largely a restatement of which REGIME that weekday falls
  in, and pruning the calendar masked the environment instead of naming it.
  Nothing about that reasoning is retracted: the difference is that this
  verdict never touches the SEARCH — no sweep is masked, no gate is scored on
  it, nothing is dropped — it only stands a promoted pair down live.
- **The worst weekday is always IDENTIFIED; it is BLOCKED only when its
  expectancy is negative.** Two answers, deliberately kept apart. Five
  profitable weekdays have a worst one too, and blocking it removes realised
  edge in exchange for nothing — "fifth of five" is not evidence against a
  session. `--block-worst-always` overrides that and is recorded as
  `block_rule: "worst_always"`; what it must not be is the silent default.
  Collapsing the two fields would make "every weekday made money"
  indistinguishable from "the stage did not run".
- **The rank is expectancy ascending, ties on win rate ascending, then on the
  largest share of the deepest drawdown.** Expectancy is the mean NET P&L per
  trade — the metric a trade decision is made on and the one the
  counterfactual moves — and it is computed ONCE under that name rather than
  twice beside `avg_pnl`, which is arithmetically the same number.
- **A weekday below `--min-trades` (default 20) is NEVER ranked**, however
  badly it scored, and is listed in `below_floor` with its count. That floor
  is `backtest.report.losing_weekdays`', for its reason: a weekday holds a
  fifth of the sample and condemning one on eight trades is precisely how a
  day-of-week filter manufactures an in-sample Sharpe. A run where nothing
  reaches the floor blocks nothing and says so.
- **The grouping is `report.day_of_week_breakdown`'s, imported.** That function
  already owns the two decisions that change the answer — attribution by the
  ENTRY, keyed on the CME SESSION date — and a second grouping here would be
  free to disagree with the day-of-week table Stage 1 and Stage 4 already
  print, with both summing to the same totals. The DAILY realised-return table
  beside it is keyed on the EXIT date, because that is when a return is
  realised and that is the index `BacktestResult.returns` carries; the two are
  reported side by side and never reconciled, and the decision is taken on the
  ENTRY table because the entry is what the gate acts on.
- **`max_dd_contribution` is a share of the run's DEEPEST episode, not a
  per-weekday drawdown.** A weekday has no equity curve of its own, and five
  drawdowns over five disjoint slices are five numbers that sum to nothing.
  The episode is located on the trade-by-trade equity curve in REALISATION
  order and every trade inside it is attributed to its ENTRY weekday. Shares
  can total more than 100% because profitable weekdays inside the episode
  offset the losing ones; that is printed rather than normalised away.
- **The counterfactual is Version A, over the SAME bars, differing only in
  `cfg.exclude_days`.** `ml=False`: the question is about the calendar, and a
  classifier refitting per completed trade would double the stage to answer
  it. **The report carries the candidate count, never the trade count alone** —
  a filter removes candidate TRIGGERS and the walk holds one position at a
  time, so declining an early trigger can leave the strategy flat for a later
  one it would have been holding through, and a trade count that went UP is
  not evidence the filter failed to bind.
- **Two files leave the stage.** `dow_gate_<SYMBOL>_<TF>.json` per pair is the
  AUTHORITATIVE verdict and `stage45_dow_summary.json` is the index over them,
  the division Stage 3 draws between its audits and its summary. The per-pair
  name carries the TIMEFRAME and there is **no unsuffixed form**: a blocked
  weekday is a fact about one (symbol, timeframe) pair, and an unsuffixed file
  would hold whichever timeframe ran last while every promotion cited it —
  each `meta.json` carrying a plausible weekday with two of them wrong. The
  summary MERGES across timeframes for the same reason Stage 3's does.
  `pipeline.stage45_blocked_days` is the one place that mapping is read back,
  keyed per pair exactly as `stage1_exclude_days` is. The handoff's stage id is
  the INTEGER `pipeline.STAGE45` (45) rather than 4.5, because `write_stage`
  stamps `int(stage)` and a float would truncate to 4 and make every Stage 4.5
  file satisfy a Stage 4 `read_stage` check; `STAGE_LABELS` is what spells
  "4.5" in the banner.
- **The orchestrator runs it under `check=False`**, unlike Stages 1-4. Those
  are a chain — a stage that failed has not written the handoff the next one
  reads. This one is not: nothing downstream needs it to exist, and a
  day-of-week profiler that raised must not discard certifications that
  already cleared Gate R. The affected packages record
  `day_of_week_gate.status: "NOT EVALUATED"`, which is what actually happened.

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

**These three are the `bt-run` dual-version audit, and since 2026-08-21 they no
longer decide a Stage 3 CERTIFICATION.** `backtest/audit_gates.py` folds in
**Gate R** — profit factor >= 1.00 over >= 30 trades inside the one quadrant
Stage 1 designated, measured on the holdout — and takes its verdict from that
alone; Gates 1, 2 and 3 are computed and reported there as evidence and cannot
fail a certification. Everything in this section is unchanged for `bt-run`,
which evaluates Gate 1 and reports the other two as NOT EVALUATED. See the
`backtest/audit_gates.py` entry above for why the verdict moved and what is
kept on the file so the move is legible (`aggregate_status`).

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
- **`day_of_week_gate` is Stage 4.5's verdict, and it is what the live loop
  acts on.** `strategies/approved_incubator/<id>/meta.json` is the only file
  `realtime/live_dispatcher.py` reads about the calendar, so the blocked
  weekday has to land there; a verdict left in the pipeline directory is one
  the dispatcher would have to go looking for. `load_dow_gate` finds the
  per-pair file at the conventional path, `--dow-gate` names one explicitly,
  and the orchestrator passes it explicitly so a relocated `--out-dir` cannot
  silently record NOT EVALUATED on a pair that was profiled. **THREE STATES,
  kept apart exactly as the `risk` block keeps its three**: `status: "NOT
  EVALUATED"` (Stage 4.5 did not run for this pair — written, never omitted),
  `blocked_weekdays: []` under `EVALUATED` (it ran and every session cleared),
  and `[4]`. An unreadable verdict is `UNREADABLE` and blocks nothing, as a
  warning rather than a refusal: a certification that cleared Gate R is not
  thrown away because a day-of-week artifact was truncated. **The key spelling
  is shared with `live_dispatcher.DOW_GATE_KEY`** — a rename on one side alone
  turns the gate off silently, the block simply never found and every log line
  reading correctly.
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

## Commands

```bash
# THE FIVE-STAGE PIPELINE. Each stage prints the next stage's command and
# stops; nothing chains automatically, because the point of the stages is that
# a human reads the evidence between them. Handoff files live in
# <BT_ARTIFACTS>/pipeline/<strategy>/ - see backtest/pipeline.py.
python3 backtest/baseline.py    --strat X --symbols ALL --tf 15m \
    --start 2013-01-01 --end 2022-12-31       # 1: drop PF<1.0 contracts
python3 backtest/scan.py        --strat X            # 2: optimise the survivors
# STAGE 2 ONLY: with no --symbols and no --tf it sweeps Stage 1's EXACT
# surviving (symbol, timeframe) pairs, and --start/--end default to the charter
# window. An --end reaching 2023-01-01 is REFUSED, with no override: Stage 2
# fits what it reads, so a holdout it has optimised over is not a holdout.

# Stages 1 and 2 take a comma-separated --tf and evaluate each in turn.
# The lake derives 5m/15m/30m/1h/2h/4h from the 1m parquet, so nothing
# resamples in the stage. Stages 3 and 4 take ONE timeframe.
python3 backtest/baseline.py --strat X --symbols ALL --tf 1m,5m,15m,30m \
    --start 2013-01-01 --end 2022-12-31
python3 backtest/audit_gates.py --strat X --tf 15m            # 3: certify
# STAGE 3 ONLY: with no --symbols it certifies the EXACT (symbol, timeframe)
# pairs stage2_summary.json optimised, with the parameters LOCKED. The windows
# default to the charter's, and --holdout-end defaults to the PRESENT rather
# than a hardcoded year. The VERDICT is Gate R: profit factor >= 1.00 over
# >= 30 trades inside the ONE quadrant Stage 1 designated, measured on the
# holdout. Gates 1-3 are computed and reported as EVIDENCE and cannot fail a
# certification - nothing is pruned on a blended-sample metric and no
# prop-firm rule is applied. A pass is SHA-256 sealed and staged into
# strategies/approved_incubator/, never git-committed.
python3 backtest/audit_gates.py --strat X --tf 15m --no-promote   # certify only
python3 backtest/audit_gates.py --strat X --tf 15m \
    --regime-min-pf 1.25 --regime-min-trades 50   # tighten Gate R's bars
python3 backtest/verify_full.py --strat X --tf 15m \
    --start 2010-01-01 --end 2026-01-01       # 4: tear sheets + cost drag
python3 backtest/dow_gate.py    --strat X --tf 15m \
    --start 2010-01-01 --end 2026-01-01       # 4.5: the day-of-week gate
# STAGE 4.5 ONLY: names the worst weekday, runs the counterfactual, and
# writes dow_gate_<SYMBOL>_<TF>.json for stage 5 to put into the promoted
# meta.json. It PRUNES NOTHING - every configuration advances - and it
# certifies nothing: the weekday is chosen IN-SAMPLE, on a window that spans
# the stage 3 holdout. The worst session is always NAMED and is BLOCKED only
# when its expectancy is negative; --block-worst-always blocks it regardless
# and says so on the handoff. --min-trades is the floor a weekday must clear
# to be ranked at all (default 20).
python3 backtest/dow_gate.py --strat X --tf 15m --min-trades 40
python3 backtest/dow_gate.py --strat X --tf 15m --block-worst-always
python3 backtest/promote.py --strat X --version A --source <module.py> \
    --audit-file /mnt/backtest/artifacts/pipeline/X/gate_audit_NQ_15m.json \
    --dow-gate /mnt/backtest/artifacts/pipeline/X/dow_gate_NQ_15m.json  # 5

# WHICH WEEKDAYS EACH PROMOTED PACKAGE MAY ACTUALLY TRADE, and whether its
# meta.json still agrees with the stage 4.5 verdict on disk. meta.json is
# written at PROMOTION time and never revisited, so a package promoted before
# stage 4.5 ran trades the session it was stood down from with every log line
# reading correctly. Read-only; exits 1 on a DISAGREEMENT only.
python3 scripts/check_strategy_days.py            # or: days-check
python3 scripts/check_strategy_days.py --strat X --json
```

```bash
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
# HOW OFTEN VERSION B REFITS. Default 0.10; 0.0 is the pre-2026-08-24
# refit-on-every-completed-trade rule, bit for bit, and is ~60x slower.
python3 backtest/baseline.py --strat X --symbols NQ --tf 15m --ml-refit-growth 0.0
```

```bash
# TEMPORAL CHUNKING, for a sweep that will not fit in RAM. OFF by default.
# The peak of a Stage 2 sweep is NOT the bars - it is the four stacked signal
# masks, 4 x n_bars x n_combinations bytes, allocated in full before the first
# vectorbt call (so --max-cells cannot reduce them). One contract of 1m bars
# over 16 years against a 432-cell grid is 9.0 GiB. The banner prints that
# projection and names a --chunk-years when it exceeds --memory-budget-gib;
# nothing is chunked automatically, because a chunked sweep is an
# APPROXIMATION and switching it on silently would make two runs of the same
# grid incomparable.
python3 backtest/scan.py --strat X --symbols NQ --tf 1m --chunk-years 2
python3 backtest/scan.py --strat X --chunk-years 2 --chunk-warmup 2000 \
    --chunk-settlement 4000        # explicit warm-up and settlement tail
```
