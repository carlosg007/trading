# The Backtest Pipeline — Stages 1 through 5 (including 4.5)

Reference for rebuilding or reasoning about the strategy-certification pipeline.
Everything here is drawn from `.claude/rules/backtest-pipeline.md` (the
authoritative scoped rule for this subsystem) and verified against the current
source in `backtest/*.py` and `realtime/live_dispatcher.py`. Where a claim
comes only from the rules doc and was not independently re-derived from code
in this pass, it is marked "per rules doc, not re-verified here."

The failure mode every stage exists to prevent: **an overfitted backtest that
looks right and fails in live markets.** The pipeline's design is a sequence
of increasingly strict defenses against curve-fitting and against selection
effects invisible in a single number.

---

## 1. Overview — data flow

```
Stage 1  baseline.py        Regime-Aware Screening Firewall
            |  surviving_assets.json  (exact (symbol, timeframe, version) pairs,
            |                          each scoped to the ONE quadrant it cleared)
            v
Stage 2  scan.py             Parameter sweep / plateau selection
            |  best_params_<SYMBOL>_<TF>.json, stage2_summary.json
            v
Stage 3  audit_gates.py      Gate R certification (out-of-sample, in-quadrant)
            |  gate_audit_<SYMBOL>_<TF>.json, stage3_audit_summary.json
            |  -> seals a PASS into strategies/approved_incubator/<strat>/
            v
Stage 4  verify_full.py      Full-lifecycle tear sheets + cost drag (in-sample by construction)
            |  dual_metrics_<SYMBOL>.json, HTML reports
            v
Stage 4.5 dow_gate.py        Day-of-week live-trading instruction (prunes nothing)
            |  dow_gate_<SYMBOL>_<TF>.json, stage45_dow_summary.json
            v
Stage 5  promote.py          Promotion into strategies/approved_incubator/, git commit
            |  meta.json (certification block, day_of_week_gate block, risk block)
            v
      realtime/live_dispatcher.py reads meta.json and trades (or stands down) live
```

Nothing chains automatically. Each stage prints the next stage's command and
stops — the point of the staging is that a human reads the evidence between
stages. Handoff files live in `<BT_ARTIFACTS>/pipeline/<strategy>/`
(`backtest/pipeline.py`), one directory per strategy, written atomically
(temp file + `os.replace`).

Two dates bound every stage and live in `backtest/pipeline.py` rather than
being copied into each one: `CHARTER_IS_START` / `CHARTER_IS_END`
(2013-01-01 .. 2022-12-31, the in-sample window) and `HOLDOUT_START`
(2023-01-01, the untouched 3-year holdout). Stage 1 screens on the in-sample
window, Stage 2 optimizes on it and refuses to read past it, Stage 3 measures
retention on the years after it.

---

## 2. Stage 1 — `backtest/baseline.py` — Regime-Aware Screening Firewall

**Purpose.** Decide which (symbol, timeframe) configurations are even worth
optimizing, without letting a parameter sweep hide behind a symbol's own
degrees of freedom. Runs Version A (rules-based) and, by default, Version B
(same signals plus an ML confirmation filter) on **default, unswept
parameters** — deliberately, because sweeping here would screen on the best
of N per contract and reward whichever symbol has the most parameters to
hide behind. The comparison is meant to be between contracts, not between
parameter sets.

**Inputs.** Lake bars for the requested symbols/timeframes (`--symbols ALL`
or an explicit list, `--tf` comma-separated), over the charter in-sample
window by default (`--start`/`--end` default to 2013-01-01..2022-12-31).
Reading past `HOLDOUT_START` here means screening on the bars Gate 3 later
measures retention against.

**Mechanism.** Each (symbol, timeframe, version) is run once and profiled by
`backtest.profiler.RegimeProfiler` into four quadrants — **ADX(14) > 25 is
Trending, ATR(14) above the contract's own median is High Volatility** (note:
the live daemon's comparator is also `>`, not `>=`; a bar with ADX exactly
25.00000 is Ranging). Survival is decided **per quadrant, not on the blended
sample**: `optimal_regime_PF >= 1.00` (lowered from 1.15 on 2026-08-20 by
operator instruction) over `>= max(50, 10% of placed trades)` trades in that
same quadrant, with positive net P&L there — on **either** version. Among
quadrants that clear, the designated one is the highest **alpha score**
(`net P&L x profit factor`). A strategy module can restrict its own
designation with `TARGET_QUADRANTS` (e.g. `("Q3",)`); a configuration that
misses the bar there is **dropped, not re-homed** into whichever quadrant
scored highest (this replaced silent re-homing, which had certified 20/20
trend-drift-archetype instances into Q1/Q2 including the Q3-declaring
module).

This stage carries **no gate table** — nothing here is entitled to a gate
verdict; three lines of "NOT EVALUATED" teach a reader not to look for one
until Stage 3.

**Outputs.**

| File | Contents |
|---|---|
| `surviving_assets.json` | `surviving_pairs`: exact `{symbol, tf, version, status, optimal_regime, quadrant, regime_pf, regime_trade_count, regime_win_rate, regime_net_pnl, kill_switch_regimes}` per survivor. `surviving`: the symbol union (what `scan.py` defaults `--symbols` to when none is named). `dropped`: every configuration that cleared nothing, with a `reason`. |
| `regime_profile_<SYMBOL>_<TF>_version_<a\|b>.json` | Full four-quadrant breakdown per version. |
| `stage1_baseline_report.md` | Full metrics for both versions (Sharpe, Sortino, PF, win rate, max DD, net return, trades, friction), the regime matrix, day-of-week attribution (now purely descriptive), entry-filter audit, drop reasons. Rewritten from scratch after each configuration so a killed run still leaves a complete partial report. |

**Kill switch.** `kill_switch_regimes` is **derived** as "the other three
quadrants," not measured — a quadrant that failed the bar and one the
strategy never traded in are the same instruction to a live supervisor.

**Gate/pass criteria.** Per-quadrant PF >= 1.00 over >= 30 trades (Stage 1's
own floor is `max(50, 10%)`, which is *not* Gate R's holdout floor of 30).
`--min-profit-factor` / `--min-trades` override.

**Handoff to Stage 2.** `pipeline.stage1_pairs` reads `surviving_assets.json`
back as exact `(symbol, timeframe)` pairs with each one's regime scope
(version, quadrant, optimal regime, kill switch) attached — this is what
Stage 2 sweeps by default.

---

## 3. Stage 2 — `backtest/scan.py` — Parameter sweep / plateau selection

**Purpose.** Optimize each Stage 1 survivor's parameters, on the in-sample
window only, without silently drifting past the holdout or masking the
sweep to an in-sample-chosen quadrant (which would stack a second
in-sample selection under the first).

**Inputs.** Stage 1's `surviving_assets.json`, as exact `(symbol, timeframe)`
pairs (not the cross product of `--symbols` and `--tf`, which are independent
axes over a ragged survivor set — e.g. NQ at 5m and 15m, GC at 15m only).
A pair Stage 1 did not promote can still be swept explicitly, but is flagged
(`in_stage1: false`, `unscreened_pairs`) so it can never reach Stage 3 looking
like a screened one.

**Mechanism.** A strategy declares `PARAM_GRID = {...}`; every combination
becomes a **column** of one `vbt.Portfolio.from_signals` call (a 27-cell grid
costs roughly one backtest, not 27; columns are batched via `MAX_CELLS` so
peak RAM tracks bars x columns-per-batch). The winner is the best **Sharpe
plateau**, not the best single cell: `plateau_scores` scores every cell
`min(own Sharpe, mean Sharpe of its one-step grid neighbours)` — the `min` is
deliberate, since averaging would score a spike's neighbour highly and
promote the cell beside it. Two minimum robustness bars, since 2026-08-25,
gate *eligibility* before Gate 1 preference is applied (this is the one
departure from "Stage 2 prunes nothing" — both are non-performance checks):
`is_spike` (neighbours must stay within `PLATEAU_SPIKE_RATIO` = 0.5 of the
cell's own Sharpe) and a ruin guard (`RUIN_MIN_DRAWDOWN_PCT` = -100% in-sample
drawdown). A pair whose entire grid fails both is `PRUNED_FRAGILE` in
`stage2_summary.json` with **no** `best_params` file, so Stage 3 cannot
certify it.

**Gate 1 here is a ranking preference, never a filter** — a grid where
nothing clears PF >= 1.00 / Sharpe informational still produces a winner
under `selection: "HIGHEST SHARPE · NO COMBINATION CLEARED GATE 1"`.

The sweep runs the **whole in-sample window**, never masked to the winning
quadrant (`regime_applied_to_sweep: false`) — the quadrant is transported
onto the output as `stage1_regime`, never applied as a filter on the search.

**Entry filters** (`--exclude-days`, `--news-filter`) are applied inside the
sweep itself (since 2026-08-19), inherited from Stage 1's per-pair
`exclude_days` unless overridden explicitly (`resolve_exclude_days`
precedence, with provenance recorded).

**Version B.** Stage 2 ranks the grid on Version A and, on the single winning
cell only, re-runs with the ML filter on (`ml_confirm_winner`) — never ranks
the grid itself on B, because B's classifier is fitted on the same in-sample
bars and would stack a second in-sample selection under the first. This is
labelled **in-sample**; Stage 3 is where B is certified, on the holdout.

**Outputs.**

| File | Contents |
|---|---|
| `best_params_<SYMBOL>_<TF>.json` | Winning parameter set, `stage1_regime`, `entry_filters`, `chunking` (if used), `variants_tested`, `ml_confirmation`. |
| `stage2_summary.json` | Every configuration asked to optimize, errors included (`params: "NOT OPTIMIZED"`), through `pipeline.write_stage`. |
| `stage2_summary_matrix.csv` | Same, as a human-readable table. |

**Gate/pass criteria.** None that prune (see robustness bars above for the
one exception). Reports `coverage` (targets vs. optimized configs); a
shortfall is a **run failure**, not a screening result — exits 1.

**Multi-timeframe note.** On a multi-timeframe run Stage 2 writes one
suffixed file per timeframe and **no unsuffixed file** — Stage 3 must be told
which timeframe it certifies rather than inheriting whichever ran last.

**Temporal chunking** (`backtest/data_loader.py`, `--chunk-years`, off by
default). For sweeps that OOM: `iter_temporal_chunks` yields chronological
calendar blocks with a warm-up tail. This is explicitly an **approximation**
— `chunking` travels onto `best_params` so Stage 3 can see the parameters it
certifies were selected on one. `chunking.truncated_trades` counts trades
still open when a chunk's frame ends, rather than dropping them silently.

**Handoff to Stage 3.** `stage2_targets` reads `stage2_summary.json` back as
exact `(symbol, timeframe)` pairs with locked parameters.

---

## 4. Stage 3 — `backtest/audit_gates.py` — Gate R certification

**Purpose.** The only script that produces a gate verdict anybody may act
on. Locks Stage 2's winning parameters, runs them **once** over the untouched
holdout, and answers: does the edge survive out-of-sample, inside the one
market environment (quadrant) it was designated for? This is the stage that
converts "looks good in-sample" into "certified."

**Inputs.** `stage2_summary.json` (locked parameters, per exact pair — not a
glob of `best_params_*.json`, which could contain a superseded winner).
`--holdout-end` defaults to the **present**, not a hardcoded year, so
certification never silently stops covering new bars.

**Mechanism — Gate R.** Since the "Regime-Switching Incubator Charter"
(2026-08-21), the certification verdict is **Gate R alone**, not the roll-up
of Gates 1-3: `regime_gate` scores only the `optimal_regime` Stage 1
designated, on the holdout — `MIN_REGIME_PROFIT_FACTOR` (1.00) over
`MIN_REGIME_TRADES` (30) in that one quadrant, imported from
`backtest.baseline` so screen and certification can never diverge on the
number. Three distinct outcomes: FAIL on trade count (including a quadrant
never traded out-of-sample — a starvation case, not "the strategy lost
money"), FAIL on the factor, or **NOT EVALUATED** when no quadrant was
designated (not a pass). `[REGIME STARVATION]` is printed and recorded
specifically for the trade-count failure, because "Gate R FAIL" alone reads
identically whether the edge died or the strategy simply never re-entered its
own environment.

**Gates 1, 2, 3 are still computed and reported in full, but cannot fail a
certification** — they score the blended sample across every market state,
and a live-supervised strategy that stands itself down outside its quadrant
never trades that blended sample. `aggregate_status` preserves the
pre-charter roll-up verbatim so the change is legible, but `status`/`passed`
come from Gate R alone. A configuration can be **certified with a failing
Gate 1** — this is the intended effect of the charter.

| Gate | Criterion | Threshold |
|---|---|---|
| **Gate R** (certification verdict) | PF, in the designated quadrant, on the holdout | >= 1.00 over >= 30 trades |
| 1 · In-Sample (evidence only) | Sharpe | informational |
| | Profit factor | >= 1.00 |
| | Trades | >= 100, scaled to >= 30/year, capped at 200 |
| | Max drawdown | <= 12.0% |
| 2 · Robustness (evidence only) | WFO efficiency | >= 0.50 |
| | Monte Carlo 95% max DD | <= 18.0% |
| 3 · OOS Holdout (evidence only) | Holdout/IS retention | >= 0.80 |

**No prop-firm rule reaches a verdict.** `_assert_no_prop_firm_rules` refuses
a config carrying `trailing_drawdown_pct` or `daily_loss_limit`.

**Outputs.**

| File | Contents |
|---|---|
| `gate_audit_<SYMBOL>_<TF>.json` | Per-pair, **authoritative** verdict. Also written unsuffixed for the last-run timeframe (the documented Stage 5 command uses this). |
| `stage3_audit_summary.json` | Campaign-level index (via `pipeline.write_stage`); transcribes, never recomputes, so it cannot disagree with the audits it indexes. Merges across timeframes rather than overwriting. |

**Promotion staging.** A PASS is SHA-256 sealed (code, `best_params`,
`gate_audit`) and staged into `strategies/approved_incubator/<strategy>/`
via `backtest.promote.promote` — reused, not reimplemented, so Stage 3 and
Stage 5 cannot disagree about what was promoted. **Nothing is git-committed
here** (`commit=False`, always); the commit and the four-choice menu remain
Stage 5's, in front of a human.

**Handoff to Stage 4/4.5/5.** Stage 4 and 4.5 run on the same strategy/symbol
regardless of Gate R's verdict — they characterize the strategy further; only
Stage 5's promotion checks Gate R's PASS (or requires `--force`).

---

## 5. Stage 4 — `backtest/verify_full.py` — Full lifecycle, tear sheets, cost drag

**Purpose.** One full-lifecycle run per contract for narrative/diagnostic
evidence — tear sheets, full trade log, cost drag. **Explicitly not a
certification** (`is_certification: false` in the JSON) — its window
contains the Stage 3 holdout, so its metrics are in-sample by construction.
No gate table is printed.

**Inputs.** A symbol/timeframe and a window spanning both in-sample and
holdout (e.g. `--start 2010-01-01 --end 2026-01-01`), Stage 2's locked
parameters.

**Mechanism.**
- **Cost drag** reported as a total, per trade, and as a share of gross
  profit — the third figure decides whether an edge is real, and is `None`
  (not `0%`) when there's no gross profit for costs to be a share of.
- **`friction_by_regime`** (2026-08-24) breaks trades/gross/costs/net P&L out
  by the **entry bar's** quadrant, written into `dual_metrics_<SYMBOL>.json`.
  Labels are read from the same `RegimeProfiler` frame this run classified —
  never re-derived, so the friction table and the regime profile can't assign
  the same trade to different quadrants. This exists because a regime-gated
  strategy trades one quadrant, and the certified quadrant can be handing 80%
  of gross to the broker while the blended figure reads a comfortable 35%.
- **`--slippage-atr-mult`** (2026-08-24, off by default): switches slippage
  from a constant tick count to `M x ATR(N)` per side. Opt-in because it
  changes every P&L figure; which model ran is recorded as `slippage_model`.

**Outputs.** `dual_metrics_<SYMBOL>.json`, HTML tear sheets, full trade-log
CSV, all in a timestamped directory under `/mnt/backtest/artifacts/`. Ends by
printing the `discord_reporter.py --stage 4 --artifacts <this run's dir>`
command.

**Gate/pass criteria.** None — diagnostic only.

**Handoff.** Stage 4.5 runs on "Stage 4's own window" (per the rules doc: its
`--start`/`--end` are passed by the orchestrator, not defaulted, so the two
stages describe the same bars).

---

## 6. Stage 4.5 — `backtest/dow_gate.py` — Day-of-week live-trading gate

**Purpose.** Added 2026-09-06 at operator instruction. Names each
configuration's worst weekday, runs a counterfactual with that weekday's
entries suppressed, and writes a verdict the **live loop** (not the
backtest search) can act on. This re-instates as a **live gate** what Stage 1
demoted to purely descriptive on 2026-08-20 — the reasoning that "which
weekday loses is largely a restatement of which regime that weekday falls
in" is not retracted; the difference is this verdict never touches the
search (no sweep masked, no gate scored on it, nothing dropped) — it only
stands a promoted pair down **live**.

**Inputs.** Every configuration Stage 2 left parameters for, over Stage 4's
window.

**Mechanism.**
- **It prunes nothing.** `promotion_gate()` returns `promote: True` for every
  verdict; everything reaching this stage advances to Stage 5 regardless of
  the weekday table. The output is an instruction for the live supervisor.
- The worst weekday is **always identified**; it is **blocked only when its
  expectancy is negative** (mean net P&L/trade). `--block-worst-always`
  overrides this and is recorded as `block_rule: "worst_always"` — it must
  never be the silent default, since "every weekday made money" must stay
  distinguishable from "the stage didn't run."
- Ranking: expectancy ascending, ties on win rate ascending, then largest
  share of the deepest drawdown episode. A weekday below `--min-trades`
  (default 20) is **never ranked**, however badly it scored, and is listed in
  `below_floor`.
- Attribution is by the trade's **entry bar**, keyed on the CME session date
  (imported from `report.day_of_week_breakdown`, not re-derived).
- **One verdict per version.** Version B's trade list is a subset of A's, so
  its weekday table is a different table; the per-pair file carries a
  `versions` map, each with its own profile/worst-day/block/counterfactual.
  `--ml` runs Version B's own profile; without it, a B package's lookup is
  `NOT EVALUATED`, not an empty (implicitly-all-clear) block list.
- The counterfactual reports the **candidate count**, not just the trade
  count — a filter removes candidate triggers, and the single-position walk
  can make a trade count go *up* when a filter binds (see Gotchas below).

**Outputs.**

| File | Contents |
|---|---|
| `dow_gate_<SYMBOL>_<TF>.json` | Per-pair, **authoritative**. No unsuffixed form exists — a blocked weekday is a fact about one (symbol, timeframe) pair. |
| `stage45_dow_summary.json` | Index over the per-pair files; merges across timeframes. |

The handoff's internal stage id is the **integer** `pipeline.STAGE45 = 45`
(not 4.5 — `write_stage` stamps `int(stage)`, and a float would truncate to
4 and make a Stage 4.5 file pass a Stage 4 `read_stage` check).
`STAGE_LABELS[45] = "4.5"` is what spells it in banners.

**Gate/pass criteria.** None that prune. `selected_in_sample: true` and
`is_certification: false` on every file.

**Orchestrator behavior.** Runs under `check=False`, unlike Stages 1-4 (which
form a chain where a failed stage hasn't written what the next reads). A
day-of-week profiler that raised must not discard certifications Gate R
already granted; affected packages record
`day_of_week_gate.status: "NOT EVALUATED"`.

**Handoff to Stage 5.** `pipeline.stage45_blocked_days` reads the mapping
back keyed per `(symbol, timeframe, VERSION)`.

---

## 7. Stage 5 — `backtest/promote.py` — Promotion

**Purpose.** Human decision point. Promotes exactly one version of one
strategy (on one symbol's evidence) into
`strategies/approved_incubator/<strat>/` and commits it to git. Nothing is
promoted without a human choosing from the four-choice menu.

**The Dual-Version Workflow** (what happens when a backtest finishes,
steps 1-4 automatic, step 5 human):

1. Console scorecard + gate audit — Version A
2. Standalone HTML report with trade inspector — Version A
3. Console scorecard + gate audit — Version B
4. Standalone HTML report with trade inspector — Version B
5. **The four-choice menu** — `[1] Promote A` `[2] Promote B`
   `[3] Parameter Sweep/Sensitivity` `[4] Keep in Experimental`

**Inputs.**
- `--audit-file <gate_audit_SYMBOL.json>` — the **authoritative** Stage 3
  gate verdict. Promotion is refused unless it says PASS (`--force`
  overrides and records it).
- `--dow-gate <dow_gate_SYMBOL_TF.json>` — Stage 4.5's verdict. The
  orchestrator passes it explicitly; a hand run must too, or the promoted
  package's weekday instruction goes unrecorded.
- `--metrics <dual_metrics_SYMBOL.json>` — the locked metrics snapshot
  (optional; absence yields `metrics_status: "NOT RECORDED"`, never an
  invented snapshot).
- `--source <module.py>` — the strategy code, promoted byte-for-byte with its
  own SHA-256 recorded (so a "cleaned up" file is provably a different
  strategy).

**Mechanism / what `meta.json` becomes.**

| Field | What it holds |
|---|---|
| `certification` | Audit path, symbol, windows, SHA-256, or the literal string `"NOT CERTIFIED"` (written, never omitted). |
| `params` / `params_source` | The **run's** parameters (module defaults, layered with the metrics snapshot's `meta.params`, layered with any explicit `--params`) — not the module's bare defaults, since a Stage 2 winner may differ from them entirely. |
| `risk` | Stop/target/trailing under readable names for CrossTrade governance. Three states: `"NOT DECLARED"` (no such parameter), `null` (parameter exists, modelled off — e.g. no take-profit), or a value. |
| `day_of_week_gate` | Stage 4.5's verdict — the only file `realtime/live_dispatcher.py` reads about the calendar (`DOW_GATE_KEY = "day_of_week_gate"`, `realtime/live_dispatcher.py:280`). **The version is part of the lookup** — a `..._VB` package gets B's weekday, never A's. Four possible states, kept apart (verified against `realtime/live_dispatcher.py` and `backtest/promote.py`): no key at all / `NOT EVALUATED` (no file — Stage 4.5 never ran for this pair, or ran without covering this version); `blocked_weekdays: []` under `EVALUATED` (ran, every session cleared); a populated block list; `UNREADABLE` (warns, blocks nothing — a truncated artifact doesn't discard a Gate R pass). |
| `seal` (from Stage 3) | SHA-256 of code, `best_params`, `gate_audit` — three hashes because the same code under a different winning cell is a different strategy with the same code checksum. |

**Version A** is promoted byte-for-byte. **Version B is a pipeline, not a
file**: `baseline.py` (verbatim source) plus a generated `strat.py` wrapper
applying `apply_ml_signal_filter` at the run's threshold.

**Outputs.** `strategies/approved_incubator/<strat>/`: `strat.py`,
`meta.json`, `baseline.py` (Version B only) — and a **git commit of that
directory alone**.

**Gate/pass criteria.** Refused unless the gate audit is PASS; `--force`
overrides and sets `gates_overridden: true`.

**Note.** Being in `approved_incubator/` is a record that a version was
chosen — **not permission to trade it.** That permission is a separate,
later decision (systemd units, CrossTrade account routing).

**Handoff downstream.** `realtime/live_dispatcher.py` and `master_live.py`
read `meta.json` at runtime — `day_of_week_gate` for the weekday
stand-down, `certification`/`risk` for governance context. Regime
gating live is computed fresh each cycle (not read from `meta.json`) against
the live daemon's published quadrant.

---

## 8. Glossary

**Regime quadrant.** One of four states from `mdlib/regimes.py`:
1 High-Vol/Trending, 2 High-Vol/Ranging, 3 Low-Vol/Trending, 4
Low-Vol/Ranging. **0 = UNDEFINED** (`QUADRANT_UNDEFINED`, indicator
warm-up) — 0 is not a quadrant, and a naive `NaN > 25.0` encoding files
every warm-up bar under Low-Vol/Ranging if this isn't handled explicitly.
`config/portfolios.json` schema 1.0.0 numbered these incompatibly (every
digit named a different environment); schema 1.1.0 relabels them to agree,
and **`canonical_quadrant` is the only supported resolver** between a live
daemon's numbering and a backtest's.

**Gate R.** The Stage 3 certification verdict, introduced 2026-08-21: profit
factor >= 1.00 over >= 30 trades, measured **out-of-sample** (the holdout),
**inside the one quadrant Stage 1 designated**. Replaces the pre-2026-08-21
roll-up of Gates 1-3 as the thing that decides PASS/FAIL; those three gates
are still computed and reported as evidence but cannot fail a certification.

**theta_vol.** A per-(symbol, timeframe) volatility anchor **loaded from the
pinned 2013-01-01..2022-12-31 in-sample window**, never computed live —
written into each regime parquet and read back through `provenance()`. It is
a property of the anchor, not of the request window: a median taken over a
caller's own window would let a quiet morning read as high-volatility. It is
per (symbol, TIMEFRAME) specifically — NQ is 7.90 at 15m and 11.33 at 30m, a
43% different boundary on the same tape (per CLAUDE.md; consistent with the
stage docs' treatment of ATR/regime profiling as computed once and
transported, never re-derived downstream).

**variants_tested.** The number of parameter combinations a Stage 2 sweep
actually evaluated, carried onto every report/leaderboard row. A Sharpe read
without this number is not a measurement — it doesn't say how many chances
the sweep had to find it by luck. Multi-timeframe runs also track
`variants_tested_all_timeframes` (cells x timeframes), since `variants_tested`
alone understates the search.

**The 3-year OOS holdout.** `HOLDOUT_START = 2023-01-01`. Bars from this
date forward are never touched during Stage 1 screening or Stage 2
optimization; Stage 3 is the only stage permitted to measure against them,
and only once parameters are locked. This — not the NT8 cross-feed tree — is
the real out-of-sample gate.

**meta.json.** The single artifact `strategies/approved_incubator/<id>/`
carries per promoted strategy version. Written once, at promotion time, and
**never revisited** — a package promoted before Stage 4.5 existed simply has
no `day_of_week_gate` key, and a re-run of Stage 4.5 after promotion does not
update it.

**day_of_week_gate key.** `meta["day_of_week_gate"]`, written by
`backtest/promote.py`, read by `realtime/live_dispatcher.py`
(`DOW_GATE_KEY`). A rename on either side alone silently turns the live gate
off — the block is simply never found, and every log line still reads
correctly.

**Version A / Version B.** A = the rules-based strategy as written. B = the
same entry signals passed through an ML confirmation filter
(`apply_ml_signal_filter`) that stands down trades the classifier expects to
lose. B's trade list is always a **subset** of A's. Both can independently
survive Stage 1, be swept/certified/promoted, and are tracked as **separate
packages** (`<strategy>_<SYM>_<TF>_VA` / `..._VB`) with separate `meta.json`
files throughout Stages 2-5.

---

## 9. Gotchas for a rebuild

Pulled forward from CLAUDE.md's "Findings that are not recoverable from the
code" section, scoped to what's pipeline-relevant. These are things a fresh
reimplementation would very plausibly get wrong because the code alone
doesn't state the reasoning:

1. **`day_of_week_gate` is a single string key shared across two files that
   are never imported from a common source of truth for the key name itself**
   — `backtest/promote.py` writes it, `realtime/live_dispatcher.py` reads it
   as `DOW_GATE_KEY`. There are **three states to keep distinct**, not two:
   no key at all (promoted before Stage 4.5 existed), `blocked_weekdays: []`
   (Stage 4.5 ran and every session cleared), and an actual blocked day. A
   rebuild that collapses "no key" and "empty list" into one meaning will
   silently either over-block or under-block live trading.

2. **The verdict in `day_of_week_gate` is per (symbol, timeframe, VERSION)**,
   not per strategy. Version B's trades are a subset of A's, so applying A's
   weekday table to a B package stands it down on a session measured on a
   strategy that was never actually deployed.

3. **The live gate keys on the FILL bar's session weekday, not the last
   closed bar's** — because the engine fills at the *next* bar's open. Keying
   on the closed bar's weekday would let through exactly the
   Thursday-evening signal that fills into Friday's session.

4. **Regime quadrant numbering is `mdlib/regimes.py`'s alone.** Any other
   subsystem (a config schema, a dashboard, a second implementation) that
   numbers quadrants independently risks a live daemon and a backtest
   disagreeing about what "Q1" means — invisible downstream, because the
   strategy is stood down in the environment it was certified for and turned
   loose in the one it never traded, with every log line reading correctly.
   Always resolve through `canonical_quadrant`.

5. **`theta_vol` must be loaded per (symbol, timeframe) from the pinned
   anchor window, never computed over the caller's live request window.** A
   median computed live is a property of the request, not the market.

6. **A filter can only remove candidate triggers, never realized trades
   directly** — the single-position walk means declining an early trigger
   can leave the strategy flat for (and then take) a later trigger it would
   otherwise have been holding through already. A trade *count* going up
   after adding a filter is not evidence the filter failed to bind; compare
   candidate counts or the trade list, never the realized trade count alone.

7. **Trades are attributed by their ENTRY bar/session everywhere** — regime
   quadrant, friction-by-regime, and the day-of-week gate all key off entry,
   never exit or a later bar. This must be consistent across Stage 1's
   profiling, Stage 4's `friction_by_regime`, and Stage 4.5's weekday
   breakdown, or the three reports will disagree about which trades belong
   to which bucket even though they're describing the same trade list.

8. **Costs (commission + slippage) must be present from the very first
   test.** Variant rankings measurably change once 1-tick-each-way slippage
   and commission are applied — an unfunded comparison of "candidate
   strategies" before costs is not evidence about which one is better.

9. **`variants_tested` must be carried through every stage's output**, not
   just computed once at Stage 2. A Sharpe number without it cannot be
   judged for how much of a search produced it.

10. **The 3-year holdout (`HOLDOUT_START`), not the NT8 cross-feed tree, is
    the actual out-of-sample gate.** The NT8 tree is a thin sanity check
    spliced onto NinjaTrader's own contract-roll rules and must not be
    mistaken for validation rigor.

11. **Stage 1's quadrant-restriction (`TARGET_QUADRANTS`) changes which
    environment a strategy is judged in, never whether it passes.** All of
    Stage 1's normal bars (including its own `max(50, 10%)` sample floor,
    which is intentionally *not* the same as Gate R's floor of 30) still bind
    on the declared quadrant exactly as on any other. A rebuild must not
    read a declared quadrant as a relaxation of the pass bar.
