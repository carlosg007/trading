---
name: agents-synthesis
description: "The three agent tiers, what is implemented vs scaffold, the Monte Carlo sanitation, the Version B refit cadence, and the CrossTrade compliance rulebook."
paths:
  - "agents/**"
  - "compliance_rules/*.json"
  - "dashboard/**"
  - "tests/test_tier*.py"
---

# The agent tiers and model-generated code

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
- **THE MONTE CARLO SANITATION, from 2026-08-24.** A NaN reaching the drawdown
  distribution is the one corruption in `run_monte_carlo_simulation` that reads
  as SAFETY rather than as an error, and three routes to one are now closed —
  each RECORDED on the result rather than fixed quietly.
  - **Non-finite inputs are dropped** (`n_dropped_nonfinite`). `.dropna()`
    removed NaN and left ±inf standing; an inf return makes the whole
    cumulative path inf, and inf/inf is NaN in the drawdown division.
  - **Returns at or below -1.00 are clipped to -1.00**
    (`n_clipped_to_total_loss`). Below -1.00 the equity factor `1 + r` goes
    negative and `cumprod` FLIPS THE SIGN of the rest of the path — not a
    deeper drawdown but arithmetic that has stopped describing an account. It
    reported a **-142% drawdown on an account that cannot lose more than it
    holds**. Clipped, the account is ruined: equity 0, and 0 thereafter, which
    is a 100% drawdown and the true reading.
  - **A zero running peak no longer divides.** A path whose FIRST resampled
    trade is a total loss had a peak of 0 and computed 0/0; those cells are
    guarded with `where=` and set to a -100% drawdown. Every later cell was
    always safe — the peak is monotone non-decreasing, so once positive it
    stays positive.
  - **What all three produced was `max_drawdown_pct_at_confidence: NaN` beside
    `prob_max_loss_breach: 0.0`**, because the breach test is
    `mean(max_dds <= -limit)` and `NaN <= -8.0` is False. A bootstrap over an
    array containing an infinite loss reported a ZERO percent chance of
    breaching the loss limit — the strongest possible safety reading, from the
    most corrupt possible input — Gate 2 scored the NaN tail NOT EVALUATED, and
    nothing on the console named the array as the reason. A non-finite path
    surviving all three guards now returns `ok: False` rather than a tail
    computed over it, which puts Gate 2 into NOT EVALUATED, and NOT EVALUATED
    is not a pass.
  - **The clean path is bit-identical**, verified against the pre-fix function
    on the same seed: sanitation only ever removes or clips values that were
    already corrupting the result.
- **THE VERSION B REFIT CADENCE, from 2026-08-24.** `apply_ml_signal_filter`
  refits when the closed-trade pool has grown by `ML_REFIT_GROWTH` (0.10) of
  what it was last fitted on, not on every completed trade.
  - **A fit costs 17-40 ms whatever the sample size** — the 100 boosting
    iterations dominate, not the rows — so the old rule cost one fixed fit per
    CANDIDATE ENTRY. `t3_braid_scalp_20260823` on NQ 15m over the charter
    window signals 6,506 longs and 6,211 shorts: ~12,700 fits, **269.6 s for
    the long side alone**, ~9 minutes for one (symbol, timeframe) and ~50
    across a 4-symbol 2-timeframe screen — with nothing printed between the
    RUNNING line and the row. It read as a hang and was a silent loop.
  - **Staleness is safe; lookahead is not.** Between refits the model is one
    fitted on FEWER, STRICTLY OLDER closed trades, so a decision can only ever
    know less than the exact rule — never anything from at or after the signal
    bar, which is the one property this function exists to guarantee. At 0.10
    every decision uses a model trained on at least ~91% of the trades that had
    closed before it.
  - **It is an APPROXIMATION and it moves Version B's numbers.** On that NQ run
    14.83% of long candidates are decided differently from the exact rule. That
    number is mostly a fact about the FILTER, not about the cadence: halving the
    cadence to 0.05 doubles the fits and only moves it to 13.82%, because these
    probabilities sit near the 0.50 threshold and flip on any change of training
    window. Read it as evidence about how stable Version B's vetoes are.
  - **Measured end to end on that configuration**: 418.7 s at `0.0` against
    9.2 s at the 0.10 default, a 45x wall-clock difference — and BOTH runs
    report Version A 0.96 PF, Version B 0.95 PF and the same SURVIVES verdict.
    The 14.83% of individually flipped vetoes did not move the screening
    decision here. That is one configuration, not a guarantee: it is evidence
    that the aggregate is far more stable than the per-candidate probabilities,
    which is what would be expected if those probabilities sit near 0.50.
  - **`--ml-refit-growth 0.0` restores the old rule bit for bit** —
    `max(1, floor(n * 0.0))` is 1, which is the original `n_available !=
    fitted_n`. `tests/test_tier3_workers.py` pins that equality against an
    inline oracle of the pre-cadence loop. Which cadence ran is recorded per
    side on `metrics["meta"]["ml_refit"]`, because two Version Bs fitted on
    different cadences are not comparable and that has to travel with the
    numbers.
  - **Predictions are BATCHED and that changes nothing.** The refit schedule
    reads only the trade count, never a prediction, so it is resolved before any
    fit and each model then scores its whole segment in one call. A single-row
    `predict_proba` is ~0.75 ms of call overhead around ~2 us of work: 6,500 of
    them cost 4.8 s against 9 ms batched, a 507x overhead saving with identical
    probabilities.
  - **`ML_MAX_TRAIN_ROWS` (50,000) caps one fit's training trades**, MOST
    RECENT kept — dropping the oldest preserves causality. It does not bind on
    this repository's workloads (the largest slice measured is ~6,500) and so
    changes no existing number.
  - **`HistGradientBoostingClassifier` takes no `n_jobs`.** It parallelises
    through OpenMP, so the bound is a thread count read when the native library
    loads — `BT_ML_THREADS`, default 1, and the runners already pin 1 at their
    process boundary. **More threads is SLOWER here**, measured: 50 fits of
    1,000x5 take 1.52 s at 1 thread and 2.00 s at 4, because the fits are small
    and numerous and each pays pool overhead exceeding the work it distributes.
- `system_monitor.py`: circuit breaker tracking RAM and runaway execution loops.

**`compliance_rules/`** — prop-firm constraint sets as JSON, one per program
(`fundednext_rapid.json`). Each rule carries its unit, its basis, and an
`enforcement` block. Post-overhaul this is the **specification handed to
CrossTrade NAM**, not a research gate.
