# Phase 3 — System Verification Report

**Date:** 2026-08-14
**Repo state:** branch `vectorbt-engine`, at `158ec95` *(Add the CrossTrade
dispatcher and incubator sync bridge)*, working tree otherwise clean.
**Environment:** Python 3.13.14 in `.venv`; `pandas 3.0.5`, `pyarrow 25.0.1`,
`numpy 2.5.2`, `duckdb 1.5.5`, `scikit-learn 1.9.0`, `streamlit 1.59.1`.

This records a full verification sweep of the built system. It is a statement
about the **code**, not about any strategy: nothing here is evidence that a
strategy works, and section 5 lists exactly what the sweep does not cover.

---

## 1. Test suites

Every suite is a plain script that exits non-zero on failure — there is no
pytest config in this repo, and `pytest` is not installed in `.venv`. Each was
run individually; the exit code, not the printed text, is the pass criterion.

| Suite | Checks | Exit | Covers |
|---|---:|---:|---|
| `tests/test_tier1.py` | 65 | 0 | intent routing, symbol/timeframe parsing, ruleset resolution, vault reporting, campaign event stream, synthesis error taxonomy |
| `tests/test_tier2.py` | 47 | 0 | consistency rule, profit target, trailing drawdown, robustness, lifecycle bands |
| `tests/test_tier3_workers.py` | 62 | 0 | worker tools, metrics, generated-code validation, RAM ceiling |
| `tests/test_dispatcher.py` | 63 | 0 | CrossTrade payload validation, HTTP 200/500/timeout handling, NT8 fill-log parsing |
| `tests/test_clean_signals.py` | 43 | 0 | per-symbol signals vs the interleaved-frame trap |
| `tests/test_streaming_lake.py` | 49 | 0 | `iter_bars` and the streaming engine built on it |
| `tests/test_engine_batching.py` | 97 | 0 | chunked == unchunked, trade for trade |
| `tests/test_engine_vbt.py` | 42 | 0 | vectorbt P&L reconciled against the legacy-loop oracle |
| **Total** | **468** | **0 failures** | |

### Naming discrepancy resolved

The verification request named `test_tier1.py`, `test_tier2.py`, and
`test_tier3.py`. Only `test_tier2.py` existed under that name; Tier 3's suite is
`test_tier3_workers.py`, and **Tier 1 had no test file at all** — its
implemented surface (`run_campaign`, `classify_intent`,
`synthesize_strategy_code`, and the vault/ruleset helpers the dashboard chat
calls) was untested while being reachable from the UI.

`tests/test_tier1.py` was written to close that gap. It touches neither the
network nor the lake: the GenAI client is stubbed and vault fixtures are
written to a temp dir. Its load-bearing checks are that an unrecognised prompt
routes to `conversational` rather than spending compute on a speculative
backtest; that symbols match on word boundaries (substring matching finds "ES"
inside "strategies"); that an unknown ruleset raises instead of silently
defaulting to FundedNext; that a vault directory without `meta.json` is
reported rather than skipped; and that `MissingAPIKey` (an expected
configuration state) stays distinguishable from `SynthesisError` (a failure),
since collapsing the two would let a template fallback pass as model-authored
code.

---

## 2. Dashboard

`dashboard/app.py` verified three ways:

1. `py_compile` — compiles, no syntax errors.
2. Bare-mode import with the module registered in `sys.modules` — executes to
   completion. The only output is Streamlit's expected
   `missing ScriptRunContext!` warnings.
3. `streamlit run dashboard/app.py --server.headless true` on port 8599 —
   served **HTTP 200** on `/` and `/healthz`, no traceback in the server log.
   The server was stopped afterwards.

The agent backend behind the UI is still mocked and labelled as such in the UI.

---

## 3. Data reader integrity

`mdlib/lake.py` is **unchanged and uncommitted-clean**:

- Last modified in `051b4b0` (2026-08-13, *"Remove the frame-in entry point"*) —
  **21 commits ago**. No commit in the Tier 2 / Tier 3 / dashboard / dispatcher
  work touched it.
- `sha256(mdlib/lake.py) = 458774a1f16fa3fdca191a216ecc9a6eb68ec2e621bdd564e5296c8a2c8e2b96`
- `git status` reports no modification to `mdlib/`.

The reader is also exercised, not merely untouched: `test_streaming_lake.py`
(49 checks) drives `iter_bars` and the streaming engine against real lake data,
and `test_clean_signals.py` (43 checks) verifies signals are computed per
symbol rather than across an interleaved frame.

**Lake bytes:** `python scripts/generate_manifest.py --verify --quick` over
**11,081 files** → `missing: 0  new: 0  modified: 0`, *"clean — every file
matches the manifest"*. Size/mtime only; a full SHA-256 pass was not run in
this sweep.

---

## 4. Compliance rulesets

One ruleset is active: `compliance_rules/fundednext_rapid.json` (schema 1,
FundedNext Futures — Rapid Challenge, $100,000 initial balance). Percentages are
authoritative; USD figures resolve from the balance.

| Rule | Value | Basis | Enforcement | Where |
|---|---:|---|---|---|
| `max_trailing_drawdown` | 8.0% | high-water mark | `ENFORCED` | `BacktestConfig.trailing_drawdown_pct`, engine |
| `profit_target` | 8.0% | initial balance | `ENFORCED_AT_TIER2` | `agents.tier2_supervisors.evaluate_compliance` |
| `consistency` | 40.0% | total profit | `ENFORCED_AT_TIER2` | `agents.tier2_supervisors.evaluate_compliance` |
| `max_daily_loss` | 5.0% | initial balance | **`NOT_ENFORCED`** | `BacktestConfig.daily_loss_limit` exists but is dead code |

**A clean `BacktestResult.breach` is still not evidence of compliance.** Only
the trailing drawdown is checked inside the simulation; profit target and
consistency are evaluated after the fact by Tier 2, and the daily loss limit is
not evaluated anywhere. A run that blows the 5% daily loss on one session and
recovers will report no breach.

---

## 5. What this sweep does *not* establish

Recorded so no one reads a green board as more than it is.

- **No strategy has passed Phase 3.** The held-back final 3 years of the
  Databento dataset remain untouched. Nothing in this report is evidence that
  any strategy has an edge.
- **`max_daily_loss` is unenforced** (section 4). Implementing it is the
  highest-value open compliance item.
- **The dispatcher has never sent a real request.** All 63 checks run against a
  fake opener. `live/config.json` still holds the placeholder webhook, which
  `send_execution_signal` refuses by design — the first live dispatch against
  the sim account still has to be done by hand. *(2026-08-25: that account was
  `Sim101` when this sweep ran; NinjaTrader has since renamed the four to
  `SimIncubator1`/`SimIncubator2`/`SimProp1`/`SimProp2`, which is what
  `target_account` now carries. The finding stands — nothing has been sent.)*
- **`evaluate_incubator_sync` has not run on a real NT8 export.** It is tested
  on synthetic CSV/JSON fixtures only; `/mnt/backtest/artifacts/incubator_logs/`
  has no data yet.
- **17 symbols still have UNVERIFIED contract specs** (PL, grains, LE, FX,
  crypto, micros) — no definition data downloaded. A wrong multiplier silently
  scales every P&L figure for that symbol.
- **The NT8 tree is unreachable through the reader.** `get_bars` still has no
  `source` parameter, so `lake/futures_nt8/` cannot be read via `mdlib.lake`.
  (It is a cross-feed sanity check, never the OOS gate.)
- **LightGBM is unpinned.** The Dual-Version Mandate references it; it is not in
  `requirements.txt`, and no Version B has been built.
- **Manifest verification was `--quick`** — size and mtime, not content hashes.
