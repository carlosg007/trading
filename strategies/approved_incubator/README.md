# approved_incubator/

Strategies that have cleared initial review and are staged for deeper
evaluation. The dashboard's Strategy Vault reads this directory.

**Being here is not an approval to deploy.** It means a strategy is under
active evaluation. Deployment requires clearing the gates in
`docs/STRATEGY_DEVELOPMENT.md` and the Phase 3 holdout.

## Expected layout

One directory per strategy:

```
approved_incubator/
    <strategy_name>/
        meta.json          required - what this is and how it was tested
        strat.py           written by backtest/promote.py - the promoted version
        baseline.py        version B only - the rule-based module strat.py filters
        dual_metrics.json  written by promote.py when --metrics was supplied
        strategy.py        optional - signal_fn(bars) -> (entries, exits)
        returns.parquet    optional - saved BacktestResult.returns
        trades.parquet     optional - saved BacktestResult.trades
        equity.parquet     optional - saved BacktestResult.equity
```

`strat.py` is what `backtest/promote.py` writes; `strategy.py` is the older
hand-placed name. Nothing reads either programmatically — the dashboard and
`agents.tier1_master` describe a strategy from `meta.json` alone — so both are
listed rather than one being silently wrong.

A strategy directory with no `meta.json` is listed by the dashboard as
incomplete rather than skipped silently — an unlabelled strategy is worse than
a missing one.

## `meta.json`

```json
{
  "name": "ES Opening Range Breakout",
  "version": "A",
  "family": "breakout",
  "description": "One line on what the strategy does.",
  "symbols": ["ES", "NQ"],
  "timeframe": "30m",
  "start": "2013-01-01",
  "end": "2023-12-31",
  "variants_tested": 24,
  "costs_included": true,
  "ruleset_id": "fundednext_rapid",
  "oos_status": "not_run",
  "notes": "Anything a reader needs to judge the numbers."
}
```

`variants_tested` and `costs_included` are not optional bookkeeping. A Sharpe
ratio read without knowing how many variants it was selected from, or whether
commissions and slippage were applied, is not interpretable — see
`backtest/report.py`, which requires both.

`version` follows the Dual-Version Mandate: `A` for the rule-based baseline,
`B` for the ML-augmented filter. ML is adopted only if B beats A
out-of-sample without breaching the prop-firm limits.

`backtest/promote.py` writes the same file with extra keys, all of them there
so a reader can tell what a number is worth:

| Key | Why it is recorded |
|---|---|
| `source`, `source_sha256`, `promoted_sha256` | The promoted file provably is the file that was backtested. |
| `metrics`, `metrics_status` | The locked snapshot, or `"NOT RECORDED"` when none was supplied. Never an invented number. |
| `gate_audit`, `gate_audit_status` | Gate 1/2/3 as PASS / FAIL / NOT EVALUATED. `NOT EVALUATED` is not a pass. |
| `gates_overridden` | True when `--force` promoted past a gate that did not clear. |
| `audit_warnings` | Structural objections to the source module, reported not enforced. |
| `ml_threshold`, `baseline` | Version B only: the filter threshold and the module it filters. |

## Why the parquet files are optional

The dashboard degrades: `meta.json` alone gives you a listed strategy with no
performance panel. It never fabricates a curve for a strategy that has not been
run — an empty panel is the correct display for a strategy with no results.
