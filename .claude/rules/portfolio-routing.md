---
name: portfolio-routing
description: "Portfolio-to-account routing, ATR position sizing, signal netting, the forward-incubation promotion rule and the incubator tracker."
paths:
  - "portfolio/**"
  - "config/portfolios.json"
  - "scripts/incubator_tracker.py"
  - "data/incubator_ledger.json"
  - "tests/test_portfolio_*.py"
  - "tests/test_incubator_tracker.py"
---

# Portfolio routing, sizing, and forward incubation

**`portfolio/`** — portfolio-to-account routing and position sizing, added
2026-08-23. Sits ABOVE `backtest/` in the dependency chain and reads from it;
**nothing in `backtest/` may import from here**, because a research run that
depended on which live account a strategy would be routed to would be tuned to
a funding program rather than to a market.

- **`config_loader.py`** reads `config/portfolios.json` — the four-account
  partition (Incubator-Odd/Even, Prop-Odd/Even), each with a basket, a regime
  scope and a risk envelope — validates it, derives
  `allowable_forward_dd_usd` (`max_trailing_drawdown_usd x
  max_forward_incubation_dd_pct`, a fraction of the TRAILING LIMIT and not of
  the account), and answers `get_portfolio_by_account`,
  `get_portfolio_for_strategy` and `get_asset_spec`. Two reconciliations run on
  every load and REFUSE the config on a disagreement: `asset_metadata` against
  `backtest/specs.py` (a wrong multiplier silently scales every P&L figure),
  and the regime labels against `backtest/profiler.py`. Routing has no
  fallback — an unassigned strategy raises rather than being placed by a rule
  nobody wrote down. **Schema 1.0.0 numbered its quadrants incompatibly with
  `mdlib/regimes.py`** — every digit named a different environment — and 1.1.0
  relabels them to agree; `canonical_quadrant` is the only supported way to
  resolve a label.
- **`volatility_sizer.py`** — `contracts = floor(risk_budget / (ATR x
  stop_atr_mult x point_value))`, clamped. FLOOR rather than round, so the
  budget is a ceiling on intended risk. The `min_contracts` clamp CAN breach
  the budget (one MCL at ATR 3.00 risks $300 against $250) and `size_detail`
  reports `budget_breached` rather than rounding it away. `stop_atr_mult`
  defaults to 1.0 and must be passed the strategy's own `sl_atr_mult`, or the
  position is sized for a stop the strategy will not use.
- **`portfolio_manager.py`** — `aggregate_signals` nets many strategies into
  one position per (portfolio, symbol); `build_order_plan` sizes each and
  records every DECLINED one with its reason; `build_order_payloads` reduces
  that to the wire format. Payloads are built by
  `live.dispatcher.format_crosstrade_payload`, the only order formatter in this
  repository. Two decisions the spec left open are written down in the module:
  conviction SCALES the size (so stacking is observable in the order) with the
  clamp bounding the breach, and a symbol outside its portfolio's quadrant is
  STOOD DOWN rather than sized smaller. It never emits `FLATTEN` — it does not
  know what is open.
- **`promotion_daemon.py`** — the forward-incubation promotion rule and the
  file surgery that acts on it, added 2026-08-23. Stage 3 certified a strategy
  on historical bars; this asks whether it kept working on bars nobody had
  then, from the FORWARD PAPER TRADES in `data/incubator_ledger.json` and never
  from a backtest. `evaluate_strategy_promotion` scores four LOOSE criteria —
  >= 14 calendar days (and >= 10 active sessions where the ledger records
  them), >= 14 closed trades, realized profit factor strictly > 1.00, and a
  realized forward drawdown inside `derived.allowable_forward_dd_usd`
  ($2,500 x 0.40 = $1,000 as shipped). `promote_strategy` moves the strategy
  `Incubator-Odd` -> `Prop-Odd` (or `Even`) in `config/portfolios.json` and
  stamps `GRADUATED_PROP` / `graduated_at` / `target_portfolio` onto the
  ledger.
  - **A missing metric FAILS its criterion; it is never a zero.** An absent
    trade count is a ledger nobody filled in, not a strategy that placed no
    trades, and the two must not be indistinguishable at the moment an account
    is handed over. `active_sessions` is the one documented exception and
    reports NOT RECORDED without blocking, so a ledger written before the field
    existed still promotes.
  - **An entry carrying its own `trades` is scored on them**, and when the
    summary beside them disagrees the evaluation FAILS rather than picking one:
    the readings are "the summary is stale" and "the trade list is
    incomplete", and both are reasons not to move an account. `metrics.source`
    records which path ran. A profit factor with no losing trade is `None` and
    falls back to the sign of net P&L — never the profiler's 999 sentinel,
    which means "undefined" and sorts like the best result on the board.
  - **The route is a TABLE, not a name.** `PROMOTION_ROUTES` is spelled out, so
    `Incubator-Test` raises instead of promoting onto a `Prop-Test` account
    that does not exist. Session dates come from
    `backtest.event_calendar.session_date` (imported, not reimplemented) and
    the allowable drawdown is READ from the loader's `derived` block rather
    than multiplied out a second time.
  - **Two files, one decision, and no atomic write across both.** Each is
    written temp-then-`os.replace`, and the config is validated through
    `load_portfolio_config` while it is still the temp file, so a mutation that
    would make the routing table unloadable never reaches the real path. The
    config is written FIRST because that ordering has a recovery: a failed
    ledger write rolls the config back. The reverse would leave a ledger
    reading GRADUATED_PROP over a config still routing to the sim account —
    invisible, and skipped by every later run as already done.
- **`scripts/incubator_tracker.py`** is the CLI over it: resolve each ledger
  entry to its incubator account, print the ASCII status table, and — ONLY
  under `--auto-promote` — graduate what cleared. It computes nothing itself.
  The account comes from `active_strategies` in the config and there is NO
  fallback to the `portfolio` a ledger entry names: taking it would produce a
  PROMOTE verdict for a strategy no incubator portfolio holds, which
  `promote_strategy` then refuses, so the row would clear every criterion on
  the table and fail on the way out. Such a row is UNROUTED and the note says
  which portfolio the ledger claims.
- **No risk management anywhere in this package.** The drawdown figures in the
  config are a specification handed to CrossTrade NAM, like
  `compliance_rules/*.json`; nothing here reads an account balance.

## Commands

```bash
# THE INCUBATOR TRACKER. Audits the forward paper trades in
# data/incubator_ledger.json against the four loose promotion criteria and,
# with --auto-promote, moves a strategy Incubator-Odd -> Prop-Odd (or Even) in
# config/portfolios.json. Reads no bars and runs no backtest. It WRITES ONLY
# under --auto-promote; --dry-run wins if both are given. Posts a summary embed
# when $DISCORD_WEBHOOK_URL or $BT_DISCORD_WEBHOOK is set — --no-discord never
# posts.
python3 scripts/incubator_tracker.py                       # evaluate and print
python3 scripts/incubator_tracker.py --auto-promote        # act on it
python3 scripts/incubator_tracker.py --dry-run --no-discord
python3 scripts/incubator_tracker.py --ledger data/incubator_ledger.json \
    --config config/portfolios.json --json /tmp/incubator_audit.json
```
