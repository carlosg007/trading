---
name: portfolio-routing
description: "Portfolio-to-account routing, ATR position sizing, signal netting, the forward-incubation promotion rule, the fill recorder that feeds it and the incubator tracker."
paths:
  - "portfolio/**"
  - "config/portfolios.json"
  - "scripts/incubator_tracker.py"
  - "scripts/record_incubator_fills.py"
  - "data/incubator_ledger.json"
  - "tests/test_portfolio_*.py"
  - "tests/test_incubator_tracker.py"
  - "tests/test_incubator_recorder.py"
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
- **`incubator_recorder.py`** — the INPUT side of the promotion gate, added
  2026-08-25. `promotion_daemon` grades a strategy on the closed trades in its
  ledger entry and nothing put any there: the ledger shipped empty, so every
  incubating strategy was invisible to the audit that graduates it. This turns
  NT8's export into those trades. It computes the criteria's INPUTS and never
  the criteria, and writes no declared summary at all — the daemon derives
  every figure from the trade list, and a summary written here would be a
  second implementation of the gate, free to disagree with the one that
  promotes.
  - **Fills pair FIFO into round turns, per (strategy, symbol), matched in
    pieces.** A 2-lot entry closed by two 1-lot exits is TWO trades priced
    separately; averaged into one, a winner hides a loser and the trade count
    — criterion two — is halved. **What is still open at the end of the log is
    not a trade**: its outcome is unknown, and counting it would let a strategy
    reach the fourteen-trade bar on positions it has not exited. Open lots are
    COUNTED in the report instead. A TRADE-level export (NT8's Trades tab)
    already carries the round turn and is taken as it is.
  - **Costs are in every P&L figure**, because a forward profit factor is
    compared against 1.00 and the commission on fourteen round turns is the
    whole decision at that boundary. `--cost-basis auto` takes a declared-NET
    figure as net and charges a gross one the log's own commission (pro-rated
    by size) or the spec's round turn; `log` leaves an already-net export
    alone; `specs` always applies the spec's. Every trade records which priced
    it — the right answer depends on how the NT8 template was configured, and
    that is an operator's knowledge, not a default.
  - **The multiplier is READ from `backtest/specs.py`, never assumed.** A
    symbol with no ContractSpec has its rows reported and skipped: a guessed
    multiplier scales every P&L figure for that contract and nothing
    downstream would look wrong.
  - **Attribution is evidence, not inference.** The strategy the log NAMES
    wins (`strategy_tag` is what `format_crosstrade_json` already sends).
    Without one, a row is attributed only when the account holds exactly ONE
    strategy trading that contract; two candidates is UNATTRIBUTED and
    reported. A trade filed under the wrong strategy is a promotion decided on
    somebody else's P&L. `NT8_ACCOUNT_ALIASES` (`Sim101` -> `Incubator-Odd`) is
    the only place NT8's account names and the routing table's are tied
    together, and portfolio ids resolve to themselves.
  - **Re-reading the same export changes nothing.** Every trade carries a
    `trade_id` derived from what it IS, so an evening cron over a growing file
    adds what is new — duplicates would make a strategy look like it cleared
    the fourteen-trade bar twice as fast as it did. A GRADUATED entry is never
    appended to: its trades are being taken on a prop account and are not
    incubation evidence.
  - **The NT8 reader is `live/dispatcher.py`'s**, exported as `read_fill_log` /
    `normalize_fill_row` / `fill_status` / `FILL_ALIASES` rather than copied.
    One alias table and one definition of "this row was filled", so a spelling
    that module stops recognising cannot become a fill the recorder silently
    stops recording.
- **`scripts/incubator_tracker.py`** is the CLI over it: resolve each ledger
  entry to its incubator account, print the ASCII status table, and — ONLY
  under `--auto-promote` — graduate what cleared. It computes nothing itself.
  The account comes from `active_strategies` in the config and there is NO
  fallback to the `portfolio` a ledger entry names: taking it would produce a
  PROMOTE verdict for a strategy no incubator portfolio holds, which
  `promote_strategy` then refuses, so the row would clear every criterion on
  the table and fail on the way out. Such a row is UNROUTED and the note says
  which portfolio the ledger claims. The table shows PROGRESS against each bar
  (`8/14`, not `8`), with the threshold transcribed from the daemon's own
  report; it is grouped by ACCOUNT, because the two incubator books are
  separate risk envelopes and are read separately; and the Discord card
  carries one compact line per incubating strategy — `Day 8/14 | 12/14 trades
  | PF 1.45 | DD $350 / $1,000` — because an embed clips a wide code block on
  a phone and the table is the half that gets clipped.
- **No risk management anywhere in this package.** The drawdown figures in the
  config are a specification handed to CrossTrade NAM, like
  `compliance_rules/*.json`; nothing here reads an account balance.

## Commands

```bash
# THE FILL RECORDER, first half of the post-market job. Turns the NT8 exports
# in /mnt/backtest/artifacts/incubator_logs/ into closed trades on the ledger
# the tracker grades. WRITES ONLY under --write. Exit 1 = no log directory
# (no feed, which is not "no trades"); exit 2 = ran but could not use some
# rows; 0 = clean.
python3 scripts/record_incubator_fills.py                  # show, write nothing
python3 scripts/record_incubator_fills.py --write          # record them
python3 scripts/record_incubator_fills.py --logs /tmp/nt8_export.csv
python3 scripts/record_incubator_fills.py --cost-basis log --write

# THE POST-MARKET CRON PAIR, in this order. Two commands rather than one:
# they fail differently, and an operator has to be able to grade the ledger
# without re-reading the exports.
#   30 17 * * 1-5  cd ~/src/trading && .venv/bin/python3 scripts/record_incubator_fills.py --write \
#                    && .venv/bin/python3 scripts/incubator_tracker.py --auto-promote

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
