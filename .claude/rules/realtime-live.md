---
name: realtime-live
description: "The live regime daemon, the reader, the CrossTrade formatters, the live execution loop and the only module that puts an order on the wire."
paths:
  - "realtime/**"
  - "master_live.py"
  - "live/**"
  - "tests/test_regime_daemon.py"
  - "tests/test_live_dispatcher.py"
  - "tests/test_dispatcher.py"
---

# The live regime service and the execution loop

**`realtime/`** — the live regime service, added 2026-08-23. Sits ABOVE
`backtest/` and beside `portfolio/`; **nothing in `backtest/` may import from
here**, for the same reason nothing there may import from `portfolio/` — a
research number that depended on live account or feed state would be tuned to a
broker rather than to a market. Three modules, split by what they do: the
daemon WRITES, the reader READS, the formatter FORMATS and sends nothing.

- **`regime_daemon.py`** — `MasterRegimeDaemon`. `calculate_regime(symbol,
  bars)` labels the last bar into `Q1_HIGH_VOL_TREND` / `Q2_HIGH_VOL_CHOP` /
  `Q3_LOW_VOL_TREND` / `Q4_LOW_VOL_MEAN_REVERSION`, `evaluate_ml_gate` runs the
  Version B confirmation model, and `update_state` publishes to
  `data/live_regime_state.json`.
  - **Not one threshold is restated here.** `ADX_TREND_THRESHOLD`, the
    indicator lengths, the four-way encoding and `classify()` itself are
    IMPORTED from `mdlib/regimes.py`; the schema labels are inverted out of
    `portfolio.config_loader.CANONICAL_QUADRANT`. A live daemon and a backtest
    that disagree about what Q1 means is the failure nothing downstream can
    detect: a strategy certified in High-Vol/Trending would be stood down in
    the environment it was certified for and turned loose in the one it never
    traded, with every log line reading correctly.
  - **The ADX comparator is strictly `>`, not `>=`.** The specification for
    this module was written `ADX >= 25`; `mdlib/regimes.py` and
    `backtest/profiler.py` have always used `>`, and every cached quadrant,
    every Stage 1 designation and every Gate R verdict was drawn on it.
    Matching the wording in the live daemon alone would move the boundary there
    and leave the backtests behind it. At ADX exactly 25.00000 the daemon says
    Ranging.
  - **theta_vol is LOADED, never computed.** It is the pinned in-sample median
    ATR(14) written into each `{SYMBOL}_{TF}_regime.parquet` by
    `mdlib.regimes`, read back through `provenance()`, and overridable by an
    operator anchors file (`$BT_THETA_ANCHORS`, else
    `config/theta_vol_anchors.json`). A symbol with no anchor raises
    `ThetaAnchorMissing` and is not classified. Taking a median of the live
    window instead would make the boundary a property of the REQUEST: on a
    quiet morning every bar reads high-volatility, and a Q1-certified strategy
    is handed permission for a market it is not in, with a plausible-looking
    equity curve behind it.
  - **An anchor is per (symbol, TIMEFRAME).** NQ's is 7.90 at 15m and 11.33 at
    30m — the same tape, a 43% different boundary — so an anchor applied at the
    wrong timeframe silently relabels roughly a third of the session. The
    timeframe is part of the key, part of the state file and part of every
    error message. **ES and CL have no regime cache on this box**, so they
    cannot be classified until `scripts/precompute_regimes.py` has run for
    them; the daemon says so on stderr at construction rather than at the first
    signal.
  - **Micros resolve to their full-size parent** (`MNQ`→`NQ`, `MES`→`ES`,
    `MCL`→`CL`, `MGC`→`GC`) because they quote the same price series at the
    same tick size — only the multiplier differs, and a multiplier appears
    nowhere in an ADX or an ATR. The tick sizes are RECONCILED against
    `backtest/specs.py` on every construction rather than asserted in a
    comment: were they ever to differ, every ATR comparison for that contract
    would be wrong by the same factor, in price units, silently.
  - **The warm-up is `Q0_UNDEFINED_WARMUP`, never Q4.** `NaN > theta` is False,
    so the naive encoding files every warm-up bar under Low-Vol/Ranging — a
    populated label on bars where no indicator exists. Fewer than
    `2 x ADX_LENGTH + 1` bars reports Q0 with NULL indicators, and a consumer
    must treat it as "no regime" exactly as `mdlib.regimes` requires of
    quadrant 0.
  - **The ML gate fails loud, not open and not closed.** No model registered
    for a strategy returns **True** — the documented pass-through that keeps a
    rule-based Version A trading. A model that is registered but fails to load,
    declares no feature order, or is missing a feature the caller did not
    supply **raises `MLGateError`**. True there would trade unfiltered while
    every log line read "ML confirmed"; False is indistinguishable from a model
    that looked and vetoed. Both are silent. Feature ORDER comes from the
    model's sidecar and never from the caller's dict — see `models/README.md`.
  - **Every record carries `bar_ts` beside `updated_at`.** The first is the
    market's clock and the second is the daemon's; a daemon looping over a dead
    feed keeps `updated_at` fresh forever while `bar_ts` stops, and telling
    those apart is the job.

- **`regime_reader.py`** — `get_current_regime(symbol)`, and deliberately
  dependency-light: `json`, `os`, `pathlib`, `datetime`, and nothing else. It
  imports neither pandas nor the daemon, so it keeps answering when the
  daemon's numeric stack is what is broken. Non-blocking with no locks, because
  the daemon publishes through `os.replace` and a reader sees the previous
  complete document or the new one. **It returns no defaults**: a missing file
  or an unpublished symbol RAISES, since `{"regime": None}` becomes "not in the
  permitted quadrant" downstream and stands a strategy down for a missing file
  in a way that looks exactly like a market that moved. `age_seconds` and
  `bar_age_seconds` are reported on every read and `max_age_s` turns either
  into a refusal; `is_regime_permitted` accepts an id or a schema label and
  never permits Q0.

- **`crosstrade_formatter.py`** — the two wire forms, and it SENDS NOTHING;
  `live/dispatcher.py` remains the only module in this repository that puts an
  order on the wire. `format_crosstrade_command` builds the semicolon
  plain-text place order (upper-cased, carrying `key` and `tif`),
  `format_crosstrade_json` the structured object (lower-cased, carrying
  `strategy_tag` and no key — that endpoint takes the credential in the
  request, and a key in the body is a key in every payload log), and
  `format_flatten_command` the flatten, which carries no side and no quantity
  because a flatten closes whatever is open and a wrong guess at the position
  opens the opposite one. Field ORDER in the text form is part of the contract.
  Legal sides and order types are IMPORTED from `live.dispatcher` rather than
  restated; MARKET only, since none of these signatures carries a price and a
  LIMIT defaulted to the market turns a bounded entry into an unbounded one.
  `redact()` exists because a log file outlives the session that wrote it and a
  command copied out of one is directly replayable.

**`master_live.py` and `realtime/live_dispatcher.py`** — the end-to-end live
execution loop, added 2026-08-23. `LiveExecutionDispatcher` is the pipeline
(signals → regime gate → ML gate → netting and ATR sizing → CrossTrade);
`master_live.py` is the CLI, the interval loop and the shutdown handling, so
the wiring is unit-testable without a clock. `process_bar_cycle` returns the
per-stage record — signals, declines, ML vetoes, exits, plan, payloads,
dispatches, errors — and a four-key summary over it (`processed_at`,
`evaluated_signals`, `approved_signals`, `orders`) DERIVED from those records
rather than counted beside them: a counter incremented alongside a list is one
early return away from disagreeing with the list it summarises, and the
summary is the half a caller reads. `orders` IS `payloads`, the same list, so
the two cannot disagree about what was sent; a FLAT record is an evaluation
that produced no entry and is never counted as an approval.

- **THE PIPELINE OPENS POSITIONS AND CANNOT CLOSE THEM.** `PortfolioManager`
  does not know what is open and never emits FLATTEN, and nothing here invents
  the position state it would need to. An EXIT signal on the last bar is
  COUNTED and REPORTED (`exit_signals`) and is not turned into an order, so a
  loop run without something reconciling positions on the CrossTrade side
  accumulates entries and never leaves. That is a deliberate stopping point:
  closing a position this process cannot see would shut positions it never
  opened, and the failure is silent in the direction that costs money.
- **Two processes, one direction.** This loop does NOT classify regimes —
  `realtime/regime_daemon.py` publishes the state file and this reads it. A
  slow indicator pass can therefore never stall a dispatch, and the loop's view
  of the market is a file somebody can inspect afterwards rather than a value
  that existed for one millisecond inside a process that has exited. With no
  state file every signal is declined, which is correct: an unknown environment
  is not a permitted one.
- **The regime is checked twice and the two cannot disagree.** Once per
  (strategy, symbol) before a signal is emitted — so a decline is recorded
  against the STRATEGY with its reason rather than disappearing into an empty
  payload list — and once inside `build_order_plan`, which is the authoritative
  gate. Both read the same state file through the same reader and compare
  against the same `derived.canonical_quadrants`; removing the early one would
  change no order. **A regime decline cannot change the net** (the gate is per
  portfolio+symbol, so every strategy on that pair gets the same verdict), but
  **an ML veto CAN**: vetoing one side of an opposing pair turns a position
  that would have netted flat into a live order. `ml_vetoes` is what makes that
  visible.
- **`active_strategies` grants permission; `approved_incubator/` supplies the
  code.** Being in the directory is explicitly not permission to trade. **The
  promoted code is SHA-256 checked against its `meta.json` before it is run** —
  that hash exists so a promoted file provably IS the file the metrics
  describe, and a live loop that ignored it would trade an edited module under
  a certified name. The certified `symbols` are checked too, through the same
  micro/full-size alias: a strategy certified on ZS cannot trade MNQ because a
  config line put them in one basket.
- **The bar feed resolves micros to their parent.** The baskets hold
  MNQ/MES/MCL/MGC and the lake holds only the full-size contracts, so without
  the alias this loop reads nothing and no-ops forever — looking exactly like a
  market with no signals. Same price series, same tick size, reconciled against
  `backtest/specs.py`; the substitution is PRINTED, and the order is still for
  the micro and sized on the micro's own point value.
- **The stop multiplier when contributors disagree: the WIDEST wins.**
  `build_order_plan` takes one `stop_atr_mult` per symbol while `sl_atr_mult`
  belongs to a strategy, so the plan is built PER PORTFOLIO and within it the
  widest stop is used — the tightest would over-size relative to the strategy
  holding the wide one and breach its budget. `stop_atr_mult_source` records
  the value, whose it was, and everything that was in contention.
- **Retries are narrow on purpose.** A retried market order is a DUPLICATE
  POSITION this process cannot undo, so a retry happens only when the error
  proves nothing reached the broker (refused connection, DNS failure, no route
  to host). **A timeout is never retried, and neither is a 5xx** — both leave
  the outcome unknown. An unrecognised error defaults to NOT retrying, so a new
  upstream failure mode cannot silently become a duplicate order.
- **`--dry-run` runs every stage except the socket**: strategies loaded and
  hash-checked, signals computed, both gates applied, positions netted and
  sized, payloads formatted and validated. Run it first and after any config
  change. **It is also the DEFAULT, and `--live` is the only flag that arms
  the socket** — the mode an operator gets by forgetting a flag has to be the
  recoverable one, because a market order this process cannot see is not
  undone by noticing the mistake. Asking for both is REFUSED rather than
  resolved: either resolution leaves half the command describing a run that
  did not happen, and the wrong half is the one about whether real orders went
  out. **Live mode refuses to START without a webhook URL** rather than
  discovering it at the first order — which on a console reads exactly like a
  quiet market. Credentials come from `.env` (`CROSSTRADE_WEBHOOK_URL`,
  `CROSSTRADE_API_KEY`) via a fifteen-line reader rather than a new pinned
  dependency, and are never put into `os.environ`, where every subprocess would
  inherit them. Only the webhook's HOST is ever printed and every logged
  command is redacted.
- **`live/dispatcher.py` is still the only module that sends.** This one calls
  `send_execution_signal`, retries it under the rule above, and accepts an
  injected sender only so the tests stay off the network.
- **SIGINT/SIGTERM set a flag and never interrupt a cycle** — an exception
  between the send and the print leaves an order on a broker with no line in
  the log saying so. The cycle finishes, the sleep is skipped, and the last
  thing on the console is always a complete cycle; a second signal exits at
  once. No sockets are held between cycles, so there is nothing to drain.

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

## Commands

```bash
# THE MASTER REGIME DAEMON. Classifies LIVE bars into the same four quadrants
# the pipeline certifies against, and publishes them to data/live_regime_state.json.
# Reads no lake and runs no backtest; the caller owns the bar feed.
python3 realtime/regime_daemon.py            # what is wired up: anchors, models
python3 realtime/regime_reader.py            # what is currently published
python3 realtime/regime_reader.py --symbol NQ

# THE LIVE EXECUTION LOOP. Signals -> regime gate -> ML gate -> netting and
# ATR sizing -> CrossTrade. Reads the regime state the daemon publishes; it
# does NOT classify. A dry run is the DEFAULT and runs every gate and formats
# every payload but opens no socket; --live is the only flag that arms it.
python3 master_live.py --dry-run --once            # one cycle, nothing sent
python3 master_live.py --dry-run --interval-sec 30
python3 master_live.py --interval-sec 60           # ALSO A DRY RUN
python3 master_live.py --live --interval-sec 60    # LIVE. SENDS REAL ORDERS.
#   --config/--state-file  the routing table and the published regime state
#   --live                 send real orders. Without it nothing reaches the
#                          wire; --dry-run and --live together are REFUSED
#   --max-regime-age-sec   refuse a regime reading older than this
#   --no-verify-hash       skip the meta.json SHA-256 check. Do not use this
#                          to trade an edited module.
```
