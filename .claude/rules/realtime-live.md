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
  - "tests/test_live_feed.py"
  - "data_pull/databento_live.py"
  - "tests/test_nt8_feed.py"
  - "tests/test_nt8_listener.py"
---

# The live regime service and the execution loop

**`realtime/`** — the live regime service, added 2026-08-23. Sits ABOVE
`backtest/` and beside `portfolio/`; **nothing in `backtest/` may import from
here**, for the same reason nothing there may import from `portfolio/` — a
research number that depended on live account or feed state would be tuned to a
broker rather than to a market. Three modules, split by what they do: the
daemon WRITES, the reader READS, the formatter FORMATS and sends nothing.
`contract_alias.py` is a fourth, and holds only data: the micro -> full-size
table every tier resolves symbols through.

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
    error message. **CL, ES, GC and NQ all load pinned anchors from
    the regime cache** over the in-sample window 2013-01-01..2022-12-31 -
    verified 2026-08-25 at 15m and 1h, and the 1m/2m/5m/30m caches exist too.
    A symbol with no cache still cannot be classified and the daemon says so on
    stderr at construction rather than at the first signal; run
    `scripts/precompute_regimes.py` for any (symbol, timeframe) before trading
    it, because an uncached pair falls back to a live median rather than the
    pinned anchor.
  - **Micros resolve to their full-size parent** (`MNQ`→`NQ`, `MES`→`ES`,
    `MCL`→`CL`, `MGC`→`GC`, `M2K`→`RTY`, `MYM`→`YM`) because they quote the
    same price series at the same tick size — only the multiplier differs, and a multiplier appears
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
  dependency-light: `json`, `os`, `pathlib`, `datetime`, and
  `realtime.contract_alias`, which is a dict and four functions over the
  standard library. It imports neither pandas nor the daemon, so it keeps
  answering when the daemon's numeric stack is what is broken. Non-blocking with no locks, because
  the daemon publishes through `os.replace` and a reader sees the previous
  complete document or the new one. **It returns no defaults**: a missing file
  or an unpublished symbol RAISES, since `{"regime": None}` becomes "not in the
  permitted quadrant" downstream and stands a strategy down for a missing file
  in a way that looks exactly like a market that moved. **A micro resolves to
  its full-size parent** through the shared table — `get_current_regime("MNQ")`
  answers from NQ's record, since the daemon publishes the contract the history
  and the pinned anchor belong to. The exact symbol always wins if it is
  published, the parent's record has to exist, and the answer carries
  `requested_symbol` / `resolved_symbol` / `symbol_aliased` so the substitution
  is on the record rather than inferred. Without it every micro-denominated
  basket read as "no live regime reading" on a full state file, which is the
  same stand-down a missing daemon produces. `age_seconds` and
  `bar_age_seconds` are reported on every read and `max_age_s` turns either
  into a refusal; `is_regime_permitted` accepts an id or a schema label and
  never permits Q0.

- **`crosstrade_formatter.py`** — the two wire forms, and it SENDS NOTHING;
  `live/dispatcher.py` remains the only module in this repository that puts an
  order on the wire. `format_crosstrade_command` builds the semicolon
  plain-text place order (upper-cased, carrying `key` and `tif`),
  `format_crosstrade_json` the structured object (lower-cased, no key — that
  endpoint takes the credential in the request, and a key in the body is a key
  in every payload log), `format_flatten_command` the flatten, which carries
  no side and no quantity because a flatten closes whatever is open and a wrong
  guess at the position opens the opposite one, and
  `format_account_flatten_command` the ACCOUNT-WIDE kill — no instrument, no
  tag, `emergency_halt` its only caller. Field ORDER in the text form is part
  of the contract.
  - **THE TEXT COMMAND IS WHAT GOES ON THE WIRE, for every order without
    exception.** The configured `/v1/send/` webhook parses the semicolon form
    and refuses a JSON object with an HTTP 400 — 125 refused entries on
    2026-08-31, none ever accepted, while dry run reported success because dry
    run opens no socket. `dispatch_order` was corrected then and
    `_send_with_retry` was not, so until 2026-09-01 every FLATTEN was still
    POSTed as a JSON object: refused by the endpoint, reported as sent, and the
    position left open — the failure direction that costs money, on the one
    path whose job is to close. The JSON object is still BUILT and kept on the
    record as evidence of what the other endpoint would have been handed; it is
    not sent. `tests/test_live_dispatcher.wire_fields` asserts the form before
    it asserts the contents, so a payload that reverts to a dict fails loudly.
  Legal sides and order types are IMPORTED from `live.dispatcher` rather than
  restated; MARKET only, since none of these signatures carries a price and a
  LIMIT defaulted to the market turns a bounded entry into an unbounded one.
  `redact()` exists because a log file outlives the session that wrote it and a
  command copied out of one is directly replayable.
  - **`strategy_tag` is CrossTrade's LOCK and every form carries it.** The
    order may act only on the position that tag opened, and the lock clears
    when a matching order closes it — matched by STRING EQUALITY. It lived on
    the JSON object alone until 2026-09-01, which is the form that is not on
    the wire (see above), so every live entry went out untagged while the cycle
    report showed a tag. An entry tagged and
    an exit untagged, or tagged differently, does not close the position: the
    flatten is sent, accepted and logged, and nothing is released.
    `live_dispatcher.compose_strategy_tag` is the ONE composer — the entry
    builds `portfolio:strategy_a+strategy_b` from the plan's contributors and
    the exit rebuilds the identical string from the POSITION BOOK, because a
    netted position took out one lock named for all of its contributors and
    flattening it under the exiting strategy's own id names a lock that never
    existed. `;`, `=` and whitespace are STRIPPED (`sanitize_strategy_tag`) —
    they are the text form's field separators — and a tag that is nothing but
    separators is refused rather than reduced to `""`, since an order that
    silently lost its lock is the failure the field exists to prevent. No tag
    appends no field at all; `strategy_tag=;` would be a lock on `""` shared by
    every untagged order on the account.

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

- **`realtime/position_book.py` is the ENTRY gate, and it is a SUBCLASS of
  the netted book rather than a second one.** Added 2026-09-01.
  `dispatch_order` consulted nothing on the way in, so a strategy whose signal
  stayed true did not enter once — it entered on EVERY cycle, one position per
  interval, none of which this loop could see as one position. `can_execute`
  is keyed by `(portfolio_id, symbol)` exactly as the book it extends: BUY is
  permitted on FLAT or SHORT, SELL on FLAT or LONG, CLOSE/FLATTEN on anything
  but FLAT, and an unrecognised action is refused. The refusal is a record and
  a console line — `HOLD {portfolio}/{symbol} {action} — position already
  active. Preventing stack.` — on `report["held"]`, because a cycle that held
  every entry and a cycle that produced no signal both print no dispatch line
  and are different facts about the account.
  - **It had to be the SAME OBJECT the exit path writes.** `record_fill`,
    `record_flat` and `plan_exits` belong to `PositionBook`; a second
    dictionary of positions — even one keyed identically — would be written by
    the entry path and not by the exit path, and would be wrong from the first
    flatten onward, refusing entries on a position already closed with every
    log line reading correctly.
  - **The key is the PAIR, never the strategy.** Sixteen strategies trade NQ
    inside Incubator-Odd and they net to ONE position, so strategy B's BUY on a
    pair strategy A opened is the same stack under a different name. A
    strategy-keyed book would have permitted it and called the result two
    positions.
  - **It knows only what THIS PROCESS opened**, so a restart permits an entry
    on a position a previous run opened. The firewall's
    `max_contracts_per_symbol` reads the durable `EngineState` and is what
    covers that; the two are complementary and neither replaces the other.
  - **A reversal is permitted and has a sharp edge.** A BUY of one contract
    against a short of one NETS FLAT at the broker rather than opening a long,
    so the book recording LONG after it is a claim about intent, not about the
    account. Reverse by flattening and then entering.
- **`MAX_QTY = 1` CLAMPS, it does not reject.** Changed 2026-09-02; it
  rejected until then. An order asking for more is reduced to `max_qty` and
  SENT, carrying its `strategy_tag`. **THE ATR SIZER ROUTINELY ASKS FOR MORE
  THAN ONE** — 5 contracts on the test fixture's MNQ — so this is the normal
  path, not an edge case, and what goes on the broker is NOT the order the
  sizer computed: the position is sized against a stop drawn for five and the
  risk model no longer describes the trade, while the record reports success.
  That is the accepted trade-off — a 1-lot expression of the edge over none —
  and it is survivable only because the reduction is never silent. The dispatch
  record carries `quantity` (what was SENT), `requested_quantity` (what the
  sizer ASKED for), `clamped: True`, `rule: "MAX_QTY"`, `detail: "sizer asked
  N, cap is 1"` and the `warning` line; the console prints `CLAMPED {symbol}
  order from {N} to 1 due to MAX_QTY cap.` on stderr and annotates the cycle
  summary's dispatch line, which would otherwise show `x1` for a clamped 5-lot
  and a genuine 1-lot alike. **A clamp does NOT land on `risk_refusals`** —
  `master_live` prints everything on that list as `RISK BLOCKED` and counts it
  a failure, and this order was sent. The caller's payload dict is COPIED, not
  rewritten, so the cycle report's `orders` still shows what was asked for. It
  duplicates the firewall's configurable `max_contracts_per_order` on purpose:
  the firewall is OPTIONAL (`firewall=None` is the default and every
  construction outside `master_live.py` leaves it there) and this cap is in the
  send path itself. `max_qty=` on the constructor exists only so the suite can
  exercise the wire; nothing in production passes it.
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
  micro/full-size alias — a certification on `['NQ']` authorizes MNQ and the
  reverse — but a strategy certified on ZS cannot trade MNQ because a config
  line put them in one basket.
- **`realtime/feed.py` is where bars come from, and the forming-bar rule lives
  there and nowhere else.** Added 2026-08-25. `load_symbol_bars` used to inline
  a lake read, which is why the loop could poll every sixty seconds against a
  tape that had stopped eighteen days earlier and report nothing wrong. One
  interface (`BarFeed.closed_bars`), two implementations (`LakeFeed`,
  `LiveFeed`), and `--feed auto|live|lake` on both `master_live.py` and the
  regime daemon's `--publish`.
  - **A bar is stamped when it OPENS**, so the 14:00 bar on an hourly feed
    covers 14:00–15:00 and is unfinished until 15:00. Acting on it is
    lookahead — a decision made with information from inside the interval being
    traded — and it is what makes a live curve diverge from its backtest for
    reasons nobody can find afterwards. `drop_forming_bar` removes EVERY
    unfinished interval, not just the last: after a reconnect a feed can return
    several, and dropping only the tail leaves a half-formed bar looking
    complete. It is a DROP, never a fill-forward and never a "close so far".
  - **The cut is at the EARLIER of the clock and the feed's horizon.** A vendor
    publishing on a lag can leave an interval whose end has passed but whose
    last minutes have not arrived: at 17:05, with data published to 16:50, the
    16:45 quarter-hour is finished by the clock and built from five minutes of
    bars — wrong high, wrong low, wrong close, a fifth of the volume, and it
    reads as a quiet quarter of an hour. `BarFeed.horizon()` makes a lagging
    feed produce FEWER bars rather than wrong ones.
  - **Live bars are aggregated by the LAKE's resampler** (`mdlib.lake.DERIVED`,
    `OHLCV`, `_resample`, imported), so a live 15m bar is assembled exactly as
    the 15m bar the strategy was certified on. A second aggregation would
    differ in the last decimal, which is enough to move a bar across an
    indicator threshold with nothing in any log to explain it.
  - **`--feed live` REFUSES when none is configured** rather than falling back.
    An operator who typed it and silently got fortnight-old bars would read a
    rehearsal as a live session, and a stale tape produces no signals — which
    looks exactly like a quiet market. `auto` falls back and says so;
    `describe()` names the feed before the first cycle and every cycle prints
    how long ago the newest bar CLOSED.
  - **The choice is NOT tied to `--dry-run`.** Dry-run is about whether a
    socket opens; a dry run against stale bars cannot rehearse a decision the
    loop would make now.
- **LIVE MARKET DATA COMES FROM THE BROKER, and Databento is historical
  only.** `realtime/nt8_feed.py`, 2026-08-25. The bars a strategy decides on
  are then the bars its orders execute against — same feed, same session
  template, same clock — and a research vendor that disagreed with a broker
  fill about a bar's close would produce slippage nobody could source. Nothing
  in `realtime/` or `master_live.py` imports `databento`, and
  `tests/test_nt8_feed.py` asserts that on the source rather than trusting it.
  `data_pull/databento_live.py` survives as a gap-fill and verification tool
  and is explicitly not the live path.
  - **The transport is a SPOOL DIRECTORY**, `/mnt/backtest/artifacts/nt8_bars/`
    or `$BT_NT8_SPOOL` — the same shared-folder channel NT8 already uses to
    deliver fill logs. A NinjaScript add-on appends one line per closed bar;
    this reads the tail. Chosen over an HTTP listener deliberately: a listener
    is a port, a process to supervise and a silent hole when it dies, while a
    file that stops growing is visible to `ls`, survives a restart on either
    side and replays after an outage. A push transport added later should
    WRITE this spool rather than bypass it.
  - **`ts` IS THE BAR'S CLOSE TIME, and this repository stamps the OPEN.**
    NinjaTrader stamps the bar that ran 16:00–17:00 as `17:00`; the lake
    resamples `label="left"` and every certification was computed that way.
    Ingested as-is, every bar shifts one period, every indicator is computed on
    misaligned data, and nothing raises. One subtraction on ingest, driven by
    the timeframe. A file may DECLARE the other convention with a `# stamp=open`
    header — in the file, because the NinjaScript is what knows, and an
    operator who changes it should not have to remember a flag on the other
    side of the mount.
  - **A naive timestamp is REFUSED, not assumed to be UTC.** NT8 writes in the
    instrument's or the workstation's timezone unless the script converts, so a
    stamp with no offset is as likely to be New York as UTC — and guessed wrong
    the series shifts by hours while still looking like a market: bars in order,
    prices sane, sessions the wrong length.
  - **A written bar is not trusted to be a closed bar.** The publisher is meant
    to append on close; every frame still goes through the one forming-bar
    rule, because a script switched to `Calculate.OnEachTick` would otherwise
    start appending live bars and nothing downstream would notice.
  - **A missing spool RAISES.** No publisher is not a quiet market. A directory
    that exists but holds no file for a symbol leaves that symbol absent, which
    the loop reports as "no bars" rather than as "no signal".
  - **THE NINJASCRIPT PUBLISHER IS NOT IN THIS REPOSITORY** and does not exist
    yet. It runs on the Windows workstation and must honour the contract in
    `realtime/nt8_feed.py`: one file per (symbol, timeframe) named
    `{SYMBOL}_{TF}.csv`, header `ts,open,high,low,close,volume`, appended on
    bar close, ISO-8601 timestamps carrying `Z` or an explicit offset. Until it
    runs, `--feed nt8` refuses and `auto` falls back to the lake.
- **`realtime/nt8_bar_listener.py` is the PUSH transport, and it WRITES THE
  SPOOL.** Added 2026-08-25. `POST /api/bars` takes one closed bar (or a list,
  to drain a backlog after a reconnect) and appends it to the same
  `{SYMBOL}_{TF}.csv` the feed reads; `GET /health` is what `trading-watchdog`
  polls. It is the transport `nt8_feed.py` said to build if a push channel were
  ever added — *write this spool rather than bypass it* — so `--feed nt8` and
  `load_symbol_bars(source="nt8")` mean exactly one thing whether a bar arrived
  over HTTP or was appended by a NinjaScript.
  - **A private bar store would have cost four things at once**: the
    forming-bar drop, the micro alias, aggregation through the LAKE's
    resampler, and the single `ts` conversion. All four live behind
    `realtime/feed.py`, and a second store read directly by `load_symbol_bars`
    reaches a strategy having skipped every one of them.
  - **The listener does NOT convert `ts`.** It writes the timestamp through
    unchanged and declares the convention in the file's `# stamp=` header;
    `read_spool` does the one subtraction. Converting in both places shifts
    every bar a period the other way, and nothing raises. The default is
    `close` because that is NT8's, it is PRINTED at startup and echoed on every
    accepted bar, because the author of the NinjaScript is who has to notice it
    is wrong.
  - **One file carries one convention.** `read_spool` takes the LAST `# stamp=`
    it sees, so a payload DECLARING a stamp that disagrees with the file's
    header is refused (409) rather than appended — it would re-interpret every
    bar already written. A payload that declares nothing has merely inherited
    this process's default, and the FILE's header outranks that, for the same
    reason it outranks it on read.
  - **A duplicate timestamp is idempotent, not an error.** A publisher retrying
    a request whose response was lost must not crash-loop and must not
    double-write. It is reported as `duplicate` and COUNTED: a publisher
    sending only duplicates is broken in a way that otherwise looks identical
    to a working one. The duplicate index and the out-of-order high-water mark
    are both re-seeded FROM THE FILE at startup — the ring buffer is memory and
    the spool is the record.
  - **What is refused at the door**, because nothing downstream re-checks it: a
    naive timestamp, a timestamp off the timeframe's grid, an incoherent bar
    (high below the close, negative volume, a non-finite price), and a
    timeframe the lake cannot build. The forming-bar rule is NOT applied here —
    it belongs to the reader and already runs there for every feed, and putting
    it in two places lets them drift.
  - **`/health` reports both clocks.** `last_bar_utc` is the market's and
    `last_post_utc` is this process's; a publisher looping over a dead feed
    keeps the second fresh forever while the first stops. `STARVED` (nothing
    ever received, and a 503) is distinct from `HEALTHY` — a health endpoint
    that says HEALTHY before its first bar is a watchdog that can never fire.
    `STALE` is also what a shut market looks like; the watchdog owns the
    session calendar that separates them.
  - **THE PORT IS A REAL EXPOSURE.** It binds `0.0.0.0:8000`, so anything that
    can route to the box can inject bars that live strategies decide on. Set
    `$BT_NT8_LISTEN_TOKEN` to require an `X-NT8-Token` header, and prefer
    binding the interface the NT8 workstation is actually on.
  - Built on **Starlette**, which is already pinned and is the ASGI layer
    FastAPI is built on. FastAPI is not installed and this is two routes over a
    JSON body; the endpoints and payloads are unchanged by that choice.
- **The bar feed resolves micros to their parent.** The baskets hold
  MNQ/MES/MCL/MGC and the tape is the full-size contract's, so without the
  alias this loop reads nothing and no-ops forever — looking exactly like a
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
- **`emergency_halt` flattens the ACCOUNT, not the instrument.** The switch is
  armed FIRST and unconditionally, so a failed flatten still stops the next
  entry. Then `command=flatten; account=...;` — no instrument, no tag, ONE per
  account, because the account command closes everything and a second for the
  next symbol is a duplicate kill that reads as a failure. **It closes
  positions this process did not open**, which the ordinary exit path refuses
  to do; a halt inverts that trade-off deliberately, since a kill that closes
  only what this loop can name leaves behind exactly the positions nobody can
  account for. WHICH accounts is still decided by the position book, so a halt
  cannot reach an account this loop was not trading, and unverified claims from
  a previous run are still reported rather than acted on. `open_positions()`
  returns a LIST — it was iterated as `.items()`, raising `AttributeError`
  inside the halt: every flatten skipped, the switch armed, and the alert
  reporting "flattened: none". The book is cleared only for a flatten that
  SUCCEEDED. `dispatcher.account_for()` maps portfolio → account, because the
  book is keyed by portfolio and a kill addressed to a portfolio id names an
  account that does not exist.
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
python3 master_live.py --dry-run --once --tf 1h --feed nt8    # the broker feed
python3 master_live.py --dry-run --once --feed lake           # against history
BT_NT8_SPOOL=/path/to/spool python3 master_live.py --feed nt8 --dry-run --once

# THE PUSH RECEIVER. NT8 POSTs closed bars; this appends them to the spool
# above, so `--feed nt8` reads pushed and file-appended bars identically.
# Binds 0.0.0.0:8000 — set $BT_NT8_LISTEN_TOKEN before exposing it.
python3 realtime/nt8_bar_listener.py
python3 realtime/nt8_bar_listener.py --port 8000 --spool-dir /mnt/backtest/artifacts/nt8_bars
#   --stamp close|open     what a posted `timestamp_utc` MEANS. Default close
#                          (NT8's own). Written into the file header; the
#                          conversion happens on READ, in read_spool, once
#   --stale-after-bars N   /health reports STALE after N bar widths of silence
curl -s localhost:8000/health
# Databento is HISTORICAL: gap-fill and verification, never the live path.
python3 data_pull/databento_live.py --symbol NQ --minutes 120
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
