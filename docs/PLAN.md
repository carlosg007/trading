# Trading Research & Deployment Plan

Drafted 2026-08-08. Living document — revise as decisions change.

---

## Goal

Build and test many algorithmic futures strategies, from simple rules through
machine learning, and combine the survivors into a diversified portfolio.

Two portfolios with different constraints (see below): intraday for prop firm
accounts, swing for personal capital.

Secondary goal, explicitly stated: **learn**. Prefer understanding the
machinery over the fastest path to a result.

---

## Environment

**Server:** `backtest` — Ubuntu 26.04 LTS on Proxmox, 192.168.102.119, 100G disk.

**Storage:** NFSv3 mount at `/mnt/backtest` from UNAS Pro (192.168.100.10), 16TB free.

Mount options are deliberate. `hard` not `soft` — a soft mount returns an I/O
error on timeout, which can truncate a Parquet write into a corrupt file that
still half-parses. `timeo=600` pairs with `hard`. NFSv3 only; UniFi Drive does
not export v4.

Ownership on the share shows a bogus numeric UID (no v3 idmapping).
Permissions are enforced server-side. **Do not chown/chmod the mount.**

**Throughput:** measured 107 MB/s (1GbE line rate) despite both ends showing
2.5 GbE on the switch. Traffic routes between VLAN 102 and VLAN 100 through
the UDM Pro, which is the likely bottleneck. Not pursued — adequate for bars.

**Python:** 3.13 via `uv`, not the system 3.14 (scientific stack wheels lag a
new minor by months). Confirmed working: duckdb 1.5.5, pyarrow 25.0.0,
pandas 3.0.5.

**Venv:** one for the whole project at `~/src/trading/.venv`.

---

## Directory layout

Code on local disk, data on NFS. Never put a venv on NFS — Python imports
thousands of small files and each is a network round trip.

    ~/src/trading/
      mdlib/         lake access; owns session and roll definitions
      data_pull/     vendor downloaders (the ONLY place that talks to a vendor API)
      backtest/      engine: fills, costs, metrics
      strategies/    signal logic only
      scripts/       utilities
      courses/       study material

    /mnt/backtest/
      raw/{futures,equities,options,crypto,forex}    original vendor files, write-once
      lake/futures/bars/symbol=X/tf=1m/year=Y/month=M/
      reference/futures/                             roll calendars, contract specs
      artifacts/                                     backtest outputs

`raw/` is immutable so the lake can be rebuilt after a parser bug without
re-downloading. Vendor name lives in the filename, not a directory level.

**DuckDB catalog stays on local disk** (`~/.local/share/mdlib/`). NFSv3 with
`local_lock=none` sends file locking over NLM, which is not safe for DuckDB.
The catalog is only an index over Parquet — rebuildable.

---

## Data

**Source:** Databento, `GLBX.MDP3` (CME Globex). Standard plan, $199/month.
Includes 16+ years of L0. Dataset begins 2010-06-06.

**Continuous contracts:** `ES.v.0` — volume roll, front month. Databento does
**not** back-adjust; prices are raw with a gap at each roll. This is the right
choice here: back-adjustment shifts historical prices away from where they
actually traded, which breaks any strategy anchored on absolute price levels.
Roll calendars are saved to `reference/futures/` so backtests can exclude
roll days.

**Symbols (30):**

| Category | Symbols |
|---|---|
| Equity index | ES, NQ, RTY, YM |
| Rates | ZT, ZF, ZN, ZB |
| Energy | CL, RB, HO, NG |
| Metals | GC, SI, PL |
| Grains | ZC, ZW, ZS |
| Softs | KC, SB, CT |
| Livestock | LE |
| FX | 6E, 6B, 6A, 6C, 6J, 6S |
| Crypto | BTC, ETH |

Micros (MES, MNQ, MBT, MET) deliberately excluded — same markets at a
different multiplier. Get micro results by changing the contract multiplier
in position sizing.

**Schemas stored:** `ohlcv-1m` and `ohlcv-1d`, both native.

Daily is pulled natively rather than derived because CME sessions run
18:00–17:00 ET; resampling 1m on UTC calendar days splits the session in the
wrong place. Databento uses correct exchange boundaries.

**Derived timeframes** (not stored — resampled on demand by `mdlib`):

    1m (native) → 30m, 1h, 4h
    1d (native) → 1w

Hourly is derived rather than pulled because an hour has no session
semantics — it is just sixty minutes, so there is no convention to get wrong.
Deriving also lets you choose the boundary (e.g. 09:30-aligned) and
guarantees consistency with the minute data the backtest fills on.

**Known data quality issue:** Databento flags some sessions as "degraded."
Observed clustering on high-volatility days — 2020-02-27, 2020-02-28 (COVID
crash), 2020-06-30 (quarter-end roll), 2024-09-18 (FOMC 50bp cut). This is
expected: exchange feeds get stressed exactly when it matters most.

**Action required:** extract the full list from the pull log and store it at
`reference/futures/degraded_days.csv`. The backtest engine should flag any
trade touching one of these dates. A strategy that looks robust through 2020
because the worst bars are missing is a dangerous conclusion.

    grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} \(degraded\)" ~/pull.log | sort -u

**Second data source:** NinjaTrader 8 export, ~2 years available.

Use this for **data-source agreement testing**, not out-of-sample validation.
Two years on daily bars is ~500 observations across one regime — far too few
to distinguish edge from luck, and no bear market in the window.

Real out-of-sample comes from holding out the last 3 years of Databento data
(develop on 2010–2022, never look at 2023–2026 until finished).

NT8 export will need `data_pull/ingest_nt8.py` to normalize its CSV format,
writing to a separate path (`lake/futures_nt8/`) so the two sources can never
be accidentally mixed in a query.

---

## Two portfolios

FundedNext does not permit overnight holds. Swing strategies are therefore
structurally impossible on prop accounts. Rather than abandon either
approach, run two portfolios against shared infrastructure.

### Portfolio A — Intraday (prop accounts)

| | |
|---|---|
| Timeframes | 30m – 1h |
| Constraint | **Flat by session close. No overnight, no weekend.** |
| Capital | FundedNext prop accounts (5 max, per household) |
| Deployment | NT8 bridge (see Deployment) |
| Binding limit | Transaction costs, trailing drawdown rules |

### Portfolio B — Swing (own capital)

| | |
|---|---|
| Timeframes | 1h – weekly |
| Constraint | None beyond normal risk management |
| Capital | Personal |
| Deployment | Broker API (direct, no NT8) |
| Binding limit | Sample size |

**Shared:** data lake, `mdlib`, backtest engine, strategy code, metrics.
The constraint set is a parameter passed to the engine, not a separate
codebase.

---

## Timeframes and the cost tradeoff

ES costs roughly 0.25 points round trip in spread plus commission.

On daily bars where the average winner is ~20 points, costs are ~1% of gross
edge. On 1-minute bars where the average winner is ~1 point, costs eat ~25%.
Portfolio A sits in between and must clear a meaningfully higher per-trade
edge than Portfolio B to be viable.

**Derived timeframes** (not stored — resampled on demand by `mdlib`):

    1m (native) → 30m, 1h, 4h
    1d (native) → 1w

Hourly is derived rather than pulled because an hour has no session
semantics — it is just sixty minutes, so there is no convention to get wrong.
Deriving also lets you choose the boundary (e.g. 09:30-aligned) and
guarantees consistency with the minute data the backtest fills on.

**Sample size is the binding constraint for Portfolio B.** A daily strategy
on ES alone over 16 years gives ~100–200 trades — too thin to distinguish
skill from luck. The same strategy across 30 symbols gives 3,000–6,000.

**Therefore: build the backtest engine cross-sectional from day one.**
Retrofitting this later is painful, and testing one symbol at a time leads to
conclusions that do not hold up.

**Portfolio A has the opposite problem.** Intraday generates plenty of trades
but each carries less edge relative to costs, so the cost model must be
realistic rather than optimistic. Assume you pay the spread.

---

## Engine requirements arising from the split

The engine takes a constraint set as a parameter. At minimum:

- `flat_by_close: bool` — force exit at session end, Portfolio A only
- `max_holding_bars: int | None`
- `trailing_drawdown: float | None` — prop rule breach check
- `cost_model` — per-symbol spread and commission

Every backtest emits a **daily return series in a standard format**,
regardless of portfolio or timeframe, so results are directly comparable and
correlatable across both portfolios.

---

## Testing discipline

**Timeframe testing is a robustness check, not a search.** Running five
timeframes and picking the winner selects on noise. A real edge should
survive perturbation — degrade gracefully on neighbouring timeframes rather
than collapse. Pick the timeframe from the logic of the idea *before*
testing, validate there, then check neighbours as a sanity test, and report
the original.

The same applies to every parameter. A strategy that works at exactly 14
periods and fails at 12 and 16 has found an artifact.

**Expected hit rate is roughly 1 in 10.** Most strategies will not work.
That is the process, not failure. The infrastructure exists so that testing
an idea costs an hour rather than a day.

**Portfolio-first, not strategy-first.** Two mediocre uncorrelated strategies
beat one good one. The engine must emit a **daily return series per strategy
in a standard format** from the start, so results can be correlated and
combined. Building strategy-by-strategy produces equity curves that cannot
be compared.

**Prop firm drawdown rules are a first-class metric.** A trailing max
drawdown means the *path* of returns matters more than the total. A strategy
with a great Sharpe and a 15% drawdown fails the account regardless of
eventual profitability. "Would this have breached the rule" must be a
standard backtest output, not an afterthought.

**Correlation across accounts:** five accounts running the same strategies
breach together. Account diversification only helps if the strategies differ.

---

## Tooling

**vectorbt.pro** — $25/mo, $240/yr, or $500 lifetime. Vectorized, fast, suits
cross-sectional testing across 30 symbols and parameter sweeps.

Installation is via a private GitHub repo (members added as collaborators),
so `ssh -T git@github.com` must authenticate first. **Currently failing —
the SSH key generated during provisioning has not been added to GitHub yet.**

Two things to check before subscribing:

1. Supported Python version — the current release adds a Rust backend and may
   not support 3.13. If not, make a separate 3.12 venv for backtesting.
2. Licence is personal use only. Given prop firm accounts are the target,
   confirm intended use qualifies — contact the author.

**Keep strategy logic framework-agnostic:** a function taking bars and
returning signals, with no vectorbt-specific machinery inside. This makes
the live bridge far easier later.

---

## Deployment — Portfolio A (prop accounts)

Prop firms do not expose broker APIs, so NinjaTrader 8 is the execution layer
by necessity. Research stays in Python.

**Architecture — one Windows VM running both NT8 and WSL:**

    NinjaScript          on each bar close, append bar to CSV
        ↓  C:\nt8bridge\
    Python (WSL)         read bars, compute target position, write JSON
        ↓  C:\nt8bridge\
    NinjaScript          read target, compare to current position, trade the gap

Three files in one folder. WSL sees `/mnt/c/nt8bridge/`, NinjaScript sees
`C:\nt8bridge\`. No network, no server, no second data feed at runtime.

**Critical design choice: emit target positions, not entry/exit signals.**

If Python emits "buy" and "sell" events, NinjaScript must track state — did
it act on this signal already? What if it missed one? What if the platform
restarted?

If Python emits *target position* ("be long 2 ES right now"), NinjaScript is
stateless. It reads the target, reads its actual position, trades the
difference. Miss a day, restart the machine, disconnect for an hour — it
self-corrects on the next read.

**NinjaScript holds no strategy logic.** No indicators, no signal rules, no
state. It is a position-reconciliation loop — the same ~200 lines regardless
of how many strategies run above it. This avoids maintaining the same
strategy in two languages, which is the most common failure mode in this
setup.

**Signal file should carry:** timestamp (so NinjaScript can refuse stale
signals), per-symbol target quantity, and a kill flag to halt everything
without touching the VM.

**Known research/live gaps:** backtests use Databento data, live uses the NT8
feed. Bar timestamping conventions differ. Continuous contract roll handling
differs. At daily/weekly frequency these are immaterial; they would matter
intraday.

---

## Build order

1. **Validation script** — row counts per symbol-year, gap detection, price
   sanity checks, degraded-day extraction. Everything downstream depends on
   the data being sound.
2. **`mdlib` reader** — one function returning bars for N symbols at a chosen
   timeframe, cross-sectional by default.
3. **GitHub SSH + vectorbt.pro install.**
4. **Backtest engine** — costs, fills, standard result object including
   drawdown-rule breach checks.
5. **Two or three simple strategies** to exercise the engine.
6. **Portfolio combination and correlation analysis.**
7. **ML** — only after the pipeline underneath is trusted. ML on price data is
   mostly an exercise in overfitting unless train/holdout separation, feature
   lookahead, and realistic costs are already handled correctly.

---

## Deployment — Portfolio B (own capital)

Direct broker API. No NT8, no bridge, no reimplementation — Python computes
signals and places orders in the same process.

This is the cleaner path and is available here only because there is no prop
firm restricting API access. Single codebase from research to execution.

Broker not yet chosen. Requirements: futures execution, a documented REST or
streaming API, and reasonable commissions at swing frequency (fill quality
matters less at multi-day holds than at intraday).

Same design principle as Portfolio A applies: **emit target positions, not
entry/exit events.** A reconciliation loop that compares desired position to
actual position is stateless and self-corrects after any interruption.

---

## Open questions

- Broker selection for Portfolio B. Needs futures execution and a documented
  API. Interactive Brokers, Tradovate (direct, non-prop), and others worth
  comparing on commissions and API quality.
- Whether `bbo-1m` (1-minute sampled bid/ask) falls under the Standard plan's
  1-year L1 allowance or is billed separately. Would give bid-ask spread as a
  per-minute liquidity feature — more relevant to Portfolio A, where cost
  modelling matters most.
- Whether to pull `statistics` (open interest, settlement) and `definition`
  (tick size, multiplier, expiry) schemas. Both small. Open interest is
  especially relevant to Portfolio B — it separates new positioning from
  position closing.
- vectorbt.pro Python version support and licence terms.

**Resolved:** FundedNext does not permit overnight holds. This drove the
two-portfolio split.
