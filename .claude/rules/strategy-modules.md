---
name: strategy-modules
description: "The strategy module contract (signal_fn, indicators, LOGIC, PARAM_GRID, ml_features), the short-side rules, and the mandatory strategy request template."
paths:
  - "strategies/**"
  - "tests/test_sma_momentum_crossover.py"
  - "tests/test_ma_anchoring_spread_20260820.py"
  - "test_dual_version.py"
---

# Writing a strategy module

**`strategies/`** — Signal logic only: take bars, return signal masks. No
cost handling, no session logic, no data access. `approved_incubator/` stages
strategies under evaluation; see its README for the required `meta.json`.

The loader (`agents/tier3_workers.load_strategy`) accepts either form:

```python
make_signal_fn(**params) -> signal_fn     # preferred when parameterised
signal_fn(bars) -> masks                  # when it takes none
```

`bars` is ONE symbol's DataFrame, oldest to newest. Return boolean Series
aligned to it, in one of the two shapes the engine accepts:

```python
(entries, exits)                                  # long only
(entries, exits, short_entries, short_exits)      # bidirectional
```

Neither is deprecated. A long-only strategy returns two — `ema_crossover_20260821` does,
and that keeps the compatibility path exercised by a real module rather than
only by a test. Anything else (a three-tuple, a bare Series) RAISES: silently
taking the first two masks of a three-tuple is how a strategy's short side
disappears into a plausible long-only equity curve. A module may declare
`TIMEFRAME`, `SYMBOLS`, and `DEFAULT_PARAMS`.

**A short is not a long with the sign flipped, and every layer has to know it.**
The places where a mirrored rule is wrong rather than merely unwritten, all of
them silent:

- The stop sits ABOVE the fill and the target BELOW it; the trailing stop
  ratchets DOWN, tracking the low-water mark since the fill. A short stop
  placed below the fill is breached by the fill bar itself.
- `gross_pnl` is `entry - exit` per contract.
- Slippage is charged on the side the order CROSSED, not on whether it opened
  or closed: buys are long entries and short exits, sells are long exits and
  short entries (`_cost_arrays`).
- The ML filter's training label is signed by side
  (`_label_baseline_trades(..., direction=)`). Labelling shorts with the long
  formula teaches the classifier the edge exactly inverted, and Version B comes
  back smooth and backwards. A bidirectional strategy gets one classifier per
  side, each trained only on its own completed trades.

### Every new strategy implements all four

The loader tolerates a module that declares none of the following — several
pre-existing ones do, and the tolerance is what keeps them loadable. It is not
permission to write another. **Any strategy Claude creates carries all four**,
because each one closes a specific way a result goes wrong silently:

```python
def signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, ...]: ...
def indicators(bars: pd.DataFrame, **params) -> dict[str, pd.Series]: ...
LOGIC = {"concept": ..., "entry": ..., "exit": ...}
PARAM_GRID = {"ema_period": [15, 20, 30], "atr_mult": [2.0, 2.5, 3.0]}
```

| Declaration | Without it |
|---|---|
| `signal_fn` | The engine cannot call the module at all. This is **the** contract — a module taking unpacked arrays is wrong, not merely unconventional. |
| `indicators` | The trade inspector draws no lines, and the only way to see why a trade fired is to recompute the series somewhere else — where it is free to disagree with the signals and draw a crossover a bar from where the entry actually happened. |
| `LOGIC` | The tear sheet's strategy card says "not declared". The alternative is inferring the rules from the trades, which is a guess printed as a fact. |
| `PARAM_GRID` | `--scan` has nothing to sweep, so the parameters are whichever ones got typed first and were never compared against anything. |

Write `indicators` in the same module and from the same column `signal_fn`
reads. A second implementation living in the report would be free to disagree
with this one, with nothing raising.

A module may also declare the search space `backtest/run.py --scan` sweeps. It
lives here because this module is the only place that knows what its parameters
mean and what its signature will accept — `load_strategy` rejects unknown
parameter names, so a stale key raises rather than being quietly ignored:

```python
PARAM_GRID = {"ema_period": [15, 20, 30], "atr_mult": [2.0, 2.5, 3.0]}
```

Keep it coarse and small. Nine combinations over 4,000 daily bars is a search
whose result can be reported honestly; a 400-cell grid over the same bars is a
machine for manufacturing an in-sample Sharpe.

Two further declarations are optional, read by the loader, and used **only by
the tear sheet** — neither can change a signal:

```python
LOGIC = {"concept": "...",                                  # plain English
         "entry": "Go Long when the Fast SMA ({fast_window}) crosses above "
                  "the Slow SMA ({slow_window}).",
         "exit":  "Exit when it crosses back below."}       # {param} slots are
                                                            # filled with the
                                                            # bound params
def indicators(bars, **params) -> dict[str, pd.Series]:     # full-length series
    ...                                                     # drawn over the
                                                            # inspector's candles
```

Write the indicator series the same way the signals are computed, in the same
module. A second implementation in the report would be free to disagree with
this one and draw a crossover a bar away from where the trade fired, with
nothing raising.

A third optional declaration, added 2026-08-18, is **not** cosmetic — it
changes which entries Version B vetoes:

```python
def ml_features(bars, **params) -> pd.DataFrame:   # one row per bar, in order
    ...                                            # the matrix the classifier
                                                   # is fitted on
```

`load_strategy` binds it as `module_info["ml_feature_fn"]` and
`run_dual_version_backtest` hands it to `apply_ml_signal_filter` as `features=`.
**A module that declares none gets `None`, which selects the shared
`causal_features`** — so every strategy written before the hook existed keeps
the Version B it always had, bit for bit. `backtest/promote.py`'s generated
Version B wrapper binds it through the same `bind_ml_features`, because a
promoted Version B fitted on different columns from the Version B whose metrics
justified promoting it is the failure the sharing exists to prevent.

Three things travel with it. **Causality is the module's responsibility** — the
filter's guarantee that it trains only on trades closed before the candidate is
undone by a column that reads the future, and a scaler fitted on the whole
frame leaks the test period's distribution into the training rows without
tripping any shift-based audit. **Shape is checked and failures raise**: a row
count that disagrees with the bars, an empty matrix, or a hook that throws is
refused rather than being aligned or quietly fallen back to the default —
unlike `indicators`, which is wrapped, because a broken chart annotation must
not throw away a completed backtest and a silently swapped model must not
survive one. And **the columns are recorded** on the run as
`metrics["meta"]["ml_features"]`, `None` when Version B did not run at all:
two strategies filtered on different matrices have Version Bs that are not
comparable, and that fact has to travel with the numbers.
`strategies/experimental/sma_momentum_crossover.py` is the first user.

- **Dual-Version Mandate:** every strategy outputs two versions. **Version A** is
  a pure rule-based baseline (e.g. an SMA crossover); **Version B** adds an ML
  filter over the same signals. ML is adopted only if B beats A out-of-sample.
  *Note: LightGBM is not currently in `requirements.txt`.*

---

## The Strategy Request Template

**Every strategy starts from a filled-in copy of this block. It is mandatory.**
No module gets written from a one-line prompt — "try a mean reversion on ES" does
not say over what period, against what costs, at what timeframe, or what would
count as it working, and every one of those gets decided anyway. Decided
silently, after the fact, by whoever is looking at the equity curve. A
specification written before the backtest is the only version of it that cannot
be adjusted to fit the result.

The sections run from what the module must declare through to how it is run and
what would settle it. Sections 2, 3 and 4 map onto the module's `LOGIC` block,
its `PARAM_GRID` and its `signal_fn`; sections 1 and 5 are the run.

```text
### STRATEGY SPECIFICATION & BACKTEST REQUEST

================================================================================
1. STRATEGY METADATA
================================================================================
- Strategy Name:        intraday_vol_mr
- Strategy Archetype:   Mean-Reversion
- Primary Timeframe:    15m
- Target Assets:        Multi: NQ,ES,CL,GC  (or ALL)

================================================================================
2. CORE CONCEPT & HYPOTHESIS (Plain English)
================================================================================
- Concept: <what inefficiency is harvested, and why it persists>

================================================================================
3. INDICATORS & PARAMETER GRID (VectorBT Scan)
================================================================================
- Indicators:            <one line each, with the exact period>
- Default Parameters:    <name = value>
- Parameter Search Grid (`PARAM_GRID`):   <name: [values]>

================================================================================
4. ENTRY & EXIT EXECUTION RULES
================================================================================
- Long Entry:            <condition>
- Short Entry:           <condition, or "Long Only">
- Take Profit:           <condition>
- Stop Loss:             <condition, or "none modelled">
- Session Rules:         <entry window, flatten time, in a named timezone>
- Execution Fill:        Next-bar open with contract-specific slippage & commission

================================================================================
5. BACKTEST EXECUTION CONTROLS
================================================================================
- In-Sample Period:      2015-01-01 to 2022-12-31
- Phase 3 Holdout Check: YES / NO   (must NOT overlap the in-sample period)
- ML Filter Comparison (Version B): YES / NO
- Run Mode:              Multi-Asset Runner (`bt-run`)
```

What each section is defending against, and where it has to be checked against
what the engine can actually do:

- **Section 2** is the part that cannot be recovered later. A strategy with no
  stated reason for existing is indistinguishable from one found by searching
  until something looked good, and the two fail differently in live markets. It
  becomes the module's `LOGIC["concept"]` close to verbatim.
- **Section 3** fixes the size of the search before it runs. `--scan` reports
  `variants_tested`, and a Sharpe read without knowing how many variants
  produced it is not a measurement.
- **Section 4 is where a request most often asks for something the engine does
  not have.** Check every line of it against the engine before writing the
  module, and state the gap in the module's docstring rather than approximating
  it silently:
  - **There are no stop or target ORDERS.** `from_signals` is driven by
    boolean masks — two for a long-only strategy, four for a bidirectional
    one — and fills them at the next bar's open. A stop can only be
    expressed as an exit signal, so it is detected on the bar that breaches it
    and filled one bar later — not at the stop price. Say so on the module, or
    every drawdown figure it produces will be read as something it is not.
  - **A trailing stop cannot be a stateless mask at all.** Its level depends on
    the high since entry, which depends on which bar opened the position. It
    needs a state machine inside the module (see
    `strategies/experimental/intraday_vol_mr.py::_walk`), and that machine must
    start its high-water mark at the FILL bar, not the signal bar.
  - **Session times must be converted through a named zone**, not a fixed UTC
    offset. `--flat-by-close` uses a fixed `session_close_utc`, so it is right
    in one half of the year and an hour off in the other; a module doing its
    own session logic should use `America/New_York` and say that the flag is
    then redundant.
- **Section 5's holdout must not overlap the in-sample period.** This is the
  one line in the template that can invalidate everything above it. Reserving
  the last three years after the run is not reserving them — by then they have
  been seen — and an in-sample window that runs into the holdout has spent it
  before Gate 3 is ever evaluated. Check the two date ranges against each other
  before starting.

---
