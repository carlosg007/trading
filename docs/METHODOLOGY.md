# Methodology

Drafted 2026-08-09. Living document.

---

## Purpose of backtesting

Backtesting is the primary tool for reducing uncertainty before capital is at
risk. Testing across different market conditions establishes realistic
expectations for:

- Win rates
- Average gains and losses
- Drawdown characteristics
- Risk-adjusted returns

It also surfaces risks that are not visible at design time:

- Market condition dependencies
- Position sizing impacts
- Correlation breakdowns
- Implementation challenges

The end goal is a mental model of the strategy — its limitations, its risk
management requirements, and what performance it can reasonably be expected
to produce.

**Building conviction.** Durable conviction comes from the intersection of
research and lived experience: understanding the strategy at a fundamental
level, testing it across multiple market conditions, experiencing real market
behaviour, and continuously updating the research as a result.

---

## System principles

The system is systematic, meaning:

- **Rule-based and programmable** — every decision expressible in code
- **Consistent and repeatable** — the same inputs always produce the same
  outputs
- **Risk management is quantifiable and automated** — not discretionary

---

## Tooling and outputs

**Engine:** VectorBT PRO for strategy and portfolio construction.

**Storage:** historical data is stored as Parquet. **Only results are written
as CSV.**

**Backtests run across multiple assets simultaneously**, not one symbol at a
time, and write results to CSV for logging and analysis.

**Every backtest produces graphs** for visual analysis alongside the numeric
results.

**Two run modes for every strategy:**

1. Baseline — no machine learning
2. ML-enabled

The baseline is the control. An ML variant that does not beat its own baseline
has not earned its complexity.

**Separate script:** market regime classification per year, output to CSV.
(See `scripts/classify_regime.py`.)

---

## Data quality

Sources: Databento (research) and NinjaTrader 8 (walk-forward validation).

Data must be clean and validated **before** any backtest runs.

### Checks

- Timestamp and timezone handling
- Duplicate detection
- Gaps in the time series
- Consistency across different timeframes
- Unusual price or volume movements
- Cross-source comparison where possible

### Practices

**Preserve raw data.** Keep the original vendor files untouched. This is what
`/mnt/backtest/raw/` exists for — it is write-once and never edited, so the
lake can be rebuilt after a parser bug without re-downloading.

**Cleaning is a script, not a manual step.** Every cleaning operation must be
rerunnable from raw. Cleaning scripts are version controlled.

**Validate after cleaning.** Check values are realistic, verify cleaning has
not introduced new problems, and compare summary statistics before and after.

**Document the cleaning process.**

### Known issue: degraded sessions

Databento flags some sessions as reduced quality. Observed clustering on
high-volatility days — 2020-02-27 and 2020-02-28 (COVID crash), 2024-09-18
(FOMC 50bp cut), quarter-end rolls.

This is expected: exchange feeds get stressed exactly when it matters most.
The practical risk is that a strategy looks robust through a crisis because
the worst bars are missing.

The full list belongs in `reference/futures/degraded_days.csv`, sourced from
Databento's dataset condition endpoint rather than scraped from log warnings
(warnings truncate after three dates).

---

## Avoiding lookahead bias

Signals must only ever use information that was actually available at the time.

### Intraday

- Wait for the current bar to **close** before generating a signal
- Execute on the **next bar's open**
- Example: on 5-minute bars, a crossover occurring during 10:00–10:05 is
  acted on at the open of the 10:05–10:10 bar
- Never calculate indicators from an incomplete bar

### Daily

- Wait for the daily close before calculating indicators
- Place orders for the next session's open
- Account for overnight gaps in strategy design
- Consider after-hours impact on the opening price

### General

- Indicators use completed bars only
- Build in realistic execution delays
- Account for tick size, liquidity, and spread when planning entries
- Test with realistic fill assumptions, not ideal prices

---

## Avoiding survivorship bias

**Note:** most standard survivorship-bias guidance is written for equities —
delisted companies, index reconstitution, market cap screens. Trading a fixed
universe of 30 continuous futures contracts, that specific form does not apply.

The futures-relevant version:

- **Contract roll handling.** The continuous series is a construction. Roll
  rules affect results, and choosing a roll rule after seeing results is a
  form of selection bias.
- **Symbol selection.** Choosing which of the 30 symbols to trade based on
  which performed well in the backtest is survivorship bias by another name.
  Decide the universe by liquidity and diversification criteria first.
- **Period selection.** Cherry-picking test periods, or excluding a bad year
  as "anomalous," produces the same distortion.
- **Costs and slippage** must be included in every test.

---

## Universe selection

Futures. Filtering criteria for the tradable universe:

- Liquidity thresholds — **tight bid-ask spread is a requirement, not a
  preference**
- Volume requirements
- Volatility minima and maxima
- Price ranges
- Asset class constraints for diversification

Also test with a **calendar filter** — excluding roll days, holidays,
half-sessions, and known degraded dates.

---

## Strategy components

### Entry and exit logic

The "when" and "why" of each trade.

**Entry conditions** may be based on technical indicators, statistical
signals, fundamental data, market microstructure signals, or alternative data.

**Exit conditions** typically include profit targets, time-based exits, signal
reversal, technical level breaches, or position size adjustments.

*Signal generation* is the process of converting raw data inputs into
actionable trading decisions through this entry and exit logic.

### Initial strategy families

- **Intraday trend following** — capitalize on momentum in price movement
- **Intraday mean reversion** — assume prices revert to a mean

---

## Risk management framework

### Position sizing

- Percentage of capital
- Volatility-based sizing
- Equal weight allocation
- Kelly criterion — noted as flawed in practice; treat as a ceiling rather
  than a target

### Risk controls

- Maximum position sizes
- Portfolio-level exposure limits
- Sector and asset class constraints
- Correlation management
- Drawdown controls

**Prop firm constraint:** trailing max drawdown means the *path* of returns
matters more than the total. A strategy with a strong Sharpe and a 15%
drawdown fails the account regardless of eventual profitability. "Would this
have breached the rule" must be a standard backtest output, not an
afterthought.

---

## Market microstructure

Understanding microstructure underpins:

- Developing realistic strategies
- Managing execution costs
- Understanding price formation
- Risk management
- Market impact analysis
