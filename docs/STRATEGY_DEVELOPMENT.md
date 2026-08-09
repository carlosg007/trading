# Strategy development and validation

Best practices distilled from course notes. Drafted 2026-08-09.

Equity-specific material (short locate fees, per-share commissions, delisting
and corporate actions, small-cap statistical setups) has been dropped — this
project trades futures only.

---

## The development pipeline

Each stage is a gate. A strategy that fails one does not advance.

| Stage | Purpose |
|---|---|
| 1. Goal | Define what the strategy is for before writing any code |
| 2. Idea | Develop the trading idea before backtesting it |
| 3. Preliminary testing | Limited data, light optimization, rules still changing |
| 4. Walk-forward | Full history, rolling in-sample/out-of-sample |
| 5. Monte Carlo | Is the result robust or lucky sequencing? |
| 6. Parameter sensitivity | Does it work only at one setting? |
| 7. Stress testing | What happens in a crisis? |
| 8. Incubation | Paper trade live, no money at risk |
| 9. Portfolio fit | Does it diversify, or just add correlated risk? |
| 10. Live | Ongoing maintenance is part of the job |

**Order matters.** Preliminary testing happens on a subset. Do not run the
full-history test until rules and filters are finalised — otherwise the "big
test" is no longer independent.

---

## Preliminary testing

- Use a limited slice of the available data.
- Keep adding and refining rules here, not later.
- Optimization at this stage is acceptable because the results are not being
  treated as validation.

**Include costs from the very first test.** A strategy evaluated without
commissions and slippage is not being evaluated at all — the ranking of
variants changes once costs are applied.

---

## Out-of-sample validation

Split history into a development segment and a validation segment.

| Segment | Share | Use |
|---|---|---|
| In-sample | 60–80% | Develop and optimize |
| Out-of-sample | 20–40% | Validate only |

### The discipline

- **Never look at out-of-sample data during development.** Not once.
- **Never modify a strategy based on out-of-sample results.** The moment you
  do, that data becomes in-sample and you have no validation left.
- Document the validation process.
- Use multiple out-of-sample periods spanning different market conditions.

### Interpreting the gap

Slightly worse out-of-sample performance is **expected and healthy**. In-sample
results benefit from optimization and represent a best case.

| Acceptable | Red flag |
|---|---|
| Slightly lower average trade | Complete failure out-of-sample |
| Modest increase in drawdown | Dramatic reversal of metrics |
| Small decrease in win rate | Loss of the core statistical edge |
| Slightly lower Sharpe | **Better** out-of-sample than in-sample |

Better out-of-sample performance is a warning, not a win — it usually means
data leakage or a methodological error.

### Data leakage

The subtlest failure. It happens unconsciously: a decision made after
glancing at later data, a feature computed across the full series before
splitting, a filter chosen because it fixed a period you had already seen.

---

## Walk-forward optimization

Rolling in-sample/out-of-sample windows rather than a single split. Optimize
on window N, test on window N+1, advance, repeat, then aggregate all the
out-of-sample results.

**The problem it solves:** optimizing across all history at once grants an
advantage that would not have existed in real time.

### Window sizing

| Strategy frequency | In-sample window |
|---|---|
| High frequency | Days to weeks |
| Daily | Months |
| Long term | Years |

For futures, 6–12 months in-sample is a reasonable starting point. Experiment
with lengths and observe how sensitive the results are to that choice — if
they swing wildly, that is itself a finding.

### Parameters to set

- **Window size** — length of each optimization cycle
- **Step size** — how far the window advances
- **Training-to-testing ratio**

**Do not change the strategy based on walk-forward results.** Same rule as
out-of-sample, for the same reason.

---

## Monte Carlo simulation

A single equity curve tells you what happened in one sequence. Monte Carlo
tells you the range of what could have happened.

### Methods

**Trade reshuffling** — reorder historical trades randomly. Reveals how
dependent the result is on a specific sequence. Run hundreds or thousands of
iterations.

*Note:* reshuffling preserves final P&L only when returns are additive (fixed
dollar risk per trade). Under compounding or percentage-of-equity sizing, the
order changes the endpoint — which is precisely what makes the test
informative.

**Trade resampling** — draw trades with replacement, so the same trade can
appear multiple times. Produces more extreme scenarios than reshuffling.
Better for stress-testing resilience.

**Permutation analysis** — reshuffle intrabar price changes to create
synthetic series with similar statistical properties but no real patterns.
Computationally expensive, but the most rigorous of the three: it tests
whether the strategy could have found an edge in noise.

### What to look for

| Positive | Warning |
|---|---|
| Tight clustering around the median | Wide dispersion of paths |
| Consistently upward-sloping median | Paths crossing into negative territory |
| Most simulations stay positive | Account blow-ups in any path |
| | Extreme outliers in either direction |

**Drawdowns.** Monte Carlo drawdowns are typically 1.5–2× the backtest
drawdown. That is normal. **3× or more is a red flag.** So are multiple
unrecoverable paths and extended underwater periods.

**Mean return** across simulations should sit within roughly 20% of the
original backtest.

### Caveat for trend-following

Trend strategies make their money from a small number of outsized winners.
Resampling may exclude those winners (understating performance) or select
them repeatedly (overstating it), producing much wider dispersion than for
strategies with consistent trade sizes. Interpret the distribution
accordingly rather than reading the tails literally.

### What Monte Carlo cannot do

It cannot rescue a flawed strategy. It validates a sound one. Using it to
justify an over-optimized strategy is a common misuse.

Its real value is calibrating expectations — knowing that a run of nine
consecutive losses is normal for this strategy, and not a sign it has broken.

---

## Parameter sensitivity

A robust strategy performs acceptably across a *range* of parameter values,
not at one specific setting.

### Why it matters

High sensitivity indicates overfitting, fragility across regime changes, and
elevated risk of live failure — small deviations from the optimum produce
large performance drops.

### Methods

1. **Manual** — vary one parameter at a time from a baseline, record metrics,
   plot the results.
2. **Grid search** — define ranges and step sizes, test all combinations
   systematically, record everything.
3. **Walk-forward** — track whether the optimal parameters stay stable across
   periods. Large parameter shifts between windows are a warning.

### Reading the results

**Heat maps** (two parameters, colour = metric): gradual colour transitions
suggest stability; sharp transitions warn of sensitivity.

**3D surface plots**: look for wide elevated **plateaus**, not sharp peaks.
A tall narrow peak is a strategy that works at exactly one setting.

| Robust | Fragile |
|---|---|
| Similar results at neighbouring values | Isolated optimal point |
| Gradual decline away from the optimum | Cliff-edge performance drops |
| Wide stable region | Many sudden peaks and valleys |

**The goal is not the best parameters — it is robust parameters.** Slightly
lower backtest performance with a wide stable region beats an optimal point
that is highly sensitive.

---

## Stress testing

Regular backtesting measures overall performance. Stress testing examines
behaviour specifically during crisis, high volatility, and unusual conditions.

Significant market stress occurs roughly every two years. Performance during
those periods determines long-term survival more than performance in
favourable conditions.

### Periods to examine

2008 financial crisis, 2020 COVID crash, 2022 bear market, and any
high-volatility episode within the data range.

**Note for this project:** the Databento history begins 2010-06-06, so 2008
and the dot-com bubble are not available. 2020 and 2022 are.

**Also note:** Databento flags some sessions as degraded, and these cluster on
exactly the high-volatility days that matter most for stress testing. Check
`reference/futures/degraded_days.csv` before drawing conclusions about crisis
performance — missing bars can make a strategy look more robust than it was.

### Method

**Basic** — identify stress periods, calculate maximum drawdown within them,
compare stress vs. normal metrics.

**Intermediate** — analyse correlation changes during stress, test position
sizing models under stress, evaluate whether liquidity assumptions still hold.

**Advanced** — multiple simultaneous stress conditions, dynamic risk
adjustment.

### Output

Stress testing sets the risk parameters: maximum leverage, position size
limits, stop levels, concentration limits.

**Set leverage from worst-case scenarios, not optimal conditions.**

---

## Transaction costs

Many strategies that look excellent in testing fail live because costs were
underestimated or ignored.

### Components (futures)

- **Broker commission** — per contract, per side
- **Exchange fees** — trading, clearing, data
- **Regulatory fees** — NFA and similar

### Modelling

Start with a **fixed per-contract cost** covering commission plus exchange
plus regulatory fees. For futures this is simple and accurate — costs scale
linearly with contract count, so the percentage-based and per-share models
used in equities do not apply.

### Practices

- Obtain the actual fee schedule from the broker; do not estimate.
- Include every fee, however small.
- Keep records of actual costs from live trading and compare against
  assumptions.
- Update backtest parameters when fee structures change.
- **Be conservative.** Better to overestimate than to be surprised.
- Document assumptions and apply them consistently across all strategies.

---

## Slippage

The difference between expected and actual execution price.

### Drivers

Volatility (higher → more slippage), liquidity (lower → more), order size
(larger → more), and market structure.

### Models, in increasing sophistication

**1. Fixed** — add a constant amount to every fill. Simple; does not reflect
variability.

**2. Probabilistic** — apply slippage to a percentage of trades only. Allows
variability, but strategies that execute at volatile moments will slip on
essentially every trade, so this can understate costs depending on the
strategy type.

**3. High-low window** — look ahead a short window (e.g. 10 seconds) from the
signal and use the worst price in that window: highest for buys, lowest for
sells. Captures real volatility around the execution point and produces
asymmetric, direction-aware slippage.

*Not applicable to this project as stored* — it requires tick or sub-minute
data, and the lake holds 1-minute bars. The bar-level equivalent is to fill
at the worst price within the execution bar, which is more conservative.

**4. Volume-and-volatility based** — model impact as a function of order size
relative to average daily volume, scaled by realized volatility. Realistic but
only relevant at institutional size. Overkill here.

**Almgren-Chriss** is the formal framework behind level 4, splitting impact
into permanent and temporary components and deriving an optimal trading
trajectory. Institutional; noted for awareness only.

### Time-of-day effects

Open and close carry higher volatility and potentially higher slippage.
Off-hours carry lower liquidity and also potentially higher slippage. Regular
hours are generally the most stable.

### Practices

- **Start conservative.** Overestimating is far safer than underestimating.
- Calibrate against actual execution data once live.
- Document the assumptions used for every backtest.

The goal is not a perfect model — it is a model appropriate to the strategy,
market, and capital involved.

---

## Performance metrics

No single metric is sufficient. Evaluate in combination.

### Returns

**CAGR** — annualized growth rate. Preferred over raw ROI because it makes
strategies with different durations comparable. Should exceed the risk-free
rate to be worth pursuing.

### Risk-adjusted

**Sharpe ratio** — excess return per unit of total volatility.

| Range | Reading |
|---|---|
| < 0.5 | Poor |
| 0.5–1.0 | Acceptable but suboptimal |
| 1.0–2.0 | Good |
| 2.0–3.0 | Excellent |
| > 3.0 | Suspicious — check for overfitting |

*Limitations:* assumes normally distributed returns, penalizes upside
volatility equally with downside, and may not capture tail risk. Taleb's
critique is relevant — a high Sharpe can mask a strategy prone to sudden
blow-ups.

Treat these bands as rough guidance. They shift with timeframe and asset
class; a daily futures strategy and an intraday one are not comparable on the
same scale.

**Sortino ratio** — same idea but only penalizes downside volatility. Better
for strategies with positive skew, and closer to how risk is actually
experienced.

### Drawdown

**Maximum drawdown** — largest peak-to-trough decline.

| Range | Reading |
|---|---|
| < 15% | Generally acceptable |
| 15–25% | Moderate risk |
| 25–50% | High risk |
| > 50% | Dangerous |

Consider **duration** alongside magnitude. Long drawdowns demand resilience
and are where strategies get abandoned mid-recovery. Track both average and
longest drawdown duration.

A recent drawdown at the end of a backtest is not a reason to delay going
live — starting during a drawdown is common and often fine.

### Trade statistics

**Win rate**

| Range | Reading |
|---|---|
| < 40% | Requires a high reward/risk ratio |
| 40–50% | Standard |
| 50–60% | Good accuracy |
| > 60% | Typical of mean reversion |

Neither high nor low is inherently better — they are different profiles.
High win rate gives stability, easier position sizing, and less psychological
strain, usually at the cost of lower returns. Low win rate can produce larger
absolute returns but requires strict adherence during long losing streaks and
depends on catching the big winners.

**Average trade** — total P&L divided by trade count. Futures benchmarks:

| Per trade | Reading |
|---|---|
| > $100 | Excellent |
| $50–100 | Strong |
| < $50 | Vulnerable to live slippage and signal timing issues |

Must exceed transaction costs by a wide margin.

### Supporting

**R²** of the equity curve against a linear fit — higher indicates more
consistent performance and helps detect deterioration.

**Standard deviation** of returns — primary volatility measure and a key
input to position sizing. Calculate on rolling windows to capture changing
risk, and compare against a benchmark.

### Practice

- Evaluate metrics across different market conditions, not just in aggregate.
- Always review yearly and monthly breakdowns alongside the headline number.
- Ensure the trade count is large enough for statistical significance. More
  trades, more reliable metrics.

---

## Portfolio fit

A strategy passing every test above is still not automatically tradeable.

**The question is whether it adds diversification or adds correlated risk.**
A profitable strategy that moves with everything else you run increases
portfolio risk without improving return.

This requires every backtest to emit a **daily return series in a standard
format**, so strategies can be correlated against each other.

Position sizing approach is decided per strategy: percentage of capital,
volatility-based, or equal weight. Kelly is noted as flawed in practice —
treat it as a ceiling, not a target.

---

## Live operation

Going live is not the end of the work.

- **Know how to tell when a strategy is broken** versus experiencing a normal
  drawdown. Monte Carlo results define what "normal" looks like — this is
  their main practical use.
- **Futures require rollover handling.** Positions must move to the new
  contract.
- **Portfolio maintenance is a recurring task**, daily and monthly. Neglect
  leads to losses.

---

## Answers to the development questions

**What should a strategy target?** Uncorrelated and profitable. Correlation
against the existing portfolio is a first-class criterion, not an
afterthought — a profitable strategy that moves with everything else adds
risk without adding return.

**What counts as acceptable incubation performance?** Same bar: uncorrelated
and profitable in paper trading.

**How many strategies for adequate diversification?** Not a fixed number. The
incubator paper-trades many strategies concurrently as they are built.
Strategies that prove profitable and stable graduate to the prop accounts.

**How to determine ranges for optimized variables?** Grid search in vectorbt
across a parameter range.

> **Caution on the last one.** Searching a range to find "the best option" is
> the over-optimization trap described in the parameter sensitivity section.
> The output of a grid search should be a **heat map or surface**, not a
> single winning combination. Pick from the middle of a wide stable plateau,
> not the peak. If the best cell is 40% better than its immediate neighbours,
> that cell is noise.
>
> A practical rule: after finding the optimum, deliberately step one or two
> increments away in every direction and use those parameters instead. If
> performance collapses, the strategy has not passed sensitivity testing.

### Still open

- How many rules should a strategy have? (Fewer is generally safer — each
  rule is a degree of freedom that can fit noise.)
- How much optimization is safe?
- How to judge an idea before testing it? "Uncorrelated and profitable" is
  the target, but neither property is knowable in advance — the practical
  filter is whether there is a plausible economic reason the edge exists.
