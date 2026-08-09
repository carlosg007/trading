# Portfolio construction

Consolidated from course notes 10.1, 10.1b, 10.1c, 10.2. Drafted 2026-08-09.

(10.1b and 10.1c were near-duplicates; merged, keeping the multi-asset
scanning material that only appeared in 10.1c.)

---

## The core claim

The edge is in the combination, not the individual strategy. Several
"good enough" uncorrelated strategies beat one highly optimized strategy.

**The maths.** For N strategies of equal volatility with average pairwise
correlation ρ, portfolio volatility scales as:

    σ_portfolio = σ × √( (1 + (N−1)ρ) / N )

Worked through:

| Correlation | Strategies needed | Result |
|---|---|---|
| 0.60 | 3–4 | ~15% risk reduction, then it stops helping |
| 0.10 | 7–8 | Risk roughly halves — Sharpe doubles |
| 0.00 | 15–20 | Return-to-risk improves up to ~5× |

At ρ = 0.60, adding a thousand strategies barely moves the number. At ρ = 0,
each addition still helps. **Correlation matters more than count.**

*Caveat:* this assumes each strategy genuinely has positive expectancy, and
that measured correlations hold going forward. Both assumptions fail in the
places that matter most — see the failure modes below.

---

## Correlation clusters

Within a cluster, contracts move together. Two picks from one cluster give
almost no diversification.

| Cluster | Contracts | Internal correlation |
|---|---|---|
| Equity indices | ES, NQ, RTY, YM | 0.85–0.95 — nearly the same trade |
| Rates | ZT, ZF, ZN, ZB | 0.80–0.95 |
| Energy (petroleum) | CL, RB, HO | 0.75–0.90 |
| Natural gas | NG | Independent of everything |
| Precious metals | GC, SI | 0.70–0.85 |
| Grains | ZC, ZW, ZS | 0.50–0.70, weather-driven |
| Softs | KC, SB, CT | Low correlation to everything |
| FX majors | 6E, 6B, 6A, 6C | 0.50–0.80 |
| FX safe haven | 6J, 6S | Often negative to risk assets |
| Crypto | BTC, ETH | Tracks NQ in stress, else independent |

### A low-correlation basket

One from each cluster:

| Contract | Cluster | Why |
|---|---|---|
| ES | Equity | Core, most liquid |
| ZN | Rates | Often negative to equities |
| CL | Energy | Supply/demand driven |
| GC | Metals | Real-rate and crisis driven |
| 6J | FX | Risk-off hedge |
| ZC | Grains | Weather-driven, no financial link |
| NG | Energy | Uncorrelated to all of the above |

Liquidity ranking, for slippage: **ES > ZN > CL > GC > 6E > NG > ZC**

Micros exist for ES, NQ, RTY, YM, CL, GC. Rates, grains, and FX have none —
position sizing there is lumpy if capital is limited.

---

## Strategy diversification beats asset diversification

**This is the part most people get wrong.**

Trend following on ES and trend following on CL are more correlated than they
look. Both lose in choppy markets, at the same time, for the same reason.

Trend following and mean reversion on the *same* asset are genuinely
diversifying — one profits precisely when the other struggles.

| Strategy type | Wins when | Loses when |
|---|---|---|
| Trend following | Sustained directional moves | Chop, range |
| Mean reversion | Range-bound, high noise | Strong trends |
| Breakout | Volatility expansion | False breaks in low vol |
| Volatility / gamma | Vol regime shifts | Stable vol |
| Carry / roll yield | Calm, stable term structure | Regime breaks |
| Seasonality | Recurring calendar patterns | Structural change |

**Combine opposite logic, not more symbols.**

### Practical structures

**Small (1–2 strategies)** — trend on ES plus mean reversion on ES. Same
instrument, opposite logic. The cheapest real diversification available.

**Medium (3–4)** — trend on ES, mean reversion on ES or ZN, breakout on CL,
seasonality on ZC or NG.

**Larger (5–7)** — add GC (crisis hedge) and 6J (risk-off). Both tend to work
when everything else fails.

Start with 2–3 well-understood strategies and add gradually. Benefits appear
early; hundreds of strategies are not required.

---

## Multi-asset scanning

Run one strategy across a basket rather than one symbol, taking whichever
signals qualify.

**Scan across clusters, never within.** Scanning ES, NQ, RTY, YM and taking
the first signal is picking randomly among near-identical trades — no risk
reduction. Scanning ES, ZN, CL, GC, ZC, NG gives genuine diversification.

### Selection rule

| Rule | When to use |
|---|---|
| **Best available** — rank qualifying signals by strength, take the top | When signal quality is definable (trend strength, distance from level) |
| **All qualifying** — take every signal at reduced size | Simpler, usually fine, spreads risk naturally |
| ~~First available~~ | Avoid — order-dependent and arbitrary, hard to backtest honestly |

### The hidden benefit: sample size

This is the biggest reason to scan, and it has nothing to do with risk.

| Setup | Signals in 5 years |
|---|---|
| 1 asset at 2/week | ~500 |
| 7 assets at 2/week | ~3,500 |

500 trades is barely enough to fit a simple ML filter. 3,500 is comfortable.

### Train one model, not seven

Pool all assets into a single training set rather than fitting per symbol.

- Add asset identity as a feature so per-asset behaviour can still be learned
- Normalize features to be comparable across instruments — ATR-relative or
  percentage-based, **never raw prices or raw dollar values**

This is cross-sectional pooling, the standard fix for the sample-size problem.

*Caution:* pooling assumes the logic means the same thing across assets. If a
signal means something different on NG than on ES, pooling averages both away.
**Check per-asset results after training, not just the aggregate.**

---

## Three things that will bite you

**1. Correlation goes to 1 in a crash.** The carefully diversified basket
becomes one trade on the day it matters most. March 2020: equities, oil, gold,
and bonds all sold off together as everyone raised cash. Diversification helps
in normal times and partly abandons you in a crisis.

**2. Same strategy on different assets is not diversification.** Five accounts
running identical logic on ES are one position at five times the size.
Correlation between them is 1.0 by construction.

**3. Correlation is not stable.** Bonds were negatively correlated to equities
for two decades, then went positive in 2022. Anything measured over one period
may reverse.

---

## Verifying correlation

Do not trust any published table. Measure on your own data.

```python
returns = prices.pct_change()
returns.corr()                # static
returns.rolling(60).corr()    # rolling — watch it move
```

| Correlation | Reading |
|---|---|
| < 0.3 | Genuine diversification |
| 0.3–0.6 | Partial |
| > 0.7 | Effectively the same trade |

**Measure strategy-return correlation, not just asset correlation.** Two
strategies on uncorrelated assets can still produce correlated P&L if they
share the same logic.

---

## Weighting methods

### Simple

**Equal weighting** — same allocation to each strategy.

Simple, minimal maintenance, automatic diversification, no overconcentration.
Ignores risk differences and correlations.

Worth noting: academic research repeatedly finds equal-weighted portfolios
outperform more complex allocations out of sample. Treat it as the benchmark
every other method must beat.

**Performance-based weighting** — allocate in proportion to historical Sharpe,
Calmar, or returns.

Rewards proven strategies, but concentrates into whatever is currently hot,
underallocates to newer strategies, and inherits the backtest period's biases.

**Fixed position sizing** — a set dollar amount per position.

Simple, controls absolute risk. Does not scale with account size or adjust for
volatility differences.

### Intermediate

**Volatility weighting** — allocate inversely proportional to each strategy's
volatility.

    weight_i = (1 / σ_i) / Σ(1 / σ_j)

Introduces risk awareness. Prevents naturally volatile strategies from
dominating. Needs periodic recalculation and still ignores correlation.

**Risk parity** — allocate so each strategy contributes equally to portfolio
risk. Accounts for correlation as well as volatility.

More balanced than volatility weighting, but requires covariance matrix
calculations, may imply leveraging low-volatility strategies, and depends on
correlations that break down under stress.

### Advanced

**Mean-variance optimization** — maximize expected return for a given risk
level using expected returns and the covariance matrix.

Theoretically optimal. In practice, highly sensitive to input estimates,
prone to extreme allocations, and frequently poor out of sample. The estimation
error usually swamps the theoretical benefit.

**Minimum variance** — minimize portfolio volatility, ignoring expected
returns. Avoids needing return estimates, which are the least reliable input.
Still depends on correlation estimates.

### Recommendation

**Start with equal weighting.** Move to volatility weighting once strategy
volatilities differ materially. Treat risk parity and optimization as things
to try *against* the equal-weight benchmark, not as defaults.

Complexity is not obviously rewarded here. Sophisticated methods routinely
fail to beat simple ones out of sample.

---

## Rebalancing

**Calendar** — fixed intervals, monthly or quarterly.
**Threshold** — rebalance when a weight drifts beyond a set band.

Every rebalance incurs commissions and slippage. More frequent rebalancing
tracks target weights better and costs more. The cost must be modelled, not
assumed away.

---

## Implications for this project

### Portfolio A strategy pairing: mean reversion + session breakout

**Decided.** Portfolio A pairs two opposite-skew strategy families rather than
running variants of one.

| | Mean reversion | Session breakout |
|---|---|---|
| Role | Bread and butter — works most days | Insurance — works on the days MR fails |
| Win rate | 60–70% | 30–40% |
| Skew | Negative — small wins, occasional large loss | Positive — small losses, occasional large win |
| Wins when | Range-bound sessions | Trend days, volatility expansion |
| Loses when | Sustained directional moves | Choppy, low-volatility ranges |

**Why not mean reversion alone**, despite it being the more natural intraday
fit:

**1. The skew is wrong for a trailing drawdown.** A trailing limit punishes
exactly one thing — the large loss. Mean reversion's return shape produces
precisely that, occasionally, when a move does not revert. Trend and breakout
shapes are more survivable under the rule: small losses stay inside the daily
limit, and the outsized win breaches nothing.

**2. All mean reversion strategies fail in the same conditions.** They lose on
strong sustained trends. Five funded accounts running five mean reversion
variants are not diversified — they are one bet that today is not a trend day.
They breach together, on the same day, for the same reason. That is failure
mode 2 above, in its worst possible setting.

**Why session breakout rather than classic trend following.** Trend following
depends on holding through sustained multi-day moves, and flat-by-close
truncates exactly that. The intraday-viable version of positive skew is
breakout and momentum on a session timescale — opening range breaks, prior-day
level breaks, volatility expansion. Same family, same skew, but the trade
completes within the session.

**Build order:** mean reversion first (easier to get working intraday), then
session breakout. Do not fill all five accounts with variants of the first
one that works.

**Classic trend following belongs in Portfolio B**, where daily-to-weekly
horizons and overnight holds are permitted.

### Other implications

**The 30-symbol universe is broader than the diversification need.** Most of
the correlation benefit comes from the seven-contract basket. The rest of the
universe earns its place through *sample size for cross-sectional training*,
not through additional diversification.

**Every backtest must emit a daily return series in a standard format.**
Correlation analysis across strategies is impossible otherwise. Already
handled by `backtest/report.py`.
