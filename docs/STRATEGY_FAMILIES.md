# Strategy families

Consolidated from course notes 6.5 (mean reversion) and 6.6 (trend following),
plus session breakout as the intraday-viable form of positive skew.
Drafted 2026-08-09.

**Portfolio A pairs mean reversion with session breakout** — opposite skew,
opposite failure conditions. **Portfolio B uses classic trend following**,
which needs multi-day holds that flat-by-close would truncate.

The organising principle: combine strategies whose losing conditions differ.
One should profit precisely when the other struggles.

---

## Side by side

| | Mean reversion | Session breakout | Trend following |
|---|---|---|---|
| Portfolio | A | A | B |
| Premise | Extremes revert | Breaks follow through | Moves persist |
| Win rate | 60–70% | 30–40% | 30–40% |
| Skew | Negative | Positive | Positive |
| Trade shape | Small wins, occasional large loss | Small losses, occasional large win | Small losses, occasional large win |
| Holding period | Short, intraday | Within session | Days to weeks |
| Wins when | Range-bound, high noise | Trend days, vol expansion | Sustained directional moves |
| Loses when | Strong trends | Quiet, choppy sessions | Chop, whipsaw |
| Main risk | A move that never reverts | Repeated false breaks | Long drawdown while waiting |

---

## Mean reversion

### Why it works

**Statistical.** Most series distribute around a central tendency. Extreme
values are less probable and unsustainable, and the reverting force grows with
the size of the deviation.

**Market structure.** Supply and demand imbalances self-correct. Institutional
rebalancing creates reverting pressure. Arbitrage pushes prices toward
equilibrium.

**Behavioural.** Investors overreact to news, pushing price away from value;
prices return as sentiment stabilizes. Loss aversion and similar biases cause
overshoot followed by correction.

### Measuring it

**Z-score** — how many standard deviations price sits from its mean. High
absolute values signal overbought or oversold conditions.

Common indicators: RSI, Bollinger Bands, distance from a moving average.

### Position management

**Position sizing, not stops, is the primary risk control.** The course notes
argue that tight stops are counterproductive, because the further price moves
against the position the higher the probability of reversion — so a stop exits
at exactly the wrong moment.

The suggested approach: smaller initial positions with room to scale,
progressive position building as deviation increases, exposure limits based on
historical volatility, and **time-based exits rather than price-based stops**.

> ### ⚠ This conflicts directly with prop firm rules
>
> A trailing maximum drawdown does not care about the probability of eventual
> reversion. If the account breaches the floor, it is closed — the reversion
> that would have arrived tomorrow is irrelevant.
>
> "No stops, scale into the deviation" is the classic route to a single
> catastrophic loss, and prop accounts are the environment least able to
> absorb one. The strategy that survives on personal capital with wide stops
> is not the same strategy on a funded account.
>
> **For Portfolio A (prop accounts), mean reversion needs a hard stop**,
> sized so the worst case is survivable within the daily loss limit. This
> costs edge — the notes are right that stops hurt mean reversion — and that
> cost must be modelled in the backtest, not discovered live.
>
> **For Portfolio B (own capital)**, the wider-stop approach is available, but
> scaling into a losing position still requires a hard maximum exposure.

### Limitations

**Timing is uncertain.** The reversion point is not predictable, and the time
to revert varies widely.

**The mean itself moves.** A level that looks extreme against a historical
average may be the new normal. This is what turns a mean reversion loss into
a large one.

**Regime dependence.** Markets shift between reverting and trending states.
Identifying the regime in advance is difficult. Combining opposite-logic
strategies is often easier than filtering the regime correctly.

### Fit for this project

Good fit for **Portfolio A**. Intraday sessions have historically been more
range-bound than trending, holding periods are short, and flat-by-close is
natural for a strategy whose positions turn over quickly.

---

## Trend following

### Why it works

**Risk premium.** Large payoffs during high-volatility periods, low
correlation to traditional asset classes, strong long-run risk-adjusted
returns.

**Psychological.** Most traders avoid it because of the low win rate, and it
requires conviction through drawdowns. That discomfort is why it has not been
arbitraged away.

**Fundamental.** Economic trends often originate in interest rate policy,
which develops slowly and incrementally, propagating into FX, trade balances,
mortgage rates, carrying charges, and equities.

**Systematic.** Works with simple robust rules, across asset classes, and
captures large moves during crises.

### Methods

**Moving averages**
- SMA — equal weight to all points; common periods 50 and 200; suits smooth
  trends
- EMA — more weight on recent prices, faster response, less lag; popular on
  shorter timeframes
- Dual MA — crossover signals, often 50/200, giving direction and strength

**Breakout systems**
- Price channels from recent highs and lows
- Donchian channels — highest high, lowest low; simple and effective
- ATR-based channels — adapt to volatility, reducing false breakouts

**Momentum**
- ADX — measures trend *strength*, not direction. Above 25 indicates a strong
  trend.

### Characteristics

Low win rate with large winners compensating for frequent small losses, and
positive skew. Traditionally medium to long holding periods, requiring
patience through drawdowns. Reversals are sudden — stairs up, elevator down.

Position sizing based on volatility, systematic exit rules, and
portfolio-level diversification.

### Disadvantages

Frequent small losses, whipsaws in choppy markets, late entries and exits, and
significant psychological demands.

### Common pitfalls

**Over-optimization** — curve fitting, too many parameters, overly complex
rule sets.

**Abandoning the strategy during drawdowns**, overtrading in chop, or
manually overriding systematic rules.

**Insufficient diversification** and inconsistent execution.

### Fit for this project

**Awkward for Portfolio A.** Trend following depends on holding through
sustained moves, and flat-by-close truncates exactly that. An intraday trend
strategy is really a momentum strategy on a session timescale — related, but
not the same edge, and it forgoes the overnight moves that supply much of
futures' total return.

**Natural for Portfolio B.** Daily-to-weekly horizons with no overnight
restriction are trend following's home ground.

**Note for Monte Carlo:** trend strategies make their money from a small
number of outsized winners. Trade resampling may exclude or duplicate those
winners, producing much wider dispersion than for other strategy types.
Interpret the tails accordingly rather than literally.

---

## Session breakout

The intraday-viable form of positive skew. Same family as trend following —
it profits from directional persistence — but the trade completes within a
session, so flat-by-close does not truncate it.

**This is the Portfolio A counterweight to mean reversion.**

### Why it works

The same reasons trend following works, compressed to a session. Directional
moves persist once underway; volatility expansion tends to continue rather
than immediately reverse; and participants who need to transact create
follow-through after a level breaks.

It is also uncomfortable to trade discretionarily — most breakouts fail — which
is part of why the edge survives.

### Common forms

**Opening range breakout** — define a range over the first N minutes of the
session, trade the break of that range. The most studied intraday breakout
structure.

**Prior-day level breaks** — trade breaks of the previous session's high or
low. Levels that many participants watch.

**Volatility expansion** — enter when realized volatility or range expands
beyond a threshold, on the reasoning that expansion tends to persist within
the session.

**ATR-based channels** — adapt the breakout threshold to current volatility,
which reduces false breaks in quiet conditions.

### Characteristics

Low win rate (30–40%), small frequent losses, occasional large wins, positive
skew. Sensitive to the choice of range window and threshold — this is a family
where parameter sensitivity testing matters especially, since it is easy to
find a window that happened to work.

### The main failure mode

**False breaks in low volatility.** A quiet session produces repeated small
breaks that immediately reverse, and the strategy bleeds. A volatility filter —
only trade when the session's early range or realized volatility exceeds some
threshold — is usually necessary rather than optional.

### Fit

Natural for **Portfolio A**. Completes within the session, positive skew suits
the trailing drawdown rule, and it wins on exactly the trend days that hurt
mean reversion.

---

## Building Portfolio A

**The pair is mean reversion + session breakout**, not two variants of one
family. They offset: on a strong trend day, mean reversion is stopping out
while breakout is running. That is the entire point.

Three things to decide before writing code:

**Stop policy on the mean reversion side.** Decide before the first backtest,
not after seeing results. Changing it afterwards is fitting. Prop rules mean
a hard stop is required, and its cost must show up in the numbers.

**Volatility filter on the breakout side.** Also decide up front. Without one,
low-volatility sessions will bleed the strategy.

**Regime filtering, or combination?** Combining opposite-logic strategies is
usually easier than correctly identifying the regime in advance. Running both
unfiltered and letting them offset is the simpler starting point — and it is
testable: compare the combined result against each strategy filtered by regime.

**Build order:** mean reversion first, since it is easier to get working
intraday. Then session breakout. Do not fill all five funded accounts with
variants of whichever works first.
