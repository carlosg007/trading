#!/usr/bin/env python3
"""
report.py - Standard backtest analysis report.

Location:  ~/src/trading/backtest/report.py

Produces the full set of numbers needed to judge a backtest result, in a
paste-ready block. Every strategy gets analysed the same way, so results are
directly comparable across strategies and across time.

Input contract
--------------
Two files (parquet or csv):

  returns.parquet   required
      date        datetime, one row per trading day
      returns     float, daily return as a decimal (0.01 = 1%)

  trades.parquet    optional, enables trade statistics
      entry_time  datetime
      exit_time   datetime
      symbol      str
      pnl         float, dollars, net of costs
      direction   str, 'long' or 'short'   (optional)

Usage
-----
    python report.py --returns results/strat1_returns.parquet \
                     --trades  results/strat1_trades.parquet \
                     --name    "Strat 1: ES trend" \
                     --variants-tested 12 \
                     --costs-included yes \
                     --out results/strat1_report

Writes <out>.txt (paste this) and <out>_yearly.csv.

The --variants-tested flag matters. A Sharpe of 1.5 from the first idea you
tried and a Sharpe of 1.5 selected from 200 sweeps are not the same result.
The report records it so the number is never read without that context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
REGIME_FILE = Path("/mnt/backtest/reference/futures/regimes.parquet")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {p}")
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def prepare_returns(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}

    date_col = next((cols[c] for c in ("date", "ts", "datetime", "time") if c in cols), None)
    ret_col = next((cols[c] for c in ("returns", "return", "ret", "pnl_pct") if c in cols), None)

    if date_col is None:
        sys.exit(f"No date column found. Columns present: {list(df.columns)}")

    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.sort_values(date_col).set_index(date_col)

    if ret_col is not None:
        r = s[ret_col].astype(float)
    else:
        # Fall back to deriving returns from an equity curve
        eq_col = next((cols[c] for c in ("equity", "value", "nav", "balance") if c in cols), None)
        if eq_col is None:
            sys.exit(f"No returns or equity column found. Columns: {list(df.columns)}")
        r = s[eq_col].astype(float).pct_change()

    return r.dropna()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def cagr(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (eq.iloc[-1] ** (1 / years) - 1) * 100


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dd = downside.std(ddof=1)
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def drawdown_stats(returns: pd.Series) -> dict:
    """Max drawdown plus duration - duration is where strategies get abandoned."""
    dd = drawdown_series(returns)
    max_dd = float(dd.min() * 100)

    underwater = dd < -1e-12
    durations, current = [], 0
    for flag in underwater:
        if flag:
            current += 1
        elif current:
            durations.append(current)
            current = 0
    if current:
        durations.append(current)

    return {
        "max_dd_pct": max_dd,
        "longest_dd_days": max(durations) if durations else 0,
        "avg_dd_days": float(np.mean(durations)) if durations else 0.0,
        "n_drawdowns": len(durations),
        "pct_time_underwater": float(underwater.mean() * 100),
    }


def equity_r2(returns: pd.Series) -> float:
    """R^2 of log equity against a straight line. Higher = steadier."""
    eq = equity_curve(returns)
    if len(eq) < 3 or (eq <= 0).any():
        return float("nan")
    y = np.log(eq.values)
    x = np.arange(len(y), dtype=float)
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr ** 2)


def trailing_drawdown_breach(returns: pd.Series, limit_pct: float,
                             starting_equity: float = 100_000.0) -> dict:
    """
    Prop firm trailing drawdown check.

    The limit trails the HIGH WATER MARK, so the path of returns matters more
    than the total. A strategy with a strong Sharpe and a deep drawdown fails
    the account regardless of eventual profitability.
    """
    eq = starting_equity * equity_curve(returns)
    peak = eq.cummax()
    floor = peak * (1 - limit_pct / 100)
    breached = eq < floor

    return {
        "breached": bool(breached.any()),
        "first_breach": str(breached.idxmax().date()) if breached.any() else None,
        "worst_margin_pct": float(((eq - floor) / peak).min() * 100),
    }


# --------------------------------------------------------------------------
# Trade statistics
# --------------------------------------------------------------------------
def trade_stats(trades: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in trades.columns}
    pnl_col = next((cols[c] for c in ("pnl", "profit", "net_pnl", "p&l") if c in cols), None)
    if pnl_col is None:
        return {}

    pnl = trades[pnl_col].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]

    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    return {
        "n_trades": len(pnl),
        "win_rate_pct": float(len(wins) / len(pnl) * 100) if len(pnl) else float("nan"),
        "avg_trade": float(pnl.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "top5_pct_of_gross": float(wins.nlargest(5).sum() / gross_win * 100) if gross_win > 0 else float("nan"),
        "max_consec_losses": max_consecutive(pnl < 0),
    }


def max_consecutive(flags: pd.Series) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


# --------------------------------------------------------------------------
# Breakdowns
# --------------------------------------------------------------------------
def yearly_table(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for year, g in returns.groupby(returns.index.year):
        dd = drawdown_stats(g)
        rows.append({
            "year": int(year),
            "return_pct": round(float(((1 + g).prod() - 1) * 100), 2),
            "sharpe": round(sharpe(g), 2),
            "max_dd_pct": round(dd["max_dd_pct"], 2),
            "n_days": len(g),
            "pct_days_positive": round(float((g > 0).mean() * 100), 1),
        })
    return pd.DataFrame(rows)


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    m = (1 + returns).resample("ME").prod() - 1
    out = pd.DataFrame({
        "year": m.index.year,
        "month": m.index.month,
        "ret": (m.values * 100).round(2),
    })
    return out.pivot(index="year", columns="month", values="ret")


def regime_join(yearly: pd.DataFrame, symbol: str = "ES") -> pd.DataFrame | None:
    """
    Attach market regime labels so 'does this only work in bull markets?'
    is answered directly rather than by inference.
    """
    if not REGIME_FILE.exists():
        return None
    try:
        reg = pd.read_parquet(REGIME_FILE)
    except Exception:
        return None

    reg = reg[reg["symbol"] == symbol][["year", "regime", "ret_pct"]]
    reg = reg.rename(columns={"ret_pct": "market_ret_pct"})
    merged = yearly.merge(reg, on="year", how="left")
    return merged if merged["regime"].notna().any() else None


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def build_report(returns: pd.Series, trades: pd.DataFrame | None,
                 name: str, variants: int | None, costs: str,
                 dd_limit: float, regime_symbol: str) -> str:
    L: list[str] = []
    add = L.append

    add("=" * 66)
    add(f"BACKTEST REPORT — {name}")
    add("=" * 66)
    add("")
    add(f"Period        : {returns.index[0].date()} → {returns.index[-1].date()}")
    add(f"Trading days  : {len(returns):,}")
    add(f"Variants tested: {variants if variants is not None else 'NOT RECORDED'}")
    add(f"Costs included: {costs}")

    if variants is None:
        add("")
        add("  ⚠ Variants tested not recorded. A result selected from many")
        add("    sweeps is far weaker evidence than a first attempt.")
    elif variants > 50:
        add("")
        add(f"  ⚠ {variants} variants tested. With that many attempts, some will")
        add("    look good by chance. Confirm on a parameter surface, not a peak.")

    dd = drawdown_stats(returns)

    add("")
    add("-" * 66)
    add("HEADLINE")
    add("-" * 66)
    add(f"  CAGR                 {cagr(returns):>10.2f} %")
    add(f"  Total return         {((equity_curve(returns).iloc[-1] - 1) * 100):>10.2f} %")
    add(f"  Sharpe               {sharpe(returns):>10.2f}")
    add(f"  Sortino              {sortino(returns):>10.2f}")
    add(f"  Max drawdown         {dd['max_dd_pct']:>10.2f} %")
    add(f"  Ann. volatility      {returns.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100:>10.2f} %")
    add(f"  Equity curve R²      {equity_r2(returns):>10.3f}")

    sr = sharpe(returns)
    if sr > 3:
        add("")
        add("  ⚠ Sharpe above 3.0 — suspicious. Check for lookahead bias,")
        add("    survivorship in symbol selection, or missing costs.")

    add("")
    add("-" * 66)
    add("DRAWDOWN")
    add("-" * 66)
    add(f"  Max drawdown         {dd['max_dd_pct']:>10.2f} %")
    add(f"  Longest drawdown     {dd['longest_dd_days']:>10} days")
    add(f"  Average drawdown     {dd['avg_dd_days']:>10.1f} days")
    add(f"  Distinct drawdowns   {dd['n_drawdowns']:>10}")
    add(f"  Time underwater      {dd['pct_time_underwater']:>10.1f} %")

    add("")
    add("-" * 66)
    add(f"PROP FIRM CHECK (trailing {dd_limit}% from high water mark)")
    add("-" * 66)
    br = trailing_drawdown_breach(returns, dd_limit)
    if br["breached"]:
        add(f"  BREACHED on {br['first_breach']} — account would have failed.")
    else:
        add(f"  No breach. Closest approach: {br['worst_margin_pct']:.2f} % of peak.")

    if trades is not None and len(trades):
        ts = trade_stats(trades)
        if ts:
            add("")
            add("-" * 66)
            add("TRADES")
            add("-" * 66)
            add(f"  Count                {ts['n_trades']:>10,}")
            add(f"  Win rate             {ts['win_rate_pct']:>10.1f} %")
            add(f"  Average trade        {ts['avg_trade']:>10.2f}")
            add(f"  Average win          {ts['avg_win']:>10.2f}")
            add(f"  Average loss         {ts['avg_loss']:>10.2f}")
            add(f"  Profit factor        {ts['profit_factor']:>10.2f}")
            add(f"  Largest win          {ts['largest_win']:>10.2f}")
            add(f"  Largest loss         {ts['largest_loss']:>10.2f}")
            add(f"  Max consec. losses   {ts['max_consec_losses']:>10}")
            add(f"  Top 5 wins as % of gross profit {ts['top5_pct_of_gross']:>7.1f} %")

            if ts["n_trades"] < 100:
                add("")
                add(f"  ⚠ Only {ts['n_trades']} trades. Too few to separate skill from")
                add("    luck. Widen the universe or lengthen the period.")
            if ts["top5_pct_of_gross"] > 50:
                add("")
                add(f"  ⚠ Top 5 trades are {ts['top5_pct_of_gross']:.0f}% of gross profit.")
                add("    Result depends on a handful of outcomes. Check whether")
                add("    they cluster in one period or one symbol.")
            if 0 < ts["avg_trade"] < 50:
                add("")
                add("  ⚠ Average trade under $50 — vulnerable to live slippage.")

    add("")
    add("-" * 66)
    add("BY YEAR")
    add("-" * 66)
    yearly = yearly_table(returns)
    reg = regime_join(yearly, regime_symbol)
    table = reg if reg is not None else yearly
    add(table.to_string(index=False))

    pos_years = int((yearly["return_pct"] > 0).sum())
    add("")
    add(f"  Positive years: {pos_years}/{len(yearly)}")

    if reg is not None:
        add("")
        for regime, g in reg.groupby("regime"):
            add(f"  {regime:<8} years: {len(g):>2}   "
                f"mean return {g['return_pct'].mean():>7.2f} %   "
                f"worst {g['return_pct'].min():>7.2f} %")
        bull = reg[reg["regime"] == "Bull"]["return_pct"]
        other = reg[reg["regime"] != "Bull"]["return_pct"]
        if len(bull) and len(other) and other.mean() < 0 < bull.mean():
            add("")
            add("  ⚠ Profitable in bull years, negative otherwise. This may be")
            add("    long market exposure rather than a strategy edge.")

    # Single-year concentration. Compare the best year's contribution against
    # total growth in log space - summing percentage returns nets to nonsense
    # when good and bad years offset.
    total_growth = float(np.log(equity_curve(returns).iloc[-1]))
    if len(yearly) > 2 and total_growth > 0.01:
        yr_growth = np.log1p(yearly["return_pct"] / 100.0)
        best_i = yr_growth.idxmax()
        share = float(yr_growth.loc[best_i]) / total_growth * 100
        if share > 60:
            add("")
            add(f"  ⚠ {int(yearly.loc[best_i, 'year'])} contributed ~{share:.0f}% of total growth.")
            add("    Check whether the edge exists outside that year.")

    add("")
    add("-" * 66)
    add("MONTHLY RETURNS (%)")
    add("-" * 66)
    add(monthly_table(returns).to_string())

    add("")
    add("=" * 66)
    add("STILL TO CHECK — not computed here")
    add("=" * 66)
    add("  □ Monte Carlo: reshuffle and resample. Drawdown >2x backtest?")
    add("  □ Parameter surface: is this a plateau or a peak?")
    add("  □ Out-of-sample: was the holdout genuinely untouched?")
    add("  □ Stress periods: 2020 COVID, 2022 bear — check degraded days.")
    add("  □ Correlation against strategies already in the portfolio.")
    add("")

    return "\n".join(L)


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Standard backtest analysis report.")
    p.add_argument("--returns", required=True, help="Parquet/CSV with date + returns")
    p.add_argument("--trades", default=None, help="Parquet/CSV with trade records")
    p.add_argument("--name", default="unnamed strategy")
    p.add_argument("--variants-tested", type=int, default=None,
                   help="How many parameter sets or variants were tried to reach this result")
    p.add_argument("--costs-included", default="UNKNOWN",
                   help="yes/no plus what was modelled, e.g. 'yes - $4.50/RT + 1 tick slippage'")
    p.add_argument("--dd-limit", type=float, default=5.0,
                   help="Prop firm trailing drawdown limit, percent. Default 5.")
    p.add_argument("--regime-symbol", default="ES",
                   help="Symbol whose regime labels to join against. Default ES.")
    p.add_argument("--out", default=None, help="Output path prefix (no extension)")
    args = p.parse_args()

    returns = prepare_returns(load_table(args.returns))
    trades = load_table(args.trades) if args.trades else None

    report = build_report(returns, trades, args.name, args.variants_tested,
                          args.costs_included, args.dd_limit, args.regime_symbol)
    print(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".txt").write_text(report)
        yearly_table(returns).to_csv(out.parent / f"{out.name}_yearly.csv", index=False)
        print(f"\nWrote {out.with_suffix('.txt')}")
        print(f"Wrote {out.parent / f'{out.name}_yearly.csv'}")


if __name__ == "__main__":
    main()
