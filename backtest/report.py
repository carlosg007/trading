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

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ, so an operator
# opening a fresh terminal never has to `source .env` first. It runs at import
# time, above the imports below, because modules resolve their BT_* variables
# while being imported and loading the file inside main() would be too late for
# those - and would work here, which is the kind of difference nobody notices
# until one runner silently uses the default path. The rules live in ONE
# module: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import argparse
import math
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

    eq_col = next((cols[c] for c in ("equity", "value", "nav", "balance")
                   if c in cols), None)

    if ret_col is not None:
        r = s[ret_col].astype(float)
    elif eq_col is not None:
        r = s[eq_col].astype(float).pct_change()
    else:
        sys.exit(f"No returns or equity column found. Columns: {list(df.columns)}")

    r = r.dropna()

    # Every ratio below annualizes with sqrt(TRADING_DAYS), so a file saved at
    # an intraday frequency has to be collapsed onto daily closes before any of
    # them is computed. Nothing about a 15m return series announces itself -
    # it is the right dtype with a sane-looking index - so the check is made
    # here, once, rather than trusted to whoever produced the parquet.
    if is_intraday(r.index):
        equity = (s[eq_col].astype(float) if eq_col is not None
                  else equity_curve(r))
        r = daily_returns(equity)
        print(f"  note: input is intraday ({len(s)} rows over "
              f"{r.index.normalize().nunique()} sessions) - metrics are "
              f"computed on daily closes.", file=sys.stderr)

    return r


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
    """
    Sortino ratio on the institutional denominator.

    Downside deviation is sqrt(sum(min(0, r)^2) / N_total) - the sum of squared
    shortfalls divided by the count of ALL periods, not just the losing ones.

    The denominator convention is not cosmetic. Dividing by the count of losing
    days only takes the dispersion of a strategy's bad days and ignores how
    rare they were, so a strategy that trades 25 times in 1,558 sessions is
    scored on the handful of days it lost and the ~98% of flat days vanish. On
    a real ES/NQ run that inverted the ratio - Sortino 0.28 against a Sharpe of
    0.53 - which reads as a strategy with unusually ugly downside when the
    truth is that it is mostly flat. Over all periods, rarity counts in the
    strategy's favour, which is the property the ratio is supposed to have and
    what Vectorbt Pro and the standard references compute.
    """
    if returns is None or len(returns) == 0:
        return float("nan")
    excess = returns - rf / TRADING_DAYS
    shortfall = excess.clip(upper=0.0)
    dd = float(np.sqrt((shortfall ** 2).sum() / len(excess)))
    # No losing period at all: undefined, not infinite. An inf here would rank
    # a two-trade sample above every real strategy in a sweep.
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


# --------------------------------------------------------------------------
# Daily standardization
#
# Every ratio in this module annualizes with sqrt(TRADING_DAYS), so every ratio
# in this module requires a DAILY return series. Handing `sharpe` a 15m return
# series neither raises nor looks wrong - it silently reports a number scaled
# by the wrong root, because sqrt(252) does not annualize 6,552 bars a year.
# These helpers put an equity curve onto daily closes first, so the
# annualization factor matches the sampling frequency.
#
# "Daily" here means one point per SESSION DATE PRESENT IN THE DATA, carrying
# that session's LAST observation. It deliberately does NOT mean
# `.resample("1D").last().ffill()`. A calendar resample manufactures rows for
# weekends and exchange holidays - days the market never traded - and each one
# carries a return of exactly 0.0. Those zeros pull the mean and the standard
# deviation toward a 365-day year while the sqrt(252) factor stays put, so the
# ratio moves for a reason that has nothing to do with the strategy. Measured
# on a 20-session synthetic run, padding weekends that way turned a Sharpe of
# -15.08 into -11.26. Grouping on the dates actually present also skips
# holidays for free, which a `B`-frequency resample would still invent.
# --------------------------------------------------------------------------
def is_intraday(index: pd.DatetimeIndex) -> bool:
    """True when the index carries more than one observation on some date."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return False
    return len(index) > index.normalize().nunique()


def to_daily_equity(equity: pd.Series) -> pd.Series:
    """
    Collapse an equity / portfolio-value curve onto daily closes: the last
    value observed in each session, indexed on that session's date.

    Idempotent. A curve already carrying one point per session comes back with
    its index normalized to midnight and nothing else changed, so this is safe
    to apply to a series that is already daily rather than something a caller
    has to know whether to skip.
    """
    if equity is None or len(equity) == 0:
        return equity
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise TypeError("to_daily_equity needs a DatetimeIndex, got "
                        f"{type(equity.index).__name__}.")
    daily = equity.groupby(equity.index.normalize()).last()
    daily.index.name = equity.index.name
    return daily


def daily_returns(equity: pd.Series,
                  initial_capital: float | None = None) -> pd.Series:
    """
    Daily simple returns off an equity curve, standardized onto daily closes.

    `initial_capital` seeds the curve one step before its first session, so
    that session's return is a real measurement instead of a dropped NaN.
    Without a seed `pct_change` has no prior close to compare the opening
    session against, and that session's P&L is spent establishing the base of
    the series rather than being scored in it. Usually a no-op - most
    strategies need a warm-up before their first exit - but it is not a no-op
    for a strategy that closes a trade on day one, and the difference is
    invisible in the output.
    """
    daily = to_daily_equity(equity)
    if daily is None or len(daily) == 0:
        return pd.Series(dtype=float)
    if initial_capital is not None:
        step = (daily.index[1] - daily.index[0] if len(daily) > 1
                else pd.Timedelta(days=1))
        seed = pd.Series([float(initial_capital)], index=[daily.index[0] - step])
        daily = pd.concat([seed, daily])
    return daily.pct_change().dropna()


def annualized_return_pct(daily_equity: pd.Series,
                          initial_capital: float) -> float:
    """
    CAGR from a daily equity curve, in percent.

    Years are counted in SESSIONS (len / TRADING_DAYS) rather than in calendar
    days, because the curve is indexed on sessions and a calendar denominator
    would charge the strategy for the weekends its index does not contain.
    NaN rather than a complex number when equity reached or passed zero.
    """
    if daily_equity is None or len(daily_equity) < 2:
        return float("nan")
    final = float(daily_equity.iloc[-1])
    if final <= 0 or initial_capital <= 0:
        return float("nan")
    years = len(daily_equity) / TRADING_DAYS
    if years <= 0:
        return float("nan")
    return float(((final / initial_capital) ** (1.0 / years) - 1.0) * 100.0)


def calmar(annualized_pct: float, max_drawdown_pct: float) -> float:
    """
    CAGR over the absolute max drawdown, both already in percent.

    NaN when there was no drawdown: a strategy that never drew down has an
    undefined Calmar, not an infinite one, and an inf here reads as a headline
    result rather than as the degenerate sample it is. Drawdown is taken on
    magnitude because the engine signs it negative.
    """
    if math.isnan(annualized_pct) or math.isnan(max_drawdown_pct):
        return float("nan")
    if max_drawdown_pct == 0:
        return float("nan")
    return float(annualized_pct / abs(max_drawdown_pct))


def daily_metrics(equity: pd.Series,
                  initial_capital: float,
                  rf: float = 0.0) -> dict:
    """
    The three risk-adjusted ratios, all off ONE daily equity series.

    Single entry point so Sharpe, Sortino and Calmar cannot end up sampled at
    different frequencies - which is exactly how a scorecard ends up printing
    a daily Sharpe next to an intraday Sortino with nothing raising. `basis`
    travels with the numbers so a reader can see which frequency and which
    risk-free rate produced them.
    """
    daily_eq = to_daily_equity(equity)
    rets = daily_returns(equity, initial_capital)
    dd = (daily_eq / daily_eq.cummax() - 1.0) if daily_eq is not None and len(daily_eq) else None
    max_dd_pct = float(dd.min() * 100) if dd is not None and len(dd) else float("nan")
    ann_pct = annualized_return_pct(daily_eq, initial_capital)
    return {
        "sharpe": sharpe(rets, rf) if len(rets) else float("nan"),
        "sortino": sortino(rets, rf) if len(rets) else float("nan"),
        "calmar": calmar(ann_pct, max_dd_pct),
        "annualized_return_pct": ann_pct,
        "max_dd_pct": max_dd_pct,
        "basis": {
            "frequency": "daily_close",
            "annualization_factor": TRADING_DAYS,
            "risk_free_rate": float(rf),
            "n_days": int(len(rets)),
        },
    }


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
DOW_COLUMNS = ["day", "weekday", "trades", "net_pnl", "gross_win", "gross_loss",
               "win_rate", "avg_pnl", "profit_factor", "pct_of_trades",
               "pct_of_net_pnl"]


def day_of_week_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """
    P&L, win rate and trade count per weekday, attributed by ENTRY session.

    Two decisions this makes, both of which change the answer:

    **Attribution is by the ENTRY, not the exit.** A trade is the consequence
    of the decision to open it, and that decision was made on the entry
    session. Attributing an overnight position to its exit day would credit
    Tuesday for a Monday setup, which is precisely backwards for the use this
    table exists for - `exclude_days` suppresses ENTRIES, so the table has to
    be keyed on the thing the parameter acts on or pruning the worst row will
    not remove the trades that made it worst.

    **The day is the CME SESSION date, not the UTC date.** See
    `backtest.event_calendar.session_date`: the session opens 18:00 ET the previous
    evening, so on intraday bars a UTC-keyed table splits every Globex evening
    off onto the wrong weekday.

    Rows are Monday through Sunday, and a weekday with no trades is present
    with zeros rather than absent. An absent row reads as "no data" when what
    it means is "this strategy never traded on a Friday", which is a finding.
    Saturday and Sunday appear only when something actually traded then - on
    session dates that should be never, and a non-zero row there is a bug
    worth seeing rather than hiding.

    `win_rate` is a fraction (0.55), matching `summarize_result`, not the
    percentage `trade_stats` reports. `profit_factor` is NaN, never inf, where
    a day has no losses - an infinity in a table read by eye is a division by
    zero wearing a disguise.
    """
    from .event_calendar import WEEKDAY_NAMES, session_date

    empty = pd.DataFrame(columns=DOW_COLUMNS)
    if trades is None or len(trades) == 0:
        return empty
    cols = {c.lower(): c for c in trades.columns}
    pnl_col = next((cols[c] for c in ("pnl", "net_pnl", "profit") if c in cols),
                   None)
    entry_col = next((cols[c] for c in ("entry_time", "entry_ts", "entry")
                      if c in cols), None)
    if pnl_col is None or entry_col is None:
        return empty

    df = pd.DataFrame({
        "pnl": pd.to_numeric(trades[pnl_col], errors="coerce"),
        "dow": session_date(trades[entry_col]).dayofweek,
    }).dropna(subset=["pnl"])
    if df.empty:
        return empty

    total_trades = len(df)
    total_pnl = float(df["pnl"].sum())

    present = set(df["dow"].unique())
    rows = []
    for d in range(7):
        if d > 4 and d not in present:
            continue                       # no weekend row unless it traded
        g = df[df["dow"] == d]
        pnl = g["pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        rows.append({
            "day": WEEKDAY_NAMES[d],
            "weekday": d,
            "trades": int(len(g)),
            "net_pnl": float(pnl.sum()),
            "gross_win": gross_win,
            "gross_loss": gross_loss,
            "win_rate": float(len(wins) / len(g)) if len(g) else float("nan"),
            "avg_pnl": float(pnl.mean()) if len(g) else float("nan"),
            "profit_factor": (gross_win / gross_loss if gross_loss > 0
                              else float("nan")),
            "pct_of_trades": (100.0 * len(g) / total_trades
                              if total_trades else float("nan")),
            # Share of the run's net P&L, against the ABSOLUTE total so the
            # sign belongs to the day rather than to the run. Dividing by a
            # signed total inverts every row of a losing backtest - the worst
            # day comes out most positive - which is exactly backwards for a
            # table whose job is to point at the day worth excluding.
            "pct_of_net_pnl": (100.0 * float(pnl.sum()) / abs(total_pnl)
                               if total_pnl else float("nan")),
        })
    return pd.DataFrame(rows, columns=DOW_COLUMNS)


def losing_weekdays(breakdown: pd.DataFrame,
                    min_trades: int = 20) -> list[int]:
    """
    The weekday integers a breakdown says lost money, for `exclude_days`.

    `min_trades` is a floor, not a formality. Over a 5-year daily backtest a
    weekday holds a fifth of the sample, and pruning a day on eight trades is
    fitting the calendar to noise - which is the specific way a day-of-week
    filter manufactures an in-sample Sharpe. Days below the floor are NOT
    returned however badly they scored, and the caller is told nothing about
    them here; `format_day_of_week` marks them on the table instead.

    This is a suggestion for a human to act on, never applied automatically.
    Nothing in the pipeline calls it and then feeds the result back into the
    same run: selecting the losing days in-sample and re-scoring on the same
    bars is circular, and the resulting number is not a measurement.
    """
    if breakdown is None or breakdown.empty:
        return []
    bad = breakdown[(breakdown["net_pnl"] < 0)
                    & (breakdown["trades"] >= int(min_trades))]
    return [int(d) for d in bad["weekday"].tolist()]


# The profit-factor bar a weekday has to clear to stay in the strategy. The same
# 1.00 Stage 1 screens a whole CONTRACT on, and the same number Gate 1 binds on,
# so a session cannot survive the day-of-week screen on a profit factor the gate
# would later reject.
DOW_MIN_PROFIT_FACTOR = 1.00


def unprofitable_weekdays(breakdown: pd.DataFrame,
                          min_trades: int = 20,
                          min_profit_factor: float = DOW_MIN_PROFIT_FACTOR
                          ) -> list[dict]:
    """
    EVERY weekday whose profit factor is below the bar - the days to exclude.

    This is the automated half of what `losing_weekdays` offers a human. Stage 1
    calls it to fill `dropped_days` / `exclude_days` in `surviving_assets.json`,
    Stage 2 masks those sessions out before it sweeps, and Stage 3 certifies on
    the same pruned week. Read the caveat in `backtest/baseline.py` first: the
    days are chosen IN-SAMPLE, on the same bars Stage 2 then optimises over, so
    this is one more layer of selection and not a free improvement. It is
    recorded as such at every handoff rather than disappearing into the
    parameters.

    The rule, in full, because "unprofitable" has more than one edge case:

      * **Profit factor, not net P&L.** PF < 1.00 means the session gave back
        more than it took in after costs, which is the same statement Stage 1
        screens a whole contract on. Ranking on dollars instead would drop a
        thin day that lost a little ahead of a heavily-traded day that lost
        more per trade.
      * **Only Monday-Friday.** A Saturday row is a bug worth seeing (see
        `day_of_week_breakdown`), not a session to prune.
      * **Only days at or above `min_trades`.** The same floor
        `losing_weekdays` enforces, for the same reason: a losing Tuesday over
        eight trades is noise, and pruning it is how a day-of-week filter
        manufactures an in-sample Sharpe. A day below the floor stays in
        however badly it scored.
      * **An UNDEFINED profit factor is never dropped.** `day_of_week_breakdown`
        returns NaN where a day had no losing trades at all, and `NaN < 1.00` is
        False in numpy as well as in Python - but relying on that would make the
        behaviour an accident of the comparison rather than a decision. A day
        that never lost is the opposite of one to exclude, and a day with zero
        trades has produced no evidence to exclude it on.

    Returns a list of row dicts in WEEKDAY order (Mon first), one per day to
    drop, each carrying `day_name` for the artifact and the metrics that
    condemned it. Zero, one or five days is all the same shape: an empty list
    means every weekday cleared the bar, which is a result, not a missing value.
    """
    from .event_calendar import WEEKDAY_FULL_NAMES

    if breakdown is None or breakdown.empty:
        return []
    out = []
    for _, r in breakdown.sort_values("weekday").iterrows():
        d = int(r["weekday"])
        pf = r["profit_factor"]
        if d > 4 or int(r["trades"]) < int(min_trades) or pd.isna(pf):
            continue
        if float(pf) >= float(min_profit_factor):
            continue
        out.append({
            "weekday": d,
            "day": str(r["day"]),
            "day_name": WEEKDAY_FULL_NAMES[d],
            "trades": int(r["trades"]),
            "net_pnl": float(r["net_pnl"]),
            "win_rate": (None if pd.isna(r["win_rate"])
                         else float(r["win_rate"])),
            "profit_factor": float(pf),
            "pct_of_net_pnl": (None if pd.isna(r["pct_of_net_pnl"])
                               else float(r["pct_of_net_pnl"])),
        })
    return out


def exclude_days_basis(min_trades: int = 20,
                       min_profit_factor: float = DOW_MIN_PROFIT_FACTOR) -> str:
    """The rule `unprofitable_weekdays` applied, in one line, for the record."""
    return (f"every Mon-Fri session with profit factor < "
            f"{float(min_profit_factor):.2f} over at least {int(min_trades)} "
            f"trades, attributed by ENTRY session date")


def format_day_of_week(breakdown: pd.DataFrame,
                       min_trades: int = 20,
                       indent: str = "  ") -> str:
    """The breakdown as a console block, thin rows flagged rather than dropped."""
    if breakdown is None or breakdown.empty:
        return f"{indent}day of week   : no trades to attribute"

    head = (f"{indent}{'day':<5}{'trades':>8}{'net P&L':>14}{'win rate':>10}"
            f"{'avg P&L':>12}{'PF':>7}  share of P&L")
    lines = [head, indent + "-" * (len(head) - len(indent) + 14)]
    for _, r in breakdown.iterrows():
        pf = r["profit_factor"]
        thin = "  (thin sample)" if r["trades"] < min_trades else ""
        lines.append(
            f"{indent}{r['day']:<5}{int(r['trades']):>8,}"
            f"{r['net_pnl']:>14,.0f}"
            f"{(r['win_rate'] * 100 if pd.notna(r['win_rate']) else float('nan')):>9.1f}%"
            f"{r['avg_pnl']:>12,.0f}"
            f"{(f'{pf:.2f}' if pd.notna(pf) else '  n/a'):>7}"
            f"{r['pct_of_net_pnl']:>13.1f}%{thin}")
    return "\n".join(lines)


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


def regime_join(yearly: pd.DataFrame,
                symbol: str = "ES") -> tuple[pd.DataFrame | None, str | None]:
    """
    Attach market regime labels so 'does this only work in bull markets?'
    is answered directly rather than by inference.

    Returns (merged, reason). Exactly one is None: on success the merged frame,
    otherwise a human-readable reason the labels are unavailable.

    The reason is not decoration. Without labels the report still renders a
    complete-looking BY YEAR table, just missing the one section that separates
    a strategy edge from long market exposure. A reader who is not told the
    check was skipped will read its absence as its passing.
    """
    if not REGIME_FILE.exists():
        return None, (f"{REGIME_FILE} does not exist - regime labels skipped. "
                      f"Generate it with: python scripts/classify_regime.py "
                      f"--threshold 10.0")
    try:
        reg = pd.read_parquet(REGIME_FILE)
    except Exception as e:
        return None, f"{REGIME_FILE} could not be read ({e}) - regime labels skipped."

    missing = {"symbol", "year", "regime", "ret_pct"} - set(reg.columns)
    if missing:
        return None, (f"{REGIME_FILE} is missing column(s) {sorted(missing)} - "
                      f"regime labels skipped. Regenerate it.")

    reg = reg[reg["symbol"] == symbol][["year", "regime", "ret_pct"]]
    if reg.empty:
        return None, (f"{REGIME_FILE} has no rows for symbol {symbol!r} - regime "
                      f"labels skipped. Check --regime-symbol.")

    reg = reg.rename(columns={"ret_pct": "market_ret_pct"})
    merged = yearly.merge(reg, on="year", how="left")
    if not merged["regime"].notna().any():
        lo, hi = int(yearly["year"].min()), int(yearly["year"].max())
        return None, (f"No regime labels overlap the backtest years {lo}-{hi} for "
                      f"symbol {symbol!r} - regime labels skipped.")
    return merged, None


# --------------------------------------------------------------------------
# Acceptance gates
# --------------------------------------------------------------------------
# The three gates a strategy clears before it is allowed near a live account.
# They are stated once, here, so Version A and Version B are judged against
# identical numbers - a scorecard where the two columns were scored on
# different bars is worse than no scorecard.
GATE_THRESHOLDS: dict[str, dict[str, float]] = {
    # Gate 1 recalibrated 2026-08-16 for single-strategy account governance.
    #
    # Sharpe is NOT a Gate 1 criterion. It is still computed on daily closes,
    # still printed on every scorecard and tear sheet, and still the metric
    # Gate 3 measures retention on - it is simply no longer a pass/fail
    # condition here. A Sharpe floor rejects a strategy that makes money after
    # costs and never draws down past 12% for having a lumpy return path,
    # which is a complaint about the shape of the equity curve rather than
    # about whether there is an edge. It stays on the Gate 1 table as an
    # INFORMATIONAL row so a reader who remembers a 1.20 threshold can see it
    # was demoted rather than silently dropped.
    #
    # What is left is the pair that a single account actually lives or dies
    # on: positive gross expectancy AFTER costs (profit factor >= 1.00, which
    # is the break-even line, not a comfort margin), and capital preservation
    # (drawdown <= 12%, tighter than the old 15% because the survivable
    # drawdown on a governed account is smaller than the survivable one on a
    # research equity curve).
    #
    # `min_trades` is a FLOOR, not the whole criterion - `_required_trades`
    # scales it by the length of the sample at `min_trades_per_year`, and
    # `max_required_trades` is the CEILING that scaling stops at.
    "gate1": {"min_profit_factor": 1.00, "min_trades": 100,
              "max_required_trades": 200, "min_trades_per_year": 30,
              "max_drawdown_pct": 12.0},
    "gate2": {"min_wfo_efficiency": 0.50, "max_mc_drawdown_pct": 18.0},
    # Retention = holdout metric / in-sample metric. 0.80 is "no more than 20%
    # degradation", expressed as a ratio because that is what gets computed.
    # Measured on Sharpe where the in-sample Sharpe is positive, and on profit
    # factor where it is not - see `audit_acceptance_gates`.
    "gate3": {"min_retention": 0.80},
}

GATE_NAMES = {
    "gate1": "GATE 1 · In-Sample",
    "gate2": "GATE 2 · Robustness",
    "gate3": "GATE 3 · OOS Holdout",
}

PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUATED = "NOT EVALUATED"
# Reported, never scored. A criterion carrying this status is displayed on the
# scorecard and the tear sheet and is skipped by `_roll_up` - it can neither
# pass a gate nor fail one.
INFO = "INFO"


def _numeric(value) -> float:
    """
    Coerce a metric to a float, mapping anything unusable to NaN.

    None, a missing key and a string that is not a number all collapse to NaN,
    which every comparison below treats as *not a pass*. A gate must never
    clear because the number that would have failed it was absent.
    """
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _pick(source, *keys) -> float:
    """First present key from a metrics-shaped dict, as a float. NaN if none."""
    if source is None:
        return float("nan")
    if not isinstance(source, dict):
        return _numeric(source)
    for k in keys:
        if k in source:
            return _numeric(source[k])
    return float("nan")


def _criterion(label: str, value: float, threshold: float | None, direction: str,
               unit: str = "", note: str | None = None,
               fmt: str = "{:.2f}", informational: bool = False) -> dict:
    """
    Score one criterion.

    `direction` is 'min' (value must be >= threshold) or 'max' (value must be
    <= threshold). A NaN value scores NOT EVALUATED rather than FAIL: "we did
    not measure this" and "this failed" are different findings, and collapsing
    them is how an unmeasured gate gets reported as a cleared one. Neither is
    a pass, so the distinction never flatters a result.

    `informational=True` scores INFO and takes no threshold: the number is
    carried onto the scorecard and the tear sheet and can never pass or fail
    the gate. It is how Sharpe stays visible on Gate 1 without being a Gate 1
    condition. Deleting the row instead would leave a reader who remembers a
    Sharpe threshold unable to tell whether it was demoted or merely not
    measured this run - which are opposite readings of the same blank space.

    Drawdowns are compared on magnitude. The engine reports `max_dd_pct` as a
    negative number and `report.drawdown_stats` agrees, but a caller handing in
    a positive 12.0 means the same drawdown - comparing the raw sign would let
    a 40% drawdown clear a 12% limit because -40 <= 12.
    """
    if direction not in ("min", "max"):
        raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")

    if informational:
        return {"label": label, "value": value, "threshold": None,
                "direction": direction, "unit": unit, "status": INFO,
                "note": note, "fmt": fmt}

    if math.isnan(value):
        status = NOT_EVALUATED
    elif direction == "min":
        status = PASS if value >= threshold else FAIL
    else:
        status = PASS if value <= threshold else FAIL

    return {"label": label, "value": value, "threshold": threshold,
            "direction": direction, "unit": unit, "status": status,
            "note": note, "fmt": fmt}


def criterion_text(check: dict) -> tuple[str, str]:
    """
    `(measured, required)` as display strings, e.g. ('11.20%', '<= 12%').

    An informational criterion has no threshold and reports 'informational' in
    place of one, so no reader can mistake a number that is merely on the
    table for a bar the run had to clear.

    The value goes through `_numeric` rather than being used raw because an
    audit is routinely read back out of JSON, and a NaN written to
    `dual_metrics.json` comes back as `null` - JSON has no NaN. This is the one
    place both the scorecard and the HTML report render a criterion, so it is
    the one place that has to survive the round trip.
    """
    fmt = check.get("fmt", "{:.2f}")
    unit = check.get("unit", "")
    value = _numeric(check.get("value"))
    measured = "n/a" if math.isnan(value) else fmt.format(value) + unit
    threshold = _numeric(check.get("threshold"))
    if check.get("status") == INFO or math.isnan(threshold):
        return measured, "informational"
    op = ">=" if check.get("direction") == "min" else "<="
    return measured, f"{op} {threshold:g}{unit}"


def _roll_up(checks: list[dict]) -> str:
    """
    A gate is only PASS when every one of its SCORED criteria passed.

    Informational criteria are skipped entirely - they carry no threshold, so
    there is nothing for them to have passed or failed. A gate made up only of
    informational rows is NOT EVALUATED, because nothing about it was tested.
    """
    scored = [c for c in checks if c["status"] != INFO]
    if not scored:
        return NOT_EVALUATED
    if any(c["status"] == FAIL for c in scored):
        return FAIL
    if any(c["status"] == NOT_EVALUATED for c in scored):
        return NOT_EVALUATED
    return PASS


def _sample_years(metrics: dict) -> float:
    """
    How many years of bars the metrics were measured over. NaN if unknown.

    Counted in SESSIONS (`n_days / TRADING_DAYS`), the same way
    `annualized_return_pct` counts them, so the trade-count bar and the CAGR
    beside it are denominated in the same year. A calendar span would make a
    strategy that only trades one session in five look like it covered more
    ground than it did.

    `metrics_basis.n_days` is the count the ratios were actually sampled on
    and is preferred; `n_days` is the equity curve's own length and is the
    fallback for a metrics dict assembled by hand.
    """
    basis = metrics.get("metrics_basis") if isinstance(metrics, dict) else None
    n_days = _pick(basis, "n_days") if isinstance(basis, dict) else float("nan")
    if math.isnan(n_days) or n_days <= 0:
        n_days = _pick(metrics, "n_days")
    if math.isnan(n_days) or n_days <= 0:
        return float("nan")
    return float(n_days) / TRADING_DAYS


def _required_trades(metrics: dict, thresholds: dict) -> tuple[float, str | None]:
    """
    Gate 1's trade-count bar, scaled to the length of the sample.

    Returns `(required, note)`.

    `min_trades` (100) is a FLOOR that every slice has to clear, however short
    it is - an out-of-sample window with 40 trades in it cannot separate an
    edge from a run of luck no matter how few months it covers. Above roughly
    3.3 years the per-year rate binds instead: a 16-year backtest that
    produced 120 trades is not comfortably clear of a 100-trade floor, it is
    seven trades a year, and every ratio computed from it is noise wearing two
    decimal places.

    `max_required_trades` (200) is a CEILING the scaling stops at, added
    2026-08-17. Unbounded, the per-year rate demanded 503 trades of a
    16.7-year lake run, which is not a significance bar - 200 trades already
    settles whether a sample is large enough to measure - but a selectivity
    bar, and the thing it rejected hardest was a Version B whose ML filter did
    its job and stood most of the baseline's trades down. A filter that
    improves every remaining trade should not fail Gate 1 for taking fewer of
    them.

    Where the sample length is unknown the floor is used alone and the note
    says so, rather than assuming a generous span and quietly lowering the bar.
    The scaled figure is rounded to a whole trade so the number displayed in
    the 'required' column is the number actually compared against.
    """
    floor = float(thresholds["min_trades"])
    ceiling = float(thresholds.get("max_required_trades", math.inf))
    per_year = float(thresholds.get("min_trades_per_year", 0.0))
    years = _sample_years(metrics)
    if math.isnan(years) or per_year <= 0:
        return floor, (f"sample length unknown, so the {floor:,.0f}-trade floor "
                       f"is applied without scaling")
    scaled = int(round(per_year * years))
    required = min(ceiling, max(floor, float(scaled)))
    if scaled <= floor:
        return required, (f"{years:.2f} years at {per_year:,.0f} trades/yr is "
                          f"{scaled:,d}; the {floor:,.0f}-trade floor binds")
    if scaled >= ceiling:
        return required, (f"{years:.2f} years at {per_year:,.0f} trades/yr is "
                          f"{scaled:,d}; the {ceiling:,.0f}-trade cap binds")
    return required, (f"{years:.2f} years at {per_year:,.0f} trades/yr")


def audit_acceptance_gates(metrics: dict,
                           robustness: dict | None = None,
                           holdout: dict | None = None,
                           version: str = "A",
                           name: str | None = None) -> dict:
    """
    Score one version of a strategy against the three acceptance gates.

    Call it once per version - `print_dual_scorecard` puts the two audits side
    by side - because a single audit over merged inputs cannot say which
    version failed.

    Parameters
    ----------
    metrics
        The in-sample metrics dict `agents.tier3_workers.summarize_result`
        returns: `sharpe`, `profit_factor`, `trade_count`, `max_drawdown_pct`,
        and `n_days` / `metrics_basis.n_days` - the last of which sets how many
        trades Gate 1 demands (`_required_trades`). A metrics dict with no day
        count is scored against the unscaled 100-trade floor.
    robustness
        Optional. `{"wfo": <run_walk_forward_analysis result or ratio>,
        "monte_carlo": <run_monte_carlo_simulation result or drawdown pct>}`.
        Omit it and Gate 2 reports NOT EVALUATED - it never reports PASS on
        evidence that was not supplied.
    holdout
        Optional. The metrics dict from the held-back final 3 years, or
        `{"sharpe": x}`, or a bare Sharpe. Gate 3 compares it against the
        in-sample Sharpe in `metrics` - or, where that ratio is undefined,
        against the in-sample profit factor. Pass the full metrics dict rather
        than a bare Sharpe when you want that fallback available.

    Returns
    -------
    dict with `gates` (gate1/gate2/gate3, each with `status` and `checks`),
    an overall `status` (PASS / FAIL / NOT EVALUATED), `passed` (True only when
    all three gates passed), and the thresholds used.

    An incomplete audit is NOT a passing audit. `passed` is True only when
    every gate cleared on real numbers, so a caller that promotes on `passed`
    cannot promote a strategy whose robustness was never run.
    """
    t1, t2, t3 = (GATE_THRESHOLDS["gate1"], GATE_THRESHOLDS["gate2"],
                  GATE_THRESHOLDS["gate3"])

    is_sharpe = _pick(metrics, "sharpe", "sharpe_ratio")
    is_pf = _pick(metrics, "profit_factor")
    min_trades, trades_note = _required_trades(metrics or {}, t1)
    gate1 = [
        # Reported, not gated. See GATE_THRESHOLDS.
        _criterion("Sharpe (not gated)", is_sharpe, None, "min",
                   informational=True),
        _criterion("Profit factor", is_pf, t1["min_profit_factor"], "min"),
        _criterion("Trades", _pick(metrics, "trade_count", "n_trades"),
                   min_trades, "min", fmt="{:,.0f}", note=trades_note),
        _criterion("Max drawdown", abs(_pick(metrics, "max_drawdown_pct", "max_dd_pct")),
                   t1["max_drawdown_pct"], "max", unit="%"),
    ]

    rb = robustness or {}
    wfo = _pick(rb.get("wfo"), "efficiency_ratio", "efficiency", "wfo_efficiency")
    if math.isnan(wfo):
        wfo = _pick(rb, "wfo_efficiency", "efficiency_ratio")
    mc = _pick(rb.get("monte_carlo"), "max_drawdown_pct_at_confidence",
               "mc_max_drawdown_pct", "max_drawdown_pct")
    if math.isnan(mc):
        mc = _pick(rb, "mc_max_drawdown_pct", "max_drawdown_pct_at_confidence")

    # The bootstrap reports the 95% tail as a signed (negative) drawdown; take
    # the magnitude for the same reason as Gate 1.
    gate2 = [
        _criterion("WFO efficiency", wfo, t2["min_wfo_efficiency"], "min",
                   note=(rb.get("wfo") or {}).get("warning")
                   if isinstance(rb.get("wfo"), dict) else None),
        _criterion("Monte Carlo 95% max DD", abs(mc),
                   t2["max_mc_drawdown_pct"], "max", unit="%"),
    ]

    # Gate 3 measures retention on Sharpe where the in-sample Sharpe is
    # positive, and falls back to profit factor where it is not. The fallback
    # exists because Sharpe is no longer a Gate 1 condition: a strategy with a
    # negative in-sample Sharpe can now reach Gate 3, and its retention ratio
    # is undefined - two negative Sharpes divide to a healthy-looking positive
    # number. Profit factor is the metric Gate 1 does bind on and it is
    # strictly positive, so it is the one that still means something there.
    # The criterion's LABEL names whichever metric was used; the two are never
    # reported under one heading.
    oos_sharpe = _pick(holdout, "sharpe", "sharpe_ratio", "holdout_sharpe")
    # Only from a mapping. `_pick` reads a bare scalar as the value of
    # whatever key it was asked for, and the documented bare form of `holdout`
    # is a SHARPE - taking 1.80 as a profit factor of 1.80 would score Gate 3
    # on a number the caller never supplied.
    oos_pf = _pick(holdout, "profit_factor") if isinstance(holdout, dict) \
        else float("nan")
    sharpe_retention = float("nan")
    if not math.isnan(oos_sharpe) and not math.isnan(is_sharpe) and is_sharpe > 0:
        sharpe_retention = oos_sharpe / is_sharpe

    retention, retention_note = sharpe_retention, None
    retention_metric = "sharpe" if not math.isnan(sharpe_retention) else None
    retention_label = "Holdout Sharpe retention"
    if retention_metric is None:
        if not math.isnan(is_sharpe) and is_sharpe <= 0:
            undefined = (f"in-sample Sharpe is {is_sharpe:.2f} (<= 0), so "
                         f"Sharpe retention is undefined, not passing")
        elif math.isnan(oos_sharpe) and not math.isnan(is_sharpe):
            undefined = "no holdout Sharpe supplied, so Sharpe retention is undefined"
        elif math.isnan(is_sharpe):
            undefined = "no in-sample Sharpe supplied, so Sharpe retention is undefined"
        else:
            undefined = None
        if not math.isnan(oos_pf) and not math.isnan(is_pf) and is_pf > 0:
            retention, retention_metric = oos_pf / is_pf, "profit_factor"
            retention_label = "Holdout profit-factor retention"
            retention_note = (f"{undefined}; measured on profit factor instead"
                              if undefined else "measured on profit factor")
        else:
            retention_note = undefined
    gate3 = [
        _criterion(retention_label, retention, t3["min_retention"], "min",
                   unit="x", note=retention_note),
    ]

    gates = {
        "gate1": {"name": GATE_NAMES["gate1"], "status": _roll_up(gate1),
                  "checks": gate1},
        "gate2": {"name": GATE_NAMES["gate2"], "status": _roll_up(gate2),
                  "checks": gate2},
        "gate3": {"name": GATE_NAMES["gate3"], "status": _roll_up(gate3),
                  "checks": gate3},
    }
    statuses = [g["status"] for g in gates.values()]
    overall = (FAIL if FAIL in statuses
               else NOT_EVALUATED if NOT_EVALUATED in statuses
               else PASS)

    return {
        "version": version,
        "name": name or (metrics or {}).get("meta", {}).get("strategy", "unnamed"),
        "gates": gates,
        "status": overall,
        "passed": overall == PASS,
        "thresholds": GATE_THRESHOLDS,
        "holdout_sharpe": oos_sharpe,
        "in_sample_sharpe": is_sharpe,
        # `sharpe_retention` stays Sharpe-only and is NaN when that ratio is
        # undefined - a reader asking for the Sharpe retention must never be
        # handed a profit-factor ratio under that name. `retention` is what
        # Gate 3 actually scored and `retention_metric` says which it was.
        "sharpe_retention": sharpe_retention,
        "retention": retention,
        "retention_metric": retention_metric,
    }


# --------------------------------------------------------------------------
# Dual-version scorecard
# --------------------------------------------------------------------------
_SCORECARD_ROWS = [
    ("Sharpe", "sharpe", "{:.2f}", "high"),
    ("Sortino", "sortino", "{:.2f}", "high"),
    ("Calmar", "calmar", "{:.2f}", "high"),
    ("Profit factor", "profit_factor", "{:.2f}", "high"),
    ("Win rate %", "win_rate_pct", "{:.1f}", "high"),
    # Compared on magnitude. The engine signs drawdown negative, so "lower is
    # better" would mark B's DEEPER drawdown as the improvement - -36.73
    # against -29.27 is a delta of -7.47, which reads as progress and is the
    # opposite of what happened.
    ("Max drawdown %", "max_drawdown_pct", "{:.2f}", "smaller"),
    ("Net return %", "total_return_pct", "{:.2f}", "high"),
    ("CAGR %", "annualized_return_pct", "{:.2f}", "high"),
    ("Trades", "trade_count", "{:,.0f}", "none"),
    # The split under the total. "none" because neither side is the better
    # one - the row is here so a two-sided result cannot be read as if it were
    # one-sided, or a one-sided result mistaken for a strategy whose short
    # signals simply never triggered. Both read 0 on a long-only run.
    ("  · long", "long_trades", "{:,.0f}", "none"),
    ("  · short", "short_trades", "{:,.0f}", "none"),
    ("Total costs $", "total_costs", "{:,.0f}", "none"),
]


def _metric(metrics: dict, key: str) -> float:
    """
    One scorecard value.

    `win_rate` is stored as a fraction by `summarize_result` and shown as a
    percent here; every other key is passed through unchanged.
    """
    if key == "win_rate_pct":
        value = _pick(metrics, "win_rate_pct")
        return value if not math.isnan(value) else _pick(metrics, "win_rate") * 100
    return _pick(metrics, key)


def _cell(metrics: dict, key: str, fmt: str) -> str:
    value = _metric(metrics, key)
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return fmt.format(value)


def _delta_cell(metrics_a: dict, metrics_b: dict, key: str, fmt: str,
                better: str) -> str:
    a, b = _metric(metrics_a, key), _metric(metrics_b, key)
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return "n/a"
    d = b - a
    mark = ""
    if better != "none" and abs(d) > 1e-12:
        if better == "high":
            improved = d > 0
        elif better == "smaller":
            improved = abs(b) < abs(a)      # magnitude, so the sign cannot lie
        else:
            improved = d < 0
        mark = " +" if improved else " -"
    return (fmt.format(d) + mark).strip()


def _gate_line(audit: dict, key: str) -> str:
    return (audit or {}).get("gates", {}).get(key, {}).get("status", NOT_EVALUATED)


def format_dual_scorecard(metrics_a: dict, metrics_b: dict | None,
                          gate_audit_a: dict | None = None,
                          gate_audit_b: dict | None = None,
                          label_a: str = "A · rule-based",
                          label_b: str = "B · ML-filtered",
                          show_gates: bool = True) -> str:
    """
    The scorecard as a string, so it can be tested and written to a file.

    `show_gates=False` drops the gate table, the per-criterion detail and the
    NOT-EVALUATED warning, leaving the metric columns and the verdict. Stage 1
    of the pipeline (`backtest/baseline.py`) asks for that: it is a survival
    screen run on default parameters, where every gate would report NOT
    EVALUATED except a Gate 1 nobody is entitled to read yet - the parameters
    have not been swept and the holdout has not been touched. Three gates of
    NOT EVALUATED printed under a heading is clutter that trains a reader to
    skip the gate table, which is the one thing it must never do at Stage 3.
    The gates are not being hidden; they are being deferred to the stage that
    can actually evaluate them.

    `metrics_b=None` means Version B was not run (`--ml` off on the batch
    runner). The B and B−A columns are then omitted entirely rather than
    printed as `n/a`: a column of dashes next to a header reading
    "B · ML-filtered" invites the reading that the filter ran and produced
    nothing, and the whole point of the mandate is that a missing comparison
    is not a settled one. An empty dict is a Version B that ran and scored
    nothing measurable, which is a different finding and still prints.
    """
    W = 78
    dual = metrics_b is not None
    L: list[str] = []
    add = L.append

    add("=" * W)
    add("DUAL-VERSION SCORECARD" if dual else "SCORECARD · VERSION A ONLY")
    add("=" * W)

    meta = (metrics_a or {}).get("meta") or (metrics_b or {}).get("meta") or {}
    if meta:
        add(f"Strategy      : {meta.get('strategy', 'unnamed')}")
        add(f"Symbol / TF   : {meta.get('symbol', '?')} / {meta.get('timeframe', '?')}")
        add(f"Period        : {str(meta.get('start', '?'))[:19]} → "
            f"{str(meta.get('end', '?'))[:19]}")
        add(f"Costs included: {'yes' if meta.get('costs_included') else 'NO — results are not comparable to live'}")
    add("")
    if dual:
        add(f"  {'Metric':<22}{label_a:>20}{label_b:>20}{'B − A':>14}")
    else:
        add(f"  {'Metric':<22}{label_a:>20}")
    add("  " + "-" * (W - 4))
    for label, key, fmt, better in _SCORECARD_ROWS:
        row = f"  {label:<22}{_cell(metrics_a, key, fmt):>20}"
        if dual:
            row += (f"{_cell(metrics_b, key, fmt):>20}"
                    f"{_delta_cell(metrics_a, metrics_b, key, fmt, better):>14}")
        add(row)

    if show_gates:
        add("")
        add("-" * W)
        add("ACCEPTANCE GATES")
        add("-" * W)
        add(f"  {'Gate':<40}{'A':>18}" + (f"{'B':>18}" if dual else ""))
        for gk in ("gate1", "gate2", "gate3"):
            name = GATE_NAMES[gk]
            add(f"  {name:<40}{_gate_line(gate_audit_a, gk):>18}"
                + (f"{_gate_line(gate_audit_b, gk):>18}" if dual else ""))
        add("  " + "-" * (W - 4))
        add(f"  {'OVERALL':<40}{(gate_audit_a or {}).get('status', NOT_EVALUATED):>18}"
            + (f"{(gate_audit_b or {}).get('status', NOT_EVALUATED):>18}"
               if dual else ""))

        # Per-criterion detail, so a FAIL says which number failed and by how much.
        detail = ((gate_audit_a, label_a), (gate_audit_b, label_b)) if dual \
            else ((gate_audit_a, label_a),)
        for audit, label in detail:
            if not audit:
                continue
            add("")
            add(f"  {label}")
            for gk in ("gate1", "gate2", "gate3"):
                gate = audit["gates"][gk]
                add(f"    {gate['name']:<32}{gate['status']}")
                for c in gate["checks"]:
                    measured, required = criterion_text(c)
                    # 15 wide: 'informational' is 13 characters, and a required
                    # column that runs into the status column reads as one word.
                    add(f"      {c['label']:<28}{measured:>10}  "
                        f"{required:<15}{c['status']}")
                    if c.get("note"):
                        add(f"        ! {c['note']}")

        audited = ([(gate_audit_a, "A"), (gate_audit_b, "B")] if dual
                   else [(gate_audit_a, "A")])
        incomplete = [lbl for audit, lbl in audited
                      if audit and audit["status"] == NOT_EVALUATED]
        if incomplete:
            add("")
            add(f"  ⚠ Version {', '.join(incomplete)}: at least one gate was NOT")
            add("    EVALUATED. That is not a pass. Run the walk-forward, the Monte")
            add("    Carlo bootstrap and the 3-year holdout before promoting.")

    add("")
    add("-" * W)
    add("VERDICT")
    add("-" * W)
    sa, sb = _pick(metrics_a, "sharpe"), _pick(metrics_b, "sharpe")
    if not dual:
        add(f"  Version A Sharpe {sa:.2f}. Version B was NOT RUN, so the "
            f"Dual-Version")
        add("  Mandate's comparison has not been made — not made is not lost.")
        add("  Re-run with --ml before reading this as a result for the "
            "baseline.")
        add("=" * W)
        return "\n".join(L)
    if math.isnan(sa) or math.isnan(sb):
        add("  Sharpe is undefined for at least one version — no comparison.")
    elif sb > sa:
        add(f"  Version B leads on Sharpe by {sb - sa:+.2f}.")
    else:
        add(f"  Version A leads on Sharpe by {sa - sb:+.2f}. The ML filter did "
            f"not earn its place.")
    add("")
    add("  The Dual-Version Mandate adopts B only if it beats A OUT-OF-SAMPLE.")
    add("  An in-sample lead is not that evidence — the filter was fitted")
    add("  walk-forward on this very period.")
    add("=" * W)
    return "\n".join(L)


def print_dual_scorecard(metrics_a: dict, metrics_b: dict | None,
                         gate_audit_a: dict | None = None,
                         gate_audit_b: dict | None = None,
                         label_a: str = "A · rule-based",
                         label_b: str = "B · ML-filtered",
                         show_gates: bool = True) -> str:
    """
    Render the side-by-side terminal scorecard for both versions.

    Returns the same text it prints, so a caller can save it next to the HTML
    reports without formatting it twice.
    """
    text = format_dual_scorecard(metrics_a, metrics_b, gate_audit_a,
                                 gate_audit_b, label_a, label_b, show_gates)
    print(text)
    return text


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
    reg, regime_note = regime_join(yearly, regime_symbol)
    table = reg if reg is not None else yearly
    add(table.to_string(index=False))

    pos_years = int((yearly["return_pct"] > 0).sum())
    add("")
    add(f"  Positive years: {pos_years}/{len(yearly)}")

    if regime_note:
        # Surfaced in the report as well as on stderr: the saved report is what
        # gets read later, and a caveat that lives only in a terminal scrollback
        # has not been recorded.
        print(f"[!] {regime_note}", file=sys.stderr, flush=True)
        add("")
        add(f"  ⚠ REGIME ANALYSIS UNAVAILABLE")
        add(f"    {regime_note}")
        add(f"    This report cannot say whether the strategy is merely long")
        add(f"    market exposure. Treat that question as unanswered.")

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
