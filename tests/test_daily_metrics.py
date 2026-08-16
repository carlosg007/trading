"""
tests/test_daily_metrics.py - the metric frequency contract.

Location: ~/src/trading/tests/test_daily_metrics.py

Sharpe, Sortino and Calmar are annualized with sqrt(252) everywhere in this
repo, which is only correct if the return series they are handed is DAILY.
Nothing about a 15m return series announces itself - it is the right dtype,
the right length and has a sane-looking index - so a regression here would
show up as a Sharpe that is simply the wrong size, on a scorecard, with
nothing raising. These checks pin the frequency itself rather than any
particular number.

Runs without the lake and without a network: the bars are synthetic and
`mdlib.lake.iter_bars` is stubbed.

    python tests/test_daily_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.report import (TRADING_DAYS, annualized_return_pct,  # noqa: E402
                             calmar, daily_metrics, daily_returns,
                             is_intraday, sharpe, to_daily_equity)

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(f"{label}{(' - ' + detail) if detail else ''}")


# --------------------------------------------------------------------------
def synthetic_15m_bars(n_sessions: int = 40, per_session: int = 26,
                       seed: int = 7) -> pd.DataFrame:
    """n_sessions business days of 15m bars, one symbol, UTC."""
    rng = np.random.default_rng(seed)
    rows, px = [], 4000.0
    for d in pd.bdate_range("2020-01-01", periods=n_sessions, tz="UTC"):
        for k in range(per_session):
            ts = d + pd.Timedelta(hours=14, minutes=30) + pd.Timedelta(minutes=15 * k)
            px *= 1 + rng.normal(0, 0.0008)
            rows.append((ts, "ES", px, px * 1.002, px * 0.998, px, 1000))
    return pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low",
                                       "close", "volume"])


# --------------------------------------------------------------------------
print("\n1. to_daily_equity collapses onto session closes")

idx = pd.DatetimeIndex(
    ["2020-01-01 14:30", "2020-01-01 20:00", "2020-01-02 14:30",
     "2020-01-02 20:00"], tz="UTC")
intraday_eq = pd.Series([100.0, 110.0, 105.0, 130.0], index=idx)
d = to_daily_equity(intraday_eq)

check("4 intraday points over 2 sessions -> 2 daily points", len(d) == 2,
      f"got {len(d)}")
check("keeps each session's LAST value, not its first",
      list(d.values) == [110.0, 130.0], f"got {list(d.values)}")
check("index is normalized to midnight",
      bool((d.index == d.index.normalize()).all()))
check("is_intraday spots the intraday index", is_intraday(intraday_eq.index))
check("is_intraday clears an already-daily index", not is_intraday(d.index))
check("to_daily_equity is idempotent", to_daily_equity(d).equals(d))


# --------------------------------------------------------------------------
print("\n2. daily returns skip the days the market never traded")

# Fri 2020-01-03 -> Mon 2020-01-06. A calendar resample would invent Sat/Sun.
eq = pd.Series([100.0, 101.0, 102.0],
               index=pd.DatetimeIndex(["2020-01-02", "2020-01-03",
                                       "2020-01-06"], tz="UTC"))
r = daily_returns(eq)
check("3 sessions across a weekend -> 2 returns, no padded zeros", len(r) == 2,
      f"got {len(r)}")

padded = eq.resample("1D").last().ffill().pct_change().dropna()
check("the calendar-resample recipe would have padded to 4",
      len(padded) == 4, f"got {len(padded)}")
check("and would have injected zero-return days",
      float((padded == 0.0).sum()) == 2.0, f"got {list(padded.values)}")


# --------------------------------------------------------------------------
print("\n3. the seed scores the first session instead of consuming it")

seeded = daily_returns(eq, initial_capital=100.0)
check("seeded series keeps one return per session", len(seeded) == 3,
      f"got {len(seeded)}")
check("first entry is the real day-one return, not a filled zero",
      abs(float(seeded.iloc[0]) - 0.0) < 1e-12,
      f"got {float(seeded.iloc[0])}")

# A strategy that closes a trade on session one: the seed is what makes that
# session's P&L visible at all.
eq_day1 = pd.Series([104.0, 104.0],
                    index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"],
                                           tz="UTC"))
s1 = daily_returns(eq_day1, initial_capital=100.0)
check("a day-one winner is scored, not swallowed by the base",
      abs(float(s1.iloc[0]) - 0.04) < 1e-12, f"got {float(s1.iloc[0])}")
check("unseeded, that same day-one P&L is invisible",
      len(daily_returns(eq_day1)) == 1)


# --------------------------------------------------------------------------
print("\n4. sqrt(252) is applied to a daily series, and only a daily one")

rng = np.random.default_rng(3)
dr = pd.Series(rng.normal(0.001, 0.01, 500),
               index=pd.bdate_range("2018-01-01", periods=500, tz="UTC"))
manual = float(dr.mean() / dr.std(ddof=1) * np.sqrt(252))
check("sharpe() == mean/std * sqrt(252) at rf=0",
      abs(sharpe(dr) - manual) < 1e-9, f"{sharpe(dr)} vs {manual}")
check("TRADING_DAYS is 252", TRADING_DAYS == 252)

# Same equity path, sampled twice. The ratio must not depend on the sampling.
n = 40
daily_eq = pd.Series(100_000 + np.cumsum(rng.normal(50, 300, n)),
                     index=pd.bdate_range("2019-01-01", periods=n, tz="UTC"))
fine_idx, fine_val = [], []
for ts, v in daily_eq.items():
    for k in range(26):                       # 26 intraday marks per session
        fine_idx.append(ts + pd.Timedelta(minutes=15 * k))
        fine_val.append(v if k == 25 else v * 1.003)   # noise inside the day
fine_eq = pd.Series(fine_val, index=pd.DatetimeIndex(fine_idx))

m_daily = daily_metrics(daily_eq, 100_000.0)
m_fine = daily_metrics(fine_eq, 100_000.0)
for key in ("sharpe", "sortino", "calmar", "annualized_return_pct", "max_dd_pct"):
    check(f"{key} is identical whether fed daily or 15m marks",
          abs(m_daily[key] - m_fine[key]) < 1e-9,
          f"{m_daily[key]} vs {m_fine[key]}")

naive = float(fine_eq.pct_change().dropna().mean()
              / fine_eq.pct_change().dropna().std(ddof=1) * np.sqrt(252))
check("and differs from the un-standardized intraday Sharpe",
      abs(naive - m_fine["sharpe"]) > 1e-6,
      "the two agree, so this test proves nothing")


# --------------------------------------------------------------------------
print("\n5. daily_metrics reports the basis it was sampled on")

b = m_daily["basis"]
check("basis names the frequency", b["frequency"] == "daily_close", str(b))
check("basis names the annualization factor",
      b["annualization_factor"] == 252, str(b))
check("basis records the risk-free rate", b["risk_free_rate"] == 0.0, str(b))
check("basis records the day count", b["n_days"] == n, str(b))

rf_m = daily_metrics(daily_eq, 100_000.0, rf=0.04)
check("a non-zero rf is recorded", rf_m["basis"]["risk_free_rate"] == 0.04)
check("a non-zero rf lowers Sharpe", rf_m["sharpe"] < m_daily["sharpe"],
      f"{rf_m['sharpe']} vs {m_daily['sharpe']}")


# --------------------------------------------------------------------------
print("\n6. calmar and CAGR come off the same daily series")

ann = annualized_return_pct(daily_eq, 100_000.0)
dd = float((daily_eq / daily_eq.cummax() - 1.0).min() * 100)
check("calmar == CAGR / |max drawdown|",
      abs(calmar(ann, dd) - ann / abs(dd)) < 1e-9)
check("calmar uses drawdown MAGNITUDE (engine signs it negative)",
      abs(calmar(ann, dd) - calmar(ann, abs(dd))) < 1e-9)
check("no drawdown -> NaN, not inf", np.isnan(calmar(ann, 0.0)))


# --------------------------------------------------------------------------
print("\n7. end to end: a 15m backtest reports DAILY stats")

import mdlib.lake as lake                                        # noqa: E402
from backtest.engine import BacktestConfig, run_backtest         # noqa: E402

bars = synthetic_15m_bars()
lake.iter_bars = (lambda symbols, tf, start=None, end=None, **kw:
                  iter([("ES", bars.reset_index(drop=True))]))


def signal_fn(g):
    e = pd.Series(np.zeros(len(g), dtype=bool))
    x = pd.Series(np.zeros(len(g), dtype=bool))
    e.iloc[::13] = True
    x.iloc[6::13] = True
    return e, x


res = run_backtest("ES", "15m", signal_fn, cfg=BacktestConfig())
n_sessions = bars["ts"].dt.normalize().nunique()

check(f"{len(bars)} 15m bars -> {n_sessions} return points",
      len(res.returns) == n_sessions, f"got {len(res.returns)}")
check("returns index is one entry per session",
      res.returns.index.equals(res.returns.index.normalize().unique()))
check("equity index matches the returns index",
      len(res.equity) == n_sessions)

for key in ("sharpe", "sortino", "calmar", "annualized_return_pct", "basis"):
    check(f"stats records {key}", key in res.stats, str(sorted(res.stats)))

check("stats sharpe == report.sharpe on the daily returns",
      abs(res.stats["sharpe"] - sharpe(res.returns)) < 1e-9,
      f"{res.stats['sharpe']} vs {sharpe(res.returns)}")
check("stats basis says daily_close",
      res.stats["basis"]["frequency"] == "daily_close")
check("stats basis day count matches the series",
      res.stats["basis"]["n_days"] == len(res.returns))

# The scorecard JSON has to carry the same numbers the run was judged on.
from agents.tier3_workers import summarize_result                # noqa: E402

sm = summarize_result(res, include_trades=False)
for key in ("sharpe", "sortino", "calmar", "annualized_return_pct"):
    a, b_ = sm[key], res.stats[key]
    ok = (np.isnan(a) and np.isnan(b_)) or abs(a - b_) < 1e-9
    check(f"summarize_result passes {key} through unchanged", ok, f"{a} vs {b_}")
check("summarize_result carries the basis into the scorecard",
      sm["metrics_basis"].get("frequency") == "daily_close",
      str(sm.get("metrics_basis")))

# A 1d run of the same path must agree with the 15m run's frequency handling.
daily_bars = (bars.set_index("ts")
                  .groupby([pd.Grouper(freq="1D"), "symbol"])
                  .agg(open=("open", "first"), high=("high", "max"),
                       low=("low", "min"), close=("close", "last"),
                       volume=("volume", "sum"))
                  .dropna().reset_index())
lake.iter_bars = (lambda symbols, tf, start=None, end=None, **kw:
                  iter([("ES", daily_bars.reset_index(drop=True))]))
res_d = run_backtest("ES", "1d", signal_fn, cfg=BacktestConfig())
check("a 1d run of the same window yields the same number of return points",
      len(res_d.returns) == len(res.returns),
      f"{len(res_d.returns)} vs {len(res.returns)}")


# --------------------------------------------------------------------------
print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All daily-metric checks passed.")
