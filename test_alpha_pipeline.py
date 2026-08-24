#!/usr/bin/env python3
"""
Integration test for the pure-alpha pipeline.

Proves the whole chain runs on real data with the harmonized contract:

    lake (NQ) -> sma_crossover.signal_fn(bars) -> run_backtest -> tear sheet

Follows the `tests/` convention: no pytest, exits non-zero on failure, prints
what it checked. Unlike the suites in `tests/`, this one NEEDS the lake mounted
at /mnt/backtest.

Every metric below is computed here in NumPy/pandas from the engine's own
daily returns and trade list. Nothing is asked of a model, and nothing is
reported that was not derived from the simulation.

    python3 test_alpha_pipeline.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import math
import resource
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.engine import BacktestConfig, run_backtest        # noqa: E402
from strategies.experimental.sma_crossover import make_signal_fn  # noqa: E402

SYMBOL = "NQ"
TIMEFRAME = "1d"
START = "2010-01-01"
END = "2023-12-31"
RAM_CEILING_GIB = 3.0
TRADING_DAYS = 252

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Record a check and print it. Mirrors the tests/ helper."""
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def peak_rss_gib() -> float:
    """
    Peak RSS for this process, in GiB.

    ru_maxrss is the high-water mark, not the current usage, which is the
    figure that matters: a run that transiently allocated 20 GiB and freed it
    would look fine under a spot reading of RSS.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


# --------------------------------------------------------------------------
# Deterministic metrics
# --------------------------------------------------------------------------
def annualized_sharpe(returns: pd.Series) -> float:
    """Mean/sd of daily returns, scaled by sqrt(252). NaN if never varies."""
    sd = returns.std(ddof=1)
    if not sd or math.isnan(sd):
        return float("nan")
    return float(returns.mean() / sd * math.sqrt(TRADING_DAYS))


def annualized_sortino(returns: pd.Series) -> float:
    """
    Sharpe with the denominator restricted to downside deviation.

    Divides by the count of ALL periods, not just losing ones. Using only the
    losing days inflates the ratio for a strategy that rarely loses, which is
    exactly the strategy this is supposed to distinguish.
    """
    downside = returns.clip(upper=0.0)
    dd = math.sqrt(float((downside ** 2).sum()) / len(returns))
    if dd == 0.0:
        return float("nan")
    return float(returns.mean() / dd * math.sqrt(TRADING_DAYS))


def annualized_return(equity: pd.Series, periods: int) -> float:
    """CAGR from the equity curve. NaN if the account was ruined."""
    if equity.empty or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    years = periods / TRADING_DAYS
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown_pct(equity: pd.Series) -> float:
    """Worst peak-to-trough decline on the daily equity curve, in percent."""
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1.0).min() * 100)


def calmar(cagr: float, max_dd_pct: float) -> float:
    """CAGR over the absolute max drawdown. NaN when there was no drawdown."""
    if math.isnan(cagr) or math.isnan(max_dd_pct) or max_dd_pct == 0:
        return float("nan")
    return float(cagr / abs(max_dd_pct / 100))


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross wins over gross losses, on NET pnl so costs are inside the ratio."""
    if trades.empty:
        return float("nan")
    pnl = trades["pnl"]
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def win_rate(trades: pd.DataFrame) -> float:
    """Share of trades with net pnl > 0. Breakeven counts as a loss."""
    if trades.empty:
        return float("nan")
    return float((trades["pnl"] > 0).mean())


def _fmt(value: float, suffix: str = "") -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:,.2f}{suffix}"


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print("=" * 72)

    baseline = peak_rss_gib()
    print(f"Baseline peak RSS before the run: {baseline:.3f} GiB\n")

    # -- a) + b) load the lake and run the engine --------------------------
    print(f"Running {SYMBOL} {TIMEFRAME} {START} -> {END} through run_backtest…")
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        contracts=1,
        slippage_ticks=1.0,          # costs on from the first test, never off
        variants_tested=1,
        notes="test_alpha_pipeline baseline SMA crossover",
    )
    signal_fn = make_signal_fn(fast_window=10, slow_window=30)

    try:
        result = run_backtest(SYMBOL, TIMEFRAME, signal_fn,
                              start=START, end=END, cfg=cfg)
    except Exception:
        print("\nENGINE RAISED:\n")
        traceback.print_exc()
        return 1

    trades = result.trades
    returns = result.returns
    equity = result.equity
    print(f"Done — {len(trades):,} trades over {len(returns):,} daily periods.\n")

    # -- c) memory ceiling --------------------------------------------------
    print("Memory")
    peak = peak_rss_gib()
    check(f"peak RSS <= {RAM_CEILING_GIB} GiB", peak <= RAM_CEILING_GIB,
          f"{peak:.3f} GiB")

    # -- integrity checks ---------------------------------------------------
    print("\nIntegrity")
    check("the engine returned trades", not trades.empty,
          f"{len(trades)} trades")
    check("a daily return series was produced", len(returns) > 0,
          f"{len(returns)} periods")
    check("costs were actually charged",
          bool(not trades.empty and trades["costs"].sum() > 0),
          f"total costs {trades['costs'].sum():,.2f}" if not trades.empty else "")
    check("net pnl is gross minus costs",
          bool(trades.empty or np.isclose(
              trades["pnl"].sum(),
              trades["gross_pnl"].sum() - trades["costs"].sum(), atol=1e-6)))
    check("every trade is NQ",
          bool(trades.empty or set(trades["symbol"].unique()) == {SYMBOL}))
    check("no trade exits before it enters",
          bool(trades.empty or (trades["exit_time"] >= trades["entry_time"]).all()))

    # -- d) the tear sheet --------------------------------------------------
    sharpe = annualized_sharpe(returns)
    sortino = annualized_sortino(returns)
    mdd = max_drawdown_pct(equity)
    cagr = annualized_return(equity, len(returns))
    cal = calmar(cagr, mdd)
    pf = profit_factor(trades)
    wr = win_rate(trades)

    print("\n" + "=" * 72)
    print(f"PURE ALPHA TEAR SHEET — {SYMBOL} {TIMEFRAME} "
          f"{START} to {END}, costs included")
    print("=" * 72)
    rows = [
        ("Annualized Sharpe", _fmt(sharpe)),
        ("Annualized Sortino", _fmt(sortino)),
        ("Calmar", _fmt(cal)),
        ("Profit Factor", _fmt(pf)),
        ("Win Rate", _fmt(wr * 100 if not math.isnan(wr) else wr, "%")),
        ("Max Drawdown", _fmt(mdd, "%")),
        ("", ""),
        ("CAGR", _fmt(cagr * 100 if not math.isnan(cagr) else cagr, "%")),
        ("Total return", _fmt(result.stats["total_return_pct"], "%")),
        ("Net P&L", _fmt(result.stats["net_pnl"])),
        ("Total costs", _fmt(result.stats["total_costs"])),
        ("Trades", f"{len(trades):,}"),
    ]
    for label, value in rows:
        print(f"  {label:<22}{value:>18}" if label else "")
    print("=" * 72)

    # Cross-check our Sharpe against the engine's own, so a divergence in the
    # metric layer surfaces here rather than in a report months from now.
    print("\nCross-check")
    engine_sharpe = result.stats["sharpe"]
    check("tear-sheet Sharpe matches the engine's",
          bool(math.isnan(sharpe) and math.isnan(engine_sharpe))
          or bool(np.isclose(sharpe, engine_sharpe, atol=1e-9)),
          f"{sharpe:.6f} vs {engine_sharpe:.6f}")
    check("tear-sheet max drawdown matches the engine's",
          bool(np.isclose(mdd, result.stats["max_dd_pct"], atol=1e-9)),
          f"{mdd:.6f}% vs {result.stats['max_dd_pct']:.6f}%")

    print("\n" + "=" * 72)
    if _failures:
        print(f"FAILED — {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("All checks passed.")
    print("\nThis is an in-sample run on a deliberately trivial baseline. It "
          "says the pipeline works, not that the edge does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
