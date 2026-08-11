"""
backtest.engine - turns signals into a tested result.

Location:  ~/src/trading/backtest/engine.py

A strategy says when to be long, short, or flat. This says what that would
have cost and whether the account would have survived.

    from mdlib.lake import get_bars
    from backtest.engine import BacktestConfig, run_backtest

    bars = get_bars("ES", tf="1h", start="2013-01-01")
    entries, exits = my_strategy(bars)

    cfg = BacktestConfig(flat_by_close=True, trailing_drawdown_pct=5.0)
    res = run_backtest(bars, entries, exits, cfg)
    res.save("/mnt/backtest/artifacts/strat1")

Then:

    python backtest/report.py \
      --returns /mnt/backtest/artifacts/strat1_returns.parquet \
      --trades  /mnt/backtest/artifacts/strat1_trades.parquet \
      --name "Strat 1" --variants-tested 1 --costs-included yes

Why this is separate from the strategy
--------------------------------------
If each strategy carried its own cost handling you would end up with twenty
slightly different cost models and no way to know whether strategy A beat
strategy B or merely assumed cheaper fills. One wrapper means every idea is
judged on identical terms.

Structure
---------
The vectorbt call is isolated in `_simulate`. Everything around it - signal
preparation, session handling, cost computation, result formatting, the
drawdown check - is plain pandas, so it can be tested without vectorbt and
swapped if the engine ever changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import get_spec


# --------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """
    Everything that is not the strategy itself.

    The constraint set is a parameter, not a separate codebase: Portfolio A
    (intraday, prop accounts) sets flat_by_close=True and a trailing drawdown
    limit; Portfolio B (swing, own capital) leaves both off. Same engine,
    same strategy code.
    """

    initial_capital: float = 100_000.0
    contracts: int = 1

    # Costs. Defaults come from backtest/specs.py per symbol; these override.
    commission_per_side: float | None = None   # dollars per contract
    slippage_ticks: float = 1.0                # each way

    # Portfolio A constraints
    flat_by_close: bool = False
    session_close_utc: str = "20:00"           # 16:00 ET during EDT
    trailing_drawdown_pct: float | None = None  # e.g. 5.0 for FundedNext
    daily_loss_limit: float | None = None       # dollars

    # Data hygiene
    exclude_degraded: bool = True
    exclude_rolls: bool = True

    # Bookkeeping - carried into the result so a number is never read without
    # the context needed to judge it.
    variants_tested: int | None = None
    notes: str = ""


@dataclass
class BacktestResult:
    """
    Standard output. Every strategy produces this shape, so results can be
    compared and correlated against each other.
    """

    returns: pd.Series          # daily, decimal (0.01 = 1%)
    trades: pd.DataFrame        # entry_time, exit_time, symbol, pnl, direction
    equity: pd.Series
    config: BacktestConfig
    breach: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def save(self, prefix: str | Path) -> tuple[Path, Path]:
        """Write in the format backtest/report.py consumes."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        r = prefix.with_name(prefix.name + "_returns.parquet")
        t = prefix.with_name(prefix.name + "_trades.parquet")

        (self.returns.rename("returns").rename_axis("date")
             .reset_index().to_parquet(r, index=False))
        self.trades.to_parquet(t, index=False)
        return r, t

    def summary(self) -> str:
        lines = [
            f"  trades          {len(self.trades):>10,}",
            f"  total return    {self.stats.get('total_return_pct', float('nan')):>10.2f} %",
            f"  sharpe          {self.stats.get('sharpe', float('nan')):>10.2f}",
            f"  max drawdown    {self.stats.get('max_dd_pct', float('nan')):>10.2f} %",
            f"  total costs     {self.stats.get('total_costs', float('nan')):>10,.0f}",
        ]
        if self.breach.get("breached"):
            lines.append(f"  PROP BREACH     {self.breach['first_breach']}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Session handling
# --------------------------------------------------------------------------
def apply_flat_by_close(bars: pd.DataFrame,
                        entries: pd.Series,
                        exits: pd.Series,
                        session_close_utc: str = "20:00") -> tuple[pd.Series, pd.Series]:
    """
    Force an exit at the last bar of each session, and block entries on it.

    Portfolio A trades prop accounts, which do not permit overnight holds.
    This is not cosmetic - it truncates exactly the moves a trend or breakout
    strategy would otherwise capture, so results change materially. That is
    the point: the constraint has to be in the backtest, not discovered live.

    An entry on the final bar would be opened and closed in the same bar, so
    those are suppressed rather than left to produce a guaranteed cost.
    """
    ts = pd.to_datetime(bars["ts"], utc=True)
    close_t = pd.Timestamp(f"2000-01-01 {session_close_utc}", tz="UTC").time()

    # Session date: bars at or after the close belong to the next session.
    session = ts.dt.normalize()
    session = session.where(ts.dt.time < close_t, session + pd.Timedelta(days=1))

    is_last = session != session.shift(-1)
    is_last.iloc[-1] = True

    exits = exits.copy()
    entries = entries.copy()
    exits[is_last.values] = True
    entries[is_last.values] = False
    return entries, exits


def clean_signals(entries: pd.Series, exits: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Remove signals that cannot execute: an entry with no prior exit, and an
    exit with no open position. Prevents double-counting.
    """
    e = entries.fillna(False).astype(bool).to_numpy()
    x = exits.fillna(False).astype(bool).to_numpy()

    in_pos = False
    ke = np.zeros(len(e), dtype=bool)
    kx = np.zeros(len(x), dtype=bool)

    for i in range(len(e)):
        if not in_pos and e[i]:
            ke[i] = True
            in_pos = True
        elif in_pos and x[i]:
            kx[i] = True
            in_pos = False

    return (pd.Series(ke, index=entries.index),
            pd.Series(kx, index=exits.index))


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------
def round_turn_cost(symbol: str, cfg: BacktestConfig) -> float:
    """
    Total cost of one round trip, one contract: commission both sides plus
    slippage both sides.

    Slippage is charged in ticks each way on the assumption you pay the
    spread. Optimistic fill assumptions are the most common reason a backtest
    overstates performance, so the default is deliberately conservative.
    """
    spec = get_spec(symbol)
    commission = (cfg.commission_per_side
                  if cfg.commission_per_side is not None else spec.commission)
    slip = spec.slippage_dollars(cfg.slippage_ticks)
    return (commission + slip) * 2


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def _simulate(bars: pd.DataFrame,
              entries: pd.Series,
              exits: pd.Series,
              symbol: str,
              cfg: BacktestConfig) -> pd.DataFrame:
    """
    Walk the bars and produce a trade list.

    Deliberately simple and explicit rather than vectorised: entries and exits
    fill at the NEXT bar's open, never the signal bar's close. Acting on the
    bar that produced the signal is lookahead bias, and it is the single most
    common way a backtest lies.

    Returns: entry_time, exit_time, symbol, direction, entry_price,
             exit_price, gross_pnl, costs, pnl
    """
    spec = get_spec(symbol)
    cost = round_turn_cost(symbol, cfg) * cfg.contracts

    ts = pd.to_datetime(bars["ts"], utc=True).to_numpy()
    op = bars["open"].to_numpy(dtype=float)
    e = entries.to_numpy(dtype=bool)
    x = exits.to_numpy(dtype=bool)

    trades = []
    in_pos = False
    entry_i = -1

    for i in range(len(ts) - 1):
        if not in_pos and e[i]:
            entry_i = i + 1          # fill next bar open
            in_pos = True
        elif in_pos and x[i]:
            exit_i = i + 1
            gross = (op[exit_i] - op[entry_i]) * spec.multiplier * cfg.contracts
            trades.append({
                "entry_time": ts[entry_i],
                "exit_time": ts[exit_i],
                "symbol": symbol,
                "direction": "long",
                "entry_price": op[entry_i],
                "exit_price": op[exit_i],
                "gross_pnl": gross,
                "costs": cost,
                "pnl": gross - cost,
            })
            in_pos = False

    return pd.DataFrame(trades)


def _daily_returns(trades: pd.DataFrame,
                   bars: pd.DataFrame,
                   initial_capital: float) -> tuple[pd.Series, pd.Series]:
    """
    Convert a trade list into a daily return series.

    P&L is attributed to the exit date, which is when it is realised. Every
    calendar day in the backtest window appears, including flat days - a
    strategy that trades rarely should show that in its return series rather
    than compressing time.
    """
    idx = pd.to_datetime(bars["ts"], utc=True).dt.normalize().drop_duplicates()
    idx = pd.DatetimeIndex(sorted(idx))

    daily_pnl = pd.Series(0.0, index=idx)
    if not trades.empty:
        by_day = (trades.assign(d=pd.to_datetime(trades["exit_time"], utc=True)
                                  .dt.normalize())
                        .groupby("d")["pnl"].sum())
        daily_pnl = daily_pnl.add(by_day.reindex(idx).fillna(0.0), fill_value=0.0)

    equity = initial_capital + daily_pnl.cumsum()
    returns = equity.pct_change().fillna(0.0)
    return returns, equity


def check_trailing_drawdown(equity: pd.Series, limit_pct: float) -> dict:
    """
    Prop firm trailing drawdown check.

    The limit trails the high water mark, so the PATH of returns matters more
    than the total. A strategy with a strong Sharpe and a deep drawdown fails
    the account regardless of eventual profitability - which is why this is a
    first-class output rather than an afterthought.
    """
    peak = equity.cummax()
    floor = peak * (1 - limit_pct / 100)
    breached = equity < floor
    return {
        "breached": bool(breached.any()),
        "first_breach": str(breached.idxmax().date()) if breached.any() else None,
        "worst_margin_pct": float(((equity - floor) / peak).min() * 100),
        "limit_pct": limit_pct,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_backtest(bars: pd.DataFrame,
                 entries: pd.Series,
                 exits: pd.Series,
                 cfg: BacktestConfig | None = None) -> BacktestResult:
    """
    Run one strategy over one or more symbols.

    `bars` is long format from mdlib.lake.get_bars. `entries` and `exits` are
    boolean Series aligned to it. For multiple symbols, signals must be
    aligned to the same long frame - each symbol is simulated independently
    and the results are pooled.
    """
    cfg = cfg or BacktestConfig()

    if bars.empty:
        raise ValueError("No bars supplied.")
    if len(entries) != len(bars) or len(exits) != len(bars):
        raise ValueError(
            f"Signal length mismatch: bars={len(bars)}, "
            f"entries={len(entries)}, exits={len(exits)}"
        )

    bars = bars.reset_index(drop=True)
    entries = pd.Series(entries).reset_index(drop=True)
    exits = pd.Series(exits).reset_index(drop=True)

    all_trades = []

    for sym, g in bars.groupby("symbol", sort=True):
        idx = g.index
        g = g.reset_index(drop=True)
        e = entries.loc[idx].reset_index(drop=True)
        x = exits.loc[idx].reset_index(drop=True)

        if cfg.flat_by_close:
            e, x = apply_flat_by_close(g, e, x, cfg.session_close_utc)

        e, x = clean_signals(e, x)
        t = _simulate(g, e, x, sym, cfg)
        if not t.empty:
            all_trades.append(t)

    trades = (pd.concat(all_trades, ignore_index=True)
              if all_trades else
              pd.DataFrame(columns=["entry_time", "exit_time", "symbol",
                                    "direction", "entry_price", "exit_price",
                                    "gross_pnl", "costs", "pnl"]))
    if not trades.empty:
        trades = trades.sort_values("exit_time").reset_index(drop=True)

    returns, equity = _daily_returns(trades, bars, cfg.initial_capital)

    breach = (check_trailing_drawdown(equity, cfg.trailing_drawdown_pct)
              if cfg.trailing_drawdown_pct else {})

    dd = equity / equity.cummax() - 1.0
    sd = returns.std(ddof=1)
    stats = {
        "n_trades": len(trades),
        "total_return_pct": float((equity.iloc[-1] / cfg.initial_capital - 1) * 100),
        "sharpe": float(returns.mean() / sd * np.sqrt(252)) if sd else float("nan"),
        "max_dd_pct": float(dd.min() * 100),
        "total_costs": float(trades["costs"].sum()) if not trades.empty else 0.0,
        "gross_pnl": float(trades["gross_pnl"].sum()) if not trades.empty else 0.0,
        "net_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
    }

    return BacktestResult(returns=returns, trades=trades, equity=equity,
                          config=cfg, breach=breach, stats=stats)
