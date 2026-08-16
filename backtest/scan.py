#!/usr/bin/env python3
"""
scan.py - vectorized parameter grid search, one symbol at a time.

Location:  ~/src/trading/backtest/scan.py

A strategy declares its own search space:

    PARAM_GRID = {"fast_window": [5, 10, 20], "slow_window": [30, 50, 100]}

`backtest/run.py --scan` sweeps that space per symbol and keeps the parameter
set with the highest Sharpe **among those that clear Gate 1**. The winning set
is then re-run through the ordinary dual-version path, so the numbers that
reach a report come from the same code that produces every other number here.

Why the grid lives in the strategy module
-----------------------------------------
The module is the only place that knows what its parameters mean and what its
signature will accept. A grid written in the runner would drift from the
function it has to bind against, and the first symptom would be a sweep that
silently tested nothing - `load_strategy` rejects unknown parameter names, so
a stale key raises here rather than being quietly ignored.

How this is vectorized
----------------------
Every parameter combination becomes a COLUMN. Signals are computed per column
in pandas (cheap - a rolling mean over a few hundred thousand rows), stacked
into one wide boolean frame, and handed to a SINGLE
`vbt.Portfolio.from_signals` call that simulates all of them at once. That is
the vectorbt-native shape for a sweep and it is why a 27-cell grid costs
roughly one backtest rather than 27.

Columns are batched so peak RAM tracks `bars x columns-per-batch` rather than
`bars x whole grid`; vectorbt allocates several full float64 arrays per column
and a wide grid on a long 1-minute history would otherwise be the largest
allocation in the process. `backtest.engine._simulate` batches BARS for the
same reason; this batches columns because the trade records of one column
cannot be stitched across a bar boundary here without reimplementing
`_chunk_bounds` per column.

Identical numbers to a single run
---------------------------------
The per-column trade list is assembled with the engine's own
`_assemble_result`, from the engine's own cost arrays, with the same one-bar
signal shift and the same next-bar-open fill. A column's metrics are therefore
the metrics `_simulate` would have produced for that parameter set on its own -
`tests/test_scan.py` asserts exactly that, trade for trade. A scanner that
ranked on its own private Sharpe would select a parameter set the real engine
then scores differently, and the discrepancy would show up as an unexplained
gap between the leaderboard and the tear sheet.

What this deliberately does not do
----------------------------------
Selecting the best of N parameter sets is a search, and a Sharpe read without
knowing N is not a measurement. The winner carries `variants_tested` (the count
of combinations that actually ran) onto the config, into the report and into
the leaderboard. Nothing here treats a swept Sharpe as evidence of an edge: the
sweep is in-sample by construction, and Gates 2 and 3 - walk-forward, bootstrap,
the held-back final three years - are separate runs that this does not perform.
"""

from __future__ import annotations

import gc
import itertools
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.engine import (BacktestConfig, TRADE_COLUMNS,   # noqa: E402
                             _assemble_result, _cost_arrays, apply_flat_by_close,
                             clean_signals)
from backtest.report import PASS, audit_acceptance_gates       # noqa: E402
from backtest.specs import get_spec                            # noqa: E402

try:
    import vectorbtpro as vbt
    _VBT_IMPORT_ERROR: Exception | None = None
except ImportError as e:      # pragma: no cover - depends on the environment
    vbt = None
    _VBT_IMPORT_ERROR = e


# Bars x columns per `from_signals` call. vectorbt holds several full-length
# float64 arrays per column (cash, position, value, order and trade records),
# so this is the knob that keeps a wide grid on a long history from becoming
# the largest allocation in the process. It changes RAM and nothing else: the
# columns in a batch do not interact, so the results are identical at any
# value.
MAX_CELLS = 4_000_000

# How a winner was chosen. Recorded verbatim in the leaderboard, because
# "highest Sharpe of the sets that cleared Gate 1" and "highest Sharpe of a
# grid where nothing cleared Gate 1" are different findings and only one of
# them is a result.
SELECTED_GATE1 = "GATE 1 PASS · highest Sharpe"
SELECTED_NO_GATE1 = "HIGHEST SHARPE · NO COMBINATION CLEARED GATE 1"
SELECTED_NONE = "NO COMBINATION PRODUCED A MEASURABLE SHARPE"


class ScanError(RuntimeError):
    """The grid could not be swept at all."""


def expand_grid(grid: dict[str, Iterable]) -> list[dict[str, Any]]:
    """
    The cartesian product of a PARAM_GRID, in declaration order.

    Declaration order rather than sorted order so the CSV a human reads varies
    its last-declared parameter fastest, which is how the grid was written. A
    scalar is accepted and treated as a one-value axis, so pinning one
    parameter does not require wrapping it in a list.
    """
    if not grid:
        return []
    keys, axes = [], []
    for key, values in grid.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            values = [values]
        values = list(values)
        if not values:
            raise ScanError(f"PARAM_GRID['{key}'] is empty - nothing to sweep")
        keys.append(key)
        axes.append(values)
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


def _shift_to_fill(signals: pd.Series) -> np.ndarray:
    """
    A signal on bar i executes on bar i+1, exactly as `_simulate` does it.

    np.roll wraps the last element to the front, where it is cleared - which is
    the "no next bar to fill on" case. Duplicated from the engine rather than
    imported because it is three lines living inside `_simulate`'s loop; if it
    ever stops matching, `tests/test_scan.py` fails on the trade list.
    """
    out = np.roll(np.asarray(signals, dtype=bool), 1)
    out[0] = False
    return out


def _combo_signals(strategy_path: str | Path,
                   bars: pd.DataFrame,
                   base_params: dict,
                   combo: dict,
                   cfg: BacktestConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill-bar entries and exits for one parameter combination.

    The module is re-imported and re-bound per combination through
    `load_strategy`, so a combination the strategy rejects - `fast_window >=
    slow_window` on the SMA baseline - raises here and is recorded as REJECTED
    rather than being swept as though it were a valid point in the space.
    """
    from agents.tier3_workers import load_strategy

    fn, _info = load_strategy(strategy_path, {**base_params, **combo})
    entries, exits = fn(bars)
    if len(entries) != len(bars) or len(exits) != len(bars):
        raise ValueError(
            f"signal_fn returned {len(entries)}/{len(exits)} signals for "
            f"{len(bars)} bars")

    entries = pd.Series(entries).reset_index(drop=True).fillna(False).astype(bool)
    exits = pd.Series(exits).reset_index(drop=True).fillna(False).astype(bool)

    if cfg.flat_by_close:
        entries, exits = apply_flat_by_close(bars, entries, exits,
                                             cfg.session_close_utc)
    entries, exits = clean_signals(entries, exits)
    return _shift_to_fill(entries), _shift_to_fill(exits)


def _batch_columns(n_bars: int, n_cols: int, max_cells: int) -> list[tuple[int, int]]:
    """`[(lo, hi), ...]` column slices whose bars x columns stays under the cap."""
    per = max(1, int(max_cells // max(1, n_bars)))
    return [(lo, min(lo + per, n_cols)) for lo in range(0, n_cols, per)]


def _simulate_columns(bars: pd.DataFrame,
                      ent: np.ndarray,
                      exi: np.ndarray,
                      symbol: str,
                      cfg: BacktestConfig,
                      max_cells: int = MAX_CELLS) -> list[pd.DataFrame]:
    """
    One trade list per column, from batched multi-column `from_signals` calls.

    `ent` and `exi` are (bars x columns) boolean arrays ALREADY shifted to the
    fill bar. Slippage is a per-bar fraction of price and does not depend on the
    column, so it broadcasts; fees do depend on the column, because the fee
    fraction is quoted against the FILL price and a bar where one column enters
    and another exits has two different fills. A single shared fee array would
    charge one of those columns the wrong side's fill and the error would be a
    fraction of a tick - invisible in the totals, wrong in every one of them.
    """
    if vbt is None:
        raise ImportError(
            "vectorbtpro is required by backtest.scan. "
            f"Import failed with: {_VBT_IMPORT_ERROR}") from _VBT_IMPORT_ERROR

    spec = get_spec(symbol)
    px = bars["open"].to_numpy(dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    n_bars, n_cols = ent.shape

    out: list[pd.DataFrame] = [pd.DataFrame(columns=TRADE_COLUMNS)
                               for _ in range(n_cols)]

    for lo, hi in _batch_columns(n_bars, n_cols, max_cells):
        cols = [f"c{j}" for j in range(lo, hi)]
        e_df = pd.DataFrame(ent[:, lo:hi], index=index, columns=cols)
        x_df = pd.DataFrame(exi[:, lo:hi], index=index, columns=cols)
        if not e_df.to_numpy().any():
            continue

        # _cost_arrays is the engine's, called once per column so the fee
        # denominator uses that column's own fills. Slippage comes back
        # identical every time - it is price and tick size only - so the first
        # column's copy is kept and the rest discarded.
        slippage = None
        fees = {}
        for j, col in zip(range(lo, hi), cols):
            s, f, size = _cost_arrays(bars, ent[:, j], exi[:, j], symbol, cfg)
            slippage = s if slippage is None else slippage
            fees[col] = f
        price = pd.Series(px, index=index)

        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=e_df,
            exits=x_df,
            price=price,
            size=size,
            size_type="amount",
            fees=pd.DataFrame(fees, index=index, columns=cols),
            slippage=pd.Series(slippage, index=index),
            # Same reasoning as the engine: futures are margined, the account
            # curve is rebuilt from realised P&L, and an unbounded balance
            # keeps vectorbt from rejecting an order on notional. It also keeps
            # the columns independent, which is the whole premise of sweeping
            # them in one call.
            init_cash=np.inf,
            direction="longonly",
            accumulate=False,
        )

        rec = pf.trades.records
        rec = rec[rec["status"] == 1]     # closed only - an open position at
        if not rec.empty:                 # the end of the data realised nothing
            for col_i, part in rec.groupby("col", sort=False):
                entry_i = part["entry_idx"].to_numpy()
                exit_i = part["exit_idx"].to_numpy()
                entry_px = px[entry_i]
                exit_px = px[exit_i]
                gross = (exit_px - entry_px) * spec.multiplier * cfg.contracts
                pnl = part["pnl"].to_numpy()
                out[lo + int(col_i)] = pd.DataFrame({
                    "entry_time": index[entry_i],
                    "exit_time": index[exit_i],
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "gross_pnl": gross,
                    "costs": gross - pnl,
                    "pnl": pnl,
                })

        del pf, rec, e_df, x_df, fees, price
        gc.collect()

    return out


def scan_symbol(strategy_path: str | Path,
                bars: pd.DataFrame,
                symbol: str,
                cfg: BacktestConfig,
                grid: dict[str, Iterable],
                base_params: dict | None = None,
                max_cells: int = MAX_CELLS,
                strat_name: str | None = None) -> dict[str, Any]:
    """
    Sweep `grid` over one symbol's bars and pick a winner.

    Selection is the highest Sharpe among the combinations whose Gate 1 audit
    is PASS. When nothing clears Gate 1 the highest Sharpe overall is returned
    with `selection` set to say so - the sweep still has to hand the runner
    something to report, and labelling it plainly is the alternative to either
    inventing a pass or refusing to produce a result at all. The gate audit on
    the eventual run will fail either way; this only records how the parameters
    on the report got there.

    Returns a dict with `table` (one row per combination), `winner`
    (`{params, metrics, gate1}` or None), `selection`, `evaluated`, and
    `rejected` (combinations the strategy itself refused, with the reason).
    """
    base_params = dict(base_params or {})
    combos = expand_grid(grid)
    if not combos:
        raise ScanError(
            f"{strat_name or Path(strategy_path).stem} declares no PARAM_GRID, "
            f"so --scan has nothing to sweep. Add one, or drop --scan.")

    n_bars = len(bars)
    if n_bars == 0:
        raise ScanError(f"no bars for {symbol} - nothing to sweep")

    valid: list[dict] = []
    rejected: list[dict] = []
    ent_cols: list[np.ndarray] = []
    exi_cols: list[np.ndarray] = []

    for combo in combos:
        try:
            e, x = _combo_signals(strategy_path, bars, base_params, combo, cfg)
        except Exception as exc:                                # noqa: BLE001
            # A strategy that refuses a combination is not a failure of the
            # sweep. sma_crossover raises on fast >= slow, which is most of a
            # square grid, and dropping those silently would report a 9-cell
            # search that only ever tested 3.
            rejected.append({"params": dict(combo),
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        valid.append(dict(combo))
        ent_cols.append(e)
        exi_cols.append(x)

    if not valid:
        raise ScanError(
            f"every one of the {len(combos)} combinations in the grid was "
            f"rejected by the strategy. First reason: "
            f"{rejected[0]['reason'] if rejected else 'unknown'}")

    ent = np.column_stack(ent_cols)
    exi = np.column_stack(exi_cols)
    del ent_cols, exi_cols

    trade_lists = _simulate_columns(bars, ent, exi, symbol, cfg,
                                    max_cells=max_cells)
    del ent, exi

    days = pd.DatetimeIndex(np.unique(
        pd.DatetimeIndex(bars["ts"]).values.astype("datetime64[D]"))
    ).tz_localize("UTC")

    from agents.tier3_workers import summarize_result

    rows: list[dict] = []
    for combo, trades in zip(valid, trade_lists):
        result = _assemble_result([trades] if not trades.empty else [],
                                  days, cfg)
        # include_trades=False: a sweep holding every column's trade log at
        # once is how a bounded search turns into an unbounded one.
        metrics = summarize_result(result, include_trades=False)
        audit = audit_acceptance_gates(metrics, version="A",
                                       name=strat_name or symbol)
        gate1 = audit["gates"]["gate1"]
        failed = [c["label"] for c in gate1["checks"] if c["status"] != PASS]
        rows.append({
            **combo,
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "profit_factor": metrics["profit_factor"],
            "trades": metrics["trade_count"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "total_return_pct": metrics["total_return_pct"],
            "total_costs": metrics["total_costs"],
            "gate1": gate1["status"],
            "gate1_shortfalls": ", ".join(failed),
            "_metrics": metrics,
        })

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_metrics"}
                          for r in rows])

    passing = [r for r in rows
               if r["gate1"] == PASS and not pd.isna(r["sharpe"])]
    pool = passing or [r for r in rows if not pd.isna(r["sharpe"])]
    if passing:
        selection = SELECTED_GATE1
    elif pool:
        selection = SELECTED_NO_GATE1
    else:
        selection = SELECTED_NONE

    winner = None
    if pool:
        best = max(pool, key=lambda r: r["sharpe"])
        winner = {
            "params": {k: best[k] for k in valid[0]},
            "metrics": best["_metrics"],
            "gate1": best["gate1"],
            "sharpe": best["sharpe"],
        }
        table["selected"] = [
            all(row.get(k) == v for k, v in winner["params"].items())
            for _, row in table.iterrows()]
    else:
        table["selected"] = False

    # Sharpe descending so the CSV opens on the interesting end. NaN Sharpes -
    # a combination that never traded - sort last rather than being dropped:
    # "this parameter set produces no trades" is a finding about the space.
    table = table.sort_values("sharpe", ascending=False,
                              na_position="last").reset_index(drop=True)

    return {
        "symbol": symbol,
        "combinations": len(combos),
        "evaluated": len(valid),
        "rejected": rejected,
        "table": table,
        "winner": winner,
        "selection": selection,
    }


def write_scan_table(scan: dict, out_dir: str | Path) -> Path:
    """`scan_<symbol>.csv` - every combination that ran, winner flagged."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scan_{scan['symbol']}.csv"
    scan["table"].to_csv(path, index=False)
    return path


def format_scan_summary(scan: dict, top: int = 5) -> str:
    """A few lines for the console: what was searched, what won, and how."""
    L = [f"  grid: {scan['combinations']} combinations, {scan['evaluated']} "
         f"evaluated, {len(scan['rejected'])} rejected by the strategy"]
    if scan["rejected"]:
        L.append(f"    (first rejection: {scan['rejected'][0]['params']} — "
                 f"{scan['rejected'][0]['reason']})")
    table = scan["table"]
    head = table.head(top)
    param_cols = [c for c in table.columns
                  if c not in ("sharpe", "sortino", "profit_factor", "trades",
                               "max_drawdown_pct", "total_return_pct",
                               "total_costs", "gate1", "gate1_shortfalls",
                               "selected")]
    L.append(f"    {'params':<34}{'Sharpe':>9}{'PF':>8}{'trades':>9}"
             f"{'maxDD%':>9}  gate1")
    for _, r in head.iterrows():
        params = ", ".join(f"{c}={r[c]}" for c in param_cols)
        mark = "*" if r.get("selected") else " "
        L.append(f"  {mark} {params:<34}{r['sharpe']:>9.2f}"
                 f"{r['profit_factor']:>8.2f}{int(r['trades']):>9,}"
                 f"{r['max_drawdown_pct']:>9.2f}  {r['gate1']}")
    if len(table) > top:
        L.append(f"    … {len(table) - top} more in scan_{scan['symbol']}.csv")
    L.append(f"  selection: {scan['selection']}")
    return "\n".join(L)
