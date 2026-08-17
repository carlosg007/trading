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

Nothing here knows or cares whether a parameter is an indicator period or a
risk setting. A grid that sweeps `sl_atr_mult`, `tp_atr_mult` and `trailing`
alongside `fast_period` is swept exactly like one that does not - the strategy
module is what decides which of its parameters mean what. Two consequences of
that are worth stating, because both are silent when they go wrong:

  * `None` is a legitimate grid VALUE (`tp_atr_mult: [2.5, 5.0, None]` means
    "no take-profit" is one of the points searched). pandas turns a column of
    floats-and-None into float64-and-NaN, and `NaN == None` is False, so the
    winner has to be matched with `_same_value` below rather than `==`.
  * A wider grid is a bigger in-sample search, not a better one. Risk axes
    multiply: adding 4 stops x 5 targets x 2 trailing flags to a 48-cell
    indicator grid is 1,920 fits to one sample. `expand_grid` refuses nothing,
    but `run.py` prints the cell count before it sweeps and warns past
    `SIZE_WARN`, and `variants_tested` carries the number onto every report.

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
                             _assemble_result, _cost_arrays, _shift_to_fill,
                             apply_flat_by_close, clean_signals_ls,
                             unpack_signals)
from backtest.report import INFO, PASS, audit_acceptance_gates  # noqa: E402
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

# Grid size past which `run.py` warns before sweeping. Not a limit - refusing
# to run a grid somebody deliberately wrote would be the wrong call, and the
# count is reported either way. It is the point at which "the best of N" stops
# being a measurement and starts being a search worth arguing about: 108 cells
# over one symbol's in-sample bars is defensible, 640 is a different claim, and
# the operator should see which one they asked for before it runs.
SIZE_WARN = 200


class ScanError(RuntimeError):
    """The grid could not be swept at all."""


def _same_value(a: Any, b: Any) -> bool:
    """
    Grid-value equality that survives the round trip through a DataFrame.

    `None` is a real point in a risk grid - `tp_atr_mult: [2.5, 5.0, None]`
    searches "no take-profit" as one of its combinations. pandas stores that
    column as float64 with NaN, and both `NaN == None` and `NaN == NaN` are
    False, so plain `==` would fail to flag the winning row as selected in
    exactly the case where the winner is the no-take-profit configuration. The
    `selected` column would then be all-False and the scan CSV would show a
    sweep with no winner while `run.py` went on to use one.

    Booleans are compared with `==` deliberately rather than with `is`: pandas
    stores a bool column as np.bool_, and `np.True_ is True` is False.
    """
    a_null = a is None or (isinstance(a, float) and a != a)
    b_null = b is None or (isinstance(b, float) and b != b)
    if a_null or b_null:
        return a_null and b_null
    try:
        return bool(a == b)
    except Exception:                                           # noqa: BLE001
        return False


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


def _combo_signals(strategy_path: str | Path,
                   bars: pd.DataFrame,
                   base_params: dict,
                   combo: dict,
                   cfg: BacktestConfig) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
    """
    Fill-bar signals for one parameter combination, all four masks.

    The module is re-imported and re-bound per combination through
    `load_strategy`, so a combination the strategy rejects - `fast_window >=
    slow_window` on the SMA baseline - raises here and is recorded as REJECTED
    rather than being swept as though it were a valid point in the space.

    Everything from unpacking to the one-bar shift is the ENGINE's own code
    (`unpack_signals`, `apply_flat_by_close`, `clean_signals_ls`,
    `_shift_to_fill`), so a swept column is prepared exactly the way a run is.
    A long-only strategy comes back with two all-False short masks and the
    sweep proceeds as it always has.
    """
    from agents.tier3_workers import load_strategy

    fn, _info = load_strategy(strategy_path, {**base_params, **combo})
    entries, exits, s_entries, s_exits = unpack_signals(fn(bars), len(bars))

    if cfg.flat_by_close:
        entries, exits = apply_flat_by_close(bars, entries, exits,
                                             cfg.session_close_utc)
        s_entries, s_exits = apply_flat_by_close(bars, s_entries, s_exits,
                                                 cfg.session_close_utc)
    entries, exits, s_entries, s_exits = clean_signals_ls(
        entries, exits, s_entries, s_exits)
    return (_shift_to_fill(entries), _shift_to_fill(exits),
            _shift_to_fill(s_entries), _shift_to_fill(s_exits))


def _batch_columns(n_bars: int, n_cols: int, max_cells: int) -> list[tuple[int, int]]:
    """`[(lo, hi), ...]` column slices whose bars x columns stays under the cap."""
    per = max(1, int(max_cells // max(1, n_bars)))
    return [(lo, min(lo + per, n_cols)) for lo in range(0, n_cols, per)]


def _simulate_columns(bars: pd.DataFrame,
                      ent: np.ndarray,
                      exi: np.ndarray,
                      symbol: str,
                      cfg: BacktestConfig,
                      max_cells: int = MAX_CELLS,
                      s_ent: np.ndarray | None = None,
                      s_exi: np.ndarray | None = None) -> list[pd.DataFrame]:
    """
    One trade list per column, from batched multi-column `from_signals` calls.

    All four mask arguments are (bars x columns) boolean arrays ALREADY shifted
    to the fill bar. Slippage is a per-bar fraction of price and does not depend
    on the column, so it broadcasts; fees do depend on the column, because the
    fee fraction is quoted against the FILL price and a bar where one column
    enters and another exits has two different fills. A single shared fee array
    would charge one of those columns the wrong side's fill and the error would
    be a fraction of a tick - invisible in the totals, wrong in every one of
    them. With shorts in play the same argument decides the SIDE of the fill as
    well, which is why the short masks reach `_cost_arrays` per column rather
    than being assumed away.

    `s_ent` / `s_exi` omitted (or all False) keeps the long-only
    `direction="longonly"` call, exactly as the engine does, so sweeping a
    long-only strategy produces the numbers it always did. With shorts present
    the four-mask long/short mode is used and `direction` is not passed at all -
    vectorbtpro refuses the two together.
    """
    if vbt is None:
        raise ImportError(
            "vectorbtpro is required by backtest.scan. "
            f"Import failed with: {_VBT_IMPORT_ERROR}") from _VBT_IMPORT_ERROR

    spec = get_spec(symbol)
    px = bars["open"].to_numpy(dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    n_bars, n_cols = ent.shape

    zeros = np.zeros_like(ent)
    s_ent = zeros if s_ent is None else s_ent
    s_exi = zeros if s_exi is None else s_exi
    bidirectional = bool(s_ent.any())

    out: list[pd.DataFrame] = [pd.DataFrame(columns=TRADE_COLUMNS)
                               for _ in range(n_cols)]

    for lo, hi in _batch_columns(n_bars, n_cols, max_cells):
        cols = [f"c{j}" for j in range(lo, hi)]
        e_df = pd.DataFrame(ent[:, lo:hi], index=index, columns=cols)
        x_df = pd.DataFrame(exi[:, lo:hi], index=index, columns=cols)
        se_df = pd.DataFrame(s_ent[:, lo:hi], index=index, columns=cols)
        sx_df = pd.DataFrame(s_exi[:, lo:hi], index=index, columns=cols)
        if not e_df.to_numpy().any() and not se_df.to_numpy().any():
            continue

        # _cost_arrays is the engine's, called once per column so the fee
        # denominator uses that column's own fills. Slippage comes back
        # identical every time - it is price and tick size only - so the first
        # column's copy is kept and the rest discarded.
        slippage = None
        fees = {}
        for j, col in zip(range(lo, hi), cols):
            s, f, size = _cost_arrays(bars, ent[:, j], exi[:, j], symbol, cfg,
                                      s_ent[:, j], s_exi[:, j])
            slippage = s if slippage is None else slippage
            fees[col] = f
        price = pd.Series(px, index=index)

        common = dict(
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
            accumulate=False,
        )
        if bidirectional:
            # No `direction=`: vectorbtpro refuses it alongside short signal
            # arrays. The four masks ARE the long/short mode.
            pf = vbt.Portfolio.from_signals(**common, short_entries=se_df,
                                            short_exits=sx_df)
        else:
            pf = vbt.Portfolio.from_signals(**common, direction="longonly")

        rec = pf.trades.records
        rec = rec[rec["status"] == 1]     # closed only - an open position at
        if not rec.empty:                 # the end of the data realised nothing
            for col_i, part in rec.groupby("col", sort=False):
                entry_i = part["entry_idx"].to_numpy()
                exit_i = part["exit_idx"].to_numpy()
                entry_px = px[entry_i]
                exit_px = px[exit_i]
                # 0 = Long, 1 = Short, from vectorbt's own record - the same
                # stamp the engine makes, for the same reason: the sign of the
                # P&L cannot tell a losing long from a winning short.
                is_short = part["direction"].to_numpy() == 1
                per_contract = np.where(is_short, entry_px - exit_px,
                                        exit_px - entry_px)
                gross = per_contract * spec.multiplier * cfg.contracts
                pnl = part["pnl"].to_numpy()
                out[lo + int(col_i)] = pd.DataFrame({
                    "entry_time": index[entry_i],
                    "exit_time": index[exit_i],
                    "symbol": symbol,
                    "direction": np.where(is_short, "short", "long"),
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "gross_pnl": gross,
                    "costs": gross - pnl,
                    "pnl": pnl,
                })

        del pf, rec, common, e_df, x_df, se_df, sx_df, fees, price
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
    s_ent_cols: list[np.ndarray] = []
    s_exi_cols: list[np.ndarray] = []

    for combo in combos:
        try:
            e, x, se, sx = _combo_signals(strategy_path, bars, base_params,
                                          combo, cfg)
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
        s_ent_cols.append(se)
        s_exi_cols.append(sx)

    if not valid:
        raise ScanError(
            f"every one of the {len(combos)} combinations in the grid was "
            f"rejected by the strategy. First reason: "
            f"{rejected[0]['reason'] if rejected else 'unknown'}")

    ent = np.column_stack(ent_cols)
    exi = np.column_stack(exi_cols)
    s_ent = np.column_stack(s_ent_cols)
    s_exi = np.column_stack(s_exi_cols)
    del ent_cols, exi_cols, s_ent_cols, s_exi_cols

    trade_lists = _simulate_columns(bars, ent, exi, symbol, cfg,
                                    max_cells=max_cells,
                                    s_ent=s_ent, s_exi=s_exi)
    del ent, exi, s_ent, s_exi

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
        # INFO rows carry no threshold, so they cannot be a shortfall. Listing
        # the informational Sharpe here would put "Sharpe (not gated)" in the
        # shortfalls column of every row in the sweep, including the winner's.
        failed = [c["label"] for c in gate1["checks"]
                  if c["status"] not in (PASS, INFO)]
        rows.append({
            **combo,
            # The whole combination as one unambiguous field, alongside the
            # per-parameter columns. Those columns go through a DataFrame,
            # where `tp_atr_mult=None` becomes NaN and writes to CSV as an
            # empty cell - indistinguishable from "not recorded". This column
            # writes `None` as the word, so the no-take-profit combination is
            # readable in the file rather than only in this process. It also
            # matches the leaderboard's `params` column verbatim, so a scan row
            # and a leaderboard row can be lined up by string.
            "params": str(dict(combo)),
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
        # Sharpe first - it IS the risk-adjusted return, which is what the
        # selection is supposed to maximise, and ranking on raw return would
        # pick the parameter set that took the most risk to get there.
        #
        # Shallower drawdown breaks ties, and ties are not hypothetical once
        # risk parameters are swept: a take-profit that no bar ever reaches and
        # `tp_atr_mult=None` produce the SAME trade list and therefore the same
        # Sharpe to the last digit. Left to `max` alone the winner would be
        # whichever the grid happened to declare first. Between two identical
        # Sharpes the one that got there through a smaller drawdown is the
        # better risk-adjusted result, and `abs` is required because the engine
        # signs drawdowns negative - comparing raw would prefer the DEEPEST.
        def _rank(r: dict) -> tuple[float, float]:
            dd = r.get("max_drawdown_pct")
            dd = float("inf") if dd is None or pd.isna(dd) else abs(float(dd))
            return (float(r["sharpe"]), -dd)

        best = max(pool, key=_rank)
        winner = {
            "params": {k: best[k] for k in valid[0]},
            "metrics": best["_metrics"],
            "gate1": best["gate1"],
            "sharpe": best["sharpe"],
        }
        # `_same_value` rather than `==`: a winning `tp_atr_mult=None` comes
        # back out of the DataFrame as NaN, and NaN equals nothing.
        table["selected"] = [
            all(_same_value(row.get(k), v)
                for k, v in winner["params"].items())
            for _, row in table.iterrows()]
        if not bool(table["selected"].any()):
            # Unreachable unless a parameter value does not survive the round
            # trip through the frame at all. Loud rather than silent: a scan
            # CSV with no selected row next to a run that used a winner is a
            # discrepancy somebody will spend an afternoon on.
            raise ScanError(
                f"the winning combination {winner['params']} could not be "
                f"matched back to a row of the scan table for {symbol}. The "
                f"CSV would show a sweep with no winner while the run used "
                f"one.")
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
    # `params` is excluded with the metric columns, not listed with the
    # parameters: it is the whole combination as one string and printing it
    # beside the per-parameter columns would render every row twice.
    param_cols = [c for c in table.columns
                  if c not in ("params", "sharpe", "sortino", "profit_factor",
                               "trades", "max_drawdown_pct",
                               "total_return_pct", "total_costs", "gate1",
                               "gate1_shortfalls", "selected")]
    # Wide enough for five parameters, which is what a grid carrying risk axes
    # alongside indicator ones has. Longer than this is truncated with an
    # ellipsis rather than wrapped - the full set is in `params` in the CSV,
    # and a console table that reflows is harder to scan than a clipped one.
    W = 58
    L.append(f"    {'params':<{W}}{'Sharpe':>9}{'PF':>8}{'trades':>9}"
             f"{'maxDD%':>9}  gate1")
    for _, r in head.iterrows():
        params = ", ".join(f"{c}={r[c]}" for c in param_cols)
        if len(params) > W - 1:
            params = params[:W - 2] + "…"
        mark = "*" if r.get("selected") else " "
        L.append(f"  {mark} {params:<{W}}{r['sharpe']:>9.2f}"
                 f"{r['profit_factor']:>8.2f}{int(r['trades']):>9,}"
                 f"{r['max_drawdown_pct']:>9.2f}  {r['gate1']}")
    if len(table) > top:
        L.append(f"    … {len(table) - top} more in scan_{scan['symbol']}.csv")
    L.append(f"  selection: {scan['selection']}")
    return "\n".join(L)
