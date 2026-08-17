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

import argparse
import ast
import gc
import itertools
import json
import math
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
from backtest.event_calendar import add_filter_args                  # noqa: E402
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


# Columns of the scan table that are metrics or bookkeeping rather than swept
# parameters. Named once because two readers need the same answer to "which of
# these columns is a parameter": `format_scan_summary` when it prints the table,
# and `scan_from_csv` when it rebuilds a winner out of one.
_NON_PARAM_COLUMNS = ("params", "sharpe", "sortino", "profit_factor", "trades",
                      "max_drawdown_pct", "total_return_pct", "total_costs",
                      "gate1", "gate1_shortfalls", "selected")


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """
    Split on `sep`, ignoring separators nested in brackets or quotes.

    `fast_period=13, tp_atr_mult=None` splits on the comma; a hypothetical
    `windows=[5, 10]` must not. A grid value is almost always a scalar, so the
    nesting case is rare - which is exactly why a naive `str.split(",")` would
    survive every test anybody thought to write and then mangle one real grid.
    """
    out, buf, depth, quote = [], [], 0, ""
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth <= 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    out.append("".join(buf))
    return [t for t in (tok.strip() for tok in out) if t]


def _literal(token: str) -> Any:
    """
    A grid value as the Python object it denotes, or the bare string.

    `13` is an int, `1.0` a float, `False` a bool and `None` the absence of a
    take-profit - all four of which arrive here as text and none of which mean
    what the text means. A token that parses as nothing (`foo`) stays a string
    rather than raising: a strategy is free to take a string parameter.
    """
    try:
        return ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return token


def parse_param_dict(params_val: Any) -> dict[str, Any]:
    """
    A parameter set as a dict, whatever shape it arrived in.

    The same combination is written three different ways in this pipeline and
    all three come back through here:

      * a native `dict`, which is what `scan_symbol` holds in memory;
      * the scan CSV's `params` column, `str(dict(combo))` - a PYTHON repr,
        with `None` and `True` spelled the Python way and keys in single
        quotes, so `json.loads` rejects it;
      * a JSON object, which is what every handoff file under `pipeline/`
        holds;
      * and the flat `fast_period=13, slow_period=34` form the console table
        prints, which is what somebody copying a row out of a log will paste.

    Types are restored, not left as text. `'tp_atr_mult': None` and
    `tp_atr_mult=None` both come back as `None` rather than the four-character
    string `"None"`, because `None` is a real point in a risk grid - it means
    no take-profit was modelled - and a string there would be bound to the
    strategy as a truthy value. That is the whole failure this function
    exists to prevent, and nothing downstream would raise on it.

    An empty or missing value is an empty dict. Anything that cannot be read as
    a mapping raises `ScanError` rather than returning `{}`: a parameter set
    silently read as "no parameters" binds the module's defaults, and the run
    would be reported under the winner's name while using none of its values.
    """
    if params_val is None:
        return {}
    if isinstance(params_val, dict):
        return {str(k): v for k, v in params_val.items()}
    if isinstance(params_val, float) and params_val != params_val:  # NaN
        return {}
    if not isinstance(params_val, str):
        raise ScanError(
            f"cannot read a parameter set from {type(params_val).__name__}: "
            f"{params_val!r}")

    s = params_val.strip()
    if not s or s.lower() in ("nan", "none", "{}"):
        return {}

    if s.startswith("{"):
        # JSON first - it is the stricter grammar, so anything it accepts is
        # unambiguous. `literal_eval` then covers the Python repr the CSV
        # holds, which JSON rejects on the single quotes alone.
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(s)
            except (ValueError, SyntaxError, TypeError):
                continue
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items()}
        raise ScanError(f"could not parse a parameter dict from {s!r}")

    # `k=v, k=v`. Split on the FIRST `=` per token so a string value containing
    # one survives.
    out: dict[str, Any] = {}
    for token in _split_top_level(s):
        if "=" not in token:
            raise ScanError(
                f"could not parse a parameter dict from {s!r}: the fragment "
                f"{token!r} is neither `key=value` nor a JSON object.")
        k, _, v = token.partition("=")
        out[k.strip()] = _literal(v.strip())
    if not out:
        raise ScanError(f"could not parse a parameter dict from {s!r}")
    return out


def _gate1_status(gate1: Any) -> str | None:
    """
    Gate 1's status, from either shape it is held in.

    The scan table stores `gate1["status"]` - a bare string. An audit dict from
    `audit_acceptance_gates` holds the whole gate. Both are read here so that
    handing this the wrong one is a no-op rather than an `AttributeError` five
    hundred lines from where the value was set.
    """
    if isinstance(gate1, dict):
        return gate1.get("status")
    return gate1 if gate1 is None or isinstance(gate1, str) else str(gate1)


def _select_best_row(rows: Iterable[dict]) -> tuple[dict | None, str]:
    """
    The winning row and the label for HOW it won, from one rule.

    Highest Sharpe among the combinations whose Gate 1 audit is PASS; if none
    cleared it, the highest Sharpe overall under a `selection` string that says
    so. This lives in its own function because two callers need the identical
    answer - `scan_symbol` sweeping the grid, and `scan_from_csv` rebuilding a
    winner from a table that was written earlier. A second copy of the rule in
    the rebuild path would be free to disagree with the sweep about which row
    won, and the two would be compared by nobody.

    Sharpe ranks because it IS the risk-adjusted return, which is what the
    selection is supposed to maximise; ranking on raw return would pick the
    parameter set that took the most risk to get there.

    Shallower drawdown breaks ties, and ties are not hypothetical once risk
    parameters are swept: a take-profit that no bar ever reaches and
    `tp_atr_mult=None` produce the SAME trade list and therefore the same
    Sharpe to the last digit. Left to `max` alone the winner would be whichever
    the grid happened to declare first. Between two identical Sharpes the one
    that got there through a smaller drawdown is the better risk-adjusted
    result, and `abs` is required because the engine signs drawdowns negative -
    comparing raw would prefer the DEEPEST.
    """
    rows = list(rows)
    passing = [r for r in rows
               if r["gate1"] == PASS and not pd.isna(r["sharpe"])]
    pool = passing or [r for r in rows if not pd.isna(r["sharpe"])]
    if passing:
        selection = SELECTED_GATE1
    elif pool:
        selection = SELECTED_NO_GATE1
    else:
        return None, SELECTED_NONE

    def _rank(r: dict) -> tuple[float, float]:
        dd = r.get("max_drawdown_pct")
        dd = float("inf") if dd is None or pd.isna(dd) else abs(float(dd))
        return (float(r["sharpe"]), -dd)

    return max(pool, key=_rank), selection


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

    best, selection = _select_best_row(rows)

    winner = None
    if best is not None:
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


def scan_from_csv(path: str | Path, symbol: str | None = None) -> dict:
    """
    Rebuild a scan result from a `scan_<SYMBOL>.csv` the sweep already wrote.

    The grid is the expensive half of Stage 2 and the export is the cheap one,
    so a crash in the export should not cost the sweep. This reads the table
    back and re-derives the winner with `_select_best_row` - the SAME rule the
    live sweep applies, not a copy of it - so a rebuilt `best_params` file
    names the row the sweep itself would have named.

    What it cannot recover, and does not invent
    -------------------------------------------
    The CSV holds one row per EVALUATED combination. Combinations the strategy
    rejected were never written to it, so `rejected` comes back as 0 and
    `combinations` equals the row count. `variants_tested` is unaffected - it
    has always been the number evaluated - but a rebuilt file records where it
    came from so the two counts are not read as a fresh sweep's.

    The winner's metrics are likewise only the columns the CSV carries. Win
    rate, Calmar and the day count were never in it; they are OMITTED from
    `in_sample` rather than defaulted, because a zero win rate beside a
    profitable profit factor is a number nobody computed.
    """
    path = Path(path)
    table = pd.read_csv(path)
    if table.empty:
        raise ScanError(f"{path} holds no rows; there is no winner to rebuild.")
    for col in ("params", "sharpe", "gate1"):
        if col not in table.columns:
            raise ScanError(
                f"{path} has no {col!r} column, so it is not a scan table "
                f"written by Stage 2.")
    if symbol is None:
        symbol = path.stem[len("scan_"):] if path.stem.startswith("scan_") \
            else path.stem

    param_cols = [c for c in table.columns if c not in _NON_PARAM_COLUMNS]
    metric_cols = {"sharpe": "sharpe", "sortino": "sortino",
                   "profit_factor": "profit_factor", "trades": "trade_count",
                   "max_drawdown_pct": "max_drawdown_pct",
                   "total_return_pct": "total_return_pct",
                   "total_costs": "total_costs"}

    rows = []
    for i, r in table.iterrows():
        rec = {c: r[c] for c in param_cols}
        rec.update({
            "_index": i,
            "params": r["params"],
            "sharpe": r["sharpe"],
            "gate1": r["gate1"],
            "max_drawdown_pct": r.get("max_drawdown_pct"),
            "_metrics": {dst: r[src] for src, dst in metric_cols.items()
                         if src in table.columns and not pd.isna(r[src])},
        })
        rows.append(rec)

    best, selection = _select_best_row(rows)

    winner = None
    if best is not None:
        # The `params` STRING, not the per-parameter columns. It is the only
        # field in the file that survives the round trip intact: pandas reads a
        # `tp_atr_mult` column of floats-and-None back as float64 with NaN, and
        # binding NaN to a strategy is not the same run as binding None. The
        # columns are the fallback for a table written before that column
        # existed.
        try:
            params = parse_param_dict(best["params"])
        except ScanError:
            params = {c: (None if pd.isna(best[c]) else best[c])
                      for c in param_cols}
        winner = {
            "params": params,
            "metrics": best["_metrics"],
            "gate1": best["gate1"],
            "sharpe": best["sharpe"],
        }
        # The table already records which row the sweep chose. If re-deriving
        # it lands somewhere else, the rule and the file disagree and the
        # rebuild is not a rebuild - raise rather than write a best_params
        # naming one row beside a CSV flagging another.
        if "selected" in table.columns:
            flagged = list(table.index[table["selected"].astype(bool)])
            if flagged and best["_index"] not in flagged:
                raise ScanError(
                    f"{path} flags row(s) {flagged} as selected, but the "
                    f"selection rule picks row {best['_index']}. The CSV and "
                    f"the rule disagree about which combination won; the "
                    f"table was not written by this version of the scanner.")

    return {
        "symbol": symbol,
        "combinations": len(table),
        "evaluated": len(table),
        "rejected": [],
        "table": table,
        "winner": winner,
        "selection": selection,
        "rebuilt_from": str(path),
    }


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
    param_cols = [c for c in table.columns if c not in _NON_PARAM_COLUMNS]
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


# --------------------------------------------------------------------------
# STAGE 2 of 5 - the dedicated sweep CLI
# --------------------------------------------------------------------------
# Everything above is the library `backtest/run.py --scan` calls. Everything
# below turns it into a stage that can be run on its own and hands Stage 3 a
# file: one `best_params_<SYMBOL>.json` per contract, holding the winning
# combination and the evidence for how it was chosen.
#
# The stage adds nothing to the search itself. Same `scan_symbol`, same Gate-1
# selection rule, same tie-break on the shallower drawdown - a second sweep
# implementation living in a CLI would be free to disagree with the one
# `--scan` uses, and the two would be compared by nobody.
def write_best_params(scan: dict, strategy: str, symbol: str, tf: str,
                      start: str | None, end: str | None,
                      base_params: dict, out_dir: Path,
                      timeframes: list[str] | None = None,
                      variants_all_timeframes: int | None = None) -> list[Path]:
    """
    `best_params_<SYMBOL>.json` - the winner, and what it was chosen from.

    `params` is the FULL effective set the winning signals were bound to, not
    just the swept axes: Stage 3 binds a run from this file, and a file holding
    only the three parameters that were swept would silently fall back to the
    module's defaults for everything else. The two runs would differ in a way
    neither report could show.

    `variants_tested` is written beside it and is not optional. The winning
    Sharpe is the best of N fits to one sample of bars; Stage 3 carries the
    number onto its audit, and a certification that cannot say N is not a
    certification.

    Naming, and why there are sometimes two files
    ---------------------------------------------
    Every sweep writes `best_params_<SYMBOL>_<TF>.json`. A single-timeframe
    run ALSO writes the unsuffixed `best_params_<SYMBOL>.json`, which is what
    Stage 3 falls back to.

    A MULTI-timeframe run deliberately does not write the unsuffixed file.
    Writing it would mean picking a timeframe, and picking the highest-Sharpe
    timeframe is a second selection layer stacked on the parameter sweep: the
    reported number becomes the best of (cells x timeframes) while
    `variants_tested` still says cells. Stage 3 is told to name a timeframe
    instead. `variants_tested_all_timeframes` records the real size of the
    search either way, so the number is available to anyone who does make that
    choice by hand.
    """
    from backtest.pipeline import BEST_PARAMS_FILE, write_stage

    winner = scan.get("winner")
    # `parse_param_dict` rather than `dict(...)`: the winner reaches here as a
    # native dict from `scan_symbol` and as the CSV's `params` string from
    # `scan_from_csv`, and `{**base_params, **"fast_period=13, ..."}` raises a
    # TypeError that names neither the file nor the parameter set.
    swept = parse_param_dict(winner["params"]) if winner else {}
    payload = {
        "symbol": symbol,
        "timeframe": tf,
        "start": start,
        "end": end,
        "params": ({**base_params, **swept} if winner else dict(base_params)),
        "swept_params": dict(swept) if winner else None,
        "base_params": dict(base_params),
        "variants_tested": int(scan["evaluated"]),
        "combinations": int(scan["combinations"]),
        "rejected": len(scan.get("rejected") or []),
        "selection": scan["selection"],
        # The winner's IN-SAMPLE metrics. Recorded so Stage 3 can be compared
        # against what the sweep believed it had found; they are not evidence
        # of anything on their own, having been selected on these very bars.
        "in_sample": _jsonable_metrics(winner["metrics"]) if winner else None,
        # `winner["gate1"]` is the STATUS STRING the scan table carries, not
        # the gate dict `audit_acceptance_gates` returns - `scan_symbol` stores
        # `gate1["status"]` on the row and the winner copies the row's value.
        # Reading it as a dict raised `AttributeError: 'str' object has no
        # attribute 'get'` here, AFTER the whole grid had been swept and the
        # CSV written: six completed sweeps reported as six errors with no
        # best_params file to show for them. The dict form is still accepted so
        # a caller holding the full gate is not a second crash.
        "gate1_in_sample": _gate1_status(winner["gate1"]) if winner else None,
        "timeframes_searched": list(timeframes or [tf]),
        # cells x timeframes. The honest N for anything selected by comparing
        # timeframes against each other, which `variants_tested` alone
        # understates by a factor of len(timeframes).
        "variants_tested_all_timeframes": (
            variants_all_timeframes
            if variants_all_timeframes is not None else int(scan["evaluated"])),
    }
    if winner is None:
        payload["warning"] = (
            "no combination produced a measurable Sharpe; `params` falls back "
            "to the base parameters and nothing was selected")

    # Provenance, written rather than omitted. A file rebuilt from a CSV counts
    # only the combinations that CSV holds - anything the strategy rejected was
    # never written to it - so `combinations` and `rejected` are floors here,
    # and `in_sample` carries only the columns the table had. Stage 3 is
    # entitled to know which of those it is reading.
    if scan.get("rebuilt_from"):
        payload["rebuilt_from"] = str(scan["rebuilt_from"])
        payload["rebuilt_note"] = (
            "regenerated from an existing scan table, not from a fresh sweep. "
            "`variants_tested` is the row count of that table (combinations "
            "the strategy rejected were never written to it), and `in_sample` "
            "holds only the metrics the table carried.")

    out_dir = Path(out_dir)
    written = [write_stage(
        out_dir / BEST_PARAMS_FILE.format(symbol=f"{symbol}_{tf}"),
        2, strategy, payload)]
    if not timeframes or len(timeframes) == 1:
        written.append(write_stage(
            out_dir / BEST_PARAMS_FILE.format(symbol=symbol), 2, strategy,
            payload))
    return written


def _jsonable_metrics(metrics: dict | None) -> dict | None:
    """The scalar metrics only - no trade frame, no equity series."""
    if not metrics:
        return None
    keep = ("sharpe", "sortino", "calmar", "profit_factor", "win_rate",
            "trade_count", "max_drawdown_pct", "total_return_pct",
            "annualized_return_pct", "total_pnl", "total_costs", "n_days")
    out = {}
    for k in keep:
        v = metrics.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            f = float(v)
            out[k] = None if (math.isnan(f) or math.isinf(f)) else f
        elif v is not None and not isinstance(v, (pd.DataFrame, pd.Series)):
            out[k] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 2/5 — sweep a strategy's PARAM_GRID over the "
                    "IN-SAMPLE window, one independent search per contract, "
                    "and write the winner to best_params_<SYMBOL>.json.")
    p.add_argument("--strat", required=True, help="Strategy name or path")
    p.add_argument("--symbols", default=None,
                   help="NQ, a list NQ,ES,CL, or ALL. Default: the survivors "
                        "Stage 1 wrote to surviving_assets.json.")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None,
                   help="Timeframe, or a comma-separated list: "
                        "'--tf 1m,5m,15m,30m' sweeps the grid independently at "
                        "each. Derived timeframes are aggregated from the 1m "
                        "parquet by the lake reader.")
    p.add_argument("--start", default="2013-01-01",
                   help="In-sample start (default 2013-01-01)")
    p.add_argument("--end", default="2022-12-31",
                   help="In-sample end (default 2022-12-31). It must NOT run "
                        "into the Stage 3 holdout — a window that overlaps the "
                        "holdout has spent it before Gate 3 is evaluated.")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="Fix a parameter OUTSIDE the sweep")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--flat-by-close", action="store_true")
    p.add_argument("--out-dir", default=None,
                   help="Override <BT_ARTIFACTS>/pipeline/<strategy>/")
    p.add_argument("--reuse-scan", action="store_true",
                   help="Do not sweep. Rebuild best_params_<SYMBOL>_<TF>.json "
                        "from the scan_<SYMBOL>.csv files already in the "
                        "artifact directory, re-deriving the winner with the "
                        "same selection rule the sweep uses. For recovering "
                        "the export after a completed grid, not for a rerun: "
                        "it reads no bars, so --start/--end/--param are "
                        "recorded from the CLI and not verified against the "
                        "table.")
    add_filter_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    import traceback

    from agents.tier3_workers import load_strategy
    from backtest.event_calendar import filter_config_kwargs
    from backtest.pipeline import (SURVIVORS_FILE, next_step, pipeline_dir,
                                   read_stage, stage_banner)
    from backtest.run import (load_bars, parse_param, parse_symbols,
                              parse_timeframes, resolve_strategy)

    args = build_parser().parse_args(argv)
    path = resolve_strategy(args.strat)
    strat_name = path.parent.name if path.stem == "strat" else path.stem
    base_params = dict(parse_param(p) for p in args.param)

    try:
        _fn, info = load_strategy(path, base_params)
        cfg_kwargs = filter_config_kwargs(args)
    except Exception as e:                                        # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    out_dir = pipeline_dir(strat_name, args.out_dir, create=True)

    # Symbols default to Stage 1's survivors. Sweeping a contract Stage 1
    # dropped is not forbidden - an operator naming it explicitly gets it - but
    # it should not happen by default: the parameters that rescue a contract
    # with no edge at default settings are the definition of a curve fit.
    source = "--symbols"
    if args.symbols:
        symbols = parse_symbols(args.symbols, info.get("symbols"))
    else:
        try:
            stage1 = read_stage(out_dir / SURVIVORS_FILE, 1, strat_name)
        except FileNotFoundError as e:
            print(f"{e}\n\nOr name the contracts explicitly with --symbols.",
                  file=sys.stderr)
            return 1
        symbols = list(stage1.get("surviving") or [])
        source = f"stage 1 survivors ({SURVIVORS_FILE})"
        if not symbols:
            print(f"Stage 1 recorded no surviving contracts in "
                  f"{out_dir / SURVIVORS_FILE}.\nThere is nothing to sweep. "
                  f"That is a result about the idea, not a\nreason to sweep "
                  f"the contracts it already failed on.", file=sys.stderr)
            return 1

    try:
        timeframes = parse_timeframes(args.tf, info.get("timeframe"))
    except ValueError as e:
        print(f"ValueError: {e}", file=sys.stderr)
        return 1
    grid = info.get("param_grid") or {}
    if not grid:
        print(f"{strat_name} declares no PARAM_GRID, so there is nothing to "
              f"sweep.\nAdd one to the module, or run Stage 3 on the defaults "
              f"with --param.", file=sys.stderr)
        return 1

    cells = len(expand_grid(grid))
    total_fits = cells * len(symbols) * len(timeframes)
    print(stage_banner(2, strat_name,
                       f"{len(symbols)} contract(s) × {len(timeframes)} "
                       f"timeframe(s) · {', '.join(timeframes)} · "
                       f"{args.start} → {args.end}"))
    print(f"  symbols    : {', '.join(symbols)}  (from {source})")
    print(f"  grid       : " + ", ".join(f"{k}={v!r}" for k, v in grid.items()))
    print(f"  grid size  : {cells:,} combination(s) per contract per timeframe")
    if args.reuse_scan:
        # No fits are run, so printing a fit count would describe a search this
        # invocation is not performing. What each file reports as
        # variants_tested is the row count of the table it was rebuilt from.
        print(f"  mode       : --reuse-scan · rebuilding the export from the "
              f"scan tables already in\n               {out_dir}. No bars are "
              f"read and no combination is re-fitted.")
    else:
        print(f"  total fits : {total_fits:,} "
              f"({cells:,} × {len(symbols)} symbol(s) × {len(timeframes)} tf)")
    if cells > SIZE_WARN and not args.reuse_scan:
        print(f"\n  [!] {cells:,} combinations is a large in-sample search. "
              f"Every winning Sharpe\n      below is the best of {cells:,} "
              f"fits to one sample of bars, and it has to be\n      read that "
              f"way. The count travels to Stage 3 as variants_tested.\n",
              file=sys.stderr, flush=True)
    if len(timeframes) > 1:
        print(f"  [!] Comparing the {len(timeframes)} timeframes against each "
              f"other afterwards is a\n      SECOND selection layer. Anything "
              f"chosen that way is the best of\n      {cells * len(timeframes):,}, "
              f"not the best of {cells:,} — recorded in each file as\n"
              f"      variants_tested_all_timeframes. No unsuffixed "
              f"best_params_<SYMBOL>.json\n      is written, so Stage 3 has to "
              f"be told which timeframe you mean.\n",
              file=sys.stderr, flush=True)

    rows, errors = [], []
    for tf in timeframes:
        if len(timeframes) > 1:
            print("\n" + "=" * 78)
            print(f"TIMEFRAME {tf}")
            print("=" * 78)
        for i, sym in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] {sym} · {tf}")
            print("-" * 78)
            try:
                tf_dir = out_dir / tf if len(timeframes) > 1 else out_dir
                if args.reuse_scan:
                    csv = tf_dir / f"scan_{sym}.csv"
                    if not csv.exists():
                        raise FileNotFoundError(
                            f"{csv} does not exist. --reuse-scan rebuilds the "
                            f"export from a sweep that already ran; there is "
                            f"no table here to rebuild from.")
                    scan = scan_from_csv(csv, sym)
                    print(f"  reused: {len(scan['table']):,} evaluated "
                          f"combination(s) from {csv}")
                    print(f"  selection: {scan['selection']}")
                else:
                    bars = load_bars(sym, tf, args.start, args.end)
                    cfg = BacktestConfig(
                        initial_capital=args.capital, contracts=args.contracts,
                        slippage_ticks=args.slippage_ticks,
                        flat_by_close=args.flat_by_close,
                        notes=f"stage 2 scan {sym} {tf}", **cfg_kwargs)
                    scan = scan_symbol(path, bars, sym, cfg, grid,
                                       base_params=base_params,
                                       strat_name=strat_name)
                    print(format_scan_summary(scan))
                    csv = write_scan_table(scan, tf_dir)
                dest = write_best_params(
                    scan, strat_name, sym, tf, args.start, args.end,
                    base_params, out_dir, timeframes=timeframes,
                    variants_all_timeframes=scan["evaluated"] * len(timeframes))
                print(f"  table      → {csv}")
                for d in dest:
                    print(f"  winner     → {d}")
                rows.append({"symbol": sym, "timeframe": tf,
                             "selection": scan["selection"],
                             "winner": (scan["winner"] or {}).get("params"),
                             "sharpe": (scan["winner"] or {}).get("sharpe"),
                             "variants_tested": scan["evaluated"]})
            except Exception as e:                                # noqa: BLE001
                errors.append({"symbol": sym, "timeframe": tf,
                               "error": f"{type(e).__name__}: {e}"})
                print(f"\n[!] {sym} {tf}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    W = 78
    print("\n" + "=" * W)
    print(f"STAGE 2 RESULT · {len(rows)}/{len(symbols) * len(timeframes)} "
          f"sweep(s) completed")
    print("=" * W)
    for r in rows:
        sharpe = r["sharpe"]
        print(f"  {r['symbol']:<6}{r['timeframe']:<5}Sharpe "
              f"{(sharpe if sharpe is not None else float('nan')):>6.2f}"
              f"  best of {r['variants_tested']:>5,}   {r['winner']}")
        if r["selection"] != SELECTED_GATE1:
            print(f"         {r['selection']}")
    for e in errors:
        print(f"  ERROR {e['symbol']:<6}{e.get('timeframe', ''):<5}{e['error']}")

    if len(timeframes) > 1:
        print(f"\n  Sorting the rows above by Sharpe picks a timeframe as well "
              f"as a\n  parameter set. That choice is yours to make and to "
              f"record: the winner\n  of that comparison is the best of "
              f"{cells * len(timeframes):,} fits, not of {cells:,}.")

    best_sym = rows[0]["symbol"] if rows else "<none>"
    print(next_step([
        "Stage 3 — certify the gates on the winning parameters. The holdout",
        "window must NOT overlap the in-sample window above:",
        "",
        f"  python3 backtest/audit_gates.py --strat {args.strat} \\",
        f"      --symbols {','.join(sorted({r['symbol'] for r in rows})) or '<none>'} "
        f"--tf {timeframes[0]} \\",
        f"      --is-start {args.start} --is-end {args.end} \\",
        "      --holdout-start 2023-01-01 --holdout-end 2026-01-01",
    ] + ([] if len(timeframes) == 1 else [
        "",
        f"--tf takes ONE timeframe there: stage 3 reads "
        f"best_params_<SYMBOL>_<TF>.json",
        f"and certifies that timeframe. Repeat it per timeframe you want "
        f"certified.",
        f"(shown above with {timeframes[0]}; {best_sym} also has "
        f"{', '.join(timeframes[1:])})",
    ])))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
