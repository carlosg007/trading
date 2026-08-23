#!/usr/bin/env python3
"""
scan.py - vectorized parameter grid search, one symbol at a time.

Location:  ~/src/trading/backtest/scan.py

A strategy declares its own search space:

    PARAM_GRID = {"fast_window": [5, 10, 20], "slow_window": [30, 50, 100]}

`backtest/run.py --scan` sweeps that space per symbol and keeps the parameter
set with the best SHARPE PLATEAU **among those that clear Gate 1**. The winning
set is then re-run through the ordinary dual-version path, so the numbers that
reach a report come from the same code that produces every other number here.

Stage 2 under the Regime-Switching Incubator Charter
----------------------------------------------------
`main()` is Stage 2 of the five-stage pipeline, and the charter fixes four
things about it. All four are enforced here rather than left to how the command
was typed:

  * **Its input is Stage 1's handoff, as EXACT PAIRS.** `surviving_assets.json`
    carries `(symbol, timeframe)` survivors and the regime quadrant each one
    cleared; `resolve_targets` sweeps those pairs. `--symbols`/`--tf` are two
    independent axes, so handing the survivors over as their cross product
    sweeps configurations the screen dropped - the survivors are ragged, and a
    contract with no baseline edge getting parameters fitted to it anyway is
    the exact curve fit the screen exists to prevent.
  * **The in-sample window is 2013-01-01 .. 2022-12-31 and the holdout is not
    read.** `check_in_sample_window` refuses a window reaching 2023-01-01
    before a bar is loaded, and there is no override flag. An optimiser fits
    what it reads, so a holdout Stage 2 has swept is not a holdout, and Gate 3
    would measure retention on bars the winner was already chosen on.
  * **Nothing is pruned.** No contract, no timeframe, no quadrant and no
    parameter set is eliminated here on an aggregate metric, and no prop-firm
    rule is applied - those are CrossTrade's, against a live balance. Gate 1
    is a ranking PREFERENCE and never a filter: a grid where nothing clears it
    still produces a winner, still writes `best_params_<SYMBOL>_<TF>.json` and
    still advances. Stage 3 is what certifies; conflating the two would drop a
    contract on an in-sample number before the gates ever ran.
  * **The winner is a plateau, not a spike.** `plateau_scores` reads the Sharpe
    surface vectorbt produced and ranks each cell on
    `min(own Sharpe, mean neighbour Sharpe)`. Nothing about a market changes
    between a 20-bar mean and a 21-bar one, so a Sharpe that does is a property
    of this sample; taking the grid's single best cell is how a sweep
    manufactures an in-sample number no walk-forward reproduces. Spikes are
    labelled, never dropped.

Two files leave the stage beside the per-contract winners: `stage2_summary.json`
(the handoff `discord_reporter.py --stage 2` posts) and
`stage2_summary_matrix.csv`, both carrying every configuration the stage was
asked to optimise - including any whose sweep failed, because a shorter table
reads as a complete one.

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

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. This runs at import
# time, above the local imports below, because several modules resolve their
# BT_* variables while being imported (backtest.run's ARTIFACTS_ROOT) - loading
# the file inside main() would be too late for those and would work here, which
# is the kind of difference nobody notices until one runner silently uses the
# default path. Existing environment variables WIN: load_dotenv does not
# override them, so an explicit `BT_ARTIFACTS=... bt-run` still beats the file.
from pathlib import Path                                           # noqa: E402
from dotenv import load_dotenv                                     # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
# ---------------------------------------------------------------------------


import argparse
import ast
import gc
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.data_loader import (DEFAULT_CHUNK_YEARS,        # noqa: E402
                                  LakeSource, auto_warmup_bars,
                                  iter_temporal_chunks, peak_rss_bytes,
                                  projected_sweep_bytes, suggest_chunk_years)
from backtest.memory_guard import (DEFAULT_GUARD,             # noqa: E402
                                   MEMORY_HALT_EXIT_CODE, MemoryGuard,
                                   MemorySafetyException)
from backtest.engine import (BacktestConfig, TRADE_COLUMNS,   # noqa: E402
                             _assemble_result, _cost_arrays, _shift_to_fill,
                             apply_flat_by_close, clean_signals_ls,
                             unpack_signals)
from backtest.event_calendar import (WEEKDAY_NAMES, add_filter_args,   # noqa: E402
                                    entry_block_mask)
from backtest.pipeline import (CHARTER_IS_END, CHARTER_IS_START,  # noqa: E402
                               HOLDOUT_START)
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
# The charter's rule, and the default from 2026-08-21: rank on the SHARPE
# PLATEAU - the mean Sharpe of a combination and its immediate grid
# neighbours - rather than on the single cell's own Sharpe. A parameter set
# whose neighbours collapse is a spike, and a spike is the shape an overfit
# takes on a grid: nothing about the market changes between `ema_period=20`
# and `21`, so a Sharpe that does is a property of this sample of bars.
SELECTED_GATE1_PLATEAU = "GATE 1 PASS · best Sharpe plateau"
SELECTED_NO_GATE1_PLATEAU = ("BEST SHARPE PLATEAU · NO COMBINATION CLEARED "
                             "GATE 1")

# The two ranking rules, named so a CLI flag, a scan dict and a leaderboard all
# spell them the same way.
RANK_PLATEAU = "plateau"
RANK_SHARPE = "sharpe"

# Selections that mean "a combination cleared Gate 1 in-sample". Both spellings
# of it, because the rank changes the string and a caller testing for one of
# them would silently report every plateau-ranked sweep as a shortfall.
CLEAN_SELECTIONS = frozenset({SELECTED_GATE1, SELECTED_GATE1_PLATEAU})

# A cell is an isolated SPIKE when its neighbours keep less than this share of
# its Sharpe. 0.5 is a judgement call and it is only ever REPORTED - nothing is
# dropped for being a spike, because Stage 2 drops nothing at all. It exists so
# a winner whose surroundings fall away is labelled as one on the card and in
# the summary, rather than being read as a plateau because it happened to win.
PLATEAU_SPIKE_RATIO = 0.5

# The plateau columns, named once. `_NON_PARAM_COLUMNS` has to exclude them
# from the swept parameters and `scan_from_csv` has to carry them across a
# rebuild; two hand-written lists would drift, and the failure mode of the
# drift is a rebuild that ranks a plateau-selected table on Sharpe and then
# raises because the row it picks is not the row the table flags.
PLATEAU_COLUMNS = ("plateau_score", "neighbour_mean", "plateau_min",
                   "plateau_neighbours", "is_spike")

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
                      "gate1", "gate1_shortfalls", "selected",
                      # The plateau columns are metrics ABOUT the grid, not
                      # points in it. Left out of this tuple they would be read
                      # back by `scan_from_csv` as swept parameters and bound
                      # to the strategy, which raises on an unknown parameter
                      # name - after the sweep, with the table already written.
                      *PLATEAU_COLUMNS)


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


def _select_best_row(rows: Iterable[dict],
                     rank: str = RANK_PLATEAU) -> tuple[dict | None, str]:
    """
    The winning row and the label for HOW it won, from one rule.

    Best SHARPE PLATEAU among the combinations whose Gate 1 audit is PASS; if
    none cleared it, the best plateau overall under a `selection` string that
    says so. This lives in its own function because two callers need the
    identical answer - `scan_symbol` sweeping the grid, and `scan_from_csv`
    rebuilding a winner from a table that was written earlier. A second copy of
    the rule in the rebuild path would be free to disagree with the sweep about
    which row won, and the two would be compared by nobody.

    Gate 1 is a PREFERENCE here and never an elimination. Under the charter
    Stage 2 drops nothing - not a strategy, not a contract, not a quadrant - so
    a grid where no combination clears Gate 1 still returns a winner, still
    writes `best_params_<SYMBOL>_<TF>.json`, and still advances to Stage 3. The
    only thing that changes is the `selection` string, which says plainly that
    nothing cleared the gate.

    Sharpe is the underlying metric because it IS the risk-adjusted return,
    which is what the selection is supposed to maximise; ranking on raw return
    would pick the parameter set that took the most risk to get there. The
    PLATEAU of it is what ranks, from 2026-08-21: the mean Sharpe one step
    away, floored against the cell's own (`plateau_scores`). A grid's single
    best cell is the cell this sample's noise helped most, and picking it is
    how a sweep manufactures an in-sample Sharpe that no walk-forward
    reproduces.

    `rank=RANK_SHARPE` restores the pre-charter rule, and the fallback is
    automatic: a table with no usable `plateau_score` column - anything written
    before the column existed, rebuilt through `scan_from_csv` - is ranked on
    Sharpe and LABELLED as Sharpe-ranked. Silently ranking a spike-blind table
    under a plateau heading would be the one failure this whole function is
    supposed to prevent.

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
    if not pool:
        return None, SELECTED_NONE

    # Every pooled row has to carry a plateau score for the plateau ranking to
    # mean anything: ranking half a table on the plateau and the other half on
    # a Sharpe standing in for it would compare two different quantities under
    # one heading.
    plateaued = (rank == RANK_PLATEAU
                 and all(not pd.isna(r.get("plateau_score", float("nan")))
                         for r in pool))
    if passing:
        selection = SELECTED_GATE1_PLATEAU if plateaued else SELECTED_GATE1
    else:
        selection = (SELECTED_NO_GATE1_PLATEAU if plateaued
                     else SELECTED_NO_GATE1)

    def _rank(r: dict) -> tuple[float, float]:
        # Ties are real once risk axes are swept - a target no bar reaches and
        # `tp_atr_mult=None` produce the same trade list - and they are MORE
        # common under the plateau rule, which floors a whole neighbourhood at
        # one number. The shallower drawdown breaks them; `abs` because the
        # engine signs drawdowns negative and comparing raw would prefer the
        # deepest.
        dd = r.get("max_drawdown_pct")
        dd = float("inf") if dd is None or pd.isna(dd) else abs(float(dd))
        score = float(r["plateau_score"]) if plateaued else float(r["sharpe"])
        return (score, -dd)

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


def check_in_sample_window(start: str | None, end: str | None) -> str:
    """
    Refuse an in-sample window that reads a single bar of the Stage 3 holdout.

    Stage 2 optimises. Whatever it reads has been fitted to, so a window that
    runs into 2023 does not merely "include" the holdout - it SPENDS it, and
    every retention number Gate 3 later prints is measured against bars this
    stage already chose parameters on. Nothing downstream can detect that: the
    audit is well-formed, the ratio is plausible, and the strategy is certified
    on a comparison that was never out of sample.

    There is deliberately no override flag. Stage 1 defaults to the charter
    window and lets a deliberate re-screen through, because a screen is a
    filter and a wider one is merely a looser filter. An optimiser is not: the
    holdout only exists once, and a flag that spends it would be used.

    An omitted `--end` is refused for the same reason it is in
    `audit_gates.check_windows` - it runs to the end of the lake, which is the
    holdout, and it is the easiest way to spend one by accident.

    Returns a one-line note naming the window and whether it is the charter's.
    """
    if not start or not str(start).strip():
        raise ScanError(
            f"--start is required. Stage 2 optimises on the charter's "
            f"in-sample window, {CHARTER_IS_START} → {CHARTER_IS_END}; an "
            f"unbounded start reads whatever the lake begins with.")
    if not end or not str(end).strip():
        raise ScanError(
            f"--end is required and must not run into the holdout. Omitted, "
            f"it reads to the end of the lake — which IS the Stage 3 holdout "
            f"({HOLDOUT_START} onward). Pass --end {CHARTER_IS_END}.")
    start, end = str(start).strip(), str(end).strip()
    if start > end:
        raise ScanError(f"--start {start} is after --end {end}.")
    if end >= HOLDOUT_START:
        raise ScanError(
            f"--end {end} reaches into the Stage 3 holdout, which begins "
            f"{HOLDOUT_START}. Stage 2 fits parameters to every bar it reads, "
            f"so a holdout it has optimised over is no longer a holdout and "
            f"Gate 3's retention would be measured on bars this sweep already "
            f"chose the winner on. The charter window is "
            f"{CHARTER_IS_START} → {CHARTER_IS_END}.")
    if start >= HOLDOUT_START:                      # unreachable given the above
        raise ScanError(f"--start {start} is inside the holdout.")
    charter = (start == CHARTER_IS_START and end == CHARTER_IS_END)
    return (f"{start} → {end} · "
            + ("charter in-sample window (holdout untouched)" if charter else
               f"operator window, inside the charter's holdout boundary "
               f"({HOLDOUT_START})"))


# --------------------------------------------------------------------------
# plateau detection
#
# The sweep already scores every combination through one vectorbt call; this is
# what turns that surface into a choice. Ranking a grid on the single best cell
# picks the cell most helped by this sample's noise, and the parameter axes are
# continuous: nothing about a market changes between a 20-bar mean and a
# 21-bar one, so a Sharpe that changes sharply between them is a fact about
# these bars rather than about the edge.
# --------------------------------------------------------------------------

def axis_order(grid: dict[str, Iterable]) -> dict[str, list]:
    """
    Each axis's values in NEIGHBOUR order - which is what "adjacent" means when
    a plateau is measured.

    Numeric axes sort ascending, because `[30, 15, 20]` declared in that order
    still has 15 next to 20. A `None` in a risk axis sorts LAST rather than
    being dropped: `tp_atr_mult=None` is "no take-profit", the limit of an
    ever-wider target, so it belongs at the far end of the axis and its
    neighbour is the widest target that was tested.

    A non-numeric axis (booleans, strings) keeps its DECLARED order. There is
    no distance between `True` and `False` to sort by, and the declared order
    is the only ordering the module actually asserts.
    """
    out: dict[str, list] = {}
    for key, raw in grid.items():
        values = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
        # Bools are ints in Python, and sorting `[True, False]` numerically
        # would silently reorder a flag axis into an ordering nobody declared.
        numeric = [v for v in values
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        nulls = [v for v in values if v is None]
        if len(numeric) + len(nulls) == len(values) and numeric:
            out[key] = sorted(numeric) + nulls
        else:
            out[key] = values
    return out


def _combo_key(combo: dict) -> tuple:
    """A hashable identity for a parameter set, stable across `None`."""
    return tuple(sorted((str(k), "\0None" if v is None else v)
                        for k, v in combo.items()))


def _finite(value: Any) -> float:
    """
    A Sharpe as a number for neighbourhood arithmetic, with NaN read as 0.0.

    NaN here means the combination produced no trades. Dropping those from the
    neighbourhood would let a spike sitting next to a hole in the grid average
    out as a plateau - the hole is exactly the evidence that the surrounding
    parameter space does not trade, and it has to count. The row's own Sharpe
    stays NaN in the table and such a row can still never be selected.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


def plateau_scores(combos: list[dict], sharpes: Sequence[Any],
                   grid: dict[str, Iterable],
                   spike_ratio: float = PLATEAU_SPIKE_RATIO) -> list[dict]:
    """
    One plateau record per evaluated combination: is this a shelf or a spike?

    A neighbour is a combination differing on exactly ONE axis by one step in
    that axis's order (`axis_order`). That is the local structure of the grid
    itself - not a distance in parameter units, which would be meaningless
    across axes measured in bars and in ATR multiples.

    Per row:
      `plateau_score`      `min(own Sharpe, mean neighbour Sharpe)` - what
                           this parameter set delivers, and what it degrades
                           to one step away, whichever is worse. This is what
                           selection ranks on.
      `neighbour_mean`     mean over the neighbours alone; NaN with none.
      `plateau_min`        the worst Sharpe in the neighbourhood - what the
                           parameter set degrades to if the market moves one
                           step away from the fit.
      `plateau_neighbours` how many neighbours were actually evaluated. A
                           combination the strategy REJECTED is not in the
                           grid's evaluated set and cannot be a neighbour, so
                           an edge cell is scored on fewer of them and the
                           count says so rather than being padded with zeros.
      `is_spike`           the neighbours keep less than `spike_ratio` of this
                           cell's Sharpe. Reported, never acted on.

    The `min` is the whole point and a mean would defeat it. Averaging the cell
    with its neighbours scores a spike's NEIGHBOUR highly - it inherits the
    spike's Sharpe through the average while delivering none of it - so the
    sweep would answer an overfit by promoting the cell next to it. Taking the
    worse of the two says the only thing a plateau claim is entitled to say:
    this set works, and so does the parameter space around it.

    A grid with a single value on every axis has no neighbours anywhere, so
    every `plateau_score` equals the cell's own Sharpe and the ranking
    degenerates to the Sharpe rule. That is the correct degeneration: with
    nothing adjacent tested, there is no evidence of a plateau either way.
    """
    axes = {k: v for k, v in axis_order(grid).items() if len(v) > 1}
    index = {_combo_key(c): i for i, c in enumerate(combos)}
    scores: list[dict] = []

    for combo in combos:
        own = _finite(sharpes[index[_combo_key(combo)]])
        neighbours: list[float] = []
        for key, values in axes.items():
            if key not in combo:
                continue
            try:
                pos = next(i for i, v in enumerate(values)
                           if _same_value(v, combo[key]))
            except StopIteration:               # a --param pin outside the axis
                continue
            for step in (-1, 1):
                j = pos + step
                if not 0 <= j < len(values):
                    continue
                probe = index.get(_combo_key({**combo, key: values[j]}))
                if probe is not None:
                    neighbours.append(_finite(sharpes[probe]))

        pool = [own] + neighbours
        n_mean = (sum(neighbours) / len(neighbours)) if neighbours else float("nan")
        scores.append({
            "plateau_score": min(own, n_mean) if neighbours else own,
            "neighbour_mean": n_mean,
            "plateau_min": min(pool),
            "plateau_neighbours": len(neighbours),
            "is_spike": bool(neighbours and own > 0.0
                             and n_mean < own * float(spike_ratio)),
        })
    return scores


def _entry_block_mask(bars: pd.DataFrame,
                      cfg: BacktestConfig
                      ) -> tuple[np.ndarray | None, dict | None]:
    """
    The sweep's entry-suppression mask, or `(None, None)` when nothing is on.

    Built from the same `BacktestConfig` fields `run_backtest` reads and through
    the same `entry_block_mask`, so a column of this sweep is filtered exactly
    the way the eventual run of the winning parameters will be. A second
    implementation here would be free to disagree about the fill-bar widening
    or the session-date roll, and the disagreement would surface as a winner
    whose re-run scores differently for no visible reason.

    Returning `None` rather than an all-False mask when both filters are off is
    deliberate: it lets `_combo_signals` skip the copy entirely, and it keeps
    "no filter was configured" distinguishable from "a filter ran and blocked
    nothing" everywhere the info dict is read.
    """
    if not (cfg.news_filter or cfg.exclude_days):
        return None, None
    mask, info = entry_block_mask(
        bars["ts"],
        news_filter=cfg.news_filter,
        news_window_minutes=cfg.news_window_minutes,
        news_kinds=cfg.news_kinds,
        exclude_days=cfg.exclude_days)
    return mask, info


def _combo_signals(strategy_path: str | Path,
                   bars: pd.DataFrame,
                   base_params: dict,
                   combo: dict,
                   cfg: BacktestConfig,
                   block_mask: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray, int, int]:
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

    `block_mask` is the ENTRY suppression mask - the news window, the excluded
    weekdays, or both - already widened to the fill bar by `entry_block_mask`.
    It is applied HERE, immediately after `unpack_signals` and before
    `apply_flat_by_close`, which is exactly where `run_backtest` applies it. The
    ordering is not cosmetic: `clean_signals_ls` resolves a bar signalling both
    sides while flat by taking NEITHER, so suppressing an entry after the
    three-state machine has run would leave the OTHER side's signal resolved
    against a trade that no longer exists. Entries only, both sides, never the
    exits - a blocked exit holds a position through the session the filter was
    added to stay out of.

    Before this, the sweep ignored `cfg.news_filter` and `cfg.exclude_days`
    entirely: the flags parsed, reached the config, printed on the console and
    changed nothing. Every parameter set was selected on the unfiltered week
    and then re-run filtered, so the winner was the winner of a different
    strategy.

    Returns the four fill-bar masks plus `(entries_offered, entries_suppressed)`
    summed over both sides, so a caller can say whether the filter actually bit.
    """
    from agents.tier3_workers import load_strategy

    fn, _info = load_strategy(strategy_path, {**base_params, **combo})
    entries, exits, s_entries, s_exits = unpack_signals(fn(bars), len(bars))

    offered = suppressed = 0
    if block_mask is not None:
        # The same two lines `event_calendar.apply_entry_filters._suppress`
        # runs, on the mask that function would have built. Written out rather
        # than called because the mask depends only on the timestamps, so it is
        # built ONCE per contract instead of once per grid cell - a 1,296-cell
        # sweep would otherwise rebuild the same macro calendar 1,296 times.
        keep = ~np.asarray(block_mask, dtype=bool)
        offered = sum(int(np.asarray(side.values, dtype=bool).sum())
                      for side in (entries, s_entries))
        entries = pd.Series(np.asarray(entries.values, dtype=bool) & keep,
                            index=entries.index, name=entries.name)
        s_entries = pd.Series(np.asarray(s_entries.values, dtype=bool) & keep,
                              index=s_entries.index, name=s_entries.name)
        suppressed = offered - sum(int(np.asarray(side.values, dtype=bool).sum())
                                   for side in (entries, s_entries))

    if cfg.flat_by_close:
        entries, exits = apply_flat_by_close(bars, entries, exits,
                                             cfg.session_close_utc)
        s_entries, s_exits = apply_flat_by_close(bars, s_entries, s_exits,
                                                 cfg.session_close_utc)
    entries, exits, s_entries, s_exits = clean_signals_ls(
        entries, exits, s_entries, s_exits)
    return (_shift_to_fill(entries), _shift_to_fill(exits),
            _shift_to_fill(s_entries), _shift_to_fill(s_exits),
            offered, suppressed)


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
                      s_exi: np.ndarray | None = None,
                      report_open: bool = False):
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

    `report_open` returns `(trade_lists, open_entry_indices)` instead of the
    trade lists alone: one array per column holding the ENTRY BAR INDEX of
    every position still open when the bars ran out. Closed trades are the
    only ones with realised P&L, so those are all this function has ever
    returned - and on a contiguous run the open one at the very end is a
    position the strategy genuinely still holds, which realised nothing and is
    correctly absent.

    ON A TEMPORALLY CHUNKED RUN IT IS NOT THAT. A position open when a CHUNK
    ends is a trade the contiguous run would have closed, dropped from the
    results by the `status == 1` filter below - which is exactly the failure
    CLAUDE.md names for calendar chunking: "a position open on 31 December is
    silently dropped, which flatters results". Dropped trades are not neutral;
    they remove losers as readily as winners and the equity curve still looks
    plausible. `scan_symbol_chunked` asks for these indices so the drop is
    COUNTED and reported rather than being invisible.
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
    open_entries: list[np.ndarray] = [np.empty(0, dtype=np.int64)
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
        if report_open:
            # Read BEFORE the status filter below removes them. An open record
            # carries a real `entry_idx` and no exit, which is precisely the
            # trade a chunk boundary cut in half.
            still_open = rec[rec["status"] == 0]
            for col_i, part in still_open.groupby("col", sort=False):
                open_entries[lo + int(col_i)] = np.asarray(
                    part["entry_idx"].to_numpy(), dtype=np.int64)
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

    return (out, open_entries) if report_open else out


def session_days(frames) -> pd.DatetimeIndex:
    """
    The unique SESSION days spanned by one or more bar frames, as the daily
    index `_assemble_result` collapses an equity curve onto.

    Accepts several frames because a chunked sweep computes this per payload
    and has to union them: every risk-adjusted ratio in this repository is
    computed on DAILY closes (`backtest/engine.py`), so a day count assembled
    from a subset of the chunks would annualise the whole run by the wrong
    root. Duplicates across chunk payloads cannot arise — the payloads
    partition the history — but they are dropped anyway, because a day counted
    twice would deflate the mean return per day and nothing would raise.
    """
    if isinstance(frames, pd.DataFrame):
        frames = [frames]
    stamps = [pd.DatetimeIndex(f["ts"]).values.astype("datetime64[D]")
              for f in frames if len(f)]
    if not stamps:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.DatetimeIndex(np.unique(np.concatenate(stamps))).tz_localize("UTC")


def _finalise_scan(symbol: str,
                   cfg: BacktestConfig,
                   grid: dict[str, Iterable],
                   rank: str,
                   strat_name: str | None,
                   valid: list[dict],
                   rejected: list[dict],
                   n_combinations: int,
                   trade_lists: list[pd.DataFrame],
                   days: pd.DatetimeIndex,
                   filter_info: dict | None,
                   offered_total: int,
                   suppressed_total: int,
                   extra: dict | None = None) -> dict[str, Any]:
    """
    Turn one trade list per surviving combination into the scan result.

    Everything after the simulation — the per-column metrics, the Gate 1 audit,
    the plateau surface, the winner and the sorted table — lives here so the
    CONTIGUOUS sweep (`scan_symbol`) and the CHUNKED one
    (`scan_symbol_chunked`) cannot disagree about how a winner is chosen. Two
    copies of this would be free to rank differently, and the two would be
    compared by nobody: a chunked run exists precisely where the contiguous one
    could not be run at all, so there would be no second opinion to catch it.

    `days` is supplied rather than derived from the bars because the chunked
    path unions its payloads — see `session_days`.

    `extra` is merged into the returned dict and is how the chunked path
    attaches its `chunking` record. It cannot overwrite a key this function
    computed; a collision raises, because a scan whose `winner` came from
    somewhere other than the ranking above is the one thing no reader would
    think to check.
    """
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

    # The plateau surface, over the combinations that actually ran. Rejected
    # combinations are absent from `valid`, so an edge cell simply has fewer
    # neighbours - it is never padded with a zero, which would read as an
    # adjacent parameter set that was tested and made nothing.
    for row, score in zip(rows, plateau_scores(valid,
                                               [r["sharpe"] for r in rows],
                                               grid)):
        row.update(score)

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_metrics"}
                          for r in rows])

    best, selection = _select_best_row(rows, rank=rank)
    # What the pre-charter rule would have picked, kept for one comparison:
    # whether ranking on the plateau moved the winner at all. It is recorded,
    # never acted on.
    spike_best, _spike_sel = _select_best_row(rows, rank=RANK_SHARPE)

    winner = None
    if best is not None:
        winner = {
            "params": {k: best[k] for k in valid[0]},
            "metrics": best["_metrics"],
            "gate1": best["gate1"],
            "sharpe": best["sharpe"],
            # The winner's own plateau record. Carried onto the handoff so
            # Stage 3 certifies a parameter set knowing whether the sweep found
            # a shelf or the best cell of a noisy surface - two findings that
            # are identical in every other number on the file.
            "plateau": {k: best.get(k) for k in PLATEAU_COLUMNS},
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

    if filter_info is not None:
        # Summed across every evaluated column, and labelled as such. A per-
        # column number would be the more precise thing to report, but the
        # totals answer the question that decides whether the sweep meant
        # anything: did the filter remove entries at all, or did it print on
        # the console and cut nothing.
        filter_info = dict(filter_info)
        filter_info["entries_offered_all_columns"] = int(offered_total)
        filter_info["entries_suppressed_all_columns"] = int(suppressed_total)
        filter_info["columns"] = len(valid)

    result = {
        "symbol": symbol,
        "combinations": int(n_combinations),
        "evaluated": len(valid),
        "rejected": rejected,
        "table": table,
        "winner": winner,
        "selection": selection,
        "rank": rank,
        # The shape of the surface the winner was taken off, summarised. A
        # sweep whose every cell is a spike has found no plateau anywhere, and
        # that is a statement about the strategy's parameter sensitivity that
        # no single winning row can carry.
        "plateau": {
            "rule": ("min(own Sharpe, mean Sharpe of the grid neighbours one "
                     "step away on a single axis)"),
            "spike_ratio": PLATEAU_SPIKE_RATIO,
            "spikes": int(sum(1 for r in rows if r.get("is_spike"))),
            "columns_scored": len(rows),
            # Did ranking on the plateau change the answer? When it did, the
            # cell the old rule would have taken is named - that is the
            # evidence that the sweep declined a spike, and it is worth exactly
            # one line in the file.
            "sharpe_rank_winner": (str(dict((k, spike_best[k])
                                            for k in valid[0]))
                                   if spike_best is not None else None),
            "differs_from_sharpe_rank": bool(
                best is not None and spike_best is not None
                and best is not spike_best),
        },
        # None when neither filter was configured, which is a different
        # statement from a filter that ran and cut nothing - and the two sweeps
        # are indistinguishable from their tables alone.
        "entry_filters": filter_info,
    }

    for key, value in (extra or {}).items():
        if key in result:
            # A caller cannot overwrite the ranking's own output. The chunked
            # path attaches metadata about HOW the sweep was run; if it could
            # also replace `winner` or `table`, a scan result would carry a
            # winner that this function did not choose and every consumer -
            # the CSV, the handoff, Stage 3's locked parameters - would trust
            # it.
            raise ScanError(
                f"a scan extra tried to overwrite {key!r}, which the ranking "
                f"itself produced")
        result[key] = value
    return result


def scan_symbol(strategy_path: str | Path,
                bars: pd.DataFrame,
                symbol: str,
                cfg: BacktestConfig,
                grid: dict[str, Iterable],
                base_params: dict | None = None,
                max_cells: int = MAX_CELLS,
                strat_name: str | None = None,
                rank: str = RANK_PLATEAU) -> dict[str, Any]:
    """
    Sweep `grid` over one symbol's bars and pick a winner.

    Every combination becomes a COLUMN of one `vbt.Portfolio.from_signals`
    call, so the whole grid costs roughly one backtest rather than one per
    cell. The resulting Sharpe surface is then read for a PLATEAU: selection is
    the best `plateau_score` (`plateau_scores`) among the combinations whose
    Gate 1 audit is PASS, which is the highest Sharpe that still survives one
    step in any direction on the grid. When nothing clears Gate 1 the best
    plateau overall is returned with `selection` set to say so.

    Nothing is ever dropped for failing Gate 1. The sweep always hands back a
    winner when any combination produced a measurable Sharpe, because Stage 2's
    job under the charter is to optimise every configuration Stage 1 passed it,
    not to re-screen them: labelling how the winner got there is the
    alternative to either inventing a pass or refusing to produce a result.
    The gate audit on the eventual run will fail either way.

    Returns a dict with `table` (one row per combination, including the plateau
    columns), `winner` (`{params, metrics, gate1, plateau}` or None),
    `selection`, `rank`, `plateau` (the surface's summary), `evaluated`, and
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

    # The ENTRY suppression mask, built ONCE for the whole sweep. It reads
    # timestamps and nothing else - no price, no parameter - so it is identical
    # for every column, and rebuilding it per combination would reload the
    # macro calendar once per grid cell. `filter_info` is returned on the scan
    # so a table swept with Monday masked out is never read as one swept on the
    # whole week: the two produce different winners from the same grid, and
    # nothing in the CSV would otherwise say which had happened.
    block_mask, filter_info = _entry_block_mask(bars, cfg)

    valid: list[dict] = []
    rejected: list[dict] = []
    ent_cols: list[np.ndarray] = []
    exi_cols: list[np.ndarray] = []
    s_ent_cols: list[np.ndarray] = []
    s_exi_cols: list[np.ndarray] = []
    offered_total = suppressed_total = 0

    for combo in combos:
        try:
            e, x, se, sx, offered, suppressed = _combo_signals(
                strategy_path, bars, base_params, combo, cfg,
                block_mask=block_mask)
        except Exception as exc:                                # noqa: BLE001
            # A strategy that refuses a combination is not a failure of the
            # sweep. sma_crossover raises on fast >= slow, which is most of a
            # square grid, and dropping those silently would report a 9-cell
            # search that only ever tested 3.
            rejected.append({"params": dict(combo),
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        valid.append(dict(combo))
        offered_total += offered
        suppressed_total += suppressed
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

    days = session_days(bars)
    return _finalise_scan(
        symbol=symbol, cfg=cfg, grid=grid, rank=rank, strat_name=strat_name,
        valid=valid, rejected=rejected, n_combinations=len(combos),
        trade_lists=trade_lists, days=days, filter_info=filter_info,
        offered_total=offered_total, suppressed_total=suppressed_total)



def _span_years(bars: pd.DataFrame) -> float:
    """The calendar span of a bar frame, in years. Zero for an empty frame."""
    if len(bars) < 2:
        return 0.0
    ts = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    return float((ts[-1] - ts[0]).total_seconds()) / (365.25 * 24 * 3600.0)


def warn_if_sweep_will_not_fit(n_bars: int,
                               n_combinations: int,
                               budget_gib: float,
                               symbol: str,
                               tf: str,
                               span_years: float) -> str | None:
    """
    Print the mask projection when a contiguous sweep is about to allocate more
    than the budget, and name the `--chunk-years` that would fit.

    ADVISORY ONLY, AND THAT IS THE POINT. Chunking is an approximation of the
    contiguous sweep, so switching it on automatically would change the numbers
    a run reports without the operator having asked — and the table would carry
    no sign of it beyond a key most readers do not know to look for. Every
    other decision of this kind in the pipeline is printed and left to a human,
    and this one is more consequential than most: a sweep that OOMs is loud,
    while a sweep that quietly reports approximate metrics is not.

    Returns the message it printed, or None when the sweep fits.
    """
    projected = projected_sweep_bytes(n_bars, n_combinations)
    budget = float(budget_gib) * (2 ** 30)
    if projected <= budget:
        return None
    # `span_years` is the CALENDAR span the bars cover and is passed in rather
    # than derived from their count: bars-per-year is 252 at 1d and ~350,000 at
    # 1m, so a suggestion computed from the count alone is wrong by three
    # orders of magnitude on exactly the timeframe that needs it.
    suggestion = suggest_chunk_years(n_bars, n_combinations,
                                     max(float(span_years), 1.0 / 252.0),
                                     int(budget))
    msg = (f"  MEMORY: {symbol} {tf} — {n_bars:,} bars x {n_combinations} "
           f"combination(s) needs about "
           f"{projected / 2 ** 30:.1f} GiB of signal masks, over the "
           f"{budget_gib:.1f} GiB budget. The masks are allocated in full "
           f"before the first vectorbt call, so --max-cells does not reduce "
           f"this. Re-run with --chunk-years "
           f"{suggestion or 1} to sweep it in blocks (and read what "
           f"`chunking` records: a chunked sweep is an approximation).")
    print(msg, flush=True)
    return msg


def scan_symbol_chunked(strategy_path: str | Path,
                        source: Any,
                        symbol: str,
                        cfg: BacktestConfig,
                        grid: dict[str, Iterable],
                        base_params: dict | None = None,
                        max_cells: int = MAX_CELLS,
                        strat_name: str | None = None,
                        rank: str = RANK_PLATEAU,
                        chunk_years: int = DEFAULT_CHUNK_YEARS,
                        warmup_bars: int | str = "auto",
                        settlement_bars: int | None = None,
                        start: Any = None,
                        end: Any = None,
                        tf: str | None = None,
                        progress: bool = True,
                        guard: MemoryGuard | None = None) -> dict[str, Any]:
    """
    `scan_symbol`, sweeping one temporal chunk at a time instead of holding the
    whole history.

    THE REASON THIS EXISTS is the arithmetic in `backtest/data_loader.py`: the
    peak of a Stage 2 sweep is not the bars, it is the four stacked boolean
    masks, `4 x n_bars x n_combinations` bytes, all resident before the first
    vectorbt call. One contract of 1-minute bars over sixteen years against the
    432-cell grid `double_rsi_macd_scalp_20260823` declares is 9.0 GiB in four
    allocations. Eight 2-year chunks divide `n_bars` by eight and the same
    sweep peaks near 1.1 GiB.

    `source` is anything `iter_temporal_chunks` accepts - a frame, a parquet
    path, a `LakeSource`, or `(symbol, tf)`. Only the last two reduce the LOAD
    peak as well; a frame the caller is already holding cannot be un-held.

    WHAT IS DIFFERENT FROM THE CONTIGUOUS SWEEP, and it is not nothing:

      * **Trades are attributed by ENTRY timestamp** to exactly one chunk's
        payload, so the concatenation across chunks is a partition. A trade
        entering in the warm-up belongs to the previous chunk; one entering in
        the settlement tail belongs to the next.
      * **A trade still open when its chunk's frame ends is LOST**, because
        `vbt.Portfolio.trades` realises nothing for an open position. That is
        the calendar-chunking failure CLAUDE.md forbids, and it is not
        prevented here - it is bounded by the settlement tail and COUNTED.
        `chunking.truncated_trades` is the count and
        `chunking.truncated_columns` is how many grid cells lost at least one.
        A non-zero count means the tail is too short for this strategy's
        holding period, and the answer is a longer `settlement_bars`, not a
        smaller grid.
      * **A path-dependent strategy can resolve a boundary differently.** Its
        position state at the payload's first bar is whatever the warm-up
        produced, which is not guaranteed to be what sixteen contiguous years
        would have produced. The warm-up makes it converge; nothing makes it
        identical.

    So: a chunked sweep is an APPROXIMATION and the contiguous one is not. Use
    it where the contiguous sweep would not complete at all, and read the
    winner as a candidate to be certified contiguously at Stage 3 - which is
    what Stage 3 does anyway, with the parameters locked.

    The result dict is `scan_symbol`'s, with one extra key, `chunking`, that
    records every one of the above so a table produced this way is never read
    as one produced contiguously.

    MEMORY IS GUARDED AT TWO GRANULARITIES, and they catch different things.
    The chunk boundary is guarded by `iter_temporal_chunks` itself. The GRID
    CELL is guarded here, because that is where the allocation actually is: the
    four boolean masks are built one column at a time and `np.column_stack`ed,
    so a sweep dies part-way through building a list of 432 of them, not at the
    boundary between chunks. A guard that only fired between chunks would check
    at every point except the one where the memory goes.

    On a halt the exception carries `partial`: which chunks completed, which
    combinations bound, and how many trades each column had accumulated. That
    is what `main` writes to disk before exiting — see MEMORY_HALT_EXIT_CODE.
    """
    guard = guard or DEFAULT_GUARD
    base_params = dict(base_params or {})
    combos = expand_grid(grid)
    if not combos:
        raise ScanError(
            f"{strat_name or Path(strategy_path).stem} declares no PARAM_GRID, "
            f"so --scan has nothing to sweep. Add one, or drop --scan.")

    if isinstance(warmup_bars, str) and warmup_bars.strip().isdigit():
        warmup_bars = int(warmup_bars.strip())
    elif isinstance(warmup_bars, str) and warmup_bars.lower().strip() != "auto":
        raise ScanError(
            f"--chunk-warmup must be a whole number of bars or 'auto'; got "
            f"{warmup_bars!r}")
    if isinstance(warmup_bars, str) and warmup_bars.lower().strip() == "auto":
        # Sized from the strategy's OWN declared parameters, not from a
        # constant. Read `auto_warmup_bars` before trusting it: a strategy
        # whose slowest indicator is a module CONSTANT rather than a parameter
        # is invisible to it, which is why the number used is printed below
        # rather than only recorded.
        warmup_bars = auto_warmup_bars(params=base_params, grid=grid)
    warmup_bars = int(warmup_bars)
    settlement = int(warmup_bars if settlement_bars is None
                     else settlement_bars)

    valid: list[dict] = []
    rejected: list[dict] = []
    per_column: list[list[pd.DataFrame]] = []
    day_frames: list[pd.DataFrame] = []
    chunk_rows: list[dict] = []
    filter_info: dict | None = None
    offered_total = suppressed_total = 0
    truncated_total = 0
    truncated_columns: set[int] = set()
    rss_before = peak_rss_bytes()
    n_bars_total = 0

    def _partial() -> dict:
        """
        What the sweep managed before it was stopped.

        Deliberately NOT the trade frames themselves — writing hundreds of
        megabytes at the moment the machine is out of memory is how a graceful
        halt becomes an ungraceful one. Counts and spans are what an operator
        needs to know how far it got and where to resume.
        """
        return {
            "symbol": symbol,
            "timeframe": tf,
            "strategy": strat_name,
            "chunks_completed": list(chunk_rows),
            "combinations_declared": len(combos),
            "combinations_evaluated": len(valid),
            "bars_swept": int(n_bars_total),
            "trades_by_column": [sum(len(f) for f in parts)
                                 for parts in per_column],
            "truncated_trades": int(truncated_total),
        }

    # Driven by hand rather than with `for`, so a halt raised INSIDE the
    # generator can be caught here and given this sweep's progress before it
    # propagates. `iter_temporal_chunks` guards its own chunk boundary and
    # knows nothing about grid cells or accumulated trades, so a halt from
    # there would otherwise arrive with `partial=None` and `main` would write a
    # file recording only that the run stopped.
    chunk_iter = iter_temporal_chunks(source, chunk_years=chunk_years,
                                      warmup_bars=warmup_bars,
                                      settlement_bars=settlement,
                                      start=start, end=end,
                                      symbol=symbol, tf=tf, guard=guard)
    while True:
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            break
        except MemorySafetyException as exc:
            if exc.partial is None:
                exc.partial = _partial()
            raise
        bars = chunk.frame
        n_bars_total += chunk.payload_len
        if progress:
            print(f"  {chunk.describe()}", flush=True)

        block_mask, info = _entry_block_mask(bars, cfg)

        ent_cols: list[np.ndarray] = []
        exi_cols: list[np.ndarray] = []
        s_ent_cols: list[np.ndarray] = []
        s_exi_cols: list[np.ndarray] = []
        chunk_valid: list[dict] = []

        for cell, combo in enumerate(combos, 1):
            # THE GRID CELL, which is where the memory actually goes: four
            # boolean masks per combination, accumulated in a list before they
            # are stacked. `partial=` so a halt here carries what the sweep had
            # already done rather than only the fact that it stopped.
            guard.enforce(
                f"scan.grid_iteration[{symbol} {tf or ''} chunk "
                f"{chunk.index + 1} cell {cell}/{len(combos)}]".replace(
                    "  ", " "),
                partial=_partial())
            try:
                e, x, se, sx, offered, suppressed = _combo_signals(
                    strategy_path, bars, base_params, combo, cfg,
                    block_mask=block_mask)
            except Exception as exc:                            # noqa: BLE001
                if valid and any(_combo_key(combo) == _combo_key(v)
                                 for v in valid):
                    # A combination that bound on an earlier chunk and refuses
                    # this one is not a rejection, it is a data-dependent
                    # failure - and dropping the column here would misalign
                    # every trade list after it against `valid`. Loud, because
                    # a silently shorter column list would attribute one
                    # combination's trades to another's parameters.
                    raise ScanError(
                        f"{symbol}: combination {combo} bound on an earlier "
                        f"chunk and raised on chunk {chunk.index + 1} "
                        f"({chunk.payload_start_ts:%Y-%m-%d}): "
                        f"{type(exc).__name__}: {exc}") from exc
                if chunk.index == 0:
                    rejected.append({"params": dict(combo),
                                     "reason": f"{type(exc).__name__}: {exc}"})
                continue
            chunk_valid.append(dict(combo))
            offered_total += offered
            suppressed_total += suppressed
            ent_cols.append(e)
            exi_cols.append(x)
            s_ent_cols.append(se)
            s_exi_cols.append(sx)

        if not chunk_valid:
            raise ScanError(
                f"every one of the {len(combos)} combinations in the grid was "
                f"rejected by the strategy. First reason: "
                f"{rejected[0]['reason'] if rejected else 'unknown'}")
        if not valid:
            valid = chunk_valid
            per_column = [[] for _ in valid]
            filter_info = dict(info) if info else None
        elif len(chunk_valid) != len(valid):
            raise ScanError(
                f"{symbol}: chunk {chunk.index + 1} evaluated "
                f"{len(chunk_valid)} combinations where the first chunk "
                f"evaluated {len(valid)}. The columns would no longer line up "
                f"with the parameters they belong to.")

        ent = np.column_stack(ent_cols)
        exi = np.column_stack(exi_cols)
        s_ent = np.column_stack(s_ent_cols)
        s_exi = np.column_stack(s_exi_cols)
        del ent_cols, exi_cols, s_ent_cols, s_exi_cols

        trade_lists, open_entries = _simulate_columns(
            bars, ent, exi, symbol, cfg, max_cells=max_cells,
            s_ent=s_ent, s_exi=s_exi, report_open=True)
        del ent, exi, s_ent, s_exi

        lo_ts = chunk.payload_start_ts
        hi_ts = chunk.payload_end_ts
        pay = chunk.payload_slice
        chunk_truncated = 0
        for j, trades in enumerate(trade_lists):
            if len(trades):
                # ATTRIBUTION BY ENTRY, which is what makes the concatenation a
                # partition rather than a pile. A trade whose entry is in this
                # chunk's warm-up was already counted by the previous chunk;
                # one entering in the settlement tail is the next chunk's.
                entry = pd.DatetimeIndex(trades["entry_time"])
                keep = (entry >= lo_ts) & (entry <= hi_ts)
                if keep.any():
                    per_column[j].append(trades.loc[keep])
            if not chunk.is_last:
                cut = open_entries[j]
                if cut.size:
                    n_cut = int(((cut >= pay.start) & (cut < pay.stop)).sum())
                    if n_cut:
                        chunk_truncated += n_cut
                        truncated_columns.add(j)
        truncated_total += chunk_truncated

        day_frames.append(chunk.payload_frame[["ts"]].copy())
        chunk_rows.append({
            "index": chunk.index,
            "start": str(lo_ts),
            "end": str(hi_ts),
            "bars": int(chunk.payload_len),
            "warmup": int(chunk.warmup_len),
            "settlement": int(chunk.settlement_len),
            "warmup_short": bool(chunk.warmup_short and not chunk.is_first),
            "truncated_trades": int(chunk_truncated),
        })

        del bars, trade_lists, open_entries, block_mask
        # The request's explicit reference cleanup. `_simulate_columns` already
        # does this per COLUMN batch; this is the per-CHUNK one, and it is what
        # keeps the previous chunk's frame from staying resident behind the
        # next chunk's masks.
        gc.collect()

    if not valid:
        raise ScanError(
            f"no temporal chunk of {symbol} held any bars - nothing to sweep")

    merged: list[pd.DataFrame] = []
    for parts in per_column:
        if parts:
            merged.append(pd.concat(parts, ignore_index=True))
        else:
            merged.append(pd.DataFrame(columns=TRADE_COLUMNS))
    del per_column
    gc.collect()

    if filter_info is not None:
        filter_info = dict(filter_info)
        filter_info["entries_offered_all_columns"] = int(offered_total)
        filter_info["entries_suppressed_all_columns"] = int(suppressed_total)
        filter_info["columns"] = len(valid)

    chunking = {
        "chunk_years": int(chunk_years),
        "warmup_bars": int(warmup_bars),
        "settlement_bars": int(settlement),
        "chunks": chunk_rows,
        "bars_total": int(n_bars_total),
        "bars_largest_chunk": int(max((r["bars"] for r in chunk_rows),
                                      default=0)),
        # The number the whole exercise is for. Reported both ways so the
        # saving is legible without re-deriving it.
        "projected_bytes_contiguous": projected_sweep_bytes(n_bars_total,
                                                            len(valid)),
        "projected_bytes_chunked": projected_sweep_bytes(
            max((r["bars"] for r in chunk_rows), default=0), len(valid)),
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_rss_delta_bytes": peak_rss_bytes() - rss_before,
        # Non-zero means trades were dropped at a boundary. See the docstring:
        # the fix is a longer settlement tail, and the number must never be
        # read as noise.
        "truncated_trades": int(truncated_total),
        "truncated_columns": len(truncated_columns),
        "warmup_short_chunks": sum(1 for r in chunk_rows if r["warmup_short"]),
        "equivalence": (
            "APPROXIMATE - trades are attributed by entry to one chunk and a "
            "trade outliving its settlement tail is dropped and counted. Not "
            "comparable bar-for-bar with a contiguous sweep."),
        # What the guard did while this ran. A sweep that spent an hour
        # throttling and finished is a different result from one that never
        # noticed anything, and only this says which happened.
        "memory_guard": guard.summary(),
    }
    if progress:
        gib = 2 ** 30
        print(f"  chunked sweep: {len(chunk_rows)} chunk(s), "
              f"{n_bars_total:,} bars, {len(valid)} column(s) — "
              f"mask peak {chunking['projected_bytes_chunked'] / gib:.2f} GiB "
              f"vs {chunking['projected_bytes_contiguous'] / gib:.2f} GiB "
              f"contiguous", flush=True)
        if truncated_total:
            print(f"  WARNING: {truncated_total} trade(s) across "
                  f"{len(truncated_columns)} column(s) were still open at a "
                  f"chunk boundary and are NOT in these results. Raise "
                  f"--chunk-settlement above {settlement} bars.", flush=True)

    return _finalise_scan(
        symbol=symbol, cfg=cfg, grid=grid, rank=rank, strat_name=strat_name,
        valid=valid, rejected=rejected, n_combinations=len(combos),
        trade_lists=merged, days=session_days(day_frames),
        filter_info=filter_info, offered_total=offered_total,
        suppressed_total=suppressed_total, extra={"chunking": chunking})


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
        # The plateau columns, when the table has them. Copied rather than
        # recomputed: the surface is a property of the sweep that ran, and a
        # rebuild reads no bars. Without them `_select_best_row` would fall
        # back to the Sharpe rule and pick a different row than the one the
        # table flags as `selected` - which the disagreement check below then
        # raises on, turning every rebuild of a plateau-ranked table into an
        # error.
        for col in PLATEAU_COLUMNS:
            if col in table.columns:
                rec[col] = r[col]
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
            "plateau": ({k: best[k] for k in PLATEAU_COLUMNS if k in best}
                        or None),
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
        # Which rule this rebuild actually applied, which is not a choice - it
        # is whatever the table can support. `selection` says the same thing in
        # words; both are written because Stage 3 reads the file, not the
        # console.
        "rank": (RANK_PLATEAU if "plateau_score" in table.columns
                 else RANK_SHARPE),
        "plateau": ({"rebuilt": True,
                     "spikes": int(table["is_spike"].astype(bool).sum())
                     if "is_spike" in table.columns else None,
                     "columns_scored": len(table)}
                    if "plateau_score" in table.columns else None),
        # A rebuild cannot know what was filtered. The scan table records
        # metrics, not the mask that produced them, so this is None - "not
        # recorded" - and never False, which would assert that the sweep ran
        # unfiltered on no evidence at all.
        "entry_filters": None,
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
    has_plateau = "plateau_score" in table.columns
    L.append(f"    {'params':<{W}}{'Sharpe':>9}{'plateau':>9}{'PF':>8}"
             f"{'trades':>9}{'maxDD%':>9}  gate1")
    for _, r in head.iterrows():
        params = ", ".join(f"{c}={r[c]}" for c in param_cols)
        if len(params) > W - 1:
            params = params[:W - 2] + "…"
        mark = "*" if r.get("selected") else " "
        # `!` marks an isolated spike: this cell's Sharpe, and not the
        # parameter space around it. Printed beside the row rather than only in
        # the CSV, because the whole reason the plateau column exists is that
        # a spike and a shelf are indistinguishable from the Sharpe alone.
        spike = "!" if has_plateau and bool(r.get("is_spike")) else " "
        plateau = (f"{r['plateau_score']:>9.2f}" if has_plateau
                   else f"{'--':>9}")
        L.append(f"  {mark}{spike}{params:<{W}}{r['sharpe']:>9.2f}"
                 f"{plateau}{r['profit_factor']:>8.2f}{int(r['trades']):>9,}"
                 f"{r['max_drawdown_pct']:>9.2f}  {r['gate1']}")
    if len(table) > top:
        L.append(f"    … {len(table) - top} more in scan_{scan['symbol']}.csv")
    L.append(f"  selection: {scan['selection']}")
    surface = scan.get("plateau")
    if surface and surface.get("columns_scored"):
        spikes = surface.get("spikes")
        L.append(f"    plateau: {spikes} of {surface['columns_scored']} "
                 f"combination(s) are isolated spikes (marked !) — the "
                 f"neighbours one step away keep under "
                 f"{PLATEAU_SPIKE_RATIO:.0%} of their Sharpe. "
                 f"Nothing is dropped for it.")
        if surface.get("differs_from_sharpe_rank"):
            L.append(f"             ranking on the plateau DECLINED the "
                     f"highest-Sharpe cell "
                     f"{surface.get('sharpe_rank_winner')}.")
    info = scan.get("entry_filters")
    if info:
        from backtest.event_calendar import describe_filters
        L.append(describe_filters(info))
        L.append(f"    entries cut across all {info.get('columns', 0)} "
                 f"column(s): "
                 f"{info.get('entries_suppressed_all_columns', 0):,} of "
                 f"{info.get('entries_offered_all_columns', 0):,} — every "
                 f"combination above was selected on the FILTERED week.")
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
                      variants_all_timeframes: int | None = None,
                      entry_filters: dict | None = None,
                      stage1_pair: dict | None = None) -> list[Path]:
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
        # The entry filters the sweep ran UNDER, and where they came from.
        # Carried onto the winner rather than left in the console, because a
        # parameter set selected with Mondays masked out is not the same
        # parameter set as one selected on the whole week, and Stage 3 has to
        # certify the one that was actually chosen. `provenance` names the
        # source - Stage 1's Drop Unprofitable Days decision, an explicit
        # --exclude-days, or neither - so a certification can say whether the
        # excluded sessions were picked on the very bars being certified.
        "entry_filters": dict(entry_filters) if entry_filters else None,
        "entry_filters_audit": scan.get("entry_filters"),
        # HOW the winner was ranked, and what the surface around it looks like.
        # A parameter set chosen off a plateau and one chosen off a spike are
        # identical in every other field of this file, and they are not the
        # same claim: Stage 3 certifies one of them, and the walk-forward is
        # where the difference shows up.
        "rank": scan.get("rank"),
        "plateau": (scan["winner"] or {}).get("plateau"),
        "plateau_surface": scan.get("plateau"),
        # None for a contiguous sweep, which is the exact one. A dict here
        # means the winner was chosen from a TEMPORALLY CHUNKED sweep, whose
        # trade list is an approximation of the contiguous one - it carries the
        # chunk spans, the warm-up and settlement the boundaries were run with,
        # and `truncated_trades`, the count of trades that outlived a
        # settlement tail and are absent from the metrics above. Stage 3
        # certifies the parameters this file names, so it has to be able to see
        # that they were selected on an approximation and how good one it was.
        "chunking": scan.get("chunking"),
        # The regime scope Stage 1 screened this pair in, carried through
        # UNCHANGED. Stage 2 sweeps the whole window on purpose - masking the
        # search to a quadrant that was itself chosen as the best of four on
        # these same bars stacks a second in-sample selection under the first -
        # so this is transported, not applied. It is here because the parameter
        # set eventually reaches a live supervisor, and one that arrives with
        # no environment attached reads as a licence to trade it in all four.
        "stage1_regime": (dict(stage1_pair) if stage1_pair else None),
        # The designation, lifted to the TOP LEVEL of this file as well as
        # nested under `stage1_regime`. Stage 3 reads `optimal_regime` here
        # first, and a reader (or the CrossTrade supervisor) should not have to
        # know which stage's sub-object a certification target is buried in.
        # `optimal_regime` stays a plain STRING - the name - because that is
        # what it is everywhere else in this repo; the `Q1`..`Q4` code sits
        # beside it in `target_quadrant` rather than turning one key into
        # sometimes-a-string-sometimes-a-dict.
        "optimal_regime": (stage1_pair or {}).get("optimal_regime"),
        "target_quadrant": (stage1_pair or {}).get("quadrant"),
        # The whole scored four-quadrant table the designation beat, keyed by
        # regime: in-sample profit factor, net P&L, trade count, alpha score,
        # and - for anything ineligible - which bar it missed. Without it a
        # target quadrant on this file is a name with no evidence under it, and
        # Gate R would be certifying a choice nobody can audit.
        "regime_scores": (stage1_pair or {}).get("regime_scores") or {},
        # Positive-expectancy runners-up. Metadata for the live supervisor,
        # never a second certification target: two permitted quadrants give
        # Gate R two chances at a 1.00 holdout profit factor, which is the
        # best-of-N selection the single-quadrant rule exists to prevent.
        "secondary_regimes": (stage1_pair or {}).get("secondary_regimes") or [],
        "regime_applied_to_sweep": False,
        # The window is written whole rather than left to `start`/`end` alone,
        # because "was the holdout touched" has to be answerable from the file
        # itself. `check_in_sample_window` refuses anything reaching
        # HOLDOUT_START, so a file that exists is a file whose sweep did not.
        "in_sample_window": {
            "start": start, "end": end,
            "charter_default": bool(start == CHARTER_IS_START
                                    and end == CHARTER_IS_END),
            "charter": {"start": CHARTER_IS_START, "end": CHARTER_IS_END},
            "holdout_starts": HOLDOUT_START,
            "holdout_touched": False,
        },
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


def summary_matrix_rows(rows: list[dict], errors: list[dict]) -> list[dict]:
    """
    The Stage 2 summary matrix: one row per configuration the stage was ASKED
    to optimise, optimised and failed alike, in one flat shape.

    Errors are rows, not omissions. Stage 2 prunes nothing - every Stage 1
    survivor is entitled to an optimised parameter set and a place in this
    matrix - so a configuration whose sweep raised has to appear here saying
    OPTIMIZED=no with the reason, rather than vanishing into a shorter table
    that reads as a complete one. The count in the summary is what tells an
    operator whether the stage covered its input.

    `params` is the winner as a string; the typed version lives in
    `best_params_<SYMBOL>_<TF>.json`, which is what Stage 3 reads. A CSV round
    trip turns `tp_atr_mult=None` into an empty cell, so this column is for
    reading, never for binding.
    """
    out = [{
        "symbol": r["symbol"],
        "timeframe": r["timeframe"],
        "status": "OPTIMIZED",
        "quadrant": (r.get("stage1") or {}).get("quadrant"),
        "optimal_regime": (r.get("stage1") or {}).get("optimal_regime"),
        "stage1_version": (r.get("stage1") or {}).get("version"),
        "in_stage1": bool(r.get("in_stage1")),
        "params": (", ".join(f"{k}={v}" for k, v in (r.get("winner") or {}).items())
                   or "(no winner)"),
        "profit_factor": r.get("profit_factor"),
        "sharpe": r.get("sharpe"),
        "max_drawdown_pct": r.get("max_drawdown_pct"),
        "trades": r.get("trades"),
        "plateau_score": r.get("plateau_score"),
        "plateau_neighbours": r.get("plateau_neighbours"),
        "is_spike": r.get("is_spike"),
        # The rank this configuration was ACTUALLY selected under, which is
        # not always the one the CLI asked for: a --reuse-scan of a table
        # written before the plateau columns existed can only be ranked on
        # Sharpe. Recording the request instead would put "plateau" on a card
        # announcing a spike-blind selection.
        "rank": r.get("rank"),
        "selection": r.get("selection"),
        "variants_tested": r.get("variants_tested"),
        "exclude_days": ", ".join(r.get("exclude_days_named") or []),
        "error": "",
    } for r in rows]
    out.extend({
        "symbol": e["symbol"],
        "timeframe": e.get("timeframe"),
        "status": "ERROR",
        "quadrant": (e.get("stage1") or {}).get("quadrant"),
        "optimal_regime": (e.get("stage1") or {}).get("optimal_regime"),
        "stage1_version": (e.get("stage1") or {}).get("version"),
        "in_stage1": bool(e.get("in_stage1")),
        # Not "(no winner)". A sweep that raised produced no parameters at all,
        # which is a different statement from a grid that produced no
        # measurable Sharpe, and the two must not share a cell.
        "params": "NOT OPTIMIZED",
        "profit_factor": None, "sharpe": None, "max_drawdown_pct": None,
        "trades": None, "plateau_score": None, "plateau_neighbours": None,
        "is_spike": None, "rank": None, "selection": None,
        "variants_tested": None,
        "exclude_days": "",
        "error": e.get("error", ""),
    } for e in errors)
    return out


def write_stage2_summary(strategy: str, rows: list[dict], errors: list[dict],
                         out_dir: Path, start: str | None, end: str | None,
                         timeframes: list[str], target_source: str,
                         unscreened: list[str], rank: str,
                         grid: dict | None = None) -> list[Path]:
    """
    `stage2_summary.json` + `stage2_summary_matrix.csv` - the stage's own
    handoff, beside the per-contract `best_params` files.

    Two forms of the same rows, deliberately. The JSON goes through
    `pipeline.write_stage`, so it carries the stage number and strategy name
    that let `read_stage` refuse the wrong file - which is what
    `discord_reporter.py --stage 2` reads, and a card posted off the wrong
    strategy's sweep is exactly the artifact nobody cross-checks. The CSV is
    the same matrix for a human and a spreadsheet and is read back by nothing.

    `coverage` is the charter's own check written down: how many configurations
    the stage was asked to optimise and how many carry an optimised parameter
    set. Stage 2 eliminates nothing, so anything other than "all of them"
    is a failure of the RUN, not a screening result, and it has to be legible
    without re-reading the log.
    """
    from backtest.pipeline import (STAGE2_MATRIX_FILE, STAGE2_SUMMARY_FILE,
                                   write_stage)

    matrix = summary_matrix_rows(rows, errors)
    applied = {str(r["rank"]) for r in matrix if r.get("rank")}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / STAGE2_MATRIX_FILE
    pd.DataFrame(matrix).to_csv(csv_path, index=False)

    payload = {
        "start": start,
        "end": end,
        "in_sample_window": {
            "start": start, "end": end,
            "charter_default": bool(start == CHARTER_IS_START
                                    and end == CHARTER_IS_END),
            "charter": {"start": CHARTER_IS_START, "end": CHARTER_IS_END},
            "holdout_starts": HOLDOUT_START,
            "holdout_touched": False,
        },
        "timeframes": list(timeframes),
        "target_source": target_source,
        "unscreened_pairs": list(unscreened),
        # What was asked for, and what was applied. They differ when a table
        # cannot support the plateau rank, and the difference has to be on the
        # handoff rather than reconstructed: `discord_reporter --stage 2`
        # prints this line, and a card claiming a plateau rank over a
        # Sharpe-ranked selection is a false statement about how the parameters
        # were chosen.
        "rank_requested": rank,
        "rank": (applied.pop() if len(applied) == 1
                 else "mixed" if applied else rank),
        "grid": {k: list(v) if isinstance(v, (list, tuple)) else v
                 for k, v in (grid or {}).items()},
        # The charter's no-pruning guarantee, as data rather than as prose.
        "coverage": {
            "targets": len(matrix),
            "optimized": len(rows),
            "errors": len(errors),
            "complete": not errors,
            "rule": ("Stage 2 drops nothing: every configuration Stage 1 "
                     "promoted receives an optimised parameter set and "
                     "advances to Stage 3. A row here that is not OPTIMIZED "
                     "is a run failure, not a screening decision."),
        },
        "results": matrix,
        "matrix_csv": str(csv_path),
    }
    return [write_stage(out_dir / STAGE2_SUMMARY_FILE, 2, strategy, payload),
            csv_path]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 2/5 — optimise every configuration Stage 1 "
                    "promoted, over the charter's in-sample window "
                    f"({CHARTER_IS_START}..{CHARTER_IS_END}, holdout never "
                    "read), one independent search per (symbol, timeframe), "
                    "ranking on the Sharpe PLATEAU. Nothing is pruned: every "
                    "survivor gets best_params_<SYMBOL>_<TF>.json and "
                    "advances to Stage 3.")
    p.add_argument("--strat", required=True, help="Strategy name or path")
    p.add_argument("--symbols", default=None,
                   help="NQ, a list NQ,ES,CL, or ALL. Default: the contracts "
                        "Stage 1 promoted, each at the timeframes IT survived "
                        "at — the exact pairs from surviving_assets.json, not "
                        "their cross product.")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None,
                   help="Timeframe, or a comma-separated list. Omit it: the "
                        "timeframes come from Stage 1's surviving pairs. "
                        "Naming them here is an explicit override and sweeps "
                        "the full cross product, including pairs Stage 1 "
                        "dropped (which are flagged). Derived timeframes are "
                        "aggregated from the 1m parquet by the lake reader.")
    p.add_argument("--start", default=CHARTER_IS_START,
                   help=f"In-sample start (default {CHARTER_IS_START}, the "
                        f"charter window)")
    p.add_argument("--end", default=CHARTER_IS_END,
                   help=f"In-sample end (default {CHARTER_IS_END}). REFUSED if "
                        f"it reaches {HOLDOUT_START} or later: Stage 2 fits "
                        f"parameters to every bar it reads, so a holdout it "
                        f"has optimised over is no longer a holdout. There is "
                        f"no override.")
    p.add_argument("--select", dest="rank", default=RANK_PLATEAU,
                   choices=[RANK_PLATEAU, RANK_SHARPE],
                   help=f"How the winner is ranked. {RANK_PLATEAU} (default): "
                        f"the best Sharpe that survives one step in any "
                        f"direction on the grid — min(own Sharpe, mean "
                        f"neighbour Sharpe). {RANK_SHARPE}: the single "
                        f"highest-Sharpe cell, which is the pre-charter rule "
                        f"and picks whichever cell this sample's noise helped "
                        f"most.")
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
                        "table — and so is the resolved exclude_days, which is "
                        "written with unverified_rebuild: true.")
    p.add_argument("--ignore-stage1-exclude-days", action="store_true",
                   help="Sweep the whole week even when Stage 1's "
                        "surviving_assets.json names losing days per pair. "
                        "Use it to measure what the exclusion is actually "
                        "worth: those days were chosen on this same in-sample "
                        "window, so the only way to see whether it helped is "
                        "to sweep both ways and compare.")
    p.add_argument("--chunk-years", type=int, default=None, metavar="N",
                   help="Sweep the history in N-year temporal chunks instead "
                        "of holding it all at once. OFF by default, and that "
                        "default is deliberate: a chunked sweep is an "
                        "APPROXIMATION of the contiguous one (trades are "
                        "attributed to the chunk their ENTRY falls in, and a "
                        "trade outliving the settlement tail is dropped and "
                        "COUNTED as chunking.truncated_trades), so turning it "
                        "on silently would make two runs of the same grid "
                        "incomparable for a reason nothing on the table says. "
                        "Use it when the contiguous sweep will not fit in RAM "
                        "— the banner prints the projection that decides that. "
                        "With it on, the bars are read one chunk of years at a "
                        "time through the lake's year partitions, so the load "
                        "peak drops with the sweep peak.")
    p.add_argument("--chunk-warmup", default="auto", metavar="BARS",
                   help="Bars of history prepended to each chunk so its "
                        "indicators are not cold-started (default: auto). "
                        "`auto` sizes the window from the strategy's own "
                        "declared parameters via "
                        "backtest.data_loader.auto_warmup_bars. A RECURSIVE "
                        "indicator never forgets its seed exactly, only "
                        "exponentially: a span EMA(200) needs 1,382 bars to "
                        "get the seed under 1e-6 and a Wilder(200) needs "
                        "2,758, so the 500 that reads as generous is short by "
                        "a factor of three for a 200-bar trend filter. A "
                        "strategy whose slowest length is a module CONSTANT is "
                        "invisible to `auto` — check the number it printed "
                        "against what the module actually computes.")
    p.add_argument("--chunk-settlement", type=int, default=None, metavar="BARS",
                   help="Bars appended to each chunk so a trade opened near "
                        "the end of the chunk can still close inside it "
                        "(default: the same as --chunk-warmup). This is what "
                        "stands between temporal chunking and the failure "
                        "CLAUDE.md names for it — a position open at the "
                        "boundary is dropped from the results, and dropped "
                        "trades remove losers as readily as winners. Whatever "
                        "it is set to, the trades that still outlive it are "
                        "counted and printed rather than lost quietly.")
    p.add_argument("--memory-budget-gib", type=float, default=8.0,
                   metavar="GIB",
                   help="The mask allocation above which the banner warns and "
                        "suggests a --chunk-years (default: 8.0). Advisory "
                        "only: nothing is chunked automatically, because that "
                        "would change the numbers a run reports without the "
                        "operator asking.")
    add_filter_args(p)
    return p


def resolve_targets(stage1: dict | None,
                    symbols: list[str] | None,
                    timeframes: list[str] | None,
                    default_timeframes: list[str]
                    ) -> tuple[list[dict], str, list[str]]:
    """
    WHAT this sweep optimises: one entry per `(symbol, timeframe)`, with the
    Stage 1 scope each one carries.

    The charter's default is Stage 1's survivors, as EXACT PAIRS. That is the
    whole reason this function exists. `--symbols` and `--tf` are two
    independent axes, so the survivors can only be handed over as their cross
    product - and the survivors are ragged (NQ at 5m and 15m, GC at 15m only),
    so the product is a SUPERSET that sweeps configurations the screen dropped.
    Fitting parameters to a configuration with no baseline edge is the curve
    fit the screen exists to prevent, and once the parameters exist nothing
    downstream records that the pair was never screened.

    Three cases, and the precedence between them is the point:

      1. **Neither flag** — the exact surviving pairs. The default, and the
         only one that sweeps precisely what Stage 1 promoted.
      2. **`--symbols` alone** — those contracts, each at the timeframes IT
         survived at. A contract Stage 1 never promoted still gets swept (an
         operator naming it is making that decision) at the module's default
         timeframes, and it is FLAGGED, because a parameter set for an
         unscreened pair must not reach Stage 3 looking like a screened one.
      3. **`--tf` given** — an explicit cross product with whatever symbols
         resolved. An operator naming timeframes is overriding the handoff on
         purpose; the pairs Stage 1 did not promote are still flagged.

    Returns `(targets, source, unscreened)`. Each target is
    `{"symbol", "tf", "stage1", "in_stage1"}`; `stage1` is the survivor record
    - version, quadrant, optimal regime, kill switch - or None. `unscreened`
    names the `SYMBOL·TF` pairs that are not Stage 1 survivors, so the banner
    can say so before the sweep rather than after it.
    """
    from backtest.pipeline import stage1_pairs

    pairs = stage1_pairs(stage1)
    by_pair = {(p["symbol"], p["tf"]): p for p in pairs}

    if not symbols and not timeframes:
        targets = [{"symbol": p["symbol"], "tf": p["tf"], "stage1": p,
                    "in_stage1": True} for p in pairs]
        return targets, "stage 1 survivors · exact pairs", []

    if symbols and not timeframes:
        targets = []
        for sym in symbols:
            tfs = [p["tf"] for p in pairs if p["symbol"] == sym]
            for tf in (tfs or default_timeframes):
                targets.append({"symbol": sym, "tf": tf,
                                "stage1": by_pair.get((sym, tf)),
                                "in_stage1": (sym, tf) in by_pair})
        src = ("--symbols × the timeframes each contract survived at "
               "(module defaults where it survived at none)")
        return targets, src, [f"{t['symbol']}·{t['tf']}" for t in targets
                              if not t["in_stage1"]]

    tfs = timeframes or default_timeframes
    syms = symbols or sorted({p["symbol"] for p in pairs})
    targets = [{"symbol": sym, "tf": tf, "stage1": by_pair.get((sym, tf)),
                "in_stage1": (sym, tf) in by_pair}
               for tf in tfs for sym in syms]
    return (targets, "--symbols × --tf · explicit cross product",
            [f"{t['symbol']}·{t['tf']}" for t in targets if not t["in_stage1"]])


def resolve_exclude_days(symbol: str, tf: str,
                         cli_days: tuple[int, ...] | None,
                         stage1_map: dict[tuple[str, str], tuple[int, ...]],
                         ignore_stage1: bool = False
                         ) -> tuple[tuple[int, ...] | None, str]:
    """
    Which sessions this `(symbol, tf)` sweep masks out, and on whose authority.

    There may be none, one, or all five. The list travels whole; nothing here
    reduces it to a single day.

    Precedence, in one place so the three call sites cannot drift:

      1. **`--exclude-days` on the command line wins outright.** An operator
         who names days is making the decision themselves, and a Stage 1
         artifact quietly widening or narrowing that list would mean the run
         did not do what the command said. It applies to every pair, which is
         what a global flag means.
      2. **Otherwise Stage 1's per-pair `exclude_days`** — every weekday whose
         profit factor was below 1.00 — from the Drop Unprofitable Days
         contract. This is the automated path and needs no flag. Zero, one or
         five days is the same code path.
      3. **Otherwise nothing** - the whole week is swept.

    `--ignore-stage1-exclude-days` removes step 2 only; an explicit
    `--exclude-days` still wins, because a flag that silently disabled another
    flag would be the least discoverable behaviour available.

    Returns `(days_or_None, provenance)`. The provenance string is written onto
    `best_params_<SYMBOL>_<TF>.json` and printed, so "Monday was excluded" is
    never recorded without "and here is who decided that".
    """
    from backtest.pipeline import SURVIVORS_FILE

    if cli_days:
        return tuple(cli_days), "--exclude-days (CLI, overrides stage 1)"
    if ignore_stage1:
        return None, ("none — stage 1's losing days ignored "
                      "(--ignore-stage1-exclude-days)")
    days = stage1_map.get((symbol, tf))
    if days:
        return tuple(days), (f"stage 1 Drop Losing Days ({SURVIVORS_FILE}), "
                             f"selected IN-SAMPLE on this window")
    return None, "none"


def winners_leaderboard(rows: list[dict]) -> str:
    """
    STAGE 2 OPTIMIZED WINNERS LEADERBOARD - the table the stage ends on.

    Sorted by in-sample Sharpe descending, which is the metric the sweep
    SELECTED on; ranking the summary on anything else would show a winner
    beneath a row it beat. NaN sorts last rather than being dropped - a
    contract whose grid produced no measurable Sharpe is a finding about the
    space, and an absent row reads as a sweep that never ran.

    Every number here is IN-SAMPLE and selected from `variants_tested`
    combinations on these very bars. `EXCLUDED` names the weekdays that were
    masked out while it was selected: two rows with the same parameters and
    different exclusions were fitted to different weeks, and without the column
    the table would present them as comparable.

    `QUAD` is the quadrant Stage 1 screened the pair in, transcribed. It is not
    a mask on the sweep - Stage 2 optimises the whole window - it is the scope
    the parameters are eventually allowed to trade in, and a Stage 2 table that
    dropped it would hand a reader a winner with no environment attached.
    `PLATEAU` is what the winner degrades to one step away on the grid, with
    `!` where the cell is an isolated spike; a Sharpe and a plateau that
    disagree is the single most useful thing on this table.
    """
    from backtest.pipeline import leaderboard

    def _f(v, spec=".2f", na="n/a") -> str:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return na
        return na if f != f else format(f, spec)

    def _key(r: dict) -> tuple[int, float]:
        try:
            v = float(r.get("sharpe"))
        except (TypeError, ValueError):
            return (1, 0.0)
        return (1, 0.0) if v != v else (0, -v)

    body = []
    for r in sorted(rows, key=_key):
        plateau = _f(r.get("plateau_score"))
        if r.get("is_spike"):
            plateau += " !"
        body.append([
            r["symbol"], r["timeframe"],
            (r.get("stage1") or {}).get("quadrant") or "--",
            _f(r.get("profit_factor")),
            _f(r.get("sharpe")),
            plateau,
            _f(r.get("max_drawdown_pct"), ".2f") + "%"
            if r.get("max_drawdown_pct") is not None else "n/a",
            ", ".join(f"{k}={v}" for k, v in (r.get("winner") or {}).items())
            or "(no winner)",
            ", ".join(r.get("exclude_days_named") or []) or "none",
        ])
    return leaderboard(
        "STAGE 2 OPTIMIZED WINNERS LEADERBOARD",
        ["SYMBOL", "TF", "QUAD", "IS PF", "IS SHARPE", "PLATEAU", "MAX DD",
         "WINNING PARAMS", "EXCLUDED DAYS"],
        body, align=["<", "<", "<", ">", ">", ">", ">", "<", "<"],
        empty="no sweep completed")


def main(argv: list[str] | None = None) -> int:
    import traceback

    from agents.tier3_workers import load_strategy
    from backtest.event_calendar import filter_config_kwargs
    from backtest.pipeline import (SURVIVORS_FILE, next_step, pipeline_dir,
                                   read_stage, stage1_exclude_days,
                                   stage_banner)
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

    # The in-sample window, checked BEFORE a bar is read. Stage 2 fits
    # parameters to everything it loads, so a window that runs into the holdout
    # has spent it - and the only moment that is cheap to prevent is now.
    try:
        window_note = check_in_sample_window(args.start, args.end)
    except ScanError as e:
        print(f"ScanError: {e}", file=sys.stderr)
        return 1

    # Stage 1's handoff is the INPUT to this stage, not merely a default for
    # --symbols. It carries the exact surviving (symbol, timeframe) pairs, the
    # regime quadrant each one cleared, and any weekday exclusion attached to
    # it. It is read whether or not --symbols was given: an operator naming
    # contracts explicitly still wants each one swept in the scope Stage 1
    # screened it in. A missing file is fatal only when the pair list is what
    # was needed from it — with --symbols supplied, an absent handoff simply
    # means no scope and no exclusions, and the run says so rather than dying.
    stage1: dict = {}
    try:
        stage1 = read_stage(out_dir / SURVIVORS_FILE, 1, strat_name)
    except FileNotFoundError as e:
        if not args.symbols:
            print(f"{e}\n\nOr name the contracts explicitly with --symbols.",
                  file=sys.stderr)
            return 1
        print(f"  [i] no stage 1 handoff at {out_dir / SURVIVORS_FILE}; "
              f"sweeping --symbols with no regime scope and no inherited "
              f"exclusions.", file=sys.stderr)
    except ValueError as e:
        # Wrong stage or wrong strategy. Never tolerated, with or without
        # --symbols: optimising one strategy's contracts against another's
        # screen is a mistake nothing downstream could detect.
        print(f"ValueError: {e}", file=sys.stderr)
        return 1

    symbols_arg = (parse_symbols(args.symbols, info.get("symbols"))
                   if args.symbols else None)
    try:
        tf_arg = parse_timeframes(args.tf, None) if args.tf else None
        default_tfs = parse_timeframes(None, info.get("timeframe"))
    except ValueError as e:
        print(f"ValueError: {e}", file=sys.stderr)
        return 1

    targets, source, unscreened = resolve_targets(
        stage1, symbols_arg, tf_arg, default_tfs)
    if not targets:
        print(f"Stage 1 recorded no surviving configurations in "
              f"{out_dir / SURVIVORS_FILE}.\nThere is nothing to sweep. "
              f"That is a result about the idea, not a\nreason to sweep the "
              f"contracts it already failed on.", file=sys.stderr)
        return 1

    symbols = list(dict.fromkeys(t["symbol"] for t in targets))
    # The distinct timeframes this run covers, in the order they appear. It is
    # what decides whether an unsuffixed best_params_<SYMBOL>.json is written:
    # a run spanning two timeframes must not leave Stage 3 a file that silently
    # names one of them.
    timeframes = list(dict.fromkeys(t["tf"] for t in targets))
    grid = info.get("param_grid") or {}
    if not grid:
        print(f"{strat_name} declares no PARAM_GRID, so there is nothing to "
              f"sweep.\nAdd one to the module, or run Stage 3 on the defaults "
              f"with --param.", file=sys.stderr)
        return 1

    # The Drop Unprofitable Days map, resolved once so the banner can print
    # what the sweep is about to do rather than only reporting it afterwards.
    stage1_map = ({} if args.ignore_stage1_exclude_days
                  else stage1_exclude_days(stage1))
    cli_days = cfg_kwargs.get("exclude_days")

    cells = len(expand_grid(grid))
    total_fits = cells * len(targets)
    print(stage_banner(2, strat_name,
                       f"{len(targets)} configuration(s) · "
                       f"{len(symbols)} contract(s) × "
                       f"{len(timeframes)} timeframe(s) · "
                       f"{', '.join(timeframes)} · {args.start} → {args.end}"))
    print(f"  window     : {window_note}")
    print(f"  targets    : {', '.join(t['symbol'] + '·' + t['tf'] for t in targets)}")
    print(f"               (from {source})")
    if unscreened:
        # Named, not counted. A parameter set for a pair Stage 1 dropped is
        # still written and still advances - nothing here prunes - but it is
        # not evidence of the same thing as one for a screened pair, and the
        # only place that distinction can be made visible before the fact is
        # here.
        print(f"  [!] not stage 1 survivors: {', '.join(unscreened)}\n"
              f"      They are swept because they were asked for. Their "
              f"parameters carry no\n      regime scope, and no screen says "
              f"the contract has an edge to optimise.")
    print(f"  grid       : " + ", ".join(f"{k}={v!r}" for k, v in grid.items()))
    print(f"  grid size  : {cells:,} combination(s) per configuration")
    print(f"  rank       : {args.rank}"
          + ("  · best Sharpe that survives one step on the grid"
             if args.rank == RANK_PLATEAU
             else "  · the single highest-Sharpe cell (pre-charter rule)"))
    if args.reuse_scan:
        # No fits are run, so printing a fit count would describe a search this
        # invocation is not performing. What each file reports as
        # variants_tested is the row count of the table it was rebuilt from.
        print(f"  mode       : --reuse-scan · rebuilding the export from the "
              f"scan tables already in\n               {out_dir}. No bars are "
              f"read and no combination is re-fitted.")
    else:
        print(f"  total fits : {total_fits:,} "
              f"({cells:,} × {len(targets)} configuration(s))")
    if cli_days:
        print(f"  exclude    : {list(cli_days)} on every contract  "
              f"(--exclude-days, overrides stage 1)")
    elif stage1_map:
        named = ", ".join(f"{sym}·{tf}={list(d)}"
                          for (sym, tf), d in sorted(stage1_map.items())
                          if sym in symbols and tf in timeframes)
        print(f"  exclude    : stage 1 Drop Losing Days  "
              f"({named or 'none of the pairs being swept'})")
        if named:
            print("               [!] Those sessions were chosen on THIS "
                  "in-sample window. Every\n"
                  "                   winning Sharpe below is the best of "
                  "(combinations × this\n"
                  "                   pruning). --ignore-stage1-exclude-days "
                  "sweeps the whole week\n"
                  "                   for comparison.")
    elif args.ignore_stage1_exclude_days:
        print("  exclude    : none — stage 1's losing days ignored "
              "(--ignore-stage1-exclude-days)")
    else:
        print("  exclude    : none — the whole week is swept")
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
    # Grouped by timeframe so a multi-timeframe log reads as one block per
    # timeframe, and so `tf_dir` changes once rather than per contract. The
    # targets themselves are exact pairs, so a timeframe here carries only the
    # contracts that are actually being swept at it.
    for tf in timeframes:
        at_tf = [t for t in targets if t["tf"] == tf]
        if len(timeframes) > 1:
            print("\n" + "=" * 78)
            print(f"TIMEFRAME {tf}")
            print("=" * 78)
        for i, target in enumerate(at_tf, 1):
            sym = target["symbol"]
            scope = target.get("stage1") or {}
            print(f"\n[{i}/{len(at_tf)}] {sym} · {tf}")
            print("-" * 78)
            if scope.get("quadrant") or scope.get("optimal_regime"):
                # Transcribed from the handoff and applied to nothing. The
                # sweep runs the whole window; this is the scope the winning
                # parameters are eventually allowed to trade in, printed so a
                # reader of the log knows the search was not masked to it.
                print(f"  stage 1    : {scope.get('quadrant') or '--'} "
                      f"{scope.get('optimal_regime') or ''} · regime PF "
                      f"{scope.get('regime_pf')} over "
                      f"{scope.get('regime_trade_count')} trade(s) · version "
                      f"{scope.get('version') or '--'}  "
                      f"(scope carried, NOT applied to the sweep)")
            elif not target["in_stage1"]:
                print("  stage 1    : not a survivor — swept because it was "
                      "asked for, with no regime scope")
            excl, excl_source = resolve_exclude_days(
                sym, tf, cli_days, stage1_map,
                args.ignore_stage1_exclude_days)
            pair_kwargs = {**cfg_kwargs, "exclude_days": excl}
            filters_record = {
                # --reuse-scan reads no bars and runs no simulation, so this
                # describes what a sweep WOULD run under, not what the table in
                # front of it was swept under. Flagged rather than omitted: an
                # absent block reads as "no filter", which is a claim about the
                # original sweep that this invocation cannot make.
                "unverified_rebuild": bool(args.reuse_scan),
                "exclude_days": list(excl) if excl else [],
                "exclude_days_named": ([WEEKDAY_NAMES[d] for d in excl]
                                       if excl else []),
                "exclude_days_source": excl_source,
                "news_filter": bool(pair_kwargs.get("news_filter")),
                "news_window_minutes": pair_kwargs.get("news_window_minutes"),
                "news_kinds": (list(pair_kwargs.get("news_kinds"))
                               if pair_kwargs.get("news_kinds") else None),
            }
            if excl:
                print(f"  exclude    : {list(excl)} "
                      f"({', '.join(WEEKDAY_NAMES[d] for d in excl)}) "
                      f"— {excl_source}")
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
                    cfg = BacktestConfig(
                        initial_capital=args.capital, contracts=args.contracts,
                        slippage_ticks=args.slippage_ticks,
                        flat_by_close=args.flat_by_close,
                        notes=f"stage 2 scan {sym} {tf}", **pair_kwargs)
                    if args.chunk_years:
                        # The bars are never materialised whole: the source is
                        # a lake spec and each chunk reads only its own years
                        # through the year-partition pruning in
                        # `mdlib.lake._read_native`.
                        scan = scan_symbol_chunked(
                            path, LakeSource(symbol=sym, tf=tf), sym, cfg,
                            grid, base_params=base_params,
                            strat_name=strat_name, rank=args.rank,
                            chunk_years=args.chunk_years,
                            warmup_bars=args.chunk_warmup,
                            settlement_bars=args.chunk_settlement,
                            start=args.start, end=args.end, tf=tf)
                    else:
                        bars = load_bars(sym, tf, args.start, args.end)
                        warn_if_sweep_will_not_fit(
                            len(bars), len(expand_grid(grid)),
                            args.memory_budget_gib, sym, tf,
                            span_years=_span_years(bars))
                        scan = scan_symbol(path, bars, sym, cfg, grid,
                                           base_params=base_params,
                                           strat_name=strat_name,
                                           rank=args.rank)
                    print(format_scan_summary(scan))
                    csv = write_scan_table(scan, tf_dir)
                dest = write_best_params(
                    scan, strat_name, sym, tf, args.start, args.end,
                    base_params, out_dir, timeframes=timeframes,
                    variants_all_timeframes=scan["evaluated"] * len(timeframes),
                    entry_filters=filters_record,
                    stage1_pair=(scope or None))
                print(f"  table      → {csv}")
                for d in dest:
                    print(f"  winner     → {d}")
                wm = (scan["winner"] or {}).get("metrics") or {}
                wp = (scan["winner"] or {}).get("plateau") or {}
                rows.append({"symbol": sym, "timeframe": tf,
                             # The Stage 1 scope, travelling with the result so
                             # the leaderboard, the summary matrix and the
                             # Discord card all name the same quadrant without
                             # re-reading the handoff.
                             "stage1": scope or None,
                             "in_stage1": bool(target["in_stage1"]),
                             "selection": scan["selection"],
                             "rank": scan.get("rank"),
                             "plateau_score": wp.get("plateau_score"),
                             "plateau_neighbours": wp.get("plateau_neighbours"),
                             "is_spike": wp.get("is_spike"),
                             "trades": wm.get("trade_count"),
                             "winner": (scan["winner"] or {}).get("params"),
                             "sharpe": (scan["winner"] or {}).get("sharpe"),
                             # The winner's own in-sample profit factor and
                             # drawdown, lifted out of the metrics dict so the
                             # leaderboard reads them without re-opening the
                             # CSV. `.get` rather than indexing: a rebuild from
                             # a table written before a column existed carries
                             # only what that table held.
                             "profit_factor": wm.get("profit_factor"),
                             "max_drawdown_pct": wm.get("max_drawdown_pct"),
                             "variants_tested": scan["evaluated"],
                             "exclude_days": list(excl) if excl else [],
                             "exclude_days_named": ([WEEKDAY_NAMES[d]
                                                     for d in excl]
                                                    if excl else []),
                             "exclude_days_source": excl_source})
            except MemorySafetyException as e:
                # BEFORE the broad handler below, and that ordering is the
                # whole point. `except Exception` would swallow a memory halt
                # and move on to the NEXT configuration — straight back into
                # the allocation the halt was raised to prevent, on a machine
                # that is now no emptier. The run has to stop.
                #
                # What is saved is small on purpose: counts and spans, not
                # trade frames. Writing hundreds of megabytes at the moment the
                # box is out of memory turns a graceful halt into the ungraceful
                # one this whole module exists to avoid.
                partial_path = tf_dir / f"stage2_partial_{sym}_{tf}.json"
                try:
                    tf_dir.mkdir(parents=True, exist_ok=True)
                    partial_path.write_text(json.dumps({
                        "halted_on": "memory",
                        "message": str(e),
                        "reading": e.status,
                        "context": e.context,
                        "partial": e.partial,
                        "completed_configurations": rows,
                    }, indent=2, default=str), encoding="utf-8")
                    saved = str(partial_path)
                except Exception as write_err:            # noqa: BLE001
                    # A failed flush must not replace the memory diagnosis with
                    # an I/O one: the operator needs to know WHY the run
                    # stopped more than they need the partial file.
                    saved = f"NOT WRITTEN ({type(write_err).__name__}: "\
                            f"{write_err})"

                print(f"\n[!] MEMORY HALT · {sym} {tf}: {e}", file=sys.stderr)
                print(f"    partial results: {saved}", file=sys.stderr)
                print(f"    exiting {MEMORY_HALT_EXIT_CODE} (EX_TEMPFAIL) "
                      f"rather than 137: this process was NOT killed, it "
                      f"stopped itself. Retry with a smaller --chunk-years, a "
                      f"smaller grid, or more RAM.", file=sys.stderr)
                return MEMORY_HALT_EXIT_CODE
            except Exception as e:                                # noqa: BLE001
                # Recorded as a ROW of the summary matrix, not as an absence.
                # Stage 2 prunes nothing, so a configuration that failed to
                # sweep is a run failure that has to stay visible: a shorter
                # table reads as a complete one.
                errors.append({"symbol": sym, "timeframe": tf,
                               "stage1": scope or None,
                               "in_stage1": bool(target["in_stage1"]),
                               "error": f"{type(e).__name__}: {e}"})
                print(f"\n[!] {sym} {tf}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    W = 78
    print("\n" + "=" * W)
    print(f"STAGE 2 RESULT · {len(rows)}/{len(targets)} configuration(s) "
          f"optimised")
    print("=" * W)
    print(winners_leaderboard(rows))
    for r in rows:
        if r["selection"] not in CLEAN_SELECTIONS:
            # A note, never a removal. A configuration whose grid cleared no
            # Gate 1 combination still has an optimised parameter set, still
            # has a best_params file and still advances: Stage 2 optimises,
            # Stage 3 certifies, and collapsing the two here would drop a
            # contract on an in-sample aggregate that the charter reserves for
            # the gate audit.
            print(f"  [i] {r['symbol']} {r['timeframe']}: {r['selection']} "
                  f"— optimised and advancing anyway; Stage 3 decides.")
    for e in errors:
        print(f"  ERROR {e['symbol']:<6}{e.get('timeframe', ''):<5}{e['error']}")

    spiky = [r for r in rows if r.get("is_spike")]
    if spiky:
        print(f"\n  {len(spiky)} of {len(rows)} winner(s) are isolated SPIKES: "
              f"the parameters one step\n  away on the grid keep under "
              f"{PLATEAU_SPIKE_RATIO:.0%} of the winning Sharpe. Nothing is "
              f"dropped for it —\n  it is a warning about how much of that "
              f"Sharpe is a property of this sample.")
        for r in spiky:
            print(f"    {r['symbol']:<6}{r['timeframe']:<5}"
                  f"Sharpe {r.get('sharpe')}  plateau {r.get('plateau_score')}")

    # The handoff and the matrix, written even when every sweep failed: a
    # summary of a run that produced nothing is the summary that matters most,
    # and an absent file reads as a stage that never ran.
    try:
        written = write_stage2_summary(
            strat_name, rows, errors, out_dir, args.start, args.end,
            timeframes, source, unscreened, args.rank, grid)
        for w in written:
            print(f"\n  summary    → {w}")
    except Exception as e:                                        # noqa: BLE001
        print(f"\n  [!] the summary matrix could not be written: "
              f"{type(e).__name__}: {e}", file=sys.stderr)

    print(f"\n  coverage   : {len(rows)}/{len(targets)} configuration(s) carry "
          f"an optimised parameter set.")
    if errors:
        print("               Stage 2 drops nothing, so the shortfall is a RUN "
              "FAILURE, not a\n               screening result. Fix the "
              "errors above and re-run those pairs;\n               "
              "--reuse-scan rebuilds any export whose grid already completed.")
    else:
        print("               Nothing was pruned, filtered or eliminated: "
              "every configuration\n               advances to Stage 3.")

    pruned = [r for r in rows if r.get("exclude_days")]
    if pruned:
        print(f"\n  {len(pruned)} of {len(rows)} sweep(s) ran with weekdays "
              f"masked out. Those parameters were\n"
              f"  selected on the pruned week, so Stage 3 must certify on the "
              f"pruned week too —\n"
              f"  it reads the exclusion back out of best_params_<SYMBOL>_<TF>"
              f".json and applies it.\n"
              f"  Gate 3's holdout retention is the only evidence that the "
              f"pruning generalised.")
        for r in pruned:
            print(f"    {r['symbol']:<6}{r['timeframe']:<5}"
                  f"{r['exclude_days']}  ({r['exclude_days_source']})")

    if len(timeframes) > 1:
        print(f"\n  Sorting the rows above by Sharpe picks a timeframe as well "
              f"as a\n  parameter set. That choice is yours to make and to "
              f"record: the winner\n  of that comparison is the best of "
              f"{cells * len(timeframes):,} fits, not of {cells:,}.")

    # One Stage 3 command per timeframe, naming only the contracts that were
    # actually optimised AT that timeframe. A single command over the union of
    # symbols would certify pairs this stage never swept - `--symbols` and
    # `--tf` are two axes there as well - and Stage 3 would fail on a missing
    # best_params file rather than on anything about the strategy.
    lines = [
        "Stage 3 — certify the gates on the winning parameters. The holdout",
        f"window begins {HOLDOUT_START} and no stage before this one has read "
        f"a bar of it:",
    ]
    for tf in timeframes:
        at_tf = sorted({r["symbol"] for r in rows if r["timeframe"] == tf})
        lines += [
            "",
            f"  python3 backtest/audit_gates.py --strat {args.strat} \\",
            f"      --symbols {','.join(at_tf) or '<none>'} --tf {tf} \\",
            f"      --is-start {args.start} --is-end {args.end} \\",
            f"      --holdout-start {HOLDOUT_START} --holdout-end 2026-01-01",
        ]
    if len(timeframes) > 1:
        lines += [
            "",
            "--tf takes ONE timeframe there: Stage 3 reads "
            "best_params_<SYMBOL>_<TF>.json",
            "and certifies that timeframe. Both commands are listed because "
            "both were swept.",
        ]
    lines += [
        "",
        "Post this stage's summary to Discord (reads the handoff, recomputes",
        "nothing; --dry-run prints the payload and sends nothing):",
        "",
        f"  python3 backtest/discord_reporter.py --stage 2 "
        f"--strat {strat_name}",
    ]
    print(next_step(lines))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
