#!/usr/bin/env python3
"""
run.py - the multi-asset batch runner.

Location:  ~/src/trading/backtest/run.py

    python3 backtest/run.py --strat sma_crossover --symbols NQ --tf 1d
    python3 backtest/run.py --strat sma_crossover --symbols NQ,ES,CL --tf 15m
    python3 backtest/run.py --strat sma_crossover --symbols ALL --scan --bg
    bt-run --strat sma_crossover --symbols NQ,ES --tf 1d --ml

Each symbol is a SEPARATE backtest
----------------------------------
The loop below runs one contract at a time and pools nothing. There is no
blended portfolio anywhere in this file, and that is a correctness constraint
rather than a stylistic one:

  * `mdlib.lake.get_bars` returns rows sorted by `(ts, symbol)`, so a
    concatenated frame INTERLEAVES instruments. A `close.rolling(200).mean()`
    over that averages 27 different contracts into every window. The signals
    come out the right length and the right dtype, nothing raises, and the
    equity curve looks plausible - it produced 608,079 trades where the correct
    per-symbol signals give 86,035. Bars are read through `iter_bars`, one
    symbol at a time, precisely so that frame is never built.
  * P&L is contract-specific. `backtest/specs.py` supplies the multiplier, the
    tick size and the commission per symbol, and `_cost_arrays` looks them up
    per simulation. A pooled run would have to pick one multiplier, and a wrong
    multiplier silently rescales every P&L figure while the backtest still
    looks fine.
  * A 27-symbol equity curve answers a portfolio question. This runner answers
    a per-market one: does the edge exist on THIS contract. Those are different
    questions and the second has to be settled first.

Cross-sectional work still matters - a daily strategy on ES alone over 16 years
is 100-200 trades, too thin to separate skill from luck. The answer is running
the same strategy across many symbols and reading the leaderboard, which is
what this does. It is not the same thing as merging their trades.

The five documented steps, per symbol
-------------------------------------
    1. console scorecard and gate audit - Version A
    2. standalone HTML report with the trade inspector - Version A
    3. console scorecard and gate audit - Version B      (--ml)
    4. standalone HTML report with the trade inspector - Version B  (--ml)
    5. the four-choice menu

Steps 1-4 happen here, per symbol. Step 5 is printed once at the end and stops:
nothing is promoted without a human, and this script has no path that promotes
anything.

Version B is OFF by default (`--ml` turns it on). The classifier refits once
per completed trade, and 27 symbols of that is hours rather than minutes. A
skipped Version B is reported as NOT RUN everywhere it appears - never as a
Version B that scored nothing - because a comparison that was not made is not a
comparison the baseline won.

What this does NOT do
---------------------
Gates 2 and 3 need a walk-forward, a Monte Carlo bootstrap, and the held-back
final three years. Those are separate runs, this is not them, and they report
NOT EVALUATED here. That is not a pass. A strategy leaving this script with
Gate 1 green has cleared one gate of three.

The run is IN-SAMPLE over whatever period is asked for, and `--scan` makes that
sharper rather than softer: the winning parameters were selected on the very
bars they are scored on. The number of combinations tested is carried onto
every report and every leaderboard row for that reason.

Outputs
-------
`/mnt/backtest/artifacts/<strat_name>_<timestamp>/` holds

    report_<SYMBOL>_version_a.html      one per symbol
    report_<SYMBOL>_version_b.html      one per symbol, with --ml
    dual_metrics_<SYMBOL>.json          the snapshot promote.py locks in
    scan_<SYMBOL>.csv                   every parameter combination, with --scan
    summary_leaderboard.csv             ONE row per symbol, A and B side by side
    job.json                            the finished progress record
    run.log                             stdout, with --bg

The timestamp is on the directory, so a re-run never overwrites the evidence an
earlier promotion decision was made on.

`summary_leaderboard.csv` columns, in order:

    timestamp strategy symbol tf params
    sharpe_a pf_a win_rate_a max_dd_a trades_a gate1_a
    sharpe_b gate1_b selected_version html_report
    status error variants_tested scan_selection
    sl_atr_mult tp_atr_mult trailing

`timestamp` is the RUN's stamp, identical on every row and equal to the
directory's, so leaderboards from several runs can be concatenated and grouped.
`params` is the FULL effective parameter set the signals were bound to - the
module's defaults with `--param` and `--scan`'s winner layered over them, not
just what the CLI was handed. `selected_version` records which version led
IN-SAMPLE - it is not a promotion and not the Dual-Version Mandate's verdict,
which requires B to beat A out-of-sample and cannot be established by any run
this script performs.

The last seven columns sit after the declared schema rather than inside it, so
a reader slicing the first fifteen gets exactly the specified file. Without
`status`/`error` a symbol that failed to load reads as a strategy that produced
nothing; without `variants_tested` a `--scan` Sharpe is the best of an unstated
N; and `sl_atr_mult`/`tp_atr_mult`/`trailing` lift the winning risk settings out
of the `params` string into columns that can be filtered and sorted, which is
how "did the take-profit earn its place across contracts" gets answered. In
those three, a BLANK cell means the strategy declares no such parameter, while
the literal word `None` in `tp_atr_mult` means the strategy modelled no
take-profit - two different statements that must not collapse into one.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. This runs at import
# time, above the local imports below, because several modules resolve their
# BT_* variables while being imported (backtest.run's ARTIFACTS_ROOT) - loading
# the file inside main() would be too late for those and would work here, which
# is the kind of difference nobody notices until one runner silently uses the
# default path. The rules - the repository root derived from __file__ rather
# than the working directory, existing variables winning over the file, the
# CrossTrade credentials withheld from os.environ - live in ONE module rather
# than in a block copied into every runner: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # `python3 backtest/x.py` puts backtest/ on sys.path, not the repository
    # root, so mdlib is not importable until this runs.
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------


import os

# Set before numpy and sklearn are imported, because both read the thread count
# at import time. Version B refits its classifier once per completed trade on a
# few dozen rows; on a 16-core box each fit's thread pool costs far more than
# the fit itself. Overridable - this only fills the variable when it is unset.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                   # noqa: E402
import subprocess                                                 # noqa: E402
import sys                                                        # noqa: E402
import time                                                       # noqa: E402
import traceback                                                  # noqa: E402
from datetime import datetime, timezone                           # noqa: E402
from pathlib import Path                                          # noqa: E402

import pandas as pd                                               # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier1_master import (_strategy_indicators,            # noqa: E402
                                 run_dual_version_backtest)
from agents.tier3_workers import load_strategy                    # noqa: E402
from backtest.engine import BacktestConfig                        # noqa: E402
from backtest.event_calendar import (add_filter_args,            # noqa: E402
                                     describe_filters,
                                     filter_config_kwargs)
from backtest.report import NOT_EVALUATED, print_dual_scorecard   # noqa: E402
from backtest.report_html import write_dual_reports               # noqa: E402
from backtest.specs import SPECS, get_spec                        # noqa: E402
from backtest.status import DONE, FAILED, JobTracker              # noqa: E402
from mdlib.lake import available_symbols, iter_bars               # noqa: E402

SEARCH_DIRS = ("strategies/experimental", "strategies")
ARTIFACTS_ROOT = Path(os.environ.get("BT_ARTIFACTS", "/mnt/backtest/artifacts"))

# One row per symbol per version. Everything a reader needs to judge a row
# without opening the report it points at - including how many parameter sets
# it was selected from, because a Sharpe read without that is not a
# measurement.
LEADERBOARD_COLUMNS = [
    # The declared schema, in the declared order. One row per SYMBOL, with
    # Version A and Version B side by side - not one row per version - because
    # the question a leaderboard answers is "which contract, and did the filter
    # earn its place there", and that comparison has to be readable across a
    # row rather than by pairing two of them up by eye.
    "timestamp", "strategy", "symbol", "tf", "params",
    "sharpe_a", "pf_a", "win_rate_a", "max_dd_a", "trades_a", "gate1_a",
    "sharpe_b", "gate1_b", "selected_version", "html_report",
    # Appended, after the declared columns and never among them. A reader
    # slicing the fifteen names above gets exactly the schema that was
    # specified; these exist because dropping them would make the file lie
    # rather than merely make it shorter:
    #   status/error   - a symbol that failed to load has no Sharpe, and a row
    #                    of blanks with no reason beside it reads as a strategy
    #                    that produced nothing rather than a run that broke.
    #   variants_tested/scan_selection
    #                  - under --scan the Sharpe is the best of N, chosen
    #                    in-sample. Reporting it without N is the single thing
    #                    this project's conventions exist to prevent.
    #   sl_atr_mult/tp_atr_mult/trailing
    #                  - the winning RISK settings, lifted out of `params` into
    #                    columns of their own. They are already inside the
    #                    `params` string, but a leaderboard is read by
    #                    filtering and sorting, and "did the target earn its
    #                    place across contracts" is the question a risk sweep
    #                    exists to answer. Blank for a strategy that declares
    #                    no such parameters - blank means "this strategy has no
    #                    such setting", never "the setting was off".
    "status", "error", "variants_tested", "scan_selection",
    "sl_atr_mult", "tp_atr_mult", "trailing",
]

# The risk parameters promoted to their own leaderboard columns. Named here
# rather than discovered, so a strategy that invents `stop_mult` gets a blank
# column and an obvious question instead of a quietly missing one. `promote.py`
# reads the same three names out of the metrics snapshot.
RISK_PARAMS = ("sl_atr_mult", "tp_atr_mult", "trailing")

# `selected_version` values. It records which version LED IN-SAMPLE and nothing
# more. It is not a promotion, not a gate result, and not the Dual-Version
# Mandate's verdict - that requires B to beat A out-of-sample, which no run this
# script performs can establish. promote.py still refuses a version whose gate
# audit is not PASS, whatever this column says.
SEL_A = "A"
SEL_B = "B"
SEL_A_ONLY = "A (B not run)"
SEL_NONE = "NONE (no measurable Sharpe)"


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
def resolve_strategy(name: str) -> Path:
    """
    A module path from either a path or a bare strategy name.

    `strategies/approved_incubator/<name>/strat.py` is searched last and only
    by exact name. Being in the incubator records that a version was chosen; it
    is not a reason to prefer it over the file somebody is currently editing.
    """
    p = Path(name)
    if p.suffix == ".py":
        if not p.exists():
            raise SystemExit(f"strategy module not found: {p}")
        return p.resolve()

    for d in SEARCH_DIRS:
        cand = REPO / d / f"{name}.py"
        if cand.exists():
            return cand.resolve()
    cand = REPO / "strategies" / "approved_incubator" / name / "strat.py"
    if cand.exists():
        return cand.resolve()

    tried = [f"{d}/{name}.py" for d in SEARCH_DIRS]
    tried.append(f"strategies/approved_incubator/{name}/strat.py")
    raise SystemExit(f"no strategy called {name!r}. Tried: {', '.join(tried)}")


def parse_param(text: str) -> tuple[str, object]:
    """
    `key=value`, typed as it looks.

    int before float before bool before string, so `fast_window=10` binds an
    int and not the string "10" - a window that is a string does not raise, it
    just makes `rolling` fail somewhere less obvious.
    """
    if "=" not in text:
        raise SystemExit(f"--param needs key=value, got {text!r}")
    key, _, raw = text.partition("=")
    key, raw = key.strip(), raw.strip()
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            continue
    return key, raw


def parse_symbols(text: str | None, module_symbols: list | None) -> list[str]:
    """
    `NQ`, `NQ,ES,CL`, or `ALL`, falling back to the module's own SYMBOLS.

    Every symbol is checked against the lake AND against `backtest/specs.py`
    before a single bar is read. Both checks are up front on purpose: a batch
    over 27 contracts takes long enough that discovering the twentieth has no
    contract spec, after the first nineteen have run, wastes the run. A missing
    spec is fatal rather than skippable - the multiplier is what turns a price
    move into a dollar, and there is no defensible default for it.
    """
    if text:
        raw = text.strip()
        if raw.upper() == "ALL":
            symbols = list(available_symbols())
        else:
            symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    elif module_symbols:
        symbols = [str(s).strip().upper() for s in module_symbols]
    else:
        raise SystemExit(
            "--symbols is required: the module declares no SYMBOLS, and the "
            "symbol sets the contract multiplier, tick size and commission. "
            "Guessing it would silently rescale every P&L figure.")

    # Preserve the requested order, drop repeats. A symbol listed twice would
    # otherwise be backtested twice and appear twice on the leaderboard.
    seen, ordered = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    try:
        in_lake = set(available_symbols())
    except Exception as e:                                      # noqa: BLE001
        raise SystemExit(f"could not list the lake: {type(e).__name__}: {e}")

    missing = [s for s in ordered if s not in in_lake]
    if missing:
        raise SystemExit(
            f"not in the lake: {', '.join(missing)}. Available: "
            f"{', '.join(sorted(in_lake))}")

    if not ordered:
        raise SystemExit("--symbols resolved to nothing. Pass a symbol, a "
                         "comma-separated list, or ALL.")

    unspecced = [s for s in ordered if s not in SPECS]
    if unspecced:
        raise SystemExit(
            f"no contract spec for: {', '.join(unspecced)}. Add them to "
            f"backtest/specs.py - a backtest cannot compute P&L without the "
            f"multiplier, and a default would silently rescale every figure.")

    return ordered


def parse_timeframes(text: str | None, module_tf: str | None,
                     fallback: str = "15m") -> list[str]:
    """
    `--tf 1m,5m,15m,30m` -> `["1m", "5m", "15m", "30m"]`, validated.

    One timeframe is the normal case and comes back as a one-element list, so
    every caller loops and there is no second code path for the single-tf run
    to diverge along.

    Validated against what `mdlib.lake` actually serves, and validated HERE
    rather than at the first read: an unknown timeframe otherwise surfaces
    several minutes into a sweep, after the first contract has already been
    swept at the timeframes that were spelled correctly.

    Nothing is resampled by the caller. `1m` and `1d` are stored natively and
    `5m/15m/30m/1h/2h/4h` are derived from the 1-minute parquet inside the
    reader (`mdlib.lake.DERIVED`), so `--tf 1m,5m,15m,30m` reads the same 1m
    files four times and aggregates each differently. Resampling in a stage
    instead would be a second implementation of the lake's aggregation, free to
    disagree with it about bar boundaries and about the Sunday session merge.

    Duplicates are collapsed and the ORDER GIVEN is preserved - a run's log
    should read in the order the operator asked for, and sorting "1m,5m,30m"
    into some canonical order makes a long log harder to follow, not easier.
    """
    from mdlib.lake import DERIVED, NATIVE_TFS

    raw = (text or "").strip()
    if not raw:
        return [module_tf or fallback]

    known = set(NATIVE_TFS) | set(DERIVED)
    out: list[str] = []
    for part in raw.split(","):
        tf = part.strip()
        if not tf or tf in out:
            continue
        if tf not in known:
            raise ValueError(
                f"unknown timeframe {tf!r}. The lake serves "
                f"{', '.join(sorted(known))} — {', '.join(sorted(NATIVE_TFS))} "
                f"natively and the rest derived from 1m.")
        out.append(tf)
    if not out:
        raise ValueError(f"--tf {text!r} names no timeframe")
    return out


def load_bars(symbol: str, tf: str, start: str | None,
              end: str | None) -> pd.DataFrame:
    """
    One symbol's bars, oldest first, exactly as the engine would read them.

    `iter_bars` rather than `get_bars`: one frame, one instrument, no
    interleaving, and peak RAM tracks the largest single symbol rather than the
    lake.
    """
    for sym, frame in iter_bars([symbol], tf, start, end):
        if sym == symbol and len(frame):
            return frame.reset_index(drop=True)
    raise ValueError(
        f"the lake returned no {tf} bars for {symbol} over "
        f"{start or 'the start of history'} → {end or 'the end'}")


# --------------------------------------------------------------------------
# Leaderboard
# --------------------------------------------------------------------------
def _gate(audit: dict | None, key: str) -> str:
    return (audit or {}).get("gates", {}).get(key, {}).get("status",
                                                           NOT_EVALUATED)


def _win_rate_pct(metrics: dict) -> float | None:
    """
    Win rate as a percent.

    `summarize_result` stores it as a fraction. This is the only unit
    conversion anywhere on the leaderboard path; every other figure is carried
    through untouched so a row and the report it points at cannot disagree.
    """
    pct = metrics.get("win_rate_pct")
    if pct is not None:
        return pct
    frac = metrics.get("win_rate")
    return None if frac is None else float(frac) * 100


def select_version(metrics_a: dict | None, metrics_b: dict | None) -> str:
    """
    Which version led IN-SAMPLE. Not a promotion and not a mandate verdict.

    Version B is adopted only if it beats A OUT-OF-SAMPLE, and no run this
    script performs can establish that - the filter was fitted walk-forward on
    the very period it is being scored over. So `B` here means "B's in-sample
    Sharpe was higher", which is a fact about this run and not a
    recommendation. When B was never run the value says so explicitly rather
    than defaulting to `A`: an uncontested A is not a winning A.
    """
    sa = (metrics_a or {}).get("sharpe")
    if metrics_b is None:
        return SEL_A_ONLY if sa is not None and not pd.isna(sa) else SEL_NONE
    sb = metrics_b.get("sharpe")
    a_ok = sa is not None and not pd.isna(sa)
    b_ok = sb is not None and not pd.isna(sb)
    if not a_ok and not b_ok:
        return SEL_NONE
    if not b_ok:
        return SEL_A
    if not a_ok:
        return SEL_B
    return SEL_B if float(sb) > float(sa) else SEL_A


def leaderboard_row(timestamp: str, strategy: str, symbol: str, tf: str,
                    metrics_a: dict | None, audit_a: dict | None,
                    metrics_b: dict | None = None,
                    audit_b: dict | None = None,
                    reports: dict | None = None,
                    **extra) -> dict:
    """
    One row per symbol, Version A and Version B side by side.

    `gate1_b` is left blank rather than filled with NOT EVALUATED when Version
    B did not run. NOT EVALUATED is a statement about a gate that was reached
    and had no evidence behind it; a version that never ran did not reach it,
    and printing the same token for both would make a skipped B look like a B
    whose robustness run is merely outstanding.

    `html_report` is the SELECTED version's tear sheet. Its sibling is one
    substitution away - the filenames are `report_<SYMBOL>_version_a.html` and
    `..._version_b.html` in the same directory - so naming one loses nothing.
    """
    a = metrics_a or {}
    reports = reports or {}
    selected = extra.pop("selected_version", None) or select_version(metrics_a,
                                                                    metrics_b)
    report = (reports.get("report_version_b") if selected == SEL_B
              else reports.get("report_version_a"))

    row = {
        "timestamp": timestamp,
        "strategy": strategy,
        "symbol": symbol,
        "tf": tf,
        "params": None,
        "sharpe_a": a.get("sharpe"),
        "pf_a": a.get("profit_factor"),
        "win_rate_a": _win_rate_pct(a),
        "max_dd_a": a.get("max_drawdown_pct"),
        "trades_a": a.get("trade_count"),
        "gate1_a": _gate(audit_a, "gate1") if audit_a else None,
        "sharpe_b": (metrics_b or {}).get("sharpe") if metrics_b else None,
        "gate1_b": _gate(audit_b, "gate1") if audit_b else None,
        "selected_version": selected,
        "html_report": str(report) if report else None,
        "status": "OK",
        "error": None,
        "variants_tested": None,
        "scan_selection": None,
        "sl_atr_mult": None,
        "tp_atr_mult": None,
        "trailing": None,
    }
    row.update(extra)
    return {c: row.get(c) for c in LEADERBOARD_COLUMNS}


def risk_columns(bound_params: dict | None) -> dict:
    """
    The winning risk settings, for their own leaderboard columns.

    Read from the strategy's BOUND parameters - the module's DEFAULT_PARAMS
    with the CLI's `--param` and `--scan`'s winner layered over them - rather
    than from the CLI arguments alone. A run with neither flag uses the
    module's defaults, and reporting those as blank would say "no stop was
    modelled" about a strategy that stopped out of half its trades.

    A parameter the strategy does not declare stays None, which the CSV writes
    as an empty cell. Blank therefore means "this strategy has no such
    setting". `tp_atr_mult=None` is a DIFFERENT statement - "no take-profit was
    modelled" - and is written as the word so the two cannot be confused.
    """
    bound = bound_params or {}
    out: dict[str, object] = {}
    for k in RISK_PARAMS:
        if k not in bound:
            out[k] = None
            continue
        v = bound[k]
        # `str(None)` rather than None, so this column distinguishes "the
        # strategy swept the target off" from "the strategy has no target".
        out[k] = "None" if v is None else v
    return out


def write_leaderboard(rows: list[dict], out_dir: Path) -> Path:
    """
    Rewrite `summary_leaderboard.csv` from scratch after every symbol.

    Rewritten rather than appended so the file is always a complete, sorted
    view of everything finished so far - a batch killed at symbol 14 leaves a
    readable leaderboard of 14, not a half-written line.

    Sorted by Version A's Sharpe, best first, with errored symbols last. Sorted
    on A rather than on the selected version because A is the column every row
    has: ranking on a mixture of A and B Sharpes would put a symbol whose ML
    filter happened to run above one where it did not, which is a fact about
    the flags rather than about the market.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary_leaderboard.csv"
    df = pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)
    if not df.empty:
        df = df.assign(_err=df["status"].ne("OK"))
        df = (df.sort_values(["_err", "sharpe_a"], ascending=[True, False],
                             na_position="last", kind="stable")
                .drop(columns="_err")
                .reset_index(drop=True))
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Backtest a strategy across one or many contracts, one "
                    "independent simulation each, and write a tear sheet and "
                    "a leaderboard row for every one.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strat", required=True,
                    help="strategy name (searched under strategies/) or a path "
                         "to a .py module")
    ap.add_argument("--symbols", help="one symbol (NQ), a comma-separated list "
                                      "(NQ,ES,CL), or ALL for every symbol in "
                                      "the lake. Defaults to the module's "
                                      "SYMBOLS.")
    ap.add_argument("--symbol", dest="symbols_legacy",
                    help=argparse.SUPPRESS)   # the pre-batch spelling
    ap.add_argument("--tf", help="timeframe. Defaults to the module's "
                                 "TIMEFRAME, then 15m. 1m and 1d are stored; "
                                 "the rest are derived by the lake reader.")
    ap.add_argument("--scan", action="store_true",
                    help="sweep the module's PARAM_GRID per symbol and keep "
                         "the highest-Sharpe set that clears Gate 1")
    ap.add_argument("--bg", action="store_true",
                    help="detach and run the batch as a background process. "
                         "Track it with bt-status.")
    ap.add_argument("--ml", action="store_true",
                    help="also run Version B (the ML filter) per symbol. Off "
                         "by default: the classifier refits per completed "
                         "trade, which across many symbols is hours.")
    ap.add_argument("--start", help="first bar, e.g. 2016-01-01")
    ap.add_argument("--end", help="last bar, e.g. 2023-12-31")
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="strategy parameter, repeatable. Merges over the "
                         "module's DEFAULT_PARAMS, and pins that parameter "
                         "under --scan only if the grid does not sweep it.")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="P(win) at or above which Version B keeps an entry "
                         "(default: 0.50)")
    ap.add_argument("--capital", type=float, default=100_000.0,
                    help="starting equity (default: 100,000)")
    ap.add_argument("--contracts", type=int, default=1,
                    help="contracts per trade (default: 1)")
    ap.add_argument("--slippage-ticks", type=float, default=1.0,
                    help="slippage charged each way, in ticks (default: 1.0). "
                         "Costs are not optional here — a variant ranking "
                         "changes once they are applied.")
    ap.add_argument("--flat-by-close", action="store_true",
                    help="close any open position at the session close "
                         "(Portfolio A / intraday setting)")
    ap.add_argument("--variants-tested", type=int, default=None,
                    help="how many variants this result was selected from. "
                         "Carried onto every report. --scan fills it in with "
                         "the size of the grid it actually evaluated.")
    ap.add_argument("--out", help="directory for the artifacts. Defaults to "
                                  "<artifacts>/<strat_name>_<timestamp>/")
    ap.add_argument("--no-reports", action="store_true",
                    help="skip the HTML tear sheets; scorecards and the "
                         "leaderboard are still written")
    # The news and day-of-week entry filters, defined once in
    # backtest/event_calendar.py so the batch runner and the five pipeline
    # stages cannot drift apart on what --exclude-days 0 means.
    add_filter_args(ap)
    return ap


MENU = """
Next step — your call. Nothing here promotes anything.

    [1] Promote Version A to Incubator     [2] Promote Version B to Incubator
    [3] Parameter Sweep / Sensitivity      [4] Keep in Experimental / New Idea

    python3 backtest/promote.py --strat {strat} --version A \\
        --source {source} \\
        --metrics {metrics}

Promotion is per strategy, and this run covered {n} contract(s). Pick the
symbol whose evidence you are promoting on and pass that symbol's
dual_metrics_<SYMBOL>.json — the file records which contract produced the
numbers.
"""


# --------------------------------------------------------------------------
# Background launch
# --------------------------------------------------------------------------
def relaunch_detached(art_dir: Path) -> int:
    """
    Re-exec this script without --bg, detached, with stdout in the run's log.

    `start_new_session` puts the child in its own session, so it survives the
    terminal that started it closing - which is the point of --bg. The argv is
    the user's own, minus --bg and with --out pinned to the directory the
    parent already created, so the parent can print where the log will be
    before the child has written a byte to it.
    """
    argv = [a for a in sys.argv[1:] if a != "--bg"]
    # Both spellings, or `--out=/x` would slip past the check and the child
    # would be handed a second --out. argparse takes the last one, which is
    # this one, so the user's own choice would be silently discarded.
    if not any(a == "--out" or a.startswith("--out=") for a in argv):
        argv += ["--out", str(art_dir)]

    art_dir.mkdir(parents=True, exist_ok=True)
    log = art_dir / "run.log"
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *argv],
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(REPO), start_new_session=True)

    print(f"Batch detached.  pid {proc.pid}")
    print(f"  artifacts : {art_dir}")
    print(f"  log       : {log}")
    print(f"  progress  : bt-status          (or python3 backtest/status.py)")
    print(f"              bt-status --watch  to follow it")
    return 0


# --------------------------------------------------------------------------
# One symbol
# --------------------------------------------------------------------------
def run_symbol(symbol: str,
               path: Path,
               info: dict,
               args: argparse.Namespace,
               tf: str,
               params: dict,
               art_dir: Path,
               strat_name: str,
               stamp: str) -> dict:
    """
    One contract, start to finish: bars, optional sweep, A (and B), reports.

    Returns this symbol's single leaderboard row, with Version A and Version B
    side by side. It raises freely; the batch loop catches everything, so one
    bad contract cannot end a 27-symbol run.
    """
    t0 = time.time()
    spec = get_spec(symbol)

    print("\n" + "=" * 78)
    print(f"{symbol}  ·  {tf}  ·  {strat_name}")
    print("=" * 78)
    print(f"  contract   : x{spec.multiplier:g} per point, tick {spec.tick_size:g} "
          f"(${spec.tick_value:.2f}), ${spec.commission:.2f}/side")

    bars = load_bars(symbol, tf, args.start, args.end)
    # Belt and braces on the constraint the whole file is built around. The
    # reader yields one symbol per frame; if that ever changed, every rolling
    # window below would quietly start averaging across contracts.
    if "symbol" in bars.columns and bars["symbol"].nunique() > 1:
        raise ValueError(f"{symbol}: the lake returned an interleaved frame "
                         f"({bars['symbol'].nunique()} symbols)")
    print(f"  bars       : {len(bars):,}  "
          f"{bars['ts'].iloc[0]} → {bars['ts'].iloc[-1]}")

    cfg = BacktestConfig(
        initial_capital=args.capital,
        contracts=args.contracts,
        slippage_ticks=args.slippage_ticks,
        flat_by_close=args.flat_by_close,
        variants_tested=args.variants_tested,
        notes=f"backtest/run.py batch {symbol} {tf}",
        **filter_config_kwargs(args))

    run_params = dict(params)
    scan_selection = None
    if args.scan:
        from backtest.scan import (SIZE_WARN, expand_grid,       # noqa: F401
                                   format_scan_summary, scan_symbol,
                                   write_scan_table)

        grid = info.get("param_grid") or {}
        print("  scanning   : "
              + ", ".join(f"{k}={v!r}" for k, v in grid.items()))
        # The size of the search, before it runs rather than after. Risk axes
        # multiply onto indicator axes, so a grid that looks like five short
        # lists is routinely several hundred fits to one sample of bars, and
        # the best of several hundred is a different claim from the best of
        # nine. It is printed always and flagged past SIZE_WARN; nothing is
        # refused, because a grid somebody deliberately wrote is theirs to run.
        cells = len(expand_grid(grid))
        print(f"  grid size  : {cells:,} combination(s) per symbol")
        if cells > SIZE_WARN:
            print(f"\n  [!] {cells:,} combinations is a large in-sample "
                  f"search. The winning Sharpe is the best\n"
                  f"      of {cells:,} fits to these same bars, and that is "
                  f"how it has to be read — the count\n"
                  f"      is carried onto every report and every leaderboard "
                  f"row as variants_tested.\n", file=sys.stderr, flush=True)
        scan = scan_symbol(path, bars, symbol, cfg, grid,
                           base_params=params, strat_name=strat_name)
        print(format_scan_summary(scan))
        write_scan_table(scan, art_dir)
        scan_selection = scan["selection"]
        if scan["winner"]:
            run_params = {**params, **scan["winner"]["params"]}
        # The search is part of the result. Selecting the best of N in-sample
        # and reporting the Sharpe without N is the headline failure this
        # project is built to avoid, so the count goes onto the config the
        # reported run uses, not just into the scan CSV.
        if args.variants_tested is None:
            cfg.variants_tested = scan["evaluated"]

    print(f"  parameters : {run_params or '(module defaults)'}")

    # Rebind against the parameters this run actually uses. `info` was bound to
    # the CLI's --param at startup, and under --scan the winning set is not
    # those: drawing the overlay from the stale binding would put a Fast SMA(10)
    # line under trades taken by a Fast SMA(20) crossover, with the chart a bar
    # or two away from where the entry fired and nothing raising. The line a
    # reader watches cross has to be the array the entry came from.
    run_fn, run_info = load_strategy(path, run_params)
    del run_fn

    out = run_dual_version_backtest(
        str(path), bars, freq=tf, symbol=symbol, cfg=cfg, params=run_params,
        threshold=args.threshold, ml=args.ml,
        emit_reports=False, strat_name=strat_name)

    a = out["version_a"]
    b = out["version_b"]

    filters = a["metrics"].get("entry_filters") or {}
    if filters:
        print("\n" + describe_filters(filters))

    # Steps 1 and 3. Version A is the left column, because Version B only means
    # anything measured against it.
    print()
    print_dual_scorecard(a["metrics"], b["metrics"] if b else None,
                         a["gate_audit"], b["gate_audit"] if b else None)

    # Steps 2 and 4. Prefixed with the symbol: one directory holds the whole
    # batch, and an unprefixed NQ report and an unprefixed ES report would
    # leave only the second with nothing raising.
    reports, report_error = {}, None
    if not args.no_reports:
        try:
            reports = write_dual_reports(
                out, bars=bars, out_dir=art_dir, strat_name=strat_name,
                indicators=_strategy_indicators(run_info, bars), prefix=symbol)
            print("\n  Version A  : " + str(reports["report_version_a"]))
            if reports.get("report_version_b"):
                print("  Version B  : " + str(reports["report_version_b"]))
            print("  Metrics    : " + str(reports["metrics_json"]))
        except Exception as e:                                  # noqa: BLE001
            # Recorded, printed, and the run continues. A completed backtest is
            # not thrown away because an NFS mount was busy.
            report_error = f"{type(e).__name__}: {e}"
            print(f"\n[!] {symbol}: reports were NOT written: {report_error}",
                  file=sys.stderr, flush=True)

    meta = out["meta"]
    row = leaderboard_row(
        timestamp=stamp, strategy=strat_name, symbol=symbol, tf=tf,
        metrics_a=a["metrics"], audit_a=a["gate_audit"],
        metrics_b=b["metrics"] if b else None,
        audit_b=b["gate_audit"] if b else None,
        reports=reports,
        # The FULL effective set, not just what the CLI passed: under --scan
        # this is the winning combination, and under neither flag it is the
        # module's own defaults. `run_info["bound_params"]` is what the signal
        # function was actually bound to, so a row and the report it points at
        # cannot disagree about which parameters produced the numbers.
        params=str(run_info.get("bound_params") or meta.get("params") or {}),
        variants_tested=cfg.variants_tested,
        scan_selection=scan_selection,
        error=report_error,
        **risk_columns(run_info.get("bound_params")))

    if b is not None:
        cmp_ = out["comparison"]
        print(f"\n  The filter suppressed {cmp_['entries_suppressed']:,} of "
              f"{cmp_['entries_a']:,} entries. B beats A on Sharpe: "
              f"{cmp_['b_beats_a']}.")
        print(f"  In-sample lead: Version {row['selected_version']}. That is a "
              f"fact about this run, not a\n  promotion — the mandate adopts B "
              f"only if it beats A out-of-sample.")
    else:
        print("\n  Version B (ML filter) was NOT RUN. Pass --ml to evaluate it; "
              "until then the\n  Dual-Version Mandate's comparison is "
              "outstanding, not settled.")

    print(f"  ({round(time.time() - t0, 1)}s)")
    return row


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path = resolve_strategy(args.strat)
    params = dict(parse_param(p) for p in args.param)
    strat_name = path.parent.name if path.stem == "strat" else path.stem

    # Loaded once up front, purely to read its declarations and to fail on a
    # broken module before a lake read that can take minutes.
    try:
        _fn, info = load_strategy(path, params)
    except Exception as e:                                      # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    symbols = parse_symbols(args.symbols or args.symbols_legacy,
                            info.get("symbols"))
    tf = args.tf or info.get("timeframe") or "15m"

    if args.scan and not info.get("param_grid"):
        print(f"--scan was passed but {strat_name} declares no PARAM_GRID, so "
              f"there is nothing to sweep.\nAdd one to the module, or drop "
              f"--scan.", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    art_dir = (Path(args.out) if args.out
               else ARTIFACTS_ROOT / f"{strat_name}_{stamp}")

    if args.bg:
        return relaunch_detached(art_dir)

    art_dir.mkdir(parents=True, exist_ok=True)

    print(f"Strategy   : {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    print(f"Symbols    : {len(symbols)} — {', '.join(symbols)}")
    print(f"Timeframe  : {tf}")
    print(f"Parameters : {info.get('bound_params') or '(module defaults)'}")
    print(f"Scan       : {'PARAM_GRID sweep per symbol' if args.scan else 'off'}")
    print(f"Version B  : {'on' if args.ml else 'off (--ml to evaluate the ML filter)'}")
    print(f"Indicators : "
          f"{'declared — drawn on the trade inspector' if info.get('indicator_fn') else 'none declared'}")
    print(f"Artifacts  : {art_dir}")
    print("\nEach symbol is an independent simulation on its own contract "
          "spec. Nothing is pooled.")

    job = JobTracker(job_id=art_dir.name, strategy=strat_name, symbols=symbols,
                     timeframe=tf, artifact_dir=art_dir, scan=args.scan,
                     ml=args.ml)
    job.start()

    rows: list[dict] = []
    failures = 0
    # Written before the loop as well as inside it, so a batch that dies on its
    # very first symbol still leaves a leaderboard file rather than nothing.
    board = write_leaderboard(rows, art_dir)
    for symbol in symbols:
        job.start_symbol(symbol)
        try:
            row = run_symbol(symbol, path, info, args, tf, params,
                             art_dir, strat_name, stamp)
            rows.append(row)
            job.finish_symbol(symbol, {
                "status": "OK",
                "sharpe": row["sharpe_a"],
                "profit_factor": row["pf_a"],
                "trades": row["trades_a"],
                "max_drawdown_pct": row["max_dd_a"],
                "gate1": row["gate1_a"],
                "error": row["error"],
            })
        except Exception as e:                                  # noqa: BLE001
            # One contract's failure is one contract's failure. A missing spec,
            # an empty slice of the lake or a strategy that raises on this
            # symbol's data must not end a 27-symbol batch that has been
            # running for an hour - the row records what happened and the loop
            # moves on.
            failures += 1
            reason = f"{type(e).__name__}: {e}"
            print(f"\n[!] {symbol} FAILED: {reason}", file=sys.stderr)
            traceback.print_exc()
            rows.append(leaderboard_row(
                timestamp=stamp, strategy=strat_name, symbol=symbol, tf=tf,
                metrics_a=None, audit_a=None, status="ERROR",
                selected_version=SEL_NONE, params=str(params), error=reason))
            job.finish_symbol(symbol, {"status": "ERROR", "error": reason})

        board = write_leaderboard(rows, art_dir)

    job.finish(DONE if failures < len(symbols) else FAILED,
               error=None if not failures else f"{failures} symbol(s) failed")
    job.snapshot(art_dir)

    print("\n" + "=" * 78)
    print("BATCH COMPLETE")
    print("=" * 78)
    ok = len([r for r in rows if r["status"] == "OK"])
    print(f"  {ok}/{len(symbols)} symbols completed"
          + (f", {failures} failed" if failures else ""))
    print(f"  Leaderboard : {board}")
    print(f"  Artifacts   : {art_dir}")
    print("\n  Read each logic card before its metrics. It states what the "
          "engine actually did —\n  fills on the next bar's open, no "
          "stop-loss, no take-profit — and the trade inspector\n  draws the "
          "strategy's own indicator lines over the candles of any trade you "
          "click.")
    print("\n  Every result here is IN-SAMPLE. Gates 2 and 3 report NOT "
          "EVALUATED, which is not a\n  pass: the walk-forward, the bootstrap "
          "and the held-back final three years are\n  separate runs, and this "
          "script does not touch them.")
    if args.scan:
        print("\n  --scan selected each symbol's parameters on the same bars "
              "it scored them on.\n  Every leaderboard row carries the number "
              "of combinations that produced it.")

    print(MENU.format(
        strat=strat_name,
        source=path.relative_to(REPO) if path.is_relative_to(REPO) else path,
        metrics=str(art_dir / f"dual_metrics_{symbols[0]}.json"),
        n=len(symbols)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
