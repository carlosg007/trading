#!/usr/bin/env python3
"""
run.py - run one strategy through the dual-version workflow, end to end.

Location:  ~/src/trading/backtest/run.py

    python3 backtest/run.py --strat sma_crossover --symbol NQ --tf 1d
    python3 backtest/run.py --strat sma_crossover --symbol NQ --tf 15m \\
        --start 2018-01-01 --end 2023-12-31 --param fast_window=10

The five documented steps, in order:

    1. console scorecard and gate audit - Version A
    2. standalone HTML report with the trade inspector - Version A
    3. console scorecard and gate audit - Version B
    4. standalone HTML report with the trade inspector - Version B
    5. the four-choice menu

Steps 1-4 happen here. Step 5 is printed and stops: nothing is promoted
without a human, and this script has no path that promotes anything. The
scorecard renders A and B as two columns of one table, so 1 and 3 arrive
together with a B-minus-A delta between them - which is the comparison the
Dual-Version Mandate is actually about.

What this does NOT do
---------------------
Gates 2 and 3 need a walk-forward, a Monte Carlo bootstrap, and the held-back
final three years. Those are separate runs, this is not them, and they report
NOT EVALUATED here. That is not a pass. A strategy leaving this script with
Gate 1 green has cleared one gate of three.

The run is IN-SAMPLE over whatever period is asked for. Reserve the last three
years or the holdout is not a holdout.

Bars come from `mdlib.lake.iter_bars`, one symbol at a time - never `get_bars`,
whose frame interleaves symbols and would have a rolling mean averaging across
contracts.
"""

from __future__ import annotations

import os

# Set before numpy and sklearn are imported, because both read the thread count
# at import time. Version B refits its classifier once per completed trade on a
# few dozen rows; on a 16-core box each fit's thread pool costs far more than
# the fit itself. Overridable - this only fills the variable when it is unset.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                   # noqa: E402
import sys                                                        # noqa: E402
import traceback                                                  # noqa: E402
from pathlib import Path                                          # noqa: E402

import pandas as pd                                               # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier1_master import run_dual_version_backtest          # noqa: E402
from agents.tier3_workers import load_strategy                     # noqa: E402
from backtest.engine import BacktestConfig                         # noqa: E402
from backtest.report import print_dual_scorecard                   # noqa: E402
from mdlib.lake import iter_bars                                   # noqa: E402

SEARCH_DIRS = ("strategies/experimental", "strategies")


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


def load_bars(symbol: str, tf: str, start: str | None,
              end: str | None) -> pd.DataFrame:
    """One symbol's bars, oldest first, exactly as the engine would read them."""
    for sym, frame in iter_bars([symbol], tf, start, end):
        if sym == symbol and len(frame):
            return frame.reset_index(drop=True)
    raise SystemExit(
        f"the lake returned no {tf} bars for {symbol} over "
        f"{start or 'the start of history'} → {end or 'the end'}.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run a strategy as Version A and Version B and write both "
                    "tear sheets.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strat", required=True,
                    help="strategy name (searched under strategies/) or a path "
                         "to a .py module")
    ap.add_argument("--symbol", help="contract to run. Defaults to the "
                                     "module's first declared SYMBOLS entry.")
    ap.add_argument("--tf", help="timeframe. Defaults to the module's "
                                 "TIMEFRAME. 1m and 1d are stored; the rest "
                                 "are derived by the lake reader.")
    ap.add_argument("--start", help="first bar, e.g. 2016-01-01")
    ap.add_argument("--end", help="last bar, e.g. 2023-12-31")
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="strategy parameter, repeatable. Merges over the "
                         "module's DEFAULT_PARAMS.")
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
                         "Carried onto the report — a Sharpe read without it "
                         "is not a measurement.")
    ap.add_argument("--out", help="directory for the reports. Defaults to "
                                  "/mnt/backtest/artifacts/<strat>_<timestamp>/")
    ap.add_argument("--no-reports", action="store_true",
                    help="skip the HTML tear sheets and print the scorecard only")
    return ap


MENU = """
Next step — your call. Nothing here promotes anything.

    [1] Promote Version A to Incubator     [2] Promote Version B to Incubator
    [3] Parameter Sweep / Sensitivity      [4] Keep in Experimental / New Idea

    python3 backtest/promote.py --strat {strat} --version A \\
        --source {source} \\
        --metrics {metrics}
"""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path = resolve_strategy(args.strat)
    params = dict(parse_param(p) for p in args.param)

    # Loaded once up front, purely to read its declarations and to fail on a
    # broken module before a lake read that can take minutes.
    try:
        _fn, info = load_strategy(path, params)
    except Exception as e:                                      # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    symbol = args.symbol or (info.get("symbols") or [None])[0]
    tf = args.tf or info.get("timeframe") or "1d"
    if symbol is None:
        print("--symbol is required: the module declares no SYMBOLS, and the "
              "symbol sets the contract multiplier, tick size and commission. "
              "Guessing it would silently rescale every P&L figure.",
              file=sys.stderr)
        return 1

    print(f"Strategy   : {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    print(f"Symbol     : {symbol}   Timeframe: {tf}")
    print(f"Parameters : {info.get('bound_params') or '(module defaults)'}")
    print(f"Indicators : "
          f"{'declared — drawn on the trade inspector' if info.get('indicator_fn') else 'none declared'}")
    print(f"\nReading {symbol} {tf} bars"
          f"{f' {args.start} → {args.end}' if args.start or args.end else ''}…")

    try:
        bars = load_bars(symbol, tf, args.start, args.end)
    except SystemExit:
        raise
    except Exception:
        print("\nLAKE READ FAILED:\n", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(f"{len(bars):,} bars, {bars['ts'].iloc[0]} → {bars['ts'].iloc[-1]}\n")

    cfg = BacktestConfig(
        initial_capital=args.capital,
        contracts=args.contracts,
        slippage_ticks=args.slippage_ticks,
        flat_by_close=args.flat_by_close,
        variants_tested=args.variants_tested,
        notes=f"backtest/run.py {symbol} {tf}")

    print("Running Version A (rule-based) and Version B (ML-filtered) under "
          "identical costs…")
    try:
        out = run_dual_version_backtest(
            str(path), bars, freq=tf, symbol=symbol, cfg=cfg, params=params,
            threshold=args.threshold,
            emit_reports=not args.no_reports,
            report_dir=args.out,
            strat_name=path.parent.name if path.stem == "strat" else path.stem)
    except Exception:
        print("\nTHE RUN RAISED:\n", file=sys.stderr)
        traceback.print_exc()
        return 1

    a, b = out["version_a"], out["version_b"]

    # Steps 1 and 3: the scorecard puts both versions side by side with the
    # gate table under them. Version A is the left column, because Version B
    # only means anything measured against it.
    print_dual_scorecard(a["metrics"], b["metrics"],
                         a["gate_audit"], b["gate_audit"])

    # Steps 2 and 4.
    reports = out.get("reports") or {}
    metrics_path = "<dual_metrics.json>"
    if reports.get("error"):
        print(f"\n[!] HTML reports were NOT written: {reports['error']}",
              file=sys.stderr)
    elif reports:
        print("\nReports")
        print(f"  Version A : {reports['report_version_a']}")
        print(f"  Version B : {reports['report_version_b']}")
        print(f"  Metrics   : {reports['metrics_json']}")
        print("  Read the strategy logic card before the metrics. It states "
              "what the engine actually did —\n  fills on the next bar's open, "
              "no stop-loss, no take-profit — and the trade inspector draws "
              "the\n  strategy's own indicator lines over the candles of any "
              "trade you click.")
        metrics_path = str(reports["metrics_json"])
    elif args.no_reports:
        print("\n(--no-reports: no tear sheets were written.)")

    cmp_ = out["comparison"]
    print(f"\nThe filter suppressed {cmp_['entries_suppressed']:,} of "
          f"{cmp_['entries_a']:,} entries. B beats A on Sharpe: "
          f"{cmp_['b_beats_a']}.")
    print("This run is IN-SAMPLE. B winning here is not the evidence the "
          "mandate asks for — that is\nthe held-back final three years, which "
          "this script does not touch.")

    # Step 5.
    print(MENU.format(
        strat=path.parent.name if path.stem == "strat" else path.stem,
        source=path.relative_to(REPO) if path.is_relative_to(REPO) else path,
        metrics=metrics_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
