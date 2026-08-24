"""
backtest/verify_full.py - STAGE 4 of 5: the whole lifecycle, on the record.

Location: ~/src/trading/backtest/verify_full.py

One run per contract across every bar the lake holds - in-sample, holdout and
everything on either side of them - producing the artifacts a promotion
decision is read from: the full tear sheets with the trade inspector, the trade
log as a CSV, and the cost drag broken out year by year.

    python3 backtest/verify_full.py --strat ema_trend_filter --symbols NQ \\
        --tf 15m --start 2010-01-01 --end 2026-01-01

What this stage is, and what it is NOT
--------------------------------------
**It is not a gate, and it cannot certify one.** Stage 3 already spent the
holdout; this run covers the holdout as part of a continuous window, so every
metric it reports is contaminated by construction. The banner says so, the
JSON records `is_certification: false`, and no gate table is printed. What it
is for is the questions the gates do not ask: does the equity curve survive
2015 and 2020, is the edge concentrated in two years, what did the commissions
actually take.

**It is the answer to "what would this have felt like to hold".** Sixteen years
of an equity curve, a monthly heatmap and every trade in a sortable table is a
different kind of evidence from a ratio that cleared a threshold, and it is the
kind that catches a strategy whose entire profit came from one quarter.

Cost drag
---------
Reported three ways, because the same costs read differently at each:

    total          dollars of commission and slippage over the whole run
    per trade      what one round turn cost, on average
    as % of gross  the share of gross P&L handed to the broker and the spread

The third is the one that decides whether a strategy is real. An edge whose
costs are 85% of its gross profit is an edge that dies on a wider spread or one
extra tick of slippage, and it will look fine in every ratio above. Commission
is split out from slippage where the arithmetic permits (see `_cost_split`);
where the remainder would come out negative the split is abandoned and the
combined figure is reported, rather than printing an impossible column.

Per-year rows sit under the total. A strategy whose cost share climbs steadily
is one whose average trade is shrinking - the signal is decaying while the
headline Sharpe holds up on the early years.
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

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                   # noqa: E402
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
from agents.tier3_workers import load_strategy                     # noqa: E402
from backtest.engine import BacktestConfig, round_turn_cost        # noqa: E402
from backtest.event_calendar import (add_filter_args,              # noqa: E402
                                     describe_filters,
                                     filter_config_kwargs)
from backtest.pipeline import (BEST_PARAMS_FILE, VERIFY_FILE,      # noqa: E402
                               next_step, pipeline_dir, read_stage,
                               stage_banner, write_stage)
from backtest.profiler import RegimeProfiler                       # noqa: E402
from backtest.report import (day_of_week_breakdown,                # noqa: E402
                             format_day_of_week, print_dual_scorecard)
from backtest.report_html import _cost_split, write_dual_reports   # noqa: E402
from backtest.audit_gates import discover_symbols                  # noqa: E402
from backtest.run import (load_bars, parse_param, parse_symbols,    # noqa: E402
                          resolve_strategy)
from backtest.specs import get_spec                                # noqa: E402


def cost_drag(trades: pd.DataFrame, result, symbol: str,
              cfg: BacktestConfig) -> dict:
    """
    What the broker and the spread took, in total and per year.

    `cost_share_pct` is costs as a percentage of GROSS profit, and it is
    undefined - reported as None, never as 0 - when gross P&L is not positive.
    A strategy that lost money gross has no profit for its costs to be a share
    of, and printing 0% there would read as a run that cost nothing.
    """
    empty = {"trades": 0, "total_costs": 0.0, "gross_pnl": 0.0,
             "net_pnl": 0.0, "cost_per_trade": None, "cost_share_pct": None,
             "commission": None, "slippage": None, "split_available": False,
             "modelled_round_turn": None, "by_year": []}
    if trades is None or len(trades) == 0 or "costs" not in trades.columns:
        return empty

    costs = trades["costs"].astype(float)
    gross = trades["gross_pnl"].astype(float)
    net = trades["pnl"].astype(float)

    split = _cost_split(trades, result, symbol)
    fees, slip = split if split else (None, None)

    def _share(g: float, c: float):
        return (100.0 * c / g) if g > 0 else None

    try:
        modelled = round_turn_cost(symbol, cfg) * (cfg.contracts or 1)
    except Exception:                                             # noqa: BLE001
        modelled = None

    out = {
        "trades": int(len(trades)),
        "total_costs": float(costs.sum()),
        "gross_pnl": float(gross.sum()),
        "net_pnl": float(net.sum()),
        "cost_per_trade": float(costs.mean()),
        "cost_share_pct": _share(float(gross.sum()), float(costs.sum())),
        "commission": float(fees.sum()) if fees is not None else None,
        "slippage": float(slip.sum()) if slip is not None else None,
        "split_available": split is not None,
        # What the cost model says one round turn should cost, alongside what
        # the run actually charged. The two disagreeing is a spec problem, and
        # a wrong tick size silently rescales every P&L figure for the symbol.
        "modelled_round_turn": modelled,
        "by_year": [],
    }

    year = pd.DatetimeIndex(trades["exit_time"]).year
    for y, g in trades.assign(_y=year).groupby("_y", sort=True):
        gy = float(g["gross_pnl"].astype(float).sum())
        cy = float(g["costs"].astype(float).sum())
        out["by_year"].append({
            "year": int(y), "trades": int(len(g)),
            "gross_pnl": gy, "costs": cy,
            "net_pnl": float(g["pnl"].astype(float).sum()),
            "cost_per_trade": cy / len(g) if len(g) else None,
            "cost_share_pct": _share(gy, cy),
        })
    return out


def format_cost_drag(drag: dict, indent: str = "  ") -> str:
    """The cost block for the console."""
    if not drag or not drag["trades"]:
        return f"{indent}cost drag     : no trades"

    def money(v):
        return "n/a" if v is None else f"{v:,.0f}"

    def pct(v):
        return "n/a" if v is None else f"{v:.1f}%"

    L = [f"{indent}trades        : {drag['trades']:,}",
         f"{indent}gross P&L     : {money(drag['gross_pnl'])}",
         f"{indent}total costs   : {money(drag['total_costs'])}"]
    if drag["split_available"]:
        L.append(f"{indent}  commission  : {money(drag['commission'])}")
        L.append(f"{indent}  slippage    : {money(drag['slippage'])}")
    else:
        L.append(f"{indent}  (commission/slippage split unavailable — the "
                 f"remainder came out\n{indent}   negative or the symbol has no "
                 f"spec, so only the combined figure is real)")
    L.append(f"{indent}net P&L       : {money(drag['net_pnl'])}")
    L.append(f"{indent}cost / trade  : {money(drag['cost_per_trade'])}"
             + (f"   (model says {money(drag['modelled_round_turn'])})"
                if drag["modelled_round_turn"] is not None else ""))
    L.append(f"{indent}cost share    : {pct(drag['cost_share_pct'])} of gross "
             f"profit")
    if drag["cost_share_pct"] is not None and drag["cost_share_pct"] > 50:
        L.append(f"{indent}  [!] over half the gross edge goes to costs. One "
                 f"extra tick of\n{indent}      slippage is the difference "
                 f"between this and nothing.")

    if drag["by_year"]:
        L.append("")
        L.append(f"{indent}{'year':<8}{'trades':>8}{'gross':>13}{'costs':>13}"
                 f"{'net':>13}{'cost share':>13}")
        for r in drag["by_year"]:
            L.append(f"{indent}{r['year']:<8}{r['trades']:>8,}"
                     f"{r['gross_pnl']:>13,.0f}{r['costs']:>13,.0f}"
                     f"{r['net_pnl']:>13,.0f}{pct(r['cost_share_pct']):>13}")
    return "\n".join(L)


def verify_symbol(symbol: str, path: Path, tf: str, params: dict,
                  args: argparse.Namespace, cfg_kwargs: dict,
                  art_dir: Path, strat_name: str) -> dict:
    """One contract over the whole lifecycle. Raises; the caller records it."""
    t0 = time.time()
    print("\n" + "-" * 78)
    print(f"{symbol}  ·  {tf}  ·  full lifecycle")
    print("-" * 78)
    spec = get_spec(symbol)
    print(f"  contract   : x{spec.multiplier:g} per point, tick "
          f"{spec.tick_size:g} (${spec.tick_value:.2f}), "
          f"${spec.commission:.2f}/side")

    bars = load_bars(symbol, tf, args.start, args.end)
    if "symbol" in bars.columns and bars["symbol"].nunique() > 1:
        raise ValueError(f"{symbol}: the lake returned an interleaved frame "
                         f"({bars['symbol'].nunique()} symbols)")
    print(f"  bars       : {len(bars):,}  "
          f"{bars['ts'].iloc[0]} → {bars['ts'].iloc[-1]}")

    cfg = BacktestConfig(
        initial_capital=args.capital, contracts=args.contracts,
        slippage_ticks=args.slippage_ticks, flat_by_close=args.flat_by_close,
        variants_tested=args.variants_tested,
        notes=f"stage 4 full verification {symbol} {tf}", **cfg_kwargs)

    out = run_dual_version_backtest(
        str(path), bars, freq=tf, symbol=symbol, cfg=cfg, params=params,
        threshold=args.threshold, ml=args.ml, emit_reports=False,
        strat_name=strat_name)

    a, b = out["version_a"], out["version_b"]
    metrics_a = a["metrics"]
    metrics_b = b["metrics"] if b else None

    print()
    # No gate table: this window includes the holdout Stage 3 already spent, so
    # a gate verdict computed on it would be a certification of contaminated
    # bars wearing the same badge as a real one.
    print_dual_scorecard(metrics_a, metrics_b, show_gates=False)

    filters = metrics_a.get("entry_filters") or {}
    if filters:
        print()
        print(describe_filters(filters))

    # Run Dynamic Regime Profiler. Version A's result, matching the cost drag
    # and the day-of-week table below it: the profile describes the rule-based
    # baseline, not whatever an ML filter left of it.
    #
    # `a["result"]` is a BacktestResult, NOT a vbt.Portfolio - the engine
    # builds one portfolio per chunk inside `_simulate` and deletes it, so no
    # portfolio object survives the run. RegimeProfiler reads the trade list
    # off either shape.
    #
    # `art_dir.parent` is the stage's own out directory - `pipeline_dir` with
    # `--out-dir` already applied - so the profile lands beside the handoffs
    # rather than under a hardcoded root, and at a STABLE path: the live
    # supervisor reads the latest profile, and burying it in this run's
    # timestamped directory would make it unfindable without one.
    try:
        profiler = RegimeProfiler(bars, a.get("result"), strat_name, symbol, tf,
                                  out_dir=art_dir.parent)
        profiler.generate_profile()
    except Exception as e:                                        # noqa: BLE001
        print(f"[!] Regime Profiler failed for {symbol}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)

    drag = cost_drag(metrics_a.get("trades"), a.get("result"), symbol, cfg)
    print("\n  COST DRAG · Version A")
    print(format_cost_drag(drag))

    dow = day_of_week_breakdown(metrics_a.get("trades"))
    print("\n  DAY OF WEEK · Version A, whole lifecycle")
    print(format_day_of_week(dow))

    # Tear sheets, into a timestamped directory. Evidence is never overwritten:
    # a promotion cites a specific run, and a re-run that replaced it would
    # leave a decision pointing at numbers that are no longer the ones it was
    # made on.
    _fn, run_info = load_strategy(path, params)
    del _fn
    reports, report_error = {}, None
    if not args.no_reports:
        try:
            reports = write_dual_reports(
                out, bars=bars, out_dir=art_dir, strat_name=strat_name,
                indicators=_strategy_indicators(run_info, bars), prefix=symbol)
            print(f"\n  Version A  : {reports['report_version_a']}")
            if reports.get("report_version_b"):
                print(f"  Version B  : {reports['report_version_b']}")
            print(f"  Metrics    : {reports['metrics_json']}")
        except Exception as e:                                    # noqa: BLE001
            report_error = f"{type(e).__name__}: {e}"
            print(f"\n[!] {symbol}: reports were NOT written: {report_error}",
                  file=sys.stderr, flush=True)

    # The trade log as a CSV alongside the HTML. The tear sheet's table caps at
    # MAX_TRADE_ROWS and says so; this file is every trade, for the analysis
    # nobody has thought of yet.
    trade_csv = None
    trades = metrics_a.get("trades")
    if trades is not None and len(trades):
        art_dir.mkdir(parents=True, exist_ok=True)
        trade_csv = art_dir / f"trades_{symbol}_version_a.csv"
        trades.to_csv(trade_csv, index=False)
        print(f"  Trade log  : {trade_csv}  ({len(trades):,} rows)")
        if metrics_b is not None and metrics_b.get("trades") is not None:
            tb = art_dir / f"trades_{symbol}_version_b.csv"
            metrics_b["trades"].to_csv(tb, index=False)
            print(f"               {tb}  ({len(metrics_b['trades']):,} rows)")

    print(f"  ({round(time.time() - t0, 1)}s)")
    return {
        "symbol": symbol,
        "timeframe": tf,
        "params": run_info.get("bound_params") or params,
        "window": {"start": str(bars["ts"].iloc[0]),
                   "end": str(bars["ts"].iloc[-1]),
                   "bars": int(len(bars))},
        "entry_filters": cfg_kwargs,
        "metrics_a": {k: v for k, v in metrics_a.items()
                      if not isinstance(v, (pd.DataFrame, pd.Series))},
        "metrics_b": ({k: v for k, v in metrics_b.items()
                       if not isinstance(v, (pd.DataFrame, pd.Series))}
                      if metrics_b else None),
        "cost_drag": drag,
        "dow_breakdown": dow.to_dict("records"),
        "reports": {k: str(v) for k, v in reports.items()},
        "trade_log": str(trade_csv) if trade_csv else None,
        "report_error": report_error,
        # Said in the file as well as on screen. A JSON with gate-shaped
        # metrics in it and no such flag would be read as a certification by
        # the next thing that parses it.
        "is_certification": False,
        "certification_note": (
            "This window includes the Stage 3 holdout, so these metrics are "
            "in-sample by construction. Gate verdicts come from "
            "gate_audit_<SYMBOL>.json, never from here."),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 4/5 — full-lifecycle run across all available "
                    "data. Tear sheets, cost drag and trade logs. Not a "
                    "certification: this window includes the Stage 3 holdout.")
    p.add_argument("--strat", required=True)
    p.add_argument("--symbols", default=None,
                   help="NQ, a list, or ALL. Default: every contract Stage 2 "
                        "selected parameters for.")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None)
    p.add_argument("--start", default="2010-01-01",
                   help="Lifecycle start (default 2010-01-01)")
    p.add_argument("--end", default="2026-01-01",
                   help="Lifecycle end (default 2026-01-01)")
    p.add_argument("--param", action="append", default=[], metavar="K=V")
    p.add_argument("--defaults", action="store_true",
                   help="Use the module's DEFAULT_PARAMS instead of Stage 2's "
                        "winner")
    p.add_argument("--ml", action="store_true", help="Also run Version B")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--flat-by-close", action="store_true")
    p.add_argument("--variants-tested", type=int, default=None,
                   help="Carried onto the reports. Defaults to the count "
                        "Stage 2 recorded for this contract.")
    p.add_argument("--no-reports", action="store_true",
                   help="Skip the HTML tear sheets")
    p.add_argument("--out-dir", default=None)
    add_filter_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path = resolve_strategy(args.strat)
    strat_name = path.parent.name if path.stem == "strat" else path.stem
    out_dir = pipeline_dir(strat_name, args.out_dir, create=True)
    overrides = dict(parse_param(p) for p in args.param)

    try:
        _fn, info = load_strategy(path, overrides)
        cfg_kwargs = filter_config_kwargs(args)
    except Exception as e:                                        # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # ONE timeframe: a lifecycle run produces one tear sheet per contract, and
    # a comma-separated list here would overwrite each contract's report with
    # the next timeframe's under the same filename.
    tfs = [t.strip() for t in str(args.tf or "").split(",") if t.strip()]
    if len(tfs) > 1:
        print(f"--tf takes ONE timeframe here, got {args.tf!r}. Stage 4 writes "
              f"one tear\nsheet per contract; several timeframes would "
              f"overwrite each other. Run it\nonce per timeframe.",
              file=sys.stderr)
        return 1
    tf = (tfs[0] if tfs else None) or info.get("timeframe") or "15m"

    if args.symbols:
        symbols = parse_symbols(args.symbols, info.get("symbols"))
    else:
        symbols = discover_symbols(out_dir, tf)
        if not symbols:
            print(f"No best_params_*.json in {out_dir} and no --symbols. Run "
                  f"stage 2 first,\nor name the contracts.", file=sys.stderr)
            return 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    art_dir = out_dir / f"verify_{stamp}"

    print(stage_banner(4, strat_name,
                       f"{len(symbols)} contract(s) · {tf} · "
                       f"{args.start} → {args.end}"))
    print("  This run covers the Stage 3 holdout. Its metrics are IN-SAMPLE by")
    print("  construction and certify nothing — no gate table is printed. The")
    print("  certified verdicts are in gate_audit_<SYMBOL>.json from Stage 3.")
    print(f"  artifacts  : {art_dir}")

    rows, errors = [], []
    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")
        try:
            params = dict(overrides)
            variants = args.variants_tested
            # The timeframe-specific winner first, as in stage 3.
            bp = out_dir / BEST_PARAMS_FILE.format(symbol=f"{sym}_{tf}")
            if not bp.exists():
                bp = out_dir / BEST_PARAMS_FILE.format(symbol=sym)
            if not args.defaults and bp.exists():
                blob = read_stage(bp, 2, strat_name)
                params = {**(blob.get("params") or {}), **overrides}
                if variants is None:
                    variants = blob.get("variants_tested")
                print(f"  parameters : {params}  (stage 2 winner, best of "
                      f"{variants})")
            else:
                print(f"  parameters : {params or '(module defaults)'}  "
                      f"({'--defaults' if args.defaults else 'no stage 2 file'})")
            args.variants_tested = variants
            row = verify_symbol(sym, path, tf, params, args, cfg_kwargs,
                                art_dir, strat_name)
            rows.append(row)
            write_stage(out_dir / VERIFY_FILE.format(symbol=sym), 4,
                        strat_name, row)
        except Exception as e:                                    # noqa: BLE001
            errors.append({"symbol": sym, "error": f"{type(e).__name__}: {e}"})
            print(f"\n[!] {sym}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    W = 78
    print("\n" + "=" * W)
    print(f"STAGE 4 RESULT · {len(rows)}/{len(symbols)} contract(s) verified")
    print("=" * W)
    print(f"  {'symbol':<8}{'Sharpe':>8}{'PF':>7}{'maxDD%':>9}{'trades':>9}"
          f"{'cost share':>12}")
    for r in rows:
        m, d = r["metrics_a"], r["cost_drag"]
        share = d["cost_share_pct"]
        print(f"  {r['symbol']:<8}{m.get('sharpe', float('nan')):>8.2f}"
              f"{m.get('profit_factor', float('nan')):>7.2f}"
              f"{m.get('max_drawdown_pct', float('nan')):>9.2f}"
              f"{int(m.get('trade_count', 0)):>9,}"
              f"{('n/a' if share is None else f'{share:.1f}%'):>12}")
    for e in errors:
        print(f"  ERROR {e['symbol']:<6}{e['error']}")
    print(f"\n  artifacts  → {art_dir}")

    sym = rows[0]["symbol"] if rows else (symbols[0] if symbols else "NQ")
    print(next_step([
        "Read the tear sheets, then decide. Stage 5 promotes ONE version and",
        "refuses any whose Stage 3 audit is not PASS:",
        "",
        f"  python3 backtest/promote.py --strat {strat_name} --version A \\",
        f"      --source {path} \\",
        f"      --audit-file {out_dir / f'gate_audit_{sym}.json'} \\",
        f"      --metrics {art_dir / f'dual_metrics_{sym}.json'}",
        "",
        "[1] Promote Version A   [2] Promote Version B",
        "[3] Back to Stage 2     [4] Keep in experimental",
    ]))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
