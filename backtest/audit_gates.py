"""
backtest/audit_gates.py - STAGE 3 of 5: certification. The gates, on the record.

Location: ~/src/trading/backtest/audit_gates.py

The only stage that produces a gate verdict anybody may act on. It runs the
three separate pieces of evidence the gates need and writes one
`gate_audit_<SYMBOL>.json` per contract carrying the official PASS/FAIL that
Stage 5 refuses to promote without.

    python3 backtest/audit_gates.py --strat ema_trend_filter --symbols NQ,ES \\
        --tf 15m --is-start 2013-01-01 --is-end 2022-12-31 \\
        --holdout-start 2023-01-01 --holdout-end 2026-01-01

    Gate 1  in-sample     profit factor, trade count, max drawdown
    Gate 2  robustness    walk-forward efficiency, Monte Carlo 95% drawdown
    Gate 3  OOS holdout   retention of the in-sample result on unseen bars

Each is a separate run, because each needs different bars
---------------------------------------------------------
Gate 1 scores the in-sample window. Gate 2 rolls train/test folds forward
inside that same window and bootstraps the in-sample trade sequence. Gate 3
runs the held-back years once, at the end. A single backtest cannot produce
all three, which is why `run_dual_version_backtest` reports Gates 2 and 3 as
NOT EVALUATED and why that is not a pass.

**The holdout must not overlap the in-sample window, and this refuses to run
when it does.** That check is the one thing in this file that can invalidate
everything else in it: an in-sample window running into the holdout has spent
the holdout before Gate 3 is evaluated, and the retention ratio it computes is
a strategy scored against itself. It is checked before any bars are read, so
the run fails in a second rather than after an hour.

**Parameters come from Stage 2, not from the module.** Without
`best_params_<SYMBOL>.json` this would certify the defaults while the operator
believed it had certified the winner of the sweep. `--param` overrides
explicitly, and whichever source was used is recorded in the audit file next to
`variants_tested` - a certification that cannot say how many variants its
parameters were selected from is not one.

**The walk-forward runs WITHOUT a parameter grid by default.** With fixed
parameters it compares two time periods rather than fitted-versus-unseen, which
`run_walk_forward_analysis` says plainly and which is recorded in the audit as
`wfo_optimized: false`. Pass `--wfo-grid` to re-select per fold from the
module's PARAM_GRID; that is the version that says something about overfitting,
and it costs one full sweep per fold.

Version B
---------
`--ml` certifies the ML-filtered version alongside the baseline, and both
audits are written. The filter is refit inside each window it is scored on, so
Gate 3 for Version B is a genuine out-of-sample test of the pipeline rather
than of one fitted classifier.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                   # noqa: E402
import sys                                                        # noqa: E402
import time                                                       # noqa: E402
import traceback                                                  # noqa: E402
from pathlib import Path                                          # noqa: E402

import pandas as pd                                               # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier1_master import run_dual_version_backtest          # noqa: E402
from agents.tier3_workers import (load_strategy,                   # noqa: E402
                                  run_monte_carlo_simulation,
                                  run_walk_forward_analysis,
                                  trade_returns_from_result)
from backtest.engine import BacktestConfig                         # noqa: E402
from backtest.event_calendar import (add_filter_args,              # noqa: E402
                                     filter_config_kwargs)
from backtest.pipeline import (BEST_PARAMS_FILE, GATE_AUDIT_FILE,  # noqa: E402
                               next_step, pipeline_dir, read_stage,
                               stage_banner, write_stage)
from backtest.report import (FAIL, NOT_EVALUATED, PASS,            # noqa: E402
                             audit_acceptance_gates, criterion_text,
                             day_of_week_breakdown, format_day_of_week)
from backtest.run import (load_bars, parse_param, parse_symbols,    # noqa: E402
                          resolve_strategy)
from backtest.scan import expand_grid                              # noqa: E402


class WindowOverlapError(ValueError):
    """The holdout has already been seen. Nothing downstream can fix that."""


def check_windows(is_start: str | None, is_end: str | None,
                  ho_start: str, ho_end: str) -> None:
    """
    Refuse an in-sample window that runs into the holdout.

    An open-ended in-sample window (`--is-end` omitted) is refused for the same
    reason as an overlapping one: it runs to the end of the lake, which
    includes every holdout bar. There is no safe default here, so there is no
    default.
    """
    if not is_end:
        raise WindowOverlapError(
            "--is-end is required. An in-sample window with no end runs to the "
            "end of the lake, which consumes the holdout and makes Gate 3 a "
            "strategy scored against itself.")
    is_e = pd.Timestamp(is_end)
    ho_s, ho_e = pd.Timestamp(ho_start), pd.Timestamp(ho_end)
    if ho_s >= ho_e:
        raise WindowOverlapError(
            f"the holdout window is empty or inverted: {ho_start} → {ho_end}")
    if is_e >= ho_s:
        raise WindowOverlapError(
            f"the in-sample window ends {is_end}, on or after the holdout "
            f"starts {ho_start}. The holdout has been seen, so Gate 3 would "
            f"measure retention of a result on its own training data. Move "
            f"--is-end back before --holdout-start.")
    if is_start and pd.Timestamp(is_start) >= is_e:
        raise WindowOverlapError(
            f"the in-sample window is empty or inverted: {is_start} → {is_end}")


def load_params(strat_name: str, symbol: str, out_dir: Path,
                overrides: dict, use_defaults: bool) -> tuple[dict, dict]:
    """
    The parameters to certify, and where they came from.

    Returns `(params, provenance)`. Missing Stage 2 output is an ERROR unless
    `--defaults` was passed: certifying the module's defaults while the
    operator believes the sweep's winner was certified is the failure this
    argument exists to make deliberate.
    """
    path = Path(out_dir) / BEST_PARAMS_FILE.format(symbol=symbol)
    if use_defaults or not path.exists():
        if not use_defaults:
            raise FileNotFoundError(
                f"{path} does not exist. It is written by stage 2 "
                f"(backtest/scan.py) — run that first, or pass --defaults to "
                f"certify the module's DEFAULT_PARAMS knowing that is what you "
                f"are certifying.")
        return dict(overrides), {
            "params_source": "module DEFAULT_PARAMS with --param over them",
            "variants_tested": None,
            "scan_selection": None,
        }

    blob = read_stage(path, 2, strat_name)
    params = {**(blob.get("params") or {}), **overrides}
    return params, {
        "params_source": f"stage 2 winner ({path.name})"
                         + (" with --param over it" if overrides else ""),
        "variants_tested": blob.get("variants_tested"),
        "scan_selection": blob.get("selection"),
        "scan_in_sample": blob.get("in_sample"),
    }


def _dual(path: Path, bars: pd.DataFrame, symbol: str, tf: str, params: dict,
          cfg: BacktestConfig, ml: bool, threshold: float,
          strat_name: str) -> dict:
    return run_dual_version_backtest(
        str(path), bars, freq=tf, symbol=symbol, cfg=cfg, params=params,
        threshold=threshold, ml=ml, emit_reports=False, strat_name=strat_name)


def gate2_evidence(path: Path, symbol: str, tf: str, params: dict,
                   cfg: BacktestConfig, is_start: str, is_end: str,
                   result, args: argparse.Namespace,
                   grid: dict | None) -> dict:
    """
    Walk-forward efficiency and the bootstrap drawdown, both from IN-SAMPLE bars.

    The Monte Carlo resamples the in-sample trade sequence rather than the
    holdout's: it asks how bad the drawdown could have been had the same edge
    arrived in a different order, and the holdout has too few trades to answer
    that. The result is an OPTIMISTIC floor on the risk either way - resampling
    with replacement destroys the autocorrelation real losing streaks have.
    """
    out: dict = {}
    start_year = pd.Timestamp(is_start).year if is_start else 2010
    end_year = pd.Timestamp(is_end).year

    param_grid = None
    if grid:
        param_grid = expand_grid(grid)

    wfo = run_walk_forward_analysis(
        path, symbol, train_years=args.wfo_train, test_years=args.wfo_test,
        start_year=start_year, end_year=end_year, params=params,
        param_grid=param_grid, tf=tf, cfg=cfg)
    out["wfo"] = {k: v for k, v in wfo.items() if k != "folds"}
    out["wfo"]["n_folds"] = len(wfo.get("folds") or [])
    out["wfo_optimized"] = bool(param_grid)

    returns = trade_returns_from_result(result)
    if returns is None or len(returns) == 0:
        out["monte_carlo"] = {"ok": False,
                              "error": "no in-sample trades to bootstrap"}
    else:
        out["monte_carlo"] = run_monte_carlo_simulation(
            returns, n_iterations=args.mc_iterations,
            initial_capital=cfg.initial_capital, returns_are_dollars=True,
            seed=args.mc_seed)
    return out


def certify_symbol(symbol: str, path: Path, tf: str, args: argparse.Namespace,
                   cfg_kwargs: dict, out_dir: Path, strat_name: str,
                   grid: dict | None) -> dict:
    """One contract through all three gates. Raises; the caller records it."""
    t0 = time.time()
    print("\n" + "-" * 78)
    print(f"{symbol}  ·  {tf}")
    print("-" * 78)

    overrides = dict(parse_param(p) for p in args.param)
    params, prov = load_params(strat_name, symbol, out_dir, overrides,
                               args.defaults)
    print(f"  parameters : {params or '(module defaults)'}")
    print(f"               ({prov['params_source']})")
    n_variants = prov["variants_tested"]
    print(f"  variants   : "
          f"{n_variants if n_variants is not None else 'NOT RECORDED'}")

    cfg = BacktestConfig(
        initial_capital=args.capital, contracts=args.contracts,
        slippage_ticks=args.slippage_ticks, flat_by_close=args.flat_by_close,
        variants_tested=prov["variants_tested"],
        notes=f"stage 3 audit {symbol} {tf}", **cfg_kwargs)

    # -- Gate 1 evidence: the in-sample run -------------------------------
    print(f"\n  [1/3] in-sample   {args.is_start or 'lake start'} → {args.is_end}")
    is_bars = load_bars(symbol, tf, args.is_start, args.is_end)
    is_dual = _dual(path, is_bars, symbol, tf, params, cfg, args.ml,
                    args.threshold, strat_name)

    # -- Gate 3 evidence: the holdout, run once ---------------------------
    print(f"  [3/3] holdout     {args.holdout_start} → {args.holdout_end}")
    ho_bars = load_bars(symbol, tf, args.holdout_start, args.holdout_end)
    ho_dual = _dual(path, ho_bars, symbol, tf, params, cfg, args.ml,
                    args.threshold, strat_name)

    # -- Gate 2 evidence: walk-forward and bootstrap, in-sample only ------
    print(f"  [2/3] robustness  walk-forward + {args.mc_iterations:,}-path "
          f"bootstrap")
    versions = {}
    for label, key in (("A", "version_a"), ("B", "version_b")):
        block = is_dual.get(key)
        if block is None:
            continue
        rb = gate2_evidence(path, symbol, tf, params, cfg, args.is_start,
                            args.is_end, block["result"], args,
                            grid if args.wfo_grid else None)
        ho_block = ho_dual.get(key) or {}
        audit = audit_acceptance_gates(
            block["metrics"], robustness=rb, holdout=ho_block.get("metrics"),
            version=label, name=strat_name)
        versions[label] = {
            "metrics_in_sample": _scalars(block["metrics"]),
            "metrics_holdout": _scalars(ho_block.get("metrics")),
            "robustness": _scalars_deep(rb),
            "gate_audit": _scalars_deep(audit),
            "dow_in_sample": day_of_week_breakdown(
                block["metrics"].get("trades")).to_dict("records"),
            "dow_holdout": day_of_week_breakdown(
                (ho_block.get("metrics") or {}).get("trades")).to_dict("records"),
        }

    _print_audit(symbol, versions)

    dow_ho = day_of_week_breakdown(
        ((ho_dual.get("version_a") or {}).get("metrics") or {}).get("trades"))
    if not dow_ho.empty:
        print("\n  DAY OF WEEK · Version A, HOLDOUT")
        print(format_day_of_week(dow_ho))

    payload = {
        "symbol": symbol,
        "timeframe": tf,
        "params": params,
        **prov,
        "in_sample": {"start": args.is_start, "end": args.is_end},
        "holdout": {"start": args.holdout_start, "end": args.holdout_end},
        "entry_filters": cfg_kwargs,
        "wfo": {"train_years": args.wfo_train, "test_years": args.wfo_test,
                "optimized": bool(args.wfo_grid)},
        "monte_carlo": {"iterations": args.mc_iterations, "seed": args.mc_seed},
        "versions": versions,
        # The certified verdict per version, lifted out of the nested audit so
        # Stage 5 and a human reading the file see it without traversing.
        "status": {k: v["gate_audit"]["status"] for k, v in versions.items()},
        "passed": {k: bool(v["gate_audit"]["passed"])
                   for k, v in versions.items()},
    }
    dest = write_stage(out_dir / GATE_AUDIT_FILE.format(symbol=symbol), 3,
                       strat_name, payload)
    print(f"\n  audit      → {dest}")
    print(f"  ({round(time.time() - t0, 1)}s)")
    return {"symbol": symbol, "path": dest,
            "status": payload["status"], "passed": payload["passed"]}


def _scalars(metrics: dict | None) -> dict | None:
    """Metrics without the trade frame or the equity series."""
    if not metrics:
        return None
    return {k: v for k, v in metrics.items()
            if not isinstance(v, (pd.DataFrame, pd.Series))}


def _scalars_deep(obj):
    """The same, recursively, so a nested result is JSON-writable."""
    if isinstance(obj, dict):
        return {k: _scalars_deep(v) for k, v in obj.items()
                if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(obj, (list, tuple)):
        return [_scalars_deep(v) for v in obj]
    return obj


def _print_audit(symbol: str, versions: dict) -> None:
    """The gate table in full - every criterion, measured against its bar."""
    W = 78
    print("\n" + "=" * W)
    print(f"GATE CERTIFICATION · {symbol}")
    print("=" * W)
    for label, block in versions.items():
        audit = block["gate_audit"]
        print(f"\n  Version {label}   OVERALL: {audit['status']}")
        for gk in ("gate1", "gate2", "gate3"):
            gate = audit["gates"][gk]
            print(f"    {gate['name']:<32}{gate['status']}")
            for c in gate["checks"]:
                measured, required = criterion_text(c)
                print(f"      {c['label']:<28}{measured:>10}  "
                      f"{required:<15}{c['status']}")
                if c.get("note"):
                    print(f"        ! {c['note']}")
        if audit["status"] == NOT_EVALUATED:
            print("    ⚠ NOT EVALUATED is not a pass. A gate with no evidence "
                  "behind it\n      has not been cleared.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 3/5 — certify Gate 1 (in-sample), Gate 2 "
                    "(walk-forward + Monte Carlo) and Gate 3 (OOS holdout) "
                    "on the parameters Stage 2 selected.")
    p.add_argument("--strat", required=True)
    p.add_argument("--symbols", default=None,
                   help="NQ, a list, or ALL. Default: every contract with a "
                        "best_params file from Stage 2.")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None)
    p.add_argument("--is-start", default="2013-01-01", help="In-sample start")
    p.add_argument("--is-end", default="2022-12-31",
                   help="In-sample end. Required, and must fall before "
                        "--holdout-start.")
    p.add_argument("--holdout-start", default="2023-01-01",
                   help="Holdout start (default 2023-01-01)")
    p.add_argument("--holdout-end", default="2026-01-01",
                   help="Holdout end (default 2026-01-01)")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="Override a parameter from the Stage 2 winner")
    p.add_argument("--defaults", action="store_true",
                   help="Certify the module's DEFAULT_PARAMS instead of "
                        "Stage 2's winner. Say so deliberately: without this, "
                        "a missing best_params file is an error rather than a "
                        "silent fallback.")
    p.add_argument("--ml", action="store_true",
                   help="Also certify Version B (the ML-filtered pipeline)")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--wfo-train", type=int, default=2,
                   help="Walk-forward train window in years (default 2)")
    p.add_argument("--wfo-test", type=int, default=1,
                   help="Walk-forward test window in years (default 1)")
    p.add_argument("--wfo-grid", action="store_true",
                   help="Re-select parameters per fold from the module's "
                        "PARAM_GRID. This is the version of the walk-forward "
                        "that measures overfitting; it costs a full sweep per "
                        "fold.")
    p.add_argument("--mc-iterations", type=int, default=1000)
    p.add_argument("--mc-seed", type=int, default=42)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--flat-by-close", action="store_true")
    p.add_argument("--out-dir", default=None)
    add_filter_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path = resolve_strategy(args.strat)
    strat_name = path.parent.name if path.stem == "strat" else path.stem
    out_dir = pipeline_dir(strat_name, args.out_dir, create=True)

    try:
        check_windows(args.is_start, args.is_end, args.holdout_start,
                      args.holdout_end)
    except WindowOverlapError as e:
        print(f"\nWindowOverlapError: {e}", file=sys.stderr)
        return 2

    try:
        _fn, info = load_strategy(path, dict(parse_param(p) for p in args.param))
        cfg_kwargs = filter_config_kwargs(args)
    except Exception as e:                                        # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if args.symbols:
        symbols = parse_symbols(args.symbols, info.get("symbols"))
    else:
        symbols = sorted(p.stem.replace("best_params_", "")
                         for p in out_dir.glob("best_params_*.json"))
        if not symbols:
            print(f"No best_params_*.json in {out_dir}. Run stage 2 "
                  f"(backtest/scan.py) first,\nor name contracts with "
                  f"--symbols.", file=sys.stderr)
            return 1

    tf = args.tf or info.get("timeframe") or "15m"
    grid = info.get("param_grid") or {}

    print(stage_banner(3, strat_name, f"{len(symbols)} contract(s) · {tf}"))
    print(f"  in-sample  : {args.is_start} → {args.is_end}")
    print(f"  holdout    : {args.holdout_start} → {args.holdout_end}  "
          f"(untouched until now)")
    print(f"  walk-fwd   : {args.wfo_train}y train / {args.wfo_test}y test, "
          f"{'re-optimized per fold' if args.wfo_grid else 'FIXED parameters'}")
    if not args.wfo_grid:
        print("               fixed parameters means the ratio compares two "
              "time periods,\n               not fitted-versus-unseen. "
              "Recorded as wfo_optimized: false.")
    print(f"  Version B  : {'certified' if args.ml else 'NOT RUN (--ml is off)'}")

    results, errors = [], []
    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")
        try:
            results.append(certify_symbol(sym, path, tf, args, cfg_kwargs,
                                          out_dir, strat_name, grid))
        except Exception as e:                                    # noqa: BLE001
            errors.append({"symbol": sym, "error": f"{type(e).__name__}: {e}"})
            print(f"\n[!] {sym}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    W = 78
    print("\n" + "=" * W)
    print("STAGE 3 RESULT · certified verdicts")
    print("=" * W)
    passing = []
    for r in results:
        for ver, status in r["status"].items():
            mark = {PASS: "PASS", FAIL: "FAIL"}.get(status, "NOT EVAL")
            print(f"  {r['symbol']:<6}Version {ver}   {mark:<9}{r['path'].name}")
            if r["passed"][ver]:
                passing.append((r["symbol"], ver))
    for e in errors:
        print(f"  ERROR {e['symbol']:<6}{e['error']}")

    if not passing:
        print("\n  Nothing was certified. A FAIL or a NOT EVALUATED is not a "
              "pass, and\n  Stage 5 will refuse both without --force.")
    else:
        print(f"\n  {len(passing)} certification(s) passed: "
              + ", ".join(f"{s}/{v}" for s, v in passing))

    sym, ver = passing[0] if passing else (symbols[0], "A")
    print(next_step([
        "Stage 4 — the full lifecycle run, for the tear sheets and the cost drag:",
        "",
        f"  python3 backtest/verify_full.py --strat {args.strat} "
        f"--symbols {sym} --tf {tf} \\",
        "      --start 2010-01-01 --end 2026-01-01",
        "",
        "Stage 5 — promote, once a human has read the evidence:",
        "",
        f"  python3 backtest/promote.py --strat {strat_name} --version {ver} \\",
        f"      --source {path} \\",
        f"      --audit-file {out_dir / GATE_AUDIT_FILE.format(symbol=sym)}",
    ]))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
