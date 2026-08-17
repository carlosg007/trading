"""
backtest/baseline.py - STAGE 1 of 5: does this idea carry on this contract?

Location: ~/src/trading/backtest/baseline.py

Runs Version A (rules) and, with `--ml`, Version B (the same signals, ML
filtered) on the strategy's DEFAULT parameters, one independent simulation per
contract, and answers one question per symbol: is there anything here at all.
Contracts whose Version A profit factor is below 1.00 are dropped, the rest are
written to `surviving_assets.json`, and Stage 2 sweeps only those.

    python3 backtest/baseline.py --strat ema_trend_filter --symbols ALL --tf 15m \\
        --start 2013-01-01 --end 2022-12-31

Why defaults, and why no gate table
-----------------------------------
**Default parameters, deliberately unswept.** A sweep at this stage would
select the best of N per contract and then screen on the result, which promotes
whichever symbol had the most parameters to hide behind. Running one fixed set
everywhere makes the comparison across contracts a comparison of the
CONTRACTS. The parameters get their turn in Stage 2, on the survivors.

**No Gate 1/2/3 block.** Nothing here is entitled to a gate verdict: the
parameters are unswept, the walk-forward has not run, and the holdout must stay
untouched until Stage 3. Three lines of NOT EVALUATED under an ACCEPTANCE GATES
heading is clutter that teaches a reader to skip the gate table, which is the
one thing they must not do when Stage 3 prints a real one. The scorecard is
rendered with `show_gates=False`; the gates are deferred, not hidden.

**Profit factor is the screen, not Sharpe.** PF < 1.00 means the strategy lost
money gross of nothing - it took in less than it gave back after costs, on this
contract. That is a fact about the symbol. A Sharpe screen at this stage would
drop a contract for a lumpy equity path, which is the complaint Gate 1 stopped
making when its Sharpe threshold was demoted to informational.

**The screen runs on Version A even when `--ml` is on.** Version B refits a
classifier once per completed trade on these same bars; surviving on B would
advance a contract on the fitted version of a comparison the Dual-Version
Mandate says is only settled out-of-sample. B's numbers are printed and
recorded per symbol, and they do not decide survival.

The day-of-week table
---------------------
Every symbol gets P&L, win rate and trade count by weekday, attributed by ENTRY
session (see `backtest.report.day_of_week_breakdown`). It is DESCRIPTIVE. A
losing weekday here is a candidate for `--exclude-days` in a later run, not a
parameter this script applies: excluding the days that lost money in-sample and
re-scoring on those same bars is circular, and the Sharpe it produces is not a
measurement. The suggestion is printed with the trade count beside it so a
reader can see whether the row is an edge or eleven trades.
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
from agents.tier3_workers import load_strategy                     # noqa: E402
from backtest.event_calendar import (add_filter_args, describe_filters,   # noqa: E402
                               filter_config_kwargs)
from backtest.engine import BacktestConfig                         # noqa: E402
from backtest.pipeline import (SURVIVORS_FILE, next_step,          # noqa: E402
                               pipeline_dir, stage_banner, write_stage)
from backtest.report import (day_of_week_breakdown,                # noqa: E402
                             format_day_of_week, losing_weekdays,
                             print_dual_scorecard)
from backtest.run import (load_bars, parse_param, parse_symbols,    # noqa: E402
                          parse_timeframes, resolve_strategy)

# The survival bar. 1.00 is break-even after costs, not a comfort margin - the
# same number Gate 1 binds on, so a contract cannot survive Stage 1 on a profit
# factor Gate 1 would later reject.
MIN_PROFIT_FACTOR = 1.00

# Below this, the day-of-week row is reported but never suggested for exclusion.
DOW_MIN_TRADES = 20


def screen(metrics: dict | None,
           min_profit_factor: float = MIN_PROFIT_FACTOR) -> tuple[bool, str]:
    """
    Did this contract carry the edge? Returns `(survived, reason)`.

    A run that produced no trades is dropped with its own reason rather than
    being folded into "profit factor too low". No trades is a fact about the
    strategy on this contract - the signals never fired - and a reader chasing
    a 0.00 profit factor would go looking for losses that do not exist.
    """
    if not metrics or not metrics.get("ok", True):
        return False, f"run failed: {(metrics or {}).get('error', 'unknown')}"
    n = int(metrics.get("trade_count", 0) or 0)
    if n == 0:
        return False, "no trades - the signals never fired on this contract"
    pf = metrics.get("profit_factor")
    if pf is None or pd.isna(pf):
        # `_profit_factor` returns inf for a run with no losing trades. That is
        # not a missing value, and it survives.
        return False, f"profit factor undefined over {n:,} trades"
    if float(pf) < min_profit_factor:
        return False, f"profit factor {float(pf):.2f} < {min_profit_factor:.2f}"
    return True, f"profit factor {float(pf):.2f} over {n:,} trades"


def _row(symbol: str, metrics_a: dict, metrics_b: dict | None,
         survived: bool, reason: str, dow: pd.DataFrame) -> dict:
    def g(m, k, default=None):
        return None if m is None else m.get(k, default)

    return {
        "symbol": symbol,
        "survived": bool(survived),
        "reason": reason,
        "profit_factor_a": g(metrics_a, "profit_factor"),
        "sharpe_a": g(metrics_a, "sharpe"),
        "max_drawdown_pct_a": g(metrics_a, "max_drawdown_pct"),
        "trades_a": g(metrics_a, "trade_count"),
        "win_rate_a": g(metrics_a, "win_rate"),
        "net_pnl_a": g(metrics_a, "total_pnl"),
        # Blank rather than 0 when Version B never ran. A skipped comparison is
        # not one the baseline won.
        "profit_factor_b": g(metrics_b, "profit_factor"),
        "sharpe_b": g(metrics_b, "sharpe"),
        "trades_b": g(metrics_b, "trade_count"),
        "ml_evaluated": metrics_b is not None,
        "dow_breakdown": dow.to_dict("records"),
        "losing_weekdays": losing_weekdays(dow, DOW_MIN_TRADES),
    }


def run_symbol(symbol: str, path: Path, tf: str, params: dict,
               args: argparse.Namespace, cfg_kwargs: dict) -> dict:
    """One contract, defaults only, no sweep. Raises; the caller records it."""
    t0 = time.time()
    print("\n" + "-" * 78)
    print(f"{symbol}  ·  {tf}")
    print("-" * 78)

    bars = load_bars(symbol, tf, args.start, args.end)
    if "symbol" in bars.columns and bars["symbol"].nunique() > 1:
        raise ValueError(f"{symbol}: the lake returned an interleaved frame "
                         f"({bars['symbol'].nunique()} symbols)")
    print(f"  bars       : {len(bars):,}  "
          f"{bars['ts'].iloc[0]} → {bars['ts'].iloc[-1]}")

    cfg = BacktestConfig(
        initial_capital=args.capital, contracts=args.contracts,
        slippage_ticks=args.slippage_ticks, flat_by_close=args.flat_by_close,
        notes=f"stage 1 baseline {symbol} {tf}", **cfg_kwargs)

    out = run_dual_version_backtest(
        str(path), bars, freq=tf, symbol=symbol, cfg=cfg, params=params,
        threshold=args.threshold, ml=args.ml, emit_reports=False,
        strat_name=path.stem)

    a, b = out["version_a"], out["version_b"]
    metrics_a = a["metrics"]
    metrics_b = b["metrics"] if b else None

    # Stage 1's scorecard: the metrics and the verdict, no gate table.
    print()
    print_dual_scorecard(metrics_a, metrics_b, show_gates=False)

    filters = metrics_a.get("entry_filters") or {}
    if filters:
        print()
        print(describe_filters(filters))

    dow = day_of_week_breakdown(metrics_a.get("trades"))
    print("\n  DAY OF WEEK · Version A, attributed by entry session")
    print(format_day_of_week(dow, DOW_MIN_TRADES))
    losers = losing_weekdays(dow, DOW_MIN_TRADES)
    if losers:
        names = ", ".join(dow.loc[dow["weekday"].isin(losers), "day"])
        print(f"\n  {names} lost money on this contract. That is a CANDIDATE "
              f"for\n  --exclude-days {','.join(str(d) for d in losers)}, not a "
              f"result: excluding the days that lost\n  in-sample and "
              f"re-scoring the same bars proves nothing.")

    survived, reason = screen(metrics_a, args.min_profit_factor)
    print(f"\n  SCREEN     : {'SURVIVES' if survived else 'DROPPED'} — {reason}")
    print(f"  ({round(time.time() - t0, 1)}s)")
    return _row(symbol, metrics_a, metrics_b, survived, reason, dow)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1/5 — baseline survival screen on default "
                    "parameters. Drops contracts whose Version A profit "
                    "factor is below 1.00 and writes the survivors for "
                    "Stage 2.")
    p.add_argument("--strat", required=True, help="Strategy name or path")
    p.add_argument("--symbols", default=None,
                   help="NQ, a list NQ,ES,CL, or ALL")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None,
                   help="Timeframe, or a comma-separated list: "
                        "'--tf 1m,5m,15m,30m' screens each in turn. Derived "
                        "timeframes are aggregated from the 1m parquet by the "
                        "lake reader. Default: the module's, then 15m.")
    p.add_argument("--start", default=None, help="In-sample start, YYYY-MM-DD")
    p.add_argument("--end", default=None, help="In-sample end, YYYY-MM-DD")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="Override a default parameter. Stage 1 runs one fixed "
                        "set across every contract; this changes that set, it "
                        "does not sweep.")
    p.add_argument("--ml", action="store_true",
                   help="Also run Version B. Off by default: the classifier "
                        "refits once per completed trade, and 27 contracts of "
                        "that is hours. Survival is decided on A either way.")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Version B: P(win) at or above which an entry is kept")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--flat-by-close", action="store_true")
    p.add_argument("--min-profit-factor", type=float, default=MIN_PROFIT_FACTOR,
                   help=f"Survival bar (default {MIN_PROFIT_FACTOR:.2f})")
    p.add_argument("--out-dir", default=None,
                   help="Override <BT_ARTIFACTS>/pipeline/<strategy>/")
    add_filter_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path = resolve_strategy(args.strat)
    strat_name = path.parent.name if path.stem == "strat" else path.stem
    params = dict(parse_param(p) for p in args.param)

    try:
        _fn, info = load_strategy(path, params)
    except Exception as e:                                        # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        cfg_kwargs = filter_config_kwargs(args)
    except Exception as e:                                        # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    symbols = parse_symbols(args.symbols, info.get("symbols"))
    try:
        timeframes = parse_timeframes(args.tf, info.get("timeframe"))
    except ValueError as e:
        print(f"ValueError: {e}", file=sys.stderr)
        return 1
    bound = info.get("bound_params") or {}

    print(stage_banner(1, strat_name,
                       f"{len(symbols)} contract(s) × {len(timeframes)} "
                       f"timeframe(s) · {', '.join(timeframes)} · "
                       f"{args.start or 'lake start'} → {args.end or 'lake end'}"))
    print(f"  parameters : {bound or '(module defaults)'}")
    print(f"  screen     : Version A profit factor >= "
          f"{args.min_profit_factor:.2f}")
    print(f"  Version B  : {'evaluated' if args.ml else 'NOT RUN (--ml is off)'}")
    if cfg_kwargs["news_filter"] or cfg_kwargs["exclude_days"]:
        print(f"  filters    : news={cfg_kwargs['news_filter']} "
              f"exclude_days={cfg_kwargs['exclude_days']}")
    if len(timeframes) > 1:
        print(f"  [!] {len(symbols) * len(timeframes)} independent screens. A "
              f"contract that survives on\n      ONE timeframe survives this "
              f"stage — which is a weaker statement than\n      surviving on "
              f"the timeframe you meant, and is why the per-timeframe\n"
              f"      breakdown below is the part to read.")

    # Timeframe-major: a whole timeframe's screen completes before the next
    # starts, so a run killed part way leaves complete timeframes rather than
    # a partial row on each.
    rows, errors = [], []
    for tf in timeframes:
        if len(timeframes) > 1:
            print("\n" + "=" * 78)
            print(f"TIMEFRAME {tf}")
            print("=" * 78)
        for i, sym in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] {sym} · {tf}")
            try:
                row = run_symbol(sym, path, tf, params, args, cfg_kwargs)
                row["timeframe"] = tf
                rows.append(row)
            except Exception as e:                                # noqa: BLE001
                # One bad contract does not end the screen. Recorded as an
                # ERROR row rather than as a symbol that produced nothing:
                # those read identically in a survivor list and mean opposite
                # things.
                errors.append({"symbol": sym, "timeframe": tf,
                               "error": f"{type(e).__name__}: {e}"})
                print(f"\n[!] {sym} {tf}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    by_tf = {}
    for tf in timeframes:
        tf_rows = [r for r in rows if r["timeframe"] == tf]
        by_tf[tf] = {
            "surviving": [r["symbol"] for r in tf_rows if r["survived"]],
            "dropped": [{"symbol": r["symbol"], "reason": r["reason"],
                         "profit_factor": r["profit_factor_a"]}
                        for r in tf_rows if not r["survived"]],
        }

    # The union across timeframes, and it is labelled as one. Stage 2 sweeps
    # each survivor at every requested timeframe anyway, so a contract that
    # cleared on one is worth sweeping; what must not happen is the union
    # being read as "survived at 15m" when it survived at 1m only. `by_timeframe`
    # is where that question is answered.
    survivors = sorted({r["symbol"] for r in rows if r["survived"]})
    dropped = [{"symbol": r["symbol"], "timeframe": r["timeframe"],
                "reason": r["reason"], "profit_factor": r["profit_factor_a"]}
               for r in rows if not r["survived"]]

    out_dir = pipeline_dir(strat_name, args.out_dir, create=True)
    dest = write_stage(out_dir / SURVIVORS_FILE, 1, strat_name, {
        # Singular when one timeframe was screened, so a downstream reader that
        # expects the pre-multi-timeframe shape still finds what it looks for;
        # None when several were, because there is no single answer and a
        # plausible-looking wrong one is worse than an absent one.
        "timeframe": timeframes[0] if len(timeframes) == 1 else None,
        "timeframes": timeframes,
        "by_timeframe": by_tf,
        "surviving_is_union_across_timeframes": len(timeframes) > 1,
        "start": args.start,
        "end": args.end,
        "params": bound,
        "params_source": "module DEFAULT_PARAMS with --param over them",
        "criterion": f"version_a profit_factor >= "
                     f"{args.min_profit_factor:.2f}",
        "ml_evaluated": bool(args.ml),
        "entry_filters": cfg_kwargs,
        "surviving": survivors,
        "dropped": dropped,
        "errors": errors,
        "assets": rows,
    })

    W = 78
    print("\n" + "=" * W)
    print(f"STAGE 1 RESULT · {len(survivors)}/{len(symbols)} contract(s) survived")
    print("=" * W)
    for r in sorted(rows, key=lambda r: (not r["survived"],
                                         -(r["sharpe_a"] or 0))):
        mark = "keep" if r["survived"] else "drop"
        pf = r["profit_factor_a"]
        print(f"  {mark:<5}{r['symbol']:<6}{r['timeframe']:<5}"
              f"PF {('%.2f' % pf) if pf is not None and not pd.isna(pf) else '  n/a':>6}"
              f"   Sharpe {(r['sharpe_a'] if r['sharpe_a'] is not None else float('nan')):>6.2f}"
              f"   {int(r['trades_a'] or 0):>7,} trades   {r['reason']}")
    for e in errors:
        print(f"  ERROR {e['symbol']:<6}{e.get('timeframe', ''):<5}{e['error']}")

    if len(timeframes) > 1:
        print("\n  by timeframe:")
        for tf_, block in by_tf.items():
            kept = block["surviving"]
            print(f"    {tf_:<5}{len(kept)}/{len(symbols)} survived"
                  + (f"  ({', '.join(kept)})" if kept else ""))
        print("\n  The list written to surviving_assets.json is the UNION: a "
              "contract is in it\n  if it survived on ANY of these timeframes. "
              "Read by_timeframe before\n  treating that as a result for the "
              "timeframe you care about.")
    print(f"\n  survivors → {dest}")

    if not survivors:
        print("\n  Nothing survived. That is a result about the idea on these "
              "contracts,\n  not a run to repeat with different parameters "
              "until something does.")

    print(next_step([
        "Stage 2 — sweep the survivors' parameters in-sample:",
        "",
        f"  python3 backtest/scan.py --strat {args.strat} "
        f"--symbols {','.join(survivors) if survivors else '<none survived>'} \\",
        f"      --tf {','.join(timeframes)} "
        f"--start {args.start or '2013-01-01'} "
        f"--end {args.end or '2022-12-31'}",
    ]))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
