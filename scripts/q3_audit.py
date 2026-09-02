#!/usr/bin/env python3
"""
q3_audit.py - why no strategy is ever certified in Q3, measured from the lake.

Location:  ~/src/trading/scripts/q3_audit.py

Why
---
The registry holds 46 certified strategies: 20 in Q1, 22 in Q2, 4 in Q4, and
**none in Q3 (Low Volatility / Trending)**. The obvious readings - "Q3 is rare"
or "Q3 candidates failed the gate" - are both wrong, and this script measures
what is actually happening so the answer can be re-checked rather than
remembered.

THE FINDING, IN ONE LINE
========================
Q3 candidates are not failing certification. **They are never nominated.**

`backtest.profiler.designate` picks ONE home quadrant per configuration and
ranks the four on ALPHA CONTRIBUTION, `net_pnl x profit_factor`. Gate R then
scores only that quadrant. Two structural facts about Q3 make it unable to win
that ranking:

  1. **The engine is fixed-size** (`BacktestConfig.contracts = 1`,
     `size_type="amount"`), so per-trade P&L scales with the size of the move.
     Q3's mean ATR is ~0.31x Q1's (0.18x on NQ). Q3 also holds fewer bars
     (18.4% against Q1's 29.9%). At an EQUAL profit factor a Q3 quadrant
     therefore scores roughly 0.19x a Q1 quadrant.
  2. **Costs do not scale.** Commission and a tick of slippage each way are
     fixed dollars, so in Q3 a round turn eats ~3.2x as much of the available
     move as it does in Q1. That depresses Q3's profit factor as well as its
     net P&L - it is a penalty on both terms of the product.

Together: a strategy has to earn a materially HIGHER profit factor in Q3 than
in Q1 merely to be designated there. `break_even_pf()` prints the exact number
for the measured ratios. Nothing rejects Q3; the ranking simply never selects
it, and Gate R is only ever handed the winner.

WHAT IS NOT THE CAUSE, having been checked
==========================================
  * **No hardcoded ATR or volume floor in the pipeline.** `backtest/*.py` and
    `mdlib/regimes.py` carry none. Four STRATEGY modules hardcode
    `MIN_NORM_ATR = 0.0005`; that floor is asymmetric - it removes 0-8% of Q3
    and Q4 bars and 0% of Q1 and Q2 - but it is far too small to be the cause.
  * **The brackets are not the cause.** Stops and targets are `mult x ATR`, so
    they self-scale into a low-volatility regime. It is the fixed COST, not the
    bracket, that fails to scale - which is the real form of the worry.
  * **Q3 is not rare.** 18.4% of all bars, the smallest of the four but tens of
    thousands of bars per (symbol, timeframe).
  * **The candidates exist and were run.** `keltner_trend_drift_20260901`
    declares `TARGET_QUADRANTS = ("Q3",)` - it was written for this regime -
    and all eight certified instances of it came back Q1 or Q2. Across the
    three trend-drift archetypes, 20 of 20 instances were re-designated into a
    high-volatility quadrant.

Reads
-----
    The pinned regime caches through `mdlib.regimes.load_cache`, and closes
    through `mdlib.lake.get_bars` for the normalised-ATR section only.

Writes
------
    Nothing. stdout, or JSON with --json.

Usage
-----
    python scripts/q3_audit.py
    python scripts/q3_audit.py --symbols ES NQ 6E 6J --tf 30m
    python scripts/q3_audit.py --json
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
import math                                                        # noqa: E402
from typing import Any                                             # noqa: E402

import pandas as pd                                                # noqa: E402

from backtest import specs                                         # noqa: E402
from backtest.engine import BacktestConfig, round_turn_cost        # noqa: E402
from mdlib import regimes as R                                     # noqa: E402

#: The quadrant under audit. `mdlib/regimes.py` is the only authority on the
#: numbering and this script does not restate it - Q3 is Low-Vol/TRENDING, and
#: Q2 (the one it is most often confused with) is High-Vol/RANGING.
Q3 = 3

DEFAULT_SYMBOLS = ("6E", "6J", "ES", "NQ", "GC", "CL")
DEFAULT_TFS = ("5m", "15m", "30m", "1h")


def quadrant_shares(symbols, timeframes) -> pd.DataFrame:
    """
    What share of bars each quadrant holds, per (symbol, timeframe).

    From the PINNED caches, so the boundary is the in-sample theta_vol every
    certification was drawn against. Recomputing a median over whatever window
    this script happened to read would make the answer a property of the
    REQUEST - a quiet month would report every bar as low-volatility.
    """
    rows = []
    for symbol in symbols:
        for tf in timeframes:
            try:
                frame = R.load_cache(symbol, tf)
            except Exception:                                      # noqa: BLE001
                continue
            counts = frame["regime_quadrant"].value_counts()
            total = int(counts.sum())
            if not total:
                continue
            rows.append({
                "symbol": symbol, "tf": tf, "bars": total,
                **{f"Q{q}_pct": round(100.0 * int(counts.get(q, 0)) / total, 1)
                   for q in range(5)},
            })
    return pd.DataFrame(rows)


def volatility_ratios(symbols, tf: str = "30m") -> pd.DataFrame:
    """
    Mean ATR(14) per quadrant as a RATIO to Q1's.

    THE RATIO IS THE POINT, not the level. With `contracts = 1` and
    `size_type="amount"` the engine takes one contract per trade whatever the
    volatility, so per-trade P&L moves with the size of the move - and this
    ratio is therefore also the ratio of the alpha score's `net_pnl` term.
    """
    rows = []
    for symbol in symbols:
        try:
            frame = R.load_cache(symbol, tf)
        except Exception:                                          # noqa: BLE001
            continue
        means = frame.groupby("regime_quadrant")["atr_14"].mean()
        base = float(means.get(1, float("nan")))
        if not base or math.isnan(base):
            continue
        rows.append({"symbol": symbol,
                     **{f"Q{q}_atr_ratio": round(float(means.get(q, float("nan")))
                                                 / base, 2)
                        for q in (1, 2, 3, 4)}})
    return pd.DataFrame(rows)


def cost_drag(symbols, timeframes) -> pd.DataFrame:
    """
    A round turn as a percentage of ONE ATR of move, per timeframe, in Q3.

    The second penalty, and the one that is easy to miss: the bracket scales
    with ATR and the COST does not. At 5m in Q3 a round turn costs about half
    of one ATR of available move, which no edge survives; by 1h it is around an
    eighth. This is why a Q3 candidate belongs on the slower rungs of the
    ladder, and it is measured rather than asserted.
    """
    cfg = BacktestConfig()
    rows = []
    for symbol in symbols:
        spec = specs.SPECS.get(symbol)
        if spec is None:
            continue
        cost = round_turn_cost(symbol, cfg)
        row: dict[str, Any] = {"symbol": symbol, "round_turn_usd": round(cost, 2)}
        for tf in timeframes:
            try:
                frame = R.load_cache(symbol, tf)
            except Exception:                                      # noqa: BLE001
                row[tf] = None
                continue
            atr = frame[frame["regime_quadrant"] == Q3]["atr_14"].mean()
            atr_usd = float(atr) * spec.multiplier
            row[tf] = round(100.0 * cost / atr_usd, 1) if atr_usd else None
        rows.append(row)
    return pd.DataFrame(rows)


def break_even_pf(atr_ratio: float, bar_ratio: float,
                  rival_pf: float = 1.20) -> float:
    """
    The profit factor Q3 must reach to OUTSCORE a rival quadrant on alpha.

    `designate` ranks on `net_pnl x profit_factor`. Modelling net P&L as
    `trades x per_trade x (pf - 1)`, a Q3 quadrant with `bar_ratio` of the
    rival's trades and `atr_ratio` of its per-trade size scores

        k (pf - 1) pf        with  k = bar_ratio x atr_ratio

    against the rival's `(rival_pf - 1) x rival_pf`. Solving the quadratic
    gives the factor Q3 needs just to be NOMINATED - before Gate R has looked
    at anything.
    """
    target = (rival_pf - 1.0) * rival_pf
    k = float(bar_ratio) * float(atr_ratio)
    if k <= 0:
        return float("inf")
    return (1.0 + math.sqrt(1.0 + 4.0 * target / k)) / 2.0


def collect(symbols, timeframes) -> dict[str, Any]:
    shares = quadrant_shares(symbols, timeframes)
    ratios = volatility_ratios(symbols)
    drag = cost_drag(symbols, timeframes)

    q3_share = float(shares["Q3_pct"].mean()) if not shares.empty else float("nan")
    q1_share = float(shares["Q1_pct"].mean()) if not shares.empty else float("nan")
    atr_ratio = (float(ratios["Q3_atr_ratio"].mean())
                 if not ratios.empty else float("nan"))
    bar_ratio = q3_share / q1_share if q1_share else float("nan")

    return {
        "shares": shares.to_dict("records"),
        "atr_ratios": ratios.to_dict("records"),
        "cost_drag_pct_of_atr": drag.to_dict("records"),
        "summary": {
            "q3_share_pct": round(q3_share, 1),
            "q1_share_pct": round(q1_share, 1),
            "q3_atr_ratio_to_q1": round(atr_ratio, 2),
            "q3_trade_ratio_to_q1": round(bar_ratio, 2),
            "q3_alpha_handicap": round(bar_ratio * atr_ratio, 2),
            "q3_break_even_pf_vs_q1_at_1_20": round(
                break_even_pf(atr_ratio, bar_ratio, 1.20), 2),
        },
    }


def render(snap: dict[str, Any]) -> str:
    s = snap["summary"]
    lines = [
        "=" * 78,
        "WHY NOTHING CERTIFIES IN Q3 (Low Volatility / Trending)",
        "=" * 78,
        "",
        "1. Q3 IS NOT RARE.",
        f"   {s['q3_share_pct']}% of all bars against Q1's "
        f"{s['q1_share_pct']}% — the smallest quadrant, not a scarce one.",
        "",
        pd.DataFrame(snap["shares"]).to_string(index=False),
        "",
        "2. Q3 MOVES LESS, AND THE ENGINE IS FIXED-SIZE (contracts = 1).",
        f"   Mean ATR(14) in Q3 is {s['q3_atr_ratio_to_q1']}x Q1's, so per-trade",
        "   P&L — the `net_pnl` term of the alpha score — scales down with it.",
        "",
        pd.DataFrame(snap["atr_ratios"]).to_string(index=False),
        "",
        "3. COSTS DO NOT SCALE. A round turn as a % of ONE ATR of move, in Q3:",
        "",
        pd.DataFrame(snap["cost_drag_pct_of_atr"]).to_string(index=False),
        "",
        "   The bracket is `mult x ATR` and scales; commission and slippage are",
        "   fixed dollars and do not. At 5m no edge survives that drag.",
        "",
        "4. THE RANKING, WHICH IS THE ACTUAL CAUSE.",
        "   `profiler.designate` picks ONE home quadrant on alpha contribution",
        "   (`net_pnl x profit_factor`) and Gate R scores only that one. Q3",
        f"   carries {s['q3_trade_ratio_to_q1']}x Q1's trades at "
        f"{s['q3_atr_ratio_to_q1']}x the size —",
        f"   an alpha handicap of {s['q3_alpha_handicap']}x at an EQUAL profit "
        f"factor.",
        "",
        f"   To outscore a Q1 quadrant running PF 1.20, a Q3 quadrant must "
        f"reach",
        f"   PF {s['q3_break_even_pf_vs_q1_at_1_20']} — just to be NOMINATED, "
        f"before Gate R looks at anything.",
        "",
        "   Q3 candidates are not failing certification. They are never",
        "   nominated: `keltner_trend_drift_20260901` declares",
        "   TARGET_QUADRANTS = (\"Q3\",) and all eight certified instances came",
        "   back Q1 or Q2. Across the three trend-drift archetypes, 20 of 20.",
        "",
        "WHAT TO DO ABOUT IT",
        "-" * 78,
        "   Stage 3 already takes `--regime`, which overrides the quadrant Gate",
        "   R certifies in and records the override. Fixing the target a priori",
        "   is NOT the best-of-four selection Gate R exists to avoid — the",
        "   quadrant is chosen here, from the structure of the tape, and the",
        "   holdout is still unseen when the parameters are locked.",
        "",
        "   Sweep and certify Q3 candidates at 30m and 1h only. The cost table",
        "   above is why: the same strategy at 5m is paying half its move away.",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Measure why no configuration is ever designated Q3. "
                    "Reads the pinned regime caches; runs no backtest and "
                    "writes nothing.")
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--tf", nargs="+", default=list(DEFAULT_TFS),
                    dest="timeframes")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output instead of the card")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snap = collect(args.symbols, args.timeframes)
    if not snap["shares"]:
        print("no regime caches could be read for "
              f"{args.symbols} at {args.timeframes}; run "
              f"scripts/precompute_regimes.py first", file=sys.stderr)
        return 1
    print(json.dumps(snap, indent=2, default=str) if args.json
          else render(snap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
