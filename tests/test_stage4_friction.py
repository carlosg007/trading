#!/usr/bin/env python3
"""
Stage 4's per-quadrant friction breakdown, and the ATR-scaled slippage model.

Location:  ~/src/trading/tests/test_stage4_friction.py

Reads no bars and runs no backtest - every case is a small synthetic fixture
whose answer is known by hand, which is the only way to tell a cost model that
is right from one that merely produces plausible dollars.

What is pinned here, and why each one is a way a number could mislead:

  * Friction is attributed by the trade's ENTRY bar, which is the rule
    `RegimeProfiler` uses for its own breakdown. A second attribution rule
    would put the same trade in Q1 on one artifact and Q2 on the artifact
    beside it, with both tables still summing to the same totals.
  * All four quadrants are always rows. An absent row reads as missing data
    when it means "this strategy never entered a low-volatility range".
  * `cost_share_pct` is None, never 0.0, where gross P&L was not positive -
    `cost_drag`'s rule. A quadrant that lost money gross has no profit for its
    costs to be a share of, and 0% there reads as a quadrant that cost nothing.
  * A trade in NO quadrant (inside the 14-bar ADX/ATR warm-up) is counted and
    named, not quietly left out of a table read as the whole run.
  * The CONSTANT-tick slippage path is bit-identical to the pre-2026-08-24
    engine. Every existing certification was measured on it.
  * A non-finite `slippage_ticks` column RAISES. A NaN there makes the fill
    price NaN and drops the trade from the P&L with nothing raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.engine import BacktestConfig, _cost_arrays           # noqa: E402
from backtest.profiler import REGIMES                              # noqa: E402
from backtest.specs import get_spec                                # noqa: E402
from backtest.verify_full import (attach_atr_slippage,             # noqa: E402
                                  friction_by_regime)

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Six trades over six labelled bars, with the arithmetic done by hand:

        Q1  2 trades  gross 100 + 50 = 150   costs 10 + 10 = 20   -> 13.33%
        Q2  2 trades  gross -40 + 10 = -30   costs  5 +  5 = 10   -> None
        Q3  0 trades                                              -> None
        Q4  1 trade   gross 200              costs 8              ->  4.00%
        and one trade entered on an UNKNOWN (warm-up) bar -> unplaced
    """
    ts = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    bars = pd.DataFrame(
        {"Regime": [REGIMES[0], REGIMES[0], REGIMES[1], REGIMES[1],
                    REGIMES[3], "UNKNOWN"]}, index=ts)
    trades = pd.DataFrame({
        "entry_time": list(ts),
        "gross_pnl": [100.0, 50.0, -40.0, 10.0, 200.0, 999.0],
        "costs": [10.0, 10.0, 5.0, 5.0, 8.0, 7.0],
        "pnl": [90.0, 40.0, -45.0, 5.0, 192.0, 992.0],
    })
    return bars, trades


def test_attribution() -> None:
    print("\nfriction by regime quadrant")
    bars, trades = _fixture()
    f = friction_by_regime(trades, bars)
    by = {r["quadrant"]: r for r in f["by_regime"]}

    check("all four quadrants are rows, traded or not",
          [r["quadrant"] for r in f["by_regime"]] == ["Q1", "Q2", "Q3", "Q4"])

    exp = {"Q1": (2, 150.0, 20.0, 130.0),
           "Q2": (2, -30.0, 10.0, -40.0),
           "Q3": (0, 0.0, 0.0, 0.0),
           "Q4": (1, 200.0, 8.0, 192.0)}
    bad = [q for q, e in exp.items()
           if (by[q]["trades"], by[q]["gross_pnl"], by[q]["costs"],
               by[q]["net_pnl"]) != e]
    check("trades, gross, costs and net are attributed by the ENTRY bar",
          not bad, f"wrong: {bad}")

    check("cost share is measured where gross is positive",
          abs(by["Q1"]["cost_share_pct"] - (100.0 * 20.0 / 150.0)) < 1e-9
          and abs(by["Q4"]["cost_share_pct"] - 4.0) < 1e-9)

    check("cost share is None, never 0.0, where gross is not positive",
          by["Q2"]["cost_share_pct"] is None and by["Q3"]["cost_share_pct"] is None,
          f"Q2={by['Q2']['cost_share_pct']} Q3={by['Q3']['cost_share_pct']}")

    check("a warm-up trade is counted as unplaced, not dropped in silence",
          f["attributed_trades"] == 5 and f["unplaced_trades"] == 1
          and f["unplaced_note"],
          f"attributed={f['attributed_trades']} unplaced={f['unplaced_trades']}")

    check("the rows sum to the attributed trades, and say so",
          sum(r["trades"] for r in f["by_regime"]) == f["attributed_trades"])


def test_degrades_to_a_finding() -> None:
    print("\nunavailable is a finding, not a crash")
    bars, trades = _fixture()

    check("no trades -> available False with a reason",
          friction_by_regime(pd.DataFrame(), bars)["available"] is False)
    check("unlabelled bars -> available False with a reason",
          friction_by_regime(trades, pd.DataFrame())["available"] is False)
    check("no profiler frame at all -> available False",
          friction_by_regime(trades, None)["available"] is False)
    check("an unavailable breakdown still carries four rows",
          len(friction_by_regime(trades, None)["by_regime"]) == 4)

    naive = bars.copy()
    naive.index = naive.index.tz_localize(None)
    f = friction_by_regime(trades, naive)
    check("a tz-naive bar index still matches (not zero everywhere)",
          f["available"] and f["attributed_trades"] == 5,
          f"attributed={f.get('attributed_trades')}")


def _bars(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    px = 4000 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"),
        "open": px, "high": px + 3, "low": px - 3,
        "close": px + rng.normal(0, 1, n), "volume": 1000})


def test_slippage_models() -> None:
    print("\nslippage: constant vs ATR-scaled")
    n = 300
    bars = _bars(n)
    ent = np.zeros(n, bool); ent[10] = True
    exi = np.zeros(n, bool); exi[20] = True
    cfg = BacktestConfig(slippage_ticks=1.0)
    tick = get_spec("NQ").tick_size

    slip, _fees, _size = _cost_arrays(bars.copy(), ent, exi, "NQ", cfg)
    check("the constant path is still ticks x tick SIZE / price",
          np.allclose(slip, tick / bars["open"].to_numpy()))

    b2 = bars.copy()
    rec = attach_atr_slippage(b2, "NQ", mult=0.10, period=14, floor_ticks=1.0)
    s2, _f2, _z2 = _cost_arrays(b2, ent, exi, "NQ", cfg)
    check("an ATR column makes slippage vary per bar",
          s2.min() != s2.max() and np.all(np.isfinite(s2)) and np.all(s2 > 0))
    check("the ATR model never charges less than its floor",
          rec["ticks_min"] >= 1.0 and rec["bars_at_floor"] >= 13,
          f"min={rec['ticks_min']} at_floor={rec['bars_at_floor']}")
    check("the model records what it charged, so two runs are comparable",
          rec["model"] == "atr_scaled" and rec["atr_mult"] == 0.10
          and rec["atr_period"] == 14)

    # The warm-up is the failure that matters: ATR is NaN for its first
    # `period - 1` bars, and a NaN reaching the engine makes the fill price NaN
    # and drops the trade from the P&L with nothing raising.
    check("ATR warm-up bars take the floor rather than a NaN",
          np.all(np.isfinite(b2["slippage_ticks"].to_numpy())))

    b3 = bars.copy(); b3["slippage_ticks"] = np.nan
    try:
        _cost_arrays(b3, ent, exi, "NQ", cfg)
        check("a non-finite slippage column RAISES", False)
    except ValueError:
        check("a non-finite slippage column RAISES", True)

    b4 = bars.copy(); b4["slippage_ticks"] = -1.0
    try:
        _cost_arrays(b4, ent, exi, "NQ", cfg)
        check("a negative slippage column RAISES", False)
    except ValueError:
        check("a negative slippage column RAISES", True)


def test_chunk_boundaries() -> None:
    """
    The column lives on the BARS, so `_simulate`'s `bars.iloc[lo:hi]` slices it
    in step for free. A per-bar array on the CONFIG would have to be sliced by
    hand at every chunk boundary, and a slice that drifted would charge each
    bar another bar's slippage - wrong in every trade, invisible in a total.
    """
    print("\nthe column slices with its own rows")
    n = 300
    bars = _bars(n)
    ent = np.zeros(n, bool); ent[10] = True
    exi = np.zeros(n, bool); exi[20] = True
    cfg = BacktestConfig(slippage_ticks=1.0)
    attach_atr_slippage(bars, "NQ", mult=0.10, period=14)

    whole, _f, _z = _cost_arrays(bars, ent, exi, "NQ", cfg)
    lo, hi = 100, 200
    part, _f2, _z2 = _cost_arrays(bars.iloc[lo:hi], ent[lo:hi], exi[lo:hi],
                                  "NQ", cfg)
    check("a sliced frame charges each bar its OWN slippage",
          np.allclose(part, whole[lo:hi]))


if __name__ == "__main__":
    print("=" * 70)
    print("  STAGE 4 - per-quadrant friction and the slippage model")
    print("=" * 70)
    test_attribution()
    test_degrades_to_a_finding()
    test_slippage_models()
    test_chunk_boundaries()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")
