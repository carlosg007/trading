#!/usr/bin/env python3
"""
test_engine_vbt.py - does the vectorbt engine compute the same P&L?

Location:  ~/src/trading/tests/test_engine_vbt.py

Run:  python tests/test_engine_vbt.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

What it checks
--------------
1. A hand-computed trade. Five bars, one round trip, every number worked out
   on paper first. If the unit conversions in _cost_arrays are wrong this is
   what catches it.
2. Parity with the pre-vectorbt loop over real lake data and random signals.
   The loop is slow and obviously correct; the vectorbt path has to match it
   trade-for-trade. Run over several symbols with different multipliers and
   tick sizes, because the multiplier cancels in the slippage fraction and a
   sign or factor error would only show up on some contracts.
3. The date-aware tick. ZT's tick halved on 2019-01-13, so slippage per tick
   must differ either side of that date - and the legacy loop, which uses one
   scalar tick, must DISAGREE with the new engine before that date. A test
   that passed here would mean the date-awareness is not wired up.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from backtest.engine import (BacktestConfig, _simulate, _simulate_legacy,
                             round_turn_cost, run_backtest)
from backtest.specs import get_spec

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def bars_from(prices: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame({
        "ts": idx, "symbol": "ES",
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [1000] * len(prices),
    })


# --------------------------------------------------------------------------
def test_hand_computed() -> None:
    """
    Five bars: open = 100, 101, 102, 103, 104.

    Entry signal on bar 0 -> fills at bar 1 open = 101.
    Exit signal on bar 2  -> fills at bar 3 open = 103.

    ES: multiplier 50, tick 0.25, tick value $12.50, commission $2.29/side.
    One contract, one tick of slippage each way.

        gross = (103 - 101) * 50            = $100.00
        slippage = 1 tick each way          = 2 * 12.50 = $25.00
        commission = 2 sides                = 2 *  2.29 =  $4.58
        costs                               = $29.58
        net                                 = $70.42

    Fills should land at 101.25 and 102.75 - a tick against us on both sides.
    """
    print("\n[1] hand-computed trade")
    bars = bars_from([100.0, 101.0, 102.0, 103.0, 104.0])
    entries = pd.Series([True, False, False, False, False])
    exits = pd.Series([False, False, True, False, False])

    cfg = BacktestConfig(contracts=1, slippage_ticks=1.0)
    t = _simulate(bars, entries, exits, "ES", cfg)

    check("one trade produced", len(t) == 1, f"got {len(t)}")
    if len(t) != 1:
        return
    r = t.iloc[0]

    check("entry fills at next bar open (101)", r.entry_price == 101.0, f"{r.entry_price}")
    check("exit fills at next bar open (103)", r.exit_price == 103.0, f"{r.exit_price}")
    check("gross = $100.00", abs(r.gross_pnl - 100.0) < 1e-9, f"{r.gross_pnl:.4f}")
    check("costs = $29.58", abs(r.costs - 29.58) < 1e-9, f"{r.costs:.4f}")
    check("net = $70.42", abs(r.pnl - 70.42) < 1e-9, f"{r.pnl:.4f}")

    # The cost the engine charged must equal the cost model, independently.
    expected = round_turn_cost("ES", cfg) * cfg.contracts
    check("costs == round_turn_cost()", abs(r.costs - expected) < 1e-9,
          f"engine {r.costs:.4f} vs model {expected:.4f}")


def test_slippage_is_tick_size_not_tick_value() -> None:
    """
    The conversion this whole refactor turns on.

    A slippage fraction lives in price space, so it is tick_size/price. Had it
    been built from tick_value (dollars, = multiplier * tick_size) the fill
    would move by multiplier ticks - 50x on ES - and one tick of slippage on
    one ES contract would cost $625 instead of $12.50.
    """
    print("\n[2] slippage uses tick size, not tick value")
    bars = bars_from([100.0, 101.0, 102.0, 103.0, 104.0])
    entries = pd.Series([True, False, False, False, False])
    exits = pd.Series([False, False, True, False, False])

    spec = get_spec("ES")
    zero = _simulate(bars, entries, exits, "ES",
                     BacktestConfig(slippage_ticks=0.0)).iloc[0]
    one = _simulate(bars, entries, exits, "ES",
                    BacktestConfig(slippage_ticks=1.0)).iloc[0]

    delta = one.costs - zero.costs
    check("1 tick round turn costs 2 * tick_value",
          abs(delta - 2 * spec.tick_value) < 1e-9,
          f"{delta:.4f} vs {2 * spec.tick_value:.4f}")
    check("not inflated by the multiplier",
          abs(delta - 2 * spec.tick_value * spec.multiplier) > 1.0,
          f"multiplier-inflated would be {2 * spec.tick_value * spec.multiplier:.2f}")


def test_parity_with_legacy() -> None:
    """vectorbt path vs the old loop, on real bars, across contract types."""
    print("\n[3] parity with the legacy loop on real data")
    from mdlib.lake import get_bars

    rng = np.random.default_rng(20260813)
    for sym in ["ES", "NQ", "ZN", "6E", "GC"]:
        try:
            bars = get_bars([sym], "1d", "2022-01-01", "2024-01-01")
        except Exception as e:
            check(f"{sym}: bars loaded", False, str(e))
            continue
        if bars.empty:
            check(f"{sym}: bars loaded", False, "empty")
            continue

        bars = bars.reset_index(drop=True)
        n = len(bars)
        entries = pd.Series(rng.random(n) < 0.05)
        exits = pd.Series(rng.random(n) < 0.05)

        from backtest.engine import clean_signals
        entries, exits = clean_signals(entries, exits)

        cfg = BacktestConfig(contracts=2, slippage_ticks=1.0)
        new = _simulate(bars, entries, exits, sym, cfg)
        old = _simulate_legacy(bars, entries, exits, sym, cfg)

        if len(new) != len(old):
            check(f"{sym}: same trade count", False, f"vbt {len(new)} vs loop {len(old)}")
            continue
        check(f"{sym}: same trade count", True, f"{len(new)} trades")

        for col, tol in (("gross_pnl", 1e-6), ("costs", 1e-6), ("pnl", 1e-6)):
            diff = float(np.abs(new[col].to_numpy() - old[col].to_numpy()).max()) if len(new) else 0.0
            check(f"{sym}: {col} matches", diff < tol, f"max diff {diff:.10f}")

        same_t = (len(new) == 0) or bool(
            (pd.to_datetime(new["entry_time"], utc=True).to_numpy()
             == pd.to_datetime(old["entry_time"], utc=True).to_numpy()).all())
        check(f"{sym}: entry times match", same_t)


def test_date_aware_tick_zt() -> None:
    """
    ZT's tick halved on 2019-01-13, so slippage per tick halved with it.

    The new engine must charge the old, larger tick before that date and the
    smaller one after. The legacy loop uses one scalar tick for the whole
    backtest, so it must DISAGREE across the boundary - if it agreed, the
    date-aware lookup would not be wired in.
    """
    print("\n[4] date-aware tick on ZT")
    from mdlib.lake import get_bars

    spec = get_spec("ZT")
    cfg = BacktestConfig(contracts=1, slippage_ticks=1.0)

    windows = {"pre-2019": ("2017-01-01", "2018-01-01", 1 / 128),
               "post-2019": ("2020-01-01", "2021-01-01", 1 / 256)}

    for label, (start, end, tick) in windows.items():
        bars = get_bars(["ZT"], "1d", start, end).reset_index(drop=True)
        if bars.empty:
            check(f"ZT {label}: bars loaded", False, "empty")
            continue
        n = len(bars)
        entries = pd.Series([i % 20 == 0 for i in range(n)])
        exits = pd.Series([i % 20 == 10 for i in range(n)])

        new = _simulate(bars, entries, exits, "ZT", cfg)
        if new.empty:
            check(f"ZT {label}: trades produced", False)
            continue

        expected_slip = 2 * spec.multiplier * tick          # both sides
        expected_cost = expected_slip + 2 * spec.commission
        got = float(new["costs"].iloc[0])
        check(f"ZT {label}: cost per trade = ${expected_cost:.4f}",
              abs(got - expected_cost) < 1e-6, f"got ${got:.4f}")

        old = _simulate_legacy(bars, entries, exits, "ZT", cfg)
        agrees = abs(float(old["costs"].iloc[0]) - got) < 1e-6
        if label == "pre-2019":
            check("ZT pre-2019: legacy loop disagrees (date-awareness is live)",
                  not agrees,
                  f"legacy ${float(old['costs'].iloc[0]):.4f} vs vbt ${got:.4f}")
        else:
            check("ZT post-2019: legacy loop agrees", agrees)


def test_end_to_end() -> None:
    """run_backtest still returns a coherent result through the new engine."""
    print("\n[5] end-to-end run_backtest")

    def crossover(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = bars["close"]
        fast, slow = close.rolling(10).mean(), close.rolling(30).mean()
        return (((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False),
                ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False))

    res = run_backtest(["ES"], "1d", crossover, "2022-01-01", "2024-01-01",
                       BacktestConfig(contracts=1, trailing_drawdown_pct=5.0))

    check("trades produced", len(res.trades) > 0, f"{len(res.trades)} trades")
    check("net = gross - costs",
          abs(res.stats["net_pnl"] - (res.stats["gross_pnl"] - res.stats["total_costs"])) < 1e-6)
    check("equity ends at capital + net P&L",
          abs(res.equity.iloc[-1] - (100_000 + res.stats["net_pnl"])) < 1e-6,
          f"{res.equity.iloc[-1]:,.2f}")
    check("costs are positive", res.stats["total_costs"] > 0,
          f"${res.stats['total_costs']:,.2f}")
    print("\n" + res.summary())


if __name__ == "__main__":
    test_hand_computed()
    test_slippage_is_tick_size_not_tick_value()
    test_parity_with_legacy()
    test_date_aware_tick_zt()
    test_end_to_end()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  All checks passed.")
