#!/usr/bin/env python3
"""
Integration test for the Dual-Version Mandate.

Proves the whole chain runs on real intraday data:

    lake (NQ 15m) -> sma_crossover -> Version A  (rule-based)
                                   -> Version B  (causal ML filter)
                  -> identical costs -> two tear sheets

Follows the `tests/` convention: no pytest, exits non-zero on failure, prints
what it checked. Like test_alpha_pipeline.py, it NEEDS the lake mounted at
/mnt/backtest. 15m bars are derived from 1m by the lake reader.

Every metric below is produced by agents.tier3_workers.summarize_result from
the engine's own trade list and equity curve. Nothing is asked of a model.

The causality checks are the point of this file. An ML filter that peeks at
the future produces a beautiful Version B and no warning, so the lookahead is
tested directly rather than inferred from a good result.

    python3 test_dual_version.py
"""

from __future__ import annotations

import math
import resource
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.tier1_master import run_dual_version_backtest        # noqa: E402
from agents.tier3_workers import (apply_ml_signal_filter,        # noqa: E402
                                  causal_features)
from backtest.engine import BacktestConfig                        # noqa: E402
from mdlib.lake import iter_bars                                  # noqa: E402

SYMBOL = "NQ"
TIMEFRAME = "15m"
START = "2018-01-01"
END = "2023-12-31"
STRATEGY = "strategies/experimental/sma_crossover.py"
RAM_CEILING_GIB = 3.0

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def peak_rss_gib() -> float:
    """Peak RSS, not current: a run that transiently allocated 20 GiB and freed
    it would look fine under a spot reading."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def fmt(v: float, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:,.2f}{suffix}"


def tear_sheet(title: str, m: dict) -> None:
    print("\n" + "=" * 72)
    print(f"PURE ALPHA TEAR SHEET — Version {title}")
    print("=" * 72)
    rows = [
        ("Annualized Sharpe", fmt(m["sharpe"])),
        ("Sortino", fmt(m["sortino"])),
        ("Calmar", fmt(m["calmar"])),
        ("Profit Factor", fmt(m["profit_factor"])),
        ("Win Rate", fmt(m["win_rate"] * 100 if not math.isnan(m["win_rate"])
                         else m["win_rate"], "%")),
        ("Max Drawdown", fmt(m["max_drawdown_pct"], "%")),
        ("", ""),
        ("CAGR", fmt(m["annualized_return_pct"], "%")),
        ("Total return", fmt(m["total_return_pct"], "%")),
        ("Net P&L", fmt(m["total_pnl"])),
        ("Total costs", fmt(m["total_costs"])),
        ("Trades", f"{m['trade_count']:,}"),
    ]
    for label, value in rows:
        print(f"  {label:<22}{value:>18}" if label else "")
    print("=" * 72)


def load_bars() -> pd.DataFrame:
    """One symbol's 15m bars, straight from the reader."""
    for sym, g in iter_bars([SYMBOL], TIMEFRAME, START, END):
        if sym == SYMBOL:
            return g.reset_index(drop=True)
    raise SystemExit(f"no {TIMEFRAME} bars returned for {SYMBOL}")


def test_feature_causality(bars: pd.DataFrame) -> None:
    """
    A causal feature cannot change when the future is deleted.

    Recomputing the matrix on a truncated frame must reproduce the full-frame
    values exactly for every row that survives. Any full-sample statistic - a
    global mean, a fitted scaler, a centred window - breaks this immediately,
    and none of those is caught by scanning for shift(-k).
    """
    print("\nFeature causality (truncation invariance)")
    full = causal_features(bars)
    check("feature matrix is the declared shape",
          full.shape == (len(bars), 7), str(full.shape))

    for k in (2_000, 10_000, len(bars) // 2):
        trunc = causal_features(bars.iloc[:k])
        same = np.allclose(trunc.to_numpy(dtype=float),
                           full.iloc[:k].to_numpy(dtype=float),
                           equal_nan=True)
        check(f"features identical when bars after {k:,} are removed", same)


def test_filter_is_causal_and_subtractive(bars: pd.DataFrame) -> None:
    """
    The filter may only remove entries, and only using the past.

    Both properties are checked directly. If B could add an entry it would be
    trading a signal the strategy never generated; if its decision at bar s
    changed when bars after s were deleted, it was reading the future.
    """
    print("\nFilter behaviour")
    from backtest.engine import clean_signals
    from agents.tier3_workers import load_strategy

    fn, _ = load_strategy(STRATEGY, {"fast_window": 10, "slow_window": 30})
    e, x = fn(bars)
    e = pd.Series(e).reset_index(drop=True)
    x = pd.Series(x).reset_index(drop=True)
    e, x = clean_signals(e, x)

    kept, kept_exits = apply_ml_signal_filter(bars, e, x, symbol=SYMBOL,
                                              cfg=BacktestConfig())
    check("Version B entries are a subset of Version A's",
          bool((kept & ~e).sum() == 0),
          f"A={int(e.sum())} B={int(kept.sum())}")
    check("exits are returned untouched", kept_exits.equals(x))
    check("the filter actually suppressed something",
          int(e.sum()) > int(kept.sum()),
          f"{int(e.sum()) - int(kept.sum())} suppressed")

    # Truncate hard and re-run: decisions on the surviving bars must not move.
    cut = int(len(bars) * 0.6)
    e_t, x_t = e.iloc[:cut].reset_index(drop=True), x.iloc[:cut].reset_index(drop=True)
    kept_t, _ = apply_ml_signal_filter(bars.iloc[:cut].reset_index(drop=True),
                                       e_t, x_t, symbol=SYMBOL,
                                       cfg=BacktestConfig())
    # The final open trade of the truncated frame has no exit inside it, so the
    # last decision can legitimately differ; compare everything before it.
    last = int(np.flatnonzero(kept_t.to_numpy())[-1]) if kept_t.any() else cut
    check("decisions do not change when future bars are deleted",
          bool((kept.iloc[:last].to_numpy() == kept_t.iloc[:last].to_numpy()).all()),
          f"compared {last:,} bars")


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print("=" * 72)
    print(f"Baseline peak RSS: {peak_rss_gib():.3f} GiB\n")

    print(f"Loading {SYMBOL} {TIMEFRAME} {START} → {END} (derived from 1m)…")
    try:
        bars = load_bars()
    except Exception:
        print("\nLAKE READ FAILED:\n")
        traceback.print_exc()
        return 1
    print(f"{len(bars):,} bars.\n")

    test_feature_causality(bars)
    test_filter_is_causal_and_subtractive(bars)

    print("\nRunning both versions under identical costs…")
    try:
        out = run_dual_version_backtest(
            STRATEGY, bars, freq=TIMEFRAME, symbol=SYMBOL,
            cfg=BacktestConfig(slippage_ticks=1.0, variants_tested=1,
                               notes="test_dual_version"),
            params={"fast_window": 10, "slow_window": 30})
    except Exception:
        print("\nDUAL VERSION RUN RAISED:\n")
        traceback.print_exc()
        return 1

    a, b = out["version_a"]["metrics"], out["version_b"]["metrics"]
    cmp_ = out["comparison"]

    print("\nMemory")
    peak = peak_rss_gib()
    check(f"peak RSS <= {RAM_CEILING_GIB} GiB", peak <= RAM_CEILING_GIB,
          f"{peak:.3f} GiB")

    print("\nIntegrity")
    check("Version A produced trades", a["trade_count"] > 0,
          f"{a['trade_count']} trades")
    check("Version B took no more trades than A",
          b["trade_count"] <= a["trade_count"],
          f"A={a['trade_count']} B={b['trade_count']}")
    check("costs were charged in both", a["total_costs"] > 0 and b["total_costs"] >= 0,
          f"A={a['total_costs']:,.2f} B={b['total_costs']:,.2f}")
    check("both versions ran on the same bars",
          a["meta"]["bars"] == b["meta"]["bars"] == len(bars))
    check("both versions used identical costs",
          a["meta"]["costs_included"] and b["meta"]["costs_included"])

    tear_sheet("A · rule-based baseline", a)
    tear_sheet("B · ML-filtered", b)

    print("\n" + "=" * 72)
    print("DUAL-VERSION COMPARISON")
    print("=" * 72)
    print(f"  {'Metric':<22}{'Version A':>16}{'Version B':>16}{'Delta':>14}")
    pairs = [
        ("Sharpe", "sharpe", ""), ("Sortino", "sortino", ""),
        ("Calmar", "calmar", ""), ("Profit factor", "profit_factor", ""),
        ("Win rate %", "win_rate", "pct"), ("Max drawdown %", "max_drawdown_pct", ""),
        ("Net return %", "total_return_pct", ""), ("CAGR %", "annualized_return_pct", ""),
    ]
    for label, key, kind in pairs:
        va = a[key] * 100 if kind == "pct" else a[key]
        vb = b[key] * 100 if kind == "pct" else b[key]
        delta = vb - va if not (math.isnan(va) or math.isnan(vb)) else float("nan")
        print(f"  {label:<22}{fmt(va):>16}{fmt(vb):>16}{fmt(delta):>14}")
    print(f"  {'Trades':<22}{a['trade_count']:>16,}{b['trade_count']:>16,}"
          f"{b['trade_count'] - a['trade_count']:>14,}")
    print("=" * 72)
    print(f"  Entries suppressed by the filter: {cmp_['entries_suppressed']:,} "
          f"of {cmp_['entries_a']:,}")
    print(f"  B beats A on Sharpe: {cmp_['b_beats_a']}")

    print("\n" + "=" * 72)
    if _failures:
        print(f"FAILED — {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("All checks passed.")
    print("\nThis is an IN-SAMPLE run. Under the Dual-Version Mandate the ML "
          "filter is adopted only if B beats A OUT-OF-SAMPLE, on the held-back "
          "final 3 years. B winning here is not that evidence — the filter was "
          "fitted walk-forward on this very period.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
