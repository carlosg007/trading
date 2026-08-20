#!/usr/bin/env python3
"""
The profiler's regime source, on synthetic bars with known answers.

Run as a script; exits non-zero on failure. No lake, no network.

What this pins:
  1. When the bars carry `regime_quadrant`, the profiler USES it - every trade
     is attributed to the quadrant the cache assigned its entry bar, not to one
     the profiler recomputed. Without this the cache is written, joined, and
     ignored, and Stage 1 reports "precomputed" while doing its own pass.
  2. Quadrant 0 (warm-up) is UNKNOWN, so those trades count as unplaced rather
     than being filed under a real regime.
  3. The fallback still works when no column is present, and it is recorded as
     a DIFFERENT source - the two draw their volatility threshold from
     different windows and will disagree.
  4. The integer -> label map matches REGIMES. A transposed map would move
     every trade between quadrants with every total still correct.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402

from backtest import profiler                                       # noqa: E402
from mdlib import regimes                                           # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(label)


def bars_and_trades(n: int = 600):
    ts = pd.date_range("2015-01-05", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0, 1.0, n))
    bars = pd.DataFrame({
        "ts": ts, "symbol": "SYN",
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 500,
    })
    # A deliberate, hand-built quadrant column: blocks of 100 bars cycling
    # 1,2,3,4 after a 40-bar warm-up. Hand-built rather than computed, so the
    # expected attribution below is arithmetic and not a second run of the
    # code under test.
    quad = np.zeros(n, dtype="uint8")
    for i, q in enumerate((1, 2, 3, 4, 1, 2)):
        lo = 40 + i * 90
        quad[lo:lo + 90] = q
    bars["regime_quadrant"] = quad
    bars["adx_14"] = np.float32(30.0)
    bars["atr_14"] = np.float32(2.0)
    bars["is_trending"] = True
    bars["is_high_vol"] = True

    # One trade entering on every 10th bar, alternating win/loss.
    idx = np.arange(5, n - 5, 10)
    trades = pd.DataFrame({
        "entry_time": ts[idx],
        "pnl": np.where(np.arange(len(idx)) % 2 == 0, 100.0, -50.0),
    })
    return bars, trades, quad, idx


def test_uses_precomputed_column() -> None:
    print("\n[1] the cached quadrant column is what gets used")
    bars, trades, quad, idx = bars_and_trades()

    with tempfile.TemporaryDirectory() as tmp:
        prof = profiler.RegimeProfiler(bars, trades, "syn", "SYN", "15m",
                                       out_dir=tmp, version="a", quiet=True
                                       ).generate_profile()

        check("regime_source is precomputed_cache",
              prof.get("regime_source") == "precomputed_cache",
              str(prof.get("regime_source")))

        # Expected counts, derived from the hand-built column by arithmetic.
        entry_quads = quad[idx]
        expected = {}
        for q in (1, 2, 3, 4):
            n = int((entry_quads == q).sum())
            if n:
                expected[profiler.QUADRANT_TO_REGIME[q]] = n
        got = {k: v["trade_count"]
               for k, v in prof["regime_breakdown"].items()}
        check("trade counts per quadrant match the cached column exactly",
              got == expected, f"got={got} expected={expected}")

        # 2: warm-up entries are unplaced, not filed under a regime.
        warm = int((entry_quads == 0).sum())
        check("the fixture has warm-up entries to place", warm > 0, str(warm))
        check("quadrant-0 entries are counted as unplaced",
              prof["trades_unplaced"] == warm,
              f"{prof['trades_unplaced']} vs {warm}")
        check("profiled + unplaced == the whole trade list",
              prof["trades_profiled"] + prof["trades_unplaced"] == len(trades))

        # The profiler must not have run its own indicator pass.
        check("no ADX/ATR columns were appended to the frame",
              not any(c.startswith(("ADX", "ATR")) for c in bars.columns),
              str([c for c in bars.columns if c.startswith(("ADX", "ATR"))]))


def test_fallback_when_absent() -> None:
    print("\n[2] fallback when the bars carry no cached quadrant")
    bars, trades, quad, idx = bars_and_trades()
    plain = bars.drop(columns=["regime_quadrant", "adx_14", "atr_14",
                               "is_trending", "is_high_vol"])

    with tempfile.TemporaryDirectory() as tmp:
        prof = profiler.RegimeProfiler(plain, trades, "syn", "SYN", "15m",
                                       out_dir=tmp, version="a", quiet=True
                                       ).generate_profile()
        check("regime_source is recomputed_live",
              prof.get("regime_source") == "recomputed_live",
              str(prof.get("regime_source")))
        check("it still produces a breakdown",
              len(prof["regime_breakdown"]) > 0)
        check("the fallback records a threshold basis that names the window",
              "handed to the profiler" in
              (prof.get("volatility_threshold_basis") or ""))

        # 3: the two sources genuinely disagree, so recording which ran matters.
        cached = profiler.RegimeProfiler(bars, trades, "syn", "SYN", "15m",
                                         out_dir=tmp, version="b", quiet=True
                                         ).generate_profile()
        check("cached and recomputed attribute trades differently",
              {k: v["trade_count"] for k, v in prof["regime_breakdown"].items()}
              != {k: v["trade_count"]
                  for k, v in cached["regime_breakdown"].items()},
              "identical - the source field would be recording nothing")


def test_map_orientation() -> None:
    print("\n[3] the integer -> label map is not transposed")
    check("quadrant map matches REGIMES in order",
          tuple(profiler.QUADRANT_TO_REGIME[q] for q in (1, 2, 3, 4))
          == profiler.REGIMES)
    check("1 is High Volatility / Trending",
          profiler.QUADRANT_TO_REGIME[1] == "High Volatility / Trending")
    check("4 is Low Volatility / Ranging",
          profiler.QUADRANT_TO_REGIME[4] == "Low Volatility / Ranging")
    check("the map is built from mdlib.regimes, not respelled",
          all(profiler.QUADRANT_TO_REGIME[q] == regimes.QUADRANT_LABELS[q]
              for q in (1, 2, 3, 4)))


def test_screen_threshold() -> None:
    print("\n[4] the survival hurdle is the one the operator asked for")
    from backtest import baseline
    check("MIN_REGIME_PROFIT_FACTOR is 1.00",
          baseline.MIN_REGIME_PROFIT_FACTOR == 1.00,
          str(baseline.MIN_REGIME_PROFIT_FACTOR))
    check("MIN_REGIME_TRADES is 30",
          baseline.MIN_REGIME_TRADES == 30, str(baseline.MIN_REGIME_TRADES))

    # max(PF_A, PF_B) >= 1.00 AND N >= 30, per quadrant, either version.
    R = profiler.REGIMES
    def prof_with(pf, n, regime=R[0]):
        return {"trades_profiled": n,
                "regime_breakdown": {regime: {"profit_factor": pf,
                                              "trade_count": n,
                                              "win_rate": 50.0,
                                              "net_pnl": 1.0}}}

    ok, why, best = baseline.screen({"A": prof_with(1.02, 40), "B": None})
    check("PF 1.02 over 40 trades survives at the 1.00 bar", ok, why)

    ok, _, _ = baseline.screen({"A": prof_with(0.99, 40), "B": None})
    check("PF 0.99 does not survive", not ok)

    ok, _, _ = baseline.screen({"A": prof_with(1.50, 29), "B": None})
    check("PF 1.50 over 29 trades does not survive (trade floor binds)", not ok)

    # Version B carries it when A cannot - the max() in the operator's rule.
    ok, why, best = baseline.screen({"A": prof_with(0.80, 40),
                                     "B": prof_with(1.30, 40)})
    check("Version B alone can carry the quadrant", ok and best["version"] == "B",
          why)

    # Both bars must bind on the SAME quadrant.
    split = {"trades_profiled": 100,
             "regime_breakdown": {R[0]: {"profit_factor": 1.90, "trade_count": 11,
                                         "win_rate": 50.0, "net_pnl": 1.0},
                                  R[1]: {"profit_factor": 0.90, "trade_count": 400,
                                         "win_rate": 50.0, "net_pnl": -1.0}}}
    ok, _, _ = baseline.screen({"A": split, "B": None})
    check("a high PF in one quadrant cannot borrow another's trade count",
          not ok)


if __name__ == "__main__":
    print("=" * 70)
    print("PROFILER · PRE-COMPUTED REGIME INGESTION")
    print("=" * 70)
    test_uses_precomputed_column()
    test_fallback_when_absent()
    test_map_orientation()
    test_screen_threshold()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")
