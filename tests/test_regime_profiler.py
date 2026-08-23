"""
tests/test_regime_profiler.py - the Stage 1 regime seam, end to end.

Location: ~/src/trading/tests/test_regime_profiler.py

Covers the join that no other suite exercises:

    backtest/profiler.py   RegimeProfiler.generate_profile on REAL bars
    backtest/baseline.py   screen / best_quadrant / kill_switch_regimes,
                           driven by the profiler's own output

`tests/test_pipeline_filters.py` already pins the survival RULE, but it feeds
`screen()` hand-built profile dicts. That leaves the seam between the two
modules untested in both directions: a profiler that stopped placing trades in
quadrants, or a breakdown whose keys drifted from `REGIMES`, would return an
empty screen and read as a strategy with no edge anywhere. Nothing would raise,
and Stage 1 would simply drop every contract.

The checks that matter most, and why they are here rather than assumed:

  * **Every closed trade lands somewhere, or is COUNTED as unplaced.** The four
    printed rows are read as the whole run, so `profiled + unplaced` must equal
    the trade list. A trade silently in no bucket shrinks a quadrant's count
    toward the 30-trade floor from below.
  * **The version suffix is on the artifact path.** Stage 1 profiles both
    versions of one configuration; without the suffix Version B overwrites
    Version A at a path named only for the contract.
  * **A no-trade run yields an EMPTY kill switch, never all four regimes.**
    "Stand down everywhere" is a live-trading instruction, and deriving one
    from a strategy that never traded is the failure the profiler exists to
    refuse quietly.
  * **The two modules agree on the four regime NAMES.** Both sides import
    `REGIMES`, and this asserts the breakdown keys are drawn from it - a second
    spelling would make a kill switch name a quadrant no table ever prints.

Runs without the lake and without a network: the bars are synthetic and the
trade list is built by hand in the engine's `BacktestResult` shape. `pandas-ta`
is required (pinned in `requirements.txt`), since the profiler classifies bars
with ADX(14) and ATR(14).

    python tests/test_regime_profiler.py     # script, exits non-zero on failure
    pytest tests/test_regime_profiler.py     # same checks, real assertions
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    """
    Print the check, then RAISE when it fails.

    Deliberately different from the sibling suites, which only collect the
    failure and let the function run on. Those are script-only, and under
    pytest a collector-style `check` reports a green test for a failed check -
    the whole suite passes while pinning nothing. Raising means pytest sees a
    real failure; `main()` catches it per test function so the script form
    still reports every section rather than stopping at the first.
    """
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
        raise AssertionError(f"{label}" + (f" [{detail}]" if detail else ""))
    return True


# --------------------------------------------------------------------------
# Fixtures. A frame with two genuinely different halves, so the ADX/ATR
# classification has something to separate: a smooth high-range trend, then a
# tight random chop. The exact quadrant each half lands in is NOT asserted -
# that is pandas-ta's business, and pinning it here would make this suite fail
# on an indicator revision that changed nothing about the seam under test.
# --------------------------------------------------------------------------
def synthetic_bars(n: int = 1200) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(7)
    half = n // 2
    trend = np.concatenate([np.linspace(0, 400, half),
                            400 + rng.normal(0, 1.0, n - half).cumsum()])
    close = 15_000 + trend
    band = np.concatenate([np.full(half, 20.0), np.full(n - half, 2.0)])
    return pd.DataFrame({"ts": ts, "symbol": "NQ", "open": close,
                         "high": close + band, "low": close - band,
                         "close": close, "volume": 1000})


class _Result:
    """
    The engine's `BacktestResult` as the profiler reads it.

    A stand-in rather than a real run: `_closed_trades` takes the `trades`
    frame off the result and nothing else, and building one through the engine
    would need the lake for no extra coverage of this seam.
    """

    def __init__(self, trades: pd.DataFrame):
        self.trades = trades


def synthetic_trades(bars: pd.DataFrame, n_win: int = 60,
                     n_loss: int = 60) -> pd.DataFrame:
    """Winners in the trending half, losers in the chop, engine column names."""
    ts = pd.DatetimeIndex(bars["ts"])
    entries = list(ts[100:100 + n_win * 5:5]) + list(ts[700:700 + n_loss * 5:5])
    pnl = [120.0] * n_win + [-40.0] * n_loss
    return pd.DataFrame({"entry_time": entries, "exit_time": entries,
                         "pnl": pnl, "direction": "long"})


def _profile(bars, trades, symbol="NQ", tf="15m", version="a", out_dir=None):
    from backtest.profiler import RegimeProfiler
    return RegimeProfiler(bars, _Result(trades), "regime_smoke", symbol, tf,
                          out_dir=out_dir, version=version,
                          quiet=True).generate_profile()


# --------------------------------------------------------------------------
# 1. The profiler on real bars
# --------------------------------------------------------------------------
def _check_generate_profile(tmp_dir: str | None = None) -> None:
    print("\n1. RegimeProfiler.generate_profile on synthetic bars")
    from backtest.profiler import REGIMES

    bars = synthetic_bars()
    trades = synthetic_trades(bars)
    prof = _profile(bars, trades, out_dir=tmp_dir)

    breakdown = prof["regime_breakdown"]
    check("every breakdown key is one of the four declared REGIMES",
          set(breakdown).issubset(set(REGIMES)), str(list(breakdown)))
    check("no closed trade is silently dropped from the breakdown",
          prof["trades_profiled"] + prof["trades_unplaced"] == len(trades),
          f"{prof['trades_profiled']} placed + {prof['trades_unplaced']} "
          f"unplaced of {len(trades)}")
    check("the artifact carries the VERSION suffix, so B cannot overwrite A",
          str(prof["artifact"]).endswith("regime_profile_NQ_15m_version_a.json")
          and os.path.exists(prof["artifact"]), str(prof["artifact"]))
    check("the optimal quadrant carries its OWN pf and trade count",
          prof["optimal_profit_factor"] is not None
          and prof["optimal_trade_count"] >= 0,
          f"{prof['optimal_regime']} pf={prof['optimal_profit_factor']} "
          f"n={prof['optimal_trade_count']}")
    check("the kill switch is the other THREE quadrants",
          sorted(prof["kill_switch_conditions"])
          == sorted(r for r in REGIMES if r != prof["optimal_regime"]),
          str(prof["kill_switch_conditions"]))
    check("a version B profile writes to its own path",
          str(_profile(bars, trades, version="b", out_dir=tmp_dir)["artifact"]
              ).endswith("_version_b.json"))


# --------------------------------------------------------------------------
# 2. The screen, driven by the profiler's own output rather than a fixture
# --------------------------------------------------------------------------
def _check_screen_consumes_a_real_profile(tmp_dir: str | None = None) -> None:
    print("\n2. baseline.screen on a REAL profiler dict")
    from backtest.baseline import screen

    bars = synthetic_bars()
    prof = _profile(bars, synthetic_trades(bars), out_dir=tmp_dir)

    kept, why, best = screen({"A": prof, "B": None}, 1.15, 30)
    check("a profitable configuration survives on Version A", kept, why)
    check("the winning quadrant is named with the version that produced it",
          best is not None and best["version"] == "A"
          and best["regime"] in prof["regime_breakdown"], why)
    check("BOTH bars are cleared in the SAME quadrant",
          best["profit_factor"] >= 1.15 and best["trade_count"] >= 30,
          f"pf={best['profit_factor']} n={best['trade_count']}")
    check("the screen's quadrant matches the breakdown row it came from",
          prof["regime_breakdown"][best["regime"]]["trade_count"]
          == best["trade_count"])


# --------------------------------------------------------------------------
# 3. The no-trade run - the shape that must not become a stand-down everywhere
# --------------------------------------------------------------------------
def _check_no_trades(tmp_dir: str | None = None) -> None:
    print("\n3. a configuration that never traded")
    from backtest.baseline import screen

    bars = synthetic_bars()
    empty = _profile(bars, synthetic_trades(bars).iloc[0:0], tf="1h",
                     version="b", out_dir=tmp_dir)

    check("a no-trade run returns a dict rather than raising",
          isinstance(empty, dict) and empty["trades_profiled"] == 0)
    check("...and writes NO artifact", empty["artifact"] is None)
    check("...and its kill switch is EMPTY, never all four regimes",
          empty["kill_switch_conditions"] == [],
          str(empty["kill_switch_conditions"]))

    kept, why, best = screen({"A": empty, "B": None}, 1.15, 30)
    check("...so the screen DROPS it, naming the cause",
          not kept and best is None and "no trades" in why, why)


# --------------------------------------------------------------------------
# 4. Either version may carry a configuration
# --------------------------------------------------------------------------
def _check_version_b_alone(tmp_dir: str | None = None) -> None:
    print("\n4. Version B alone can carry a configuration")
    from backtest.baseline import screen

    bars = synthetic_bars()
    good = _profile(bars, synthetic_trades(bars), out_dir=tmp_dir)
    empty = _profile(bars, synthetic_trades(bars).iloc[0:0], tf="1h",
                     version="b", out_dir=tmp_dir)

    kept, why, best = screen({"A": empty, "B": good}, 1.15, 30)
    check("a Version A with no trades does not veto a Version B that cleared",
          kept and best["version"] == "B", why)
    check("a skipped Version B (None) is not read as a failed one",
          screen({"A": good, "B": None}, 1.15, 30)[0], "A still survives")


# --------------------------------------------------------------------------
# pytest entry points. `BT_ARTIFACTS` is redirected per test so nothing here
# ever writes to the NFS mount at /mnt/backtest.
#
# The sections above are `_check_*`, NOT `test_*`, and the underscore is the
# whole point: pytest collects any module-level `test_*` it can call, and these
# take a defaulted argument, so under their old names pytest ran them BOTH
# directly and through these wrappers. The direct call skips the redirect below
# and writes real regime profiles into /mnt/backtest/artifacts/pipeline/ - a
# test suite quietly depositing artifacts on the NFS mount beside the ones a
# promotion cites. Renaming them back reintroduces exactly that.
# --------------------------------------------------------------------------
def _check_true_home_regime(tmp_dir: str | None = None) -> None:
    """
    TRUE HOME REGIME DISCOVERY — the designation rule, on hand-built
    breakdowns where the right answer is arithmetic rather than a simulation.

    Every check here is a case where ranking on profit factor alone gives the
    WRONG quadrant, which is what the rule was changed on 2026-08-21 to stop.
    """
    print("\n5. designate(): alpha contribution, the sample floor, secondaries")
    from backtest.profiler import (DESIGNATION_MIN_TRADES, REGIMES,
                                   SCORE_PF_CEILING, designate,
                                   designation_floor, quadrant_score,
                                   rank_quadrants)

    def q(pf, n, net, win=50.0):
        return {"profit_factor": pf, "trade_count": n, "net_pnl": net,
                "win_rate": win}

    check("the score is net P&L x profit factor",
          quadrant_score(q(1.28, 120, 12_000.0)) == 12_000.0 * 1.28,
          str(quadrant_score(q(1.28, 120, 12_000.0))))
    check("a missing term scores None, never 0.0 — which is a score a "
          "quadrant can legitimately have",
          quadrant_score({"trade_count": 40}) is None
          and quadrant_score(q(1.2, 40, 0.0)) == 0.0)

    check(f"the floor is the LARGER of {DESIGNATION_MIN_TRADES} and 10%",
          (designation_floor(120), designation_floor(5000),
           designation_floor(0)) == (50, 500, 50),
          f"{designation_floor(120)} {designation_floor(5000)}")
    check("...and a fraction of 0 reduces it to the flat count, which is the "
          "pre-2026-08-21 behaviour",
          designation_floor(5000, 30, 0.0) == 30)
    check("the share ROUNDS UP — 10% of 761 is 77, not 76",
          designation_floor(761) == 77, str(designation_floor(761)))

    # The engine beats the corner. Under the old profit-factor rank this
    # designated Q3, and Gate R then certified in a quadrant holding a
    # handful of holdout trades while the money was made in Q1.
    d = designate({REGIMES[0]: q(1.28, 120, 12_000.0),
                   REGIMES[2]: q(1.55, 45, 4_500.0)}, 165)
    check("the ALPHA ENGINE is designated, not the sharpest per-trade edge",
          d["primary"]["quadrant"] == "Q1", str(d["primary"]))
    check("...and the runner-up is a SECONDARY carrying why it was not "
          "designated",
          [x["quadrant"] for x in d["secondaries"]] == ["Q3"]
          and "sample floor" in d["secondaries"][0]["reason"],
          str(d["secondaries"]))

    # A quadrant that LOST money is never a home regime, whatever its factor
    # rounds to: net_pnl x PF is only monotone above zero, and at PF 0.00 the
    # product is exactly 0.0 and would outrank every losing quadrant.
    d = designate({REGIMES[0]: q(0.0, 80, -9_000.0),
                   REGIMES[1]: q(0.5, 80, -5_000.0)}, 160)
    check("a total-loss quadrant (PF 0.00) does not score 0.0 into first "
          "place — nothing is designated at all",
          d["primary"] is None, str(d["primary"]))
    check("...and the reason names the bar that failed, not a trade count",
          "not positive" in d["reason"], d["reason"])

    # The 999 sentinel is "gross loss was zero", not a measured factor.
    d = designate({REGIMES[0]: q(1.30, 4000, 900_000.0),
                   REGIMES[3]: q(999, 60, 9_000.0)}, 4060)
    check("an unbeaten 60-trade quadrant cannot outrank a 4,000-trade engine "
          "on a sentinel",
          d["primary"]["quadrant"] == "Q1", str(d["primary"]))
    capped = {r["quadrant"]: r["pf_capped"] for r in d["scores"]}
    check(f"...the cap at {SCORE_PF_CEILING:.0f} is RECORDED on the row it "
          f"bit, and the reported profit factor is left as measured",
          capped["Q4"] is True and capped["Q1"] is False
          and next(r for r in d["scores"]
                   if r["quadrant"] == "Q4")["profit_factor"] == 999,
          str(capped))

    # Every quadrant comes back, including the ones that lost. A ranking that
    # dropped them makes "disqualified on sample size" and "never traded"
    # the same absent row.
    rows = rank_quadrants({REGIMES[0]: q(1.4, 90, 9_000.0),
                           REGIMES[1]: q(0.4, 300, -3_000.0)}, floor=50)
    check("every quadrant present in the breakdown is scored and returned",
          [r["quadrant"] for r in rows] == ["Q1", "Q2"], str(rows))
    check("...an absent quadrant is absent, not invented as a zero row",
          all(r["quadrant"] not in ("Q3", "Q4") for r in rows))

    # An exact score tie resolves on evidence, not on declaration order.
    d = designate({REGIMES[0]: q(1.20, 100, 6_000.0),
                   REGIMES[3]: q(1.20, 300, 6_000.0)}, 400)
    check("an exact tie on score breaks on the LARGER trade count",
          d["primary"]["quadrant"] == "Q4", str(d["primary"]["quadrant"]))

    print("\n5b. the artifact carries the designation and what it beat")
    bars = synthetic_bars()
    prof = _profile(bars, synthetic_trades(bars), out_dir=tmp_dir)
    for key in ("optimal_quadrant", "optimal_score", "optimal_net_pnl",
                "regime_scores", "secondary_regimes", "designation"):
        check(f"...the profile carries {key!r}", key in prof,
              str(sorted(prof)))
    check("the designation block states the rule and the floor it applied",
          "Net_PnL" in prof["designation"]["rule"]
          and isinstance(prof["designation"]["sample_floor"], int),
          str(prof["designation"]))
    check("optimal_quadrant and optimal_regime are the SAME statement",
          (prof["optimal_regime"] == "None")
          or (prof["optimal_quadrant"]
              == {"High Volatility / Trending": "Q1",
                  "High Volatility / Ranging": "Q2",
                  "Low Volatility / Trending": "Q3",
                  "Low Volatility / Ranging": "Q4"}[prof["optimal_regime"]]),
          f"{prof['optimal_regime']} / {prof['optimal_quadrant']}")
    check("regime_scores is keyed by regime and covers every traded quadrant",
          set(prof["regime_scores"]) == set(prof["regime_breakdown"]),
          str(sorted(prof["regime_scores"])))


def _isolated(fn):
    with tempfile.TemporaryDirectory() as tmp:
        prior = os.environ.get("BT_ARTIFACTS")
        os.environ["BT_ARTIFACTS"] = tmp
        try:
            fn(None)
        finally:
            if prior is None:
                os.environ.pop("BT_ARTIFACTS", None)
            else:
                os.environ["BT_ARTIFACTS"] = prior


def test_profiler_generates_a_four_quadrant_profile():
    _isolated(_check_generate_profile)


def test_screen_reads_the_profiler_output():
    _isolated(_check_screen_consumes_a_real_profile)


def test_a_configuration_with_no_trades_is_dropped():
    _isolated(_check_no_trades)


def test_either_version_can_carry_a_configuration():
    _isolated(_check_version_b_alone)


def test_true_home_regime_designation():
    _isolated(_check_true_home_regime)


def main() -> int:
    print("=" * 60)
    print("  the Stage 1 regime seam: profiler -> screen")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        for fn in (_check_generate_profile, _check_screen_consumes_a_real_profile,
                   _check_no_trades, _check_version_b_alone,
                   _check_true_home_regime):
            try:
                fn(None)
            except AssertionError:
                pass                      # already recorded by `check`
            except Exception as e:        # noqa: BLE001
                print(f"  FAIL  {fn.__name__} raised "
                      f"{type(e).__name__}: {e}")
                _failures.append(fn.__name__)

    print("\n" + "=" * 60)
    if _failures:
        print(f"  {len(_failures)} CHECK(S) FAILED")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
