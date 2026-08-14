#!/usr/bin/env python3
"""
test_tier3_workers.py - the Tier 3 execution and quantitative-testing tools.

Location:  ~/src/trading/tests/test_tier3_workers.py

Run:  python tests/test_tier3_workers.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

What it checks
--------------
1. Metrics are arithmetically right, verified against the trade log rather
   than against a remembered number.
2. Costs actually reached the P&L (gross - net == costs).
3. The Monte Carlo bootstrap matches a naive, obviously-correct reference,
   so the in-place chunked implementation cannot drift from it silently.
4. Known-answer Monte Carlo cases: all-winners has no drawdown, all-losers
   breaches with certainty.
5. Integer parameter perturbation lands where it should. 50 * 1.1 is
   55.00000000000001 in binary floating point, so a ceil() would turn a 10%
   step into 56 - the rounding is checked explicitly.
6. Walk-forward folds tile the requested span, and an undefined efficiency
   ratio stays out of the aggregate. A NaN reaching np.mean turns the whole
   study into NaN, and a negative in-sample return makes the ratio actively
   misleading rather than merely absent.
7. A blown account is flagged (`ruined`) rather than only showing up as a NaN
   annualized return, which reads like missing data.
8. Every load failure raises rather than returning a null strategy. A backtest
   over a strategy that failed to import produces no trades, which downstream
   is indistinguishable from a strategy that never triggered.
9. The generated boilerplate imports and runs. The first version of the
   template assigned to its own parameter name inside signal_fn, which made it
   a local and raised UnboundLocalError on the first call.
10. Peak RSS on a real multi-symbol run stays under the ceiling.
"""

from __future__ import annotations

import math
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tier3_workers import (  # noqa: E402
    StrategyLoadError, generate_strategy_boilerplate, load_strategy,
    run_monte_carlo_simulation, run_parameter_sensitivity,
    run_strategy_backtest, run_walk_forward_analysis, _fold_windows,
    trade_returns_from_result,
)

FAILURES: list[str] = []

SYMBOLS = ["ES", "NQ", "GC"]
START, END = "2016-01-01", "2022-12-31"
RSS_CEILING_GIB = 3.0


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def dummy_strategy(tmp: Path) -> Path:
    """The fixture is produced by the generator, so the template is tested too."""
    return generate_strategy_boilerplate(
        "SMA Crossover Dummy", "test fixture",
        params={"fast": 20, "slow": 50}, symbols=SYMBOLS,
        timeframe="1d", out_dir=tmp, overwrite=True)


# --------------------------------------------------------------------------
def test_boilerplate_runs(tmp: Path) -> None:
    print("\ngenerated boilerplate")
    path = dummy_strategy(tmp)
    check("file written", path.exists(), str(path.name))

    fn, info = load_strategy(path, {"fast": 20, "slow": 50})
    check("declares TIMEFRAME", info["timeframe"] == "1d", str(info["timeframe"]))

    # The template once shadowed its own parameter inside signal_fn.
    bars = pd.DataFrame({"close": 100 + np.random.default_rng(0).standard_normal(300).cumsum()})
    try:
        entries, exits = fn(bars)
        ran = True
    except Exception as e:
        ran = False
        check("signal_fn runs", False, f"{type(e).__name__}: {e}")
    if ran:
        check("signal_fn runs", True)
        check("returns bool Series", entries.dtype == bool and exits.dtype == bool,
              f"{entries.dtype}/{exits.dtype}")
        check("aligned to bars", len(entries) == len(bars) == len(exits))


def test_metrics_are_arithmetic(tmp: Path) -> None:
    print("\nbacktest metrics")
    path = dummy_strategy(tmp)
    r = run_strategy_backtest(path, SYMBOLS, START, END, params={"fast": 20, "slow": 50})
    check("ok", r["ok"] is True)
    check("has trades", r["trade_count"] > 0, f"{r['trade_count']} trades")

    t = r["trades"]
    pnl = t["pnl"]
    exp_wr = float((pnl > 0).sum() / len(t))
    gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())

    check("win_rate matches trade log", abs(r["win_rate"] - exp_wr) < 1e-12)
    check("profit_factor matches trade log", abs(r["profit_factor"] - gp / gl) < 1e-9)
    check("total_pnl matches trade log", abs(r["total_pnl"] - float(pnl.sum())) < 1e-6)

    # If this drifts to zero the cost model has been bypassed somewhere.
    check("costs reached the P&L",
          abs((r["gross_pnl"] - r["total_pnl"]) - r["total_costs"]) < 1e-6
          and r["total_costs"] > 0,
          f"costs {r['total_costs']:.2f}")

    check("trade_log is opt-in", "trade_log" not in r)
    r2 = run_strategy_backtest(path, ["ES"], "2020-01-01", "2021-12-31",
                               params={"fast": 20, "slow": 50},
                               include_trade_records=True)
    check("trade_log present when asked", "trade_log" in r2)


def test_monte_carlo_matches_reference() -> None:
    print("\nmonte carlo")

    def reference(arr, n_iter, seed, conf=0.95, max_loss=8.0):
        rng = np.random.default_rng(seed)
        s = rng.choice(arr, size=(n_iter, arr.size), replace=True)
        eq = np.cumprod(1.0 + s, axis=1)
        dd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0) * 100.0
        mdd = dd.min(axis=1)
        return (float(np.percentile(mdd, (1 - conf) * 100)),
                float(np.mean(mdd <= -max_loss)))

    rng = np.random.default_rng(0)
    all_match = True
    for n_trades, n_iter in ((50, 300), (500, 200), (2000, 100)):
        arr = rng.normal(0.002, 0.02, n_trades)
        got = run_monte_carlo_simulation(arr, n_iterations=n_iter, seed=11)
        exp = reference(arr, n_iter, 11)
        if not (abs(got["max_drawdown_pct_at_confidence"] - exp[0]) < 1e-9
                and abs(got["prob_max_loss_breach"] - exp[1]) < 1e-12):
            all_match = False
    check("chunked in-place == naive reference", all_match)

    pos = run_monte_carlo_simulation([0.01] * 50, n_iterations=200, seed=1)
    check("all winners: no drawdown",
          pos["max_drawdown_pct_at_confidence"] == 0.0 and pos["prob_profit"] == 1.0)

    neg = run_monte_carlo_simulation([-0.01] * 50, n_iterations=200,
                                     max_loss_pct=8.0, seed=1)
    check("all losers: certain breach", neg["prob_max_loss_breach"] == 1.0)

    check("empty input is structured, not raised",
          run_monte_carlo_simulation([])["ok"] is False)

    for bad, kwargs in (("n_iterations", {"n_iterations": 0}),
                        ("confidence_pct", {"confidence_pct": 1.5})):
        try:
            run_monte_carlo_simulation([0.01], **kwargs)
            check(f"rejects bad {bad}", False)
        except ValueError:
            check(f"rejects bad {bad}", True)


def test_integer_perturbation() -> None:
    print("\nparameter perturbation arithmetic")

    def shift(value, sign, pct=0.10):
        s = value * (1.0 + sign * pct)
        if isinstance(value, int):
            s = int(round(s))
            if s == value:
                s = value + (1 if sign > 0 else -1)
        return s

    # 50 * 1.1 == 55.00000000000001, so ceil() would give 56.
    cases = {3: (4, 2), 5: (6, 4), 10: (11, 9), 20: (22, 18),
             50: (55, 45), 100: (110, 90)}
    ok = all((shift(v, 1), shift(v, -1)) == exp for v, exp in cases.items())
    check("integers land on the intended step", ok,
          "50 -> " + str((shift(50, 1), shift(50, -1))))
    check("always moves off the base",
          all(shift(v, 1) != v and shift(v, -1) != v for v in cases))


def test_sensitivity(tmp: Path) -> None:
    print("\nparameter sensitivity")
    path = generate_strategy_boilerplate(
        "Mixed Param Dummy", "fixture with non-numeric params",
        params={"fast": 20, "slow": 50, "use_filter": False, "label": "x"},
        symbols=SYMBOLS, timeframe="1d", out_dir=tmp, overwrite=True)

    s = run_parameter_sensitivity(
        path, base_params={"fast": 20, "slow": 50, "use_filter": False, "label": "x"},
        perturbation_pct=0.10, symbols=SYMBOLS,
        start_date="2018-01-01", end_date="2022-12-31")

    check("two numeric params, two directions", s["n_variations"] == 4,
          f"{s['n_variations']}")
    check("non-numerics skipped, not dropped", s["n_skipped"] == 2,
          str([k["param"] for k in s["skipped"]]))
    check("every variation ran", all(v["error"] is None for v in s["variations"]))

    vals = {(v["param"], v["direction"]): v["value"] for v in s["variations"]}
    check("perturbed values correct",
          vals[("fast", "up")] == 22 and vals[("fast", "down")] == 18
          and vals[("slow", "up")] == 55 and vals[("slow", "down")] == 45,
          str(vals))

    try:
        run_parameter_sensitivity(path, {"fast": 20}, 0.0, symbols=SYMBOLS)
        check("rejects perturbation_pct=0", False)
    except ValueError:
        check("rejects perturbation_pct=0", True)


def test_walk_forward(tmp: Path) -> None:
    print("\nwalk-forward")
    windows = _fold_windows(2010, 2015, train_years=2, test_years=1)
    check("folds tile the span", len(windows) == 4, f"{len(windows)} folds")
    check("test follows train",
          all(w["test_start"] > w["train_end"] for w in windows))
    check("test windows do not overlap",
          all(windows[i]["test_end"] < windows[i + 1]["test_start"]
              for i in range(len(windows) - 1)))

    path = dummy_strategy(tmp)
    w = run_walk_forward_analysis(path, SYMBOLS, train_years=2, test_years=1,
                                  start_year=2014, end_year=2022,
                                  params={"fast": 20, "slow": 50})
    check("unoptimized run is labelled",
          w["optimized"] is False and w["warning"] is not None)

    eff = w["efficiency_ratio"]
    check("aggregate is not NaN", eff is None or not math.isnan(eff), str(eff))
    check("undefined folds excluded from aggregate",
          w["n_scored"] + w["n_undefined"] + w["n_failed"] == w["n_folds"],
          f"scored {w['n_scored']} undefined {w['n_undefined']} of {w['n_folds']}")
    check("every scored fold is finite",
          all(not math.isnan(f["efficiency"]) for f in w["folds"]
              if f.get("efficiency") is not None))

    empty = run_walk_forward_analysis(path, ["ES"], train_years=5, test_years=5,
                                      start_year=2020, end_year=2022)
    check("no room for folds is structured, not raised",
          empty["ok"] is False and "no folds fit" in empty["error"])


def test_ruin_is_flagged(tmp: Path) -> None:
    print("\nblown account")
    path = dummy_strategy(tmp)
    # 2022 loses more than the account on this placeholder, so CAGR is
    # undefined; the point is that it says so rather than only returning NaN.
    r = run_strategy_backtest(path, SYMBOLS, "2022-01-01", "2022-12-31",
                              params={"fast": 20, "slow": 50})
    if r["final_equity"] <= 0:
        check("ruin flagged", r["ruined"] is True,
              f"final equity {r['final_equity']:.2f}")
        check("annualized return withheld when undefined",
              math.isnan(r["annualized_return_pct"]))
    else:
        check("ruin flag consistent with equity", r["ruined"] is False,
              f"final equity {r['final_equity']:.2f} (no ruin in this window)")


def test_load_errors(tmp: Path) -> None:
    print("\nstrategy loading")
    (tmp / "syntax_err.py").write_text("this is not python(((\n")
    (tmp / "no_signal.py").write_text("X = 1\n")
    (tmp / "raises.py").write_text('raise RuntimeError("boom")\n')
    (tmp / "plain_fn.py").write_text(
        "import pandas as pd\n"
        "def signal_fn(bars):\n"
        "    return bars['close'] > 0, bars['close'] < 0\n")
    good = dummy_strategy(tmp)

    cases = [
        ("missing file", tmp / "nope.py", None),
        ("syntax error", tmp / "syntax_err.py", None),
        ("raises at import", tmp / "raises.py", None),
        ("no signal_fn", tmp / "no_signal.py", None),
        ("params without a factory", tmp / "plain_fn.py", {"fast": 5}),
        ("unknown param", good, {"nope": 1}),
        ("factory rejects the value", good, {"fast": -5}),
    ]
    for label, p, params in cases:
        try:
            load_strategy(p, params)
            check(f"raises on {label}", False, "returned a strategy")
        except StrategyLoadError:
            check(f"raises on {label}", True)
        except Exception as e:
            check(f"raises on {label}", False, f"wrong type: {type(e).__name__}")

    try:
        _, info = load_strategy(tmp / "plain_fn.py")
        check("plain signal_fn with no params loads", info["bound_params"] == {})
    except Exception as e:
        check("plain signal_fn with no params loads", False, str(e))

    try:
        run_strategy_backtest(good, ["ES"], "2020-01-01", "2020-12-31",
                              params={"fast": 20, "config": {"not_a_field": 1}})
        check("rejects unknown BacktestConfig field", False)
    except StrategyLoadError:
        check("rejects unknown BacktestConfig field", True)


def test_memory(tmp: Path) -> None:
    print("\nmemory")
    path = dummy_strategy(tmp)
    r = run_strategy_backtest(path, SYMBOLS, START, END,
                              params={"fast": 20, "slow": 50})
    tr = trade_returns_from_result(r)
    run_monte_carlo_simulation(tr, n_iterations=1000)
    peak = rss_gib()
    check(f"peak RSS under {RSS_CEILING_GIB} GiB", peak < RSS_CEILING_GIB,
          f"{peak:.2f} GiB")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_boilerplate_runs(tmp)
        test_load_errors(tmp)
        test_metrics_are_arithmetic(tmp)
        test_monte_carlo_matches_reference()
        test_integer_perturbation()
        test_sensitivity(tmp)
        test_walk_forward(tmp)
        test_ruin_is_flagged(tmp)
        test_memory(tmp)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
