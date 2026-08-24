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
11. Model-generated code is rejected before it is imported when it imports
    outside the allowlist, reaches for eval/dunders, contains a negative shift
    or reversed slice, or returns prices instead of booleans. Importing a
    module executes it, so everything checkable statically is checked first.
"""

from __future__ import annotations

import math
import os
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tier3_workers import (  # noqa: E402
    GeneratedCodeError, StrategyLoadError, apply_ml_signal_filter,
    generate_strategy_boilerplate,
    load_strategy, run_monte_carlo_simulation, run_parameter_sensitivity,
    run_strategy_backtest, run_walk_forward_analysis, _fold_windows,
    trade_returns_from_result, write_and_validate_strategy,
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


def test_monte_carlo_sanitation() -> None:
    """
    The bootstrap must never report a NaN tail beside a 0.0 breach
    probability. That pairing is what an unsanitised array produced, and it is
    the one corruption here that reads as SAFETY: the breach test is
    `mean(max_dds <= -limit)`, `NaN <= -8.0` is False, so an array holding an
    infinite loss reported a ZERO percent chance of breaching the loss limit.
    """
    print("\nmonte carlo sanitation")

    # An inf return used to make the whole path inf, then inf/inf -> NaN.
    r = run_monte_carlo_simulation([float("inf"), 0.01, -0.02],
                                   n_iterations=300, seed=1)
    check("inf is dropped, not propagated to a NaN tail",
          r["ok"] and math.isfinite(r["max_drawdown_pct_at_confidence"])
          and r["n_dropped_nonfinite"] == 1,
          f"tail={r.get('max_drawdown_pct_at_confidence')} "
          f"dropped={r.get('n_dropped_nonfinite')}")

    # A total loss: equity hits 0, the running peak is 0 on the first trade of
    # some paths, and the unguarded divide was 0/0.
    r = run_monte_carlo_simulation([-1.0, 0.02, 0.03, -0.01],
                                   n_iterations=500, seed=1)
    check("a -1.00 return gives a finite -100% floor, not NaN",
          r["ok"] and math.isfinite(r["max_drawdown_pct_at_confidence"])
          and math.isfinite(r["worst_max_drawdown_pct"])
          and abs(r["worst_max_drawdown_pct"] + 100.0) < 1e-9,
          f"worst={r.get('worst_max_drawdown_pct')}")
    check("ruin is reported as a BREACH, not as 0.0 probability",
          r["prob_max_loss_breach"] > 0.0,
          f"breach={r.get('prob_max_loss_breach')}")

    # Below -1.00 the equity factor goes negative and cumprod flips the sign of
    # the rest of the path: it reported -142% on an account that cannot lose
    # more than it holds.
    r = run_monte_carlo_simulation([-1.4, 0.05, 0.02], n_iterations=400, seed=1)
    check("a return below -1.00 is clipped, never a >100% drawdown",
          r["ok"] and r["n_clipped_to_total_loss"] == 1
          and r["worst_max_drawdown_pct"] >= -100.0 - 1e-9,
          f"worst={r.get('worst_max_drawdown_pct')} "
          f"clipped={r.get('n_clipped_to_total_loss')}")

    r = run_monte_carlo_simulation([float("nan")] * 4)
    check("an all-non-finite array is refused, not resampled",
          r["ok"] is False and r["n_dropped_nonfinite"] == 4)

    # No result may ever carry a non-finite headline number.
    bad = []
    for arr in ([-1.0, 0.01], [-2.0, 0.5], [float("inf"), 0.01],
                [0.01, -0.02, 0.03]):
        r = run_monte_carlo_simulation(arr, n_iterations=200, seed=3)
        if not r["ok"]:
            continue
        for k in ("max_drawdown_pct_at_confidence", "prob_max_loss_breach",
                  "median_max_drawdown_pct", "worst_max_drawdown_pct",
                  "median_final_return_pct", "prob_profit"):
            if not math.isfinite(r[k]):
                bad.append((arr, k))
    check("no headline metric is ever non-finite", not bad, str(bad))

    # The clean path must not have moved: sanitation only removes or clips what
    # was already corrupting the result.
    rng = np.random.default_rng(7)
    arr = rng.normal(0.001, 0.02, 400)
    r = run_monte_carlo_simulation(arr, n_iterations=300, seed=3)
    ref_rng = np.random.default_rng(3)
    s = ref_rng.choice(arr, size=(300, arr.size), replace=True)
    eq = np.cumprod(1.0 + s, axis=1)
    mdd = ((eq / np.maximum.accumulate(eq, axis=1)) - 1.0).min(axis=1) * 100.0
    check("a clean array is bit-identical to the unguarded arithmetic",
          abs(r["max_drawdown_pct_at_confidence"]
              - float(np.percentile(mdd, 5.0))) < 1e-12
          and r["n_dropped_nonfinite"] == 0
          and r["n_clipped_to_total_loss"] == 0)


def test_monte_carlo_scale_regression() -> None:
    """
    The bootstrap must be fed FRACTIONS once, not twice.

    `trade_returns_from_result` already divides per-trade dollar P&L by the
    starting capital - its docstring says "per-trade returns as fractions of
    starting equity". Stage 3 then passed `returns_are_dollars=True`, which
    divided by `initial_capital` a SECOND time, so every return reaching the
    drawdown distribution was `initial_capital` times too small.

    This is not the NaN corruption the sanitation above guards. Nothing is
    non-finite, nothing is clipped, and `ok` is True: the array is merely
    SHRUNK, and a shrunk array bootstraps to a drawdown of roughly zero. The
    signature is a `max_drawdown_pct_at_confidence` of -0.00% beside a
    `prob_max_loss_breach` of 0.0 - the strongest possible safety reading -
    produced for every strategy whatever its equity path, including runs whose
    accounts ended below zero.

    Both readings are pinned here. Asserting only the correct one would let a
    future caller reintroduce the double division and still pass, because a
    near-zero drawdown is a perfectly well-formed number.
    """
    print("\nmonte carlo · return scale")
    capital = 100_000.0
    rng = np.random.default_rng(11)

    # A ruinous series: 13,089 trades averaging a loss, netting roughly
    # -1.79x the starting capital, which is the shape of a real losing run.
    pnl = rng.normal(-13.66, 500.0, 13_089)
    pnl -= pnl.mean() - (-178_868.0 / 13_089)
    result = {"trades": pd.DataFrame({"pnl": pnl}),
              "meta": {"initial_capital": capital}}

    fractions = trade_returns_from_result(result, capital)
    check("trade_returns_from_result returns FRACTIONS, not dollars",
          bool(np.isclose(fractions, pnl / capital).all()),
          f"mean {float(fractions.mean()):.3e}")

    correct = run_monte_carlo_simulation(
        fractions, n_iterations=300, initial_capital=capital,
        returns_are_dollars=False, seed=42)
    check("the bootstrap ran cleanly", correct["ok"] is True)
    check("nothing was dropped or clipped - this is a SCALE bug, not a NaN one",
          correct["n_dropped_nonfinite"] == 0
          and correct["n_clipped_to_total_loss"] == 0)
    dd = correct["max_drawdown_pct_at_confidence"]
    check("a ruinous series bootstraps to a drawdown deeper than 50%",
          dd < -50.0, f"dd@95% = {dd:.4f}%")
    check("a ruinous series breaches the loss limit with probability 1.0",
          correct["prob_max_loss_breach"] == 1.0,
          f"breach_prob = {correct['prob_max_loss_breach']}")

    # The bug's own signature, pinned so it cannot come back unnoticed.
    doubled = run_monte_carlo_simulation(
        fractions, n_iterations=300, initial_capital=capital,
        returns_are_dollars=True, seed=42)
    check("dividing a fraction by the capital again reports near-zero risk",
          abs(doubled["max_drawdown_pct_at_confidence"]) < 0.1
          and doubled["prob_max_loss_breach"] == 0.0,
          f"dd@95% = {doubled['max_drawdown_pct_at_confidence']:.6f}%, "
          f"breach = {doubled['prob_max_loss_breach']}")
    check("the two readings differ by orders of magnitude",
          abs(dd) > 100 * abs(doubled["max_drawdown_pct_at_confidence"]),
          f"{dd:.4f}% vs {doubled['max_drawdown_pct_at_confidence']:.6f}%")

    # And the caller that matters. A unit test on the function cannot stop
    # Stage 3 from passing the wrong flag, and Stage 3's audit is what a
    # promotion rests on - so the call site is asserted directly.
    # Comments are stripped first: this file's own prose explains the bug and
    # names the wrong flag, and so does the call site's. Matching raw text
    # would fail on the explanation rather than on the code.
    src = (Path(__file__).resolve().parent.parent
           / "backtest" / "audit_gates.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    check("audit_gates does NOT tell the bootstrap its fractions are dollars",
          "returns_are_dollars=True" not in code,
          "backtest/audit_gates.py passes returns_are_dollars=True")
    check("audit_gates passes returns_are_dollars=False explicitly",
          "returns_are_dollars=False" in code)


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


GENERATED_OK = """
import pandas as pd

def signal_fn(bars, lookback=20):
    close = bars["close"]
    prior_high = bars["high"].rolling(lookback).max().shift(1)
    prior_low = bars["low"].rolling(lookback).min().shift(1)
    entries = (close > prior_high).fillna(False)
    exits = (close < prior_low).fillna(False)
    return entries, exits
"""


def test_generated_code_validation(tmp: Path) -> None:
    """
    Model-authored code is executed by importing it, so everything checkable
    without running it is checked first. These are the cases that must never
    reach an import.
    """
    print("\ngenerated code validation")

    path = write_and_validate_strategy("gen_ok", GENERATED_OK, out_dir=tmp)
    check("valid bars-signature module is accepted", path.exists())

    fenced = write_and_validate_strategy(
        "gen_fenced", "```python\n" + GENERATED_OK + "\n```", out_dir=tmp)
    check("markdown fences are stripped",
          "```" not in fenced.read_text())

    # The generated signature now matches what the engine calls, so the adapter
    # no longer reshapes arguments. It still binds params and forces the return
    # to boolean, which is the part the model cannot be trusted with.
    check("engine adapter is appended",
          "make_signal_fn" in path.read_text())
    fn, _ = load_strategy(path)
    bars = pd.DataFrame({
        "open": np.arange(100.0, 340.0), "high": np.arange(101.0, 341.0),
        "low": np.arange(99.0, 339.0), "close": np.arange(100.0, 340.0),
        "volume": np.ones(240, dtype="uint64"),
    })
    entries, exits = fn(bars)
    check("adapted strategy returns aligned booleans",
          len(entries) == len(bars) and entries.dtype == bool
          and exits.dtype == bool)

    rejects = [
        ("syntax error", "def signal_fn(((", SyntaxError),
        ("empty code", "   ", GeneratedCodeError),
        ("no signal_fn", "import numpy as np\ndef other(x):\n    return x\n",
         GeneratedCodeError),
        ("os import",
         "import os\ndef signal_fn(bars):\n"
         "    c = bars['close']\n    return c>0, c<0\n",
         GeneratedCodeError),
        ("subprocess import",
         "import subprocess\ndef signal_fn(bars):\n"
         "    c = bars['close']\n    return c>0, c<0\n",
         GeneratedCodeError),
        # The open-source twin. Same-looking API, different simulation
        # semantics, and nothing downstream would report the substitution.
        ("open-source vectorbt import",
         "import vectorbt as vbt\ndef signal_fn(bars):\n"
         "    c = bars['close']\n    return c>0, c<0\n",
         GeneratedCodeError),
        ("eval",
         "def signal_fn(bars):\n    eval('1')\n"
         "    c = bars['close']\n    return c>0, c<0\n",
         GeneratedCodeError),
        ("dunder escape",
         "def signal_fn(bars):\n    c = bars['close']\n"
         "    x = c.__class__\n    return c>0, c<0\n",
         GeneratedCodeError),
        ("lookahead shift(-1)",
         "import pandas as pd\ndef signal_fn(bars):\n"
         "    c = bars['close']\n    f = c.shift(-1)\n    return c>0, c<0\n",
         GeneratedCodeError),
        # The same lookahead written through a variable. A literal-only check
        # let this through, which is why the audit rejects any unary minus.
        ("lookahead shift(-k) via a variable",
         "import pandas as pd\ndef signal_fn(bars, k=1):\n"
         "    c = bars['close']\n    f = c.shift(-k)\n    return c>0, c<0\n",
         GeneratedCodeError),
        ("reversed slice",
         "def signal_fn(bars):\n    c = bars['close']\n"
         "    r = c[::-1]\n    return c>0, c<0\n",
         GeneratedCodeError),
        ("returns one array",
         "def signal_fn(bars):\n    return bars['close']>0\n",
         GeneratedCodeError),
        # Blanket astype(bool) would make this True on every nonzero bar: a
        # position opened every bar, and an equity curve that looks like
        # leverage rather than a bug.
        ("returns prices, not booleans",
         "def signal_fn(bars):\n    c = bars['close']\n    return c, c\n",
         GeneratedCodeError),
    ]
    for label, code, expected in rejects:
        try:
            write_and_validate_strategy(label.replace(" ", "_"), code, out_dir=tmp)
            check(f"rejects {label}", False, "accepted it")
        except Exception as e:
            check(f"rejects {label}", isinstance(e, expected),
                  f"{type(e).__name__}")

    ok_int = write_and_validate_strategy(
        "gen_ints",
        "def signal_fn(bars):\n    c = bars['close']\n"
        "    z = (c > c.mean()).astype(int)\n    return z, 1 - z\n",
        out_dir=tmp)
    check("accepts 0/1 integer signals", ok_int.exists())


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


# --------------------------------------------------------------------------
def _ml_fixture(n: int = 1200, seed: int = 7):
    """
    Bars with enough alternating entries to make the refit cadence bite.

    Synthetic and small on purpose: the property under test is how OFTEN the
    classifier is refitted, which does not need a lake.
    """
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    bars = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"),
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1000.0,
    })
    entries = pd.Series(np.zeros(n, dtype=bool))
    exits = pd.Series(np.zeros(n, dtype=bool))
    entries.iloc[2::6] = True          # one trade every six bars
    exits.iloc[5::6] = True
    return bars, entries, exits


def _legacy_filter(bars, entries, exits, threshold=0.5):
    """
    The refit-on-every-completed-trade loop as it stood before the cadence,
    as an ORACLE. Kept here rather than referenced, because the point is to
    detect the day the real one silently stops agreeing with it.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from agents.tier3_workers import (MIN_TRAIN_TRADES, _feature_matrix,
                                      _label_baseline_trades)
    entries = pd.Series(entries).fillna(False).astype(bool)
    exits = pd.Series(exits).fillna(False).astype(bool)
    trades = _label_baseline_trades(bars, entries, exits, None, None,
                                    direction="long")
    kept = entries.to_numpy(dtype=bool).copy()
    signal_bars = np.flatnonzero(kept)
    matrix = _feature_matrix(None, bars)
    train_rows = matrix[trades["signal_idx"]]
    labels, exit_idx = trades["label"], trades["exit_idx"]
    model, fitted_n, fits = None, -1, 0
    for s in signal_bars:
        n = int(np.searchsorted(exit_idx, s, side="left"))
        if n < MIN_TRAIN_TRADES:
            continue
        y = labels[:n]
        if np.unique(y).size < 2:
            continue
        if n != fitted_n:
            model = HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=5, early_stopping=False, random_state=0)
            model.fit(train_rows[:n], y)
            fitted_n, fits = n, fits + 1
        if float(model.predict_proba(matrix[s:s + 1])[0, 1]) < threshold:
            kept[s] = False
    return pd.Series(kept, index=entries.index), fits


def test_refit_step() -> None:
    print("\nrefit cadence arithmetic")
    from agents.tier3_workers import _refit_step
    # 0.0 IS the original rule: refit whenever the pool grew at all.
    check("growth 0.0 steps by 1", _refit_step(500, 0.0) == 1)
    check("never below 1", _refit_step(3, 0.10) == 1, str(_refit_step(3, 0.10)))
    check("scales with the pool", _refit_step(500, 0.10) == 50)
    check("unfitted refits at once", _refit_step(0, 0.10) == 1)
    # A negative or absurd growth must not stall the loop forever.
    check("negative growth still progresses", _refit_step(500, -1.0) == 1)


def test_refit_growth_zero_is_the_old_rule() -> None:
    print("\nrefit_growth=0.0 reproduces the pre-cadence filter")
    bars, e, x = _ml_fixture()
    oracle, oracle_fits = _legacy_filter(bars, e, x)
    stats: dict = {}
    got, _ = apply_ml_signal_filter(bars, e, x, threshold=0.5,
                                    refit_growth=0.0, stats=stats)
    check("identical entry mask",
          bool((got.to_numpy() == oracle.to_numpy()).all()),
          f"{int((got.to_numpy() != oracle.to_numpy()).sum())} differ")
    check("same number of fits", stats["fits"] == oracle_fits,
          f"{stats['fits']} vs oracle {oracle_fits}")
    check("something was actually suppressed",
          int((~got.to_numpy() & e.to_numpy()).sum()) > 0)


def test_refit_cadence_cuts_fits() -> None:
    print("\nthe cadence is what makes the screen finish")
    bars, e, x = _ml_fixture()
    exact: dict = {}
    apply_ml_signal_filter(bars, e, x, threshold=0.5, refit_growth=0.0,
                           stats=exact)
    paced: dict = {}
    apply_ml_signal_filter(bars, e, x, threshold=0.5, refit_growth=0.10,
                           stats=paced)
    check("far fewer fits", paced["fits"] < exact["fits"] / 3,
          f"{paced['fits']} vs {exact['fits']}")
    check("candidates unchanged", paced["candidates"] == exact["candidates"])
    check("cadence is recorded", paced["refit_growth"] == 0.10)
    check("elapsed is recorded", paced["elapsed_s"] >= 0.0)


def test_cadence_never_sees_the_future() -> None:
    """
    The property the cadence must not break. A stale model is fitted on FEWER,
    OLDER closed trades - so a decision can only ever know less than the exact
    rule, never anything from at or after the signal bar. Asserted on the
    training window the loop would use, because a lookahead here is invisible
    in the equity curve: it just makes Version B look brilliant.
    """
    print("\ncadence staleness is backwards-only")
    from agents.tier3_workers import (MIN_TRAIN_TRADES, _label_baseline_trades,
                                      _refit_step)
    bars, e, x = _ml_fixture()
    trades = _label_baseline_trades(bars, e, x, None, None, direction="long")
    exit_idx = trades["exit_idx"]
    fitted_n, worst_ratio, violations = -1, 1.0, 0
    for s in np.flatnonzero(e.to_numpy()):
        n = int(np.searchsorted(exit_idx, s, side="left"))
        if n < MIN_TRAIN_TRADES:
            continue
        if fitted_n < 0 or (n - fitted_n) >= _refit_step(fitted_n, 0.10):
            fitted_n = n
        # Every trade in the fitted window closed strictly before this bar.
        if fitted_n and exit_idx[fitted_n - 1] >= s:
            violations += 1
        worst_ratio = min(worst_ratio, fitted_n / n)
    check("no trade in the window closed at or after the signal bar",
          violations == 0, f"{violations} violations")
    check("the model is never fitted on MORE than is available",
          worst_ratio <= 1.0)
    check("staleness stays bounded", worst_ratio > 0.80,
          f"worst {worst_ratio:.3f} of available trades")


def test_train_row_cap() -> None:
    print("\nthe training cap keeps the MOST RECENT trades")
    bars, e, x = _ml_fixture()
    capped: dict = {}
    got, _ = apply_ml_signal_filter(bars, e, x, threshold=0.5,
                                    refit_growth=0.0, max_train_rows=40,
                                    stats=capped)
    check("no fit exceeded the cap", capped["train_rows_max"] <= 40,
          str(capped["train_rows_max"]))
    check("the cap did not change which entries were judged",
          int(np.asarray(e).sum()) == capped["candidates"])
    uncapped: dict = {}
    apply_ml_signal_filter(bars, e, x, threshold=0.5, refit_growth=0.0,
                           stats=uncapped)
    check("the default cap does not bind on this sample",
          uncapped["train_rows_max"] < 50_000,
          f"{uncapped['train_rows_max']} rows")


def test_ml_thread_count() -> None:
    """
    Bounded parallelism, and the only form this estimator has:
    `HistGradientBoostingClassifier` takes no `n_jobs`, so the knob is the
    OpenMP thread count.
    """
    print("\nML thread bound")
    import inspect as _inspect
    from sklearn.ensemble import HistGradientBoostingClassifier
    from agents.tier3_workers import ML_THREADS_ENV, ml_thread_count
    check("the estimator really has no n_jobs",
          "n_jobs" not in _inspect.signature(
              HistGradientBoostingClassifier.__init__).parameters)
    before = os.environ.get(ML_THREADS_ENV)
    try:
        os.environ[ML_THREADS_ENV] = "4"
        check("BT_ML_THREADS is read", ml_thread_count() == 4)
        os.environ[ML_THREADS_ENV] = "nonsense"
        check("a bad value falls back to 1", ml_thread_count() == 1)
    finally:
        if before is None:
            os.environ.pop(ML_THREADS_ENV, None)
        else:
            os.environ[ML_THREADS_ENV] = before


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_boilerplate_runs(tmp)
        test_generated_code_validation(tmp)
        test_load_errors(tmp)
        test_metrics_are_arithmetic(tmp)
        test_monte_carlo_matches_reference()
        test_monte_carlo_sanitation()
        test_monte_carlo_scale_regression()
        test_integer_perturbation()
        test_sensitivity(tmp)
        test_walk_forward(tmp)
        test_ruin_is_flagged(tmp)
        test_refit_step()
        test_refit_growth_zero_is_the_old_rule()
        test_refit_cadence_cuts_fits()
        test_cadence_never_sees_the_future()
        test_train_row_cap()
        test_ml_thread_count()
        test_memory(tmp)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
