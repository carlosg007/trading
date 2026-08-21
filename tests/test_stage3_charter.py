"""
tests/test_stage3_charter.py - the Regime-Switching Incubator Stage 3 Charter.

Location: ~/src/trading/tests/test_stage3_charter.py

Covers the two modules the charter binds at Stage 3:

    backtest/audit_gates.py       ingestion and the parameter lock, holdout
                                  isolation, Gate R, the no-pruning
                                  guarantee, retention scoring, and the
                                  cryptographic seal
    backtest/discord_reporter.py  the Stage 3 gate-audit card

The six clauses, and how each one fails silently when nothing pins it:

  1. **Ingestion, and the LOCK.** The input is Stage 2's artifacts read as
     EXACT (symbol, timeframe) pairs out of `stage2_summary.json`, with the
     parameters bound verbatim from `best_params_<SYMBOL>_<TF>.json`. A glob
     of the directory certifies a superseded sweep's winner and nothing
     raises; a `--param` that slipped through unrecorded certifies a strategy
     that was never optimised, and the file would not say so.
  2. **Holdout isolation.** The verdict is the holdout, from `HOLDOUT_START`
     to the PRESENT. An in-sample window reaching into it is refused before a
     bar is read - the retention it would compute is a strategy scored
     against itself. A hardcoded holdout end is the quieter failure: it stops
     certifying against the newest bars the moment a year rolls over, and the
     verdict looks identical either way.
  3. **No pruning, no aggregate penalties.** Gate R is the certification: the
     edge has to hold out of sample INSIDE the one quadrant Stage 1
     designated, at PF >= 1.00 over >= 30 trades. Gates 1, 2 and 3 are
     computed and reported and cannot fail a certification - they score the
     blended sample across every market state, which a regime-gated strategy
     never trades.
  4. **No prop-firm rules.** No daily loss limit, no trailing drawdown. Those
     answer "would this funding program have tolerated the equity path",
     which is a fact about a rulebook; applied here they fail strategies for
     the shape of the road to the same money.
  5. **Retention scoring.** Profit factor, Sharpe, max drawdown and win rate,
     in-sample against holdout. Reported and never scored - but a holdout
     profit factor printed without the in-sample one beside it lets a
     collapsing edge read as a healthy one.
  6. **Promotion and the seal.** A pass is staged into
     `strategies/approved_incubator/<strategy>/` with `meta.json` carrying
     SHA-256 of the strategy code, the winning parameter file and the gate
     audit. Three hashes because they can be separated: the same code under a
     different winning cell is a different strategy with the same code
     checksum.

Runs without the lake and without a network: Gate R is exercised against
hand-built regime profiles whose answers are known, the seal against real
files in a temporary directory, and the Discord transport in --dry-run only.

    python tests/test_stage3_charter.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import backtest.discord_reporter as dr                            # noqa: E402
from backtest.audit_gates import (GATE_R, PROP_FIRM_FIELDS,       # noqa: E402
                                  UnknownRegimeError,
                                  WindowOverlapError,
                                  _assert_no_prop_firm_rules,
                                  _seal_hashes, certification_leaderboard,
                                  charter_audit, check_windows,
                                  load_stage2_summary, regime_gate,
                                  retention_scores, seal_and_promote,
                                  resolve_targets, stage2_targets,
                                  target_regime, write_stage3_summary)
from backtest.baseline import (MIN_REGIME_PROFIT_FACTOR,          # noqa: E402
                               MIN_REGIME_TRADES)
from backtest.engine import BacktestConfig                        # noqa: E402
from backtest.pipeline import (BEST_PARAMS_FILE, CHARTER_IS_END,  # noqa: E402
                               CHARTER_IS_START, GATE_AUDIT_FILE,
                               HOLDOUT_START, STAGE2_SUMMARY_FILE,
                               STAGE3_SUMMARY_FILE, write_stage)
from backtest.profiler import REGIMES                             # noqa: E402
from backtest.promote import sha256                               # noqa: E402
from backtest.report import FAIL, NOT_EVALUATED, PASS             # noqa: E402

_failures: list[str] = []

TRENDING, HV_RANGING, LV_TRENDING, LV_RANGING = REGIMES


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
    return bool(ok)


def raises(fn, exc) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:                                              # noqa: BLE001
        return True, str(e)
    except Exception as e:                                        # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return False, "nothing raised"


def _profile(**by_regime) -> dict:
    """A hand-built regime profile in the shape `RegimeProfiler` returns."""
    return {"regime_breakdown": dict(by_regime)}


def _q(pf, n, win_rate=50.0, net=1000.0) -> dict:
    return {"profit_factor": pf, "trade_count": n, "win_rate": win_rate,
            "net_pnl": net}


# --------------------------------------------------------------------------
# 1. Ingestion and the parameter lock
# --------------------------------------------------------------------------
def test_ingestion(tmp: Path) -> None:
    print("\n1. Stage 3 certifies Stage 2's EXACT pairs, with the parameters "
          "LOCKED")
    from backtest.audit_gates import load_params

    d = tmp / "ingest"
    d.mkdir(parents=True, exist_ok=True)
    write_stage(d / STAGE2_SUMMARY_FILE, 2, "demo", {"results": [
        {"symbol": "NQ", "timeframe": "5m", "status": "OPTIMIZED",
         "quadrant": "Q1", "optimal_regime": TRENDING, "in_stage1": True},
        {"symbol": "NQ", "timeframe": "15m", "status": "OPTIMIZED",
         "quadrant": "Q1", "optimal_regime": TRENDING, "in_stage1": True},
        {"symbol": "GC", "timeframe": "15m", "status": "ERROR",
         "quadrant": "Q4", "optimal_regime": LV_RANGING, "in_stage1": True,
         "error": "sweep raised"},
    ]})
    blob = load_stage2_summary("demo", d)
    rows = stage2_targets(blob, "15m")

    # The ragged-survivor trap, one stage further down. NQ was optimised at
    # both 5m and 15m and GC only at 15m; crossing the symbol axis with the
    # timeframe axis would certify NQ 5m inside a 15m run, against parameters
    # selected on different bars.
    check("only the pairs AT the requested timeframe are certified",
          {(r["symbol"], r["timeframe"]) for r in rows}
          == {("NQ", "15m"), ("GC", "15m")}, str(rows))
    check("...so NQ 5m is never crossed into a 15m certification",
          not any(r["timeframe"] == "5m" for r in rows))
    check("the regime scope Stage 1 designated travels with the pair",
          all(r["optimal_regime"] for r in rows), str(rows))

    # Stage 2 prunes nothing, so its matrix carries the sweep that raised. A
    # Stage 3 input silently shorter than the Stage 2 output is how "the sweep
    # never ran" becomes "this was certified and failed".
    gc = next(r for r in rows if r["symbol"] == "GC")
    check("a configuration Stage 2 could not sweep is a ROW, flagged, never "
          "dropped", gc["certifiable"] is False and gc["stage2_error"],
          str(gc))
    ok, msg = raises(lambda: load_stage2_summary("other", d), Exception)
    check("another strategy's summary is REFUSED - a card and a verdict "
          "posted under the wrong name is what nobody cross-checks", ok,
          msg[:80])
    check("a summary that was never written is not an error - --symbols "
          "still names contracts by hand",
          load_stage2_summary("demo", tmp / "nothing-here") is None)

    # The lock.
    bp = d / BEST_PARAMS_FILE.format(symbol="NQ_15m")
    write_stage(bp, 2, "demo", {
        "symbol": "NQ", "timeframe": "15m",
        "params": {"fast": 5, "slow": 50}, "variants_tested": 1296,
        "stage1_regime": {"optimal_regime": TRENDING, "quadrant": "Q1",
                          "version": "A"}})

    params, prov = load_params("demo", "NQ", d, {}, False, tf="15m")
    check("the winner is bound VERBATIM", params == {"fast": 5, "slow": 50},
          str(params))
    check("...and the file says so: params_locked",
          prov["params_locked"] is True and "nothing re-tuned"
          in prov["params_lock_note"], prov["params_lock_note"])
    check("the search size travels with it - a Sharpe read without N is not "
          "a measurement", prov["variants_tested"] == 1296)
    check("the regime scope reaches Stage 3 on the winner",
          (prov["stage1_regime"] or {}).get("optimal_regime") == TRENDING)
    check("the parameter file is named, so the seal can hash the right one",
          Path(prov["best_params_file"]).name == bp.name)

    # `--param` is allowed and is re-tuning. It must never be silent.
    params2, prov2 = load_params("demo", "NQ", d, {"slow": 80}, False,
                                 tf="15m")
    check("--param still overrides - an operator correcting the record "
          "outranks a file", params2["slow"] == 80)
    check("...but it BREAKS THE LOCK, in the file, by name",
          prov2["params_locked"] is False
          and "slow" in prov2["params_overridden"]
          and "LOCK BROKEN" in prov2["params_lock_note"],
          prov2["params_lock_note"])

    ok, msg = raises(lambda: load_params("demo", "ES", d, {}, False, tf="15m"),
                     FileNotFoundError)
    check("a missing best_params is an ERROR, never a silent fall back to "
          "the module defaults", ok, msg[:80])
    _p, prov3 = load_params("demo", "ES", d, {}, True, tf="15m")
    check("--defaults is the deliberate way to say so, and reports NO lock "
          "rather than claiming one",
          prov3["params_locked"] is False
          and "DEFAULT_PARAMS" in prov3["params_lock_note"],
          prov3["params_lock_note"])

    # `resolve_targets` is where the three sources meet.
    args = SimpleNamespace(symbols=None, stage2_summary=None)
    targets, skipped, source, blob2 = resolve_targets("demo", args, d, {},
                                                      "15m")
    check("with no --symbols the targets ARE the summary's exact pairs",
          [t["symbol"] for t in targets] == ["NQ"]
          and STAGE2_SUMMARY_FILE in source, f"{targets} / {source}")
    check("...and Stage 2's failed sweep is carried as a skip with a reason",
          [t["symbol"] for t in skipped] == ["GC"] and skipped[0]["error"],
          str(skipped))

    named = SimpleNamespace(symbols="NQ,ES", stage2_summary=None)
    targets, _sk, source, _b = resolve_targets("demo", named, d, {}, "15m")
    check("--symbols is an explicit override and names contracts directly",
          [t["symbol"] for t in targets] == ["NQ", "ES"], str(targets))
    es = next(t for t in targets if t["symbol"] == "ES")
    check("...and a pair no earlier stage screened is FLAGGED, not refused - "
          "certifying it by hand is allowed, reading the result as screened "
          "is not",
          es["in_stage1"] is False and es["stage2_status"] == "NOT IN STAGE 2",
          str(es))
    nq = next(t for t in targets if t["symbol"] == "NQ")
    check("...while a named pair Stage 2 DID optimise keeps its scope",
          nq["optimal_regime"] == TRENDING and nq["in_stage1"] is True,
          str(nq))

    bare = tmp / "bare"
    bare.mkdir(parents=True, exist_ok=True)
    write_stage(bare / BEST_PARAMS_FILE.format(symbol="CL_15m"), 2, "demo",
                {"params": {}})
    targets, _sk, source, blob3 = resolve_targets("demo", args, bare, {},
                                                  "15m")
    check("with no summary at all it falls back to the per-contract files - "
          "a single-contract Stage 2 run by hand leaves nothing else",
          [t["symbol"] for t in targets] == ["CL"] and blob3 is None
          and "no stage 2 summary" in source, f"{targets} / {source}")

    explicit = SimpleNamespace(symbols=None,
                               stage2_summary=str(d / "not-here.json"))
    ok, _msg = raises(
        lambda: resolve_targets("demo", explicit, d, {}, "15m"),
        FileNotFoundError)
    check("a summary named EXPLICITLY and missing is an error - an operator "
          "naming a file meant that file", ok)


# --------------------------------------------------------------------------
# 2. Holdout isolation
# --------------------------------------------------------------------------
def test_holdout_isolation() -> None:
    print("\n2. The verdict is the holdout, and it runs to the PRESENT")

    check_windows(CHARTER_IS_START, CHARTER_IS_END, HOLDOUT_START, None)
    check("the charter split is accepted with no holdout end at all", True)
    check("...and the charter's own dates are what the stage defaults to",
          (CHARTER_IS_END, HOLDOUT_START) == ("2022-12-31", "2023-01-01"),
          f"{CHARTER_IS_END} / {HOLDOUT_START}")

    # An open-ended HOLDOUT spends nothing - it is the window this stage
    # exists to spend. An open-ended IN-SAMPLE window eats it.
    ok, msg = raises(lambda: check_windows("2013-01-01", None,
                                           HOLDOUT_START, None),
                     WindowOverlapError)
    check("an open-ended IN-SAMPLE window is refused - it runs to the end of "
          "the lake and consumes the holdout", ok, msg[:70])
    ok, _ = raises(lambda: check_windows("2013-01-01", "2023-06-30",
                                         HOLDOUT_START, None),
                   WindowOverlapError)
    check("an in-sample window running INTO the holdout is refused", ok)
    ok, _ = raises(lambda: check_windows("2013-01-01", HOLDOUT_START,
                                         HOLDOUT_START, None),
                   WindowOverlapError)
    check("...including one that merely touches it", ok)
    ok, _ = raises(lambda: check_windows("2013-01-01", "2022-12-31",
                                         "2026-01-01", "2023-01-01"),
                   WindowOverlapError)
    check("an inverted holdout window is refused", ok)

    # The check has to run before any bars are read, or a 108-configuration
    # sweep fails an hour in on a window that was wrong from the first line.
    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "audit_gates.py"),
         "--strat", "sma_crossover", "--is-end", "2023-06-30"],
        capture_output=True, text=True, cwd=REPO, timeout=180)
    check("the CLI refuses a spent holdout before it reads a bar, and exits 2",
          out.returncode == 2 and "holdout" in out.stderr.lower(),
          f"rc={out.returncode} {out.stderr[-120:]}")


# --------------------------------------------------------------------------
# 3. Gate R, and no pruning on an aggregate
# --------------------------------------------------------------------------
def test_gate_r() -> None:
    print("\n3. Gate R is the certification, measured in ONE quadrant")

    prof = _profile(**{
        TRENDING: _q(1.42, 88),
        HV_RANGING: _q(0.61, 400, win_rate=39.0, net=-5000.0),
    })
    g = regime_gate(prof, TRENDING)
    check("an edge that holds in its designated quadrant CERTIFIES",
          g["status"] == PASS, g["status"])
    check("...and the quadrant id is on the verdict",
          g["quadrant"] == "Q1" and g["target_regime"] == TRENDING)
    check("...with the numbers it was measured on, not just the name",
          g["measured"]["profit_factor"] == 1.42
          and g["measured"]["trade_count"] == 88)

    g2 = regime_gate(prof, HV_RANGING)
    check("a losing quadrant FAILS on 400 trades - volume is not evidence",
          g2["status"] == FAIL)
    check("...and the note names WHICH bar was missed, since a thin sample "
          "and a losing one are fixed by different work",
          any("below" in (c.get("note") or "") for c in g2["checks"]),
          str([c.get("note") for c in g2["checks"]]))

    # The two bars bind on the SAME quadrant. A 9.00 over eleven trades and a
    # 0.90 over four hundred describe a strategy with no environment.
    thin = _profile(**{TRENDING: _q(9.0, MIN_REGIME_TRADES - 1)})
    check(f"a 9.00 profit factor over {MIN_REGIME_TRADES - 1} trades does NOT "
          f"certify", regime_gate(thin, TRENDING)["status"] == FAIL)
    edge = _profile(**{TRENDING: _q(MIN_REGIME_PROFIT_FACTOR,
                                    MIN_REGIME_TRADES)})
    check("...and both bars clear exactly at the boundary, not one past it",
          regime_gate(edge, TRENDING)["status"] == PASS)

    absent = regime_gate(prof, LV_TRENDING)
    check("a quadrant the strategy never traded out of sample FAILS",
          absent["status"] == FAIL and absent["measured"]["trade_count"] == 0)
    check("...and says it never traded there, which is not the same finding "
          "as losing there", "never traded" in absent["checks"][0]["note"],
          absent["checks"][0]["note"])

    none = regime_gate(prof, None)
    check("no designated quadrant is NOT EVALUATED, and NOT EVALUATED is "
          "never a pass",
          none["status"] == NOT_EVALUATED and none["status"] != PASS)

    # The bars are Stage 1's, imported rather than restated. Two copies would
    # let a screen at 1.00 feed a certification at 1.15, and the strategies
    # lost in the gap would look like ones that failed out of sample.
    check("Gate R's bars ARE Stage 1's screening bars",
          (MIN_REGIME_PROFIT_FACTOR, MIN_REGIME_TRADES) == (1.00, 30),
          f"{MIN_REGIME_PROFIT_FACTOR} / {MIN_REGIME_TRADES}")

    # A quadrant name outside REGIMES is a broken handoff, not a result.
    ok, msg = raises(lambda: target_regime({}, None, override="HV/Trend"),
                     UnknownRegimeError)
    check("a quadrant name no profiler produces RAISES rather than reading "
          "as a strategy that stopped trading", ok, msg[:80])
    reg, src = target_regime({}, None, override=TRENDING)
    check("--regime is honoured and recorded as an OVERRIDE, not as stage 1",
          reg == TRENDING and "override" in src, src)
    reg, src = target_regime({"stage1_regime": {"optimal_regime": "None"}},
                             {"optimal_regime": LV_RANGING})
    check("the JSON token 'None' is not a regime - the summary is the "
          "fallback", reg == LV_RANGING and STAGE2_SUMMARY_FILE in src,
          f"{reg} / {src}")


def test_no_aggregate_pruning() -> None:
    print("\n4. Gates 1-3 are evidence and cannot fail a certification")

    passing_r = regime_gate(_profile(**{TRENDING: _q(1.42, 88)}), TRENDING)
    failing_r = regime_gate(_profile(**{TRENDING: _q(0.61, 88)}), TRENDING)
    ret = retention_scores({"profit_factor": 2.4}, {"profit_factor": 1.1})

    def _aggregate(g1, g2, g3, overall):
        return {"version": "A", "status": overall, "passed": overall == PASS,
                "gates": {"gate1": {"name": "g1", "status": g1, "checks": []},
                          "gate2": {"name": "g2", "status": g2, "checks": []},
                          "gate3": {"name": "g3", "status": g3,
                                    "checks": []}}}

    # The clause, at its sharpest. Before 2026-08-21 this combination was
    # NOT CERTIFIED on the strength of the Gate 1 FAIL.
    a = charter_audit(_aggregate(FAIL, PASS, NOT_EVALUATED, FAIL),
                      passing_r, ret)
    check("a FAILING Gate 1 and an unrun Gate 2 still CERTIFY when Gate R "
          "passes", a["status"] == PASS and a["passed"] is True, a["status"])
    check("...and the pre-charter roll-up survives as aggregate_status, so "
          "the verdict reads as MOVED rather than quietly dropped",
          a["aggregate_status"] == FAIL and a["aggregate_passed"] is False
          and a["aggregate_is_advisory"] is True)
    check("...and all four gates are still on the file in full",
          set(a["gates"]) == {"gate1", "gate2", "gate3", GATE_R})
    check("the file says IN WORDS which gate the verdict came from",
          a["verdict_gate"] == GATE_R and "clause 3" in a["verdict_basis"])

    b = charter_audit(_aggregate(PASS, PASS, PASS, PASS), failing_r, ret)
    check("three PASSING aggregate gates do NOT certify a failed Gate R - "
          "the blend cannot vouch for the quadrant either",
          b["status"] == FAIL and b["passed"] is False, b["status"])

    c = charter_audit(_aggregate(PASS, PASS, PASS, PASS),
                      regime_gate(_profile(), None), ret)
    check("no designated quadrant is NOT CERTIFIED however the gates read",
          c["status"] == NOT_EVALUATED and c["passed"] is False)


def test_no_prop_firm_rules() -> None:
    print("\n5. No prop-firm rule reaches a research verdict")

    clean = BacktestConfig()
    block = _assert_no_prop_firm_rules(clean)
    check("a clean research config passes and RECORDS the absence rather "
          "than leaving it to a missing key",
          block["applied"] is False and set(block["fields_checked"])
          == set(PROP_FIRM_FIELDS), str(block["fields_checked"]))
    check("...and names CrossTrade as where those rules live",
          "CrossTrade" in block["rule"])

    for field in PROP_FIRM_FIELDS:
        ok, msg = raises(
            lambda f=field: _assert_no_prop_firm_rules(
                BacktestConfig(**{f: 5.0})), ValueError)
        check(f"a config carrying {field} is REFUSED - it would cut the "
              f"equity curve short and change nothing else on the console",
              ok, msg[:70])


# --------------------------------------------------------------------------
# 6. Retention
# --------------------------------------------------------------------------
def test_retention() -> None:
    print("\n6. Retention is calculated on all four metrics, and scored on "
          "none")

    r = retention_scores(
        {"profit_factor": 2.0, "sharpe": 1.5, "max_drawdown_pct": -20.0,
         "win_rate": 55.0},
        {"profit_factor": 1.0, "sharpe": 0.75, "max_drawdown_pct": -10.0,
         "win_rate": 44.0})
    m = r["metrics"]
    check("all four metrics the charter names are present",
          set(m) == {"profit_factor", "sharpe", "max_drawdown_pct",
                     "win_rate"}, str(sorted(m)))
    check("profit factor retention is holdout / in-sample",
          abs(m["profit_factor"]["retention"] - 0.5) < 1e-9)
    check("Sharpe likewise", abs(m["sharpe"]["retention"] - 0.5) < 1e-9)

    # The one that inverts. The engine signs drawdowns negative, so a raw
    # oos/is would score a strategy that drew down TWICE as deep at 2.00 and
    # sort it to the top of the table.
    check("a HALVED drawdown scores ABOVE 1.00, like every other row",
          abs(m["max_drawdown_pct"]["retention"] - 2.0) < 1e-9,
          str(m["max_drawdown_pct"]["retention"]))
    worse = retention_scores({"max_drawdown_pct": -10.0},
                             {"max_drawdown_pct": -40.0})
    check("...and a drawdown four times deeper scores 0.25, not 4.00",
          abs(worse["metrics"]["max_drawdown_pct"]["retention"] - 0.25) < 1e-9,
          str(worse["metrics"]["max_drawdown_pct"]["retention"]))
    check("every row carries its direction, so the number cannot be read the "
          "wrong way round",
          m["max_drawdown_pct"]["direction"] == "lower"
          and m["profit_factor"]["direction"] == "higher")

    check("a metric nobody could compute is None, not 0.0 - 'retained "
          "nothing' is a different finding",
          retention_scores({"sharpe": None}, {"sharpe": 1.0}
                           )["metrics"]["sharpe"]["retention"] is None)
    check("a zero denominator is None rather than an infinity",
          retention_scores({"profit_factor": 0.0}, {"profit_factor": 1.0}
                           )["metrics"]["profit_factor"]["retention"] is None)
    check("the block states that it is NOT a pruning criterion",
          r["scored"] is False and "not a pruning criterion" in r["rule"])


# --------------------------------------------------------------------------
# 7. The seal, and the incubator
# --------------------------------------------------------------------------
def _audit_for(tmp: Path, status: str, name: str) -> Path:
    return write_stage(tmp / GATE_AUDIT_FILE.format(symbol=name), 3, "demo", {
        "symbol": "NQ", "timeframe": "15m",
        "params": {"fast": 5, "slow": 50}, "variants_tested": 1296,
        "in_sample": {"start": CHARTER_IS_START, "end": CHARTER_IS_END},
        "holdout": {"start": HOLDOUT_START, "end": None},
        "versions": {"A": {"gate_audit": {
            "status": status, "passed": status == PASS,
            "gates": {"gate1": {"status": FAIL}, "gate2": {"status": PASS},
                      "gate3": {"status": PASS},
                      GATE_R: {"status": status}}}}}})


def test_seal_and_incubator(tmp: Path) -> None:
    print("\n7. A certified version is staged, sealed, and NOT committed")

    d = tmp / "seal"
    d.mkdir(parents=True, exist_ok=True)
    inc = d / "incubator"
    src = d / "demo_strat.py"
    src.write_text("import pandas as pd\n"
                   "DEFAULT_PARAMS = {'fast': 5, 'slow': 50}\n"
                   "def signal_fn(bars, **p):\n"
                   "    c = bars['close']\n"
                   "    return c > c, c < c\n")
    bp = write_stage(d / BEST_PARAMS_FILE.format(symbol="NQ_15m"), 2, "demo",
                     {"symbol": "NQ", "timeframe": "15m",
                      "params": {"fast": 5, "slow": 50}})
    prov = {"best_params_file": str(bp), "variants_tested": 1296,
            "params_locked": True, "target_regime": TRENDING,
            "target_quadrant": "Q1"}

    good = _audit_for(d, PASS, "NQ_pass")
    out = seal_and_promote("demo", "A", src, "NQ", "15m",
                           {"fast": 5, "slow": 50}, prov, good,
                           metrics_file=None, threshold=0.5, incubator=inc)
    check("a Gate R PASS is staged into the incubator",
          out["promoted"] is True and Path(out["dir"]) == inc / "demo",
          out.get("error") or str(out.get("dir")))

    dest = Path(out["dir"])
    meta = json.loads((dest / "meta.json").read_text())
    seal = meta["seal"]
    check("meta.json carries a seal block", bool(seal))

    # Version A is copied byte for byte, so the promoted file provably IS the
    # file that was certified.
    check("the CODE hash matches the promoted file AND the source",
          seal["strategy_code"]["sha256"] == sha256(dest / "strat.py")
          == sha256(src), seal["strategy_code"]["sha256"][:16])
    check("the WINNING PARAMETER file is hashed - the same code under a "
          "different cell is a different strategy with the same code checksum",
          seal["winning_parameters"]["sha256"] == sha256(bp))
    check("the GATE AUDIT is hashed - which holdout produced the verdict",
          seal["gate_audit"]["sha256"] == sha256(good))
    check("all three are full 64-character digests",
          all(len(seal[k]["sha256"]) == 64
              for k in ("strategy_code", "winning_parameters", "gate_audit")))
    check("a metrics handoff nobody supplied reads NOT AVAILABLE rather than "
          "being omitted - an absent key is a field nobody filled in",
          seal["metrics_handoff"]["sha256"] == "NOT AVAILABLE")
    check("the certified scope is sealed with the hashes",
          seal["target_regime"] == TRENDING
          and seal["target_quadrant"] == "Q1"
          and seal["params_locked"] is True)
    check("a FAILING advisory Gate 1 did not block the staging",
          meta["gate_audit"]["gate1"] == FAIL
          and meta["gate_audit_status"] == PASS, str(meta["gate_audit"]))
    check("nothing was git-committed - the commit is Stage 5's, in front of "
          "a human", "committed" not in meta)

    bad = seal_and_promote("demo", "A", src, "NQ", "15m", {}, prov,
                           _audit_for(d, FAIL, "NQ_fail"), metrics_file=None,
                           threshold=0.5, incubator=inc)
    check("a Gate R FAIL is refused staging", bad["promoted"] is False)
    check("...and the refusal is RETURNED, so a completed certification is "
          "not thrown away because staging failed",
          bool(bad["error"]) and bad["dir"] is None, str(bad))

    gone = seal_and_promote("demo", "A", d / "missing.py", "NQ", "15m", {},
                            prov, good, metrics_file=None, threshold=0.5,
                            incubator=inc)
    check("a source that vanished is an error row too, never an exception "
          "out of the loop", gone["promoted"] is False and bool(gone["error"]))


# --------------------------------------------------------------------------
# 8. The handoff
# --------------------------------------------------------------------------
class _Args:
    is_start = CHARTER_IS_START
    is_end = CHARTER_IS_END
    holdout_start = HOLDOUT_START
    holdout_end = None
    tf = "15m"
    regime_min_pf = MIN_REGIME_PROFIT_FACTOR
    regime_min_trades = MIN_REGIME_TRADES


def test_summary_handoff(tmp: Path) -> dict:
    print("\n8. stage3_audit_summary.json covers everything it was asked to "
          "certify")

    d = tmp / "handoff"
    d.mkdir(parents=True, exist_ok=True)
    audit = _audit_for(d, PASS, "NQ_15m")
    results = [{
        "symbol": "NQ", "timeframe": "15m", "path": audit,
        "status": {"A": PASS}, "passed": {"A": True},
        "target_regime": TRENDING, "target_quadrant": "Q1",
        "params": {"fast": 5, "slow": 50}, "params_locked": True,
        "in_stage1": True,
        "gates": {"A": {"gate1": FAIL, "gate2": PASS, "gate3": PASS,
                        GATE_R: PASS}},
        "regime_measured": {"A": {"profit_factor": 1.42, "trade_count": 88,
                                  "win_rate": 52.0}},
        "retention": {"A": retention_scores(
            {"profit_factor": 1.90, "sharpe": 1.2, "max_drawdown_pct": -9.0,
             "win_rate": 52.0},
            {"profit_factor": 1.05, "sharpe": 0.7, "max_drawdown_pct": -7.0,
             "win_rate": 48.0})["metrics"]},
        "incubator": {"A": {"promoted": True, "dir": d / "inc" / "demo",
                            "seal": _seal_hashes(audit, None, audit, None),
                            "error": ""}},
        "exclude_days": [0],
    }]
    errors = [{"symbol": "ES", "timeframe": "15m", "quadrant": "Q2",
               "optimal_regime": HV_RANGING, "in_stage1": True,
               "error": "ValueError: the lake returned no 15m bars"}]
    skipped = [{"symbol": "GC", "timeframe": "15m", "quadrant": "Q4",
                "optimal_regime": LV_RANGING, "in_stage1": True,
                "error": "stage 2 recorded no parameters"}]
    targets = [{"symbol": s, "timeframe": "15m"} for s in ("NQ", "ES", "GC")]

    path = write_stage3_summary("demo", d, results, errors, skipped, _Args(),
                                targets, f"{STAGE2_SUMMARY_FILE} (exact pairs)")
    check("written to the name the charter specifies",
          path.name == STAGE3_SUMMARY_FILE, path.name)
    blob = json.loads(path.read_text())

    check("through write_stage, so read_stage can refuse the wrong file",
          blob.get("stage") == 3 and blob.get("strategy") == "demo")
    rows = blob["results"]
    check("every configuration the stage was ASKED to certify is a row - "
          "errors and skips included, since a shorter table reads as a "
          "complete one", len(rows) == 3, str(len(rows)))
    by_sym = {r["symbol"]: r for r in rows}
    check("a run that raised is NOT AUDITED, never FAIL - 'the run broke' "
          "and 'the edge did not generalise' must not share a token",
          by_sym["ES"]["status"] == "NOT AUDITED"
          and by_sym["ES"]["certified"] is False, by_sym["ES"]["status"])
    check("...and so is a configuration Stage 2 never optimised",
          by_sym["GC"]["status"] == "NOT AUDITED" and by_sym["GC"]["error"])

    nq = by_sym["NQ"]
    check("the certified row carries the quadrant and its OWN numbers",
          nq["quadrant"] == "Q1" and nq["oos_profit_factor"] == 1.42
          and nq["oos_trade_count"] == 88)
    check("...and the IS and OOS blended factors are SEPARATE fields, so "
          "neither can be read as the other",
          nq["is_profit_factor"] == 1.90
          and nq["holdout_profit_factor"] == 1.05)
    check("...and all four retention ratios",
          set(nq["retention"]) == {"profit_factor", "sharpe",
                                   "max_drawdown_pct", "win_rate"},
          str(sorted(nq["retention"])))
    check("...and the seal, and where it was staged",
          nq["seal"] and nq["incubator_dir"], str(nq.get("incubator_dir")))
    check("the audit file is hashed on the summary too, so the index cannot "
          "drift from the verdict it points at",
          len(nq["audit_sha256"]) == 64)

    check("both windows are on the handoff",
          blob["in_sample"]["end"] == CHARTER_IS_END
          and blob["holdout"]["start"] == HOLDOUT_START)
    check("an open-ended holdout is null WITH the word beside it, never "
          "stamped with today's date as though the lake reached it",
          blob["holdout"]["end"] is None
          and "present" in blob["holdout"]["end_basis"])
    check("the verdict rule is on the handoff, not only in the log",
          blob["certification_rule"]["verdict_gate"] == GATE_R
          and blob["certification_rule"]["aggregate_gates_are_advisory"]
          is True)
    check("the absence of prop-firm rules is STATED",
          blob["prop_firm_rules"]["applied"] is False)
    cov = blob["coverage"]
    check("coverage counts targets against verdicts, and a shortfall is a "
          "run failure rather than a screening result",
          cov["targets"] == 3 and cov["certified"] == 1
          and cov["errors"] == 1 and cov["skipped"] == 1
          and cov["complete"] is False, str(cov))
    check("...and says so in words", "never a screening decision"
          in cov["rule"])
    return blob


# --------------------------------------------------------------------------
# 9. The Discord card
# --------------------------------------------------------------------------
def test_stage3_card(blob: dict) -> None:
    print("\n9. The Stage 3 gate-audit card")

    embed = dr.build_stage3_embed("demo", blob, source="/x/summary.json")
    desc = embed["description"]
    fields = {f["name"]: f["value"] for f in embed["fields"]}

    check("titled as a Stage 3 gate audit, naming the strategy",
          "Stage 3" in embed["title"] and "demo" in embed["title"],
          embed["title"])
    check("the strategy is a field too", fields.get("Strategy") == "`demo`")
    check("the timeframe is a field", fields.get("Timeframe") == "`15m`")
    check("the IN-SAMPLE window is on the card - it says what the parameters "
          "were fitted to", CHARTER_IS_START in desc and CHARTER_IS_END in desc)
    check("the OUT-OF-SAMPLE window is on the card, and an open end reads as "
          "'present' rather than as a blank",
          HOLDOUT_START in desc and "present" in desc)
    for col in ("SYMBOL", "TF", "VER", "QUAD", "GATE R", "REG PF", "REG N",
                "IS PF", "OOS PF", "SEAL"):
        check(f"the table carries the {col!r} column", col in desc)
    check("the target quadrant legend is built FROM the rows, so no second "
          "spelling of a regime name lives in the reporter",
          f"`Q1` {TRENDING}" in desc, desc[-500:])
    check("IS PF and OOS PF are BOTH on the row - a holdout factor alone "
          "lets a collapsing edge read as a healthy one",
          "1.90" in desc and "1.05" in desc)
    check("Gate R's own quadrant profit factor and trade count are there too",
          "1.42" in desc and "88" in desc)
    check("the card says in words that Gates 1-3 cannot fail a certification",
          "cannot fail a certification" in desc)
    check("Certified counts Gate R's passes",
          fields.get("Certified → Incubator") == "1")
    check("a NOT AUDITED configuration is still a row - the card is never "
          "shorter than the stage's input", "ES" in desc and "GC" in desc)

    digest = ((blob["results"][0].get("seal") or {})
              .get("strategy_code", {}).get("sha256", ""))
    check("the table shows a hash PREFIX, not the full digest",
          digest[:dr.SEAL_PREFIX_CHARS] in desc and digest not in desc)
    seals = "\n".join(f["value"] for f in embed["fields"]
                      if f["name"].startswith("Seals"))
    check("...and the full seals are their own field, labelled by what each "
          "one covers",
          digest in seals and "code" in seals and "params" in seals
          and "audit" in seals, seals[:120])
    check("only STAGED configurations get a seal block - a checksum of a "
          "file the reader cannot find is worse than none",
          "ES " not in seals and "GC " not in seals, seals)

    size = dr._embed_size(embed)
    check(f"the embed fits Discord's {dr.MAX_EMBED_TOTAL}-character limit",
          size <= dr.MAX_EMBED_TOTAL, str(size))
    check(f"the description fits the {dr.MAX_EMBED_DESCRIPTION}-character "
          f"limit", len(desc) <= dr.MAX_EMBED_DESCRIPTION, str(len(desc)))
    check("teal, and deliberately not the promotion green: a certification "
          "is not a decision to trade", embed["color"] == dr.TEAL)

    nothing = dict(blob)
    nothing["results"] = [dict(r, certified=False, gate_regime="FAIL")
                          for r in blob["results"]]
    check("amber when nothing certified, never red - a holdout that "
          "certified nothing is a result to read, not a crash",
          dr.build_stage3_embed("demo", nothing)["color"] == dr.AMBER)

    # The reporter computes nothing. A row whose numbers look like a pass but
    # whose recorded status is FAIL must post as a FAIL.
    lying = dict(blob)
    lying["results"] = [dict(blob["results"][0], certified=False,
                             gate_regime="FAIL", oos_profit_factor=99.0)]
    e = dr.build_stage3_embed("demo", lying)
    check("a huge quadrant factor does not become a PASS - the status is "
          "transcribed, never re-derived",
          "FAIL" in e["description"]
          and [f for f in e["fields"]
               if f["name"] == "Certified → Incubator"][0]["value"] == "0")


def test_mode_resolution() -> None:
    print("\n10. --stage 3 and --mode audit are one choice")

    check("--stage 3 resolves to the audit card",
          dr.resolve_mode(None, "3") == "audit")
    check("--mode audit alone does too", dr.resolve_mode("audit", None)
          == "audit")
    check("both together agree", dr.resolve_mode("audit", "3") == "audit")
    ok, msg = raises(lambda: dr.resolve_mode("baseline", "3"), ValueError)
    check("--mode baseline --stage 3 is REFUSED, never guessed at: picking "
          "either silently posts the wrong card", ok, msg[:70])
    check("the other three modes are untouched",
          dr.resolve_mode(None, "1") == "baseline"
          and dr.resolve_mode(None, "2") == "scan"
          and dr.resolve_mode(None, None) == "promotion")


def test_cli(tmp: Path, blob: dict) -> None:
    print("\n11. The CLI, end to end, sending nothing")

    d = tmp / "cli"
    d.mkdir(parents=True, exist_ok=True)
    write_stage(d / STAGE3_SUMMARY_FILE, 3, "demo",
                {k: v for k, v in blob.items()
                 if k not in ("stage", "strategy", "generated_utc")})

    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "discord_reporter.py"),
         "--stage", "3", "--strat", "demo",
         "--audit", str(d / STAGE3_SUMMARY_FILE), "--dry-run"],
        capture_output=True, text=True, cwd=REPO, timeout=180)
    check("--stage 3 --dry-run exits 0 and sends nothing",
          out.returncode == 0 and "DRY RUN" in out.stdout,
          f"rc={out.returncode} {out.stderr[-160:]}")
    if out.returncode == 0:
        payload = json.loads(out.stdout.split("DRY RUN")[0])
        check("...and prints one embed",
              len(payload.get("embeds") or []) == 1)

    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "discord_reporter.py"),
         "--stage", "3", "--strat", "other",
         "--audit", str(d / STAGE3_SUMMARY_FILE), "--dry-run"],
        capture_output=True, text=True, cwd=REPO, timeout=180)
    check("another strategy's summary is refused, and nothing is posted",
          out.returncode == 1 and "FAILED" in out.stderr,
          f"rc={out.returncode} {out.stderr[-160:]}")

    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "discord_reporter.py"),
         "--stage", "3", "--strat", "demo",
         "--audit", str(d / "does-not-exist.json"), "--dry-run"],
        capture_output=True, text=True, cwd=REPO, timeout=180)
    check("a missing handoff is a refusal, not an invented card",
          out.returncode == 1, f"rc={out.returncode}")

    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "audit_gates.py"), "--help"],
        capture_output=True, text=True, cwd=REPO, timeout=180)
    text = out.stdout
    check("audit_gates --help exits 0", out.returncode == 0,
          out.stderr[-160:])
    for flag in ("--regime-min-pf", "--regime-min-trades", "--no-promote",
                 "--stage2-summary", "--regime", "--incubator"):
        check(f"...and offers {flag}", flag in text)
    check("...and documents the holdout default as the PRESENT",
          "PRESENT" in text)
    check("...and documents that gates 1-3 cannot fail a certification",
          "cannot fail a certification" in text)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage3charter_") as td:
        tmp = Path(td)
        test_ingestion(tmp)
        test_holdout_isolation()
        test_gate_r()
        test_no_aggregate_pruning()
        test_no_prop_firm_rules()
        test_retention()
        test_seal_and_incubator(tmp)
        blob = test_summary_handoff(tmp)
        test_stage3_card(blob)
        test_mode_resolution()
        test_cli(tmp, blob)

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
