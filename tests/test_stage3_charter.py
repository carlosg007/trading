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
from backtest.audit_gates import (GATE_RUIN, RUIN_CHECK_FAILED,  # noqa: E402
                                  GATE_R, GATE_Q, PROP_FIRM_FIELDS,  # noqa: E402
                                  UnknownRegimeError,
                                  WindowOverlapError,
                                  _assert_no_prop_firm_rules,
                                  _seal_hashes, certification_leaderboard,
                                  charter_audit, check_windows,
                                  load_stage2_summary, regime_gate,
                                  retention_scores, seal_and_promote,
                                  resolve_targets, stage2_targets,
                                  target_regime, write_stage3_summary,
                                  audit_to_result, consolidated_audits,
                                  merge_stage3_rows)
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


def _raises(fn) -> bool:
    """Did it raise? A broken handoff has to stop the run, not be measured."""
    try:
        fn()
    except Exception:                                          # noqa: BLE001
        return True
    return False


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
         "--strat", "ema_crossover_20260821", "--is-end", "2023-06-30"],
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


def test_regime_starvation() -> None:
    print("\n3b. REGIME STARVATION — a Gate R that failed on sample, not edge")
    from backtest.audit_gates import (regime_for_quadrant, regime_starvation,
                                      target_regime)

    scores = {TRENDING: {"regime": TRENDING, "quadrant": "Q1",
                         "score": 150_132.0, "trade_count": 436},
              LV_RANGING: {"regime": LV_RANGING, "quadrant": "Q4",
                           "score": 1_335.0, "trade_count": 167}}

    # One holdout trade in the designated quadrant. The FACTOR row passes -
    # the profiler's 999 sentinel clears 1.00 - so a reader looking only at
    # the factor sees a healthy number under a FAIL, which is exactly the
    # case the diagnostic exists for.
    starved = _profile(**{LV_RANGING: _q(999, 1, win_rate=100.0, net=450.42)})
    g = regime_gate(starved, LV_RANGING, regime_scores=scores)
    check("a designated quadrant with 1 holdout trade FAILS", g["status"] == FAIL)
    diag = g.get("regime_starvation")
    check("...and carries a starvation diagnostic", bool(diag), str(diag))
    check("...naming the target quadrant, its holdout count, and where the "
          "candidate was actually dominant IN SAMPLE",
          diag["message"] == ("[REGIME STARVATION] Quadrant Q4 "
                              "(Low Volatility / Ranging) had only 1 holdout "
                              "trades. Candidate was dominant in Q1 "
                              "(High Volatility / Trending) in sample."),
          diag["message"])
    check("...and the dominance is read from the STAGE 2 handoff, never "
          "re-derived from the holdout — naming a new quadrant off the "
          "holdout is the best-of-four pick Gate R exists to avoid",
          diag["dominant_basis"].startswith("in-sample")
          and diag["dominant_quadrant"] == "Q1")

    # A quadrant that traded enough and lost is NOT starvation. Attaching the
    # diagnostic there would send the operator to re-designate a quadrant
    # whose problem is that the edge is dead.
    fed = _profile(**{TRENDING: _q(0.61, 400, win_rate=39.0, net=-5000.0)})
    g2 = regime_gate(fed, TRENDING, regime_scores=scores)
    check("a quadrant that traded 400 times and lost is a dead edge, not "
          "starvation — no diagnostic",
          g2["status"] == FAIL and g2.get("regime_starvation") is None)

    # A PASS never carries one either.
    g3 = regime_gate(_profile(**{TRENDING: _q(1.42, 88)}), TRENDING,
                     regime_scores=scores)
    check("a PASS carries no starvation diagnostic",
          g3["status"] == PASS and g3.get("regime_starvation") is None)

    # No scored table on the handoff: say so rather than guessing.
    bare = regime_gate(starved, LV_RANGING, regime_scores=None)
    check("with no scored table on the handoff the diagnostic still prints "
          "the starvation and declines to name a dominant quadrant",
          "REGIME STARVATION" in bare["regime_starvation"]["message"]
          and bare["regime_starvation"]["dominant_quadrant"] is None
          and "cannot be stated" in bare["regime_starvation"]["message"],
          bare["regime_starvation"]["message"])

    print("\n3c. Gate R's target is parsed DYNAMICALLY from best_params")
    check("Q1..Q4 codes resolve to their regime names",
          regime_for_quadrant("Q1") == TRENDING
          and regime_for_quadrant("q4") == LV_RANGING
          and regime_for_quadrant("Q9") is None)

    file_prov = {"best_params_file": "/x/best_params_NQ_30m.json"}
    check("the TOP-LEVEL optimal_regime stage 2 now writes is read first",
          target_regime({**file_prov, "best_params_regime": TRENDING},
                        None)[0] == TRENDING)
    check("...a handoff carrying only the QUADRANT CODE still resolves",
          target_regime({**file_prov, "best_params_quadrant": "Q1"},
                        None)[0] == TRENDING)
    check("...and a file written before the lift falls back to the nested "
          "stage1_regime, so an old handoff still certifies",
          target_regime({**file_prov,
                         "stage1_regime": {"optimal_regime": LV_RANGING}},
                        None)[0] == LV_RANGING)
    regime, source = target_regime(
        {**file_prov, "best_params_regime": TRENDING,
         "stage1_regime": {"optimal_regime": LV_RANGING}}, None)
    check("when the two DISAGREE the top level wins and the disagreement is "
          "RECORDED in the source, not resolved silently",
          regime == TRENDING and "disagrees" in source and LV_RANGING in source,
          source)
    check("a quadrant name no profiler produces still RAISES rather than "
          "being measured as zero trades",
          _raises(lambda: target_regime(
              {**file_prov, "best_params_regime": "Sideways Chop"}, None)))


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

    # A SURVIVING account. From 2026-08-24 the ruin guard is a hard bar, so
    # every case below has to say the account lived - otherwise these would be
    # testing the ruin guard rather than the clause they are named for.
    ALIVE = {"ruined": False, "final_equity": 138_400.0,
             "max_drawdown_pct": -17.5}

    # The clause, at its sharpest. Before 2026-08-21 this combination was
    # NOT CERTIFIED on the strength of the Gate 1 FAIL.
    # `require_all_quadrants=False` throughout this section: these cases
    # assert the CHARTER's verdict, which is Gate R alone, and that is now the
    # waived mode. Gate Q - the all-quadrant bar added 2026-09-07 - is
    # exercised in its own section below, on profiles built for it. Passing it
    # a fixture with no holdout breakdown would test nothing except that NOT
    # EVALUATED is not a pass, which `test_all_quadrant_gate` states directly.
    a = charter_audit(_aggregate(FAIL, PASS, NOT_EVALUATED, FAIL),
                      passing_r, ret, in_sample_metrics=ALIVE,
                      require_all_quadrants=False)
    check("a FAILING Gate 1 and an unrun Gate 2 still CERTIFY when Gate R "
          "passes", a["status"] == PASS and a["passed"] is True, a["status"])
    check("...and the pre-charter roll-up survives as aggregate_status, so "
          "the verdict reads as MOVED rather than quietly dropped",
          a["aggregate_status"] == FAIL and a["aggregate_passed"] is False
          and a["aggregate_is_advisory"] is True)
    check("...and all six gates are still on the file in full - Gate Q "
          "included, since 2026-09-08 it is always measured and recorded "
          "even when advisory",
          set(a["gates"]) == {"gate1", "gate2", "gate3", GATE_R, GATE_RUIN,
                              GATE_Q},
          str(sorted(a["gates"])))
    check("...and with no holdout profile Gate Q says NOT EVALUATED rather "
          "than inventing a pass",
          a["gates"][GATE_Q]["status"] == NOT_EVALUATED
          and a["gates"][GATE_Q]["advisory"] is True)
    check("the file says IN WORDS which gate the verdict came from",
          a["verdict_gate"] == GATE_R and "clause 3" in a["verdict_basis"])

    b = charter_audit(_aggregate(PASS, PASS, PASS, PASS), failing_r, ret,
                      in_sample_metrics=ALIVE, require_all_quadrants=False)
    check("three PASSING aggregate gates do NOT certify a failed Gate R - "
          "the blend cannot vouch for the quadrant either",
          b["status"] == FAIL and b["passed"] is False, b["status"])

    c = charter_audit(_aggregate(PASS, PASS, PASS, PASS),
                      regime_gate(_profile(), None), ret,
                      in_sample_metrics=ALIVE, require_all_quadrants=False)
    check("no designated quadrant is NOT CERTIFIED however the gates read",
          c["status"] == NOT_EVALUATED and c["passed"] is False)


def test_ruin_guard() -> None:
    """
    The ONE blended-sample bar that can refuse a certification.

    Gates 1-3 grade edge quality on a sample a regime-gated strategy never
    trades, which is why the charter made them advisory. Ruin is not a grade:
    it says the account the trades were placed in reached zero, and there is no
    quadrant restriction that makes a blown account survivable.

    Three configurations of `t3_braid_scalp_20260823` reached the incubator
    with `ruined: true` in sample - NQ 15m ended the charter window at -$78,868
    and NQ 30m at -$104,159 - because Gate R binds profit factor and trade
    count and nothing else. Both are pinned here as fixtures.
    """
    print("\n4b. Ruin is the one hard bar (2026-08-24)")

    passing_r = regime_gate(_profile(**{TRENDING: _q(1.42, 88)}), TRENDING)
    ret = retention_scores({"profit_factor": 2.4}, {"profit_factor": 1.1})
    agg = {"version": "A", "status": PASS, "passed": True,
           "gates": {"gate1": {"name": "g1", "status": PASS, "checks": []}}}

    def _audit(metrics):
        return charter_audit(agg, passing_r, ret, in_sample_metrics=metrics,
                             require_all_quadrants=False)

    alive = _audit({"ruined": False, "final_equity": 207_890.0,
                    "max_drawdown_pct": -49.56})
    check("a surviving account certifies exactly as before",
          alive["status"] == PASS and alive["passed"] is True, alive["status"])
    check("...and the guard is recorded on the file even when it passes",
          alive["gates"][GATE_RUIN]["status"] == PASS
          and alive["ruin_guard"]["status"] == PASS)

    # The two real promotions this gate exists to have refused.
    for label, metrics in (
            ("NQ 15m", {"ruined": True, "final_equity": -78_868.0,
                        "max_drawdown_pct": -194.27}),
            ("NQ 30m", {"ruined": True, "final_equity": -104_159.0,
                        "max_drawdown_pct": -205.94})):
        a = _audit(metrics)
        check(f"{label}: a PASSING Gate R does NOT certify a blown account",
              a["status"] == RUIN_CHECK_FAILED and a["passed"] is False,
              a["status"])
        check(f"{label}: Gate R's own PASS is left untouched on the file",
              a["gates"][GATE_R]["status"] == PASS)

    dd_only = _audit({"ruined": False, "final_equity": 5_000.0,
                      "max_drawdown_pct": -119.36})
    check("a drawdown past -100% refuses on its own - an account cannot lose "
          "more than it holds",
          dd_only["status"] == RUIN_CHECK_FAILED and dd_only["passed"] is False)

    equity_only = _audit({"final_equity": -1.0})
    check("a negative final equity refuses without the engine's flag - a "
          "metrics dict written before `ruined` existed still fails",
          equity_only["status"] == RUIN_CHECK_FAILED)

    silent = _audit({"profit_factor": 1.4})
    check("metrics that say NOTHING about survival are NOT EVALUATED, and "
          "NOT EVALUATED is not a pass - absent evidence of survival is not "
          "evidence of survival",
          silent["gates"][GATE_RUIN]["status"] == NOT_EVALUATED
          and silent["passed"] is False)

    check("the refusal token names the bar that was missed, so a blown "
          "account and a dead edge are never the same word",
          RUIN_CHECK_FAILED != FAIL and "RUIN" in RUIN_CHECK_FAILED)


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
    # `<strategy>_<SYMBOL>_<TF>_V<A|B>`, not `<strategy>` and no longer the
    # bare pair either. Stage 3 stages one certified PAIR AND VERSION, and a
    # campaign certifies several of each: sharing one directory per strategy
    # meant each staging overwrote the last, and sharing one per PAIR meant
    # Version B collided with the Version A that got there first. On
    # 2026-08-29 t3_braid NQ 1h certified on both - A at OOS PF 1.25, B at
    # 1.22 - and promote.py exited 1 on B, losing a package no gate refused.
    check("a Gate R PASS is staged under its PAIR *and VERSION* id",
          out["promoted"] is True
          and Path(out["dir"]) == inc / "demo_NQ_15m_VA",
          out.get("error") or str(out.get("dir")))
    check("...so the other version of the same pair does not collide with it",
          Path(out["dir"]) != inc / "demo_NQ_15m_VB")

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
    for col in ("SYM", "TF", "QD", "GATE R", "PF", "N", "STATUS"):
        check(f"the table carries the {col!r} column", col in desc)
    for gone in ("SYMBOL", "REG PF", "IS PF", "OOS PF", "SEAL"):
        check(f"...and NOT the wide {gone!r} column: a fixed-width table that "
              f"overruns a phone viewport wraps, and a wrapped row is two "
              f"rows with the second one unlabelled", gone not in desc)
    table = desc.split("```text\n")[1].split("\n```")[0]
    widest = max(len(line) for line in table.splitlines())
    check(f"every row fits {dr.STAGE3_TABLE_WIDTH} characters, which is what "
          f"keeps it from wrapping", widest <= dr.STAGE3_TABLE_WIDTH,
          f"{widest} chars")
    check("the target quadrant legend is built FROM the rows, so no second "
          "spelling of a regime name lives in the reporter",
          f"`Q1` {TRENDING}" in desc, desc[-500:])
    check("Gate R's own quadrant profit factor and trade count are on the "
          "row - they are what the verdict was measured on",
          "1.42" in table and "88" in table)
    check("...and the description says which factor that is, so a lone "
          "number under a regime-gated verdict is never guessed at",
          "INSIDE the target quadrant" in desc)
    check("the card says in words that Gates 1-3 cannot fail a certification",
          "cannot fail a certification" in desc)
    # Why a Gate R failed, on the row. The PASS/FAIL token is transcribed;
    # only the reason is worked out, and only from the thresholds the handoff
    # itself recorded.
    rule = {"min_profit_factor": 1.00, "min_trades": 30}
    starved = {"symbol": "NQ", "timeframe": "15m", "quadrant": "Q3",
               "target_regime": TRENDING, "gate_regime": FAIL,
               "certified": False, "status": FAIL,
               "oos_profit_factor": 999.0, "oos_trade_count": 1}
    dead = dict(starved, symbol="CL", oos_profit_factor=0.98,
                oos_trade_count=36)
    broke = {"symbol": "GC", "timeframe": "15m", "status": "NOT AUDITED",
             "certified": False, "gate_regime": "NOT AUDITED"}
    check("a Gate R that failed on the SAMPLE reads FAIL·N and STARVED - 'it "
          "never traded there again' and 'the edge died' are fixed by "
          "different work and must not share a token",
          dr._gate_r_cell(starved, rule) == "FAIL·N"
          and dr._status_cell(starved, rule) == "STARVED",
          dr._status_cell(starved, rule))
    check("...and one that failed on the FACTOR reads FAIL·PF and REJECTED",
          dr._gate_r_cell(dead, rule) == "FAIL·PF"
          and dr._status_cell(dead, rule) == "REJECTED",
          dr._status_cell(dead, rule))
    check("the 999 sentinel renders as `--`: a quadrant with one winning "
          "holdout trade has no measured profit factor, and 999.00 beside a "
          "FAIL reads as the strongest row on the card",
          dr._regime_pf_cell(starved) == "--", dr._regime_pf_cell(starved))
    check("a run that broke is NO AUDIT in both columns, never a FAIL - the "
          "run failing and the edge failing are different findings",
          dr._gate_r_cell(broke, rule) == "NO AUDIT"
          and dr._status_cell(broke, rule) == "NO AUDIT",
          dr._status_cell(broke, rule))
    # The table lists CERTIFIED configurations only. None of the three above
    # is one, so none of them is a row - the card is read to answer "what may
    # be promoted", and that is the only row anybody acts on.
    small, hidden_small, _l = dr.format_stage3_table([starved, dead, broke],
                                                     10, rule)
    check("a STARVED, a REJECTED and a NO AUDIT row are all OFF the table - "
          "only certified configurations are listed",
          "NQ" not in small and "CL" not in small and "GC" not in small,
          small)
    check("...and with nothing certified the block says so under the header, "
          "rather than rendering as an empty table that reads as a failure "
          "to draw one",
          dr.STAGE3_NO_ROWS_NOTE in small
          and small.splitlines()[0].startswith("SYM"), small)
    check("...and nothing is counted as merely HIDDEN by the row cap: the "
          "cap counts certified rows that did not fit, never rows the filter "
          "removed", hidden_small == 0, str(hidden_small))
    certified_only, _h, _l = dr.format_stage3_table(
        [starved, dict(starved, symbol="ES", certified=True,
                       gate_regime=PASS, status=PASS,
                       oos_profit_factor=1.42, oos_trade_count=88)], 10, rule)
    check("...while a certified configuration IS listed, with its Gate R "
          "numbers beside it",
          "ES" in certified_only and "CERTIFIED" in certified_only
          and "NQ" not in certified_only, certified_only)
    check("with no threshold on the handoff the reason is left off rather "
          "than guessed, and the cell stays a bare FAIL",
          dr.gate_r_reason(dict(dead, regime_starvation=None), {}) == "",
          dr.gate_r_reason(dict(dead, regime_starvation=None), {}))
    check("...but Stage 3's own starvation record outranks the arithmetic",
          dr.gate_r_reason(dict(dead, regime_starvation="[REGIME STARVATION] "
                                "Quadrant Q3 had only 36 holdout trades."),
                           rule) == "N")

    check("Certified counts Gate R's passes",
          fields.get("Certified → Incubator") == "1")
    check("an uncertified configuration is NOT a row - the table lists only "
          "what may be promoted", "ES" not in desc and "GC" not in desc)
    check("...but the card still COUNTS every configuration the run covered, "
          "so a filtered table can never read as a shorter certification run",
          fields.get("Configurations") == "3" and fields.get("Audited")
          == "1/3", f"{fields.get('Configurations')} {fields.get('Audited')}")
    check("...and says in words that the table is filtered",
          "CERTIFIED configurations only" in desc)

    digest = ((blob["results"][0].get("seal") or {})
              .get("strategy_code", {}).get("sha256", ""))
    promo = "\n".join(f["value"] for f in embed["fields"]
                      if f["name"].startswith(dr.PROMO_READY_TITLE))
    check("the seals are a PREFIX beside the parameters they seal, never the "
          "30-line dump of full digests that nobody verified from a phone",
          digest[:dr.SEAL_PREFIX_CHARS] in promo and digest not in promo)
    check("only STAGED configurations carry a seal - a checksum of a file "
          "the reader cannot find is worse than none",
          "ES " not in promo and "GC " not in promo, promo)

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
          "transcribed, never re-derived, so the row is filtered OFF the "
          "table and counted as nothing certified",
          dr.STAGE3_NO_ROWS_NOTE in e["description"]
          and "99.00" not in e["description"]
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


# --------------------------------------------------------------------------
# 12. The summary spans every timeframe the campaign certified
# --------------------------------------------------------------------------
def _result_at(d: Path, symbol: str, tf: str, passed: bool) -> dict:
    audit = _audit_for(d, PASS if passed else FAIL, f"{symbol}_{tf}")
    return {
        "symbol": symbol, "timeframe": tf, "path": audit,
        "status": {"A": PASS if passed else FAIL}, "passed": {"A": passed},
        "target_regime": TRENDING, "target_quadrant": "Q1",
        "params": {"fast": 5, "slow": 50}, "params_locked": True,
        "in_stage1": True,
        "gates": {"A": {"gate1": PASS, "gate2": PASS, "gate3": PASS,
                        GATE_R: PASS if passed else FAIL}},
        "regime_measured": {"A": {"profit_factor": 1.07 if passed else 0.98,
                                  "trade_count": 348 if passed else 12}},
        "regime_starvation": {"A": None if passed else
                              "[REGIME STARVATION] Quadrant Q1 (High "
                              "Volatility / Trending) had only 12 holdout "
                              "trades."},
        "retention": {"A": {}},
        "incubator": {"A": {"promoted": passed,
                            "dir": d / "inc" / "demo" if passed else None,
                            "seal": _seal_hashes(audit, None, audit, None)
                            if passed else None, "error": ""}},
        "exclude_days": [],
    }


def test_multi_timeframe_summary(tmp: Path) -> dict:
    print("\n12. stage3_audit_summary.json MERGES across timeframes")

    check("the merge keeps other timeframes and lets this run replace its own",
          merge_stage3_rows(
              [{"symbol": "CL", "timeframe": "5m", "version": "A"},
               {"symbol": "CL", "timeframe": "15m", "version": "A",
                "stale": True}],
              [{"symbol": "CL", "timeframe": "15m", "version": "A"}],
              {"15m"})[0]
          == [{"symbol": "CL", "timeframe": "15m", "version": "A"},
              {"symbol": "CL", "timeframe": "5m", "version": "A"}])
    check("...so a re-certification cannot leave a superseded verdict for the "
          "same pair standing beside the new one",
          all(not r.get("stale") for r in merge_stage3_rows(
              [{"symbol": "CL", "timeframe": "15m", "version": "A",
                "stale": True}],
              [{"symbol": "CL", "timeframe": "15m", "version": "A"}],
              {"15m"})[0]))

    d = tmp / "multitf"
    d.mkdir(parents=True, exist_ok=True)

    class _A5(_Args):
        tf = "5m"

    # Two Stage 3 runs, exactly as the pipeline drives them: one per timeframe.
    write_stage3_summary("demo", d, [_result_at(d, "CL", "15m", True)], [], [],
                         _Args(), [{"symbol": "CL", "timeframe": "15m"}],
                         "stage2 (exact pairs)")
    path = write_stage3_summary(
        "demo", d, [_result_at(d, "CL", "5m", True),
                    _result_at(d, "NQ", "5m", False)], [], [],
        _A5(), [{"symbol": s, "timeframe": "5m"} for s in ("CL", "NQ")],
        "stage2 (exact pairs)")
    blob = json.loads(path.read_text())
    rows = blob["results"]

    check("the second run did not overwrite the first - a card reading this "
          "file would otherwise announce one timeframe and silently drop the "
          "certifications from the others",
          {(r["symbol"], r["timeframe"]) for r in rows}
          == {("CL", "15m"), ("CL", "5m"), ("NQ", "5m")},
          str(sorted((r["symbol"], r["timeframe"]) for r in rows)))
    check("every timeframe the file indexes is named on it",
          blob["timeframes"] == ["15m", "5m"], str(blob.get("timeframes")))
    check("...and `timeframe` still records the run that wrote it, as a "
          "SEPARATE field rather than one that changes meaning",
          blob["timeframe"] == "5m")
    check("coverage is summed over every invocation, not the last one",
          blob["coverage"]["targets"] == 3
          and blob["coverage"]["certified"] == 2, str(blob["coverage"]))
    check("...and each invocation is on the record under its own timeframe",
          set(blob["runs"]) == {"15m", "5m"}
          and blob["runs"]["15m"]["certified"] == 1, str(list(blob["runs"])))

    audits = blob["audits"]
    check("the per-pair audits are indexed, one entry each",
          len(audits) == 3 and all(a["path"] for a in audits), str(len(audits)))
    check("...naming the file, its hash and the verdict inside it, so nothing "
          "has to glob a directory holding one audit per timeframe",
          all(a["exists"] and len(a["sha256"]) == 64 for a in audits))
    check("...and the audits stay AUTHORITATIVE - the index transcribes their "
          "verdict rather than restating one",
          {(a["symbol"], a["timeframe"], a["certified"]) for a in audits}
          == {("CL", "15m", True), ("CL", "5m", True), ("NQ", "5m", False)})
    check("a row with no audit file is kept in results and left out of the "
          "index - an entry pointing at nothing is worse than none",
          consolidated_audits([{"symbol": "X", "audit_file": None}]) == [])

    starved = [r for r in rows if r["symbol"] == "NQ"][0]
    check("a Gate R failure records WHETHER the quadrant starved, because "
          "'the edge died' and 'it never traded there again' are fixed by "
          "different work", "STARVATION" in (starved["regime_starvation"] or ""),
          str(starved["regime_starvation"]))

    # An audit read back off disk reproduces the row it was written from.
    replayed = audit_to_result(json.loads(Path(audits[0]["path"]).read_text()),
                               Path(audits[0]["path"]))
    check("a per-pair audit replays into the shape the summary indexes, so a "
          "rebuild re-scores nothing - the verdict is transcribed out of the "
          "file, including from the nested block when the top-level copy was "
          "never written",
          replayed["passed"] == {"A": True}
          and replayed["status"] == {"A": PASS}
          and replayed["gates"]["A"][GATE_R] == PASS, str(replayed["status"]))
    return blob


# --------------------------------------------------------------------------
# 13. The promotion section of the card
# --------------------------------------------------------------------------
def test_promotion_section(blob: dict) -> None:
    print("\n13. The card's promotion section")

    title, lines, pairs, hidden = dr.format_stage3_promotions("demo", blob)
    text = "\n".join(lines)
    check("headed READY FOR PROMOTION until something actually promoted",
          title == dr.PROMO_READY_TITLE, title)
    check("only CERTIFIED configurations are listed", len(pairs) == 2
          and all("NQ" not in p for p in pairs), str(pairs))
    check("the pairs are the compact answer to 'did anything certify' - the "
          "detail is in the block below them",
          all(p.startswith("`CL ") for p in pairs), str(pairs))
    check("each bullet carries its quadrant and the factor Gate R scored",
          text.count("`Q1`") == 2 and text.count("PF **1.07**") == 2,
          text[:200])
    check("the winning parameter plateau is on the bullet, abbreviated "
          "through the module's own collision-safe shortener",
          text.count("`f=5 s=50`") == 2, text[:200])
    check("...and the blended pair behind it, so an edge that collapsed from "
          "2.40 to 1.10 cannot read as a healthy 1.10",
          text.count("blended IS ") == 2, text[:400])
    check("ONE command promotes the certified set, and it is --promote-only: "
          "--auto-promote re-runs Stages 1-4 first and overwrites the very "
          "handoff this card was built from",
          "--promote-only" in dr.promotion_footer("demo", blob)
          and "--auto-promote" not in dr.promotion_footer("demo", blob),
          dr.promotion_footer("demo", blob))
    check("...and it is not one three-line bash block per pair any more",
          "backtest/promote.py" not in text, text[:200])
    check("a staged configuration says it is staged and NOT committed",
          "STAGED by Stage 3" in text and "nothing is committed" in text)

    promoted = dict(blob)
    promoted["auto_promotion"] = {
        "ran": True, "commit": "deadbee",
        "promotions": [{"symbol": "CL", "timeframe": "15m", "version": "A",
                        "promoted": True, "commit": "deadbee",
                        "incubator_dir": "/x/inc/demo", "error": ""},
                       {"symbol": "CL", "timeframe": "5m", "version": "A",
                        "promoted": False, "commit": None,
                        "error": "promote.py exited 1"}]}
    title2, lines2, _pairs2, _h = dr.format_stage3_promotions("demo", promoted)
    text2 = "\n".join(lines2)
    check("once auto-promotion ran the heading says so, with the commit",
          title2.startswith(dr.PROMO_DONE_TITLE) and "deadbee" in title2,
          title2)
    check("...and the heading is the RECORD's, never inferred from a seal - a "
          "sealed configuration was staged by Stage 3 and committed by nobody",
          dr.format_stage3_promotions("demo", blob)[0] == dr.PROMO_READY_TITLE)
    check("a promoted row names the commit it landed under",
          "promoted `deadbee`" in text2, text2[:200])
    check("a FAILED promotion says so and keeps its OWN per-pair command - "
          "the recovery path is one pair, not the set, and it cites that "
          "pair's own audit rather than the unsuffixed file, which holds "
          "whichever timeframe ran last",
          "NOT PROMOTED" in text2 and "promote.py exited 1" in text2
          and "gate_audit_CL_5m.json" in text2
          and "backtest/promote.py --strat demo --version A" in text2)
    check("the one-command footer counts only what is still outstanding, so "
          "it never tells a reader to re-run a promotion that committed",
          "1 configuration(s)" in dr.promotion_footer("demo", promoted),
          dr.promotion_footer("demo", promoted))

    embed = dr.build_stage3_embed("demo", promoted, source="/x/summary.json")
    names = [f["name"] for f in embed["fields"]]
    check("the section is its own field on the embed, and the command is a "
          "field of its own - the chunker splits a block on a blank line, and "
          "half a command is a command that runs and does something else",
          any(n.startswith(dr.PROMO_DONE_TITLE) for n in names)
          and dr.PROMO_COMMAND_FIELD in names, str(names))
    check("...and the embed still fits Discord's limit",
          dr._embed_size(embed) <= dr.MAX_EMBED_TOTAL,
          str(dr._embed_size(embed)))
    check("the timeframe field carries EVERY timeframe the summary indexes, "
          "not the last run's",
          all(t in [f for f in embed["fields"]
                    if f["name"] == "Timeframe"][0]["value"]
              for t in ("15m", "5m")))

    none_certified = dict(blob)
    none_certified["results"] = [dict(r, certified=False)
                                 for r in blob["results"]]
    _t, l3, p3, _ = dr.format_stage3_promotions("demo", none_certified)
    check("nothing certified means no section at all, rather than an empty "
          "heading that reads as a promotion nobody can find",
          not l3 and not p3)


# --------------------------------------------------------------------------
# 14. The card's TWO inputs: the campaign index and a single per-pair audit
# --------------------------------------------------------------------------
def _full_pair_audit(d: Path, symbol: str, tf: str, status: str,
                     pf: float = 1.22, n: int = 387) -> Path:
    """
    A `gate_audit_<SYMBOL>_<TF>.json` with everything Stage 3 puts on one:
    the measured quadrant numbers, the retention block, the entry filters and
    the incubator seal. `_audit_for` above is deliberately minimal (it exists
    to be sealed); the reporter's adapter has to survive a full one.
    """
    audit = {"status": status, "passed": status == PASS,
             "gates": {"gate1": {"status": FAIL}, "gate2": {"status": PASS},
                       "gate3": {"status": PASS},
                       GATE_R: {"status": status, "quadrant": "Q2",
                                "target_regime": HV_RANGING,
                                "regime_starvation": None,
                                "measured": {"profit_factor": pf,
                                             "trade_count": n,
                                             "win_rate": 53.75,
                                             "net_pnl": 96912.54}}}}
    return write_stage(d / GATE_AUDIT_FILE.format(symbol=f"{symbol}_{tf}"),
                       3, "demo", {
        "symbol": symbol, "timeframe": tf,
        "params": {"t3_period": 5, "sl_atr_mult": 2.0, "tp_atr_mult": None},
        "params_locked": True, "variants_tested": 162,
        "target_regime": HV_RANGING, "target_quadrant": "Q2",
        "in_sample": {"start": CHARTER_IS_START, "end": CHARTER_IS_END},
        "holdout": {"start": HOLDOUT_START, "end": "2026-01-01"},
        "entry_filters": {"news_filter": False, "exclude_days": [0]},
        "certification_rule": {"verdict_gate": GATE_R,
                               "min_profit_factor": MIN_REGIME_PROFIT_FACTOR,
                               "min_trades": MIN_REGIME_TRADES,
                               "aggregate_gates_are_advisory": True},
        "prop_firm_rules": {"applied": False},
        "versions": {"A": {
            "gate_audit": audit,
            "retention": retention_scores(
                {"profit_factor": 1.06, "sharpe": 0.40,
                 "max_drawdown_pct": -49.6, "win_rate": 52.0},
                {"profit_factor": 1.13, "sharpe": 0.90,
                 "max_drawdown_pct": -29.0, "win_rate": 53.8}),
        }},
        "status": {"A": status}, "passed": {"A": status == PASS},
        "incubator": {"A": {"promoted": status == PASS,
                            "dir": str(d / "inc" / "demo"),
                            "seal": {"strategy_code": {"sha256": "a" * 64},
                                     "winning_parameters":
                                         {"sha256": "b" * 64}},
                            "error": ""}},
    })


def test_pair_audit_ingestion(tmp: Path) -> None:
    print("\n14. A single gate_audit_<SYMBOL>_<TF>.json builds the same card "
          "as the summary")

    d = tmp / "pair"
    d.mkdir(parents=True, exist_ok=True)
    path = _full_pair_audit(d, "NQ", "1h", PASS)
    blob = json.loads(path.read_text())

    check("a per-pair audit is recognised as one, on CONTENT and not on the "
          "filename - an operator can point --audit at either shape",
          dr.is_pair_audit(blob) is True)

    # ------------------------------------------------------------------
    # THE DRIFT PIN. Both builders are handed the SAME file and must produce
    # the same row: `audit_gates.write_stage3_summary` writes the index and
    # `discord_reporter.stage3_rows_from_audit` reads a pair audit directly,
    # and the reporter cannot import the first (it pulls in the engine and
    # vectorbtpro, and a notifier that dies on the simulation stack is a quiet
    # pipeline). Two transcriptions of one file are two things that can
    # disagree, and a card that disagreed with the index would announce a
    # verdict Stage 3 never wrote.
    # ------------------------------------------------------------------
    summary = json.loads(write_stage3_summary(
        "demo", d, [audit_to_result(blob, path)], [], [], _Args(),
        [{"symbol": "NQ", "timeframe": "1h"}], "test").read_text())
    want = summary["results"][0]
    got = dr.stage3_rows_from_audit(blob, path)[0]
    diff = {k: (want.get(k), got.get(k)) for k in set(want) | set(got)
            if want.get(k) != got.get(k)}
    check("the reporter's adapter and Stage 3's own summary writer produce "
          "an IDENTICAL row from one audit - field for field, same names",
          not diff and set(want) == set(got), str(diff)[:300])

    rows = dr.stage3_rows_from_audit(blob, path)
    check("one row per VERSION, since A and B reach separate verdicts",
          len(rows) == 1 and rows[0]["version"] == "A")
    check("Gate R's quadrant numbers are transcribed, not re-derived",
          rows[0]["oos_profit_factor"] == 1.22
          and rows[0]["oos_trade_count"] == 387
          and rows[0]["gate_regime"] == PASS)
    check("...and the certification flag is the audit's own `passed`",
          rows[0]["certified"] is True)
    check("the audit is hashed into the row, so a card cannot drift from the "
          "verdict it points at", len(rows[0]["audit_sha256"]) == 64)

    # The card itself.
    assembled = dr.stage3_blob_from_audits([(blob, path)])
    embed = dr.build_stage3_embed("demo", assembled, source=path)
    fields = {f["name"]: f["value"] for f in embed["fields"]}
    table = embed["description"].split("```")[1]
    row = [ln for ln in table.splitlines() if ln.startswith("NQ")]
    check("the card renders the ONE configuration as a one-row leaderboard",
          len(row) == 1, table)
    cells = row[0].split() if row else []
    check("...carrying SYM TF QD GATE R PF N STATUS",
          cells == ["NQ", "1h", "Q2", "PASS", "1.22", "387", "CERTIFIED"],
          str(cells))
    check("Audited counts the audit that was read", fields["Audited"] == "1/1",
          fields["Audited"])
    check("Certified -> Incubator is Gate R's own flag",
          fields["Certified \u2192 Incubator"] == "1")
    check("Configurations is 1, not a campaign's count",
          fields["Configurations"] == "1")
    check("the timeframe is the audit's, never the module's declaration",
          fields["Timeframe"] == "`1h`", fields["Timeframe"])
    check("both windows come off the audit itself",
          f"{CHARTER_IS_START}" in embed["description"]
          and "2026-01-01" in embed["description"])
    check("coverage says what these counts DESCRIBE - the audits read, not "
          "the campaign Stage 3 was asked to certify",
          "cannot appear here" in assembled["coverage"]["rule"])

    # A FAILED pair audit is still a card, and it certifies nothing.
    bad = _full_pair_audit(d, "ES", "1h", FAIL, pf=0.81, n=120)
    fail_blob = json.loads(bad.read_text())
    fail_rows = dr.stage3_rows_from_audit(fail_blob, bad)
    check("a Gate R FAIL is transcribed as one and certifies nothing",
          fail_rows[0]["gate_regime"] == FAIL
          and fail_rows[0]["certified"] is False)
    fail_embed = dr.build_stage3_embed(
        "demo", dr.stage3_blob_from_audits([(fail_blob, bad)]), source=bad)
    check("...and its card carries the no-rows note rather than a table of "
          "one uncertified row, which is what the filter is for",
          dr.STAGE3_NO_ROWS_NOTE in fail_embed["description"])


def test_stage3_input_resolution(tmp: Path) -> None:
    print("\n15. Which file the Stage 3 card is built from")

    d = tmp / "resolve"
    d.mkdir(parents=True, exist_ok=True)
    pair = _full_pair_audit(d, "NQ", "1h", PASS)
    _full_pair_audit(d, "NQ", "30m", PASS, pf=1.09, n=1077)
    # The unsuffixed duplicate of whichever timeframe ran last. It must never
    # be indexed beside the suffixed file it copies - one verdict under two
    # names is a campaign that reads as twice the size it was.
    (d / GATE_AUDIT_FILE.format(symbol="NQ")).write_text(pair.read_text())

    blob, src, what = dr.resolve_stage3_input("demo", audit=pair, out_dir=d)
    check("--audit on a pair file resolves to that ONE verdict",
          len(dr.stage3_rows(blob)) == 1 and Path(src) == pair, what)

    found = dr.discover_pair_audits("demo", d)
    check("discovery reads the SUFFIXED audits only - the unsuffixed file is "
          "a duplicate of the last timeframe, and indexing both counts one "
          "verdict twice", len(found) == 2, str([p.name for p in found]))

    blob, src, what = dr.resolve_stage3_input("demo", out_dir=d)
    check("with no summary on disk and no flag, EVERY per-pair audit is "
          "indexed - picking one would announce a single certification while "
          "the rest sat on disk unread",
          len(dr.stage3_rows(blob)) == 2, what)
    check("...and the card spans both timeframes",
          dr.stage3_timeframes(blob) == ["1h", "30m"],
          str(dr.stage3_timeframes(blob)))

    summary = write_stage3_summary(
        "demo", d, [audit_to_result(json.loads(pair.read_text()), pair)],
        [], [], _Args(), [{"symbol": "NQ", "timeframe": "1h"}], "test")
    blob, src, what = dr.resolve_stage3_input("demo", out_dir=d)
    check("once the summary exists it is PREFERRED, since it spans the whole "
          "campaign rather than the pairs that happen to be on disk",
          Path(src) == summary, what)
    blob, src, _ = dr.resolve_stage3_input("demo", summary=summary, out_dir=d)
    check("--summary reads the index", Path(src) == summary)

    ok, msg = raises(lambda: dr.resolve_stage3_input(
        "demo", summary=pair, out_dir=d), ValueError)
    check("--summary REFUSES a per-pair audit rather than adapting it - a "
          "flag that accepts either shape makes the two words mean nothing",
          ok and "--audit" in msg, msg[:120])
    ok, msg = raises(lambda: dr.resolve_stage3_input(
        "demo", audit=pair, summary=summary, out_dir=d), ValueError)
    check("both flags at once is refused rather than one silently winning",
          ok, msg[:120])

    empty = tmp / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    ok, msg = raises(lambda: dr.resolve_stage3_input("demo", out_dir=empty),
                     FileNotFoundError)
    check("nothing on disk names BOTH files it looked for and the command "
          "that writes them, rather than posting an empty card",
          ok and STAGE3_SUMMARY_FILE in msg and "audit_gates.py" in msg,
          msg[:160])



# --------------------------------------------------------------------------
# The sample floor, and what a starved quadrant is allowed to become
# --------------------------------------------------------------------------
def test_regime_starvation_is_never_certified() -> None:
    print("\nGate R's sample floor: 30 trades inside the designated quadrant")

    from backtest.audit_gates import (MIN_REGIME_TRADES, charter_audit,
                                      regime_gate)

    REGIME = "High Volatility / Ranging"

    def profile(n, pf=2.50):
        return {"regime_breakdown": {REGIME: {
            "trade_count": n, "profit_factor": pf, "win_rate": 0.55,
            "net_pnl": 1000.0}}}

    check("the floor is 30 and it comes from Stage 1's screen, not a second "
          "number here", MIN_REGIME_TRADES == 30, str(MIN_REGIME_TRADES))

    # The boundary, from both sides. A floor written `>` instead of `>=` fails
    # exactly one configuration in the whole campaign and nothing says so.
    for n, want in ((0, False), (1, False), (29, False), (30, True),
                    (31, True)):
        g = regime_gate(profile(n), REGIME, 1.00, MIN_REGIME_TRADES)
        check(f"{n} holdout trade(s) in the quadrant "
              f"{'PASSES' if want else 'FAILS'} the sample floor",
              (g["status"] == "PASS") is want, g["status"])

    # A magnificent profit factor over three trades is the case the floor
    # exists for, and it must not survive it.
    starved = regime_gate(profile(3, pf=999.0), REGIME, 1.00,
                          MIN_REGIME_TRADES)
    check("a 999 profit factor over 3 trades still fails",
          starved["status"] == "FAIL", starved["status"])
    check("...the profit-factor row itself PASSES, so the two reasons stay "
          "distinguishable",
          [c["status"] for c in starved["checks"]] == ["FAIL", "PASS"],
          str([c["status"] for c in starved["checks"]]))
    check("...and REGIME_STARVATION is attached with a message an operator "
          "can act on",
          bool((starved.get("regime_starvation") or {}).get("message")),
          str((starved.get("regime_starvation") or {}).get("message"))[:90])

    # Starvation is keyed on the TRADE COUNT, not on the overall verdict. A
    # quadrant that traded 90 times and lost is a different finding and must
    # not carry the diagnostic that says it was never sampled.
    lost = regime_gate(profile(90, pf=0.60), REGIME, 1.00, MIN_REGIME_TRADES)
    check("a quadrant that traded enough and LOST is not marked starved",
          lost["status"] == "FAIL" and lost.get("regime_starvation") is None,
          str(lost.get("regime_starvation")))

    # And the verdict: a starved configuration is NOT CERTIFIED. Gate R is the
    # verdict gate under charter clause 3, so this is the whole promotion
    # decision - `promote` refuses anything whose status is not PASS.
    survived = {"ruined": False, "max_drawdown_pct": -18.0}
    audit = {"status": "PASS", "passed": True, "gates": {
        "gate1": {"status": "PASS"}, "gate2": {"status": "PASS"},
        "gate3": {"status": "PASS"}}}
    folded = charter_audit(dict(audit), starved, {"metrics": {}},
                           in_sample_metrics=survived,
                           require_all_quadrants=False)
    check("a starved quadrant is NOT CERTIFIED even with Gates 1-3 all "
          "passing", folded["passed"] is False, str(folded["passed"]))
    check("...and the verdict gate is named as Gate R",
          folded["verdict_gate"] == GATE_R, folded["verdict_gate"])
    check("...while the advisory roll-up is preserved, so the loosening stays "
          "legible", folded.get("aggregate_status") == "PASS",
          str(folded.get("aggregate_status")))

    # The same fold with a healthy quadrant certifies, which is what makes the
    # check above a measurement rather than a tautology.
    healthy = regime_gate(profile(61), REGIME, 1.00, MIN_REGIME_TRADES)
    ok = charter_audit(dict(audit), healthy, {"metrics": {}},
                       in_sample_metrics=survived,
                       require_all_quadrants=False)
    check("a quadrant with 61 trades and PF 2.50 IS certified",
          ok["passed"] is True and ok["status"] == "PASS", ok["status"])


def test_version_b_survivor_is_audited_as_version_b() -> None:
    print("\nA Version B survivor is certified on Version B, with no --ml")

    from backtest.audit_gates import resolve_version_b, stage2_targets

    # Stage 2's summary row is where Stage 3 reads the version from, and it is
    # the hop the answer used to die on.
    targets = stage2_targets({"results": [
        {"symbol": "NQ", "timeframe": "30m", "status": "OPTIMIZED",
         "stage1_version": "B", "quadrant": "Q2"},
        {"symbol": "ES", "timeframe": "30m", "status": "OPTIMIZED",
         "stage1_version": "A", "quadrant": "Q3"},
    ]}, "30m")
    versions = {t["symbol"]: t["stage1_version"] for t in targets}
    check("Stage 2's summary row carries the version into Stage 3's targets",
          versions == {"NQ": "B", "ES": "A"}, str(versions))

    args = SimpleNamespace(ml=False, no_stage1_ml=False)
    verdicts = {t["symbol"]: resolve_version_b(t, args)[0] for t in targets}
    check("the B survivor is audited as Version B without --ml being typed",
          verdicts == {"NQ": True, "ES": False}, str(verdicts))

def test_all_quadrant_gate() -> None:
    """
    Gate Q: the edge held in EVERY quadrant, not only its designated one.

    Requested 2026-09-07 and enforced by DEFAULT, so the case that matters
    most is the specialist: Stage 1 designates one home quadrant and the live
    supervisor stands the strategy down in the other three, so a Q4 range fade
    never trades Q1 and fails this gate by construction. That is the intended
    effect of the request rather than a defect, and it is pinned here so
    nobody later reads the resulting wall of refusals as a bug in the gate.
    """
    from backtest.audit_gates import (all_quadrant_gate, ALL_QUADRANTS_FAILED,
                                      GATE_Q)
    from backtest.profiler import REGIMES

    print("\nGate Q: all four quadrants, individually (2026-09-07)")

    def _qrisk(net, sharpe, dd, n=40):
        return {"trade_count": n, "net_pnl": net, "sharpe_trade": sharpe,
                "max_drawdown_pnl": dd, "profit_factor": 1.3,
                "win_rate": 50.0}

    healthy = {"regime_breakdown": {r: _qrisk(5000, 0.20, -3000)
                                    for r in REGIMES}}
    check("four healthy quadrants PASS",
          all_quadrant_gate(healthy)["status"] == PASS)

    # The specialist. Only Q4 has any trades at all.
    specialist = {"regime_breakdown": {REGIMES[3]: _qrisk(9000, 0.30, -2000)}}
    spec = all_quadrant_gate(specialist)
    check("a quadrant the strategy NEVER TRADED is a FAIL, not a skip",
          spec["status"] == FAIL
          and sum(1 for r in spec["quadrants"] if r["status"] != PASS) == 3,
          spec["status"])
    check("...and the note says a specialist is EXPECTED to fail it, and "
          "that the gate is advisory",
          "specialist" in spec["note"].lower()
          and "advisory" in spec["note"].lower()
          and "--require-all-quadrants" in spec["note"])

    # Each bar, one at a time, so a pass cannot come from the wrong column.
    for label, bad in (
            ("negative expectancy", _qrisk(-800, 0.20, -3000)),
            ("Sharpe below the floor", _qrisk(5000, 0.01, -3000)),
            ("too few trades to measure", _qrisk(5000, 0.20, -3000, n=6)),
            ("no Sharpe recorded at all", _qrisk(5000, None, -3000))):
        prof = {"regime_breakdown": {r: _qrisk(5000, 0.20, -3000)
                                     for r in REGIMES}}
        prof["regime_breakdown"][REGIMES[0]] = bad
        check(f"one quadrant with {label} fails the gate",
              all_quadrant_gate(prof)["status"] == FAIL, label)

    check("a deep quadrant drawdown no longer fails the gate: the dollar cap "
          "was retired 2026-09-08 as account governance, and belongs to "
          "CrossTrade",
          all_quadrant_gate({"regime_breakdown": {
              r: _qrisk(5000, 0.20, -22_000) for r in REGIMES}})["status"]
          == PASS)
    check("...but it is still MEASURED and reported on the row, so retiring "
          "the bar did not retire the evidence",
          all(r["max_drawdown_pnl"] == -22_000 for r in all_quadrant_gate(
              {"regime_breakdown": {r: _qrisk(5000, 0.20, -22_000)
                                    for r in REGIMES}})["quadrants"]))
    check("...and a caller that passes a cap explicitly still gets the bar",
          all_quadrant_gate({"regime_breakdown": {
              r: _qrisk(5000, 0.20, -22_000) for r in REGIMES}},
              max_drawdown=15_000.0)["status"] == FAIL)

    check("no breakdown at all is NOT EVALUATED, which is not a pass",
          all_quadrant_gate({})["status"] == NOT_EVALUATED)

    # The fold: Gate Q can refuse a certification Gate R passed, under its own
    # status token, and waiving it is recorded by the gate's ABSENCE.
    alive = {"ruined": False, "final_equity": 138_400.0,
             "max_drawdown_pct": -17.5}
    passing_r = regime_gate(_profile(**{TRENDING: _q(1.42, 88)}), TRENDING)
    ret = retention_scores({"profit_factor": 2.4}, {"profit_factor": 1.1})
    # Spelled out rather than reaching for `test_no_aggregate_pruning`'s
    # nested `_aggregate`: these sections are called independently by main()
    # and a helper borrowed across them would tie their order together.
    agg = {"version": "A", "status": PASS, "passed": True,
           "gates": {"gate1": {"name": "g1", "status": PASS, "checks": []},
                     "gate2": {"name": "g2", "status": PASS, "checks": []},
                     "gate3": {"name": "g3", "status": PASS, "checks": []}}}

    # ADVISORY by default from 2026-09-08. The specialist that Gate Q was
    # written to refuse is exactly the strategy this pipeline is built to
    # produce - Stage 1 designates one home quadrant and the supervisor stands
    # it down in the other three - so the gate is measured, reported, and
    # certifies nothing on its own.
    advisory = charter_audit(agg, passing_r, ret, in_sample_metrics=alive,
                             holdout_profile=specialist)
    check("a specialist that Gate Q fails is still CERTIFIED: the gate is "
          "advisory and Gate R is the verdict",
          advisory["passed"] is True and advisory["status"] == PASS,
          advisory["status"])
    check("...and Gate Q is still recorded, still FAIL, and marked advisory - "
          "the measurement does not disappear with its authority",
          GATE_Q in advisory["gates"]
          and advisory["gates"][GATE_Q]["status"] == FAIL
          and advisory["gates"][GATE_Q]["advisory"] is True)

    refused = charter_audit(agg, passing_r, ret, in_sample_metrics=alive,
                            holdout_profile=specialist,
                            require_all_quadrants=True)
    check("Gate Q refuses a certification Gate R passed when it is ARMED",
          refused["passed"] is False
          and refused["status"] == ALL_QUADRANTS_FAILED, refused["status"])
    check("...under its OWN token, so it is never read as a blown account",
          refused["status"] != "FAILED_RUIN_CHECK"
          and GATE_Q in refused["gates"]
          and refused["gates"][GATE_Q]["advisory"] is False)

    passed = charter_audit(agg, passing_r, ret, in_sample_metrics=alive,
                           holdout_profile=healthy)
    check("...and four healthy quadrants certify, so the refusal above is a "
          "measurement and not a tautology",
          passed["passed"] is True and passed["status"] == PASS,
          passed["status"])

    waived = charter_audit(agg, passing_r, ret, in_sample_metrics=alive,
                           holdout_profile=specialist,
                           require_all_quadrants=False)
    check("an advisory Gate Q reports its REAL status, never a written PASS: "
          "a gate switched off must not be readable as four quadrants it "
          "measured and liked",
          waived["passed"] is True
          and waived["gates"][GATE_Q]["status"] == FAIL
          and waived["all_quadrant_gate"] is not None)

    # Ruin outranks it: a blown account must not be relabelled by a
    # robustness bar it also missed.
    blown = {"ruined": True, "final_equity": -78_868.0,
             "max_drawdown_pct": -194.0}
    both = charter_audit(agg, passing_r, ret, in_sample_metrics=blown,
                         holdout_profile=specialist)
    check("a blown account still reads as RUIN, not as a quadrant failure",
          both["status"] == "FAILED_RUIN_CHECK", both["status"])


def test_quadrant_risk_is_quadrant_local() -> None:
    """
    `profiler._quadrant_risk` - the two figures Gate Q reads.

    Both are quadrant-local and neither is an account number. The Sharpe is
    UNANNUALISED because a non-contiguous subset of the calendar has no time
    basis that is not invented, and the drawdown is in dollars because a
    percent would need a capital base that describes the whole account.
    """
    from backtest.profiler import _quadrant_risk

    print("\nGate Q's inputs: quadrant-local risk")

    mixed = _quadrant_risk([100, -50, 80, -30, 60])
    check("Sharpe is mean/stdev over the quadrant's own trades",
          abs(mixed["sharpe_trade"] - 0.473) < 0.01, str(mixed))
    check("drawdown is the deepest fall of the quadrant's cumulative P&L",
          mixed["max_drawdown_pnl"] == -50.0, str(mixed))
    check("a quadrant that only made new highs draws 0, not None",
          _quadrant_risk([10, 20, 30])["max_drawdown_pnl"] == 0.0)
    check("ONE trade has no dispersion, so the Sharpe is None rather than "
          "infinite - and Gate Q refuses a None",
          _quadrant_risk([42.0])["sharpe_trade"] is None)
    check("an empty quadrant reports None for both",
          _quadrant_risk([]) == {"sharpe_trade": None,
                                 "max_drawdown_pnl": None})
    check("a losing quadrant carries a negative Sharpe through",
          _quadrant_risk([-10, -20, -5])["sharpe_trade"] < 0)


def test_the_starvation_only_fallback() -> None:
    """
    Gate R may certify on the PRE-DECLARED secondary, and only on starvation.

    Requested 2026-09-08. The rule that keeps this one test rather than two:
    a primary that traded `min_trades` times and lost is a hard FAIL with no
    second look. Starvation is different in kind - the primary was never
    measured - so a quadrant Stage 1 declared IN SAMPLE stands in for it.

    Pinned here because every part of it is load bearing and none of it is
    visible from the value alone: a fallback that also fired on a performance
    failure would be the best-of-N selection this gate exists to prevent, and
    a fallback that accepted an INELIGIBLE runner-up would certify in an
    environment the screen refused.
    """
    from backtest.audit_gates import (regime_gate, CERTIFIED_ON_PRIMARY,
                                      CERTIFIED_ON_FALLBACK)
    print("\nGate R: the starvation-only secondary fallback (2026-09-08)")

    HVT = "High Volatility / Trending"

    def prof(**rows):
        return {"regime_breakdown": {
            r: {"trade_count": n, "profit_factor": pf, "win_rate": 50.0,
                "net_pnl": 1000.0, "sharpe_trade": 0.1,
                "max_drawdown_pnl": -500.0}
            for r, (n, pf) in rows.items()}}

    eligible = [{"regime": HVT, "quadrant": "Q1", "eligible": True,
                 "profit_factor": 1.01, "trade_count": 411, "net_pnl": 9000.0,
                 "score": 5000.0}]

    starved = prof(**{LV_RANGING: (16, 1.40), HVT: (300, 1.20)})
    g = regime_gate(starved, LV_RANGING, 1.00, 30, secondary_regimes=eligible)
    check("a STARVED primary falls back to the pre-declared secondary",
          g["status"] == PASS
          and g["certified_on"] == CERTIFIED_ON_FALLBACK, str(g["status"]))
    check("...and the CERTIFIED quadrant becomes the one that carried it, so "
          "the live supervisor is handed a single environment",
          g["quadrant"] == "Q1" and g["target_regime"] == HVT, g["quadrant"])
    check("...with the designation kept beside it rather than overwritten",
          g["primary_quadrant"] == "Q4" and g["primary_regime"] == LV_RANGING)
    check("...and the starvation diagnostic still reports the PRIMARY's "
          "drought, which is why the fallback ran at all",
          bool(g["regime_starvation"]))

    # THE RULE THAT MAKES IT ONE TEST AND NOT TWO.
    lost = prof(**{LV_RANGING: (300, 0.88), HVT: (300, 1.20)})
    g = regime_gate(lost, LV_RANGING, 1.00, 30, secondary_regimes=eligible)
    check("a primary that traded ENOUGH and LOST is a hard FAIL, and the "
          "secondary is never looked at",
          g["status"] == FAIL and g["fallback"] is None
          and g["certified_on"] == CERTIFIED_ON_PRIMARY, str(g["fallback"]))

    # An INELIGIBLE runner-up is not an environment the screen would designate.
    ineligible = [{"regime": HVT, "quadrant": "Q1", "eligible": False,
                   "profit_factor": 1.40, "trade_count": 12}]
    g = regime_gate(starved, LV_RANGING, 1.00, 30,
                    secondary_regimes=ineligible)
    check("an INELIGIBLE secondary is refused - the screen declined that "
          "quadrant and a fallback cannot reach past it",
          g["status"] == FAIL and g["fallback"] is None)

    # The fallback can fail on its own terms.
    both_thin = prof(**{LV_RANGING: (16, 1.40), HVT: (9, 1.20)})
    g = regime_gate(both_thin, LV_RANGING, 1.00, 30,
                    secondary_regimes=eligible)
    check("a secondary that is itself starved FAILS, and the target stays "
          "the designation rather than moving to a quadrant that certified "
          "nothing",
          g["status"] == FAIL and g["quadrant"] == "Q4"
          and g["fallback"]["status"] == FAIL, str(g["quadrant"]))

    # And with nothing declared, the gate is exactly what it was.
    g = regime_gate(starved, LV_RANGING, 1.00, 30, secondary_regimes=None)
    check("no declared secondary leaves the pre-2026-09-08 behaviour intact",
          g["status"] == FAIL and g["fallback"] is None
          and g["certified_on"] == CERTIFIED_ON_PRIMARY)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage3charter_") as td:
        tmp = Path(td)
        test_ingestion(tmp)
        test_holdout_isolation()
        test_gate_r()
        test_regime_starvation()
        test_no_aggregate_pruning()
        test_ruin_guard()
        test_no_prop_firm_rules()
        test_retention()
        test_seal_and_incubator(tmp)
        blob = test_summary_handoff(tmp)
        test_stage3_card(blob)
        test_mode_resolution()
        test_cli(tmp, blob)
        multi = test_multi_timeframe_summary(tmp)
        test_promotion_section(multi)
        test_pair_audit_ingestion(tmp)
        test_stage3_input_resolution(tmp)
        test_regime_starvation_is_never_certified()
        test_version_b_survivor_is_audited_as_version_b()
        test_all_quadrant_gate()
        test_the_starvation_only_fallback()
        test_quadrant_risk_is_quadrant_local()

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
