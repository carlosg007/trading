"""
tests/test_stage1_charter.py - the Regime-Switching Incubator Stage 1 Charter.

Location: ~/src/trading/tests/test_stage1_charter.py

Covers the two modules the charter binds:

    backtest/baseline.py          the dual simulation, the quadrant screen, and
                                  the surviving_assets.json handoff
    backtest/discord_reporter.py  the Stage 1 leaderboard card

The five clauses, and the way each one fails silently if nothing pins it:

  1. **Dual simulation.** Survival is decided on EITHER version, so a run that
     screened Version A alone cannot distinguish "neither version carried it"
     from "only one of them was asked". The default has to BE both, and the
     in-sample window has to stop at 2022-12-31 - a screen that reads into the
     Stage 3 holdout picks its survivors on the bars Gate 3 later measures
     retention against, and every stage downstream then reports a holdout it
     has already been shown.
  2. **Quadrant attribution.** Q1..Q4 are the charter's names for the four
     ADX x ATR quadrants. A transposed map would move every trade between
     quadrants with every total still adding up, so the ids are checked against
     `mdlib.regimes`, which is where the encoding is written down.
  3. **The hurdle is the QUADRANT, never the blend.** Both bars bind on the
     same quadrant, and no aggregate Sharpe or profit factor may drop a
     configuration. Pinned with a run whose blended numbers are dreadful.
  4. **Optimal tagging.** The winner is named; the other three are muted.
  5. **The handoff.** symbol, tf, version, optimal_regime, quadrant metrics -
     and `status: DROPPED` when no quadrant qualified.

Runs without the lake and without a network: every fixture is a synthetic
profile dict, and the Discord transport is exercised in --dry-run only.

    python tests/test_stage1_charter.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import backtest.discord_reporter as dr                            # noqa: E402
from backtest.baseline import (CHARTER_IS_END, CHARTER_IS_START,   # noqa: E402
                               QUADRANT_ID, _row, best_quadrant,
                               build_parser, dropped_from,
                               kill_switch_regimes, quadrant_id,
                               screen, screen_results_from,
                               surviving_pairs_from, survivors_leaderboard)
from backtest.pipeline import SURVIVORS_FILE, write_stage          # noqa: E402
from backtest.profiler import REGIMES                              # noqa: E402
from mdlib import regimes as regime_cache                          # noqa: E402

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
    return bool(ok)


def raises(fn, exc=Exception) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:
        return True, f"{type(e).__name__}: {e}"
    except Exception as e:                                        # noqa: BLE001
        return False, f"wrong exception: {type(e).__name__}: {e}"
    return False, "did not raise"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
_KEYS = {"hvt": REGIMES[0], "hvr": REGIMES[1],
         "lvt": REGIMES[2], "lvr": REGIMES[3]}

# Deliberately awful blended metrics. Every row built on these has a negative
# Sharpe, a profit factor well under 1.00 and a 41% drawdown - the profile of a
# configuration the OLD blended screen dropped on sight. Clause 3 says none of
# that may decide anything.
AWFUL = {"ok": True, "trade_count": 812, "profit_factor": 0.41,
         "sharpe": -3.90, "sortino": -2.7, "calmar": -0.2, "win_rate": 0.28,
         "max_drawdown_pct": -41.0, "total_pnl": -88_400.0,
         "gross_pnl": 12_000.0, "total_costs": 6_100.0, "n_days": 2_517}

EMPTY_DOW = pd.DataFrame(columns=["weekday", "day", "trades", "net_pnl",
                                  "win_rate", "profit_factor"])


def profile(**quadrants) -> dict:
    """`profile(hvt=(1.28, 120))` -> one version's four-quadrant breakdown."""
    breakdown, total = {}, 0
    for key, (pf, n) in quadrants.items():
        breakdown[_KEYS[key]] = {"trade_count": n, "profit_factor": pf,
                                 "win_rate": 55.0, "net_pnl": 100.0 * n}
        total += n
    return {"regime_breakdown": breakdown, "trades_profiled": total,
            "trades_unplaced": 0, "regime_source": "precomputed_cache"}


def row(symbol: str, tf: str, profiles: dict, ml: bool = True) -> dict:
    ok, why, best = screen(profiles)
    metrics_b = dict(AWFUL) if (ml and profiles.get("B")) else (
        dict(AWFUL) if ml else None)
    return _row(symbol, tf, dict(AWFUL), metrics_b, ok, why, EMPTY_DOW,
                10_000, 1.0, profiles=profiles, best=best)


# --------------------------------------------------------------------------
# 1. Dual simulation, over the charter window
# --------------------------------------------------------------------------
def test_dual_simulation_defaults() -> None:
    print("\n1. Stage 1 runs BOTH versions, over the charter's in-sample window")
    parser = build_parser()
    args = parser.parse_args(["--strat", "demo"])

    check("Version B is ON by default - survival is decided on EITHER "
          "version, so a screen that ran only the rules answers a narrower "
          "question than the one it reports", args.ml is True, str(args.ml))
    check("...--no-ml is the deliberate way to decline it",
          parser.parse_args(["--strat", "demo", "--no-ml"]).ml is False)
    check("...and the older explicit --ml still means the same thing",
          parser.parse_args(["--strat", "demo", "--ml"]).ml is True)

    check("the in-sample window defaults to the charter's 2013-01-01",
          args.start == CHARTER_IS_START == "2013-01-01", str(args.start))
    check("...through 2022-12-31, which stops SHORT of the Stage 3 holdout",
          args.end == CHARTER_IS_END == "2022-12-31", str(args.end))
    check("...and neither default is None any more - reading the lake end to "
          "end would screen on the held-back years",
          args.start is not None and args.end is not None)

    over = parser.parse_args(["--strat", "demo", "--end", "2026-01-01"])
    check("an operator can still override the window explicitly",
          over.end == "2026-01-01", over.end)


# --------------------------------------------------------------------------
# 2. Quadrant attribution
# --------------------------------------------------------------------------
def test_quadrant_ids() -> None:
    print("\n2. Q1..Q4 are the charter's four ADX(14) x ATR(14) quadrants")
    check("Q1 is High Volatility / Trending",
          quadrant_id(REGIMES[0]) == "Q1", str(quadrant_id(REGIMES[0])))
    check("Q2 is High Volatility / Ranging", quadrant_id(REGIMES[1]) == "Q2")
    check("Q3 is Low Volatility / Trending", quadrant_id(REGIMES[2]) == "Q3")
    check("Q4 is Low Volatility / Ranging", quadrant_id(REGIMES[3]) == "Q4")

    check("the ids are INVERTED from mdlib.regimes rather than spelled out "
          "again - one map, so 'Q1' and its label cannot come apart",
          all(QUADRANT_ID[regime_cache.QUADRANT_LABELS[q]] == f"Q{q}"
              for q in (1, 2, 3, 4)))
    check("the cache's 0 (indicator warm-up) is NOT a quadrant and gets no id",
          regime_cache.QUADRANT_LABELS[0] not in QUADRANT_ID,
          regime_cache.QUADRANT_LABELS[0])
    check("an absent or unknown regime yields None, never a placeholder 'Q0' "
          "that would read as a fifth environment",
          quadrant_id(None) is None and quadrant_id("Some Other Regime") is None)


# --------------------------------------------------------------------------
# 3. The survival hurdle
# --------------------------------------------------------------------------
def test_hurdle() -> None:
    print("\n3. The hurdle is PF >= 1.00 AND positive net P&L AND "
          "N >= max(50, 10% of placed), on ONE quadrant")
    keep, why, best = screen({"A": profile(hvt=(1.00, 50)), "B": None})
    check("exactly 1.00 over exactly 50 trades is on the boundary and clears",
          keep and best["trade_count"] == 50, why)
    check("0.99 over 400 trades does not clear",
          not screen({"A": profile(hvt=(0.99, 400)), "B": None})[0])
    check("2.40 over 49 trades does not clear - a factor over 49 trades is "
          "not an environment",
          not screen({"A": profile(hvt=(2.40, 49)), "B": None})[0])
    check("the floor SCALES: 60 of 5,000 placed trades is 1.2% of the run "
          "and does not clear, where the flat 50 would have",
          not screen({"A": profile(hvt=(1.80, 60), lvr=(0.90, 4940)),
                      "B": None})[0])
    check("the best factor and the largest count in DIFFERENT quadrants do "
          "not combine into a pass",
          not screen({"A": profile(hvt=(1.90, 11), lvr=(0.90, 900)),
                      "B": None})[0])

    print("\n3b. NOTHING is dropped on an unsegmented aggregate")
    profiles = {"A": profile(hvt=(1.31, 96), hvr=(0.42, 300),
                             lvt=(0.55, 210), lvr=(0.30, 206)), "B": None}
    r = row("NQ", "15m", profiles, ml=False)
    check("a configuration with a -3.90 blended SHARPE survives on one "
          "quadrant - Sharpe is not a Stage 1 screen at any aggregation",
          r["survived"] and r["sharpe_a"] == -3.90, str(r["sharpe_a"]))
    check("...and a 0.41 blended PROFIT FACTOR does not drop it either - the "
          "screen asks whether there is an environment, not whether it made "
          "money on every bar",
          r["survived"] and r["profit_factor_a"] == 0.41,
          str(r["profit_factor_a"]))
    check("...nor does a 41% max drawdown, which is a Gate 1 question and "
          "not a Stage 1 one",
          r["survived"] and r["max_drawdown_pct_a"] == -41.0)
    check("the reason names the QUADRANT that carried it, not the blend",
          REGIMES[0] in r["reason"] and "1.31" in r["reason"], r["reason"])

    print("\n3c. Either version can carry a configuration")
    both = screen({"A": profile(hvt=(0.87, 400)), "B": profile(lvt=(1.31, 60))})
    check("Version B carries one Version A failed, and is named as such",
          both[0] and both[2]["version"] == "B", both[1])
    check("...with Version B not run, Version A alone decides",
          not screen({"A": profile(hvt=(0.87, 400)), "B": None})[0])


# --------------------------------------------------------------------------
# 4. Optimal tagging and the muted kill switches
# --------------------------------------------------------------------------
def test_optimal_tagging() -> None:
    print("\n4. The winning quadrant is tagged; the other three are muted")
    # The ALPHA ENGINE wins, not the sharpest corner. Q1 contributes
    # $12,000 x 1.28 = 15,360; Q3's 1.55 factor over 45 trades is both the
    # smaller contribution and below the 50-trade floor. Before 2026-08-21 the
    # rank was profit factor alone and this same fixture designated Q3 - which
    # is what sent Gate R to certify in a quadrant holding a handful of holdout
    # trades while the money was being made next door.
    profiles = {"A": profile(hvt=(1.28, 120), lvt=(1.55, 45)), "B": None}
    r = row("NQ", "15m", profiles, ml=False)
    check("the quadrant that CONTRIBUTES the alpha wins, not the one with the "
          "sharpest per-trade edge",
          r["optimal_regime"] == REGIMES[0] and r["regime_pf"] == 1.28,
          f"{r['optimal_regime']} {r['regime_pf']}")
    check("...tagged with its Q id as well as its name",
          r["optimal_quadrant"] == "Q1", str(r["optimal_quadrant"]))
    check("...and its own trade count, win rate and net P&L travel with it",
          (r["regime_trade_count"], r["regime_win_rate"], r["regime_net_pnl"])
          == (120, 55.0, 12000.0), str(r["regime_trade_count"]))
    check("...with the alpha score that chose it and the floor it cleared",
          (r["regime_score"], r["regime_sample_floor"]) == (15360.0, 50),
          f"{r['regime_score']} {r['regime_sample_floor']}")
    check("the runner-up is recorded as a SECONDARY, with why it could not "
          "be designated - a quadrant dropped on sample size and one that "
          "lost money are not the same instruction to a supervisor",
          [x["quadrant"] for x in r["secondary_regimes"]] == ["Q3"]
          and "sample floor" in r["secondary_regimes"][0]["reason"],
          str(r["secondary_regimes"]))
    check("...and the whole scored table travels, all four bars visible",
          set(r["regime_scores"]) == {REGIMES[0], REGIMES[2]}
          and r["regime_scores"][REGIMES[2]]["eligible"] is False,
          str(sorted(r["regime_scores"])))
    check("the other THREE quadrants are the kill switch, in regime order",
          r["kill_switch_regimes"] == [REGIMES[1], REGIMES[2], REGIMES[3]],
          str(r["kill_switch_regimes"]))
    check("a failing quadrant and one never traded are muted identically - "
          "'no evidence' must not read as 'permitted'",
          REGIMES[1] in r["kill_switch_regimes"]
          and REGIMES[1] not in profiles["A"]["regime_breakdown"])

    dropped = row("CL", "15m", {"A": profile(hvt=(1.90, 12)), "B": None},
                  ml=False)
    check("a DROPPED configuration names NO optimal regime and NO quadrant",
          dropped["optimal_regime"] is None
          and dropped["optimal_quadrant"] is None
          and dropped["regime_pf"] is None)
    check("...and derives an EMPTY kill switch, never all four - 'trade "
          "nowhere' is an instruction that has to come from a decision",
          dropped["kill_switch_regimes"] == [])
    check("kill_switch_regimes refuses to invent one from nothing",
          kill_switch_regimes(None) == []
          and kill_switch_regimes("not a regime") == [])

    tied = best_quadrant(profile(hvt=(1.40, 90), lvr=(1.40, 900)))
    check("the larger engine wins on score, not on the shared factor",
          tied["regime"] == REGIMES[3] and tied["quadrant"] == "Q4",
          str(tied))
    # A genuine score tie: same net P&L, same factor, different counts.
    even = {"regime_breakdown": {
                REGIMES[0]: {"trade_count": 100, "profit_factor": 1.20,
                             "win_rate": 55.0, "net_pnl": 6000.0},
                REGIMES[3]: {"trade_count": 300, "profit_factor": 1.20,
                             "win_rate": 55.0, "net_pnl": 6000.0}},
            "trades_profiled": 400}
    check("an exact tie on score breaks on the LARGER trade count - the "
          "better-evidenced claim, not whichever the quadrant order put first",
          best_quadrant(even)["quadrant"] == "Q4",
          str(best_quadrant(even)["quadrant"]))


# --------------------------------------------------------------------------
# 5. The handoff
# --------------------------------------------------------------------------
def stage1_rows() -> list[dict]:
    return [
        row("NQ", "15m", {"A": profile(hvt=(1.28, 120), lvr=(0.4, 300)),
                          "B": profile(lvr=(0.9, 80))}),
        row("GC", "30m", {"A": profile(hvr=(0.61, 400)),
                          "B": profile(lvt=(1.61, 60))}),
        row("CL", "15m", {"A": profile(hvt=(0.94, 500)),
                          "B": profile(hvt=(0.88, 260))}),
    ]


def test_handoff(tmp: Path) -> None:
    print("\n5. surviving_assets.json - the certified survivors and the drops")
    rows = stage1_rows()
    pairs = surviving_pairs_from(rows)
    drops = dropped_from(rows)
    results = screen_results_from(rows)

    check("two of three configurations were promoted",
          [p["symbol"] for p in pairs] == ["NQ", "GC"], str(pairs))
    required = {"symbol", "tf", "version", "status", "optimal_regime",
                "quadrant", "regime_pf", "regime_trade_count",
                "regime_win_rate", "regime_net_pnl", "kill_switch_regimes"}
    check("every survivor carries symbol, tf, version, the optimal regime and "
          "that quadrant's metrics",
          all(required <= set(p) for p in pairs),
          str(sorted(required - set(pairs[0]))))
    check("...the status is the word PROMOTED",
          all(p["status"] == "PROMOTED" for p in pairs))
    nq, gc = pairs
    check("...the VERSION that carried each pair is recorded, and the two "
          "differ here - a quadrant a classifier found is not one the rules "
          "did", (nq["version"], gc["version"]) == ("A", "B"),
          f"{nq['version']} {gc['version']}")
    check("...with the quadrant id beside the regime name",
          (nq["quadrant"], gc["quadrant"]) == ("Q1", "Q3"),
          f"{nq['quadrant']} {gc['quadrant']}")
    check("...and the pair the screen decided on: PF at a trade count",
          (nq["regime_pf"], nq["regime_trade_count"]) == (1.28, 120))

    check("the dropped configuration is marked DROPPED",
          [d["symbol"] for d in drops] == ["CL"]
          and drops[0]["status"] == "DROPPED", str(drops))
    check("...with the reason it fell short, so an operator knows whether it "
          "failed on edge or on sample size",
          "0.94" in drops[0]["reason"], drops[0]["reason"])
    check("...and an explicitly null optimal_regime beside an empty kill "
          "switch - a drop is never handed a live-trading instruction",
          drops[0]["optimal_regime"] is None
          and drops[0]["kill_switch_regimes"] == [])

    check("screen_results carries EVERY configuration evaluated, in one shape",
          len(results) == 3
          and [r["status"] for r in results]
              == ["PROMOTED", "PROMOTED", "DROPPED"], str(results))
    check("...with identical keys on promoted and dropped rows, so a "
          "leaderboard cannot print one under the other's heading",
          len({tuple(sorted(r)) for r in results}) == 1)

    # It has to survive the round trip through JSON: the handoff is a file.
    blob = {"timeframes": ["15m", "30m"], "start": CHARTER_IS_START,
            "end": CHARTER_IS_END, "ml_evaluated": True,
            "criterion": "optimal_regime_PF >= 1.00 AND "
                         "optimal_regime_trade_count >= 30",
            "min_profit_factor": 1.0, "min_trades": 30,
            "min_trade_fraction": 0.1,
            "surviving_pairs": pairs, "dropped": drops,
            "screen_results": results}
    dest = write_stage(tmp / SURVIVORS_FILE, 1, "demo", blob)
    back = json.loads(Path(dest).read_text())
    check("the handoff round-trips through JSON with the schema intact",
          back["surviving_pairs"][0]["quadrant"] == "Q1"
          and back["screen_results"][2]["status"] == "DROPPED")
    return back


def test_console_leaderboard() -> None:
    print("\n5b. The console leaderboard names the quadrant")
    out = survivors_leaderboard(stage1_rows())
    check("the QUAD column is present", "QUAD" in out)
    check("...alongside the full regime name and its profit factor",
          "Q1" in out and REGIMES[0] in out and "1.28" in out)
    body = [ln for ln in out.splitlines()
            if ln.strip().startswith(("NQ", "GC", "CL"))]
    check("only survivors are listed", len(body) == 2, str(body))
    check("sorted by the REGIME profit factor, descending",
          body[0].strip().startswith("GC"), str(body))


# --------------------------------------------------------------------------
# The Discord card
# --------------------------------------------------------------------------
def test_mode_resolution() -> None:
    print("\n6. --stage 1 and --mode baseline are one choice, two spellings")
    check("no flag at all is the promotion card this script has always posted",
          dr.resolve_mode(None, None) == "promotion")
    check("--stage 1 selects the baseline card",
          dr.resolve_mode(None, "1") == "baseline")
    check("--mode baseline selects the same card",
          dr.resolve_mode("baseline", None) == "baseline")
    check("--stage 5 and --mode promotion agree, and agreeing is fine",
          dr.resolve_mode("promotion", "5") == "promotion")
    ok, msg = raises(lambda: dr.resolve_mode("baseline", "5"), ValueError)
    check("...but --mode baseline --stage 5 RAISES rather than guessing - "
          "either guess posts the wrong card", ok, msg)


def test_stage1_card(blob: dict) -> None:
    print("\n7. The Stage 1 leaderboard card")
    embed = dr.build_stage1_embed("demo", blob, source="/mnt/backtest/x.json")
    text = embed["description"]

    check("the card is titled as a Stage 1 screen, with the strategy named "
          "in the body rather than in the title",
          "Stage 1" in embed["title"] and "demo" in text, embed["title"])
    fields = {f["name"]: f["value"] for f in embed["fields"]}
    check("it counts what was evaluated, promoted and dropped",
          (fields["Total Evaluated"], fields["Promoted → Stage 2"],
           fields["Dropped"]) == ("3", "2", "1"), str(fields))
    check("the per-pair leaderboard is NOT on the card - it was a fixed-width "
          "table inside a container that reflows, so every row wrapped on a "
          "narrow client and the alignment it existed for was the first "
          "casualty",
          "```" not in text and "NQ" not in text, text)
    check("...and the card names the two files that DO carry the detail, so "
          "a reader is pointed somewhere rather than left with five numbers",
          "stage1_survivors.csv" in text
          and "stage1_baseline_report.md" in text, text)
    check("...and the DROPPED one is off the table entirely - the card is "
          "read to answer 'what goes to Stage 2', and a drop is not that",
          "CL" not in text and "DROPPED" not in text, text)
    check("...while the counts still cover the whole screen",
          (fields["Total Evaluated"], fields["Dropped"]) == ("3", "1"),
          str(fields))
    check("...with the timeframes screened named as a field",
          "`15m`" in fields["Timeframes Screened"]
          and "`30m`" in fields["Timeframes Screened"])
    # The regime name, the quadrant PF, the trade count, both versions'
    # factors and the carrying version were columns of the removed table.
    # They are NOT lost - every one of them is a column of
    # stage1_survivors.csv, which the card names - but they are no longer
    # ASSERTED HERE, because the card no longer claims to carry them.
    check("no per-row metric is left stranded on the card - a number without "
          "the row it belonged to is worse than no number",
          not any(t in text for t in ("PF (A)", "PF (B)", "VA", "VB",
                                      "1.28", "1.61")), text)
    check("the in-sample window is on the card - a leaderboard whose bars "
          "nobody can name is a table of unlabelled figures",
          CHARTER_IS_START in text and CHARTER_IS_END in text)
    check("...as is the screening rule, as the two bars a row is checked "
          "against and not as the full designation mathematics",
          "Regime PF `>= 1.00`" in text
          and "Regime Trades `>= max(30, 10% placed)`" in text
          and blob["criterion"] not in text, text)
    check("...with both bars TRANSCRIBED from the thresholds the screen "
          "recorded, so a raised bar is not reported as the default",
          "max(50, 25% placed)" in dr.build_stage1_embed(
              "demo", {**blob, "min_trades": 50,
                       "min_trade_fraction": 0.25})["description"])
    check("...and a handoff recording neither says so, rather than printing "
          "a bar of zero nobody set",
          "Regime Trades `>= not recorded`" in dr.build_stage1_embed(
              "demo", {k: v for k, v in blob.items()
                       if k not in ("min_trades", "min_trade_fraction")}
              )["description"])
    check("Version B is reported as evaluated, not merely absent",
          fields["Version B"] == "evaluated", fields["Version B"])
    check("the handoff path is rendered as code, not as a dead link",
          fields["Handoff"].startswith("`"), fields["Handoff"])
    check("the whole embed is inside Discord's 6000-character limit",
          dr._embed_size(embed) <= dr.MAX_EMBED_TOTAL,
          str(dr._embed_size(embed)))
    check("...and _embed_size counts the DESCRIPTION, which is most of this "
          "card - omitting it would let a 400 through unnoticed",
          dr._embed_size(embed) > len(text))

    empty = dr.build_stage1_embed("demo", {**blob, "surviving_pairs": [],
                                           "screen_results":
                                           [r for r in blob["screen_results"]
                                            if r["status"] == "DROPPED"]})
    check("a screen where nothing survived is AMBER, not green - an empty "
          "screen is a result, and nothing on it is an approval",
          empty["color"] == dr.AMBER and embed["color"] == dr.EMERALD_GREEN)
    check("...and it still reports the counts, so an empty screen is legible "
          "as a screen that ran and promoted nothing rather than as a card "
          "which failed to draw",
          {f["name"]: f["value"] for f in empty["fields"]}["Promoted → Stage 2"]
          == "0", empty["description"])


def test_stage1_card_edges() -> None:
    print("\n8. What the card refuses to do")
    many = [{"symbol": f"S{i:02d}", "tf": "15m", "status": "PROMOTED",
             "version": "A", "optimal_regime": REGIMES[0], "quadrant": "Q1",
             "regime_pf": 1.0 + i / 100, "regime_trade_count": 100 + i}
            for i in range(108)]
    embed = dr.build_stage1_embed("wide", {"screen_results": many})
    check("108 configurations do not blow the description limit",
          len(embed["description"]) <= dr.MAX_EMBED_DESCRIPTION
          and dr._embed_size(embed) <= dr.MAX_EMBED_TOTAL,
          str(len(embed["description"])))
    check("...because the description no longer grows with the screen at "
          "all - there is no table to truncate, so there is no truncation to "
          "under-report",
          len(embed["description"]) < 512, str(len(embed["description"])))
    check("the totals describe the whole screen",
          {f["name"]: f["value"]
           for f in embed["fields"]}["Total Evaluated"] == "108")

    old = dr.build_stage1_embed("legacy", {
        "surviving_pairs": [{"symbol": "NQ", "tf": "15m",
                             "optimal_regime": REGIMES[0], "regime_pf": 1.28,
                             "kill_switch_regimes": list(REGIMES[1:])}],
        "dropped": [{"symbol": "CL", "timeframe": "30m", "reason": "..."}]})
    fields = {f["name"]: f["value"] for f in old["fields"]}
    check("a handoff written before screen_results existed is still "
          "reassembled from surviving_pairs + dropped, and COUNTED",
          fields["Total Evaluated"] == "2" and fields["Dropped"] == "1",
          str(fields))

    ml_off = dr.build_stage1_embed("x", {"screen_results": [],
                                         "ml_evaluated": False})
    unknown = dr.build_stage1_embed("x", {"screen_results": []})
    vb = lambda e: {f["name"]: f["value"] for f in e["fields"]}["Version B"]
    check("a --no-ml screen says NOT RUN, and a handoff that recorded nothing "
          "says 'not recorded' - three states, never two",
          (vb(ml_off), vb(unknown)) == ("NOT RUN", "not recorded"),
          f"{vb(ml_off)} / {vb(unknown)}")

    card = dr.build_stage1_embed("x", {"screen_results": [
        {"symbol": "NQ", "tf": "15m", "status": "DROPPED", "version": "A",
         "optimal_regime": REGIMES[0], "quadrant": "Q1", "regime_pf": 9.99,
         "regime_trade_count": 5_000}]})
    check("the card filters on the STATUS the stage recorded and never "
          "re-derives it - a notifier that re-applied the hurdle would table "
          "this 9.99-over-5,000-trades row Stage 1 dropped",
          "9.99" not in card["description"]
          and {f["name"]: f["value"]
               for f in card["fields"]}["Promoted → Stage 2"] == "0"
          and {f["name"]: f["value"]
               for f in card["fields"]}["Total Evaluated"] == "1",
          card["description"])


def test_cli(tmp: Path, blob: dict) -> None:
    print("\n9. The CLI, end to end (--dry-run: nothing is sent)")
    path = tmp / SURVIVORS_FILE

    def run(*argv, expect: int = 0) -> str:
        p = subprocess.run([sys.executable,
                            str(REPO / "backtest" / "discord_reporter.py"),
                            *argv], capture_output=True, text=True,
                           timeout=120)
        check(f"  exit {expect}: {' '.join(argv[:4])}",
              p.returncode == expect, f"rc={p.returncode} {p.stderr[-300:]}")
        return p.stdout + p.stderr

    out = run("--stage", "1", "--strat", "demo", "--survivors", str(path),
              "--dry-run")
    check("--stage 1 posts the Stage 1 card, and --dry-run sends nothing",
          "Stage 1" in out and "DRY RUN" in out and "demo" in out)
    payload = json.loads(out[out.index("{"): out.rindex("}") + 1])
    check("...as a single well-formed embed",
          len(payload["embeds"]) == 1 and payload["embeds"][0]["description"])

    check("--mode baseline is the same card",
          "Stage 1" in run("--mode", "baseline", "--strat", "demo",
                           "--survivors", str(path), "--dry-run"))

    out = run("--strat", "demo", "--symbol", "NQ", "--tf", "15m", "--pf",
              "1.42", "--dd", "-8.3", "--regime", REGIMES[0], "--dry-run")
    check("the promotion card is unchanged and still the default mode",
          "Incubation Promotion" in out and "1.42" in out, out[:200])
    out = run("--strat", "demo", "--dry-run", expect=1)
    check("...and it refuses to build a card with no --symbol / --tf rather "
          "than heading one '?' - that is a promotion on no instrument",
          "--symbol" in out and "--tf" in out, out[-200:])

    out = run("--stage", "1", "--strat", "demo", "--survivors",
              str(tmp / "nope.json"), "--dry-run", expect=1)
    check("a missing handoff is refused, naming the stage that writes it",
          "stage 1" in out.lower() and "does not exist" in out, out[-200:])

    # A clean payload: `write_stage` spreads the payload OVER its own
    # provenance keys, so handing it a blob that already carries
    # `strategy: "demo"` would write a file that names demo whatever the
    # argument says.
    other = write_stage(tmp / "other.json", 1, "another_strategy",
                        {k: v for k, v in blob.items()
                         if k not in ("stage", "stage_name", "strategy",
                                      "generated_utc")})
    out = run("--stage", "1", "--strat", "demo", "--survivors", str(other),
              "--dry-run", expect=1)
    check("a handoff belonging to ANOTHER strategy is refused - a card is "
          "exactly the artifact nobody cross-checks",
          "another_strategy" in out, out[-200:])

    wrong = write_stage(tmp / "stage3.json", 3, "demo", {})
    out = run("--stage", "1", "--strat", "demo", "--survivors", str(wrong),
              "--dry-run", expect=1)
    check("...as is a file written by a different STAGE",
          "stage 3" in out, out[-200:])

    out = run("--stage", "1", "--mode", "promotion", "--strat", "demo",
              "--survivors", str(path), "--dry-run", expect=1)
    check("contradictory --stage / --mode is refused, not guessed at",
          "disagree" in out, out[-200:])

    out = run("--stage", "1", "--strat", "demo", "--survivors", str(path),
              "--webhook", "https://discord.example/api/webhooks/1/SECRET-TOKEN",
              expect=1)
    check("a failed POST never echoes the webhook - it is a credential, and "
          "requests' own exception text embeds the whole URL",
          "SECRET-TOKEN" not in out and "discord.example" in out, out[-300:])


def test_survivors_csv(tmp: Path) -> None:
    print("\n10. The Stage 1 CSV leaderboard")
    import csv as _csv
    from backtest.baseline import SURVIVORS_CSV_COLUMNS, write_survivors_csv

    rows = [
        {"symbol": "NQ", "timeframe": "1h", "status": "PROMOTED",
         "regime_version": "A", "optimal_regime": REGIMES[0],
         "regime_pf": 1.13, "regime_trade_count": 801,
         "profit_factor_a": 1.06, "profit_factor_b": 1.10,
         "sharpe_a": 0.43, "sharpe_b": 0.42,
         "max_drawdown_pct_a": -12.5, "reason": None},
        {"symbol": "ES", "timeframe": "15m", "status": "DROPPED",
         "regime_version": None, "optimal_regime": None,
         "regime_pf": None, "regime_trade_count": None,
         "profit_factor_a": 0.83, "profit_factor_b": 0.91,
         "sharpe_a": -0.24, "sharpe_b": -0.16,
         "max_drawdown_pct_a": -31.0, "reason": "no quadrant cleared"},
    ]
    dest = write_survivors_csv(tmp / "stage1_survivors.csv", rows)
    text = dest.read_text()
    parsed = list(_csv.DictReader(text.splitlines()))

    check("the DROPPED configuration is in the file, not filtered out - a "
          "survivors-only sheet cannot answer what was tried",
          [r["status"] for r in parsed] == ["PROMOTED", "DROPPED"],
          str([r["status"] for r in parsed]))
    check("the columns are the fixed schema, in order - a sheet whose "
          "columns move between runs cannot be diffed",
          list(parsed[0]) == list(SURVIVORS_CSV_COLUMNS), str(list(parsed[0])))
    check("no index column is written",
          not text.startswith(",") and text.split(",")[0] == "symbol")
    check("a dropped row leaves the quadrant columns EMPTY rather than zero - "
          "0.0 reads as a measurement that came back bad, which is a "
          "different finding from one never taken",
          (parsed[1]["optimal_regime"], parsed[1]["optimal_regime_pf"],
           parsed[1]["optimal_regime_trades"],
           parsed[1]["version_selected"]) == ("", "", "", ""), str(parsed[1]))
    check("...while still carrying the metrics it DOES have, so a drop can be "
          "understood rather than merely counted",
          parsed[1]["baseline_pf_va"].startswith("0.83")
          and parsed[1]["reason"] == "no quadrant cleared", str(parsed[1]))
    check("the trade count is written as an integer - a gap in the column "
          "must not render every surviving row as '801.0'",
          parsed[0]["optimal_regime_trades"] == "801",
          parsed[0]["optimal_regime_trades"])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage1charter_") as td:
        tmp = Path(td)
        test_dual_simulation_defaults()
        test_quadrant_ids()
        test_hurdle()
        test_optimal_tagging()
        blob = test_handoff(tmp)
        test_console_leaderboard()
        test_mode_resolution()
        test_stage1_card(blob)
        test_stage1_card_edges()
        test_cli(tmp, blob)
        test_survivors_csv(tmp)

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
