#!/usr/bin/env python3
"""
The Stage 4 card: what `discord_reporter.py --stage 4` may say about a
full-lifecycle run, and what it must refuse to say.

Location:  ~/src/trading/tests/test_stage4_card.py

Reads no bars and runs no backtest - every case builds the artifacts Stage 4
writes (`dual_metrics_<SYMBOL>.json`, `regime_profile_<SYM>_<TF>.json`) as
small JSON fixtures and asserts what the card makes of them. That is the whole
surface: the reporter transcribes, and the one thing it derives is the friction
share.

What is pinned here, and why each one is a way the card could mislead:

  * A metric nobody measured renders `--`, never `0`. A zero CAGR beside a
    zero drawdown reads as a run that was measured and found flat.
  * FRIC is undefined when gross P&L is not positive - `verify_full.cost_drag`'s
    rule, and the reason `0%` is wrong there.
  * ALPHA comes from the UNSUFFIXED regime profile only. The `_version_a` file
    beside it is Stage 1's, profiled over the charter window alone.
  * A snapshot that cannot be read is a ROW, not a dropped file, and a card
    shorter than the run it announces reads as a shorter run.
  * Another strategy's snapshot is refused outright.
  * `--stage 4` and `--mode audit` together are refused rather than guessed at.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import backtest.discord_reporter as dr                            # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def raises(fn, exc=Exception) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:                                              # noqa: BLE001
        return True, f"{type(e).__name__}: {e}"
    return False, "no exception"


NAN = float("nan")


def _snapshot(run: Path, symbol: str, tf: str = "15m", strategy: str = "demo",
              cagr=11.5, pnl=125_000.0, gross=200_000.0, costs=75_000.0,
              trades=1_200, version_b: bool = False,
              start="2010-01-04 00:00:00", end="2026-01-01 00:00:00") -> Path:
    """One `dual_metrics_<SYMBOL>.json`, shaped like write_dual_reports'."""
    blob = {
        "meta": {"strategy": strategy, "symbol": symbol, "timeframe": tf,
                 "start": start, "end": end},
        "version_a": {"metrics": {
            "annualized_return_pct": cagr, "total_pnl": pnl,
            "gross_pnl": gross, "total_costs": costs, "trade_count": trades}},
        "version_b": ({"metrics": {"trade_count": 900}} if version_b else None),
    }
    path = run / f"dual_metrics_{symbol}.json"
    path.write_text(json.dumps(blob))
    return path


def _profile(directory: Path, symbol: str, tf: str, regime: str | None,
             quadrant: str | None, score: float | None,
             suffix: str = "") -> Path:
    """One regime profile, as `RegimeProfiler.generate_profile` writes it."""
    path = directory / f"regime_profile_{symbol}_{tf}{suffix}.json"
    path.write_text(json.dumps({
        "optimal_regime": regime or "None",
        "optimal_quadrant": quadrant,
        "optimal_score": score,
        "optimal_profit_factor": 1.42 if regime else None,
        "optimal_trade_count": 512 if regime else 0}))
    return path


def _run_dir(tmp: Path, name: str = "verify_20260824_101112") -> Path:
    run = tmp / "pipeline" / "demo" / name
    run.mkdir(parents=True, exist_ok=True)
    return run


# --------------------------------------------------------------------------

def test_ingestion(tmp: Path) -> list[dict]:
    print("\n1. every dual_metrics_<SYMBOL>.json in the directory is a row")
    run = _run_dir(tmp)
    _snapshot(run, "NQ")
    _snapshot(run, "ES", pnl=-4_200.0, gross=-1_000.0, costs=900.0,
              trades=87, version_b=True)
    _snapshot(run, "GC", cagr=NAN, pnl=2_450_000.0, gross=3_000_000.0,
              costs=500_000.0, trades=15_000)
    (run / "dual_metrics_CL.json").write_text("{ not json")

    rows = dr.stage4_rows(run, "demo")
    by = {r["symbol"]: r for r in rows}

    check("one row per snapshot, the unreadable one included",
          len(rows) == 4 and set(by) == {"NQ", "ES", "GC", "CL"},
          str(sorted(by)))
    check("the symbol and timeframe come from the snapshot's meta",
          by["NQ"]["tf"] == "15m")
    check("CAGR, net P&L and the trade count are transcribed",
          by["NQ"]["cagr_pct"] == 11.5 and by["NQ"]["net_pnl"] == 125_000.0
          and by["NQ"]["trades"] == 1_200)
    check("a corrupt snapshot carries an error and no metrics",
          by["CL"]["error"] and by["CL"].get("cagr_pct") is None,
          str(by["CL"]))
    check("Version B is recorded as present or absent, never as a zero",
          by["ES"]["version_b"] is True and by["NQ"]["version_b"] is False)
    check("rows are ordered by contract, not ranked on a metric",
          [r["symbol"] for r in dr.stage4_rows(run, "demo")]
          == ["CL", "ES", "GC", "NQ"])

    ok, msg = raises(lambda: dr.stage4_rows(tmp / "pipeline" / "demo", "demo"),
                     FileNotFoundError)
    check("a directory with no snapshots is REFUSED, not posted empty",
          ok, msg)
    ok, msg = raises(lambda: dr.stage4_rows(run / "nope", "demo"),
                     FileNotFoundError)
    check("a path that is not a directory is refused", ok, msg)
    return rows


def test_friction_share() -> None:
    print("\n2. FRIC is costs over GROSS profit, undefined when there is none")
    check("the ratio is 100 x costs / gross",
          dr.friction_share({"gross_pnl": 200_000.0,
                             "total_costs": 75_000.0}) == 37.5)
    check("gross P&L of zero is UNDEFINED, never 0%",
          dr.friction_share({"gross_pnl": 0.0, "total_costs": 900.0}) is None)
    check("a negative gross P&L is undefined too",
          dr.friction_share({"gross_pnl": -1_000.0,
                             "total_costs": 900.0}) is None)
    check("a missing cost figure is undefined, not free",
          dr.friction_share({"gross_pnl": 10.0}) is None)
    check("NaN is a missing measurement, not a number",
          dr.friction_share({"gross_pnl": NAN, "total_costs": 5.0}) is None)
    check("an undefined share renders as -- and never as 0.0%",
          dr._fmt_pct(None) == "--" and dr._fmt_pct(0.0) == "0.0%")
    check("NaN renders as -- everywhere on the card",
          dr._fmt_pct(NAN) == "--" and dr._fmt_money(NAN) == "--")
    check("money is compact so the table holds its width",
          (dr._fmt_money(123_456.0), dr._fmt_money(2_450_000.0),
           dr._fmt_money(-4_200.0)) == ("123.5k", "2.45M", "-4,200"),
          f"{dr._fmt_money(123_456.0)} {dr._fmt_money(2_450_000.0)}")


def test_top_regime_alpha(tmp: Path) -> None:
    print("\n3. ALPHA comes from Stage 4's OWN profile, never Stage 1's")
    run = _run_dir(tmp, "verify_regimes")
    pipeline = run.parent
    _profile(pipeline, "NQ", "15m", "High Volatility / Trending", "Q1",
             456_789.0)
    _profile(run, "ES", "15m", None, None, None)
    # Stage 1's, over the charter window alone. It must NOT be read.
    _profile(pipeline, "GC", "15m", "Low Volatility / Ranging", "Q4",
             9_000_000.0, suffix="_version_a")

    nq = dr.top_regime_alpha(run, "NQ", "15m")
    es = dr.top_regime_alpha(run, "ES", "15m")
    gc = dr.top_regime_alpha(run, "GC", "15m")

    check("the designated quadrant and its alpha score are transcribed",
          nq and nq["quadrant"] == "Q1" and nq["score"] == 456_789.0
          and nq["designated"] is True)
    check("the profile is found in the pipeline dir the profiler writes to",
          nq and nq["source"].endswith("regime_profile_NQ_15m.json"))
    check("a profile designating NO home regime says so, with no quadrant",
          es and es["designated"] is False and es["quadrant"] is None
          and es["score"] is None)
    check("a Stage 1 _version_a profile is NOT read as this run's",
          gc is None, str(gc))
    check("a contract with no profile at all reports none",
          dr.top_regime_alpha(run, "CL", "15m") is None)


def test_wrong_strategy(tmp: Path) -> None:
    print("\n4. another strategy's snapshot is refused")
    run = _run_dir(tmp, "verify_wrongstrat")
    _snapshot(run, "NQ", strategy="other_strategy")
    ok, msg = raises(lambda: dr.stage4_rows(run, "demo"), ValueError)
    check("a snapshot written for another strategy is REFUSED", ok, msg)

    # `approved_incubator/<strat>/strat.py` is module `strat` under directory
    # `<strat>`, so that one spelling has to be accepted or no promoted
    # strategy could ever post a card.
    promoted = _run_dir(tmp, "verify_promoted")
    _snapshot(promoted, "NQ", strategy="strat")
    rows = dr.stage4_rows(promoted, "demo")
    check("the promoted module spelling ('strat') is accepted",
          len(rows) == 1 and not rows[0]["error"])


def test_table(tmp: Path, rows: list[dict]) -> None:
    print("\n5. the table: no invented zeros, a legend built from the rows")
    text, hidden, legend = dr.format_stage4_table(rows)
    lines = text.splitlines()
    width = max(len(line) for line in lines)

    check("the table holds its design width",
          width <= dr.STAGE4_TABLE_WIDTH, f"{width} chars")
    check("every contract is a row, the unreadable one included",
          len(lines) == 2 + len(rows), f"{len(lines)} lines")
    check("an unreadable snapshot shows -- in every metric column, not 0",
          [c for c in lines[2].split() if c] [2:] == ["--"] * 6,
          lines[2])
    check("nothing was hidden at the default row cap", hidden == 0)
    check("the legend maps only the quadrants that appear",
          legend == {}, str(legend))

    run = _run_dir(tmp, "verify_legend")
    _snapshot(run, "NQ")
    _profile(run.parent, "NQ", "15m", "High Volatility / Trending", "Q1", 1e5)
    text2, _h, legend2 = dr.format_stage4_table(dr.stage4_rows(run, "demo"))
    check("a designated quadrant reaches the table and the legend",
          legend2 == {"Q1": "High Volatility / Trending"}
          and "Q1" in text2, str(legend2))

    many = [dict(r, symbol=f"S{i}") for i, r in enumerate(rows * 8)]
    _t, hidden_many, _l = dr.format_stage4_table(many, max_rows=5)
    check("rows past the cap are COUNTED, never dropped in silence",
          hidden_many == len(many) - 5, str(hidden_many))


def test_card(tmp: Path, rows: list[dict]) -> None:
    print("\n6. the card says what it is not, and counts what it left out")
    run = _run_dir(tmp)
    embed = dr.build_stage4_embed("demo", rows, source=run)
    text = embed["description"]

    check("the embed is inside Discord's limits",
          dr._embed_size(embed) <= dr.MAX_EMBED_TOTAL
          and len(text) <= dr.MAX_EMBED_DESCRIPTION,
          str(dr._embed_size(embed)))
    check("it states outright that this is NOT a certification",
          "Not a certification" in text and "holdout" in text)
    check("a field says the stage certifies nothing",
          any(f["name"] == "Certifies" and "nothing" in f["value"]
              for f in embed["fields"]))
    check("no gate verdict appears anywhere on the card",
          not any(tok in text for tok in ("PASS", "FAIL", "CERTIFIED")))
    check("the metrics are named as Version A's",
          dr.STAGE4_VERSION_NOTE in text)
    check("FRIC is labelled as a share of GROSS profit",
          "share of GROSS profit" in text)
    check("the colour is Stage 4's own, not the promotion green",
          embed["color"] == dr.GRAPHITE
          and embed["color"] != dr.EMERALD_GREEN)
    check("the artifacts directory is named on the card",
          any(str(run) in f["value"] for f in embed["fields"]))
    check("unreadable snapshots are counted in a field AND in the text",
          any(f["name"] == "Unreadable" and f["value"] == "1"
              for f in embed["fields"])
          and "could not be read" in text)
    check("contracts are counted whole, and the verified count separately",
          any(f["name"] == "Contracts" and f["value"] == "4"
              for f in embed["fields"])
          and any(f["name"] == "Verified" and f["value"] == "3"
                  for f in embed["fields"]))
    check("nothing is summed across contracts",
          not any(f["name"].lower().startswith("total net")
                  for f in embed["fields"]))

    # A missing profile and a designated-nothing profile are different findings
    # behind the same `--`, and the note has to keep them apart.
    run2 = _run_dir(tmp, "verify_notes")
    # ZS has no profile anywhere; ES's designates nothing. Two `--` cells in
    # the ALPHA column, two different findings.
    _snapshot(run2, "ZS")
    _snapshot(run2, "ES")
    _profile(run2, "ES", "15m", None, None, None)
    note = dr.stage4_regime_note(dr.stage4_rows(run2, "demo"))
    check("the note counts no-profile and no-home-regime separately",
          "1 contract(s) have no profile" in note
          and "1 designated no home quadrant" in note, note)

    nothing = [{"symbol": "NQ", "tf": None, "error": "boom", "source": "x"}]
    dead = dr.build_stage4_embed("demo", nothing, source=run)
    check("a run where nothing could be read is amber, not graphite",
          dead["color"] == dr.AMBER)

    print("\n7. the lifecycle window is stated, or said to vary")
    check("one shared window prints as one span",
          "2010-01-04" in dr.stage4_window(
              [r for r in rows if not r.get("error")]))
    mixed = [dict(rows[1]), dict(rows[1])]
    mixed[1] = dict(mixed[1], window={"start": "2015-01-01", "end": "b"})
    check("contracts with different spans are not collapsed into one",
          dr.stage4_window(mixed) == "varies by contract — see the tear sheets",
          dr.stage4_window(mixed))
    check("no window at all is 'not recorded', not a guess",
          dr.stage4_window([]) == "not recorded")


def test_mode_resolution() -> None:
    print("\n8. --stage 4 is --mode verify, and a disagreement is refused")
    check("--stage 4 resolves to verify",
          dr.resolve_mode(None, "4") == "verify")
    check("--mode verify alone resolves to verify",
          dr.resolve_mode("verify", None) == "verify")
    check("the two agreeing is fine",
          dr.resolve_mode("verify", "4") == "verify")
    ok, msg = raises(lambda: dr.resolve_mode("audit", "4"), ValueError)
    check("--mode audit --stage 4 is REFUSED, not guessed at", ok, msg)
    check("the other stages are untouched",
          (dr.resolve_mode(None, "1"), dr.resolve_mode(None, "3"),
           dr.resolve_mode(None, None)) == ("baseline", "audit", "promotion"))


def test_default_dir(tmp: Path) -> None:
    print("\n9. the default artifacts directory is the NEWEST verify run")
    # `pipeline_dir(strat, out_dir)` treats out_dir as the pipeline directory
    # ITSELF, so that is what is passed here - the same override every stage's
    # --out-dir takes.
    base = tmp / "newest" / "demo"
    for name in ("verify_20260101_000000", "verify_20260824_101112",
                 "verify_20260501_120000"):
        (base / name).mkdir(parents=True, exist_ok=True)
    picked = dr.default_verify_dir("demo", str(base))
    check("the newest STAMP wins, sorted by name rather than mtime",
          picked.name == "verify_20260824_101112", picked.name)
    ok, msg = raises(
        lambda: dr.default_verify_dir("nobody", str(tmp / "empty")),
        FileNotFoundError)
    check("no verify run at all is a refusal naming stage 4", ok, msg)


def test_cli(tmp: Path) -> None:
    print("\n10. the CLI posts nothing on --dry-run and refuses what it must")
    run = _run_dir(tmp, "verify_cli")
    _snapshot(run, "NQ")
    script = REPO / "backtest" / "discord_reporter.py"

    def call(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(script), "--strat", "demo", *extra],
            capture_output=True, text=True, timeout=120)

    out = call("--stage", "4", "--artifacts", str(run), "--dry-run")
    check("--stage 4 --dry-run exits 0 and sends nothing",
          out.returncode == 0 and "DRY RUN" in out.stdout, out.stderr[-200:])
    payload = json.loads(out.stdout.split("DRY RUN")[0])
    check("the payload is one embed titled Stage 4",
          len(payload["embeds"]) == 1
          and "Stage 4" in payload["embeds"][0]["title"])
    check("no webhook is needed for a dry run",
          "webhook" not in out.stderr.lower())

    bad = call("--stage", "4", "--artifacts", str(tmp), "--dry-run")
    check("a directory with no snapshots exits 1 and posts nothing",
          bad.returncode == 1 and "dual_metrics" in bad.stderr, bad.stderr[-200:])

    clash = call("--stage", "4", "--mode", "scan", "--dry-run")
    check("--stage 4 --mode scan exits 1 rather than posting either card",
          clash.returncode == 1 and "disagree" in clash.stderr,
          clash.stderr[-200:])

    helptext = call("--help").stdout
    check("--stage 4 and --artifacts are documented in --help",
          "--artifacts" in helptext and "--stage 4" in helptext)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage4card_") as td:
        tmp = Path(td)
        rows = test_ingestion(tmp)
        test_friction_share()
        test_top_regime_alpha(tmp)
        test_wrong_strategy(tmp)
        test_table(tmp, rows)
        test_card(tmp, rows)
        test_mode_resolution()
        test_default_dir(tmp)
        test_cli(tmp)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: " + "; ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
