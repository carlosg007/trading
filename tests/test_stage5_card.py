#!/usr/bin/env python3
"""
The Stage 5 card: what `discord_reporter.py --stage 5` resolves on its own,
and what it must refuse to resolve.

Location:  ~/src/trading/tests/test_stage5_card.py

Reads no bars and runs no backtest. Every case builds the files a promotion
leaves behind - `approved_incubator/<strat>/meta.json`, the `dual_metrics.json`
beside it and the Stage 3 `gate_audit_<SYMBOL>_<TF>.json` the first cites - as
small JSON fixtures, and asserts what the card makes of them.

What is pinned here, and why each one is a way the card could mislead:

  * The OUT-OF-SAMPLE profit factor is Gate R's or nothing. `dual_metrics.json`
    and meta.json's snapshot both carry a profit factor measured over a window
    that CONTAINS the holdout; either one under that heading is an in-sample
    number on a promotion announcement.
  * The contract and the timeframe resolve as a PAIR, from one file. Taking the
    symbol from a certification and the timeframe from the module's own
    declaration is how a card announces NQ at 5m for a run certified at 1h -
    with both halves individually true.
  * The command line always wins, and a `--symbol` naming a different contract
    from the certification is FLAGGED rather than silently accepted.
  * The WIN RATE comes from the same sample as the profit factor beside it -
    Gate R's quadrant on the holdout - and falls back to the blended holdout
    and then to the run snapshot, saying which it read. Its unit comes from the
    SOURCE: the profiler writes a percentage and `summarize_result` writes a
    fraction, and a magnitude test cannot tell 0.52 from 52.0 reliably.
  * PORTFOLIO MEMBERSHIP is read from the routing table and is never typed.
    Staged under `approved_incubator/` is explicitly NOT permission to trade;
    `active_strategies` is what grants that, and the two must not read as one
    green embed. A registry that cannot be opened reports NOT RESOLVED rather
    than the staging token.
  * A value no file carries stays NOT REPORTED and the card says where it
    looked. A zero drawdown for a run nobody measured is the one failure a
    status notifier can cause on its own.
  * Another strategy's meta.json, metrics snapshot or gate audit is refused.
  * A file named explicitly and missing RAISES; one this went looking for on
    its own is a note.
  * `--audit-file`/`--metrics`/`--incubator` on any other card are refused
    rather than parsed and ignored.
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


STRAT = "demo_strat_20260824"


def _audit(path: Path, *, strategy: str = STRAT, symbol: str = "NQ",
           tf: str = "1h", version: str = "A", pf: float = 1.22,
           trades: int = 387, holdout_dd: float = -28.96,
           regime: str = "High Volatility / Ranging",
           quadrant: str = "Q2", stage: int = 3,
           gate_r_win: float | None = 53.75,
           holdout_win: float | None = 0.5233) -> Path:
    """One Stage 3 `gate_audit_<SYMBOL>_<TF>.json`, shaped like audit_gates'."""
    blob = {
        "stage": stage,
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": tf,
        "target_regime": regime,
        "target_quadrant": quadrant,
        "versions": {version: {
            "metrics_in_sample": {"max_drawdown_pct": -49.55,
                                  "profit_factor": 1.06},
            # A FRACTION here, the way `summarize_result` writes it, against
            # Gate R's percentage below. The two units in one fixture are the
            # point: a resolver that sniffed the unit from the magnitude would
            # pass every case here and be wrong on a 52% win rate.
            "metrics_holdout": {"max_drawdown_pct": holdout_dd,
                                "profit_factor": 1.11,
                                **({} if holdout_win is None
                                   else {"win_rate": holdout_win})},
            "gate_audit": {"status": "PASS", "passed": True, "gates": {
                "gate1": {"status": "FAIL"},
                "gate_regime": {
                    "status": "PASS",
                    "target_regime": regime,
                    "quadrant": quadrant,
                    "measured": {"profit_factor": pf, "trade_count": trades,
                                 "net_pnl": 96_912.54,
                                 **({} if gate_r_win is None
                                    else {"win_rate": gate_r_win})},
                }}}}},
    }
    path.write_text(json.dumps(blob))
    return path


def _meta(home: Path, *, name: str = STRAT, version: str = "A",
          audit_file: Path | None = None, symbol: str = "NQ",
          declared_symbols: list[str] | None = None,
          declared_tf: str = "5m",
          snapshot_pf: float | None = 1.09,
          snapshot_dd: float | None = -46.39,
          snapshot_win: float | None = 0.5089) -> Path:
    """One `approved_incubator/<strat>/meta.json`, shaped like promote.py's."""
    home.mkdir(parents=True, exist_ok=True)
    metrics = None
    if snapshot_pf is not None or snapshot_dd is not None:
        metrics = {"profit_factor": snapshot_pf,
                   "max_drawdown_pct": snapshot_dd, "sharpe": 0.48,
                   **({} if snapshot_win is None
                      else {"win_rate": snapshot_win})}
    blob = {
        "name": name,
        "version": version,
        # The MODULE's declarations: every contract it targets, at the
        # timeframe it prefers. Deliberately not the certified pair.
        "symbols": (declared_symbols if declared_symbols is not None
                    else ["NQ", "ES", "CL", "GC"]),
        "timeframe": declared_tf,
        "metrics": metrics,
        "gate_audit_status": "PASS",
        "certification": ({"audit_file": str(audit_file),
                           "audit_symbol": symbol,
                           "status": "PASS"} if audit_file else {}),
    }
    path = home / dr.PROMOTED_META_FILE
    path.write_text(json.dumps(blob))
    return path


def _metrics(home: Path, *, strategy: str = STRAT, symbol: str = "NQ",
             tf: str = "1h", version: str = "A", pf: float = 1.09,
             dd: float = -46.39, win: float | None = 0.5089,
             report: str = "/mnt/x/report_NQ_version_a.html",
             start: str = "2010-06-07 00:00:00+00:00",
             end: str = "2025-12-31 21:00:00+00:00") -> Path:
    """The `dual_metrics.json` a promotion locked beside its strategy."""
    home.mkdir(parents=True, exist_ok=True)
    key = "version_b" if version.upper() == "B" else "version_a"
    blob = {
        "meta": {"strategy": strategy, "symbol": symbol, "timeframe": tf,
                 "start": start, "end": end},
        key: {"metrics": {"profit_factor": pf, "max_drawdown_pct": dd,
                          "sharpe": 0.48, "trade_count": 4279,
                          **({} if win is None else {"win_rate": win})}},
        "reports": {key: report},
    }
    path = home / dr.PROMOTED_METRICS_FILE
    path.write_text(json.dumps(blob))
    return path


def _portfolios(path: Path, allocations: dict[str, list[str]] | None = None,
                account_types: dict[str, str] | None = None) -> Path:
    """
    A `config/portfolios.json`, cut down to what the card reads.

    Written per case rather than pointing at the repository's own file: what
    `active_strategies` holds today is a live routing decision that changes,
    and a test that asserted the card's membership from it would start failing
    the first time somebody allocated a strategy.
    """
    allocations = allocations or {}
    account_types = account_types or {}
    portfolios = {}
    for pid in ("Incubator-Odd", "Incubator-Even", "Prop-Odd", "Prop-Even"):
        portfolios[pid] = {
            "portfolio_id": pid,
            "account_type": account_types.get(
                pid, "incubator_sim" if pid.startswith("Incubator")
                else "prop_eval"),
            "active_strategies": allocations.get(pid, []),
        }
    path.write_text(json.dumps({"version": "1.1.0", "portfolios": portfolios}))
    return path


# --------------------------------------------------------------------------

def test_full_resolution(tmp: Path) -> None:
    print("\nEverything resolved, from the files a promotion leaves behind")
    home = tmp / "full" / STRAT
    audit = _audit(tmp / "gate_audit_NQ_1h.json")
    _meta(home, audit_file=audit)
    _metrics(home)

    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "full")

    check("the certified contract, not the module's declared one",
          (res["symbol"], res["tf"]) == ("NQ", "1h"),
          f"{res['symbol']} {res['tf']}")
    check("the pair came from the gate audit",
          res["sources"]["symbol"] == audit.name == res["sources"]["tf"],
          str(res["sources"]))
    check("the profit factor is Gate R's, to 2dp", res["pf"] == "1.22", res["pf"])
    check("its basis names the window, the quadrant and the sample",
          all(t in dict((lbl, b) for lbl, b, _ in res["resolved"])["Out-of-Sample PF"]
              for t in ("holdout", "Q2", "387")),
          str(res["resolved"]))
    check("the win rate is Gate R's own quadrant, not rescaled",
          res["win"] == "53.75", res["win"])
    check("its basis names the same holdout quadrant the factor was scored in",
          all(t in dict((lbl, b) for lbl, b, _ in res["resolved"])["Win Rate"]
              for t in ("holdout", "Q2", "387")),
          str(res["resolved"]))
    check("the drawdown is the HOLDOUT's, as a magnitude",
          res["dd"] == "28.96", res["dd"])
    check("the regime carries the quadrant and its name",
          res["regime"] == "Q2 · High Volatility / Ranging", res["regime"])
    check("the tear sheet comes from the metrics snapshot",
          res["report"].endswith("report_NQ_version_a.html")
          and res["sources"]["report"] == dr.PROMOTED_METRICS_FILE,
          f"{res['report']} via {res['sources'].get('report')}")
    check("nothing is left NOT REPORTED", res["missing"] == [], str(res["missing"]))
    check("every auto-resolved value names the file it came from",
          all(where for _, _, where in res["resolved"]), str(res["resolved"]))


def test_pf_is_gate_r_only(tmp: Path) -> None:
    print("\nThe out-of-sample profit factor is Gate R's or nothing")
    home = tmp / "nopf" / STRAT
    # No certification on disk: both remaining files carry a profit factor,
    # and both measured it over a window that contains the holdout.
    _meta(home, audit_file=None)
    _metrics(home)

    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "nopf")

    check("the snapshot's profit factor is NOT promoted to the OOS field",
          res["pf"] == "", res["pf"])
    check("the card will print its NOT REPORTED token",
          dr.build_embed(STRAT, res["symbol"], res["tf"], res["pf"], res["dd"],
                         res["regime"], res["report"],
                         resolution=res)["fields"][1]["value"] == "NOT REPORTED")
    check("and the card says the number was declined, not absent",
          any("Gate R's or nothing" in n for n in res["notes"]),
          str(res["notes"]))
    check("the drawdown still resolves, labelled NOT the holdout",
          res["dd"] == "46.39"
          and "NOT the holdout" in dict((l, b) for l, b, _ in res["resolved"])["Max Drawdown"],
          f"{res['dd']} {res['resolved']}")

    # Nothing on disk carries one at all: then "not found" is the whole story
    # and the note would describe a decision nobody had to make.
    bare = tmp / "bare" / STRAT
    _meta(bare, audit_file=None, snapshot_pf=None, snapshot_dd=None)
    quiet = dr.resolve_promotion_fields(STRAT, incubator=tmp / "bare")
    check("no such note when no file carried a profit factor at all",
          not any("Gate R's or nothing" in n for n in quiet["notes"]),
          str(quiet["notes"]))


def test_pair_is_never_mixed(tmp: Path) -> None:
    print("\nThe contract and the timeframe come from ONE file")
    home = tmp / "pair" / STRAT
    # The certification says NQ at 1h; the module declares 5m. Only meta.json
    # survives - the audit it cites is gone.
    _meta(home, audit_file=tmp / "gone" / "gate_audit_NQ_1h.json",
          declared_tf="5m")

    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "pair")
    check("the timeframe comes off the certification, not the declaration",
          (res["symbol"], res["tf"]) == ("NQ", "1h"),
          f"{res['symbol']} {res['tf']}")
    check("a cited audit that is not on disk is a note, not a crash",
          any("not on disk" in n for n in res["notes"]), str(res["notes"]))

    # A module that names exactly one contract has nothing to pick between, so
    # its declarations are usable - and are labelled as declarations.
    single = tmp / "single" / STRAT
    _meta(single, audit_file=None, declared_symbols=["ZS"], declared_tf="30m")
    res2 = dr.resolve_promotion_fields(STRAT, incubator=tmp / "single")
    check("a single-symbol module resolves from its own declarations",
          (res2["symbol"], res2["tf"]) == ("ZS", "30m"),
          f"{res2['symbol']} {res2['tf']}")

    # Four declared contracts and no certification: nothing to pick between
    # them, so the pair stays unresolved rather than being guessed at.
    many = tmp / "many" / STRAT
    _meta(many, audit_file=None, declared_symbols=["NQ", "ES"], declared_tf="5m")
    res3 = dr.resolve_promotion_fields(STRAT, incubator=tmp / "many")
    check("a multi-symbol module resolves NO contract rather than the first",
          (res3["symbol"], res3["tf"]) == ("", ""),
          f"{res3['symbol']} {res3['tf']}")


def test_cli_wins(tmp: Path) -> None:
    print("\nThe command line outranks every file, and disagreement is flagged")
    home = tmp / "cli" / STRAT
    audit = _audit(tmp / "gate_audit_NQ_1h.json")
    _meta(home, audit_file=audit)
    _metrics(home)

    res = dr.resolve_promotion_fields(
        STRAT, symbol="ES", pf="2.50", regime="Q1 · High Volatility / Trending",
        report="https://example.invalid/r.html", incubator=tmp / "cli")

    check("--pf wins over Gate R", res["pf"] == "2.50", res["pf"])
    check("--regime wins over the certification target",
          res["regime"].startswith("Q1"), res["regime"])
    check("--report wins over the snapshot's tear sheet",
          res["report"].startswith("https://"), res["report"])
    check("--symbol wins, and the timeframe still comes off the audit",
          (res["symbol"], res["tf"]) == ("ES", "1h"),
          f"{res['symbol']} {res['tf']}")
    check("a --symbol that disagrees with the certification is FLAGGED",
          any("different contract" in n for n in res["notes"]), str(res["notes"]))
    check("what came from the command line is not listed as auto-resolved",
          not any(lbl in ("Out-of-Sample PF", "Certified Regime Firewall",
                          "Artifacts / Report")
                  for lbl, _, _ in res["resolved"]), str(res["resolved"]))
    check("a value typed on the command line records the flag as its source",
          res["sources"]["pf"] == "--pf", str(res["sources"]))


def test_missing_values(tmp: Path) -> None:
    print("\nA value no file carries stays NOT REPORTED, and says where it looked")
    home = tmp / "thin" / STRAT
    _meta(home, audit_file=None, snapshot_pf=None, snapshot_dd=None,
          declared_symbols=["ZS"], declared_tf="30m")

    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "thin")
    embed = dr.build_embed(STRAT, res["symbol"], res["tf"], res["pf"],
                           res["dd"], res["regime"], res["report"],
                           resolution=res, win=res["win"],
                           membership=res["membership"])
    values = {f["name"]: f["value"] for f in embed["fields"]}

    check("the profit factor is NOT REPORTED, never 0.00",
          values["Out-of-Sample PF"] == "NOT REPORTED")
    check("the drawdown is NOT REPORTED, never 0.00 %",
          values["Max Drawdown"] == "NOT REPORTED")
    check("the regime is NOT DECLARED", values["Certified Regime Firewall"] == "NOT DECLARED")
    check("the provenance field names what was inspected",
          dr.PROMOTED_META_FILE in values["Auto-resolved"], values["Auto-resolved"])
    check("and lists every field it could not fill",
          all(f"`{lbl}` · not found" in values["Auto-resolved"]
              for lbl in ("Out-of-Sample PF", "Max Drawdown")),
          values["Auto-resolved"])


def test_sentinel_and_version(tmp: Path) -> None:
    print("\nThe 999 sentinel, and the version that was actually promoted")
    audit = _audit(tmp / "gate_audit_GC_15m.json", symbol="GC", tf="15m",
                   pf=dr.REGIME_PF_SENTINEL, trades=1)
    home = tmp / "sentinel" / STRAT
    _meta(home, audit_file=audit, symbol="GC")
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "sentinel")
    check("a 999 profit factor renders as NOT MEASURED, never as 999.00",
          res["pf"] == "NOT MEASURED", res["pf"])
    check("and says why - the quadrant never had a losing trade",
          "no losing trade" in dict((l, b) for l, b, _ in res["resolved"])["Out-of-Sample PF"],
          str(res["resolved"]))

    # The promotion recorded Version B; the audit carries only A. Reading A's
    # Gate R numbers under a Version B promotion is the substitution this card
    # could not survive, so they are left off and the gap is stated.
    bhome = tmp / "versionb" / STRAT
    baudit = _audit(tmp / "gate_audit_ES_30m.json", symbol="ES", tf="30m",
                    version="A")
    _meta(bhome, audit_file=baudit, symbol="ES", version="B")
    _metrics(bhome, symbol="ES", tf="30m", version="B", dd=-12.5,
             report="/mnt/x/report_ES_version_b.html")
    resb = dr.resolve_promotion_fields(STRAT, incubator=tmp / "versionb")
    check("Version B's promotion reads Version B's blocks",
          resb["version"] == "B" and resb["report"].endswith("version_b.html"),
          f"{resb['version']} {resb['report']}")
    check("an audit carrying only Version A supplies no Gate R numbers",
          resb["pf"] == "" and resb["regime"] == "",
          f"{resb['pf']!r} {resb['regime']!r}")
    check("and the gap is stated rather than filled from the other version",
          any("not Version B" in n for n in resb["notes"]), str(resb["notes"]))
    check("the contract still resolves from that audit's own header",
          (resb["symbol"], resb["tf"]) == ("ES", "30m"),
          f"{resb['symbol']} {resb['tf']}")


def test_win_rate_sources(tmp: Path) -> None:
    print("\nThe win rate: which sample it came from, and in whose units")

    # 1. Gate R's own quadrant, already a percentage.
    home = tmp / "win_gater" / STRAT
    _meta(home, audit_file=_audit(tmp / "gate_audit_NQ_1h.json"))
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "win_gater")
    check("Gate R's percentage is transcribed, not multiplied by 100",
          res["win"] == "53.75", res["win"])

    # 2. An audit whose Gate R block never recorded one: the blended holdout,
    #    written as a fraction, and the basis has to say it is blended.
    blended = tmp / "win_blended" / STRAT
    _meta(blended, audit_file=_audit(tmp / "gate_audit_ES_15m.json",
                                     symbol="ES", tf="15m", gate_r_win=None))
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "win_blended")
    check("the blended holdout fraction is scaled to a percentage",
          res["win"] == "52.33", res["win"])
    check("and the card says it is blended, not Gate R's quadrant",
          "blended" in dict((l, b) for l, b, _ in res["resolved"])["Win Rate"],
          str(res["resolved"]))

    # 3. No certification at all: the run snapshot, which is NOT the holdout
    #    and must say so - the field claims no window, so unlike the profit
    #    factor it is reported rather than declined.
    snap = tmp / "win_snapshot" / STRAT
    _meta(snap, audit_file=None)
    _metrics(snap)
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "win_snapshot")
    check("the snapshot supplies a win rate where no certification is readable",
          res["win"] == "50.89", res["win"])
    check("and its basis says the window is NOT the holdout",
          "NOT the holdout" in
          dict((l, b) for l, b, _ in res["resolved"])["Win Rate"],
          str(res["resolved"]))
    check("while the profit factor beside it is still DECLINED",
          res["pf"] == "" and "Win Rate" not in res["missing"],
          f"{res['pf']!r} {res['missing']}")

    # 4. Nowhere at all.
    none = tmp / "win_none" / STRAT
    _meta(none, audit_file=None, snapshot_pf=None, snapshot_dd=None,
          snapshot_win=None)
    _metrics(none, win=None)
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "win_none")
    check("a win rate no file carries stays empty and is listed as missing",
          res["win"] == "" and "Win Rate" in res["missing"],
          f"{res['win']!r} {res['missing']}")

    # 5. The command line still wins.
    res = dr.resolve_promotion_fields(STRAT, win="47.5",
                                      incubator=tmp / "win_gater")
    check("--win outranks every file", res["win"] == "47.5", res["win"])


def test_portfolio_membership(tmp: Path) -> None:
    print("\nPortfolio membership: staged, allocated, or unreadable")
    home = tmp / "member" / STRAT
    _meta(home, audit_file=_audit(tmp / "gate_audit_NQ_1h.json"))

    empty = _portfolios(tmp / "portfolios_empty.json")
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "member",
                                      portfolio_config=empty)
    check("staged and named by no portfolio reads as incubator staging",
          res["membership"] == dr.INCUBATOR_STAGING, res["membership"])
    check("and the provenance names the routing table it checked",
          any(lbl == "Portfolio Membership" and where == empty.name
              for lbl, _, where in res["resolved"]), str(res["resolved"]))

    live = _portfolios(tmp / "portfolios_live.json", {"Prop-Odd": [STRAT]})
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "member",
                                      portfolio_config=live)
    check("an allocated strategy names its portfolio",
          res["membership"] == "Active Prop-Odd (Allocated)", res["membership"])

    both = _portfolios(tmp / "portfolios_both.json",
                       {"Incubator-Odd": [STRAT], "Prop-Odd": [STRAT]})
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "member",
                                      portfolio_config=both)
    check("one strategy on both tracks names both - that is what they are for",
          res["membership"] == "Active Incubator-Odd + Prop-Odd (Allocated)",
          res["membership"])
    check("and it is not flagged, because two TRACKS is a normal config",
          not any("refuses that config" in n for n in res["notes"]),
          str(res["notes"]))

    twice = _portfolios(tmp / "portfolios_twice.json",
                        {"Prop-Odd": [STRAT], "Prop-Even": [STRAT]})
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "member",
                                      portfolio_config=twice)
    check("two portfolios on ONE track is flagged, the way the loader refuses it",
          any("refuses that config" in n for n in res["notes"]),
          str(res["notes"]))

    cased = _portfolios(tmp / "portfolios_case.json",
                        {"Prop-Even": [STRAT.upper()]})
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "member",
                                      portfolio_config=cased)
    check("an id differing only in case is the same strategy, not staging",
          res["membership"] == "Active Prop-Even (Allocated)", res["membership"])

    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "member",
                                      portfolio_config=tmp / "no_such.json")
    check("an unreadable registry is NOT RESOLVED, never the staging token",
          res["membership"] == dr.MEMBERSHIP_UNRESOLVED, res["membership"])
    check("and says why, rather than claiming an allocation nobody checked",
          any("not readable" in n for n in res["notes"]), str(res["notes"]))

    broken = tmp / "portfolios_broken.json"
    broken.write_text("{not json")
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "member",
                                      portfolio_config=broken)
    check("so is a corrupt one - and the card is still built",
          res["membership"] == dr.MEMBERSHIP_UNRESOLVED
          and res["symbol"] == "NQ", str(res["membership"]))


def test_refusals(tmp: Path) -> None:
    print("\nWhat it refuses to read")
    home = tmp / "refuse" / STRAT
    audit = _audit(tmp / "gate_audit_NQ_1h.json")
    _meta(home, audit_file=audit)

    other = _audit(tmp / "gate_audit_ZZ_5m.json", strategy="somebody_else",
                   symbol="ZZ", tf="5m")
    ok, detail = raises(lambda: dr.load_promotion_audit(other, STRAT), ValueError)
    check("another strategy's gate audit is refused", ok, detail)

    stage2 = _audit(tmp / "gate_audit_XX_5m.json", symbol="XX", tf="5m", stage=2)
    ok, detail = raises(lambda: dr.load_promotion_audit(stage2, STRAT), ValueError)
    check("a file written by another stage is refused", ok, detail)

    wrong_home = tmp / "wrongname" / STRAT
    _meta(wrong_home, name="a_different_strategy", audit_file=None)
    ok, detail = raises(
        lambda: dr.resolve_promotion_fields(STRAT, incubator=tmp / "wrongname"),
        ValueError)
    check("another strategy's meta.json is refused", ok, detail)

    metrics_home = tmp / "wrongmetrics" / STRAT
    _meta(metrics_home, audit_file=None)
    _metrics(metrics_home, strategy="a_different_strategy")
    ok, detail = raises(
        lambda: dr.resolve_promotion_fields(STRAT, incubator=tmp / "wrongmetrics"),
        ValueError)
    check("another strategy's metrics snapshot is refused", ok, detail)

    ok, detail = raises(
        lambda: dr.resolve_promotion_fields(STRAT, audit_file=tmp / "nope.json",
                                            incubator=tmp / "refuse"),
        FileNotFoundError)
    check("an --audit-file that is not there RAISES rather than defaulting",
          ok, detail)
    ok, detail = raises(
        lambda: dr.resolve_promotion_fields(STRAT, metrics_file=tmp / "nope.json",
                                            incubator=tmp / "refuse"),
        FileNotFoundError)
    check("so does a --metrics that is not there", ok, detail)

    bad = tmp / "broken.json"
    bad.write_text("{not json")
    ok, detail = raises(lambda: dr.load_promotion_audit(bad, STRAT), ValueError)
    check("a corrupt file is a refusal, not a card with holes in it", ok, detail)

    # No promoted directory at all: nothing to read, nothing to raise about.
    # The membership is still resolved - and says NOT STAGED, which is the one
    # honest thing to say about a strategy with no promotion and no allocation.
    registry = _portfolios(tmp / "refuse_portfolios.json")
    empty = dr.resolve_promotion_fields(STRAT, incubator=tmp / "does_not_exist",
                                        portfolio_config=registry)
    check("a strategy that was never promoted resolves no metric and raises "
          "nothing",
          empty["symbol"] == "" and empty["pf"] == "" and empty["win"] == "",
          str(empty))
    check("and is NOT STAGED rather than staged-and-waiting",
          empty["membership"] == dr.NOT_STAGED
          and [lbl for lbl, _, _ in empty["resolved"]] == ["Portfolio Membership"],
          str(empty["resolved"]))


def test_embed_shape(tmp: Path) -> None:
    print("\nThe card itself")
    home = tmp / "card" / STRAT
    audit = _audit(tmp / "gate_audit_NQ_1h.json")
    _meta(home, audit_file=audit)
    _metrics(home)
    res = dr.resolve_promotion_fields(STRAT, incubator=tmp / "card")

    embed = dr.build_embed(STRAT, res["symbol"], res["tf"], res["pf"],
                           res["dd"], res["regime"], res["report"],
                           resolution=res, win=res["win"],
                           membership=res["membership"])
    names = [f["name"] for f in embed["fields"]]
    check("the seven fields are in order, win rate on the metrics row and "
          "membership above the firewall",
          names[:7] == ["Asset / Timeframe", "Out-of-Sample PF", "Win Rate",
                        "Max Drawdown", "Portfolio Membership",
                        "Certified Regime Firewall", "Artifacts / Report"],
          str(names))
    inline = {f["name"]: f["inline"] for f in embed["fields"]}
    check("the four metrics-row fields are inline and the rest are not",
          all(inline[n] for n in ("Asset / Timeframe", "Out-of-Sample PF",
                                  "Win Rate", "Max Drawdown"))
          and not any(inline[n] for n in ("Portfolio Membership",
                                          "Certified Regime Firewall",
                                          "Artifacts / Report")),
          str(inline))
    values = {f["name"]: f["value"] for f in embed["fields"]}
    check("the win rate is printed as a percentage",
          values["Win Rate"] == "53.75 %", values["Win Rate"])
    check("the provenance field is appended last", names[-1] == "Auto-resolved")
    check("the embed is well inside Discord's limits",
          dr._embed_size(embed) <= dr.MAX_EMBED_TOTAL
          and all(len(f["value"]) <= dr.MAX_FIELD_VALUE for f in embed["fields"]),
          str(dr._embed_size(embed)))

    typed = dr.build_embed(STRAT, "NQ", "15m", "1.42", "8.30", "Q1", "/x.html")
    check("a card built with no resolution carries NO provenance field",
          [f["name"] for f in typed["fields"]] == names[:7], str(typed))
    check("and is byte-identical to what it was before auto-resolution existed",
          typed == dr.build_embed(STRAT, "NQ", "15m", "1.42", "8.30", "Q1",
                                  "/x.html", resolution=None))
    typed_values = {f["name"]: f["value"] for f in typed["fields"]}
    check("a membership nobody resolved is NOT RESOLVED, never the staging "
          "token - that token is a claim about a file this never opened",
          typed_values["Portfolio Membership"] == dr.MEMBERSHIP_UNRESOLVED,
          typed_values["Portfolio Membership"])
    check("and a win rate nobody measured is NOT REPORTED, never 0.00 %",
          typed_values["Win Rate"] == "NOT REPORTED", typed_values["Win Rate"])


def test_cli(tmp: Path) -> None:
    print("\nCLI")
    home = tmp / "cli2" / STRAT
    audit = _audit(tmp / "gate_audit_NQ_1h.json")
    _meta(home, audit_file=audit)
    _metrics(home)

    def call(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO / "backtest" / "discord_reporter.py"),
             *argv],
            capture_output=True, text=True)

    run = call("--stage", "5", "--strat", STRAT, "--incubator",
               str(tmp / "cli2"), "--dry-run")
    check("--stage 5 with nothing else builds the card and sends nothing",
          run.returncode == 0 and "DRY RUN" in run.stdout, run.stderr[-300:])
    payload = json.loads(run.stdout[: run.stdout.rindex("}") + 1])
    fields = {f["name"]: f["value"] for f in payload["embeds"][0]["fields"]}
    check("the posted card carries the certified pair and Gate R's factor",
          fields["Asset / Timeframe"] == "**NQ** · `1h`"
          and fields["Out-of-Sample PF"] == "1.22", str(fields))

    gone = call("--stage", "5", "--strat", "never_promoted", "--incubator",
                str(tmp / "cli2"), "--dry-run")
    check("an unresolvable contract exits 1 and names where it looked",
          gone.returncode == 1 and "--symbol" in gone.stderr, gone.stderr[-300:])

    manual = call("--stage", "5", "--strat", "never_promoted", "--incubator",
                  str(tmp / "cli2"), "--symbol", "CL", "--tf", "15m",
                  "--pf", "1.30", "--dry-run")
    check("the same strategy posts fine when the values are typed",
          manual.returncode == 0 and "1.30" in manual.stdout,
          manual.stderr[-300:])

    stray = call("--stage", "3", "--strat", STRAT, "--metrics",
                 str(home / dr.PROMOTED_METRICS_FILE), "--dry-run")
    check("--metrics on another card is refused, not silently ignored",
          stray.returncode == 1 and "--metrics" in stray.stderr,
          stray.stderr[-300:])

    registry = _portfolios(tmp / "cli_portfolios.json",
                           {"Prop-Odd": [STRAT]})
    allocated = call("--stage", "5", "--strat", STRAT, "--incubator",
                     str(tmp / "cli2"), "--portfolios", str(registry),
                     "--dry-run")
    payload = json.loads(allocated.stdout[: allocated.stdout.rindex("}") + 1])
    fields = {f["name"]: f["value"] for f in payload["embeds"][0]["fields"]}
    check("an allocated strategy posts as Active <portfolio> (Allocated)",
          fields["Portfolio Membership"] == "Active Prop-Odd (Allocated)",
          fields["Portfolio Membership"])
    check("and the win rate rides the metrics row",
          fields["Win Rate"] == "53.75 %", fields["Win Rate"])

    typed_win = call("--stage", "5", "--strat", STRAT, "--incubator",
                     str(tmp / "cli2"), "--portfolios", str(registry),
                     "--win", "41.0", "--dry-run")
    check("--win overrides the resolved one",
          '"41.00 %"' in typed_win.stdout, typed_win.stdout[-400:])

    stray_portfolios = call("--stage", "1", "--strat", STRAT, "--portfolios",
                            str(registry), "--dry-run")
    check("--portfolios on another card is refused, not silently ignored",
          stray_portfolios.returncode == 1
          and "--portfolios" in stray_portfolios.stderr,
          stray_portfolios.stderr[-300:])

    helptext = call("--help").stdout
    check("the new flags are documented in --help",
          all(flag in helptext for flag in ("--audit-file", "--metrics",
                                            "--incubator", "--win",
                                            "--portfolios")))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage5card_") as td:
        tmp = Path(td)
        test_full_resolution(tmp)
        test_pf_is_gate_r_only(tmp)
        test_pair_is_never_mixed(tmp)
        test_cli_wins(tmp)
        test_missing_values(tmp)
        test_win_rate_sources(tmp)
        test_portfolio_membership(tmp)
        test_sentinel_and_version(tmp)
        test_refusals(tmp)
        test_embed_shape(tmp)
        test_cli(tmp)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: " + "; ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
