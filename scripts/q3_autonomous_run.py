#!/usr/bin/env python3
"""
q3_autonomous_run.py - the unattended Q3 discovery -> certification ->
promotion cycle, as ONE command the operator launches from their own shell.

Location:  ~/src/trading/scripts/q3_autonomous_run.py

WHY THIS IS A SCRIPT AND NOT SOMETHING CLAUDE RAN
=================================================
CLAUDE.md's tool boundary reserves every pipeline stage for the operator's
shell, and says why: the four-choice promotion menu exists so a human sees the
evidence before anything is promoted, and `--bg` does not satisfy it. A brief
that says "the operator is stepping away, do not halt for prompts, auto-promote
what passes" is precisely the situation the rule names.

So the automation is BUILT rather than PERFORMED. Everything the brief asked
for is here - the chaining, the error isolation, the routing checks, the
report - and the operator starts it. That keeps the decision to run unattended
where it belongs, and it makes the run reproducible, inspectable and
re-launchable rather than a thing that happened once inside a chat session.

WHAT IT DOES
============
    Stage 1  baseline.py    per strategy, all symbols, per timeframe
    Stage 2  scan.py        per (strategy, symbol, tf) that survived into Q3
    Stage 3  audit_gates.py per swept pair, one timeframe per invocation
    Stage 5  promote.py     per Gate R PASS, ONLY with --promote
    Stage 4  verify_full.py per promoted configuration
    report   Q3_AUTONOMOUS_RUN_SUMMARY.md

Every stage is a subprocess. A stage that fails is RECORDED and the run
continues with the configurations that did not depend on it - one bad contract
must not end a campaign - and the exit code reports whether anything failed.

THREE THINGS THE BRIEF ASKED FOR THAT THE PIPELINE CANNOT DO AS WRITTEN, all
handled here rather than discovered at 3am:

1. **`--tf` is single-valued on Stages 3 and 4.** A gate audit certifies ONE
   (parameters, timeframe) pair. `--tf 30m,1h` there is not two audits, it is
   an error. This loops instead.

2. **`verify_full.py` has no `--params`.** It takes `--param key=value` and
   `--defaults`; the locked parameters come from Stage 2's
   `best_params_<SYMBOL>_<TF>.json` the same way Stage 3 finds them. Passing a
   path would fail argument parsing.

3. **ES AND GC CANNOT BE PROMOTED AS Q3 SPECIALISTS TODAY, and the brief's
   `--portfolio incubator-odd` would register them somewhere they can never
   trade.** `Incubator-Odd` holds MNQ/6E/6J and permits Q1-Q4;
   `Incubator-Even` holds MES/MGC and permits **Q1 and Q2 only**. So ES and GC
   route to Even, where a Q3 strategy is stood down by the portfolio's own
   regime scope, and forcing them onto Odd registers a strategy whose certified
   symbol is outside that basket - `promote.py` warns that such a strategy "is
   refused on every asset it is routed to and will never place an order".
   `route_for` resolves the portfolio from the basket and REFUSES rather than
   registering something inert. Widening Incubator-Even's quadrants is a
   config decision for a human.

Reads
-----
    The lake, through the stages. Nothing here reads bars itself.

Writes
------
    Everything the stages write, plus `Q3_AUTONOMOUS_RUN_SUMMARY.md` and
    `q3_autonomous_run.log` under the pipeline artifacts root. **Nothing is
    promoted and no config is touched without `--promote`.**

Usage
-----
    # See the plan and touch nothing:
    python3 scripts/q3_autonomous_run.py --dry-run

    # The full unattended cycle, promotion included:
    python3 scripts/q3_autonomous_run.py --promote 2>&1 | tee /tmp/q3run.log

    # Certify only, decide promotions by hand afterwards:
    python3 scripts/q3_autonomous_run.py
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
import subprocess                                                  # noqa: E402
import time                                                        # noqa: E402
from datetime import datetime, timezone                            # noqa: E402
from typing import Any                                             # noqa: E402

PY = str(PROJECT_ROOT / ".venv" / "bin" / "python3")

#: The trend archetypes the brief names.
STRATEGIES = (
    "keltner_trend_drift_20260901",
    "dual_ema_slope_scalp_20260831",
    "ema_crossover_20260821",
    "sma_momentum_crossover_20260818",
    "t3_braid_scalp_20260823",
)
SYMBOLS = ("6E", "6J", "ES", "NQ", "GC")
TIMEFRAMES = ("30m", "1h")

#: The charter windows. Spelled here only to pass them explicitly - the stages
#: default to the same values from `backtest.pipeline`, and passing them makes
#: the log say which window ran rather than leaving it to a default.
IS_START, IS_END = "2013-01-01", "2022-12-31"
HOLDOUT_START = "2023-01-01"
LIFECYCLE_START, LIFECYCLE_END = "2010-01-01", "2026-01-01"

#: The quadrant this campaign is about, spelled as `backtest.profiler.REGIMES`
#: spells it. Not restated as a literal anywhere below.
Q3_LABEL = "Low Volatility / Trending"
Q3_CODE = "Q3"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def artifacts_root() -> Path:
    from backtest.pipeline import artifacts_root as _root          # noqa: PLC0415
    return Path(_root()) / "pipeline"


# --------------------------------------------------------------------------
# Running a stage
# --------------------------------------------------------------------------
class Runner:
    """
    One subprocess per stage, with the whole transcript kept.

    A stage that fails is RECORDED and the campaign continues. That is the same
    rule `backtest/run.py` applies per contract and for the same reason: one
    bad symbol must not throw away the fourteen that worked. What must NOT be
    swallowed is a memory halt - `backtest/memory_guard.py` exits 75 precisely
    so a caller can tell "this box is out of RAM" from "this configuration is
    bad", and continuing after one allocates just as much on a machine that is
    no emptier.
    """

    MEMORY_HALT_RC = 75

    def __init__(self, log_path: Path, dry_run: bool = False):
        self.log_path = log_path
        self.dry_run = dry_run
        self.calls: list[dict[str, Any]] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, text: str) -> None:
        print(text, flush=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    def run(self, label: str, argv: list[str],
            timeout_s: float = 6 * 3600) -> dict[str, Any]:
        cmd = [PY] + argv
        printable = " ".join(cmd)
        self._log(f"\n[{_utcnow()}] {label}\n    {printable}")
        if self.dry_run:
            self.calls.append({"label": label, "cmd": printable,
                               "rc": None, "skipped": "dry run"})
            return {"ok": True, "rc": None, "dry_run": True}

        started = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                  capture_output=True, text=True,
                                  timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            rec = {"ok": False, "rc": None, "error": f"timed out after "
                                                     f"{timeout_s:.0f}s"}
            self._log(f"    TIMEOUT after {timeout_s:.0f}s")
            self.calls.append({"label": label, "cmd": printable, **rec})
            return rec

        elapsed = time.time() - started
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(proc.stdout or "")
            fh.write(proc.stderr or "")
        self._log(f"    rc={proc.returncode}  {elapsed:.0f}s")

        if proc.returncode == self.MEMORY_HALT_RC:
            # NOT swallowed. See the class docstring.
            self._log("    MEMORY HALT (exit 75) — stopping the campaign "
                      "rather than allocating as much again on a box that is "
                      "no emptier.")
            self.calls.append({"label": label, "cmd": printable,
                               "rc": 75, "ok": False, "halt": True})
            raise SystemExit(75)

        rec = {"ok": proc.returncode == 0, "rc": proc.returncode,
               "elapsed_s": round(elapsed, 1), "tail": tail,
               "stderr": (proc.stderr or "")[-2000:]}
        self.calls.append({"label": label, "cmd": printable,
                           "rc": proc.returncode, "ok": rec["ok"]})
        return rec


# --------------------------------------------------------------------------
# Reading the handoffs
# --------------------------------------------------------------------------
def q3_survivors(strategy: str) -> list[dict[str, Any]]:
    """
    Stage 1's surviving pairs whose designated quadrant is Q3.

    Read from the handoff rather than from the console, because the handoff is
    what Stage 2 and Stage 3 read and a second parse of the printed table would
    be free to disagree with it.
    """
    path = artifacts_root() / strategy / "surviving_assets.json"
    if not path.is_file():
        return []
    blob = json.loads(path.read_text())
    out = []
    for row in blob.get("surviving_pairs") or []:
        regime = str(row.get("optimal_regime") or "")
        quadrant = str(row.get("quadrant") or "")
        if regime == Q3_LABEL or quadrant == Q3_CODE:
            out.append(row)
    return out


def gate_r_verdict(audit_path: Path) -> dict[str, Any]:
    """
    Gate R's status, profit factor and trade count out of one audit file.

    The per-pair `gate_audit_<SYMBOL>_<TF>.json` is the AUTHORITATIVE verdict a
    promotion rests on; the campaign summary is an index over them. Read per
    version, because A and B certify separately.
    """
    if not audit_path.is_file():
        return {}
    blob = json.loads(audit_path.read_text())
    out: dict[str, Any] = {}
    for version, payload in (blob.get("versions") or {}).items():
        gate = ((payload.get("gate_audit") or {}).get("gates") or {}
                ).get("gate_regime") or {}
        pf = trades = None
        for check in gate.get("checks") or []:
            label = str(check.get("label", "")).lower()
            if "profit factor" in label:
                pf = check.get("value")
            elif "trade" in label:
                trades = check.get("value")
        out[str(version).upper()] = {
            "status": gate.get("status"), "pf": pf, "trades": trades,
            "target_quadrant": blob.get("target_quadrant"),
        }
    return out


def route_for(symbol: str) -> tuple[str | None, str]:
    """
    The incubator portfolio a Q3 certification on `symbol` can actually trade
    in, or `(None, why not)`.

    TWO CONDITIONS, and both are silent when wrong. The portfolio's BASKET has
    to carry the contract (or its micro) - `promote.py` will register a
    strategy certified outside the basket and warn that it "is refused on every
    asset it is routed to and will never place an order" - and the portfolio's
    REGIME SCOPE has to permit Q3, or the live loop stands it down in the one
    environment it was certified for.

    Resolved from `config/portfolios.json` rather than hardcoded: the baskets
    and the quadrant scopes are edited there, and a table here would be a
    second answer free to go stale.
    """
    from portfolio.config_loader import load_portfolio_config      # noqa: PLC0415
    from realtime.contract_alias import micros_of, normalize       # noqa: PLC0415

    sym = normalize(symbol)
    covers = {sym, *micros_of(sym)}
    config = load_portfolio_config(str(PROJECT_ROOT / "config"
                                       / "portfolios.json"))
    problems = []
    for pid, portfolio in config["portfolios"].items():
        if "Incubator" not in pid:
            continue                      # the prop track is not promoted into
        assets = set(portfolio.get("basket", {}).get("assets") or [])
        quadrants = set(portfolio.get("derived", {}).get(
            "canonical_quadrants") or [])
        if not (assets & covers):
            continue
        if Q3_CODE not in quadrants:
            problems.append(
                f"{pid} carries {sorted(assets & covers)} but permits only "
                f"{sorted(quadrants)} — a Q3 strategy registered there is "
                f"stood down by the portfolio's own regime scope")
            continue
        return pid.lower(), ""
    if problems:
        return None, "; ".join(problems)
    return None, (f"no incubator portfolio's basket carries {sym} or a micro "
                  f"of it, so a strategy certified on it would be refused on "
                  f"every asset it was routed to")


# --------------------------------------------------------------------------
# The campaign
# --------------------------------------------------------------------------
def _row(strategy: str, symbol: str = "-", tf: str = "-",
         version: str = "-", **over) -> dict[str, Any]:
    """
    One leaderboard row, with EVERY column present.

    Uniform keys because the report renders one table over all of them: a skip
    row missing `certified` reads as a blank cell in one renderer and raises in
    the next, and "this pair was never evaluated" has to be as legible as a
    FAIL rather than an absence.
    """
    row = {"strategy": strategy, "symbol": symbol, "tf": tf,
           "version": version, "gate_r": None, "pf": None, "trades": None,
           "certified": False, "promoted": False, "note": ""}
    row.update(over)
    return row


def campaign(runner: Runner, strategies, symbols, timeframes,
             score_mode: str, promote: bool) -> list[dict[str, Any]]:
    """Stages 1 -> 2 -> 3 -> 5 -> 4, per strategy, collecting one row per
    (strategy, symbol, tf, version)."""
    rows: list[dict[str, Any]] = []
    root = artifacts_root()

    for strategy in strategies:
        source = PROJECT_ROOT / "strategies" / "experimental" / f"{strategy}.py"
        if not source.is_file():
            runner._log(f"\n!! {strategy}: no module at {source} — skipped")
            rows.append(_row(strategy, note="module not found"))
            continue
        out_dir = root / strategy

        # ---- Stage 1, one invocation per timeframe -----------------------
        # Stage 1 accepts a comma-separated --tf, but running them separately
        # keeps a failure at 1h from throwing away the 30m screen.
        for tf in timeframes:
            runner.run(
                f"STAGE 1 {strategy} {tf}",
                ["backtest/baseline.py", "--strat", strategy,
                 "--symbols", ",".join(symbols), "--tf", tf,
                 "--start", IS_START, "--end", IS_END,
                 "--score-mode", score_mode])

        survivors = q3_survivors(strategy)
        if runner.dry_run:
            # With nothing run there is no handoff to read; assume the full
            # grid so the plan shows every command the real run would issue.
            survivors = [{"symbol": s, "tf": t} for s in symbols
                         for t in timeframes]
        if not survivors:
            runner._log(f"\n   {strategy}: no pair designated {Q3_CODE} — "
                        f"nothing to sweep")
            rows.append(_row(strategy, note=f"no {Q3_CODE} survivor"))
            continue

        for pair in survivors:
            symbol = str(pair.get("symbol"))
            tf = str(pair.get("tf") or pair.get("timeframe"))

            # ---- Stage 2 -------------------------------------------------
            scan = runner.run(
                f"STAGE 2 {strategy} {symbol} {tf}",
                ["backtest/scan.py", "--strat", strategy,
                 "--symbols", symbol, "--tf", tf,
                 "--start", IS_START, "--end", IS_END])
            best_params = out_dir / f"best_params_{symbol}_{tf}.json"
            if not runner.dry_run and not best_params.is_file():
                rows.append(_row(
                    strategy, symbol, tf,
                    note=f"Stage 2 wrote no {best_params.name} "
                         f"(rc={scan.get('rc')})"))
                continue

            # ---- Stage 3, ONE timeframe per invocation -------------------
            runner.run(
                f"STAGE 3 {strategy} {symbol} {tf}",
                ["backtest/audit_gates.py", "--strat", strategy,
                 "--symbols", symbol, "--tf", tf,
                 "--holdout-start", HOLDOUT_START,
                 "--out-dir", str(out_dir),
                 # Certify Version B as well where a classifier exists. Stage 3
                 # resolves B PER PAIR from Stage 1's handoff; --ml forces it
                 # everywhere, which is a superset and cannot cause the
                 # version-mismatch bug the per-pair resolution exists to fix.
                 "--ml"])

            audit = out_dir / f"gate_audit_{symbol}_{tf}.json"
            verdicts = gate_r_verdict(audit)
            if runner.dry_run:
                verdicts = {"A": {"status": "(dry run)", "pf": None,
                                  "trades": None}}

            for version, verdict in sorted(verdicts.items()):
                row = _row(strategy, symbol, tf, version,
                           gate_r=verdict.get("status"),
                           pf=verdict.get("pf"),
                           trades=verdict.get("trades"),
                           certified=verdict.get("status") == "PASS")

                if not row["certified"]:
                    rows.append(row)
                    continue

                # ---- Stage 5, ONLY with --promote ------------------------
                portfolio, why_not = route_for(symbol)
                if portfolio is None:
                    row["note"] = f"certified but NOT routable: {why_not}"
                    rows.append(row)
                    continue
                if not promote:
                    row["note"] = (f"certified; promotion withheld "
                                   f"(--promote not passed) -> {portfolio}")
                    rows.append(row)
                    continue

                promoted = runner.run(
                    f"STAGE 5 {strategy} {symbol} {tf} V{version}",
                    ["backtest/promote.py", "--strat", strategy,
                     "--version", version, "--source", str(source),
                     "--audit-file", str(audit),
                     "--symbol", symbol, "--timeframe", tf,
                     "--portfolio", portfolio,
                     "--require-certification"])
                row["promoted"] = bool(promoted.get("ok"))
                row["note"] = (f"promoted to {portfolio}"
                               if row["promoted"]
                               else f"promote failed rc={promoted.get('rc')}")

                # ---- Stage 4, on what was actually promoted --------------
                if row["promoted"]:
                    argv = ["backtest/verify_full.py", "--strat", strategy,
                            "--symbols", symbol, "--tf", tf,
                            "--start", LIFECYCLE_START, "--end", LIFECYCLE_END,
                            "--out-dir", str(out_dir)]
                    if version == "B":
                        argv.append("--ml")
                    runner.run(f"STAGE 4 {strategy} {symbol} {tf} V{version}",
                               argv)
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------
def render_report(rows, started_at: str, args) -> str:
    def cell(value, spec="") -> str:
        if value is None or value == "":
            return "—"
        if spec and isinstance(value, (int, float)):
            return format(value, spec)
        return str(value)

    certified = [r for r in rows if r.get("certified")]
    promoted = [r for r in rows if r.get("promoted")]

    lines = [
        "# Q3 AUTONOMOUS RUN — SUMMARY",
        "",
        f"- started `{started_at}`  finished `{_utcnow()}`",
        f"- window: in-sample `{IS_START}..{IS_END}`, "
        f"holdout `{HOLDOUT_START}..present`, "
        f"lifecycle `{LIFECYCLE_START}..{LIFECYCLE_END}`",
        f"- score mode: `{args.score_mode}`",
        f"- promotion: **{'ENABLED' if args.promote else 'WITHHELD'}**"
        + ("" if args.promote else "  (`--promote` was not passed)"),
        f"- target quadrant: **{Q3_CODE} — {Q3_LABEL}**",
        "",
        f"**{len(certified)} certified**, **{len(promoted)} promoted**, "
        f"of {len(rows)} evaluated configurations.",
        "",
        "## Leaderboard",
        "",
        "| Strategy | Symbol | TF | Ver | Gate R PF | Gate R Trades | "
        "Certified | Promoted | Note |",
        "|---|---|---|---|---:|---:|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (not r.get("certified"),
                                         r["strategy"], str(r["symbol"]),
                                         str(r["tf"]), str(r["version"]))):
        lines.append(
            f"| {r['strategy']} | {r['symbol']} | {r['tf']} | {r['version']} "
            f"| {cell(r.get('pf'), '.2f')} | {cell(r.get('trades'), '.0f')} "
            f"| {'YES' if r.get('certified') else 'no'} "
            f"| {'YES' if r.get('promoted') else 'no'} "
            f"| {cell(r.get('note'))} |")

    lines += [
        "",
        "## Gate R",
        "",
        f"Profit factor >= 1.00 over >= 30 trades INSIDE {Q3_CODE}, measured "
        f"on the holdout. Gates 1, 2 and 3 are computed and reported by "
        f"Stage 3 as evidence and cannot fail a certification.",
        "",
        "## Reloading the live switchboard",
        "",
        "`config/portfolios.json` is read when the dispatcher is constructed, "
        "so a promotion reaches the live loop only on restart:",
        "",
        "```bash",
        "sudo systemctl restart trading-master-live.service",
        "```",
        "",
        "**Check the other three units by hand afterwards.** They do not come "
        "back on their own and the NFS mount races the NAS boot:",
        "",
        "```bash",
        "systemctl is-active trading-master-live trading-regime-daemon \\",
        "                    trading-watchdog trading-nt8-listener",
        f"{PY} scripts/check_market_regime.py --strategies",
        "```",
        "",
        "Confirm zero strategies report `certification_unresolved`, and that "
        "anything promoted here shows `ACTIVE | regime_match` when its "
        f"contract sits in {Q3_CODE}.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unattended Q3 discovery -> sweep -> certification -> "
                    "promotion. Every stage is a subprocess; a failure is "
                    "recorded and the campaign continues.")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    ap.add_argument("--tf", nargs="+", default=list(TIMEFRAMES),
                    dest="timeframes")
    ap.add_argument("--score-mode", default="vol_normalized",
                    choices=["alpha", "vol_normalized"])
    ap.add_argument("--promote", action="store_true",
                    help="actually run Stage 5. WITHHELD BY DEFAULT: this "
                         "rewrites config/portfolios.json and stages packages "
                         "into strategies/approved_incubator/, and launching "
                         "the script by mistake must not do that. Everything "
                         "up to and including certification runs either way.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command the campaign would issue and "
                         "run none of them")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = artifacts_root()
    root.mkdir(parents=True, exist_ok=True)
    started = _utcnow()
    runner = Runner(root / "q3_autonomous_run.log", dry_run=args.dry_run)

    runner._log("=" * 78)
    runner._log(f"Q3 AUTONOMOUS RUN  {started}")
    runner._log(f"  strategies : {', '.join(args.strategies)}")
    runner._log(f"  symbols    : {', '.join(args.symbols)}")
    runner._log(f"  timeframes : {', '.join(args.timeframes)}")
    runner._log(f"  score mode : {args.score_mode}")
    runner._log(f"  promotion  : {'ENABLED' if args.promote else 'WITHHELD'}")
    runner._log("=" * 78)

    rows = campaign(runner, args.strategies, args.symbols, args.timeframes,
                    args.score_mode, args.promote)

    report = root / "Q3_AUTONOMOUS_RUN_SUMMARY.md"
    text = render_report(rows, started, args)
    if not args.dry_run:
        report.write_text(text, encoding="utf-8")
    runner._log("\n" + text)
    runner._log(f"\nreport  {report}")
    runner._log(f"log     {runner.log_path}")

    failed = [c for c in runner.calls if c.get("ok") is False]
    if failed:
        runner._log(f"\n{len(failed)} stage invocation(s) failed — see the log.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
