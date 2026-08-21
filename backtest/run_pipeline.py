#!/usr/bin/env python3
"""
backtest/run_pipeline.py - the unified CLI orchestrator for Stages 1-4.

Location: ~/src/trading/backtest/run_pipeline.py

Runs Stage 1 (baseline.py), Stage 2 (scan.py), Stage 3 (audit_gates.py) and
Stage 4 (verify_full.py) in sequence with `subprocess.run(..., check=True)`,
posting the Discord card for Stages 1-3 with `--report-discord` and optionally
promoting Stage 3's certified configurations with `--auto-promote` - or with
`--promote-only`, which runs that promotion pass ALONE against the
certifications already on the handoff.

WHAT THIS TRADES AWAY, STATED PLAINLY
-------------------------------------
CLAUDE.md's five-stage pipeline deliberately does NOT chain: each stage prints
the next one's command and stops, so a human reads the evidence in between.
This script removes that checkpoint. That is its entire purpose and it is a
real cost - the stages are sequenced so a screen that promoted nothing, a sweep
whose winner is a spike, or a certification resting on 31 holdout trades is
seen by somebody before the next stage spends hours building on it. Nothing
here re-adds that judgement; it only makes the sequence cheap to launch.

Two guard rails survive, because they are not judgement calls:

  * `--auto-promote` promotes ONLY what Stage 3 already recorded as
    `certified: true`, through `promote.py --require-certification`, and never
    passes `--force`. It cannot promote a configuration the gate audit refused.
    It is still not a substitute for the four-choice menu: it commits to git
    what a human did not look at. It iterates EVERY certified row across every
    timeframe, each against that pair's own `gate_audit_<SYMBOL>_<TF>.json` -
    never the unsuffixed file, which holds whichever timeframe ran last - and
    writes the outcome back onto Stage 3's summary so the Discord card can say
    what was promoted and at which commit instead of printing a command that
    has already run. That is also why the Stage 3 card is posted AFTER the
    promotions when this flag is on.

  * `--promote-only` is that same Stage 5 pass with Stages 1-4 SKIPPED, and it
    is the command the Stage 3 Discord card prints. The distinction is the
    whole reason it exists: `--auto-promote` re-runs the screen and the sweep
    first, which overwrites the handoff the card was built from, so the
    winners it promotes are a fresh sweep's and not the ones the reader is
    looking at. `--symbols`, `--tf` and the window are not used by that path -
    `auto_promote` promotes every certified row on the handoff and always has.

The run ends on one table of every configuration Stage 3 indexed: what was
promoted, what was certified and left for a human, and for the rest WHY - a
Gate R that failed on the factor, a quadrant that starved, a NOT EVALUATED with
no designated quadrant, and a run that broke are four different findings fixed
by four different pieces of work, and one "not promoted" token hides all of
them.
  * The in-sample window is refused before Stage 1 reads a bar if it reaches
    HOLDOUT_START, the same rule `scan.check_in_sample_window` enforces. An
    orchestrator that spends the holdout in Stage 1 spends it for every stage
    downstream, and nothing after it can detect that.

WHY THE ARGUMENTS ARE NOT FORWARDED BLINDLY
-------------------------------------------
The four stages do not take the same flags, and forwarding `--symbols/--tf/
--start/--end` to each in turn produces a run that looks correct and is not:

  * Stage 2 takes NO `--symbols` and NO `--tf` here. With both omitted it
    sweeps Stage 1's EXACT surviving (symbol, timeframe) pairs. Passing them
    is an override that crosses the two axes, and the survivors are RAGGED -
    NQ at 5m and 15m, GC at 15m only - so the cross product sweeps
    configurations the screen dropped.
  * Stage 3 has no `--start`/`--end` at all. It takes `--is-start`/`--is-end`
    and `--holdout-start`/`--holdout-end`. Forwarding `--start` would fail on
    an unrecognised flag, which is the good case; forwarding a window is the
    bad one.
  * Stages 3 and 4 take ONE timeframe. A multi-timeframe run is expanded into
    one Stage 3 and one Stage 4 invocation PER timeframe, and only for the
    timeframes Stage 2 actually optimised - read back off its summary. Running
    Stage 3 at a timeframe nothing survived exits 1 ("Nothing to certify"),
    and under `check=True` that would abort a pipeline over a screening
    result, which is not a failure.
  * Stage 4's window is the lifecycle, not the in-sample window: it spans the
    holdout by construction and reports `is_certification: false`. Its `--end`
    is passed explicitly as today rather than left to the module's hardcoded
    2026-01-01 default, which silently stops covering the newest bars.

Discord failures are NOT fatal. A card is an announcement; a completed
certification is evidence. Losing the second because the first got a 500 back
is the wrong trade, so the reporter runs with `check=False` and a warning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.pipeline import (  # noqa: E402
    CHARTER_IS_START,
    CHARTER_IS_END,
    HOLDOUT_START,
    STAGE2_SUMMARY_FILE,
    STAGE3_SUMMARY_FILE,
    leaderboard,
    pipeline_dir,
    read_stage,
    write_stage,
)

BACKTEST = REPO / "backtest"

STAGE_SCRIPTS = {
    1: BACKTEST / "baseline.py",
    2: BACKTEST / "scan.py",
    3: BACKTEST / "audit_gates.py",
    4: BACKTEST / "verify_full.py",
}
DISCORD_SCRIPT = BACKTEST / "discord_reporter.py"
PROMOTE_SCRIPT = BACKTEST / "promote.py"

#: Stages whose handoff has a Discord card. Stage 4 has none - it is a
#: lifecycle report, not a verdict, and there is nothing to announce.
DISCORD_STAGES = (1, 2, 3)

W = 78


def _py() -> list[str]:
    """
    The interpreter prefix, unbuffered.

    `-u` because these are long background runs whose only progress indicator
    is the log: buffered, a stage that dies at hour three leaves a log ending
    an hour before the failure.
    """
    return [sys.executable, "-u"]


# ---------------------------------------------------------------------------
# window and timeframe validation
# ---------------------------------------------------------------------------

def parse_timeframes(raw: str | None) -> list[str]:
    """Split a comma-separated --tf into an ordered, de-duplicated list."""
    seen: list[str] = []
    for part in str(raw or "").split(","):
        tf = part.strip()
        if tf and tf not in seen:
            seen.append(tf)
    return seen


def check_in_sample_window(start: str | None, end: str | None) -> None:
    """
    Refuse an in-sample window that reaches the holdout, before Stage 1 runs.

    The same rule `scan.check_in_sample_window` applies, hoisted to the front
    of the chain: Stage 1 screens on this window and Stage 2 optimises on it,
    so a window that runs into the holdout has spent it before Gate 3 is ever
    evaluated - and every stage downstream then reports a holdout it has
    already been fitted to. There is deliberately no override flag. The
    holdout can only be spent once.
    """
    if not start:
        raise ValueError("--start is required (it defaults to the charter window)")
    if not end:
        raise ValueError(
            "--end is required. Omitted, it runs to the end of the lake, which "
            f"eats the holdout beginning {HOLDOUT_START}."
        )
    if end >= HOLDOUT_START:
        raise ValueError(
            f"in-sample --end {end} reaches the holdout ({HOLDOUT_START}). "
            f"Stages 1 and 2 fit what they read, so a holdout they have "
            f"screened and optimised over is not a holdout. Use "
            f"--end {CHARTER_IS_END}."
        )


# ---------------------------------------------------------------------------
# command builders - pure, so the wiring is testable without running a stage
# ---------------------------------------------------------------------------

def stage1_cmd(strat: str, symbols: str, tfs: Sequence[str], start: str,
               end: str, out_dir: str | None = None) -> list[str]:
    """Stage 1: the regime screen, over every requested timeframe at once."""
    cmd = _py() + [str(STAGE_SCRIPTS[1]), "--strat", strat,
                   "--symbols", symbols, "--tf", ",".join(tfs),
                   "--start", start, "--end", end]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def stage2_cmd(strat: str, start: str, end: str,
               out_dir: str | None = None) -> list[str]:
    """
    Stage 2: the sweep, over Stage 1's EXACT surviving pairs.

    No `--symbols` and no `--tf`, deliberately - see the module docstring.
    That omission IS the inheritance, and adding either would replace the
    ragged survivor list with the cross product of two axes.
    """
    cmd = _py() + [str(STAGE_SCRIPTS[2]), "--strat", strat,
                   "--start", start, "--end", end]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def stage3_cmd(strat: str, tf: str, is_start: str, is_end: str,
               holdout_start: str = HOLDOUT_START,
               holdout_end: str | None = None,
               promote: bool = True,
               out_dir: str | None = None) -> list[str]:
    """
    Stage 3: certify one timeframe against the holdout.

    `--holdout-end` is left OFF unless asked for, because the stage defaults
    it to the present; pinning a year here would stop certifying against the
    newest bars the moment one rolled over, and the verdict looks identical
    either way. No `--symbols`: omitted, it certifies the exact pairs Stage 2
    optimised, with the parameters locked.
    """
    cmd = _py() + [str(STAGE_SCRIPTS[3]), "--strat", strat, "--tf", tf,
                   "--is-start", is_start, "--is-end", is_end,
                   "--holdout-start", holdout_start]
    if holdout_end:
        cmd += ["--holdout-end", holdout_end]
    if not promote:
        cmd += ["--no-promote"]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def stage4_cmd(strat: str, tf: str, start: str, end: str,
               out_dir: str | None = None) -> list[str]:
    """
    Stage 4: the lifecycle run for one timeframe.

    The window spans the holdout on purpose - this is not a certification and
    the module records `is_certification: false`. `--end` is passed rather
    than defaulted for the reason given in the module docstring.
    """
    cmd = _py() + [str(STAGE_SCRIPTS[4]), "--strat", strat, "--tf", tf,
                   "--start", start, "--end", end]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def discord_cmd(strat: str, stage: int, dry_run: bool = False,
                out_dir: str | None = None) -> list[str]:
    """The card for one stage, read from that stage's own handoff."""
    if stage not in DISCORD_STAGES:
        raise ValueError(
            f"stage {stage} has no Discord card; cards exist for "
            f"{', '.join(str(s) for s in DISCORD_STAGES)}"
        )
    cmd = _py() + [str(DISCORD_SCRIPT), "--stage", str(stage), "--strat", strat]
    if dry_run:
        cmd += ["--dry-run"]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def promote_cmd(strat: str, version: str, source: str | Path,
                audit_file: str | Path, symbol: str,
                timeframe: str) -> list[str]:
    """
    Stage 5 for ONE certified configuration.

    `--require-certification` is always passed and `--force` never is: this
    path may only ratify a verdict Stage 3 already reached. A promotion that
    could override a gate is a decision, and decisions are the operator's.
    """
    return _py() + [str(PROMOTE_SCRIPT), "--strat", strat,
                    "--version", version, "--source", str(source),
                    "--audit-file", str(audit_file), "--symbol", symbol,
                    "--timeframe", timeframe, "--require-certification"]


# ---------------------------------------------------------------------------
# handoff readers
# ---------------------------------------------------------------------------

def stage2_timeframes(summary: dict[str, Any] | None) -> list[str]:
    """
    The timeframes Stage 2 actually optimised something at.

    Read off the summary's rows rather than taken from the CLI, because a
    timeframe every contract was screened out at has no parameters to certify
    and Stage 3 exits 1 there. Under `check=True` that would end the run on a
    screening result. Rows Stage 2 recorded as an ERROR are excluded for the
    opposite reason: the sweep never produced a winner, so there is nothing
    locked to certify against.
    """
    rows = (summary or {}).get("results") or []
    tfs: list[str] = []
    for row in rows:
        if str(row.get("status", "")).upper() != "OPTIMIZED":
            continue
        tf = row.get("timeframe") or row.get("tf")
        if tf and tf not in tfs:
            tfs.append(str(tf))
    return tfs


def certified_rows(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Stage 3's certified configurations, as it recorded them.

    Keyed on the `certified` flag the audit wrote, never re-derived from the
    gate fields: a second opinion about what passed would be free to disagree
    with the audit it is promoting, and the two would be compared by nobody.
    """
    rows = (summary or {}).get("results") or []
    return [r for r in rows if r.get("certified") is True]


def _strategy_source(strat: str) -> Path:
    """
    The module path `promote.py --source` cites, behind a seam.

    Imported lazily because `backtest.run` pulls in the engine and vectorbtpro,
    which an orchestrator that never promotes has no reason to pay for - and
    which a unit test has no way to avoid otherwise.
    """
    from backtest.run import resolve_strategy
    return resolve_strategy(strat)


def _read_summary(strat: str, filename: str,
                  expect_stage: int, out_dir: str | None) -> dict[str, Any] | None:
    path = pipeline_dir(strat, out_dir) / filename
    if not path.exists():
        return None
    return read_stage(path, expect_stage=expect_stage, expect_strategy=strat)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def _git_head() -> str | None:
    """
    The short SHA of HEAD, or None.

    A seam, and deliberately a separate one from `run_step`'s runner: this is
    the only place the orchestrator reads git, and it must never be able to
    fail a pipeline. Every exception is swallowed and reported as "no commit
    recorded" - a promotion that happened without its hash landing on the card
    is a smaller problem than a completed four-stage run ending on a
    subprocess error after promote.py already committed.
    """
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=str(REPO), capture_output=True, text=True,
                              check=False)
    except Exception:                                             # noqa: BLE001
        return None
    out = (getattr(proc, "stdout", "") or "").strip()
    return out or None


def failure_reason(row: dict[str, Any]) -> str:
    """
    Why one configuration was not promoted, in one line.

    The distinctions are the point, because they are fixed by different work
    and a single "Gate R FAIL" hides all of them:

      * NOT AUDITED - the run broke or Stage 2 left no parameters. Nothing was
        measured, so this is not a statement about the strategy at all.
      * NOT EVALUATED - no quadrant was designated, so Gate R had no target.
        Not a pass and not a failure of the edge.
      * REGIME STARVATION - the quadrant it was certified in barely traded out
        of sample. The edge was never re-tested; the in-sample best-of-four
        pick simply did not recur.
      * Gate R FAIL on the factor - the edge WAS re-tested in its own
        environment and lost money there. This is the honest failure.

    Transcribed from what Stage 3 recorded, never re-derived: this module
    promotes on the audit's own `certified` flag, and a reason computed from
    the numbers beside it would be free to disagree with the verdict.
    """
    status = str(row.get("status") or "UNKNOWN").upper()
    gate = str(row.get("gate_regime") or "NOT EVALUATED").upper()
    if status == "NOT AUDITED":
        return f"NOT AUDITED · {row.get('error') or 'the audit did not run'}"
    if row.get("regime_starvation"):
        return f"REGIME STARVATION · {row['regime_starvation']}"
    if "NOT EVALUATED" in gate:
        return ("Gate R NOT EVALUATED · Stage 1 designated no home quadrant, "
                "so there was no target to certify")
    pf = row.get("oos_profit_factor")
    quad = row.get("quadrant") or "the target quadrant"
    n = row.get("oos_trade_count")
    where = f"{quad}" + (f" over {n} holdout trades" if n is not None else "")
    return (f"Gate R {gate} · OOS PF "
            + ("not measured" if pf is None else f"{float(pf):.2f}")
            + f" in {where}")


def promotion_summary_table(rows: list[dict[str, Any]],
                            promotions: list[dict[str, Any]]) -> str:
    """
    The table the pipeline ends on: what was promoted, and why the rest was
    not.

    Every configuration Stage 3 indexed is a row, certified or not. A table of
    the winners alone reads as a run in which nothing else happened, and the
    reason column is the half an operator acts on - "Gate R failed" sends
    someone to look at the edge, "regime starvation" sends them to look at
    whether the strategy ever met its own environment again, and "not audited"
    sends them to the log.

    Rendered through `pipeline.leaderboard` for the same reason the stages'
    own tables are: these are read as a sequence, and a Symbol column aligned
    one way here and another at Stage 3 makes two tables of the same contracts
    look like tables of different things.
    """
    done = {(str(d.get("symbol")), str(d.get("timeframe")),
             str(d.get("version") or "A")): d for d in promotions}
    body: list[list[str]] = []
    for row in rows:
        key = (str(row.get("symbol")), str(row.get("timeframe")),
               str(row.get("version") or "A"))
        if row.get("certified") is True:
            record = done.get(key)
            if record is None:
                outcome, detail = "CERTIFIED", "not promoted (--auto-promote off)"
            elif record.get("promoted"):
                outcome = "PROMOTED"
                detail = str(record.get("incubator_dir")
                             or record.get("commit") or "committed")
            else:
                outcome = "PROMOTE FAILED"
                detail = str(record.get("error") or "promote.py exited non-zero")
        else:
            outcome, detail = "NOT CERTIFIED", failure_reason(row)
        body.append([
            str(row.get("symbol") or "?"), str(row.get("timeframe") or "?"),
            str(row.get("version") or "--"),
            str(row.get("quadrant") or "--"),
            str(row.get("gate_regime") or "NOT EVALUATED"),
            outcome, detail,
        ])
    body.sort(key=lambda r: ({"PROMOTED": 0, "CERTIFIED": 1,
                              "PROMOTE FAILED": 2}.get(r[5], 3),
                             r[0], r[1]))
    return leaderboard(
        "PIPELINE RESULT · certification and promotion",
        ["SYMBOL", "TF", "VER", "QUAD", "GATE R", "OUTCOME", "DETAIL"],
        body,
        align=["<", "<", "<", "<", "<", "<", "<"],
        empty="Stage 3 indexed no configuration - nothing reached a verdict")


def record_auto_promotion(strat: str, out_dir: str | None,
                          promotions: list[dict[str, Any]],
                          commit: str | None) -> Path | None:
    """
    Write the auto-promotion outcome back onto Stage 3's summary.

    The Discord card reads this to decide whether its promotion section is
    headed READY FOR PROMOTION or AUTOMATICALLY PROMOTED, and to print the
    commit. It is written HERE rather than by Stage 3 because Stage 3 stages
    into the incubator and never commits - the hash does not exist until
    Stage 5 has run, and a card that inferred a promotion from a seal would
    announce every staged-but-uncommitted configuration as promoted.

    Only the `auto_promotion` key is added; `results`, `audits` and `runs` are
    rewritten byte for byte from what was read, so this cannot restate a
    verdict. A summary that cannot be read is reported and skipped - the
    promotions themselves are already committed and are not undone by a
    failure to annotate an index.
    """
    path = pipeline_dir(strat, out_dir) / STAGE3_SUMMARY_FILE
    try:
        blob = read_stage(path, expect_stage=3, expect_strategy=strat)
    except Exception as e:                                        # noqa: BLE001
        print(f"  WARNING  could not annotate {path.name} with the promotion "
              f"outcome ({type(e).__name__}: {e}); the promotions themselves "
              f"stand.", file=sys.stderr)
        return None
    blob["auto_promotion"] = {
        "ran": True,
        "commit": commit,
        "promoted": sum(1 for d in promotions if d.get("promoted")),
        "failed": sum(1 for d in promotions if not d.get("promoted")),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "promotions": promotions,
    }
    return write_stage(path, 3, strat, blob)


def _banner(text: str) -> str:
    return f"\n{'=' * W}\n{text}\n{'=' * W}"


def run_step(cmd: Sequence[str], label: str, *, dry_run: bool = False,
             check: bool = True,
             runner=None) -> int:
    """
    One subprocess, announced before it starts.

    `runner` is resolved at CALL time - `runner or subprocess.run`, never a
    default argument bound to `subprocess.run` at definition time. The
    difference is the whole testability of this module: a default binds the
    real function into the signature, so patching `run_pipeline.subprocess.run`
    has no effect and a test that believes it is recording commands silently
    LAUNCHES all four stages instead.

    `check=True` for the four stages: a stage that failed has not produced the
    handoff the next one reads, and continuing would report the PREVIOUS run's
    artifacts as this run's result.
    """
    print(_banner(f"{label}\n$ {' '.join(str(c) for c in cmd)}"), flush=True)
    if dry_run:
        print("  DRY RUN  not executed.", flush=True)
        return 0
    proc = (runner or subprocess.run)(list(cmd), cwd=str(REPO), check=check)
    return int(getattr(proc, "returncode", 0) or 0)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=("Run Stages 1-4 in sequence. NOTE: the stages are "
                     "designed NOT to chain, so a human reads the evidence "
                     "between them; this removes that checkpoint."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--strat", required=True, help="Strategy module name")
    p.add_argument("--symbols", default="ALL",
                   help="Stage 1 universe: NQ, NQ,ES or ALL (default ALL)")
    p.add_argument("--tf", default="15m",
                   help="Timeframe(s) for Stage 1, comma-separated. Stages 3 "
                        "and 4 run once per timeframe Stage 2 optimised.")
    p.add_argument("--start", default=CHARTER_IS_START,
                   help=f"In-sample start (default {CHARTER_IS_START})")
    p.add_argument("--end", default=CHARTER_IS_END,
                   help=f"In-sample end (default {CHARTER_IS_END}). Refused if "
                        f"it reaches {HOLDOUT_START}.")
    p.add_argument("--report-discord", action="store_true",
                   help="Post the Discord card after Stages 1, 2 and 3")
    p.add_argument("--auto-promote", action="store_true",
                   help="Run Stage 5 for every configuration Stage 3 recorded "
                        "as certified. Never overrides a gate.")
    p.add_argument("--promote-only", action="store_true",
                   help="Skip Stages 1-4 and run Stage 5 alone against the "
                        "certifications already on the handoff. This is the "
                        "command the Stage 3 Discord card prints.")
    p.add_argument("--out-dir", default=None,
                   help="Override the artifacts root (passed to every stage)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every command in order and run nothing")
    return p


def _promote_only(args: argparse.Namespace) -> int:
    """
    Stage 5 alone, against the certifications already on Stage 3's handoff.

    This is what the Stage 3 Discord card tells a reader to run, and the whole
    point of it is that it reads the evidence the card was built from rather
    than making new evidence. `--auto-promote` re-runs Stages 1-4 first, which
    OVERWRITES that handoff: the winners it then promotes are a fresh sweep's,
    not the ones on the card a reader is looking at, and the two are compared
    by nobody. One flag is the difference between promoting what was certified
    and re-certifying from scratch, so they are separate flags.

    `--symbols`, `--tf` and the window are ignored here and say so, because
    `auto_promote` promotes every certified row on the handoff and always has -
    a symbol list that silently narrowed nothing would read as one that scoped
    the promotion.
    """
    strat, out_dir, dry = args.strat, args.out_dir, args.dry_run
    print(_banner(
        f"PROMOTE ONLY  ·  {strat}\n"
        f"  Stages 1-4 are NOT run. Stage 5 promotes what Stage 3 already\n"
        f"  recorded as certified, through promote.py "
        f"--require-certification.\n"
        f"  --symbols/--tf/--start/--end are not used by this path."))

    outcome = auto_promote(strat, out_dir=out_dir, dry_run=dry)
    rc = int(outcome["returncode"])
    rows, promotions = outcome["rows"], outcome["promotions"]

    if args.report_discord:
        # After the promotions, exactly as in the --auto-promote path: the
        # outcome is on the handoff by now, so the card reads AUTOMATICALLY
        # PROMOTED with the commit instead of printing a command that has
        # already run. check=False - a webhook outage must not fail a
        # promotion that already happened.
        code = run_step(discord_cmd(strat, 3, out_dir=out_dir),
                        "DISCORD · Stage 3 card", dry_run=dry, check=False)
        if code:
            print(f"  WARNING  Stage 3 Discord card failed (exit {code}); the "
                  f"promotion itself is unaffected.", file=sys.stderr,
                  flush=True)

    if rows:
        print("\n" + promotion_summary_table(rows, promotions))
    print(_banner("PROMOTE ONLY COMPLETE"))
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.promote_only:
        return _promote_only(args)

    try:
        check_in_sample_window(args.start, args.end)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    tfs = parse_timeframes(args.tf)
    if not tfs:
        print("REFUSED: --tf named no timeframe", file=sys.stderr)
        return 2

    strat, out_dir, dry = args.strat, args.out_dir, args.dry_run

    print(_banner(
        f"UNIFIED PIPELINE  ·  {strat}\n"
        f"  in-sample   {args.start} -> {args.end}\n"
        f"  holdout     {HOLDOUT_START} -> present   (Stage 3 only)\n"
        f"  symbols     {args.symbols}\n"
        f"  timeframes  {', '.join(tfs)}\n"
        f"  discord     {'yes' if args.report_discord else 'no'}\n"
        f"  auto-promote {'yes (certified only)' if args.auto_promote else 'no'}\n"
        f"  NOTE  the stages are designed not to chain; this run reads no\n"
        f"        evidence between them."))

    def discord(stage: int) -> None:
        if not args.report_discord:
            return
        # check=False: a card is an announcement, the stage's artifacts are the
        # evidence. A webhook outage must not discard a completed stage.
        rc = run_step(discord_cmd(strat, stage, out_dir=out_dir),
                      f"DISCORD · Stage {stage} card", dry_run=dry, check=False)
        if rc:
            print(f"  WARNING  Stage {stage} Discord card failed (exit {rc}); "
                  f"the stage itself is unaffected.", file=sys.stderr, flush=True)

    try:
        run_step(stage1_cmd(strat, args.symbols, tfs, args.start, args.end, out_dir),
                 "STAGE 1 · baseline.py · regime screen", dry_run=dry)
        discord(1)

        run_step(stage2_cmd(strat, args.start, args.end, out_dir),
                 "STAGE 2 · scan.py · parameter sweep over Stage 1's exact pairs",
                 dry_run=dry)
        discord(2)
    except subprocess.CalledProcessError as e:
        print(f"\nABORTED: {' '.join(str(c) for c in e.cmd)} exited "
              f"{e.returncode}. Later stages read this one's handoff, so "
              f"continuing would report an earlier run's artifacts as this "
              f"run's result.", file=sys.stderr)
        return int(e.returncode or 1)

    # Which timeframes have something to certify is a fact about Stage 2's
    # output, not about the CLI.
    if dry:
        certify_tfs = list(tfs)
        print("\n  DRY RUN  assuming every requested timeframe reached Stage 3.")
    else:
        summary = _read_summary(strat, STAGE2_SUMMARY_FILE, 2, out_dir)
        certify_tfs = stage2_timeframes(summary)
        skipped = [t for t in tfs if t not in certify_tfs]
        if skipped:
            print(f"\n  Stage 2 optimised nothing at {', '.join(skipped)} - "
                  f"skipping Stages 3 and 4 there. That is a screening "
                  f"result, not a failure.", flush=True)

    if not certify_tfs:
        print(_banner("STOPPED AFTER STAGE 2\n"
                      "  No configuration survived to a parameter set, so "
                      "there is nothing to\n  certify. This is a result: the "
                      "screen found no environment to sweep."))
        return 0

    try:
        for tf in certify_tfs:
            run_step(stage3_cmd(strat, tf, args.start, args.end,
                                out_dir=out_dir),
                     f"STAGE 3 · audit_gates.py · certify {tf} on the holdout",
                     dry_run=dry)
        # The card is posted AFTER Stage 3 has run at every timeframe, because
        # Stage 3 merges each invocation into one summary and the card reads
        # that summary: posted per timeframe it would announce the same
        # campaign several times, each edition missing the timeframes that had
        # not run yet. With --auto-promote it waits longer still - see below.
        if not args.auto_promote:
            discord(3)

        for tf in certify_tfs:
            run_step(stage4_cmd(strat, tf, args.start, _today(), out_dir),
                     f"STAGE 4 · verify_full.py · lifecycle {tf}", dry_run=dry)
    except subprocess.CalledProcessError as e:
        print(f"\nABORTED: {' '.join(str(c) for c in e.cmd)} exited "
              f"{e.returncode}.", file=sys.stderr)
        return int(e.returncode or 1)

    rc = 0
    promotions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if args.auto_promote:
        outcome = auto_promote(strat, out_dir=out_dir, dry_run=dry)
        rc = int(outcome["returncode"])
        promotions, rows = outcome["promotions"], outcome["rows"]
        # Now the card, and only now: the promotion outcome is on the handoff,
        # so the section reads AUTOMATICALLY PROMOTED with the commit rather
        # than telling a reader to run a command that has already run.
        discord(3)
    elif not dry:
        rows = ((_read_summary(strat, STAGE3_SUMMARY_FILE, 3, out_dir) or {})
                .get("results") or [])

    if rows:
        print("\n" + promotion_summary_table(rows, promotions))

    print(_banner("PIPELINE COMPLETE"))
    return rc


def auto_promote(strat: str, *, out_dir: str | None = None,
                 dry_run: bool = False) -> dict[str, Any]:
    """
    Stage 5 for every configuration Stage 3 recorded as certified.

    Returns `{"returncode", "promotions", "rows", "commit"}` rather than a bare
    exit code, because the caller prints the end-of-run table from it: a
    summary rebuilt by re-reading the handoff could disagree with what this
    function actually launched, and the two would be compared by nobody.

    Reads the flag the audit wrote and promotes through
    `promote.py --require-certification`, so a configuration the gate audit
    refused cannot be promoted here however this was invoked. A promotion that
    fails is recorded and the rest continue: the certifications are already on
    the record either way.

    **It iterates EVERY certified row across every timeframe.** Stage 3
    certifies one timeframe per invocation and merges its verdicts into one
    summary, so a campaign that certified CL at both 5m and 15m has two rows
    here and each is promoted against its OWN
    `gate_audit_<SYMBOL>_<TF>.json` - the per-pair audit named on the row,
    never the unsuffixed file, which holds whichever timeframe ran last.
    """
    print(_banner("STAGE 5 · promote.py · certified configurations only"))
    if dry_run:
        print("  DRY RUN  Stage 3's summary is not read.")
        return {"returncode": 0, "promotions": [], "rows": [], "commit": None}

    summary = _read_summary(strat, STAGE3_SUMMARY_FILE, 3, out_dir)
    if summary is None:
        print("  Stage 3 wrote no summary; nothing to promote.", file=sys.stderr)
        return {"returncode": 1, "promotions": [], "rows": [], "commit": None}

    rows = (summary or {}).get("results") or []
    winners = certified_rows(summary)
    refused = [r for r in rows if r.get("certified") is not True]
    for r in refused:
        print(f"  NOT PROMOTED  {r.get('symbol')} {r.get('timeframe')} "
              f"version {r.get('version')} · {failure_reason(r)}")
    if not winners:
        print("  Nothing was certified. Nothing promoted.")
        return {"returncode": 0, "promotions": [], "rows": rows, "commit": None}

    try:
        source = _strategy_source(strat)
    except (SystemExit, Exception) as e:                          # noqa: BLE001
        # resolve_strategy raises SystemExit. Uncaught, it would end the
        # process here - after all four stages completed - and the run would
        # report nothing about the certifications already on the record.
        print(f"  Cannot resolve the strategy module to promote from: {e}",
              file=sys.stderr)
        return {"returncode": 1, "promotions": [], "rows": rows, "commit": None}

    promotions: list[dict[str, Any]] = []
    failures = 0
    for r in winners:
        record = {"symbol": str(r.get("symbol")),
                  "timeframe": str(r.get("timeframe")),
                  "version": str(r.get("version") or "A"),
                  "audit_file": r.get("audit_file"),
                  "incubator_dir": r.get("incubator_dir"),
                  "promoted": False, "commit": None, "error": ""}
        audit = r.get("audit_file")
        if not audit:
            record["error"] = ("the row records no audit file, so there is no "
                               "certification to cite")
            print(f"  SKIPPED  {record['symbol']} {record['timeframe']}: "
                  f"{record['error']}", file=sys.stderr)
            promotions.append(record)
            failures += 1
            continue
        cmd = promote_cmd(strat, record["version"], source, audit,
                          record["symbol"], record["timeframe"])
        rc = run_step(cmd, f"PROMOTE {record['symbol']} {record['timeframe']} "
                           f"version {record['version']}", check=False)
        if rc:
            record["error"] = f"promote.py exited {rc}"
            print(f"  WARNING  promotion exited {rc}", file=sys.stderr)
            failures += 1
        else:
            # The hash AFTER this promotion's own commit, not one taken once
            # at the end: promote.py commits per configuration, so a single
            # hash for a batch of four would name three of them wrongly.
            record["promoted"] = True
            record["commit"] = _git_head()
        promotions.append(record)

    commit = next((d["commit"] for d in reversed(promotions) if d.get("commit")),
                  None)
    written = record_auto_promotion(strat, out_dir, promotions, commit)
    if written:
        print(f"\n  promotion outcome recorded → {written}")
    return {"returncode": 1 if failures else 0, "promotions": promotions,
            "rows": rows, "commit": commit}


if __name__ == "__main__":
    sys.exit(main())
