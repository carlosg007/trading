#!/usr/bin/env python3
"""
backtest/run_pipeline.py - the unified CLI orchestrator for Stages 1-4.

Location: ~/src/trading/backtest/run_pipeline.py

Runs Stage 1 (baseline.py), Stage 2 (scan.py), Stage 3 (audit_gates.py) and
Stage 4 (verify_full.py) in sequence with `subprocess.run(..., check=True)`,
posting the Discord card for every stage with `--report-discord` and optionally
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

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. This runs at import
# time, above the local imports below, because several modules resolve their
# BT_* variables while being imported (backtest.run's ARTIFACTS_ROOT) - loading
# the file inside main() would be too late for those and would work here, which
# is the kind of difference nobody notices until one runner silently uses the
# default path. The rules - the repository root derived from __file__ rather
# than the working directory, existing variables winning over the file, the
# CrossTrade credentials withheld from os.environ - live in ONE module rather
# than in a block copied into every runner: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # `python3 backtest/x.py` puts backtest/ on sys.path, not the repository
    # root, so mdlib is not importable until this runs.
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------


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

from backtest.promote import unrouted_packages  # noqa: E402
from backtest.run import DEFAULT_TFS, TF_GROUPS  # noqa: E402
from backtest.pipeline import (  # noqa: E402
    ML_THRESHOLD_DEFAULT,
    CHARTER_IS_START,
    CHARTER_IS_END,
    HOLDOUT_START,
    STAGE2_SUMMARY_FILE,
    STAGE3_SUMMARY_FILE,
    STAGE45,
    STAGE45_SUMMARY_FILE,
    DOW_GATE_FILE,
    leaderboard,
    artifacts_root,
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
    # Keyed on the integer stage id `pipeline.STAGE45` stamps into the
    # handoff, not on 4.5: `write_stage` writes `int(stage)` and a float key
    # here would truncate to 4 and collide with the lifecycle stage above.
    STAGE45: BACKTEST / "dow_gate.py",
}
DISCORD_SCRIPT = BACKTEST / "discord_reporter.py"
PROMOTE_SCRIPT = BACKTEST / "promote.py"

#: Stages with a Discord card, and the ORDER they are transmitted in. Cards
#: are posted in strict numerical sequence 1..5 - a reader scrolling a channel
#: reconstructs the campaign from the order the cards arrive in, so a Stage 3
#: card landing after Stage 5's promotion describes a decision that had
#: already been taken by the time it was announced.
DISCORD_STAGES = (1, 2, 3, 4, 5)

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
    """
    Split a comma-separated --tf into an ordered, de-duplicated list.

    A named group (`ALL_DAY_TRADING`) expands here to the same tuple
    `backtest.run.TF_GROUPS` gives every other stage, imported rather than
    restated: two spellings of "the day-trading ladder" that could drift apart
    would put Stage 1 and Stage 3 on different timeframe sets with every log
    line reading correctly.
    """
    text = str(raw or "").strip()
    if text.upper() in TF_GROUPS:
        return list(TF_GROUPS[text.upper()])
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
               end: str, out_dir: str | None = None,
               ml_threshold: float = ML_THRESHOLD_DEFAULT) -> list[str]:
    """Stage 1: the regime screen, over every requested timeframe at once."""
    cmd = _py() + [str(STAGE_SCRIPTS[1]), "--strat", strat,
                   "--symbols", symbols, "--tf", ",".join(tfs),
                   "--start", start, "--end", end,
                   "--ml-threshold", str(ml_threshold)]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def stage2_cmd(strat: str, start: str, end: str,
               out_dir: str | None = None,
               ml_threshold: float = ML_THRESHOLD_DEFAULT,
               allow_isolated_spikes: bool = False) -> list[str]:
    """
    Stage 2: the sweep, over Stage 1's EXACT surviving pairs.

    No `--symbols` and no `--tf`, deliberately - see the module docstring.
    That omission IS the inheritance, and adding either would replace the
    ragged survivor list with the cross product of two axes.
    """
    # --ml-threshold is NOT a scope flag. The omission of --symbols and --tf
    # is what makes Stage 2 inherit Stage 1's ragged survivor list; a
    # threshold constrains no pair and adding it changes nothing about which
    # configurations are swept.
    cmd = _py() + [str(STAGE_SCRIPTS[2]), "--strat", strat,
                   "--start", start, "--end", end,
                   "--ml-threshold", str(ml_threshold)]
    # Also not a scope flag: it changes which cell of an already-swept grid is
    # ELIGIBLE to win, never which pairs are swept.
    if allow_isolated_spikes:
        cmd += ["--allow-isolated-spikes"]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def stage3_cmd(strat: str, tf: str, is_start: str, is_end: str,
               holdout_start: str = HOLDOUT_START,
               holdout_end: str | None = None,
               promote: bool = True,
               out_dir: str | None = None,
               ml_threshold: float = ML_THRESHOLD_DEFAULT) -> list[str]:
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
                   "--holdout-start", holdout_start,
                   "--ml-threshold", str(ml_threshold)]
    if holdout_end:
        cmd += ["--holdout-end", holdout_end]
    if not promote:
        cmd += ["--no-promote"]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def stage4_cmd(strat: str, tf: str, start: str, end: str,
               out_dir: str | None = None,
               ml_threshold: float = ML_THRESHOLD_DEFAULT,
               symbols: str | None = None) -> list[str]:
    """
    Stage 4: the lifecycle run for one timeframe.

    The window spans the holdout on purpose - this is not a certification and
    the module records `is_certification: false`. `--end` is passed rather
    than defaulted for the reason given in the module docstring.

    `symbols` NAMES THE CONTRACTS, and since 2026-09-08 the caller passes
    Stage 3's CERTIFIED pairs rather than leaving it off.

    Every other stage omits `--symbols` so the scope is INHERITED from the
    previous stage's artifacts, and Stage 4 inheriting Stage 2's the same way
    was consistent rather than accidental. But Stage 4 runs AFTER Stage 3, and
    its output is what `promote.py` locks into the promoted package - so the
    pairs it needs are the ones that can still BE promoted. On
    t3_braid_scalp_20260823 that difference was 31 sixteen-year lifecycle runs
    for 12 promotable pairs: 19 full tear sheets, trade logs and per-year cost
    tables for configurations Gate R had already refused.

    `--all-optimized` restores the wider sweep, for a diagnostic run where the
    lifecycle curve of a pair that FAILED is the thing being read.
    """
    cmd = _py() + [str(STAGE_SCRIPTS[4]), "--strat", strat, "--tf", tf,
                   "--start", start, "--end", end,
                   # --ml IS REQUIRED HERE, and its absence was a silent
                   # pipeline break. verify_full's --ml is store_true and this
                   # command never passed it, so Stage 4 wrote
                   # `version_b: null` into dual_metrics_<SYM>.json - and
                   # Stage 5, handed a Version B that Stage 3 had certified,
                   # looked for the `version_b` block, did not find it, and
                   # died in load_metrics with "carries no `version_b` block".
                   # On sma_momentum_crossover_20260818 that lost all NINE
                   # certified Version B packages: promoted 17, failed 9.
                   "--ml",
                   # At the SAME bar Stage 1 screened and Stage 3 certified
                   # at. Stage 4's numbers are what promote.py locks into the
                   # promoted module, so a filter measured here at a different
                   # threshold would deploy metrics for a Version B nobody
                   # certified.
                   "--ml-threshold", str(ml_threshold)]
    if symbols:
        cmd += ["--symbols", symbols]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    return cmd


def stage45_cmd(strat: str, tf: str, start: str, end: str,
                out_dir: str | None = None,
                min_trades: int | None = None,
                block_worst_always: bool = False,
                ml_threshold: float = ML_THRESHOLD_DEFAULT) -> list[str]:
    """
    Stage 4.5: the day-of-week gate for one timeframe.

    THE SAME WINDOW AS STAGE 4, and passed rather than defaulted for that
    reason. The two stages describe the same bars, so the weekday table here
    and the one Stage 4's tear sheet prints are statements about one run; a
    different window would produce two day-of-week tables of the same strategy
    that disagree, with nothing on either saying why.

    `--ml` IS passed, for exactly the reason `stage4_cmd` passes it. Version B
    is Version A's entries minus the ones a classifier expected to lose, so
    its trade list is a SUBSET and its weekday table is a different table; a
    pair Stage 3 certified as B and this stage profiled only as A would be
    promoted carrying a weekday measured on a strategy nobody deployed. Both
    versions get their own verdict inside the one per-pair file, and
    `promote.py` picks the one matching the package it is writing.

    `--ml-threshold` travels with it: Stage 3 certifies at that bar and
    `promote.py` embeds it, so profiling at a different one would describe a
    Version B nobody certified.

    ONE TIMEFRAME per invocation, as with Stages 3 and 4: the blocked weekday
    is a fact about one (symbol, timeframe) pair, and the per-pair file is
    named for both.
    """
    cmd = _py() + [str(STAGE_SCRIPTS[STAGE45]), "--strat", strat, "--tf", tf,
                   "--start", start, "--end", end,
                   "--ml", "--ml-threshold", str(ml_threshold)]
    if min_trades is not None:
        cmd += ["--min-trades", str(int(min_trades))]
    if block_worst_always:
        cmd += ["--block-worst-always"]
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


def stage4_metrics_exist(strat: str, out_dir: str | None = None) -> bool:
    """
    Whether THIS campaign produced a Stage 4 lifecycle snapshot to announce.

    The Stage 4 card is built from the `dual_metrics_<SYMBOL>.json` files in a
    `verify_<stamp>/` directory, and refuses a directory holding none. On the
    `--promote-only` path Stage 4 is not run, so the card is posted only when
    an earlier run left snapshots behind - otherwise the reporter would exit
    non-zero and print a warning about a stage this invocation never ran.
    """
    home = pipeline_dir(strat, out_dir)
    try:
        stamps = sorted((d for d in home.glob("verify_*") if d.is_dir()),
                        key=lambda d: d.name, reverse=True)
    except OSError:
        return False
    return any(any(d.glob("dual_metrics_*.json")) for d in stamps)


def stage5_card_targets(strat: str) -> list[str]:
    """
    The id(s) the Stage 5 card must be posted under, module name included.

    Stages 1-4 measure a MODULE and their cards are headed by its name. Stage 5
    does not: `promote.py` writes one package per certified pair
    (`<module>_<SYMBOL>_<TF>_V<A|B>`) and a promotion card names ONE contract.
    So the module name this orchestrator passes to every other card resolves
    to no package at all here, and the card refused for a missing `--symbol` -
    a message that sends an operator looking for a contract when what was
    actually wrong is that a module is not a promotion.

    Returns the packages when there are any, and NO TARGET when the enumeration
    succeeded and found none. `promotion_packages` already returns the module
    directory itself when a strategy was promoted under its own module name
    (the older layout), so an empty result there does not mean "the old layout"
    - it means NOTHING WAS PROMOTED. Falling back to the module name in that
    case posted a promotion card for a promotion that does not exist, which
    could only ever fail, and it failed with "the contract could not be
    resolved from portfolios.json" - a message that sends an operator to a
    config file to debug a run whose real outcome was simply that no
    configuration cleared its gates. A run that certified nothing has nothing
    to announce.

    The enumeration RAISING is a different case and still falls back, because
    then we genuinely do not know whether anything was promoted.
    """
    try:
        from backtest.discord_reporter import promotion_packages
        found = [p.name for p in promotion_packages(strat)]
    except Exception as exc:                                      # noqa: BLE001
        # A card is an announcement; failing to enumerate packages must not
        # end a run that has already promoted. Fall back to what was asked
        # for and let the reporter print its own refusal.
        print(f"  NOTE  could not enumerate promoted packages for {strat} "
              f"({exc}); posting one card under the name as given.",
              file=sys.stderr, flush=True)
        return [strat]
    return found


def post_card(strat: str, stage: int, *, out_dir: str | None = None,
              dry_run: bool = False) -> int:
    """
    Post one stage's card, and never let a webhook outage fail the run.

    `check=False` throughout: a card is an ANNOUNCEMENT and the stage's
    artifacts are the evidence. A completed certification must not be
    discarded because Discord was unreachable.

    Stage 5 posts ONE CARD PER PROMOTED PACKAGE - see `stage5_card_targets`.
    The worst return code wins, so a run whose second card failed still says
    so rather than being reported by whichever went last.
    """
    targets = stage5_card_targets(strat) if stage == 5 else [strat]
    if stage == 5 and not targets:
        print(f"  NOTE  no promoted package for {strat}; there is no Stage 5 "
              f"card to post.", flush=True)
        return 0
    worst = 0
    for target in targets:
        label = f"DISCORD · Stage {stage} card"
        if len(targets) > 1:
            label += f" · {target}"
        rc = run_step(discord_cmd(target, stage, out_dir=out_dir),
                      label, dry_run=dry_run, check=False)
        if rc:
            print(f"  WARNING  Stage {stage} Discord card for {target} failed "
                  f"(exit {rc}); the stage itself is unaffected.",
                  file=sys.stderr, flush=True)
            worst = worst or rc
    return worst


def dow_gate_file(strat: str, symbol: str, timeframe: str,
                  out_dir: str | None = None) -> Path | None:
    """
    This PAIR's Stage 4.5 verdict, or None when the stage did not cover it.

    The SUFFIXED name only, and there is no unsuffixed fallback - the same
    rule `auto_promote` follows for gate audits, for the same reason. A
    campaign certifies one strategy at several timeframes and each has its own
    weakest session; a file that dropped the timeframe would hold whichever
    ran last while every promotion cited it, and each meta.json would carry a
    plausible weekday with two of them wrong.

    None rather than a guessed path, so `promote_cmd` omits the flag and
    `promote.py` records `NOT EVALUATED` - which is what actually happened.
    """
    path = (pipeline_dir(strat, out_dir)
            / DOW_GATE_FILE.format(symbol=str(symbol).upper(),
                                   tf=str(timeframe).lower()))
    return path if path.exists() else None


def _generated_utc(path: str | Path | None) -> datetime | None:
    """`generated_utc` off a pipeline artifact, or None if it is unreadable."""
    if not path:
        return None
    try:
        stamp = json.loads(Path(path).read_text()).get("generated_utc")
        return datetime.fromisoformat(str(stamp)) if stamp else None
    except (OSError, ValueError, TypeError):
        return None


def dow_gate_staleness(dow_gate: str | Path | None,
                       audit_file: str | Path | None) -> str | None:
    """
    Why this pair's Stage 4.5 verdict should not be trusted, or None.

    THE FAILURE THIS CATCHES, observed on
    `sma_momentum_crossover_20260818_YM_30m_VB` on 2026-09-09. `dow_gate_file`
    asks only whether the artifact EXISTS, and an artifact from an earlier
    Stage 4.5 run exists just as convincingly as a current one. That package
    was promoted at 08:11:19 against a gate audit generated at 08:11:19, and
    the pair's Stage 4.5 verdict was not rewritten until 08:33:00 - so the
    weekday stamped into its meta.json came from a PREVIOUS run. It blocked
    Wednesday; the current verdict blocks nothing, and the strategy sat
    stood-down on a session no live evidence supported, with every log line
    reading correctly.

    The test is ordering, not age: a verdict generated BEFORE the
    certification it is being attached to was measured on a configuration that
    is no longer the one being promoted. Equal timestamps pass - Stage 4.5 and
    Stage 3 inside one `run_pipeline` invocation can share a second.

    A REASON, NOT A REFUSAL. Stage 4.5 prunes nothing and its weekday is an
    instruction for the live supervisor rather than evidence about an edge, so
    a stale one must not discard a certification that already cleared Gate R.
    The caller records this and promotes anyway, and
    `scripts/check_strategy_days.py` is the standing check that catches what
    slips through.

    Unreadable timestamps return None. Neither file having a `generated_utc`
    is the pre-2026 artifact shape, and inventing staleness from a missing
    field would warn on every older campaign.
    """
    dow_at = _generated_utc(dow_gate)
    audit_at = _generated_utc(audit_file)
    if dow_at is None or audit_at is None:
        return None
    if dow_at >= audit_at:
        return None
    return (f"the stage 4.5 verdict was generated {dow_at.isoformat()}, "
            f"BEFORE the gate audit it is being attached to "
            f"({audit_at.isoformat()}) - it describes an earlier run's "
            f"configuration. Re-run stage 4.5 for this pair and re-promote, "
            f"or the package carries a weekday nobody measured for it.")


def promote_cmd(strat: str, version: str, source: str | Path,
                audit_file: str | Path, symbol: str,
                timeframe: str,
                metrics: str | Path | None = None,
                dow_gate: str | Path | None = None) -> list[str]:
    """
    Stage 5 for ONE certified configuration.

    `--require-certification` is always passed and `--force` never is: this
    path may only ratify a verdict Stage 3 already reached. A promotion that
    could override a gate is a decision, and decisions are the operator's.

    `--metrics` is passed when `resolve_metrics` found this PAIR's Stage 4
    snapshot, and omitted when it did not. Omitted, `promote.py` records
    `metrics_status: "NOT RECORDED"`, which is the honest reading - the
    alternative is another timeframe's lifecycle numbers locked into this
    pair's meta.json under its certification.
    """
    cmd = _py() + [str(PROMOTE_SCRIPT), "--strat", strat,
                   "--version", version, "--source", str(source),
                   "--audit-file", str(audit_file), "--symbol", symbol,
                   "--timeframe", timeframe, "--require-certification"]
    if metrics:
        cmd += ["--metrics", str(metrics)]
    # PASSED EXPLICITLY rather than left to promote.py's own lookup, even
    # though the two resolve to the same file. `--out-dir` moves the pipeline
    # directory and the promotion command is the artifact an operator re-runs
    # by hand; a flag naming the file is reproducible, and a lookup that
    # silently found nothing under a relocated root would record NOT EVALUATED
    # on a pair that was profiled.
    if dow_gate:
        cmd += ["--dow-gate", str(dow_gate)]
    return cmd


# ---------------------------------------------------------------------------
# handoff readers
# ---------------------------------------------------------------------------

def version_b_pairs(strat: str, out_dir: str | None = None
                    ) -> list[tuple[str, str]]:
    """
    The `(symbol, timeframe)` pairs `surviving_assets.json` says cleared Stage
    1 on VERSION B - the ML-filtered pipeline.

    Read here so the orchestrator can PRINT the plan before Stage 2 runs and
    CHECK it after Stage 3, but never to pass a global `--ml`: that flag turns
    the filter on for every pair, and a Version A survivor certified with a
    classifier over it is the mirror image of the bug being fixed. Both stages
    resolve the version per pair from the handoff they already read; this is
    the orchestrator's independent second reading of the same file, which is
    what makes a silent regression in either of them visible.
    """
    from backtest.pipeline import SURVIVORS_FILE, stage1_pairs

    path = pipeline_dir(strat, out_dir) / SURVIVORS_FILE
    if not path.exists():
        return []
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [(str(p["symbol"]), str(p["tf"])) for p in stage1_pairs(blob)
            if str(p.get("version") or "").strip().upper() == "B"]


def check_version_b_certified(strat: str, expected: list[tuple[str, str]],
                              out_dir: str | None = None) -> list[str]:
    """
    Every Stage 1 Version B survivor that Stage 3 did NOT audit as Version B.

    Returned as messages, printed by the caller, and never fatal - the run is
    already over by the time this can be checked and the certifications on
    disk are real. It exists because the failure it detects is silent in every
    other output: a B survivor certified as A only produces a complete,
    plausible Stage 3 summary in which every gate is filled in and the version
    column reads `A`, and nothing else in the pipeline compares that column
    against the handoff that designated it.
    """
    summary = _read_summary(strat, STAGE3_SUMMARY_FILE, 3, out_dir)
    rows = (summary or {}).get("results") or []
    audited = {(str(r.get("symbol")), str(r.get("timeframe")),
                str(r.get("version") or "").upper()) for r in rows}
    missing = []
    for symbol, tf in expected:
        if (symbol, tf, "B") in audited:
            continue
        if not any(r for r in rows if str(r.get("symbol")) == symbol
                   and str(r.get("timeframe")) == tf):
            missing.append(f"{symbol} {tf}: cleared Stage 1 on Version B and "
                           f"never reached Stage 3")
            continue
        missing.append(
            f"{symbol} {tf}: cleared Stage 1 on Version B but Stage 3 audited "
            f"only {sorted(v for s, t, v in audited if (s, t) == (symbol, tf))}"
            f". The certification does not describe the version that earned "
            f"the survivorship.")
    return missing


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


def _snapshot_timeframe(path: Path) -> str | None:
    """The timeframe a `dual_metrics_<SYMBOL>.json` was produced at, or None."""
    try:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    meta = blob.get("meta") if isinstance(blob, dict) else None
    tf = (meta or {}).get("timeframe")
    return str(tf).strip().lower() if tf else None


def resolve_metrics(strat: str, symbol: str, timeframe: str,
                    out_dir: str | None = None) -> tuple[Path | None, str]:
    """
    The Stage 4 lifecycle snapshot for ONE certified pair, and where it is.

    Returns `(path, basis)`; `path` is None when no snapshot for this pair
    could be found, which `promote.py` already reports as
    `metrics_status: "NOT RECORDED"` rather than inventing one.

    **The timeframe must MATCH, and this is the whole reason the function
    exists rather than a glob.** Stage 4 writes one `verify_<stamp>/` per
    invocation and a campaign runs it once per timeframe, so the directory
    names are stamps and carry no timeframe at all: on this repository the
    NEWEST `verify_*` for `t3_braid_scalp_20260823` is a 1h run, while 15m and
    30m sit in older ones. Taking "the latest" would hand the 15m promotion a
    1h lifecycle snapshot - and since `promote.snapshot_params` layers that
    snapshot's parameters over the module's defaults, meta.json would then
    record 1h's stop and target for a 15m promotion, beside 1h's Sharpe, under
    the 15m certification. Every field individually true, the whole thing
    describing a run nobody made.

    So the timeframe is read out of each candidate's own `meta.timeframe` and
    the first NEWEST match wins. A snapshot that records no timeframe is
    accepted only when nothing else matched, and says so - it cannot be
    checked, which is a weaker claim than a checked match and a stronger one
    than nothing.

    Three places are searched, narrowing outward:

      1. `<pipeline>/verify_<stamp>/dual_metrics_<SYM>.json` - Stage 4's own
         output, newest stamp first. Sorted by NAME rather than mtime, the way
         the Stage 4 card resolves its default directory: mtime moves when a
         directory is copied off the NFS mount.
      2. `<pipeline>/dual_metrics_<SYM>.json` - the artifact root fallback.
      3. `<BT_ARTIFACTS>/<strat>_<stamp>/dual_metrics_<SYM>.json` - a `bt-run`
         batch, newest stamp first. It is last because a batch run is not a
         stage: its window is whatever was typed.
    """
    tf = str(timeframe).strip().lower()
    home = pipeline_dir(strat, out_dir)
    name = f"dual_metrics_{symbol}.json"

    candidates: list[tuple[str, Path]] = []
    for directory in sorted(home.glob("verify_*"), reverse=True):
        if (directory / name).is_file():
            candidates.append((f"{directory.name}/{name}", directory / name))
    if (home / name).is_file():
        candidates.append((name, home / name))
    try:
        root = artifacts_root()
    except Exception:                                             # noqa: BLE001
        root = None
    if root is not None:
        for directory in sorted(root.glob(f"{strat}_*"), reverse=True):
            if (directory / name).is_file():
                candidates.append((f"{directory.name}/{name}",
                                   directory / name))

    if not candidates:
        return None, f"no {name} for {strat} under {home}"

    unchecked: tuple[str, Path] | None = None
    seen: list[str] = []
    for label, path in candidates:
        found = _snapshot_timeframe(path)
        if found == tf:
            return path, f"{label} · meta.timeframe {found}"
        if found is None and unchecked is None:
            unchecked = (label, path)
        elif found:
            seen.append(found)

    if unchecked is not None:
        label, path = unchecked
        return path, (f"{label} · records NO timeframe, so it could not be "
                      f"checked against {tf}")
    return None, (f"no {name} at {tf} - the {len(candidates)} on disk are "
                  f"{', '.join(sorted(set(seen))) or 'unreadable'}. A snapshot "
                  f"from another timeframe is NOT used: it would lock that "
                  f"run's metrics and parameters into this pair's meta.json.")


def certified_symbols(strat: str, tf: str,
                      out_dir: str | None = None) -> list[str]:
    """
    The contracts Stage 3 certified AT THIS TIMEFRAME, for Stage 4's scope.

    Read from Stage 3's summary through `certified_rows`, so the list is the
    audit's own `certified` flag and never a second opinion about what passed.

    Returns [] when the summary is missing or nothing certified at `tf`, and
    the caller must read that as "run NOTHING" rather than "run everything".
    An empty scope list falling through to an unscoped command is precisely
    the failure this narrowing exists to close: it would profile the whole
    Stage 2 universe while reporting that it had narrowed.
    """
    rows = certified_rows(_read_summary(strat, STAGE3_SUMMARY_FILE, 3, out_dir))
    return sorted({str(r["symbol"]) for r in rows
                   if r.get("symbol") and str(r.get("timeframe")) == str(tf)})


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
                          commit: str | None,
                          deferred: str | None = None) -> Path | None:
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
        # WHY the promotion loop promoted nothing, when it promoted nothing.
        # `promoted: 0` alone is ambiguous - it reads the same for a batch the
        # fan-out bar deferred, a batch where every promotion failed, and a
        # Stage 3 that certified nothing. The card and any unattended reader
        # need those apart.
        "deferred": deferred,
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
    # `--timeframes` is the same option, not a second one: argparse folds both
    # spellings onto one dest, so they cannot drift the way two constants can.
    # It exists because it is the name people reach for and type.
    #
    # The default is CORE_DAY_TRADING, NOT a single 15m and NOT the full
    # ladder. Omitting --tf used to mean 15m alone, which quietly made the
    # unflagged run a one-timeframe run; ALL_DAY_TRADING remains one word away
    # for anyone who wants 1m/2m/3m back.
    p.add_argument("--tf", "--timeframes", dest="tf",
                   default=",".join(DEFAULT_TFS),
                   help=f"Timeframe(s) for Stage 1, comma-separated, or a "
                        f"group name ({', '.join(sorted(TF_GROUPS))}). "
                        f"Default {','.join(DEFAULT_TFS)}. Stages 3 and 4 run "
                        f"once per timeframe Stage 2 optimised.")
    p.add_argument("--start", default=CHARTER_IS_START,
                   help=f"In-sample start (default {CHARTER_IS_START})")
    p.add_argument("--end", default=CHARTER_IS_END,
                   help=f"In-sample end (default {CHARTER_IS_END}). Refused if "
                        f"it reaches {HOLDOUT_START}.")
    p.add_argument("--ml-threshold", type=float,
                   default=ML_THRESHOLD_DEFAULT,
                   help=(f"Version B: P(win) at or above which an entry is "
                         f"kept (default {ML_THRESHOLD_DEFAULT}). Forwarded "
                         f"to every stage, so one run cannot screen, certify "
                         f"and measure at three different bars"))
    p.add_argument("--dow-min-trades", type=int, default=None, metavar="N",
                   help="Stage 4.5: a weekday must place at least this many "
                        "trades before it can be condemned (default 20, "
                        "backtest/dow_gate.py). A day below the floor is "
                        "listed and never ranked — cutting a session on a "
                        "thinner sample is how a day-of-week filter "
                        "manufactures an in-sample Sharpe.")
    p.add_argument("--dow-block-worst-always", action="store_true",
                   help="Stage 4.5: block the worst weekday even when its "
                        "expectancy is POSITIVE. Off by default; being fifth "
                        "of five is not evidence against a session.")
    p.add_argument("--all-optimized", dest="all_optimized",
                   action="store_true", default=False,
                   help="Stage 4: profile every pair Stage 2 optimised, not "
                        "only the ones Stage 3 certified. Off by default "
                        "since 2026-09-08 — Stage 4 runs after Stage 3 and "
                        "its output is what promote.py locks into a package, "
                        "so the pairs worth sixteen years of tear sheets are "
                        "the promotable ones. Turn it on for a diagnostic "
                        "run, where the lifecycle curve of a pair that FAILED "
                        "is the thing being read.")
    p.add_argument("--allow-isolated-spikes", dest="allow_isolated_spikes",
                   action="store_true", default=False,
                   help="Forwarded to Stage 2: let a cell whose grid "
                        "neighbours keep under half its Sharpe still win its "
                        "sweep. NOTE this re-rolls WHICH parameters Stage 2 "
                        "locks, so Stage 3 then certifies a different set - "
                        "it is not a way to reproduce an earlier run's "
                        "verdicts with more candidates. Does not touch the "
                        "ruin bar.")
    p.add_argument("--report-discord", action="store_true",
                   help="Post the Discord card after Stages 1, 2 and 3")
    p.add_argument("--promote-max", type=int, default=DEFAULT_PROMOTE_MAX,
                   metavar="N",
                   help=f"Refuse an unattended promotion of more than N "
                        f"configurations (default {DEFAULT_PROMOTE_MAX}; 0 or "
                        f"less removes the bar). Applies to --auto-promote "
                        f"and --promote-only alike. One command over the "
                        f"23-contract universe at four timeframes can certify "
                        f"dozens of pairs, and each promotion is a package on "
                        f"disk, a routing-table row and its own commit that "
                        f"nobody read a card for.")
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
        f"  --symbols/--tf/--start/--end are not used by this path.\n"
        f"  STAGE 4.5 IS NOT RUN EITHER. Each promotion takes whatever\n"
        f"  dow_gate_<SYM>_<TF>.json is already on disk - a verdict older\n"
        f"  than the certification is stamped in as though it were current,\n"
        f"  so any pair reported STALE below needs stage 4.5 re-run and the\n"
        f"  package re-promoted."))

    outcome = auto_promote(strat, out_dir=out_dir, dry_run=dry,
                           promote_max=args.promote_max)
    rc = int(outcome["returncode"])
    rows, promotions = outcome["rows"], outcome["promotions"]

    if args.report_discord:
        # Strict numerical order, the same sequence the full pipeline posts in:
        # Stage 3's certification, then Stage 4's lifecycle, then Stage 5's
        # promotion. All three go out AFTER the promotions have run, because
        # this path exists to promote what is already on the handoff - so the
        # Stage 3 card reads AUTOMATICALLY PROMOTED with the commit rather than
        # printing a command that has already run, and Stage 5 reports the
        # outcome of the run the operator is watching.
        #
        # Stage 4 is NOT run by this path. Its card is posted only when an
        # earlier run left lifecycle snapshots behind; with none, the reporter
        # would refuse the empty directory and warn about a stage this
        # invocation never ran. Under --dry-run nothing is read, so the card is
        # traced unconditionally to keep the dry run a faithful rehearsal of
        # the order.
        post_card(strat, 3, out_dir=out_dir, dry_run=dry)
        if dry or stage4_metrics_exist(strat, out_dir):
            post_card(strat, 4, out_dir=out_dir, dry_run=dry)
        else:
            print("  Stage 4 card skipped: this campaign has no "
                  "verify_<stamp>/dual_metrics_*.json to announce.",
                  flush=True)
        post_card(strat, 5, out_dir=out_dir, dry_run=dry)

    if rows:
        print("\n" + promotion_summary_table(rows, promotions))
    print(_banner("PROMOTE ONLY COMPLETE"
                  + (f" · PROMOTION DEFERRED\n  {outcome.get('deferred')}"
                     if outcome.get("deferred") else "")))
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
        f"  stage 4.5   day-of-week gate, "
        f"{'block worst always' if args.dow_block_worst_always else 'block only a negative expectancy'}"
        f" (floor {args.dow_min_trades or 20} trades)\n"
        f"  NOTE  the stages are designed not to chain; this run reads no\n"
        f"        evidence between them."))

    def discord(stage: int) -> None:
        if not args.report_discord:
            return
        post_card(strat, stage, out_dir=out_dir, dry_run=dry)

    try:
        run_step(stage1_cmd(strat, args.symbols, tfs, args.start, args.end,
                            out_dir, args.ml_threshold),
                 "STAGE 1 · baseline.py · regime screen", dry_run=dry)
        discord(1)

        run_step(stage2_cmd(strat, args.start, args.end, out_dir,
                            args.ml_threshold,
                            allow_isolated_spikes=getattr(
                                args, "allow_isolated_spikes", False)),
                 "STAGE 2 · scan.py · parameter sweep over Stage 1's exact pairs",
                 dry_run=dry)
        discord(2)
    except subprocess.CalledProcessError as e:
        print(f"\nABORTED: {' '.join(str(c) for c in e.cmd)} exited "
              f"{e.returncode}. Later stages read this one's handoff, so "
              f"continuing would report an earlier run's artifacts as this "
              f"run's result.", file=sys.stderr)
        return int(e.returncode or 1)

    # WHICH PAIRS CARRY A CLASSIFIER, read straight off Stage 1's handoff.
    # Printed rather than turned into a flag: Stages 2 and 3 each resolve the
    # version per pair from the handoff, and a global `--ml` here would run the
    # filter over the Version A survivors too.
    b_pairs = [] if dry else version_b_pairs(strat, out_dir)
    if b_pairs:
        print(f"\n  VERSION B · {len(b_pairs)} pair(s) cleared Stage 1 on the "
              f"ML-filtered version:\n    "
              + ", ".join(f"{s}·{t}" for s, t in b_pairs)
              + "\n    Stage 2 confirms the filter on each winner and Stage 3 "
                "certifies Version B for them,\n    resolved per pair from "
                "surviving_assets.json. No global --ml is passed.", flush=True)

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
            # `--no-promote`, ALWAYS, AND ONLY ON THIS PATH.
            #
            # Stage 3 stages a package for everything it certifies
            # (`audit_gates.seal_and_promote`, Charter clause 6), and run
            # standalone that is correct: nothing else is going to write one,
            # and the directory is the record that a version was certified.
            #
            # Under THIS orchestrator it is not, because Stage 4.5 has not run
            # yet and cannot have. `promote.load_dow_gate` looks up
            # `dow_gate_<SYM>_<TF>.json` by convention when no path is passed,
            # so Stage 3's staging is not merely missing the flag - the file it
            # would read does not exist for another ten minutes. Every package
            # Stage 3 stages here is therefore stamped
            # `day_of_week_gate.status: NOT EVALUATED` BY CONSTRUCTION, with a
            # reason reading "Stage 4.5 was not run for this pair" that is true
            # only of the instant it was written.
            #
            # That is invisible when Stage 5 runs, because promote.py rewrites
            # meta.json wholesale. It is NOT invisible when Stage 5 does not -
            # a run without --auto-promote, or one where the --promote-max bar
            # refused the batch - and what is left on disk is a package that
            # looks promoted, reads as though the week was profiled and cleared
            # to a hurried eye, and stands nothing down. On
            # dbb_momentum_breakout_20260909 that was 15 packages written
            # 18:02-18:10 against a gate written 18:11-18:14, and CL 30m alone
            # blocked Friday on Version A and Monday on Version B - two
            # measured negative-expectancy sessions both packages would have
            # traded.
            #
            # So the invariant this orchestrator enforces is: A PACKAGE
            # APPEARS ON DISK ONLY AFTER STAGE 4.5 HAS RETURNED A VERDICT FOR
            # ITS PAIR. Stage 3 still certifies, and `certified: true` on the
            # summary is set from the gate verdict and not from the staging
            # (`audit_gates` line ~2490), so `auto_promote`'s winners are
            # unchanged. What is given up is the `seal` block, which only
            # Stage 3 writes - and Stage 5 overwrites it anyway whenever it
            # runs, which is why 110 of the 187 packages on disk have none.
            run_step(stage3_cmd(strat, tf, args.start, args.end,
                                out_dir=out_dir, promote=False,
                                ml_threshold=args.ml_threshold),
                     f"STAGE 3 · audit_gates.py · certify {tf} on the holdout",
                     dry_run=dry)
        # The card is posted AFTER Stage 3 has run at every timeframe, because
        # Stage 3 merges each invocation into one summary and the card reads
        # that summary: posted per timeframe it would announce the same
        # campaign several times, each edition missing the timeframes that had
        # not run yet.
        #
        # It is posted HERE and not after the promotion, even under
        # --auto-promote. Cards go out in strict numerical order, so Stage 3's
        # certification is announced before Stage 4's lifecycle and Stage 5's
        # promotion rather than after them. The cost is that under
        # --auto-promote this card is built before `auto_promotion` is on the
        # handoff, so its promotion section reads READY FOR PROMOTION / STAGED
        # and prints the --promote-only command that this same run is about to
        # execute. That is the correct division: Stage 3 announces what was
        # CERTIFIED, and the promotion outcome is Stage 5's card to carry.
        discord(3)

        if b_pairs:
            gaps = check_version_b_certified(strat, b_pairs, out_dir)
            if gaps:
                print("\n  ! VERSION B GAP — a pair that only the ML-filtered "
                      "version carried was\n    certified without it:",
                      file=sys.stderr)
                for gap in gaps:
                    print(f"      {gap}", file=sys.stderr)
            else:
                print(f"\n  Version B preserved end to end for all "
                      f"{len(b_pairs)} pair(s).", flush=True)

        for tf in certify_tfs:
            # Stage 3's certified pairs, not Stage 2's optimised ones - see
            # `stage4_cmd`. In a --dry-run there is no Stage 3 summary to
            # read, so the scope is left off and the printed command is the
            # unscoped one; that is the honest thing to print, because a
            # scope resolved from a handoff that does not exist would be a
            # different command from the one a real run would issue.
            scope = None if (dry or args.all_optimized) else ",".join(
                certified_symbols(strat, tf, out_dir))
            if not dry and not args.all_optimized and not scope:
                print(f"  SKIPPED  Stage 4 at {tf}: Stage 3 certified nothing "
                      f"here, so there is no promotable pair to profile. "
                      f"--all-optimized runs the lifecycle anyway.",
                      flush=True)
                continue
            run_step(stage4_cmd(strat, tf, args.start, _today(), out_dir,
                                args.ml_threshold, symbols=scope),
                     f"STAGE 4 · verify_full.py · lifecycle {tf}"
                     + (f" · {scope}" if scope else " · ALL optimised pairs"),
                     dry_run=dry)
        # Card 4 after every timeframe, for the same reason Card 3 waits: it
        # reads ONE run's artifacts directory and defaults to the newest.
        discord(4)

        # STAGE 4.5, BETWEEN THE LIFECYCLE RUN AND THE PROMOTION.
        #
        # Here rather than earlier because it profiles the SAME window Stage 4
        # ran, and before Stage 5 because Stage 5 is what writes the blocked
        # weekday into the promoted meta.json - run after it, the verdict
        # would land in the pipeline directory one promotion too late and the
        # live loop would trade the session until somebody re-promoted.
        #
        # `check=False`, unlike Stages 1-4. Those are a chain: a stage that
        # failed has not written the handoff the next one reads. This one is
        # not - it produces an INSTRUCTION for the live supervisor, and
        # nothing downstream needs it to exist. A day-of-week profiler that
        # raised must not discard certifications that already cleared Gate R;
        # the promotion proceeds and every affected package records
        # `day_of_week_gate.status: NOT EVALUATED`, which is exactly what
        # happened.
        for tf in certify_tfs:
            rc45 = run_step(
                stage45_cmd(strat, tf, args.start, _today(), out_dir,
                            min_trades=args.dow_min_trades,
                            block_worst_always=args.dow_block_worst_always,
                            ml_threshold=args.ml_threshold),
                f"STAGE 4.5 · dow_gate.py · day-of-week gate {tf}",
                dry_run=dry, check=False)
            if rc45:
                print(f"  WARNING  Stage 4.5 exited {rc45} at {tf}. Packages "
                      f"promoted below will record NO blocked weekday for "
                      f"the pairs it did not profile; nothing else is "
                      f"affected.", file=sys.stderr, flush=True)
    except subprocess.CalledProcessError as e:
        print(f"\nABORTED: {' '.join(str(c) for c in e.cmd)} exited "
              f"{e.returncode}.", file=sys.stderr)
        return int(e.returncode or 1)

    rc = 0
    deferred: str | None = None
    promotions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if args.auto_promote:
        outcome = auto_promote(strat, out_dir=out_dir, dry_run=dry,
                               promote_max=args.promote_max)
        rc = int(outcome["returncode"])
        promotions, rows = outcome["promotions"], outcome["rows"]
        deferred = outcome.get("deferred")
        # Card 5 last, and only now: the promotion outcome is on the handoff,
        # so the card reports what was actually promoted and to which commit
        # rather than a promotion that had not happened when it was built.
        discord(5)
    elif not dry:
        rows = ((_read_summary(strat, STAGE3_SUMMARY_FILE, 3, out_dir) or {})
                .get("results") or [])

    if rows:
        print("\n" + promotion_summary_table(rows, promotions))

    # The banner says which of the two clean endings this was. A run whose
    # promotion was deferred by the fan-out bar exits 0 like any other
    # success, so the exit code alone cannot say that one step is outstanding.
    print(_banner("PIPELINE COMPLETE"
                  + (f" · PROMOTION DEFERRED\n  {deferred}"
                     if deferred else "")))
    return rc


def render_unrouted(groups: dict[str, list[str]], width: int = 78) -> str:
    """
    The end-of-run tally of promoted packages no portfolio routes.

    WHY THIS PRINTS AT ALL. `register_portfolio` refuses, correctly, when no
    incubator basket carries a certified contract, and says so on the console
    at the moment it happens. In a `--auto-promote` run that is one `!` line
    among up to 168 configurations, and it scrolls. Seventy-five accumulated
    that way before anybody counted them, and the only reason they surfaced
    was somebody looking for two specific packages in a CSV and not finding
    them.

    Grouped by CONTRACT rather than listed by package, because the remedy is
    per contract: widening one basket routes every package certified on that
    symbol, and a flat list of eighty ids says nothing about how many
    decisions that is.
    """
    if not groups:
        return ""
    total = sum(len(v) for v in groups.values())
    lines = ["", "-" * width,
             f"  {total} promoted package(s) are NOT routed by any portfolio",
             ""]
    # Biggest first: the contract worth deciding about is the one carrying the
    # most certified work.
    for symbol, ids in sorted(groups.items(),
                              key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"    {symbol:<6} {len(ids):>3}")
    lines += [
        "",
        "  These cleared Gate R and were written to the incubator, and nothing",
        "  armed them. For most, no basket carries the contract - that is the",
        "  refusal working, not a failure: a strategy routed to an account",
        "  that cannot trade its symbol is refused on every asset, forever.",
        "  Rows marked (carried; re-promote to route) are the other case -",
        "  the basket carries them NOW but did not when they were promoted,",
        "  and routing is decided at promotion time and never revisited.",
        "",
        "  To arm them, add the contract to a basket in",
        "  config/portfolios.json (with asset_metadata matching",
        "  backtest/specs.py) and re-promote. To review them:",
        "      python3 tools/portfolio_inventory.py --include-unallocated",
        "-" * width]
    return "\n".join(lines)


#: The unattended fan-out bar. RAISED FROM 5 TO 50 on 2026-09-10.
#:
#: 5 was set on 2026-09-07 after an audit of approved_incubator/ found 160
#: packages on disk against the 52 the routing table actually routes, and the
#: number chosen was small enough to stop a runaway. It was also below the
#: legitimate yield of ONE charter-compliant campaign, which is the case it
#: had to allow: `dbb_momentum_breakout_20260909` certified 15 configurations
#: across 3 timeframes and 9 contracts in a single sweep, and every one of
#: them cleared Gate R on the holdout. The bar refused the batch, promoted
#: nothing, and left Stages 1-4.5 complete on disk with no package to show for
#: 30 minutes of compute - a bar that fires on the ordinary case is not a
#: guard, it is an outage.
#:
#: 50 sits above one campaign's realistic yield and below the runaway the
#: audit actually found (92 certifiable pairs, 472 commits over 14 days), so
#: the bar still binds exactly where it was designed to.
DEFAULT_PROMOTE_MAX = 50


def auto_promote(strat: str, *, out_dir: str | None = None,
                 dry_run: bool = False,
                 promote_max: int | None = DEFAULT_PROMOTE_MAX,
                 ) -> dict[str, Any]:
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

    # THE UNATTENDED FAN-OUT BAR, added 2026-09-07 after an audit of
    # approved_incubator/. `promote.py` is per-pair and refuses an ambiguous
    # multi-pair promotion outright; the bulk arrives HERE, because this loop
    # promotes every certified row and the shell helpers sweep 23 contracts x
    # 4 timeframes. One command therefore reached 92 certifiable pairs, and
    # 472 promotion commits over 14 days put 160 packages on disk against 52
    # the routing table actually routes.
    #
    # A CAP RATHER THAN A PROMPT: this path exists to run unattended, so
    # blocking on stdin would hang a detached run rather than protect it. The
    # limit is refused LOUDLY and names the flag that lifts it, so promoting
    # forty configurations stays possible and stops being accidental.
    if (promote_max is not None and int(promote_max) > 0
            and len(winners) > int(promote_max)):
        # A DEFERRAL, NOT A FAILURE - `returncode: 0`, changed 2026-09-10.
        #
        # Nothing went wrong when this fires. Stages 1-4.5 completed, every
        # artifact is on disk, and 15 certifications are on the handoff; the
        # only thing that did not happen is a decision the bar exists to keep
        # from being made unattended. Reporting that as exit 1 tells a cron
        # runner the campaign FAILED, and the difference between "this run
        # broke" and "this run is waiting for you" is the whole value of an
        # exit code to an unattended operator.
        #
        # WHAT THIS DELIBERATELY DOES NOT DO IS PROMOTE THE TOP N BY METRIC.
        # It was considered and rejected on the evidence. Every certified row
        # carries `oos_profit_factor`, so it is mechanically easy - and it
        # would make the 3-year holdout a SELECTION set. Gate R is pass/fail
        # against the charter; ranking the survivors by their holdout profit
        # factor and keeping the top slice chooses on the same window that
        # validated them, which is the curve-fit this repository exists to
        # avoid. It also ranks backwards: on this campaign it would put
        # ETH 30m B (PF 1.79 on 39 OOS trades) above CL 30m A (PF 1.27 on
        # 395), because a profit factor over a thin sample is noise, so the
        # rule would systematically prefer the configurations with the least
        # evidence behind them. A cap that silently drops certified work by an
        # invented ranking is worse than one that stops and says so.
        deferral = (f"Stage 3 certified {len(winners)} configurations and "
                    f"--auto-promote will not register more than "
                    f"{int(promote_max)} unattended.")
        print(f"\n  DEFERRED  {deferral}\n"
              f"            Each one is a package on disk, a row in "
              f"config/portfolios.json and its own git commit, and no human "
              f"has read the Stage 3 card for any of them.\n"
              f"            NOTHING FAILED - every stage completed and the "
              f"certifications are on the handoff. The promotion is the only "
              f"step outstanding.\n"
              f"            Read the card, then promote what you chose:\n"
              f"              python3 backtest/run_pipeline.py --strat {strat} "
              f"--promote-only\n"
              f"            Or lift the bar deliberately:\n"
              f"              --promote-max {len(winners)}",
              file=sys.stderr, flush=True)
        # Recorded on the handoff for the same reason a promotion is: an
        # unattended reader that sees `auto_promotion: null` cannot tell a
        # deferral from a Stage 5 that never ran at all.
        record_auto_promotion(strat, out_dir, [], None, deferred=deferral)
        return {"returncode": 0, "promotions": [], "rows": rows,
                "commit": None, "deferred": deferral}

    try:
        source = _strategy_source(strat)
    except (SystemExit, Exception) as e:                          # noqa: BLE001
        # resolve_strategy raises SystemExit. Uncaught, it would end the
        # process here - after all four stages completed - and the run would
        # report nothing about the certifications already on the record.
        print(f"  Cannot resolve the strategy module to promote from: {e}",
              file=sys.stderr)
        return {"returncode": 1, "promotions": [], "rows": rows, "commit": None}

    # STAGE 4.5 COVERAGE, STATED BEFORE THE FIRST ROW IS PROMOTED.
    #
    # The ordering itself is positional - `main()` runs Stage 4.5 to completion
    # and only then calls this function, and `run_step` is a blocking
    # `subprocess.run` - but positional ordering is exactly what nobody can
    # check afterwards from a promoted package. This says out loud how many of
    # the rows about to be promoted have a verdict on disk, at the moment the
    # loop starts, so a run that promoted 15 packages against 0 gate files is
    # legible in its own log rather than reconstructable from mtimes.
    #
    # A REPORT, NOT A REFUSAL, and deliberately so. Stage 4.5 prunes nothing:
    # its weekday is an instruction for the live supervisor, not evidence about
    # an edge, and a certification that cleared Gate R must not be discarded
    # because a day-of-week profiler raised. `--promote-only` legitimately runs
    # with no Stage 4.5 at all, and a package promoted before the stage existed
    # is a third valid state.
    covered = {(str(r.get("symbol")), str(r.get("timeframe")))
               for r in winners
               if dow_gate_file(strat, str(r.get("symbol")),
                                str(r.get("timeframe")), out_dir) is not None}
    pairs = {(str(r.get("symbol")), str(r.get("timeframe"))) for r in winners}
    print(f"\n  STAGE 4.5 COVERAGE  {len(covered)}/{len(pairs)} certified "
          f"pair(s) have a dow_gate verdict on disk"
          + ("" if len(covered) == len(pairs) else
             "; the rest promote as NOT EVALUATED and block no weekday"),
          flush=True)

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
            # A handoff written before Stage 3 recorded the path on the row.
            # The file is still where Stage 3 always writes it, so the
            # convention is tried before the row is given up on - and only the
            # SUFFIXED name, never the unsuffixed `gate_audit_<SYM>.json`,
            # which holds whichever timeframe ran last and would certify this
            # pair against another one's verdict.
            guess = (pipeline_dir(strat, out_dir)
                     / f"gate_audit_{record['symbol']}_{record['timeframe']}.json")
            if guess.exists():
                audit = str(guess)
                record["audit_file"] = audit
                print(f"  {record['symbol']} {record['timeframe']}: the row "
                      f"records no audit file; using {guess.name}, which is "
                      f"where Stage 3 writes it.")
        if not audit:
            record["error"] = ("the row records no audit file, so there is no "
                               "certification to cite")
            print(f"  SKIPPED  {record['symbol']} {record['timeframe']}: "
                  f"{record['error']}", file=sys.stderr)
            promotions.append(record)
            failures += 1
            continue
        metrics, metrics_basis = resolve_metrics(
            strat, record["symbol"], record["timeframe"], out_dir)
        record["metrics_file"] = str(metrics) if metrics else None
        record["metrics_basis"] = metrics_basis
        print(f"  METRICS  {record['symbol']} {record['timeframe']}: "
              + (f"{metrics_basis}" if metrics
                 else f"NOT RECORDED · {metrics_basis}"))
        dow = dow_gate_file(strat, record["symbol"], record["timeframe"],
                            out_dir)
        record["dow_gate_file"] = str(dow) if dow else None
        if dow is None:
            print(f"  DOW GATE {record['symbol']} {record['timeframe']}: "
                  f"NOT EVALUATED — stage 4.5 left no verdict for this pair, "
                  f"so no weekday is blocked on the promoted package.")
        else:
            # EXISTING IS NOT THE SAME AS CURRENT. An artifact left by an
            # earlier campaign passes `dow_gate_file`'s existence test and
            # would be stamped into meta.json as this promotion's weekday.
            stale = dow_gate_staleness(dow, audit)
            record["dow_gate_stale"] = stale
            if stale:
                print(f"  ! DOW GATE {record['symbol']} "
                      f"{record['timeframe']}: STALE — {stale}",
                      file=sys.stderr, flush=True)
        cmd = promote_cmd(strat, record["version"], source, audit,
                          record["symbol"], record["timeframe"], metrics,
                          dow_gate=dow)
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

    # Read from the incubator and the routing table, never counted from the
    # loop above: the number that matters is how many are unrouted in TOTAL,
    # not how many this invocation happened to refuse.
    unrouted = unrouted_packages()
    card = render_unrouted(unrouted)
    if card:
        print(card)
    return {"returncode": 1 if failures else 0, "promotions": promotions,
            "rows": rows, "commit": commit,
            # Reported, never a failure: an unrouted package is a certified
            # strategy nobody armed, not a broken run. Moving `returncode` on
            # it would fail every pipeline that promotes onto a contract no
            # basket carries yet, which is the normal case.
            "unrouted": unrouted}


if __name__ == "__main__":
    sys.exit(main())
