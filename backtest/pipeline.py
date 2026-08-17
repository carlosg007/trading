"""
backtest/pipeline.py - the contract BETWEEN the five stages.

Location: ~/src/trading/backtest/pipeline.py

Not a stage. This module holds the handoff: where a stage writes its output,
what the next one reads, and the header every stage prints so a log says which
one produced it. It exists because the alternative is each stage hard-coding
the same paths, and a Stage 3 that looks for `best_params_NQ.json` one
directory away from where Stage 2 wrote it fails by finding nothing - which is
indistinguishable from a sweep that produced no winner.

    Stage 1  baseline.py     surviving_assets.json      which contracts carry it
    Stage 2  scan.py         best_params_<SYM>.json     what parameters won
    Stage 3  audit_gates.py  gate_audit_<SYM>.json      PASS/FAIL, on the record
    Stage 4  verify_full.py  verify_<SYM>.json + sheets the full lifecycle
    Stage 5  promote.py      approved_incubator/<strat> the decision

Everything lands under `<BT_ARTIFACTS>/pipeline/<strategy>/`, one directory per
strategy rather than per run, because these files are a CHAIN: Stage 3 has to
find Stage 2's winner without being told a timestamp. That means a re-run
overwrites - deliberately, and only for the small JSON handoffs. Stage 4's tear
sheets keep the repo's usual timestamped-directory rule, since those are the
evidence a promotion cites and evidence is never overwritten.

Every handoff file records the stage that wrote it, the strategy, the date
window and a UTC timestamp. `read_stage` checks the strategy name matches what
the reader expects and raises when it does not: two strategies' pipelines run
in sequence, and picking up the wrong `best_params_NQ.json` would certify one
strategy's gates against another's parameters with nothing raising.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

STAGE_NAMES = {
    1: "BASELINE · which contracts carry the edge",
    2: "SCAN · in-sample parameter selection",
    3: "GATE AUDIT · certification",
    4: "FULL VERIFICATION · whole lifecycle",
    5: "PROMOTION · into the incubator",
}

SURVIVORS_FILE = "surviving_assets.json"
# Stage 1's human-readable half. The JSON above is what Stage 2 reads; this is
# what a person reads, and it is written on EVERY run - including one where
# nothing survived, which is the run whose detail matters most. Markdown rather
# than a console dump because the console is now a progress line per
# configuration: the evidence has to land somewhere, and somewhere is a file.
BASELINE_REPORT_FILE = "stage1_baseline_report.md"
BEST_PARAMS_FILE = "best_params_{symbol}.json"
GATE_AUDIT_FILE = "gate_audit_{symbol}.json"
VERIFY_FILE = "verify_{symbol}.json"


def artifacts_root() -> Path:
    """
    Read at call time, never at import.

    `backtest/run.py` binds `$BT_ARTIFACTS` at import, which is correct for a
    long-lived batch and wrong for a test process that sets the variable after
    importing the module it is testing. These paths are read once per stage, so
    the lookup costs nothing.
    """
    return Path(os.environ.get("BT_ARTIFACTS", "/mnt/backtest/artifacts"))


def pipeline_dir(strategy: str, out_dir: str | Path | None = None,
                 create: bool = False) -> Path:
    """`<artifacts>/pipeline/<strategy>/`, or an explicit override."""
    d = Path(out_dir) if out_dir else artifacts_root() / "pipeline" / strategy
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def stage_banner(stage: int, strategy: str, detail: str = "") -> str:
    """The header every stage prints. One line says which stage a log is."""
    W = 78
    return ("\n" + "=" * W
            + f"\nSTAGE {stage}/5 · {STAGE_NAMES[stage]}"
            + f"\n{strategy}" + (f"  ·  {detail}" if detail else "")
            + "\n" + "=" * W)


def write_stage(path: Path, stage: int, strategy: str,
                payload: dict[str, Any]) -> Path:
    """
    Write a handoff file with its provenance attached.

    Atomic: a temp file then `os.replace`, the same rule `status.py` follows.
    A stage killed mid-write would otherwise leave a truncated JSON that the
    next stage reads as a parse error at best and as a short symbol list at
    worst.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "stage": int(stage),
        "stage_name": STAGE_NAMES[int(stage)],
        "strategy": strategy,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_stage(path: Path, expect_stage: int | None = None,
               expect_strategy: str | None = None) -> dict[str, Any]:
    """
    Read a handoff file and refuse the wrong one.

    A missing file raises `FileNotFoundError` naming the stage that should have
    written it, because "run Stage 2 first" is the actual answer and a bare
    "no such file" sends somebody looking for a bug in Stage 3.
    """
    path = Path(path)
    if not path.exists():
        want = f"stage {expect_stage}" if expect_stage else "an earlier stage"
        raise FileNotFoundError(
            f"{path} does not exist. It is written by {want} — run that first.")
    blob = json.loads(path.read_text(encoding="utf-8"))

    if expect_stage is not None and int(blob.get("stage", -1)) != int(expect_stage):
        raise ValueError(
            f"{path} was written by stage {blob.get('stage')}, not stage "
            f"{expect_stage}.")
    if expect_strategy is not None and blob.get("strategy") != expect_strategy:
        raise ValueError(
            f"{path} belongs to strategy {blob.get('strategy')!r}, not "
            f"{expect_strategy!r}. Certifying one strategy's gates against "
            f"another's parameters is a mistake nothing downstream could "
            f"detect.")
    return blob


def next_step(lines: list[str]) -> str:
    """
    The block every stage ends with: the exact command for the next one.

    Printed rather than run. The tool boundary in CLAUDE.md is that a human is
    at the console when the evidence appears, and a stage that chained straight
    into the next one would put the four-choice menu in front of nobody.
    """
    W = 78
    out = ["", "-" * W, "NEXT STEP", "-" * W]
    out.extend(f"  {ln}" for ln in lines)
    out.append("-" * W)
    return "\n".join(out)
