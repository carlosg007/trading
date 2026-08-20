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

Stage 1 also hands Stage 2 a DECISION, not only a list: the Drop Unprofitable
Days contract. Each surviving pair carries its own `dropped_days` and
`exclude_days` - every weekday whose profit factor was below 1.00 - and
`stage1_exclude_days` below is the one place that mapping is read. Keyed per
`(symbol, timeframe)` rather than globally, because which weekdays lose is a
fact about a contract at a timeframe and a single list applied to every survivor
would prune a session that is profitable on one of them.

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


def stage1_exclude_days(blob: dict[str, Any] | None
                        ) -> dict[tuple[str, str], tuple[int, ...]]:
    """
    Stage 1's Drop Unprofitable Days decision, keyed by `(symbol, timeframe)`.

    The handoff carries one `exclude_days` list PER SURVIVING PAIR - zero, one
    or five days - because which weekdays lose is a fact about a contract at a
    timeframe. Flattening them into one list and applying it everywhere would
    prune a session that is profitable on NQ in order to fix one that is not on
    GC, and nothing downstream could tell that had happened.

    Pairs with an empty list are omitted from the mapping rather than mapped to
    `()`. `()` and "no entry" mean the same thing to every caller here, and the
    absent key keeps `if key in mapping` an honest test of "did Stage 1 exclude
    anything for this pair".

    A blob written before the contract existed - or by a run with
    `--no-drop-losing-days` - simply yields an empty mapping. That is the correct
    reading: no exclusion was decided, so Stage 2 sweeps the whole week.
    """
    out: dict[tuple[str, str], tuple[int, ...]] = {}
    for pair in (blob or {}).get("surviving_pairs") or []:
        if not isinstance(pair, dict):
            continue
        sym, tf = pair.get("symbol"), pair.get("tf")
        days = tuple(sorted({int(d) for d in (pair.get("exclude_days") or [])}))
        if sym and tf and days:
            out[(str(sym), str(tf))] = days
    return out


def leaderboard(title: str, header: list[str], rows: list[list[str]],
                align: list[str] | None = None,
                empty: str = "nothing to report") -> str:
    """
    The end-of-stage table, rendered the same way by every stage.

    One implementation rather than three, because the point of printing a
    leaderboard at each stage is that an operator reads them as a sequence: a
    Symbol column that is left-aligned in Stage 1 and right-aligned in Stage 2
    makes two tables of the same contracts look like tables of different things.

    Columns are sized to their widest CELL, not to a fixed width, so a long
    parameter set widens its own column instead of being silently clipped -
    a truncated winning parameter set reads as a complete one, and the whole
    value of the row is that it names the parameters exactly.

    `align` is one of `"<"` or `">"` per column, defaulting to left for the
    first and right for the rest, which is the shape every one of these tables
    has: identifiers on the left, numbers on the right. A stage that ends with
    no rows still prints the heading and says so - an absent table reads as a
    stage that did not finish.
    """
    align = align or (["<"] + [">"] * (len(header) - 1))
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(header[i]), *(len(r[i]) for r in cells)) if cells
              else len(header[i]) for i in range(len(header))]

    def _line(vals: list[str]) -> str:
        # Right-stripped: a padded final column leaves trailing spaces on every
        # row, which survive into a log file and show up as a diff against the
        # same table pasted anywhere else.
        return ("  " + "  ".join(f"{v:{align[i]}{widths[i]}}"
                                 for i, v in enumerate(vals))).rstrip()

    rule = "  " + "  ".join("-" * w for w in widths)
    total = max(len(rule), len(title) + 2)
    out = ["", "=" * total, f"  {title}", "=" * total, _line(header), rule]
    out.extend(_line(r) for r in cells)
    if not cells:
        out.append(f"  ({empty})")
    out.append("=" * total)
    return "\n".join(out)


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
