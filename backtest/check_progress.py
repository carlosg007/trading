#!/usr/bin/env python3
"""
backtest/check_progress.py - what the pipeline is doing, in English.

Location:  ~/src/trading/backtest/check_progress.py

    python3 backtest/check_progress.py           # one card
    python3 backtest/check_progress.py --watch 10
    bt-check / bt-progress                       # the aliases

WHY THIS EXISTS BESIDE status.py
--------------------------------
`backtest/status.py` reads `active_job.json`, a file `backtest/run.py` WRITES.
That is the right design for the batch runner and it is unavailable here:
`backtest/run_pipeline.py` writes no progress file at all. It shells out to
four stage scripts in sequence with `subprocess.run(..., check=True)`, and the
only running record of where it has got to is its own stdout - which, for a
run started under tmux or redirected to a log, is exactly the thing nobody is
watching.

So this reader infers rather than reads, from two sources that exist whether
or not anybody instrumented the run:

  * the PROCESS TABLE - the orchestrator and whichever stage script is its
    live descendant, which is what says "Stage 2, right now" rather than
    "Stage 2 has written some files";
  * the ARTIFACT TREE under `<BT_ARTIFACTS>/pipeline/<strategy>/`, whose file
    names carry the symbol and timeframe of every unit of work that finished.

Inference is weaker evidence than a written progress file and the readout says
so where it matters. It is never used to state a RESULT - no Sharpe, no gate
verdict, no certification is reported here. This answers "where is it up to",
and the four-choice promotion menu still answers "was it any good".

HOW THE STAGE 2 WORK PLAN IS RECONSTRUCTED EXACTLY
--------------------------------------------------
The interesting number - `[3/5] RB` - is not a guess. Stage 2 sweeps Stage 1's
EXACT surviving pairs, and `scan.resolve_targets` with neither `--symbols` nor
`--tf` (which is how `run_pipeline.stage2_cmd` always calls it) returns those
pairs in `surviving_assets.json` order, unsorted. `scan.main` then groups them
with `list(dict.fromkeys(...))`, so both the timeframe order and the symbol
order within a timeframe are first-appearance order in that one file.

Reading the same file the same way reproduces the plan the sweep is walking,
and the artifacts on disk say how far along it is. The symbol currently being
swept is the FIRST PLANNED ONE WITH NO ARTIFACT - it has none precisely
because it has not finished - which is why the position is derived from the
plan and not from the newest file's name.

That reconstruction is only valid for a sweep launched by the orchestrator. An
operator who runs `scan.py --symbols ...` by hand has overridden the handoff,
and the readout says the plan is unavailable instead of showing a position in
a plan that is not being walked.

WHICH FILE MEANS WHICH STAGE
----------------------------
    Stage 1  baseline.py     regime_profile_<SYM>_<TF>_version_<a|b>.json  live
                             surviving_assets.json + stage1_baseline_report.md
    Stage 2  scan.py         <TF>/scan_<SYM>.csv        one per swept pair
                             best_params_<SYM>_<TF>.json  one per WINNER
                             stage2_summary.json          at the end
    Stage 3  audit_gates.py  gate_audit_<SYM>_<TF>.json
                             stage3_audit_summary.json    at the end
    Stage 4  verify_full.py  verify_<stamp>/dual_metrics_<SYM>.json
    Stage 5  promote.py      the `auto_promotion` block on Stage 3's summary,
                             and strategies/approved_incubator/<id>/meta.json

`scan_<SYM>.csv` is the Stage 2 progress marker rather than
`best_params_<SYM>_<TF>.json`, because the sweep writes the table for every
pair it evaluates and withholds the parameter file when every combination
failed the fragility bar. Counting parameter files would report a pair that
ran and was pruned as a pair that has not run yet.

FRESH VERSUS CARRIED OVER
-------------------------
The artifacts tree is ONE DIRECTORY PER STRATEGY, not per run, so a re-run
overwrites the small JSON handoffs and leaves everything it has not reached
yet in place. A reader that only asks "does gate_audit_NQ_15m.json exist"
therefore reports the PREVIOUS campaign's Stage 3 as this campaign's, at
minute four of a nine-hour run, with every line reading correctly.

So while a run is live, every artifact is compared against the orchestrator's
start time, and anything older is counted separately and labelled `earlier
run`. Stale evidence reading as fresh is the failure this repository is built
around; a status tool is not permitted to introduce it.

NOTHING HERE RUNS A BACKTEST
----------------------------
This reads `ps`, `stat` and small JSON handoffs. It imports no strategy
module, loads no bars, and touches the lake at `/mnt/backtest/raw` not at all.
It is safe to run at any time, including from inside a session, and it is the
only part of the pipeline that is.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. This runs at import
# time, above the local imports below, because several modules resolve their
# BT_* variables while being imported - loading the file inside main() would be
# too late for those and would work here, which is the kind of difference
# nobody notices until one runner silently uses the default path. The rules -
# the repository root derived from __file__ rather than the working directory,
# existing variables winning over the file, the CrossTrade credentials withheld
# from os.environ - live in ONE module rather than in a block copied into every
# runner: see mdlib/env.py.
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
import difflib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# `backtest.pipeline` is the contract module: it declares the handoff names and
# imports nothing but the standard library and mdlib.env, so importing it here
# costs nothing and keeps ONE spelling of `surviving_assets.json`. A status
# tool holding its own copy of those names would keep reporting PENDING after
# somebody renamed a handoff, which is the failure mode it exists to catch.
from backtest.pipeline import (  # noqa: E402
    BASELINE_REPORT_FILE,
    STAGE2_SUMMARY_FILE,
    STAGE3_SUMMARY_FILE,
    SURVIVORS_FILE,
    artifacts_root,
    pipeline_dir,
)

W = 80

#: The orchestrator, and the stage script each of its children belongs to.
#: Matched on the BASENAME of an argument, so `/home/x/src/trading/backtest/
#: scan.py` and a relative `backtest/scan.py` resolve identically.
ORCHESTRATOR = "run_pipeline.py"
STAGE_OF_SCRIPT = {
    "baseline.py": 1,
    "scan.py": 2,
    "audit_gates.py": 3,
    "verify_full.py": 4,
    "promote.py": 5,
    # Not a stage. Named so a run sitting in a Discord POST is reported as
    # posting a card rather than as an orchestrator with no live stage, which
    # reads like a hang.
    "discord_reporter.py": 0,
}
STAGE_TITLES = {
    1: "Stage 1 (Baseline)",
    2: "Stage 2 (Optimize)",
    3: "Stage 3 (Certify)",
    4: "Stage 4 (Verify)",
}
STAGE_ACTIONS = {
    1: "Stage 1 Baseline Screen",
    2: "Stage 2 Optimization",
    3: "Stage 3 Certification",
    4: "Stage 4 Full Verification",
    5: "Stage 5 Promotion",
    0: "Discord card",
}

RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
PENDING = "PENDING"
FAILED = "FAILED"
PARTIAL = "PARTIAL"

#: A `verify_<stamp>/` lifecycle directory, and the timestamp inside its name.
VERIFY_DIR_RE = re.compile(r"^verify_(\d{8}_\d{6})$")


# ==========================================================================
# formatting helpers
# ==========================================================================

def elapsed_text(seconds: float | None) -> str:
    """`08h 52m 14s`, zero-padded so consecutive readouts line up."""
    if seconds is None:
        return "unknown"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def ago_text(seconds: float | None) -> str:
    """`14s ago`, `6m ago`, `2h 11m ago` - the coarse form a human reads."""
    if seconds is None:
        return "unknown"
    seconds = int(max(0, seconds))
    if seconds < 90:
        return f"{seconds}s ago"
    m, s = divmod(seconds, 60)
    if m < 90:
        return f"{m}m {s:02d}s ago"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h {m:02d}m ago"
    return f"{h // 24}d {h % 24}h ago"


def stamp_text(epoch: float | None) -> str:
    """A local wall-clock stamp. Local, not UTC: this is read at the console."""
    if not epoch:
        return "unknown"
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def truncated_list(items: Sequence[str], keep: int = 11) -> str:
    """`ES, NQ, RTY, ... (+12 more)` - a universe named, not dumped."""
    items = list(items)
    if not items:
        return "none"
    if len(items) <= keep:
        return ", ".join(items)
    return ", ".join(items[:keep]) + f", ... (+{len(items) - keep} more)"


def rel(path: Path | None) -> str:
    return str(path) if path else "n/a"


# ==========================================================================
# the process table
# ==========================================================================

@dataclass
class Proc:
    """One row of `ps`, with only the columns this readout uses."""
    pid: int
    ppid: int
    etimes: int          # seconds alive, from ps - no lstart parsing
    pcpu: float
    pmem: float
    args: str

    @property
    def started(self) -> float:
        return time.time() - self.etimes

    def script(self) -> str | None:
        """
        The basename of the first `.py` argument, or None.

        Taken from the ARGUMENTS rather than from the command name, because
        every one of these runs as `python3 <script>` - and under the shell
        helper as `nice -n 19 ionice -c 3 python3 <script>`, where the command
        name is still python3. The path is basenamed so an absolute and a
        relative invocation of the same stage are one thing.
        """
        for token in self.args.split():
            if token.endswith(".py"):
                return os.path.basename(token)
        return None


def ps_snapshot() -> list[Proc] | None:
    """
    Every process, as `Proc` rows - or None when the table could not be read.

    `etimes` rather than `lstart`: elapsed seconds is one integer field, while
    `lstart` is five space-separated ones that would have to be parsed back
    through a locale.

    None rather than an empty list for a `ps` that is missing, refuses, or does
    not support the format, because the caller has to tell those apart from a
    table it read successfully. "Nothing is running" is a fact about the
    pipeline; "I could not look" is a fact about this box, and printing the
    first when the second is true is how a live nine-hour run gets reported as
    idle.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,etimes=,pcpu=,pmem=,args="],
            capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None

    procs: list[Proc] = []
    for line in out.stdout.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            procs.append(Proc(pid=int(parts[0]), ppid=int(parts[1]),
                              etimes=int(parts[2]), pcpu=float(parts[3]),
                              pmem=float(parts[4]), args=parts[5]))
        except ValueError:
            # A kernel thread or a row whose args contain something unparseable
            # is not a pipeline; skipping it is the whole handling.
            continue
    return procs


def _is_self(proc: Proc, me: int) -> bool:
    """
    This reader, its `ps` child, and any shell that has it on the line.

    Matched on the SCRIPT NAME rather than only on the pid, because
    `bt-check | tee` and `watch bt-check` put this file's path on a second
    process's command line - and a status tool that reports itself as a
    running pipeline is worse than one that reports nothing.
    """
    return (proc.pid == me or proc.pid == os.getppid()
            or "check_progress.py" in proc.args)


def find_orchestrators(procs: Iterable[Proc]) -> list[Proc]:
    """
    Live `run_pipeline.py` processes, LONGEST RUNNING FIRST.

    The order is the choice: callers report `[0]`, and two orchestrators alive
    at once means somebody launched a second campaign over the first. The one
    that has been running for eight hours is the one whose artifacts fill the
    directory, so it is the one the card describes - and reporting the
    two-minute-old process instead would attribute an entire campaign's tree to
    a run that has barely started.
    """
    me = os.getpid()
    found = [p for p in procs
             if not _is_self(p, me) and p.script() == ORCHESTRATOR]
    return sorted(found, key=lambda p: p.etimes, reverse=True)


def find_stage_procs(procs: Iterable[Proc]) -> list[Proc]:
    """Live stage scripts, whoever launched them."""
    me = os.getpid()
    return [p for p in procs
            if not _is_self(p, me) and p.script() in STAGE_OF_SCRIPT]


def descendants(procs: Sequence[Proc], root_pid: int) -> list[Proc]:
    """
    Every process below `root_pid`, at any depth.

    A full walk rather than a ppid match, because the chain from the shell
    helper is `bash -> nice -> python3 run_pipeline.py -> python3 scan.py` and
    a sweep parallelised over symbols adds another level below that. One
    generation of lookup would find the stage script today and silently find
    nothing the day a stage grows a worker pool.
    """
    children: dict[int, list[Proc]] = {}
    for p in procs:
        children.setdefault(p.ppid, []).append(p)
    out, stack, seen = [], [root_pid], {root_pid}
    while stack:
        for child in children.get(stack.pop(), []):
            if child.pid in seen:
                continue
            seen.add(child.pid)
            out.append(child)
            stack.append(child.pid)
    return out


def parse_cli_flags(args: str) -> dict[str, Any]:
    """
    `--strat X --tf 1m,5m --report-discord` as a dict.

    A hand-rolled scanner rather than argparse: this parses the command line of
    ANOTHER process, which may have been started by a version of the script
    with flags this one does not declare. argparse would exit on the first of
    those; here an unknown flag is simply carried, and a flag that is missing
    reads as absent rather than as a crash.

    A bare flag maps to True. Quoting is lost by `ps` before this sees it, so
    only values without spaces survive - which is every flag the pipeline takes.
    """
    flags: dict[str, Any] = {}
    tokens = args.split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            if "=" in key:
                key, _, val = key.partition("=")
                flags[key] = val
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[key] = tokens[i + 1]
                i += 1
            else:
                flags[key] = True
        i += 1
    return flags


# ==========================================================================
# the artifact tree
# ==========================================================================

def read_json(path: Path) -> dict[str, Any] | None:
    """
    A handoff, or None when it is absent, truncated or not a mapping.

    None for a TRUNCATED file as well as a missing one, and that is the whole
    reason this wrapper exists. Stages write through `os.replace`, but a status
    tool polling an NFS mount during a nine-hour run will eventually read one
    mid-flight, and a JSONDecodeError surfacing from a progress card looks like
    a broken pipeline rather than a broken read. The next poll succeeds.
    """
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


@dataclass
class FileGroup:
    """A set of artifacts split by whether THIS run wrote them."""
    fresh: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return self.fresh + self.stale

    def __len__(self) -> int:
        return len(self.fresh) + len(self.stale)


@dataclass
class Stage2Timeframe:
    """
    One timeframe's slice of the Stage 2 plan, and how far into it we are.

    `covered` is what counts as done, and WHICH FILES FEED IT is the whole
    subtlety. While a run is live it is the tables THIS run wrote; with nothing
    running it is every table on disk. A re-run starts against a directory
    already holding the previous campaign's tables for every timeframe, so
    counting those would report the sweep as four timeframes ahead of where it
    is - and would put the `[n/N]` position on a contract it has not reached.

    `carried` names the symbols the earlier run covered and this one has not
    got back to yet, so the tables are reported rather than silently ignored.
    """
    tf: str
    planned: list[str]
    done: FileGroup
    status: str
    covered: list[str] = field(default_factory=list)
    carried: list[str] = field(default_factory=list)
    current: str | None = None       # the symbol being swept, if this is the one
    position: int | None = None      # its 1-based index into `planned`

    @property
    def pending(self) -> list[str]:
        return [s for s in self.planned if s not in set(self.covered)]


class Campaign:
    """
    One strategy's artifact directory, read as progress rather than as results.

    `since` is the orchestrator's start time when a run is live and None when
    it is not. Every artifact is classified against it, so a card printed four
    minutes into a re-run does not present the previous campaign's Stage 3 as
    this one's. With no run in flight there is nothing to be fresh relative to,
    and the split is not drawn at all.
    """

    def __init__(self, strategy: str, out_dir: str | Path | None = None,
                 since: float | None = None) -> None:
        self.strategy = strategy
        self.dir = pipeline_dir(strategy, out_dir)
        self.since = since
        # One tolerance for the whole card. Clocks on an NFS server and on this
        # box are not identical, and a file written in the run's first seconds
        # must not be filed under the previous campaign because the mount's
        # clock is thirty seconds behind.
        self.slack = 120.0

    # -- existence ------------------------------------------------------
    def exists(self) -> bool:
        return self.dir.is_dir()

    def _glob(self, pattern: str) -> list[Path]:
        try:
            return sorted(self.dir.glob(pattern))
        except OSError:
            return []

    def _fresh(self, path: Path) -> bool:
        if self.since is None:
            return True
        m = mtime(path)
        return m is not None and m >= (self.since - self.slack)

    def _group(self, paths: Iterable[Path], label) -> FileGroup:
        g = FileGroup()
        for p in paths:
            (g.fresh if self._fresh(p) else g.stale).append(label(p))
        return g

    # -- Stage 1 --------------------------------------------------------
    def survivors(self) -> dict[str, Any] | None:
        return read_json(self.dir / SURVIVORS_FILE)

    def stage1(self, universe: Sequence[str] | None,
               timeframes: Sequence[str] | None,
               live: bool) -> dict[str, Any]:
        """
        Stage 1's state, and while it runs, how much of the screen is done.

        The screen writes `regime_profile_<SYM>_<TF>_version_<a|b>.json` as it
        goes and `surviving_assets.json` only at the end, so the profiles are
        the live counter and the handoff is the completion marker. The
        denominator is symbols x timeframes off the ORCHESTRATOR's command
        line - the screen's own record of what it evaluated does not exist
        until it has finished, and a progress fraction with no denominator is
        a count dressed as progress.
        """
        blob = self.survivors()
        handoff = self.dir / SURVIVORS_FILE
        report = self.dir / BASELINE_REPORT_FILE
        pairs = self._screened_pairs()

        if blob is not None and self._fresh(handoff):
            n_pairs = len(blob.get("surviving_pairs") or [])
            n_eval = blob.get("evaluated")
            syms = sorted({str(p.get("symbol")) for p in
                           (blob.get("surviving_pairs") or [])
                           if p.get("symbol")})
            detail = (f"{n_pairs} pair(s) promoted across "
                      f"{len(syms)} contract(s)")
            if n_eval:
                detail += f", from {n_eval} screened"
            return {"status": COMPLETED, "detail": detail,
                    "pairs": n_pairs, "symbols": syms,
                    "report": report if report.exists() else None}

        if live:
            total = (len(universe) * len(timeframes)
                     if universe and timeframes else None)
            done = len(pairs)
            detail = (f"{done} of {total} (symbol x timeframe) screened"
                      if total else f"{done} (symbol x timeframe) screened")
            if blob is not None:
                detail += " · the handoff on disk is an earlier run's"
            return {"status": RUNNING, "detail": detail, "pairs": 0,
                    "symbols": [], "report": None}

        if blob is not None:
            n_pairs = len(blob.get("surviving_pairs") or [])
            return {"status": COMPLETED, "detail":
                    f"{n_pairs} pair(s) promoted (earlier run)",
                    "pairs": n_pairs, "symbols": [], "report":
                    report if report.exists() else None}
        return {"status": PENDING, "detail": "no surviving_assets.json yet",
                "pairs": 0, "symbols": [], "report": None}

    def _screened_pairs(self) -> set[tuple[str, str]]:
        """The `(symbol, tf)` pairs Stage 1 has written a regime profile for."""
        out: set[tuple[str, str]] = set()
        for p in self._glob("regime_profile_*.json"):
            if not self._fresh(p):
                continue
            # regime_profile_<SYM>_<TF>_version_<v>[_holdout].json
            m = re.match(r"^regime_profile_([A-Z0-9]+)_([0-9]+[a-z])_version_",
                         p.name)
            if m:
                out.add((m.group(1), m.group(2)))
        return out

    # -- Stage 2 --------------------------------------------------------
    def stage2_plan(self) -> list[tuple[str, list[str]]]:
        """
        The sweep's work list, in the ORDER it walks it.

        Reconstructed from `surviving_assets.json` exactly as
        `scan.resolve_targets` + `scan.main` build it when the orchestrator
        calls them - neither `--symbols` nor `--tf`, so the pairs pass through
        unsorted and are grouped by first appearance. See the module docstring.

        An empty list means the handoff is absent or carries no usable pair,
        and the readout then declines to show a position rather than showing
        one in a plan nobody is walking.
        """
        blob = self.survivors()
        if not blob:
            return []
        order: list[str] = []
        by_tf: dict[str, list[str]] = {}
        for pair in blob.get("surviving_pairs") or []:
            if not isinstance(pair, dict):
                continue
            sym = pair.get("symbol")
            tf = pair.get("tf") or pair.get("timeframe")
            if not sym or not tf:
                # `pipeline.stage1_pairs` skips these too. A survivor whose
                # timeframe cannot be read is a handoff bug, and inventing one
                # would put a symbol in the wrong timeframe's queue.
                continue
            sym, tf = str(sym), str(tf)
            if tf not in by_tf:
                by_tf[tf] = []
                order.append(tf)
            if sym not in by_tf[tf]:
                by_tf[tf].append(sym)
        return [(tf, by_tf[tf]) for tf in order]

    def _scan_tables(self, tf: str, multi_tf: bool) -> list[Path]:
        """
        The `scan_<SYM>.csv` tables written at one timeframe.

        `scan.main` writes into `<out_dir>/<tf>/` only when the run spans MORE
        THAN ONE timeframe, and straight into `<out_dir>/` when it spans one.
        Both are checked rather than only the expected one: a campaign whose
        earlier stages ran multi-timeframe and whose latest sweep ran single
        leaves tables in both places, and looking in one would report the
        other's work as not done.
        """
        found = self._glob(f"{tf}/scan_*.csv")
        if not multi_tf or not found:
            found = found + self._glob("scan_*.csv")
        return found

    def stage2(self, plan: Sequence[tuple[str, list[str]]],
               live_tf: str | None, live: bool) -> list[Stage2Timeframe]:
        """
        Per-timeframe progress through the sweep.

        `live_tf` is the timeframe the running `scan.py` is actually on, when
        that can be established; otherwise the first timeframe THIS run has not
        finished is taken as current, which is the order the sweep walks them
        in.

        A live sweep is measured only against the tables it has written itself.
        `scan.py` re-writes every table it is asked for, so the previous
        campaign's tables sitting in the same directory say nothing about this
        run's progress - and taken as progress they would place the position
        several timeframes ahead of the contract actually being swept.
        """
        multi = len(plan) > 1
        rows: list[Stage2Timeframe] = []

        for tf, planned in plan:
            tables = self._scan_tables(tf, multi)
            done = self._group(tables, lambda p: p.stem[len("scan_"):])
            wanted = set(planned)
            # The BASIS is `since`, not `live`. A run in flight between two
            # stages is still a run whose Stage 2 progress is its own - keying
            # this to "the sweep is the current stage" would re-admit the
            # previous campaign's tables the moment scan.py exited.
            counted = done.fresh if self.since is not None else done.all
            covered = [s for s in planned if s in set(counted) & wanted]
            carried = [s for s in planned
                       if s in (set(done.stale) & wanted) and s not in covered]

            if covered and len(covered) >= len(planned):
                status = COMPLETED
            elif covered:
                status = PARTIAL
            else:
                status = PENDING
            rows.append(Stage2Timeframe(tf=tf, planned=list(planned), done=done,
                                        status=status, covered=covered,
                                        carried=carried))

        if not live:
            return rows

        # The one timeframe in flight. Named explicitly when the running
        # process says so; otherwise the first one the sweep has not finished.
        for row in rows:
            if live_tf is not None and row.tf != live_tf:
                continue
            if live_tf is None and row.status == COMPLETED:
                continue
            row.status = RUNNING
            pending = row.pending
            if pending:
                row.current = pending[0]
                row.position = row.planned.index(pending[0]) + 1
            break
        return rows

    def stage2_summary(self) -> dict[str, Any] | None:
        return read_json(self.dir / STAGE2_SUMMARY_FILE)

    def grid_size(self) -> tuple[int | None, str | None, bool]:
        """
        How many combinations one sweep evaluates, and which file said so.

        Read off the most recent `best_params_<SYM>_<TF>.json` as
        `variants_tested`, which is the count that ACTUALLY RAN - the grid's
        cross product includes combinations the sweep rejects before
        simulating. `stage2_summary.json` is not used for this: it is written
        when Stage 2 finishes, so during a live sweep it describes the previous
        run's grid while looking exactly like this one's.
        """
        newest, newest_m = None, -1.0
        for p in self._glob("best_params_*.json"):
            m = mtime(p) or -1.0
            if m > newest_m:
                newest, newest_m = p, m
        if newest is None:
            return None, None, False
        blob = read_json(newest) or {}
        n = blob.get("variants_tested")
        return ((int(n) if isinstance(n, (int, float)) else None),
                newest.name, self._fresh(newest))

    # -- Stage 3 --------------------------------------------------------
    def stage3(self, live: bool) -> dict[str, Any]:
        """
        Certification: which pairs have a gate audit, and what the index says.

        The per-pair `gate_audit_<SYMBOL>_<TF>.json` is the authoritative
        verdict and the unsuffixed `gate_audit_<SYMBOL>.json` is NOT counted -
        it holds whichever timeframe ran last, so counting it would double one
        pair and name no timeframe for it.
        """
        audits = [p for p in self._glob("gate_audit_*.json")
                  if re.match(r"^gate_audit_[A-Z0-9]+_[0-9]+[a-z]\.json$",
                              p.name)]
        done = self._group(audits, lambda p: p.stem[len("gate_audit_"):])
        summary = read_json(self.dir / STAGE3_SUMMARY_FILE)
        summary_fresh = (summary is not None
                         and self._fresh(self.dir / STAGE3_SUMMARY_FILE))

        targets = None
        certified = None
        if summary_fresh:
            cov = summary.get("coverage") or {}
            targets = cov.get("targets")
            certified = cov.get("certified")

        if live and done.fresh and not summary_fresh:
            status = RUNNING
        elif summary_fresh:
            status = COMPLETED
        elif done.fresh:
            status = PARTIAL
        elif done.stale or summary is not None:
            status = PENDING
        else:
            status = PENDING

        return {"status": status, "audited": done, "targets": targets,
                "certified": certified, "summary": summary,
                "summary_fresh": summary_fresh}

    # -- Stage 4 --------------------------------------------------------
    def stage4(self, live: bool) -> dict[str, Any]:
        """
        The lifecycle run: `verify_<stamp>/` directories holding dual metrics.

        Timestamped directories, so unlike every other stage here nothing is
        overwritten and the freshness split is drawn on the STAMP rather than
        on an mtime.
        """
        dirs = []
        try:
            for d in self.dir.iterdir():
                if d.is_dir() and VERIFY_DIR_RE.match(d.name):
                    dirs.append(d)
        except OSError:
            pass
        dirs.sort(key=lambda d: d.name)

        fresh_syms: set[str] = set()
        stale_syms: set[str] = set()
        newest = dirs[-1].name if dirs else None
        for d in dirs:
            try:
                metrics = sorted(d.glob("dual_metrics_*.json"))
            except OSError:
                continue
            bucket = fresh_syms if self._fresh(d) else stale_syms
            for m in metrics:
                bucket.add(m.stem[len("dual_metrics_"):])

        if fresh_syms and live:
            status = RUNNING
        elif fresh_syms:
            status = COMPLETED
        elif stale_syms:
            status = PENDING
        else:
            status = PENDING
        return {"status": status, "fresh": sorted(fresh_syms),
                "stale": sorted(stale_syms), "runs": len(dirs),
                "newest": newest}

    # -- Stage 5 --------------------------------------------------------
    def stage5(self) -> dict[str, Any]:
        """
        Promotion, read from the block Stage 5 writes back onto Stage 3.

        `run_pipeline.record_auto_promotion` adds `auto_promotion` to the
        Stage 3 summary once promote.py has COMMITTED, which is the only
        record that distinguishes a promotion from a staging. The incubator
        directories are counted alongside it because `--promote-only` and a
        hand-run promotion both land there, and neither annotates a summary
        this campaign may not even have.
        """
        summary = read_json(self.dir / STAGE3_SUMMARY_FILE) or {}
        promo = summary.get("auto_promotion") or {}
        incubator = REPO / "strategies" / "approved_incubator"
        ids: list[str] = []
        try:
            for d in sorted(incubator.iterdir()):
                if d.is_dir() and d.name.startswith(self.strategy):
                    ids.append(d.name)
        except OSError:
            pass
        return {"ran": bool(promo.get("ran")),
                "promoted": promo.get("promoted"),
                "failed": promo.get("failed"),
                "commit": promo.get("commit"),
                "at": promo.get("at"),
                "incubator_ids": ids}

    # -- the tree as a whole --------------------------------------------
    def latest_artifact(self) -> tuple[Path | None, float | None]:
        """
        The most recently modified file anywhere under the strategy directory.

        A full walk. The tree runs to a few hundred small files - the tear
        sheets and scan tables are the large ones and none of them is READ
        here, only stat'ed - so this costs one directory traversal on the NFS
        mount and nothing else. Nothing is opened, which is the rule for
        anything under /mnt/backtest.
        """
        newest, newest_m = None, -1.0
        try:
            for root, _dirs, files in os.walk(self.dir):
                for name in files:
                    p = Path(root) / name
                    m = mtime(p)
                    if m is not None and m > newest_m:
                        newest, newest_m = p, m
        except OSError:
            return None, None
        return newest, (newest_m if newest is not None else None)


def newest_campaign(out_dir: str | Path | None = None) -> str | None:
    """
    The strategy whose artifacts were touched most recently.

    What `bt-check` reports when no run is live and nobody named a strategy.
    Chosen on the newest FILE rather than on the directory mtime: a directory's
    mtime moves when a subdirectory is created and not when a file inside one
    is rewritten, so a sweep writing into `5m/` for six hours never touches the
    parent and the wrong campaign would win.
    """
    ranked = list_campaigns(out_dir)
    return ranked[0][0] if ranked else None


def list_campaigns(out_dir: str | Path | None = None
                   ) -> list[tuple[str, float | None]]:
    """
    Every campaign directory with an artifact tree, NEWEST FIRST.

    Ranked on the newest FILE anywhere beneath each directory, for the reason
    `newest_campaign` gives: a directory's own mtime moves when a subdirectory
    is created and not when a file inside one is rewritten, so a sweep writing
    into `5m/` for six hours never touches the parent.

    A directory with no readable file at all is kept, with `None` for its time,
    and sorts last. Dropping it would make an empty campaign - a run that died
    before Stage 1 wrote anything - indistinguishable from one that was never
    launched, and that is exactly the state somebody runs this to diagnose.
    """
    root = Path(out_dir) if out_dir else artifacts_root() / "pipeline"
    try:
        candidates = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return []
    out: list[tuple[str, float | None]] = []
    for d in candidates:
        best_m: float | None = None
        try:
            for sub, _dirs, files in os.walk(d):
                for name in files:
                    m = mtime(Path(sub) / name)
                    if m is not None and (best_m is None or m > best_m):
                        best_m = m
        except OSError:
            pass
        out.append((d.name, best_m))
    out.sort(key=lambda r: (r[1] is not None, r[1] or 0.0), reverse=True)
    return out


def match_campaign(requested: str, out_dir: str | Path | None = None
                   ) -> tuple[str | None, list[str]]:
    """
    Resolve a user-typed strategy to a real campaign directory.

    Returns `(name, suggestions)`. An exact hit returns `(name, [])` without
    consulting the rest, so a strategy whose name is a substring of another
    still resolves to ITSELF rather than to an ambiguity.

    Falling back: case-insensitive equality, then case-insensitive substring.
    A substring hit is only accepted when it is UNIQUE - `ema_crossover` across
    three dated campaigns is an ambiguity, and silently reporting on the newest
    of them would put a card with the wrong provenance in front of somebody who
    asked a precise question. Ambiguous and missing both come back as
    `(None, suggestions)`; the caller prints them and exits non-zero.
    """
    names = [n for n, _ in list_campaigns(out_dir)]
    if requested in names:
        return requested, []
    folded = requested.casefold()
    ci = [n for n in names if n.casefold() == folded]
    if len(ci) == 1:
        return ci[0], []
    sub = [n for n in names if folded in n.casefold()]
    if len(sub) == 1:
        return sub[0], []
    if sub:
        return None, sub
    close = difflib.get_close_matches(requested, names, n=5, cutoff=0.6)
    return None, close


def running_rows(procs: Sequence[Proc]) -> list[dict[str, Any]]:
    """
    One row per live orchestrator: pid, strategy, stage, timeframes, elapsed.

    The strategy and the timeframes are read off the ORCHESTRATOR's own command
    line with `parse_cli_flags`, not off the artifacts, because that is the
    only source that says what THIS process was asked to do - two campaigns
    writing into the same tree are told apart by their arguments and by nothing
    else.

    The stage is the live descendant, resolved through the same `descendants`
    walk the single-campaign card uses: a stage script is a grandchild under
    `nice`, and a sweep parallelised over symbols puts it deeper still.
    """
    rows: list[dict[str, Any]] = []
    for orch in find_orchestrators(procs):
        flags = parse_cli_flags(orch.args)
        kids = [p for p in descendants(procs, orch.pid)
                if p.script() in STAGE_OF_SCRIPT]
        stage = STAGE_OF_SCRIPT.get(kids[0].script() or "") if kids else None
        strat = flags.get("strat")
        tf = flags.get("tf")
        rows.append({
            "pid": orch.pid,
            "strategy": strat if isinstance(strat, str) and strat else "?",
            "stage": stage,
            "tf": tf if isinstance(tf, str) and tf else "?",
            "etimes": orch.etimes,
        })
    return rows


def _stage_text(stage: int | None) -> str:
    if stage is None:
        return "-"
    if stage == 0:
        return "discord"
    return f"{stage}"


def render_running_table(rows: Sequence[dict[str, Any]]) -> str:
    """The compact multi-pipeline summary, one line per live orchestrator."""
    L = ["=" * W, f"BACKTEST PIPELINE STATUS  ·  {len(rows)} RUNS ACTIVE", "=" * W,
         f"{'PID':>7}  {'STAGE':>5}  {'ELAPSED':>11}  {'TIMEFRAMES':<18}  STRATEGY",
         "-" * W]
    for r in rows:
        L.append(f"{r['pid']:>7}  {_stage_text(r['stage']):>5}  "
                 f"{elapsed_text(r['etimes']):>11}  {r['tf'][:18]:<18}  "
                 f"{r['strategy']}")
    L += ["-" * W,
          "  More than one campaign is live, so no single card describes the box.",
          "  Full Stage 1-5 detail for one of them:",
          "      bt-check <strategy>",
          "=" * W]
    return "\n".join(L)


def render_campaign_list(out_dir: str | None = None) -> str:
    """Every campaign with an artifact tree, newest first."""
    root = Path(out_dir) if out_dir else artifacts_root() / "pipeline"
    ranked = list_campaigns(out_dir)
    L = ["=" * W, "CAMPAIGNS WITH AN ARTIFACT TREE", "=" * W, f"  {root}", ""]
    if not ranked:
        L += ["  (none)", "=" * W]
        return "\n".join(L)
    now = time.time()
    for name, m in ranked:
        when = ago_text(now - m) if m is not None else "no files"
        L.append(f"  {when:>14}   {name}")
    L += ["", f"  {len(ranked)} campaign(s).  Detail: bt-check <strategy>",
          "=" * W]
    return "\n".join(L)


def render_all(out_dir: str | None = None,
               procs: Sequence[Proc] | None = None) -> str:
    """
    Running pipelines and recent campaigns in one view.

    Deliberately two SEPARATE sections rather than one merged table. A live run
    and a finished tree are different kinds of fact - one is a process, the
    other is what is on disk - and a row that blends them cannot say whether
    "Stage 3" means running now or stopped there.
    """
    table = ps_snapshot() if procs is None else list(procs)
    rows = running_rows(table or [])
    L = ["=" * W, "ALL PIPELINES", "=" * W, ""]
    if table is None:
        L.append("  the process table could not be read - live runs unknown.")
    elif rows:
        L.append(f"  RUNNING ({len(rows)}):")
        L.append(f"      {'PID':>7}  {'STAGE':>5}  {'ELAPSED':>11}  STRATEGY")
        for r in rows:
            L.append(f"      {r['pid']:>7}  {_stage_text(r['stage']):>5}  "
                     f"{elapsed_text(r['etimes']):>11}  {r['strategy']}")
    else:
        L.append("  RUNNING: nothing.")
    L.append("")
    L.append(render_campaign_list(out_dir))
    return "\n".join(L)


# ==========================================================================
# assembling one report
# ==========================================================================

def resolve_universe(flags: dict[str, Any],
                     campaign: Campaign) -> tuple[list[str], str]:
    """
    The contracts this run covers, and where the list came from.

    `--symbols` is usually an explicit comma list (the shell helper passes 24
    of them), in which case it IS the answer. `ALL` and the other group
    keywords are resolved by the stage against the lake catalog, which is not
    reachable from here without importing the reader - so the screen's own
    record is used instead, and when that does not exist yet the keyword is
    reported verbatim rather than turned into a number nobody can check.
    """
    raw = flags.get("symbols")
    if isinstance(raw, str) and "," in raw:
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
        return syms, "--symbols"

    blob = campaign.survivors()
    if blob:
        screened = [str(r.get("symbol")) for r in (blob.get("screen_results") or [])
                    if r.get("symbol")]
        if screened:
            return sorted(set(screened), key=screened.index), \
                "resolved by Stage 1"
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()], "--symbols (group, unresolved)"
    return [], "unknown"


def build_report(strategy: str | None = None,
                 out_dir: str | None = None,
                 procs: Sequence[Proc] | None = None) -> tuple[str, bool]:
    """
    The whole card as a string, plus whether a run is live.

    Returned as a string rather than printed so `--watch` can redraw it and so
    it can be asserted against in a test without a terminal.

    `procs` overrides the live process table, and exists for the tests. Without
    it every case that renders a card would read the REAL `ps` - so the suite
    would pass on an idle box and fail on the same box while a pipeline ran,
    which is precisely when somebody runs the gate. A test whose result depends
    on what else is running is not a test.
    """
    table = ps_snapshot() if procs is None else list(procs)
    ps_ok = table is not None
    procs = table or []
    orchestrators = find_orchestrators(procs)

    # An orchestrator is preferred, but a stage script run by hand is a real
    # pipeline too - and reporting IDLE while scan.py is burning a core would
    # be the single most misleading thing this tool could say.
    # When a strategy was NAMED, bind to ITS orchestrator - never to the
    # longest-running one. Concurrent campaigns are the whole reason this
    # argument exists, and `orchestrators[0]` under a name produced a card
    # headed with the requested strategy and filled with another run's PID,
    # elapsed time, CPU and timeframes. Every line read correctly and the card
    # described a different campaign; a strategy that was not running at all
    # was reported as Running.
    #
    # No match means this strategy is not live. That is a REPORT, not a
    # fallback: reading the artifacts of one campaign beside the process of
    # another is what produced the wrong card in the first place.
    if strategy:
        orchestrators = [o for o in orchestrators
                         if parse_cli_flags(o.args).get("strat") == strategy]
    orch = orchestrators[0] if orchestrators else None
    if orch is not None:
        stage_procs = [p for p in descendants(procs, orch.pid)
                       if p.script() in STAGE_OF_SCRIPT]
    else:
        stage_procs = find_stage_procs(procs)
        # Same wrong-provenance trap as the orchestrator above, one level down.
        # A hand-run stage carries its own --strat, and without this filter a
        # named strategy that is NOT running borrowed whichever stage script
        # happened to be alive - `bt-check t3_braid...` reported Running at
        # 99% CPU on a scan.py belonging to sma_momentum_crossover.
        if strategy:
            stage_procs = [p for p in stage_procs
                           if parse_cli_flags(p.args).get("strat") == strategy]

    stage_proc = stage_procs[0] if stage_procs else None
    live = orch is not None or stage_proc is not None

    flags = parse_cli_flags(orch.args if orch else
                            (stage_proc.args if stage_proc else ""))
    name = strategy or flags.get("strat")
    if not isinstance(name, str) or not name:
        name = newest_campaign(out_dir)

    if not name:
        return _no_campaign(out_dir, ps_ok), False

    since = orch.started if orch is not None else (
        stage_proc.started if stage_proc is not None else None)
    campaign = Campaign(name, out_dir or flags.get("out-dir"), since=since)
    return _render(campaign, orch, stage_proc, stage_procs, flags, ps_ok), live


def _no_campaign(out_dir: str | None, ps_ok: bool) -> str:
    root = Path(out_dir) if out_dir else artifacts_root() / "pipeline"
    L = ["=" * W, "BACKTEST PIPELINE STATUS", "=" * W,
         f"{'Overall Status':<19}: IDLE",
         f"{'Reason':<19}: no pipeline is running and {root} holds no campaign."]
    if not ps_ok:
        L.append(f"{'Note':<19}: the process table could not be read, so a "
                 f"running job would not")
        L.append(f"{'':<19}  have been seen either.")
    L.append("=" * W)
    return "\n".join(L)


def _row(label: str, value: str) -> str:
    return f"{label:<19}: {value}"


def _render(c: Campaign, orch: Proc | None, stage_proc: Proc | None,
            stage_procs: Sequence[Proc], flags: dict[str, Any],
            ps_ok: bool) -> str:
    live = orch is not None or stage_proc is not None
    L: list[str] = ["=" * W, "BACKTEST PIPELINE STATUS", "=" * W]

    # ---- headline -----------------------------------------------------
    L.append(_row("Strategy Name", c.strategy))

    if orch is not None:
        L.append(_row("Overall Status",
                      f"Running (PID: {orch.pid} | CPU: {orch.pcpu:.1f}% | "
                      f"Memory: {orch.pmem:.1f}%)"))
        L.append(_row("Elapsed Time", elapsed_text(orch.etimes)))
        L.append(_row("Started At", stamp_text(orch.started)))
    elif stage_proc is not None:
        L.append(_row("Overall Status",
                      f"Running one stage by hand — no run_pipeline.py "
                      f"(PID: {stage_proc.pid} | CPU: {stage_proc.pcpu:.1f}% | "
                      f"Memory: {stage_proc.pmem:.1f}%)"))
        L.append(_row("Elapsed Time", elapsed_text(stage_proc.etimes)))
        L.append(_row("Started At", stamp_text(stage_proc.started)))
    else:
        L.append(_row("Overall Status", "IDLE — no pipeline process is running"))
        if not ps_ok:
            L.append(_row("Note", "the process table could not be read; a "
                                  "running job would not have been seen."))

    universe, uni_source = resolve_universe(flags, c)
    tfs_flag = str(flags.get("tf") or "")
    tfs = [t.strip() for t in tfs_flag.split(",") if t.strip()]

    if universe:
        L.append(_row("Target Universe",
                      f"{len(universe)} Symbols ({truncated_list(universe)})"
                      + (f"  [{uni_source}]" if uni_source != "--symbols" else "")))
    if tfs:
        L.append(_row("Target Timeframes", ", ".join(tfs)))

    if not c.exists():
        L.append("")
        L.append(_row("Artifact Directory", f"{c.dir} — does not exist yet"))
        L.append("=" * W)
        return "\n".join(L)

    plan = c.stage2_plan()
    plan_tfs = [tf for tf, _ in plan]
    if not tfs and plan_tfs:
        L.append(_row("Target Timeframes",
                      ", ".join(plan_tfs) + "  [from Stage 1's survivors]"))

    s1 = c.stage1(universe, tfs or plan_tfs,
                  live=(stage_proc is not None and stage_proc.script() == "baseline.py"))

    live_stage = STAGE_OF_SCRIPT.get(stage_proc.script()) if stage_proc else None
    live_flags = parse_cli_flags(stage_proc.args) if stage_proc else {}
    live_tf = None
    if live_stage in (3, 4) and isinstance(live_flags.get("tf"), str):
        live_tf = live_flags["tf"]

    # `scan.py` is called with NO --tf by the orchestrator - that omission IS
    # how it inherits Stage 1's ragged pairs - so under the pipeline the
    # timeframe in flight is not on the command line, and it is taken as the
    # first one this run has not finished, which is how the sweep chooses it.
    #
    # An operator sweeping ONE timeframe by hand did put it there, and then it
    # is better evidence than the plan: a hand-run `scan.py --tf 5m` against a
    # directory where 3m is unfinished is on 5m, and inferring 3m from the plan
    # would name a timeframe nothing is sweeping. Only a single timeframe is
    # taken - `--tf 1m,5m` names two and picks out neither.
    s2_tf = None
    if live_stage == 2 and isinstance(live_flags.get("tf"), str):
        named = [t.strip() for t in live_flags["tf"].split(",") if t.strip()]
        s2_tf = named[0] if len(named) == 1 else None
    s2 = c.stage2(plan, s2_tf, live=(live_stage == 2))
    s3 = c.stage3(live=(live_stage == 3))
    s4 = c.stage4(live=(live_stage == 4))
    s5 = c.stage5()

    # ---- execution progress -------------------------------------------
    L.append("")
    L.append("--- Execution Progress ---")
    if stage_proc is not None:
        action = STAGE_ACTIONS.get(live_stage, stage_proc.script() or "?")
        detail = ""
        if live_stage == 2:
            cur = next((r for r in s2 if r.status == RUNNING), None)
            if cur is not None:
                if cur.current:
                    detail = (f" (Timeframe: {cur.tf} | Asset: "
                              f"[{cur.position}/{len(cur.planned)}] "
                              f"{cur.current})")
                else:
                    detail = f" (Timeframe: {cur.tf} | finishing)"
            elif not plan:
                detail = " (no Stage 1 handoff — the swept plan is not known)"
        elif live_stage in (3, 4) and live_tf:
            detail = f" (Timeframe: {live_tf})"
        elif live_stage == 1:
            detail = f" ({s1['detail']})"
        L.append(_row("Current Step", f"{action}{detail}"))
        L.append(_row("Stage Runtime", elapsed_text(stage_proc.etimes)))
        if len(stage_procs) > 1:
            L.append(_row("Also Running", ", ".join(
                sorted({p.script() or "?" for p in stage_procs[1:]}))))
    elif orch is not None:
        L.append(_row("Current Step",
                      "the orchestrator is between stages — no stage script "
                      "is running"))
    else:
        L.append(_row("Current Step", "nothing is running"))

    grid, grid_src, grid_fresh = c.grid_size()
    if grid:
        # The size is read off a parameter file the sweep has ALREADY written,
        # so on a fresh run it is the previous campaign's number until the
        # first winner lands. Saying "evaluating" of that would state a grid
        # this run has not been observed to use.
        if live_stage == 2 and grid_fresh:
            note = "evaluating"
        elif grid_fresh:
            note = "per sweep, last recorded"
        else:
            note = "per sweep, measured by an earlier run"
        L.append(_row("Current Grid",
                      f"{grid:,} parameter combinations {note}"
                      f"  [{grid_src}]"))

    # ---- stage breakdown ----------------------------------------------
    L.append("")
    L.append("--- Stage Breakdown ---")
    L.append(_row(STAGE_TITLES[1], f"{s1['status']:<9} ({s1['detail']})"))

    s2_done = [r.tf for r in s2 if r.status == COMPLETED]
    s2_run = [r for r in s2 if r.status == RUNNING]
    s2_part = [r.tf for r in s2 if r.status == PARTIAL]
    s2_wait = [r.tf for r in s2 if r.status == PENDING]
    if not plan:
        s2_status, s2_detail = PENDING, "no Stage 1 handoff to sweep"
    elif s2_run:
        s2_status = RUNNING
        bits = []
        if s2_done:
            bits.append(f"Completed: {', '.join(s2_done)}")
        bits.append(f"In Progress: {s2_run[0].tf} "
                    f"({len(s2_run[0].planned) - len(s2_run[0].pending)}"
                    f"/{len(s2_run[0].planned)})")
        if s2_part:
            bits.append(f"Part-done: {', '.join(s2_part)}")
        if s2_wait:
            bits.append(f"Pending: {', '.join(s2_wait)}")
        s2_detail = " | ".join(bits)
    elif s2_done and not s2_part and not s2_wait:
        s2_status = COMPLETED
        s2_detail = f"all {len(s2_done)} timeframe(s): {', '.join(s2_done)}"
    elif s2_done or s2_part:
        s2_status = PARTIAL
        bits = []
        if s2_done:
            bits.append(f"Completed: {', '.join(s2_done)}")
        if s2_part:
            bits.append(f"Part-done: {', '.join(s2_part)}")
        if s2_wait:
            bits.append(f"Never started: {', '.join(s2_wait)}")
        s2_detail = " | ".join(bits)
    else:
        s2_status, s2_detail = PENDING, f"queued: {', '.join(plan_tfs)}"
    L.append(_row(STAGE_TITLES[2], f"{s2_status:<9} ({s2_detail})"))
    carried = [f"{r.tf}: {len(r.carried)}" for r in s2 if r.carried]
    if carried:
        # Reported rather than counted. These tables exist and this run has not
        # rewritten them yet, and the difference between "swept" and "swept
        # last time" is the whole reason the split is drawn.
        L.append(f"{'':<19}  an earlier run's tables are still on disk and "
                 f"are not counted above — "
                 + ", ".join(carried) + " pair(s) awaiting a rewrite")

    audited = s3["audited"]
    if s3["status"] == COMPLETED and s3["certified"] is not None:
        s3_detail = (f"{s3['certified']} of {s3['targets']} certified across "
                     f"{len(audited)} audit(s)")
    elif len(audited):
        s3_detail = f"{len(audited.fresh)} pair(s) audited"
        if audited.stale:
            s3_detail += f" ({len(audited.stale)} more from an earlier run)"
    else:
        s3_detail = "nothing certified yet"
    L.append(_row(STAGE_TITLES[3], f"{s3['status']:<9} ({s3_detail})"))

    if s4["fresh"]:
        s4_detail = (f"{len(s4['fresh'])} contract(s) verified: "
                     f"{truncated_list(s4['fresh'])}")
    elif s4["stale"]:
        s4_detail = (f"{s4['runs']} lifecycle run(s) on disk, all from an "
                     f"earlier campaign")
    else:
        s4_detail = "no verify_<stamp>/ lifecycle run yet"
    L.append(_row(STAGE_TITLES[4], f"{s4['status']:<9} ({s4_detail})"))

    if s5["ran"]:
        s5_detail = (f"{s5['promoted']} promoted, {s5['failed']} failed"
                     + (f", commit {str(s5['commit'])[:12]}" if s5["commit"] else ""))
        s5_status = COMPLETED
    elif s5["incubator_ids"]:
        s5_detail = (f"{len(s5['incubator_ids'])} id(s) in the incubator: "
                     f"{truncated_list(s5['incubator_ids'], keep=3)} — no "
                     f"auto-promotion recorded for this campaign")
        s5_status = PARTIAL
    else:
        s5_detail = "nothing promoted"
        s5_status = PENDING
    L.append(_row("Stage 5 (Promote)", f"{s5_status:<9} ({s5_detail})"))

    # ---- artifacts ----------------------------------------------------
    L.append("")
    L.append("--- Artifacts & Storage ---")
    L.append(_row("Artifact Directory", f"{c.dir}/"))
    newest, newest_m = c.latest_artifact()
    if newest is not None:
        L.append(_row("Latest File Written",
                      f"{newest} ({ago_text(time.time() - (newest_m or 0))})"))
    else:
        L.append(_row("Latest File Written", "nothing has been written yet"))

    if orch is not None:
        discord = "Active (--report-discord enabled)" if flags.get("report-discord") \
            else "Off (no --report-discord on the command line)"
        promote = "on (certified rows register unattended)" \
            if flags.get("auto-promote") else "off"
        L.append(_row("Discord Alerts", discord))
        L.append(_row("Auto-Promote", promote))
        if flags.get("dry-run"):
            L.append(_row("Dry Run", "YES — the stages are printed, not run"))

    if not live:
        L.append("")
        L.append("--- Last Run ---")
        L.append(_row("Last Run Strategy", c.strategy))
        L.append(_row("Last Artifact At", stamp_text(newest_m)))
        L.append(_row("Reached", _reached(s1, s2_status, s3, s4, s5)))
        # Deliberately not the word SUCCESS. Nothing on disk records the exit
        # code of a run that has already gone, so a verdict drawn from which
        # files exist would be an inference presented as a fact - and a run
        # killed at hour eight leaves a tree indistinguishable from one that
        # finished, right up to the stage it died in.
        L.append(_row("Note", "no exit code is written to the tree, so this "
                              "is how far the"))
        L.append(f"{'':<19}  artifacts reach, not a pass/fail verdict. The "
                 f"gate audit is that.")

    L.append("=" * W)
    return "\n".join(L)


def _reached(s1: dict[str, Any], s2_status: str, s3: dict[str, Any],
             s4: dict[str, Any], s5: dict[str, Any]) -> str:
    """The furthest stage the tree shows evidence of, in plain English."""
    if s5["ran"]:
        return "Stage 5 — promotion recorded"
    if s4["fresh"] or s4["stale"]:
        return "Stage 4 — a lifecycle verification ran"
    if s3["status"] in (COMPLETED, PARTIAL) or len(s3["audited"]):
        return "Stage 3 — certification ran"
    if s2_status in (COMPLETED, PARTIAL, RUNNING):
        return f"Stage 2 — the sweep {'finished' if s2_status == COMPLETED else 'stopped part-way'}"
    if s1["status"] == COMPLETED:
        return "Stage 1 — the screen finished"
    return "Stage 1 — incomplete"


# ==========================================================================
# CLI
# ==========================================================================

def active_note(procs: Sequence[Proc] | None = None) -> str:
    """One line per live campaign, for the bottom of --help."""
    table = ps_snapshot() if procs is None else list(procs)
    if table is None:
        return "running now: the process table could not be read."
    rows = running_rows(table)
    if not rows:
        return "running now: nothing."
    return "running now:\n" + "\n".join(
        f"    {r['strategy']}  (pid {r['pid']}, stage "
        f"{_stage_text(r['stage'])}, {elapsed_text(r['etimes'])})"
        for r in rows)


def build_parser(epilog: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
        description="What the backtest pipeline is doing right now, in "
                    "English. Reads processes and artifacts; runs nothing.")
    # The positional and --strategy are ONE option with two spellings, folded
    # onto separate dests only so that supplying both can be caught rather than
    # silently resolved by argparse's last-wins. `bt-check <name>` is what
    # people type; `--strategy` predates it and stays.
    ap.add_argument("strategy_pos", nargs="?", default=None, metavar="STRATEGY",
                    help="report on this strategy (same as --strategy)")
    ap.add_argument("--strategy", "--strat", dest="strategy", default=None,
                    help="report on this strategy instead of the running one "
                         "(or, when nothing is running, the most recent)")
    ap.add_argument("-a", "--all", action="store_true",
                    help="running pipelines and recent campaigns in one view")
    ap.add_argument("-l", "--list", dest="list_campaigns", action="store_true",
                    help="list every strategy with an artifact tree")
    ap.add_argument("--out-dir", default=None,
                    help="an explicit artifacts directory, matching the same "
                         "flag on the stages")
    ap.add_argument("--watch", type=float, nargs="?", const=10.0, default=None,
                    metavar="SECONDS",
                    help="redraw every SECONDS until nothing is running "
                         "(default 10)")
    return ap


def _requested_strategy(args: argparse.Namespace) -> str | None:
    """
    The strategy the caller named, from either spelling.

    Both given and disagreeing is an ERROR rather than a precedence rule.
    Either answer would be defensible, which is the problem: somebody who
    types `bt-check A --strategy B` has two strategies in mind and needs to
    be told, not handed a card for one of them that reads exactly like a card
    for the other.
    """
    pos, flag = args.strategy_pos, args.strategy
    if pos and flag and pos != flag:
        raise SystemExit(f"bt-check: named two strategies, {pos!r} and "
                         f"{flag!r}. Pass one.")
    return pos or flag


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    # The live list is built ONLY for --help. It costs a `ps`, and every other
    # path already takes its own snapshot; paying for one on each invocation to
    # fill an epilog nobody is reading would be a waste on the tool whose whole
    # point is answering quickly.
    epilog = active_note() if ("-h" in raw or "--help" in raw) else None
    args = build_parser(epilog).parse_args(raw)

    if args.list_campaigns:
        print(render_campaign_list(args.out_dir))
        return 0
    if args.all:
        print(render_all(args.out_dir))
        return 0

    requested = _requested_strategy(args)
    if requested:
        # Resolve BEFORE reporting. Without this a typo silently produced an
        # empty card for a campaign that does not exist, which reads like a run
        # that has not started rather than like a name nobody recognises.
        resolved, suggestions = match_campaign(requested, args.out_dir)
        if resolved is None:
            print(f"bt-check: no campaign matches {requested!r}.",
                  file=sys.stderr)
            if suggestions:
                print("  did you mean:", file=sys.stderr)
                for n in suggestions:
                    print(f"      {n}", file=sys.stderr)
            else:
                print("  bt-check --list shows every campaign.",
                      file=sys.stderr)
            return 2
        args.strategy = resolved
    else:
        # Nobody named one and more than one campaign is live: no single card
        # describes the box, so show the table instead of silently picking the
        # longest-running of them.
        table = ps_snapshot()
        rows = running_rows(table or [])
        if len(rows) > 1:
            print(render_running_table(rows))
            return 0

    if args.watch is None:
        card, _live = build_report(args.strategy, args.out_dir)
        print(card)
        return 0

    try:
        while True:
            card, live = build_report(args.strategy, args.out_dir)
            # Redraw in place. Scrolled, a card reprinted every ten seconds
            # buries the line that changed under twenty copies of the ones
            # that did not.
            print("\033[2J\033[H", end="")
            print(card, flush=True)
            if not live:
                return 0
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
