#!/usr/bin/env python3
"""
tools/tool_backtest_runner.py - queue sweeps in market hours, run them off-peak.

Location:  ~/src/trading/tools/tool_backtest_runner.py

    python3 tools/tool_backtest_runner.py status
    python3 tools/tool_backtest_runner.py queue --note "ES 30m sweep" -- \
        backtest/run.py --symbol ES --timeframe 30m --start 2013-01-01 --end 2022-12-31
    python3 tools/tool_backtest_runner.py list
    python3 tools/tool_backtest_runner.py drain              # prints; runs nothing
    python3 tools/tool_backtest_runner.py drain --execute    # actually runs

The Backtest agent's admission control. This box runs the live loop; a sweep
that saturates it between 08:00 and 17:00 ET competes with the process that is
placing orders. So work submitted in market hours is QUEUED, and only drained
outside that window, wrapped in the cgroup the architecture specifies:

    systemd-run --scope -p CPUQuota=75% nice -n 19 <argv>

THIS DOES NOT RUN ANYTHING UNLESS YOU SAY --execute
---------------------------------------------------
`drain` prints the exact commands it would run and exits. Nothing starts
without `--execute`. That default is deliberate and it is not timidity:
`CLAUDE.md` puts a hard boundary around starting a backtest, because a sweep
takes the human out of the loop at exactly the point the promotion evidence is
being generated. An agent may prepare, order and describe the work; a person
decides that it starts. `--execute` is how a person says so, and `--execute`
issued inside market hours is still refused.

WHAT MAY BE QUEUED
------------------
An argv LIST, never a shell string, and its first element must be one of the
entrypoints in `ALLOWED_ENTRYPOINTS`. Nothing is passed through a shell, so a
queued argument cannot become a command. A queue that an agent can write to
and that executes arbitrary strings is a remote shell with extra steps; this
one can start a sweep and nothing else.

The queue is a JSON file written whole and atomically on each mutation - it
holds tens of entries, and a partial write is the one state that would make it
run a job twice.

WHAT THIS DOES NOT DO
---------------------
**It does not know whether the lake is mounted, or whether a sweep succeeded.**
It starts a process under a scope and records the exit code. Reading the run's
output and judging it is `bt-status`'s job and a human's.

**It is not a scheduler.** Nothing drains the queue on its own; a timer or an
operator calls `drain --execute`.

Exit codes:  0 fine · 1 refused (market hours, or nothing to do) ·
             2 could not run · 3 a drained job failed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
QUEUE_PATH = REPO_ROOT / "data" / "backtest_queue.json"

ET = ZoneInfo("America/New_York")

#: The protected window. Inside it, sweeps queue and never start.
PEAK_START = time(8, 0)
PEAK_END = time(17, 0)

#: The cgroup wrapper the architecture specifies for off-peak compute.
SCOPE_WRAPPER = ("systemd-run", "--scope", "-p", "CPUQuota=75%", "nice", "-n", "19")

#: Only these may be queued. A path outside this set is refused at submission,
#: so a bad entry never reaches the drain.
ALLOWED_ENTRYPOINTS = (
    "backtest/run.py",
    "run_pipeline.py",
    "scripts/report_strategy_performance.py",
)

STATE_QUEUED = "QUEUED"
STATE_DONE = "DONE"
STATE_FAILED = "FAILED"


class RunnerError(RuntimeError):
    """The request could not be accepted."""


# --------------------------------------------------------------------------
# the clock
# --------------------------------------------------------------------------

def now_et(now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(ET)


def in_peak_window(now: datetime | None = None) -> bool:
    """
    True inside 08:00-17:00 ET on a weekday.

    Weekends are off-peak in full: the live loop is not trading and the box is
    free. The window is checked in ET rather than UTC because it is a
    statement about the trading day, and a UTC window would drift by an hour
    across the DST boundary - starting the sweep an hour into the session
    twice a year, which is exactly when nobody is looking for it.
    """
    moment = now_et(now)
    if moment.weekday() >= 5:
        return False
    return PEAK_START <= moment.timetz().replace(tzinfo=None) < PEAK_END


def window_note(now: datetime | None = None) -> str:
    moment = now_et(now)
    state = "PEAK — sweeps queue" if in_peak_window(now) else "off-peak — sweeps may run"
    return (f"{moment:%Y-%m-%d %H:%M:%S %Z} ({moment:%a}) · {state} "
            f"(peak {PEAK_START:%H:%M}-{PEAK_END:%H:%M} ET, weekdays)")


# --------------------------------------------------------------------------
# the queue
# --------------------------------------------------------------------------

def load_queue(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "jobs": []}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(
            f"{path}: unreadable ({exc}). Refusing to treat an unparseable "
            f"queue as an empty one — 'nothing is pending' is the belief that "
            f"loses queued work silently.") from None
    blob.setdefault("jobs", [])
    if not isinstance(blob["jobs"], list):
        raise RunnerError(f"{path}: 'jobs' is not a list")
    return blob


def save_queue(path: Path, blob: dict[str, Any]) -> None:
    """Whole-document atomic write; a torn queue is a job that runs twice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent),
                                   prefix=".backtest_queue.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def validate_argv(argv: list[str]) -> list[str]:
    """
    Accept an argv list whose entrypoint is on the allowlist.

    A leading `python`/`python3`/venv-python is stripped: the drain supplies
    the interpreter itself, so accepting one here would let a submission
    choose a different Python than the pinned venv.
    """
    if not argv:
        raise RunnerError("nothing to queue — pass the command after `--`")
    argv = list(argv)
    first = Path(argv[0]).name
    if first in ("python", "python3", "python3.13"):
        argv = argv[1:]
        if not argv:
            raise RunnerError("an interpreter with no script is not a sweep")

    script = argv[0]
    try:
        relative = Path(script).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        relative = script
    if relative not in ALLOWED_ENTRYPOINTS:
        raise RunnerError(
            f"{script!r} is not a queueable entrypoint. Allowed: "
            f"{', '.join(ALLOWED_ENTRYPOINTS)}. The queue starts sweeps and "
            f"nothing else — an arbitrary command here would make it a shell.")
    return [relative] + argv[1:]


def add_job(blob: dict[str, Any], argv: list[str], note: str) -> dict[str, Any]:
    job = {
        "id": uuid.uuid4().hex[:8],
        "argv": validate_argv(argv),
        "note": note,
        "state": STATE_QUEUED,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "scope": None,
    }
    blob["jobs"].append(job)
    return job


def pending(blob: dict[str, Any]) -> list[dict[str, Any]]:
    return [j for j in blob["jobs"] if j.get("state") == STATE_QUEUED]


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def build_command(job: dict[str, Any], wrap: bool = True) -> list[str]:
    """The exact argv a drain would spawn."""
    inner = [str(VENV_PYTHON), str(REPO_ROOT / job["argv"][0]), *job["argv"][1:]]
    if not wrap:
        return inner
    return [*SCOPE_WRAPPER, *inner]


def scope_available() -> bool:
    return shutil.which("systemd-run") is not None


def render_command(argv: list[str]) -> str:
    import shlex                                                # noqa: PLC0415
    return " ".join(shlex.quote(a) for a in argv)


def drain(blob: dict[str, Any], path: Path, execute: bool, limit: int,
          now: datetime | None = None) -> tuple[int, list[str]]:
    """
    Run (or print) the queued jobs. Returns `(exit_code, lines)`.

    Refuses inside the peak window even with `--execute`: the whole point of
    the queue is that market hours belong to the live loop, and an override
    that ignores it would make the window advisory.
    """
    lines: list[str] = [window_note(now)]
    jobs = pending(blob)
    if not jobs:
        lines.append("  the queue is empty — nothing to drain.")
        return 1, lines

    if in_peak_window(now):
        lines.append(f"  REFUSED — {len(jobs)} job(s) pending, but it is "
                     f"market hours. They stay queued.")
        return 1, lines

    wrap = scope_available()
    if not wrap:
        lines.append("  NOTE systemd-run is not on PATH; the CPUQuota/nice "
                     "scope cannot be applied. Jobs are shown UNWRAPPED and "
                     "will not be executed.")

    selected = jobs[:limit] if limit > 0 else jobs
    lines.append(f"  {len(selected)} of {len(jobs)} pending job(s):")

    if not execute or not wrap:
        for job in selected:
            lines.append(f"    [{job['id']}] {job.get('note') or '—'}")
            lines.append(f"      {render_command(build_command(job, wrap))}")
        lines.append("")
        lines.append("  Nothing was started. Re-run with --execute to start "
                     "them." if wrap else
                     "  Nothing was started (no systemd-run).")
        return 0, lines

    worst = 0
    for job in selected:
        command = build_command(job)
        lines.append(f"    [{job['id']}] {job.get('note') or '—'}")
        lines.append(f"      {render_command(command)}")
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        job["scope"] = render_command(SCOPE_WRAPPER)
        save_queue(path, blob)
        try:
            completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
            code = completed.returncode
        except OSError as exc:
            code = -1
            lines.append(f"      could not start: {exc}")
        job["returncode"] = code
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        job["state"] = STATE_DONE if code == 0 else STATE_FAILED
        save_queue(path, blob)
        lines.append(f"      -> {job['state']} (exit {code})")
        worst = max(worst, 0 if code == 0 else 3)
    return worst, lines


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------

def format_list(blob: dict[str, Any], now: datetime | None = None) -> str:
    lines = ["BACKTEST QUEUE", f"  {window_note(now)}",
             f"  {QUEUE_PATH}"]
    jobs = blob.get("jobs", [])
    if not jobs:
        lines.append("  the queue is empty.")
        return "\n".join(lines)
    lines.append("")
    lines.append(f"  {'ID':<9} {'STATE':<7} {'RC':>4}  {'QUEUED':<20} NOTE / ARGV")
    for job in jobs:
        rc = "—" if job.get("returncode") is None else str(job["returncode"])
        lines.append(f"  {job['id']:<9} {job.get('state', '?'):<7} {rc:>4}  "
                     f"{str(job.get('queued_at', ''))[:19]:<20} "
                     f"{job.get('note') or '—'}")
        lines.append(f"      {render_command(job.get('argv', []))}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Queue backtest sweeps during market hours; drain them "
                    "off-peak under a CPU-capped scope.")
    parser.add_argument("--queue-file", default=str(QUEUE_PATH),
                        help=f"queue file (default {QUEUE_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="market window and queue depth")
    sub.add_parser("list", help="every job in the queue")

    queue_cmd = sub.add_parser("queue", help="submit a sweep")
    queue_cmd.add_argument("--note", default="", help="what this sweep is for")
    queue_cmd.add_argument("argv", nargs=argparse.REMAINDER,
                           help="the command, after `--`")

    drain_cmd = sub.add_parser("drain", help="run the queued sweeps (off-peak)")
    drain_cmd.add_argument("--execute", action="store_true",
                           help="actually start them; without this the "
                                "commands are printed and nothing runs")
    drain_cmd.add_argument("--max", type=int, default=0,
                           help="drain at most this many (0 = all)")

    remove_cmd = sub.add_parser("remove", help="drop a job by id")
    remove_cmd.add_argument("job_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.queue_file)

    try:
        blob = load_queue(path)
    except RunnerError as exc:
        print(f"backtest runner could not run: {exc}", file=sys.stderr)
        return 2

    if args.command == "status":
        print("BACKTEST RUNNER")
        print(f"  {window_note()}")
        print(f"  queue      {path}")
        print(f"  pending    {len(pending(blob))} of {len(blob['jobs'])}")
        print(f"  scope      {'systemd-run available' if scope_available() else 'systemd-run NOT on PATH'}")
        print(f"  interpreter{'':1}{VENV_PYTHON}"
              f"{'' if VENV_PYTHON.exists() else '  (MISSING)'}")
        return 0

    if args.command == "list":
        print(format_list(blob))
        return 0

    if args.command == "queue":
        submitted = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        try:
            job = add_job(blob, submitted, args.note)
        except RunnerError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        save_queue(path, blob)
        print(f"queued [{job['id']}] {job.get('note') or '—'}")
        print(f"  {render_command(build_command(job))}")
        print(f"  {window_note()}")
        if not in_peak_window():
            print("  It is off-peak — `drain --execute` will start it.")
        return 0

    if args.command == "remove":
        before = len(blob["jobs"])
        blob["jobs"] = [j for j in blob["jobs"] if j.get("id") != args.job_id]
        if len(blob["jobs"]) == before:
            print(f"no job with id {args.job_id!r}", file=sys.stderr)
            return 2
        save_queue(path, blob)
        print(f"removed {args.job_id}")
        return 0

    if args.command == "drain":
        code, lines = drain(blob, path, args.execute, args.max)
        print("\n".join(lines))
        return code

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
