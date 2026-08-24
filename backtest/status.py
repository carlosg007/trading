#!/usr/bin/env python3
"""
status.py - what the multi-asset batch is doing right now.

Location:  ~/src/trading/backtest/status.py

    python3 backtest/status.py             # one snapshot
    python3 backtest/status.py --watch 5   # refresh every 5 seconds
    bt-status                              # the alias

Two halves of the same file. `JobTracker` is what `backtest/run.py` writes
while a batch is running; `main()` is what a human reads while it runs. They
share one schema so the reader cannot drift from the writer.

Why a file rather than a log tail
---------------------------------
A batch started with `--bg` is detached: its terminal is gone and its stdout is
a log file nobody is watching. The question a human actually has is "which
symbol is it on and how many are left", and answering that by parsing a
free-form log means the answer breaks whenever a print statement changes.
`active_job.json` is a small declared structure, rewritten after every symbol,
so the answer is a read rather than a parse.

The write is atomic - a temporary file in the same directory, then
`os.replace`. A reader that catches the writer mid-`json.dump` gets a
truncated file and a JSONDecodeError, which on a run that takes hours would
show up as an intermittent crash in the status tool and nowhere else.

`active_job.json` is deliberately a SINGLE well-known path, overwritten by each
new batch. It is a progress indicator, not evidence: the run's own directory
gets `job.json` at the end, and that is the copy a later reader should trust.
The distinction matters because the artifacts tree is append-only by design -
a re-run must never overwrite the numbers an earlier promotion decision was
made on - and a live progress file is the one thing in it that is not a record.

Staleness
---------
A job whose state is RUNNING but whose PID is gone was killed, or the machine
went down. The reader says STALE rather than showing a frozen progress bar
forever, because a bar sitting at 12/27 looks identical whether the run is
slow or dead.
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
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_JOB_FILE = Path(
    os.environ.get("BT_ACTIVE_JOB", "/mnt/backtest/artifacts/active_job.json"))

RUNNING = "RUNNING"
DONE = "DONE"
FAILED = "FAILED"


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------
class JobTracker:
    """
    The batch runner's progress file.

    One tracker per batch. `start()` on the way in, `start_symbol()` before
    each symbol's bars are read, `finish_symbol()` with that symbol's
    mini-scorecard, `finish()` on the way out. Every one of those rewrites the
    file, so a reader is never more than one symbol behind.

    Nothing here raises. A batch that has run for two hours is not thrown away
    because the progress file could not be written - the failure is printed and
    the run continues, which is the same rule the report writer follows.
    """

    def __init__(self,
                 job_id: str,
                 strategy: str,
                 symbols: list[str],
                 timeframe: str,
                 artifact_dir: str | Path,
                 scan: bool = False,
                 ml: bool = False,
                 path: str | Path | None = None) -> None:
        self.path = Path(path or DEFAULT_JOB_FILE)
        self.started = time.time()
        self.state: dict[str, Any] = {
            "job_id": job_id,
            "pid": os.getpid(),
            "strategy": strategy,
            "timeframe": timeframe,
            "artifact_dir": str(artifact_dir),
            "symbols": list(symbols),
            "total_symbols": len(symbols),
            "completed_symbols": 0,
            "current_symbol": None,
            "scan": bool(scan),
            "ml": bool(ml),
            "state": RUNNING,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "started_epoch": self.started,
            "updated_utc": None,
            "finished_utc": None,
            "results": [],
            "error": None,
        }

    # -- writing --------------------------------------------------------
    def _write(self) -> None:
        self.state["updated_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        self.state["elapsed_s"] = round(time.time() - self.started, 1)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.state, indent=2, default=str),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as e:                                  # noqa: BLE001
            print(f"[!] could not update {self.path}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    def start(self) -> None:
        self._write()

    def start_symbol(self, symbol: str) -> None:
        self.state["current_symbol"] = symbol
        self._symbol_started = time.time()
        self._write()

    def finish_symbol(self, symbol: str, row: dict[str, Any]) -> None:
        """
        Record one symbol's outcome.

        `row` is the mini-scorecard - status, Sharpe, profit factor, trades,
        max drawdown, Gate 1 - the same numbers the leaderboard row carries, so
        the two cannot disagree about what happened.
        """
        elapsed = round(time.time() - getattr(self, "_symbol_started",
                                              self.started), 1)
        self.state["results"].append({"symbol": symbol, "elapsed_s": elapsed,
                                      **row})
        self.state["completed_symbols"] = len(self.state["results"])
        self.state["current_symbol"] = None
        self._write()

    def finish(self, state: str = DONE, error: str | None = None) -> None:
        self.state["state"] = state
        self.state["error"] = error
        self.state["current_symbol"] = None
        self.state["finished_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        self._write()

    def snapshot(self, dest: str | Path) -> Path | None:
        """
        Copy the finished job into the run's own directory as `job.json`.

        `active_job.json` is overwritten by the next batch; this copy is the
        one that stays with the reports it describes.
        """
        try:
            dest = Path(dest) / "job.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(self.state, indent=2, default=str),
                            encoding="utf-8")
            return dest
        except Exception as e:                                  # noqa: BLE001
            print(f"[!] could not write job.json: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            return None


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------
def read_job(path: str | Path | None = None) -> dict[str, Any] | None:
    """The job file, or None when there isn't one / it is mid-write."""
    p = Path(path or DEFAULT_JOB_FILE)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Only reachable if os.replace is not atomic on this filesystem. Say so
        # rather than crashing - the next poll will succeed.
        return {"_unreadable": str(p)}


def pid_alive(pid: Any) -> bool:
    """True when the writing process still exists."""
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def format_elapsed(seconds: float) -> str:
    """`2h 14m 03s`, `14m 03s`, `43s`."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "?"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


FILL, EMPTY = "█", "░"


def progress_bar(done: int, total: int, width: int = 40) -> str:
    """
    `[████░░░░] 12/27  44.4%`.

    Block characters rather than ASCII hashes: at a glance the bar is a bar
    rather than a run of punctuation, and both glyphs occupy one cell in every
    monospace font this will be read in, so the bar's width does not change as
    it fills.
    """
    total = max(0, int(total or 0))
    done = max(0, min(int(done or 0), total))
    if total == 0:
        return f"[{EMPTY * width}] 0/0"
    filled = int(round(width * done / total))
    pct = 100.0 * done / total
    return (f"[{FILL * filled}{EMPTY * (width - filled)}] "
            f"{done}/{total}  {pct:5.1f}%")


def _fmt(value: Any, spec: str = "{:.2f}") -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "n/a"
    if math.isinf(f):
        return "inf"
    return spec.format(f)


def format_status(job: dict[str, Any] | None,
                  path: str | Path | None = None) -> str:
    """The whole readout as a string, so it can be tested without a terminal."""
    W = 78
    p = Path(path or DEFAULT_JOB_FILE)
    L: list[str] = []
    add = L.append

    add("=" * W)
    add("BATCH JOB STATUS")
    add("=" * W)

    if job is None:
        add(f"  No job file at {p}.")
        add("  Nothing has run yet, or the artifacts mount is not available.")
        add("=" * W)
        return "\n".join(L)
    if job.get("_unreadable"):
        add(f"  {job['_unreadable']} was mid-write. Try again.")
        add("=" * W)
        return "\n".join(L)

    state = job.get("state", "?")
    alive = pid_alive(job.get("pid"))
    if state == RUNNING and not alive:
        # A bar frozen at 12/27 looks the same whether the run is slow or dead.
        state = f"{RUNNING} · STALE (pid {job.get('pid')} is gone)"

    total = job.get("total_symbols", 0)
    done = job.get("completed_symbols", 0)
    started = job.get("started_epoch")
    elapsed = (time.time() - started) if (started and job.get("state") == RUNNING) \
        else job.get("elapsed_s")

    add(f"  Job        : {job.get('job_id', '?')}")
    add(f"  Strategy   : {job.get('strategy', '?')}   "
        f"TF: {job.get('timeframe', '?')}   "
        f"scan: {'on' if job.get('scan') else 'off'}   "
        f"ML: {'on' if job.get('ml') else 'off'}")
    add(f"  State      : {state}")
    add(f"  Started    : {job.get('started_utc', '?')} UTC")
    add(f"  Elapsed    : {format_elapsed(elapsed)}")
    add(f"  Artifacts  : {job.get('artifact_dir', '?')}")
    add("")
    add(f"  {progress_bar(done, total)}")

    current = job.get("current_symbol")
    if current:
        add(f"  Evaluating : {current}")
    elif job.get("state") == RUNNING:
        add("  Evaluating : (between symbols)")

    # A remaining-time estimate from the symbols that have actually finished.
    # Not shown on the first symbol: one sample is not a rate, and a wildly
    # wrong ETA is worse than none.
    results = job.get("results") or []
    if job.get("state") == RUNNING and len(results) >= 2 and total > done:
        per = sum(float(r.get("elapsed_s") or 0) for r in results) / len(results)
        add(f"  Remaining  : ~{format_elapsed(per * (total - done))} "
            f"at {format_elapsed(per)} per symbol so far")

    if job.get("error"):
        add("")
        add(f"  ERROR: {job['error']}")

    add("")
    add("-" * W)
    add("COMPLETED SYMBOLS")
    add("-" * W)
    if not results:
        add("  (none yet)")
    else:
        add(f"  {'Symbol':<8}{'Status':<9}{'Sharpe':>8}{'PF':>8}{'Trades':>9}"
            f"{'maxDD%':>9}{'Gate 1':>9}{'Time':>10}")
        add("  " + "-" * (W - 4))
        for r in results:
            add(f"  {str(r.get('symbol', '?')):<8}"
                f"{str(r.get('status', '?')):<9}"
                f"{_fmt(r.get('sharpe')):>8}"
                f"{_fmt(r.get('profit_factor')):>8}"
                f"{_fmt(r.get('trades'), '{:,.0f}'):>9}"
                f"{_fmt(r.get('max_drawdown_pct')):>9}"
                f"{str(r.get('gate1', '—')):>9}"
                f"{format_elapsed(r.get('elapsed_s')):>10}")
            if r.get("error"):
                add(f"      ! {r['error']}")

    add("")
    add(f"  Leaderboard: {job.get('artifact_dir', '?')}/summary_leaderboard.csv")
    add("=" * W)
    return "\n".join(L)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Show the progress of the running multi-asset batch.")
    ap.add_argument("--file", default=None,
                    help=f"job file to read (default: {DEFAULT_JOB_FILE}, "
                         f"overridable with $BT_ACTIVE_JOB)")
    ap.add_argument("--watch", type=float, nargs="?", const=5.0, default=None,
                    metavar="SECONDS",
                    help="refresh every SECONDS until the job leaves RUNNING "
                         "(default 5)")
    ap.add_argument("--json", action="store_true",
                    help="print the raw job file instead of the readout")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.file or DEFAULT_JOB_FILE

    if args.json:
        job = read_job(path)
        print(json.dumps(job, indent=2, default=str) if job else "null")
        return 0 if job else 1

    if args.watch is None:
        job = read_job(path)
        print(format_status(job, path))
        return 0 if job else 1

    try:
        while True:
            job = read_job(path)
            # Redraw in place rather than scrolling: a 27-symbol table
            # reprinted every 5 seconds buries the one at the bottom.
            print("\033[2J\033[H", end="")
            print(format_status(job, path))
            if not job or job.get("state") != RUNNING or not pid_alive(job.get("pid")):
                return 0 if job else 1
            time.sleep(max(0.5, args.watch))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
