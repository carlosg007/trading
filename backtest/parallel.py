"""
backtest/parallel.py - run one independent unit of work per (symbol, timeframe)
across several processes, and stop before the box does.

Location: ~/src/trading/backtest/parallel.py

WHAT THIS PARALLELISES, AND WHAT IT DELIBERATELY DOES NOT
========================================================
It parallelises the LOOP, never the math. Stage 1 walks
`[(sym, tf) for sym in symbols for tf in timeframes]` and each iteration is a
complete, independent simulation on that contract's own multiplier, tick size
and commission. Nothing is shared between iterations, so running eight of them
in eight processes produces the same rows as running them one after another -
`tests/test_parallel.py` pins that equality on a synthetic fixture.

**It does NOT broadcast several symbols into one `vbt.Portfolio.from_signals`
call, and that is not an oversight.** One call takes one set of costs and one
notion of what a point is worth. `backtest/specs.py` gives ES a 50x multiplier
and CL a 1000x; a blended call has to pick one, and the equity curve it returns
still looks entirely plausible. The concatenated frame `mdlib.lake.get_bars`
returns interleaves instruments for the same reason a `rolling(200)` over it
averages 27 contracts. The speedup here comes from using 16 cores instead of 1,
which is the honest 8-12x - not from collapsing 27 different instruments into
one array.

WHY THE WORKER COUNT IS SIZED ON MEMORY AND NOT ON CORES
========================================================
This VM has 16 vCPUs and 24.9 GiB. The binding constraint is the second number.
One configuration's peak is its bars plus the signal masks built over them; a
16-year 1-minute contract is 5.6M rows, and `backtest/data_loader.py` already
documents a Stage 2 sweep of one such contract against a 432-cell grid at
9.0 GiB. Eight of those at once is not a faster screen, it is the OOM killer
choosing a victim by a score this process does not control.

So `resolve_jobs` takes the MINIMUM of three bounds - cores minus a reserve,
available RAM divided by a per-worker estimate, and the number of units there
actually are - and prints which one bound it. A run that silently used two
workers on a 16-core box because memory was tight would read as a run that
failed to parallelise.

**The default is 1.** Parallelism is opt-in through `--jobs`, because a screen
that changes its worker count between two runs of the same grid is a screen
whose timings are not comparable, and because every existing artifact in
`/mnt/backtest` was produced serially.

THE MEMORY GUARD UNDER N WORKERS
================================
`backtest/memory_guard.py`'s HALT trigger is SYSTEM-WIDE. Serially that is a
run declining to start one more configuration; with N workers it is N processes
all seeing the same 90% and all raising at once, which is correct - the box is
full whoever filled it - but it means a halt must reach the parent and cancel
what has not started. `MemorySafetyException` is therefore re-raised out of the
worker, caught here, and turned into `halted=True` with every pending unit left
UNSUBMITTED rather than merely unfinished. The caller exits 75 (EX_TEMPFAIL) on
it exactly as the serial path does.

Each worker pins its own BLAS/OpenMP thread count to 1 before importing numpy.
Sixteen processes each opening a 16-thread pool is 256 threads fighting over 16
cores, which is measurably slower than serial.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

#: Reserved cores, so a full-tilt screen leaves the box usable and does not
#: starve the parent's own report writing.
CORE_RESERVE = 2

#: Per-worker RAM estimate, GiB. Deliberately generous: the cost of
#: over-estimating is one fewer worker, and the cost of under-estimating is the
#: OOM killer. A 1-minute contract's bars alone are ~0.5 GiB before any mask.
DEFAULT_WORKER_GIB = 3.0

#: What the serial path returns, so `--jobs 1` stays the documented default.
SERIAL = 1


def available_gib() -> float:
    """
    Memory this process could actually obtain, GiB.

    MemAvailable, not MemFree: the kernel's own estimate of what is obtainable
    without swapping, which counts reclaimable page cache. MemFree on a box
    that has just read 8 GiB of parquet reads near zero while 20 GiB is in
    fact available, and sizing on it would refuse to parallelise at all.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def resolve_jobs(requested: int | str | None, n_units: int, *,
                 worker_gib: float = DEFAULT_WORKER_GIB) -> tuple[int, str]:
    """
    How many workers to run, and the ONE bound that decided it.

    `requested` is an integer, or "auto" to derive one. An explicit integer is
    honoured up to the unit count - an operator who says 12 on a quiet box gets
    12 - because the memory estimate is an estimate and the person at the
    console can see the machine. It is still clamped to `n_units`: eight
    workers for three configurations is five processes that start, find nothing
    to do and exit.

    Returns `(jobs, reason)`. The reason is printed, never inferred, because
    "2 workers" is the same string whether memory bound it or the operator
    asked for 2.
    """
    if n_units <= 0:
        return SERIAL, "no units to run"

    raw = str(requested).strip().lower() if requested is not None else "1"
    if raw in ("", "1"):
        return SERIAL, "serial (--jobs 1, the default)"

    if raw == "auto":
        cores = max(1, (os.cpu_count() or 1) - CORE_RESERVE)
        avail = available_gib()
        by_mem = max(1, int(avail // worker_gib)) if avail > 0 else 1
        jobs = min(cores, by_mem, n_units)
        if jobs == n_units:
            reason = f"auto: {n_units} unit(s), fewer than any other bound"
        elif jobs == by_mem <= cores:
            reason = (f"auto: memory-bound — {avail:.1f} GiB available / "
                      f"{worker_gib:.1f} GiB per worker")
        else:
            reason = (f"auto: core-bound — {os.cpu_count()} cores less "
                      f"{CORE_RESERVE} reserved")
        return jobs, reason

    try:
        asked = int(raw)
    except ValueError:
        raise ValueError(
            f"--jobs {requested!r} is neither an integer nor 'auto'") from None
    if asked < 1:
        raise ValueError(f"--jobs must be >= 1; got {asked}")
    jobs = min(asked, n_units)
    if jobs < asked:
        return jobs, (f"explicit --jobs {asked}, clamped to {n_units} unit(s)")
    return jobs, f"explicit --jobs {asked}"


def _init_worker() -> None:
    """
    Pin every native thread pool to 1 in the child, before numpy is touched.

    These are read when the native library LOADS, so setting them after an
    import is a no-op that looks like throttling - see
    `backtest/memory_guard.py`. A fresh process is the one place the variable
    can still bind, which is why this is an initializer and not a call.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "BT_ML_THREADS"):
        os.environ[var] = "1"


@dataclass
class Progress:
    """
    A one-line progress record an operator can watch, and a log can hold.

    ETA is from COMPLETED WORK, never from a per-unit constant: configurations
    differ by an order of magnitude (a 1-minute contract against a 1-hour one),
    so a fixed estimate is wrong in the direction that matters - it promises an
    hour on a screen that takes six. With `n` done out of `total` it is the
    mean completed duration times what is left, divided by the worker count.
    """
    total: int
    jobs: int
    started: float = field(default_factory=time.monotonic)
    done: int = 0
    failed: int = 0

    def tick(self, ok: bool = True) -> None:
        self.done += 1
        if not ok:
            self.failed += 1

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def eta_seconds(self) -> float | None:
        """None until something has finished - an ETA from zero samples is a
        guess presented as a measurement."""
        if self.done <= 0 or self.done >= self.total:
            return None
        per_unit = self.elapsed / self.done
        return per_unit * (self.total - self.done) / max(1, self.jobs)

    def line(self, label: str = "") -> str:
        pct = 100.0 * self.done / self.total if self.total else 100.0
        eta = self.eta_seconds()
        eta_s = f"ETA {_hms(eta)}" if eta is not None else "ETA --:--"
        fail = f" · {self.failed} error(s)" if self.failed else ""
        tail = f" · {label}" if label else ""
        return (f"[{self.done}/{self.total} {pct:5.1f}%] "
                f"elapsed {_hms(self.elapsed)} · {eta_s} · "
                f"{self.jobs} worker(s){fail}{tail}")


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    s = int(max(0.0, seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


@dataclass
class MapResult:
    """
    What a parallel run produced, and how it ended.

    `results` is keyed by the unit's index so the caller can restore SUBMISSION
    order: completion order is arbitrary under a pool, and a leaderboard whose
    rows arrive in whichever order the scheduler finished them is a different
    table on every run of the same screen.

    `halted` is kept apart from `errors` on purpose. A halt means the box ran
    out of memory and the remaining units were never attempted; an error means
    a configuration was attempted and failed. Collapsing them would report an
    unfinished screen as a completed one with some bad contracts in it.
    """
    results: dict[int, Any] = field(default_factory=dict)
    errors: dict[int, BaseException] = field(default_factory=dict)
    halted: bool = False
    halt_error: BaseException | None = None
    unsubmitted: list[int] = field(default_factory=list)
    progress: Progress | None = None

    def ordered(self) -> list[Any]:
        """Completed results in SUBMISSION order, gaps dropped."""
        return [self.results[i] for i in sorted(self.results)]


def map_units(fn: Callable[..., Any],
              units: Sequence[Any],
              *,
              jobs: int,
              halt_exceptions: tuple[type[BaseException], ...] = (),
              on_result: Callable[[int, Any, Progress], None] | None = None,
              on_error: Callable[[int, BaseException, Progress], None] | None = None,
              label_of: Callable[[Any], str] | None = None,
              ) -> MapResult:
    """
    Run `fn(unit)` over `units`, serially at `jobs == 1` and pooled above it.

    **`jobs == 1` does not create a pool.** It calls `fn` in this process, in
    order, which is what makes `--jobs 1` bit-identical to the loop this
    replaced rather than merely equivalent to it - no pickling, no fork, no
    re-import of the strategy module, and a traceback that points at the real
    frame.

    `halt_exceptions` are the ones that stop the whole run rather than marking
    one unit bad - `MemorySafetyException` in practice. On one, no further unit
    is SUBMITTED and the indices that never ran come back in `unsubmitted`.

    `on_result` / `on_error` fire in the parent as each unit lands, which is
    where the caller rewrites its partial report: a screen killed at 14 of 108
    must still leave a complete report of 14, and under a pool that property
    only holds if the parent writes on every completion rather than at the end.
    """
    prog = Progress(total=len(units), jobs=max(1, jobs))
    out = MapResult(progress=prog)
    if not units:
        return out

    def _record(idx: int, value: Any) -> None:
        out.results[idx] = value
        prog.tick(ok=True)
        if on_result:
            on_result(idx, value, prog)

    def _record_err(idx: int, exc: BaseException) -> None:
        out.errors[idx] = exc
        prog.tick(ok=False)
        if on_error:
            on_error(idx, exc, prog)

    # ---- serial ---------------------------------------------------------
    if jobs <= 1:
        for idx, unit in enumerate(units):
            try:
                _record(idx, fn(unit))
            except halt_exceptions as e:                      # noqa: PERF203
                out.halted, out.halt_error = True, e
                out.unsubmitted = list(range(idx, len(units)))
                return out
            except Exception as e:                            # noqa: BLE001
                _record_err(idx, e)
        return out

    # ---- pooled ---------------------------------------------------------
    # Submission is BOUNDED, not one-shot. Handing a 108-unit screen to the
    # pool in one call queues 108 pickled argument sets immediately; bounding
    # in flight to a small multiple of the worker count keeps the queue short,
    # so a halt cancels work that has not been handed over yet instead of work
    # already sitting in a worker's inbox.
    in_flight: dict[Any, int] = {}
    pending = list(enumerate(units))
    ex = ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker)
    try:
        while pending or in_flight:
            while pending and len(in_flight) < jobs * 2 and not out.halted:
                idx, unit = pending.pop(0)
                in_flight[ex.submit(fn, unit)] = idx
            if not in_flight:
                break
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done:
                idx = in_flight.pop(fut)
                try:
                    _record(idx, fut.result())
                except halt_exceptions as e:
                    if not out.halted:
                        out.halted, out.halt_error = True, e
                except Exception as e:                        # noqa: BLE001
                    _record_err(idx, e)
            if out.halted and pending:
                # Everything still queued here was never handed to a worker.
                # Captured ONCE, on the iteration the halt is first seen: the
                # loop keeps running to drain the futures already in flight,
                # and by the next pass `pending` is empty - reassigning then
                # would overwrite the real list with [] and report a halted
                # screen as one that had started everything.
                out.unsubmitted = [i for i, _ in pending]
                pending.clear()
    finally:
        # cancel_futures so a halt does not wait out a full queue. The workers
        # already running are allowed to finish - killing one mid-write is how
        # a partial artifact reaches disk looking complete.
        ex.shutdown(wait=True, cancel_futures=True)
        gc.collect()
    return out


def describe_plan(n_units: int, jobs: int, reason: str,
                  worker_gib: float = DEFAULT_WORKER_GIB) -> str:
    """The allocation banner, printed once before any unit starts."""
    avail = available_gib()
    cores = os.cpu_count() or 1
    mode = "SERIAL" if jobs <= 1 else f"PARALLEL x{jobs}"
    return (f"  execution  : {mode} · {n_units} configuration(s)\n"
            f"  allocation : {reason}\n"
            f"  host       : {cores} core(s) · {avail:.1f} GiB available · "
            f"~{worker_gib:.1f} GiB budgeted per worker")
