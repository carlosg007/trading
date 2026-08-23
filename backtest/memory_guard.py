"""
The memory guard: watch system RAM and this process's RSS, and stop a long run
deliberately rather than letting the kernel stop it.

Location:  ~/src/trading/backtest/memory_guard.py

WHAT THIS IS FOR
================
A full-lake Stage 1 screen or a 432-cell Stage 2 sweep runs for hours and
allocates in bursts. When it allocates past what the box has, the Linux OOM
killer sends SIGKILL — which cannot be caught, cannot flush anything, and
leaves an artifact directory half written with no record of why. Everything
computed since the last write is gone and the log's last line is whatever was
printing at the time.

This module makes that outcome reachable only after a deliberate one: check
before each expensive step, collect garbage while there is still room to, and
raise `MemorySafetyException` with the numbers attached while the process is
still alive enough to write them down.

IT IS A GUARD, NOT A BUDGET. It cannot stop an allocation, cap one, or make a
sweep fit. It only decides whether the NEXT step starts.

FOUR THINGS THAT ARE EASY TO GET WRONG HERE
===========================================
1. **THE HALT TRIGGER IS SYSTEM-WIDE, SO SOMEBODY ELSE'S PROCESS CAN STOP YOUR
   SWEEP.** `psutil.virtual_memory().percent` is the whole box, not this
   process. A browser or a second backtest pushing the machine past `halt_pct`
   halts a six-hour run that was itself using two gigabytes. That is the
   intended trade — the OOM killer is also system-wide, and it picks its victim
   by a score this process does not control — but it means a halt is NOT
   evidence that the run was the problem. `check_memory` returns `rss_gib`
   beside `sys_mem_pct` for exactly that reason, and every halt message carries
   both.

2. **`max_rss_gib` IS THE PROCESS-SCOPED BOUND, AND ITS DEFAULT IS INERT ON
   THIS MACHINE.** The default of 24.0 GiB comes from the specification. This
   VM has 24.9 GiB of RAM in total, so a process holding 24 GiB has already
   taken 96% of the box and the 90% SYSTEM threshold fired long before —
   somewhere near 22.4 GiB used machine-wide. The RSS ceiling as defaulted can
   therefore never be the trigger here. It is kept because it IS the right
   bound on a bigger box (CLAUDE.md sizes this VM at 32-64 GB), and
   `MemoryGuard.for_this_machine()` derives one that actually binds.

3. **`percent` IS NOT `used / total`.** On Linux psutil computes it from
   `available`, which already excludes reclaimable page cache. That is the
   number worth gating on: a box showing 80% "used" mostly in cache is not
   under pressure and gating on `used` would throttle a run that had plenty of
   room. Reading /proc/meminfo by hand and dividing is the mistake this note
   exists to prevent — the fallback below uses `MemAvailable` for the same
   reason.

4. **A "NON-BLOCKING COOLDOWN SLEEP" IS A CONTRADICTION.** The throttle sleeps,
   and `time.sleep` blocks by definition. What it buys is real and worth being
   precise about: it yields the CPU so the allocator can return freed arenas to
   the kernel and so reclaim can make progress, neither of which happens while
   a tight loop is allocating. At the point this is called the pipeline is
   single-threaded anyway, so blocking is what is wanted — there is no other
   work to do.

WHAT IT DELIBERATELY DOES NOT DO: THROTTLE THREADS
==================================================
The objective asks for worker-thread throttling and this module does not
implement it, because on this stack it cannot be done honestly. The heavy
allocation happens inside vectorbt / numba / BLAS, whose thread pools are sized
when the library is imported. Setting `OMP_NUM_THREADS` afterwards does not
resize a pool that already exists, so the call would look like throttling,
change nothing, and remove the reason anyone would look further. The repository
pins `OMP_NUM_THREADS=1` at the process boundary instead, which is where it
works.

The lever that does exist is `backtest/data_loader.py`: fewer bars per chunk is
fewer bytes per step, and `--chunk-years` is how an operator applies it.

DISABLING IT
============
`BT_MEMORY_GUARD=off` (or `0`, `false`, `no`) turns every guard into a no-op
that still returns a well-formed status dict. That exists for two reasons: a
loaded workstation must not fail a test suite over its own browser, and an
operator who has decided to run at 95% RAM should be able to, out loud, in the
command they typed. It is read once per `MemoryGuard`, at construction, so a
long run cannot change behaviour halfway through.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

try:                                        # pragma: no cover - env dependent
    import psutil
except ImportError:                         # pragma: no cover - env dependent
    psutil = None

GIB = float(2 ** 30)

OK, WARN, THROTTLE, HALT = "OK", "WARN", "THROTTLE", "HALT"

# The specification's thresholds.
DEFAULT_WARN_PCT = 75.0
DEFAULT_THROTTLE_PCT = 82.0
DEFAULT_HALT_PCT = 90.0
DEFAULT_MAX_RSS_GIB = 24.0

# How long the throttle yields for. See note 4 in the module docstring.
DEFAULT_COOLDOWN_S = 1.0

# The same warning line can be reached thousands of times in a sweep. The CHECK
# is never rate-limited — it costs about 20 microseconds and skipping one is
# how a spike goes unnoticed — but the LOG is, so a run under sustained
# pressure prints a line every few seconds rather than one per grid cell.
DEFAULT_LOG_INTERVAL_S = 5.0

# How often the WARN tier may actually collect.
#
# `gc.collect()` on a heap holding several million-row frames costs O(100ms).
# The sweep in `backtest/scan.py` reaches its guard once per grid cell — 432 of
# them per chunk — so a run sitting just over `warn_pct` would spend ~40s a
# chunk in the collector, most of it re-walking a heap the previous call had
# already cleaned. WARN therefore collects at most this often.
#
# THROTTLE AND HALT ARE NEVER RATE-LIMITED. WARN is advisory pressure and one
# collection per interval is the useful part of it; the two tiers above it are
# the ones standing between the run and the OOM killer.
#
# The FIRST warn always collects, whatever the interval: a guard that skipped
# it would do nothing at all on a short run, which is precisely the run where a
# single collection is most likely to be enough.
DEFAULT_GC_INTERVAL_S = 2.0

# The exit code the CLI uses after a memory halt.
#
# NOT 137. That is 128 + SIGKILL, which is what the shell reports when the OOM
# killer has actually killed a process — the precise outcome this module exists
# to avoid. Exiting 137 after halting CLEANLY would tell every log scraper and
# CI dashboard that the thing we prevented is what happened. 75 is EX_TEMPFAIL
# from sysexits.h: a temporary failure, the caller is invited to retry — which
# is exactly right, because the retry is `--chunk-years 2` or a bigger box.
MEMORY_HALT_EXIT_CODE = 75

_TRUTHY_OFF = {"off", "0", "false", "no", "disabled"}


class MemorySafetyException(RuntimeError):
    """
    Raised when a memory ceiling is breached, before the allocation that would
    have crossed it.

    Carries the reading that triggered it (`status`) and, when a caller has
    attached one, the partial work it managed to save (`partial`). A halt that
    reached a human as a bare traceback would leave them guessing whether the
    run was the cause or the casualty — see note 1 in the module docstring.

    Inherits `RuntimeError` rather than `Exception` directly so it is
    catchable, but note the consequence at every integration site: a loop
    wrapped in `except Exception` will SWALLOW it and carry on into the
    allocation it was raised to prevent. `backtest/baseline.py` re-raises it
    explicitly for that reason.
    """

    def __init__(self, message: str, status: dict | None = None,
                 context: str = "", partial: Any = None) -> None:
        super().__init__(message)
        self.status = dict(status or {})
        self.context = context
        self.partial = partial


def guard_enabled() -> bool:
    """False when `BT_MEMORY_GUARD` names an off value."""
    return str(os.environ.get("BT_MEMORY_GUARD", "on")).strip().lower() \
        not in _TRUTHY_OFF


# --------------------------------------------------------------------------
# Reading the numbers
# --------------------------------------------------------------------------
def _read_psutil() -> tuple[float, float, float]:
    """`(sys_mem_pct, rss_gib, available_gib)` from psutil."""
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    return float(vm.percent), rss / GIB, float(vm.available) / GIB


def _read_proc() -> tuple[float, float, float]:
    """
    The same three numbers from /proc, for a box without psutil.

    psutil is pinned in `requirements.txt`, so this should be unreachable. It
    exists because this module is imported by `data_loader`, `scan` and
    `baseline` — three modules the whole pipeline runs through — and a missing
    optional dependency turning a SAFETY mechanism into an ImportError at the
    top of every stage is a worse failure than the one being guarded against.

    `MemAvailable` rather than `MemFree`: it is the kernel's own estimate of
    what a new allocation could get, page cache included, which is what psutil
    reports and what note 3 in the module docstring is about.
    """
    info: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            info[key] = float(parts[0]) * 1024.0     # kB -> bytes

    total = info.get("MemTotal", 0.0)
    available = info.get("MemAvailable", info.get("MemFree", 0.0))
    pct = 100.0 * (1.0 - available / total) if total else 0.0

    rss = 0.0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = float(line.split()[1]) * 1024.0
            break
    return pct, rss / GIB, available / GIB


_WARNED_NO_PSUTIL = False


def read_memory() -> tuple[float, float, float]:
    """`(sys_mem_pct, rss_gib, available_gib)`, psutil first."""
    global _WARNED_NO_PSUTIL
    if psutil is not None:
        return _read_psutil()
    if not _WARNED_NO_PSUTIL:
        _WARNED_NO_PSUTIL = True
        print("[memory_guard] psutil is not importable; falling back to "
              "/proc. It is pinned in requirements.txt — `uv pip install -r "
              "requirements.txt` restores the intended reader.",
              file=sys.stderr, flush=True)
    return _read_proc()


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------
@dataclass
class MemoryGuard:
    """
    Thresholds, and the decision they imply.

    Escalating, and checked in descending order so the most severe wins: HALT
    at or above `halt_pct` OR at or above `max_rss_gib` of process RSS,
    THROTTLE at `throttle_pct`, WARN at `warn_pct`, OK below.

    `>=` throughout rather than `>`: a threshold written as 90 should fire AT
    90. The alternative makes every bound off by one reading, which nobody
    notices until the one time it matters.
    """

    warn_pct: float = DEFAULT_WARN_PCT
    throttle_pct: float = DEFAULT_THROTTLE_PCT
    halt_pct: float = DEFAULT_HALT_PCT
    max_rss_gib: float = DEFAULT_MAX_RSS_GIB
    cooldown_s: float = DEFAULT_COOLDOWN_S
    log_interval_s: float = DEFAULT_LOG_INTERVAL_S
    gc_interval_s: float = DEFAULT_GC_INTERVAL_S
    enabled: bool = field(default_factory=guard_enabled)

    # Counters, so a run can report what the guard did rather than only what it
    # prevented. A sweep that spent an hour throttling and finished is a
    # different result from one that never noticed anything.
    warns: int = field(default=0, init=False)
    throttles: int = field(default=0, init=False)
    checks: int = field(default=0, init=False)
    collections: int = field(default=0, init=False)
    _last_log: float = field(default=0.0, init=False)
    # `None` rather than 0.0 so the first WARN always collects — see
    # DEFAULT_GC_INTERVAL_S. A 0.0 sentinel would be `interval` seconds in the
    # past only if the process had been up that long, which on a short run it
    # has not been.
    _last_gc: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not (0 < self.warn_pct <= self.throttle_pct <= self.halt_pct
                <= 100):
            raise ValueError(
                f"thresholds must ascend within (0, 100]: warn "
                f"{self.warn_pct}, throttle {self.throttle_pct}, halt "
                f"{self.halt_pct}. Out of order, a run would halt before it "
                f"ever warned and the two milder tiers would be dead code.")
        if self.max_rss_gib <= 0:
            raise ValueError(
                f"max_rss_gib must be > 0; got {self.max_rss_gib}")

    # -- construction helpers ------------------------------------------
    @classmethod
    def for_this_machine(cls, headroom_gib: float = 2.0, **kwargs
                         ) -> "MemoryGuard":
        """
        A guard whose RSS ceiling actually binds on the box it is running on.

        The default `max_rss_gib` of 24.0 is inert on a 24.9 GiB VM — see note
        2 in the module docstring. This sets it to total RAM less
        `headroom_gib`, so the process-scoped bound trips before the machine is
        exhausted rather than after the system-wide one already has.
        """
        try:
            total = (float(psutil.virtual_memory().total) if psutil is not None
                     else _proc_total_bytes())
        except Exception:                                   # noqa: BLE001
            total = DEFAULT_MAX_RSS_GIB * GIB
        kwargs.setdefault("max_rss_gib",
                          max(1.0, total / GIB - float(headroom_gib)))
        return cls(**kwargs)

    # -- reading -------------------------------------------------------
    def check_memory(self, context: str = "") -> dict:
        """
        One reading and the status it implies.

        `{"status", "sys_mem_pct", "rss_gib", "available_gib"}`, plus the
        context and the threshold that decided it. Never raises and never
        sleeps — `enforce` is what acts. Keeping the reading separate from the
        action is what lets a caller log or record pressure without also
        pausing for it.

        A disabled guard still returns a well-formed dict, with status OK and
        `enabled: False`, so a caller reading `["status"]` does not have to
        know whether the guard is on.
        """
        pct, rss_gib, available_gib = read_memory()
        self.checks += 1

        if not self.enabled:
            status, reason = OK, "guard disabled (BT_MEMORY_GUARD)"
        elif rss_gib >= self.max_rss_gib:
            status = HALT
            reason = f"process RSS {rss_gib:.2f} GiB >= {self.max_rss_gib} GiB"
        elif pct >= self.halt_pct:
            status, reason = HALT, f"system {pct:.1f}% >= {self.halt_pct}%"
        elif pct >= self.throttle_pct:
            status = THROTTLE
            reason = f"system {pct:.1f}% >= {self.throttle_pct}%"
        elif pct >= self.warn_pct:
            status, reason = WARN, f"system {pct:.1f}% >= {self.warn_pct}%"
        else:
            status, reason = OK, f"system {pct:.1f}% < {self.warn_pct}%"

        return {
            "status": status,
            "sys_mem_pct": pct,
            "rss_gib": rss_gib,
            "available_gib": available_gib,
            "context": context,
            "reason": reason,
            "enabled": bool(self.enabled),
        }

    # -- acting --------------------------------------------------------
    def enforce(self, context: str = "", partial: Any = None) -> dict:
        """
        Read, then do what the reading says. Returns the status dict.

        WARN      log, `gc.collect()`.
        THROTTLE  log, `gc.collect()`, then sleep `cooldown_s`.
        HALT      raise `MemorySafetyException`.

        `gc.collect()` at WARN is not decoration. This pipeline holds large
        frames in reference cycles — a `BacktestResult` refers to frames that
        refer back through closures — so the collector, not refcounting, is
        what returns them. Calling it while there is still room is the whole
        point of a threshold below the ceiling.

        `partial` is attached to the exception on HALT, so a caller that has
        accumulated work can hand it to whatever writes it down. Nothing here
        writes to disk: a guard that did I/O on the way out would be doing it
        at exactly the moment the machine is least able to.
        """
        status = self.check_memory(context)
        state = status["status"]

        if state == OK:
            return status

        if state == HALT:
            raise MemorySafetyException(
                f"Memory safety ceiling breached in {context}: "
                f"{status['sys_mem_pct']:.1f}% RAM used "
                f"({status['rss_gib']:.2f} GiB RSS). Halting gracefully "
                f"before OOM.",
                status=status, context=context, partial=partial)

        if state == WARN:
            self.warns += 1
            collected = self._collect()
            self._log(f"[memory_guard] WARN {context}: "
                      f"{status['sys_mem_pct']:.1f}% RAM, "
                      f"{status['rss_gib']:.2f} GiB RSS, "
                      f"{status['available_gib']:.2f} GiB available"
                      + (" — collecting" if collected else
                         f" — collected within the last "
                         f"{self.gc_interval_s:g}s, skipping"))
            status["collected"] = collected
            return status

        # THROTTLE
        self.throttles += 1
        self._log(f"[memory_guard] THROTTLE {context}: "
                  f"{status['sys_mem_pct']:.1f}% RAM, "
                  f"{status['rss_gib']:.2f} GiB RSS, "
                  f"{status['available_gib']:.2f} GiB available — collecting "
                  f"and pausing {self.cooldown_s:g}s")
        # Never rate-limited: at this tier the collection IS the mitigation.
        gc.collect()
        self.collections += 1
        self._last_gc = time.monotonic()
        time.sleep(self.cooldown_s)
        status["collected"] = True
        return status

    def _collect(self) -> bool:
        """Collect if the interval allows. True when it actually ran."""
        now = time.monotonic()
        if self._last_gc is not None and now - self._last_gc < self.gc_interval_s:
            return False
        gc.collect()
        self.collections += 1
        self._last_gc = now
        return True

    def _log(self, message: str) -> None:
        """Rate-limited, to stderr. See DEFAULT_LOG_INTERVAL_S."""
        now = time.monotonic()
        if now - self._last_log < self.log_interval_s:
            return
        self._last_log = now
        print(message, file=sys.stderr, flush=True)

    def summary(self) -> dict:
        """What the guard did over a run, for a stage to record."""
        return {"checks": self.checks, "warns": self.warns,
                "throttles": self.throttles, "collections": self.collections,
                "enabled": bool(self.enabled),
                "warn_pct": self.warn_pct, "throttle_pct": self.throttle_pct,
                "halt_pct": self.halt_pct, "max_rss_gib": self.max_rss_gib}


def _proc_total_bytes() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return float(line.split()[1]) * 1024.0
    return DEFAULT_MAX_RSS_GIB * GIB


# --------------------------------------------------------------------------
# The shared default, and the two wrappers
# --------------------------------------------------------------------------
# One guard for the pipeline, so its counters describe the whole run rather
# than whichever module happened to make one. A caller wanting different
# thresholds constructs its own and passes it in — every integration point
# takes a `guard=` argument for that reason.
DEFAULT_GUARD = MemoryGuard()


def enforce(context: str = "", partial: Any = None,
            guard: MemoryGuard | None = None) -> dict:
    """Module-level `enforce` against `DEFAULT_GUARD`."""
    return (guard or DEFAULT_GUARD).enforce(context, partial=partial)


def check_memory(context: str = "", guard: MemoryGuard | None = None) -> dict:
    """Module-level `check_memory` against `DEFAULT_GUARD`."""
    return (guard or DEFAULT_GUARD).check_memory(context)


class MemoryContext(ContextDecorator):
    """
    Guard a block, or decorate a function, with one object.

    Checks on ENTRY and again on EXIT, and both matter for different reasons:
    the entry check refuses to start a step there is no room for, and the exit
    check catches the step that has just allocated the thing that will kill the
    NEXT one — which is the reading a loop wants, because the next iteration is
    where it would have died.

    The exit check is skipped when the block is already unwinding an exception.
    Raising a memory halt on the way out of a failure would replace the
    original traceback with one about memory, and the first one is the one that
    explains what happened.

        with MemoryContext("stage1.symbol_loop"):
            ...

        @MemoryContext("scan.sweep")
        def sweep(...):
            ...
    """

    def __init__(self, context: str = "", guard: MemoryGuard | None = None,
                 on_exit: bool = True) -> None:
        self.context = context
        self.guard = guard or DEFAULT_GUARD
        self.on_exit = on_exit

    def __enter__(self) -> "MemoryContext":
        self.guard.enforce(self.context)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.on_exit and exc_type is None:
            self.guard.enforce(self.context)
        return False


def guard_memory(context: str | None = None,
                 guard: MemoryGuard | None = None,
                 **thresholds) -> Callable:
    """
    Decorate a function so every call is guarded.

        @guard_memory("scan.grid_iteration", halt_pct=95.0)
        def sweep(...): ...

    Threshold keywords build a guard for this function alone; with none given
    it uses `DEFAULT_GUARD`, so the counters stay pooled across the run.
    """
    own = MemoryGuard(**thresholds) if thresholds else guard

    def decorate(fn: Callable) -> Callable:
        label = context or f"{fn.__module__}.{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            with MemoryContext(label, guard=own):
                return fn(*args, **kwargs)
        return wrapper
    return decorate


__all__ = ["MemoryGuard", "MemorySafetyException", "MemoryContext",
           "guard_memory", "enforce", "check_memory", "read_memory",
           "guard_enabled", "DEFAULT_GUARD", "MEMORY_HALT_EXIT_CODE",
           "OK", "WARN", "THROTTLE", "HALT"]
