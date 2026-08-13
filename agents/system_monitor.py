"""
agents.system_monitor - the circuit breaker.

Location:  ~/src/trading/agents/system_monitor.py

SCAFFOLD ONLY. Interfaces and the measured thresholds are defined here; the
monitoring loop is not written yet.

What this is for
----------------
An autonomous agent system can spend an afternoon burning compute on a loop
nobody is reading, or take the box down by asking for more memory than it has.
This watches for both and stops them. It is deliberately the dumbest component
in the system: fixed thresholds, no model in the loop, nothing to reason its
way into an exception.

Thresholds
----------
The box is a Proxmox VM with ~24 GB usable against a 26 GB ceiling, and the
numbers below are measured on this lake rather than guessed:

    full 1-minute lake                  109.8M rows across 27 symbols
    run_backtest, all 27 symbols        peak 2.9 GiB, ~110s
    largest single symbol               5.6M rows (GC)
    get_bars on the full lake           peak 15.4 GiB  <- avoid in agent runs

A backtest that has passed a few GiB is not working harder, it is doing
something the engine was rebuilt to stop doing - most likely assembling a
whole-lake frame instead of streaming per symbol. That is worth killing early,
which is why the warning threshold sits well below the physical limit.

Runaway detection is about wall time and repetition, not correctness: a worker
still running long after its cohort finished, or a tier re-issuing work orders
it has already issued, is looping. The monitor does not try to work out why. It
stops the process and records what it was doing so a human can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Physical ceiling of the VM, and the fractions of it that mean something.
RAM_LIMIT_GIB = 26.0
RAM_WARN_GIB = 8.0          # ~3x a healthy full-lake run; something is wrong
RAM_KILL_GIB = 18.0         # close enough to the ceiling to act before the OOM

# A full-lake backtest is ~110s. An order still running after this is stuck.
WORKER_TIMEOUT_S = 45 * 60
# Identical work orders re-issued this many times is a loop, not persistence.
REPEAT_ORDER_LIMIT = 3


@dataclass
class ProcessSnapshot:
    """One sample of one tracked process."""

    pid: int
    tier: str
    label: str
    rss_gib: float
    elapsed_s: float


@dataclass
class Breach:
    """Why the monitor intervened, and what it did."""

    kind: str                              # ram | timeout | repeat
    snapshot: ProcessSnapshot | None = None
    detail: str = ""
    action: str = "none"                   # none | warned | killed
    context: dict[str, Any] = field(default_factory=dict)


def sample(pid: int) -> ProcessSnapshot:
    """Read current RSS and elapsed time for a tracked process."""
    raise NotImplementedError("system_monitor: not implemented yet")


def check_memory(snap: ProcessSnapshot) -> Breach | None:
    """Warn above RAM_WARN_GIB, kill above RAM_KILL_GIB."""
    raise NotImplementedError("system_monitor: not implemented yet")


def check_runaway(snap: ProcessSnapshot) -> Breach | None:
    """Flag a process past WORKER_TIMEOUT_S."""
    raise NotImplementedError("system_monitor: not implemented yet")


def check_repeat_orders(history: list[Any]) -> Breach | None:
    """Flag a tier re-issuing work it has already issued."""
    raise NotImplementedError("system_monitor: not implemented yet")


def watch(pids: dict[int, str], interval_s: float = 5.0) -> None:
    """Main loop. Samples tracked processes and acts on breaches."""
    raise NotImplementedError("system_monitor: not implemented yet")


def main() -> int:
    raise NotImplementedError("system_monitor: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
