"""
tests/test_parallel.py - backtest/parallel.py, the (symbol, timeframe) executor.

Location: ~/src/trading/tests/test_parallel.py

Reads no bars and runs no stage. Everything here is synthetic, because the one
property that matters cannot be established by watching a fast screen finish:
**a pooled run must produce the same rows, in the same order, as the serial
loop it replaced.** A parallel screen that is merely FAST and quietly drops or
reorders a configuration looks exactly like a correct one on the console, and
the leaderboard it writes is a different table on every run.

The halt cases matter for the same reason. `MemorySafetyException` must stop a
screen rather than mark one configuration bad, and the units that never ran
have to be reported as never-run rather than folded in with the ones that were
tried and failed - an unfinished screen presented as a completed one with some
bad contracts in it is the reading that gets a strategy promoted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.memory_guard import MemorySafetyException          # noqa: E402
from backtest.parallel import (CORE_RESERVE, MapResult, Progress,  # noqa: E402
                               available_gib, describe_plan,
                               map_units, resolve_jobs)

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail and not ok else ""))
    if not ok:
        _failures.append(name)


# --- module-level workers: a pool pickles by NAME, so these cannot be local --

def square(u):
    return u * u


def slow_square(u):
    time.sleep(0.02)
    return u * u


def odd_raises(u):
    if u % 2:
        raise ValueError(f"odd unit {u}")
    return u * u


def halt_at_three(u):
    if u == 3:
        raise MemorySafetyException("synthetic halt")
    return u * u


def slow_halt_at_three(u):
    """
    Halts at unit 3, and every unit costs real time.

    The sleep is the point. `map_units` bounds submission to `jobs * 2` in
    flight so that a halt cancels work that has not been handed to a worker
    yet - but with an INSTANT workload the whole queue drains before the third
    future is ever examined, and `unsubmitted` comes back empty for a reason
    that says nothing about the mechanism. Slow units are what make the bound
    observable, and an unbounded submission would fail this where the instant
    version passes.
    """
    time.sleep(0.05)
    if u == 3:
        raise MemorySafetyException("synthetic halt")
    return u * u


def test_resolve_jobs() -> None:
    print("\n1. Worker allocation is bounded, and says which bound bound it")
    check("no units means serial", resolve_jobs("auto", 0)[0] == 1)
    check("the default is SERIAL, so existing screens are unchanged",
          resolve_jobs(None, 50)[0] == 1 and resolve_jobs(1, 50)[0] == 1)
    j, why = resolve_jobs("auto", 50)
    check("auto is >= 1 and never exceeds the core reserve",
          1 <= j <= max(1, __import__("os").cpu_count() - CORE_RESERVE), f"{j}")
    check("...and names the bound rather than leaving it to be inferred",
          ("memory-bound" in why or "core-bound" in why or "unit" in why), why)
    check("auto is clamped to the unit count", resolve_jobs("auto", 2)[0] <= 2)
    check("an explicit count is honoured", resolve_jobs(6, 50)[0] == 6)
    check("...and still clamped to the units there are",
          resolve_jobs(99, 4)[0] == 4)
    for bad in ("banana", 0, -3):
        try:
            resolve_jobs(bad, 10)
            check(f"--jobs {bad!r} is refused", False, "no raise")
        except ValueError:
            check(f"--jobs {bad!r} is refused", True)
    check("the plan banner names the host", "core(s)" in describe_plan(4, 2, "x"))
    check("available_gib reads a real number", available_gib() >= 0.0)


def test_serial_and_parallel_agree() -> None:
    print("\n2. A pooled run == the serial loop, value for value and in order")
    units = list(range(12))
    ser = map_units(square, units, jobs=1)
    par = map_units(square, units, jobs=4)
    check("serial produced every unit", len(ser.results) == 12)
    check("parallel produced every unit", len(par.results) == 12)
    check("SAME VALUES", ser.ordered() == par.ordered() == [u * u for u in units],
          f"{par.ordered()}")
    check("results are keyed by SUBMISSION index, not completion",
          all(par.results[i] == i * i for i in range(12)))
    check("neither halted", not ser.halted and not par.halted)
    check("no errors", not ser.errors and not par.errors)

    # Out-of-order completion is the interesting case: without index keying a
    # scheduler that finishes unit 7 before unit 2 silently transposes them.
    par2 = map_units(slow_square, list(range(8)), jobs=4)
    check("ordering survives genuinely concurrent completion",
          par2.ordered() == [u * u for u in range(8)], f"{par2.ordered()}")


def test_errors_are_per_unit_not_fatal() -> None:
    print("\n3. One bad configuration does not end the screen")
    for jobs in (1, 3):
        r = map_units(odd_raises, list(range(8)), jobs=jobs)
        check(f"jobs={jobs}: the even units still completed",
              r.ordered() == [0, 4, 16, 36], f"{r.ordered()}")
        check(f"jobs={jobs}: the odd units are recorded as errors",
              sorted(r.errors) == [1, 3, 5, 7], f"{sorted(r.errors)}")
        check(f"jobs={jobs}: an error is NOT a halt", not r.halted)
        check(f"jobs={jobs}: the error keeps its index, so the caller can name "
              f"the pair", isinstance(r.errors.get(1), ValueError))


def test_halt_stops_the_screen() -> None:
    print("\n4. A memory halt stops the screen and reports what never ran")
    for jobs in (1, 2):
        worker = halt_at_three if jobs == 1 else slow_halt_at_three
        r = map_units(worker, list(range(24)), jobs=jobs,
                      halt_exceptions=(MemorySafetyException,))
        check(f"jobs={jobs}: the run is marked halted", r.halted)
        check(f"jobs={jobs}: the halt is kept, not flattened into errors",
              isinstance(r.halt_error, MemorySafetyException)
              and not r.errors, f"errors={sorted(r.errors)}")
        check(f"jobs={jobs}: units that never started are reported as such, "
              f"so an unfinished screen never reads as a completed one",
              bool(r.unsubmitted), f"{r.unsubmitted}")
        check(f"jobs={jobs}: the halt actually SAVED work - fewer units ran "
              f"than were asked for",
              len(r.results) + len(r.errors) < 24,
              f"{len(r.results)} done of 24")
        check(f"jobs={jobs}: work completed before the halt is KEPT - a screen "
              f"stopped at N leaves a complete report of N",
              all(r.results[i] == i * i for i in r.results))
        check(f"jobs={jobs}: nothing is both completed and unsubmitted",
              not (set(r.results) & set(r.unsubmitted)))

    # Without halt_exceptions the SAME exception is an ordinary per-unit error.
    r = map_units(halt_at_three, list(range(6)), jobs=1)
    check("a halt type not declared as one is just an error",
          not r.halted and 3 in r.errors)


def test_progress_and_eta() -> None:
    print("\n5. Progress is measured, and an ETA from no samples is withheld")
    p = Progress(total=10, jobs=2)
    check("no ETA before anything finishes", p.eta_seconds() is None)
    p.tick(); p.tick()
    check("an ETA appears once there are samples", p.eta_seconds() is not None)
    check("the line carries percent, elapsed, ETA and workers",
          all(t in p.line() for t in ("2/10", "%", "elapsed", "ETA", "worker")),
          p.line())
    for _ in range(8):
        p.tick()
    check("no ETA once complete", p.eta_seconds() is None)
    p2 = Progress(total=4, jobs=1)
    p2.tick(ok=False)
    check("failures are counted and shown", "1 error(s)" in p2.line(), p2.line())


def test_empty_and_single() -> None:
    print("\n6. Degenerate inputs")
    r = map_units(square, [], jobs=4)
    check("no units is not an error", not r.halted and not r.results)
    r = map_units(square, [7], jobs=8)
    check("one unit still runs", r.ordered() == [49])
    check("MapResult.ordered drops gaps rather than inventing them",
          MapResult(results={0: "a", 2: "c"}).ordered() == ["a", "c"])


def main() -> int:
    print("=" * 62)
    print("  backtest/parallel.py - the configuration executor")
    print("=" * 62)
    test_resolve_jobs()
    test_serial_and_parallel_agree()
    test_errors_are_per_unit_not_fatal()
    test_halt_stops_the_screen()
    test_progress_and_eta()
    test_empty_and_single()
    print("\n" + "=" * 62)
    if _failures:
        print(f"  {len(_failures)} CHECK(S) FAILED")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
