#!/usr/bin/env python3
"""
test_memory_guard.py — the memory guard: that each threshold produces the tier
it claims, that WARN collects, that THROTTLE pauses, that HALT raises with the
numbers attached, and that the process-RSS ceiling is a second independent
trigger.

Location:  ~/src/trading/tests/test_memory_guard.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_memory_guard.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_memory_guard.py

EVERY READING IS MOCKED, DELIBERATELY. A test that waited for the machine to
reach 90% RAM would be a test that never runs, and one that asserted on the
machine's ACTUAL memory would pass or fail on what else is open. `psutil` is
patched at the point `backtest.memory_guard` looks it up, so every tier is
exercised at a reading chosen here.

THE ONE THING THAT CANNOT BE MOCKED IS THE INTEGRATION, and section 5 covers it
against the real modules: that `iter_temporal_chunks` guards each chunk, that
the sweep guards each grid cell, and — the part that matters most — that
`baseline.py` and `scan.py` RE-RAISE a halt past their `except Exception`
handlers instead of swallowing it and walking into the allocation it was raised
to prevent.
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.memory_guard import (DEFAULT_HALT_PCT,            # noqa: E402
                                   DEFAULT_MAX_RSS_GIB,
                                   DEFAULT_THROTTLE_PCT,
                                   DEFAULT_WARN_PCT, GIB, HALT,
                                   MEMORY_HALT_EXIT_CODE, OK, THROTTLE, WARN,
                                   MemoryContext, MemoryGuard,
                                   MemorySafetyException, guard_memory)

GUARD_MODULE = "backtest.memory_guard"

# A crossover with a declared grid, for the end-to-end halt case. Kept here
# rather than imported from `test_batch_runner`, which is a script-style suite:
# importing it would execute its module body under pytest.
_PROBE_STRATEGY = '''
import pandas as pd

TIMEFRAME = "1d"
SYMBOLS = ["ES"]
DEFAULT_PARAMS = {"fast": 5, "slow": 20}
PARAM_GRID = {"fast": [3, 5, 10], "slow": [10, 20, 40]}


def signal_fn(bars, fast=5, slow=20):
    if fast >= slow:
        raise ValueError(f"fast must be < slow; got {fast} >= {slow}")
    c = bars["close"]
    f = c.rolling(fast, min_periods=fast).mean()
    s = c.rolling(slow, min_periods=slow).mean()
    above = f > s
    was = above.shift(1).fillna(False).astype(bool)
    return ((above & ~was).fillna(False).astype(bool),
            (~above & was).fillna(False).astype(bool))


def make_signal_fn(fast=5, slow=20):
    def _bound(bars):
        return signal_fn(bars, fast=fast, slow=slow)
    return _bound
'''


def fake_memory(pct: float, rss_gib: float = 1.0,
                total_gib: float = 32.0):
    """
    Patch both readings `MemoryGuard` takes, at the point it takes them.

    `available` is derived from `pct` rather than passed separately, so a case
    cannot accidentally describe a machine that is 90% full with 30 GiB free —
    the guard reports `available_gib` and a reader has to be able to trust it.
    """
    vm = mock.Mock()
    vm.percent = float(pct)
    vm.total = float(total_gib) * GIB
    vm.available = float(total_gib) * GIB * (1.0 - float(pct) / 100.0)

    proc = mock.Mock()
    proc.memory_info.return_value = mock.Mock(rss=float(rss_gib) * GIB)

    psutil_mock = mock.Mock()
    psutil_mock.virtual_memory.return_value = vm
    psutil_mock.Process.return_value = proc
    return mock.patch(f"{GUARD_MODULE}.psutil", psutil_mock)


def guard(**kwargs) -> MemoryGuard:
    """A guard that is on regardless of the environment's kill switch."""
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("cooldown_s", 0.0)
    return MemoryGuard(**kwargs)


def raises(fn, *args, **kwargs) -> MemorySafetyException:
    try:
        fn(*args, **kwargs)
    except MemorySafetyException as exc:
        return exc
    raise AssertionError("MemorySafetyException was not raised")


# ==========================================================================
# 1. Normal execution
# ==========================================================================
def test_normal_load_is_ok_and_does_nothing() -> None:
    """
    THE REQUEST'S FIRST CLAUSE: below `warn_pct`, status is OK.

    And nothing happens — no collection, no sleep. A guard that collected on
    every check would turn an O(20us) reading into an O(100ms) one at every
    grid cell, which is a performance regression disguised as safety.
    """
    with fake_memory(pct=40.0, rss_gib=2.0):
        g = guard()
        with mock.patch.object(gc, "collect") as collect:
            status = g.check_memory("unit")
            g.enforce("unit")
        assert status["status"] == OK, status
        assert collect.call_count == 0, "OK collected garbage"

    assert status["sys_mem_pct"] == 40.0
    assert status["rss_gib"] == 2.0
    assert abs(status["available_gib"] - 32.0 * 0.6) < 1e-6, status
    assert status["context"] == "unit"
    assert status["enabled"] is True
    assert set(status) >= {"status", "sys_mem_pct", "rss_gib",
                           "available_gib"}


def test_each_threshold_produces_its_own_tier() -> None:
    """
    The tiers are escalating and `>=`, so a threshold written as 75 fires AT
    75. `>` would make every bound off by one reading — invisible until the
    once it matters.
    """
    cases = [
        (10.0, OK), (74.9, OK),
        (75.0, WARN), (78.0, WARN), (81.9, WARN),
        (82.0, THROTTLE), (85.0, THROTTLE), (89.9, THROTTLE),
        (90.0, HALT), (92.0, HALT), (99.9, HALT),
    ]
    for pct, want in cases:
        with fake_memory(pct=pct, rss_gib=1.0):
            got = guard().check_memory("tiers")["status"]
        assert got == want, f"{pct}% gave {got}, expected {want}"

    assert (DEFAULT_WARN_PCT, DEFAULT_THROTTLE_PCT, DEFAULT_HALT_PCT) == \
        (75.0, 82.0, 90.0)


def test_thresholds_out_of_order_are_refused() -> None:
    """
    With halt below warn a run would stop before it ever warned, and the two
    milder tiers would be unreachable code that reads as active protection.
    """
    with pytest.raises(ValueError, match="ascend"):
        MemoryGuard(warn_pct=90.0, throttle_pct=82.0, halt_pct=75.0)
    with pytest.raises(ValueError, match="ascend"):
        MemoryGuard(warn_pct=0.0)
    with pytest.raises(ValueError, match="max_rss_gib"):
        MemoryGuard(max_rss_gib=0.0)


# ==========================================================================
# 2. WARN — the request's second clause
# ==========================================================================
def test_warning_logs_and_collects() -> None:
    """
    THE REQUEST'S SECOND CLAUSE: at 78% the guard warns and `gc.collect()`
    runs.

    The collection is not decoration. This pipeline holds large frames in
    reference CYCLES — a result refers to frames that refer back through
    closures — so the collector, not refcounting, is what returns them, and
    calling it while there is still room is the entire purpose of a threshold
    below the ceiling.
    """
    with fake_memory(pct=78.0, rss_gib=3.0):
        g = guard()
        with mock.patch.object(gc, "collect") as collect:
            status = g.enforce("scan.grid_iteration")
        assert collect.call_count == 1, (
            f"gc.collect() ran {collect.call_count} times at 78%")

    assert status["status"] == WARN, status
    assert status["collected"] is True
    assert g.warns == 1 and g.throttles == 0
    assert "78" in status["reason"]


def test_the_first_warn_always_collects_and_the_next_ones_are_paced() -> None:
    """
    `gc.collect()` on a heap of million-row frames costs O(100ms), and the
    sweep reaches its guard once per grid cell — 432 a chunk. Left unpaced, a
    run sitting just over `warn_pct` would spend ~40s a chunk re-walking a heap
    the previous call had already cleaned.

    THE FIRST ONE ALWAYS RUNS, whatever the interval. A guard that skipped it
    would do nothing at all on a short run, which is exactly the run where a
    single collection is most likely to be enough.
    """
    with fake_memory(pct=78.0, rss_gib=3.0):
        g = guard(gc_interval_s=60.0)
        with mock.patch.object(gc, "collect") as collect:
            for _ in range(10):
                g.enforce("paced")
            assert collect.call_count == 1, (
                f"{collect.call_count} collections in 10 warns")
        assert g.warns == 10, "the CHECK must never be rate-limited"
        assert g.collections == 1

        # With no interval, every warn collects.
        g2 = guard(gc_interval_s=0.0)
        with mock.patch.object(gc, "collect") as collect:
            for _ in range(5):
                g2.enforce("unpaced")
            assert collect.call_count == 5


# ==========================================================================
# 3. THROTTLE
# ==========================================================================
def test_throttle_collects_and_pauses() -> None:
    """
    At 82% the guard collects AND sleeps. The sleep is what the specification
    calls a "non-blocking cooldown" and it is blocking — `time.sleep` is, by
    definition. What it buys is real: it yields the CPU so the allocator can
    return freed arenas and reclaim can make progress, neither of which happens
    inside a tight allocating loop. At this point the pipeline is
    single-threaded, so there is no other work to block.

    THROTTLE IS NEVER PACED, unlike WARN: at this tier the collection is the
    mitigation rather than advisory pressure.
    """
    with fake_memory(pct=85.0, rss_gib=4.0):
        g = guard(cooldown_s=0.25, gc_interval_s=60.0)
        with mock.patch.object(gc, "collect") as collect, \
                mock.patch.object(time, "sleep") as sleep:
            for _ in range(3):
                status = g.enforce("scan.grid_iteration")
        assert collect.call_count == 3, "THROTTLE was rate-limited"
        assert sleep.call_count == 3
        assert sleep.call_args[0][0] == 0.25
    assert status["status"] == THROTTLE
    assert g.throttles == 3 and g.warns == 0


def test_the_cooldown_actually_elapses() -> None:
    """`time.sleep` is mocked above; here it is not, so the pause is real."""
    with fake_memory(pct=85.0, rss_gib=4.0):
        g = guard(cooldown_s=0.15)
        started = time.monotonic()
        g.enforce("real-sleep")
        assert time.monotonic() - started >= 0.15


# ==========================================================================
# 4. HALT — the request's third and fourth clauses
# ==========================================================================
def test_halt_raises_with_accurate_diagnostic_context() -> None:
    """
    THE REQUEST'S THIRD CLAUSE: at 92% the guard raises, and the message
    carries the numbers.

    BOTH numbers, and that is the point of the message rather than a detail of
    it. The halt trigger is SYSTEM-WIDE, so an unrelated process can stop a
    six-hour sweep that was itself using two gigabytes — a halt is not evidence
    that the run was the problem, and `rss_gib` beside `sys_mem_pct` is what
    lets a reader tell the two apart.
    """
    with fake_memory(pct=92.0, rss_gib=6.5):
        g = guard()
        with mock.patch.object(gc, "collect") as collect, \
                mock.patch.object(time, "sleep") as sleep:
            exc = raises(g.enforce, "scan.grid_iteration")
        # A halt does not collect or sleep: there is no room left for either to
        # help, and the caller is about to write partial results.
        assert collect.call_count == 0 and sleep.call_count == 0

    message = str(exc)
    assert "Memory safety ceiling breached in scan.grid_iteration" in message
    assert "92.0% RAM used" in message, message
    assert "6.50 GiB RSS" in message, message
    assert "Halting gracefully before OOM." in message, message

    assert exc.context == "scan.grid_iteration"
    assert exc.status["status"] == HALT
    assert exc.status["sys_mem_pct"] == 92.0
    assert exc.status["rss_gib"] == 6.5
    assert isinstance(exc, RuntimeError)


def test_the_process_rss_ceiling_is_a_second_independent_trigger() -> None:
    """
    THE REQUEST'S FOURTH CLAUSE: RSS over `max_rss_gib` halts, even on a box
    with plenty of free RAM.

    The two triggers are independent on purpose. The system percentage is what
    the OOM killer looks at; the RSS ceiling is what bounds THIS process, and
    it is the one that fires when a single runaway sweep is the problem rather
    than the machine.
    """
    # 30% system RAM — nothing else would fire — but 25 GiB resident.
    with fake_memory(pct=30.0, rss_gib=25.0, total_gib=128.0):
        g = guard(max_rss_gib=24.0)
        exc = raises(g.enforce, "baseline.screen")
        assert g.check_memory("x")["status"] == HALT

    assert "25.00 GiB RSS" in str(exc)
    assert "30.0% RAM used" in str(exc)
    assert "process RSS" in exc.status["reason"], exc.status
    assert "24.0" in exc.status["reason"]

    # And just under the ceiling is not a halt.
    with fake_memory(pct=30.0, rss_gib=23.9, total_gib=128.0):
        assert guard(max_rss_gib=24.0).check_memory("x")["status"] == OK


def test_the_default_rss_ceiling_is_inert_on_this_machine() -> None:
    """
    THE DEFAULT IS 24.0 GiB AND THIS VM HAS 24.9 GiB OF RAM, so a process
    holding 24 GiB has taken 96% of the box and the 90% SYSTEM threshold fired
    long ago — around 22.4 GiB used machine-wide. The RSS ceiling as defaulted
    can never be the trigger here.

    That is not a bug and the default is the specification's, but a ceiling
    that cannot fire is worth a failing test rather than a silent assumption
    that it protects something. `for_this_machine()` derives one that binds.
    """
    assert DEFAULT_MAX_RSS_GIB == 24.0
    with fake_memory(pct=95.0, rss_gib=23.0, total_gib=24.9):
        status = guard().check_memory("inert")
        # The SYSTEM threshold is what caught it, not the RSS ceiling.
        assert status["status"] == HALT
        assert "system" in status["reason"], status["reason"]

    with fake_memory(pct=50.0, rss_gib=1.0, total_gib=24.9):
        derived = MemoryGuard.for_this_machine(headroom_gib=2.0)
        assert derived.max_rss_gib == pytest.approx(22.9, abs=0.05), (
            derived.max_rss_gib)
        assert derived.max_rss_gib < 24.9, "the derived ceiling must fit"


def test_the_exit_code_is_not_137() -> None:
    """
    137 is 128 + SIGKILL — what the shell reports when the OOM killer has
    actually killed a process, which is the precise outcome this module exists
    to avoid. Exiting 137 after halting CLEANLY would tell every log scraper
    that the thing we prevented is what happened.
    """
    assert MEMORY_HALT_EXIT_CODE != 137
    assert MEMORY_HALT_EXIT_CODE == 75          # EX_TEMPFAIL: retry smaller


# ==========================================================================
# 5. The kill switch, the wrappers, and the real integration points
# ==========================================================================
def test_the_environment_kill_switch_disables_every_tier() -> None:
    """
    A loaded workstation must not fail a test suite over its own browser, and
    an operator who has decided to run at 95% should be able to say so in the
    command they typed. A disabled guard still returns a well-formed dict, so
    a caller reading `["status"]` never has to know.
    """
    for value in ("off", "0", "false", "no", "OFF"):
        with fake_memory(pct=99.0, rss_gib=99.0), \
                mock.patch.dict("os.environ", {"BT_MEMORY_GUARD": value}):
            g = MemoryGuard()
            status = g.enforce("disabled")      # must not raise
            assert status["status"] == OK, value
            assert status["enabled"] is False

    with fake_memory(pct=99.0, rss_gib=1.0), \
            mock.patch.dict("os.environ", {"BT_MEMORY_GUARD": "on"}):
        raises(MemoryGuard().enforce, "enabled")


def test_the_context_manager_and_decorator_both_guard() -> None:
    """
    `MemoryContext` checks on ENTRY and again on EXIT, and they catch different
    things: entry refuses to start a step there is no room for, exit catches
    the step that has just allocated what will kill the NEXT one.
    """
    with fake_memory(pct=40.0, rss_gib=1.0):
        g = guard()
        with MemoryContext("block", guard=g):
            pass
        assert g.checks == 2, "entry and exit"

    with fake_memory(pct=95.0, rss_gib=1.0):
        g = guard()
        with pytest.raises(MemorySafetyException):
            with MemoryContext("block", guard=g):
                pytest.fail("the body ran despite a halt on entry")

    # The exit check is skipped while unwinding: replacing an in-flight
    # traceback with one about memory hides what actually happened.
    with fake_memory(pct=40.0, rss_gib=1.0):
        g = guard()
        with pytest.raises(ZeroDivisionError):
            with MemoryContext("block", guard=g):
                raise ZeroDivisionError("the real failure")
        assert g.checks == 1, "the exit check ran while unwinding"

    with fake_memory(pct=40.0, rss_gib=1.0):
        g = guard()

        @guard_memory("decorated", guard=g)
        def work(x):
            return x * 2

        assert work(21) == 42
        assert g.checks == 2


def test_the_data_loader_guards_every_chunk_boundary() -> None:
    """
    Against the real `iter_temporal_chunks`, not a stand-in. The guard is
    enforced immediately before each chunk is handed out — after the read that
    is the largest thing the loader holds, and before the consumer allocates
    several times that on top of it.
    """
    import numpy as np
    import pandas as pd

    from backtest.data_loader import iter_temporal_chunks

    n = 3 * 365 * 24 * 4
    close = 1000.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.4, n))
    bars = pd.DataFrame({
        "ts": pd.date_range("2013-03-01", periods=n, freq="15min", tz="UTC"),
        "symbol": "ES", "open": close, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 1000.0})

    with fake_memory(pct=40.0, rss_gib=1.0):
        g = guard()
        chunks = list(iter_temporal_chunks(bars, chunk_years=2,
                                           warmup_bars=100, guard=g))
        assert len(chunks) >= 2, len(chunks)
        assert g.checks == len(chunks), (
            f"{g.checks} checks for {len(chunks)} chunks")

    # A halt stops the generator rather than yielding a chunk it cannot afford.
    with fake_memory(pct=95.0, rss_gib=1.0):
        g = guard()
        with pytest.raises(MemorySafetyException, match="iter_temporal_chunks"):
            list(iter_temporal_chunks(bars, chunk_years=2, warmup_bars=100,
                                      guard=g))


def test_the_runners_re_raise_a_halt_past_their_broad_handlers() -> None:
    """
    THE INTEGRATION FAILURE THAT WOULD BE INVISIBLE.

    Both runners wrap each configuration in `except Exception`, which is right
    for a missing spec or an empty slice of the lake — one bad contract must
    not end a 108-configuration screen. It is exactly wrong for a memory halt:
    swallowing it moves on to the next configuration, which allocates as much
    as the one just refused, on a machine that is no emptier.

    Checked at the SOURCE, because the behaviour only shows up on a box that is
    actually out of memory, and a test that waited for that would never run.
    """
    # scan.py still catches it inline, in its own loop.
    for name in ("scan.py",):
        src = (REPO / "backtest" / name).read_text()
        assert "except MemorySafetyException" in src, (
            f"{name} does not catch MemorySafetyException explicitly, so its "
            f"`except Exception` handler will swallow a halt")
        halt_at = src.index("except MemorySafetyException")
        broad_at = src.index("except Exception", halt_at)
        assert halt_at < broad_at, (
            f"{name} catches Exception before MemorySafetyException; the "
            f"broad handler wins and the halt is swallowed")
        assert "MEMORY_HALT_EXIT_CODE" in src, (
            f"{name} does not exit with the memory-halt code")

    # baseline.py delegates its configuration loop to `backtest/parallel.py`
    # so it can run several at once, so the broad handler that could swallow a
    # halt now lives THERE. The invariant is unchanged and is checked in both
    # places: baseline must declare the halt type to the executor and act on
    # the halt it reports, and the executor must catch that type BEFORE its own
    # `except Exception`. Checking only baseline would pass a version that
    # declared the type to an executor which then filed it as an ordinary
    # per-unit error, which is precisely the swallow this test exists to catch.
    src = (REPO / "backtest" / "baseline.py").read_text()
    assert "halt_exceptions=(MemorySafetyException,)" in src, (
        "baseline.py does not declare MemorySafetyException as a halt to "
        "backtest/parallel.py, so a halt will be recorded as one bad "
        "configuration and the screen will continue allocating")
    assert "outcome.halted" in src, (
        "baseline.py never acts on the executor's halt flag")
    assert "MEMORY_HALT_EXIT_CODE" in src, (
        "baseline.py does not exit with the memory-halt code")

    src = (REPO / "backtest" / "parallel.py").read_text()
    assert src.count("except halt_exceptions") == 2, (
        "backtest/parallel.py must catch the declared halt types in BOTH its "
        "serial and pooled paths; a path that misses it swallows the halt for "
        "whichever --jobs value takes it")
    for halt_at in [i for i in range(len(src))
                    if src.startswith("except halt_exceptions", i)]:
        broad_at = src.index("except Exception", halt_at)
        assert halt_at < broad_at, (
            "backtest/parallel.py catches Exception before the declared halt "
            "types; the broad handler wins and the halt becomes a per-unit "
            "error")


def test_a_mid_run_halt_carries_the_work_already_done() -> None:
    """
    THE POINT OF `partial`, AGAINST THE REAL SWEEP.

    A halt at the very first check carries an empty record, which is honest but
    proves nothing. This runs the sweep until it has completed chunks and
    accumulated trades, THEN halts, and requires the exception to carry all of
    it — because that record is what `main` writes to disk, and a file saying
    only "the run stopped" is not worth the write.

    The guard tightens after a fixed number of checks rather than on a real
    reading: waiting for the machine to actually fill would be a test that
    never runs.
    """
    import numpy as np
    import pandas as pd

    from backtest.engine import BacktestConfig
    from backtest.scan import scan_symbol_chunked

    class TighteningGuard(MemoryGuard):
        """Passes `budget` checks, then halts — a run that hits the wall."""

        def __init__(self, budget: int, **kwargs):
            super().__init__(**kwargs)
            self._budget = budget
            self._seen = 0

        def check_memory(self, context: str = "") -> dict:
            status = super().check_memory(context)
            self._seen += 1
            if self._seen > self._budget:
                return {**status, "status": HALT, "sys_mem_pct": 93.4,
                        "rss_gib": 7.25, "reason": "system 93.4% >= 90.0%"}
            return status

    import tempfile
    strategy = Path(tempfile.mkdtemp()) / "guard_probe.py"
    strategy.write_text(_PROBE_STRATEGY, encoding="utf-8")

    n = 12 * 365
    rng = np.random.default_rng(7)
    close = (4000.0 + rng.normal(0.4, 12.0, n).cumsum()
             + 60.0 * np.sin(np.arange(n) / 45.0))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(6.0, 3.0, n))
    bars = pd.DataFrame({
        "ts": pd.date_range("2013-01-02", periods=n, freq="D", tz="UTC"),
        "symbol": "ES", "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread, "close": close,
        "volume": rng.integers(10_000, 100_000, n).astype(float)})

    exc = raises(
        scan_symbol_chunked, strategy, bars, "ES", BacktestConfig(),
        {"fast": [3, 5, 10], "slow": [10, 20, 40]}, strat_name="probe",
        chunk_years=2, warmup_bars=600, settlement_bars=600, progress=False,
        guard=TighteningGuard(25, enabled=True, cooldown_s=0.0))

    partial = exc.partial
    assert partial is not None, "the halt carried no partial work"
    assert "grid_iteration" in exc.context, exc.context
    assert len(partial["chunks_completed"]) >= 1, partial
    assert partial["bars_swept"] > 0
    assert 0 < partial["combinations_evaluated"] <= \
        partial["combinations_declared"]
    assert sum(partial["trades_by_column"]) > 0, (
        "no trades were recorded, so the flush would preserve nothing")
    for row in partial["chunks_completed"]:
        assert {"start", "end", "bars"} <= set(row), row
    assert exc.status["sys_mem_pct"] == 93.4


def test_the_sweep_guards_the_grid_cell_not_only_the_chunk() -> None:
    """
    The four boolean masks are built one column at a time and stacked, so a
    sweep dies part-way through building a list of 432 of them — not at a chunk
    boundary. A guard that only fired between chunks would check at every point
    except the one where the memory goes.
    """
    src = (REPO / "backtest" / "scan.py").read_text()
    assert "scan.grid_iteration" in src, (
        "scan.py does not guard the grid cell")
    cell_at = src.index("scan.grid_iteration")
    loop_at = src.index("for cell, combo in enumerate(combos, 1):")
    assert loop_at < cell_at, "the grid guard is outside the combo loop"
    # And the halt carries the partial work rather than only the fact of it.
    assert "partial=_partial()" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
