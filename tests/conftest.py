"""
pytest configuration for this directory, and the reason it has to exist.

TWO RUNNER CONVENTIONS LIVE HERE, AND ONLY ONE OF THEM IS PYTEST-SHAPED
=======================================================================
Most suites in this directory are SCRIPTS. They collect results with a
`check(name, ok)` helper that appends to a module-level FAILURES list, and
`main()` exits non-zero at the end — which is exactly what CLAUDE.md documents
("No pytest config - each is a script that exits non-zero on failure").

Run under pytest, those suites do two bad things and one useless one:

  * **They report green while failing.** pytest calls the `test_*` functions
    and sees them return None. `check()` recording a failure is invisible to
    it, and `main()` — the only thing that turns FAILURES into an exit code —
    is never called. A suite with ten broken checks passes.
  * **They interfere with each other.** `pytest tests/` runs every suite in ONE
    process, sharing `lru_cache`d lake lookups, module-level caches and any
    environment variable a suite sets. Measured on this tree:
    `test_regime_cache` and `test_temporal_chunking` both PASS alone and both
    FAIL together, on state neither of them owns.
  * **They error on collection.** Their functions take positional arguments —
    `test_promote_certification(tmp)`, `test_stage2_card(blob)` — which pytest
    tries to satisfy as fixtures and cannot. That produced 59 collection errors
    on this directory, none of them a defect in the code under test.

Supplying `tmp` and `blob` as fixtures would be the obvious fix and it is the
wrong one: those functions are called by `main()` in a fixed ORDER with shared
state — `blob` is literally the return value of an earlier case — so letting
pytest call them independently would run them in a configuration nobody
designed, and the greens would mean less than the errors did.

WHAT THIS FILE DOES INSTEAD
===========================
Script-style suites are excluded from normal collection and run by
`test_suite_runners.py` as SUBPROCESSES, one pytest test each, asserting the
exit code the convention already defines. Each therefore runs in its own
process, in its own order, exactly as designed — which fixes the false green,
the interference and the collection errors together.

Assert-based suites are collected normally and keep their per-case granularity.
They are pytest-shaped on purpose: every case fails through `assert`, so both
runners report the same thing.

The classification is by the `def check(` marker rather than a hardcoded list,
so a new suite lands in the right bucket by how it is written rather than by
somebody remembering to edit this file.

NAMING A SCRIPT SUITE DIRECTLY
==============================
`collect_ignore` is consulted when pytest walks the DIRECTORY. An explicitly
named file argument - `pytest tests/test_run_pipeline.py` - skips that walk, so
the file was collected function by function after all and every case taking a
positional argument errored with `fixture 'tmp' not found`. Fifteen errors on
`test_run_pipeline`, four on `test_dow_gate`, and both of them noise: the code
under test was fine.

`pytest_collect_file` below closes that hole by handing a directly named script
suite to the SAME subprocess wrapper the directory walk uses. So the command
works, reports the suite's real verdict, and adds no errors.

WHY NOT JUST SUPPLY THE FIXTURE. Measured on 2026-09-10, on this tree, with a
`tmp` fixture added and one check deliberately broken in `test_dow_gate.py`:

    pytest test_dow_gate.py   ->  18 passed
    python  test_dow_gate.py  ->  exit 1

pytest calls each `test_*`, sees it return None, and reports green; `check()`
recording the failure is invisible to it because only `main()` turns FAILURES
into an exit code. The fixture does not make the direct run work - it makes it
LIE, and a green that cannot go red is worse than the error it replaced.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Long enough for the lake-reading suites, short enough that a hang is
#: reported as a hang. Kept in step with `test_suite_runners.SUITE_TIMEOUT_S`.
SUITE_TIMEOUT_S = 1200

TESTS_DIR = Path(__file__).resolve().parent

# The marker that identifies a script-style suite: the `check()` helper whose
# results pytest cannot see.
SCRIPT_MARKER = "\ndef check("


def is_script_suite(path: Path) -> bool:
    """True for a suite whose results only its own `main()` can report."""
    try:
        return SCRIPT_MARKER in path.read_text(encoding="utf-8")
    except OSError:
        return False


def script_suites() -> list[Path]:
    """Every script-style suite in this directory, sorted."""
    return sorted(p for p in TESTS_DIR.glob("test_*.py")
                  if is_script_suite(p))


# pytest reads this at collection: these files are handed to
# `test_suite_runners.py` instead of being collected function by function.
#
# This covers the DIRECTORY walk only. A file named directly on the command
# line never reaches it - see `pytest_collect_file`.
collect_ignore = [p.name for p in script_suites()]


class ScriptSuiteItem(pytest.Item):
    """One script-style suite, run in its own process, passing iff it exits 0."""

    def __init__(self, *, suite: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.suite = suite

    def runtest(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(self.suite)],
            capture_output=True, text=True, timeout=SUITE_TIMEOUT_S,
            # The repo pins this for the suites that refit a classifier per
            # trade: on a 16-core box each fit's thread pool costs far more
            # than the fit. Kept identical to `test_suite_runners.py` - two
            # runners that launch the same suite under different environments
            # would disagree about it, and only one of them would be believed.
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            cwd=str(self.suite.resolve().parent.parent))
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-40:])
            err = "\n".join((proc.stderr or "").splitlines()[-20:])
            raise AssertionError(
                f"{self.suite.name} exited {proc.returncode}\n"
                f"--- stdout (last 40 lines) ---\n{tail}\n"
                f"--- stderr (last 20 lines) ---\n{err}")

    def repr_failure(self, excinfo, style=None):
        """The suite's own output IS the message - it prints a
        `N CHECK(S) FAILED` block naming each one, which is more useful than
        anything this wrapper could reconstruct from a traceback."""
        if isinstance(excinfo.value, AssertionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)

    def reportinfo(self):
        return self.path, 0, f"script suite: {self.suite.stem}"


class ScriptSuiteFile(pytest.File):
    def collect(self):
        yield ScriptSuiteItem.from_parent(
            self, name=self.path.stem, suite=Path(self.path))


def pytest_collect_file(file_path, parent):
    """
    Hand a DIRECTLY NAMED script suite to the subprocess wrapper.

    Returns None for everything else, including every assert-based suite,
    which keeps its per-case granularity under both runners.

    No double-run under `pytest tests/`: the directory walk consults
    `collect_ignore` first and the file never reaches this hook, so
    `test_suite_runners.py` remains the only thing that launches it there.
    """
    path = Path(file_path)
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    if path.name == "test_suite_runners.py" or not is_script_suite(path):
        return None
    return ScriptSuiteFile.from_parent(parent, path=file_path)


def _is_script_suite_path(path: Path) -> bool:
    return (path.suffix == ".py" and path.name.startswith("test_")
            and path.name != "test_suite_runners.py" and is_script_suite(path))


def pytest_collection_modifyitems(session, config, items) -> None:
    """
    Drop the per-function items pytest's own collector added for a script
    suite, leaving the one subprocess wrapper.

    `pytest_collect_file` is NOT a first-result hook: every implementation
    contributes, so naming a script suite directly collected the
    ScriptSuiteFile above AND the python plugin's Module for the same file.
    The wrapper ran the suite correctly and the Module's functions still
    errored on `fixture 'tmp' not found` beside it - the errors this exists to
    remove, now with a passing wrapper next to them.

    Removed HERE rather than by refusing collection, because
    `pytest_ignore_collect` runs before `pytest_collect_file` and would
    suppress the wrapper too. Fixture resolution happens at setup, after this
    hook, so dropping the items now is what prevents the errors rather than
    merely hiding them.
    """
    keep = []
    for item in items:
        path = Path(getattr(item, "path", "") or "")
        if (not isinstance(item, ScriptSuiteItem)
                and path.name and _is_script_suite_path(path)):
            continue
        keep.append(item)
    items[:] = keep
