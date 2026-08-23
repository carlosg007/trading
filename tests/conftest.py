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
"""

from __future__ import annotations

from pathlib import Path

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
collect_ignore = [p.name for p in script_suites()]
