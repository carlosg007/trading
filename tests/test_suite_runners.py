#!/usr/bin/env python3
"""
Runs every script-style suite in this directory as a subprocess, one pytest
test each.

Location:  ~/src/trading/tests/test_suite_runners.py

WHY A SUBPROCESS RATHER THAN A DIRECT CALL. See `conftest.py` for the whole
argument; the short version is that those suites record results with a
`check()` helper pytest cannot see, and only their own `main()` turns that into
an exit code. Calling `main()` in-process would fix the false green and leave
the other two problems — the suites would still share one interpreter's caches
and environment, and `test_regime_cache` and `test_temporal_chunking` would
still fail together while passing alone.

So each suite gets its own process, which is how CLAUDE.md documents running
them and the only configuration any of them was designed for.

THE COST IS REAL: this is the whole test suite, so `pytest tests/` now takes as
long as running every script by hand — around ten minutes, most of it in the
three suites that read the lake. That is the price of a gate that means
something. Run a single suite directly while iterating:

    OMP_NUM_THREADS=1 python tests/test_risk_params.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import script_suites

# Long enough for the lake-reading suites (test_engine_batching reads GC's
# whole 1-minute history twice), short enough that a hang is reported as a hang.
SUITE_TIMEOUT_S = 1200

SUITES = script_suites()


def test_there_are_script_suites_to_run() -> None:
    """
    A guard on the guard. If the `def check(` marker ever stops matching, every
    suite silently drops out of collection and `pytest tests/` goes green by
    running nothing at all — which is the exact failure this whole arrangement
    exists to remove, one level up.
    """
    assert SUITES, (
        "no script-style suites were found. Either they were all converted to "
        "assert-based (in which case delete this file and conftest's "
        "collect_ignore), or the marker in conftest.SCRIPT_MARKER no longer "
        "matches and pytest is now collecting nothing.")


@pytest.mark.parametrize("suite", SUITES, ids=lambda p: p.stem)
def test_script_suite(suite: Path) -> None:
    """
    One suite, in its own process. Passes iff it exits 0.

    On failure the suite's own output is the assertion message — the
    `check()`-style suites print a `N CHECK(S) FAILED` block naming each one,
    which is more useful than anything this wrapper could reconstruct.
    """
    proc = subprocess.run(
        [sys.executable, str(suite)],
        capture_output=True, text=True, timeout=SUITE_TIMEOUT_S,
        # The repo pins this for the suites that refit a classifier per trade:
        # on a 16-core box each fit's thread pool costs far more than the fit.
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        cwd=str(suite.resolve().parent.parent))

    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        err = "\n".join((proc.stderr or "").splitlines()[-20:])
        pytest.fail(
            f"{suite.name} exited {proc.returncode}\n"
            f"--- stdout (last 40 lines) ---\n{tail}\n"
            f"--- stderr (last 20 lines) ---\n{err}")


if __name__ == "__main__":
    pytest.main([__file__])
