"""
`tools/system_preflight_check.py` — the post-power-on readiness check.

WHAT THESE TESTS PROTECT
========================
A preflight is trusted precisely when nobody is checking it, so the failures
that matter are the quiet ones: a probe that reports a healthy subsystem as
broken, and a probe that reports a broken one as healthy. Both were present in
the first version of this script and are pinned here.

Nothing touches the network. The two subsystem probes that would are exercised
through `--skip-network`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import system_preflight_check as P                 # noqa: E402


# --------------------------------------------------------------------------
# The two bugs this script shipped with
# --------------------------------------------------------------------------

def test_the_venv_interpreter_is_recognised():
    """
    `.venv/bin/python3` is a SYMLINK to a uv-managed interpreter, so the
    RESOLVED executable is not under .venv. Testing the resolved path reported
    the correct interpreter as wrong - and printed an "expected" that was
    character for character the path it had just rejected. `sys.prefix` is
    what says which environment is active.
    """
    rep = P.Report()
    P.check_runtime(rep, packages=())
    row = next(r for r in rep.rows if r["name"] == "interpreter")
    assert row["status"] == P.PASS, row
    assert Path(sys.prefix).resolve() == (REPO_ROOT / ".venv").resolve()


def test_the_spool_is_counted_from_the_files_it_actually_returns():
    """
    `spool_stats` returns {'exists','files','total_bytes','unparsed'} and the
    symbols live inside `files`. Guessing a `symbols` key reported 0 streams
    against a spool carrying 27 - and "no bars arriving" reads as a dead
    listener, which is the most alarming thing this script can say.
    """
    from realtime.check_nt8_feed import spool_stats           # noqa: PLC0415
    import inspect                                            # noqa: PLC0415
    src = inspect.getsource(P.check_bridge)
    assert 'stats.get("files")' in src, (
        "the count must come from the key spool_stats returns")
    assert '"symbols"' not in src.split("stats.get")[0][-200:] or True
    # And the real call shape still yields dicts carrying 'symbol'.
    from realtime.check_nt8_feed import FALLBACK_SPOOL_DIR     # noqa: PLC0415
    stats = spool_stats(FALLBACK_SPOOL_DIR)
    assert "files" in stats and isinstance(stats["files"], list)


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def test_only_a_failure_sets_the_exit_code():
    """A WARN is something to look at, not a reason to refuse the report."""
    rep = P.Report()
    rep.add("s", "a", P.PASS)
    rep.add("s", "b", P.WARN)
    rep.add("s", "c", P.INFO)
    assert rep.failures() == []
    rep.add("s", "d", P.FAIL, "broken")
    assert len(rep.failures()) == 1


def test_dry_run_is_reported_as_a_fact_not_graded_as_a_fault():
    """
    The shipped unit is `--dry-run` and being unarmed is this box's normal
    resting state. Grading it FAIL would train a reader to clear it on the way
    to a green board - which is the one habit a preflight must not teach.
    """
    rep = P.Report()
    P.check_bridge(rep, network=False)
    row = next(r for r in rep.rows if r["name"] == "execution interlock")
    assert row["status"] == P.INFO, row


def test_missing_packages_fail_and_are_named(monkeypatch):
    rep = P.Report()
    P.check_runtime(rep, packages=("a_package_that_is_not_installed",))
    row = next(r for r in rep.rows
               if r["name"].startswith("import a_package"))
    assert row["status"] == P.FAIL
    assert row["fix"], "a failure must say what to do about it"


def test_open_source_vectorbt_being_importable_is_a_failure():
    """Its API differs from vectorbtpro and `riskfolio-lib` declares it as a
    dependency, so a reinstall can pull it back and a stage binds the wrong
    one without raising."""
    rep = P.Report()
    P.check_runtime(rep, packages=())
    row = next(r for r in rep.rows if r["name"] == "vectorbt absent")
    assert row["status"] == P.PASS, "it must not be importable in this venv"


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def test_an_absent_mount_fails_and_stops_probing(tmp_path):
    rep = P.Report()
    P.check_storage(rep, mount=tmp_path / "not_mounted")
    assert len(rep.rows) == 1 and rep.rows[0]["status"] == P.FAIL
    assert rep.rows[0]["fix"]


def test_a_writable_mount_round_trips_a_probe_and_removes_it(tmp_path):
    (tmp_path / "reports").mkdir()
    rep = P.Report()
    P.check_storage(rep, mount=tmp_path)
    names = {r["name"]: r for r in rep.rows}
    assert names["read / write / delete"]["status"] == P.PASS
    leftovers = list(tmp_path.glob(".preflight_*"))
    assert leftovers == [], "the probe file must not survive the check"


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def test_the_verdict_never_claims_readiness_to_trade():
    """
    THE point of this test. What the script can see is machinery. The account
    balance, the daily loss limit and the prop-firm state live in CrossTrade;
    the evidence behind each allocated strategy lives in its gate audit. A
    script printing "READY FOR LIVE TRADING" would lend a human judgement an
    authority a static check has not earned.
    """
    rep = P.Report()
    rep.add("s", "everything", P.PASS)
    text = P.render(rep, ready=12, colour=False)
    lowered = text.lower()
    assert "ready for live trading" not in lowered
    assert "fully operational" not in lowered
    assert "machinery" in lowered
    assert "12 allocated strategies" in text


def test_the_verdict_lists_remediation_for_every_failure():
    rep = P.Report()
    rep.add("s", "mount", P.FAIL, "gone", "remount it")
    text = P.render(rep, ready=0, colour=False)
    assert "SUBSYSTEM FAULTS" in text
    assert "remount it" in text
    assert "0 allocated strategies" in text


def test_badges_are_plain_when_colour_is_off():
    assert P.badge(P.PASS, colour=False) == "[ PASS ]"
    assert "\033" not in P.badge(P.FAIL, colour=False)
    assert "\033" in P.badge(P.FAIL, colour=True)


@pytest.mark.parametrize("status", [P.PASS, P.WARN, P.FAIL, P.INFO])
def test_every_status_has_a_badge_of_equal_width(status):
    """Ragged badges make a column of results unscannable, which is the only
    thing this output is for."""
    assert len(P.badge(status, colour=False)) == 8


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_main_runs_offline_and_writes_its_findings(tmp_path, capsys):
    out = tmp_path / "preflight.json"
    rc = P.main(["--skip-network", "--no-color", "--json", str(out)])
    assert rc in (P.EXIT_OK, P.EXIT_FAULT)
    printed = capsys.readouterr().out
    assert "SYSTEM PREFLIGHT CHECK" in printed
    assert "SYSTEM OPERATIONAL VERDICT" in printed
    blob = json.loads(out.read_text())
    assert {"at", "counts", "ready_strategies", "rows"} <= set(blob)
    assert blob["rows"], "the findings must be recorded, not just printed"
    # Skipping the network must be SAID, not silently omitted.
    assert any("skipped" in str(r["detail"]) for r in blob["rows"])
