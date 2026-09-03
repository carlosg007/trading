"""
tests/test_regime_staleness_guard.py - `master_live.regime_write_age` and the
`--max-regime-write-age-sec` cycle guard.

ASSERT-BASED, so `tests/conftest.py` collects it normally. No `def check(`
marker: that is what routes a suite to the subprocess runner.

WHAT THIS IS GUARDING
=====================
A regime reading is a PERMISSION. `trading-regime-daemon.timer` publishes
`data/live_regime_state.json` and the live loop reads it; if the publisher
stops, the quadrant freezes at whatever it last said and the loop keeps
granting permissions for a market that has since moved. Nothing in the
quadrant itself distinguishes a quiet tape from a dead publisher - only the
write clock does.

THE WRITE AGE IS NOT THE BAR AGE, and that distinction is the reason this
guard is separate from `--max-regime-age-sec`:

  * `age_seconds`      the DAEMON's clock. Timeframe-independent. 900s is a
                       sensible default at any bar width.
  * `bar_age_seconds`  the MARKET's clock. On a 1h feed it reaches ~3,600s in
                       the ordinary course of an hour that has not closed yet.

`--max-regime-age-sec` compares `max(write, bar)` and so cannot be defaulted:
at 900 it refuses three quarters of every hour on the 1h unit this repository
actually runs, and at 3,600 it fails to notice a publisher fifty minutes dead.
It ships as None for that reason and this guard does not touch it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_live import build_parser, regime_write_age            # noqa: E402

UNIT = REPO_ROOT / "deploy" / "systemd" / "trading-master-live.service"


def _state(tmp_path: Path, written_at: dict[str, str],
           bar_ts: str = "2026-09-02T23:00:00+00:00") -> str:
    """
    A state file in the shape `MasterRegimeDaemon.update_state` writes, with
    per-symbol write clocks the caller chooses.
    """
    symbols = {
        sym: {"symbol": sym, "tf": "1h", "quadrant": "Q1",
              "regime": "Q1_HIGH_VOL_TREND", "bar_ts": bar_ts,
              "written_at": stamp, "updated_at": stamp}
        for sym, stamp in written_at.items()
    }
    blob = {"symbols": symbols,
            "by_timeframe": {s: {"1h": r} for s, r in symbols.items()}}
    path = tmp_path / "live_regime_state.json"
    path.write_text(json.dumps(blob))
    return str(path)


def _now_iso(offset_s: float = 0.0) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc)
            - timedelta(seconds=offset_s)).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# 1. The measurement
# --------------------------------------------------------------------------
def test_a_fresh_publisher_reports_a_small_age(tmp_path):
    age, detail = regime_write_age(
        _state(tmp_path, {"NQ": _now_iso(5), "ES": _now_iso(3)}))
    assert age is not None and age < 60, detail
    assert "2 of 2 symbols timestamped" in detail


def test_the_age_is_the_WORST_symbol_not_the_average(tmp_path):
    """
    One symbol the daemon has stopped refreshing is a dead publisher for that
    symbol's strategies, and averaging would hide it behind four fresh ones.
    """
    age, detail = regime_write_age(_state(tmp_path, {
        "NQ": _now_iso(5), "ES": _now_iso(5), "GC": _now_iso(4000)}))
    assert age > 3900, detail
    assert "GC" in detail, "the offending symbol is not named"


def test_a_dead_publisher_reports_a_large_age(tmp_path):
    age, _ = regime_write_age(
        _state(tmp_path, {"NQ": "2026-09-01T00:00:00+00:00"}))
    assert age is not None and age > 3600


# --------------------------------------------------------------------------
# 2. Unreadable is treated as stale, never as fresh
# --------------------------------------------------------------------------
def test_a_missing_state_file_is_not_freshness(tmp_path):
    age, detail = regime_write_age(str(tmp_path / "nothing.json"))
    assert age is None, "an absent state file reported an age"
    assert "unreadable" in detail or "no symbols" in detail


def test_a_record_with_no_write_clock_is_not_freshness(tmp_path):
    """
    An unreadable clock is not evidence of freshness. Returning 0.0 here would
    make a malformed record the SAFEST possible reading, which is the
    inversion this whole guard exists to avoid.
    """
    blob = {"symbols": {"NQ": {"symbol": "NQ", "tf": "1h", "quadrant": "Q1"}},
            "by_timeframe": {"NQ": {"1h": {"symbol": "NQ", "tf": "1h",
                                           "quadrant": "Q1"}}}}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(blob))
    age, detail = regime_write_age(str(path))
    assert age is None, detail


def test_an_empty_state_is_not_freshness(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"symbols": {}, "by_timeframe": {}}))
    age, detail = regime_write_age(str(path))
    assert age is None
    assert "no symbols" in detail


# --------------------------------------------------------------------------
# 3. The flag, and why it is not the other one
# --------------------------------------------------------------------------
def test_the_write_age_guard_is_on_by_default():
    assert build_parser().parse_args([]).max_regime_write_age_sec == 900.0


def test_the_bar_age_flag_is_still_off_by_default():
    """
    `--max-regime-age-sec` compares max(write, bar). A 1h bar is legitimately
    ~3,600s old before it closes, so defaulting it to 900 would refuse three
    quarters of every hour on the unit this repository runs. It stays None.
    """
    assert build_parser().parse_args([]).max_regime_age_sec is None


def test_zero_disables_the_guard():
    assert build_parser().parse_args(
        ["--max-regime-write-age-sec", "0"]).max_regime_write_age_sec == 0.0


def test_the_guard_stands_the_whole_cycle_down_not_one_symbol():
    """
    Every symbol reads the SAME file, so a stale file is not a fact about one
    contract. The loop `continue`s before any bar is loaded, so nothing is
    dispatched and `dispatch_exits` is never reached - open inventory keeps
    whatever brackets it already has rather than being flattened by a guard.
    """
    import inspect

    import master_live as M

    src = inspect.getsource(M.main)
    assert "CRITICAL regime state is STALE" in src
    assert "max_regime_write_age_sec" in src
    # It must sit BEFORE the bucket loop, or a stale cycle would still load
    # bars and evaluate strategies against the frozen quadrant.
    guard = src.index("CRITICAL regime state is STALE")
    buckets = src.index("for bucket_tf in buckets")
    assert guard < buckets, "the guard runs after bars are already loaded"


def test_the_critical_line_names_the_publisher_to_check():
    """A CRITICAL that does not say which unit to look at is a page with no
    next action."""
    import inspect

    import master_live as M

    src = inspect.getsource(M.main)
    assert "trading-regime-daemon" in src


# --------------------------------------------------------------------------
# 4. The unit file
# --------------------------------------------------------------------------
def test_the_shipped_unit_sets_the_guard():
    text = UNIT.read_text()
    assert "--max-regime-write-age-sec 900" in text


def test_the_shipped_unit_does_not_set_the_bar_age_flag():
    """
    It runs `--tf 1h`. Setting `--max-regime-age-sec` here would refuse most
    of every hour, and the two flags read almost identically.

    Scoped to the ExecStart, not the whole file: the unit's comment NAMES the
    flag to explain why it is not used, and a check over the file would fail
    on the documentation rather than on a setting.
    """
    text = UNIT.read_text()
    exec_start = text.split("ExecStart=", 1)[1].split("\n\n", 1)[0]
    assert "--max-regime-write-age-sec 900" in exec_start
    assert "--max-regime-age-sec" not in exec_start.replace(
        "--max-regime-write-age-sec", "")
