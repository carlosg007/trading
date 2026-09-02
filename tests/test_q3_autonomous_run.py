"""
tests/test_q3_autonomous_run.py - `scripts/q3_autonomous_run.py`.

ASSERT-BASED, so `tests/conftest.py` collects it normally. No `def check(`
marker: that is what routes a suite to the subprocess runner.

NO STAGE IS EVER EXECUTED HERE. Every case runs against `--dry-run` or against
the pure functions, so the suite reads handoffs and builds command lines and
never touches the lake. A test that launched Stage 1 would take hours and would
be testing the pipeline rather than the runner.

WHAT THIS IS GUARDING
=====================
The runner exists so an unattended campaign is reproducible and inspectable
rather than something that happened once. The failures worth catching are the
ones that would only show up at 3am with nobody watching:

  * a Q3 certification registered into a portfolio that cannot trade Q3, or
    whose basket does not carry the contract - inert, and silent
  * `--tf` passed as a list to a stage that certifies ONE pair
  * promotion running because the script was launched, rather than because
    somebody asked for it
  * a memory halt swallowed and the campaign continuing on a full box
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.q3_autonomous_run import (                            # noqa: E402
    HOLDOUT_START,
    IS_END,
    IS_START,
    Q3_CODE,
    Q3_LABEL,
    Runner,
    build_parser,
    campaign,
    q3_survivors,
    render_report,
    route_for,
)


def _runner(tmp_path) -> Runner:
    return Runner(tmp_path / "run.log", dry_run=True)


def _plan(tmp_path, **over) -> list[str]:
    """Every command line the campaign would issue, as strings."""
    runner = _runner(tmp_path)
    kwargs = dict(strategies=["keltner_trend_drift_20260901"],
                  symbols=["6E"], timeframes=["30m"],
                  score_mode="vol_normalized", promote=False)
    kwargs.update(over)
    campaign(runner, **kwargs)
    return [c["cmd"] for c in runner.calls]


# --------------------------------------------------------------------------
# 1. Routing — the failure that is silent
# --------------------------------------------------------------------------
def test_every_target_symbol_routes_to_a_portfolio_that_permits_q3():
    """
    Two conditions and both are silent when wrong. `promote.py` will register
    a strategy certified outside a portfolio's basket and warn that it "will
    never place an order"; a portfolio whose regime scope excludes Q3 stands a
    Q3 strategy down in the one environment it was certified for.

    Every portfolio was widened to all four quadrants on 2026-09-02, so all
    five target contracts now route. This asserts the OUTCOME against the live
    table on purpose - it is the check that would catch a portfolio being
    narrowed again, or a contract dropping out of a basket, which is exactly
    when a Q3 campaign would start registering strategies that can never
    trade. `test_a_narrowed_portfolio_is_still_refused` pins the guard itself
    on a fixture that cannot move.
    """
    from portfolio.config_loader import clear_cache

    clear_cache()
    expected = {"6E": "incubator-odd", "6J": "incubator-odd",
                "NQ": "incubator-odd", "ES": "incubator-even",
                "GC": "incubator-even"}
    for symbol, portfolio in expected.items():
        got, why = route_for(symbol)
        assert got == portfolio, f"{symbol} -> {got} ({why})"


def test_a_narrowed_portfolio_is_still_refused(monkeypatch, tmp_path):
    """
    THE GUARD, on a fixture that cannot move. The case above reads the live
    table and would stop testing anything the moment every account permits
    everything - which is now true. This narrows one and checks the refusal
    still fires, so the routing check cannot quietly become a no-op.
    """
    import json

    import scripts.q3_autonomous_run as R
    from portfolio.config_loader import clear_cache

    blob = json.loads((REPO_ROOT / "config" / "portfolios.json").read_text())
    for pid, portfolio in blob["portfolios"].items():
        if "Incubator" in pid:
            portfolio["basket"]["regime_quadrants"] = [
                "Q1_HIGH_VOL_TREND", "Q2_HIGH_VOL_CHOP"]
    narrowed = tmp_path / "portfolios.json"
    narrowed.write_text(json.dumps(blob, indent=2) + "\n")

    real = R.PROJECT_ROOT
    monkeypatch.setattr(R, "PROJECT_ROOT", tmp_path.parent)
    clear_cache()
    try:
        # route_for builds the path from PROJECT_ROOT/config/portfolios.json,
        # so stand that tree up rather than patching the resolver.
        (tmp_path.parent / "config").mkdir(exist_ok=True)
        (tmp_path.parent / "config" / "portfolios.json").write_text(
            narrowed.read_text())
        portfolio, why = R.route_for("ES")
        assert portfolio is None, f"ES routed to {portfolio} on a Q1/Q2 table"
        assert "Q1" in why and "Q2" in why, why
    finally:
        monkeypatch.setattr(R, "PROJECT_ROOT", real)
        clear_cache()


def test_an_unroutable_certification_is_recorded_not_promoted():
    """A certified-but-unroutable configuration must appear on the report with
    its reason, not vanish and not get registered."""
    rows = [{"strategy": "s", "symbol": "ES", "tf": "1h", "version": "A",
             "gate_r": "PASS", "pf": 1.20, "trades": 88, "certified": True,
             "promoted": False,
             "note": "certified but NOT routable: permits only ['Q1', 'Q2']"}]

    class _Args:
        score_mode, promote = "vol_normalized", True

    report = render_report(rows, "2026-09-02T00:00:00+00:00", _Args())
    assert "NOT routable" in report
    assert "| YES | no |" in report.replace("  ", " ")


# --------------------------------------------------------------------------
# 2. The command lines the stages will actually accept
# --------------------------------------------------------------------------
def test_stage_three_gets_one_timeframe_per_invocation(tmp_path):
    """
    A gate audit certifies ONE (parameters, timeframe) pair, and Stages 3 and 4
    refuse a comma-separated `--tf`. Passing the campaign's list straight
    through would fail argument parsing hours into an unattended run.
    """
    plan = _plan(tmp_path, timeframes=["30m", "1h"])
    for cmd in plan:
        if "audit_gates.py" in cmd or "verify_full.py" in cmd:
            tf = cmd.split("--tf ")[1].split()[0]
            assert "," not in tf, f"Stage 3/4 got a timeframe list: {cmd}"


def test_stage_one_carries_the_charter_window_and_the_score_mode(tmp_path):
    stage1 = [c for c in _plan(tmp_path) if "baseline.py" in c]
    assert stage1, "no Stage 1 command was planned"
    for cmd in stage1:
        assert f"--start {IS_START}" in cmd and f"--end {IS_END}" in cmd
        assert "--score-mode vol_normalized" in cmd


def test_stage_two_never_reads_the_holdout(tmp_path):
    """
    Stage 2 fits what it reads. `check_in_sample_window` refuses a window
    reaching the holdout, but the runner must not be the thing that tries.
    """
    for cmd in (c for c in _plan(tmp_path) if "scan.py" in c):
        assert f"--end {IS_END}" in cmd
        assert HOLDOUT_START not in cmd


def test_verify_full_is_never_passed_a_params_path(tmp_path):
    """
    The brief asked for `--params <best_params_path>`; `verify_full.py` has no
    such flag - it takes `--param key=value` and finds the locked set the same
    way Stage 3 does. Passing a path would fail argument parsing.
    """
    for cmd in (c for c in _plan(tmp_path, promote=True)
                if "verify_full.py" in c):
        assert "--params " not in cmd


# --------------------------------------------------------------------------
# 3. Promotion is opt-in
# --------------------------------------------------------------------------
def test_promotion_is_withheld_unless_asked_for(tmp_path):
    """
    Stage 5 rewrites `config/portfolios.json` and stages packages into the
    incubator. Launching the script by mistake must not do that; everything up
    to certification runs either way.
    """
    assert build_parser().parse_args([]).promote is False
    assert build_parser().parse_args(["--promote"]).promote is True
    assert not [c for c in _plan(tmp_path, promote=False)
                if "promote.py" in c]


def test_the_report_says_which_mode_ran():
    class _Withheld:
        score_mode, promote = "vol_normalized", False

    class _Enabled:
        score_mode, promote = "alpha", True

    assert "WITHHELD" in render_report([], "t", _Withheld())
    assert "ENABLED" in render_report([], "t", _Enabled())


# --------------------------------------------------------------------------
# 4. Reading the handoff
# --------------------------------------------------------------------------
def test_only_q3_survivors_are_swept(tmp_path, monkeypatch):
    """A pair Stage 1 designated Q1 is not a Q3 candidate, whatever else is on
    the handoff."""
    import scripts.q3_autonomous_run as R

    root = tmp_path / "pipeline"
    (root / "strat_x").mkdir(parents=True)
    (root / "strat_x" / "surviving_assets.json").write_text(
        '{"surviving_pairs": ['
        '{"symbol": "6J", "tf": "1h", "optimal_regime": "Low Volatility / '
        'Trending", "quadrant": "Q3"},'
        '{"symbol": "NQ", "tf": "30m", "optimal_regime": "High Volatility / '
        'Trending", "quadrant": "Q1"}]}')
    monkeypatch.setattr(R, "artifacts_root", lambda: root)

    got = q3_survivors("strat_x")
    assert [(r["symbol"], r["tf"]) for r in got] == [("6J", "1h")]


def test_a_strategy_with_no_q3_survivor_is_recorded_and_skipped(
        tmp_path, monkeypatch):
    import scripts.q3_autonomous_run as R

    root = tmp_path / "pipeline"
    (root / "keltner_trend_drift_20260901").mkdir(parents=True)
    (root / "keltner_trend_drift_20260901" / "surviving_assets.json"
     ).write_text('{"surviving_pairs": []}')
    monkeypatch.setattr(R, "artifacts_root", lambda: root)

    runner = Runner(tmp_path / "log", dry_run=False)
    runner.run = lambda *a, **k: {"ok": True, "rc": 0}      # no subprocesses
    rows = campaign(runner, ["keltner_trend_drift_20260901"], ["6E"], ["30m"],
                    "vol_normalized", False)
    assert len(rows) == 1
    assert Q3_CODE in rows[0]["note"]
    assert rows[0]["certified"] is not True


def test_a_missing_module_is_a_row_rather_than_a_crash(tmp_path):
    runner = _runner(tmp_path)
    rows = campaign(runner, ["no_such_strategy_20990101"], ["6E"], ["30m"],
                    "alpha", False)
    assert rows and rows[0]["note"] == "module not found"


# --------------------------------------------------------------------------
# 5. A memory halt is not a bad configuration
# --------------------------------------------------------------------------
def test_a_memory_halt_stops_the_campaign(tmp_path, monkeypatch):
    """
    `memory_guard` exits 75 precisely so a caller can tell "this box is out of
    RAM" from "this configuration is bad". Continuing allocates just as much on
    a machine that is no emptier.
    """
    import subprocess

    class _Halt:
        returncode, stdout, stderr = Runner.MEMORY_HALT_RC, "", ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Halt())
    runner = Runner(tmp_path / "log", dry_run=False)
    with pytest.raises(SystemExit) as excinfo:
        runner.run("STAGE 1", ["backtest/baseline.py"])
    assert excinfo.value.code == Runner.MEMORY_HALT_RC


def test_an_ordinary_stage_failure_is_recorded_and_does_not_raise(
        tmp_path, monkeypatch):
    """One bad contract must not end a campaign — the same rule the batch
    runner applies per symbol."""
    import subprocess

    class _Fail:
        returncode, stdout, stderr = 1, "boom", "traceback"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Fail())
    runner = Runner(tmp_path / "log", dry_run=False)
    rec = runner.run("STAGE 2", ["backtest/scan.py"])
    assert rec["ok"] is False and rec["rc"] == 1


# --------------------------------------------------------------------------
# 6. The report
# --------------------------------------------------------------------------
def test_the_report_carries_the_requested_leaderboard_columns():
    class _Args:
        score_mode, promote = "vol_normalized", True

    report = render_report(
        [{"strategy": "keltner", "symbol": "6J", "tf": "1h", "version": "A",
          "gate_r": "PASS", "pf": 1.08, "trades": 116, "certified": True,
          "promoted": True, "note": "promoted to incubator-odd"}],
        "2026-09-02T00:00:00+00:00", _Args())

    for column in ("Strategy", "Symbol", "TF", "Ver", "Gate R PF",
                   "Gate R Trades", "Certified", "Promoted"):
        assert column in report
    assert "1.08" in report and "116" in report
    assert Q3_LABEL in report


def test_the_report_names_the_restart_and_the_other_three_units():
    """
    A promotion reaches the live loop only on restart — and the other three
    units do not come back on their own after the NFS mount races the NAS
    boot, so a report that named only the restart would leave an operator with
    a dispatcher running against a dead regime daemon.
    """
    class _Args:
        score_mode, promote = "vol_normalized", True

    report = render_report([], "t", _Args())
    assert "systemctl restart trading-master-live.service" in report
    for unit in ("trading-regime-daemon", "trading-watchdog",
                 "trading-nt8-listener"):
        assert unit in report
    assert "check_market_regime.py --strategies" in report
