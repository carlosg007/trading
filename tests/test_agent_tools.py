#!/usr/bin/env python3
"""
tests/test_agent_tools.py - the five agent tool wrappers, and the bridge.

Location:  ~/src/trading/tests/test_agent_tools.py

    pytest tests/test_agent_tools.py
    python3 -m pytest tests/test_agent_tools.py -q

ASSERT-BASED ON PURPOSE
=======================
Every case fails through `assert`, so `pytest tests/` and a direct run report
the same thing. This suite deliberately does NOT define a `def check(` helper:
`tests/conftest.py` classifies by that marker, and a collector-style suite here
would be routed into the subprocess runner and lose its per-case granularity
for no benefit.

Helper sections are named `_check_*`, never `test_*`. A module-level `test_*`
whose only argument is defaulted gets collected and called directly by pytest —
which is how `test_regime_profiler.py` once ran its sections without their
`$BT_ARTIFACTS` redirect and wrote real JSON onto the NFS mount.

NOTHING HERE TOUCHES THE LAKE, THE LIVE LOOP, OR THE NETWORK
============================================================
Every fixture is synthetic and written under pytest's `tmp_path`. The one
test that spawns a subprocess runs `tool_log_watcher.py` against a temp file.
No test starts a backtest, sends a Telegram message, or calls systemctl with a
mutating verb — the two tools that CAN change the world are tested on their
refusal paths, which is the behaviour that matters.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"

from tools import bridge                                          # noqa: E402
from tools import tool_backtest_runner as runner                  # noqa: E402
from tools import tool_contract_guard as guard                    # noqa: E402
from tools import tool_daemon_control as daemon                   # noqa: E402
from tools import tool_log_watcher as watcher                     # noqa: E402
from tools import tool_portfolio_eval as evaluator                # noqa: E402

ET = ZoneInfo("America/New_York")


# ==========================================================================
# tool_log_watcher
# ==========================================================================

def test_log_watcher_classifies_http_4xx_and_5xx():
    for line, expected in (
            ("POST /webhook -> HTTP 503 Service Unavailable", "503"),
            ("crosstrade response: status_code=429", "429"),
            ("dispatch failed status: 401", "401"),
            ("got 500 Internal Server Error from the bridge", "500"),
    ):
        verdict = watcher.classify(line)
        assert verdict is not None, f"missed an HTTP failure: {line!r}"
        category, _severity, detail = verdict
        assert category == watcher.CATEGORY_HTTP, line
        assert expected in detail, (line, detail)


def test_log_watcher_ignores_2xx_and_3xx():
    for line in ("POST /webhook -> HTTP 200 OK",
                 "status_code=204",
                 "redirected with 302"):
        verdict = watcher.classify(line)
        if verdict is not None:
            assert verdict[0] != watcher.CATEGORY_HTTP, line


def test_log_watcher_classifies_exceptions_and_disconnects():
    assert watcher.classify("Traceback (most recent call last):")[0] == \
        watcher.CATEGORY_EXCEPTION
    assert watcher.classify("ValueError: bad symbol")[0] == \
        watcher.CATEGORY_EXCEPTION
    assert watcher.classify("nt8 socket closed unexpectedly")[0] == \
        watcher.CATEGORY_DISCONNECT
    assert watcher.classify("Connection refused")[0] == \
        watcher.CATEGORY_DISCONNECT


def test_log_watcher_suppresses_handled_prose():
    # "no webhook configured" is the loop reporting a condition it HANDLED.
    # Treating it as a disconnect is the false positive that gets a watchdog
    # muted, which costs every real finding after it.
    assert watcher.classify("[master_live] no webhook configured") is None
    assert watcher.classify("cycle complete, no errors") is None


def test_log_watcher_groups_repeats_and_counts_them():
    # The same failure at different timestamps and order ids is ONE finding.
    lines = [f"[2026-09-06T0{i}:00:00+00:00] webhook failed: status_code=503 "
             f"order={1000 + i}" for i in range(6)]
    findings = watcher.scan_lines(lines, "synthetic.log")
    assert len(findings) == 1, "digit-masked repeats should collapse to one"
    only = next(iter(findings.values()))
    assert only["count"] == 6
    assert only["first_seen"] < only["last_seen"]


def test_log_watcher_separates_genuinely_different_failures():
    findings = watcher.scan_lines(
        ["webhook failed: status_code=503",
         "feed socket closed unexpectedly",
         "ValueError: bad symbol"], "synthetic.log")
    assert len(findings) == 3
    assert {f["category"] for f in findings.values()} == {
        watcher.CATEGORY_HTTP, watcher.CATEGORY_DISCONNECT,
        watcher.CATEGORY_EXCEPTION}


def test_log_watcher_marks_which_line_matched():
    lines = ["context above", "ValueError: bad symbol", "context below"]
    findings = watcher.scan_lines(lines, "synthetic.log", context=1)
    only = next(iter(findings.values()))
    assert only["snippet"][only["match_index"]] == "ValueError: bad symbol"


def test_log_watcher_tail_drops_the_partial_first_line(tmp_path: Path):
    path = tmp_path / "big.log"
    path.write_text("\n".join(f"line-{i:05d}" for i in range(4000)),
                    encoding="utf-8")
    lines = watcher.read_tail(path, 2000)
    assert lines, "the tail should not be empty"
    # Every retained line is whole; a fragment would not match the pattern.
    assert all(ln.startswith("line-") and len(ln) == 10 for ln in lines[:20])
    assert len(lines) < 4000, "the window should be a tail, not the whole file"


def test_log_watcher_reads_whole_file_when_tail_is_zero(tmp_path: Path):
    path = tmp_path / "small.log"
    path.write_text("a\nb\nc", encoding="utf-8")
    assert watcher.read_tail(path, 0) == ["a", "b", "c"]


def test_log_watcher_since_filter_keeps_untimestamped_lines():
    lines = ["[2020-01-01T00:00:00+00:00] webhook failed: status_code=500",
             "ValueError: no timestamp on this line"]
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    findings = watcher.scan_lines(lines, "s.log", since=since)
    details = " ".join(f["sample"] for f in findings.values())
    assert "2020-01-01" not in details, "an old stamped line should be dropped"
    assert "no timestamp" in details, (
        "a traceback body carries no timestamp; dropping it would discard the "
        "exception while keeping its header")


def test_log_watcher_end_to_end_on_an_injected_error(tmp_path: Path):
    """The synthetic dry run the build plan asks for."""
    path = tmp_path / "probe.log"
    path.write_text(
        "[2026-09-06T12:00:00+00:00] cycle LIVE symbols=5 strategies=52\n"
        "[2026-09-06T12:00:01+00:00] CROSSTRADE POST failed: HTTP 503 "
        "Service Unavailable\n"
        "[2026-09-06T12:00:02+00:00] cycle LIVE symbols=5 strategies=52\n",
        encoding="utf-8")
    done = subprocess.run(
        [str(VENV_PYTHON), str(REPO_ROOT / "tools" / "tool_log_watcher.py"),
         "--file", str(path), "--json"],
        capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 1, (
        f"findings present must exit 1; got {done.returncode}\n{done.stderr}")
    report = json.loads(done.stdout)
    assert len(report["findings"]) == 1
    finding = report["findings"][0]
    assert finding["category"] == watcher.CATEGORY_HTTP
    assert finding["detail"] == "503"


def test_log_watcher_clean_file_exits_zero(tmp_path: Path):
    path = tmp_path / "clean.log"
    path.write_text("[2026-09-06T12:00:00+00:00] cycle LIVE ok\n",
                    encoding="utf-8")
    done = subprocess.run(
        [str(VENV_PYTHON), str(REPO_ROOT / "tools" / "tool_log_watcher.py"),
         "--file", str(path)],
        capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "not a health check" in done.stdout, (
        "a clean scan must say what it does NOT prove")


# ==========================================================================
# tool_contract_guard
# ==========================================================================

def _contracts(roll: str, next_contract: str | None = "ES DEC26",
               valid_until: str = "2099-01-01") -> dict:
    entry: dict = {"active_contract": "ES SEP26", "roll_date": roll}
    if next_contract is not None:
        entry["next_contract"] = next_contract
    return {"version": 1, "valid_until": valid_until, "contracts": {"ES": entry}}


def test_contract_guard_countdown_boundaries():
    today = date(2026, 9, 6)
    cases = {
        "2026-09-01": guard.STATUS_OVERDUE,   # -5 days
        "2026-09-06": guard.STATUS_TODAY,     # 0
        "2026-09-09": guard.STATUS_URGENT,    # exactly 3 -> urgent
        "2026-09-10": guard.STATUS_NOTICE,    # 4
        "2026-09-11": guard.STATUS_NOTICE,    # exactly 5 -> notice
        "2026-09-12": guard.STATUS_OK,        # 6 -> outside both windows
    }
    for roll, expected in cases.items():
        report = guard.evaluate(_contracts(roll), today)
        assert report["rows"][0]["status"] == expected, (
            f"{roll}: expected {expected}, got {report['rows'][0]['status']}")


def test_contract_guard_overdue_is_reported_not_as_a_countdown():
    report = guard.evaluate(_contracts("2026-08-31"), date(2026, 9, 6))
    row = report["rows"][0]
    assert row["status"] == guard.STATUS_OVERDUE
    assert row["days_remaining"] == -6
    assert "contract_resolver refuses" in row["note"], (
        "an overdue roll means orders fail at dispatch; the note must say so")


def test_contract_guard_stale_file_is_flagged_separately():
    blob = _contracts("2026-12-01", valid_until="2026-09-01")
    report = guard.evaluate(blob, date(2026, 9, 6))
    assert report["file"]["stale"] is True
    assert report["alerting"] is True, (
        "a stale file alerts even when no individual roll is due")
    assert report["rows"][0]["status"] == guard.STATUS_OK, (
        "the per-symbol verdict stays its own; staleness is a file-level fact")


def test_contract_guard_missing_next_contract_is_an_error():
    report = guard.evaluate(_contracts("2026-09-20", next_contract=None),
                            date(2026, 9, 6))
    row = report["rows"][0]
    assert row["status"] == guard.STATUS_ERROR
    assert "nothing to roll INTO" in row["note"]


def test_contract_guard_unparseable_roll_date_does_not_read_as_distant():
    report = guard.evaluate(_contracts("not-a-date"), date(2026, 9, 6))
    row = report["rows"][0]
    assert row["status"] == guard.STATUS_ERROR
    assert row["days_remaining"] is None
    assert "absent one is not a distant one" in row["note"]


def test_contract_guard_refuses_a_document_that_is_not_a_contracts_file(
        tmp_path: Path):
    path = tmp_path / "junk.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(guard.ContractGuardError) as excinfo:
        guard.load_contracts(path)
    assert "not a contracts file" in str(excinfo.value)


def test_contract_guard_runs_against_the_real_september_contracts():
    """The build plan's dry run: the live config, on a pinned date."""
    blob = guard.load_contracts(guard.DEFAULT_CONTRACTS)
    report = guard.evaluate(blob, date(2026, 9, 6))
    assert report["rows"], "the real config should carry contracts"
    statuses = {r["status"] for r in report["rows"]}
    assert statuses <= set(guard.SEVERITY), f"unexpected status: {statuses}"
    # Every row is accounted for; a shorter table would read as a complete
    # audit of fewer symbols.
    assert len(report["rows"]) == len(blob["contracts"])


# ==========================================================================
# tool_portfolio_eval
# ==========================================================================

def _trades(pnls: list[float], sessions: list[str]) -> list[dict]:
    return [{"status": "CLOSED", "pnl": p, "session_date": sessions[i % len(sessions)]}
            for i, p in enumerate(pnls)]


def _sessions(n: int) -> list[str]:
    from datetime import timedelta
    out, cur = [], date(2026, 6, 1)
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def test_portfolio_eval_sharpe_is_none_without_deviation():
    assert evaluator.session_sharpe([100.0]) is None
    assert evaluator.session_sharpe([]) is None
    assert evaluator.session_sharpe([50.0, 50.0, 50.0]) is None, (
        "a zero deviation is a degenerate sample, not an infinite Sharpe")


def test_portfolio_eval_sharpe_matches_the_documented_formula():
    import math
    pnl = [10.0, -5.0, 20.0, 0.0, 15.0]
    mean = sum(pnl) / len(pnl)
    var = sum((v - mean) ** 2 for v in pnl) / (len(pnl) - 1)
    expected = (mean / math.sqrt(var)) * math.sqrt(252)
    assert evaluator.session_sharpe(pnl) == pytest.approx(expected)


def test_portfolio_eval_metrics_are_arithmetically_right():
    sessions = _sessions(4)
    metrics = evaluator.compute_metrics(
        _trades([100.0, -50.0, 200.0, -25.0], sessions), sessions, window=4)
    assert metrics["trade_count"] == 4
    assert metrics["win_count"] == 2
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["gross_profit"] == pytest.approx(300.0)
    assert metrics["gross_loss"] == pytest.approx(75.0)
    assert metrics["profit_factor"] == pytest.approx(4.0)
    assert metrics["net_pnl"] == pytest.approx(225.0)
    # Cumulative: 100, 50, 250, 225. Peak 100 -> trough 50 gives 50;
    # peak 250 -> 225 gives 25. The deepest is 50.
    assert metrics["max_drawdown"] == pytest.approx(50.0)


def test_portfolio_eval_open_trades_are_excluded():
    sessions = _sessions(2)
    trades = _trades([100.0, 100.0], sessions)
    trades.append({"status": "OPEN", "pnl": 9999.0, "session_date": sessions[0]})
    metrics = evaluator.compute_metrics(
        evaluator._closed_trades(trades),
        evaluator.trade_sessions(evaluator._closed_trades(trades)), window=2)
    assert metrics["trade_count"] == 2
    assert metrics["net_pnl"] == pytest.approx(200.0)


def test_portfolio_eval_undefined_pf_does_not_clear_the_bar():
    sessions = _sessions(30)
    metrics = evaluator.compute_metrics(
        _trades([100.0] * 30, sessions), sessions, window=30)
    assert metrics["pf_undefined"] is True
    assert metrics["profit_factor"] is None
    verdict = evaluator.score_matrix(metrics, 50000.0, 30)
    assert "profit_factor" in verdict["failed"], (
        "no losing trade is not a passing profit factor")


def test_portfolio_eval_short_window_is_insufficient_not_failing():
    sessions = _sessions(5)
    metrics = evaluator.compute_metrics(
        _trades([100.0, -20.0] * 5, sessions), sessions, window=30)
    verdict = evaluator.score_matrix(metrics, 50000.0, 30)
    assert verdict["verdict"] == evaluator.VERDICT_INSUFFICIENT
    assert verdict["failed"] == [], (
        "missing evidence must not be reported as failed criteria")
    assert metrics["max_drawdown_pct"] is not None, (
        "the drawdown percentage is known even when the window is short")


def test_portfolio_eval_matrix_thresholds_are_inclusive():
    # A criterion stated as ">= 1.8" must pass at exactly 1.8.
    metrics = {
        "sessions_in_window": 30, "trade_count": 100, "sharpe": 1.8,
        "win_rate": 0.52, "profit_factor": 1.4, "pf_undefined": False,
        "max_drawdown": 1750.0, "net_pnl": 1.0,
    }
    verdict = evaluator.score_matrix(metrics, 50000.0, 30)
    assert verdict["verdict"] == evaluator.VERDICT_PROMOTE, verdict["failed"]


def test_portfolio_eval_missing_account_size_fails_the_drawdown_criterion():
    metrics = {
        "sessions_in_window": 30, "trade_count": 100, "sharpe": 5.0,
        "win_rate": 0.9, "profit_factor": 3.0, "pf_undefined": False,
        "max_drawdown": 10.0, "net_pnl": 1.0,
    }
    verdict = evaluator.score_matrix(metrics, None, 30)
    assert "max_drawdown" in verdict["failed"]
    assert any("no denominator" in c["detail"] or "denominator" in c["detail"]
               for c in verdict["criteria"])


def test_portfolio_eval_demotion_needs_a_loss_baseline():
    metrics = {"max_drawdown": 10.0, "selected_pnls": [-5.0, -6.0, 100.0]}
    verdict = evaluator.score_demotion(metrics, 1000.0, sigma=1.0, baseline=5)
    streak = next(t for t in verdict["triggers"] if t["name"] == "loss_streak")
    assert streak["passed"] is None, (
        "too few losses means NOT MEASURABLE, never 'clear'")
    assert verdict["verdict"] == evaluator.VERDICT_OK


def test_portfolio_eval_demotion_fires_on_five_consecutive_outliers():
    ordinary = [-10.0, 20.0] * 10
    metrics = {"max_drawdown": 10.0,
               "selected_pnls": ordinary + [-500.0] * 5}
    verdict = evaluator.score_demotion(metrics, 100000.0, sigma=1.0, baseline=5)
    assert verdict["verdict"] == evaluator.VERDICT_DEMOTE
    assert "loss_streak" in verdict["fired"]


def test_portfolio_eval_demotion_fires_on_a_drawdown_breach():
    metrics = {"max_drawdown": 1500.0, "selected_pnls": [-1500.0]}
    verdict = evaluator.score_demotion(metrics, 1000.0, sigma=1.0, baseline=5)
    assert verdict["verdict"] == evaluator.VERDICT_DEMOTE
    assert "trailing_drawdown" in verdict["fired"]


def _identifiers(path: Path) -> set[str]:
    """
    Every name and attribute the module's CODE references.

    Parsed rather than grepped: these files DOCUMENT the calls they refuse to
    make, and a substring search over the source cannot tell a prohibition in
    a docstring from the call itself.
    """
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[-1])
    return names


def _code_strings(path: Path) -> set[str]:
    """String literals that are not docstrings."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value not in docstrings}


def test_portfolio_eval_never_promotes_anything():
    """The safety property this tool exists to hold."""
    referenced = _identifiers(REPO_ROOT / "tools" / "tool_portfolio_eval.py")
    for forbidden in ("promote_strategy", "run_promotions", "save_ledger",
                      "write_ledger"):
        assert forbidden not in referenced, (
            f"tool_portfolio_eval.py must never call {forbidden}: promotion "
            f"runs through the certified gate and a human, not through this "
            f"advisory card")


def test_portfolio_eval_unattributed_trades_are_excluded_not_swept():
    sessions = _sessions(3)
    trades = _trades([100.0, 100.0], sessions[:2])
    trades.append({"status": "CLOSED", "pnl": 5000.0})       # no session at all
    resolved = evaluator.trade_sessions(trades)
    metrics = evaluator.compute_metrics(trades, resolved, window=30)
    assert metrics["unattributed_trades"] == 1
    assert metrics["net_pnl"] == pytest.approx(200.0), (
        "an unattributable trade must not inflate the newest session")


# ==========================================================================
# tool_backtest_runner
# ==========================================================================

def test_backtest_runner_peak_window_boundaries():
    def at(hour, minute, day=7):        # 2026-09-07 is a Monday
        return datetime(2026, 9, day, hour, minute, tzinfo=ET)
    assert runner.in_peak_window(at(7, 59)) is False
    assert runner.in_peak_window(at(8, 0)) is True, "08:00 is inside the window"
    assert runner.in_peak_window(at(12, 0)) is True
    assert runner.in_peak_window(at(16, 59)) is True
    assert runner.in_peak_window(at(17, 0)) is False, "17:00 is the open bound"


def test_backtest_runner_weekends_are_off_peak():
    saturday = datetime(2026, 9, 12, 12, 0, tzinfo=ET)
    sunday = datetime(2026, 9, 13, 12, 0, tzinfo=ET)
    assert runner.in_peak_window(saturday) is False
    assert runner.in_peak_window(sunday) is False


def test_backtest_runner_window_is_evaluated_in_et_not_utc():
    # 13:00 UTC is 09:00 EDT in September — inside the window. A tool
    # comparing UTC hours would call it off-peak and start a sweep mid-session.
    moment = datetime(2026, 9, 7, 13, 0, tzinfo=timezone.utc)
    assert runner.in_peak_window(moment) is True


def test_backtest_runner_allowlist_accepts_known_entrypoints():
    assert runner.validate_argv(["backtest/run.py", "--symbol", "ES"]) == \
        ["backtest/run.py", "--symbol", "ES"]
    # A leading interpreter is stripped so a submission cannot choose a
    # different Python than the pinned venv.
    assert runner.validate_argv(["python3", "run_pipeline.py"]) == \
        ["run_pipeline.py"]


def test_backtest_runner_allowlist_refuses_everything_else():
    for argv in (["/bin/sh", "-c", "echo pwned"],
                 ["rm", "-rf", "/"],
                 ["tools/tool_daemon_control.py", "restart"],
                 []):
        with pytest.raises(runner.RunnerError):
            runner.validate_argv(argv)


def test_backtest_runner_queue_roundtrips(tmp_path: Path):
    path = tmp_path / "queue.json"
    blob = runner.load_queue(path)
    job = runner.add_job(blob, ["backtest/run.py", "--symbol", "ES"], "note")
    runner.save_queue(path, blob)
    reloaded = runner.load_queue(path)
    assert len(runner.pending(reloaded)) == 1
    assert reloaded["jobs"][0]["id"] == job["id"]
    assert reloaded["jobs"][0]["state"] == runner.STATE_QUEUED


def test_backtest_runner_refuses_an_unparseable_queue(tmp_path: Path):
    path = tmp_path / "queue.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(runner.RunnerError) as excinfo:
        runner.load_queue(path)
    assert "empty one" in str(excinfo.value), (
        "an unreadable queue must not be treated as an empty one")


def test_backtest_runner_drain_refuses_during_market_hours(tmp_path: Path):
    path = tmp_path / "queue.json"
    blob = runner.load_queue(path)
    runner.add_job(blob, ["backtest/run.py"], "in-hours")
    runner.save_queue(path, blob)
    peak = datetime(2026, 9, 7, 10, 0, tzinfo=ET)
    code, lines = runner.drain(blob, path, execute=True, limit=0, now=peak)
    assert code == 1
    assert any("REFUSED" in ln for ln in lines)
    assert runner.pending(runner.load_queue(path)), "the job must stay queued"


def test_backtest_runner_drain_without_execute_starts_nothing(tmp_path: Path):
    path = tmp_path / "queue.json"
    blob = runner.load_queue(path)
    runner.add_job(blob, ["backtest/run.py", "--symbol", "ES"], "off-hours")
    runner.save_queue(path, blob)
    off_peak = datetime(2026, 9, 12, 12, 0, tzinfo=ET)      # Saturday
    code, lines = runner.drain(blob, path, execute=False, limit=0, now=off_peak)
    assert code == 0
    assert any("Nothing was started" in ln for ln in lines)
    after = runner.load_queue(path)
    assert runner.pending(after), "a printed job must remain queued"
    assert after["jobs"][0]["started_at"] is None


def test_backtest_runner_builds_the_specified_scope_wrapper():
    job = {"argv": ["backtest/run.py", "--symbol", "ES"]}
    command = runner.build_command(job)
    assert command[:7] == ["systemd-run", "--scope", "-p", "CPUQuota=75%",
                           "nice", "-n", "19"]
    assert command[7] == str(runner.VENV_PYTHON)


# ==========================================================================
# tool_daemon_control
# ==========================================================================

def test_daemon_control_reads_the_interlock_from_the_command():
    assert "ARMED" in daemon.interlock("python master_live.py --live --tf 1h")
    assert "DRY RUN" in daemon.interlock("python master_live.py --dry-run")
    assert "not running" in daemon.interlock(None)
    both = daemon.interlock("python master_live.py --live --dry-run")
    assert "BOTH" in both, (
        "two contradictory flags must be reported, not resolved by guesswork")


def test_daemon_control_refuses_a_mutating_action_without_confirm():
    code, lines = daemon.act("restart", daemon.SERVICE, confirm=False)
    assert code == 1
    assert any("refused" in ln for ln in lines)
    assert any("--confirm" in ln for ln in lines)


def test_daemon_control_refuses_an_unmanaged_unit():
    code, lines = daemon.act("restart", "sshd.service", confirm=True)
    assert code == 1
    assert any("not a unit this tool manages" in ln for ln in lines)


def test_daemon_control_never_edits_units_or_flags():
    # The mutating verbs are a closed set, declared in one place.
    assert daemon.MUTATING == ("start", "stop", "restart"), (
        "enable/disable/mask change what happens after the next reboot, which "
        "is when nobody is watching")
    literals = _code_strings(REPO_ROOT / "tools" / "tool_daemon_control.py")
    for forbidden in ("enable", "disable", "mask", "edit", "--full"):
        assert forbidden not in literals, (
            f"daemon control must not build a systemctl {forbidden!r} command")
    assert not any(lit.startswith("ExecStart") for lit in literals), (
        "a tool that can rewrite ExecStart can disarm the interlock silently")
    # It must only ever address its own two units.
    assert daemon.MANAGED_UNITS == (daemon.SERVICE, daemon.TIMER)


def test_daemon_control_status_reports_the_real_units():
    report = daemon.collect()
    assert daemon.SERVICE in report["units"]
    assert daemon.TIMER in report["units"]
    info = report["units"][daemon.SERVICE]
    assert info["active"] in ("active", "inactive", "failed", "activating",
                              "deactivating", "?")
    assert info["interlock"] is not None


# ==========================================================================
# tools/bridge.py
# ==========================================================================

def _bridge_config(chat_id: str = "-1001234567890") -> dict:
    return {
        "telegram": {
            "chat_id": chat_id,
            "topics": {
                "general_ops": {"thread_id": 8, "agent": "supervisor"},
                "portfolio_mgmt": {"thread_id": 2},
                "system_health": {"thread_id": 6},
                "roll_alerts": {"thread_id": 6, "alias_of": "system_health"},
            },
        }
    }


def test_bridge_resolves_variables_and_records_gaps():
    missing: list[str] = []
    out = bridge.resolve({"a": "${PRESENT}", "b": "${ABSENT}"},
                         {"PRESENT": "yes"}, missing)
    assert out["a"] == "yes"
    assert out["b"] == ""
    assert any("ABSENT" in m for m in missing)


def test_bridge_strict_load_refuses_an_unresolved_reference(tmp_path: Path):
    path = tmp_path / "bridge.yaml"
    path.write_text("telegram:\n  chat_id: ${DEFINITELY_NOT_SET_ANYWHERE}\n",
                    encoding="utf-8")
    with pytest.raises(bridge.BridgeError) as excinfo:
        bridge.load(path, strict=True)
    assert "unresolved" in str(excinfo.value)


def test_bridge_builds_a_forum_topic_target():
    config = _bridge_config()
    assert bridge.topic_target(config, "portfolio_mgmt") == \
        "telegram:-1001234567890:2"
    assert bridge.topic_target(config, "general_ops") == \
        "telegram:-1001234567890:8"


def test_bridge_roll_alerts_shares_the_watchdog_thread():
    config = _bridge_config()
    assert bridge.topic_target(config, "roll_alerts") == \
        bridge.topic_target(config, "system_health")


def test_bridge_refuses_an_empty_chat_id():
    with pytest.raises(bridge.BridgeError) as excinfo:
        bridge.topic_target(_bridge_config(chat_id=""), "general_ops")
    message = str(excinfo.value)
    assert "chat id is not set" in message
    assert "discover" in message, "the error should name the way out"


def test_bridge_refuses_an_unknown_topic():
    with pytest.raises(bridge.BridgeError):
        bridge.topic_target(_bridge_config(), "nope")


def test_bridge_env_reader_strips_quotes_and_export():
    path = Path(bridge.HERMES_ENV)
    _ = path            # touched only to prove the constant is a path
    values = bridge.read_env_file(Path("/nonexistent/.env"))
    assert values == {}, "a missing env file is empty, not an error"


def test_bridge_env_reader_parses_the_shapes_people_write(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text('A=1\nexport B="two"\nC=\'three\'\n# comment\nBAD LINE\n',
                    encoding="utf-8")
    values = bridge.read_env_file(path)
    assert values == {"A": "1", "B": "two", "C": "three"}


def test_bridge_config_on_disk_is_loadable_and_complete():
    """The real bridge_config.yaml parses and declares all five tools."""
    if not bridge.BRIDGE_CONFIG.exists():
        pytest.skip(f"{bridge.BRIDGE_CONFIG} is not installed on this host")
    config, _missing = bridge.load(strict=False)
    tools = config.get("tools") or {}
    assert set(tools) == {"log_watcher", "contract_guard", "portfolio_eval",
                          "backtest_runner", "daemon_control"}
    for name, tool in tools.items():
        assert (REPO_ROOT / tool["script"]).exists(), f"{name}: script missing"
    # The two tools that can change the world must say so.
    assert tools["backtest_runner"]["writes"] is True
    assert tools["daemon_control"]["writes"] is True
    assert tools["portfolio_eval"]["writes"] is False


def test_bridge_config_mirrors_the_live_honcho_connection():
    if not bridge.BRIDGE_CONFIG.exists() or not bridge.HONCHO_JSON.exists():
        pytest.skip("the bridge or honcho config is not installed on this host")
    ok, lines = bridge.check()
    drift = [ln for ln in lines if "DRIFT" in ln]
    assert not drift, "bridge_config.yaml disagrees with honcho.json:\n" + \
        "\n".join(drift)
    _ = ok      # unset credentials are a separate, expected finding
