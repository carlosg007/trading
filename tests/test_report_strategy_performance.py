"""
tests/test_report_strategy_performance.py -
`scripts/report_strategy_performance.py`.

ASSERT-BASED, so `tests/conftest.py` collects it normally. No `def check(`
marker: that is what routes a suite to the subprocess runner.

Every case builds its own audit files under `tmp_path`. Nothing reads the real
artifacts root - 557 audits on an NFS mount is not a unit test, and a suite
that asserted against them would fail whenever somebody ran a campaign.

WHAT THIS IS GUARDING
=====================
The tool transcribes; the failures worth catching are all transcription
errors, and every one of them produces a plausible-looking number:

  * `status` read as a string when it is `{"A": "FAIL", "B": "PASS"}`, which
    collapses a passing Version B into a failing Version A
  * `win_rate` (a FRACTION on the artifact) printed under a column headed
    `Win_Rate_Pct`
  * the unsuffixed `gate_audit_<SYMBOL>.json` counted beside the per-pair
    file, putting one configuration on the leaderboard twice
  * an in-sample profit factor ranked against out-of-sample ones
  * a row with no metrics sorting as though it scored zero
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_strategy_performance import (                  # noqa: E402
    COLUMNS,
    build_parser,
    collect,
    main,
    render,
    rows_from_audit,
    sort_rows,
    write_csv,
)

HELPERS = REPO_ROOT / "deploy" / "shell" / "trading_helpers.sh"


def _metrics(pf=1.25, sharpe=0.9, dd=-7.5, win=0.42, trades=120) -> dict:
    """The REAL key names, as `metrics_holdout` carries them."""
    return {"profit_factor": pf, "sharpe": sharpe, "max_drawdown_pct": dd,
            "win_rate": win, "trade_count": trades}


def _audit(directory: Path, name: str, *, strategy="strat_x", symbol="NQ",
           timeframe="1h", status=None, versions=None, generated="2026-09-01",
           holdout_key="metrics_holdout") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    versions = versions or {"A": _metrics()}
    blob = {
        "strategy": strategy, "symbol": symbol, "timeframe": timeframe,
        "generated_utc": generated,
        "status": status if status is not None
        else {v: "PASS" for v in versions},
        "versions": {v: {holdout_key: m} for v, m in versions.items()},
    }
    path = directory / name
    path.write_text(json.dumps(blob))
    return path


# --------------------------------------------------------------------------
# 1. Reading one audit
# --------------------------------------------------------------------------
def test_one_row_per_version_because_status_is_per_version(tmp_path):
    """
    `{"A": "FAIL", "B": "PASS"}` is the shape on disk and the two disagree
    routinely. Collapsed to one row, the passing half disappears.
    """
    path = _audit(tmp_path, "gate_audit_NQ_1h.json",
                  status={"A": "FAIL", "B": "PASS"},
                  versions={"A": _metrics(pf=0.9), "B": _metrics(pf=1.4)})
    rows = rows_from_audit(path)
    assert [r["Ver"] for r in rows] == ["A", "B"]
    assert [r["Status"] for r in rows] == ["FAIL", "PASS"]
    assert rows[1]["Profit_Factor"] == pytest.approx(1.4)


def test_the_real_metric_key_names_are_read(tmp_path):
    rows = rows_from_audit(_audit(tmp_path, "gate_audit_NQ_1h.json"))
    r = rows[0]
    assert r["Profit_Factor"] == pytest.approx(1.25)
    assert r["Sharpe"] == pytest.approx(0.9)
    assert r["Max_DD_Pct"] == pytest.approx(-7.5)
    assert r["Total_Trades"] == pytest.approx(120)


def test_win_rate_is_a_fraction_on_disk_and_a_percent_in_the_column(tmp_path):
    """
    THE SCALE TRAP. `win_rate` is 0.42 on the artifact and the column is headed
    `Win_Rate_Pct`. Printed unconverted it reads as a 0.42% win rate, which is
    a plausible-looking number and wrong by a factor of a hundred.
    """
    rows = rows_from_audit(_audit(tmp_path, "gate_audit_NQ_1h.json",
                                  versions={"A": _metrics(win=0.42)}))
    assert rows[0]["Win_Rate_Pct"] == pytest.approx(42.0)


def test_the_requests_key_spellings_are_accepted_too(tmp_path):
    """`sharpe_ratio`/`max_drawdown`/`total_trades` are not what the schema
    uses, but a rename in either direction must not empty a column."""
    path = _audit(tmp_path, "gate_audit_NQ_1h.json", versions={"A": {
        "profit_factor": 1.1, "sharpe_ratio": 0.5, "max_drawdown": -3.0,
        "win_rate": 0.5, "total_trades": 80}})
    r = rows_from_audit(path)[0]
    assert r["Sharpe"] == pytest.approx(0.5)
    assert r["Max_DD_Pct"] == pytest.approx(-3.0)
    assert r["Total_Trades"] == pytest.approx(80)


def test_a_bare_string_status_still_reads(tmp_path):
    """An audit written before `status` became per-version."""
    rows = rows_from_audit(_audit(tmp_path, "gate_audit_NQ_1h.json",
                                  status="PASS"))
    assert rows[0]["Status"] == "PASS"


def test_a_malformed_file_yields_no_rows_rather_than_raising(tmp_path):
    bad = tmp_path / "gate_audit_NQ_1h.json"
    bad.write_text("{not json")
    assert rows_from_audit(bad) == []


# --------------------------------------------------------------------------
# 2. Out of sample is preferred, and the choice is reported
# --------------------------------------------------------------------------
def test_the_holdout_block_wins_over_the_in_sample_one(tmp_path):
    """
    An in-sample profit factor and a holdout one are different claims. Mixed
    silently under one heading, a configuration fitted on those bars outranks
    one measured on bars it never saw.
    """
    directory = tmp_path / "s"
    directory.mkdir()
    (directory / "gate_audit_NQ_1h.json").write_text(json.dumps({
        "strategy": "s", "symbol": "NQ", "timeframe": "1h",
        "status": {"A": "PASS"},
        "versions": {"A": {"metrics_in_sample": _metrics(pf=9.9),
                           "metrics_holdout": _metrics(pf=1.1)}}}))
    row = rows_from_audit(directory / "gate_audit_NQ_1h.json")[0]
    assert row["Profit_Factor"] == pytest.approx(1.1)
    assert row["_metrics_source"] == "metrics_holdout"

    in_sample = rows_from_audit(directory / "gate_audit_NQ_1h.json",
                                prefer_oos=False)[0]
    assert in_sample["Profit_Factor"] == pytest.approx(9.9)
    assert in_sample["_metrics_source"] == "metrics_in_sample"


def test_the_footer_names_which_block_was_read(tmp_path):
    """The table must say whether it is showing in-sample or holdout numbers;
    the two are different claims and the heading is the same either way."""
    _audit(tmp_path / "s", "gate_audit_NQ_1h.json")
    rows, counts = collect(tmp_path)
    assert "metrics_holdout" in render(rows, counts)


# --------------------------------------------------------------------------
# 3. The unsuffixed duplicate
# --------------------------------------------------------------------------
def test_an_unsuffixed_audit_beside_a_per_pair_one_is_skipped(tmp_path):
    """
    Stage 3 writes both, and the unsuffixed file holds whichever timeframe ran
    LAST. Counting both puts one configuration on the leaderboard twice, under
    a timeframe it may not belong to.
    """
    directory = tmp_path / "s"
    _audit(directory, "gate_audit_NQ_1h.json")
    _audit(directory, "gate_audit_NQ.json")
    rows, counts = collect(tmp_path)
    assert len(rows) == 1
    assert counts["skipped_unsuffixed"] == 1


def test_an_unsuffixed_audit_with_no_sibling_is_kept(tmp_path):
    """It is the only record of that configuration; dropping it would hide a
    certification rather than a duplicate."""
    _audit(tmp_path / "s", "gate_audit_ES.json", symbol="ES")
    rows, counts = collect(tmp_path)
    assert len(rows) == 1
    assert counts["skipped_unsuffixed"] == 0


def test_include_unsuffixed_keeps_both(tmp_path):
    directory = tmp_path / "s"
    _audit(directory, "gate_audit_NQ_1h.json")
    _audit(directory, "gate_audit_NQ.json")
    rows, _counts = collect(tmp_path, include_unsuffixed=True)
    assert len(rows) == 2


# --------------------------------------------------------------------------
# 4. Sorting
# --------------------------------------------------------------------------
def test_a_row_with_no_profit_factor_sorts_last_not_as_zero():
    """
    "This audit recorded no metrics" is a finding about the artifact. Sorted
    as 0.0 it would sit among the genuinely losing configurations and read as
    one of them.
    """
    rows = [{"Profit_Factor": None, "Sharpe": None, "_generated": "c"},
            {"Profit_Factor": 0.5, "Sharpe": 0.1, "_generated": "b"},
            {"Profit_Factor": 2.0, "Sharpe": 1.0, "_generated": "a"}]
    got = sort_rows(rows, "pf")
    assert [r["Profit_Factor"] for r in got] == [2.0, 0.5, None]


def test_timestamp_sort_is_newest_first():
    rows = [{"Profit_Factor": 1.0, "_generated": "2026-01-01"},
            {"Profit_Factor": 1.0, "_generated": "2026-09-01"}]
    assert sort_rows(rows, "timestamp")[0]["_generated"] == "2026-09-01"


# --------------------------------------------------------------------------
# 5. The CSV
# --------------------------------------------------------------------------
def test_the_csv_carries_the_declared_columns_and_parses_as_numbers(tmp_path):
    _audit(tmp_path / "s", "gate_audit_NQ_1h.json")
    rows, _counts = collect(tmp_path)
    path, latest = write_csv(rows, tmp_path / "reports")

    parsed = list(csv.DictReader(path.open()))
    assert list(parsed[0]) == COLUMNS
    for column in ("Profit_Factor", "Sharpe", "Max_DD_Pct", "Win_Rate_Pct",
                   "Total_Trades"):
        float(parsed[0][column])            # raises if it is not a number
    assert latest.read_text() == path.read_text(), (
        "the canonical latest file does not match the stamped one")


def test_latest_is_a_copy_rather_than_a_symlink(tmp_path):
    """
    The reports directory is read by whatever an operator points at it. A
    dangling symlink after the stamped file is tidied away reads as an EMPTY
    report rather than a missing one.
    """
    _audit(tmp_path / "s", "gate_audit_NQ_1h.json")
    rows, _counts = collect(tmp_path)
    _path, latest = write_csv(rows, tmp_path / "reports")
    assert not latest.is_symlink()
    assert latest.is_file() and latest.read_text().strip()


def test_a_missing_metric_is_an_empty_cell_not_a_zero(tmp_path):
    """A zero is a measurement a configuration can legitimately have."""
    _audit(tmp_path / "s", "gate_audit_NQ_1h.json",
           versions={"A": {"profit_factor": 1.2}})
    rows, _counts = collect(tmp_path)
    path, _latest = write_csv(rows, tmp_path / "reports")
    parsed = list(csv.DictReader(path.open()))[0]
    assert parsed["Profit_Factor"] == "1.2"
    assert parsed["Sharpe"] == ""


# --------------------------------------------------------------------------
# 6. The CLI
# --------------------------------------------------------------------------
def test_min_trades_defaults_to_showing_everything():
    """Nothing is hidden unless asked for - but the flag exists because
    ranking on profit factor puts the thinnest samples on top."""
    assert build_parser().parse_args([]).min_trades == 0
    assert build_parser().parse_args(["--min-trades", "30"]).min_trades == 30


def test_a_thin_sample_warning_is_printed(tmp_path):
    _audit(tmp_path / "s", "gate_audit_NQ_1h.json",
           versions={"A": _metrics(pf=9.0, trades=4)})
    rows, counts = collect(tmp_path)
    assert "FEWER THAN 30" in render(rows, counts)


def test_an_empty_artifacts_root_exits_non_zero(tmp_path, capsys):
    rc = main(["--artifacts", str(tmp_path), "--no-csv"])
    assert rc == 1
    assert "no gate audits found" in capsys.readouterr().out


def test_a_missing_artifacts_root_is_reported(tmp_path, capsys):
    rc = main(["--artifacts", str(tmp_path / "nope"), "--no-csv"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_the_filters_narrow_the_table(tmp_path, capsys):
    _audit(tmp_path / "a", "gate_audit_NQ_1h.json", strategy="alpha_one")
    _audit(tmp_path / "b", "gate_audit_ES_30m.json", strategy="beta_two",
           symbol="ES", timeframe="30m")
    assert main(["--artifacts", str(tmp_path), "--strategy", "alpha",
                 "--no-csv"]) == 0
    out = capsys.readouterr().out
    assert "alpha_one" in out and "beta_two" not in out


# --------------------------------------------------------------------------
# 7. The alias
# --------------------------------------------------------------------------
def test_the_alias_uses_the_venv_python():
    """
    `~/.bashrc` sources this file, so the alias reaches an interactive shell
    either way - and `_TRADING_PY` is the venv interpreter, which is the half
    the request asked for explicitly. A bare `python3` would miss pandas.
    """
    text = HELPERS.read_text()
    assert 'alias strat-perf="${_TRADING_PY} ' in text
    assert "report_strategy_performance.py" in text
