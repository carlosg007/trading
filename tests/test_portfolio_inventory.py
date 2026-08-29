"""
`tools/portfolio_inventory.py` - the join across config, package and audit.

WHY THIS SUITE EXISTS
=====================
The inventory's whole value is that it does NOT recompute. Every number is
transcribed from the artifact that recorded it, so the failure to guard against
is a column quietly filled from the wrong place - a metric fabricated where none
was measured, or Gate R's quadrant-scoped profit factor swapped for the
whole-holdout one. Both produce a table that disagrees with the certification it
claims to summarise, and the disagreement reads as a finding.

Nothing here touches the real config or the artifact mount.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import portfolio_inventory as inv                     # noqa: E402


def _audit(path: Path, version: str = "B") -> Path:
    path.write_text(json.dumps({"versions": {version: {
        "metrics_in_sample": {"profit_factor": 1.0748, "trade_count": 1719,
                              "win_rate": 0.5183},
        "metrics_holdout": {"profit_factor": 1.1545, "trade_count": 673},
        "regime_profile_holdout": {
            "optimal_quadrant": "Q2",
            "optimal_regime": "High Volatility / Ranging",
            "optimal_profit_factor": 1.22,
            "optimal_trade_count": 280,
            "kill_switch_conditions": ["High Volatility / Trending",
                                       "Low Volatility / Ranging"]}}}}))
    return path


@pytest.fixture
def world(tmp_path):
    """One portfolio, one Version B package, one audit."""
    inc = tmp_path / "incubator"
    pkg = inc / "demo_NQ_1h_VB"
    pkg.mkdir(parents=True)
    audit = _audit(tmp_path / "gate_audit_NQ_1h.json")
    (pkg / "meta.json").write_text(json.dumps({
        "strategy": "demo", "symbol": "NQ", "timeframe": "1h", "version": "B",
        "ml_threshold": 0.48, "promoted_utc": "2026-08-29T20:47:49+00:00",
        "gate_audit_status": "PASS",
        "certification": {"audit_file": str(audit)}}))
    cfg = {"portfolios": {"Incubator-Odd": {
        "active_strategies": ["demo_NQ_1h_VB"],
        "strategy_allocations": {"demo_NQ_1h_VB": {"symbol": "NQ",
                                                   "timeframe": "1h"}}}}}
    return cfg, inc


def test_a_row_is_transcribed_not_recomputed(world):
    cfg, inc = world
    row, = inv.collect_rows(cfg, inc)
    assert row["strategy_family"] == "demo"
    assert (row["version"], row["ml_threshold"]) == ("B", 0.48)
    assert row["in_sample_pf"] == 1.0748
    assert row["in_sample_trades"] == 1719
    assert row["optimal_regime"] == "Q2 High Volatility / Ranging"
    assert row["disk_status"] == inv.EXISTS
    assert row["promoted_at"] == "2026-08-29T20:47:49+00:00"


def test_gate_r_pf_is_the_quadrant_not_the_whole_holdout(world):
    """
    The two are different measurements and swapping them invents a
    disagreement with the certification. Gate R judges the DESIGNATED
    QUADRANT; metrics_holdout covers all four.
    """
    cfg, inc = world
    row, = inv.collect_rows(cfg, inc)
    assert (row["gate_r_pf"], row["gate_r_trades"]) == (1.22, 280)
    assert (row["oos_holdout_pf"], row["oos_holdout_trades"]) == (1.1545, 673)
    assert row["gate_r_pf"] != row["oos_holdout_pf"]


def test_win_rate_is_a_percentage_not_a_fraction(world):
    """A column headed win_rate showing 0.52 beside one showing 52 is read
    wrong once and trusted afterwards."""
    cfg, inc = world
    row, = inv.collect_rows(cfg, inc)
    assert row["in_sample_win_rate"] == 51.83


def test_a_missing_package_is_reported_not_dropped(tmp_path):
    """The dangling allocation is the thing this tool was written to catch:
    the live dispatcher imports a strategy by that path."""
    cfg = {"portfolios": {"Incubator-Odd": {
        "active_strategies": ["gone_NQ_1h_VA"],
        "strategy_allocations": {"gone_NQ_1h_VA": {"symbol": "NQ",
                                                   "timeframe": "1h"}}}}}
    row, = inv.collect_rows(cfg, tmp_path / "incubator")
    assert row["disk_status"] == inv.MISSING
    assert row["strategy_id"] == "gone_NQ_1h_VA"
    assert inv.MISSING in inv.summarise([row])
    assert "gone_NQ_1h_VA" in inv.summarise([row]), "named, never just counted"


def test_unreadable_evidence_leaves_cells_empty_not_zero(tmp_path):
    """An empty cell and a 0.00 are different claims - one says nobody
    measured, the other says somebody measured nothing."""
    inc = tmp_path / "incubator"
    (inc / "demo_NQ_1h_VA").mkdir(parents=True)      # no meta.json at all
    cfg = {"portfolios": {"P": {
        "active_strategies": ["demo_NQ_1h_VA"],
        "strategy_allocations": {"demo_NQ_1h_VA": {"symbol": "NQ",
                                                   "timeframe": "1h"}}}}}
    row, = inv.collect_rows(cfg, inc)
    assert row["disk_status"] == inv.EXISTS, "the directory IS there"
    assert row["in_sample_pf"] == "" and row["gate_r_pf"] == ""
    assert row["oos_holdout_pf"] == "" and row["optimal_regime"] == ""


def test_version_a_carries_no_ml_threshold(tmp_path):
    """promote.py writes it only for B, so filling 0.48 here would claim a
    filter that is not in the package."""
    inc = tmp_path / "incubator"
    pkg = inc / "demo_NQ_1h_VA"
    pkg.mkdir(parents=True)
    (pkg / "meta.json").write_text(json.dumps({
        "strategy": "demo", "symbol": "NQ", "timeframe": "1h",
        "version": "A"}))
    cfg = {"portfolios": {"P": {"active_strategies": ["demo_NQ_1h_VA"],
                                "strategy_allocations": {}}}}
    row, = inv.collect_rows(cfg, inc)
    assert row["ml_threshold"] == ""


def test_filters(world):
    cfg, inc = world
    rows = inv.collect_rows(cfg, inc)
    assert len(inv.apply_filters(rows, version="B")) == 1
    assert len(inv.apply_filters(rows, version="VB")) == 1, "VB == B"
    assert inv.apply_filters(rows, version="A") == []
    assert len(inv.apply_filters(rows, symbol="nq")) == 1, "case-insensitive"
    assert inv.apply_filters(rows, portfolio="Nope") == [], (
        "an unmatched filter returns nothing, never the whole inventory")


def test_csv_has_the_fixed_column_order(world, tmp_path):
    cfg, inc = world
    out = inv.write_csv(inv.collect_rows(cfg, inc), tmp_path / "r" / "i.csv")
    assert out.exists(), "the parent directory is created"
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert list(rows[0]) == list(inv.COLUMNS), (
        "a sheet whose columns move between runs cannot be diffed")


def test_exit_code_flags_a_dangling_allocation(tmp_path, capsys):
    cfg = tmp_path / "p.json"
    cfg.write_text(json.dumps({"portfolios": {"P": {
        "active_strategies": ["gone_NQ_1h_VA"], "strategy_allocations": {}}}}))
    rc = inv.main(["--config", str(cfg), "--no-csv"])
    assert rc == 1, "non-zero so it chains into a pre-flight check"
