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
        "gate_audit": {"gates": {"gate_regime": {
            "status": "PASS",
            "measured": {"profit_factor": 1.22, "trade_count": 280}}}},
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
        # Q4 here DISAGREES with the audit's `regime_profile_holdout`, which
        # says Q2. That disagreement is the point: it is the only way a test
        # can tell which source the tool actually read.
        "certification": {"audit_file": str(audit),
                          "target_quadrant": "Q4",
                          "target_regime": "Low Volatility / Ranging"}}))
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
    assert row["target_quadrant"] == "Q4"
    assert row["target_regime"] == "Low Volatility / Ranging"
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
    assert "report_html" in row
    assert row["gate_r_pf"] != row["oos_holdout_pf"]


def test_gate_r_comes_from_the_gate_not_the_holdout_profile(tmp_path):
    """
    Both blocks carry an "optimal" profit factor and they are NOT the same
    measurement. `regime_profile_holdout` describes the best quadrant WITHIN
    THE HOLDOUT; Gate R judges the quadrant DESIGNATED IN SAMPLE.

    Reading the profile reported sma_momentum GC 1h Version B as certified on
    0 trades - its holdout had no dominant quadrant - when Gate R had measured
    46 trades at PF 1.20 in Q1 and passed it. A read-only summary must not
    manufacture a "certified on nothing" finding out of the wrong field.
    """
    inc = tmp_path / "inc"
    pkg = inc / "demo_GC_1h_VB"
    pkg.mkdir(parents=True)
    audit = tmp_path / "a.json"
    audit.write_text(json.dumps({"versions": {"B": {
        "metrics_holdout": {"profit_factor": 1.0018, "trade_count": 93},
        "gate_audit": {"gates": {"gate_regime": {
            "measured": {"profit_factor": 1.20, "trade_count": 46}}}},
        # The holdout had no dominant quadrant. This is the trap.
        "regime_profile_holdout": {"optimal_profit_factor": None,
                                   "optimal_trade_count": 0,
                                   "optimal_regime": None}}}}))
    (pkg / "meta.json").write_text(json.dumps({
        "strategy": "demo", "symbol": "GC", "timeframe": "1h", "version": "B",
        "certification": {"audit_file": str(audit)}}))
    cfg = {"portfolios": {"P": {"active_strategies": ["demo_GC_1h_VB"],
                                "strategy_allocations": {}}}}
    row, = inv.collect_rows(cfg, inc)
    assert (row["gate_r_pf"], row["gate_r_trades"]) == (1.2, 46), (
        "the gate's own measurement, not the holdout profile's 0")
    assert row["gate_r_trades"] != 0


def test_report_html_resolves_or_says_missing(tmp_path, monkeypatch):
    """MISSING rather than a guessed path - a column offering a file that is
    not there wastes more time than an empty one."""
    root = tmp_path / "artifacts"
    run = root / "pipeline" / "demo" / "verify_20260829_120000"
    run.mkdir(parents=True)
    (run / "report_NQ_version_b.html").write_text("<html/>")
    # A NEWER run wins, because a re-run leaves the old tearsheet describing
    # parameters the promoted module no longer uses.
    newer = root / "pipeline" / "demo" / "verify_20260830_090000"
    newer.mkdir(parents=True)
    (newer / "report_NQ_version_b.html").write_text("<html/>")
    monkeypatch.setenv("BT_ARTIFACTS", str(root))

    meta = {"strategy": "demo", "symbol": "NQ"}
    assert inv._report_html(meta, "B").endswith(
        "verify_20260830_090000/report_NQ_version_b.html")
    assert inv._report_html({"strategy": "demo", "symbol": "ES"}, "B") == "MISSING"
    assert inv._report_html({}, "B") == "MISSING"
    assert inv._report_html(meta, None) == "MISSING"


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
    assert row["oos_holdout_pf"] == ""
    assert row["target_quadrant"] == "" and row["target_regime"] == ""


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


def test_the_quadrant_comes_from_the_seal_not_the_holdout_profile(world):
    """The provenance fix of 2026-08-31.

    The tool used to read `regime_profile_holdout.optimal_quadrant`, which is
    the best-of-four quadrant WITHIN THE HOLDOUT. Gate R judges the quadrant
    DESIGNATED IN SAMPLE, and re-picking the best of four on the holdout is
    exactly the selection Gate R exists to prevent. Across the real incubator
    the two disagreed on 37 of 118 packages.

    The fixture's seal says Q4 and its holdout profile says Q2, so this fails
    on any implementation that reads the profile.
    """
    cfg, inc = world
    row, = inv.collect_rows(cfg, inc)
    assert row["target_quadrant"] == "Q4", (
        "the holdout profile's Q2 leaked into a column that names the "
        "certified quadrant")


def test_a_seal_with_no_quadrant_reports_empty_not_a_guess(world):
    """No fallback, on purpose.

    Two other sources are readable and both are wrong - the holdout profile
    above, and `TARGET_QUADRANTS` in the module source, which is the premise's
    NOMINATION rather than anything Stage 1 measured. An empty cell says
    nobody recorded it; a guess from either would say something false in a
    column a reader has no way to check.
    """
    cfg, inc = world
    meta_path = inc / "demo_NQ_1h_VB" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["certification"].pop("target_quadrant")
    meta["certification"].pop("target_regime")
    meta_path.write_text(json.dumps(meta))
    row, = inv.collect_rows(cfg, inc)
    assert row["target_quadrant"] == "", (
        "a missing seal quadrant must be EMPTY, never filled from the "
        "holdout profile or the module source")
    assert row["target_regime"] == ""


def test_certified_regime_reads_only_the_certification_block():
    """Unit-level: the resolver ignores every other quadrant-shaped field."""
    assert inv._certified_regime({}) == ("", "")
    assert inv._certified_regime({"certification": None}) == ("", "")
    # A module-source nomination and a holdout profile, both present, both
    # ignored.
    meta = {"TARGET_QUADRANTS": ("Q3", "Q1"),
            "regime_profile_holdout": {"optimal_quadrant": "Q2"},
            "certification": {"target_quadrant": "Q4",
                              "target_regime": "Low Volatility / Ranging"}}
    assert inv._certified_regime(meta) == ("Q4", "Low Volatility / Ranging")


def test_unallocated_packages_are_found_and_marked(world):
    """A promoted package no portfolio routes is invisible on the default
    board, because the board is keyed on allocations.

    On 2026-08-31 eighty packages were in that state and four of them were the
    Q4 pair somebody went looking for in the CSV.
    """
    cfg, inc = world
    stray = inc / "demo_ES_5m_VA"
    stray.mkdir(parents=True)
    (stray / "meta.json").write_text(json.dumps({
        "strategy": "demo", "symbol": "ES", "timeframe": "5m", "version": "A",
        "certification": {"target_quadrant": "Q3",
                          "target_regime": "Low Volatility / Trending"}}))

    assert [r["strategy_id"] for r in inv.collect_rows(cfg, inc)] == \
        ["demo_NQ_1h_VB"], "the default board must not grow"

    extra = inv.collect_unallocated_rows(cfg, inc)
    assert [r["strategy_id"] for r in extra] == ["demo_ES_5m_VA"]
    row, = extra
    assert row["portfolio_name"] == inv.UNALLOCATED
    assert row["disk_status"] == inv.EXISTS
    # The evidence is transcribed the same way as for an allocated row.
    assert row["target_quadrant"] == "Q3"
    assert row["version"] == "A"


def test_a_directory_without_meta_is_not_reported_as_a_strategy(world):
    """A stray directory is not a promoted package, and inventing one would be
    the same class of false finding as a guessed quadrant."""
    cfg, inc = world
    (inc / "not_a_package").mkdir(parents=True)
    assert inv.collect_unallocated_rows(cfg, inc) == []


def test_both_collectors_build_a_row_the_same_way(world):
    """One row builder, on purpose.

    An allocated row and an unallocated one differ in exactly `portfolio_name`
    - a second builder would be free to read a different field and the two
    tables would disagree about a package while both looked right.
    """
    cfg, inc = world
    allocated, = inv.collect_rows(cfg, inc)
    # Same package, reached the other way.
    unrouted, = inv.collect_unallocated_rows(
        {"portfolios": {}}, inc)
    differing = {k for k in allocated
                 if allocated[k] != unrouted.get(k)}
    assert differing == {"portfolio_name"}, differing


def test_allocated_ids_spans_every_portfolio(world):
    cfg, _inc = world
    cfg["portfolios"]["Second"] = {"active_strategies": ["other_GC_1h_VA"]}
    assert inv.allocated_ids(cfg) == {"demo_NQ_1h_VB", "other_GC_1h_VA"}
