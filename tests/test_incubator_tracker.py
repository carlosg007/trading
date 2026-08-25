#!/usr/bin/env python3
"""
test_incubator_tracker.py — the forward-incubation promotion rule and the file
surgery that acts on it: that each of the four criteria fails on its own,
that a promotion actually moves a strategy between accounts in both files, and
that the failure modes leave the routing table where it was.

Location:  ~/src/trading/tests/test_incubator_tracker.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_incubator_tracker.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_incubator_tracker.py

EVERY CASE FAILS THROUGH `assert`, deliberately, like `test_portfolio_config.py`
and `test_portfolio_manager.py`. The older convention in this directory (a
`check(name, ok)` helper and `sys.exit(1)` in `main`) is invisible to pytest —
see `conftest.py`.

Nothing here needs the lake, a network or a Discord webhook. Every file the
promotion cases touch is a COPY in `tmp_path`: a test that promoted a strategy
in the real `config/portfolios.json` would edit a live routing table, and the
change would look exactly like one a human made on purpose.

WHAT THIS COVERS, and why each case is here rather than assumed:

  * **EACH CRITERION FAILS ALONE.** The four cases from the objective hold
    three criteria comfortably clear and break one. A single "everything is
    wrong" fixture would pass just as well against a function that returned
    False unconditionally, and the four are fixed by completely different work
    — a short sample is time, a dead profit factor is the strategy.
  * **THE DRAWDOWN BAR IS DERIVED, NOT TYPED.** $2,500 x 0.40 = $1,000. The
    $1,250 case is a breach and the $450 case is not, and the case pins the bar
    against the config rather than against the literal, so a config edit moves
    the test with it.
  * **A MISSING METRIC IS A FAIL, NEVER A ZERO.** An absent trade count is an
    unfilled ledger, not a strategy that placed no trades, and defaulting it
    would make those indistinguishable at the moment an account is handed over.
  * **THE PROMOTION MOVES BOTH FILES OR NEITHER.** `Incubator-Odd` ->
    `Prop-Odd` in `portfolios.json` AND `INCUBATING` -> `GRADUATED_PROP` in the
    ledger, with the strategy REMOVED from the source list — an append that
    left it in place would be a second live assignment of the same strategy,
    which is the one outcome that doubles a position.
  * **THE ROUTE IS A TABLE, NOT A NAME.** A portfolio that is not an incubator
    raises rather than promoting onto a `Prop-` account derived from its name.
  * **DERIVED METRICS BEAT A STALE SUMMARY, AND A CONTRADICTION FAILS.** An
    entry carrying its own closed trades is scored on them; when the summary
    beside them disagrees, the evaluation fails rather than picking one.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio import config_loader                                # noqa: E402
from portfolio.promotion_daemon import (                           # noqa: E402
    MIN_DAYS_ACTIVE,
    MIN_PROFIT_FACTOR,
    MIN_TRADE_COUNT,
    STATUS_GRADUATED,
    STATUS_INCUBATING,
    PromotionError,
    allowable_forward_dd,
    evaluate_strategy_promotion,
    extract_metrics,
    promote_strategy,
    target_portfolio_for,
)

CONFIG_PATH = REPO_ROOT / "config" / "portfolios.json"
LEDGER_PATH = REPO_ROOT / "data" / "incubator_ledger.json"

# The account profile every case is scored against: $2,500 trailing limit at
# 40%, so $1,000 of allowable forward drawdown on the $50,000 profile. Read
# from the real config rather than restated, so an edit there moves the tests.
PORTFOLIO = config_loader.load_portfolio_config()["portfolios"]["Incubator-Odd"]
ALLOWABLE_DD = allowable_forward_dd(PORTFOLIO)


def entry(days: int, trades: int, pf: float, max_dd: float,
          **extra) -> dict:
    """A ledger entry that declares its metrics — the shape a tracker writes
    after a session, and the shape the objective's four cases are stated in."""
    return {
        "status": STATUS_INCUBATING,
        "portfolio": "Incubator-Odd",
        "days_active": days,
        "trade_count": trades,
        "realized_pf": pf,
        "max_drawdown": max_dd,
        **extra,
    }


# --------------------------------------------------------------------------
# the allowable drawdown
# --------------------------------------------------------------------------

def test_allowable_dd_is_a_fraction_of_the_trailing_limit() -> None:
    """
    $2,500 x 0.40 = $1,000, which is 2% of the $50,000 account and not 40% of
    it. The two readings differ by a factor of twenty and both produce a
    plausible dollar figure, so the basis is pinned as well as the value.
    """
    profile = PORTFOLIO["risk_profile"]
    expected = (profile["max_trailing_drawdown_usd"]
                * profile["max_forward_incubation_dd_pct"])
    assert ALLOWABLE_DD == pytest.approx(expected)
    assert ALLOWABLE_DD == pytest.approx(1000.0)
    assert ALLOWABLE_DD != pytest.approx(
        PORTFOLIO["default_account_size"]
        * profile["max_forward_incubation_dd_pct"])


def test_allowable_dd_from_a_raw_config_matches_the_derived_one() -> None:
    """A config that never went through the loader has no `derived` block, and
    the fallback product must give the same bar — otherwise the criterion moves
    depending on which reader loaded the file."""
    raw = json.loads(CONFIG_PATH.read_text())["portfolios"]["Incubator-Odd"]
    assert "derived" not in raw
    assert allowable_forward_dd(raw) == pytest.approx(ALLOWABLE_DD)


# --------------------------------------------------------------------------
# the four criteria, each failing alone
# --------------------------------------------------------------------------

def test_a_qualifying_strategy_passes() -> None:
    """15 days, 18 trades, PF 1.15, max DD $450 — every bar cleared."""
    report = evaluate_strategy_promotion("qualifier",
                                         entry(15, 18, 1.15, 450.0),
                                         PORTFOLIO)
    assert report["passed"] is True, report["reasons"]
    assert report["failed"] == []
    assert report["strategy_id"] == "qualifier"
    assert all(r.startswith("PASS") for r in report["reasons"]), report["reasons"]
    assert report["metrics"]["allowable_dd"] == pytest.approx(ALLOWABLE_DD)
    # The four criteria are each reported, whether they passed or not: a report
    # that listed only failures would make "cleared" and "not checked" the same
    # absent line.
    assert {c["id"] for c in report["criteria"]} == {
        "window", "sample", "expectancy", "drawdown"}


def test_failing_trade_count() -> None:
    """15 days, 10 trades, PF 1.40, max DD $300. The strongest profit factor of
    the four cases, on ten trades — which is the point: expectancy measured on
    a sample this small is not measured."""
    report = evaluate_strategy_promotion("thin_sample",
                                         entry(15, 10, 1.40, 300.0),
                                         PORTFOLIO)
    assert report["passed"] is False
    assert report["failed"] == ["sample"], report["reasons"]
    reason = next(r for r in report["reasons"] if r.startswith("FAIL sample"))
    assert "10 closed forward trades" in reason
    assert f">= {MIN_TRADE_COUNT}" in reason


def test_failing_profit_factor() -> None:
    """16 days, 20 trades, PF 0.92, max DD $600. Enough evidence, and what it
    evidences is a loss."""
    report = evaluate_strategy_promotion("no_edge",
                                         entry(16, 20, 0.92, 600.0),
                                         PORTFOLIO)
    assert report["passed"] is False
    assert report["failed"] == ["expectancy"], report["reasons"]
    reason = next(r for r in report["reasons"]
                  if r.startswith("FAIL expectancy"))
    assert "0.92" in reason and f"{MIN_PROFIT_FACTOR:.2f}" in reason


def test_failing_drawdown() -> None:
    """16 days, 25 trades, PF 1.20, max DD $1,250 against a $1,000 envelope. A
    profitable strategy is still refused: the bar is the account's, not the
    strategy's."""
    report = evaluate_strategy_promotion("too_deep",
                                         entry(16, 25, 1.20, 1250.0),
                                         PORTFOLIO)
    assert report["passed"] is False
    assert report["failed"] == ["drawdown"], report["reasons"]
    reason = next(r for r in report["reasons"] if r.startswith("FAIL drawdown"))
    assert "1,250" in reason and "1,000" in reason


def test_failing_window() -> None:
    """13 days is one short of the fortnight, with everything else clear."""
    report = evaluate_strategy_promotion("too_young",
                                         entry(13, 30, 1.60, 200.0),
                                         PORTFOLIO)
    assert report["passed"] is False
    assert report["failed"] == ["window"], report["reasons"]


def test_active_sessions_bind_when_recorded() -> None:
    """
    The parenthetical half of criterion 1. Recorded and short of ten, it fails
    the window; recorded and clear, it passes — and an entry that does not
    record it at all is NOT failed on it, which is the documented exception and
    the reason a ledger written before the field existed still promotes.
    """
    sparse = evaluate_strategy_promotion(
        "sparse", entry(20, 20, 1.30, 200.0, active_sessions=6), PORTFOLIO)
    assert sparse["passed"] is False
    assert sparse["failed"] == ["window"]

    dense = evaluate_strategy_promotion(
        "dense", entry(20, 20, 1.30, 200.0, active_sessions=14), PORTFOLIO)
    assert dense["passed"] is True, dense["reasons"]

    silent = evaluate_strategy_promotion("silent", entry(20, 20, 1.30, 200.0),
                                         PORTFOLIO)
    assert silent["passed"] is True
    assert any("sessions NOT RECORDED" in r for r in silent["reasons"])


def test_a_drawdown_recorded_negative_is_compared_on_magnitude() -> None:
    """A ledger recording -1,250 means the same excursion as 1,250. Comparing
    raw would let every negative figure clear every limit — the same rule the
    acceptance gates apply to a signed max drawdown."""
    report = evaluate_strategy_promotion("signed",
                                         entry(16, 25, 1.20, -1250.0),
                                         PORTFOLIO)
    assert report["passed"] is False
    assert report["failed"] == ["drawdown"]


# -- the three fixtures the objective names, at its own figures -------------
# `test_failing_window` breaks the horizon by ONE day and this breaks it by
# nine; both are the same verdict and they are not the same evidence. A
# strategy nine days short is one nobody should be looking at yet, and a table
# that showed it as PROMOTE would be read as "ready" by whoever is on the
# other end of `--auto-promote`.
def test_a_strategy_five_days_into_incubation_fails_the_horizon() -> None:
    """Day 5 of 14, with the other three criteria comfortably clear."""
    report = evaluate_strategy_promotion("day_five", entry(5, 30, 1.60, 200.0),
                                         PORTFOLIO)
    assert report["passed"] is False
    assert report["failed"] == ["window"]
    window = next(c for c in report["criteria"] if c["id"] == "window")
    assert "5 calendar days" in window["detail"]
    assert f"{MIN_DAYS_ACTIVE}" in window["detail"]


def test_a_twelve_hundred_dollar_drawdown_fails_the_risk_gate() -> None:
    """
    $1,200 against the $1,000 the account allows — a PROFITABLE strategy,
    refused.

    The bar is the account's, not the strategy's: a prop evaluation is lost on
    the drawdown and not on the profit factor, so an edge that ran $1,200
    underwater to earn it is not one this account can carry.
    """
    report = evaluate_strategy_promotion("too_deep_1200",
                                         entry(20, 30, 1.55, 1200.0),
                                         PORTFOLIO)
    assert report["passed"] is False
    assert report["failed"] == ["drawdown"]
    assert ALLOWABLE_DD == 1000.0, ALLOWABLE_DD
    reason = next(r for r in report["reasons"] if r.startswith("FAIL drawdown"))
    assert "1,200" in reason and "1,000" in reason


def test_a_missing_metric_fails_rather_than_defaulting_to_zero() -> None:
    """
    An entry with no trade count is an unfilled ledger, not a strategy that
    placed no trades. Zero would clear nothing either — but it would report
    "0 trades" as a measurement, and the difference is which of the two a human
    goes and fixes.
    """
    incomplete = entry(20, 20, 1.30, 200.0)
    del incomplete["trade_count"]
    report = evaluate_strategy_promotion("incomplete", incomplete, PORTFOLIO)
    assert report["passed"] is False
    assert report["failed"] == ["sample"]
    assert report["metrics"]["trade_count"] is None
    assert any("NOT RECORDED" in r for r in report["reasons"])


def test_an_already_graduated_entry_is_not_a_candidate() -> None:
    """Every metric clears and the status does not. Re-promoting a graduated
    strategy would append it to a prop list it is already on."""
    graduated = entry(30, 40, 1.50, 100.0)
    graduated["status"] = STATUS_GRADUATED
    report = evaluate_strategy_promotion("done", graduated, PORTFOLIO)
    assert report["passed"] is False
    assert "status" in report["failed"]


# --------------------------------------------------------------------------
# metrics derived from the trade list
# --------------------------------------------------------------------------

def _trades(pnls: list[float], start_day: int = 1) -> list[dict]:
    """One closed trade per weekday-ish calendar day, at 15:00 UTC — inside the
    session it is stamped in, so the session attribution is not what is under
    test here."""
    out = []
    for i, pnl in enumerate(pnls):
        day = start_day + i
        out.append({"closed_at": f"2026-08-{day:02d}T15:00:00+00:00",
                    "pnl": pnl, "status": "CLOSED"})
    return out


def test_metrics_are_derived_from_closed_trades() -> None:
    """
    Gross profit / gross loss, and the deepest peak-to-trough of cumulative
    realised P&L. The fixture wins 100 six times and loses 100 twice in a row,
    so the drawdown is 200 and the factor is 3.00 — both hand-checkable, which
    is the point of a fixture this small.
    """
    pnls = [100, 100, 100, -100, -100, 100, 100, 100]
    metrics = extract_metrics({"trades": _trades(pnls)})
    assert metrics["source"] == "derived_from_trades"
    assert metrics["trade_count"] == 8
    assert metrics["realized_pf"] == pytest.approx(600 / 200)
    assert metrics["max_drawdown"] == pytest.approx(200.0)
    assert metrics["net_pnl"] == pytest.approx(400.0)
    # Eight trades on eight consecutive days is an eight-day window, counted
    # inclusively at both ends.
    assert metrics["days_active"] == pytest.approx(8.0)
    assert metrics["active_sessions"] == pytest.approx(8.0)


def test_open_trades_do_not_count_toward_the_sample() -> None:
    """An open position has no realised P&L. Counting it would let a strategy
    reach the 14-trade bar on the trades whose outcome is still unknown."""
    trades = _trades([100.0] * 3)
    trades.append({"closed_at": "2026-08-09T15:00:00+00:00", "pnl": 0.0,
                   "status": "OPEN"})
    assert extract_metrics({"trades": trades})["trade_count"] == 3


def test_an_undefined_profit_factor_is_not_a_999_sentinel() -> None:
    """
    No losing trade means the factor is undefined, not enormous. The profiler
    writes 999 in that case and the Discord card has to render it as `--`; a
    promotion must not inherit a number that means "undefined" and sorts like
    the best result on the board. Expectancy then falls back to the sign of the
    net P&L, which is what "any net-positive expectancy" means with nothing to
    divide by.
    """
    metrics = extract_metrics({"trades": _trades([50.0] * 16)})
    assert metrics["realized_pf"] is None
    assert metrics["pf_undefined"] is True

    report = evaluate_strategy_promotion(
        "unbeaten",
        {"status": STATUS_INCUBATING, "trades": _trades([50.0] * 16)},
        PORTFOLIO)
    expectancy = next(c for c in report["criteria"] if c["id"] == "expectancy")
    assert expectancy["status"] == "PASS"
    assert "UNDEFINED" in expectancy["detail"]
    assert report["passed"] is True, report["reasons"]


def test_a_declared_figure_stands_in_for_one_the_trades_cannot_give() -> None:
    """
    Trades with no timestamps cannot date a window. The entry's own
    `days_active` is then the only evidence there is, and it must be USED —
    a merge that let the derived None shadow it would report a 16-day window
    as NOT RECORDED beside a trade list that simply carried no clocks.
    """
    entry_ = {
        "status": STATUS_INCUBATING,
        "days_active": 16,
        "trades": [{"pnl": p, "status": "CLOSED"} for p in [100.0] * 15
                   + [-50.0]],
    }
    metrics = extract_metrics(entry_)
    assert metrics["source"] == "derived_from_trades"
    assert metrics["trade_count"] == 16          # from the trades
    assert metrics["days_active"] == 16          # from the entry
    assert metrics["active_sessions"] is None    # nothing can say
    assert evaluate_strategy_promotion("clockless", entry_,
                                       PORTFOLIO)["passed"] is True


def test_metrics_are_read_from_a_nested_block_too() -> None:
    """A hand-written row puts the figures at the top level and a
    tracker-written one nests them under `metrics`. Refusing either spelling
    would be a schema argument fought at the cost of a promotion."""
    nested = {"status": STATUS_INCUBATING,
              "metrics": {"days_active": 15, "trade_count": 18,
                          "realized_pf": 1.15, "max_drawdown": 450.0}}
    assert evaluate_strategy_promotion("nested", nested,
                                       PORTFOLIO)["passed"] is True


def test_a_null_top_level_metric_does_not_hide_the_nested_one() -> None:
    """A half-migrated row carries `realized_pf: null` beside a `metrics` block
    holding the real figure. Reading the null as the answer would report a
    recorded metric as missing and hold a qualifying strategy."""
    half = {"status": STATUS_INCUBATING, "days_active": 15, "trade_count": 18,
            "realized_pf": None, "max_drawdown": 450.0,
            "metrics": {"realized_pf": 1.15}}
    report = evaluate_strategy_promotion("half_migrated", half, PORTFOLIO)
    assert report["metrics"]["realized_pf"] == pytest.approx(1.15)
    assert report["passed"] is True, report["reasons"]


def test_a_summary_contradicting_its_own_trades_fails() -> None:
    """
    The two readings of a disagreement are "the summary is stale" and "the
    trade list is incomplete". Both are reasons not to move a live account, so
    neither is chosen: the evaluation fails and names the discrepancy.
    """
    contradictory = {
        "status": STATUS_INCUBATING,
        "trades": _trades([100.0] * 16),      # 16 closed trades
        "trade_count": 40,                    # ...and a summary claiming 40
        "days_active": 16,
    }
    report = evaluate_strategy_promotion("stale", contradictory, PORTFOLIO)
    assert report["passed"] is False
    assert "ledger" in report["failed"]
    assert any("trade_count" in r and "FAIL ledger" in r
               for r in report["reasons"])


# --------------------------------------------------------------------------
# the promotion itself
# --------------------------------------------------------------------------

@pytest.fixture()
def workspace(tmp_path: Path):
    """
    Copies of the real config and a ledger holding one qualifying strategy on
    `Incubator-Odd`. Copies, not the originals: a test that promoted a strategy
    in the live routing table would make a change indistinguishable from one a
    human made on purpose.
    """
    config_path = tmp_path / "portfolios.json"
    ledger_path = tmp_path / "incubator_ledger.json"
    shutil.copy(CONFIG_PATH, config_path)

    config = json.loads(config_path.read_text())
    config["portfolios"]["Incubator-Odd"]["active_strategies"] = ["alpha_x"]
    config_path.write_text(json.dumps(config, indent=2))

    ledger_path.write_text(json.dumps({
        "version": "1.0.0",
        "strategies": {"alpha_x": entry(15, 18, 1.15, 450.0)},
    }, indent=2))

    config_loader.clear_cache()
    yield config_path, ledger_path
    config_loader.clear_cache()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_promotion_updates_both_files(workspace) -> None:
    """`Incubator-Odd` -> `Prop-Odd` in the routing table, and INCUBATING ->
    GRADUATED_PROP in the ledger. The removal from the source list is as much
    the point as the append to the target: leaving it on both would be a second
    live assignment of the same strategy against the same signal."""
    config_path, ledger_path = workspace
    assert promote_strategy("alpha_x", "Incubator-Odd",
                            config_path=str(config_path),
                            ledger_path=str(ledger_path)) is True

    portfolios = _read(config_path)["portfolios"]
    assert portfolios["Incubator-Odd"]["active_strategies"] == []
    assert portfolios["Prop-Odd"]["active_strategies"] == ["alpha_x"]
    # The other two accounts are untouched: a promotion moves one strategy.
    assert portfolios["Incubator-Even"]["active_strategies"] == []
    assert portfolios["Prop-Even"]["active_strategies"] == []

    record = _read(ledger_path)["strategies"]["alpha_x"]
    assert record["status"] == STATUS_GRADUATED
    assert record["target_portfolio"] == "Prop-Odd"
    assert record["source_portfolio"] == "Incubator-Odd"
    assert record["graduated_at"].startswith("20")
    # The forward evidence the decision rested on survives the stamp.
    assert record["trade_count"] == 18


def test_the_written_config_still_loads(workspace) -> None:
    """A routing table the loader refuses is worse than an unpromoted
    strategy: every subsequent run, including the live router, reads the same
    file."""
    config_path, ledger_path = workspace
    promote_strategy("alpha_x", "Incubator-Odd", config_path=str(config_path),
                     ledger_path=str(ledger_path))
    reloaded = config_loader.load_portfolio_config(str(config_path),
                                                   use_cache=False)
    assert config_loader.get_portfolio_for_strategy(
        "alpha_x", is_incubating=False, config=reloaded) == "Prop-Odd"


def test_promotion_is_idempotent(workspace) -> None:
    """A second pass over the same ledger finds the strategy graduated and on
    the target already. That is the one no-op that is not an error, and it
    returns False rather than raising — an evening's cron run must not exit
    non-zero because last night's promotion held."""
    config_path, ledger_path = workspace
    assert promote_strategy("alpha_x", "Incubator-Odd",
                            config_path=str(config_path),
                            ledger_path=str(ledger_path)) is True
    assert promote_strategy("alpha_x", "Incubator-Odd",
                            config_path=str(config_path),
                            ledger_path=str(ledger_path)) is False
    assert _read(config_path)["portfolios"]["Prop-Odd"][
        "active_strategies"] == ["alpha_x"]


def test_promoting_a_strategy_the_source_does_not_hold_raises(workspace) -> None:
    """Appending it to the prop list without removing it from anywhere is not a
    move. The config is left exactly as it was."""
    config_path, ledger_path = workspace
    before = config_path.read_text()
    ledger = _read(ledger_path)
    ledger["strategies"]["ghost"] = entry(20, 20, 1.30, 100.0)
    ledger_path.write_text(json.dumps(ledger, indent=2))

    with pytest.raises(PromotionError, match="active_strategies"):
        promote_strategy("ghost", "Incubator-Odd", config_path=str(config_path),
                         ledger_path=str(ledger_path))
    assert config_path.read_text() == before


def test_promoting_a_strategy_with_no_ledger_entry_raises(workspace) -> None:
    """The stamp goes onto the record of the forward trades the decision rests
    on. With no record there is nothing to stamp, and the config must not move
    on its own."""
    config_path, ledger_path = workspace
    config = _read(config_path)
    config["portfolios"]["Incubator-Odd"]["active_strategies"].append("orphan")
    config_path.write_text(json.dumps(config, indent=2))
    before = config_path.read_text()

    with pytest.raises(PromotionError, match="no entry"):
        promote_strategy("orphan", "Incubator-Odd",
                         config_path=str(config_path),
                         ledger_path=str(ledger_path))
    assert config_path.read_text() == before


def test_the_route_is_a_table_not_a_name() -> None:
    """`Prop-` + suffix would promote `Incubator-Test` onto a `Prop-Test`
    account that does not exist, and the failure would surface as a KeyError
    several layers from the typo."""
    assert target_portfolio_for("Incubator-Odd") == "Prop-Odd"
    assert target_portfolio_for("Incubator-Even") == "Prop-Even"
    for bad in ("Prop-Odd", "Incubator-Test", "incubator-odd", ""):
        with pytest.raises(PromotionError):
            target_portfolio_for(bad)


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------

def _tracker():
    """
    The CLI's own module, imported from the script path.

    `scripts/` is not a package, so there is no `import scripts.
    incubator_tracker`. The alternative — asserting on the table through
    `subprocess` stdout — would test the same strings through a pipe and could
    not reach `build_embed` at all, and the card is the half of Module D
    nobody sees until it is posted.
    """
    import importlib.util as _util
    path = REPO_ROOT / "scripts" / "incubator_tracker.py"
    spec = _util.spec_from_file_location("incubator_tracker_under_test", path)
    module = _util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_shows_progress_against_each_bar() -> None:
    """
    `8/14`, not `8`. The bare number is what a strategy HAS; the pair is what
    it has against what it needs, and the second is the question anybody
    reading this table is asking.

    The threshold is transcribed from the report the daemon produced, so a
    table cell and the criterion that promotes cannot show different bars.
    """
    tracker = _tracker()
    config = config_loader.load_portfolio_config()
    ledger = {"strategies": {"partway": entry(8, 12, 1.45, 350.0)}}
    config = json.loads(json.dumps(config))
    config["portfolios"]["Incubator-Odd"]["active_strategies"] = ["partway"]

    rows = tracker.evaluate_all(ledger, config)
    table = tracker.format_table(rows)

    assert f"8/{MIN_DAYS_ACTIVE}" in table
    assert f"12/{MIN_TRADE_COUNT}" in table
    assert "1.45" in table
    assert f"$350 / ${ALLOWABLE_DD:,.0f}" in table
    assert "HOLD" in table


def test_the_card_carries_one_line_per_incubating_strategy() -> None:
    """
    The four criteria in the compact form, because a Discord embed clips a
    wide code block on a phone and the table is the half that gets clipped.
    Every figure is transcribed from the same report the table renders.
    """
    tracker = _tracker()
    config = json.loads(json.dumps(config_loader.load_portfolio_config()))
    config["portfolios"]["Incubator-Odd"]["active_strategies"] = ["partway"]
    ledger = {"strategies": {"partway": entry(8, 12, 1.45, 350.0)}}

    rows = tracker.evaluate_all(ledger, config)
    embed = tracker.build_embed(rows, acted=False)
    field = next(f for f in embed["fields"]
                 if f["name"].startswith("Incubating"))

    assert field["value"].startswith("`partway`")
    assert f"Day 8/{MIN_DAYS_ACTIVE}" in field["value"]
    assert f"12/{MIN_TRADE_COUNT} trades" in field["value"]
    assert "PF 1.45" in field["value"]
    assert "DD $350" in field["value"]


def test_the_card_announces_a_promotion_and_says_so_when_it_did_not_act(
) -> None:
    """An empty promotion section read the morning after has two readings —
    "nothing qualified" and "nobody passed --auto-promote" — and only one of
    them is a reason to go and look."""
    tracker = _tracker()
    quiet = tracker.build_embed([{"strategy_id": "x", "account": "Incubator-Odd",
                                  "status": "HOLD", "note": "", "report": None}],
                                acted=False)
    assert any("--auto-promote" in f["value"] for f in quiet["fields"])

    promoted = tracker.build_embed(
        [{"strategy_id": "x", "account": "Prop-Odd", "status": "PROMOTED",
          "note": "Incubator-Odd -> Prop-Odd", "promoted_to": "Prop-Odd",
          "report": None}], acted=True)
    field = next(f for f in promoted["fields"] if f["name"] == "Promoted to prop")
    assert "`x`" in field["value"] and "Prop-Odd" in field["value"]


def test_the_table_groups_the_two_incubator_accounts() -> None:
    """Sim101's board and Sim102's are separate risk envelopes and are read
    separately; ledger order interleaves them and makes an operator scan the
    account column to answer "how is Sim101 doing"."""
    tracker = _tracker()
    rows = [{"strategy_id": "b_even", "account": "Incubator-Even",
             "status": "HOLD", "note": "", "report": None},
            {"strategy_id": "a_odd", "account": "Incubator-Odd",
             "status": "HOLD", "note": "", "report": None},
            {"strategy_id": "a_even", "account": "Incubator-Even",
             "status": "HOLD", "note": "", "report": None}]

    assert [r["strategy_id"] for r in tracker.group_by_account(rows)] == [
        "a_even", "b_even", "a_odd"]
    # and the audit itself is untouched: `evaluate_all` returns ledger order,
    # because an audit that reorders its input has to be diffed rather than read
    assert [r["strategy_id"] for r in rows] == ["b_even", "a_odd", "a_even"]


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """--no-discord on every invocation: a test suite must not post to a real
    channel, and `$BT_DISCORD_WEBHOOK` is populated from the repository's own
    `.env` on any box where the pipeline runs."""
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "incubator_tracker.py"),
         "--no-discord", *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120)


def test_cli_dry_run_writes_nothing(workspace) -> None:
    config_path, ledger_path = workspace
    before = (config_path.read_text(), ledger_path.read_text())
    proc = _run_cli("--config", str(config_path), "--ledger", str(ledger_path),
                    "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "alpha_x" in proc.stdout
    assert "PROMOTE" in proc.stdout
    assert (config_path.read_text(), ledger_path.read_text()) == before


def test_cli_default_is_evaluation_only(workspace) -> None:
    """No flags means no writes. The default of a script that edits a live
    routing table must be to print."""
    config_path, ledger_path = workspace
    before = config_path.read_text()
    proc = _run_cli("--config", str(config_path), "--ledger", str(ledger_path))
    assert proc.returncode == 0, proc.stderr
    assert "--auto-promote" in proc.stdout
    assert config_path.read_text() == before


def test_cli_dry_run_wins_over_auto_promote(workspace) -> None:
    """An operator who typed both meant to be careful. Guessing the other way
    writes an account move nobody sanctioned."""
    config_path, ledger_path = workspace
    before = config_path.read_text()
    proc = _run_cli("--config", str(config_path), "--ledger", str(ledger_path),
                    "--dry-run", "--auto-promote")
    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    assert config_path.read_text() == before


def test_cli_auto_promote_moves_the_strategy(workspace) -> None:
    config_path, ledger_path = workspace
    proc = _run_cli("--config", str(config_path), "--ledger", str(ledger_path),
                    "--auto-promote")
    assert proc.returncode == 0, proc.stderr
    assert "PROMOTED" in proc.stdout
    assert _read(config_path)["portfolios"]["Prop-Odd"][
        "active_strategies"] == ["alpha_x"]
    assert _read(ledger_path)["strategies"]["alpha_x"]["status"] == \
        STATUS_GRADUATED


def test_a_strategy_the_routing_table_does_not_name_is_unrouted(
        workspace) -> None:
    """
    A ledger entry naming its own portfolio is NOT a fallback. Taking it would
    produce a PROMOTE verdict for a strategy no incubator portfolio holds,
    which `promote_strategy` then refuses — so the row would clear every
    criterion on the table and fail on the way out. It is UNROUTED, the note
    says which portfolio the ledger claims, and nothing moves.
    """
    config_path, ledger_path = workspace
    ledger = _read(ledger_path)
    ledger["strategies"]["unrouted"] = entry(20, 20, 1.30, 100.0)
    ledger["strategies"]["unrouted"]["portfolio"] = "Incubator-Even"
    ledger_path.write_text(json.dumps(ledger, indent=2))

    proc = _run_cli("--config", str(config_path), "--ledger", str(ledger_path),
                    "--auto-promote")
    assert proc.returncode == 0, proc.stderr
    assert "UNROUTED" in proc.stdout
    assert "Incubator-Even" in proc.stdout
    assert _read(config_path)["portfolios"]["Prop-Even"][
        "active_strategies"] == []
    assert _read(ledger_path)["strategies"]["unrouted"]["status"] == \
        STATUS_INCUBATING
    # The qualifying strategy on the same run still promoted: one unroutable
    # row must not stall the evening's audit.
    assert _read(config_path)["portfolios"]["Prop-Odd"][
        "active_strategies"] == ["alpha_x"]


def test_a_disagreement_between_the_two_files_promotes_nothing(
        workspace) -> None:
    """The config has alpha_x on Incubator-Odd and the ledger says Even. The
    readings are "the ledger is stale" and "the table was hand-edited", and
    under the first a promotion moves a strategy off an account it was never
    on."""
    config_path, ledger_path = workspace
    ledger = _read(ledger_path)
    ledger["strategies"]["alpha_x"]["portfolio"] = "Incubator-Even"
    ledger_path.write_text(json.dumps(ledger, indent=2))
    before = config_path.read_text()

    proc = _run_cli("--config", str(config_path), "--ledger", str(ledger_path),
                    "--auto-promote")
    assert proc.returncode == 0, proc.stderr
    assert "UNROUTED" in proc.stdout
    assert config_path.read_text() == before


def test_the_shipped_ledger_and_config_evaluate_cleanly() -> None:
    """The real files, read-only. An empty ledger is a valid state and must
    print a table rather than raise — this is the state the repository ships
    in, so it is the first thing an operator will run."""
    proc = _run_cli()
    assert proc.returncode == 0, proc.stderr
    assert "INCUBATOR PERFORMANCE TRACKER" in proc.stdout
    assert _read(LEDGER_PATH)["strategies"] == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
