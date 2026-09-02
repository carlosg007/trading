"""
tests/test_regime_score_modes.py - the two designation ranking rules.

ASSERT-BASED, so `tests/conftest.py` collects it normally. No `def check(`
marker: that is what routes a suite to the subprocess runner.

WHAT THIS IS GUARDING
=====================
`vol_normalized` scoring exists because the default one has a measured
direction. The engine is fixed-size (`BacktestConfig.contracts = 1`), so
per-trade P&L moves with the size of the move; Q3's mean ATR is 0.28x Q1's and
Q3 holds 0.67x the bars, so at an EQUAL profit factor a Q3 quadrant scores
about 0.19x a Q1 one. That is why 20 of 20 instances of the three trend-drift
archetypes were designated into a high-volatility quadrant - including
`keltner_trend_drift_20260901`, a module that declares
`TARGET_QUADRANTS = ("Q3",)`.

The cases below pin three things, and the FIRST is the one that protects the
live registry: `alpha` is still the default and still ranks exactly as it did,
because every strategy in `config/portfolios.json` was designated under it and
a silently switched rule would leave the registry's quadrants and the rule that
produced them disagreeing with nothing raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.profiler import (                                    # noqa: E402
    SCORE_MODES,
    designate,
    quadrant_score,
    rank_quadrants,
    score_for,
    vol_normalized_score,
)

PER_TRADE = 100.0


def _breakdown(q3_pf: float = 1.60, q1_pf: float = 1.20) -> dict:
    """
    A strategy that is genuinely BEST in Q3 - highest profit factor of the four
    - with the measured structure around it: Q3 carries fewer trades (18% of
    bars against Q1's 30%) and each is 0.31x the size.
    """
    return {
        "High Volatility / Trending": dict(
            trade_count=300, profit_factor=q1_pf, win_rate=50.0,
            net_pnl=300 * PER_TRADE * (q1_pf - 1.0),
            avg_trade_abs_pnl=PER_TRADE),
        "High Volatility / Ranging": dict(
            trade_count=270, profit_factor=1.15, win_rate=50.0,
            net_pnl=270 * PER_TRADE * 0.88 * 0.15,
            avg_trade_abs_pnl=PER_TRADE * 0.88),
        "Low Volatility / Trending": dict(
            trade_count=180, profit_factor=q3_pf, win_rate=50.0,
            net_pnl=180 * PER_TRADE * 0.31 * (q3_pf - 1.0),
            avg_trade_abs_pnl=PER_TRADE * 0.31),
        "Low Volatility / Ranging": dict(
            trade_count=240, profit_factor=0.95, win_rate=50.0,
            net_pnl=-240 * PER_TRADE * 0.30 * 0.05,
            avg_trade_abs_pnl=PER_TRADE * 0.30),
    }


# --------------------------------------------------------------------------
# 1. The default is unchanged
# --------------------------------------------------------------------------
def test_alpha_is_the_default_and_is_unchanged():
    """
    THE CASE THAT PROTECTS THE LIVE REGISTRY. Every strategy in
    config/portfolios.json was designated under `alpha`; a switched default
    would leave the registry's quadrants and the rule that produced them
    disagreeing, invisibly.
    """
    bd = _breakdown()
    assert designate(bd, 990)["score_mode"] == "alpha"
    assert designate(bd, 990)["primary"]["quadrant"] == "Q1", (
        "the default rule stopped designating what it always designated")

    # And the score itself is still net P&L x capped PF, unmodified.
    stats = bd["High Volatility / Trending"]
    assert quadrant_score(stats) == pytest.approx(
        stats["net_pnl"] * stats["profit_factor"])


def test_the_default_ranking_ignores_the_new_divisor_entirely():
    """A breakdown with no `avg_trade_abs_pnl` - every profile written before
    this change - must designate exactly as it did."""
    bd = {k: {kk: vv for kk, vv in v.items() if kk != "avg_trade_abs_pnl"}
          for k, v in _breakdown().items()}
    assert designate(bd, 990)["primary"]["quadrant"] == "Q1"


# --------------------------------------------------------------------------
# 2. The new mode does what it claims
# --------------------------------------------------------------------------
def test_vol_normalized_designates_the_quadrant_with_the_real_edge():
    bd = _breakdown()
    d = designate(bd, 990, score_mode="vol_normalized")
    assert d["primary"]["quadrant"] == "Q3"
    assert d["score_mode"] == "vol_normalized"


def test_the_divisor_is_the_quadrants_own_average_trade():
    stats = dict(trade_count=100, profit_factor=1.50, net_pnl=1000.0,
                 avg_trade_abs_pnl=50.0)
    assert vol_normalized_score(stats) == pytest.approx(
        (1000.0 / 50.0) * 1.50)


def test_a_quadrant_with_no_scale_is_not_scored_rather_than_divided_by_zero():
    """A quadrant whose trades all closed exactly flat has no scale to divide
    by, and 0/0 is not a ranking."""
    for scale in (0.0, None, float("nan")):
        stats = dict(profit_factor=1.5, net_pnl=100.0, avg_trade_abs_pnl=scale)
        assert vol_normalized_score(stats) is None


def test_normalizing_by_theta_vol_would_have_changed_nothing():
    """
    The request suggested dividing by `theta_vol`. It is ONE scalar per
    (symbol, TIMEFRAME) - the boundary between high and low volatility, not a
    per-quadrant statistic - so it is the same divisor for all four quadrants
    and the ranking is identical. Pinned so the reasoning is not re-litigated.
    """
    bd = _breakdown()
    theta = 7.90                                   # NQ at 15m, any constant
    scaled = {k: {**v, "net_pnl": v["net_pnl"] / theta} for k, v in bd.items()}
    assert ([r["regime"] for r in designate(scaled, 990)["scores"]]
            == [r["regime"] for r in designate(bd, 990)["scores"]])


# --------------------------------------------------------------------------
# 3. Both scores travel, whichever one sorts
# --------------------------------------------------------------------------
def test_every_row_carries_both_scores_in_either_mode():
    """A reader has to be able to see what the other rule would have done
    without re-running anything."""
    for mode in SCORE_MODES:
        for row in designate(_breakdown(), 990, score_mode=mode)["scores"]:
            assert "alpha_score" in row and "vol_normalized_score" in row
            assert row["score_mode"] == mode
            assert row["score"] == row[
                "alpha_score" if mode == "alpha" else "vol_normalized_score"]


def test_a_disagreement_is_reported_and_agreement_is_not():
    """
    `would_designate` is None when the two rules agree, so a disagreement is
    never buried in a field that is always populated.
    """
    disagreeing = designate(_breakdown(), 990)
    assert disagreeing["would_designate"] is not None
    assert disagreeing["would_designate"]["quadrant"] == "Q3"
    assert disagreeing["primary"]["quadrant"] == "Q1"

    # Make Q1 win on both by giving it the edge outright.
    agreeing = designate(_breakdown(q3_pf=1.02, q1_pf=2.00), 990)
    assert agreeing["primary"]["quadrant"] == "Q1"
    assert agreeing["would_designate"] is None, (
        "the rules agree; there is nothing to report")


def test_the_score_formula_string_names_the_rule_that_ran():
    assert "avg_trade_abs_pnl" not in designate(
        _breakdown(), 990)["score_formula"]
    assert "avg_trade_abs_pnl" in designate(
        _breakdown(), 990, score_mode="vol_normalized")["score_formula"]


# --------------------------------------------------------------------------
# 4. Neither mode relaxes a bar
# --------------------------------------------------------------------------
def test_the_designation_bars_bind_identically_in_both_modes():
    """
    The scoring rule decides the ORDER, never eligibility. A losing quadrant
    stays ineligible under both, or the new mode would be a lowered bar wearing
    a ranking's clothes.
    """
    bd = _breakdown()
    for mode in SCORE_MODES:
        rows = {r["regime"]: r
                for r in designate(bd, 990, score_mode=mode)["scores"]}
        assert rows["Low Volatility / Ranging"]["eligible"] is False, (
            f"a negative-expectancy quadrant became eligible under {mode}")
        assert rows["Low Volatility / Trending"]["eligible"] is True


def test_an_unknown_score_mode_raises_rather_than_falling_back():
    """A run that silently ranked on something other than what was asked for
    is a designation nobody can check."""
    with pytest.raises(ValueError, match="score_mode"):
        designate(_breakdown(), 990, score_mode="sharpe")
    with pytest.raises(ValueError, match="score_mode"):
        rank_quadrants(_breakdown(), 50, 1.00, "nonsense")
    with pytest.raises(ValueError, match="score_mode"):
        score_for({"profit_factor": 1.0, "net_pnl": 1.0}, "nonsense")


def test_stage1_threads_the_mode_through_to_the_handoff():
    """`best_quadrant` is Stage 1's entry point and must carry the choice, or
    the flag would parse and change no designation."""
    from backtest.baseline import best_quadrant

    profile = {"regime_breakdown": _breakdown(), "trades_profiled": 990}
    assert best_quadrant(profile)["quadrant"] == "Q1"
    assert best_quadrant(profile, score_mode="vol_normalized")["quadrant"] \
        == "Q3"
    assert best_quadrant(profile)["score_mode"] == "alpha"
    assert best_quadrant(profile)["would_designate"]["quadrant"] == "Q3"


# --------------------------------------------------------------------------
# 5. TARGET_QUADRANTS restricts the designation
#
# A module declares the environment its premise is about. Before this, Stage 1
# measured all four and designated whichever scored highest, so
# `keltner_trend_drift_20260901` - which declares `TARGET_QUADRANTS = ("Q3",)`
# - was certified into Q1 or Q2 on all eight of its instances, and 20 of 20
# across the three trend-drift archetypes. The declaration now decides which
# quadrant the strategy is JUDGED in. It never decides that it passed.
# --------------------------------------------------------------------------
def test_a_declared_quadrant_wins_over_a_higher_scoring_one():
    """(a) Q3 declared, Q1 has the higher alpha, Q3 clears the bars."""
    d = designate(_breakdown(), 990, target_quadrants=("Q3",))
    assert d["primary"]["quadrant"] == "Q3"
    assert d["designation_restricted"] is True
    assert d["declared_quadrants"] == ["Q3"]
    # And the unrestricted screen really would have said Q1 — otherwise this
    # case proves nothing.
    assert designate(_breakdown(), 990)["primary"]["quadrant"] == "Q1"


def test_a_declared_quadrant_that_fails_drops_the_pair(caplog=None):
    """
    (b) THE HALF THAT MATTERS. Q3 declared and failing must DROP the pair, not
    re-home it into Q1 because higher volatility swung larger dollars.
    """
    d = designate(_breakdown(q3_pf=0.90), 990, target_quadrants=("Q3",))
    assert d["primary"] is None, (
        "the pair was re-homed into a quadrant the module does not claim")
    assert "declared target Q3" in d["reason"]
    # The quadrant that DID qualify is named, so the drop is legible.
    assert d["unrestricted_primary"]["quadrant"] == "Q1"
    assert "NOT substituted" in d["reason"]


def test_an_undeclared_strategy_keeps_the_existing_behaviour():
    """(c) No declaration — the unrestricted best-of-four, exactly as before."""
    d = designate(_breakdown(), 990)
    assert d["primary"]["quadrant"] == "Q1"
    assert d["designation_restricted"] is False
    assert d["declared_quadrants"] == []
    assert d["unrestricted_primary"] is None


def test_the_declaration_restricts_candidates_and_relaxes_no_bar():
    """
    THE LINE THIS MUST NOT CROSS. Stage 1's sample floor is max(50, 10% of
    placed) - NOT Gate R's holdout floor of 30 - and a declaration must not
    lower it. Otherwise a module could certify on thinner evidence by writing
    a constant in itself.
    """
    thin = {
        "Low Volatility / Trending": dict(
            trade_count=35, profit_factor=2.00, win_rate=50.0,
            net_pnl=35 * 31.0, avg_trade_abs_pnl=31.0),
    }
    # 35 trades clears Gate R's 30 but not Stage 1's floor of 50.
    d = designate(thin, 350, target_quadrants=("Q3",))
    assert d["sample_floor"] == 50
    assert d["primary"] is None, (
        "the declaration lowered the sample floor to Gate R's")
    assert "below the sample floor" in d["reason"]


def test_both_declaration_spellings_designate_identically():
    for declared in (("Q3",), ("Low Volatility / Trending",)):
        d = designate(_breakdown(), 990, target_quadrants=declared)
        assert d["primary"]["quadrant"] == "Q3"
        assert d["declared_quadrants"] == ["Q3"]


def test_a_declaration_naming_several_quadrants_picks_the_best_among_them():
    """A module may declare more than one; the score still orders WITHIN the
    declared set rather than being ignored."""
    d = designate(_breakdown(), 990, target_quadrants=("Q2", "Q3"))
    assert d["primary"]["quadrant"] == "Q3", "Q3 outscores Q2 among the two"
    assert set(d["declared_quadrants"]) == {"Q2", "Q3"}


def test_the_declaration_composes_with_vol_normalized_scoring():
    """The two mechanisms are independent: one picks the candidate set, the
    other orders it."""
    d = designate(_breakdown(), 990, score_mode="vol_normalized",
                  target_quadrants=("Q3",))
    assert d["primary"]["quadrant"] == "Q3"
    assert d["score_mode"] == "vol_normalized"


def test_stage1_reads_the_declaration_off_the_module():
    """
    `load_strategy` has to surface `TARGET_QUADRANTS`, or the flag would exist
    and no module would ever reach it. Checked against the real module that
    started this: `keltner_trend_drift_20260901` declares Q3.
    """
    from agents.tier3_workers import load_strategy

    _fn, info = load_strategy(
        str(REPO_ROOT / "strategies" / "experimental"
            / "keltner_trend_drift_20260901.py"))
    assert info["target_quadrants"] == ("Q3",)


def test_stage1_best_quadrant_threads_the_declaration():
    from backtest.baseline import best_quadrant

    profile = {"regime_breakdown": _breakdown(), "trades_profiled": 990}
    assert best_quadrant(profile)["quadrant"] == "Q1"
    assert best_quadrant(profile, target_quadrants=("Q3",))["quadrant"] == "Q3"

    failing = {"regime_breakdown": _breakdown(q3_pf=0.90),
               "trades_profiled": 990}
    assert best_quadrant(failing, target_quadrants=("Q3",)) is None, (
        "Stage 1 re-homed a pair that missed its own declared target")


def test_stage1_screen_drops_a_pair_that_misses_its_declared_target():
    """`screen` is what decides PROMOTED vs DROPPED, so the drop has to survive
    the whole way out rather than only inside `designate`."""
    from backtest.baseline import screen

    profiles = {"A": {"regime_breakdown": _breakdown(q3_pf=0.90),
                      "trades_profiled": 990}}
    survived, reason, best = screen(profiles, target_quadrants=("Q3",))
    assert survived is False
    assert best is None
    assert "Q3" in reason

    ok = {"A": {"regime_breakdown": _breakdown(), "trades_profiled": 990}}
    survived, _reason, best = screen(ok, target_quadrants=("Q3",))
    assert survived is True
    assert best["quadrant"] == "Q3"


def test_a_misspelled_declaration_raises_rather_than_being_ignored():
    """
    Reduced to an empty set it would restore the unrestricted best-of-four and
    the module would be screened on dollar alpha exactly as if it had declared
    nothing — the failure this mechanism exists to remove, with every log line
    reading correctly.
    """
    for bad in (("Q5",), ("Low Vol Trend",), ("q3 ",)[:0] + ("Quadrant 3",)):
        with pytest.raises(ValueError, match="TARGET_QUADRANTS"):
            designate(_breakdown(), 990, target_quadrants=bad)
