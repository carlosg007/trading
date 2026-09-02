"""
tests/test_q3_audit.py - `scripts/q3_audit.py`.

ASSERT-BASED, so `tests/conftest.py` collects it normally. No `def check(`
marker in this file: that is what routes a suite to the subprocess runner, and
a suite recording results in a list instead of asserting would report green
while failing.

The arithmetic is tested hermetically. The lake-reading half is not mocked and
not exercised here - it reads the pinned regime caches on `/mnt/backtest`, and
a unit test that stood up a fake cache would be testing the fake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.q3_audit import (                                     # noqa: E402
    Q3,
    break_even_pf,
    build_parser,
)


def test_the_quadrant_under_audit_is_mdlibs_q3():
    """
    Pinned against the authority rather than restated. Q3 is Low-Vol/TRENDING;
    Q2 - the one it is most often confused with - is High-Vol/RANGING, and an
    audit that drifted onto the wrong digit would measure the wrong regime and
    report it under the right name.
    """
    from mdlib.regimes import QUADRANT_LABELS

    assert Q3 == 3
    assert QUADRANT_LABELS[Q3] == "Low Volatility / Trending"
    assert QUADRANT_LABELS[2] == "High Volatility / Ranging"


def test_the_break_even_solves_the_alpha_score_it_claims_to():
    """
    `designate` ranks on `net_pnl x profit_factor`. The returned factor must
    make Q3's score EQUAL the rival's, or the number printed on the card is a
    plausible-looking figure that settles nothing.
    """
    atr_ratio, bar_ratio, rival = 0.28, 0.67, 1.20
    pf = break_even_pf(atr_ratio, bar_ratio, rival)

    k = atr_ratio * bar_ratio
    q3_score = k * (pf - 1.0) * pf
    rival_score = (rival - 1.0) * rival
    assert q3_score == pytest.approx(rival_score, rel=1e-9)


def test_a_handicapped_quadrant_needs_a_higher_factor_than_its_rival():
    """The direction of the result, which is the whole claim: Q3 has to be
    BETTER than Q1 on profit factor merely to be nominated."""
    assert break_even_pf(0.28, 0.67, 1.20) > 1.20
    assert break_even_pf(0.18, 0.60, 1.20) > break_even_pf(0.33, 0.75, 1.20), (
        "a heavier handicap must demand a higher factor")


def test_no_handicap_means_no_premium():
    """With equal size and equal trade count the bar is the rival's own factor
    - the model has to degenerate correctly or its slope means nothing."""
    assert break_even_pf(1.0, 1.0, 1.20) == pytest.approx(1.20, rel=1e-9)


def test_a_quadrant_that_never_trades_can_never_be_nominated():
    assert break_even_pf(0.0, 0.5, 1.20) == float("inf")
    assert break_even_pf(0.3, 0.0, 1.20) == float("inf")


def test_the_parser_accepts_the_documented_flags():
    args = build_parser().parse_args(
        ["--symbols", "ES", "NQ", "--tf", "30m", "1h", "--json"])
    assert args.symbols == ["ES", "NQ"]
    assert args.timeframes == ["30m", "1h"]
    assert args.json is True
