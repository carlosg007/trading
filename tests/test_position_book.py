"""
tests.test_position_book - the entry gate, and the size cap beside it.

WHAT THIS SUITE IS DEFENDING
============================
A live loop re-evaluates its strategies every cycle. A strategy whose signal
stays true therefore does not enter once - it enters on every pass, and on a
62-second interval that is a new position a minute on an account this process
believed held one. `aggregate_signals` cannot catch it: it nets the signals
arriving in ONE cycle against each other and knows nothing about the position
the previous cycle opened. `can_execute` is the check that does, and every case
below is a way that check could quietly stop working.

Assert-based and pytest-shaped: every case fails through `assert`, so
`pytest tests/` and running this file directly report the same thing. There is
no `check()` helper here on purpose - see `tests/conftest.py` for what that
marker does to a suite's results.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest                                                     # noqa: E402

from portfolio.portfolio_manager import (                         # noqa: E402
    FLAT,
    LONG,
    SHORT,
    PositionBook as NettedPositionBook,
)
from realtime.position_book import (                              # noqa: E402
    HOLD_TEMPLATE,
    MAX_QTY,
    PositionBook,
    quantity_refusal,
)

PORTFOLIO = "Incubator-Odd"
OTHER = "Incubator-Even"
SYMBOL = "NQ"
STRATEGY_A = "keltner_trend_drift_20260901_NQ_15m_VA"
STRATEGY_B = "ema_crossover_20260821_NQ_1h_VA"


# --------------------------------------------------------------------------
# 1. the stacking loop, which is what this exists to stop
# --------------------------------------------------------------------------
def test_consecutive_buys_evaluate_true_then_false():
    """
    THE 62-SECOND LOOP. The first BUY opens; every BUY after it, on the same
    signal that has not gone away, is a second position on top of the first.
    """
    book = PositionBook()
    assert book.can_execute(PORTFOLIO, SYMBOL, "BUY") is True

    book.record_fill(PORTFOLIO, SYMBOL, LONG, 1, [STRATEGY_A])

    assert book.can_execute(PORTFOLIO, SYMBOL, "BUY") is False
    # ...and it stays False. A gate that reopened after one refusal would turn
    # a position per cycle into a position every other cycle.
    assert book.can_execute(PORTFOLIO, SYMBOL, "BUY") is False


def test_a_second_strategy_in_the_same_portfolio_is_gated_too():
    """
    THE KEY IS `(portfolio_id, symbol)` AND NOT THE STRATEGY. Sixteen
    strategies trade NQ inside Incubator-Odd and they net to ONE position, so
    strategy B's BUY on a pair strategy A already opened is the same stack -
    it just arrives under a different name. A strategy-keyed book would have
    permitted it and called the result two positions.
    """
    book = PositionBook()
    book.record_fill(PORTFOLIO, SYMBOL, LONG, 1, [STRATEGY_A])

    assert book.can_execute(PORTFOLIO, SYMBOL, "BUY") is False
    # The book does not even take a strategy name: there is one position on
    # that pair and it belongs to every strategy holding it.
    assert book.get(PORTFOLIO, SYMBOL)["strategies"] == [STRATEGY_A]


def test_another_portfolio_on_the_same_symbol_is_not_gated():
    """
    The baskets route to DIFFERENT ACCOUNTS. Incubator-Odd being long NQ says
    nothing about what Incubator-Even may do on NQ, and a gate that keyed on
    the symbol alone would stand down a whole account for another's position.
    """
    book = PositionBook()
    book.record_fill(PORTFOLIO, SYMBOL, LONG, 1, [STRATEGY_A])

    assert book.can_execute(OTHER, SYMBOL, "BUY") is True


def test_a_different_symbol_in_the_same_portfolio_is_not_gated():
    book = PositionBook()
    book.record_fill(PORTFOLIO, SYMBOL, LONG, 1, [STRATEGY_A])
    assert book.can_execute(PORTFOLIO, "GC", "BUY") is True


# --------------------------------------------------------------------------
# 2. the full round trip
# --------------------------------------------------------------------------
def test_buy_then_close_walks_flat_to_long_to_flat():
    """FLAT -> LONG -> FLAT, with the gate agreeing at every step."""
    book = PositionBook()
    assert book.state(PORTFOLIO, SYMBOL) == FLAT
    assert book.can_execute(PORTFOLIO, SYMBOL, "BUY") is True
    # Nothing is open, so there is nothing to close.
    assert book.can_execute(PORTFOLIO, SYMBOL, "CLOSE") is False

    book.record_fill(PORTFOLIO, SYMBOL, LONG, 1, [STRATEGY_A])
    assert book.state(PORTFOLIO, SYMBOL) == LONG
    assert book.can_execute(PORTFOLIO, SYMBOL, "CLOSE") is True

    book.record_flat(PORTFOLIO, SYMBOL)
    assert book.state(PORTFOLIO, SYMBOL) == FLAT
    # ...and the pair is tradable again, which is the half a gate that only
    # ever refused would also pass.
    assert book.can_execute(PORTFOLIO, SYMBOL, "BUY") is True


def test_a_short_round_trip_walks_the_other_way():
    book = PositionBook()
    assert book.can_execute(PORTFOLIO, SYMBOL, "SELL") is True

    book.record_fill(PORTFOLIO, SYMBOL, SHORT, 1, [STRATEGY_B])
    assert book.state(PORTFOLIO, SYMBOL) == SHORT
    assert book.can_execute(PORTFOLIO, SYMBOL, "SELL") is False
    assert book.can_execute(PORTFOLIO, SYMBOL, "FLATTEN") is True

    book.record_flat(PORTFOLIO, SYMBOL)
    assert book.can_execute(PORTFOLIO, SYMBOL, "SELL") is True


# --------------------------------------------------------------------------
# 3. the rest of the table
# --------------------------------------------------------------------------
@pytest.mark.parametrize("held, action, allowed", [
    (FLAT,  "BUY",     True),    # open
    (FLAT,  "SELL",    True),    # open short
    (FLAT,  "CLOSE",   False),   # nothing recorded to close
    (FLAT,  "FLATTEN", False),
    (LONG,  "BUY",     False),   # THE STACK
    (LONG,  "SELL",    True),    # reverse
    (LONG,  "CLOSE",   True),
    (LONG,  "FLATTEN", True),
    (SHORT, "BUY",     True),    # reverse
    (SHORT, "SELL",    False),   # THE STACK
    (SHORT, "CLOSE",   True),
    (SHORT, "FLATTEN", True),
])
def test_the_whole_state_table(held, action, allowed):
    book = PositionBook()
    if held != FLAT:
        book.record_fill(PORTFOLIO, SYMBOL, held, 1, [STRATEGY_A])
    assert book.can_execute(PORTFOLIO, SYMBOL, action) is allowed


@pytest.mark.parametrize("action", ["buy", " Buy ", "sell", "close"])
def test_the_action_is_read_case_insensitively(action):
    """
    `build_order_payloads` writes `BUY`, an operator types `buy`, and the
    formatter upper-cases on the way out. A gate that only recognised one
    spelling would silently permit everything written in the other.
    """
    book = PositionBook()
    book.record_fill(PORTFOLIO, SYMBOL, LONG, 1, [STRATEGY_A])
    expected = action.strip().upper() in ("SELL", "CLOSE")
    assert book.can_execute(PORTFOLIO, SYMBOL, action) is expected


@pytest.mark.parametrize("action", ["", "HOLD", "REVERSE", None, "BUY_TO_OPEN"])
def test_an_unrecognised_action_is_refused(action):
    """
    A gate that let a word it did not understand through is a gate that stops
    working the first time somebody invents a side.
    """
    assert PositionBook().can_execute(PORTFOLIO, SYMBOL, action) is False


# --------------------------------------------------------------------------
# 4. one book, not two
# --------------------------------------------------------------------------
def test_the_gate_reads_the_same_state_the_exit_path_writes():
    """
    THE REASON THIS IS A SUBCLASS. `record_fill`, `record_flat` and
    `plan_exits` are the exit path's, and the gate has to answer from the
    object they update. A second dictionary of positions would be written by
    the entry path and not by the exit path, and would be wrong from the first
    flatten onward - refusing entries on a position already closed, with every
    log line reading correctly.
    """
    book = PositionBook()
    assert isinstance(book, NettedPositionBook)

    book.record_fill(PORTFOLIO, SYMBOL, LONG, 1, [STRATEGY_A])
    intents = book.plan_exits([{"portfolio_id": PORTFOLIO, "symbol": SYMBOL,
                                "strategy_id": STRATEGY_A}])
    assert intents[0]["emit"] is True, "the inherited exit path still works"

    book.record_flat(PORTFOLIO, SYMBOL)
    assert book.can_execute(PORTFOLIO, SYMBOL, "BUY") is True


def test_the_symbol_is_matched_the_way_the_base_class_keys_it():
    """
    `_key` upper-cases the symbol. A gate that compared raw strings would read
    `nq` as a different pair from the `NQ` the fill was recorded under and
    permit a stack on it.
    """
    book = PositionBook()
    book.record_fill(PORTFOLIO, "nq", LONG, 1, [STRATEGY_A])
    assert book.can_execute(PORTFOLIO, "NQ", "BUY") is False
    assert book.can_execute(PORTFOLIO, "nq", "BUY") is False


# --------------------------------------------------------------------------
# 5. the message
# --------------------------------------------------------------------------
def test_the_hold_line_names_the_portfolio_and_the_symbol():
    """
    Not the strategy: the position is netted, so what is already active belongs
    to every strategy on that pair, and naming one would report the block
    against a strategy that may not have signalled this cycle.
    """
    book = PositionBook()
    assert book.hold_reason(PORTFOLIO, SYMBOL, "buy") == (
        "HOLD Incubator-Odd/NQ BUY — position already active. "
        "Preventing stack.")
    assert HOLD_TEMPLATE.count("{") == 3


# --------------------------------------------------------------------------
# 6. the size cap
# --------------------------------------------------------------------------
def test_the_cap_is_one_contract():
    assert MAX_QTY == 1


def test_an_oversized_order_is_rejected_and_never_clamped():
    """
    A clamp sends a DIFFERENT order from the one the sizer computed and reports
    success: a position sized against a 3-contract stop goes on at 1, and the
    risk model no longer describes the trade that is open. The rejection names
    what was asked for so the drop is visible rather than silent.
    """
    refusal = quantity_refusal(3)
    assert refusal == {"ok": False, "rule": "MAX_QTY",
                       "detail": "sizer asked 3, cap is 1"}


@pytest.mark.parametrize("qty", [0, 1, -4])
def test_a_size_at_or_under_the_cap_is_not_this_rule_s_business(qty):
    """
    `None` for "no objection". A zero or negative quantity IS wrong, and it is
    refused by the formatter and the firewall with a message about what is
    wrong with it - answering "cap is 1" to a quantity of 0 would name the
    wrong problem.
    """
    assert quantity_refusal(qty) is None


@pytest.mark.parametrize("qty", [None, "3", 2.0, True])
def test_a_non_integer_quantity_is_left_to_the_validators_that_own_it(qty):
    assert quantity_refusal(qty) is None


def test_the_cap_can_be_raised_for_a_caller_that_passes_one():
    assert quantity_refusal(3, 5) is None
    assert quantity_refusal(6, 5)["detail"] == "sizer asked 6, cap is 5"


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
