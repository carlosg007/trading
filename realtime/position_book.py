"""
realtime.position_book - the ENTRY gate over the netted position book.

WHY THIS IS A SUBCLASS AND NOT A SECOND BOOK
============================================
`portfolio.portfolio_manager.PositionBook` already tracks what this process
opened, already keys it by `(portfolio_id, symbol)`, and is already what
`plan_exits`, `record_fill` and `record_flat` read and write. What it never had
is a gate on the way IN: `dispatch_order` consulted nothing, so a strategy that
keeps signalling BUY while its position is open sent a fresh entry on every
cycle, and the account accumulated a position per cycle that nothing here could
see as one position.

So this module adds the DECISION and inherits the STATE. A second dictionary of
positions - even one keyed identically - would be written by the entry path and
not by the exit path, and it would be wrong from the first flatten onwards: the
gate would keep refusing entries on a position that had already been closed, or
permit one on a position still open, and every log line would read correctly.
The book that decides whether to enter must be the same object the fill and the
flatten update.

`LONG`, `SHORT` and `FLAT` are IMPORTED for the same reason the formatter
imports its own action list. Three string constants restated in a second module
is how "long" and "LONG" come to mean different things in two files that both
look right.

WHAT THE GATE ANSWERS, AND WHAT IT CANNOT
=========================================
`can_execute` answers one question - *does this process already hold a position
that makes this order a stack?* - about a book that only knows what THIS
PROCESS opened. It is not a position report from the broker. A restart begins
believing it holds nothing (the base class does not persist, deliberately), so
after a restart the gate permits an entry on a position a previous run opened.
That is the same limitation `plan_exits` has and it is the safe direction for
an EXIT to be wrong in; for an ENTRY it is not, and the firewall's
`max_contracts_per_symbol` - which reads the durable `EngineState` - is what
covers that case. The two are complementary and neither replaces the other.

THE REVERSAL RULE HAS A SHARP EDGE
==================================
A BUY is permitted when the book says SHORT, and a SELL when it says LONG,
because a reversal is a real intention and refusing it would strand a position
whose signal has flipped. But `MAX_QTY` is 1: a BUY of one contract against a
short of one contract NETS FLAT at the broker, it does not open a long. The
book, told to record the fill, would then read LONG while the account is flat.

Nothing here can fix that from the entry side - the size that would reverse
rather than close is 2, and the cap forbids it. **Reverse by flattening and
then entering**, which is two orders and two states the book can represent.
The gate permits the single-order reversal because it was specified, and this
paragraph is the warning that the book's LONG after one is a claim about
intent rather than about the account.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.portfolio_manager import (                          # noqa: E402
    FLAT,
    LONG,
    SHORT,
    PositionBook as NettedPositionBook,
)

#: THE HARD CAP ON ONE ORDER, and it REJECTS rather than clamps.
#:
#: A clamp sends a different order from the one the sizer computed and reports
#: success, so a 3-contract position sized against a 3-contract stop goes on at
#: 1 and the risk model silently no longer describes the trade. A rejection is
#: visible: no order, a record saying what was asked for, and a number in the
#: cycle report that an operator can act on.
#:
#: IT DUPLICATES `max_contracts_per_order` IN THE FIREWALL ON PURPOSE. That
#: limit is configurable and the firewall is OPTIONAL - `LiveExecutionDispatcher`
#: accepts `firewall=None` and every construction outside `master_live.py`
#: leaves it there. This cap is neither: it is in the send path itself, so an
#: oversized order cannot reach the wire through a dispatcher nobody armed.
MAX_QTY = 1

#: The sides an entry can carry, and the two spellings of "close it".
#: `FLATTEN` is what this repository puts on the wire; `CLOSE` is what an
#: operator types (`send_test_probe.py --action CLOSE`) and what CrossTrade's
#: own documentation calls it. Both are accepted here so a caller using either
#: word gets the same verdict rather than an unrecognised-action refusal that
#: looks like a gate decision.
BUY, SELL = "BUY", "SELL"
CLOSING_ACTIONS = frozenset({"CLOSE", "FLATTEN"})

#: The exact line the dispatcher logs when the gate blocks an entry. Built here
#: rather than formatted at the call site so the wording cannot drift between
#: the console, the cycle report and the tests that assert on it.
HOLD_TEMPLATE = ("HOLD {portfolio_id}/{symbol} {action} — position already "
                 "active. Preventing stack.")


class PositionBook(NettedPositionBook):
    """
    The netted position book, plus `can_execute`.

    Everything about the state - the `(portfolio_id, symbol)` key, `record_fill`,
    `record_flat`, `direction`, `plan_exits` - is inherited unchanged from
    `portfolio.portfolio_manager.PositionBook`. This class adds the entry gate
    and nothing else, so there is exactly one record of what this process holds.
    """

    def state(self, portfolio_id: str, symbol: str) -> str:
        """
        `LONG`, `SHORT` or `FLAT` for one `(portfolio_id, symbol)`.

        A NAME FOR WHAT `direction()` ALREADY RETURNS, because the gate reads as
        a state machine and "state" is the word the rest of this module uses.
        It is not a second lookup: an alias, so there is no second answer.
        """
        return self.direction(portfolio_id, symbol)

    def can_execute(self, portfolio_id: str, symbol: str,
                    action: str) -> bool:
        """
        May this order be sent, given what this process already holds?

            BUY      FLAT -> open        SHORT -> reverse     LONG  -> NO
            SELL     FLAT -> open        LONG  -> reverse     SHORT -> NO
            CLOSE    anything but FLAT                        FLAT  -> NO
            FLATTEN  (the same as CLOSE)

        THE TWO `NO`s ARE THE WHOLE POINT. A BUY on a book that already says
        LONG is a second position on top of the first, and the loop re-evaluates
        the same signal every cycle - so a strategy that stays long-signalled
        does not enter once, it enters on every pass until something else stops
        it. Netting cannot catch this: `aggregate_signals` nets the signals
        arriving in ONE cycle against each other and knows nothing about the
        position the previous cycle opened.

        A CLOSE on a FLAT book is refused for the mirror reason: there is
        nothing recorded to close, and a flatten sent anyway acts on whatever
        the shared account happens to hold - including a position placed by
        hand, by NinjaTrader, or by a previous run. `plan_exits` enforces the
        same rule on the exit path and states it the same way.

        An unrecognised action returns **False**. A gate that let a word it did
        not understand through would be a gate that stops working the first
        time a caller invents a side.
        """
        current = self.state(portfolio_id, symbol)
        act = str(action).strip().upper()

        if act == BUY:
            return current in (FLAT, SHORT)
        if act == SELL:
            return current in (FLAT, LONG)
        if act in CLOSING_ACTIONS:
            return current != FLAT
        return False

    def hold_reason(self, portfolio_id: str, symbol: str,
                    action: str) -> str:
        """
        The line to log for an order `can_execute` refused.

        Carries the PORTFOLIO and the SYMBOL, not the strategy: the position is
        netted, so the thing already active belongs to every strategy on that
        pair and naming one of them would report the block against a strategy
        that may not even have signalled this cycle.
        """
        return HOLD_TEMPLATE.format(portfolio_id=portfolio_id, symbol=symbol,
                                    action=str(action).strip().upper())


def quantity_refusal(quantity: object, cap: int | None = None) -> dict | None:
    """
    The refusal record for an oversized order, or `None` when the size is fine.

    Returns a RECORD rather than raising. Every other refusal in the send path
    is a record too - one blocked order must not end a cycle that has others to
    place, and the record is the evidence that it was blocked and why. `rule`
    and `detail` are the same two keys the firewall's refusals carry, so
    whatever reads one reads the other.

    A non-integer or non-positive quantity is left ALONE here: the formatter
    and the firewall both refuse it with a message about what is wrong with it,
    and answering "cap is 1" to a quantity of `None` would name the wrong
    problem.
    """
    # READ AT CALL TIME so `MAX_QTY` can be raised in a test without every
    # caller having to thread a cap through. A default bound at def time
    # cannot be, and the tests that need an order to reach the wire would
    # then have to be written around the cap instead of against it.
    cap = MAX_QTY if cap is None else int(cap)
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        return None
    if quantity <= cap:
        return None
    return {"ok": False, "rule": "MAX_QTY",
            "detail": f"sizer asked {quantity}, cap is {cap}"}


__all__ = ["BUY", "CLOSING_ACTIONS", "FLAT", "HOLD_TEMPLATE", "LONG",
           "MAX_QTY", "PositionBook", "SELL", "SHORT", "quantity_refusal"]
