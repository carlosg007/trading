#!/usr/bin/env python3
"""
test_risk_firewall.py — the pre-trade gate and the durable record: that every
limit refuses on its own, that a check which cannot be evaluated blocks rather
than passes, and that a restart cannot re-send an order it already placed.

Location:  ~/src/trading/tests/test_risk_firewall.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_risk_firewall.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_risk_firewall.py

EVERY CASE FAILS THROUGH `assert`. Nothing here touches a broker: the firewall
is a pure function of a payload, a clock and a state file.

WHAT THIS COVERS, and why each case is here rather than assumed:

  * **EACH LIMIT REFUSES ALONE.** Every case holds the others comfortably clear
    and breaks one. A single "everything is wrong" fixture would pass against a
    gate that refused unconditionally, and the limits are fixed by completely
    different work.
  * **FAIL CLOSED.** A kill switch that cannot be read counts as ENGAGED, and a
    payload with no quantity is refused. The failure directions are not
    symmetric: unreadable-means-open disables a halt nobody notices until it
    was needed.
  * **THE DUPLICATE GUARD IS KEYED ON THE SIGNAL.** The loop acts on the last
    CLOSED bar, so a crash and restart inside one interval re-evaluates the
    same bar. The key is (strategy, symbol, bar_ts) — a fact about the signal —
    so it works without knowing anything about the broker.
  * **THE STATE FILE IS A RECORD, NOT A BELIEF.** Recovered dispatches come
    back UNVERIFIED and nothing acts on them, because the reconciliation that
    would make them positions does not exist.
  * **A CORRUPT STATE FILE REFUSES TO START.** "This process has sent nothing"
    is exactly the belief that re-sends an order it already placed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.portfolio_manager import PositionBook              # noqa: E402
from realtime.lifecycle import EngineState, emergency_halt         # noqa: E402
from realtime.risk_firewall import (RiskFirewall,                  # noqa: E402
                                    RiskViolation,
                                    arm_kill_switch,
                                    disarm_kill_switch,
                                    kill_switch_engaged)

ORDER = {"account": "SimIncubator1", "symbol": "MNQ", "action": "BUY",
         "orderType": "MARKET", "quantity": 1}
BAR = "2026-08-25T16:00:00Z"


def firewall(tmp_path: Path, limits=None, with_state=True) -> RiskFirewall:
    state = EngineState(tmp_path / "state.json") if with_state else None
    return RiskFirewall(limits=limits, state=state,
                        kill_switch=tmp_path / "KILL")


def refused(fw: RiskFirewall, order=None, **kwargs) -> str:
    with pytest.raises(RiskViolation) as caught:
        fw.check_order(dict(order or ORDER), **kwargs)
    return caught.value.rule


# --------------------------------------------------------------------------
# 1. the kill switch
# --------------------------------------------------------------------------

def test_a_clean_order_passes(tmp_path: Path) -> None:
    """The gate must not be a gate that refuses everything — that would pass
    every other case in this file for the wrong reason."""
    firewall(tmp_path).check_order(dict(ORDER), strategy_tag="t3", bar_ts=BAR)


def test_the_kill_switch_blocks_the_next_order_not_the_next_cycle(
        tmp_path: Path) -> None:
    """An operator arming it mid-cycle means now. It is checked before EVERY
    order rather than once per pass."""
    fw = firewall(tmp_path)
    fw.check_order(dict(ORDER), strategy_tag="t3", bar_ts=BAR)

    arm_kill_switch("drill", tmp_path / "KILL")
    assert refused(fw, strategy_tag="t3", bar_ts="LATER") == "kill_switch"

    assert disarm_kill_switch(tmp_path / "KILL") is True
    fw.check_order(dict(ORDER), strategy_tag="t3", bar_ts="LATER2")


def test_an_unreadable_kill_switch_counts_as_engaged(tmp_path: Path) -> None:
    """
    The failure directions are not symmetric. Unreadable-means-open lets a
    permissions mistake arm a halt nobody asked for, and somebody notices in
    minutes. Unreadable-means-closed lets the same mistake DISABLE the halt,
    and nobody notices until it was needed.
    """
    path = tmp_path / "KILL"
    path.mkdir()          # a directory where a file belongs: read() raises
    engaged, why = kill_switch_engaged(path)
    assert engaged is True
    assert "could not be read" in why


def test_the_switch_records_who_and_when(tmp_path: Path) -> None:
    path = arm_kill_switch("watchdog: feed degraded", tmp_path / "KILL")
    text = path.read_text()
    assert "watchdog: feed degraded" in text
    assert text[:4].isdigit(), "the timestamp leads the line"
    # Arming twice must not overwrite the FIRST reason — the original cause is
    # what an operator needs, not the most recent re-arm.
    arm_kill_switch("something else", path)
    assert "watchdog: feed degraded" in path.read_text()


# --------------------------------------------------------------------------
# 2. each limit, alone
# --------------------------------------------------------------------------

def test_an_oversized_order_is_refused(tmp_path: Path) -> None:
    fw = firewall(tmp_path, {"max_contracts_per_order": 5})
    assert refused(fw, dict(ORDER, quantity=6)) == "max_contracts_per_order"


def test_a_quantity_that_is_not_a_positive_whole_number_is_refused(
        tmp_path: Path) -> None:
    """No size limit can be evaluated against it, so the order does not go."""
    fw = firewall(tmp_path)
    for bad in (0, -1, 1.5, None, "1", True):
        assert refused(fw, dict(ORDER, quantity=bad)) == "quantity"


def test_the_session_cutoff_closes_new_entries(tmp_path: Path) -> None:
    fw = firewall(tmp_path, {"session_cutoff_utc": "20:55"})
    before = datetime(2026, 8, 25, 20, 54, tzinfo=timezone.utc)
    after = datetime(2026, 8, 25, 20, 55, tzinfo=timezone.utc)

    fw.check_order(dict(ORDER), now=before)
    assert refused(fw, now=after) == "session_cutoff"


def test_an_unparseable_cutoff_refuses_rather_than_trading_past_it(
        tmp_path: Path) -> None:
    with pytest.raises(RiskViolation, match="session_cutoff"):
        RiskFirewall(limits={"session_cutoff_utc": "eight-ish"})


def test_a_misspelled_limit_is_refused_at_construction() -> None:
    """A limit that is not enforced would LOOK configured, which is worse than
    one that is missing."""
    with pytest.raises(RiskViolation, match="unknown risk limit"):
        RiskFirewall(limits={"max_contracts_per_trade": 2})


def test_the_per_symbol_cap_counts_what_is_already_open(tmp_path: Path) -> None:
    state = EngineState(tmp_path / "state.json")
    fw = RiskFirewall(limits={"max_contracts_per_symbol": 3}, state=state,
                      kill_switch=tmp_path / "KILL")
    for i in range(3):
        state.record_order({"account": "SimIncubator1", "symbol": "MNQ",
                            "action": "BUY", "quantity": 1, "ok": True},
                           strategy="t3", bar_ts=f"bar{i}")
    assert refused(fw) == "max_contracts_per_symbol"


def test_the_order_count_stops_a_loop_that_is_looping(tmp_path: Path) -> None:
    state = EngineState(tmp_path / "state.json")
    fw = RiskFirewall(limits={"max_orders_per_session": 2}, state=state,
                      kill_switch=tmp_path / "KILL")
    for i in range(2):
        state.record_order({"account": "SimIncubator1", "symbol": "MES",
                            "action": "BUY", "quantity": 1, "ok": True},
                           strategy="t3", bar_ts=f"bar{i}")
    assert refused(fw) == "max_orders_per_session"


def test_the_local_loss_cap_halts_this_loop(tmp_path: Path) -> None:
    """
    A cap on THIS PROCESS's realised P&L — not the funding program's rule.
    That one resolves against a live account balance this process cannot see,
    and CrossTrade NAM enforces it. Two risk systems disagreeing about one
    number is worse than one enforcing it.
    """
    state = EngineState(tmp_path / "state.json")
    fw = RiskFirewall(limits={"max_session_loss_usd": 500.0}, state=state,
                      kill_switch=tmp_path / "KILL")
    state.record_fill("MNQ", -200.0)
    fw.check_order(dict(ORDER))

    state.record_fill("MNQ", -310.0)
    rule = refused(fw)
    assert rule == "max_session_loss_usd"
    assert "NAM" in fw.refusals[-1]["detail"], (
        "the refusal must say which system owns the rulebook limit")


# --------------------------------------------------------------------------
# 3. the duplicate guard
# --------------------------------------------------------------------------

def test_the_same_signal_cannot_be_sent_twice(tmp_path: Path) -> None:
    """
    THE CRASH-RECOVERY CASE. The loop acts on the last CLOSED bar; a crash and
    restart inside one interval re-evaluates the same bar and would re-send the
    same order. The key is a fact about the SIGNAL, so it needs to know nothing
    about the broker.
    """
    state = EngineState(tmp_path / "state.json")
    fw = RiskFirewall(state=state, kill_switch=tmp_path / "KILL")

    fw.check_order(dict(ORDER), strategy_tag="t3", bar_ts=BAR)
    state.record_dispatch("t3", "MNQ", BAR, ORDER)

    assert refused(fw, strategy_tag="t3", bar_ts=BAR) == "duplicate_bar"
    # A DIFFERENT bar is a different signal and must still go.
    fw.check_order(dict(ORDER), strategy_tag="t3",
                   bar_ts="2026-08-25T17:00:00Z")


def test_the_guard_survives_a_restart(tmp_path: Path) -> None:
    """The whole point: a NEW process, reading the file the old one left."""
    path = tmp_path / "state.json"
    first = EngineState(path)
    first.record_dispatch("t3", "MNQ", BAR, ORDER)

    restarted = EngineState(path)
    assert restarted.already_dispatched("t3", "MNQ", BAR) is True
    assert restarted.recovered is True
    assert restarted.reconciled is False, (
        "nothing read back from a previous run has been confirmed against the "
        "broker")


def test_a_corrupt_state_file_refuses_to_start(tmp_path: Path) -> None:
    """
    An empty history is exactly the belief that re-sends an order already
    placed, so an unreadable file is moved aside and the start is refused.
    """
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to start"):
        EngineState(path)
    assert path.with_suffix(".json.corrupt").exists()


def test_the_dispatch_is_recorded_before_the_socket(tmp_path: Path) -> None:
    """
    Deliberately the unsafe-looking order. If the process dies between the
    record and the send, the restart declines to re-send: a missed entry is a
    trade not taken, a double entry is a position nothing here can unwind. The
    unsent-but-recorded case is listed for a human instead.
    """
    state = EngineState(tmp_path / "state.json")
    state.record_dispatch("t3", "MNQ", BAR, ORDER)

    entry = state.blob["dispatched"][state.signal_key("t3", "MNQ", BAR)]
    assert entry["confirmed"] is False

    state.record_order({"account": "SimIncubator1", "symbol": "MNQ",
                        "action": "BUY", "quantity": 1, "ok": True},
                       strategy="t3", bar_ts=BAR)
    assert state.blob["dispatched"][
        state.signal_key("t3", "MNQ", BAR)]["confirmed"] is True


def test_the_state_write_is_atomic(tmp_path: Path) -> None:
    """Temp file then os.replace. A half-written state file is read on the next
    start as a SHORTER history: fewer orders sent, no record of the position
    that is actually open."""
    path = tmp_path / "state.json"
    state = EngineState(path)
    state.record_order({"account": "A", "symbol": "MNQ", "action": "BUY",
                        "quantity": 1, "ok": True}, strategy="t3", bar_ts=BAR)

    assert json.loads(path.read_text())["orders"], "the file parses"
    assert not list(tmp_path.glob(".engine_state.*.tmp")), "no temp left behind"


# --------------------------------------------------------------------------
# 4. the emergency
# --------------------------------------------------------------------------

def test_the_halt_arms_the_switch_before_it_tries_to_flatten(
        tmp_path: Path) -> None:
    """
    The order matters. Armed first and unconditionally, no further entry can be
    sent even if the flatten fails — broker down, network gone, dispatcher in a
    bad state. Flatten-then-arm could send an entry in between on the next
    cycle.
    """
    class ExplodingDispatcher:
        positions = None

        def dispatch_flatten(self, *_a, **_k):
            raise RuntimeError("broker unreachable")

    switch = tmp_path / "KILL"
    out = emergency_halt("test", dispatcher=ExplodingDispatcher(),
                         kill_switch=switch)

    assert switch.exists(), "armed even though the flatten path was unusable"
    assert kill_switch_engaged(switch)[0] is True
    assert out["reason"] == "test"


def test_the_halt_reports_unverified_claims_rather_than_closing_them(
        tmp_path: Path) -> None:
    """A blind flatten closes whatever is on a shared account, including
    positions placed by hand. Claims from a previous run are reported for a
    human."""
    state = EngineState(tmp_path / "state.json")
    state.record_dispatch("t3", "MNQ", BAR, ORDER)
    state.record_order({"account": "SimIncubator1", "symbol": "MNQ",
                        "action": "BUY", "quantity": 1, "ok": True},
                       strategy="t3", bar_ts=BAR)
    state.blob["dispatched"][state.signal_key("t3", "MNQ", BAR)][
        "confirmed"] = False

    out = emergency_halt("test", state=state, kill_switch=tmp_path / "KILL")
    assert out["unverified"], "the claim is surfaced"
    assert out["flattened"] == [], "and nothing was closed on it"


class HaltDispatcher:
    """
    A dispatcher with a REAL `PositionBook`, which is the whole point.

    The halt read `open_positions()` as a mapping - `.items()` on a list - and
    raised `AttributeError` inside the loop, so every flatten was skipped while
    the switch still armed and the alert still reported "flattened: none". It
    survived because the only case that reached this path used a stand-in whose
    `positions` was None. A hand-rolled fake would have hidden it again.
    """

    def __init__(self, accounts=None, fail=False):
        self.positions = PositionBook()
        self.accounts = (accounts if accounts is not None
                         else {"Incubator-Odd": "SimIncubator1"})
        self.fail = fail
        self.calls: list[str] = []

    def account_for(self, portfolio_id):
        try:
            return self.accounts[portfolio_id]
        except KeyError:
            raise RuntimeError(
                f"portfolio {portfolio_id!r} declares no target_account") from None

    def dispatch_account_flatten(self, account):
        self.calls.append(account)
        return {"account": account, "action": "FLATTEN_ACCOUNT",
                "scope": "account", "ok": not self.fail,
                "error": "broker refused" if self.fail else None}


def test_the_halt_actually_reaches_the_flatten_with_a_real_position_book(
        tmp_path: Path) -> None:
    """
    THE REGRESSION. `PositionBook.open_positions()` returns a LIST. Iterated as
    a mapping it raised inside the halt, and an `AttributeError` there is the
    worst possible place for one: the switch is armed, the alert goes out
    saying nothing needed closing, and the positions are still open.
    """
    d = HaltDispatcher()
    d.positions.record_fill("Incubator-Odd", "MNQ", "long", 1, ["t3_braid"])

    out = emergency_halt("test", dispatcher=d, kill_switch=tmp_path / "KILL")

    assert d.calls == ["SimIncubator1"], "the flatten was actually sent"
    assert out["failed"] == []
    assert out["flattened"] == [{"account": "SimIncubator1", "scope": "account",
                                 "covered": ["MNQ"], "error": None}]


def test_the_halt_sends_ONE_flatten_per_account_not_per_position(
        tmp_path: Path) -> None:
    """
    The account command closes everything on the account, so a second one for
    the next symbol is a duplicate kill with nothing left to close - an error
    response to read and a `failed` entry that reads like the halt did not
    work.
    """
    d = HaltDispatcher(accounts={"Incubator-Odd": "SimIncubator1",
                                 "Incubator-Even": "SimIncubator2"})
    d.positions.record_fill("Incubator-Odd", "MNQ", "long", 1, ["a"])
    d.positions.record_fill("Incubator-Odd", "MGC", "short", 1, ["b"])
    d.positions.record_fill("Incubator-Even", "MES", "long", 1, ["c"])

    out = emergency_halt("test", dispatcher=d, kill_switch=tmp_path / "KILL")

    assert d.calls == ["SimIncubator1", "SimIncubator2"]
    covered = {f["account"]: f["covered"] for f in out["flattened"]}
    assert covered == {"SimIncubator1": ["MGC", "MNQ"],
                       "SimIncubator2": ["MES"]}


def test_the_halt_stops_the_book_claiming_what_it_just_closed(
        tmp_path: Path) -> None:
    """
    Left standing, this process believes it still holds them, and the next exit
    flattens an account that is already flat - which on a shared account is not
    a no-op.
    """
    d = HaltDispatcher()
    d.positions.record_fill("Incubator-Odd", "MNQ", "long", 1, ["t3_braid"])

    emergency_halt("test", dispatcher=d, kill_switch=tmp_path / "KILL")
    assert d.positions.open_positions() == []


def test_a_failed_flatten_leaves_the_book_claiming_the_position(
        tmp_path: Path) -> None:
    """
    The book is cleared on a send that SUCCEEDED. Clearing it on a refusal
    would leave this process believing it is flat while the position is open,
    and nothing would try again.
    """
    d = HaltDispatcher(fail=True)
    d.positions.record_fill("Incubator-Odd", "MNQ", "long", 1, ["t3_braid"])

    out = emergency_halt("test", dispatcher=d, kill_switch=tmp_path / "KILL")

    assert out["flattened"] == []
    assert out["failed"][0]["error"] == "broker refused"
    assert d.positions.open_positions(), "still claimed, so it is still visible"


def test_a_portfolio_with_no_account_is_reported_not_guessed(
        tmp_path: Path) -> None:
    """
    The book is keyed by PORTFOLIO and orders route to an ACCOUNT. Sending the
    kill to a portfolio id would address an account that does not exist, and it
    would fail at the one moment nothing else is going to stop the loop.
    """
    d = HaltDispatcher(accounts={})
    d.positions.record_fill("Incubator-Odd", "MNQ", "long", 1, ["t3_braid"])

    out = emergency_halt("test", dispatcher=d, kill_switch=tmp_path / "KILL")

    assert d.calls == [], "nothing was sent to a guessed account"
    assert out["flattened"] == []
    assert "no account to flatten on" in out["failed"][0]["error"]
    assert out["failed"][0]["portfolio_id"] == "Incubator-Odd"


# --------------------------------------------------------------------------
# 5. without state, the gate says what it is not enforcing
# --------------------------------------------------------------------------

def test_a_firewall_with_no_state_says_which_limits_are_dark(
        tmp_path: Path) -> None:
    """Enforcing fewer rules than the operator believes is the failure mode of
    every risk system that reports only what it checked."""
    fw = firewall(tmp_path, with_state=False)
    fw.check_order(dict(ORDER), strategy_tag="t3", bar_ts=BAR)
    assert "NOT being enforced" in fw.describe()
    # The limits that need no history still bite.
    assert refused(fw, dict(ORDER, quantity=99)) == "max_contracts_per_order"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
