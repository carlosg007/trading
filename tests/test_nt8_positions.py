"""
tests/test_nt8_positions.py - broker position reconciliation.

ASSERT-BASED, so `tests/conftest.py` collects it normally. No `def check(`
marker: that is what routes a suite to the subprocess runner.

WHAT THIS IS GUARDING
=====================
The gap this closes, exactly: a strategy's stop IS computed live -
`signal_fn` runs the whole `_walk_loop` every cycle and `exits.iloc[-1]`
reaches `report["exit_signals"]` - and is then dropped by `plan_exits`, whose
FIRST condition is `is_open(pid, symbol)`. `PositionBook` holds only what THIS
PROCESS opened, so after a restart every stop was calculated and never sent,
under the message "no position opened by this process".

The cases below pin the reconciliation that fixes it and, more importantly,
the four ways a fix like this goes wrong quietly:

  * inventing a position from a stale snapshot
  * reading a malformed snapshot as "flat" and standing the account down
  * clearing the book on an account the publisher said nothing about
  * adopting a position the loop then still cannot exit
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.nt8_positions import (                                # noqa: E402
    MAX_SNAPSHOT_AGE_S,
    PositionSnapshotError,
    describe,
    load_snapshot,
    parse_position,
    reconcile,
)
from realtime.position_book import PositionBook                     # noqa: E402

ACCOUNTS = {"Incubator-Odd": "SimIncubator1",
            "Incubator-Even": "SimIncubator2"}
PORTFOLIOS = list(ACCOUNTS)


def account_for(pid: str) -> str:
    return ACCOUNTS[pid]


def _write(tmp_path: Path, positions: list[dict],
           age_s: float = 0.0) -> Path:
    stamp = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    path = tmp_path / "positions.json"
    path.write_text(json.dumps({"published_utc": stamp.isoformat(),
                                "positions": positions}))
    return path


def _pos(symbol="MNQ", qty=2, direction="long", account="SimIncubator1"):
    return {"account": account, "symbol": symbol, "quantity": qty,
            "direction": direction}


# --------------------------------------------------------------------------
# 1. Reading a snapshot
# --------------------------------------------------------------------------
def test_an_absent_snapshot_is_none_rather_than_an_error(tmp_path):
    """
    Until the NinjaScript publisher ships there is no snapshot, and raising
    would take the loop down over a feed it has never had. `None` leaves every
    caller in exactly the state it is in today.
    """
    assert load_snapshot(tmp_path / "nothing.json") is None


def test_a_stale_snapshot_is_refused(tmp_path):
    """
    A snapshot is a claim about NOW. One written an hour ago may describe
    positions closed by hand, by a bracket, or by the prop-firm layer since.
    """
    path = _write(tmp_path, [_pos()], age_s=MAX_SNAPSHOT_AGE_S + 60)
    with pytest.raises(PositionSnapshotError, match="published"):
        load_snapshot(path)


def test_a_naive_timestamp_is_refused(tmp_path):
    """
    The publisher runs on a Windows workstation in the instrument's or the
    operator's timezone. Read as UTC, a five-hour-old snapshot looks fresh.
    """
    path = tmp_path / "positions.json"
    path.write_text(json.dumps({"published_utc": "2026-09-03T01:00:00",
                                "positions": []}))
    with pytest.raises(PositionSnapshotError, match="NAIVE"):
        load_snapshot(path)


def test_a_malformed_snapshot_raises_rather_than_reading_as_flat(tmp_path):
    """
    THE INVERSION THAT MATTERS. A file that exists and cannot be parsed must
    not become "no positions" - that would be the loop deciding the account is
    flat because a publisher wrote bad JSON, and it would then decline every
    flatten it should have sent.
    """
    path = tmp_path / "positions.json"
    path.write_text("{not json")
    with pytest.raises(PositionSnapshotError):
        load_snapshot(path)


def test_a_negative_quantity_is_refused_rather_than_reinterpreted():
    """`{"quantity": -2, "direction": "long"}` is two statements about the
    side that disagree, and there is no correct way to resolve it."""
    with pytest.raises(PositionSnapshotError, match="UNSIGNED"):
        parse_position(_pos(qty=-2))


def test_a_flat_row_needs_no_direction():
    """An explicit `quantity: 0` is how a publisher says CLOSED, rather than
    leaving the reader to infer it from an absent row."""
    assert parse_position({"account": "A", "symbol": "X",
                           "quantity": 0})["direction"] == "flat"


# --------------------------------------------------------------------------
# 2. The gap this closes
# --------------------------------------------------------------------------
def test_an_unreconciled_book_refuses_the_exit_this_fixes(tmp_path):
    """The behaviour BEFORE reconciliation, pinned so the fix is measured
    against it rather than asserted."""
    book = PositionBook()
    intents = book.plan_exits(
        [{"portfolio_id": "Incubator-Odd", "symbol": "MNQ",
          "strategy_id": "s"}], {})
    assert intents[0]["emit"] is False
    assert "no position opened by this process" in intents[0]["reason"]


def test_an_adopted_position_becomes_flattenable(tmp_path):
    """
    THE WHOLE POINT. After reconciliation the same exit emits a FLATTEN, so a
    stop the strategy computed actually reaches the wire.
    """
    book = PositionBook()
    result = reconcile(book, load_snapshot(_write(tmp_path, [_pos()])),
                       account_for, PORTFOLIOS)
    assert result["reconciled"] is True
    assert len(result["adopted"]) == 1
    assert book.state("Incubator-Odd", "MNQ") == "long"

    intents = book.plan_exits(
        [{"portfolio_id": "Incubator-Odd", "symbol": "MNQ",
          "strategy_id": "any_strategy"}], {})
    assert intents[0]["emit"] is True, intents[0]["reason"]


def test_an_adopted_position_is_exitable_by_any_strategy(tmp_path):
    """
    Adopted positions carry NO owner: the loop did not open them, so no
    strategy owns them. `plan_exits`' owner check treats an unknown owner set
    as closeable, which is the correct direction here - otherwise reconciling
    would hand the loop inventory it still could not exit.
    """
    book = PositionBook()
    reconcile(book, load_snapshot(_write(tmp_path, [_pos()])),
              account_for, PORTFOLIOS)
    assert (book.get("Incubator-Odd", "MNQ") or {}).get("strategies") == []


def test_an_adopted_position_blocks_a_duplicate_entry(tmp_path):
    """The stack gate has to see it too, or the loop would open a second
    position on top of one the broker already holds."""
    book = PositionBook()
    reconcile(book, load_snapshot(_write(tmp_path, [_pos()])),
              account_for, PORTFOLIOS)
    assert book.can_execute("Incubator-Odd", "MNQ", "BUY") is False


# --------------------------------------------------------------------------
# 3. The three outcomes, kept apart
# --------------------------------------------------------------------------
def test_a_matching_position_is_confirmed_not_adopted(tmp_path):
    book = PositionBook()
    book.record_fill("Incubator-Odd", "MNQ", "long", 2, strategies=["s"])
    result = reconcile(book, load_snapshot(_write(tmp_path, [_pos()])),
                       account_for, PORTFOLIOS)
    assert len(result["confirmed"]) == 1 and not result["adopted"]
    assert (book.get("Incubator-Odd", "MNQ") or {}).get("strategies") == ["s"]


def test_a_disagreement_resolves_in_the_brokers_favour(tmp_path):
    """It is the account. The book is a belief about it."""
    book = PositionBook()
    book.record_fill("Incubator-Odd", "MNQ", "long", 1, strategies=["s"])
    result = reconcile(book,
                       load_snapshot(_write(tmp_path, [_pos(qty=3)])),
                       account_for, PORTFOLIOS)
    assert len(result["conflicts"]) == 1
    assert book.get("Incubator-Odd", "MNQ")["quantity"] == 3
    # The owning strategy survives a size correction - it still opened it.
    assert book.get("Incubator-Odd", "MNQ")["strategies"] == ["s"]


def test_a_position_closed_elsewhere_is_dropped_from_the_book(tmp_path):
    """
    Flattened by hand, by a bracket or by the prop-firm layer. Left in the
    book the loop keeps trying to flatten something already gone.
    """
    book = PositionBook()
    book.record_fill("Incubator-Odd", "MNQ", "long", 2, strategies=["s"])
    result = reconcile(book, load_snapshot(_write(tmp_path, [
        _pos(symbol="MNQ", qty=0, direction="flat")])), account_for,
        PORTFOLIOS)
    assert len(result["closed_elsewhere"]) == 1
    assert book.state("Incubator-Odd", "MNQ") == "flat"


def test_an_absent_row_on_a_covered_account_also_closes(tmp_path):
    """A publisher that reported the account and did not mention the pair is
    saying the pair is closed."""
    book = PositionBook()
    book.record_fill("Incubator-Odd", "MNQ", "long", 2, strategies=["s"])
    reconcile(book, load_snapshot(_write(tmp_path, [_pos(symbol="6J")])),
              account_for, PORTFOLIOS)
    assert book.state("Incubator-Odd", "MNQ") == "flat"


def test_an_account_the_snapshot_never_mentioned_is_left_alone(tmp_path):
    """
    THE SILENCE TRAP. A publisher that sent SimIncubator1's positions says
    nothing about SimIncubator2, and clearing the book on that silence would
    drop this loop's own record of a live position.
    """
    book = PositionBook()
    book.record_fill("Incubator-Even", "MES", "long", 1, strategies=["s"])
    reconcile(book, load_snapshot(_write(tmp_path, [_pos()])),
              account_for, PORTFOLIOS)
    assert book.state("Incubator-Even", "MES") == "long", (
        "an untouched account's position was cleared on silence")


def test_a_row_for_an_unrouted_account_is_reported_not_dropped(tmp_path):
    book = PositionBook()
    result = reconcile(book, load_snapshot(_write(
        tmp_path, [_pos(account="SomeOtherAccount")])), account_for,
        PORTFOLIOS)
    assert len(result["skipped"]) == 1
    assert "no portfolio routes" in result["skipped"][0]["reason"]


# --------------------------------------------------------------------------
# 4. No publisher, and the EngineState flag
# --------------------------------------------------------------------------
def test_no_snapshot_leaves_the_book_exactly_as_it_was(tmp_path):
    book = PositionBook()
    book.record_fill("Incubator-Odd", "MNQ", "long", 2, strategies=["s"])
    result = reconcile(book, None, account_for, PORTFOLIOS)
    assert result["reconciled"] is False
    assert "no broker snapshot" in result["reason"]
    assert book.state("Incubator-Odd", "MNQ") == "long"
    assert "NOT RECONCILED" in describe(result)


def test_reconciling_sets_the_flag_engine_state_has_always_carried(tmp_path):
    """
    `EngineState.reconciled` ships False with the comment "Only an explicit
    reconciliation may set this True" and nothing ever did. This is it.
    """
    class _State:
        reconciled = False

    state = _State()
    reconcile(PositionBook(), load_snapshot(_write(tmp_path, [_pos()])),
              account_for, PORTFOLIOS, state=state)
    assert state.reconciled is True


def test_a_failed_reconciliation_does_not_set_the_flag(tmp_path):
    class _State:
        reconciled = False

    state = _State()
    reconcile(PositionBook(), None, account_for, PORTFOLIOS, state=state)
    assert state.reconciled is False


# --------------------------------------------------------------------------
# 5. The wiring
# --------------------------------------------------------------------------
def test_the_live_loop_reconciles_before_it_reads_a_signal():
    """
    Order matters: reconciling AFTER the plan is built would leave the first
    cycle's exits refused on the empty book it was about to fix.
    """
    import inspect

    import master_live as M

    src = inspect.getsource(M.main)
    assert "reconcile(dispatcher.positions" in src
    assert src.index("reconcile(dispatcher.positions") < src.index(
        "for bucket_tf in buckets")


def test_reconciliation_is_on_by_default_and_can_be_declined():
    import master_live as M

    assert M.build_parser().parse_args([]).reconcile_positions is True
    assert M.build_parser().parse_args(
        ["--no-position-reconcile"]).reconcile_positions is False


def test_the_listener_exposes_the_positions_route():
    from realtime.nt8_bar_listener import create_app

    assert "/api/positions" in {r.path for r in create_app().routes}


def test_a_refused_snapshot_does_not_become_the_processes_exit_code():
    """
    THE REGRESSION THIS PINS. `failures` is `main`'s return value and it is
    CUMULATIVE over the run, so one refusal at hour one made the SIGTERM at
    hour twelve exit 1 and systemd report a clean shutdown as
    `status=1/FAILURE`. Observed on 2026-09-03.

    A stale snapshot is a degraded INPUT the loop is built to handle - the
    book stays unreconciled, which is the state it was in before this feed
    existed. It belongs on stderr and in the watchdog, not in the exit code.
    """
    import inspect

    import master_live as M

    src = inspect.getsource(M.main)
    handler = src.index("position snapshot REFUSED")
    # The next `failures += 1` after the handler must not belong to it. The
    # bar-load failure below is pre-existing and IS a run failure.
    following = src[handler:handler + 900]
    assert "failures += 1" not in following, (
        "a refused snapshot still increments the exit-code counter")


def test_the_snapshot_is_read_every_cycle_not_once_at_startup():
    """
    A position closed by hand, by a bracket or by the prop-firm layer between
    cycles has to leave the book too, or the loop keeps trying to flatten
    something already gone.
    """
    import inspect

    import master_live as M

    src = inspect.getsource(M.main)
    assert src.index("load_snapshot(args.positions_snapshot)") > src.index(
        "while True:"), "reconciliation sits outside the cycle loop"
