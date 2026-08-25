#!/usr/bin/env python3
"""
test_incubator_recorder.py — NT8 forward fills into ledger trades: that the
pairing is the round turn the account actually took, that costs are in every
P&L figure, that a row nobody can attribute is reported rather than guessed,
and that re-reading the same export changes nothing.

Location:  ~/src/trading/tests/test_incubator_recorder.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_incubator_recorder.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_incubator_recorder.py

EVERY CASE FAILS THROUGH `assert`, like `test_incubator_tracker.py`. Nothing
here needs the lake, a network, or `/mnt/backtest`: every fixture is a small
CSV or JSON export written into `tmp_path`, which is also the only honest way
to test this today — `/mnt/backtest/artifacts/incubator_logs/` has never held a
file, so a suite that read the real directory would be testing that it is
still empty.

WHAT THIS COVERS, and why each case is here rather than assumed:

  * **THE PAIRING IS THE ACCOUNT'S, NOT A SUMMARY.** FIFO, matched in pieces,
    so a 2-lot entry closed by two 1-lot exits is two trades priced
    separately. Averaged into one they would net to a single figure and a
    winner would hide a loser — and the trade COUNT is criterion two.
  * **AN OPEN POSITION IS NOT A TRADE.** It has no realised P&L. Counting it
    would let a strategy reach the fourteen-trade bar on positions whose
    outcome nobody has seen.
  * **COSTS ARE IN EVERY FIGURE.** A forward profit factor is compared against
    1.00; the commission on fourteen round turns is the whole decision at that
    boundary. Each of the three bases is pinned, and each trade records which
    one priced it.
  * **THE MULTIPLIER IS READ, NEVER GUESSED.** A symbol with no ContractSpec
    is reported and skipped. A guessed multiplier scales every P&L figure for
    that contract and nothing downstream would look wrong.
  * **ATTRIBUTION IS EVIDENCE.** The strategy the log names wins; a log that
    names none is attributed only when the account holds exactly one strategy
    trading that contract. Two candidates is UNATTRIBUTED — a trade filed
    under the wrong strategy is a promotion decided on somebody else's P&L.
  * **RE-READING IS A NO-OP.** The cron job reads a growing export every
    evening. Duplicated trades would make a strategy look like it cleared the
    fourteen-trade bar twice as fast as it did.
  * **THE RECORDER FEEDS THE GATE.** The last case runs a recorded ledger
    through `evaluate_strategy_promotion`, because the trade shape written
    here is only correct if the daemon can score it.
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

from backtest.specs import get_spec                                # noqa: E402
from portfolio import config_loader                                # noqa: E402
from portfolio.incubator_recorder import (                         # noqa: E402
    NT8_ACCOUNT_ALIASES,
    RecorderError,
    attribute,
    build_trades,
    discover_logs,
    merge_into_ledger,
    parse_rows,
    portfolio_for_account,
    root_symbol,
)
from portfolio.promotion_daemon import (                           # noqa: E402
    STATUS_GRADUATED,
    STATUS_INCUBATING,
    evaluate_strategy_promotion,
)

CONFIG_PATH = REPO_ROOT / "config" / "portfolios.json"

# MNQ: $2 a point, and the round turn from the spec rather than a literal, so
# a commission change moves the expectations with the table.
MNQ = get_spec("MNQ")
POINT = float(MNQ.multiplier)
ROUND_TURN = float(MNQ.round_turn_cost)

# A routing table with one strategy on one account, written out rather than
# copied from the real config: these cases are about attribution, and pinning
# them to whatever is registered today would make them fail the next time
# somebody promotes something.
CONFIG = {"portfolios": {
    "Incubator-Odd": {
        "portfolio_id": "Incubator-Odd",
        "target_account": "SimIncubator1",
        "account_type": "incubator_sim",
        "active_strategies": ["alpha"],
        "strategy_allocations": {"alpha": {"symbol": "NQ"}},
        "basket": {"assets": ["MNQ", "MCL"]},
    },
    "Incubator-Even": {
        "portfolio_id": "Incubator-Even",
        "target_account": "SimIncubator2",
        "account_type": "incubator_sim",
        "active_strategies": [],
        "strategy_allocations": {},
        "basket": {"assets": ["MES", "MGC"]},
    },
}}

EXEC_HEADER = "Time,Instrument,Account,Action,Quantity,Price\n"


def write_exec_log(directory: Path, name: str, rows: list[str]) -> Path:
    """An NT8 EXECUTIONS-tab export: one row per fill, no P&L column."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(EXEC_HEADER + "".join(rows), encoding="utf-8")
    return path


def fill(ts: str, action: str, qty: int, price: float,
         instrument: str = "MNQ 12-26", account: str = "SimIncubator1") -> str:
    return f"{ts},{instrument},{account},{action},{qty},{price}\n"


def built_for(path: Path, config: dict | None = None, **kwargs) -> dict:
    return build_trades(parse_rows(path), config or CONFIG, **kwargs)


# --------------------------------------------------------------------------
# 1. reading the export
# --------------------------------------------------------------------------

def test_the_expiry_is_dropped_from_the_instrument() -> None:
    """`MNQ 12-26` is MNQ. A contract spec has no month, and keeping one would
    turn every roll into a new symbol the recorder refuses to price."""
    assert root_symbol("MNQ 12-26") == "MNQ"
    assert root_symbol("  es 03-26 ") == "ES"
    assert root_symbol("MNQ") == "MNQ"
    assert root_symbol("") == ""
    assert root_symbol(None) == ""


def test_a_missing_log_directory_raises_and_an_empty_one_does_not(
        tmp_path: Path) -> None:
    """Different failures. No directory means the NT8 export was never wired
    up; an empty one means a quiet session, which is an ordinary thing for a
    post-market job to find."""
    with pytest.raises(RecorderError, match="no incubator log directory"):
        discover_logs(tmp_path / "never_created")

    empty = tmp_path / "logs"
    empty.mkdir()
    assert discover_logs(empty) == []


def test_a_json_export_reads_the_same_as_a_csv_one(tmp_path: Path) -> None:
    """Both shapes come through the ONE reader in `live/dispatcher.py`, so a
    template that exports JSON is not a second parser."""
    csv_log = write_exec_log(tmp_path, "a.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-01 15:30:00", "Sell", 1, 20010.0)])
    json_log = tmp_path / "b.json"
    json_log.write_text(json.dumps([
        {"Time": "2026-08-01 14:30:00", "Instrument": "MNQ 12-26",
         "Account": "SimIncubator1", "Action": "Buy", "Quantity": 1,
         "Price": 20000.0},
        {"Time": "2026-08-01 15:30:00", "Instrument": "MNQ 12-26",
         "Account": "SimIncubator1", "Action": "Sell", "Quantity": 1,
         "Price": 20010.0}]), encoding="utf-8")

    from_csv = built_for(csv_log)["trades"]["alpha"]
    from_json = built_for(json_log)["trades"]["alpha"]
    assert len(from_csv) == len(from_json) == 1
    assert from_csv[0]["pnl"] == from_json[0]["pnl"]
    assert from_csv[0]["trade_id"] == from_json[0]["trade_id"]


# --------------------------------------------------------------------------
# 2. pairing
# --------------------------------------------------------------------------

def test_fills_pair_fifo_and_partial_closes_are_separate_trades(
        tmp_path: Path) -> None:
    """
    A 2-lot entry closed by a 1-lot winner and a 1-lot loser is TWO trades.

    Averaged into one they net to a single figure, the winner hides the loser,
    and the trade count — criterion two — is halved.
    """
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 2, 20000.0),
        fill("2026-08-01 15:30:00", "Sell", 1, 20010.0),
        fill("2026-08-01 16:30:00", "Sell", 1, 19990.0)])
    trades = built_for(log)["trades"]["alpha"]

    assert len(trades) == 2
    assert [t["quantity"] for t in trades] == [1.0, 1.0]
    assert trades[0]["pnl"] == round(10 * POINT - ROUND_TURN, 2)
    assert trades[1]["pnl"] == round(-10 * POINT - ROUND_TURN, 2)
    assert all(t["direction"] == "LONG" for t in trades)
    assert all(t["status"] == "CLOSED" for t in trades)


def test_a_short_is_priced_in_the_direction_it_was_taken(
        tmp_path: Path) -> None:
    """Sell high, buy back low, and the P&L is POSITIVE. A recorder that
    priced every round turn as a long would report the profitable half of a
    two-sided strategy as its losses."""
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Sell", 1, 20000.0),
        fill("2026-08-01 15:30:00", "Buy", 1, 19980.0)])
    trades = built_for(log)["trades"]["alpha"]

    assert len(trades) == 1
    assert trades[0]["direction"] == "SHORT"
    assert trades[0]["pnl"] == round(20 * POINT - ROUND_TURN, 2)


def test_an_open_position_is_reported_and_is_not_a_trade(
        tmp_path: Path) -> None:
    """Its outcome is unknown. Counting it would let a strategy reach the
    fourteen-trade bar on positions it has not exited."""
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-01 15:30:00", "Sell", 1, 20010.0),
        fill("2026-08-02 14:30:00", "Buy", 3, 20050.0)])
    built = built_for(log)

    assert len(built["trades"]["alpha"]) == 1
    assert built["open_positions"] == {"alpha/MNQ": 3}


def test_a_rejected_fill_is_not_a_trade(tmp_path: Path) -> None:
    """NT8 logs the order that never filled beside the one that did, and the
    fill test is `live/dispatcher.py`'s — one definition, so a status this
    repository stops recognising is not a trade the recorder starts
    inventing."""
    path = tmp_path / "exec.csv"
    path.write_text(
        "Time,Instrument,Account,Action,Quantity,Price,Status\n"
        "2026-08-01 14:30:00,MNQ 12-26,SimIncubator1,Buy,1,20000.0,Filled\n"
        "2026-08-01 15:00:00,MNQ 12-26,SimIncubator1,Sell,1,20005.0,Rejected\n"
        "2026-08-01 15:30:00,MNQ 12-26,SimIncubator1,Sell,1,20010.0,Filled\n",
        encoding="utf-8")
    trades = built_for(path)["trades"]["alpha"]

    assert len(trades) == 1
    assert trades[0]["exit_price"] == 20010.0


def test_an_unpairable_row_is_counted_rather_than_dropped(
        tmp_path: Path) -> None:
    """An execution the recorder could not read is a trade the ledger is
    missing, and the count is how anybody notices."""
    path = tmp_path / "exec.csv"
    path.write_text(
        "Time,Instrument,Account,Action,Quantity,Price\n"
        "2026-08-01 14:30:00,MNQ 12-26,SimIncubator1,Buy,1,20000.0\n"
        "2026-08-01 15:00:00,MNQ 12-26,SimIncubator1,Adjust,1,20005.0\n"
        "2026-08-01 15:30:00,MNQ 12-26,SimIncubator1,Sell,1,20010.0\n",
        encoding="utf-8")
    built = built_for(path)

    assert len(built["trades"]["alpha"]) == 1
    assert len(built["problems"]) == 1
    assert "not BUY/SELL" in built["problems"][0]


def test_each_trade_carries_the_session_it_exited_in(tmp_path: Path) -> None:
    """
    The 18:00 ET roll, from `backtest.event_calendar` and not re-derived.

    An exit at 22:00 UTC on the 3rd is 18:00 ET, which is the FOURTH session.
    `active_sessions` is half of criterion one, and a recorder that stamped
    the calendar date would move a trade into the session before it.
    """
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-03T21:00:00+00:00", "Buy", 1, 20000.0),
        fill("2026-08-03T22:30:00+00:00", "Sell", 1, 20010.0)])
    trades = built_for(log)["trades"]["alpha"]

    assert trades[0]["session_date"] == "2026-08-04"


# --------------------------------------------------------------------------
# 3. costs
# --------------------------------------------------------------------------

def test_costs_are_applied_when_the_export_carries_none(
        tmp_path: Path) -> None:
    """The spec's round turn, because a forward profit factor is compared
    against 1.00 and a gross figure clears bars a net one does not."""
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-01 15:30:00", "Sell", 1, 20005.0)])
    trade = built_for(log)["trades"]["alpha"][0]

    assert trade["pnl"] == round(5 * POINT - ROUND_TURN, 2)
    assert trade["cost_basis"] == "specs round turn"


def test_the_logs_own_commission_is_preferred_and_prorated(
        tmp_path: Path) -> None:
    """
    A commission column describes the WHOLE row. A 1-lot exit against a 2-lot
    entry paid half of the entry's fee, and charging the entry's full
    commission to the first close would price the second one free.
    """
    path = tmp_path / "exec.csv"
    path.write_text(
        "Time,Instrument,Account,Action,Quantity,Price,Commission\n"
        "2026-08-01 14:30:00,MNQ 12-26,SimIncubator1,Buy,2,20000.0,1.00\n"
        "2026-08-01 15:30:00,MNQ 12-26,SimIncubator1,Sell,1,20010.0,0.50\n"
        "2026-08-01 16:30:00,MNQ 12-26,SimIncubator1,Sell,1,20020.0,0.50\n",
        encoding="utf-8")
    trades = built_for(path)["trades"]["alpha"]

    # half of the entry's $1.00 plus the whole of this exit's $0.50
    assert trades[0]["pnl"] == round(10 * POINT - 1.00, 2)
    assert trades[1]["pnl"] == round(20 * POINT - 1.00, 2)
    assert all(t["cost_basis"] == "log commission" for t in trades)


def test_cost_basis_log_leaves_an_already_net_export_alone(
        tmp_path: Path) -> None:
    """For a template that exports net of fees. Subtracting again would charge
    the strategy twice, which at the 1.00 boundary is the decision."""
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-01 15:30:00", "Sell", 1, 20005.0)])
    trade = built_for(log, cost_basis="log")["trades"]["alpha"][0]

    assert trade["pnl"] == round(5 * POINT, 2)
    assert "log" in trade["cost_basis"]


def test_an_unknown_cost_basis_raises_rather_than_defaulting() -> None:
    with pytest.raises(RecorderError, match="cost_basis"):
        build_trades([], CONFIG, cost_basis="free")


def test_a_symbol_with_no_contract_spec_is_skipped_and_reported(
        tmp_path: Path) -> None:
    """A guessed multiplier scales every P&L figure for that contract, and
    nothing downstream would look wrong."""
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 100.0, instrument="ZZZ 12-26"),
        fill("2026-08-01 15:30:00", "Sell", 1, 105.0, instrument="ZZZ 12-26")])
    config = json.loads(json.dumps(CONFIG))
    config["portfolios"]["Incubator-Odd"]["basket"]["assets"].append("ZZZ")
    config["portfolios"]["Incubator-Odd"]["strategy_allocations"] = {}
    built = built_for(log, config)

    assert built["trades"] == {}
    assert built["unpriceable_symbols"] == {"ZZZ": 2}


# --------------------------------------------------------------------------
# 4. attribution
# --------------------------------------------------------------------------

def test_the_nt8_account_resolves_to_its_portfolio() -> None:
    """
    `SimIncubator1` is the routing table's `Incubator-Odd`, and the map is the
    only place the two are tied together.

    NinjaTrader prefixes a simulation account with `Sim`; the portfolio keeps
    the Odd/Even id, which encodes the basket split. Matched case-insensitively
    because an export's capitalisation is NT8's business, and an id or a
    `target_account` resolves to itself so a log already written in
    routing-table names needs no entry in the map at all.
    """
    assert NT8_ACCOUNT_ALIASES["SIMINCUBATOR1"] == "Incubator-Odd"
    assert NT8_ACCOUNT_ALIASES["SIMPROP2"] == "Prop-Even"
    assert portfolio_for_account("SimIncubator1", CONFIG) == "Incubator-Odd"
    assert portfolio_for_account("simincubator2", CONFIG) == "Incubator-Even"
    assert portfolio_for_account("Incubator-Odd", CONFIG) == "Incubator-Odd"
    assert portfolio_for_account("SimNotOurs", CONFIG) is None
    assert portfolio_for_account("", CONFIG) is None


def test_the_alias_map_and_the_routing_table_name_the_same_accounts() -> None:
    """
    THE TWO HALVES OF ONE ROUND TRIP, RECONCILED.

    `target_account` is what the live loop SENDS an order to; the alias map is
    what the recorder reads a fill BACK through. They are written in two files
    and nothing but this makes them agree. An order sent to an account
    NinjaTrader does not have is rejected at one end of the day; a fill
    recorded under an account no portfolio claims is unattributed at the
    other, and neither failure mentions the other file.
    """
    portfolios = config_loader.load_portfolio_config()["portfolios"]
    for pid, portfolio in portfolios.items():
        account = portfolio["target_account"]
        assert NT8_ACCOUNT_ALIASES.get(account.upper()) == pid, (
            f"{pid} executes on {account!r}, which the alias map resolves to "
            f"{NT8_ACCOUNT_ALIASES.get(account.upper())!r}")
    assert set(NT8_ACCOUNT_ALIASES.values()) == set(portfolios)


def test_the_strategy_the_log_names_wins() -> None:
    """`format_crosstrade_json` already sends `strategy_tag`; when the export
    carries it back, nothing has to be inferred."""
    row = {"symbol": "MNQ 12-26", "account": "SimIncubator1", "strategy": "named_one"}
    strategy, portfolio, why = attribute(row, CONFIG)
    assert (strategy, portfolio) == ("named_one", "Incubator-Odd")
    assert why == "named by the log"


def test_a_row_naming_no_strategy_is_attributed_only_when_it_is_unambiguous(
) -> None:
    """One strategy on that account trading that contract is evidence. Two is
    a guess, and a trade filed under the wrong strategy is a promotion decided
    on somebody else's P&L."""
    row = {"symbol": "MNQ 12-26", "account": "SimIncubator1"}
    assert attribute(row, CONFIG)[0] == "alpha"

    crowded = json.loads(json.dumps(CONFIG))
    odd = crowded["portfolios"]["Incubator-Odd"]
    odd["active_strategies"] = ["alpha", "beta"]
    odd["strategy_allocations"]["beta"] = {"symbol": "NQ"}
    strategy, portfolio, why = attribute(row, crowded)
    assert strategy is None
    assert portfolio == "Incubator-Odd"
    assert "refusing to guess" in why


def test_a_strategy_registered_on_the_full_size_contract_owns_the_micros_fills(
) -> None:
    """`alpha` is registered on NQ and the account fills MNQ. Same price
    series, and the order was always for the micro — resolved through the one
    table in `realtime/contract_alias.py`."""
    assert attribute({"symbol": "MNQ 12-26", "account": "SimIncubator1"},
                     CONFIG)[0] == "alpha"


def test_an_unknown_account_is_unattributed_rather_than_assigned(
        tmp_path: Path) -> None:
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0, account="SimNotOurs"),
        fill("2026-08-01 15:30:00", "Sell", 1, 20010.0, account="SimNotOurs")])
    built = built_for(log)

    assert built["trades"] == {}
    assert len(built["unattributed"]) == 2
    assert "matches no portfolio" in built["unattributed"][0]["reason"]


# --------------------------------------------------------------------------
# 5. trade-level exports
# --------------------------------------------------------------------------

def test_a_trade_level_export_is_taken_as_the_round_turn_it_is(
        tmp_path: Path) -> None:
    """NT8's Trades tab already carries the round turn and its profit; pairing
    it again would be inventing an entry the export already stated."""
    path = tmp_path / "trades.csv"
    path.write_text(
        "Strategy,Instrument,Account,Entry time,Exit time,Quantity,Profit,"
        "Commission,Action\n"
        "alpha,MNQ 12-26,SimIncubator1,2026-08-01 14:30:00,2026-08-01 15:30:00,"
        "1,40.00,1.40,Buy\n",
        encoding="utf-8")
    trades = built_for(path)["trades"]["alpha"]

    assert len(trades) == 1
    assert trades[0]["shape"] == "trade"
    assert trades[0]["pnl"] == 38.60
    assert trades[0]["cost_basis"] == "log commission"


def test_a_declared_net_figure_is_not_charged_again(tmp_path: Path) -> None:
    path = tmp_path / "trades.csv"
    path.write_text(
        "Strategy,Instrument,Account,Exit time,Quantity,Net_pnl\n"
        "alpha,MNQ 12-26,SimIncubator1,2026-08-01 15:30:00,1,38.60\n",
        encoding="utf-8")
    trade = built_for(path)["trades"]["alpha"][0]

    assert trade["pnl"] == 38.60
    assert trade["cost_basis"] == "log, declared net"


# --------------------------------------------------------------------------
# 6. merging into the ledger
# --------------------------------------------------------------------------

def test_reading_the_same_export_twice_records_it_once(
        tmp_path: Path) -> None:
    """The cron job reads a growing export every evening. Duplicates would
    make a strategy look like it cleared the fourteen-trade bar twice as fast
    as it did."""
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-01 15:30:00", "Sell", 1, 20010.0)])
    built = built_for(log)
    ledger = {"version": "1.0.0", "strategies": {}}

    first = merge_into_ledger(ledger, built)
    second = merge_into_ledger(ledger, built)

    assert first["added"] == {"alpha": 1}
    assert first["created"] == ["alpha"]
    assert second["added"] == {"alpha": 0}
    assert second["skipped"] == {"alpha": 1}
    assert len(ledger["strategies"]["alpha"]["trades"]) == 1


def test_a_new_entry_records_its_account_and_its_first_trade(
        tmp_path: Path) -> None:
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-05 15:30:00", "Sell", 1, 20010.0)])
    ledger = {"version": "1.0.0", "strategies": {}}
    merge_into_ledger(ledger, built_for(log))
    entry = ledger["strategies"]["alpha"]

    assert entry["status"] == STATUS_INCUBATING
    assert entry["portfolio"] == "Incubator-Odd"
    assert entry["started_at"].startswith("2026-08-01")
    # Nothing declared: the daemon derives every criterion from the trades,
    # and a summary written here would be a second implementation of the gate
    # that is free to disagree with it.
    assert "realized_pf" not in entry
    assert "trade_count" not in entry


def test_a_started_at_later_than_the_first_trade_is_pulled_back(
        tmp_path: Path) -> None:
    """The window is criterion one. A bookkeeping stamp later than the trades
    themselves would shorten a window the ledger already proves."""
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-05 15:30:00", "Sell", 1, 20010.0)])
    ledger = {"version": "1.0.0", "strategies": {"alpha": {
        "status": STATUS_INCUBATING, "portfolio": "Incubator-Odd",
        "started_at": "2026-08-04T00:00:00+00:00", "trades": []}}}
    merge_into_ledger(ledger, built_for(log))

    assert ledger["strategies"]["alpha"]["started_at"].startswith("2026-08-01")


def test_a_graduated_entry_is_never_appended_to(tmp_path: Path) -> None:
    """Its forward trades are being taken on a prop account. They are not
    incubation evidence for a decision that has already been made."""
    log = write_exec_log(tmp_path, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-01 15:30:00", "Sell", 1, 20010.0)])
    ledger = {"version": "1.0.0", "strategies": {"alpha": {
        "status": STATUS_GRADUATED, "portfolio": "Prop-Odd", "trades": []}}}
    report = merge_into_ledger(ledger, built_for(log))

    assert ledger["strategies"]["alpha"]["trades"] == []
    assert any(STATUS_GRADUATED in note for note in report["refused"])


# --------------------------------------------------------------------------
# 7. the recorded ledger is what the gate reads
# --------------------------------------------------------------------------

def test_a_recorded_entry_scores_through_the_promotion_daemon(
        tmp_path: Path) -> None:
    """
    The shape written here is only correct if the daemon can grade it.

    Fifteen sessions of one winning round turn each: the window, the sample
    and the drawdown all read off the trade list the recorder produced, with
    no declared summary anywhere in the entry.
    """
    rows = []
    for day in range(1, 16):
        stamp = f"2026-08-{day:02d}"
        rows.append(fill(f"{stamp} 14:30:00", "Buy", 1, 20000.0))
        rows.append(fill(f"{stamp} 15:30:00", "Sell", 1, 20010.0))
    log = write_exec_log(tmp_path, "exec.csv", rows)

    ledger = {"version": "1.0.0", "strategies": {}}
    merge_into_ledger(ledger, built_for(log))
    entry = ledger["strategies"]["alpha"]

    portfolio = config_loader.load_portfolio_config()["portfolios"]["Incubator-Odd"]
    report = evaluate_strategy_promotion("alpha", entry, portfolio)

    assert report["metrics"]["source"] == "derived_from_trades"
    assert report["metrics"]["trade_count"] == 15
    assert report["metrics"]["days_active"] == 15
    assert report["metrics"]["active_sessions"] == 15
    assert report["passed"] is True, report["reasons"]


# --------------------------------------------------------------------------
# 8. the CLI
# --------------------------------------------------------------------------

def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "scripts" / "record_incubator_fills.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120)


@pytest.fixture()
def cli_workspace(tmp_path: Path):
    """A copy of the real routing table with one strategy on Incubator-Odd, an
    empty ledger, and one export. Copies, not the originals: a test that wrote
    the live ledger would look exactly like a recorded session."""
    config_path = tmp_path / "portfolios.json"
    shutil.copy(CONFIG_PATH, config_path)
    config = json.loads(config_path.read_text())
    odd = config["portfolios"]["Incubator-Odd"]
    odd["active_strategies"] = ["alpha"]
    odd["strategy_allocations"] = {"alpha": {
        "strat": "alpha", "symbol": "NQ", "timeframe": "1h", "version": "A",
        "allocation": 1, "status": "incubating"}}
    config_path.write_text(json.dumps(config, indent=2))

    ledger_path = tmp_path / "incubator_ledger.json"
    ledger_path.write_text(json.dumps({"version": "1.0.0", "strategies": {}}),
                           encoding="utf-8")

    logs = tmp_path / "logs"
    write_exec_log(logs, "exec.csv", [
        fill("2026-08-01 14:30:00", "Buy", 1, 20000.0),
        fill("2026-08-01 15:30:00", "Sell", 1, 20010.0)])

    config_loader.clear_cache()
    yield config_path, ledger_path, logs
    config_loader.clear_cache()


def test_cli_writes_nothing_without_the_flag(cli_workspace) -> None:
    """The recoverable mode is the one you get by forgetting a flag."""
    config_path, ledger_path, logs = cli_workspace
    before = ledger_path.read_text()
    proc = _run_cli("--logs", str(logs), "--config", str(config_path),
                    "--ledger", str(ledger_path))

    assert proc.returncode == 0, proc.stderr
    assert "NOTHING WAS WRITTEN" in proc.stdout
    assert "alpha" in proc.stdout
    assert ledger_path.read_text() == before


def test_cli_write_records_the_trades(cli_workspace) -> None:
    config_path, ledger_path, logs = cli_workspace
    proc = _run_cli("--logs", str(logs), "--config", str(config_path),
                    "--ledger", str(ledger_path), "--write")

    assert proc.returncode == 0, proc.stderr
    entry = json.loads(ledger_path.read_text())["strategies"]["alpha"]
    assert len(entry["trades"]) == 1
    assert entry["status"] == STATUS_INCUBATING


def test_cli_exits_1_when_the_export_directory_does_not_exist(
        cli_workspace, tmp_path: Path) -> None:
    """No feed is not no trades, and a cron job has to be told the
    difference."""
    config_path, ledger_path, _ = cli_workspace
    proc = _run_cli("--logs", str(tmp_path / "nowhere"),
                    "--config", str(config_path),
                    "--ledger", str(ledger_path))

    assert proc.returncode == 1
    assert "no incubator log directory" in proc.stderr


def test_cli_exits_2_when_rows_could_not_be_used(cli_workspace) -> None:
    """Ran, but dropped something. A run that exits 0 having silently lost a
    day of fills is how a strategy reaches its fourteenth session with nine
    trades on the board and nobody asking why."""
    config_path, ledger_path, logs = cli_workspace
    write_exec_log(logs, "orphan.csv", [
        fill("2026-08-02 14:30:00", "Buy", 1, 20000.0, account="SimNotOurs")])
    proc = _run_cli("--logs", str(logs), "--config", str(config_path),
                    "--ledger", str(ledger_path))

    assert proc.returncode == 2, proc.stdout
    assert "UNATTRIBUTED" in proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
