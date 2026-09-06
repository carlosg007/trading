"""
Tests for `realtime/contract_resolver.py`.

ASSERT-BASED. The collector-style suites in this tree record failures by
appending to a module-level list and only signal through `sys.exit(1)` in
`main()`, which bare pytest never calls, so they watch checks fail and report
green; `tests/conftest.py` routes those. This one uses plain asserts and needs
no routing.

WHAT THIS DEFENDS, and why each case is here rather than being obvious:

  * NinjaTrader refuses a bare root - "Instrument 'MNQ' not found" - and it
    refuses it AFTER CrossTrade has returned 200. The rejection lands
    downstream of everything this repository logs, so the dispatch record says
    `ok` on an order that never reached an account. Resolution being right is
    not checkable from the logs; it is only checkable here.
  * The two expiry guards fail in opposite directions from the same data.
    `valid_until` catches a table nobody has refreshed; `roll_date` catches
    the common case of a table refreshed for the index roll while the rates
    quietly rolled a fortnight earlier.
  * The boundary is END of the roll day, not the start. `fromisoformat` on a
    bare date returns MIDNIGHT from Python 3.11, which refused the rates a
    day early on the first run.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from realtime.contract_resolver import (                          # noqa: E402
    DEFAULT_CONTRACTS_PATH,
    ContractResolver,
    ContractResolverError,
    ContractTableExpiredError,
    UnknownSymbolError,
)

#: Inside the shipped table's `valid_until` and on or before EVERY
#: `roll_date` in it, so the mapping cases test the MAPPING rather than the
#: calendar. Pinned rather than "now" so this suite does not start failing on
#: a date nobody changed anything on.
#:
#: The earliest roll in the table is now the index complex at 2026-09-10,
#: which is what fixes this to the tenth. It was 2026-08-31 while the rates
#: complex still held that slot; when the rates were rolled to DEC26 this
#: moved with them, and the ZB expectation below moved from SEP26 to DEC26.
#: That coupling is the point: this fixture tracks the table, and when the two
#: disagree it is the fixture that is wrong.
#:
#: Note it also sits one day inside the table's own `valid_until`
#: (2026-09-10T23:59:59Z). Both expiries have to be refreshed together at the
#: index roll.
IN_WINDOW = "2026-09-10T12:00:00Z"


@pytest.fixture(scope="module")
def resolver() -> ContractResolver:
    return ContractResolver(DEFAULT_CONTRACTS_PATH)


def _table(tmp_path: Path, contracts: dict, valid_until: str) -> Path:
    p = tmp_path / "contracts.json"
    p.write_text(json.dumps({"version": 1, "valid_until": valid_until,
                             "contracts": contracts}))
    return p


# ---------------------------------------------------------------------------
# Mapping, across the families that roll differently
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("root,expected", [
    ("MES", "MES SEP26"),     # index micro, quarterly
    ("MNQ", "MNQ SEP26"),
    ("MGC", "MGC DEC26"),     # gold SKIPS Oct - the reason this is a table
    ("6E", "6E SEP26"),       # FX, quarterly
    ("6J", "6J SEP26"),
    ("HO", "HO OCT26"),       # energy, monthly
    ("ZB", "ZB DEC26"),       # rates, quarterly - rolled off SEP26 2026-09-06
    ("ZC", "ZC DEC26"),       # grains, own cycle
])
def test_roots_map_to_their_active_contract(resolver, root, expected):
    assert resolver.resolve_contract(root, current_time=IN_WINDOW) == expected


def test_gold_is_why_this_is_a_table_and_not_a_rule(resolver):
    """GC lists Feb/Apr/Jun/Aug/Oct/Dec and its liquidity SKIPS months.

    In August 2026 the active contract is DEC26, not the nearer OCT26. A rule
    taking "the next listed month" would route gold into a thin contract and
    look correct doing it.
    """
    assert resolver.resolve_contract("MGC", current_time=IN_WINDOW) \
        == "MGC DEC26"
    assert "OCT" not in resolver.resolve_contract("GC", current_time=IN_WINDOW)


def test_resolution_is_case_insensitive(resolver):
    assert (resolver.resolve_contract("mnq", current_time=IN_WINDOW)
            == resolver.resolve_contract("MNQ", current_time=IN_WINDOW))


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("explicit", [
    "MES SEP26",    # the month spelled out
    "MNQ 09-26",    # NinjaTrader's numeric form
    "MES 1!",       # continuous
    "MGC DEC26",
    "ZS NOV26",
])
def test_an_explicit_contract_is_returned_verbatim(resolver, explicit):
    assert resolver.resolve_contract(explicit, current_time=IN_WINDOW) \
        == explicit


def test_resolving_twice_changes_nothing(resolver):
    once = resolver.resolve_contract("MNQ", current_time=IN_WINDOW)
    assert resolver.resolve_contract(once, current_time=IN_WINDOW) == once


def test_an_explicit_contract_survives_an_expired_table(tmp_path):
    """The one path that must keep working when the table is stale.

    A caller who names the contract has taken the decision the table exists to
    make. Refusing them would break `send_test_probe.py`, which is how an
    operator finds out what NinjaTrader wants in the first place.
    """
    p = _table(tmp_path, {"MES": {"active_contract": "MES SEP26"}},
               valid_until="2020-01-01T00:00:00Z")
    r = ContractResolver(p)
    assert r.resolve_contract("MES SEP26") == "MES SEP26"
    with pytest.raises(ContractTableExpiredError):
        r.resolve_contract("MES")


# ---------------------------------------------------------------------------
# The two expiry guards
# ---------------------------------------------------------------------------
def test_the_whole_table_expires(tmp_path):
    p = _table(tmp_path, {"MES": {"active_contract": "MES SEP26"}},
               valid_until="2026-09-10T23:59:59Z")
    r = ContractResolver(p)
    assert r.resolve_contract("MES", current_time="2026-09-10T23:00:00Z")
    with pytest.raises(ContractTableExpiredError, match="expired"):
        r.resolve_contract("MES", current_time="2026-09-11T00:00:01Z")


def test_one_symbol_can_roll_while_the_table_is_still_valid(tmp_path):
    """The case that actually bites: a table refreshed for the index roll
    while the rates rolled a fortnight earlier."""
    p = _table(tmp_path, {
        "MES": {"active_contract": "MES SEP26", "next_contract": "MES DEC26",
                "roll_date": "2026-09-10"},
        "ZB": {"active_contract": "ZB SEP26", "next_contract": "ZB DEC26",
               "roll_date": "2026-08-31"},
    }, valid_until="2026-09-10T23:59:59Z")
    r = ContractResolver(p)
    when = "2026-09-05T12:00:00Z"
    assert r.resolve_contract("MES", current_time=when) == "MES SEP26"
    with pytest.raises(ContractTableExpiredError, match="rolled"):
        r.resolve_contract("ZB", current_time=when)


def test_a_rolled_symbol_names_its_successor_and_refuses_to_substitute(tmp_path):
    """It says what the next contract is and does NOT return it.

    `config/contracts.json` is the record of what Market Analyzer shows, and
    this module is not entitled to promote a guess into that record.
    """
    p = _table(tmp_path, {
        "ZB": {"active_contract": "ZB SEP26", "next_contract": "ZB DEC26",
               "roll_date": "2026-08-31"}},
        valid_until="2026-12-31T23:59:59Z")
    r = ContractResolver(p)
    with pytest.raises(ContractTableExpiredError) as caught:
        r.resolve_contract("ZB", current_time="2026-09-01T00:00:01Z")
    assert "ZB DEC26" in str(caught.value)


def test_the_roll_boundary_is_the_end_of_the_day_not_the_start(tmp_path):
    """THE REGRESSION THIS PINS.

    From Python 3.11 `fromisoformat("2026-08-31")` SUCCEEDS and returns
    MIDNIGHT, so a try/except never reaches the end-of-day branch and a
    contract with roll_date 2026-08-31 is refused from 00:00 on the
    thirty-first. Measured: the rates rolled a day early on the first run.
    """
    p = _table(tmp_path, {
        "ZB": {"active_contract": "ZB SEP26", "next_contract": "ZB DEC26",
               "roll_date": "2026-08-31"}},
        valid_until="2026-12-31T23:59:59Z")
    r = ContractResolver(p)
    for when in ("2026-08-31T00:00:01Z", "2026-08-31T12:00:00Z",
                 "2026-08-31T23:59:00Z"):
        assert r.resolve_contract("ZB", current_time=when) == "ZB SEP26", when
    with pytest.raises(ContractTableExpiredError):
        r.resolve_contract("ZB", current_time="2026-09-01T00:00:01Z")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_an_unknown_symbol_is_refused(resolver):
    with pytest.raises(UnknownSymbolError, match="Unknown symbol"):
        resolver.resolve_contract("ZZZ", current_time=IN_WINDOW)


def test_cl_is_absent_on_purpose_and_fails_loudly(resolver):
    """There is no CL series on this box's NT8 feed and no basket carries it.

    An order for one should fail at the formatter rather than resolve into a
    contract nothing can trade.
    """
    for root in ("CL", "MCL"):
        with pytest.raises(UnknownSymbolError):
            resolver.resolve_contract(root, current_time=IN_WINDOW)


def test_an_empty_instrument_is_refused(resolver):
    for bad in ("", "   "):
        with pytest.raises(UnknownSymbolError):
            resolver.resolve_contract(bad, current_time=IN_WINDOW)


def test_a_missing_table_raises_rather_than_falling_back(tmp_path):
    """Every order names a contract month, so there is nothing to fall back
    to. A resolver that returned the bare root would send exactly what NT8
    refuses."""
    r = ContractResolver(tmp_path / "does_not_exist.json")
    with pytest.raises(ContractResolverError, match="cannot read"):
        r.resolve_contract("MES")


def test_a_table_without_valid_until_is_refused(tmp_path):
    p = tmp_path / "contracts.json"
    p.write_text(json.dumps({"contracts": {"MES": {}}}))
    with pytest.raises(ContractResolverError, match="valid_until"):
        ContractResolver(p).resolve_contract("MES")


# ---------------------------------------------------------------------------
# The mtime cache
# ---------------------------------------------------------------------------
def test_an_edit_on_disk_is_picked_up_without_a_restart(tmp_path):
    """The reason the table is a file at all.

    A roll happens on a Monday when nobody wants to be editing Python and
    redeploying a live service to fix an order being refused right now.
    """
    p = _table(tmp_path, {"MES": {"active_contract": "MES SEP26"}},
               valid_until="2026-12-31T23:59:59Z")
    r = ContractResolver(p)
    assert r.resolve_contract("MES", current_time=IN_WINDOW) == "MES SEP26"

    time.sleep(0.01)
    p.write_text(json.dumps({"version": 1,
                             "valid_until": "2026-12-31T23:59:59Z",
                             "contracts": {
                                 "MES": {"active_contract": "MES DEC26"}}}))
    assert r.resolve_contract("MES", current_time=IN_WINDOW) == "MES DEC26", (
        "the edit was not picked up - the mtime cache is stale")


def test_the_shipped_table_covers_every_symbol_the_live_baskets_hold():
    """A symbol a basket can hold but the table cannot map is an order refused
    at the formatter, so the two files have to agree."""
    cfg = json.loads((REPO / "config" / "portfolios.json").read_text())
    held = {a for p in cfg["portfolios"].values()
            for a in (p.get("basket") or {}).get("assets", [])}
    table = json.loads(DEFAULT_CONTRACTS_PATH.read_text())["contracts"]
    missing = sorted(held - set(table))
    assert not missing, (
        f"config/portfolios.json routes {missing} but config/contracts.json "
        f"has no contract month for them")


def test_describe_names_the_symbols_past_their_roll_date(resolver):
    line = resolver.describe()
    assert "contract table:" in line
    assert "valid_until" in line
