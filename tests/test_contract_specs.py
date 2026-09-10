"""
tests/test_contract_specs.py - the contract specifications, against the exchange.

Location:  ~/src/trading/tests/test_contract_specs.py

    .venv/bin/python3 -m pytest tests/test_contract_specs.py

ASSERT-BASED, DELIBERATELY. `tests/conftest.py` classifies a suite by the
`\\ndef check(` marker: script-style suites are run as subprocesses asserting
an exit code. This suite has no `check()` helper, so bare pytest and the suite
runner report the same thing and per-case granularity survives.

WHY THIS FILE EXISTS. "Verified" was prose. CLAUDE.md's Known-gaps list named
PL, LE and crypto as UNVERIFIED in `backtest/specs.py`, and
`scripts/register_incubator_batch.py` repeated it - but neither statement was
checked by anything, and both had been overtaken: `python -m backtest.specs`
reconciles every one of them against the Databento definition files and
reports a problem for M2K and MYM alone.

A comment claiming a multiplier is right is worth nothing, because the failure
it guards against is silent. THE MULTIPLIER AND THE TICK ARE WHAT EVERY P&L
FIGURE FOR A CONTRACT IS BUILT FROM: get one wrong and the equity curve, the
Sharpe, the gate audit's net P&L and the routing decision that rests on it are
all wrong together, all internally consistent, and nothing raises. So the
numbers are RETYPED HERE FROM THE EXCHANGE SPECIFICATION rather than read out
of the file under test - the one place in this suite that compares the config
against the world instead of against itself.

The reconciler is NOT called here. It reads the definition files on the NFS
mount, which would make this suite environment-dependent and slow; it is the
other half of the check and belongs at `python -m backtest.specs`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.specs import (                                     # noqa: E402
    SPECS, TICK_HISTORY, ContractSpec,
)

# The exchange specification, retyped. `(multiplier, tick_size, tick_value)`.
#
#   ETH  CME Ether            50 ether/contract, $0.50/ether   -> $25.00/tick
#   LE   CME Live Cattle      40,000 lb quoted in cents/lb, so the multiplier
#                             is 400 dollars per full cent, 0.025c tick
#                                                              -> $10.00/tick
#   PL   NYMEX Platinum       50 troy oz, $0.10/oz             -> $5.00/tick
#
# Written as the ARITHMETIC rather than as three independent numbers, because
# `tick_value` is derived (`multiplier * tick_size`) and a table that restated
# it could agree with itself while disagreeing with the contract.
EXCHANGE = {
    "ETH": {"multiplier": 50.0, "tick_size": 0.50, "tick_value": 25.00,
            "exchange": "CME", "name": "Ether"},
    "LE":  {"multiplier": 400.0, "tick_size": 0.025, "tick_value": 10.00,
            "exchange": "CME", "name": "Live Cattle"},
    "PL":  {"multiplier": 50.0, "tick_size": 0.10, "tick_value": 5.00,
            "exchange": "NYMEX", "name": "Platinum"},
}

#: Every non-micro contract is charged the same way: `1.29 + 1.00` per side,
#: commission plus estimated exchange and clearing fees. Retyped so a contract
#: quietly given its own rate fails here rather than producing a cheaper
#: backtest than its neighbours.
NON_MICRO_COMMISSION = 2.29


@pytest.mark.parametrize("symbol", sorted(EXCHANGE))
def test_the_verified_contracts_match_the_exchange(symbol: str) -> None:
    """Multiplier and tick, against the specification rather than the file."""
    spec = SPECS[symbol]
    want = EXCHANGE[symbol]
    assert isinstance(spec, ContractSpec)
    assert spec.multiplier == want["multiplier"], (
        f"{symbol} multiplier {spec.multiplier} != {want['multiplier']} - "
        f"every P&L figure for this contract is built from it")
    assert spec.tick_size == want["tick_size"], symbol
    assert spec.exchange == want["exchange"], symbol
    assert spec.name == want["name"], symbol


@pytest.mark.parametrize("symbol", sorted(EXCHANGE))
def test_the_tick_value_is_the_one_the_exchange_publishes(symbol: str) -> None:
    """
    `tick_value` is DERIVED - `multiplier * tick_size` - and this is the case
    that catches a pair of errors that cancel. A multiplier ten times too
    small with a tick ten times too large passes both fields individually and
    prices every trade correctly, right up to the first slippage calculation.
    """
    spec = SPECS[symbol]
    assert spec.tick_value == pytest.approx(EXCHANGE[symbol]["tick_value"]), (
        f"{symbol} tick value ${spec.tick_value:.4f} != "
        f"${EXCHANGE[symbol]['tick_value']:.2f}")
    # One tick of slippage costs one tick, on one contract, on one side.
    assert spec.slippage_dollars(1) == pytest.approx(spec.tick_value)


@pytest.mark.parametrize("symbol", sorted(EXCHANGE))
def test_they_are_charged_like_every_other_non_micro(symbol: str) -> None:
    """A contract quietly given its own rate produces a cheaper backtest than
    its neighbours, and the ranking between them stops meaning anything."""
    spec = SPECS[symbol]
    assert spec.is_micro is False, symbol
    assert spec.commission == pytest.approx(NON_MICRO_COMMISSION), symbol
    assert spec.round_turn_cost == pytest.approx(NON_MICRO_COMMISSION * 2)


def test_the_three_are_charged_the_same_as_a_known_verified_peer() -> None:
    """
    Cross-checked against GC, which has never been in anybody's UNVERIFIED
    list. Costs are mandatory in every test from the first one, and a contract
    whose costs differ from its peers' for no stated reason is the kind of
    difference that shows up as alpha.
    """
    peer = SPECS["GC"]
    for symbol in EXCHANGE:
        assert SPECS[symbol].commission == peer.commission, symbol


def test_eth_keeps_the_tick_that_was_in_force_on_the_bar() -> None:
    """
    ETH's tick is NOT 0.50 for the whole history: it was 0.25 until
    2021-12-06. `tick_size` is the CURRENT tick, and `TICK_HISTORY` carries
    the rest.

    Pinned because "ETH tick_size = 0.50" is true and, applied to a 2021 bar,
    wrong - slippage on every pre-December-2021 ETH trade would be double what
    the market charged, on a contract this repository has four routed packages
    on.
    """
    history = TICK_HISTORY.get("ETH")
    assert history, "ETH lost its tick history; pre-2021 slippage is now wrong"
    assert ("2021-12-06", 0.50) in [(d, float(t)) for d, t in history]
    assert ("2021-02-08", 0.25) in [(d, float(t)) for d, t in history]
    # The current tick agrees with the last entry in the history.
    assert SPECS["ETH"].tick_size == pytest.approx(float(history[-1][1]))


def test_a_micro_ether_is_deliberately_absent() -> None:
    """
    CME lists Micro Ether (MET, 0.1 ether). It is NOT in SPECS, and adding it
    without pulling its definition file would create exactly the state this
    file exists to close: a contract whose numbers nothing has checked, which
    `python -m backtest.specs` would report UNVERIFIED beside M2K and MYM.

    The micros are documented as position-sizing entries only - the backtest
    runs on the full-size symbol and scales - so nothing is currently blocked
    by its absence. This case is the record of that decision; delete it in the
    commit that adds MET and its definition data together.
    """
    assert "MET" not in SPECS
    from realtime.contract_alias import MICRO_TO_PARENT     # noqa: PLC0415
    assert "MET" not in MICRO_TO_PARENT, (
        "MET routes to a parent but has no ContractSpec, so a basket holding "
        "it would size against a multiplier that does not exist")


def test_every_micro_that_routes_has_a_spec_to_size_against() -> None:
    """
    The general form of the case above. `basket_index` resolves a micro in a
    basket to its parent for routing, and the live sizer needs the MICRO's own
    multiplier - so a micro that can be routed and cannot be priced is a
    position sized against nothing.
    """
    from realtime.contract_alias import MICRO_TO_PARENT     # noqa: PLC0415
    for micro, parent in MICRO_TO_PARENT.items():
        assert micro in SPECS, f"{micro} routes but has no ContractSpec"
        assert parent in SPECS, f"{micro} -> {parent}, which has no spec"
        assert SPECS[micro].is_micro is True, micro
        assert SPECS[parent].is_micro is False, parent
