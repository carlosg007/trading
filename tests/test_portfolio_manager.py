#!/usr/bin/env python3
"""
test_portfolio_manager.py — the portfolio engine: that ATR sizing produces the
contract counts the four point values imply, that opposing signals cancel and
reinforcing ones stack, that each basket's orders reach its own account, and
that a basket outside its regime stands down instead of trading.

Location:  ~/src/trading/tests/test_portfolio_manager.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_portfolio_manager.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_portfolio_manager.py

EVERY CASE FAILS THROUGH `assert`, DELIBERATELY — pytest never calls a `main()`
that exits non-zero, so a suite built on that convention reports green while its
checks fail.

Nothing here needs the lake or a network. No order is dispatched: the payloads
are built and inspected, never sent.

WHAT THIS COVERS, and why each one is here rather than assumed:

  * **THE SIZING MATH, AGAINST HAND-WORKED NUMBERS.** Every expected contract
    count below is computed on paper from the point value and the ATR, not from
    the module. A sizer compared only against itself is pinned, not verified —
    and the failure mode is silent: a wrong multiplier gives a well-formed
    integer and an order that fills normally.
  * **THE FLOOR, NOT ROUNDING.** 2.5 contracts is 2. Rounding up overshoots the
    budget by 25% in the same direction every time, invisibly.
  * **THE FLOOR CLAMP BREACHING THE BUDGET, AND SAYING SO.** One MCL contract
    at ATR 3.00 risks $300 against a $250 budget, and the sizer cannot express
    0.83 contracts. The clamp stands and `budget_breached` reports it; a limit
    that is silently exceeded is worse than one that is not enforced.
  * **NETTING AS A PARTITION.** Opposing signals cancel to zero, reinforcing
    ones stack, and a symbol routed to a basket that does not hold it RAISES —
    the point of a basket is that it names what may be held there.
  * **ROUTING BY BASKET AND BY TRACK.** MNQ/MCL to Odd, MES/MGC to Even, on
    whichever of the incubator and prop tracks the signal named. The `account`
    on the payload is what decides where money moves.
  * **THE REGIME GATE.** A symbol whose quadrant is not the portfolio's
    produces NO order. Outside its quadrant the basket is a strategy in the
    environment nobody certified it for.
  * **THE PAYLOAD IS THE DISPATCHER'S.** Checked through
    `live.dispatcher.format_crosstrade_payload`'s own validation, so a payload
    this suite accepts is one the dispatcher would send.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.dispatcher import (VALID_ACTIONS,                      # noqa: E402
                             VALID_ORDER_TYPES)
from portfolio.config_loader import (PortfolioConfigError,       # noqa: E402
                                     clear_cache,
                                     load_portfolio_config)
from portfolio.portfolio_manager import (FLAT, LONG,             # noqa: E402
                                         PortfolioError,
                                         PortfolioManager, SHORT)
from portfolio.volatility_sizer import (SizingError,             # noqa: E402
                                        calculate_position_size,
                                        size_detail)

# The point values, retyped from the specification rather than read from the
# config. This is the one place the suite compares the SIZER against the
# numbers a human signed off; reading them out of the file under test would
# pass whatever they were changed to.
POINT_VALUE = {"MNQ": 2.0, "MES": 5.0, "MCL": 100.0, "MGC": 10.0}

ODD_ASSETS = ("MNQ", "MCL")
EVEN_ASSETS = ("MES", "MGC")
# The quadrants each basket is configured to trade, as this repository's ids.
ODD_QUADRANTS = ("Q1", "Q2", "Q3", "Q4")   # ALL FOUR since 2026-08-24
EVEN_QUADRANTS = ("Q1", "Q2")       # the two HIGH-volatility quadrants


def manager() -> PortfolioManager:
    clear_cache()
    return PortfolioManager()


def narrowed_manager(pid: str, quadrants: list[str]) -> PortfolioManager:
    """
    A manager on a temp config where ONE portfolio forbids a quadrant.

    The regime gate can only be tested against an account that declines
    something. Every portfolio in `config/portfolios.json` declares all four
    quadrants since 2026-09-02 - Odd was widened on 2026-08-24 and Even
    followed so a Q3-certified ES or GC could be routed at all - so a case
    reading the live table has nothing left to stand down and would pass while
    testing nothing.

    The gate itself is unchanged and is what this pins: `build_order_plan`
    reads `basket.regime_quadrants` and nothing else, so an account declaring
    all four permits every strategy on it everywhere. That is a real property
    of the current config and the reason this case has to keep being tested on
    a fixture rather than on the file.
    """
    blob = json.loads(REPO.joinpath("config", "portfolios.json")
                      .read_text(encoding="utf-8"))
    blob["portfolios"][pid]["basket"]["regime_quadrants"] = list(quadrants)
    path = Path(tempfile.mkdtemp(prefix="pm_narrow_")) / "portfolios.json"
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    clear_cache()
    return PortfolioManager(str(path))


def regimes(**overrides) -> dict:
    """
    A regime reading per symbol, each in a quadrant its own basket permits, so
    a case that is not about the gate is never blocked by it.

    The ATRs are chosen to make the hand arithmetic clean: every one of them
    puts the risk per contract at a round $50 or $100.
    """
    base = {
        "MNQ": {"atr_14": 50.0, "quadrant": "Q3"},    # 50 x $2   = $100/ct
        "MCL": {"atr_14": 1.00, "quadrant": "Q3"},    # 1 x $100  = $100/ct
        "MES": {"atr_14": 10.0, "quadrant": "Q1"},    # 10 x $5   = $50/ct
        "MGC": {"atr_14": 5.00, "quadrant": "Q1"},    # 5 x $10   = $50/ct
    }
    base.update(overrides)
    return base


def signal(symbol: str, direction: str, portfolio_id: str | None = None,
           **extra) -> dict:
    out = {"symbol": symbol, "direction": direction, **extra}
    if portfolio_id:
        out["portfolio_id"] = portfolio_id
    return out


def raises(exc, fn, *args, **kwargs) -> str:
    try:
        fn(*args, **kwargs)
    except exc as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


# ==========================================================================
# 1. Volatility sizing — the request's first testing clause
# ==========================================================================
def test_the_sizing_math_matches_the_four_point_values() -> None:
    """
    THE REQUEST'S FIRST CLAUSE: MNQ $2, MES $5, MCL $100, MGC $10.

    Every expectation is worked on paper:

        contracts = floor( risk_budget / (ATR x point_value) )

    A $250 budget against a $100-per-contract risk is 2.5, which FLOORS to 2 —
    not 3. Rounding up overshoots the budget by 25% on every such signal, in
    the same direction every time, and the overshoot is invisible: the order is
    well formed and the fill is normal.
    """
    cases = [
        # symbol, atr, budget, expected contracts, expected $/contract
        ("MNQ", 50.0, 250.0, 2, 100.0),     # 250/100 = 2.5  -> 2
        ("MNQ", 25.0, 250.0, 5, 50.0),      # 250/50  = 5.0  -> 5
        ("MES", 10.0, 250.0, 5, 50.0),      # 250/50  = 5.0  -> 5
        ("MES", 20.0, 250.0, 2, 100.0),     # 250/100 = 2.5  -> 2
        ("MCL", 1.00, 250.0, 2, 100.0),     # 250/100 = 2.5  -> 2
        ("MCL", 0.50, 250.0, 5, 50.0),      # 250/50  = 5.0  -> 5
        ("MGC", 5.00, 250.0, 5, 50.0),      # 250/50  = 5.0  -> 5
        ("MGC", 12.5, 250.0, 2, 125.0),     # 250/125 = 2.0  -> 2
    ]
    for symbol, atr, budget, want, want_risk in cases:
        detail = size_detail(symbol, atr, risk_budget_usd=budget)
        assert detail["point_value"] == POINT_VALUE[symbol], (
            f"{symbol}: point value is {detail['point_value']}, the "
            f"specification says {POINT_VALUE[symbol]}")
        assert detail["risk_per_contract_usd"] == want_risk, (
            f"{symbol} ATR {atr}: ${detail['risk_per_contract_usd']}/contract, "
            f"hand-worked {want_risk}")
        got = calculate_position_size(symbol, atr, risk_budget_usd=budget)
        assert got == want == detail["contracts"], (
            f"{symbol} ATR {atr} budget {budget}: {got} contracts, "
            f"hand-worked {want} (raw {detail['raw_contracts']})")
        assert isinstance(got, int) and not isinstance(got, bool)


def test_a_fractional_size_floors_rather_than_rounding() -> None:
    """
    2.5 -> 2 and 2.99 -> 2. The budget is a CEILING on intended risk, so the
    error the sizer makes is always downward — recoverable, and the one a
    reader would rather find.
    """
    # 250 / (60 x 2) = 2.083
    assert calculate_position_size("MNQ", 60.0) == 2
    # 250 / (50.5 x 2) = 2.475
    assert calculate_position_size("MNQ", 50.5) == 2
    # 250 / (41.7 x 2) = 2.998 — floors to 2, and rounding would say 3
    d = size_detail("MNQ", 41.7)
    assert 2.99 < d["raw_contracts"] < 3.0, d["raw_contracts"]
    assert d["contracts"] == 2, d
    assert d["risk_usd"] <= d["risk_budget_usd"], d


def test_the_clamps_bound_the_size_at_both_ends() -> None:
    """
    The ceiling caps conviction; the floor keeps the smallest tradeable
    position tradeable. Both are recorded, so a size that was clamped is never
    mistaken for one the arithmetic produced.
    """
    # A tiny ATR would buy hundreds of contracts: clamped to 5.
    big = size_detail("MNQ", 0.5)
    assert big["contracts"] == 5, big
    assert big["clamp"] == "ceiling", big
    assert big["raw_contracts"] > 5

    # A huge ATR would buy less than one: clamped UP to 1.
    small = size_detail("MCL", 10.0)
    assert small["contracts"] == 1, small
    assert small["clamp"] == "floor", small
    assert small["raw_contracts"] < 1

    exact = size_detail("MES", 10.0)
    assert exact["clamp"] == "none" and exact["contracts"] == 5

    # And the clamps are honoured when the caller moves them.
    assert calculate_position_size("MNQ", 0.5, max_contracts=3) == 3
    assert calculate_position_size("MCL", 10.0, min_contracts=2) == 2


def test_the_floor_clamp_breaches_the_budget_and_reports_it() -> None:
    """
    ONE MCL CONTRACT AT ATR 3.00 RISKS $300 AGAINST A $250 BUDGET, and the
    sizer cannot express 0.83 contracts. The clamp stands — refusing the trade
    outright is a different strategy from the one configured — so what matters
    is that the breach is on the record rather than rounded away.

    A limit that is silently exceeded is worse than one that is not enforced:
    the account carries more risk than the configuration says, the order is
    well formed, and nothing anywhere reads differently.
    """
    d = size_detail("MCL", 3.0, risk_budget_usd=250.0)
    assert d["contracts"] == 1, d
    assert d["clamp"] == "floor"
    assert d["risk_per_contract_usd"] == 300.0
    assert d["risk_usd"] == 300.0
    assert d["budget_breached"] is True, d

    # And it is NOT set when the size fits inside the budget.
    ok = size_detail("MCL", 1.0, risk_budget_usd=250.0)
    assert ok["budget_breached"] is False, ok
    assert ok["risk_usd"] == 200.0          # 2 contracts x $100

    # min_contracts=0 is the way to decline instead of breaching.
    stood_down = size_detail("MCL", 3.0, min_contracts=0)
    assert stood_down["contracts"] == 0 and not stood_down["budget_breached"]


def test_the_stop_multiplier_scales_the_risk_per_contract() -> None:
    """
    The default sizes as though the stop sits ONE ATR from the fill. Every
    strategy here declares its own `sl_atr_mult`, and the two must agree or the
    position is sized for a stop the strategy will not use — at 1.5x against a
    default of 1.0, the realised loss on a stop-out is 1.5 times the budget.
    """
    one = size_detail("MNQ", 50.0)
    one_five = size_detail("MNQ", 50.0, stop_atr_mult=1.5)
    assert one["risk_per_contract_usd"] == 100.0
    assert one_five["risk_per_contract_usd"] == 150.0
    assert one_five["stop_distance_pts"] == 75.0
    assert one_five["contracts"] == 1, one_five      # 250/150 = 1.67 -> 1
    assert one["contracts"] == 2


def test_a_degenerate_input_raises_rather_than_sizing() -> None:
    """
    Each of these returns a plausible integer if it is let through. The
    dangerous one is the ZERO ATR: it divides to infinity and clamps to
    `max_contracts`, which reads as maximum conviction on an instrument that
    has not moved — exactly what a halted contract or a dead overnight hour
    produces.
    """
    assert "must be > 0" in raises(SizingError, calculate_position_size,
                                   "MNQ", 0.0)
    assert "maximum conviction" in raises(SizingError,
                                          calculate_position_size, "MNQ", 0.0)
    assert "must be > 0" in raises(SizingError, calculate_position_size,
                                   "MNQ", -5.0)
    assert "NaN" in raises(SizingError, calculate_position_size,
                           "MNQ", float("nan"))
    assert "NaN" in raises(SizingError, calculate_position_size,
                           "MNQ", float("inf"))
    assert "must be > 0" in raises(SizingError, calculate_position_size,
                                   "MNQ", 50.0, 0.0)
    assert "above max_contracts" in raises(
        SizingError, calculate_position_size, "MNQ", 50.0, 250.0, 5, 2)
    assert "no asset_metadata" in raises(PortfolioConfigError,
                                         calculate_position_size, "NQ", 50.0)


# ==========================================================================
# 2. Signal netting — the request's second testing clause
# ==========================================================================
def test_opposing_signals_cancel_to_flat() -> None:
    """
    THE REQUEST'S SECOND CLAUSE, first half. A long and a short on the same
    symbol in the same basket net to zero — and the record keeps BOTH sides, so
    "two strategies disagreed" stays distinguishable from "nobody signalled".
    """
    pm = manager()
    net = pm.aggregate_signals([
        signal("MNQ", LONG, "Incubator-Odd", strategy_id="a"),
        signal("MNQ", SHORT, "Incubator-Odd", strategy_id="b"),
    ])
    record = net["Incubator-Odd"]["MNQ"]
    assert record["net_units"] == 0, record
    assert record["direction"] == FLAT
    assert record["long_units"] == 1 and record["short_units"] == 1
    assert record["signals"] == 2
    assert [c["strategy_id"] for c in record["contributors"]] == ["a", "b"]

    # Unequal opposition nets to the remainder, with the sign of the majority.
    net = pm.aggregate_signals([
        signal("MNQ", LONG, "Incubator-Odd"),
        signal("MNQ", LONG, "Incubator-Odd"),
        signal("MNQ", SHORT, "Incubator-Odd"),
    ])
    assert net["Incubator-Odd"]["MNQ"]["net_units"] == 1
    assert net["Incubator-Odd"]["MNQ"]["direction"] == LONG


def test_reinforcing_signals_stack() -> None:
    """
    THE REQUEST'S SECOND CLAUSE, second half. Two longs make +2, three make +3,
    and a weighted signal contributes its own units.
    """
    pm = manager()
    net = pm.aggregate_signals([
        signal("MCL", LONG, "Prop-Odd", strategy_id="a"),
        signal("MCL", LONG, "Prop-Odd", strategy_id="b"),
        signal("MCL", LONG, "Prop-Odd", strategy_id="c"),
    ])
    assert net["Prop-Odd"]["MCL"]["net_units"] == 3
    assert net["Prop-Odd"]["MCL"]["direction"] == LONG

    weighted = pm.aggregate_signals([
        signal("MCL", LONG, "Prop-Odd", units=2),
        signal("MCL", SHORT, "Prop-Odd", units=1),
    ])
    assert weighted["Prop-Odd"]["MCL"]["net_units"] == 1

    shorts = pm.aggregate_signals([
        signal("MGC", SHORT, "Prop-Even"),
        signal("MGC", SHORT, "Prop-Even"),
    ])
    assert shorts["Prop-Even"]["MGC"]["net_units"] == -2
    assert shorts["Prop-Even"]["MGC"]["direction"] == SHORT


def test_flat_signals_are_counted_and_contribute_nothing() -> None:
    """
    "Three strategies looked and none wanted a position" is a different fact
    from "nothing ran", and only one of them is a reason to check the feed.
    """
    pm = manager()
    net = pm.aggregate_signals([
        signal("MNQ", FLAT, "Incubator-Odd", strategy_id="a"),
        signal("MNQ", FLAT, "Incubator-Odd", strategy_id="b"),
    ])
    r = net["Incubator-Odd"]["MNQ"]
    assert r["net_units"] == 0 and r["direction"] == FLAT
    assert r["signals"] == 2 and len(r["contributors"]) == 2
    assert r["long_units"] == 0 and r["short_units"] == 0


def test_netting_is_per_portfolio_and_per_symbol() -> None:
    """
    The same symbol on two DIFFERENT portfolios is two positions, not one.
    Incubator-Odd and Prop-Odd hold the same basket on purpose — paper
    validation mirroring live evaluation — and netting them together would make
    one account's signal cancel the other's.
    """
    pm = manager()
    net = pm.aggregate_signals([
        signal("MNQ", LONG, "Incubator-Odd"),
        signal("MNQ", SHORT, "Prop-Odd"),
        signal("MCL", LONG, "Incubator-Odd"),
    ])
    assert net["Incubator-Odd"]["MNQ"]["net_units"] == 1
    assert net["Prop-Odd"]["MNQ"]["net_units"] == -1
    assert net["Incubator-Odd"]["MCL"]["net_units"] == 1
    assert set(net) == {"Incubator-Odd", "Prop-Odd"}
    assert set(net["Incubator-Odd"]) == {"MNQ", "MCL"}


def test_a_symbol_outside_the_basket_is_refused() -> None:
    """
    The point of a basket is that it names what may be held there. Netting MES
    onto the Odd book would put a position on an account that does not trade
    the contract, and the correlation partition the four baskets exist for
    would be fictional.
    """
    pm = manager()
    msg = raises(PortfolioError, pm.aggregate_signals,
                 [signal("MES", LONG, "Incubator-Odd")])
    assert "MES" in msg and "fictional" in msg, msg
    assert "no portfolio" not in msg

    assert "not in" in raises(PortfolioError, pm.aggregate_signals,
                              [signal("MNQ", LONG, "Prop-Sideways")])
    assert "expected one of" in raises(PortfolioError, pm.aggregate_signals,
                                       [signal("MNQ", "buy", "Prop-Odd")])
    assert "units must be >= 0" in raises(
        PortfolioError, pm.aggregate_signals,
        [signal("MNQ", LONG, "Prop-Odd", units=-1)])
    assert "neither portfolio_id nor strategy_id" in raises(
        PortfolioError, pm.aggregate_signals, [{"symbol": "MNQ",
                                                "direction": LONG}])


def test_an_unassigned_strategy_cannot_be_routed() -> None:
    """
    With no explicit `portfolio_id` the strategy must be named in some
    portfolio's `active_strategies`. There is no fallback: a portfolio inferred
    from a strategy's name would be a live account chosen by a rule nobody
    wrote down.
    """
    pm = manager()
    msg = raises(PortfolioError, pm.aggregate_signals,
                 [{"strategy_id": "nobody_assigned_me", "symbol": "MNQ",
                   "direction": LONG}])
    assert "not assigned to any incubator portfolio" in msg, msg

    # Assigned, and it routes — on the track the signal asked for.
    cfg = load_portfolio_config()
    cfg["portfolios"]["Incubator-Odd"]["active_strategies"].append("strat_x")
    cfg["portfolios"]["Prop-Odd"]["active_strategies"].append("strat_x")
    pm2 = PortfolioManager(config=cfg)
    inc = pm2.aggregate_signals([{"strategy_id": "strat_x", "symbol": "MNQ",
                                  "direction": LONG}])
    assert set(inc) == {"Incubator-Odd"}
    prop = pm2.aggregate_signals([{"strategy_id": "strat_x", "symbol": "MNQ",
                                   "direction": LONG,
                                   "is_incubating": False}])
    assert set(prop) == {"Prop-Odd"}


# ==========================================================================
# 3. Account routing — the request's third testing clause
# ==========================================================================
def test_each_basket_routes_to_its_own_account() -> None:
    """
    THE REQUEST'S THIRD CLAUSE. MNQ and MCL reach an Odd account; MES and MGC
    reach an Even one; and the `account` field on the payload is what decides
    where money actually moves.

    That field carries the portfolio's `target_account` — NinjaTrader's name
    for the account, which is not the portfolio id: NT8 prefixes a simulation
    account with `Sim`, so `Incubator-Odd` executes on `SimIncubator1`. The
    names are READ from the routing table here rather than retyped, because
    what this case is about is which BASKET reaches which account;
    `tests/test_portfolio_config.py` is where the four names are checked
    against the specification, and a second copy would fail there twice for
    one rename.
    """
    pm = manager()
    accounts = {pid: portfolio["target_account"]
                for pid, portfolio in load_portfolio_config()["portfolios"].items()}
    net = pm.aggregate_signals([
        signal("MNQ", LONG, "Incubator-Odd"),
        signal("MCL", LONG, "Prop-Odd"),
        signal("MES", SHORT, "Incubator-Even"),
        signal("MGC", LONG, "Prop-Even"),
    ])
    payloads = pm.build_order_payloads(net, regimes())
    routed = {p["symbol"]: p["account"] for p in payloads}
    assert routed == {"MNQ": accounts["Incubator-Odd"],
                      "MCL": accounts["Prop-Odd"],
                      "MES": accounts["Incubator-Even"],
                      "MGC": accounts["Prop-Even"]}, routed

    odd_accounts = {accounts["Incubator-Odd"], accounts["Prop-Odd"]}
    even_accounts = {accounts["Incubator-Even"], accounts["Prop-Even"]}
    for p in payloads:
        assert (p["symbol"] in ODD_ASSETS) == (p["account"] in odd_accounts), p
        assert (p["symbol"] in EVEN_ASSETS) == (p["account"] in even_accounts), p


def test_the_payload_is_the_dispatchers_own_shape() -> None:
    """
    Built by `live.dispatcher.format_crosstrade_payload`, which is the only
    place in this repository that formats an order — it validates the action,
    refuses a fractional or non-positive quantity, and rejects an order type
    that carries no price. A second formatter here would be free to disagree
    with it, and the disagreement would be discovered by a broker.

    Exactly five keys, so the result can be POSTed as it stands.
    """
    pm = manager()
    net = pm.aggregate_signals([signal("MNQ", LONG, "Incubator-Odd"),
                                signal("MES", SHORT, "Prop-Even")])
    payloads = pm.build_order_payloads(net, regimes())
    assert len(payloads) == 2
    for p in payloads:
        assert set(p) == {"account", "action", "symbol", "orderType",
                          "quantity"}, sorted(p)
        assert p["action"] in VALID_ACTIONS
        assert p["orderType"] in VALID_ORDER_TYPES
        assert isinstance(p["quantity"], int) and p["quantity"] > 0
        assert not isinstance(p["quantity"], bool)

    by_symbol = {p["symbol"]: p for p in payloads}
    assert by_symbol["MNQ"]["action"] == "BUY"
    assert by_symbol["MES"]["action"] == "SELL"
    # FLATTEN is never emitted — this manager does not know what is open.
    assert all(p["action"] != "FLATTEN" for p in payloads)


def test_conviction_scales_the_order_and_the_clamp_bounds_it() -> None:
    """
    Two strategies long MNQ net to +2, and the order is twice the unit size —
    which is what makes "reinforcing signals stack" observable in the order
    rather than only in a field nobody reads.

    THE COST IS REAL AND ON THE RECORD: two units risk twice the configured
    budget. `max_contracts` is what bounds it, and `budget_breached` is what
    says it happened.
    """
    pm = manager()
    one = pm.build_order_plan(
        pm.aggregate_signals([signal("MNQ", LONG, "Incubator-Odd")]),
        regimes())[0]
    two = pm.build_order_plan(
        pm.aggregate_signals([signal("MNQ", LONG, "Incubator-Odd"),
                              signal("MNQ", LONG, "Incubator-Odd")]),
        regimes())[0]

    assert one["payload"]["quantity"] == 2          # 250/(50x2) = 2.5 -> 2
    assert two["payload"]["quantity"] == 4          # 2 x 2 units
    assert two["sizing"]["unit_contracts"] == 2
    assert two["sizing"]["conviction"] == 2
    assert one["sizing"]["budget_breached"] is False
    assert two["sizing"]["budget_breached"] is True, two["sizing"]
    assert two["sizing"]["risk_usd"] == 400.0
    assert two["sizing"]["risk_budget_usd"] == 250.0

    # The ceiling bounds it: 3 units of a 2-contract unit size is 6, capped
    # at the configured max of 5.
    three = pm.build_order_plan(
        pm.aggregate_signals([signal("MNQ", LONG, "Incubator-Odd")] * 3),
        regimes())[0]
    assert three["payload"]["quantity"] == 5, three["sizing"]


# ==========================================================================
# 4. The regime gate and the declined orders
# ==========================================================================
def test_a_basket_outside_its_quadrant_stands_down() -> None:
    """
    Each portfolio declares the quadrants it trades, and outside them the
    basket is a strategy in the environment nobody certified it for - so it
    produces NO order rather than a smaller one.

    NARROWED AS A FIXTURE. Every portfolio in the live table declares all four
    quadrants - Odd from 2026-08-24 so a Nasdaq strategy certified in a
    HIGH-volatility quadrant could route at all, Even from 2026-09-02 so a
    Q3-certified ES or GC could - so there is no forbidden quadrant left in
    the file to stand anything down in, and a case reading it would pass while
    testing nothing. That widening is exactly why this has to keep being
    tested: the live gate reads `basket.regime_quadrants` and nothing else, so
    an account declaring all four permits every strategy on it everywhere.
    """
    pm = narrowed_manager("Prop-Even", ["Q1_HIGH_VOL_TREND",
                                        "Q2_HIGH_VOL_CHOP"])
    assert tuple(pm.portfolios["Incubator-Odd"]["derived"][
        "canonical_quadrants"]) == ODD_QUADRANTS
    assert tuple(pm.portfolios["Prop-Even"]["derived"][
        "canonical_quadrants"]) == EVEN_QUADRANTS

    net = pm.aggregate_signals([signal("MES", LONG, "Prop-Even")])
    # Q3 is a LOW-volatility quadrant; the Even basket trades Q1 and Q2.
    blocked = pm.build_order_plan(net, regimes(
        MES={"atr_14": 50.0, "quadrant": "Q3"}))
    assert blocked[0]["payload"] is None
    assert "Q3" in blocked[0]["skipped_reason"]
    assert "certified" in blocked[0]["skipped_reason"]
    assert pm.build_order_payloads(net, regimes(
        MES={"atr_14": 50.0, "quadrant": "Q3"})) == []

    # And it trades in a quadrant it does permit.
    allowed = pm.build_order_plan(net, regimes(
        MES={"atr_14": 50.0, "quadrant": "Q2"}))
    assert allowed[0]["payload"] is not None


def test_a_schema_label_resolves_the_same_as_the_canonical_id() -> None:
    """
    A caller holding Stage 1's handoff has `Q3`; one reading the portfolio
    config has `Q3_LOW_VOL_TREND`. Neither should have to know which the other
    used, and the translation is `canonical_quadrant` — the only supported one.
    """
    pm = manager()
    net = pm.aggregate_signals([signal("MNQ", LONG, "Incubator-Odd")])
    by_id = pm.build_order_payloads(net, regimes(
        MNQ={"atr_14": 50.0, "quadrant": "Q3"}))
    by_label = pm.build_order_payloads(net, regimes(
        MNQ={"atr_14": 50.0, "quadrant": "Q3_LOW_VOL_TREND"}))
    assert by_id == by_label and len(by_id) == 1

    assert "unknown regime label" in raises(
        PortfolioConfigError, pm.build_order_plan, net,
        regimes(MNQ={"atr_14": 50.0, "quadrant": "Q9_MADE_UP"}))


def test_a_missing_regime_or_atr_stands_down_with_its_own_reason() -> None:
    """
    An unknown environment is not a permitted one, and the ATR is what sizes
    the position. The two reasons are kept apart because they are fixed by
    different work: one is a gap in the regime feed, the other a market the
    basket is not for.
    """
    pm = manager()
    net = pm.aggregate_signals([signal("MNQ", LONG, "Incubator-Odd")])

    absent = pm.build_order_plan(net, {})[0]
    assert absent["payload"] is None
    assert "no regime reading" in absent["skipped_reason"]

    no_atr = pm.build_order_plan(net, {"MNQ": {"quadrant": "Q3"}})[0]
    assert no_atr["payload"] is None
    assert "no atr_14" in no_atr["skipped_reason"]

    assert "must be a dict" in raises(PortfolioError, pm.build_order_plan,
                                      net, {"MNQ": "Q3"})


def test_a_flat_net_produces_no_order_and_no_flatten() -> None:
    """
    A net that cancels to zero is a skip, not a close. This manager does not
    know whether anything is open, and a FLATTEN sent on that assumption would
    shut positions it never opened.
    """
    pm = manager()
    net = pm.aggregate_signals([signal("MCL", LONG, "Prop-Odd"),
                                signal("MCL", SHORT, "Prop-Odd")])
    plan = pm.build_order_plan(net, regimes())
    assert plan[0]["payload"] is None
    assert "netted flat" in plan[0]["skipped_reason"]
    assert "FLATTEN" in plan[0]["skipped_reason"]
    assert pm.build_order_payloads(net, regimes()) == []


def test_the_plan_records_what_the_payload_list_cannot() -> None:
    """
    An empty payload list cannot distinguish a basket standing down outside its
    regime from a day with no signals at all, and those are different facts
    about a trading day. The plan carries every decision, declined ones
    included.
    """
    pm = manager()
    # The regime-blocked leg is on the EVEN track: the Odd basket declares all
    # four quadrants since 2026-08-24, so nothing routed there can be stood
    # down on regime any more.
    net = pm.aggregate_signals([
        signal("MNQ", LONG, "Incubator-Odd"),     # trades
        signal("MCL", LONG, "Incubator-Odd"),     # trades
        signal("MES", LONG, "Prop-Even"),         # blocked by regime
        signal("MGC", LONG, "Prop-Even"),         # netted flat
        signal("MGC", SHORT, "Prop-Even"),
    ])
    reading = regimes(MES={"atr_14": 1.0, "quadrant": "Q3"})
    plan = pm.build_order_plan(net, reading)
    payloads = pm.build_order_payloads(net, reading)

    assert len(plan) == 4, [r["symbol"] for r in plan]
    assert len(payloads) == 2
    assert payloads == [r["payload"] for r in plan if r["payload"]]

    by_symbol = {r["symbol"]: r for r in plan}
    assert by_symbol["MNQ"]["payload"] is not None
    assert by_symbol["MCL"]["payload"] is not None
    assert "Q3" in by_symbol["MES"]["skipped_reason"]
    assert "netted flat" in by_symbol["MGC"]["skipped_reason"]

    text = pm.describe_plan(plan)
    for symbol in ("MNQ", "MCL", "MES", "MGC"):
        assert symbol in text, symbol
    assert "SKIP" in text and "BUY" in text


def test_the_manager_holds_a_private_copy_of_the_config() -> None:
    """
    A caller mutating a record it got back must not change where the next order
    is routed. The loader deep-copies on every access for the same reason.
    """
    pm = manager()
    net = pm.aggregate_signals([signal("MNQ", LONG, "Incubator-Odd")])
    net["Incubator-Odd"]["MNQ"]["net_units"] = 99
    again = pm.aggregate_signals([signal("MNQ", LONG, "Incubator-Odd")])
    assert again["Incubator-Odd"]["MNQ"]["net_units"] == 1

    plan = pm.build_order_plan(net, regimes())
    plan[0]["sizing"]["contracts"] = 999
    assert pm.build_order_plan(net, regimes())[0]["payload"][
        "quantity"] != 999


# ==========================================================================
# The script runner. `assert` is the failure mechanism, so pytest and this
# report the same thing.
# ==========================================================================
def main() -> int:
    cases = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    print(f"portfolio engine — {len(cases)} cases\n")
    for name, fn in cases:
        clear_cache()
        try:
            fn()
        except Exception as e:                   # noqa: BLE001 - reported below
            failures.append((name, e))
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
        else:
            print(f"  PASS  {name}")
    print()
    if failures:
        print(f"{len(failures)} of {len(cases)} FAILED: "
              + ", ".join(n for n, _ in failures))
        return 1
    print(f"all {len(cases)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
