"""
ATR position sizing: how many contracts a fixed dollar risk budget buys, given
what the instrument is currently moving.

Location:  ~/src/trading/portfolio/volatility_sizer.py

THE ARITHMETIC, AND THE ONE THING IT DEPENDS ON
===============================================
    risk per contract  =  atr_14 x stop_atr_mult x point_value
    contracts          =  floor( risk_budget_usd / risk per contract )
    size               =  clamp(contracts, min_contracts, max_contracts)

`point_value` is what turns a price move into dollars, and it is looked up
through `portfolio.config_loader`, which has already reconciled it against
`backtest/specs.py`. That reconciliation is the whole reason this module does
not take a multiplier as an argument: a wrong one does not raise, it silently
scales every position for that symbol, and the backtest it is compared against
still looks plausible.

The four contracts in the baskets, for reading the numbers below by eye:

    MNQ   $2/pt    tick 0.25   ->  $0.50 a tick
    MES   $5/pt    tick 0.25   ->  $1.25 a tick
    MCL   $100/pt  tick 0.01   ->  $1.00 a tick
    MGC   $10/pt   tick 0.10   ->  $1.00 a tick

FLOOR, NOT ROUND
================
`floor`, so the budget is a CEILING on intended risk rather than a midpoint.
Rounding 1.6 contracts up to 2 overshoots the budget by 25% on every such
signal, in the same direction every time, and the overshoot is invisible: the
order is well formed, the fill is normal, and the account simply carries more
risk than the configuration says it does. Rounding down costs opportunity,
which is recoverable, and is the error a reader would rather find.

THE FLOOR CLAMP BREACHES THE BUDGET, AND THAT IS REPORTED RATHER THAN HIDDEN
===========================================================================
`min_contracts=1` means the smallest tradeable position is one contract even
when one contract risks MORE than the budget. On MCL at $100 a point, an ATR of
3.00 puts a single contract's 1x-ATR risk at $300 against a $250 budget — the
sizer cannot express "0.83 contracts" and refusing the trade outright is a
different strategy from the one that was configured.

So the clamp stands, and `size_detail` reports `budget_breached` with the
actual dollar risk beside the budgeted one. A number that silently exceeds its
own stated limit is exactly the kind of thing this repository writes down
rather than rounds away. `min_contracts=0` is accepted for a caller that would
rather stand the trade down than breach — it is not the default, because a
sizer that can return zero makes "no signal" and "signal too big" the same
observation downstream.

WHAT THIS IS NOT
================
NOT a risk manager. It converts a budget into a contract count and stops there.
Drawdown lockouts, daily loss caps and the prop-firm rulebook are CrossTrade
NAM's, enforced against a live account balance this process cannot see — see
`portfolio/config_loader.py` for the separation. Nothing here reads a balance,
a position or a P&L.

NOT a stop-distance decision either. `stop_atr_mult` defaults to 1.0, which
sizes as though the stop sits one ATR from the fill. EVERY STRATEGY IN THIS
REPOSITORY DECLARES ITS OWN `sl_atr_mult`, and the two must agree or the
position is sized for a stop the strategy will not use: at `sl_atr_mult=1.5`
against the default 1.0 here, the realised loss on a stop-out is 1.5x the
budget. Pass the strategy's own multiplier.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.config_loader import (DEFAULT_CONFIG_PATH,        # noqa: E402
                                     PortfolioConfigError, get_asset_spec)

DEFAULT_RISK_BUDGET_USD = 250.0
DEFAULT_MIN_CONTRACTS = 1
DEFAULT_MAX_CONTRACTS = 5


class SizingError(ValueError):
    """A position size that cannot be computed from the inputs given."""


def _validate(symbol: str, atr_14: float, risk_budget_usd: float,
              min_contracts: int, max_contracts: int,
              stop_atr_mult: float) -> None:
    """
    Refuse inputs that cannot produce a meaningful size.

    Every one of these would otherwise return a plausible integer. A NaN ATR
    makes the division NaN and `int(NaN)` raises somewhere further away; a zero
    ATR makes it infinite, which clamps to `max_contracts` and reads as maximum
    conviction on an instrument that has not moved at all. That second one is
    the dangerous one, and it is exactly what a halted contract or a dead
    overnight hour produces.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise SizingError(f"symbol must be a non-empty string; got {symbol!r}")

    for name, value in (("atr_14", atr_14),
                        ("risk_budget_usd", risk_budget_usd),
                        ("stop_atr_mult", stop_atr_mult)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SizingError(f"{name} must be a number; got {value!r}")
        if not math.isfinite(float(value)):
            raise SizingError(
                f"{name} is {value!r}. A NaN or infinite {name} produces a "
                f"size that is either an exception several frames away or a "
                f"clamp to the maximum — and a zero-volatility instrument "
                f"clamping to maximum conviction is what a halted contract "
                f"looks like.")
        if float(value) <= 0:
            raise SizingError(
                f"{name} must be > 0; got {value!r}. "
                + ("A zero ATR divides to infinity and clamps to "
                   "max_contracts, which reads as maximum conviction on an "
                   "instrument that did not move."
                   if name == "atr_14" else
                   "A zero budget sizes every position at the floor, which is "
                   "not the same statement as declining to trade."))

    for name, value in (("min_contracts", min_contracts),
                        ("max_contracts", max_contracts)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SizingError(
                f"{name} must be a whole number of contracts; got {value!r}")
        if value < 0:
            raise SizingError(f"{name} must be >= 0; got {value!r}")
    if min_contracts > max_contracts:
        raise SizingError(
            f"min_contracts ({min_contracts}) is above max_contracts "
            f"({max_contracts}) — no contract count satisfies both.")
    if max_contracts == 0:
        raise SizingError(
            "max_contracts is 0, so every position is zero. Decline the "
            "signal rather than sizing it to nothing: the two are different "
            "statements and only one of them is visible downstream.")


def size_detail(symbol: str,
                atr_14: float,
                risk_budget_usd: float = DEFAULT_RISK_BUDGET_USD,
                min_contracts: int = DEFAULT_MIN_CONTRACTS,
                max_contracts: int = DEFAULT_MAX_CONTRACTS,
                stop_atr_mult: float = 1.0,
                config: dict | None = None,
                config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    The full sizing record: the contract count AND everything it was derived
    from.

    `calculate_position_size` returns the integer alone, which is what an order
    needs. This returns the reasoning, which is what a reader needs when the
    integer is surprising — and, more importantly, `budget_breached`, the one
    fact the integer cannot carry.

    Keys:

        contracts            the clamped, floored size — what to trade
        raw_contracts        the unclamped, unfloored quotient
        clamp                "floor", "ceiling" or "none"
        point_value          dollars per point, from the reconciled metadata
        stop_distance_pts    atr_14 x stop_atr_mult
        risk_per_contract_usd
        risk_budget_usd      what was asked for
        risk_usd             what `contracts` actually risks
        budget_breached      risk_usd > risk_budget_usd, which the FLOOR clamp
                             can cause and nothing else can
    """
    _validate(symbol, atr_14, risk_budget_usd, min_contracts, max_contracts,
              stop_atr_mult)
    spec = get_asset_spec(symbol, config=config, config_path=config_path)
    point_value = float(spec["point_value"])

    stop_distance = float(atr_14) * float(stop_atr_mult)
    risk_per_contract = stop_distance * point_value
    raw = float(risk_budget_usd) / risk_per_contract

    floored = int(math.floor(raw))
    contracts = max(min_contracts, min(max_contracts, floored))
    if contracts > floored:
        clamp = "floor"
    elif contracts < floored:
        clamp = "ceiling"
    else:
        clamp = "none"

    risk_usd = contracts * risk_per_contract
    return {
        "symbol": spec["symbol"],
        "contracts": int(contracts),
        "raw_contracts": raw,
        "floored_contracts": floored,
        "clamp": clamp,
        "min_contracts": int(min_contracts),
        "max_contracts": int(max_contracts),
        "point_value": point_value,
        "tick_size": float(spec["tick_size"]),
        "atr_14": float(atr_14),
        "stop_atr_mult": float(stop_atr_mult),
        "stop_distance_pts": stop_distance,
        "risk_per_contract_usd": risk_per_contract,
        "risk_budget_usd": float(risk_budget_usd),
        "risk_usd": risk_usd,
        # Only the FLOOR clamp can cause this: flooring the quotient can only
        # reduce risk, and the ceiling clamp only reduces it further.
        "budget_breached": bool(risk_usd > float(risk_budget_usd) + 1e-9),
    }


def calculate_position_size(symbol: str,
                            atr_14: float,
                            risk_budget_usd: float = DEFAULT_RISK_BUDGET_USD,
                            min_contracts: int = DEFAULT_MIN_CONTRACTS,
                            max_contracts: int = DEFAULT_MAX_CONTRACTS,
                            stop_atr_mult: float = 1.0,
                            config: dict | None = None,
                            config_path: str = DEFAULT_CONFIG_PATH) -> int:
    """
    Contracts to trade for `symbol`, given its current ATR(14) and a dollar
    risk budget.

    Worked, at a $250 budget and a 1x-ATR stop:

        MNQ  ATR 50.0  -> 50.0 x $2   = $100/contract -> 2.5 -> 2 contracts
        MES  ATR 10.0  -> 10.0 x $5   = $50/contract  -> 5.0 -> 5 contracts
        MCL  ATR 1.00  -> 1.00 x $100 = $100/contract -> 2.5 -> 2 contracts
        MGC  ATR 5.00  -> 5.00 x $10  = $50/contract  -> 5.0 -> 5 contracts

    Raises `SizingError` rather than returning a default for a non-positive or
    non-finite ATR, budget or multiplier, and `PortfolioConfigError` for a
    symbol with no metadata. See `size_detail` for why the floor clamp can put
    the realised risk ABOVE the budget, and for the record that says when it
    did.
    """
    return int(size_detail(symbol, atr_14, risk_budget_usd, min_contracts,
                           max_contracts, stop_atr_mult, config,
                           config_path)["contracts"])


__all__ = ["SizingError", "calculate_position_size", "size_detail",
           "DEFAULT_RISK_BUDGET_USD", "DEFAULT_MIN_CONTRACTS",
           "DEFAULT_MAX_CONTRACTS", "PortfolioConfigError"]
