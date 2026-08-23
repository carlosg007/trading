"""
realtime.crosstrade_formatter - CrossTrade order payloads, in both wire forms.

This module FORMATS and it does not SEND. `live/dispatcher.py` is still the
only module in this repository that puts an order on the wire; everything here
returns a string or a dict to a caller who then decides what to do with it.
That split is why the validation below is worth having twice: a malformed
payload built here is caught before it reaches the one place that can act on
it.

Two forms, because CrossTrade accepts two
-----------------------------------------
* The semicolon plain-text command, which is what the CrossTrade webhook and
  the NinjaTrader add-on both parse:

      key=...; command=place; account=...; instrument=...; action=BUY; qty=1;
      order_type=MARKET; tif=DAY;

* The structured JSON object, for the HTTP endpoint.

They are NOT the same payload with different punctuation and the difference is
deliberate rather than cosmetic: the plain-text form carries the API `key` and
a `tif`, the JSON form carries a `strategy_tag` and no key (the key travels in
the request, not in the body). Casing differs too - the text command is
upper-cased because that is what the add-on's parser compares against, the JSON
is lower-cased because that is what the endpoint's schema declares. Neither is
a preference. Writing one and posting it to the other's endpoint fails.

What is validated, and why each check exists
--------------------------------------------
* **The side.** Checked against `live.dispatcher.VALID_ACTIONS` - IMPORTED,
  never restated. A second list of legal sides in a second module is how BUY
  and SELL come to mean different things in two files that both look right.
  `place` additionally refuses FLATTEN: `command=place; action=flatten;` is not
  a flatten, it is a place order with a side the receiver does not recognise.
  `format_flatten_command` is how a flatten is expressed.
* **The order type.** MARKET only, from `live.dispatcher.VALID_ORDER_TYPES`.
  None of these three signatures carries a price field, and defaulting a LIMIT
  order to the market is how a risk-managed entry becomes an unbounded one.
  The parameter is kept in the signature (a price-bearing form will need it)
  and a LIMIT passed to it raises rather than being silently marketed.
* **The quantity.** A positive whole number of contracts. `bool` is rejected
  explicitly because it is an `int` subclass and `True` would otherwise become
  a live order for one contract.
* **`tif`.** Checked against a declared set. A typo'd time-in-force is not
  rejected by every broker; some substitute a default, which is a different
  order from the one the caller wrote.

The key is a credential
-----------------------
It appears in the plain-text command because the wire format requires it - and
nowhere else. It is never echoed into an exception message (see `_reject`), and
`redact` exists so a formatted command can be logged or printed without
publishing the account's key to a log file that outlives it. An empty `key`
falls back to `$CROSSTRADE_KEY`; an explicit argument always wins.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live.dispatcher import VALID_ACTIONS, VALID_ORDER_TYPES  # noqa: E402

# `place` takes a side. FLATTEN is a COMMAND here, not a side - see the module
# docstring - so it is removed from the set a place order may carry rather than
# being spelled out as a second literal that could drift from the import.
PLACE_ACTIONS = frozenset(VALID_ACTIONS) - {"FLATTEN"}

# Time-in-force values CrossTrade forwards to NinjaTrader. DAY and GTC are the
# only two that mean anything for a futures account trading a single session;
# the rest are rejected rather than passed through, because a broker that does
# not recognise a TIF substitutes its own and the resulting order is not the
# one that was written.
VALID_TIF = frozenset({"DAY", "GTC"})

# The environment variable an empty `key` falls back to.
KEY_ENV_VAR = "CROSSTRADE_KEY"

# What `redact` replaces a key with. Fixed-width and obviously not a key, so a
# redacted command cannot be copied out of a log and replayed.
REDACTED = "***REDACTED***"

_KEY_FIELD = re.compile(r"(key=)([^;]*)(;)")


class CrossTradeFormatError(ValueError):
    """A payload was refused. Never carries the API key - see `_reject`."""


def _reject(message: str) -> "CrossTradeFormatError":
    """
    Build the refusal.

    A function rather than a bare `raise` so there is one place responsible for
    the rule that an exception from this module never quotes the key. Callers
    log exceptions; a key in an exception message is a key in a log file.
    """
    return CrossTradeFormatError(message)


# --------------------------------------------------------------------------
# field validation
# --------------------------------------------------------------------------
def _clean_account(account: str) -> str:
    acct = str(account).strip()
    if not acct:
        raise _reject("account is empty. There is no default account: an order "
                      "formatted for an unnamed account would be routed by the "
                      "receiver's own default, which is not a decision this "
                      "repository makes.")
    if ";" in acct or "=" in acct:
        raise _reject(f"account {acct!r} contains ';' or '=', which are the "
                      f"plain-text command's field separators. The receiver "
                      f"would parse the remainder as extra fields.")
    return acct


def _clean_instrument(instrument: str) -> str:
    """
    Upper-cased and stripped, and NOT otherwise translated.

    A CrossTrade instrument is a NinjaTrader instrument name - `NQ 12-26`,
    `MNQ`, `ES 03-26` - not a lake symbol, and this module does not know the
    contract calendar. Mapping `NQ` onto a specific expiry is a decision that
    belongs where the roll calendar lives; guessed here it would send an order
    to whichever contract month a stale rule named.
    """
    ins = str(instrument).strip().upper()
    if not ins:
        raise _reject("instrument is empty")
    if ";" in ins or "=" in ins:
        raise _reject(f"instrument {ins!r} contains ';' or '=', the plain-text "
                      f"command's field separators.")
    return ins


def _clean_action(action: str, *, allowed: frozenset[str]) -> str:
    act = str(action).strip().upper()
    if act not in allowed:
        raise _reject(
            f"action {action!r} is not one of {sorted(allowed)}. Refusing to "
            f"format an unrecognised side."
            + ("" if "FLATTEN" in allowed else
               " FLATTEN is a command, not a side: use "
               "format_flatten_command().")
        )
    return act


def _clean_qty(qty: Any) -> int:
    # bool is an int subclass; True would otherwise become one contract.
    if isinstance(qty, bool):
        raise _reject("qty must be a number of contracts, not a bool")
    try:
        qty_f = float(qty)
    except (TypeError, ValueError):
        raise _reject(f"qty {qty!r} is not a number") from None
    if qty_f != int(qty_f):
        raise _reject(f"qty {qty!r} is not a whole number of contracts. A "
                      f"fractional futures order has no meaning and the "
                      f"receiver would round it by a rule nobody chose.")
    out = int(qty_f)
    if out <= 0:
        raise _reject(f"qty must be positive, got {qty!r}. A zero-quantity "
                      f"order is not a flatten - see format_flatten_command().")
    return out


def _clean_order_type(order_type: str) -> str:
    ot = str(order_type).strip().upper()
    if ot not in VALID_ORDER_TYPES:
        raise _reject(
            f"order_type {order_type!r} is not supported. These signatures "
            f"carry no price field, so only {sorted(VALID_ORDER_TYPES)} can be "
            f"formatted safely - a price-bearing type defaulted to the market "
            f"turns a bounded entry into an unbounded one. Add a price to the "
            f"signature first.")
    return ot


def _clean_tif(tif: str) -> str:
    value = str(tif).strip().upper()
    if value not in VALID_TIF:
        raise _reject(f"tif {tif!r} is not one of {sorted(VALID_TIF)}. A "
                      f"time-in-force the receiver does not recognise is "
                      f"replaced with its default rather than refused.")
    return value


def resolve_key(key: str = "") -> str:
    """
    The API key to put on the wire: the argument when given, else
    `$CROSSTRADE_KEY`, else the empty string.

    Returned rather than logged, and never included in an exception. An empty
    result is NOT an error here - the plain-text form is also used in tests and
    in dry-run output, where there is no key to carry and inventing one would
    be worse. `send_execution_signal` is where a missing credential has to
    fail, because that is where it stops being a string.
    """
    if str(key).strip():
        return str(key).strip()
    return os.environ.get(KEY_ENV_VAR, "").strip()


# --------------------------------------------------------------------------
# the three payloads
# --------------------------------------------------------------------------
def format_crosstrade_command(account: str,
                              instrument: str,
                              action: str,
                              qty: int,
                              order_type: str = "market",
                              key: str = "",
                              tif: str = "day") -> str:
    """
    The semicolon-delimited plain-text place command.

        key={key}; command=place; account={account}; instrument={instrument};
        action={ACTION}; qty={qty}; order_type={ORDER_TYPE}; tif={TIF};

    Field ORDER is fixed and part of the contract - several NinjaTrader add-on
    builds parse positionally when a field is absent, so a reordered command is
    not an equivalent command.

    Raises `CrossTradeFormatError` on an unknown side, a non-positive or
    fractional quantity, a price-bearing order type, an unknown time-in-force,
    or an account or instrument carrying a field separator.
    """
    resolved_key = resolve_key(key)
    acct = _clean_account(account)
    ins = _clean_instrument(instrument)
    act = _clean_action(action, allowed=PLACE_ACTIONS)
    quantity = _clean_qty(qty)
    ot = _clean_order_type(order_type)
    time_in_force = _clean_tif(tif)

    return (f"key={resolved_key}; command=place; account={acct}; "
            f"instrument={ins}; action={act}; qty={quantity}; "
            f"order_type={ot}; tif={time_in_force};")


def format_crosstrade_json(account: str,
                           instrument: str,
                           action: str,
                           qty: int,
                           order_type: str = "market",
                           strategy_tag: str = "") -> dict[str, Any]:
    """
    The structured JSON place order.

        {"command": "place", "account": ..., "instrument": ...,
         "action": "buy", "qty": 1, "order_type": "market",
         "strategy_tag": ...}

    Lower-cased, which is what the endpoint's schema declares - and the
    opposite of the plain-text form, deliberately. See the module docstring.

    `strategy_tag` is free text and is what ties a fill in the NT8 log back to
    the strategy that asked for it (`live.dispatcher.evaluate_incubator_sync`
    reconciles on it). It defaults to empty rather than to the account or the
    instrument, because a tag invented here would attribute a fill to a
    strategy that did not place it.

    No API key: the JSON endpoint takes the credential in the request, not in
    the body. Putting one here would publish it into every payload log.
    """
    acct = _clean_account(account)
    ins = _clean_instrument(instrument)
    act = _clean_action(action, allowed=PLACE_ACTIONS)
    quantity = _clean_qty(qty)
    ot = _clean_order_type(order_type)

    tag = str(strategy_tag).strip()
    return {
        "command": "place",
        "account": acct,
        "instrument": ins,
        "action": act.lower(),
        "qty": quantity,
        "order_type": ot.lower(),
        "strategy_tag": tag,
    }


def format_flatten_command(account: str,
                           instrument: str,
                           key: str = "") -> str:
    """
    The plain-text flatten.

        key={key}; command=flatten; account={account}; instrument={instrument};

    No side and no quantity, and that is the whole point: a flatten closes
    whatever is open, and neither this module nor the caller knows what that
    is. Expressing it as `action=sell; qty=N` requires guessing the position,
    and a wrong guess does not close a position - it opens the opposite one.
    """
    resolved_key = resolve_key(key)
    acct = _clean_account(account)
    ins = _clean_instrument(instrument)
    return (f"key={resolved_key}; command=flatten; account={acct}; "
            f"instrument={ins};")


# --------------------------------------------------------------------------
# logging safety
# --------------------------------------------------------------------------
def redact(command: str) -> str:
    """
    A formatted plain-text command with its `key=` field replaced.

    Use this on every path that prints, logs or reports a command. The key is
    a bearer credential for a live account: a log file outlives the session
    that wrote it, and a command copied out of one is directly replayable.
    A command with no `key=` field passes through unchanged.
    """
    return _KEY_FIELD.sub(rf"\g<1>{REDACTED}\g<3>", str(command))
