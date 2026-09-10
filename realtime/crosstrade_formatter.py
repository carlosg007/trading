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
      order_type=MARKET; tif=DAY; strategy_tag=...;

* The structured JSON object, for the HTTP endpoint.

They are NOT the same payload with different punctuation and the difference is
deliberate rather than cosmetic: the plain-text form carries the API `key` and
a `tif`, the JSON form carries neither (the key travels in the request, not in
the body). BOTH carry `strategy_tag` - see below. Casing differs too - the
text command is upper-cased because that is what the add-on's parser compares
against, the JSON
object is lower-cased because that is what the endpoint's schema declares.
Neither is a preference. Writing one and posting it to the other's endpoint fails.

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

The strategy tag is a LOCK, and both forms carry it
---------------------------------------------------
CrossTrade keys a strategy lock on `strategy_tag`: an order carrying one may
only act on the position that tag opened, and the lock clears when a matching
order closes it. That makes the tag part of the ORDER, not part of the
reporting around it - an entry tagged and an exit untagged does not close the
position, it is refused against a lock nothing releases.

So the tag is optional in both forms and, when given, is the SAME string in
both. It used to live only in the JSON object, which is the form that is not
on the wire: `live_dispatcher.dispatch_order` sends the semicolon command, so
every live entry went out untagged while the tag sat on an unsent dict beside
it.

It is sanitised rather than escaped - `;`, `=` and whitespace are removed by
`sanitize_strategy_tag`, because they are the plain-text command's field
separators and a tag containing one would be parsed as extra fields. A tag
that is nothing BUT separators is refused rather than reduced to the empty
string: a caller that asked for a lock and silently got an untagged order is
the failure this field exists to prevent. An empty or omitted tag appends no
field at all.

The key is a credential
-----------------------
It appears in the plain-text command because the wire format requires it - and
nowhere else. It is never echoed into an exception message (see `_reject`), and
`redact` exists so a formatted command can be logged or printed without
publishing the account's key to a log file that outlives it. An empty `key`
falls back to `$CROSSTRADE_KEY`; an explicit argument always wins.
"""

from __future__ import annotations

import datetime as _dt
import json
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


#: THE CONTRACT TABLE LIVES IN A FILE, NOT HERE.
#:
#: `realtime/contract_resolver.py` reads `config/contracts.json` and re-reads
#: it when the file changes on disk. A contract month is not a constant - it
#: is a fact about this week that stops being true on a roll date, and a roll
#: happens on a Monday when nobody wants to be editing Python and redeploying
#: a live service to fix an order being refused right now. The in-code
#: `ACTIVE_CONTRACT` dict this replaced could only be corrected by a commit
#: and a restart.
#:
#: The name is re-exported so existing callers and tests keep importing it
#: from here.
from realtime.contract_resolver import (  # noqa: E402
    ContractResolverError,
    ContractTableExpiredError,
    UnknownSymbolError,
    resolve_contract,
)


def _resolved(instrument: str) -> str:
    """
    `resolve_contract`, with its refusal translated into this module's error.

    WHY THE TRANSLATION IS THE FIX AND NOT A CONVENIENCE. This module's whole
    contract is the sentence at the top of the file: a malformed payload built
    here is caught before it reaches the one place that can act on it. A
    symbol that cannot be resolved is a payload that cannot be built, so it
    belongs to the same refusal - but `ContractResolverError` is a sibling of
    `CrossTradeFormatError` under ValueError, not a subclass, and
    `realtime/live_dispatcher.py` catches only the latter.

    So every unresolvable symbol RAISED THROUGH the handler that exists to
    record a refused order and return it. Three ways to hit it, and the third
    is not hypothetical: an unknown root, an empty instrument, and a symbol
    PAST ITS ROLL DATE - which on 2026-09-10 was ES, NQ, YM, RTY and their
    four micros, every one of them carrying live strategies. The dispatcher
    would not have recorded "order refused"; the exception would have left the
    per-order handler entirely.

    The order is refused either way. What changes is that it is now refused
    the way this module already documents, as one recorded error on one order,
    instead of escaping into the caller.
    """
    try:
        return resolve_contract(instrument)
    except ContractResolverError as exc:
        raise _reject(str(exc)) from exc


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


#: What is stripped out of a strategy tag: the plain-text command's two field
#: separators, and whitespace. Whitespace is included because the command is
#: split on `;` and read as `name=value` - a tag with a space in it survives
#: that split but reaches the receiver with the space in the value, and two
#: spellings of one strategy are two different locks.
_TAG_STRIP = re.compile(r"[\s;=]+")


#: What CrossTrade says when its STRATEGY LOCK refuses an order. Observed on
#: 2026-09-04 against live accounts:
#:
#:     Trade blocked: MGC DEC26 is managed by strategy 'Incubator-Even:...'
#:     Trade blocked: 6J SEP26 is managed by strategy 'Incubator-Odd:...'
#:
#: Matched case-insensitively on the DECODED BODY, and deliberately narrow -
#: two phrases CrossTrade writes when it declines, not a general search for
#: the word "error". A false positive marks a filled order as failed, which
#: leaves the position book denying a position that exists.
REFUSAL_MARKERS = ("trade blocked", "is managed by strategy")


def refusal_reason(response_body: Any) -> str | None:
    """
    The refusal CrossTrade wrote into a 2xx body, or None.

    **A 2xx IS NOT A FILL.** `live.dispatcher.send_execution_signal` sets `ok`
    from the HTTP status alone, which is the right rule for a transport: the
    request arrived and was understood. But CrossTrade ACCEPTS the webhook and
    then declines the trade in the body, so an order its strategy lock refused
    came back 200 and every log line read `OK`.

    That is the expensive direction on the FLATTEN path. `dispatch_exits`
    clears the position book only for a flatten that succeeded - so a refusal
    read as success cleared the book, the loop believed it was flat, and the
    position stayed open at the broker with nothing in any log saying so. On
    2026-09-04 sixty-two flattens were logged `OK` this way.

    Returns the body TEXT rather than a bool, because "which lock is it
    holding" is the next question and the answer is in that sentence.
    """
    if response_body is None:
        return None
    text = (response_body if isinstance(response_body, str)
            else json.dumps(response_body) if isinstance(response_body,
                                                         (dict, list))
            else str(response_body))
    low = text.lower()
    if any(marker in low for marker in REFUSAL_MARKERS):
        return " ".join(text.split())[:300]
    return None


def sanitize_strategy_tag(strategy_tag: Any) -> str:
    """
    A tag safe to put in either wire form, or `""` for no tag at all.

    `;`, `=` and whitespace are REMOVED rather than escaped - the plain-text
    command has no escape syntax, so a tag carrying a separator is not a tag
    with an odd character in it, it is extra fields the receiver will parse.

    `None` and `""` mean "no tag" and return `""`; the caller then omits the
    field entirely. A tag that is non-empty but sanitises to nothing (`"; ;"`)
    RAISES instead, because a caller that asked for a strategy lock and
    silently got an untagged order has the failure this field exists to
    prevent - CrossTrade would place the order against no lock, and the exit
    that expects one would have nothing to clear.
    """
    if strategy_tag is None:
        return ""
    raw = str(strategy_tag)
    tag = _TAG_STRIP.sub("", raw)
    if raw.strip() and not tag:
        raise _reject(f"strategy_tag {raw!r} is nothing but field separators. "
                      f"An order formatted from it would carry no tag at all, "
                      f"and CrossTrade would hold no lock for the exit to "
                      f"clear.")
    return tag


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
                              tif: str = "day",
                              strategy_tag: str | None = None) -> str:
    """
    The semicolon-delimited plain-text place command.

        key={key}; command=place; account={account}; instrument={instrument};
        action={ACTION}; qty={qty}; order_type={ORDER_TYPE}; tif={TIF};
        strategy_tag={TAG};

    Field ORDER is fixed and part of the contract - several NinjaTrader add-on
    builds parse positionally when a field is absent, so a reordered command is
    not an equivalent command. `strategy_tag` is therefore APPENDED, after the
    last field of the form that existed before it, and is omitted entirely
    rather than sent empty when no tag is given: a trailing `strategy_tag=;` is
    a field with a value, and a receiver holding a lock on `""` is not the same
    as one holding no lock.

    The tag is CrossTrade's strategy lock - the order may act only on the
    position that tag opened. The matching `format_flatten_command` must carry
    the SAME tag or the exit does not clear it.

    Raises `CrossTradeFormatError` on an unknown side, a non-positive or
    fractional quantity, a price-bearing order type, an unknown time-in-force,
    or an account or instrument carrying a field separator.
    """
    resolved_key = resolve_key(key)
    acct = _clean_account(account)
    # THE ROOT IS RESOLVED TO A CONTRACT MONTH BEFORE IT IS CLEANED.
    # NinjaTrader refuses a bare root outright - "Instrument 'MNQ' not
    # found" - and `resolve_contract` is idempotent, so a caller who
    # already named the month gets exactly what they typed.
    ins = _clean_instrument(_resolved(instrument))
    act = _clean_action(action, allowed=PLACE_ACTIONS)
    quantity = _clean_qty(qty)
    ot = _clean_order_type(order_type)
    time_in_force = _clean_tif(tif)

    tag = sanitize_strategy_tag(strategy_tag)

    return (f"key={resolved_key}; command=place; account={acct}; "
            f"instrument={ins}; action={act}; qty={quantity}; "
            f"order_type={ot}; tif={time_in_force};"
            + (f" strategy_tag={tag};" if tag else ""))


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
    # THE ROOT IS RESOLVED TO A CONTRACT MONTH BEFORE IT IS CLEANED.
    # NinjaTrader refuses a bare root outright - "Instrument 'MNQ' not
    # found" - and `resolve_contract` is idempotent, so a caller who
    # already named the month gets exactly what they typed.
    ins = _clean_instrument(_resolved(instrument))
    act = _clean_action(action, allowed=PLACE_ACTIONS)
    quantity = _clean_qty(qty)
    ot = _clean_order_type(order_type)

    # THE SAME SANITISER AS THE TEXT FORM. The tag is a lock key matched by
    # string equality, so `keltner trend` reaching one endpoint as
    # `keltner trend` and the other as `keltnertrend` is two locks.
    tag = sanitize_strategy_tag(strategy_tag)
    return {
        "command": "place",
        "account": acct,
        "instrument": ins,
        "action": act.lower(),
        "qty": quantity,
        "order_type": ot.lower(),
        "strategy_tag": tag,
    }


def format_flatten_json(account: str,
                       instrument: str,
                       strategy_tag: str = "") -> dict[str, Any]:
    """
    The structured JSON flatten, the exact analogue of `format_crosstrade_json`.

        {"command": "flatten", "account": ..., "instrument": ...,
         "strategy_tag": ...}

    It exists because `live.dispatcher.send_execution_signal` POSTs a DICT, and
    the only flatten this module had was the plain-text form. Wrapping that
    string in an invented envelope would put a payload shape on the wire that
    no endpoint schema declares, so the JSON form is spelled out here beside
    the place order it mirrors - same lower-casing, same `strategy_tag`, same
    absence of a key.

    NO `action` AND NO `qty`, exactly as in the plain-text flatten. A flatten
    closes whatever is open; expressing it as a side and a size requires
    guessing the position, and a wrong guess does not close a position - it
    opens the opposite one. Their absence here is the contract, not an omission.
    """
    return {
        "command": "flatten",
        "account": _clean_account(account),
        # RESOLVED, like the three commands it mirrors. It was the one payload
        # here that did not resolve a root to its contract month, which made
        # the flatten this dispatcher keeps on the record disagree with the
        # flatten it puts on the wire - `MNQ` against `MNQ SEP26` - about which
        # contract was closed.
        "instrument": _clean_instrument(_resolved(instrument)),
        "strategy_tag": sanitize_strategy_tag(strategy_tag),
    }


def format_flatten_command(account: str,
                           instrument: str,
                           key: str = "",
                           strategy_tag: str | None = None) -> str:
    """
    The plain-text flatten.

        key={key}; command=flatten; account={account}; instrument={instrument};
        strategy_tag={TAG};

    No side and no quantity, and that is the whole point: a flatten closes
    whatever is open, and neither this module nor the caller knows what that
    is. Expressing it as `action=sell; qty=N` requires guessing the position,
    and a wrong guess does not close a position - it opens the opposite one.

    `strategy_tag` IS the exception to "this flatten knows nothing about the
    position", and it is not a contradiction: it does not describe what is
    open, it names the lock the entry took out. It must be BYTE-IDENTICAL to
    the tag the entry carried - CrossTrade matches the lock on the string, so
    a flatten tagged with one contributor of a netted position, or with a
    differently-spelled tag, does not clear the lock it was meant to release.
    `live_dispatcher` rebuilds it from the position book for exactly that
    reason. Omitted when no tag is given, which is an UNTAGGED flatten: it acts
    on whatever the account holds, as it always has.
    """
    resolved_key = resolve_key(key)
    acct = _clean_account(account)
    # THE ROOT IS RESOLVED TO A CONTRACT MONTH BEFORE IT IS CLEANED.
    # NinjaTrader refuses a bare root outright - "Instrument 'MNQ' not
    # found" - and `resolve_contract` is idempotent, so a caller who
    # already named the month gets exactly what they typed.
    ins = _clean_instrument(_resolved(instrument))
    tag = sanitize_strategy_tag(strategy_tag)
    return (f"key={resolved_key}; command=flatten; account={acct}; "
            f"instrument={ins};"
            + (f" strategy_tag={tag};" if tag else ""))


def format_account_flatten_command(account: str, key: str = "") -> str:
    """
    The ACCOUNT-LEVEL flatten: close everything on the account, full stop.

        key={key}; command=flatten; account={account};

    NO INSTRUMENT AND NO STRATEGY TAG, and both absences are the point. This
    is the emergency kill: an instrument scopes a flatten to one contract and a
    tag scopes it to one strategy's lock, so a halt built from either closes
    the positions it can name and leaves behind the ones it cannot - which,
    during the failure that triggered a halt, are exactly the ones nobody can
    account for.

    A SEPARATE FUNCTION FROM `format_flatten_command`, NOT AN EMPTY
    `instrument=` ON IT. The two commands differ by one absent field and
    differ in blast radius by the whole account, so a bug that drops an
    instrument must not be able to silently escalate one into the other.
    Reaching this hammer requires naming it.

    **IT CLOSES POSITIONS THIS PROCESS DID NOT OPEN.** On a shared account that
    includes anything placed by hand, by NinjaTrader, by a previous run of this
    loop or by another tool, and it is silent, immediate and unrecoverable.
    `realtime.lifecycle.emergency_halt` is the only caller, and it is called
    when continuing to trade is the larger risk. Nothing on the ordinary exit
    path may use it: `dispatch_exits` flattens per instrument, under the tag
    the entry took out, and only for a position this process recorded opening.
    """
    resolved_key = resolve_key(key)
    acct = _clean_account(account)
    return f"key={resolved_key}; command=flatten; account={acct};"


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
