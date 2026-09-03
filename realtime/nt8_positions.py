"""
realtime.nt8_positions - the BROKER's open positions, and the reconciliation
that lets the live loop close what it did not open.

WHY THIS EXISTS
===============
The loop knows three things about positions and none of them is the account:

    PositionBook            what THIS PROCESS opened. In memory, per process,
                            gone on restart - deliberately, because believing
                            a position survived a crash is the expensive way
                            to be wrong.
    EngineState             durable, and a record of what was SENT rather than
                            what is HELD. On restart `recovered` is True,
                            `reconciled` is False, and the flatten path
                            refuses to act on the claims until something
                            confirms them.
    incubator fill logs     NT8's own export, paired FIFO into round turns by
                            `portfolio.incubator_recorder`. It reports open
                            lots - but it is an END OF DAY artifact and the
                            directory is not being written.

So the account holds positions the loop cannot see, and the consequence is
exact and visible in the log: a strategy's stop IS computed live -
`signal_fn` runs the whole `_walk_loop` every cycle and `exits.iloc[-1]`
reaches `report["exit_signals"]` - and is then dropped by
`PortfolioManager.plan_exits`, whose FIRST condition is `is_open(pid,
symbol)`. An empty book turns every exit into "no position opened by this
process for <pid>/<symbol>; declining to flatten an account this loop cannot
vouch for". The stop is calculated and never sent.

`EngineState.reconciled` was written for this and nothing ever set it. This
module is the thing that does.

WHAT IT IS NOT
==============
**Not a second position store.** It reads a snapshot the broker side
publishes and loads it INTO `PositionBook`, which stays the one object
`record_fill`, `record_flat`, `plan_exits` and `can_execute` all read. A
parallel store would be written by one path and not the other and would be
wrong from the first flatten onward, with every log line reading correctly.

**Not a fill feed.** A snapshot is what the account holds NOW. Pairing fills
into round turns is `incubator_recorder`'s job and it answers a different
question - what did this strategy earn - on a different cadence.

THE PUBLISHER IS NOT IN THIS REPOSITORY
=======================================
Same as the bar feed: a NinjaScript add-on on the Windows workstation writes
the snapshot, either into the spool directory or by POSTing
`/api/positions` to `realtime/nt8_bar_listener.py`. Until it runs,
`load_snapshot` finds nothing and says so, and the loop behaves exactly as it
does today - no position is invented. The contract is:

    {"published_utc": "2026-09-03T01:00:00Z",
     "positions": [{"account": "SimIncubator1", "symbol": "MNQ",
                    "quantity": 2, "direction": "long",
                    "average_price": 29138.25}, ...]}

`quantity` is unsigned and `direction` carries the side, because a signed
quantity and a side field disagreeing is a position nobody can resolve. A
FLAT row may be sent explicitly (`quantity: 0`) and is how a publisher says
"this pair is closed" rather than leaving the reader to infer it from
absence.

WHY A SNAPSHOT IS REFUSED WHEN STALE
====================================
A snapshot is a claim about NOW. Acting on one written an hour ago is acting
on a position that may have been closed by hand, by a bracket, or by the
prop-firm layer since. `MAX_SNAPSHOT_AGE_S` refuses it, and a refusal leaves
the book exactly as it was - unreconciled and declining to flatten - which is
the same safe state the loop already has rather than a new one.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Where the publisher writes. `$BT_NT8_POSITIONS` overrides, and it sits
#: beside the bar spool because one shared folder is what NT8 already has.
DEFAULT_SNAPSHOT = "/mnt/backtest/artifacts/nt8_positions/positions.json"
SNAPSHOT_ENV_VAR = "BT_NT8_POSITIONS"

#: A snapshot older than this is refused. Two minutes is four bar-listener
#: posts and two live cycles at the shipped 60s interval - long enough that a
#: slow publisher is not flapping, short enough that a dead one is caught
#: before the next entry.
MAX_SNAPSHOT_AGE_S = 120.0

LONG, SHORT, FLAT = "long", "short", "flat"


class PositionSnapshotError(RuntimeError):
    """A snapshot that cannot be trusted. Never raised for an ABSENT one -
    see `load_snapshot`."""


def snapshot_path(path: str | Path | None = None) -> Path:
    """
    Read at CALL time, never at import. `backtest.pipeline.artifacts_root`
    documents why: a module-level default binds the variable before a test
    that sets it has run.
    """
    if path:
        return Path(path)
    return Path(os.environ.get(SNAPSHOT_ENV_VAR, DEFAULT_SNAPSHOT))


def _parse_stamp(value: Any) -> datetime:
    """
    An ISO-8601 timestamp, tz-AWARE.

    A NAIVE STAMP IS REFUSED rather than assumed to be UTC. The publisher runs
    on a Windows workstation in the instrument's or the operator's timezone,
    so a stamp with no offset is as likely to be New York as UTC - and guessed
    wrong, a five-hour-old snapshot reads as fresh. `nt8_feed` refuses a naive
    bar timestamp for exactly this reason.
    """
    if not value:
        raise PositionSnapshotError("snapshot carries no `published_utc`")
    text = str(value).strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PositionSnapshotError(
            f"`published_utc` is not ISO-8601: {value!r} ({exc})") from exc
    if stamp.tzinfo is None:
        raise PositionSnapshotError(
            f"`published_utc` {value!r} is NAIVE. The publisher runs in the "
            f"workstation's timezone, so a stamp with no offset cannot be "
            f"read as UTC without guessing - and guessed wrong, an hours-old "
            f"snapshot reads as fresh. Send an offset or a trailing Z.")
    return stamp.astimezone(timezone.utc)


def parse_position(row: Any) -> dict[str, Any]:
    """
    One position row, validated. Raises on anything it would have to guess at.

    `quantity` is UNSIGNED and `direction` carries the side. A signed quantity
    beside a side field is two statements that can disagree, and there is no
    correct way to resolve `{"quantity": -2, "direction": "long"}` - so a
    negative quantity is refused rather than reinterpreted.
    """
    if not isinstance(row, dict):
        raise PositionSnapshotError(f"position row is not an object: {row!r}")

    account = str(row.get("account") or "").strip()
    symbol = str(row.get("symbol") or "").strip().upper()
    if not account or not symbol:
        raise PositionSnapshotError(
            f"position row needs both `account` and `symbol`; got {row!r}")

    try:
        quantity = int(row.get("quantity", 0))
    except (TypeError, ValueError) as exc:
        raise PositionSnapshotError(
            f"{account}/{symbol} quantity is not an integer: "
            f"{row.get('quantity')!r}") from exc
    if quantity < 0:
        raise PositionSnapshotError(
            f"{account}/{symbol} quantity is {quantity}. Quantity is UNSIGNED "
            f"and `direction` carries the side; a negative one is a second "
            f"statement about the side that can disagree with the first.")

    direction = str(row.get("direction") or "").strip().lower()
    if quantity == 0:
        # An explicit flat row is how a publisher says "closed" rather than
        # leaving the reader to infer it from an absent row.
        direction = FLAT
    elif direction not in (LONG, SHORT):
        raise PositionSnapshotError(
            f"{account}/{symbol} holds {quantity} but its direction is "
            f"{row.get('direction')!r}; expected {LONG!r} or {SHORT!r}.")

    price = row.get("average_price")
    try:
        average_price = None if price is None else float(price)
    except (TypeError, ValueError):
        average_price = None

    return {"account": account, "symbol": symbol, "quantity": quantity,
            "direction": direction, "average_price": average_price}


def load_snapshot(path: str | Path | None = None,
                  max_age_s: float | None = MAX_SNAPSHOT_AGE_S,
                  now: datetime | None = None) -> dict[str, Any] | None:
    """
    The broker's positions, or `None` when no publisher is running.

    ABSENT IS NOT AN ERROR. Until the NinjaScript side ships there is no
    snapshot, and raising would take the loop down over a feed it has never
    had. `None` leaves every caller in exactly the state it is in today.

    A snapshot that EXISTS and cannot be trusted RAISES: unparseable, naive,
    stale, or carrying a row that would have to be guessed at. That is the
    opposite direction on purpose - a publisher that is running and wrong is a
    fact about the account, and reading it as "no positions" would be the
    loop deciding it is flat because the file was malformed.
    """
    target = snapshot_path(path)
    if not target.is_file():
        return None
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PositionSnapshotError(
            f"{target} exists and is not readable JSON: {exc}") from exc
    if not isinstance(blob, dict):
        raise PositionSnapshotError(
            f"{target} holds {type(blob).__name__}, expected an object with "
            f"`published_utc` and `positions`")

    published = _parse_stamp(blob.get("published_utc"))
    moment = now or datetime.now(timezone.utc)
    age = (moment - published).total_seconds()
    if max_age_s is not None and age > float(max_age_s):
        raise PositionSnapshotError(
            f"{target} was published {age:.0f}s ago, past "
            f"{float(max_age_s):.0f}s. A snapshot is a claim about NOW; this "
            f"one may describe positions closed by hand, by a bracket or by "
            f"the prop-firm layer since. Refusing it leaves the book "
            f"unreconciled, which is the state the loop already handles.")

    rows = blob.get("positions")
    if not isinstance(rows, list):
        raise PositionSnapshotError(
            f"{target} carries no `positions` list")

    return {"published_utc": published.isoformat(),
            "age_seconds": age,
            "source": str(target),
            "positions": [parse_position(r) for r in rows]}


def reconcile(book, snapshot: dict[str, Any] | None,
              account_for, portfolios, state=None) -> dict[str, Any]:
    """
    Load the broker's positions INTO the book, and report every difference.

    `account_for(portfolio_id) -> account` is the dispatcher's own mapping, so
    the account a snapshot names and the account an order goes to can never be
    resolved two different ways.

    THREE OUTCOMES PER PAIR, and they are kept apart because they are fixed by
    different work:

      adopted    the broker holds it and the book did not. The book is
                 updated, and the position becomes closeable - this is the
                 whole point.
      confirmed  both agree. Nothing changes.
      conflict   both hold it and they DISAGREE on side or size. The BROKER
                 wins, because it is the account, and the disagreement is
                 reported rather than resolved silently.

    A pair the BOOK holds and the broker does not is `closed_elsewhere`: the
    position was flattened by hand, by a bracket, or by the prop-firm layer.
    The book is cleared, which is what stops the loop trying to flatten
    something that is already gone.

    ADOPTED POSITIONS CARRY NO STRATEGY. `record_fill(strategies=[])` is
    deliberate: the loop did not open them, so no strategy owns them, and
    `plan_exits`' owner check treats an unknown owner set as closeable rather
    than refusing. That is the correct direction here - an adopted position
    must be closeable by whichever strategy trades that pair, or reconciling
    would hand the loop inventory it still could not exit.
    """
    result = {"reconciled": False, "adopted": [], "confirmed": [],
              "conflicts": [], "closed_elsewhere": [], "skipped": [],
              "snapshot": None}
    if snapshot is None:
        result["reason"] = ("no broker snapshot: the NinjaScript publisher is "
                            "not running, so the book keeps only what this "
                            "process opened")
        return result

    result["snapshot"] = {k: snapshot[k]
                          for k in ("published_utc", "age_seconds", "source")}

    # The account each portfolio trades on, inverted so a snapshot row can be
    # resolved back to the portfolio the book is keyed by.
    by_account: dict[str, list[str]] = {}
    for pid in portfolios:
        try:
            by_account.setdefault(str(account_for(pid)), []).append(pid)
        except Exception:                                          # noqa: BLE001
            continue

    seen: set[tuple[str, str]] = set()
    for row in snapshot["positions"]:
        pids = by_account.get(row["account"]) or []
        if not pids:
            result["skipped"].append(
                {**row, "reason": f"no portfolio routes to account "
                                  f"{row['account']!r}"})
            continue
        # A basket may hold the contract in more than one portfolio only if
        # two portfolios share an account, which the routing table forbids;
        # taking the first is therefore taking the only one.
        pid = pids[0]
        key = (pid, row["symbol"])
        seen.add(key)
        held = book.get(pid, row["symbol"]) or {}

        if row["direction"] == FLAT:
            if held:
                book.record_flat(pid, row["symbol"])
                result["closed_elsewhere"].append(
                    {"portfolio_id": pid, **row,
                     "was": held.get("direction")})
            continue

        if not held:
            book.record_fill(pid, row["symbol"], row["direction"],
                             row["quantity"], strategies=[])
            result["adopted"].append({"portfolio_id": pid, **row})
        elif (held.get("direction") == row["direction"]
              and int(held.get("quantity", 0)) == row["quantity"]):
            result["confirmed"].append({"portfolio_id": pid, **row})
        else:
            book.record_fill(pid, row["symbol"], row["direction"],
                             row["quantity"],
                             strategies=held.get("strategies") or [])
            result["conflicts"].append(
                {"portfolio_id": pid, **row,
                 "book_direction": held.get("direction"),
                 "book_quantity": held.get("quantity"),
                 "resolution": "broker wins; it is the account"})

    # Anything the book holds that the snapshot does not mention at all.
    for record in list(book.open_positions()):
        key = (record.get("portfolio_id"), record.get("symbol"))
        if key in seen:
            continue
        account = None
        try:
            account = account_for(record.get("portfolio_id"))
        except Exception:                                          # noqa: BLE001
            pass
        # Only for accounts the snapshot actually covered. A publisher that
        # sent one account's positions says nothing about another's, and
        # clearing a book entry on that silence would flatten the loop's own
        # record of a live position.
        if account not in {r["account"] for r in snapshot["positions"]}:
            continue
        book.record_flat(record["portfolio_id"], record["symbol"])
        result["closed_elsewhere"].append(
            {"portfolio_id": record.get("portfolio_id"),
             "account": account, "symbol": record.get("symbol"),
             "quantity": 0, "direction": FLAT,
             "was": record.get("direction"),
             "reason": "held by the book, absent from the broker snapshot"})

    result["reconciled"] = True
    if state is not None:
        # THE FLAG `EngineState` HAS CARRIED SINCE IT WAS WRITTEN. Its own
        # comment says only an explicit reconciliation may set it, and this is
        # that reconciliation - the claims from a previous run have now been
        # checked against what the account actually holds.
        state.reconciled = True
    return result


def describe(result: dict[str, Any]) -> str:
    """The console line. One per outcome, because 'adopted 2' and 'conflicts
    2' are different facts about an account."""
    if not result.get("reconciled"):
        return f"[positions] NOT RECONCILED — {result.get('reason', 'unknown')}"
    snap = result.get("snapshot") or {}
    parts = [f"[positions] reconciled against {snap.get('source')} "
             f"({snap.get('age_seconds', 0):.0f}s old)"]
    for name in ("adopted", "confirmed", "conflicts", "closed_elsewhere",
                 "skipped"):
        rows = result.get(name) or []
        if rows:
            parts.append(f"   {name:<16} {len(rows)}")
            for row in rows[:6]:
                parts.append(f"      {row.get('account')}/{row.get('symbol')} "
                             f"{row.get('direction')} x{row.get('quantity')}"
                             + (f"  ({row['reason']})" if row.get("reason")
                                else ""))
    if result.get("conflicts"):
        parts.append("   CONFLICTS were resolved in the BROKER's favour — it "
                     "is the account. Check what opened them.")
    return "\n".join(parts)


__all__ = ["DEFAULT_SNAPSHOT", "FLAT", "LONG", "MAX_SNAPSHOT_AGE_S",
           "PositionSnapshotError", "SHORT", "describe", "load_snapshot",
           "parse_position", "reconcile", "snapshot_path"]
