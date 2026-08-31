"""
realtime.contract_resolver - a root symbol to the contract NT8 will accept.

NinjaTrader refuses a bare root outright: "Instrument 'MNQ' not found". It
refuses it AFTER CrossTrade has returned 200, so the rejection lands
downstream of everything this repository logs - the dispatch record says `ok`
on an order that never reached an account. Nothing here can see that, which is
why the mapping has to be right rather than merely present.

WHY THE TABLE IS A FILE AND NOT A DICT IN THE CODE
==================================================
A contract month is not a constant. It is a fact about this week that stops
being true on a roll date, and a roll happens on a Monday when nobody wants to
be editing Python and redeploying a live service to fix an order that is being
refused right now. `config/contracts.json` is data an operator can correct
from the same shell they are reading the logs in, and this module re-reads it
when the file's mtime changes - so a correction takes effect on the next
cycle, with no restart.

That is also why nothing here computes a roll. Index and FX are honestly
computable - quarterly H/M/U/Z, third Friday - but the metals and the grains
are not: GC lists Feb/Apr/Jun/Aug/Oct/Dec and its liquidity SKIPS months, so
in August 2026 the active contract is DEC26 rather than the nearer OCT26. A
rule taking "the next listed month" would route gold into a thin contract and
look correct doing it. The file is what Market Analyzer says; a human keeps it
true.

TWO EXPIRIES, AND THEY FAIL FOR DIFFERENT REASONS
=================================================
`valid_until` is the whole table's. Past it nothing resolves, because a table
nobody has looked at in a month is not evidence about any symbol.

`roll_date` is per contract, and it is the one that catches the common case: a
table refreshed for the index roll while the rates quietly rolled a fortnight
earlier. Past a symbol's own roll date this REFUSES and names
`next_contract` - it does not silently substitute it, because the file is the
record of what Market Analyzer shows and this module is not entitled to
promote a guess into that record.

Both refusals cost a rejected order, which is visible in the first cycle. The
alternative is a FILLED order in an expired contract, which nothing
downstream can detect.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACTS_PATH = REPO_ROOT / "config" / "contracts.json"

#: An instrument that already names its contract. Matched so resolution is
#: IDEMPOTENT: a caller who spells the month out - `send_test_probe.py
#: --symbol "MNQ SEP26"`, or a config that pins a back month - gets exactly
#: what they typed, and never "MNQ SEP26 SEP26".
#:
#:     "MNQ SEP26"   an explicit month
#:     "MNQ 09-26"   NinjaTrader's numeric form
#:     "MNQ 1!"      a continuous contract
_QUALIFIED = re.compile(
    r"(?:\s(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s?\d{2}"
    r"|\s\d{1,2}-\d{2}"
    r"|\s?\d+!)\s*$",
    re.IGNORECASE)


class ContractResolverError(ValueError):
    """Base for every refusal this module makes."""


class ContractTableExpiredError(ContractResolverError):
    """The table as a whole, or one symbol's row, is past its date."""


class UnknownSymbolError(ContractResolverError):
    """No row for this root, and it is not already a qualified contract."""


def _parse_stamp(value: Any) -> dt.datetime:
    """An ISO date or timestamp as an AWARE UTC datetime.

    A naive value is read as UTC rather than as local time: this file is
    edited on a workstation and read on a box that runs in UTC, and reading a
    date as local would move a roll boundary by hours in whichever direction
    nobody expected.
    """
    text = str(value).strip().replace("Z", "+00:00")
    # A BARE DATE MEANS THE END OF THAT DAY, and it is matched explicitly
    # rather than by letting `fromisoformat` fail. From Python 3.11
    # `fromisoformat("2026-08-31")` SUCCEEDS and returns MIDNIGHT, so a
    # try/except would never reach this branch - and a contract with
    # roll_date 2026-08-31 would be refused from 00:00 on the thirty-first
    # instead of being tradable through it. Measured: the rates rolled a day
    # early on the first run.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        stamp = dt.datetime.combine(dt.date.fromisoformat(text),
                                    dt.time(23, 59, 59))
    else:
        try:
            stamp = dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ContractResolverError(
                f"cannot read {value!r} as a date or timestamp: {exc}") from exc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp.astimezone(dt.timezone.utc)


class ContractResolver:
    """
    `config/contracts.json`, reloaded when it changes on disk.

    The cache is keyed on the file's mtime and size, so an edit is picked up
    on the next call without a restart - which is the point of the file. Size
    is part of the key because a filesystem with one-second mtime granularity
    can hide an edit made inside the same second as the read that preceded it.

    Thread-safe: `master_live` is single-threaded today, but this object is a
    natural module-level singleton and a lock is cheaper than the class of bug
    a torn reload would produce.
    """

    def __init__(self, path: Path | str = DEFAULT_CONTRACTS_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._stamp: tuple[float, int] | None = None
        self._table: dict[str, Any] = {}

    # -- loading ----------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise ContractResolverError(
                f"cannot read the contract table at {self.path}: {exc}. Every "
                f"order names a contract month, so there is nothing to fall "
                f"back to.") from exc

        stamp = (stat.st_mtime, stat.st_size)
        with self._lock:
            if stamp == self._stamp and self._table:
                return self._table
            try:
                blob = json.loads(self.path.read_text())
            except (OSError, ValueError) as exc:
                raise ContractResolverError(
                    f"{self.path} is not readable JSON: {exc}") from exc
            if not isinstance(blob.get("contracts"), dict):
                raise ContractResolverError(
                    f"{self.path} has no `contracts` object")
            if not blob.get("valid_until"):
                raise ContractResolverError(
                    f"{self.path} declares no `valid_until`. A table with no "
                    f"expiry is one nobody is obliged to refresh.")
            self._table = blob
            self._stamp = stamp
            return self._table

    def reload(self) -> dict[str, Any]:
        """Force a re-read. The mtime check makes this unnecessary; it exists
        for a caller that has just written the file itself."""
        with self._lock:
            self._stamp = None
        return self._load()

    # -- the query --------------------------------------------------------
    def resolve_contract(self, symbol: str,
                         current_time: dt.datetime | str | None = None) -> str:
        """
        The instrument string to put on the wire.

            MNQ         -> MNQ SEP26
            MNQ SEP26   -> MNQ SEP26     (already qualified, returned verbatim)
            MNQ 1!      -> MNQ 1!
            ZZZ         -> UnknownSymbolError

        `current_time` is for testing the two expiry guards without waiting
        for the calendar. Naive values are read as UTC.

        AN ALREADY-QUALIFIED INSTRUMENT IS RETURNED BEFORE ANY DATE IS
        CHECKED. A caller who names the contract has taken the decision the
        table exists to make, and refusing them because the table is stale
        would break the one path that still works when it is - including
        `send_test_probe.py`, which is how an operator finds out what NT8
        wants.
        """
        raw = str(symbol).strip()
        if not raw:
            raise UnknownSymbolError("instrument is empty")
        ins = raw.upper()
        if _QUALIFIED.search(ins):
            return ins

        table = self._load()
        now = (_parse_stamp(current_time) if current_time is not None
               else dt.datetime.now(dt.timezone.utc))

        valid_until = _parse_stamp(table["valid_until"])
        if now > valid_until:
            raise ContractTableExpiredError(
                f"Contract table expired on {table['valid_until']} and it is "
                f"now {now.isoformat()}. Refusing to route stale symbols - "
                f"refresh {self.path} against NinjaTrader's Market Analyzer. "
                f"A wrong contract month is FILLED, not rejected.")

        row = table["contracts"].get(ins)
        if row is None:
            raise UnknownSymbolError(
                f"Unknown symbol: {ins}. Add it to {self.path}, or pass the "
                f"fully qualified instrument (e.g. '{ins} SEP26'). A bare "
                f"root is what NinjaTrader refuses.")

        roll = row.get("roll_date")
        if roll and now > _parse_stamp(roll):
            nxt = row.get("next_contract") or "the next contract"
            raise ContractTableExpiredError(
                f"{ins} rolled on {roll} and it is now {now.date()}. "
                f"{row.get('active_contract')!r} is no longer the front "
                f"month; the table names {nxt!r}. Refusing rather than "
                f"substituting it - {self.path} is the record of what Market "
                f"Analyzer shows, and this module does not get to promote a "
                f"guess into it.")

        active = row.get("active_contract")
        if not active:
            raise ContractResolverError(
                f"{ins} has a row in {self.path} with no `active_contract`")
        return str(active).strip().upper()

    # -- introspection, for the status cards ------------------------------
    def describe(self) -> str:
        """One line an operator can read before trading."""
        try:
            table = self._load()
        except ContractResolverError as exc:
            return f"contract table: UNUSABLE - {exc}"
        now = dt.datetime.now(dt.timezone.utc)
        rows = table["contracts"]
        rolled = sorted(
            sym for sym, row in rows.items()
            if row.get("roll_date") and now > _parse_stamp(row["roll_date"]))
        expired = now > _parse_stamp(table["valid_until"])
        parts = [f"{len(rows)} contracts from {self.path.name}",
                 f"valid_until {table['valid_until']}"
                 + (" (EXPIRED)" if expired else "")]
        if rolled:
            parts.append(f"{len(rolled)} past their roll date: "
                         + ", ".join(rolled))
        return "contract table: " + " | ".join(parts)


#: The module-level instance every caller shares, so one mtime check serves
#: the whole process rather than one per formatter call.
_RESOLVER = ContractResolver()


def resolve_contract(symbol: str,
                     current_time: dt.datetime | str | None = None) -> str:
    """Module-level convenience over the shared `ContractResolver`."""
    return _RESOLVER.resolve_contract(symbol, current_time=current_time)


def describe() -> str:
    return _RESOLVER.describe()


if __name__ == "__main__":                                # pragma: no cover
    import sys
    print(describe())
    for arg in sys.argv[1:]:
        try:
            print(f"  {arg:12} -> {resolve_contract(arg)}")
        except ContractResolverError as exc:
            print(f"  {arg:12} -> REFUSED: {exc}")
