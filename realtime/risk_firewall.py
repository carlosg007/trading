#!/usr/bin/env python3
"""
realtime.risk_firewall - the last thing that runs before an order goes out.

Every order this repository sends passes through `check_order` inside
`LiveExecutionDispatcher.dispatch_order`, before the socket. Not beside it, not
as a decorator a caller can forget: a pre-trade gate that can be bypassed is a
pre-trade gate that will be, on the day somebody adds a second dispatch path in
a hurry.

WHAT THIS LAYER KNOWS, AND WHAT IT DELIBERATELY DOES NOT
========================================================
It enforces what THIS PROCESS can observe and prove:

    * the kill switch                      a file, checked every order
    * session cutoff                       a clock
    * contracts per order / per symbol     the payload and this process's book
    * orders per session                   a counter this process owns
    * duplicate bar guard                  the signal ledger
    * realised loss booked BY THIS LOOP    its own fills

It does NOT enforce the funding program's trailing drawdown or its daily loss
percentage, and that is a decision rather than an omission. Those rules resolve
against a LIVE ACCOUNT BALANCE this process cannot see: the account is shared
with NinjaTrader, with the operator's own hands and with previous runs, so a
balance computed here from this loop's fills alone would be a second, weaker
opinion about a number CrossTrade NAM already enforces authoritatively. Two
risk systems that disagree are worse than one, because the account is halted by
whichever is stricter while the operator reads the other.

`compliance_rules/*.json` remains the SPECIFICATION handed to NAM. What lives
here is a local circuit breaker: a hard dollar cap on what this loop may lose
on its own recorded fills before it stops sending, which is a statement about
this process and not a restatement of the rulebook.

FAIL CLOSED, ALWAYS
===================
Every check that cannot be evaluated REFUSES. A missing state file, an
unreadable kill switch, a payload with no quantity, a clock that will not parse
- each of them blocks the order and says which check could not run. The
alternative is a gate that passes when it is broken, and the first time anybody
notices is on a fill.

THE KILL SWITCH IS A FILE
=========================
`data/KILL_SWITCH` (or `$BT_KILL_SWITCH`). A file because it works from any
shell on the box, needs no socket into a process that may be wedged, survives a
restart, and is visible to `ls` when somebody asks whether trading is halted. It
is checked before EVERY order rather than once per cycle: an operator arming it
mid-cycle means "now", not "after the orders already queued".

Arming it blocks new orders. It does NOT flatten by itself - see
`realtime/lifecycle.py::emergency_halt`, which arms the switch, then closes
what this process opened, then alerts. Halting and flattening are separate
because "stop trading" and "close my positions" are different instructions, and
a switch that did both would be one nobody dares touch.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KILL_SWITCH_VAR = "BT_KILL_SWITCH"
DEFAULT_KILL_SWITCH = "data/KILL_SWITCH"

# Local ceilings. Deliberately conservative, deliberately in absolute units,
# and deliberately NOT the funding program's percentages - see the module
# docstring. An operator raises these on purpose; nothing raises them
# automatically.
DEFAULT_LIMITS: dict[str, Any] = {
    "max_contracts_per_order": 5,      # matches the sizer's own clamp
    "max_contracts_per_symbol": 5,     # net, across strategies, this process
    "max_open_positions": 4,           # one per portfolio-symbol in flight
    "max_orders_per_session": 40,      # a runaway loop stops here
    "max_session_loss_usd": 1000.0,    # THIS loop's realised loss, not NAM's
    "session_cutoff_utc": None,        # "20:55" halts new entries before close
}


class RiskViolation(Exception):
    """An order was refused before the socket. Carries the rule that refused it."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


def kill_switch_path(path: str | Path | None = None) -> Path:
    """The switch file: the argument, then `$BT_KILL_SWITCH`, then the default."""
    if path is not None:
        return Path(path)
    override = (os.environ.get(KILL_SWITCH_VAR) or "").strip()
    candidate = Path(override) if override else Path(DEFAULT_KILL_SWITCH)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def kill_switch_engaged(path: str | Path | None = None) -> tuple[bool, str]:
    """
    `(engaged, reason)`. A switch that cannot be READ counts as ENGAGED.

    The failure directions are not symmetric. Unreadable-means-open lets a
    permissions mistake arm a halt nobody asked for, and somebody notices in
    minutes. Unreadable-means-closed lets the same mistake disable the halt,
    and nobody notices until it was needed.
    """
    target = kill_switch_path(path)
    try:
        if not target.exists():
            return False, ""
        reason = target.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return True, (f"kill switch at {target} exists but could not be read "
                      f"({exc}). Treating it as ENGAGED: a switch nobody can "
                      f"read is not a switch anybody may ignore.")
    return True, reason or f"kill switch armed at {target} (no reason recorded)"


def arm_kill_switch(reason: str, path: str | Path | None = None) -> Path:
    """Write the switch. Idempotent; an existing reason is not overwritten."""
    target = kill_switch_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        target.write_text(f"{stamp} {reason}\n", encoding="utf-8")
    return target


def disarm_kill_switch(path: str | Path | None = None) -> bool:
    """Remove the switch. True when one was there."""
    target = kill_switch_path(path)
    if not target.exists():
        return False
    target.unlink()
    return True


def _parse_cutoff(value: Any) -> clock_time | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        hour, minute = (int(part) for part in text.split(":", 1))
        return clock_time(hour, minute, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        raise RiskViolation(
            "session_cutoff",
            f"cutoff {value!r} is not HH:MM UTC, so the session boundary "
            f"cannot be evaluated. Refusing rather than trading past a limit "
            f"nobody could read.") from None


class RiskFirewall:
    """
    The pre-trade gate. Deterministic, local, and refuses when unsure.

    `state` is a `realtime.lifecycle.EngineState` (or anything exposing the
    same three reads) so the counters a limit is measured against are the ones
    that survive a restart. Passed None, the firewall still enforces every
    check that needs no history - and says so, rather than silently enforcing
    fewer rules than the operator believes.
    """

    def __init__(self, limits: dict[str, Any] | None = None,
                 state: Any | None = None,
                 kill_switch: str | Path | None = None) -> None:
        merged = dict(DEFAULT_LIMITS)
        merged.update(limits or {})
        unknown = set(merged) - set(DEFAULT_LIMITS)
        if unknown:
            raise RiskViolation(
                "config",
                f"unknown risk limit(s) {sorted(unknown)}. A misspelled limit "
                f"is a limit that is not enforced, and it would look "
                f"configured.")
        self.limits = merged
        self.state = state
        self.kill_switch = kill_switch
        self.cutoff = _parse_cutoff(merged.get("session_cutoff_utc"))
        self.refusals: list[dict[str, Any]] = []

    # -- the gate ---------------------------------------------------------
    def check_order(self, payload: dict, *, strategy_tag: str = "",
                    bar_ts: str | None = None,
                    now: datetime | None = None) -> None:
        """
        Raise `RiskViolation` if this order must not go out. Return None if it
        may.

        Checked in the order a human would: is trading halted at all, is the
        session still open, is the order itself sane, has this exact signal
        already been sent, and has this loop lost more than it is allowed to.
        """
        stamp = now or datetime.now(timezone.utc)

        engaged, reason = kill_switch_engaged(self.kill_switch)
        if engaged:
            self._refuse("kill_switch", reason or "armed", payload)

        if self.cutoff is not None and stamp.timetz() >= self.cutoff:
            self._refuse(
                "session_cutoff",
                f"{stamp:%H:%M} UTC is past the {self.limits['session_cutoff_utc']} "
                f"cutoff. New entries are closed for the session.", payload)

        quantity = payload.get("quantity")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            self._refuse(
                "quantity", f"quantity {quantity!r} is not a positive whole "
                f"number of contracts, so no size limit can be evaluated "
                f"against it.", payload)
        cap = int(self.limits["max_contracts_per_order"])
        if quantity > cap:
            self._refuse("max_contracts_per_order",
                         f"{quantity} contracts exceeds the {cap} allowed on "
                         f"one order.", payload)

        symbol = str(payload.get("symbol") or "").upper()
        account = str(payload.get("account") or "")
        if not symbol or not account:
            self._refuse("payload",
                         f"order names symbol={symbol!r} account={account!r}; "
                         f"a limit cannot be attributed to an order that does "
                         f"not say what it is.", payload)

        if self.state is not None:
            self._check_against_state(payload, symbol, account, quantity,
                                      strategy_tag, bar_ts)

    def _check_against_state(self, payload, symbol, account, quantity,
                             strategy_tag, bar_ts) -> None:
        """The limits that need history. Skipped entirely when there is none."""
        held = abs(int(self.state.net_contracts(account, symbol)))
        symbol_cap = int(self.limits["max_contracts_per_symbol"])
        if held + quantity > symbol_cap:
            self._refuse(
                "max_contracts_per_symbol",
                f"{account}/{symbol} would hold {held + quantity} contracts "
                f"against a cap of {symbol_cap} ({held} already open by this "
                f"process).", payload)

        open_cap = int(self.limits["max_open_positions"])
        if self.state.open_position_count() >= open_cap and held == 0:
            self._refuse(
                "max_open_positions",
                f"this process already holds {self.state.open_position_count()} "
                f"positions, the cap is {open_cap}, and this order opens "
                f"another.", payload)

        order_cap = int(self.limits["max_orders_per_session"])
        if self.state.orders_sent_today() >= order_cap:
            self._refuse(
                "max_orders_per_session",
                f"{self.state.orders_sent_today()} orders already sent today "
                f"against a cap of {order_cap}. A loop sending more than this "
                f"is not trading, it is looping.", payload)

        loss_cap = float(self.limits["max_session_loss_usd"])
        booked = float(self.state.realised_pnl_today())
        if booked <= -abs(loss_cap):
            self._refuse(
                "max_session_loss_usd",
                f"this loop has booked {booked:,.2f} today against a local cap "
                f"of -{abs(loss_cap):,.2f}. This is THIS PROCESS's realised "
                f"P&L, not the funding program's rule - CrossTrade NAM "
                f"enforces that one against the live balance.", payload)

        if bar_ts and strategy_tag:
            if self.state.already_dispatched(strategy_tag, symbol, bar_ts):
                self._refuse(
                    "duplicate_bar",
                    f"{strategy_tag} already sent an order for {symbol} on the "
                    f"bar at {bar_ts}. A restart mid-cycle re-evaluates the "
                    f"same closed bar, and acting on it twice is a double "
                    f"position nothing downstream can undo.", payload)

    def _refuse(self, rule: str, detail: str, payload: dict) -> None:
        self.refusals.append({
            "rule": rule, "detail": detail,
            "symbol": payload.get("symbol"), "account": payload.get("account"),
            "quantity": payload.get("quantity"),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        raise RiskViolation(rule, detail)

    # -- reporting --------------------------------------------------------
    def describe(self) -> str:
        engaged, reason = kill_switch_engaged(self.kill_switch)
        lines = [f"RiskFirewall  kill switch: "
                 f"{'ENGAGED — ' + reason if engaged else 'clear'}"]
        for name, value in sorted(self.limits.items()):
            lines.append(f"  {name:<28} {value}")
        if self.state is None:
            lines.append("  NOTE: no engine state — the limits that need "
                         "history (per-symbol size, open positions, orders "
                         "per session, session loss, duplicate bars) are NOT "
                         "being enforced.")
        return "\n".join(lines)
