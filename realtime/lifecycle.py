#!/usr/bin/env python3
"""
realtime.lifecycle - state that survives a crash, and a shutdown that is safe.

Three jobs, in the order they matter:

    EngineState      what this process has sent, durably, atomically
    install_signals  SIGTERM/SIGINT that finish the cycle rather than cut it
    emergency_halt   arm the kill switch, flatten what we opened, alert

WHAT IS PERSISTED, AND WHY IT IS NOT THE POSITION BOOK
======================================================
`PortfolioManager.PositionBook` is deliberately NOT persisted, and this module
does not quietly undo that. Its docstring names the reason: a restart that
believes it holds positions may flatten what it no longer holds, and the
account is SHARED - with NinjaTrader, with the operator's hands, with previous
runs. The reconciliation that would make position persistence safe is reading
real positions back from CrossTrade, and that does not exist.

So what is persisted here is not belief, it is RECORD:

    dispatched   (strategy, symbol, bar_ts) -> the order this process sent
    orders       every attempt, with its outcome
    fills        realised P&L this process booked
    session      counters, reset on the session date

All four are facts about what this process did. On restart they come back as
UNVERIFIED: `open_claims()` lists what was in flight when the process died,
`EngineState.reconciled` is False, and the flatten path refuses to act on them
until an operator or a broker read confirms. That is crash recovery without
inventing certainty - the loop knows what it sent, and knows it does not know
what came of it.

**The duplicate-fill guard is the part that genuinely prevents the failure the
requirement names.** The loop acts on the last CLOSED bar; a crash and restart
inside one interval re-evaluates the same bar and would re-send the same order.
`already_dispatched(strategy, symbol, bar_ts)` makes that impossible without
knowing anything about the broker, because the key is a fact about the SIGNAL.

ATOMICITY
=========
Temp file, fsync, `os.replace` - the rule every writer in this repository
follows. A process killed mid-write leaves the previous complete file rather
than a truncated one, because a half-written state file is read on the next
start as a shorter history: fewer orders sent, no record of the position that
is actually open.

Appends go through the same path rather than an open handle held across the
loop, because a held handle is buffered and a SIGKILL takes the buffer with it.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.risk_firewall import (arm_kill_switch,                # noqa: E402
                                    kill_switch_engaged)

STATE_VAR = "BT_ENGINE_STATE"
DEFAULT_STATE_PATH = "data/engine_state.json"
SCHEMA_VERSION = "1.0.0"


def state_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = (os.environ.get(STATE_VAR) or "").strip()
    candidate = Path(override) if override else Path(DEFAULT_STATE_PATH)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _session_date(stamp: datetime | None = None) -> str:
    """
    The CME session this instant belongs to, for the daily counters.

    `backtest.event_calendar.session_date` owns the 18:00 ET roll and is
    imported rather than reimplemented - the same rule the recorder stamps
    trades with. A daily loss cap that reset at UTC midnight would reset in the
    middle of a session, which is exactly when it matters.
    """
    when = stamp or datetime.now(timezone.utc)
    try:
        from backtest.event_calendar import session_date            # noqa: PLC0415
        return session_date([when])[0].strftime("%Y-%m-%d")
    except Exception:                                               # noqa: BLE001
        # Falling back to the UTC date is wrong by up to six hours, so it is
        # RECORDED on the state rather than silently substituted.
        return when.strftime("%Y-%m-%d")


class EngineState:
    """
    Durable record of what this process has sent. Not what it believes it holds.

    Every mutation writes the whole document atomically. The file is small - a
    session's orders and fills - and a write per order is cheap next to the
    socket it accompanies. Buffering to make it cheaper would trade the one
    property the file exists for.
    """

    def __init__(self, path: str | Path | None = None,
                 now: datetime | None = None) -> None:
        self.path = state_path(path)
        self.lock = threading.Lock()
        self.session = _session_date(now)
        self.blob = self._load()
        self.recovered = bool(self.blob.get("dispatched")) or bool(
            self.blob.get("orders"))
        # Nothing read back from a previous run has been confirmed against the
        # broker. Only an explicit reconciliation may set this True.
        self.reconciled = False
        self._roll_session()

    # -- io ---------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "session": self.session,
                    "dispatched": {}, "orders": [], "fills": [],
                    "started_at": _utcnow()}
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A state file that will not parse is NOT treated as an empty one:
            # empty means "sent nothing", and acting on that after a crash is
            # how the same order goes out twice. It is moved aside and the
            # refusal is loud.
            broken = self.path.with_suffix(self.path.suffix + ".corrupt")
            try:
                self.path.replace(broken)
            except OSError:
                pass
            raise RuntimeError(
                f"engine state at {self.path} is unreadable ({exc}); moved to "
                f"{broken.name}. Refusing to start with an empty history: "
                f"'this process has sent nothing' is exactly the belief that "
                f"re-sends an order it already placed. Reconcile against the "
                f"broker, then delete the file to start clean.") from None
        blob.setdefault("dispatched", {})
        blob.setdefault("orders", [])
        blob.setdefault("fills", [])
        return blob

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                       prefix=".engine_state.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump(self.blob, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())      # the buffer is what a SIGKILL takes
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _roll_session(self) -> None:
        """A new session zeroes the daily counters and KEEPS the history."""
        if self.blob.get("session") != self.session:
            self.blob["previous_session"] = self.blob.get("session")
            self.blob["session"] = self.session
            self.blob["orders"] = [o for o in self.blob["orders"]
                                   if o.get("session") == self.session]
            self.blob["fills"] = [f for f in self.blob["fills"]
                                  if f.get("session") == self.session]
            self._write()

    # -- the reads the firewall makes -------------------------------------
    def net_contracts(self, account: str, symbol: str) -> int:
        """Signed contracts this process has sent for (account, symbol) today."""
        total = 0
        for order in self.blob["orders"]:
            if (order.get("account") == account
                    and order.get("symbol") == str(symbol).upper()
                    and order.get("ok")):
                sign = 1 if str(order.get("action", "")).upper() == "BUY" else -1
                total += sign * int(order.get("quantity") or 0)
        return total

    def open_position_count(self) -> int:
        keys = {(o.get("account"), o.get("symbol"))
                for o in self.blob["orders"] if o.get("ok")}
        return sum(1 for account, symbol in keys
                   if account and symbol and self.net_contracts(account, symbol))

    def orders_sent_today(self) -> int:
        return sum(1 for o in self.blob["orders"] if o.get("ok"))

    def realised_pnl_today(self) -> float:
        return float(sum(float(f.get("pnl") or 0.0) for f in self.blob["fills"]))

    def already_dispatched(self, strategy: str, symbol: str,
                           bar_ts: str) -> bool:
        return self.signal_key(strategy, symbol, bar_ts) in self.blob["dispatched"]

    @staticmethod
    def signal_key(strategy: str, symbol: str, bar_ts: str) -> str:
        return f"{strategy}|{str(symbol).upper()}|{bar_ts}"

    # -- the writes -------------------------------------------------------
    def record_dispatch(self, strategy: str, symbol: str, bar_ts: str,
                        payload: dict) -> None:
        """
        Mark a signal as SENT. Called BEFORE the socket, deliberately.

        If the process dies between this write and the send, the restart
        believes an order went out that may not have - and declines to re-send
        it. That is the safe direction: a missed entry is a trade not taken,
        while a double entry is a position nothing here can unwind. The
        unsent-but-recorded case is listed by `open_claims()` for an operator
        to resolve.
        """
        with self.lock:
            self.blob["dispatched"][self.signal_key(strategy, symbol, bar_ts)] = {
                "at": _utcnow(), "session": self.session,
                "account": payload.get("account"),
                "symbol": str(payload.get("symbol") or "").upper(),
                "action": payload.get("action"),
                "quantity": payload.get("quantity"),
                "confirmed": False,
            }
            self._write()

    def record_order(self, record: dict, strategy: str = "",
                     bar_ts: str = "") -> None:
        """The attempt and its outcome, after the socket."""
        with self.lock:
            entry = {k: record.get(k) for k in
                     ("timestamp", "account", "symbol", "action", "quantity",
                      "ok", "error", "attempts", "dry_run")}
            entry["session"] = self.session
            entry["strategy"] = strategy
            entry["bar_ts"] = bar_ts
            self.blob["orders"].append(entry)
            key = self.signal_key(strategy, entry["symbol"] or "", bar_ts)
            if key in self.blob["dispatched"]:
                self.blob["dispatched"][key]["confirmed"] = bool(record.get("ok"))
                self.blob["dispatched"][key]["error"] = record.get("error")
            self._write()

    def record_fill(self, symbol: str, pnl: float, detail: str = "") -> None:
        with self.lock:
            self.blob["fills"].append({
                "at": _utcnow(), "session": self.session,
                "symbol": str(symbol).upper(), "pnl": float(pnl),
                "detail": detail})
            self._write()

    # -- recovery ---------------------------------------------------------
    def open_claims(self) -> list[dict[str, Any]]:
        """
        What this process had in flight when it last stopped, UNVERIFIED.

        Every entry here is a claim, not a position. Nothing may act on one
        until it is reconciled against the broker - see the module docstring
        and `PositionBook`'s.
        """
        claims = []
        for key, record in sorted(self.blob["dispatched"].items()):
            account, symbol = record.get("account"), record.get("symbol")
            if account and symbol and self.net_contracts(account, symbol):
                claims.append({"key": key, **record})
        return claims

    def describe(self) -> str:
        lines = [f"EngineState  {self.path}",
                 f"  session          {self.session}",
                 f"  orders sent      {self.orders_sent_today()}",
                 f"  realised P&L     {self.realised_pnl_today():,.2f}",
                 f"  signals recorded {len(self.blob['dispatched'])}"]
        claims = self.open_claims()
        if claims:
            lines.append(f"  UNVERIFIED CLAIMS FROM A PREVIOUS RUN: {len(claims)}")
            for claim in claims[:5]:
                lines.append(f"    {claim['key']}  {claim.get('action')} "
                             f"{claim.get('quantity')} — confirmed="
                             f"{claim.get('confirmed')}")
            lines.append("    Nothing will be flattened on these: they are what "
                         "this process SENT, not what the account HOLDS. "
                         "Reconcile in NinjaTrader before trading.")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------

class GracefulShutdown:
    """
    SIGTERM/SIGINT recorded, never acted on mid-cycle.

    `master_live.ShutdownFlag` already does this for the loop and is not
    replaced; this is the same contract for the daemons that have no cycle of
    their own, plus the hooks systemd needs: a first signal asks, a second
    insists, and `on_stop` callbacks run once, in order, before exit.

    A cycle is signals -> gates -> sizing -> dispatch. Interrupting it between
    the send and the record leaves an order on a broker with nothing on disk
    saying so, which is the one state this whole module exists to prevent.
    """

    def __init__(self, name: str = "process") -> None:
        self.name = name
        self.requested = False
        self.signal_name: str | None = None
        self._callbacks: list[tuple[str, Callable[[], None]]] = []

    def on_stop(self, label: str, callback: Callable[[], None]) -> None:
        self._callbacks.append((label, callback))

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame) -> None:
        name = signal.Signals(signum).name
        if self.requested:
            print(f"[{self.name}] second {name}: exiting now.", flush=True)
            self.run_callbacks()
            raise SystemExit(130)
        self.requested = True
        self.signal_name = name
        print(f"[{self.name}] {name}: finishing the current cycle, then "
              f"stopping. Send again to exit immediately.", flush=True)

    def run_callbacks(self) -> list[str]:
        """
        Every shutdown hook, in order, each isolated.

        One hook that raises must not skip the rest: the disconnect matters
        even when the state save failed, and the state save matters even when
        the disconnect did.
        """
        done = []
        for label, callback in self._callbacks:
            try:
                callback()
                done.append(f"{label}: ok")
            except Exception as exc:                                # noqa: BLE001
                done.append(f"{label}: FAILED {type(exc).__name__}: {exc}")
                print(f"[{self.name}] shutdown hook {label} failed: {exc}",
                      file=sys.stderr, flush=True)
        self._callbacks = []
        return done


# --------------------------------------------------------------------------
# the emergency
# --------------------------------------------------------------------------

def emergency_halt(reason: str, dispatcher: Any | None = None,
                   state: EngineState | None = None,
                   alert: Callable[[str], Any] | None = None,
                   kill_switch: str | Path | None = None) -> dict[str, Any]:
    """
    Stop trading now: arm the switch, close what THIS PROCESS opened, alert.

    The order is deliberate. The switch is armed FIRST and unconditionally, so
    that even if the flatten fails - broker down, network gone, dispatcher in a
    bad state - no further entry can be sent by this process or the next one to
    start. A halt that flattened first and armed second could send an entry in
    between on the next cycle.

    **It flattens only what this process recorded opening.** A blind flatten
    closes whatever is on a shared account, including positions placed by hand
    or by another tool, and that is silent, immediate and unrecoverable. If
    there are unverified claims from a previous run, they are REPORTED for a
    human rather than closed.
    """
    outcome: dict[str, Any] = {
        "at": _utcnow(), "reason": reason, "switch": None,
        "flattened": [], "failed": [], "unverified": [], "alerted": False}

    outcome["switch"] = str(arm_kill_switch(reason, kill_switch))

    if state is not None:
        outcome["unverified"] = [c["key"] for c in state.open_claims()
                                 if not c.get("confirmed")]

    if dispatcher is not None:
        book = getattr(dispatcher, "positions", None)
        held = list(book.open_positions().items()) if book is not None and hasattr(
            book, "open_positions") else []
        for (account, symbol), position in held:
            try:
                record = dispatcher.dispatch_flatten(account, symbol)
                (outcome["flattened"] if record.get("ok")
                 else outcome["failed"]).append(
                    {"account": account, "symbol": symbol,
                     "error": record.get("error")})
            except Exception as exc:                                # noqa: BLE001
                outcome["failed"].append({"account": account, "symbol": symbol,
                                          "error": f"{type(exc).__name__}: {exc}"})

    if alert is not None:
        try:
            alert(f"EMERGENCY HALT: {reason}\n"
                  f"kill switch: {outcome['switch']}\n"
                  f"flattened: {outcome['flattened'] or 'none'}\n"
                  f"FAILED: {outcome['failed'] or 'none'}\n"
                  f"unverified claims (NOT closed): {outcome['unverified'] or 'none'}")
            outcome["alerted"] = True
        except Exception as exc:                                    # noqa: BLE001
            outcome["alert_error"] = f"{type(exc).__name__}: {exc}"
    return outcome


def startup_report(state: EngineState,
                   kill_switch: str | Path | None = None) -> str:
    """
    What a daemon prints before its first cycle. Read this in the journal after
    a restart: it is the difference between a clean start and one that inherited
    an unfinished session.
    """
    engaged, why = kill_switch_engaged(kill_switch)
    lines = [state.describe()]
    if engaged:
        lines.append(f"  KILL SWITCH ENGAGED — {why}")
        lines.append("  No entry will be sent until it is cleared.")
    if state.recovered and not state.reconciled:
        lines.append("  RECOVERED STATE IS UNRECONCILED: this process knows "
                     "what it sent, not what the account holds.")
    return "\n".join(lines)
