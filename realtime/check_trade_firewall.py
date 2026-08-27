#!/usr/bin/env python3
"""
realtime/check_trade_firewall.py - why is nothing trading?

Location:  ~/src/trading/realtime/check_trade_firewall.py

    python3 realtime/check_trade_firewall.py
    python3 realtime/check_trade_firewall.py --watch 30
    firewall-check / trade-gate                    # the aliases

`nt8-check` says bars are arriving. `signal-check` says what the loop decided.
This says WHY an entry cannot happen, layer by layer, and ends with the list of
blockers in the order they bite.

Six things can stop an entry and they fail in different places, so a single
"not trading" verdict sends an operator to the wrong one:

    1  the execution interlock   `--dry-run`: orders are formatted, not sent
    2  the global kill switch    a file; halts every order this process would send
    3  the session firewall      caps on size, count, exposure and realised loss
    4  the execution bridge      no webhook configured means a LIVE loop refuses
                                 to start at all
    5  the regime gate           the strategy's certified quadrant vs the live one
    6  the feed                  no fresh bars means nothing is evaluated

THE INTERLOCK IS CHECKED IN THREE PLACES, AND THAT IS THE POINT
---------------------------------------------------------------
`--dry-run` can be present in the repo unit, in what systemd would actually
run, and in the process that is running now - and those three can disagree:

  * `systemctl edit --full` writes an override under /etc/systemd/system/ that
    SHADOWS the repo unit. It survives `deploy/redeploy.sh` and git never sees
    it, so the tracked unit can say `--dry-run` while the installed one does
    not.
  * a unit edited but not restarted leaves the OLD flag on the running
    process, so the tracked config, the installed config and reality are three
    different answers.

Each disagreement is reported by name rather than collapsed into one verdict.
A card that read only the repo file would have told an operator they were in
dry run while a live loop was sending orders.

WHAT THIS DOES NOT DO
---------------------
**It does not probe the webhook.** "Reachable" for CrossTrade means POSTing to
an endpoint whose entire purpose is placing orders on a funded account. A
status tool that pings it to see whether it answers is a status tool that can
open a position. Configuration is verified - the variables are set, the URL
parses, the host is named - and the first REAL request is left to the loop,
under the interlock and the kill switch, where it belongs.

**It does not report open positions from disk, because they are not there.**
`PositionBook` is deliberately not persisted and `EngineState` records what was
SENT, not what the account HOLDS. Only the running process knows its own
exposure, and NinjaTrader knows the account's. Printing `0 / 4` from an absent
file would be a claim about an account this tool cannot see.

**It computes no indicator and re-runs no gate.** Every number here was written
by the process that made the decision.

COST
----
No pandas, no engine. `risk_firewall` (6.6ms) and `lifecycle` (10.7ms) are
imported for the caps and the state path - both standard-library only.
`regime_daemon`, `live_dispatcher` and `mdlib.regimes` are NOT: they cost
253ms, 2.7s and 268ms respectively. The three constants needed from them are
restated below and pinned against the originals by
`tests/test_check_trade_firewall.py`.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. The CrossTrade
# variables are what layer 4 checks for, and mdlib/env.py is the one module
# that knows they are withheld from the environment rather than exported.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import (                                            # noqa: E402
    DEFAULT_ENV_FILE, ENV_FILE_VAR, NO_EXPORT, load_env)

load_env()
# ---------------------------------------------------------------------------

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
import os                                                          # noqa: E402
import re                                                          # noqa: E402
import subprocess                                                  # noqa: E402
import time                                                        # noqa: E402
from datetime import datetime, timezone                            # noqa: E402
from typing import Any                                             # noqa: E402
from urllib.parse import urlparse                                  # noqa: E402

REPO = PROJECT_ROOT

from realtime.risk_firewall import (                               # noqa: E402
    DEFAULT_LIMITS, kill_switch_path)
from realtime.lifecycle import state_path                          # noqa: E402
from realtime.check_nt8_feed import resolve_spool, spool_stats     # noqa: E402

W = 80
UNIT = "trading-master-live.service"
REPO_UNIT = REPO / "deploy" / "systemd" / "trading-master-live.service"
REGIME_STATE = REPO / "data" / "live_regime_state.json"
MASTER_LOG = REPO / "logs" / "master_live.log"
PORTFOLIO_CONFIG = REPO / "config" / "portfolios.json"

#: Restated from `mdlib.regimes` and `realtime.regime_daemon`, which cost 268ms
#: and 253ms to import because both pull pandas. Pinned against the originals
#: by `test_the_restated_constants_match_their_sources`, so the copy is checked
#: rather than trusted - a warm-up threshold that drifted would report a
#: strategy as ready eleven bars early.
ADX_LENGTH = 14
MIN_BARS_FOR_REGIME = 2 * ADX_LENGTH + 1        # 29

#: Restated from `realtime.live_dispatcher` (2.7s to import). Pinned likewise.
ENV_URL = "CROSSTRADE_WEBHOOK_URL"
ENV_KEY = "CROSSTRADE_API_KEY"

#: A spooled contract older than this has stopped feeding the loop. The
#: listener's own flag is 3 bar widths; this is the coarser "the tape is gone"
#: threshold, and only the streams a strategy needs are judged on it.
FEED_STALE_SECONDS = 600.0

PASS = "PASS"
BLOCKED = "BLOCKED"
UNKNOWN = "UNKNOWN"


# ==========================================================================
# formatting
# ==========================================================================

def age_text(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        h, m = divmod(int(seconds // 60), 60)
        return f"{h}h {m:02d}m"
    d, h = divmod(int(seconds // 3600), 24)
    return f"{d}d {h:02d}h"


def num(value: Any, spec: str = ".2f", missing: str = "n/a") -> str:
    """
    A reading that may be absent, formatted WITHOUT EVER RAISING.

    `f"{None:.2f}"` raises TypeError. That expression took the regime daemon
    down 163 times on 2026-08-26 - publishing correctly, then dying while
    formatting its own log line. Every branch here returns a string.
    """
    if value is None:
        return missing
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def short(path: Path) -> str:
    """Relative to the repo when inside it, absolute otherwise - never raises."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def wrap(text: str, indent: int = 6, width: int = W) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for word in words:
        if cur and len(cur) + 1 + len(word) > width - indent:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def row(label: str, value: str) -> str:
    return f"{label:<23}: {value}"


# ==========================================================================
# Layer 1 - the execution interlock, in three places
# ==========================================================================

def interlock() -> dict[str, Any]:
    """
    Whether the loop is ARMED, asked of git, of systemd, of the process, and
    of the loop's own banner.

    IT LOOKS FOR `--live`, NOT FOR THE ABSENCE OF `--dry-run`, and that
    distinction cost a deployment on 2026-08-27.
    `master_live.resolve_dry_run` returns `not args.live`: dry run is the
    DEFAULT, `--live` is the only thing that clears it, and a unit carrying
    NEITHER flag sends no order. The first version of this function inferred
    "armed" from a missing `--dry-run` and therefore reported
    `LIVE EXECUTION — the running loop will send orders` about a loop that was
    formatting payloads into a socket it never opened. A card built to answer
    "why is nothing trading" told an operator the opposite of the truth.

    FOUR sources, deliberately not merged. The first three are configuration
    and can each be edited without the others; the fourth is the loop's OWN
    verdict, printed by `LiveExecutionDispatcher.describe()` after
    `resolve_dry_run` has run, and it is the only one that reflects what the
    process actually decided. Where they disagree, that is the finding.
    """
    out: dict[str, Any] = {"repo": None, "effective": None, "running": None,
                           "banner": None, "banner_at": None,
                           "drop_ins": None, "pids": [], "unit_active": None,
                           "agree": None}

    def armed(text: str | None) -> bool | None:
        """
        `--live` present, as a whole flag.

        A substring test would match `master_live.py` on every command line
        this reads and report every dry run as armed.
        """
        if text is None:
            return None
        return re.search(r"(?<![\w-])--live(?![\w-])", text) is not None

    def exec_start(unit_text: str) -> str:
        """
        The ExecStart DIRECTIVE only, with its line continuations joined.

        Reading the whole file was wrong and wrong in the dangerous direction:
        this unit's comments explain what `--live` does, so a whole-file match
        reported the repo unit as ARMED while its ExecStart carried no such
        flag. A card that mistakes documentation for configuration is the same
        error as one that mistakes a missing flag for an armed loop.
        """
        joined, capture = [], False
        for raw in unit_text.splitlines():
            line = raw.strip()
            if line.startswith("#"):
                continue
            if line.startswith("ExecStart"):
                capture = True
            if capture:
                joined.append(line.rstrip("\\").strip())
                if not line.endswith("\\"):
                    break
        return " ".join(joined)

    try:
        out["repo"] = armed(exec_start(REPO_UNIT.read_text(encoding="utf-8")))
    except OSError:
        pass

    try:
        # `systemctl show -p ExecStart` resolves DROP-INS, so this is what
        # would run on the next restart - not what the repo file says.
        res = subprocess.run(["systemctl", "show", UNIT, "-p", "ExecStart",
                              "--value"], capture_output=True, text=True,
                             timeout=8, check=False)
        if res.returncode == 0 and res.stdout.strip():
            out["effective"] = armed(res.stdout)
        drops = subprocess.run(["systemctl", "show", UNIT, "-p", "DropInPaths",
                                "--value"], capture_output=True, text=True,
                               timeout=8, check=False)
        out["drop_ins"] = [d for d in (drops.stdout or "").split() if d]
        act = subprocess.run(["systemctl", "is-active", UNIT],
                             capture_output=True, text=True, timeout=8,
                             check=False)
        out["unit_active"] = act.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass

    me = os.getpid()
    try:
        ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                            text=True, timeout=10, check=False)
        for line in ps.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            pid, args = parts
            if "master_live.py" not in args or int(pid) == me:
                continue
            if "check_trade_firewall" in args:
                continue
            out["pids"].append(int(pid))
            out["running"] = armed(args)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    # THE LOOP'S OWN VERDICT. `describe()` prints `mode=LIVE` or
    # `mode=DRY RUN` after `resolve_dry_run` has decided, so this is the only
    # source that reports what the PROCESS concluded rather than what its
    # command line looks like. When it disagrees with the flags, believe it.
    try:
        with MASTER_LOG.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            back = min(fh.tell(), 262144)
            fh.seek(-back, os.SEEK_END)
            for line in reversed(fh.read().decode("utf-8", "replace")
                                 .splitlines()):
                if "LiveExecutionDispatcher  mode=" in line:
                    out["banner"] = "DRY RUN" not in line
                    out["banner_at"] = mtime(MASTER_LOG)
                    break
    except OSError:
        pass

    answers = [v for v in (out["repo"], out["effective"], out["running"],
                           out["banner"]) if v is not None]
    out["agree"] = (len(set(answers)) <= 1) if answers else None
    return out


# ==========================================================================
# Layer 2 - the kill switch
# ==========================================================================

def kill_switch() -> dict[str, Any]:
    path = kill_switch_path()
    out: dict[str, Any] = {"path": path, "armed": path.exists(),
                           "text": None, "mtime": mtime(path)}
    if out["armed"]:
        try:
            out["text"] = path.read_text(encoding="utf-8").strip()[:300]
        except OSError:
            out["text"] = "(armed, but the file could not be read)"
    return out


# ==========================================================================
# Layer 3 - the session firewall
# ==========================================================================

def session() -> dict[str, Any]:
    """
    The counters the firewall checks each order against.

    OPEN POSITIONS ARE ABSENT ON PURPOSE. `PositionBook` is deliberately not
    persisted and `EngineState` records what was SENT, not what the account
    HOLDS - so exposure is knowable only inside the running process, and the
    account's true position only in NinjaTrader. A `0 / 4` printed from an
    absent file would be a claim about an account this tool cannot see.
    """
    path = state_path()
    out: dict[str, Any] = {"path": path, "exists": path.exists(),
                           "mtime": mtime(path), "session": None,
                           "orders_ok": None, "orders_attempted": None,
                           "realised": None, "dispatched": None,
                           "error": None, "limits": dict(DEFAULT_LIMITS)}
    if not out["exists"]:
        return out
    blob = read_json(path)
    if blob is None:
        out["error"] = "unreadable or mid-write"
        return out
    orders = blob.get("orders") or []
    fills = blob.get("fills") or []
    out["session"] = blob.get("session")
    out["orders_ok"] = sum(1 for o in orders if o.get("ok"))
    out["orders_attempted"] = len(orders)
    out["realised"] = sum(float(f.get("pnl") or 0.0) for f in fills)
    out["dispatched"] = len(blob.get("dispatched") or {})
    return out


# ==========================================================================
# Layer 4 - the execution bridge
# ==========================================================================

def _credential_sources() -> dict[str, str]:
    """
    The CrossTrade credentials, resolved THE WAY THE DISPATCHER RESOLVES THEM.

    They are NOT in `os.environ` on a correctly configured box, and that is
    deliberate: `mdlib.env.NO_EXPORT` withholds them because anything in the
    environment is inherited by every subprocess, which is how a webhook key -
    the credential for a funded account - reaches an unrelated tool's debug
    output.

    `realtime.live_dispatcher` reads them with
    `env.get(NAME) or os.environ.get(NAME)`: the .env FILE first, the
    environment second. That precedence is mirrored here rather than guessed
    at, and pinned against the real `load_env_file` by the tests. Checking
    `os.environ` alone - which the first version of this file did - reports
    NOT SET on a box that is correctly configured, and would have told an
    operator their bridge was down while it was fine.

    Values are never returned to the caller beyond "is it set", and never
    printed. Only the parsed host reaches the card.
    """
    from dotenv import dotenv_values                          # noqa: PLC0415

    path = Path(os.environ.get(ENV_FILE_VAR) or DEFAULT_ENV_FILE)
    out: dict[str, Any] = {"_source": {}}
    file_values: dict[str, str | None] = {}
    if path.is_file():
        try:
            file_values = dotenv_values(path)
        except Exception:                                     # noqa: BLE001
            file_values = {}
    for name in (ENV_URL, ENV_KEY):
        from_file = (file_values.get(name) or "").strip()
        from_env = (os.environ.get(name) or "").strip()
        value = from_file or from_env
        out[name] = value
        out["_source"][name] = (
            f"{short(path)}" if from_file else
            "the environment" if from_env else None)
    return out


def bridge() -> dict[str, Any]:
    """
    Whether a LIVE loop could dispatch - CONFIGURATION ONLY, never a request.

    `LiveExecutionDispatcher` refuses to construct with no webhook URL, so an
    unset variable is not a warning here: it is a live loop that will not
    start. What is checked is that the variables are set and the URL parses.

    Nothing is sent. "Reachable" for this endpoint means POSTing to the thing
    that places orders on a funded account, and a status tool that pings it to
    see whether it answers is a status tool that can open a position.
    """
    values = _credential_sources()
    url = values.get(ENV_URL, "")
    key = values.get(ENV_KEY, "")
    out: dict[str, Any] = {"url_set": bool(url), "key_set": bool(key),
                           "host": None, "scheme": None, "parse_error": None,
                           "source": values.get("_source", {})}
    if not url:
        return out
    try:
        parsed = urlparse(url)
        out["host"] = parsed.hostname
        out["scheme"] = parsed.scheme
        if not parsed.scheme or not parsed.hostname:
            out["parse_error"] = "not an absolute URL"
    except ValueError as e:
        out["parse_error"] = f"{type(e).__name__}: {e}"
    return out


# ==========================================================================
# Layer 5 - the regime gate, per allocated strategy
# ==========================================================================

def gates() -> dict[str, Any]:
    """Every allocated strategy, with the switchboard's verdict and warm-up."""
    out: dict[str, Any] = {"rows": [], "error": None,
                           "regime_mtime": mtime(REGIME_STATE)}
    cfg = read_json(PORTFOLIO_CONFIG)
    if cfg is None:
        out["error"] = f"{short(PORTFOLIO_CONFIG)} is unreadable"
        return out
    regime = read_json(REGIME_STATE) or {}
    switchboard = regime.get("strategies") or {}
    symbols = regime.get("symbols") or {}

    for pname, p in sorted((cfg.get("portfolios") or {}).items()):
        for sid in (p.get("active_strategies") or []):
            gate = switchboard.get(sid)
            sym = (gate or {}).get("symbol")
            live = symbols.get(str(sym or "").upper()) or {}
            n_bars = live.get("n_bars")
            out["rows"].append({
                "id": sid, "portfolio": pname, "gate": gate,
                "symbol": sym, "timeframe": (gate or {}).get("timeframe"),
                "n_bars": n_bars,
                "warm": (None if n_bars is None
                         else int(n_bars) >= MIN_BARS_FOR_REGIME),
                "short_by": (None if n_bars is None
                             else max(0, MIN_BARS_FOR_REGIME - int(n_bars))),
            })
    return out


# ==========================================================================
# Layer 6 - the feed
# ==========================================================================

def feed(rows: list[dict]) -> dict[str, Any]:
    """Whether the streams the allocated strategies need are still arriving."""
    directory, source = resolve_spool(None)
    stats = spool_stats(directory)
    needed: set[str] = set()
    for r in rows:
        if r.get("symbol"):
            needed.add(str(r["symbol"]).upper())
    now = time.time()
    streams = []
    for f in stats.get("files", []):
        if f["symbol"].upper() not in needed:
            continue
        age = now - f["mtime"]
        streams.append({**f, "age": age, "stale": age > FEED_STALE_SECONDS})
    return {"dir": directory, "source": source, "exists": stats["exists"],
            "needed": sorted(needed), "streams": streams,
            "missing": sorted(needed - {s["symbol"].upper() for s in streams})}


# ==========================================================================
# assembly
# ==========================================================================

def collect() -> dict[str, Any]:
    g = gates()
    return {"now": time.time(), "interlock": interlock(),
            "kill": kill_switch(), "session": session(), "bridge": bridge(),
            "gates": g, "feed": feed(g["rows"])}


def blockers(snap: dict[str, Any]) -> list[str]:
    """
    Every reason an entry cannot happen, in the order the stack applies them.

    ORDER MATTERS. The interlock is first because it makes every layer below
    it moot: a dry-run loop evaluates gates, sizes orders and sends nothing, so
    "the regime is wrong" is true and irrelevant until the flag is off.
    """
    out: list[str] = []
    il, kill, ses, br = (snap["interlock"], snap["kill"], snap["session"],
                         snap["bridge"])

    armed_now = (il["banner"] if il["banner"] is not None else il["running"])
    if not il["pids"]:
        out.append("No loop is running, so nothing is being evaluated at all.")
    elif armed_now is False:
        out.append("The RUNNING loop is in DRY RUN: it formats payloads and "
                   "opens no socket. `--live` is what arms it — REMOVING "
                   "`--dry-run` does nothing, because dry run is the default.")
    elif armed_now is None:
        out.append("Could not establish whether the loop is armed: neither "
                   "its flags nor its banner could be read.")

    if il["agree"] is False:
        out.append(f"The interlock DISAGREES across sources — repo="
                   f"{il['repo']}, systemd={il['effective']}, "
                   f"running={il['running']}, loop-says={il['banner']}. The "
                   f"loop's own banner is the one to believe; resolve this "
                   f"before trusting any other line on this card.")

    if kill["armed"]:
        out.append(f"The kill switch is ARMED ({short(kill['path'])}). Every "
                   f"new order is refused until it is cleared.")

    if not br["url_set"]:
        out.append(f"${ENV_URL} is not set. A LIVE loop REFUSES TO START "
                   f"without it — this is not a warning.")
    elif br["parse_error"]:
        out.append(f"${ENV_URL} is set but {br['parse_error']}.")
    if br["url_set"] and not br["key_set"]:
        out.append(f"${ENV_KEY} is not set alongside the webhook URL.")

    lim = ses["limits"]
    if ses["exists"] and not ses.get("error"):
        if (ses["orders_ok"] or 0) >= lim["max_orders_per_session"]:
            out.append(f"Session order cap reached: {ses['orders_ok']} of "
                       f"{lim['max_orders_per_session']}.")
        loss_cap = -abs(float(lim["max_session_loss_usd"]))
        if (ses["realised"] or 0.0) <= loss_cap:
            out.append(f"Session loss cap reached: "
                       f"${ses['realised']:,.2f} against ${loss_cap:,.2f}. "
                       f"Stop for the day; do not raise the cap.")

    for r in snap["gates"]["rows"]:
        gate = r["gate"]
        if gate is None:
            out.append(f"{r['id']} is allocated but has NO switchboard entry — "
                       f"the regime daemon has published no status for it.")
            continue
        if r["warm"] is False:
            out.append(f"Indicator warm-up incomplete for {r['symbol']} "
                       f"{r['timeframe']}: {r['n_bars']}/{MIN_BARS_FOR_REGIME} "
                       f"bars, short by {r['short_by']}.")
        elif not gate.get("entries_allowed"):
            out.append(f"{r['id']} gate is {gate.get('status')} "
                       f"({gate.get('reason')}): live "
                       f"{gate.get('live_quadrant')} against certified "
                       f"{gate.get('optimal_regime')}.")

    for sym in snap["feed"]["missing"]:
        out.append(f"No spool file for {sym}: the strategy that needs it "
                   f"cannot be evaluated.")
    for s in snap["feed"]["streams"]:
        if s["stale"]:
            out.append(f"{s['symbol']} spool has not been written for "
                       f"{age_text(s['age'])}; the tape has stopped.")
    return out


def render(snap: dict[str, Any]) -> str:
    now = snap["now"]
    il, kill, ses, br = (snap["interlock"], snap["kill"], snap["session"],
                         snap["bridge"])
    stops = blockers(snap)
    L: list[str] = ["=" * W, "LIVE TRADE FIREWALL & ENTRY GATE DIAGNOSTICS",
                    "=" * W]

    if stops:
        L.append(row("Overall Trading Status",
                     f"BLOCKED — {len(stops)} reason(s), listed at the bottom"))
    else:
        L.append(row("Overall Trading Status",
                     "ACTIVE — every layer permits an entry"))

    # The BANNER is preferred over the flags: it is what the process concluded,
    # and the flags are only what it was asked.
    effective_mode = il["banner"] if il["banner"] is not None else il["running"]
    if not il["pids"]:
        mode = (f"no loop running (unit: {il['unit_active'] or 'unknown'}); "
                f"systemd would start it "
                + ("LIVE" if il["effective"] else "in DRY RUN — `--live` is "
                                                 "absent, which is the default"))
    elif effective_mode is True:
        mode = "LIVE EXECUTION — the running loop WILL send orders"
    elif effective_mode is False:
        mode = ("DRY RUN — the loop formats payloads and opens no socket "
                "(`--live` absent; that is the default)")
    else:
        mode = "UNKNOWN — could not read the flags or the loop's banner"
    L.append(row("Execution Mode", mode))
    L.append(row("Kill Switch State",
                 f"TRIPPED ({short(kill['path'])})" if kill["armed"]
                 else "CLEAR"))

    # ---- layers ------------------------------------------------------
    L.append("")
    L.append("--- Protective Firewall Layers ---")

    armed_now = (il["banner"] if il["banner"] is not None else il["running"])
    verdict = PASS if armed_now is True else BLOCKED
    L.append(f"1. Execution Interlock    : {verdict}")
    for label, val in (("repo unit", il["repo"]),
                       ("systemd (effective)", il["effective"]),
                       ("running process", il["running"])):
        text = ("--live PRESENT (armed)" if val is True else
                "--live ABSENT (dry run — the default)" if val is False
                else "not readable")
        L.append(f"   • {label:<22} {text}")
    banner = ("mode=LIVE" if il["banner"] is True else
              "mode=DRY RUN" if il["banner"] is False else "not found")
    L.append(f"   • {'the loop says':<22} {banner}"
             + ("   <-- believe this one" if il["banner"] is not None else ""))
    if il["drop_ins"]:
        L.append(f"   • drop-in overrides     {', '.join(il['drop_ins'])}")
        L.append("     these SHADOW the repo unit and git does not see them")
    if il["agree"] is False:
        L.append("   • [!] THESE DISAGREE — see the summary below")

    L.append(f"2. Global Kill Switch     : "
             f"{BLOCKED if kill['armed'] else PASS}")
    if kill["armed"]:
        for ln in wrap(kill["text"] or "(no reason recorded)", indent=6):
            L.append(f"     {ln}")

    lim = ses["limits"]
    if not ses["exists"]:
        L.append(f"3. Session Risk Firewall  : {UNKNOWN} — no session recorded")
        L.append(f"   • {short(ses['path'])} does not exist, so nothing has "
                 f"been counted.")
        L.append("     That is NOT the same as zero orders.")
    elif ses.get("error"):
        L.append(f"3. Session Risk Firewall  : {UNKNOWN} — {ses['error']}")
    else:
        L.append(f"3. Session Risk Firewall  : {PASS}")
        L.append(f"   • Session date           {ses['session']}")
        L.append(f"   • Orders dispatched      {ses['orders_ok']} confirmed of "
                 f"{ses['orders_attempted']} attempted / "
                 f"{lim['max_orders_per_session']} max")
        L.append(f"   • Realised P&L           "
                 f"${ses['realised']:,.2f} / "
                 f"-${float(lim['max_session_loss_usd']):,.0f} cap")
        L.append(f"   • Signals dispatched     {ses['dispatched']}")
    L.append(f"   • Max contracts/order    {lim['max_contracts_per_order']}  "
             f"per symbol {lim['max_contracts_per_symbol']}")
    L.append(f"   • Max open positions     {lim['max_open_positions']} "
             f"(current exposure NOT knowable here — see below)")

    bridge_ok = br["url_set"] and br["key_set"] and not br["parse_error"]
    L.append(f"4. Execution Bridge       : {PASS if bridge_ok else BLOCKED}")
    src = br.get("source") or {}
    L.append(f"   • ${ENV_URL:<24} "
             f"{'set' if br['url_set'] else 'NOT SET'}"
             + (f" → {br['scheme']}://{br['host']}" if br["host"] else "")
             + (f"  [from {src[ENV_URL]}]" if src.get(ENV_URL) else ""))
    L.append(f"   • ${ENV_KEY:<24} "
             f"{'set' if br['key_set'] else 'NOT SET'}"
             + (f"  [from {src[ENV_KEY]}]" if src.get(ENV_KEY) else ""))
    L.append(f"   • withheld from os.environ by design "
             f"({', '.join(sorted(NO_EXPORT))})")
    L.append("   • configuration only — this tool never POSTs to the endpoint "
             "that places orders")

    # ---- strategy gates ----------------------------------------------
    L.append("")
    L.append("--- Strategy & Regime Gate Breakdown ---")
    if snap["gates"]["error"]:
        L.append(f"  [!] {snap['gates']['error']}")
    if not snap["gates"]["rows"]:
        L.append("  nothing is allocated, so no gate can open.")
    for r in snap["gates"]["rows"]:
        gate = r["gate"]
        L.append(f"  • {r['id']}")
        L.append(f"    Symbol / TF     : {r['symbol']} @ {r['timeframe']} "
                 f"({r['portfolio']})")
        if gate is None:
            L.append("    Gate Status     : NO SWITCHBOARD ENTRY — the daemon "
                     "has published nothing for this id")
            continue
        L.append(f"    Gate Status     : "
                 f"{'OPEN' if gate.get('entries_allowed') else 'BLOCKED'} "
                 f"({gate.get('status')} · {gate.get('reason')}) — entries "
                 f"{'ALLOWED' if gate.get('entries_allowed') else 'BLOCKED'}, "
                 f"exits "
                 f"{'ALLOWED' if gate.get('exits_allowed') else 'BLOCKED'}")
        if r["n_bars"] is not None:
            bar = ("complete" if r["warm"]
                   else f"short by {r['short_by']} (~{r['short_by']}h at "
                        f"{r['timeframe']})")
            L.append(f"    Warm-up         : {r['n_bars']} / "
                     f"{MIN_BARS_FOR_REGIME} bars — {bar}")
        L.append(f"    Market Regime   : {gate.get('live_quadrant')} "
                 f"({gate.get('live_regime')}) · certified for "
                 f"{gate.get('optimal_regime')} "
                 f"({gate.get('optimal_regime_label')})")
        L.append(f"    Indicators      : ADX={num(gate.get('adx_14'))} "
                 f"ATR={num(gate.get('atr_14'), '.4f')} "
                 f"theta={num(gate.get('theta_vol'), '.4f')}")

    # ---- feed --------------------------------------------------------
    fd = snap["feed"]
    if fd["needed"]:
        L.append("")
        L.append("--- Feed For The Allocated Strategies ---")
        for s in sorted(fd["streams"], key=lambda x: x["symbol"]):
            L.append(f"  • {s['symbol']:<5} {s['name']:<14} written "
                     f"{age_text(s['age'])} ago"
                     + ("  [STALE]" if s["stale"] else ""))
        for sym in fd["missing"]:
            L.append(f"  • {sym:<5} NO SPOOL FILE")

    # ---- root cause --------------------------------------------------
    L.append("")
    L.append("--- Root Cause Summary (Why Trades Are Not Firing) ---")
    if not stops:
        L.append("  Nothing is blocking an entry. The next qualifying signal "
                 "will be sent.")
    for i, reason in enumerate(stops, 1):
        body = wrap(reason, indent=14)
        L.append(f"  {i}. [BLOCKER] {body[0] if body else ''}")
        for ln in body[1:]:
            L.append(f"{'':<14}{ln}")
    if stops:
        L.append("")
        L.append("  Listed in the order the stack applies them: the interlock "
                 "makes every layer")
        L.append("  below it moot, so clear them top down.")

    L.append("=" * W)
    return "\n".join(L)


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Why a live entry cannot happen, layer by layer. Reads "
                    "state other processes wrote; sends nothing and never "
                    "probes the order endpoint.")
    ap.add_argument("--watch", type=float, nargs="?", const=30.0, default=None,
                    metavar="SECONDS", help="redraw every SECONDS (default 30)")
    ap.add_argument("--json", action="store_true",
                    help="print the collected state instead of the card")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def once() -> tuple[str, int]:
        snap = collect()
        n = len(blockers(snap))
        if args.json:
            return json.dumps({**snap, "blockers": blockers(snap)},
                              indent=2, default=str), n
        return render(snap), n

    if args.watch is None:
        text, n = once()
        print(text)
        # 0 when NOTHING blocks an entry, 1 when something does. So
        # `firewall-check && ...` runs only on a stack that would actually
        # trade - and on this box, in dry run, that is deliberately non-zero.
        return 0 if n == 0 else 1

    try:
        while True:
            text, _ = once()
            print("\033[2J\033[H", end="")
            print(text, flush=True)
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
