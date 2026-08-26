#!/usr/bin/env python3
"""
realtime/check_live_signals.py - is anything actually being evaluated?

Location:  ~/src/trading/realtime/check_live_signals.py

    python3 realtime/check_live_signals.py         # one card
    python3 realtime/check_live_signals.py --watch 20
    signal-check / live-signals                    # the aliases

`nt8-check` answers "are bars arriving". This answers the next question: given
that they are, is anything DECIDING anything with them - which strategy is
loaded, what environment the gate thinks it is in, whether entries are
permitted, and what the firewall has counted today.

WHAT IS PERSISTED, AND WHAT IS NOT
----------------------------------
This reads state that other processes WROTE. It computes no indicator and
re-evaluates no gate, and the distinction matters more here than in any other
status tool in this repository, because the numbers involved look exactly the
same whether they were read or recomputed.

Available, and reported:

    data/live_regime_state.json   the live quadrant per symbol, and the
                                  SWITCHBOARD - per strategy: status, reason,
                                  a plain-English detail, entries_allowed,
                                  exits_allowed, the certified quadrant, and
                                  the ADX/ATR the gate was drawn on
    data/engine_state.json        the session's dispatches, orders and fills
    config/portfolios.json        allocation, target account, basket
    approved_incubator/<id>/      meta.json: the promoted risk parameters and
                                  the certification behind them
    data/KILL_SWITCH              armed or clear, with its reason
    realtime.risk_firewall        the caps every order is checked against

NOT available, and therefore NOT printed:

  * **The strategy's own indicators** - fast RSI, slow RSI, MACD histogram and
    the rest. Nothing persists them. This tool COULD load bars and recompute
    them, and must not: the engine evaluates on the bars IT loaded, with its
    own warm-up and its own last-closed-bar rule, and a second computation
    here would differ at exactly the boundaries that matter and would carry a
    status tool's authority while doing it. What IS shown is ADX/ATR, because
    those are read back from the file the daemon wrote rather than recomputed.
  * **A signal line for a symbol with no strategy.** Bars arrive for contracts
    nothing trades; those are not evaluated, so they have no verdict. Printing
    `FLAT` beside them would report an evaluation that never ran. They are
    listed instead under what the feed carries that nothing is watching, which
    is the true and more useful statement.

LATENCY AND PER-STRATEGY VERDICTS DO EXIST, IN THE LOG
------------------------------------------------------
`master_live.py` itself has no `perf_counter`, which is why an earlier version
of this file declared latency unavailable. That was wrong: the timing lives in
`LiveExecutionDispatcher.describe_cycle`, which prints one block per cycle to
`logs/master_live.log`:

    [2026-08-26T15:21:21+00:00] cycle DRY RUN  symbols=3  strategies=1  458.6ms
           HOLD t3_braid_scalp_20260823_NQ_1h MNQ - MNQ (regime read from NQ)
           is in Q0 ...; Incubator-Odd trades ['Q1','Q2','Q3','Q4']. Standing
           down rather than trading the environment nobody certified.

So the elapsed milliseconds and the per-strategy verdict - with the
dispatcher's own plain-English reason - are both recorded, and are parsed back
out here rather than restated. `OK`, `SKIP`, `VETO`, `HOLD`, `ERROR` and `EXIT`
are `describe_cycle`'s vocabulary, not this file's.

That log is EMPTY on a box where the loop has never been armed, which is the
normal state here. An absent cycle block is reported as absent; the switchboard
still says what the gate WOULD decide, and the two are labelled differently
because one is a decision that was taken and the other is one that would be.

OFFLINE IS A NORMAL STATE
-------------------------
`trading-master-live` ships as `--dry-run` and is deliberately not armed. A
stopped loop with a live feed is this box's resting posture, not an outage, so
a missing `data/engine_state.json` is reported as "no session recorded" with
the age of whatever last wrote - never as an error, and never as zero orders,
which is a different claim.

COST
----
No pandas, no engine, no lake reader. `risk_firewall` (6ms) and `lifecycle`
(11ms) ARE imported: both are standard-library only, and one spelling of the
caps and of the state path beats a local copy that drifts.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. `$BT_ENGINE_STATE`
# and `$BT_KILL_SWITCH` decide which files this reports on, and a tool that
# resolved them before the file was loaded would describe the wrong ones. The
# rules live in ONE module rather than a block copied into every runner: see
# mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # `python3 realtime/x.py` puts realtime/ on sys.path, not the repository
    # root, so mdlib is not importable until this runs.
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

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

REPO = PROJECT_ROOT

# Both are standard-library only - measured 6.4ms and 10.7ms - so the caps and
# the state path have ONE spelling rather than a copy here that drifts the
# first time somebody raises a limit.
from realtime.risk_firewall import DEFAULT_LIMITS, kill_switch_path  # noqa: E402
from realtime.lifecycle import state_path                          # noqa: E402
from realtime.contract_alias import micros_of                      # noqa: E402
# The spool's location and filename shape, from the tool that already owns
# them. Importing a sibling status card rather than restating two constants:
# it is 32ms, pulls nothing heavy, and `nt8-check` and `signal-check`
# disagreeing about which directory the feed writes to would be a fault that
# reads as two correct answers.
from realtime.check_nt8_feed import (                              # noqa: E402
    resolve_spool, spool_stats)

W = 80

PORTFOLIO_CONFIG = REPO / "config" / "portfolios.json"
REGIME_STATE = REPO / "data" / "live_regime_state.json"
INCUBATOR = REPO / "strategies" / "approved_incubator"
MASTER_LOG = REPO / "logs" / "master_live.log"

#: How stale the regime file may be before the gate it describes is not a
#: statement about now. The daemon republishes every 5 minutes, so twice that
#: is a missed cycle rather than a slow one.
REGIME_STALE_SECONDS = 600.0


# ==========================================================================
# formatting
# ==========================================================================

def parse_iso(text: Any) -> float | None:
    if not text:
        return None
    s = str(text).strip().replace("Z", "+00:00")
    # The regime file writes `bar_ts` as a pandas stamp - `2026-08-26
    # 14:00:00+00:00`, a space rather than a T - which `fromisoformat` accepts
    # on 3.11+. Normalised anyway so a reader is not relying on that.
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def utc_text(epoch: float | None) -> str:
    if epoch is None:
        return "unknown"
    return datetime.fromtimestamp(epoch, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC")


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
    A reading that is allowed to be absent.

    The same rule `regime_daemon._fmt_opt` follows, and for the same reason:
    `adx_14` and `atr_14` are None all through the ADX warm-up, and
    `f"{None:.2f}"` raises TypeError. That exact line took the regime daemon
    down 163 times on 2026-08-26; a status card is not going to repeat it.
    """
    if value is None:
        return missing
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def read_json(path: Path) -> dict[str, Any] | None:
    """A state file, or None when absent, truncated or not a mapping."""
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


def wrap(text: str, indent: int = 21, width: int = W) -> list[str]:
    """Wrap a long `detail` under its label rather than letting it run off."""
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


def short(path: Path) -> str:
    """
    A path relative to the repo when it is inside it, absolute otherwise.

    `Path.relative_to` RAISES for anything outside the tree, and these paths
    are overridable: `$BT_ENGINE_STATE` and `$BT_KILL_SWITCH` may point at
    /var/lib or anywhere else. A bare `.relative_to(REPO)` therefore turns a
    supported configuration into a traceback from a status card.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def row(label: str, value: str) -> str:
    return f"{label:<19}: {value}"


# ==========================================================================
# the loop
# ==========================================================================

def engine_process() -> dict[str, Any]:
    """
    Is `master_live.py` running, and in which mode.

    Read off the ARGUMENTS, not the command name: every one of these runs as
    `python3 master_live.py`, and under systemd the shipped unit carries
    `--dry-run`. Mode is taken from the live command line rather than from the
    unit file, because an operator who edited the unit to arm the loop and an
    operator who did not are exactly what this line has to tell apart.
    """
    info: dict[str, Any] = {"pids": [], "dry_run": None, "args": None,
                            "unit_active": None, "ps_ok": False}
    me = os.getpid()
    try:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                             text=True, timeout=10, check=False)
        info["ps_ok"] = out.returncode == 0
        for line in out.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            pid, args = parts
            if "master_live.py" not in args:
                continue
            if "check_live_signals" in args or int(pid) == me:
                continue
            info["pids"].append(int(pid))
            info["args"] = args
            info["dry_run"] = "--dry-run" in args
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    try:
        act = subprocess.run(
            ["systemctl", "is-active", "trading-master-live.service"],
            capture_output=True, text=True, timeout=8, check=False)
        info["unit_active"] = act.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return info


#: `LiveExecutionDispatcher.describe_cycle`'s header and its verdict lines.
#: Matched rather than reconstructed: the vocabulary belongs to that method,
#: and a copy here that drifted would silently stop recognising a verdict and
#: report a cycle as having decided nothing.
CYCLE_HEADER_RE = re.compile(
    r"^\[([^\]]+)\]\s+cycle\s+(DRY RUN|LIVE)\s+symbols=(\d+)\s+"
    r"strategies=(\d+)\s+([\d.]+)ms\s*$")
CYCLE_VERDICTS = ("OK", "FAIL", "SKIP", "VETO", "HOLD", "ERROR", "EXIT")


def last_cycle() -> dict[str, Any]:
    """
    The most recent cycle block from the loop's own log.

    `describe_cycle` emits a header carrying the elapsed milliseconds and then
    one indented line per outcome. Both are parsed back rather than restated:
    the reasons are the DISPATCHER's, written when the decision was taken, and
    a status tool paraphrasing them would be offering its own account of a
    decision it did not make.

    An empty or absent log means the loop has never run here, which is this
    box's normal state and is reported as such.
    """
    out: dict[str, Any] = {
        "exists": MASTER_LOG.exists(), "mtime": mtime(MASTER_LOG), "bytes": 0,
        "started_at": None, "mode": None, "symbols": None, "strategies": None,
        "elapsed_ms": None, "verdicts": [], "bar_line": None, "bar_ts": None,
    }
    if not out["exists"]:
        return out
    try:
        out["bytes"] = MASTER_LOG.stat().st_size
        with MASTER_LOG.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            back = min(fh.tell(), 65536)
            fh.seek(-back, os.SEEK_END)
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    # The newest bar line, wherever it sits - it is printed once per cycle.
    for line in reversed(lines):
        if "newest closed bar" in line:
            out["bar_line"] = line.strip()
            m = re.search(r"newest closed bar (\S+ \S+)", line)
            if m:
                out["bar_ts"] = parse_iso(m.group(1))
            break

    # The LAST header, and every verdict line under it. Scanning backwards for
    # the header and then forwards for its body, so a truncated tail that cuts
    # a block in half yields the previous complete one rather than a mix.
    start = None
    for i in range(len(lines) - 1, -1, -1):
        if CYCLE_HEADER_RE.match(lines[i].strip()):
            start = i
            break
    if start is None:
        return out

    m = CYCLE_HEADER_RE.match(lines[start].strip())
    out["started_at"] = parse_iso(m.group(1))
    out["mode"] = m.group(2)
    out["symbols"] = int(m.group(3))
    out["strategies"] = int(m.group(4))
    out["elapsed_ms"] = float(m.group(5))

    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped or CYCLE_HEADER_RE.match(stripped):
            break
        if stripped.startswith("[master_live]") or stripped.startswith("["):
            break
        kind = stripped.split(None, 1)[0] if stripped.split() else ""
        if kind not in CYCLE_VERDICTS:
            # `no orders — every net position was declined:` and similar
            # narration. Kept as context rather than dropped, so a reader sees
            # what the dispatcher saw.
            out["verdicts"].append({"kind": "NOTE", "text": stripped})
            continue
        out["verdicts"].append({
            "kind": kind,
            "text": stripped[len(kind):].strip()})
    return out


def engine_state() -> dict[str, Any]:
    """The session counters, or an honest absence."""
    path = state_path()
    out: dict[str, Any] = {"path": path, "exists": path.exists(),
                           "mtime": mtime(path), "blob": None,
                           "orders": None, "fills_pnl": None,
                           "dispatched": None, "session": None}
    if not out["exists"]:
        return out
    blob = read_json(path)
    if blob is None:
        out["error"] = "unreadable or mid-write"
        return out
    out["blob"] = blob
    out["session"] = blob.get("session")
    orders = blob.get("orders") or []
    fills = blob.get("fills") or []
    out["orders"] = sum(1 for o in orders if o.get("ok"))
    out["orders_total"] = len(orders)
    out["fills_pnl"] = sum(float(f.get("pnl") or 0.0) for f in fills)
    out["dispatched"] = len(blob.get("dispatched") or {})
    return out


def kill_switch() -> dict[str, Any]:
    path = kill_switch_path()
    out: dict[str, Any] = {"path": path, "armed": path.exists(),
                           "text": None, "mtime": mtime(path)}
    if out["armed"]:
        try:
            out["text"] = path.read_text(encoding="utf-8").strip()[:200]
        except OSError:
            pass
    return out


# ==========================================================================
# strategies
# ==========================================================================

def strategies() -> dict[str, Any]:
    """
    Every allocated strategy, with the routing table's view and the gate's.

    Three files meet here and none of them is authoritative alone:
    `portfolios.json` says what is ALLOCATED and to which account,
    `approved_incubator/<id>/meta.json` says what was PROMOTED (the risk
    parameters, the certification), and the switchboard in the regime state
    says what the gate DECIDED on the last bar. An id present in one and
    absent from another is a real and invisible fault - a strategy allocated
    but never certified, or certified and never republished - so each is
    reported as found or missing rather than merged into a single row.
    """
    out: dict[str, Any] = {"rows": [], "errors": []}
    cfg = read_json(PORTFOLIO_CONFIG)
    if cfg is None:
        out["errors"].append(f"{PORTFOLIO_CONFIG} is unreadable")
        return out
    regime = read_json(REGIME_STATE) or {}
    switchboard = regime.get("strategies") or {}
    symbols_block = regime.get("symbols") or {}

    for pname, p in sorted((cfg.get("portfolios") or {}).items()):
        allocations = p.get("strategy_allocations") or {}
        for sid in (p.get("active_strategies") or []):
            alloc = allocations.get(sid) or {}
            gate = switchboard.get(sid)
            meta = read_json(INCUBATOR / sid / "meta.json")
            symbol = (gate or {}).get("symbol") or alloc.get("symbol")
            tf = (gate or {}).get("timeframe") or alloc.get("timeframe")
            live = symbols_block.get(str(symbol or "").upper()) or {}
            out["rows"].append({
                "id": sid,
                "portfolio": pname,
                "account": p.get("target_account"),
                "account_type": p.get("account_type"),
                "allocation": alloc.get("allocation"),
                "alloc_status": alloc.get("status"),
                "symbol": symbol,
                "timeframe": tf,
                "micros": list(micros_of(symbol)) if symbol else [],
                "basket": list((p.get("basket") or {}).get("assets") or []),
                "meta": meta,
                "gate": gate,
                "live": live,
            })
    return out


def unwatched_feed_symbols(strategy_rows: list[dict]) -> list[str]:
    """
    Contracts the SPOOL carries that no allocated strategy looks at.

    Read from the spool on disk, not from the regime state. The daemon only
    classifies its registered targets, so the regime file lists one symbol
    while the feed carries ten - and sourcing this from the regime file would
    compare a set against itself and report nothing, every time, which is
    exactly as wrong as it is quiet.

    Not a fault: an operator may be warming a tape deliberately. But it is the
    difference between "the feed is healthy" and "the feed is healthy AND
    something is watching it", and only one of those is a reason to leave the
    box alone.
    """
    watched = {str(r.get("symbol") or "").upper() for r in strategy_rows}
    for r in strategy_rows:
        watched.update(m.upper() for m in r.get("micros") or [])
    directory, _source = resolve_spool(None)
    arriving = {f["symbol"].upper() for f in spool_stats(directory)["files"]}
    return sorted(arriving - watched)


# ==========================================================================
# the card
# ==========================================================================

def collect() -> dict[str, Any]:
    rows = strategies()
    return {
        "now": time.time(),
        "engine": engine_process(),
        "cycle": last_cycle(),
        "state": engine_state(),
        "kill": kill_switch(),
        "strategies": rows,
        "regime_mtime": mtime(REGIME_STATE),
        "regime": read_json(REGIME_STATE),
        "unwatched": unwatched_feed_symbols(rows["rows"]),
        "limits": dict(DEFAULT_LIMITS),
    }


def render(snap: dict[str, Any]) -> str:
    now = snap["now"]
    eng, cyc, st = snap["engine"], snap["cycle"], snap["state"]
    L: list[str] = ["=" * W, "STRATEGY & SIGNAL EVALUATION STATUS", "=" * W]

    # ---- the loop ----------------------------------------------------
    if eng["pids"]:
        mode = "DRY RUN (formatted, nothing sent)" if eng["dry_run"] \
            else "LIVE EXECUTION — orders will be sent"
        L.append(row("Engine Mode", mode))
        L.append(row("Cycle Status",
                     f"ACTIVE (PID: {', '.join(str(p) for p in eng['pids'])})"))
    else:
        L.append(row("Engine Mode", "IDLE — master_live.py is not running"))
        L.append(row("Cycle Status",
                     "STOPPED. The shipped unit is --dry-run and is not armed "
                     "by default;"))
        L.append(f"{'':<19}  a stopped loop under a live feed is this box's "
                 f"resting state.")
        if eng.get("unit_active"):
            L.append(row("Unit State", f"trading-master-live: {eng['unit_active']}"))
        if not eng["ps_ok"]:
            L.append(row("Note", "the process table could not be read, so a "
                                 "running loop would not have been seen."))

    if cyc["bar_ts"]:
        L.append(row("Last Evaluated Bar",
                     f"{utc_text(cyc['bar_ts'])} "
                     f"({age_text(now - cyc['bar_ts'])} ago)"))
    elif cyc["bar_line"]:
        L.append(row("Last Evaluated Bar", cyc["bar_line"]))
    else:
        L.append(row("Last Evaluated Bar",
                     "no cycle recorded — logs/master_live.log is "
                     + ("empty" if cyc["exists"] else "absent")))

    if cyc["elapsed_ms"] is not None:
        L.append(row("Evaluation Latency",
                     f"{cyc['elapsed_ms']:.1f} ms "
                     f"({cyc['mode']}, {cyc['symbols']} symbol(s), "
                     f"{cyc['strategies']} strateg"
                     f"{'y' if cyc['strategies'] == 1 else 'ies'})"))
        L.append(row("Last Cycle Ran",
                     f"{utc_text(cyc['started_at'])} "
                     f"({age_text(now - cyc['started_at'])} ago)"
                     if cyc["started_at"] else "unknown"))
    else:
        # Stated, not left blank: an absent latency reads as a fast one. The
        # number is the dispatcher's own `elapsed_ms`, printed once per cycle;
        # with no cycle in the log there is nothing to report and nothing to
        # infer.
        L.append(row("Evaluation Latency",
                     "no cycle logged — nothing has been timed"))

    # ---- strategies --------------------------------------------------
    L.append("")
    L.append("--- Active Strategies & Portfolio Allocation ---")
    rows = snap["strategies"]["rows"]
    for err in snap["strategies"]["errors"]:
        L.append(f"  [!] {err}")
    if not rows:
        L.append("  none — no portfolio lists an active strategy, so nothing "
                 "is being evaluated.")
    for r in rows:
        gate, meta, live = r["gate"], r["meta"], r["live"]
        L.append(f"  • Strategy       : {r['id']}")
        micro = f" (Execution: {', '.join(r['micros'])})" if r["micros"] else ""
        L.append(f"    Target Symbol  : {r['symbol']}{micro}")
        L.append(f"    Timeframe      : {r['timeframe']}"
                 + (f" (resampled from the 1m spool)" if r["timeframe"] != "1m"
                    else ""))
        L.append(f"    Portfolio      : {r['portfolio']} "
                 f"(account {r['account']} · {r['account_type']}) "
                 f"allocation {r['allocation']} · {r['alloc_status']}")

        if meta is None:
            L.append(f"    Promoted Risk  : NOT FOUND — no "
                     f"approved_incubator/{r['id']}/meta.json. The routing "
                     f"table allocates a strategy that was never promoted.")
        else:
            risk = meta.get("risk") or {}
            L.append(f"    SL / TP        : SL={num(risk.get('sl_atr_mult'), '.1f')} ATR, "
                     f"TP={num(risk.get('tp_atr_mult'), '.1f')} ATR, "
                     f"trailing={risk.get('trailing')}")
            L.append(f"    Certified Tape : {meta.get('symbols')} at "
                     f"{meta.get('timeframe')} · version {meta.get('version')}")

        if gate is None:
            L.append("    Strategy Gate  : NOT ON THE SWITCHBOARD — the regime "
                     "daemon has not published a status for this id.")
            L.append("                     Entries are not permitted by "
                     "anything that reads this file.")
        else:
            L.append(f"    Certified For  : {gate.get('optimal_regime')} "
                     f"({gate.get('optimal_regime_label')})")
            L.append(f"    Market Regime  : {gate.get('live_quadrant')} "
                     f"({gate.get('live_regime')}) · "
                     f"ADX={num(gate.get('adx_14'))} "
                     f"ATR={num(gate.get('atr_14'), '.4f')} "
                     f"theta={num(gate.get('theta_vol'), '.4f')}")
            L.append(f"    Strategy Gate  : {gate.get('status')} "
                     f"({gate.get('reason')}) — entries "
                     f"{'ALLOWED' if gate.get('entries_allowed') else 'BLOCKED'}, "
                     f"exits "
                     f"{'ALLOWED' if gate.get('exits_allowed') else 'BLOCKED'}")
            for ln in wrap(gate.get("detail") or "", indent=21):
                L.append(f"{'':<21}{ln}")
            bar = parse_iso(gate.get("bar_ts"))
            L.append(f"    Gate Drawn On  : bar {utc_text(bar)} "
                     f"({age_text(now - bar)} ago)" if bar else
                     "    Gate Drawn On  : unknown")
            if live.get("n_bars") is not None:
                L.append(f"    Bars Seen      : {live.get('n_bars')} at "
                         f"{live.get('tf')}")

    # ---- risk --------------------------------------------------------
    L.append("")
    L.append("--- Risk & Order Firewall Heartbeat ---")
    kill = snap["kill"]
    if kill["armed"]:
        L.append(row("Kill Switch",
                     f"ARMED — NO NEW ORDER WILL BE SENT ({kill['path']})"))
        for ln in wrap(kill["text"] or "(no reason recorded)", indent=21):
            L.append(f"{'':<21}{ln}")
    else:
        L.append(row("Kill Switch", "CLEAR (no orders are being blocked by it)"))

    lim = snap["limits"]
    if not st["exists"]:
        L.append(row("Session Record",
                     f"none — {short(st['path'])} does not exist, "
                     f"so no session has run."))
        L.append(f"{'':<19}  That is NOT the same as zero orders; nothing has "
                 f"been counted.")
    elif st.get("error"):
        L.append(row("Session Record", f"{st['path'].name}: {st['error']}"))
    else:
        L.append(row("Session Date", str(st.get("session") or "unknown")))
        L.append(row("Realised P&L", f"${st['fills_pnl']:,.2f} USD "
                                     f"(this loop's fills only)"))
        L.append(row("Orders Dispatched",
                     f"{st['orders']} confirmed of {st.get('orders_total', 0)} "
                     f"attempted / {lim['max_orders_per_session']} max per "
                     f"session"))
        L.append(row("Signals Dispatched", f"{st['dispatched']} this session"))
        L.append(row("State Written", f"{age_text(now - st['mtime'])} ago"
                     if st["mtime"] else "unknown"))
    L.append(row("Firewall Caps",
                 f"{lim['max_open_positions']} open positions · "
                 f"{lim['max_contracts_per_order']} contracts/order · "
                 f"${lim['max_session_loss_usd']:,.0f} session loss"))

    # ---- evaluation summary ------------------------------------------
    L.append("")
    L.append("--- Signal Evaluation Summary ---")
    reg_age = (now - snap["regime_mtime"]) if snap["regime_mtime"] else None

    if cyc["verdicts"]:
        # What the loop ACTUALLY DECIDED, in the dispatcher's own words. This
        # outranks the gate view below: the gate says what would be permitted,
        # this says what was done about it.
        L.append(f"  Last cycle's decisions ({cyc['mode']}, "
                 f"{age_text(now - cyc['started_at'])} ago):"
                 if cyc["started_at"] else "  Last cycle's decisions:")
        for v in cyc["verdicts"]:
            head = f"  {v['kind']:<6}" if v["kind"] != "NOTE" else "        "
            body = wrap(v["text"], indent=10)
            L.append(f"{head}{body[0] if body else ''}")
            for ln in body[1:]:
                L.append(f"{'':<8}{ln}")
        L.append("")
        L.append("  The gate's current view (what would be permitted on the "
                 "latest published bar):")

    if not rows:
        L.append("  nothing is allocated, so nothing is evaluated.")
    for r in rows:
        gate = r["gate"]
        if gate is None:
            L.append(f"  {str(r['symbol']):<4} ({r['timeframe']}): "
                     f"NOT EVALUATED  [no switchboard entry]")
            continue
        allowed = gate.get("entries_allowed")
        verdict = "ENTRIES PERMITTED" if allowed else "HOLD / NO NEW ENTRY"
        L.append(f"  {str(r['symbol']):<4} ({r['timeframe']}): {verdict:<19} "
                 f"[{gate.get('status')} · {gate.get('reason')}]")
    if reg_age is not None and reg_age > REGIME_STALE_SECONDS:
        L.append("")
        L.append(f"  [!] the regime file was written {age_text(reg_age)} ago. "
                 f"The daemon republishes every")
        L.append(f"      5 minutes, so these verdicts describe a bar that is "
                 f"no longer current.")

    if snap["unwatched"]:
        L.append("")
        L.append(f"  Feed symbols no allocated strategy watches "
                 f"({len(snap['unwatched'])}):")
        for ln in wrap(", ".join(snap["unwatched"]), indent=6):
            L.append(f"      {ln}")
        # NOT "classified". The daemon classifies only its REGISTERED targets,
        # so these are ingested to the spool and nothing more - no quadrant is
        # computed for them and no gate is drawn. Saying "classified" would
        # credit the stack with work it did not do.
        L.append("  They are spooled and nothing else: no quadrant is computed "
                 "for them and no gate is")
        L.append("  drawn, so they have no verdict here. That is expected "
                 "unless one was meant to trade.")

    L.append("=" * W)
    return "\n".join(L)


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="What the live engine is evaluating, and what the gate is "
                    "letting through. Reads state other processes wrote; "
                    "computes no indicator and sends nothing.")
    ap.add_argument("--watch", type=float, nargs="?", const=20.0, default=None,
                    metavar="SECONDS", help="redraw every SECONDS (default 20)")
    ap.add_argument("--json", action="store_true",
                    help="print the collected state instead of the card")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def once() -> tuple[str, bool]:
        snap = collect()
        running = bool(snap["engine"]["pids"])
        if args.json:
            return json.dumps(snap, indent=2, default=str), running
        return render(snap), running

    if args.watch is None:
        text, running = once()
        print(text)
        # 0 when the loop is running, 1 when it is not - so the alias chains.
        # An IDLE loop is a legitimate resting state on this box, so this exit
        # code means "is it evaluating", NOT "is something wrong".
        return 0 if running else 1

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
