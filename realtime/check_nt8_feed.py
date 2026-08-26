#!/usr/bin/env python3
"""
realtime/check_nt8_feed.py - is NinjaTrader still feeding this box?

Location:  ~/src/trading/realtime/check_nt8_feed.py

    python3 realtime/check_nt8_feed.py            # one card
    python3 realtime/check_nt8_feed.py --watch 15
    nt8-check / feed-status                       # the aliases

WHAT THIS ANSWERS THAT /health DOES NOT
---------------------------------------
`curl localhost:8000/health` already reports the listener's own view, and the
runbook's pre-flight pipes it through `json.tool`. That is the RECEIVER's
answer to "am I getting bars", and it is necessarily blind to three things a
person standing in front of this box actually needs:

  * **whether the process is even there.** A connection refused and a listener
    reporting STARVED look identical through `curl` unless you read the exit
    code, and they are opposite findings: one is a dead service, the other is
    a live service with a quiet publisher at the far end.
  * **whether the streams arriving are the streams the LIVE STACK NEEDS.** The
    listener keys on whatever it is sent. A spool full of healthy bars for
    contracts nothing trades is a healthy listener and a broken feed, and only
    a reader that knows `config/portfolios.json` can tell the difference.
  * **whether the rest of the chain moved.** The regime state file and the
    watchdog are downstream of these bars. Bars arriving while the regime file
    has not been rewritten in an hour is a different fault from no bars at all.

So this reads `/health` as its primary source and then checks it against the
process table, the spool on disk, the routing table and the two files
downstream of the feed.

WHAT IT REFUSES TO DECIDE
-------------------------
**Whether the market is open.** `nt8_bar_listener.health()` says so in its own
docstring: STALE "is also what a shut market looks like ... the watchdog owns
the session calendar that separates them." A second, cheaper session guess
here would be a second calendar to disagree with the first, and it would
disagree exactly at the session edges where somebody is looking. A stream past
its window is reported as stale with its age; the watchdog line lower down is
what says whether that is expected at this hour.

**Anything about order flow.** This reads. It sends nothing, writes nothing,
and opens no file under `/mnt/backtest/raw`.

THRESHOLDS, AND WHY THERE ARE TWO
---------------------------------
The listener flags a stream stale after `stale_after_bars` (3.0) x the bar
width - 3 minutes on a 1m stream. This tool's own escalation is 5 minutes, per
the operator brief. Those disagree by design over a 2-minute band, so BOTH are
shown: a stream inside this tool's window that the listener has already flagged
prints `listener: STALE` beside its status. Two tools on one box quietly
contradicting each other about the same stream is the failure this note exists
to prevent.

COST
----
No pandas, no engine, no lake reader. Measured on this box: `pandas` alone is
247ms to import, against 18ms for the standard library this needs.
`backtest.pipeline` (16ms, dotenv only) IS imported, for `split_strategy_id`
and nothing else - a second spelling of how `<strategy>_<SYMBOL>_<TF>` comes
apart is precisely the drift that module exists to prevent.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. `$BT_NT8_SPOOL`
# decides which directory this reports on, and a tool that resolved it before
# the file was loaded would confidently describe the wrong spool. The rules -
# repository root from __file__ rather than the working directory, existing
# variables winning over the file, CrossTrade credentials withheld - live in
# ONE module rather than in a block copied into every runner: see mdlib/env.py.
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

# `contract_alias` is 0.3ms and stdlib-only; `pipeline` is 16ms and pulls
# dotenv. Both are cheap enough that ONE spelling of their rules beats a local
# copy. `backtest.specs` is NOT imported - it pulls numpy and pandas.
from realtime.contract_alias import micros_of, resolve_parent      # noqa: E402
from backtest.pipeline import split_strategy_id                    # noqa: E402

W = 80
DEFAULT_URL = "http://localhost:8000/health"
DEFAULT_PORT = 8000

#: The listener's default, copied from `realtime.nt8_feed.DEFAULT_SPOOL_DIR`
#: rather than imported - that module pulls pandas. It is only ever the LAST
#: fallback: `/health` reports the directory the listener is actually writing
#: to, and that answer is preferred over this one wherever it is available.
FALLBACK_SPOOL_DIR = Path("/mnt/backtest/artifacts/nt8_bars")
SPOOL_DIR_VAR = "BT_NT8_SPOOL"

#: The suffixes `realtime.nt8_feed.SPOOL_SUFFIXES` reads, and the shape
#: `spool_path` builds. Restated rather than imported for the same reason as
#: the directory above - that module pulls pandas - and pinned against the real
#: constant by `tests/test_check_nt8_feed.py`, so the copy cannot drift
#: silently.
SPOOL_SUFFIXES = (".csv", ".txt", ".tsv", ".jsonl")
SPOOL_NAME_RE = r"([A-Z0-9]+)_(\d+[a-z])(?:" + "|".join(
    re.escape(x) for x in SPOOL_SUFFIXES) + ")"

#: The operator brief's escalation. Deliberately LARGER than the listener's own
#: 3-bar window, so the two disagree over a band rather than at a point; see
#: the module docstring.
DEFAULT_STALE_SECONDS = 300.0

PORTFOLIO_CONFIG = REPO / "config" / "portfolios.json"
REGIME_STATE = REPO / "data" / "live_regime_state.json"
WATCHDOG_LOG = REPO / "logs" / "watchdog.log"
HELPERS_SH = REPO / "deploy" / "shell" / "trading_helpers.sh"

STREAMING = "STREAMING"
IDLE = "IDLE / OK"
STALE = "STALE / WAITING"

_TF_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


# ==========================================================================
# small helpers
# ==========================================================================

def tf_seconds(tf: str) -> float | None:
    """`1m` -> 60.0. None for anything this does not recognise."""
    m = re.fullmatch(r"(\d+)\s*([mhdw])", str(tf or "").strip().lower())
    if not m:
        return None
    return float(m.group(1)) * _TF_SECONDS[m.group(2)]


def parse_iso(text: Any) -> float | None:
    """An ISO-8601 stamp (`Z` or offset) as an epoch, or None."""
    if not text:
        return None
    s = str(text).strip().replace("Z", "+00:00")
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
    """`12s`, `4.2 min`, `2h 11m`, `3d 04h`."""
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


def size_text(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    return f"{size:.1f} GB"


def truncated(items, keep: int = 11) -> str:
    items = list(items)
    if not items:
        return "none"
    if len(items) <= keep:
        return ", ".join(items)
    return ", ".join(items[:keep]) + f", ... (+{len(items) - keep} more)"


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
# the listener
# ==========================================================================

def fetch_health(url: str = DEFAULT_URL,
                 timeout: float = 4.0) -> dict[str, Any]:
    """
    `/health`, as a verdict rather than as an exception.

    Returns `{"ok", "code", "payload", "error"}`. A 503 is NOT an error here:
    the endpoint answers 503 while STARVED, which is a live listener that has
    received nothing yet - and after every restart, for up to one bar width.
    Treating a non-2xx as a failure would report a healthy restart as an
    outage, which is the exact mistake the runbook warns an uptime monitor
    against.

    Connection refused, DNS failure and timeout are reported by NAME rather
    than as a traceback: "the listener is not running" and "the listener did
    not answer in 4s" send you to different places.

    `urllib` is imported HERE rather than at module scope. It costs 24ms and
    drags in `http.client`, and `realtime/check_live_signals.py` imports this
    module for `resolve_spool`/`spool_stats` alone - two filesystem helpers
    that have no business pulling in an HTTP stack. This tool pays the 24ms on
    every run either way, so nothing is lost by paying it here.
    """
    import urllib.error                                       # noqa: PLC0415
    import urllib.request                                     # noqa: PLC0415

    out: dict[str, Any] = {"ok": False, "code": None, "payload": None,
                           "error": None, "url": url}
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["code"] = resp.getcode()
            body = resp.read()
    except urllib.error.HTTPError as e:
        # 503/STARVED arrives here. The body is still the health document.
        out["code"] = e.code
        try:
            body = e.read()
        except Exception:                                     # noqa: BLE001
            body = b""
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        name = type(reason).__name__ if not isinstance(reason, str) else "URLError"
        if isinstance(reason, ConnectionRefusedError) or \
                "refused" in str(reason).lower():
            out["error"] = "connection refused"
        elif "timed out" in str(reason).lower():
            out["error"] = f"no answer within {timeout:.0f}s"
        else:
            out["error"] = f"{name}: {reason}"
        return out
    except Exception as e:                                    # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    try:
        payload = json.loads(body or b"")
    except (ValueError, UnicodeDecodeError) as e:
        out["error"] = f"the endpoint answered {out['code']} but not JSON: {e}"
        return out
    if not isinstance(payload, dict):
        out["error"] = f"the endpoint answered {out['code']} with a non-object"
        return out
    out["payload"] = payload
    out["ok"] = True
    return out


def listener_process(port: int = DEFAULT_PORT) -> dict[str, Any]:
    """
    Who holds the port, and whether systemd thinks it owns them.

    Asked of the SOCKET (`ss`), not of `pgrep`. `pgrep -f` matches whole
    command lines, so any shell whose argv happens to mention the script - a
    wrapper, an editor, the command grepping for it - counts as a listener.
    `deploy/redeploy.sh` learned this the hard way and asks `ss` for the same
    reason; a status tool and the deploy script disagreeing about whether the
    port is held would be worse than either being wrong alone.
    """
    info: dict[str, Any] = {"pids": [], "unit_active": None, "unit_pid": None,
                            "source": None}
    try:
        out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=8,
                             check=False)
        if out.returncode == 0:
            info["pids"] = sorted({int(p) for p in
                                   re.findall(r"pid=(\d+)", out.stdout)})
            info["source"] = "ss"
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    try:
        act = subprocess.run(
            ["systemctl", "is-active", "trading-nt8-listener.service"],
            capture_output=True, text=True, timeout=8, check=False)
        info["unit_active"] = act.stdout.strip() or None
        mp = subprocess.run(
            ["systemctl", "show", "trading-nt8-listener.service",
             "-p", "MainPID", "--value"],
            capture_output=True, text=True, timeout=8, check=False)
        pid = (mp.stdout or "").strip()
        info["unit_pid"] = int(pid) if pid.isdigit() and pid != "0" else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return info


# ==========================================================================
# the spool
# ==========================================================================

def resolve_spool(health: dict[str, Any] | None) -> tuple[Path, str]:
    """
    Which directory to report on, and how that was decided.

    `/health` names the directory the listener is ACTUALLY writing to, so it
    wins whenever the listener is answering. Falling back to the environment
    when a live answer exists would let this tool describe one spool while the
    listener fills another - the two differ exactly when somebody has just
    changed `$BT_NT8_SPOOL` and not restarted the unit, which is the moment
    the question gets asked.
    """
    if health:
        named = str(health.get("spool_dir") or "").strip()
        if named:
            return Path(named), "reported by the listener"
    override = (os.environ.get(SPOOL_DIR_VAR) or "").strip()
    if override:
        return Path(override), f"${SPOOL_DIR_VAR}"
    return FALLBACK_SPOOL_DIR, "the built-in default"


def spool_stats(directory: Path) -> dict[str, Any]:
    """
    What is on disk: one entry per spool file, plus the total.

    `mtime` is when the file was last APPENDED TO, which is not the bar's
    timestamp - a distinction the readout keeps, because a spool written at
    14:21 carrying a 13:00 bar is a real and confusing state and the two
    numbers are what tell you it is happening.
    """
    out: dict[str, Any] = {"exists": directory.is_dir(), "files": [],
                           "total_bytes": 0, "unparsed": []}
    if not out["exists"]:
        return out
    try:
        entries = sorted(directory.iterdir())
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    for p in entries:
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out["total_bytes"] += st.st_size
        m = re.fullmatch(SPOOL_NAME_RE, p.name)
        if not m:
            # Reported, not skipped. A name this does not recognise is how a
            # publisher rolled onto a physical contract shows up - see the
            # RUNBOOK's rollover section - and a silent skip is what let it
            # go unnoticed for a session.
            out["unparsed"].append(p.name)
            continue
        out["files"].append({"symbol": m.group(1), "timeframe": m.group(2),
                             "name": p.name, "bytes": st.st_size,
                             "mtime": st.st_mtime})
    return out


# ==========================================================================
# what the live stack needs from this feed
# ==========================================================================

def required_streams() -> dict[str, Any]:
    """
    The `(symbol, timeframe)` pairs the ROUTING TABLE needs bars for.

    Read from `config/portfolios.json` with the standard library, not through
    the daemon: `realtime.regime_daemon` pulls pandas, and this file's whole
    claim is that it does not.

    Each active strategy id splits to a certified `(symbol, timeframe)` - the
    tape the regime gate is drawn on - and the EXECUTION tier trades that
    contract's micros, so those are listed too. A spool carrying MNQ but not NQ
    answers the loop and starves the daemon, and both report correctly while it
    happens; that is the split the RUNBOOK's "FULL-SIZE tape" section is about,
    and this is the reader that can see it coming.
    """
    out: dict[str, Any] = {"pairs": [], "error": None, "ids": []}
    try:
        blob = json.loads(PORTFOLIO_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    seen: set[tuple[str, str, str]] = set()
    for pname, p in sorted((blob.get("portfolios") or {}).items()):
        for sid in (p.get("active_strategies") or []):
            out["ids"].append({"id": sid, "portfolio": pname})
            _strategy, symbol, tf = split_strategy_id(sid)
            if not symbol or not tf:
                # A bare id names no pair. Recorded rather than guessed at:
                # inventing a symbol here would demand bars nothing certified.
                continue
            for sym in [symbol] + list(micros_of(symbol)):
                key = (sym, tf, sid)
                if key in seen:
                    continue
                seen.add(key)
                out["pairs"].append({
                    "symbol": sym, "timeframe": tf, "for": sid,
                    "role": "certified tape" if sym == symbol
                            else f"execution micro of {symbol}"})
    return out


def satisfied_by(symbol: str, tf: str,
                 on_disk: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Which spool file would answer a request for `(symbol, tf)`, and how.

    MIRRORS `realtime.nt8_feed.spool_path` + `closed_bars`, which is the only
    reason this is not a one-line `(symbol, tf) in have` test:

      * `spool_path` tries the symbol AND its full-size parent, so a request
        for MNQ is answered by `MNQ_*` or by `NQ_*`;
      * `closed_bars` takes a NATIVE `{SYM}_{TF}` file when one exists and
        otherwise reads `{SYM}_1m` and resamples with `to_timeframe`.

    So a spool holding only `NQ_1m.csv` fully answers `NQ 1h` - and a reader
    that demanded `NQ_1h.csv` would report a healthy feed as MISSING, send an
    operator to fix a publisher that is working, and be wrong in the direction
    that wastes a session. That is not hypothetical: it is what the first
    version of this file did.

    `nt8_feed` cannot be imported here (it pulls pandas), so the rule is
    restated and `tests/test_check_nt8_feed.py` pins this function against the
    real `spool_path` on a temporary directory. The copy is checked, not
    trusted.
    """
    sym = str(symbol).upper()
    parent = resolve_parent(sym)
    index = {(f["symbol"], f["timeframe"]): f for f in on_disk}
    for name, via in ((sym, "own"), (parent, f"parent {parent}")):
        if name == sym and via != "own":
            continue
        native = index.get((name, str(tf)))
        if native is not None:
            return {**native, "how": "native", "via": via}
    for name, via in ((sym, "own"), (parent, f"parent {parent}")):
        base = index.get((name, "1m"))
        if base is not None:
            return {**base, "how": f"resampled from 1m", "via": via}
    return None


def configured_universe() -> list[str]:
    """
    The research universe, read from the shell helper that defines it.

    `_TRADING_UNIVERSE` in `deploy/shell/trading_helpers.sh` is the single
    place that list exists, so it is parsed rather than restated. It is CONTEXT
    on this card and not a requirement: it is what multi-asset backtests sweep,
    and nothing obliges the live feed to carry all of it. What the feed must
    carry is `required_streams()` above.
    """
    try:
        text = HELPERS_SH.read_text(encoding="utf-8")
    except OSError:
        return []
    m = re.search(r'^_TRADING_UNIVERSE="([^"]*)"', text, re.MULTILINE)
    if not m:
        return []
    return [s.strip().upper() for s in m.group(1).split(",") if s.strip()]


# ==========================================================================
# downstream of the feed
# ==========================================================================

def regime_state() -> dict[str, Any]:
    """The regime file's own stamp, and the file's mtime. They can differ."""
    out: dict[str, Any] = {"path": REGIME_STATE, "exists": REGIME_STATE.exists(),
                           "mtime": None, "updated_at": None, "error": None}
    if not out["exists"]:
        return out
    try:
        out["mtime"] = REGIME_STATE.stat().st_mtime
    except OSError:
        pass
    try:
        blob = json.loads(REGIME_STATE.read_text(encoding="utf-8"))
        out["updated_at"] = parse_iso(blob.get("updated_at"))
        strategies = blob.get("strategies") or {}
        out["strategies"] = {k: v.get("status") for k, v in strategies.items()}
    except (OSError, ValueError) as e:
        # A mid-write read is a transient, not a fault. The next poll succeeds.
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def watchdog_verdict() -> dict[str, Any]:
    """
    The LAST watchdog run, read off its log.

    The log is one line per check and a run is the group of lines sharing a
    timestamp, so the last run is the last such group - not the last line,
    which would report one check out of four as though it were the verdict.

    The log is read rather than `scripts/watchdog.py` re-run: the watchdog
    opens the lake and the spool, and a status card that re-runs it turns a
    read into a second, slower opinion that can disagree with the one systemd
    already recorded.
    """
    out: dict[str, Any] = {"exists": WATCHDOG_LOG.exists(), "stamp": None,
                           "checks": [], "error": None}
    if not out["exists"]:
        return out
    try:
        # The tail is enough and the file is rotated; reading it whole would
        # grow into a megabyte read on every invocation of a "fast" tool.
        with WATCHDOG_LOG.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            back = min(fh.tell(), 8192)
            fh.seek(-back, os.SEEK_END)
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    parsed = []
    for line in lines:
        m = re.match(r"^\[([^\]]+)\]\s+(\S+)\s+(\S+)\s*(.*)$", line)
        if m:
            parsed.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    if not parsed:
        return out
    last_stamp = parsed[-1][0]
    out["stamp"] = parse_iso(last_stamp)
    out["checks"] = [{"status": s, "check": c, "detail": d}
                     for ts, s, c, d in parsed if ts == last_stamp]
    return out


# ==========================================================================
# classification
# ==========================================================================

def classify(age: float | None, bar_seconds: float | None,
             stale_seconds: float, listener_stale: bool | None) -> str:
    """
    One stream's status, with the listener's own verdict carried alongside.

    Three levels: a bar inside its own width is STREAMING, anything up to
    `stale_seconds` is between bars and fine, past that is stale. The
    listener's flag is APPENDED rather than merged - it fires at 3 bar widths
    and this fires at 5 minutes, so on a 1m stream there is a two-minute band
    where they differ. Showing only one of them would make two tools on this
    box contradict each other about the same stream with no way to see why.
    """
    if age is None:
        label = "UNKNOWN"
    elif bar_seconds is not None and age <= bar_seconds:
        label = STREAMING
    elif age <= stale_seconds:
        label = IDLE
    else:
        label = STALE
    if listener_stale and label != STALE:
        label += " · listener: STALE"
    return label


# ==========================================================================
# the card
# ==========================================================================

def port_of(url: str) -> int:
    """
    The port named in a URL, defaulting to this listener's.

    The process check has to follow the URL. Asked about a listener on some
    other port while reporting on whoever holds 8000, the card prints
    `Listener Status: STOPPED` and `Listener Service: active` on consecutive
    lines - two true statements about two different processes, reading as one
    self-contradicting answer.
    """
    m = re.search(r"^\w+://[^/:]+:(\d+)", str(url or ""))
    return int(m.group(1)) if m else DEFAULT_PORT


def collect(url: str = DEFAULT_URL, stale_seconds: float = DEFAULT_STALE_SECONDS,
            port: int | None = None) -> dict[str, Any]:
    """Everything the card needs, as plain data, so it can be asserted on."""
    port = port_of(url) if port is None else port
    health = fetch_health(url)
    payload = health.get("payload")
    proc = listener_process(port)
    spool_path, spool_source = resolve_spool(payload)
    return {
        "now": time.time(),
        "port": port,
        "health": health,
        "process": proc,
        "spool": {**spool_stats(spool_path), "path": spool_path,
                  "source": spool_source},
        "required": required_streams(),
        "universe": configured_universe(),
        "regime": regime_state(),
        "watchdog": watchdog_verdict(),
        "stale_seconds": stale_seconds,
    }


def render(snap: dict[str, Any]) -> str:
    now = snap["now"]
    health, proc, spool = snap["health"], snap["process"], snap["spool"]
    payload = health.get("payload") or {}
    L: list[str] = ["=" * W, "NINJATRADER 8 LIVE FEED STATUS", "=" * W]

    # ---- listener ----------------------------------------------------
    status = str(payload.get("status") or "").upper()
    if health.get("error"):
        if health["error"] == "connection refused":
            L.append(row("Listener Status",
                         "STOPPED (connection refused on "
                         f"{snap['health']['url']})"))
        else:
            L.append(row("Listener Status", f"UNREACHABLE ({health['error']})"))
    elif status == "STARVED":
        L.append(row("Listener Status",
                     f"WARMING UP (HTTP {health['code']}, STARVED — up, and "
                     f"no bar has arrived yet)"))
    elif status == "STALE":
        L.append(row("Listener Status",
                     f"RUNNING (HTTP {health['code']}, but a stream is past "
                     f"its window)"))
    elif status == "HEALTHY":
        L.append(row("Listener Status", f"RUNNING (Healthy, HTTP {health['code']})"))
    else:
        L.append(row("Listener Status",
                     f"ANSWERING (HTTP {health['code']}, status "
                     f"{status or 'not reported'})"))

    pids = proc.get("pids") or []
    unit = proc.get("unit_active")
    port = snap.get("port", DEFAULT_PORT)
    if pids:
        owner = ("systemd" if proc.get("unit_pid") in pids
                 else "NOT the systemd unit — started by hand")
        L.append(row("Listener Service",
                     f"{unit or 'unit state unknown'} "
                     f"(PID: {', '.join(str(p) for p in pids)} | "
                     f"Port: {port} | {owner})"))
    else:
        L.append(row("Listener Service",
                     f"nothing is bound to port {port}"
                     + (f" (unit: {unit})" if unit else "")))

    if payload:
        up = payload.get("uptime_seconds")
        L.append(row("Listener Uptime", age_text(up) if up is not None else "unknown"))

    # ---- streams -----------------------------------------------------
    streams = payload.get("streams") or []
    active = sorted({str(s.get("symbol")) for s in streams if s.get("symbol")})
    universe = snap["universe"]
    if payload:
        L.append(row("Active Symbols",
                     f"{len(active)} Active ({truncated(active)})"))
        tfs = sorted({str(s.get("timeframe")) for s in streams if s.get("timeframe")})
        L.append(row("Timeframes", ", ".join(tfs) if tfs else "none"))
    if universe:
        L.append(row("Research Universe",
                     f"{len(universe)} symbols (what backtests sweep — the "
                     f"feed is not obliged to carry it)"))

    last_bar = parse_iso(payload.get("last_bar_utc"))
    if payload:
        L.append(row("Last Bar Received",
                     f"{utc_text(last_bar)} ({age_text(now - last_bar)} ago)"
                     if last_bar else "none yet"))
        c = payload.get("counters") or {}
        L.append(row("Bars Ingested",
                     f"{c.get('accepted', 0):,} accepted, "
                     f"{c.get('rejected', 0):,} rejected, "
                     f"{c.get('duplicate', 0):,} duplicate (this process)"))

    # ---- per symbol --------------------------------------------------
    L.append("")
    L.append("--- Symbol Ingestion Breakdown ---")
    if not payload:
        L.append("  the listener is not answering; falling back to the spool "
                 "on disk below.")
    elif not streams:
        L.append("  no stream has posted a bar since this listener started.")
    else:
        for s in sorted(streams, key=lambda x: (x.get("bar_age_seconds") or 0)):
            age = s.get("bar_age_seconds")
            bar_s = tf_seconds(s.get("timeframe"))
            label = classify(age, bar_s, snap["stale_seconds"], s.get("stale"))
            ts = parse_iso(s.get("last_bar_utc"))
            L.append(f"  • {str(s.get('symbol')):<5} {str(s.get('timeframe')):<4} "
                     f"last bar {utc_text(ts)} ({age_text(age)} ago) "
                     f"— {label}")

    # ---- what the live stack needs -----------------------------------
    req = snap["required"]
    L.append("")
    L.append("--- Streams The Live Stack Needs ---")
    if req.get("error"):
        L.append(f"  config/portfolios.json unreadable: {req['error']}")
    elif not req["pairs"]:
        L.append("  none — no portfolio lists an active strategy, so nothing "
                 "downstream is waiting on this feed.")
    else:
        files = spool.get("files", [])
        live = {str(s.get("symbol")) for s in streams}
        for p in req["pairs"]:
            hit = satisfied_by(p["symbol"], p["timeframe"], files)
            if hit is None:
                verdict = "MISSING — no spool file would answer this"
            else:
                if not payload:
                    # The listener did not answer, so whether these streams
                    # are POSTING is unknown. "NOT posting" would be a claim
                    # this tool made no observation to support.
                    fresh = "cannot confirm — the listener did not answer"
                elif hit["symbol"] in live:
                    fresh = "posting"
                else:
                    fresh = "NOT posting to this listener"
                verdict = (f"served by {hit['name']} ({hit['how']}"
                           + (f", {hit['via']}" if hit["via"] != "own" else "")
                           + f") — {fresh}")
            L.append(f"  • {p['symbol']:<5} {p['timeframe']:<4} "
                     f"{verdict}")
            L.append(f"{'':<19}  ({p['role']}, for {p['for']})")

    # ---- integrity ---------------------------------------------------
    L.append("")
    L.append("--- System Integrity & Heartbeat ---")
    reg = snap["regime"]
    if not reg["exists"]:
        L.append(row("Regime State File", f"MISSING ({reg['path']})"))
    else:
        stamp = reg.get("updated_at") or reg.get("mtime")
        L.append(row("Regime State File",
                     f"updated {age_text(now - stamp)} ago "
                     f"({short(reg['path'])})"
                     if stamp else f"unreadable ({reg.get('error')})"))
        for sid, st in sorted((reg.get("strategies") or {}).items()):
            L.append(f"{'':<19}  {sid}: {st}")

    wd = snap["watchdog"]
    if not wd["exists"]:
        L.append(row("Watchdog Status", "no log at logs/watchdog.log"))
    elif not wd["checks"]:
        L.append(row("Watchdog Status", "the log holds no parseable run"))
    else:
        bad = [c for c in wd["checks"] if c["status"].lower() != "ok"]
        verdict = "PASS (all checks green)" if not bad else \
            "FAIL: " + ", ".join(c["check"] for c in bad)
        L.append(row("Watchdog Status",
                     f"{verdict} — last run {age_text(now - wd['stamp'])} ago"
                     if wd["stamp"] else verdict))
        for c in bad:
            L.append(f"{'':<19}  {c['check']}: {c['detail']}")

    if not spool["exists"]:
        L.append(row("Spool Directory", f"MISSING — {spool['path']}"))
    else:
        L.append(row("Spool Directory",
                     f"{spool['path']}  "
                     f"({len(spool['files'])} file(s), "
                     f"{size_text(spool['total_bytes'])} on disk) "
                     f"[{spool['source']}]"))
        for f in sorted(spool["files"], key=lambda x: x["symbol"]):
            L.append(f"{'':<19}  {f['name']:<16} "
                     f"{size_text(f['bytes']):>9}  "
                     f"written {age_text(now - f['mtime'])} ago")
        if spool["unparsed"]:
            # See the RUNBOOK's rollover section: a name that is not
            # <ROOT>_<TF>.csv is how a publisher rolled onto a physical
            # contract, and the spool it forks is one nothing reads.
            L.append(f"{'':<19}  [!] UNRECOGNISED: "
                     f"{', '.join(spool['unparsed'])}")
            L.append(f"{'':<19}      a spool name that is not <ROOT>_<TF>.csv "
                     f"is read by nothing — see the RUNBOOK's rollover section.")

    L.append("=" * W)
    return "\n".join(L)


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Whether NinjaTrader is still feeding this box. Reads the "
                    "listener, the spool and the two files downstream of it; "
                    "sends nothing and writes nothing.")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"the health endpoint (default {DEFAULT_URL})")
    ap.add_argument("--port", type=int, default=None,
                    help="the port to ask the process table about. Defaults "
                         "to the port named in --url, so the two lines of the "
                         "card cannot describe different processes.")
    ap.add_argument("--stale-after-seconds", type=float,
                    default=DEFAULT_STALE_SECONDS, metavar="SECONDS",
                    help=f"escalate a stream to {STALE} past this age "
                         f"(default {DEFAULT_STALE_SECONDS:.0f}). The "
                         f"listener's own 3-bar flag is shown regardless.")
    ap.add_argument("--watch", type=float, nargs="?", const=15.0, default=None,
                    metavar="SECONDS", help="redraw every SECONDS (default 15)")
    ap.add_argument("--json", action="store_true",
                    help="print the collected data instead of the card")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def once() -> tuple[str, bool]:
        snap = collect(args.url, args.stale_after_seconds, args.port)
        if args.json:
            return json.dumps(snap, indent=2, default=str), bool(
                snap["health"].get("ok"))
        return render(snap), bool(snap["health"].get("ok"))

    if args.watch is None:
        text, ok = once()
        print(text)
        # Non-zero when the listener could not be reached AT ALL, so this is
        # usable in a shell `&&` chain. A 503/STARVED is reachable and returns
        # 0: it is a healthy restart, not an outage.
        return 0 if ok else 1

    try:
        while True:
            text, _ok = once()
            print("\033[2J\033[H", end="")
            print(text, flush=True)
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
