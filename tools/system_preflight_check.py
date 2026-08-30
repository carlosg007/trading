#!/usr/bin/env python3
"""
Is this box in a fit state to be trusted? Answered subsystem by subsystem.

    python3 tools/system_preflight_check.py
    python3 tools/system_preflight_check.py --json /tmp/preflight.json
    python3 tools/system_preflight_check.py --skip-network

WHAT THIS ANSWERS, AND WHAT IT CANNOT
=====================================
It answers "is the machinery sound" — the mount is writable, the interpreter
has its packages, the routing table names packages that exist, the bridge
resolves and presents a certificate this box trusts.

**IT DOES NOT CERTIFY THAT TRADING SHOULD BEGIN, and it will not print that it
does.** Three of the things that decide it are outside this repository or
outside a static check entirely:

  * the ACCOUNT. Contract sizing, daily loss limits, trailing drawdown and
    prop-firm challenge state live in CrossTrade, enforced against a live
    balance. Nothing here can see a balance.
  * the EVIDENCE. A strategy allocated to an account is one a gate certified,
    not one anybody has decided to risk money on. This script counts them; it
    has no opinion on whether their holdout numbers are good.
  * the ARMING. The loop ships `--dry-run`. Being unarmed is the normal
    resting state of this box and is reported as a FACT, never as a fault to
    be cleared on the way to a green verdict.

So the verdict says whether the subsystems are sound and states the arming
position plainly. "Ready for live trading" is a judgement a human makes with
the runbook open, and a script that printed it would be lending that judgement
an authority it has not earned.

IT CHANGES NOTHING. Every probe is a read, a DNS lookup or a TLS handshake.
The one write is a probe file on the mount, removed immediately — that is the
only way to answer "is it writable" honestly. No order endpoint is contacted:
`check_crosstrade_connection`'s rule holds here, that the URL's path IS the
credential and a completed handshake proves everything a preflight needs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"

#: Only FAIL sets the exit code. A WARN is something to look at before an
#: order goes out, not a reason to refuse to report the rest.
EXIT_OK, EXIT_FAULT = 0, 1

MOUNT = Path("/mnt/backtest")
DISK_WARN_PCT = 90.0

#: Imported because a stage cannot run without them. `vectorbtpro` is checked
#: by that name and never as `vectorbt`: the open-source package has a
#: different API, and a run that silently bound the wrong one would produce
#: numbers nobody could reproduce.
REQUIRED_PACKAGES = ("vectorbtpro", "pandas", "numpy", "scipy",
                     "sklearn", "pytest")

_COLOUR = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m",
           INFO: "\033[36m"}
_RESET, _BOLD = "\033[0m", "\033[1m"


def _use_colour(stream=sys.stdout) -> bool:
    """Colour only for a terminal. Piped into a file or a log, escape codes
    are noise a reader has to strip before they can grep the result."""
    return bool(getattr(stream, "isatty", lambda: False)()) and \
        os.environ.get("NO_COLOR") is None


def badge(status: str, colour: bool = True) -> str:
    text = f"[ {status:^4} ]"
    return f"{_COLOUR.get(status,'')}{text}{_RESET}" if colour else text


class Report:
    """Ordered checks with their verdicts. Nothing is thrown away: a check
    that could not run is recorded as one, because a preflight that silently
    skipped a subsystem is worse than one that says it could not look."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, section: str, name: str, status: str,
            detail: str = "", fix: str = "") -> None:
        self.rows.append({"section": section, "name": name, "status": status,
                          "detail": detail, "fix": fix})

    def counts(self) -> dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0, INFO: 0}
        for r in self.rows:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out

    def failures(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == FAIL]

    def warnings(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == WARN]


# ==========================================================================
# A. Shared storage
# ==========================================================================

def check_storage(rep: Report, mount: Path = MOUNT) -> None:
    sec = "A · Shared storage"
    if not mount.is_dir():
        rep.add(sec, f"{mount} mounted", FAIL, "not present",
                f"mount the NFS share at {mount}")
        return
    rep.add(sec, f"{mount} mounted", PASS, str(mount))

    probe = mount / f".preflight_{os.getpid()}"
    try:
        probe.write_text("probe\n", encoding="utf-8")
        ok = probe.read_text(encoding="utf-8").strip() == "probe"
        probe.unlink()
        rep.add(sec, "read / write / delete", PASS if ok else FAIL,
                "round-tripped a probe file" if ok
                else "wrote a probe but read it back wrong")
    except OSError as exc:
        # A hard mount that has stalled BLOCKS rather than raising, so
        # reaching this line means a real permission or space problem.
        rep.add(sec, "read / write / delete", FAIL, str(exc)[:90],
                "check the export's permissions for this user")

    reports = REPO_ROOT / "reports"
    target = mount / "reports"
    if not reports.exists():
        rep.add(sec, "reports/ resolves", FAIL, "missing",
                f"ln -s {target} {reports}")
    elif reports.is_symlink():
        dest = reports.resolve()
        ok = dest == target.resolve()
        rep.add(sec, "reports/ resolves", PASS if ok else WARN,
                f"-> {dest}" + ("" if ok else f" (expected {target})"),
                "" if ok else f"repoint the link at {target}")
    else:
        # Not a fault. It is a local directory that will not be shared, and
        # saying so is more use than a FAIL for a state somebody may want.
        rep.add(sec, "reports/ resolves", WARN,
                "a real directory, not a symlink — reports stay local",
                f"ln -s {target} {reports} to share them")

    try:
        usage = shutil.disk_usage(mount)
        pct = 100.0 * usage.used / usage.total if usage.total else 0.0
        free_gib = usage.free / 1024 ** 3
        rep.add(sec, "free space", WARN if pct > DISK_WARN_PCT else PASS,
                f"{pct:.1f}% used, {free_gib:,.0f} GiB free",
                "" if pct <= DISK_WARN_PCT else
                "a full lake truncates a run mid-write")
    except OSError as exc:
        rep.add(sec, "free space", WARN, str(exc)[:80])


# ==========================================================================
# B. Runtime
# ==========================================================================

def check_runtime(rep: Report,
                  packages: tuple[str, ...] = REQUIRED_PACKAGES) -> None:
    sec = "B · Python runtime"
    venv = REPO_ROOT / ".venv"
    # `sys.prefix`, not the resolved executable path. `.venv/bin/python3` is a
    # SYMLINK to a uv-managed interpreter under ~/.local/share/uv, so the
    # resolved path is NOT under .venv and a startswith test on it reports the
    # correct interpreter as wrong - printing an "expected" that is character
    # for character the path it just rejected. sys.prefix is what actually
    # says which environment is active.
    inside = Path(sys.prefix).resolve() == venv.resolve()
    running = Path(sys.executable)
    rep.add(sec, "interpreter", PASS if inside else FAIL,
            f"{running} (prefix {sys.prefix})",
            "" if inside else
            "run through .venv/bin/python3 — a second interpreter resolves "
            "different package versions than every stage was pinned against")
    rep.add(sec, "python version", INFO,
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}")

    for name in packages:
        try:
            mod = importlib.import_module(name)
            rep.add(sec, f"import {name}", PASS,
                    getattr(mod, "__version__", "version not reported"))
        except Exception as exc:                             # noqa: BLE001
            rep.add(sec, f"import {name}", FAIL,
                    f"{type(exc).__name__}: {str(exc)[:70]}",
                    f"uv pip install {name}")

    # The open-source package must NOT be importable. It has a different API,
    # and `riskfolio-lib` declares it as a dependency, so a reinstall can pull
    # it back and a stage would bind the wrong one without raising.
    try:
        importlib.import_module("vectorbt")
        rep.add(sec, "vectorbt absent", FAIL,
                "open-source `vectorbt` is importable",
                "uv pip uninstall vectorbt — its API differs from "
                "vectorbtpro and the wrong bind is silent")
    except ImportError:
        rep.add(sec, "vectorbt absent", PASS,
                "open-source package not importable, as required")


# ==========================================================================
# C. Feed and bridge
# ==========================================================================

def check_bridge(rep: Report, network: bool = True) -> None:
    sec = "C · Feed and bridge"
    try:
        from realtime.check_nt8_feed import (FALLBACK_SPOOL_DIR,  # noqa: PLC0415
                                             SPOOL_DIR_VAR, spool_stats)
        spool = Path(os.environ.get(SPOOL_DIR_VAR) or FALLBACK_SPOOL_DIR)
        stats = spool_stats(spool)
        # DISTINCT SYMBOLS, from `files`. There is no `symbols` key, and
        # guessing one reported 0 streams against a spool carrying 27 - the
        # exact false alarm a preflight must not raise, since the honest
        # reading of "no bars arriving" is that the listener is down.
        symbols = sorted({f.get("symbol") for f in (stats.get("files") or [])
                          if f.get("symbol")})
        n = len(symbols)
        rep.add(sec, "NT8 spool", PASS if n else FAIL,
                f"{n} contract stream(s) at {spool}",
                "" if n else "start the NT8 listener; no bars are arriving")
    except Exception as exc:                                 # noqa: BLE001
        rep.add(sec, "NT8 spool", WARN,
                f"{type(exc).__name__}: {str(exc)[:70]}")

    try:
        from realtime.check_trade_firewall import interlock   # noqa: PLC0415
        il = interlock()
        armed = il.get("running")
        # REPORTED, NOT GRADED. Dry run is the shipped default and the normal
        # resting state; grading it as a failure would train a reader to clear
        # it on the way to a green board.
        rep.add(sec, "execution interlock", INFO,
                "ARMED — the loop will send orders" if armed is True
                else "dry run — `--live` absent (the default)"
                if armed is False else "no loop running")
    except Exception as exc:                                 # noqa: BLE001
        rep.add(sec, "execution interlock", WARN, str(exc)[:80])

    if not network:
        rep.add(sec, "bridge reachability", INFO, "skipped (--skip-network)")
        return
    try:
        from realtime.check_crosstrade_connection import (    # noqa: PLC0415
            resolve, safe_origin, tls_handshake)
        url = os.environ.get("CROSSTRADE_WEBHOOK_URL") or ""
        origin, _meta = safe_origin(url) if url else (None, {})
        if not origin:
            rep.add(sec, "bridge reachability", WARN,
                    "no CROSSTRADE_WEBHOOK_URL configured",
                    "set it in .env before arming")
            return
        host = origin.split("://", 1)[-1].split("/", 1)[0]
        dns = resolve(host)
        rep.add(sec, "bridge DNS", PASS if dns.get("ok") else FAIL,
                f"{host} -> {', '.join(dns.get('addresses') or []) or 'no answer'}")
        if dns.get("ok"):
            tls = tls_handshake(host)
            rep.add(sec, "bridge TLS", PASS if tls.get("ok") else FAIL,
                    # ZERO bytes of HTTP. A completed handshake proves the
                    # name resolves, the route works, something is listening
                    # and it presents a certificate this box trusts. No path
                    # is requested, so no endpoint can act on it.
                    "handshake completed; no HTTP sent"
                    if tls.get("ok") else str(tls.get("error"))[:70])
    except Exception as exc:                                 # noqa: BLE001
        rep.add(sec, "bridge reachability", WARN,
                f"{type(exc).__name__}: {str(exc)[:70]}")


# ==========================================================================
# D. Incubator and routing
# ==========================================================================

def check_portfolio(rep: Report) -> int:
    """Returns the number of allocations that resolve to a package on disk."""
    sec = "D · Incubator and routing"
    try:
        from tools.portfolio_inventory import collect_rows, MISSING  # noqa: PLC0415
        cfg = json.loads((REPO_ROOT / "config" / "portfolios.json")
                         .read_text(encoding="utf-8"))
        rows = collect_rows(cfg)
    except Exception as exc:                                 # noqa: BLE001
        rep.add(sec, "routing table", FAIL,
                f"{type(exc).__name__}: {str(exc)[:70]}")
        return 0

    missing = [r["strategy_id"] for r in rows if r["disk_status"] == MISSING]
    rep.add(sec, "allocations resolve on disk", FAIL if missing else PASS,
            f"{len(rows) - len(missing)}/{len(rows)} present"
            + (f"; MISSING: {', '.join(missing[:4])}" if missing else ""),
            "" if not missing else
            "the dispatcher imports a strategy by that path — prune the "
            "allocation or restore the package")

    # CL cannot trade from this box: NinjaTrader streams no CL series, so a
    # CL allocation is a row that can never place an order.
    cl = [r["strategy_id"] for r in rows
          if str(r.get("symbol", "")).upper() in ("CL", "MCL")]
    rep.add(sec, "no CL / MCL allocations", FAIL if cl else PASS,
            ", ".join(cl) if cl else "none",
            "" if not cl else "no CL feed on this box; prune these")

    bad_b: list[str] = []
    version_b = [r for r in rows if r["version"] == "B"]
    for r in version_b:
        pkg = REPO_ROOT / "strategies" / "approved_incubator" / r["strategy_id"]
        strat, base = pkg / "strat.py", pkg / "baseline.py"
        try:
            embeds = "ML_THRESHOLD" in strat.read_text(encoding="utf-8")
        except OSError:
            embeds = False
        if not (embeds and base.is_file()):
            bad_b.append(r["strategy_id"])
    rep.add(sec, "Version B packages complete", FAIL if bad_b else PASS,
            f"{len(version_b) - len(bad_b)}/{len(version_b)} embed "
            f"ML_THRESHOLD and carry baseline.py"
            + (f"; incomplete: {', '.join(bad_b[:3])}" if bad_b else ""),
            "" if not bad_b else
            "a Version B without its wrapper runs as bare rules — the "
            "filter its certification measured would not be applied")
    return len(rows) - len(missing)


# ==========================================================================
# E. Outbound connectivity
# ==========================================================================

def check_network(rep: Report, network: bool = True) -> None:
    sec = "E · Outbound connectivity"
    if not network:
        rep.add(sec, "DNS", INFO, "skipped (--skip-network)")
        return
    try:
        from realtime.check_crosstrade_connection import (    # noqa: PLC0415
            resolve, safe_origin)
    except Exception as exc:                                 # noqa: BLE001
        rep.add(sec, "DNS", WARN, str(exc)[:70])
        return

    hook = os.environ.get("DISCORD_WEBHOOK_URL") or ""
    origin, _ = safe_origin(hook) if hook else (None, {})
    if origin:
        host = origin.split("://", 1)[-1].split("/", 1)[0]
        dns = resolve(host)
        # The webhook's PATH is the credential, so only the host is named and
        # nothing is posted. A preflight that announced a run would be a
        # preflight nobody could run twice.
        rep.add(sec, "Discord reporting host", PASS if dns.get("ok") else WARN,
                f"{host} resolves" if dns.get("ok")
                else f"{host} did not resolve")
    else:
        rep.add(sec, "Discord reporting host", WARN,
                "no DISCORD_WEBHOOK_URL configured",
                "stage cards will not be delivered")


# ==========================================================================
# Rendering
# ==========================================================================

def render(rep: Report, ready: int, colour: bool = True) -> str:
    W = 78
    bold = (lambda s: f"{_BOLD}{s}{_RESET}") if colour else (lambda s: s)
    out = ["=" * W, bold("SYSTEM PREFLIGHT CHECK"), "=" * W]
    section = None
    for r in rep.rows:
        if r["section"] != section:
            section = r["section"]
            out += ["", bold(section), "-" * W]
        out.append(f"  {badge(r['status'], colour)}  {r['name']:<32}"
                   f"{r['detail']}")
    counts = rep.counts()
    out += ["", "=" * W, bold("SYSTEM OPERATIONAL VERDICT"), "=" * W,
            f"  {counts[PASS]} passed · {counts[WARN]} warning(s) · "
            f"{counts[FAIL]} failure(s)",
            f"  {ready} allocated strateg{'y' if ready == 1 else 'ies'} "
            f"resolve to a package on disk"]

    if rep.failures():
        out += ["", "  SUBSYSTEM FAULTS — fix these before arming:"]
        for r in rep.failures():
            out.append(f"    - {r['name']}: {r['detail']}")
            if r["fix"]:
                out.append(f"        {r['fix']}")
    else:
        out += ["", "  All checked subsystems are sound."]
    if rep.warnings():
        out += ["", "  WARNINGS — look before an order goes out:"]
        for r in rep.warnings():
            out.append(f"    - {r['name']}: {r['detail']}")

    # THE LINE THIS SCRIPT WILL NOT PRINT is "ready for live trading". What it
    # can see is machinery. Whether to risk money additionally depends on the
    # ACCOUNT (CrossTrade holds the balance, the daily loss limit and the
    # prop-firm state), on the EVIDENCE behind each allocated strategy, and on
    # a human with the runbook open. Saying it here would lend that judgement
    # an authority a static check has not earned.
    out += ["", "  This check covers MACHINERY, not the decision to trade.",
            "  Account state (CrossTrade), each strategy's holdout evidence,",
            "  and the arming position are the rest of that call.", "=" * W]
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Post-power-on readiness check. Reads, resolves and "
                     "handshakes; sends no order and posts no card."))
    p.add_argument("--skip-network", action="store_true",
                   help="no DNS or TLS — for an offline box")
    p.add_argument("--json", dest="json_out", metavar="PATH",
                   help="also write the raw findings here")
    p.add_argument("--no-color", action="store_true")
    return p


def run_all(network: bool = True) -> tuple[Report, int]:
    rep = Report()
    check_storage(rep)
    check_runtime(rep)
    check_bridge(rep, network=network)
    ready = check_portfolio(rep)
    check_network(rep, network=network)
    return rep, ready


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rep, ready = run_all(network=not args.skip_network)
    colour = _use_colour() and not args.no_color
    print(render(rep, ready, colour))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "counts": rep.counts(), "ready_strategies": ready,
             "rows": rep.rows}, indent=2) + "\n", encoding="utf-8")
    return EXIT_FAULT if rep.failures() else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
