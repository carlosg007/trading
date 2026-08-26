#!/usr/bin/env python3
"""
realtime/check_crosstrade_connection.py - can this box reach CrossTrade?

Location:  ~/src/trading/realtime/check_crosstrade_connection.py

    python3 realtime/check_crosstrade_connection.py
    python3 realtime/check_crosstrade_connection.py --http-probe
    crosstrade-check / ct-check                    # the aliases

`firewall-check` reports the bridge as configured or not. This answers the
next question: is the host actually reachable from here, and is the routing
pointed where you think it is.

WHAT IT SENDS: NOTHING, BY DEFAULT
----------------------------------
The webhook URL is the credential for a funded account and its entire purpose
is placing orders. So reachability is established WITHOUT an application-layer
request:

    1. DNS      `getaddrinfo(host)` - the name resolves, and to what
    2. TCP      a socket connects to host:443
    3. TLS      a real handshake completes and the certificate validates

That chain proves everything "reachable" needs to mean - the name resolves,
the route works, something is listening, and it presents a certificate this
box trusts - and it sends ZERO bytes of HTTP. No path is requested, so no
endpoint can act on it. A TLS handshake cannot place an order.

`--http-probe` adds a GET, and even then only to the ORIGIN (`https://host/`),
never to the webhook path. It is opt-in rather than default because the
marginal information - that an HTTP server answers, which the TLS handshake
already implied - is not worth a request against a host whose other paths take
orders. The probe REFUSES to run against anything but a bare origin.

WHAT IT PRINTS: NEVER THE SECRET
--------------------------------
The configured URL carries a 4-segment path that IS the credential. Only
`scheme://host` is ever shown; the path is reported as a segment count, which
is enough to confirm one is present and useless to anybody reading over a
shoulder. The API key is shown as its length and last four characters - the
standard identification tail, enough to tell two keys apart and not enough to
use one.

Neither value is put into `os.environ`. `mdlib.env.NO_EXPORT` withholds them
because anything in the environment is inherited by every subprocess, which is
how a webhook key reaches an unrelated tool's debug output. Resolution goes
through `check_trade_firewall._credential_sources`, which mirrors the
dispatcher's own file-then-environment precedence and is pinned against
`live_dispatcher.load_env_file` by that module's tests.

COST
----
No pandas, no engine. `check_trade_firewall` (22ms) is imported for the
credential resolution and the interlock, so those rules have one spelling.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
import socket                                                      # noqa: E402
import ssl                                                         # noqa: E402
import time                                                        # noqa: E402
from typing import Any                                             # noqa: E402
from urllib.parse import urlparse                                  # noqa: E402

REPO = PROJECT_ROOT

from realtime.check_trade_firewall import (                        # noqa: E402
    ENV_KEY, ENV_URL, _credential_sources, interlock, read_json, short)
from realtime.contract_alias import MICRO_TO_PARENT                # noqa: E402

W = 80
PORTFOLIO_CONFIG = REPO / "config" / "portfolios.json"
DEFAULT_TIMEOUT = 6.0

PASS = "PASS"
FAIL = "FAIL"


# ==========================================================================
# masking
# ==========================================================================

def mask_secret(value: str, tail: int = 4) -> str:
    """
    `43 chars, ending ...9f3a` - length and an identification tail.

    THE PREFIX IS NOT SHOWN, deliberately, and that is a departure from the
    usual `ct_live_****3a9f` house style. For an opaque 43-character token the
    first eight characters are ~19% of the secret; the last four identify it
    against another key and are useless on their own. Terminal output gets
    pasted into tickets and chat, so the tail is the half worth showing.
    """
    value = str(value or "")
    if not value:
        return "not set"
    if len(value) <= tail:
        return f"{len(value)} chars (too short to mask safely)"
    return f"{len(value)} chars, ending ...{value[-tail:]}"


def safe_origin(url: str) -> tuple[str | None, dict[str, Any]]:
    """
    `scheme://host` and a description of the path, WITHOUT the path itself.

    The configured webhook carries a multi-segment path that IS the
    credential. Reporting how many segments it has confirms one is present -
    a URL with no path would be a misconfiguration worth seeing - while
    printing none of it.
    """
    info: dict[str, Any] = {"scheme": None, "host": None, "port": None,
                            "path_segments": 0, "has_query": False,
                            "error": None}
    try:
        p = urlparse(str(url or ""))
    except ValueError as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info
    info["scheme"] = p.scheme or None
    info["host"] = p.hostname
    info["port"] = p.port
    info["path_segments"] = len([s for s in (p.path or "").split("/") if s])
    info["has_query"] = bool(p.query)
    if not p.scheme or not p.hostname:
        info["error"] = "not an absolute URL"
        return None, info
    if p.scheme != "https":
        # A webhook carrying a credential over http would put it on the wire
        # in clear text at every hop.
        info["error"] = f"scheme is {p.scheme!r}, not https"
    origin = f"{p.scheme}://{p.hostname}" + (f":{p.port}" if p.port else "")
    return origin, info


# ==========================================================================
# reachability, without an application-layer request
# ==========================================================================

def resolve(host: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """DNS only. Records the families and how long the lookup took."""
    out: dict[str, Any] = {"ok": False, "addresses": [], "ms": None,
                           "error": None}
    started = time.perf_counter()
    try:
        socket.setdefaulttimeout(timeout)
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        out["error"] = f"DNS lookup failed: {e}"
        return out
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    finally:
        socket.setdefaulttimeout(None)
    out["ms"] = (time.perf_counter() - started) * 1000
    seen: list[str] = []
    for family, _t, _p, _c, sockaddr in infos:
        addr = sockaddr[0]
        if addr not in seen:
            seen.append(addr)
    out["addresses"] = seen
    out["ok"] = bool(seen)
    return out


def tls_handshake(host: str, port: int = 443,
                  timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """
    Connect, negotiate TLS, read the certificate, close. NO HTTP IS SENT.

    This is the whole reachability test, and it is deliberately the whole of
    it: a completed handshake proves the name resolved, the route works,
    something is listening on 443 and it presents a certificate this box
    trusts. Not one byte of application data crosses the socket, so there is
    no path and no endpoint that could act on it.

    The certificate's validity dates are read back because an expiring
    certificate on the order bridge is the kind of thing that takes a stack
    down at a weekend, and it is free to check here.
    """
    out: dict[str, Any] = {"ok": False, "ms": None, "error": None,
                           "peer": None, "tls_version": None, "cipher": None,
                           "not_after": None, "subject": None, "issuer": None}
    ctx = ssl.create_default_context()
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as tls:
                out["ms"] = (time.perf_counter() - started) * 1000
                out["peer"] = tls.getpeername()[0]
                out["tls_version"] = tls.version()
                cipher = tls.cipher()
                out["cipher"] = cipher[0] if cipher else None
                cert = tls.getpeercert() or {}
                out["not_after"] = cert.get("notAfter")
                subj = dict(x[0] for x in cert.get("subject", ()) if x)
                issuer = dict(x[0] for x in cert.get("issuer", ()) if x)
                out["subject"] = subj.get("commonName")
                out["issuer"] = issuer.get("organizationName") or \
                    issuer.get("commonName")
                out["ok"] = True
    except ssl.SSLCertVerificationError as e:
        out["error"] = f"certificate did NOT verify: {e.verify_message or e}"
    except ssl.SSLError as e:
        out["error"] = f"TLS failed: {e}"
    except socket.timeout:
        out["error"] = f"no answer within {timeout:.0f}s"
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def http_probe(origin: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """
    An opt-in GET against the ORIGIN ONLY. Refuses anything with a path.

    The guard is not decoration. This module's one hard rule is that it never
    addresses the webhook path, and the way that rule gets broken later is
    somebody passing a full URL into this function because it takes a string.
    It re-parses what it was given and refuses if a path, query or fragment
    survived - so the rule is enforced here rather than remembered upstream.
    """
    out: dict[str, Any] = {"ok": False, "code": None, "ms": None,
                           "error": None, "requested": None}
    p = urlparse(origin)
    if p.path not in ("", "/") or p.query or p.fragment:
        out["error"] = ("REFUSED: the probe addresses a bare origin only, and "
                        "was handed a URL carrying a path. The webhook path "
                        "is the credential and takes orders.")
        return out

    import urllib.error                                       # noqa: PLC0415
    import urllib.request                                     # noqa: PLC0415

    target = f"{p.scheme}://{p.netloc}/"
    out["requested"] = target
    req = urllib.request.Request(target, method="GET",
                                 headers={"User-Agent": "crosstrade-check"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["code"] = resp.getcode()
            resp.read(0)
            out["ok"] = True
    except urllib.error.HTTPError as e:
        # A 4xx from the origin still proves an HTTP server answered.
        out["code"] = e.code
        out["ok"] = True
    except Exception as e:                                    # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    out["ms"] = (time.perf_counter() - started) * 1000
    return out


# ==========================================================================
# routing
# ==========================================================================

def routing() -> dict[str, Any]:
    """Which account the orders would be addressed to, and on which contracts."""
    out: dict[str, Any] = {"portfolios": [], "error": None,
                           "micro_map": dict(sorted(
                               (parent, micro)
                               for micro, parent in MICRO_TO_PARENT.items()))}
    cfg = read_json(PORTFOLIO_CONFIG)
    if cfg is None:
        out["error"] = f"{short(PORTFOLIO_CONFIG)} is unreadable"
        return out
    for name, p in sorted((cfg.get("portfolios") or {}).items()):
        active = list(p.get("active_strategies") or [])
        if not active:
            continue
        out["portfolios"].append({
            "portfolio": name,
            "account": p.get("target_account"),
            "account_type": p.get("account_type"),
            "assets": list((p.get("basket") or {}).get("assets") or []),
            "active": active})
    return out


# ==========================================================================
# assembly
# ==========================================================================

def collect(probe: bool = False,
            timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    creds = _credential_sources()
    url, key = creds.get(ENV_URL, ""), creds.get(ENV_KEY, "")
    origin, url_info = safe_origin(url)
    snap: dict[str, Any] = {
        "now": time.time(),
        "url_set": bool(url), "key_set": bool(key),
        "key_mask": mask_secret(key),
        "origin": origin, "url_info": url_info,
        "source": creds.get("_source", {}),
        "dns": None, "tls": None, "http": None,
        "routing": routing(), "interlock": interlock(),
    }
    host = url_info.get("host")
    if host:
        snap["dns"] = resolve(host, timeout)
        if snap["dns"]["ok"]:
            snap["tls"] = tls_handshake(host, url_info.get("port") or 443,
                                        timeout)
            if probe and snap["tls"]["ok"] and origin:
                snap["http"] = http_probe(origin, timeout)
    return snap


def render(snap: dict[str, Any]) -> str:
    L: list[str] = ["=" * W, "CROSSTRADE CONNECTION & WEBHOOK STATUS", "=" * W]
    info, src = snap["url_info"], snap["source"]

    if not snap["url_set"]:
        L.append(f"{'Webhook Target':<24}: NOT CONFIGURED — ${ENV_URL} is unset")
        L.append(f"{'':<24}  A LIVE loop REFUSES TO START without it.")
    else:
        note = "parsed and validated" if not info.get("error") else \
            f"PROBLEM: {info['error']}"
        L.append(f"{'Webhook Target':<24}: {snap['origin'] or '(unparseable)'} "
                 f"({note})")
        L.append(f"{'':<24}  + a {info['path_segments']}-segment path, not "
                 f"shown — the path IS the credential"
                 + ("  [from " + src[ENV_URL] + "]" if src.get(ENV_URL) else ""))
    L.append(f"{'API Key State':<24}: "
             + ("CONFIGURED — " + snap["key_mask"] if snap["key_set"]
                else f"NOT SET (${ENV_KEY})")
             + ("  [from " + src[ENV_KEY] + "]"
                if src.get(ENV_KEY) and snap["key_set"] else ""))

    dns, tls = snap["dns"], snap["tls"]
    if dns is None:
        L.append(f"{'Connection Status':<24}: NOT TESTED — no host to resolve")
    elif not dns["ok"]:
        L.append(f"{'Connection Status':<24}: UNREACHABLE — {dns['error']}")
    elif tls is None or not tls["ok"]:
        why = tls["error"] if tls else "not attempted"
        L.append(f"{'Connection Status':<24}: DNS OK, TLS FAILED — {why}")
    else:
        L.append(f"{'Connection Status':<24}: REACHABLE "
                 f"(DNS resolved · TLS handshake OK · certificate verified)")
        L.append(f"{'Handshake Latency':<24}: DNS {dns['ms']:.1f} ms · "
                 f"TLS {tls['ms']:.1f} ms")

    # ---- routing -----------------------------------------------------
    L.append("")
    L.append("--- Account Routing & Payload Config ---")
    rt = snap["routing"]
    if rt["error"]:
        L.append(f"  [!] {rt['error']}")
    if not rt["portfolios"]:
        L.append("  no portfolio lists an active strategy, so no order would "
                 "be addressed anywhere.")
    for p in rt["portfolios"]:
        L.append(f"  • {p['portfolio']:<16} account {p['account']} "
                 f"({p['account_type']})")
        L.append(f"    trades {', '.join(p['assets']) or 'nothing'} · "
                 f"{len(p['active'])} active strateg"
                 f"{'y' if len(p['active']) == 1 else 'ies'}")
    L.append(f"  Micro routing        : "
             + " | ".join(f"{parent} → {micro}"
                          for parent, micro in rt["micro_map"].items()))
    L.append("  The regime is read on the FULL-SIZE tape and the order is "
             "sent for the micro.")

    il = snap["interlock"]
    if il["running"] is True:
        guard = "DRY RUN — the running loop formats orders and sends none"
    elif il["running"] is False:
        guard = "LIVE — the running loop WILL send orders"
    else:
        guard = (f"no loop running; systemd would start it "
                 + ("in DRY RUN" if il["effective"] else "LIVE"))
    L.append(f"  Firewall Interlock   : {guard}")

    # ---- what was sent -----------------------------------------------
    L.append("")
    L.append("--- Safe Connectivity Test ---")
    L.append(f"  • DNS resolution      : "
             + (f"{PASS} — {', '.join(dns['addresses'][:3])}"
                + (f" (+{len(dns['addresses']) - 3} more)"
                   if len(dns["addresses"]) > 3 else "")
                if dns and dns["ok"] else f"{FAIL}"))
    if tls and tls["ok"]:
        L.append(f"  • TLS handshake       : {PASS} — {tls['tls_version']}, "
                 f"{tls['cipher']}")
        L.append(f"  • Certificate         : {PASS} — CN={tls['subject']}, "
                 f"issued by {tls['issuer']}")
        L.append(f"{'':<26}valid until {tls['not_after']}")
    elif tls:
        L.append(f"  • TLS handshake       : {FAIL} — {tls['error']}")

    http = snap["http"]
    if http is None:
        L.append("  • HTTP probe          : NOT RUN (default). The TLS "
                 "handshake already proves the")
        L.append("                          host answers; pass --http-probe "
                 "to GET the origin.")
    elif http["error"]:
        L.append(f"  • HTTP probe          : {FAIL} — {http['error']}")
    else:
        L.append(f"  • HTTP probe          : {PASS} — GET {http['requested']} "
                 f"→ {http['code']} in {http['ms']:.1f} ms")

    L.append("  • Order safeguard     : ACTIVE — zero requests to the webhook "
             "path. Nothing this")
    L.append("                          tool sends can place an order.")
    L.append("=" * W)
    return "\n".join(L)


def verdict(snap: dict[str, Any]) -> bool:
    """True when the bridge is configured AND the host is reachable."""
    return bool(snap["url_set"] and snap["key_set"]
                and not (snap["url_info"] or {}).get("error")
                and (snap["dns"] or {}).get("ok")
                and (snap["tls"] or {}).get("ok"))


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Whether CrossTrade is configured and reachable. Sends no "
                    "application data by default and never addresses the "
                    "webhook path.")
    ap.add_argument("--http-probe", action="store_true",
                    help="additionally GET the ORIGIN (never the webhook "
                         "path). Off by default: the TLS handshake already "
                         "proves the host answers.")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    metavar="SECONDS", help=f"per-step timeout "
                                            f"(default {DEFAULT_TIMEOUT:.0f})")
    ap.add_argument("--json", action="store_true",
                    help="print the collected state instead of the card")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snap = collect(probe=args.http_probe, timeout=args.timeout)
    if args.json:
        print(json.dumps(snap, indent=2, default=str))
    else:
        print(render(snap))
    return 0 if verdict(snap) else 1


if __name__ == "__main__":
    sys.exit(main())
