#!/usr/bin/env python3
"""
tests/test_check_crosstrade_connection.py - the CrossTrade reachability card.

Location: ~/src/trading/tests/test_check_crosstrade_connection.py

    .venv/bin/python3 -m pytest tests/test_check_crosstrade_connection.py -v

ASSERT-BASED so `tests/conftest.py` collects it case by case. Every helper is
named `_...`: pytest collects any module-level `test_*` it can call, including
one whose only argument is defaulted, and `tests/test_regime_profiler.py` was
bitten by exactly that.

NO TEST HERE TOUCHES THE REAL ENDPOINT. Reachability is exercised against a
throwaway TLS server on 127.0.0.1 with a self-signed certificate, and the one
case that must never happen — a request to the webhook PATH — is asserted
against the guard rather than by trying it.

WHAT IS WORTH PINNING
---------------------
Two things, and neither is the layout:

  * **The secret never reaches stdout.** The webhook URL's path IS the
    credential and the API key is the account. A card that printed either is
    a card that leaks a funded account into a terminal that gets pasted into
    tickets. Both are asserted absent from the rendered output.
  * **The probe cannot address a path.** The module's one hard rule is that it
    never requests the webhook path, and the way that rule gets broken later
    is somebody passing a full URL into `http_probe` because it takes a
    string. The guard is tested with a path, a query and a fragment.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest                                                      # noqa: E402

from realtime import check_crosstrade_connection as ct             # noqa: E402


# ==========================================================================
# fixtures
# ==========================================================================

def _env(tmp: Path, url: str | None, key: str | None) -> None:
    """
    Point the credential resolver at a throwaway .env.

    Written to a FILE rather than os.environ because that is where the real
    ones live: `mdlib.env.NO_EXPORT` withholds them from the environment, and
    a test that set them there would exercise the fallback rather than the
    path production uses.
    """
    f = tmp / ".env"
    lines = []
    if url is not None:
        lines.append(f"CROSSTRADE_WEBHOOK_URL={url}")
    if key is not None:
        lines.append(f"CROSSTRADE_API_KEY={key}")
    f.write_text("\n".join(lines) + "\n")
    os.environ["BT_ENV_FILE"] = str(f)
    os.environ.pop(ct.ENV_URL, None)
    os.environ.pop(ct.ENV_KEY, None)


def _unenv() -> None:
    os.environ.pop("BT_ENV_FILE", None)


def _selfsigned(tmp: Path) -> tuple[Path, Path]:
    """A throwaway certificate, so a TLS handshake can be tested offline."""
    pytest.importorskip("cryptography")
    from datetime import datetime, timedelta, timezone   # noqa: PLC0415
    from cryptography import x509                        # noqa: PLC0415
    from cryptography.x509.oid import NameOID            # noqa: PLC0415
    from cryptography.hazmat.primitives import hashes, serialization  # noqa: PLC0415,E501
    from cryptography.hazmat.primitives.asymmetric import rsa        # noqa: PLC0415,E501

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName(
                [x509.DNSName("localhost")]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp / "cert.pem", tmp / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    return cert_path, key_path


def _tls_server(tmp: Path):
    """An HTTPS server on 127.0.0.1 that records every path it is asked for."""
    cert, key = _selfsigned(tmp)
    asked: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                    # noqa: N802
            asked.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_a):                          # noqa: A003
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_port, asked, cert


# ==========================================================================
# the secret must not reach stdout
# ==========================================================================

def test_neither_the_webhook_path_nor_the_key_is_printed(tmp_path):
    """
    THE CASE THIS FILE EXISTS FOR. The URL's path IS the credential and the
    key IS the account. Terminal output gets pasted into tickets and chat.
    """
    _env(tmp_path,
         "https://app.crosstrade.io/hook/AAAA/BBBB/SECRETPATHSEGMENT",
         "ct_live_KEYMUSTNOTAPPEAR_0123456789abcdef")
    try:
        card = ct.render(ct.collect())
    finally:
        _unenv()
    assert "SECRETPATHSEGMENT" not in card
    assert "KEYMUSTNOTAPPEAR" not in card
    assert "/hook/" not in card
    # ...and what IS shown is enough to act on.
    assert "https://app.crosstrade.io" in card
    assert "4-segment path" in card, "hook/AAAA/BBBB/SECRET is four"


def test_the_key_is_masked_to_length_and_a_tail():
    """
    The PREFIX is deliberately not shown. For an opaque 43-character token the
    first eight characters are ~19% of the secret; the last four identify it
    against another key and are useless alone.
    """
    masked = ct.mask_secret("ct_live_0123456789abcdefghijklmnop_9f3a")
    assert masked.endswith("...9f3a")
    assert "ct_live" not in masked
    assert "chars" in masked
    assert ct.mask_secret("") == "not set"
    assert "too short" in ct.mask_secret("ab")


def test_safe_origin_reports_the_path_without_revealing_it():
    origin, info = ct.safe_origin("https://host.example/a/b/c?q=1")
    assert origin == "https://host.example"
    assert info["path_segments"] == 3
    assert info["has_query"] is True
    assert "a/b/c" not in json.dumps(info)


def test_a_non_https_webhook_is_called_out():
    """A credential-bearing URL over http is on the wire in clear text."""
    _origin, info = ct.safe_origin("http://host.example/hook/x")
    assert "not https" in (info["error"] or "")


def test_an_unparseable_url_does_not_raise():
    origin, info = ct.safe_origin("not-a-url")
    assert origin is None
    assert info["error"]


# ==========================================================================
# the probe cannot address the webhook path
# ==========================================================================

def test_the_probe_refuses_anything_but_a_bare_origin():
    """
    The module's one hard rule. It gets broken later by somebody passing a
    full URL in, because the function takes a string — so the guard re-parses
    what it was handed rather than trusting the caller.
    """
    for bad in ("https://app.crosstrade.io/hook/secret",
                "https://app.crosstrade.io/?token=abc",
                "https://app.crosstrade.io/#frag"):
        out = ct.http_probe(bad)
        assert out["ok"] is False
        assert "REFUSED" in out["error"]
        assert out["requested"] is None, "it must not even build a target"


def test_the_probe_requests_only_the_root_when_it_does_run(tmp_path):
    """Against a real TLS server that records every path it is asked for."""
    srv, port, asked, cert = _tls_server(tmp_path)
    try:
        os.environ["SSL_CERT_FILE"] = str(cert)
        out = ct.http_probe(f"https://localhost:{port}", timeout=5.0)
        assert out["ok"] is True, out["error"]
        assert out["code"] == 200
        assert asked == ["/"], f"the probe asked for {asked}"
    finally:
        os.environ.pop("SSL_CERT_FILE", None)
        srv.shutdown()


def test_the_probe_is_off_by_default(tmp_path):
    _env(tmp_path, "https://app.crosstrade.io/hook/a/b", "k" * 40)
    try:
        snap = ct.collect(probe=False)
    finally:
        _unenv()
    assert snap["http"] is None
    assert "NOT RUN (default)" in ct.render(snap)


# ==========================================================================
# reachability, offline
# ==========================================================================

def test_a_real_tls_handshake_reports_the_certificate(tmp_path):
    srv, port, _asked, cert = _tls_server(tmp_path)
    try:
        os.environ["SSL_CERT_FILE"] = str(cert)
        out = ct.tls_handshake("localhost", port, timeout=5.0)
        assert out["ok"] is True, out["error"]
        assert out["tls_version"].startswith("TLSv1")
        assert out["subject"] == "localhost"
        assert out["not_after"], "an expiring bridge certificate is worth seeing"
        assert out["ms"] is not None
    finally:
        os.environ.pop("SSL_CERT_FILE", None)
        srv.shutdown()


def test_an_untrusted_certificate_fails_closed(tmp_path):
    """
    Verification is NOT disabled anywhere in this module. A card that reported
    REACHABLE against a certificate it could not verify would be reporting on
    a host it has not identified.
    """
    srv, port, _asked, _cert = _tls_server(tmp_path)
    try:
        out = ct.tls_handshake("localhost", port, timeout=5.0)
        assert out["ok"] is False
        assert "certificate did NOT verify" in out["error"]
    finally:
        srv.shutdown()


def test_a_name_that_does_not_resolve_is_a_verdict_not_a_traceback():
    out = ct.resolve("no-such-host.invalid", timeout=3.0)
    assert out["ok"] is False
    assert "DNS lookup failed" in out["error"]


def test_a_closed_port_is_reported_cleanly():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    out = ct.tls_handshake("127.0.0.1", port, timeout=3.0)
    assert out["ok"] is False and out["error"]
    assert "Traceback" not in out["error"]


# ==========================================================================
# missing and malformed credentials
# ==========================================================================

def test_a_missing_url_is_reported_as_a_live_loop_blocker(tmp_path):
    _env(tmp_path, None, "k" * 40)
    try:
        snap = ct.collect()
        card = ct.render(snap)
    finally:
        _unenv()
    assert snap["url_set"] is False
    assert "NOT CONFIGURED" in card
    assert "REFUSES TO START" in card
    assert ct.verdict(snap) is False
    assert snap["dns"] is None, "nothing to resolve, so nothing was attempted"


def test_a_missing_key_alongside_a_url_is_not_a_pass(tmp_path):
    _env(tmp_path, "https://app.crosstrade.io/hook/a/b", None)
    try:
        snap = ct.collect()
        assert snap["url_set"] is True and snap["key_set"] is False
        assert ct.verdict(snap) is False
        assert "NOT SET" in ct.render(snap)
    finally:
        _unenv()


def test_credentials_come_from_the_file_not_the_environment(tmp_path):
    """
    `mdlib.env.NO_EXPORT` withholds these from os.environ on purpose, so the
    FILE is the production path. Pinned against the resolver the firewall card
    already shares, which is itself pinned against the dispatcher's reader.
    """
    _env(tmp_path, "https://host.example/hook/a", "abc123")
    try:
        creds = ct._credential_sources()
        assert creds[ct.ENV_URL] == "https://host.example/hook/a"
        assert creds[ct.ENV_KEY] == "abc123"
        assert creds["_source"][ct.ENV_URL].endswith(".env")
    finally:
        _unenv()


# ==========================================================================
# routing
# ==========================================================================

def test_the_micro_routing_is_the_alias_tables_own(tmp_path):
    """
    Restating the mapping here would let this card and the loop disagree about
    which contract an order is for.
    """
    from realtime.contract_alias import MICRO_TO_PARENT      # noqa: PLC0415
    rt = ct.routing()
    assert rt["micro_map"] == {p: m for m, p in MICRO_TO_PARENT.items()}
    card = ct.render(ct.collect())
    for parent, micro in rt["micro_map"].items():
        assert f"{parent} → {micro}" in card


# ==========================================================================
# cost and the executable
# ==========================================================================

def test_the_tool_does_not_import_pandas():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import realtime.check_crosstrade_connection;"
         "print(','.join(m for m in ('pandas','numpy','vectorbtpro')"
         "                if m in sys.modules))" % str(REPO)],
        capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"pulled in: {out.stdout.strip()}"


def test_it_runs_from_any_directory():
    out = subprocess.run(
        [sys.executable,
         str(REPO / "realtime" / "check_crosstrade_connection.py")],
        capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=180)
    assert out.returncode in (0, 1)
    assert "CROSSTRADE CONNECTION & WEBHOOK STATUS" in out.stdout
    assert "Order safeguard" in out.stdout
    assert "Traceback" not in out.stderr
