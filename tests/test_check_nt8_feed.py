#!/usr/bin/env python3
"""
tests/test_check_nt8_feed.py - the NT8 live-feed status card.

Location: ~/src/trading/tests/test_check_nt8_feed.py

    .venv/bin/python3 -m pytest tests/test_check_nt8_feed.py -v
    .venv/bin/python3 tests/test_check_nt8_feed.py            # as a script

ASSERT-BASED, and pytest-shaped on purpose. `tests/conftest.py` routes any
suite carrying the `check()` collector to a subprocess runner because pytest
cannot see its results; this one fails through `assert`, so both runners
report the same thing.

Every helper is named `_...`. `tests/test_regime_profiler.py` was bitten by the
other spelling: pytest collects any module-level `test_*` it can call,
INCLUDING one whose only argument is defaulted, and a helper collected that way
ran without its `$BT_ARTIFACTS` redirect and wrote real JSON onto the NFS
mount. Nothing here may touch `/mnt/backtest`, so nothing here is named so it
could be called by accident.

WHAT IS WORTH PINNING HERE
--------------------------
Two things, and neither is the formatting:

  * **`satisfied_by` against the REAL `nt8_feed.spool_path`.** The tool cannot
    import `nt8_feed` (pandas, 247ms), so it restates that module's lookup
    rule. A restated rule that drifts is how this card would start reporting a
    working publisher as MISSING - which the first version of it did, by
    demanding `NQ_1h.csv` from a spool that correctly holds only `NQ_1m.csv`.
    The test imports the real module and compares the two on a temp directory.
  * **that the tool's import graph stays free of pandas.** The whole claim of
    the file is that it is cheap enough to run in a shell prompt. A stray
    `from backtest.specs import ...` would cost 247ms and nothing would fail.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest                                                      # noqa: E402

from realtime import check_nt8_feed as cf                          # noqa: E402


# ==========================================================================
# a throwaway /health, so no test needs the real listener
# ==========================================================================

def _serve(payload, code: int = 200, raw: bytes | None = None):
    """
    An ephemeral HTTP server yielding its URL.

    A REAL socket rather than a monkeypatched `urlopen`, because the thing
    under test is how this tool behaves against HTTP - a 503 carrying a body,
    a body that is not JSON - and a patched function would test the patch.
    """
    body = raw if raw is not None else json.dumps(payload).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                    # noqa: N802
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):                          # noqa: A003
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}/health"


def _closed_port() -> int:
    """A port nothing is listening on."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _health(status="HEALTHY", streams=None, **kw):
    return {"status": status, "last_bar_utc": "2026-08-26T14:27:00Z",
            "symbols_active": sorted({s["symbol"] for s in (streams or [])}),
            "last_post_utc": "2026-08-26T14:27:42Z",
            "started_at_utc": "2026-08-26T00:06:33Z", "uptime_seconds": 51600.0,
            "spool_dir": "/tmp/spool", "default_stamp": "close",
            "stale_after_bars": 3.0,
            "counters": {"accepted": 3435, "duplicate": 0, "rejected": 0},
            "streams": streams or [], **kw}


def _stream(symbol="NQ", tf="1m", age=42.0, stale=False):
    return {"symbol": symbol, "timeframe": tf,
            "last_bar_utc": "2026-08-26T14:27:00Z", "bar_age_seconds": age,
            "stale": stale, "stamp": "close", "buffered": 400,
            "spool": f"{symbol}_{tf}.csv"}


def _spool(tmp: Path, names) -> Path:
    d = tmp / "spool"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_text("ts,open,high,low,close,volume\n", encoding="utf-8")
    return d


# ==========================================================================
# reaching the listener
# ==========================================================================

def test_a_refused_connection_is_a_verdict_not_a_traceback():
    """
    The acceptance criterion, stated as written: a listener that is not there
    must read as STOPPED, not as a socket traceback in an operator's terminal.
    """
    url = f"http://127.0.0.1:{_closed_port()}/health"
    out = cf.fetch_health(url, timeout=2.0)
    assert out["ok"] is False
    assert out["error"] == "connection refused"
    assert out["payload"] is None

    card = cf.render({**cf.collect(url=url), "now": time.time()})
    assert "STOPPED" in card
    assert "connection refused" in card
    assert "Traceback" not in card


def test_a_503_is_reachable_and_not_an_error():
    """
    `/health` answers 503 while STARVED, and after EVERY restart for up to one
    bar width. Treating a non-2xx as failure would report a healthy restart as
    an outage - the exact mistake the RUNBOOK warns an uptime monitor against,
    so the tool must not make it either.
    """
    srv, url = _serve(_health(status="STARVED", streams=[]), code=503)
    try:
        out = cf.fetch_health(url, timeout=3.0)
        assert out["ok"] is True, "503 is reachable"
        assert out["code"] == 503
        assert out["payload"]["status"] == "STARVED"
        card = cf.render(cf.collect(url=url))
        assert "WARMING UP" in card
        assert "STARVED" in card
    finally:
        srv.shutdown()


def test_a_healthy_listener_reads_as_running():
    srv, url = _serve(_health(streams=[_stream("NQ"), _stream("ES")]))
    try:
        card = cf.render(cf.collect(url=url))
        assert "RUNNING (Healthy, HTTP 200)" in card
        assert "2 Active (ES, NQ)" in card
        assert "3,435 accepted, 0 rejected" in card
    finally:
        srv.shutdown()


def test_a_body_that_is_not_json_does_not_crash():
    srv, url = _serve(None, code=200, raw=b"<html>nope</html>")
    try:
        out = cf.fetch_health(url, timeout=3.0)
        assert out["ok"] is False
        assert "not JSON" in out["error"]
        assert "Traceback" not in cf.render(cf.collect(url=url))
    finally:
        srv.shutdown()


def test_a_json_body_that_is_not_an_object_is_refused():
    srv, url = _serve(None, code=200, raw=b"[1, 2, 3]")
    try:
        out = cf.fetch_health(url, timeout=3.0)
        assert out["ok"] is False and out["payload"] is None
    finally:
        srv.shutdown()


# ==========================================================================
# the lookup rule this file restates
# ==========================================================================

def test_satisfied_by_matches_the_real_spool_path(tmp_path):
    """
    THE COPY IS CHECKED, NOT TRUSTED.

    `check_nt8_feed` cannot import `realtime.nt8_feed` - that module pulls
    pandas and the tool's whole claim is that it does not - so it restates
    `spool_path`'s rule. This imports the real one and compares them, which is
    the only thing standing between a future edit and a status card that
    reports a working publisher as MISSING.
    """
    from realtime.nt8_feed import spool_path                 # noqa: PLC0415

    d = _spool(tmp_path, ["NQ_1m.csv", "MNQ_1m.csv", "ES_1h.csv"])
    files = cf.spool_stats(d)["files"]

    cases = [("NQ", "1m"), ("NQ", "1h"), ("MNQ", "1m"), ("MNQ", "1h"),
             ("ES", "1h"), ("ES", "1m"), ("GC", "1m"), ("CL", "1h")]
    for symbol, tf in cases:
        mine = cf.satisfied_by(symbol, tf, files)
        theirs = spool_path(symbol, tf, d) or spool_path(symbol, "1m", d)
        if theirs is None:
            assert mine is None, f"{symbol} {tf}: I found {mine}, the feed none"
        else:
            assert mine is not None, \
                f"{symbol} {tf}: the feed resolves {theirs.name}, I found none"
            assert mine["name"] == theirs.name, \
                f"{symbol} {tf}: I say {mine['name']}, the feed says {theirs.name}"


def test_an_hourly_request_is_served_by_the_one_minute_spool(tmp_path):
    """
    The bug this function exists to prevent, pinned on its own.

    `closed_bars` takes a native `{SYM}_{TF}` file when one exists and
    otherwise reads `{SYM}_1m` and resamples. The first version of this card
    demanded `NQ_1h.csv`, reported a correct spool as MISSING, and would have
    sent an operator to fix a publisher that was working.
    """
    files = cf.spool_stats(_spool(tmp_path, ["NQ_1m.csv"]))["files"]
    hit = cf.satisfied_by("NQ", "1h", files)
    assert hit is not None
    assert hit["name"] == "NQ_1m.csv"
    assert hit["how"] == "resampled from 1m"


def test_a_micro_is_answered_by_its_parents_spool(tmp_path):
    """`spool_path` tries the symbol AND its full-size parent, one-directional."""
    files = cf.spool_stats(_spool(tmp_path, ["NQ_1m.csv"]))["files"]
    hit = cf.satisfied_by("MNQ", "1h", files)
    assert hit is not None and hit["name"] == "NQ_1m.csv"
    assert "parent NQ" in hit["via"]

    # ...and NOT the reverse. A spool of MNQ answers the loop and starves the
    # daemon, which asks for the certification symbol NQ.
    micro_only = cf.spool_stats(_spool(tmp_path / "b", ["MNQ_1m.csv"]))["files"]
    assert cf.satisfied_by("NQ", "1h", micro_only) is None


def test_a_native_file_beats_the_resampled_one(tmp_path):
    files = cf.spool_stats(_spool(tmp_path, ["NQ_1m.csv", "NQ_1h.csv"]))["files"]
    hit = cf.satisfied_by("NQ", "1h", files)
    assert hit["name"] == "NQ_1h.csv" and hit["how"] == "native"


# ==========================================================================
# the spool on disk
# ==========================================================================

def test_every_suffix_the_feed_reads_is_recognised(tmp_path):
    from realtime.nt8_feed import SPOOL_SUFFIXES               # noqa: PLC0415
    assert tuple(cf.SPOOL_SUFFIXES) == tuple(SPOOL_SUFFIXES), \
        "the restated suffix list has drifted from realtime.nt8_feed"

    names = [f"NQ_1m{s}" for s in SPOOL_SUFFIXES]
    stats = cf.spool_stats(_spool(tmp_path, names))
    assert len(stats["files"]) == len(names)
    assert stats["unparsed"] == []


def test_a_rolled_contract_spool_is_flagged_not_skipped(tmp_path):
    """
    `NQ12-26_1m.csv` is how a publisher rolled onto a PHYSICAL contract shows
    up: the listener accepts it, opens a new spool, and nothing reads it. A
    silent skip is what let that go unnoticed for a session, so it is reported.
    """
    stats = cf.spool_stats(_spool(tmp_path, ["NQ_1m.csv", "NQ12-26_1m.csv"]))
    assert [f["symbol"] for f in stats["files"]] == ["NQ"]
    assert stats["unparsed"] == ["NQ12-26_1m.csv"]


def test_a_missing_spool_directory_is_reported_not_raised(tmp_path):
    stats = cf.spool_stats(tmp_path / "nope")
    assert stats["exists"] is False and stats["files"] == []


def test_the_listener_names_the_spool_it_is_actually_writing(tmp_path):
    """
    `/health` wins over `$BT_NT8_SPOOL`. The two differ exactly when somebody
    changed the variable and did not restart the unit - which is the moment
    this question gets asked, and the moment a card reading the variable would
    describe a directory the listener is not filling.
    """
    os.environ["BT_NT8_SPOOL"] = str(tmp_path / "from_env")
    try:
        path, source = cf.resolve_spool({"spool_dir": "/from/health"})
        assert str(path) == "/from/health" and "listener" in source
        path, source = cf.resolve_spool(None)
        assert path == tmp_path / "from_env" and "BT_NT8_SPOOL" in source
    finally:
        os.environ.pop("BT_NT8_SPOOL", None)


# ==========================================================================
# the card must not contradict itself
# ==========================================================================

def test_the_process_check_follows_the_url():
    """
    Asked about a listener on one port while reporting on whoever holds 8000,
    the card printed `Listener Status: STOPPED` and `Listener Service: active`
    on consecutive lines - two true statements about two different processes,
    reading as one self-contradicting answer.
    """
    assert cf.port_of("http://localhost:8000/health") == 8000
    assert cf.port_of("http://127.0.0.1:9/health") == 9
    assert cf.port_of("http://localhost/health") == cf.DEFAULT_PORT
    assert cf.port_of("") == cf.DEFAULT_PORT

    port = _closed_port()
    card = cf.render(cf.collect(url=f"http://127.0.0.1:{port}/health"))
    assert "STOPPED" in card
    assert f"nothing is bound to port {port}" in card
    assert f"Port: {cf.DEFAULT_PORT}" not in card, \
        "the service line described a different process from the status line"


def test_an_unreachable_listener_does_not_claim_a_stream_is_not_posting(tmp_path):
    """
    With the listener down, whether a stream is POSTING is unobserved. Saying
    `NOT posting` states a fact this tool did not gather - and points an
    operator at the publisher when the thing that is down is the receiver.
    """
    os.environ["BT_NT8_SPOOL"] = str(_spool(tmp_path, ["NQ_1m.csv", "MNQ_1m.csv"]))
    try:
        card = cf.render(cf.collect(
            url=f"http://127.0.0.1:{_closed_port()}/health"))
    finally:
        os.environ.pop("BT_NT8_SPOOL", None)
    assert "cannot confirm" in card
    assert "NOT posting" not in card


# ==========================================================================
# staleness
# ==========================================================================

def test_the_three_freshness_levels():
    minute = 60.0
    assert cf.classify(12.0, minute, 300.0, False) == cf.STREAMING
    assert cf.classify(72.0, minute, 300.0, False) == cf.IDLE
    assert cf.classify(600.0, minute, 300.0, False) == cf.STALE
    assert cf.classify(None, minute, 300.0, False) == "UNKNOWN"


def test_the_listeners_own_verdict_is_shown_where_the_two_disagree():
    """
    The listener flags stale at 3 bar widths (180s on a 1m stream) and this
    tool escalates at 300s, so there is a two-minute band where they differ.
    Showing only one would make two tools on one box contradict each other
    about the same stream with nothing naming the disagreement.
    """
    label = cf.classify(240.0, 60.0, 300.0, listener_stale=True)
    assert label.startswith(cf.IDLE)
    assert "listener: STALE" in label

    # Past this tool's own threshold there is no disagreement left to report.
    assert cf.classify(600.0, 60.0, 300.0, listener_stale=True) == cf.STALE


def test_a_bar_inside_its_own_width_is_streaming():
    """An hourly bar 20 minutes old is mid-bar, not late."""
    assert cf.classify(1200.0, 3600.0, 300.0, False) == cf.STREAMING


# ==========================================================================
# parsing
# ==========================================================================

def test_timeframe_seconds():
    assert cf.tf_seconds("1m") == 60
    assert cf.tf_seconds("15m") == 900
    assert cf.tf_seconds("1h") == 3600
    assert cf.tf_seconds("1d") == 86400
    assert cf.tf_seconds("nonsense") is None
    assert cf.tf_seconds(None) is None


def test_iso_parsing_handles_z_and_offsets_and_junk():
    a = cf.parse_iso("2026-08-26T14:27:00Z")
    b = cf.parse_iso("2026-08-26T14:27:00+00:00")
    assert a == b and a is not None
    assert cf.parse_iso("not a date") is None
    assert cf.parse_iso(None) is None
    # Naive stamps are read as UTC rather than as local time: the spool is
    # UTC by contract, and a local reading would be silently hours out.
    assert cf.parse_iso("2026-08-26T14:27:00") == a


def test_sizes_and_ages_are_human():
    assert cf.size_text(0) == "0 B"
    assert cf.size_text(1536).endswith("KB")
    assert cf.age_text(12) == "12s"
    assert cf.age_text(300) == "5.0 min"
    assert cf.age_text(None) == "unknown"


# ==========================================================================
# the cheapness claim
# ==========================================================================

def test_the_tool_does_not_import_pandas():
    """
    The whole point of the file. pandas is 247ms on this box against 18ms for
    the standard library it needs, and a stray heavy import would cost that on
    every shell prompt with nothing failing.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import realtime.check_nt8_feed;"
         "heavy = [m for m in ('pandas','numpy','vectorbtpro','sklearn')"
         "         if m in sys.modules];"
         "print(','.join(heavy))" % str(REPO)],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", \
        f"check_nt8_feed pulled in: {out.stdout.strip()}"


# ==========================================================================
# the executable
# ==========================================================================

def test_it_runs_from_any_directory_and_exits_nonzero_when_unreachable():
    """
    `nt8-check` from somewhere that is not the repository. The module's own
    sys.path bootstrap is what is under test - `python3 realtime/x.py` puts
    realtime/ on the path, not the repository root.

    The exit code is the other half: 1 when the listener cannot be reached, so
    the alias is usable in a shell `&&` chain.
    """
    url = f"http://127.0.0.1:{_closed_port()}/health"
    out = subprocess.run(
        [sys.executable, str(REPO / "realtime" / "check_nt8_feed.py"),
         "--url", url],
        capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=120)
    assert out.returncode == 1, "unreachable listener must exit non-zero"
    assert "NINJATRADER 8 LIVE FEED STATUS" in out.stdout
    assert "STOPPED" in out.stdout
    assert "Traceback" not in out.stderr


def test_the_json_mode_is_machine_readable():
    srv, url = _serve(_health(streams=[_stream("NQ")]))
    try:
        out = subprocess.run(
            [sys.executable, str(REPO / "realtime" / "check_nt8_feed.py"),
             "--url", url, "--json"],
            capture_output=True, text=True, cwd=tempfile.gettempdir(),
            timeout=120)
        assert out.returncode == 0, out.stderr
        blob = json.loads(out.stdout)
        assert blob["health"]["payload"]["status"] == "HEALTHY"
    finally:
        srv.shutdown()


def test_it_never_writes_anything(tmp_path):
    """A status tool that creates what it reports on has changed its subject."""
    os.environ["BT_NT8_SPOOL"] = str(tmp_path / "spool_that_does_not_exist")
    try:
        before = sorted(p.name for p in tmp_path.iterdir())
        cf.collect(url=f"http://127.0.0.1:{_closed_port()}/health")
        assert sorted(p.name for p in tmp_path.iterdir()) == before
    finally:
        os.environ.pop("BT_NT8_SPOOL", None)


def main() -> int:
    """A script entry point, so this file behaves the same under both runners."""
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            import inspect
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
