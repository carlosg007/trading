#!/usr/bin/env python3
"""
test_nt8_listener.py — the NT8 push receiver: that a posted bar lands in the
spool the live feed already reads, that it lands on the timestamp this
repository means, and that the bars nobody should trade on are refused at the
door rather than downstream.

Location:  ~/src/trading/tests/test_nt8_listener.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_nt8_listener.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_nt8_listener.py

EVERY CASE FAILS THROUGH `assert`. Nothing here opens a port: the app is
exercised through Starlette's `TestClient`, and every spool write goes to
`tmp_path`. Inner helpers are named `_check_*` — a module-level `test_*` whose
only argument is defaulted gets COLLECTED and run outside its fixtures, which
is how a suite ends up writing real files onto the NFS mount.

WHAT THIS COVERS, and why each case is here rather than assumed:

  * **THE ROUND TRIP IS THE POINT.** A bar posted to `/api/bars` has to come
    back out of `realtime.nt8_feed`, because writing the spool instead of
    holding a private buffer is the entire design. The assertion is on the
    FEED's output, not on the file's bytes.
  * **THE STAMP CONVENTION, WHICH SILENTLY RUINS EVERYTHING.** NT8 stamps a
    bar's CLOSE; this repository stamps the OPEN. The listener writes `ts`
    through unchanged and declares the convention in the file header, so the
    single subtraction stays in `read_spool`. Applied twice, or not at all,
    every bar shifts a period and nothing raises — so the test asserts the
    timestamp that reaches a strategy, end to end.
  * **ONE FILE, ONE CONVENTION.** `read_spool` takes the last `# stamp=` it
    sees, so an open-stamped bar appended under a close-stamped header would
    re-read every bar already written. Refused.
  * **A NAIVE TIMESTAMP IS REFUSED.** Assumed to be UTC and wrong, the series
    shifts by hours and still looks like a market.
  * **DUPLICATES ARE IDEMPOTENT, NOT ERRORS.** A publisher retrying a request
    whose response was lost must not crash-loop, and must not double-write.
  * **AN INCOHERENT BAR IS REFUSED.** Nothing downstream re-checks OHLC; every
    indicator treats a high below the close as a fact.
  * **A RESTART DOES NOT RE-APPEND.** The ring buffer is memory; the file is
    the record, and a fresh process re-seeds its duplicate index from it.
  * **`/health` SEPARATES THE TWO CLOCKS.** `last_bar_utc` is the market's and
    `last_post_utc` is the process's; a publisher looping over a dead feed
    keeps the second fresh forever while the first stops.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.nt8_bar_listener import (BarRejected, BarSpool,      # noqa: E402
                                       SPOOL_COLUMNS, create_app,
                                       parse_bar, parse_timestamp)
from realtime.nt8_feed import NT8BarFeed, read_spool, spool_path    # noqa: E402


def _bar(ts: str = "2026-08-25T20:00:00Z", symbol: str = "NQ",
         tf: str = "1m", **overrides) -> dict:
    """The payload shape NT8 posts, as documented in the listener."""
    payload = {"symbol": symbol, "timeframe": tf, "timestamp_utc": ts,
               "open": 19850.0, "high": 19855.5, "low": 19848.25,
               "close": 19852.0, "volume": 120}
    payload.update(overrides)
    return payload


def _client(tmp_path: Path, **kwargs):
    from starlette.testclient import TestClient
    spool = BarSpool(spool_dir=tmp_path, **kwargs)
    return TestClient(create_app(spool, token="")), spool


# --------------------------------------------------------------------------
# ingestion, and the round trip that is the whole design
# --------------------------------------------------------------------------

def test_posted_bar_is_accepted_and_acknowledged(tmp_path: Path) -> None:
    """The documented response, exactly: status, symbol, bar_ts."""
    client, _ = _client(tmp_path)
    response = client.post("/api/bars", json=_bar())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["symbol"] == "NQ"
    assert body["bar_ts"] == "2026-08-25T20:00:00Z"
    assert body["written"] is True


def test_posted_bar_lands_in_the_spool_the_live_feed_reads(
        tmp_path: Path) -> None:
    """
    THE ROUND TRIP. The listener writes the NT8 spool rather than a private
    store, so a pushed bar has to come back out of `NT8BarFeed` — which is what
    `master_live.load_symbol_bars(source="nt8")` reads. Asserted on the feed's
    output, because the file's bytes are an implementation detail and the
    frame a strategy sees is not.
    """
    client, _ = _client(tmp_path)
    for minute in range(5):
        stamp = (datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc)
                 + timedelta(minutes=minute))
        assert client.post("/api/bars", json=_bar(
            ts=stamp.isoformat().replace("+00:00", "Z"),
            close=19852.0 + minute,
            high=19855.5 + minute)).status_code == 200

    assert spool_path("NQ", "1m", tmp_path) == tmp_path / "NQ_1m.csv"

    feed = NT8BarFeed(spool_dir=tmp_path)
    # `now` is long after these bars, so none of them is still forming.
    bars, sources = feed.closed_bars(["NQ"], "1m", lookback_bars=50)
    assert "NQ" in bars, f"the pushed bars did not reach the feed: {sources}"
    assert len(bars["NQ"]) == 5
    assert list(bars["NQ"].columns)[:7] == ["ts", "symbol", "open", "high",
                                            "low", "close", "volume"]
    assert bars["NQ"]["close"].iloc[-1] == pytest.approx(19856.0)


def test_close_stamp_is_converted_once_on_read_not_twice(
        tmp_path: Path) -> None:
    """
    THE FIELD THAT SILENTLY RUINS EVERYTHING.

    NT8 stamps the bar that ran 19:59–20:00 as `20:00`. The listener writes
    that through UNCHANGED and declares `# stamp=close`; `read_spool` does the
    one subtraction. A bar posted at 20:00 must therefore reach a strategy as
    the bar OPENING at 19:59 — converted here as well it would arrive at
    19:58, and not converted at all it would arrive at 20:00. Both look like a
    market.
    """
    client, _ = _client(tmp_path, stamp="close")
    client.post("/api/bars", json=_bar(ts="2026-08-25T20:00:00Z", tf="1m"))

    written = (tmp_path / "NQ_1m.csv").read_text().splitlines()
    assert written[0] == "# stamp=close", "the convention must be IN the file"
    assert written[1] == ",".join(SPOOL_COLUMNS)
    assert written[2].startswith("2026-08-25T20:00:00Z"), \
        "the listener must not convert; read_spool owns that subtraction"

    bars, _ = read_spool(tmp_path / "NQ_1m.csv", "1m")
    assert str(bars["ts"].iloc[0]) == "2026-08-25 19:59:00+00:00"


def test_open_stamp_is_declared_and_not_shifted(tmp_path: Path) -> None:
    """A publisher that posts OPEN times says so, and its bars are not moved."""
    client, _ = _client(tmp_path, stamp="open")
    client.post("/api/bars", json=_bar(ts="2026-08-25T20:00:00Z", tf="1m"))

    assert (tmp_path / "NQ_1m.csv").read_text().splitlines()[0] == "# stamp=open"
    bars, convention = read_spool(tmp_path / "NQ_1m.csv", "1m")
    assert convention == "open"
    assert str(bars["ts"].iloc[0]) == "2026-08-25 20:00:00+00:00"


def test_one_file_carries_one_convention(tmp_path: Path) -> None:
    """
    `read_spool` takes the LAST `# stamp=` header it sees, so appending an
    open-stamped bar into a close-stamped file would re-interpret every bar
    already in it — a period at a time, silently. Refused with 409.
    """
    client, _ = _client(tmp_path, stamp="close")
    assert client.post("/api/bars", json=_bar()).status_code == 200
    response = client.post("/api/bars", json=_bar(
        ts="2026-08-25T20:01:00Z", stamp="open"))
    assert response.status_code == 409, response.text
    assert "one convention" in response.json()["reason"]
    assert len((tmp_path / "NQ_1m.csv").read_text().splitlines()) == 3


def test_a_files_declared_stamp_beats_the_processs_default(
        tmp_path: Path) -> None:
    """
    The NinjaScript is what knows which end it writes, so an existing file's
    header wins over this process's `--stamp` — the same precedence
    `read_spool` applies. Otherwise restarting the listener with the wrong
    flag would append bars under a convention the file does not declare.
    """
    (tmp_path / "NQ_1m.csv").write_text(
        "# stamp=open\n" + ",".join(SPOOL_COLUMNS) + "\n")
    client, _ = _client(tmp_path, stamp="close")

    # Says nothing about its stamp: it inherits, so the file's header wins and
    # the bar is written under `open` without being moved.
    body = client.post("/api/bars", json=_bar()).json()
    assert body["status"] == "ok"
    assert body["stamp"] == "open"
    bars, convention = read_spool(tmp_path / "NQ_1m.csv", "1m")
    assert convention == "open"
    assert str(bars["ts"].iloc[0]) == "2026-08-25 20:00:00+00:00"

    # DECLARES the other one: that is a claim about its own timestamps, and it
    # conflicts with the file. Refused rather than reconciled.
    clash = client.post("/api/bars",
                        json=_bar(ts="2026-08-25T20:01:00Z", stamp="close"))
    assert clash.status_code == 409, clash.text


# --------------------------------------------------------------------------
# what is refused, and why
# --------------------------------------------------------------------------

def test_naive_timestamp_is_refused_not_assumed_utc(tmp_path: Path) -> None:
    """
    NT8 writes in the instrument's or the workstation's timezone unless the
    script converts. Guessed wrong the whole series shifts by hours and still
    looks like a market: bars in order, prices sane, sessions the wrong length.
    """
    client, _ = _client(tmp_path)
    response = client.post("/api/bars",
                           json=_bar(ts="2026-08-25T20:00:00"))
    assert response.status_code == 400, response.text
    assert "no UTC offset" in response.json()["reason"]
    assert not (tmp_path / "NQ_1m.csv").exists()

    with pytest.raises(BarRejected, match="no UTC offset"):
        parse_timestamp("2026-08-25 20:00:00")
    assert parse_timestamp("2026-08-25T16:00:00-04:00").hour == 20


def test_incoherent_bars_are_refused(tmp_path: Path) -> None:
    """
    Nothing downstream re-checks OHLC. A high below the close reaches every
    indicator as a fact, and no error is ever raised on it.
    """
    client, _ = _client(tmp_path)
    for bad, why in (({"high": 19800.0}, "high below open/close"),
                     ({"low": 19900.0}, "low above open/close"),
                     ({"volume": -5}, "negative volume"),
                     ({"close": "n/a"}, "non-numeric price")):
        response = client.post("/api/bars", json=_bar(**bad))
        assert response.status_code == 400, f"{why} was accepted: {response.text}"
    assert not (tmp_path / "NQ_1m.csv").exists()


def test_missing_fields_and_unknown_timeframes_are_refused(
        tmp_path: Path) -> None:
    """
    The timeframe is checked against the LAKE's own table, so a timeframe this
    repository cannot build bars at is one the listener will not accept.
    """
    client, _ = _client(tmp_path)
    incomplete = _bar()
    incomplete.pop("volume")
    response = client.post("/api/bars", json=incomplete)
    assert response.status_code == 400
    assert "volume" in response.json()["reason"]

    response = client.post("/api/bars", json=_bar(tf="7m"))
    assert response.status_code == 400
    assert "7m" in response.json()["reason"]


def test_misaligned_intraday_timestamp_is_refused(tmp_path: Path) -> None:
    """
    A 15m bar stamped 20:07 means the publisher's clock or convention is
    wrong. Accepted it resamples into the wrong interval, or reaches a strategy
    on a boundary no certification ever used.
    """
    client, _ = _client(tmp_path)
    response = client.post("/api/bars",
                           json=_bar(ts="2026-08-25T20:07:00Z", tf="15m"))
    assert response.status_code == 400
    assert "grid" in response.json()["reason"]
    assert client.post("/api/bars",
                       json=_bar(ts="2026-08-25T20:15:00Z",
                                 tf="15m")).status_code == 200


def test_body_that_is_not_json_is_refused(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    response = client.post("/api/bars", content=b"not json",
                           headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert "not JSON" in response.json()["reason"]


# --------------------------------------------------------------------------
# duplicates, batches, restarts
# --------------------------------------------------------------------------

def test_duplicate_is_idempotent_and_writes_nothing(tmp_path: Path) -> None:
    """
    A publisher retrying a request whose response was lost must not crash-loop
    and must not double-write. Reported as `duplicate` rather than errored, and
    COUNTED — a publisher sending only duplicates is broken in a way that
    otherwise looks exactly like a working one.
    """
    client, spool = _client(tmp_path)
    assert client.post("/api/bars", json=_bar()).json()["status"] == "ok"
    repeat = client.post("/api/bars", json=_bar())
    assert repeat.status_code == 200
    assert repeat.json() == {**repeat.json(), "status": "duplicate",
                             "symbol": "NQ", "bar_ts": "2026-08-25T20:00:00Z",
                             "written": False}
    assert len((tmp_path / "NQ_1m.csv").read_text().splitlines()) == 3
    assert spool.counters == {"accepted": 1, "duplicate": 1, "rejected": 0}


def test_restart_reseeds_duplicates_from_the_file(tmp_path: Path) -> None:
    """
    The ring buffer is memory and the FILE is the record. A restarted listener
    that had forgotten what is on disk would re-append every bar the publisher
    replayed after the outage — and `normalize_bars` would dedupe it on read,
    so the spool would grow without bound and nothing would say why.
    """
    client, _ = _client(tmp_path)
    client.post("/api/bars", json=_bar())

    fresh, spool = _client(tmp_path)               # a new process's buffer
    assert fresh.post("/api/bars", json=_bar()).json()["status"] == "duplicate"
    assert spool.counters["accepted"] == 0
    assert len((tmp_path / "NQ_1m.csv").read_text().splitlines()) == 3


def test_batch_post_judges_each_bar_on_its_own(tmp_path: Path) -> None:
    """
    A list drains a backlog in one request after a reconnect. One bad bar must
    not discard the good ones around it, and must not be quietly dropped
    either — it comes back in `results` with its reason.
    """
    client, _ = _client(tmp_path)
    response = client.post("/api/bars", json=[
        _bar(ts="2026-08-25T20:00:00Z"),
        _bar(ts="2026-08-25T20:01:00", close=19853.0),      # naive: refused
        _bar(ts="2026-08-25T20:02:00Z", close=19854.0),
    ])
    body = response.json()
    assert body["status"] == "partial"
    assert body["accepted"] == 2
    assert [r["status"] for r in body["results"]] == ["ok", "rejected", "ok"]
    assert "no UTC offset" in body["results"][1]["reason"]
    assert len((tmp_path / "NQ_1m.csv").read_text().splitlines()) == 4


def test_out_of_order_arrival_is_reported_not_corrected(
        tmp_path: Path) -> None:
    """
    `normalize_bars` sorts on read, so an out-of-order arrival is harmless to
    the frame — but it means the publisher is replaying or its clock moved,
    and that is a fact about the feed rather than something to silently fix.
    """
    client, _ = _client(tmp_path)
    client.post("/api/bars", json=_bar(ts="2026-08-25T20:05:00Z"))
    late = client.post("/api/bars", json=_bar(ts="2026-08-25T20:04:00Z"))
    assert late.json()["status"] == "ok"
    assert late.json()["out_of_order"] is True


def test_a_rejection_is_counted_once(tmp_path: Path) -> None:
    """
    The counters are what tells a broken publisher from a quiet one, so they
    have to be countable. A stamp conflict raises inside `append` and is caught
    in the route; incrementing in both places double-counted it, which reads as
    twice the failure rate on the one number an operator would act on.
    """
    client, spool = _client(tmp_path, stamp="close")
    client.post("/api/bars", json=_bar())
    assert client.post("/api/bars", json=_bar(ts="2026-08-25T20:01:00Z",
                                              stamp="open")).status_code == 409
    assert client.post("/api/bars", json=_bar(ts="nonsense")).status_code == 400
    assert spool.counters == {"accepted": 1, "duplicate": 0, "rejected": 2}


def test_out_of_order_survives_a_restart(tmp_path: Path) -> None:
    """
    The high-water mark is re-seeded from the file with the duplicate index.
    Without it the first bar after a restart always reads as in-order however
    far back it is — which is exactly the bar a replaying publisher sends.
    """
    client, _ = _client(tmp_path)
    client.post("/api/bars", json=_bar(ts="2026-08-25T20:05:00Z"))

    fresh, _ = _client(tmp_path)
    late = fresh.post("/api/bars", json=_bar(ts="2026-08-25T20:04:00Z"))
    assert late.json()["status"] == "ok"
    assert late.json()["out_of_order"] is True, \
        "a restarted listener forgot where the tape had got to"


# --------------------------------------------------------------------------
# /health, for the watchdog
# --------------------------------------------------------------------------

def test_health_before_any_bar_is_starved_not_healthy(tmp_path: Path) -> None:
    """
    A listener that has never received a bar is not healthy, and 200/HEALTHY
    there is a watchdog that can never fire. STARVED, and a failing HTTP check.
    """
    client, _ = _client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "STARVED"
    assert body["last_bar_utc"] is None
    assert body["symbols_active"] == []


def test_health_reports_the_documented_shape(tmp_path: Path) -> None:
    """The three keys `trading-watchdog` reads, on a live-looking bar."""
    client, spool = _client(tmp_path)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    client.post("/api/bars", json=_bar(
        ts=now.isoformat().replace("+00:00", "Z")))

    # The same contract at a second timeframe: one stream each, ONE symbol.
    # A watchdog reading `symbols_active` as a set of contracts should not see
    # NQ twice because somebody publishes it at 1m and 15m.
    client.post("/api/bars", json=_bar(
        ts=now.replace(minute=now.minute // 15 * 15).isoformat()
              .replace("+00:00", "Z"), tf="15m"))

    body = client.get("/health").json()
    assert body["status"] == "HEALTHY"
    assert body["last_bar_utc"] == now.isoformat().replace("+00:00", "Z")
    assert body["symbols_active"] == ["NQ"]
    assert len(body["streams"]) == 2
    assert body["spool_dir"] == str(tmp_path)


def test_health_separates_the_market_clock_from_the_process_clock(
        tmp_path: Path) -> None:
    """
    A publisher looping over a dead feed keeps `last_post_utc` fresh forever
    while `bar_ts` stops. Telling those apart is the job, so both are on the
    record and the STATUS is drawn on the bar's clock.
    """
    client, spool = _client(tmp_path, stale_after_bars=3.0)
    client.post("/api/bars", json=_bar(ts="2026-08-25T20:00:00Z"))

    stale_now = datetime(2026, 8, 25, 20, 30, tzinfo=timezone.utc)
    report = spool.health(now=stale_now)
    assert report["status"] == "STALE", "30 min of silence on a 1m feed"
    assert report["last_bar_utc"] == "2026-08-25T20:00:00Z"
    assert report["last_post_utc"] is not None, "the process clock kept moving"
    assert report["streams"][0]["bar_age_seconds"] == pytest.approx(1800.0)

    fresh_now = datetime(2026, 8, 25, 20, 1, tzinfo=timezone.utc)
    assert spool.health(now=fresh_now)["status"] == "HEALTHY"


# --------------------------------------------------------------------------
# the port is a real exposure
# --------------------------------------------------------------------------

def test_token_gate_when_configured(tmp_path: Path) -> None:
    """
    Binding 0.0.0.0 means anything that can route here can inject bars that
    live strategies decide on. The gate is optional, and it works when set.
    """
    from starlette.testclient import TestClient
    client = TestClient(create_app(BarSpool(spool_dir=tmp_path), token="s3cr3t"))
    assert client.post("/api/bars", json=_bar()).status_code == 401
    assert client.post("/api/bars", json=_bar(),
                       headers={"X-NT8-Token": "s3cr3t"}).status_code == 200


# --------------------------------------------------------------------------
# the listener does not become a second bar store
# --------------------------------------------------------------------------

def test_listener_holds_no_private_store_and_sends_nothing() -> None:
    """
    Asserted on the SOURCE rather than trusted. The listener writes the spool
    `nt8_feed` reads; a private `data/live_bars` store would have bypassed the
    forming-bar rule, the micro alias, the lake's resampler and the single `ts`
    conversion. And nothing in `realtime/` other than `live/dispatcher.py` puts
    anything on the wire.
    """
    source = (REPO_ROOT / "realtime" / "nt8_bar_listener.py").read_text()
    assert "data/live_bars" not in source
    assert "databento" not in source.lower()
    for sender in ("requests.post", "httpx.post", "urlopen",
                   "send_execution_signal"):
        assert sender not in source, f"the listener must not send: {sender}"
    # THE ONE SUBTRACTION HAPPENS IN `read_spool`, ONCE. The listener may ask
    # `tf_delta` for a width — to validate a timeframe, to check grid
    # alignment, to age a bar for /health — but it must never move a
    # timestamp by one, or every bar shifts a period back the other way and
    # nothing raises.
    for shift in ("- tf_delta", "-tf_delta", "- width", "+ width"):
        assert shift not in source.replace("opened + width", ""), \
            f"the listener must not shift a bar's ts: {shift}"


def test_load_symbol_bars_accepts_a_named_source() -> None:
    """
    `load_symbol_bars(..., source="nt8")` is what `--feed nt8` gets, and an
    INJECTED feed still wins — re-resolving per call is how a run reads one
    tape while its startup banner names another.
    """
    import inspect

    import master_live
    signature = inspect.signature(master_live.load_symbol_bars)
    assert "source" in signature.parameters
    assert signature.parameters["source"].default is None

    class _Feed:
        def closed_bars(self, symbols, tf, lookback_bars):
            return ({"NQ": "injected"}, {"NQ": "NQ"})

    bars, _ = master_live.load_symbol_bars(["NQ"], "1m", 10, feed=_Feed(),
                                           source="lake")
    assert bars == {"NQ": "injected"}, "an injected feed must win"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
