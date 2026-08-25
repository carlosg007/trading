#!/usr/bin/env python3
"""
realtime.nt8_bar_listener - a push receiver for NT8 bars that WRITES THE SPOOL.

NinjaTrader posts one closed bar per request; this validates it and appends it
to the same `{SYMBOL}_{TF}.csv` spool `realtime/nt8_feed.py` already reads. The
live loop and the regime daemon then consume it through the feed seam they
already use, unchanged:

    NT8 (Windows) --POST /api/bars--> this listener --appends--> spool dir
                                                                    |
                                              realtime/nt8_feed.py  | reads
                                                                    v
                                   realtime/feed.py -> master_live.py
                                                    -> regime_daemon.py

WHY THIS WRITES THE SPOOL INSTEAD OF BEING ITS OWN BAR STORE
============================================================
`realtime/nt8_feed.py` chose a spool directory over an HTTP listener
deliberately, and says so: a listener is a port, a process to supervise and a
silent hole when it dies, while a file that stops growing is visible to `ls`,
survives a restart on either side and replays after an outage. It also says
what to do if a push transport is added anyway - **write this spool rather
than bypass it, so there is one format and one place to look when a bar is
missing.** This module is that transport.

A second bar store read directly by `load_symbol_bars` would have cost four
things that are not obvious until they are gone: `drop_forming_bar` (the one
anti-lookahead rule), the micro->parent alias, aggregation through the LAKE's
resampler (so a live 15m bar is assembled exactly as the certified one), and
the single `ts` conversion below. Writing the spool keeps all four, and keeps
`--feed nt8` meaning one thing.

The durable record is therefore the FILE. The in-memory ring buffer is an
index over what this process has accepted - it answers `/health` and rejects
duplicates - and losing it on restart loses nothing, because it is re-seeded
from the spool.

`ts` AND THE ONE SUBTRACTION THAT MUST NOT HAPPEN TWICE
======================================================
NinjaTrader stamps a bar with its CLOSE; this repository stamps the OPEN. That
conversion lives in `nt8_feed.read_spool` and NOWHERE ELSE, so this listener
writes `timestamp_utc` through UNCHANGED and records which end it means in the
file's `# stamp=` header. Converting here as well would shift every bar a full
period back the other way - and nothing would raise.

The default is `close`, because that is NT8's own and `nt8_feed.DEFAULT_STAMP`.
A publisher that posts OPEN times must say so - `"stamp": "open"` on the
payload, or `--stamp open` on this process. The convention is PRINTED at
startup, echoed on every accepted bar and reported by `/health`, because the
author of the NinjaScript on the other side of the mount is the person who has
to notice it is wrong, and a bar shifted one period still looks like a market.

**One file carries one convention.** A payload whose `stamp` disagrees with the
header already in the file is REFUSED rather than appended: `read_spool` takes
the last `# stamp=` it sees, so a file with two would silently re-interpret
every bar written before the switch.

WHAT IS REFUSED, AND WHY EACH ONE
=================================
* **A naive timestamp.** NT8 writes in the instrument's or the workstation's
  timezone unless the script converts, so a stamp with no offset is as likely
  to be New York as UTC. Guessed wrong, the series shifts by hours and still
  looks like a market: bars in order, prices sane, sessions the wrong length.
* **A timestamp off the timeframe's grid.** A 15m bar at 20:07 means the
  publisher's convention or clock is wrong. Accepted, it either lands in the
  wrong resample bin or reaches a strategy on a boundary the certification
  never used.
* **An incoherent bar** - `high` below `close`, `low` above `open`, a negative
  volume, a non-finite price. Nothing downstream re-checks OHLC; every
  indicator treats it as a fact.
* **An unknown timeframe**, checked against the lake's own table through
  `realtime.feed.tf_delta`, so a timeframe this repository cannot build is one
  this listener will not accept bars for.
* **A duplicate timestamp**, which is ACCEPTED-as-duplicate rather than
  errored: a publisher retrying after a network blip must not crash-loop, and
  re-posting a bar is idempotent. It is reported as `"status": "duplicate"` and
  counted, because a publisher that sends only duplicates is broken in a way
  that otherwise looks identical to a working one.

WHAT IS NOT DONE HERE
=====================
* **The forming-bar rule.** A written bar is not trusted to be a closed bar,
  but that check belongs to the reader and already runs there for every feed.
  Doing it here as well would put the rule in two places and let them drift.
  A NinjaScript switched to `Calculate.OnEachTick` will get its forming bars
  into the spool and `drop_forming_bar` will refuse them downstream.
* **Sending anything.** `live/dispatcher.py` remains the only module in this
  repository that puts an order on the wire.
* **Filling a gap.** A missing bar is a missing bar. Nothing is interpolated.

THE PORT IS A REAL EXPOSURE
===========================
This binds `0.0.0.0:8000` as specified, which means anything that can route to
this box can inject bars that live strategies decide on. Set
`$BT_NT8_LISTEN_TOKEN` to require an `X-NT8-Token` header, and prefer binding
to the interface the NT8 workstation is actually on.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mdlib.env import load_env                                     # noqa: E402
from realtime.feed import BAR_COLUMNS, FeedError, tf_delta         # noqa: E402
from realtime.nt8_feed import (DEFAULT_STAMP, STAMP_CONVENTIONS,   # noqa: E402
                               configured_spool_dir)

load_env()

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# The spool's header, derived from the feed's column contract rather than
# retyped, so the writer and the reader cannot drift apart on column ORDER.
SPOOL_COLUMNS = tuple(c for c in BAR_COLUMNS if c != "symbol")

# How many accepted bars per (symbol, timeframe) the in-memory index keeps.
# It exists to answer /health and to reject duplicates; the file is the record.
RING_CAPACITY = 1500

# `/health` reports STALE after this many bar-widths with nothing arriving.
DEFAULT_STALE_AFTER_BARS = 3.0

# Optional shared secret for `X-NT8-Token`. Unset means the endpoint is open.
TOKEN_VAR = "BT_NT8_LISTEN_TOKEN"

REQUIRED_FIELDS = ("symbol", "timeframe", "timestamp_utc",
                   "open", "high", "low", "close", "volume")


class BarRejected(ValueError):
    """A posted bar that will not be written, and the reason it will not."""

    def __init__(self, reason: str, status: int = 400) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def parse_timestamp(raw: Any) -> datetime:
    """
    An ISO-8601 stamp carrying an explicit offset -> an aware UTC datetime.

    A NAIVE stamp is refused rather than assumed to be UTC. See the module
    docstring: guessed wrong it moves the whole series by hours and the result
    still reads as a market.
    """
    if isinstance(raw, datetime):
        parsed = raw
    else:
        text = str(raw).strip()
        if not text:
            raise BarRejected("timestamp_utc is empty")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BarRejected(
                f"timestamp_utc {raw!r} is not ISO-8601 ({exc}). Write "
                f"e.g. 2026-08-25T20:00:00Z") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BarRejected(
            f"timestamp_utc {raw!r} carries no UTC offset. NT8 writes in the "
            f"instrument's or the workstation's timezone unless the script "
            f"converts, so this cannot be assumed to be UTC - guessed wrong "
            f"it shifts every bar by hours and still looks like a market. "
            f"Send ISO-8601 with `Z` or an explicit offset.")
    return parsed.astimezone(timezone.utc)


def _price(payload: dict, field: str) -> float:
    raw = payload[field]
    if isinstance(raw, bool):
        raise BarRejected(f"{field} is a boolean, not a price")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BarRejected(f"{field} {raw!r} is not a number") from exc
    if not math.isfinite(value):
        raise BarRejected(f"{field} is {raw!r}; a price must be finite")
    return value


def check_alignment(ts: datetime, tf: str) -> None:
    """
    An intraday bar must sit on its timeframe's grid.

    A 15m bar stamped 20:07 says the publisher's clock or its convention is
    wrong. Accepted, it lands in the wrong resample bin or reaches a strategy
    on a boundary no certification used - and both are invisible afterwards.
    Daily bars are exempt: their boundary is a session, not a UTC multiple.
    """
    width = tf_delta(tf)
    if width >= timedelta(days=1):
        return
    seconds = int(width.total_seconds())
    if seconds <= 0:
        return
    offset = int(ts.timestamp()) % seconds
    if offset or ts.microsecond:
        raise BarRejected(
            f"timestamp_utc {ts.isoformat().replace('+00:00', 'Z')} is not on "
            f"the {tf} grid (off by {offset}s). A bar stamped off its own "
            f"boundary means the publisher's convention or clock is wrong; "
            f"accepted it would resample into the wrong interval.")


def parse_bar(payload: Any, default_stamp: str = DEFAULT_STAMP) -> dict:
    """
    A posted JSON object -> a validated bar, or `BarRejected`.

    Returns `symbol`, `timeframe`, `ts` (aware UTC, UNCONVERTED - see the
    module docstring), `open/high/low/close`, `volume` and the `stamp`
    convention the poster means by `ts`.
    """
    if not isinstance(payload, dict):
        raise BarRejected(
            f"expected a JSON object with {list(REQUIRED_FIELDS)}, got "
            f"{type(payload).__name__}")

    missing = [f for f in REQUIRED_FIELDS if payload.get(f) is None]
    if missing:
        raise BarRejected(f"missing required field(s): {missing}")

    symbol = str(payload["symbol"]).strip().upper()
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
        raise BarRejected(f"symbol {payload['symbol']!r} is not a contract code")

    tf = str(payload["timeframe"]).strip()
    try:
        tf_delta(tf)
    except FeedError as exc:
        raise BarRejected(
            f"timeframe {tf!r} is not one this repository builds bars at "
            f"({exc}).") from exc

    # DECLARED vs INHERITED, and the difference decides a conflict below. A
    # payload that says `"stamp": "open"` is making a claim about its own
    # timestamps; one that says nothing has merely taken this process's
    # default, which must not be allowed to overrule a file's header.
    declared_stamp = payload.get("stamp") is not None
    stamp = str(payload.get("stamp") or default_stamp).strip().lower()
    if stamp not in STAMP_CONVENTIONS:
        raise BarRejected(
            f"stamp must be one of {STAMP_CONVENTIONS}; got {payload['stamp']!r}")

    ts = parse_timestamp(payload["timestamp_utc"])
    check_alignment(ts, tf)

    bar = {field: _price(payload, field)
           for field in ("open", "high", "low", "close")}

    volume = _price(payload, "volume")
    if volume < 0:
        raise BarRejected(f"volume {volume} is negative")

    # NOTHING DOWNSTREAM RE-CHECKS THIS. An incoherent bar reaches every
    # indicator as a fact, and a high below the close is not a market.
    if bar["high"] < bar["low"]:
        raise BarRejected(
            f"high {bar['high']} is below low {bar['low']}")
    if bar["high"] < max(bar["open"], bar["close"]):
        raise BarRejected(
            f"high {bar['high']} is below open/close "
            f"({bar['open']}/{bar['close']})")
    if bar["low"] > min(bar["open"], bar["close"]):
        raise BarRejected(
            f"low {bar['low']} is above open/close "
            f"({bar['open']}/{bar['close']})")

    bar.update(symbol=symbol, timeframe=tf, ts=ts, volume=volume, stamp=stamp,
               stamp_declared=declared_stamp)
    return bar


# --------------------------------------------------------------------------
# the spool writer, and the ring buffer over it
# --------------------------------------------------------------------------

def iso_z(ts: datetime) -> str:
    """UTC, ISO-8601, `Z` - the spelling the spool contract asks NT8 for."""
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value: float) -> str:
    """`repr` round-trips a float exactly; `%g` silently truncates 19855.25."""
    return repr(int(value)) if float(value).is_integer() else repr(float(value))


class BarSpool:
    """
    The in-memory ring buffer AND the spool writer, which are one object
    because they must not be able to disagree about what was accepted.

    A bar is written to the file and indexed under the same lock, so a
    duplicate check that passed can never be followed by a write that did not
    happen - and `/health` cannot report a bar the spool does not hold.
    """

    def __init__(self, spool_dir: Path | str | None = None,
                 stamp: str = DEFAULT_STAMP,
                 capacity: int = RING_CAPACITY,
                 stale_after_bars: float = DEFAULT_STALE_AFTER_BARS) -> None:
        if str(stamp).strip().lower() not in STAMP_CONVENTIONS:
            raise ValueError(
                f"stamp must be one of {STAMP_CONVENTIONS}; got {stamp!r}")
        self.spool_dir = configured_spool_dir(spool_dir)
        self.stamp = str(stamp).strip().lower()
        self.capacity = int(capacity)
        self.stale_after_bars = float(stale_after_bars)
        self.started_at = datetime.now(timezone.utc)
        self.last_post_at: datetime | None = None
        self.counters = {"accepted": 0, "duplicate": 0, "rejected": 0}
        self._keys: dict[tuple[str, str], dict] = {}
        self._lock = RLock()

    # -- files ------------------------------------------------------------

    def path_for(self, symbol: str, tf: str) -> Path:
        """`{SYMBOL}_{TF}.csv` - the name `nt8_feed.spool_path` looks for."""
        return self.spool_dir / f"{str(symbol).upper()}_{tf}.csv"

    def _seed(self, symbol: str, tf: str) -> dict:
        """
        Open a (symbol, timeframe) for writing, re-reading whatever the spool
        already holds.

        Two things come off the existing file and neither can be assumed: its
        DECLARED stamp convention, which is authoritative exactly as it is for
        `read_spool` (the NinjaScript is what knows which end it writes), and
        its recent timestamps, so a restarted listener does not re-append bars
        that are already on disk.
        """
        path = self.path_for(symbol, tf)
        state = {"path": path, "stamp": self.stamp,
                 "ring": deque(maxlen=self.capacity),
                 "seen": deque(maxlen=self.capacity),
                 "seen_set": set(), "declared": False, "newest_ts": None}

        if path.is_file():
            stamps: list[str] = []
            for line in path.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
                text = line.strip()
                if not text:
                    continue
                if text.startswith("#"):
                    token = text.lstrip("#").strip().lower()
                    if token.startswith("stamp="):
                        declared = token.split("=", 1)[1].strip()
                        if declared in STAMP_CONVENTIONS:
                            state["stamp"] = declared
                            state["declared"] = True
                    continue
                stamps.append(text.split(",", 1)[0].strip())
            for raw in stamps[-self.capacity:]:
                if raw.lower() == "ts":
                    continue
                try:
                    known = iso_z(parse_timestamp(raw))
                except (BarRejected, ValueError):
                    continue          # a hand-edited line is not this loop's
                self._remember(state, known)
                parsed = parse_timestamp(known)
                if state.get("newest_ts") is None or parsed > state["newest_ts"]:
                    state["newest_ts"] = parsed
        else:
            self.spool_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(f"# stamp={state['stamp']}\n")
                handle.write(",".join(SPOOL_COLUMNS) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            state["declared"] = True

        self._keys[(symbol, tf)] = state
        return state

    @staticmethod
    def _remember(state: dict, key: str) -> None:
        """Index a timestamp, evicting the oldest in step with the ring."""
        if key in state["seen_set"]:
            return
        if len(state["seen"]) == state["seen"].maxlen:
            state["seen_set"].discard(state["seen"][0])
        state["seen"].append(key)
        state["seen_set"].add(key)

    # -- ingest -----------------------------------------------------------

    def append(self, bar: dict) -> dict:
        """
        Write one validated bar and index it. Returns the ingest record.

        `status` is `ok` for a bar that reached the file and `duplicate` for a
        timestamp already there. A duplicate is not an error: a publisher
        retrying a request whose response was lost must be able to do so
        without a crash loop, and re-posting a bar is idempotent.
        """
        symbol, tf = bar["symbol"], bar["timeframe"]
        with self._lock:
            self.last_post_at = datetime.now(timezone.utc)
            state = self._keys.get((symbol, tf)) or self._seed(symbol, tf)

            # ONE FILE, ONE CONVENTION. `read_spool` takes the LAST `# stamp=`
            # header it sees, so appending an open-stamped bar under a
            # close-stamped header would re-interpret every bar already
            # written - silently, and a period at a time.
            #
            # Only a DECLARED conflict is refused. A payload that said nothing
            # about its stamp carries this process's default, and the FILE's
            # header outranks that for the same reason it outranks it in
            # `read_spool`: the NinjaScript is what knows which end it writes,
            # and an operator restarting this listener with a stale `--stamp`
            # should not thereby relabel a spool.
            if bar["stamp"] != state["stamp"] and bar.get("stamp_declared"):
                raise BarRejected(
                    f"{state['path'].name} is {state['stamp']}-stamped and "
                    f"this bar declares stamp={bar['stamp']!r}. One spool "
                    f"file carries one convention: mixing them re-reads every "
                    f"bar already in the file. Post to a new file or restate "
                    f"the header deliberately.",
                    status=409)

            bar = {**bar, "stamp": state["stamp"]}
            key = iso_z(bar["ts"])
            newest = state.get("newest_ts")
            if key in state["seen_set"]:
                self.counters["duplicate"] += 1
                return {"status": "duplicate", "symbol": symbol,
                        "bar_ts": key, "timeframe": tf,
                        "stamp": state["stamp"], "spool": state["path"].name,
                        "written": False}

            line = ",".join([key] + [_number(bar[c]) for c in SPOOL_COLUMNS[1:]])
            with open(state["path"], "a", encoding="utf-8", newline="") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            state["ring"].append({"ts": key, "received_at": iso_z(self.last_post_at),
                                  **{c: bar[c] for c in SPOOL_COLUMNS[1:]}})
            out_of_order = newest is not None and bar["ts"] < newest
            if newest is None or bar["ts"] > newest:
                state["newest_ts"] = bar["ts"]
            self._remember(state, key)
            self.counters["accepted"] += 1

            # REPORTED, never corrected. `normalize_bars` sorts on read, so an
            # out-of-order arrival is harmless to the frame - but it means the
            # publisher is replaying or its clock moved, and that is a fact
            # about the feed an operator should see rather than infer.
            return {"status": "ok", "symbol": symbol, "bar_ts": key,
                    "timeframe": tf, "stamp": state["stamp"],
                    "spool": state["path"].name, "written": True,
                    "out_of_order": out_of_order,
                    "buffered": len(state["ring"])}

    # -- health -----------------------------------------------------------

    def health(self, now: datetime | None = None) -> dict:
        """
        What `trading-watchdog` reads.

        `last_bar_utc` is the MARKET's clock and `last_post_utc` is this
        process's. A publisher looping over a dead feed keeps the second fresh
        forever while the first stops, and telling those apart is the job - so
        both are on the record and the status is drawn on the first.

        STALE is also what a shut market looks like. It says "no bar for N bar
        widths", which is a true statement in both cases; the watchdog owns the
        session calendar that separates them.
        """
        stamp_now = now or datetime.now(timezone.utc)
        with self._lock:
            symbols, newest = [], None
            for (symbol, tf), state in sorted(self._keys.items()):
                if not state["ring"]:
                    continue
                last = state["ring"][-1]["ts"]
                opened = parse_timestamp(last)
                width = tf_delta(tf)
                # Measured from when the bar CLOSED. A bar stamped at its open
                # is always a full width "old" by any other reading, and a
                # staleness test drawn on that fires every healthy cycle.
                closed = opened if state["stamp"] == "close" else opened + width
                age = (stamp_now - closed).total_seconds()
                limit = self.stale_after_bars * width.total_seconds()
                symbols.append({"symbol": symbol, "timeframe": tf,
                                "last_bar_utc": last,
                                "bar_age_seconds": round(age, 3),
                                "stale": age > limit,
                                "stamp": state["stamp"],
                                "buffered": len(state["ring"]),
                                "spool": state["path"].name})
                if newest is None or last > newest:
                    newest = last

            if not symbols:
                status = "STARVED"
            elif any(s["stale"] for s in symbols):
                status = "STALE"
            else:
                status = "HEALTHY"

            return {"status": status,
                    "last_bar_utc": newest,
                    "symbols_active": sorted({s["symbol"] for s in symbols}),
                    "last_post_utc": iso_z(self.last_post_at)
                                     if self.last_post_at else None,
                    "started_at_utc": iso_z(self.started_at),
                    "uptime_seconds": round(
                        (stamp_now - self.started_at).total_seconds(), 1),
                    "spool_dir": str(self.spool_dir),
                    "default_stamp": self.stamp,
                    "stale_after_bars": self.stale_after_bars,
                    "counters": dict(self.counters),
                    "streams": symbols}


# --------------------------------------------------------------------------
# the ASGI app
# --------------------------------------------------------------------------

def create_app(spool: BarSpool | None = None, token: str | None = None):
    """
    The Starlette app, with the spool injected so tests never open a port.

    Starlette rather than FastAPI: it is the ASGI layer FastAPI is built on,
    it is already pinned in `requirements.txt` and already installed, and this
    is two routes over a JSON body. FastAPI is not installed and CLAUDE.md
    forbids adding it here; the endpoints and the payloads are as specified.
    """
    from starlette.applications import Starlette                  # noqa: PLC0415
    from starlette.responses import JSONResponse                  # noqa: PLC0415
    from starlette.routing import Route                           # noqa: PLC0415

    buffer = spool if spool is not None else BarSpool()
    secret = token if token is not None else (
        os.environ.get(TOKEN_VAR) or "").strip()

    def authorized(request) -> bool:
        if not secret:
            return True
        return request.headers.get("x-nt8-token", "") == secret

    async def post_bars(request):
        if not authorized(request):
            return JSONResponse(
                {"status": "rejected",
                 "reason": f"missing or wrong X-NT8-Token (${TOKEN_VAR} is "
                           f"set on this listener)"}, status_code=401)
        try:
            payload = json.loads(await request.body() or b"")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            buffer.counters["rejected"] += 1
            return JSONResponse({"status": "rejected",
                                 "reason": f"body is not JSON: {exc}"},
                                status_code=400)

        # A list is accepted so a publisher can drain a backlog after a
        # reconnect in one request. Each bar is judged on its own: a bad one
        # in a batch must not discard the good ones around it, and it must not
        # be quietly dropped either.
        batch = payload if isinstance(payload, list) else [payload]
        if not batch:
            return JSONResponse({"status": "rejected",
                                 "reason": "empty batch"}, status_code=400)

        results, worst = [], 200
        for item in batch:
            try:
                results.append(buffer.append(parse_bar(item, buffer.stamp)))
            except BarRejected as exc:
                buffer.counters["rejected"] += 1
                results.append({"status": "rejected", "reason": exc.reason,
                                "symbol": (item or {}).get("symbol")
                                          if isinstance(item, dict) else None,
                                "bar_ts": (item or {}).get("timestamp_utc")
                                          if isinstance(item, dict) else None})
                worst = max(worst, exc.status)

        if not isinstance(payload, list):
            return JSONResponse(results[0], status_code=worst)
        return JSONResponse({"status": "ok" if worst == 200 else "partial",
                             "accepted": sum(r["status"] == "ok"
                                             for r in results),
                             "results": results}, status_code=worst)

    async def health(request):
        report = buffer.health()
        # 200 while HEALTHY or STALE, 503 while STARVED: a watchdog polling a
        # listener that has never received a bar should see a failing check,
        # while a quiet overnight tape should not page anybody. The `status`
        # field is the one to alert on.
        code = 503 if report["status"] == "STARVED" else 200
        return JSONResponse(report, status_code=code)

    app = Starlette(routes=[
        Route("/api/bars", post_bars, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ])
    app.state.spool = buffer
    return app


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="realtime/nt8_bar_listener.py",
        description="Receive closed bars pushed by NT8 and append them to the "
                    "spool `realtime/nt8_feed.py` reads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The live loop then reads them with `--feed nt8`. Nothing here "
               "sends an order.")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--spool-dir", default=None,
                    help=f"where bars are written. Default: $BT_NT8_SPOOL, "
                         f"else the NT8 feed's own default")
    ap.add_argument("--stamp", default=DEFAULT_STAMP, choices=STAMP_CONVENTIONS,
                    help="what a posted `timestamp_utc` MEANS when the payload "
                         "does not say: the bar's close (NinjaTrader's own "
                         "convention, the default) or its open (this "
                         "repository's). Written into each spool file's "
                         "header; the conversion itself happens on READ, once")
    ap.add_argument("--stale-after-bars", type=float,
                    default=DEFAULT_STALE_AFTER_BARS,
                    help="/health reports STALE after this many bar widths "
                         "without a bar")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    spool = BarSpool(spool_dir=args.spool_dir, stamp=args.stamp,
                     stale_after_bars=args.stale_after_bars)
    spool.spool_dir.mkdir(parents=True, exist_ok=True)

    secret = (os.environ.get(TOKEN_VAR) or "").strip()
    print(f"[nt8_listener] spool      {spool.spool_dir}")
    print(f"[nt8_listener] listening  http://{args.host}:{args.port}"
          f"  (POST /api/bars, GET /health)")
    print(f"[nt8_listener] ts means   the bar's {spool.stamp.upper()} unless a "
          f"payload says otherwise")
    if spool.stamp == "close":
        print("[nt8_listener]            NT8's own convention; bars are shifted "
              "back one timeframe on READ, in nt8_feed.read_spool, once.")
    print(f"[nt8_listener] auth       "
          f"{'X-NT8-Token required' if secret else 'OPEN - anything that can '
             'route here can inject bars. Set $' + TOKEN_VAR + '.'}")
    print("[nt8_listener] consume with: python3 master_live.py --dry-run "
          "--once --feed nt8", flush=True)

    import uvicorn                                                # noqa: PLC0415
    uvicorn.run(create_app(spool), host=args.host, port=args.port,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
