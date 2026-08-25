#!/usr/bin/env python3
"""
data_pull.databento_live - the newest 1-minute bars, from the vendor.

This is the only module in the live path that talks to Databento, because
`data_pull/` is the only layer in this repository that touches a vendor API.
`realtime/feed.py` consumes it through a one-method contract and never imports
`databento` itself.

    from data_pull.databento_live import DatabentoBarClient
    client.minute_bars("NQ", 900)     # -> DataFrame of 1m bars, UTC

WHY THE HISTORICAL ENDPOINT AND NOT THE LIVE ONE
================================================
`databento.Live` is a streaming session: you subscribe, and records arrive on
a socket for as long as the process holds it open. `master_live.py` is a
polling loop that asks for bars every `--interval-sec` and acts on the last
CLOSED one, so what it needs is request/response, and a streaming session
inside it would mean a background thread, a rolling buffer, reconnect and gap
handling - state that can be wrong in ways a poll cannot.

The historical endpoint answers that shape directly. What it costs is LAG, and
the lag was measured rather than assumed - sampling the dataset's published
availability once a minute on 2026-08-25:

    17:08:21  available_end=17:00:00  lag= 501s
    17:09:21  available_end=17:00:00  lag= 562s
    ...                                     (unchanged for six minutes)
    17:13:30  available_end=17:00:00  lag= 810s
    17:14:30  available_end=17:10:00  lag= 271s   <- a chunk lands

**GLBX.MDP3 publishes `ohlcv-1m` in ten-minute chunks**, so the newest bar is
between about 4 and 14 minutes old depending on where in that cycle you ask.
That number decides which timeframes this transport can honestly serve:

    1h    fine. A bar is acted on within ~15 minutes of closing, and the
          engine's model - fill at the next bar's open - is off by that much
          rather than by a whole bar.
    15m   MARGINAL. A 14-minute lag on a 15-minute bar means acting almost a
          full bar late; the fill lands deep inside the bar the backtest
          assumed it entered at the open of.
    5m and faster
          NO. The lag exceeds the bar.

`databento.Live` - a streaming session, which this module deliberately does
not implement - is the transport for anything below 1h. Until it exists,
`--tf 1h` is the timeframe this feed supports and `master_live.py` prints the
age of the newest bar every cycle so the lag is visible rather than inferred.

SAME SYMBOLOGY AS THE LAKE, OR THE SERIES BREAKS AT THE SEAM
============================================================
The lake was built from CONTINUOUS contracts on a volume roll (`NQ.v.0`,
`stype_in="continuous"`, dataset `GLBX.MDP3`) by
`data_pull/pull_futures.py`, and those constants are IMPORTED from it rather
than retyped. A live feed pulling the front month by its physical symbol, or
rolling on open interest instead of volume, would hand the strategy a price
series that jumps at different dates than the one it was certified on - and
every bar would still look perfectly well formed.

COST, AND WHY EVERY CALL IS BOUNDED
===================================
Databento bills historical requests by bytes. Every call here is bounded to a
window computed from the bars actually needed, and `minute_bars` refuses a
request for more than `MAX_MINUTES` rather than silently pulling a month of
1-minute data on a mistyped lookback. The loop asks for the same few hundred
bars every cycle; a run left going overnight makes about 1,440 small requests
per symbol per day, which is the number to look at before raising the poll
rate.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load ~/src/trading/.env before anything here reads os.environ, so
# $DATABENTO_API_KEY works whether it was exported or written in the file. The
# rules live in ONE module: see mdlib/env.py. An exported value still wins.
from mdlib.env import load_env                                     # noqa: E402

load_env()

import pandas as pd                                                # noqa: E402

# The lake's own vendor constants. Imported, never retyped - see the module
# docstring: a second opinion about the dataset or the roll rule is a second
# price series.
from data_pull.pull_futures import (DATASET,                       # noqa: E402
                                    continuous_symbol)

API_KEY_VAR = "DATABENTO_API_KEY"
SCHEMA = "ohlcv-1m"

# The WARM-UP ceiling, in minutes of history, and also how much 1m history a
# symbol retains in memory. 64,800 is 45 days of clock - enough for the deepest
# lookback the loop uses (500 hourly bars is roughly 500 trading hours, about
# three trading weeks) with room for weekends and holidays inside it. It bounds
# the ONE big request per symbol per process; everything after that is an
# incremental top-up of a few rows. A larger ask is a typo, and it is billed.
MAX_MINUTES = 64_800

# How far back to reach for a given number of minutes of BARS. Futures trade
# ~23 hours a day with a maintenance break and closed weekends, so a request
# for N minutes of bars has to span more than N minutes of clock. 2.2x plus a
# weekend covers both without pulling a month.
_CALENDAR_SLACK = 2.2
_WEEKEND_HOURS = 72


class DatabentoUnavailable(RuntimeError):
    """No key, no client, or the vendor refused."""


def availability() -> tuple[bool, str]:
    """
    `(usable, why not)` — checked WITHOUT spending anything.

    Import and credential only. It deliberately does not call the API: a
    readiness probe that billed for a metadata request would be a readiness
    probe nobody runs.
    """
    try:
        import databento                                           # noqa: F401,PLC0415
    except Exception as exc:                                       # noqa: BLE001
        return False, f"the `databento` package is not importable ({exc})"
    if not (os.environ.get(API_KEY_VAR) or "").strip():
        return False, (f"${API_KEY_VAR} is not set — export it, or add it to "
                       f"the repository's .env")
    return True, "databento client and API key present"


class DatabentoBarClient:
    """
    The newest 1-minute bars for one continuous contract.

    Constructed lazily: the vendor client is not built until the first
    request, so importing this module costs nothing and a process that never
    asks for bars never authenticates.
    """

    def __init__(self, api_key: str | None = None,
                 dataset: str = DATASET) -> None:
        self.dataset = dataset
        self._api_key = api_key
        self._client = None
        # Per-symbol 1m history, warmed once and topped up. See `minute_bars`:
        # without it the loop re-buys the same fortnight of bars every minute.
        self._cache: dict[str, pd.DataFrame] = {}
        # Counters, so a session can be asked what it actually spent.
        self.requests = 0
        self.rows_fetched = 0

    # -- plumbing ---------------------------------------------------------
    def _connect(self):
        if self._client is not None:
            return self._client
        key = self._api_key or (os.environ.get(API_KEY_VAR) or "").strip()
        if not key:
            raise DatabentoUnavailable(
                f"${API_KEY_VAR} is not set, so there is no live feed. Export "
                f"it, or run with --feed lake and know you are reading "
                f"historical bars.")
        try:
            import databento as db                                 # noqa: PLC0415
        except Exception as exc:                                   # noqa: BLE001
            raise DatabentoUnavailable(
                f"the `databento` package is not importable: {exc}") from exc
        self._client = db.Historical(key)
        return self._client

    def available_end(self, max_age_s: float = 20.0) -> pd.Timestamp:
        """
        The newest instant this dataset will answer for, from the vendor.

        **The request must not ask past it.** Databento refuses an `end` after
        its published availability with a 422 rather than clamping, so a query
        stamped `now` fails outright — which is what a naive poll does every
        single cycle.

        `metadata.get_dataset_range` is not billed, and the answer is cached
        for `max_age_s` so a four-symbol poll makes one call rather than four.
        The cache is short because this value is the whole point: it advances
        as the vendor publishes, and a stale copy would silently pin the feed
        to the moment the loop started.
        """
        now = datetime.now(timezone.utc)
        cached = getattr(self, "_range_cache", None)
        if cached is not None and (now - cached[0]).total_seconds() < max_age_s:
            return cached[1]

        client = self._connect()
        try:
            span = client.metadata.get_dataset_range(dataset=self.dataset)
        except Exception as exc:                                   # noqa: BLE001
            raise DatabentoUnavailable(
                f"could not read {self.dataset}'s available range "
                f"({type(exc).__name__}: {exc})") from exc
        raw = span.get("schema", {}).get(SCHEMA, span).get("end", span.get("end"))
        end = pd.Timestamp(raw)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        self._range_cache = (now, end)
        return end

    def _window(self, minutes: int, end: pd.Timestamp | None = None
                ) -> tuple[datetime, datetime]:
        """
        The clock span to request for `minutes` of BARS.

        Wider than `minutes` because the market is shut for part of it: futures
        break daily and stop for the weekend, so N minutes of bars span more
        than N minutes of clock. The END is the vendor's published
        availability, never `now` — see `available_end`. Whatever forming bar
        comes back inside that window is dropped by
        `realtime.feed.drop_forming_bar`, which is the only place that rule
        lives.
        """
        stop = pd.Timestamp(end) if end is not None else self.available_end()
        span = timedelta(minutes=int(minutes) * _CALENDAR_SLACK
                         ) + timedelta(hours=_WEEKEND_HOURS)
        return (stop - span).to_pydatetime(), stop.to_pydatetime()

    def horizon(self) -> pd.Timestamp:
        """
        Where this feed's data ends — the vendor's published availability.

        `realtime.feed.BarFeed.horizon` uses it to cut bars at the earlier of
        the clock and the data, so an interval the vendor has published only
        half of is held back rather than handed over as a complete bar with
        the wrong high, low, close and volume.
        """
        return self.available_end()

    # -- the contract -----------------------------------------------------
    def _request(self, symbol: str, start, end) -> pd.DataFrame:
        """One bounded call to the vendor. The only place that spends money."""
        client = self._connect()
        try:
            store = client.timeseries.get_range(
                dataset=self.dataset,
                schema=SCHEMA,
                symbols=[continuous_symbol(str(symbol).upper())],
                stype_in="continuous",
                start=start,
                end=end,
            )
            frame = store.to_df()
        except Exception as exc:                                   # noqa: BLE001
            raise DatabentoUnavailable(
                f"{symbol}: Databento request failed "
                f"({type(exc).__name__}: {exc})") from exc
        self.requests += 1
        if frame is None or len(frame) == 0:
            return pd.DataFrame()
        out = frame.reset_index()
        ts_col = "ts_event" if "ts_event" in out.columns else out.columns[0]
        out = out.rename(columns={ts_col: "ts"})
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        self.rows_fetched += len(out)
        return out

    def minute_bars(self, symbol: str, minutes: int) -> pd.DataFrame:
        """
        The newest `minutes` 1-minute bars for `symbol`, as a DataFrame.

        **CACHED AND TOPPED UP, because the alternative is unaffordable.** The
        loop polls every 60 seconds and a strategy on 400 hourly bars needs
        ~24,000 minute bars behind it; re-requesting that window every cycle
        would pull ~100,000 rows a minute across four symbols, on an endpoint
        billed by bytes. So the first call for a symbol fetches the window
        once, and every call after it asks only for the minutes since the last
        bar already held — a handful of rows.

        `symbol` is a ROOT (`NQ`), resolved to the continuous contract the lake
        was built from (`NQ.v.0`) by `pull_futures.continuous_symbol`. The
        frame keeps the vendor's own column spelling and is normalized by
        `realtime.feed.normalize_bars`, so there is exactly one normalizer
        between the live feed and the lake.

        An empty response is an empty frame, not an error: a request landing
        in the maintenance break or over a weekend legitimately has no bars,
        and raising there would take the loop down every Saturday.
        """
        want = int(minutes)
        if want <= 0:
            raise ValueError(f"minutes must be positive; got {minutes}")
        if want > MAX_MINUTES:
            raise DatabentoUnavailable(
                f"refusing to warm up {want:,} minutes of 1m bars (cap "
                f"{MAX_MINUTES:,}). Historical requests are billed by bytes, "
                f"and a lookback this large is a typo more often than a "
                f"requirement.")

        key = str(symbol).upper()
        end = self.available_end()
        held = self._cache.get(key)

        if held is None or held.empty:
            start, stop = self._window(want, end=end)
            fresh = self._request(key, start, stop)
        elif pd.Timestamp(held["ts"].iloc[-1]) >= end - pd.Timedelta("1min"):
            # Nothing new has been published since the last top-up. Poll
            # faster than the vendor publishes and this is the common case;
            # spending a request to be told so again would be the waste.
            return held.tail(want)
        else:
            resume = pd.Timestamp(held["ts"].iloc[-1]) + pd.Timedelta("1min")
            fresh = self._request(key, resume.to_pydatetime(),
                                  end.to_pydatetime())

        if fresh.empty:
            return held.tail(want) if held is not None else pd.DataFrame()

        merged = (fresh if held is None or held.empty
                  else pd.concat([held, fresh], ignore_index=True))
        merged = (merged.sort_values("ts")
                        .drop_duplicates(subset=["ts"], keep="last")
                        .tail(MAX_MINUTES)
                        .reset_index(drop=True))
        self._cache[key] = merged
        return merged.tail(want)


def main(argv: list[str] | None = None) -> int:
    """`python3 data_pull/databento_live.py --symbol NQ` — one bounded probe."""
    import argparse

    ap = argparse.ArgumentParser(description="Probe the live bar feed.")
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--minutes", type=int, default=120)
    args = ap.parse_args(argv)

    ok, why = availability()
    print(f"availability: {'OK' if ok else 'UNAVAILABLE'} — {why}")
    if not ok:
        return 1

    from realtime.feed import normalize_bars                       # noqa: PLC0415
    client = DatabentoBarClient()
    raw = client.minute_bars(args.symbol, args.minutes)
    bars = normalize_bars(raw, args.symbol)
    if bars.empty:
        print("no bars returned for that window (market closed?)")
        return 0
    print(f"{len(bars)} 1m bars  {bars['ts'].iloc[0]} -> {bars['ts'].iloc[-1]}")
    print(bars.tail(3).to_string(index=False))
    age = (datetime.now(timezone.utc)
           - bars["ts"].iloc[-1].to_pydatetime()).total_seconds()
    print(f"newest bar opened {age:,.0f}s ago")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
