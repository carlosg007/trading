#!/usr/bin/env python3
"""
realtime.feed - where the bars come from, and the one rule about which ones.

`master_live.py` and the regime daemon both need the same thing: the newest
CLOSED bars for a set of contracts, as `{symbol: DataFrame}`. Until now that
was a lake read inlined in `master_live.load_symbol_bars`, which is why the
loop could run forever against a tape that stopped 18 days ago and report
nothing wrong. This module is the seam: one interface, two implementations,
and the anti-lookahead rule written down once.

THE FORMING BAR IS THE WHOLE POINT
==================================
A bar is stamped with the time it OPENED - that is the lake's convention
(`label="left", closed="left"`) and Databento's - so the bar stamped 14:00 on a
1h feed covers 14:00 to 15:00 and is not finished until 15:00. Ask a live feed
for bars at 14:30 and the last row it hands back is half a bar: its close is
the current price, its high and low are whatever has printed so far.

**Acting on that row is lookahead.** The engine fills at the next bar's open,
so a signal computed on a bar that has not closed is a decision made with
information from inside the interval it is trading. Backtested that way it
looks brilliant; live it is a different strategy. `drop_forming_bar` removes
it, every feed passes through it, and it is the reason this module exists as a
seam rather than as two callers each doing their own thing.

It is a DROP, never a fill-forward and never a truncation to "close so far". A
partial bar is not a bar; the honest answer is that the last complete one is
the newest fact available.

LIVE BARS ARE BUILT THE SAME WAY LAKE BARS ARE
==============================================
Both feeds return 1-minute data aggregated by `mdlib.lake`'s own resampler,
through `mdlib.lake.DERIVED` and `mdlib.lake.OHLCV` - imported, never restated.
A 15m bar assembled here is therefore assembled exactly as the 15m bar the
strategy was certified on. Reimplementing the aggregation would give the live
loop bars that differ from the backtest's in the last decimal, which is enough
to put a bar on the other side of an indicator threshold with nothing in any
log to say why.

THE MICRO ALIAS LIVES HERE TOO
==============================
The baskets hold MNQ and the tape is NQ's - same price series, same tick size,
different multiplier - so every feed resolves a micro to its full-size parent
through `realtime/contract_alias.py` and REPORTS the substitution in the
`sources` map. The order is still for the micro and still sized on the micro's
point value; only the bars come from the parent.

WHERE LIVE BARS COME FROM, AND WHERE THEY DO NOT
================================================
The live feed is the BROKER's: NinjaTrader 8, through `realtime/nt8_feed.py`.
The bars a strategy decides on are then the bars its orders execute against -
same feed, same session template, same clock - and a research vendor that
disagreed with a broker fill about a bar's close would produce slippage nobody
could source.

**Databento is historical only**, for backtests and model training, through
`data_pull/pull_futures.py` and the lake. Nothing in this module and nothing in
the live path imports it.

`resolve_feed("auto")` returns the NT8 feed when it is publishing and the lake
otherwise, and says which in `describe()`. The choice is deliberately NOT tied
to `--dry-run`: dry-run is about whether a socket opens, and a dry run against
stale bars cannot rehearse a decision the loop would make now. Pass
`--feed lake` to force the historical path when you mean it.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd                                                # noqa: E402

from mdlib.lake import DERIVED, LONG_COLUMNS, NATIVE_TFS, OHLCV    # noqa: E402
from realtime.contract_alias import resolve_parent                 # noqa: E402

BAR_COLUMNS = tuple(LONG_COLUMNS)     # ts, symbol, open, high, low, close, volume

# `live` is kept as a spelling of `nt8` so existing commands and documented
# runbooks keep working; there is only one live feed and it is the broker's.
FEED_MODES = ("auto", "nt8", "live", "lake")


class FeedError(RuntimeError):
    """The feed cannot answer, and will not guess."""


# --------------------------------------------------------------------------
# timeframes
# --------------------------------------------------------------------------

def tf_delta(tf: str) -> pd.Timedelta:
    """
    One bar's WIDTH, from the same table the lake builds its bars with.

    Derived from `mdlib.lake.DERIVED` rather than parsed out of the string, so
    a timeframe this repository cannot build is a timeframe this module
    refuses rather than one it invents a width for.
    """
    key = str(tf).strip()
    if key in DERIVED:
        return pd.Timedelta(DERIVED[key][1])
    if key in NATIVE_TFS:
        return pd.Timedelta("1min" if key == "1m" else "1D")
    raise FeedError(
        f"unknown timeframe {tf!r}. Known: {sorted(set(DERIVED) | NATIVE_TFS)}")


def infer_timeframe(bars, default: str | None = None) -> str | None:
    """
    A frame's bar WIDTH, read back as a timeframe token.

    The inverse of `tf_delta`, and it exists so a mismatch between a strategy's
    certified timeframe and the bars it is handed can be CAUGHT rather than
    trusted. Until 2026-08-27 `master_live.py` loaded one timeframe and gave it
    to every strategy, so a 3m certification was evaluated on 1h bars with
    nothing in the stack noticing — the signals were real, the log lines were
    correct, and the certification described a different tape.

    THE MODE, NOT THE MEAN OR THE MEDIAN. A session break is a six-hour gap
    between two adjacent rows and a holiday is longer; both drag an average and
    can drag a median on a short frame. The most COMMON spacing is the bar
    width by construction, because every bar inside a session sits one width
    from its neighbour.

    Returns `default` when the frame is too short to have a spacing (fewer than
    two rows) or when the modal spacing matches no timeframe this repository
    can build. Guessing there would defeat the point: an unrecognised width has
    to reach the caller as "unknown", not as the nearest token.
    """
    import pandas as pd                                            # noqa: PLC0415

    if bars is None or len(bars) < 2:
        return default
    try:
        ts = bars["ts"] if "ts" in getattr(bars, "columns", ()) else bars.index
        stamps = pd.to_datetime(pd.Series(list(ts)), utc=True).sort_values()
        deltas = stamps.diff().dropna()
        if deltas.empty:
            return default
        modal = deltas.mode()
        if modal.empty:
            return default
        width = pd.Timedelta(modal.iloc[0])
    except (KeyError, TypeError, ValueError):
        return default
    if width <= pd.Timedelta(0):
        return default

    for token in sorted(set(DERIVED) | NATIVE_TFS):
        try:
            if tf_delta(token) == width:
                return token
        except Exception:                                     # noqa: BLE001
            # Broad on purpose. `tf_delta` raises FeedError for an unknown
            # token, but `1w` is a KNOWN one whose rule is the anchored offset
            # `W-MON` rather than a Timedelta string, so pandas raises a bare
            # ValueError inside it. A token this loop cannot measure is simply
            # not the answer; letting that decide the whole lookup would make
            # one unmeasurable entry hide every token after it alphabetically.
            continue
    return default


def source_tf(tf: str) -> str:
    """The NATIVE timeframe `tf` is built from — 1m for everything intraday."""
    key = str(tf).strip()
    if key in NATIVE_TFS:
        return key
    if key in DERIVED:
        return DERIVED[key][0]
    raise FeedError(f"unknown timeframe {tf!r}")


# --------------------------------------------------------------------------
# the frame contract
# --------------------------------------------------------------------------

def normalize_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Any bar frame -> the canonical one: `ts, symbol, open, high, low, close,
    volume`, UTC, sorted, one row per timestamp.

    The same shape `mdlib.lake` returns and `data_pull/pull_futures.normalize`
    writes, because downstream code cannot tell where a frame came from and
    must not have to. A missing column RAISES: a strategy handed a frame with
    no `volume` does not fail, it silently computes a different signal.
    """
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=list(BAR_COLUMNS))

    df = frame.reset_index()
    if "ts" not in df.columns:
        for candidate in ("ts_event", "timestamp", "time", "index"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "ts"})
                break
    lowered = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=lowered)

    missing = [c for c in ("ts", "open", "high", "low", "close", "volume")
               if c not in df.columns]
    if missing:
        raise FeedError(
            f"{symbol}: bar frame is missing {missing}. Got {list(df.columns)}. "
            f"A frame short a column does not fail downstream — it produces a "
            f"different signal, quietly.")

    out = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
    out["symbol"] = str(symbol).upper()
    out = (out[list(BAR_COLUMNS)]
           .dropna(subset=["open", "high", "low", "close"])
           .sort_values("ts")
           .drop_duplicates(subset=["ts"], keep="last")
           .reset_index(drop=True))
    return out


def canonical_order(frame: pd.DataFrame) -> pd.DataFrame:
    """
    The canonical columns FIRST, anything else after.

    The lake's resampler appends `symbol` after the OHLCV block, so an
    aggregated frame comes back in a different column order from a native one
    — same data, different shape, and code that reads a frame positionally
    would read the wrong column. Extra columns (the lake's joined regime
    block) are kept: they are useful, and dropping them would make the lake
    path lose information for the sake of symmetry.
    """
    lead = [c for c in BAR_COLUMNS if c in frame.columns]
    rest = [c for c in frame.columns if c not in lead]
    return frame[lead + rest]


def to_timeframe(minute_bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    1-minute bars -> `tf` bars, through the LAKE's resampler.

    `mdlib.lake._resample` is the implementation and `DERIVED` is the rule
    table; both are imported. A second aggregation written here would differ
    from the one the certification was computed on somewhere in the last
    decimal, and the difference would surface as a bar on the wrong side of an
    indicator threshold with nothing in any log explaining it.
    """
    key = str(tf).strip()
    if key in NATIVE_TFS or minute_bars.empty:
        return minute_bars.reset_index(drop=True)
    if key not in DERIVED:
        raise FeedError(f"unknown timeframe {tf!r}")
    from mdlib.lake import _resample                              # noqa: PLC0415
    return canonical_order(
        _resample(minute_bars, DERIVED[key][1]).reset_index(drop=True))


def drop_forming_bar(bars: pd.DataFrame, tf: str,
                     now: datetime | pd.Timestamp | None = None
                     ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """
    `(closed_bars, dropped_ts)` — the anti-lookahead rule, in one place.

    A bar is stamped with the time it OPENED, so the bar at `ts` covers
    `[ts, ts + tf)` and is finished only once `now >= ts + tf`. Every row that
    fails that test is dropped, not just the last one: a feed can hand back
    several unfinished intervals after a reconnect, and dropping only the tail
    would leave the second-to-last half-formed bar in place looking complete.

    `dropped_ts` is the newest timestamp removed, or None when nothing was —
    it is REPORTED rather than silently discarded, so an operator can see the
    loop declining to act on a bar that is still filling in.
    """
    if bars is None or len(bars) == 0:
        return (bars if bars is not None
                else pd.DataFrame(columns=list(BAR_COLUMNS))), None

    width = tf_delta(tf)
    stamp = pd.Timestamp(now if now is not None
                         else datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")

    closes_at = bars["ts"] + width
    forming = closes_at > stamp
    if not forming.any():
        return bars.reset_index(drop=True), None
    dropped = pd.Timestamp(bars.loc[forming, "ts"].max())
    return bars.loc[~forming].reset_index(drop=True), dropped


# --------------------------------------------------------------------------
# feeds
# --------------------------------------------------------------------------

class BarFeed(ABC):
    """
    One method, one contract: the newest CLOSED bars per symbol.

    Returns `({symbol: DataFrame}, {symbol: source_symbol})` — the second map
    records where each frame actually came from, because a micro's bars are
    its parent's and the operator has to be able to see that rather than infer
    it from a table.
    """

    name = "feed"

    @abstractmethod
    def fetch_minutes(self, source_symbol: str, minutes: int) -> pd.DataFrame:
        """The newest `minutes` 1-minute bars for ONE full-size contract."""

    def horizon(self) -> pd.Timestamp | None:
        """
        The newest instant this feed has DATA for, or None when it cannot say.

        This is not the wall clock, and the difference is a real bug. A vendor
        that publishes on a lag can leave an interval whose end has passed but
        whose last minutes have not arrived: at 17:05, with data published to
        16:50, the 16:45 quarter-hour is finished by the clock and built from
        five minutes of bars. Treated as complete it is a bar with the wrong
        high, the wrong low, the wrong close and the wrong volume — and it
        looks exactly like a quiet quarter of an hour.

        `closed_bars` therefore cuts at the EARLIER of the clock and this, so a
        lagging feed produces fewer bars rather than wrong ones.
        """
        return None

    def describe(self) -> str:
        return self.name

    def closed_bars(self, symbols, tf: str, lookback_bars: int = 500
                    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """
        `({symbol: closed bars}, {symbol: which contract supplied them})`.

        Every frame is normalized, aggregated to `tf` through the lake's
        resampler, and passed through `drop_forming_bar`. A symbol the feed
        cannot answer for is ABSENT from the result rather than present and
        empty: "no bars for this symbol" and "this symbol has no signal" are
        different statements and the loop reports them differently.
        """
        wanted = {str(s).upper(): resolve_parent(s) for s in symbols}
        width = tf_delta(tf)
        minutes = max(int(lookback_bars) + 2, 1) * max(
            int(width / pd.Timedelta("1min")), 1)

        # The earlier of the clock and what the feed actually has. See
        # `horizon`: a published-on-a-lag interval that is over by the clock is
        # still incomplete in the data.
        wall = pd.Timestamp(datetime.now(timezone.utc))
        edge = self.horizon()
        cutoff = wall if edge is None else min(wall, pd.Timestamp(edge))

        cache: dict[str, pd.DataFrame] = {}
        bars: dict[str, pd.DataFrame] = {}
        sources: dict[str, str] = {}
        self.last_forming: dict[str, pd.Timestamp] = {}

        for symbol, parent in wanted.items():
            if parent not in cache:
                frame = self.fetch_minutes(parent, minutes)
                cache[parent] = normalize_bars(frame, parent)
            minute_bars = cache[parent]
            if minute_bars.empty:
                continue
            shaped = to_timeframe(minute_bars, tf)
            closed, dropped = drop_forming_bar(shaped, tf, now=cutoff)
            if dropped is not None:
                self.last_forming[symbol] = dropped
            if closed.empty:
                continue
            out = closed.tail(int(lookback_bars)).reset_index(drop=True)
            out["symbol"] = symbol          # the contract being TRADED
            bars[symbol] = canonical_order(out)
            sources[symbol] = parent
        return bars, sources


class LakeFeed(BarFeed):
    """
    The historical lake. What the loop has always read.

    Correct and never current: the lake is a batch store, so its newest bar is
    the newest bar somebody ingested. It is the right feed for a rehearsal
    against known data and the wrong one for deciding what to do now.
    """

    name = "lake"

    def __init__(self, session_merge: bool = True) -> None:
        self.session_merge = session_merge

    def describe(self) -> str:
        return "lake (historical parquet — as current as the last ingest)"

    def fetch_minutes(self, source_symbol: str, minutes: int) -> pd.DataFrame:
        from mdlib.lake import iter_bars                          # noqa: PLC0415
        for _, frame in iter_bars([source_symbol], "1m", None, None):
            if frame is not None and not frame.empty:
                return frame.tail(int(minutes))
        return pd.DataFrame(columns=list(BAR_COLUMNS))

    def closed_bars(self, symbols, tf: str, lookback_bars: int = 500):
        """
        Read the lake at `tf` DIRECTLY rather than pulling minutes and
        resampling.

        Same bars either way — `iter_bars` does that aggregation internally —
        but reading 500 hourly bars instead of 30,000 minute bars is the
        difference between a 350ms cycle and a several-second one, and the
        loop runs this every interval.
        """
        from mdlib.lake import iter_bars                          # noqa: PLC0415
        wanted = {str(s).upper(): resolve_parent(s) for s in symbols}
        frames = {}
        for source, frame in iter_bars(sorted(set(wanted.values())), tf,
                                       None, None):
            if frame is not None and not frame.empty:
                frames[source] = frame

        bars, sources = {}, {}
        self.last_forming = {}
        for symbol, parent in wanted.items():
            frame = frames.get(parent)
            if frame is None:
                continue
            # The lake holds only completed bars, but the rule is applied here
            # too rather than assumed: an ingest that ran mid-interval would
            # otherwise put a partial bar into a live decision, and nothing
            # downstream could tell.
            closed, dropped = drop_forming_bar(
                frame.sort_values("ts").reset_index(drop=True), tf)
            if dropped is not None:
                self.last_forming[symbol] = dropped
            if closed.empty:
                continue
            out = closed.tail(int(lookback_bars)).reset_index(drop=True)
            out["symbol"] = symbol
            bars[symbol] = canonical_order(out)
            sources[symbol] = parent
        return bars, sources


class LiveFeed(BarFeed):
    """
    A real feed, through whatever vendor client is handed in.

    The client is injected rather than constructed here for two reasons: this
    package must not reach a vendor API (`data_pull/` is the layer that does
    that), and a feed whose transport is a constructor argument can be tested
    against a fake that returns known bars.

    The client contract is one method:

        client.minute_bars(symbol: str, minutes: int) -> DataFrame

    with any column spelling `normalize_bars` accepts.
    """

    name = "live"

    def __init__(self, client: Any, label: str | None = None) -> None:
        if not hasattr(client, "minute_bars"):
            raise FeedError(
                f"{type(client).__name__} is not a bar client: it has no "
                f"`minute_bars(symbol, minutes)`.")
        self.client = client
        self.label = label or type(client).__name__

    def describe(self) -> str:
        return f"live ({self.label})"

    def horizon(self) -> pd.Timestamp | None:
        """The vendor's published edge, when the client exposes one."""
        getter = getattr(self.client, "horizon", None)
        if getter is None:
            return None
        try:
            return pd.Timestamp(getter())
        except Exception:                                          # noqa: BLE001
            # A feed that cannot say where its data ends falls back to the
            # clock rather than refusing to produce bars: the drop rule still
            # applies, it is just no longer tightened by the vendor's edge.
            return None

    def fetch_minutes(self, source_symbol: str, minutes: int) -> pd.DataFrame:
        return self.client.minute_bars(source_symbol, minutes)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def live_feed_available() -> tuple[bool, str]:
    """
    `(is_available, why not)` for the live feed — which is NT8's, and only
    NT8's.

    Live market data comes from the BROKER. Databento is historical data for
    backtests and model training, and nothing in the live path imports it: the
    bars a strategy decides on should be the bars its orders execute against,
    and a research vendor and a broker fill that disagree about a bar's close
    produce slippage nobody can source.
    """
    try:
        from realtime.nt8_feed import availability                 # noqa: PLC0415
    except Exception as exc:                                       # noqa: BLE001
        return False, f"realtime.nt8_feed is unimportable ({exc})"
    return availability()


def resolve_feed(mode: str = "auto", **kwargs) -> BarFeed:
    """
    The feed for this run.

    `live` RAISES when no live feed is configured rather than falling back —
    an operator who asked for live bars and silently got 18-day-old ones would
    be reading a rehearsal as a live session. `auto` falls back, and says so.
    """
    choice = str(mode or "auto").strip().lower()
    if choice not in FEED_MODES:
        raise FeedError(f"--feed must be one of {FEED_MODES}; got {mode!r}")

    if choice == "lake":
        return LakeFeed(**kwargs)

    from realtime.nt8_feed import NT8BarFeed                       # noqa: PLC0415

    ok, why = live_feed_available()
    if choice in ("nt8", "live") and not ok:
        raise FeedError(
            f"--feed {choice} was requested and the NT8 feed is not "
            f"publishing: {why}. Refusing to fall back to the lake: a "
            f"rehearsal against stale bars reads exactly like a live session "
            f"that found no signal.")
    if not ok:
        return LakeFeed(**kwargs)
    return NT8BarFeed(**kwargs)
