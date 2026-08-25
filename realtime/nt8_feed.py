#!/usr/bin/env python3
"""
realtime.nt8_feed - live bars from NinjaTrader 8, the account's own tape.

The live path takes its market data from the BROKER, not from a research
vendor. The bars a strategy decides on are then the bars the orders execute
against: same feed, same session template, same instrument, same clock. A
vendor feed and a broker fill can disagree about the last price of a bar, and
when they do the difference shows up as slippage nobody can source.

Databento stays where it belongs - historical bars for backtests and model
training, through `data_pull/pull_futures.py` and the lake. Nothing in this
module and nothing in the live path imports it.

    NT8 (Windows)  --writes-->  spool directory  --reads-->  NT8BarFeed
                                                             master_live.py
                                                             regime_daemon.py

THE TRANSPORT IS A SPOOL DIRECTORY, NOT A SOCKET
================================================
NinjaTrader runs on Windows and this loop runs on Linux, and the channel that
already exists between them is a shared directory - it is how NT8 delivers fill
logs to `live/dispatcher.py::evaluate_incubator_sync`. A bar spool works the
same way: a NinjaScript add-on appends one line per CLOSED bar, this reads the
tail.

That choice is deliberate over an HTTP listener. A listener means a port open
on this box, a process to supervise, and a silent hole when it dies; a file
that stops growing is visible to `ls`, survives a restart on either side, and
replays after an outage because the history is still in it. If a push
transport is added later it should WRITE THIS SPOOL rather than bypass it, so
there is one format and one place to look when a bar is missing.

THE FILE CONTRACT, AND THE ONE FIELD THAT SILENTLY RUINS EVERYTHING
===================================================================
The spool lives at `/mnt/backtest/artifacts/nt8_bars/`, or wherever
`$BT_NT8_SPOOL` points. One file per (symbol, timeframe), named
`{SYMBOL}_{TF}.csv`, appended:

    ts,open,high,low,close,volume
    2026-08-25T17:00:00Z,29260.00,29280.25,29250.75,29274.75,12345

**`ts` IS THE BAR'S CLOSE TIME, because that is what NinjaTrader stamps**, and
this repository stamps bars by their OPEN - the lake resamples `label="left"`,
every certification was computed that way, and Databento publishes that way.
Ingested as-is, an NT8 hourly bar stamped 17:00 would be read as the bar
STARTING at 17:00: every bar shifted one period, every indicator computed on
misaligned data, and not one error anywhere. The default is `close` because
that is NT8's, and a file DECLARES it otherwise with a header line:

    # stamp=open

The declaration lives in the file rather than in a flag on this machine
because the NinjaScript is the thing that knows which end it writes, and an
operator who changes that script should not have to remember a command-line
argument on the other side of the mount.

Timestamps MUST carry an explicit UTC offset (`Z` or `+00:00`). A naive
timestamp is REFUSED rather than assumed to be UTC: NinjaTrader writes in the
instrument's or the workstation's timezone unless the script converts, so a
naive stamp is exactly as likely to be New York as UTC, and guessing wrong
shifts the whole series by hours in a way that still looks like a market.

WHAT THIS MODULE WILL NOT DO
============================
* **Invent a bar.** A gap in the spool is a gap. Nothing is forward-filled: a
  synthetic bar has a real timestamp and a fictional close, and every
  indicator downstream treats it as a fact.
* **Trust that a written bar is a closed bar.** The NinjaScript is supposed to
  append on bar close, but every frame still goes through
  `realtime.feed.drop_forming_bar` - one rule, one place, applied to every
  feed. A script switched to `Calculate.OnEachTick` would otherwise start
  appending forming bars and nothing would notice.
* **Guess which instrument a file is about.** The filename names the symbol
  and the timeframe; a file whose name does not parse is reported, not read.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd                                                # noqa: E402

from realtime.contract_alias import resolve_parent                 # noqa: E402
from realtime.feed import (BAR_COLUMNS, BarFeed, FeedError,        # noqa: E402
                           canonical_order, normalize_bars,
                           tf_delta, to_timeframe)

# Where NT8 drops bars. Beside the fill logs it already writes, because one
# share between the two machines is one thing to mount, one thing to permission
# and one place to look when something is missing.
DEFAULT_SPOOL_DIR = Path("/mnt/backtest/artifacts/nt8_bars")

# Where the share is mounted, when it is not the default. Read at CALL time,
# not at import: `mdlib.env.load_env` runs at the top of every entrypoint, and
# a module-level read here would capture the value from before the file was
# loaded and then be wrong in a way that looks like a missing publisher.
SPOOL_DIR_VAR = "BT_NT8_SPOOL"


def configured_spool_dir(spool_dir: Path | str | None = None) -> Path:
    """The spool to read: the argument, then `$BT_NT8_SPOOL`, then the default."""
    if spool_dir is not None:
        return Path(spool_dir)
    override = (os.environ.get(SPOOL_DIR_VAR) or "").strip()
    return Path(override) if override else DEFAULT_SPOOL_DIR

# `ts` means the bar's... close (NinjaTrader's default) or open (this
# repository's). Spelled out rather than inferred: a heuristic that guessed
# from the first timestamp would be right until the first session that opened
# on the hour.
STAMP_CONVENTIONS = ("close", "open")
DEFAULT_STAMP = "close"

SPOOL_SUFFIXES = (".csv", ".txt", ".tsv", ".jsonl")


class NT8FeedError(FeedError):
    """The NT8 spool cannot answer, and will not be guessed at."""


def spool_path(symbol: str, tf: str, spool_dir: Path | str | None = None,
               suffixes: tuple[str, ...] = SPOOL_SUFFIXES) -> Path | None:
    """
    The file NT8 writes `(symbol, tf)` into, or None when there is none.

    Tried under the symbol NT8 actually trades AND its full-size parent, so a
    spool of MNQ bars answers a request for MNQ and a spool of NQ bars answers
    it too - the same alias every other tier resolves through.
    """
    directory = configured_spool_dir(spool_dir)
    if not directory.is_dir():
        return None
    for name in (str(symbol).upper(), resolve_parent(symbol)):
        for suffix in suffixes:
            candidate = directory / f"{name}_{tf}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def read_spool(path: Path, tf: str, stamp: str = DEFAULT_STAMP
               ) -> tuple[pd.DataFrame, str]:
    """
    One spool file -> `(canonical bars, the stamp convention used)`.

    The file may override the convention with a `# stamp=open` header line,
    because the NinjaScript that writes it is the thing that knows, and an
    operator changing the script should not have to remember a CLI flag on
    another machine.

    Close-stamped bars are shifted back one timeframe so every frame in this
    repository means the same thing by `ts`. That subtraction is the whole
    reason this function exists rather than a call to `pd.read_csv`.
    """
    convention = str(stamp).strip().lower()
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        payload = []
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                token = text.lstrip("#").strip().lower()
                if token.startswith("stamp="):
                    declared = token.split("=", 1)[1].strip()
                    if declared not in STAMP_CONVENTIONS:
                        raise NT8FeedError(
                            f"{path.name}: header declares stamp={declared!r}; "
                            f"it must be one of {STAMP_CONVENTIONS}.")
                    convention = declared
                continue
            payload.append(line)
        if not payload:
            return pd.DataFrame(columns=list(BAR_COLUMNS)), convention
        rows = list(csv.DictReader(payload,
                                   dialect=_sniff("".join(payload[:20]))))

    if convention not in STAMP_CONVENTIONS:
        raise NT8FeedError(
            f"stamp must be one of {STAMP_CONVENTIONS}; got {stamp!r}")

    symbol = path.stem.split("_")[0].upper()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(BAR_COLUMNS)), convention

    _refuse_naive_timestamps(frame, path)
    bars = normalize_bars(frame, symbol)
    if convention == "close":
        # NinjaTrader stamps the CLOSE. Everything here stamps the OPEN. One
        # subtraction, in one place, or every bar in the system is off by one
        # period with nothing raising.
        bars["ts"] = bars["ts"] - tf_delta(tf)
    return bars, convention


def _sniff(sample: str) -> csv.Dialect | type[csv.Dialect]:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _refuse_naive_timestamps(frame: pd.DataFrame, path: Path) -> None:
    """
    A timestamp with no offset is REFUSED, not assumed to be UTC.

    NinjaTrader writes in the instrument's timezone, or the workstation's,
    unless the script converts explicitly - so a naive stamp is as likely to be
    New York as UTC. Assumed wrong it shifts the entire series by hours, and
    the result still looks like a market: bars in order, prices sane, sessions
    the wrong length in a way nobody reads off a chart.
    """
    column = next((c for c in frame.columns
                   if str(c).strip().lower() in ("ts", "timestamp", "time",
                                                 "datetime", "ts_event")), None)
    if column is None:
        return
    sample = str(frame[column].dropna().iloc[0]) if len(frame) else ""
    if not sample:
        return
    parsed = pd.to_datetime(sample, errors="coerce")
    if parsed is not pd.NaT and parsed.tzinfo is None:
        raise NT8FeedError(
            f"{path.name}: timestamp {sample!r} carries no UTC offset. NT8 "
            f"writes in the instrument's or the workstation's timezone unless "
            f"the script converts, so this cannot be assumed to be UTC - "
            f"guessed wrong it shifts every bar by hours and still looks like "
            f"a market. Write ISO-8601 with `Z` or an explicit offset.")


class NT8BarFeed(BarFeed):
    """
    The broker's own bars, from the spool NT8 appends to.

    Prefers a spool published at the requested timeframe; falls back to the 1m
    spool and aggregates through the lake's resampler, so a NinjaScript
    publishing only minutes still serves a 15m or 1h strategy with bars built
    exactly as the certification's were.
    """

    name = "nt8"

    def __init__(self, spool_dir: Path | str | None = None,
                 stamp: str = DEFAULT_STAMP) -> None:
        if str(stamp).strip().lower() not in STAMP_CONVENTIONS:
            raise NT8FeedError(
                f"stamp must be one of {STAMP_CONVENTIONS}; got {stamp!r}")
        self.spool_dir = configured_spool_dir(spool_dir)
        self.stamp = str(stamp).strip().lower()
        self.last_forming: dict[str, pd.Timestamp] = {}
        self.spooled: dict[str, str] = {}

    def describe(self) -> str:
        state = "present" if self.spool_dir.is_dir() else "MISSING"
        return (f"NT8 broker feed (spool {self.spool_dir}, {state}, "
                f"ts={self.stamp}-stamped)")

    def fetch_minutes(self, source_symbol: str, minutes: int) -> pd.DataFrame:
        """The 1-minute spool, for the base-class path. See `closed_bars`."""
        path = spool_path(source_symbol, "1m", self.spool_dir)
        if path is None:
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        bars, _ = read_spool(path, "1m", self.stamp)
        return bars.tail(int(minutes))

    def closed_bars(self, symbols, tf: str, lookback_bars: int = 500):
        """
        `({symbol: closed bars}, {symbol: which spool supplied them})`.

        A MISSING spool directory raises: it means the NT8 publisher was never
        wired up, and that is not the same fact as a market with no bars. A
        directory that exists with no file for a symbol leaves that symbol
        ABSENT from the result, which the loop reports as "no bars" rather
        than as "no signal".
        """
        if not self.spool_dir.is_dir():
            raise NT8FeedError(
                f"no NT8 bar spool at {self.spool_dir}. Nothing is publishing "
                f"bars there, so there is no live feed - this is not a quiet "
                f"market. Install the NinjaScript bar publisher on the NT8 "
                f"workstation (see the file contract in this module), or run "
                f"with --feed lake and know you are reading history.")

        bars: dict[str, pd.DataFrame] = {}
        sources: dict[str, str] = {}
        self.last_forming = {}
        self.spooled = {}
        cache: dict[tuple[str, str], pd.DataFrame] = {}

        from realtime.feed import drop_forming_bar                 # noqa: PLC0415

        for raw in symbols:
            symbol = str(raw).upper()
            native = spool_path(symbol, tf, self.spool_dir)
            key = (symbol, tf if native is not None else "1m")

            if key not in cache:
                path = native or spool_path(symbol, "1m", self.spool_dir)
                if path is None:
                    continue
                frame, convention = read_spool(path, key[1], self.stamp)
                self.spooled[symbol] = f"{path.name} ({convention}-stamped)"
                cache[key] = frame if native is not None else to_timeframe(
                    frame, tf)
            shaped = cache[key]
            if shaped.empty:
                continue

            closed, dropped = drop_forming_bar(shaped, tf)
            if dropped is not None:
                self.last_forming[symbol] = dropped
            if closed.empty:
                continue
            out = closed.tail(int(lookback_bars)).reset_index(drop=True)
            out["symbol"] = symbol
            bars[symbol] = canonical_order(out)
            sources[symbol] = symbol if native is not None else f"{symbol} 1m"
        return bars, sources


def availability(spool_dir: Path | str | None = None) -> tuple[bool, str]:
    """`(usable, why not)` for the NT8 feed — a directory check, nothing more."""
    directory = configured_spool_dir(spool_dir)
    if not directory.is_dir():
        return False, (f"no NT8 bar spool at {directory} — the NinjaScript "
                       f"publisher is not installed or the share is not "
                       f"mounted")
    files = [p for p in directory.iterdir()
             if p.is_file() and p.suffix.lower() in SPOOL_SUFFIXES]
    if not files:
        return False, f"{directory} exists but holds no bar files yet"
    return True, f"{len(files)} spool file(s) in {directory}"
