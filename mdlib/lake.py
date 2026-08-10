"""
mdlib.lake - the single way to get bars out of the lake.

Location:  ~/src/trading/mdlib/lake.py

Every strategy reads through this module. That is the point: one definition
of what "ES 1-hour" means, so two backtests can never silently disagree.

    from mdlib.lake import get_bars

    df = get_bars(["ES", "NQ", "CL"], tf="1d", start="2013-01-01")

Design decisions encoded here
-----------------------------

**Sunday sessions are merged into Monday** (daily and weekly only).

CME opens Sunday 18:00 ET, so a UTC calendar day boundary produces ~51 short
Sunday "days" per year. ES 2022-01-02 carries 13,421 volume against 1,267,352
the next day.

Left alone, these break every lookback window: a 20-day moving average
computed over bars including Sunday stubs is really ~17 days plus 3 fragments,
each weighted equally with a full session. The distortion is irregular too -
some weeks have a Sunday bar and some do not - so it cannot be corrected with
a constant. And "yesterday's close" would sometimes mean a thin Sunday evening
print rather than a real settlement.

Pass `session_merge=False` for the raw bars as Databento delivered them.

**The merge happens here, not in the lake.** Stored data stays exactly as
received. If this decision proves wrong it is one function to change, not a
re-ingest.

**Only 1m and 1d are stored.** Everything else is derived:

    1m (native) -> 30m, 1h, 4h
    1d (native) -> 1w

An hour has no session semantics - it is sixty minutes, so there is no
convention to get wrong. Deriving also guarantees the higher timeframes agree
with the minute data a backtest actually fills on.

**Long format by default**, one row per symbol per bar. Cross-sectional work
across many symbols is the normal case, not the exception. Use `wide()` to
pivot when a single column per symbol is more convenient.

Coverage caveat
---------------
1-minute density is much lower before 2013 for ten of the symbols - the same
volume recorded across fewer distinct minutes. Nothing is missing (volume
reconciles exactly against the native daily bars), but a 30m or 1h bar built
from sparse minutes behaves differently from one built from dense minutes.

`intraday_start_year()` returns the year from which each symbol is dense.
Respect it for intraday work; daily and swing can use the full history.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

# --------------------------------------------------------------------------
LAKE = Path("/mnt/backtest/lake/futures/bars")
REF = Path("/mnt/backtest/reference/futures")

NATIVE_TFS = {"1m", "1d"}

# Which native timeframe each derived timeframe is built from, and the pandas
# resample rule to use.
DERIVED = {
    "5m":  ("1m", "5min"),
    "15m": ("1m", "15min"),
    "30m": ("1m", "30min"),
    "1h":  ("1m", "1h"),
    "2h":  ("1m", "2h"),
    "4h":  ("1m", "4h"),
    "1w":  ("1d", "W-MON"),
}

OHLCV = {
    "open": ("open", "first"),
    "high": ("high", "max"),
    "low": ("low", "min"),
    "close": ("close", "last"),
    "volume": ("volume", "sum"),
}


class LakeError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def available_symbols() -> tuple[str, ...]:
    """Symbols present in the lake."""
    if not LAKE.exists():
        raise LakeError(f"Lake not found: {LAKE}")
    return tuple(sorted(
        d.name.split("=", 1)[1] for d in LAKE.iterdir()
        if d.is_dir() and d.name.startswith("symbol=")
    ))


@lru_cache(maxsize=1)
def coverage() -> pd.DataFrame:
    """Per-symbol coverage table from scripts/coverage_summary.py."""
    p = REF / "coverage.csv"
    if not p.exists():
        raise LakeError(f"{p} not found. Run scripts/coverage_summary.py first.")
    return pd.read_csv(p)


def intraday_start_year(symbol: str) -> int | None:
    """
    First year from which this symbol's 1-minute data is dense enough for
    intraday work. Earlier years are not wrong - just coarser.
    """
    cov = coverage()
    row = cov[cov["symbol"] == symbol]
    if row.empty:
        return None
    v = row["intraday_start_year"].iloc[0]
    return None if pd.isna(v) else int(v)


@lru_cache(maxsize=1)
def degraded_days() -> frozenset[date]:
    """
    Sessions Databento flagged as reduced quality.

    31 days across 2010-2026. Small enough to simply exclude, and worth
    excluding from stress tests in particular - a strategy that looks robust
    through a crisis because the worst bars are missing is a dangerous
    conclusion.
    """
    p = REF / "degraded_days.csv"
    if not p.exists():
        return frozenset()
    df = pd.read_csv(p)
    if df.empty or "date" not in df.columns:
        return frozenset()
    return frozenset(pd.to_datetime(df["date"]).dt.date)


@lru_cache(maxsize=64)
def roll_dates(symbol: str) -> tuple[date, ...]:
    """
    Dates on which the continuous series switched contracts.

    Databento does not back-adjust, so there is a genuine price gap at each
    roll. Exclude these days rather than trading the gap.
    """
    p = REF / f"roll_calendar_{symbol}.json"
    if not p.exists():
        return ()
    try:
        data = json.loads(p.read_text())
    except Exception:
        return ()
    intervals = data.get("result", {}).get(f"{symbol}.v.0", [])
    out = []
    for iv in intervals:
        d0 = iv.get("d0")
        if d0:
            try:
                out.append(pd.Timestamp(d0).date())
            except Exception:
                pass
    return tuple(sorted(set(out)))


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _read_native(symbol: str, tf: str,
                 start: pd.Timestamp | None,
                 end: pd.Timestamp | None) -> pd.DataFrame:
    """Read a native timeframe, using partition pruning on year."""
    base = LAKE / f"symbol={symbol}" / f"tf={tf}"
    if not base.exists():
        return pd.DataFrame()

    # Only touch the year directories the request needs. At ~1ms NFS latency
    # per file, skipping irrelevant years matters more than it looks.
    years = sorted(int(d.name.split("=")[1]) for d in base.iterdir()
                   if d.is_dir() and d.name.startswith("year="))
    if start is not None:
        years = [y for y in years if y >= start.year]
    if end is not None:
        years = [y for y in years if y <= end.year]
    if not years:
        return pd.DataFrame()

    # pyarrow accepts a list of FILES or a single directory, but not a list of
    # directories - so collect the parquet files explicitly.
    files = []
    for y in years:
        files.extend(str(p) for p in (base / f"year={y}").rglob("*.parquet"))
    if not files:
        return pd.DataFrame()

    try:
        table = ds.dataset(sorted(files), format="parquet").to_table()
    except Exception as e:
        raise LakeError(f"{symbol} {tf}: {e}") from e

    df = table.to_pandas()
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")

    if start is not None:
        df = df[df["ts"] >= start]
    if end is not None:
        df = df[df["ts"] <= end]

    keep = ["ts", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    df["symbol"] = symbol
    return df


def _merge_sunday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fold Sunday bars into the following Monday.

    Sunday (dayofweek 6) is the opening stub of the week's session. Shifting
    it forward one day and re-aggregating gives a bar that spans the whole
    session: Sunday's open, the range across both, Monday's close, summed
    volume.

    Edge case: if the Monday is a holiday with no bar, the Sunday stub becomes
    the Monday bar on its own. Rare, and better than leaving a fragment.
    """
    if df.empty:
        return df

    d = df.copy()
    is_sunday = d["ts"].dt.dayofweek == 6
    d["session"] = d["ts"].dt.normalize()
    d.loc[is_sunday, "session"] = d.loc[is_sunday, "session"] + pd.Timedelta(days=1)

    if not is_sunday.any():
        return df

    out = (d.groupby("session", sort=True)
             .agg(**OHLCV)
             .reset_index()
             .rename(columns={"session": "ts"}))
    out["symbol"] = df["symbol"].iloc[0]
    return out


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    sym = df["symbol"].iloc[0]
    out = (df.set_index("ts")
             .resample(rule, label="left", closed="left")
             .agg(**OHLCV)
             .dropna(subset=["open"])
             .reset_index())
    out["symbol"] = sym
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def get_bars(symbols: str | list[str],
             tf: str = "1d",
             start: str | pd.Timestamp | None = None,
             end: str | pd.Timestamp | None = None,
             session_merge: bool = True,
             exclude_degraded: bool = False,
             exclude_rolls: bool = False,
             respect_coverage: bool = False) -> pd.DataFrame:
    """
    Return bars in long format: ts, symbol, open, high, low, close, volume.

    Parameters
    ----------
    symbols
        One symbol or a list. A list is the normal case - cross-sectional
        testing across many symbols is far stronger evidence than one symbol
        at a time.
    tf
        Native: "1m", "1d". Derived: "5m", "15m", "30m", "1h", "2h", "4h", "1w".
    session_merge
        Fold Sunday stubs into Monday. Applies to daily and weekly only, where
        the bar is meant to represent a session. Default True - see module
        docstring for why.
    exclude_degraded
        Drop the 31 sessions Databento flagged as reduced quality. Worth
        turning on for stress tests.
    exclude_rolls
        Drop contract roll days. Databento does not back-adjust, so those bars
        contain a genuine price gap that is not a tradeable move.
    respect_coverage
        Clamp the start date to each symbol's intraday_start_year. Only
        meaningful for intraday timeframes.
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    start_ts = pd.Timestamp(start, tz="UTC") if start is not None else None
    end_ts = pd.Timestamp(end, tz="UTC") if end is not None else None

    if tf in NATIVE_TFS:
        source_tf, rule = tf, None
    elif tf in DERIVED:
        source_tf, rule = DERIVED[tf]
    else:
        raise ValueError(
            f"Unknown timeframe {tf!r}. "
            f"Native: {sorted(NATIVE_TFS)}. Derived: {sorted(DERIVED)}."
        )

    is_session_tf = tf in ("1d", "1w")
    frames = []

    for sym in symbols:
        sym_start = start_ts
        if respect_coverage and source_tf == "1m":
            y = intraday_start_year(sym)
            if y is not None:
                floor = pd.Timestamp(f"{y}-01-01", tz="UTC")
                sym_start = floor if sym_start is None else max(sym_start, floor)

        df = _read_native(sym, source_tf, sym_start, end_ts)
        if df.empty:
            continue

        # Merge Sunday BEFORE resampling to weekly, so the weekly bar is built
        # from whole sessions rather than fragments.
        if session_merge and is_session_tf and source_tf == "1d":
            df = _merge_sunday(df)

        if rule is not None:
            df = _resample(df, rule)

        if exclude_degraded:
            bad = degraded_days()
            if bad:
                df = df[~df["ts"].dt.date.isin(bad)]

        if exclude_rolls:
            rolls = set(roll_dates(sym))
            if rolls:
                df = df[~df["ts"].dt.date.isin(rolls)]

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high",
                                     "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    cols = ["ts", "symbol", "open", "high", "low", "close", "volume"]
    return out[cols].sort_values(["ts", "symbol"]).reset_index(drop=True)


def wide(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """
    Pivot long-format bars to one column per symbol.

    Useful for correlation analysis and for feeding vectorbt, which expects
    a column per configuration.
    """
    return df.pivot(index="ts", columns="symbol", values=field).sort_index()


def describe(symbols: str | list[str] | None = None) -> pd.DataFrame:
    """Quick summary of what is available, for use at a prompt."""
    cov = coverage()
    if symbols:
        if isinstance(symbols, str):
            symbols = [symbols]
        cov = cov[cov["symbol"].isin(symbols)]
    cols = ["symbol", "first_date", "last_date", "mature_bars_per_day",
            "intraday_start_year", "median_daily_volume"]
    return cov[[c for c in cols if c in cov.columns]]
