"""
The pre-computed regime feature cache: ADX(14), ATR(14) and the four-quadrant
volatility/trend label, computed ONCE per (symbol, timeframe) and read back by
`mdlib.lake`.

Why a cache at all
------------------
Every pipeline stage that profiles a result recomputes Wilder's ADX(14) and
ATR(14) over the same bars. On a 27-contract x 4-timeframe screen that is the
same few hundred indicator passes repeated in every stage, and the numbers are
a pure function of the bars - nothing about a strategy changes them.

Why the volatility threshold is pinned to the IN-SAMPLE window
--------------------------------------------------------------
The quadrant boundary is a median ATR, and a median computed over whatever
window the caller happened to ask for is not a property of the contract - it is
a property of the request. Two consequences, both silent:

  * A backtest reading 2013-2022 and a holdout run reading 2023-2026 would
    label the SAME bar differently, because each took its own median. Gate 3
    would then measure retention between two strategies whose regime
    definitions disagree.
  * A median taken over a window that includes the holdout has read the
    holdout. It is a mild leak, but it is a leak, and it is invisible.

So theta_vol is computed strictly from the in-sample window (default
2013-01-01 .. 2022-12-31), written into the file, and applied to the WHOLE
series - including the out-of-sample years, which are then labelled by a
boundary that never saw them.

**This differs from `backtest.profiler.RegimeProfiler`**, which takes the median
of whatever frame it was handed. This module does NOT change that class; the
two are separate, and Stage 1's screening numbers are unaffected by anything
here. See `docs`/the task notes - reconciling them is its own scoped change.

The warm-up sentinel
--------------------
ADX(14) and ATR(14) are undefined for the first bars of a series. `NaN > 25.0`
is False, so a naive comparison silently files every warm-up bar under
"Low Volatility / Ranging" - a real quadrant label, on bars where no indicator
exists. `regime_quadrant` is `uint8` and cannot hold NaN, so those bars are
stamped `0 = UNDEFINED` instead. A consumer must treat 0 as "no regime", never
as a quadrant.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------
# The lake root that actually exists on this box. The task specification named
# `/mnt/data/lake/regimes/`; there is no `/mnt/data` mount here and the futures
# lake lives under `/mnt/backtest/lake`, so the cache is placed beside it
# rather than at a path nothing could read. `$BT_REGIME_CACHE` overrides.
DEFAULT_CACHE_DIR = Path("/mnt/backtest/lake/regimes")

ADX_LENGTH = 14
ATR_LENGTH = 14

# ADX above this is a trend. The same 25.0 `backtest.profiler` uses; a second
# spelling of the boundary in a second module is how a cached quadrant and a
# freshly-profiled one come to disagree with nothing raising.
ADX_TREND_THRESHOLD = 25.0

# The encoding, and the ONLY place it is written down. Order matches
# `backtest.profiler.REGIMES` so a quadrant integer and a profiler label are
# the same statement about the same bar.
QUADRANT_UNDEFINED = 0
QUADRANT_LABELS: dict[int, str] = {
    0: "Undefined (indicator warm-up)",
    1: "High Volatility / Trending",
    2: "High Volatility / Ranging",
    3: "Low Volatility / Trending",
    4: "Low Volatility / Ranging",
}

# The 10-year in-sample window the pipeline optimises on.
DEFAULT_IS_START = "2013-01-01"
DEFAULT_IS_END = "2022-12-31"

REGIME_COLUMNS = ["adx_14", "atr_14", "is_trending", "is_high_vol",
                  "regime_quadrant"]


class RegimeCacheError(RuntimeError):
    pass


def cache_dir() -> Path:
    """
    Read at CALL time, never bound at import - a test that sets
    `$BT_REGIME_CACHE` must be able to set it after this module is imported.
    """
    return Path(os.environ.get("BT_REGIME_CACHE", str(DEFAULT_CACHE_DIR)))


def cache_path(symbol: str, tf: str, root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else cache_dir()
    return base / f"{symbol}_{tf}_regime.parquet"


# --------------------------------------------------------------------------
# The math
# --------------------------------------------------------------------------
def _wilder_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Wilder's ADX(14) and ATR(14) over `bars`, as float32 columns named
    `adx_14` / `atr_14`, indexed by the bar timestamp.

    Computed with `pandas_ta` in the same call `backtest.profiler` makes
    (`.ta.adx(length=14)` / `.ta.atr(length=14)`, the latter defaulting to the
    RMA smoothing that IS Wilder's) so the cache and a live profile are the
    same numbers rather than two good-faith implementations of the same paper.
    """
    import pandas_ta as ta  # noqa: F401  - registers the .ta accessor

    if bars.empty:
        raise RegimeCacheError("no bars: refusing to write a regime file that "
                               "describes nothing")
    for col in ("high", "low", "close"):
        if col not in bars.columns:
            raise RegimeCacheError(f"bars carry no {col!r} column")

    ts = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    if not ts.is_monotonic_increasing:
        raise RegimeCacheError("bars are not sorted by ts; a rolling Wilder "
                               "average over unsorted bars is meaningless")
    if ts.has_duplicates:
        raise RegimeCacheError("duplicate timestamps in the bar frame; the "
                               "left-join back onto the lake would fan out")

    work = pd.DataFrame(
        {"high": bars["high"].to_numpy(dtype="float64"),
         "low": bars["low"].to_numpy(dtype="float64"),
         "close": bars["close"].to_numpy(dtype="float64")},
        index=ts,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        work.ta.adx(length=ADX_LENGTH, append=True)
        work.ta.atr(length=ATR_LENGTH, append=True)

    adx_cols = [c for c in work.columns if c.startswith("ADX_")]
    atr_cols = [c for c in work.columns
                if c.startswith("ATRr_") or c.startswith("ATRe_")
                or c.startswith("ATR_")]
    if not adx_cols or not atr_cols:
        raise RegimeCacheError(
            f"pandas_ta produced no ADX/ATR column (got {list(work.columns)}); "
            f"refusing to guess which column is which")

    out = pd.DataFrame(index=ts)
    out.index.name = "timestamp"
    out["adx_14"] = work[adx_cols[0]].astype("float32").to_numpy()
    out["atr_14"] = work[atr_cols[0]].astype("float32").to_numpy()
    return out


def in_sample_theta(atr: pd.Series,
                    is_start: str = DEFAULT_IS_START,
                    is_end: str = DEFAULT_IS_END) -> float:
    """
    theta_vol = median(ATR_14) over the in-sample window ALONE.

    Raises when the window holds no defined ATR. A NaN threshold compares False
    against everything, which would stamp the entire series "low volatility" -
    a fully-populated file, four quadrants collapsed into two, and nothing
    raising anywhere downstream.
    """
    lo = pd.Timestamp(is_start, tz="UTC")
    hi = pd.Timestamp(is_end, tz="UTC")
    if hi <= lo:
        raise RegimeCacheError(f"in-sample window ends at or before it starts: "
                               f"{is_start} .. {is_end}")

    window = atr.loc[(atr.index >= lo) & (atr.index <= hi)].dropna()
    if window.empty:
        raise RegimeCacheError(
            f"no defined ATR({ATR_LENGTH}) inside the in-sample window "
            f"{is_start} .. {is_end}; theta_vol would be NaN and every bar "
            f"would be labelled low-volatility")
    return float(window.median())


def classify(frame: pd.DataFrame, theta_vol: float) -> pd.DataFrame:
    """
    Add `is_trending`, `is_high_vol` and `regime_quadrant` to a frame that
    already carries `adx_14` / `atr_14`.

    Warm-up bars - either indicator NaN - get quadrant 0, not a quadrant.
    """
    out = frame.copy()
    defined = out["adx_14"].notna() & out["atr_14"].notna()

    out["is_trending"] = (out["adx_14"] > ADX_TREND_THRESHOLD).fillna(False)
    out["is_high_vol"] = (out["atr_14"] > theta_vol).fillna(False)
    out["is_trending"] = out["is_trending"].astype(bool) & defined
    out["is_high_vol"] = out["is_high_vol"].astype(bool) & defined

    # 1 HV/Trend, 2 HV/Range, 3 LV/Trend, 4 LV/Range - see QUADRANT_LABELS.
    quad = pd.Series(QUADRANT_UNDEFINED, index=out.index, dtype="uint8")
    hv, tr = out["is_high_vol"], out["is_trending"]
    quad[defined & hv & tr] = 1
    quad[defined & hv & ~tr] = 2
    quad[defined & ~hv & tr] = 3
    quad[defined & ~hv & ~tr] = 4
    out["regime_quadrant"] = quad
    return out[REGIME_COLUMNS]


def build(bars: pd.DataFrame,
          is_start: str = DEFAULT_IS_START,
          is_end: str = DEFAULT_IS_END) -> tuple[pd.DataFrame, float]:
    """Full regime frame for one symbol's bars, plus the theta_vol used."""
    frame = _wilder_frame(bars)
    theta = in_sample_theta(frame["atr_14"], is_start, is_end)
    return classify(frame, theta), theta


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
# Provenance travels INSIDE the parquet, as schema key-value metadata, rather
# than in a sidecar JSON that can be separated from the file it describes.
# theta_vol is the one number a reader cannot recover from the columns, and a
# quadrant read without knowing which window drew its boundary is not a
# measurement.
_META_KEY = b"bt_regime_provenance"


def write_cache(regimes: pd.DataFrame,
                symbol: str,
                tf: str,
                theta_vol: float,
                is_start: str,
                is_end: str,
                bar_flags: dict | None = None,
                root: Path | str | None = None) -> Path:
    """Write one `{SYMBOL}_{TF}_regime.parquet`, atomically."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = cache_path(symbol, tf, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    quad = regimes["regime_quadrant"]
    provenance = {
        "symbol": symbol,
        "tf": tf,
        "theta_vol": float(theta_vol),
        "is_start": is_start,
        "is_end": is_end,
        "adx_length": ADX_LENGTH,
        "atr_length": ATR_LENGTH,
        "adx_trend_threshold": ADX_TREND_THRESHOLD,
        "rows": int(len(regimes)),
        # The hygiene flags the BARS were read under. A cache built on bars
        # that included roll days holds an ATR shaped by those gaps; joined
        # onto a run that excluded them, every timestamp still matches and the
        # numbers are quietly from a different series.
        "bar_flags": bar_flags or {},
        "first_ts": str(regimes.index[0]) if len(regimes) else None,
        "last_ts": str(regimes.index[-1]) if len(regimes) else None,
        "undefined_bars": int((quad == QUADRANT_UNDEFINED).sum()),
        "quadrant_labels": {str(k): v for k, v in QUADRANT_LABELS.items()},
    }

    table = pa.Table.from_pandas(regimes, preserve_index=True)
    md = dict(table.schema.metadata or {})
    md[_META_KEY] = json.dumps(provenance, indent=2).encode()
    table = table.replace_schema_metadata(md)

    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def provenance(symbol: str, tf: str,
               root: Path | str | None = None) -> dict | None:
    """The build record stored inside the file, or None when there is none."""
    import pyarrow.parquet as pq

    path = cache_path(symbol, tf, root)
    if not path.exists():
        return None
    md = pq.read_schema(path).metadata or {}
    raw = md.get(_META_KEY)
    return json.loads(raw.decode()) if raw else None


def load_cache(symbol: str, tf: str,
               root: Path | str | None = None) -> pd.DataFrame | None:
    """The cached regime frame, or None when no file exists."""
    path = cache_path(symbol, tf, root)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RegimeCacheError(f"{path} is not indexed by timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"
    return df


# One warning per (symbol, tf) per process. A miss inside a 27-contract loop
# would otherwise print 27 identical lines, which trains a reader to skip them.
_warned: set[tuple] = set()


def _warn_once(key: tuple, message: str) -> None:
    """
    Announce once, on STDERR.

    Not stdout: `mdlib.lake` is a library, and several callers here parse a
    child process's stdout as data (`tests/test_streaming_lake.py` reads four
    whitespace-separated numbers off one). A cache-miss line printed there
    corrupts the payload rather than informing anybody, and the caller fails
    with a parse error that says nothing about regimes.
    """
    if key not in _warned:
        _warned.add(key)
        print(f"[regimes] {message}", file=sys.stderr, flush=True)


def attach(bars: pd.DataFrame,
           symbol: str,
           tf: str,
           bar_flags: dict | None = None,
           root: Path | str | None = None) -> pd.DataFrame:
    """
    Left-join the cached regime columns onto ONE symbol's bars, on `ts`.

    On a cache miss the frame is returned UNCHANGED - no regime columns at
    all - and the miss is announced once. It is deliberately not filled with
    quadrant 0: a consumer testing `"regime_quadrant" in bars.columns` then
    gets a truthful answer, where an all-zero column would read as a cache that
    computed the regime and found none.

    Computing ADX/ATR on the fly here instead was rejected. theta_vol is a
    median over the in-sample window, and a reader handed an arbitrary
    `start`/`end` cannot take it without either re-reading years the caller did
    not ask for or silently substituting the median of the requested window -
    which makes a bar's quadrant depend on the date range it was read under.
    Build the cache; do not improvise it.
    """
    if bars.empty:
        return bars

    key = (symbol, tf)
    cached = load_cache(symbol, tf, root)
    if cached is None:
        _warn_once(key + ("miss",),
                   f"no cache for {symbol} {tf} at "
                   f"{cache_path(symbol, tf, root)} - bars returned without "
                   f"regime columns. Build it with "
                   f"scripts/precompute_regimes.py --symbols {symbol} "
                   f"--tf {tf}")
        return bars

    if bar_flags is not None:
        built = (provenance(symbol, tf, root) or {}).get("bar_flags")
        if built is not None and built != bar_flags:
            _warn_once(key + ("flags",),
                       f"{symbol} {tf}: regime cache was built on bars read "
                       f"with {built}, joined onto bars read with {bar_flags}. "
                       f"Timestamps still match; the ADX/ATR behind them came "
                       f"from a different bar series.")

    out = bars.merge(cached, how="left", left_on="ts", right_index=True)

    # A left join fills a non-matching row with NaN, which turns uint8 into
    # float64 and bool into object. Those bars are outside the cache's span,
    # so they are UNDEFINED, and the dtypes stated in the contract are
    # restored rather than left as whatever the join produced.
    missing = out["regime_quadrant"].isna()
    if missing.any():
        _warn_once(key + ("gap",),
                   f"{symbol} {tf}: {int(missing.sum())} of {len(out)} bars "
                   f"fall outside the regime cache's span "
                   f"({cached.index[0]} .. {cached.index[-1]}) and are "
                   f"stamped quadrant {QUADRANT_UNDEFINED}. Rebuild the cache "
                   f"to cover them.")
    out["regime_quadrant"] = out["regime_quadrant"].fillna(
        QUADRANT_UNDEFINED).astype("uint8")
    out["is_trending"] = out["is_trending"].fillna(False).astype(bool)
    out["is_high_vol"] = out["is_high_vol"].fillna(False).astype(bool)
    out["adx_14"] = out["adx_14"].astype("float32")
    out["atr_14"] = out["atr_14"].astype("float32")
    return out
