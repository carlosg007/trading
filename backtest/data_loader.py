"""
Temporal chunking for the sweep: rolling multi-year blocks of one contract's
bars, each carrying a warm-up tail from the block before it, so a 16-year
parameter search never holds 16 years of signal masks in RAM at once.

Location:  ~/src/trading/backtest/data_loader.py

WHAT THE MEMORY PROBLEM ACTUALLY IS
===================================
Worth stating precisely, because the obvious answer is wrong and the real peak
is somewhere else.

`mdlib.lake.iter_bars` already streams one SYMBOL at a time, so a full-lake run
never holds the lake — peak tracks the largest single contract (5.6M 1-minute
rows, ~2.9 GiB for a whole run). That half of the problem was solved in 2026-08.

The peak that actually kills a Stage 2 sweep is in `backtest/scan.py`, and it is
not the bars. `scan_symbol` builds FOUR boolean masks per grid cell over the
whole history and then `np.column_stack`s them:

    bytes ~ 4 masks x n_bars x n_combinations x 1 byte

For one contract of 1-minute bars over 16 years against a 432-cell grid:

    4 x 5,600,000 x 432  =  9.7 GiB, in four allocations, before the sweep runs

`_simulate_columns` batches COLUMNS inside the vectorbt call (`MAX_CELLS`), but
that batching happens AFTER the full masks exist, so it does not touch this
peak. Chunking the bars is what touches it: eight 2-year chunks divide `n_bars`
by eight and the same sweep peaks at ~1.2 GiB.

    `projected_sweep_bytes` below is that arithmetic, and Stage 2 prints it.

THIS IS NOT THE ENGINE'S CHUNKING, AND THE DIFFERENCE IS THE WHOLE RISK
=======================================================================
`backtest/engine.py::_chunk_bounds` already splits a run into batches of bars,
and it is EXACT: it places a boundary only where the strategy is flat, so the
trade list is bit-identical at any chunk size (`tests/test_engine_batching.py`
pins that). CLAUDE.md states the reason in one line — **"Never chunk by
calendar year — a position open on 31 December is silently dropped, which
flatters results."**

This module chunks by CALENDAR YEAR, which is exactly the thing that line
forbids, and it does so because a data loader cannot know where the trades are:
the boundaries have to be fixed before any signal exists. So the prohibition is
not ignored here, it is PAID FOR, in three parts:

1. **A settlement tail.** Each chunk's frame runs past its payload by
   `settlement_bars`, so a trade opened just inside the payload has room to
   close inside the same simulation.
2. **Attribution by ENTRY.** A trade belongs to the chunk whose PAYLOAD window
   contains its entry timestamp, and to no other. Trades entering in the warm-up
   belong to the previous chunk and are discarded here; trades entering in the
   settlement tail belong to the next chunk and are discarded here. That is what
   makes the concatenation across chunks a partition rather than a pile — no
   trade is counted twice and none is attributed to two windows.
3. **Truncation is COUNTED, never silently dropped.** A trade that is still open
   when its chunk's frame ends is invisible to `vbt.Portfolio.trades`
   (`status == 1` keeps closed trades only), which is precisely the flattering
   failure the CLAUDE.md line describes. `backtest.scan` asks
   `_simulate_columns` for the OPEN-trade count per column and reports it as
   `chunking.truncated_trades`. A sweep whose count is non-zero has dropped
   trades and says so; the fix is a longer `settlement_bars`, not a smaller
   number.

**A chunked sweep is therefore an APPROXIMATION of the contiguous one, and the
engine's batching is not.** Two residual differences survive even with a
generous tail, both of them properties of a path-dependent strategy rather than
bugs:

  * a strategy whose position state at the payload boundary depends on trades
    that opened before the warm-up can resolve a boundary trade differently from
    the contiguous run;
  * a trade longer than `settlement_bars` is lost from the chunk that opened it
    (and counted, see 3).

Do not use this to produce a number that will be compared against a contiguous
run of the same parameters. Use it to make a sweep that would otherwise OOM
complete, then re-run the WINNER contiguously — which is what Stage 3 does
anyway, with the parameters locked.

WARM-UP IS NOT FREE AND 500 BARS IS NOT ENOUGH FOR THIS REPOSITORY'S OWN
STRATEGIES
=======================================================================
The point of the warm-up is that an indicator at the first payload bar reads
what it would have read in the contiguous run. For a WINDOWED indicator that is
exactly true once the window is full: an SMA(200) or a `rolling(20).std()` needs
200 and 20 bars and is then identical to the last bit.

For a RECURSIVE one it is never exactly true, only exponentially close. An EMA
seeded at the wrong value carries `(1 - alpha)^w` of that error after `w` bars,
so the warm-up needed to get the seed's influence under a tolerance is

    w  =  ceil( ln(tolerance) / ln(1 - alpha) )

which `required_warmup_bars` computes. At the 1e-6 tolerance this repository's
tests use:

    span EMA(20)     alpha = 0.0952      138 bars
    span EMA(50)     alpha = 0.0392      346 bars
    Wilder(14)       alpha = 0.0714      187 bars      (RSI, ATR)
    span EMA(200)    alpha = 0.00995   1,382 bars
    Wilder(200)      alpha = 0.005     2,757 bars

THOSE FIGURES ARE RELATIVE TO THE SEED ERROR, WHICH IS PRICE-SCALE, so they
are a FLOOR and not the answer for an absolute tolerance. `ewm` restarts the
recursion at the chunk's first value, so the error to be shed is hundreds of
points on a 4,000-point contract rather than order one: a span EMA(50) warmed
for the 346 bars above still carries ~1e-4 IN PRICE UNITS, which
`tests/test_temporal_chunking.py` measures at 9.4e-05. Sizing against an
absolute tolerance means passing the price level as `seed_error`, which adds
`ln(seed_error) / -ln(1 - alpha)` bars — another 207 for that EMA.
`auto_warmup_bars` does this by default through `SEED_ERROR_SCALE`.

**The requested default of 500 bars is enough for a Wilder(14) ATR and a span
EMA(50), and is NOT enough for the 200-period trend EMA that
`double_rsi_macd_scalp_20260823` and `ema_trend_filter` both use** — at 500 bars
a 200-EMA still carries 0.67% of its seed error, roughly four thousand times the
1e-6 tolerance. It is kept as the signature default because the specification
asks for it; `backtest.scan` does NOT use it, and passes
`warmup_bars="auto"`, which reads the strategy's own declared parameters and
sizes the window from the slowest one. `tests/test_temporal_chunking.py` pins
both halves of this — that the derived warm-up meets 1e-6 and that 500 bars
does not.

CHAINED recursions compound: `t3_braid_scalp_20260823` chains six EMAs, so its
convergence is the single-stage decay times a polynomial in `w`. Pass
`stages=6` for a conservative bound rather than assuming one stage.

ZERO-COPY, AND WHERE IT STOPS
=============================
For an in-memory frame the chunk is `frame.iloc[lo:hi]` over a column
projection, which under pandas 3's copy-on-write shares the parent's buffers —
`tests/test_temporal_chunking.py` asserts the shared memory rather than
trusting it. **That means an in-memory source cannot reduce the peak of holding
the frame it was handed**, only the peak of everything computed downstream of
it. The load-side saving needs a source that can be read a slice at a time: a
parquet path, or `LakeSource`, which re-reads each chunk's own years through
`mdlib.lake` and its year-partition pruning.

The cost of `LakeSource` is honest and worth naming: consecutive chunks overlap
by the warm-up, so a boundary year is read twice. On an NFS mount that is the
price of not holding 16 years of bars, and it is paid once per chunk rather than
once per grid cell.
"""

from __future__ import annotations

import gc
import math
import resource
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.memory_guard import (DEFAULT_GUARD,                # noqa: E402
                                   MemoryGuard, MemorySafetyException)

# The columns a sweep actually reads. Everything else in a lake frame is
# carried through every slice, every copy and every chunk for nothing.
#
# `regime_quadrant` is in the list because Stage 1's screen and the profiler
# read it and it arrives on the frame from `mdlib.lake`'s regime join; a
# projection that dropped it would make a chunked frame silently un-profilable.
# A column that is absent from the source is skipped rather than raising — the
# regime cache MISSES for uncached (symbol, tf) pairs by design, and a miss
# there means the column is simply not present.
DEFAULT_COLUMNS: tuple[str, ...] = (
    "ts", "symbol", "open", "high", "low", "close", "volume",
    "regime_quadrant",
)

# The request's defaults. See the module docstring for why the warm-up one is
# not what `backtest.scan` uses.
DEFAULT_CHUNK_YEARS = 2
DEFAULT_WARMUP_BARS = 500

# The tolerance the warm-up arithmetic is quoted at, and the one the test suite
# holds a chunked indicator to.
WARMUP_TOLERANCE = 1e-6

# The seed error `auto_warmup_bars` sizes against, in PRICE UNITS. A recursion
# restarted at a chunk boundary begins at that chunk's first value, so the
# error it has to shed is price-scale rather than order-one — see
# `required_warmup_bars`. 10,000 covers every contract in this lake (NQ trades
# near 20,000 and CL near 70, and over-estimating costs warm-up bars while
# under-estimating costs correctness); it is a CONSTANT rather than a reading
# of the bars because the warm-up has to be known before the bars are read.
SEED_ERROR_SCALE = 10_000.0

# A final block shorter than this fraction of a full chunk is absorbed into the
# block before it rather than becoming a chunk of its own. A 3-week eighth
# chunk would pay a full warm-up to sweep almost nothing, and — worse — it
# would be a payload window too short for any trade to close inside, so every
# trade opened in it would count as truncated.
MIN_TAIL_FRACTION = 0.5

# Nominal bar durations, for turning a warm-up expressed in BARS into a date
# window a partitioned reader can prune on. Deliberately generous: the lake has
# session gaps, holidays and thin overnight hours, so a window sized at the
# nominal rate would come back short of bars exactly where the market was
# closed. `_lookback_window` multiplies by `LOOKBACK_SLACK` and the result is
# then trimmed to the exact bar count, so over-reading costs one read and
# under-reading costs a silently short warm-up.
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
               "4h": 240, "1d": 1440, "1w": 10080}
LOOKBACK_SLACK = 4.0


class ChunkingError(RuntimeError):
    """A temporal chunking request that cannot be satisfied as asked."""


# --------------------------------------------------------------------------
# Warm-up arithmetic
# --------------------------------------------------------------------------
def required_warmup_bars(period: int,
                         kind: str = "span",
                         tolerance: float = WARMUP_TOLERANCE,
                         stages: int = 1,
                         seed_error: float = 1.0) -> int:
    """
    Bars of warm-up needed before a `period`-length indicator has forgotten its
    seed to within `tolerance`.

    `kind` selects the recursion:

        "sma" / "rolling"   a windowed statistic. EXACT once the window is
                            full, so the answer is `period` and the tolerance
                            does not enter. Returned as `period + 1` because
                            every windowed indicator in this repository that
                            matters is computed on a `.diff()`, which costs one
                            bar before the window can start.
        "span"              a conventional EMA, `alpha = 2 / (period + 1)` —
                            the MACD family, the trend EMAs, Tillson's T3.
        "wilder"            Wilder's smoothing, `alpha = 1 / period` — the RSI
                            and the ATR. Roughly half the speed of a span EMA
                            at the same length, so it needs about twice the
                            warm-up, which is the trap in reading a "14-period"
                            label and assuming one number covers both.

    `stages` is for CHAINED recursions (T3 chains six EMAs). The bound returned
    is `stages` times the single-stage requirement, which is conservative
    rather than tight: the true decay of a chain is the single-stage decay
    multiplied by a polynomial in the bar count, and a conservative warm-up
    costs memory while a tight one costs correctness.

    `seed_error` IS NOT OPTIONAL DECORATION AND GETTING IT WRONG IS THE WHOLE
    TRAP. What decays is the SEED ERROR — the gap between the value the
    recursion starts at and the value it should have had — so `tolerance` is
    read RELATIVE to it:

        w  =  ceil( ln(tolerance / seed_error) / ln(1 - alpha) )

    At the default `seed_error = 1.0` the answer is the number of bars to shed
    a factor of `1/tolerance`, which is the right question when the tolerance
    is relative. IT IS THE WRONG QUESTION WHEN THE TOLERANCE IS ABSOLUTE, and
    the difference is not small. `ewm` seeds at the chunk's first value, so on
    a 4,000-point contract the seed error is hundreds of points; a span EMA(50)
    warmed for the 346 bars this returns at `seed_error=1.0` still carries
    ~1e-4 in PRICE UNITS, a hundred times an absolute 1e-6 tolerance. The
    measured figure on this repository's own fixture is 9.4e-05 —
    `tests/test_temporal_chunking.py` pins it. Pass the contract's price level
    as `seed_error` to size a warm-up against an absolute tolerance; it adds
    `ln(seed_error) / -ln(1 - alpha)` bars, which for a 4,000-point contract
    and a span EMA(50) is another 207.

    Raises on a non-positive period rather than returning a default — a zero
    period reaching here means a parameter was not bound, and a silently
    plausible warm-up would hide it until the numbers were wrong.
    """
    if period is None or period <= 0:
        raise ChunkingError(
            f"required_warmup_bars needs a positive period; got {period!r}")
    if not 0.0 < tolerance < 1.0:
        raise ChunkingError(
            f"tolerance must be within (0, 1); got {tolerance!r}")
    if stages < 1:
        raise ChunkingError(f"stages must be >= 1; got {stages!r}")

    period = int(period)
    kind = str(kind).lower()
    if kind in ("sma", "rolling", "window"):
        return stages * (period + 1)
    if kind == "span":
        alpha = 2.0 / (period + 1.0)
    elif kind == "wilder":
        alpha = 1.0 / period
    else:
        raise ChunkingError(
            f"kind must be one of 'span', 'wilder', 'sma'; got {kind!r}")

    if seed_error <= 0 or not math.isfinite(seed_error):
        raise ChunkingError(
            f"seed_error must be a finite number > 0; got {seed_error!r}")
    if alpha >= 1.0:                      # period 1: the "average" is the bar
        return stages
    ratio = tolerance / float(seed_error)
    if ratio >= 1.0:
        # The tolerance is already wider than the error being shed, so no
        # warm-up is needed on this indicator's account. One bar, not zero:
        # every recursion here still needs its first observation.
        return stages
    return stages * int(math.ceil(math.log(ratio) / math.log(1.0 - alpha)))


def auto_warmup_bars(params: dict | None = None,
                     grid: dict | None = None,
                     tolerance: float = WARMUP_TOLERANCE,
                     floor: int = DEFAULT_WARMUP_BARS,
                     stages: int = 1,
                     seed_error: float = SEED_ERROR_SCALE) -> int:
    """
    A warm-up sized from a strategy's OWN declared parameters, as the safest
    thing a caller can do without being told which of them are periods.

    THIS IS A HEURISTIC AND IT IS KEYED ON VALUES, NOT ON MEANING. It takes the
    largest positive whole number in `params` and every value of `grid`, and
    sizes the warm-up for a Wilder recursion of that length — Wilder because it
    is the slowest of the three forms, so the answer covers a span EMA of the
    same length as well.

    What it cannot know, stated rather than discovered later:

      * a parameter that is a whole number and NOT a period — a contract count,
        a bar offset — inflates the warm-up. That costs memory and nothing else.
      * a strategy whose slowest indicator length is a MODULE CONSTANT rather
        than a parameter is invisible here. `double_rsi_macd_scalp_20260823`
        pins its 200-bar trend EMA as `EMA_TREND_PERIOD` and does not expose
        it, so the value this returns for that module is sized off its RSI
        windows and is far too short. That is why the `floor` exists and why
        `backtest.scan` prints the warm-up it used: a number nobody can see is
        a number nobody can check.

    The floor is `DEFAULT_WARMUP_BARS`, so the answer is never SHORTER than the
    specification's default — only longer.

    `seed_error` defaults to `SEED_ERROR_SCALE`, a futures PRICE LEVEL, so the
    tolerance this sizes against is absolute in price units rather than
    relative to whatever the seed error happened to be — see
    `required_warmup_bars` for why that distinction is worth a constant.
    """
    values: list[int] = []
    for source in (params or {}, ):
        for v in source.values():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, np.integer)) and int(v) > 0:
                values.append(int(v))
    for axis in (grid or {}).values():
        if isinstance(axis, (str, bytes)) or not isinstance(axis, Sequence):
            axis = [axis]
        for v in axis:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, np.integer)) and int(v) > 0:
                values.append(int(v))

    if not values:
        return int(floor)
    longest = max(values)
    return int(max(floor, required_warmup_bars(longest, "wilder", tolerance,
                                               stages=stages,
                                               seed_error=seed_error)))


# --------------------------------------------------------------------------
# Memory arithmetic
# --------------------------------------------------------------------------
def projected_sweep_bytes(n_bars: int, n_columns: int, masks: int = 4) -> int:
    """
    The bytes `scan_symbol` will allocate for its stacked signal masks.

    Four boolean masks — long entries, long exits, short entries, short exits —
    each `n_bars x n_columns` at one byte a cell, and all four resident at once
    because `np.column_stack` materialises them before the sweep starts. This
    is the number that decides whether a sweep OOMs; the bars themselves are an
    order of magnitude smaller.

    It is a FLOOR, not a total: the per-combination masks are held in a list
    before they are stacked, so the true transient peak is close to twice this.
    Reported as the floor because the floor is the part that is certain.
    """
    return int(masks) * int(n_bars) * int(n_columns)


def suggest_chunk_years(n_bars: int,
                        n_columns: int,
                        span_years: float,
                        budget_bytes: int) -> int | None:
    """
    The largest whole number of years per chunk whose projected mask allocation
    fits inside `budget_bytes`, or None when the contiguous sweep already fits.

    Returns at least 1. A sweep that cannot fit inside the budget even at one
    year per chunk gets 1 and the caller is expected to say so rather than
    pretend the budget was met — chunking below a year is possible but a
    payload window that short leaves most trades truncated at a boundary.
    """
    if span_years <= 0 or n_bars <= 0 or n_columns <= 0:
        return None
    if projected_sweep_bytes(n_bars, n_columns) <= budget_bytes:
        return None
    bars_per_year = n_bars / float(span_years)
    affordable_bars = budget_bytes / (4.0 * n_columns)
    return max(1, int(math.floor(affordable_bars / max(bars_per_year, 1.0))))


def peak_rss_bytes() -> int:
    """
    This process's peak resident set size, in bytes.

    `ru_maxrss` is in KILOBYTES on Linux and in BYTES on macOS. This repository
    runs on Ubuntu and the conversion is written for it; the constant is named
    rather than inlined so the platform assumption is visible to whoever ports
    it.

    It is a HIGH-WATER MARK for the whole process and never falls, which is
    what makes it the right measure for "did this OOM" and the wrong one for
    "how much is live now". A test asserting a ceiling has to record it before
    and after and compare the DELTA, and even then it is measuring the
    interpreter's peak, not the chunker's.
    """
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# --------------------------------------------------------------------------
# Calendar spans
# --------------------------------------------------------------------------
def calendar_spans(first: pd.Timestamp,
                   last: pd.Timestamp,
                   chunk_years: int = DEFAULT_CHUNK_YEARS,
                   min_tail_fraction: float = MIN_TAIL_FRACTION
                   ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    The half-open payload spans `[start, end)` covering `first`..`last`.

    ANCHORED ON A GLOBAL GRID OF `chunk_years`-YEAR BLOCKS, not on the first
    bar: the first edge is January 1 of the largest multiple of `chunk_years`
    at or before the first bar's year. Anchoring on the first bar's own year
    would give NQ (history from 2013) and MNQ (from 2019) different edges, so
    the same flags would cut the two contracts differently and their chunked
    sweeps would not be comparable — and a data pull that moved a first bar by
    one year would silently re-cut every block of that contract. On the global
    grid both land on the same even years.

    The first span therefore often begins BEFORE the first bar, and that is
    inert: the payload starts at the first bar that exists, and a span holding
    no bars at all is never yielded as a chunk.

    Half-open throughout: `[2013-01-01, 2015-01-01)`. A closed end would put the
    bar stamped exactly `2015-01-01T00:00:00Z` in two chunks, and the trade
    attribution in `backtest.scan` would count it twice.

    The last span always reaches `last` inclusive of the final bar, and a tail
    shorter than `min_tail_fraction` of a full chunk is absorbed into the block
    before it — see `MIN_TAIL_FRACTION`.
    """
    if chunk_years <= 0:
        raise ChunkingError(f"chunk_years must be >= 1; got {chunk_years}")
    first = pd.Timestamp(first)
    last = pd.Timestamp(last)
    if first.tz is None or last.tz is None:
        raise ChunkingError(
            "calendar_spans needs tz-aware timestamps; the lake stores UTC and "
            "a naive bound would be compared against an aware index, which "
            "raises under pandas 3 rather than being silently read as UTC")
    if last < first:
        raise ChunkingError(f"last ({last}) is before first ({first})")

    tz = first.tz
    anchor_year = first.year - (first.year % int(chunk_years))
    anchor = pd.Timestamp(year=anchor_year, month=1, day=1, tz=tz)
    edges: list[pd.Timestamp] = []
    cursor = anchor
    # `last` is inclusive, so the final edge has to sit strictly past it.
    while cursor <= last:
        edges.append(cursor)
        cursor = pd.Timestamp(year=cursor.year + int(chunk_years), month=1,
                              day=1, tz=tz)
    edges.append(cursor)

    spans = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    if len(spans) > 1:
        # The tail's real extent is `last`, not the calendar edge past it.
        tail_len = (last - spans[-1][0]).total_seconds()
        full_len = (spans[-1][1] - spans[-1][0]).total_seconds()
        if full_len > 0 and tail_len < min_tail_fraction * full_len:
            merged = (spans[-2][0], spans[-1][1])
            spans = spans[:-2] + [merged]
    return spans


# --------------------------------------------------------------------------
# The chunk
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalChunk:
    """
    One block of bars, plus everything a consumer needs to know which part of
    it is real.

    `frame` is `warm-up + payload + settlement`, in that order, on a fresh
    `RangeIndex`. THE WHOLE FRAME IS FOR COMPUTING AND ONLY THE PAYLOAD IS FOR
    KEEPING: indicators are calculated over all of it so the payload's first
    bar reads what it would have read in the contiguous run, and results are
    then attributed to `payload_slice` alone. A consumer that keeps the warm-up
    rows double-counts every one of them against the previous chunk.

    The index is reset because the sweep is positional end to end —
    `unpack_signals`, `_shift_to_fill` and `_cost_arrays` all index by position
    — and a slice carrying the parent's original labels would put row 0 of a
    mask against label 1,400,000 of the frame.
    """

    frame: pd.DataFrame
    index: int
    n_chunks: int
    warmup_len: int
    payload_len: int
    settlement_len: int
    span_start: pd.Timestamp
    span_end: pd.Timestamp
    warmup_requested: int = 0
    settlement_requested: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.index == self.n_chunks - 1

    @property
    def payload_slice(self) -> slice:
        """Positional `[start, stop)` of the payload inside `frame`."""
        return slice(self.warmup_len, self.warmup_len + self.payload_len)

    @property
    def payload_frame(self) -> pd.DataFrame:
        return self.frame.iloc[self.payload_slice]

    @property
    def payload_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame["ts"].iloc[self.warmup_len])

    @property
    def payload_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(
            self.frame["ts"].iloc[self.warmup_len + self.payload_len - 1])

    @property
    def warmup_short(self) -> bool:
        """
        True when the warm-up is shorter than asked for.

        Always true for the FIRST chunk, which has no history behind it — that
        one is not a defect, it is the same cold start the contiguous run has.
        True for any LATER chunk it is a real shortfall and means an indicator
        in the payload may still be carrying its seed.
        """
        return self.warmup_len < self.warmup_requested

    def payload_mask(self, n: int | None = None) -> np.ndarray:
        """A boolean mask over `frame`'s rows, True on the payload."""
        n = len(self.frame) if n is None else int(n)
        mask = np.zeros(n, dtype=bool)
        mask[self.payload_slice] = True
        return mask

    def describe(self) -> str:
        """One line for the console, naming what is real and what is warm-up."""
        short = " SHORT WARM-UP" if (self.warmup_short and not self.is_first) \
            else ""
        return (f"chunk {self.index + 1}/{self.n_chunks} "
                f"{self.payload_start_ts:%Y-%m-%d} → "
                f"{self.payload_end_ts:%Y-%m-%d}  "
                f"{self.payload_len:,} bars "
                f"(+{self.warmup_len:,} warm-up, "
                f"+{self.settlement_len:,} settlement){short}")


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LakeSource:
    """
    A (symbol, timeframe) read through `mdlib.lake`, one chunk's years at a
    time.

    This is the source that actually reduces the LOAD peak: `iter_bars` prunes
    on the year partition, so a 2-year chunk touches two years of parquet
    instead of sixteen. The hygiene flags are carried verbatim so a chunked
    read and a contiguous one see the same bars — a chunked sweep run with
    `exclude_rolls` off against a contiguous one with it on would differ for a
    reason that has nothing to do with chunking.
    """

    symbol: str
    tf: str = "15m"
    session_merge: bool = True
    exclude_degraded: bool = False
    exclude_rolls: bool = False
    respect_coverage: bool = False
    regimes: bool = True


class _Source:
    """Read a date range and report the full span. One interface, three backings."""

    tf_hint: str | None = None

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        raise NotImplementedError

    def read(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError


def _project(df: pd.DataFrame, columns: Sequence[str] | None) -> pd.DataFrame:
    """
    Keep the requested columns that exist, in the requested order.

    A missing column is SKIPPED rather than raising: `regime_quadrant` is
    absent whenever the regime cache misses, which `mdlib.regimes` documents as
    a legitimate outcome rather than an error, and a projection that raised
    would make an uncached (symbol, tf) pair unscannable.

    `ts` is forced into the projection whatever the caller asked for — every
    boundary in this module is a timestamp, so a frame without it cannot be
    chunked at all.
    """
    if columns is None:
        return df
    keep = [c for c in columns if c in df.columns]
    if "ts" in df.columns and "ts" not in keep:
        keep = ["ts"] + keep
    return df[keep] if keep else df


class _FrameSource(_Source):
    """
    An already-materialised frame.

    Slicing it is zero-copy under copy-on-write, so this costs nothing per
    chunk — and saves nothing on the load side either, because the caller is
    already holding the whole thing. It is here so a caller can chunk a frame
    it has (a fixture, a test, a frame from another stage) with the same code
    path as a lake read.
    """

    def __init__(self, frame: pd.DataFrame, columns: Sequence[str] | None):
        if "ts" not in frame.columns:
            raise ChunkingError(
                "a frame source needs a `ts` column; the engine's long format "
                "carries one and a time-INDEXED frame must be reset first, "
                "because every consumer downstream of here indexes by position")
        self._frame = _project(frame, columns)
        ts = pd.DatetimeIndex(pd.to_datetime(self._frame["ts"], utc=True))
        if not ts.is_monotonic_increasing:
            raise ChunkingError(
                "a frame source must be sorted oldest-first; an unsorted frame "
                "would put a chunk boundary in the middle of the data")
        self._ts = ts

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        if len(self._ts) == 0:
            raise ChunkingError("the frame is empty — nothing to chunk")
        return self._ts[0], self._ts[-1]

    def probe_ts(self) -> pd.DatetimeIndex:
        """The whole index, free — it is already in memory."""
        return self._ts

    def bar_minutes(self) -> float:
        return _infer_bar_minutes(self._ts)

    def read(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        lo = int(self._ts.searchsorted(start, side="left"))
        hi = int(self._ts.searchsorted(end, side="right"))
        # `.iloc` on a range, not a boolean mask: a mask is a fancy-index and
        # copies, which would defeat the whole point on a 5.6M-row frame.
        return self._frame.iloc[lo:hi]


class _ParquetSource(_Source):
    """
    A parquet file, or a directory of them, read one date range at a time.

    Uses a pyarrow dataset with a `ts` filter so only the row groups a chunk
    needs are materialised. This is the source to use for a lake-shaped
    directory that is not the lake itself.
    """

    def __init__(self, path: str | Path, columns: Sequence[str] | None):
        import pyarrow.dataset as ds

        self._path = Path(path)
        if not self._path.exists():
            raise ChunkingError(f"{self._path} does not exist")
        files = ([str(self._path)] if self._path.is_file()
                 else sorted(str(p) for p in self._path.rglob("*.parquet")))
        if not files:
            raise ChunkingError(f"no parquet files under {self._path}")
        self._ds = ds.dataset(files, format="parquet")
        self._columns = columns
        self._ts_all: pd.DatetimeIndex | None = None
        self._field = ds.field
        names = set(self._ds.schema.names)
        if "ts" not in names:
            raise ChunkingError(f"{self._path} has no `ts` column")
        self._keep = ([c for c in columns if c in names]
                      if columns is not None else None)
        if self._keep is not None and "ts" not in self._keep:
            self._keep = ["ts"] + self._keep

    def probe_ts(self) -> pd.DatetimeIndex:
        """
        The whole `ts` index, as ONE column scan, cached.

        Eight bytes a row against the ~60 the projected bars cost, so this is
        the cheap way to know exactly which rows every chunk needs — and it is
        what lets the parquet path place its boundaries exactly rather than by
        widening a date window. Sorted here rather than assumed: the boundary
        is a `searchsorted` and an unsorted index would place it anywhere.
        """
        if self._ts_all is None:
            ts = self._ds.to_table(columns=["ts"]).column("ts").to_pandas()
            ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
            if len(ts) == 0:
                raise ChunkingError(f"{self._path} holds no rows")
            self._ts_all = ts if ts.is_monotonic_increasing else ts.sort_values()
        return self._ts_all

    def bar_minutes(self) -> float:
        return _infer_bar_minutes(self.probe_ts())

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        ts = self.probe_ts()
        return ts[0], ts[-1]

    def read(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        expr = (self._field("ts") >= pd.Timestamp(start)) & \
               (self._field("ts") <= pd.Timestamp(end))
        table = self._ds.to_table(columns=self._keep, filter=expr)
        df = table.to_pandas()
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.sort_values("ts").reset_index(drop=True)


class _LakeSource(_Source):
    """One contract, through `mdlib.lake.iter_bars`, a date range at a time."""

    def __init__(self, spec: LakeSource, columns: Sequence[str] | None,
                 start: str | pd.Timestamp | None,
                 end: str | pd.Timestamp | None):
        self._spec = spec
        self._columns = columns
        self._start = start
        self._end = end
        self.tf_hint = spec.tf
        self._span: tuple[pd.Timestamp, pd.Timestamp] | None = None

    @staticmethod
    def _naive(value):
        """
        A bound `mdlib.lake.iter_bars` will accept.

        It does `pd.Timestamp(start, tz="UTC")` internally, which RAISES on an
        already-aware timestamp — "Cannot pass a datetime or Timestamp with
        tzinfo with the tz parameter". Everything in this module is tz-aware
        UTC by construction, so the conversion has to happen at the boundary
        rather than being left to a caller who would have no reason to expect
        it.
        """
        if value is None:
            return None
        ts = pd.Timestamp(value)
        return ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None \
            else ts

    def _read_raw(self, start, end) -> pd.DataFrame:
        from mdlib.lake import iter_bars

        start, end = self._naive(start), self._naive(end)
        spec = self._spec
        for sym, frame in iter_bars([spec.symbol], spec.tf, start, end,
                                    session_merge=spec.session_merge,
                                    exclude_degraded=spec.exclude_degraded,
                                    exclude_rolls=spec.exclude_rolls,
                                    respect_coverage=spec.respect_coverage,
                                    regimes=spec.regimes):
            if sym == spec.symbol and len(frame):
                return frame.reset_index(drop=True)
        return pd.DataFrame()

    def _partition_years(self) -> list[int]:
        """
        The years the lake HOLDS for this contract, from the partition
        directory names alone.

        A directory listing, not a read: it discovers which years exist and
        never decides which ROWS exist, so it cannot disagree with
        `mdlib.lake` about the Sunday merge or the hygiene flags — those drop
        rows inside a year and leave the year itself in place. That
        distinction is what makes this safe where a second parquet READER
        would not be.

        A derived timeframe is built from its native one, so the partition to
        list is the native's. Returns `[]` when the lake is not mounted or the
        contract is absent, and the caller falls back to a full read.
        """
        try:
            from mdlib.lake import DERIVED, LAKE
        except Exception:                                       # noqa: BLE001
            return []
        native = DERIVED.get(self._spec.tf, (self._spec.tf, None))[0]
        base = LAKE / f"symbol={self._spec.symbol}" / f"tf={native}"
        if not base.exists():
            return []
        years = sorted(int(d.name.split("=")[1]) for d in base.iterdir()
                       if d.is_dir() and d.name.startswith("year="))
        if self._start is not None:
            years = [y for y in years if y >= pd.Timestamp(self._start).year]
        if self._end is not None:
            years = [y for y in years if y <= pd.Timestamp(self._end).year]
        return years

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        The first and last bar, read from the EDGE years only.

        The naive way to answer this is a full-range read, which would pull the
        sixteen years this whole module exists to avoid holding at once — the
        span would be free and the chunking pointless. Listing the year
        partitions costs a directory scan, and reading the first and last of
        them costs two years of bars rather than sixteen.

        A year that is present as a directory and empty after the hygiene flags
        have run is stepped over from each end, so an `exclude_rolls` run whose
        first year is entirely roll days still gets a true first bar.
        """
        if self._span is not None:
            return self._span

        years = self._partition_years()
        first = last = None
        for y in years:
            edge = self._read_raw(f"{y}-01-01", f"{y}-12-31 23:59:59")
            if len(edge):
                first = pd.Timestamp(edge["ts"].iloc[0])
                break
        for y in reversed(years):
            edge = self._read_raw(f"{y}-01-01", f"{y}-12-31 23:59:59")
            if len(edge):
                last = pd.Timestamp(edge["ts"].iloc[-1])
                break
        edge = None
        gc.collect()

        if first is None or last is None:
            # No partitions visible — an unmounted lake, or a caller that
            # narrowed the window past every year directory. Fall back to the
            # honest expensive answer rather than reporting no data.
            frame = self._read_raw(self._start, self._end)
            if frame.empty:
                raise ChunkingError(
                    f"the lake returned no {self._spec.tf} bars for "
                    f"{self._spec.symbol} over "
                    f"{self._start or 'the start of history'} → "
                    f"{self._end or 'the end'}")
            ts = pd.DatetimeIndex(pd.to_datetime(frame["ts"], utc=True))
            first, last = ts[0], ts[-1]
            del frame, ts
            gc.collect()

        # An explicit --start/--end narrows the span; it never widens it.
        if self._start is not None:
            first = max(first, pd.Timestamp(self._start, tz="UTC")
                        if pd.Timestamp(self._start).tz is None
                        else pd.Timestamp(self._start))
        if self._end is not None:
            last = min(last, pd.Timestamp(self._end, tz="UTC")
                       if pd.Timestamp(self._end).tz is None
                       else pd.Timestamp(self._end))
        self._span = (first, last)
        return self._span

    def read(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # CLAMPED AT BOTH ENDS to the span, which already carries the caller's
        # --start/--end. Clamping only the lower bound was a real bug: a final
        # calendar span reaching past the requested end read the year beyond it
        # and put those bars in the payload, so `--end 2022-12-31` silently
        # swept 2023 — the Stage 3 holdout — and nothing on the table said so.
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        if self._span:
            lo = max(lo, self._span[0])
            hi = min(hi, self._span[1])
        if hi < lo:
            return pd.DataFrame()

        # THE UPPER BOUND IS WIDENED BY ONE BAR, AND ONLY A DERIVED TIMEFRAME
        # EXPLAINS WHY. 5m/15m/30m/1h/2h/4h are RESAMPLED from the 1-minute
        # tree by `mdlib.lake`, and a resample labels each bucket at its START.
        # So a read that stops at the last bar's LABEL hands the resampler only
        # the first minute of that bucket, and the bar comes back with the high,
        # low, close and volume of one minute instead of fifteen. Measured
        # against a contiguous read of NQ 15m over 2013-2022: exactly one bar
        # differed, the last, in every column except `open` — which is `first`
        # and therefore the one field a truncated bucket still gets right. That
        # is the signature of this bug and it is nearly invisible: one bar in
        # 221,685, at the end of the window, where a strategy's last trade
        # closes.
        #
        # Widening by one bar covers the bucket. Anything extra it pulls in is
        # removed positionally by the payload boundary, so over-reading here
        # cannot leak a bar into the results.
        #
        # THE PAD STOPS DEAD AT THE CALLER'S `end`, AND THAT MATTERS MORE THAN
        # THE COMPLETE BUCKET. Stage 2 is REFUSED a window reaching
        # `HOLDOUT_START` before it loads a bar (`scan.check_in_sample_window`)
        # because whatever it reads it has fitted to. Padding past `--end
        # 2022-12-31` would read fifteen minutes of 2023 to finish a bucket —
        # a trivial amount of the holdout, spent silently, by a memory
        # optimisation. So the final bucket at the wall stays truncated, which
        # is ALSO exactly what `backtest.run.load_bars` produces for the same
        # window: a chunked sweep and a contiguous one then see the same last
        # bar, which is worth more here than either of them seeing a complete
        # one. `tests/test_temporal_chunking.py` pins the equality against a
        # real contiguous read.
        pad = _TF_MINUTES.get(self._spec.tf)
        if pad:
            hi = hi + pd.Timedelta(minutes=float(pad))
            wall = self._naive_to_utc(self._end)
            if wall is not None:
                hi = min(hi, wall)
        return _project(self._read_raw(lo, hi), self._columns)

    @staticmethod
    def _naive_to_utc(value):
        """The caller's `--end` as a tz-aware UTC bound, or None."""
        if value is None:
            return None
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _resolve_source(source: Any,
                    columns: Sequence[str] | None,
                    symbol: str | None,
                    tf: str | None,
                    start,
                    end) -> _Source:
    """
    `df_or_path` in the request's signature, widened to the three shapes a
    caller actually has: a frame, a path, or a lake spec.

    A bare `(symbol, tf)` tuple and a `symbol` string with `tf=` given are both
    read as a lake spec, because that is the only reading that could be meant —
    a string that is not an existing path cannot be a parquet source.
    """
    if isinstance(source, pd.DataFrame):
        return _FrameSource(source, columns)
    if isinstance(source, LakeSource):
        return _LakeSource(source, columns, start, end)
    if isinstance(source, tuple) and len(source) == 2:
        return _LakeSource(LakeSource(symbol=str(source[0]),
                                      tf=str(source[1])),
                           columns, start, end)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            return _ParquetSource(path, columns)
        if isinstance(source, Path) or path.suffix == ".parquet" or "/" in str(
                source):
            # Unmistakably meant as a path. Reporting it as an unrecognised
            # lake spec would send a reader looking for a symbol called
            # `/mnt/.../bars.parquet`.
            raise ChunkingError(f"{path} does not exist")
        if tf:
            return _LakeSource(LakeSource(symbol=str(source), tf=str(tf)),
                               columns, start, end)
        raise ChunkingError(
            f"{source!r} is neither an existing path nor a lake spec. Pass a "
            f"tf= to read it as a symbol, or a path that exists.")
    if symbol and tf:
        return _LakeSource(LakeSource(symbol=symbol, tf=tf), columns,
                           start, end)
    raise ChunkingError(
        f"cannot read a source of type {type(source).__name__}; pass a "
        f"DataFrame, a parquet path, a LakeSource, or (symbol, tf)")


def _infer_bar_minutes(ts: pd.DatetimeIndex) -> float:
    """
    The frame's own bar spacing, in minutes, as the MEDIAN gap.

    The median rather than the mean: a futures week has a weekend gap of ~2,600
    minutes in it, and a handful of those drag a mean far enough that a warm-up
    window sized from it would over-read by a factor of several. The median is
    the spacing of a normal bar, which is what a bar COUNT has to be converted
    through.
    """
    if len(ts) < 3:
        return 1.0
    diffs = np.diff(ts.asi8) / 60_000_000_000.0
    med = float(np.median(diffs))
    return med if med > 0 else 1.0


def _lookback_window(bars: int, bar_minutes: float) -> pd.Timedelta:
    """
    A date window generous enough to contain `bars` bars of history.

    `LOOKBACK_SLACK` is the margin for session gaps, holidays and thin
    overnight hours. Over-reading costs one read; under-reading costs a
    silently short warm-up, which is an indicator still carrying its seed into
    a payload nobody would think to check. `iter_temporal_chunks` widens and
    re-reads rather than accepting the short window, so the slack is a starting
    point rather than a guarantee.
    """
    if bars <= 0:
        return pd.Timedelta(0)
    return pd.Timedelta(minutes=float(bars) * float(bar_minutes)
                        * LOOKBACK_SLACK)


# --------------------------------------------------------------------------
# Two ways to place a boundary, and why both exist
# --------------------------------------------------------------------------
# A chunk's frame is `warm-up + payload + settlement`, and the warm-up is a
# count of BARS while a reader is addressed in DATES. Turning one into the
# other can be done exactly or approximately, and which is available depends on
# what the source can be asked cheaply:
#
#   EXACT      A source that can hand over its whole `ts` index without
#              materialising the bars — an in-memory frame, or a parquet
#              dataset where reading one column is a column scan. The boundary
#              is then a `searchsorted`, the read is for exactly the rows
#              wanted, and there is no slack anywhere.
#
#   WINDOWED   `mdlib.lake` has no metadata-only entry point, and the obvious
#              substitute — walking the year partitions with a second parquet
#              reader — would be a second reader free to disagree with the
#              first about the Sunday merge and the hygiene flags, both of
#              which DROP ROWS. Disagreeing about which rows exist is
#              disagreeing about where the boundary is. So the lake path asks
#              for a generous DATE window, counts what came back, and widens
#              and re-reads if the warm-up landed short.
#
# The windowed path is the one that can be wrong, so it is the one that checks
# its own work: a chunk whose warm-up came back short of what was asked for
# carries `warmup_short`, and `backtest.scan` prints it.
def _probe_ts(src: _Source) -> pd.DatetimeIndex | None:
    """The source's whole `ts` index, or None when it cannot be had cheaply."""
    probe = getattr(src, "probe_ts", None)
    return probe() if callable(probe) else None


def _to_utc(value) -> pd.Timestamp | None:
    """A bound as a tz-aware UTC timestamp; a naive one is READ as UTC."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _clamp_span(span: tuple[pd.Timestamp, pd.Timestamp], start, end
                ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    The source's span narrowed by the caller's `--start`/`--end`.

    Applied for EVERY source rather than only the lake one. Before this the
    bounds reached `_LakeSource` alone, so `iter_temporal_chunks(frame,
    start=..., end=...)` accepted the arguments and ignored them - the least
    visible kind of wrong, because the chunks came back looking entirely
    normal and simply covered more history than was asked for.
    """
    first, last = span
    lo, hi = _to_utc(start), _to_utc(end)
    if lo is not None:
        first = max(first, lo)
    if hi is not None:
        last = min(last, hi)
    if last < first:
        raise ChunkingError(
            f"the requested window {start} → {end} leaves no bars: the source "
            f"spans {span[0]} → {span[1]}")
    return first, last


def iter_temporal_chunks(df_or_path: Any,
                         chunk_years: int = DEFAULT_CHUNK_YEARS,
                         warmup_bars: int | str = DEFAULT_WARMUP_BARS,
                         settlement_bars: int = 0,
                         columns: Sequence[str] | None = DEFAULT_COLUMNS,
                         start: Any = None,
                         end: Any = None,
                         symbol: str | None = None,
                         tf: str | None = None,
                         min_tail_fraction: float = MIN_TAIL_FRACTION,
                         guard: MemoryGuard | None = None,
                         ) -> Iterator[TemporalChunk]:
    """
    Yield one contract's bars as chronological blocks of `chunk_years` calendar
    years, each carrying `warmup_bars` of history from the block before it.

    `df_or_path` is the request's name for the source and is any of:

        pd.DataFrame            already in memory. Sliced zero-copy; saves
                                nothing on the load side, because the caller is
                                already holding it.
        str | Path              a parquet file or a directory of them. Read one
                                date range at a time.
        LakeSource(symbol, tf)  read through `mdlib.lake` with its year-
                                partition pruning. THIS is the shape that
                                reduces the load peak.
        (symbol, tf)            the same, spelled as a tuple.

    Chunks are `[start, end)` on the calendar, anchored on January 1 so two
    contracts get the same boundaries — see `calendar_spans`. A span with no
    bars in it (a contract that had not listed yet, a gap in the lake) is
    SKIPPED and does not become an empty chunk, and `n_chunks` on every yielded
    chunk counts what will actually be yielded rather than what was planned.

    `warmup_bars` accepts the string `"auto"`, which sizes the window from the
    slowest recursive indicator this module knows how to bound — see
    `auto_warmup_bars`, and read its caveats before trusting it.

    EVERY CHUNK BOUNDARY IS GUARDED. `guard` defaults to
    `backtest.memory_guard.DEFAULT_GUARD` and is enforced immediately before
    each chunk is handed out, which is the right moment for two reasons: the
    frame for THIS chunk has just been read and is the largest single thing
    this function holds, and the consumer is about to allocate several times
    that on top of it — `scan_symbol_chunked` stacks four boolean masks over
    every grid cell. Checking after the read and before the consumer's
    allocation is the last point where collecting still helps and halting is
    still clean.

    A halt raises `MemorySafetyException` out of the generator, which
    propagates through the consumer's `for` loop. Pass `guard=MemoryGuard(
    enabled=False)` for a caller that must not be interrupted, or set
    `BT_MEMORY_GUARD=off`.

    THE CALLER IS RESPONSIBLE FOR USING ONLY THE PAYLOAD. Every chunk carries
    `payload_slice`, `payload_frame` and `payload_mask` for that, and the
    warm-up rows are in `frame` so indicators can be computed across them.
    Keeping a result attributed to a warm-up row double-counts it against the
    previous chunk; `backtest.scan` attributes trades by ENTRY timestamp for
    exactly this reason.
    """
    if isinstance(warmup_bars, str):
        if warmup_bars.lower() != "auto":
            raise ChunkingError(
                f"warmup_bars must be an int or 'auto'; got {warmup_bars!r}")
        warmup_bars = auto_warmup_bars()
    warmup_bars = int(warmup_bars)
    settlement_bars = int(settlement_bars)
    if warmup_bars < 0 or settlement_bars < 0:
        raise ChunkingError(
            f"warmup_bars and settlement_bars must be >= 0; got "
            f"{warmup_bars} and {settlement_bars}")

    src = _resolve_source(df_or_path, columns, symbol, tf, start, end)
    first, last = _clamp_span(src.span(), start, end)
    spans = calendar_spans(first, last, chunk_years, min_tail_fraction)
    ts_all = _probe_ts(src)

    plans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    exact: list[tuple[int, int]] = []
    # THE REQUESTED WINDOW IS A HARD WALL AT BOTH ENDS, and the near end
    # matters as much as the far one. A warm-up that reached back before
    # `start` would give the first chunk history the CONTIGUOUS run does not
    # have — `backtest/run.py::load_bars` reads from `start` and no earlier —
    # so chunk 1's payload would not match the contiguous sweep's head, and
    # the chunked run would look better for a reason that is not the strategy.
    bound_lo, bound_hi = 0, 0
    if ts_all is not None:
        bound_lo = int(ts_all.searchsorted(first, side="left"))
        bound_hi = int(ts_all.searchsorted(last, side="right"))
        for lo_ts, hi_ts in spans:
            lo = max(bound_lo, int(ts_all.searchsorted(lo_ts, side="left")))
            hi = min(bound_hi, int(ts_all.searchsorted(hi_ts, side="left")))
            if hi > lo:                    # a span with no bars is not a chunk
                plans.append((lo_ts, hi_ts))
                exact.append((lo, hi))
    else:
        plans = list(spans)

    n_planned = len(plans)
    if n_planned == 0:
        return

    bar_minutes = (_infer_bar_minutes(ts_all) if ts_all is not None
                   else float(_TF_MINUTES.get(getattr(src, "tf_hint", None)
                                              or "", 1.0)))

    emitted = 0
    for i, (span_start, span_end) in enumerate(plans):
        if ts_all is not None:
            lo, hi = exact[i]
            f_lo = max(bound_lo, lo - warmup_bars)
            f_hi = min(bound_hi, hi + settlement_bars)
            raw = src.read(ts_all[f_lo], ts_all[f_hi - 1])
            # The read is inclusive at both ends and the source may hold
            # duplicate timestamps, so the boundary is recomputed inside what
            # actually came back rather than assumed from the plan.
            got = pd.DatetimeIndex(pd.to_datetime(raw["ts"], utc=True))
            w_lo = int(got.searchsorted(span_start, side="left"))
            w_hi = int(got.searchsorted(span_end, side="left"))
            frame_raw, warm_lo, pay_hi = raw, w_lo, w_hi
        else:
            frame_raw, warm_lo, pay_hi = _windowed_read(
                src, span_start, span_end, warmup_bars, settlement_bars,
                bar_minutes, first, last)
            if frame_raw is None:
                continue

        payload_len = pay_hi - warm_lo
        if payload_len <= 0:
            continue

        keep_lo = max(0, warm_lo - warmup_bars)
        keep_hi = min(len(frame_raw), pay_hi + settlement_bars)
        # `.iloc` over a contiguous range keeps the copy-on-write view; the
        # index is then replaced rather than the data copied, because every
        # consumer downstream indexes by position.
        frame = frame_raw.iloc[keep_lo:keep_hi]
        frame = frame.set_axis(pd.RangeIndex(len(frame)), axis=0)

        # The chunk boundary. See the note in this function's docstring for why
        # here rather than at the top of the loop: the read has happened, the
        # consumer's allocation has not.
        (guard or DEFAULT_GUARD).enforce(
            f"data_loader.iter_temporal_chunks[{symbol or 'frame'} "
            f"chunk {emitted + 1}/{n_planned}]")

        yield TemporalChunk(
            frame=frame,
            index=emitted,
            n_chunks=n_planned,
            warmup_len=warm_lo - keep_lo,
            payload_len=payload_len,
            settlement_len=keep_hi - pay_hi,
            span_start=span_start,
            span_end=span_end,
            warmup_requested=warmup_bars,
            settlement_requested=settlement_bars,
            meta={"bar_minutes": bar_minutes,
                  "boundary": "exact" if ts_all is not None else "windowed"},
        )
        emitted += 1
        del frame, frame_raw
        gc.collect()


def _windowed_read(src: _Source,
                   span_start: pd.Timestamp,
                   span_end: pd.Timestamp,
                   warmup_bars: int,
                   settlement_bars: int,
                   bar_minutes: float,
                   first: pd.Timestamp,
                   last: pd.Timestamp):
    """
    Read one chunk from a source whose row positions are not known in advance,
    widening the date window until the warm-up and the settlement tail are
    long enough — or until the window has reached the ends of the data, where
    short is the honest answer.

    Returns `(frame, payload_lo, payload_hi)` positionally inside that frame,
    or `(None, 0, 0)` when the span holds no bars at all.

    The widening is bounded at `_MAX_WIDENINGS`. Beyond that the short read is
    RETURNED rather than retried forever: a chunk whose warm-up is short is
    visible on `TemporalChunk.warmup_short` and printed by `backtest.scan`,
    which is a better outcome than a loader that spins on a thin contract.
    """
    back = _lookback_window(warmup_bars, bar_minutes)
    fwd = _lookback_window(settlement_bars, bar_minutes)

    for attempt in range(_MAX_WIDENINGS):
        scale = 2 ** attempt
        lo_ts = span_start - back * scale
        hi_ts = span_end + fwd * scale if settlement_bars else span_end
        raw = src.read(lo_ts, hi_ts)
        if raw is None or len(raw) == 0:
            if lo_ts <= first and hi_ts >= last:
                return None, 0, 0
            continue
        got = pd.DatetimeIndex(pd.to_datetime(raw["ts"], utc=True))
        p_lo = max(int(got.searchsorted(span_start, side="left")),
                   int(got.searchsorted(first, side="left")))
        # Clamped to `last` as well as to the calendar edge. The read above is
        # widened by a bar to complete a derived timeframe's final bucket, and
        # without this clamp that extra bar would land in the payload — one bar
        # past the window the operator asked for, on the far side of a holdout
        # boundary.
        p_hi = min(int(got.searchsorted(span_end, side="left")),
                   int(got.searchsorted(last, side="right")))
        if p_hi <= p_lo:
            return None, 0, 0
        have_warm = p_lo
        have_settle = len(got) - p_hi
        reached_start = lo_ts <= first
        reached_end = hi_ts >= last
        if ((have_warm >= warmup_bars or reached_start)
                and (have_settle >= settlement_bars or reached_end)):
            return raw, p_lo, p_hi
        del raw, got
        gc.collect()

    raw = src.read(span_start - back * (2 ** (_MAX_WIDENINGS - 1)),
                   span_end + fwd * (2 ** (_MAX_WIDENINGS - 1)))
    if raw is None or len(raw) == 0:
        return None, 0, 0
    got = pd.DatetimeIndex(pd.to_datetime(raw["ts"], utc=True))
    p_lo = max(int(got.searchsorted(span_start, side="left")),
               int(got.searchsorted(first, side="left")))
    p_hi = min(int(got.searchsorted(span_end, side="left")),
               int(got.searchsorted(last, side="right")))
    return (raw, p_lo, p_hi) if p_hi > p_lo else (None, 0, 0)


# Four doublings takes a 1-minute warm-up window from 4x nominal to 64x, which
# covers a contract that trades a few hours a week. Past that the shortfall is
# a property of the data rather than of the window.
_MAX_WIDENINGS = 4


def chunk_plan(df_or_path: Any,
               chunk_years: int = DEFAULT_CHUNK_YEARS,
               warmup_bars: int = DEFAULT_WARMUP_BARS,
               settlement_bars: int = 0,
               columns: Sequence[str] | None = DEFAULT_COLUMNS,
               start: Any = None,
               end: Any = None,
               symbol: str | None = None,
               tf: str | None = None) -> list[dict]:
    """
    What `iter_temporal_chunks` WOULD yield, without yielding any bars.

    One row per chunk: the payload span, its bar count, and the warm-up and
    settlement it would carry. For a source that can hand over its `ts` index
    this costs no bar reads at all, which makes it the right thing to print in
    a banner before a sweep starts — an operator can see the eight blocks and
    their sizes before committing to the run.

    For a source that cannot (the lake), this falls back to iterating the
    chunks for real, so it is NOT free there and says so in the row's
    `boundary` field.
    """
    src = _resolve_source(df_or_path, columns, symbol, tf, start, end)
    first, last = src.span()
    spans = calendar_spans(first, last, chunk_years)
    ts_all = _probe_ts(src)
    if ts_all is None:
        return [{"index": c.index, "n_chunks": c.n_chunks,
                 "start": c.payload_start_ts, "end": c.payload_end_ts,
                 "bars": c.payload_len, "warmup": c.warmup_len,
                 "settlement": c.settlement_len, "boundary": "windowed"}
                for c in iter_temporal_chunks(
                    df_or_path, chunk_years, warmup_bars, settlement_bars,
                    columns, start, end, symbol, tf)]

    rows: list[dict] = []
    kept = [(lo_ts, hi_ts,
             int(ts_all.searchsorted(lo_ts, side="left")),
             int(ts_all.searchsorted(hi_ts, side="left")))
            for lo_ts, hi_ts in spans]
    kept = [k for k in kept if k[3] > k[2]]
    for i, (lo_ts, _hi_ts, lo, hi) in enumerate(kept):
        rows.append({
            "index": i,
            "n_chunks": len(kept),
            "start": ts_all[lo],
            "end": ts_all[hi - 1],
            "bars": hi - lo,
            "warmup": lo - max(0, lo - warmup_bars),
            "settlement": min(len(ts_all), hi + settlement_bars) - hi,
            "boundary": "exact",
        })
    return rows
