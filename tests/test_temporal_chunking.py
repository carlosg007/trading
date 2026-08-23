#!/usr/bin/env python3
"""
test_temporal_chunking.py — the universe-wide temporal chunker: that its
payloads partition the history exactly, that a chunked indicator matches a
contiguous one to 1e-6, that the peak it exists to remove is actually removed,
and that the trades a chunk boundary cuts are counted rather than lost.

Location:  ~/src/trading/tests/test_temporal_chunking.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_temporal_chunking.py
    OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
        tests/test_temporal_chunking.py

EVERY CASE FAILS THROUGH `assert`, DELIBERATELY — see the note in
`tests/test_double_rsi_macd_scalp_20260823.py`. The older convention in this
directory (a `check(name, ok)` helper and `sys.exit(1)` in `main`) is invisible
to pytest, which collects those suites, watches their checks fail and reports
all green.

Nothing here needs the lake or a network. The one case that would
(`test_the_lake_source_reads_only_the_edge_years`) SKIPS LOUDLY when
`/mnt/backtest` is not mounted.

WHAT THIS COVERS, and why each one is here rather than assumed:

  * **THE PAYLOADS PARTITION THE HISTORY.** Every bar belongs to exactly one
    chunk's payload — no gaps, no overlaps, and the union of the spans is the
    full span. A gap silently removes bars from the sweep; an overlap counts
    the trades in it twice, and both leave a table that looks complete.
  * **A CHUNKED INDICATOR MATCHES A CONTIGUOUS ONE.** Checked on a windowed
    average, a span EMA and Wilder's ATR and RSI, at the warm-up
    `required_warmup_bars` says is needed for 1e-6 — and separately checked to
    FAIL at the specification's default of 500 bars for a 200-period EMA,
    because that shortfall is the single most likely way this feature produces
    quietly wrong numbers. `double_rsi_macd_scalp_20260823` and
    `ema_trend_filter` both carry a 200-bar trend EMA.
  * **THE PEAK IS ACTUALLY REMOVED.** Both as the arithmetic
    (`projected_sweep_bytes`, which is what decides whether a sweep OOMs) and
    as a measured RSS ceiling around a real chunked evaluation. The measured
    half is the weaker of the two — `ru_maxrss` is a process high-water mark
    that never falls — so it is asserted as a DELTA and with a ceiling loose
    enough to be about the chunker rather than about the interpreter.
  * **BOUNDARY TRADES ARE COUNTED, NOT LOST.** With a settlement tail long
    enough, the chunked sweep reproduces the contiguous one exactly — same
    trade counts, same Sharpe, same profit factor, in every column. With NO
    tail, trades are dropped at the boundaries, and the case requires the
    reported `truncated_trades` to equal the count that went missing, exactly.
    That equality is the whole safety property: CLAUDE.md's objection to
    calendar chunking is that "a position open on 31 December is silently
    dropped", and silence is the part this removes.
  * **ZERO-COPY AND COLUMN FILTERING**, asserted through `np.shares_memory`
    rather than trusted — copy-on-write makes the claim plausible and a
    fancy-index anywhere in the slicing path would quietly break it on a 5.6M
    row frame.
"""

from __future__ import annotations

import ast
import gc
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.data_loader import (ChunkingError,                 # noqa: E402
                                  DEFAULT_COLUMNS, LakeSource,
                                  auto_warmup_bars, calendar_spans,
                                  chunk_plan, iter_temporal_chunks,
                                  peak_rss_bytes, projected_sweep_bytes,
                                  required_warmup_bars, SEED_ERROR_SCALE,
                                  suggest_chunk_years)
from backtest.engine import BacktestConfig                        # noqa: E402
from backtest.scan import (expand_grid, scan_symbol,              # noqa: E402
                           scan_symbol_chunked, session_days,
                           warn_if_sweep_will_not_fit)

LOADER_PATH = REPO / "backtest" / "data_loader.py"
# A real temp directory, not a folder inside `tests/`. The probe strategy and
# the parquet fixture are build artifacts of a test run and have no business
# appearing in `git status` next to the repository's own files.
SCRATCH = Path(tempfile.mkdtemp(prefix="temporal_chunking_"))

# The probe strategy the sweep cases run. A crossover with a declared grid and
# a combination it REFUSES (`fast >= slow`), so the chunked path's handling of
# rejected cells is exercised rather than assumed.
STRATEGY_SRC = '''
"""A crossover with a declared grid, for the temporal-chunking test."""
import pandas as pd

TIMEFRAME = "1d"
SYMBOLS = ["ES"]
DEFAULT_PARAMS = {"fast": 5, "slow": 20}
PARAM_GRID = {"fast": [3, 5, 10], "slow": [10, 20, 40]}


def signal_fn(bars, fast=5, slow=20):
    if fast >= slow:
        raise ValueError(f"fast must be < slow; got {fast} >= {slow}")
    c = bars["close"]
    f = c.rolling(fast, min_periods=fast).mean()
    s = c.rolling(slow, min_periods=slow).mean()
    above = f > s
    was = above.shift(1).fillna(False).astype(bool)
    return ((above & ~was).fillna(False).astype(bool),
            (~above & was).fillna(False).astype(bool))


def make_signal_fn(fast=5, slow=20):
    def _bound(bars):
        return signal_fn(bars, fast=fast, slow=slow)
    return _bound
'''


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def daily_bars(years: int = 12, seed: int = 7,
               start: str = "2013-01-02") -> pd.DataFrame:
    """
    A trending, noisy daily path with real intrabar ranges, in the ENGINE's own
    shape — `ts` as a COLUMN with a positional index.

    Trending on purpose: a pure random walk gives a crossover strategy almost
    no trades, and a chunking test where every column produces two trades
    cannot distinguish "the chunks agree" from "the chunks are all empty".
    """
    n = years * 365
    rng = np.random.default_rng(seed)
    close = (4000.0 + rng.normal(0.4, 12.0, n).cumsum()
             + 60.0 * np.sin(np.arange(n) / 45.0))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(6.0, 3.0, n))
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=n, freq="D", tz="UTC"),
        "symbol": "ES",
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": rng.integers(10_000, 100_000, n).astype(float),
    })


def intraday_bars(years: int = 6, seed: int = 3) -> pd.DataFrame:
    """A 15-minute frame, for the cases that care about bar density."""
    n = years * 365 * 24 * 4
    rng = np.random.default_rng(seed)
    close = 1000.0 + rng.normal(0.0, 0.4, n).cumsum()
    return pd.DataFrame({
        "ts": pd.date_range("2013-03-01", periods=n, freq="15min", tz="UTC"),
        "symbol": "NQ",
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0),
        "regime_quadrant": np.ones(n, dtype=np.uint8),
    })


def probe_strategy() -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "chunk_probe.py"
    path.write_text(STRATEGY_SRC, encoding="utf-8")
    return path


def raises(fn, *args, **kwargs) -> str:
    """Run `fn` and return the ChunkingError message; assert if it did not raise."""
    try:
        fn(*args, **kwargs)
    except ChunkingError as exc:
        return str(exc)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise")


# ==========================================================================
# 1. Calendar spans — the boundaries, before any bars are involved
# ==========================================================================
def test_the_spans_are_half_open_and_anchored_on_january() -> None:
    """
    Anchoring on a GLOBAL grid of `chunk_years`-year blocks — not on the first
    bar's own year — is what makes two contracts with different histories
    comparable. NQ has history from 2013 and MNQ from 2019; on a first-bar
    anchor the same flags would cut them differently and their chunked sweeps
    would not line up, and a data pull that moved a first bar by a year would
    silently re-cut every block of that contract.

    Half-open throughout. A closed end would put the bar stamped exactly on the
    boundary into two chunks, and the trade attribution in `scan_symbol_chunked`
    would count it twice.
    """
    first = pd.Timestamp("2013-05-17", tz="UTC")
    last = pd.Timestamp("2022-11-02", tz="UTC")
    spans = calendar_spans(first, last, chunk_years=2)
    assert spans[0][0] <= first, spans[0]
    assert spans[0][0].month == 1 and spans[0][0].day == 1, spans[0]
    assert spans[0][0].year % 2 == 0, (
        f"the first edge {spans[0][0]} is not on the 2-year global grid")
    for (a_lo, a_hi), (b_lo, _b_hi) in zip(spans, spans[1:]):
        assert a_hi == b_lo, f"a gap or an overlap between {a_hi} and {b_lo}"
        assert a_lo < a_hi
        assert (a_hi.year - a_lo.year) == 2, (a_lo, a_hi)
    assert spans[-1][1] > last, "the last span must cover the last bar"

    # A contract whose history starts six years later gets the SAME edges
    # wherever the two overlap — which is the property the global anchor buys.
    other = calendar_spans(pd.Timestamp("2019-02-02", tz="UTC"), last, 2)
    shared = {s[0] for s in spans} & {s[0] for s in other}
    assert shared == {s[0] for s in other}, (
        f"the later contract's edges {sorted(s[0] for s in other)} are not a "
        f"subset of the earlier one's {sorted(s[0] for s in spans)}")


def test_a_stub_tail_is_absorbed_rather_than_becoming_its_own_chunk() -> None:
    """
    A final block covering three weeks would pay a full warm-up to sweep almost
    nothing — and, worse, its payload would be too short for any trade to close
    inside, so every trade opened in it would count as truncated. It is merged
    into the block before it.
    """
    # 2012-01-01 is the global-grid anchor for a 2013 start. A last bar in
    # early March leaves a two-month tail of a twenty-four-month block.
    spans = calendar_spans(pd.Timestamp("2013-01-01", tz="UTC"),
                           pd.Timestamp("2018-03-01", tz="UTC"), 2)
    assert len(spans) == 3, spans
    assert spans[-1] == (pd.Timestamp("2016-01-01", tz="UTC"),
                         pd.Timestamp("2020-01-01", tz="UTC")), spans[-1]
    # A tail past the halfway mark keeps its own chunk — eighteen months of
    # twenty-four is worth its own warm-up.
    kept = calendar_spans(pd.Timestamp("2013-01-01", tz="UTC"),
                          pd.Timestamp("2019-06-30", tz="UTC"), 2)
    assert len(kept) == 4, kept
    assert kept[-1][0] == pd.Timestamp("2018-01-01", tz="UTC"), kept[-1]


def test_a_naive_timestamp_is_refused_rather_than_assumed_to_be_utc() -> None:
    """
    The lake stores UTC and pandas 3 RAISES on comparing a naive bound against
    an aware index rather than silently reading it as UTC. Refusing here means
    the failure names the cause; letting it through means it surfaces several
    frames deeper as a comparison error inside a searchsorted.
    """
    msg = raises(calendar_spans, pd.Timestamp("2013-01-01"),
                 pd.Timestamp("2016-01-01"), 2)
    assert "tz-aware" in msg, msg
    assert "before" in raises(calendar_spans,
                              pd.Timestamp("2016-01-01", tz="UTC"),
                              pd.Timestamp("2013-01-01", tz="UTC"), 2)
    assert ">= 1" in raises(calendar_spans,
                            pd.Timestamp("2013-01-01", tz="UTC"),
                            pd.Timestamp("2016-01-01", tz="UTC"), 0)


# ==========================================================================
# 2. Chunk continuity — the request's first verification clause
# ==========================================================================
def test_the_payloads_partition_the_dataset_exactly() -> None:
    """
    THE REQUEST'S "union of date ranges across all chunks equals the full
    dataset span without gaps", asserted as the stronger property it needs to
    be: every bar belongs to exactly ONE payload.

    Checked on the timestamps rather than on the counts. Equal counts would
    pass with two chunks that overlap by a day and skip a different one — and
    that is the failure that matters, because an overlap double-counts the
    trades inside it and a gap removes them, and both leave a table that looks
    complete.
    """
    for freq_bars, years, chunk_years in ((daily_bars(12), 12, 2),
                                          (daily_bars(9), 9, 3),
                                          (intraday_bars(6), 6, 2)):
        bars = freq_bars
        seen: list[np.ndarray] = []
        spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for chunk in iter_temporal_chunks(bars, chunk_years=chunk_years,
                                          warmup_bars=200):
            payload = chunk.payload_frame
            seen.append(pd.DatetimeIndex(payload["ts"]).asi8)
            spans.append((chunk.payload_start_ts, chunk.payload_end_ts))
            assert len(payload) == chunk.payload_len

        covered = np.concatenate(seen)
        want = pd.DatetimeIndex(bars["ts"]).asi8
        assert covered.size == want.size, (
            f"{covered.size} payload bars against {want.size} in the frame — "
            f"the payloads do not cover the data")
        assert np.array_equal(np.sort(covered), np.sort(want)), (
            "the payloads are not the same set of bars as the frame")
        assert np.array_equal(covered, want), (
            "the payloads cover the right bars in the wrong order")
        assert len(np.unique(covered)) == covered.size, (
            "a bar appears in two payloads — its trades would be counted twice")
        for (_a_lo, a_hi), (b_lo, _b_hi) in zip(spans, spans[1:]):
            assert a_hi < b_lo, f"payload {a_hi} overlaps {b_lo}"


def test_the_chunk_count_and_plan_agree_with_what_is_yielded() -> None:
    """
    `chunk_plan` is what a banner prints BEFORE a sweep commits to a run, and
    `n_chunks` is what a progress line counts against. A plan that disagreed
    with the iteration would be a banner describing a different run from the
    one that followed.
    """
    bars = daily_bars(12)
    plan = chunk_plan(bars, chunk_years=2, warmup_bars=300)
    chunks = list(iter_temporal_chunks(bars, chunk_years=2, warmup_bars=300))
    assert len(plan) == len(chunks) == plan[0]["n_chunks"]
    for row, chunk in zip(plan, chunks):
        assert row["index"] == chunk.index
        assert row["bars"] == chunk.payload_len, (row, chunk.describe())
        assert row["warmup"] == chunk.warmup_len
        assert pd.Timestamp(row["start"]) == chunk.payload_start_ts
        assert pd.Timestamp(row["end"]) == chunk.payload_end_ts
    assert all(c.n_chunks == len(chunks) for c in chunks)
    assert chunks[0].is_first and chunks[-1].is_last
    assert not chunks[0].is_last


def test_a_span_with_no_bars_is_skipped_rather_than_yielded_empty() -> None:
    """
    A contract that had not listed yet, or a gap in the lake, leaves a calendar
    span with nothing in it. An empty chunk would reach the sweep as a frame of
    zero bars — where `_combo_signals` and `_entry_block_mask` both have to
    handle a degenerate case for no reason — and would make `n_chunks` count
    blocks that produced nothing.
    """
    early = daily_bars(2, start="2013-01-02")
    late = daily_bars(2, seed=9, start="2021-01-02")
    gapped = pd.concat([early, late], ignore_index=True)
    chunks = list(iter_temporal_chunks(gapped, chunk_years=2, warmup_bars=50))
    assert all(c.payload_len > 0 for c in chunks)
    years = {c.payload_start_ts.year for c in chunks}
    assert 2017 not in years and 2018 not in years, years
    assert sum(c.payload_len for c in chunks) == len(gapped)


# ==========================================================================
# 3. Warm-up integrity — the request's second verification clause
# ==========================================================================
def _wilder(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()


def _indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """
    One windowed indicator and three recursive ones, computed the way the
    strategy modules in this repository compute them.

    The mix is the point: a windowed average is EXACT once its window is full
    and would pass any warm-up test, while the recursive three converge and
    never quite arrive. A case built only on the first would pass with no
    warm-up at all.
    """
    close = bars["close"].astype(float)
    high, low = bars["high"].astype(float), bars["low"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), 14)
    loss = _wilder((-delta).clip(lower=0.0), 14)
    total = gain + loss
    return pd.DataFrame({
        "sma_50": close.rolling(50, min_periods=50).mean(),
        "ema_50": close.ewm(span=50, adjust=False, min_periods=50).mean(),
        "atr_14": _wilder(tr, 14),
        "rsi_14": 100.0 * gain / total.where(total > 0),
    })


def _chunked_indicators(bars: pd.DataFrame, chunk_years: int,
                        warmup_bars: int) -> pd.DataFrame:
    """
    The indicators computed CHUNK BY CHUNK — over `warm-up + payload`, then
    keeping the payload rows only, which is exactly the contract
    `iter_temporal_chunks` documents for its consumers.
    """
    parts = []
    for chunk in iter_temporal_chunks(bars, chunk_years=chunk_years,
                                      warmup_bars=warmup_bars):
        full = _indicators(chunk.frame)
        parts.append(full.iloc[chunk.payload_slice])
    return pd.concat(parts, ignore_index=True)


def test_a_chunked_indicator_matches_the_contiguous_one_to_1e_6() -> None:
    """
    THE REQUEST'S SECOND VERIFICATION CLAUSE, at the warm-up the arithmetic
    says is needed rather than at a number that looked generous.

    `required_warmup_bars` is asked for the slowest recursion in the fixture,
    AT THE FIXTURE'S PRICE SCALE. That second half is the part that is easy to
    get wrong and this case exists to pin: what decays is the SEED ERROR, and
    `ewm` restarts at the chunk's first value, so on a ~4,000-point series the
    error to shed is hundreds of points rather than order one. Sized without
    the scale, a span EMA(50) gets 346 bars and still lands 9.4e-05 out — a
    hundred times the tolerance. With it, 553 bars, and every payload row of
    every indicator matches the contiguous computation within 1e-6 ABSOLUTE.
    """
    bars = daily_bars(12)
    scale = float(bars["close"].abs().max())
    warmup = max(required_warmup_bars(50, "span", seed_error=scale),
                 required_warmup_bars(14, "wilder", seed_error=scale),
                 required_warmup_bars(50, "sma"))
    assert warmup > required_warmup_bars(50, "span"), (
        "the price scale did not lengthen the warm-up — the seed-error term "
        "is not doing anything")

    want = _indicators(bars)
    got = _chunked_indicators(bars, chunk_years=2, warmup_bars=warmup)
    assert len(got) == len(want) == len(bars)

    for col in want.columns:
        a, b = want[col].to_numpy(), got[col].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > len(bars) * 0.8, (
            f"{col}: only {both.sum()} comparable rows")
        worst = float(np.max(np.abs(a[both] - b[both])))
        assert worst < 1e-6, f"{col} diverged by {worst:.3e} at {warmup} bars"
        # And the NaN warm-up pattern is identical, so a chunk is not quietly
        # producing a value where the contiguous run has none.
        assert np.array_equal(np.isfinite(a), np.isfinite(b)), (
            f"{col}: the chunked run has values where the contiguous one does "
            f"not, or the reverse")


def test_the_specified_500_bar_default_is_not_enough_for_a_200_ema() -> None:
    """
    THE FINDING THAT MOTIVATES `auto`, PINNED SO IT CANNOT BE FORGOTTEN.

    A recursive average never forgets its seed exactly, only exponentially:
    after `w` bars it still carries `(1 - alpha)^w` of it. At the specified
    default of 500 bars a span EMA(200) retains 0.67% — four thousand times the
    1e-6 tolerance the clause above is checked at — and
    `double_rsi_macd_scalp_20260823` and `ema_trend_filter` both carry a
    200-bar trend EMA whose crossings are entry and exit conditions.

    So this case asserts the FAILURE, deliberately. If a future change makes it
    pass, the arithmetic in `required_warmup_bars` has become wrong or the
    fixture has stopped exercising the indicator, and both are worth a red
    test.
    """
    assert required_warmup_bars(200, "span") == 1382
    assert required_warmup_bars(200, "wilder") == 2757
    assert required_warmup_bars(14, "wilder") == 187
    # The seed-error term only ever LENGTHENS a warm-up, and a tolerance wider
    # than the error being shed asks for none at all.
    assert required_warmup_bars(200, "span", seed_error=4000.0) > 1382
    assert required_warmup_bars(200, "span", tolerance=0.5,
                                seed_error=0.1) == 1
    # A windowed statistic is exact once full and does not depend on tolerance.
    assert required_warmup_bars(50, "sma") == 51
    assert (required_warmup_bars(50, "sma", tolerance=1e-12)
            == required_warmup_bars(50, "sma", tolerance=1e-2))

    bars = daily_bars(12)
    close = bars["close"].astype(float)

    def ema200(frame):
        return frame["close"].astype(float).ewm(
            span=200, adjust=False, min_periods=200).mean()

    want = ema200(bars).to_numpy()

    def worst_at(warmup: int) -> float:
        parts = []
        for chunk in iter_temporal_chunks(bars, chunk_years=2,
                                          warmup_bars=warmup):
            parts.append(ema200(chunk.frame).iloc[chunk.payload_slice])
        got = pd.concat(parts, ignore_index=True).to_numpy()
        both = np.isfinite(want) & np.isfinite(got)
        return float(np.max(np.abs(want[both] - got[both])))

    short = worst_at(500)
    assert short > 1e-6, (
        f"a 200-period EMA at a 500-bar warm-up came within {short:.3e} — the "
        f"fixture is no longer exercising the seed decay this case exists to "
        f"show")
    long = worst_at(required_warmup_bars(200, "span"))
    assert long < short, (long, short)
    print(f"        200-EMA worst error: {short:.3e} at 500 bars, "
          f"{long:.3e} at {required_warmup_bars(200, 'span')}")


def test_the_auto_warmup_reads_the_strategys_own_parameters() -> None:
    """
    `auto` is a heuristic keyed on VALUES, not on meaning — it takes the
    largest positive whole number among the bound parameters and the grid and
    sizes a Wilder recursion of that length, Wilder being the slowest of the
    three forms so the answer covers a span EMA too.

    Its floor is the specification's 500, so the answer is never SHORTER than
    the documented default, only longer. Booleans are excluded explicitly:
    `True` is an int in Python and a strategy with `trailing=True` would
    otherwise size its warm-up for a 1-period indicator.
    """
    assert auto_warmup_bars() == 500
    assert auto_warmup_bars(params={"fast": 5, "slow": 20}) == 500
    big = auto_warmup_bars(grid={"trend_period": [50, 200]})
    assert big == required_warmup_bars(200, "wilder",
                                       seed_error=SEED_ERROR_SCALE), big
    assert big > required_warmup_bars(200, "wilder"), (
        "auto must size against the price-scale seed error, not the relative "
        "one — see required_warmup_bars")
    assert auto_warmup_bars(params={"trailing": True, "use_x": True}) == 500
    assert auto_warmup_bars(params={"tp_atr_mult": None,
                                    "sl_atr_mult": 1.5}) == 500
    assert "positive period" in raises(required_warmup_bars, 0)
    assert "tolerance" in raises(required_warmup_bars, 20, "span", 0.0)
    assert "kind" in raises(required_warmup_bars, 20, "triangular")


# ==========================================================================
# 4. Memory — the request's third verification clause
# ==========================================================================
def test_the_projected_peak_falls_with_the_chunk_count() -> None:
    """
    The arithmetic that decides whether a sweep OOMs, and the reason chunking
    is the fix rather than a smaller `--max-cells`: the four signal masks are
    `4 x n_bars x n_columns` bytes and are allocated IN FULL before the first
    vectorbt call, so column batching inside that call cannot touch them.
    """
    bars_16y_1m, combos = 5_600_000, 432
    contiguous = projected_sweep_bytes(bars_16y_1m, combos)
    assert contiguous / 2 ** 30 > 8.0, contiguous
    eight_chunks = projected_sweep_bytes(bars_16y_1m // 8, combos)
    assert eight_chunks * 8 == contiguous
    assert eight_chunks / 2 ** 30 < 1.2, eight_chunks

    # The suggestion is derived from the CALENDAR span, not the bar count: a
    # year is 252 bars at 1d and ~350,000 at 1m, so a suggestion computed from
    # the count alone is wrong by three orders of magnitude on the timeframe
    # that needs it most.
    years = suggest_chunk_years(bars_16y_1m, combos, span_years=16.0,
                                budget_bytes=int(8 * 2 ** 30))
    assert years is not None and 1 <= years <= 16, years
    assert projected_sweep_bytes(int(bars_16y_1m * years / 16.0),
                                 combos) <= 8 * 2 ** 30
    # A sweep that already fits is told nothing.
    assert suggest_chunk_years(200_000, 9, 6.0, int(8 * 2 ** 30)) is None

    warned = warn_if_sweep_will_not_fit(bars_16y_1m, combos, 8.0, "NQ", "1m",
                                        span_years=16.0)
    assert warned and "--chunk-years" in warned
    assert warn_if_sweep_will_not_fit(1000, 9, 8.0, "ES", "1d",
                                      span_years=4.0) is None


def test_the_chunked_evaluation_stays_under_a_memory_ceiling() -> None:
    """
    THE REQUEST'S THIRD VERIFICATION CLAUSE, measured rather than projected.

    The measurement is the weak half of this case and the ceiling is set
    accordingly: `ru_maxrss` is a HIGH-WATER MARK for the whole process and
    never falls, so what is asserted is the DELTA across the chunked loop and
    the ceiling is loose enough to be about the chunker rather than about
    whatever the interpreter had already touched. What makes it meaningful is
    the CONTRAST — the same work done contiguously allocates a mask block the
    chunked loop never does, and the projections beside it are exact.
    """
    bars = intraday_bars(6)                      # ~210k 15-minute bars
    n_cols = 24

    before = peak_rss_bytes()
    largest = 0
    for chunk in iter_temporal_chunks(bars, chunk_years=2, warmup_bars=500):
        # A stand-in for the sweep's own allocation: four boolean masks over
        # the chunk, which is exactly the shape `scan_symbol` builds.
        masks = [np.zeros((len(chunk.frame), n_cols), dtype=bool)
                 for _ in range(4)]
        largest = max(largest, len(chunk.frame))
        del masks
        gc.collect()
    delta = peak_rss_bytes() - before

    ceiling = projected_sweep_bytes(largest, n_cols) * 3 + 256 * 2 ** 20
    assert delta < ceiling, (
        f"the chunked evaluation added {delta / 2 ** 20:.0f} MiB of peak RSS "
        f"against a {ceiling / 2 ** 20:.0f} MiB ceiling")
    assert largest < len(bars), (
        "no chunk may be the whole frame, or nothing was chunked")
    assert projected_sweep_bytes(largest, n_cols) < \
        projected_sweep_bytes(len(bars), n_cols) / 2
    print(f"        chunked peak RSS delta {delta / 2 ** 20:.0f} MiB, "
          f"largest chunk {largest:,} bars of {len(bars):,}")


def test_the_slicing_is_zero_copy_and_the_columns_are_filtered() -> None:
    """
    Copy-on-write makes the zero-copy claim plausible; `np.shares_memory` is
    what makes it checked. A fancy-index anywhere in the slicing path — a
    boolean mask instead of a range, a `.loc` with a list — would silently
    start copying, and on a 5.6M-row frame that is the peak this module exists
    to remove, reintroduced.

    The projection is the other half: a lake frame carries columns a sweep
    never reads, and each one is copied into every chunk for nothing. A column
    that is ABSENT from the source is skipped rather than raising —
    `regime_quadrant` is missing whenever the regime cache misses, which
    `mdlib.regimes` documents as a legitimate outcome.
    """
    bars = intraday_bars(4)
    bars["unused_column"] = bars["close"] * 3.0
    parent = bars["close"].to_numpy()

    for chunk in iter_temporal_chunks(bars, chunk_years=2, warmup_bars=100):
        child = chunk.frame["close"].to_numpy()
        assert np.shares_memory(parent, child), (
            "the chunk copied its price data — the slicing path is not "
            "zero-copy any more")
        assert "unused_column" not in chunk.frame.columns
        assert "regime_quadrant" in chunk.frame.columns
        assert list(chunk.frame.columns) == [c for c in DEFAULT_COLUMNS
                                             if c in bars.columns]
        assert isinstance(chunk.frame.index, pd.RangeIndex), (
            "the chunk kept the parent's labels; every consumer downstream "
            "indexes by position")
        assert chunk.frame.index[0] == 0

    # A frame with no regime column still chunks, with the column simply absent.
    plain = bars.drop(columns=["regime_quadrant", "unused_column"])
    one = next(iter_temporal_chunks(plain, chunk_years=2, warmup_bars=100))
    assert "regime_quadrant" not in one.frame.columns
    # And a caller can ask for its own projection.
    narrow = next(iter_temporal_chunks(bars, chunk_years=2, warmup_bars=100,
                                       columns=("ts", "close")))
    assert list(narrow.frame.columns) == ["ts", "close"]


# ==========================================================================
# 5. The sweep across chunks — boundaries, attribution, and what is lost
# ==========================================================================
def _sweep_pair(bars: pd.DataFrame, settlement: int, warmup: int = 600):
    """The same grid swept contiguously and in 2-year chunks."""
    path = probe_strategy()
    cfg = BacktestConfig(variants_tested=None)
    grid = {"fast": [3, 5, 10], "slow": [10, 20, 40]}
    contiguous = scan_symbol(path, bars, "ES", cfg, grid,
                             strat_name="chunk_probe")
    chunked = scan_symbol_chunked(path, bars, "ES", cfg, grid,
                                  strat_name="chunk_probe", chunk_years=2,
                                  warmup_bars=warmup,
                                  settlement_bars=settlement, progress=False)
    return contiguous, chunked


def test_a_chunked_sweep_with_a_settlement_tail_reproduces_the_contiguous_one() -> None:
    """
    The property the whole design rests on: with a warm-up long enough for the
    indicators and a settlement tail longer than the holding period, sweeping
    in chunks and sweeping contiguously produce THE SAME ANSWER — the same
    trade count, Sharpe and profit factor in every column, and the same winner.

    This is stronger than the module promises (it documents the chunked sweep
    as an approximation) and it is asserted anyway, because the approximation
    is meant to be tight where the tail covers the trades. A divergence here
    with a 600-bar tail on a strategy whose trades last days means the
    attribution is wrong, not that the approximation is loose.
    """
    bars = daily_bars(12)
    contiguous, chunked = _sweep_pair(bars, settlement=600)

    assert chunked["chunking"]["truncated_trades"] == 0, (
        chunked["chunking"]["truncated_trades"])
    assert chunked["evaluated"] == contiguous["evaluated"]
    assert chunked["combinations"] == contiguous["combinations"]
    assert len(chunked["rejected"]) == len(contiguous["rejected"]) == 1, (
        "the grid's fast >= slow cell must still be recorded as REJECTED")

    a = contiguous["table"].set_index("params")
    b = chunked["table"].set_index("params")
    assert set(a.index) == set(b.index)
    for col in ("trades", "sharpe", "profit_factor", "max_drawdown_pct",
                "total_return_pct"):
        # Both reindexed onto ONE order. The two tables are each sorted by
        # their own Sharpe, so comparing them positionally would line up
        # different parameter sets and pass or fail for the wrong reason.
        left = a[col].sort_index()
        right = b[col].reindex(left.index)
        assert np.allclose(left.to_numpy(dtype=float),
                           right.to_numpy(dtype=float),
                           rtol=0, atol=1e-9, equal_nan=True), (
            f"{col} diverged between the chunked and contiguous sweeps:\n"
            f"{pd.DataFrame({'contiguous': left, 'chunked': right})}")
    assert (chunked["winner"]["params"] == contiguous["winner"]["params"])
    assert chunked["selection"] == contiguous["selection"]


def test_boundary_trades_are_counted_exactly_when_they_are_lost() -> None:
    """
    THE SAFETY PROPERTY, and the reason this module is allowed to do the thing
    CLAUDE.md forbids.

    Calendar chunking drops a position that is open when a chunk ends —
    "silently dropped, which flatters results". With the settlement tail
    removed entirely, that is exactly what happens here. What must NOT happen
    is the silence: `chunking.truncated_trades` has to equal the number of
    trades that went missing against the contiguous sweep, EXACTLY, so an
    operator reading the record knows the size of what is not in the numbers.

    Anything less than exact equality would be worse than useless — a count
    that under-reports is a licence to trust a table that lost more than it
    admits.
    """
    bars = daily_bars(12)
    contiguous, chunked = _sweep_pair(bars, settlement=0)

    a = contiguous["table"].set_index("params")["trades"]
    b = chunked["table"].set_index("params")["trades"].reindex(a.index)
    missing = int((a - b).sum())
    assert missing > 0, (
        "no trade was lost with the settlement tail removed — the fixture is "
        "not exercising a boundary at all")
    assert chunked["chunking"]["truncated_trades"] == missing, (
        f"{chunked['chunking']['truncated_trades']} trades reported as "
        f"truncated against {missing} actually missing")
    assert chunked["chunking"]["truncated_columns"] == int((a - b).gt(0).sum())
    # Never the other way: a chunk may lose a trade, never invent one.
    assert (b <= a).all(), "the chunked sweep produced trades the contiguous "\
                           "sweep did not"
    print(f"        settlement=0 dropped {missing} trade(s), all counted")


def test_every_trade_is_attributed_to_exactly_one_chunk() -> None:
    """
    Attribution is by ENTRY timestamp into one chunk's payload, which is what
    makes concatenating the chunks a partition rather than a pile. An overlap
    would double-count the trades in it — inflating the trade count, the
    profit factor's denominator and the Sharpe's sample all at once, and
    leaving a table that looks entirely normal.
    """
    bars = daily_bars(12)
    _contiguous, chunked = _sweep_pair(bars, settlement=600)
    spans = [(pd.Timestamp(c["start"]), pd.Timestamp(c["end"]))
             for c in chunked["chunking"]["chunks"]]
    for (_a_lo, a_hi), (b_lo, _b_hi) in zip(spans, spans[1:]):
        assert a_hi < b_lo, f"chunk payloads overlap at {a_hi} / {b_lo}"
    assert sum(c["bars"] for c in chunked["chunking"]["chunks"]) == len(bars)
    assert chunked["chunking"]["bars_total"] == len(bars)


def test_the_chunking_record_travels_with_the_result() -> None:
    """
    A chunked sweep must never be readable as a contiguous one. The `chunking`
    key is how a table, a `best_params_<SYMBOL>_<TF>.json` and eventually Stage
    3 can tell that the winner was selected on an approximation, and how good
    an approximation it was.

    A contiguous sweep carries no such key at all — `None` rather than a record
    of zero chunks, because "not chunked" and "chunked into one block" are
    different runs.
    """
    bars = daily_bars(12)
    contiguous, chunked = _sweep_pair(bars, settlement=600)
    assert "chunking" not in contiguous, sorted(contiguous)
    record = chunked["chunking"]
    for key in ("chunk_years", "warmup_bars", "settlement_bars", "chunks",
                "bars_total", "truncated_trades", "truncated_columns",
                "projected_bytes_contiguous", "projected_bytes_chunked",
                "peak_rss_bytes", "warmup_short_chunks", "equivalence"):
        assert key in record, f"the chunking record has no {key!r}"
    assert "APPROXIMATE" in record["equivalence"]
    assert record["projected_bytes_chunked"] < \
        record["projected_bytes_contiguous"]
    assert record["chunk_years"] == 2
    assert all(set(row) >= {"index", "start", "end", "bars", "warmup",
                            "settlement", "warmup_short", "truncated_trades"}
               for row in record["chunks"])


def test_a_scan_extra_cannot_overwrite_the_ranking() -> None:
    """
    `_finalise_scan` merges the chunked path's metadata into the result and
    REFUSES a key the ranking itself produced. A caller that could replace
    `winner` or `table` would put a parameter set into
    `best_params_<SYMBOL>_<TF>.json` that the selection rule did not choose,
    and every consumer downstream — the CSV, the handoff, Stage 3's locked
    parameters — would trust it.
    """
    from backtest.scan import _finalise_scan, ScanError

    days = session_days(daily_bars(1))
    try:
        _finalise_scan(symbol="ES", cfg=BacktestConfig(), grid={"fast": [3]},
                       rank="plateau", strat_name="p",
                       valid=[{"fast": 3}], rejected=[], n_combinations=1,
                       trade_lists=[pd.DataFrame()], days=days,
                       filter_info=None, offered_total=0, suppressed_total=0,
                       extra={"winner": {"params": {"fast": 99}}})
    except ScanError as exc:
        assert "overwrite" in str(exc), str(exc)
    else:
        raise AssertionError("a scan extra overwrote the ranking's own winner")


# ==========================================================================
# 6. Pandas 3 compatibility and the source shapes
# ==========================================================================
def test_the_loader_carries_no_deprecated_pandas_construct() -> None:
    """
    The request's fourth implementation clause. Checked at the source rather
    than by watching for a warning, because a `FutureWarning` raised inside a
    sweep of 432 columns is printed once and scrolls past.

    `.fillna(method=...)` was removed in pandas 3, positional datetime slicing
    with `.ix`-style labels went with it, and `inplace=` on a chained
    assignment is the copy-on-write trap that silently does nothing.
    """
    tree = ast.parse(LOADER_PATH.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else getattr(func, "id", ""))
        kwargs = {kw.arg for kw in node.keywords}
        assert not (name == "fillna" and "method" in kwargs), (
            f"fillna(method=...) at line {node.lineno} was removed in "
            f"pandas 3")
        # `iteritems` and `get_values` are unambiguous — nothing else in the
        # standard library spells them. `DataFrame.append` is NOT checked here
        # even though it is also gone: at the AST level `x.append(y)` is
        # indistinguishable from a list append, and this module uses several,
        # so the check would be a permanent false positive rather than a net.
        assert not (name in ("iteritems", "get_values")
                    and isinstance(func, ast.Attribute)), (
            f"{name}() at line {node.lineno} is gone from pandas 3")
        assert "inplace" not in kwargs, (
            f"inplace= at line {node.lineno} is a no-op under copy-on-write "
            f"often enough to be a bug")


def test_a_time_indexed_frame_is_refused_with_a_reason() -> None:
    """
    The engine's long format carries `ts` as a COLUMN, and every consumer
    downstream of the chunker indexes by POSITION. A time-indexed frame would
    slice correctly here and then misalign the moment a mask built on
    `RangeIndex` met a frame labelled by timestamp — so it is refused at the
    door, with the fix in the message.
    """
    bars = daily_bars(3)
    indexed = bars.drop(columns=["ts"]).set_index(
        pd.DatetimeIndex(bars["ts"]))
    msg = raises(lambda: list(iter_temporal_chunks(indexed, chunk_years=2)))
    assert "`ts` column" in msg, msg

    shuffled = bars.sample(frac=1.0, random_state=0)
    msg = raises(lambda: list(iter_temporal_chunks(shuffled, chunk_years=2)))
    assert "oldest-first" in msg, msg

    assert "int or 'auto'" in raises(
        lambda: list(iter_temporal_chunks(bars, warmup_bars="lots")))
    assert ">= 0" in raises(
        lambda: list(iter_temporal_chunks(bars, warmup_bars=-1)))
    assert "DataFrame" in raises(
        lambda: list(iter_temporal_chunks(12345, chunk_years=2)))


def test_a_parquet_path_is_read_one_chunk_at_a_time() -> None:
    """
    The source shape that actually reduces the LOAD peak for a non-lake
    dataset: a parquet dataset, read with a `ts` filter so only the row groups
    a chunk needs are materialised.

    Held to the same partition property as the in-memory path, and to the same
    values — a reader that returned different bars from the same file would be
    a second lake.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    bars = daily_bars(8)
    path = SCRATCH / "bars.parquet"
    bars.to_parquet(path, index=False)

    seen = []
    for chunk in iter_temporal_chunks(path, chunk_years=2, warmup_bars=100):
        assert chunk.meta["boundary"] == "exact", chunk.meta
        seen.append(chunk.payload_frame[["ts", "close"]])
    got = pd.concat(seen, ignore_index=True)
    want = bars[["ts", "close"]].reset_index(drop=True)
    assert len(got) == len(want)
    assert np.array_equal(pd.DatetimeIndex(got["ts"]).asi8,
                          pd.DatetimeIndex(want["ts"]).asi8)
    assert np.allclose(got["close"].to_numpy(), want["close"].to_numpy())

    assert "does not exist" in raises(
        lambda: list(iter_temporal_chunks(SCRATCH / "nope.parquet")))


def test_the_lake_source_reads_only_the_edge_years() -> None:
    """
    `LakeSource` is the shape that matters for `--symbols ALL`: the span comes
    from the year PARTITION names plus a read of the first and last year, not
    from a full-range read, so discovering where a contract's history starts
    does not pull the sixteen years this module exists not to hold.

    SKIPS LOUDLY without the lake. It reads one contract at the DAILY
    timeframe, which is a few thousand rows — this suite does not run a
    backtest and does not touch the 1-minute tree.
    """
    lake = Path("/mnt/backtest/lake/futures/bars")
    if not lake.exists():
        print("        SKIPPED: /mnt/backtest is not mounted")
        return
    symbols = [d.name.split("=")[1] for d in lake.iterdir()
               if d.is_dir() and d.name.startswith("symbol=")]
    candidates = [s for s in ("ES", "NQ", "CL", "GC") if s in symbols]
    if not candidates:
        print(f"        SKIPPED: none of ES/NQ/CL/GC in the lake ({symbols[:5]})")
        return

    src = LakeSource(symbol=candidates[0], tf="1d")
    plan = chunk_plan(src, chunk_years=4, warmup_bars=50)
    assert plan, f"no chunks planned for {candidates[0]}"
    assert all(row["bars"] > 0 for row in plan), plan
    spans = [(pd.Timestamp(r["start"]), pd.Timestamp(r["end"])) for r in plan]
    for (_a_lo, a_hi), (b_lo, _b_hi) in zip(spans, spans[1:]):
        assert a_hi < b_lo, f"lake payloads overlap at {a_hi} / {b_lo}"
    print(f"        {candidates[0]} 1d → {len(plan)} chunk(s), "
          f"{sum(r['bars'] for r in plan):,} bars, "
          f"{spans[0][0]:%Y-%m-%d} → {spans[-1][1]:%Y-%m-%d}")


def test_a_derived_timeframe_chunk_is_not_truncated_at_the_wall() -> None:
    """
    THE REGRESSION FOR THE ONE BUG THIS FEATURE ALMOST SHIPPED WITH, and it was
    found by comparing against a contiguous read of real bars rather than by
    any amount of reasoning about the code.

    5m/15m/30m/1h/2h/4h are RESAMPLED from the 1-minute tree by `mdlib.lake`,
    and a resample labels each bucket at its START. A chunk read that stopped
    at the last bar's LABEL therefore handed the resampler the first minute of
    that bucket and nothing else, and the bar came back carrying the high, low,
    close and volume of ONE minute instead of fifteen. Measured on NQ 15m over
    2013-2022 it was exactly one bar in 221,685 — the last one, in every column
    except `open`, which is `first` and is the one field a truncated bucket
    still gets right.

    That is as close to invisible as a data bug gets: it lands at the end of
    the window, which is where a strategy's last trade closes and where a
    holdout begins. `_LakeSource.read` widens its upper bound by one bar to
    cover the bucket, and `_windowed_read` clamps the payload back so the extra
    bar cannot leak past the requested window.

    SKIPS LOUDLY without the lake — the truncation lives in the resample, so
    only a real derived-timeframe read can exercise it.
    """
    lake = Path("/mnt/backtest/lake/futures/bars")
    if not lake.exists():
        print("        SKIPPED: /mnt/backtest is not mounted")
        return
    symbols = [d.name.split("=")[1] for d in lake.iterdir()
               if d.is_dir() and d.name.startswith("symbol=")]
    sym = next((s for s in ("ES", "NQ", "CL", "GC") if s in symbols), None)
    if sym is None:
        print(f"        SKIPPED: none of ES/NQ/CL/GC in the lake")
        return

    from backtest.run import load_bars

    start, end, tf = "2018-01-01", "2019-12-31", "15m"
    contiguous = load_bars(sym, tf, start, end)
    # settlement_bars=0 on purpose: with a tail every chunk but the last reads
    # well past its payload and the truncation only bites at the far wall.
    # With no tail EVERY chunk's final payload bar sits on a wall, which is the
    # configuration that would have shown the bug on every boundary.
    parts = [c.payload_frame for c in iter_temporal_chunks(
        LakeSource(symbol=sym, tf=tf), chunk_years=1, warmup_bars=500,
        settlement_bars=0, start=start, end=end)]
    assert len(parts) >= 2, f"only {len(parts)} chunk(s) — no boundary to test"
    chunked = pd.concat(parts, ignore_index=True)

    assert len(chunked) == len(contiguous), (
        f"{len(chunked)} chunked payload bars against {len(contiguous)} "
        f"contiguous")
    assert np.array_equal(pd.DatetimeIndex(chunked["ts"]).asi8,
                          pd.DatetimeIndex(contiguous["ts"]).asi8), (
        "the chunked payloads are not the contiguous read's bars, in order")
    for col in ("open", "high", "low", "close", "volume"):
        bad = int((~np.isclose(contiguous[col].to_numpy(dtype=float),
                               chunked[col].to_numpy(dtype=float))).sum())
        assert bad == 0, (
            f"{bad} {col} value(s) differ from the contiguous read — a "
            f"resampled bucket was cut off at a chunk boundary")
    print(f"        {sym} {tf} {start}..{end}: {len(chunked):,} bars, "
          f"{len(parts)} chunks, bit-identical to the contiguous read")



# ==========================================================================
# The script runner. `assert` is the failure mechanism, so pytest and this
# report the same thing — see the module docstring.
# ==========================================================================
def main() -> int:
    cases = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    print(f"temporal chunking — {len(cases)} cases\n")
    for name, fn in cases:
        try:
            fn()
        except Exception as e:                   # noqa: BLE001 - reported below
            failures.append((name, e))
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
        else:
            print(f"  PASS  {name}")
    print()
    if failures:
        print(f"{len(failures)} of {len(cases)} FAILED: "
              + ", ".join(n for n, _ in failures))
        return 1
    print(f"all {len(cases)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
