#!/usr/bin/env python3
"""
The regime feature cache, against hand-checkable answers.

Run as a script; it exits non-zero on failure. No lake and no network.

What each check is defending:
  1. The quadrant integer and the two boolean columns are ONE statement. If
     they can disagree, a consumer reading `regime_quadrant` and one reading
     `is_high_vol` describe different bars and neither raises.
  2. Warm-up bars are 0, not 4. `NaN > 25.0` is False, so the naive encoding
     files every warm-up bar under Low-Vol/Ranging - a populated column of a
     regime that was never measured.
  3. theta_vol is the in-sample median and nothing else. Widening the window
     changes the boundary, so the OOS years must not move it.
  4. The dtypes survive the parquet round trip. uint8 that comes back float64
     still compares equal to 1, so nothing downstream would notice.
  5. `attach` aligns on ts. An off-by-one join gives every bar its neighbour's
     regime, and the column is fully populated either way.
  6. Our ADX/ATR are the numbers `backtest.profiler` computes. Two good-faith
     Wilder implementations that differ in the last decimal put bars on
     opposite sides of a boundary.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402

from mdlib import regimes                                           # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(f"{label} {detail}")


def synthetic_bars(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Bars with a deliberate volatility regime change, so all four quadrants
    are populated rather than the fixture testing one branch four times."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2013-01-02", periods=n, freq="15min", tz="UTC")
    # A calm first half and a violent, trending second half.
    vol = np.where(np.arange(n) < n // 2, 0.3, 1.6)
    drift = np.where(np.arange(n) < n // 2, 0.0, 0.06)
    close = 100 + np.cumsum(rng.normal(drift, vol, n))
    rng2 = np.random.default_rng(seed + 1)
    span = np.abs(rng2.normal(0, vol, n)) + 0.05
    return pd.DataFrame({
        "ts": ts,
        "symbol": "SYNTH",
        "open": close - rng2.normal(0, vol * 0.2, n),
        "high": close + span,
        "low": close - span,
        "close": close,
        "volume": rng2.integers(100, 1000, n),
    })


def test_encoding_and_warmup() -> None:
    print("\n[1] quadrant encoding, booleans and the warm-up sentinel")
    bars = synthetic_bars()
    frame, theta = regimes.build(bars, "2013-01-02", "2013-01-20")

    check("columns are exactly the declared contract",
          list(frame.columns) == regimes.REGIME_COLUMNS,
          str(list(frame.columns)))
    check("regime_quadrant is uint8", frame["regime_quadrant"].dtype == "uint8",
          str(frame["regime_quadrant"].dtype))
    check("adx_14 is float32", frame["adx_14"].dtype == "float32")
    check("atr_14 is float32", frame["atr_14"].dtype == "float32")
    check("is_trending is bool", frame["is_trending"].dtype == bool)
    check("is_high_vol is bool", frame["is_high_vol"].dtype == bool)
    check("index is a UTC DatetimeIndex named timestamp",
          isinstance(frame.index, pd.DatetimeIndex)
          and str(frame.index.tz) == "UTC"
          and frame.index.name == "timestamp")

    defined = frame["adx_14"].notna() & frame["atr_14"].notna()

    # 2: the sentinel. Every undefined bar is 0, every defined bar is 1-4.
    check("every warm-up bar is quadrant 0",
          bool((frame.loc[~defined, "regime_quadrant"] == 0).all()),
          f"{int((frame.loc[~defined, 'regime_quadrant'] != 0).sum())} leaked")
    check("no defined bar is quadrant 0",
          bool((frame.loc[defined, "regime_quadrant"] != 0).all()))
    check("there ARE warm-up bars to sentinel", int((~defined).sum()) > 0,
          f"{int((~defined).sum())}")
    check("warm-up bars are not flagged trending or high-vol",
          not frame.loc[~defined, "is_trending"].any()
          and not frame.loc[~defined, "is_high_vol"].any())

    # 1: the integer and the booleans are one statement, checked by rebuilding
    # the integer from the booleans rather than by re-running the same code.
    expect = pd.Series(0, index=frame.index, dtype="uint8")
    hv, tr = frame["is_high_vol"], frame["is_trending"]
    expect[defined & hv & tr] = 1
    expect[defined & hv & ~tr] = 2
    expect[defined & ~hv & tr] = 3
    expect[defined & ~hv & ~tr] = 4
    check("quadrant integer agrees with the boolean columns, bar for bar",
          bool((expect == frame["regime_quadrant"]).all()),
          f"{int((expect != frame['regime_quadrant']).sum())} disagree")

    # The boundaries themselves, checked against the raw indicator.
    check("is_trending is exactly ADX > 25.0 on defined bars",
          bool((frame.loc[defined, "is_trending"]
                == (frame.loc[defined, "adx_14"] > 25.0)).all()))
    check("is_high_vol is exactly ATR > theta_vol on defined bars",
          bool((frame.loc[defined, "is_high_vol"]
                == (frame.loc[defined, "atr_14"] > theta)).all()))

    counts = frame["regime_quadrant"].value_counts()
    check("the fixture populates all four quadrants",
          all(int(counts.get(q, 0)) > 0 for q in (1, 2, 3, 4)),
          str(dict(counts)))


def test_theta_is_in_sample_only() -> None:
    print("\n[2] theta_vol comes from the in-sample window alone")
    bars = synthetic_bars()
    frame_all = regimes._wilder_frame(bars)
    atr = frame_all["atr_14"]

    is_start, is_end = "2013-01-02", "2013-01-20"
    theta = regimes.in_sample_theta(atr, is_start, is_end)

    lo = pd.Timestamp(is_start, tz="UTC")
    hi = pd.Timestamp(is_end, tz="UTC")
    manual = float(atr.loc[(atr.index >= lo) & (atr.index <= hi)]
                   .dropna().median())
    check("theta_vol == median(ATR) over the window, computed by hand",
          np.isclose(theta, manual), f"{theta} vs {manual}")

    full = float(atr.dropna().median())
    check("the in-sample median genuinely differs from the full-series median",
          not np.isclose(theta, full),
          f"in-sample {theta:.4f} vs full {full:.4f}")

    # The OOS tail must not move the boundary.
    truncated = bars[bars["ts"] <= pd.Timestamp("2013-02-15", tz="UTC")]
    theta_trunc = regimes.in_sample_theta(
        regimes._wilder_frame(truncated)["atr_14"], is_start, is_end)
    check("adding or removing OOS bars leaves theta_vol unchanged",
          np.isclose(theta, theta_trunc), f"{theta} vs {theta_trunc}")

    # An empty window must raise, not return NaN.
    try:
        regimes.in_sample_theta(atr, "1999-01-01", "1999-12-31")
        check("an empty in-sample window raises", False, "it returned")
    except regimes.RegimeCacheError:
        check("an empty in-sample window raises", True)


def test_roundtrip_and_provenance() -> None:
    print("\n[3] parquet round trip, dtypes and provenance")
    bars = synthetic_bars()
    frame, theta = regimes.build(bars, "2013-01-02", "2013-01-20")

    with tempfile.TemporaryDirectory() as tmp:
        path = regimes.write_cache(frame, "SYNTH", "15m", theta,
                                   "2013-01-02", "2013-01-20",
                                   bar_flags={"session_merge": True},
                                   root=tmp)
        check("file lands at {SYMBOL}_{TF}_regime.parquet",
              Path(path).name == "SYNTH_15m_regime.parquet", Path(path).name)
        check("no .tmp file is left behind",
              not list(Path(tmp).glob("*.tmp")))

        back = regimes.load_cache("SYNTH", "15m", root=tmp)
        check("regime_quadrant survives as uint8",
              back["regime_quadrant"].dtype == "uint8",
              str(back["regime_quadrant"].dtype))
        check("adx_14 survives as float32", back["adx_14"].dtype == "float32")
        check("atr_14 survives as float32", back["atr_14"].dtype == "float32")
        check("is_trending survives as bool", back["is_trending"].dtype == bool)
        check("index survives as UTC timestamps",
              isinstance(back.index, pd.DatetimeIndex)
              and str(back.index.tz) == "UTC")
        check("values are unchanged by the round trip",
              bool((back["regime_quadrant"] == frame["regime_quadrant"]).all())
              and np.allclose(back["atr_14"], frame["atr_14"], equal_nan=True))

        prov = regimes.provenance("SYNTH", "15m", root=tmp)
        check("provenance records theta_vol",
              prov is not None and np.isclose(prov["theta_vol"], theta))
        check("provenance records the in-sample window",
              prov["is_start"] == "2013-01-02" and prov["is_end"] == "2013-01-20")
        check("provenance records the bar hygiene flags",
              prov["bar_flags"] == {"session_merge": True})
        check("provenance records the warm-up count",
              prov["undefined_bars"] == int((frame["regime_quadrant"] == 0).sum()))
        check("a missing file gives None, not an exception",
              regimes.load_cache("NOPE", "15m", root=tmp) is None
              and regimes.provenance("NOPE", "15m", root=tmp) is None)


def test_attach_alignment() -> None:
    print("\n[4] attach: alignment, cache miss, and out-of-span bars")
    bars = synthetic_bars()
    frame, theta = regimes.build(bars, "2013-01-02", "2013-01-20")

    with tempfile.TemporaryDirectory() as tmp:
        regimes.write_cache(frame, "SYNTH", "15m", theta,
                            "2013-01-02", "2013-01-20", root=tmp)
        regimes._warned.clear()

        joined = regimes.attach(bars, "SYNTH", "15m", root=tmp)
        check("row count is unchanged by the join",
              len(joined) == len(bars), f"{len(joined)} vs {len(bars)}")
        check("original columns are all still present",
              all(c in joined.columns for c in bars.columns))
        check("regime columns are all present",
              all(c in joined.columns for c in regimes.REGIME_COLUMNS))
        check("regime_quadrant is uint8 after the join",
              joined["regime_quadrant"].dtype == "uint8")

        # 5: alignment. Compare against a direct per-timestamp lookup, not
        # against the join's own ordering.
        lookup = frame["regime_quadrant"].reindex(
            pd.DatetimeIndex(joined["ts"])).to_numpy()
        check("every bar carries its OWN timestamp's quadrant",
              bool((lookup == joined["regime_quadrant"].to_numpy()).all()),
              f"{int((lookup != joined['regime_quadrant'].to_numpy()).sum())} misaligned")

        # And prove the check would catch a shift.
        shifted = np.roll(lookup, 1)
        check("a one-bar shift would be detected by that comparison",
              not bool((shifted == joined["regime_quadrant"].to_numpy()).all()))

        # A subset of bars joins correctly too.
        sub = bars.iloc[500:900].reset_index(drop=True)
        j2 = regimes.attach(sub, "SYNTH", "15m", root=tmp)
        exp = frame["regime_quadrant"].reindex(
            pd.DatetimeIndex(sub["ts"])).to_numpy()
        check("a mid-series slice joins to the same quadrants",
              bool((exp == j2["regime_quadrant"].to_numpy()).all()))

        # A cache miss returns the frame UNCHANGED - no columns invented.
        miss = regimes.attach(bars, "ABSENT", "15m", root=tmp)
        check("a cache miss returns the bars unchanged",
              list(miss.columns) == list(bars.columns) and len(miss) == len(bars))
        check("a cache miss adds no regime_quadrant column",
              "regime_quadrant" not in miss.columns)

        # Bars beyond the cache's span are stamped undefined, not dropped.
        extra = bars.copy()
        extra.loc[:, "ts"] = extra["ts"] + pd.Timedelta(days=3650)
        j3 = regimes.attach(extra, "SYNTH", "15m", root=tmp)
        check("out-of-span bars are kept, not dropped", len(j3) == len(extra))
        check("out-of-span bars are stamped quadrant 0",
              bool((j3["regime_quadrant"] == 0).all()))
        check("out-of-span bars stay uint8",
              j3["regime_quadrant"].dtype == "uint8")


def test_matches_profiler() -> None:
    print("\n[5] the cache's ADX/ATR are the profiler's ADX/ATR")
    bars = synthetic_bars()
    ours = regimes._wilder_frame(bars)

    # The exact call backtest/profiler.py makes, on the same bars.
    import pandas_ta  # noqa: F401
    theirs = bars.set_index(pd.DatetimeIndex(bars["ts"]))[
        ["open", "high", "low", "close"]].astype("float64").copy()
    theirs.ta.adx(length=14, append=True)
    theirs.ta.atr(length=14, append=True)
    adx_col = [c for c in theirs.columns if c.startswith("ADX")][0]
    atr_col = [c for c in theirs.columns
               if c.startswith("ATRe") or c.startswith("ATR")][0]

    check("profiler's ADX column is ADX_14, not ADXR",
          adx_col == "ADX_14", adx_col)
    check("ADX matches the profiler bar for bar",
          np.allclose(ours["adx_14"].to_numpy(dtype="float64"),
                      theirs[adx_col].to_numpy(dtype="float64"),
                      equal_nan=True, rtol=1e-5))
    check("ATR matches the profiler bar for bar",
          np.allclose(ours["atr_14"].to_numpy(dtype="float64"),
                      theirs[atr_col].to_numpy(dtype="float64"),
                      equal_nan=True, rtol=1e-5))

    # And the 25.0 boundary is the same constant on both sides.
    from backtest import profiler
    check("quadrant labels match backtest.profiler.REGIMES exactly",
          tuple(regimes.QUADRANT_LABELS[q] for q in (1, 2, 3, 4))
          == tuple(profiler.REGIMES),
          f"{[regimes.QUADRANT_LABELS[q] for q in (1,2,3,4)]} vs "
          f"{list(profiler.REGIMES)}")


def test_input_guards() -> None:
    print("\n[6] bad input raises rather than writing a plausible file")
    bars = synthetic_bars(200)

    try:
        regimes._wilder_frame(bars.iloc[0:0])
        check("empty bars raise", False, "it returned")
    except regimes.RegimeCacheError:
        check("empty bars raise", True)

    try:
        regimes._wilder_frame(bars.drop(columns=["high"]))
        check("a missing OHLC column raises", False, "it returned")
    except regimes.RegimeCacheError:
        check("a missing OHLC column raises", True)

    try:
        regimes._wilder_frame(bars.iloc[::-1].reset_index(drop=True))
        check("unsorted bars raise", False, "it returned")
    except regimes.RegimeCacheError:
        check("unsorted bars raise", True)

    try:
        regimes._wilder_frame(pd.concat([bars, bars]).reset_index(drop=True))
        check("duplicate timestamps raise", False, "it returned")
    except regimes.RegimeCacheError:
        check("duplicate timestamps raise", True)

    try:
        regimes.build(bars, "2022-01-01", "2013-01-01")
        check("an inverted in-sample window raises", False, "it returned")
    except regimes.RegimeCacheError:
        check("an inverted in-sample window raises", True)


def test_lake_integration() -> None:
    """
    The join as `mdlib.lake` actually performs it. Needs the lake and a built
    cache; skips LOUDLY without either, because a section that quietly does
    nothing reads as a section that passed.
    """
    print("\n[7] mdlib.lake integration")
    try:
        from mdlib import lake
        cached = [s for s in ("NQ", "GC")
                  if regimes.cache_path(s, "15m").exists()]
        uncached = [s for s in ("ES", "CL", "ZN")
                    if not regimes.cache_path(s, "15m").exists()]
        if not cached or not uncached:
            print("  SKIPPED - needs at least one cached and one uncached "
                  f"symbol (cached={cached}, uncached={uncached})")
            return
        sym, other = cached[0], uncached[0]
        start, end = "2023-06-01", "2023-06-05"
        probe = lake.get_bars(sym, "15m", start, end, regimes=False)
        if probe.empty:
            print("  SKIPPED - no lake")
            return
    except Exception as e:                                       # noqa: BLE001
        print(f"  SKIPPED - {type(e).__name__}: {e}")
        return

    df = lake.get_bars(sym, "15m", start, end)
    check("get_bars attaches regime_quadrant automatically",
          "regime_quadrant" in df.columns)
    check("regime_quadrant is uint8 off get_bars",
          df["regime_quadrant"].dtype == "uint8", str(df["regime_quadrant"].dtype))
    check("the join added no rows", len(df) == len(probe))

    # Each contract gets its OWN regime. The regime file is keyed by timestamp
    # alone, so a join on the concatenated frame would give every symbol the
    # first contract's ADX and the column would still be fully populated.
    if len(cached) > 1:
        both = lake.get_bars(cached[:2], "15m", start, end)
        agree = True
        for s2 in cached[:2]:
            c = regimes.load_cache(s2, "15m")
            part = both[both["symbol"] == s2]
            exp = c["regime_quadrant"].reindex(
                pd.DatetimeIndex(part["ts"])).to_numpy()
            agree &= bool((exp == part["regime_quadrant"].to_numpy()).all())
        check("each symbol carries its own contract's quadrant", agree)

        a = both[both.symbol == cached[0]].set_index("ts")["regime_quadrant"]
        b = both[both.symbol == cached[1]].set_index("ts")["regime_quadrant"]
        shared = a.index.intersection(b.index)
        check("the two contracts' quadrants genuinely differ, so that check "
              "has teeth", int((a[shared] != b[shared]).sum()) > 0,
              f"{int((a[shared] != b[shared]).sum())} of {len(shared)}")

    # iter_bars and get_bars must keep returning the same frame -
    # tests/test_streaming_lake.py pins this and the regime join runs on both.
    frames = [d for _, d in lake.iter_bars([sym], "15m", start, end)]
    re = (pd.concat(frames, ignore_index=True)
            .sort_values(["ts", "symbol"]).reset_index(drop=True))
    check("iter_bars reassembles to exactly get_bars", re.equals(df))

    # A cache miss adds no columns at all.
    regimes._warned.clear()
    miss = lake.get_bars(other, "15m", start, end)
    check("an uncached symbol comes back with no regime columns",
          "regime_quadrant" not in miss.columns, str(list(miss.columns)))

    # The mixed case: concat unions columns, so the uncached symbol's rows
    # arrive as NaN and silently demote uint8 to float64.
    mixed = lake.get_bars([sym, other], "15m", start, end)
    check("a mixed cached/uncached request keeps regime_quadrant uint8",
          mixed["regime_quadrant"].dtype == "uint8",
          str(mixed["regime_quadrant"].dtype))
    check("a mixed request keeps is_trending bool",
          mixed["is_trending"].dtype == bool, str(mixed["is_trending"].dtype))
    check("the uncached symbol's rows are quadrant 0, not NaN",
          bool((mixed.loc[mixed["symbol"] == other,
                          "regime_quadrant"] == 0).all())
          and not mixed["regime_quadrant"].isna().any())
    check("the cached symbol still carries real quadrants in a mixed request",
          int((mixed.loc[mixed["symbol"] == sym,
                         "regime_quadrant"] != 0).sum()) > 0)

    off = lake.get_bars(sym, "15m", start, end, regimes=False)
    check("regimes=False opts out entirely",
          "regime_quadrant" not in off.columns)


if __name__ == "__main__":
    print("=" * 70)
    print("REGIME FEATURE CACHE")
    print("=" * 70)
    test_encoding_and_warmup()
    test_theta_is_in_sample_only()
    test_roundtrip_and_provenance()
    test_attach_alignment()
    test_matches_profiler()
    test_input_guards()
    test_lake_integration()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
