#!/usr/bin/env python3
"""
classify_regime.py - Label each symbol-year as Bull, Bear, or Neutral.

Location:  ~/src/trading/scripts/classify_regime.py

Why
---
Backtest results mean very little without knowing what market they ran in.
A strategy that made money 2016-2021 and lost it in 2022 isn't broken - it's
long-biased. This produces a small table you join onto backtest output so
performance can be read per regime.

Method
------
Primary label comes from calendar-year return on daily closes:

    return >  +threshold  -> Bull
    return <  -threshold  -> Bear
    otherwise             -> Neutral

Default threshold is 10%. Change with --threshold.

Supporting columns are also written so you can reclassify later without
recomputing anything:

    ret_pct          calendar-year return, %
    max_drawdown_pct largest peak-to-trough decline within the year, %
    realized_vol_pct annualized stdev of daily log returns, %
    pct_days_above_200sma   share of days closing above the 200-day SMA
    trend_label      Bull/Bear/Neutral from the 200SMA measure alone
    n_days           trading days observed (data-quality check)

trend_label is a second opinion: >60% of days above the 200SMA is Bull,
<40% is Bear, else Neutral. When it disagrees with the return label, the
year was probably choppy or had a sharp reversal - worth looking at.

Reads
-----
    Daily bars via mdlib.lake.iter_bars (which applies the Sunday-session
    merge), from /mnt/backtest/lake/futures/bars/symbol=X/tf=1d/year=*/month=*/

Writes
------
    /mnt/backtest/reference/futures/regimes.parquet
    /mnt/backtest/reference/futures/regimes.csv

Usage
-----
    python classify_regime.py                          # all symbols in lake
    python classify_regime.py --symbols ES NQ
    python classify_regime.py --threshold 15
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ, so an operator
# opening a fresh terminal never has to `source .env` first. It runs at import
# time, above the imports below, because modules resolve their BT_* variables
# while being imported and loading the file inside main() would be too late for
# those - and would work here, which is the kind of difference nobody notices
# until one runner silently uses the default path. The rules live in ONE
# module: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run as `python scripts/classify_regime.py`, so sys.path[0] is scripts/ and the
# repo root is absent. mdlib is the only sanctioned lake reader, so put the root
# on the path rather than reaching into the Parquet files directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mdlib.lake import iter_bars  # noqa: E402

LAKE = Path("/mnt/backtest/lake/futures/bars")
OUT_DIR = Path("/mnt/backtest/reference/futures")

TRADING_DAYS = 252

# 200SMA second-opinion bands
SMA_BULL = 0.60
SMA_BEAR = 0.40


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[!] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
def discover_symbols() -> list[str]:
    if not LAKE.exists():
        sys.exit(f"Lake not found: {LAKE}")
    syms = sorted(
        d.name.split("=", 1)[1]
        for d in LAKE.iterdir()
        if d.is_dir() and d.name.startswith("symbol=")
    )
    return syms


def load_daily(sym: str) -> pd.DataFrame | None:
    """
    Daily bars for one symbol, via the sanctioned reader.

    This used to read symbol=<SYM>/tf=1d/data.parquet directly - a flat file
    above the year= partitions. Those files were the double-counting hazard
    validate_lake.py flags as strays, and were removed on 2026-08-14, at which
    point this function silently found nothing for all 27 symbols.

    Going through mdlib.lake also buys the Sunday-session merge, which matters
    here specifically: CME opens Sunday 18:00 ET, so a raw UTC day boundary
    manufactures ~51 thin stub "days" a year. Those distort both the 200-day
    SMA and the realized-vol estimate this script reports.
    """
    try:
        frames = [df for _, df in iter_bars([sym], "1d", None, None)]
    except Exception as e:
        warn(f"{sym}: could not read daily bars ({e})")
        return None
    if not frames or frames[0].empty:
        warn(f"{sym}: no daily bars in the lake")
        return None

    df = frames[0]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------
def max_drawdown_pct(closes: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative percentage."""
    running_max = closes.cummax()
    dd = (closes / running_max) - 1.0
    return float(dd.min() * 100.0)


def annualized_vol_pct(closes: pd.Series) -> float:
    """Annualized stdev of daily log returns, in percent."""
    if len(closes) < 3:
        return float("nan")
    logret = np.log(closes / closes.shift(1)).dropna()
    if logret.empty:
        return float("nan")
    return float(logret.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0)


def label_from_return(ret_pct: float, threshold: float) -> str:
    if ret_pct > threshold:
        return "Bull"
    if ret_pct < -threshold:
        return "Bear"
    return "Neutral"


def label_from_sma(pct_above: float) -> str:
    if np.isnan(pct_above):
        return "Unknown"
    if pct_above > SMA_BULL:
        return "Bull"
    if pct_above < SMA_BEAR:
        return "Bear"
    return "Neutral"


# --------------------------------------------------------------------------
def classify(sym: str, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per calendar year."""
    df = df.copy()

    # 200SMA computed on the FULL series, not per year - otherwise January
    # would have no SMA at all. Sliced per year afterwards.
    df["sma200"] = df["close"].rolling(200, min_periods=200).mean()
    df["above_sma"] = df["close"] > df["sma200"]
    df["year"] = df["ts"].dt.year

    rows = []
    for year, g in df.groupby("year", sort=True):
        g = g.sort_values("ts")
        closes = g["close"]

        if len(closes) < 2:
            warn(f"{sym} {year}: only {len(closes)} days, skipping")
            continue

        first, last = float(closes.iloc[0]), float(closes.iloc[-1])
        ret_pct = (last / first - 1.0) * 100.0

        sma_valid = g["sma200"].notna()
        pct_above = (
            float(g.loc[sma_valid, "above_sma"].mean())
            if sma_valid.any() else float("nan")
        )

        rows.append({
            "symbol": sym,
            "year": int(year),
            "regime": label_from_return(ret_pct, threshold),
            "ret_pct": round(ret_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct(closes), 2),
            "realized_vol_pct": round(annualized_vol_pct(closes), 2),
            "pct_days_above_200sma": (
                round(pct_above, 3) if not np.isnan(pct_above) else np.nan
            ),
            "trend_label": label_from_sma(pct_above),
            "first_close": round(first, 2),
            "last_close": round(last, 2),
            "n_days": int(len(g)),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Label each symbol-year Bull/Bear/Neutral.")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Symbols to classify. Default: everything in the lake.")
    p.add_argument("--threshold", type=float, default=10.0,
                   help="Annual return %% band for Bull/Bear. Default 10.")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    symbols = args.symbols or discover_symbols()
    if not symbols:
        sys.exit(f"No symbols found in {LAKE}. Run pull_futures.py first.")

    log(f"Classifying {len(symbols)} symbol(s) at +/-{args.threshold}% bands")

    frames = []
    for sym in symbols:
        daily = load_daily(sym)
        if daily is None:
            continue
        res = classify(sym, daily, args.threshold)
        if not res.empty:
            frames.append(res)
            log(f"{sym}: {len(res)} years")

    if not frames:
        sys.exit("Nothing classified. Is there daily data in the lake?")

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "year"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / "regimes.parquet"
    csv = out_dir / "regimes.csv"

    out.to_parquet(pq, engine="pyarrow", compression="zstd", index=False)
    out.to_csv(csv, index=False)

    print()
    print(out.to_string(index=False))
    print()

    # Flag years where the two methods disagree - usually choppy or reversal years
    disagree = out[
        (out["trend_label"] != "Unknown") & (out["regime"] != out["trend_label"])
    ]
    if not disagree.empty:
        print("Years where return-based and 200SMA-based labels disagree:")
        print(disagree[["symbol", "year", "regime", "trend_label",
                        "ret_pct", "pct_days_above_200sma"]].to_string(index=False))
        print()

    log(f"Wrote {pq}")
    log(f"Wrote {csv}")


if __name__ == "__main__":
    main()
