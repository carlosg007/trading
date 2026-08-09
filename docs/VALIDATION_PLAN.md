# Data validation plan

Drafted 2026-08-09, to execute next session.

The download completed: **110,113,410 rows across 30 symbols, 2 schemas
(`ohlcv-1m`, `ohlcv-1d`), no failures.** Range 2010-06-06 to 2026-08-08,
except where a contract launched later.

Nothing gets built on this data until it has been validated. A silently
truncated or gap-ridden file is worse than a missing one, because a backtest
will run on it happily and produce a number you believe.

---

## Deliverables

Three scripts and three reference files.

| Script | Location | Produces |
|---|---|---|
| `validate_lake.py` | `scripts/` | Console report + `artifacts/validation/` CSVs |
| `fetch_degraded_days.py` | `data_pull/` | `reference/futures/degraded_days.csv` |
| `mdlib/lake.py` | `mdlib/` | The reader everything else uses |

---

## Phase 1 — Inventory

Confirm what actually landed before checking whether it is any good.

**Checks**

- All 30 symbol directories exist under `lake/futures/bars/`
- Both `tf=1m` and `tf=1d` present for each symbol
- File count per symbol-year — a missing month is a silent hole
- Total file count and disk size
- Raw DBN archive count in `raw/futures/` matches the lake
- Roll calendar JSON exists for all 30 symbols in `reference/futures/`

**Output:** `artifacts/validation/inventory.csv` — one row per
symbol × timeframe × year with file count and row count.

---

## Phase 2 — Row count sanity

Expected counts, to catch partial downloads.

| Timeframe | Expected per year | Notes |
|---|---|---|
| 1m, full Globex | ~345,000 | 23h sessions × ~250 days |
| 1m, thinner contracts | Lower, legitimately | Illiquid markets have gaps with no trades |
| 1d | ~250–260 | **Observed 310–313 — investigate** |

**The daily count needs explaining before anything else.** ZN 2022 returned
310 daily bars and YM 2015 returned 312, against ~252 trading days in a
calendar year. That is 23% more bars than trading days.

Possible explanations:
- Databento's trading-day convention (Sunday evening open starting Monday's
  session) creating extra boundary bars
- Separate bars around contract rolls
- Partial or holiday sessions counted separately
- Something else entirely

**Do not assume.** Pull the actual date sequence for one symbol-year, print
it, and look at what is there. Weekly resampling inherits whatever this is,
so it must be understood, not guessed at.

```bash
python -c "
import pandas as pd
df = pd.read_parquet('/mnt/backtest/lake/futures/bars/symbol=ZN/tf=1d')
d = df[df.ts.dt.year==2022].ts.dt.date
print(len(d)); print(d.head(20).tolist()); print(d.value_counts().head())
"
```

Check specifically for duplicate dates and for bars on weekends.

---

## Phase 3 — Structural integrity

Per symbol, per timeframe:

- **Duplicate timestamps** — should be zero; the puller de-duplicates, so any
  present indicate a deeper problem
- **Monotonic ordering** — timestamps strictly increasing
- **Timezone** — all UTC, tz-aware, no naive timestamps
- **Gaps** — missing sessions, missing days, unexplained holes mid-session
- **Month boundaries** — no rows landing in the wrong partition
- **Schema** — every file has `ts, open, high, low, close, volume, symbol`
  with consistent dtypes

**Output:** `artifacts/validation/gaps.csv` — symbol, timeframe, gap start,
gap end, duration, whether it coincides with a known holiday.

---

## Phase 4 — Price and volume sanity

- **OHLC relationships** — `high >= max(open, close)`, `low <= min(open,
  close)`, `high >= low`. Violations mean corrupt data.
- **Zero or negative prices** — should not exist
- **Zero volume bars** — legitimate in thin markets, but a run of them is
  suspicious
- **Extreme moves** — flag bars where the return exceeds some multiple of
  trailing volatility. Most will be real (limit moves, gaps, the 2020 crash),
  but this catches decimal errors and bad ticks.
- **Roll gaps** — Databento does not back-adjust, so a price jump at every
  roll is *expected*. Cross-reference the flagged jumps against the roll
  calendars. A jump that is not on a roll date needs investigation.

**Output:** `artifacts/validation/anomalies.csv` — symbol, timestamp, what
tripped, the values involved.

---

## Phase 5 — Cross-timeframe consistency

The daily bars were pulled natively, not derived, so they should be checked
against the minute bars rather than assumed to agree.

For a sample of symbol-days:

- Does daily high equal the max of that session's 1m highs?
- Does daily low equal the min?
- Does daily volume equal the sum?
- Where do the session boundaries actually fall?

**Disagreement is informative, not necessarily wrong** — it reveals
Databento's session definition, which is exactly what needs to be encoded in
`mdlib` for resampling. Document what it turns out to be.

---

## Phase 6 — Degraded days

Databento flagged sessions as reduced quality throughout the download. The log
warnings truncate after three dates (`...`), so the log is not a reliable
source.

**Pull the real list from the dataset condition endpoint.** It is a metadata
call, so it costs nothing.

```python
client.metadata.get_dataset_condition(
    dataset="GLBX.MDP3", start_date="2010-06-06", end_date="2026-08-08"
)
```

**Output:** `reference/futures/degraded_days.csv` — date, condition.

**Why this matters more than it looks.** The observed degraded dates cluster
on high-volatility days: 2020-02-27 and 2020-02-28 (COVID crash), 2024-09-18
(FOMC 50bp cut), quarter-end rolls. Exchange feeds get stressed exactly when
it matters most.

The practical risk: a strategy looks robust through a crisis because the worst
bars are missing. Every stress test must check this file first.

---

## Phase 7 — Coverage summary

One table, one row per symbol:

| Column | Meaning |
|---|---|
| symbol | |
| first_date / last_date | Actual coverage, not requested range |
| n_years | |
| rows_1m / rows_1d | |
| pct_days_with_data | Against an expected trading calendar |
| median_daily_volume | Liquidity ranking for slippage assumptions |
| n_gaps | From phase 3 |
| n_anomalies | From phase 4 |
| n_degraded_days | From phase 6 |

**Output:** `reference/futures/coverage.csv`

This is the reference for every later decision about which symbols are
tradeable and what history is usable.

Expected findings, to confirm rather than discover later:
- BTC starts around December 2017
- ETH starts around February 2021 (2021 was the first year returning rows)
- Thinner contracts (PL, LE, KC, SB, CT) will have materially lower 1m row
  counts than ES — legitimate, not an error

---

## Phase 8 — `mdlib` reader

Only after validation passes. One function every strategy uses.

```python
get_bars(symbols, tf="1d", start=None, end=None,
         session="rth"|"globex"|None) -> pd.DataFrame
```

Requirements:

- **Cross-sectional by default** — accepts a list of symbols, returns a long
  or wide frame. Testing one symbol at a time leads to conclusions that do not
  hold up.
- **Reads native `1m` and `1d`; derives `30m`, `1h`, `4h` from 1m and `1w`
  from 1d.** Nothing else is stored.
- **Owns the session definition** discovered in phase 5.
- **Exposes roll dates** from the roll calendars so backtests can exclude them.
- **Exposes degraded days** so stress tests can check.
- DuckDB catalog stays on local disk (`~/.local/share/mdlib/`) — NFSv3 file
  locking over NLM is not safe for DuckDB. The catalog is only an index over
  the Parquet and is rebuildable.

---

## Order of work

1. Phase 1 inventory — fast, tells you if anything is structurally missing
2. **Phase 2 daily-count investigation — do this early, it may change how
   `mdlib` resamples**
3. Phases 3–4 structural and price checks
4. Phase 6 degraded days — independent, can run any time
5. Phase 5 cross-timeframe — needs phase 2 resolved first
6. Phase 7 coverage summary
7. Phase 8 `mdlib`

---

## Gate

**No strategy work begins until phases 1–7 are clean, or every exception is
documented and understood.**

An unexplained anomaly is not a small problem — it is a number that will
silently propagate into every backtest built on top of it.
