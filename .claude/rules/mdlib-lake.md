---
name: mdlib-lake
description: "The single lake reader, the .env bootstrap, the pre-computed regime cache and theta_vol, plus lake validation and data-pull commands."
paths:
  - "mdlib/**"
  - "scripts/precompute_regimes.py"
  - "scripts/validate_lake.py"
  - "scripts/generate_manifest.py"
  - "scripts/coverage_summary.py"
  - "scripts/classify_regime.py"
  - "data_pull/**"
  - "tests/test_streaming_lake.py"
  - "tests/test_regime_cache.py"
  - "tests/test_profiler_precomputed.py"
  - "tests/test_env_bootstrap.py"
---

# mdlib, the lake, and the regime cache

**`mdlib/env.py`** — the ONE place `.env` is loaded and the ONE place the
Discord webhook is resolved, added 2026-08-24. It lives at the BOTTOM of the
dependency chain so `backtest/`, `scripts/`, `portfolio/`, `realtime/`,
`data_pull/` and `master_live.py` can all import it, and it pulls in `os`,
`pathlib` and `dotenv` and nothing else — it runs at the top of every
entrypoint, above the thread-count variables `backtest/run.py` sets before
numpy is imported.

- **Every operator entrypoint calls `load_env()` at import time**, so a script
  run from a fresh terminal never needs `source .env` first. Twelve runners
  used to carry a byte-identical fourteen-line bootstrap block and the rest
  carried none; the block is now three lines and a call, and
  `tests/test_env_bootstrap.py` fails if a new `__main__` script reads
  `os.environ` without it. A missing file is a fact (`exists: False`), not an
  error — `.env` is optional and every consumer has a default.
- **The repository root comes from `__file__`, never from the working
  directory.** `find_dotenv()` walks up from the CALLER's cwd, and the runs
  that matter start from `/mnt/backtest`, from a `--bg` daemon and from cron;
  from any of those it finds nothing, silently, and the script then uses every
  default path as though the file did not exist. `$BT_ENV_FILE` overrides the
  derived path for a second checkout.
- **An existing environment variable WINS**, so `BT_ARTIFACTS=/tmp/x bt-run`
  still beats the file. The file is parsed ONCE per path: twelve importers
  reach it on a single `bt-run`, and re-reading per import would let two of
  them disagree if the file changed mid-run.
- **`NO_EXPORT` (the CrossTrade credentials) is read from the file and NOT put
  into `os.environ`.** `realtime/live_dispatcher.load_env_file` reads them
  directly and documents why: everything in `os.environ` is inherited by every
  subprocess, which is how a broker key reaches an unrelated tool's debug
  output. A blanket `load_dotenv()` in `master_live.py` would have undone that
  silently. Withholding them costs nothing — `resolve_credentials` reads the
  file before it reads the environment — and a credential the operator
  exported themselves is untouched, because this module never removes a name.
- **`DISCORD_WEBHOOK_VARS` is the one alias chain**, highest precedence first:
  `--webhook`, then `$BT_DISCORD_WEBHOOK`, `$DISCORD_WEBHOOK_URL`,
  `$DISCORD_WEBHOOK`. Before this `backtest/discord_reporter.py` read
  `$BT_DISCORD_WEBHOOK` alone while `scripts/incubator_tracker.py` tried
  `$DISCORD_WEBHOOK_URL` first — so one `.env` configured one card and not the
  other, and the symptom is a report that is simply never posted, which is
  indistinguishable from a quiet pipeline. Neither module spells the chain out
  any more.
- **An empty or whitespace value is UNSET, at every step.** `DISCORD_WEBHOOK=`
  left in a file is a name somebody meant to fill in, and treating it as set
  shadows the alias carrying the URL and fails with the one message ("no
  webhook") that sends the operator to look at the wrong variable.
- **`describe_webhook` returns the VARIABLE NAME beside the URL**, and that
  name is what the success line prints — a webhook in a log outlives the
  session that wrote it and is directly replayable, so the URL is never
  printed. With two channels configured, the name is what says which one
  received the card.

**`mdlib/lake.py`** — The single reader. Two entry points over the same bars,
both returning **long format** (`ts, symbol, open, high, low, close, volume`,
UTC). `wide(df, field)` pivots when a column per symbol is needed.

- **`iter_bars(symbols, tf, start, end, ...)`** yields `(symbol, df)` one symbol
  at a time. **This is what backtests use** — it never builds the monolith, and
  a rolling window on a single-symbol frame cannot bleed across instruments.
- **`get_bars(...)`** concatenates those frames and sorts by `(ts, symbol)`. Use
  it only when a single cross-sectional frame is genuinely needed (correlation
  work, `wide()`); on the full 1m lake it costs ~15 GiB, and the result
  interleaves symbols — see the engine note below.
- **Only `1m` and `1d` are stored.** `5m/15m/30m/1h/2h/4h` derive from 1m, `1w`
  from 1d. Adding a timeframe means a `DERIVED` entry, not a re-pull.
- **Sunday sessions merge into Monday** for `1d`/`1w` (`session_merge=True`).
  CME opens Sunday 18:00 ET, so a UTC day boundary otherwise creates ~51 thin
  stub "days" a year that silently corrupt every lookback window. The merge
  lives in the reader, not the lake — one function to change if it is wrong.
- Hygiene flags: `exclude_degraded`, `exclude_rolls`, `respect_coverage`.
- **`regimes=True` (default) left-joins the pre-computed regime cache** onto
  each symbol's frame. Applied inside `iter_bars`, so `get_bars` inherits it
  and the two keep returning the same columns —
  `tests/test_streaming_lake.py` pins that equality. Joined PER SYMBOL, inside
  the loop, because the regime file is keyed by timestamp alone and a join on
  the concatenated frame would hand every contract NQ's ADX with the column
  still fully populated. A cache MISS is not an error: the frame comes back
  with no regime columns at all, rather than a column of zeros that would read
  as a regime that was computed and found absent. Nothing is computed on the
  fly — see `mdlib/regimes.py` for why an improvised threshold is worse than
  no threshold.
- Reference lookups (`coverage()`, `degraded_days()`, `roll_dates()`) read from
  `/mnt/backtest/reference/futures/` and are `lru_cache`d.
- **Dual-dataset routing is not implemented** — `get_bars` has no `source`
  parameter, so the NT8 tree is unreachable through the reader. See Open Tasks.

**`mdlib/regimes.py`** — the pre-computed regime feature cache, and the only
place the quadrant encoding is written down. Wilder's ADX(14) and ATR(14) are a
pure function of the bars, and every stage that profiles a result was
recomputing them over the same series.

- **`regime_quadrant` is `uint8`: 1 High-Vol/Trending, 2 High-Vol/Ranging,
  3 Low-Vol/Trending, 4 Low-Vol/Ranging, and 0 = UNDEFINED.** The order matches
  `backtest.profiler.REGIMES`, so a quadrant integer and a profiler label are
  the same statement about the same bar. **0 is not a quadrant.** ADX and ATR
  are undefined during their 14-bar warm-up, and `NaN > 25.0` is False — the
  naive encoding files every warm-up bar under Low-Vol/Ranging, a populated
  column of a regime nobody measured. Bars outside the cache's span get 0 for
  the same reason. Anything reading the column must treat 0 as "no regime".
- **The volatility boundary is pinned to the IN-SAMPLE window**, default
  2013-01-01..2022-12-31, and then applied to the whole series including the
  holdout years — which are therefore labelled by a boundary that never saw
  them. A median taken over whatever window the caller asked for is a property
  of the REQUEST, not of the contract: the same bar would be labelled one way
  by an in-sample run and another by a holdout run, and Gate 3 would measure
  retention between two strategies whose regime definitions disagree. `theta_vol`
  is stored in the file, because a quadrant read without knowing which window
  drew its boundary is not a measurement.
- **`backtest.profiler.RegimeProfiler` consumes this cache from 2026-08-20.**
  When the bars carry `regime_quadrant` — which they do whenever they came
  through `mdlib.lake` — the profiler labels every bar from the cached column
  and runs no indicator pass of its own. It falls back to the original live
  ADX/ATR pass otherwise, and the two are NOT equivalent: the fallback takes
  the median ATR of whatever frame it was handed, so its boundary moves with
  the requested date range while the cache's does not. Which one ran is
  recorded on every profile artifact as `regime_source`
  (`precomputed_cache` / `recomputed_live`) alongside the threshold that drew
  the quadrants, and Stage 1 collects them per configuration into
  `regime_screen.regime_source`. **Switching a (symbol, tf) from recomputed to
  cached MOVES its quadrant boundary and therefore its Stage 1 verdict** — that
  is the intended effect, not drift, but a before/after is not comparable
  across the change.
- The integer→label map lives in `backtest/profiler.py` as
  `QUADRANT_TO_REGIME`, built FROM `QUADRANT_LABELS` here and **checked against
  `REGIMES` at import**, which raises on disagreement. A transposed map would
  move every trade between quadrants with every total still adding up.
- **Provenance lives inside the parquet**, as schema metadata, not in a sidecar
  that can be separated from what it describes: `theta_vol`, the in-sample
  window, the indicator lengths, and the lake hygiene flags the bars were read
  under. A cache built on bars that included roll days holds an ATR shaped by
  those gaps; joined onto a run that excluded them every timestamp still
  matches, so `attach` compares the flags and says so.
- **Warnings go to stderr, once per (symbol, tf).** `mdlib.lake` is a library
  and several callers parse a child process's stdout as data.

## Commands

```bash
# Data manifest: path, size, SHA-256, row count, ts range per file.
# Answers "have the bytes changed?"; validate_lake.py answers "is it sane?".
# Both must pass. --verify exits 1 on drift, so it gates a pipeline.
python scripts/generate_manifest.py                   # rebuild manifest.json
python scripts/generate_manifest.py --verify          # detect drift or corruption
python scripts/generate_manifest.py --verify --quick  # size/mtime only, no hashing

# Validate the lake (phases 1-4: inventory, row counts, structure, price sanity)
python scripts/validate_lake.py                     # all symbols
python scripts/validate_lake.py --symbols ES NQ --tf 1d
python scripts/validate_lake.py --quick             # skip price checks

# Build the pre-computed regime cache (ADX14/ATR14/quadrant per symbol+tf).
# Writes /mnt/backtest/lake/regimes/{SYMBOL}_{TF}_regime.parquet ($BT_REGIME_CACHE
# overrides). mdlib.lake picks these up automatically; re-run after a data pull.
python scripts/precompute_regimes.py --symbols NQ,GC --tf 15m,30m
python scripts/precompute_regimes.py --symbols ALL --tf 15m --force
```

```bash
# Rebuild the per-symbol coverage reference (writes reference/futures/coverage*.csv)
python scripts/coverage_summary.py

# Label symbol-years Bull/Bear/Neutral (writes reference/futures/regimes.{parquet,csv})
python scripts/classify_regime.py --threshold 10.0

# Refresh the degraded-sessions calendar (metadata call, free)
export DATABENTO_API_KEY="db-..."
python data_pull/fetch_degraded_days.py

# Download bars. Prints cost first; downloads nothing without --confirm.
python data_pull/pull_futures.py --symbols ES NQ --start 2016-01-01 --end 2026-01-01
python data_pull/pull_futures.py --symbols ES NQ --start 2016-01-01 --end 2026-01-01 --confirm

# Analysis report from a saved BacktestResult
python backtest/report.py \
  --returns /mnt/backtest/artifacts/strat1_returns.parquet \
  --trades  /mnt/backtest/artifacts/strat1_trades.parquet \
  --name "Strat 1" --variants-tested 12 --costs-included yes \
  --out /mnt/backtest/artifacts/strat1_report
```
```
