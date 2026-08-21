"""
backtest/baseline.py - STAGE 1 of 5: does this idea carry on this contract?

Location: ~/src/trading/backtest/baseline.py

Runs BOTH versions - Version A (pure rules) and Version B (the same signals,
ML filtered) - on the strategy's DEFAULT parameters, one independent simulation
per (symbol, timeframe) CONFIGURATION, and answers one question per
configuration: is there anything here at all. Configurations that clear the
screen are written to `surviving_assets.json` as exact (symbol, timeframe)
pairs, and Stage 2 sweeps only those.

    python3 backtest/baseline.py --strat ema_trend_filter --symbols ALL --tf 15m

**The dual simulation is the default, and so is the window.** Survival is
decided on EITHER version, so a screen that ran only the rules cannot
distinguish a configuration neither version carried from one only half of which
was ever asked; `--no-ml` is the deliberate way to accept that narrower answer
for a cheaper run. `--start` / `--end` default to the charter's in-sample
window, 2013-01-01..2022-12-31 - see CHARTER_IS_START. Reading past it means
screening on the Stage 3 holdout.

Where the output goes
---------------------
**The console is a progress line per configuration, and a table of the
WINNERS.** A 27-contract × 4-timeframe screen is 108 scorecards, 108 day-of-week
tables and 108 filter audits; printed in full that is several thousand lines
whose only reliable effect is that nobody reads the last one. The terminal
answers "how far along is it, and what came out"; everything else - both
versions' full metrics, the day-of-week attribution, the entry-filter audit,
and the drop reason for every configuration that failed - is written to
`stage1_baseline_report.md` in the pipeline directory, on every run, whether
anything survived or not.

The report is rewritten from scratch after every configuration, so a screen
killed at 14 of 108 leaves a complete report of 14 rather than nothing.

Why defaults, and why no gate table
-----------------------------------
**Default parameters, deliberately unswept.** A sweep at this stage would
select the best of N per contract and then screen on the result, which promotes
whichever symbol had the most parameters to hide behind. Running one fixed set
everywhere makes the comparison across contracts a comparison of the
CONTRACTS. The parameters get their turn in Stage 2, on the survivors.

**No Gate 1/2/3 block.** Nothing here is entitled to a gate verdict: the
parameters are unswept, the walk-forward has not run, and the holdout must stay
untouched until Stage 3. Three lines of NOT EVALUATED under an ACCEPTANCE GATES
heading is clutter that teaches a reader to skip the gate table, which is the
one thing they must not do when Stage 3 prints a real one. The gates are
deferred, not hidden.

**The screen is a REGIME, not a blended average.** Every configuration is
profiled into four quadrants by `backtest.profiler.RegimeProfiler` — ADX(14)
above 25 is Trending, ATR(14) above the contract's own in-sample median is
High Volatility — and it survives when ONE quadrant carries a profit factor of
1.00 or better over at least 30 trades in that quadrant, on either version.

Since 2026-08-20 the quadrant labels are READ from the pre-computed regime
cache (`mdlib/regimes.py`, joined onto every frame by `mdlib.lake`) rather than
recomputed per stage and per version. The threshold that separates high from
low volatility is therefore the same number in every stage, pinned to the
in-sample window, instead of a median of whatever date range each caller
happened to request.

That is a different question from the one this stage used to ask. A blended
profit factor over the whole window asks whether a strategy makes money on
every bar; a quadrant asks whether there is an environment in which it does.
The second is the honest question for a strategy that will be governed by a
live supervisor able to stand it down — and it is a LOOSER screen.

The bar sat at 1.15 for that reason: the quadrant is picked as the best of
four, so a bar it clears by a hair is a bar it clears by selection rather than
by edge. **It was lowered to 1.00 on 2026-08-20 by operator instruction**, and
that objection is not answered by the change — it is accepted. Stage 1 is now
a wide net feeding the Stage 2 sweep, not a verdict about an edge. See
MIN_REGIME_PROFIT_FACTOR.

**Both bars bind on the SAME quadrant.** A 1.80 profit factor over eleven
trades in one quadrant and a 0.90 over four hundred in another describe a
strategy with no environment; pairing the best factor with the largest count
would advance exactly that.

**A survivor is scoped, and the scope travels with it.** Each surviving pair
carries `status` (PROMOTED), the `version` that carried it, its
`optimal_regime` and `quadrant` (Q1..Q4), that quadrant's own four metrics
(`regime_pf`, `regime_trade_count`, `regime_win_rate`, `regime_net_pnl`), and
`kill_switch_regimes` — the three quadrants the contract must NOT trade in.
A configuration that cleared no quadrant is written to `dropped` with
`status: "DROPPED"`, a `reason`, and an explicitly null `optimal_regime`;
`screen_results` carries every configuration evaluated, in one shape, with the
same keys. The kill switch is DERIVED from the
optimal regime rather than measured: a quadrant that failed the bar and a
quadrant the strategy never traded in are the same instruction to a supervisor,
and reading "no evidence" as "permitted" is what puts a contract into the one
environment nobody sampled.

**This is an in-sample selection layer, and the handoff says so.** The quadrant
is chosen on the same bars Stage 2 sweeps and Stage 3 certifies, so a winning
Sharpe is the best of (parameter combinations × this best-of-four pick).
`regime_screen.selected_in_sample` records it, and Gate 3's holdout retention
is the only evidence it generalised. Stage 2 deliberately sweeps the WHOLE
window rather than masking to the winning quadrant — fitting parameters to a
subset that was itself chosen as the best of four on these same bars would
stack a second selection layer under the first.

Calendar-day pruning is gone
----------------------------
**The Drop Unprofitable Days contract has been REMOVED from this stage.** No
weekday is blacklisted, `surviving_assets.json` carries no `exclude_days`, and
`pipeline.stage1_exclude_days` therefore yields an empty mapping — which that
function already documents as the correct reading of a handoff with nothing
excluded, so Stage 2 sweeps the whole week unless an explicit `--exclude-days`
is passed to it directly.

The day-of-week table is still written to the markdown report and is now purely
DESCRIPTIVE — nothing downstream reads it. Which weekday loses is largely a
restatement of which regime that weekday tends to fall in, and pruning the
calendar masked the environment instead of naming it. Naming it is what the
quadrant does.

`--exclude-days` and `--news-filter` remain on the CLI. They come from the
shared `add_filter_args` and are spelled identically on all five stages and on
`bt-run`; they are explicit operator instructions applied by the engine, not a
pruning decision this stage makes.

Artifacts
---------
Besides `surviving_assets.json` and `stage1_baseline_report.md`, every
configuration writes one regime profile per version —
`regime_profile_<SYMBOL>_<TF>_version_<a|b>.json` — into the same pipeline
directory. The report carries the four-quadrant matrix for every asset
EVALUATED, survivors and drops alike: the matrix of a configuration that failed
is how an operator sees whether it failed on its edge or on its sample size.

`backtest/discord_reporter.py --stage 1` formats this stage's leaderboard
straight off `surviving_assets.json` and posts it to $BT_DISCORD_WEBHOOK. It
recomputes nothing — every number on that card is one this stage wrote.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                   # noqa: E402
import math                                                       # noqa: E402
import sys                                                        # noqa: E402
import time                                                       # noqa: E402
import traceback                                                  # noqa: E402
from datetime import datetime, timezone                           # noqa: E402
from pathlib import Path                                          # noqa: E402

import pandas as pd                                               # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier1_master import run_dual_version_backtest          # noqa: E402
from agents.tier3_workers import load_strategy                     # noqa: E402
from backtest.event_calendar import (add_filter_args, describe_filters,   # noqa: E402
                               filter_config_kwargs)
from backtest.engine import BacktestConfig                         # noqa: E402
from backtest.pipeline import (BASELINE_REPORT_FILE,               # noqa: E402
                               CHARTER_IS_END, CHARTER_IS_START,
                               SURVIVORS_FILE, leaderboard, next_step,
                               pipeline_dir, stage_banner, write_stage)
from backtest.profiler import (QUADRANT_TO_REGIME, REGIMES,         # noqa: E402
                               RegimeProfiler)
from backtest.report import day_of_week_breakdown                   # noqa: E402
from backtest.run import (load_bars, parse_param, parse_symbols,    # noqa: E402
                          parse_timeframes, resolve_strategy)

# The survival bar, applied to ONE regime quadrant rather than to the whole
# sample.
#
# **Lowered from 1.15 to 1.00 on 2026-08-20 by operator instruction.** The
# reason 1.15 was chosen still stands and is not answered by the change: a
# quadrant is a SUBSET chosen after the fact as the best of four, so a bar it
# clears by a hair is a bar it clears by SELECTION rather than by edge. At
# 1.00 a configuration advances on a quadrant that merely broke even, and
# break-even on the best of four looks the same as no edge anywhere.
#
# What this makes Stage 1: a wider net feeding a sweep, not a verdict. Nothing
# downstream loosens - Gate 1 still binds profit factor at 1.00 on the BLENDED
# sample over >= 100 trades, and Gate 3 still has to see the retention hold
# out of sample. Raise it back with --min-profit-factor at any time; the value
# actually used is printed in the stage banner and written onto
# surviving_assets.json, so no handoff records a survivor without recording
# the bar it cleared.
MIN_REGIME_PROFIT_FACTOR = 1.00

# The trade floor the winning QUADRANT must clear on its own trade count - not
# the configuration's total. A 1.60 profit factor over nine trades in one
# quadrant is not an environment, and the whole point of screening per quadrant
# is that the counts get smaller.
MIN_REGIME_TRADES = 30

# The in-sample window the Regime-Switching Incubator Charter fixes for Stage
# 1, and the DEFAULT for --start / --end since 2026-08-21.
#
# It was `None` before, which reads the lake end to end - and the last three
# years of the lake are the Stage 3 HOLDOUT. A screen that quietly included
# them selected which contracts advance on bars Gate 3 then measures retention
# against, and nothing downstream could see that it had happened: every stage
# after this one would report a holdout it had already been shown. The window
# is a default rather than a hard limit, so a deliberate re-screen elsewhere is
# still one flag away, and whichever window ran is written onto the handoff
# beside `window_source` - a survivor is never recorded without recording the
# bars it survived on.
# Imported from `backtest.pipeline`, not declared here. Stage 2 enforces the
# same two dates and Stage 3's holdout begins the day after them; a second copy
# in this module would be one edit away from screening on a window Stage 2 then
# optimises past.

# Q1..Q4, the charter's own shorthand for the four quadrants, INVERTED from the
# profiler's integer map rather than spelled out again: `mdlib.regimes` numbers
# them and `backtest.profiler` checks that numbering against `REGIMES` at
# import, so there is exactly one place where "Q1" and "High Volatility /
# Trending" are the same statement. A second literal map here would be free to
# transpose two quadrants, and every count in every table would still add up.
QUADRANT_ID = {label: f"Q{q}" for q, label in QUADRANT_TO_REGIME.items()}

# Ascending bar length, so the Stage 2 command lists timeframes the way an
# operator reads them (1m, 5m, 15m, 30m) rather than the way `sorted` does
# ("15m", "1m", "30m", "5m"). An unknown token sorts last rather than raising -
# `parse_timeframes` has already validated these against the lake.
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
               "4h": 240, "1d": 1440, "1w": 10080}


# --------------------------------------------------------------------------
# Formatting. Every one of these takes None, NaN and inf, because every one of
# them is handed a metric from a run that may have produced no trades.
# --------------------------------------------------------------------------
def _num(v) -> float | None:
    """A finite float, or None for anything that is not one."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _fmt(v, spec: str = ".2f", na: str = "n/a", suffix: str = "") -> str:
    f = _num(v)
    if f is None:
        return na
    if math.isinf(f):
        return ("inf" if f > 0 else "-inf") + suffix
    return format(f, spec) + suffix


def _human_bars(n: int) -> str:
    """`3.2M`, `661k`, `4,200` - the bar count as a reader would say it."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return f"{n:,}"


def _pair_label(symbol: str, tf: str) -> str:
    return f"{symbol:<5}· {tf:<4}"


def quadrant_id(regime: str | None) -> str | None:
    """
    `Q1` for "High Volatility / Trending", and None for anything that is not
    one of the four.

    None rather than a placeholder string: a configuration that cleared no
    quadrant has no quadrant, and `"Q0"` or `"-"` on a handoff would read as a
    fifth environment nobody defined.
    """
    return QUADRANT_ID.get(regime) if regime else None


# --------------------------------------------------------------------------
# The regime screen
# --------------------------------------------------------------------------
def best_quadrant(profile: dict | None,
                  min_profit_factor: float = MIN_REGIME_PROFIT_FACTOR,
                  min_trades: int = MIN_REGIME_TRADES) -> dict | None:
    """
    The best quadrant of one version's profile that CLEARS BOTH bars, or None.

    Both bars are applied to the same quadrant, which is the whole point: a
    1.80 profit factor in a quadrant with eleven trades and a 0.90 in one with
    four hundred describe a strategy with no environment, and pairing the best
    factor with the largest count would let exactly that through.

    Ranked on profit factor, ties broken on the LARGER trade count. Ties are
    real - a quadrant no bar reaches and one the strategy never traded in both
    round to the same number - and between two equal factors the one measured
    over more trades is the better-evidenced claim, not the one the quadrant
    order happened to put first.

    A profit factor of `inf` (the quadrant never lost) is a result and clears;
    the profiler's 999 sentinel for the same case is left as it is rather than
    normalised here, because rewriting another module's sentinel in a screen is
    how two modules come to disagree about what 999 meant.
    """
    if not profile:
        return None
    best = None
    for regime in REGIMES:
        stats = (profile.get("regime_breakdown") or {}).get(regime)
        if not stats:
            continue
        pf = _num(stats.get("profit_factor"))
        n = int(stats.get("trade_count", 0) or 0)
        if pf is None or pf < float(min_profit_factor) or n < int(min_trades):
            continue
        cand = {"regime": regime, "quadrant": quadrant_id(regime),
                "profit_factor": pf, "trade_count": n,
                "win_rate": _num(stats.get("win_rate")),
                "net_pnl": _num(stats.get("net_pnl"))}
        if best is None or (pf, n) > (best["profit_factor"],
                                      best["trade_count"]):
            best = cand
    return best


def _top_quadrant(profile: dict | None) -> dict | None:
    """
    The highest-profit-factor quadrant IGNORING both bars — for the drop
    reason only, never for survival.

    "best profit factor 0.94 in High Volatility / Trending, below 1.00" tells
    an operator what to change. A bare "no quadrant cleared the bar" sends them
    to re-run the stage to find out how close it was.
    """
    rows = [(_num(v.get("profit_factor")), int(v.get("trade_count", 0) or 0), k)
            for k, v in ((profile or {}).get("regime_breakdown") or {}).items()
            if _num(v.get("profit_factor")) is not None]
    if not rows:
        return None
    pf, n, regime = max(rows)
    return {"regime": regime, "profit_factor": pf, "trade_count": n}


def kill_switch_regimes(optimal_regime: str | None) -> list[str]:
    """
    The three quadrants a survivor must NOT trade in, derived from the one it
    must.

    Derived rather than measured on purpose. A quadrant that failed the bar and
    a quadrant the strategy never traded in are the same instruction to a live
    supervisor — stand down — and treating "no evidence" as "permitted" is the
    reading that puts a strategy into the one environment nobody sampled.

    An unknown or absent optimal regime yields an EMPTY list, never all four.
    "Trade nowhere" is a live-trading instruction, and it must come from a
    decision rather than from a missing value.
    """
    if not optimal_regime or optimal_regime not in REGIMES:
        return []
    return [r for r in REGIMES if r != optimal_regime]


def screen(profiles: dict | None,
           min_profit_factor: float = MIN_REGIME_PROFIT_FACTOR,
           min_trades: int = MIN_REGIME_TRADES) -> tuple[bool, str, dict | None]:
    """
    Did this configuration carry an edge in ANY ONE regime? Returns
    `(survived, reason, best)`.

    `profiles` is `{"A": profile_or_None, "B": profile_or_None}` as
    `RegimeProfiler.generate_profile` returns them. A configuration survives
    when EITHER version has a quadrant at or above `min_profit_factor` over at
    least `min_trades` trades IN THAT QUADRANT, and `best` carries the winning
    quadrant with the version that produced it.

    This replaces the blended profit-factor screen, and it is a different
    question. The old one asked whether the strategy made money across every
    bar of the window; this asks whether there is an environment in which it
    does, and it will advance a contract whose blended factor is below 1.00 on
    the strength of one quadrant — which is the intended loosening. The bar
    sits at 1.00 since 2026-08-20; it sat at 1.15 before, because a quadrant
    cleared by a hair on a best-of-four pick is cleared by selection.

    **The quadrant is chosen in-sample, on the same bars Stage 2 sweeps and
    Stage 3 certifies.** Best-of-four is a selection layer stacked on the
    parameter search, exactly as the weekday pruning it replaces was. The
    handoff records it; Gate 3's holdout retention is the only evidence it
    generalised.

    `profiles["B"] = None` means Version B did not run, and only Version A is
    considered. A skipped comparison is not one the baseline won.
    """
    views = [(k, v) for k, v in (("A", (profiles or {}).get("A")),
                                 ("B", (profiles or {}).get("B"))) if v]
    if not views:
        return False, "no regime profile — the run produced nothing to profile", None

    if not any(int(v.get("trades_profiled", 0) or 0) for _, v in views):
        return False, ("no trades landed in any regime — the signals never "
                       "fired, or every entry sat inside the indicator "
                       "warm-up"), None

    clearing = []
    for label, prof in views:
        q = best_quadrant(prof, min_profit_factor, min_trades)
        if q:
            clearing.append({**q, "version": label})
    if clearing:
        best = max(clearing, key=lambda q: (q["profit_factor"],
                                            q["trade_count"]))
        return True, (f"Version {best['version']} profit factor "
                      f"{best['profit_factor']:.2f} over "
                      f"{best['trade_count']:,} trades in "
                      f"{best['regime']}"), best

    near = [(t, label) for label, prof in views
            if (t := _top_quadrant(prof)) is not None]
    if not near:
        return False, ("no quadrant has a defined profit factor over any "
                       "trades"), None
    top, label = max(near, key=lambda x: x[0]["profit_factor"])
    if top["profit_factor"] < float(min_profit_factor):
        return False, (f"best quadrant {top['profit_factor']:.2f} (Version "
                       f"{label}, {top['regime']}) < "
                       f"{float(min_profit_factor):.2f}"), None
    return False, (f"Version {label} clears {float(min_profit_factor):.2f} in "
                   f"{top['regime']} but on only {top['trade_count']:,} "
                   f"trades < {int(min_trades)}"), None


# --------------------------------------------------------------------------
# The per-configuration record
# --------------------------------------------------------------------------
# What the markdown scorecard prints, in the order it prints it. Kept here
# rather than inline so the report and the JSON snapshot carry the same fields:
# a metric visible on the page and absent from the handoff is the kind of gap
# nobody notices until they go looking for it.
SCORECARD_ROWS = [
    ("Sharpe (daily closes)", "sharpe", ".2f", ""),
    ("Sortino", "sortino", ".2f", ""),
    ("Calmar", "calmar", ".2f", ""),
    ("Profit factor", "profit_factor", ".2f", ""),
    ("Win rate", "win_rate_pct", ".1f", "%"),
    ("Max drawdown", "max_drawdown_pct", ".2f", "%"),
    ("Net return", "total_return_pct", ".2f", "%"),
    ("Annualised return", "annualized_return_pct", ".2f", "%"),
    ("Net P&L", "total_pnl", ",.0f", ""),
    ("Trades", "trade_count", ",.0f", ""),
    ("Friction costs", "total_costs", ",.0f", ""),
    ("Friction per trade", "cost_per_trade", ",.2f", ""),
    ("Friction / gross profit", "cost_share_of_gross_pct", ".1f", "%"),
]


def _scalars(m: dict | None) -> dict | None:
    """
    The scalar metrics of one version - no trade frame, no equity series.

    Rows are serialized into `surviving_assets.json`, and a DataFrame carried
    in one would be stringified by `default=str` into something that is neither
    readable nor parseable.
    """
    if m is None:
        return None
    out = {k: _num(m.get(k)) for k in (
        "sharpe", "sortino", "calmar", "profit_factor", "win_rate",
        "max_drawdown_pct", "total_return_pct", "annualized_return_pct",
        "total_pnl", "gross_pnl", "total_costs", "n_days")}
    out["ok"] = bool(m.get("ok", True))
    out["trade_count"] = int(m.get("trade_count", 0) or 0)
    wr = out.get("win_rate")
    out["win_rate_pct"] = None if wr is None else wr * 100.0
    # Friction, in the two forms that answer different questions: what it cost
    # per trade, and what share of the gross the costs ate. The second is None
    # rather than 0% when there is no gross profit for them to be a share of -
    # the same rule Stage 4 follows.
    costs, gross, n = out.get("total_costs"), out.get("gross_pnl"), out["trade_count"]
    out["cost_per_trade"] = (costs / n) if (costs is not None and n) else None
    out["cost_share_of_gross_pct"] = (
        100.0 * costs / gross if (costs is not None and gross is not None
                                  and gross > 0) else None)
    return out


def _row(symbol: str, tf: str, metrics_a: dict, metrics_b: dict | None,
         survived: bool, reason: str, dow: pd.DataFrame,
         bars: int, elapsed: float,
         profiles: dict | None = None,
         best: dict | None = None) -> dict:
    """
    One configuration's complete record — both versions' metrics, both
    versions' four-quadrant regime breakdowns, and the screen's verdict.

    `best` is the winning quadrant `screen` returned, or None when nothing
    cleared. The three regime fields are written from it and are the ones the
    handoff carries: `optimal_regime`, `regime_pf`, `kill_switch_regimes`.
    A dropped configuration gets `None` and `[]` rather than a plausible
    second-best — a kill switch derived from a quadrant that failed the bar is
    a live-trading instruction nothing certified.
    """
    a, b = _scalars(metrics_a), _scalars(metrics_b)
    profiles = profiles or {}
    optimal = (best or {}).get("regime")

    return {
        "symbol": symbol,
        "timeframe": tf,
        "survived": bool(survived),
        "reason": reason,
        "bars": int(bars),
        "elapsed_s": round(float(elapsed), 1),
        "profit_factor_a": (a or {}).get("profit_factor"),
        "sharpe_a": (a or {}).get("sharpe"),
        "max_drawdown_pct_a": (a or {}).get("max_drawdown_pct"),
        "trades_a": (a or {}).get("trade_count"),
        "win_rate_a": (a or {}).get("win_rate"),
        "net_pnl_a": (a or {}).get("total_pnl"),
        # Blank rather than 0 when Version B never ran. A skipped comparison is
        # not one the baseline won.
        "profit_factor_b": (b or {}).get("profit_factor"),
        "sharpe_b": (b or {}).get("sharpe"),
        "trades_b": (b or {}).get("trade_count"),
        "ml_evaluated": metrics_b is not None,
        "metrics_a": a,
        "metrics_b": b,
        "entry_filters": dict(metrics_a.get("entry_filters") or {}),
        # Descriptive only, and it no longer decides anything. The weekday
        # table is kept because the attribution is worth reading; the pruning
        # that used to be derived from it is gone, and the regime breakdown
        # below is what the screen acts on.
        "dow_breakdown": dow.to_dict("records"),
        # The full four-quadrant matrix per version, so the report can print
        # what the screen saw rather than a summary of it.
        "regime_profile_a": profiles.get("A"),
        "regime_profile_b": profiles.get("B"),
        "regime_version": (best or {}).get("version"),
        # The fields the handoff carries, in the schema the live supervisor
        # reads. `status` is written rather than left to be re-derived from
        # `survived`: PROMOTED and DROPPED are the two words the charter uses,
        # and a reader who has to invert a boolean to find out which one this
        # is will eventually invert it the other way.
        "status": "PROMOTED" if survived else "DROPPED",
        "optimal_regime": optimal,
        "optimal_quadrant": (best or {}).get("quadrant"),
        "regime_pf": (best or {}).get("profit_factor"),
        "regime_trade_count": (best or {}).get("trade_count"),
        # The winning quadrant's other two numbers, carried because the screen
        # decided on the PAIR (profit factor at a trade count) and a supervisor
        # reading the handoff should not have to re-derive a win rate from a
        # breakdown where it is free to apply a different trade floor.
        "regime_win_rate": (best or {}).get("win_rate"),
        "regime_net_pnl": (best or {}).get("net_pnl"),
        "kill_switch_regimes": kill_switch_regimes(optimal),
    }


# --------------------------------------------------------------------------
# The markdown report
# --------------------------------------------------------------------------
def _cell(v) -> str:
    """A pipe inside a cell ends the column early and shifts every one after
    it, silently - which on a scorecard means reading Version B's number under
    Version A's heading."""
    return str(v).replace("|", "\\|").replace("\n", " ")


def _md_table(header: list[str], rows: list[list[str]],
              align: list[str] | None = None) -> list[str]:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(_cell(h) for h in header) + " |",
           "|" + "|".join(f" {a} " for a in align) + "|"]
    out.extend("| " + " | ".join(_cell(c) for c in r) + " |" for r in rows)
    return out


def _md_scorecard(row: dict) -> list[str]:
    a, b = row.get("metrics_a") or {}, row.get("metrics_b")
    header = ["Metric", "Version A · rules", "Version B · ML-filtered", "B − A"]
    body = []
    for label, key, spec, suffix in SCORECARD_ROWS:
        va, vb = a.get(key), (b or {}).get(key)
        if b is None:
            # No `n/a` column under a "Version B" header: a column of n/a
            # invites the reading that the filter ran and produced nothing.
            cell_b, delta = "NOT RUN", "—"
        else:
            cell_b = _fmt(vb, spec, suffix=suffix)
            fa, fb = _num(va), _num(vb)
            delta = ("—" if fa is None or fb is None
                     or math.isinf(fa) or math.isinf(fb)
                     else format(fb - fa, spec) + suffix)
        body.append([label, _fmt(va, spec, suffix=suffix), cell_b, delta])
    return _md_table(header, body, ["---", "---:", "---:", "---:"])


def _regime_cell(row: dict) -> str:
    """
    The optimal-regime column, in the two states it has.

    `High Volatility / Trending (1.28)` — a quadrant cleared both bars.
    `none`                             — nothing did, so nothing is promoted
                                          and no kill switch is derived.

    A dropped configuration never prints a regime name. Naming its best
    quadrant here would read as an environment the strategy was cleared to
    trade in, which is the one thing the screen just decided against.
    """
    regime = row.get("optimal_regime")
    if not regime:
        return "none"
    return f"{regime} ({_fmt(row.get('regime_pf'))})"


def _md_regime_matrix(profile: dict | None, label: str,
                      min_pf: float = MIN_REGIME_PROFIT_FACTOR,
                      min_trades: int = MIN_REGIME_TRADES) -> list[str]:
    """
    The four-quadrant breakdown matrix for ONE version.

    Every one of the four regimes is a row, including those the strategy never
    traded in. An absent row reads as missing data when it means "this
    strategy never took a trade in a low-volatility range", which is a finding
    about the strategy rather than a gap in the table — the same rule the
    day-of-week breakdown follows for a weekday with no trades.

    The VERDICT column states which bar a quadrant missed, so a reader can see
    whether a regime failed on its edge or on its sample size. Those are fixed
    by different work.
    """
    if profile is None:
        return [f"_Version {label} was NOT RUN, so it has no regime profile._"]
    breakdown = profile.get("regime_breakdown") or {}
    if not breakdown:
        return [f"_Version {label} placed no trades in any regime "
                f"({int(profile.get('trades_unplaced', 0) or 0):,} outside "
                f"every quadrant)._"]

    body = []
    for regime in REGIMES:
        stats = breakdown.get(regime)
        if not stats:
            body.append([regime, "0", "—", "—", "—", "no trades in this regime"])
            continue
        pf = _num(stats.get("profit_factor"))
        n = int(stats.get("trade_count", 0) or 0)
        if pf is not None and pf >= min_pf and n >= min_trades:
            verdict = "**CLEARS**"
        elif pf is not None and pf < min_pf:
            verdict = f"PF < {min_pf:.2f}"
        elif n < min_trades:
            verdict = f"{n} trades < {min_trades}"
        else:
            verdict = "profit factor undefined"
        body.append([
            regime, f"{n:,}",
            _fmt(stats.get("win_rate"), ".1f", suffix="%"),
            _fmt(pf),
            _fmt(stats.get("net_pnl"), ",.0f"),
            verdict,
        ])
    lines = _md_table(
        ["Regime", "Trades", "Win rate", "PF", "Net P&L", "Verdict"], body,
        ["---", "---:", "---:", "---:", "---:", "---"])

    unplaced = int(profile.get("trades_unplaced", 0) or 0)
    if unplaced:
        lines += ["",
                  f"_{unplaced:,} trade(s) are in NO quadrant — the entry is "
                  f"outside this frame, or inside the 14-bar ADX/ATR warm-up "
                  f"where the regime is undefined — and are excluded from "
                  f"every row above._"]
    return lines


def _md_regimes(row: dict) -> list[str]:
    """Both versions' matrices, and what the screen concluded from them."""
    lines = ["**Version A · rules**", ""]
    lines += _md_regime_matrix(row.get("regime_profile_a"), "A")
    lines += ["", "**Version B · ML-filtered**", ""]
    lines += _md_regime_matrix(row.get("regime_profile_b"), "B")
    lines.append("")
    if row.get("optimal_regime"):
        lines += [
            f"**Optimal regime: {row['optimal_regime']}** "
            f"(Version {row.get('regime_version')}, PF "
            f"{_fmt(row.get('regime_pf'))} over "
            f"{int(row.get('regime_trade_count') or 0):,} trades)  ",
            f"**Kill switch — do NOT trade in:** "
            f"{', '.join(row.get('kill_switch_regimes') or []) or '(none)'}",
            "",
            "_The quadrant is the best of four, chosen IN-SAMPLE on THESE "
            "bars — the same bars Stage 2 then sweeps and Stage 3 certifies. "
            "It is a selection layer stacked on the parameter search, not a "
            "free improvement, and Gate 3's holdout retention is the only "
            "evidence that it generalised._",
        ]
    else:
        lines.append("_No quadrant cleared both bars, so no optimal regime "
                     "and no kill switch are derived._")
    return lines


def _md_dow(row: dict) -> list[str]:
    """
    The weekday attribution, DESCRIPTIVE ONLY since the regime firewall
    replaced the Drop Unprofitable Days contract.

    It reports all five weekdays whether they made money or lost it, and
    nothing downstream reads it. A losing weekday is no longer a session this
    stage prunes: which weekday loses is largely a restatement of which regime
    that weekday tends to fall in, and pruning the calendar masked the
    environment instead of naming it.
    """
    recs = row.get("dow_breakdown") or []
    if not recs:
        return ["_No trades to attribute._"]
    body = []
    for r in recs:
        wr = _num(r.get("win_rate"))
        body.append([
            str(r.get("day", "?")),
            f"{int(r.get('trades', 0) or 0):,}",
            _fmt(r.get("net_pnl"), ",.0f"),
            _fmt(None if wr is None else wr * 100.0, ".1f", suffix="%"),
            _fmt(r.get("avg_pnl"), ",.0f"),
            _fmt(r.get("profit_factor"), ".2f"),
            _fmt(r.get("pct_of_net_pnl"), ".1f", suffix="%"),
        ])
    return _md_table(
        ["Day", "Trades", "Net P&L", "Win rate", "Avg P&L", "PF",
         "Share of net P&L"], body,
        ["---", "---:", "---:", "---:", "---:", "---:", "---:"])


def build_markdown_report(strat_name: str, header: dict,
                          rows: list[dict], errors: list[dict]) -> str:
    """
    The whole screen as one markdown document. Pure - it reads no files.

    Written on every run, including one where nothing survived: the detail of
    WHY nothing survived is the most useful output that run produces, and the
    console no longer carries it.
    """
    survivors = [r for r in rows if r["survived"]]
    W = []
    W.append(f"# Stage 1 · Baseline screen — `{strat_name}`")
    W.append("")
    W.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    W.append("")
    W.append("## Run")
    W.append("")
    W.extend(_md_table(["Field", "Value"],
                       [[k, str(v)] for k, v in header.items()]))
    W.append("")
    W.append("## Result")
    W.append("")
    W.append(f"**{len(survivors)} of {len(rows)} evaluated configuration(s) "
             f"survived.**"
             + (f" {len(errors)} errored." if errors else ""))
    W.append("")
    if rows:
        body = []
        for r in sorted(rows, key=lambda r: (not r["survived"],
                                             -(_num(r["sharpe_a"]) or 0))):
            body.append([
                "**SURVIVES**" if r["survived"] else "DROPPED",
                r["symbol"], r["timeframe"],
                _fmt(r["profit_factor_a"]), _fmt(r["profit_factor_b"],
                                                 na="not run"),
                _fmt(r["sharpe_a"]),
                _fmt(r["max_drawdown_pct_a"], ".2f", suffix="%"),
                f"{int(r['trades_a'] or 0):,}",
                _regime_cell(r),
                r["reason"],
            ])
        W.extend(_md_table(
            ["Status", "Symbol", "TF", "PF (A)", "PF (B)", "Sharpe (A)",
             "Max DD (A)", "Trades (A)", "Optimal regime", "Reason"], body,
            ["---", "---", "---", "---:", "---:", "---:", "---:", "---:",
             "---", "---"]))
        W.append("")

    if errors:
        W.append("### Errors")
        W.append("")
        W.extend(_md_table(["Symbol", "TF", "Error"],
                           [[e["symbol"], e.get("timeframe", ""), f"`{e['error']}`"]
                            for e in errors]))
        W.append("")

    W.append("## Configurations")
    W.append("")
    if not rows:
        W.append("_Nothing was evaluated._")
        W.append("")
    for r in rows:
        status = "SURVIVES" if r["survived"] else "DROPPED"
        W.append(f"### {r['symbol']} · {r['timeframe']} — {status}")
        W.append("")
        W.append(f"{r['reason']}  ")
        W.append(f"_{int(r['bars']):,} bars · {r['elapsed_s']}s · Version B "
                 f"{'evaluated' if r['ml_evaluated'] else 'NOT RUN'}_")
        W.append("")
        W.append("#### Dual-version scorecard")
        W.append("")
        W.extend(_md_scorecard(r))
        W.append("")
        W.append("#### Regime breakdown · the four-quadrant screening matrix")
        W.append("")
        W.extend(_md_regimes(r))
        W.append("")
        W.append("#### Day of week · Version A, attributed by entry session "
                 "(descriptive — nothing is pruned from it)")
        W.append("")
        W.extend(_md_dow(r))
        W.append("")
        W.append("#### Entry filter audit")
        W.append("")
        W.append("```")
        W.append(describe_filters(r.get("entry_filters")))
        W.append("```")
        W.append("")
    return "\n".join(W) + "\n"


def write_markdown_report(path: Path, text: str) -> Path:
    """
    Atomic, for the same reason the JSON handoffs are.

    The report is rewritten after every configuration, so a screen killed mid
    write would otherwise leave a truncated page - which reads as a screen that
    stopped there rather than one that was interrupted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# One configuration
# --------------------------------------------------------------------------
def profile_versions(bars: pd.DataFrame, out: dict, symbol: str, tf: str,
                     strat_name: str, out_dir: Path | str | None = None,
                     quiet: bool = True) -> dict[str, dict | None]:
    """
    The four-quadrant regime profile of BOTH versions of one configuration.

    Each version is profiled from its own `BacktestResult` against the SAME
    bar frame, so the ADX/ATR classification of every bar is identical and the
    two breakdowns are comparable. Profiling Version B against a frame rebuilt
    for it would let the quadrant boundaries move between the two columns of
    the same table, and nothing would raise.

    Returns `{"A": profile, "B": profile_or_None}`. `B` is None when `--ml` was
    off — a version that never ran has no regime profile, which is a different
    statement from one that has an empty breakdown.

    `quiet` by default: this stage's console is one progress line per
    configuration, and 108 configurations x 2 versions is 216 ten-line tables.
    The matrices go into `stage1_baseline_report.md`, and each version's own
    `regime_profile_<SYMBOL>_<TF>_version_<v>.json` artifact is written
    regardless.
    """
    profiles: dict[str, dict | None] = {"A": None, "B": None}
    for label, key in (("A", "version_a"), ("B", "version_b")):
        version = out.get(key)
        if not version or version.get("result") is None:
            continue
        profiles[label] = RegimeProfiler(
            bars, version["result"], strat_name, symbol, tf,
            out_dir=out_dir, version=label.lower(), quiet=quiet
        ).generate_profile()
    return profiles


def run_symbol(symbol: str, path: Path, tf: str, params: dict,
               args: argparse.Namespace, cfg_kwargs: dict,
               tag: str = "", out_dir: Path | None = None) -> dict:
    """
    One (symbol, timeframe) configuration, defaults only, no sweep.

    Prints the RUNNING progress line and nothing else; every number it produces
    goes into the returned row, and from there into the markdown report. Raises
    on failure - the caller records it as an ERROR row and moves on.
    """
    t0 = time.time()

    bars = load_bars(symbol, tf, args.start, args.end)
    if "symbol" in bars.columns and bars["symbol"].nunique() > 1:
        raise ValueError(f"{symbol}: the lake returned an interleaved frame "
                         f"({bars['symbol'].nunique()} symbols)")

    work = "Resampling & ML fitting..." if args.ml else "Resampling & simulating..."
    print(f"{tag} RUNNING   {_pair_label(symbol, tf)} "
          f"({_human_bars(len(bars))} bars) | {work}", flush=True)

    cfg = BacktestConfig(
        initial_capital=args.capital, contracts=args.contracts,
        slippage_ticks=args.slippage_ticks, flat_by_close=args.flat_by_close,
        notes=f"stage 1 baseline {symbol} {tf}", **cfg_kwargs)

    out = run_dual_version_backtest(
        str(path), bars, freq=tf, symbol=symbol, cfg=cfg, params=params,
        threshold=args.threshold, ml=args.ml, emit_reports=False,
        strat_name=path.stem)

    a, b = out["version_a"], out["version_b"]
    metrics_a = a["metrics"]
    metrics_b = b["metrics"] if b else None

    strat_name = path.parent.name if path.stem == "strat" else path.stem
    profiles = profile_versions(bars, out, symbol, tf, strat_name, out_dir)
    survived, reason, best = screen(profiles, args.min_profit_factor,
                                    args.min_trades)

    # Version A's trades, deliberately, even when --ml ran: the weekday table
    # describes what the RULES did, and attributing it on Version B's surviving
    # trades would describe what a classifier left behind. Descriptive only -
    # nothing downstream reads it since the regime firewall replaced the
    # Drop Unprofitable Days contract.
    dow = day_of_week_breakdown(metrics_a.get("trades"))
    return _row(symbol, tf, metrics_a, metrics_b, survived, reason, dow,
                len(bars), time.time() - t0, profiles=profiles, best=best)


def survivors_leaderboard(rows: list[dict]) -> str:
    """
    STAGE 1 SURVIVORS LEADERBOARD - the table the stage ends on.

    Survivors only, sorted by the REGIME profit factor the screen decided on,
    descending — not by either version's blended factor. The blended number is
    not what advanced the configuration and ranking on it would put a contract
    with a broad mediocre edge above one with a sharp edge in a single
    environment, which inverts the question this stage now asks.

    `OPTIMAL REGIME` and `REGIME PF` are the two fields the live supervisor
    reads. The kill switch is NOT a column here: it is always the other three
    quadrants, so spelling it out would repeat the optimal regime three times
    per row and push the table past 170 characters — where a terminal wraps it
    and the alignment that makes a leaderboard readable is gone. It is printed
    in full, per survivor, in the REGIME FIREWALL block below the table, and
    carried in full on every handoff row.

    A Version B that never ran prints NOT RUN, never a dash and never 0.00: a
    comparison that was not made is not one the baseline won.
    """
    survivors = [r for r in rows if r["survived"]]

    body = []
    for r in sorted(survivors, key=lambda r: _num(r.get("regime_pf")) or 0.0,
                    reverse=True):
        body.append([
            r["symbol"], r["timeframe"],
            _fmt(r["profit_factor_a"]),
            _fmt(r["profit_factor_b"], na="NOT RUN")
            if r["ml_evaluated"] else "NOT RUN",
            r.get("optimal_quadrant") or "--",
            r.get("optimal_regime") or "none",
            _fmt(r.get("regime_pf")),
            f"{int(r.get('regime_trade_count') or 0):,}",
            f"V{r.get('regime_version') or '?'}",
        ])
    return leaderboard(
        "STAGE 1 SURVIVORS LEADERBOARD",
        ["SYMBOL", "TF", "PF (A)", "PF (B)", "QUAD", "OPTIMAL REGIME",
         "REGIME PF", "REGIME TRADES", "VER"],
        body, align=["<", "<", ">", ">", "<", "<", ">", ">", "<"],
        empty="no configuration cleared the regime firewall")


def _evaluated_line(tag: str, row: dict) -> str:
    """`[2/8] EVALUATED NQ · 5m -> Version A: 1.02 PF | Version B: 0.93 PF [SURVIVES]`"""
    pf_a = _fmt(row["profit_factor_a"], ".2f")
    pf_b = (f"{_fmt(row['profit_factor_b'], '.2f')} PF"
            if row["ml_evaluated"] else "NOT RUN")
    return (f"{tag} EVALUATED {_pair_label(row['symbol'], row['timeframe'])} "
            f"-> Version A: {pf_a} PF | Version B: {pf_b} "
            f"[{'SURVIVES' if row['survived'] else 'DROPPED'}]")


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1/5 — the REGIME-AWARE SCREENING FIREWALL. Runs "
                    "each (symbol, timeframe) configuration on default "
                    "parameters, profiles both versions into the four "
                    "volatility/trend quadrants, and keeps only the "
                    "configurations with a quadrant at profit factor "
                    f"{MIN_REGIME_PROFIT_FACTOR:.2f} or better over "
                    f"{MIN_REGIME_TRADES}+ trades. Writes each survivor's "
                    "optimal_regime, regime_pf and kill_switch_regimes for "
                    "Stage 2, and the full four-quadrant matrix per asset to "
                    "stage1_baseline_report.md.")
    p.add_argument("--strat", required=True, help="Strategy name or path")
    p.add_argument("--symbols", default=None,
                   help="NQ, a list NQ,ES,CL, or ALL")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None,
                   help="Timeframe, or a comma-separated list: "
                        "'--tf 1m,5m,15m,30m' screens each in turn. Derived "
                        "timeframes are aggregated from the 1m parquet by the "
                        "lake reader. Default: the module's, then 15m.")
    p.add_argument("--start", default=CHARTER_IS_START,
                   help=f"In-sample start, YYYY-MM-DD (default "
                        f"{CHARTER_IS_START} - the charter window)")
    p.add_argument("--end", default=CHARTER_IS_END,
                   help=f"In-sample end, YYYY-MM-DD (default {CHARTER_IS_END}). "
                        f"The years after it are the Stage 3 HOLDOUT: a screen "
                        f"that reads them selects which contracts advance on "
                        f"the bars Gate 3 then measures retention against.")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="Override a default parameter. Stage 1 runs one fixed "
                        "set across every contract; this changes that set, it "
                        "does not sweep.")
    # ON by default since 2026-08-21. The charter's Stage 1 is a DUAL
    # simulation: survival is decided on EITHER version, so a screen that ran
    # only the rules cannot say whether a configuration was dropped because
    # neither version carried it or because only one of them was ever asked.
    # The cost is real - the classifier refits once per completed trade - and
    # `--no-ml` is the deliberate way to pay less for a narrower answer.
    p.add_argument("--ml", dest="ml", action="store_true", default=True,
                   help="Run Version B, the ML-filtered twin (DEFAULT). "
                        "Survival is decided on either version, so both are "
                        "run.")
    p.add_argument("--no-ml", dest="ml", action="store_false",
                   help="Skip Version B. Expensive to run - the classifier "
                        "refits once per completed trade - but a skipped "
                        "Version B is reported as NOT RUN everywhere, never "
                        "as a Version B that scored nothing.")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Version B: P(win) at or above which an entry is kept")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--flat-by-close", action="store_true")
    p.add_argument("--min-profit-factor", type=float,
                   default=MIN_REGIME_PROFIT_FACTOR,
                   help=f"Profit-factor bar ONE regime quadrant must clear "
                        f"(default {MIN_REGIME_PROFIT_FACTOR:.2f}). Above the "
                        f"1.00 break-even a blended screen would use, because "
                        f"the quadrant is the best of four and a bar cleared "
                        f"by a hair is a bar cleared by selection.")
    p.add_argument("--min-trades", type=int, default=MIN_REGIME_TRADES,
                   help=f"Trade floor the winning QUADRANT must clear on its "
                        f"own trade count, not the configuration's total "
                        f"(default {MIN_REGIME_TRADES})")
    p.add_argument("--out-dir", default=None,
                   help="Override <BT_ARTIFACTS>/pipeline/<strategy>/")
    add_filter_args(p)
    return p


def stage2_command(strat: str, pairs: list[dict], start: str | None,
                   end: str | None) -> list[str]:
    """
    The exact Stage 2 command, over the surviving pairs and nothing else.

    `--symbols` and `--tf` are the two axes Stage 2 accepts, so ragged
    survivors (NQ at 5m and 15m, GC at 15m only) can only be expressed as the
    cross product of the two unions - which is a SUPERSET of what survived.
    The caller says so out loud when it is: a command that quietly sweeps GC at
    5m after Stage 1 dropped it is how a contract with no baseline edge gets
    parameters fitted to it anyway.
    """
    symbols = sorted({p["symbol"] for p in pairs})
    tfs = sorted({p["tf"] for p in pairs},
                 key=lambda t: (_TF_MINUTES.get(t, 10 ** 6), t))
    if not pairs:
        return ["Stage 2 — nothing survived, so there is nothing to sweep."]
    lines = [
        "Stage 2 — sweep the surviving configurations in-sample:",
        "",
        f"  python3 backtest/scan.py --strat {strat} "
        f"--symbols {','.join(symbols)} \\",
        f"      --tf {','.join(tfs)} "
        f"--start {start or '2013-01-01'} "
        f"--end {end or '2022-12-31'}",
    ]
    profiled = [p for p in pairs if p.get("optimal_regime")]
    if profiled:
        lines += [
            "",
            f"  {SURVIVORS_FILE} carries each pair's optimal_regime, "
            f"regime_pf and",
            f"  kill_switch_regimes — {len(profiled)} of {len(pairs)} "
            f"pair(s). Stage 2 sweeps the WHOLE",
            "  window; the regime is a live-execution instruction, not a mask "
            "on the sweep.",
            "  Masking the sweep to the winning quadrant would fit parameters "
            "to a subset",
            "  that was itself chosen as the best of four on these same bars.",
        ]
    if len(pairs) < len(symbols) * len(tfs):
        kept = ", ".join(f"{p['symbol']}·{p['tf']}" for p in pairs)
        lines += [
            "",
            f"  [!] The survivors are ragged, so that command is a SUPERSET: "
            f"{len(symbols)}x{len(tfs)}",
            f"      = {len(symbols) * len(tfs)} combinations for "
            f"{len(pairs)} that survived. What survived is:",
            f"      {kept}",
        ]
    return lines


# --------------------------------------------------------------------------
# The handoff lists. Three shapes over the same rows, named rather than built
# inline in `main` so the schema the charter fixes can be checked without a
# lake, a strategy module or a simulation.
# --------------------------------------------------------------------------
def surviving_pairs_from(rows: list[dict]) -> list[dict]:
    """
    The PROMOTED configurations, in the schema Stage 2 and the live supervisor
    read.

    `kill_switch_regimes` is DERIVED from `optimal_regime` rather than
    measured, and that is deliberate. A quadrant that failed the profit-factor
    bar and a quadrant the strategy never traded in are the same instruction to
    a supervisor - stand down - and treating "no evidence" as "permitted" is
    the reading that puts a contract into the one environment nobody sampled.
    """
    return [{"symbol": r["symbol"],
             "tf": r["timeframe"],
             # WHICH version carried it. Survival is decided on either, so a
             # survivor with no version recorded is a pair nobody can
             # reproduce: Version B is a classifier fitted on these same bars,
             # and a pair that only B cleared is a different claim from one
             # the rules carried on their own.
             "version": r.get("regime_version"),
             "status": "PROMOTED",
             "optimal_regime": r["optimal_regime"],
             "quadrant": r.get("optimal_quadrant"),
             # The winning quadrant's own metrics, all four of them. The screen
             # decided on the PAIR (profit factor at a trade count); recording
             # the factor alone leaves the supervisor to re-derive the count
             # from a breakdown where it is free to apply a different floor
             # than the one that chose the name.
             "regime_pf": r["regime_pf"],
             "regime_trade_count": r.get("regime_trade_count"),
             "regime_win_rate": r.get("regime_win_rate"),
             "regime_net_pnl": r.get("regime_net_pnl"),
             "kill_switch_regimes": list(r.get("kill_switch_regimes") or [])}
            for r in rows if r["survived"]]


def dropped_from(rows: list[dict]) -> list[dict]:
    """
    The DROPPED configurations, with the reason they were.

    DROPPED is written as a word, not left to be inferred from absence, and
    `optimal_regime` is explicitly None beside it: a configuration that cleared
    no quadrant has no environment, and a best-effort second place here would
    read as one it was cleared to trade in.
    """
    return [{"symbol": r["symbol"], "timeframe": r["timeframe"],
             "status": "DROPPED",
             "optimal_regime": None,
             "kill_switch_regimes": [],
             "reason": r["reason"], "profit_factor": r["profit_factor_a"]}
            for r in rows if not r["survived"]]


def screen_results_from(rows: list[dict]) -> list[dict]:
    """
    Every configuration EVALUATED, promoted and dropped alike, in one flat list
    with the same keys - the table Stage 1 ends on, and what
    `backtest/discord_reporter.py --stage 1` formats.

    A separate list from `surviving_pairs` and `dropped` because those two are
    handoffs (Stage 2 reads the first; the second is the audit trail for why a
    contract is not in it) and this one is a REPORT. Joining a leaderboard back
    together from two lists with different keys is how a dropped configuration
    ends up printed under a promoted heading.
    """
    return [{"symbol": r["symbol"],
             "tf": r["timeframe"],
             "status": r.get("status"),
             "version": r.get("regime_version"),
             "optimal_regime": r.get("optimal_regime"),
             "quadrant": r.get("optimal_quadrant"),
             "regime_pf": r.get("regime_pf"),
             "regime_trade_count": r.get("regime_trade_count"),
             "regime_win_rate": r.get("regime_win_rate"),
             "regime_net_pnl": r.get("regime_net_pnl"),
             "kill_switch_regimes": list(r.get("kill_switch_regimes") or []),
             "profit_factor_a": r.get("profit_factor_a"),
             "profit_factor_b": r.get("profit_factor_b"),
             "trades_a": r.get("trades_a"),
             "trades_b": r.get("trades_b"),
             "ml_evaluated": bool(r.get("ml_evaluated")),
             "reason": r.get("reason")}
            for r in rows]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path = resolve_strategy(args.strat)
    strat_name = path.parent.name if path.stem == "strat" else path.stem
    params = dict(parse_param(p) for p in args.param)

    try:
        _fn, info = load_strategy(path, params)
    except Exception as e:                                        # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        cfg_kwargs = filter_config_kwargs(args)
    except Exception as e:                                        # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    symbols = parse_symbols(args.symbols, info.get("symbols"))
    try:
        timeframes = parse_timeframes(args.tf, info.get("timeframe"))
    except ValueError as e:
        print(f"ValueError: {e}", file=sys.stderr)
        return 1
    bound = info.get("bound_params") or {}

    out_dir = pipeline_dir(strat_name, args.out_dir, create=True)
    report_path = out_dir / BASELINE_REPORT_FILE
    criterion = (f"optimal_regime_PF >= {args.min_profit_factor:.2f} AND "
                 f"optimal_regime_trade_count >= {args.min_trades} in ANY of "
                 f"the 4 regime quadrants, on either version")

    # Symbol-major, so the progress counter reads the way an operator watches
    # it: one contract taken through every timeframe before the next starts.
    pairs = [(sym, tf) for sym in symbols for tf in timeframes]

    charter_window = (args.start == CHARTER_IS_START
                      and args.end == CHARTER_IS_END)
    window_source = ("charter default (holdout untouched)" if charter_window
                     else "operator override — check it against the Stage 3 "
                          "holdout before reading Gate 3")

    header = {
        "Strategy": strat_name,
        "Module": str(path),
        "Window": f"{args.start or 'lake start'} → {args.end or 'lake end'} "
                  f"· {window_source}",
        "Symbols": f"{len(symbols)} · {', '.join(symbols)}",
        "Timeframes": f"{len(timeframes)} · {', '.join(timeframes)}",
        "Configurations": len(pairs),
        "Parameters": bound or "(module defaults)",
        "Parameter source": "module DEFAULT_PARAMS with --param over them "
                            "(NOT swept — that is Stage 2)",
        "Screen": criterion,
        "Regimes": " · ".join(REGIMES),
        "Regime classification": "ADX(14) > 25 is Trending; ATR(14) above the "
                                 "contract's own median ATR is High "
                                 "Volatility. Both thresholds are per "
                                 "(symbol, timeframe) — the ATR median is "
                                 "computed on THESE bars.",
        "Version B": ("evaluated" if args.ml
                      else "NOT RUN (--no-ml). Survival was decided on Version "
                           "A alone."),
        "Costs": f"{args.slippage_ticks:g} tick slippage each way, "
                 f"{args.contracts} contract(s), "
                 f"${args.capital:,.0f} capital",
        "Entry filters": (f"news={cfg_kwargs['news_filter']} "
                          f"exclude_days={cfg_kwargs['exclude_days']} "
                          f"(explicit CLI filters only — this stage decides "
                          f"no calendar pruning of its own)"),
        "Day-of-week pruning": "REMOVED. The weekday table is descriptive and "
                               "nothing downstream reads it; the regime "
                               "quadrant replaced it as the screen.",
    }

    print(stage_banner(1, strat_name,
                       f"{len(symbols)} contract(s) × {len(timeframes)} "
                       f"timeframe(s) = {len(pairs)} configuration(s) · "
                       f"{args.start or 'lake start'} → "
                       f"{args.end or 'lake end'}"))
    print(f"  parameters : {bound or '(module defaults)'}")
    print(f"  screen     : {criterion}")
    print(f"  window     : {args.start} -> {args.end} · {window_source}")
    print(f"  regimes    : ADX(14)>25 = Trending · ATR(14) > per-contract "
          f"median = High Volatility")
    print(f"  Version B  : {'evaluated' if args.ml else 'NOT RUN (--no-ml)'}")
    print(f"  report     : {report_path}")
    print()

    rows, errors = [], []
    total = len(pairs)
    for i, (sym, tf) in enumerate(pairs, 1):
        tag = f"[{i}/{total}]"
        try:
            row = run_symbol(sym, path, tf, params, args, cfg_kwargs, tag,
                             out_dir=out_dir)
            rows.append(row)
            print(_evaluated_line(tag, row), flush=True)
        except Exception as e:                                # noqa: BLE001
            # One bad configuration does not end the screen. Recorded as an
            # ERROR row rather than as one that produced nothing: those read
            # identically in a survivor list and mean opposite things.
            errors.append({"symbol": sym, "timeframe": tf,
                           "error": f"{type(e).__name__}: {e}"})
            print(f"{tag} ERROR     {_pair_label(sym, tf)} -> "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
        # Rewritten from scratch after every configuration, so a screen killed
        # at 14 of 108 leaves a complete report of 14.
        try:
            write_markdown_report(
                report_path,
                build_markdown_report(strat_name, header, rows, errors))
        except Exception as e:                                # noqa: BLE001
            print(f"[!] the markdown report was not written: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)

    by_tf = {}
    for tf in timeframes:
        tf_rows = [r for r in rows if r["timeframe"] == tf]
        by_tf[tf] = {
            "surviving": [r["symbol"] for r in tf_rows if r["survived"]],
            "dropped": [{"symbol": r["symbol"], "reason": r["reason"],
                         "profit_factor": r["profit_factor_a"]}
                        for r in tf_rows if not r["survived"]],
        }

    # WHERE each version's quadrant labels came from, one entry per
    # configuration that produced a profile - so a mixed run, some pairs
    # cached and some not, says so per pair rather than under a single banner
    # that would be true of only half of it.
    regime_sources: dict[str, str] = {}
    for r in rows:
        for label in ("a", "b"):
            prof = r.get(f"regime_profile_{label}")
            if prof:
                regime_sources[
                    f"{r['symbol']}_{r['timeframe']}_version_{label}"] = (
                    prof.get("regime_source", "unrecorded"))

    surviving_pairs = surviving_pairs_from(rows)
    # And the symbol union, kept because `scan.py` defaults `--symbols` to it.
    # It is labelled as a union so it is never read as "survived at 15m" when
    # it survived at 1m only; `by_timeframe` and `surviving_pairs` are where
    # that question is answered.
    survivors = sorted({p["symbol"] for p in surviving_pairs})
    dropped = dropped_from(rows)
    screen_results = screen_results_from(rows)

    dest = write_stage(out_dir / SURVIVORS_FILE, 1, strat_name, {
        # Singular when one timeframe was screened, so a downstream reader that
        # expects the pre-multi-timeframe shape still finds what it looks for;
        # None when several were, because there is no single answer and a
        # plausible-looking wrong one is worse than an absent one.
        "timeframe": timeframes[0] if len(timeframes) == 1 else None,
        "timeframes": timeframes,
        "by_timeframe": by_tf,
        "surviving_is_union_across_timeframes": len(timeframes) > 1,
        "start": args.start,
        "end": args.end,
        # Which window ran and who chose it. The charter fixes
        # 2013-01-01..2022-12-31 and the years after it are the Stage 3
        # holdout; a screen run over a wider window picked its survivors on
        # bars Gate 3 later measures retention against, and that has to be
        # legible from the handoff rather than reconstructed from a shell
        # history.
        "in_sample_window": {"start": args.start, "end": args.end,
                             "charter_default": bool(charter_window),
                             "source": window_source,
                             "charter": {"start": CHARTER_IS_START,
                                         "end": CHARTER_IS_END}},
        "params": bound,
        "params_source": "module DEFAULT_PARAMS with --param over them",
        "criterion": criterion,
        "min_profit_factor": args.min_profit_factor,
        "min_trades": args.min_trades,
        "ml_evaluated": bool(args.ml),
        "entry_filters": cfg_kwargs,
        "surviving_pairs": surviving_pairs,
        "surviving": survivors,
        "screen_results": screen_results,
        "evaluated": len(rows),
        "promoted": len(surviving_pairs),
        "regime_screen": {
            "regimes": list(REGIMES),
            "min_profit_factor": float(args.min_profit_factor),
            "min_trades": int(args.min_trades),
            "rule": criterion,
            "classification": ("ADX(14) > 25 is Trending; ATR(14) above the "
                               "contract's own median ATR is High Volatility. "
                               "Both thresholds are per (symbol, timeframe)."),
            # WHERE the quadrant labels came from, per configuration, and it is
            # not decoration. A cached label is drawn against a threshold
            # pinned to the in-sample window; a recomputed one is drawn against
            # the median of whatever window the caller asked for. The two put
            # the same bar in different quadrants, so a handoff that records a
            # survivor's optimal_regime without recording which threshold drew
            # it is not reproducible.
            "regime_source": regime_sources,
            "regime_source_note": (
                "precomputed_cache = read from "
                "<lake>/regimes/{SYMBOL}_{TF}_regime.parquet, whose volatility "
                "threshold is the median ATR(14) over the cache's own "
                "in-sample window and does NOT move with this run's --start / "
                "--end. recomputed_live = no cached quadrant on the bars, so "
                "the profiler took the median ATR of this window instead. "
                "Build the cache with scripts/precompute_regimes.py."),
            # Written as a field rather than left to a reader to infer. The
            # quadrant is the best of four picked on the bars Stage 2 then
            # optimises over and Stage 3 certifies, exactly as the weekday
            # pruning it replaced was, and a gate verdict on a strategy scoped
            # to one regime has to carry the fact that the regime was chosen on
            # the bars being certified.
            "selected_in_sample": True,
            "note": ("The optimal regime is the best of four quadrants, "
                     "selected on THIS in-sample window, which Stage 2 then "
                     "optimises over and Stage 3 certifies. It is a layer of "
                     "in-sample selection on top of the parameter sweep, not "
                     "a free improvement, and the retention Gate 3 measures "
                     "is the only evidence that it generalised."),
            "applied_to_stage2": False,
            "applied_to_stage2_note": (
                "Stage 2 sweeps the WHOLE in-sample window, not the winning "
                "quadrant. Masking the sweep to a subset that was itself "
                "chosen as the best of four on these same bars would stack a "
                "second selection layer under the first. The regime is a "
                "live-execution instruction for the supervisor."),
        },
        "day_of_week_pruning": {
            "enabled": False,
            "note": ("REMOVED. Stage 1 no longer blacklists weekdays. The "
                     "day-of-week table in the report is descriptive and "
                     "nothing downstream reads it; `stage1_exclude_days` "
                     "therefore yields an empty mapping and Stage 2 sweeps "
                     "the whole week unless an explicit --exclude-days is "
                     "passed to it."),
        },
        "dropped": dropped,
        "errors": errors,
        "report": str(report_path),
        "assets": rows,
    })

    W = 78
    print("\n" + "=" * W)
    print(f"STAGE 1 RESULT · {len(surviving_pairs)}/{total} configuration(s) "
          f"survived")
    print("=" * W)
    print(survivors_leaderboard(rows))
    if not surviving_pairs:
        print("\n  Nothing cleared the regime firewall. That is a result about")
        print("  the idea on these contracts — there is no environment in")
        print("  which it works — not a run to repeat with different")
        print("  parameters until something does.")
    else:
        print(f"\n  REGIME FIREWALL · {len(surviving_pairs)} surviving "
              f"pair(s), each scoped to ONE environment:")
        for pair in surviving_pairs:
            print(f"    {pair['symbol']:<6}{pair['tf']:<5}TRADE ONLY IN  "
                  f"{pair['optimal_regime']}  (PF {_fmt(pair['regime_pf'])})")
            # In full, never as a count. "3 regimes" beside a symbol is not an
            # instruction a supervisor can act on.
            for killed in pair["kill_switch_regimes"]:
                print(f"    {'':<11}KILL SWITCH    {killed}")
        print("    [!] The quadrant is the BEST OF FOUR, chosen on THESE "
              "bars, which Stage 2 then\n"
              "        sweeps and Stage 3 certifies. It is an in-sample "
              "selection layer stacked on\n"
              "        the parameter search, recorded as such in the handoff. "
              "Gate 3's holdout\n"
              "        retention is the only evidence that it generalised.")

    dropped_n = len(rows) - len(surviving_pairs)
    print(f"\n  {dropped_n} dropped, {len(errors)} errored — every one with "
          f"its reason and its full four-quadrant matrix in the report.")
    print(f"  report    → {report_path}")
    print(f"  survivors → {dest}")
    print(f"  profiles  → {out_dir}/regime_profile_<SYMBOL>_<TF>_version_"
          f"<a|b>.json")

    lines = stage2_command(args.strat, surviving_pairs, args.start, args.end)
    lines += [
        "",
        "Post this leaderboard to Discord (reads the handoff, recomputes "
        "nothing):",
        "",
        f"  python3 backtest/discord_reporter.py --stage 1 "
        f"--strat {strat_name} \\",
        f"      --survivors {dest}",
    ]
    print(next_step(lines))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
