"""
backtest/baseline.py - STAGE 1 of 5: does this idea carry on this contract?

Location: ~/src/trading/backtest/baseline.py

Runs Version A (rules) and, with `--ml`, Version B (the same signals, ML
filtered) on the strategy's DEFAULT parameters, one independent simulation per
(symbol, timeframe) CONFIGURATION, and answers one question per configuration:
is there anything here at all. Configurations that clear the screen are written
to `surviving_assets.json` as exact (symbol, timeframe) pairs, and Stage 2
sweeps only those.

    python3 backtest/baseline.py --strat ema_trend_filter --symbols ALL --tf 15m \\
        --start 2013-01-01 --end 2022-12-31

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

**Profit factor is the screen, not Sharpe.** PF < 1.00 means the strategy took
in less than it gave back after costs, on this contract. That is a fact about
the symbol. A Sharpe screen at this stage would drop a contract for a lumpy
equity path, which is the complaint Gate 1 stopped making when its Sharpe
threshold was demoted to informational.

**The screen also carries a trade floor.** A profit factor over eleven trades
is not a measurement, and 1.00 is a bar it clears by accident often enough to
matter across a 108-cell screen. `MIN_TRADES` (30) is deliberately far below
Gate 1's 100 - Stage 1 is asking whether an idea is worth sweeping, not
certifying it - but it is not zero.

**A configuration survives on EITHER version** (`max(PF_a, PF_b)`), each held
to the trade floor on its own trade count. This is a change from the earlier
rule, which decided survival on Version A alone even when `--ml` was on, and it
is a real loosening: Version B's classifier is fitted on these same bars, so
advancing a configuration on B advances it on the fitted side of a comparison
the Dual-Version Mandate says is only settled out-of-sample. What that buys is
that a filter which rescues a marginal contract gets its parameters swept in
Stage 2 rather than being dropped here. What it costs is that a Stage 1
survivor is no longer necessarily a contract with an unfiltered edge. Which
version carried a configuration is recorded in its `reason` and in the report,
so the distinction survives into Stage 2 rather than being averaged away.

The day-of-week table
---------------------
Every configuration gets P&L, win rate and trade count by weekday in the
markdown report, attributed by ENTRY session (see
`backtest.report.day_of_week_breakdown`). It is DESCRIPTIVE. A losing weekday
is a candidate for `--exclude-days` in a later run, not a parameter this script
applies: excluding the days that lost money in-sample and re-scoring on those
same bars is circular, and the Sharpe it produces is not a measurement. The
suggestion is written with the trade count beside it so a reader can see
whether the row is an edge or eleven trades.
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
from backtest.pipeline import (BASELINE_REPORT_FILE, SURVIVORS_FILE,  # noqa: E402
                               next_step, pipeline_dir, stage_banner,
                               write_stage)
from backtest.report import (day_of_week_breakdown,                # noqa: E402
                             losing_weekdays)
from backtest.run import (load_bars, parse_param, parse_symbols,    # noqa: E402
                          parse_timeframes, resolve_strategy)

# The survival bar. 1.00 is break-even after costs, not a comfort margin - the
# same number Gate 1 binds on, so a configuration cannot survive Stage 1 on a
# profit factor Gate 1 would later reject.
MIN_PROFIT_FACTOR = 1.00

# The trade floor the surviving version must clear on its OWN trade count.
# Below Gate 1's 100 on purpose: this stage decides what is worth sweeping, not
# what is worth trading.
MIN_TRADES = 30

# Below this, the day-of-week row is reported but never suggested for exclusion.
DOW_MIN_TRADES = 20

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


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------
def screen(metrics: dict | None,
           min_profit_factor: float = MIN_PROFIT_FACTOR,
           metrics_b: dict | None = None,
           min_trades: int = MIN_TRADES) -> tuple[bool, str]:
    """
    Did this configuration carry the edge? Returns `(survived, reason)`.

    A configuration survives when EITHER version's profit factor is at or above
    `min_profit_factor` AND that same version produced at least `min_trades`
    trades. The two bars are applied to one version at a time on purpose: a
    Version B profit factor of 1.4 over nine surviving trades is not rescued by
    Version A's four hundred, and pairing the best PF with the largest trade
    count would let exactly that through.

    A run that produced no trades is dropped with its own reason rather than
    being folded into "profit factor too low". No trades is a fact about the
    strategy on this contract - the signals never fired - and a reader chasing
    a 0.00 profit factor would go looking for losses that do not exist.

    `metrics_b=None` means Version B did not run, and only Version A is
    considered. A skipped comparison is not one the baseline won.
    """
    if not metrics or not metrics.get("ok", True):
        return False, f"run failed: {(metrics or {}).get('error', 'unknown')}"

    views = [("A", metrics)]
    if metrics_b is not None and metrics_b.get("ok", True):
        views.append(("B", metrics_b))

    scored = []                       # (pf, label, n) for a defined PF
    counts = []                       # n per version, defined or not
    for label, m in views:
        n = int(m.get("trade_count", 0) or 0)
        counts.append(n)
        pf = m.get("profit_factor")
        # `_profit_factor` returns inf for a run with no losing trades. That is
        # not a missing value, and it survives.
        if pf is not None and not pd.isna(pf):
            scored.append((float(pf), label, n))

    if not any(counts):
        return False, "no trades - the signals never fired on this contract"

    clearing = [s for s in scored
                if s[0] >= min_profit_factor and s[2] >= int(min_trades)]
    if clearing:
        pf, label, n = max(clearing)
        return True, f"Version {label} profit factor {pf:.2f} over {n:,} trades"

    if not scored:
        return False, f"profit factor undefined over {max(counts):,} trades"

    pf, label, n = max(scored)
    if pf < min_profit_factor:
        return False, (f"best profit factor {pf:.2f} (Version {label}) < "
                       f"{min_profit_factor:.2f} over {n:,} trades")
    return False, (f"Version {label} profit factor {pf:.2f} but only {n:,} "
                   f"trades < {int(min_trades)}")


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
         bars: int, elapsed: float) -> dict:
    a, b = _scalars(metrics_a), _scalars(metrics_b)

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
        "dow_breakdown": dow.to_dict("records"),
        "losing_weekdays": losing_weekdays(dow, DOW_MIN_TRADES),
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


def _md_dow(row: dict) -> list[str]:
    recs = row.get("dow_breakdown") or []
    if not recs:
        return ["_No trades to attribute._"]
    body = []
    for r in recs:
        thin = " ⚠ thin" if int(r.get("trades", 0) or 0) < DOW_MIN_TRADES else ""
        wr = _num(r.get("win_rate"))
        body.append([
            str(r.get("day", "?")),
            f"{int(r.get('trades', 0) or 0):,}",
            _fmt(r.get("net_pnl"), ",.0f"),
            _fmt(None if wr is None else wr * 100.0, ".1f", suffix="%"),
            _fmt(r.get("avg_pnl"), ",.0f"),
            _fmt(r.get("profit_factor"), ".2f"),
            _fmt(r.get("pct_of_net_pnl"), ".1f", suffix="%") + thin,
        ])
    lines = _md_table(
        ["Day", "Trades", "Net P&L", "Win rate", "Avg P&L", "PF",
         "Share of net P&L"], body,
        ["---", "---:", "---:", "---:", "---:", "---:", "---:"])
    losers = row.get("losing_weekdays") or []
    if losers:
        names = ", ".join(str(r.get("day")) for r in recs
                          if int(r.get("weekday", -1)) in losers)
        lines += ["",
                  f"_{names} lost money on this configuration. That is a "
                  f"CANDIDATE for `--exclude-days "
                  f"{','.join(str(d) for d in losers)}`, not a result: "
                  f"excluding the days that lost in-sample and re-scoring the "
                  f"same bars proves nothing._"]
    return lines


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
                r["reason"],
            ])
        W.extend(_md_table(
            ["Status", "Symbol", "TF", "PF (A)", "PF (B)", "Sharpe (A)",
             "Max DD (A)", "Trades (A)", "Reason"], body,
            ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---"]))
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
        W.append("#### Day of week · Version A, attributed by entry session")
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
def run_symbol(symbol: str, path: Path, tf: str, params: dict,
               args: argparse.Namespace, cfg_kwargs: dict,
               tag: str = "") -> dict:
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

    dow = day_of_week_breakdown(metrics_a.get("trades"))
    survived, reason = screen(metrics_a, args.min_profit_factor, metrics_b,
                              args.min_trades)
    return _row(symbol, tf, metrics_a, metrics_b, survived, reason, dow,
                len(bars), time.time() - t0)


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
        description="Stage 1/5 — baseline survival screen on default "
                    "parameters. Drops (symbol, timeframe) configurations "
                    "whose best profit factor is below 1.00, writes the "
                    "surviving pairs for Stage 2, and writes the full detail "
                    "to stage1_baseline_report.md.")
    p.add_argument("--strat", required=True, help="Strategy name or path")
    p.add_argument("--symbols", default=None,
                   help="NQ, a list NQ,ES,CL, or ALL")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None,
                   help="Timeframe, or a comma-separated list: "
                        "'--tf 1m,5m,15m,30m' screens each in turn. Derived "
                        "timeframes are aggregated from the 1m parquet by the "
                        "lake reader. Default: the module's, then 15m.")
    p.add_argument("--start", default=None, help="In-sample start, YYYY-MM-DD")
    p.add_argument("--end", default=None, help="In-sample end, YYYY-MM-DD")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="Override a default parameter. Stage 1 runs one fixed "
                        "set across every contract; this changes that set, it "
                        "does not sweep.")
    p.add_argument("--ml", action="store_true",
                   help="Also run Version B. Off by default: the classifier "
                        "refits once per completed trade, and 27 contracts of "
                        "that is hours.")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Version B: P(win) at or above which an entry is kept")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--flat-by-close", action="store_true")
    p.add_argument("--min-profit-factor", type=float, default=MIN_PROFIT_FACTOR,
                   help=f"Survival bar on max(PF A, PF B) "
                        f"(default {MIN_PROFIT_FACTOR:.2f})")
    p.add_argument("--min-trades", type=int, default=MIN_TRADES,
                   help=f"Trade floor the surviving version must clear on its "
                        f"own trade count (default {MIN_TRADES})")
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
    criterion = (f"max(profit_factor_a, profit_factor_b) >= "
                 f"{args.min_profit_factor:.2f} with >= {args.min_trades} "
                 f"trades on the version that cleared it")

    # Symbol-major, so the progress counter reads the way an operator watches
    # it: one contract taken through every timeframe before the next starts.
    pairs = [(sym, tf) for sym in symbols for tf in timeframes]

    header = {
        "Strategy": strat_name,
        "Module": str(path),
        "Window": f"{args.start or 'lake start'} → {args.end or 'lake end'}",
        "Symbols": f"{len(symbols)} · {', '.join(symbols)}",
        "Timeframes": f"{len(timeframes)} · {', '.join(timeframes)}",
        "Configurations": len(pairs),
        "Parameters": bound or "(module defaults)",
        "Parameter source": "module DEFAULT_PARAMS with --param over them "
                            "(NOT swept — that is Stage 2)",
        "Screen": criterion,
        "Version B": "evaluated" if args.ml else "NOT RUN (--ml is off)",
        "Costs": f"{args.slippage_ticks:g} tick slippage each way, "
                 f"{args.contracts} contract(s), "
                 f"${args.capital:,.0f} capital",
        "Entry filters": (f"news={cfg_kwargs['news_filter']} "
                          f"exclude_days={cfg_kwargs['exclude_days']}"),
    }

    print(stage_banner(1, strat_name,
                       f"{len(symbols)} contract(s) × {len(timeframes)} "
                       f"timeframe(s) = {len(pairs)} configuration(s) · "
                       f"{args.start or 'lake start'} → "
                       f"{args.end or 'lake end'}"))
    print(f"  parameters : {bound or '(module defaults)'}")
    print(f"  screen     : {criterion}")
    print(f"  Version B  : {'evaluated' if args.ml else 'NOT RUN (--ml is off)'}")
    print(f"  report     : {report_path}")
    print()

    rows, errors = [], []
    total = len(pairs)
    for i, (sym, tf) in enumerate(pairs, 1):
        tag = f"[{i}/{total}]"
        try:
            row = run_symbol(sym, path, tf, params, args, cfg_kwargs, tag)
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

    # The exact configurations that survived, as (symbol, tf) pairs. This is
    # the handoff Stage 2 should sweep.
    surviving_pairs = [{"symbol": r["symbol"], "tf": r["timeframe"]}
                       for r in rows if r["survived"]]
    # And the symbol union, kept because `scan.py` defaults `--symbols` to it.
    # It is labelled as a union so it is never read as "survived at 15m" when
    # it survived at 1m only; `by_timeframe` and `surviving_pairs` are where
    # that question is answered.
    survivors = sorted({p["symbol"] for p in surviving_pairs})
    dropped = [{"symbol": r["symbol"], "timeframe": r["timeframe"],
                "reason": r["reason"], "profit_factor": r["profit_factor_a"]}
               for r in rows if not r["survived"]]

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
        "params": bound,
        "params_source": "module DEFAULT_PARAMS with --param over them",
        "criterion": criterion,
        "min_profit_factor": args.min_profit_factor,
        "min_trades": args.min_trades,
        "ml_evaluated": bool(args.ml),
        "entry_filters": cfg_kwargs,
        "surviving_pairs": surviving_pairs,
        "surviving": survivors,
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
    if surviving_pairs:
        print(f"  {'SYMBOL':<8}{'TF':<6}{'PF (A)':>8}{'PF (B)':>9}"
              f"{'SHARPE':>9}{'MAX DD':>9}{'TRADES':>9}")
        for r in sorted((r for r in rows if r["survived"]),
                        key=lambda r: -(_num(r["sharpe_a"]) or 0)):
            print(f"  {r['symbol']:<8}{r['timeframe']:<6}"
                  f"{_fmt(r['profit_factor_a']):>8}"
                  f"{_fmt(r['profit_factor_b'], na='—'):>9}"
                  f"{_fmt(r['sharpe_a']):>9}"
                  f"{_fmt(r['max_drawdown_pct_a'], '.1f', suffix='%'):>9}"
                  f"{int(r['trades_a'] or 0):>9,}")
    else:
        print("  Nothing survived. That is a result about the idea on these")
        print("  contracts, not a run to repeat with different parameters")
        print("  until something does.")
    dropped_n = len(rows) - len(surviving_pairs)
    print(f"\n  {dropped_n} dropped, {len(errors)} errored — every one with "
          f"its reason in the report.")
    print(f"  report    → {report_path}")
    print(f"  survivors → {dest}")

    print(next_step(stage2_command(args.strat, surviving_pairs,
                                   args.start, args.end)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
