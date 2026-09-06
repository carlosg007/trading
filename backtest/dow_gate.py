#!/usr/bin/env python3
"""
backtest/dow_gate.py - STAGE 4.5 of 5: the worst session of the week, named.

Location: ~/src/trading/backtest/dow_gate.py

Sits between Stage 4 (the lifecycle run) and Stage 5 (promotion). It profiles
every certified configuration by WEEKDAY, isolates the worst one, runs a
counterfactual with that weekday's ENTRIES suppressed, and writes the verdict
where the live execution loop can read it.

    python3 backtest/dow_gate.py --strat t3_braid_scalp_20260823 --tf 15m

WHAT THIS STAGE DECIDES, AND WHAT IT DELIBERATELY DOES NOT
----------------------------------------------------------
**It prunes nothing.** Every configuration handed to it advances to Stage 5,
whatever the weekday table says - that is the promotion criterion, written
down rather than left implicit, and `promotion_gate()` is the one function
that expresses it. The stage produces an INSTRUCTION for the live supervisor
(`blocked_weekday`), never a verdict about the strategy. This is the same
division Stage 1's regime firewall draws: the quadrant is an instruction for
the live loop and never a mask on the search.

**It is an IN-SAMPLE selection layer and the handoff says so.** The window is
Stage 4's - it spans the Stage 3 holdout - so the weekday is chosen on bars
that have already been optimised over and certified against. Nothing here is
unseen evidence, `is_certification` is false on every file this stage writes,
and `selected_in_sample` records it. Which weekday loses is also, largely, a
restatement of which REGIME that weekday tends to fall in; Stage 1's own
day-of-week table was demoted to descriptive for exactly that reason
(2026-08-20). This stage re-instates the calendar cut as a LIVE gate at the
operator's instruction, and the honest reading of the counterfactual below is
"what this would have looked like had the day been cut", not "what cutting it
will earn".

THE WORST WEEKDAY, AND HOW IT IS PICKED
---------------------------------------
`select_worst_weekday` ranks Monday-Friday on EXPECTANCY (mean net P&L per
trade) ascending, breaking ties on win rate and then on the share of the
deepest drawdown episode the weekday contributed. Three rules bind it:

  * **Only Mon-Fri, and only at or above `--min-trades` (default 20).** The
    floor is `backtest.report.losing_weekdays`', for its reason: over a
    16-year lake a weekday holds a fifth of the sample, and condemning a day
    on eight trades is precisely how a day-of-week filter manufactures an
    in-sample Sharpe. A Saturday row is a bug worth seeing, not a session to
    cut.
  * **The worst weekday is always IDENTIFIED; it is BLOCKED only when its
    expectancy is negative.** Five profitable weekdays have a worst one too,
    and blocking it removes realised edge in exchange for nothing. The
    identification and the block decision are recorded as separate fields
    with the rule that produced each, so "no weekday was blocked" can never
    be read as "the stage did not run".
  * **`--block-worst-always` overrides that**, and is recorded as
    `block_rule: "worst_always"`. It exists because the instruction "block the
    worst day" is a legitimate operating policy; what it must not be is the
    silent default.

ATTRIBUTION IS BY THE ENTRY SESSION, AND THAT IS NOT A DETAIL
-------------------------------------------------------------
The grouping comes from `backtest.report.day_of_week_breakdown` - imported,
never re-implemented - which keys every trade on the CME SESSION date of its
ENTRY. Both halves matter:

  * **ENTRY, not exit.** `exclude_days` suppresses entries, so a table keyed
    on the exit would point at a weekday whose removal does not remove the
    trades that made it worst.
  * **SESSION date, not UTC date.** The session opens 18:00 ET the previous
    evening (`backtest.event_calendar.session_date`), so a UTC-keyed table
    splits every Globex evening onto the wrong weekday. A "no Friday entries"
    rule keyed on the calendar date keeps trading through Thursday evening's
    Friday session and stops at Friday 18:00 ET, when the Monday session has
    already begun.

The DAILY realised-return table beside it is attributed by the EXIT date,
because that is when a return is realised and that is the index the engine's
own `returns` series carries. The two disagree for every overnight trade, and
they are reported as two tables rather than reconciled into one: the decision
is taken on the ENTRY table, because the entry is the thing the gate acts on.

MAXIMUM DRAWDOWN CONTRIBUTION
-----------------------------
Not a per-weekday drawdown - a weekday does not have an equity curve of its
own, and computing one over five disjoint slices produces five numbers that
sum to nothing. What is reported is each weekday's share of the run's DEEPEST
peak-to-trough episode: locate that episode on the trade-by-trade equity
curve, then attribute every trade realised inside it to its entry weekday.
Shares can exceed 100% in total, because profitable weekdays inside the
episode offset the losing ones; that is the honest arithmetic and it is
printed rather than normalised away.

ONE VERDICT PER VERSION, AND THAT IS NOT A DETAIL
-------------------------------------------------
Version B is Version A's entries minus the ones a classifier expected to lose,
so its trade list is a SUBSET and its weekday table is a different table. A
weekday named on Version A and written into a Version B package would stand
that package down on a session measured on a strategy nobody deployed - the
same shape as the Version B gap Stage 2 and Stage 3 were corrected for on
2026-08-25, where the answer travelled as far as a summary column and then
stopped while every output stayed complete and well-formed.

So the per-pair file carries a `versions` map and each entry has its own
profile, worst weekday, block and counterfactual. `--ml` runs Version B, and
the orchestrator passes it for exactly the reason `stage4_cmd` does: a pair
Stage 3 certified as B and this stage profiled only as A is a package promoted
with a weekday nobody measured for it. Without `--ml` the B entry is absent
and `promote.load_dow_gate` records `NOT EVALUATED` for it, which is the
honest reading - not an empty block list, which would say every session
cleared.

The COUNTERFACTUAL is run once per DISTINCT identified weekday rather than
once per version. Both versions usually name the same session, and a second
run to exclude the same day would produce the same masks over the same bars.

THE COST MODEL IS THE CONSTANT-TICK ONE
---------------------------------------
`--slippage-atr-mult` is Stage 4's flag and is deliberately not repeated here.
It is opt-in there for the reason its own docstring gives - an ATR-scaled run
is not comparable with a constant-tick one - and the orchestrator does not
pass it, so by default the two stages charge the same costs and their weekday
tables describe the same trades. An operator who ran Stage 4 with ATR-scaled
slippage should read this stage's expectancies as the constant-tick figures
they are, not as the tear sheet's.

THE COUNTERFACTUAL
------------------
One further Version A run over the SAME bars with `exclude_days` extended by
the identified weekday, so the only difference between the two curves is which
entries were suppressed. Two things are recorded because either alone
misleads:

  * **The candidate count and the trade list, never the trade count alone.**
    A filter can only remove candidate TRIGGERS, never realised trades. The
    walk holds one position at a time and ignores a trigger arriving while one
    is open, so declining an early trigger leaves the strategy flat for a
    later one it would have been holding through - measured elsewhere in this
    repository as 17 candidates removed and 11 realised entries ADDED. A trade
    count that went UP is not evidence the filter failed to bind.
  * **The engine's own `entry_filters` record**, which says how many bars the
    mask covered and how many entries it actually suppressed. A filter that
    ran and cut nothing is a different result from no filter at all, and the
    two equity curves are identical.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. This runs at import
# time, above the local imports below, because several modules resolve their
# BT_* variables while being imported (backtest.run's ARTIFACTS_ROOT) - loading
# the file inside main() would be too late for those and would work here, which
# is the kind of difference nobody notices until one runner silently uses the
# default path. The rules - the repository root derived from __file__ rather
# than the working directory, existing variables winning over the file, the
# CrossTrade credentials withheld from os.environ - live in ONE module rather
# than in a block copied into every runner: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # `python3 backtest/x.py` puts backtest/ on sys.path, not the repository
    # root, so mdlib is not importable until this runs.
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------


import os                                                          # noqa: E402

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                    # noqa: E402
import dataclasses                                                 # noqa: E402
import time                                                        # noqa: E402
import traceback                                                   # noqa: E402
from typing import Any                                             # noqa: E402

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.event_calendar import (WEEKDAY_FULL_NAMES,           # noqa: E402
                                     WEEKDAY_NAMES, add_filter_args,
                                     filter_config_kwargs,
                                     session_weekday)
from backtest.pipeline import (DOW_GATE_FILE,                      # noqa: E402
                               ML_THRESHOLD_DEFAULT,
                               BEST_PARAMS_FILE, STAGE45,
                               STAGE45_SUMMARY_FILE, leaderboard,
                               next_step, pipeline_dir, read_stage,
                               stage_banner, write_stage)
from backtest.report import day_of_week_breakdown                  # noqa: E402

#: The trade floor a weekday must clear before it can be condemned. The same
#: number `backtest.report.losing_weekdays` and `unprofitable_weekdays`
#: enforce, imported in spirit rather than re-derived: a weekday holds roughly
#: a fifth of a sample, and cutting one on eight trades is how a day-of-week
#: filter manufactures an in-sample Sharpe. A day below the floor stays in
#: however badly it scored, and the reason is recorded on the verdict.
DOW_MIN_TRADES = 20

#: The columns `weekday_profile` produces, in order. Declared so a table with
#: no rows still has the shape of one - an empty frame with no columns reads
#: as a bug in every consumer that indexes it.
PROFILE_COLUMNS = [
    "day", "day_name", "weekday", "trades", "net_pnl", "gross_win",
    "gross_loss", "win_rate", "profit_factor", "expectancy", "avg_win",
    "avg_loss", "pct_of_trades", "pct_of_net_pnl",
    "max_dd_contribution", "max_dd_contribution_pct",
    "sessions", "daily_net_pnl", "daily_return_mean_pct",
    "daily_return_sum_pct", "daily_win_rate",
]


# --------------------------------------------------------------------------
# The deepest drawdown episode, and who paid for it
# --------------------------------------------------------------------------
def worst_drawdown_episode(trades: pd.DataFrame) -> dict[str, Any]:
    """
    The deepest peak-to-trough stretch of the TRADE-BY-TRADE equity curve.

    Trade-by-trade rather than daily, because the question this answers is
    "which trades made the drawdown" and a daily curve has already pooled
    them. The curve is the cumulative net P&L in REALISATION order (sorted on
    `exit_time`), which is the order an account actually experienced.

    Returns `{"available", "reason", "peak_index", "trough_index",
    "peak_time", "trough_time", "decline"}`. `decline` is NEGATIVE - the
    dollars given back from the peak - and is 0.0 for a curve that never drew
    down, which is a finding rather than a missing value.
    """
    empty = {"available": False, "reason": "no trades", "peak_index": None,
             "trough_index": None, "peak_time": None, "trough_time": None,
             "decline": 0.0}
    if trades is None or len(trades) == 0:
        return empty
    cols = {c.lower(): c for c in trades.columns}
    pnl_col = next((cols[c] for c in ("pnl", "net_pnl", "profit")
                    if c in cols), None)
    exit_col = next((cols[c] for c in ("exit_time", "exit_ts", "exit")
                     if c in cols), None)
    if pnl_col is None:
        return {**empty, "reason": "the trade log carries no P&L column"}

    df = trades.copy()
    if exit_col is not None:
        df = df.sort_values(exit_col, kind="stable")
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0).to_numpy(float)
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(equity)
    underwater = equity - running_peak
    trough = int(np.argmin(underwater))
    decline = float(underwater[trough])
    if decline >= 0.0:
        return {**empty, "available": True,
                "reason": "the equity curve never drew down",
                "decline": 0.0}
    # The peak this trough fell from: the last index at or before the trough
    # whose equity equals the running peak there.
    peak = int(np.argmax(equity[:trough + 1]))
    stamps = (pd.to_datetime(df[exit_col], utc=True).to_numpy()
              if exit_col is not None else None)
    return {"available": True, "reason": "",
            "peak_index": peak, "trough_index": trough,
            "peak_time": (str(stamps[peak]) if stamps is not None else None),
            "trough_time": (str(stamps[trough]) if stamps is not None else None),
            "decline": decline,
            # The positional order the episode was located in, so
            # `drawdown_contributions` attributes exactly the trades inside it
            # rather than re-deriving a window that could differ by one row.
            "_order": df.index.to_numpy()}


def drawdown_contributions(trades: pd.DataFrame) -> dict[int, dict[str, float]]:
    """
    Each weekday's share of the deepest drawdown episode.

    NOT a per-weekday drawdown. A weekday has no equity curve of its own - an
    account is not flat on Tuesday because the strategy did not trade then -
    and five drawdowns computed over five disjoint slices are five numbers
    that sum to nothing and cannot be compared with the run's own. What is
    computed instead is the episode `worst_drawdown_episode` located, with
    every trade realised inside it attributed to the SESSION weekday of its
    ENTRY - the same key the rest of this stage uses, so a weekday's drawdown
    share and its net P&L are statements about the same set of trades.

    `pct` is the share of the episode's decline, positive for a weekday that
    contributed to it. **The shares can sum to more than 100%**, because
    profitable weekdays inside the episode offset the losing ones; that is the
    arithmetic and it is reported rather than normalised, since a normalised
    share would hide the fact that the drawdown was deeper than any one
    weekday made it.

    Returns `{weekday: {"dollars": float, "pct": float}}` for 0..6 - every
    weekday present, because an absent key reads as missing data where it
    means "this weekday traded nothing inside the worst drawdown".
    """
    out = {d: {"dollars": 0.0, "pct": 0.0} for d in range(7)}
    episode = worst_drawdown_episode(trades)
    if not episode["available"] or episode["decline"] >= 0.0:
        return out

    cols = {c.lower(): c for c in trades.columns}
    pnl_col = next((cols[c] for c in ("pnl", "net_pnl", "profit")
                    if c in cols), None)
    entry_col = next((cols[c] for c in ("entry_time", "entry_ts", "entry")
                      if c in cols), None)
    if pnl_col is None or entry_col is None:
        return out

    order = episode["_order"]
    inside = order[episode["peak_index"] + 1: episode["trough_index"] + 1]
    if len(inside) == 0:
        return out
    slice_ = trades.loc[inside]
    dow = session_weekday(slice_[entry_col])
    pnl = pd.to_numeric(slice_[pnl_col], errors="coerce").fillna(0.0)
    decline = episode["decline"]
    for d in range(7):
        dollars = float(pnl[dow == d].sum())
        out[d] = {"dollars": dollars,
                  # Two negatives divide to a positive share. A weekday that
                  # made money inside the episode gets a NEGATIVE share, which
                  # is the truth about it - it shortened the drawdown - and is
                  # not clipped to zero, because a clipped column reads as a
                  # weekday that was simply absent.
                  "pct": (100.0 * dollars / decline) if decline else 0.0}
    return out


# --------------------------------------------------------------------------
# The daily realised-return table
# --------------------------------------------------------------------------
def daily_return_profile(returns: pd.Series | None,
                         initial_capital: float | None = None
                         ) -> dict[int, dict[str, float]]:
    """
    The engine's DAILY realised returns, grouped by session weekday.

    `returns` is `BacktestResult.returns`: one point per session in the
    window, decimal, with each trade's P&L attributed to its EXIT date - which
    is when the money is realised, and is NOT the entry attribution the trade
    table above uses. The two are reported side by side rather than
    reconciled: an overnight trade is entered on one weekday and realised on
    the next, and pretending otherwise would make one of the tables wrong for
    every strategy that holds through a session close.

    The index is grouped through `session_weekday`, the same rule the trade
    table uses. On a UTC-midnight daily index that reduces to the plain
    weekday - which is the point: one rule, correct on both shapes, rather
    than a `.dayofweek` here that would silently disagree with the trade table
    the day somebody hands this an intraday index.

    Returns `{weekday: {"sessions", "mean_pct", "sum_pct", "win_rate"}}` for
    0..6, every weekday present.
    """
    out = {d: {"sessions": 0, "mean_pct": float("nan"),
               "sum_pct": 0.0, "win_rate": float("nan")} for d in range(7)}
    if returns is None or len(returns) == 0:
        return out
    idx = pd.DatetimeIndex(pd.to_datetime(returns.index, utc=True))
    vals = pd.to_numeric(pd.Series(np.asarray(returns, dtype=float)),
                         errors="coerce")
    dow = session_weekday(idx)
    for d in range(7):
        m = dow == d
        n = int(m.sum())
        if not n:
            continue
        g = vals[m]
        out[d] = {"sessions": n,
                  "mean_pct": float(g.mean() * 100.0),
                  "sum_pct": float(g.sum() * 100.0),
                  "win_rate": float((g > 0).sum() / n)}
    return out


# --------------------------------------------------------------------------
# The profile
# --------------------------------------------------------------------------
def weekday_profile(trades: pd.DataFrame,
                    returns: pd.Series | None = None) -> pd.DataFrame:
    """
    Net P&L, win rate, profit factor, expectancy and drawdown contribution
    per weekday - one row per Monday-Friday, plus a weekend row only if
    something actually traded there.

    The GROUPING is `backtest.report.day_of_week_breakdown`'s, imported rather
    than re-implemented. That function already owns the two decisions that
    change the answer - attribution by the ENTRY, keyed on the CME SESSION
    date - and a second grouping here would be free to disagree with the
    day-of-week table Stage 1 and Stage 4 already print, with both tables
    still summing to the same totals.

    What this adds is the three columns the gate needs and that table does not
    carry: `expectancy`, the drawdown contribution, and the daily realised
    returns.

    `expectancy` is the mean NET P&L per trade, which is arithmetically
    identical to `avg_pnl` on the breakdown - `win_rate x avg_win -
    loss_rate x |avg_loss|` expands to exactly the mean. It is computed ONCE,
    under the name the gate reads it by, rather than twice under two names
    that could drift apart; `avg_win` and `avg_loss` sit beside it because an
    expectancy is not readable without them.
    """
    base = day_of_week_breakdown(trades)
    dd = drawdown_contributions(trades)
    daily = daily_return_profile(returns)

    if base is None or base.empty:
        # Still five rows. An empty frame reads as "the stage did not run",
        # and a strategy with no trades at all is a finding about the strategy.
        rows = [{"day": WEEKDAY_NAMES[d], "day_name": WEEKDAY_FULL_NAMES[d],
                 "weekday": d, "trades": 0, "net_pnl": 0.0, "gross_win": 0.0,
                 "gross_loss": 0.0, "win_rate": float("nan"),
                 "profit_factor": float("nan"), "expectancy": float("nan"),
                 "avg_win": float("nan"), "avg_loss": float("nan"),
                 "pct_of_trades": float("nan"), "pct_of_net_pnl": float("nan"),
                 "max_dd_contribution": dd[d]["dollars"],
                 "max_dd_contribution_pct": dd[d]["pct"],
                 "sessions": daily[d]["sessions"],
                 "daily_net_pnl": 0.0,
                 "daily_return_mean_pct": daily[d]["mean_pct"],
                 "daily_return_sum_pct": daily[d]["sum_pct"],
                 "daily_win_rate": daily[d]["win_rate"]}
                for d in range(5)]
        return pd.DataFrame(rows, columns=PROFILE_COLUMNS)

    cols = {c.lower(): c for c in trades.columns}
    pnl_col = next((cols[c] for c in ("pnl", "net_pnl", "profit")
                    if c in cols), None)
    entry_col = next((cols[c] for c in ("entry_time", "entry_ts", "entry")
                      if c in cols), None)
    dow = (session_weekday(trades[entry_col])
           if entry_col is not None else np.array([], dtype=int))
    pnl = (pd.to_numeric(trades[pnl_col], errors="coerce")
           if pnl_col is not None else pd.Series(dtype=float))

    rows = []
    for _, r in base.iterrows():
        d = int(r["weekday"])
        if len(dow) and pnl_col is not None:
            g = pnl[dow == d].dropna()
            wins, losses = g[g > 0], g[g < 0]
        else:
            g = pd.Series(dtype=float)
            wins = losses = pd.Series(dtype=float)
        rows.append({
            "day": str(r["day"]),
            "day_name": WEEKDAY_FULL_NAMES[d],
            "weekday": d,
            "trades": int(r["trades"]),
            "net_pnl": float(r["net_pnl"]),
            "gross_win": float(r["gross_win"]),
            "gross_loss": float(r["gross_loss"]),
            "win_rate": float(r["win_rate"]),
            "profit_factor": float(r["profit_factor"]),
            # ONE number, under the name the gate ranks on. NaN, never 0.0,
            # for a weekday that never traded: a zero expectancy is a
            # break-even day and would sort above a losing one, which is the
            # wrong answer to "which weekday is worst" when the truth is that
            # nobody has any evidence about it.
            "expectancy": (float(g.mean()) if len(g) else float("nan")),
            "avg_win": (float(wins.mean()) if len(wins) else float("nan")),
            "avg_loss": (float(losses.mean()) if len(losses) else float("nan")),
            "pct_of_trades": float(r["pct_of_trades"]),
            "pct_of_net_pnl": float(r["pct_of_net_pnl"]),
            "max_dd_contribution": dd[d]["dollars"],
            "max_dd_contribution_pct": dd[d]["pct"],
            "sessions": daily[d]["sessions"],
            # The daily table's own net P&L is not recomputed here - the
            # returns series is a fraction of equity and the trade table is
            # dollars, and multiplying one back into the other would produce a
            # third number nobody measured. This column is the ENTRY-attributed
            # net P&L, repeated so the daily columns beside it are never read
            # as dollars.
            "daily_net_pnl": float(r["net_pnl"]),
            "daily_return_mean_pct": daily[d]["mean_pct"],
            "daily_return_sum_pct": daily[d]["sum_pct"],
            "daily_win_rate": daily[d]["win_rate"],
        })
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------
def select_worst_weekday(profile: pd.DataFrame,
                         min_trades: int = DOW_MIN_TRADES,
                         block_always: bool = False) -> dict[str, Any]:
    """
    The worst Monday-Friday session, and whether it is BLOCKED.

    Two answers, kept apart on purpose:

        `worst_weekday`    the weakest session, always named when there is a
                           qualifying row. Reported whatever it scored.
        `blocked_weekday`  the instruction the live loop acts on. None unless
                           the worst weekday's expectancy is actually
                           negative, or `block_always` was asked for.

    Collapsing them would make "every weekday made money" indistinguishable
    from "the stage did not run", and would hand the live gate a session to
    stand down on the strength of being fifth out of five.

    THE RANK, in order:

      1. `expectancy` ascending - mean net P&L per trade. The metric a trade
         decision is actually made on, and the one the counterfactual moves.
      2. `win_rate` ascending, as the tie-break. Two weekdays with the same
         expectancy and different hit rates are not equally bad: the one that
         gets there on fewer, larger wins is the one whose next quarter is
         less predictable.
      3. `max_dd_contribution_pct` descending. The largest share of the
         deepest drawdown breaks a remaining tie, which is the "largest tail
         drawdown" clause.

    A weekday below `min_trades` is NOT ranked, however badly it scored, and
    is listed in `below_floor` with its count so a reader can see it was
    considered and set aside rather than missing. A weekday with no trades at
    all has an undefined expectancy and is excluded for the same reason: there
    is no evidence to condemn it on.
    """
    out: dict[str, Any] = {
        "worst_weekday": None, "worst_day": None, "worst_day_name": None,
        "blocked_weekday": None, "blocked_day": None, "blocked_day_name": None,
        "blocked": False,
        "min_trades": int(min_trades),
        "block_rule": ("worst_always" if block_always
                       else "negative_expectancy"),
        "rank_rule": ("expectancy ascending, ties on win rate ascending, then "
                      "on the share of the deepest drawdown descending"),
        "eligible": [], "below_floor": [], "metrics": None,
        "reason": "",
    }
    if profile is None or profile.empty:
        out["reason"] = "no weekday profile: the run produced no trades"
        return out

    weekdays = profile[profile["weekday"] <= 4]
    for _, r in weekdays.sort_values("weekday").iterrows():
        entry = {"weekday": int(r["weekday"]), "day": str(r["day"]),
                 "trades": int(r["trades"])}
        if int(r["trades"]) < int(min_trades) or pd.isna(r["expectancy"]):
            out["below_floor"].append(entry)
        else:
            out["eligible"].append(entry)

    ranked = weekdays[(weekdays["trades"] >= int(min_trades))
                      & weekdays["expectancy"].notna()]
    if ranked.empty:
        out["reason"] = (
            f"no weekday reached the {int(min_trades)}-trade floor, so none "
            f"was ranked. Cutting a session on a thinner sample is how a "
            f"day-of-week filter manufactures an in-sample Sharpe.")
        return out

    ranked = ranked.sort_values(
        ["expectancy", "win_rate", "max_dd_contribution_pct"],
        ascending=[True, True, False], kind="stable")
    worst = ranked.iloc[0]
    d = int(worst["weekday"])
    out["worst_weekday"] = d
    out["worst_day"] = str(worst["day"])
    out["worst_day_name"] = WEEKDAY_FULL_NAMES[d]
    out["metrics"] = {
        "trades": int(worst["trades"]),
        "net_pnl": float(worst["net_pnl"]),
        "win_rate": (None if pd.isna(worst["win_rate"])
                     else float(worst["win_rate"])),
        "profit_factor": (None if pd.isna(worst["profit_factor"])
                          else float(worst["profit_factor"])),
        "expectancy": float(worst["expectancy"]),
        "max_dd_contribution": float(worst["max_dd_contribution"]),
        "max_dd_contribution_pct": float(worst["max_dd_contribution_pct"]),
        "pct_of_net_pnl": (None if pd.isna(worst["pct_of_net_pnl"])
                           else float(worst["pct_of_net_pnl"])),
    }

    if block_always or float(worst["expectancy"]) < 0.0:
        out["blocked"] = True
        out["blocked_weekday"] = d
        out["blocked_day"] = str(worst["day"])
        out["blocked_day_name"] = WEEKDAY_FULL_NAMES[d]
        out["reason"] = (
            f"{WEEKDAY_FULL_NAMES[d]} is the weakest session "
            f"(expectancy {float(worst['expectancy']):,.2f} per trade over "
            f"{int(worst['trades']):,} trades)"
            + ("; --block-worst-always was passed" if block_always
               and float(worst["expectancy"]) >= 0.0
               else " and its expectancy is negative"))
    else:
        out["reason"] = (
            f"{WEEKDAY_FULL_NAMES[d]} is the weakest session but its "
            f"expectancy is {float(worst['expectancy']):,.2f} per trade, "
            f"which is not negative. Nothing is blocked: cutting a profitable "
            f"session removes realised edge, and being fifth of five is not "
            f"evidence against a weekday. Pass --block-worst-always to block "
            f"it anyway.")
    return out


def promotion_gate(verdict: dict[str, Any]) -> dict[str, Any]:
    """
    Stage 4.5's promotion criterion, written down: EVERYTHING advances.

    This function exists precisely because the rule is "promote all" and a
    rule that is only an absence is a rule nobody can find. Stage 4.5 produces
    an INSTRUCTION for the live supervisor, never a verdict about a strategy -
    the same division the regime firewall draws, for the same reason: a
    weekday cut chosen in-sample on the bars it is scored on is not evidence
    that an edge is absent, and pruning on it would drop configurations Gate R
    certified on the strength of a calendar.

    Returns the row Stage 5 reads, with `promote` always True and the reason
    on the record.
    """
    return {
        "promote": True,
        "status": "ADVANCED",
        "criterion": ("every configuration reaching Stage 4.5 advances to "
                      "Stage 5. This stage names the worst session; it does "
                      "not certify, and it prunes nothing."),
        "blocked_weekday": verdict.get("blocked_weekday"),
        "blocked_day": verdict.get("blocked_day"),
        "worst_weekday": verdict.get("worst_weekday"),
    }


# --------------------------------------------------------------------------
# The console table
# --------------------------------------------------------------------------
def format_weekday_profile(profile: pd.DataFrame,
                           verdict: dict[str, Any] | None = None,
                           indent: str = "  ") -> str:
    """The weekday table, with the worst row marked and thin rows flagged."""
    if profile is None or profile.empty:
        return f"{indent}day of week   : no trades to attribute"
    verdict = verdict or {}
    worst = verdict.get("worst_weekday")
    blocked = verdict.get("blocked_weekday")
    floor = int(verdict.get("min_trades", DOW_MIN_TRADES))

    head = (f"{indent}{'day':<5}{'trades':>8}{'net P&L':>14}{'win':>8}"
            f"{'PF':>7}{'expectancy':>13}{'maxDD share':>13}"
            f"{'daily ret':>11}  note")
    lines = [head, indent + "-" * (len(head) - len(indent))]
    for _, r in profile.iterrows():
        d = int(r["weekday"])
        pf = r["profit_factor"]
        notes = []
        if int(r["trades"]) < floor:
            notes.append(f"thin (<{floor})")
        if worst is not None and d == worst:
            notes.append("WORST")
        if blocked is not None and d == blocked:
            notes.append("BLOCKED")
        # Formatted into locals first. Nesting a quoted f-string inside
        # another is legal from 3.12 and unreadable at any version; the point
        # of this table is that a human reads it.
        exp = (f"{r['expectancy']:,.2f}" if pd.notna(r["expectancy"]) else "n/a")
        pf_s = (f"{pf:.2f}" if pd.notna(pf) else "n/a")
        wr = (r["win_rate"] * 100 if pd.notna(r["win_rate"]) else float("nan"))
        dr = (f"{r['daily_return_mean_pct']:.3f}%"
              if pd.notna(r["daily_return_mean_pct"]) else "n/a")
        lines.append(
            f"{indent}{r['day']:<5}{int(r['trades']):>8,}"
            f"{r['net_pnl']:>14,.0f}{wr:>7.1f}%{pf_s:>7}{exp:>13}"
            f"{r['max_dd_contribution_pct']:>12.1f}%{dr:>11}"
            + ("  " + ", ".join(notes) if notes else ""))
    lines.append(f"{indent}expectancy is the mean NET P&L per trade, "
                 f"attributed to the ENTRY session.")
    lines.append(f"{indent}maxDD share is this weekday's share of the "
                 f"DEEPEST drawdown episode; shares can")
    lines.append(f"{indent}exceed 100% in total because profitable weekdays "
                 f"inside it offset the losing ones.")
    lines.append(f"{indent}daily ret is the mean realised daily return, "
                 f"attributed to the EXIT date - a")
    lines.append(f"{indent}different key from the columns to its left, and "
                 f"deliberately not reconciled.")
    return "\n".join(lines)


def format_counterfactual(cf: dict[str, Any], indent: str = "  ") -> str:
    """The baseline-versus-counterfactual block, for the console."""
    if not cf.get("available"):
        return f"{indent}(not run — {cf.get('reason', 'unknown')})"

    def num(v, fmt="{:,.2f}"):
        return "n/a" if v is None or (isinstance(v, float) and v != v) \
            else fmt.format(v)

    b, c = cf["baseline"], cf["counterfactual"]
    rows = [("trades", "{:,.0f}", "trade_count"),
            ("net P&L", "{:,.0f}", "total_pnl"),
            ("Sharpe", "{:.2f}", "sharpe"),
            ("profit factor", "{:.2f}", "profit_factor"),
            ("win rate %", "{:.1f}", "win_rate_pct"),
            ("max DD %", "{:.2f}", "max_drawdown_pct")]
    L = [f"{indent}{'metric':<16}{'baseline':>14}{'excl. ' + cf['excluded_day']:>14}"
         f"{'delta':>14}"]
    L.append(indent + "-" * 58)
    for label, fmt, key in rows:
        bv, cv = b.get(key), c.get(key)
        delta = (None if bv is None or cv is None
                 or (isinstance(bv, float) and bv != bv)
                 or (isinstance(cv, float) and cv != cv) else cv - bv)
        L.append(f"{indent}{label:<16}{num(bv, fmt):>14}{num(cv, fmt):>14}"
                 f"{num(delta, fmt):>14}")
    L.append("")
    L.append(f"{indent}candidate entries suppressed : "
             f"{cf['entries_suppressed']} of {cf['entries_offered']}")
    L.append(f"{indent}signal bars the mask covered : {cf['bars_blocked']}")
    L.append(f"{indent}A FILTER REMOVES CANDIDATE TRIGGERS, NEVER REALISED "
             f"TRADES. The walk holds one")
    L.append(f"{indent}position at a time, so declining an early trigger can "
             f"leave the strategy flat")
    L.append(f"{indent}for a later one it would have been holding through - a "
             f"trade count that went")
    L.append(f"{indent}UP is not evidence the filter failed to bind. Read the "
             f"suppressed-candidate")
    L.append(f"{indent}count above, or the trade list, and never the trade "
             f"count alone.")
    return "\n".join(L)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------
def _headline(metrics: dict | None) -> dict[str, Any]:
    """The six numbers the counterfactual table compares, from a metrics dict."""
    m = metrics or {}
    wr = m.get("win_rate")
    return {
        "trade_count": int(m.get("trade_count") or 0),
        "total_pnl": m.get("total_pnl"),
        "sharpe": m.get("sharpe"),
        "profit_factor": m.get("profit_factor"),
        "win_rate_pct": (None if wr is None or (isinstance(wr, float) and wr != wr)
                         else float(wr) * 100.0),
        "max_drawdown_pct": m.get("max_drawdown_pct"),
    }


def counterfactual_run(path: Path, bars: pd.DataFrame, symbol: str, tf: str,
                       params: dict, cfg, weekday: int | None,
                       strat_name: str, ml: bool = False) -> dict[str, Any]:
    """
    The same run with one weekday's ENTRIES suppressed, for every version.

    The ONLY difference from the baseline is `cfg.exclude_days`: the same
    bars, the same parameters, the same costs, the same fills. Anything else
    varying would make the comparison a measurement of the change rather than
    of the filter - the same rule `run_dual_version_backtest` holds A and B to.

    Returns `{"available", "reason", "excluded_day", "by_version": {...}}`.
    ONE call covers both versions, which is why this is keyed on a weekday
    rather than on a version: A and B usually name the same worst session, and
    a second run excluding the same day would build the same masks over the
    same bars.
    """
    from agents.tier1_master import run_dual_version_backtest       # noqa: PLC0415

    if weekday is None:
        return {"available": False, "excluded_day": None, "by_version": {},
                "reason": ("no weekday was identified, so there is nothing to "
                           "exclude and no counterfactual to run")}

    existing = tuple(cfg.exclude_days or ())
    if weekday in existing:
        return {"available": False, "by_version": {},
                "excluded_day": WEEKDAY_NAMES[weekday],
                "reason": (f"{WEEKDAY_FULL_NAMES[weekday]} is already in "
                           f"exclude_days for this run, so the baseline IS "
                           f"the counterfactual")}

    cf_cfg = dataclasses.replace(
        cfg,
        exclude_days=tuple(sorted(set(existing) | {int(weekday)})),
        notes=f"stage 4.5 counterfactual {symbol} {tf} "
              f"excl. {WEEKDAY_FULL_NAMES[weekday]}")
    out = run_dual_version_backtest(
        str(path), bars, freq=tf, symbol=symbol, cfg=cf_cfg, params=params,
        ml=ml, emit_reports=False, strat_name=strat_name)

    by_version: dict[str, dict] = {}
    for version, key in (("A", "version_a"), ("B", "version_b")):
        block = out.get(key)
        if not block:
            continue
        metrics = block.get("metrics") or {}
        filters = dict(metrics.get("entry_filters") or {})
        by_version[version] = {
            "metrics": _headline(metrics),
            # THE ENGINE'S OWN RECORD OF WHAT THE MASK DID, read straight off
            # `apply_entry_filters` rather than recomputed. A trade-count
            # delta cannot answer "did the filter bind" - the walk holds one
            # position at a time, so removing a trigger can ADD a realised
            # trade - and these three can. "NOT RECORDED" rather than 0 where
            # the key is absent: a zero would read as a mask that covered
            # nothing.
            "entries_suppressed": filters.get("entries_suppressed",
                                              "NOT RECORDED"),
            "entries_offered": (int(filters.get("long_entries_before", 0))
                                + int(filters.get("short_entries_before", 0))
                                if "long_entries_before" in filters
                                else "NOT RECORDED"),
            "bars_blocked": filters.get("dow_signal_bars_blocked",
                                        filters.get("signal_bars_blocked",
                                                    "NOT RECORDED")),
            "entry_filters": filters,
        }

    return {
        "available": True,
        "reason": "",
        "excluded_day": WEEKDAY_NAMES[weekday],
        "excluded_weekday": int(weekday),
        "exclude_days_applied": list(cf_cfg.exclude_days),
        "exclude_days_baseline": list(existing),
        "by_version": by_version,
        # Said on the file, not only in the docstring: this window spans the
        # Stage 3 holdout, so both curves are in-sample by construction.
        "is_certification": False,
        "note": ("Both curves were measured on the SAME window, which spans "
                 "the Stage 3 holdout. The weekday was chosen on these bars, "
                 "so the counterfactual is an in-sample restatement of that "
                 "choice and not evidence that cutting the day generalises."),
    }


def _version_counterfactual(cf: dict[str, Any], version: str,
                            baseline: dict[str, Any]) -> dict[str, Any]:
    """One version's slice of a shared counterfactual run, table-ready."""
    if not cf.get("available"):
        return {"available": False, "reason": cf.get("reason", "not run"),
                "excluded_day": cf.get("excluded_day")}
    side = (cf.get("by_version") or {}).get(version)
    if side is None:
        return {"available": False,
                "excluded_day": cf.get("excluded_day"),
                "reason": (f"the counterfactual run produced no Version "
                           f"{version} curve to compare against")}
    return {
        "available": True, "reason": "", "version": version,
        "excluded_day": cf["excluded_day"],
        "excluded_weekday": cf["excluded_weekday"],
        "exclude_days_applied": cf["exclude_days_applied"],
        "exclude_days_baseline": cf["exclude_days_baseline"],
        "baseline": baseline,
        "counterfactual": side["metrics"],
        "entries_suppressed": side["entries_suppressed"],
        "entries_offered": side["entries_offered"],
        "bars_blocked": side["bars_blocked"],
        "entry_filters": side["entry_filters"],
        "is_certification": False,
        "note": cf["note"],
    }


def gate_symbol(symbol: str, path: Path, tf: str, params: dict,
                args: argparse.Namespace, cfg_kwargs: dict,
                strat_name: str) -> dict[str, Any]:
    """
    One contract: profile every version, name its worst weekday, run the
    counterfactuals.

    Version B is Version A's entries minus the ones a classifier expected to
    lose, so its trade list is a SUBSET and its weekday table is a different
    table. Each version therefore gets its own profile, its own verdict and
    its own counterfactual slice - a weekday named on A and written into a B
    package would stand it down on a session measured on a strategy nobody
    deployed.
    """
    from backtest.engine import BacktestConfig                      # noqa: PLC0415
    from backtest.run import load_bars                              # noqa: PLC0415
    from agents.tier1_master import run_dual_version_backtest       # noqa: PLC0415

    t0 = time.time()
    print("\n" + "-" * 78)
    print(f"{symbol}  ·  {tf}  ·  day-of-week gate")
    print("-" * 78)

    bars = load_bars(symbol, tf, args.start, args.end)
    if "symbol" in bars.columns and bars["symbol"].nunique() > 1:
        raise ValueError(f"{symbol}: the lake returned an interleaved frame "
                         f"({bars['symbol'].nunique()} symbols)")
    print(f"  bars       : {len(bars):,}  "
          f"{bars['ts'].iloc[0]} → {bars['ts'].iloc[-1]}")
    print(f"  versions   : {'A and B' if args.ml else 'A only (no --ml)'}")

    cfg = BacktestConfig(
        initial_capital=args.capital, contracts=args.contracts,
        slippage_ticks=args.slippage_ticks, flat_by_close=args.flat_by_close,
        notes=f"stage 4.5 day-of-week gate {symbol} {tf}", **cfg_kwargs)

    out = run_dual_version_backtest(
        str(path), bars, freq=tf, symbol=symbol, cfg=cfg, params=params,
        threshold=args.threshold, ml=args.ml, emit_reports=False,
        strat_name=strat_name)

    versions: dict[str, dict[str, Any]] = {}
    for version, key in (("A", "version_a"), ("B", "version_b")):
        block = out.get(key)
        if not block:
            continue
        metrics = block.get("metrics") or {}
        result = block.get("result")
        profile = weekday_profile(metrics.get("trades"),
                                  getattr(result, "returns", None))
        verdict = select_worst_weekday(profile, min_trades=args.min_trades,
                                       block_always=args.block_worst_always)
        print(f"\n  DAY OF WEEK · Version {version} · Stage 4 lifecycle window")
        print(format_weekday_profile(profile, verdict))
        print(f"\n  VERDICT    : {verdict['reason']}")
        if verdict["below_floor"]:
            print("  below floor: "
                  + ", ".join(f"{e['day']} ({e['trades']} trades)"
                              for e in verdict["below_floor"])
                  + f"  — not ranked, floor is {args.min_trades}")
        versions[version] = {"version": version, "baseline": _headline(metrics),
                             "weekday_profile": profile.to_dict("records"),
                             "verdict": verdict}

    if not versions:
        raise ValueError(f"{symbol} {tf}: the dual-version run produced no "
                         f"Version A curve to profile")

    # ONE COUNTERFACTUAL PER DISTINCT IDENTIFIED WEEKDAY, not one per version.
    # A and B usually name the same session, and a second run excluding the
    # same day would build the same masks over the same bars. Sorted so the
    # order does not depend on dict insertion, which decides nothing here but
    # would make two runs of the same pair print in different orders.
    wanted = sorted({v["verdict"]["worst_weekday"] for v in versions.values()
                     if v["verdict"]["worst_weekday"] is not None})
    runs: dict[int | None, dict] = {}
    for day in wanted:
        runs[day] = counterfactual_run(path, bars, symbol, tf, params, cfg,
                                       day, strat_name, ml=args.ml)
    none_run = counterfactual_run(path, bars, symbol, tf, params, cfg, None,
                                  strat_name, ml=args.ml)

    for version, entry in versions.items():
        day = entry["verdict"]["worst_weekday"]
        shared = runs.get(day, none_run) if day is not None else none_run
        entry["counterfactual"] = _version_counterfactual(
            shared, version, entry["baseline"])
        entry["promotion"] = promotion_gate(entry["verdict"])
        entry["blocked_weekday"] = entry["verdict"]["blocked_weekday"]
        entry["blocked_day"] = entry["verdict"]["blocked_day"]
        entry["blocked_day_name"] = entry["verdict"]["blocked_day_name"]
        print(f"\n  COUNTERFACTUAL · Version {version} · entries on the "
              f"identified weekday suppressed")
        print(format_counterfactual(entry["counterfactual"]))
        print(f"  PROMOTION  : {entry['promotion']['status']} — "
              f"{entry['promotion']['criterion']}")

    print(f"  ({round(time.time() - t0, 1)}s)")

    return {
        "symbol": symbol,
        "timeframe": tf,
        "params": params,
        "ml": bool(args.ml),
        "ml_threshold": float(args.threshold),
        # WHICH VERSIONS WERE PROFILED, said as data. A file carrying only an
        # "A" entry because --ml was not passed and one carrying only an "A"
        # entry because Version B produced no curve are different facts, and
        # `promote.load_dow_gate` reports the missing version as NOT EVALUATED
        # either way rather than as an empty block list.
        "versions_profiled": sorted(versions),
        "versions": versions,
        "window": {"start": str(bars["ts"].iloc[0]),
                   "end": str(bars["ts"].iloc[-1]),
                   "bars": int(len(bars))},
        "entry_filters": cfg_kwargs,
        "is_certification": False,
        # The weekday was picked on the bars it is scored on. Recorded on the
        # file for the same reason Stage 1 records `selected_in_sample`: a
        # reader who does not know that reads the counterfactual as evidence.
        "selected_in_sample": True,
        "certification_note": (
            "Stage 4.5 certifies nothing. The window spans the Stage 3 "
            "holdout, the weekday was selected on these same bars, and the "
            "counterfactual is an in-sample restatement of that selection. "
            "Gate verdicts come from gate_audit_<SYMBOL>_<TF>.json."),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 4.5/5 — the day-of-week gate. Profiles every "
                    "certified configuration by weekday, names the worst "
                    "session, runs the counterfactual and writes the blocked "
                    "weekday for the live loop. It prunes nothing: every "
                    "configuration advances to Stage 5.")
    p.add_argument("--strat", required=True)
    p.add_argument("--symbols", default=None,
                   help="NQ, a list, or ALL. Default: every contract Stage 2 "
                        "selected parameters for.")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None)
    p.add_argument("--start", default="2010-01-01",
                   help="Profiling window start (default 2010-01-01, Stage "
                        "4's own default, so the two tables describe the "
                        "same bars)")
    p.add_argument("--end", default="2026-01-01",
                   help="Profiling window end (default 2026-01-01)")
    p.add_argument("--param", action="append", default=[], metavar="K=V")
    p.add_argument("--defaults", action="store_true",
                   help="Use the module's DEFAULT_PARAMS instead of Stage 2's "
                        "winner")
    p.add_argument("--min-trades", type=int, default=DOW_MIN_TRADES,
                   metavar="N",
                   help=f"A weekday must place at least this many trades "
                        f"before it can be condemned (default "
                        f"{DOW_MIN_TRADES}). A day below the floor is listed "
                        f"and never ranked.")
    p.add_argument("--block-worst-always", action="store_true",
                   help="Block the worst weekday even when its expectancy is "
                        "POSITIVE. Off by default: being fifth of five is not "
                        "evidence against a session, and cutting a profitable "
                        "one removes realised edge. Recorded on the handoff "
                        "as block_rule: worst_always.")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--flat-by-close", action="store_true")
    p.add_argument("--ml", action="store_true",
                   help="Also profile Version B, and give it its OWN worst "
                        "weekday. Version B is Version A's entries minus the "
                        "ones a classifier expected to lose, so its trade "
                        "list is a SUBSET and its weekday table is a "
                        "different table - a weekday named on A and written "
                        "into a B package stands it down on a session "
                        "measured on a strategy nobody deployed. The "
                        "orchestrator passes this for the same reason "
                        "stage4_cmd does. Without it the B entry is ABSENT "
                        "and promote.py records NOT EVALUATED for it, which "
                        "is not an empty block list.")
    p.add_argument("--ml-threshold", "--threshold", dest="threshold",
                   type=float, default=ML_THRESHOLD_DEFAULT,
                   help=(f"Version B: P(win) at or above which an entry is "
                         f"kept (default {ML_THRESHOLD_DEFAULT}). Stage 3 "
                         f"certifies at this bar, so a different value here "
                         f"profiles a filter nobody certified."))
    p.add_argument("--out-dir", default=None)
    add_filter_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from backtest.audit_gates import discover_symbols               # noqa: PLC0415
    from backtest.run import parse_param, parse_symbols, resolve_strategy  # noqa: PLC0415, E501
    from agents.tier3_workers import load_strategy                  # noqa: PLC0415

    path = resolve_strategy(args.strat)
    strat_name = path.parent.name if path.stem == "strat" else path.stem
    out_dir = pipeline_dir(strat_name, args.out_dir, create=True)
    overrides = dict(parse_param(p) for p in args.param)

    try:
        _fn, info = load_strategy(path, overrides)
        cfg_kwargs = filter_config_kwargs(args)
    except Exception as e:                                         # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # ONE timeframe, for Stage 3's and Stage 4's reason: the verdict is about
    # one (parameters, timeframe) pair, and a comma-separated list here would
    # write each pair's blocked weekday over the last one's.
    tfs = [t.strip() for t in str(args.tf or "").split(",") if t.strip()]
    if len(tfs) > 1:
        print(f"--tf takes ONE timeframe here, got {args.tf!r}. The blocked "
              f"weekday is a fact\nabout one (symbol, timeframe) pair; "
              f"several timeframes would overwrite each\nother's verdict. Run "
              f"it once per timeframe.", file=sys.stderr)
        return 1
    tf = (tfs[0] if tfs else None) or info.get("timeframe") or "15m"

    if args.symbols:
        symbols = parse_symbols(args.symbols, info.get("symbols"))
    else:
        symbols = discover_symbols(out_dir, tf)
        if not symbols:
            print(f"No best_params_*.json in {out_dir} and no --symbols. Run "
                  f"stage 2 first,\nor name the contracts.", file=sys.stderr)
            return 1

    print(stage_banner(STAGE45, strat_name,
                       f"{len(symbols)} contract(s) · {tf} · "
                       f"{args.start} → {args.end}"))
    print("  This stage names the worst weekday and writes it for the live")
    print("  loop. It certifies NOTHING and prunes NOTHING: every")
    print("  configuration here advances to Stage 5. The weekday is chosen")
    print("  IN-SAMPLE, on bars Stage 3 already spent.")
    if args.ml:
        print("  Versions A and B each get their OWN weekday: B's trade list "
              "is a")
        print("  subset of A's, so its weekday table is a different table.")
    else:
        print("  VERSION A ONLY (--ml not passed). A pair Stage 3 certified "
              "as")
        print("  Version B will record NOT EVALUATED on its promoted "
              "meta.json.")

    rows, errors = [], []
    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")
        try:
            params = dict(overrides)
            bp = out_dir / BEST_PARAMS_FILE.format(symbol=f"{sym}_{tf}")
            if not bp.exists():
                bp = out_dir / BEST_PARAMS_FILE.format(symbol=sym)
            if not args.defaults and bp.exists():
                blob = read_stage(bp, 2, strat_name)
                params = {**(blob.get("params") or {}), **overrides}
                print(f"  parameters : {params}  (stage 2 winner)")
            else:
                print(f"  parameters : {params or '(module defaults)'}  "
                      f"({'--defaults' if args.defaults else 'no stage 2 file'})")
            row = gate_symbol(sym, path, tf, params, args, cfg_kwargs,
                              strat_name)
            rows.append(row)
            write_stage(out_dir / DOW_GATE_FILE.format(symbol=sym, tf=tf),
                        STAGE45, strat_name, row)
        except Exception as e:                                     # noqa: BLE001
            errors.append({"symbol": sym, "timeframe": tf,
                           "error": f"{type(e).__name__}: {e}"})
            print(f"\n[!] {sym}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # The run's summary, MERGED across timeframes the way Stage 3's is: this
    # stage runs once per timeframe and a plain overwrite would keep only the
    # last, leaving the other timeframes' blocked weekdays on disk in their
    # per-pair files and reaching nobody through the summary.
    summary_path = out_dir / STAGE45_SUMMARY_FILE
    carried: list[dict] = []
    if summary_path.exists():
        try:
            prior = read_stage(summary_path, STAGE45, strat_name)
            carried = [r for r in (prior.get("results") or [])
                       if str(r.get("timeframe")) != str(tf)]
        except (ValueError, OSError) as e:                         # noqa: BLE001
            print(f"  (the previous Stage 4.5 summary could not be read and "
                  f"is being replaced: {e})", file=sys.stderr)

    # ONE ROW PER (symbol, timeframe, VERSION), because that is the unit that
    # gets promoted: `<strategy>_<SYMBOL>_<TF>_VA` and `..._VB` are two
    # packages with two meta.json files, and both can certify. A row per pair
    # would hand one weekday to both.
    results = carried + [
        {"symbol": r["symbol"], "timeframe": r["timeframe"],
         "version": v["version"],
         "blocked_weekday": v["blocked_weekday"],
         "blocked_day": v["blocked_day"],
         "worst_weekday": v["verdict"]["worst_weekday"],
         "worst_day": v["verdict"]["worst_day"],
         "block_rule": v["verdict"]["block_rule"],
         "min_trades": v["verdict"]["min_trades"],
         "reason": v["verdict"]["reason"],
         "promote": v["promotion"]["promote"],
         "status": v["promotion"]["status"],
         "counterfactual_available": bool(v["counterfactual"].get("available")),
         "file": str(out_dir / DOW_GATE_FILE.format(symbol=r["symbol"],
                                                    tf=r["timeframe"]))}
        for r in rows for v in r["versions"].values()]
    results += [{"symbol": e["symbol"], "timeframe": e["timeframe"],
                 "version": None, "blocked_weekday": None, "blocked_day": None,
                 "worst_weekday": None, "worst_day": None, "block_rule": None,
                 "min_trades": args.min_trades, "reason": e["error"],
                 # NOT PROFILED is deliberately not "nothing was blocked":
                 # "the run broke" and "the stage looked and found no losing
                 # session" must not share a token.
                 "promote": True, "status": "NOT PROFILED",
                 "counterfactual_available": False, "file": None}
                for e in errors]

    write_stage(summary_path, STAGE45, strat_name, {
        "timeframe": tf,
        "timeframes": sorted({str(r["timeframe"]) for r in results
                              if r.get("timeframe")}),
        "window": {"start": args.start, "end": args.end},
        "min_trades": int(args.min_trades),
        "ml": bool(args.ml),
        "ml_threshold": float(args.threshold),
        "block_rule": ("worst_always" if args.block_worst_always
                       else "negative_expectancy"),
        "selected_in_sample": True,
        "promotion_criterion": promotion_gate({})["criterion"],
        "coverage": {"requested": len(symbols), "profiled": len(rows),
                     "errors": len(errors)},
        "results": results,
    })

    # ONE ROW PER VERSION. The versions are two packages with two meta.json
    # files and can name different sessions; a table showing one row per pair
    # would present whichever version happened to be first as the pair's
    # answer.
    print(leaderboard(
        f"STAGE 4.5 · DAY-OF-WEEK GATE · {strat_name} · {tf}",
        ["SYMBOL", "TF", "VER", "WORST", "EXPECTANCY", "TRADES", "BLOCKED",
         "PROMOTE"],
        [[r["symbol"], r["timeframe"], v["version"],
          (v["verdict"]["worst_day"] or "--"),
          (f"{v['verdict']['metrics']['expectancy']:,.2f}"
           if v["verdict"]["metrics"] else "--"),
          (f"{v['verdict']['metrics']['trades']:,}"
           if v["verdict"]["metrics"] else "--"),
          (v["blocked_day"] or "none"),
          v["promotion"]["status"]]
         for r in rows for v in r["versions"].values()],
        empty="no configuration was profiled"))
    for e in errors:
        print(f"  ERROR {e['symbol']:<6}{e['error']}")

    print(f"\n  handoff    → {summary_path}")
    print(next_step([
        "Stage 5 reads the blocked weekday off this handoff and writes it",
        "into the promoted meta.json, where the live loop reads it:",
        "",
        f"  python3 backtest/promote.py --strat {strat_name} --version A \\",
        f"      --source {path} \\",
        f"      --audit-file {out_dir / f'gate_audit_<SYMBOL>_{tf}.json'} \\",
        f"      --metrics <verify_<stamp>/dual_metrics_<SYMBOL>.json>",
        "",
        "To read back what a promoted package is allowed to trade:",
        "",
        f"  python3 scripts/check_strategy_days.py --strat {strat_name}",
        "  (or the shell helper: days-check)",
    ]))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
