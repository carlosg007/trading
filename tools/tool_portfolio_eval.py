#!/usr/bin/env python3
"""
tools/tool_portfolio_eval.py - the forward book, scored two ways, promoting nothing.

Location:  ~/src/trading/tools/tool_portfolio_eval.py

    python3 tools/tool_portfolio_eval.py
    python3 tools/tool_portfolio_eval.py --markdown          # the #portfolio-mgmt card
    python3 tools/tool_portfolio_eval.py --ledger /tmp/l.json --config /tmp/c.json

The Portfolio & Promotion agent's daily read of `data/incubator_ledger.json`.
It scores every incubating strategy against TWO rule sets and prints both:

  CERTIFIED   `portfolio/promotion_daemon.py` — 14 calendar days, 10 active
              sessions, 14 closed trades, profit factor > 1.00, and a forward
              drawdown inside `allowable_forward_dd` (US dollars, derived from
              the portfolio's own trailing limit). This is the gate that
              actually governs promotion in this repo.

  MATRIX      The agent architecture's screen — Sharpe >= 1.8, win rate >= 52%,
              profit factor >= 1.4, max drawdown <= 3.5% over 30 sessions.

WHY BOTH, AND WHY THIS PROMOTES NOTHING
---------------------------------------
The two rule sets do not measure the same thing. The matrix adds Sharpe and
win rate, which the certified gate does not test at all, and states drawdown
as a PERCENTAGE where the certified gate states it in dollars against the
account the strategy will actually be governed by. Neither is a superset of
the other: a strategy can clear one and fail the other, in both directions.

So this tool **never calls `promote_strategy` and never writes to the
ledger.** Promotion stays with `scripts/incubator_tracker.py` and the human
four-choice menu. A second gate that could promote on different numbers than
the certified one is precisely the silent divergence this repo is built to
avoid: the promotion would be logged correctly, the card would read correctly,
and the strategy would be live on a rule nobody agreed to. Reporting a
disagreement is useful; acting on it unilaterally is not.

`variants_tested` is carried through to the report where the ledger records
it. A metric read without knowing how many variants produced it is not a
measurement.

HOW THE NUMBERS ARE COMPUTED
----------------------------
All of it in Python, from the entry's own CLOSED trades — an open position has
no realised P&L, and counting one would score a strategy on an outcome that is
still unknown. `_closed_trades`, `_trade_pnl` and the session-date derivation
are IMPORTED from `portfolio/promotion_daemon.py` rather than reimplemented,
so "what is a closed trade" has one definition in this repo.

  Sharpe      mean / stdev of per-SESSION net P&L, annualised by sqrt(252).
              Computed on dollars, which is identical to computing it on
              returns: a constant account size scales the mean and the
              deviation alike and cancels. Needs >= 2 sessions with a
              non-zero deviation; otherwise NOT MEASURABLE.
  Win rate    winning closed trades / closed trades.
  PF          gross profit / gross loss. With no losing trade it is
              UNDEFINED and reported as such — a 999 sentinel is not a
              measured factor, and the certified gate deliberately refuses
              one too.
  Max DD      deepest peak-to-trough excursion of CUMULATIVE closed-trade
              P&L. This is the REALISED drawdown; an open position's
              mark-to-market never enters it, so the true excursion can be
              deeper than this number.
  DD %        max drawdown / the portfolio's `default_account_size`. That
              denominator is a RESEARCH-side figure from
              `config/portfolios.json`, not a live account balance. Live
              balance, trailing drawdown and prop-firm state belong to
              CrossTrade and are deliberately not modelled here.

THE SESSION WINDOW
------------------
The matrix is scored over the most recent `--sessions` sessions (default 30).
A strategy with fewer than that many recorded sessions is INSUFFICIENT, not
failing — "not enough evidence" and "the evidence is bad" are different
verdicts and are never collapsed.

Sessions are attributed on the trade's EXIT, matching how
`promotion_daemon._session_dates` counts `active_sessions`. That is a
deliberate departure from this repo's entry-bar attribution rule for regime
and friction: using entry sessions here would make this card's session count
disagree with the certified gate's on the very same ledger, and two session
counts for one strategy is worse than one that is documented.

DEMOTION
--------
Two triggers, both advisory:

  * realised drawdown at or beyond `allowable_forward_dd` for the portfolio;
  * five CONSECUTIVE uncharacteristic losses.

"Uncharacteristic" is not defined by the architecture document, so it is
defined here and stated in the output: a losing trade whose magnitude exceeds
`mean + sigma x stdev` of that strategy's OWN prior losses (`--sigma`, default
1.0). It needs a baseline of at least `--loss-baseline` (default 5) losing
trades; below that the trigger reports NOT MEASURABLE rather than firing,
because a strategy with three losses has no characteristic loss to be
uncharacteristic of.

Exit codes:  0 ran, nothing flagged · 1 something needs a human ·
             2 could not run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.config_loader import (                              # noqa: E402
    DEFAULT_CONFIG_PATH,
    PortfolioConfigError,
    load_portfolio_config,
)
from portfolio.promotion_daemon import (                           # noqa: E402
    DEFAULT_LEDGER_PATH,
    GRADUATED_STATUSES,
    STATUS_INCUBATING,
    PromotionError,
    _closed_trades,
    _trade_pnl,
    _trade_timestamps,
    allowable_forward_dd,
    evaluate_strategy_promotion,
    load_ledger,
)
from scripts.incubator_tracker import (                            # noqa: E402
    incubator_assignments,
    resolve_account,
)

# -- the matrix, exactly as the architecture states it ----------------------
MATRIX_MIN_SHARPE = 1.8
MATRIX_MIN_WIN_RATE = 0.52
MATRIX_MIN_PROFIT_FACTOR = 1.4
MATRIX_MAX_DD_PCT = 3.5
MATRIX_SESSIONS = 30

#: Trading sessions per year, for annualising the session-level Sharpe.
TRADING_SESSIONS_PER_YEAR = 252

#: Consecutive uncharacteristic losses that trip the demotion flag.
DEMOTE_LOSS_STREAK = 5

VERDICT_PROMOTE = "PROMOTE"
VERDICT_HOLD = "HOLD"
VERDICT_INSUFFICIENT = "INSUFFICIENT"
VERDICT_DEMOTE = "DEMOTE"
VERDICT_OK = "OK"
VERDICT_UNMEASURABLE = "NOT MEASURABLE"
VERDICT_ERROR = "ERROR"
VERDICT_GRADUATED = "GRADUATED"
VERDICT_SKIPPED = "SKIPPED"
VERDICT_UNROUTED = "UNROUTED"

#: Verdicts a human has to look at.
ACTIONABLE = (VERDICT_PROMOTE, VERDICT_DEMOTE, VERDICT_ERROR, VERDICT_UNROUTED)


# --------------------------------------------------------------------------
# session attribution
# --------------------------------------------------------------------------

def trade_sessions(trades: list[dict]) -> list[str | None]:
    """
    One session date per trade, in trade order; None where unattributable.

    An explicit `session_date` on the trade wins — a recorder that already
    knows the session should not have it re-derived from a timestamp it may
    have rounded. The rest are derived in ONE batched call to
    `backtest.event_calendar.session_date`, which is where the 18:00 ET roll
    is written down; the import is lazy because it pulls pandas.
    """
    out: list[str | None] = [None] * len(trades)
    pending: list[int] = []
    stamps: list[str] = []

    for index, trade in enumerate(trades):
        explicit = trade.get("session_date")
        if isinstance(explicit, str) and explicit:
            out[index] = explicit[:10]
            continue
        found = _trade_timestamps([trade])
        if found:
            pending.append(index)
            stamps.append(found[0])

    if stamps:
        try:
            from backtest.event_calendar import session_date     # noqa: PLC0415
            for index, day in zip(pending, session_date(stamps)):
                out[index] = day.strftime("%Y-%m-%d")
        except Exception:                                        # noqa: BLE001
            # Falling back to the raw date is wrong by up to six hours for a
            # trade that exited in the evening session. It is applied rather
            # than dropping the trade, and the report says the derivation was
            # unavailable.
            for index, stamp in zip(pending, stamps):
                out[index] = str(stamp)[:10]
    return out


# --------------------------------------------------------------------------
# the arithmetic (deterministic; no model ever produces one of these numbers)
# --------------------------------------------------------------------------

def _stdev(values: list[float]) -> float | None:
    """Sample standard deviation, or None below two points."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def session_sharpe(session_pnl: list[float]) -> float | None:
    """
    Annualised Sharpe of per-session net P&L, or None when undefined.

    A zero deviation means every session returned the same amount. That is
    not an infinite Sharpe; it is a sample too degenerate to score, and it
    returns None rather than a number that would clear any threshold.
    """
    if len(session_pnl) < 2:
        return None
    deviation = _stdev(session_pnl)
    if not deviation:
        return None
    mean = sum(session_pnl) / len(session_pnl)
    return (mean / deviation) * math.sqrt(TRADING_SESSIONS_PER_YEAR)


def compute_metrics(trades: list[dict], sessions: list[str | None],
                    window: int) -> dict[str, Any]:
    """
    The matrix's inputs over the most recent `window` sessions.

    Trades with no attributable session are EXCLUDED from the window and
    counted in `unattributed`, rather than being swept into the newest
    session — which would inflate that session's P&L and the Sharpe with it.
    """
    dated = [(s, t) for s, t in zip(sessions, trades) if s]
    unattributed = len(trades) - len(dated)

    ordered_sessions = sorted({s for s, _ in dated})
    kept = set(ordered_sessions[-window:]) if window > 0 else set(ordered_sessions)
    selected = [(s, t) for s, t in dated if s in kept]

    pnls = [float(_trade_pnl(t) or 0.0) for _, t in selected]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    by_session: dict[str, float] = {}
    for session, trade in selected:
        by_session[session] = by_session.get(session, 0.0) + float(
            _trade_pnl(trade) or 0.0)
    session_pnl = [by_session[s] for s in sorted(by_session)]

    equity, peak, max_dd = 0.0, 0.0, 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return {
        "window_sessions": window,
        "sessions_in_window": len(by_session),
        "sessions_recorded": len(ordered_sessions),
        "unattributed_trades": unattributed,
        "trade_count": len(selected),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": (len(wins) / len(selected)) if selected else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": sum(pnls),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "pf_undefined": gross_loss == 0 and bool(selected),
        "max_drawdown": max_dd,
        "sharpe": session_sharpe(session_pnl),
        "session_pnl": session_pnl,
        "selected_pnls": pnls,
    }


def _criterion(name: str, passed: bool | None, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _fmt(value: float | None, decimals: int = 2, prefix: str = "") -> str:
    if value is None:
        return "NOT RECORDED"
    return f"{prefix}{value:,.{decimals}f}"


def score_matrix(metrics: dict[str, Any], account_size: float | None,
                 window: int) -> dict[str, Any]:
    """
    The architecture's matrix. A missing input FAILS its criterion.

    The one thing that is not a failure is having fewer than `window`
    sessions: that is INSUFFICIENT, reported before any criterion, because
    scoring a 4-session sample against a 30-session rule produces a verdict
    about nothing.
    """
    criteria: list[dict[str, Any]] = []
    sessions = metrics["sessions_in_window"]

    # The drawdown percentage is a fact about the trades, not about whether
    # the window is long enough to score. Recording it here keeps the column
    # populated on INSUFFICIENT rows, where a blank would read as "no
    # drawdown" rather than "not enough sessions to reach a verdict".
    metrics["max_drawdown_pct"] = (
        (metrics["max_drawdown"] / account_size) * 100.0
        if account_size else None)

    if sessions < window:
        return {
            "verdict": VERDICT_INSUFFICIENT,
            "criteria": criteria,
            "failed": [],
            "note": (f"{sessions} of {window} sessions recorded — not enough "
                     f"evidence to score, which is not the same as failing"),
        }

    sharpe = metrics["sharpe"]
    criteria.append(_criterion(
        "sharpe", sharpe is not None and sharpe >= MATRIX_MIN_SHARPE,
        f"Sharpe {_fmt(sharpe)} (>= {MATRIX_MIN_SHARPE})"
        + ("" if sharpe is not None else
           " — session P&L has no usable deviation")))

    win_rate = metrics["win_rate"]
    criteria.append(_criterion(
        "win_rate", win_rate is not None and win_rate >= MATRIX_MIN_WIN_RATE,
        f"win rate {_fmt(None if win_rate is None else win_rate * 100, 1)}% "
        f"(>= {MATRIX_MIN_WIN_RATE * 100:.0f}%) on "
        f"{metrics['trade_count']} closed trades"))

    pf = metrics["profit_factor"]
    if metrics["pf_undefined"]:
        criteria.append(_criterion(
            "profit_factor", False,
            f"profit factor UNDEFINED (no losing trade in the window); net "
            f"{_fmt(metrics['net_pnl'], 2, '$')}. A sentinel is not a "
            f"measured factor, so this does not clear the bar"))
    else:
        criteria.append(_criterion(
            "profit_factor", pf is not None and pf >= MATRIX_MIN_PROFIT_FACTOR,
            f"profit factor {_fmt(pf)} (>= {MATRIX_MIN_PROFIT_FACTOR})"))

    max_dd = metrics["max_drawdown"]
    if not account_size:
        criteria.append(_criterion(
            "max_drawdown", False,
            f"drawdown {_fmt(max_dd, 2, '$')} but the portfolio records no "
            f"'default_account_size', so the {MATRIX_MAX_DD_PCT}% bar has no "
            f"denominator and cannot be scored"))
        dd_pct = None
    else:
        dd_pct = metrics["max_drawdown_pct"]
        # Compared in DOLLARS, not percent. `1750/50000*100` is
        # 3.5000000000000004 in binary floating point, so a strategy at
        # exactly the 3.5% bar fails a `<= 3.5` test on rounding error alone.
        # `account_size * pct / 100` is exact for these magnitudes.
        dd_limit = account_size * MATRIX_MAX_DD_PCT / 100.0
        criteria.append(_criterion(
            "max_drawdown", max_dd <= dd_limit,
            f"realised drawdown {_fmt(max_dd, 2, '$')} = {_fmt(dd_pct)}% of "
            f"{_fmt(account_size, 0, '$')} (<= {MATRIX_MAX_DD_PCT}%)"))

    metrics["max_drawdown_pct"] = dd_pct
    failed = [c["name"] for c in criteria if not c["passed"]]
    return {
        "verdict": VERDICT_PROMOTE if not failed else VERDICT_HOLD,
        "criteria": criteria,
        "failed": failed,
        "note": "" if not failed else "failed: " + ", ".join(failed),
    }


def score_demotion(metrics: dict[str, Any], allowable_dd: float | None,
                   sigma: float, baseline: int) -> dict[str, Any]:
    """
    The two demotion triggers. Advisory — nothing is demoted here.

    The loss-streak trigger needs a baseline of losses to define what a
    characteristic loss IS. Below it the trigger is NOT MEASURABLE, never
    "clear": a strategy with three losses has not proved it is behaving, it
    has only not yet produced enough evidence to say it is not.
    """
    triggers: list[dict[str, Any]] = []

    max_dd = metrics["max_drawdown"]
    if allowable_dd is None or allowable_dd <= 0:
        triggers.append(_criterion(
            "trailing_drawdown", None,
            f"realised drawdown {_fmt(max_dd, 2, '$')} but the portfolio "
            f"declares no usable allowable forward drawdown to compare it to"))
    else:
        breached = max_dd >= allowable_dd
        triggers.append(_criterion(
            "trailing_drawdown", breached,
            f"realised drawdown {_fmt(max_dd, 2, '$')} against an allowable "
            f"{_fmt(allowable_dd, 2, '$')}"
            + (" — BREACHED" if breached else "")))

    pnls = metrics["selected_pnls"]
    losses = [abs(p) for p in pnls if p < 0]
    if len(losses) < baseline:
        triggers.append(_criterion(
            "loss_streak", None,
            f"{len(losses)} losing trade(s) in the window; at least "
            f"{baseline} are needed before a loss can be called "
            f"uncharacteristic"))
    else:
        mean_loss = sum(losses) / len(losses)
        deviation = _stdev(losses) or 0.0
        threshold = mean_loss + sigma * deviation
        streak = best = 0
        for pnl in pnls:
            if pnl < 0 and abs(pnl) > threshold:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        tripped = best >= DEMOTE_LOSS_STREAK
        triggers.append(_criterion(
            "loss_streak", tripped,
            f"longest run of losses beyond {_fmt(threshold, 2, '$')} "
            f"(mean {_fmt(mean_loss, 2, '$')} + {sigma:g}σ "
            f"{_fmt(deviation, 2, '$')}) is {best}; "
            f"{DEMOTE_LOSS_STREAK} trips demotion"
            + (" — TRIPPED" if tripped else "")))

    fired = [t["name"] for t in triggers if t["passed"] is True]
    unmeasured = [t["name"] for t in triggers if t["passed"] is None]
    if fired:
        verdict = VERDICT_DEMOTE
    elif len(unmeasured) == len(triggers):
        verdict = VERDICT_UNMEASURABLE
    else:
        verdict = VERDICT_OK
    return {"verdict": verdict, "triggers": triggers, "fired": fired,
            "unmeasured": unmeasured}


# --------------------------------------------------------------------------
# per-strategy assembly
# --------------------------------------------------------------------------

def evaluate_entry(strategy_id: str, entry: Any, config: dict,
                   assignments: dict, window: int, sigma: float,
                   baseline: int) -> dict[str, Any]:
    """One row. Every ledger entry gets one, including the broken ones."""
    row: dict[str, Any] = {
        "strategy_id": strategy_id, "account": "—", "status": None,
        "note": "", "metrics": None, "matrix": None, "demotion": None,
        "certified": None, "variants_tested": None,
    }
    if not isinstance(entry, dict):
        row["status"] = VERDICT_ERROR
        row["note"] = f"the ledger entry is a {type(entry).__name__}, not an object"
        return row

    row["variants_tested"] = entry.get("variants_tested")

    ledger_status = str(entry.get("status", STATUS_INCUBATING)).upper()
    account, provenance = resolve_account(strategy_id, entry, assignments)
    row["account"] = account or "—"

    if ledger_status in GRADUATED_STATUSES:
        row["status"] = VERDICT_GRADUATED
        row["account"] = entry.get("target_portfolio") or row["account"]
        row["note"] = f"graduated {entry.get('graduated_at', 'at an unrecorded time')}"
        return row
    if ledger_status != STATUS_INCUBATING:
        row["status"] = VERDICT_SKIPPED
        row["note"] = f"ledger status is {ledger_status!r}"
        return row
    if account is None:
        row["status"] = VERDICT_UNROUTED
        row["note"] = provenance
        return row

    portfolio = config["portfolios"][account]
    trades = _closed_trades(entry.get("trades"))
    metrics = compute_metrics(trades, trade_sessions(trades), window)
    row["metrics"] = metrics

    account_size = portfolio.get("default_account_size")
    try:
        allowable = allowable_forward_dd(portfolio)
    except PromotionError:
        allowable = None

    row["matrix"] = score_matrix(metrics, account_size, window)
    row["demotion"] = score_demotion(metrics, allowable, sigma, baseline)

    # The certified gate is asked on the WHOLE entry, unwindowed — that is
    # what the promotion daemon itself does, and asking it a different
    # question here would make this row disagree with the daemon's own.
    try:
        row["certified"] = evaluate_strategy_promotion(strategy_id, entry,
                                                       portfolio)
    except PromotionError as exc:
        row["certified"] = {"error": str(exc)}

    if row["demotion"]["verdict"] == VERDICT_DEMOTE:
        row["status"] = VERDICT_DEMOTE
    else:
        row["status"] = row["matrix"]["verdict"]
    row["note"] = row["matrix"]["note"]
    return row


def evaluate_all(ledger: dict, config: dict, window: int, sigma: float,
                 baseline: int) -> list[dict[str, Any]]:
    assignments = incubator_assignments(config)
    return [evaluate_entry(sid, entry, config, assignments, window, sigma,
                           baseline)
            for sid, entry in ledger.get("strategies", {}).items()]


def _certified_verdict(row: dict[str, Any]) -> str:
    report = row.get("certified")
    if not isinstance(report, dict):
        return "—"
    if "error" in report:
        return "ERROR"
    return VERDICT_PROMOTE if report.get("passed") else VERDICT_HOLD


def disagreements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rows where the two rule sets reach different verdicts.

    This is the report's most useful output and the reason both are printed.
    """
    out = []
    for row in rows:
        if row["matrix"] is None:
            continue
        matrix = row["matrix"]["verdict"]
        certified = _certified_verdict(row)
        if certified in ("—", "ERROR") or matrix == VERDICT_INSUFFICIENT:
            continue
        if matrix != certified:
            out.append({"strategy_id": row["strategy_id"],
                        "matrix": matrix, "certified": certified})
    return out


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------

HEADER = ["STRATEGY", "ACCOUNT", "SESS", "TRADES", "SHARPE", "WIN%", "PF",
          "DD%", "MATRIX", "CERTIFIED"]


def _row_cells(row: dict[str, Any]) -> list[str]:
    metrics = row["metrics"] or {}
    win = metrics.get("win_rate")
    return [
        row["strategy_id"][:44],
        str(row["account"]),
        str(metrics.get("sessions_in_window", "—")),
        str(metrics.get("trade_count", "—")),
        _fmt(metrics.get("sharpe")),
        _fmt(None if win is None else win * 100, 1),
        "UNDEF" if metrics.get("pf_undefined") else _fmt(metrics.get("profit_factor")),
        _fmt(metrics.get("max_drawdown_pct")),
        str(row["status"]),
        _certified_verdict(row),
    ]


def format_text(rows: list[dict[str, Any]], window: int) -> str:
    lines = [f"PORTFOLIO EVAL — matrix over {window} sessions, and the "
             f"certified gate",
             "  This tool promotes nothing. Both verdicts are advisory; "
             "promotion runs through scripts/incubator_tracker.py.",
             ""]
    if not rows:
        lines.append("  The ledger holds no strategies — nothing to evaluate. "
                     "That is an empty ledger, not a clean book.")
        return "\n".join(lines)

    table = [HEADER] + [_row_cells(r) for r in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(HEADER))]
    for index, cells in enumerate(table):
        lines.append("  " + "  ".join(c.ljust(widths[i])
                                      for i, c in enumerate(cells)))
        if index == 0:
            lines.append("  " + "  ".join("-" * w for w in widths))

    for row in rows:
        if row["status"] in (VERDICT_GRADUATED, VERDICT_SKIPPED):
            continue
        lines.append("")
        lines.append(f"  {row['strategy_id']}  [{row['status']}]")
        if row["variants_tested"] is not None:
            lines.append(f"    variants tested: {row['variants_tested']}")
        if row["note"]:
            lines.append(f"    {row['note']}")
        if row["matrix"]:
            for crit in row["matrix"]["criteria"]:
                mark = "PASS" if crit["passed"] else "FAIL"
                lines.append(f"    matrix   [{mark}] {crit['detail']}")
        if row["demotion"]:
            for trig in row["demotion"]["triggers"]:
                mark = ("FIRED" if trig["passed"] is True
                        else "n/a  " if trig["passed"] is None else "clear")
                lines.append(f"    demote   [{mark}] {trig['detail']}")
        certified = row.get("certified")
        if isinstance(certified, dict) and "error" in certified:
            lines.append(f"    certified [ERROR] {certified['error']}")
        elif isinstance(certified, dict):
            for crit in certified.get("criteria", []):
                # promotion_daemon._criterion records PASS/FAIL under
                # "status"; there is no "passed" key on its criteria and
                # reading for one renders every criterion as a failure.
                mark = str(crit.get("status", "FAIL"))
                lines.append(f"    certified[{mark}] {crit.get('detail', '')}")
        if row["metrics"] and row["metrics"]["unattributed_trades"]:
            lines.append(f"    NOTE {row['metrics']['unattributed_trades']} "
                         f"closed trade(s) carry no attributable session and "
                         f"are outside the window")

    clashes = disagreements(rows)
    if clashes:
        lines.append("")
        lines.append("  THE TWO RULE SETS DISAGREE:")
        for clash in clashes:
            lines.append(f"    {clash['strategy_id']}: matrix says "
                         f"{clash['matrix']}, the certified gate says "
                         f"{clash['certified']}")
        lines.append("    Neither is a superset of the other. A human picks.")
    return "\n".join(lines)


def format_markdown(rows: list[dict[str, Any]], window: int) -> str:
    """The daily card for `#portfolio-mgmt`."""
    out = ["*Portfolio evaluation*",
           f"_Matrix over {window} sessions, beside the certified gate. "
           f"Advisory only — this promotes nothing._"]
    if not rows:
        out.append("")
        out.append("The ledger holds no strategies — nothing to evaluate. "
                   "That is an empty ledger, not a clean book.")
        return "\n".join(out)

    actionable = [r for r in rows if r["status"] in ACTIONABLE]
    out.append("")
    out.append(f"{len(rows)} ledger entries · {len(actionable)} need a human")

    table = [HEADER] + [_row_cells(r) for r in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(HEADER))]
    body = "\n".join("  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
                     for cells in table)
    out.append(f"```\n{body}\n```")

    for row in actionable:
        out.append("")
        out.append(f"*{row['strategy_id']}* — `{row['status']}`")
        if row["note"]:
            out.append(row["note"])
        if row["demotion"] and row["demotion"]["verdict"] == VERDICT_DEMOTE:
            for trig in row["demotion"]["triggers"]:
                if trig["passed"] is True:
                    out.append(f"- demote: {trig['detail']}")

    clashes = disagreements(rows)
    if clashes:
        out.append("")
        out.append("*The two rule sets disagree*")
        for clash in clashes:
            out.append(f"- `{clash['strategy_id']}`: matrix {clash['matrix']}, "
                       f"certified {clash['certified']}")
        out.append("_Neither is a superset of the other. A human picks._")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the forward book against the architecture matrix "
                    "and the certified promotion gate. Promotes nothing.")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH,
                        help=f"forward-trade ledger (default {DEFAULT_LEDGER_PATH})")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help=f"portfolio routing table (default {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--sessions", type=int, default=MATRIX_SESSIONS,
                        help=f"matrix window in sessions (default {MATRIX_SESSIONS})")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="deviations above the mean loss that make a loss "
                             "uncharacteristic (default 1.0)")
    parser.add_argument("--loss-baseline", type=int, default=5,
                        help="losing trades needed before the streak trigger "
                             "can be measured (default 5)")
    parser.add_argument("--markdown", action="store_true",
                        help="emit the Markdown card for #portfolio-mgmt")
    parser.add_argument("--json", action="store_true",
                        help="emit the full report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ledger = load_ledger(args.ledger)
        config = load_portfolio_config(args.config)
    except (PromotionError, PortfolioConfigError, OSError,
            json.JSONDecodeError) as exc:
        print(f"portfolio eval could not run: {exc}", file=sys.stderr)
        return 2

    try:
        rows = evaluate_all(ledger, config, args.sessions, args.sigma,
                            args.loss_baseline)
    except Exception as exc:                                     # noqa: BLE001
        print(f"portfolio eval could not run: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"window_sessions": args.sessions, "rows": rows,
                          "disagreements": disagreements(rows)},
                         indent=2, default=str))
    elif args.markdown:
        print(format_markdown(rows, args.sessions))
    else:
        print(format_text(rows, args.sessions))

    return 1 if any(r["status"] in ACTIONABLE for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
