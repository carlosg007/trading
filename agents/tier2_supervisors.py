"""
agents.tier2_supervisors - the gates a result has to pass.

Location:  ~/src/trading/agents/tier2_supervisors.py

The deterministic evaluators are implemented: `evaluate_compliance`,
`evaluate_robustness`, `evaluate_lifecycle_state`. The agent-judgement classes
(`PropFirmSupervisor`, `OOSValidationSupervisor`, `review_all`) are still
scaffold and raise NotImplementedError, because a supervisor that defaults to
"approved" while unimplemented is worse than no supervisor at all - it
manufactures confidence.

What this tier is for
---------------------
Tier 2 decides whether a result is allowed to count. Two supervisors:

  - PROP-FIRM COMPLIANCE: trailing drawdown, profit target, daily loss limit,
    consistency rules. A strategy that breaches is rejected regardless of its
    Sharpe, because the account is closed before the edge has time to show up.
  - OOS VALIDATION: the Phase 3 gate - the HELD-BACK FINAL YEARS OF THE
    DATABENTO DATASET, untouched during optimization. A Sharpe that collapses
    across that boundary means overfitting, and the strategy is discarded
    rather than retuned.

    The NT8 tree is NOT this gate. It holds ~420 daily bars per symbol and is
    spliced on NinjaTrader's own roll rules, so a divergence against Databento
    mostly measures contract construction rather than strategy decay. It is a
    thin cross-feed sanity check. See `lake/futures_nt8/README.md`.

Why these are agents rather than asserts
----------------------------------------
The numeric checks themselves are not agent work - `backtest.engine` already
computes the breach and the stats, deterministically, and that is where they
belong. What Tier 2 adds is judgement over the *context* the numbers arrived
in: how many variants were tested to find this one, whether the OOS window
overlaps anything already used for selection, whether a "fix" between runs was
a bug fix or a fit to the test set. Those are the questions that decide whether
a number is evidence, and they are not expressible as a threshold.

The deterministic parts stay deterministic. A supervisor may read
`BacktestResult.breach` and `BacktestResult.stats`; it may not recompute them,
and it may not overrule them.

Boundary
--------
A supervisor's rejection is final. Tier 1 may not overrule it, and neither may
a human asking to "see if the strategy works" with the constraint relaxed. If
a constraint is genuinely wrong, it gets changed in `BacktestConfig` and
everything is re-run - not waived for one result.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from google import genai
    _GENAI_IMPORT_ERROR: Exception | None = None
except ImportError as e:      # pragma: no cover - depends on the environment
    genai = None
    _GENAI_IMPORT_ERROR = e

REPO = Path(__file__).resolve().parent.parent
RULES_DIR = REPO / "compliance_rules"

# Robustness thresholds. Not in the JSON rulesets because they are house
# research standards rather than anything the prop firm imposes.
MIN_WFO_EFFICIENCY = 0.50

# Lifecycle bands, as multiples of the historical max drawdown.
LIFECYCLE_PAUSE_AT = 1.00
LIFECYCLE_ESCALATE_AT = 1.35
LIFECYCLE_DECOMMISSION_AT = 1.50


class RulesetError(Exception):
    """Raised when a ruleset cannot be loaded or does not define what is asked."""


@dataclass
class Verdict:
    """
    One supervisor's decision.

    `reasons` is required, not decorative: a rejection nobody can read is a
    rejection nobody can learn from, and an approval nobody can read is not
    reviewable.
    """

    supervisor: str
    approved: bool
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Ruleset loading
# --------------------------------------------------------------------------
def load_ruleset(ruleset_path: str | Path) -> dict:
    """Load a compliance ruleset. Every failure raises rather than defaulting."""
    path = Path(ruleset_path)
    if not path.is_absolute() and not path.exists():
        candidate = RULES_DIR / path.name
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise RulesetError(f"ruleset not found: {ruleset_path}")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        raise RulesetError(f"{path.name} is not readable JSON: "
                           f"{type(e).__name__}: {e}") from e
    if not isinstance(data, dict) or "rules" not in data:
        raise RulesetError(f"{path.name} has no 'rules' block")
    return data


def _rule(ruleset: dict, name: str) -> dict:
    rule = (ruleset.get("rules") or {}).get(name)
    if not isinstance(rule, dict) or rule.get("value") is None:
        raise RulesetError(
            f"ruleset '{ruleset.get('ruleset_id', '?')}' does not define "
            f"rule '{name}'"
        )
    return rule


# --------------------------------------------------------------------------
# Equity reconstruction
# --------------------------------------------------------------------------
def daily_pnl_from_trades(trade_log_df: pd.DataFrame) -> pd.Series:
    """
    Net P&L per calendar day, booked on the day each trade EXITED.

    Trades that opened earlier contribute entirely to their exit day, which is
    how a prop firm's daily accounting works for realized P&L.
    """
    if trade_log_df is None or len(trade_log_df) == 0:
        return pd.Series(dtype=float)
    df = trade_log_df
    for col in ("exit_time", "pnl"):
        if col not in df.columns:
            raise ValueError(f"trade log is missing required column {col!r}")
    day = pd.to_datetime(df["exit_time"], utc=True).dt.date
    return df.groupby(day)["pnl"].sum().sort_index()


def equity_from_trades(trade_log_df: pd.DataFrame,
                       initial_balance: float) -> pd.Series:
    """Daily closing equity implied by the realized trade log."""
    daily = daily_pnl_from_trades(trade_log_df)
    if daily.empty:
        return pd.Series(dtype=float)
    return initial_balance + daily.cumsum()


def _trailing_drawdown(equity: pd.Series, initial_balance: float) -> dict:
    """
    Worst shortfall below the trailing high water mark, as a percent of peak.

    Matches backtest.engine.check_trailing_drawdown: the floor trails the peak,
    so the PATH decides the account rather than the total.
    """
    if equity.empty:
        return {"worst_pct": 0.0, "worst_date": None, "peak_equity": initial_balance}
    curve = pd.concat([pd.Series([initial_balance]), equity], ignore_index=True)
    peak = curve.cummax()
    dd = (curve / peak - 1.0) * 100.0
    idx = int(dd.idxmin())
    worst_date = None
    if idx > 0:
        worst_date = str(equity.index[idx - 1])
    return {"worst_pct": float(dd.min()),
            "worst_date": worst_date,
            "peak_equity": float(peak.max())}


# --------------------------------------------------------------------------
# 1. Compliance
# --------------------------------------------------------------------------
def evaluate_compliance(trade_log_df: pd.DataFrame,
                        ruleset_path: str | Path,
                        initial_balance: float | None = None,
                        equity: pd.Series | None = None,
                        consistency_pct: float | None = None) -> dict[str, Any]:
    """
    Audit a trade log against an Evaluation Phase ruleset.

    Checks, all read from the ruleset rather than hardcoded:

      - profit target reached
      - max trailing drawdown not breached
      - consistency: no single day above the ruleset's share of total profit

    Returns a strict PASS/FAIL verdict with the specific reason for each
    failure. A rule the ruleset does not define is reported as NOT_EVALUATED,
    never as passed.

    Two things the caller must know
    -------------------------------
    **The thresholds come from the ruleset file.** `consistency_pct` overrides
    the ruleset's value for a one-off audit; it does not change the file. If a
    different limit is intended, change the ruleset - a threshold that lives in
    an argument is one nothing else in the system can see.

    **A trade-log equity curve understates drawdown.** It marks P&L only when a
    trade exits, so an open position's adverse excursion is invisible. The
    engine's daily curve captures it. Pass `equity` (from
    `BacktestResult.equity`) whenever you have it; the verdict records which
    was used, because a prop account is closed on unrealized drawdown too.
    """
    ruleset = load_ruleset(ruleset_path)
    account = ruleset.get("account") or {}
    balance = float(initial_balance if initial_balance is not None
                    else account.get("initial_balance", 100_000.0))
    if balance <= 0:
        raise ValueError(f"initial_balance must be positive, got {balance}")

    reconstructed = equity is None
    curve = (equity_from_trades(trade_log_df, balance)
             if reconstructed else pd.Series(equity).dropna())

    n_trades = 0 if trade_log_df is None else len(trade_log_df)
    daily = daily_pnl_from_trades(trade_log_df)
    total_pnl = float(daily.sum()) if not daily.empty else 0.0

    checks: dict[str, dict] = {}
    failures: list[str] = []

    # -- profit target ----------------------------------------------------
    try:
        rule = _rule(ruleset, "profit_target")
        target_pct = float(rule["value"])
        target_abs = balance * target_pct / 100.0
        reached = total_pnl >= target_abs
        checks["profit_target"] = {
            "status": "PASS" if reached else "FAIL",
            "required_pct": target_pct,
            "required_abs": target_abs,
            "achieved_abs": total_pnl,
            "achieved_pct": total_pnl / balance * 100.0,
        }
        if not reached:
            failures.append(
                f"profit target not reached: {total_pnl:,.2f} of "
                f"{target_abs:,.2f} required ({target_pct}% of {balance:,.0f})"
            )
    except RulesetError as e:
        checks["profit_target"] = {"status": "NOT_EVALUATED", "reason": str(e)}
        failures.append(f"profit target could not be evaluated: {e}")

    # -- trailing drawdown ------------------------------------------------
    try:
        rule = _rule(ruleset, "max_trailing_drawdown")
        limit_pct = float(rule["value"])
        dd = _trailing_drawdown(curve, balance)
        breached = abs(dd["worst_pct"]) > limit_pct
        checks["max_trailing_drawdown"] = {
            "status": "FAIL" if breached else "PASS",
            "limit_pct": limit_pct,
            "worst_drawdown_pct": dd["worst_pct"],
            "worst_date": dd["worst_date"],
            "equity_source": "reconstructed_from_trades" if reconstructed
                             else "supplied_equity_curve",
        }
        if breached:
            failures.append(
                f"trailing drawdown breached: {abs(dd['worst_pct']):.2f}% "
                f"against a {limit_pct}% limit"
                + (f" on {dd['worst_date']}" if dd["worst_date"] else "")
            )
    except RulesetError as e:
        checks["max_trailing_drawdown"] = {"status": "NOT_EVALUATED", "reason": str(e)}
        failures.append(f"trailing drawdown could not be evaluated: {e}")

    # -- consistency ------------------------------------------------------
    try:
        rule = _rule(ruleset, "consistency")
        limit = float(consistency_pct if consistency_pct is not None
                      else rule["value"])
        checks["consistency"] = _check_consistency(daily, total_pnl, limit)
        if checks["consistency"]["status"] == "FAIL":
            c = checks["consistency"]
            failures.append(
                f"consistency breached: {c['best_day']} contributed "
                f"{c['best_day_pnl']:,.2f} which is "
                f"{c['best_day_share_pct']:.2f}% of total profit "
                f"{total_pnl:,.2f}, above the {limit}% limit"
            )
        elif checks["consistency"]["status"] == "NOT_ASSESSABLE":
            failures.append(
                f"consistency not assessable: {checks['consistency']['reason']}"
            )
    except RulesetError as e:
        checks["consistency"] = {"status": "NOT_EVALUATED", "reason": str(e)}
        failures.append(f"consistency could not be evaluated: {e}")

    passed = not failures
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "ruleset_id": ruleset.get("ruleset_id"),
        "ruleset_path": str(ruleset_path),
        "failures": failures,
        "checks": checks,
        "summary": {
            "n_trades": int(n_trades),
            "n_trading_days": int(len(daily)),
            "total_pnl": total_pnl,
            "initial_balance": balance,
            "final_equity": float(curve.iloc[-1]) if len(curve) else balance,
        },
        "equity_source": ("reconstructed_from_trades" if reconstructed
                          else "supplied_equity_curve"),
        "caveat": (
            "Equity was reconstructed from realized trade exits, so drawdown "
            "while a position was open is not visible and the figure is a "
            "LOWER BOUND. Pass equity=BacktestResult.equity for the real "
            "path." if reconstructed else None
        ),
        "unenforced_rules": _unenforced(ruleset, checks),
    }


def _check_consistency(daily: pd.Series, total_pnl: float,
                       limit_pct: float) -> dict:
    """
    Does any single day account for more than `limit_pct` of total profit?

    Only profitable days can breach: the rule targets an account carried by one
    outlier session, and a large losing day is a different problem that the
    drawdown rule already covers.

    Undefined when total profit is zero or negative. That returns
    NOT_ASSESSABLE rather than PASS - a losing account has not demonstrated
    consistency, it has simply made the ratio meaningless, and reporting that
    as a pass is how a failing strategy collects a clean verdict.
    """
    if daily.empty:
        return {"status": "NOT_ASSESSABLE", "reason": "no trading days",
                "limit_pct": limit_pct}
    if total_pnl <= 0:
        return {
            "status": "NOT_ASSESSABLE",
            "reason": (f"total profit is {total_pnl:,.2f}; the share of a "
                       f"non-positive total is not meaningful"),
            "limit_pct": limit_pct,
            "total_pnl": total_pnl,
        }

    profitable = daily[daily > 0]
    if profitable.empty:
        return {"status": "NOT_ASSESSABLE",
                "reason": "no profitable days", "limit_pct": limit_pct}

    best_day = profitable.idxmax()
    best_pnl = float(profitable.max())
    share = best_pnl / total_pnl * 100.0
    return {
        "status": "FAIL" if share > limit_pct else "PASS",
        "limit_pct": limit_pct,
        "best_day": str(best_day),
        "best_day_pnl": best_pnl,
        "best_day_share_pct": share,
        "total_pnl": total_pnl,
        "n_profitable_days": int(len(profitable)),
    }


def _unenforced(ruleset: dict, checks: dict) -> list[str]:
    """
    Rules the ruleset declares that this audit did not evaluate.

    Surfaced so a PASS cannot be read as blanket compliance. The ruleset
    records that only trailing drawdown is enforced inside the engine; a
    verdict that stays quiet about the rest invites exactly that misreading.
    """
    declared = set((ruleset.get("rules") or {}).keys())
    evaluated = {k for k, v in checks.items()
                 if v.get("status") in ("PASS", "FAIL")}
    return sorted(declared - evaluated)


# --------------------------------------------------------------------------
# 2. Robustness
# --------------------------------------------------------------------------
def evaluate_robustness(wfo_efficiency: float | None,
                        mc_95th_dd: float | None,
                        max_allowed_dd: float,
                        min_wfo_efficiency: float = MIN_WFO_EFFICIENCY
                        ) -> dict[str, Any]:
    """
    Gate a strategy on walk-forward efficiency and bootstrapped drawdown.

    Passes only when BOTH hold:

      - WFO efficiency >= `min_wfo_efficiency` (default 0.50)
      - |Monte Carlo drawdown at the confidence level| <= `max_allowed_dd`

    Sign convention: drawdowns are compared on magnitude, so it does not matter
    whether the caller passes -8.4 or 8.4. `max_allowed_dd` is a positive
    percent, e.g. the 8.0 in the FundedNext ruleset.

    A `None` input fails rather than being skipped. `run_walk_forward_analysis`
    returns `efficiency_ratio=None` when no fold produced a defined ratio, and
    treating "could not be measured" as "met the bar" is how an unvalidated
    strategy reaches production.
    """
    if max_allowed_dd is None or (isinstance(max_allowed_dd, float)
                                  and math.isnan(max_allowed_dd)):
        raise ValueError("max_allowed_dd is required and must be a number")
    max_allowed = abs(float(max_allowed_dd))

    failures: list[str] = []
    checks: dict[str, dict] = {}

    if wfo_efficiency is None or (isinstance(wfo_efficiency, float)
                                  and math.isnan(wfo_efficiency)):
        checks["wfo_efficiency"] = {
            "status": "FAIL", "required_min": min_wfo_efficiency,
            "observed": None,
            "reason": "no defined walk-forward efficiency was produced",
        }
        failures.append(
            "walk-forward efficiency is undefined - not measured is not the "
            "same as passed"
        )
    else:
        eff = float(wfo_efficiency)
        ok = eff >= min_wfo_efficiency
        checks["wfo_efficiency"] = {
            "status": "PASS" if ok else "FAIL",
            "required_min": min_wfo_efficiency, "observed": eff,
        }
        if not ok:
            failures.append(
                f"walk-forward efficiency {eff:.3f} is below the "
                f"{min_wfo_efficiency:.2f} minimum - in-sample performance is "
                f"not surviving the step forward"
            )

    if mc_95th_dd is None or (isinstance(mc_95th_dd, float)
                              and math.isnan(mc_95th_dd)):
        checks["monte_carlo_drawdown"] = {
            "status": "FAIL", "max_allowed_pct": max_allowed, "observed": None,
            "reason": "no Monte Carlo drawdown was produced",
        }
        failures.append("Monte Carlo drawdown is undefined - cannot be cleared")
    else:
        observed = abs(float(mc_95th_dd))
        ok = observed <= max_allowed
        checks["monte_carlo_drawdown"] = {
            "status": "PASS" if ok else "FAIL",
            "max_allowed_pct": max_allowed, "observed_pct": observed,
        }
        if not ok:
            failures.append(
                f"Monte Carlo drawdown {observed:.2f}% exceeds the "
                f"{max_allowed:.2f}% limit - a reordering of the same trades "
                f"closes the account"
            )

    passed = not failures
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "failures": failures,
        "checks": checks,
        "caveat": (
            "The Monte Carlo figure comes from an i.i.d. bootstrap, which "
            "destroys trade autocorrelation and therefore understates "
            "clustered drawdowns. Clearing this bar is necessary, not "
            "sufficient."
        ),
    }


# --------------------------------------------------------------------------
# 3. Lifecycle state machine
# --------------------------------------------------------------------------
def evaluate_lifecycle_state(live_dd: float,
                             historical_max_dd: float,
                             rolling_ev: float) -> dict[str, Any]:
    """
    Live execution state from drawdown against the historical envelope.

        ACTIVE           live DD <= 1.00x historical
        PAUSED           1.00x < live DD <= 1.50x
        DECOMMISSIONED   live DD > 1.50x, or rolling EV negative

    A negative rolling expectancy decommissions regardless of drawdown: a
    strategy with no edge left is not waiting to recover, it is bleeding.

    The 1.35x band
    --------------
    The specification named 1.00x, 1.35x and 1.50x but defined no state between
    1.35x and 1.50x. Rather than invent one or let the gap fall through to
    ACTIVE, that range stays PAUSED and sets `escalated=True`. Failing safe
    matters more here than guessing: the fallthrough state for a strategy in
    unusual drawdown must never be "keep trading".

    Confirm the intended behaviour before this drives real capital.
    """
    for label, value in (("live_dd", live_dd),
                         ("historical_max_dd", historical_max_dd),
                         ("rolling_ev", rolling_ev)):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError(f"{label} must be a number, got {value!r}")

    live = abs(float(live_dd))
    historical = abs(float(historical_max_dd))
    ev = float(rolling_ev)

    if historical == 0:
        # No historical envelope to compare against. Any live drawdown is
        # outside it by definition, so this cannot resolve to ACTIVE.
        state = "DECOMMISSIONED" if (live > 0 or ev < 0) else "PAUSED"
        return {
            "state": state,
            "ratio": None,
            "escalated": True,
            "reasons": [
                "historical_max_dd is zero, so there is no envelope to measure "
                "against; refusing to report ACTIVE"
            ] + (["rolling expectancy is negative"] if ev < 0 else []),
            "inputs": {"live_dd": live, "historical_max_dd": historical,
                       "rolling_ev": ev},
        }

    ratio = live / historical
    reasons: list[str] = []

    if ev < 0:
        state = "DECOMMISSIONED"
        reasons.append(
            f"rolling expectancy is negative ({ev:.6g}) - no edge remains, "
            f"which decommissions regardless of drawdown"
        )
    elif ratio > LIFECYCLE_DECOMMISSION_AT:
        state = "DECOMMISSIONED"
        reasons.append(
            f"live drawdown is {ratio:.2f}x the historical maximum, above the "
            f"{LIFECYCLE_DECOMMISSION_AT:.2f}x decommission threshold"
        )
    elif ratio > LIFECYCLE_PAUSE_AT:
        state = "PAUSED"
        reasons.append(
            f"live drawdown is {ratio:.2f}x the historical maximum, above the "
            f"{LIFECYCLE_PAUSE_AT:.2f}x pause threshold"
        )
    else:
        state = "ACTIVE"
        reasons.append(
            f"live drawdown is {ratio:.2f}x the historical maximum, within the "
            f"{LIFECYCLE_PAUSE_AT:.2f}x envelope"
        )

    escalated = state == "PAUSED" and ratio > LIFECYCLE_ESCALATE_AT
    if escalated:
        reasons.append(
            f"above the {LIFECYCLE_ESCALATE_AT:.2f}x escalation mark and inside "
            f"the band the specification left undefined - held at PAUSED "
            f"rather than resolved to a state nobody specified"
        )

    return {
        "state": state,
        "ratio": float(ratio),
        "escalated": escalated,
        "reasons": reasons,
        "inputs": {"live_dd": live, "historical_max_dd": historical,
                   "rolling_ev": ev},
        "thresholds": {
            "pause_at": LIFECYCLE_PAUSE_AT,
            "escalate_at": LIFECYCLE_ESCALATE_AT,
            "decommission_at": LIFECYCLE_DECOMMISSION_AT,
        },
    }


class PropFirmSupervisor:
    """
    Enforces the prop account rules.

    Reads the breach dict the engine already produced. Does not recompute it -
    two implementations of a drawdown rule is one more than the number that can
    be right.
    """

    name = "prop_firm"

    def review(self, result: Any) -> Verdict:
        raise NotImplementedError("tier2_supervisors: not implemented yet")


class OOSValidationSupervisor:
    """
    The Phase 3 gate: does the edge survive out of sample?

    Compares an in-sample (Databento) result against an out-of-sample (NT8)
    one and judges whether the difference is degradation within tolerance or
    collapse. Also responsible for catching the subtler failure: an OOS window
    that has already been looked at during selection is no longer out of
    sample, however it is labelled.
    """

    name = "oos_validation"

    def review(self, in_sample: Any, out_of_sample: Any) -> Verdict:
        raise NotImplementedError("tier2_supervisors: not implemented yet")


SUPERVISORS = (PropFirmSupervisor, OOSValidationSupervisor)


def review_all(result: Any, oos_result: Any | None = None) -> list[Verdict]:
    """Run every supervisor. A single rejection is enough to fail the result."""
    raise NotImplementedError("tier2_supervisors: not implemented yet")


def main() -> int:
    raise NotImplementedError("tier2_supervisors: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
