#!/usr/bin/env python3
"""
test_tier2.py - the Tier 2 risk and compliance supervisors.

Location:  ~/src/trading/tests/test_tier2.py

Run:  python tests/test_tier2.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

The central claim
-----------------
The consistency rule must FAIL a strategy carried by one outlier session and
PASS an otherwise identical one whose profit is spread across days. Both mock
books below earn the SAME total profit and clear the same profit target, so the
only thing separating them is concentration. If the rule were broken, both
would pass and nothing else in the suite would notice.

Thresholds come from the ruleset, so the rule is exercised at both the 30%
value committed in `compliance_rules/fundednext_rapid.json` and at an explicit
40% override. A test that only ever ran one number could not tell a working
threshold from a hardcoded one.

What else is checked
--------------------
- The boundary is strict: exactly at the limit passes, a hair above fails.
- A losing book is NOT_ASSESSABLE, never PASS. The share of a non-positive
  total is meaningless, and reporting it as a pass is how a failing strategy
  collects a clean verdict.
- Trailing drawdown is measured against the trailing peak, not the start.
- Robustness fails on `None` - not measured is not the same as passed.
- The lifecycle machine covers every band, including the 1.35x-1.50x range the
  specification left undefined.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tier2_supervisors import (  # noqa: E402
    RulesetError, evaluate_compliance, evaluate_lifecycle_state,
    evaluate_robustness, load_ruleset,
)

FAILURES: list[str] = []
RULESET = Path(__file__).resolve().parent.parent / "compliance_rules" / "fundednext_rapid.json"
BALANCE = 100_000.0


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def book(daily_pnl: list[float], start_day: int = 1) -> pd.DataFrame:
    """
    A mock trade log: one trade per day, exiting that day.

    Only exit_time and pnl matter to the supervisor; the other columns are
    present so the frame has the shape the engine actually produces.
    """
    rows = []
    for i, pnl in enumerate(daily_pnl):
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=start_day + i)
        rows.append({
            "entry_time": ts, "exit_time": ts, "symbol": "ES",
            "direction": 1, "entry_price": 4000.0, "exit_price": 4000.0,
            "gross_pnl": pnl, "costs": 0.0, "pnl": pnl,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def test_consistency_rule() -> None:
    """The headline claim: concentration fails, diversification passes."""
    print("\nconsistency rule - the central claim")

    # Both books earn 12,000 (12% of a 100k account, clearing the 8% target).
    # CONCENTRATED: one day is 8,000 of it = 66.7% of total profit.
    concentrated = book([8000.0] + [500.0] * 8)
    # DIVERSIFIED: the same 12,000 spread evenly, 1,333.33 per day = 11.1%.
    diversified = book([12000.0 / 9] * 9)

    conc_total = concentrated["pnl"].sum()
    div_total = diversified["pnl"].sum()
    check("both mock books earn the same total",
          abs(conc_total - div_total) < 1e-6, f"{conc_total:,.2f}")

    for limit in (30.0, 40.0):
        c = evaluate_compliance(concentrated, RULESET, initial_balance=BALANCE,
                                consistency_pct=limit)
        d = evaluate_compliance(diversified, RULESET, initial_balance=BALANCE,
                                consistency_pct=limit)
        cc, dc = c["checks"]["consistency"], d["checks"]["consistency"]

        check(f"@{limit:.0f}%: concentrated FAILS consistency",
              cc["status"] == "FAIL",
              f"best day {cc.get('best_day_share_pct', float('nan')):.1f}% of total")
        check(f"@{limit:.0f}%: diversified PASSES consistency",
              dc["status"] == "PASS",
              f"best day {dc.get('best_day_share_pct', float('nan')):.1f}% of total")
        check(f"@{limit:.0f}%: concentrated verdict is FAIL overall",
              c["verdict"] == "FAIL")
        check(f"@{limit:.0f}%: failure names the consistency rule",
              any("consistency" in f for f in c["failures"]),
              c["failures"][0][:60] if c["failures"] else "no reasons given")

    # The diversified book should clear everything, not merely consistency.
    d = evaluate_compliance(diversified, RULESET, initial_balance=BALANCE)
    check("diversified book passes the whole ruleset", d["verdict"] == "PASS",
          f"failures: {d['failures']}")


def test_consistency_boundary() -> None:
    """Exactly at the limit passes; a hair above fails."""
    print("\nconsistency boundary")
    # Total 10,000 with a best day of exactly 3,000 == 30.0%.
    exact = book([3000.0] + [1000.0] * 7)
    r = evaluate_compliance(exact, RULESET, initial_balance=BALANCE,
                            consistency_pct=30.0)
    c = r["checks"]["consistency"]
    check("exactly at the limit passes", c["status"] == "PASS",
          f"share {c['best_day_share_pct']:.4f}%")

    over = book([3001.0] + [1000.0] * 7)
    c2 = evaluate_compliance(over, RULESET, initial_balance=BALANCE,
                             consistency_pct=30.0)["checks"]["consistency"]
    check("a hair above the limit fails", c2["status"] == "FAIL",
          f"share {c2['best_day_share_pct']:.4f}%")


def test_losing_book_is_not_a_pass() -> None:
    """A non-positive total makes the share meaningless - never report PASS."""
    print("\nnon-positive profit")
    losing = book([2000.0, -5000.0, 1000.0])
    r = evaluate_compliance(losing, RULESET, initial_balance=BALANCE)
    c = r["checks"]["consistency"]
    check("losing book is NOT_ASSESSABLE, not PASS",
          c["status"] == "NOT_ASSESSABLE", c.get("reason", "")[:60])
    check("overall verdict is FAIL", r["verdict"] == "FAIL")

    empty = evaluate_compliance(pd.DataFrame(columns=["exit_time", "pnl"]),
                                RULESET, initial_balance=BALANCE)
    check("empty book is FAIL, not PASS", empty["verdict"] == "FAIL",
          empty["checks"]["consistency"]["status"])


def test_profit_target_and_drawdown() -> None:
    print("\nprofit target and trailing drawdown")
    # 12,000 profit on 100k clears the 8% target.
    r = evaluate_compliance(book([12000.0 / 9] * 9), RULESET,
                            initial_balance=BALANCE)
    check("profit target reached", r["checks"]["profit_target"]["status"] == "PASS",
          f"{r['checks']['profit_target']['achieved_pct']:.2f}%")

    short = evaluate_compliance(book([500.0] * 4), RULESET, initial_balance=BALANCE)
    check("profit target short is FAIL",
          short["checks"]["profit_target"]["status"] == "FAIL",
          f"{short['checks']['profit_target']['achieved_pct']:.2f}% of 8%")

    # Run up 20k, then give back 12k. Drawdown measures from the PEAK (120k),
    # so 12k/120k = 10% and breaches the 8% limit - even though the account is
    # still 8k above where it started. Measuring from the start would give 0%.
    path = book([20000.0, -12000.0])
    dd = evaluate_compliance(path, RULESET, initial_balance=BALANCE)
    ddc = dd["checks"]["max_trailing_drawdown"]
    check("drawdown measured from the trailing peak, not the start",
          ddc["status"] == "FAIL" and abs(abs(ddc["worst_drawdown_pct"]) - 10.0) < 1e-9,
          f"{ddc['worst_drawdown_pct']:.2f}% against {ddc['limit_pct']}%")

    check("reconstruction is disclosed",
          dd["equity_source"] == "reconstructed_from_trades"
          and dd["caveat"] is not None)

    supplied = evaluate_compliance(
        path, RULESET, initial_balance=BALANCE,
        equity=pd.Series([120000.0, 108000.0]))
    check("supplied equity curve is recorded as such",
          supplied["checks"]["max_trailing_drawdown"]["equity_source"]
          == "supplied_equity_curve")


def test_unenforced_rules_surfaced() -> None:
    """A PASS must not read as blanket compliance."""
    print("\nunevaluated rules")
    r = evaluate_compliance(book([12000.0 / 9] * 9), RULESET,
                            initial_balance=BALANCE)
    check("rules not evaluated here are listed",
          "max_daily_loss" in r["unenforced_rules"],
          str(r["unenforced_rules"]))


def test_ruleset_errors() -> None:
    print("\nruleset loading")
    for label, path in (("missing file", "compliance_rules/nope.json"),
                        ("not a ruleset", "README.md")):
        try:
            load_ruleset(path)
            check(f"raises on {label}", False)
        except RulesetError:
            check(f"raises on {label}", True)
        except Exception as e:
            check(f"raises on {label}", False, type(e).__name__)

    rs = load_ruleset(RULESET)
    check("ruleset carries a consistency value",
          rs["rules"]["consistency"]["value"] is not None,
          f"{rs['rules']['consistency']['value']}%")


def test_robustness() -> None:
    print("\nrobustness gate")
    ok = evaluate_robustness(0.62, -6.5, max_allowed_dd=8.0)
    check("clears both bars", ok["verdict"] == "PASS")

    low = evaluate_robustness(0.49, -6.5, max_allowed_dd=8.0)
    check("WFO just below 0.50 fails", low["verdict"] == "FAIL",
          low["failures"][0][:50])
    edge = evaluate_robustness(0.50, -8.0, max_allowed_dd=8.0)
    check("exactly at both bars passes", edge["verdict"] == "PASS")

    deep = evaluate_robustness(0.80, -12.0, max_allowed_dd=8.0)
    check("MC drawdown over the limit fails", deep["verdict"] == "FAIL")

    check("sign convention does not matter",
          evaluate_robustness(0.8, -6.0, 8.0)["verdict"]
          == evaluate_robustness(0.8, 6.0, 8.0)["verdict"] == "PASS")

    check("undefined WFO fails, not skipped",
          evaluate_robustness(None, -1.0, 8.0)["verdict"] == "FAIL")
    check("undefined MC fails, not skipped",
          evaluate_robustness(0.9, None, 8.0)["verdict"] == "FAIL")
    check("NaN WFO fails",
          evaluate_robustness(float("nan"), -1.0, 8.0)["verdict"] == "FAIL")


def test_lifecycle() -> None:
    print("\nlifecycle state machine")
    cases = [
        ("well inside the envelope", 0.5, 1.0, 0.1, "ACTIVE", False),
        ("exactly at 1.00x", 1.0, 1.0, 0.1, "ACTIVE", False),
        ("just over 1.00x", 1.01, 1.0, 0.1, "PAUSED", False),
        ("at 1.35x", 1.35, 1.0, 0.1, "PAUSED", False),
        ("in the undefined 1.35-1.50 band", 1.42, 1.0, 0.1, "PAUSED", True),
        ("at 1.50x", 1.50, 1.0, 0.1, "PAUSED", True),
        ("just over 1.50x", 1.51, 1.0, 0.1, "DECOMMISSIONED", False),
        ("negative EV inside the envelope", 0.2, 1.0, -0.01,
         "DECOMMISSIONED", False),
    ]
    for label, live, hist, ev, expected, escalated in cases:
        r = evaluate_lifecycle_state(live, hist, ev)
        check(f"{label} -> {expected}", r["state"] == expected,
              f"got {r['state']} at ratio {r['ratio']:.2f}")
        if escalated:
            check(f"{label} is flagged escalated", r["escalated"] is True)

    check("negative EV outranks a healthy drawdown",
          evaluate_lifecycle_state(0.1, 1.0, -0.5)["state"] == "DECOMMISSIONED")

    zero = evaluate_lifecycle_state(0.4, 0.0, 0.1)
    check("zero historical envelope never resolves ACTIVE",
          zero["state"] != "ACTIVE", zero["state"])

    for bad in (None, float("nan")):
        try:
            evaluate_lifecycle_state(bad, 1.0, 0.1)
            check(f"rejects {bad!r} input", False)
        except ValueError:
            check(f"rejects {bad!r} input", True)

    check("reasons are always given",
          all(evaluate_lifecycle_state(l, h, e)["reasons"]
              for l, h, e in ((0.5, 1.0, 0.1), (1.2, 1.0, 0.1), (2.0, 1.0, 0.1))))


if __name__ == "__main__":
    test_consistency_rule()
    test_consistency_boundary()
    test_losing_book_is_not_a_pass()
    test_profit_target_and_drawdown()
    test_unenforced_rules_surfaced()
    test_ruleset_errors()
    test_robustness()
    test_lifecycle()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
