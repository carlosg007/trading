#!/usr/bin/env python3
"""
The post-session incubator audit: score every forward-incubating strategy
against the promotion criteria, print the table, and — only when asked —
graduate the ones that cleared.

Location:  ~/src/trading/scripts/incubator_tracker.py
Reads:     data/incubator_ledger.json, config/portfolios.json
Writes:    both, and ONLY under --auto-promote.

    python3 scripts/incubator_tracker.py                      # evaluate, print
    python3 scripts/incubator_tracker.py --dry-run            # the same, said out loud
    python3 scripts/incubator_tracker.py --auto-promote       # act on it
    python3 scripts/incubator_tracker.py --ledger /tmp/l.json --config /tmp/c.json

WHAT IT DOES AND WHAT IT REFUSES TO DO
======================================
The rule lives in `portfolio/promotion_daemon.py` and this script computes
nothing. It resolves which account each strategy is on, hands the entry and
that account's risk envelope to `evaluate_strategy_promotion`, and prints what
came back. A CLI that re-derived a profit factor would be free to disagree with
the daemon it is announcing, and the two would be compared by nobody.

**Doing nothing is the default.** With no flags it evaluates and prints; only
`--auto-promote` writes. `--dry-run` and `--auto-promote` together resolve to
the dry run, loudly, rather than to the write — an operator who typed both
meant to be careful, and the failure mode of guessing the other way is an
account move nobody sanctioned.

WHERE THE ACCOUNT COMES FROM, AND WHY THE LEDGER IS NOT A FALLBACK
==================================================================
A strategy's account is whichever incubator portfolio's `active_strategies`
names it in `config/portfolios.json`. That is the table CrossTrade routes on,
so it is the only answer that describes where the orders actually go — and it
is also the only answer a promotion can ACT on, because a promotion removes the
strategy from that list. There is deliberately no fallback to the `portfolio`
field a ledger entry may carry: taking it would produce a PROMOTE verdict for a
strategy no incubator portfolio holds, which `promote_strategy` then refuses,
so the row would clear every criterion on the table and fail on the way out.

A strategy the routing table does not name is UNROUTED and is evaluated
against nothing. If the ledger names a portfolio, the note says which — "the
ledger records Incubator-Even and the routing table names it nowhere" is a
one-line description of the edit somebody has to make, whereas "not assigned"
would leave a reader to find the field themselves. A DISAGREEMENT between the
two is the same row for the same reason: the readings are "the ledger is stale"
and "the routing table was edited by hand", and under the first a promotion
would move a strategy off an account it was never on.

THE STATUS COLUMN IS NOT A GATE VERDICT
=======================================
`PROMOTE` means the four loose criteria cleared on recorded numbers. It is not
a certification — Stage 3 already did that, on historical bars — and it is not
permission to trade: the account it moves to is a prop EVALUATION account whose
lockouts are enforced by CrossTrade NAM against a live balance, not here.

THE DISCORD POST IS OPTIONAL AND SILENT WHEN UNCONFIGURED
==========================================================
`$DISCORD_WEBHOOK_URL` (or this repository's own `$BT_DISCORD_WEBHOOK`, which
every other card already uses) turns on a summary embed. The transport is
`backtest.discord_reporter.post_embed`, reused rather than reimplemented, so
the URL is treated as the credential it is: never printed, never echoed into a
failure message, only its host. A failed post is REPORTED and never raised —
losing the audit because a webhook was down would be the notifier deciding the
run failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load ~/src/trading/.env explicitly, the same call every runner in `backtest/`
# carries. It is NOT redundant: without it this script still ends up with the
# file's variables, because something down the import chain loads them — which
# means the webhook this script posts to would be supplied by an import side
# effect rather than by anything written here, and would silently stop being
# supplied the day that import moved. Existing environment variables WIN; the
# rules live in mdlib/env.py.
from mdlib.env import (                                            # noqa: E402
    WEBHOOK_HINT, discord_webhook, load_env,
)

load_env()

from portfolio.config_loader import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    INCUBATOR_ACCOUNT_TYPE,
    PortfolioConfigError,
    load_portfolio_config,
)
from portfolio.promotion_daemon import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    PROMOTION_ROUTES,
    STATUS_GRADUATED,
    STATUS_INCUBATING,
    PromotionError,
    evaluate_strategy_promotion,
    load_ledger,
    promote_strategy,
)

# `$DISCORD_WEBHOOK_URL` is the name the objective specifies and
# `$BT_DISCORD_WEBHOOK` is the one the four pipeline cards read; both are
# honoured, along with `$DISCORD_WEBHOOK`, so this script works with the
# environment an operator already has. The chain and its PRECEDENCE come from
# `mdlib.env` rather than being spelled out here - this module used to try
# `$DISCORD_WEBHOOK_URL` first while `backtest/discord_reporter.py` read
# `$BT_DISCORD_WEBHOOK` alone, so an operator with both set had two cards
# posting to two different channels with nothing saying so.

EMBED_COLOR_PROMOTED = 0x2ECC71
EMBED_COLOR_QUIET = 0x5865F2
EMBED_COLOR_ERROR = 0xE74C3C
# Discord caps an embed at 6000 characters and wraps a code block that overruns
# the viewport. The same reasoning as the Stage 3 card: rows past the cap are
# COUNTED on the embed, never silently dropped.
MAX_EMBED_ROWS = 20


# --------------------------------------------------------------------------
# resolving each strategy to an account
# --------------------------------------------------------------------------

def incubator_assignments(config: dict) -> dict[str, list[str]]:
    """`{strategy_id: [portfolio_id, ...]}` across the INCUBATOR track only.

    A list rather than a single id because a strategy named on two portfolios
    is a real state of the file and has to be reported as the conflict it is,
    not resolved by taking the first."""
    out: dict[str, list[str]] = {}
    for pid, portfolio in config["portfolios"].items():
        if portfolio.get("account_type") != INCUBATOR_ACCOUNT_TYPE:
            continue
        for sid in portfolio.get("active_strategies") or []:
            out.setdefault(sid, []).append(pid)
    return out


def resolve_account(strategy_id: str, entry: dict,
                    assignments: dict[str, list[str]]) -> tuple[str | None, str]:
    """
    `(portfolio_id, "config")`, or `(None, why not)`.

    Only the routing table can answer this — see the module docstring. The
    ledger's own `portfolio` field is read for the NOTE and never as a
    fallback, because an account it alone names is one the promotion cannot
    remove the strategy from.
    """
    declared = entry.get("portfolio") or entry.get("source_portfolio")
    declared = declared if isinstance(declared, str) and declared else None
    routed = assignments.get(strategy_id, [])

    if len(routed) > 1:
        # Both would size it against the same signal and the net position
        # would be double what either risk profile describes. The loader
        # raises on this too; here it is a row, so the rest of the table
        # still prints.
        return None, (f"assigned to {len(routed)} incubator portfolios "
                      f"({sorted(routed)})")
    if routed:
        pid = routed[0]
        if declared and declared != pid:
            return None, (f"the routing table has it on {pid} and the ledger "
                          f"records {declared}")
        return pid, "config"
    if declared:
        return None, (f"the ledger records {declared!r} and the routing table "
                      f"names it nowhere — add it to that portfolio's "
                      f"active_strategies")
    return None, "not assigned to any incubator portfolio"


# --------------------------------------------------------------------------
# the evaluation pass
# --------------------------------------------------------------------------

def evaluate_all(ledger: dict, config: dict) -> list[dict[str, Any]]:
    """
    One row per ledger entry, in ledger order.

    EVERY entry gets a row, including the already-graduated ones and the ones
    that could not be resolved to an account. A table shorter than the ledger
    reads as a complete audit of fewer strategies, and "this was skipped" is a
    different statement from "this was evaluated and did not clear".
    """
    assignments = incubator_assignments(config)
    rows: list[dict[str, Any]] = []
    for strategy_id, entry in ledger.get("strategies", {}).items():
        row: dict[str, Any] = {
            "strategy_id": strategy_id,
            "account": None,
            "status": None,
            "report": None,
            "note": "",
        }
        if not isinstance(entry, dict):
            row["status"] = "ERROR"
            row["note"] = (f"the ledger entry is a "
                           f"{type(entry).__name__}, not an object")
            rows.append(row)
            continue

        ledger_status = str(entry.get("status", STATUS_INCUBATING)).upper()
        account, provenance = resolve_account(strategy_id, entry, assignments)
        row["account"] = account or "—"

        if ledger_status == STATUS_GRADUATED:
            row["status"] = "GRADUATED"
            row["account"] = entry.get("target_portfolio") or row["account"]
            row["note"] = f"graduated {entry.get('graduated_at', 'at an unrecorded time')}"
            rows.append(row)
            continue
        if ledger_status != STATUS_INCUBATING:
            row["status"] = "SKIPPED"
            row["note"] = f"ledger status is {ledger_status!r}"
            rows.append(row)
            continue
        if account is None:
            row["status"] = "UNROUTED"
            row["note"] = provenance
            rows.append(row)
            continue

        try:
            report = evaluate_strategy_promotion(
                strategy_id, entry, config["portfolios"][account])
        except PromotionError as exc:
            row["status"] = "ERROR"
            row["note"] = str(exc)
            rows.append(row)
            continue

        row["report"] = report
        row["status"] = "PROMOTE" if report["passed"] else "HOLD"
        row["note"] = ("" if report["passed"]
                       else "failed: " + ", ".join(report["failed"]))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

HEADER = ["STRATEGY", "ACCOUNT", "DAYS", "TRADES", "REAL PF",
          "MAX DD / LIMIT", "STATUS"]


def _cell_metric(report: dict | None, key: str, decimals: int = 0) -> str:
    if report is None:
        return "—"
    value = report["metrics"].get(key)
    if value is None:
        return "n/r"          # NOT RECORDED, which is not zero
    return f"{value:,.{decimals}f}"


def _cell_dd(report: dict | None) -> str:
    if report is None:
        return "—"
    metrics = report["metrics"]
    limit = metrics.get("allowable_dd")
    dd = metrics.get("max_drawdown")
    left = "n/r" if dd is None else f"${abs(dd):,.0f}"
    right = "?" if limit is None else f"${limit:,.0f}"
    return f"{left} / {right}"


def _cell_pf(report: dict | None) -> str:
    if report is None:
        return "—"
    metrics = report["metrics"]
    if metrics.get("realized_pf") is None:
        # The profiler's 999 sentinel means "no losing trade", not a measured
        # factor. It is not printed as a number here for the same reason the
        # Stage 3 card renders it as `--`: it would be the strongest figure on
        # the table attached to the least evidence.
        return "--" if metrics.get("pf_undefined") else "n/r"
    return f"{metrics['realized_pf']:,.2f}"


def format_table(rows: list[dict[str, Any]]) -> str:
    """
    The ASCII status table. Columns size to their widest CELL, so a long
    strategy id widens its own column rather than being clipped — the same rule
    `backtest/pipeline.py::leaderboard` follows, because these are read as a
    sequence with the stage tables and a column that moves between them makes
    two tables of the same strategies look like tables of different ones.
    """
    if not rows:
        return ("INCUBATOR PERFORMANCE TRACKER\n"
                "  the ledger holds no strategies — nothing to evaluate.")

    body = []
    for row in rows:
        report = row.get("report")
        body.append([
            row["strategy_id"],
            str(row["account"]),
            _cell_metric(report, "days_active"),
            _cell_metric(report, "trade_count"),
            _cell_pf(report),
            _cell_dd(report),
            row["status"],
        ])

    widths = [max(len(HEADER[i]), max(len(r[i]) for r in body))
              for i in range(len(HEADER))]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    rule = "  ".join("-" * w for w in widths)
    out = ["INCUBATOR PERFORMANCE TRACKER", "", line(HEADER), rule]
    out += [line(cells) for cells in body]
    out.append(rule)
    return "\n".join(out)


def format_detail(rows: list[dict[str, Any]]) -> str:
    """Every criterion, per strategy that was actually evaluated. The table
    says which strategies cleared; this says which bar the others missed, and
    those are fixed by different work."""
    blocks = []
    for row in rows:
        report = row.get("report")
        if report is None:
            if row["note"]:
                blocks.append(f"{row['strategy_id']} [{row['status']}] "
                              f"{row['note']}")
            continue
        lines = [f"{row['strategy_id']} [{row['status']}] "
                 f"on {row['account']} · metrics {report['metrics']['source']}"]
        lines += [f"    {reason}" for reason in report["reasons"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# promotion
# --------------------------------------------------------------------------

def run_promotions(rows: list[dict[str, Any]], config_path: str,
                   ledger_path: str) -> list[dict[str, Any]]:
    """
    Promote every row whose verdict is PROMOTE, one at a time, recording the
    outcome on the row.

    A failure does not end the pass. Three strategies clearing on the same
    evening is normal, and a read-only checkout or a lost race on the config
    must not throw away the other two promotions — the run reports what it did
    and exits non-zero.
    """
    promoted = []
    for row in rows:
        if row["status"] != "PROMOTE":
            continue
        try:
            moved = promote_strategy(row["strategy_id"], row["account"],
                                     config_path=config_path,
                                     ledger_path=ledger_path)
        except (PromotionError, PortfolioConfigError, OSError) as exc:
            row["status"] = "FAILED"
            row["note"] = f"promotion failed: {exc}"
            continue
        target = PROMOTION_ROUTES[row["account"]]
        row["status"] = "PROMOTED" if moved else "ALREADY"
        row["promoted_to"] = target
        row["note"] = (f"{row['account']} -> {target}" if moved
                       else f"already on {target}")
        if moved:
            row["account"] = target
            promoted.append(row)
    return promoted


# --------------------------------------------------------------------------
# discord
# --------------------------------------------------------------------------

def webhook_from_env(env: dict[str, str] | None = None) -> str | None:
    """The shared resolver. An empty value is UNSET, not a webhook."""
    return discord_webhook(env=env)


def build_embed(rows: list[dict[str, Any]], acted: bool) -> dict[str, Any]:
    """
    One embed: the counts, the table, and what was promoted.

    It transcribes the verdicts this run already produced and re-scores
    nothing, so the card can never announce a promotion the daemon refused.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    promoted = [r for r in rows if r["status"] == "PROMOTED"]
    failed = [r for r in rows if r["status"] in ("FAILED", "ERROR")]

    shown = rows[:MAX_EMBED_ROWS]
    table = format_table(shown).split("\n", 2)[2] if shown else "no strategies"
    hidden = len(rows) - len(shown)
    description = f"```\n{table}\n```"
    if hidden > 0:
        description += f"\n{hidden} further row(s) not shown."

    color = (EMBED_COLOR_ERROR if failed else
             EMBED_COLOR_PROMOTED if promoted else EMBED_COLOR_QUIET)
    fields = [{
        "name": "Evaluated",
        "value": (f"{len(rows)} ledger entries · "
                  + " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))),
        "inline": False,
    }]
    if promoted:
        fields.append({
            "name": "Promoted to prop",
            "value": "\n".join(
                f"• `{r['strategy_id']}` -> **{r.get('promoted_to')}**"
                for r in promoted),
            "inline": False,
        })
    elif not acted:
        # Said outright, because an empty promotion section on a card read the
        # morning after reads as "nothing qualified" when it may mean "nobody
        # passed --auto-promote".
        fields.append({
            "name": "Promotions",
            "value": "Evaluation only — no changes were written "
                     "(`--auto-promote` was not set).",
            "inline": False,
        })
    if failed:
        fields.append({
            "name": "Failed",
            "value": "\n".join(f"• `{r['strategy_id']}`: {r['note']}"
                               for r in failed[:5]),
            "inline": False,
        })

    return {
        "title": "🌱 Incubator Performance Tracker",
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": (
            f"criteria: >= 14 days · >= 14 trades · PF > 1.00 · "
            f"forward DD within the account envelope · "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")},
    }


def post_summary(rows: list[dict[str, Any]], acted: bool,
                 webhook: str | None) -> dict[str, Any]:
    """Post the embed, or say why not. Never raises: see the module docstring."""
    if not webhook:
        return {"ok": False, "skipped": True,
                "error": f"no webhook configured: {WEBHOOK_HINT}"}
    try:
        from backtest.discord_reporter import build_payload, post_embed
    except ImportError as exc:
        return {"ok": False, "skipped": True,
                "error": f"discord transport unavailable: {exc}"}
    result = post_embed(webhook, build_payload(build_embed(rows, acted)))
    result["skipped"] = False
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="incubator_tracker.py",
        description="Audit forward-incubating strategies against the "
                    "promotion criteria and, with --auto-promote, graduate "
                    "the ones that cleared.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help=f"portfolio routing table "
                             f"(default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH,
                        help=f"forward-trade ledger "
                             f"(default: {DEFAULT_LEDGER_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate and print, writing nothing. Wins over "
                             "--auto-promote if both are given.")
    parser.add_argument("--auto-promote", action="store_true",
                        help="promote every strategy that cleared all four "
                             "criteria")
    parser.add_argument("--no-discord", action="store_true",
                        help="never post, whatever the environment holds")
    parser.add_argument("--json", dest="json_out", metavar="PATH",
                        help="also write the full evaluation to this file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_portfolio_config(args.config)
        ledger = load_ledger(args.ledger)
    except (PortfolioConfigError, PromotionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rows = evaluate_all(ledger, config)

    act = args.auto_promote and not args.dry_run
    if args.auto_promote and args.dry_run:
        print("--dry-run and --auto-promote were both given: running the DRY "
              "RUN and writing nothing.\n")

    if act:
        run_promotions(rows, args.config, args.ledger)

    print(format_table(rows))
    detail = format_detail(rows)
    if detail:
        print("\n" + detail)

    eligible = [r for r in rows if r["status"] in ("PROMOTE", "PROMOTED")]
    failed = [r for r in rows if r["status"] in ("FAILED", "ERROR")]
    print()
    if act:
        print(f"{sum(1 for r in rows if r['status'] == 'PROMOTED')} strategy/"
              f"strategies promoted to prop.")
    elif eligible:
        print(f"{len(eligible)} strategy/strategies cleared every criterion. "
              f"Nothing was written — re-run with --auto-promote to graduate "
              f"them:\n"
              f"    python3 scripts/incubator_tracker.py --auto-promote")
    else:
        print("No strategy cleared every criterion; nothing to promote.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"generated_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
             "wrote_changes": act,
             "rows": rows}, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"evaluation written to {out}")

    if not args.no_discord:
        result = post_summary(rows, act, webhook_from_env())
        if result.get("skipped"):
            print(f"discord: not posted — {result['error']}")
        elif result["ok"]:
            print("discord: summary posted.")
        else:
            print(f"discord: post FAILED — {result['error']}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
