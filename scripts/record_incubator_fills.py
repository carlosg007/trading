#!/usr/bin/env python3
"""
Record NT8 forward fills into the incubator ledger.

Location:  ~/src/trading/scripts/record_incubator_fills.py
Reads:     /mnt/backtest/artifacts/incubator_logs/*, config/portfolios.json
Writes:    data/incubator_ledger.json, and ONLY under --write.

    python3 scripts/record_incubator_fills.py                  # show, write nothing
    python3 scripts/record_incubator_fills.py --write          # record them
    python3 scripts/record_incubator_fills.py --logs /tmp/nt8_export.csv
    python3 scripts/record_incubator_fills.py --cost-basis log --write

THE FIRST HALF OF THE POST-MARKET JOB
=====================================
This produces what `scripts/incubator_tracker.py` grades. Run them in that
order: fills become closed trades here, the tracker scores those trades against
the four promotion criteria and, with `--auto-promote`, acts. Two commands
rather than one because they fail differently and an operator has to be able to
run the second without re-reading the exports - and because a recorder that
promoted would make the trade list and the decision the same commit.

The rule lives in `portfolio/incubator_recorder.py`; this file is the CLI, the
table and the exit code. Nothing is computed here.

DOING NOTHING IS THE DEFAULT
============================
With no flags it reads, pairs, and prints what it WOULD add. `--write` is the
only thing that touches the ledger, the same shape `--auto-promote` has on the
tracker: the recoverable mode is the one you get by forgetting a flag.

EXIT CODES
==========
    0  ran cleanly, whether or not anything was recorded
    1  the log directory does not exist, or the ledger/config could not be read
    2  ran, but something in the exports could not be used - unattributed rows,
       an unreadable file, a symbol with no contract spec

2 is not a crash and it is not success. A cron job that exits 0 on a run that
silently dropped a day of fills is how a strategy reaches its fourteenth
session with nine trades on the board and nobody asking why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mdlib.env import load_env                                     # noqa: E402

load_env()

from portfolio.config_loader import (                              # noqa: E402
    DEFAULT_CONFIG_PATH,
    PortfolioConfigError,
    load_portfolio_config,
)
from portfolio.incubator_recorder import (                         # noqa: E402
    COST_BASES,
    INCUBATOR_LOG_DIR,
    RecorderError,
    record,
)
from portfolio.promotion_daemon import (                           # noqa: E402
    DEFAULT_LEDGER_PATH,
    PromotionError,
)

HEADER = ["STRATEGY", "ACCOUNT", "NEW", "KNOWN", "LEDGER", "FIRST", "LAST"]


def _entry_span(entry: dict) -> tuple[str, str]:
    stamps = sorted(str(t.get("exit_time") or t.get("entry_time") or "")
                    for t in (entry.get("trades") or [])
                    if isinstance(t, dict))
    stamps = [s for s in stamps if s]
    if not stamps:
        return "—", "—"
    return stamps[0][:10], stamps[-1][:10]


def format_table(report: dict) -> str:
    """One row per strategy the run touched, plus the ledger's own totals."""
    built = report["built"]
    merged = report["merged"]
    ledger = report["ledger"]["strategies"]
    strategies = sorted(set(merged["added"]) | set(merged["skipped"]))
    if not strategies:
        return ("INCUBATOR FILL RECORDER\n"
                "  no attributable trades in these exports — nothing to record.")

    body = []
    for sid in strategies:
        entry = ledger.get(sid) or {}
        first, last = _entry_span(entry)
        body.append([
            sid,
            str(built["portfolios"].get(sid) or entry.get("portfolio") or "—"),
            f"{merged['added'].get(sid, 0):,}",
            f"{merged['skipped'].get(sid, 0):,}",
            f"{len(entry.get('trades') or []):,}",
            first,
            last,
        ])

    widths = [max(len(HEADER[i]), max(len(r[i]) for r in body))
              for i in range(len(HEADER))]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    rule = "  ".join("-" * w for w in widths)
    out = ["INCUBATOR FILL RECORDER", "", line(HEADER), rule]
    out += [line(cells) for cells in body]
    out.append(rule)
    return "\n".join(out)


def format_problems(report: dict) -> str:
    """Everything the run could NOT use, named. Counted, never dropped."""
    built = report["built"]
    lines = []
    for err in report["read_errors"]:
        lines.append(f"  UNREADABLE  {err}")
    for symbol, count in sorted(built["unpriceable_symbols"].items()):
        lines.append(f"  NO SPEC     {symbol}: {count} row(s) skipped — add it "
                     f"to backtest/specs.py; a guessed multiplier scales every "
                     f"P&L figure for that contract")
    for row in built["unattributed"][:10]:
        lines.append(f"  UNATTRIBUTED {row['source_log']} row {row['row']} "
                     f"({row.get('symbol') or 'no instrument'}): {row['reason']}")
    extra = len(built["unattributed"]) - 10
    if extra > 0:
        lines.append(f"  UNATTRIBUTED ... and {extra} more")
    for problem in built["problems"][:10]:
        lines.append(f"  UNPAIRABLE  {problem}")
    for refusal in report["merged"]["refused"]:
        lines.append(f"  REFUSED     {refusal}")
    for key, qty in sorted(built["open_positions"].items()):
        # Not a problem — a fact. An open position has no realised P&L and is
        # deliberately not a trade, and the count is the difference between
        # "flat" and "holding four".
        lines.append(f"  OPEN        {key}: {qty} contract(s) still open, not "
                     f"counted as a closed trade")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record_incubator_fills.py",
        description="Turn NT8 forward fills into closed trades on the "
                    "incubator ledger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run scripts/incubator_tracker.py afterwards to grade them.")
    parser.add_argument("--logs", default=str(INCUBATOR_LOG_DIR),
                        help=f"NT8 export file or directory "
                             f"(default: {INCUBATOR_LOG_DIR})")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="path to config/portfolios.json")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH,
                        help=f"path to the incubator ledger "
                             f"(default: {DEFAULT_LEDGER_PATH})")
    parser.add_argument("--cost-basis", default="auto", choices=COST_BASES,
                        help="how commissions are applied to a forward P&L. "
                             "auto: a declared-net figure stands, a gross one "
                             "pays the log's commission or the spec's round "
                             "turn. log: the export is already net. specs: "
                             "always the spec's round turn")
    parser.add_argument("--write", action="store_true",
                        help="WRITE the ledger. Without it nothing is changed")
    parser.add_argument("--json", dest="json_out", metavar="PATH",
                        help="also write the full report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_portfolio_config(args.config)
    except PortfolioConfigError as exc:
        print(f"REFUSING TO RUN: {exc}", file=sys.stderr)
        return 1

    try:
        report = record(log_target=args.logs, config=config,
                        ledger_path=args.ledger,
                        cost_basis=args.cost_basis, write=args.write)
    except (RecorderError, PromotionError) as exc:
        print(f"REFUSING TO RUN: {exc}", file=sys.stderr)
        return 1

    print(format_table(report))
    problems = format_problems(report)
    if problems:
        print()
        print(problems)

    built, merged = report["built"], report["merged"]
    added = sum(merged["added"].values())
    print(f"\n{len(report['logs'])} export(s), {report['rows']:,} row(s) read; "
          f"{added:,} new closed trade(s) on {len(merged['added'])} strategy "
          f"(costs: {built['cost_basis']}).")
    if report["wrote"]:
        print(f"ledger written: {report['written']}")
    elif added or merged["created"]:
        print("NOTHING WAS WRITTEN — pass --write to record these trades.")

    if args.json_out:
        out = {k: v for k, v in report.items() if k != "ledger"}
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n",
                                       encoding="utf-8")
        print(f"report written: {args.json_out}")

    unusable = (report["read_errors"] or built["unattributed"]
                or built["unpriceable_symbols"] or built["problems"]
                or merged["refused"])
    return 2 if unusable else 0


if __name__ == "__main__":
    raise SystemExit(main())
