#!/usr/bin/env python3
"""
tools/daily_portfolio_report.py - the 17:15 ET card, and what to do when it cannot be sent.

Location:  ~/src/trading/tools/daily_portfolio_report.py

    python3 tools/daily_portfolio_report.py
    python3 tools/daily_portfolio_report.py --dry-run
    python3 tools/daily_portfolio_report.py --topic system_health

Runs `tool_portfolio_eval.py --markdown` and hands the card to the
`#portfolio-mgmt` topic through `tools/bridge.py`. Driven by
`hermes-portfolio-eval.timer` at 17:15 ET, weekdays, after the 17:00 CME close.

WHY THE REPORT IS PRINTED EVEN WHEN THE SEND FAILS
--------------------------------------------------
A delivery failure - the chat id unset, the gateway down, Telegram refusing -
must not also destroy the report. The card is always written to stdout, which
under systemd means the journal, so `journalctl -u hermes-portfolio-eval` has
the evaluation even on the days nobody received it. A daily report that exists
only if the network cooperated is a daily report that silently has holes.

The exit code still reflects the failure. Delivered and not-delivered are
different outcomes and this never reports one as the other.

Exit codes:  0 evaluated and delivered · 1 evaluated, NOT delivered ·
             2 the evaluation itself could not run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
EVAL_SCRIPT = REPO_ROOT / "tools" / "tool_portfolio_eval.py"
ET = ZoneInfo("America/New_York")


def run_evaluation(extra: list[str]) -> tuple[int, str, str]:
    argv = [str(VENV_PYTHON), str(EVAL_SCRIPT), "--markdown", *extra]
    done = subprocess.run(argv, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=600, check=False)
    return done.returncode, done.stdout, done.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce the daily portfolio card and deliver it.")
    parser.add_argument("--topic", default="portfolio_mgmt",
                        help="bridge topic to deliver to")
    parser.add_argument("--dry-run", action="store_true",
                        help="produce and print the card; deliver nothing")
    parser.add_argument("--sessions", type=int, default=None,
                        help="override the matrix window")
    args = parser.parse_args(argv)

    extra: list[str] = []
    if args.sessions is not None:
        extra += ["--sessions", str(args.sessions)]

    stamp = datetime.now(ET)
    subject = f"Daily portfolio evaluation — {stamp:%Y-%m-%d} (17:15 ET)"

    try:
        code, report, errors = run_evaluation(extra)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"daily report: the evaluation could not run: {exc}",
              file=sys.stderr)
        return 2

    if code == 2 or not report.strip():
        print(f"daily report: the evaluation could not run "
              f"(exit {code})", file=sys.stderr)
        if errors.strip():
            print(errors.strip(), file=sys.stderr)
        return 2

    # The card goes to the journal FIRST, so a delivery failure never costs
    # us the evaluation itself.
    print(f"==== {subject} ====")
    print(report)
    if errors.strip():
        print(errors.strip(), file=sys.stderr)

    if args.dry_run:
        print("---- dry run: nothing delivered ----")
        return 0

    try:
        from tools.bridge import BridgeError, load, send      # noqa: PLC0415
        config, _ = load(strict=False)
        sent, detail = send(config, args.topic, report, subject)
    except BridgeError as exc:
        print(f"---- NOT DELIVERED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                                   # noqa: BLE001
        print(f"---- NOT DELIVERED: {exc}", file=sys.stderr)
        return 1

    if sent != 0:
        print(f"---- NOT DELIVERED: {detail}", file=sys.stderr)
        return 1
    print(f"---- delivered: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
