#!/usr/bin/env python3
"""
report_strategy_performance.py - every gate audit on disk as one table, and
one CSV.

Location:  ~/src/trading/scripts/report_strategy_performance.py

Why
---
Stage 3 writes `gate_audit_<SYMBOL>_<TF>.json` per certified configuration and
`stage3_audit_summary.json` per RUN. Neither answers "what does everything I
have ever certified look like side by side" - the per-pair files are
authoritative but there are 557 of them across eleven strategies, and each
campaign summary indexes only its own run. This reads them all.

READ-ONLY. It re-scores nothing: every number is transcribed from an audit
that already exists, so this table can never disagree with the verdicts it
indexes.

FOUR THINGS THE REQUEST GOT WRONG ABOUT THE ARTIFACTS, all resolved against
what is actually on disk rather than approximated:

1. **The path.** `/mnt/backtest/artifacts/gates/` does not exist. Audits live
   at `<artifacts>/pipeline/<strategy>/gate_audit_*.json`, one directory per
   strategy, plus whatever campaign directories an operator made
   (`q3_recert/`). This walks the artifacts root and finds them wherever they
   are.

2. **The metric keys.** There is no `oos_metrics`, `sharpe_ratio`,
   `max_drawdown` or `total_trades`. The out-of-sample block is
   `versions.<V>.metrics_holdout` (in-sample is `metrics_in_sample`) and the
   keys are `sharpe`, `profit_factor`, `max_drawdown_pct`, `win_rate` and
   `trade_count`. Both spellings are accepted so a schema change in either
   direction does not silently produce an empty column.

3. **`status` is a dict, not a string.** `{"A": "FAIL", "B": "PASS"}` - one
   audit certifies BOTH versions and they disagree routinely. A row per
   (audit, VERSION) is therefore the only shape that does not collapse a
   passing Version B into a failing Version A, which is why `Ver` is a column
   the request did not ask for.

4. **The scales.** `win_rate` is a FRACTION (0.42) and is rendered as a
   percent; `max_drawdown_pct` is already a percent and is NEGATIVE, because
   the engine signs drawdowns. Neither is re-derived - a second convention
   here would put a 42% win rate in a column headed 0.42 and nothing would
   raise.

THE UNSUFFIXED AUDIT IS SKIPPED WHEN A SUFFIXED SIBLING EXISTS.
Stage 3 writes both `gate_audit_<SYMBOL>_<TF>.json` and the unsuffixed
`gate_audit_<SYMBOL>.json`, and the second duplicates whichever timeframe ran
LAST. Counting both puts one configuration on the leaderboard twice, once
under a timeframe it may not belong to. `--include-unsuffixed` keeps them,
flagged.

Reads
-----
    `$BT_ARTIFACTS` (default /mnt/backtest/artifacts), recursively.

Writes
------
    `backtest/reports/strategy_performance_<UTC stamp>.csv` and refreshes
    `strategy_performance_latest.csv` beside it.

Usage
-----
    strat-perf                       # every audit, best profit factor first
    strat-perf --pass-only
    strat-perf --strategy keltner_trend_drift_20260901
    strat-perf --sort timestamp --limit 40
    strat-perf --no-csv              # console only
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import argparse                                                    # noqa: E402
import csv                                                         # noqa: E402
import json                                                        # noqa: E402
import os                                                          # noqa: E402
import re                                                          # noqa: E402
from datetime import datetime, timezone                            # noqa: E402
from typing import Any                                             # noqa: E402

#: `gate_audit_<SYMBOL>_<TF>.json`, and the unsuffixed form beside it.
AUDIT_RE = re.compile(r"^gate_audit_([A-Z0-9]+)(?:_([0-9]+[a-z]+))?\.json$")

#: The CSV contract. `Ver` is added to the requested columns because `status`
#: is per-version and A and B disagree routinely; without it two rows collapse
#: into one and the passing half disappears.
COLUMNS = ["Strategy", "Symbol", "TF", "Ver", "Status", "Profit_Factor",
           "Sharpe", "Max_DD_Pct", "Win_Rate_Pct", "Total_Trades",
           "Artifact_File"]

REPORT_DIR = PROJECT_ROOT / "backtest" / "reports"
LATEST = "strategy_performance_latest.csv"


def artifacts_root() -> Path:
    """`$BT_ARTIFACTS`, read at CALL time. Bound at import it would resolve
    before a caller that sets it has run."""
    return Path(os.environ.get("BT_ARTIFACTS", "/mnt/backtest/artifacts"))


def _num(value) -> float | None:
    """A float, or None - never a NaN masquerading as a measurement."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _first(block: dict, *names):
    """
    The first key present, in preference order.

    Both the real spelling and the one the request assumed are listed at every
    call site, so a rename in either direction produces a value rather than a
    silently empty column.
    """
    for name in names:
        if name in block:
            value = _num(block[name])
            if value is not None:
                return value
    return None


def metrics_for(version_block: dict, prefer_oos: bool = True) -> tuple[dict, str]:
    """
    `(metrics, which)` - the holdout block when it holds anything, else the
    in-sample one.

    OUT OF SAMPLE IS PREFERRED AND THE CHOICE IS REPORTED. An in-sample profit
    factor and a holdout one are different claims, and a table that mixed them
    silently under one heading would rank a configuration fitted on those bars
    above one measured on bars it had never seen.
    """
    order = (("metrics_holdout", "oos_metrics", "metrics_in_sample", "metrics")
             if prefer_oos else
             ("metrics_in_sample", "metrics", "metrics_holdout", "oos_metrics"))
    for name in order:
        block = version_block.get(name)
        if isinstance(block, dict) and block:
            return block, name
    return {}, "none"


def rows_from_audit(path: Path, prefer_oos: bool = True) -> list[dict[str, Any]]:
    """One row per VERSION in one audit file. A malformed file yields none."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    match = AUDIT_RE.match(path.name)
    symbol = blob.get("symbol") or (match.group(1) if match else "?")
    timeframe = blob.get("timeframe") or (match.group(2) if match else "?")
    strategy = blob.get("strategy") or path.parent.name
    status_map = blob.get("status")
    generated = blob.get("generated_utc") or ""

    out = []
    for version, block in sorted((blob.get("versions") or {}).items()):
        if not isinstance(block, dict):
            continue
        metrics, source = metrics_for(block, prefer_oos)
        # `status` is a per-version dict; a bare string is accepted for an
        # audit written before it became one.
        status = (status_map.get(version) if isinstance(status_map, dict)
                  else status_map) or "NOT EVALUATED"
        win = _first(metrics, "win_rate")
        out.append({
            "Strategy": strategy,
            "Symbol": str(symbol),
            "TF": str(timeframe or "?"),
            "Ver": str(version).upper(),
            "Status": str(status),
            "Profit_Factor": _first(metrics, "profit_factor"),
            "Sharpe": _first(metrics, "sharpe", "sharpe_ratio"),
            "Max_DD_Pct": _first(metrics, "max_drawdown_pct", "max_drawdown"),
            # A FRACTION on the artifact, a percent in the column. Converted
            # once, here, rather than at each renderer.
            "Win_Rate_Pct": None if win is None else win * 100.0,
            "Total_Trades": _first(metrics, "trade_count", "total_trades",
                                   "trades"),
            "Artifact_File": str(path),
            # Not CSV columns; used for sorting and the console note.
            "_generated": generated,
            "_metrics_source": source,
        })
    return out


def collect(root: Path, include_unsuffixed: bool = False,
            prefer_oos: bool = True) -> tuple[list[dict], dict[str, int]]:
    """
    Every audit under `root`, deduplicated.

    THE UNSUFFIXED FILE IS A DUPLICATE, not a separate configuration: Stage 3
    writes `gate_audit_<SYMBOL>.json` alongside the per-pair file and it holds
    whichever timeframe ran LAST. Kept, it puts one configuration on the
    leaderboard twice under a timeframe it may not belong to.
    """
    files = sorted(root.rglob("gate_audit_*.json"))
    suffixed: set[tuple[Path, str]] = set()
    for path in files:
        m = AUDIT_RE.match(path.name)
        if m and m.group(2):
            suffixed.add((path.parent, m.group(1)))

    rows: list[dict] = []
    counts = {"files": 0, "skipped_unsuffixed": 0, "unreadable": 0}
    for path in files:
        m = AUDIT_RE.match(path.name)
        if m and not m.group(2) and not include_unsuffixed:
            if (path.parent, m.group(1)) in suffixed:
                counts["skipped_unsuffixed"] += 1
                continue
        got = rows_from_audit(path, prefer_oos)
        if not got:
            counts["unreadable"] += 1
            continue
        counts["files"] += 1
        rows.extend(got)
    return rows, counts


def sort_rows(rows: list[dict], key: str) -> list[dict]:
    """
    Best first. A row with NO profit factor sorts LAST rather than as zero -
    "this audit recorded no metrics" is a finding about the artifact and must
    not be presented as the worst result.
    """
    if key == "timestamp":
        return sorted(rows, key=lambda r: r.get("_generated") or "",
                      reverse=True)
    if key == "sharpe":
        return sorted(rows, key=lambda r: (r["Sharpe"] is not None,
                                           r["Sharpe"] or 0.0), reverse=True)
    return sorted(rows, key=lambda r: (r["Profit_Factor"] is not None,
                                       r["Profit_Factor"] or 0.0), reverse=True)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _cell(value, spec: str = ".2f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def render(rows: list[dict], counts: dict, limit: int | None = None) -> str:
    headers = ["Strategy", "Sym", "TF", "V", "Status", "PF", "Sharpe",
               "MaxDD%", "Win%", "Trades"]
    body = []
    for r in (rows[:limit] if limit else rows):
        body.append([
            r["Strategy"], r["Symbol"], r["TF"], r["Ver"], r["Status"],
            _cell(r["Profit_Factor"]), _cell(r["Sharpe"]),
            _cell(r["Max_DD_Pct"]), _cell(r["Win_Rate_Pct"], ".1f"),
            _cell(r["Total_Trades"], ".0f"),
        ])

    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines = [
        "=" * min(sum(widths) + 3 * len(widths), 120),
        "STRATEGY PERFORMANCE — every gate audit on disk",
        "=" * min(sum(widths) + 3 * len(widths), 120),
        "",
    ]
    if not body:
        lines.append("no gate audits found.")
        return "\n".join(lines)

    lines.append(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * w for w in widths))
    lines.extend(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row))
                 for row in body)

    passed = sum(1 for r in rows if str(r["Status"]).upper() == "PASS")
    sources = sorted({r["_metrics_source"] for r in rows})
    lines += [
        "",
        f"{len(rows)} configuration(s) from {counts['files']} audit file(s): "
        f"{passed} PASS, {len(rows) - passed} not.",
        f"metrics read from: {', '.join(sources)}  "
        f"(holdout = out of sample; in-sample is the window the parameters "
        f"were fitted on)",
    ]
    if limit and len(rows) > limit:
        lines.append(f"showing {limit} of {len(rows)} — raise --limit or use "
                     f"the CSV for the rest.")
    # THE THIN-SAMPLE WARNING. Ranking on profit factor puts the smallest
    # samples first - a quadrant that never lost prints a huge factor over a
    # handful of trades - and a reader scanning the top of the table is
    # looking at exactly the rows least worth reading.
    shown = body and rows[:len(body)]
    thin = [r for r in (shown or []) if (r["Total_Trades"] or 0) < 30]
    if thin:
        lines.append(
            f"{len(thin)} of the {len(body)} rows shown have FEWER THAN 30 "
            f"trades — Gate R's own floor. A high profit factor over a handful "
            f"of trades is noise; --min-trades 30 removes them.")
    if counts["skipped_unsuffixed"]:
        lines.append(
            f"{counts['skipped_unsuffixed']} unsuffixed gate_audit_<SYMBOL>."
            f"json skipped: each duplicates whichever timeframe ran last, "
            f"beside a per-pair file. --include-unsuffixed keeps them.")
    if counts["unreadable"]:
        lines.append(f"{counts['unreadable']} file(s) could not be parsed.")
    return "\n".join(lines)


def write_csv(rows: list[dict], report_dir: Path) -> tuple[Path, Path]:
    """
    The stamped file and the canonical `_latest.csv` beside it.

    A COPY, not a symlink. The reports directory is read by whatever the
    operator points at it - a spreadsheet, a scp, a mounted share - and a
    dangling symlink after the stamped file is tidied away reads as an empty
    report rather than as a missing one.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"strategy_performance_{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: ("" if r.get(c) is None else r[c])
                             for c in COLUMNS})
    latest = report_dir / LATEST
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path, latest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Every gate audit on disk as one table and one CSV. "
                    "Transcribes; it re-scores nothing.")
    ap.add_argument("--artifacts", default=None,
                    help="artifacts root (default $BT_ARTIFACTS, else "
                         "/mnt/backtest/artifacts). Searched recursively.")
    ap.add_argument("--strategy", default=None,
                    help="substring filter on the strategy name")
    ap.add_argument("--symbol", default=None, help="exact symbol filter")
    ap.add_argument("--tf", default=None, help="exact timeframe filter")
    ap.add_argument("--min-trades", type=int, default=0,
                    help="drop configurations with fewer than this many "
                         "trades. DEFAULT 0 - nothing is hidden unless you "
                         "ask - but sorting by profit factor puts the "
                         "thinnest samples on top, and a PF of 6.90 over four "
                         "trades is noise wearing two decimal places. Gate R's "
                         "own floor is 30.")
    ap.add_argument("--pass-only", action="store_true",
                    help="only configurations whose status is PASS")
    ap.add_argument("--sort", default="pf",
                    choices=["pf", "sharpe", "timestamp"])
    ap.add_argument("--limit", type=int, default=50,
                    help="console rows (default 50). The CSV always holds "
                         "every row; 0 shows all.")
    ap.add_argument("--in-sample", action="store_true",
                    help="report the in-sample block instead of the holdout. "
                         "Those are different claims and the footer says "
                         "which ran.")
    ap.add_argument("--include-unsuffixed", action="store_true",
                    help="keep gate_audit_<SYMBOL>.json files that duplicate "
                         "a per-pair audit")
    ap.add_argument("--no-csv", action="store_true", help="console only")
    ap.add_argument("--reports-dir", default=str(REPORT_DIR))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.artifacts) if args.artifacts else artifacts_root()
    if not root.is_dir():
        print(f"artifacts root not found: {root}", file=sys.stderr)
        return 1

    rows, counts = collect(root, args.include_unsuffixed,
                           prefer_oos=not args.in_sample)
    if args.strategy:
        rows = [r for r in rows if args.strategy.lower()
                in r["Strategy"].lower()]
    if args.symbol:
        rows = [r for r in rows if r["Symbol"] == args.symbol.upper()]
    if args.tf:
        rows = [r for r in rows if r["TF"] == args.tf]
    if args.min_trades:
        rows = [r for r in rows
                if (r["Total_Trades"] or 0) >= args.min_trades]
    if args.pass_only:
        rows = [r for r in rows if str(r["Status"]).upper() == "PASS"]

    rows = sort_rows(rows, args.sort)
    print(render(rows, counts, args.limit or None))

    if not args.no_csv and rows:
        path, latest = write_csv(rows, Path(args.reports_dir))
        print(f"\ncsv     {path}")
        print(f"latest  {latest}")

    # Non-zero only when nothing could be READ. An empty filter result is a
    # legitimate answer, not a failure of the tool.
    return 0 if counts["files"] else 1


if __name__ == "__main__":
    sys.exit(main())
