#!/usr/bin/env python3
"""
What is allocated, where, and on what evidence.

    python3 tools/portfolio_inventory.py
    python3 tools/portfolio_inventory.py --portfolio Incubator-Odd --version B
    python3 tools/portfolio_inventory.py --out /tmp/inv.csv

WHY THIS EXISTS
===============
`config/portfolios.json` says what an account may trade. The package under
`strategies/approved_incubator/` says what was certified. The gate audit on the
artifact mount says on what numbers. Three files, and nothing joined them - so
on 2026-08-29 the routing table carried 18 allocations naming packages that had
been deleted, and it took a hand-written diff to notice. This is that diff, as a
tool.

WHAT IT DOES NOT DO
===================
**It recomputes nothing.** Every number is transcribed from the artifact that
recorded it - profit factors from the gate audit's own metrics blocks, the
designated quadrant from the package's own sealed certification block. A tool
that re-derived
a profit factor here would be free to disagree with the certification it claims
to summarise, and the disagreement would look like a finding.

A row whose evidence cannot be read is REPORTED, never dropped: `disk_status`
carries MISSING_ON_DISK for an allocation with no package, and a metric that no
artifact recorded is left EMPTY rather than filled with a zero. An empty cell
and a 0.00 are different claims - one says nobody measured, the other says
somebody measured nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PORTFOLIOS = REPO_ROOT / "config" / "portfolios.json"
INCUBATOR = REPO_ROOT / "strategies" / "approved_incubator"
DEFAULT_OUT = REPO_ROOT / "reports" / "portfolio_inventory.csv"

#: Fixed and ordered, so two runs can be diffed. A sheet whose columns move
#: when a field is added is one nobody can compare across days.
COLUMNS = (
    "portfolio_name", "strategy_id", "strategy_family", "symbol", "timeframe",
    "version", "ml_threshold", "target_quadrant", "target_regime",
    "kill_switch_regimes",
    "in_sample_pf", "in_sample_trades", "in_sample_win_rate",
    "gate_r_pf", "gate_r_trades", "oos_holdout_pf", "oos_holdout_trades",
    "gate_r_status", "report_html",
    "disk_status", "promoted_at",
)

EXISTS, MISSING = "EXISTS", "MISSING_ON_DISK"


def _read_json(path: Path | str | None) -> dict[str, Any] | None:
    """Parsed JSON, or None. An unreadable artifact is a MISSING FIELD, not a
    crash: the inventory's whole job is to report what is and is not there."""
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _family(strategy_id: str, meta: dict | None) -> str:
    """The base strategy an id belongs to.

    `meta["strategy"]` first, because the package records it directly. The id
    is parsed only as a fallback, and through `pipeline.base_strategy` rather
    than a split here - strategy names carry underscores and a date suffix, so
    counting segments turns `ma_anchoring_spread_20260820` into a family of
    `ma_anchoring`.
    """
    if isinstance(meta, dict) and meta.get("strategy"):
        return str(meta["strategy"])
    try:
        from backtest.pipeline import base_strategy
        return base_strategy(strategy_id)
    except Exception:                                            # noqa: BLE001
        return strategy_id


def _report_html(meta: dict, version: str | None) -> str:
    """
    The tearsheet Stage 4 wrote for this package, or MISSING.

    Resolved by CONVENTION, because nothing records it: meta.json carries no
    report key, so the path is rebuilt from the family, the symbol and the
    version - `<family>/verify_*/report_<SYMBOL>_version_<a|b>.html`. The
    NEWEST verify_ directory wins, since a campaign re-run leaves the older
    ones in place and the stale tearsheet describes a run whose parameters no
    longer match the promoted module.

    MISSING rather than a guessed path. A column offering a file that is not
    there wastes more time than an empty one.
    """
    strategy = str(meta.get("strategy") or "").strip()
    symbol = str(meta.get("symbol") or "").strip().upper()
    ver = str(version or "").strip().lower()
    if not (strategy and symbol and ver in ("a", "b")):
        return "MISSING"
    try:
        from backtest.pipeline import artifacts_root
        home = artifacts_root() / "pipeline" / strategy
    except Exception:                                            # noqa: BLE001
        return "MISSING"
    runs = sorted((d for d in home.glob("verify_*") if d.is_dir()),
                  key=lambda d: d.name, reverse=True)
    for run in runs:
        candidate = run / f"report_{symbol}_version_{ver}.html"
        if candidate.is_file():
            return str(candidate)
    return "MISSING"


def _certified_regime(meta: dict) -> tuple[str, str]:
    """
    The quadrant this package was CERTIFIED on, from its sealed certification
    block. `(target_quadrant, target_regime)`, or `("", "")` when the block
    does not record them.

    THE SEALED ARTIFACT IS THE ONLY AUTHORITY, AND THERE IS DELIBERATELY NO
    FALLBACK. Two other sources are readable from here and both are wrong:

      * `regime_profile_holdout.optimal_quadrant`, which this tool used until
        2026-08-31, is the best-of-four quadrant WITHIN THE HOLDOUT. Gate R
        judges the quadrant DESIGNATED IN SAMPLE, and re-picking the best of
        four on the holdout is exactly the selection Gate R exists to prevent -
        almost anything clears a profit factor of 1.00 given four attempts. The
        two coincide only when the holdout's best happens to be the designated
        one, and across the incubator they DISAGREE on 37 of 118 packages:
        `double_rsi_momentum_pullback_20260830_6E_30m` is certified on Q4 and
        was being reported as Q3, and the Q1/Q2 pair swaps twenty more times.
      * `TARGET_QUADRANTS` in the strategy's `.py` source, which is the
        premise's NOMINATION - what the author hoped for before Stage 1
        measured anything. `double_rsi_momentum_pullback_20260830` declares
        `("Q3", "Q1")` in source and holds Q4 certifications on disk. The
        module's own comment says so: "Stage 1 designates the real one by
        measuring all four quadrants, and nothing in this module reads these."

    A package whose certification block records no quadrant returns EMPTY, and
    the row still prints. An empty cell says nobody recorded it; a quadrant
    guessed from either source above would say something false in a column a
    reader has no way to check.
    """
    cert = meta.get("certification")
    if not isinstance(cert, dict):
        return "", ""
    quadrant = str(cert.get("target_quadrant") or "").strip()
    regime = str(cert.get("target_regime") or "").strip()
    return quadrant, regime


def _version_block(audit: dict | None, version: str | None) -> dict:
    """The gate audit's per-version evidence, keyed "A"/"B"."""
    if not isinstance(audit, dict) or not version:
        return {}
    block = (audit.get("versions") or {}).get(str(version).upper())
    return block if isinstance(block, dict) else {}


def _pct(value: Any) -> Any:
    """A win rate as a percentage. The audits record a FRACTION (0.52), and a
    column headed win_rate showing 0.52 beside one showing 52 is the kind of
    ambiguity that gets read wrong once and trusted after."""
    try:
        return round(float(value) * 100.0, 2)
    except (TypeError, ValueError):
        return ""


def _num(value: Any, digits: int = 4) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


def collect_rows(portfolios: dict, incubator: Path = INCUBATOR) -> list[dict]:
    """One row per ALLOCATION, joined to its package and its gate audit."""
    rows: list[dict] = []
    for name, portfolio in (portfolios.get("portfolios") or {}).items():
        if not isinstance(portfolio, dict):
            continue
        allocations = portfolio.get("strategy_allocations") or {}
        for sid in portfolio.get("active_strategies") or []:
            alloc = allocations.get(sid) or {}
            pkg = incubator / sid
            meta = _read_json(pkg / "meta.json") if pkg.is_dir() else None
            # A directory with no readable meta.json is still ON DISK. The two
            # failures are different - a deleted package and a corrupt one -
            # and collapsing them would send somebody looking in the wrong
            # place.
            disk = EXISTS if pkg.is_dir() else MISSING
            meta = meta or {}

            version = (meta.get("version")
                       or alloc.get("version") or "")
            audit = _read_json((meta.get("certification") or {}).get("audit_file"))
            block = _version_block(audit, version)
            is_m = block.get("metrics_in_sample") or {}
            oos_m = block.get("metrics_holdout") or {}
            profile = block.get("regime_profile_holdout") or {}
            gate_r = (((block.get("gate_audit") or {}).get("gates") or {})
                      .get("gate_regime") or {}).get("measured") or {}

            # FROM THE SEALED CERTIFICATION BLOCK, never from `profile`
            # beside it and never from the module source. See
            # `_certified_regime`.
            target_quadrant, target_regime = _certified_regime(meta)
            rows.append({
                "portfolio_name": name,
                "strategy_id": sid,
                "strategy_family": _family(sid, meta),
                "symbol": meta.get("symbol") or alloc.get("symbol") or "",
                "timeframe": meta.get("timeframe") or alloc.get("timeframe") or "",
                "version": str(version).upper(),
                # Absent for Version A by design - promote.py writes it only
                # for B - so the cell is empty rather than 0.48, which would
                # claim a filter that is not there.
                "ml_threshold": meta.get("ml_threshold", ""),
                "target_quadrant": target_quadrant,
                "target_regime": target_regime,
                "kill_switch_regimes": ", ".join(
                    profile.get("kill_switch_conditions") or []),
                "in_sample_pf": _num(is_m.get("profit_factor")),
                "in_sample_trades": is_m.get("trade_count", ""),
                "in_sample_win_rate": _pct(is_m.get("win_rate")),
                # FROM THE GATE'S OWN BLOCK, not from the holdout profile
                # beside it. Both carry an "optimal" profit factor and they
                # are not the same measurement: `regime_profile_holdout`
                # describes the best quadrant WITHIN THE HOLDOUT, while Gate R
                # judges the quadrant DESIGNATED IN SAMPLE. They coincide only
                # when the holdout's best happens to be the designated one.
                #
                # Reading the profile made this tool report
                # sma_momentum GC 1h Version B as certified on 0 trades - its
                # holdout had no dominant quadrant, so optimal_trade_count was
                # 0 - when Gate R had measured 46 trades at PF 1.20 in Q1 and
                # passed it on that. The gate was right; the summary was
                # reading the wrong field, and a "certified on nothing" row is
                # exactly the kind of false finding a read-only tool must not
                # manufacture.
                "gate_r_pf": _num(gate_r.get("profit_factor")),
                "gate_r_trades": gate_r.get("trade_count", ""),
                "oos_holdout_pf": _num(oos_m.get("profit_factor")),
                "oos_holdout_trades": oos_m.get("trade_count", ""),
                "gate_r_status": (meta.get("gate_audit_status")
                                  or meta.get("oos_status") or ""),
                "report_html": _report_html(meta, version),
                "disk_status": disk,
                "promoted_at": meta.get("promoted_utc") or "",
            })
    return rows


def apply_filters(rows: list[dict], portfolio: str | None = None,
                  symbol: str | None = None,
                  version: str | None = None) -> list[dict]:
    """Case-insensitive filters. A filter that matched nothing returns an
    EMPTY inventory rather than the whole one - silently ignoring an
    unmatched filter would report every strategy under a heading saying
    otherwise."""
    out = rows
    if portfolio:
        out = [r for r in out
               if r["portfolio_name"].casefold() == portfolio.casefold()]
    if symbol:
        out = [r for r in out if r["symbol"].casefold() == symbol.casefold()]
    if version:
        want = str(version).upper().removeprefix("V")
        out = [r for r in out if r["version"] == want]
    return out


def render_table(rows: list[dict]) -> str:
    """The console view: the columns somebody scans, not all seventeen."""
    if not rows:
        return "  (no allocation matched)"
    show = ("portfolio_name", "strategy_id", "version", "ml_threshold",
            "in_sample_pf", "gate_r_pf", "gate_r_trades", "oos_holdout_pf",
            "gate_r_status", "disk_status")
    head = {"portfolio_name": "PORTFOLIO", "strategy_id": "STRATEGY",
            "version": "V", "ml_threshold": "MLthr", "in_sample_pf": "IS PF",
            "gate_r_pf": "GateR PF", "gate_r_trades": "GateR n",
            "oos_holdout_pf": "OOS PF",
            "gate_r_status": "GATE R", "disk_status": "DISK"}
    width = {c: max(len(head[c]), *(len(str(r[c])) for r in rows))
             for c in show}
    lines = ["  " + "  ".join(head[c].ljust(width[c]) for c in show),
             "  " + "  ".join("-" * width[c] for c in show)]
    for r in sorted(rows, key=lambda x: (x["portfolio_name"],
                                         x["strategy_id"])):
        lines.append("  " + "  ".join(str(r[c]).ljust(width[c]) for c in show))
    return "\n".join(lines)


def summarise(rows: list[dict]) -> str:
    missing = [r for r in rows if r["disk_status"] == MISSING]
    by_version: dict[str, int] = {}
    for r in rows:
        by_version[r["version"] or "?"] = by_version.get(r["version"] or "?", 0) + 1
    parts = [f"{len(rows)} allocation(s)",
             ", ".join(f"{n} Version {v}" for v, n in sorted(by_version.items()))]
    if missing:
        # Named, not counted. "3 dangling" sends nobody anywhere.
        parts.append(f"!! {len(missing)} MISSING_ON_DISK: "
                     + ", ".join(r["strategy_id"] for r in missing))
    return "  " + "\n  ".join(p for p in parts if p)


def write_csv(rows: list[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in COLUMNS})
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Catalogue every allocated strategy with the metrics its "
                     "certification recorded. Reads config, packages and gate "
                     "audits; recomputes nothing and writes no strategy."))
    p.add_argument("--out", default=None,
                   help=f"CSV destination (default {DEFAULT_OUT})")
    p.add_argument("--portfolio", default=None, help="only this portfolio")
    p.add_argument("--symbol", default=None, help="only this root symbol")
    p.add_argument("--version", default=None, choices=("A", "B", "VA", "VB"),
                   help="only this version track")
    p.add_argument("--no-csv", action="store_true",
                   help="print the table and write nothing")
    p.add_argument("--config", default=None,
                   help=f"portfolio config to read (default {PORTFOLIOS})")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Path(args.config) if args.config else PORTFOLIOS
    portfolios = _read_json(config)
    if portfolios is None:
        print(f"FAILED  could not read {config}", file=sys.stderr)
        return 2

    rows = apply_filters(collect_rows(portfolios), args.portfolio,
                         args.symbol, args.version)

    W = 100
    print("=" * W)
    print("PORTFOLIO INVENTORY")
    print("=" * W)
    print(render_table(rows))
    print("-" * W)
    print(summarise(rows))

    if not args.no_csv:
        dest = write_csv(rows, Path(args.out) if args.out else DEFAULT_OUT)
        print(f"\n  csv → {dest}")
    # Non-zero when the routing table names a package that is not there. The
    # exit code is the whole point of running this from a check: a dangling
    # allocation cannot be imported by the live dispatcher.
    return 1 if any(r["disk_status"] == MISSING for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
