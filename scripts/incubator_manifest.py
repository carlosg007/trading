#!/usr/bin/env python3
"""
scripts/incubator_manifest.py - one row per promoted package, as CSV.

Location:  ~/src/trading/scripts/incubator_manifest.py

    .venv/bin/python3 scripts/incubator_manifest.py
    .venv/bin/python3 scripts/incubator_manifest.py --out reports/x.csv

WHAT IT READS, AND WHY EACH FIELD COMES FROM WHERE IT DOES
---------------------------------------------------------
Three sources, and none of them is re-derived from another:

  meta.json          the package's own record - symbol, timeframe, version,
                     certified regime, risk bracket, day-of-week gate.
  portfolios.json    the ROUTING, which meta.json does not know: a package on
                     disk is not a routed one, and the difference is whether
                     the live loop will ever hand it a bar. A package with no
                     row here is reported as NOT ROUTED rather than omitted.
  gate_audit_*.json  the holdout numbers and `certified_on`, cited by
                     `certification.audit_file` on the package. Read from the
                     audit rather than from meta so the CSV cannot disagree
                     with the file the certification was sealed against.

`certified_on` IS NOT DEFAULTED TO "primary". The field arrived 2026-09-08
with Gate R's starvation fallback, and audits written before it carry no
value at all. Writing "primary" for those would state a fact the artifact
does not - the honest cell is `not recorded (pre-2026-09-08 audit)`, because
"we know it was the primary" and "the file predates the question" are
different things and only one of them can be checked.

WEEKDAYS ARE SPELLED OUT, both blocked and approved. `blocked_weekdays` is a
list of ints with Monday=0, and a CSV column holding `[3]` is a column nobody
reads correctly under time pressure. Futures trade Monday to Friday, so the
approved set is those five minus whatever Stage 4.5 blocked; a package whose
gate never ran is reported as UNKNOWN rather than as all five, since "no
weekday was blocked" and "no weekday was measured" are not the same.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
INCUBATOR = REPO / "strategies" / "approved_incubator"
PORTFOLIOS = REPO / "config" / "portfolios.json"
DEFAULT_OUT = REPO / "reports" / "active_incubator_manifest.csv"

#: Monday=0, matching `datetime.weekday()` and `blocked_weekdays` on the gate.
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
#: The sessions a futures strategy can trade at all.
TRADING_WEEK = ("Mon", "Tue", "Wed", "Thu", "Fri")

COLUMNS = ("strategy_id", "strategy_family", "symbol", "timeframe", "version",
           "portfolio", "execution_account", "certified_regime",
           "certified_on", "holdout_trades", "holdout_pf",
           "approved_trading_days", "blocked_trading_days",
           "sl_atr_mult", "tp_atr_mult")

NOT_ROUTED = "NOT ROUTED"
NOT_RECORDED = "not recorded (pre-2026-09-08 audit)"


def _read_json(path: Path | str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def routing_index(cfg: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """`{strategy_id: {portfolio, execution_account, regime_filter}}`."""
    out: dict[str, dict[str, Any]] = {}
    for pid, p in ((cfg or {}).get("portfolios") or {}).items():
        allocations = p.get("strategy_allocations") or {}
        for sid in (p.get("active_strategies") or []):
            out[str(sid)] = {
                "portfolio": pid,
                "execution_account": p.get("target_account"),
                "regime_filter": (allocations.get(sid) or {}).get(
                    "regime_filter"),
            }
    return out


def family_of(strategy_id: str, meta: dict[str, Any] | None) -> str:
    """
    The module name, with the promotion's `_<SYM>_<TF>_V<A|B>` suffix removed.

    Taken from `meta["strategy"]` when it is there, because that is the module
    the package was built from; the id is only parsed as a fallback. Trailing
    date stamps are KEPT - `t3_braid_scalp_20260823` and a future
    `t3_braid_scalp_20261102` are different families and collapsing them would
    hide a re-run of the same idea beside the original.
    """
    named = str((meta or {}).get("strategy") or "").strip()
    if named:
        return named
    parts = strategy_id.split("_")
    while parts and (parts[-1].upper() in ("VA", "VB")
                     or parts[-1][:1].isdigit()
                     or parts[-1].isupper()):
        parts.pop()
    return "_".join(parts) or strategy_id


def weekday_names(indices: Any) -> list[str]:
    out = []
    for i in (indices or []):
        try:
            out.append(WEEKDAYS[int(i)])
        except (ValueError, TypeError, IndexError):
            out.append(f"?{i}")
    return out


def gate_regime_of(meta: dict[str, Any]) -> dict[str, Any]:
    """Gate R's block from the audit this package cites, or {}."""
    cert = meta.get("certification") or {}
    audit = _read_json(cert.get("audit_file") or "")
    if not audit:
        return {}
    version = str(meta.get("version") or "A")
    block = ((audit.get("versions") or {}).get(version) or {})
    gates = ((block.get("gate_audit") or block).get("gates") or {})
    return gates.get("gate_regime") or {}


def row_for(package: Path, routes: dict[str, dict[str, Any]]
            ) -> dict[str, Any]:
    sid = package.name
    meta = _read_json(package / "meta.json") or {}
    cert = meta.get("certification") or {}
    risk = meta.get("risk") or {}
    dow = meta.get("day_of_week_gate") or {}
    gate = gate_regime_of(meta)
    route = routes.get(sid) or {}

    blocked = weekday_names(dow.get("blocked_weekdays"))
    if not dow or dow.get("status") in (None, "", "NOT EVALUATED"):
        approved_cell, blocked_cell = "UNKNOWN (gate not evaluated)", "UNKNOWN"
    else:
        approved_cell = "/".join(d for d in TRADING_WEEK if d not in blocked)
        blocked_cell = "/".join(blocked) if blocked else "None"

    measured = gate.get("measured") or {}
    return {
        "strategy_id": sid,
        "strategy_family": family_of(sid, meta),
        "symbol": meta.get("symbol") or cert.get("audit_symbol"),
        "timeframe": meta.get("timeframe") or cert.get("audit_timeframe"),
        "version": meta.get("version"),
        "portfolio": route.get("portfolio") or NOT_ROUTED,
        "execution_account": route.get("execution_account") or NOT_ROUTED,
        # The package's own certified quadrant. `regime_filter` on the routing
        # row is the same value and is NOT read here: if the two ever differ
        # the package is the certification and the routing table is the bug,
        # and a CSV that quietly preferred one would hide that.
        "certified_regime": cert.get("target_quadrant")
        or gate.get("quadrant"),
        "certified_on": gate.get("certified_on") or NOT_RECORDED,
        "holdout_trades": measured.get("trade_count"),
        "holdout_pf": measured.get("profit_factor"),
        "approved_trading_days": approved_cell,
        "blocked_trading_days": blocked_cell,
        "sl_atr_mult": risk.get("sl_atr_mult"),
        # `None` is a REAL setting - no take profit, the position runs to the
        # stop or the exit rule - so it is written as the word rather than an
        # empty cell that reads as "not configured".
        "tp_atr_mult": ("None" if risk.get("tp_atr_mult") is None
                        else risk.get("tp_atr_mult")),
    }


def build(incubator: Path = INCUBATOR,
          portfolios: Path = PORTFOLIOS) -> list[dict[str, Any]]:
    routes = routing_index(_read_json(portfolios))
    packages = sorted(p for p in incubator.iterdir()
                      if p.is_dir() and (p / "meta.json").exists())
    return [row_for(p, routes) for p in packages]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--incubator", default=str(INCUBATOR))
    ap.add_argument("--portfolios", default=str(PORTFOLIOS))
    args = ap.parse_args(argv)

    rows = build(Path(args.incubator), Path(args.portfolios))
    if not rows:
        print(f"no promoted packages under {args.incubator}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    unrouted = [r["strategy_id"] for r in rows
                if r["portfolio"] == NOT_ROUTED]
    print(f"{len(rows)} package(s) -> {out}")
    if unrouted:
        print(f"  ! {len(unrouted)} NOT ROUTED (on disk, absent from "
              f"portfolios.json, cannot trade):", file=sys.stderr)
        for sid in unrouted:
            print(f"      {sid}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
