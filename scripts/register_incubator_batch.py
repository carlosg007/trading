#!/usr/bin/env python3
"""
scripts/register_incubator_batch.py - route unrouted incubator packages.

    .venv/bin/python3 scripts/register_incubator_batch.py            # dry run
    .venv/bin/python3 scripts/register_incubator_batch.py --write
    .venv/bin/python3 scripts/register_incubator_batch.py --allow-missing-dow

WRITES NOTHING WITHOUT --write. A package on disk is not a routed one, and the
difference is whether the live loop hands it a bar; this turns the first into
the second, one contract at a time.

FOUR GATES, AND A PACKAGE MUST CLEAR ALL FOUR
---------------------------------------------
Each exists because a specific failure was observed in this repository, and
each refusal names the package rather than skipping it quietly.

  1. THE SYMBOL IS IN A BASKET. `portfolio/config_loader.py` has no routing
     fallback - an unassigned strategy raises rather than being placed by a
     rule nobody wrote down - so the basket is the routing decision and this
     reads it rather than inventing one. A symbol in no basket is REFUSED, not
     added: adding an asset also means adding `asset_metadata`, which the
     loader reconciles against `backtest/specs.py` on every load, and several
     symbols here (BTC, SI, LE, the FX crosses, the grains) are UNVERIFIED in
     that file. A guessed multiplier silently scales every P&L figure for that
     contract and nothing downstream looks wrong.

  2. A theta_vol ANCHOR RESOLVES for the package's (symbol, TIMEFRAME). This
     is the ETH 30m bug generalised: a registered pair the daemon cannot
     classify does not crash anything, it just reads downstream as "not in the
     permitted quadrant" - indistinguishable from a quiet market - and the
     strategy is logged correctly forever while never taking a trade. AN
     ANCHOR IS PER TIMEFRAME, so the check is per (symbol, tf) and never per
     symbol.

  3. THE CERTIFIED QUADRANT'S NET P&L IS POSITIVE. Gate R compares
     `PF >= 1.00` against a factor `backtest/profiler.py` already rounded to
     two places, so a true factor anywhere in [0.995, 1.000) passes it. The
     net P&L beside it is the tell, and it is not rounded into ambiguity:
     negative net means the unrounded factor is below the bar whatever the
     stored 1.00 says. Three packages fail here, and two of them were pruned
     out of the routing table on 2026-09-09 for exactly this.

  4. STAGE 4.5 RAN (`day_of_week_gate.status == "EVALUATED"`). The three
     states this repository keeps apart are no key at all, `blocked_weekdays:
     []`, and a blocked day. Registering a package whose gate never ran writes
     an empty mask into the live table, which is the SECOND state - "the stage
     ran and every session cleared" - and the two are then indistinguishable
     to every later reader. `--allow-missing-dow` routes them anyway and
     stamps `day_of_week_basis: "NOT EVALUATED"` on the allocation so the
     distinction survives in the file. It is not the default.

WHAT IS WRITTEN. One contract, the package's own certified `target_quadrant`
as `regime_filter`, and the weekday mask carried from `meta.json` rather than
recomputed. Both keys move together - `active_strategies` and the
`strategy_allocations` record beside it - because clearing or setting one
without the other is a registry that disagrees with itself (6938ce8).

The config is written temp-then-`os.replace` and VALIDATED through
`portfolio.config_loader` while it is still the temp file, so a mutation that
would make the routing table unloadable never reaches the real path.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

INCUBATOR = REPO / "strategies" / "approved_incubator"
PORTFOLIOS = REPO / "config" / "portfolios.json"

#: Micro -> the full-size contract the history and the anchor live under. The
#: basket names the micro; the certification names the parent.
PARENT = {"MNQ": "NQ", "MES": "ES", "MGC": "GC", "MYM": "YM", "M2K": "RTY"}
INCUBATOR_RUNGS = ("Incubator-Odd", "Incubator-Even", "Incubator-FullSize")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(), object_pairs_hook=collections.OrderedDict)


def basket_index(cfg: dict) -> dict[str, str]:
    """`{traded_symbol: incubator_portfolio_id}`, resolving micros to parents."""
    out: dict[str, str] = {}
    for pid in INCUBATOR_RUNGS:
        port = cfg["portfolios"].get(pid) or {}
        for asset in (port.get("basket") or {}).get("assets") or []:
            out.setdefault(PARENT.get(asset, asset), pid)
    return out


def gate_regime_of(meta: dict) -> dict:
    cert = meta.get("certification") or {}
    try:
        audit = json.loads(Path(cert["audit_file"]).read_text())
    except (OSError, ValueError, KeyError):
        return {}
    block = (audit.get("versions") or {}).get(str(meta.get("version"))) or {}
    gates = ((block.get("gate_audit") or block).get("gates") or {})
    return gates.get("gate_regime") or {}


def evaluate(package: Path, cfg: dict, baskets: dict[str, str],
             theta_for, allow_missing_dow: bool) -> dict[str, Any]:
    """One package against the four gates. `refusals` empty == routable."""
    meta = json.loads((package / "meta.json").read_text())
    symbol = meta.get("symbol")
    tf = meta.get("timeframe")
    dow = meta.get("day_of_week_gate") or {}
    gate = gate_regime_of(meta)
    measured = gate.get("measured") or {}
    refusals: list[str] = []

    portfolio = baskets.get(symbol)
    if portfolio is None:
        refusals.append(f"symbol {symbol} is in no incubator basket")
    try:
        theta_for(symbol, tf)
    except Exception:
        refusals.append(f"no theta_vol anchor for {symbol}/{tf}")

    net = measured.get("net_pnl")
    if net is None:
        refusals.append("Gate R recorded no net P&L for the certified quadrant")
    elif net <= 0:
        refusals.append(
            f"certified quadrant net P&L is {net:,.2f} - the stored profit "
            f"factor {measured.get('profit_factor')} is rounded to 2dp and the "
            f"unrounded one is below 1.00")

    status = dow.get("status")
    if status != "EVALUATED" and not allow_missing_dow:
        refusals.append(f"Stage 4.5 day-of-week gate is {status!r}, not EVALUATED")

    return {
        "strategy_id": package.name, "symbol": symbol, "timeframe": tf,
        "version": meta.get("version"), "portfolio": portfolio,
        "quadrant": (meta.get("certification") or {}).get("target_quadrant"),
        "blocked_weekdays": list(dow.get("blocked_weekdays") or []),
        "dow_status": status, "refusals": refusals,
        "path": f"strategies/approved_incubator/{package.name}/strat.py",
    }


def allocation_for(row: dict) -> collections.OrderedDict:
    alloc = collections.OrderedDict([
        ("strat", row["strategy_id"]),
        ("symbol", row["symbol"]),
        ("timeframe", row["timeframe"]),
        ("version", row["version"]),
        ("allocation", 1),
        ("regime_filter", row["quadrant"]),
        ("status", "incubating"),
        ("path", row["path"]),
        ("registered_by", "scripts/register_incubator_batch.py"),
        ("blocked_weekdays", row["blocked_weekdays"]),
    ])
    # The three day-of-week states stay distinguishable IN THE FILE. An empty
    # mask written for a gate that never ran would read as "the stage ran and
    # every session cleared" to every later reader.
    if row["dow_status"] != "EVALUATED":
        alloc["day_of_week_basis"] = "NOT EVALUATED"
    return alloc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="apply the registrations (default: report only)")
    ap.add_argument("--allow-missing-dow", action="store_true",
                    help="route packages whose Stage 4.5 gate never ran")
    ap.add_argument("--portfolios", default=str(PORTFOLIOS))
    ap.add_argument("--incubator", default=str(INCUBATOR))
    args = ap.parse_args(argv)

    path = Path(args.portfolios)
    cfg = _load(path)
    baskets = basket_index(cfg)
    routed = {s for p in cfg["portfolios"].values()
              for s in (p.get("active_strategies") or [])}

    from realtime.regime_daemon import MasterRegimeDaemon
    daemon = MasterRegimeDaemon()

    rows = [evaluate(p, cfg, baskets, daemon.theta_for, args.allow_missing_dow)
            for p in sorted(Path(args.incubator).iterdir())
            if p.is_dir() and (p / "meta.json").exists() and p.name not in routed]

    eligible = [r for r in rows if not r["refusals"]]
    refused = [r for r in rows if r["refusals"]]

    print(f"unrouted packages: {len(rows)}   routable: {len(eligible)}   "
          f"refused: {len(refused)}")
    for r in eligible:
        mask = (", ".join(str(d) for d in r["blocked_weekdays"])
                or "no weekday blocked")
        print(f"  ROUTE   {r['strategy_id']:56s} -> {r['portfolio']:19s} "
              f"{r['quadrant']} 1 contract [{mask}]")
    if refused:
        print(f"\nrefused ({len(refused)}):")
        for r in refused:
            print(f"  SKIP    {r['strategy_id']}")
            for why in r["refusals"]:
                print(f"            {why}")

    if not args.write:
        print("\ndry run - nothing written. Re-run with --write to apply.")
        return 0
    if not eligible:
        print("\nnothing routable; config untouched.")
        return 1

    for r in eligible:
        port = cfg["portfolios"][r["portfolio"]]
        port.setdefault("active_strategies", []).append(r["strategy_id"])
        port.setdefault("strategy_allocations",
                        collections.OrderedDict())[r["strategy_id"]] = \
            allocation_for(r)

    # Temp-then-replace, validated while it is still the temp file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    try:
        from portfolio.config_loader import load_portfolio_config
        load_portfolio_config(str(tmp))
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"\nREFUSED: the resulting config does not load ({exc}). "
              f"{path} is untouched.", file=sys.stderr)
        return 2
    os.replace(tmp, path)
    print(f"\nregistered {len(eligible)} package(s) into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
