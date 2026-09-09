#!/usr/bin/env python3
"""
scripts/check_live_certification_health.py - daily incubator certification audit.

Location:  ~/src/trading/scripts/check_live_certification_health.py

    python3 scripts/check_live_certification_health.py
    python3 scripts/check_live_certification_health.py --dry-run
    python3 scripts/check_live_certification_health.py --webhook url

WHAT THIS CHECKS
----------------
Every strategy package under strategies/approved_incubator/ is cross-referenced
against four sources of truth, and the four core live-readiness blockers are
evaluated for each:

  1. ROUTING GAP   - package on disk but missing from config/portfolios.json
                     active allocations
  2. STAGE 4.5 GAP - meta.json lacks a day_of_week_gate evaluation
  3. ANCHOR GAP    - the traded (symbol, timeframe) has no pinned theta_vol
                     anchor in config/theta_vol_anchors.json
  4. EXPECTANCY    - certified quadrant net P&L < $0 or true holdout PF < 1.00

Plus the live service status of trading-regime-daemon.

Exit codes:  0 = 100% clean and reconciled  |  1 = uncertified/unrouted found
             2 = could not run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

INCUBATOR = REPO_ROOT / "strategies" / "approved_incubator"
PORTFOLIOS = REPO_ROOT / "config" / "portfolios.json"
THETA_ANCHORS = REPO_ROOT / "config" / "theta_vol_anchors.json"
MANIFEST = REPO_ROOT / "reports" / "active_incubator_manifest.csv"


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path}: {exc}") from exc


def load_portfolios() -> dict[str, Any]:
    return load_json(PORTFOLIOS)


def load_theta_anchors() -> dict[str, Any]:
    return load_json(THETA_ANCHORS)


def load_manifest() -> dict[str, dict[str, str]]:
    import csv
    rows: dict[str, dict[str, str]] = {}
    try:
        with open(MANIFEST, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sid = (row.get("strategy_id") or "").strip()
                if sid:
                    rows[sid] = row
    except OSError:
        pass
    return rows


def routed_ids(cfg: dict[str, Any]) -> set[str]:
    routed: set[str] = set()
    portfolios = cfg.get("portfolios") or {}
    for pdata in portfolios.values():
        if not isinstance(pdata, dict):
            continue
        for s in (pdata.get("active_strategies") or []):
            if s:
                routed.add(str(s).strip())
    return routed


def packages_on_disk() -> list[Path]:
    if not INCUBATOR.is_dir():
        return []
    return sorted(
        p for p in INCUBATOR.iterdir()
        if p.is_dir() and (p / "meta.json").exists())


def symbol_tf_from_meta(meta: dict[str, Any]) -> tuple[str, str]:
    cert = meta.get("certification") or {}
    audit_file = str(cert.get("audit_file") or "").strip()
    m = re.match(
        r"^gate_audit_(?P<symbol>[^_]+)_(?P<tf>[^_]+)\.json$",
        Path(audit_file).name)
    if m:
        return m.group("symbol").upper(), m.group("tf")
    symbols = meta.get("symbols")
    if isinstance(symbols, list) and len(symbols) == 1:
        tf = str(meta.get("timeframe") or "").strip()
        if tf:
            return str(symbols[0]).upper(), tf
    return "", ""


def day_of_week_gate_ok(meta: dict[str, Any]) -> bool:
    dow = meta.get("day_of_week_gate")
    if not isinstance(dow, dict):
        return False
    status = str(dow.get("status") or "").upper()
    return status in ("EVALUATED", "PASSED", "CLEARED")


def anchor_present(theta: dict[str, Any], symbol: str, tf: str) -> bool:
    sym_entry = theta.get(symbol.upper())
    if not isinstance(sym_entry, dict):
        return False
    tf_entry = sym_entry.get(tf)
    return isinstance(tf_entry, dict)


def evaluate_package(pkg: Path, routed: set[str],
                     manifest: dict[str, dict[str, str]],
                     theta: dict[str, Any]) -> dict[str, Any]:
    sid = pkg.name
    meta = load_json(pkg / "meta.json")
    symbol, tf = symbol_tf_from_meta(meta)

    blockers: list[dict[str, Any]] = []

    # 1. Routing gap
    if sid not in routed:
        blockers.append({
            "blocker": "ROUTING GAP",
            "detail": (f"Package {sid} is on disk but not in any portfolio's "
                       f"active_strategies."),
        })

    # 2. Stage 4.5 gap
    if not day_of_week_gate_ok(meta):
        blockers.append({
            "blocker": "STAGE 4.5 GAP",
            "detail": (f"meta.json lacks an evaluated day_of_week_gate "
                       f"(status={meta.get('day_of_week_gate', {}).get('status', 'MISSING')})."),
        })

    # 3. Anchor gap
    if symbol and tf:
        if not anchor_present(theta, symbol, tf):
            blockers.append({
                "blocker": "ANCHOR GAP",
                "detail": (f"({symbol}, {tf}) has no pinned theta_vol anchor "
                           f"in config/theta_vol_anchors.json."),
            })
    else:
        blockers.append({
            "blocker": "ANCHOR GAP",
            "detail": (f"Could not resolve (symbol, timeframe) from {sid}."),
        })

    # 4. Negative expectancy / rounding artifact
    meta_metrics = meta.get("metrics") or {}
    total_pnl = meta_metrics.get("total_pnl")
    try:
        pnl_val = float(total_pnl) if total_pnl is not None else None
    except (TypeError, ValueError):
        pnl_val = None
    if pnl_val is not None and pnl_val < 0:
        blockers.append({
            "blocker": "NEGATIVE EXPECTANCY",
            "detail": (f"Certified net P&L = ${pnl_val:,.2f} (< $0)."),
        })

    # Holdout PF from the certification audit
    cert = meta.get("certification") or {}
    audit_file = str(cert.get("audit_file") or "").strip()
    if audit_file and Path(audit_file).exists():
        try:
            audit = load_json(Path(audit_file))
            versions = audit.get("versions") or {}
            for ver in ("A", "B"):
                block = versions.get(ver)
                if isinstance(block, dict):
                    gate_r = ((block.get("gate_audit") or {}).get("gates", {})
                              .get("gate_regime", {}))
                    measured = gate_r.get("measured") or {}
                    pf = measured.get("profit_factor")
                    try:
                        holdout_pf = float(pf) if pf is not None else None
                    except (TypeError, ValueError):
                        holdout_pf = None
                    if holdout_pf is not None and holdout_pf < 1.0:
                        blockers.append({
                            "blocker": "HOLDOUT PF < 1.00",
                            "detail": (f"True holdout PF = {holdout_pf:.2f} "
                                       f"(< 1.00)."),
                        })
                    break
        except Exception:
            pass

    return {
        "strategy_id": sid,
        "symbol": symbol,
        "timeframe": tf,
        "in_routing": sid in routed,
        "in_manifest": sid in manifest,
        "dow_gate_evaluated": day_of_week_gate_ok(meta),
        "blockers": blockers,
    }


def check_daemon_status() -> dict[str, Any]:
    result: dict[str, Any] = {"service": {}, "timer": {}, "error": None}
    svc_unit = os.environ.get("REGIME_DAEMON_SERVICE",
                              "trading-regime-daemon.service")
    for unit, key in ((svc_unit, "service"),
                      (svc_unit.replace(".service", ".timer"), "timer")):
        try:
            proc = subprocess.run(
                ["systemctl", "--user", "show", unit,
                 "--property=ActiveState,SubState,LoadState,NRestarts,ExecMainStatus,Result"],
                capture_output=True, text=True, timeout=15, check=False)
            props: dict[str, str] = {}
            for line in proc.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    props[k] = v
            result[key] = {
                "load_state": props.get("LoadState", "?"),
                "active_state": props.get("ActiveState", "?"),
                "sub_state": props.get("SubState", "?"),
                "exit_status": props.get("ExecMainStatus", "?"),
                "result": props.get("Result", "?"),
                "restarts": props.get("NRestarts", "?"),
            }
        except Exception as exc:
            result[key] = {"error": str(exc)}
            result["error"] = str(exc)
    return result


def build_report(packages: list[dict[str, Any]],
                 daemon: dict[str, Any]) -> dict[str, Any]:
    unrouted = [p for p in packages if not p["in_routing"]]
    with_blockers = [p for p in packages if p["blockers"]]
    clean = [p for p in packages
             if p["in_routing"] and not p["blockers"]]

    categories: dict[str, list[str]] = {}
    for p in unrouted + with_blockers:
        if not p["blockers"]:
            continue
        primary = p["blockers"][0]["blocker"]
        categories.setdefault(primary, []).append(p["strategy_id"])

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_packages_disk": len(packages),
        "routed_count": sum(1 for p in packages if p["in_routing"]),
        "manifest_count": sum(1 for p in packages if p["in_manifest"]),
        "clean_count": len(clean),
        "unrouted_count": len(unrouted),
        "with_blockers_count": len(with_blockers),
        "categories": categories,
        "unrouted_ids": [p["strategy_id"] for p in unrouted],
        "blocker_details": [
            {"strategy_id": p["strategy_id"], "blockers": p["blockers"]}
            for p in with_blockers
        ],
        "daemon": daemon,
        "exit_code": 0 if (not unrouted and not with_blockers) else 1,
    }


def discord_embed(report: dict[str, Any]) -> dict[str, Any]:
    colour = 0xE74C3C if report["exit_code"] == 1 else 0x2ECC71

    lines = [
        f"**Packages on disk:** {report['total_packages_disk']}",
        f"**Routed in portfolios.json:** {report['routed_count']}",
        f"**In manifest:** {report['manifest_count']}",
        f"**Clean / reconciled:** {report['clean_count']}",
        f"**Unrouted:** {report['unrouted_count']}",
        f"**With blockers:** {report['with_blockers_count']}",
        "",
    ]

    if report["categories"]:
        lines.append("**Uncertified / unrouted by reason:**")
        for cat, ids in sorted(report["categories"].items()):
            lines.append(f"- {cat}: {len(ids)} packages")
            for sid in ids[:8]:
                lines.append(f"   - `{sid}`")
            if len(ids) > 8:
                lines.append(f"   ... +{len(ids)-8} more")
        lines.append("")

    if report["blocker_details"]:
        lines.append("**Blocker details:**")
        for bd in report["blocker_details"][:15]:
            for b in bd["blockers"]:
                lines.append(f"- `{bd['strategy_id']}`: {b['blocker']} - {b['detail']}")
        if len(report["blocker_details"]) > 15:
            lines.append(f"... +{len(report['blocker_details'])-15} more packages")
        lines.append("")

    d = report["daemon"]
    svc = d.get("service", {})
    tmr = d.get("timer", {})
    lines.append("**Daemon status:**")
    if "error" in svc:
        lines.append(f"- service: ERROR - {svc['error']}")
    else:
        lines.append(
            f"- service: {svc.get('active_state','?')}/{svc.get('sub_state','?')}  "
            f"(exit={svc.get('exit_status','?')}, restarts={svc.get('restarts','?')})")
    if "error" in tmr:
        lines.append(f"- timer: ERROR - {tmr['error']}")
    else:
        lines.append(
            f"- timer: {tmr.get('active_state','?')}/{tmr.get('sub_state','?')}")
    lines.append("")

    description = "\n".join(lines)

    return {
        "title": f"\U0001F50D Incubator Health & Certification Audit",
        "color": colour,
        "fields": [
            {"name": "Summary", "value": description, "inline": False},
        ],
        "footer": {"text": "scripts/check_live_certification_health.py"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Incubator strategy live-certification health audit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the report as JSON, post nothing")
    parser.add_argument("--webhook", default=None,
                        help="Discord webhook URL (default: from .env)")
    parser.add_argument("--no-post", action="store_true",
                        help="run the audit but never post to Discord")
    args = parser.parse_args(argv)

    try:
        cfg = load_portfolios()
        theta = load_theta_anchors()
        manifest = load_manifest()
        routed = routed_ids(cfg)
    except RuntimeError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 2

    packages: list[dict[str, Any]] = []
    disk_pkgs = packages_on_disk()
    for pkg in disk_pkgs:
        try:
            packages.append(evaluate_package(pkg, routed, manifest, theta))
        except Exception as exc:
            packages.append({
                "strategy_id": pkg.name,
                "symbol": "",
                "timeframe": "",
                "in_routing": pkg.name in routed,
                "in_manifest": pkg.name in manifest,
                "dow_gate_evaluated": False,
                "blockers": [{"blocker": "EVALUATION ERROR",
                               "detail": str(exc)}],
            })

    daemon = check_daemon_status()
    report = build_report(packages, daemon)

    print(f"Incubator Health Audit - {report['timestamp']}")
    print(f"  Packages on disk : {report['total_packages_disk']}")
    print(f"  Routed           : {report['routed_count']}")
    print(f"  Clean / reconciled: {report['clean_count']}")
    print(f"  Unrouted         : {report['unrouted_count']}")
    print(f"  With blockers    : {report['with_blockers_count']}")
    print(f"  Exit code        : {report['exit_code']}")
    if report["categories"]:
        print()
        print("  Uncertified by reason:")
        for cat, ids in sorted(report["categories"].items()):
            print(f"    {cat}: {len(ids)}")
            for sid in ids[:5]:
                print(f"      - {sid}")
            if len(ids) > 5:
                print(f"      ... +{len(ids)-5} more")

    if not args.no_post and not args.dry_run:
        webhook = args.webhook
        if not webhook:
            try:
                import scripts._env_bootstrap  # noqa: F401
            except Exception:
                pass
            webhook = os.environ.get("BT_DISCORD_WEBHOOK") or \
                      os.environ.get("DISCORD_WEBHOOK_URL") or \
                      os.environ.get("DISCORD_WEBHOOK") or None
        if webhook:
            from backtest.discord_reporter import post_embed, build_payload
            embed = discord_embed(report)
            payload = build_payload(embed)
            result = post_embed(webhook, payload)
            if result.get("ok"):
                print(f"  Discord: posted [HTTP {result['http_status']}]")
            else:
                print(f"  Discord: FAILED - {result.get('error') or 'unknown'}")
        else:
            print("  Discord: no webhook configured - not posting")

    if args.dry_run:
        print()
        print("--- DRY RUN JSON ---")
        print(json.dumps(report, indent=2, default=str))

    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
