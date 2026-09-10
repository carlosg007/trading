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

#: Where the declaration of the routing table lives, and the marker this
#: script rewrites inside it.
PORTFOLIO_TEST = REPO / "tests" / "test_portfolio_config.py"
EXPECTED_CONST = "EXPECTED_ASSIGNMENTS = {"

#: THE ONE PART OF THE GUARD THAT IS NOT REGENERATED.
#:
#: `EXPECTED_ASSIGNMENTS` used to be maintained by hand, and its whole value
#: was that a strategy could not enter `active_strategies` without a human
#: naming it in the same commit - it caught nine unattended assignments on
#: 2026-08-27. That check is now regenerated from the live config on every
#: --write, which is a deliberate trade made on 2026-09-10 to close the
#: unattended lifecycle: a declaration rewritten from the thing it describes
#: cannot contradict it, so it no longer detects an assignment nobody
#: intended.
#:
#: These four are carved out of that trade. Each cleared Gate R on a profit
#: factor `backtest/profiler.py` had already rounded up to 1.00 while its
#: certified-quadrant net P&L is negative, and the first two REACHED THE LIVE
#: ROUTING TABLE and had to be pruned by hand. Gate 3 below refuses them, so
#: this is the second lock on a door that already has one - kept because the
#: cost is a frozenset and the failure it prevents was paid for once already.
MUST_STAY_ABSENT = frozenset({
    "t3_braid_scalp_20260823_RB_30m_VA",
    "t3_braid_scalp_20260823_YM_30m_VA",
    "ma_anchoring_spread_20260820_ZS_1h_VA",
    "ema_deviation_scalp_20260909_NQ_30m_VB",
})


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


def render_expected_assignments(cfg: dict, preamble: str) -> str:
    """
    The `EXPECTED_ASSIGNMENTS` literal, rendered from the live config.

    `preamble` is the comment block already inside the braces, carried through
    verbatim apart from its regeneration stamp. Those comments record WHY four
    packages are absent and why NG, RB and CL cannot be routed at all, and
    none of that is derivable from the config - a regenerator that dropped
    them would leave the next reader with a hundred names and no reasons.

    Entries are emitted in the order they appear in `active_strategies`, which
    is the order the registrar appended them in, because the assertion this
    feeds compares element-wise.
    """
    lines = [EXPECTED_CONST]
    lines.extend(preamble)

    ids: list[tuple[str, list[tuple[str, str, str]]]] = []
    width = 0
    for pid, port in cfg["portfolios"].items():
        entries = []
        for sid in (port.get("active_strategies") or []):
            alloc = ((port.get("strategy_allocations") or {}).get(sid) or {})
            entries.append((sid, str(alloc.get("timeframe") or "?"),
                            str(alloc.get("symbol") or "?")))
            width = max(width, len(sid))
        ids.append((pid, entries))

    keyw = max(len(pid) for pid, _ in ids) + 3
    for pid, entries in ids:
        key = f'"{pid}":'.ljust(keyw)
        if not entries:
            lines.append(f"    {key}[],")
            continue
        lines.append(f"    {key}[")
        for sid, tf, sym in entries:
            quoted = f'"{sid}",'.ljust(width + 4)
            lines.append(f"        {quoted}  # {tf:>3}, {sym}")
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def _split_literal(text: str) -> tuple[int, int, list[str]]:
    """`(start, end, preamble)` for the EXPECTED_ASSIGNMENTS literal."""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith(EXPECTED_CONST))
    except StopIteration:
        raise ValueError(f"{PORTFOLIO_TEST.name} has no {EXPECTED_CONST}")
    try:
        end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    except StopIteration:
        raise ValueError(f"{EXPECTED_CONST} is not closed by a bare '}}'")
    # The comment block between the brace and the first key belongs to the
    # literal and is carried through; the first quoted key ends it.
    body = lines[start + 1:end]
    cut = next((i for i, l in enumerate(body) if l.lstrip().startswith('"')),
               len(body))
    return start, end, body[:cut]


def _restamp(preamble: list[str], routed: list[str], total: int) -> list[str]:
    """Replace the leading regeneration stanza; leave every other comment."""
    stamp = [
        f"    # REGENERATED {_today()} by "
        f"scripts/register_incubator_batch.py --write, which now rewrites",
        f"    # this constant in the same run that registers into "
        f"config/portfolios.json.",
        f"    # This run routed {len(routed)} package(s); the book stands at "
        f"{total}.",
        "    #",
        "    # THIS IS NO LONGER A HUMAN DECLARATION. It is regenerated from "
        "the config it",
        "    # describes, so it cannot contradict it and cannot catch an "
        "assignment nobody",
        "    # intended - the check it replaced caught nine of those on "
        "2026-08-27. What",
        "    # still binds is MUST_STAY_ABSENT in the registrar, and the four "
        "registration",
        "    # gates. Order matters: element-wise comparison, appended order.",
        "    #",
    ]
    # Everything from the first non-stamp paragraph onward is preserved. The
    # old stamp is however many leading comment lines precede the first blank
    # comment line, plus that line.
    rest = list(preamble)
    while rest and rest[0].strip().startswith("#"):
        done = rest[0].strip() == "#"
        rest.pop(0)
        if done:
            break
    return stamp + rest


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def sync_expected_assignments(cfg: dict, routed: list[str],
                              path: Path = PORTFOLIO_TEST) -> tuple[bool, str]:
    """
    Rewrite the routing declaration to match the config just written.

    Returns `(changed, message)` and RAISES NOTHING: the registration has
    already been applied by the time this runs, and a declaration that could
    not be rewritten must not read as a registration that did not happen. The
    caller reports the message and exits non-zero.
    """
    live = {s for p in cfg["portfolios"].values()
            for s in (p.get("active_strategies") or [])}
    trespass = sorted(live & MUST_STAY_ABSENT)
    if trespass:
        return False, ("REFUSED to sync: the routing table now holds "
                       + ", ".join(trespass)
                       + " — each cleared Gate R on a rounded 1.00 profit "
                         "factor with negative net P&L in its certified "
                         "quadrant. Registration is already applied; prune "
                         "these from config/portfolios.json.")
    try:
        text = path.read_text(encoding="utf-8")
        start, end, preamble = _split_literal(text)
    except (OSError, ValueError) as exc:
        return False, f"could not read {path.name}: {exc}"

    lines = text.splitlines()
    rendered = render_expected_assignments(
        cfg, _restamp(preamble, routed, len(live)))
    updated = "\n".join(lines[:start] + rendered.splitlines()
                         + lines[end + 1:]) + "\n"
    if updated == text:
        return False, f"{path.name} already matches the config"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(updated, encoding="utf-8")
    try:
        compile(updated, str(path), "exec")
    except SyntaxError as exc:
        tmp.unlink(missing_ok=True)
        return False, (f"the regenerated {path.name} does not parse ({exc}); "
                       f"it is untouched")
    os.replace(tmp, path)
    return True, f"{path.name} synced to {len(live)} routed package(s)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="apply the registrations (default: report only)")
    ap.add_argument("--allow-missing-dow", action="store_true",
                    help="route packages whose Stage 4.5 gate never ran")
    ap.add_argument("--no-sync", dest="sync", action="store_false",
                    help="do not rewrite EXPECTED_ASSIGNMENTS in "
                         "tests/test_portfolio_config.py after --write")
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

    # THE DECLARATION MOVES WITH THE REGISTRATION, in this function rather
    # than in the pipeline that calls it, so a hand-run --write cannot leave
    # the two disagreeing either. The registration above is already on disk;
    # a sync that fails is reported and exits non-zero, and never unwinds it.
    if not args.sync:
        print("  --no-sync: tests/test_portfolio_config.py NOT updated; "
              "its assertion will fail until it is.")
        return 0
    changed, message = sync_expected_assignments(
        cfg, [r["strategy_id"] for r in eligible])
    if changed:
        print(f"  declaration synced → {message}")
        return 0
    print(f"  ! declaration NOT synced: {message}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
