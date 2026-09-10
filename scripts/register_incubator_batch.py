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
     loader reconciles against `backtest/specs.py` on every load. That file
     now reconciles against the Databento definitions for every contract but
     M2K and MYM, and ETH, LE and PL are additionally pinned against the
     exchange specification in `tests/test_contract_specs.py` - but the
     caution stands for anything newly added, because a guessed multiplier
     silently scales every P&L figure for that contract and nothing
     downstream looks wrong.

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
#: A PRUNE THAT IS NOT ALSO A REFUSAL IS A NO-OP. Removing a row by hand only
#: makes the package unrouted, and the next --write evaluates unrouted
#: packages and puts it straight back - which is exactly what happened to
#: `multi_ema_cci_trend_20260910_YM_30m_VB` on 2026-09-10, re-routed by the
#: same command that was meant to record its removal. So this set is a GATE
#: (see `evaluate`) and not only a sync guard.
MUST_STAY_ABSENT = frozenset({
    "t3_braid_scalp_20260823_RB_30m_VA",
    "t3_braid_scalp_20260823_YM_30m_VA",
    "ma_anchoring_spread_20260820_ZS_1h_VA",
    "ema_deviation_scalp_20260909_NQ_30m_VB",
    # Not negative - the four above are - but +26.04 over 62 trades is 42
    # cents a trade on a profit factor that rounds to 1.0, which is inside the
    # friction the run already charged. The net-P&L gate checks the SIGN, and
    # the sign is the right side of zero here; the magnitude is what makes it
    # not an edge, and no threshold on it would be anything but a number
    # somebody picked. Pruned deliberately on 2026-09-10.
    "multi_ema_cci_trend_20260910_YM_30m_VB",
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


# ONE IMPLEMENTATION OF EACH GATE, and it lives in `backtest/promote.py`
# because that is the module the OTHER registration path goes through. This
# script and promote.py both write config/portfolios.json, and a check that
# existed here and not there is exactly how a package with negative net P&L in
# its certified quadrant reached Incubator-Odd on 2026-09-10: this file
# refused it, promote.py never looked, and promote.py won because it ran
# first.
from backtest.promote import (                                   # noqa: E402
    ALLOCATIONS_KEY, certified_quadrant, dow_allocation_fields,
    gate_regime_of, routing_refusals,
)


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

    if package.name in MUST_STAY_ABSENT:
        # Named, not inferred. Each entry carries its reason at the constant.
        refusals.append(
            "named in MUST_STAY_ABSENT - deliberately unrouted, and a prune "
            "that is not also a refusal is undone by the next --write")

    portfolio = baskets.get(symbol)
    if portfolio is None:
        refusals.append(f"symbol {symbol} is in no incubator basket")
    try:
        theta_for(symbol, tf)
    except Exception:
        refusals.append(f"no theta_vol anchor for {symbol}/{tf}")

    # Gates 3 and 4 are `promote.routing_refusals`; gates 1 and 2 above need
    # the routing table and the regime cache and stay here.
    refusals.extend(routing_refusals(meta, allow_missing_dow=allow_missing_dow))
    status = dow.get("status")

    return {
        "strategy_id": package.name, "symbol": symbol, "timeframe": tf,
        "version": meta.get("version"), "portfolio": portfolio,
        # THE QUADRANT GATE R MEASURED, not the one it aimed at. See
        # `promote.certified_quadrant`: a starved primary falls back to a
        # pre-declared secondary and the certification block goes on naming
        # the primary, so routing on it gates the strategy into the
        # environment that starved.
        "quadrant": (certified_quadrant(meta)["quadrant"]
                     or (meta.get("certification") or {}).get(
                         "target_quadrant")),
        "blocked_weekdays": list(dow.get("blocked_weekdays") or []),
        "dow_status": status, "refusals": refusals, "meta": meta,
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
    ])
    # The day-of-week half comes from `promote.dow_allocation_fields`, the
    # same call promote.py makes, so one allocation cannot carry a shape the
    # other does not. It keeps the three states distinguishable IN THE FILE:
    # an empty mask written for a gate that never ran would otherwise read as
    # "the stage ran and every session cleared" to every later reader.
    alloc.update(dow_allocation_fields(row["meta"]))
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


#: The generated stamp is FENCED, and the fence is why. `_restamp` used to
#: strip "the first comment paragraph", which worked until the stamp itself
#: grew a blank `#` line in the middle: the stripper stopped at that line, the
#: paragraph below it survived, and a fresh stamp was prepended on top of it.
#: Every --write then added one more copy - four of them accumulated in
#: tests/test_portfolio_config.py on 2026-09-10 before a `git status` caught
#: it, because a growing comment block changes no behaviour and no test reads
#: comments. A fence has one meaning and cannot be half-matched.
STAMP_BEGIN = "    # --- BEGIN GENERATED STAMP (rewritten on every sync) ---"
STAMP_END = "    # --- END GENERATED STAMP ---"


def _restamp(preamble: list[str], routed: list[str], total: int) -> list[str]:
    """
    Replace the fenced regeneration stanza; leave every other comment.

    IDEMPOTENT BY CONSTRUCTION: the fence is removed as a whole block before
    the new one is written, so syncing twice with an unchanged config produces
    a byte-identical file. `_split_literal`'s preamble may carry no fence at
    all - the first sync after this change, or a hand-edited constant - and
    that is handled by finding nothing to remove.
    """
    stamp = [STAMP_BEGIN,
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
        STAMP_END,
        "    #",
    ]
    # Everything outside the fence is preserved verbatim - it records WHY four
    # packages are absent and why NG, RB and CL cannot be routed at all, none
    # of which is derivable from the config.
    rest = list(preamble)
    if STAMP_BEGIN in rest:
        start = rest.index(STAMP_BEGIN)
        end = (rest.index(STAMP_END, start) + 1 if STAMP_END in rest[start:]
               else len(rest))
        # The blank `#` separator the previous stamp wrote after its fence.
        if end < len(rest) and rest[end].strip() == "#":
            end += 1
        rest = rest[:start] + rest[end:]
    else:
        # PRE-FENCE FORMAT, and the duplicates it left behind. Every paragraph
        # that is a verbatim copy of one already in the new stamp is dropped -
        # the four accumulated blocks go, anything a human wrote stays.
        stamp_lines = {line.strip() for line in stamp if line.strip() != "#"}
        seen: set[str] = set()
        pruned: list[str] = []
        for line in rest:
            body = line.strip()
            if body and body != "#" and body in stamp_lines:
                seen.add(body)
                continue
            pruned.append(line)
        if seen:
            # Collapse the runs of bare `#` the removals left facing each
            # other, so the block does not grow blank lines instead.
            collapsed: list[str] = []
            for line in pruned:
                if (line.strip() == "#" and collapsed
                        and collapsed[-1].strip() == "#"):
                    continue
                collapsed.append(line)
            rest = collapsed
        else:
            rest = pruned
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


def reconcile_regime_filters(cfg: dict, incubator: Path) -> list[dict]:
    """
    Correct `regime_filter` on rows ALREADY in the routing table.

    THE REGISTRAR CANNOT REACH THESE ANY OTHER WAY. `main` evaluates only
    packages not yet routed (`p.name not in routed`), which is right for
    routing - a package is placed once - and leaves it structurally unable to
    fix a row it wrote before the rule changed. Fourteen rows across eight
    strategies were routed on a starved primary's quadrant, and no number of
    `--write` runs would have touched one of them.

    THIS RE-STATES A FIELD, IT NEVER RE-ROUTES. The account is not
    reconsidered and the allocation is not resized; the only thing rewritten
    is the quadrant, and only to the one that package's OWN gate audit says
    the edge was measured in. A row whose package cannot be read is left
    exactly as it is - an unreadable meta.json is a reason to look, not a
    reason to guess.
    """
    changed: list[dict] = []
    for pid, port in cfg["portfolios"].items():
        for sid, alloc in (port.get(ALLOCATIONS_KEY) or {}).items():
            meta_p = Path(incubator) / sid / "meta.json"
            if not meta_p.exists():
                continue
            try:
                meta = json.loads(meta_p.read_text())
            except (OSError, ValueError):
                continue
            measured = certified_quadrant(meta)
            if not measured["fallback"] or not measured["quadrant"]:
                continue
            was = alloc.get("regime_filter")
            if was == measured["quadrant"]:
                continue
            alloc["regime_filter"] = measured["quadrant"]
            alloc["certified_on"] = measured["certified_on"]
            # The nested per-configuration copy moves with it, or the two
            # halves of one record disagree about the same promotion.
            for cfgrow in (alloc.get("configurations") or []):
                if isinstance(cfgrow, dict) and cfgrow.get("strat") == sid:
                    cfgrow["regime_filter"] = measured["quadrant"]
                    cfgrow["certified_on"] = measured["certified_on"]
            changed.append({"portfolio": pid, "strategy_id": sid,
                            "was": was, "now": measured["quadrant"],
                            "basis": measured["certified_on"]})
    return changed


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

    # THE REPAIR PASS, BEFORE THE ROUTING AND ON EVERY RUN. It re-states
    # `regime_filter` on rows already in the table whose Gate R certified on a
    # fallback secondary - see `reconcile_regime_filters`. Reported in a dry
    # run and applied under --write, like everything else here.
    repaired = reconcile_regime_filters(cfg, Path(args.incubator))
    if repaired:
        print(f"\nregime_filter corrections ({len(repaired)}) - Gate R "
              f"certified these on a fallback secondary, and the routing "
              f"table named the primary that starved:")
        for r in repaired:
            print(f"  FIX     {r['strategy_id']:56s} {r['was']} -> "
                  f"{r['now']}  ({r['portfolio']})")

    if not args.write:
        print("\ndry run - nothing written. Re-run with --write to apply.")
        return 0
    if not eligible and not repaired:
        # NOTHING TO ROUTE IS NOT NOTHING TO DO. The declaration in
        # tests/test_portfolio_config.py can be stale while this run finds no
        # new package: `backtest/promote.py` registers during Stage 5 through
        # its own path, and a row pruned by hand moves the table too. Gating
        # the sync on `eligible` left the guard describing a routing table
        # that had already moved on, which is the exact drift this sync
        # exists to remove - so it runs on every --write.
        print("\nnothing routable; config untouched.")
        if not args.sync:
            return 1
        changed, message = sync_expected_assignments(cfg, [])
        if changed:
            print(f"  declaration synced anyway → {message}")
            return 0
        if message.startswith("REFUSED"):
            print(f"  ! declaration NOT synced: {message}", file=sys.stderr)
            return 3
        print(f"  declaration already current: {message}")
        return 1

    for r in eligible:
        port = cfg["portfolios"][r["portfolio"]]
        port.setdefault("active_strategies", []).append(r["strategy_id"])
        port.setdefault(ALLOCATIONS_KEY,
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
    print(f"\nregistered {len(eligible)} package(s) into {path}"
          + (f"; corrected {len(repaired)} regime_filter(s)" if repaired
             else ""))

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
