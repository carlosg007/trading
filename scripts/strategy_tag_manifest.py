#!/usr/bin/env python3
"""
strategy_tag_manifest.py - every CrossTrade strategy tag each account can
emit, derived from the routing table.

Location:  ~/src/trading/scripts/strategy_tag_manifest.py

WHAT A TAG IS, AND WHY THERE IS NOTHING TO "POPULATE"
=====================================================
`live_dispatcher.compose_strategy_tag` builds `portfolio:a+b` from the
CONTRIBUTORS to one netted position, at the moment an order is dispatched.
There is no tag registry anywhere in this repository and no manual step that
could be forgotten: a tag is derived from the plan, travels on the wire in the
text command, and is rebuilt identically from the POSITION BOOK when the
position is flattened - CrossTrade matches the lock by string equality, so the
two spellings have to come from one function, and they do.

**So an account showing one tag is an account on which one distinct
contributor-set has traded.** It is not an incomplete mapping. A tag appears
on the journal the first time an order carries it and never before, which is
why a stream running dozens of approved strategies can show a single tag: on
any given cycle most of them do not signal, the ones that do net into one
position per (portfolio, symbol), and that position takes out ONE lock named
for exactly the strategies that asked for it.

WHAT THIS SCRIPT IS FOR
=======================
Pre-registering those tags in a journal or dashboard, so a lock is recognisable
before its first fill rather than after. It enumerates what each account CAN
emit, from `config/portfolios.json` and the promoted `meta.json` files - the
same two inputs `LiveExecutionDispatcher._load_active_strategies` reads, so a
tag here cannot disagree with one on the wire.

IT DOES NOT ENUMERATE EVERY POSSIBLE TAG, and that is a measurement rather than
a shortcut. A netted position names whichever subset of contributors signalled
together, so a symbol with 16 eligible strategies has 2^16 - 1 possible tags.
Listing them would be a manifest nobody could load and would imply a
combinatorial reality the market does not produce. What is listed instead:

  * the SINGLETON tag per (portfolio, symbol, strategy) - one strategy
    signalling alone, which is the common case and the one a journal most
    needs to recognise
  * the FULL-SET tag per (portfolio, symbol) - every eligible strategy
    signalling together, the upper bound

Between them sit the subsets, and `tag_pattern` on each row states the shape so
a journal can match on the prefix rather than on an enumeration.

Reads
-----
    config/portfolios.json and strategies/approved_incubator/*/meta.json.

Writes
------
    Nothing, unless `--out` is given.

Usage
-----
    strat-tags                       # the table
    strat-tags --json                # machine-readable manifest
    strat-tags --account SimPropSim
    strat-tags --out /mnt/backtest/artifacts/strategy_tags.json
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
import json                                                        # noqa: E402
from datetime import datetime, timezone                            # noqa: E402
from typing import Any                                             # noqa: E402

from portfolio.config_loader import (clear_cache,                  # noqa: E402
                                     load_portfolio_config)
from realtime.live_dispatcher import compose_strategy_tag          # noqa: E402

INCUBATOR = PROJECT_ROOT / "strategies" / "approved_incubator"


def certified_symbols(strategy_id: str) -> list[str]:
    """
    The contracts a promoted strategy is certified on, from its own
    `meta.json`. Read rather than assumed: a strategy certified on ES cannot
    contribute to an MNQ position however it is routed, and the live loop
    refuses exactly that.
    """
    meta_path = INCUBATOR / strategy_id / "meta.json"
    if not meta_path.is_file():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    symbols = meta.get("symbols") or ([meta["symbol"]]
                                      if meta.get("symbol") else [])
    return [str(s).strip().upper() for s in symbols if str(s).strip()]


def covers(certified: list[str], asset: str) -> bool:
    """
    Can a strategy certified on `certified` trade `asset`?

    Through the shared micro/full-size table, so a certification on NQ
    authorises MNQ and the reverse - the same resolution the live loop applies
    before it lets a strategy contribute. A second rule here would let this
    manifest name a tag the dispatcher would never emit.
    """
    from realtime.contract_alias import normalize, resolve_parent  # noqa: PLC0415

    # BOTH SIDES resolve to the full-size parent before comparing. Resolving
    # only the asset made this stricter than the dispatcher, which authorises
    # a certification on NQ for MNQ *and the reverse* - so a strategy
    # certified on a micro would have been left off the manifest and its lock
    # would read as unregistered.
    target = resolve_parent(normalize(asset))
    return any(resolve_parent(normalize(c)) == target for c in certified if c)


def build(config_path: str | None = None) -> dict[str, Any]:
    """One row per (portfolio, symbol), with the tags it can emit."""
    clear_cache()
    cfg = load_portfolio_config(config_path or str(
        PROJECT_ROOT / "config" / "portfolios.json"))

    rows: list[dict[str, Any]] = []
    for pid, portfolio in cfg["portfolios"].items():
        account = portfolio["target_account"]
        active = list(portfolio.get("active_strategies") or [])
        for asset in portfolio.get("basket", {}).get("assets") or []:
            eligible = sorted(
                sid for sid in active
                if covers(certified_symbols(sid), asset))
            rows.append({
                "portfolio_id": pid,
                "account": account,
                "account_type": portfolio.get("account_type"),
                "symbol": asset,
                "eligible_strategies": eligible,
                # ONE STRATEGY SIGNALLING ALONE - the common case, and what a
                # journal most needs to recognise before a first fill.
                "singleton_tags": [compose_strategy_tag(pid, [sid])
                                   for sid in eligible],
                # EVERY eligible strategy signalling together - the upper
                # bound. `None` when the pair has none, which is a real state
                # (an asset in the basket that nothing is certified on) and
                # not an empty tag.
                "full_set_tag": (compose_strategy_tag(pid, eligible)
                                 if eligible else None),
                "tag_pattern": f"{pid}:<strategy>[+<strategy>...]",
            })
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "config": str(config_path or PROJECT_ROOT / "config"
                      / "portfolios.json"),
        "composer": "realtime.live_dispatcher.compose_strategy_tag",
        "note": ("A tag names the CONTRIBUTORS to one netted position and is "
                 "composed at dispatch. Subsets between the singleton and the "
                 "full set are possible; match on `tag_pattern` rather than "
                 "on an enumeration."),
        "rows": rows,
    }


def render(manifest: dict[str, Any]) -> str:
    rows = manifest["rows"]
    lines = [
        "=" * 78,
        "STRATEGY TAG MANIFEST — what each account can emit",
        "=" * 78,
        f"generated {manifest['generated_utc']}",
        f"composed by {manifest['composer']}",
        "",
        "A tag is built AT DISPATCH from the contributors to one netted",
        "position. There is no registry to populate: an account showing one",
        "tag is one on which one contributor-set has traded.",
        "",
    ]
    header = ["Account", "Portfolio", "Sym", "Eligible", "Example tag"]
    body = []
    for r in rows:
        example = (r["singleton_tags"][0] if r["singleton_tags"] else "—")
        body.append([r["account"], r["portfolio_id"], r["symbol"],
                     str(len(r["eligible_strategies"])), example])

    widths = [len(h) for h in header]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines.append(" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    lines.append("-+-".join("-" * w for w in widths))
    lines.extend(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row))
                 for row in body)

    total = sum(len(r["singleton_tags"]) for r in rows)
    empty = [r for r in rows if not r["eligible_strategies"]]
    lines += [
        "",
        f"{total} singleton tag(s) across {len(rows)} (portfolio, symbol) "
        f"pairs on {len({r['account'] for r in rows})} accounts.",
    ]
    if empty:
        lines.append(
            f"{len(empty)} pair(s) have NO eligible strategy and can emit no "
            f"tag at all — an asset in a basket that nothing is certified on:")
        for r in empty:
            lines.append(f"   {r['account']}/{r['symbol']} ({r['portfolio_id']})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Every CrossTrade strategy tag each account can emit, "
                    "derived from the routing table and the promoted "
                    "meta.json files. Reads only; composes through the live "
                    "dispatcher's own function.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--account", default=None, help="filter to one account")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None,
                    help="also write the manifest as JSON to this path")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build(args.config)
    if args.account:
        manifest["rows"] = [r for r in manifest["rows"]
                            if r["account"] == args.account]
    print(json.dumps(manifest, indent=2) if args.json else render(manifest))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"\nmanifest {out}")
    # Non-zero only when the routing table could not be read. A pair with no
    # eligible strategy is a real state, not a failure of this tool.
    return 0 if manifest["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
