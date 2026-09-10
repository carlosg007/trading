#!/usr/bin/env python3
"""
scripts/backfill_portfolio_shapes.py - give every allocation its weekday mask.

    .venv/bin/python3 scripts/backfill_portfolio_shapes.py          # report
    .venv/bin/python3 scripts/backfill_portfolio_shapes.py --write

WRITES NOTHING WITHOUT --write, like every other tool that touches
`config/portfolios.json`. This one edits the table the live daemon reads.

WHY THE MASK IS READ FROM EACH PACKAGE AND NOT SET TO `[]`
=========================================================
`backtest/promote.py` only began writing `blocked_weekdays` onto the
allocation on 2026-09-10 (e05737f). Every row registered before that carries
no such key, and the obvious backfill - add `"blocked_weekdays": []` to
anything missing it - is wrong in a way that is silent and specific.

THE THREE STATES THIS REPOSITORY KEEPS APART:

    no key at all              a record written before the field existed
    `blocked_weekdays: []`     the stage ran and EVERY SESSION CLEARED
    `blocked_weekdays: [4]`    Friday is stood down

A blanket `[]` turns the first into the second. It does not fill a gap, it
makes a claim: that Stage 4.5 profiled this package and found nothing to
block. Measured on this tree before the backfill ran, of the 114 rows missing
the key, ONE HUNDRED had a real blocked weekday in their own meta.json -
`ema_crossover_20260821_NQ_15m_VA` blocks Friday, `dual_ema_slope_scalp
_20260831_NQ_1h_VA` blocks Wednesday, and so on. Writing `[]` over those
records "every session cleared" for a hundred packages that each have a
session measured as negative expectancy.

Nothing live reads this field today - the dispatcher takes the weekday from
`meta.json`'s `day_of_week_gate`, which is the one key CLAUDE.md documents as
the live path - so the damage would not have shown up as a bad trade. It would
have shown up the first time somebody read the routing table to answer "which
of these stand down on Friday" and got the wrong answer from a file that
looked complete.

So the mask is copied from each package's OWN meta.json through
`promote.dow_allocation_fields`, the same call both registrars make. Where a
gate genuinely never ran, that helper writes `[]` AND
`day_of_week_basis: "NOT EVALUATED"` beside it, so the third state survives
being written down rather than being flattened into the second.

A package that is routed but no longer on disk is REPORTED AND SKIPPED. A
missing meta.json is a reason to look, not a reason to guess: the row keeps
no key at all, which is the honest record of "nothing here has been checked".
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.promote import (                                    # noqa: E402
    ALLOCATIONS_KEY, dow_allocation_fields,
)

INCUBATOR = REPO / "strategies" / "approved_incubator"
PORTFOLIOS = REPO / "config" / "portfolios.json"

#: The key this tool exists to add, and the provenance marker that travels
#: with it when the gate never ran.
MASK_KEY = "blocked_weekdays"
BASIS_KEY = "day_of_week_basis"


def _load(path: Path) -> dict[str, Any]:
    """Ordered, so untouched portfolios come back out byte for byte."""
    return json.loads(path.read_text(),
                      object_pairs_hook=collections.OrderedDict)


def _meta_for(strategy_id: str, incubator: Path) -> dict | None:
    """The package's own meta.json, or None when it cannot be read."""
    path = Path(incubator) / strategy_id / "meta.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def backfill(cfg: dict, incubator: Path = INCUBATOR) -> list[dict[str, Any]]:
    """
    Add the weekday mask to every allocation that lacks one.

    Returns one record per row CHANGED. A row that already carries the key is
    left exactly as it is - including its value: this tool fills gaps and does
    not reconcile disagreements, because a mask that disagrees with its
    package is a different finding and wants a human, not a rewrite.

    The nested `configurations` copy moves with the top-level record. Both
    halves describe one promotion and a reader that opened the wrong one would
    get a different answer.
    """
    changed: list[dict[str, Any]] = []
    for pid, port in cfg.get("portfolios", {}).items():
        for sid, alloc in (port.get(ALLOCATIONS_KEY) or {}).items():
            if MASK_KEY in alloc:
                continue
            meta = _meta_for(sid, incubator)
            if meta is None:
                changed.append({"portfolio": pid, "strategy_id": sid,
                                "skipped": "no readable meta.json on disk"})
                continue
            fields = dow_allocation_fields(meta)
            alloc.update(fields)
            for row in (alloc.get("configurations") or []):
                if isinstance(row, dict) and row.get("strat") == sid:
                    row.update(fields)
            changed.append({
                "portfolio": pid, "strategy_id": sid, "skipped": None,
                MASK_KEY: fields.get(MASK_KEY),
                BASIS_KEY: fields.get(BASIS_KEY),
                "status": (meta.get("day_of_week_gate") or {}).get("status"),
            })
    return changed


def write_config(cfg: dict, path: Path) -> None:
    """
    Temp-then-replace, VALIDATED while it is still the temp file.

    The same discipline `scripts/register_incubator_batch.py` uses: a mutation
    that would make the routing table unloadable never reaches the real path,
    so the live daemon cannot be handed a file this tool broke.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    try:
        from portfolio.config_loader import load_portfolio_config
        load_portfolio_config(str(tmp))
    except Exception as exc:                                      # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"REFUSED: the resulting config does not load ({exc}). "
            f"{path} is untouched.")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="apply the backfill (default: report only)")
    ap.add_argument("--portfolios", default=str(PORTFOLIOS))
    ap.add_argument("--incubator", default=str(INCUBATOR))
    args = ap.parse_args(argv)

    path = Path(args.portfolios)
    cfg = _load(path)

    total = sum(len(p.get(ALLOCATIONS_KEY) or {})
                for p in cfg.get("portfolios", {}).values())
    changed = backfill(cfg, Path(args.incubator))
    applied = [c for c in changed if not c["skipped"]]
    skipped = [c for c in changed if c["skipped"]]

    print(f"allocations: {total}   already compliant: {total - len(changed)}   "
          f"backfilled: {len(applied)}   skipped: {len(skipped)}")

    tally = collections.Counter()
    for c in applied:
        mask = c.get(MASK_KEY) or []
        tally["a weekday is blocked" if mask
              else ("gate never ran - basis recorded" if c.get(BASIS_KEY)
                    else "every session cleared")] += 1
    for label, n in tally.most_common():
        print(f"  {label:36} {n}")

    if applied:
        print("\nbackfilled rows:")
        for c in applied:
            basis = f"  [{c[BASIS_KEY]}]" if c.get(BASIS_KEY) else ""
            print(f"  SET     {c['strategy_id']:56s} "
                  f"{MASK_KEY}={c[MASK_KEY]}{basis}")
    if skipped:
        print(f"\nskipped ({len(skipped)}) - a missing package is a reason to "
              f"look, not to guess:")
        for c in skipped:
            print(f"  SKIP    {c['strategy_id']:56s} {c['skipped']}")

    if not args.write:
        print("\ndry run - nothing written. Re-run with --write to apply.")
        return 0
    if not applied:
        print("\nnothing to backfill; config untouched.")
        return 0

    write_config(cfg, path)
    print(f"\nbackfilled {len(applied)} allocation(s) in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
