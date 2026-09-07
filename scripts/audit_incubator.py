#!/usr/bin/env python3
"""
scripts/audit_incubator.py - what is in approved_incubator/, and what routes.

Location: ~/src/trading/scripts/audit_incubator.py

    python3 scripts/audit_incubator.py                 # the table, writes nothing
    python3 scripts/audit_incubator.py --family t3_braid_scalp_20260823
    python3 scripts/audit_incubator.py --purge-unrouted --family ema_crossover_20260821
    python3 scripts/audit_incubator.py --purge-unrouted --family X --yes

THE THREE POPULATIONS THIS EXISTS TO KEEP APART
===============================================
`approved_incubator/` and `config/portfolios.json` disagree by design, and
reading either one alone gives a wrong answer to "what is deployed":

  ON DISK   a promoted package. `approved_incubator/<id>/` is a record that a
            VERSION WAS CHOSEN - Stage 5 wrote it - and is explicitly not
            permission to trade it. Deleting one throws away the audit trail
            of a certification.
  ROUTED    named in `active_strategies` on a portfolio. THIS is what the live
            dispatcher walks, and the only population whose removal changes
            what the account does.
  ORPHANED  routed, but with no package on disk. The live loop raises
            FileNotFoundError on it every cycle - measured on 2026-09-07,
            after 521 tracked files were removed from the working tree while
            the routing table still named 104 of them.

**A ROUTED PACKAGE IS NEVER PURGED HERE, WHATEVER IS ASKED.** Removing one
leaves the routing table pointing at a directory that is gone, which is the
ORPHANED state above - it does not stand a strategy down, it makes the live
loop throw. Unrouting is a `config/portfolios.json` edit and belongs to
`promotion_daemon` / the portfolio work, not to a directory sweep.

Removal is `git rm`, never `rm`. Every package here was committed by
`promote.py` in its own commit, so the deletion is a reviewable change on top
of that history rather than a hole in the working tree that the next
`git status` reports as damage.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INCUBATOR = REPO / "strategies" / "approved_incubator"
PORTFOLIOS = REPO / "config" / "portfolios.json"
META = "meta.json"


def routed_ids(config_path: Path) -> dict[str, list[str]]:
    """`active_strategies` across every portfolio: id -> the portfolios naming it.

    Read as plain JSON rather than through `portfolio.config_loader`, which
    RAISES for an unassigned strategy - the ordinary state this tool reports.
    """
    blob = json.loads(config_path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for pid, p in (blob.get("portfolios") or {}).items():
        for sid in p.get("active_strategies") or []:
            out.setdefault(str(sid), []).append(pid)
    return out


def packages(incubator: Path) -> list[str]:
    """Every promoted package on disk, by directory name."""
    if not incubator.is_dir():
        return []
    return sorted(d.name for d in incubator.iterdir()
                  if d.is_dir() and (d / META).exists())


def family_of(package_id: str) -> str:
    """
    The MODULE a package was promoted from - `<module>_<SYMBOL>_<TF>_V<A|B>`.

    Split from the RIGHT on exactly three fields, so a module whose own name
    carries an underscore-separated token (every one of them does) is not
    truncated. A package promoted under a bare module name - the older layout,
    with no pair suffix - is its own family.
    """
    parts = package_id.rsplit("_", 3)
    if len(parts) == 4 and parts[3][:1] == "V" and len(parts[3]) == 2:
        return parts[0]
    return package_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit approved_incubator/ against the routing table.")
    ap.add_argument("--incubator", default=str(INCUBATOR))
    ap.add_argument("--portfolios", default=str(PORTFOLIOS))
    ap.add_argument("--family", action="append", default=[],
                    help="restrict to this module (repeatable). Required for "
                         "--purge-unrouted: a sweep across every family at "
                         "once is the shape of the accident this audits.")
    ap.add_argument("--purge-unrouted", action="store_true",
                    help="git rm the UNROUTED packages of the named families. "
                         "Prints the plan and writes nothing without --yes.")
    ap.add_argument("--yes", action="store_true",
                    help="actually run the git rm the plan describes.")
    args = ap.parse_args(argv)

    incubator, config_path = Path(args.incubator), Path(args.portfolios)
    try:
        routed = routed_ids(config_path)
    except (OSError, ValueError) as exc:
        print(f"cannot read the routing table {config_path}: {exc}",
              file=sys.stderr)
        return 1
    on_disk = packages(incubator)

    disk_set = set(on_disk)
    orphaned = sorted(sid for sid in routed if sid not in disk_set)

    fams: dict[str, dict[str, list[str]]] = {}
    for pid in on_disk:
        fam = fams.setdefault(family_of(pid), {"routed": [], "unrouted": []})
        fam["routed" if pid in routed else "unrouted"].append(pid)

    wanted = set(args.family)
    shown = {f: v for f, v in fams.items() if not wanted or f in wanted}
    if wanted:
        for f in sorted(wanted - set(fams)):
            print(f"  NOTE  no packages on disk for family {f!r}")

    print(f"approved_incubator: {len(on_disk)} package(s), "
          f"{len(routed)} routed id(s) in {config_path.name}\n")
    print(f"  {'FAMILY':<44} {'ON DISK':>8} {'ROUTED':>7} {'UNROUTED':>9}")
    for fam in sorted(shown):
        v = shown[fam]
        total = len(v["routed"]) + len(v["unrouted"])
        print(f"  {fam:<44} {total:>8} {len(v['routed']):>7} "
              f"{len(v['unrouted']):>9}")
    tot_r = sum(len(v["routed"]) for v in shown.values())
    tot_u = sum(len(v["unrouted"]) for v in shown.values())
    print(f"  {'':<44} {tot_r + tot_u:>8} {tot_r:>7} {tot_u:>9}")

    if orphaned:
        print(f"\n  ORPHANED - routed with NO package on disk "
              f"({len(orphaned)}). The live loop raises on each of these "
              f"every cycle; restore them rather than unrouting blind:")
        for sid in orphaned:
            print(f"    {sid}  ({', '.join(routed[sid])})")
        print("    git restore -- strategies/approved_incubator/")

    if not args.purge_unrouted:
        print("\n  Nothing was written. --purge-unrouted --family <module> "
              "plans a removal.")
        return 0

    if not wanted:
        print("\n  REFUSED  --purge-unrouted needs at least one --family. "
              "Sweeping every family at once is the shape of the accident "
              "this tool audits.", file=sys.stderr)
        return 2

    doomed = [pid for fam in sorted(shown) for pid in shown[fam]["unrouted"]]
    kept = [pid for fam in sorted(shown) for pid in shown[fam]["routed"]]
    if not doomed:
        print("\n  Nothing to purge: every package in the named families is "
              "routed.")
        return 0

    # TRACKED AND UNTRACKED ARE NOT THE SAME REMOVAL, and conflating them cost
    # a whole batch: `git rm` FATALS on an untracked path and aborts every
    # other path with it, so one directory Stage 5 never committed stopped 107
    # legitimate removals. That is the harmless half of the problem. The other
    # half is that an untracked package has NO git history to restore from -
    # `git rm` stages a deletion a `git reset` undoes, while deleting an
    # untracked directory is simply gone. So they are separated here, only the
    # tracked ones are removed, and the untracked ones are REPORTED with the
    # command that would remove them rather than swept up silently.
    #
    # A THIRD population sits between them: TRACKED BUT LOCALLY MODIFIED.
    # `git rm` refuses those outright (exit 1) rather than discarding work
    # nobody committed, and `-f` is the flag that overrides it. This tool does
    # not pass `-f`. An edit sitting in the working tree is the one copy of
    # itself that exists, and a directory sweep is not where somebody should
    # discover it is gone - so these are held back and named too.
    modified = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", "--",
         str(incubator.relative_to(REPO))],
        capture_output=True, text=True, check=False).stdout.split()
    dirty = {Path(f).parts[2] for f in modified if len(Path(f).parts) > 2}

    tracked, untracked, changed = [], [], []
    for pid in doomed:
        rel = str((incubator / pid).relative_to(REPO))
        probe = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "--error-unmatch", "--", rel],
            capture_output=True, check=False)
        if probe.returncode != 0:
            untracked.append(pid)
        elif pid in dirty:
            changed.append(pid)
        else:
            tracked.append(pid)

    print(f"\n  PLAN  git rm -r {len(tracked)} tracked unrouted package(s); "
          f"{len(kept)} routed package(s) are NOT touched.")
    for pid in tracked:
        print(f"    rm   {pid}")
    for pid in kept:
        print(f"    keep {pid}  (routed: {', '.join(routed[pid])})")
    if untracked:
        print(f"\n  NOT REMOVED - {len(untracked)} unrouted package(s) are "
              f"UNTRACKED. Stage 5 never committed these, so there is no "
              f"history to restore them from and this tool will not delete "
              f"them. Remove them by hand if that is what you want:")
        for pid in untracked:
            print(f"    rm -rf strategies/approved_incubator/{pid}")
    if changed:
        print(f"\n  NOT REMOVED - {len(changed)} unrouted package(s) carry "
              f"UNCOMMITTED local modifications. `git rm` refuses these and "
              f"this tool does not pass -f: the edit in the working tree is "
              f"the only copy of itself. Commit or discard it first, then "
              f"re-run:")
        for pid in changed:
            print(f"    {pid}")
            for f in sorted(f for f in modified
                            if Path(f).parts[2:3] == (pid,)):
                print(f"        M {f}")

    if not args.yes:
        print("\n  DRY RUN  nothing was written. Re-run with --yes.")
        return 0
    if not tracked:
        print("\n  Nothing tracked to remove.")
        return 0

    paths = [str((incubator / pid).relative_to(REPO)) for pid in tracked]
    proc = subprocess.run(["git", "-C", str(REPO), "rm", "-r", "-q", "--"]
                          + paths, check=False)
    if proc.returncode:
        print(f"\n  git rm exited {proc.returncode}; nothing else was done.",
              file=sys.stderr)
        return proc.returncode
    print(f"\n  Removed {len(tracked)} package(s) and STAGED the deletion. "
          f"Review with `git diff --cached --stat` and commit with an "
          f"explicit pathspec; `git reset` undoes it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
