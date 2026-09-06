#!/usr/bin/env python3
"""
Which weekdays is each promoted strategy actually allowed to trade?

Location: ~/src/trading/scripts/check_strategy_days.py

The card to run after a backtest campaign finishes and before anything is
armed. It joins the two files that decide a live stand-down and reports where
they disagree:

    strategies/approved_incubator/<id>/meta.json   `day_of_week_gate`, which is
                                                   the ONLY thing the live loop
                                                   reads about the calendar
    <BT_ARTIFACTS>/pipeline/<strat>/dow_gate_      Stage 4.5's own verdict for
        <SYMBOL>_<TF>.json                         that pair

READ-ONLY. It classifies nothing, runs no backtest, opens no bar and writes no
file. It reads what other processes wrote, which is the whole design: a status
tool that recomputed a weekday would be free to disagree with the gate it is
describing while carrying a status tool's authority.

WHY THE TWO FILES CAN DISAGREE, AND WHY THAT IS THE POINT
---------------------------------------------------------
The blocked weekday is written into `meta.json` at PROMOTION time and never
revisited. So a package promoted BEFORE Stage 4.5 ran for its pair carries
`status: "NOT EVALUATED"` while a verdict blocking Friday sits in the pipeline
directory reaching nobody - the strategy trades the session it was stood down
from, and every log line reads correctly. That is exactly the failure this
card exists to surface, and it is invisible in `bt-inventory`, in
`signal-check` and on the Discord cards, all of which read one file or the
other and never both.

The reverse disagreement - meta.json blocking a day the current handoff does
not - is the same class of fault arriving from the other side: Stage 4.5 was
re-run with different bars or a different floor and nothing re-promoted.

EXIT CODE ANSWERS "IS ANY PACKAGE OUT OF STEP WITH ITS VERDICT", not "is
something wrong". A package with no Stage 4.5 verdict at all is a legitimate
state - the stage is newer than most of this tree - and is reported without
moving the exit code. A DISAGREEMENT is what exits 1, because it names a
package whose live behaviour is not the behaviour anybody decided on.

    python3 scripts/check_strategy_days.py
    python3 scripts/check_strategy_days.py --strat t3_braid_scalp_20260823
    python3 scripts/check_strategy_days.py --json
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mdlib.env import load_env                                     # noqa: E402

load_env()

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
from datetime import datetime, timezone                            # noqa: E402
from typing import Any                                             # noqa: E402

from backtest.pipeline import (DOW_GATE_FILE, base_strategy,        # noqa: E402
                               pipeline_dir, split_strategy_id,
                               version_of_strategy_id)

INCUBATOR = REPO / "strategies" / "approved_incubator"

#: Monday-first, matching `datetime.weekday()` - which is what
#: `backtest.event_calendar.session_weekday` returns and what every
#: `exclude_days` in this repository is written in. A Sunday-first table here
#: would shift every reported stand-down by a day while both files still
#: parsed.
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
FULL = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday")

#: The trading week. Saturday and Sunday are not "allowed days a strategy
#: declines to use" - they are not sessions, and printing them as available
#: would make a five-day week look like a restriction.
TRADING_WEEK = (0, 1, 2, 3, 4)


def _blocked_from_meta(meta: dict) -> tuple[list[int], str, str]:
    """
    `(weekdays, status, reason)` out of a promoted `meta.json`.

    Mirrors `realtime.live_dispatcher.StrategyHandle._resolve_blocked_weekdays`
    in what it accepts, because the point of this card is to report what the
    LIVE LOOP will do - a reader here that was more forgiving than the loop
    would show a block the loop does not apply.
    """
    block = meta.get("day_of_week_gate")
    if not isinstance(block, dict):
        return [], "NOT EVALUATED", ("meta.json carries no `day_of_week_gate`; "
                                     "this package predates stage 4.5 or was "
                                     "promoted without its verdict")
    raw = block.get("blocked_weekdays")
    if raw is None:
        one = block.get("blocked_weekday")
        raw = [] if one is None else [one]
    if isinstance(raw, str):
        return [], str(raw), str(block.get("reason") or "")
    days = sorted({int(d) for d in raw})
    return (days, str(block.get("status") or "EVALUATED"),
            str(block.get("reason") or ""))


def _handoff_blocked(strategy_id: str, meta: dict,
                     out_dir: str | None) -> dict[str, Any]:
    """
    Stage 4.5's verdict for this package's pair AND VERSION.

    The pair comes from `meta.json` first and from the id only as a fallback,
    the same order `promote.py` resolves it in: `meta["symbol"]` and
    `meta["timeframe"]` are what the promotion recorded, while the id is a
    string that has to be split back apart on a timeframe token.

    THE VERSION IS PART OF THE LOOKUP. The per-pair file carries a verdict per
    version because Version B is Version A's entries minus the ones a
    classifier expected to lose - a different trade list and a different
    weekday table - and both versions of one pair can be promoted as two
    packages. Reading A's verdict for a `..._VB` package would report a
    disagreement that is really this card comparing two different strategies.
    A file profiled without `--ml` carries no B entry, which is reported as
    "no verdict" and not as a disagreement.
    """
    strat = str(meta.get("strategy") or base_strategy(strategy_id))
    symbol = meta.get("symbol")
    tf = meta.get("timeframe")
    if not symbol or not tf:
        _s, sym2, tf2 = split_strategy_id(strategy_id)
        symbol, tf = symbol or sym2, tf or tf2
    version = str(meta.get("version")
                  or version_of_strategy_id(strategy_id) or "A").upper()
    if not symbol or not tf:
        return {"found": False, "path": None, "blocked": [], "version": version,
                "reason": ("this package names no (symbol, timeframe) pair, "
                           "so no per-pair verdict can be located")}
    path = (pipeline_dir(strat, out_dir)
            / DOW_GATE_FILE.format(symbol=str(symbol).upper(),
                                   tf=str(tf).lower()))
    if not path.exists():
        return {"found": False, "path": str(path), "blocked": [],
                "version": version,
                "reason": "stage 4.5 left no verdict for this pair"}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"found": False, "path": str(path), "blocked": [],
                "version": version, "reason": f"{type(e).__name__}: {e}"}
    profiled = sorted((blob.get("versions") or {}))
    entry = (blob.get("versions") or {}).get(version)
    if entry is None:
        return {"found": False, "path": str(path), "blocked": [],
                "version": version, "versions_profiled": profiled,
                "reason": (f"stage 4.5 profiled {profiled or 'nothing'} for "
                           f"this pair, not Version {version} — it runs "
                           f"Version B only with --ml")}
    day = entry.get("blocked_weekday")
    return {"found": True, "path": str(path), "version": version,
            "versions_profiled": profiled,
            "blocked": ([] if day is None else [int(day)]),
            "worst": (entry.get("verdict") or {}).get("worst_day"),
            "reason": (entry.get("verdict") or {}).get("reason", "")}


def scan(strat: str | None = None, incubator: Path = INCUBATOR,
         out_dir: str | None = None) -> list[dict[str, Any]]:
    """One row per promoted package, joined to its Stage 4.5 verdict."""
    rows: list[dict[str, Any]] = []
    if not incubator.exists():
        return rows
    for d in sorted(p for p in incubator.iterdir() if p.is_dir()):
        meta_p = d / "meta.json"
        if not meta_p.exists():
            rows.append({"strategy_id": d.name, "error": "no meta.json",
                         "blocked": [], "status": "UNREADABLE",
                         "agrees": None})
            continue
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            rows.append({"strategy_id": d.name,
                         "error": f"{type(e).__name__}: {e}",
                         "blocked": [], "status": "UNREADABLE",
                         "agrees": None})
            continue
        if strat and base_strategy(d.name) != strat and \
                str(meta.get("strategy") or "") != strat:
            continue

        blocked, status, reason = _blocked_from_meta(meta)
        handoff = _handoff_blocked(d.name, meta, out_dir)
        # AGREEMENT IS TRI-STATE. None means there is no verdict to agree
        # with, which is not the same as agreeing - collapsing the two would
        # let a package with no Stage 4.5 evidence at all read as confirmed.
        agrees = (None if not handoff["found"]
                  else sorted(blocked) == sorted(handoff["blocked"]))
        rows.append({
            "strategy_id": d.name,
            "strategy": meta.get("strategy"),
            "symbol": meta.get("symbol"),
            "timeframe": meta.get("timeframe"),
            "version": meta.get("version"),
            "blocked": blocked,
            "blocked_named": [FULL[d_] for d_ in blocked],
            "active_days": [d_ for d_ in TRADING_WEEK if d_ not in blocked],
            "status": status,
            "reason": reason,
            "handoff": handoff,
            "agrees": agrees,
            "error": "",
        })
    return rows


def session_today() -> dict[str, Any]:
    """
    Which CME session weekday it is right NOW, through the backtest's own rule.

    `backtest.event_calendar.session_weekday` and nothing else - the same
    function Stage 4.5 attributed every trade with and the live gate reads the
    fill bar through. A `datetime.now().weekday()` here would answer correctly
    for most of the day and wrongly for the six hours after 18:00 ET, which is
    precisely the window an operator is most likely to be reading this card in.
    """
    from backtest.event_calendar import session_weekday            # noqa: PLC0415

    now = datetime.now(timezone.utc)
    d = int(session_weekday([now])[0])
    return {"utc": now.isoformat(timespec="seconds"), "weekday": d,
            "day": DAYS[d], "day_name": FULL[d]}


def render(rows: list[dict[str, Any]], session: dict[str, Any]) -> str:
    W = 78
    out = ["", "=" * W,
           "  STRATEGY ACTIVE DAYS  ·  what stage 4.5 blocked, and what the",
           "  live loop will actually enforce",
           "=" * W,
           f"  session now : {session['day_name']} "
           f"(weekday {session['weekday']}, {session['utc']})",
           "                CME sessions roll at 18:00 ET, so this is the "
           "SESSION day,",
           "                not the calendar day.",
           ""]
    if not rows:
        out += ["  (no promoted packages under "
                f"{INCUBATOR.relative_to(REPO)}/)", "=" * W, ""]
        return "\n".join(out)

    head = (f"  {'STRATEGY':<44}{'SYM':<6}{'TF':<5}{'VER':<5}{'BLOCKED':<10}"
            f"{'ACTIVE':<20}{'TODAY':<9}CHECK")
    out += [head, "  " + "-" * (len(head) - 2)]
    for r in rows:
        if r.get("error"):
            out.append(f"  {r['strategy_id']:<44}{'':<6}{'':<5}{'':<5}"
                       f"{'?':<10}{'?':<20}{'?':<9}{r['error']}")
            continue
        blocked = "".join(DAYS[d] for d in r["blocked"]) or "none"
        active = " ".join(DAYS[d] for d in r["active_days"])
        # WHAT THE LOOP WOULD DO RIGHT NOW. The one column an operator
        # actually came for: "is this thing muted at this moment".
        today = ("MUTED" if session["weekday"] in r["blocked"] else "trading")
        if r["agrees"] is True:
            check = "ok"
        elif r["agrees"] is False:
            check = (f"DISAGREES with stage 4.5 "
                     f"({''.join(DAYS[d] for d in r['handoff']['blocked']) or 'none'})")
        else:
            check = f"no verdict ({r['handoff']['reason']})"
        out.append(f"  {r['strategy_id']:<44}{str(r['symbol'] or '?'):<6}"
                   f"{str(r['timeframe'] or '?'):<5}"
                   f"{str(r['version'] or '?'):<5}{blocked:<10}{active:<20}"
                   f"{today:<9}{check}")

    stale = [r for r in rows if r.get("agrees") is False]
    unevaluated = [r for r in rows if r.get("agrees") is None
                   and not r.get("error")]
    out += ["", "  " + "-" * (W - 4)]
    out.append(f"  {len(rows)} package(s)   "
               f"{len([r for r in rows if r.get('blocked')])} with a blocked "
               f"session   {len(stale)} out of step   "
               f"{len(unevaluated)} with no verdict")
    if stale:
        out += ["",
                "  ! OUT OF STEP. meta.json is what the live loop reads, and it "
                "is written at",
                "    PROMOTION time and never revisited. These packages will "
                "enforce the",
                "    weekday in their own meta.json, not the one stage 4.5 last "
                "decided:"]
        for r in stale:
            out.append(f"      {r['strategy_id']}: meta says "
                       f"{[DAYS[d] for d in r['blocked']] or 'nothing'}, "
                       f"stage 4.5 says "
                       f"{[DAYS[d] for d in r['handoff']['blocked']] or 'nothing'}")
        out += ["", "    Re-promote to bring them into step:",
                "      python3 backtest/promote.py --strat <strategy> "
                "--version <A|B> \\",
                "          --source <module.py> --audit-file "
                "<gate_audit_<SYM>_<TF>.json> \\",
                "          --dow-gate <dow_gate_<SYM>_<TF>.json>"]
    if unevaluated:
        out += ["",
                "  These have no stage 4.5 verdict at all, which is a "
                "legitimate state and",
                "  not a fault: no weekday is blocked and none was cleared. "
                "Run stage 4.5",
                "  for the pair, then re-promote, if you want one:",
                "      python3 backtest/dow_gate.py --strat <strategy> "
                "--tf <TF>"]
    out += ["=" * W, ""]
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Which weekdays each promoted strategy may trade, and "
                    "whether meta.json still agrees with stage 4.5. Read-only.")
    p.add_argument("--strat", default=None,
                   help="Only packages belonging to this strategy module")
    p.add_argument("--incubator", default=str(INCUBATOR),
                   help=f"Promoted packages root (default {INCUBATOR})")
    p.add_argument("--out-dir", default=None,
                   help="An explicit pipeline DIRECTORY to read the stage 4.5 "
                        "verdicts from, the same override every stage takes. "
                        "It replaces the whole "
                        "<BT_ARTIFACTS>/pipeline/<strategy>/ path, so it is "
                        "only meaningful alongside --strat; to move the root "
                        "for every strategy at once, set $BT_ARTIFACTS.")
    p.add_argument("--json", action="store_true",
                   help="Emit the rows as JSON instead of the card")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = scan(args.strat, Path(args.incubator), args.out_dir)
    session = session_today()
    if args.json:
        print(json.dumps({"session": session, "packages": rows},
                         indent=2, default=str))
    else:
        print(render(rows, session))
    # Only a DISAGREEMENT moves the exit code. A package with no verdict is a
    # legitimate state and a package that agrees is the normal one; failing on
    # either would make this unusable as a chained pre-flight check.
    return 1 if any(r.get("agrees") is False or r.get("error")
                    for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
