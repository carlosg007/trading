#!/usr/bin/env python3
"""
realtime/check_portfolio_assets.py - who is allowed to trade what, and where.

Location:  ~/src/trading/realtime/check_portfolio_assets.py

    python3 realtime/check_portfolio_assets.py
    portfolio-check / assets-check                 # the aliases

The other cards answer questions about MOTION - are bars arriving, what did
the loop decide, why is nothing firing. This one answers a question about
CONFIGURATION: which portfolios exist, which account each addresses, what each
is permitted to trade, and which strategy is allocated to which contract.

FOUR FILES MEET HERE AND NONE IS AUTHORITATIVE ALONE
-----------------------------------------------------
    config/portfolios.json            what is ALLOCATED, and to which account
    approved_incubator/<id>/meta.json what was PROMOTED - the risk parameters
                                      and the certification behind them
    data/live_regime_state.json       what the gate DECIDED on the last bar,
                                      and how far into warm-up the tape is
    the NT8 spool                     what is actually ARRIVING

An id present in one and absent from another is a real and quiet fault - a
strategy allocated but never promoted, or promoted and never republished to
the switchboard - so each source is reported as found or missing rather than
merged into a single row that hides which one is empty.

THE FULL-SIZE / MICRO SPLIT IS THE THING TO GET RIGHT
-----------------------------------------------------
The regime is measured on the FULL-SIZE tape and the order is sent for the
MICRO. `realtime/contract_alias.py` is one-directional on purpose: a micro
resolves to its parent, never the reverse. So a strategy certified on NQ is
gated on NQ's quadrant and executed as MNQ, and the routing matrix below is
read out of that table rather than restated - two spellings of "MNQ means NQ"
that could drift would put an order on the wrong contract with every log line
reading correctly.

WHAT IT DOES NOT DO
-------------------
It reads. It computes no indicator, re-runs no gate, and reports no open
position: `PositionBook` is deliberately not persisted, so exposure is
knowable only inside the running process. What is shown here is CAPACITY - the
caps an order would be checked against - not usage.

COST
----
No pandas, no engine. `check_trade_firewall` (22ms) and `check_nt8_feed` are
imported so the interlock and the spool rules have one spelling. The six micro
CONTRACT NAMES are restated rather than read from `backtest.specs`, which
pulls numpy and pandas; they are pinned against it by the tests.
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
import time                                                        # noqa: E402
from typing import Any                                             # noqa: E402

REPO = PROJECT_ROOT

from realtime.check_trade_firewall import (                        # noqa: E402
    MIN_BARS_FOR_REGIME, interlock, num, read_json, short, wrap)
from realtime.check_nt8_feed import resolve_spool, spool_stats     # noqa: E402
from realtime.contract_alias import (                              # noqa: E402
    MICRO_TO_PARENT, micros_of, not_traded_reason)
from realtime.risk_firewall import DEFAULT_LIMITS                  # noqa: E402

W = 80
PORTFOLIO_CONFIG = REPO / "config" / "portfolios.json"
REGIME_STATE = REPO / "data" / "live_regime_state.json"
INCUBATOR = REPO / "strategies" / "approved_incubator"

#: The six micro contract names, restated from `backtest.specs.SPECS` because
#: that module pulls numpy and pandas (~250ms) for six strings. Pinned against
#: it by `test_the_micro_names_match_the_contract_specs`, so the copy is
#: checked rather than trusted. The MAPPING itself is never restated - it is
#: read from `contract_alias.MICRO_TO_PARENT`, which is the only place it may
#: live.
MICRO_NAMES = {
    "MNQ": "Micro E-mini Nasdaq",
    "MES": "Micro E-mini S&P 500",
    "M2K": "Micro E-mini Russell",
    "MYM": "Micro E-mini Dow",
    "MGC": "Micro Gold",
    "MCL": "Micro Crude Oil",
}


# ==========================================================================
# reading
# ==========================================================================

def portfolios() -> dict[str, Any]:
    """
    Every configured portfolio, with the three overlays applied per strategy.

    Portfolios with NO active strategy are kept rather than filtered out. An
    empty portfolio is a deliberate state - three of the four on this box are
    empty by design - and a card that showed only the populated one would make
    "nothing is allocated here" indistinguishable from "this portfolio does
    not exist".
    """
    out: dict[str, Any] = {"rows": [], "error": None, "version": None}
    cfg = read_json(PORTFOLIO_CONFIG)
    if cfg is None:
        out["error"] = f"{short(PORTFOLIO_CONFIG)} is unreadable"
        return out
    out["version"] = cfg.get("version")
    regime = read_json(REGIME_STATE) or {}
    switchboard = regime.get("strategies") or {}
    live_symbols = regime.get("symbols") or {}

    for name, p in sorted((cfg.get("portfolios") or {}).items()):
        basket = p.get("basket") or {}
        risk = p.get("risk_profile") or {}
        clamp = risk.get("clamping") or {}
        allocations = p.get("strategy_allocations") or {}
        strategies = []
        for sid in (p.get("active_strategies") or []):
            alloc = allocations.get(sid) or {}
            gate = switchboard.get(sid)
            meta = read_json(INCUBATOR / sid / "meta.json")
            symbol = (gate or {}).get("symbol") or alloc.get("symbol")
            tf = (gate or {}).get("timeframe") or alloc.get("timeframe")
            live = live_symbols.get(str(symbol or "").upper()) or {}
            n_bars = live.get("n_bars")
            strategies.append({
                "id": sid,
                "symbol": symbol,
                "timeframe": tf,
                "micros": micros_of(symbol) if symbol else [],
                "allocation": alloc.get("allocation"),
                "alloc_status": alloc.get("status"),
                "alloc_regime": alloc.get("regime_filter"),
                "meta": meta,
                "gate": gate,
                "n_bars": n_bars,
                "warm": (None if n_bars is None
                         else int(n_bars) >= MIN_BARS_FOR_REGIME),
            })
        out["rows"].append({
            "name": name,
            "account": p.get("target_account"),
            "account_type": p.get("account_type"),
            "account_size": p.get("default_account_size"),
            "assets": list(basket.get("assets") or []),
            "correlation_group": basket.get("correlation_group"),
            "clamp_min": clamp.get("min_contracts"),
            "clamp_max": clamp.get("max_contracts"),
            "risk_budget": risk.get("fixed_risk_budget_usd"),
            "max_dd": risk.get("max_trailing_drawdown_usd"),
            "strategies": strategies,
        })
    return out


def unallocated(rows: list[dict]) -> dict[str, Any]:
    """
    Contracts arriving in the spool that no allocated strategy watches.

    Read from the SPOOL, not the regime file: the daemon classifies only its
    registered targets, so the regime file lists one symbol while the feed
    carries a dozen. Sourcing this from the regime file compares a set against
    itself and reports nothing, every time.

    A symbol declared in `contract_alias.NOT_TRADED` is separated out rather
    than hidden - the decision is recorded, so it is reported as a decision.
    """
    watched: set[str] = set()
    for p in rows:
        watched.update(a.upper() for a in p["assets"])
        for s in p["strategies"]:
            if s["symbol"]:
                watched.add(str(s["symbol"]).upper())
            watched.update(m.upper() for m in s["micros"])

    directory, source = resolve_spool(None)
    stats = spool_stats(directory)
    arriving = {f["symbol"].upper() for f in stats.get("files", [])}
    loose = sorted(arriving - watched)
    declared = [(s, not_traded_reason(s)) for s in loose]
    return {"dir": directory, "source": source, "exists": stats["exists"],
            "watched": sorted(watched), "arriving": sorted(arriving),
            "unallocated": [s for s, why in declared if not why],
            "declared": [(s, why) for s, why in declared if why]}


# ==========================================================================
# the card
# ==========================================================================

def collect() -> dict[str, Any]:
    pf = portfolios()
    return {"now": time.time(), "portfolios": pf,
            "unallocated": unallocated(pf["rows"]),
            "interlock": interlock(), "limits": dict(DEFAULT_LIMITS)}


def render(snap: dict[str, Any]) -> str:
    pf, un, il = snap["portfolios"], snap["unallocated"], snap["interlock"]
    rows = pf["rows"]
    lim = snap["limits"]
    L: list[str] = ["=" * W, "PORTFOLIO ASSETS & ALLOCATION STATUS", "=" * W]

    if pf["error"]:
        L.append(f"  [!] {pf['error']}")
        L.append("=" * W)
        return "\n".join(L)

    populated = [r for r in rows if r["strategies"]]
    n_strats = sum(len(r["strategies"]) for r in rows)
    accounts = sorted({r["account"] for r in populated if r["account"]})

    L.append(f"{'Portfolios Configured':<28}: {len(rows)} "
             f"({len(populated)} with an active allocation, "
             f"{len(rows) - len(populated)} empty by design)")
    L.append(f"{'Strategies Allocated':<28}: {n_strats} across "
             f"{len(accounts)} account(s)"
             + (f" — {', '.join(accounts)}" if accounts else ""))
    if pf["version"]:
        L.append(f"{'Routing Table Schema':<28}: {pf['version']}")

    # `interlock()` reports ARMED: its `armed()` helper matches `--live` as a
    # whole flag, so True means the loop sends orders. This read the polarity
    # backwards and printed LIVE precisely BECAUSE the loop was unarmed - and
    # would have printed DRY RUN once somebody armed it. Both directions are
    # wrong and the second is the dangerous one.
    #
    # The same inversion in check_trade_firewall.py "cost a deployment on
    # 2026-08-27" (see `interlock`'s docstring); the fix never reached here.
    # Dry run is the DEFAULT - a unit carrying neither flag sends nothing - so
    # anything that is not positively armed is reported as dry run.
    if il["running"] is True:
        mode = "LIVE — the running loop WILL send orders"
    elif il["running"] is False:
        mode = "DRY RUN — the running loop formats orders and sends none"
    else:
        mode = ("no loop running; systemd would start it "
                + ("LIVE" if il["effective"] else "in DRY RUN"))
    L.append(f"{'Execution Mode':<28}: {mode}")

    # ---- per portfolio -----------------------------------------------
    for r in rows:
        L.append("")
        L.append(f"--- Portfolio: {r['name']} ---")
        L.append(f"  Account            : {r['account']} "
                 f"(type {r['account_type']}"
                 + (f", size ${r['account_size']:,}" if r["account_size"]
                    else "") + ")")
        L.append(f"  Permitted Basket   : "
                 f"{', '.join(r['assets']) if r['assets'] else 'nothing'}"
                 + (f"  [{r['correlation_group']}]"
                    if r["correlation_group"] else ""))
        L.append(f"  Sizing Clamp       : {r['clamp_min']}–{r['clamp_max']} "
                 f"contracts per order (portfolio) · "
                 f"{lim['max_contracts_per_order']} (firewall ceiling)")
        L.append(f"  Capacity           : {lim['max_open_positions']} open "
                 f"positions max · risk budget "
                 f"${num(r['risk_budget'], ',.0f')} · trailing DD "
                 f"${num(r['max_dd'], ',.0f')}")
        L.append("                       (caps, not usage — live exposure is "
                 "not persisted anywhere)")

        if not r["strategies"]:
            L.append("  Allocations        : none — this portfolio is empty")
            continue

        L.append(f"  Allocations ({len(r['strategies'])}):")
        for i, s in enumerate(r["strategies"], 1):
            L.append(f"   {i}. {s['id']}")
            micro = ", ".join(s["micros"]) if s["micros"] else "none"
            L.append(f"      Full-size tape : {s['symbol']} @ "
                     f"{s['timeframe']}"
                     + (" (resampled from the 1m spool)"
                        if s["timeframe"] != "1m" else ""))
            L.append(f"      Execution      : {micro} · allocation "
                     f"{s['allocation']} · {s['alloc_status']}")

            meta = s["meta"]
            if meta is None:
                L.append(f"      Promoted spec  : NOT FOUND — no "
                         f"approved_incubator/{s['id']}/meta.json.")
                L.append("                       The table allocates a "
                         "strategy nothing promoted.")
            else:
                risk = meta.get("risk") or {}
                L.append(f"      Parameters     : "
                         f"SL={num(risk.get('sl_atr_mult'), '.1f')} ATR, "
                         f"TP={num(risk.get('tp_atr_mult'), '.1f')} ATR, "
                         f"trailing={risk.get('trailing')} · version "
                         f"{meta.get('version')}")
                L.append(f"      Certified on   : {meta.get('symbols')} @ "
                         f"{meta.get('timeframe')}")

            gate = s["gate"]
            if gate is None:
                L.append("      Gate status    : NO SWITCHBOARD ENTRY — the "
                         "regime daemon publishes nothing for this id")
                continue
            target = gate.get("optimal_regime")
            if s["alloc_regime"] and s["alloc_regime"] != target:
                # The routing table and the certification disagreeing about
                # which quadrant this is for is invisible downstream: the
                # daemon gates on one and a reader trusts the other.
                L.append(f"      [!] regime_filter in the routing table is "
                         f"{s['alloc_regime']}, certification says {target}")
            L.append(f"      Target regime  : {target} "
                     f"({gate.get('optimal_regime_label')})")
            if s["n_bars"] is not None and not s["warm"]:
                L.append(f"      Current status : WARM-UP "
                         f"{s['n_bars']}/{MIN_BARS_FOR_REGIME} bars · "
                         f"{gate.get('live_regime')}")
            else:
                L.append(f"      Current status : "
                         f"{'ENTRIES ALLOWED' if gate.get('entries_allowed') else 'BLOCKED'}"
                         f" ({gate.get('status')} · {gate.get('reason')}) · "
                         f"live {gate.get('live_quadrant')}")

    # ---- routing matrix ----------------------------------------------
    L.append("")
    L.append("--- Micro Execution Routing Matrix ---")
    for parent, micro in sorted((v, k) for k, v in MICRO_TO_PARENT.items()):
        L.append(f"  • {parent:<4} → {micro:<4} "
                 f"{MICRO_NAMES.get(micro, '')}")
    L.append("  The regime and every indicator are computed on the FULL-SIZE "
             "tape; the order is")
    L.append("  sent for the micro. The alias table is one-directional: a "
             "micro resolves to its")
    L.append("  parent, never the reverse.")

    # ---- unallocated -------------------------------------------------
    L.append("")
    L.append("--- Ingested Streams Without An Allocation ---")
    if not un["exists"]:
        L.append(f"  the spool directory {un['dir']} does not exist.")
    elif un["unallocated"]:
        L.append(f"  {len(un['unallocated'])} contract(s) arriving that no "
                 f"allocated strategy watches:")
        for ln in wrap(", ".join(un["unallocated"]), indent=6):
            L.append(f"      {ln}")
        L.append("  Spooled for research and nothing else: no quadrant is "
                 "computed and no gate is")
        L.append("  drawn for them. Expected unless one was meant to trade.")
    else:
        L.append("  none — every arriving contract is watched by an "
                 "allocation.")
    for sym, why in un["declared"]:
        L.append(f"  • {sym} — DECLARED NOT TRADED:")
        for ln in wrap(why, indent=8):
            L.append(f"        {ln}")

    L.append("=" * W)
    return "\n".join(L)


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Which portfolios exist, what each may trade, and which "
                    "strategy is allocated where. Reads configuration and "
                    "state; changes nothing.")
    ap.add_argument("--json", action="store_true",
                    help="print the collected configuration instead of the card")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snap = collect()
    if args.json:
        print(json.dumps(snap, indent=2, default=str))
    else:
        print(render(snap))
    # Non-zero only when the configuration could not be read. An empty
    # portfolio is a legitimate state, not a failure, so it does not set this.
    return 1 if snap["portfolios"]["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
