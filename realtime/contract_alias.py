"""
realtime.contract_alias - the micro / full-size contract table, in ONE place.

The execution tier trades MICROS. `config/portfolios.json` holds MNQ, MES,
MCL and MGC because that is what the accounts are sized for. Everything that
MEASURES the market - the lake, the theta_vol anchors, the regime daemon's
published state - is keyed on the FULL-SIZE parent, because that is where the
history is and where the in-sample anchor was pinned.

So a micro symbol has to resolve to its parent at every seam between the two,
and this module is the only table that says how. A micro and its parent quote
the SAME price series at the SAME tick size; only the multiplier differs, and
a multiplier appears nowhere in an ADX, an ATR or a quadrant boundary. That -
and nothing about the two being "related products" - is what makes the
substitution legitimate. `realtime.regime_daemon._verify_alias_tick_sizes`
reconciles every pair here against `backtest/specs.py` on every daemon
construction, rather than trusting this paragraph.

WHY THE TABLE LIVES HERE AND NOT IN THE DAEMON
==============================================
It used to live in `realtime/regime_daemon.py` as `THETA_ANCHOR_ALIAS`, and
`realtime/regime_reader.py` could not use it: the reader deliberately imports
neither pandas nor the writer, so that it still answers when the writer's
dependency stack is the thing that is broken. The result was a reader that
knew nothing about micros, so `get_current_regime("MNQ")` raised "no regime
published for 'MNQ'" while NQ sat in the file - and the dispatcher turned that
into `HOLD - no live regime reading` on every cycle, for a strategy the daemon
had already marked ACTIVE. A strategy stood down by a naming convention looks
exactly like a strategy stood down by the market.

This module therefore imports NOTHING but the standard library's typing, so
every tier can share the one table: the daemon (which anchors theta_vol on the
parent), the reader (which answers a micro's regime from the parent's record),
the dispatcher (which accepts a certification on NQ as covering MNQ),
`portfolio/config_loader.py` (which checks a basket covers a strategy) and
`master_live.py` (which reads a micro's bars from the parent's tape).

The daemon re-exports it as `THETA_ANCHOR_ALIAS`, which is still that module's
public name for it.

THE MAP IS ONE-DIRECTIONAL, AND DELIBERATELY SHORT
==================================================
Micro -> parent only. There is no parent -> micro entry, because the reverse is
not a function: a question about NQ is answered by NQ, never by routing through
a micro. Callers that need the other direction (the daemon publishing every
micro of a parent it classified) enumerate this table instead of inverting it.

Nothing but a genuine size variant of the SAME contract belongs here. Two
contracts on correlated markets are not aliases: substituting one for the other
would hand a strategy a regime reading, a bar series or a certification drawn
from a tape it was never measured on, and every log line would still read
correctly.
"""

from __future__ import annotations

# Micro -> full-size parent. Reconciled against `backtest/specs.py` tick sizes
# by `realtime.regime_daemon._verify_alias_tick_sizes`, which is called on
# every `MasterRegimeDaemon` construction and prints what it finds.
MICRO_TO_PARENT: dict[str, str] = {
    "MNQ": "NQ",     # Micro E-mini Nasdaq-100
    "MES": "ES",     # Micro E-mini S&P 500
    "MCL": "CL",     # Micro Crude Oil
    "MGC": "GC",     # Micro Gold
    "M2K": "RTY",    # Micro E-mini Russell 2000
    "MYM": "YM",     # Micro E-mini Dow
}


def normalize(symbol) -> str:
    """`' mnq '` -> `'MNQ'`. One spelling rule, applied at every seam."""
    return str(symbol or "").strip().upper()


def parent_of(symbol) -> str | None:
    """
    The full-size parent of a micro, or None when `symbol` is not a micro.

    None rather than the symbol itself, so a caller can tell "this is already
    a full-size contract" from "this micro resolves to itself" - the second is
    not a state this table can produce, and a caller that cannot distinguish
    them writes a fallback that silently swallows an unknown symbol.
    """
    return MICRO_TO_PARENT.get(normalize(symbol))


def resolve_parent(symbol) -> str:
    """
    `symbol` normalized, mapped to its parent when it is a micro.

    Use this where a full-size key is the only thing that can answer - a
    theta_vol anchor, a lake read. Where an exact reading might also exist,
    prefer `resolve_published`, which tries the symbol itself first.
    """
    sym = normalize(symbol)
    return MICRO_TO_PARENT.get(sym, sym)


def resolve_published(symbol, published) -> str:
    """
    Which key in `published` answers for `symbol`: the symbol, else its parent.

    THE EXACT SYMBOL WINS. If the daemon is ever pointed at a micro's own tape,
    that reading is the better answer and must not be displaced by its parent's
    - the alias exists to cover a symbol nobody measured, not to override one
    somebody did.

    Returns the NORMALIZED symbol unchanged when neither is present, so the
    caller raises its own error naming what it actually looked for. This
    function never decides that a lookup has failed.
    """
    sym = normalize(symbol)
    if sym in published:
        return sym
    parent = MICRO_TO_PARENT.get(sym)
    if parent is not None and parent in published:
        return parent
    return sym


def micros_of(symbol) -> list[str]:
    """Every micro that resolves to `symbol`, sorted. Empty for a micro."""
    sym = normalize(symbol)
    return sorted(m for m, full in MICRO_TO_PARENT.items() if full == sym)
