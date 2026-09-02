#!/usr/bin/env python3
"""
check_market_regime.py - what quadrant is each active contract in RIGHT NOW,
and which of the 46 registered strategies that permits to open a position.

Location:  ~/src/trading/scripts/check_market_regime.py

Why
---
`realtime/regime_daemon.py` classifies and PUBLISHES; `realtime/regime_reader.py`
reads what was published. Neither answers "classify the tape as it stands and
show me the working", which is the question an operator has when a strategy is
muted and the console will not say whether the market moved or the feed did.

This is a READ-ONLY inspection card. It classifies in-process and prints; it
never calls `update_state`/`refresh` and never touches
`data/live_regime_state.json`. Writing would race the live daemon, and a
diagnostic that can change the state every other tier reads is not a
diagnostic.

WHAT IT DOES NOT DO: DEFINE A REGIME
====================================
Not one threshold, indicator length or quadrant number is spelled here. The
quadrant comes from `MasterRegimeDaemon.calculate_regime`, which computes
Wilder's ADX(14)/ATR(14) through `mdlib.regimes` and compares them against the
PINNED in-sample theta_vol for that (symbol, timeframe). A second spelling of
the boundary in a second module is exactly how a cached quadrant and a live one
come to disagree with nothing raising, and this file is a reader of that engine
rather than a second implementation of it.

THE QUADRANTS ARE VOLATILITY x TREND. THEY ARE NOT DIRECTIONAL.
===============================================================
    Q1 High Volatility / Trending      Q2 High Volatility / Ranging
    Q3 Low Volatility / Trending       Q4 Low Volatility / Ranging
    Q0 UNDEFINED - indicator warm-up, and NOT a quadrant

There is no Bull or Bear quadrant, and this card does not invent one. Every
strategy in `strategies/approved_incubator/` was certified against a Q1..Q4
label drawn on that axis pair, and a card that relabelled them "Bull Trend /
High Vol" would be a SECOND quadrant vocabulary - the failure
`config/portfolios.json` schema 1.0.0 already shipped once, where every digit
named a different environment and a strategy was stood down in the regime it
was certified for while every log line read correctly.

Direction is still a fair thing to want, so it is reported SEPARATELY as
`trend_bias` (UP/DOWN/FLAT, from ROC) in its own column. It is a diagnostic. It
gates nothing, no strategy is certified against it, and it is never folded into
the quadrant.

SUPPORTING METRICS ARE DIAGNOSTICS, NOT INPUTS
==============================================
`norm_atr`, `roc_5`, `roc_15`, `volume_z` and `ribbon_slope` are computed here
and shown so the label can be argued with. NONE of them feeds the
classification - the quadrant would be identical if this whole block were
deleted. Where the tree already had a spelling for one, that spelling is
reused rather than re-derived:

    norm_atr    ATR(14) / close, the `atr_norm` of the promoted strategies'
                ML feature matrices - a raw ATR in points would say "2015"
                rather than "quiet" across a 16-year price range.
    volume_z    volume against its own 20-bar rolling mean and stdev, with the
                same 1e-8 guard those modules use for a dead-flat window.
    ribbon      EMA(50)/EMA(200), the ratio `keltner_trend_drift` calls
                `ribbon_ratio`; the slope is that fast leg's percent change
                over `RIBBON_SLOPE_LOOKBACK` bars.

Reads
-----
    Bars through `realtime.feed.resolve_feed` - the same feed `master_live.py`
    runs on, so the forming-bar drop, the micro alias, the lake's own resampler
    and the single NT8 `ts` conversion all apply here exactly as they do live.
    A card that read bars its own way could disagree with the loop about what
    the last closed bar even was.

    theta_vol anchors and the strategy registry through `MasterRegimeDaemon`.

Writes
------
    Nothing. stdout only.

Usage
-----
    python scripts/check_market_regime.py
    python scripts/check_market_regime.py --json
    python scripts/check_market_regime.py --symbols NQ ES --tf 15m 1h
    python scripts/check_market_regime.py --symbols MNQ MES MGC   # -> NQ/ES/GC
    python scripts/check_market_regime.py --feed lake     # rehearsal on history
    python scripts/check_market_regime.py --strategies    # per-strategy detail
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Before ANYTHING reads os.environ. Modules resolve their BT_* variables while
# being imported, so loading inside main() would be too late for those - and
# would appear to work here, which is the kind of difference nobody notices
# until one runner silently uses a default path. Rules live in mdlib/env.py.
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

import pandas as pd                                                # noqa: E402

from realtime.contract_alias import normalize, resolve_parent      # noqa: E402
from realtime.feed import FeedError, resolve_feed                  # noqa: E402
from realtime.regime_daemon import (                               # noqa: E402
    MasterRegimeDaemon,
    RegimeDaemonError,
    STATUS_ACTIVE,
    ThetaAnchorMissing,
    UNDEFINED_LABEL,
)

# --------------------------------------------------------------------------
# Supporting-metric periods. DIAGNOSTIC ONLY - see the module docstring. These
# are not regime parameters and changing one cannot move a quadrant boundary.
# --------------------------------------------------------------------------
VOLUME_Z_PERIOD = 20        # as `sma_momentum_crossover`'s feature matrix
EMA_FAST = 50               # as `keltner_trend_drift`'s ribbon
EMA_SLOW = 200
RIBBON_SLOPE_LOOKBACK = 10
ROC_FAST = 5
ROC_SLOW = 15

# `trend_bias` is UP/DOWN only outside this dead band, in percent of price.
# Without it every reading is UP or DOWN and the column says nothing: a tape
# that moved 0.002% in fifteen bars is flat, and calling it a direction is how
# a diagnostic column starts being read as a signal.
TREND_BIAS_DEADBAND_PCT = 0.05

# Enough for EMA(200) to be defined with room to spare. The quadrant itself
# needs only 2*ADX_LENGTH+1 = 29, so a short window still classifies - it is
# the ribbon that goes None first, and it says so rather than reporting a
# number computed over fewer bars than it names.
DEFAULT_LOOKBACK_BARS = 500


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# What to look at
# --------------------------------------------------------------------------
def registry_coverage(daemon: MasterRegimeDaemon
                      ) -> tuple[list[str], list[str]]:
    """
    The symbols and timeframes the REGISTERED strategies are certified on.

    Derived from the switchboard rather than hardcoded, because a hardcoded
    list is one promotion away from omitting a contract - and a symbol missing
    from this card is a symbol whose strategies silently report "no regime
    published", which reads exactly like a market that never entered its
    quadrant.
    """
    symbols, timeframes = set(), set()
    for entry in daemon.registry.values():
        if entry.get("symbol"):
            symbols.add(normalize(entry["symbol"]))
        if entry.get("timeframe"):
            timeframes.add(str(entry["timeframe"]))
    return sorted(symbols), sorted(timeframes, key=_tf_sort_key)


def _tf_sort_key(tf: str):
    """5m before 15m before 30m before 1h, rather than lexicographically."""
    try:
        from realtime.feed import tf_delta                         # noqa: PLC0415
        return (0, tf_delta(tf))
    except Exception:                                              # noqa: BLE001
        return (1, pd.Timedelta(0), tf)


# --------------------------------------------------------------------------
# Supporting metrics
# --------------------------------------------------------------------------
def supporting_metrics(bars: pd.DataFrame, atr_14: float | None
                       ) -> dict[str, Any]:
    """
    The diagnostic columns for one symbol's last closed bar.

    Every one is None when its own window is not filled. Reporting a ribbon
    slope computed over 40 bars in a column headed EMA(200) would be a number
    that looks like a measurement and is not one.
    """
    out: dict[str, Any] = {
        "close": None, "norm_atr": None, "roc_5": None, "roc_15": None,
        "volume_z": None, "ribbon_ratio": None, "ribbon_slope": None,
        "trend_bias": "n/a",
    }
    if bars is None or bars.empty:
        return out

    close = bars["close"].astype(float)
    last_close = float(close.iloc[-1])
    out["close"] = last_close

    # ATR as a FRACTION OF PRICE. `atr_14` comes from the engine, so this is a
    # rescale of the engine's number and never a second ATR.
    if atr_14 is not None and last_close:
        out["norm_atr"] = float(atr_14) / last_close

    for label, span in (("roc_5", ROC_FAST), ("roc_15", ROC_SLOW)):
        if len(close) > span:
            prior = float(close.iloc[-1 - span])
            if prior:
                out[label] = (last_close - prior) / prior * 100.0

    if "volume" in bars.columns:
        volume = bars["volume"].astype(float)
        if len(volume) >= VOLUME_Z_PERIOD:
            mean = volume.rolling(VOLUME_Z_PERIOD,
                                  min_periods=VOLUME_Z_PERIOD).mean()
            std = volume.rolling(VOLUME_Z_PERIOD,
                                 min_periods=VOLUME_Z_PERIOD).std()
            # The 1e-8 keeps a dead-flat volume window off a divide by zero,
            # exactly as the promoted feature matrices spell it.
            z = (volume - mean) / (std + 1e-8)
            if pd.notna(z.iloc[-1]):
                out["volume_z"] = float(z.iloc[-1])

    if len(close) >= EMA_SLOW:
        fast = close.ewm(span=EMA_FAST, adjust=False).mean()
        slow = close.ewm(span=EMA_SLOW, adjust=False).mean()
        if float(slow.iloc[-1]):
            out["ribbon_ratio"] = float(fast.iloc[-1]) / float(slow.iloc[-1])
        if len(fast) > RIBBON_SLOPE_LOOKBACK:
            prior = float(fast.iloc[-1 - RIBBON_SLOPE_LOOKBACK])
            if prior:
                out["ribbon_slope"] = (
                    (float(fast.iloc[-1]) - prior) / prior * 100.0)

    # DIRECTION, AND IT IS NOT THE QUADRANT. Read off the slow ROC, with a dead
    # band so a flat tape reads FLAT instead of borrowing a sign from noise.
    roc = out["roc_15"] if out["roc_15"] is not None else out["roc_5"]
    if roc is not None:
        if roc > TREND_BIAS_DEADBAND_PCT:
            out["trend_bias"] = "UP"
        elif roc < -TREND_BIAS_DEADBAND_PCT:
            out["trend_bias"] = "DOWN"
        else:
            out["trend_bias"] = "FLAT"
    return out


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------
def collect(symbols: list[str] | None = None,
            timeframes: list[str] | None = None,
            feed_mode: str = "auto",
            lookback_bars: int = DEFAULT_LOOKBACK_BARS,
            config_path: str | None = None,
            anchors_path: str | None = None,
            incubator_dir: str | None = None,
            feed: Any | None = None,
            daemon: MasterRegimeDaemon | None = None) -> dict[str, Any]:
    """
    One snapshot: every (symbol, timeframe) classified, plus the switchboard
    evaluated against those classifications.

    `feed` and `daemon` are injected by the tests. Nothing in production passes
    them; they exist so a case can run against deterministic bars without a
    lake, a mount or a broker.

    A FAILURE FOR ONE PAIR IS A ROW, NOT AN EXCEPTION. A missing theta anchor
    for 6J at 5m must not cost the operator NQ's reading - the card is most
    wanted when something is already broken.
    """
    snap: dict[str, Any] = {
        "collected_at": _utcnow(),
        "feed": None,
        "feed_error": None,
        "rows": [],
        "errors": [],
        "strategies": {},
        "registry_size": 0,
        "registry_conflicts": [],
    }

    kwargs: dict[str, Any] = {"strict_config": False}
    if config_path:
        kwargs["config_path"] = config_path
    if anchors_path:
        kwargs["anchors_path"] = anchors_path
    if incubator_dir:
        kwargs["incubator_dir"] = incubator_dir
    try:
        daemon = daemon or MasterRegimeDaemon(**kwargs)
    except (RegimeDaemonError, OSError, ValueError) as exc:
        snap["errors"].append(f"regime daemon unavailable: "
                              f"{type(exc).__name__}: {exc}")
        return snap

    snap["registry_size"] = len(daemon.registry)
    snap["registry_conflicts"] = list(daemon.registry_conflicts)

    reg_symbols, reg_tfs = registry_coverage(daemon)
    # DEFAULTS COME FROM THE REGISTRY, not from a list in this file. See
    # `registry_coverage`.
    wanted_symbols = [normalize(s) for s in (symbols or reg_symbols) if s]
    wanted_tfs = [str(t) for t in (timeframes or reg_tfs) if t]
    snap["symbols_requested"] = wanted_symbols
    snap["timeframes_requested"] = wanted_tfs
    snap["registry_symbols"] = reg_symbols
    snap["registry_timeframes"] = reg_tfs

    if not wanted_symbols or not wanted_tfs:
        snap["errors"].append(
            "nothing to classify: no symbols or timeframes were requested and "
            "the strategy registry named none either.")
        return snap

    try:
        feed = feed or resolve_feed(feed_mode)
        snap["feed"] = feed.describe()
    except (FeedError, ImportError, OSError) as exc:
        snap["feed_error"] = f"{type(exc).__name__}: {exc}"
        snap["errors"].append(f"no bar feed: {snap['feed_error']}")
        return snap

    # The synthetic state the switchboard is evaluated against. Built in
    # memory and NEVER written: `update_state` would race the live daemon, and
    # a read-only card that can move the file every other tier reads is not
    # read-only. Shaped exactly as `update_state` shapes it, so
    # `_published_record` resolves it by the same lookup.
    state: dict[str, Any] = {"symbols": {}, "by_timeframe": {}}

    for tf in wanted_tfs:
        try:
            bars_by_symbol, sources = feed.closed_bars(
                wanted_symbols, tf, lookback_bars=lookback_bars)
        except Exception as exc:                                   # noqa: BLE001
            snap["errors"].append(
                f"{tf}: bar load failed: {type(exc).__name__}: {exc}")
            continue

        for symbol in wanted_symbols:
            row: dict[str, Any] = {
                "symbol": symbol,
                "tf": tf,
                "source_symbol": sources.get(symbol),
                "aliased": resolve_parent(symbol) != symbol,
                "error": None,
            }
            bars = bars_by_symbol.get(symbol)
            if bars is None or bars.empty:
                # ABSENT, not empty. "no bars for this symbol" and "this symbol
                # has no signal" are different statements about the account.
                row["error"] = "no closed bars"
                snap["rows"].append(row)
                continue

            try:
                regime = daemon.calculate_regime(symbol, bars, tf=tf)
            except ThetaAnchorMissing as exc:
                row["error"] = f"no pinned theta_vol: {exc}"
                snap["rows"].append(row)
                continue
            except (RegimeDaemonError, ValueError, KeyError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                snap["rows"].append(row)
                continue

            row.update(regime)
            row.update(supporting_metrics(bars, regime.get("atr_14")))
            row["n_bars"] = int(len(bars))
            snap["rows"].append(row)

            record = dict(regime)
            record["written_at"] = snap["collected_at"]
            state["symbols"][symbol] = record
            state["by_timeframe"].setdefault(symbol, {})[tf] = record

    snap["state"] = state
    try:
        snap["strategies"] = daemon.evaluate_switchboard(state=state)
    except Exception as exc:                                       # noqa: BLE001
        snap["errors"].append(
            f"switchboard evaluation failed: {type(exc).__name__}: {exc}")
    return snap


def strategies_for(snap: dict[str, Any], symbol: str, tf: str
                   ) -> tuple[list[str], list[str]]:
    """
    `(active, suppressed)` strategy ids certified on this exact (symbol, tf).

    KEYED ON THE CERTIFIED PAIR, never on the symbol alone. theta_vol is per
    (symbol, TIMEFRAME) - NQ's is 7.90 at 15m and 11.33 at 30m - so a 15m
    reading is not an answer about a strategy certified at 1h, and pooling them
    into one "NQ" bucket would credit a strategy with a quadrant match drawn
    against a boundary it was never certified on.
    """
    active, suppressed = [], []
    parent = resolve_parent(symbol)
    for sid, status in sorted((snap.get("strategies") or {}).items()):
        if str(status.get("timeframe")) != str(tf):
            continue
        certified = normalize(status.get("symbol") or "")
        if certified not in (normalize(symbol), parent):
            continue
        (active if status.get("status") == STATUS_ACTIVE
         else suppressed).append(sid)
    return active, suppressed


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _fmt(value: Any, spec: str = ".2f", missing: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return missing
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """
    An ASCII table sized to its contents.

    Widths are computed rather than fixed: a hardcoded column is one long
    strategy id away from either wrapping or silently truncating, and a
    truncated symbol in a regime card is a wrong answer that looks like a
    right one.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    rule = "-+-".join("-" * w for w in widths)
    out = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), rule]
    out.extend(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row))
               for row in rows)
    return out


def render(snap: dict[str, Any], show_strategies: bool = False) -> str:
    lines = [
        "=" * 78,
        "CURRENT MARKET REGIME QUADRANT",
        "=" * 78,
        f"collected  {snap['collected_at']}",
        f"feed       {snap.get('feed') or snap.get('feed_error') or 'none'}",
        f"registry   {snap.get('registry_size', 0)} strategies registered",
        "",
        # STATED EVERY TIME, not in a footnote. The axes are what the whole
        # card means, and an operator reading "Trending" as "going up" is the
        # misreading that costs money.
        "Quadrants are VOLATILITY x TREND, not direction:",
        "  Q1 High-Vol/Trending   Q2 High-Vol/Ranging",
        "  Q3 Low-Vol/Trending    Q4 Low-Vol/Ranging   Q0 warm-up (NOT a quadrant)",
        "Trend Bias is a separate diagnostic from ROC. It gates nothing.",
        "",
    ]

    headers = ["Symbol", "TF", "Quadrant", "Regime", "ADX(14)", "NormATR%",
               "ROC5%", "ROC15%", "VolZ", "Ribbon%", "Bias", "Active",
               "Muted"]
    body: list[list[str]] = []
    for row in snap.get("rows", []):
        if row.get("error"):
            body.append([row["symbol"], row["tf"], "--", row["error"][:44],
                         "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
                         "-", "-"])
            continue
        active, suppressed = strategies_for(snap, row["symbol"], row["tf"])
        norm_atr = row.get("norm_atr")
        body.append([
            row["symbol"] + ("*" if row.get("aliased") else ""),
            row["tf"],
            str(row.get("quadrant") or "--"),
            str(row.get("regime_name") or row.get("regime") or "--"),
            _fmt(row.get("adx_14")),
            _fmt(None if norm_atr is None else norm_atr * 100.0, ".3f"),
            _fmt(row.get("roc_5"), "+.2f"),
            _fmt(row.get("roc_15"), "+.2f"),
            _fmt(row.get("volume_z"), "+.2f"),
            _fmt(row.get("ribbon_slope"), "+.2f"),
            str(row.get("trend_bias") or "n/a"),
            str(len(active)),
            str(len(suppressed)),
        ])

    if body:
        lines.extend(_table(headers, body))
    else:
        lines.append("no rows: nothing could be classified.")

    if any(r.get("aliased") for r in snap.get("rows", [])):
        lines += ["", "* micro contract; the tape and the pinned theta_vol are "
                      "the full-size parent's."]

    strategies = snap.get("strategies") or {}
    if strategies:
        active_n = sum(1 for s in strategies.values()
                       if s.get("status") == STATUS_ACTIVE)
        lines += ["",
                  f"SWITCHBOARD  {active_n} ACTIVE / "
                  f"{len(strategies) - active_n} MUTED  of {len(strategies)}"]
        # WHY, grouped. A count alone cannot distinguish "the market is not in
        # their quadrant" from "the feed published nothing", and those are an
        # operator's two completely different next actions.
        reasons: dict[str, int] = {}
        for status in strategies.values():
            if status.get("status") != STATUS_ACTIVE:
                reasons[str(status.get("reason"))] = \
                    reasons.get(str(status.get("reason")), 0) + 1
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"   muted: {reason:<28} {n}")

    if show_strategies and strategies:
        lines += ["", "PER STRATEGY"]
        detail = [[sid,
                   str(s.get("symbol") or "?"),
                   str(s.get("timeframe") or "?"),
                   str(s.get("optimal_regime") or "?"),
                   str(s.get("live_quadrant") or "--"),
                   str(s.get("status")),
                   str(s.get("reason"))]
                  for sid, s in sorted(strategies.items())]
        lines.extend("  " + line for line in _table(
            ["Strategy", "Sym", "TF", "Certified", "Live", "Status", "Reason"],
            detail))

    for err in snap.get("errors", []):
        lines.append(f"  ERROR {err}")
    for conflict in snap.get("registry_conflicts", []):
        lines.append(f"  REGISTRY {conflict}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Classify each active contract into the four-quadrant "
                    "regime standard and report which strategies that "
                    "permits to enter. Reads bars and configuration; writes "
                    "nothing, and never publishes to the live state file.")
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="contracts to classify. Default: every symbol the "
                         "registered strategies are certified on. Micros are "
                         "resolved to their full-size parent's tape.")
    ap.add_argument("--tf", "--timeframes", nargs="+", default=None,
                    dest="timeframes",
                    help="timeframes. Default: every timeframe the registered "
                         "strategies are certified at — a timeframe left out "
                         "leaves its strategies reporting no published regime.")
    ap.add_argument("--feed", default="auto",
                    help="auto|lake|nt8|live. Default auto: the broker feed "
                         "when it is publishing, else the lake, and it says "
                         "which.")
    ap.add_argument("--lookback-bars", type=int, default=DEFAULT_LOOKBACK_BARS,
                    help=f"bars per symbol (default {DEFAULT_LOOKBACK_BARS}; "
                         f"EMA({EMA_SLOW}) needs {EMA_SLOW})")
    ap.add_argument("--strategies", action="store_true",
                    help="add the per-strategy table")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable JSON instead of the card")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snap = collect(symbols=args.symbols,
                   timeframes=args.timeframes,
                   feed_mode=args.feed,
                   lookback_bars=args.lookback_bars)
    if args.json:
        # The switchboard verdict per row, resolved into the payload rather
        # than left for the consumer to recompute against `strategies_for`'s
        # (symbol, timeframe) rule - a second implementation of that join is a
        # second chance to pool two timeframes into one bucket.
        for row in snap.get("rows", []):
            if not row.get("error"):
                active, suppressed = strategies_for(snap, row["symbol"],
                                                    row["tf"])
                row["active_strategies"] = active
                row["suppressed_strategies"] = suppressed
        print(json.dumps(snap, indent=2, default=str))
    else:
        print(render(snap, show_strategies=args.strategies))

    # Non-zero only when nothing could be classified at all. A single symbol
    # without an anchor, or a quadrant that happens to suit no strategy, is a
    # real state of the market rather than a failure of this tool - exiting
    # non-zero for it would make the card useless in a monitoring loop.
    classified = [r for r in snap.get("rows", []) if not r.get("error")]
    return 0 if classified else 1


if __name__ == "__main__":
    sys.exit(main())
