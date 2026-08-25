"""
realtime.regime_daemon - the Master Regime Daemon.

The live half of the regime firewall. Stage 1 designated a home quadrant for a
strategy on ten years of history; a live supervisor has to know which quadrant
the market is in RIGHT NOW to decide whether that strategy is permitted to
trade. This module answers that question, runs the ML confirmation gate, and
caches the answer to `data/live_regime_state.json` for whatever executes.

THE QUADRANT STANDARD IS NOT RESTATED HERE
==========================================
Every threshold, every comparator and the four-way encoding are IMPORTED from
`mdlib/regimes.py`, and the schema labels come from
`portfolio/config_loader.py`. There is no second copy of the rule in this file,
because a live daemon and a backtest that disagree about what "Q1" means is the
one failure nothing downstream can detect: a strategy certified in High-Vol /
Trending would be stood down in the environment it was certified for and turned
loose in the one it never traded, with every log line reading correctly.

    Q1  HIGH_VOL_TREND            ADX(14) > 25   ATR(14) >  theta_vol
    Q2  HIGH_VOL_CHOP             ADX(14) <= 25  ATR(14) >  theta_vol
    Q3  LOW_VOL_TREND             ADX(14) > 25   ATR(14) <= theta_vol
    Q4  LOW_VOL_MEAN_REVERSION    ADX(14) <= 25  ATR(14) <= theta_vol
    Q0  UNDEFINED                 either indicator inside its 14-bar warm-up

**The ADX comparator is strictly `>`, not `>=`.** The specification for this
module was written as `ADX >= 25`; `mdlib/regimes.py` and
`backtest/profiler.py` have both always used `>`, so every quadrant in this
repository's caches, every Stage 1 designation and every Gate R verdict was
drawn on `>`. Changing the comparator here to match the wording would move the
boundary in the live daemon alone and leave the backtests behind it - so the
implementation follows the repository and this paragraph records the
difference. At ADX exactly 25.00000 the daemon says Ranging. The two rules
disagree on a measure-zero set of bars and the divergence they would cause is
permanent.

WHY theta_vol IS LOADED AND NEVER COMPUTED
==========================================
theta_vol is the median ATR(14) over the PINNED in-sample window
(2013-01-01..2022-12-31), which is exactly what `mdlib/regimes.py` writes into
each `{SYMBOL}_{TF}_regime.parquet`. This daemon reads that number and refuses
to trade without it.

It does NOT take a median of the recent bars it was handed, and that refusal is
the single most important line in the module. A median over a live window is a
property of the REQUEST, not of the contract: on a quiet morning every bar
looks high-volatility relative to its neighbours, so a rolling theta would
label a dead tape Q1 and hand a High-Vol/Trending strategy permission to trade
a market it was never certified in. The boundary that decided a certification
is the only boundary a live permission may be drawn against. A symbol with no
anchor raises - see `ThetaAnchorMissing`.

**An anchor is per (symbol, TIMEFRAME).** NQ's theta_vol is 7.90 at 15m and
11.33 at 30m - the same tape, a 43% different boundary - so an anchor applied
at the wrong timeframe silently relabels roughly a third of the session. The
timeframe is part of the key, part of the state file, and part of every error
message.

MICRO CONTRACTS
===============
The four portfolios trade micros (MNQ, MES, MCL, MGC) and the regime caches are
keyed on the full-size contracts (NQ, ES, CL, GC). `THETA_ANCHOR_ALIAS` - the
table in `realtime/contract_alias.py`, re-exported here under the name this
module has always used - maps each micro onto its full-size parent, because
they quote the SAME price series at the SAME tick size - only the multiplier
differs, and a multiplier does not appear anywhere in an ADX or an ATR. The tick sizes are RECONCILED against
`backtest/specs.py` at construction rather than asserted in a comment: if a
future contract change made them differ, the alias would be quietly wrong in
price units and every quadrant with it.

THE ML GATE FAILS LOUD, NOT OPEN AND NOT CLOSED
===============================================
`evaluate_ml_gate` returns True when NO model is registered for a strategy -
that is the documented pass-through, and it is what keeps a rule-based
Version A strategy trading. A model that IS registered but cannot be loaded, or
whose feature order cannot be established, RAISES instead: returning True there
would trade unfiltered while every log said "ML confirmed", and returning False
is indistinguishable from a model that looked at the features and vetoed. Both
are silent; the raise is not.

Feature ORDER is taken from the model's sidecar (`{model}.json`, key
`features`) and never from the iteration order of the caller's dict. A
classifier fed its columns in the wrong order does not fail - it returns a
confident probability computed from a matrix that means nothing.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ, so an operator
# opening a fresh terminal never has to `source .env` first. It runs at import
# time, above the imports below, because modules resolve their BT_* variables
# while being imported and loading the file inside main() would be too late for
# those - and would work here, which is the kind of difference nobody notices
# until one runner silently uses the default path. The rules live in ONE
# module: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mdlib.regimes import (  # noqa: E402
    ADX_LENGTH,
    ADX_TREND_THRESHOLD,
    ATR_LENGTH,
    DEFAULT_IS_END,
    DEFAULT_IS_START,
    QUADRANT_LABELS,
    QUADRANT_UNDEFINED,
    _wilder_frame,
    classify,
)
from mdlib import regimes as _regime_cache  # noqa: E402
from portfolio.config_loader import (  # noqa: E402
    CANONICAL_QUADRANT,
    DEFAULT_CONFIG_PATH,
    PortfolioConfigError,
    load_portfolio_config,
)
# The switchboard's signal vocabulary is BUILT from these rather than typed out
# again: `live.dispatcher` owns the actions that may reach the wire and
# `portfolio.portfolio_manager` owns the direction tokens a strategy emits. A
# token this module invented would be one nothing upstream sends, so its gate
# would never fire while every log line read as though it had.
from live.dispatcher import VALID_ACTIONS  # noqa: E402
from portfolio.portfolio_manager import FLAT, LONG, SHORT  # noqa: E402
# The reader owns path resolution, and the writer borrows it rather than
# spelling the rule twice: a daemon that resolved `data/live_regime_state.json`
# differently from its readers would publish to a file nothing reads, and both
# sides would log success. The dependency runs writer -> reader only; the
# reader imports nothing from here, so it keeps answering when this module's
# numeric stack is what is broken.
from realtime.contract_alias import (MICRO_TO_PARENT,   # noqa: E402
                                     micros_of,
                                     normalize,
                                     parent_of)
from realtime.regime_reader import resolve_state_path  # noqa: E402

# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------
DEFAULT_STATE_FILE = "data/live_regime_state.json"
DEFAULT_MODEL_DIR = "models/"

# The timeframe an anchor is assumed to describe when a caller names none.
# 15m is the pipeline's working timeframe and the one both existing caches were
# built at. It is a DEFAULT, not a fact about a symbol - see the module
# docstring on why the timeframe is part of the key.
DEFAULT_TF = "15m"

# The symbols the specification names. Used only to report which anchors are
# MISSING at construction; nothing restricts `calculate_regime` to this list.
REFERENCE_SYMBOLS = ("NQ", "ES", "CL", "GC")

# The schema label for each quadrant id, INVERTED from the config loader's
# table rather than written out again. `CANONICAL_QUADRANT` maps label -> id;
# this maps id -> label, and building it by inversion means a relabelled schema
# moves both directions at once.
QUADRANT_TO_LABEL: dict[str, str] = {
    quad: label for label, quad in CANONICAL_QUADRANT.items()
}
if sorted(QUADRANT_TO_LABEL) != ["Q1", "Q2", "Q3", "Q4"]:
    raise ImportError(
        f"portfolio.config_loader.CANONICAL_QUADRANT does not cover Q1..Q4 "
        f"exactly (got {sorted(QUADRANT_TO_LABEL)}). The live daemon cannot "
        f"name a quadrant the routing table does not recognise.")

# Quadrant 0 is not a quadrant. It is what `mdlib.regimes` stamps on a bar
# inside the ADX/ATR warm-up, and a consumer that reads it as a regime is
# reading a label nobody measured. It gets its own token so it can never be
# confused with Q4, which is where a naive `NaN > theta` comparison files it.
UNDEFINED_LABEL = "Q0_UNDEFINED_WARMUP"
QUADRANT_TO_LABEL[f"Q{QUADRANT_UNDEFINED}"] = UNDEFINED_LABEL

# pandas_ta's ADX needs two full Wilder windows before it produces a number:
# one for the directional movement, one for the smoothing of DX. Fewer bars
# than this is not an error - it is a warm-up, and it reports as Q0.
MIN_BARS_FOR_REGIME = 2 * ADX_LENGTH + 1

# Micro -> full-size parent for theta_vol lookup. The two quote the same price
# series at the same tick size; `_verify_alias_tick_sizes` checks that against
# `backtest/specs.py` on every construction rather than trusting this comment.
#
# The table itself lives in `realtime/contract_alias.py` and is re-exported
# here under the name this module has always used for it. It moved because
# `realtime/regime_reader.py` needs the same map and refuses to import this
# module (pandas, and the writer): a reader carrying its own copy would be
# free to disagree, and a disagreement about whether MNQ means NQ resolves a
# strategy into the wrong market with every log line reading correctly.
THETA_ANCHOR_ALIAS: dict[str, str] = MICRO_TO_PARENT

# Operator-supplied anchor overrides, read before the regime caches. The env
# var wins so an operator can pin a boundary without editing the repository.
ANCHORS_ENV_VAR = "BT_THETA_ANCHORS"
DEFAULT_ANCHORS_PATH = "config/theta_vol_anchors.json"

# The state file's own version. A downstream reader that finds a version it
# does not know must say so rather than parsing optimistically.
STATE_SCHEMA_VERSION = "1.0.0"

# Default probability a model must reach to confirm an entry, when its sidecar
# declares none. 0.50 is the classifier's own decision boundary - it is the
# threshold that makes `predict_proba >= t` equivalent to `predict`, which is
# the only defensible value to assume on behalf of a model whose author did
# not state one.
DEFAULT_ML_THRESHOLD = 0.50

MODEL_SUFFIXES = (".pkl", ".joblib", ".onnx")


# --------------------------------------------------------------------------
# The strategy registry and the switchboard vocabulary
# --------------------------------------------------------------------------
# Where a promoted strategy's CERTIFICATION lives. `config/portfolios.json`
# grants the permission (`active_strategies`) and carries the routing record;
# `meta.json` carries the quadrant Gate R actually certified against. Both are
# read, and they are RECONCILED - see `load_strategy_registry`.
DEFAULT_INCUBATOR_DIR = "strategies/approved_incubator"

STATUS_ACTIVE = "ACTIVE"
STATUS_MUTED = "MUTED"

DECISION_ALLOWED = "ALLOWED"
DECISION_BLOCKED = "BLOCKED"

KIND_ENTRY = "ENTRY"
KIND_EXIT = "EXIT"

# Why a strategy is muted. The status alone is not actionable - "MUTED" reads
# identically whether the market moved out of the strategy's quadrant, the feed
# died, or its certification could not be resolved, and those have completely
# different fixes.
MUTE_REGIME_MISMATCH = "regime_mismatch"
MUTE_NO_REGIME = "no_regime_published"
MUTE_WARMUP = "indicator_warmup"
MUTE_TIMEFRAME = "timeframe_mismatch"
MUTE_UNCERTIFIED = "certification_unresolved"

# FLATTEN is the wire action that closes a position. Pinned against
# `live.dispatcher` at import rather than assumed: if that vocabulary were ever
# renamed, the exit set below would quietly lose its only wire token and a
# muted strategy would be unable to flatten what it is already holding.
_WIRE_EXIT = {"FLATTEN"}
if not _WIRE_EXIT <= set(VALID_ACTIONS):
    raise ImportError(
        f"live.dispatcher.VALID_ACTIONS is {sorted(VALID_ACTIONS)} and does "
        f"not contain {sorted(_WIRE_EXIT)}. The switchboard builds its exit "
        f"vocabulary from that set; without the flatten token a muted "
        f"strategy could not close an open position.")

# ENTRY tokens open exposure; EXIT tokens reduce or close it. The switchboard
# mutes the first and never the second.
ENTRY_SIGNALS = frozenset(
    {str(a).upper() for a in VALID_ACTIONS if str(a).upper() not in _WIRE_EXIT}
    | {LONG.upper(), SHORT.upper()}
    | {"ENTRY", "ENTER", "ENTER_LONG", "ENTER_SHORT"})
EXIT_SIGNALS = frozenset(
    _WIRE_EXIT
    | {FLAT.upper()}
    | {"EXIT", "CLOSE", "EXIT_LONG", "EXIT_SHORT"})
if ENTRY_SIGNALS & EXIT_SIGNALS:
    raise ImportError(
        f"a signal token is both an entry and an exit: "
        f"{sorted(ENTRY_SIGNALS & EXIT_SIGNALS)}. The switchboard would gate "
        f"it as whichever branch it tested first.")


class RegimeDaemonError(RuntimeError):
    """The daemon cannot answer. Never a value that reads like an answer."""


class ThetaAnchorMissing(RegimeDaemonError):
    """
    No pinned in-sample theta_vol for a (symbol, timeframe).

    Its own type because it has its own fix - build the regime cache, or write
    the anchor into `config/theta_vol_anchors.json` - and because a caller may
    legitimately want to catch it and stand the symbol down while letting a
    genuine data error propagate.
    """


class MLGateError(RegimeDaemonError):
    """A registered model could not be evaluated. See the module docstring."""


class UnknownStrategy(RegimeDaemonError):
    """
    A strategy id the switchboard does not carry.

    Its own type because it is a CONFIGURATION fault, not a market state. It
    must never be caught into "muted": a caller that treated an unregistered id
    as muted would silently stop trading a strategy somebody forgot to add to
    `active_strategies`, and the console for that is a quiet market.
    """


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# theta_vol anchors
# --------------------------------------------------------------------------
def _anchors_path(explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(ANCHORS_ENV_VAR, "").strip()
    if env:
        return Path(env)
    return REPO_ROOT / DEFAULT_ANCHORS_PATH


def load_theta_anchors(anchors_path: str | Path | None = None,
                       symbols: tuple[str, ...] = REFERENCE_SYMBOLS,
                       timeframes: tuple[str, ...] = (DEFAULT_TF,),
                       cache_root: str | Path | None = None,
                       ) -> dict[tuple[str, str], dict[str, Any]]:
    """
    The pinned in-sample median ATR(14) per `(symbol, timeframe)`.

    Two sources, in this order, and every entry records WHICH one it came from:

      1. An operator anchors file - `$BT_THETA_ANCHORS`, else
         `config/theta_vol_anchors.json`. Shape:
         `{"NQ": {"15m": {"theta_vol": 7.8957, "is_start": "...",
                          "is_end": "..."}}}`.
         A bare number (`{"NQ": {"15m": 7.8957}}`) is accepted and recorded
         with its window UNKNOWN, because a theta with no window beside it is
         not a measurement - it is a number somebody typed.
      2. The regime cache's own provenance block,
         `mdlib.regimes.provenance(symbol, tf)["theta_vol"]`, which is the
         value every backtest of that (symbol, tf) was labelled with.

    The file wins because it is the deliberate act; the cache is what exists.
    A symbol present in neither is ABSENT from the result rather than defaulted
    to anything - see `ThetaAnchorMissing`.
    """
    anchors: dict[tuple[str, str], dict[str, Any]] = {}

    # -- source 2 first, so source 1 overwrites it and the override is visible.
    for symbol in symbols:
        for tf in timeframes:
            try:
                prov = _regime_cache.provenance(symbol, tf, root=cache_root)
            except Exception as exc:            # unreadable parquet, no mount
                prov = None
                _warn(f"regime cache for {symbol} {tf} is unreadable: "
                      f"{type(exc).__name__}: {exc}")
            if not prov or prov.get("theta_vol") is None:
                continue
            anchors[(symbol, tf)] = {
                "theta_vol": float(prov["theta_vol"]),
                "is_start": prov.get("is_start"),
                "is_end": prov.get("is_end"),
                "atr_length": prov.get("atr_length", ATR_LENGTH),
                "adx_length": prov.get("adx_length", ADX_LENGTH),
                "adx_trend_threshold": prov.get("adx_trend_threshold",
                                                ADX_TREND_THRESHOLD),
                "source": "regime_cache",
                "source_detail": str(
                    _regime_cache.cache_path(symbol, tf, root=cache_root)),
            }

    path = _anchors_path(anchors_path)
    if path.is_file():
        try:
            blob = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RegimeDaemonError(
                f"anchors file {path} could not be read ({exc}). Refusing to "
                f"fall back to the regime caches silently: an anchors file "
                f"that exists is a deliberate override, and ignoring a broken "
                f"one would label live bars with a boundary its author "
                f"replaced.") from None
        for symbol, per_tf in (blob or {}).items():
            if not isinstance(per_tf, dict):
                raise RegimeDaemonError(
                    f"{path}: {symbol!r} must map a TIMEFRAME to a theta_vol. "
                    f"An anchor with no timeframe cannot be applied - NQ's "
                    f"theta_vol is 7.90 at 15m and 11.33 at 30m.")
            for tf, entry in per_tf.items():
                if isinstance(entry, (int, float)) and not isinstance(entry, bool):
                    record = {"theta_vol": float(entry),
                              "is_start": None, "is_end": None}
                elif isinstance(entry, dict) and "theta_vol" in entry:
                    record = dict(entry)
                    record["theta_vol"] = float(entry["theta_vol"])
                else:
                    raise RegimeDaemonError(
                        f"{path}: {symbol}/{tf} is neither a number nor an "
                        f"object carrying 'theta_vol'.")
                if not (record["theta_vol"] > 0):
                    raise RegimeDaemonError(
                        f"{path}: {symbol}/{tf} theta_vol is "
                        f"{record['theta_vol']!r}. A non-positive volatility "
                        f"boundary labels every bar high-volatility.")
                record.setdefault("atr_length", ATR_LENGTH)
                record.setdefault("adx_length", ADX_LENGTH)
                record.setdefault("adx_trend_threshold", ADX_TREND_THRESHOLD)
                record["source"] = "anchors_file"
                record["source_detail"] = str(path)
                anchors[(str(symbol), str(tf))] = record

    return anchors


def _warn(message: str) -> None:
    """
    Announce on STDERR, never stdout.

    `mdlib.lake` sets this precedent and it holds here for the same reason:
    several tools in this repository parse a child process's stdout as data,
    and a warning printed there corrupts the payload instead of informing
    anybody.
    """
    print(f"[regime_daemon] {message}", file=sys.stderr, flush=True)


def _verify_alias_tick_sizes() -> list[str]:
    """
    Check every micro/full-size alias quotes the same tick size.

    Returns the disagreements as strings rather than raising, so a daemon can
    still classify the symbols that ARE consistent - but the list is surfaced
    on the daemon and printed, because an alias that is wrong in price units
    makes every ATR comparison for that contract wrong by the same factor.
    """
    try:
        from backtest.specs import SPECS
    except Exception as exc:                     # pragma: no cover - import path
        return [f"backtest.specs is unimportable ({type(exc).__name__}), so "
                f"the micro/full-size tick sizes were NOT reconciled"]

    problems = []
    for micro, full in THETA_ANCHOR_ALIAS.items():
        a, b = SPECS.get(micro), SPECS.get(full)
        if a is None or b is None:
            problems.append(f"{micro}->{full}: one of them has no ContractSpec")
            continue
        if float(a.tick_size) != float(b.tick_size):
            problems.append(
                f"{micro}->{full}: tick sizes differ ({a.tick_size} vs "
                f"{b.tick_size}). They do not quote the same price series, so "
                f"{full}'s theta_vol is not {micro}'s volatility boundary.")
    return problems


# --------------------------------------------------------------------------
# Module A - the strategy registry
# --------------------------------------------------------------------------
def _as_quadrant(value: Any) -> tuple[str | None, str | None]:
    """
    Resolve `Q2` or `Q2_HIGH_VOL_CHOP` to the canonical id, or explain why not.

    Returns `(quadrant, error)`; exactly one of the two is None. Both spellings
    are accepted because a Stage 1 handoff writes the id and a portfolio
    basket writes the schema label, and a resolver that took only one of them
    would report every strategy written the other way as uncertified.
    """
    text = str(value or "").strip()
    if not text:
        return None, None
    if text.upper() in QUADRANT_TO_LABEL and text.upper() != "Q0":
        return text.upper(), None
    if text in CANONICAL_QUADRANT:
        return CANONICAL_QUADRANT[text], None
    if text.upper() == "Q0" or text == UNDEFINED_LABEL:
        return None, (f"{text!r} is the warm-up sentinel, not a quadrant. "
                      f"Nothing may be certified in a regime nobody measured.")
    return None, (f"{text!r} is not a quadrant. Expected one of "
                  f"{sorted(q for q in QUADRANT_TO_LABEL if q != 'Q0')} or a "
                  f"schema label from {sorted(CANONICAL_QUADRANT)}.")


def _read_meta(incubator_dir: Path, strategy_id: str) -> tuple[dict, str | None]:
    """`meta.json` for one promoted strategy, or the reason there is none."""
    path = Path(incubator_dir) / strategy_id / "meta.json"
    if not path.is_file():
        return {}, (f"no {path} - the strategy is named in active_strategies "
                    f"but nothing in the incubator records what it was "
                    f"certified for")
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{path} is unreadable: {exc}"
    if not isinstance(blob, dict):
        return {}, f"{path} is not an object"
    return blob, None


def load_strategy_registry(config: dict,
                           incubator_dir: str | Path = DEFAULT_INCUBATOR_DIR,
                           ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Every strategy the routing table PERMITS, with the quadrant it was
    certified in. Returns `(registry, conflicts)`.

    Permission comes from `active_strategies` and nothing else. Being in
    `strategies/approved_incubator/` is not permission to trade - the directory
    is a shelf, the config is the grant - so a promoted strategy nobody
    activated is absent from the registry rather than muted in it.

    THE CERTIFIED QUADRANT IS READ TWICE AND RECONCILED
    ---------------------------------------------------
    `strategy_allocations[id].regime_filter` in `config/portfolios.json` and
    `certification.target_quadrant` in `meta.json` are two records of the same
    fact, written by the same promotion but at different times. When they
    DISAGREE the strategy is left with `optimal_regime = None`, which the
    switchboard mutes: picking either one would be picking at random which
    market a certified strategy is turned loose in, and the log line would read
    correctly whichever way it went. The timeframe is reconciled the same way
    and for a sharper reason - theta_vol is per (symbol, TIMEFRAME), so the
    wrong one silently relabels roughly a third of the session.

    Conflicts are RETURNED, never raised. A daemon that refused to start over
    one strategy's metadata would take the regime feed down for every other
    strategy on the account, and the config somebody is midway through fixing
    is exactly the one that must keep loading.
    """
    incubator_dir = Path(incubator_dir)
    registry: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []

    for pid in sorted((config.get("portfolios") or {})):
        portfolio = config["portfolios"][pid] or {}
        records = portfolio.get("strategy_allocations") or {}
        basket = portfolio.get("basket") or {}
        permitted = list((portfolio.get("derived") or {})
                         .get("canonical_quadrants") or [])
        for strategy_id in (portfolio.get("active_strategies") or []):
            strategy_id = str(strategy_id)
            record = records.get(strategy_id) or {}
            meta, meta_error = _read_meta(incubator_dir, strategy_id)
            problems: list[str] = []
            if meta_error:
                problems.append(meta_error)

            # -- the certified quadrant, from both records -----------------
            from_config, err = _as_quadrant(record.get("regime_filter"))
            if err:
                problems.append(f"{pid}.strategy_allocations: {err}")
            from_meta, err = _as_quadrant(
                (meta.get("certification") or {}).get("target_quadrant"))
            if err:
                problems.append(f"meta.json certification: {err}")

            if from_config and from_meta and from_config != from_meta:
                problems.append(
                    f"certified quadrant disagrees: portfolios.json says "
                    f"{from_config} and meta.json says {from_meta}. Refusing "
                    f"to choose - one of them decides which market this "
                    f"strategy is turned loose in.")
                optimal = None
            else:
                optimal = from_config or from_meta
            if optimal is None and not problems:
                problems.append(
                    "no certified quadrant in either portfolios.json "
                    "(regime_filter) or meta.json (certification."
                    "target_quadrant)")

            # -- symbol and timeframe, reconciled the same way -------------
            cfg_symbol = str(record.get("symbol") or "").strip().upper()
            meta_symbol = str(meta.get("symbol") or "").strip().upper()
            if cfg_symbol and meta_symbol and cfg_symbol != meta_symbol:
                problems.append(
                    f"certified symbol disagrees: portfolios.json says "
                    f"{cfg_symbol} and meta.json says {meta_symbol}")
                symbol = None
            else:
                symbol = cfg_symbol or meta_symbol or None
            if symbol is None and not any("symbol" in p for p in problems):
                problems.append("no certified symbol in either record")

            cfg_tf = str(record.get("timeframe") or "").strip()
            meta_tf = str(meta.get("timeframe") or "").strip()
            if cfg_tf and meta_tf and cfg_tf != meta_tf:
                problems.append(
                    f"certified timeframe disagrees: portfolios.json says "
                    f"{cfg_tf} and meta.json says {meta_tf}. theta_vol is per "
                    f"(symbol, TIMEFRAME), so the two name different "
                    f"volatility boundaries on the same tape.")
                timeframe = None
            else:
                timeframe = cfg_tf or meta_tf or None
            if timeframe is None and not any("timeframe" in p
                                             for p in problems):
                problems.append(
                    "no certified timeframe in either record, so no theta_vol "
                    "anchor can be selected")

            entry = {
                "strategy_id": strategy_id,
                "portfolio_id": pid,
                "portfolio_ids": [pid],
                "symbol": symbol,
                "timeframe": timeframe,
                "version": str(record.get("version")
                               or meta.get("version") or "") or None,
                "optimal_regime": optimal,
                "optimal_regime_label": (QUADRANT_TO_LABEL.get(optimal)
                                         if optimal else None),
                "optimal_regime_sources": {"portfolios.json": from_config,
                                           "meta.json": from_meta},
                "portfolio_quadrants": permitted,
                "basket_assets": [str(a).upper()
                                  for a in (basket.get("assets") or [])],
                "status_in_config": record.get("status"),
                "module_path": record.get("path") or meta.get("source"),
                "meta_path": str(incubator_dir / strategy_id / "meta.json"),
                "problems": problems,
            }

            if strategy_id in registry:
                # The same id activated on two accounts. Not automatically an
                # error, but if the two grants disagree about the certified
                # quadrant there is no single answer to "is it active", and
                # the switchboard publishes one status per id.
                first = registry[strategy_id]
                first["portfolio_ids"].append(pid)
                if first["optimal_regime"] != entry["optimal_regime"]:
                    note = (f"{strategy_id} is active on "
                            f"{first['portfolio_ids']} with different "
                            f"certified quadrants "
                            f"({first['optimal_regime']} vs "
                            f"{entry['optimal_regime']})")
                    first["problems"].append(note)
                    first["optimal_regime"] = None
                    first["optimal_regime_label"] = None
                continue
            registry[strategy_id] = entry

    for strategy_id, entry in sorted(registry.items()):
        for problem in entry["problems"]:
            conflicts.append(f"{strategy_id}: {problem}")

    return registry, conflicts


# --------------------------------------------------------------------------
# The daemon
# --------------------------------------------------------------------------
class MasterRegimeDaemon:
    """
    Classify live bars, gate entries through the ML confirmation models, and
    cache the result for whatever executes.

    Construction reads three things and holds none of them open: the portfolio
    routing table, the theta_vol anchors, and whatever models are in
    `ml_model_dir`. Nothing here reads bars - the caller owns the feed and
    hands frames to `calculate_regime`.
    """

    def __init__(self,
                 config_path: str = DEFAULT_CONFIG_PATH,
                 state_file: str = DEFAULT_STATE_FILE,
                 ml_model_dir: str = DEFAULT_MODEL_DIR,
                 anchors_path: str | Path | None = None,
                 default_tf: str = DEFAULT_TF,
                 symbols: tuple[str, ...] | None = None,
                 timeframes: tuple[str, ...] | None = None,
                 cache_root: str | Path | None = None,
                 incubator_dir: str | Path = DEFAULT_INCUBATOR_DIR,
                 strict_config: bool = True) -> None:
        self.config_path = str(config_path)
        self.incubator_dir = Path(incubator_dir)
        self.state_file = resolve_state_path(state_file)
        self.model_dir = Path(ml_model_dir)
        self.default_tf = str(default_tf)
        self.cache_root = cache_root

        # ---- portfolio routing table -------------------------------------
        # Loaded through the config loader, which reconciles `asset_metadata`
        # against `backtest/specs.py` and the regime labels against
        # `backtest/profiler.py` - so a daemon that constructed at all is one
        # whose quadrant vocabulary agrees with the pipeline's.
        try:
            self.config = load_portfolio_config(self.config_path)
            self.config_error: str | None = None
        except (PortfolioConfigError, FileNotFoundError, OSError) as exc:
            if strict_config:
                raise
            self.config = {}
            self.config_error = f"{type(exc).__name__}: {exc}"
            _warn(f"portfolio config not loaded: {self.config_error}")

        self.tradeable_symbols = tuple(sorted({
            asset
            for p in (self.config.get("portfolios") or {}).values()
            for asset in (p.get("basket", {}).get("assets") or [])
        }))

        # ---- the strategy registry (Module A) ----------------------------
        # Permission from `active_strategies`, certification from `meta.json`,
        # and the two reconciled. Conflicts are held, printed and published -
        # never raised, or one strategy's stale metadata would take the regime
        # feed down for every other strategy on the account.
        self.registry, self.registry_conflicts = load_strategy_registry(
            self.config, self.incubator_dir)
        for conflict in self.registry_conflicts:
            _warn(f"strategy registry: {conflict}")

        # ---- theta_vol anchors -------------------------------------------
        self.alias_problems = _verify_alias_tick_sizes()
        for problem in self.alias_problems:
            _warn(f"theta anchor alias: {problem}")

        # What the REGISTRY needs, which is the only set that actually has to
        # resolve. `default_tf` is a default and the baskets name contracts,
        # but a certification names a (symbol, TIMEFRAME) pair - and an anchor
        # loaded at the wrong timeframe is not a near miss: NQ's theta_vol is
        # 7.90 at 15m and 16.23 at 1h. Loading only the default timeframe left
        # a strategy certified at 1h permanently unclassifiable, reported as
        # "no regime published", which reads exactly like a quiet market.
        registry_pairs = tuple(
            (str(e["symbol"]), str(e["timeframe"]))
            for e in self.registry.values() if e["symbol"] and e["timeframe"])
        registry_symbols = tuple(
            dict.fromkeys(
                [sym for sym, _ in registry_pairs]
                + [THETA_ANCHOR_ALIAS[sym] for sym, _ in registry_pairs
                   if sym in THETA_ANCHOR_ALIAS]))

        wanted_symbols = tuple(symbols) if symbols else tuple(
            dict.fromkeys(REFERENCE_SYMBOLS
                          + tuple(THETA_ANCHOR_ALIAS[s]
                                  for s in self.tradeable_symbols
                                  if s in THETA_ANCHOR_ALIAS)
                          + tuple(s for s in self.tradeable_symbols
                                  if s not in THETA_ANCHOR_ALIAS)
                          + registry_symbols))
        wanted_tfs = tuple(timeframes) if timeframes else tuple(
            dict.fromkeys((self.default_tf,)
                          + tuple(tf for _, tf in registry_pairs)))

        self.theta_anchors = load_theta_anchors(
            anchors_path=anchors_path,
            symbols=wanted_symbols,
            timeframes=wanted_tfs,
            cache_root=cache_root,
        )
        # Scoped to the reference symbols that were actually REQUESTED. A
        # caller narrowing to one contract does not need to hear about the
        # three it did not ask for, and a warning that fires every time trains
        # a reader to skip the one that matters.
        # Two sources of "should have resolved": the reference symbols at the
        # timeframe this daemon defaults to, and every (symbol, timeframe) a
        # REGISTERED strategy was certified on. The cross product of all
        # requested symbols and all requested timeframes is not it - it would
        # report CL/1h missing because some other strategy trades 1h, and a
        # warning that fires every time trains a reader to skip the one that
        # matters.
        should_resolve = {(s, self.default_tf) for s in wanted_symbols
                          if s in REFERENCE_SYMBOLS}
        should_resolve |= {(THETA_ANCHOR_ALIAS.get(sym, sym), tf)
                           for sym, tf in registry_pairs}
        if symbols or timeframes:
            should_resolve = {(s, tf) for s in wanted_symbols
                              if s in REFERENCE_SYMBOLS for tf in wanted_tfs}
        self.missing_anchors = tuple(sorted(
            pair for pair in should_resolve
            if pair not in self.theta_anchors
            and (THETA_ANCHOR_ALIAS.get(pair[0], pair[0]), pair[1])
            not in self.theta_anchors))
        if self.missing_anchors:
            _warn(f"no pinned theta_vol for {list(self.missing_anchors)} - "
                  f"those symbols CANNOT be classified. Build the regime cache "
                  f"(scripts/precompute_regimes.py) or write the anchor into "
                  f"{_anchors_path(anchors_path)}.")

        # ---- ML confirmation models --------------------------------------
        self.models: dict[str, dict[str, Any]] = {}
        self.model_errors: dict[str, str] = {}
        self._discover_models()

        # ---- state -------------------------------------------------------
        # Read the existing file so an update for ONE symbol cannot delete the
        # others. A daemon restarted mid-session would otherwise publish a
        # state file holding whichever symbol ticked first, and a reader asking
        # about any other would be told the symbol is unknown.
        self.state: dict[str, Any] = self._read_state()

    # -- construction helpers ---------------------------------------------
    def _read_state(self) -> dict[str, Any]:
        if not self.state_file.is_file():
            return {"schema_version": STATE_SCHEMA_VERSION,
                    "updated_at": None,
                    "quadrant_standard": dict(QUADRANT_TO_LABEL),
                    "symbols": {}, "by_timeframe": {}, "strategies": {}}
        try:
            blob = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _warn(f"existing state file {self.state_file} is unreadable "
                  f"({exc}); starting from an empty state.")
            return {"schema_version": STATE_SCHEMA_VERSION,
                    "updated_at": None,
                    "quadrant_standard": dict(QUADRANT_TO_LABEL),
                    "symbols": {}, "by_timeframe": {}, "strategies": {}}
        blob.setdefault("schema_version", STATE_SCHEMA_VERSION)
        blob.setdefault("symbols", {})
        blob.setdefault("by_timeframe", {})
        blob.setdefault("strategies", {})
        blob["quadrant_standard"] = dict(QUADRANT_TO_LABEL)
        return blob

    def _discover_models(self) -> None:
        """
        Register every model file in `ml_model_dir`, keyed by STRATEGY ID.

        The strategy id is the filename stem, optionally suffixed with a
        symbol: `mean_rev_v3.pkl` registers for every symbol,
        `mean_rev_v3__NQ.pkl` for NQ alone and takes precedence. The double
        underscore is the separator because strategy ids in this repository
        contain single ones (`sma_momentum_crossover`).

        A file that fails to load is recorded in `model_errors` and NOT
        registered as a working model - `evaluate_ml_gate` then raises for that
        strategy rather than passing the entry through, because a broken model
        and an absent one must not produce the same verdict.
        """
        if not self.model_dir.is_dir():
            return
        for path in sorted(self.model_dir.iterdir()):
            if path.suffix.lower() not in MODEL_SUFFIXES:
                continue
            stem = path.stem
            strategy_id, _, symbol = stem.partition("__")
            key = f"{strategy_id}__{symbol}" if symbol else strategy_id
            try:
                record = self._load_model(path)
            except Exception as exc:
                self.model_errors[key] = f"{type(exc).__name__}: {exc}"
                _warn(f"model {path.name} failed to load: "
                      f"{type(exc).__name__}: {exc}")
                continue
            record.update({"strategy_id": strategy_id,
                           "symbol": symbol or None,
                           "path": str(path)})
            self.models[key] = record

    @staticmethod
    def _load_model(path: Path) -> dict[str, Any]:
        """
        Load one model plus its sidecar.

        The sidecar `{stem}.json` carries `threshold` and, critically,
        `features` - the column order the model was FITTED on. A dict of
        features has no order, so without that list the daemon cannot build the
        matrix and refuses to guess: a classifier handed its columns permuted
        returns a confident probability computed from a matrix that means
        nothing, and nothing raises.
        """
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            try:
                import onnxruntime                       # noqa: F401
            except ImportError:
                raise ImportError(
                    "onnxruntime is not installed, so an .onnx model cannot "
                    "be evaluated. It is not in requirements.txt - pin it "
                    "before shipping ONNX models, or export to joblib."
                ) from None
            import onnxruntime as ort
            model = ort.InferenceSession(str(path))
            kind = "onnx"
        else:
            import joblib
            model = joblib.load(path)
            kind = "sklearn"
            if not hasattr(model, "predict_proba"):
                raise TypeError(
                    f"{path.name} has no predict_proba. The gate compares a "
                    f"PROBABILITY against a threshold; a bare predict() gives "
                    f"a class, and thresholding a class is not gating.")

        sidecar = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if sidecar.is_file():
            meta = json.loads(sidecar.read_text())

        features = meta.get("features")
        if features is not None:
            features = [str(f) for f in features]
        threshold = float(meta.get("threshold", DEFAULT_ML_THRESHOLD))
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"{sidecar.name} declares threshold {threshold!r}; a "
                f"probability gate outside (0, 1] either passes everything or "
                f"nothing.")
        return {"model": model, "kind": kind, "features": features,
                "threshold": threshold, "sidecar": str(sidecar)
                if sidecar.is_file() else None}

    # -- regime -----------------------------------------------------------
    def theta_for(self, symbol: str, tf: str | None = None) -> dict[str, Any]:
        """
        The anchor record for a symbol, resolving a micro to its full-size
        parent. Raises `ThetaAnchorMissing` rather than returning a default.
        """
        tf = str(tf or self.default_tf)
        sym = str(symbol).strip().upper()
        for candidate in (sym, THETA_ANCHOR_ALIAS.get(sym)):
            if candidate and (candidate, tf) in self.theta_anchors:
                record = dict(self.theta_anchors[(candidate, tf)])
                record["anchor_symbol"] = candidate
                record["aliased"] = candidate != sym
                record["tf"] = tf
                return record
        known = sorted({f"{s}/{t}" for s, t in self.theta_anchors})
        raise ThetaAnchorMissing(
            f"no pinned in-sample theta_vol for {sym} at {tf}. Known anchors: "
            f"{known}. Refusing to take a median of the live bars instead: a "
            f"boundary drawn on the recent window is a property of the window, "
            f"and it would label a quiet tape high-volatility and permit a "
            f"strategy certified for a regime it is not in. Run "
            f"`scripts/precompute_regimes.py --symbols {sym} --tf {tf}` or add "
            f"the anchor to {_anchors_path(None)}.")

    def calculate_regime(self, symbol: str, recent_bars_df,
                         tf: str | None = None) -> dict[str, Any]:
        """
        Classify the LAST bar of `recent_bars_df` into the four-quadrant
        standard.

        `recent_bars_df` is one symbol's bars, oldest to newest, carrying
        `high`/`low`/`close` and either a `ts` column or a DatetimeIndex - the
        same shape `mdlib.lake` yields. Indicators are computed by
        `mdlib.regimes` itself (Wilder's ADX(14)/ATR(14) through the identical
        pandas_ta call the profiler makes), so a live label and a cached one
        are the same number rather than two good-faith implementations.

        Returns the specified dict, plus the fields a live consumer cannot
        work without:

          `quadrant`      the `Q1`..`Q4` id, `Q0` inside the warm-up
          `bar_ts`        the timestamp of the bar this describes. `updated_at`
                          is wall clock and says only when the daemon ran; a
                          frozen feed keeps `updated_at` fresh while `bar_ts`
                          stops, and telling those apart is the whole job.
          `theta_source`  where the boundary came from, and over which window
          `n_bars`        how many bars the indicators were computed over

        Too few bars is a WARM-UP, not an error: the result is `Q0`,
        `regime == "Q0_UNDEFINED_WARMUP"`, with `adx_14`/`atr_14` None. A
        consumer must treat that as "no regime", never as a quadrant - it is
        the same contract `mdlib.regimes` defines for quadrant 0.
        """
        sym = str(symbol).strip().upper()
        tf = str(tf or self.default_tf)
        anchor = self.theta_for(sym, tf)
        theta = float(anchor["theta_vol"])

        bars = _normalise_bars(recent_bars_df, sym)
        n_bars = len(bars)

        base = {
            "symbol": sym,
            "tf": tf,
            "theta_vol": theta,
            "theta_source": anchor.get("source"),
            "theta_window": [anchor.get("is_start"), anchor.get("is_end")],
            "theta_anchor_symbol": anchor.get("anchor_symbol"),
            "theta_aliased": bool(anchor.get("aliased")),
            "adx_trend_threshold": ADX_TREND_THRESHOLD,
            "n_bars": n_bars,
            "updated_at": _utcnow(),
        }

        if n_bars < MIN_BARS_FOR_REGIME:
            return {**base,
                    "regime": UNDEFINED_LABEL,
                    "quadrant": f"Q{QUADRANT_UNDEFINED}",
                    "adx_14": None,
                    "atr_14": None,
                    "is_high_vol": False,
                    "is_trending": False,
                    "bar_ts": (str(bars["ts"].iloc[-1]) if n_bars else None),
                    "note": (f"{n_bars} bars is inside the ADX({ADX_LENGTH}) "
                             f"warm-up; {MIN_BARS_FOR_REGIME} are needed "
                             f"before a quadrant exists.")}

        frame = _wilder_frame(bars)
        labelled = classify(frame, theta)
        last = labelled.iloc[-1]
        quad = int(last["regime_quadrant"])

        adx = float(last["adx_14"]) if pd.notna(last["adx_14"]) else None
        atr = float(last["atr_14"]) if pd.notna(last["atr_14"]) else None

        return {
            **base,
            "regime": QUADRANT_TO_LABEL[f"Q{quad}"],
            "quadrant": f"Q{quad}",
            "regime_name": QUADRANT_LABELS[quad],
            "adx_14": adx,
            "atr_14": atr,
            "is_high_vol": bool(last["is_high_vol"]),
            "is_trending": bool(last["is_trending"]),
            "bar_ts": str(labelled.index[-1]),
        }

    # -- ML gate ----------------------------------------------------------
    def has_model(self, strategy_id: str, symbol: str | None = None) -> bool:
        """True when a model is REGISTERED for this strategy (and symbol)."""
        return self._model_key(strategy_id, symbol) is not None

    def _model_key(self, strategy_id: str, symbol: str | None) -> str | None:
        """
        The registered model to use, most specific first.

        A per-contract model is a deliberate override of a shared one, so it is
        checked first. It also SHADOWS the shared model when it exists and is
        broken - `_candidate_keys` is what `evaluate_ml_gate` walks, and it
        stops at the first key that names either a working model or a failed
        one. Falling past a failed override onto the shared model would filter
        the symbol with the classifier its author replaced.
        """
        for key in self._candidate_keys(strategy_id, symbol):
            if key in self.models:
                return key
            if key in self.model_errors:
                return None
        return None

    def _candidate_keys(self, strategy_id: str,
                        symbol: str | None) -> list[str]:
        sym = str(symbol).strip().upper() if symbol else ""
        keys = [f"{strategy_id}__{sym}"] if sym else []
        keys.append(strategy_id)
        return keys

    def evaluate_ml_gate(self, strategy_id: str, symbol: str,
                         features: dict) -> bool:
        """
        True when the ML confirmation model permits this entry.

        THREE OUTCOMES, AND ONLY TWO OF THEM ARE BOOLEANS:

          * No model registered for `strategy_id` -> **True**. The documented
            pass-through. A rule-based strategy has no classifier and must keep
            trading.
          * A model registered -> `predict_proba(features)[1] >= threshold`.
          * A model registered but broken, or its feature order unknown, or a
            feature missing from `features` -> **raises `MLGateError`**.
            Returning True there trades unfiltered while the log reads "ML
            confirmed"; returning False is indistinguishable from a model that
            looked and vetoed. Both are silent, and a live filter that silently
            stopped filtering is exactly the failure this repository is built
            to avoid.

        `features` is a mapping. Its ORDER is ignored - the model's sidecar
        `features` list is what orders the matrix. See `_load_model`.
        """
        key = self._model_key(strategy_id, symbol)
        if key is None:
            # A model that FAILED to load is registered in `model_errors`, not
            # in `models`, so this is where a broken model is caught - without
            # this check it would look exactly like a strategy that has none.
            broken = [k for k in self._candidate_keys(strategy_id, symbol)
                      if k in self.model_errors]
            if broken:
                raise MLGateError(
                    f"a model for {strategy_id!r} exists but failed to load "
                    f"({'; '.join(self.model_errors[b] for b in broken)}). "
                    f"Refusing to pass the entry through as unfiltered: the "
                    f"run would be reported as ML-confirmed.")
            return True

        record = self.models[key]
        columns = record["features"]
        if not columns:
            raise MLGateError(
                f"model {record['path']} has no sidecar declaring `features`, "
                f"so the column ORDER it was fitted on is unknown. A "
                f"classifier fed permuted columns returns a confident "
                f"probability from a matrix that means nothing. Write "
                f"{Path(record['path']).with_suffix('.json').name} with "
                f'{{"features": [...], "threshold": 0.55}}.')

        missing = [c for c in columns if c not in features]
        if missing:
            raise MLGateError(
                f"model {record['path']} needs features {missing} which the "
                f"caller did not supply (got {sorted(features)}). Imputing "
                f"them - with a zero, a mean, or anything else - feeds the "
                f"classifier a bar that did not happen.")

        row = [[float(features[c]) for c in columns]]
        try:
            proba = self._probability(record, row)
        except Exception as exc:
            raise MLGateError(
                f"model {record['path']} raised during inference: "
                f"{type(exc).__name__}: {exc}") from None

        if proba is None or not (0.0 <= proba <= 1.0):
            raise MLGateError(
                f"model {record['path']} returned {proba!r}, which is not a "
                f"probability. Refusing to threshold it.")
        return bool(proba >= record["threshold"])

    @staticmethod
    def _probability(record: dict[str, Any], row: list[list[float]]):
        """P(class 1) for one row, for either backend."""
        model = record["model"]
        if record["kind"] == "onnx":
            import numpy as np
            name = model.get_inputs()[0].name
            outputs = model.run(None, {name: np.asarray(row, dtype="float32")})
            # ONNX classifiers conventionally emit [labels, probabilities];
            # the probabilities are either a dict per row or an array.
            probs = outputs[-1]
            first = probs[0]
            if isinstance(first, dict):
                return float(first.get(1, first.get("1")))
            return float(first[1])
        proba = model.predict_proba(row)
        return float(proba[0][1])

    # -- state ------------------------------------------------------------
    # -- Module C: the deterministic switchboard --------------------------
    @staticmethod
    def _symbol_candidates(symbol: str) -> list[str]:
        """
        Every contract whose published regime describes `symbol`'s tape.

        Both directions of the micro alias: a strategy certified on NQ is
        answered by a published MNQ reading and the reverse, because the two
        quote the same price series at the same tick size and only the
        multiplier differs - and a multiplier appears nowhere in an ADX or an
        ATR. Without this a strategy certified on NQ and routed to a basket of
        MNQ would find no reading and sit muted forever, which on a console
        looks exactly like a market that never entered its quadrant.
        """
        sym = normalize(symbol)
        out = [sym] if sym else []
        parent = parent_of(sym)
        if parent:
            out.append(parent)
        out.extend(micros_of(sym))
        return list(dict.fromkeys(out))

    def _published_record(self, symbol: str, tf: str,
                          state: dict[str, Any] | None = None
                          ) -> tuple[dict[str, Any] | None, str]:
        """
        The published regime covering `(symbol, tf)`, or `(None, why not)`.

        THE TIMEFRAME IS PART OF THE LOOKUP. A record published for NQ at 15m
        does not answer a question about NQ at 1h: theta_vol is 7.90 at 15m and
        16.23 at 1h on the same tape, so the two labels are drawn against
        boundaries that differ by more than a factor of two. Falling back to
        "whatever was published for this symbol" would answer with the wrong
        one and nothing would raise.
        """
        state = self.state if state is None else state
        by_tf = state.get("by_timeframe") or {}
        symbols = state.get("symbols") or {}
        candidates = self._symbol_candidates(symbol)
        for candidate in candidates:
            record = (by_tf.get(candidate) or {}).get(tf)
            if record:
                return dict(record), "published"
        # Documents written before `by_timeframe` existed, and any writer that
        # publishes only the flat map. Accepted only when the record NAMES the
        # timeframe asked for.
        for candidate in candidates:
            record = symbols.get(candidate)
            if record and str(record.get("tf")) == str(tf):
                return dict(record), "published"
        seen = sorted({f"{sym}/{rec.get('tf')}"
                       for sym, rec in symbols.items() if isinstance(rec, dict)})
        if any(c in symbols for c in candidates):
            return None, (f"nothing published for {'/'.join(candidates)} at "
                          f"{tf}; published: {seen}")
        return None, (f"no regime published for {'/'.join(candidates)}; "
                      f"published: {seen}")

    def strategy_status(self, strategy_id: str,
                        state: dict[str, Any] | None = None
                        ) -> dict[str, Any]:
        """
        Is this strategy permitted to OPEN a position right now, and why.

        `ACTIVE` when the live quadrant for its certified (symbol, timeframe)
        equals its certified quadrant. `MUTED` in every other case, including
        every case where the answer is not known - an unreadable, absent or
        warm-up regime is not permission, and defaulting it to ACTIVE would
        trade a market nobody has measured.

        `exits_allowed` is unconditionally True and is not a regime decision.
        A muted strategy still owns whatever it is already holding, and a gate
        that blocked its exit would strand inventory in the one environment its
        certification says it should not be in.
        """
        sid = str(strategy_id)
        entry = self.registry.get(sid)
        if entry is None:
            raise UnknownStrategy(
                f"{sid!r} is not in the switchboard. Registered: "
                f"{sorted(self.registry)}. An unregistered id is a config "
                f"fault, not a muted strategy - `active_strategies` in "
                f"{self.config_path} is what grants permission.")

        status = {
            "strategy_id": sid,
            "portfolio_id": entry["portfolio_id"],
            "portfolio_ids": list(entry["portfolio_ids"]),
            "symbol": entry["symbol"],
            "timeframe": entry["timeframe"],
            "version": entry["version"],
            "optimal_regime": entry["optimal_regime"],
            "optimal_regime_label": entry["optimal_regime_label"],
            "live_quadrant": None,
            "live_regime": None,
            "resolved_symbol": None,
            "bar_ts": None,
            "written_at": None,
            "status": STATUS_MUTED,
            "reason": MUTE_UNCERTIFIED,
            "detail": "",
            "entries_allowed": False,
            # NEVER conditional. See the docstring.
            "exits_allowed": True,
            "problems": list(entry["problems"]),
            "evaluated_at": _utcnow(),
        }

        if not entry["optimal_regime"] or not entry["symbol"] \
                or not entry["timeframe"]:
            status["detail"] = ("; ".join(entry["problems"])
                                or "certification incomplete")
            return status

        record, why = self._published_record(entry["symbol"],
                                             entry["timeframe"], state=state)
        if record is None:
            status["reason"] = MUTE_NO_REGIME
            status["detail"] = why
            return status

        status["resolved_symbol"] = record.get("symbol")
        status["live_quadrant"] = record.get("quadrant")
        status["live_regime"] = record.get("regime")
        status["bar_ts"] = record.get("bar_ts")
        status["written_at"] = record.get("written_at")
        status["adx_14"] = record.get("adx_14")
        status["atr_14"] = record.get("atr_14")
        status["theta_vol"] = record.get("theta_vol")

        if record.get("quadrant") in (None, "", f"Q{QUADRANT_UNDEFINED}") \
                or record.get("regime") == UNDEFINED_LABEL:
            status["reason"] = MUTE_WARMUP
            status["detail"] = (
                f"{record.get('symbol')} is inside the ADX({ADX_LENGTH})/"
                f"ATR({ATR_LENGTH}) warm-up. Q0 is not a quadrant, so it "
                f"matches nothing - including the strategy's own.")
            return status

        if str(record.get("tf")) != str(entry["timeframe"]):
            # Defence in depth: `_published_record` already keys on the
            # timeframe, so reaching this means a writer put a record under a
            # key that disagrees with its own `tf` field.
            status["reason"] = MUTE_TIMEFRAME
            status["detail"] = (
                f"the published record is at {record.get('tf')} and the "
                f"strategy is certified at {entry['timeframe']}; theta_vol is "
                f"per (symbol, TIMEFRAME) and the two boundaries differ.")
            return status

        if record.get("quadrant") == entry["optimal_regime"]:
            status["status"] = STATUS_ACTIVE
            status["entries_allowed"] = True
            status["reason"] = "regime_match"
            status["detail"] = (
                f"{record.get('symbol')} {record.get('tf')} is in "
                f"{record.get('quadrant')} ({record.get('regime')}), the "
                f"quadrant {sid} was certified in.")
            return status

        status["reason"] = MUTE_REGIME_MISMATCH
        status["detail"] = (
            f"{record.get('symbol')} {record.get('tf')} is in "
            f"{record.get('quadrant')} ({record.get('regime')}) and {sid} is "
            f"certified in {entry['optimal_regime']} "
            f"({entry['optimal_regime_label']}). New entries are blocked; "
            f"exits remain permitted so open inventory can still be managed.")
        return status

    def evaluate_switchboard(self, state: dict[str, Any] | None = None
                             ) -> dict[str, dict[str, Any]]:
        """Every registered strategy's status, against one state snapshot."""
        return {sid: self.strategy_status(sid, state=state)
                for sid in sorted(self.registry)}

    def is_strategy_active(self, strategy_id: str,
                           state: dict[str, Any] | None = None) -> bool:
        """True only when entries are permitted. Never a default."""
        return self.strategy_status(strategy_id,
                                    state=state)["status"] == STATUS_ACTIVE

    @staticmethod
    def classify_signal(signal: Any) -> str:
        """
        `KIND_ENTRY` or `KIND_EXIT` for a signal token, or RAISE.

        A token in neither vocabulary is refused rather than defaulted. Either
        default is wrong in a way nothing reports: treated as an entry it is
        blocked, and a strategy silently stops flattening; treated as an exit
        it is allowed, and a muted strategy opens a position in the one
        environment it was certified against.
        """
        token = str(signal or "").strip().upper()
        if token in EXIT_SIGNALS:
            return KIND_EXIT
        if token in ENTRY_SIGNALS:
            return KIND_ENTRY
        raise RegimeDaemonError(
            f"{signal!r} is not a signal this switchboard recognises. Entries: "
            f"{sorted(ENTRY_SIGNALS)}. Exits: {sorted(EXIT_SIGNALS)}. Refusing "
            f"to guess - one default blocks a flatten and the other opens a "
            f"position in a muted regime.")

    def gate_signal(self, strategy_id: str, signal: Any,
                    state: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        The switchboard verdict for one signal: `ALLOWED` or `BLOCKED`.

        Entries are allowed only while the strategy is ACTIVE. **Exits are
        allowed in every regime, including one where the strategy is muted and
        one where no regime has been published at all** - the muting exists to
        stop new exposure, and a position already open has to be closable from
        whatever state the daemon is in.
        """
        kind = self.classify_signal(signal)
        status = self.strategy_status(strategy_id, state=state)
        if kind == KIND_EXIT:
            decision, why = DECISION_ALLOWED, (
                "exit signals are permitted in every regime: muting blocks "
                "new exposure, it does not trap open inventory")
        elif status["status"] == STATUS_ACTIVE:
            decision, why = DECISION_ALLOWED, status["detail"]
        else:
            decision, why = DECISION_BLOCKED, status["detail"]
        return {"strategy_id": str(strategy_id),
                "signal": str(signal or "").strip().upper(),
                "kind": kind,
                "decision": decision,
                "status": status["status"],
                "reason": status["reason"],
                "detail": why,
                "live_quadrant": status["live_quadrant"],
                "optimal_regime": status["optimal_regime"],
                "evaluated_at": status["evaluated_at"]}

    # -- Module E: publication --------------------------------------------
    def _stamp(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """
        Finish a state document: the standards it was written under, and the
        switchboard computed FROM THAT DOCUMENT.

        The switchboard is derived from `candidate` rather than from
        `self.state`, so every status published in a file was drawn on the
        quadrants published in the same file. Computing it from memory would
        let a reader see a status that disagrees with the reading printed
        beside it, with both looking correct.
        """
        candidate["schema_version"] = STATE_SCHEMA_VERSION
        candidate["quadrant_standard"] = dict(QUADRANT_TO_LABEL)
        candidate["adx_trend_threshold"] = ADX_TREND_THRESHOLD
        candidate["in_sample_window"] = [DEFAULT_IS_START, DEFAULT_IS_END]
        # Published so the reader can classify a signal without importing this
        # module and without a second copy of the vocabulary.
        candidate["signal_vocabulary"] = {
            "entry": sorted(ENTRY_SIGNALS),
            "exit": sorted(EXIT_SIGNALS),
            "note": ("exit tokens are permitted in every regime; muting "
                     "blocks new exposure only"),
        }
        candidate["strategies"] = self.evaluate_switchboard(state=candidate)
        candidate["registry_conflicts"] = list(self.registry_conflicts)
        return candidate

    def publish_switchboard(self) -> dict[str, dict[str, Any]]:
        """
        Recompute and republish the switchboard against the regimes already on
        disk, without a new bar.

        Needed because a strategy's permission can change with no market
        movement at all - an operator adding an id to `active_strategies`, or a
        promotion rewriting a certified quadrant. Without this the switchboard
        would only ever refresh on the next tick of a symbol somebody happened
        to be watching.
        """
        candidate = dict(self.state)
        candidate.setdefault("symbols", {})
        candidate.setdefault("by_timeframe", {})
        candidate["updated_at"] = _utcnow()
        self._stamp(candidate)
        write_state_atomic(self.state_file, candidate)
        self.state = candidate
        return candidate["strategies"]

    def update_state(self, symbol: str, regime_data: dict) -> None:
        """
        Merge one symbol's regime into the cache and rewrite the whole file
        atomically.

        Atomic means temp file in the SAME directory then `os.replace`: a
        reader either sees the previous complete state or the new complete
        state, never a half-written one. `os.replace` is atomic only within a
        filesystem, which is why the temp file is not in /tmp.

        The whole file is rewritten rather than one key patched, because the
        alternative is a read-modify-write of a JSON document by two processes
        at once. Merging into `self.state` first means an update for NQ cannot
        drop ES.
        """
        sym = str(symbol).strip().upper()
        if not sym:
            raise RegimeDaemonError("update_state was given an empty symbol")
        if not isinstance(regime_data, dict):
            raise RegimeDaemonError(
                f"regime_data must be the dict calculate_regime returns, got "
                f"{type(regime_data).__name__}")

        entry = dict(regime_data)
        entry.setdefault("symbol", sym)
        entry.setdefault("updated_at", _utcnow())
        # `written_at` is the daemon's clock and `bar_ts` is the market's. A
        # reader needs both to tell "the tape is quiet" from "the feed died".
        entry["written_at"] = _utcnow()

        # Built as a CANDIDATE and committed to `self.state` only after the
        # write succeeds. Mutating in place first would leave a failed write
        # poisoning the daemon: the rejected entry would still be in memory,
        # so every later update - for any symbol - would serialise it again and
        # fail again, while the disk kept serving the last good document and
        # nothing looked wrong.
        candidate = dict(self.state)
        candidate["symbols"] = {**self.state.get("symbols", {}), sym: entry}
        # `symbols` holds the LATEST reading per contract, which is what a
        # caller asking "what is NQ doing" wants. `by_timeframe` holds one per
        # (symbol, timeframe), because they are different measurements: NQ's
        # theta_vol is 7.90 at 15m and 16.23 at 1h, so a 15m reading is not an
        # answer about a strategy certified at 1h. Keeping only the flat map
        # would let the last symbol to tick decide which boundary every
        # strategy on it was judged against.
        tf_key = str(entry.get("tf") or self.default_tf)
        by_tf = {s: dict(v) for s, v in
                 (self.state.get("by_timeframe") or {}).items()}
        by_tf.setdefault(sym, {})[tf_key] = entry
        candidate["by_timeframe"] = by_tf
        candidate["updated_at"] = entry["written_at"]
        self._stamp(candidate)

        write_state_atomic(self.state_file, candidate)
        self.state = candidate

    def refresh(self, symbol: str, recent_bars_df,
                tf: str | None = None) -> dict[str, Any]:
        """Classify and publish in one call. Returns what was written."""
        data = self.calculate_regime(symbol, recent_bars_df, tf=tf)
        self.update_state(symbol, data)
        return data

    # -- reporting --------------------------------------------------------
    def describe(self) -> str:
        lines = [f"MasterRegimeDaemon  config={self.config_path}  "
                 f"state={self.state_file}  models={self.model_dir}",
                 f"  quadrant standard: " + ", ".join(
                     f"{q}={QUADRANT_TO_LABEL[q]}"
                     for q in ("Q1", "Q2", "Q3", "Q4")),
                 f"  ADX trend threshold: > {ADX_TREND_THRESHOLD} "
                 f"(strictly greater - see the module docstring)"]
        lines.append("  theta_vol anchors:")
        if not self.theta_anchors:
            lines.append("    NONE. No symbol can be classified.")
        for (sym, tf), rec in sorted(self.theta_anchors.items()):
            lines.append(f"    {sym:<5} {tf:<4} theta={rec['theta_vol']:.6g} "
                         f"[{rec.get('is_start')}..{rec.get('is_end')}] "
                         f"via {rec.get('source')}")
        if self.missing_anchors:
            lines.append(f"  MISSING anchors: "
                         f"{', '.join(f'{s}/{t}' for s, t in self.missing_anchors)}")
        lines.append("  switchboard (active_strategies x certified quadrant):")
        if not self.registry:
            lines.append("    NONE registered. `active_strategies` in "
                         f"{self.config_path} grants the permission; the "
                         "incubator directory does not.")
        for sid, entry in sorted(self.registry.items()):
            lines.append(
                f"    {sid:<36} {str(entry['symbol']):<5} "
                f"{str(entry['timeframe']):<5} certified "
                f"{entry['optimal_regime'] or 'UNRESOLVED'} "
                f"({entry['optimal_regime_label'] or '-'}) "
                f"on {entry['portfolio_id']}")
        if self.registry_conflicts:
            lines.append("  REGISTRY CONFLICTS (these strategies stay muted):")
            lines.extend(f"    {c}" for c in self.registry_conflicts)
        lines.append(f"  models registered: {sorted(self.models) or 'none'}")
        if self.model_errors:
            lines.append(f"  models FAILED to load: {sorted(self.model_errors)}")
        if self.alias_problems:
            lines.extend(f"  ALIAS PROBLEM: {p}" for p in self.alias_problems)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# helpers shared with the reader
# --------------------------------------------------------------------------
def write_state_atomic(path: str | Path, payload: dict) -> Path:
    """
    Write `payload` as JSON, atomically, into `path`.

    Temp file in the destination directory, flush, `os.fsync`, `os.replace`.
    The fsync is what makes the guarantee survive a power loss rather than only
    a crash: without it `os.replace` can be durable while the bytes it points
    at are not.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent),
                               prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False,
                      default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        # A failed write must not leave a temp file that a directory listing
        # reads as state.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def _json_default(value):
    """
    Serialise what `json` cannot, and STRINGIFY NOTHING NUMERIC.

    `default=str` is the obvious shortcut and it is a silent corruption: a
    `numpy.float32` that reached the payload would be written as `"7.89"`, and
    a reader comparing an ATR against a threshold would compare a string.
    numpy scalars are unwrapped to their Python equivalent; a timestamp becomes
    its ISO form; anything else RAISES, because a value nobody taught this
    function to write is a value nobody has decided the shape of.
    """
    if hasattr(value, "item") and hasattr(value, "dtype"):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(
        f"live regime state cannot hold a {type(value).__name__} "
        f"({value!r}). Convert it at the point it is produced rather than "
        f"letting it be written as whatever str() makes of it.")


def _normalise_bars(df, symbol: str) -> pd.DataFrame:
    """
    Coerce a live bar frame into the shape `mdlib.regimes._wilder_frame` takes:
    a `ts` column plus high/low/close, sorted ascending.

    A DatetimeIndex is accepted and moved into `ts`. Sorting is CHECKED rather
    than performed - a live frame arriving out of order is a feed problem, and
    silently sorting it hides the gap that caused it while producing a plausible
    ADX.
    """
    if df is None:
        raise RegimeDaemonError(f"{symbol}: no bars")
    if not isinstance(df, pd.DataFrame):
        raise RegimeDaemonError(
            f"{symbol}: recent_bars_df must be a DataFrame, got "
            f"{type(df).__name__}")
    bars = df.copy()
    if "ts" not in bars.columns:
        if isinstance(bars.index, pd.DatetimeIndex):
            bars = bars.reset_index().rename(columns={bars.index.name or "index": "ts"})
        else:
            raise RegimeDaemonError(
                f"{symbol}: bars carry no 'ts' column and are not indexed by "
                f"timestamp.")
    missing = [c for c in ("high", "low", "close") if c not in bars.columns]
    if missing:
        raise RegimeDaemonError(f"{symbol}: bars are missing {missing}")
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    if len(bars) and not bars["ts"].is_monotonic_increasing:
        raise RegimeDaemonError(
            f"{symbol}: bars are not in ascending time order. A Wilder average "
            f"over unsorted bars is meaningless, and sorting them here would "
            f"hide the feed problem that produced them.")
    return bars.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """
    `python3 realtime/regime_daemon.py` - print what the daemon would use.

    Reads no bars and runs no classification: this is the "is it wired up"
    command. It says which anchors exist, which are missing, which models
    registered and which failed.
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    ap.add_argument("--models", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tf", default=DEFAULT_TF)
    ap.add_argument("--incubator-dir", default=DEFAULT_INCUBATOR_DIR)
    ap.add_argument(
        "--publish", action="store_true",
        help="classify each registered strategy's (symbol, timeframe) from "
             "the lake and publish the state file. Without it this command "
             "reads no bars and only reports what is wired up.")
    ap.add_argument("--bars", type=int, default=400,
                    help="how many trailing bars to classify over "
                         f"(minimum {MIN_BARS_FOR_REGIME})")
    ap.add_argument("--feed", default="auto",
                    choices=("auto", "nt8", "live", "lake"),
                    help="where --publish reads bars from. auto: the NT8 "
                         "broker feed when it is publishing, the lake "
                         "otherwise. A quadrant published from the lake "
                         "describes the newest bar somebody INGESTED, which "
                         "is not the market")
    args = ap.parse_args(argv)

    daemon = MasterRegimeDaemon(config_path=args.config,
                                state_file=args.state_file,
                                ml_model_dir=args.models,
                                default_tf=args.tf,
                                incubator_dir=args.incubator_dir,
                                strict_config=False)
    print(daemon.describe())

    if not args.publish:
        return 1 if daemon.missing_anchors or daemon.model_errors else 0

    # BARS ARE READ HERE AND NOWHERE ELSE IN THIS MODULE. `MasterRegimeDaemon`
    # owns no feed - the live loop hands it frames - and the import is inside
    # `main()` so that stays true: a class that could read a feed would grow a
    # code path where a live permission was drawn on historical bars.
    #
    # The feed is `realtime/feed.py`'s, the same seam `master_live.py` uses, so
    # the quadrant this publishes and the bars the loop trades on come from one
    # source. Published from the lake while the loop reads a live feed, the
    # switchboard would gate today's signals on a fortnight-old regime and
    # every log line would look right.
    from realtime.feed import resolve_feed                        # noqa: PLC0415

    feed = resolve_feed(args.feed)
    print(f"\nbar feed: {feed.describe()}")

    published, failed = 0, 0
    targets = sorted({(e["symbol"], e["timeframe"])
                      for e in daemon.registry.values()
                      if e["symbol"] and e["timeframe"]})
    if not targets:
        print("\nnothing to publish: no registered strategy names both a "
              "symbol and a timeframe.")
        return 1
    print()
    depth = max(int(args.bars), MIN_BARS_FOR_REGIME)
    for symbol, tf in targets:
        try:
            frames, _ = feed.closed_bars([symbol], tf, depth)
            bars = frames.get(str(symbol).upper())
            if bars is None or not len(bars):
                raise RegimeDaemonError(f"the feed returned no {tf} bars")
            tail = bars.tail(depth)
            data = daemon.refresh(symbol, tail, tf=tf)
        except Exception as exc:
            failed += 1
            print(f"  {symbol} {tf}: FAILED - {type(exc).__name__}: {exc}")
            continue
        published += 1
        print(f"  {symbol} {tf}: {data['quadrant']} ({data['regime']})  "
              f"ADX={data['adx_14']:.2f}  ATR={data['atr_14']:.4f}  "
              f"theta={data['theta_vol']:.4f}  bar={data['bar_ts']}")

    # Republish the switchboard even when nothing classified, so the file
    # always carries a status for every registered strategy rather than an
    # absence a reader would have to interpret.
    for sid, st in sorted(daemon.publish_switchboard().items()):
        print(f"  {sid}: {st['status']} ({st['reason']}) - entries "
              f"{'ALLOWED' if st['entries_allowed'] else 'BLOCKED'}, exits "
              f"{'ALLOWED' if st['exits_allowed'] else 'BLOCKED'}")
    print(f"\npublished {published} reading(s) to {daemon.state_file}"
          + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
