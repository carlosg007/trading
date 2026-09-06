#!/usr/bin/env python3
"""
promote.py - Move a strategy from experimental to the approved incubator.

Location:  ~/src/trading/backtest/promote.py

    python3 backtest/promote.py --strat sma_crossover --version A \
        --source strategies/experimental/sma_crossover.py \
        --metrics /mnt/backtest/artifacts/sma_crossover_20260815_120000/dual_metrics.json

Writes `strategies/approved_incubator/<strat>/`:

    strat.py         the promoted signal logic
    baseline.py      version B only - the rule-based module strat.py filters
    meta.json        what this is, which version, and the metrics it was
                     promoted on
    dual_metrics.json  version B/A: the full snapshot, copied verbatim

and commits that directory.

Being in the incubator is not permission to trade
-------------------------------------------------
It records that a decision was made, on numbers that are written down. The
gates in `docs/STRATEGY_DEVELOPMENT.md` and the 3-year holdout are what grant
deployment, and this script refuses to pretend otherwise: promoting on an
audit that did not pass every gate requires `--force`, and the reason is
recorded in `meta.json` where the next reader will see it.

Version A promotes the source module byte for byte, and records its SHA-256.
The file that gets promoted is then provably the file that was backtested -
a "cleaned up on the way through" strategy is a different strategy.

The parameters recorded are the RUN's, not the module's
-------------------------------------------------------
`meta.json` reads `params` out of the metrics snapshot when one is supplied,
layered over the module's DEFAULT_PARAMS and under any explicit `--params`.
That ordering matters as soon as `--scan` sweeps anything: the run's Sharpe
came from the winning grid cell, and recording the module's defaults beside it
would describe a strategy nobody backtested - with a stop distance that never
ran sitting next to metrics that assume one that did. `params_source` in
meta.json says which layer won.

The stop, the take-profit and the trailing flag are repeated in a `risk` block
so they can be read without knowing what a given strategy called its periods.
Three states are kept distinct there: `"NOT DECLARED"` (the strategy has no
such parameter), `null` (it has one and this run modelled it off - for
`tp_atr_mult` that means no take-profit at all), and a value.

Version B is the baseline plus the causal ML filter, which is a pipeline
rather than a file. `baseline.py` is the verbatim source; `strat.py` is a
generated wrapper that applies `agents.tier3_workers.apply_ml_signal_filter`
to the baseline's signals - the same call `run_dual_version_backtest` makes,
with the same threshold, so the promoted module reproduces the Version B that
was measured.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. This runs at import
# time, above the local imports below, because several modules resolve their
# BT_* variables while being imported (backtest.run's ARTIFACTS_ROOT) - loading
# the file inside main() would be too late for those and would work here, which
# is the kind of difference nobody notices until one runner silently uses the
# default path. The rules - the repository root derived from __file__ rather
# than the working directory, existing variables winning over the file, the
# CrossTrade credentials withheld from os.environ - live in ONE module rather
# than in a block copied into every runner: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # `python3 backtest/x.py` puts backtest/ on sys.path, not the repository
    # root, so mdlib is not importable until this runs.
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import discord_webhook, load_env                    # noqa: E402
from backtest.pipeline import (DOW_GATE_FILE, ML_THRESHOLD_DEFAULT,  # noqa: E402
                               STAGE45, base_strategy, pipeline_dir,
                               read_stage, strategy_id)

load_env()
# ---------------------------------------------------------------------------


import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
INCUBATOR = REPO_ROOT / "strategies" / "approved_incubator"

# The risk parameters lifted into their own `risk` block in meta.json. Named
# rather than discovered, so a strategy that invents `stop_mult` shows up as
# NOT DECLARED and prompts a question, instead of silently contributing an
# extra key nobody reads. `backtest/run.py` puts the same three on the
# leaderboard, and `strategies/experimental/*.py` spell them the same way.
RISK_KEYS = ("sl_atr_mult", "tp_atr_mult", "trailing")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# The Version B wrapper
# --------------------------------------------------------------------------
VERSION_B_TEMPLATE = '''"""
{strat} - Version B (ML-filtered), promoted {stamp}.

Baseline signals from `baseline.py` (SHA-256 {sha}), with the causal ML
filter applied on top. This is the pipeline, not a new idea: every entry here
is an entry Version A also produced, minus the ones the classifier expected to
lose.

The filter is an expanding-window walk-forward. For a candidate entry on bar
`s` it is fitted only on trades that had already CLOSED before `s`, so no
decision uses an outcome that did not exist when it was made. Refitting is
per completed trade, not per bar.

Promoted from: {source}
ML threshold : {threshold} - keep the entry when P(win) >= this.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from agents.tier3_workers import apply_ml_signal_filter, bind_ml_features
from backtest.engine import BacktestConfig

TIMEFRAME = {timeframe!r}
SYMBOLS = {symbols!r}
DEFAULT_PARAMS = {params!r}
ML_THRESHOLD = {threshold!r}

_BASELINE_PATH = Path(__file__).with_name("baseline.py")


def _baseline():
    """Load the promoted rule-based module sitting next to this file."""
    spec = importlib.util.spec_from_file_location(
        "{module}_baseline", _BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load baseline from {{_BASELINE_PATH}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def signal_fn(bars: pd.DataFrame, **params):
    """
    Version B: the baseline's signals, with losing entries suppressed.

    `bars` is ONE symbol's frame, oldest to newest - the engine calls this once
    per symbol. The symbol is read from the frame when the lake reader put it
    there and falls back to the promoted SYMBOLS entry, because it decides the
    contract multiplier, tick size and commission the filter's training labels
    are net of. Without it the classifier learns from gross outcomes and keeps
    trades that lose money after costs.

    Returns whichever shape the baseline returns: two masks for a long-only
    strategy, four for a bidirectional one. A bidirectional baseline gets one
    classifier per side, each trained on its own completed trades with its own
    P&L sign - see `apply_ml_signal_filter`. Filtering only the long side here
    would ship a promoted Version B whose shorts never met the filter it is
    named for.
    """
    threshold = params.pop("threshold", ML_THRESHOLD)
    cfg = params.pop("cfg", None) or BacktestConfig()

    merged = dict(DEFAULT_PARAMS)
    merged.update(params)

    base = _baseline()
    out = base.signal_fn(bars, **merged)
    if len(out) == 4:
        entries, exits, s_entries, s_exits = out
    else:
        entries, exits = out
        s_entries = s_exits = None

    symbol = None
    if "symbol" in bars.columns:
        present = pd.unique(pd.Series(bars["symbol"]).dropna())
        if len(present) > 1:
            raise ValueError(
                f"bars carry {{len(present)}} symbols. Pass one symbol's bars: "
                f"a rolling window over an interleaved frame averages across "
                f"contracts and the result looks fine.")
        if len(present) == 1:
            symbol = str(present[0])
    if symbol is None and SYMBOLS:
        symbol = SYMBOLS[0]

    def _b(s):
        return pd.Series(s).fillna(False).astype(bool)

    # The baseline's own feature matrix when it declares an `ml_features` hook,
    # None otherwise - and None is what selects the shared `causal_features`,
    # so a baseline without the hook is filtered by exactly the model it was
    # backtested under. Omitting this would ship a promoted Version B fitted on
    # different columns from the Version B whose metrics justified promoting
    # it, with nothing raising and no field on the page saying so. Resolved
    # once and handed to both sides, as `run_dual_version_backtest` does.
    _features_fn = bind_ml_features(base, merged)
    _features = _features_fn(bars) if _features_fn is not None else None

    entries, exits = apply_ml_signal_filter(
        bars, entries, exits, symbol=symbol, cfg=cfg, threshold=threshold,
        direction="long", features=_features)
    if s_entries is None:
        return _b(entries), _b(exits)

    s_entries, s_exits = apply_ml_signal_filter(
        bars, _b(s_entries), _b(s_exits), symbol=symbol, cfg=cfg,
        threshold=threshold, direction="short", features=_features)
    return _b(entries), _b(exits), _b(s_entries), _b(s_exits)


def make_signal_fn(**params):
    """Bind parameters for `agents.tier3_workers.load_strategy`."""
    def _bound(bars: pd.DataFrame):
        return signal_fn(bars, **params)

    return _bound
'''


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_source(path: Path) -> dict[str, Any]:
    """
    Read the strategy module's declared contract without importing it.

    `ast` rather than `import` because importing runs the module, and this
    function is the thing deciding whether the module should be trusted at all.
    TIMEFRAME, SYMBOLS and DEFAULT_PARAMS are read as literals; anything
    computed is skipped rather than guessed at.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    found: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                    "TIMEFRAME", "SYMBOLS", "DEFAULT_PARAMS"):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass

    defs = {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    return {
        "timeframe": found.get("TIMEFRAME"),
        "symbols": found.get("SYMBOLS") or [],
        "params": found.get("DEFAULT_PARAMS") or {},
        "has_signal_fn": "signal_fn" in defs,
        "has_make_signal_fn": "make_signal_fn" in defs,
        "functions": sorted(defs),
    }


def audit_notes(path: Path) -> list[str]:
    """
    Structural objections from the generated-code audit, as warnings.

    Reported, not enforced. That audit is the gate on MODEL-generated code,
    where an unexpected import is a sandbox escape. A hand-written module that
    is already in the repo has cleared review by a different route, and
    refusing to promote it over an import the allowlist does not know about
    would be the wrong tool doing the wrong job. Surfaced anyway, because if a
    strategy that reaches promotion is calling `open()`, somebody should see it.
    """
    try:
        from agents.tier3_workers import _audit_ast
    except Exception as e:                                      # noqa: BLE001
        return [f"safety audit unavailable: {type(e).__name__}: {e}"]
    try:
        return sorted(set(_audit_ast(ast.parse(path.read_text(encoding="utf-8")))))
    except SyntaxError as e:
        return [f"module does not parse: line {e.lineno}: {e.msg}"]


def load_metrics(path: Path | None, version: str) -> tuple[dict | None, dict | None, str]:
    """
    Pull the locked metrics and gate audit for one version out of a snapshot.

    Accepts the `dual_metrics.json` written next to the HTML reports, or any
    JSON with a `version_a`/`version_b` block, or a bare metrics dict.

    Returns `(metrics, gate_audit, status)` where status is the string recorded
    in meta.json. A missing file yields (None, None, "NOT RECORDED") rather
    than an invented number - meta.json says the metrics were never captured,
    which is a fact somebody can act on.

    A snapshot that WAS read reads `RECORDED · locked from <file>`. The leading
    token is what a reader scans for and the filename is what makes the record
    checkable - a bare `RECORDED` cannot be reconciled against the run it came
    from, and there are six `verify_<stamp>/` directories for this strategy
    alone.
    """
    if path is None:
        return None, None, "NOT RECORDED"
    if not path.exists():
        raise FileNotFoundError(f"metrics snapshot not found: {path}")

    blob = json.loads(path.read_text(encoding="utf-8"))
    key = "version_a" if version.upper() == "A" else "version_b"
    block = blob.get(key)
    if isinstance(block, dict) and "metrics" in block:
        return (block.get("metrics"), block.get("gate_audit"),
                f"RECORDED · locked from {path.name}")
    if "sharpe" in blob:
        return blob, blob.get("gate_audit"), f"RECORDED · locked from {path.name}"
    raise ValueError(
        f"{path} carries no `{key}` block and is not a metrics dict — cannot "
        f"tell which numbers belong to Version {version.upper()}")


def load_gate_certification(path: Path | None,
                            version: str) -> tuple[dict | None, dict]:
    """
    Read a Stage 3 `gate_audit_<SYMBOL>.json` and pull out one version's verdict.

    Returns `(gate_audit, provenance)`. The provenance block is what meta.json
    records: which file, which contract, which windows, and the file's SHA-256,
    so a promotion can be traced back to the exact certification run rather
    than to "a gate audit that said PASS at the time".

    This is the AUTHORITATIVE gate audit. The one inside `dual_metrics.json`
    comes from a single dual-version run, which can only ever evaluate Gate 1 -
    Gates 2 and 3 need a walk-forward, a bootstrap and the held-back years,
    which are separate runs. A promotion resting on the dual run's audit is
    resting on two gates that reported NOT EVALUATED, and NOT EVALUATED is not
    a pass. When both files are supplied this one wins and the disagreement is
    recorded rather than resolved silently.
    """
    if path is None:
        return None, {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"gate audit not found: {path}")

    blob = json.loads(path.read_text(encoding="utf-8"))
    stage = blob.get("stage")
    if stage is not None and int(stage) != 3:
        raise ValueError(
            f"{path} was written by stage {stage}, not stage 3. Only the "
            f"certification stage (backtest/audit_gates.py) produces a gate "
            f"verdict that may be promoted on.")

    ver = version.upper()
    versions = blob.get("versions") or {}
    block = versions.get(ver)
    if block is None:
        have = ", ".join(sorted(versions)) or "none"
        raise ValueError(
            f"{path} carries no certification for Version {ver} (has: {have}). "
            f"Re-run stage 3 with --ml to certify Version B.")

    audit = block.get("gate_audit")
    if not isinstance(audit, dict) or "status" not in audit:
        raise ValueError(
            f"{path} Version {ver} carries no gate_audit block. A file that "
            f"cannot state a verdict is not a certification.")

    prov = {
        "audit_file": path.as_posix(),
        "audit_sha256": sha256(path),
        "audit_symbol": blob.get("symbol"),
        # The certified PAIR and the quadrant it was certified inside, carried
        # so a later reader does not have to dig the timeframe out of the
        # audit's filename or re-derive Gate R's target. A promotion is one
        # contract at one timeframe in one quadrant, and the module's own
        # SYMBOLS/TIMEFRAME are its declarations rather than that pair - see
        # `certified_scope`.
        "audit_timeframe": blob.get("timeframe"),
        "target_quadrant": blob.get("target_quadrant"),
        "target_regime": blob.get("target_regime"),
        "audit_generated_utc": blob.get("generated_utc"),
        "in_sample": blob.get("in_sample"),
        "holdout": blob.get("holdout"),
        "params": blob.get("params"),
        "params_source": blob.get("params_source"),
        "variants_tested": blob.get("variants_tested"),
        "wfo": blob.get("wfo"),
        "monte_carlo": blob.get("monte_carlo"),
        "entry_filters": blob.get("entry_filters"),
        "status": audit.get("status"),
    }
    return audit, prov


def snapshot_params(metrics: dict | None) -> dict:
    """
    The parameters the promoted run ACTUALLY used, out of the metrics snapshot.

    This is the fix for a gap that mattered as soon as `--scan` started
    sweeping risk parameters. `inspect_source` reads the module's
    DEFAULT_PARAMS, which are the values somebody typed while writing the file
    - under `--scan` the run used a different set entirely, chosen from the
    grid. Recording the defaults in meta.json would state a stop multiplier the
    backtest never ran, next to a metrics block from the run that used the
    winning one. Nothing would raise, and the promoted strategy would be a
    strategy nobody measured.

    `run_dual_version_backtest` writes the bound set to `metrics["meta"]
    ["params"]`, which is the same dict the tear sheet's logic card was
    rendered from. A snapshot without that block returns `{}`, and the caller
    falls back to the module's declarations - the old behaviour.
    """
    meta = (metrics or {}).get("meta") or {}
    params = meta.get("params")
    return dict(params) if isinstance(params, dict) else {}


def certified_params(gate_audit: dict | None) -> dict:
    """
    The parameter set Stage 3 LOCKED and certified, out of the gate audit.

    Stage 3 takes Stage 2's winning cell, locks it, and runs it once over the
    holdout; `params` on the audit is that set. It is a stronger claim about
    what should be promoted than anything a lifecycle snapshot carries -
    Stage 4's window CONTAINS the holdout and reports `is_certification:
    false`, so its parameters describe a run, not a verdict.

    It is a PARTIAL set: only what Stage 2 swept is in it, so it layers over
    the module's defaults rather than replacing them. An audit that records
    none returns `{}` and the caller falls back to the snapshot - the
    pre-certification behaviour, unchanged.
    """
    params = (gate_audit or {}).get("params")
    return dict(params) if isinstance(params, dict) else {}


def _same_param(a: Any, b: Any) -> bool:
    """
    Two bound parameter values that mean the same thing.

    `1.5` and `1.5` arrive as int and float from JSON round-trips, and `None`
    is a legitimate value meaning "no take-profit modelled" - `scan.py` makes
    the same distinction with `_same_value` and for the same reason. Compared
    with `==` alone, `0` and `False` would also collapse, so bools are held
    apart from numbers explicitly.
    """
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def risk_settings(params: dict) -> dict[str, Any]:
    """
    The stop / target / trailing settings, called out of the parameter set.

    They are already in `meta["params"]`; this repeats them under a name a
    reader and the CrossTrade governance layer can find without knowing what a
    given strategy called its periods. The values are what actually ran, so
    this is the record of the risk the promoted numbers were earned under.

    Three states, deliberately distinguished, because collapsing any two of
    them would misdescribe the promoted strategy:

        absent from the dict  -> "NOT DECLARED"; the strategy has no such
                                 parameter and models that risk control not at
                                 all.
        present and None      -> None; the strategy HAS the parameter and this
                                 run swept it off. For `tp_atr_mult` that means
                                 no take-profit was modelled - the position
                                 runs to the stop, the signal exit or the bell.
        present with a value  -> the value.
    """
    out: dict[str, Any] = {}
    for key in RISK_KEYS:
        out[key] = params[key] if key in params else "NOT DECLARED"
    return out


#: What `meta["day_of_week_gate"]["status"]` says when Stage 4.5 never ran for
#: this pair. WRITTEN, never omitted, and deliberately not an empty list: an
#: absent key reads as a field nobody filled in, and `[]` reads as "the stage
#: looked at the week and found nothing to block". Those are different facts,
#: and `realtime/live_dispatcher.py` keeps them apart on the way back in.
DOW_NOT_EVALUATED = "NOT EVALUATED"


def load_dow_gate(strat: str, symbol: str | None, timeframe: str | None,
                  version: str | None = None,
                  path: Path | None = None,
                  out_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Stage 4.5's verdict for ONE promoted package, as the block for meta.json.

    `strategies/approved_incubator/<id>/meta.json` is the only file the live
    loop reads about a promoted package, so this is where the blocked weekday
    has to land - a verdict left in the pipeline directory is one the
    dispatcher would have to go looking for, keyed on a strategy id it would
    have to split apart first.

    THE FILE IS PER (SYMBOL, TIMEFRAME) AND THE VERDICT INSIDE IT IS PER
    VERSION, and both halves matter. There is no unsuffixed
    `dow_gate_<SYMBOL>.json`: `t3_braid_scalp_20260823` certified NQ at 15m,
    30m and 1h, and one would hold whichever timeframe ran last while all
    three promotions read it. And Version B is Version A's entries minus the
    ones a classifier expected to lose, so its trade list is a SUBSET and its
    weekday table is a different table - both versions of one pair can certify
    and be promoted as two packages, and one weekday handed to both would
    stand one of them down on a session measured on a strategy nobody
    deployed.

    Four outcomes, and the middle two are the reason this returns a dict
    rather than an integer:

        no file            `{"status": "NOT EVALUATED", ...}` - Stage 4.5 was
                           not run for this pair.
        no such version    also NOT EVALUATED, naming the versions that WERE
                           profiled. This is what a Version B package gets
                           when Stage 4.5 ran without `--ml`.
        version, no block  `blocked_weekdays: []` under `EVALUATED` - the
                           stage ran and every session cleared.
        version, blocked   `blocked_weekdays: [4]`.

    A file that cannot be read is `status: "UNREADABLE"` with the error, and
    blocks nothing. It is NOT raised: a certification that cleared Gate R must
    not be thrown away because a day-of-week artifact was truncated by a
    killed run, and a package promoted with the gate off and the reason on its
    own meta.json is recoverable by re-running Stage 4.5.
    """
    ver = str(version or "A").upper()
    if path is None:
        if not symbol or not timeframe:
            return {"status": DOW_NOT_EVALUATED, "blocked_weekdays": [],
                    "blocked_weekday": None, "version": ver, "source": None,
                    "reason": ("this promotion is not scoped to a (symbol, "
                               "timeframe) pair, so there is no Stage 4.5 "
                               "verdict to attach")}
        base = pipeline_dir(base_strategy(strat), out_dir)
        path = base / DOW_GATE_FILE.format(symbol=str(symbol).upper(),
                                           tf=str(timeframe).lower())
    path = Path(path)
    if not path.exists():
        return {"status": DOW_NOT_EVALUATED, "blocked_weekdays": [],
                "blocked_weekday": None, "version": ver, "source": str(path),
                "reason": (f"{path.name} does not exist. Stage 4.5 "
                           f"(backtest/dow_gate.py) was not run for this "
                           f"pair; no weekday is blocked and none was "
                           f"cleared.")}
    try:
        blob = read_stage(path, STAGE45, base_strategy(strat))
    except Exception as e:                                        # noqa: BLE001
        return {"status": "UNREADABLE", "blocked_weekdays": [],
                "blocked_weekday": None, "version": ver, "source": str(path),
                "reason": f"{type(e).__name__}: {e}"}

    profiled = sorted((blob.get("versions") or {}))
    entry = (blob.get("versions") or {}).get(ver)
    if entry is None:
        # NOT EVALUATED, never an empty block list. This is what a Version B
        # package gets when Stage 4.5 ran without --ml, and reporting it as
        # "every session cleared" would be a claim nobody measured.
        return {"status": DOW_NOT_EVALUATED, "blocked_weekdays": [],
                "blocked_weekday": None, "version": ver, "source": str(path),
                "versions_profiled": profiled,
                "reason": (f"{path.name} carries no Version {ver} verdict "
                           f"(profiled: {profiled or 'none'}). Stage 4.5 runs "
                           f"Version B only with --ml, and a weekday measured "
                           f"on Version A does not describe Version B's "
                           f"trades.")}

    day = entry.get("blocked_weekday")
    verdict = entry.get("verdict") or {}
    return {
        "status": "EVALUATED",
        "version": ver,
        "versions_profiled": profiled,
        # A LIST, even though the stage names at most one weekday. The live
        # handle reads a list, `exclude_days` is a list everywhere else in
        # this repository, and a scalar that some readers treat as a set is
        # how a second blocked day would silently be dropped later.
        "blocked_weekdays": ([] if day is None else [int(day)]),
        "blocked_weekday": (None if day is None else int(day)),
        "blocked_day": entry.get("blocked_day"),
        "blocked_day_name": entry.get("blocked_day_name"),
        # The worst session is carried even when it was NOT blocked. "Friday
        # was the weakest week and still made money" is the finding that
        # explains an empty block list, and without it the two look identical.
        "worst_weekday": verdict.get("worst_weekday"),
        "worst_day": verdict.get("worst_day"),
        "block_rule": verdict.get("block_rule"),
        "min_trades": verdict.get("min_trades"),
        "reason": verdict.get("reason"),
        "selected_in_sample": bool(blob.get("selected_in_sample", True)),
        "note": ("The weekday was chosen in-sample, on a window that spans "
                 "the Stage 3 holdout. It is a live-supervision instruction, "
                 "not evidence: no gate was scored on it and nothing was "
                 "pruned for it."),
        "source": str(path),
        "source_sha256": sha256(path),
    }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def promote(strat: str,
            version: str,
            source: Path,
            metrics_path: Path | None = None,
            audit_path: Path | None = None,
            symbol: str | None = None,
            timeframe: str | None = None,
            params: dict | None = None,
            threshold: float = 0.50,
            notes: str = "",
            variants_tested: int | None = None,
            force: bool = False,
            commit: bool = True,
            require_certification: bool = False,
            dow_gate: Path | None = None,
            out_dir: str | Path | None = None,
            incubator: Path = INCUBATOR) -> dict[str, Any]:
    """Write the promoted strategy directory and return what was written."""
    version = version.upper()
    if version not in ("A", "B"):
        raise ValueError(f"version must be A or B, got {version!r}")
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"source strategy not found: {source}")
    if source.suffix != ".py":
        raise ValueError(f"source must be a .py module, got {source.name}")

    info = inspect_source(source)
    if not info["has_signal_fn"]:
        raise ValueError(
            f"{source} defines no signal_fn (found: "
            f"{', '.join(info['functions']) or 'nothing'}). The engine calls "
            f"signal_fn(bars) -> (entries, exits); a module without it cannot "
            f"be run, let alone promoted.")

    metrics, gate_audit, metrics_status = load_metrics(metrics_path, version)

    # Stage 3's certification outranks whatever gate block travelled with the
    # metrics snapshot. The snapshot's audit comes from one dual-version run,
    # which cannot evaluate Gates 2 or 3 at all.
    cert_audit, certification = load_gate_certification(audit_path, version)
    snapshot_status = (gate_audit or {}).get("status", "NOT EVALUATED")
    if cert_audit is not None:
        certification["snapshot_gate_status"] = snapshot_status
        certification["agrees_with_snapshot"] = (
            snapshot_status == cert_audit.get("status")
            if gate_audit else None)
        gate_audit = cert_audit
    gate_status = (gate_audit or {}).get("status", "NOT EVALUATED")

    if gate_audit and gate_status != "PASS" and not force:
        where = (f" (from {Path(audit_path).name})" if audit_path
                 else " (from the metrics snapshot)")
        raise SystemExit(
            f"Version {version} gate audit is {gate_status}, not PASS"
            f"{where}. Promotion refused.\n"
            f"  Re-run with --force to promote anyway; the override is "
            f"recorded in meta.json.")
    # Defaults to False so the documented `bt-run` workflow - promote on a
    # dual_metrics.json, with meta.json recording NOT RECORDED - keeps working
    # unchanged. `--require-certification` turns it into a refusal for anyone
    # running the five-stage pipeline, where an uncertified promotion means a
    # stage was skipped rather than a snapshot mislaid.
    if gate_audit is None and require_certification and not force:
        raise SystemExit(
            f"No gate certification supplied. Promotion refused.\n"
            f"  Pass --audit-file <gate_audit_SYMBOL.json> from stage 3 "
            f"(backtest/audit_gates.py),\n"
            f"  or --force to promote an uncertified strategy — the override "
            f"is recorded in meta.json.")

    # The pair this promotion is for, resolved before the directory is named.
    # The certification's own symbol and timeframe are preferred over the
    # module's declarations for the reason `certified_scope` documents: a
    # module declares every contract it targets at the timeframe it prefers,
    # and a promotion is one contract at one timeframe.
    scope_symbol = symbol or (cert_audit or {}).get("symbol")
    scope_tf = timeframe or (cert_audit or {}).get("timeframe")
    symbols = ([scope_symbol] if scope_symbol
               else list(info["symbols"] or []))
    tf = scope_tf or info["timeframe"]

    # ONE DIRECTORY PER CERTIFIED PAIR **AND VERSION**. `strategy_id` returns
    # the bare strategy name when the pair is not known, which is the `bt-run`
    # workflow's id and is left exactly as it was - a dual-version run is not
    # scoped to a certified pair and there is nothing to name.
    #
    # The version segment is why this line changed. Version A and Version B of
    # one pair are two different strategies, and both can certify: on
    # 2026-08-29 NQ 1h passed Gate R on A at OOS PF 1.25 and on B at 1.22.
    # Without the segment both resolved to `<strat>_NQ_1h`, promote.py wrote A,
    # found the directory taken when it reached B, and exited 1. The Version B
    # package was lost with no gate having refused it - `promoted: 1, failed:
    # 1` in that run's auto_promotion block.
    #
    # Forward-only. Every id already in config/portfolios.json and on disk was
    # written without a version and keeps its exact spelling; `strategy_id`
    # appends nothing when no version is passed.
    # `--strat` IS ACCEPTED IN EITHER SPELLING, and normalising it here is what
    # makes that true. `strategy_id` concatenates, so handing it an id that is
    # already qualified produced
    # `keltner_trend_drift_20260901_6J_1h_VA_6J_1h_VA` - a second directory for
    # a pair that already had one, under a name nothing else resolves.
    # `base_strategy` splits on a KNOWN timeframe token, so a bare name passes
    # through unchanged and this is idempotent.
    strat = base_strategy(strat)

    # Collected here and folded into `audit_warnings` below, because that is
    # the list the CLI prints and `meta.json` records - a note about a bare id
    # has to reach the same place a reader already looks.
    pair_notes: list[str] = []

    # A CERTIFIED PROMOTION TAKES ITS PAIR FROM THE CERTIFICATION.
    # `scope_symbol`/`scope_tf` above read the audit BODY's top-level keys,
    # which `dual_ema_slope_scalp_20260831`'s audit did not carry - so
    # `strategy_id` fell back to the bare name and registered it beside the
    # `dual_ema_slope_scalp_20260831_6J_1h_VA` that already held the same
    # certification, with meta.json's top-level `symbol`/`timeframe` filled
    # from the MODULE's declarations (NQ/ES at 5m) while `certification.*`
    # correctly read 6J at 1h. One file, two answers, and the live registry
    # refused both for disagreeing with each other.
    #
    # `certified_scope` is the authority on that pair - it reads the
    # certification block and falls back to the audit's FILENAME - so consult
    # it before giving up. The module's declarations are still never used
    # here: its SYMBOLS is every contract it targets and its TIMEFRAME the one
    # it prefers, and a promotion is ONE pair.
    if cert_audit is not None and not (scope_symbol and scope_tf):
        resolved = certified_scope(certification, None, audit_path)
        from_cert_sym = resolved.get("symbol")
        from_cert_tf = resolved.get("timeframe")
        scope_symbol = scope_symbol or (
            from_cert_sym if from_cert_sym != NOT_RESOLVED else None)
        scope_tf = scope_tf or (
            from_cert_tf if from_cert_tf != NOT_RESOLVED else None)

    # NOT AN ERROR WHEN THE CERTIFICATION GENUINELY NAMES NO PAIR. An
    # unsuffixed `gate_audit_<SYMBOL>.json` from a dual-version `bt-run` is a
    # real artifact and promoting from it is a documented workflow, so raising
    # here would break a path that predates the bug above. The id stays bare,
    # as it always did, and the NOTE says so - which is the visible half the
    # duplicate registration never had.
    if cert_audit is not None and not (scope_symbol and scope_tf):
        missing = [n for n, v in (("symbol", scope_symbol),
                                  ("timeframe", scope_tf)) if not v]
        pair_notes.append(
            f"{strat} was promoted against a certification that records no "
            f"{' and no '.join(missing)}, so it is registered under the BARE "
            f"strategy name rather than a canonical <strategy>_<SYMBOL>_<TF> "
            f"id. If another promotion already holds this certification under "
            f"a canonical id, the two are separate registrations of one "
            f"configuration and the live registry will refuse both. Pass "
            f"--symbol/--timeframe, or point --audit-file at a per-pair "
            f"gate_audit_<SYMBOL>_<TF>.json.")

    promoted_id = strategy_id(strat, scope_symbol, scope_tf, version)
    dest = Path(incubator) / promoted_id
    dest.mkdir(parents=True, exist_ok=True)

    # FOUR layers, weakest first, and the third is the one a certification
    # rests on. Under `--scan` the run's parameters are the winning grid cell
    # rather than the module's DEFAULT_PARAMS, so promoting the defaults beside
    # that run's metrics would record a strategy nobody backtested; and the
    # LOCKED set on the gate audit is the one Stage 3 certified, which is a
    # stronger claim than the one a lifecycle snapshot happens to carry - Stage
    # 4's window contains the holdout and its snapshot is not a certification.
    # `--params` still wins outright, because an operator correcting the record
    # on purpose outranks a file.
    from_snapshot = snapshot_params(metrics)
    from_audit = certified_params(cert_audit)
    merged_params = dict(info["params"] or {})
    merged_params.update(from_snapshot)
    merged_params.update(from_audit)
    merged_params.update(params or {})

    # A snapshot and a certification that disagree about a shared parameter
    # describe two different strategies, and the metrics locked into meta.json
    # would then be measurements of the one that was NOT certified. Recorded
    # and printed rather than resolved silently - the certification wins, and
    # the fact that it had to is the finding.
    params_conflict = sorted(
        k for k in from_audit
        if k in from_snapshot and not _same_param(from_snapshot[k],
                                                  from_audit[k]))

    sources: list[str] = []
    if from_snapshot:
        sources.append(f"locked from {metrics_path.name}")
    if from_audit:
        sources.append(f"certified by {Path(audit_path).name}"
                       if audit_path else "certified by the gate audit")
    if params:
        sources.append("--params")
    params_source = " -> ".join(sources) or f"{source.name} DEFAULT_PARAMS"

    stamp = datetime.now(timezone.utc)
    written: list[Path] = []
    source_sha = sha256(source)

    if version == "A":
        # Byte-for-byte, so the promoted file hashes to the file that was
        # backtested. No header, no reformatting.
        strat_py = dest / "strat.py"
        shutil.copyfile(source, strat_py)
        written.append(strat_py)
        promoted_sha = sha256(strat_py)
    else:
        baseline_py = dest / "baseline.py"
        shutil.copyfile(source, baseline_py)
        written.append(baseline_py)
        strat_py = dest / "strat.py"
        strat_py.write_text(VERSION_B_TEMPLATE.format(
            strat=strat, stamp=stamp.strftime("%Y-%m-%d"), sha=source_sha,
            source=source.as_posix(), module=strat.replace("-", "_"),
            timeframe=tf, symbols=symbols, params=merged_params,
            threshold=threshold), encoding="utf-8")
        written.append(strat_py)
        promoted_sha = sha256(strat_py)

    if metrics_path and metrics_path.exists():
        snap = dest / "dual_metrics.json"
        shutil.copyfile(metrics_path, snap)
        written.append(snap)

    warnings = audit_notes(source) + pair_notes

    # Read AFTER `scope_symbol`/`tf` have been resolved, because the verdict
    # is keyed on the pair and resolving it from `--symbol`/`--timeframe`
    # alone would miss the ones that came off the certification.
    # `scope_tf` first, not `tf`: `tf` was bound before the certification's
    # own pair was resolved above and falls back to the MODULE's preferred
    # timeframe, which is not necessarily the pair Stage 4.5 profiled.
    dow_block = load_dow_gate(strat, scope_symbol, scope_tf or tf,
                              version=version, path=dow_gate, out_dir=out_dir)
    if dow_block["status"] == "UNREADABLE":
        # Recorded as a warning rather than raised. A certification that
        # cleared Gate R is not thrown away because a day-of-week artifact was
        # truncated; the package is promoted with the gate OFF and the reason
        # on its own meta.json, which is recoverable by re-running Stage 4.5.
        warnings.append(f"Stage 4.5 verdict could not be read "
                        f"({dow_block['reason']}); no weekday is blocked for "
                        f"this package.")

    meta = {
        # The PAIR's id: the directory name, the id in `active_strategies`,
        # and what `--strat` names on the Stage 5 card. `strategy` beside it is
        # the module this pair belongs to, and the two are equal for a bare
        # `bt-run` promotion.
        "name": promoted_id,
        "strategy": strat,
        "symbol": scope_symbol or (symbols[0] if len(symbols) == 1 else None),
        "version": version,
        "description": (f"Promoted Version {version} "
                        f"({'rule-based baseline' if version == 'A' else 'baseline + causal ML filter'})."),
        "symbols": symbols,
        "timeframe": tf,
        "params": merged_params,
        "params_source": params_source,
        # Where the locked metrics and the certification disagree about a
        # parameter. An empty list is written rather than omitted: "checked,
        # and they agree" is a different statement from "nobody looked".
        "params_conflict": params_conflict,
        # The stop, the target and the trailing flag the promoted numbers were
        # earned under, repeated where they can be found without knowing what
        # this strategy called its periods. `NOT DECLARED` means the strategy
        # has no such parameter; a literal null under `tp_atr_mult` means it
        # has one and this run modelled no take-profit. Those are different
        # facts about what a live account would be running.
        "risk": risk_settings(merged_params),
        # STAGE 4.5's BLOCKED WEEKDAY, and it is what the live loop acts on.
        # `realtime/live_dispatcher.py` reads this key and nothing else about
        # the calendar; `DOW_GATE_KEY` there and this spelling are the same
        # string, and a rename on one side alone turns the gate off silently -
        # the block would never be found, the strategy would trade the session
        # it was stood down from, and every log line would read correctly.
        #
        # Written on EVERY promotion, including one where Stage 4.5 never ran,
        # because an absent key reads as a field nobody filled in while
        # `status: "NOT EVALUATED"` says plainly that no weekday was profiled.
        "day_of_week_gate": dow_block,
        "source": source.as_posix(),
        "source_sha256": source_sha,
        "promoted_sha256": promoted_sha,
        "promoted_utc": stamp.isoformat(timespec="seconds"),
        "costs_included": bool((metrics or {}).get("meta", {}).get("costs_included", False)),
        "variants_tested": (
            variants_tested
            if variants_tested is not None
            else (metrics or {}).get("meta", {}).get("variants_tested")),
        "metrics_status": metrics_status,
        "metrics": _locked_metrics(metrics),
        "gate_audit_status": gate_status,
        "gate_audit": _gate_summary(gate_audit),
        # Where the verdict came from. `NOT CERTIFIED` is written rather than
        # omitted: an absent key reads as a field nobody filled in, and this
        # one is the difference between a strategy that cleared stage 3 and
        # one that was promoted past it.
        "certification": certification or "NOT CERTIFIED",
        "gates_overridden": bool(force and gate_status != "PASS"),
        "oos_status": "not_run" if gate_status != "PASS" else "passed_gate3",
        "audit_warnings": warnings,
        "notes": notes,
    }
    if version == "B":
        meta["ml_threshold"] = threshold
        meta["baseline"] = "baseline.py"

    meta_p = dest / "meta.json"
    meta_p.write_text(json.dumps(meta, indent=2, default=str) + "\n",
                      encoding="utf-8")
    written.append(meta_p)

    out = {"dir": dest, "files": written, "meta": meta,
           "strategy_id": promoted_id, "symbol": scope_symbol, "timeframe": tf,
           "warnings": warnings, "committed": False, "commit_output": ""}

    if commit:
        out.update(git_commit(dest, promoted_id, version, gate_status))
    return out


def _locked_metrics(metrics: dict | None) -> dict | None:
    """
    The headline numbers, flattened for meta.json.

    A subset, deliberately: meta.json is read by humans and by the dashboard,
    and the full snapshot is already sitting next to it in dual_metrics.json.
    """
    if not metrics:
        return None
    keys = ("sharpe", "sortino", "calmar", "profit_factor", "win_rate",
            "max_drawdown_pct", "total_return_pct", "annualized_return_pct",
            "trade_count", "total_pnl", "total_costs", "final_equity", "n_days")
    out = {}
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            out[k] = None            # NaN/inf are not JSON; None says "undefined"
        else:
            out[k] = v
    meta = metrics.get("meta") or {}
    for k in ("start", "end", "bars", "initial_capital"):
        if k in meta:
            out[k] = meta[k]
    return out


def _gate_summary(gate_audit: dict | None) -> dict | None:
    """Gate 1/2/3 status only. The criteria live in dual_metrics.json."""
    if not gate_audit:
        return None
    return {k: v.get("status") for k, v in (gate_audit.get("gates") or {}).items()}


# --------------------------------------------------------------------------
# Portfolio registration
# --------------------------------------------------------------------------
# `config/portfolios.json` is read and written here as PLAIN JSON, never
# through `portfolio.config_loader`. The dependency runs one way - `portfolio/`
# sits above `backtest/` and reads from it, and nothing in `backtest/` may
# import from there - and `backtest/discord_reporter.py` reads the same file
# the same way, for the same reason. The cost is that this module cannot run
# the loader's validation; what it CAN do is refuse to write anything it could
# not first parse, and re-parse what it wrote before it replaces the real file.
# See `_write_portfolio_config`.
PORTFOLIO_CONFIG = REPO_ROOT / "config" / "portfolios.json"

# Promotion registers onto the INCUBATOR track and only ever onto it.
# `approved_incubator/<strat>/` is a record that a version was chosen, and the
# incubator account is where a chosen version is evaluated on forward paper
# trades. Graduating Incubator -> Prop is a different decision, made on those
# forward trades rather than on a backtest, and it belongs to
# `portfolio/promotion_daemon.py` and `scripts/incubator_tracker.py`. A
# `--portfolio Prop-Odd` here would put a strategy on an evaluation account on
# the strength of a certification alone, which is the one step the forward
# incubation exists to sit between.
INCUBATOR_ACCOUNT_TYPE = "incubator_sim"

# The status stamped on a freshly registered allocation.
# `portfolio/promotion_daemon.py` is what moves it on, and it stamps
# `GRADUATED_PROP` on the ledger when it does.
ALLOCATION_STATUS = "incubating"

# One contract. A promoted strategy has never traded forward, so the opening
# allocation is the smallest position the account can hold; `--allocation`
# raises it deliberately. This is a DECLARATION, like everything else in that
# file - `portfolio/volatility_sizer.py` sizes from ATR against the portfolio's
# own risk budget and clamps, and nothing reads this number to place an order
# yet. It is recorded because the intended allocation is part of what was
# decided at promotion, and reconstructing it afterwards from a clamp shared by
# every strategy on the account is guessing.
DEFAULT_ALLOCATION = 1

# WHERE THE RICH RECORD GOES, AND WHY IT IS NOT IN `active_strategies`.
#
# `active_strategies` is a flat list of strategy-id STRINGS, and three separate
# consumers already read it that way:
#
#   portfolio.config_loader.get_portfolio_for_strategy  `id in active_strategies`
#   realtime.live_dispatcher._load_active_strategies    builds
#                                                       approved_incubator/<id>/
#                                                       out of each element
#   backtest.discord_reporter.portfolio_membership      lowercases each element
#                                                       and compares
#
# Putting a dict in that list breaks all three, and it breaks them QUIETLY:
# the router reports the strategy as unassigned and raises, the live loop looks
# for a directory named after a stringified dict, and the Stage 5 card prints
# the staging token for a strategy that is live on an account. None of those
# says "the schema changed".
#
# So the PERMISSION stays exactly where those three already look for it, and
# the PAYLOAD goes in a sibling object keyed by the same id. The two halves can
# drift apart, which is the one real cost of the split - so they are written
# together, here, by one function, and reconciled at load time by
# `portfolio.config_loader` (`allocation_reconciliation` on the loaded config).
ALLOCATIONS_KEY = "strategy_allocations"

# A field no artifact carried. Written rather than omitted, for the reason
# meta.json writes `NOT CERTIFIED`: an absent key reads as a field nobody
# filled in, and this one is the difference between a scope that was resolved
# and one that was never available.
NOT_RESOLVED = "NOT RESOLVED"


# THE INCUBATOR TRACK IS CREATED IF IT IS NOT THERE, AND NEVER INVENTED
# AROUND WHAT IS.
#
# `register_portfolio` used to raise on a routing table with no `portfolios`
# object, which meant a fresh checkout - or a `--portfolios` pointed at a path
# that does not exist yet - turned a completed, committed promotion into a
# printed error and an unallocated strategy. What it must NOT do instead is
# quietly write a plausible-looking account: `default_account_size`,
# `fixed_risk_budget_usd` and `max_trailing_drawdown_usd` are the numbers a
# live position is sized against, and a default that nobody chose is exactly
# the "rule nobody wrote down" that `portfolio/config_loader.py` refuses to
# apply when it declines to infer a portfolio from a strategy's name.
#
# So the bootstrap does both halves explicitly. It creates the two incubator
# portfolios with `active_strategies: []`, filled from
# `INCUBATOR_TRACK_TEMPLATE` below, and it PRINTS that the risk envelope is a
# placeholder. The template mirrors the shipped `config/portfolios.json`
# exactly, because `portfolio.config_loader` validates every field of it and a
# skeleton it refuses takes the live loop, the incubator tracker and the Stage
# 5 card down together - on the next run rather than on this one.
#
# The four-account partition is created whole. `REQUIRED_PORTFOLIOS` in
# `portfolio/config_loader.py` refuses a table missing any of the four ("a
# partition with a hole in it routes some strategy nowhere"), so writing only
# the incubator half would produce a file this module could read back and that
# loader could not.
# `execution_account` is the NinjaTrader account the portfolio's orders are
# SENT to, and it is not the portfolio id: NT8 prefixes a simulation account
# with `Sim`, so `Incubator-Odd` executes on `SimIncubator1`. It has to be
# spelled here as well as in `portfolio/incubator_recorder.py` because this
# module may not import that package - see the layering rule at the top of
# `.claude/rules/portfolio-routing.md`. The two are RECONCILED by
# `tests/test_portfolio_config.py` against the shipped table rather than
# trusted to stay equal: a bootstrap that wrote an account NinjaTrader does not
# have would have every order rejected, on a config that loads perfectly.
INCUBATOR_TRACK_TEMPLATE: dict[str, dict[str, Any]] = {
    "Incubator-Odd":  {"account_type": "incubator_sim",
                       "execution_account": "SimIncubator1",
                       "assets": ["MNQ", "MCL"],
                       "correlation_group": "Index_Energy_Uncorrelated",
                       "regime_quadrants": ["Q3_LOW_VOL_TREND",
                                            "Q4_LOW_VOL_MEAN_REVERSION"]},
    "Incubator-Even": {"account_type": "incubator_sim",
                       "execution_account": "SimIncubator2",
                       "assets": ["MES", "MGC"],
                       "correlation_group": "Index_Metals_Uncorrelated",
                       "regime_quadrants": ["Q1_HIGH_VOL_TREND",
                                            "Q2_HIGH_VOL_CHOP"]},
    # The evaluation rung, added 2026-09-03. Its own `account_type` because
    # the loader's orthogonality check reads that field as the LADDER STAGE:
    # sharing `prop_eval` with the funded book made it compare two stages of
    # one stream and refuse Eval-Odd beside Prop-Odd for both holding MNQ.
    "Eval-Odd":       {"account_type": "prop_evaluation",
                       "execution_account": "SimPropSim",
                       "assets": ["MNQ", "MCL"],
                       "correlation_group": "Index_Energy_Uncorrelated",
                       "regime_quadrants": ["Q3_LOW_VOL_TREND",
                                            "Q4_LOW_VOL_MEAN_REVERSION"]},
    "Eval-Even":      {"account_type": "prop_evaluation",
                       "execution_account": "Sim101",
                       "assets": ["MES", "MGC"],
                       "correlation_group": "Index_Metals_Uncorrelated",
                       "regime_quadrants": ["Q1_HIGH_VOL_TREND",
                                            "Q2_HIGH_VOL_CHOP"]},
    "Prop-Odd":       {"account_type": "prop_eval",
                       "execution_account": "SimProp1",
                       "assets": ["MNQ", "MCL"],
                       "correlation_group": "Index_Energy_Uncorrelated",
                       "regime_quadrants": ["Q3_LOW_VOL_TREND",
                                            "Q4_LOW_VOL_MEAN_REVERSION"]},
    "Prop-Even":      {"account_type": "prop_eval",
                       "execution_account": "SimProp2",
                       "assets": ["MES", "MGC"],
                       "correlation_group": "Index_Metals_Uncorrelated",
                       "regime_quadrants": ["Q1_HIGH_VOL_TREND",
                                            "Q2_HIGH_VOL_CHOP"]},
}

# The placeholder risk envelope, and the reason it is announced every time it
# is written rather than only the first time.
BOOTSTRAP_RISK_PROFILE: dict[str, Any] = {
    "fixed_risk_budget_usd": 250.0,
    "max_trailing_drawdown_usd": 2500.0,
    "max_forward_incubation_dd_pct": 0.4,
    "clamping": {"min_contracts": 1, "max_contracts": 5},
}
BOOTSTRAP_ACCOUNT_SIZE = 50000

# The asset metadata a bootstrapped basket refers to. READ from
# `backtest/specs.py` rather than restated: `portfolio.config_loader`
# reconciles this block against that module on every load and REFUSES the
# config when they disagree, because a wrong multiplier silently scales every
# P&L figure for that symbol. A hardcoded copy here would be a second source
# of truth that the loader exists to catch - so it is generated from the first.
BOOTSTRAP_SECTORS = {"MNQ": "Equity_Index", "MES": "Equity_Index",
                     "MCL": "Energy", "MGC": "Metals"}


def _bootstrap_asset_metadata() -> dict[str, dict[str, Any]]:
    """`asset_metadata` for the template's baskets, from `backtest/specs.py`."""
    from backtest.specs import SPECS
    out: dict[str, dict[str, Any]] = {}
    for symbol, sector in BOOTSTRAP_SECTORS.items():
        spec = SPECS.get(symbol)
        if spec is None:
            raise ValueError(
                f"backtest/specs.py declares no {symbol}, so a bootstrapped "
                f"routing table cannot state its point value. Create "
                f"{PORTFOLIO_CONFIG.name} by hand.")
        out[symbol] = {"point_value": float(spec.multiplier),
                       "tick_size": float(spec.tick_size),
                       "sector": sector}
    return out


def _template_portfolio(pid: str) -> dict[str, Any]:
    """One portfolio of the four-account partition, schema-valid and empty."""
    tpl = INCUBATOR_TRACK_TEMPLATE[pid]
    return {
        "portfolio_id": pid,
        "account_type": tpl["account_type"],
        "target_account": tpl["execution_account"],
        "default_account_size": BOOTSTRAP_ACCOUNT_SIZE,
        "risk_profile": json.loads(json.dumps(BOOTSTRAP_RISK_PROFILE)),
        "basket": {"assets": list(tpl["assets"]),
                   "correlation_group": tpl["correlation_group"],
                   "regime_quadrants": list(tpl["regime_quadrants"]),
                   "structures": ["MOMENTUM_TREND", "MEAN_REVERSION_SCALP"]},
        "active_strategies": [],
        ALLOCATIONS_KEY: {},
    }


def ensure_portfolio_groups(blob: dict | None) -> tuple[dict, list[str]]:
    """
    A routing table with both incubator groups present, and what was created.

    Returns `(blob, notes)`. Every note is PRINTED at promotion, because each
    one describes a number an operator has to confirm before the account it
    describes is traded: a portfolio created here carries
    `BOOTSTRAP_RISK_PROFILE`, which is the shipped envelope and not a decision
    anybody made about this account.

    Three repairs, all of them additive - nothing that is already there is
    rewritten, because a table somebody edited by hand outranks a template:

      * no file at all, or no `portfolios` object -> the whole four-account
        partition, each with `active_strategies: []`.
      * a missing group -> that group alone.
      * `active_strategies` missing on a group that exists -> an empty list.
        A group with no such key grants nothing and is the state a hand-edit
        leaves behind; refusing it here would fail a promotion over a key that
        means exactly what an empty list means.

    `active_strategies` present and NOT a list is NOT repaired. An empty list
    and a string are the same to `if name in active`, so replacing one would
    throw away a permission that is currently granted - `register_portfolio`
    raises on it, and that is the right outcome for a shape this module does
    not recognise.
    """
    notes: list[str] = []
    blob = dict(blob) if isinstance(blob, dict) else {}

    portfolios = blob.get("portfolios")
    if not isinstance(portfolios, dict) or not portfolios:
        portfolios = {}
        blob["portfolios"] = portfolios
        blob.setdefault("version", "1.1.0")
        blob.setdefault("base_currency", "USD")
        blob.setdefault("asset_metadata", _bootstrap_asset_metadata())
        for pid in INCUBATOR_TRACK_TEMPLATE:
            portfolios[pid] = _template_portfolio(pid)
        notes.append(
            f"created the four-account partition "
            f"({', '.join(INCUBATOR_TRACK_TEMPLATE)}), each with "
            f"active_strategies: []. The risk envelope on every one of them is "
            f"the SHIPPED DEFAULT ({BOOTSTRAP_RISK_PROFILE['fixed_risk_budget_usd']:.0f} "
            f"USD risk budget, "
            f"{BOOTSTRAP_RISK_PROFILE['max_trailing_drawdown_usd']:.0f} USD "
            f"trailing limit, {BOOTSTRAP_ACCOUNT_SIZE} account) and not a "
            f"decision anybody made about these accounts - confirm it before "
            f"anything trades against them.")
        return blob, notes

    for pid in INCUBATOR_TRACK_TEMPLATE:
        if pid not in portfolios:
            portfolios[pid] = _template_portfolio(pid)
            notes.append(
                f"created {pid} with active_strategies: [] - it was missing "
                f"from the routing table. Its risk envelope is the shipped "
                f"default, not a decision about this account.")
            continue
        block = portfolios[pid]
        if isinstance(block, dict) and "active_strategies" not in block:
            block["active_strategies"] = []
            notes.append(f"{pid} declared no active_strategies; initialised "
                         f"it to [].")
    return blob, notes


def _load_portfolio_config(path: Path) -> tuple[dict, list[str]]:
    """Read the routing table, creating what is missing. See above."""
    if not Path(path).exists():
        blob, notes = ensure_portfolio_groups(None)
        return blob, [f"{Path(path).name} did not exist"] + notes
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    return ensure_portfolio_groups(blob)


# The key a per-configuration record is deduplicated on.
#
# A promotion is ONE contract at ONE timeframe, and a campaign certifies
# several: `t3_braid_scalp_20260823` certified NQ at 15m, 30m and 1h. Keyed by
# strategy id alone, the third promotion silently REPLACED the first two - the
# routing table ended up describing one allocation for a strategy that had
# been promoted three times, and nothing on the console said which two had
# been dropped. So the record keeps a `configurations` list, one entry per
# (strat, symbol, timeframe), and re-promoting a pair UPDATES its entry in
# place rather than appending a second one - two entries for one pair would
# size the same signal twice on one account.
#
# The list lives INSIDE the id-keyed record rather than replacing the id key
# with a composite one, and that is not a stylistic choice. Three consumers
# read this block by strategy id:
#
#   portfolio.config_loader._reconcile_allocations   matches each key against
#                                                    active_strategies
#   portfolio.promotion_daemon.promote_strategy      moves the record by id
#                                                    when a strategy graduates
#   tests/test_promote_registration.py               the shape they pin
#
# A composite key breaks all three quietly: the reconciliation reports every
# record as an orphan and every permission as unallocated, and a graduation
# moves the permission while leaving the record on the incubator account -
# which is precisely the half-moved pair the daemon documents itself as
# existing to prevent.
def _configuration_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """`(strat, symbol, timeframe)`, case-folded. The dedup identity."""
    return (str(record.get("strat") or "").strip().lower(),
            str(record.get("symbol") or "").strip().upper(),
            str(record.get("timeframe") or "").strip().lower())


def merge_configuration(existing: list[dict[str, Any]] | None,
                        record: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """
    `record` folded into the per-pair list. Returns `(list, replaced)`.

    Order is preserved and an updated entry keeps its POSITION rather than
    moving to the end, so re-running a promotion that changed nothing leaves
    the file byte-identical apart from its timestamp. A promotion that is
    re-run must not churn the routing table - a diff on it is how an operator
    sees what actually moved.
    """
    out = [dict(r) for r in (existing or []) if isinstance(r, dict)]
    key = _configuration_key(record)
    for i, prior in enumerate(out):
        if _configuration_key(prior) == key:
            out[i] = dict(record)
            return out, True
    out.append(dict(record))
    return out, False


def _normalise_pid(name: Any) -> str:
    """
    A portfolio id reduced to what two spellings of it have in common.

    `--portfolio incubator-odd`, `Incubator-Odd` and `INCUBATOR_ODD` are the
    same account. Matching them exactly would refuse the spelling an operator
    actually types, and refusing on case is how somebody ends up editing the
    routing table by hand - which is the thing this function exists to stop
    being necessary.
    """
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def incubator_portfolios(portfolios: dict) -> list[str]:
    """The incubator-track portfolio ids, in a stable order."""
    return sorted(pid for pid, p in portfolios.items()
                  if isinstance(p, dict)
                  and p.get("account_type") == INCUBATOR_ACCOUNT_TYPE)


def _basket_covers(symbol: str, assets: list[str]) -> bool:
    """
    Can a strategy certified on `symbol` trade anything in `assets`?

    Delegates to `portfolio.config_loader`, which delegates in turn to the
    micro/full-size table `realtime/regime_daemon.py` owns and reconciles
    against `backtest/specs.py`. Three modules now ask this question - the
    loader, the live dispatcher and this one - and a second copy of the rule
    here would be free to disagree with the one that actually gates the order.

    `backtest/` may not import `portfolio/`, so this is a LAZY import inside
    the one function that needs it. That rule exists so a research number can
    never depend on live account state; this is a promotion writing a routing
    table, which is the account side of the line and the only place in this
    module that touches it.
    """
    from portfolio.config_loader import _basket_covers as _covers, _micro_alias
    return _covers(symbol, [str(a).upper() for a in assets], _micro_alias())


def resolve_portfolio(portfolios: dict,
                      requested: str | None = None,
                      strat: str | None = None,
                      quadrant: str | None = None,
                      symbol: str | None = None,
                      timeframe: str | None = None) -> tuple[str, str]:
    """
    Which incubator portfolio this promotion is registered onto, and why.

    Returns `(portfolio_id, basis)`. The basis is recorded on the allocation
    and printed, because "which account is this strategy on" answered by a rule
    nobody wrote down is exactly what `portfolio.config_loader` refuses to do
    when it declines to infer a portfolio from a strategy's name.

    An explicit `--portfolio` wins outright and is matched case- and
    punctuation-insensitively. Failing that, a strategy ALREADY named on an
    incubator portfolio stays on it. Otherwise the target is the incubator
    portfolio holding the FEWER active strategies, ties broken alphabetically -
    which is round-robin across successive promotions, because the portfolio
    that just took one is the one with more next time. It is a deterministic
    rule and that is the point: two operators promoting the same strategy get
    the same account, and a promotion that is re-run does not land somewhere
    else.

    THE STAY-PUT STEP IS WHAT MAKES THE ROUND-ROBIN CORRECT, and it was added
    on 2026-08-24 because a campaign that certifies several timeframes
    promotes ONE strategy several times in a row. Balanced by count alone,
    each of those promotions found its own strategy on the fuller account and
    MOVED it to the emptier one - so three certified timeframes bounced the
    strategy Even -> Odd -> Even, each move announced as a routing decision and
    each one flipping which quadrants the account permits. The rule balances
    NEW strategies across the two accounts; a strategy that has one is not new,
    and re-promoting a second timeframe of it is not a reason to re-open the
    question of where it lives.

    THE CERTIFIED QUADRANT NOW ROUTES, and this reverses what this function
    used to do. The old rule ignored it deliberately: a strategy id covered a
    whole module, its `regime_filter` was one of several certified quadrants,
    and letting one of them pick the account would have made the ACCOUNT a
    property of one Stage 1 designation.

    That argument died with the per-pair strategy id. An id is now ONE
    certified pair with ONE quadrant, and the two incubator accounts hold
    disjoint quadrant permissions - Odd trades Q3/Q4, Even trades Q1/Q2.
    Balancing purely by headcount therefore sends about half of every campaign
    to an account that forbids the one quadrant the pair was certified in, and
    `realtime/live_dispatcher.py` stands those down forever. Every count on
    every table still adds up and the symptom is silence, which is the failure
    `register_portfolio` already warns about and could not prevent.

    So: the incubator accounts DECLARING the certified quadrant are the
    candidates, and the headcount rule then balances among them. When none
    declares it - or no quadrant was resolved - every incubator account is a
    candidate and the headcount rule decides alone, exactly as before; the
    mismatch is still reported by `register_portfolio` rather than resolved by
    widening a basket, because `regime_quadrants` is the account's permission
    and every strategy on it inherits anything added there.
    """
    candidates = incubator_portfolios(portfolios)
    if not candidates:
        raise ValueError(
            f"{PORTFOLIO_CONFIG.name} declares no portfolio with "
            f"account_type {INCUBATOR_ACCOUNT_TYPE!r}, so there is no "
            f"incubator account to register onto.")

    if requested:
        wanted = _normalise_pid(requested)
        matches = [pid for pid in sorted(portfolios)
                   if _normalise_pid(pid) == wanted
                   or _normalise_pid((portfolios[pid] or {}).get("portfolio_id")
                                     if isinstance(portfolios[pid], dict)
                                     else None) == wanted]
        if not matches:
            raise ValueError(
                f"no portfolio named {requested!r}. Known: "
                f"{sorted(portfolios)} (matched ignoring case and "
                f"punctuation).")
        if len(matches) > 1:
            raise ValueError(
                f"{requested!r} matches more than one portfolio "
                f"({matches}); name it exactly.")
        pid = matches[0]
        if pid not in candidates:
            account = (portfolios[pid] or {}).get("account_type")
            raise ValueError(
                f"{pid} is an {account!r} account, not "
                f"{INCUBATOR_ACCOUNT_TYPE!r}. A promotion registers onto the "
                f"incubator track only - graduating to the prop track is "
                f"decided on FORWARD paper trades by "
                f"scripts/incubator_tracker.py --auto-promote, not on a "
                f"certification.")
        return pid, f"--portfolio {requested}"

    if strat:
        wanted = _normalise_pid(strat)
        for pid in candidates:
            names = portfolios[pid].get("active_strategies") or []
            if any(_normalise_pid(n) == wanted for n in names):
                return pid, (f"already registered on {pid}; a re-promotion "
                             f"does not move an account")

    # THE CERTIFIED SYMBOL FILTERS FIRST, and it filters harder than the
    # quadrant, because the two failures are not equally recoverable.
    #
    # A quadrant mismatch stands the strategy down in its own environment: the
    # account still carries it, and widening `basket.regime_quadrants` fixes it
    # in place. A SYMBOL mismatch is terminal -
    # `realtime.live_dispatcher.trades_symbol` refuses a strategy on any asset
    # its certification does not cover, so an account holding none of its
    # contract refuses it on everything it reaches, forever. Ranking the
    # quadrant first sent three NQ promotions of `t3_braid_scalp_20260823` to
    # Incubator-Even - which declares their Q1/Q2 and trades MES and MGC -
    # where they were correct on quadrant and could never place an order.
    #
    # An empty result is NOT resolved here. It means no incubator account
    # trades this contract at all, which is a partition that cannot carry the
    # strategy rather than a routing choice, and `register_portfolio` refuses
    # it with the contract and the basket named.
    pool, scope_note = candidates, ""
    symbol = str(symbol or "").strip().upper()
    if symbol and symbol != NOT_RESOLVED:
        carrying = [pid for pid in candidates
                    if _basket_covers(symbol,
                                      (portfolios[pid].get("basket") or {})
                                      .get("assets") or [])]
        if carrying:
            pool = carrying
            scope_note = f"trades {symbol}; "
        else:
            scope_note = (f"no incubator account trades {symbol}, so the "
                          f"certified contract could not route this; ")

    quadrant = str(quadrant or "").strip()
    if quadrant and quadrant != NOT_RESOLVED:
        permitting = [pid for pid in pool
                      if any(str(q).startswith(quadrant + "_")
                             for q in ((portfolios[pid].get("basket") or {})
                                       .get("regime_quadrants") or []))]
        if permitting:
            pool = permitting
            scope_note += f"declares {quadrant}; "
        else:
            scope_note += (f"no account in that pool declares {quadrant}, so "
                           f"the certified quadrant could not narrow it "
                           f"further; ")

    counts = {pid: len([s for s in (portfolios[pid].get("active_strategies")
                                    or [])])
              for pid in pool}
    pid = sorted(pool, key=lambda p: (counts[p], p))[0]
    tally = ", ".join(f"{p}={counts[p]}" for p in pool)

    # THE TIMEFRAME IS RECORDED, NOT FILTERED ON, and the distinction is
    # deliberate.
    #
    # Routing narrows by symbol and then by quadrant because a portfolio
    # DECLARES both - `basket.assets` and `basket.regime_quadrants` - so there
    # is something to match against. Nothing in the config declares which bar
    # widths the loop serving an account reads: that came from the systemd
    # unit's `--tf`, which this process cannot see.
    #
    # On 2026-08-27 a 3m certification was auto-registered onto an account
    # served by a 1h loop and was evaluated on hourly bars. The fix for that
    # is in the LOOP, which now reads every width its roster names and refuses
    # any strategy handed the wrong one - so the destination no longer has a
    # single timeframe for this function to validate against.
    #
    # What was missing and IS fixable here is the record: the basis said which
    # symbol and which quadrant decided the account and never which bar width
    # was being registered, so the one fact that turned out to matter was the
    # one nobody could read back afterwards.
    tf_note = ""
    if timeframe and str(timeframe).strip() and timeframe != NOT_RESOLVED:
        tf_note = f"; certified on {str(timeframe).strip()} bars"
    return pid, (f"{scope_note}fewest active strategies ({tally}), ties "
                 f"alphabetical{tf_note}")


def certified_scope(certification: Any,
                    meta: dict | None = None,
                    audit_path: Path | None = None) -> dict[str, Any]:
    """
    The contract, timeframe and regime quadrant this promotion was CERTIFIED
    on, each with the file it came from.

    A promotion is ONE contract at ONE timeframe, and a strategy module's
    `SYMBOLS`/`TIMEFRAME` are its declarations - every contract it targets, at
    the timeframe it prefers. `t3_braid_scalp_20260823` declares `NQ,ES,CL,GC`
    at 5m and was certified on NQ at 1h; taking the pair from the module would
    register NQ at 5m, an allocation for a run nobody made, with both halves
    individually true. This is the same trap `backtest/discord_reporter.py`
    documents for the Stage 5 card, resolved the same way: the certification
    supplies the pair, and the module's declarations are used only where they
    name exactly one symbol and nothing better is available.

    The timeframe falls back to the audit's FILENAME
    (`gate_audit_<SYMBOL>_<TF>.json`) when the audit body records none, because
    Stage 3 writes the pair into the name whether or not it writes it into the
    file.

    `regime_filter` is Gate R's certification target - the ONE quadrant Stage 1
    designated and Stage 3 measured the holdout inside. It is transcribed, not
    re-derived: naming a quadrant here from anything but the audit would be a
    second best-of-four pick, which is the selection Gate R exists to prevent.
    The `Q1`..`Q4` code comes from `backtest.profiler.REGIME_TO_QUADRANT`, so
    no spelling of a regime name lives in this module.
    """
    meta = meta or {}
    cert = certification if isinstance(certification, dict) else {}

    # A certification block written before this module recorded the pair and
    # the quadrant names the audit FILE but not its contents. Every promotion
    # made before that change is in exactly that state, so the fields are read
    # back out of the file the block already cites rather than reported as
    # absent - the certification says where the answer is, and the answer has
    # not moved. Recorded in `resolved_from` as the audit rather than as the
    # certification, because they are two different reads and only one of them
    # was verified against the SHA-256 in meta.json.
    from_audit: dict[str, Any] = {}
    audit_on_disk = Path(cert.get("audit_file") or (audit_path or ""))
    if audit_on_disk.name and audit_on_disk.exists():
        try:
            from_audit = json.loads(audit_on_disk.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            from_audit = {}
    for key in ("audit_timeframe", "target_quadrant", "target_regime"):
        if not cert.get(key):
            value = from_audit.get(
                "timeframe" if key == "audit_timeframe" else key)
            if value:
                cert = dict(cert)
                cert[key] = value
                cert.setdefault("_from_audit", []).append(key)

    out: dict[str, Any] = {"symbol": NOT_RESOLVED,
                           "timeframe": NOT_RESOLVED,
                           "regime_filter": NOT_RESOLVED,
                           "regime": NOT_RESOLVED,
                           "resolved_from": {}}
    src = out["resolved_from"]
    name = Path(cert.get("audit_file") or (audit_path or "")).name

    symbol = cert.get("audit_symbol")
    if symbol:
        out["symbol"] = str(symbol)
        src["symbol"] = f"{name} (certification)" if name else "certification"
    else:
        symbols = [s for s in (meta.get("symbols") or []) if s]
        if len(symbols) == 1:
            out["symbol"] = str(symbols[0])
            src["symbol"] = ("meta.json symbols - the module declares exactly "
                             "one contract")
        else:
            src["symbol"] = (f"no certification symbol, and the module "
                             f"declares {len(symbols)} contracts")

    recovered = cert.get("_from_audit") or []
    tf = cert.get("audit_timeframe")
    if tf:
        out["timeframe"] = str(tf)
        src["timeframe"] = (
            f"{name} (the audit itself)" if "audit_timeframe" in recovered
            else (f"{name} (certification)" if name else "certification"))
    else:
        # gate_audit_<SYMBOL>_<TF>.json -> the last underscore-separated part.
        parts = Path(name).stem.split("_") if name else []
        if len(parts) >= 4 and parts[0] == "gate" and parts[1] == "audit":
            out["timeframe"] = parts[-1]
            src["timeframe"] = f"{name} (the audit's filename)"
        elif meta.get("timeframe"):
            out["timeframe"] = str(meta["timeframe"])
            src["timeframe"] = ("meta.json timeframe - the MODULE's declared "
                                "preference, not a certified pair")
        else:
            src["timeframe"] = "no certification and no module declaration"

    quadrant = cert.get("target_quadrant")
    regime = cert.get("target_regime")
    if not quadrant and regime:
        try:
            from backtest.profiler import REGIME_TO_QUADRANT
            quadrant = REGIME_TO_QUADRANT.get(str(regime))
        except Exception:                                       # noqa: BLE001
            quadrant = None
    if quadrant:
        out["regime_filter"] = str(quadrant)
        out["regime"] = str(regime or NOT_RESOLVED)
        src["regime_filter"] = (
            f"{name} (Gate R's certification target"
            + (", read from the audit itself)"
               if ("target_quadrant" in recovered
                   or "target_regime" in recovered) else ")")
            if name else "certification")
    else:
        src["regime_filter"] = ("the certification records no target quadrant "
                                "- Gate R was NOT EVALUATED, or the audit "
                                "predates the charter")
    return out


def _write_portfolio_config(path: Path, blob: dict) -> None:
    """
    Replace the routing table atomically, and never with something unreadable.

    Temp file, then `os.replace`, the way `portfolio/promotion_daemon.py` and
    `backtest/status.py` write theirs: a reader - including a live dispatcher
    mid-cycle - sees the previous complete document or the new one, never half
    of either.

    The temp file is PARSED BACK before it is moved into place. This module
    cannot call `portfolio.config_loader.load_portfolio_config` to validate
    what it wrote (the dependency runs one way), so the one guarantee it can
    still make is that the bytes it is about to install are valid JSON
    describing the same four portfolios. A routing table that will not parse
    takes down the live loop, the incubator tracker and the Stage 5 card
    together, and it would do it on the next run rather than on this one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    try:
        check = json.loads(tmp.read_text(encoding="utf-8"))
        if set(check.get("portfolios") or {}) != set(blob.get("portfolios") or {}):
            raise ValueError("the written portfolios do not round-trip")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def unrouted_packages(config_path: Path | str = PORTFOLIO_CONFIG,
                      incubator: Path = INCUBATOR) -> dict[str, list[str]]:
    """
    Promoted packages that no portfolio routes, grouped by certified contract.

    DERIVED FROM STATE, NOT FROM THIS RUN'S LOG. A tally counted while
    promoting would only see the configurations this invocation touched, and
    the number an operator needs is how many are unrouted IN TOTAL - a
    campaign that refuses four today, on a tree that already held seventy-six,
    has a problem of size eighty. Reading the incubator against the routing
    table also cannot drift from what the live loop will actually load, which
    a counter incremented in a loop can.

    THIS IS NOT A FAULT LIST. A promotion records that a version was CHOSEN,
    not that it was armed, and `register_portfolio` refuses on purpose when no
    incubator basket carries the certified contract - routing a strategy to an
    account that cannot trade its symbol would have it refused on every asset
    it reaches, forever. Grouping by symbol is what makes the remedy legible:
    the question is which contracts the incubator should carry, and that is a
    decision about accounts and risk rather than anything a promotion can
    settle.
    """
    try:
        blob = json.loads(Path(config_path).read_text())
    except (OSError, ValueError):
        return {}
    routed: set[str] = set()
    for block in (blob.get("portfolios") or {}).values():
        if isinstance(block, dict):
            routed.update(block.get("active_strategies") or [])

    # Which contracts an INCUBATOR basket can carry. The two reasons a package
    # is unrouted are fixed by different work and must not be pooled: a
    # contract no basket carries needs a decision about accounts and risk,
    # while a contract a basket carries already needs only a re-promotion -
    # it was certified before the basket was widened, and routing is decided
    # at promotion time and never revisited.
    carried: list[str] = []
    for block in (blob.get("portfolios") or {}).values():
        if (isinstance(block, dict)
                and block.get("account_type") == INCUBATOR_ACCOUNT_TYPE):
            carried += list((block.get("basket") or {}).get("assets") or [])

    out: dict[str, list[str]] = {}
    if not Path(incubator).is_dir():
        return out
    for pkg in sorted(Path(incubator).iterdir()):
        meta_path = pkg / "meta.json"
        if not pkg.is_dir() or pkg.name in routed or not meta_path.is_file():
            # No meta.json means it is a stray directory, not a promoted
            # package. Counting one would invent an unrouted strategy.
            continue
        try:
            symbol = str(json.loads(meta_path.read_text()).get("symbol") or "")
        except (OSError, ValueError):
            symbol = ""
        key = symbol or "UNKNOWN"
        if symbol and _basket_covers(symbol, carried):
            # Marked, not dropped. Saying "no basket carries their contract"
            # over a package whose contract IS carried would be false on the
            # card, and the reader would go looking for a basket edit that is
            # already done.
            key = f"{symbol} (carried; re-promote to route)"
        out.setdefault(key, []).append(pkg.name)
    return out


def register_portfolio(strat: str,
                       *,
                       version: str,
                       scope: dict,
                       allocation: int = DEFAULT_ALLOCATION,
                       portfolio: str | None = None,
                       status: str = ALLOCATION_STATUS,
                       config_path: Path = PORTFOLIO_CONFIG,
                       incubator: Path = INCUBATOR) -> dict[str, Any]:
    """
    Register a promoted strategy onto an incubator portfolio.

    Writes BOTH halves together, which is the whole contract of this function:

        active_strategies      the id, as a string. This is the PERMISSION,
                               and it is the only thing
                               `get_portfolio_for_strategy`, the live
                               dispatcher and the Stage 5 card read.
        strategy_allocations   the record: strat, symbol, timeframe, version,
                               allocation, regime_filter, status, path.

    Idempotent. Registering the same strategy twice updates the record in
    place; it never appends the id a second time, which would size one signal
    twice on one account.

    A strategy already named on the OTHER incubator portfolio is MOVED, not
    added - `portfolio.config_loader` refuses a config naming one strategy on
    two portfolios of one track, because both would size the same signal
    independently and the net position would be double what either risk
    profile describes. The move is returned as `moved_from` and printed.

    THE REGIME SCOPE IS RECORDED, NEVER APPLIED. `regime_filter` is this
    strategy's certified quadrant; `basket.regime_quadrants` is the ACCOUNT's
    permission and is shared by every strategy on it, so widening it to admit
    this one would hand every other strategy on that account a quadrant nobody
    certified it for - silently, and in the direction that trades. This
    function therefore never edits the basket. When the two disagree the
    conflict is returned in `notes`, printed at promotion, and recorded on the
    loaded config by `portfolio.config_loader` as
    `allocation_reconciliation` - which is where the authoritative comparison
    lives, because that is where the schema-label -> `Q1`..`Q4` mapping lives.
    """
    path = Path(config_path)
    # Creates the incubator track when it is missing rather than failing a
    # promotion that is already written and committed - see
    # `ensure_portfolio_groups`, which announces every default it had to
    # supply instead of writing a plausible account silently.
    blob, bootstrap_notes = _load_portfolio_config(path)
    portfolios = blob.get("portfolios")
    if not isinstance(portfolios, dict) or not portfolios:
        raise ValueError(f"{path} carries no `portfolios` object")

    pid, basis = resolve_portfolio(portfolios, portfolio, strat,
                                   scope.get("regime_filter"),
                                   scope.get("symbol"),
                                   scope.get("timeframe"))
    target = portfolios[pid]
    active = target.get("active_strategies")
    if not isinstance(active, list):
        raise ValueError(
            f"{pid}: active_strategies is {type(active).__name__}, not a list. "
            f"Refusing to write into a routing table this module does not "
            f"recognise.")

    allocation = int(allocation)
    if allocation < 1:
        raise ValueError(f"--allocation must be at least 1 contract, got "
                         f"{allocation}")

    wanted = _normalise_pid(strat)
    notes: list[str] = list(bootstrap_notes)

    # A BARE registration this pair's id supersedes.
    #
    # Before the per-pair id, a campaign that certified NQ at 15m, 30m and 1h
    # registered all three under one id - `t3_braid_scalp_20260823` - each
    # promotion replacing the last. That entry grants permission to
    # `approved_incubator/<strategy>/`, a directory whose meta.json describes
    # exactly one of the pairs, so leaving it beside the three isolated ids
    # would arm a fourth allocation nobody certified and would double-size
    # whichever pair happened to be in it. It is removed, and the removal is
    # announced - a permission withdrawn silently is as bad as one granted
    # silently.
    base = base_strategy(strat)
    superseded = (_normalise_pid(base)
                  if base and _normalise_pid(base) != wanted else None)

    # Every per-pair configuration already on the record, from wherever the
    # record currently is. Collected BEFORE the move below pops it off the
    # other portfolio: a strategy certified on NQ 15m on one account and then
    # promoted at 1h onto the other would otherwise arrive with an empty list
    # and the 15m allocation would vanish with nothing saying so.
    prior_configurations: list[dict[str, Any]] = []
    for block in portfolios.values():
        if not isinstance(block, dict):
            continue
        allocs = block.get(ALLOCATIONS_KEY)
        if not isinstance(allocs, dict):
            continue
        for key, prior in allocs.items():
            if _normalise_pid(key) != wanted or not isinstance(prior, dict):
                continue
            carried = prior.get("configurations")
            if isinstance(carried, list) and carried:
                prior_configurations = [dict(c) for c in carried
                                        if isinstance(c, dict)]
            elif prior.get("symbol") and prior.get("timeframe"):
                # A record written before `configurations` existed. Seeded
                # from its top-level fields rather than dropped: that record
                # IS a promotion somebody made, and losing it here is exactly
                # the silent replacement this list was added to stop.
                prior_configurations = [dict(prior)]

    # Remove the id and any stale allocation record from every OTHER portfolio
    # on this track. Leaving one behind is the double-sizing config the loader
    # refuses, and it would be refused on the next load rather than here.
    retired: list[str] = []
    if superseded:
        for block in portfolios.values():
            if not isinstance(block, dict):
                continue
            names = block.get("active_strategies")
            if isinstance(names, list):
                gone = [n for n in names if _normalise_pid(n) == superseded]
                if gone:
                    # IN PLACE. `active` below is a reference to the TARGET
                    # portfolio's list, taken before this runs; rebinding the
                    # key to a fresh list here leaves `active` aliasing the old
                    # object, so the id this function then appends never
                    # reaches the file. The promotion reports a registration
                    # and the routing table grants nothing - the one failure
                    # mode where the console and the config disagree.
                    names[:] = [n for n in names
                                if _normalise_pid(n) != superseded]
                    retired.extend(str(g) for g in gone)
            allocs = block.get(ALLOCATIONS_KEY)
            if isinstance(allocs, dict):
                for key in [k for k in allocs
                            if _normalise_pid(k) == superseded]:
                    allocs.pop(key)
    if retired:
        notes.append(
            f"retired the bare registration {', '.join(sorted(set(retired)))} "
            f"- it is superseded by the per-pair ids. That entry granted "
            f"permission to approved_incubator/{base}/, whose meta.json "
            f"describes ONE of the certified pairs, so leaving it beside them "
            f"would arm a fourth allocation nobody certified.")

    moved_from: list[str] = []
    for other in incubator_portfolios(portfolios):
        if other == pid:
            continue
        block = portfolios[other]
        names = block.get("active_strategies")
        if isinstance(names, list):
            kept = [n for n in names if _normalise_pid(n) != wanted]
            if len(kept) != len(names):
                names[:] = kept                 # in place - see the note above
                moved_from.append(other)
        allocs = block.get(ALLOCATIONS_KEY)
        if isinstance(allocs, dict):
            for key in [k for k in allocs if _normalise_pid(k) == wanted]:
                allocs.pop(key)

    already = [n for n in active if _normalise_pid(n) == wanted]
    if already:
        # Keep the spelling already in the file rather than overwriting it with
        # this invocation's. The id is a directory name under
        # approved_incubator/ and the file's copy is the one the live loop has
        # been resolving; silently re-casing it here would move which directory
        # is loaded without saying so.
        strat_id = str(already[0])
        if len(already) > 1:
            active[:] = [n for n in active if _normalise_pid(n) != wanted]
            active.append(strat_id)
            notes.append(f"{pid} named {strat} {len(already)} times; collapsed "
                         f"to one entry")
    else:
        strat_id = str(strat)
        active.append(strat_id)

    record = {
        "strat": strat_id,
        "symbol": scope.get("symbol", NOT_RESOLVED),
        "timeframe": scope.get("timeframe", NOT_RESOLVED),
        "version": str(version).upper(),
        "allocation": allocation,
        "regime_filter": scope.get("regime_filter", NOT_RESOLVED),
        "status": status,
        "path": (Path(incubator) / strat_id / "strat.py")
                 .relative_to(REPO_ROOT).as_posix()
                 if Path(incubator).is_absolute()
                 and str(Path(incubator)).startswith(str(REPO_ROOT))
                 else (Path(incubator) / strat_id / "strat.py").as_posix(),
        "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registered_by": "backtest/promote.py",
        "routing_basis": basis,
        "resolved_from": dict(scope.get("resolved_from") or {}),
    }

    # One entry per certified (strat, symbol, timeframe), deduplicated on that
    # triple. The top-level fields above describe THIS promotion - the most
    # recent one - and stay exactly where `portfolio.config_loader` and
    # `portfolio.promotion_daemon` already read them; `configurations` is the
    # complete list, so a campaign that certified three timeframes stops
    # collapsing into whichever was promoted last. See `merge_configuration`.
    configurations, replaced = merge_configuration(prior_configurations,
                                                   record)
    record["configurations"] = configurations
    if replaced:
        notes.append(
            f"{record['symbol']} {record['timeframe']} was already registered; "
            f"its allocation record was UPDATED in place rather than added a "
            f"second time - two entries for one pair would size the same "
            f"signal twice on one account.")
    elif len(configurations) > 1:
        others = ", ".join(f"{c.get('symbol')} {c.get('timeframe')}"
                           for c in configurations
                           if _configuration_key(c) != _configuration_key(record))
        notes.append(
            f"{strat_id} now holds {len(configurations)} certified "
            f"configurations on {pid} ({others}, and this one). "
            f"approved_incubator/{strat_id}/ holds ONE module and ONE "
            f"meta.json, and they describe the promotion that ran LAST - so "
            f"the live loop trades this pair's parameters for every "
            f"configuration listed here until each is promoted into its own "
            f"strategy id.")

    allocs = target.get(ALLOCATIONS_KEY)
    if not isinstance(allocs, dict):
        allocs = {}
        target[ALLOCATIONS_KEY] = allocs
    for key in [k for k in allocs if _normalise_pid(k) == wanted]:
        allocs.pop(key)
    allocs[strat_id] = record

    declared = list((target.get("basket") or {}).get("regime_quadrants") or [])
    quadrant = record["regime_filter"]
    if quadrant == NOT_RESOLVED:
        notes.append(
            "no certified quadrant was resolved, so `regime_filter` is "
            "NOT RESOLVED. The live regime gate reads the ACCOUNT's "
            "basket.regime_quadrants, not this field, so nothing is widened "
            "or narrowed by it - but a promotion whose Gate R target cannot "
            "be stated was certified by an audit that predates the charter, "
            "or by one where Gate R was NOT EVALUATED.")
    elif declared and not any(str(q).startswith(quadrant + "_")
                              for q in declared):
        notes.append(
            f"{strat_id} is certified in {quadrant}, and {pid} declares "
            f"{declared}. The basket was NOT widened: regime_quadrants is the "
            f"ACCOUNT's permission and every strategy on it inherits any "
            f"quadrant added here. Until they agree the live gate "
            f"(realtime/live_dispatcher.py) stands this strategy down in the "
            f"one quadrant it was certified for, so it will never trade. Fix "
            f"it by routing to the incubator portfolio that already declares "
            f"{quadrant}, or by editing basket.regime_quadrants deliberately.")

    # THE SYMBOL CHECK IS A REFUSAL, where the quadrant check above is a note.
    #
    # The difference is whether the strategy can EVER trade. A quadrant
    # mismatch stands it down in its own environment, which is wrong and
    # recoverable - the account still carries it and widening the basket's
    # quadrants fixes it in place. A symbol mismatch is terminal:
    # `realtime.live_dispatcher.trades_symbol` refuses a strategy on any asset
    # its certification does not cover, so a strategy routed to a basket
    # holding none of its contract is refused on every asset it reaches and
    # can never place an order. There is no market state in which it starts
    # working, and the console for it is indistinguishable from a quiet market.
    #
    # This is what put three promotions of `t3_braid_scalp_20260823` on
    # `Incubator-Even`: certified on NQ, routed to a basket of MES and MGC,
    # correct on quadrant and dead on arrival, with `active_strategies` naming
    # them and every table adding up.
    #
    # AN EXPLICIT `--portfolio` IS STILL HONOURED, with a loud note. The three
    # bad promotions were routed AUTOMATICALLY; an operator naming an account
    # is a decision, and this module's rule everywhere else is that an explicit
    # instruction outranks a file. Refusing it outright would also make the
    # account unnameable while the basket is being fixed.
    assets = [str(a).upper()
              for a in ((target.get("basket") or {}).get("assets") or [])]
    symbol = str(record.get("symbol") or "").upper()
    covered = (not symbol or symbol == NOT_RESOLVED or not assets
               or _basket_covers(symbol, assets))
    explicit = str(basis or "").startswith("--portfolio")
    if not covered and explicit:
        notes.append(
            f"{strat_id} is certified on {symbol} and {pid} trades {assets}. "
            f"The basket was NOT widened. You named this account explicitly, "
            f"so the registration stands - but the live loop checks the "
            f"certified symbol against the basket, so this strategy is "
            f"refused on every asset it is routed to and will never place an "
            f"order until {pid}'s basket carries {symbol} or its micro.")
    elif not covered:
        raise ValueError(
            f"{strat_id} is certified on {symbol} and {pid} trades {assets}. "
            f"The live loop checks the certified symbol against the basket, "
            f"so this strategy would be refused on every asset it is routed "
            f"to and could never place an order. Route it to an incubator "
            f"portfolio whose basket carries {symbol} (or its micro), or add "
            f"the contract to {pid}'s basket.assets deliberately - widening a "
            f"basket grants every strategy on that account the new contract.")

    if record["symbol"] == NOT_RESOLVED or record["timeframe"] == NOT_RESOLVED:
        notes.append(
            "the certified contract or timeframe could not be resolved, so "
            "the allocation records NOT RESOLVED rather than the module's "
            "declarations. Pass --audit-file from stage 3, or --symbol / "
            "--timeframe to state the pair by hand.")

    _write_portfolio_config(path, blob)
    return {"portfolio_id": pid, "basis": basis, "record": record,
            "config_path": path, "moved_from": moved_from,
            "retired": sorted(set(retired)),
            "declared_quadrants": declared, "notes": notes,
            "was_registered": bool(already)}


def post_stage5_card(strat: str, *, dry_run: bool = False,
                     python: str | None = None) -> dict[str, Any]:
    """
    Run `discord_reporter.py --stage 5` for this strategy.

    A SUBPROCESS rather than an import, for two reasons. The card resolves
    every value it prints from what the promotion just wrote - meta.json, the
    snapshot beside it, the gate audit those cite and now the routing table -
    so running it as its own process reads the files as they are ON DISK,
    which is the state a human re-running the same command by hand would see.
    And a card that fails cannot take a completed promotion with it.

    Failures are RETURNED, never raised, the same way `git_commit` reports
    one: the strategy is promoted and registered either way, and the exact
    command to re-run is printed.
    """
    reporter = REPO_ROOT / "backtest" / "discord_reporter.py"
    cmd = [python or sys.executable, str(reporter), "--stage", "5",
           "--strat", strat]
    if dry_run:
        cmd.append("--dry-run")
    if not reporter.exists():
        return {"posted": False, "cmd": cmd, "returncode": None,
                "output": f"{reporter} does not exist"}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          check=False)
    return {"posted": proc.returncode == 0, "cmd": cmd,
            "returncode": proc.returncode,
            "output": (proc.stdout or "").strip() or (proc.stderr or "").strip()}


# --------------------------------------------------------------------------
# CLI auto-resolution
# --------------------------------------------------------------------------
# `--strat` names one strategy, and everything else on the command line is
# derivable from it plus what Stage 3 already wrote. Before 2026-08-24
# `--version` and `--source` were both `required=True`, which made the
# ORCHESTRATOR the only practical caller: an operator promoting one certified
# pair by hand had to retype a module path and a version letter that the gate
# audit beside it already records, and a mistyped version letter promotes a
# Version B wrapper under Version A's metrics with nothing raising.
#
# Every resolution here NAMES THE FILE IT CAME FROM and refuses to guess when
# more than one answer is available. That is the same rule the Stage 5 Discord
# card follows: a resolved value is only better than a typed one while it is
# traceable back to the artifact that supplied it.

# Where a strategy module is looked for, in order. `strategies/<strat>.py` is
# the path the orchestrator's documentation spells; in this repository the
# modules actually live one level down in `experimental/`, so both are tried
# and the one that exists wins. `approved_incubator/<strat>/strat.py` is
# deliberately NOT a candidate - promoting from the previous promotion's own
# copy is circular, and it would silently re-promote a generated Version B
# wrapper as though it were a source module.
SOURCE_CANDIDATES = ("strategies/{strat}.py",
                     "strategies/experimental/{strat}.py")

# The version a promotion defaults to when neither the CLI nor the gate audit
# names one. Version A is the rule-based baseline and is what Stage 3
# certifies unless a run said otherwise; defaulting to B would promote a
# generated ML wrapper on the strength of an audit of the baseline.
DEFAULT_VERSION = "A"


def resolve_source(strat: str, explicit: str | Path | None = None,
                   repo_root: Path = REPO_ROOT) -> tuple[Path, str]:
    """
    The strategy module `--source` cites, and where it was found.

    An explicit `--source` is honoured verbatim, including one that does not
    exist - `promote` raises on it with the path in the message, which is what
    an operator who typed a path wants to see. Without one the candidates in
    `SOURCE_CANDIDATES` are tried in order and the FIRST that exists wins.

    More than one candidate existing is not resolved silently: two modules
    named for one strategy are two different strategies, and promoting
    whichever the tuple happened to list first would record a SHA-256 for a
    file nobody chose. It raises and names both.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path, "--source"

    found = [repo_root / c.format(strat=strat) for c in SOURCE_CANDIDATES]
    found = [p for p in found if p.exists()]
    if not found:
        tried = ", ".join(c.format(strat=strat) for c in SOURCE_CANDIDATES)
        raise FileNotFoundError(
            f"no strategy module for {strat!r}. Tried {tried} relative to "
            f"{repo_root}. Name it with --source.")
    if len(found) > 1:
        names = ", ".join(str(p.relative_to(repo_root)) for p in found)
        raise ValueError(
            f"{strat!r} names more than one module ({names}). Two modules "
            f"under one strategy name are two different strategies and the "
            f"promoted SHA-256 would describe whichever was listed first. "
            f"Name one with --source.")
    return found[0], f"resolved from --strat ({found[0].relative_to(repo_root)})"


def _pipeline_dir(strat: str, out_dir: str | Path | None = None) -> Path:
    """
    Stage 3's handoff directory, behind a seam.

    Imported lazily and from `backtest.pipeline`, which pulls in `json`, `os`
    and `mdlib.env` and nothing else - promoting must not depend on the engine
    or on vectorbtpro being importable, since a promotion reads artifacts and
    runs no simulation.
    """
    from backtest.pipeline import pipeline_dir
    return pipeline_dir(strat, out_dir)


def resolve_audit_file(strat: str,
                       symbol: str | None = None,
                       timeframe: str | None = None,
                       explicit: str | Path | None = None,
                       out_dir: str | Path | None = None
                       ) -> tuple[Path | None, str]:
    """
    The Stage 3 certification this promotion rests on, and where it came from.

    Returns `(path, basis)`; `path` is None when nothing could be resolved,
    which promote() reports as NOT CERTIFIED rather than treating as a pass.

    Four routes, tried in order, each narrower than the one after it:

      1. `--audit-file`, verbatim. A file named by hand and missing RAISES -
         the operator asked for a specific certification and the answer that
         it is not there is the useful one.
      2. `--symbol` + `--timeframe` -> `gate_audit_<SYMBOL>_<TF>.json` in
         `<BT_ARTIFACTS>/pipeline/<strat>/`. This is the pair Stage 3 writes
         and the one the orchestrator passes.
      3. `stage3_audit_summary.json`, when it records EXACTLY ONE certified
         configuration. More than one is refused and all of them are named:
         picking one would promote a pair nobody chose while the others sat on
         disk, and every field on the resulting card would still read
         correctly.
      4. Exactly one `gate_audit_<SYMBOL>_<TF>.json` on disk. Only the
         SUFFIXED files are considered - the unsuffixed `gate_audit_<SYM>.json`
         duplicates whichever timeframe ran last, so counting it would make two
         files look like two certifications and refuse a directory holding one.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(
                f"--audit-file {path} does not exist. A certification named "
                f"by hand and missing is refused rather than resolved to "
                f"another one.")
        return path, "--audit-file"

    try:
        directory = _pipeline_dir(strat, out_dir)
    except Exception as exc:                                      # noqa: BLE001
        return None, (f"the pipeline directory could not be resolved "
                      f"({type(exc).__name__}: {exc})")

    if symbol and timeframe:
        path = directory / f"gate_audit_{symbol}_{timeframe}.json"
        if path.exists():
            return path, (f"resolved from --symbol/--timeframe "
                          f"({path.name} in {directory})")
        return None, (f"no {path.name} in {directory} - Stage 3 has not "
                      f"certified {symbol} at {timeframe}")

    summary = directory / "stage3_audit_summary.json"
    if summary.exists():
        try:
            rows = (json.loads(summary.read_text(encoding="utf-8"))
                    .get("results") or [])
        except (OSError, ValueError):
            rows = []
        certified = [r for r in rows if r.get("certified") is True]
        if len(certified) == 1:
            row = certified[0]
            named = row.get("audit_file")
            path = (Path(named) if named
                    else directory / f"gate_audit_{row.get('symbol')}_"
                                     f"{row.get('timeframe')}.json")
            if path.exists():
                return path, (f"resolved from {summary.name} - the one "
                              f"certified configuration ({row.get('symbol')} "
                              f"{row.get('timeframe')})")
        elif len(certified) > 1:
            pairs = ", ".join(f"{r.get('symbol')} {r.get('timeframe')}"
                              for r in certified)
            raise ValueError(
                f"{summary.name} records {len(certified)} certified "
                f"configurations ({pairs}). Promoting one of them would leave "
                f"the rest on disk with nothing saying they were not chosen. "
                f"Name the pair with --symbol/--timeframe, the file with "
                f"--audit-file, or promote all of them with "
                f"`python3 backtest/run_pipeline.py --strat {strat} "
                f"--promote-only`.")

    audits = sorted(p for p in directory.glob("gate_audit_*_*.json")
                    if p.is_file())
    if len(audits) == 1:
        return audits[0], (f"resolved from {directory} - the one gate audit "
                           f"on disk ({audits[0].name})")
    if len(audits) > 1:
        raise ValueError(
            f"{directory} holds {len(audits)} gate audits "
            f"({', '.join(p.name for p in audits)}) and no summary naming one "
            f"certified configuration. Name the pair with "
            f"--symbol/--timeframe or the file with --audit-file.")
    return None, (f"no gate audit for {strat} in {directory}")


def resolve_version(audit_path: Path | None = None,
                    explicit: str | None = None) -> tuple[str, str]:
    """
    Which version is being promoted, and on whose authority.

    An explicit `--version` wins. Otherwise the gate audit says which twin it
    certified, and promoting the other one would attach a generated Version B
    wrapper to an audit of the baseline. With neither, `DEFAULT_VERSION`.

    A gate audit does not carry a scalar `version` - it carries `passed` and
    `status` as objects KEYED BY VERSION (`{"A": true}`), because Stage 3
    audits both twins in one pass. So the version is the one whose `passed` is
    True, and a scalar `version` is still read first for a handoff that
    carries one.

    Two things it will not do. It never picks between two PASSING versions:
    which twin to trade is the Dual-Version Mandate's decision, made on
    whether B beat A out of sample, and it belongs to the operator - so both
    passing falls through to `DEFAULT_VERSION` and says so. And anything that
    is not A or B is IGNORED rather than passed through: `--version` is a
    two-value choice everywhere else in this module, and a third token would
    reach `promote()` and select the Version B branch by not being "A".
    """
    if explicit:
        return str(explicit).upper(), "--version"
    if audit_path is not None and Path(audit_path).exists():
        name = Path(audit_path).name
        try:
            blob = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            blob = {}
        recorded = str(blob.get("version") or "").strip().upper()
        if recorded in ("A", "B"):
            return recorded, f"{name} (the certification)"
        passed = blob.get("passed")
        if isinstance(passed, dict):
            won = [str(k).upper() for k, v in passed.items()
                   if v is True and str(k).upper() in ("A", "B")]
            if len(won) == 1:
                return won[0], f"{name} (the version whose Gate R passed)"
            if len(won) > 1:
                return DEFAULT_VERSION, (
                    f"default - {name} certified {', '.join(sorted(won))} and "
                    f"choosing between them is the Dual-Version Mandate's "
                    f"decision, not this script's. Name one with --version.")
        audited = [str(k).upper() for k in (blob.get("versions") or {})
                   if str(k).upper() in ("A", "B")]
        if len(audited) == 1:
            return audited[0], f"{name} (the one version it audited)"
    return DEFAULT_VERSION, (f"default - neither --version nor a gate audit "
                             f"named one")


# --------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------
def git_commit(dest: Path, strat: str, version: str,
               gate_status: str) -> dict[str, Any]:
    """
    Stage and commit the promoted directory. Only that directory.

    Failures are reported, not raised: the files are already written and
    correct, and a promotion is not undone because git had an opinion. The
    caller is told exactly what to run by hand.
    """
    def run(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(argv, cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False)

    if run("git", "rev-parse", "--git-dir").returncode != 0:
        return {"committed": False,
                "commit_output": f"{REPO_ROOT} is not a git repository"}

    add = run("git", "add", "--", str(dest))
    if add.returncode != 0:
        return {"committed": False, "commit_output": add.stderr.strip()}

    staged = run("git", "diff", "--cached", "--name-only", "--", str(dest))
    if not staged.stdout.strip():
        return {"committed": False,
                "commit_output": "nothing to commit - the promoted files are "
                                 "already identical to what is in git"}

    message = (f"Promote {strat} Version {version} to the incubator\n\n"
               f"Gate audit: {gate_status}. Being in approved_incubator/ is a\n"
               f"record that this version was chosen, not permission to trade\n"
               f"it - deployment still needs the gates in\n"
               f"docs/STRATEGY_DEVELOPMENT.md and the 3-year holdout.\n")
    commit = run("git", "commit", "-m", message, "--", str(dest))
    if commit.returncode != 0:
        return {"committed": False,
                "commit_output": (commit.stderr or commit.stdout).strip()}
    return {"committed": True, "commit_output": commit.stdout.strip()}


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Promote a strategy version into approved_incubator/.")
    p.add_argument("--strat", required=True, help="Strategy name (directory name)")
    p.add_argument("--version", default=None, choices=["A", "B", "a", "b"],
                   help="A = rule-based baseline, B = baseline + ML filter. "
                        "Omitted, it is read from the gate audit's own "
                        f"`version`, and failing that defaults to "
                        f"{DEFAULT_VERSION}.")
    p.add_argument("--source", default=None,
                   help="Path to the experimental strategy module. Omitted, "
                        "it resolves to the first of "
                        + " / ".join(SOURCE_CANDIDATES) + " that exists.")
    p.add_argument("--metrics", default=None,
                   help="dual_metrics.json from the run being promoted on. "
                        "Without it meta.json records NOT RECORDED rather than "
                        "a metrics snapshot nobody produced.")
    p.add_argument("--audit-file", default=None,
                   help="gate_audit_<SYMBOL>.json from stage 3 "
                        "(backtest/audit_gates.py). This is the AUTHORITATIVE "
                        "gate verdict — the audit inside dual_metrics.json "
                        "comes from a single run, which can only evaluate "
                        "Gate 1. Promotion is refused unless it says PASS.")
    p.add_argument("--require-certification", action="store_true",
                   help="Refuse to promote with no --audit-file at all. The "
                        "five-stage pipeline should pass this; the older "
                        "bt-run workflow predates certification and does not.")
    p.add_argument("--symbol", default=None,
                   help="Override the module's SYMBOLS")
    p.add_argument("--timeframe", default=None,
                   help="Override the module's TIMEFRAME")
    p.add_argument("--params", default=None,
                   help="JSON dict merged over the module's DEFAULT_PARAMS")
    # The SAME default as the three stages upstream. A promotion that baked
    # in a different number would deploy a Version B nobody certified.
    p.add_argument("--ml-threshold", "--threshold", dest="threshold",
                   type=float, default=ML_THRESHOLD_DEFAULT,
                   help=(f"Version B: P(win) at or above which an entry is "
                         f"kept (default {ML_THRESHOLD_DEFAULT}). Embedded "
                         f"into the promoted module as ML_THRESHOLD"))
    p.add_argument("--variants-tested", type=int, default=None,
                   help="How many variants were tried to reach this result")
    p.add_argument("--notes", default="", help="Free text for meta.json")
    p.add_argument("--force", action="store_true",
                   help="Promote even when the gate audit did not pass. "
                        "Recorded in meta.json.")
    p.add_argument("--dow-gate", default=None, metavar="PATH",
                   help="dow_gate_<SYMBOL>_<TF>.json from stage 4.5 "
                        "(backtest/dow_gate.py). Omitted, it is looked up at "
                        "the conventional path for THIS pair; a missing file "
                        "records `day_of_week_gate.status: NOT EVALUATED` "
                        "rather than an empty block list, because 'nobody "
                        "looked' and 'every session cleared' are different "
                        "facts and the live loop keeps them apart.")
    p.add_argument("--no-commit", action="store_true",
                   help="Write the files but do not touch git")
    p.add_argument("--portfolio", default=None,
                   help="Incubator portfolio to register onto (e.g. "
                        "incubator-odd). Matched ignoring case and "
                        "punctuation. Without it the target is whichever "
                        "incubator portfolio holds FEWER active strategies, "
                        "ties broken alphabetically — round-robin across "
                        "successive promotions. The prop track is refused: "
                        "graduating there is decided on forward paper trades "
                        "by scripts/incubator_tracker.py --auto-promote.")
    p.add_argument("--allocation", type=int, default=DEFAULT_ALLOCATION,
                   help=f"Contracts recorded on the allocation (default "
                        f"{DEFAULT_ALLOCATION}). A DECLARATION: "
                        f"portfolio/volatility_sizer.py sizes from ATR "
                        f"against the portfolio's own budget and clamps, and "
                        f"nothing places an order off this number yet.")
    p.add_argument("--portfolios", default=None,
                   help=f"The routing table to register into (default: "
                        f"{PORTFOLIO_CONFIG})")
    p.add_argument("--no-register", action="store_true",
                   help="Promote without touching config/portfolios.json. "
                        "The strategy is staged and committed and stays "
                        "unallocated, which is what approved_incubator/ meant "
                        "before this flag's default became registration.")
    p.add_argument("--no-discord", action="store_true",
                   help="Never post the stage 5 card. Without it the card is "
                        "posted when a webhook is configured (see "
                        "mdlib/env.py) and skipped, quietly, when none is.")
    p.add_argument("--discord-dry-run", action="store_true",
                   help="Run the stage 5 card with --dry-run: it prints the "
                        "payload and sends nothing.")
    args = p.parse_args(argv)

    params = json.loads(args.params) if args.params else None

    # ---- resolve what the command line did not say -----------------------
    # `--strat` plus what Stage 3 already wrote is enough to promote. Each
    # value NAMES the file it came from and is printed below, because a
    # resolved argument is only better than a typed one while it stays
    # traceable to the artifact that supplied it. A resolution that cannot be
    # made unambiguously RAISES here rather than picking one - see
    # `resolve_audit_file`, which refuses a directory holding several
    # certifications instead of promoting whichever sorted first.
    try:
        source, source_basis = resolve_source(args.strat, args.source)
        audit_path, audit_basis = resolve_audit_file(
            args.strat, args.symbol, args.timeframe, args.audit_file)
        version, version_basis = resolve_version(audit_path, args.version)
    except (FileNotFoundError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print("Resolving the promotion:")
    print(f"  source         {source}")
    print(f"                 ({source_basis})")
    print(f"  version        {version}")
    print(f"                 ({version_basis})")
    print(f"  audit file     {audit_path or 'NOT RESOLVED'}")
    print(f"                 ({audit_basis})")
    print()

    out = promote(
        strat=args.strat, version=version, source=source,
        metrics_path=Path(args.metrics) if args.metrics else None,
        audit_path=audit_path,
        symbol=args.symbol, timeframe=args.timeframe, params=params,
        threshold=args.threshold, notes=args.notes,
        variants_tested=args.variants_tested, force=args.force,
        require_certification=args.require_certification,
        dow_gate=Path(args.dow_gate) if args.dow_gate else None,
        commit=not args.no_commit)

    meta = out["meta"]
    promoted_id = out["strategy_id"]
    print(f"Promoted {promoted_id} Version {meta['version']} → {out['dir']}")
    if promoted_id != args.strat:
        print(f"  strategy id    {promoted_id}  (one certified pair: "
              f"{out['symbol']} {out['timeframe']})")
    for f in out["files"]:
        print(f"  wrote {f.relative_to(REPO_ROOT)}")
    print(f"  symbols        {meta['symbols'] or 'NOT RECORDED'}")
    print(f"  timeframe      {meta['timeframe'] or 'NOT RECORDED'}")
    print(f"  params         {meta['params'] or '{}'}")
    print(f"                 ({meta['params_source']})")
    if meta["params_conflict"]:
        print(f"\n  ! The metrics snapshot and the certification disagree on "
              f"{', '.join(meta['params_conflict'])}.\n"
              f"    The CERTIFIED values are recorded; the locked metrics were "
              f"measured on the\n    others, so they describe a run of "
              f"different parameters than the ones promoted.")
    risk = meta["risk"]
    print("  risk           "
          + ", ".join(f"{k}={'no take-profit modelled' if k == 'tp_atr_mult' and v is None else v}"
                      for k, v in risk.items()))
    # THE BLOCKED WEEKDAY, ON THE PROMOTION LINE. It is a standing
    # restriction on a live account and the operator reading this output is
    # the last human in the chain; printed nowhere, the first anybody would
    # know of it is a Friday with no orders.
    dow = meta["day_of_week_gate"]
    if dow.get("blocked_weekdays"):
        print(f"  day-of-week    BLOCKED {dow['blocked_day_name']} "
              f"(weekday {dow['blocked_weekday']}, Version "
              f"{dow.get('version')}) — {dow.get('reason', '')}")
    elif dow.get("status") == "EVALUATED":
        print(f"  day-of-week    no session blocked for Version "
              f"{dow.get('version')} "
              f"(worst was {dow.get('worst_day') or 'not identified'})")
    else:
        print(f"  day-of-week    {dow.get('status')} — {dow.get('reason', '')}")
    print(f"  metrics        {meta['metrics_status']}")
    print(f"  gate audit     {meta['gate_audit_status']}"
          + ("  (OVERRIDDEN with --force)" if meta["gates_overridden"] else ""))
    cert = meta["certification"]
    if isinstance(cert, dict):
        print(f"  certified by   {Path(cert['audit_file']).name}  "
              f"({cert.get('audit_symbol') or 'symbol not recorded'})")
        ho = cert.get("holdout") or {}
        print(f"                 holdout {ho.get('start', '?')} → "
              f"{ho.get('end', '?')}, sha {cert['audit_sha256'][:12]}")
        if cert.get("agrees_with_snapshot") is False:
            print("\n  ! The stage 3 certification and the metrics snapshot's "
                  "own gate block\n    disagree. The certification is what was "
                  "enforced; the snapshot's\n    block can only ever evaluate "
                  "Gate 1.")
    else:
        print("  certified by   NOT CERTIFIED")
        print("\n  ! No --audit-file was supplied, so no stage 3 gate "
              "certification is\n    recorded. Gates 2 and 3 cannot be "
              "evaluated by any single run — a\n    promotion without one "
              "rests on Gate 1 alone.")

    if meta["metrics_status"] == "NOT RECORDED":
        print("\n  ! No metrics snapshot was supplied, so meta.json records "
              "none.\n    Pass --metrics <dual_metrics.json> to lock the "
              "numbers this\n    promotion was decided on.")
    if out["warnings"]:
        print("\n  ! Audit warnings on the source module:")
        for w in out["warnings"]:
            print(f"      {w}")

    if out["committed"]:
        print(f"\nCommitted:\n{out['commit_output']}")
    else:
        print(f"\nNot committed: {out['commit_output'] or 'skipped (--no-commit)'}")
        print(f"  git add {out['dir'].relative_to(REPO_ROOT)} && git commit")

    # ---- register onto an incubator portfolio ---------------------------
    # AFTER the directory is written and committed, never before. The record
    # points at `approved_incubator/<strat>/strat.py` and grants a live loop
    # permission to load it; writing that permission first would name a file
    # that does not exist yet, and the window is exactly as long as a promotion
    # that then fails.
    registration = None
    if args.no_register:
        print("\n  registration    SKIPPED (--no-register). "
              "config/portfolios.json is unchanged, so this strategy is "
              "staged and\n                  unallocated — "
              "approved_incubator/ is a record that a version was chosen, "
              "not\n                  permission to trade it.")
    else:
        cfg = Path(args.portfolios) if args.portfolios else PORTFOLIO_CONFIG
        scope = certified_scope(meta.get("certification"), meta, audit_path)
        # An explicit --symbol/--timeframe outranks the certification, the
        # same way --params does: an operator correcting the record on
        # purpose beats a file. It is recorded as the source so the override
        # is never silent.
        if args.symbol:
            scope["symbol"] = args.symbol
            scope["resolved_from"]["symbol"] = "--symbol"
        if args.timeframe:
            scope["timeframe"] = args.timeframe
            scope["resolved_from"]["timeframe"] = "--timeframe"
        try:
            registration = register_portfolio(
                promoted_id, version=meta["version"], scope=scope,
                allocation=args.allocation, portfolio=args.portfolio,
                config_path=cfg)
        except (OSError, ValueError) as exc:
            # Reported, not raised, for git_commit's reason: the promotion is
            # written and committed, and it is not undone because the routing
            # table could not be updated. The operator is told what to run.
            print(f"\n  ! Portfolio registration FAILED: {exc}")
            print(f"    The promotion stands. Re-run registration with:\n"
                  f"      python3 backtest/promote.py --strat {args.strat} "  # noqa: E501
                  f"--version {meta['version']} \\\n"
                  f"          --source {source} --portfolio "
                  f"<incubator-odd|incubator-even>")
        else:
            rec = registration["record"]
            print(f"\n  registered      {registration['portfolio_id']}  "
                  f"({registration['basis']})")
            print(f"                  {rec['symbol']} {rec['timeframe']} "
                  f"version {rec['version']} · {rec['allocation']} contract"
                  f"{'s' if rec['allocation'] != 1 else ''} · regime "
                  f"{rec['regime_filter']}")
            print(f"                  status {rec['status']} · "
                  f"{registration['config_path'].name} "
                  f"(active_strategies + {ALLOCATIONS_KEY})")
            for field, where in sorted(rec["resolved_from"].items()):
                print(f"                    {field:<12} {where}")
            for gone in registration["retired"]:
                print(f"                  RETIRED {gone} — the bare "
                      f"registration this pair's id supersedes")
            if registration["moved_from"]:
                print(f"                  MOVED from "
                      f"{', '.join(registration['moved_from'])} — one "
                      f"strategy on two portfolios of one track would size "
                      f"the same signal twice")
            for note in registration["notes"]:
                print(f"\n  ! {note}")

    # ---- refresh the strategy tag manifest --------------------------------
    # A registration changes which strategies can contribute to a netted
    # position, and therefore which CrossTrade locks this account can take
    # out. The manifest a journal pre-registers those locks from is stale the
    # moment `active_strategies` changes, so it is regenerated here.
    #
    # AFTER the registration and NEVER FATAL. `export` does not raise: a
    # promotion that succeeded and then reported an error would leave an
    # operator unsure whether the strategy was registered, and the manifest is
    # a convenience for a journal rather than part of the promotion. Imported
    # inside the branch so a `--no-register` run does not pay for a module
    # that pulls in the live dispatcher.
    if not args.no_register:
        from scripts.strategy_tag_manifest import export   # noqa: PLC0415
        exported = export(config_path=str(
            Path(args.portfolios) if args.portfolios else PORTFOLIO_CONFIG))
        if exported["ok"]:
            print(f"\n  tag manifest    {exported['path']} "
                  f"({exported['singleton_tags']} tag(s) over "
                  f"{exported['rows']} pair(s))")
        else:
            print(f"\n  ! tag manifest NOT refreshed: {exported['error']}")
            print(f"    The promotion stands. Regenerate with: strat-tags --out")

    # ---- the stage 5 card ------------------------------------------------
    if args.no_discord:
        pass
    elif discord_webhook() is None:
        print("\n  stage 5 card    not posted: no webhook configured "
              "(BT_DISCORD_WEBHOOK / DISCORD_WEBHOOK_URL / DISCORD_WEBHOOK)")
    else:
        card = post_stage5_card(promoted_id, dry_run=args.discord_dry_run)
        if card["posted"]:
            print(f"\n  stage 5 card    "
                  f"{'rendered (--dry-run, nothing sent)' if args.discord_dry_run else 'posted'}")
        else:
            print(f"\n  ! The stage 5 card did not post "
                  f"(exit {card['returncode']}). The promotion stands.")
            if card["output"]:
                for line in card["output"].splitlines()[-6:]:
                    print(f"      {line}")
            print("    Re-run: " + " ".join(card["cmd"]))

    print("\nIn approved_incubator/ means under evaluation, not cleared to "
          "trade.\nDeployment needs the gates in docs/STRATEGY_DEVELOPMENT.md "
          "and the\n3-year holdout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
