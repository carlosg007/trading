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
    """
    if path is None:
        return None, None, "NOT RECORDED"
    if not path.exists():
        raise FileNotFoundError(f"metrics snapshot not found: {path}")

    blob = json.loads(path.read_text(encoding="utf-8"))
    key = "version_a" if version.upper() == "A" else "version_b"
    block = blob.get(key)
    if isinstance(block, dict) and "metrics" in block:
        return block.get("metrics"), block.get("gate_audit"), f"locked from {path.name}"
    if "sharpe" in blob:
        return blob, blob.get("gate_audit"), f"locked from {path.name}"
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

    dest = Path(incubator) / strat
    dest.mkdir(parents=True, exist_ok=True)

    symbols = ([symbol] if symbol else list(info["symbols"] or []))
    tf = timeframe or info["timeframe"]

    # Three layers, weakest first. The middle one is the important addition:
    # under `--scan` the run's parameters are the winning grid cell, not the
    # module's DEFAULT_PARAMS, and promoting the defaults beside that run's
    # metrics would record a strategy nobody backtested. `--params` still wins,
    # because an operator correcting the record on purpose outranks a file.
    from_snapshot = snapshot_params(metrics)
    merged_params = dict(info["params"] or {})
    merged_params.update(from_snapshot)
    merged_params.update(params or {})

    if from_snapshot:
        params_source = f"locked from {metrics_path.name}"
    elif params:
        params_source = "--params"
    else:
        params_source = f"{source.name} DEFAULT_PARAMS"

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

    warnings = audit_notes(source)

    meta = {
        "name": strat,
        "version": version,
        "description": (f"Promoted Version {version} "
                        f"({'rule-based baseline' if version == 'A' else 'baseline + causal ML filter'})."),
        "symbols": symbols,
        "timeframe": tf,
        "params": merged_params,
        "params_source": params_source,
        # The stop, the target and the trailing flag the promoted numbers were
        # earned under, repeated where they can be found without knowing what
        # this strategy called its periods. `NOT DECLARED` means the strategy
        # has no such parameter; a literal null under `tp_atr_mult` means it
        # has one and this run modelled no take-profit. Those are different
        # facts about what a live account would be running.
        "risk": risk_settings(merged_params),
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
           "warnings": warnings, "committed": False, "commit_output": ""}

    if commit:
        out.update(git_commit(dest, strat, version, gate_status))
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


def resolve_portfolio(portfolios: dict,
                      requested: str | None = None) -> tuple[str, str]:
    """
    Which incubator portfolio this promotion is registered onto, and why.

    Returns `(portfolio_id, basis)`. The basis is recorded on the allocation
    and printed, because "which account is this strategy on" answered by a rule
    nobody wrote down is exactly what `portfolio.config_loader` refuses to do
    when it declines to infer a portfolio from a strategy's name.

    An explicit `--portfolio` wins outright and is matched case- and
    punctuation-insensitively. Without one the target is the incubator
    portfolio holding the FEWER active strategies, ties broken alphabetically -
    which is round-robin across successive promotions, because the portfolio
    that just took one is the one with more next time. It is a deterministic
    rule and that is the point: two operators promoting the same strategy get
    the same account, and a promotion that is re-run does not land somewhere
    else.

    THE REGIME SCOPE IS NOT PART OF THIS CHOICE. A portfolio's
    `basket.regime_quadrants` is a permission held by the ACCOUNT, shared by
    every strategy on it; routing a promotion to whichever account happened to
    declare the certified quadrant would make the account a property of one
    strategy's Stage 1 designation. The certified quadrant is recorded on the
    allocation as `regime_filter` and reconciled against the account's declared
    quadrants at load time - see `register_portfolio`.
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

    counts = {pid: len([s for s in (portfolios[pid].get("active_strategies")
                                    or [])])
              for pid in candidates}
    pid = sorted(candidates, key=lambda p: (counts[p], p))[0]
    tally = ", ".join(f"{p}={counts[p]}" for p in candidates)
    return pid, (f"fewest active strategies ({tally}), ties alphabetical")


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
    blob = json.loads(path.read_text(encoding="utf-8"))
    portfolios = blob.get("portfolios")
    if not isinstance(portfolios, dict) or not portfolios:
        raise ValueError(f"{path} carries no `portfolios` object")

    pid, basis = resolve_portfolio(portfolios, portfolio)
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
    notes: list[str] = []

    # Remove the id and any stale allocation record from every OTHER portfolio
    # on this track. Leaving one behind is the double-sizing config the loader
    # refuses, and it would be refused on the next load rather than here.
    moved_from: list[str] = []
    for other in incubator_portfolios(portfolios):
        if other == pid:
            continue
        block = portfolios[other]
        names = block.get("active_strategies")
        if isinstance(names, list):
            kept = [n for n in names if _normalise_pid(n) != wanted]
            if len(kept) != len(names):
                block["active_strategies"] = kept
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

    if record["symbol"] == NOT_RESOLVED or record["timeframe"] == NOT_RESOLVED:
        notes.append(
            "the certified contract or timeframe could not be resolved, so "
            "the allocation records NOT RESOLVED rather than the module's "
            "declarations. Pass --audit-file from stage 3, or --symbol / "
            "--timeframe to state the pair by hand.")

    _write_portfolio_config(path, blob)
    return {"portfolio_id": pid, "basis": basis, "record": record,
            "config_path": path, "moved_from": moved_from,
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
    p.add_argument("--version", required=True, choices=["A", "B", "a", "b"],
                   help="A = rule-based baseline, B = baseline + ML filter")
    p.add_argument("--source", required=True,
                   help="Path to the experimental strategy module")
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
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Version B: P(win) at or above which an entry is kept")
    p.add_argument("--variants-tested", type=int, default=None,
                   help="How many variants were tried to reach this result")
    p.add_argument("--notes", default="", help="Free text for meta.json")
    p.add_argument("--force", action="store_true",
                   help="Promote even when the gate audit did not pass. "
                        "Recorded in meta.json.")
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

    out = promote(
        strat=args.strat, version=args.version, source=Path(args.source),
        metrics_path=Path(args.metrics) if args.metrics else None,
        audit_path=Path(args.audit_file) if args.audit_file else None,
        symbol=args.symbol, timeframe=args.timeframe, params=params,
        threshold=args.threshold, notes=args.notes,
        variants_tested=args.variants_tested, force=args.force,
        require_certification=args.require_certification,
        commit=not args.no_commit)

    meta = out["meta"]
    print(f"Promoted {args.strat} Version {meta['version']} → {out['dir']}")
    for f in out["files"]:
        print(f"  wrote {f.relative_to(REPO_ROOT)}")
    print(f"  symbols        {meta['symbols'] or 'NOT RECORDED'}")
    print(f"  timeframe      {meta['timeframe'] or 'NOT RECORDED'}")
    print(f"  params         {meta['params'] or '{}'}")
    print(f"                 ({meta['params_source']})")
    risk = meta["risk"]
    print("  risk           "
          + ", ".join(f"{k}={'no take-profit modelled' if k == 'tp_atr_mult' and v is None else v}"
                      for k, v in risk.items()))
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
        scope = certified_scope(meta.get("certification"), meta,
                                Path(args.audit_file) if args.audit_file
                                else None)
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
                args.strat, version=meta["version"], scope=scope,
                allocation=args.allocation, portfolio=args.portfolio,
                config_path=cfg)
        except (OSError, ValueError) as exc:
            # Reported, not raised, for git_commit's reason: the promotion is
            # written and committed, and it is not undone because the routing
            # table could not be updated. The operator is told what to run.
            print(f"\n  ! Portfolio registration FAILED: {exc}")
            print(f"    The promotion stands. Re-run registration with:\n"
                  f"      python3 backtest/promote.py --strat {args.strat} "
                  f"--version {meta['version']} \\\n"
                  f"          --source {args.source} --portfolio "
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
            if registration["moved_from"]:
                print(f"                  MOVED from "
                      f"{', '.join(registration['moved_from'])} — one "
                      f"strategy on two portfolios of one track would size "
                      f"the same signal twice")
            for note in registration["notes"]:
                print(f"\n  ! {note}")

    # ---- the stage 5 card ------------------------------------------------
    if args.no_discord:
        pass
    elif discord_webhook() is None:
        print("\n  stage 5 card    not posted: no webhook configured "
              "(BT_DISCORD_WEBHOOK / DISCORD_WEBHOOK_URL / DISCORD_WEBHOOK)")
    else:
        card = post_stage5_card(args.strat, dry_run=args.discord_dry_run)
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
