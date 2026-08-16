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

import argparse
import ast
import hashlib
import json
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

from agents.tier3_workers import apply_ml_signal_filter
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


def signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, pd.Series]:
    """
    Version B: the baseline's signals, with losing entries suppressed.

    `bars` is ONE symbol's frame, oldest to newest - the engine calls this once
    per symbol. The symbol is read from the frame when the lake reader put it
    there and falls back to the promoted SYMBOLS entry, because it decides the
    contract multiplier, tick size and commission the filter's training labels
    are net of. Without it the classifier learns from gross outcomes and keeps
    trades that lose money after costs.
    """
    threshold = params.pop("threshold", ML_THRESHOLD)
    cfg = params.pop("cfg", None) or BacktestConfig()

    merged = dict(DEFAULT_PARAMS)
    merged.update(params)

    entries, exits = _baseline().signal_fn(bars, **merged)

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

    entries, exits = apply_ml_signal_filter(
        bars, entries, exits, symbol=symbol, cfg=cfg, threshold=threshold)
    return (pd.Series(entries).fillna(False).astype(bool),
            pd.Series(exits).fillna(False).astype(bool))


def make_signal_fn(**params):
    """Bind parameters for `agents.tier3_workers.load_strategy`."""
    def _bound(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
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
            symbol: str | None = None,
            timeframe: str | None = None,
            params: dict | None = None,
            threshold: float = 0.50,
            notes: str = "",
            variants_tested: int | None = None,
            force: bool = False,
            commit: bool = True,
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

    gate_status = (gate_audit or {}).get("status", "NOT EVALUATED")
    if gate_audit and gate_status != "PASS" and not force:
        raise SystemExit(
            f"Version {version} gate audit is {gate_status}, not PASS. "
            f"Promotion refused.\n"
            f"  Re-run with --force to promote anyway; the override is "
            f"recorded in meta.json.")

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
        "gates_overridden": bool(force and gate_audit and gate_status != "PASS"),
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
    args = p.parse_args(argv)

    params = json.loads(args.params) if args.params else None

    out = promote(
        strat=args.strat, version=args.version, source=Path(args.source),
        metrics_path=Path(args.metrics) if args.metrics else None,
        symbol=args.symbol, timeframe=args.timeframe, params=params,
        threshold=args.threshold, notes=args.notes,
        variants_tested=args.variants_tested, force=args.force,
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

    print("\nIn approved_incubator/ means under evaluation, not cleared to "
          "trade.\nDeployment needs the gates in docs/STRATEGY_DEVELOPMENT.md "
          "and the\n3-year holdout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
