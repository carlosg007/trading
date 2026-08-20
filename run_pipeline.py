#!/usr/bin/env python3
"""
run_pipeline.py - the master orchestrator for the institutional backtest stages.

Location:  ~/src/trading/run_pipeline.py

Runs Stage 1 -> Stage 2 -> Stage 3 in order, promotes the strategy module, and
optionally posts the promotion card to Discord. Each stage is a SUBPROCESS: the
stages are the implementations, and re-deriving any part of what they do here
would create a second implementation free to disagree with the first.

Date anchors are fixed in this file
-----------------------------------
`IS_START`/`IS_END`/`OOS_START`/`OOS_END` are module constants, not CLI flags,
which is the point of the script. The in-sample window ends 2022-12-31 and the
holdout begins 2023-01-01, so they do not overlap by construction - an
in-sample window that runs into the holdout spends it before Gate 3 is ever
evaluated, and that is the one mistake nothing downstream can detect.

Read this before trusting the promotion step
--------------------------------------------
**Stage 3 exits 0 when nothing is certified.** `audit_gates.py` returns
`1 if errors else 0`, where `errors` counts raised EXCEPTIONS - a contract whose
gates came back FAIL, or NOT EVALUATED, is a successful run reporting a
negative result. So "promote if no command returned non-zero" would stamp
`PROMOTED_TO_LIVE` on a strategy Stage 3 had just declared NOT CERTIFIED, and
the pipeline would report success while doing it.

This script therefore READS Stage 3's own `gate_audit_<SYMBOL>.json` and
promotes only when `passed` is true for at least one version. That is not a
second gate implementation - it is the verdict Stage 3 already wrote, lifted out
of the file it wrote it to. `--force-promote` overrides it and the override is
recorded in `meta.json` as `gates_overridden: true`, because a promotion nobody
certified must at minimum say so on its own record.

What this is NOT
----------------
- **Not Stage 5.** `backtest/promote.py` is Stage 5: it writes a directory
  (`strat.py`, `baseline.py` for Version B, a full `meta.json` with the run's
  effective parameters, the risk block and the locked metrics snapshot) and
  commits it. The promotion here is the flat-file form this script was asked
  for; it copies the module and records a hash. Prefer `promote.py` for
  anything a human will later trade.
- **Not a substitute for reading the evidence.** The four-choice menu and the
  Stage 4 tear sheets exist so a person sees the numbers before anything is
  promoted. Automating past them is a decision, not a default.
- **Not to be run by Claude Code.** Per CLAUDE.md this drives real backtests
  against the lake. `--dry-run` prints every command and executes none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from backtest.pipeline import GATE_AUDIT_FILE, pipeline_dir   # noqa: E402

# --------------------------------------------------------------------------
# The date anchors. Fixed here on purpose - see the module docstring.
# --------------------------------------------------------------------------
IS_START = "2013-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2026-01-01"

EXPERIMENTAL_DIR = REPO_ROOT / "strategies" / "experimental"
INCUBATOR_DIR = REPO_ROOT / "strategies" / "approved_incubator"
REPORTER = REPO_ROOT / "backtest" / "discord_reporter.py"

# The stages run under the SAME interpreter as this script, not a bare
# `python3`. Outside an activated venv `python3` is /usr/bin/python3, which has
# neither pandas nor vectorbtpro, and the stage would die on import.
PYTHON = sys.executable or "python3"

# Placeholder tokens for the Discord card. Deliberately NOT plausible numbers:
# `discord_reporter.py` passes a non-numeric value through verbatim, so the card
# reads "NOT PARSED" rather than a fabricated 1.85 that would be indistinguish-
# able from a measured profit factor. Replaced when the JSON parsing lands.
PLACEHOLDER = "NOT PARSED (pending JSON extraction)"


# --------------------------------------------------------------------------
# stage execution
# --------------------------------------------------------------------------

def run_stage(label: str, cmd: list[str], dry_run: bool = False) -> None:
    """
    Run one stage. A non-zero exit stops the pipeline immediately.

    Nothing downstream of a failed stage is meaningful: Stage 2 sweeps Stage 1's
    survivors and Stage 3 certifies Stage 2's winner, so continuing past a
    failure certifies parameters that were never selected.
    """
    printable = " ".join(cmd)
    print("\n" + "=" * 78)
    print(f"  {label}")
    print("=" * 78)
    print(f"  $ {printable}\n", flush=True)

    if dry_run:
        print(f"  DRY RUN  {label} not executed.")
        return

    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        print(f"\n[!] {label} FAILED with exit code {completed.returncode}.",
              file=sys.stderr)
        print("    Pipeline halted. Nothing was promoted and no alert was sent.",
              file=sys.stderr)
        sys.exit(completed.returncode)
    print(f"\n  {label} OK.")


def first_timeframe(tf: str) -> str:
    """
    Stages 1 and 2 accept a comma-separated `--tf`; Stages 3 and 4 REFUSE one -
    a gate audit certifies exactly one (parameters, timeframe) pair. Take the
    first and say so, rather than letting Stage 3 raise on the full list.
    """
    return tf.split(",")[0].strip()


def first_symbol(symbols: str) -> str:
    return symbols.split(",")[0].strip()


# --------------------------------------------------------------------------
# certification, read back from Stage 3's own artifact
# --------------------------------------------------------------------------

def read_certification(strat: str, symbol: str) -> tuple[bool, str]:
    """
    Return (certified, explanation) from `gate_audit_<SYMBOL>.json`.

    Does NOT re-evaluate a gate. It reads the `passed` map Stage 3 lifted out of
    the nested audit for exactly this purpose. A missing file is NOT certified -
    an absent verdict is not a pass.
    """
    audit_path = pipeline_dir(strat) / GATE_AUDIT_FILE.format(symbol=symbol)
    if not audit_path.exists():
        return False, f"no gate audit at {audit_path}"

    try:
        payload = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable gate audit {audit_path}: {exc}"

    # write_stage nests the stage payload; tolerate either shape.
    body = payload.get("payload", payload)
    passed = body.get("passed") or {}
    status = body.get("status") or {}
    if not isinstance(passed, dict) or not passed:
        return False, f"gate audit {audit_path.name} records no verdict"

    winners = [ver for ver, ok in passed.items() if ok]
    detail = ", ".join(f"{v}={status.get(v, 'UNKNOWN')}" for v in sorted(passed))
    if winners:
        return True, f"CERTIFIED ({detail})"
    return False, f"NOT CERTIFIED ({detail})"


# --------------------------------------------------------------------------
# promotion
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def promote(strat: str, symbol: str, tf: str, certified: bool,
            reason: str, forced: bool, dry_run: bool) -> Path:
    """
    Copy the module into the incubator byte for byte and write `meta.json`.

    Copied verbatim, never tidied on the way through: the recorded SHA-256 is
    what makes the promoted file provably the file that was backtested, and a
    strategy cleaned up in transit is a different strategy.
    """
    source = EXPERIMENTAL_DIR / f"{strat}.py"
    if not source.exists():
        # In a dry run this is a warning, not a stop: the point of --dry-run is
        # to see the whole plan, and aborting here would hide the alert step.
        if dry_run:
            print(f"\n[!] DRY RUN: no strategy module at {source}. A real run "
                  f"would stop here.", file=sys.stderr)
            return pipeline_dir(strat) / "meta.json"
        print(f"\n[!] No strategy module at {source}. Nothing promoted.",
              file=sys.stderr)
        sys.exit(1)

    digest = sha256_file(source)
    dest = INCUBATOR_DIR / f"{strat}.py"
    meta_dir = pipeline_dir(strat)
    meta_path = meta_dir / "meta.json"

    meta = {
        "strategy": strat,
        "symbol": symbol,
        "timeframe": tf,
        "certified_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": digest,
        "source": str(source),
        "promoted_to": str(dest),
        "status": "PROMOTED_TO_LIVE",
        # The window every number behind this promotion came from. A status
        # without its dates cannot be re-derived later.
        "in_sample": {"start": IS_START, "end": IS_END},
        "holdout": {"start": OOS_START, "end": OOS_END},
        # Whether Stage 3 actually certified it, kept BESIDE the status rather
        # than folded into it. `gates_overridden: true` beside
        # `PROMOTED_TO_LIVE` is the record that a human forced this.
        "certification": reason,
        "gates_overridden": bool(forced and not certified),
        "promoted_by": "run_pipeline.py",
        "note": ("Flat-file promotion. backtest/promote.py is Stage 5 and "
                 "writes the full incubator directory with the run's effective "
                 "parameters, risk block and locked metrics snapshot."),
    }

    print("\n" + "=" * 78)
    print("  PROMOTION")
    print("=" * 78)
    print(f"  source     : {source}")
    print(f"  sha256     : {digest}")
    print(f"  dest       : {dest}")
    print(f"  meta.json  : {meta_path}")

    if dry_run:
        print("\n  DRY RUN  nothing copied, nothing written.")
        return meta_path

    INCUBATOR_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    meta_dir.mkdir(parents=True, exist_ok=True)
    # Atomic, matching how every other handoff in the pipeline is written: a
    # half-written meta.json beside a copied module is a promotion whose record
    # cannot be read.
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(meta_path)
    print("\n  Promoted. Being in approved_incubator/ is a record that a "
          "version was\n  chosen - it is NOT permission to trade it.")
    return meta_path


# --------------------------------------------------------------------------
# discord
# --------------------------------------------------------------------------

def send_alert(webhook: str, strat: str, symbol: str, tf: str,
               artifact_dir: Path, dry_run: bool) -> None:
    cmd = [
        PYTHON, str(REPORTER),
        "--webhook", webhook,
        "--strat", strat,
        "--symbol", symbol,
        "--tf", tf,
        "--pf", PLACEHOLDER,
        "--dd", PLACEHOLDER,
        "--regime", PLACEHOLDER,
        "--report", str(artifact_dir),
    ]
    print("\n" + "=" * 78)
    print("  DISCORD ALERT")
    print("=" * 78)
    # The webhook is a credential; print the command with it masked.
    masked = [("<webhook>" if c == webhook else c) for c in cmd]
    print(f"  $ {' '.join(masked)}\n", flush=True)

    if dry_run:
        print("  DRY RUN  no alert sent.")
        return

    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        # A failed notification does NOT fail the pipeline: the promotion has
        # already happened and is on disk. Exiting non-zero here would report a
        # completed promotion as a failed one.
        print("\n[!] Discord alert failed. The promotion above still stands.",
              file=sys.stderr)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Master orchestrator: Stage 1 -> 2 -> 3, promote, alert.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"In-sample {IS_START}..{IS_END}   holdout {OOS_START}..{OOS_END}\n"
            "The anchors are fixed in the script and are not overridable."
        ),
    )
    p.add_argument("--strat", required=True, help="Strategy name, e.g. sma_momentum_crossover")
    p.add_argument("--symbols", required=True, help="One (NQ), a list (NQ,ES), or ALL")
    p.add_argument("--tf", required=True,
                   help="Timeframe(s). Stages 1-2 take a list; Stage 3 uses the FIRST.")
    p.add_argument("--webhook", default=None,
                   help="Discord webhook. Omitted means no alert is sent.")
    p.add_argument("--force-promote", action="store_true",
                   help="Promote even when Stage 3 did not certify. Recorded "
                        "in meta.json as gates_overridden.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every command and execute none. Nothing is run, "
                        "copied, written or sent.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    strat = args.strat
    symbols = args.symbols
    tf_all = args.tf
    tf_one = first_timeframe(tf_all)
    symbol_one = first_symbol(symbols)

    print("=" * 78)
    print(f"  PIPELINE · {strat}")
    print("=" * 78)
    print(f"  symbols    : {symbols}")
    print(f"  timeframes : {tf_all}   (Stage 3 certifies {tf_one})")
    print(f"  in-sample  : {IS_START} → {IS_END}")
    print(f"  holdout    : {OOS_START} → {OOS_END}   (untouched until Stage 3)")
    print(f"  interpreter: {PYTHON}")
    if args.dry_run:
        print("  MODE       : DRY RUN — nothing will be executed.")

    run_stage("STAGE 1 · baseline (regime-aware screening firewall)", [
        PYTHON, "backtest/baseline.py",
        "--strat", strat,
        "--symbols", symbols,
        "--tf", tf_all,
        "--start", IS_START,
        "--end", IS_END,
        "--ml",
    ], args.dry_run)

    run_stage("STAGE 2 · scan (parameter sweep over the survivors)", [
        PYTHON, "backtest/scan.py",
        "--strat", strat,
        "--symbols", symbols,
        "--tf", tf_all,
        "--start", IS_START,
        "--end", IS_END,
    ], args.dry_run)

    run_stage("STAGE 3 · audit_gates (certification against the holdout)", [
        PYTHON, "backtest/audit_gates.py",
        "--strat", strat,
        "--symbols", symbols,
        "--tf", tf_one,
        "--is-start", IS_START,
        "--is-end", IS_END,
        "--holdout-start", OOS_START,
        "--holdout-end", OOS_END,
    ], args.dry_run)

    # Stage 3 exits 0 whether or not anything was certified, so the exit codes
    # above are NOT evidence of a pass. Read the verdict it wrote.
    if args.dry_run:
        certified, reason = False, "DRY RUN — no gate audit read"
    else:
        certified, reason = read_certification(strat, symbol_one)
    print("\n" + "=" * 78)
    print("  CERTIFICATION CHECK")
    print("=" * 78)
    print(f"  {symbol_one}: {reason}")

    if not certified and not args.force_promote and not args.dry_run:
        print("\n[!] Stage 3 did not certify this strategy. Nothing promoted, "
              "no alert sent.", file=sys.stderr)
        print("    A FAIL and a NOT EVALUATED are both refusals. Re-run with "
              "--force-promote\n    to override, which is recorded in "
              "meta.json as gates_overridden.", file=sys.stderr)
        return 1
    if not certified and args.force_promote:
        print("\n  [!] OVERRIDDEN — promoting an uncertified strategy because "
              "--force-promote\n      was passed. This is recorded in meta.json.")

    promote(strat, symbol_one, tf_one, certified, reason,
            args.force_promote, args.dry_run)

    if args.webhook:
        send_alert(args.webhook, strat, symbol_one, tf_one,
                   pipeline_dir(strat), args.dry_run)
    else:
        print("\n  No --webhook given; no Discord alert sent.")

    print("\n" + "=" * 78)
    print("  PIPELINE COMPLETE")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
