"""
backtest/audit_gates.py - STAGE 3 of 5: certification. The gates, on the record.

Location: ~/src/trading/backtest/audit_gates.py

The only stage that produces a gate verdict anybody may act on. It locks the
parameters Stage 2 selected, runs them once over the untouched holdout, and
writes one `gate_audit_<SYMBOL>.json` per contract carrying the official
PASS/FAIL - plus `stage3_audit_summary.json` over the whole run.

**The summary MERGES across timeframes** (2026-08-21). This stage certifies one
timeframe per invocation, so a campaign that audits 5m, 15m and 30m runs it
three times into that one file. Written as a plain overwrite, it kept only the
last: three certifications existed as `gate_audit_<SYMBOL>_<TF>.json` on disk
and the Discord card, which reads the summary, announced one of them. Each run
now replaces its OWN timeframe's rows and carries every other timeframe's
forward verbatim, `runs` records what each invocation covered, and `audits` is
the consolidated index over the per-pair files - which remain the authoritative
verdict a promotion rests on. `--rebuild-summary` reconstructs that index from
the audits already on disk, reading no bars and re-scoring no gate, for the
campaigns the old overwrite already flattened.

    python3 backtest/audit_gates.py --strat ema_trend_filter --tf 15m

    Gate R  REGIME        the certification. Generalization of the edge inside
                          the quadrant Stage 1 designated, measured on the
                          HOLDOUT: PF >= 1.00 over >= 30 trades in that
                          quadrant, and nowhere else.
    Gate 1  in-sample     profit factor, trade count, max drawdown   EVIDENCE
    Gate 2  robustness    walk-forward efficiency, MC 95% drawdown   EVIDENCE
    Gate 3  OOS holdout   retention of the in-sample result          EVIDENCE

The Regime-Switching Incubator Charter binds this stage (2026-08-21)
--------------------------------------------------------------------
All five clauses are enforced in the module rather than left to how the
command was typed, for the same reason Stage 2's are: a charter honoured only
by convention is honoured until somebody is in a hurry.

**Its input is Stage 2's artifacts, and the parameters are LOCKED.** With no
`--symbols`, targets come from `stage2_summary.json` as exact (symbol,
timeframe) pairs - not from a glob of whatever `best_params_*.json` files
happen to be in the directory, which would certify a stale winner from a
superseded sweep with nothing raising. Each pair's parameters are read from
`best_params_<SYMBOL>_<TF>.json` and bound verbatim. **Nothing is re-tuned
here.** `--param` still overrides, because an operator correcting the record
on purpose outranks a file, but it BREAKS THE LOCK and says so: the audit
records `params_locked: false` and names every key that was overridden. A
certification whose parameters were adjusted after the sweep is a
certification of a strategy that was never optimised.

**The verdict is the holdout, and only the holdout.** `--holdout-end` defaults
to the present rather than to a hardcoded year, so a certification run today
scores every bar the lake has. `check_windows` refuses an in-sample window
reaching `HOLDOUT_START` before a bar is read - see below.

**Nothing is pruned on an aggregate score.** This is the clause that changed
what Stage 3 IS. The certification is `Gate R` and nothing else: the
strategy's edge has to survive out of sample inside the ONE quadrant Stage 1
designated for it, at `MIN_REGIME_PROFIT_FACTOR` over `MIN_REGIME_TRADES` -
the same two bars Stage 1 screened on, imported from `backtest.baseline` so
the two can never drift apart. Gates 1, 2 and 3 are computed in full and
reported in full, and **they cannot fail a certification**. They measure the
BLENDED sample across every market state, and a strategy governed by a live
supervisor that stands it down outside its quadrant is not trading the
blended sample - failing it for a mediocre number there prunes on a result
nobody will ever realise. What they are is evidence a human reads next to the
verdict, which is why they are still on the card, on the leaderboard and in
the file.

**No prop-firm rules.** No daily loss limit, no trailing drawdown, no
consistency cap. Those are enforced by CrossTrade NAM against a live account
balance and have no meaning against a research equity curve; `_assert_no_prop_firm_rules`
refuses a config carrying them rather than trusting that nobody added one.

**Passing configurations are sealed and staged.** A PASS writes
`strategies/approved_incubator/<strategy>/` through `backtest.promote`, and
the `meta.json` it produces gains a `seal` block: SHA-256 of the strategy
code, of `best_params_<SYMBOL>_<TF>.json` and of `gate_audit_<SYMBOL>.json`.
Three hashes rather than one because the three can be separated - a promoted
module hashes to the code, but says nothing about which parameters or which
holdout produced the numbers beside it. **Nothing is git-committed here.**
Staging into the incubator is a record that a version was certified; the
commit, and the four-choice menu in front of a human, stay Stage 5's.

Each gate is a separate run, because each needs different bars
--------------------------------------------------------------
Gate 1 scores the in-sample window. Gate 2 rolls train/test folds forward
inside that same window and bootstraps the in-sample trade sequence. Gate 3
runs the held-back years once, at the end. A single backtest cannot produce
all three, which is why `run_dual_version_backtest` reports Gates 2 and 3 as
NOT EVALUATED and why that is not a pass.

**The holdout must not overlap the in-sample window, and this refuses to run
when it does.** That check is the one thing in this file that can invalidate
everything else in it: an in-sample window running into the holdout has spent
the holdout before Gate 3 is evaluated, and the retention ratio it computes is
a strategy scored against itself. It is checked before any bars are read, so
the run fails in a second rather than after an hour.

**Parameters come from Stage 2, not from the module.** Without
`best_params_<SYMBOL>.json` this would certify the defaults while the operator
believed it had certified the winner of the sweep. `--param` overrides
explicitly, and whichever source was used is recorded in the audit file next to
`variants_tested` - a certification that cannot say how many variants its
parameters were selected from is not one.

**The walk-forward runs WITHOUT a parameter grid by default.** With fixed
parameters it compares two time periods rather than fitted-versus-unseen, which
`run_walk_forward_analysis` says plainly and which is recorded in the audit as
`wfo_optimized: false`. Pass `--wfo-grid` to re-select per fold from the
module's PARAM_GRID; that is the version that says something about overfitting,
and it costs one full sweep per fold.

Version B
---------
`--ml` certifies the ML-filtered version alongside the baseline, and both
audits are written. The filter is refit inside each window it is scored on, so
Gate 3 for Version B is a genuine out-of-sample test of the pipeline rather
than of one fitted classifier.
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
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------


import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                   # noqa: E402
import json                                                       # noqa: E402
import sys                                                        # noqa: E402
import time                                                       # noqa: E402
import traceback                                                  # noqa: E402
from datetime import datetime, timezone                           # noqa: E402
from pathlib import Path                                          # noqa: E402

import pandas as pd                                               # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier1_master import run_dual_version_backtest          # noqa: E402
from agents.tier3_workers import (load_strategy,                   # noqa: E402
                                  run_monte_carlo_simulation,
                                  run_walk_forward_analysis,
                                  trade_returns_from_result)
from backtest.engine import BacktestConfig                         # noqa: E402
from backtest.event_calendar import (WEEKDAY_NAMES, add_filter_args,  # noqa: E402
                                     filter_config_kwargs)
# The two survival bars, imported from STAGE 1 rather than restated here.
# Gate R asks the same question Stage 1 asked - "is there an environment in
# which this works" - of different bars, so the hurdle has to be the same
# number. Two copies would let a screen at 1.00 feed a certification at 1.15,
# and the configurations lost in the gap would look like strategies that
# failed out of sample rather than strategies nobody ever certified at the bar
# they were screened on.
from backtest.baseline import (MIN_REGIME_PROFIT_FACTOR,           # noqa: E402
                               MIN_REGIME_TRADES, quadrant_id)
from backtest.pipeline import (ML_THRESHOLD_DEFAULT,
                               BEST_PARAMS_FILE, CHARTER_IS_END,   # noqa: E402
                               CHARTER_IS_START, GATE_AUDIT_FILE,
                               HOLDOUT_START, RUIN_MIN_DRAWDOWN_PCT,
                               STAGE2_SUMMARY_FILE,
                               STAGE3_SUMMARY_FILE, next_step, pipeline_dir,
                               read_stage, stage_banner, write_stage)
from backtest.profiler import (REGIME_TO_QUADRANT, REGIMES,        # noqa: E402
                               RegimeProfiler)
from backtest.promote import INCUBATOR, promote, sha256            # noqa: E402
from backtest.report import (FAIL, NOT_EVALUATED, PASS,            # noqa: E402
                             audit_acceptance_gates, criterion_text,
                             day_of_week_breakdown, format_day_of_week)
from backtest.run import (load_bars, parse_param, parse_symbols,    # noqa: E402
                          resolve_strategy)
from backtest.scan import (expand_grid,                          # noqa: E402
                          STAGE2_PRUNED_FRAGILE)

# Gate R's own name, kept beside the three it sits with so a reader of the
# audit file never has to guess which key holds the verdict.
GATE_R = "gate_regime"

# The prop-firm surfaces `BacktestConfig` still carries, and which this stage
# refuses to run under. Listed by NAME rather than checked inline, so adding a
# third one to the engine and forgetting it here is a one-line fix in an
# obvious place instead of a silent hole.
PROP_FIRM_FIELDS = ("trailing_drawdown_pct", "daily_loss_limit")


def discover_symbols(out_dir: Path, tf: str | None) -> list[str]:
    """
    Which contracts Stage 2 left parameters for.

    `best_params_NQ_15m.json` must resolve to the symbol `NQ`, not to
    `NQ_15m` — stripping the prefix alone would hand `load_bars` a symbol the
    lake has never heard of, one stage after the sweep succeeded. The timeframe
    suffix is matched explicitly when one is known, and otherwise only
    unsuffixed files are considered rather than guessing where a symbol name
    ends.
    """
    out_dir = Path(out_dir)
    found: set[str] = set()
    for p in out_dir.glob("best_params_*.json"):
        stem = p.stem[len("best_params_"):]
        if tf and stem.endswith(f"_{tf}"):
            found.add(stem[: -(len(tf) + 1)])
        elif "_" not in stem:
            found.add(stem)
    return sorted(found)


def default_stage2_summary(out_dir: Path) -> Path:
    """`<pipeline dir>/stage2_summary.json` — where Stage 2 left its matrix."""
    return Path(out_dir) / STAGE2_SUMMARY_FILE


def load_stage2_summary(strat_name: str, out_dir: Path,
                        path: str | Path | None = None) -> dict | None:
    """
    Stage 2's summary, or None when the stage was never run here.

    Read through `pipeline.read_stage`, so a file written by another stage or
    belonging to another strategy is REFUSED rather than certified: gates run
    against one strategy's parameters and reported under another's name is a
    mistake nothing downstream could detect, and this file is what decides
    which contracts get audited at all.

    A missing summary is not an error. `--symbols` still names contracts
    directly, and a single-contract Stage 2 run driven by hand leaves
    `best_params_<SYMBOL>.json` without a matrix beside it. What it costs is
    the regime scope and the coverage check, and the stage says so on the
    banner rather than proceeding as though it had them. An EXPLICIT
    `--stage2-summary` that does not exist IS an error - an operator naming a
    file meant that file.
    """
    if path is not None:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"stage 2 summary not found: {target}")
    else:
        target = default_stage2_summary(out_dir)
        if not target.exists():
            return None
    blob = read_stage(target, 2, strat_name)
    blob["_path"] = str(target)
    return blob


def stage2_targets(blob: dict | None, tf: str | None) -> list[dict]:
    """
    The EXACT (symbol, timeframe) pairs Stage 2 optimised, with their scope.

    Not the cross product of the symbol union and the timeframe union, for the
    same reason Stage 2 does not take Stage 1's survivors that way: the pairs
    are RAGGED - NQ optimised at 5m and 15m, GC at 15m only - and crossing the
    axes manufactures configurations no earlier stage ever screened or swept.
    Certifying one of those would produce a `gate_audit_<SYMBOL>.json` for a
    parameter set that does not exist, which fails as a missing file rather
    than as a wrong answer, but only because `load_params` refuses to guess.

    `tf` filters to ONE timeframe, because a gate audit certifies one
    (parameters, timeframe) pair. With no `tf` the caller gets every pair and
    is responsible for that choice.

    Rows that ERRORED in Stage 2 are RETURNED, carrying `certifiable: False`.
    Stage 2 prunes nothing, so a configuration whose sweep raised is still on
    its matrix with `params: "NOT OPTIMIZED"`; dropping it here would make the
    Stage 3 input silently shorter than the Stage 2 output, and "the sweep
    never produced parameters" would become indistinguishable from "this was
    certified and failed". The caller reports it as a skip with a reason.
    """
    out: list[dict] = []
    for row in ((blob or {}).get("results") or []):
        sym = row.get("symbol")
        row_tf = row.get("timeframe")
        if not sym or not row_tf:
            continue
        if tf and str(row_tf) != str(tf):
            continue
        status = str(row.get("status") or "").upper()
        out.append({
            "symbol": str(sym),
            "timeframe": str(row_tf),
            "stage2_status": status or "UNKNOWN",
            "certifiable": status == "OPTIMIZED",
            "quadrant": row.get("quadrant"),
            "optimal_regime": row.get("optimal_regime"),
            "stage1_version": row.get("stage1_version"),
            "in_stage1": bool(row.get("in_stage1")),
            "stage2_error": row.get("error") or "",
        })
    return out


def stage2_skip_reason(status: str) -> str:
    """
    Why a Stage 2 pair is not certifiable, in the words of the status itself.

    Stage 2 goes to real trouble to keep three states apart: OPTIMIZED, a
    PRUNED_FRAGILE grid that RAN and locked no winner, and an ERROR row whose
    sweep raised and wrote `params: "NOT OPTIMIZED"` (`scan.py` - "a sweep that
    raised produced no parameters at all, which is a different statement from a
    grid that produced no measurable Sharpe, and the two must not share a
    cell"). An ERROR row carries `error` text and never reaches this function.

    A fragility prune carries NO error text, because it is not an error - so
    the bare fallback that used to stand here relabelled the one deliberate
    ANTI-OVERFITTING verdict in Stage 2 as "stage 2 recorded no parameters",
    which reads as a broken run. An operator triaging that table goes looking
    for a crash that never happened, and - worse - the finding that every cell
    in the grid was an isolated spike is precisely the evidence AGAINST this
    configuration. Losing it is losing the result.
    """
    if str(status).upper() == STAGE2_PRUNED_FRAGILE:
        return ("stage 2 swept this configuration and locked no winner: every "
                "combination was an isolated spike or ruinous in sample "
                f"({STAGE2_PRUNED_FRAGILE}). The sweep ran; this is a "
                "fragility verdict, not a failed run.")
    return (f"stage 2 recorded no parameters for this configuration "
            f"({status or 'UNKNOWN'})")


class WindowOverlapError(ValueError):
    """The holdout has already been seen. Nothing downstream can fix that."""


def check_windows(is_start: str | None, is_end: str | None,
                  ho_start: str, ho_end: str | None) -> None:
    """
    Refuse an in-sample window that runs into the holdout.

    An open-ended in-sample window (`--is-end` omitted) is refused for the same
    reason as an overlapping one: it runs to the end of the lake, which
    includes every holdout bar. There is no safe default here, so there is no
    default.

    An open-ended HOLDOUT is the opposite case and is ACCEPTED. `--holdout-end`
    omitted means "to the present", which is the charter's default and the only
    one that stays correct as the lake grows: a hardcoded end silently stops
    certifying against the newest bars the moment a year rolls over, and the
    verdict it produces looks exactly like one that scored them. Reading
    further forward spends nothing - the holdout is the window this stage
    exists to spend.
    """
    if not is_end:
        raise WindowOverlapError(
            "--is-end is required. An in-sample window with no end runs to the "
            "end of the lake, which consumes the holdout and makes Gate 3 a "
            "strategy scored against itself.")
    is_e = pd.Timestamp(is_end)
    ho_s = pd.Timestamp(ho_start)
    if ho_end and ho_s >= pd.Timestamp(ho_end):
        raise WindowOverlapError(
            f"the holdout window is empty or inverted: {ho_start} → {ho_end}")
    if is_e >= ho_s:
        raise WindowOverlapError(
            f"the in-sample window ends {is_end}, on or after the holdout "
            f"starts {ho_start}. The holdout has been seen, so Gate 3 would "
            f"measure retention of a result on its own training data. Move "
            f"--is-end back before --holdout-start.")
    if is_start and pd.Timestamp(is_start) >= is_e:
        raise WindowOverlapError(
            f"the in-sample window is empty or inverted: {is_start} → {is_end}")


def _assert_no_prop_firm_rules(cfg: BacktestConfig) -> dict:
    """
    Refuse a config carrying a prop-firm constraint, and record that it did not.

    Charter clause 4. Account governance - daily loss limits, trailing
    drawdown, consistency caps - is CrossTrade NAM's, enforced against a live
    balance. Against a research equity curve those rules answer "would this
    particular funding program have tolerated the path", which is a fact about
    a rulebook rather than about the market; a certification that quietly
    applied one would fail strategies for having a lumpy road to the same
    money.

    `BacktestConfig` still carries the fields - they are legacy, see the
    separation of concerns in CLAUDE.md - and this stage never sets them, so
    the check reads as paranoia. It is written down anyway because the
    alternative to checking is trusting, and what it guards is invisible: a
    `trailing_drawdown_pct` set here would populate `result.breach`, cut the
    equity curve short, and change nothing else on the console.

    Returns the block the audit records, so the file STATES the absence rather
    than leaving it to be inferred from a missing key.
    """
    carrying = {f: getattr(cfg, f, None) for f in PROP_FIRM_FIELDS
                if getattr(cfg, f, None) is not None}
    if carrying:
        raise ValueError(
            f"stage 3 refuses a prop-firm constraint on the research config: "
            f"{carrying}. Daily loss limits and trailing drawdown are enforced "
            f"by CrossTrade NAM against a live account balance, not by a gate "
            f"audit - see the separation of concerns in CLAUDE.md.")
    return {
        "applied": False,
        "fields_checked": list(PROP_FIRM_FIELDS),
        "rule": ("Charter clause 4: no prop-firm daily loss limit, trailing "
                 "drawdown or consistency cap is applied at this stage. "
                 "Account governance is CrossTrade NAM's, against a live "
                 "balance."),
    }


def load_params(strat_name: str, symbol: str, out_dir: Path,
                overrides: dict, use_defaults: bool,
                tf: str | None = None) -> tuple[dict, dict]:
    """
    The parameters to certify, and where they came from.

    Returns `(params, provenance)`. Missing Stage 2 output is an ERROR unless
    `--defaults` was passed: certifying the module's defaults while the
    operator believes the sweep's winner was certified is the failure this
    argument exists to make deliberate.

    The TIMEFRAME-SPECIFIC file wins. A multi-timeframe Stage 2 writes
    `best_params_<SYMBOL>_<TF>.json` per timeframe and deliberately writes no
    unsuffixed file, so certifying at 5m picks up the 5m sweep's winner rather
    than whichever timeframe happened to be written last. The unsuffixed file
    is the fallback for a single-timeframe sweep, which writes both.

    `variants_tested_all_timeframes` is carried through beside
    `variants_tested`. When a timeframe was itself chosen by comparing
    leaderboards, the honest N is the larger one, and the audit records both
    rather than making that judgement here.
    """
    out_dir = Path(out_dir)
    candidates = []
    if tf:
        candidates.append(out_dir / BEST_PARAMS_FILE.format(
            symbol=f"{symbol}_{tf}"))
    candidates.append(out_dir / BEST_PARAMS_FILE.format(symbol=symbol))
    path = next((c for c in candidates if c.exists()), candidates[0])

    if use_defaults or not path.exists():
        if not use_defaults:
            raise FileNotFoundError(
                f"{' or '.join(str(c) for c in candidates)} does not exist. It "
                f"is written by stage 2 (backtest/scan.py) — run that first, "
                f"or pass --defaults to certify the module's DEFAULT_PARAMS "
                f"knowing that is what you are certifying.")
        return dict(overrides), {
            "params_source": "module DEFAULT_PARAMS with --param over them",
            "variants_tested": None,
            "variants_tested_all_timeframes": None,
            "scan_selection": None,
            "entry_filters": None,
            "stage1_regime": None,
            "best_params_file": None,
            # Nothing was locked because nothing was read. `--defaults` is the
            # deliberate way to certify the module's own parameters, and the
            # lock has to report that honestly rather than claiming a Stage 2
            # winner is bound.
            "params_locked": False,
            "params_lock_note": (
                "--defaults: the module's DEFAULT_PARAMS are being certified, "
                "not a Stage 2 winner. There is no locked parameter set."),
            "params_overridden": sorted(overrides),
        }

    blob = read_stage(path, 2, strat_name)
    params = {**(blob.get("params") or {}), **overrides}
    scanned_tf = blob.get("timeframe")
    prov = {
        "params_source": f"stage 2 winner ({path.name})"
                         + (" with --param over it" if overrides else ""),
        "variants_tested": blob.get("variants_tested"),
        "variants_tested_all_timeframes": blob.get(
            "variants_tested_all_timeframes"),
        "timeframes_searched": blob.get("timeframes_searched"),
        "scan_selection": blob.get("selection"),
        "scan_in_sample": blob.get("in_sample"),
        # The entry filters Stage 2 SWEPT UNDER. Certifying the winning
        # parameters on the whole week when they were selected with Monday
        # masked out certifies a strategy nobody optimised, and the gap would
        # show up as a Gate 1 that disagrees with the sweep's own in-sample
        # metrics for no visible reason. See `_resolve_filters`.
        "entry_filters": blob.get("entry_filters"),
        # The regime scope Stage 1 designated and Stage 2 transported without
        # applying. THIS is what Gate R certifies in. It arrives here having
        # been chosen in-sample as the best of four quadrants, which is exactly
        # why the holdout is the only evidence it generalised - and why the
        # quadrant has to travel rather than be re-derived: re-picking the best
        # quadrant on the HOLDOUT would make Gate R a best-of-four selection on
        # the very bars it is meant to be unseen evidence about, and it would
        # pass almost everything.
        "stage1_regime": blob.get("stage1_regime"),
        # The designation as Stage 2 writes it at the TOP LEVEL of the same
        # file, since 2026-08-21. Read as well as `stage1_regime` rather than
        # instead of it: a `best_params` written before the lift carries only
        # the nested copy, and `target_regime` prefers whichever is present
        # without either shape being the "new" one that invalidates the other.
        "best_params_regime": _named_regime(blob.get("optimal_regime")),
        "best_params_quadrant": blob.get("target_quadrant"),
        # The four-quadrant table the designation beat, and the
        # positive-expectancy runners-up. `secondary_regimes` is scored in ONE
        # case only - `regime_gate`'s starvation-only fallback, when the
        # primary placed too few holdout trades to be measured. Certifying in
        # a secondary as a matter of course would give Gate R two chances at a
        # 1.00 holdout profit factor, so the gate refuses it whenever the
        # primary traded enough and lost. `regime_scores` is what the REGIME
        # STARVATION diagnostic reads to say where the candidate was actually
        # dominant in sample.
        "regime_scores": blob.get("regime_scores") or (
            (blob.get("stage1_regime") or {}).get("regime_scores") or {}),
        "secondary_regimes": blob.get("secondary_regimes") or (
            (blob.get("stage1_regime") or {}).get("secondary_regimes") or []),
        "best_params_file": str(path),
        # The parameter lock, charter clause 1. True when every bound value
        # came from Stage 2's winner untouched. `--param` is still allowed -
        # an operator correcting the record on purpose outranks a file - but it
        # is re-tuning, and a certification that was re-tuned has to say so in
        # the file rather than only in a shell history nobody keeps.
        "params_locked": not bool(overrides),
        "params_lock_note": (
            f"locked verbatim from {path.name}; nothing re-tuned at stage 3"
            if not overrides else
            f"LOCK BROKEN: --param overrode "
            f"{', '.join(sorted(overrides))} after the sweep. These are not "
            f"the parameters Stage 2 selected."),
        "params_overridden": sorted(overrides),
    }
    if tf and scanned_tf and scanned_tf != tf:
        # Only reachable through the unsuffixed fallback. Certifying 5m bars
        # with a winner selected on 15m bars is a different strategy than the
        # one the sweep scored, and nothing downstream could detect it.
        raise ValueError(
            f"{path.name} holds a winner selected on {scanned_tf} bars, but "
            f"this audit is running on {tf}. Re-run stage 2 at {tf}, or "
            f"certify at {scanned_tf}.")
    return params, prov


def _resolve_filters(cfg_kwargs: dict, prov: dict) -> tuple[dict, str]:
    """
    The entry filters this certification runs under, and where they came from.

    Same precedence as Stage 2's `resolve_exclude_days`: an explicit
    `--exclude-days` on this command wins, otherwise the exclusion Stage 2
    recorded on `best_params_<SYMBOL>_<TF>.json` is inherited, otherwise none.

    Inheriting is the point. Stage 1 names every weekday below a 1.00 profit
    factor, Stage 2 selects the parameters with those sessions masked out, and
    a Stage 3 that then certified the whole week would be scoring a strategy
    that was never optimised - Gate 1 would disagree with the sweep's own in-sample numbers,
    Gate 3 would measure retention between two different strategies, and
    nothing in either file would say why. `--defaults` and a best_params
    written before this contract existed both yield no exclusion, which is the
    correct reading of both.

    The news filter is deliberately NOT inherited. Its provenance can be a
    rule-generated calendar whose dates are approximate, and silently carrying
    that onto a certification would put an approximate blocking window inside
    a gate verdict. Stage 3 takes `--news-filter` from its own command line or
    not at all.
    """
    out = dict(cfg_kwargs)
    if out.get("exclude_days"):
        return out, "--exclude-days (CLI, overrides stage 2)"
    recorded = (prov.get("entry_filters") or {}).get("exclude_days") or []
    if not recorded:
        return out, "none"
    out["exclude_days"] = tuple(sorted(int(d) for d in recorded))
    src = ((prov.get("entry_filters") or {}).get("exclude_days_source")
           or "stage 2")
    return out, f"inherited from stage 2 — {src}"


# --------------------------------------------------------------------------
# Gate R - the certification. Charter clause 3.
# --------------------------------------------------------------------------

def _num(value) -> float | None:
    """A float, or None. `NaN` is None: it is not a number and must not sort."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (f != f) else f


def _named_regime(value) -> str | None:
    """
    A regime NAME, or None - with the JSON round trip's fake names removed.

    `"None"` is what a JSON round trip makes of the profiler's own token for
    "no quadrant cleared the bar". It is not a regime name, and treating it as
    one sends Gate R to a breakdown key that never exists, which comes back as
    zero trades and reads on every table as a strategy that stopped trading.

    Module level rather than a closure since 2026-08-21, because `_load_params`
    now normalises Stage 2's top-level `optimal_regime` with the same rule
    `target_regime` applies to the nested one. Two copies would be one edit
    away from the two keys disagreeing about what `"None"` means.
    """
    text = str(value or "").strip()
    return text if text and text not in ("None", "null", "nan") else None


def regime_for_quadrant(code) -> str | None:
    """
    `"Q1"` -> `"High Volatility / Trending"`, and None for anything else.

    Inverted from `profiler.REGIME_TO_QUADRANT`, which is itself inverted from
    the integer map `mdlib.regimes` owns and `profiler` checks at import - so
    there is still exactly one place where a code and a name are the same
    statement. A `best_params` carrying a quadrant CODE and no name is
    resolvable here rather than being reported as an undeclared regime.
    """
    text = str(code or "").strip().upper()
    for regime, quad in REGIME_TO_QUADRANT.items():
        if quad == text:
            return regime
    return None


class UnknownRegimeError(ValueError):
    """A quadrant name no profiler produces. Gate R would read it as silence."""


def target_regime(prov: dict, target: dict | None,
                  override: str | None = None) -> tuple[str | None, str]:
    """
    The ONE quadrant Gate R certifies in, and where it came from.

    Stage 1 designated it, Stage 2 transported it onto
    `best_params_<SYMBOL>_<TF>.json` without applying it, and this reads it
    back. The Stage 2 summary row is the fallback, because a `best_params`
    written before the scope travelled carries none and its matrix still does.

    **It is never re-derived here.** Picking the best quadrant of the holdout
    profile would turn Gate R into a best-of-four selection made on the very
    bars it exists to be unseen evidence about - four chances at a 1.00 profit
    factor, which almost anything clears once. The quadrant is an input to this
    stage, not an output of it, and a configuration that arrives without one
    cannot be certified at all.
    """
    _named = _named_regime

    def _checked(name: str, where: str) -> tuple[str, str]:
        # A name outside `REGIMES` is a BROKEN HANDOFF, not a strategy result,
        # and it has to raise. Left alone it would send Gate R to a breakdown
        # key that never exists, which comes back as zero trades and reads on
        # every table as a strategy that stopped trading its own environment
        # out of sample.
        if name not in REGIMES:
            raise UnknownRegimeError(
                f"{where} names the quadrant {name!r}, which is not one of "
                f"the four the profiler produces ({', '.join(REGIMES)}). Gate "
                f"R would measure it as zero trades and report a broken "
                f"handoff as a strategy that failed out of sample.")
        return name, where

    if override:
        return _checked(str(override).strip(),
                        "--regime (operator override, not stage 1)")

    where = prov.get("best_params_file")
    where_name = Path(where).name if where else "the stage 2 handoff"

    # Stage 2 writes the designation at the top level of `best_params` since
    # 2026-08-21, and nested under `stage1_regime` as it always did. The
    # top-level copy is read FIRST and the nested one is the fallback, so a
    # file written before the lift still certifies and a file written after it
    # does not need the nested copy to agree. When both are present and they
    # DISAGREE the top-level wins and the disagreement is recorded in the
    # source string rather than resolved silently - a handoff whose two
    # designations differ is a bug worth seeing on the audit.
    top = _named(prov.get("best_params_regime")) or regime_for_quadrant(
        prov.get("best_params_quadrant"))
    nested = _named((prov.get("stage1_regime") or {}).get("optimal_regime"))
    if top:
        if nested and nested != top:
            return _checked(top, f"stage 2 handoff {where_name} "
                                 f"(top-level optimal_regime; its nested "
                                 f"stage1_regime disagrees and names "
                                 f"{nested!r})")
        return _checked(top, f"stage 1, via {where_name}")
    if nested:
        return _checked(nested, f"stage 1, via {where_name}")

    # Last resort: the Stage 2 SUMMARY row, for a `best_params` written before
    # the regime scope travelled at all. Its matrix still carries the quadrant.
    row = _named((target or {}).get("optimal_regime")) or regime_for_quadrant(
        (target or {}).get("quadrant"))
    if row:
        return _checked(row, f"stage 1, via {STAGE2_SUMMARY_FILE}")
    return None, "not declared"


def regime_starvation(regime: str | None, n_holdout: int,
                      regime_scores: dict | None) -> dict | None:
    """
    The diagnostic for a Gate R that failed on SAMPLE, not on edge.

    "Gate R FAIL" reads identically whether the strategy lost money in its own
    environment or simply never entered it again, and those are fixed by
    completely different work: the first is a dead edge, the second is a
    designation pointing at a quadrant the holdout barely contains. The second
    is also the more common failure of a best-of-four in-sample pick, and it is
    invisible on the gate table - a quadrant with one holdout trade prints a
    profit factor of 999 and a PASS on the factor row.

    Returns None when the gate did not starve, so a caller can print it or not
    without deciding anything. `dominant_quadrant` is where the candidate
    actually made its money IN SAMPLE, read from the scored table Stage 2
    embedded - not recomputed, and never taken from the holdout: naming a new
    quadrant off the holdout is the best-of-four selection Gate R exists to
    avoid, and this diagnostic must not smuggle one in through a print
    statement.
    """
    if not regime:
        return None
    rows = [r for r in (regime_scores or {}).values()
            if isinstance(r, dict) and r.get("score") is not None]
    dominant = max(rows, key=lambda r: r["score"]) if rows else None
    return {
        "target_regime": regime,
        "target_quadrant": quadrant_id(regime),
        "holdout_trades": int(n_holdout),
        "dominant_regime": (dominant or {}).get("regime"),
        "dominant_quadrant": (dominant or {}).get("quadrant"),
        "dominant_score": (dominant or {}).get("score"),
        "dominant_basis": "in-sample alpha score, from the stage 2 handoff",
        "message": (
            f"[REGIME STARVATION] Quadrant "
            f"{quadrant_id(regime) or '?'} ({regime}) had only "
            f"{int(n_holdout)} holdout trades."
            + (f" Candidate was dominant in "
               f"{(dominant or {}).get('quadrant')} "
               f"({(dominant or {}).get('regime')}) in sample."
               if dominant else
               " No in-sample quadrant scores travelled with this handoff, so "
               "where it was dominant cannot be stated.")),
    }


CERTIFIED_ON_PRIMARY = "primary"
CERTIFIED_ON_FALLBACK = "secondary_starvation_fallback"


def _evaluate_quadrant(profile: dict | None, regime: str,
                       min_profit_factor: float, min_trades: int) -> dict:
    """
    One quadrant against Gate R's two bars. The primary and any fallback are
    measured HERE, by one body of code, so the secondary can never be held to
    a different bar than the quadrant it stands in for.
    """
    stats = ((profile or {}).get("regime_breakdown") or {}).get(regime) or {}
    pf = _num(stats.get("profit_factor"))
    n = int(stats.get("trade_count", 0) or 0)
    count_ok = n >= int(min_trades)
    # An undefined profit factor over enough trades cannot clear a bar it was
    # never measured against. It is only reachable when the quadrant has trades
    # but the profiler recorded no factor for them, which is a broken profile
    # rather than a break-even one - `pf is None and count_ok` must not read as
    # a pass through `None >= 1.00` raising or, worse, being skipped.
    pf_ok = pf is not None and pf >= float(min_profit_factor)
    return {
        "stats": stats, "pf": pf, "n": n, "quad": quadrant_id(regime),
        "count_ok": count_ok, "pf_ok": pf_ok,
        "checks": [
            {"label": "Trades in quadrant", "value": n,
             "threshold": f">= {int(min_trades)}",
             "status": PASS if count_ok else FAIL,
             "note": ("" if count_ok else
                      f"the strategy never traded {regime} out of sample"
                      if n == 0 else
                      f"{n} trade(s) in {regime} is too thin to separate an "
                      f"edge from a run of luck")},
            {"label": "Profit factor in quadrant",
             "value": "not measured" if pf is None else round(pf, 2),
             "threshold": f">= {float(min_profit_factor):.2f}",
             "status": PASS if pf_ok else FAIL,
             "note": ("" if pf_ok else
                      f"no profit factor was recorded for {regime}"
                      if pf is None else
                      f"{pf:.2f} in {regime}, below "
                      f"{float(min_profit_factor):.2f}")},
        ],
    }


def _fallback_candidate(secondary_regimes: Any) -> dict | None:
    """
    The pre-declared secondary a starved primary may fall back to, or None.

    ELIGIBLE ONLY. `profiler.designate_regime` keeps two kinds of runner-up:
    quadrants that cleared every designation bar and lost on score, and
    profitable ones disqualified on sample size. Only the first kind is a
    quadrant the screen was willing to designate, and certifying in one the
    screen refused would let a fallback reach an environment the primary path
    could never have been given.

    The list arrives ranked, so the first eligible entry is the highest-scoring
    one. Nothing here reads the holdout.
    """
    for row in (secondary_regimes or []):
        if isinstance(row, dict) and row.get("eligible") and row.get("regime"):
            return row
    return None


def regime_gate(profile: dict | None, regime: str | None,
                min_profit_factor: float = MIN_REGIME_PROFIT_FACTOR,
                min_trades: int = MIN_REGIME_TRADES,
                regime_scores: dict | None = None,
                secondary_regimes: Any = None) -> dict:
    """
    Did the edge survive out of sample INSIDE its designated quadrant?

    This is the certification, and the only thing in this file that can refuse
    one. Both bars are applied to the SAME quadrant - `MIN_REGIME_PROFIT_FACTOR`
    over `MIN_REGIME_TRADES`, imported from Stage 1 so a screen and a
    certification can never be held to different numbers - and the quadrant is
    the one Stage 1 named, measured on the HOLDOUT profile.

    Three outcomes, kept distinct because they are fixed by different work and
    only one of them is a statement about the strategy:

    - **NOT EVALUATED** when no quadrant was designated. Not a pass. A
      configuration with no environment attached is one nothing screened, and
      certifying it would hand a live supervisor a parameter set with no
      instruction about when to stand it down.
    - **FAIL on the trade count** when the strategy traded its own environment
      fewer than `min_trades` times out of sample - including zero times. That
      is a real finding: the quadrant was chosen in-sample as the best of four,
      and an edge that never appears in it again over three years of unseen
      bars did not generalise. It is a different failure from losing money
      there, and the note says which.
    - **FAIL on the profit factor** when it traded enough and did not pay.

    THE STARVATION-ONLY FALLBACK (2026-09-08)
    =========================================
    A primary quadrant that came in UNDER `min_trades` was never measured, and
    `secondary_regimes` - the positive-expectancy runner-up Stage 1 declared
    from IN-SAMPLE evidence and Stage 2 carried - is evaluated in its place,
    against these same two bars.

    The rule that keeps this one test rather than two: **the fallback runs only
    on starvation, never after a performance failure.** A primary that traded
    `min_trades` times and finished below the factor is a hard FAIL and stops
    there. Looking at a second quadrant after seeing a real result in the first
    is the best-of-N selection this gate exists to prevent - it would give Gate
    R two chances at a 1.00 holdout profit factor and pass close to everything.
    Starvation is the opposite case: there is no result to be disappointed by,
    so the pre-declared alternative is a first measurement, not a second.

    Two further limits. The secondary is named IN SAMPLE and never read off the
    holdout, so nothing here chooses a quadrant because these bars liked it.
    And only a secondary marked `eligible` may be used - one the screen
    disqualified is not an environment the primary path could have reached
    either.

    `certified_on` records which quadrant carried it, `target_regime` becomes
    the one that did, and `primary_regime` keeps the designation beside it. A
    live supervisor is still handed exactly ONE quadrant.

    A profit factor of `inf` (the quadrant never lost) clears, and the
    profiler's 999 sentinel for the same case is passed through rather than
    normalised - rewriting another module's sentinel inside a gate is how two
    modules come to disagree about what 999 meant.
    """
    name = "Gate R · Regime Generalization (HOLDOUT)"
    if not regime:
        return {
            "name": name, "status": NOT_EVALUATED, "target_regime": None,
            "quadrant": None, "measured": None, "checks": [],
            "note": ("no optimal_regime was designated for this "
                     "configuration, so there is no environment to certify "
                     "in. NOT EVALUATED is not a pass - re-run stage 1 so the "
                     "scope travels, or name it with --regime."),
        }

    prim_regime = regime
    prim = _evaluate_quadrant(profile, regime, min_profit_factor, min_trades)
    stats, pf, n, quad = prim["stats"], prim["pf"], prim["n"], prim["quad"]
    count_ok, pf_ok, checks = prim["count_ok"], prim["pf_ok"], prim["checks"]

    # ---- the STARVATION-ONLY fallback (2026-09-08) ---------------------
    #
    # Reached only when the primary quadrant produced FEWER THAN `min_trades`
    # holdout trades - that is, when it was never measured at all. A primary
    # that traded enough and finished below the profit factor is a hard FAIL
    # and stops here: looking at a second quadrant after seeing a real result
    # in the first is the best-of-N selection this gate exists to prevent, and
    # it is the difference between one test and two.
    #
    # The secondary is PRE-DECLARED from in-sample evidence by
    # `profiler.designate_regime` and travels on the Stage 2 handoff. It is
    # never chosen from the holdout, and only a secondary that cleared the
    # designation bars in sample (`eligible`) is eligible here - a quadrant the
    # screen refused is not a quadrant this may certify.
    fallback = None
    if not count_ok:
        cand = _fallback_candidate(secondary_regimes)
        if cand is not None:
            alt_regime = str(cand.get("regime"))
            alt = _evaluate_quadrant(profile, alt_regime,
                                     min_profit_factor, min_trades)
            fallback = {
                "attempted": True,
                "regime": alt_regime,
                "quadrant": quadrant_id(alt_regime) or cand.get("quadrant"),
                "trigger": (f"the primary quadrant {quad} placed {n} holdout "
                            f"trade(s), below the {int(min_trades)} needed to "
                            f"measure it at all"),
                "in_sample": {k: cand.get(k) for k in
                              ("profit_factor", "trade_count", "net_pnl",
                               "score", "eligible")},
                "measured": {"profit_factor": alt["pf"],
                             "trade_count": alt["n"]},
                "status": PASS if (alt["count_ok"] and alt["pf_ok"]) else FAIL,
            }
            checks = checks + [{**c, "label": f"{c['label']} (secondary)"}
                               for c in alt["checks"]]
            if alt["count_ok"] and alt["pf_ok"]:
                # The certification MOVES to the quadrant that carried it, and
                # `target_regime` is what promote.py writes into the package's
                # regime_filter - so the live supervisor is told the one
                # quadrant this was actually certified in, exactly as before.
                regime, quad = alt_regime, fallback["quadrant"]
                stats, pf, n = alt["stats"], alt["pf"], alt["n"]
                count_ok, pf_ok = True, True
    # An undefined profit factor over enough trades cannot clear a bar it was
    # never measured against. It is only reachable when the quadrant has trades
    # but the profiler recorded no factor for them, which is a broken profile
    # rather than a break-even one - `pf is None and count_ok` must not read as
    # a pass through `None >= 1.00` raising or, worse, being skipped.
    pf_ok = pf is not None and pf >= float(min_profit_factor)

    # Starvation is a FAIL on the trade count, whatever the factor did. A
    # quadrant with three holdout trades and a 999 profit factor fails here and
    # passes the factor row, so keying the diagnostic on the overall status
    # would attach it to exactly the cases where it is least needed. It is
    # reported on the PRIMARY quadrant even when a fallback then carried the
    # certification: the drought is what happened, and a card that dropped the
    # diagnostic the moment the fallback worked would hide why it ran.
    starved = (None if prim["count_ok"] else
               regime_starvation(prim_regime, prim["n"], regime_scores))
    certified_on = (CERTIFIED_ON_FALLBACK
                    if (fallback or {}).get("status") == PASS
                    else CERTIFIED_ON_PRIMARY)
    return {
        "name": name,
        "status": PASS if (count_ok and pf_ok) else FAIL,
        "target_regime": regime,
        "quadrant": quad,
        # What Stage 1 designated, kept beside the certified quadrant so the
        # two can never be confused once a fallback has moved one of them.
        "primary_regime": prim_regime,
        "primary_quadrant": prim["quad"],
        "certified_on": certified_on,
        "fallback": fallback,
        "regime_starvation": starved,
        "measured": {
            "profit_factor": pf,
            "trade_count": n,
            "win_rate": _num(stats.get("win_rate")),
            "net_pnl": _num(stats.get("net_pnl")),
        },
        "checks": checks,
        "note": (
            ("the certification: the edge is measured only where Stage 1 "
             "said it lives. Performance in the other three quadrants is "
             "reported and never scored - a live supervisor stands the "
             "strategy down there.")
            if certified_on == CERTIFIED_ON_PRIMARY else
            (f"CERTIFIED ON THE SECONDARY. {prim['quad']} placed "
             f"{prim['n']} holdout trade(s), below the {int(min_trades)} "
             f"needed to measure it, so the pre-declared secondary {quad} was "
             f"evaluated instead - one test, not two. The primary was never "
             f"judged on edge and this is not a second look at a quadrant "
             f"that failed. {quad} is what the live supervisor is told to "
             f"trade; it is stood down everywhere else, including "
             f"{prim['quad']}.")),
    }


# The retention table. Higher is better for the first three; for a drawdown
# the SMALLER magnitude is, so its ratio is inverted and labelled - a raw
# oos/is on a drawdown would score a strategy that drew down twice as deep at
# 2.00 and put it top of the table.
RETENTION_METRICS = (
    ("profit_factor", "Profit factor", "higher"),
    ("sharpe", "Sharpe", "higher"),
    ("max_drawdown_pct", "Max drawdown", "lower"),
    ("win_rate", "Win rate", "higher"),
)


def retention_scores(is_metrics: dict | None,
                     ho_metrics: dict | None) -> dict:
    """
    IS vs OOS retention for the four headline metrics. Charter clause 5.

    **Reported, never scored.** Nothing in this file fails a configuration on a
    retention ratio: these are blended-sample numbers across every market
    state, and clause 3 is that a strategy is not pruned on those. They are
    here because "the holdout profit factor is 1.10" and "the holdout profit
    factor is 1.10, down from 2.40" are different findings, and a certification
    that printed only the first would let a collapsing edge look like a
    healthy one.

    A ratio is `None` rather than 0.0 when either side is missing or the
    denominator is zero: a metric nobody could compute and a metric that
    retained nothing are different statements, and a 0.00 in this column reads
    as the second. `direction` travels with every row so the number is never
    read the wrong way round - on `max_drawdown_pct` the ratio is
    `|in-sample| / |holdout|`, so above 1.00 still means "held up", the same as
    every other row.
    """
    rows = {}
    for key, label, direction in RETENTION_METRICS:
        a = _num((is_metrics or {}).get(key))
        b = _num((ho_metrics or {}).get(key))
        if a is None or b is None:
            ratio = None
        elif direction == "lower":
            # Magnitudes: the engine signs drawdowns negative, and -40/-20
            # would read as a strategy that improved.
            num, den = abs(a), abs(b)
            ratio = (num / den) if den else None
        else:
            ratio = (b / a) if a else None
        rows[key] = {
            "label": label,
            "in_sample": a,
            "holdout": b,
            "retention": ratio,
            "direction": direction,
            "basis": ("|in-sample| / |holdout|; above 1.00 means the holdout "
                      "drew down LESS" if direction == "lower"
                      else "holdout / in-sample; above 1.00 means the holdout "
                           "was better"),
        }
    return {
        "metrics": rows,
        "scored": False,
        "rule": ("Charter clause 5: retention is CALCULATED and reported. "
                 "Charter clause 3: it is not a pruning criterion - no "
                 "configuration is failed here for a blended-sample ratio. "
                 "Gate R is the verdict."),
    }


GATE_RUIN = "gate_ruin"
RUIN_CHECK_FAILED = "FAILED_RUIN_CHECK"

# The ruin boundary is `backtest.pipeline`'s, so Stage 2's parameter selection
# and this gate refuse the same equity curves. Re-exported under its original
# name because every caller in this module and its tests reads it from here.
# It is the same reading the Monte Carlo sanitation clips to -1.00 and calls
# ruin.


def ruin_guard(metrics: dict | None) -> dict:
    """
    Did the account survive the IN-SAMPLE window? Charter clause 3, hard bar.

    **This is the one blended-sample check that CAN refuse a certification**,
    and it is deliberately not one of Gates 1-3. Those score edge quality -
    profit factor, Sharpe retention, a drawdown budget - and the charter's
    reasoning for demoting them is sound: a regime-gated strategy never trades
    the blended sample they measure, so pruning on it prunes on a result nobody
    will realise.

    Ruin is a different kind of statement. It is not "this edge is weaker than
    we would like across states the supervisor will stand it down in" - it is
    "on the bars this strategy WAS run on, the account reached zero." A
    supervisor cannot stand a strategy down out of an account that no longer
    exists, and there is no quadrant restriction that makes a blown account
    into a survivable one, because the equity path that blew it is the path the
    certified quadrant's trades are embedded in.

    Three readings of the same event, checked together because a result can
    show any one of them:

    - `ruined` - the engine's own flag, `final_equity <= 0`;
    - `final_equity` at or below zero, checked directly for a metrics dict
      written before that flag existed;
    - `max_drawdown_pct` at or past -100%, which is ruin recorded on the
      drawdown rather than on the balance - a path can touch zero and recover
      on paper, and the recovery is fictional.

    A metrics dict carrying NONE of the three is NOT EVALUATED rather than a
    pass. Absent evidence of survival is not evidence of survival, and this
    gate exists precisely because the failure it catches was invisible.
    """
    name = "Gate Ruin · In-Sample Account Survival (HARD)"
    if not metrics:
        return {"name": name, "status": NOT_EVALUATED, "checks": [],
                "ruined": None,
                "note": ("no in-sample metrics were supplied, so account "
                         "survival could not be checked. NOT EVALUATED is not "
                         "a pass.")}

    ruined_flag = metrics.get("ruined")
    equity = _num(metrics.get("final_equity"))
    dd = _num(metrics.get("max_drawdown_pct"))

    checks, verdicts = [], []

    if ruined_flag is None:
        checks.append({"label": "Engine ruin flag", "value": None,
                       "threshold": "false", "status": NOT_EVALUATED,
                       "note": "the metrics dict carries no `ruined` field"})
    else:
        ok = not bool(ruined_flag)
        verdicts.append(ok)
        checks.append({"label": "Engine ruin flag", "value": bool(ruined_flag),
                       "threshold": "false", "status": PASS if ok else FAIL,
                       "note": "" if ok else
                               "the engine recorded the account as ruined"})

    if equity is None:
        checks.append({"label": "Final equity", "value": None,
                       "threshold": "> 0", "status": NOT_EVALUATED,
                       "note": "no final equity recorded"})
    else:
        ok = equity > 0.0
        verdicts.append(ok)
        checks.append({"label": "Final equity", "value": equity,
                       "threshold": "> 0", "status": PASS if ok else FAIL,
                       "note": "" if ok else
                               f"the account ended at {equity:,.2f}"})

    if dd is None:
        checks.append({"label": "Max drawdown", "value": None,
                       "threshold": f"> {RUIN_MIN_DRAWDOWN_PCT:.0f}%",
                       "status": NOT_EVALUATED,
                       "note": "no drawdown recorded"})
    else:
        ok = dd > RUIN_MIN_DRAWDOWN_PCT
        verdicts.append(ok)
        checks.append({"label": "Max drawdown", "value": dd,
                       "threshold": f"> {RUIN_MIN_DRAWDOWN_PCT:.0f}%",
                       "status": PASS if ok else FAIL,
                       "note": "" if ok else
                               (f"{dd:,.2f}% is past total loss - an account "
                                f"cannot lose more than it holds")})

    if not verdicts:
        status = NOT_EVALUATED
        note = ("nothing on the metrics dict states whether the account "
                "survived. NOT EVALUATED is not a pass.")
    elif all(verdicts):
        status = PASS
        note = "the account survived the in-sample window"
    else:
        status = FAIL
        note = ("the account did NOT survive the in-sample window. No "
                "quadrant restriction makes a blown account survivable: the "
                "equity path that blew it is the path the certified "
                "quadrant's trades are embedded in.")

    return {"name": name, "status": status, "checks": checks,
            "ruined": (None if not verdicts else status != PASS),
            "final_equity": equity, "max_drawdown_pct": dd,
            "note": note}


GATE_Q = "gate_all_quadrants"
ALL_QUADRANTS_FAILED = "FAILED_ALL_QUADRANT_CHECK"

# The all-quadrant bars. Deliberately NOT imported from Stage 1: the screen's
# `MIN_REGIME_*` pair is what designates a HOME quadrant, and reusing it here
# would say the bar for "this is where the edge lives" and the bar for "it also
# survives the other three" are the same statement. They are not.
ALL_QUADRANT_MIN_TRADES = 20
ALL_QUADRANT_MIN_SHARPE = 0.05
# RETIRED 2026-09-08. Was 15_000.0: a quadrant-local cap in DOLLARS, on an
# assumed $100,000 of starting capital. That makes it a statement about an
# ACCOUNT, not about an edge - the same measurement on a $250,000 account is a
# different verdict on identical bars - and CLAUDE.md puts anything that
# depends on an account balance behind CrossTrade NAM, which enforces it
# against a real one. The quadrant-local drawdown is still MEASURED and still
# reported on every row; it simply no longer refuses a certification here.
#
# This is NOT the ruin guard, which stays: `charter_audit`'s ruin bar and
# `scan.fragility_of` both key on RUIN_MIN_DRAWDOWN_PCT (-100%), a scale-free
# statement that the account reached zero on its own selection bars. Position
# sizing moves a dollar drawdown; it does not move an equity curve that hit
# zero.
ALL_QUADRANT_MAX_DRAWDOWN_PNL = None


def all_quadrant_gate(profile: dict | None,
                      min_trades: int = ALL_QUADRANT_MIN_TRADES,
                      min_sharpe: float = ALL_QUADRANT_MIN_SHARPE,
                      max_drawdown: float | None = ALL_QUADRANT_MAX_DRAWDOWN_PNL,
                      ) -> dict:
    """
    Did the edge hold in EVERY quadrant individually, not just its own?

    Requested 2026-09-07. Each of the four quadrants must separately show
    positive expectancy (net P&L > 0), a per-trade Sharpe at or above
    `min_sharpe`, and a quadrant-local drawdown no deeper than `max_drawdown` -
    over at least `min_trades` trades, because a bar cleared on six trades is
    not a measurement.

    THIS CONTRADICTS THE CHARTER'S PREMISE, AND THAT IS THE POINT OF WRITING
    IT DOWN HERE
    ===========================================================================
    Everything else in this pipeline is built for a regime SPECIALIST. Stage 1
    designates ONE home quadrant, Gate R certifies the edge only there, and the
    live supervisor stands the strategy down in the other three - so a Q4 range
    fade is not merely untested in Q1, it is DESIGNED not to trade there and
    will be stood down before it can. Requiring it to pay in Q1 as well asks a
    specialist to be an all-weather system, and most will fail this gate by
    construction rather than by defect. That is the intended, stated effect of
    the request, not a bug to tune the thresholds around.

    The consequence to expect: near-zero promotions until either the
    thresholds move or the strategies change.

    THAT IS WHAT HAPPENED, AND THE GATE IS NOW ADVISORY (2026-09-08)
    ---------------------------------------------------------------
    Enforced for one day, this refused every configuration of
    `t3_braid_scalp_20260823` across 43 pairs. The paragraph above predicted
    exactly that, so the gate is not wrong - it is answering a question the
    rest of the pipeline does not ask. It is still measured on every audit and
    still reported in full; it simply no longer refuses a certification unless
    a caller passes `--require-all-quadrants`. The dollar drawdown cap that
    used to be one of its bars is retired outright - see
    ALL_QUADRANT_MAX_DRAWDOWN_PNL.

    A quadrant the strategy NEVER TRADED is a FAIL, not a skip. The profiler
    omits a zero-trade quadrant from the breakdown entirely, so an absent key
    and a losing key are the same verdict here and the reason distinguishes
    them - reading absence as "nothing to object to" would pass exactly the
    specialist this gate exists to catch.

    `sharpe_trade` is the profiler's per-trade, UNANNUALISED ratio. It is not
    comparable to an annualised Sharpe from anywhere else; see
    `profiler._quadrant_risk`.
    """
    name = "Gate Q · All-Quadrant Robustness (HOLDOUT)"
    breakdown = (profile or {}).get("regime_breakdown") or {}
    rows: list[dict] = []
    for regime in REGIMES:
        quad = REGIME_TO_QUADRANT.get(regime)
        stats = breakdown.get(regime)
        if not stats:
            rows.append({
                "regime": regime, "quadrant": quad, "status": FAIL,
                "trade_count": 0, "net_pnl": None, "sharpe_trade": None,
                "max_drawdown_pnl": None,
                "reason": f"the strategy never traded {regime} out of sample"})
            continue
        n = int(stats.get("trade_count", 0) or 0)
        net = _num(stats.get("net_pnl"))
        sharpe = _num(stats.get("sharpe_trade"))
        dd = _num(stats.get("max_drawdown_pnl"))
        why: list[str] = []
        if n < int(min_trades):
            why.append(f"{n} trade(s) is below the {int(min_trades)} needed "
                       f"to measure the bars below")
        if net is None or net <= 0:
            why.append("net P&L is not positive"
                       if net is None else f"net P&L {net:,.2f} <= 0")
        # A missing Sharpe is a REFUSAL, never a skip: it means one trade, or
        # a quadrant whose trades had no dispersion at all, and neither is a
        # ratio that cleared a bar.
        if sharpe is None or sharpe < float(min_sharpe):
            why.append("no per-trade Sharpe was recorded" if sharpe is None
                       else f"Sharpe {sharpe:.3f} < {float(min_sharpe):.2f}")
        # `max_drawdown=None` is the default since 2026-09-08: the quadrant
        # drawdown is REPORTED on the row below and grades nothing. A caller
        # that passes a number gets the old dollar bar back.
        if max_drawdown is not None and (dd is None
                                         or abs(dd) > float(max_drawdown)):
            why.append("no drawdown was recorded" if dd is None
                       else f"drawdown ${abs(dd):,.2f} past the "
                            f"${float(max_drawdown):,.0f} cap")
        rows.append({
            "regime": regime, "quadrant": quad,
            "status": PASS if not why else FAIL,
            "trade_count": n, "net_pnl": net, "sharpe_trade": sharpe,
            "max_drawdown_pnl": dd, "reason": "; ".join(why)})

    if not breakdown:
        return {"name": name, "status": NOT_EVALUATED, "quadrants": rows,
                "checks": [], "note": ("no regime breakdown was recorded, so "
                                       "no quadrant could be measured. NOT "
                                       "EVALUATED is not a pass.")}

    failed = [r for r in rows if r["status"] != PASS]
    checks = [{"label": f"{r['quadrant']} · {r['regime']}",
               "value": (f"n={r['trade_count']} net={r['net_pnl']} "
                         f"sharpe={r['sharpe_trade']} dd={r['max_drawdown_pnl']}"),
               "threshold": (f"n>={int(min_trades)}, net>0, "
                             f"sharpe>={float(min_sharpe):.2f}"
                             + (f", |dd|<=${float(max_drawdown):,.0f}"
                                if max_drawdown is not None
                                else "; drawdown reported, not graded")),
               "status": r["status"], "note": r["reason"]} for r in rows]
    return {
        "name": name,
        "status": PASS if not failed else FAIL,
        "quadrants": rows,
        "checks": checks,
        "thresholds": {"min_trades": int(min_trades),
                       "min_sharpe": float(min_sharpe),
                       "max_drawdown_pnl": (None if max_drawdown is None
                                            else float(max_drawdown))},
        "note": ("every quadrant cleared" if not failed else
                 f"{len(failed)} of 4 quadrant(s) failed: "
                 + "; ".join(f"{r['quadrant']} ({r['reason']})"
                             for r in failed)
                 + ". A regime SPECIALIST is expected to fail this - it is "
                   "stood down outside its own quadrant and never trades the "
                   "others, so this is ADVISORY and refuses nothing by "
                   "itself. Pass --require-all-quadrants to give it "
                   "authority, for an all-weather mandate."),
    }


def charter_audit(audit: dict, gate_r: dict, retention: dict,
                  in_sample_metrics: dict | None = None,
                  holdout_profile: dict | None = None,
                  require_all_quadrants: bool = False) -> dict:
    """
    Fold Gate R into the audit and make it the verdict. Charter clause 3.

    The returned dict keeps the shape everything downstream already reads -
    `status`, `passed`, `gates` - so `promote.load_gate_certification`,
    `promote._gate_summary` and the certification leaderboard need no change.
    What moves is what `status` MEANS: it was the roll-up of Gates 1, 2 and 3
    and it is now Gate R alone.

    **This is a deliberate loosening and it is recorded as one.** Gates 1-3 are
    still computed in full, still carried under `gates`, and still printed;
    they simply cannot refuse a certification any more, because they score the
    BLENDED sample across every market state and clause 3 is that nothing is
    pruned on that. Their old roll-up survives verbatim as `aggregate_status`
    so a reader who remembers Stage 3 refusing on a Gate 1 can see the verdict
    was moved rather than quietly dropped - and so a before/after against an
    audit written before the charter is legible instead of merely different.

    A configuration can therefore be CERTIFIED with a failing Gate 1. That is
    the intended effect: a strategy whose supervisor stands it down outside its
    quadrant never trades the blended sample the gate measured.

    **THE ONE EXCEPTION IS RUIN, added 2026-08-24.** `ruin_guard` is a HARD
    bar and it can refuse a certification that Gate R passed. Three
    configurations of `t3_braid_scalp_20260823` reached the incubator with
    `ruined: true` in sample - NQ 15m ended the charter window at -$78,868 and
    NQ 30m at -$104,159, on $100,000 of starting capital, drawing -194% and
    -206% - because Gate R binds profit factor and trade count inside one
    quadrant and has no drawdown or survival bar at all. Gate 1's 12% drawdown
    limit would have caught both, and clause 3 had removed its authority.

    The distinction that keeps this consistent with the charter: Gates 1-3
    grade EDGE QUALITY on a blended sample a regime-gated strategy never
    trades, which is why they are advisory. Ruin is not a grade. It is a
    statement that the account the trades were placed in reached zero, and a
    supervisor cannot stand a strategy down out of an account that no longer
    exists.

    When it fails, `status` is `FAILED_RUIN_CHECK` rather than `FAIL`. Every
    consumer refuses a non-PASS status, so the token cannot leak a promotion
    through, and it says on sight which bar was missed - "Gate R failed" and
    "the account was blown" send an operator to completely different work.
    """
    out = dict(audit)
    gates = dict(out.get("gates") or {})
    aggregate = out.get("status", NOT_EVALUATED)
    gates[GATE_R] = gate_r
    ruin = ruin_guard(in_sample_metrics)
    gates[GATE_RUIN] = ruin
    # Gate Q is ADVISORY by default since 2026-09-08, and is ALWAYS measured.
    #
    # It was added 2026-09-07 as a hard promotion requirement, with its own
    # docstring conceding that it "contradicts the charter's premise", that a
    # regime specialist "will fail this gate by construction rather than by
    # defect", and that the consequence to expect was "near-zero promotions".
    # That is what happened. Certifying on it asked every specialist this
    # pipeline is built to produce to also be an all-weather system.
    #
    # So it now scores like Gates 1-3, for the same stated reason they are
    # advisory: it grades market states a regime-gated strategy is STOOD DOWN
    # in and never trades. `require_all_quadrants=True` restores its authority
    # for a caller that wants an all-weather bar.
    #
    # It is recorded under `gates` either way, so the measurement never
    # disappears - the earlier "absence means waived" convention is replaced by
    # an explicit `advisory` marker, because a gate that is missing and a gate
    # that is present-but-toothless are different things to read off an audit,
    # and only one of them can be told apart from a gate that never ran.
    gate_q = all_quadrant_gate(holdout_profile)
    gate_q["advisory"] = not require_all_quadrants
    gates[GATE_Q] = gate_q
    out["gates"] = gates
    q_ok = (not require_all_quadrants) or gate_q["status"] == PASS
    if ruin["status"] != PASS:
        # A NOT EVALUATED ruin guard is not a pass either: it means nothing on
        # the metrics dict states the account survived, and this gate exists
        # because that failure was invisible.
        out["status"] = RUIN_CHECK_FAILED
    elif not q_ok:
        # Ordered AFTER ruin deliberately: a blown account is the more serious
        # finding and must not be relabelled by a robustness bar it also
        # missed. Its own token, so "failed in Q1" and "the account was blown"
        # never arrive under the same word.
        out["status"] = ALL_QUADRANTS_FAILED
    else:
        out["status"] = gate_r["status"]
    out["passed"] = (gate_r["status"] == PASS
                     and ruin["status"] == PASS and q_ok)
    out["all_quadrant_gate"] = gate_q
    out["ruin_guard"] = ruin
    out["verdict_gate"] = GATE_R
    out["verdict_basis"] = (
        "Regime-Switching Incubator Charter clause 3: the certification is "
        "Gate R - out-of-sample profit factor and trade count inside the "
        "quadrant Stage 1 designated. Gates 1, 2 and 3 are computed and "
        "reported as EVIDENCE and cannot fail a certification; they score the "
        "blended sample across every market state, which a regime-gated "
        "strategy does not trade. Gate Q joined them as ADVISORY on "
        "2026-09-08 for the same reason - it grades the three quadrants the "
        "supervisor stands the strategy down in - and is measured and "
        "reported on every audit. The ONE hard exception is the ruin guard: "
        "an in-sample account that reached zero refuses certification whatever "
        "Gate R measured, because no quadrant restriction makes a blown "
        "account survivable.")
    # The pre-charter verdict, kept rather than overwritten.
    out["aggregate_status"] = aggregate
    out["aggregate_passed"] = aggregate == PASS
    out["aggregate_is_advisory"] = True
    out["target_regime"] = gate_r.get("target_regime")
    out["target_quadrant"] = gate_r.get("quadrant")
    out["retention_scores"] = retention
    return out


# --------------------------------------------------------------------------
# Charter clause 6 - promotion and the cryptographic seal
# --------------------------------------------------------------------------

def _seal_hashes(source: Path, best_params: Path | None,
                 audit_file: Path | None,
                 metrics_file: Path | None) -> dict:
    """
    SHA-256 of everything a certification rests on, and a word when one is absent.

    Three artifacts rather than one, because they can be separated and each
    answers a question the others cannot. The CODE hash says the promoted
    module is the module that was run. The PARAMETER hash says which of the
    sweep's cells it was run with - the same code under a different winning
    cell is a different strategy with the same checksum. The AUDIT hash says
    which holdout produced the verdict beside them.

    A missing artifact is recorded as the literal `"NOT AVAILABLE"` rather than
    omitted. An absent key in a seal reads as a field nobody filled in, which
    is exactly the ambiguity a seal exists to remove.
    """
    def _h(path: Path | None) -> dict:
        if path is None or not Path(path).exists():
            return {"path": (str(path) if path else None),
                    "sha256": "NOT AVAILABLE"}
        return {"path": str(path), "sha256": sha256(Path(path))}

    return {
        "algorithm": "sha256",
        "strategy_code": _h(source),
        "winning_parameters": _h(best_params),
        "gate_audit": _h(audit_file),
        "metrics_handoff": _h(metrics_file),
        "rule": ("The three hashes are sealed together because they can be "
                 "separated: the same code under a different winning cell is "
                 "a different strategy with the same code checksum."),
    }


def seal_and_promote(strat_name: str, version: str, source: Path,
                     symbol: str, tf: str, params: dict, prov: dict,
                     audit_file: Path, metrics_file: Path | None,
                     threshold: float,
                     incubator: Path = INCUBATOR) -> dict:
    """
    Stage the certified version into the incubator and seal its `meta.json`.

    Charter clause 6. The directory is written by `backtest.promote.promote`
    rather than by a second implementation here: that function is where Version
    A is copied byte for byte, where Version B's wrapper is generated against
    the same ML feature hook the run used, and where the `risk` block keeps
    "no take-profit was modelled" distinct from "this strategy has no
    take-profit parameter". A Stage 3 copy of that logic would be free to
    disagree with Stage 5's about what was promoted, and the two would be
    compared by nobody.

    **Nothing is committed to git.** `commit=False`, always. Being in
    `approved_incubator/` is a record that a version was certified, not
    permission to trade it, and the commit belongs with the human reading the
    evidence at Stage 5 - which is the whole reason the stages do not chain.

    `promote` refuses a version whose audit is not PASS, which is exactly the
    behaviour wanted here: `audit_file` is the audit this stage just wrote, so
    the refusal reads the charter verdict (Gate R) and not the aggregate gates.
    A refusal is RETURNED as `{"promoted": False, "error": ...}` rather than
    raised - a certification that succeeded must not be thrown away because
    staging it hit a read-only checkout.
    """
    try:
        out = promote(
            strat=strat_name, version=version, source=Path(source),
            metrics_path=metrics_file, audit_path=Path(audit_file),
            symbol=symbol, timeframe=tf, params=None, threshold=threshold,
            notes=(f"Stage 3 certified {symbol} {tf} in "
                   f"{prov.get('target_regime') or 'an undeclared regime'}; "
                   f"staged by backtest/audit_gates.py, not committed."),
            variants_tested=prov.get("variants_tested"),
            force=False, commit=False, require_certification=True,
            incubator=Path(incubator))
    except (SystemExit, ValueError, FileNotFoundError, OSError) as exc:
        return {"promoted": False, "dir": None, "seal": None,
                "error": f"{type(exc).__name__}: {exc}"}

    dest = Path(out["dir"])
    best_params = prov.get("best_params_file")
    seal = _seal_hashes(
        source=dest / ("baseline.py" if version.upper() == "B" else "strat.py"),
        best_params=Path(best_params) if best_params else None,
        audit_file=Path(audit_file),
        metrics_file=Path(metrics_file) if metrics_file else None)
    seal["certified_symbol"] = symbol
    seal["certified_timeframe"] = tf
    seal["target_regime"] = prov.get("target_regime")
    seal["target_quadrant"] = prov.get("target_quadrant")
    seal["params"] = dict(params)
    seal["params_locked"] = bool(prov.get("params_locked"))

    # Re-read rather than patch `out["meta"]`: `promote` wrote the file and is
    # entitled to have written something this stage did not model, and the copy
    # in memory is not what the next reader opens.
    meta_p = dest / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["seal"] = seal
    meta_p.write_text(json.dumps(meta, indent=2, default=str) + "\n",
                      encoding="utf-8")

    return {"promoted": True, "dir": dest, "seal": seal,
            "files": [str(f) for f in out.get("files") or []],
            "meta": str(meta_p), "error": ""}


def profile_holdout(bars: pd.DataFrame, result, strat_name: str, symbol: str,
                    tf: str, version: str, out_dir: Path) -> dict | None:
    """
    The holdout's four-quadrant breakdown, which Gate R reads.

    Written as `regime_profile_<SYMBOL>_<TF>_version_<v>_holdout.json` beside
    the audit - a DIFFERENT filename from Stage 1's, deliberately. Stage 1
    profiled the in-sample window to the unsuffixed name, and letting this
    overwrite it would destroy the in-sample breakdown the quadrant was chosen
    from, leaving a file whose name says nothing about which window it
    describes sitting under a handoff that claims the quadrant was picked
    in-sample.

    Quiet, and it returns the same dict it writes. A profiler failure is
    RAISED, not swallowed: Gate R has nothing to read without it, and a
    certification that silently degraded to NOT EVALUATED because a profile
    could not be built would look identical to one whose quadrant was never
    designated.
    """
    profiler = RegimeProfiler(bars, result, strat_name, symbol, tf,
                              out_dir=str(out_dir),
                              version=f"{version.lower()}_holdout", quiet=True)
    return profiler.generate_profile()


def _dual(path: Path, bars: pd.DataFrame, symbol: str, tf: str, params: dict,
          cfg: BacktestConfig, ml: bool, threshold: float,
          strat_name: str) -> dict:
    return run_dual_version_backtest(
        str(path), bars, freq=tf, symbol=symbol, cfg=cfg, params=params,
        threshold=threshold, ml=ml, emit_reports=False, strat_name=strat_name)


def gate2_evidence(path: Path, symbol: str, tf: str, params: dict,
                   cfg: BacktestConfig, is_start: str, is_end: str,
                   result, args: argparse.Namespace,
                   grid: dict | None) -> dict:
    """
    Walk-forward efficiency and the bootstrap drawdown, both from IN-SAMPLE bars.

    The Monte Carlo resamples the in-sample trade sequence rather than the
    holdout's: it asks how bad the drawdown could have been had the same edge
    arrived in a different order, and the holdout has too few trades to answer
    that. The result is an OPTIMISTIC floor on the risk either way - resampling
    with replacement destroys the autocorrelation real losing streaks have.
    """
    out: dict = {}
    start_year = pd.Timestamp(is_start).year if is_start else 2010
    end_year = pd.Timestamp(is_end).year

    param_grid = None
    if grid:
        param_grid = expand_grid(grid)

    wfo = run_walk_forward_analysis(
        path, symbol, train_years=args.wfo_train, test_years=args.wfo_test,
        start_year=start_year, end_year=end_year, params=params,
        param_grid=param_grid, tf=tf, cfg=cfg)
    out["wfo"] = {k: v for k, v in wfo.items() if k != "folds"}
    out["wfo"]["n_folds"] = len(wfo.get("folds") or [])
    out["wfo_optimized"] = bool(param_grid)

    # `trade_returns_from_result` ALREADY divides by the starting capital - it
    # returns "per-trade returns as fractions of starting equity". Passing
    # `returns_are_dollars=True` made `run_monte_carlo_simulation` divide by
    # `initial_capital` a SECOND time, so every bootstrapped return reaching
    # the drawdown distribution was 100,000x too small.
    #
    # This is the same failure the 2026-08-24 sanitation exists to prevent,
    # arriving through a different door: it did not corrupt the array, it
    # SHRANK it, and a shrunk array bootstraps to a drawdown of roughly zero.
    # Every audit in the repository reported `max_drawdown_pct_at_confidence`
    # of -0.00% beside `prob_max_loss_breach: 0.0` - the strongest possible
    # safety reading, produced for every strategy regardless of its equity
    # path, while the runs behind those numbers included accounts that ended
    # BELOW ZERO. Measured on a series matched to a real NQ 15m run: -0.0029%
    # and a 0.0 breach probability as it was called, against -95.29% and a 1.0
    # breach probability correct.
    #
    # The guard is the docstring of the function that produces the array, not
    # a magnitude test here: a caller cannot tell a fraction from a dollar
    # figure by looking at it, which is exactly why the flag exists.
    returns = trade_returns_from_result(result)
    if returns is None or len(returns) == 0:
        out["monte_carlo"] = {"ok": False,
                              "error": "no in-sample trades to bootstrap"}
    else:
        out["monte_carlo"] = run_monte_carlo_simulation(
            returns, n_iterations=args.mc_iterations,
            initial_capital=cfg.initial_capital, returns_are_dollars=False,
            seed=args.mc_seed)
    return out


# --------------------------------------------------------------------------
# Which VERSION this pair qualified on
# --------------------------------------------------------------------------
# Stage 1 screens both versions and a pair survives on EITHER, recording which
# one carried it. That answer travelled as far as the Stage 2 summary row
# (`stage1_version`) and then stopped: `--ml` was a single global flag typed by
# an operator, so a pair that only Version B cleared was certified as Version A
# unless somebody remembered. Version B is a different claim from Version A -
# it is the rules plus a classifier that suppressed some of their entries - so
# certifying A against a Stage 1 designation that B earned tests a strategy
# nobody screened, passes or fails it on that basis, and stages the result
# under a name the metrics do not describe.
STAGE1_ML_VERSION = "B"


def resolve_version_b(target: dict | None, args: argparse.Namespace
                      ) -> tuple[bool, str]:
    """
    `(run_version_b, why)` for ONE pair.

    Three inputs, in precedence order:

      1. `--no-stage1-ml` - the operator overriding the handoff. It exists
         because a Version B survivor whose classifier cannot be rebuilt has to
         remain auditable as Version A, and the alternative would be editing
         `surviving_assets.json`.
      2. `--ml` - certify BOTH versions for every pair, whatever Stage 1 said.
         Still a global, still supported: it is a superset, so it cannot cause
         a B survivor to be certified as A.
      3. Stage 1's own answer, `stage1_version == "B"`, per pair. This is the
         one that was missing.

    A pair with NO recorded version is Version A, and says so. It is what every
    handoff written before the version travelled looks like, and defaulting it
    to B would run an hours-long classifier over pairs nobody asked it for.
    """
    if getattr(args, "no_stage1_ml", False):
        return bool(args.ml), ("--no-stage1-ml: Stage 1's version is ignored; "
                               f"--ml is {'on' if args.ml else 'off'}")
    if args.ml:
        return True, "--ml: both versions certified for every pair"
    version = str((target or {}).get("stage1_version") or "").strip().upper()
    if version == STAGE1_ML_VERSION:
        return True, (f"stage1_version={version}: this pair cleared Stage 1 on "
                      f"the ML-filtered version, so it is certified on it")
    if version:
        return False, f"stage1_version={version}: rule-based, no ML filter"
    return False, ("no stage1_version recorded for this pair; certifying "
                   "Version A only")


def certify_symbol(symbol: str, path: Path, tf: str, args: argparse.Namespace,
                   cfg_kwargs: dict, out_dir: Path, strat_name: str,
                   grid: dict | None, target: dict | None = None) -> dict:
    """
    One configuration, certified. Raises; the caller records it.

    `target` is the row `stage2_targets` produced for this pair, and is the
    fallback source of the designated quadrant when a `best_params` file
    written before the scope travelled carries none. Everything else comes off
    the locked parameter file.
    """
    t0 = time.time()
    print("\n" + "-" * 78)
    print(f"{symbol}  ·  {tf}")
    print("-" * 78)

    overrides = dict(parse_param(p) for p in args.param)
    params, prov = load_params(strat_name, symbol, out_dir, overrides,
                               args.defaults, tf=tf)
    print(f"  parameters : {params or '(module defaults)'}")
    print(f"               ({prov['params_source']})")
    n_variants = prov["variants_tested"]
    print(f"  variants   : "
          f"{n_variants if n_variants is not None else 'NOT RECORDED'}")

    cfg_kwargs, filter_source = _resolve_filters(cfg_kwargs, prov)
    if cfg_kwargs.get("exclude_days"):
        print(f"  exclude    : {list(cfg_kwargs['exclude_days'])} "
              f"({filter_source})")
        print("               Both the in-sample and the holdout runs below "
              "are filtered.")

    regime, regime_source = target_regime(prov, target, args.regime)
    prov["target_regime"] = regime
    prov["target_quadrant"] = quadrant_id(regime) if regime else None
    prov["target_regime_source"] = regime_source
    quad = prov["target_quadrant"]

    # WHICH VERSION, resolved per pair from Stage 1's handoff rather than from
    # a global flag. Printed beside the regime because the two together are
    # the whole certification target: this parameter set, on this version, in
    # this quadrant.
    use_ml, ml_reason = resolve_version_b(target, args)
    prov["stage1_version"] = (target or {}).get("stage1_version")
    prov["version_b_certified"] = bool(use_ml)
    prov["version_b_source"] = ml_reason
    print(f"  versions   : A" + ("  +  B (ML-filtered)" if use_ml else " only")
          + f"   [{ml_reason}]")
    print(f"  regime     : "
          + (f"{regime} ({quad})" if regime else "NOT DECLARED")
          + f"   [{regime_source}]")
    if not prov["params_locked"]:
        print(f"  ! LOCK BROKEN: {prov['params_lock_note']}")

    cfg = BacktestConfig(
        initial_capital=args.capital, contracts=args.contracts,
        slippage_ticks=args.slippage_ticks, flat_by_close=args.flat_by_close,
        variants_tested=prov["variants_tested"],
        notes=f"stage 3 audit {symbol} {tf}", **cfg_kwargs)
    prop_firm = _assert_no_prop_firm_rules(cfg)

    # -- Gate 1 evidence: the in-sample run -------------------------------
    print(f"\n  [1/3] in-sample   {args.is_start or 'lake start'} → {args.is_end}")
    is_bars = load_bars(symbol, tf, args.is_start, args.is_end)
    is_dual = _dual(path, is_bars, symbol, tf, params, cfg, use_ml,
                    args.threshold, strat_name)

    # -- Gate R and Gate 3 evidence: the holdout, run once ----------------
    print(f"  [3/3] holdout     {args.holdout_start} → "
          f"{args.holdout_end or 'present'}")
    ho_bars = load_bars(symbol, tf, args.holdout_start, args.holdout_end)
    ho_dual = _dual(path, ho_bars, symbol, tf, params, cfg, use_ml,
                    args.threshold, strat_name)

    # -- Gate 2 evidence: walk-forward and bootstrap, in-sample only ------
    print(f"  [2/3] robustness  walk-forward + {args.mc_iterations:,}-path "
          f"bootstrap")
    versions = {}
    for label, key in (("A", "version_a"), ("B", "version_b")):
        block = is_dual.get(key)
        if block is None:
            continue
        rb = gate2_evidence(path, symbol, tf, params, cfg, args.is_start,
                            args.is_end, block["result"], args,
                            grid if args.wfo_grid else None)
        ho_block = ho_dual.get(key) or {}
        audit = audit_acceptance_gates(
            block["metrics"], robustness=rb, holdout=ho_block.get("metrics"),
            version=label, name=strat_name)

        # Gate R. The holdout is profiled into the same four quadrants Stage 1
        # used, and the designated one is read off it. Profiled from the
        # HOLDOUT bars and the HOLDOUT result, so the quadrant boundaries are
        # drawn once and both the numerator and the denominator of every
        # quadrant statistic come from the same unseen window.
        ho_profile = (profile_holdout(ho_bars, ho_block["result"], strat_name,
                                      symbol, tf, label, out_dir)
                      if ho_block.get("result") is not None else None)
        gate_r = regime_gate(ho_profile, regime,
                             args.regime_min_pf, args.regime_min_trades,
                             regime_scores=prov.get("regime_scores"),
                             # PRE-DECLARED in sample and carried on the
                             # Stage 2 handoff. Passed rather than looked up
                             # from the holdout profile: the whole point of
                             # the fallback is that the alternative quadrant
                             # was named before these bars were read.
                             secondary_regimes=prov.get("secondary_regimes"))
        # Printed on the console the moment it is known, not left to be found
        # in the JSON. A Gate R that failed on sample size and one that failed
        # on edge print the same word on the certification leaderboard, and the
        # operator reading that table is the person who has to tell them apart.
        if gate_r.get("regime_starvation"):
            print(f"  {gate_r['regime_starvation']['message']}", flush=True)
        retention = retention_scores(block["metrics"],
                                     ho_block.get("metrics"))

        versions[label] = {
            "metrics_in_sample": _scalars(block["metrics"]),
            "metrics_holdout": _scalars(ho_block.get("metrics")),
            "robustness": _scalars_deep(rb),
            # The IN-SAMPLE metrics feed the ruin guard. They are the window
            # the strategy was actually run over to select these parameters,
            # so an account that died there died on the bars the certification
            # rests on.
            # The HOLDOUT profile feeds Gate Q, on the same window Gate R is
            # measured on: an all-quadrant bar scored in sample would be
            # graded on the bars the parameters were fitted to.
            "gate_audit": _scalars_deep(charter_audit(
                audit, gate_r, retention,
                in_sample_metrics=block.get("metrics"),
                holdout_profile=ho_profile,
                require_all_quadrants=getattr(
                    args, "require_all_quadrants", False))),
            # The WHOLE holdout breakdown, not only the certified quadrant.
            # Gate R scores one row of it; the other three are what a live
            # supervisor is being told to stand the strategy down in, and a
            # kill switch nobody can see the evidence for is an instruction on
            # trust.
            "regime_profile_holdout": _scalars_deep(ho_profile),
            "retention": retention,
            "dow_in_sample": day_of_week_breakdown(
                block["metrics"].get("trades")).to_dict("records"),
            "dow_holdout": day_of_week_breakdown(
                (ho_block.get("metrics") or {}).get("trades")).to_dict("records"),
        }

    _print_audit(symbol, versions)

    dow_ho = day_of_week_breakdown(
        ((ho_dual.get("version_a") or {}).get("metrics") or {}).get("trades"))
    if not dow_ho.empty:
        print("\n  DAY OF WEEK · Version A, HOLDOUT")
        print(format_day_of_week(dow_ho))

    payload = {
        "symbol": symbol,
        "timeframe": tf,
        "params": params,
        **prov,
        "in_sample": {"start": args.is_start, "end": args.is_end},
        # `end: null` is the charter default and MEANS the present. Written as
        # null with the word beside it rather than stamped with today's date,
        # because a date stamped here would claim the lake reached it.
        "holdout": {"start": args.holdout_start, "end": args.holdout_end,
                    "end_basis": ("the present - every bar the lake holds"
                                  if not args.holdout_end
                                  else "explicit --holdout-end")},
        "entry_filters": cfg_kwargs,
        "entry_filters_source": filter_source,
        # Charter clause 4, stated rather than left to be inferred from the
        # absence of a key.
        "prop_firm_rules": prop_firm,
        # Charter clause 3, likewise: what the verdict rests on, and what it
        # deliberately does not.
        "certification_rule": {
            "verdict_gate": GATE_R,
            "target_regime": regime,
            "target_quadrant": prov["target_quadrant"],
            "target_regime_source": regime_source,
            "min_profit_factor": float(args.regime_min_pf),
            "min_trades": int(args.regime_min_trades),
            "measured_on": "holdout",
            "aggregate_gates_are_advisory": True,
            "rule": ("PF >= min_profit_factor over >= min_trades trades "
                     "INSIDE the designated quadrant, on the holdout. Gates "
                     "1-3 are evidence and cannot fail a certification; no "
                     "configuration is pruned on a blended-sample metric."),
        },
        "wfo": {"train_years": args.wfo_train, "test_years": args.wfo_test,
                "optimized": bool(args.wfo_grid)},
        "monte_carlo": {"iterations": args.mc_iterations, "seed": args.mc_seed},
        "versions": versions,
        # The certified verdict per version, lifted out of the nested audit so
        # Stage 5 and a human reading the file see it without traversing.
        "status": {k: v["gate_audit"]["status"] for k, v in versions.items()},
        "passed": {k: bool(v["gate_audit"]["passed"])
                   for k, v in versions.items()},
    }

    # Written per PAIR as well as per contract. Stage 3 certifies one
    # timeframe per run and `gate_audit_<SYMBOL>.json` holds one verdict, so
    # certifying NQ at 30m after certifying it at 15m would otherwise replace
    # the 15m verdict with no trace. The suffixed file is the durable record;
    # the unsuffixed one stays because it is the path Stage 5's documented
    # command names, and it is overwritten deliberately and out loud.
    plain = out_dir / GATE_AUDIT_FILE.format(symbol=symbol)
    if plain.exists():
        try:
            was = read_stage(plain, 3, strat_name).get("timeframe")
        except Exception:                                         # noqa: BLE001
            was = None
        if was and str(was) != str(tf):
            print(f"  ! replacing {plain.name}, which certified {was}. The "
                  f"{was} verdict remains at "
                  f"{GATE_AUDIT_FILE.format(symbol=f'{symbol}_{was}')}.")
    dest = write_stage(out_dir / GATE_AUDIT_FILE.format(
        symbol=f"{symbol}_{tf}"), 3, strat_name, payload)
    write_stage(plain, 3, strat_name, payload)
    print(f"\n  audit      → {dest}")

    # Charter clause 6. Staged only for what Gate R certified, and only ever
    # staged - never committed.
    seals = {}
    for ver, block in versions.items():
        if not block["gate_audit"]["passed"]:
            continue
        if not args.promote:
            seals[ver] = {"promoted": False, "dir": None, "seal": None,
                          "error": "--no-promote"}
            continue
        seals[ver] = seal_and_promote(
            strat_name, ver, path, symbol, tf, params, prov, dest,
            metrics_file=None, threshold=args.threshold,
            incubator=Path(args.incubator))
        if seals[ver]["promoted"]:
            print(f"  sealed     → {seals[ver]['dir']}  "
                  f"(code {seals[ver]['seal']['strategy_code']['sha256'][:12]})")
        else:
            print(f"  ! Version {ver} certified but NOT staged: "
                  f"{seals[ver]['error']}", file=sys.stderr)
    if seals:
        payload["incubator"] = _scalars_deep(seals)
        write_stage(dest, 3, strat_name, payload)
        write_stage(plain, 3, strat_name, payload)

    print(f"  ({round(time.time() - t0, 1)}s)")
    return {"symbol": symbol, "timeframe": tf, "path": dest,
            "status": payload["status"], "passed": payload["passed"],
            "target_regime": regime,
            "target_quadrant": prov["target_quadrant"],
            "params": params,
            "params_locked": bool(prov["params_locked"]),
            "in_stage1": bool((target or {}).get("in_stage1", True)),
            # WHICH version Stage 1 qualified this pair on, and whether the
            # ML-filtered version was actually certified for it. Carried so
            # `stage3_audit_summary.json` records the version that won Stage 1
            # beside the versions that were audited: a B survivor certified as
            # A only is a gap, and it has to be visible in the summary rather
            # than discoverable by comparing two files.
            "stage1_version": (target or {}).get("stage1_version"),
            "version_b_certified": bool(prov.get("version_b_certified")),
            "version_b_source": prov.get("version_b_source"),
            "incubator": seals,
            # Gate R and the three advisory gates individually, not only the
            # rolled-up verdict. A FAIL and a NOT EVALUATED are fixed by
            # different work - the first is a result about the strategy, the
            # second is a run that has not been done - and one PASS/FAIL column
            # makes them indistinguishable.
            "gates": {ver: {g: v["gate_audit"]["gates"][g]["status"]
                            for g in ("gate1", "gate2", "gate3", GATE_R)}
                      for ver, v in versions.items()},
            "regime_measured": {
                ver: (v["gate_audit"]["gates"][GATE_R].get("measured") or {})
                for ver, v in versions.items()},
            # WHY Gate R failed, not only that it did. A quadrant that
            # starved out of sample and one whose edge inverted both print
            # `FAIL` on the table, and they are fixed by completely different
            # work - the first is a strategy that never met its own
            # environment again, the second is a strategy that did and lost.
            # Carried onto the summary so an orchestrator reporting a failed
            # configuration can say which happened without re-reading the
            # per-pair audit.
            "regime_starvation": {
                ver: ((v["gate_audit"]["gates"][GATE_R]
                       .get("regime_starvation") or {}).get("message"))
                for ver, v in versions.items()},
            # WHICH quadrant carried the certification. "primary" or
            # "secondary_starvation_fallback" - a strategy certified on its
            # fallback is trading a quadrant Stage 1 ranked second, and a
            # reader who cannot see that from the handoff would have to open
            # the per-pair audit to find out which environment is live.
            "certified_on": {
                ver: v["gate_audit"]["gates"][GATE_R].get("certified_on")
                for ver, v in versions.items()},
            "primary_quadrant": {
                ver: v["gate_audit"]["gates"][GATE_R].get("primary_quadrant")
                for ver, v in versions.items()},
            "retention": {ver: v["retention"]["metrics"]
                          for ver, v in versions.items()},
            "exclude_days": list(cfg_kwargs.get("exclude_days") or [])}


def _scalars(metrics: dict | None) -> dict | None:
    """Metrics without the trade frame or the equity series."""
    if not metrics:
        return None
    return {k: v for k, v in metrics.items()
            if not isinstance(v, (pd.DataFrame, pd.Series))}


def _scalars_deep(obj):
    """The same, recursively, so a nested result is JSON-writable."""
    if isinstance(obj, dict):
        return {k: _scalars_deep(v) for k, v in obj.items()
                if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(obj, (list, tuple)):
        return [_scalars_deep(v) for v in obj]
    return obj


def _print_audit(symbol: str, versions: dict) -> None:
    """
    The gate table in full - every criterion, measured against its bar.

    Gate R prints FIRST and under its own heading, because it is the verdict.
    The three that follow print under an ADVISORY heading saying in words that
    they cannot fail a certification: a reader who has seen this table before
    the charter would otherwise read a `FAIL` on Gate 1 above a `CERTIFIED`
    verdict as a bug in the roll-up.
    """
    W = 78

    def _checks(gate: dict) -> None:
        for c in gate.get("checks") or []:
            measured, required = criterion_text(c)
            print(f"      {c['label']:<28}{measured:>10}  "
                  f"{required:<15}{c['status']}")
            if c.get("note"):
                print(f"        ! {c['note']}")

    print("\n" + "=" * W)
    print(f"GATE CERTIFICATION · {symbol}")
    print("=" * W)
    for label, block in versions.items():
        audit = block["gate_audit"]
        print(f"\n  Version {label}   CERTIFICATION: {audit['status']}")

        gate = audit["gates"][GATE_R]
        print(f"\n    {gate['name']:<40}{gate['status']}   <- THE VERDICT")
        target = gate.get("target_regime")
        scope = (f"{target} ({gate.get('quadrant')})" if target
                 else "NOT DECLARED")
        print(f"      {'target quadrant':<28}{scope}")
        _checks(gate)
        if gate.get("note"):
            print(f"      ! {gate['note']}")

        print(f"\n    ADVISORY EVIDENCE — these cannot fail a certification")
        print(f"    (blended sample, every market state; roll-up was "
              f"{audit.get('aggregate_status', NOT_EVALUATED)})")
        for gk in ("gate1", "gate2", "gate3"):
            gate = audit["gates"][gk]
            print(f"    {gate['name']:<40}{gate['status']}")
            _checks(gate)

        rows = (block.get("retention") or {}).get("metrics") or {}
        if rows:
            print("\n    RETENTION · in-sample → holdout (reported, not scored)")
            for row in rows.values():
                ret = row.get("retention")
                print(f"      {row['label']:<28}"
                      f"{_fmt(row.get('in_sample')):>10} → "
                      f"{_fmt(row.get('holdout')):>10}   "
                      f"{('n/a' if ret is None else f'{ret:.2f}x'):>8}")

        if audit["status"] == NOT_EVALUATED:
            print("\n    ⚠ NOT EVALUATED is not a pass. Gate R had no "
                  "designated quadrant to\n      certify in, so nothing was "
                  "measured.")


def _fmt(value) -> str:
    """One console cell. `n/a` for a number nobody could compute."""
    v = _num(value)
    return "n/a" if v is None else f"{v:.2f}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 3/5 — certify Stage 2's LOCKED parameters on the "
                    f"untouched holdout ({HOLDOUT_START} to the present). The "
                    "verdict is Gate R: profit factor and trade count inside "
                    "the ONE quadrant Stage 1 designated. Gates 1 "
                    "(in-sample), 2 (walk-forward + Monte Carlo) and 3 (OOS "
                    "retention) are computed and reported as evidence and "
                    "cannot fail a certification — nothing is pruned on a "
                    "blended-sample metric, and no prop-firm rule is applied. "
                    "A pass is sealed with SHA-256 and staged into the "
                    "incubator, never committed.")
    p.add_argument("--strat", required=True)
    p.add_argument("--symbols", default=None,
                   help="NQ, a list, or ALL. Default: the exact (symbol, "
                        "timeframe) pairs stage2_summary.json optimised.")
    p.add_argument("--tf", "--timeframe", dest="tf", default=None)
    # The charter window, from `pipeline` rather than restated here. Three
    # copies of these dates across the stages would be one edit away from a
    # Stage 2 sweep that runs a day into a holdout Stage 3 then scores as
    # unseen.
    p.add_argument("--is-start", default=CHARTER_IS_START,
                   help=f"In-sample start (default {CHARTER_IS_START})")
    p.add_argument("--is-end", default=CHARTER_IS_END,
                   help=f"In-sample end (default {CHARTER_IS_END}). Required, "
                        f"and must fall before --holdout-start.")
    p.add_argument("--require-all-quadrants", dest="require_all_quadrants",
                   action="store_true", default=False,
                   help="Give Gate Q, the all-quadrant robustness bar, the "
                        "authority to REFUSE a certification. Off by default: "
                        "Gate Q is measured and reported on every audit but "
                        "is ADVISORY, like Gates 1-3, because it grades the "
                        "three quadrants a regime-gated strategy is stood "
                        "down in and never trades. It requires positive "
                        "expectancy and per-trade Sharpe >= "
                        f"{ALL_QUADRANT_MIN_SHARPE:.2f} in EACH of Q1-Q4 over "
                        f">= {ALL_QUADRANT_MIN_TRADES} holdout trades. Pass "
                        "this only for an all-weather mandate: a regime "
                        "SPECIALIST fails it by construction, which is why it "
                        "no longer certifies anything on its own.")
    p.add_argument("--holdout-start", default=HOLDOUT_START,
                   help=f"Holdout start (default {HOLDOUT_START})")
    p.add_argument("--holdout-end", default=None,
                   help="Holdout end. Default: THE PRESENT — every bar the "
                        "lake holds. A hardcoded end silently stops certifying "
                        "against the newest bars once the year rolls over, and "
                        "the verdict looks the same either way.")
    p.add_argument("--stage2-summary", default=None,
                   help=f"Path to {STAGE2_SUMMARY_FILE}. Default: the "
                        f"strategy's pipeline directory. Named explicitly, a "
                        f"missing file is an error; found by default, it is "
                        f"not.")
    p.add_argument("--regime", default=None,
                   help="Override the designated quadrant Gate R certifies "
                        "in, by its full name (e.g. 'High Volatility / "
                        "Trending'). For a configuration whose Stage 1 scope "
                        "never travelled; recorded as an override.")
    p.add_argument("--regime-min-pf", type=float,
                   default=MIN_REGIME_PROFIT_FACTOR,
                   help=f"Gate R profit factor bar inside the designated "
                        f"quadrant (default {MIN_REGIME_PROFIT_FACTOR:.2f}, "
                        f"the same bar Stage 1 screened on)")
    p.add_argument("--regime-min-trades", type=int, default=MIN_REGIME_TRADES,
                   help=f"Gate R trade floor inside the designated quadrant "
                        f"(default {MIN_REGIME_TRADES})")
    p.add_argument("--no-promote", dest="promote", action="store_false",
                   help="Certify, but do not stage a passing version into "
                        "the incubator. Nothing is git-committed either way — "
                        "the commit is Stage 5's, in front of a human.")
    p.set_defaults(promote=True)
    p.add_argument("--incubator", default=str(INCUBATOR),
                   help=f"Where a certified version is staged (default "
                        f"{INCUBATOR})")
    p.add_argument("--rebuild-summary", action="store_true",
                   help="Rebuild stage3_audit_summary.json from the "
                        "gate_audit_<SYMBOL>_<TF>.json files already on disk, "
                        "across EVERY timeframe. Reads no bars and re-scores "
                        "no gate — it recovers the index for campaigns whose "
                        "earlier timeframes an older Stage 3 overwrote.")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="Override a parameter from the Stage 2 winner")
    p.add_argument("--defaults", action="store_true",
                   help="Certify the module's DEFAULT_PARAMS instead of "
                        "Stage 2's winner. Say so deliberately: without this, "
                        "a missing best_params file is an error rather than a "
                        "silent fallback.")
    p.add_argument("--ml", action="store_true",
                   help="Certify Version B (the ML-filtered pipeline) for "
                        "EVERY pair. Not needed for a pair Stage 1 qualified "
                        "on B - that is resolved per pair from the handoff.")
    p.add_argument("--no-stage1-ml", action="store_true",
                   help="Ignore stage1_version and do NOT run Version B on a "
                        "pair that qualified on it. For a B survivor whose "
                        "classifier cannot be rebuilt; the audit records that "
                        "the version was overridden.")
    p.add_argument("--ml-threshold", "--threshold", dest="threshold",
                   type=float, default=ML_THRESHOLD_DEFAULT,
                   help=(f"Version B: P(win) at or above which an entry is "
                         f"kept (default {ML_THRESHOLD_DEFAULT})"))
    p.add_argument("--wfo-train", type=int, default=2,
                   help="Walk-forward train window in years (default 2)")
    p.add_argument("--wfo-test", type=int, default=1,
                   help="Walk-forward test window in years (default 1)")
    p.add_argument("--wfo-grid", action="store_true",
                   help="Re-select parameters per fold from the module's "
                        "PARAM_GRID. This is the version of the walk-forward "
                        "that measures overfitting; it costs a full sweep per "
                        "fold.")
    p.add_argument("--mc-iterations", type=int, default=1000)
    p.add_argument("--mc-seed", type=int, default=42)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--flat-by-close", action="store_true")
    p.add_argument("--out-dir", default=None)
    add_filter_args(p)
    return p


def certification_leaderboard(results: list[dict]) -> str:
    """
    STAGE 3 GATE CERTIFICATION LEADERBOARD - the table the stage ends on.

    One row per (contract, version). `GATE R` is the verdict and carries the
    quadrant it was measured in beside the two numbers it was measured on, so
    the row states the whole claim: this edge, in this environment, over this
    many unseen trades, at this profit factor. A PF column with no quadrant
    column would be a blended-sample number under a regime-gated verdict.

    Gates 1, 2 and 3 are still shown, still individually, and still say NOT
    EVAL when they were not run - but under an `advisory` heading, because they
    fail for different reasons, are fixed by different work, and since the
    charter cannot fail a certification. Collapsing them into one column would
    make a walk-forward that never ran indistinguishable from one that did not
    hold up, and only the second is a statement about the strategy.

    `FINAL STATUS` is `audit["passed"]`, which is Gate R alone - so a NOT
    EVALUATED reads as NOT CERTIFIED, which is what Stage 5 enforces. Passing
    rows sort first, then by symbol and version, so a screen of twenty
    contracts opens on whatever cleared.
    """
    from backtest.pipeline import leaderboard

    short = {PASS: "PASS", FAIL: "FAIL"}
    body = []
    for r in results:
        for ver in sorted(r["status"]):
            g = (r.get("gates") or {}).get(ver, {})
            measured = (r.get("regime_measured") or {}).get(ver) or {}
            pf = _num(measured.get("profit_factor"))
            n = measured.get("trade_count")
            body.append([
                r["symbol"], r.get("timeframe", "?"), ver,
                r.get("target_quadrant") or "--",
                short.get(g.get(GATE_R), "NOT EVAL"),
                "n/a" if pf is None else f"{pf:.2f}",
                "--" if n is None else f"{int(n):,}",
                *(short.get(g.get(k), "NOT EVAL")
                  for k in ("gate1", "gate2", "gate3")),
                "CERTIFIED" if r["passed"].get(ver) else "NOT CERTIFIED",
                ", ".join(WEEKDAY_NAMES[int(d)]
                          for d in (r.get("exclude_days") or [])) or "none",
            ])
    body.sort(key=lambda row: (row[10] != "CERTIFIED", row[0], row[2]))
    return leaderboard(
        "STAGE 3 GATE CERTIFICATION LEADERBOARD",
        ["SYMBOL", "TF", "VER", "QUAD", "GATE R (OOS REGIME)", "OOS PF",
         "OOS N", "GATE 1 (IS)", "GATE 2 (WFO/MC)", "GATE 3 (OOS)",
         "FINAL STATUS", "EXCLUDED DAYS"],
        body,
        align=["<", "<", "<", "<", ">", ">", ">", ">", ">", ">", ">", "<"],
        empty="nothing was certified — no contract completed the audit")


def _row_key(row: dict) -> tuple[str, str, str]:
    """A configuration's identity on the summary: contract, timeframe, version."""
    return (str(row.get("symbol") or ""), str(row.get("timeframe") or ""),
            str(row.get("version") or ""))


def load_previous_summary(out_dir: Path, strat_name: str) -> dict | None:
    """
    The `stage3_audit_summary.json` an earlier timeframe's run left behind.

    Read through `read_stage`, so a file written by another stage or belonging
    to another strategy is REFUSED rather than merged - carrying one
    strategy's certifications into another's summary is precisely what the
    stage guard exists to prevent, and the Discord card would post the result.

    A file that cannot be read is treated as ABSENT rather than fatal. This
    run's verdicts are already on disk as `gate_audit_<SYMBOL>_<TF>.json` and
    must not be thrown away because an earlier run left a truncated index.
    """
    path = Path(out_dir) / STAGE3_SUMMARY_FILE
    if not path.exists():
        return None
    try:
        return read_stage(path, 3, strat_name)
    except Exception as e:                                        # noqa: BLE001
        print(f"  ! ignoring the existing {path.name} ({type(e).__name__}: "
              f"{e}); this run's summary replaces it.", file=sys.stderr)
        return None


def merge_stage3_rows(previous: list[dict], current: list[dict],
                      run_timeframes: set[str]) -> tuple[list[dict], int]:
    """
    This run's rows plus every row an earlier run certified at ANOTHER
    timeframe. Returns `(rows, carried)`.

    Stage 3 certifies ONE timeframe per invocation - a gate audit is a verdict
    about one (parameters, timeframe) pair - so a multi-timeframe pipeline
    calls it once per timeframe. Without this merge each call OVERWROTE the
    summary, and the card announced whichever timeframe happened to run last:
    a run that certified CL at 5m and again at 15m posted one of them, the
    other reached nobody, and the per-pair audit sat on disk unread with
    nothing raising.

    **This run is authoritative for the timeframes it ran.** Every prior row
    at one of them is DROPPED rather than merged: a re-certification that no
    longer covers a contract - Stage 2 stopped optimising it, or the audit
    raised - must not leave the earlier verdict standing beside the new ones,
    where it reads as current. Rows at other timeframes are carried verbatim,
    transcribed and never re-scored, exactly like everything else in this file.
    """
    keys = {_row_key(r) for r in current}
    carried = [r for r in previous
               if str(r.get("timeframe") or "") not in run_timeframes
               and _row_key(r) not in keys]
    return list(current) + carried, len(carried)


def consolidated_audits(rows: list[dict]) -> list[dict]:
    """
    The index over the per-pair `gate_audit_<SYMBOL>_<TF>.json` files.

    One entry per configuration that reached a verdict, naming the file, its
    SHA-256 and the verdict inside it. The per-pair audits stay
    AUTHORITATIVE - this is a list of WHERE they are, so a reader, the Discord
    card and Stage 5 can find every timeframe's certification without globbing
    a directory in which a superseded sweep's audit sits indistinguishable
    from a current one.

    A row with no audit file is omitted here and kept in `results`: an index
    entry pointing at nothing is worse than no entry, and the NOT AUDITED row
    is already on the record where the card reads it. `exists` is checked
    rather than assumed - the pipeline directory is one artifact root and a
    hand-cleaned one leaves rows whose file is gone.
    """
    out: list[dict] = []
    for row in rows:
        path = row.get("audit_file")
        if not path:
            continue
        out.append({
            "symbol": row.get("symbol"),
            "timeframe": row.get("timeframe"),
            "version": row.get("version"),
            "path": str(path),
            "sha256": row.get("audit_sha256"),
            "exists": Path(str(path)).exists(),
            "status": row.get("status"),
            "certified": bool(row.get("certified")),
            "gate_regime": row.get("gate_regime"),
        })
    return out


def write_stage3_summary(strat_name: str, out_dir: Path, results: list[dict],
                         errors: list[dict], skipped: list[dict],
                         args: argparse.Namespace,
                         targets: list[dict], target_source: str,
                         source_path: str | Path | None = None) -> Path:
    """
    `stage3_audit_summary.json` - the stage's own handoff over the whole run.

    Written through `pipeline.write_stage`, so `read_stage` can refuse a file
    written by another stage or belonging to another strategy. That matters
    more here than anywhere else in the pipeline: this is what
    `discord_reporter.py --stage 3` posts, and a certification card announcing
    one strategy's holdout under another's name is the artifact nobody
    cross-checks.

    It computes NOTHING. Every value is transcribed from a
    `gate_audit_<SYMBOL>_<TF>.json` this run already wrote, so the summary can
    never disagree with the per-contract verdicts it indexes - the audits stay
    authoritative and this is the index over them.

    **Errors and skips are rows, not omissions**, for the same reason Stage 2's
    matrix keeps them. `coverage` counts the configurations the stage was ASKED
    to certify against those it reached a verdict on, and a shortfall is a run
    failure or a missing Stage 2 parameter set - never a screening result,
    because this stage screens nothing on an aggregate. A shorter table reads
    as a complete one.

    **It MERGES across timeframes rather than overwriting.** Stage 3 certifies
    one timeframe per invocation, so a multi-timeframe pipeline runs it several
    times into this one file; written as a plain overwrite it kept only the
    last, and every earlier timeframe's certification vanished from the index
    (and from the Discord card) while its `gate_audit_<SYMBOL>_<TF>.json` sat
    on disk unread. `merge_stage3_rows` carries the other timeframes' rows
    forward verbatim and lets this run replace its own; `runs` records what
    each invocation covered, so `coverage` describes the whole certification
    campaign rather than its final slice; and `audits` is the consolidated
    index over the per-pair files, which remain the authoritative verdict.
    """
    rows = []
    for r in results:
        for ver in sorted(r["status"]):
            g = (r.get("gates") or {}).get(ver, {})
            measured = (r.get("regime_measured") or {}).get(ver) or {}
            ret = (r.get("retention") or {}).get(ver) or {}
            staged = (r.get("incubator") or {}).get(ver) or {}
            seal = staged.get("seal") or {}
            rows.append({
                "symbol": r["symbol"],
                "timeframe": r.get("timeframe"),
                "version": ver,
                # The version STAGE 1 qualified the pair on, beside the version
                # this row audits. They are different questions and only one of
                # them was ever recorded: `version` is what ran here,
                # `stage1_version` is what earned the survivorship. Stage 5
                # promotes on the first and the charter designated on the
                # second, so a summary that carried only one of them cannot say
                # whether they agree.
                "stage1_version": r.get("stage1_version"),
                "stage1_version_certified": (
                    None if not r.get("stage1_version")
                    else str(r.get("stage1_version")).upper() == ver),
                "version_b_certified": bool(r.get("version_b_certified")),
                "version_b_source": r.get("version_b_source"),
                "status": r["status"][ver],
                "certified": bool(r["passed"].get(ver)),
                "target_regime": r.get("target_regime"),
                "quadrant": r.get("target_quadrant"),
                "gate_regime": g.get(GATE_R, NOT_EVALUATED),
                "oos_profit_factor": _num(measured.get("profit_factor")),
                "oos_trade_count": measured.get("trade_count"),
                "oos_win_rate": _num(measured.get("win_rate")),
                # The blended in-sample and holdout profit factors, side by
                # side with the quadrant numbers above. They are the pair the
                # Discord card prints, and they are NOT what Gate R scored -
                # the field names say which is which rather than leaving one
                # `profit_factor` to be read as either.
                "is_profit_factor": _num(
                    (ret.get("profit_factor") or {}).get("in_sample")),
                "holdout_profit_factor": _num(
                    (ret.get("profit_factor") or {}).get("holdout")),
                "retention": {k: v.get("retention") for k, v in ret.items()},
                "gate1": g.get("gate1", NOT_EVALUATED),
                "gate2": g.get("gate2", NOT_EVALUATED),
                "gate3": g.get("gate3", NOT_EVALUATED),
                # Present only when Gate R failed on the TRADE COUNT. `None`
                # is "the quadrant was not starved", which is a different
                # statement from "the strategy passed" - the status field
                # above is what says that.
                "regime_starvation": (r.get("regime_starvation")
                                      or {}).get(ver),
                "certified_on": (r.get("certified_on") or {}).get(ver),
                "primary_quadrant": (r.get("primary_quadrant")
                                     or {}).get(ver),
                "params": r.get("params") or {},
                "params_locked": bool(r.get("params_locked")),
                "in_stage1": bool(r.get("in_stage1", True)),
                "exclude_days": list(r.get("exclude_days") or []),
                "audit_file": str(r["path"]),
                "audit_sha256": (sha256(Path(r["path"]))
                                 if Path(r["path"]).exists()
                                 else "NOT AVAILABLE"),
                "incubator_dir": (str(staged["dir"])
                                  if staged.get("dir") else None),
                "incubator_error": staged.get("error") or "",
                "seal": seal or None,
                "error": "",
            })
    for e in errors + skipped:
        rows.append({
            "symbol": e.get("symbol"), "timeframe": e.get("timeframe"),
            "version": None,
            # Not FAIL. A configuration whose audit raised, or whose Stage 2
            # sweep never produced parameters, did not reach a gate - and
            # "the run broke" must not share a token with "the edge did not
            # generalise".
            "status": "NOT AUDITED", "certified": False,
            "target_regime": e.get("optimal_regime"),
            "quadrant": e.get("quadrant"),
            "gate_regime": NOT_EVALUATED,
            "oos_profit_factor": None, "oos_trade_count": None,
            "oos_win_rate": None, "is_profit_factor": None,
            "holdout_profit_factor": None, "retention": {},
            "gate1": NOT_EVALUATED, "gate2": NOT_EVALUATED,
            "gate3": NOT_EVALUATED, "regime_starvation": None,
            "certified_on": None, "primary_quadrant": None,
            "params": {}, "params_locked": False,
            "in_stage1": bool(e.get("in_stage1", True)),
            "exclude_days": [], "audit_file": None,
            "audit_sha256": "NOT AVAILABLE", "incubator_dir": None,
            "seal": None,
            "error": e.get("error", ""),
        })

    # This run's rows first, then everything an earlier timeframe's run
    # certified. The merge is what makes this file the campaign's index rather
    # than the last invocation's.
    previous = load_previous_summary(out_dir, strat_name)
    run_tf = str(args.tf)
    rows, carried = merge_stage3_rows(
        (previous or {}).get("results") or [], rows, {run_tf})
    if carried:
        print(f"  merged     {carried} row(s) from earlier timeframe(s) "
              f"already in {STAGE3_SUMMARY_FILE}")

    audited = {(r["symbol"], r.get("timeframe")) for r in results}
    # What THIS invocation covered, keyed by the timeframe it certified.
    # `coverage` below is summed over these, so a campaign of three Stage 3
    # runs reports what all three did rather than what the last one did.
    runs = {k: v for k, v in ((previous or {}).get("runs") or {}).items()
            if str(k) != run_tf}
    runs[run_tf] = {
        "timeframe": run_tf,
        "targets": len(targets) or len(results) + len(errors) + len(skipped),
        "audited": len(audited),
        "certified": sum(1 for r in rows
                         if r.get("certified")
                         and str(r.get("timeframe") or "") == run_tf),
        "errors": len(errors),
        "skipped": len(skipped),
        "complete": not (errors or skipped),
        "target_source": target_source,
        "in_sample": {"start": args.is_start, "end": args.is_end},
        "holdout": {"start": args.holdout_start, "end": args.holdout_end},
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    timeframes = sorted({str(r.get("timeframe")) for r in rows
                         if r.get("timeframe")})

    payload = {
        "strategy": strat_name,
        # The module Stage 5 promotes FROM. Recorded because the Discord card
        # prints the exact `promote.py` command for a certified configuration,
        # and a command missing --source is a command nobody can paste.
        "strategy_source": (str(source_path) if source_path else None),
        "in_sample": {"start": args.is_start, "end": args.is_end},
        # The bar Version B's holdout inference ran at. Gate R's verdict on a
        # Version B candidate is a verdict on the FILTER, and the filter is
        # this number - a certification that does not record it cannot be
        # matched against the ML_THRESHOLD promote.py bakes into the module
        # it deploys, which is the one comparison that says the promoted
        # strategy is the certified one.
        # getattr, not args.threshold. This function is also reached by
        # `rebuild_stage3_summary`, which reconstructs a partial namespace from
        # gate_audit files on disk - a rebuild must not crash because the
        # rebuilt args carry no CLI flag. Absent records None, which is the
        # honest answer: not recorded is not 0.48.
        "ml_threshold": (float(t) if (t := getattr(args, "threshold", None))
                         is not None else None),
        "holdout": {
            "start": args.holdout_start, "end": args.holdout_end,
            "end_basis": ("the present - every bar the lake holds"
                          if not args.holdout_end else "explicit --holdout-end"),
        },
        "charter": {"start": CHARTER_IS_START, "end": CHARTER_IS_END,
                    "holdout_starts": HOLDOUT_START},
        # The timeframe THIS run certified, kept for every reader that has
        # only ever seen one; `timeframes` is the whole campaign, and the two
        # are separate fields rather than one that changes meaning.
        "timeframe": args.tf,
        "timeframes": timeframes,
        "target_source": target_source,
        "certification_rule": {
            "verdict_gate": GATE_R,
            "min_profit_factor": float(args.regime_min_pf),
            "min_trades": int(args.regime_min_trades),
            "measured_on": "holdout, inside the designated quadrant only",
            "aggregate_gates_are_advisory": True,
        },
        "prop_firm_rules": {"applied": False,
                            "fields_checked": list(PROP_FIRM_FIELDS)},
        # Summed over every Stage 3 invocation in `runs`, not only this one.
        # With a single timeframe these are exactly the numbers they always
        # were; with three they describe the campaign, which is what the file
        # now indexes.
        "coverage": {
            "targets": sum(int(r.get("targets") or 0) for r in runs.values()),
            "audited": sum(int(r.get("audited") or 0) for r in runs.values()),
            "certified": sum(1 for r in rows if r.get("certified")),
            "errors": sum(int(r.get("errors") or 0) for r in runs.values()),
            "skipped": sum(int(r.get("skipped") or 0) for r in runs.values()),
            "complete": all(bool(r.get("complete")) for r in runs.values()),
            "timeframes": sorted(runs),
            "rule": ("Stage 3 prunes nothing on an aggregate metric. A row "
                     "that is NOT AUDITED did not reach a gate: the run "
                     "broke, Stage 2 left no parameter set, or Stage 2 "
                     "pruned the whole grid as fragile. It is never a "
                     "screening decision by THIS stage, and the row's "
                     "reason says which of the three it was."),
        },
        "runs": runs,
        # The consolidated index over the per-pair audits. Those files remain
        # the verdict a promotion rests on; this says where each one is.
        "audits": consolidated_audits(rows),
        "results": rows,
    }
    return write_stage(Path(out_dir) / STAGE3_SUMMARY_FILE, 3, strat_name,
                       payload)


def audit_to_result(blob: dict, path: Path) -> dict:
    """
    One `gate_audit_<SYMBOL>_<TF>.json` back in the shape `certify_symbol`
    returns, so `write_stage3_summary` can index it without re-running it.

    A pure transcription of the audit, field for field. Nothing is recomputed
    and no verdict is re-derived: the audit is the authority, and a rebuilt
    summary that scored anything itself would be free to disagree with the
    file it claims to index.
    """
    versions = blob.get("versions") or {}
    gate_names = ("gate1", "gate2", "gate3", GATE_R)

    def gates_of(v: dict) -> dict:
        g = ((v.get("gate_audit") or {}).get("gates") or {})
        return {name: (g.get(name) or {}).get("status", NOT_EVALUATED)
                for name in gate_names}

    def gate_r_of(v: dict) -> dict:
        return (((v.get("gate_audit") or {}).get("gates") or {})
                .get(GATE_R) or {})

    # The top-level `status`/`passed` are lifted out of the nested audits by
    # `certify_symbol`; when they are absent the per-version block still holds
    # them verbatim. Reading them from there is transcription, not a second
    # opinion - it is the same field, one level down.
    status = blob.get("status") or {
        ver: (v.get("gate_audit") or {}).get("status", NOT_EVALUATED)
        for ver, v in versions.items()}
    passed = blob.get("passed") or {
        ver: bool((v.get("gate_audit") or {}).get("passed"))
        for ver, v in versions.items()}

    return {
        "symbol": blob.get("symbol"),
        "timeframe": blob.get("timeframe"),
        "path": path,
        "status": status,
        "passed": passed,
        "target_regime": blob.get("target_regime"),
        "target_quadrant": blob.get("target_quadrant"),
        "params": blob.get("params") or {},
        "params_locked": bool(blob.get("params_locked")),
        "in_stage1": bool(blob.get("in_stage1", True)),
        "incubator": blob.get("incubator") or {},
        "gates": {ver: gates_of(v) for ver, v in versions.items()},
        "regime_measured": {ver: (gate_r_of(v).get("measured") or {})
                            for ver, v in versions.items()},
        "regime_starvation": {
            ver: ((gate_r_of(v).get("regime_starvation") or {}).get("message"))
            for ver, v in versions.items()},
        "certified_on": {ver: gate_r_of(v).get("certified_on")
                         for ver, v in versions.items()},
        "retention": {ver: ((v.get("retention") or {}).get("metrics") or {})
                      for ver, v in versions.items()},
        "exclude_days": list((blob.get("entry_filters") or {})
                             .get("exclude_days") or []),
    }


def rebuild_stage3_summary(strat_name: str, out_dir: Path,
                           args: argparse.Namespace) -> int:
    """
    Rebuild `stage3_audit_summary.json` from the per-pair audits on disk.

    Stage 2's `--reuse-scan` for Stage 3, and it exists for the same reason:
    the expensive half of the stage is already on disk and the cheap half is
    an index over it. It reads NO bars and runs NO simulation.

    It is needed because the summary used to be OVERWRITTEN by each
    invocation. Stage 3 certifies one timeframe per run, so a campaign that
    audited CL at 5m, 15m and 30m left three verdicts in three
    `gate_audit_<SYMBOL>_<TF>.json` files and a summary describing only the
    last - and the Discord card, which reads the summary, announced one
    timeframe and silently dropped the certifications from the others. The
    merge in `write_stage3_summary` stops that happening again; this recovers
    the campaigns it already happened to.

    Only the SUFFIXED files are read. The unsuffixed `gate_audit_<SYMBOL>.json`
    is a duplicate of whichever timeframe ran last, and reading both would
    index one verdict twice under two names.

    The windows come from the audits themselves, per timeframe, never from
    this invocation's `--is-start`/`--holdout-end`: a rebuilt file must
    describe the bars its verdicts were measured on, not the flags that
    happened to be typed while rebuilding it.
    """
    prefix = GATE_AUDIT_FILE.format(symbol="")[: -len(".json")]
    found: dict[str, list[tuple[dict, Path]]] = {}
    for path in sorted(Path(out_dir).glob(f"{prefix}*_*.json")):
        try:
            blob = read_stage(path, 3, strat_name)
        except Exception as e:                                    # noqa: BLE001
            print(f"  ! skipping {path.name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        tf = str(blob.get("timeframe") or "")
        if not tf or not blob.get("versions"):
            print(f"  ! skipping {path.name}: it records no timeframe or no "
                  f"version", file=sys.stderr)
            continue
        found.setdefault(tf, []).append((blob, path))

    if not found:
        print(f"No per-pair gate audits under {out_dir}. There is nothing to "
              f"rebuild from - the audits ARE the source, and this mode never "
              f"re-runs one.", file=sys.stderr)
        return 1

    # Deleted rather than merged into. This mode reconstructs the whole index
    # from the authoritative files, so a stale row for a pair whose audit has
    # since been removed must not survive the rebuild.
    summary = Path(out_dir) / STAGE3_SUMMARY_FILE
    if summary.exists():
        summary.unlink()

    print(stage_banner(3, strat_name,
                       f"REBUILD · {sum(len(v) for v in found.values())} "
                       f"audit(s) across {len(found)} timeframe(s)"))
    print("  no bars are read and no gate is re-scored; the per-pair audits "
          "are the source\n  and this writes only the index over them.")

    path_out = summary
    for tf in sorted(found):
        results = [audit_to_result(blob, path) for blob, path in found[tf]]
        first = found[tf][0][0]
        args.tf = tf
        args.is_start = (first.get("in_sample") or {}).get("start")
        args.is_end = (first.get("in_sample") or {}).get("end")
        args.holdout_start = (first.get("holdout") or {}).get("start")
        args.holdout_end = (first.get("holdout") or {}).get("end")
        path_out = write_stage3_summary(
            strat_name, out_dir, results, [], [], args, results,
            f"REBUILT from {GATE_AUDIT_FILE.format(symbol='<SYMBOL>_<TF>')}",
            source_path=resolve_strategy(args.strat))
        print(f"  {tf:<5}{len(results)} audit(s) indexed")

    print(certification_leaderboard(
        [r for tf in sorted(found) for r in
         (audit_to_result(b, p) for b, p in found[tf])]))
    print(f"\n  summary    → {path_out}")
    print(next_step([
        "Post the Stage 3 certification card, now covering every timeframe:",
        "",
        f"  python3 backtest/discord_reporter.py --stage 3 "
        f"--strat {strat_name}",
    ]))
    return 0


def resolve_targets(strat_name: str, args: argparse.Namespace,
                    out_dir: Path, info: dict,
                    tf: str) -> tuple[list[dict], list[dict], str, dict | None]:
    """
    What this run certifies: `(targets, skipped, source, stage2_summary)`.

    Charter clause 1. With no `--symbols` the targets are the EXACT (symbol,
    timeframe) pairs `stage2_summary.json` records, filtered to `tf`. Not a
    glob of `best_params_*.json`, which was the old default and which certifies
    whatever files happen to be in the directory: a winner from a superseded
    sweep sits there indistinguishable from a current one, and Stage 3 would
    stamp a verdict on it with nothing raising.

    `--symbols` is an explicit override and names contracts directly, at `tf`.
    A pair it names that Stage 2 did not optimise is FLAGGED (`in_stage1`,
    `stage2_status`) rather than refused - an operator certifying something by
    hand meant to, and the flag is what stops the result reading as a screened
    one downstream.

    Configurations Stage 2 recorded as ERROR are returned as `skipped`, not
    dropped. Stage 2 prunes nothing, so its matrix carries them with
    `params: "NOT OPTIMIZED"`; a Stage 3 input silently shorter than the Stage
    2 output is how "the sweep never ran" becomes "this was certified and
    failed".
    """
    summary = load_stage2_summary(strat_name, out_dir, args.stage2_summary)

    if args.symbols:
        wanted = parse_symbols(args.symbols, info.get("symbols"))
        by_symbol = {t["symbol"]: t for t in stage2_targets(summary, tf)}
        targets = [dict(by_symbol.get(sym) or {
            "symbol": sym, "timeframe": tf, "stage2_status": "NOT IN STAGE 2",
            "certifiable": True, "quadrant": None, "optimal_regime": None,
            "stage1_version": None, "in_stage1": False, "stage2_error": "",
        }) for sym in wanted]
        for t in targets:
            t["timeframe"] = tf
        return targets, [], "--symbols (explicit)", summary

    if summary is not None:
        rows = stage2_targets(summary, tf)
        if rows:
            targets = [r for r in rows if r["certifiable"]]
            skipped = [{**r, "error": (r["stage2_error"]
                                       or stage2_skip_reason(
                                           r["stage2_status"]))}
                       for r in rows if not r["certifiable"]]
            return targets, skipped, f"{STAGE2_SUMMARY_FILE} (exact pairs)", \
                summary

    # No summary, or none of its pairs are at this timeframe. Fall back to the
    # per-contract files, which is what a single-contract Stage 2 driven by
    # hand leaves behind. The regime scope may be on them even when no matrix
    # was written, so this is a weaker default rather than a broken one.
    symbols = discover_symbols(out_dir, tf)
    targets = [{"symbol": sym, "timeframe": tf,
                "stage2_status": "OPTIMIZED", "certifiable": True,
                "quadrant": None, "optimal_regime": None,
                "stage1_version": None, "in_stage1": False,
                "stage2_error": ""} for sym in symbols]
    return targets, [], "best_params_*.json (no stage 2 summary found)", summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path = resolve_strategy(args.strat)
    strat_name = path.parent.name if path.stem == "strat" else path.stem
    out_dir = pipeline_dir(strat_name, args.out_dir, create=True)

    # Before the window check and before a strategy is loaded: this mode reads
    # audits, not bars, and the windows it would be checking are the ones
    # already recorded on those audits.
    if getattr(args, "rebuild_summary", False):
        return rebuild_stage3_summary(strat_name, out_dir, args)

    try:
        check_windows(args.is_start, args.is_end, args.holdout_start,
                      args.holdout_end)
    except WindowOverlapError as e:
        print(f"\nWindowOverlapError: {e}", file=sys.stderr)
        return 2

    # ONE timeframe. Stage 3 certifies a specific (parameters, timeframe) pair
    # against a specific holdout; sweeping timeframes here would produce
    # several audits per contract under one filename.
    tfs = [t.strip() for t in str(args.tf or "").split(",") if t.strip()]
    if len(tfs) > 1:
        print(f"--tf takes ONE timeframe here, got {args.tf!r}. A gate audit "
              f"certifies one\n(parameters, timeframe) pair against one "
              f"holdout, and gate_audit_<SYMBOL>.json\nholds one verdict. Run "
              f"this once per timeframe you want certified.", file=sys.stderr)
        return 1

    try:
        _fn, info = load_strategy(path, dict(parse_param(p) for p in args.param))
        cfg_kwargs = filter_config_kwargs(args)
    except Exception as e:                                        # noqa: BLE001
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    tf = (tfs[0] if tfs else None) or info.get("timeframe") or "15m"
    args.tf = tf
    grid = info.get("param_grid") or {}

    try:
        targets, skipped, target_source, summary = resolve_targets(
            strat_name, args, out_dir, info, tf)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if not targets and not skipped:
        print(f"Nothing to certify at {tf}. Run stage 2 (backtest/scan.py) "
              f"first,\nor name contracts with --symbols.", file=sys.stderr)
        return 1

    print(stage_banner(3, strat_name,
                       f"{len(targets)} configuration(s) · {tf}"))
    print(f"  targets    : {target_source}")
    print(f"  in-sample  : {args.is_start} → {args.is_end}   (evidence only)")
    print(f"  holdout    : {args.holdout_start} → "
          f"{args.holdout_end or 'present'}   (the verdict; untouched "
          f"until now)")
    print(f"  Gate R     : PF >= {args.regime_min_pf:.2f} over "
          f"{args.regime_min_trades}+ trades INSIDE the designated quadrant")
    print("               Gates 1-3 are computed and reported as EVIDENCE. "
          "They cannot\n               fail a certification: they score the "
          "blended sample across every\n               market state, which a "
          "regime-gated strategy does not trade.")
    print(f"  pruning    : none on an aggregate metric; no prop-firm rule "
          f"applied\n               ({', '.join(PROP_FIRM_FIELDS)} are "
          f"CrossTrade NAM's, against a live balance)")
    print(f"  walk-fwd   : {args.wfo_train}y train / {args.wfo_test}y test, "
          f"{'re-optimized per fold' if args.wfo_grid else 'FIXED parameters'}")
    if not args.wfo_grid:
        print("               fixed parameters means the ratio compares two "
              "time periods,\n               not fitted-versus-unseen. "
              "Recorded as wfo_optimized: false.")
    # Per pair, from Stage 1's handoff - not a single global answer any more.
    b_pairs = [f"{t['symbol']}·{t['timeframe']}" for t in targets
               if resolve_version_b(t, args)[0]]
    if args.ml:
        print(f"  Version B  : certified for ALL {len(targets)} pair(s) (--ml)")
    elif getattr(args, "no_stage1_ml", False):
        print("  Version B  : NOT RUN (--no-stage1-ml overrides the handoff)")
    elif b_pairs:
        print(f"  Version B  : certified for {len(b_pairs)} of {len(targets)} "
              f"pair(s) that cleared Stage 1 on it: {', '.join(b_pairs)}")
    else:
        print("  Version B  : NOT RUN - no target pair records "
              "stage1_version=B, and --ml is off")
    print(f"  incubator  : "
          + (f"certified versions staged into {args.incubator} "
             f"(never committed)" if args.promote
             else "NOT staged (--no-promote)"))
    unscreened = [f"{t['symbol']}·{t['timeframe']}" for t in targets
                  if not t.get("in_stage1", True)]
    if unscreened:
        print(f"  ! {len(unscreened)} configuration(s) were not promoted by "
              f"stage 1: {', '.join(unscreened)}\n    Certified anyway and "
              f"flagged in_stage1=false — a verdict on an unscreened pair "
              f"must\n    not reach an incubator looking like a screened one.")
    for sk in skipped:
        print(f"  ! SKIP {sk['symbol']}·{sk['timeframe']}: {sk['error']}")

    results, errors = [], []
    for i, target in enumerate(targets, 1):
        sym = target["symbol"]
        print(f"\n[{i}/{len(targets)}] {sym}")
        try:
            results.append(certify_symbol(sym, path, tf, args, cfg_kwargs,
                                          out_dir, strat_name, grid,
                                          target=target))
        except Exception as e:                                    # noqa: BLE001
            errors.append({"symbol": sym, "timeframe": tf,
                           "quadrant": target.get("quadrant"),
                           "optimal_regime": target.get("optimal_regime"),
                           "in_stage1": target.get("in_stage1", True),
                           "error": f"{type(e).__name__}: {e}"})
            print(f"\n[!] {sym}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    W = 78
    print("\n" + "=" * W)
    print("STAGE 3 RESULT · certified verdicts")
    print("=" * W)
    print(certification_leaderboard(results))
    passing = [(r["symbol"], ver) for r in results
               for ver, ok in r["passed"].items() if ok]
    for e in errors:
        print(f"  ERROR {e['symbol']:<6}{e['error']}")
    for r in results:
        print(f"  {r['symbol']:<6}{r['timeframe']:<5}→ {r['path'].name}"
              + (f"   (certified with exclude_days={r['exclude_days']})"
                 if r.get("exclude_days") else ""))

    summary_path = write_stage3_summary(strat_name, out_dir, results, errors,
                                        skipped, args, targets, target_source,
                                        source_path=path)
    print(f"\n  summary    → {summary_path}")

    staged = [(r["symbol"], ver, blk["dir"])
              for r in results
              for ver, blk in (r.get("incubator") or {}).items()
              if blk.get("promoted")]
    if staged:
        print(f"\n  {len(staged)} version(s) staged into the incubator, "
              f"sealed and NOT committed:")
        for sym, ver, dest in staged:
            print(f"    {sym}/{ver}  →  {dest}")

    if not passing:
        print("\n  Nothing was certified. Gate R FAIL means the edge did not "
              "hold in its own\n  quadrant out of sample; NOT EVALUATED means "
              "no quadrant was designated.\n  Neither is a pass, and Stage 5 "
              "will refuse both without --force.")
    else:
        print(f"\n  {len(passing)} certification(s) passed: "
              + ", ".join(f"{s}/{v}" for s, v in passing))

    sym, ver = passing[0] if passing else (targets[0]["symbol"], "A")
    print(next_step([
        "Post the Stage 3 certification card to Discord:",
        "",
        f"  python3 backtest/discord_reporter.py --stage 3 "
        f"--strat {strat_name}",
        "",
        "Stage 4 — the full lifecycle run, for the tear sheets and the cost drag:",
        "",
        f"  python3 backtest/verify_full.py --strat {args.strat} "
        f"--symbols {sym} --tf {tf} \\",
        "      --start 2010-01-01 --end 2026-01-01",
        "",
        "Stage 5 — commit the promotion, once a human has read the evidence:",
        "",
        f"  python3 backtest/promote.py --strat {strat_name} --version {ver} \\",
        f"      --source {path} \\",
        f"      --audit-file "
        f"{out_dir / GATE_AUDIT_FILE.format(symbol=f'{sym}_{tf}')}",
    ]))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
