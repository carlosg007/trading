"""
backtest/pipeline.py - the contract BETWEEN the five stages.

Location: ~/src/trading/backtest/pipeline.py

Not a stage. This module holds the handoff: where a stage writes its output,
what the next one reads, and the header every stage prints so a log says which
one produced it. It exists because the alternative is each stage hard-coding
the same paths, and a Stage 3 that looks for `best_params_NQ.json` one
directory away from where Stage 2 wrote it fails by finding nothing - which is
indistinguishable from a sweep that produced no winner.

    Stage 1  baseline.py     surviving_assets.json      which contracts carry it
    Stage 2  scan.py         best_params_<SYM>.json     what parameters won
    Stage 3  audit_gates.py  gate_audit_<SYM>.json      PASS/FAIL, on the record
    Stage 4  verify_full.py  verify_<SYM>.json + sheets the full lifecycle
    Stage 5  promote.py      approved_incubator/<strat> the decision

Everything lands under `<BT_ARTIFACTS>/pipeline/<strategy>/`, one directory per
strategy rather than per run, because these files are a CHAIN: Stage 3 has to
find Stage 2's winner without being told a timestamp. That means a re-run
overwrites - deliberately, and only for the small JSON handoffs. Stage 4's tear
sheets keep the repo's usual timestamped-directory rule, since those are the
evidence a promotion cites and evidence is never overwritten.

Stage 1 also hands Stage 2 a DECISION, not only a list: the Drop Unprofitable
Days contract. Each surviving pair carries its own `dropped_days` and
`exclude_days` - every weekday whose profit factor was below 1.00 - and
`stage1_exclude_days` below is the one place that mapping is read. Keyed per
`(symbol, timeframe)` rather than globally, because which weekdays lose is a
fact about a contract at a timeframe and a single list applied to every survivor
would prune a session that is profitable on one of them.

Every handoff file records the stage that wrote it, the strategy, the date
window and a UTC timestamp. `read_stage` checks the strategy name matches what
the reader expects and raises when it does not: two strategies' pipelines run
in sequence, and picking up the wrong `best_params_NQ.json` would certify one
strategy's gates against another's parameters with nothing raising.
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


import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

#: Stage 4.5's stage id, as an INTEGER, because that is what `write_stage`
#: stamps and `read_stage` compares. `4.5` cannot be used: `write_stage` writes
#: `int(stage)`, which truncates it to 4, and every Stage 4.5 handoff would
#: then be indistinguishable from a Stage 4 one to `read_stage` - a Stage 4.5
#: summary would satisfy a Stage 4 check and vice versa, silently. The DISPLAY
#: label is separate (`STAGE_LABELS`), so the banner still reads "STAGE 4.5/5".
STAGE45 = 45

STAGE_NAMES = {
    1: "BASELINE · which contracts carry the edge",
    2: "SCAN · in-sample parameter selection",
    3: "GATE AUDIT · certification",
    4: "FULL VERIFICATION · whole lifecycle",
    STAGE45: "DAY-OF-WEEK GATE · the worst session, named",
    5: "PROMOTION · into the incubator",
}

#: How a stage id is SPELLED in a banner. Only Stage 4.5 differs from its id,
#: and it differs because the id has to be an integer (see `STAGE45`) while
#: the thing a human reads is "4.5". Keyed rather than computed so a future
#: half-stage lands here rather than in a conditional inside `stage_banner`.
STAGE_LABELS = {STAGE45: "4.5"}

# The charter's IN-SAMPLE window, and the first day of the holdout. Defined
# here rather than in a stage because more than one stage has to agree about
# it: Stage 1 screens on it, Stage 2 optimises on it, and Stage 3 measures
# retention against the years AFTER it. Two stages holding their own copies of
# these dates would drift by one edit, and the failure is silent - a Stage 2
# sweep that ran a day into 2023 selects parameters on bars Gate 3 then scores
# as unseen, and every number downstream still looks well-formed.
CHARTER_IS_START = "2013-01-01"
CHARTER_IS_END = "2022-12-31"
# Everything from here on is the Stage 3 holdout. Stages 1 and 2 must not read
# a bar of it - not to be conservative, but because a holdout that has been
# optimised over is not a holdout, and nothing downstream can detect that it
# was spent.
HOLDOUT_START = "2023-01-01"

SURVIVORS_FILE = "surviving_assets.json"
#: The same screen as a spreadsheet: every configuration evaluated, promoted
#: and dropped alike, one row each. Named beside the JSON handoff rather than
#: derived from it at read time so the Discord card can point somebody at a
#: file that exists, and so a reader who wants the leaderboard does not have to
#: parse a 128 KB nested document to get thirteen columns.
SURVIVORS_CSV = "stage1_survivors.csv"

# An account cannot lose more than it holds. The engine's equity is
# `initial_capital + cumsum(net P&L)` with NO ruin barrier, so a strategy whose
# cumulative losses exceed the starting capital produces a NEGATIVE equity, and
# `equity / peak - 1` then reports a drawdown past -100% - which is not a
# deeper loss but arithmetic that has stopped describing an account.
#
# It lives HERE, in the module every stage already imports, because Stage 2 and
# Stage 3 both have to draw the line in the same place and they cannot import
# each other: `audit_gates` imports `scan.expand_grid`, so the dependency runs
# Stage 3 -> Stage 2 and a constant owned by Stage 3 is unreachable from Stage
# 2. Two spellings of "ruin" would let a parameter set Stage 2 called
# survivable be the same one Stage 3 calls ruined, with the sweep advancing it
# and the gate killing it, and nothing naming the disagreement.
RUIN_MIN_DRAWDOWN_PCT = -100.0
# Stage 1's human-readable half. The JSON above is what Stage 2 reads; this is
# what a person reads, and it is written on EVERY run - including one where
# nothing survived, which is the run whose detail matters most. Markdown rather
# than a console dump because the console is now a progress line per
# configuration: the evidence has to land somewhere, and somewhere is a file.
BASELINE_REPORT_FILE = "stage1_baseline_report.md"
BEST_PARAMS_FILE = "best_params_{symbol}.json"
# Stage 2's summary matrix: one row per (symbol, timeframe) optimised, in two
# forms. The JSON is the handoff `discord_reporter.py --stage 2` reads and is
# written through `write_stage`, so it carries the stage and strategy stamp
# that lets `read_stage` refuse the wrong one. The CSV is the same rows for a
# human and for a spreadsheet - it is NOT read back by any stage, because a
# CSV round trip loses the types a parameter set is made of.
STAGE2_SUMMARY_FILE = "stage2_summary.json"
#: Version B keeps an entry when the classifier's P(win) is at or above this.
#:
#: ONE constant, imported by all four stages, because a threshold that differs
#: between them is invisible and wrong in the worst direction. Stage 3 certifies
#: a filter at one value and promote.py bakes another into the module it
#: deploys; the promoted strategy then reproduces a Version B that was never
#: certified, and every log line reads correctly. Each stage still takes
#: `--ml-threshold` to override it, and the value that was APPLIED is written
#: into that stage's handoff so a run can be read back without guessing.
#:
#: 0.48 rather than 0.50 since 2026-08-29. A coin-flip bar filtered out roughly
#: half of Version B's entries, which shrank the holdout sample in the
#: designated quadrant toward Gate R's 30-trade floor - so B failed on COUNT
#: rather than on edge, whatever its profit factor did. Lowering the bar keeps
#: more entries; it does not make the filter better, and Gate R is still the
#: only evidence either way.
ML_THRESHOLD_DEFAULT = 0.48

STAGE2_MATRIX_FILE = "stage2_summary_matrix.csv"
GATE_AUDIT_FILE = "gate_audit_{symbol}.json"

# Stage 3's own handoff, beside the per-contract gate audits. One file for the
# whole certification run, written through `write_stage` for the same reason
# Stage 2's summary is: `discord_reporter.py --stage 3` reads it, and a
# certification card posted off another strategy's audit is exactly the
# artifact nobody cross-checks. The per-contract `gate_audit_<SYMBOL>.json`
# stays the AUTHORITATIVE verdict a promotion rests on - this is the index
# over them, not a replacement for one.
STAGE3_SUMMARY_FILE = "stage3_audit_summary.json"

VERIFY_FILE = "verify_{symbol}.json"

# Stage 4.5's two handoffs. The per-pair file is the AUTHORITATIVE verdict -
# it carries the full weekday table, the counterfactual and the rule that
# produced the block - and the summary is the index over them, the same
# division Stage 3 draws between `gate_audit_<SYM>_<TF>.json` and
# `stage3_audit_summary.json`.
#
# The per-pair name carries the TIMEFRAME and there is no unsuffixed form, on
# purpose. A blocked weekday is a fact about one (symbol, timeframe) pair -
# `t3_braid_scalp_20260823` certified NQ at 15m, 30m and 1h with a different
# quadrant at each - and an unsuffixed file would hold whichever timeframe ran
# last while Stage 5 promoted all three against it. The failure is invisible:
# every meta.json would carry a plausible weekday, two of them wrong.
DOW_GATE_FILE = "dow_gate_{symbol}_{tf}.json"
STAGE45_SUMMARY_FILE = "stage45_dow_summary.json"


# --------------------------------------------------------------------------
# The strategy id: one certified (strategy, symbol, timeframe), named
# --------------------------------------------------------------------------
# A promotion is ONE contract at ONE timeframe, and a campaign certifies
# several - `t3_braid_scalp_20260823` cleared Gate R on NQ at 15m, 30m AND 1h,
# each with its own winning parameter plateau and its own certified quadrant
# (Q1 at 15m, Q2 at the other two). Promoted into `approved_incubator/<strat>/`
# they shared one directory, so each promotion overwrote the last: one
# `strat.py`, one `meta.json`, describing whichever ran last. The routing table
# then granted one permission, and `realtime/live_dispatcher.py` would have
# traded 15m's parameters for all three pairs - every log line reading
# correctly, because the module IS the same module and only the bound
# parameters and the permitted quadrant differ.
#
# So the unit that gets promoted, registered and traded is the PAIR, and it is
# named: `<strategy>_<SYMBOL>_<TIMEFRAME>`. That name is the directory under
# `approved_incubator/`, the id in a portfolio's `active_strategies`, and the
# `name` in the promoted `meta.json` - one spelling in three places, because
# the live dispatcher builds the second from the first and the Stage 5 card
# reads the third.
#
# It is spelled HERE, in the module that already owns what one stage hands the
# next, rather than in `backtest/promote.py` - `backtest/discord_reporter.py`
# has to split an id back apart to check that a snapshot belongs to the
# promotion citing it, and it deliberately imports no module that pulls in the
# engine.

# The timeframe tokens an id may end with. `mdlib.lake` owns the real
# vocabulary (1m and 1d are stored, the rest derived), and this is a
# deliberately separate, frozen copy: `split_strategy_id` runs inside a
# Discord card and a routing-table read, and neither may import the lake
# reader to parse a directory name. A token missing from here does not break a
# promotion - it makes the id unsplittable, so the base strategy is reported
# as the whole id and a snapshot check falls back to demanding an exact match.
TIMEFRAME_TOKENS = ("1m", "2m", "3m", "5m", "10m", "15m", "30m",
                    "1h", "2h", "3h", "4h", "6h", "8h", "12h",
                    "1d", "1w")


#: The version suffixes a promotion id may carry. Spelled once so the builder
#: and the parser cannot disagree about what a version segment looks like.
VERSION_TOKENS = {"VA": "A", "VB": "B"}


def strategy_id(strategy: str, symbol: str | None = None,
                timeframe: str | None = None,
                version: str | None = None) -> str:
    """
    The id for one certified pair: `<strategy>_<SYMBOL>_<TF>`.

    With no symbol or no timeframe the BARE strategy name is returned
    unchanged, and that is not a fallback to be tidied away - it is the id the
    documented `bt-run` workflow promotes under, where a dual-version run is
    not scoped to a certified pair and there is nothing to name. Half an id
    (`<strategy>_NQ`) is never produced: it would split back to a timeframe of
    None and read as a pair whose timeframe nobody recorded.

    The symbol is upper-cased and the timeframe lower-cased, so `nq`/`NQ` and
    `1H`/`1h` cannot produce two directories for one pair.
    """
    strategy = str(strategy or "").strip()
    sym = str(symbol or "").strip().upper()
    tf = str(timeframe or "").strip().lower()
    if not strategy:
        raise ValueError("a strategy id needs a strategy name")
    if not sym or not tf:
        return strategy
    sid = f"{strategy}_{sym}_{tf}"
    # The version segment exists because Version A and Version B of the SAME
    # pair are two different strategies that used to resolve to one directory.
    # Both can certify - NQ 1h certified on both on 2026-08-29, A at OOS PF
    # 1.25 and B at 1.22 - and promote.py then wrote A, found the path taken
    # when it reached B, and exited 1. The Version B package was lost with no
    # gate having refused it.
    #
    # Appended ONLY when a version is given, so every id already on disk and
    # in config/portfolios.json keeps its exact spelling. This is forward-only:
    # nothing is renamed.
    ver = str(version or "").strip().upper().removeprefix("V")
    if ver:
        if f"V{ver}" not in VERSION_TOKENS:
            raise ValueError(f"unknown version {version!r}; expected A or B")
        sid = f"{sid}_V{ver}"
    return sid


def split_strategy_id(sid: str) -> tuple[str, str | None, str | None]:
    """
    `<strategy>_<SYMBOL>_<TF>` back into its three parts.

    Returns `(strategy, symbol, timeframe)`; the last two are None for a bare
    id. The split is anchored on the TIMEFRAME, not on the underscore count:
    strategy names in this repository carry underscores and a date suffix
    (`t3_braid_scalp_20260823`), so counting from the left splits them apart
    and counting a fixed number from the right would turn
    `ma_anchoring_spread_20260820` into a symbol of `spread` at a timeframe of
    `20260820`.

    An id is only split when its LAST segment is a known timeframe token and
    the segment before it is non-empty. Anything else is a strategy whose name
    happens to contain underscores, and is returned whole - which is the safe
    direction: the callers use the base name to decide whether an artifact
    belongs to a promotion, and reporting the whole id there demands an exact
    match instead of accepting a looser one.
    """
    text = str(sid or "").strip()
    if "_" not in text:
        return text, None, None
    # Strip a trailing version segment BEFORE the timeframe anchor. Without
    # this the parse fails outright on a version-qualified id and returns
    # (whole_id, None, None) - and `LiveDispatcher._resolve_timeframe` falls
    # back to exactly this call when a meta.json declares no timeframe, so a
    # promoted Version B would have been admitted with its timeframe unknown
    # rather than checked. The version itself is NOT returned here: the tuple
    # is three-wide and every caller unpacks it that way. Use
    # `version_of_strategy_id` for the letter.
    head0, _, last = text.rpartition("_")
    if last.upper() in VERSION_TOKENS and "_" in head0:
        text = head0
    head, _, tf = text.rpartition("_")
    if tf.lower() not in TIMEFRAME_TOKENS:
        return text, None, None
    strategy, _, symbol = head.rpartition("_")
    if not strategy or not symbol:
        return text, None, None
    return strategy, symbol, tf.lower()


def version_of_strategy_id(sid: str) -> str | None:
    """
    "A", "B", or None for an id that carries no version segment.

    None is NOT Version A. Every id promoted before 2026-08-29 is unqualified,
    and reading those as A would assert a fact about them that the name never
    recorded - the version is in their meta.json and that is where it should be
    read from.
    """
    text = str(sid or "").strip()
    head, _, last = text.rpartition("_")
    if last.upper() in VERSION_TOKENS and "_" in head:
        return VERSION_TOKENS[last.upper()]
    return None


def base_strategy(sid: str) -> str:
    """
    The strategy a promotion id belongs to - `t3_braid_scalp_20260823` for
    `t3_braid_scalp_20260823_NQ_1h`, and the id itself for a bare one.

    This is what an artifact-ownership check compares against. A
    `dual_metrics_NQ.json` records the MODULE's name and a promotion is
    registered under the pair's id, so the two differ by construction; without
    this the Stage 5 card would refuse the very snapshot the promotion cites.
    """
    return split_strategy_id(sid)[0]


def artifacts_root() -> Path:
    """
    Read at call time, never at import.

    `backtest/run.py` binds `$BT_ARTIFACTS` at import, which is correct for a
    long-lived batch and wrong for a test process that sets the variable after
    importing the module it is testing. These paths are read once per stage, so
    the lookup costs nothing.
    """
    return Path(os.environ.get("BT_ARTIFACTS", "/mnt/backtest/artifacts"))


def pipeline_dir(strategy: str, out_dir: str | Path | None = None,
                 create: bool = False) -> Path:
    """`<artifacts>/pipeline/<strategy>/`, or an explicit override."""
    d = Path(out_dir) if out_dir else artifacts_root() / "pipeline" / strategy
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def stage_banner(stage: int, strategy: str, detail: str = "") -> str:
    """The header every stage prints. One line says which stage a log is."""
    W = 78
    label = STAGE_LABELS.get(int(stage), str(int(stage)))
    return ("\n" + "=" * W
            + f"\nSTAGE {label}/5 · {STAGE_NAMES[stage]}"
            + f"\n{strategy}" + (f"  ·  {detail}" if detail else "")
            + "\n" + "=" * W)


def write_stage(path: Path, stage: int, strategy: str,
                payload: dict[str, Any]) -> Path:
    """
    Write a handoff file with its provenance attached.

    Atomic: a temp file then `os.replace`, the same rule `status.py` follows.
    A stage killed mid-write would otherwise leave a truncated JSON that the
    next stage reads as a parse error at best and as a short symbol list at
    worst.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "stage": int(stage),
        "stage_name": STAGE_NAMES[int(stage)],
        "strategy": strategy,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_stage(path: Path, expect_stage: int | None = None,
               expect_strategy: str | None = None) -> dict[str, Any]:
    """
    Read a handoff file and refuse the wrong one.

    A missing file raises `FileNotFoundError` naming the stage that should have
    written it, because "run Stage 2 first" is the actual answer and a bare
    "no such file" sends somebody looking for a bug in Stage 3.
    """
    path = Path(path)
    if not path.exists():
        want = f"stage {expect_stage}" if expect_stage else "an earlier stage"
        raise FileNotFoundError(
            f"{path} does not exist. It is written by {want} — run that first.")
    blob = json.loads(path.read_text(encoding="utf-8"))

    if expect_stage is not None and int(blob.get("stage", -1)) != int(expect_stage):
        raise ValueError(
            f"{path} was written by stage {blob.get('stage')}, not stage "
            f"{expect_stage}.")
    if expect_strategy is not None and blob.get("strategy") != expect_strategy:
        raise ValueError(
            f"{path} belongs to strategy {blob.get('strategy')!r}, not "
            f"{expect_strategy!r}. Certifying one strategy's gates against "
            f"another's parameters is a mistake nothing downstream could "
            f"detect.")
    return blob


def stage1_exclude_days(blob: dict[str, Any] | None
                        ) -> dict[tuple[str, str], tuple[int, ...]]:
    """
    Stage 1's Drop Unprofitable Days decision, keyed by `(symbol, timeframe)`.

    The handoff carries one `exclude_days` list PER SURVIVING PAIR - zero, one
    or five days - because which weekdays lose is a fact about a contract at a
    timeframe. Flattening them into one list and applying it everywhere would
    prune a session that is profitable on NQ in order to fix one that is not on
    GC, and nothing downstream could tell that had happened.

    Pairs with an empty list are omitted from the mapping rather than mapped to
    `()`. `()` and "no entry" mean the same thing to every caller here, and the
    absent key keeps `if key in mapping` an honest test of "did Stage 1 exclude
    anything for this pair".

    A blob written before the contract existed - or by a run with
    `--no-drop-losing-days` - simply yields an empty mapping. That is the correct
    reading: no exclusion was decided, so Stage 2 sweeps the whole week.
    """
    out: dict[tuple[str, str], tuple[int, ...]] = {}
    for pair in (blob or {}).get("surviving_pairs") or []:
        if not isinstance(pair, dict):
            continue
        sym, tf = pair.get("symbol"), pair.get("tf")
        days = tuple(sorted({int(d) for d in (pair.get("exclude_days") or [])}))
        if sym and tf and days:
            out[(str(sym), str(tf))] = days
    return out


def stage45_blocked_days(blob: dict[str, Any] | None
                         ) -> dict[tuple[str, str], tuple[int, ...]]:
    """
    Stage 4.5's day-of-week decision, keyed by `(symbol, timeframe)`.

    The twin of `stage1_exclude_days`, and keyed the same way for the same
    reason: which weekday loses is a fact about a contract AT A TIMEFRAME.
    `t3_braid_scalp_20260823` cleared Gate R on NQ at 15m, 30m and 1h with a
    different quadrant at each, and one blocked weekday flattened across all
    three would stand the strategy down on a session two of them trade
    profitably - with every log line reading correctly.

    A pair whose verdict blocked NOTHING is ABSENT from the mapping rather
    than mapped to `()`. `()` and "no entry" mean the same thing to every
    caller, and the absent key keeps `if key in mapping` an honest test of
    "did Stage 4.5 block anything for this pair".

    That absence is deliberately NOT how "the stage did not run" is expressed.
    A handoff written before this contract existed - or no handoff at all -
    yields `{}`, which is the same shape as a run where nothing was blocked;
    the difference is whether the FILE exists, and that is the caller's to
    check. `backtest/promote.py` does, and writes `"NOT EVALUATED"` into
    `meta.json` rather than an empty list, because an empty list is a
    statement about the week and "nobody looked" is not.

    Each row carries ONE weekday or none. The stage names the worst session,
    not a set: a rule that could block two would be selecting a subset of the
    calendar in-sample on the bars it is scored on, which is the curve fit the
    firewall replaced this contract with in the first place.
    """
    out: dict[tuple[str, str], tuple[int, ...]] = {}
    for row in (blob or {}).get("results") or []:
        if not isinstance(row, dict):
            continue
        sym, tf = row.get("symbol"), row.get("timeframe") or row.get("tf")
        day = row.get("blocked_weekday")
        if not sym or not tf or day is None:
            continue
        d = int(day)
        if not 0 <= d <= 6:
            raise ValueError(
                f"stage 4.5 recorded blocked_weekday={day!r} for {sym} {tf}; "
                f"weekdays are 0-6 (Mon-Sun). Read as anything else this "
                f"stands a strategy down on a day nobody profiled.")
        out[(str(sym), str(tf))] = (d,)
    return out


def stage1_pairs(blob: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Stage 1's SURVIVING configurations, as exact `(symbol, timeframe)` pairs
    with the regime scope each one cleared.

    This is what Stage 2 sweeps. The pairs are exact rather than a cross
    product because the survivors are ragged - NQ may survive at 5m and 15m
    while GC survives only at 15m - and `--symbols`/`--tf` are two independent
    axes, so expressing them as a product sweeps `GC 5m`, a configuration the
    screen just dropped. Fitting parameters to a contract with no baseline edge
    is the definition of the curve fit the screen exists to prevent, and
    nothing downstream records that it happened.

    The regime scope travels WITH the pair - `quadrant`, `optimal_regime`,
    `kill_switch_regimes` - because Stage 2's output is what a live supervisor
    eventually reads, and a parameter set that arrives without the environment
    it was screened in reads as a licence to trade it everywhere. Nothing here
    applies the scope: Stage 2 sweeps the whole window on purpose (masking the
    sweep to a quadrant chosen as the best of four on these same bars stacks a
    second in-sample selection under the first). It is carried, not enforced.

    A pair with no `tf` is skipped rather than defaulted to the blob's
    timeframe: a survivor whose timeframe cannot be read is a handoff bug, and
    guessing it sweeps something nobody screened.
    """
    out: list[dict[str, Any]] = []
    for pair in (blob or {}).get("surviving_pairs") or []:
        if not isinstance(pair, dict):
            continue
        sym, tf = pair.get("symbol"), pair.get("tf") or pair.get("timeframe")
        if not sym or not tf:
            continue
        out.append({
            "symbol": str(sym),
            "tf": str(tf),
            "version": pair.get("version"),
            "quadrant": pair.get("quadrant"),
            "optimal_regime": pair.get("optimal_regime"),
            "regime_pf": pair.get("regime_pf"),
            "regime_trade_count": pair.get("regime_trade_count"),
            "regime_net_pnl": pair.get("regime_net_pnl"),
            "regime_win_rate": pair.get("regime_win_rate"),
            # The alpha score that designated this quadrant, the floor it
            # cleared, the full four-quadrant table and the positive
            # runners-up. Carried for the same reason the quadrant is: Stage
            # 2's output is what a live supervisor eventually reads, and a
            # target quadrant that arrives with no record of what it beat
            # cannot be told apart from one somebody typed.
            "regime_score": pair.get("regime_score"),
            "regime_sample_floor": pair.get("regime_sample_floor"),
            "regime_scores": pair.get("regime_scores") or {},
            "secondary_regimes": pair.get("secondary_regimes") or [],
            "kill_switch_regimes": list(pair.get("kill_switch_regimes") or []),
        })
    return out


def leaderboard(title: str, header: list[str], rows: list[list[str]],
                align: list[str] | None = None,
                empty: str = "nothing to report") -> str:
    """
    The end-of-stage table, rendered the same way by every stage.

    One implementation rather than three, because the point of printing a
    leaderboard at each stage is that an operator reads them as a sequence: a
    Symbol column that is left-aligned in Stage 1 and right-aligned in Stage 2
    makes two tables of the same contracts look like tables of different things.

    Columns are sized to their widest CELL, not to a fixed width, so a long
    parameter set widens its own column instead of being silently clipped -
    a truncated winning parameter set reads as a complete one, and the whole
    value of the row is that it names the parameters exactly.

    `align` is one of `"<"` or `">"` per column, defaulting to left for the
    first and right for the rest, which is the shape every one of these tables
    has: identifiers on the left, numbers on the right. A stage that ends with
    no rows still prints the heading and says so - an absent table reads as a
    stage that did not finish.
    """
    align = align or (["<"] + [">"] * (len(header) - 1))
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(header[i]), *(len(r[i]) for r in cells)) if cells
              else len(header[i]) for i in range(len(header))]

    def _line(vals: list[str]) -> str:
        # Right-stripped: a padded final column leaves trailing spaces on every
        # row, which survive into a log file and show up as a diff against the
        # same table pasted anywhere else.
        return ("  " + "  ".join(f"{v:{align[i]}{widths[i]}}"
                                 for i, v in enumerate(vals))).rstrip()

    rule = "  " + "  ".join("-" * w for w in widths)
    total = max(len(rule), len(title) + 2)
    out = ["", "=" * total, f"  {title}", "=" * total, _line(header), rule]
    out.extend(_line(r) for r in cells)
    if not cells:
        out.append(f"  ({empty})")
    out.append("=" * total)
    return "\n".join(out)


def next_step(lines: list[str]) -> str:
    """
    The block every stage ends with: the exact command for the next one.

    Printed rather than run. The tool boundary in CLAUDE.md is that a human is
    at the console when the evidence appears, and a stage that chained straight
    into the next one would put the four-choice menu in front of nobody.
    """
    W = 78
    out = ["", "-" * W, "NEXT STEP", "-" * W]
    out.extend(f"  {ln}" for ln in lines)
    out.append("-" * W)
    return "\n".join(out)


if __name__ == "__main__":
    # Not a stage, and running it as one has to FAIL rather than exit 0.
    #
    # This module is the contract between the stages: it declares paths, the
    # charter dates and the handoff readers, and it runs nothing. Without this
    # block `python3 backtest/pipeline.py --strat X --symbols ALL --tf 15m`
    # imports it, binds those constants, ignores every argument and exits 0 -
    # producing an empty log, no artifacts and no Discord card, which is
    # indistinguishable at the console from a pipeline that ran and found
    # nothing. There is no argparse here on purpose: an unknown flag must not
    # be the thing that raises, because the command is wrong even when every
    # flag on it is spelled correctly.
    import sys

    print(
        "\n".join(
            [
                "",
                "=" * 78,
                "backtest/pipeline.py IS NOT A STAGE - it is the contract between them.",
                "=" * 78,
                "",
                "It declares where each stage writes and what the next one reads. It",
                "spawns nothing, so run as a script it would exit 0 with an empty log.",
                "",
                "Nothing chains automatically: each stage prints the next one's command",
                "and stops, so a human reads the evidence in between. Run them by hand:",
                "",
                "  1  python3 -u backtest/baseline.py    --strat X --symbols ALL --tf 15m",
                "  2  python3 -u backtest/scan.py        --strat X",
                "  3  python3 -u backtest/audit_gates.py --strat X --tf 15m",
                "  4  python3 -u backtest/verify_full.py --strat X --tf 15m",
                "  5  python3 -u backtest/promote.py     --strat X --version A \\",
                "         --source <module.py> --audit-file <gate_audit_SYMBOL_TF.json>",
                "",
                "Discord cards are posted per stage, from that stage's handoff:",
                "",
                "  python3 -u backtest/discord_reporter.py --stage {1,2,3} --strat X",
                "  python3 -u backtest/discord_reporter.py --stage 3 --strat X --dry-run",
                "",
                f"Handoffs live under {artifacts_root() / 'pipeline'}/<strategy>/",
                "=" * 78,
                "",
            ]
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)
