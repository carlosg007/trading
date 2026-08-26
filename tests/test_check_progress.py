#!/usr/bin/env python3
"""
tests/test_check_progress.py - the pipeline status reader, on synthetic trees.

Location: ~/src/trading/tests/test_check_progress.py

    .venv/bin/python3 -m pytest tests/test_check_progress.py -q
    .venv/bin/python3 tests/test_check_progress.py            # as a script

ASSERT-BASED, and pytest-shaped on purpose. Every case fails through `assert`,
so `pytest tests/` and running this file directly report the same thing - the
convention `tests/conftest.py` documents for suites that are not written around
the `check(name, ok)` collector.

Every helper is named `_...` rather than `test_...`. `tests/test_regime_profiler
.py` was bitten by the other spelling: pytest collects any module-level `test_*`
it can call, INCLUDING one whose only argument is defaulted, and a helper
collected that way ran its sections without their `$BT_ARTIFACTS` redirect and
wrote real JSON onto the NFS mount. Nothing here may touch /mnt/backtest, so
nothing here is named so it could be called by accident.

WHAT IS ACTUALLY BEING TESTED
-----------------------------
`check_progress.py` INFERS - from the process table and from file names - and
the inferences worth pinning down are the ones a human would otherwise have to
re-derive by hand:

  * the Stage 2 work plan reconstructed from `surviving_assets.json` in the
    exact order `scan.py` walks it, including the `[3/5] RB` position, which is
    the number the whole tool exists to print;
  * the freshness split, which is what stops a re-run's card from presenting
    the previous campaign's Stage 3 as this one's;
  * that a truncated handoff, a missing directory and an unreadable process
    table each produce a readout rather than a traceback.

No test here runs a stage, reads a bar, or writes outside a temporary
directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# BT_ARTIFACTS is redirected BEFORE the module is imported, and every case
# redirects it again around its own tree. `pipeline.artifacts_root()` reads the
# variable at call time, so this is belt and braces - but the belt is what
# keeps a bug in the module under test from writing to the mount.
os.environ.setdefault("BT_ARTIFACTS", tempfile.mkdtemp(prefix="bt-check-guard-"))

from backtest import check_progress as cp  # noqa: E402


# ==========================================================================
# fixtures - a strategy directory, built by hand
# ==========================================================================

def _tree(root: Path, strategy: str) -> Path:
    d = root / "pipeline" / strategy
    d.mkdir(parents=True, exist_ok=True)
    return d


def _survivors(d: Path, strategy: str, pairs: list[tuple[str, str]],
               screened: list[str] | None = None) -> Path:
    """
    A Stage 1 handoff with `surviving_pairs` in the given ORDER.

    Order is the payload of this fixture, not an incidental detail: Stage 2
    walks the pairs as they appear, so a fixture that sorted them would test a
    plan the sweep never follows.
    """
    blob = {
        "stage": 1,
        "stage_name": "BASELINE",
        "strategy": strategy,
        "generated_utc": "2026-08-26T00:00:00+00:00",
        "timeframes": sorted({tf for _, tf in pairs}),
        "evaluated": len(screened or []) or len(pairs),
        "surviving_pairs": [{"symbol": s, "tf": t, "version": "A",
                             "quadrant": "Q1"} for s, t in pairs],
        "screen_results": [{"symbol": s, "tf": "1m"} for s in (screened or [])],
    }
    p = d / "surviving_assets.json"
    p.write_text(json.dumps(blob), encoding="utf-8")
    return p


def _scan_table(d: Path, tf: str, symbol: str, multi_tf: bool = True) -> Path:
    """One `scan_<SYM>.csv`, in the directory the sweep would write it to."""
    target = (d / tf) if multi_tf else d
    target.mkdir(parents=True, exist_ok=True)
    p = target / f"scan_{symbol}.csv"
    p.write_text("params,sharpe\n", encoding="utf-8")
    return p


def _best_params(d: Path, symbol: str, tf: str, variants: int = 360) -> Path:
    p = d / f"best_params_{symbol}_{tf}.json"
    p.write_text(json.dumps({"stage": 2, "symbol": symbol, "timeframe": tf,
                             "variants_tested": variants}), encoding="utf-8")
    return p


def _gate_audit(d: Path, symbol: str, tf: str) -> Path:
    p = d / f"gate_audit_{symbol}_{tf}.json"
    p.write_text(json.dumps({"stage": 3, "certified": False}), encoding="utf-8")
    return p


def _age(path: Path, seconds: float) -> None:
    """Backdate a file, so the freshness split has something to split."""
    when = time.time() - seconds
    os.utime(path, (when, when))


def _proc(pid: int, ppid: int, args: str, etimes: int = 600,
          pcpu: float = 89.4, pmem: float = 1.5) -> cp.Proc:
    return cp.Proc(pid=pid, ppid=ppid, etimes=etimes, pcpu=pcpu, pmem=pmem,
                   args=args)


def _orchestrator(strategy: str, symbols: str = "ES,NQ,RTY",
                  tf: str = "1m,5m", pid: int = 1171969) -> cp.Proc:
    return _proc(pid, 4242,
                 f"/home/x/src/trading/.venv/bin/python3 -u "
                 f"/home/x/src/trading/backtest/run_pipeline.py "
                 f"--strat {strategy} --symbols {symbols} --tf {tf} "
                 f"--start 2013-01-01 --end 2022-12-31 --report-discord "
                 f"--auto-promote", etimes=31934)


# ==========================================================================
# the process table
# ==========================================================================

def test_script_is_read_through_nice_and_ionice():
    """
    The shell helper launches `nice -n 19 ionice -c 3 python3 <stage>`.

    The command NAME is python3 in every one of those, so the stage has to be
    read off the arguments. A reader that matched on the command name would
    report every pipeline the documented helper starts as not running.
    """
    p = _proc(1, 0, "nice -n 19 ionice -c 3 /venv/bin/python3 -u "
                    "/repo/backtest/scan.py --strat demo")
    assert p.script() == "scan.py"
    assert cp.STAGE_OF_SCRIPT[p.script()] == 2


def test_relative_and_absolute_invocations_are_one_thing():
    a = _proc(1, 0, "python3 backtest/audit_gates.py --strat demo --tf 15m")
    b = _proc(2, 0, "python3 /repo/backtest/audit_gates.py --strat demo --tf 15m")
    assert a.script() == b.script() == "audit_gates.py"


def test_the_reader_never_reports_itself():
    """
    `watch bt-check` and `bt-check | tee` put this file's path on a second
    command line. Reporting that as a running pipeline is worse than reporting
    nothing, because it is a running pipeline that will never finish.
    """
    procs = [_proc(os.getpid(), 1, "python3 backtest/check_progress.py"),
             _proc(999999, 1, "watch -n5 python3 "
                              "/repo/backtest/check_progress.py --watch")]
    assert cp.find_orchestrators(procs) == []
    assert cp.find_stage_procs(procs) == []


def test_descendants_walks_more_than_one_generation():
    """
    bash -> nice -> run_pipeline.py -> scan.py, and a worker below that.

    One generation of ppid lookup finds scan.py today and silently finds
    nothing the day a stage grows a pool, so the walk is recursive.
    """
    procs = [_proc(10, 1, "bash"),
             _proc(20, 10, "python3 /repo/backtest/run_pipeline.py --strat x"),
             _proc(30, 20, "python3 /repo/backtest/scan.py --strat x"),
             _proc(40, 30, "python3 -c worker"),
             _proc(50, 1, "unrelated")]
    found = {p.pid for p in cp.descendants(procs, 20)}
    assert found == {30, 40}, found
    stages = [p for p in cp.descendants(procs, 20)
              if p.script() in cp.STAGE_OF_SCRIPT]
    assert [p.pid for p in stages] == [30]


def test_a_cycle_in_the_process_table_terminates():
    """A ppid loop must not hang a status tool. Contrived, and cheap to rule out."""
    procs = [_proc(10, 20, "a"), _proc(20, 10, "b")]
    assert len(cp.descendants(procs, 10)) <= 2


def test_flags_are_parsed_without_argparse():
    """
    This parses ANOTHER process's command line, possibly written by a version
    of the script with flags this one does not declare. argparse would exit on
    the first of those; an unknown flag has to survive as data.
    """
    flags = cp.parse_cli_flags(
        "python3 backtest/run_pipeline.py --strat demo_20260823 "
        "--symbols ES,NQ --tf 1m,5m --report-discord --auto-promote "
        "--out-dir=/tmp/x --some-future-flag 7")
    assert flags["strat"] == "demo_20260823"
    assert flags["symbols"] == "ES,NQ"
    assert flags["tf"] == "1m,5m"
    assert flags["report-discord"] is True
    assert flags["auto-promote"] is True
    assert flags["out-dir"] == "/tmp/x"
    assert flags["some-future-flag"] == "7"


def test_an_unreadable_process_table_is_not_an_idle_pipeline():
    """
    `ps` missing or refusing is a fact about this box, not about the pipeline.
    The two must not print the same card - one of them means "nothing is
    running" and the other means "I could not tell".
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        card = cp._no_campaign(None, ps_ok=False)
        assert "process table could not be read" in card
        clean = cp._no_campaign(None, ps_ok=True)
        assert "process table could not be read" not in clean


# ==========================================================================
# the Stage 2 plan - the number this tool exists to print
# ==========================================================================

def test_the_stage2_plan_is_rebuilt_in_the_order_the_sweep_walks_it():
    """
    `scan.main` groups Stage 1's pairs with `list(dict.fromkeys(...))`, so both
    the timeframe order and the symbol order inside one are FIRST APPEARANCE in
    surviving_assets.json - not sorted. A reader that sorted either would print
    a position in a plan nobody is walking.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("ES", "3m"), ("NQ", "3m"), ("RTY", "1m"),
                               ("NQ", "1m"), ("RTY", "3m")])
        plan = cp.Campaign("demo").stage2_plan()
        assert [tf for tf, _ in plan] == ["3m", "1m"], plan
        assert dict(plan)["3m"] == ["ES", "NQ", "RTY"]
        assert dict(plan)["1m"] == ["RTY", "NQ"]


def test_a_pair_with_no_timeframe_is_skipped_not_defaulted():
    """
    `pipeline.stage1_pairs` skips these, so this must too. Guessing a timeframe
    would queue a contract under one nobody screened it at.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        blob = json.loads((_survivors(d, "demo", [("ES", "3m")])).read_text())
        blob["surviving_pairs"].append({"symbol": "NQ", "version": "A"})
        blob["surviving_pairs"].append({"tf": "5m", "version": "A"})
        (d / "surviving_assets.json").write_text(json.dumps(blob))
        plan = cp.Campaign("demo").stage2_plan()
        assert plan == [("3m", ["ES"])], plan


def test_the_current_asset_is_the_first_one_with_no_table():
    """
    `[3/5] RB` is derived from the PLAN and the gaps in it, never from the
    newest file's name. The symbol being swept has no artifact precisely
    because it has not finished, so the newest file names the PREVIOUS one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("NQ", "1m"), ("CL", "1m")]
                   + [(s, "5m") for s in ("RTY", "NG", "RB", "6B", "6A")])
        for sym in ("NQ", "CL"):
            _scan_table(d, "1m", sym)
        for sym in ("RTY", "NG"):
            _scan_table(d, "5m", sym)

        c = cp.Campaign("demo")
        rows = c.stage2(c.stage2_plan(), None, live=True)
        by_tf = {r.tf: r for r in rows}
        assert by_tf["1m"].status == cp.COMPLETED
        live = by_tf["5m"]
        assert live.status == cp.RUNNING
        assert live.current == "RB", live.current
        assert live.position == 3, live.position
        assert len(live.planned) == 5
        assert live.pending == ["RB", "6B", "6A"], live.pending


def test_a_rerun_does_not_inherit_the_previous_runs_position():
    """
    THE RE-RUN TRAP, and the reason the freshness split reaches into Stage 2's
    position and not only into its labels.

    A re-run starts against a directory already holding the previous campaign's
    `scan_<SYM>.csv` for every timeframe. `scan.py` will rewrite every one of
    them, so as progress they are worth nothing - but counted, they put the
    card four timeframes and three contracts ahead of the sweep. The operator
    then reads `[3/5] RB` at minute two of a nine-hour run, on a contract the
    sweep will not reach until hour six, with every line reading correctly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [(s, "1m") for s in ("ES", "NQ", "RTY")]
                   + [(s, "5m") for s in ("RTY", "NG", "RB")])
        for tf, syms in (("1m", ("ES", "NQ", "RTY")), ("5m", ("RTY", "NG"))):
            for sym in syms:
                _age(_scan_table(d, tf, sym), 86_400)     # yesterday's campaign

        started = time.time() - 120                       # two minutes in
        c = cp.Campaign("demo", since=started)
        rows = c.stage2(c.stage2_plan(), None, live=True)
        first = rows[0]
        assert first.tf == "1m"
        assert first.status == cp.RUNNING, first.status
        assert first.current == "ES", first.current
        assert first.position == 1, first.position
        assert first.carried == ["ES", "NQ", "RTY"], first.carried
        # ...and the timeframe the stale tables make look part-done is simply
        # queued, because this run has not written a table anywhere yet.
        assert rows[1].status == cp.PENDING, rows[1].status
        assert rows[1].covered == []

        card = cp._render(c, _orchestrator("demo"), None, [], {}, ps_ok=True)
        assert "still on disk and are not counted above" in card


def test_a_hand_run_sweep_is_placed_at_the_timeframe_it_names():
    """
    `scan.py --tf 5m` run by hand says where it is; the plan only guesses.

    Under the orchestrator there is no --tf to read - the omission is how the
    sweep inherits Stage 1's ragged pairs - so the position comes from the
    plan. An operator who named one has better evidence, and preferring the
    plan there would report a timeframe nothing is sweeping.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("ES", "3m"), ("NQ", "3m"),
                               ("RTY", "5m"), ("NG", "5m")])
        _scan_table(d, "5m", "RTY")
        stage = _proc(777, 1, "/venv/bin/python3 /repo/backtest/scan.py "
                              "--strat demo --tf 5m", etimes=90)
        c = cp.Campaign("demo", since=stage.started)
        card = cp._render(c, None, stage, [stage],
                          cp.parse_cli_flags(stage.args), ps_ok=True)
        # 3m is untouched and earlier in the plan, so the plan alone would put
        # the sweep there.
        assert "Timeframe: 5m | Asset: [2/2] NG" in card, card

        # Two timeframes name neither, and the plan takes over again.
        stage = _proc(778, 1, "/venv/bin/python3 /repo/backtest/scan.py "
                              "--strat demo --tf 3m,5m", etimes=90)
        card = cp._render(c, None, stage, [stage],
                          cp.parse_cli_flags(stage.args), ps_ok=True)
        assert "Timeframe: 3m | Asset: [1/2] ES" in card, card


def test_a_pruned_pair_counts_as_swept():
    """
    Stage 2 writes `scan_<SYM>.csv` for every pair it EVALUATES and withholds
    `best_params_<SYM>_<TF>.json` when every combination failed the fragility
    bar. Counting parameter files instead would report a pair that ran and was
    pruned as one that has not started - the sweep would look stuck on it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("NQ", "1m"), ("CL", "1m")])
        _scan_table(d, "1m", "NQ")
        _scan_table(d, "1m", "CL")           # swept
        _best_params(d, "NQ", "1m")          # ...but only NQ produced a winner
        c = cp.Campaign("demo")
        rows = c.stage2(c.stage2_plan(), None, live=False)
        assert rows[0].status == cp.COMPLETED, rows[0].status


def test_a_single_timeframe_sweep_writes_beside_the_handoff():
    """
    `scan.main` uses `<out_dir>/<tf>/` only when the run spans MORE than one
    timeframe and `<out_dir>/` when it spans one. Looking in the wrong place
    reports a finished sweep as never started.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("NQ", "15m"), ("CL", "15m")])
        _scan_table(d, "15m", "NQ", multi_tf=False)
        _scan_table(d, "15m", "CL", multi_tf=False)
        c = cp.Campaign("demo")
        rows = c.stage2(c.stage2_plan(), None, live=False)
        assert rows[0].status == cp.COMPLETED, rows[0].status


def test_the_grid_size_comes_from_the_newest_parameter_file():
    """
    `variants_tested` is what ACTUALLY ran; the grid's cross product includes
    combinations the sweep rejects before simulating. And it is read from the
    newest file rather than from stage2_summary.json, which is written when
    Stage 2 FINISHES and therefore describes the previous run mid-sweep.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        old = _best_params(d, "NQ", "1m", variants=120)
        _age(old, 4000)
        _best_params(d, "NG", "5m", variants=360)
        n, src, fresh = cp.Campaign("demo").grid_size()
        assert n == 360, n
        assert src == "best_params_NG_5m.json", src
        assert fresh is True

        # ...and on a run that started well after every parameter file was
        # written, the number is the PREVIOUS sweep's and the card has to say
        # so rather than present it as the grid this run is evaluating. Aged
        # past the clock-skew tolerance, which is what makes it a different
        # run rather than a slow mount.
        _age(d / "best_params_NG_5m.json", 3600)
        n, src, fresh = cp.Campaign("demo", since=time.time()).grid_size()
        assert n == 360 and fresh is False, (n, fresh)


# ==========================================================================
# freshness - the re-run trap
# ==========================================================================

def test_an_earlier_campaigns_audits_are_not_this_runs():
    """
    The artifacts tree is one directory per STRATEGY, not per run, so a re-run
    overwrites the handoffs and leaves everything it has not reached in place.
    Four minutes into a nine-hour re-run, a reader that only asks "does
    gate_audit_NQ_15m.json exist" reports the PREVIOUS campaign's certification
    as this one's, with every line reading correctly. That is the exact failure
    this repository is built around, so the split is asserted rather than
    assumed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        old = _gate_audit(d, "NQ", "15m")
        _age(old, 86_400)                     # yesterday's run
        fresh = _gate_audit(d, "CL", "15m")   # this one's

        started = time.time() - 600
        s3 = cp.Campaign("demo", since=started).stage3(live=True)
        assert s3["audited"].fresh == ["CL_15m"], s3["audited"].fresh
        assert s3["audited"].stale == ["NQ_15m"], s3["audited"].stale
        assert s3["status"] == cp.RUNNING

        # With no run in flight there is nothing to be fresh RELATIVE to, so
        # the split is not drawn and both audits simply exist.
        idle = cp.Campaign("demo", since=None).stage3(live=False)
        assert idle["audited"].stale == []
        assert sorted(idle["audited"].fresh) == ["CL_15m", "NQ_15m"]
        assert fresh.exists()


def test_the_unsuffixed_gate_audit_is_not_counted():
    """
    `gate_audit_<SYMBOL>.json` holds whichever timeframe ran LAST. Counting it
    beside the suffixed files would double one pair and name no timeframe for
    the copy.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _gate_audit(d, "NQ", "15m")
        (d / "gate_audit_NQ.json").write_text("{}", encoding="utf-8")
        s3 = cp.Campaign("demo").stage3(live=False)
        assert s3["audited"].all == ["NQ_15m"], s3["audited"].all


def test_clock_skew_does_not_file_a_fresh_file_under_the_old_run():
    """
    The NFS server's clock and this box's are not identical. A file written in
    the run's first seconds must not land in the previous campaign because the
    mount is thirty seconds behind, so the split carries a tolerance.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        p = _gate_audit(d, "NQ", "15m")
        _age(p, 30)
        c = cp.Campaign("demo", since=time.time())   # "started" just now
        assert c._fresh(p), "a 30s skew must not reclassify this run's own file"


# ==========================================================================
# partial writes, missing trees, and other things that must not raise
# ==========================================================================

def test_a_truncated_handoff_is_not_a_crash():
    """
    Stages write through `os.replace`, but a status tool polling an NFS mount
    for nine hours will eventually catch one mid-flight. A JSONDecodeError
    surfacing out of a progress card reads as a broken pipeline rather than a
    broken read, and the next poll would have succeeded.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        (d / "surviving_assets.json").write_text(
            '{"stage": 1, "surviving_pairs": [{"symbol": "NQ",',
            encoding="utf-8")
        c = cp.Campaign("demo")
        assert c.survivors() is None
        assert c.stage2_plan() == []
        card, live = cp.build_report("demo", procs=[])
        assert "BACKTEST PIPELINE STATUS" in card
        assert live is False


def test_a_handoff_that_is_not_a_mapping_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        (d / "surviving_assets.json").write_text("[1, 2, 3]", encoding="utf-8")
        assert cp.Campaign("demo").survivors() is None


def test_a_campaign_directory_that_does_not_exist_still_renders():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        (Path(tmp) / "pipeline").mkdir(parents=True)
        card, live = cp.build_report("never_ran", procs=[])
        assert "does not exist yet" in card
        assert live is False


def test_an_empty_artifacts_root_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        (Path(tmp) / "pipeline").mkdir(parents=True)
        card, live = cp.build_report(procs=[])
        assert "IDLE" in card
        assert "holds no campaign" in card
        assert live is False


def test_the_newest_campaign_is_chosen_on_files_not_directory_mtime():
    """
    A directory's mtime moves when a subdirectory is CREATED and not when a
    file inside one is rewritten - so a sweep writing into `5m/` for six hours
    never touches its parent, and picking on directory mtime would report the
    wrong campaign for the whole sweep.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        old = _tree(Path(tmp), "older")
        new = _tree(Path(tmp), "newer")
        stale = _scan_table(old, "1m", "NQ")
        recent = _scan_table(new, "5m", "NG")
        _age(stale, 90_000)
        _age(old, 1)                       # the DIRECTORY is the newest thing
        _age(recent, 60)
        assert cp.newest_campaign() == "newer", cp.newest_campaign()


# ==========================================================================
# the rendered card
# ==========================================================================

def _running_card(tmp: str) -> str:
    """A card for a live Stage 2 sweep, built without a real process."""
    d = _tree(Path(tmp), "demo_20260823")
    _survivors(d, "demo_20260823",
               [("NQ", "1m"), ("CL", "1m")]
               + [(s, "5m") for s in ("RTY", "NG", "RB", "6B", "6A")],
               screened=["ES", "NQ", "RTY", "CL", "NG", "RB", "6B", "6A"])
    for sym in ("NQ", "CL"):
        _scan_table(d, "1m", sym)
    for sym in ("RTY", "NG"):
        _scan_table(d, "5m", sym)
    _best_params(d, "NG", "5m", variants=360)

    orch = _orchestrator("demo_20260823", symbols="ES,NQ,RTY,CL,NG,RB,6B,6A",
                         tf="1m,5m")
    stage = _proc(1171970, orch.pid,
                  "/venv/bin/python3 -u /repo/backtest/scan.py "
                  "--strat demo_20260823 --start 2013-01-01 --end 2022-12-31",
                  etimes=3600)
    c = cp.Campaign("demo_20260823", since=orch.started)
    return cp._render(c, orch, stage, [stage],
                      cp.parse_cli_flags(orch.args), ps_ok=True)


def test_the_running_card_reads_as_english():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        card = _running_card(tmp)
        assert "Strategy Name      : demo_20260823" in card
        assert "Running (PID: 1171969" in card
        assert "Elapsed Time       : 08h 52m 14s" in card, card
        assert "Stage 2 Optimization (Timeframe: 5m | Asset: [3/5] RB)" in card
        assert "360 parameter combinations evaluating" in card
        assert "8 Symbols (ES, NQ, RTY, CL, NG, RB, 6B, 6A)" in card
        assert "Target Timeframes  : 1m, 5m" in card
        assert "Discord Alerts     : Active (--report-discord enabled)" in card
        assert "Auto-Promote       : on" in card


def test_the_running_card_names_the_stage_breakdown():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        card = _running_card(tmp)
        assert "Stage 1 (Baseline) : COMPLETED" in card
        assert "Stage 2 (Optimize) : RUNNING" in card
        assert "Completed: 1m" in card
        assert "In Progress: 5m (2/5)" in card
        assert "Stage 3 (Certify)  : PENDING" in card
        # Stage 4 is verify_full.py - the LIFECYCLE run. Registration is Stage
        # 5, and labelling the lifecycle run "Register" would name a stage for
        # something it does not do.
        assert "Stage 4 (Verify)   : PENDING" in card
        assert "Stage 5 (Promote)  : PENDING" in card


def test_no_result_is_reported_anywhere_on_the_card():
    """
    This tool answers "where is it up to", never "was it any good". A Sharpe or
    a profit factor on a progress card is a promotion decision taken without
    the gate audit in front of it, which is the checkpoint the whole pipeline
    is sequenced around.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        card = _running_card(tmp).lower()
        for banned in ("sharpe", "profit factor", "profit_factor",
                       "drawdown", "win rate", "net pnl"):
            assert banned not in card, banned


def test_a_stage_run_by_hand_is_still_a_running_pipeline():
    """
    An operator running `scan.py` alone has no orchestrator above it. Reporting
    IDLE while a sweep burns a core is the single most misleading thing this
    tool could print.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("NQ", "1m"), ("CL", "1m")])
        _scan_table(d, "1m", "NQ")
        stage = _proc(777, 1, "/venv/bin/python3 /repo/backtest/scan.py "
                              "--strat demo", etimes=120)
        c = cp.Campaign("demo", since=stage.started)
        card = cp._render(c, None, stage, [stage],
                          cp.parse_cli_flags(stage.args), ps_ok=True)
        assert "Running one stage by hand" in card
        assert "Stage 2 Optimization" in card
        # No orchestrator means no --report-discord to report. Saying "Off"
        # would be a claim about a command line this tool never saw.
        assert "Discord Alerts" not in card


def test_an_orchestrator_between_stages_is_not_a_hang():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("NQ", "1m")])
        orch = _orchestrator("demo")
        c = cp.Campaign("demo", since=orch.started)
        card = cp._render(c, orch, None, [],
                          cp.parse_cli_flags(orch.args), ps_ok=True)
        assert "between stages" in card


def test_the_idle_card_refuses_to_invent_a_verdict():
    """
    Nothing on disk records the exit code of a run that has already gone, and a
    run killed at hour eight leaves a tree indistinguishable from one that
    finished up to the stage it died in. So the idle card reports HOW FAR the
    artifacts reach and says in as many words that this is not a pass/fail.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("NQ", "1m")])
        _scan_table(d, "1m", "NQ")
        _gate_audit(d, "NQ", "1m")
        c = cp.Campaign("demo")
        card = cp._render(c, None, None, [], {}, ps_ok=True)
        assert "IDLE" in card
        assert "Last Run Strategy  : demo" in card
        assert "Stage 3 — certification ran" in card
        assert "not a pass/fail verdict" in card


# ==========================================================================
# the executable, end to end
# ==========================================================================

def test_the_script_runs_clean_from_any_directory():
    """
    The acceptance criterion, exercised as written: `bt-check` from somewhere
    that is not the repository. The module's own sys.path bootstrap is the
    thing under test - `python3 backtest/x.py` puts backtest/ on the path and
    not the repository root, so mdlib would be unimportable without it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, BT_ARTIFACTS=tmp)
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("NQ", "1m")])
        _scan_table(d, "1m", "NQ")
        out = subprocess.run(
            [sys.executable, str(REPO / "backtest" / "check_progress.py"),
             "--strategy", "demo"],
            capture_output=True, text=True, cwd=tempfile.gettempdir(),
            env=env, timeout=120)
        assert out.returncode == 0, out.stderr
        assert "BACKTEST PIPELINE STATUS" in out.stdout
        assert "Strategy Name      : demo" in out.stdout
        assert "Traceback" not in out.stderr
        # The strategy is named rather than left to auto-detection ON PURPOSE.
        # Unpinned, this case reports whichever campaign is newest on the real
        # mount - so it would pass on an idle box and fail on the same box the
        # moment somebody ran the gate during a sweep.


def test_an_injected_process_table_is_what_the_card_is_built_from():
    """
    The seam itself. `procs=[]` means "the table was read and holds no
    pipeline", which must render IDLE no matter what is running on this box -
    otherwise the suite passes when idle and fails during the sweep somebody
    ran it to check on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        d = _tree(Path(tmp), "demo")
        _survivors(d, "demo", [("NQ", "1m")])
        card, live = cp.build_report("demo", procs=[])
        assert live is False
        assert "IDLE" in card
        assert "process table could not be read" not in card

        orch = _orchestrator("demo")
        card, live = cp.build_report("demo", procs=[orch])
        assert live is True
        assert f"Running (PID: {orch.pid}" in card


def test_the_script_accepts_an_explicit_strategy():
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, BT_ARTIFACTS=tmp)
        for name in ("alpha", "beta"):
            _survivors(_tree(Path(tmp), name), name, [("NQ", "1m")])
        out = subprocess.run(
            [sys.executable, str(REPO / "backtest" / "check_progress.py"),
             "--strategy", "alpha"],
            capture_output=True, text=True, cwd=tempfile.gettempdir(),
            env=env, timeout=120)
        assert out.returncode == 0, out.stderr
        assert "Strategy Name      : alpha" in out.stdout


def test_it_never_writes_to_the_artifacts_tree():
    """
    A status tool that creates a directory to report on it has changed the
    thing it was asked to observe. `pipeline_dir(..., create=False)` is the
    default and this pins it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BT_ARTIFACTS"] = tmp
        before = sorted(p.name for p in Path(tmp).iterdir())
        cp.build_report("no_such_strategy", procs=[])
        after = sorted(p.name for p in Path(tmp).iterdir())
        assert before == after, (before, after)


# ==========================================================================
# formatting
# ==========================================================================

def test_elapsed_is_zero_padded_so_readouts_line_up():
    assert cp.elapsed_text(31934) == "08h 52m 14s"
    assert cp.elapsed_text(0) == "00h 00m 00s"
    assert cp.elapsed_text(None) == "unknown"


def test_ago_is_coarse_where_precision_would_be_noise():
    assert cp.ago_text(14) == "14s ago"
    assert cp.ago_text(3600).startswith("60m")
    assert cp.ago_text(7 * 3600).startswith("7h")
    assert cp.ago_text(72 * 3600).startswith("3d")


def test_a_universe_is_named_not_dumped():
    assert cp.truncated_list([]) == "none"
    assert cp.truncated_list(["ES", "NQ"]) == "ES, NQ"
    long = cp.truncated_list([f"S{i}" for i in range(23)])
    assert long.endswith("(+12 more)"), long
    assert long.count(",") == 11


def main() -> int:
    """
    A script entry point, so this file behaves the same under pytest and alone.

    Assert-based throughout, so both runners see the same failure - unlike the
    collector-style suites, where pytest watches checks fail and reports green.
    """
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:                            # noqa: PERF203
            failures += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                 # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
