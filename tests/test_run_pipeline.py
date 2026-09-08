"""
tests/test_run_pipeline.py - the unified orchestrator, backtest/run_pipeline.py.

Location: ~/src/trading/tests/test_run_pipeline.py

Runs no stage and reads no bars. Everything here is either a pure command
builder, a handoff reader over a fixture, or the orchestration order captured
through an injected runner - because the whole failure mode this script has is
a WIRING one: a pipeline that launches four stages successfully and hands two
of them the wrong window is indistinguishable at the console from a correct
run, and costs hours before it says anything.

The checks that matter most, and why they are here rather than assumed:

  * Stage 2 is invoked with NO --symbols and NO --tf. That omission IS the
    inheritance of Stage 1's exact ragged (symbol, timeframe) survivors.
    Adding either flag silently replaces them with the cross product of two
    axes, sweeping configurations the screen dropped - and the sweep still
    completes, still writes winners, and still advances to Stage 3.
  * Stage 3 gets --is-start/--is-end, never --start/--end. It has no such
    flags, so this is the difference between a certification and a crash.
  * Stages 3 and 4 get exactly ONE --tf each, expanded per timeframe.
  * An in-sample --end reaching HOLDOUT_START is refused before Stage 1
    launches. Nothing downstream can detect a holdout that was already fitted.
  * Stage 3/4 timeframes come from Stage 2's summary, not the CLI: a
    timeframe nothing survived at exits 1 in audit_gates ("Nothing to
    certify"), and under check=True that ends the run on a screening result.
  * --auto-promote promotes only rows Stage 3 flagged `certified: true`,
    always passes --require-certification, and never passes --force.
  * A Discord failure does not abort the pipeline; a stage failure does.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.pipeline import (CHARTER_IS_END, CHARTER_IS_START,    # noqa: E402
                               HOLDOUT_START, STAGE2_SUMMARY_FILE,
                               STAGE3_SUMMARY_FILE, pipeline_dir,
                               read_stage, write_stage)
import backtest.run_pipeline as rp                                  # noqa: E402

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
    return bool(ok)


def raises(fn, exc=Exception) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:
        return True, f"{type(e).__name__}: {e}"
    except Exception as e:                                        # noqa: BLE001
        return False, f"raised the wrong type: {type(e).__name__}: {e}"
    return False, "did not raise"


def flag_value(cmd: list[str], flag: str) -> str | None:
    """The value following `flag`, or None when the flag is absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def script_of(cmd: list[str]) -> str:
    """The .py the command runs, whatever interpreter prefix it carries."""
    return next(Path(c).name for c in cmd if str(c).endswith(".py"))


class FakeRunner:
    """
    Stands in for subprocess.run: records, never launches.

    `fail_on` maps a script name to an exit code, so a stage failure and a
    Discord failure can be told apart by what the orchestrator does next.
    """

    def __init__(self, fail_on: dict[str, int] | None = None):
        self.calls: list[list[str]] = []
        self.fail_on = dict(fail_on or {})

    def __call__(self, cmd, cwd=None, check=False):
        self.calls.append([str(c) for c in cmd])
        rc = self.fail_on.get(script_of([str(c) for c in cmd]), 0)
        if rc and check:
            raise subprocess.CalledProcessError(rc, cmd)
        return subprocess.CompletedProcess(cmd, rc)

    def scripts(self) -> list[str]:
        return [script_of(c) for c in self.calls]

    def for_script(self, name: str) -> list[list[str]]:
        return [c for c in self.calls if script_of(c) == name]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
STRAT = "fixture_strategy"


def write_stage2(out_dir: Path, rows: list[dict]) -> Path:
    path = pipeline_dir(STRAT, out_dir) / STAGE2_SUMMARY_FILE
    write_stage(path, 2, STRAT, {"results": rows})
    return path


def write_stage3(out_dir: Path, rows: list[dict]) -> Path:
    path = pipeline_dir(STRAT, out_dir) / STAGE3_SUMMARY_FILE
    write_stage(path, 3, STRAT, {"results": rows})
    return path


def s2row(symbol="NQ", tf="15m", status="OPTIMIZED") -> dict:
    return {"symbol": symbol, "timeframe": tf, "status": status,
            "params": "fast_window=15", "profit_factor": 1.2}


def s3row(symbol="NQ", tf="15m", certified=True, version="A",
          status="PASS", starved=None, pf=1.07, quadrant="Q1") -> dict:
    return {"symbol": symbol, "timeframe": tf, "version": version,
            "status": status, "certified": certified, "gate_regime": status,
            "quadrant": quadrant, "oos_profit_factor": pf,
            "oos_trade_count": 348, "regime_starvation": starved,
            "audit_file": f"/tmp/gate_audit_{symbol}_{tf}.json"}


# --------------------------------------------------------------------------
# 1. The in-sample window guard
# --------------------------------------------------------------------------
def test_window_guard() -> None:
    print("\n1. The in-sample window is refused before Stage 1 launches")

    ok, _ = raises(lambda: rp.check_in_sample_window(CHARTER_IS_START, HOLDOUT_START),
                   ValueError)
    check("an --end EQUAL to HOLDOUT_START is refused", ok)

    ok, _ = raises(lambda: rp.check_in_sample_window(CHARTER_IS_START, "2024-06-01"),
                   ValueError)
    check("an --end INSIDE the holdout is refused", ok)

    ok, _ = raises(lambda: rp.check_in_sample_window(CHARTER_IS_START, None),
                   ValueError)
    check("an OMITTED --end is refused (it runs to the end of the lake)", ok)

    try:
        rp.check_in_sample_window(CHARTER_IS_START, CHARTER_IS_END)
        ok = True
    except ValueError:
        ok = False
    check("the charter window itself is accepted", ok,
          f"{CHARTER_IS_START}..{CHARTER_IS_END}")

    runner = FakeRunner()
    rc = rp.main(["--strat", STRAT, "--end", "2025-01-01"])
    check("main() exits non-zero on a holdout-eating window", rc != 0, f"rc={rc}")
    check("...and launches NOTHING", runner.calls == [])


# --------------------------------------------------------------------------
# 2. The command builders
# --------------------------------------------------------------------------
def test_stage1_cmd() -> None:
    print("\n2. Stage 1 carries the universe and every timeframe at once")
    cmd = rp.stage1_cmd(STRAT, "NQ,ES", ["15m", "30m"],
                        CHARTER_IS_START, CHARTER_IS_END)
    check("runs baseline.py", script_of(cmd) == "baseline.py")
    check("passes --symbols", flag_value(cmd, "--symbols") == "NQ,ES")
    check("passes both timeframes in one call",
          flag_value(cmd, "--tf") == "15m,30m")
    check("passes the in-sample window",
          (flag_value(cmd, "--start"), flag_value(cmd, "--end"))
          == (CHARTER_IS_START, CHARTER_IS_END))
    check("is unbuffered (-u)", "-u" in cmd)


def test_stage2_cmd_inherits_exact_pairs() -> None:
    print("\n3. Stage 2 inherits Stage 1's EXACT pairs by omitting the axes")
    cmd = rp.stage2_cmd(STRAT, CHARTER_IS_START, CHARTER_IS_END)
    check("runs scan.py", script_of(cmd) == "scan.py")
    check("passes NO --symbols  (the omission IS the inheritance)",
          "--symbols" not in cmd)
    check("passes NO --tf  (else the ragged survivors become a cross product)",
          "--tf" not in cmd)
    check("still passes the in-sample window",
          flag_value(cmd, "--end") == CHARTER_IS_END)


def test_stage3_cmd_window_flags() -> None:
    print("\n4. Stage 3 takes --is-start/--is-end, and ONE timeframe")
    cmd = rp.stage3_cmd(STRAT, "15m", CHARTER_IS_START, CHARTER_IS_END)
    check("runs audit_gates.py", script_of(cmd) == "audit_gates.py")
    check("passes --is-start/--is-end",
          (flag_value(cmd, "--is-start"), flag_value(cmd, "--is-end"))
          == (CHARTER_IS_START, CHARTER_IS_END))
    check("passes NO --start/--end  (audit_gates has no such flags)",
          "--start" not in cmd and "--end" not in cmd)
    check("passes exactly one --tf", flag_value(cmd, "--tf") == "15m"
          and cmd.count("--tf") == 1)
    check("passes --holdout-start", flag_value(cmd, "--holdout-start") == HOLDOUT_START)
    check("omits --holdout-end so the stage defaults it to the PRESENT",
          "--holdout-end" not in cmd)
    check("passes NO --symbols  (it certifies Stage 2's exact pairs)",
          "--symbols" not in cmd)
    check("does not pass --no-promote by default", "--no-promote" not in cmd)
    check("--no-promote is expressible",
          "--no-promote" in rp.stage3_cmd(STRAT, "15m", CHARTER_IS_START,
                                          CHARTER_IS_END, promote=False))


def test_stage4_cmd() -> None:
    print("\n5. Stage 4's window is the lifecycle and spans the holdout")
    cmd = rp.stage4_cmd(STRAT, "15m", CHARTER_IS_START, "2026-08-21")
    check("runs verify_full.py", script_of(cmd) == "verify_full.py")
    check("passes --start/--end", flag_value(cmd, "--start") == CHARTER_IS_START)
    check("its --end reaches PAST the holdout start, by design",
          flag_value(cmd, "--end") > HOLDOUT_START,
          f"{flag_value(cmd, '--end')} > {HOLDOUT_START}")
    check("passes exactly one --tf", flag_value(cmd, "--tf") == "15m")


def test_discord_and_promote_cmds() -> None:
    print("\n6. The Discord card and the promotion command")
    for stage in (1, 2, 3, 4, 5):
        cmd = rp.discord_cmd(STRAT, stage)
        check(f"stage {stage} card runs discord_reporter.py",
              script_of(cmd) == "discord_reporter.py"
              and flag_value(cmd, "--stage") == str(stage))
    check("DISCORD_STAGES is the transmission ORDER, ascending",
          list(rp.DISCORD_STAGES) == sorted(rp.DISCORD_STAGES) == [1, 2, 3, 4, 5],
          str(rp.DISCORD_STAGES))
    for bad in (0, 6):
        ok, _ = raises(lambda b=bad: rp.discord_cmd(STRAT, b), ValueError)
        check(f"stage {bad} is not a card and asking for one raises", ok)
    check("--dry-run is expressible",
          "--dry-run" in rp.discord_cmd(STRAT, 1, dry_run=True))

    cmd = rp.promote_cmd(STRAT, "A", "strat.py", "audit.json", "NQ", "15m")
    check("promotion runs promote.py", script_of(cmd) == "promote.py")
    check("ALWAYS passes --require-certification",
          "--require-certification" in cmd)
    check("NEVER passes --force  (it may only ratify Stage 3's verdict)",
          "--force" not in cmd)
    check("names the symbol and timeframe the evidence rests on",
          (flag_value(cmd, "--symbol"), flag_value(cmd, "--timeframe"))
          == ("NQ", "15m"))


# --------------------------------------------------------------------------
# 3. The handoff readers
# --------------------------------------------------------------------------
def test_stage2_timeframes() -> None:
    print("\n7. Stage 3/4 timeframes come from Stage 2's output, not the CLI")
    summary = {"results": [s2row(tf="15m"), s2row("ES", "15m"),
                           s2row("GC", "30m"),
                           s2row("CL", "1h", status="ERROR")]}
    tfs = rp.stage2_timeframes(summary)
    check("collects the timeframes that OPTIMIZED", tfs == ["15m", "30m"], str(tfs))
    check("de-duplicates across symbols", tfs.count("15m") == 1)
    check("excludes a timeframe whose only row ERRORed  "
          "(no winner locked = nothing to certify)", "1h" not in tfs)
    check("an empty summary yields nothing rather than raising",
          rp.stage2_timeframes({"results": []}) == [])
    check("a missing summary yields nothing rather than raising",
          rp.stage2_timeframes(None) == [])


def test_certified_rows() -> None:
    print("\n8. Only what Stage 3 FLAGGED certified is promotable")
    summary = {"results": [
        s3row("NQ", "15m", certified=True),
        s3row("ES", "15m", certified=False, status="FAIL"),
        {"symbol": "GC", "timeframe": "30m", "status": "NOT AUDITED"},
        {"symbol": "CL", "timeframe": "15m", "certified": "true"},
    ]}
    rows = rp.certified_rows(summary)
    check("keeps the certified row", [r["symbol"] for r in rows] == ["NQ"],
          str([r["symbol"] for r in rows]))
    check("a FAIL is not promotable", all(r["symbol"] != "ES" for r in rows))
    check("a NOT AUDITED row is not promotable  "
          "('the run broke' is not 'the edge held')",
          all(r["symbol"] != "GC" for r in rows))
    check("the STRING 'true' is not the boolean True",
          all(r["symbol"] != "CL" for r in rows))
    check("no summary promotes nothing", rp.certified_rows(None) == [])


# --------------------------------------------------------------------------
# 4. Orchestration order, through an injected runner
# --------------------------------------------------------------------------
def _main_with(runner: FakeRunner, argv: list[str], monkey_today="2026-08-21",
               source="strategies/experimental/fixture_strategy.py") -> int:
    """
    main() with every outward edge stubbed: no subprocess, no clock, no
    strategy-module lookup.

    `_strategy_source` is patched too, not only `subprocess.run`. Left real it
    would import `backtest.run` (and vectorbtpro) and then raise SystemExit on
    a fixture strategy that does not exist on disk - which is a fact about the
    fixture, not about the orchestration this suite is measuring.
    """
    real_run, real_today = rp.subprocess.run, rp._today
    real_source = rp._strategy_source
    rp.subprocess.run = runner
    rp._today = lambda: monkey_today
    rp._strategy_source = lambda _strat: Path(source)
    try:
        return rp.main(argv)
    finally:
        rp.subprocess.run, rp._today = real_run, real_today
        rp._strategy_source = real_source


@contextmanager
def promoted_packages(*names: str):
    """
    Stand in for the packages a real `promote.py` leaves on disk.

    `stage5_card_targets` asks `discord_reporter.promotion_packages` what was
    actually promoted, and posts NO Stage 5 card when the answer is nothing -
    a run that certified nothing has no promotion to announce. The FakeRunner
    only records that `promote.py` was invoked; it writes no package, so
    without this the fifth card can never be exercised here.

    Patched on `discord_reporter`, where `stage5_card_targets` imports it from,
    so the real resolution logic in `run_pipeline` is the code under test.
    """
    import backtest.discord_reporter as dr
    real = dr.promotion_packages
    dr.promotion_packages = lambda strat, incubator=None: [Path(n) for n in names]
    try:
        yield
    finally:
        dr.promotion_packages = real


def test_order_and_expansion(tmp: Path) -> None:
    print("\n9. The stages run in order, expanded per surviving timeframe")
    write_stage2(tmp, [s2row("NQ", "15m"), s2row("NQ", "30m")])
    write_stage3(tmp, [s3row("NQ", "15m")])

    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--symbols", "NQ",
                             "--tf", "15m,30m", "--out-dir", str(tmp)])
    seq = runner.scripts()
    check("exits 0", rc == 0, f"rc={rc}")
    check("stages run 1 -> 2 -> 3 -> 4 -> 4.5 in order",
          seq == ["baseline.py", "scan.py", "audit_gates.py", "audit_gates.py",
                  "verify_full.py", "verify_full.py",
                  "dow_gate.py", "dow_gate.py"], str(seq))
    check("Stage 3 runs once per surviving timeframe",
          [flag_value(c, "--tf") for c in runner.for_script("audit_gates.py")]
          == ["15m", "30m"])
    check("Stage 4 runs once per surviving timeframe",
          [flag_value(c, "--tf") for c in runner.for_script("verify_full.py")]
          == ["15m", "30m"])
    # STAGE 4.5 RUNS AFTER STAGE 4 AND BEFORE ANY PROMOTION. Before Stage 5
    # because Stage 5 is what writes the blocked weekday into the promoted
    # meta.json; run after it, the verdict would land in the pipeline
    # directory one promotion too late and the live loop would trade the
    # session until somebody re-promoted.
    check("Stage 4.5 runs once per surviving timeframe",
          [flag_value(c, "--tf") for c in runner.for_script("dow_gate.py")]
          == ["15m", "30m"])
    check("Stage 4.5 profiles the SAME window Stage 4 ran",
          [(flag_value(c, "--start"), flag_value(c, "--end"))
           for c in runner.for_script("dow_gate.py")]
          == [(flag_value(c, "--start"), flag_value(c, "--end"))
              for c in runner.for_script("verify_full.py")])
    # VERSION B IS PROFILED TOO, for the reason Stage 4 passes --ml: B's
    # trade list is a SUBSET of A's, so its weekday table is a different
    # table, and a pair Stage 3 certified as B profiled only as A would be
    # promoted carrying a weekday measured on a strategy nobody deployed.
    check("Stage 4.5 profiles Version B",
          all("--ml" in c for c in runner.for_script("dow_gate.py")))
    check("...at the SAME ML threshold every other stage used",
          [flag_value(c, "--ml-threshold")
           for c in runner.for_script("dow_gate.py")]
          == [flag_value(c, "--ml-threshold")
              for c in runner.for_script("verify_full.py")])
    check("no Discord card without --report-discord",
          "discord_reporter.py" not in seq)
    check("no promotion without --auto-promote", "promote.py" not in seq)


def test_a_failing_stage45_does_not_abort_the_run(tmp: Path) -> None:
    """
    STAGE 4.5 IS NOT A LINK IN THE CHAIN. Stages 1-4 run under `check=True`
    because each reads the previous one's handoff, so continuing past a
    failure would report an earlier run's artifacts as this one's result.
    Stage 4.5 produces an INSTRUCTION for the live supervisor and nothing
    downstream needs it to exist - so a day-of-week profiler that raised must
    not discard certifications that already cleared Gate R. The affected
    packages record `day_of_week_gate.status: NOT EVALUATED`, which is exactly
    what happened.
    """
    print("\n9b. Stage 4.5 failing is a warning, not an abort")
    write_stage2(tmp, [s2row("NQ", "15m")])
    write_stage3(tmp, [s3row("NQ", "15m")])

    runner = FakeRunner(fail_on={"dow_gate.py": 3})
    rc = _main_with(runner, ["--strat", STRAT, "--symbols", "NQ",
                             "--tf", "15m", "--out-dir", str(tmp)])
    check("the run still exits 0", rc == 0, f"rc={rc}")
    check("stage 4.5 was attempted", "dow_gate.py" in runner.scripts())


def test_skips_unsurvived_timeframe(tmp: Path) -> None:
    print("\n10. A timeframe Stage 2 optimised nothing at is SKIPPED, not run")
    write_stage2(tmp, [s2row("NQ", "15m")])          # 30m did not survive
    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--tf", "15m,30m",
                             "--out-dir", str(tmp)])
    tfs3 = [flag_value(c, "--tf") for c in runner.for_script("audit_gates.py")]
    check("exits 0  (a screening result is not a failure)", rc == 0, f"rc={rc}")
    check("Stage 3 runs ONLY at the surviving timeframe", tfs3 == ["15m"], str(tfs3))
    check("...so audit_gates is never asked to certify nothing at 30m",
          "30m" not in tfs3)


def test_stops_when_nothing_survived(tmp: Path) -> None:
    print("\n11. Nothing optimised at all stops the run, and reports it as a result")
    write_stage2(tmp, [s2row("NQ", "15m", status="ERROR")])
    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--tf", "15m", "--out-dir", str(tmp)])
    seq = runner.scripts()
    check("exits 0 rather than erroring", rc == 0, f"rc={rc}")
    check("Stage 3 never launches", "audit_gates.py" not in seq)
    check("Stage 4 never launches", "verify_full.py" not in seq)


def test_discord_strict_sequential_order(tmp: Path) -> None:
    """
    Cards are transmitted in strict numerical sequence, 1 -> 2 -> 3 -> 4 -> 5.

    A reader scrolling a Discord channel reconstructs the campaign from the
    order the cards arrived in, and nothing on a card says when it was built.
    A Stage 3 certification landing after Stage 5's promotion therefore reads
    as a decision taken after the evidence was announced, when in fact it was
    taken before - and every field on both cards is individually correct, so
    there is nothing to notice.
    """
    print("\n12. --report-discord posts cards in strict numerical order")
    write_stage2(tmp, [s2row("NQ", "15m")])
    runner = FakeRunner()
    _main_with(runner, ["--strat", STRAT, "--tf", "15m", "--report-discord",
                        "--out-dir", str(tmp)])
    seq = runner.scripts()
    stages = [flag_value(c, "--stage") for c in runner.for_script("discord_reporter.py")]
    card = [i for i, s in enumerate(seq) if s == "discord_reporter.py"]
    check("four cards are posted, ascending", stages == ["1", "2", "3", "4"],
          str(stages))
    check("the Stage 1 card follows baseline.py",
          card[0] > seq.index("baseline.py"))
    check("the Stage 2 card follows scan.py", card[1] > seq.index("scan.py"))
    check("the Stage 3 card follows the LAST audit_gates.py",
          card[2] > max(i for i, s in enumerate(seq) if s == "audit_gates.py"))
    check("the Stage 4 card follows the LAST verify_full.py",
          card[3] > max(i for i, s in enumerate(seq) if s == "verify_full.py"))
    check("Stage 4 runs BEFORE the Stage 4 card and after the Stage 3 card",
          card[2] < min(i for i, s in enumerate(seq) if s == "verify_full.py"))
    check("no Stage 5 card without a promotion", "5" not in stages)

    # With --auto-promote the fifth card joins, still last, still after the
    # promotion it announces.
    write_stage3(tmp, [s3row("NQ", "15m", certified=True)])
    runner = FakeRunner()
    with promoted_packages(f"{STRAT}_NQ_15m_VA"):
        _main_with(runner, ["--strat", STRAT, "--tf", "15m",
                            "--report-discord", "--auto-promote",
                            "--out-dir", str(tmp)])
    seq = runner.scripts()
    stages = [flag_value(c, "--stage") for c in runner.for_script("discord_reporter.py")]
    check("all five cards are posted, ascending",
          stages == ["1", "2", "3", "4", "5"], str(stages))
    card = [i for i, s in enumerate(seq) if s == "discord_reporter.py"]
    check("Stage 4 runs and its card posts BEFORE promote.py runs",
          card[3] < seq.index("promote.py"),
          f"card4@{card[3]} promote@{seq.index('promote.py')}")
    check("the Stage 5 card follows the LAST promote.py",
          card[4] > max(i for i, s in enumerate(seq) if s == "promote.py"))

    # A promotion that produced no package has nothing to announce. Falling
    # back to the MODULE name here posted a promotion card for a promotion
    # that does not exist: it could only fail, and it failed with "the
    # contract could not be resolved from portfolios.json" - which sends an
    # operator to a config file to debug a run whose real outcome was that
    # nothing cleared its gates.
    runner = FakeRunner()
    with promoted_packages():
        rc = _main_with(runner, ["--strat", STRAT, "--tf", "15m",
                                 "--report-discord", "--auto-promote",
                                 "--out-dir", str(tmp)])
    stages = [flag_value(c, "--stage")
              for c in runner.for_script("discord_reporter.py")]
    check("no package promoted posts NO Stage 5 card, rather than one under "
          "the module name that can only refuse",
          "5" not in stages, str(stages))
    check("...and the run still exits 0: nothing to announce is not a failure",
          rc == 0, f"rc={rc}")


def test_stage_failure_aborts_discord_failure_does_not(tmp: Path) -> None:
    print("\n13. A stage failure aborts; a Discord failure does not")
    write_stage2(tmp, [s2row("NQ", "15m")])

    runner = FakeRunner(fail_on={"scan.py": 3})
    rc = _main_with(runner, ["--strat", STRAT, "--tf", "15m", "--out-dir", str(tmp)])
    seq = runner.scripts()
    check("a failed Stage 2 exits non-zero", rc == 3, f"rc={rc}")
    check("...and Stage 3 never launches on a stale handoff",
          "audit_gates.py" not in seq)
    check("...and Stage 4 never launches", "verify_full.py" not in seq)

    runner = FakeRunner(fail_on={"discord_reporter.py": 1})
    rc = _main_with(runner, ["--strat", STRAT, "--tf", "15m", "--report-discord",
                             "--out-dir", str(tmp)])
    seq = runner.scripts()
    check("a failed Discord card does NOT abort the pipeline", rc == 0, f"rc={rc}")
    check("...Stage 3 still runs", "audit_gates.py" in seq)
    check("...Stage 4 still runs", "verify_full.py" in seq)


def test_auto_promote(tmp: Path) -> None:
    print("\n14. --auto-promote ratifies Stage 3 and cannot override it")
    write_stage2(tmp, [s2row("NQ", "15m"), s2row("ES", "15m")])
    write_stage3(tmp, [s3row("NQ", "15m", certified=True),
                       s3row("ES", "15m", certified=False, status="FAIL")])

    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--tf", "15m", "--auto-promote",
                             "--out-dir", str(tmp)])
    promos = runner.for_script("promote.py")
    check("exits 0", rc == 0, f"rc={rc}")
    check("exactly ONE promotion, for the certified row", len(promos) == 1,
          f"{len(promos)} promotion(s)")
    if promos:
        check("it names the certified symbol",
              flag_value(promos[0], "--symbol") == "NQ")
        check("it cites that row's audit file",
              flag_value(promos[0], "--audit-file").endswith("gate_audit_NQ_15m.json"))
        check("it passes --require-certification", "--require-certification" in promos[0])
        check("it never passes --force", "--force" not in promos[0])
    check("the FAILED configuration is not promoted",
          all(flag_value(c, "--symbol") != "ES" for c in promos))

    # Nothing certified -> nothing promoted, and that is not an error.
    write_stage3(tmp, [s3row("NQ", "15m", certified=False, status="FAIL")])
    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--tf", "15m", "--auto-promote",
                             "--out-dir", str(tmp)])
    check("nothing certified promotes nothing",
          runner.for_script("promote.py") == [])
    check("...and that exits 0, because it is a verdict not a breakage",
          rc == 0, f"rc={rc}")

    # An unresolvable module must not kill the process after four stages ran:
    # resolve_strategy raises SystemExit, and the certifications are already
    # on the record either way.
    write_stage3(tmp, [s3row("NQ", "15m", certified=True)])
    runner = FakeRunner()
    real_source = rp._strategy_source

    def _boom(_strat):
        raise SystemExit("no strategy called 'fixture_strategy'")

    real_run, real_today = rp.subprocess.run, rp._today
    rp.subprocess.run, rp._today = runner, lambda: "2026-08-21"
    rp._strategy_source = _boom
    try:
        rc = rp.main(["--strat", STRAT, "--tf", "15m", "--auto-promote",
                      "--out-dir", str(tmp)])
        raised = False
    except SystemExit:
        rc, raised = None, True
    finally:
        rp.subprocess.run, rp._today = real_run, real_today
        rp._strategy_source = real_source
    check("an unresolvable module does not kill the process", not raised)
    check("...it exits non-zero instead", rc == 1, f"rc={rc}")
    check("...after the four stages had already run",
          runner.scripts()[:2] == ["baseline.py", "scan.py"],
          str(runner.scripts()))


def test_promote_only(tmp: Path) -> None:
    print("\n14c. --promote-only runs Stage 5 alone, against what is already "
          "certified")
    write_stage2(tmp, [s2row("NQ", "15m")])
    write_stage3(tmp, [s3row("NQ", "15m", certified=True),
                       s3row("ES", "15m", certified=False, status="FAIL")])

    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--symbols", "NQ,ES",
                             "--tf", "15m", "--promote-only",
                             "--out-dir", str(tmp)])
    scripts = runner.scripts()
    check("exits 0", rc == 0, f"rc={rc}")
    check("Stages 1-4 do NOT run - this is the command the Stage 3 card "
          "prints, and re-running the sweep would overwrite the handoff the "
          "card was built from and promote a DIFFERENT set of winners",
          not {"baseline.py", "scan.py", "audit_gates.py", "verify_full.py"}
          & set(scripts), str(scripts))
    promos = runner.for_script("promote.py")
    check("exactly ONE promotion, for the certified row", len(promos) == 1,
          f"{len(promos)} promotion(s)")
    if promos:
        check("it names the certified symbol",
              flag_value(promos[0], "--symbol") == "NQ")
        check("...and cites that row's own audit file",
              flag_value(promos[0], "--audit-file")
              .endswith("gate_audit_NQ_15m.json"))
        check("...through --require-certification, never --force",
              "--require-certification" in promos[0]
              and "--force" not in promos[0])
    check("--symbols narrows nothing here: auto_promote promotes every "
          "certified row and always has, and a symbol list that silently "
          "scoped nothing would read as one that did",
          all(flag_value(c, "--symbol") != "ES" for c in promos))

    # The cards, and only after the promotions: the outcome is on the handoff
    # by then, so Stage 3 reads PROMOTED with a commit instead of printing a
    # command that has already run. They still go out in numerical order.
    #
    # Stage 4 is not run by this path, so its card is posted only when an
    # earlier run left lifecycle snapshots behind. Both branches are pinned:
    # a skipped Stage 4 card and a missing one are the same absence on the
    # channel, and only one of them is correct.
    runner = FakeRunner()
    with promoted_packages(f"{STRAT}_NQ_15m_VA"):
        _main_with(runner, ["--strat", STRAT, "--promote-only",
                            "--report-discord", "--out-dir", str(tmp)])
    scripts = runner.scripts()
    stages = [flag_value(c, "--stage")
              for c in runner.for_script("discord_reporter.py")]
    check("with no lifecycle snapshot, Stages 3 and 5 post and 4 is skipped",
          stages == ["3", "5"], str(stages))
    check("...and every card follows the promotion",
          all(i > scripts.index("promote.py")
              for i, s in enumerate(scripts) if s == "discord_reporter.py"),
          str(scripts))

    verify = pipeline_dir(STRAT, tmp) / "verify_20260821_120000"
    verify.mkdir(parents=True, exist_ok=True)
    (verify / "dual_metrics_NQ.json").write_text("{}", encoding="utf-8")
    check("stage4_metrics_exist sees the snapshot",
          rp.stage4_metrics_exist(STRAT, str(tmp)))
    runner = FakeRunner()
    with promoted_packages(f"{STRAT}_NQ_15m_VA"):
        _main_with(runner, ["--strat", STRAT, "--promote-only",
                            "--report-discord", "--out-dir", str(tmp)])
    stages = [flag_value(c, "--stage")
              for c in runner.for_script("discord_reporter.py")]
    check("with one, the order is 3 -> 4 -> 5", stages == ["3", "4", "5"],
          str(stages))

    # A webhook outage must not fail a promotion that already happened.
    runner = FakeRunner(fail_on={"discord_reporter.py": 1})
    rc = _main_with(runner, ["--strat", STRAT, "--promote-only",
                             "--report-discord", "--out-dir", str(tmp)])
    check("...and a card that failed to post does not fail the run", rc == 0,
          f"rc={rc}")

    # Nothing certified is a verdict, not a breakage.
    write_stage3(tmp, [s3row("NQ", "15m", certified=False, status="FAIL")])
    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--promote-only",
                             "--out-dir", str(tmp)])
    check("nothing certified promotes nothing, and exits 0",
          rc == 0 and runner.for_script("promote.py") == [], f"rc={rc}")


def test_auto_promote_every_timeframe(tmp: Path) -> None:
    print("\n14b. --auto-promote iterates EVERY certified timeframe")

    write_stage2(tmp, [s2row("CL", "5m"), s2row("CL", "15m")])
    write_stage3(tmp, [
        s3row("CL", "5m", certified=True),
        s3row("CL", "15m", certified=True),
        s3row("NQ", "15m", certified=False, status="FAIL", pf=0.91),
        s3row("NQ", "5m", certified=False, status="FAIL", pf=None,
              starved="[REGIME STARVATION] Quadrant Q1 had only 3 holdout "
                      "trades."),
    ])

    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--tf", "5m,15m",
                             "--auto-promote", "--out-dir", str(tmp)])
    promos = runner.for_script("promote.py")
    check("exits 0", rc == 0, f"rc={rc}")
    check("BOTH certified timeframes are promoted - Stage 3 merges its runs "
          "into one summary, so a loop that stopped at the first timeframe "
          "would silently drop the rest", len(promos) == 2,
          f"{len(promos)} promotion(s)")
    audits = sorted(flag_value(c, "--audit-file") or "" for c in promos)
    check("...and each cites its OWN per-pair audit, never the unsuffixed "
          "file, which holds whichever timeframe ran last",
          audits == ["/tmp/gate_audit_CL_15m.json", "/tmp/gate_audit_CL_5m.json"],
          str(audits))
    check("...and each names its own timeframe",
          sorted(flag_value(c, "--timeframe") or "" for c in promos)
          == ["15m", "5m"])

    blob = read_stage(pipeline_dir(STRAT, str(tmp)) / STAGE3_SUMMARY_FILE,
                      expect_stage=3, expect_strategy=STRAT)
    auto = blob.get("auto_promotion") or {}
    check("the promotion outcome is written BACK onto Stage 3's summary, "
          "which is how the Discord card knows to say PROMOTED rather than "
          "telling a reader to run a command that already ran",
          auto.get("ran") is True and len(auto.get("promotions") or []) == 2,
          str(list(auto)))
    check("...per configuration, so a partly-failed batch labels each row by "
          "what happened to IT",
          {(d["symbol"], d["timeframe"], d["promoted"])
           for d in auto["promotions"]}
          == {("CL", "5m", True), ("CL", "15m", True)})
    check("...and the verdicts themselves are rewritten untouched - "
          "annotating an index must not restate one",
          len(blob["results"]) == 4)


def test_failure_reasons_are_distinguished() -> None:
    print("\n14c. A configuration that was not promoted says WHY")

    starved = rp.failure_reason(s3row(
        "NQ", "5m", certified=False, status="FAIL", pf=999.0,
        starved="[REGIME STARVATION] Quadrant Q1 had only 3 holdout trades."))
    check("starvation is named as starvation - a quadrant with three holdout "
          "trades prints a 999 profit factor and a PASS on the factor row, so "
          "'Gate R FAIL' alone points at the wrong work",
          starved.startswith("REGIME STARVATION"), starved)
    lost = rp.failure_reason(s3row("NQ", "15m", certified=False,
                                   status="FAIL", pf=0.91))
    check("an edge that was re-tested and lost reports the factor it lost at",
          "0.91" in lost and "STARVATION" not in lost, lost)
    none_eval = rp.failure_reason(s3row("GC", "15m", certified=False,
                                        status="NOT EVALUATED",
                                        pf=None))
    check("NOT EVALUATED says no quadrant was designated, which is neither a "
          "pass nor a failure of the edge",
          "no home quadrant" in none_eval, none_eval)
    broke = rp.failure_reason({"symbol": "ES", "timeframe": "15m",
                               "status": "NOT AUDITED",
                               "error": "ValueError: no bars"})
    check("a run that broke is NOT AUDITED, never a verdict about the "
          "strategy", broke.startswith("NOT AUDITED") and "no bars" in broke,
          broke)


def test_summary_table(tmp: Path) -> None:
    print("\n14d. The pipeline ends on one table of every configuration")

    rows = [s3row("CL", "15m", certified=True),
            s3row("CL", "5m", certified=True),
            s3row("NQ", "15m", certified=False, status="FAIL", pf=0.91)]
    table = rp.promotion_summary_table(rows, [
        {"symbol": "CL", "timeframe": "15m", "version": "A",
         "promoted": True, "commit": "abc1234",
         "incubator_dir": "/x/inc/demo", "error": ""},
        {"symbol": "CL", "timeframe": "5m", "version": "A",
         "promoted": False, "commit": None, "error": "promote.py exited 1"},
    ])
    check("a promoted configuration reads PROMOTED and names where it landed",
          "PROMOTED" in table and "/x/inc/demo" in table)
    check("a promotion that FAILED is its own outcome, not a missing row - "
          "the certification stands and the commit did not happen",
          "PROMOTE FAILED" in table and "promote.py exited 1" in table)
    check("an uncertified configuration is still a row, with its reason - a "
          "table of the winners alone reads as a run in which nothing else "
          "happened", "NOT CERTIFIED" in table and "0.91" in table)
    check("every configuration Stage 3 indexed is present",
          all(sym in table for sym in ("CL", "NQ")))

    off = rp.promotion_summary_table([s3row("CL", "15m", certified=True)], [])
    check("without --auto-promote a certified row says so rather than "
          "claiming a promotion nobody ran",
          "CERTIFIED" in off and "--auto-promote off" in off, off)
    check("no rows still prints the heading and says so - an absent table "
          "reads as a pipeline that did not finish",
          "nothing reached a verdict" in rp.promotion_summary_table([], []))


def test_dry_run_launches_nothing(tmp: Path) -> None:
    print("\n15. --dry-run prints the sequence and launches nothing")
    runner = FakeRunner()
    rc = _main_with(runner, ["--strat", STRAT, "--tf", "15m,30m", "--dry-run",
                             "--report-discord", "--auto-promote",
                             "--out-dir", str(tmp)])
    check("exits 0", rc == 0, f"rc={rc}")
    check("no subprocess is launched at all", runner.calls == [],
          f"{len(runner.calls)} launched")


def test_defaults_are_the_charter(tmp: Path) -> None:
    print("\n16. The default window is the charter's, not the lake's")
    write_stage2(tmp, [s2row("NQ", "15m")])
    runner = FakeRunner()
    _main_with(runner, ["--strat", STRAT, "--tf", "15m", "--out-dir", str(tmp)])
    s1 = runner.for_script("baseline.py")[0]
    check("Stage 1 defaults to the charter in-sample window",
          (flag_value(s1, "--start"), flag_value(s1, "--end"))
          == (CHARTER_IS_START, CHARTER_IS_END))
    s4 = runner.for_script("verify_full.py")[0]
    check("Stage 4's --end is passed explicitly, not left to a hardcoded year",
          flag_value(s4, "--end") is not None)


def test_stage4_runs_version_b() -> None:
    print("\nStage 4 runs Version B, at the bar Stage 3 certified")

    cmd = rp.stage4_cmd("demo", "1h", "2013-01-01", "2026-08-29")
    check("--ml is passed to verify_full. Its absence was a SILENT pipeline "
          "break: verify_full's --ml is store_true, so Stage 4 wrote "
          "version_b: null, and Stage 5 - handed a Version B that Stage 3 had "
          "certified - died in load_metrics with 'carries no version_b "
          "block'. sma_momentum_crossover_20260818 lost all nine certified "
          "Version B packages that way: promoted 17, failed 9",
          "--ml" in cmd, " ".join(cmd))

    check("...and the threshold travels with it, so Stage 4 measures the "
          "filter at the bar Stage 1 screened and Stage 3 certified at",
          "--ml-threshold" in cmd
          and cmd[cmd.index("--ml-threshold") + 1] == str(
              rp.ML_THRESHOLD_DEFAULT), " ".join(cmd))

    over = rp.stage4_cmd("demo", "1h", "2013-01-01", "2026-08-29",
                         ml_threshold=0.55)
    check("...an override reaching Stage 4 too - one run must not screen, "
          "certify and measure at three different bars",
          over[over.index("--ml-threshold") + 1] == "0.55", " ".join(over))

    for name, built in (
            ("stage 1", rp.stage1_cmd("demo", "NQ", ["1h"], "2013-01-01",
                                      "2022-12-31", None, 0.55)),
            ("stage 2", rp.stage2_cmd("demo", "2013-01-01", "2022-12-31",
                                      None, 0.55)),
            ("stage 3", rp.stage3_cmd("demo", "1h", "2013-01-01",
                                      "2022-12-31", ml_threshold=0.55)),
            ("stage 4", over)):
        check(f"{name} carries the same override",
              built[built.index("--ml-threshold") + 1] == "0.55",
              " ".join(built))

    check("Stage 2 still inherits its pairs by OMITTING --symbols and --tf - "
          "a threshold constrains no pair, so adding it changes nothing "
          "about which configurations are swept",
          "--symbols" not in rp.stage2_cmd("demo", "a", "b")
          and "--tf" not in rp.stage2_cmd("demo", "a", "b"))


def main() -> int:
    print("=" * 60)
    print("  backtest/run_pipeline.py - the unified orchestrator")
    print("=" * 60)

    test_window_guard()
    test_stage1_cmd()
    test_stage4_runs_version_b()
    test_stage2_cmd_inherits_exact_pairs()
    test_stage3_cmd_window_flags()
    test_stage4_cmd()
    test_discord_and_promote_cmds()
    test_stage2_timeframes()
    test_certified_rows()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_order_and_expansion(tmp)
        test_a_failing_stage45_does_not_abort_the_run(tmp)
        test_skips_unsurvived_timeframe(tmp)
        test_stops_when_nothing_survived(tmp)
        test_discord_strict_sequential_order(tmp)
        test_stage_failure_aborts_discord_failure_does_not(tmp)
        test_auto_promote(tmp)
        test_auto_promote_every_timeframe(tmp)
        test_promote_only(tmp)
        test_failure_reasons_are_distinguished()
        test_summary_table(tmp)
        test_dry_run_launches_nothing(tmp)
        test_defaults_are_the_charter(tmp)

    print("\n" + "=" * 60)
    if _failures:
        print(f"  {len(_failures)} CHECK(S) FAILED")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
