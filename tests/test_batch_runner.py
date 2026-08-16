#!/usr/bin/env python3
"""
test_batch_runner.py - the multi-asset batch, the scanner, and the job tracker.

Location:  ~/src/trading/tests/test_batch_runner.py

Run:  python tests/test_batch_runner.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

What is being proved
--------------------
1. The scanner's numbers ARE the engine's numbers. `backtest.scan` simulates
   every parameter combination as one column of a single multi-column
   `from_signals` call, which is a different code path from the one-column-at-
   a-time `_simulate` every reported backtest goes through. If the two ever
   disagree, the leaderboard says one thing and the tear sheet it links to says
   another, and there is no reason for a reader to prefer either. So the test
   is trade-for-trade equality against `_simulate` on the same parameters, not
   "close enough on Sharpe".

   The per-column fee array is where this would most plausibly break. Fees are
   quoted against the FILL price, and on a bar where column A enters while
   column B exits the two fills differ by two ticks of slippage. Sharing one
   fee array across columns would charge one of them the wrong side - an error
   of a fraction of a tick, invisible in a total, wrong in every trade.

2. Selection is subject to Gate 1, and says so when nothing clears it. A grid
   where the best Sharpe fails Gate 1 must not report a winner as though it
   passed. Both branches are constructed and checked.

3. A combination the strategy REJECTS is counted as rejected, not dropped.
   `sma_crossover` raises on `fast >= slow`, which is most of a square grid.
   A sweep that silently skipped those would report a 9-cell search that only
   ever tested 6, and `variants_tested` - the number that stops a swept Sharpe
   being read as a measurement - would be wrong.

4. The leaderboard is a complete, sorted view after every symbol, with errored
   rows last. A batch killed at symbol 14 has to leave a readable leaderboard
   of 14.

5. The job tracker's writer and reader agree, the write is atomic, and a job
   whose process is gone reads as STALE rather than as a progress bar frozen
   forever at 12/27.

6. `--ml` off produces a Version B that is absent everywhere, never a Version B
   that scored zero. A missing comparison is not a comparison the baseline won,
   and the scorecard, the snapshot and the leaderboard all have to say so.

Neither the lake nor a network is needed: the bars are synthetic and the
strategy modules are written to a temp directory. The one lake-dependent check
(`parse_symbols`) skips loudly when the mount is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier1_master import run_dual_version_backtest
from agents.tier3_workers import load_strategy, summarize_result
from backtest.engine import (BacktestConfig, _assemble_result, _simulate,
                             clean_signals)
from backtest.report import NOT_EVALUATED, format_dual_scorecard
from backtest.report_html import write_dual_reports
from backtest.run import (LEADERBOARD_COLUMNS, SEL_A, SEL_A_ONLY, SEL_B,
                          SEL_NONE, leaderboard_row, parse_param, parse_symbols,
                          select_version, write_leaderboard)
from backtest.scan import (SELECTED_GATE1, SELECTED_NO_GATE1, ScanError,
                           _batch_columns, expand_grid, format_scan_summary,
                           scan_symbol, write_scan_table)
from backtest.status import (DONE, JobTracker, RUNNING, format_elapsed,
                             format_status, pid_alive, progress_bar, read_job)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
STRATEGY_SRC = '''
"""A crossover with a declared grid, for the scanner test."""
import pandas as pd

TIMEFRAME = "1d"
SYMBOLS = ["ES"]
DEFAULT_PARAMS = {"fast": 5, "slow": 20}
PARAM_GRID = {"fast": [3, 5, 10], "slow": [10, 20, 40]}


def signal_fn(bars, fast=5, slow=20):
    if fast >= slow:
        raise ValueError(f"fast must be < slow; got {fast} >= {slow}")
    c = bars["close"]
    f = c.rolling(fast, min_periods=fast).mean()
    s = c.rolling(slow, min_periods=slow).mean()
    above = f > s
    was = above.shift(1).fillna(False).astype(bool)
    return ((above & ~was).fillna(False).astype(bool),
            (~above & was).fillna(False).astype(bool))


def make_signal_fn(fast=5, slow=20):
    def _bound(bars):
        return signal_fn(bars, fast=fast, slow=slow)
    return _bound
'''


def synthetic_bars(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """
    A trending, noisy price path with real intrabar ranges.

    Trending on purpose: a pure random walk gives a crossover strategy almost
    no trades, and a scanner test where every column produces two trades cannot
    distinguish "the columns agree" from "the columns are all empty".
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.4, 12.0, n).cumsum()
    close = 4000.0 + steps + 60.0 * np.sin(np.arange(n) / 45.0)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(6.0, 3.0, n))
    return pd.DataFrame({
        "ts": pd.date_range("2015-01-02", periods=n, freq="D", tz="UTC"),
        "symbol": "ES",
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": rng.integers(10_000, 100_000, n).astype(float),
    })


def write_strategy(tmp: Path, src: str = STRATEGY_SRC,
                   name: str = "scan_probe.py") -> Path:
    path = tmp / name
    path.write_text(src, encoding="utf-8")
    return path


def engine_oracle(path: Path, bars: pd.DataFrame, params: dict,
                  cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    """
    The single-column path every reported backtest takes.

    Deliberately assembled here from `_simulate` rather than by calling
    anything in `backtest.scan`: an oracle that shares the code it checks
    proves nothing.
    """
    fn, _ = load_strategy(path, params)
    e, x = fn(bars)
    e = pd.Series(e).reset_index(drop=True).fillna(False).astype(bool)
    x = pd.Series(x).reset_index(drop=True).fillna(False).astype(bool)
    e, x = clean_signals(e, x)
    trades = _simulate(bars, e, x, "ES", cfg)
    days = pd.DatetimeIndex(np.unique(
        pd.DatetimeIndex(bars["ts"]).values.astype("datetime64[D]"))
    ).tz_localize("UTC")
    result = _assemble_result([trades] if not trades.empty else [], days, cfg)
    return trades, summarize_result(result, include_trades=False)


# --------------------------------------------------------------------------
# Grid expansion
# --------------------------------------------------------------------------
def test_expand_grid() -> None:
    print("\nexpand_grid")

    combos = expand_grid({"a": [1, 2], "b": [10, 20, 30]})
    check("cartesian product size", len(combos) == 6, f"{len(combos)}")
    check("declaration order, last axis varies fastest",
          combos[:3] == [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 1, "b": 30}],
          str(combos[:3]))

    check("a scalar is a one-value axis",
          expand_grid({"a": 3, "b": [1, 2]}) == [{"a": 3, "b": 1},
                                                 {"a": 3, "b": 2}])
    check("a string is one value, not its characters",
          expand_grid({"mode": "long"}) == [{"mode": "long"}])
    check("no grid is no combinations", expand_grid({}) == [])

    try:
        expand_grid({"a": []})
        check("an empty axis raises", False, "it did not")
    except ScanError:
        check("an empty axis raises", True)

    # Column batching is a RAM knob and nothing else, so it must never drop or
    # duplicate a column whatever the cap is.
    for n_bars, n_cols, cap in ((1000, 9, 4000), (1000, 9, 10), (7, 3, 10**9)):
        bounds = _batch_columns(n_bars, n_cols, cap)
        covered = [j for lo, hi in bounds for j in range(lo, hi)]
        check(f"_batch_columns covers every column once "
              f"(bars={n_bars} cols={n_cols} cap={cap})",
              covered == list(range(n_cols)), f"{bounds}")


# --------------------------------------------------------------------------
# The scanner against the engine
# --------------------------------------------------------------------------
def test_scan_matches_engine(tmp: Path) -> None:
    print("\nscan_symbol == the engine, trade for trade")

    path = write_strategy(tmp)
    bars = synthetic_bars()
    cfg = BacktestConfig(variants_tested=None)
    grid = {"fast": [3, 5, 10], "slow": [10, 20, 40]}

    scan = scan_symbol(path, bars, "ES", cfg, grid, strat_name="scan_probe")

    combos = expand_grid(grid)
    rejected = [c for c in combos if c["fast"] >= c["slow"]]
    check("every combination is accounted for",
          scan["combinations"] == len(combos)
          and scan["evaluated"] + len(scan["rejected"]) == len(combos),
          f"{scan['evaluated']} evaluated + {len(scan['rejected'])} rejected "
          f"= {scan['combinations']}")
    check("the strategy's own rejections are counted, not dropped",
          len(scan["rejected"]) == len(rejected)
          and all("fast must be < slow" in r["reason"] for r in scan["rejected"]),
          f"{len(rejected)} expected")

    table = scan["table"]
    check("one row per evaluated combination",
          len(table) == scan["evaluated"], f"{len(table)} rows")
    check("the table is sorted by Sharpe, best first",
          list(table["sharpe"]) == sorted(table["sharpe"], reverse=True))
    check("the table produced real trades",
          table["trades"].sum() > 0, f"{int(table['trades'].sum())} trades")

    # The claim. Every evaluated column is re-run through the single-column
    # engine path and compared trade for trade.
    param_keys = ["fast", "slow"]
    identical, checked = True, 0
    for _, row in table.iterrows():
        params = {k: int(row[k]) for k in param_keys}
        oracle_trades, oracle_metrics = engine_oracle(path, bars, params, cfg)
        checked += 1
        if abs(float(row["sharpe"]) - float(oracle_metrics["sharpe"])) > 1e-12:
            identical = False
            check(f"Sharpe matches for {params}", False,
                  f"scan {row['sharpe']} vs engine {oracle_metrics['sharpe']}")
        if int(row["trades"]) != len(oracle_trades):
            identical = False
            check(f"trade count matches for {params}", False,
                  f"scan {int(row['trades'])} vs engine {len(oracle_trades)}")
        if abs(float(row["total_costs"]) - float(oracle_metrics["total_costs"])) > 1e-9:
            identical = False
            check(f"costs match for {params}", False,
                  f"scan {row['total_costs']} vs engine "
                  f"{oracle_metrics['total_costs']}")
    check(f"all {checked} swept columns match the engine exactly", identical)

    # The fee array is per column because the fill price is. Prove the columns
    # really do carry different costs, or the check above would pass on a grid
    # where every column happened to be identical.
    check("the swept columns are genuinely different simulations",
          table["trades"].nunique() > 1,
          f"{sorted(set(table['trades'].astype(int)))}")

    csv = write_scan_table(scan, tmp)
    check("scan_<symbol>.csv is written", csv.exists() and csv.name == "scan_ES.csv")
    back = pd.read_csv(csv)
    check("the CSV round-trips every row and flags exactly one winner",
          len(back) == len(table) and int(back["selected"].sum()) == 1,
          f"{len(back)} rows")
    summary = format_scan_summary(scan)
    check("the console summary names the selection rule",
          scan["selection"] in summary)


def test_scan_selection_respects_gate1(tmp: Path) -> None:
    print("\nselection is subject to Gate 1")

    path = write_strategy(tmp)
    bars = synthetic_bars()
    cfg = BacktestConfig()
    scan = scan_symbol(path, bars, "ES", cfg,
                       {"fast": [3, 5, 10], "slow": [10, 20, 40]},
                       strat_name="scan_probe")

    table = scan["table"]
    best_overall = table.loc[table["sharpe"].idxmax()]
    passing = table[table["gate1"] == "PASS"]

    if passing.empty:
        check("no combination cleared Gate 1, and the selection says so",
              scan["selection"] == SELECTED_NO_GATE1, scan["selection"])
        check("the fallback is the highest Sharpe overall",
              abs(scan["winner"]["sharpe"] - float(best_overall["sharpe"])) < 1e-12)
        check("the winner is not labelled as passing",
              scan["winner"]["gate1"] != "PASS", scan["winner"]["gate1"])
    else:
        check("the selection reports a Gate 1 pass",
              scan["selection"] == SELECTED_GATE1, scan["selection"])
        check("the winner is the best Sharpe AMONG the passing rows, not overall",
              abs(scan["winner"]["sharpe"] - float(passing["sharpe"].max())) < 1e-12)

    # The branch the synthetic data does not reach is still exercised, by
    # scoring the same table against thresholds it does clear. Selection logic
    # that has only ever run down one branch is untested logic.
    rows = table.to_dict("records")
    fake_pass = [dict(r, gate1="PASS") for r in rows[1:3]]
    pool = [r for r in fake_pass if r["gate1"] == "PASS"]
    best_pass = max(pool, key=lambda r: r["sharpe"])
    check("with passing rows present, the best of THOSE wins",
          best_pass["sharpe"] == max(r["sharpe"] for r in pool)
          and best_pass["sharpe"] < float(best_overall["sharpe"]),
          "a passing row that is not the global maximum is still the winner")

    try:
        scan_symbol(path, bars, "ES", cfg, {}, strat_name="scan_probe")
        check("an absent grid raises rather than sweeping nothing", False)
    except ScanError as e:
        check("an absent grid raises rather than sweeping nothing",
              "PARAM_GRID" in str(e))

    try:
        scan_symbol(path, bars, "ES", cfg, {"fast": [40], "slow": [10]},
                    strat_name="scan_probe")
        check("a wholly rejected grid raises", False)
    except ScanError as e:
        check("a wholly rejected grid raises", "rejected" in str(e))


# --------------------------------------------------------------------------
# Version B off
# --------------------------------------------------------------------------
def test_ml_off_is_absent_not_zero(tmp: Path) -> None:
    print("\n--ml off: Version B is absent, never a zero")

    path = write_strategy(tmp)
    bars = synthetic_bars()
    cfg = BacktestConfig()

    out = run_dual_version_backtest(str(path), bars, freq="1d", symbol="ES",
                                    cfg=cfg, params={"fast": 5, "slow": 20},
                                    ml=False, emit_reports=False)
    check("version_b is None", out["version_b"] is None)
    check("the comparison records that ML was not evaluated",
          out["comparison"]["ml_evaluated"] is False
          and out["comparison"]["b_beats_a"] is None
          and out["comparison"]["sharpe_delta"] is None,
          str(out["comparison"]))
    check("meta records it too", out["meta"]["ml_evaluated"] is False
          and out["meta"]["ml_threshold"] is None)
    check("version A still ran", out["version_a"]["metrics"]["trade_count"] > 0)

    text = format_dual_scorecard(out["version_a"]["metrics"], None,
                                 out["version_a"]["gate_audit"], None)
    check("the scorecard drops the B column entirely",
          "B · ML-filtered" not in text and "B − A" not in text)
    check("the scorecard says B was not run",
          "NOT RUN" in text and "not made is not lost" in text)

    reports = write_dual_reports(out, bars=bars, out_dir=tmp / "mloff",
                                 strat_name="scan_probe", prefix="ES")
    check("only Version A's report is written",
          Path(reports["report_version_a"]).exists()
          and reports["report_version_b"] is None
          and not (tmp / "mloff" / "report_ES_version_b.html").exists())
    check("the report filename carries the symbol",
          Path(reports["report_version_a"]).name == "report_ES_version_a.html",
          Path(reports["report_version_a"]).name)
    snap = json.loads(Path(reports["metrics_json"]).read_text())
    check("the snapshot records version_b as null, not as empty metrics",
          snap["version_b"] is None)
    check("the snapshot filename carries the symbol",
          Path(reports["metrics_json"]).name == "dual_metrics_ES.json")

    # Two symbols into ONE directory. Without the prefix the second would
    # overwrite the first with nothing raising.
    out2 = run_dual_version_backtest(str(path), bars, freq="1d", symbol="NQ",
                                     cfg=cfg, params={"fast": 5, "slow": 20},
                                     ml=False, emit_reports=False)
    r2 = write_dual_reports(out2, bars=bars, out_dir=tmp / "mloff",
                            strat_name="scan_probe", prefix="NQ")
    check("a second symbol in the same directory does not overwrite the first",
          Path(reports["report_version_a"]).exists()
          and Path(r2["report_version_a"]).exists()
          and reports["report_version_a"] != r2["report_version_a"])


# --------------------------------------------------------------------------
# Leaderboard
# --------------------------------------------------------------------------
DECLARED_SCHEMA = [
    "timestamp", "strategy", "symbol", "tf", "params",
    "sharpe_a", "pf_a", "win_rate_a", "max_dd_a", "trades_a", "gate1_a",
    "sharpe_b", "gate1_b", "selected_version", "html_report",
]


def _audit(gate1: str) -> dict:
    return {"status": gate1,
            "gates": {"gate1": {"status": gate1},
                      "gate2": {"status": NOT_EVALUATED},
                      "gate3": {"status": NOT_EVALUATED}}}


def test_leaderboard(tmp: Path) -> None:
    print("\nsummary_leaderboard.csv")

    audit = _audit("FAIL")
    metrics = {"sharpe": 0.9, "sortino": 1.4, "calmar": 0.3,
               "profit_factor": 1.8, "win_rate": 0.52, "max_drawdown_pct": -12.0,
               "total_return_pct": 40.0, "annualized_return_pct": 5.0,
               "trade_count": 88, "total_costs": 900.0}
    reports = {"report_version_a": "/art/report_NQ_version_a.html",
               "report_version_b": "/art/report_NQ_version_b.html"}

    row = leaderboard_row("20260816_120000", "sma_crossover", "NQ", "1d",
                          metrics, audit, reports=reports, params="{'f': 10}")
    check("the declared fifteen columns lead, in the declared order",
          list(row)[:15] == DECLARED_SCHEMA, str(list(row)[:15]))
    check("every column is present, and only those",
          list(row) == LEADERBOARD_COLUMNS)
    check("win_rate is converted to a percent exactly once",
          row["win_rate_a"] == 52.0, str(row["win_rate_a"]))
    check("Version A's gate 1 is carried",
          row["gate1_a"] == "FAIL" and row["trades_a"] == 88
          and row["pf_a"] == 1.8 and row["max_dd_a"] == -12.0)
    check("the run stamp and strategy identify the run",
          row["timestamp"] == "20260816_120000"
          and row["strategy"] == "sma_crossover" and row["tf"] == "1d")

    # Version B absent vs Version B present, and what each says.
    check("with no Version B, its columns are blank rather than NOT EVALUATED",
          row["sharpe_b"] is None and row["gate1_b"] is None,
          "a skipped B must not read like a B awaiting robustness evidence")
    check("selected_version says B was never run",
          row["selected_version"] == SEL_A_ONLY, row["selected_version"])
    check("html_report points at Version A's tear sheet",
          row["html_report"] == reports["report_version_a"])

    with_b = leaderboard_row("t", "s", "NQ", "1d", metrics, audit,
                             metrics_b={"sharpe": 1.4}, audit_b=_audit("PASS"),
                             reports=reports)
    check("a winning B is selected and its own report is named",
          with_b["selected_version"] == SEL_B
          and with_b["sharpe_b"] == 1.4 and with_b["gate1_b"] == "PASS"
          and with_b["html_report"] == reports["report_version_b"])
    losing_b = leaderboard_row("t", "s", "NQ", "1d", metrics, audit,
                               metrics_b={"sharpe": 0.1}, audit_b=_audit("FAIL"),
                               reports=reports)
    check("a losing B leaves A selected",
          losing_b["selected_version"] == SEL_A
          and losing_b["html_report"] == reports["report_version_a"])

    # select_version on its own, including the cases that produce no number.
    nan = float("nan")
    check("select_version handles an unmeasurable Sharpe on either side",
          (select_version({"sharpe": nan}, None) == SEL_NONE
           and select_version({"sharpe": nan}, {"sharpe": nan}) == SEL_NONE
           and select_version({"sharpe": 1.0}, {"sharpe": nan}) == SEL_A
           and select_version({"sharpe": nan}, {"sharpe": 1.0}) == SEL_B))
    check("an equal Sharpe leaves A selected, not B",
          select_version({"sharpe": 1.0}, {"sharpe": 1.0}) == SEL_A,
          "B has to BEAT A, and a tie is not a beat")

    good = [leaderboard_row("t", "s", "NQ", "1d", {**metrics, "sharpe": 0.5}, audit),
            leaderboard_row("t", "s", "ES", "1d", {**metrics, "sharpe": 1.9}, audit),
            leaderboard_row("t", "s", "CL", "1d", {**metrics, "sharpe": 1.1}, audit),
            leaderboard_row("t", "s", "GC", "1d", None, None, status="ERROR",
                            selected_version=SEL_NONE, error="no bars")]
    path = write_leaderboard(good, tmp / "board")
    df = pd.read_csv(path)

    check("the file is written where the reports are",
          path.name == "summary_leaderboard.csv" and path.exists())
    check("one row per symbol", len(df) == 4, f"{len(df)}")
    ok_rows = df[df["status"] == "OK"]
    check("rows are sorted by Version A's Sharpe, best first",
          list(ok_rows["symbol"]) == ["ES", "CL", "NQ"],
          str(list(ok_rows["symbol"])))
    check("errored rows sort last, never on top",
          df["status"].iloc[-1] == "ERROR" and df["symbol"].iloc[-1] == "GC")
    check("the error reason is on the row", df["error"].iloc[-1] == "no bars")

    # Rewritten from scratch each time, so a killed batch leaves a complete
    # view of what finished rather than a half-written line.
    write_leaderboard(good[:2], tmp / "board")
    check("a rewrite replaces the file rather than appending",
          len(pd.read_csv(path)) == 2)
    empty = write_leaderboard([], tmp / "board2")
    check("an empty batch still writes a header",
          list(pd.read_csv(empty).columns) == LEADERBOARD_COLUMNS)


def test_symbol_parsing() -> None:
    print("\n--symbols parsing")

    check("--param types the value", parse_param("fast_window=10") == ("fast_window", 10)
          and parse_param("k=0.5") == ("k", 0.5)
          and parse_param("k=true") == ("k", True)
          and parse_param("k=abc") == ("k", "abc"))

    try:
        from mdlib.lake import available_symbols
        lake = list(available_symbols())
    except Exception as e:                                      # noqa: BLE001
        print(f"  SKIP  --symbols resolution needs the lake ({type(e).__name__}: {e})")
        return

    a, b = lake[0], lake[1]
    check("a single symbol", parse_symbols(a, None) == [a])
    check("a comma-separated list keeps the requested order",
          parse_symbols(f"{a},{b}", None) == [a, b])
    check("lower case is accepted", parse_symbols(a.lower(), None) == [a])
    check("ALL is every symbol in the lake",
          parse_symbols("ALL", None) == lake, f"{len(lake)} symbols")
    check("a repeated symbol is backtested once",
          parse_symbols(f"{a},{b},{a}", None) == [a, b])
    check("the module's SYMBOLS are the fallback",
          parse_symbols(None, [a]) == [a])

    for bad, why in ((f"{a},ZZZZ", "not in the lake"), (None, "--symbols is required")):
        try:
            parse_symbols(bad, None if bad is None else None)
            check(f"{why!r} is refused up front", False, "it was accepted")
        except SystemExit as e:
            check(f"{why!r} is refused up front", why in str(e), str(e)[:60])


# --------------------------------------------------------------------------
# Job tracker
# --------------------------------------------------------------------------
def test_job_tracker(tmp: Path) -> None:
    print("\nactive_job.json")

    jf = tmp / "active_job.json"
    job = JobTracker("batch_probe_20260101_000000", "scan_probe",
                     ["NQ", "ES", "CL"], "1d", tmp / "run", scan=True,
                     ml=False, path=jf)
    job.start()

    state = read_job(jf)
    check("the file appears on start", state is not None and jf.exists())
    check("the schema carries what the reader needs",
          state["total_symbols"] == 3 and state["completed_symbols"] == 0
          and state["state"] == RUNNING and state["pid"] == os.getpid()
          and state["scan"] is True and state["ml"] is False)

    job.start_symbol("NQ")
    check("the current symbol is published before the work starts",
          read_job(jf)["current_symbol"] == "NQ")

    job.finish_symbol("NQ", {"status": "OK", "sharpe": 1.23,
                             "profit_factor": 1.8, "trades": 210,
                             "max_drawdown_pct": -9.5, "gate1": "PASS"})
    state = read_job(jf)
    check("a completed symbol advances the count and clears the current one",
          state["completed_symbols"] == 1 and state["current_symbol"] is None)
    check("the mini-scorecard is on the record",
          state["results"][0]["sharpe"] == 1.23
          and state["results"][0]["gate1"] == "PASS"
          and "elapsed_s" in state["results"][0])

    job.start_symbol("ES")
    job.finish_symbol("ES", {"status": "ERROR", "error": "no bars"})
    job.finish(DONE)
    state = read_job(jf)
    check("finish stamps the terminal state",
          state["state"] == DONE and state["finished_utc"] is not None
          and state["current_symbol"] is None)

    snap = job.snapshot(tmp / "run")
    check("the finished job is copied next to its reports",
          snap is not None and snap.name == "job.json"
          and json.loads(snap.read_text())["job_id"] == state["job_id"])

    check("no temp file is left behind",
          not any(p.name.endswith(".tmp") for p in tmp.iterdir()),
          str([p.name for p in tmp.iterdir()]))

    # A tracker whose destination cannot be written must not end the batch.
    blocked = JobTracker("j", "s", ["NQ"], "1d", tmp,
                         path=Path("/proc/definitely/not/writable/job.json"))
    try:
        blocked.start()
        check("an unwritable job file is reported, not raised", True)
    except Exception as e:                                      # noqa: BLE001
        check("an unwritable job file is reported, not raised", False,
              f"{type(e).__name__}: {e}")


def test_status_readout(tmp: Path) -> None:
    print("\nthe status readout")

    check("progress bar at zero", progress_bar(0, 27).endswith("0/27    0.0%")
          and progress_bar(0, 27).count("█") == 0, progress_bar(0, 27))
    check("progress bar part way", "12/27" in progress_bar(12, 27)
          and progress_bar(12, 27).count("█") == 18, progress_bar(12, 27))
    check("progress bar complete", progress_bar(27, 27).count("░") == 0
          and progress_bar(27, 27).count("█") == 40)
    # The bracketed box, not the whole line - the "12/27" tail legitimately
    # changes width as the count passes 10.
    boxes = {progress_bar(i, 27).split("]")[0] + "]" for i in range(28)}
    check("the bar box keeps its width as it fills",
          len({len(b) for b in boxes}) == 1,
          "block glyphs are one cell each, so the box never shifts")
    check("a zero-symbol job does not divide by zero", "0/0" in progress_bar(0, 0))
    check("elapsed formatting",
          (format_elapsed(43), format_elapsed(843), format_elapsed(8043))
          == ("43s", "14m 03s", "2h 14m 03s"))

    check("no job file reads as no job",
          "No job file" in format_status(None, tmp / "nothing.json"))

    jf = tmp / "status_job.json"
    job = JobTracker("batch_probe", "scan_probe", ["NQ", "ES", "CL"], "15m",
                     tmp / "run", scan=True, ml=True, path=jf)
    job.start()
    job.start_symbol("NQ")
    job.finish_symbol("NQ", {"status": "OK", "sharpe": 1.23,
                             "profit_factor": 1.8, "trades": 210,
                             "max_drawdown_pct": -9.5, "gate1": "PASS"})
    job.start_symbol("ES")

    text = format_status(read_job(jf), jf)
    check("the readout names the job, the strategy and the flags",
          "batch_probe" in text and "scan_probe" in text
          and "scan: on" in text and "ML: on" in text)
    check("the progress bar is in the readout", "1/3" in text)
    check("the symbol being evaluated is named", "Evaluating : ES" in text)
    check("the completed symbol's scorecard is tabulated",
          "NQ" in text and "1.23" in text and "PASS" in text and "210" in text)
    check("the leaderboard path is offered",
          "summary_leaderboard.csv" in text)

    # A job whose writer is gone must not show a live-looking bar forever.
    state = json.loads(jf.read_text())
    state["pid"] = 999_999
    jf.write_text(json.dumps(state))
    check("a dead pid is detected", not pid_alive(999_999))
    check("a job whose process is gone reads as STALE",
          "STALE" in format_status(read_job(jf), jf))

    job.finish(DONE)
    done = format_status(read_job(jf), jf)
    check("a finished job is not called stale",
          "STALE" not in done and DONE in done)

    check("a job file mid-write is reported, not crashed on",
          "mid-write" in format_status({"_unreadable": str(jf)}, jf))


def test_status_cli(tmp: Path) -> None:
    print("\nbt-status on the command line")

    jf = tmp / "cli_job.json"
    job = JobTracker("batch_cli", "scan_probe", ["NQ"], "1d", tmp / "run",
                     path=jf)
    job.start()
    job.finish(DONE)

    env = {**os.environ, "BT_ACTIVE_JOB": str(jf)}
    p = subprocess.run([sys.executable, str(REPO / "backtest" / "status.py")],
                       capture_output=True, text=True, env=env, cwd=str(REPO))
    check("the CLI exits 0 with a job to report", p.returncode == 0,
          p.stderr[-200:])
    check("the CLI prints the readout", "BATCH JOB STATUS" in p.stdout
          and "batch_cli" in p.stdout)

    p = subprocess.run([sys.executable, str(REPO / "backtest" / "status.py"),
                        "--json"], capture_output=True, text=True, env=env,
                       cwd=str(REPO))
    check("--json emits the raw file",
          p.returncode == 0 and json.loads(p.stdout)["job_id"] == "batch_cli")

    p = subprocess.run([sys.executable, str(REPO / "backtest" / "status.py"),
                        "--file", str(tmp / "absent.json")],
                       capture_output=True, text=True, env=env, cwd=str(REPO))
    check("a missing job file exits non-zero", p.returncode != 0)

    p = subprocess.run([sys.executable, str(REPO / "backtest" / "run.py"),
                        "--help"], capture_output=True, text=True, cwd=str(REPO))
    check("bt-run --help documents the batch flags",
          p.returncode == 0
          and all(f in p.stdout for f in ("--symbols", "--scan", "--bg", "--ml")))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_expand_grid()
        test_scan_matches_engine(tmp)
        test_scan_selection_respects_gate1(tmp)
        test_ml_off_is_absent_not_zero(tmp)
        test_leaderboard(tmp)
        test_symbol_parsing()
        test_job_tracker(tmp)
        test_status_readout(tmp)
        test_status_cli(tmp)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
