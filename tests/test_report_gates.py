#!/usr/bin/env python3
"""
Unit tests for the dual-version evaluation, reporting and promotion toolchain.

    backtest/report.py       audit_acceptance_gates, print_dual_scorecard
    backtest/report_html.py  generate_html_report, write_dual_reports
    backtest/promote.py      promote, inspect_source, meta.json

Follows the `tests/` convention: no pytest, exits non-zero on failure, prints
what it checked. Needs neither the lake nor a network — every input here is
synthetic, and that is the point: the gate logic has to be checkable without a
six-hour backtest in front of it.

The checks that matter most are the negative ones. A gate that reports PASS on
evidence nobody produced is worse than no gate, so most of this file is about
what happens when a number is missing, undefined, or signed the other way.

    python3 tests/test_report_gates.py
"""

from __future__ import annotations

import os

# Set before numpy/sklearn are imported, because both read the thread count at
# import time. The Version B check refits a HistGradientBoostingClassifier once
# per completed trade on a few dozen rows; on a 16-core box each of those fits
# spawns a thread pool that costs far more than the fit, and the check goes
# from under two seconds to not finishing. Nothing here is compute-bound - the
# real work is a 1.6-second filter over 3,000 synthetic bars.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import ast                                                       # noqa: E402
import importlib.util                                            # noqa: E402
import json                                                      # noqa: E402
import re                                                        # noqa: E402
import subprocess                                                # noqa: E402
import sys                                                       # noqa: E402
import tempfile                                                  # noqa: E402
from pathlib import Path                                         # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backtest.promote import (VERSION_B_TEMPLATE, inspect_source,   # noqa: E402
                              promote, sha256)
from backtest.report import (FAIL, GATE_THRESHOLDS, NOT_EVALUATED,  # noqa: E402
                             PASS, audit_acceptance_gates,
                             format_dual_scorecard, print_dual_scorecard)
from backtest.report_html import (generate_html_report,             # noqa: E402
                                  write_dual_reports)

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
    return bool(ok)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def clearing_metrics(**over) -> dict:
    """Metrics that clear Gate 1 with room to spare, before overrides."""
    m = {
        "ok": True, "ruined": False,
        "sharpe": 1.60, "sortino": 2.10, "calmar": 1.30,
        "profit_factor": 1.90, "win_rate": 0.56,
        "max_drawdown_pct": -9.40, "total_return_pct": 74.0,
        "annualized_return_pct": 12.5, "total_pnl": 74_000.0,
        "gross_pnl": 82_000.0, "total_costs": 8_000.0,
        "final_equity": 174_000.0, "trade_count": 640, "n_days": 1_500,
        "meta": {"strategy": "sma_crossover", "symbol": "NQ",
                 "timeframe": "15m", "start": "2018-01-01", "end": "2023-12-31",
                 "bars": 149_000, "params": {"fast_window": 10},
                 "initial_capital": 100_000.0, "costs_included": True,
                 "variants_tested": 1},
    }
    m.update(over)
    return m


FULL_ROBUSTNESS = {"wfo": {"efficiency_ratio": 0.63},
                   "monte_carlo": {"max_drawdown_pct_at_confidence": -15.8}}


def synthetic_result(n_days: int = 900, n_trades: int = 300, seed: int = 7):
    """A BacktestResult-shaped object: returns, trades, equity."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-02", periods=n_days, freq="B", tz="UTC")
    returns = pd.Series(rng.normal(0.0006, 0.008, n_days), index=idx)
    entry = idx[:n_trades]
    trades = pd.DataFrame({
        "entry_time": entry, "exit_time": idx[1:n_trades + 1],
        "symbol": "NQ", "direction": "long",
        "entry_price": rng.normal(15_000, 40, n_trades),
        "exit_price": rng.normal(15_000, 40, n_trades),
        "gross_pnl": rng.normal(140, 850, n_trades), "costs": 9.0,
    })
    trades["pnl"] = trades["gross_pnl"] - trades["costs"]

    class _R:
        pass

    r = _R()
    r.returns = returns
    r.trades = trades
    r.equity = (1 + returns).cumprod() * 100_000
    return r


# --------------------------------------------------------------------------
# 1. Gate thresholds
# --------------------------------------------------------------------------
def test_thresholds() -> None:
    print("\nGate thresholds are the documented ones")
    g1, g2, g3 = (GATE_THRESHOLDS["gate1"], GATE_THRESHOLDS["gate2"],
                  GATE_THRESHOLDS["gate3"])
    check("Gate 1 Sharpe >= 1.20", g1["min_sharpe"] == 1.20)
    check("Gate 1 profit factor >= 1.50", g1["min_profit_factor"] == 1.50)
    check("Gate 1 trades >= 200", g1["min_trades"] == 200)
    check("Gate 1 max drawdown <= 15.0%", g1["max_drawdown_pct"] == 15.0)
    check("Gate 2 WFO efficiency >= 0.50", g2["min_wfo_efficiency"] == 0.50)
    check("Gate 2 MC 95% max DD <= 18.0%", g2["max_mc_drawdown_pct"] == 18.0)
    check("Gate 3 Sharpe retention >= 0.85 (<=15% degradation)",
          g3["min_sharpe_retention"] == 0.85)


# --------------------------------------------------------------------------
# 2. Gate 1
# --------------------------------------------------------------------------
def test_gate1() -> None:
    print("\nGate 1 — in-sample")
    full = audit_acceptance_gates(clearing_metrics(), FULL_ROBUSTNESS,
                                  {"sharpe": 1.45})
    check("a clearing run passes all three gates",
          full["status"] == PASS and full["passed"], full["status"])

    for label, over in (("Sharpe 1.19", {"sharpe": 1.19}),
                        ("profit factor 1.49", {"profit_factor": 1.49}),
                        ("199 trades", {"trade_count": 199}),
                        ("15.01% drawdown", {"max_drawdown_pct": -15.01})):
        a = audit_acceptance_gates(clearing_metrics(**over), FULL_ROBUSTNESS,
                                   {"sharpe": 1.45})
        check(f"{label} fails Gate 1",
              a["gates"]["gate1"]["status"] == FAIL and not a["passed"])

    for label, over in (("Sharpe 1.20", {"sharpe": 1.20}),
                        ("profit factor 1.50", {"profit_factor": 1.50}),
                        ("200 trades", {"trade_count": 200}),
                        ("15.00% drawdown", {"max_drawdown_pct": -15.0})):
        a = audit_acceptance_gates(clearing_metrics(**over), FULL_ROBUSTNESS,
                                   {"sharpe": 1.45})
        check(f"{label} is exactly on the boundary and passes",
              a["gates"]["gate1"]["status"] == PASS)

    # The engine signs drawdown negative; a caller handing in a positive number
    # means the same drawdown. Comparing raw would let -40 clear a 15 limit.
    neg = audit_acceptance_gates(clearing_metrics(max_drawdown_pct=-40.0))
    pos = audit_acceptance_gates(clearing_metrics(max_drawdown_pct=40.0))
    check("a 40% drawdown fails whichever sign it arrives with",
          neg["gates"]["gate1"]["status"] == FAIL
          and pos["gates"]["gate1"]["status"] == FAIL)

    nan = audit_acceptance_gates(clearing_metrics(sharpe=float("nan")),
                                 FULL_ROBUSTNESS, {"sharpe": 1.45})
    check("a NaN Sharpe is NOT EVALUATED, never a pass",
          nan["gates"]["gate1"]["status"] == NOT_EVALUATED and not nan["passed"])

    missing = audit_acceptance_gates({}, FULL_ROBUSTNESS, {"sharpe": 1.4})
    check("an empty metrics dict passes nothing",
          missing["gates"]["gate1"]["status"] == NOT_EVALUATED
          and not missing["passed"])


# --------------------------------------------------------------------------
# 3. Gate 2
# --------------------------------------------------------------------------
def test_gate2() -> None:
    print("\nGate 2 — robustness")
    none = audit_acceptance_gates(clearing_metrics(), None, {"sharpe": 1.45})
    check("no robustness evidence -> NOT EVALUATED, not PASS",
          none["gates"]["gate2"]["status"] == NOT_EVALUATED)
    check("and the audit as a whole does not pass", not none["passed"])

    a = audit_acceptance_gates(clearing_metrics(),
                               {"wfo": {"efficiency_ratio": 0.49},
                                "monte_carlo": {"max_drawdown_pct_at_confidence": -10.0}})
    check("WFO efficiency 0.49 fails", a["gates"]["gate2"]["status"] == FAIL)

    b = audit_acceptance_gates(clearing_metrics(),
                               {"wfo": {"efficiency_ratio": 0.50},
                                "monte_carlo": {"max_drawdown_pct_at_confidence": -18.0}})
    check("0.50 efficiency and an 18.0% MC drawdown are both on the boundary",
          b["gates"]["gate2"]["status"] == PASS)

    c = audit_acceptance_gates(clearing_metrics(),
                               {"wfo": {"efficiency_ratio": 0.60},
                                "monte_carlo": {"max_drawdown_pct_at_confidence": -18.01}})
    check("an 18.01% Monte Carlo drawdown fails",
          c["gates"]["gate2"]["status"] == FAIL)

    d = audit_acceptance_gates(clearing_metrics(),
                               {"wfo": 0.61, "monte_carlo": -12.0})
    check("bare floats are accepted for both criteria",
          d["gates"]["gate2"]["status"] == PASS)

    # A WFO run with no param_grid is not an overfitting test. The warning that
    # says so has to survive into the audit, or the ratio gets over-read.
    e = audit_acceptance_gates(
        clearing_metrics(),
        {"wfo": {"efficiency_ratio": 0.7, "warning": "No param_grid supplied"},
         "monte_carlo": -10.0})
    note = e["gates"]["gate2"]["checks"][0]["note"]
    check("the WFO 'no param_grid' warning is carried into the audit",
          bool(note) and "param_grid" in note)


# --------------------------------------------------------------------------
# 4. Gate 3
# --------------------------------------------------------------------------
def test_gate3() -> None:
    print("\nGate 3 — OOS holdout")
    none = audit_acceptance_gates(clearing_metrics(), FULL_ROBUSTNESS, None)
    check("no holdout -> NOT EVALUATED, not PASS",
          none["gates"]["gate3"]["status"] == NOT_EVALUATED and not none["passed"])

    base = clearing_metrics(sharpe=2.00)
    on = audit_acceptance_gates(base, FULL_ROBUSTNESS, {"sharpe": 1.70})
    check("exactly 15% degradation (2.00 -> 1.70) passes",
          on["gates"]["gate3"]["status"] == PASS,
          f"retention {on['sharpe_retention']:.3f}")
    off = audit_acceptance_gates(base, FULL_ROBUSTNESS, {"sharpe": 1.68})
    check("16% degradation (2.00 -> 1.68) fails",
          off["gates"]["gate3"]["status"] == FAIL,
          f"retention {off['sharpe_retention']:.3f}")

    up = audit_acceptance_gates(base, FULL_ROBUSTNESS, {"sharpe": 2.40})
    check("a holdout that improves on in-sample passes",
          up["gates"]["gate3"]["status"] == PASS)

    bare = audit_acceptance_gates(base, FULL_ROBUSTNESS, 1.80)
    check("a bare holdout Sharpe is accepted",
          bare["gates"]["gate3"]["status"] == PASS)

    # Two negative Sharpes divide to a healthy-looking positive ratio.
    neg = audit_acceptance_gates(clearing_metrics(sharpe=-0.40),
                                 FULL_ROBUSTNESS, {"sharpe": -0.36})
    check("a negative in-sample Sharpe does not manufacture 0.90 retention",
          neg["gates"]["gate3"]["status"] != PASS,
          neg["gates"]["gate3"]["status"])
    check("and the reason is recorded on the criterion",
          "undefined" in (neg["gates"]["gate3"]["checks"][0]["note"] or ""))


# --------------------------------------------------------------------------
# 5. Scorecard
# --------------------------------------------------------------------------
def test_scorecard() -> None:
    print("\nDual scorecard")
    a = clearing_metrics()
    b = clearing_metrics(sharpe=1.10, trade_count=180, profit_factor=2.30,
                         max_drawdown_pct=-7.10, win_rate=0.61)
    audit_a = audit_acceptance_gates(a, FULL_ROBUSTNESS, {"sharpe": 1.45}, version="A")
    audit_b = audit_acceptance_gates(b, version="B")

    text = format_dual_scorecard(a, b, audit_a, audit_b)
    check("both versions appear", "A · rule-based" in text and "B · ML-filtered" in text)
    check("all three gates are named",
          all(g in text for g in ("GATE 1", "GATE 2", "GATE 3")))
    check("A's overall PASS and B's FAIL are both shown",
          "PASS" in text and "FAIL" in text)
    check("an unevaluated gate is labelled, not blank",
          NOT_EVALUATED in text)

    # The caveat is about an INCOMPLETE audit, so it needs a version whose
    # overall status is NOT EVALUATED rather than FAIL - a FAIL outranks a
    # missing gate and needs no warning, because nobody reads FAIL as cleared.
    ok_b = clearing_metrics(sharpe=1.30)
    incomplete = format_dual_scorecard(
        a, ok_b, audit_a, audit_acceptance_gates(ok_b, version="B"))
    check("the incomplete-audit caveat fires when a gate was never run",
          "NOT" in incomplete and "not a pass" in incomplete.lower())
    check("a FAILing version does not get the incomplete caveat instead of FAIL",
          "FAIL" in text)
    check("the out-of-sample caveat is always present",
          "OUT-OF-SAMPLE" in text)
    check("Sharpe delta is reported against A",
          "Version A leads on Sharpe" in text)
    check("win rate is shown as a percent, not a fraction",
          "56.0" in text and "61.0" in text)
    check("costs-included provenance is on the card", "Costs included" in text)

    printed = print_dual_scorecard(a, b, audit_a, audit_b)
    check("print_dual_scorecard returns what it printed", printed == text)

    # Missing audits must not crash the card, and must not read as passes.
    bare = format_dual_scorecard(a, b, None, None)
    check("a scorecard with no audits still renders",
          "DUAL-VERSION SCORECARD" in bare and PASS not in bare.split("VERDICT")[0])

    nan = format_dual_scorecard({"sharpe": float("nan")}, {"sharpe": float("nan")})
    check("NaN metrics render as n/a rather than raising", "n/a" in nan)

    # An audit read back out of dual_metrics.json has None where a NaN was:
    # JSON has no NaN. Both renderers have to survive that round trip.
    round_tripped = json.loads(json.dumps(audit_b, default=str))
    text_rt = format_dual_scorecard(a, b, audit_a, round_tripped)
    check("an audit round-tripped through JSON still renders",
          NOT_EVALUATED in text_rt and "n/a" in text_rt)

    # Drawdowns are signed negative. Comparing raw, a deeper drawdown is a
    # negative delta, which reads as an improvement and is the opposite of what
    # happened - this is the real 2026-08-15 sma_crossover result.
    deep_a = clearing_metrics(max_drawdown_pct=-29.27)
    deep_b = clearing_metrics(max_drawdown_pct=-36.73)
    dd_row = [ln for ln in format_dual_scorecard(deep_a, deep_b).splitlines()
              if "Max drawdown" in ln][0]
    check("a deeper drawdown is marked worse, not better",
          dd_row.rstrip().endswith("-"), dd_row.strip())
    shallow = [ln for ln in format_dual_scorecard(deep_b, deep_a).splitlines()
               if "Max drawdown" in ln][0]
    check("a shallower drawdown is marked better",
          shallow.rstrip().endswith("+"), shallow.strip())


# --------------------------------------------------------------------------
# 6. HTML report
# --------------------------------------------------------------------------
def test_html(tmp: Path) -> None:
    print("\nHTML report")
    result = synthetic_result()
    m = clearing_metrics()
    audit = audit_acceptance_gates(m, FULL_ROBUSTNESS, {"sharpe": 1.45})

    out = generate_html_report(result, m, audit, tmp / "report_version_a.html",
                               version_label="Version A · rule-based")
    html = out.read_text(encoding="utf-8")

    check("the file was written", out.exists(), f"{out.stat().st_size / 1e6:.2f} MB")
    check("it is a complete HTML document",
          html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>"))
    check("gate badges for all three gates are present",
          all(g in html for g in ("GATE 1", "GATE 2", "GATE 3")))
    check("the alpha metrics table is present", "Alpha metrics" in html)
    check("the monthly matrix is present", "Monthly returns" in html)
    check("the trade log is present", "Trade log" in html)
    check("provenance is present", "Provenance" in html)

    # Self-contained: plotly inlined, nothing fetched at render time. A CDN
    # script tag renders an empty rectangle the first time the file is opened
    # offline, months later, when someone is checking why a strategy was
    # promoted.
    #
    # Checked as TAGS, not as substrings. The inlined bundle contains
    # "cdn.plot.ly" and "unpkg.com" in its own source - plotly's default
    # topojson URL and its mapbox marker icons - and neither is ever requested
    # by a scatter chart. Grepping for the string would fail a file that is in
    # fact standalone.
    ext_scripts = re.findall(r"<script[^>]*\ssrc\s*=", html)
    ext_links = re.findall(r"<link[^>]*\shref\s*=", html)
    check("plotly is inlined", "plotly" in html.lower() and len(html) > 1_000_000)
    check("no external script tag", not ext_scripts, str(ext_scripts[:2]))
    check("no external stylesheet tag", not ext_links, str(ext_links[:2]))

    # Interpolated strings are escaped.
    evil = clearing_metrics()
    evil["meta"] = dict(evil["meta"], strategy='x<script>alert(1)</script>')
    esc = generate_html_report(result, evil, audit, tmp / "escaped.html")
    body = esc.read_text(encoding="utf-8")
    check("interpolated names are HTML-escaped",
          "<script>alert(1)</script>" not in body and "&lt;script&gt;" in body)

    # A report with no audit must not read as a cleared strategy.
    no_audit = generate_html_report(result, m, None, tmp / "noaudit.html")
    head = no_audit.read_text(encoding="utf-8").split("Alpha metrics")[0]
    check("no gate audit renders as 'no claim', never as a pass",
          "no claim" in head and ">PASS<" not in head)

    # The trade log cap has to announce itself.
    capped = generate_html_report(result, m, audit, tmp / "capped.html",
                                  max_trade_rows=25)
    cap_html = capped.read_text(encoding="utf-8")
    check("a truncated trade log says so on the page",
          "Showing the first 25" in cap_html and "300" in cap_html)
    check("and it really is truncated",
          cap_html.count("<tr>") < html.count("<tr>"))

    # Degrades rather than raising on an empty result.
    class _Empty:
        returns = pd.Series(dtype=float)
        trades = pd.DataFrame()
        equity = pd.Series(dtype=float)

    empty = generate_html_report(_Empty(), {"meta": {}}, None, tmp / "empty.html")
    check("an empty result produces a report instead of a traceback",
          empty.exists() and "nothing to plot" in empty.read_text(encoding="utf-8"))


def test_write_dual_reports(tmp: Path) -> None:
    print("\nwrite_dual_reports")
    a, b = clearing_metrics(), clearing_metrics(sharpe=1.10, trade_count=180)
    dual = {
        "version_a": {"metrics": a, "result": synthetic_result(seed=1),
                      "gate_audit": audit_acceptance_gates(a, FULL_ROBUSTNESS,
                                                           {"sharpe": 1.45}, version="A")},
        "version_b": {"metrics": b, "result": synthetic_result(seed=2),
                      "gate_audit": audit_acceptance_gates(b, version="B")},
        "comparison": {"b_beats_a": False, "sharpe_delta": -0.5},
        "meta": a["meta"],
    }
    out = write_dual_reports(dual, out_dir=tmp / "run", max_trade_rows=50)

    check("report_version_a.html was written",
          out["report_version_a"].name == "report_version_a.html"
          and out["report_version_a"].exists())
    check("report_version_b.html was written",
          out["report_version_b"].name == "report_version_b.html"
          and out["report_version_b"].exists())
    check("the two reports are labelled A and B",
          "Version A" in out["report_version_a"].read_text(encoding="utf-8")
          and "Version B" in out["report_version_b"].read_text(encoding="utf-8"))

    snap = json.loads(out["metrics_json"].read_text(encoding="utf-8"))
    check("dual_metrics.json is valid JSON with both versions",
          "version_a" in snap and "version_b" in snap)
    check("it carries the gate audits", snap["version_a"]["gate_audit"]["status"] == PASS)
    check("pandas objects were dropped from the snapshot",
          "trades" not in snap["version_a"]["metrics"])

    # The default destination is <root>/<name>_<timestamp>/, so a re-run never
    # overwrites the evidence a promotion decision was made on.
    auto = write_dual_reports(dual, strat_name="sma_crossover",
                              artifacts_root=tmp / "artifacts", max_trade_rows=10)
    check("the default directory is <strat>_<timestamp>",
          auto["dir"].name.startswith("sma_crossover_")
          and auto["dir"].parent == tmp / "artifacts")


# --------------------------------------------------------------------------
# 7. Promotion
# --------------------------------------------------------------------------
SOURCE = REPO / "strategies" / "experimental" / "sma_crossover.py"


def test_inspect_source() -> None:
    print("\npromote.inspect_source")
    info = inspect_source(SOURCE)
    check("TIMEFRAME is read without importing", info["timeframe"] == "1d")
    check("SYMBOLS is read", info["symbols"] == ["NQ"])
    check("DEFAULT_PARAMS is read",
          info["params"] == {"fast_window": 10, "slow_window": 30})
    check("signal_fn is found", info["has_signal_fn"])
    check("make_signal_fn is found", info["has_make_signal_fn"])


def test_promote_version_a(tmp: Path) -> None:
    print("\npromote — Version A")
    metrics_path = tmp / "dual_metrics.json"
    a = clearing_metrics()
    metrics_path.write_text(json.dumps({
        "version_a": {"metrics": a,
                      "gate_audit": audit_acceptance_gates(
                          a, FULL_ROBUSTNESS, {"sharpe": 1.45}, version="A")},
        "version_b": {"metrics": clearing_metrics(sharpe=1.1), "gate_audit": None},
    }, default=str), encoding="utf-8")

    inc = tmp / "incubator"
    out = promote("sma_crossover", "A", SOURCE, metrics_path=metrics_path,
                  commit=False, incubator=inc)
    dest = inc / "sma_crossover"

    check("strat.py was written", (dest / "strat.py").exists())
    check("meta.json was written", (dest / "meta.json").exists())
    check("the metrics snapshot was copied", (dest / "dual_metrics.json").exists())
    check("Version A is promoted byte for byte",
          sha256(dest / "strat.py") == sha256(SOURCE))
    check("no baseline.py for Version A", not (dest / "baseline.py").exists())

    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    check("meta records the version", meta["version"] == "A")
    check("meta records the symbol", meta["symbols"] == ["NQ"])
    check("meta records the timeframe", meta["timeframe"] == "1d")
    check("meta records the parameters",
          meta["params"] == {"fast_window": 10, "slow_window": 30})
    check("meta records the source sha256", meta["source_sha256"] == sha256(SOURCE))
    check("meta records a timestamp", bool(meta["promoted_utc"]))
    check("meta locks the metrics snapshot",
          meta["metrics"]["sharpe"] == 1.60 and "locked from" in meta["metrics_status"])
    check("meta records the gate audit", meta["gate_audit_status"] == PASS
          and meta["gate_audit"]["gate1"] == PASS)
    check("meta records costs were included", meta["costs_included"] is True)
    check("nothing was committed", out["committed"] is False)

    # Overrides win over what the module declares.
    out2 = promote("sma_over", "A", SOURCE, symbol="ES", timeframe="30m",
                   params={"fast_window": 20}, commit=False, incubator=inc,
                   variants_tested=42)
    m2 = out2["meta"]
    check("--symbol overrides the module", m2["symbols"] == ["ES"])
    check("--timeframe overrides the module", m2["timeframe"] == "30m")
    check("--params merges over DEFAULT_PARAMS",
          m2["params"] == {"fast_window": 20, "slow_window": 30})
    check("--variants-tested is recorded", m2["variants_tested"] == 42)


def test_promote_version_b(tmp: Path) -> None:
    print("\npromote — Version B")
    inc = tmp / "incubator_b"
    out = promote("sma_b", "B", SOURCE, threshold=0.55, commit=False, incubator=inc)
    dest = inc / "sma_b"

    check("baseline.py is the verbatim source",
          sha256(dest / "baseline.py") == sha256(SOURCE))
    check("strat.py is the ML wrapper, not a copy",
          sha256(dest / "strat.py") != sha256(SOURCE))

    src = (dest / "strat.py").read_text(encoding="utf-8")
    ast.parse(src)                       # raises if the template is malformed
    check("the wrapper is syntactically valid Python", True)
    check("the wrapper applies the ML filter", "apply_ml_signal_filter" in src)
    check("the wrapper carries the threshold it was promoted with",
          "ML_THRESHOLD = 0.55" in src)

    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    check("meta records version B", meta["version"] == "B")
    check("meta records the ML threshold", meta["ml_threshold"] == 0.55)
    check("meta points at the baseline", meta["baseline"] == "baseline.py")

    # The promoted wrapper has to actually run, and has to be subtractive.
    spec = importlib.util.spec_from_file_location("promoted_b", dest / "strat.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(3)
    n = 3_000
    px = 100 + np.cumsum(rng.normal(0, 0.6, n))
    bars = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"),
        "symbol": "NQ", "open": px, "high": px + 0.5, "low": px - 0.5,
        "close": px, "volume": rng.integers(80, 900, n)})

    entries_b, exits_b = mod.signal_fn(bars)
    base_spec = importlib.util.spec_from_file_location("promoted_base",
                                                       dest / "baseline.py")
    base = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(base)
    entries_a, _ = base.signal_fn(bars, **mod.DEFAULT_PARAMS)

    check("the promoted wrapper returns boolean Series aligned to the bars",
          entries_b.dtype == bool and exits_b.dtype == bool
          and len(entries_b) == len(bars))
    check("Version B is subtractive — it never enters where A did not",
          not bool((entries_b & ~entries_a).any()),
          f"A={int(entries_a.sum())} B={int(entries_b.sum())}")

    # A multi-symbol frame is the trap the engine has no entry point for.
    two = pd.concat([bars.assign(symbol="NQ"), bars.assign(symbol="ES")])
    raised = False
    try:
        mod.signal_fn(two)
    except ValueError:
        raised = True
    check("the wrapper refuses a multi-symbol frame", raised)


def test_promote_refuses(tmp: Path) -> None:
    print("\npromote — refusals and honest gaps")
    inc = tmp / "incubator_r"

    # No metrics: recorded as absent, not invented.
    out = promote("no_metrics", "A", SOURCE, commit=False, incubator=inc)
    check("no metrics snapshot -> metrics_status NOT RECORDED",
          out["meta"]["metrics_status"] == "NOT RECORDED"
          and out["meta"]["metrics"] is None)
    check("and gate_audit_status is NOT EVALUATED, not PASS",
          out["meta"]["gate_audit_status"] == NOT_EVALUATED)

    # A failing audit blocks promotion.
    failing = tmp / "failing.json"
    bad = clearing_metrics(sharpe=0.4)
    failing.write_text(json.dumps({"version_a": {
        "metrics": bad,
        "gate_audit": audit_acceptance_gates(bad, FULL_ROBUSTNESS, {"sharpe": 0.3}),
    }}, default=str), encoding="utf-8")

    refused = False
    try:
        promote("bad", "A", SOURCE, metrics_path=failing, commit=False, incubator=inc)
    except SystemExit:
        refused = True
    check("a FAIL gate audit refuses promotion", refused)

    forced = promote("bad", "A", SOURCE, metrics_path=failing, force=True,
                     commit=False, incubator=inc)
    check("--force promotes anyway", (inc / "bad" / "strat.py").exists())
    check("and the override is recorded in meta.json",
          forced["meta"]["gates_overridden"] is True)

    # A module with no signal_fn cannot be run, let alone promoted.
    stub = tmp / "nosignal.py"
    stub.write_text("def something_else(bars):\n    return None\n", encoding="utf-8")
    raised = False
    try:
        promote("nosignal", "A", stub, commit=False, incubator=inc)
    except ValueError as e:
        raised = "signal_fn" in str(e)
    check("a module with no signal_fn is refused", raised)

    raised = False
    try:
        promote("missing", "A", tmp / "does_not_exist.py", commit=False, incubator=inc)
    except FileNotFoundError:
        raised = True
    check("a missing source is refused", raised)

    raised = False
    try:
        promote("badver", "C", SOURCE, commit=False, incubator=inc)
    except ValueError:
        raised = True
    check("a version other than A or B is refused", raised)


def test_promote_cli(tmp: Path) -> None:
    """The CLI is the documented interface; exercise it as a subprocess."""
    print("\npromote — CLI")
    scratch = tmp / "cli_repo"
    (scratch / "strategies" / "approved_incubator").mkdir(parents=True)

    proc = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "promote.py"), "--help"],
        capture_output=True, text=True, cwd=REPO)
    check("--help exits 0", proc.returncode == 0)
    for flag in ("--strat", "--version", "--source", "--metrics", "--force",
                 "--no-commit"):
        check(f"{flag} is documented", flag in proc.stdout)


# --------------------------------------------------------------------------
def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print("=" * 72)

    with tempfile.TemporaryDirectory(prefix="gates_") as td:
        tmp = Path(td)
        for sub in ("html", "dual", "pa", "pb", "pr", "cli"):
            (tmp / sub).mkdir(parents=True, exist_ok=True)
        test_thresholds()
        test_gate1()
        test_gate2()
        test_gate3()
        test_scorecard()
        test_html(tmp / "html")
        test_write_dual_reports(tmp / "dual")
        test_inspect_source()
        test_promote_version_a(tmp / "pa")
        test_promote_version_b(tmp / "pb")
        test_promote_refuses(tmp / "pr")
        test_promote_cli(tmp / "cli")

    print("\n" + "=" * 72)
    if _failures:
        print(f"FAILED — {len(_failures)} check(s):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
