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
import shutil                                                    # noqa: E402
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
from backtest.engine import BacktestConfig                          # noqa: E402
from backtest.report_html import (build_inspector,                  # noqa: E402
                                  generate_html_report,
                                  module_docstring, normalize_indicators,
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
                 "variants_tested": 1,
                 # What the strategy module declares about itself, with its
                 # bound parameters already filled in - the logic card states
                 # this rather than inferring anything from the trades.
                 "logic": {
                     "concept": "Trend following on a moving-average pair.",
                     "entry": "Go Long when the Fast SMA (10) crosses above "
                              "the Slow SMA (30).",
                     "exit": "Exit when the Fast SMA (10) crosses back below "
                             "the Slow SMA (30).",
                 }},
    }
    m.update(over)
    return m


FULL_ROBUSTNESS = {"wfo": {"efficiency_ratio": 0.63},
                   "monte_carlo": {"max_drawdown_pct_at_confidence": -15.8}}


COMMISSION = 2.25          # per side, per contract
SLIP_PER_TRADE = 10.0      # 1 tick each way on NQ: 2 * 0.25 * 20
COSTS_PER_TRADE = 2 * COMMISSION + SLIP_PER_TRADE


def synthetic_bars(n: int = 3_000, seed: int = 4) -> pd.DataFrame:
    """One symbol's 15-minute OHLCV frame, oldest first, as the lake returns it."""
    rng = np.random.default_rng(seed)
    px = 15_000 + np.cumsum(rng.normal(0, 4.0, n))
    return pd.DataFrame({
        "ts": pd.date_range("2021-01-04", periods=n, freq="15min", tz="UTC"),
        "symbol": "NQ", "open": px, "high": px + 3.0, "low": px - 3.0,
        "close": px + rng.normal(0, 1.0, n), "volume": rng.integers(50, 900, n),
    })


def synthetic_indicators(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """Two moving averages, the shape a strategy's `indicators()` hook returns."""
    close = bars["close"]
    return {"Fast SMA (10)": close.rolling(10, min_periods=10).mean(),
            "Slow SMA (30)": close.rolling(30, min_periods=30).mean()}


def synthetic_result(bars: pd.DataFrame | None = None, n_days: int = 900,
                     n_trades: int = 60, seed: int = 7):
    """
    A BacktestResult-shaped object: returns, trades, equity, config.

    Trades are placed on real bar timestamps when `bars` is given, so the trade
    inspector has something to line up against. A trade whose entry_time is not
    a bar in the frame would still produce a chart - searchsorted finds the
    nearest position - which is exactly why the alignment is tested rather than
    assumed.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-02", periods=n_days, freq="B", tz="UTC")
    returns = pd.Series(rng.normal(0.0006, 0.008, n_days), index=idx)

    if bars is not None:
        ts = pd.DatetimeIndex(bars["ts"])
        step = max(len(ts) // (n_trades + 2), 25)
        e_pos = np.arange(step, step * (n_trades + 1), step)[:n_trades]
        x_pos = np.minimum(e_pos + rng.integers(4, 25, len(e_pos)), len(ts) - 1)
        entry_t, exit_t = ts[e_pos], ts[x_pos]
        entry_px = bars["open"].to_numpy()[e_pos]
        exit_px = bars["open"].to_numpy()[x_pos]
    else:
        n_trades = min(n_trades, n_days - 1)
        entry_t, exit_t = idx[:n_trades], idx[1:n_trades + 1]
        entry_px = rng.normal(15_000, 40, n_trades)
        exit_px = rng.normal(15_000, 40, n_trades)

    trades = pd.DataFrame({
        "entry_time": entry_t, "exit_time": exit_t,
        "symbol": "NQ", "direction": "long",
        "entry_price": entry_px, "exit_price": exit_px,
    })
    trades["gross_pnl"] = (trades["exit_price"] - trades["entry_price"]) * 20.0
    trades["costs"] = COSTS_PER_TRADE
    trades["pnl"] = trades["gross_pnl"] - trades["costs"]

    class _R:
        pass

    r = _R()
    r.returns = returns
    r.trades = trades
    r.equity = (1 + returns).cumprod() * 100_000
    r.config = BacktestConfig(slippage_ticks=1.0, commission_per_side=COMMISSION,
                              flat_by_close=True, variants_tested=3)
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
    bars = synthetic_bars()
    result = synthetic_result(bars)
    m = clearing_metrics(trade_count=len(result.trades))
    audit = audit_acceptance_gates(m, FULL_ROBUSTNESS, {"sharpe": 1.45})

    out = generate_html_report(bars, result, m, audit,
                               tmp / "report_version_a.html",
                               strat_name="sma_crossover",
                               version_label="Version A · rule-based",
                               indicators=synthetic_indicators(bars))
    html = out.read_text(encoding="utf-8")

    check("the file was written", out.exists(), f"{out.stat().st_size / 1e6:.2f} MB")
    check("it is a complete HTML document",
          html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>"))
    problems = _unbalanced(html)
    check("the generated markup is balanced", not problems, "; ".join(problems[:3]))
    check("every section rendered", html.count('<div class="card">') == 7,
          f"{html.count('<div class=' + chr(34) + 'card' + chr(34) + '>')} cards")
    check("gate badges for all three gates are present",
          all(g in html for g in ("GATE 1", "GATE 2", "GATE 3")))
    check("the alpha metrics table is present", "Alpha metrics" in html)
    check("the monthly heatmap is present",
          "Monthly returns" in html and "background:rgba(" in html)
    check("the heatmap carries a legend and prints every value",
          'class="ramp"' in html and "colour is intensity only" in html)
    check("the trade log is present", "Trade log" in html)
    check("provenance is present", "Provenance" in html)
    check("the strategy name is the header",
          "<h1>sma_crossover" in html.replace("\n", ""))

    # (g) The logic card. Stops are the entry that matters most: the engine
    # models none, and a reader who assumes an unstated stop is reading a
    # different strategy.
    check("the strategy logic card is present", "Strategy logic" in html)
    card = _card(html, "Strategy logic")
    for label in ("Core concept", "Entry trigger", "Exit rule",
                  "Risk management", "Execution"):
        check(f"the logic card states {label.lower()}", f">{label}<" in html)
    check("the entry trigger is the module's own declared sentence",
          "Go Long when the Fast SMA (10) crosses above the Slow SMA (30)."
          in html)
    check("the exit rule is too",
          "Exit when the Fast SMA (10) crosses back below the Slow SMA (30)."
          in html)
    check("it states that no stop-loss is modelled",
          "Stop loss: NONE MODELLED" in html)
    check("and that no take-profit is either",
          "Take profit: NONE MODELLED" in html)
    check("it names the fill bar, not the signal bar",
          "NEXT bar&#x27;s open" in html or "NEXT bar's open" in html)
    check("session flatten reads off the config, not a guess",
          "20:00 UTC" in html)
    check("slippage and commission are stated per side",
          f"${COMMISSION:,.2f} per side" in html
          and "1 tick charged each way" in html)
    check("counts and their nouns agree", "1 contract per trade" in html
          and "contract(s)" not in card and "tick(s)" not in card)
    check("the risk and execution facts are bullets, not a paragraph",
          html.count('<ul class="bullets">') == 2)

    # The card is for a reader deciding whether to trade this, so the module's
    # own vocabulary must not leak into it. These are the names that used to be
    # printed verbatim.
    for jargon in ("signal_fn", "clean_signals", "BacktestConfig",
                   "multi-symbol", "long-format", "entry mask", "boolean",
                   "backtest/specs.py", "flat_by_close"):
        check(f"the card is free of '{jargon}'", jargon not in card)

    # A module that declares nothing gets an honest blank, not an invented
    # description — and the docstring fallback prints its summary paragraph
    # only, because the rest of a strategy docstring is notes for whoever edits
    # the module.
    silent = clearing_metrics()
    silent["meta"] = {k: v for k, v in silent["meta"].items() if k != "logic"}
    quiet = generate_html_report(
        bars, result, silent, audit, tmp / "nologic.html",
        strat_description="Buys dips in an uptrend.\n\n"
                          "`signal_fn` returns two boolean masks aligned to "
                          "the long-format frame.")
    qh = quiet.read_text(encoding="utf-8")
    qcard = _card(qh, "Strategy logic")
    check("an undeclared concept says so rather than inventing one",
          "Not declared by the strategy module" in qcard)
    check("the docstring summary is kept", "Buys dips in an uptrend." in qcard)
    check("the implementation notes under it are not",
          "long-format" not in qcard and "signal_fn" not in qcard)

    # (e) Trade log columns, search, and sort.
    for col in ("Side", "Entry price", "Exit price", "Return %", "Fees $",
                "Slippage $", "Net P&amp;L $"):
        check(f"the trade log has a {col} column", f">{col}<" in html)
    check("the trade log is searchable", 'id="trade-search"' in html)
    # Counted against the declared column list rather than a frozen number, so
    # adding a column (Side arrived when the engine learned to go short) does
    # not fail a check about sortability.
    from backtest.report_html import _TRADE_COLS
    check("the trade log is sortable",
          html.count('aria-sort="none"') == len(_TRADE_COLS)
          and 'data-sort="0"' in html,
          f"{html.count('aria-sort=' + chr(34) + 'none' + chr(34))} sortable "
          f"headers vs {len(_TRADE_COLS)} declared columns")
    check("rows carry numeric sort keys, not rendered text",
          'data-v="' in html)

    # Fees and slippage are split out of the engine's single cost figure.
    check("commission is reported per round trip",
          f">{2 * COMMISSION:,.2f}<" in html, f"{2 * COMMISSION:.2f}")
    check("slippage is the remainder of the cost figure",
          f">{SLIP_PER_TRADE:,.2f}<" in html, f"{SLIP_PER_TRADE:.2f}")

    # (h) The trade inspector.
    check("the trade inspector modal is present", 'id="trade-modal"' in html)
    check("rows are clickable and keyboard-reachable",
          'data-trade="0"' in html and 'role="button"' in html)
    check("the bar windows are embedded, not fetched",
          "window.INSPECTOR=" in html and "candlestick" in html)
    check("the indicator lines are embedded with them",
          '"Fast SMA (10)"' in html and '"Slow SMA (30)"' in html)
    check("and the modal plots them as named lines",
          "mode: 'lines'" in html and "showlegend: lines.length > 0" in html)

    # A report whose caller has no indicators to hand still draws candles.
    plain = generate_html_report(bars, result, m, audit, tmp / "noind.html")
    check("no indicators still produces a working inspector",
          '"ind":[]' in plain.read_text(encoding="utf-8"))

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

    # Interpolated strings are escaped - including the description, which now
    # comes from a module docstring and is therefore attacker-adjacent in the
    # one case that matters: a strategy a model just wrote.
    evil = clearing_metrics()
    # No declared concept, so the handed-in description is what the card
    # prints — which is the case the escaping has to hold for.
    evil["meta"] = {k: v for k, v in evil["meta"].items() if k != "logic"}
    evil["meta"]["strategy"] = 'x<script>alert(1)</script>'
    esc = generate_html_report(bars, result, evil, audit, tmp / "escaped.html",
                               strat_description="<img src=x onerror=alert(2)>")
    body = esc.read_text(encoding="utf-8")
    check("interpolated names are HTML-escaped",
          "<script>alert(1)</script>" not in body and "&lt;script&gt;" in body)
    check("the description is escaped too",
          "<img src=x" not in body and "&lt;img src=x" in body)

    # A report with no audit must not read as a cleared strategy.
    no_audit = generate_html_report(bars, result, m, None, tmp / "noaudit.html")
    head = no_audit.read_text(encoding="utf-8").split("Alpha metrics")[0]
    check("no gate audit renders as 'no claim', never as a pass",
          "no claim" in head and ">PASS<" not in head)

    # The trade log cap has to announce itself.
    capped = generate_html_report(bars, result, m, audit, tmp / "capped.html",
                                  max_trade_rows=25)
    cap_html = capped.read_text(encoding="utf-8")
    check("a truncated trade log says so on the page",
          "Showing the first 25" in cap_html
          and f"{len(result.trades):,}" in cap_html)
    check("and it really is truncated",
          cap_html.count('data-trade="') < html.count('data-trade="'))
    check("the inspector is truncated with it — no orphan windows",
          '"trades":[' in cap_html
          and cap_html.count('data-trade="') == 25)

    # Without bars, every other section still renders and the inspector says
    # it is unavailable rather than silently vanishing.
    no_bars = generate_html_report(None, result, m, audit, tmp / "nobars.html")
    nb = no_bars.read_text(encoding="utf-8")
    check("a report without bars still renders", "Trade log" in nb)
    check("and says the inspector is unavailable",
          "Pass the bars frame" in nb and 'id="trade-modal"' not in nb)

    # Degrades rather than raising on an empty result.
    class _Empty:
        returns = pd.Series(dtype=float)
        trades = pd.DataFrame()
        equity = pd.Series(dtype=float)
        config = None

    empty = generate_html_report(None, _Empty(), {"meta": {}}, None,
                                 tmp / "empty.html")
    check("an empty result produces a report instead of a traceback",
          empty.exists() and "nothing to plot" in empty.read_text(encoding="utf-8"))


def _card(markup: str, heading: str) -> str:
    """
    Just the named card's markup.

    Split on the rendered `<h2>`, not on the bare title: the stylesheet carries
    a `/* Strategy logic card */` comment above the page's own body, so a plain
    substring split hands back the whole document — including the 4.9 MB inlined
    plotly bundle, which contains every word anyone might grep for.
    """
    after = markup.split(f"<h2>{heading}", 1)
    if len(after) < 2:
        return ""
    return after[1].split("<h2>", 1)[0]        # up to the next card's heading


def _unbalanced(markup: str) -> list[str]:
    """
    Tags left open or closed out of order in the markup WE generate.

    The inlined plotly bundle is stripped first: minified JavaScript is full of
    angle brackets that are not markup, and parsing them as tags reports
    failures that do not exist. What remains is this module's own output, which
    a browser will silently paper over if it is wrong - a stray unclosed div
    swallows every card below it and the page still "renders".
    """
    from html.parser import HTMLParser

    void = {"meta", "br", "hr", "img", "input", "link", "source", "col"}
    body = re.sub(r"<script[^>]*>.*?</script>", "", markup, flags=re.S)

    class _P(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.err: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if not self.stack:
                self.err.append(f"</{tag}> with nothing open")
            elif self.stack[-1] != tag:
                self.err.append(f"</{tag}> closes <{self.stack[-1]}>")
                if tag in self.stack:
                    while self.stack and self.stack.pop() != tag:
                        pass
            else:
                self.stack.pop()

    p = _P()
    p.feed(body)
    return p.err + [f"<{t}> never closed" for t in p.stack]


def test_inspector_payload(tmp: Path) -> None:
    """The windows the modal draws, checked against the bars they came from."""
    print("\nTrade inspector payload")
    bars = synthetic_bars(n=1_200)
    result = synthetic_result(bars, n_trades=12)
    trades = result.trades
    ts = pd.DatetimeIndex(bars["ts"])

    ins = build_inspector(bars, trades)
    B, T = ins["bars"], ins["trades"]
    check("one window per trade", len(T) == len(trades), f"{len(T)} windows")

    # The epoch-unit trap: pandas indexes are microsecond-resolution here, so
    # asi8 // 1e6 yields SECONDS and every candle lands in 1970.
    first = pd.to_datetime(B["t"][0], unit="ms", utc=True)
    check("timestamps are epoch milliseconds",
          first.year == ts[0].year, str(first))

    ok_align = True
    for i in range(len(T)):
        t = T[i]
        e_time = pd.to_datetime(B["t"][t["e"]], unit="ms", utc=True)
        x_time = pd.to_datetime(B["t"][t["x"]], unit="ms", utc=True)
        if (e_time != trades["entry_time"].iloc[i]
                or x_time != trades["exit_time"].iloc[i]
                or not (t["lo"] <= t["e"] <= t["x"] <= t["hi"])):
            ok_align = False
            break
    check("every window's entry and exit point at the right bars", ok_align)

    widths = [t["hi"] - t["lo"] + 1 for t in T]
    holds = [t["x"] - t["e"] for t in T]
    check("each window is 20 bars before the entry to 10 after the exit",
          all(w == h + 31 for w, h in zip(widths, holds))
          or T[0]["lo"] == 0,        # the first trade can be clipped at the start
          f"widths {widths[:3]}")
    check("prices match the bars they were taken from",
          B["o"][T[0]["e"]] == round(float(trades["entry_price"].iloc[0]), 6))

    # Overlapping windows are shared, not duplicated per trade.
    check("overlapping windows are deduplicated",
          ins["n"] <= sum(widths), f"{ins['n']} bars vs {sum(widths)} naive")

    # Indicator overlays ride the same compact index as the candles. If they
    # did not, a line would be drawn against the wrong bars and the crossover
    # would appear where it did not happen.
    ind = synthetic_indicators(bars)
    with_ind = build_inspector(bars, trades, indicators=ind)
    lines = with_ind["bars"]["ind"]
    check("one line per indicator handed in", len(lines) == 2,
          ", ".join(s["name"] for s in lines))
    check("each line is as long as the bar payload",
          all(len(s["v"]) == len(with_ind["bars"]["t"]) for s in lines))
    check("lines carry a colour, a dash and a legend name",
          all(s["color"] and s["dash"] and s["name"] for s in lines))
    check("indicator colours are distinct from each other",
          len({s["color"] for s in lines}) == 2)

    # Sample a bar and compare against the source series at that timestamp.
    fast = ind["Fast SMA (10)"].to_numpy(dtype=float)
    probe = with_ind["trades"][3]["e"]
    src_pos = int(ts.searchsorted(pd.to_datetime(with_ind["bars"]["t"][probe],
                                                 unit="ms", utc=True)))
    check("a line's value at a bar is that bar's indicator value",
          lines[0]["v"][probe] == round(float(fast[src_pos]), 6),
          f"{lines[0]['v'][probe]} vs {fast[src_pos]:.6f}")
    check("the warm-up is null, not zero and not carried forward",
          build_inspector(bars, trades.iloc[:1], pre=1_000,
                          indicators=ind)["bars"]["ind"][1]["v"][0] is None)

    # A series that is not the length of the frame is DROPPED. Reindexing it
    # would draw a line one bar out of step with the candles, silently.
    check("a mis-sized series is dropped, never realigned",
          normalize_indicators({"short": pd.Series([1.0, 2.0])}, len(bars)) == [])
    check("a non-numeric series is dropped too",
          normalize_indicators({"txt": pd.Series(["a"] * len(bars))},
                               len(bars)) == [])
    check("an all-NaN series is dropped rather than drawn as a blank legend",
          normalize_indicators({"nan": pd.Series([np.nan] * len(bars))},
                               len(bars)) == [])
    check("a DataFrame of columns is accepted as well as a mapping",
          len(normalize_indicators(pd.DataFrame(ind), len(bars))) == 2)
    check("junk is ignored instead of raising",
          normalize_indicators("not a series", len(bars)) == []
          and normalize_indicators(None, len(bars)) == [])

    # Degenerate inputs return an empty payload rather than half a chart.
    check("no bars -> empty payload", build_inspector(None, trades)["trades"] == [])
    check("no trades -> empty payload", build_inspector(bars, None)["trades"] == [])
    shuffled = bars.iloc[::-1]
    check("an out-of-order frame is refused, not searchsorted blindly",
          build_inspector(shuffled, trades)["trades"] == [])


def test_inspector_dom(tmp: Path) -> None:
    """
    Run the report's own JavaScript against a stub DOM.

    Python can prove the modal markup is on the page. Only this can prove that
    clicking a row draws the right bars - an off-by-one in the window slice
    renders a beautiful chart of the wrong trade.
    """
    print("\nTrade inspector behaviour (node)")
    harness = REPO / "tests" / "inspector_dom_test.js"
    node = shutil.which("node")
    if node is None:
        print("  SKIP  node is not installed — the DOM harness did not run")
        return

    bars = synthetic_bars(n=800)
    result = synthetic_result(bars, n_trades=13)
    m = clearing_metrics(trade_count=13)
    page = generate_html_report(bars, result, m, audit_acceptance_gates(m),
                                tmp / "dom.html", strat_name="sma_crossover",
                                indicators=synthetic_indicators(bars))
    proc = subprocess.run([node, str(harness), str(page)],
                          capture_output=True, text=True, cwd=REPO)
    for line in proc.stdout.strip().splitlines():
        if line.strip().startswith(("PASS", "FAIL")):
            print("  " + line.strip())
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
    check("the trade log's JavaScript passes its own suite",
          proc.returncode == 0,
          f"{proc.stdout.count('PASS')} checks")


def test_write_dual_reports(tmp: Path) -> None:
    print("\nwrite_dual_reports")
    bars = synthetic_bars(n=1_500)
    a, b = clearing_metrics(), clearing_metrics(sharpe=1.10, trade_count=180)
    dual = {
        "version_a": {"metrics": a, "result": synthetic_result(bars, seed=1, n_trades=20),
                      "gate_audit": audit_acceptance_gates(a, FULL_ROBUSTNESS,
                                                           {"sharpe": 1.45}, version="A")},
        "version_b": {"metrics": b, "result": synthetic_result(bars, seed=2, n_trades=8),
                      "gate_audit": audit_acceptance_gates(b, version="B")},
        "comparison": {"b_beats_a": False, "sharpe_delta": -0.5},
        "meta": a["meta"],
    }
    for m in (a, b):
        m["meta"] = dict(m["meta"], ml_threshold=0.55)
    out = write_dual_reports(dual, bars=bars, out_dir=tmp / "run",
                             max_trade_rows=50,
                             indicators=synthetic_indicators(bars))

    check("report_version_a.html was written",
          out["report_version_a"].name == "report_version_a.html"
          and out["report_version_a"].exists())
    check("report_version_b.html was written",
          out["report_version_b"].name == "report_version_b.html"
          and out["report_version_b"].exists())
    a_html = out["report_version_a"].read_text(encoding="utf-8")
    b_html = out["report_version_b"].read_text(encoding="utf-8")
    check("the two reports are labelled A and B",
          "Version A" in a_html and "Version B" in b_html)
    check("both carry their own trade inspector",
          "window.INSPECTOR=" in a_html and "window.INSPECTOR=" in b_html)
    check("each inspector holds that version's own trades",
          a_html.count('data-trade="') == 20 and b_html.count('data-trade="') == 8)
    check("both draw the same indicator lines — B filters A's entries, it does "
          "not recompute them",
          '"Fast SMA (10)"' in a_html and '"Fast SMA (10)"' in b_html)

    # The dual runner hands BOTH versions the same meta dict, so the ML
    # threshold is on Version A's metrics too. Only B applies the filter, so
    # only B's card may claim one.
    check("only Version B's card mentions the ML filter",
          "machine-learning filter" in _card(b_html, "Strategy logic")
          and "machine-learning filter" not in _card(a_html, "Strategy logic"))

    snap = json.loads(out["metrics_json"].read_text(encoding="utf-8"))
    check("dual_metrics.json is valid JSON with both versions",
          "version_a" in snap and "version_b" in snap)
    check("it carries the gate audits", snap["version_a"]["gate_audit"]["status"] == PASS)
    check("pandas objects were dropped from the snapshot",
          "trades" not in snap["version_a"]["metrics"])

    # The default destination is <root>/<name>_<timestamp>/, so a re-run never
    # overwrites the evidence a promotion decision was made on.
    auto = write_dual_reports(dual, bars=bars, strat_name="sma_crossover",
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
        test_inspector_payload(tmp / "html")
        test_inspector_dom(tmp / "html")
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
