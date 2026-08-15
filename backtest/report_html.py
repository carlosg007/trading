#!/usr/bin/env python3
"""
report_html.py - Self-contained HTML tear sheet for one version of a strategy.

Location:  ~/src/trading/backtest/report_html.py

`backtest/report.py` produces the paste-ready text block. This produces the
file you open in a browser: the same numbers, plus an interactive equity and
drawdown chart, the monthly return matrix and the trade log.

    from backtest.report_html import generate_html_report
    generate_html_report(result, metrics, gate_audit,
                         "/mnt/backtest/artifacts/x/report_version_a.html",
                         version_label="Version A")

Self-contained means self-contained
-----------------------------------
Plotly is inlined (`include_plotlyjs="inline"`), so the file renders on a
machine with no network and nothing cached. That costs ~4.5 MB per report.
The alternative - a CDN script tag - produces a file that renders today and
shows an empty rectangle the first time it is opened offline, months later,
when someone is trying to work out why a strategy was promoted. Reports are
evidence; evidence has to keep.

Every number displayed here was computed by the deterministic engine and
handed in. This module formats, it does not calculate - the one exception is
the drawdown series and the monthly matrix, both derived from the returns
series by `backtest/report.py`'s own functions so the HTML and the text report
cannot disagree.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.report import (FAIL, NOT_EVALUATED, PASS, criterion_text,
                             drawdown_series, drawdown_stats, equity_curve,
                             monthly_table, yearly_table)

# The trade log is rendered in full up to this many rows. A 27-symbol 1-minute
# run produces 535,563 trades; writing them all into an HTML table makes a file
# no browser will open. When the cap bites, the report SAYS so - a silently
# truncated log reads as a complete one.
MAX_TRADE_ROWS = 2_000

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _esc(value: Any) -> str:
    """Everything interpolated into the page goes through here."""
    return html.escape(str(value), quote=True)


def _num(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any, spec: str = "{:.2f}", suffix: str = "") -> str:
    v = _num(value)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "∞"
    return spec.format(v) + suffix


def _sign_class(value: Any, better: str = "high") -> str:
    v = _num(value)
    if math.isnan(v) or abs(v) < 1e-12:
        return "neutral"
    good = v > 0 if better == "high" else v < 0
    return "pos" if good else "neg"


def _unpack(result: Any) -> tuple[pd.Series | None, pd.DataFrame | None, pd.Series | None]:
    """
    Pull (returns, trades, equity) out of a BacktestResult or a dict.

    Accepting both is deliberate: the engine hands back a BacktestResult, while
    `summarize_result` hands back a dict carrying the same series. Neither
    caller should have to reshape its result to get a report.
    """
    if result is None:
        return None, None, None
    if isinstance(result, dict):
        return (result.get("returns"), result.get("trades"), result.get("equity"))
    return (getattr(result, "returns", None), getattr(result, "trades", None),
            getattr(result, "equity", None))


def _daily_index(returns: pd.Series) -> pd.Series:
    """Returns as a Series with a DatetimeIndex, whatever shape it arrived in."""
    if returns is None or len(returns) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(returns).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
        s = s[s.index.notna()]
    return s.astype(float)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def _badge(status: str) -> str:
    cls = {PASS: "pass", FAIL: "fail"}.get(status, "unknown")
    return f'<span class="badge {cls}">{_esc(status)}</span>'


def _gates_html(gate_audit: dict | None) -> str:
    if not gate_audit:
        return ('<div class="card"><h2>Acceptance gates</h2>'
                '<p class="warn">No gate audit was supplied. This report makes '
                'no claim about whether this version cleared any gate.</p></div>')

    cards, detail = [], []
    for key in ("gate1", "gate2", "gate3"):
        gate = gate_audit.get("gates", {}).get(key)
        if not gate:
            continue
        cards.append(
            f'<div class="gate {("pass" if gate["status"] == PASS else "fail" if gate["status"] == FAIL else "unknown")}">'
            f'<div class="gate-name">{_esc(gate["name"])}</div>'
            f'{_badge(gate["status"])}</div>')
        rows = []
        for c in gate["checks"]:
            measured, required = criterion_text(c)
            note = (f'<div class="note">{_esc(c["note"])}</div>'
                    if c.get("note") else "")
            rows.append(
                f'<tr><td>{_esc(c["label"])}{note}</td>'
                f'<td class="num">{_esc(measured)}</td>'
                f'<td class="num dim">{_esc(required)}</td>'
                f'<td>{_badge(c["status"])}</td></tr>')
        detail.append(
            f'<h3>{_esc(gate["name"])}</h3>'
            f'<table class="grid"><thead><tr><th>Criterion</th>'
            f'<th class="num">Measured</th><th class="num">Required</th>'
            f'<th>Result</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')

    overall = gate_audit.get("status", NOT_EVALUATED)
    caveat = ""
    if overall == NOT_EVALUATED:
        caveat = ('<p class="warn">At least one gate was NOT EVALUATED. That is '
                  'not a pass — it means the evidence was never produced. Run the '
                  'walk-forward, the Monte Carlo bootstrap and the 3-year holdout '
                  'before treating this strategy as cleared.</p>')

    return (f'<div class="card"><h2>Acceptance gates '
            f'<span class="overall">overall {_badge(overall)}</span></h2>'
            f'<div class="gates">{"".join(cards)}</div>{caveat}'
            f'{"".join(detail)}</div>')


_METRIC_ROWS = [
    ("Sharpe", "sharpe", "{:.2f}", "", "high"),
    ("Sortino", "sortino", "{:.2f}", "", "high"),
    ("Calmar", "calmar", "{:.2f}", "", "high"),
    ("Profit factor", "profit_factor", "{:.2f}", "", "high"),
    ("Win rate", "win_rate_pct", "{:.1f}", " %", "high"),
    # Not colour-coded. A drawdown is signed negative and is always a cost;
    # painting it green because the sign happens to match "lower is better"
    # tells the reader nothing true.
    ("Max drawdown", "max_drawdown_pct", "{:.2f}", " %", "none"),
    ("Total return", "total_return_pct", "{:.2f}", " %", "high"),
    ("CAGR", "annualized_return_pct", "{:.2f}", " %", "high"),
    ("Net P&L", "total_pnl", "{:,.0f}", " $", "high"),
    ("Gross P&L", "gross_pnl", "{:,.0f}", " $", "high"),
    ("Total costs", "total_costs", "{:,.0f}", " $", "none"),
    ("Final equity", "final_equity", "{:,.0f}", " $", "none"),
    ("Trades", "trade_count", "{:,.0f}", "", "none"),
    ("Trading days", "n_days", "{:,.0f}", "", "none"),
]


def _metrics_html(metrics: dict, returns: pd.Series) -> str:
    m = dict(metrics or {})
    if "win_rate_pct" not in m and "win_rate" in m:
        m["win_rate_pct"] = _num(m["win_rate"]) * 100

    cells = []
    for label, key, spec, suffix, better in _METRIC_ROWS:
        cls = _sign_class(m.get(key), better) if better != "none" else "neutral"
        cells.append(f'<tr><td>{_esc(label)}</td>'
                     f'<td class="num {cls}">{_fmt(m.get(key), spec, suffix)}</td></tr>')

    extra = ""
    if len(returns):
        dd = drawdown_stats(returns)
        extra = (
            '<table class="grid"><tbody>'
            f'<tr><td>Longest drawdown</td><td class="num">{dd["longest_dd_days"]:,} days</td></tr>'
            f'<tr><td>Average drawdown</td><td class="num">{dd["avg_dd_days"]:.1f} days</td></tr>'
            f'<tr><td>Distinct drawdowns</td><td class="num">{dd["n_drawdowns"]:,}</td></tr>'
            f'<tr><td>Time underwater</td><td class="num">{dd["pct_time_underwater"]:.1f} %</td></tr>'
            f'<tr><td>Ann. volatility</td>'
            f'<td class="num">{returns.std(ddof=1) * np.sqrt(252) * 100:.2f} %</td></tr>'
            '</tbody></table>')

    if m.get("ruined"):
        extra += ('<p class="warn">Equity reached zero or below. This is a blown '
                  'account, not a bad quarter — the derived ratios above are '
                  'not meaningful.</p>')

    return (f'<div class="card"><h2>Alpha metrics</h2><div class="cols">'
            f'<table class="grid"><tbody>{"".join(cells)}</tbody></table>'
            f'{extra}</div></div>')


def _chart_html(returns: pd.Series, initial_capital: float) -> str:
    """
    Equity and drawdown, stacked on a shared x-axis.

    Returns a placeholder card rather than raising when plotly is missing or
    there is nothing to plot: a report with no chart is still evidence, and a
    traceback at the end of a six-hour backtest is not an acceptable trade.
    """
    if not len(returns):
        return ('<div class="card"><h2>Equity &amp; drawdown</h2>'
                '<p class="warn">No daily returns series — nothing to plot.</p></div>')
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:                                  # noqa: BLE001
        return (f'<div class="card"><h2>Equity &amp; drawdown</h2>'
                f'<p class="warn">plotly is not importable ({_esc(type(e).__name__)}), '
                f'so the chart was skipped. The numbers above are unaffected.</p></div>')

    eq = equity_curve(returns) * float(initial_capital or 1.0)
    dd = drawdown_series(returns) * 100.0

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.68, 0.32],
                        subplot_titles=("Equity", "Drawdown %"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity",
                             line=dict(color="#4ea1ff", width=1.6),
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                             fill="tozeroy", line=dict(color="#ff6b6b", width=1.0),
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"),
                  row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=560, showlegend=False,
        margin=dict(l=60, r=24, t=40, b=40),
        paper_bgcolor="#12161d", plot_bgcolor="#12161d",
        font=dict(family="ui-sans-serif, system-ui, sans-serif", size=12,
                  color="#c8d1dc"),
        hovermode="x unified")
    fig.update_xaxes(gridcolor="#232a34", zerolinecolor="#232a34")
    fig.update_yaxes(gridcolor="#232a34", zerolinecolor="#232a34")

    # include_plotlyjs="inline" is what makes the file standalone.
    div = fig.to_html(full_html=False, include_plotlyjs="inline",
                      config={"displaylogo": False, "responsive": True})
    return f'<div class="card"><h2>Equity &amp; drawdown</h2>{div}</div>'


def _monthly_html(returns: pd.Series) -> str:
    if not len(returns):
        return ""
    matrix = monthly_table(returns)
    head = "".join(f"<th>{m}</th>" for m in MONTH_NAMES)
    rows = []
    for year, row in matrix.iterrows():
        cells = []
        for month in range(1, 13):
            v = row.get(month, float("nan"))
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append('<td class="num empty">·</td>')
            else:
                cells.append(f'<td class="num {_sign_class(v)}">{v:.2f}</td>')
        total = float(((1 + returns[returns.index.year == year]).prod() - 1) * 100)
        rows.append(f'<tr><th class="rowhead">{int(year)}</th>{"".join(cells)}'
                    f'<td class="num total {_sign_class(total)}">{total:.2f}</td></tr>')

    yearly = yearly_table(returns)
    positive = int((yearly["return_pct"] > 0).sum())
    return (f'<div class="card"><h2>Monthly returns (%)</h2>'
            f'<div class="scroll"><table class="grid matrix"><thead><tr><th></th>'
            f'{head}<th>Year</th></tr></thead><tbody>{"".join(rows)}</tbody>'
            f'</table></div>'
            f'<p class="dim">Positive years: {positive}/{len(yearly)}</p></div>')


def _trades_html(trades: pd.DataFrame | None,
                 max_rows: int = MAX_TRADE_ROWS) -> str:
    if trades is None or len(trades) == 0:
        return ('<div class="card"><h2>Trade log</h2>'
                '<p class="warn">No trades.</p></div>')

    total = len(trades)
    shown = trades.head(max_rows) if total > max_rows else trades
    cols = [c for c in ("entry_time", "exit_time", "symbol", "direction",
                        "entry_price", "exit_price", "gross_pnl", "costs", "pnl")
            if c in shown.columns]
    if not cols:
        cols = list(shown.columns)

    head = "".join(f'<th>{_esc(c)}</th>' for c in cols)
    body = []
    for row in shown[cols].itertuples(index=False, name=None):
        cells = []
        for col, value in zip(cols, row):
            if isinstance(value, (pd.Timestamp, datetime)):
                cells.append(f'<td class="mono">{_esc(str(value)[:19])}</td>')
            elif isinstance(value, (int, float, np.integer, np.floating)):
                cls = _sign_class(value) if col in ("pnl", "gross_pnl") else "num"
                spec = "{:,.2f}" if col not in ("costs",) else "{:,.2f}"
                cells.append(f'<td class="num {cls}">{_fmt(value, spec)}</td>')
            else:
                cells.append(f'<td>{_esc(value)}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")

    cap = ""
    if total > max_rows:
        cap = (f'<p class="warn">Showing the first {max_rows:,} of {total:,} '
               f'trades. The rest are in the saved trades parquet, not here — '
               f'a full table at this size will not open in a browser.</p>')

    return (f'<div class="card"><h2>Trade log <span class="dim">({total:,})</span></h2>'
            f'{cap}<div class="scroll tall"><table class="grid trades"><thead><tr>'
            f'{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>')


def _meta_html(metrics: dict, gate_audit: dict | None) -> str:
    meta = (metrics or {}).get("meta", {}) or {}
    rows = [
        ("Strategy", meta.get("strategy", "unnamed")),
        ("Module", meta.get("strategy_path", "—")),
        ("Symbol", meta.get("symbol", "—")),
        ("Timeframe", meta.get("timeframe", "—")),
        ("Period", f"{str(meta.get('start', '—'))[:19]} → {str(meta.get('end', '—'))[:19]}"),
        ("Bars", f"{meta.get('bars', 0):,}" if meta.get("bars") else "—"),
        ("Parameters", json.dumps(meta.get("params", {}), default=str)),
        ("Initial capital", _fmt(meta.get("initial_capital"), "{:,.0f}", " $")),
        ("Costs included", "yes" if meta.get("costs_included") else "NO"),
        ("Variants tested", meta.get("variants_tested", "NOT RECORDED")),
    ]
    if meta.get("ml_threshold") is not None:
        rows.append(("ML threshold", meta["ml_threshold"]))

    body = "".join(f'<tr><td>{_esc(k)}</td><td class="mono">{_esc(v)}</td></tr>'
                   for k, v in rows)

    warn = ""
    if not meta.get("costs_included"):
        warn = ('<p class="warn">Costs were not recorded as included. Variant '
                'rankings change once commissions and slippage are applied — '
                'treat every figure in this report as provisional.</p>')
    if meta.get("variants_tested") in (None, "NOT RECORDED"):
        warn += ('<p class="warn">Variants tested was not recorded. A Sharpe '
                 'selected from many sweeps is far weaker evidence than a first '
                 'attempt, and this report cannot tell you which this is.</p>')
    return (f'<div class="card"><h2>Provenance</h2>'
            f'<table class="grid"><tbody>{body}</tbody></table>{warn}</div>')


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
_CSS = """
:root {
  --bg:#0d1117; --panel:#12161d; --line:#232a34; --ink:#c8d1dc;
  --ink-dim:#7d8794; --accent:#4ea1ff; --pos:#3ddc84; --neg:#ff6b6b;
  --warn:#ffb454;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:14px; line-height:1.5; }
.wrap { max-width:1180px; margin:0 auto; padding:28px 20px 64px; }
header { border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:24px; }
h1 { font-size:24px; margin:0 0 6px; font-weight:650; letter-spacing:-0.01em; }
h2 { font-size:15px; margin:0 0 14px; font-weight:600; text-transform:uppercase;
  letter-spacing:0.08em; color:var(--ink-dim); }
h3 { font-size:13px; margin:20px 0 8px; font-weight:600; color:var(--ink); }
.sub { color:var(--ink-dim); font-size:13px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:20px; margin-bottom:20px; }
table.grid { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
table.grid th { text-align:left; font-weight:600; color:var(--ink-dim);
  font-size:12px; text-transform:uppercase; letter-spacing:0.05em;
  border-bottom:1px solid var(--line); padding:6px 10px; }
table.grid td { padding:5px 10px; border-bottom:1px solid rgba(35,42,52,0.6); }
table.grid tr:last-child td { border-bottom:none; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
.neutral { color:var(--ink); } .dim, .empty { color:var(--ink-dim); }
.cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:20px; }
.gates { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:12px; margin-bottom:8px; }
.gate { border:1px solid var(--line); border-radius:8px; padding:14px;
  display:flex; flex-direction:column; gap:8px; background:rgba(0,0,0,0.18); }
.gate.pass { border-color:rgba(61,220,132,0.45); }
.gate.fail { border-color:rgba(255,107,107,0.45); }
.gate.unknown { border-color:rgba(255,180,84,0.45); }
.gate-name { font-size:12px; color:var(--ink-dim); text-transform:uppercase;
  letter-spacing:0.06em; }
.badge { display:inline-block; padding:2px 9px; border-radius:999px;
  font-size:11px; font-weight:700; letter-spacing:0.06em; }
.badge.pass { background:rgba(61,220,132,0.16); color:var(--pos); }
.badge.fail { background:rgba(255,107,107,0.16); color:var(--neg); }
.badge.unknown { background:rgba(255,180,84,0.16); color:var(--warn); }
.overall { float:right; text-transform:none; letter-spacing:0; }
.warn { color:var(--warn); background:rgba(255,180,84,0.08);
  border-left:3px solid var(--warn); padding:10px 12px; border-radius:0 6px 6px 0;
  margin:14px 0 0; font-size:13px; }
.note { color:var(--warn); font-size:12px; margin-top:3px; }
.scroll { overflow-x:auto; }
.scroll.tall { max-height:520px; overflow-y:auto; }
table.matrix td, table.matrix th { padding:4px 8px; font-size:12px; }
.rowhead { color:var(--ink-dim); font-weight:600; }
.total { border-left:1px solid var(--line); font-weight:600; }
table.trades td { font-size:12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
footer { color:var(--ink-dim); font-size:12px; border-top:1px solid var(--line);
  padding-top:16px; margin-top:8px; }
@media (max-width:640px) { .wrap { padding:16px 12px 40px; } .card { padding:14px; } }
"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def generate_html_report(result: Any,
                         metrics: dict,
                         gate_audit: dict | None,
                         out_path: str | Path,
                         version_label: str = "Version A",
                         max_trade_rows: int = MAX_TRADE_ROWS) -> Path:
    """
    Write a self-contained dark-themed HTML tear sheet for ONE version.

    Parameters
    ----------
    result
        A `BacktestResult`, or a dict carrying `returns`, `trades`, `equity`.
    metrics
        The metrics dict from `agents.tier3_workers.summarize_result`.
    gate_audit
        The dict from `backtest.report.audit_acceptance_gates`, or None. None
        renders as "no gate audit was supplied" — never as a pass.
    out_path
        Destination `.html` file. Parent directories are created.

    Returns the path written.
    """
    returns, trades, _equity = _unpack(result)
    daily = _daily_index(returns)

    meta = (metrics or {}).get("meta", {}) or {}
    capital = _num(meta.get("initial_capital"))
    if math.isnan(capital) or capital <= 0:
        capital = 100_000.0

    name = meta.get("strategy", "unnamed strategy")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    period = (f"{daily.index[0].date()} → {daily.index[-1].date()}"
              if len(daily) else "no daily returns")

    sections = [
        _gates_html(gate_audit),
        _metrics_html(metrics or {}, daily),
        _chart_html(daily, capital),
        _monthly_html(daily),
        _trades_html(trades, max_trade_rows),
        _meta_html(metrics or {}, gate_audit),
    ]

    doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(name)} — {_esc(version_label)}</title>
<style>{_CSS}</style>
</head><body><div class="wrap">
<header>
  <h1>{_esc(name)} <span class="dim">·</span> {_esc(version_label)}</h1>
  <div class="sub">{_esc(period)} &nbsp;·&nbsp; {_esc(meta.get('symbol', '—'))}
    {_esc(meta.get('timeframe', ''))} &nbsp;·&nbsp; generated {_esc(generated)}</div>
</header>
{''.join(sections)}
<footer>
  Every figure on this page was computed by the deterministic backtest engine
  and formatted here. No model produced a number. In-sample results are not
  evidence of an edge until the 3-year holdout says so.
</footer>
</div></body></html>"""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def write_dual_reports(dual: dict,
                       out_dir: str | Path | None = None,
                       strat_name: str | None = None,
                       artifacts_root: str | Path = "/mnt/backtest/artifacts",
                       timestamp: str | None = None,
                       max_trade_rows: int = MAX_TRADE_ROWS) -> dict[str, Any]:
    """
    Write `report_version_a.html` and `report_version_b.html` for a dual run.

    `dual` is what `agents.tier1_master.run_dual_version_backtest` returns.
    Both versions get a report whether or not either cleared a gate — a failing
    version is exactly the one somebody will want to read.

    Without `out_dir`, the destination is
    `<artifacts_root>/<strat_name>_<timestamp>/`. The timestamp is part of the
    directory rather than the filename so a re-run never overwrites the
    evidence a promotion decision was made on.
    """
    va, vb = dual["version_a"], dual["version_b"]
    meta = dual.get("meta", {}) or {}

    if out_dir is None:
        name = strat_name or meta.get("strategy") or "strategy"
        stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path(artifacts_root) / f"{name}_{stamp}"
    out_dir = Path(out_dir)

    paths = {
        "version_a": generate_html_report(
            va.get("result"), va.get("metrics", {}), va.get("gate_audit"),
            out_dir / "report_version_a.html",
            version_label="Version A · rule-based", max_trade_rows=max_trade_rows),
        "version_b": generate_html_report(
            vb.get("result"), vb.get("metrics", {}), vb.get("gate_audit"),
            out_dir / "report_version_b.html",
            version_label="Version B · ML-filtered", max_trade_rows=max_trade_rows),
    }

    # The metrics snapshot promote.py locks into meta.json. Written next to the
    # reports so a promotion always cites numbers from a specific run rather
    # than whatever was on screen at the time.
    snapshot = {
        "meta": meta,
        "comparison": dual.get("comparison", {}),
        "version_a": {"metrics": _jsonable(va.get("metrics", {})),
                      "gate_audit": _jsonable(va.get("gate_audit"))},
        "version_b": {"metrics": _jsonable(vb.get("metrics", {})),
                      "gate_audit": _jsonable(vb.get("gate_audit"))},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reports": {k: str(v) for k, v in paths.items()},
    }
    snap_path = out_dir / "dual_metrics.json"
    snap_path.write_text(json.dumps(snapshot, indent=2, default=str),
                         encoding="utf-8")

    return {"dir": out_dir, "report_version_a": paths["version_a"],
            "report_version_b": paths["version_b"], "metrics_json": snap_path}


def _jsonable(obj: Any) -> Any:
    """
    Drop the pandas objects and normalise NaN/inf so json.dumps cannot fail.

    NaN becomes None rather than the bare `NaN` token json.dumps emits, which
    is not valid JSON and makes the snapshot unreadable by anything strict.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()
                if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    return obj
