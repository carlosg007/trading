#!/usr/bin/env python3
"""
report_html.py - Self-contained HTML tear sheet for one version of a strategy.

Location:  ~/src/trading/backtest/report_html.py

`backtest/report.py` produces the paste-ready text block. This produces the
file you open in a browser:

    a) Gate 1 / 2 / 3 pass-fail badges
    b) the alpha metrics table
    c) interactive equity and underwater drawdown curves
    d) the monthly return heatmap
    e) a searchable, sortable trade log
    f) the strategy name header
    g) the strategy logic card - entries, exits, stops, session flatten
    h) the trade inspector: click a row for a candlestick of that trade,
       with the strategy's own indicator lines drawn over it

    from backtest.report_html import generate_html_report
    generate_html_report(bars, result, metrics, gate_audit,
                         "/mnt/backtest/artifacts/x/report_version_a.html",
                         strat_name="sma_crossover",
                         version_label="Version A",
                         indicators={"Fast SMA (10)": fast, "Slow SMA (30)": slow})

Self-contained means self-contained
-----------------------------------
Plotly is inlined (`include_plotlyjs="inline"`), so the file renders on a
machine with no network and nothing cached. That costs ~4.5 MB per report.
The alternative - a CDN script tag - produces a file that renders today and
shows an empty rectangle the first time it is opened offline, months later,
when someone is trying to work out why a strategy was promoted. Reports are
evidence; evidence has to keep.

The same rule drives the trade inspector. Rather than re-reading the lake when
a row is clicked - which would need the lake, and a server - the bar windows
each shown trade needs are extracted at build time and embedded. Overlapping
windows are deduplicated, so the cost is roughly one row per bar in the union
of the windows rather than 31 rows per trade.

Indicator lines follow the bars into that payload. They are handed in already
calculated - by the strategy module's own `indicators()` hook, so the line the
reader sees is drawn from the same array the signal was taken from, not from a
second implementation of the same moving average that might disagree with it.
This module samples them onto the window and picks their colours; it does not
compute one.

Every number displayed here was computed by the deterministic engine and
handed in. This module formats, it does not calculate. The exceptions are all
derivations of handed-in numbers: the drawdown series and monthly matrix come
from `backtest/report.py`'s own functions so the HTML and the text report
cannot disagree, and the fee/slippage split is arithmetic on the engine's own
cost figure - see `_cost_split`.

Chart colors
------------
The equity/drawdown pair and the candlestick bodies use the validated dark
categorical steps (blue `#3987e5`, red `#e66767`): adjacent CVD ΔE 19.2,
normal-vision ΔE 29.0 against this surface. The entry and exit markers are the
green and red a trader expects, and that pair FAILS colorblind separation on
its own (deutan ΔE 4.1) - so the distinction is carried by shape (triangle up
against triangle down), by position, and by a printed IN / OUT label. Color is
the last of the four channels, never the only one.

The indicator lines sit on the same surface as the candles, so they avoid the
blue/red the candles already own: amber, violet, teal, then a light slate. Each
also gets its own dash pattern and a legend label, so two lines are told apart
by stroke and by name with the colour ignored entirely - which matters here
more than anywhere else on the page, because "which mean is the fast one" is
the whole question a crossover chart is being read to answer.
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

# The trade inspector's window: bars before the entry, bars after the exit.
INSPECT_PRE = 20
INSPECT_POST = 10

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Validated against this page's surface - see the module docstring.
C_UP = "#3987e5"        # equity, rising candles
C_DOWN = "#e66767"      # drawdown, falling candles
C_ENTRY = "#0ca30c"     # status good  - shape and label carry it too
C_EXIT = "#d03b3b"      # status critical

# Indicator overlays, in the order a strategy hands them over. Colour is never
# the only channel: the dash pattern and the legend label carry the same
# distinction, so the fast and slow means stay tellable apart in greyscale.
INDICATOR_STYLES = [
    ("#f0b429", "solid"),      # amber
    ("#b48ef2", "dash"),       # violet
    ("#38c7b8", "dot"),        # teal
    ("#9aa7b6", "dashdot"),    # slate
    ("#f08fc4", "longdash"),   # pink
    ("#8fd14f", "longdashdot"),  # lime
]


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


def _config(result: Any) -> Any:
    """The BacktestConfig behind a result, or None."""
    return getattr(result, "config", None)


def _cost_split(trades: pd.DataFrame, result: Any,
                symbol: str | None) -> tuple[pd.Series, pd.Series] | None:
    """
    Split the engine's single `costs` figure into commission and slippage.

    The engine records `costs = gross_pnl - pnl`, which is both sides'
    commission plus both sides' slippage. Commission is the deterministic half:

        commission_total = 2 * commission_per_side * contracts

    and slippage is the remainder. That is arithmetic on the engine's own
    number, not a re-derivation of it - the split cannot disagree with the
    total, because the total is what it is split from.

    Returns None rather than guessing when the symbol has no spec or the
    remainder comes out negative. A negative slippage means the assumption
    above is wrong for this run, and a column of impossible numbers is worse
    than one honest combined column.
    """
    cfg = _config(result)
    if cfg is None or trades is None or "costs" not in trades.columns:
        return None
    try:
        from backtest.specs import get_spec
        spec = get_spec(symbol) if symbol else None
    except Exception:                                          # noqa: BLE001
        spec = None

    per_side = getattr(cfg, "commission_per_side", None)
    if per_side is None:
        per_side = getattr(spec, "commission", None)
    if per_side is None:
        return None

    contracts = getattr(cfg, "contracts", 1) or 1
    fees = pd.Series(float(per_side) * 2.0 * contracts, index=trades.index)
    slippage = trades["costs"].astype(float) - fees
    # A cent of float noise is fine; a real negative is not.
    if bool((slippage < -0.01).any()):
        return None
    return fees, slippage


def _trade_return_pct(trades: pd.DataFrame) -> pd.Series:
    """
    Per-trade return as a percentage of the entry price, signed by direction.

    A price return, not a return on account equity: the account figure depends
    on position size and starting capital, both of which live in the config
    rather than in the trade. The header says which it is.
    """
    if trades is None or "entry_price" not in trades.columns:
        return pd.Series(dtype=float)
    entry = trades["entry_price"].astype(float)
    exit_ = trades["exit_price"].astype(float)
    raw = (exit_ / entry - 1.0) * 100.0
    if "direction" in trades.columns:
        short = trades["direction"].astype(str).str.lower().isin(["short", "sell", "-1"])
        raw = raw.where(~short, -raw)
    return raw.replace([np.inf, -np.inf], float("nan"))


def module_docstring(path: str | Path | None) -> str | None:
    """
    A strategy module's docstring, read WITHOUT importing it.

    `ast.parse` rather than `import` because a report is generated from a
    module that may have been written by a model minutes earlier, and importing
    a module runs it. Nothing here should have that power.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists() or p.suffix != ".py":
        return None
    try:
        import ast
        return ast.get_docstring(ast.parse(p.read_text(encoding="utf-8")))
    except Exception:                                          # noqa: BLE001
        return None


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
# Trade inspector payload
# --------------------------------------------------------------------------
def normalize_indicators(indicators: Any,
                         n_bars: int) -> list[tuple[str, np.ndarray]]:
    """
    Named float arrays aligned to the bars, from whatever shape arrived.

    Accepts a `{name: series}` mapping or a DataFrame of one column per line.
    A series that is not the length of the frame is DROPPED rather than padded
    or reindexed: an indicator drawn one bar out of step with the candles is a
    chart that shows a crossover happening where it did not, and silently
    aligning it is exactly how that gets shipped.

    Never raises. A broken indicator hook costs the overlay, not the report.
    """
    if indicators is None or n_bars <= 0:
        return []
    if isinstance(indicators, pd.DataFrame):
        items = [(str(c), indicators[c]) for c in indicators.columns]
    elif isinstance(indicators, pd.Series):
        items = [(str(indicators.name or "Indicator"), indicators)]
    elif isinstance(indicators, dict):
        items = [(str(k), v) for k, v in indicators.items()]
    else:
        return []

    out: list[tuple[str, np.ndarray]] = []
    for name, values in items:
        try:
            arr = np.asarray(pd.Series(values).to_numpy(), dtype=float)
        except (TypeError, ValueError):
            continue                       # a non-numeric column, not a line
        if arr.ndim != 1 or arr.size != n_bars:
            continue
        if not np.isfinite(arr).any():
            continue                       # all-NaN: nothing to draw
        out.append((name, arr))
    return out


def build_inspector(bars: pd.DataFrame | None,
                    trades: pd.DataFrame | None,
                    pre: int = INSPECT_PRE,
                    post: int = INSPECT_POST,
                    indicators: Any = None) -> dict[str, Any]:
    """
    The bar windows the trade inspector draws, as compact parallel arrays.

    For each trade, `pre` bars before the entry through `post` bars after the
    exit. Windows overlap heavily on an active strategy, so the union of bar
    indices is deduplicated and every trade points into it by position - on a
    real 15-minute run that turns ~31 rows per trade into closer to 8.

    Every window is a contiguous run of bar indices, so after the union is
    sorted it is still contiguous, and a trade's window is a plain slice. That
    is what lets the browser do `t.slice(lo, hi + 1)` with no index map.

    `indicators` are the strategy's own calculated series - a `{name: series}`
    mapping or a DataFrame, each the full length of `bars`. They are sampled
    onto the same compact index as the candles, so the browser slices them with
    the same `lo:hi` and cannot draw a line offset from the bars under it.
    NaN survives as JSON `null`, which Plotly renders as a gap: a moving
    average's warm-up is left blank rather than drawn flat at zero.

    Returns `{"bars": {...}, "trades": [...], "n": int}` - empty when there are
    no bars to draw, which is the correct payload for a result whose frame was
    not handed in.
    """
    empty = {"bars": {"t": [], "o": [], "h": [], "l": [], "c": [], "ind": []},
             "trades": [], "n": 0}
    if bars is None or len(bars) == 0 or trades is None or len(trades) == 0:
        return empty
    if not {"open", "high", "low", "close"} <= set(bars.columns):
        return empty

    ts = (pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
          if "ts" in bars.columns else
          pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True)))
    if not ts.is_monotonic_increasing:
        # The engine only ever hands over one symbol's frame, oldest first. An
        # unsorted frame means something else arrived, and searchsorted would
        # silently return nonsense positions for it.
        return empty

    entry_t = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"], utc=True))
    exit_t = pd.DatetimeIndex(pd.to_datetime(trades["exit_time"], utc=True))
    n_bars = len(ts)
    e_pos = np.clip(ts.searchsorted(entry_t, side="left"), 0, n_bars - 1)
    x_pos = np.clip(ts.searchsorted(exit_t, side="left"), 0, n_bars - 1)

    los = np.maximum(e_pos - pre, 0)
    his = np.minimum(x_pos + post, n_bars - 1)

    needed = np.zeros(n_bars, dtype=bool)
    for lo, hi in zip(los.tolist(), his.tolist()):
        needed[lo:hi + 1] = True
    idx = np.flatnonzero(needed)
    if idx.size == 0:
        return empty

    # Position of each original bar index inside the compact arrays.
    remap = np.full(n_bars, -1, dtype=np.int64)
    remap[idx] = np.arange(idx.size, dtype=np.int64)

    def col(name: str) -> list[float]:
        return [round(float(v), 6) for v in bars[name].to_numpy(dtype=float)[idx]]

    lines = []
    for i, (label, arr) in enumerate(normalize_indicators(indicators, n_bars)):
        color, dash = INDICATOR_STYLES[i % len(INDICATOR_STYLES)]
        window = arr[idx]
        lines.append({
            "name": label, "color": color, "dash": dash,
            # None, not NaN: `NaN` is not valid JSON, and Plotly reads null as
            # a gap - which is what a warm-up period actually is.
            "v": [None if not np.isfinite(v) else round(float(v), 6)
                  for v in window],
        })

    payload_bars = {
        # Epoch milliseconds: Date-constructible in the browser and about half
        # the bytes of an ISO string.
        #
        # `as_unit("ms")` rather than dividing asi8 by a million. Since pandas
        # 2.0 a DatetimeIndex carries its own resolution and asi8 is in THAT
        # unit, so the division only happens to be right for nanosecond
        # indexes. On the microsecond index this lake actually produces it
        # yields seconds, and every candlestick renders in 1970 - a wrong
        # chart, not an error.
        "t": ts[idx].as_unit("ms").asi8.tolist(),
        "o": col("open"), "h": col("high"),
        "l": col("low"), "c": col("close"),
        "ind": lines,
    }
    payload_trades = [
        {"lo": int(remap[lo]), "hi": int(remap[hi]),
         "e": int(remap[ep]), "x": int(remap[xp])}
        for lo, hi, ep, xp in zip(los.tolist(), his.tolist(),
                                  e_pos.tolist(), x_pos.tolist())
    ]
    return {"bars": payload_bars, "trades": payload_trades, "n": idx.size}


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
                             line=dict(color=C_UP, width=2),
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Underwater",
                             fill="tozeroy", line=dict(color=C_DOWN, width=1.5),
                             fillcolor="rgba(230,103,103,0.22)",
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


def _heat_style(value: float, vmax: float) -> str:
    """
    Diverging blue/red cell fill, intensity by magnitude.

    A diverging scale needs two hues and a neutral middle, so a month near zero
    fades to the panel rather than to a third colour. Alpha stops at 0.55 so
    the printed number keeps its contrast: the value is readable with the fill
    ignored entirely, which is what makes the colour an accent rather than the
    only channel carrying the data.
    """
    v = _num(value)
    if math.isnan(v) or vmax <= 0:
        return ""
    weight = min(abs(v) / vmax, 1.0) ** 0.65        # eases small months up
    alpha = round(0.06 + 0.49 * weight, 3)
    rgb = "57,135,229" if v > 0 else "230,103,103"
    return f' style="background:rgba({rgb},{alpha})"'


def _monthly_html(returns: pd.Series) -> str:
    if not len(returns):
        return ""
    matrix = monthly_table(returns)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 0.0
    if vmax <= 0:
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0

    head = "".join(f"<th>{m}</th>" for m in MONTH_NAMES)
    rows = []
    for year, row in matrix.iterrows():
        cells = []
        for month in range(1, 13):
            v = row.get(month, float("nan"))
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append('<td class="num empty">·</td>')
            else:
                cells.append(f'<td class="num heat"{_heat_style(v, vmax)}>{v:.2f}</td>')
        total = float(((1 + returns[returns.index.year == year]).prod() - 1) * 100)
        rows.append(f'<tr><th class="rowhead">{int(year)}</th>{"".join(cells)}'
                    f'<td class="num total {_sign_class(total)}">{total:.2f}</td></tr>')

    yearly = yearly_table(returns)
    positive = int((yearly["return_pct"] > 0).sum())
    legend = (f'<div class="legend"><span class="dim">−{vmax:.1f}%</span>'
              f'<span class="ramp" aria-hidden="true"></span>'
              f'<span class="dim">+{vmax:.1f}%</span>'
              f'<span class="dim legend-note">colour is intensity only — every '
              f'cell prints its value</span></div>')
    return (f'<div class="card"><h2>Monthly returns (%)</h2>{legend}'
            f'<div class="scroll"><table class="grid matrix"><thead><tr><th></th>'
            f'{head}<th>Year</th></tr></thead><tbody>{"".join(rows)}</tbody>'
            f'</table></div>'
            f'<p class="dim">Positive years: {positive}/{len(yearly)}</p></div>')


def _first_paragraph(text: str | None) -> str | None:
    """
    The opening paragraph of a docstring, which is the plain-English half.

    A strategy module's docstring keeps going into the calling contract, the
    array shapes and the traps - notes for whoever edits the module, and noise
    to whoever is deciding whether to trade it. The summary paragraph is the
    part written for a reader; the rest stays in the module where it is useful.
    """
    if not text:
        return None
    for block in text.strip().split("\n\n"):
        para = " ".join(line.strip() for line in block.splitlines()).strip()
        if para:
            return para
    return None


def _plain_flatten(cfg: Any) -> str:
    flat = getattr(cfg, "flat_by_close", None)
    if flat:
        return (f"Session flatten: ON — any position still open is closed at "
                f"{getattr(cfg, 'session_close_utc', '?')} UTC, so nothing is "
                f"carried overnight.")
    if flat is None:
        return ("Session flatten: not recorded — no run configuration reached "
                "this report.")
    return ("Session flatten: OFF — positions are held through the session "
            "close and across days.")


def _plural(value: Any, singular: str, plural: str | None = None) -> str:
    """`1 tick`, `1.5 ticks` - a count and its noun agreeing with each other."""
    v = _num(value)
    if math.isnan(v):
        return f"an unrecorded number of {plural or singular + 's'}"
    text = f"{v:g}"
    return f"{text} {singular if v == 1 else (plural or singular + 's')}"


def _is_version_b(version_label: str) -> bool:
    """
    Whether this report is the ML-filtered version.

    Read off the label because the metrics cannot answer it: the dual runner
    hands BOTH versions the same meta dict, so `ml_threshold` is present on
    Version A's metrics as well. Printing the filter sentence off that would
    have Version A's card describe a filter Version A never applied.
    """
    label = (version_label or "").strip().lower()
    return label.startswith("b") or label.startswith("version b")


def _logic_card(metrics: dict, result: Any, description: str | None,
                trades: pd.DataFrame | None,
                ml_filtered: bool = False) -> str:
    """
    What this strategy does and what the engine actually did, in plain English.

    Written for a reader deciding whether to trade the thing, not for whoever
    maintains the module. No function names, no array shapes, no repository
    paths: the entry and exit lines name the indicators and the numbers, and
    the execution lines name the price a fill was taken at and what it cost.

    The concept, entry and exit sentences are DECLARED by the strategy module
    (its `LOGIC` block, with the run's own parameters filled in) and carried
    here through the metrics meta. Nothing on this card is inferred from the
    signal arrays - a description guessed from the trades would be a guess
    printed as a fact.

    The execution half is read off the run configuration rather than described
    from memory, because these are the assumptions that quietly drift: which
    bar a signal fills on, whether the position is flattened at the session
    close, what a round trip cost.

    Stops are stated as absent because they are absent. The engine models no
    stop-loss and no take-profit - a position is opened by the entry rule and
    closed by the exit rule or the session flatten, and by nothing else. A
    reader who assumes an unstated 2% stop is reading a different strategy.
    """
    cfg = _config(result)
    meta = (metrics or {}).get("meta", {}) or {}
    params = meta.get("params") or {}
    logic = meta.get("logic") or {}

    concept = logic.get("concept") or (
        "Not declared by the strategy module — read the description above and "
        "the parameters below.")
    entry = logic.get("entry") or (
        "The strategy's own rule fires the entry; the position is opened on "
        "the next bar.")
    exit_rule = logic.get("exit") or (
        "The strategy's own rule fires the exit; the position is closed on the "
        "next bar.")

    if ml_filtered and meta.get("ml_threshold") is not None:
        entry += (f" This version then skips any of those entries the "
                  f"machine-learning filter scores below a "
                  f"{float(meta['ml_threshold']) * 100:g}% chance of winning. "
                  f"The filter can only remove entries, never add one.")

    risk = [
        "Stop loss: NONE MODELLED. No protective stop is placed on any trade.",
        "Take profit: NONE MODELLED. No profit target closes a trade early.",
        _plain_flatten(cfg),
        "Every exit comes from the exit rule above or the session flatten — "
        "read the drawdown figures knowing nothing else cuts a loser.",
    ]
    if getattr(cfg, "trailing_drawdown_pct", None) is not None:
        risk.insert(3, f"Trailing drawdown: {cfg.trailing_drawdown_pct}% — "
                       f"descriptive only. Account limits are enforced by the "
                       f"live execution bridge, not by this backtest.")

    execution = ["Fill price: the NEXT bar's open. Filling on the close of the "
                 "bar that produced the signal would be trading on information "
                 "the strategy did not have yet."]
    if cfg is not None:
        commission = (f"${cfg.commission_per_side:,.2f} per side, per contract"
                      if getattr(cfg, "commission_per_side", None) is not None
                      else "the contract's own published rate")
        execution += [
            f"Slippage: {_plural(getattr(cfg, 'slippage_ticks', None), 'tick')} "
            f"charged each way, on entry and on exit.",
            f"Commission: {commission}, charged both sides.",
            f"Position size: "
            f"{_plural(getattr(cfg, 'contracts', 1), 'contract')} per trade.",
        ]
    else:
        execution.append("Costs: not recorded — no run configuration reached "
                         "this report.")

    rows: list[tuple[str, str | list[str]]] = [
        ("Core concept", concept),
        ("Entry trigger", entry),
        ("Exit rule", exit_rule),
        ("Risk management", risk),
        ("Execution", execution),
    ]
    if params:
        rows.append(("Settings used",
                     " · ".join(f"{k} = {v}" for k, v in params.items())))
    if trades is not None and len(trades) and "direction" in trades.columns:
        sides = {str(d).lower() for d in trades["direction"].unique()}
        if sides <= {"long", "buy", "1"}:
            taken = "Long only — this strategy never sold short."
        elif sides <= {"short", "sell", "-1"}:
            taken = "Short only — this strategy never bought."
        else:
            taken = "Both directions — long and short trades were taken."
        rows.append(("Direction", taken))

    def value(v: str | list[str]) -> str:
        if isinstance(v, list):
            return ('<ul class="bullets">'
                    + "".join(f"<li>{_esc(item)}</li>" for item in v)
                    + "</ul>")
        return _esc(v)

    body = "".join(f'<tr><td class="lbl">{_esc(k)}</td><td>{value(v)}</td></tr>'
                   for k, v in rows)
    # The declared concept already says what this is; repeating the module's
    # summary above it just makes the reader compare two sentences.
    prose = None if logic.get("concept") else _first_paragraph(description)
    desc = f'<p class="desc">{_esc(prose)}</p>' if prose else ""
    return (f'<div class="card"><h2>Strategy logic</h2>{desc}'
            f'<table class="grid logic"><tbody>{body}</tbody></table></div>')


_TRADE_COLS = [
    ("#", "num", "Trade number, in exit order"),
    ("Entry time", "txt", "UTC"),
    ("Exit time", "txt", "UTC"),
    ("Entry price", "num", "The raw bar open, before slippage"),
    ("Exit price", "num", "The raw bar open, before slippage"),
    ("Return %", "num", "Price return from entry to exit, signed by direction"),
    ("Fees $", "num", "Commission, both sides"),
    ("Slippage $", "num", "The rest of the engine's cost figure"),
    ("Net P&L $", "num", "After all costs"),
]


def _trades_html(trades: pd.DataFrame | None, result: Any = None,
                 symbol: str | None = None,
                 max_rows: int = MAX_TRADE_ROWS,
                 inspectable: bool = False) -> str:
    """
    The trade log: searchable, sortable, and clickable when bars were supplied.

    Sort keys are carried in `data-v` rather than parsed out of the rendered
    text, so "1,234.50" and "2026-08-15 13:45" both sort as what they are
    instead of as strings.
    """
    if trades is None or len(trades) == 0:
        return ('<div class="card"><h2>Trade log</h2>'
                '<p class="warn">No trades.</p></div>')

    total = len(trades)
    shown = (trades.head(max_rows) if total > max_rows else trades).reset_index(drop=True)

    split = _cost_split(shown, result, symbol)
    fees, slip = split if split else (None, None)
    rets = _trade_return_pct(shown)

    head = "".join(
        f'<th class="{cls}" data-sort="{i}" tabindex="0" role="columnheader" '
        f'aria-sort="none" title="{_esc(tip)}">{_esc(label)}'
        f'<span class="arrow" aria-hidden="true"></span></th>'
        for i, (label, cls, tip) in enumerate(_TRADE_COLS))

    def cell(value: float, spec: str = "{:,.2f}", cls: str = "") -> str:
        v = _num(value)
        sortable = "" if math.isnan(v) else f' data-v="{v:.6f}"'
        return f'<td class="num {cls}"{sortable}>{_fmt(value, spec)}</td>'

    body = []
    for i in range(len(shown)):
        row = shown.iloc[i]
        entry_t, exit_t = str(row.get("entry_time"))[:19], str(row.get("exit_time"))[:19]
        pnl = _num(row.get("pnl"))
        costs = _num(row.get("costs"))
        f_val = fees.iloc[i] if fees is not None else float("nan")
        s_val = slip.iloc[i] if slip is not None else costs
        cells = [
            f'<td class="num" data-v="{i + 1}">{i + 1}</td>',
            f'<td class="mono" data-v="{_esc(entry_t)}">{_esc(entry_t)}</td>',
            f'<td class="mono" data-v="{_esc(exit_t)}">{_esc(exit_t)}</td>',
            cell(row.get("entry_price")),
            cell(row.get("exit_price")),
            cell(rets.iloc[i] if len(rets) else float("nan"), "{:,.3f}",
                 _sign_class(rets.iloc[i] if len(rets) else float("nan"))),
            cell(f_val),
            cell(s_val),
            cell(pnl, "{:,.2f}", _sign_class(pnl)),
        ]
        attrs = (f' data-trade="{i}" tabindex="0" role="button" '
                 f'aria-label="Inspect trade {i + 1}"' if inspectable else "")
        body.append(f'<tr{attrs}>{"".join(cells)}</tr>')

    notes = []
    if total > max_rows:
        notes.append(
            f'<p class="warn">Showing the first {max_rows:,} of {total:,} '
            f'trades. The rest are in the saved trades parquet, not here — a '
            f'full table at this size will not open in a browser.</p>')
    if split is None:
        notes.append(
            '<p class="warn">Fees and slippage could not be separated for this '
            'run — the Slippage column carries the engine\'s whole cost figure '
            'and Fees reads n/a. Splitting them needs the contract\'s '
            'commission from backtest/specs.py.</p>')
    hint = ('<p class="dim">Click a row to inspect that trade on the chart.</p>'
            if inspectable else
            '<p class="dim">Pass the bars frame to enable the trade inspector.</p>')

    return (f'<div class="card"><h2>Trade log <span class="dim">({total:,})</span></h2>'
            f'{"".join(notes)}'
            f'<div class="toolbar">'
            f'<label class="sr-only" for="trade-search">Search trades</label>'
            f'<input id="trade-search" type="search" placeholder="Search trades — '
            f'date, price, P&amp;L…" autocomplete="off">'
            f'<span class="dim" id="trade-count"></span></div>'
            f'{hint}'
            f'<div class="scroll tall"><table class="grid trades" id="trade-table">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody>'
            f'</table></div></div>')


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

/* Strategy logic card */
table.logic td { vertical-align:top; padding-top:9px; padding-bottom:9px; }
table.logic td.lbl { color:var(--ink-dim); white-space:nowrap; width:170px;
  font-weight:600; }
ul.bullets { margin:0; padding-left:18px; }
ul.bullets li { margin:0 0 4px; }
ul.bullets li:last-child { margin-bottom:0; }
.desc { color:var(--ink-dim); margin:0 0 14px; white-space:pre-wrap;
  border-left:2px solid var(--line); padding-left:12px; }

/* Monthly heatmap */
td.heat { color:var(--ink); }
.legend { display:flex; align-items:center; gap:8px; margin:0 0 12px;
  font-size:12px; flex-wrap:wrap; }
.ramp { width:140px; height:10px; border-radius:2px; display:inline-block;
  background:linear-gradient(90deg, rgba(230,103,103,0.55), rgba(230,103,103,0.06),
    rgba(57,135,229,0.06), rgba(57,135,229,0.55)); }
.legend-note { margin-left:4px; }

/* Trade log toolbar, sorting, row affordance */
.toolbar { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.toolbar input { flex:1; max-width:340px; background:#0b0f15; color:var(--ink);
  border:1px solid var(--line); border-radius:6px; padding:7px 10px;
  font-size:13px; font-family:inherit; }
.toolbar input:focus { outline:2px solid var(--accent); outline-offset:1px; }
table.trades thead th { cursor:pointer; user-select:none; position:sticky; top:0;
  background:var(--panel); z-index:1; }
table.trades thead th:focus-visible { outline:2px solid var(--accent);
  outline-offset:-2px; }
.arrow { display:inline-block; width:10px; color:var(--ink-dim); }
th[aria-sort="ascending"] .arrow::after { content:"\\2191"; }
th[aria-sort="descending"] .arrow::after { content:"\\2193"; }
table.trades tbody tr[data-trade] { cursor:pointer; }
table.trades tbody tr[data-trade]:hover,
table.trades tbody tr[data-trade]:focus-visible {
  background:rgba(57,135,229,0.14); outline:none; }
.sr-only { position:absolute; width:1px; height:1px; overflow:hidden;
  clip:rect(0 0 0 0); white-space:nowrap; }

/* Trade inspector modal */
.modal[hidden] { display:none; }
.modal { position:fixed; inset:0; z-index:50; display:flex; align-items:center;
  justify-content:center; padding:24px; background:rgba(4,7,11,0.74); }
.modal-box { background:var(--panel); border:1px solid var(--line);
  border-radius:12px; width:min(1000px,100%); max-height:92vh; overflow:auto;
  padding:20px; box-shadow:0 24px 64px rgba(0,0,0,0.55); }
.modal-head { display:flex; align-items:flex-start; justify-content:space-between;
  gap:16px; margin-bottom:12px; }
.modal-head h3 { margin:0; font-size:16px; }
.modal-facts { display:flex; flex-wrap:wrap; gap:6px 18px; margin:6px 0 0;
  font-size:12px; color:var(--ink-dim); font-variant-numeric:tabular-nums; }
.modal-close { background:none; border:1px solid var(--line); color:var(--ink);
  border-radius:6px; width:32px; height:32px; font-size:16px; cursor:pointer;
  flex:none; }
.modal-close:hover { border-color:var(--accent); color:var(--accent); }
"""

# Search, column sort, and the click-to-inspect modal. Vanilla, because the
# page has to work from a file:// URL with no network - a script tag pointing
# at a table library would break exactly when the report is being read as
# evidence. Plotly is already inlined for the equity chart, so the candlestick
# costs nothing extra.
_JS = """
(function () {
  var TABLE = document.getElementById('trade-table');
  if (!TABLE) return;
  var TBODY = TABLE.tBodies[0];
  var ROWS = Array.prototype.slice.call(TBODY.rows);
  var COUNT = document.getElementById('trade-count');

  function sortKey(cell) {
    if (!cell) return '';
    var v = cell.getAttribute('data-v');
    if (v === null) return cell.textContent.trim();
    var n = parseFloat(v);
    return isNaN(n) ? v : n;
  }

  /* Search ------------------------------------------------------------- */
  var search = document.getElementById('trade-search');
  function applyFilter() {
    var q = (search && search.value || '').trim().toLowerCase();
    var shown = 0;
    ROWS.forEach(function (tr) {
      var hit = !q || tr.textContent.toLowerCase().indexOf(q) !== -1;
      tr.style.display = hit ? '' : 'none';
      if (hit) shown++;
    });
    if (COUNT) {
      COUNT.textContent = q ? shown + ' of ' + ROWS.length + ' shown'
                            : ROWS.length + ' rows';
    }
  }
  if (search) search.addEventListener('input', applyFilter);
  applyFilter();

  /* Sort --------------------------------------------------------------- */
  var heads = Array.prototype.slice.call(TABLE.tHead.rows[0].cells);
  function sortBy(i, dir) {
    var sorted = ROWS.slice().sort(function (a, b) {
      var x = sortKey(a.cells[i]), y = sortKey(b.cells[i]);
      if (x < y) return -dir;
      if (x > y) return dir;
      return 0;
    });
    sorted.forEach(function (tr) { TBODY.appendChild(tr); });
  }
  heads.forEach(function (th, i) {
    function toggle() {
      var next = th.getAttribute('aria-sort') === 'ascending'
        ? 'descending' : 'ascending';
      heads.forEach(function (o) { o.setAttribute('aria-sort', 'none'); });
      th.setAttribute('aria-sort', next);
      sortBy(i, next === 'ascending' ? 1 : -1);
    }
    th.addEventListener('click', toggle);
    th.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    });
  });

  /* Trade inspector ----------------------------------------------------- */
  var modal = document.getElementById('trade-modal');
  if (!modal || !window.INSPECTOR || !INSPECTOR.trades.length) return;
  var B = INSPECTOR.bars, T = INSPECTOR.trades;
  var lastFocus = null;

  function closeModal() {
    modal.hidden = true;
    if (window.Plotly) Plotly.purge('modal-chart');
    if (lastFocus) lastFocus.focus();
  }

  function showTrade(i) {
    var t = T[i];
    if (!t || !window.Plotly) return;
    var lo = t.lo, hi = t.hi + 1;
    var x = B.t.slice(lo, hi).map(function (ms) { return new Date(ms); });
    var candles = {
      type: 'candlestick', name: 'Price',
      x: x,
      open: B.o.slice(lo, hi), high: B.h.slice(lo, hi),
      low: B.l.slice(lo, hi), close: B.c.slice(lo, hi),
      increasing: { line: { color: '%(up)s' }, fillcolor: '%(up)s' },
      decreasing: { line: { color: '%(down)s' }, fillcolor: '%(down)s' }
    };
    /* The strategy's own indicator series, sliced to the same window as the
       candles so a crossover is drawn on the bar it happened on. Colour, dash
       pattern and legend label all carry the distinction - "which one is the
       fast mean" has to survive greyscale. Nulls are the warm-up and are left
       as gaps rather than joined across. */
    var lines = (B.ind || []).map(function (s) {
      return {
        type: 'scatter', mode: 'lines', name: s.name,
        x: x, y: s.v.slice(lo, hi), connectgaps: false,
        line: { color: s.color, width: 1.7, dash: s.dash || 'solid' },
        hovertemplate: s.name + ' %%{y:,.2f}<extra></extra>'
      };
    });
    /* Shape AND label carry entry vs exit - the green/red pair alone is not
       separable under deuteranopia. */
    var marks = [
      { type: 'scatter', mode: 'markers+text', name: 'Entry',
        x: [new Date(B.t[t.e])], y: [B.o[t.e]],
        text: ['IN'], textposition: 'bottom center',
        textfont: { color: '%(entry)s', size: 11 },
        marker: { symbol: 'triangle-up', size: 15, color: '%(entry)s',
                  line: { color: '#12161d', width: 1.5 } },
        hovertemplate: 'Entry %%{x|%%Y-%%m-%%d %%H:%%M}<br>%%{y:,.2f}<extra></extra>' },
      { type: 'scatter', mode: 'markers+text', name: 'Exit',
        x: [new Date(B.t[t.x])], y: [B.o[t.x]],
        text: ['OUT'], textposition: 'top center',
        textfont: { color: '%(exit)s', size: 11 },
        marker: { symbol: 'triangle-down', size: 15, color: '%(exit)s',
                  line: { color: '#12161d', width: 1.5 } },
        hovertemplate: 'Exit %%{x|%%Y-%%m-%%d %%H:%%M}<br>%%{y:,.2f}<extra></extra>' }
    ];
    var row = ROWS[i];
    var cells = row ? row.cells : [];
    var fact = function (n) { return cells[n] ? cells[n].textContent.trim() : '—'; };
    document.getElementById('modal-title').textContent =
      'Trade ' + fact(0) + ' · ' + fact(1) + ' → ' + fact(2);
    document.getElementById('modal-facts').innerHTML =
      ['Entry ' + fact(3), 'Exit ' + fact(4), 'Return ' + fact(5) + '%%',
       'Fees ' + fact(6), 'Slippage ' + fact(7), 'Net P&L ' + fact(8)]
      .map(function (s) { return '<span>' + s + '</span>'; }).join('');

    modal.hidden = false;
    /* Markers last so they draw on top of the indicator lines. */
    Plotly.newPlot('modal-chart', [candles].concat(lines, marks), {
      template: 'plotly_dark', height: 460,
      showlegend: lines.length > 0,
      legend: { orientation: 'h', yanchor: 'bottom', y: 1.0, x: 0,
                font: { size: 11 }, bgcolor: 'rgba(0,0,0,0)' },
      margin: { l: 62, r: 20, t: lines.length ? 34 : 10, b: 40 },
      paper_bgcolor: '#12161d', plot_bgcolor: '#12161d',
      font: { family: 'ui-sans-serif, system-ui, sans-serif', size: 12,
              color: '#c8d1dc' },
      xaxis: { gridcolor: '#232a34', rangeslider: { visible: false } },
      yaxis: { gridcolor: '#232a34' }
    }, { displaylogo: false, responsive: true });
  }

  ROWS.forEach(function (tr) {
    var i = parseInt(tr.getAttribute('data-trade'), 10);
    if (isNaN(i)) return;
    tr.addEventListener('click', function () { lastFocus = tr; showTrade(i); });
    tr.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault(); lastFocus = tr; showTrade(i);
      }
    });
  });
  modal.addEventListener('click', function (e) {
    if (e.target === modal) closeModal();
  });
  document.getElementById('modal-close').addEventListener('click', closeModal);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !modal.hidden) closeModal();
  });
  window.showTrade = showTrade;      /* exercised by tests/test_report_gates */
  window.closeTrade = closeModal;
})();
""" % {"up": C_UP, "down": C_DOWN, "entry": C_ENTRY, "exit": C_EXIT}

_MODAL = """
<div class="modal" id="trade-modal" role="dialog" aria-modal="true"
     aria-labelledby="modal-title" hidden>
  <div class="modal-box">
    <div class="modal-head">
      <div>
        <h3 id="modal-title">Trade</h3>
        <div class="modal-facts" id="modal-facts"></div>
      </div>
      <button class="modal-close" id="modal-close" aria-label="Close">&#215;</button>
    </div>
    <div id="modal-chart"></div>
    <p class="dim">%d bars before the entry through %d after the exit.
      Prices are the raw bar values; the fill is the open of the bar the
      marker sits on. Any indicator lines are the strategy's own series, taken
      from the same arrays the signals were read off.</p>
  </div>
</div>
""" % (INSPECT_PRE, INSPECT_POST)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def generate_html_report(bars: pd.DataFrame | None,
                         result: Any,
                         metrics: dict,
                         gate_audit: dict | None,
                         out_path: str | Path,
                         strat_name: str | None = None,
                         strat_description: str | None = None,
                         version_label: str = "Version A",
                         max_trade_rows: int = MAX_TRADE_ROWS,
                         indicators: Any = None) -> Path:
    """
    Write a self-contained dark-themed HTML tear sheet for ONE version.

    Parameters
    ----------
    bars
        The symbol's OHLCV frame the run was made on. Powers the trade
        inspector; pass None and every other section still renders, with the
        inspector reported as unavailable rather than quietly missing.
    result
        A `BacktestResult`, or a dict carrying `returns`, `trades`, `equity`.
        The `config` on a BacktestResult is what fills the strategy logic
        card's execution half, so hand over the result rather than just its
        series when you have it.
    metrics
        The metrics dict from `agents.tier3_workers.summarize_result`.
    gate_audit
        The dict from `backtest.report.audit_acceptance_gates`, or None. None
        renders as "no gate audit was supplied" — never as a pass.
    out_path
        Destination `.html` file. Parent directories are created.
    strat_name, strat_description
        Header name and the prose above the logic card. Both default to what
        the metrics meta and the strategy module's own docstring say, so a
        caller that has nothing extra to add can leave them out.
    indicators
        `{name: series}` (or a DataFrame) of the strategy's own calculated
        series, each the full length of `bars`. Drawn over the candles in the
        trade inspector so the crossover or band touch behind a trade is
        visible next to the entry and exit markers. Pass the arrays the signals
        were actually taken from — recomputing them here would let the line and
        the signal disagree. A mis-sized series is dropped, not realigned.

    Returns the path written.
    """
    returns, trades, _equity = _unpack(result)
    daily = _daily_index(returns)

    meta = (metrics or {}).get("meta", {}) or {}
    capital = _num(meta.get("initial_capital"))
    if math.isnan(capital) or capital <= 0:
        capital = 100_000.0

    name = strat_name or meta.get("strategy") or "unnamed strategy"
    symbol = meta.get("symbol")
    description = strat_description
    if description is None:
        description = module_docstring(meta.get("strategy_path"))

    inspector = build_inspector(bars, trades.head(max_trade_rows)
                                if trades is not None else None,
                                indicators=indicators)
    inspectable = bool(inspector["trades"])

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    period = (f"{daily.index[0].date()} → {daily.index[-1].date()}"
              if len(daily) else "no daily returns")

    sections = [
        _gates_html(gate_audit),
        _metrics_html(metrics or {}, daily),
        _chart_html(daily, capital),
        _logic_card(metrics or {}, result, description, trades,
                    ml_filtered=_is_version_b(version_label)),
        _monthly_html(daily),
        _trades_html(trades, result, symbol, max_trade_rows, inspectable),
        _meta_html(metrics or {}, gate_audit),
    ]

    payload = (f'<script>window.INSPECTOR={json.dumps(inspector, separators=(",", ":"))};</script>'
               if inspectable else "")

    doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(name)} — {_esc(version_label)}</title>
<style>{_CSS}</style>
</head><body><div class="wrap">
<header>
  <h1>{_esc(name)} <span class="dim">·</span> {_esc(version_label)}</h1>
  <div class="sub">{_esc(period)} &nbsp;·&nbsp; {_esc(symbol or '—')}
    {_esc(meta.get('timeframe', ''))} &nbsp;·&nbsp; generated {_esc(generated)}</div>
</header>
{''.join(sections)}
<footer>
  Every figure on this page was computed by the deterministic backtest engine
  and formatted here. No model produced a number. In-sample results are not
  evidence of an edge until the 3-year holdout says so.
</footer>
</div>
{_MODAL if inspectable else ""}
{payload}
<script>{_JS}</script>
</body></html>"""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def write_dual_reports(dual: dict,
                       bars: pd.DataFrame | None = None,
                       out_dir: str | Path | None = None,
                       strat_name: str | None = None,
                       artifacts_root: str | Path = "/mnt/backtest/artifacts",
                       timestamp: str | None = None,
                       strat_description: str | None = None,
                       max_trade_rows: int = MAX_TRADE_ROWS,
                       indicators: Any = None,
                       prefix: str = "") -> dict[str, Any]:
    """
    Write `report_version_a.html` and `report_version_b.html` for a dual run.

    `dual` is what `agents.tier1_master.run_dual_version_backtest` returns.
    Both versions get a report whether or not either cleared a gate — a failing
    version is exactly the one somebody will want to read.

    `bars` is the frame both versions ran on. It is the same frame for A and B
    by construction — that is the whole point of the dual run — so one is
    passed to both reports and each extracts its own trades' windows from it.
    `indicators` are shared for the same reason: Version B filters Version A's
    entries, it does not recompute them, so both charts draw the same lines.

    `prefix` names the files `report_<prefix>_version_a.html` and
    `dual_metrics_<prefix>.json`. The multi-asset batch passes the symbol,
    because one directory holds one run and a run covers many contracts — an
    unprefixed NQ report and an unprefixed ES report written to the same folder
    would leave only the second, with nothing raising and a leaderboard row
    still pointing at both.

    `dual["version_b"]` may be None (`run_dual_version_backtest(ml=False)`).
    No Version B report is written then, and the snapshot records
    `version_b: null` rather than an empty metrics block — a reader must not be
    able to mistake a Version B that was never run for one that ran and scored
    nothing.

    Without `out_dir`, the destination is
    `<artifacts_root>/<strat_name>_<timestamp>/`. The timestamp is part of the
    directory rather than the filename so a re-run never overwrites the
    evidence a promotion decision was made on.
    """
    va, vb = dual["version_a"], dual.get("version_b")
    meta = dual.get("meta", {}) or {}
    name = strat_name or meta.get("strategy") or "strategy"
    tag = f"_{prefix}" if prefix else ""

    if out_dir is None:
        stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path(artifacts_root) / f"{name}_{stamp}"
    out_dir = Path(out_dir)

    paths = {
        "version_a": generate_html_report(
            bars, va.get("result"), va.get("metrics", {}), va.get("gate_audit"),
            out_dir / f"report{tag}_version_a.html", strat_name=name,
            strat_description=strat_description,
            version_label="Version A · rule-based", max_trade_rows=max_trade_rows,
            indicators=indicators),
    }
    if vb is not None:
        paths["version_b"] = generate_html_report(
            bars, vb.get("result"), vb.get("metrics", {}), vb.get("gate_audit"),
            out_dir / f"report{tag}_version_b.html", strat_name=name,
            strat_description=strat_description,
            version_label="Version B · ML-filtered", max_trade_rows=max_trade_rows,
            indicators=indicators)

    # The metrics snapshot promote.py locks into meta.json. Written next to the
    # reports so a promotion always cites numbers from a specific run rather
    # than whatever was on screen at the time.
    snapshot = {
        "meta": meta,
        "comparison": dual.get("comparison", {}),
        "version_a": {"metrics": _jsonable(va.get("metrics", {})),
                      "gate_audit": _jsonable(va.get("gate_audit"))},
        "version_b": ({"metrics": _jsonable(vb.get("metrics", {})),
                       "gate_audit": _jsonable(vb.get("gate_audit"))}
                      if vb is not None else None),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reports": {k: str(v) for k, v in paths.items()},
    }
    snap_path = out_dir / f"dual_metrics{tag}.json"
    snap_path.write_text(json.dumps(snapshot, indent=2, default=str),
                         encoding="utf-8")

    return {"dir": out_dir, "report_version_a": paths["version_a"],
            "report_version_b": paths.get("version_b"),
            "metrics_json": snap_path}


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
