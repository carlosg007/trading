#!/usr/bin/env python3
"""
app.py - CIO Command Center.

Location:  ~/src/trading/dashboard/app.py

Streamlit front end over the Pure Alpha research stack.

    streamlit run dashboard/app.py

Scope
-----
**Tab 1 - CIO Terminal.** Takes a natural-language alpha hypothesis and drives
`agents.tier1_master.run_campaign`: Gemini synthesises a `signal_fn` module,
`agents.tier3_workers.write_and_validate_strategy` audits it statically (AST
parse, import allowlist, forbidden builtins, negative-shift and reversed-slice
lookahead scan) before anything is imported, and the survivor is backtested by
`backtest.engine.run_backtest` over the real lake at
`/mnt/backtest/lake/futures/bars/`. The result is rendered as an Institutional
Pure Alpha Tear Sheet.

**Tab 2 - Strategy Vault.** Catalogues `strategies/experimental/` and
`strategies/approved_incubator/`. Modules are inspected with `ast`, never
imported: importing a module runs it, and this directory is where
model-generated code lands. The catalogue therefore reports what a module
declares, and checks it against the one strategy contract:

    signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, pd.Series]

**Tab 3 - CrossTrade Governance.** The prop-firm rulesets, shown as the
specification handed to CrossTrade NAM. They are NOT a research gate and no
PASS/FAIL verdict is rendered anywhere in this app. Account governance is
enforced against a live balance on the execution bridge; a backtest cannot
evaluate it, and a green tick here would only ever have been a statement about
a funding program rather than about an edge.

What is real and what is not
----------------------------
Every number on the tear sheet is computed by `agents.tier3_workers` from the
engine's own trade list and daily equity curve. No model produces a metric.

A campaign without a `GEMINI_API_KEY`, or one whose generated code fails the
audit, falls back to boilerplate whose signal logic is an explicitly labelled
placeholder. The UI says so on the panel, in the verdict, and on the tear sheet
itself - a command centre that renders a clean Sharpe over generated
boilerplate is how a research pipeline starts reporting conclusions nobody
reached. The same rule governs the vault: a strategy with no saved results gets
an empty panel, never a placeholder equity curve.

The campaign generator is iterated on Streamlit's own script thread. That is
why it is a generator: Streamlit re-executes this file per interaction, and a
worker thread loses its ScriptRunContext, so any st.* call from it writes into
a context that no longer exists. Yielding returns control between steps, so
progress renders normally without threading.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parent.parent

# Streamlit puts the SCRIPT's directory on sys.path, not the repo root, so
# `import agents...` resolves only by accident of the working directory.
# Without this the terminal tab breaks the moment the app is launched from
# anywhere other than ~/src/trading.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RULES_DIR = REPO / "compliance_rules"
INCUBATOR = REPO / "strategies" / "approved_incubator"
EXPERIMENTAL = REPO / "strategies" / "experimental"
LAKE = Path("/mnt/backtest/lake/futures/bars")

# Mirrors backtest.engine's default. Shown so the equity axis is never read as
# a percentage, and so a P&L figure has a stated denominator.
INITIAL_CAPITAL = 100_000.0

TIMEFRAMES = ["1d", "1w", "4h", "2h", "1h", "30m", "15m", "5m", "1m"]


# --------------------------------------------------------------------------
# Backend access
# --------------------------------------------------------------------------
def load_tier1() -> tuple[object | None, str | None]:
    """
    Import the master agent, returning the failure rather than raising.

    A broken backend must render as a named error inside the page. An uncaught
    ImportError at module scope takes the whole command centre down, including
    the vault and the governance spec, which are readable without it.
    """
    try:
        from agents import tier1_master
        return tier1_master, None
    except Exception as e:                                    # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@dataclass
class Ruleset:
    """One parsed compliance ruleset, or the reason it could not be parsed."""
    path: Path
    name: str
    data: dict | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def label(self) -> str:
        if not self.ok:
            return f"⚠ {self.path.name}"
        return str((self.data or {}).get("display_name") or self.name)


@dataclass
class StrategyEntry:
    """
    One catalogued strategy - either a bare module in `experimental/` or a
    staged directory in `approved_incubator/`.

    `contract_ok` is None when the module could not be parsed at all, which is
    a different state from a module that parsed and does not conform.
    """
    name: str
    path: Path
    stage: str                                   # "experimental" | "incubator"
    module_path: Path | None = None
    docstring: str | None = None
    entry_point: str | None = None               # signal_fn / make_signal_fn
    signature: str | None = None
    timeframe: str | None = None
    symbols: list | None = None
    default_params: dict | None = None
    contract_ok: bool | None = None
    contract_notes: list[str] = field(default_factory=list)
    contract_problems: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    error: str | None = None
    returns: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    equity: pd.DataFrame | None = None

    @property
    def has_results(self) -> bool:
        return self.returns is not None or self.equity is not None


@st.cache_data(show_spinner=False)
def load_rulesets(_dir: str) -> list[dict]:
    """
    Parse every ruleset in compliance_rules/.

    A malformed file is carried through as an error entry rather than dropped.
    Silently skipping an unparseable ruleset would let the governance spec
    appear to cover a constraint set that never loaded.
    """
    d = Path(_dir)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if not isinstance(data, dict):
                raise ValueError("top level is not a JSON object")
            out.append({"path": str(p), "name": p.stem, "data": data, "error": None})
        except Exception as e:                                # noqa: BLE001
            out.append({"path": str(p), "name": p.stem, "data": None,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def get_rulesets() -> list[Ruleset]:
    return [Ruleset(Path(r["path"]), r["name"], r["data"], r["error"])
            for r in load_rulesets(str(RULES_DIR))]


def _read_parquet(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(p) if p.exists() else None
    except Exception:                                         # noqa: BLE001
        return None


# -- static module inspection ----------------------------------------------
def _literal(node: ast.AST):
    """Best-effort literal, or None. A computed value is not worth executing."""
    try:
        return ast.literal_eval(node)
    except Exception:                                         # noqa: BLE001
        return None


def inspect_module(path: Path) -> dict:
    """
    Describe a strategy module WITHOUT importing it.

    `strategies/experimental/` is where model-generated code lands, and
    importing a module executes it. A catalogue that runs every candidate
    strategy on each Streamlit rerun is a code-execution surface disguised as a
    directory listing, so everything here comes from the AST.

    Reports the declared entry point and whether it matches the one contract
    the engine calls:

        signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, pd.Series]
    """
    out: dict = {"docstring": None, "entry_point": None, "signature": None,
                 "timeframe": None, "symbols": None, "default_params": None,
                 "contract_ok": None, "notes": [], "problems": [], "error": None}
    try:
        tree = ast.parse(path.read_text())
    except Exception as e:                                    # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["problems"].append("module does not parse - it cannot be loaded either")
        return out

    out["docstring"] = ast.get_docstring(tree)

    functions = {n.name: n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "TIMEFRAME":
                out["timeframe"] = _literal(node.value)
            elif target.id == "SYMBOLS":
                out["symbols"] = _literal(node.value)
            elif target.id == "DEFAULT_PARAMS":
                out["default_params"] = _literal(node.value)

    # make_signal_fn is the preferred form: the engine calls signal_fn(bars)
    # with no parameters, so anything parameterised needs the factory for
    # load_strategy to bind against.
    fn = functions.get("make_signal_fn") or functions.get("signal_fn")
    if fn is None:
        out["contract_ok"] = False
        out["problems"].append(
            "defines neither signal_fn nor make_signal_fn - load_strategy "
            "will refuse it")
        return out

    out["entry_point"] = fn.name
    try:
        out["signature"] = f"{fn.name}({ast.unparse(fn.args)})"
    except Exception:                                         # noqa: BLE001
        out["signature"] = f"{fn.name}(…)"

    notes: list[str] = []
    problems: list[str] = []

    # The function the ENGINE ends up calling. Usually module level, but the
    # preferred form is a factory returning a closure, and that closure is the
    # thing that receives `bars` - checking only module scope would report the
    # whole factory form as unverifiable.
    target = functions.get("signal_fn")
    if target is None and "make_signal_fn" in functions:
        target = next((n for n in ast.walk(functions["make_signal_fn"])
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and n is not functions["make_signal_fn"]), None)
        if target is None:
            notes.append("make_signal_fn returns something this static check "
                         "cannot follow - verify by hand that it takes `bars`")

    if target is not None:
        args = target.args
        positional = [a.arg for a in args.posonlyargs + args.args]
        if not positional:
            problems.append(f"`{target.name}` takes no positional argument; "
                            f"the engine calls it with one symbol's DataFrame")
        elif positional[0] != "bars":
            problems.append(
                f"`{target.name}`'s first parameter is `{positional[0]}`, not "
                f"`bars`. The engine passes ONE symbol's DataFrame; a module "
                f"that unpacks arrays in its signature is the pre-2026-08-15 "
                f"contract and is now wrong.")

    out["notes"] = notes
    out["problems"] = problems
    # Notes are informational. Only a real mismatch clears this flag - a
    # warning icon on a conforming module trains people to ignore the icon.
    out["contract_ok"] = not problems
    return out


def _catalogue_module(path: Path, stage: str) -> StrategyEntry:
    info = inspect_module(path)
    return StrategyEntry(
        name=path.stem, path=path, stage=stage, module_path=path,
        docstring=info["docstring"], entry_point=info["entry_point"],
        signature=info["signature"], timeframe=info["timeframe"],
        symbols=info["symbols"], default_params=info["default_params"],
        contract_ok=info["contract_ok"], contract_notes=info["notes"],
        contract_problems=info["problems"], error=info["error"],
    )


def get_experimental() -> list[StrategyEntry]:
    """Every candidate module in strategies/experimental/. Nothing here is a result."""
    if not EXPERIMENTAL.exists():
        return []
    return [_catalogue_module(p, "experimental")
            for p in sorted(EXPERIMENTAL.glob("*.py"))
            if p.name != "__init__.py"]


def get_incubator() -> list[StrategyEntry]:
    """
    Scan the incubator. Directories without meta.json are surfaced as
    incomplete rather than hidden - an unlabelled strategy is worse than a
    missing one.
    """
    if not INCUBATOR.exists():
        return []
    out: list[StrategyEntry] = []
    for d in sorted(x for x in INCUBATOR.iterdir() if x.is_dir()):
        if d.name.startswith((".", "__")):
            continue
        s = StrategyEntry(name=d.name, path=d, stage="incubator")
        meta_p = d / "meta.json"
        if not meta_p.exists():
            s.error = "no meta.json - cannot describe what this strategy is"
        else:
            try:
                s.meta = json.loads(meta_p.read_text())
            except Exception as e:                            # noqa: BLE001
                s.error = f"meta.json unreadable - {type(e).__name__}: {e}"

        modules = sorted(d.glob("*.py"))
        if modules:
            s.module_path = modules[0]
            info = inspect_module(s.module_path)
            s.docstring = info["docstring"]
            s.entry_point = info["entry_point"]
            s.signature = info["signature"]
            s.timeframe = info["timeframe"]
            s.symbols = info["symbols"]
            s.default_params = info["default_params"]
            s.contract_ok = info["contract_ok"]
            s.contract_notes = info["notes"]
            s.contract_problems = info["problems"]

        s.returns = _read_parquet(d / "returns.parquet")
        s.trades = _read_parquet(d / "trades.parquet")
        s.equity = _read_parquet(d / "equity.parquet")
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.2rem; max-width: 1400px; }
          h1, h2, h3 { letter-spacing: -0.01em; }
          [data-testid="stMetricValue"] {
              font-variant-numeric: tabular-nums;
              font-size: 1.45rem;
          }
          [data-testid="stMetricLabel"] {
              text-transform: uppercase;
              font-size: 0.68rem;
              letter-spacing: 0.09em;
              opacity: 0.7;
          }
          .cc-tag {
              display: inline-block; padding: 0.14rem 0.5rem; border-radius: 3px;
              font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em;
              text-transform: uppercase; font-family: ui-monospace, monospace;
          }
          .cc-on   { background: rgba(34,160,90,0.15);  color: #1f9254; }
          .cc-off  { background: rgba(200,60,60,0.15);  color: #c0392b; }
          .cc-warn { background: rgba(220,150,20,0.15); color: #b8860b; }
          .cc-rule {
              border-left: 3px solid rgba(128,128,128,0.35);
              padding: 0.35rem 0 0.35rem 0.7rem; margin-bottom: 0.55rem;
          }
          .cc-mono { font-family: ui-monospace, monospace; font-size: 0.8rem; opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tag(text: str, kind: str) -> str:
    return f'<span class="cc-tag cc-{kind}">{text}</span>'


def fmt(value, suffix: str = "", digits: int = 2) -> str:
    """
    Format a metric, refusing to render an undefined one as a number.

    NaN and inf are real outcomes here - no trades, no losing trades, a ruined
    account - and each would otherwise print as `nan` or a headline `inf`.
    """
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "—"
    if math.isinf(v):
        return "∞" if v > 0 else "-∞"
    return f"{v:,.{digits}f}{suffix}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
@dataclass
class RunSettings:
    """Everything the terminal needs to launch a campaign."""
    symbols: list[str]
    timeframe: str
    start: str
    end: str
    dual_version: bool = False


def render_sidebar(tier1) -> RunSettings:
    st.sidebar.title("CIO Command Center")
    st.sidebar.caption("Pure alpha discovery · Databento futures lake")
    st.sidebar.divider()

    st.sidebar.subheader("Campaign Universe")

    default_start = getattr(tier1, "DEFAULT_START", "2018-01-01")
    default_end = getattr(tier1, "DEFAULT_END", "2023-12-31")
    default_tf = getattr(tier1, "DEFAULT_TIMEFRAME", "1d")

    raw_symbols = st.sidebar.text_input(
        "Symbols", value="ES, NQ",
        help="Comma separated. Leave empty to let the router parse them out of "
             "the hypothesis text instead.",
    )
    symbols = [s.strip().upper() for s in raw_symbols.replace(",", " ").split()
               if s.strip()]

    timeframe = st.sidebar.selectbox(
        "Timeframe", TIMEFRAMES,
        index=TIMEFRAMES.index(default_tf) if default_tf in TIMEFRAMES else 0,
        help="Only 1m and 1d are stored; everything else is derived by the "
             "lake reader.",
    )

    c1, c2 = st.sidebar.columns(2)
    start = c1.date_input("Start", value=date.fromisoformat(default_start))
    end = c2.date_input("End", value=date.fromisoformat(default_end))

    dual_version = st.sidebar.checkbox(
        "Dual-version (A vs B)", value=False,
        help="After the campaign, re-run the staged strategy on the FIRST "
             "symbol as a rule-based baseline and again with the causal ML "
             "filter. Slower: it refits a classifier walk-forward.",
    )

    if len(symbols) == 1:
        st.sidebar.warning(
            "One symbol. A daily strategy on a single instrument over 16 years "
            "is ~100-200 trades — too thin to separate skill from luck.",
            icon="⚠️",
        )
    if start >= end:
        st.sidebar.error("Start is not before end — the campaign will find no bars.")

    st.sidebar.divider()
    st.sidebar.subheader("Environment")

    lake_ok = LAKE.exists()
    st.sidebar.markdown(
        f"{tag('lake mounted' if lake_ok else 'lake missing', 'on' if lake_ok else 'off')}",
        unsafe_allow_html=True)
    st.sidebar.caption(f"`{LAKE}`")
    if not lake_ok:
        st.sidebar.error(
            "The bars directory is not reachable. Campaigns will fail at the "
            "backtest step — nothing to read."
        )

    key_var = None
    if tier1 is not None:
        finder = getattr(tier1, "_find_api_key", None)
        key_var = finder() if callable(finder) else None
    st.sidebar.markdown(
        tag("gemini key found" if key_var else "no gemini key",
            "on" if key_var else "warn"),
        unsafe_allow_html=True)
    if not key_var:
        st.sidebar.caption(
            "Campaigns fall back to placeholder boilerplate. Set "
            "`GEMINI_API_KEY` to synthesise the hypothesis."
        )

    st.sidebar.divider()
    st.sidebar.caption(
        f"{len(get_experimental())} experimental · {len(get_incubator())} "
        f"staged · {len(get_rulesets())} governance ruleset(s)"
    )
    st.sidebar.caption(
        "Prop-firm governance is enforced by CrossTrade NAM against a live "
        "account, not here."
    )

    return RunSettings(symbols=symbols, timeframe=timeframe,
                       start=start.isoformat(), end=end.isoformat(),
                       dual_version=dual_version)


# --------------------------------------------------------------------------
# Tear sheet
# --------------------------------------------------------------------------
def _equity_series(artifacts: dict | None) -> pd.Series | None:
    eq = (artifacts or {}).get("equity")
    if eq is None or len(eq) == 0:
        return None
    return pd.Series(eq)


def equity_figure(equity: pd.Series) -> go.Figure:
    """Account equity in dollars, with the starting balance marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, mode="lines", name="Equity",
        line=dict(width=1.6, color="#1f77b4"),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=float(equity.iloc[0]), line_dash="dot",
                  line_color="rgba(128,128,128,0.6)",
                  annotation_text="initial capital",
                  annotation_position="bottom right")
    fig.update_layout(
        height=340, margin=dict(l=8, r=8, t=34, b=8),
        title="Equity curve (net of costs)", hovermode="x unified",
        xaxis_title=None, yaxis_title=None, showlegend=False,
    )
    return fig


def underwater_figure(equity: pd.Series) -> go.Figure:
    """
    Drawdown from the running high water mark.

    Plotted separately from equity because depth and DURATION are what get a
    strategy abandoned in live trading, and both are invisible on a rising
    equity curve.
    """
    dd = (equity / equity.cummax() - 1.0) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", fill="tozeroy",
        line=dict(width=1.0, color="#c0392b"),
        fillcolor="rgba(192,57,43,0.25)", name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=8, r=8, t=34, b=8),
        title="Underwater — drawdown from high water mark (%)",
        hovermode="x unified", xaxis_title=None, yaxis_title=None,
        showlegend=False,
    )
    return fig


def trade_distribution_figure(trades: pd.DataFrame) -> go.Figure | None:
    """Histogram of per-trade net P&L, split at breakeven."""
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return None
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return None

    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    fig = go.Figure()
    if not losses.empty:
        fig.add_trace(go.Histogram(x=losses.values, name="Losers",
                                   marker_color="#c0392b", opacity=0.75,
                                   nbinsx=60))
    if not wins.empty:
        fig.add_trace(go.Histogram(x=wins.values, name="Winners",
                                   marker_color="#1f9254", opacity=0.75,
                                   nbinsx=60))
    fig.add_vline(x=float(pnl.mean()), line_dash="dot",
                  line_color="rgba(128,128,128,0.8)",
                  annotation_text=f"mean {pnl.mean():,.0f}",
                  annotation_position="top right")
    fig.update_layout(
        height=300, margin=dict(l=8, r=8, t=34, b=8),
        title="Trade P&L distribution (net, dollars)",
        barmode="overlay", xaxis_title=None, yaxis_title="Trades",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return fig


def render_tear_sheet(run: dict) -> None:
    """
    The Institutional Pure Alpha Tear Sheet.

    Every figure here was computed by `agents.tier3_workers.summarize_result`
    from the engine's trade list and daily equity curve. Nothing on this panel
    came from a language model.
    """
    metrics = run.get("metrics") or {}
    meta = metrics.get("meta") or {}
    artifacts = run.get("artifacts") or {}
    equity = _equity_series(artifacts)
    trades = artifacts.get("trades")

    st.subheader("Pure Alpha Tear Sheet")
    st.caption(
        f"{', '.join(meta.get('symbols') or []) or '—'} · "
        f"{meta.get('timeframe', '—')} · {meta.get('start', '—')} → "
        f"{meta.get('end', '—')} · costs included · "
        f"initial capital ${meta.get('initial_capital', INITIAL_CAPITAL):,.0f}"
    )

    if run.get("strategy_is_placeholder"):
        st.error(
            "**These figures describe generated boilerplate, not your "
            "hypothesis.** The signal logic is a placeholder crossover that was "
            "staged because synthesis was unavailable or rejected. Read it as a "
            "test of the pipeline, not as a measurement of an edge.",
            icon="🚨",
        )

    if metrics.get("ruined"):
        st.error(
            "**Account ruined** — equity reached zero or below. Annualized "
            "figures are undefined and shown as `—`.",
            icon="🚨",
        )

    r1 = st.columns(3)
    r1[0].metric("Annualized Sharpe", fmt(metrics.get("sharpe")))
    r1[1].metric("Sortino", fmt(metrics.get("sortino")))
    r1[2].metric("Calmar", fmt(metrics.get("calmar")))

    st.caption(
        "Ratios are annualized on 252 trading days. Sortino uses the "
        "institutional denominator — squared shortfalls over ALL periods, so a "
        "strategy that is mostly flat is credited for the rarity of its losing "
        "days rather than judged only on their dispersion."
    )

    r2 = st.columns(3)
    r2[0].metric("Profit factor", fmt(metrics.get("profit_factor")))
    win = metrics.get("win_rate")
    r2[1].metric("Win rate",
                 fmt(win * 100 if isinstance(win, float) and not math.isnan(win)
                     else win, "%"))
    r2[2].metric("Total trades", f"{metrics.get('trade_count', 0):,}")

    r3 = st.columns(3)
    r3[0].metric("Max drawdown", fmt(metrics.get("max_drawdown_pct"), "%"))
    r3[1].metric("Total net return", fmt(metrics.get("total_return_pct"), "%"))
    r3[2].metric("CAGR", fmt(metrics.get("annualized_return_pct"), "%"))

    r4 = st.columns(3)
    r4[0].metric("Net P&L", fmt(metrics.get("total_pnl"), digits=0))
    r4[1].metric("Total costs", fmt(metrics.get("total_costs"), digits=0))
    r4[2].metric("Trading days", f"{metrics.get('n_days', 0):,}")

    variants = meta.get("variants_tested")
    st.caption(
        f"Search recorded: `variants_tested = {variants if variants is not None else 'unrecorded'}`. "
        f"A Sharpe read without knowing how many variants produced it is not a "
        f"measurement."
    )

    if metrics.get("trade_count", 0) == 0:
        st.info(
            "The strategy loaded and ran but never triggered. That is a result, "
            "not an error — there is nothing to plot.", icon="📄")
        return

    if equity is None:
        st.info("No equity curve was returned with this run.", icon="📄")
        return

    st.plotly_chart(equity_figure(equity), width="stretch")
    st.plotly_chart(underwater_figure(equity), width="stretch")

    dist = trade_distribution_figure(trades)
    if dist is not None:
        st.plotly_chart(dist, width="stretch")

    if trades is not None and not trades.empty:
        with st.expander(f"Trade log — {len(trades):,} trades"):
            # Bounded on purpose: a 1-minute cross-sectional run produces
            # hundreds of thousands of trades and st.dataframe will happily try
            # to ship all of them to the browser.
            st.caption("Showing the 500 most recent trades.")
            st.dataframe(trades.tail(500), hide_index=True, width="stretch")

    st.caption(
        "⚖️ No prop-firm compliance was evaluated. Trailing drawdown, daily "
        "loss, consistency and sizing are enforced by CrossTrade NAM against a "
        "live balance. These figures describe the edge only."
    )


# --------------------------------------------------------------------------
# Dual-Version Mandate: A vs B
# --------------------------------------------------------------------------
VERSION_A_COLOR = "#1f77b4"      # blue
VERSION_B_COLOR = "#10b981"      # emerald

# label, metrics key, formatting, and whether a higher number is better.
# "better" drives nothing but the delta's sign hint - drawdown improving is a
# smaller magnitude, and colouring it naively would call a deeper drawdown a win.
DUAL_ROWS = [
    ("Sharpe ratio", "sharpe", "ratio", True),
    ("Sortino ratio", "sortino", "ratio", True),
    ("Calmar ratio", "calmar", "ratio", True),
    ("Profit factor", "profit_factor", "ratio", True),
    ("Win rate", "win_rate", "pct_frac", True),
    ("Total trades", "trade_count", "int", None),
    ("Max drawdown", "max_drawdown_pct", "pct", False),
    ("Net return", "total_return_pct", "pct", True),
    ("CAGR", "annualized_return_pct", "pct", True),
]


def _dual_value(metrics: dict, key: str, kind: str) -> float:
    v = metrics.get(key)
    if kind == "pct_frac":
        try:
            return float(v) * 100.0
        except (TypeError, ValueError):
            return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def dual_comparison_table(a: dict, b: dict) -> pd.DataFrame:
    """Side-by-side metrics with the delta B - A."""
    rows = []
    for label, key, kind, _better in DUAL_ROWS:
        va, vb = _dual_value(a, key, kind), _dual_value(b, key, kind)
        delta = vb - va if not (math.isnan(va) or math.isnan(vb)) else float("nan")
        suffix = "%" if kind in ("pct", "pct_frac") else ""
        digits = 0 if kind == "int" else 2
        rows.append({
            "Metric": label,
            "A · rule-based": fmt(va, suffix, digits),
            "B · ML-filtered": fmt(vb, suffix, digits),
            "Delta (B − A)": ("—" if math.isnan(delta)
                              else f"{delta:+,.{digits}f}{suffix}"),
        })
    return pd.DataFrame(rows)


def dual_equity_figure(equity_a: pd.Series, equity_b: pd.Series) -> go.Figure:
    """
    Both equity curves on one axis.

    Overlaid rather than stacked side by side: the question is which curve is
    above the other and where they separated, and two panels with independent
    y-axes hide exactly that.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_a.index, y=equity_a.values, mode="lines",
        name="A · rule-based", line=dict(width=1.6, color=VERSION_A_COLOR),
        hovertemplate="%{x|%Y-%m-%d}<br>A  $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=equity_b.index, y=equity_b.values, mode="lines",
        name="B · ML-filtered", line=dict(width=1.6, color=VERSION_B_COLOR),
        hovertemplate="%{x|%Y-%m-%d}<br>B  $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=float(equity_a.iloc[0]), line_dash="dot",
                  line_color="rgba(128,128,128,0.6)",
                  annotation_text="initial capital",
                  annotation_position="bottom right")
    fig.update_layout(
        height=380, margin=dict(l=8, r=8, t=34, b=8),
        title="Version A vs Version B — equity, identical costs",
        hovermode="x unified", xaxis_title=None, yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return fig


def render_dual_version(dual: dict) -> None:
    """The Dual-Version Mandate comparison: rule-based baseline against the
    ML-filtered variant, same bars, same costs."""
    a = dual["version_a"]["metrics"]
    b = dual["version_b"]["metrics"]
    cmp_ = dual.get("comparison", {})
    meta = dual.get("meta", {})

    st.subheader("Dual-Version Mandate — A vs B")
    st.caption(
        f"{meta.get('symbol', '—')} · {meta.get('timeframe', '—')} · "
        f"{meta.get('bars', 0):,} bars · identical costs · "
        f"P(win) ≥ {meta.get('ml_threshold', 0.5):.2f}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Entries · A", f"{cmp_.get('entries_a', 0):,}")
    c2.metric("Entries · B", f"{cmp_.get('entries_b', 0):,}")
    c3.metric("Suppressed", f"{cmp_.get('entries_suppressed', 0):,}")

    st.dataframe(dual_comparison_table(a, b), hide_index=True, width="stretch")

    verdict = cmp_.get("b_beats_a")
    if verdict:
        st.success(
            "Version B beats Version A on Sharpe **in sample**. Under the "
            "Dual-Version Mandate that is not adoption: the filter is adopted "
            "only if B beats A on the held-back final 3 years.", icon="🧪")
    else:
        st.info(
            "Version B does not beat Version A on Sharpe. The mandate's answer "
            "is to keep the rule-based baseline — an ML filter that loses "
            "in-sample has nothing to prove out-of-sample.", icon="🧪")

    eq_a = dual["version_a"]["result"].equity
    eq_b = dual["version_b"]["result"].equity
    if eq_a is not None and len(eq_a) and eq_b is not None and len(eq_b):
        st.plotly_chart(dual_equity_figure(eq_a, eq_b), width="stretch")

    st.caption(
        "The filter passes entries through untouched until enough trades have "
        "closed to train on, so the two versions share their early history and "
        "are not independent samples. Every decision is fitted only on trades "
        "that closed strictly before the signal."
    )


# --------------------------------------------------------------------------
# Tab: CIO Terminal
# --------------------------------------------------------------------------
STATUS_ICONS = {
    "routing": "🧭", "planning": "📋", "generating": "🧱",
    "backtesting": "⚙️", "auditing": "🔎", "complete": "✅",
    "rejected": "🚫", "error": "❌", "warning": "⚠️",
}


def render_terminal(tier1, tier1_error: str | None, settings: RunSettings) -> None:
    st.subheader("CIO Terminal")

    tier_cols = st.columns(4)
    tiers = [
        ("Tier 1 · CIO", "tier1_master", "Routing / synthesis", "partial"),
        ("Tier 2 · Supervisors", "tier2_supervisors", "Robustness / lifecycle", "partial"),
        ("Tier 3 · Workers", "tier3_workers", "Audit / backtest / metrics", "partial"),
        ("Monitor", "system_monitor", "RAM / runaway loops", "off"),
    ]
    labels = {"partial": ("live", "on"), "off": ("not implemented", "off")}
    for col, (label, module, role, state) in zip(tier_cols, tiers):
        text, kind = labels[state]
        with col:
            st.markdown(f"**{label}**")
            st.markdown(tag(text, kind), unsafe_allow_html=True)
            st.caption(f"`agents/{module}.py` — {role}")

    st.caption(
        "Describe an alpha hypothesis in plain English. The campaign "
        "synthesises a `signal_fn(bars, **params)` module with Gemini, audits "
        "it statically (import allowlist, forbidden builtins, negative-shift "
        "and reversed-slice lookahead), smoke-tests it, then backtests it over "
        "the real lake with costs applied. Universe and window come from the "
        "sidebar."
    )

    if tier1_error:
        st.error(
            f"**Master Agent unavailable** — `agents.tier1_master` could not be "
            f"imported: {tier1_error}"
        )
        return

    st.divider()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("ts"):
                st.caption(msg["ts"])

    if not st.session_state.chat:
        st.caption(
            "No hypotheses submitted yet. Try: *“go long when the 20-day "
            "momentum is positive and volatility is contracting”*, *“what's in "
            "the vault?”*, or *“what can you do?”*"
        )

    prompt = st.chat_input("Describe an alpha hypothesis…")
    if prompt:
        _run_prompt(tier1, prompt, settings)

    last = st.session_state.get("last_run")
    if last and last.get("ran_backtest"):
        st.divider()
        render_tear_sheet(last)

    dual = st.session_state.get("last_dual")
    if dual:
        st.divider()
        if dual.get("error"):
            st.error(f"**Dual-version run failed** — {dual['error']}")
        else:
            render_dual_version(dual)

    if st.session_state.chat and st.button("Clear transcript", type="secondary"):
        st.session_state.chat = []
        st.session_state.pop("last_run", None)
        st.rerun()


def _run_prompt(tier1, prompt: str, settings: RunSettings) -> None:
    """
    Drive one campaign, rendering each yielded event as it lands.

    The generator is iterated on Streamlit's own script thread. That is the
    point of the generator: a worker thread would lose its ScriptRunContext and
    any st.* call from it writes into a context that no longer exists. Here
    control returns to the script between steps, so progress renders normally.

    Nothing is written to the transcript until the campaign produces a final
    payload. A half-finished run left in the history would read as a result.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    st.session_state.chat.append({"role": "user", "content": prompt, "ts": now})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final: dict | None = None
        with st.status("Working…", expanded=True) as status:
            try:
                for event in tier1.run_campaign(
                        prompt,
                        symbols=settings.symbols or None,
                        start_date=settings.start,
                        end_date=settings.end,
                        timeframe=settings.timeframe,
                        include_artifacts=True):
                    icon = STATUS_ICONS.get(event.get("status", ""), "•")
                    st.write(f"{icon} {event.get('message', '')}")
                    final = event
            except Exception as e:                            # noqa: BLE001
                # The generator raising is a real failure, not a verdict. Say
                # so plainly rather than leaving a spinner that never resolves.
                body = f"**Campaign crashed** — {type(e).__name__}: {e}"
                status.update(label="Failed", state="error")
                st.error(body)
                st.session_state.chat.append(
                    {"role": "assistant", "content": body, "ts": now})
                st.session_state.pop("last_run", None)
                return

            outcome = (final or {}).get("status")
            status.update(
                label={"complete": "Done", "rejected": "Not run",
                       "error": "Failed"}.get(outcome, "Finished"),
                state="error" if outcome == "error" else "complete",
                expanded=False,
            )

        body = (final or {}).get("response") or (
            "The campaign produced no final payload. Nothing was concluded.")
        st.markdown(body)
        st.caption(now)

    st.session_state.chat.append(
        {"role": "assistant", "content": body, "ts": now})

    # Keep the tear sheet only for a run that actually reached the engine. A
    # rejected or conversational turn must not leave the previous campaign's
    # metrics on screen next to a new answer.
    if (final or {}).get("ran_backtest"):
        st.session_state.last_run = {
            "prompt": prompt, "ts": now,
            "metrics": final.get("metrics") or {},
            "artifacts": final.get("artifacts") or {},
            "strategy_path": final.get("strategy_path"),
            "strategy_is_placeholder": final.get("strategy_is_placeholder", True),
            "ran_backtest": True,
        }
    else:
        st.session_state.pop("last_run", None)

    st.session_state.pop("last_dual", None)
    if settings.dual_version and (final or {}).get("ran_backtest"):
        _run_dual_version(tier1, final, settings)

    st.rerun()


def _run_dual_version(tier1, final: dict, settings: RunSettings) -> None:
    """
    Re-run the staged strategy as Version A and Version B on one symbol.

    Deliberately one symbol: `run_dual_version_backtest` takes a single frame,
    and handing it an interleaved multi-symbol frame is the averaging-across-
    contracts bug the engine has no entry point for. The symbol comes from the
    campaign's own metadata rather than the sidebar, so it matches what was
    actually backtested even when the router parsed the symbols out of the
    prompt.
    """
    meta = (final.get("metrics") or {}).get("meta") or {}
    symbols = meta.get("symbols") or settings.symbols
    strategy_path = final.get("strategy_path")
    if not symbols or not strategy_path:
        st.session_state.last_dual = {
            "error": "the campaign reported no symbol or staged module to re-run"}
        return

    symbol = str(symbols[0])
    tf = meta.get("timeframe") or settings.timeframe
    with st.status(f"Dual-version on {symbol} {tf} — fitting walk-forward…",
                   expanded=True) as status:
        try:
            from mdlib.lake import iter_bars
            st.write(f"📥 Reading {symbol} {tf} bars…")
            bars = None
            for sym, g in iter_bars([symbol], tf, meta.get("start"),
                                    meta.get("end")):
                if sym == symbol:
                    bars = g.reset_index(drop=True)
                    break
            if bars is None or bars.empty:
                raise ValueError(f"no {tf} bars returned for {symbol}")

            st.write(f"⚙️ {len(bars):,} bars · running A, then the ML filter…")
            dual = tier1.run_dual_version_backtest(
                strategy_path, bars, freq=tf, symbol=symbol)
            st.session_state.last_dual = dual
            status.update(label="Dual-version complete", state="complete",
                          expanded=False)
        except Exception as e:                                # noqa: BLE001
            st.session_state.last_dual = {"error": f"{type(e).__name__}: {e}"}
            status.update(label="Dual-version failed", state="error")


# --------------------------------------------------------------------------
# Tab: Strategy Vault
# --------------------------------------------------------------------------
def render_contract(s: StrategyEntry) -> None:
    """Show what the module declares and whether it matches the engine's call."""
    if s.entry_point is None and s.module_path is None:
        st.caption("No Python module in this directory — nothing to inspect.")
        return

    if s.contract_ok is None:
        st.warning(f"Module could not be parsed: {s.error}", icon="⚠️")
    elif s.contract_ok:
        st.markdown(tag("contract ok", "on"), unsafe_allow_html=True)
    else:
        st.markdown(tag("contract mismatch", "off"), unsafe_allow_html=True)

    if s.signature:
        st.code(f"def {s.signature}", language="python")
    for problem in s.contract_problems:
        st.error(problem, icon="🚨")
    for note in s.contract_notes:
        st.caption(f"ℹ️ {note}")


def render_strategy(s: StrategyEntry, rulesets: list[Ruleset]) -> None:
    meta = s.meta or {}
    title = meta.get("name") or s.name
    version = meta.get("version")
    header = f"{title}" + (f"  ·  Version {version}" if version else "")

    with st.expander(header, expanded=False):
        if s.error:
            st.error(f"**{s.name}** — {s.error}")

        description = meta.get("description") or s.docstring
        if description:
            st.write(description.strip().split("\n\n")[0])

        cols = st.columns(4)
        symbols = meta.get("symbols") or s.symbols or []
        cols[0].metric("Symbols", ", ".join(map(str, symbols)) if symbols else "—")
        cols[1].metric("Timeframe", meta.get("timeframe") or s.timeframe or "—")
        cols[2].metric("Entry point", s.entry_point or "—")
        cols[3].metric("Results", "yes" if s.has_results else "no")

        render_contract(s)

        if s.default_params:
            st.caption("Declared defaults")
            st.json(s.default_params, expanded=False)

        if s.stage == "incubator":
            variants = meta.get("variants_tested")
            costs = meta.get("costs_included")
            c1, c2 = st.columns(2)
            c1.metric("Variants tested", variants if variants is not None else "—")
            c2.metric("Costs included",
                      "yes" if costs is True else "no" if costs is False else "—")
            if variants is None or costs is None:
                st.warning(
                    "`variants_tested` or `costs_included` missing from "
                    "`meta.json`. A Sharpe ratio cannot be judged without "
                    "knowing how many variants it was selected from and whether "
                    "costs were applied.",
                    icon="⚠️",
                )

            rid = meta.get("ruleset_id")
            if rid:
                known = {r.name for r in rulesets}
                if rid in known:
                    st.caption(
                        f"References governance ruleset `{rid}` — a CrossTrade "
                        f"execution constraint, not a research gate.")
                else:
                    st.error(
                        f"`meta.json` references ruleset `{rid}`, which is not "
                        f"present in `compliance_rules/`. The stated "
                        f"constraints cannot be reproduced."
                    )

            st.markdown("**Saved results**")
            if s.returns is None and s.trades is None:
                st.info(
                    "No saved results in this directory. Add `returns.parquet` "
                    "/ `trades.parquet` from a `BacktestResult`.", icon="📄")
            else:
                summary = []
                if s.returns is not None:
                    summary.append(("Return rows", f"{len(s.returns):,}"))
                if s.trades is not None:
                    summary.append(("Trades", f"{len(s.trades):,}"))
                st.dataframe(pd.DataFrame(summary, columns=["Metric", "Value"]),
                             hide_index=True, width="stretch")

        if s.module_path is not None:
            rel = s.module_path.relative_to(REPO)
            st.caption(f"`{rel}`")
            with st.expander("Source"):
                # Read, not imported. Importing a module executes it, and this
                # is where model-generated code lands.
                try:
                    st.code(s.module_path.read_text(), language="python")
                except Exception as e:                        # noqa: BLE001
                    st.error(f"Could not read the module: {type(e).__name__}: {e}")


def render_vault(rulesets: list[Ruleset]) -> None:
    st.subheader("Strategy Vault")

    experimental = get_experimental()
    incubator = get_incubator()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experimental", len(experimental))
    c2.metric("Staged", len(incubator))
    c3.metric("With results", len([s for s in incubator if s.has_results]))
    nonconforming = [s for s in experimental + incubator if s.contract_ok is False]
    c4.metric("Contract issues", len(nonconforming))

    if nonconforming:
        st.warning(
            f"{len(nonconforming)} module(s) do not match "
            f"`signal_fn(bars, **params)`: "
            + ", ".join(f"`{s.name}`" for s in nonconforming)
            + ". `load_strategy` will reject or mis-bind them.",
            icon="⚠️",
        )

    st.divider()

    st.markdown("#### `strategies/experimental/`")
    st.caption(
        "Candidate modules, including everything Gemini has synthesised. "
        "**Nothing here has been evaluated** — a file in this directory is a "
        "hypothesis someone typed."
    )
    if not experimental:
        st.info("No candidate modules staged yet.", icon="🧪")
    for s in experimental:
        render_strategy(s, rulesets)

    st.divider()

    st.markdown("#### `strategies/approved_incubator/`")
    st.caption(
        "Promoted only after a run with costs across a symbol list that "
        "survives walk-forward and sensitivity checks. One directory per "
        "strategy, each with a `meta.json`."
    )
    if not INCUBATOR.exists():
        st.error(f"`{INCUBATOR}` does not exist.")
        return
    if not incubator:
        st.info(
            "No strategies staged yet. See the directory's `README.md` for the "
            "expected layout.", icon="🗄️")
        return

    incomplete = [s for s in incubator if s.error]
    if incomplete:
        st.warning(
            f"{len(incomplete)} strategy directory(ies) missing or with "
            f"unreadable `meta.json`: "
            + ", ".join(f"`{s.name}`" for s in incomplete)
        )
    for s in incubator:
        render_strategy(s, rulesets)


# --------------------------------------------------------------------------
# Tab: CrossTrade Governance
# --------------------------------------------------------------------------
def render_governance(rulesets: list[Ruleset]) -> None:
    st.subheader("CrossTrade Governance Spec")
    st.info(
        "**This is not a research gate.** Trailing drawdown, daily loss, "
        "consistency and contract sizing are enforced by CrossTrade NAM on the "
        "Windows execution bridge, against a live account balance. These "
        "rulesets are the declarative specification handed to it. A backtest "
        "cannot evaluate them, so this app renders no compliance verdict.",
        icon="⚖️",
    )

    if not RULES_DIR.exists():
        st.error(f"`compliance_rules/` not found at `{RULES_DIR}`.")
        return
    if not rulesets:
        st.warning("No rulesets found. Add a `.json` file to `compliance_rules/`.")
        return

    broken = [r for r in rulesets if not r.ok]
    if broken:
        st.error(
            f"{len(broken)} ruleset file(s) failed to parse: "
            + ", ".join(f"`{b.path.name}`" for b in broken))

    # Default to the first ruleset that actually parsed. Files sort
    # alphabetically, so without this a single malformed JSON whose name sorts
    # early becomes the default selection and the tab opens in an error state.
    default = next((i for i, r in enumerate(rulesets) if r.ok), 0)
    idx = st.selectbox("Ruleset", range(len(rulesets)), index=default,
                       format_func=lambda i: rulesets[i].label)
    rs = rulesets[idx]

    if not rs.ok:
        st.error(f"`{rs.path.name}` could not be parsed:\n\n{rs.error}")
        return

    data = rs.data or {}
    st.markdown(f"**{rs.label}** · `{rs.path.name}`")
    if data.get("description"):
        st.caption(data["description"])

    acct = data.get("account") or {}
    if acct.get("initial_balance") is not None:
        st.caption(f"Account basis: {acct.get('currency', '')} "
                   f"{acct['initial_balance']:,.0f}".strip())

    rules = data.get("rules")
    if not isinstance(rules, dict) or not rules:
        st.warning("This ruleset defines no `rules`.")
        return

    rows = []
    for key, r in rules.items():
        if not isinstance(r, dict):
            continue
        enf = r.get("enforcement") or {}
        rows.append({
            "Rule": key.replace("_", " ").title(),
            "Value": (f"{r.get('value')}%" if r.get("unit") == "percent"
                      else f"{r.get('value')} {r.get('unit', '')}".strip()),
            "Basis": r.get("basis", "—"),
            "Status": str(enf.get("status", "UNKNOWN")).replace("_", " "),
            "Engine field": enf.get("engine_field") or "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    for key, r in rules.items():
        if not isinstance(r, dict):
            continue
        enf = r.get("enforcement") or {}
        detail = enf.get("detail")
        caveat = enf.get("modelling_caveat")
        if detail or caveat:
            with st.expander(f"{key.replace('_', ' ').title()} — enforcement notes"):
                if r.get("description"):
                    st.write(r["description"])
                if detail:
                    st.markdown(f"**Status:** `{enf.get('status', 'UNKNOWN')}` — {detail}")
                if caveat:
                    st.warning(caveat, icon="⚠️")
                for note in r.get("evaluation_notes") or []:
                    st.caption(f"• {note}")

    if str(data.get("provenance", {}).get("verification_status", "")).startswith("UNVERIFIED"):
        st.info("Ruleset values are unverified against the provider.")

    summary = data.get("enforcement_summary") or {}
    if summary.get("warning"):
        st.error(summary["warning"], icon="🚨")


# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="CIO Command Center",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    tier1, tier1_error = load_tier1()
    settings = render_sidebar(tier1)
    rulesets = get_rulesets()

    st.title("CIO Command Center")
    st.caption(
        "Pure alpha discovery over the Databento futures lake · synthesis is "
        "audited before it runs, and every metric is computed deterministically "
        "from the engine's own trades"
    )

    tab_cmd, tab_vault, tab_rules = st.tabs(
        ["CIO Terminal", "Strategy Vault", "CrossTrade Governance"]
    )
    with tab_cmd:
        render_terminal(tier1, tier1_error, settings)
    with tab_vault:
        render_vault(rulesets)
    with tab_rules:
        render_governance(rulesets)


# Streamlit executes this file top to bottom on every rerun, so main() is
# called unconditionally. No sys.exit() here: raising SystemExit out of a
# Streamlit script aborts the run rather than returning a shell status.
main()
