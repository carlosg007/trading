#!/usr/bin/env python3
"""
app.py - CIO Command Center.

Location:  ~/src/trading/dashboard/app.py

Streamlit front end over the agent tiers, the compliance rulesets, and the
strategy incubator.

    streamlit run dashboard/app.py

Scope
-----
Layout, file reading, and error handling only. **The agent backend is mocked.**
agents/tier1_master.py raises NotImplementedError on every entry point, and
this app does not pretend otherwise: the chat panel is visibly labelled as a
mock and returns a fixed acknowledgement rather than anything resembling an
analysis.

That labelling is the point. A command centre that renders plausible agent
replies over a backend that does not exist is how a research pipeline starts
reporting conclusions nobody reached. The same rule governs the Strategy Vault:
a strategy with no saved results gets an empty panel, never a placeholder
equity curve that could be mistaken for a result.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
RULES_DIR = REPO / "compliance_rules"
INCUBATOR = REPO / "strategies" / "approved_incubator"

MOCK_BANNER = "Agent backend is not implemented. Replies below are mocked."


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
class Strategy:
    """One incubator strategy directory."""
    path: Path
    name: str
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
    Silently skipping an unparseable ruleset would let a compliance run appear
    to cover a constraint set that never loaded.
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
        except Exception as e:
            out.append({"path": str(p), "name": p.stem, "data": None,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def get_rulesets() -> list[Ruleset]:
    return [Ruleset(Path(r["path"]), r["name"], r["data"], r["error"])
            for r in load_rulesets(str(RULES_DIR))]


def _read_parquet(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(p) if p.exists() else None
    except Exception:
        return None


def get_strategies() -> list[Strategy]:
    """
    Scan the incubator. Directories without meta.json are surfaced as
    incomplete rather than hidden - an unlabelled strategy is worse than a
    missing one.
    """
    if not INCUBATOR.exists():
        return []
    out: list[Strategy] = []
    for d in sorted(x for x in INCUBATOR.iterdir() if x.is_dir()):
        if d.name.startswith((".", "__")):
            continue
        s = Strategy(path=d, name=d.name)
        meta_p = d / "meta.json"
        if not meta_p.exists():
            s.error = "no meta.json - cannot describe what this strategy is"
        else:
            try:
                s.meta = json.loads(meta_p.read_text())
            except Exception as e:
                s.error = f"meta.json unreadable - {type(e).__name__}: {e}"
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


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> Ruleset | None:
    st.sidebar.title("CIO Command Center")
    st.sidebar.caption("Prop-firm constrained research")
    st.sidebar.divider()
    st.sidebar.subheader("Compliance Ruleset")

    if not RULES_DIR.exists():
        st.sidebar.error(
            f"`compliance_rules/` not found at `{RULES_DIR}`.\n\n"
            "Create it and add a ruleset JSON."
        )
        return None

    rulesets = get_rulesets()
    if not rulesets:
        st.sidebar.warning(
            "No rulesets found. Add a `.json` file to `compliance_rules/`."
        )
        return None

    broken = [r for r in rulesets if not r.ok]
    labels = [r.label for r in rulesets]

    # Default to the first ruleset that actually parsed. Files sort
    # alphabetically, so without this a single malformed JSON whose name sorts
    # early becomes the default selection and the whole command centre opens in
    # an error state. Broken files stay listed and stay loud - they are just not
    # what you land on.
    default = next((i for i, r in enumerate(rulesets) if r.ok), 0)
    idx = st.sidebar.selectbox(
        "Active ruleset", range(len(rulesets)),
        index=default,
        format_func=lambda i: labels[i],
        help="Detected from compliance_rules/*.json",
    )
    chosen = rulesets[idx]

    if broken:
        st.sidebar.error(
            f"{len(broken)} ruleset file(s) failed to parse: "
            + ", ".join(f"`{b.path.name}`" for b in broken)
        )

    if not chosen.ok:
        st.sidebar.error(f"`{chosen.path.name}` could not be parsed:\n\n{chosen.error}")
        return chosen

    render_ruleset_summary(chosen)
    st.sidebar.divider()
    st.sidebar.caption(
        f"{len(rulesets)} ruleset(s) · {len(get_strategies())} incubator strategy(ies)"
    )
    return chosen


def render_ruleset_summary(rs: Ruleset) -> None:
    """Sidebar detail for the selected ruleset, enforcement status foremost."""
    data = rs.data or {}
    rules = data.get("rules")
    if not isinstance(rules, dict) or not rules:
        st.sidebar.warning("Ruleset has no `rules` block — nothing to display.")
        return

    acct = data.get("account") or {}
    if acct.get("initial_balance") is not None:
        st.sidebar.caption(
            f"Account basis: {acct.get('currency', '')} "
            f"{acct['initial_balance']:,.0f}".strip()
        )

    enforced, unenforced = [], []
    for key, r in rules.items():
        if not isinstance(r, dict):
            continue
        status = str((r.get("enforcement") or {}).get("status", "UNKNOWN")).upper()
        (enforced if status == "ENFORCED" else unenforced).append((key, r, status))

    for key, r, status in enforced + unenforced:
        value, unit = r.get("value"), r.get("unit", "")
        shown = f"{value}%" if unit == "percent" else f"{value} {unit}".strip()
        kind = {"ENFORCED": "on", "NOT_ENFORCED": "off"}.get(status, "warn")
        st.sidebar.markdown(
            f'<div class="cc-rule"><b>{key.replace("_", " ").title()}</b> '
            f'<span class="cc-mono">{shown}</span><br>{tag(status.replace("_", " "), kind)}</div>',
            unsafe_allow_html=True,
        )

    if unenforced:
        st.sidebar.warning(
            f"**{len(unenforced)} of {len(enforced) + len(unenforced)} rules are not "
            f"enforced by the engine.** A clean backtest is not evidence of "
            f"compliance with them."
        )

    if str(data.get("provenance", {}).get("verification_status", "")).startswith("UNVERIFIED"):
        st.sidebar.info("Ruleset values are unverified against the provider.")


# --------------------------------------------------------------------------
# Tab: Command Center
# --------------------------------------------------------------------------
def mock_agent_reply(command: str, rs: Ruleset | None) -> str:
    """
    Stand-in for agents.tier1_master.

    Deliberately does not analyse, estimate, or speculate about the command.
    It echoes what was received and states what would happen, so nothing here
    can be mistaken for a result the system actually produced.
    """
    name = rs.label if rs and rs.ok else "none selected"
    return (
        f"**Mock acknowledgement — no agent ran.**\n\n"
        f"Received: `{command}`\n\n"
        f"Active ruleset: **{name}**\n\n"
        f"Wired up, this would dispatch to `agents.tier1_master.propose_goals()`, "
        f"which currently raises `NotImplementedError`. No backtest was queued, "
        f"no data was read, and no conclusion was formed."
    )


def render_command_center(rs: Ruleset | None) -> None:
    st.subheader("Master Agent")
    st.warning(MOCK_BANNER, icon="🧪")

    tier_cols = st.columns(4)
    tiers = [
        ("Tier 1 · CIO", "tier1_master", "Goal generation"),
        ("Tier 2 · Supervisors", "tier2_supervisors", "Compliance / OOS"),
        ("Tier 3 · Workers", "tier3_workers", "Backtest execution"),
        ("Monitor", "system_monitor", "RAM / runaway loops"),
    ]
    for col, (label, module, role) in zip(tier_cols, tiers):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(tag("not implemented", "off"), unsafe_allow_html=True)
            st.caption(f"`agents/{module}.py` — {role}")

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
            "No commands issued yet. Try: *“Propose five mean-reversion "
            "hypotheses for the rates complex.”*"
        )

    prompt = st.chat_input("Send a command to the Master Agent…")
    if prompt:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        st.session_state.chat.append({"role": "user", "content": prompt, "ts": now})
        st.session_state.chat.append(
            {"role": "assistant", "content": mock_agent_reply(prompt, rs), "ts": now}
        )
        st.rerun()

    if st.session_state.chat and st.button("Clear transcript", type="secondary"):
        st.session_state.chat = []
        st.rerun()


# --------------------------------------------------------------------------
# Tab: Strategy Vault
# --------------------------------------------------------------------------
def equity_figure(strategy: Strategy) -> go.Figure | None:
    """
    Plotly equity curve from saved results. Returns None when there is nothing
    real to plot - the caller renders an empty state rather than a fake curve.
    """
    df = strategy.equity if strategy.equity is not None else strategy.returns
    if df is None or df.empty:
        return None

    frame = df.reset_index()
    time_col = next((c for c in frame.columns
                     if str(c).lower() in ("ts", "date", "datetime", "index")), None)
    value_col = next((c for c in frame.columns
                      if str(c).lower() in ("equity", "value", "cum", "cumulative")), None)

    if value_col is None:
        numeric = frame.select_dtypes("number").columns
        ret_col = next((c for c in numeric
                        if str(c).lower() in ("ret", "return", "returns", "pnl")), None)
        if ret_col is None:
            return None
        frame["_equity"] = (1.0 + frame[ret_col].fillna(0)).cumprod()
        value_col = "_equity"

    if time_col is None:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame[time_col], y=frame[value_col], mode="lines",
        name="Equity", line=dict(width=1.6),
    ))
    fig.update_layout(
        height=340, margin=dict(l=8, r=8, t=28, b=8),
        title="Equity curve", hovermode="x unified",
        xaxis_title=None, yaxis_title=None, showlegend=False,
    )
    return fig


def render_strategy(s: Strategy, rulesets: list[Ruleset]) -> None:
    meta = s.meta or {}
    title = meta.get("name") or s.name
    version = meta.get("version")
    header = f"{title}" + (f"  ·  Version {version}" if version else "")

    with st.expander(header, expanded=False):
        if s.error:
            st.error(f"**{s.name}** — {s.error}")

        if meta.get("description"):
            st.write(meta["description"])

        cols = st.columns(4)
        cols[0].metric("Symbols", len(meta.get("symbols") or []) or "—")
        cols[1].metric("Timeframe", meta.get("timeframe") or "—")
        variants = meta.get("variants_tested")
        cols[2].metric("Variants tested", variants if variants is not None else "—")
        costs = meta.get("costs_included")
        cols[3].metric("Costs included",
                       "yes" if costs is True else "no" if costs is False else "—")

        if variants is None or costs is None:
            st.warning(
                "`variants_tested` or `costs_included` missing from `meta.json`. "
                "A Sharpe ratio cannot be judged without knowing how many variants "
                "it was selected from and whether costs were applied.",
                icon="⚠️",
            )

        rid = meta.get("ruleset_id")
        if rid:
            known = {r.name for r in rulesets}
            if rid in known:
                st.caption(f"Evaluated against ruleset `{rid}`.")
            else:
                st.error(
                    f"`meta.json` references ruleset `{rid}`, which is not present "
                    f"in `compliance_rules/`. The stated constraints cannot be "
                    f"reproduced."
                )

        st.markdown("**Backtest summary**")
        if s.returns is None and s.trades is None:
            st.info(
                "No saved results in this directory. Add `returns.parquet` / "
                "`trades.parquet` from a `BacktestResult`.",
                icon="📄",
            )
        else:
            summary = []
            if s.returns is not None:
                summary.append(("Return rows", f"{len(s.returns):,}"))
            if s.trades is not None:
                summary.append(("Trades", f"{len(s.trades):,}"))
            st.dataframe(
                pd.DataFrame(summary, columns=["Metric", "Value"]),
                hide_index=True, width="stretch",
            )

        fig = equity_figure(s)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption(
                "No equity curve — nothing has been run for this strategy. "
                "Vectorbt Pro figures render here once results are saved."
            )


def render_vault(rulesets: list[Ruleset]) -> None:
    st.subheader("Strategy Vault")
    st.caption(f"Reading `{INCUBATOR.relative_to(REPO)}`")

    if not INCUBATOR.exists():
        st.error(
            f"`{INCUBATOR}` does not exist. Create it and add one directory per "
            f"strategy, each with a `meta.json`."
        )
        return

    strategies = get_strategies()
    if not strategies:
        st.info(
            "No strategies staged yet. Each strategy is a directory containing "
            "`meta.json` and, once run, `returns.parquet` / `trades.parquet`. "
            "See the directory's `README.md` for the expected layout.",
            icon="🗄️",
        )
        return

    incomplete = [s for s in strategies if s.error]
    with_results = [s for s in strategies if s.has_results]
    c1, c2, c3 = st.columns(3)
    c1.metric("Staged", len(strategies))
    c2.metric("With results", len(with_results))
    c3.metric("Incomplete", len(incomplete))

    if incomplete:
        st.warning(
            f"{len(incomplete)} strategy directory(ies) missing or with unreadable "
            f"`meta.json`: " + ", ".join(f"`{s.name}`" for s in incomplete)
        )

    st.divider()
    for s in strategies:
        render_strategy(s, rulesets)


# --------------------------------------------------------------------------
# Tab: Ruleset detail
# --------------------------------------------------------------------------
def render_ruleset_detail(rs: Ruleset | None) -> None:
    st.subheader("Ruleset Detail")
    if rs is None:
        st.info("No ruleset selected.")
        return
    if not rs.ok:
        st.error(f"`{rs.path.name}` could not be parsed:\n\n{rs.error}")
        return

    data = rs.data or {}
    st.markdown(f"**{rs.label}** · `{rs.path.name}`")
    if data.get("description"):
        st.caption(data["description"])

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
            "Enforced": "yes" if str(enf.get("status", "")).upper() == "ENFORCED" else "no",
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

    active = render_sidebar()
    rulesets = get_rulesets()

    st.title("CIO Command Center")
    st.caption(
        "Autonomous quantitative research under prop-firm constraints · "
        "frontend scaffold, agent backend not implemented"
    )

    tab_cmd, tab_vault, tab_rules = st.tabs(
        ["Command Center", "Strategy Vault", "Ruleset Detail"]
    )
    with tab_cmd:
        render_command_center(active)
    with tab_vault:
        render_vault(rulesets)
    with tab_rules:
        render_ruleset_detail(active)


# Streamlit executes this file top to bottom on every rerun, so main() is
# called unconditionally. No sys.exit() here: raising SystemExit out of a
# Streamlit script aborts the run rather than returning a shell status.
main()
