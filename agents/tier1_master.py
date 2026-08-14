"""
agents.tier1_master - the CIO agent.

Location:  ~/src/trading/agents/tier1_master.py

`run_campaign` and its intent router are implemented. The LLM-backed planning
functions (`build_client`, `propose_goals`, `prioritise`, `review`, `main`) are
still scaffold and raise NotImplementedError rather than returning a
plausible-looking placeholder, because a stub that silently returns an empty
result is exactly how a research pipeline starts reporting numbers nobody
generated.

What `run_campaign` does NOT do
------------------------------
It does not generate a strategy from the prompt. `research_campaign` stages a
module via `tier3_workers.generate_strategy_boilerplate`, whose signal logic is
an explicitly-labelled placeholder crossover. The hypothesis in the prompt is
recorded, not implemented.

So the compliance verdict at the end of a campaign describes THE TEMPLATE, not
the idea that was asked for. Every event this generator yields says so, and the
final payload carries `strategy_is_placeholder=True`. A pipeline that renders a
clean PASS over generated boilerplate would be manufacturing exactly the
confidence this repo exists to withhold - real strategy synthesis is Tier 1's
remaining unimplemented work, not something the router quietly stands in for.

What this tier is for
---------------------
Tier 1 owns the global optimisation goal: which hypotheses are worth spending
compute on, in what order, and when a line of enquiry is dead. It does not run
backtests itself - it decides what should be run and reads what came back.

    Tier 1 (this file)     what to investigate, and when to stop
    Tier 2 (supervisors)   whether a result is allowed to count
    Tier 3 (workers)       actually running the thing

The tiers are separate processes, not layers of one function, so a runaway
worker cannot take the planner down with it.

Boundaries that are not the model's to negotiate
------------------------------------------------
These come from the project's research discipline, and they exist because the
failure mode here is an overfitted backtest that looks right and fails live:

  - Tier 1 may not relax a prop-firm constraint, remove a cost model, or skip
    the out-of-sample gate to make a strategy pass. Those are Tier 2's to
    enforce and nobody's to override.
  - Every result Tier 1 reasons about must carry `variants_tested`. A Sharpe
    read without knowing how many variants it was selected from is not
    evidence, and the CIO's whole job is deciding what counts as evidence.
  - A strategy that survives in-sample is a candidate, not a result, until it
    has survived Phase 3 on out-of-sample NT8 data.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterator

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# The current Google GenAI SDK. Guarded so this module stays importable while
# the tier is still a scaffold and in environments without the SDK.
try:
    from google import genai
    _GENAI_IMPORT_ERROR: Exception | None = None
except ImportError as e:      # pragma: no cover - depends on the environment
    genai = None
    _GENAI_IMPORT_ERROR = e


DEFAULT_MODEL = "gemini-2.5-pro"


@dataclass
class ResearchGoal:
    """
    One line of enquiry the CIO is pursuing.

    `hypothesis` is prose on purpose. The point of writing it down before any
    backtest runs is that it can be checked afterwards against what was
    actually tested - a goal quietly rewritten to match a good result is the
    cheapest possible way to fool yourself.
    """

    hypothesis: str
    symbols: list[str]
    timeframe: str
    max_variants: int                      # the search budget, fixed up front
    portfolio: str = "A"                   # A = intraday/prop, B = swing/own
    notes: str = ""


@dataclass
class GoalOutcome:
    """What came back, and whether it is allowed to count."""

    goal: ResearchGoal
    variants_tested: int
    survived_oos: bool = False
    supervisor_verdicts: dict[str, Any] = field(default_factory=dict)
    verdict: str = "pending"               # pending | pursue | discard


# --------------------------------------------------------------------------
# Intent routing
# --------------------------------------------------------------------------
VAULT_TERMS = (
    "vault", "approved", "incubator", "staged", "shortlist", "portfolio",
    "what strategies", "which strategies", "list strategies", "show strategies",
    "how many strategies", "strategy status",
)

RESEARCH_TERMS = (
    "backtest", "back-test", "research", "generate", "build", "create",
    "develop", "test a strategy", "run a strategy", "campaign", "hypothesis",
    "breakout", "mean reversion", "mean-reversion", "momentum", "trend follow",
    "crossover", "optimise", "optimize", "sweep", "walk forward",
    "walk-forward",
)

# Asked in the imperative these read as research, but paired with a vault term
# they are a query about what already exists.
_VAULT_PRIORITY = ("vault", "approved", "incubator")


def classify_intent(prompt: str) -> dict[str, Any]:
    """
    Bucket a prompt into vault_status, research_campaign, or conversational.

    Deliberately keyword-based rather than a model call. This runs on every
    keystroke-submitted command and decides whether to spend minutes of compute
    on a backtest; a cheap, inspectable, deterministic rule is the right tool,
    and a misroute here is visible rather than mysterious.

    `conversational` is the fallback, not an error bucket. A prompt that
    matches nothing gets a straight answer about what the system can do - it
    does not get a backtest run on a guess.
    """
    text = (prompt or "").strip().lower()
    if not text:
        return {"intent": "conversational", "matched": [],
                "reason": "empty prompt"}

    vault_hits = [t for t in VAULT_TERMS if t in text]
    research_hits = [t for t in RESEARCH_TERMS if t in text]

    # A vault noun beats a research verb: "show me the approved strategies"
    # contains neither, but "list the strategies we've approved" would
    # otherwise trip on nothing and "generate a report on the vault" should
    # still be a vault query.
    if vault_hits and any(t in text for t in _VAULT_PRIORITY):
        return {"intent": "vault_status", "matched": vault_hits,
                "reason": "asks about staged or approved strategies"}
    if research_hits:
        return {"intent": "research_campaign", "matched": research_hits,
                "reason": "asks for a strategy to be built or tested"}
    if vault_hits:
        return {"intent": "vault_status", "matched": vault_hits,
                "reason": "asks about staged or approved strategies"}
    return {"intent": "conversational", "matched": [],
            "reason": "no research or vault intent detected"}


def parse_symbols(prompt: str) -> list[str]:
    """
    Pull known lake symbols out of a prompt.

    Matched on word boundaries against the lake's actual symbol list. Substring
    matching would find "ES" inside "test" and "strategies", and silently
    running a campaign on the wrong instrument is worse than finding none.
    """
    try:
        from mdlib.lake import available_symbols
        known = list(available_symbols())
    except Exception:
        known = ["ES", "NQ", "GC", "CL", "ZN", "RTY", "YM", "SI", "NG"]

    found = []
    for sym in known:
        if re.search(rf"\b{re.escape(sym)}\b", prompt or "", re.IGNORECASE):
            found.append(sym)
    return found


def parse_timeframe(prompt: str) -> str | None:
    """Explicit timeframe in the prompt, or None to fall back to the default."""
    text = (prompt or "").lower()
    for pattern, tf in (
        (r"\b1\s?m\b|\bone[- ]minute\b|\bminute\b", "1m"),
        (r"\b5\s?m\b|\bfive[- ]minute\b", "5m"),
        (r"\b15\s?m\b|\bfifteen[- ]minute\b", "15m"),
        (r"\b30\s?m\b|\bthirty[- ]minute\b|\bhalf[- ]hour\b", "30m"),
        (r"\b1\s?h\b|\bhourly\b|\bone[- ]hour\b", "1h"),
        (r"\b4\s?h\b|\bfour[- ]hour\b", "4h"),
        (r"\b1\s?d\b|\bdaily\b|\bend[- ]of[- ]day\b", "1d"),
        (r"\b1\s?w\b|\bweekly\b", "1w"),
    ):
        if re.search(pattern, text):
            return tf
    return None


# --------------------------------------------------------------------------
# Campaign defaults
# --------------------------------------------------------------------------
DEFAULT_RULESET = "fundednext_rapid.json"
RULES_DIR = _REPO / "compliance_rules"
INCUBATOR = _REPO / "strategies" / "approved_incubator"

# A bounded default window. A campaign kicked off from a chat box should not
# silently start a full-lake 1-minute run; the caller can widen it explicitly.
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2023-12-31"
DEFAULT_TIMEFRAME = "1d"
DEFAULT_SYMBOLS = ["ES", "NQ"]


def resolve_ruleset(ruleset_path: str | Path | None) -> Path:
    """Resolve a ruleset argument to a real file, defaulting to FundedNext."""
    if ruleset_path:
        p = Path(ruleset_path)
        if p.exists():
            return p
        candidate = RULES_DIR / p.name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"ruleset not found: {ruleset_path}")
    default = RULES_DIR / DEFAULT_RULESET
    if not default.exists():
        raise FileNotFoundError(f"default ruleset missing: {default}")
    return default


def read_vault() -> list[dict[str, Any]]:
    """
    Every staged strategy in approved_incubator/, described by its meta.json.

    A directory without a readable meta.json is returned with an `error` rather
    than skipped. An unlabelled strategy sitting in the vault is a thing
    somebody needs to see, not a thing to quietly omit from the count.
    """
    if not INCUBATOR.exists():
        return []
    out = []
    for d in sorted(x for x in INCUBATOR.iterdir() if x.is_dir()):
        if d.name.startswith((".", "__")):
            continue
        entry: dict[str, Any] = {"name": d.name, "path": str(d), "error": None,
                                 "meta": {}, "has_results": False}
        meta_p = d / "meta.json"
        if not meta_p.exists():
            entry["error"] = "no meta.json"
        else:
            try:
                entry["meta"] = json.loads(meta_p.read_text())
            except Exception as e:
                entry["error"] = f"meta.json unreadable: {type(e).__name__}: {e}"
        entry["has_results"] = any((d / f).exists() for f in
                                   ("returns.parquet", "equity.parquet"))
        out.append(entry)
    return out


# --------------------------------------------------------------------------
# The campaign generator
# --------------------------------------------------------------------------
def _event(status: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "message": message, **extra}


def run_campaign(prompt: str,
                 ruleset_path: str | Path | None = None,
                 symbols: list[str] | None = None,
                 start_date: str = DEFAULT_START,
                 end_date: str = DEFAULT_END,
                 timeframe: str | None = None) -> Iterator[dict[str, Any]]:
    """
    Route a prompt and run the resulting work, yielding progress as it goes.

    A generator rather than a blocking call so a UI can drive the loop on its
    own thread and render each step as it lands. Streamlit re-executes the
    script per interaction and a worker thread loses its ScriptRunContext, so
    anything that renders from a background thread is writing into a context
    that no longer exists. Yielding hands control back to the caller between
    steps and sidesteps that entirely.

    Yields
    ------
    dict with at least `status` and `message`. Statuses:

        routing        the intent router ran
        planning       inputs resolved
        generating     a strategy module was staged
        backtesting    Tier 3 is running
        auditing       Tier 2 is evaluating
        complete       final payload, carries `response`
        rejected       understood, deliberately not run
        error          something failed, named

    The final event always carries `response` (markdown for a chat transcript)
    and `intent`. Callers should render the last event and may render the rest
    as progress.
    """
    intent_info = classify_intent(prompt)
    intent = intent_info["intent"]
    yield _event("routing", f"Routing prompt → **{intent}**",
                 intent=intent, matched=intent_info["matched"],
                 reason=intent_info["reason"])

    if intent == "vault_status":
        yield from _run_vault_status(prompt, intent_info)
        return
    if intent == "conversational":
        yield from _run_conversational(prompt, intent_info)
        return
    yield from _run_research_campaign(
        prompt, intent_info, ruleset_path, symbols,
        start_date, end_date, timeframe)


# -- vault ------------------------------------------------------------------
def _run_vault_status(prompt: str, intent_info: dict) -> Iterator[dict]:
    yield _event("planning", "Reading `strategies/approved_incubator/`…")
    try:
        vault = read_vault()
    except Exception as e:
        yield _event("error", f"Could not read the vault: {type(e).__name__}: {e}",
                     intent="vault_status",
                     response=f"**Vault unreadable** — {type(e).__name__}: {e}")
        return

    if not vault:
        response = (
            "**Strategy Vault is empty.**\n\n"
            "No strategies are staged in `strategies/approved_incubator/`. "
            "Each staged strategy is a directory containing `meta.json` and, "
            "once run, `returns.parquet` / `trades.parquet`."
        )
        yield _event("complete", "Vault is empty.", intent="vault_status",
                     response=response, vault=[], n_staged=0)
        return

    with_results = [v for v in vault if v["has_results"]]
    broken = [v for v in vault if v["error"]]

    lines = [f"**Strategy Vault — {len(vault)} staged**", ""]
    lines.append(f"- With saved results: {len(with_results)}")
    lines.append(f"- Incomplete: {len(broken)}")
    lines.append("")
    for v in vault:
        meta = v["meta"] or {}
        name = meta.get("name") or v["name"]
        bits = []
        if meta.get("version"):
            bits.append(f"Version {meta['version']}")
        if meta.get("symbols"):
            bits.append(", ".join(meta["symbols"]))
        if meta.get("timeframe"):
            bits.append(meta["timeframe"])
        detail = " · ".join(bits) if bits else "no metadata"
        flag = f" — ⚠ {v['error']}" if v["error"] else ""
        results = "results saved" if v["has_results"] else "not yet run"
        lines.append(f"- **{name}** ({detail}) — {results}{flag}")

    if broken:
        lines += ["", f"⚠ {len(broken)} directory(ies) have no readable "
                      f"`meta.json`. They are listed rather than hidden — an "
                      f"unlabelled strategy is worse than a missing one."]

    yield _event("complete", f"{len(vault)} strategy(ies) staged.",
                 intent="vault_status", response="\n".join(lines),
                 vault=vault, n_staged=len(vault))


# -- conversational ---------------------------------------------------------
CAPABILITIES = """**CIO Command Center — what I can do**

- **Run a research campaign.** Ask me to build, generate, or backtest a
  strategy and name the instruments, e.g. *“backtest a breakout on ES and NQ”*.
  I stage a strategy module, run it through the streaming engine with costs,
  and audit the result against the active compliance ruleset.
- **Report the vault.** Ask what is staged or approved and I read
  `strategies/approved_incubator/`.
- **Answer questions about the system** — this.

**What I will not do**

- Run a backtest on a guess. If I cannot tell which instruments you mean, I
  ask rather than picking some.
- Relax a prop-firm constraint or drop the cost model to make something pass.
  Those are Tier 2's to enforce and nobody's to override."""


def _run_conversational(prompt: str, intent_info: dict) -> Iterator[dict]:
    text = (prompt or "").strip()
    yield _event("planning", "No research or vault intent — answering directly.")

    if not text:
        response = "I did not get a command. " + CAPABILITIES
    elif re.search(r"\b(hi|hello|hey|thanks|thank you|good morning|"
                   r"good afternoon)\b", text.lower()):
        response = "Hello. " + CAPABILITIES
    else:
        response = (
            f"I could not map that to a research campaign or a vault query, so "
            f"I have not run anything.\n\n"
            f"> {text[:200]}\n\n" + CAPABILITIES
        )

    yield _event("complete", "Answered without running a backtest.",
                 intent="conversational", response=response,
                 ran_backtest=False)


# -- research ---------------------------------------------------------------
def _run_research_campaign(prompt: str, intent_info: dict,
                           ruleset_path: str | Path | None,
                           symbols: list[str] | None,
                           start_date: str, end_date: str,
                           timeframe: str | None) -> Iterator[dict]:
    # Step A: resolve inputs -------------------------------------------------
    yield _event("planning", "Resolving symbols, timeframe and ruleset…")

    resolved_symbols = symbols or parse_symbols(prompt)
    if not resolved_symbols:
        response = (
            "**No campaign run — I could not tell which instruments you mean.**\n\n"
            f"> {prompt[:200]}\n\n"
            f"Name at least one symbol from the lake, e.g. *“backtest a "
            f"breakout on ES and NQ”*.\n\n"
            f"I will not pick instruments for you: a campaign silently run on "
            f"the wrong market produces a number that looks fine and means "
            f"nothing. Available: `ES NQ GC CL ZN RTY YM SI NG …`"
        )
        yield _event("rejected", "No symbols found in the prompt — not guessing.",
                     intent="research_campaign", response=response,
                     ran_backtest=False)
        return

    tf = timeframe or parse_timeframe(prompt) or DEFAULT_TIMEFRAME
    try:
        rules = resolve_ruleset(ruleset_path)
    except FileNotFoundError as e:
        yield _event("error", str(e), intent="research_campaign",
                     response=f"**Campaign aborted** — {e}")
        return

    yield _event("planning",
                 f"Symbols **{', '.join(resolved_symbols)}** · timeframe "
                 f"**{tf}** · {start_date} → {end_date} · ruleset "
                 f"`{rules.name}`",
                 symbols=resolved_symbols, timeframe=tf,
                 ruleset=str(rules), start=start_date, end=end_date)

    # Step B: stage a strategy ----------------------------------------------
    yield _event("generating",
                 "Staging a strategy module — **placeholder logic**, the "
                 "prompt's hypothesis is recorded, not implemented.")
    try:
        from agents.tier3_workers import (generate_strategy_boilerplate,
                                          run_strategy_backtest)
        from agents.tier2_supervisors import evaluate_compliance
    except Exception as e:
        yield _event("error", f"Could not import the worker tiers: {e}",
                     intent="research_campaign",
                     response=f"**Campaign aborted** — {type(e).__name__}: {e}")
        return

    stamp = date.today().isoformat().replace("-", "")
    name = f"campaign {'_'.join(resolved_symbols[:3])} {stamp}"
    try:
        strategy_path = generate_strategy_boilerplate(
            name,
            description=f"Auto-staged for campaign: {prompt[:120]}",
            params={"fast": 20, "slow": 50},
            symbols=resolved_symbols, timeframe=tf, overwrite=True)
    except Exception as e:
        yield _event("error", f"Could not stage a strategy: {e}",
                     intent="research_campaign",
                     response=f"**Campaign aborted** — {type(e).__name__}: {e}")
        return

    yield _event("generating", f"Staged `{strategy_path.name}`.",
                 strategy_path=str(strategy_path))

    # Step C: backtest -------------------------------------------------------
    yield _event("backtesting",
                 f"Running the streaming engine over "
                 f"{len(resolved_symbols)} symbol(s) at {tf}, costs included…")
    try:
        metrics = run_strategy_backtest(
            strategy_path, resolved_symbols, start_date, end_date,
            params={"fast": 20, "slow": 50}, tf=tf)
    except Exception as e:
        yield _event("error", f"Backtest failed: {type(e).__name__}: {e}",
                     intent="research_campaign",
                     response=f"**Backtest failed** — {type(e).__name__}: {e}")
        return

    yield _event("backtesting",
                 f"{metrics['trade_count']:,} trades · Sharpe "
                 f"{metrics['sharpe']:.2f} · max DD "
                 f"{metrics['max_drawdown_pct']:.2f}%",
                 metrics={k: v for k, v in metrics.items()
                          if k not in ("trades", "trade_log")})

    # Step D: audit ----------------------------------------------------------
    yield _event("auditing", f"Auditing against `{rules.name}`…")
    try:
        # Pass the engine's own daily equity curve rather than letting the
        # supervisor rebuild one from trade exits. A reconstructed curve marks
        # P&L only when a position closes, so drawdown suffered while a trade
        # was open is invisible and the figure is a lower bound - and a prop
        # account is closed on unrealized drawdown too.
        compliance = evaluate_compliance(
            metrics["trades"], rules,
            initial_balance=metrics["meta"]["initial_capital"],
            equity=metrics.get("equity"))
    except Exception as e:
        yield _event("error", f"Compliance audit failed: {type(e).__name__}: {e}",
                     intent="research_campaign",
                     response=f"**Audit failed** — {type(e).__name__}: {e}")
        return

    # Step E: verdict --------------------------------------------------------
    yield _event("complete", f"Campaign complete — {compliance['verdict']}.",
                 intent="research_campaign",
                 response=_render_campaign(prompt, resolved_symbols, tf,
                                           start_date, end_date, rules,
                                           strategy_path, metrics, compliance),
                 metrics={k: v for k, v in metrics.items()
                          if k not in ("trades", "trade_log")},
                 compliance=compliance,
                 strategy_path=str(strategy_path),
                 strategy_is_placeholder=True,
                 ran_backtest=True)


def _render_campaign(prompt: str, symbols: list[str], tf: str,
                     start: str, end: str, rules: Path, strategy_path: Path,
                     metrics: dict, compliance: dict) -> str:
    verdict = compliance["verdict"]
    icon = "✅" if verdict == "PASS" else "❌"

    lines = [
        f"### {icon} Campaign verdict: **{verdict}**",
        "",
        "> ⚠️ **The strategy is generated boilerplate with placeholder "
        "crossover logic.** The hypothesis in your prompt was recorded, not "
        "implemented, so this verdict describes the template — not the idea "
        "you asked for. Replace the signal logic in "
        f"`{strategy_path.relative_to(_REPO)}` before reading anything into it.",
        "",
        f"**Setup** — {', '.join(symbols)} · {tf} · {start} → {end} · "
        f"ruleset `{rules.name}` · costs included",
        "",
        "**Performance**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total P&L | {metrics['total_pnl']:,.2f} |",
        f"| Total return | {metrics['total_return_pct']:.2f}% |",
        f"| Sharpe | {metrics['sharpe']:.2f} |",
        f"| Max drawdown | {metrics['max_drawdown_pct']:.2f}% |",
        f"| Win rate | {metrics['win_rate'] * 100:.1f}% |",
        f"| Profit factor | {metrics['profit_factor']:.2f} |",
        f"| Trades | {metrics['trade_count']:,} |",
        "",
    ]

    if metrics.get("ruined"):
        lines += ["> 🚨 **Account ruined** — equity reached zero or below, so "
                  "annualized figures are undefined.", ""]

    lines.append("**Compliance**")
    lines.append("")
    for rule, c in (compliance.get("checks") or {}).items():
        status = c.get("status", "?")
        mark = {"PASS": "✅", "FAIL": "❌"}.get(status, "⚪")
        lines.append(f"- {mark} `{rule}` — {status}")
    if compliance.get("failures"):
        lines += ["", "**Why it failed**", ""]
        lines += [f"- {f}" for f in compliance["failures"]]
    if compliance.get("unenforced_rules"):
        lines += ["", f"⚠ Not evaluated here: "
                      f"{', '.join(compliance['unenforced_rules'])}. A PASS "
                      f"does not cover them."]
    if compliance.get("caveat"):
        lines += ["", f"⚠ {compliance['caveat']}"]

    return "\n".join(lines)


def build_client(api_key: str | None = None):
    """Construct the GenAI client used by this tier."""
    raise NotImplementedError("tier1_master: not implemented yet")


def propose_goals(context: dict[str, Any], n: int = 5) -> list[ResearchGoal]:
    """Generate candidate research goals from the current state of the book."""
    raise NotImplementedError("tier1_master: not implemented yet")


def prioritise(goals: list[ResearchGoal]) -> list[ResearchGoal]:
    """Order goals by expected information gain per unit of compute."""
    raise NotImplementedError("tier1_master: not implemented yet")


def review(outcome: GoalOutcome) -> str:
    """
    Decide `pursue` or `discard` for a completed goal.

    Must refuse to return `pursue` for anything a Tier 2 supervisor rejected.
    """
    raise NotImplementedError("tier1_master: not implemented yet")


def main() -> int:
    raise NotImplementedError("tier1_master: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
