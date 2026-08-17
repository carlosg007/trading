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

import pandas as pd

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


# Verified against the live API on 2026-08-15 (`python test_gemini_api.py`).
# `gemini-2.5-pro` and `gemini-2.5-flash` still appear in `client.models.list()`
# but 404 on generateContent for this key ("no longer available to new users"),
# so listing a model is not evidence it can be called. Pinned rather than the
# `gemini-pro-latest` alias: DEFAULT_MODEL is printed into every campaign
# verdict as provenance, and a floating alias makes that record untrue the day
# Google repoints it. When this 404s, re-run the test and pin the successor.
DEFAULT_MODEL = "gemini-3.1-pro-preview"


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
                 timeframe: str | None = None,
                 genai_client: Any = None,
                 include_artifacts: bool = False) -> Iterator[dict[str, Any]]:
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
        complete       final payload, carries `response`
        rejected       understood, deliberately not run
        error          something failed, named

    `ruleset_path` is accepted for API compatibility and is no longer applied:
    prop-firm governance moved to CrossTrade NAM, so a research campaign
    reports statistics and does not issue a compliance verdict.

    The final event always carries `response` (markdown for a chat transcript)
    and `intent`. Callers should render the last event and may render the rest
    as progress.

    `include_artifacts=True` additionally attaches the raw daily equity Series
    and trade DataFrame to the final research event under `artifacts`. It is
    opt-in because the default consumer is a chat transcript, and a 500k-row
    trade list held in an event dict is most of a gigabyte for a caller that
    only wanted the summary line. A UI that plots the equity path asks for it.
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
        start_date, end_date, timeframe, genai_client, include_artifacts)


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
                           timeframe: str | None,
                           genai_client: Any = None,
                           include_artifacts: bool = False) -> Iterator[dict]:
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

    yield _event("planning",
                 f"Symbols **{', '.join(resolved_symbols)}** · timeframe "
                 f"**{tf}** · {start_date} → {end_date}",
                 symbols=resolved_symbols, timeframe=tf,
                 start=start_date, end=end_date)

    # Step B: synthesise a strategy -----------------------------------------
    try:
        from agents.tier3_workers import (GeneratedCodeError,
                                          generate_strategy_boilerplate,
                                          run_strategy_backtest,
                                          write_and_validate_strategy)
    except Exception as e:
        yield _event("error", f"Could not import the worker tiers: {e}",
                     intent="research_campaign",
                     response=f"**Campaign aborted** — {type(e).__name__}: {e}")
        return

    stamp = date.today().isoformat().replace("-", "")
    name = f"campaign {'_'.join(resolved_symbols[:3])} {stamp}"
    strategy_path = None
    synthesized = False

    yield _event("generating",
                 f"Synthesising a strategy for **{resolved_symbols[0]}** "
                 f"with `{DEFAULT_MODEL}`…")
    try:
        code = synthesize_strategy_code(prompt, resolved_symbols[0],
                                        client=genai_client, timeframe=tf)
        strategy_path = write_and_validate_strategy(name, code)
        synthesized = True
        yield _event("generating",
                     f"Synthesised and validated `{strategy_path.name}` — "
                     f"parsed, audited for unsafe imports and lookahead, and "
                     f"smoke-tested.",
                     strategy_path=str(strategy_path), synthesized=True)
    except MissingAPIKey as e:
        yield _event("warning",
                     f"⚠️ **No GenAI credential** ({e}). Falling back to the "
                     f"safe template — its logic is a PLACEHOLDER, so any "
                     f"verdict describes the template and not your hypothesis.",
                     fallback_reason="missing_api_key")
    except (SynthesisError, SyntaxError, GeneratedCodeError) as e:
        # The model was reachable and produced something unusable. Say what
        # was wrong rather than silently retrying or quietly degrading - a
        # rejected strategy is a finding about the generator.
        yield _event("warning",
                     f"⚠️ **Synthesis rejected** — {type(e).__name__}: {e}. "
                     f"Falling back to the safe template, whose logic is a "
                     f"PLACEHOLDER.",
                     fallback_reason=f"{type(e).__name__}: {e}")

    if strategy_path is None:
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
        yield _event("generating",
                     f"Staged template `{strategy_path.name}` — placeholder logic.",
                     strategy_path=str(strategy_path), synthesized=False)

    # Step C: backtest -------------------------------------------------------
    yield _event("backtesting",
                 f"Running the streaming engine over "
                 f"{len(resolved_symbols)} symbol(s) at {tf}, costs included…")
    # A synthesised module names its own parameters, so the template's
    # fast/slow would be rejected by its signature. Its defaults are used
    # instead - the system prompt requires the signature to carry them.
    run_params: dict[str, Any] = {} if synthesized else {"fast": 20, "slow": 50}
    try:
        metrics = run_strategy_backtest(
            strategy_path, resolved_symbols, start_date, end_date,
            params=run_params, tf=tf)
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

    # Step D: report ---------------------------------------------------------
    # The equity path and trade list, for a caller that plots them. Handed over
    # by reference rather than copied: the alternative is a second full copy of
    # the trade list in memory purely to render a histogram.
    artifacts = {"equity": metrics.get("equity"),
                 "trades": metrics.get("trades")} if include_artifacts else None

    # There is deliberately no compliance audit here. Prop-firm balance math is
    # enforced by CrossTrade NAM against a live account, not against a
    # backtest, and running it here produced a PASS/FAIL that read as a verdict
    # on the edge when it was a verdict on a funding program. Research reports
    # the statistics and stops.
    yield _event("complete",
                 f"Campaign complete — Sharpe {metrics['sharpe']:.2f} over "
                 f"{metrics['trade_count']:,} trades.",
                 intent="research_campaign",
                 response=_render_campaign(prompt, resolved_symbols, tf,
                                           start_date, end_date,
                                           strategy_path, metrics,
                                           synthesized),
                 metrics={k: v for k, v in metrics.items()
                          if k not in ("trades", "trade_log", "equity")},
                 strategy_path=str(strategy_path),
                 synthesized=synthesized,
                 strategy_is_placeholder=not synthesized,
                 artifacts=artifacts,
                 ran_backtest=True)


def _render_campaign(prompt: str, symbols: list[str], tf: str,
                     start: str, end: str, strategy_path: Path,
                     metrics: dict, synthesized: bool = False) -> str:
    rel = strategy_path.relative_to(_REPO)

    if synthesized:
        provenance = (
            f"> 🤖 **Strategy synthesised by `{DEFAULT_MODEL}`** and validated "
            f"before running: parsed, audited for unsafe imports and lookahead, "
            f"and smoke-tested on a synthetic frame. Source: `{rel}`.\n"
            f">\n"
            f"> The audit catches negative shifts and reversed slices. It "
            f"cannot catch every form of lookahead, and it says nothing about "
            f"whether the logic implements what you asked for. **Read the "
            f"generated code before acting on this verdict.**"
        )
    else:
        provenance = (
            f"> ⚠️ **The strategy is generated boilerplate with placeholder "
            f"crossover logic.** The hypothesis in your prompt was recorded, "
            f"not implemented, so this verdict describes the template — not "
            f"the idea you asked for. Replace the signal logic in `{rel}` "
            f"before reading anything into it."
        )

    lines = [
        f"### 📊 Campaign result — {', '.join(symbols)} {tf}",
        "",
        provenance,
        "",
        f"**Setup** — {', '.join(symbols)} · {tf} · {start} → {end} · "
        f"costs included",
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

    lines += [
        "> ⚖️ **No prop-firm compliance was evaluated.** Account governance — "
        "trailing drawdown, daily loss, consistency, sizing — is enforced by "
        "CrossTrade NAM against a live balance. These figures describe the "
        "edge only. A strong Sharpe here is not clearance to trade a funded "
        "account.",
        "",
        f"**Next gate** — hold out the final 3 years and re-run before "
        f"believing any of this. `variants_tested` is "
        f"{metrics.get('meta', {}).get('variants_tested', 'unrecorded')}.",
    ]

    return "\n".join(lines)


# --------------------------------------------------------------------------
# The Dual-Version Mandate
# --------------------------------------------------------------------------
def _resolve_strategy(strategy_code: str, params: dict[str, Any] | None):
    """
    Accept either a path to a strategy module or raw source, and return a bound
    signal_fn.

    Raw source is written through `write_and_validate_strategy`, so it takes
    the same AST audit - import allowlist, forbidden builtins, negative shifts,
    reversed slices - and the same smoke test as anything Gemini produces.
    There is deliberately no path that imports a code string directly.
    """
    from agents.tier3_workers import load_strategy, write_and_validate_strategy

    text = (strategy_code or "").strip()
    if not text:
        raise ValueError("strategy_code is empty")

    # A path is one line ending in .py. Source always carries a newline, so the
    # two cannot be confused by a filename that happens to contain "def".
    if "\n" not in text and text.endswith(".py"):
        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"strategy module not found: {path}")
        return load_strategy(path, params)

    stamp = date.today().isoformat().replace("-", "")
    staged = write_and_validate_strategy(f"dual version {stamp}", text)
    return load_strategy(staged, params)


def _strategy_indicators(info: dict, bars: pd.DataFrame) -> dict | None:
    """
    The strategy's own indicator series, or None when it declares none.

    Cosmetic - these are drawn on the tear sheet and read by nothing else - so
    a hook that raises costs the overlay and says so on stderr, rather than
    throwing away a completed backtest over a chart annotation. The same reason
    the report writer itself is wrapped.
    """
    fn = info.get("indicator_fn")
    if fn is None:
        return None
    try:
        series = fn(bars)
    except Exception as e:                                      # noqa: BLE001
        print(f"[!] indicators() raised, so the inspector has no overlay: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return None
    return series


def run_dual_version_backtest(strategy_code: str,
                              df: pd.DataFrame,
                              freq: str = "15m",
                              symbol: str | None = None,
                              cfg: Any = None,
                              params: dict[str, Any] | None = None,
                              threshold: float = 0.50,
                              ml: bool = True,
                              robustness: dict[str, Any] | None = None,
                              holdout: dict[str, Any] | None = None,
                              emit_reports: bool = True,
                              report_dir: Any = None,
                              artifacts_root: str = "/mnt/backtest/artifacts",
                              strat_name: str | None = None) -> dict[str, Any]:
    """
    Run a strategy as Version A (rule-based) and Version B (ML-filtered) over
    the same bars, under identical costs.

    This is the comparison the Dual-Version Mandate is built on: ML is adopted
    only if B beats A out-of-sample. Both versions see the same frame, the same
    fills - next bar's open - and the same cost arrays, so the only difference
    between the two equity curves is which entries the classifier suppressed.
    Nothing else is allowed to vary, because if it did, the comparison would be
    measuring the change rather than the filter.

    `df` is ONE symbol's OHLCV frame. A multi-symbol frame raises rather than
    being silently accepted: rows sorted by (ts, symbol) interleave instruments,
    and a rolling window over that averages across contracts. It produced a
    plausible equity curve and 608,079 trades where the correct per-symbol
    signals gave 86,035, which is why the engine has no frame-in entry point at
    all. This function needs one, so it checks.

    Parameters
    ----------
    strategy_code
        A path to a strategy module, or raw source (audited before it runs).
    freq
        The timeframe the frame is already at. Recorded as provenance and used
        for nothing else - no resampling happens here, because a silent
        resample is how a 15m result gets reported as a 1m one.
    threshold
        P(win) at or above which Version B keeps an entry.
    ml
        Run Version B at all. `ml=False` returns `version_b: None` and a
        comparison whose every comparative field is None under
        `ml_evaluated: False` - not `b_beats_a: False`, which is not the same
        thing as a Version B that ran and lost. A caller reading a missing B
        as a defeated B would
        credit the baseline with a win nobody contested. It exists for the
        multi-asset batch, where the classifier refits once per completed trade
        and 27 symbols of that is hours, not minutes. Note the Dual-Version
        Mandate still wants B before anything is adopted; skipping it defers
        that comparison, it does not settle it.
    robustness, holdout
        Optional per-version evidence for Gates 2 and 3, keyed by version:
        `{"A": {...}, "B": {...}}`. See
        `backtest.report.audit_acceptance_gates` for the shapes. Omitted, those
        gates report NOT EVALUATED - which is not a pass. Nothing here can
        produce them: a walk-forward and a bootstrap are separate runs, and
        inventing a number to fill the slot is the failure this project exists
        to avoid.
    emit_reports
        Write `report_version_a.html`, `report_version_b.html` and
        `dual_metrics.json` into
        `<artifacts_root>/<strat_name>_<timestamp>/` (or `report_dir`). A
        failure to write is recorded in the returned dict, never raised - a
        completed backtest is not thrown away because an NFS mount was busy.

    Returns
    -------
    dict with `version_a` and `version_b`, each carrying `metrics` (the same
    dict shape `run_strategy_backtest` returns), `result` (a `BacktestResult`
    with returns, trades and equity) and `gate_audit`, plus a `comparison`
    block, `reports` and `meta`.

    Note `result` is the engine's BacktestResult, not a raw vectorbt Portfolio.
    `_simulate` feeds vectorbt in chunks and concatenates the trade records, so
    there is no single Portfolio object to hand back; returning one would mean
    disabling the batching that keeps peak RAM flat.
    """
    import numpy as np

    from backtest.engine import (BacktestConfig, _assemble_result, _simulate,
                                 clean_signals_ls, unpack_signals)
    from agents.tier3_workers import apply_ml_signal_filter, summarize_result

    config = cfg or BacktestConfig()

    if df is None or len(df) == 0:
        raise ValueError("df is empty - nothing to simulate")

    bars = df
    if "ts" not in bars.columns:
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError(
                "df needs a `ts` column or a DatetimeIndex; got an index of "
                f"type {type(bars.index).__name__}")
        bars = bars.assign(ts=pd.DatetimeIndex(bars.index).tz_localize("UTC")
                           if bars.index.tz is None else bars.index)
    bars = bars.reset_index(drop=True)

    missing = {"open", "high", "low", "close", "volume"} - set(bars.columns)
    if missing:
        raise ValueError(f"df is missing OHLCV columns: {sorted(missing)}")

    if "symbol" in bars.columns:
        present = pd.unique(bars["symbol"].dropna())
        if len(present) > 1:
            raise ValueError(
                f"df carries {len(present)} symbols ({', '.join(map(str, present[:5]))}). "
                f"Pass one symbol's bars: a rolling window over an interleaved "
                f"frame averages across contracts and the result looks fine.")
        if symbol is None and len(present) == 1:
            symbol = str(present[0])
    if symbol is None:
        raise ValueError(
            "symbol could not be determined from df and none was given. It "
            "sets the contract multiplier, tick size and commission - guessing "
            "it would silently rescale every P&L figure.")

    signal_fn, info = _resolve_strategy(strategy_code, params)

    # -- Version A: the rule-based baseline --------------------------------
    # Two masks or four - see `backtest.engine.unpack_signals`. A long-only
    # strategy comes back with all-False short masks and everything below is
    # the run it always was.
    entries, exits, s_entries, s_exits = unpack_signals(signal_fn(bars),
                                                        len(bars))

    # The news and day-of-week entry filters, applied here for the same reason
    # and in the same place as in `run_backtest` - this function drives
    # `_simulate` directly rather than going through it, so a filter wired only
    # into the engine's entry point would be silently inert for every run the
    # batch runner and the pipeline stages make. Version B filters these same
    # cleaned signals, so both versions inherit it and the comparison stays a
    # comparison of the classifier.
    filter_info: dict[str, Any] = {}
    if config.news_filter or config.exclude_days:
        from backtest.event_calendar import apply_entry_filters
        entries, s_entries, filter_info = apply_entry_filters(
            bars["ts"], entries, s_entries,
            news_filter=config.news_filter,
            news_window_minutes=config.news_window_minutes,
            news_kinds=config.news_kinds,
            exclude_days=config.exclude_days)
        filter_info["symbol"] = symbol

    if config.flat_by_close:
        from backtest.engine import apply_flat_by_close
        entries, exits = apply_flat_by_close(bars, entries, exits,
                                             config.session_close_utc)
        s_entries, s_exits = apply_flat_by_close(bars, s_entries, s_exits,
                                                 config.session_close_utc)

    entries_a, exits_a, s_entries_a, s_exits_a = clean_signals_ls(
        entries, exits, s_entries, s_exits)

    days = pd.DatetimeIndex(np.unique(
        pd.DatetimeIndex(bars["ts"]).values.astype("datetime64[D]"))
    ).tz_localize("UTC")

    trades_a = _simulate(bars, entries_a, exits_a, symbol, config,
                         s_entries_a, s_exits_a)
    result_a = _assemble_result([trades_a] if not trades_a.empty else [],
                                days, config, filter_info=filter_info)

    # -- Version B: the same signals, ML-filtered ---------------------------
    # Filtering the CLEANED signals, not the raw ones, so the trades the
    # classifier learns from are exactly the trades Version A took.
    result_b = None
    metrics_b = None
    entries_b = None
    if ml:
        filtered, exits_b = apply_ml_signal_filter(
            bars, entries_a, exits_a, symbol=symbol, cfg=config,
            threshold=threshold, direction="long")
        # The short side gets its own classifier, trained on its own completed
        # trades with the short P&L sign. Reusing the long filter here would
        # score every short against a model whose training set is entirely
        # longs; passing the shorts through unfiltered would label the run
        # "ML-filtered" while half its trades never met the classifier.
        s_filtered, s_exits_b = apply_ml_signal_filter(
            bars, s_entries_a, s_exits_a, symbol=symbol, cfg=config,
            threshold=threshold, direction="short")
        entries_b, exits_b, s_entries_b, s_exits_b = clean_signals_ls(
            filtered, exits_b, s_filtered, s_exits_b)

        trades_b = _simulate(bars, entries_b, exits_b, symbol, config,
                             s_entries_b, s_exits_b)
        # The same filter report as A: Version B filters A's already-filtered
        # entries, so the news and weekday suppressions are common to both and
        # only the classifier's cut differs between the two columns.
        result_b = _assemble_result([trades_b] if not trades_b.empty else [],
                                    days, config, filter_info=filter_info)
        metrics_b = summarize_result(result_b)

    metrics_a = summarize_result(result_a)

    meta = {
        "strategy": info["module"],
        "strategy_path": info["path"],
        "params": info.get("bound_params", {}),
        "symbol": symbol,
        "timeframe": freq,
        "bars": int(len(bars)),
        "start": str(bars["ts"].iloc[0]),
        "end": str(bars["ts"].iloc[-1]),
        "costs_included": True,
        "initial_capital": config.initial_capital,
        "ml_threshold": threshold if ml else None,
        "ml_evaluated": bool(ml),
        # Plain-English sentences the module declares about itself, with the
        # bound parameters filled in. Presentation only - the tear sheet's
        # strategy card reads these, and nothing else does.
        "logic": info.get("logic") or {},
    }
    for m in (metrics_a, metrics_b):
        if m is not None:
            m["meta"] = meta

    # Gate audit per version. Gates 2 and 3 report NOT EVALUATED unless the
    # caller supplied the walk-forward, bootstrap and holdout evidence, because
    # this function does not produce them and a blank gate is not a cleared one.
    from backtest.report import audit_acceptance_gates

    rb = robustness or {}
    ho = holdout or {}
    audit_a = audit_acceptance_gates(metrics_a, rb.get("A"), ho.get("A"),
                                     version="A", name=info["module"])
    audit_b = (audit_acceptance_gates(metrics_b, rb.get("B"), ho.get("B"),
                                      version="B", name=info["module"])
               if ml else None)

    if ml:
        comparison = {
            "ml_evaluated": True,
            "entries_a": int(entries_a.sum()),
            "entries_b": int(entries_b.sum()),
            "entries_suppressed": int(entries_a.sum() - entries_b.sum()),
            "sharpe_delta": metrics_b["sharpe"] - metrics_a["sharpe"],
            "b_beats_a": bool(metrics_b["sharpe"] > metrics_a["sharpe"]),
        }
    else:
        # Every field a caller would compare on is None rather than 0 or False.
        # `b_beats_a: False` here would read as "the filter was tried and lost".
        comparison = {
            "ml_evaluated": False,
            "entries_a": int(entries_a.sum()),
            "entries_b": None,
            "entries_suppressed": None,
            "sharpe_delta": None,
            "b_beats_a": None,
        }

    out = {
        "version_a": {"label": "A · rule-based", "metrics": metrics_a,
                      "result": result_a, "gate_audit": audit_a},
        "version_b": ({"label": "B · ML-filtered", "metrics": metrics_b,
                       "result": result_b, "gate_audit": audit_b}
                      if ml else None),
        "comparison": comparison,
        "reports": None,
        "meta": meta,
    }

    if emit_reports:
        from backtest.report_html import write_dual_reports
        try:
            name = strat_name or Path(info["path"]).stem
            # `bars` powers the trade inspector: the report embeds the window
            # around each trade at build time, so a reader clicking a row does
            # not need the lake, a server, or this process still being alive.
            # `indicators` are drawn over those candles, and come from the
            # strategy module itself so the line a reader watches cross is the
            # array the entry was taken from.
            out["reports"] = write_dual_reports(
                out, bars=bars, out_dir=report_dir, strat_name=name,
                artifacts_root=artifacts_root,
                indicators=_strategy_indicators(info, bars))
        except Exception as e:                                  # noqa: BLE001
            # Recorded rather than raised, and recorded loudly enough that a
            # caller cannot mistake "no reports" for "reports somewhere else".
            out["reports"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[!] HTML reports were not written: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    return out


API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")

SYNTHESIS_SYSTEM_PROMPT = """\
Generate a strictly compliant Vectorbt Pro `signal_fn` module for futures \
trading.

SIGNATURE (exact):
    def signal_fn(bars: pd.DataFrame, **params) -> tuple[pd.Series, pd.Series]

`bars` is ONE instrument's OHLCV DataFrame, ordered oldest to newest, with \
lowercase columns `open`, `high`, `low`, `close`, `volume` and a UTC \
DatetimeIndex. Do NOT unpack it into separate arrays in the signature, and do \
NOT accept a multi-symbol frame — the engine calls this once per symbol.

Return a tuple `(entries, exits)` of two BOOLEAN pandas Series indexed by \
`bars.index`, the same length as `bars`.

HARD CONSTRAINTS:
- Read prices as `bars["close"]`, `bars["high"]`, etc. Use pandas/NumPy \
vectorized operations or Vectorbt Pro indicators. No Python row loops.
- NEVER use future-looking data. A value at index i may depend only on \
indices <= i. No negative shifts, no reversed slices, no centred windows. \
The engine fills at the NEXT bar's open, so a signal computed from bar i's \
close is legitimate.
- Return booleans, not prices. `bars["close"] > ma` is a signal; \
`bars["close"]` is not.
- Warm-up periods must be False, not NaN-coerced-to-True. Finish with \
`.fillna(False).astype(bool)` on both Series.
- Preserve the index: the returned Series must align with `bars.index`.
- Import only from: numpy, pandas, math, vectorbtpro, numba.
- No file, network, or OS access. No eval/exec/__import__/open.
- Output clean, executable Python only. No markdown backticks, no prose, no \
explanation outside comments.
- Give every numeric parameter a sensible default in the signature so the \
module runs with no arguments.
"""


def build_client(api_key: str | None = None):
    """
    Construct the GenAI client used by this tier.

    Raises rather than returning None when the SDK or key is missing, so a
    caller cannot mistake an unusable client for a working one.
    """
    if genai is None:
        raise RuntimeError(
            f"google-genai is not importable: {_GENAI_IMPORT_ERROR}"
        )
    key = api_key or _find_api_key()
    if not key:
        raise MissingAPIKey(
            f"no API key found in {' / '.join(API_KEY_VARS)}"
        )
    return genai.Client(api_key=key)


def _find_api_key() -> str | None:
    import os
    for var in API_KEY_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


class MissingAPIKey(RuntimeError):
    """No GenAI credential is available. Callers fall back to the template."""


class SynthesisError(RuntimeError):
    """The model was reachable but did not return usable code."""


def synthesize_strategy_code(prompt: str,
                             symbol: str,
                             model: str = DEFAULT_MODEL,
                             client: Any = None,
                             timeframe: str = "1d") -> str:
    """
    Ask Gemini for a strategy module implementing `prompt` for `symbol`.

    Returns raw Python source. It is NOT validated here - that is
    `tier3_workers.write_and_validate_strategy`, which parses it, audits it for
    unsafe imports and lookahead, and smoke-tests it before anything runs. This
    function's only job is to get text back.

    Raises `MissingAPIKey` when no credential is configured, which
    `run_campaign` catches to fall back to the template. That distinction
    matters: no key is an expected configuration state, while a model that
    returns nothing usable is a failure worth reporting.
    """
    if client is None:
        client = build_client()          # raises MissingAPIKey when unset

    user_prompt = (
        f"Instrument: {symbol}\n"
        f"Timeframe: {timeframe}\n"
        f"Strategy to implement: {prompt}\n\n"
        f"Write the module now."
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={"system_instruction": SYNTHESIS_SYSTEM_PROMPT,
                    "temperature": 0.2},
        )
    except Exception as e:
        raise SynthesisError(
            f"{model} call failed: {type(e).__name__}: {e}"
        ) from e

    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise SynthesisError(f"{model} returned no code")
    return text


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
