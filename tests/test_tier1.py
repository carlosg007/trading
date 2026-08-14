#!/usr/bin/env python3
"""
test_tier1.py - the Tier 1 CIO router and strategy synthesis.

Location:  ~/src/trading/tests/test_tier1.py

Run:  python tests/test_tier1.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure. Nothing here reaches the network or the lake: the GenAI
client is a stub, and the vault fixtures are written to a temp dir.

Why this suite exists
---------------------
`agents/tier1_master.py` had no test file. Its implemented surface -
`classify_intent`, `parse_symbols`, `parse_timeframe`, `resolve_ruleset`,
`read_vault`, `run_campaign`, `synthesize_strategy_code` - is what decides
whether a chat prompt spends minutes of compute on a backtest and which
instrument it spends them on.

The central claims
------------------
1. A prompt that asks for nothing recognisable does NOT get a research
   campaign. `conversational` is the fallback bucket, so an ambiguous prompt
   gets an answer rather than a speculative backtest.
2. Symbols are matched on word boundaries. Substring matching finds "ES" inside
   "strategies" and "test", and running a campaign on the wrong instrument is
   worse than finding no instrument at all.
3. `parse_timeframe` returns None rather than guessing, so the documented
   default applies visibly.
4. An unknown ruleset raises. Silently falling back to FundedNext would audit a
   strategy against constraints nobody asked for.
5. A vault directory with no meta.json is REPORTED, not skipped - an unlabelled
   strategy in the vault is exactly what somebody needs to see.
6. Missing credentials raise MissingAPIKey (an expected configuration state the
   caller downgrades on), while a reachable model returning nothing usable
   raises SynthesisError (a failure). Collapsing the two would let a silent
   template fallback look like a model-authored strategy.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import tier1_master  # noqa: E402
from agents.tier1_master import (  # noqa: E402
    API_KEY_VARS, DEFAULT_SYMBOLS, MissingAPIKey, SynthesisError,
    classify_intent, parse_symbols, parse_timeframe, read_vault,
    resolve_ruleset, run_campaign, synthesize_strategy_code,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

def test_intent_routing():
    print("\nINTENT ROUTING")
    research = [
        "backtest a breakout strategy on ES",
        "build me a mean reversion system for NQ",
        "run a walk forward on the momentum idea",
        "generate a new strategy hypothesis",
    ]
    for p in research:
        check(f"research: {p[:38]!r}",
              classify_intent(p)["intent"] == "research_campaign",
              classify_intent(p)["intent"])

    vault = [
        "what strategies are in the vault?",
        "show me the approved strategies",
        "how many strategies are staged in the incubator",
    ]
    for p in vault:
        check(f"vault: {p[:38]!r}",
              classify_intent(p)["intent"] == "vault_status",
              classify_intent(p)["intent"])

    ambiguous = [
        "hello",
        "what can you do?",
        "explain trailing drawdown to me",
        "",
        "   ",
    ]
    for p in ambiguous:
        got = classify_intent(p)["intent"]
        check(f"no compute on a guess: {p[:30]!r}", got == "conversational", got)

    check("a reason is always given",
          all(classify_intent(p)["reason"]
              for p in ("backtest ES", "show the vault", "hello", "")))
    check("a vault noun beats a research verb",
          classify_intent("generate a report on the vault")["intent"] == "vault_status",
          classify_intent("generate a report on the vault")["intent"])


# --------------------------------------------------------------------------
# prompt parsing
# --------------------------------------------------------------------------

def test_parse_symbols():
    print("\nSYMBOL PARSING (word boundaries, not substrings)")
    check("finds an explicit symbol", parse_symbols("backtest ES please") == ["ES"],
          str(parse_symbols("backtest ES please")))
    check("case-insensitive", parse_symbols("backtest es please") == ["ES"])
    check("finds several, lake order",
          set(parse_symbols("compare ES and NQ")) == {"ES", "NQ"},
          str(parse_symbols("compare ES and NQ")))

    for trap in ("run some tests on strategies", "the strategies look fine",
                 "escalate this", "best guess"):
        found = parse_symbols(trap)
        check(f"no substring hit in {trap!r}", "ES" not in found, str(found))

    check("nothing found is an empty list, not a default",
          parse_symbols("do the thing") == [])


def test_parse_timeframe():
    print("\nTIMEFRAME PARSING")
    for prompt, tf in (("run it daily", "1d"), ("a 5m breakout", "5m"),
                       ("hourly momentum", "1h"), ("weekly swing", "1w"),
                       ("30m opening range", "30m"), ("end-of-day signals", "1d")):
        check(f"{prompt!r} -> {tf}", parse_timeframe(prompt) == tf,
              str(parse_timeframe(prompt)))
    check("no timeframe -> None (the default applies visibly, not a guess)",
          parse_timeframe("backtest a breakout on ES") is None,
          str(parse_timeframe("backtest a breakout on ES")))


# --------------------------------------------------------------------------
# ruleset + vault
# --------------------------------------------------------------------------

def test_resolve_ruleset():
    print("\nRULESET RESOLUTION")
    default = resolve_ruleset(None)
    check("default resolves to a real file",
          default.is_file() and default.name == "fundednext_rapid.json", str(default))
    check("bare filename resolves inside compliance_rules/",
          resolve_ruleset("fundednext_rapid.json") == default)
    check("full path is honoured", resolve_ruleset(str(default)) == default)

    try:
        resolve_ruleset("no_such_program.json")
        check("unknown ruleset raises rather than defaulting", False, "no exception")
    except FileNotFoundError:
        check("unknown ruleset raises rather than defaulting", True)

    # The resolved file must still be a usable ruleset, not just a file.
    rules = json.loads(default.read_text())["rules"]
    check("ruleset carries enforcement status on every rule",
          all("enforcement" in r and r["enforcement"].get("status") for r in rules.values()),
          ", ".join(f"{k}={v['enforcement']['status']}" for k, v in rules.items()))


def test_read_vault():
    print("\nVAULT READING")
    original = tier1_master.INCUBATOR
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "good").mkdir()
        (tmp / "good" / "meta.json").write_text(json.dumps({"name": "good", "version": "A"}))
        (tmp / "unlabelled").mkdir()                       # no meta.json
        (tmp / "broken").mkdir()
        (tmp / "broken" / "meta.json").write_text("{not json")
        (tmp / "__pycache__").mkdir()

        tier1_master.INCUBATOR = tmp
        try:
            vault = read_vault()
        finally:
            tier1_master.INCUBATOR = original

    by_name = {e["name"]: e for e in vault}
    check("dunder dirs skipped", "__pycache__" not in by_name, str(sorted(by_name)))
    check("labelled strategy read", by_name["good"]["meta"]["version"] == "A")
    check("no error on a good entry", by_name["good"]["error"] is None)
    check("unlabelled strategy surfaced, not skipped",
          by_name["unlabelled"]["error"] == "no meta.json",
          str(by_name["unlabelled"]["error"]))
    check("unreadable meta.json surfaced",
          "unreadable" in (by_name["broken"]["error"] or ""),
          str(by_name["broken"]["error"]))

    tier1_master.INCUBATOR = original / "does_not_exist"
    try:
        check("missing vault dir is an empty list, not a crash", read_vault() == [])
    finally:
        tier1_master.INCUBATOR = original


# --------------------------------------------------------------------------
# campaign generator (non-research routes - no lake, no network)
# --------------------------------------------------------------------------

def test_run_campaign_routes():
    print("\nCAMPAIGN GENERATOR (conversational + vault routes)")
    for prompt, intent in (("what can you do?", "conversational"),
                           ("what strategies are in the vault?", "vault_status")):
        events = list(run_campaign(prompt))
        check(f"{intent}: events are produced", len(events) >= 2, str(len(events)))
        check(f"{intent}: first event is the routing decision",
              events[0]["status"] == "routing" and events[0]["intent"] == intent,
              f"{events[0]['status']}/{events[0].get('intent')}")
        check(f"{intent}: every event carries status and message",
              all(e.get("status") and e.get("message") for e in events))
        final = events[-1]
        check(f"{intent}: final event carries a response for the transcript",
              bool(final.get("response")), str(final.get("status")))
        check(f"{intent}: final event is terminal",
              final["status"] in ("complete", "rejected", "error"), final["status"])

    check("DEFAULT_SYMBOLS is a bounded default, not the whole lake",
          0 < len(DEFAULT_SYMBOLS) <= 4, str(DEFAULT_SYMBOLS))


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------

class _StubModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kw):
        self.calls.append(kw)
        if isinstance(self.text, Exception):
            raise self.text
        return type("R", (), {"text": self.text})()


class _StubClient:
    def __init__(self, text):
        self.models = _StubModels(text)


def test_synthesis():
    print("\nSTRATEGY SYNTHESIS")
    code = "def signal_fn(open_, high, low, close, volume, **p):\n    return None, None\n"
    client = _StubClient(code)
    got = synthesize_strategy_code("a breakout", "ES", client=client, timeframe="1d")
    check("model text is returned verbatim", got == code)

    sent = client.models.calls[0]
    check("symbol and timeframe reach the model",
          "ES" in sent["contents"] and "1d" in sent["contents"])
    check("system instruction is attached",
          bool(sent["config"].get("system_instruction")))
    check("temperature is low for code generation",
          sent["config"].get("temperature", 1) <= 0.3,
          str(sent["config"].get("temperature")))

    for label, text in (("empty string", ""), ("whitespace only", "   \n"),
                        ("None", None)):
        try:
            synthesize_strategy_code("x", "ES", client=_StubClient(text))
            check(f"unusable model output ({label}) raises", False, "no exception")
        except SynthesisError:
            check(f"unusable model output ({label}) raises SynthesisError", True)

    try:
        synthesize_strategy_code("x", "ES", client=_StubClient(RuntimeError("boom")))
        check("a failed model call raises", False, "no exception")
    except SynthesisError as e:
        check("a failed model call raises SynthesisError, naming the cause",
              "boom" in str(e), str(e)[:70])

    # No credential is a DIFFERENT state from a failed call: run_campaign
    # catches MissingAPIKey to fall back to the template and says so.
    saved = {v: os.environ.pop(v, None) for v in API_KEY_VARS}
    try:
        synthesize_strategy_code("x", "ES")
        check("no API key raises", False, "no exception")
    except MissingAPIKey:
        check("no API key raises MissingAPIKey, not SynthesisError", True)
    except RuntimeError as e:
        # google-genai absent entirely - still a raise, never a usable client.
        check("no API key raises MissingAPIKey, not SynthesisError",
              isinstance(e, MissingAPIKey), f"{type(e).__name__}: {e}")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    check("MissingAPIKey and SynthesisError are distinguishable",
          not issubclass(SynthesisError, MissingAPIKey)
          and not issubclass(MissingAPIKey, SynthesisError))


def test_scaffold_is_honest():
    print("\nSCAFFOLD BOUNDARY")
    for name in ("propose_goals", "prioritise", "review", "main"):
        fn = getattr(tier1_master, name)
        try:
            if name == "propose_goals":
                fn({})
            elif name == "prioritise":
                fn([])
            elif name == "review":
                fn(None)
            else:
                fn()
            check(f"{name}() does not fake a result", False, "returned a value")
        except NotImplementedError:
            check(f"{name}() raises NotImplementedError rather than a placeholder", True)
        except Exception as exc:
            check(f"{name}() raises NotImplementedError rather than a placeholder",
                  False, f"raised {type(exc).__name__}")


if __name__ == "__main__":
    test_intent_routing()
    test_parse_symbols()
    test_parse_timeframe()
    test_resolve_ruleset()
    test_read_vault()
    test_run_campaign_routes()
    test_synthesis()
    test_scaffold_is_honest()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
