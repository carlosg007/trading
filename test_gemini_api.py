#!/usr/bin/env python3
"""
Connectivity check for the Google AI Synthesizer.

Verifies `GEMINI_API_KEY` is present, the google-genai SDK imports, enumerates
the models the key can actually reach, and round-trips a minimal prompt against
the pinned synthesis model plus a set of fallback candidates.

    python test_gemini_api.py

This is the procedure `agents/tier1_master.py` points at when `DEFAULT_MODEL`
starts 404ing: run it, read the summary, pin the successor.

Unlike the suites in `tests/`, this one needs a network and a live API key, so
it is deliberately not part of that directory.

**Listing a model is not evidence it can be called.** `client.models.list()`
advertises models that 404 on `generateContent` for a given key - the 2.5
family went that way on 2026-08-15 with "no longer available to new users".
Only section 5, which actually calls the model, is evidence.

Exit status tracks the PINNED model specifically, not the batch: a run where
some other candidate answered and `DEFAULT_MODEL` did not is a broken
synthesizer, and exiting 0 on it would be the failure this file exists to
catch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# The model the repo actually uses, read from the source of truth rather than
# retyped here. A hardcoded copy drifts, and a connectivity check that verifies
# a model nobody calls is worse than no check - it reports PASS for a
# synthesizer that cannot run.
try:
    from agents.tier1_master import DEFAULT_MODEL
except Exception as exc:                                      # noqa: BLE001
    print(f"WARN: could not read DEFAULT_MODEL from agents.tier1_master: "
          f"{type(exc).__name__}: {exc}")
    DEFAULT_MODEL = None

# Fallbacks, tried after the pinned model so a failure comes with a successor
# to pin. Retired entries are kept on purpose: the 404 stays on the record
# instead of being rediscovered next time.
FALLBACK_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
    "gemini-flash-latest",
    "gemini-2.5-pro",       # 404s: "no longer available to new users" (2026-08-15)
    "gemini-2.5-flash",     # 404s: as above
    "gemini-2.0-flash",     # retired
    "gemini-1.5-pro",       # retired
]

PROMPT = "Ping. Respond only with: PONG"


def candidates() -> list[str]:
    """The pinned model first, then fallbacks, without duplicates."""
    ordered = ([DEFAULT_MODEL] if DEFAULT_MODEL else []) + FALLBACK_MODELS
    seen, out = set(), []
    for m in ordered:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def main() -> int:
    models = candidates()

    print("=== 1. Environment ===")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("FAIL: GEMINI_API_KEY is not set in os.environ.")
        print("Run: export GEMINI_API_KEY='your-key-here'")
        return 1
    # Masked, never printed whole: this output gets pasted into issues and chat.
    masked = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    print(f"OK: GEMINI_API_KEY present ({masked}, length {len(api_key)})")
    print(f"Pinned DEFAULT_MODEL: {DEFAULT_MODEL or 'UNKNOWN (import failed)'}")

    print("\n=== 2. SDK import ===")
    try:
        from google import genai
    except ImportError as exc:
        print(f"FAIL: cannot import google.genai: {exc}")
        print("Run: uv pip install google-genai")
        return 1
    print(f"OK: google-genai {getattr(genai, '__version__', 'unknown')}")

    print("\n=== 3. Client ===")
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:                                  # noqa: BLE001
        print(f"FAIL: could not build client: {type(exc).__name__}: {exc}")
        return 1
    print("OK: client built")

    print("\n=== 4. Advertised models ===")
    available: set[str] = set()
    try:
        for model in client.models.list():
            name = getattr(model, "name", "") or ""
            available.add(name.split("/")[-1])
            actions = getattr(model, "supported_actions", None) or []
            print(f"  {name}{f'  [{chr(44).join(actions)}]' if actions else ''}")
    except Exception as exc:                                  # noqa: BLE001
        # Not fatal. The listing is informational; section 5 is the evidence.
        print(f"WARN: could not list models: {type(exc).__name__}: {exc}")
    if available:
        print(f"({len(available)} models visible to this key)")

    print("\n=== 5. Generation ping ===")
    results: dict[str, str] = {}
    for model in models:
        pin = " [PINNED]" if model == DEFAULT_MODEL else ""
        listed = "listed" if model in available else "not listed"
        print(f"--> {model} ({listed}){pin} ...")
        try:
            response = client.models.generate_content(model=model, contents=PROMPT)
            text = (response.text or "").strip()
        except Exception as exc:                              # noqa: BLE001
            results[model] = f"FAIL ({type(exc).__name__}: {exc})"
            print(f"    FAIL: {type(exc).__name__}: {exc}")
            continue
        if not text:
            results[model] = "FAIL (empty response)"
            print("    FAIL: empty response")
            continue
        results[model] = f"OK ({text})"
        print(f"    OK: {text}")

    print("\n=== Summary ===")
    for model in models:
        print(f"{'* ' if model == DEFAULT_MODEL else '  '}{model}: {results[model]}")

    working = [m for m in models if results[m].startswith("OK")]
    print(f"\n{len(working)}/{len(models)} model(s) responded.")

    if DEFAULT_MODEL is None:
        print("RESULT: FAIL - could not determine the pinned model, so nothing "
              "was actually verified.")
        return 1
    if results.get(DEFAULT_MODEL, "").startswith("OK"):
        print(f"RESULT: PASS - the pinned model {DEFAULT_MODEL} is reachable.")
        return 0

    print(f"RESULT: FAIL - the pinned model {DEFAULT_MODEL} did NOT respond. "
          f"Synthesis will fall back to placeholder boilerplate.")
    if working:
        print(f"Pin one of these in agents/tier1_master.py instead: "
              f"{', '.join(working)}")
    else:
        print("No candidate responded. Check the key, the quota, and the network "
              "before repinning.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
