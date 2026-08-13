"""
agents.tier1_master - the CIO agent.

Location:  ~/src/trading/agents/tier1_master.py

SCAFFOLD ONLY. Interfaces and constraints are defined here; the agent logic is
not written yet. Every function below raises NotImplementedError rather than
returning a plausible-looking placeholder, because a stub that silently returns
an empty result is exactly how a research pipeline starts reporting numbers
nobody generated.

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

from dataclasses import dataclass, field
from typing import Any

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
