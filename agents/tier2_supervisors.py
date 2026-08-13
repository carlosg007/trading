"""
agents.tier2_supervisors - the gates a result has to pass.

Location:  ~/src/trading/agents/tier2_supervisors.py

SCAFFOLD ONLY. Interfaces are defined here; the logic is not written yet.
Every check below raises NotImplementedError rather than returning a passing
verdict, because a supervisor that defaults to "approved" while unimplemented
is worse than no supervisor at all - it manufactures confidence.

What this tier is for
---------------------
Tier 2 decides whether a result is allowed to count. Two supervisors:

  - PROP-FIRM COMPLIANCE: trailing drawdown, daily loss limit, consistency
    rules. A strategy that breaches is rejected regardless of its Sharpe,
    because the account is closed before the edge has time to show up.
  - OOS VALIDATION: the Phase 3 gate. In-sample performance on Databento data
    is a candidate; a strategy is only valid if it survives on out-of-sample
    NT8 data. A Sharpe that collapses across that boundary means overfitting,
    and the strategy is discarded rather than retuned.

Why these are agents rather than asserts
----------------------------------------
The numeric checks themselves are not agent work - `backtest.engine` already
computes the breach and the stats, deterministically, and that is where they
belong. What Tier 2 adds is judgement over the *context* the numbers arrived
in: how many variants were tested to find this one, whether the OOS window
overlaps anything already used for selection, whether a "fix" between runs was
a bug fix or a fit to the test set. Those are the questions that decide whether
a number is evidence, and they are not expressible as a threshold.

The deterministic parts stay deterministic. A supervisor may read
`BacktestResult.breach` and `BacktestResult.stats`; it may not recompute them,
and it may not overrule them.

Boundary
--------
A supervisor's rejection is final. Tier 1 may not overrule it, and neither may
a human asking to "see if the strategy works" with the constraint relaxed. If
a constraint is genuinely wrong, it gets changed in `BacktestConfig` and
everything is re-run - not waived for one result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from google import genai
    _GENAI_IMPORT_ERROR: Exception | None = None
except ImportError as e:      # pragma: no cover - depends on the environment
    genai = None
    _GENAI_IMPORT_ERROR = e


@dataclass
class Verdict:
    """
    One supervisor's decision.

    `reasons` is required, not decorative: a rejection nobody can read is a
    rejection nobody can learn from, and an approval nobody can read is not
    reviewable.
    """

    supervisor: str
    approved: bool
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class PropFirmSupervisor:
    """
    Enforces the prop account rules.

    Reads the breach dict the engine already produced. Does not recompute it -
    two implementations of a drawdown rule is one more than the number that can
    be right.
    """

    name = "prop_firm"

    def review(self, result: Any) -> Verdict:
        raise NotImplementedError("tier2_supervisors: not implemented yet")


class OOSValidationSupervisor:
    """
    The Phase 3 gate: does the edge survive out of sample?

    Compares an in-sample (Databento) result against an out-of-sample (NT8)
    one and judges whether the difference is degradation within tolerance or
    collapse. Also responsible for catching the subtler failure: an OOS window
    that has already been looked at during selection is no longer out of
    sample, however it is labelled.
    """

    name = "oos_validation"

    def review(self, in_sample: Any, out_of_sample: Any) -> Verdict:
        raise NotImplementedError("tier2_supervisors: not implemented yet")


SUPERVISORS = (PropFirmSupervisor, OOSValidationSupervisor)


def review_all(result: Any, oos_result: Any | None = None) -> list[Verdict]:
    """Run every supervisor. A single rejection is enough to fail the result."""
    raise NotImplementedError("tier2_supervisors: not implemented yet")


def main() -> int:
    raise NotImplementedError("tier2_supervisors: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
