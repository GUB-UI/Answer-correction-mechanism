"""Draft one proposal from failure cases. The proposal is not applied.

A cycle names one changed component. evaluation_results stays empty until a
later evaluation fills the design metrics.
"""

from __future__ import annotations

import uuid

from harness_v2.core.contracts import FailureCase, ImprovementProposal


def _cause(cases: list[FailureCase]) -> str:
    counts: dict[str, int] = {}
    for case in cases:
        label = case.errors[0] if case.errors else "unspecified"
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=lambda label: (counts[label], label))


def _one_component(cause: str) -> str:
    text = cause.lower()
    if "json" in text or "parse" in text:
        return "prompt"
    if "evidence" in text:
        return "recognition"
    if "sympy" in text or "math" in text:
        return "validator"
    return "rubric"


def build_proposal(
    cases: list[FailureCase],
    *,
    baseline_commit: str = "",
) -> ImprovementProposal | None:
    if not cases:
        return None
    cause = _cause(cases)
    return ImprovementProposal(
        proposal_id=uuid.uuid4().hex,
        baseline_commit=baseline_commit,
        candidate_commit="",
        cause=cause,
        changed_components=[_one_component(cause)],
        expected_effect="One component change, reviewed by a person before it is applied.",
        evaluation_results={},
        regressions=[],
        teacher_reviews=[],
        decision="pending",
    )
