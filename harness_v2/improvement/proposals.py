"""Draft a proposal from failure cases. The proposal is not applied."""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone

from harness_v2.core.contracts import FailureCase, ImprovementProposal


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_proposal(cases: list[FailureCase]) -> ImprovementProposal | None:
    if not cases:
        return None
    counts = Counter(case.failure_kind for case in cases)
    kind, count = counts.most_common(1)[0]
    matched = [case.case_id for case in cases if case.failure_kind == kind]
    target = "prompt" if "parse" in kind or "json" in kind else "rubric"
    return ImprovementProposal(
        proposal_id=uuid.uuid4().hex,
        source_case_ids=matched,
        target=target,
        problem=f"{count} case(s) failed check {kind}",
        recommended_action="A person reviews these cases before any prompt or rubric change.",
        severity="high" if count >= 3 else "medium",
        pattern_count=count,
        status="proposed",
        experiment_id=cases[0].experiment_id,
        created_at=_now(),
    )
