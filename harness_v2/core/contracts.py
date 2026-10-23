"""Frozen v0.1 records for grading, failures, and experiment runs.

ItemResult field names are the ones the workstream named. FailureCase and
ImprovementProposal are the v0.1 set used by this harness: the design file
was not in the repository, so these fields are the contract until a design
doc replaces them. Nothing here invents a numeric score for a missing item.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GradingStatus = Literal["validated", "review_required", "failed"]
ProposalStatus = Literal["proposed", "accepted", "rejected"]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class RubricItem:
    item_id: str
    max_points: float
    description: str = ""
    acceptable_expression: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceSpan:
    text: str
    start: int | None = None
    end: int | None = None
    source: str = "answer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ItemResult:
    item_id: str
    max_points: float
    score: float | None
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    math_reason: str = ""
    uncertainty: float | None = None
    review_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "max_points": self.max_points,
            "score": self.score,
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "math_reason": self.math_reason,
            "uncertainty": self.uncertainty,
            "review_reason": self.review_reason,
        }


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GradingResult:
    status: GradingStatus
    items: list[ItemResult]
    total_score: float | None
    max_points: float | None
    failure_reason: str = ""
    request_id: str = ""
    provider: str = ""
    model_id: str = ""
    raw_response: str = ""
    checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
            "total_score": self.total_score,
            "max_points": self.max_points,
            "failure_reason": self.failure_reason,
            "request_id": self.request_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "raw_response": self.raw_response,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass
class FailureCase:
    case_id: str
    experiment_id: str
    request_id: str
    input_id: str
    provider: str
    model_id: str
    rubric_version: str
    prompt_version: str
    failure_kind: str
    observed: str
    expected_check: str
    item_id: str = ""
    artifact_ref: str = ""
    commit: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImprovementProposal:
    proposal_id: str
    source_case_ids: list[str]
    target: str
    problem: str
    recommended_action: str
    severity: str
    pattern_count: int
    status: ProposalStatus = "proposed"
    experiment_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    experiment_id: str
    commit: str
    provider: str
    model_id: str
    rubric_version: str
    prompt_version: str
    input_ids: list[str]
    outputs: dict[str, Any]
    checks: list[dict[str, Any]]
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    failure_reason: str
    proposal_link: str
    request_id: str
    run_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = _jsonable(self.outputs)
        payload["checks"] = _jsonable(self.checks)
        return payload


@dataclass
class ProviderExchange:
    request_id: str
    model_requested: str
    model_returned: str
    settings: dict[str, Any]
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    raw_response: str
    status: str
    provider_response_id: str = ""
    error: str = ""
    attempt_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
