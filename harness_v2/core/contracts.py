"""Frozen v0.1 records for grading, failures, and experiment runs.

FailureCase and ImprovementProposal field names match docs/harness_v2/design_v0.1.md.
Nothing here invents a numeric score for a missing item.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GradingStatus = Literal["validated", "review_required", "failed"]

FAILURE_CASE_FIELDS = (
    "case_id",
    "problem_id",
    "input_version",
    "model_version",
    "harness_version",
    "rubric_version",
    "predicted_score",
    "reference_score",
    "rubric_results",
    "evidence",
    "errors",
    "uncertainty",
    "execution_log",
)

IMPROVEMENT_PROPOSAL_FIELDS = (
    "proposal_id",
    "baseline_commit",
    "candidate_commit",
    "cause",
    "changed_components",
    "expected_effect",
    "evaluation_results",
    "regressions",
    "teacher_reviews",
    "decision",
)


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
    evidence_absent: bool = False
    absent_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "max_points": self.max_points,
            "score": self.score,
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "math_reason": self.math_reason,
            "uncertainty": self.uncertainty,
            "review_reason": self.review_reason,
            "evidence_absent": self.evidence_absent,
            "absent_reason": self.absent_reason,
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
    problem_id: str
    input_version: str
    model_version: str
    harness_version: str
    rubric_version: str
    predicted_score: float | None
    reference_score: float | None
    rubric_results: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    errors: list[str]
    uncertainty: float | None
    execution_log: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImprovementProposal:
    proposal_id: str
    baseline_commit: str
    candidate_commit: str
    cause: str
    changed_components: list[str]
    expected_effect: str
    evaluation_results: dict[str, Any]
    regressions: list[dict[str, Any]]
    teacher_reviews: list[dict[str, Any]]
    decision: str

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
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.attempts and self.attempt_count == 1:
            self.attempts = [{"latency_ms": self.latency_ms, "status": self.status,
                              "error_class": "", "input_tokens": self.input_tokens,
                              "output_tokens": self.output_tokens}]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
