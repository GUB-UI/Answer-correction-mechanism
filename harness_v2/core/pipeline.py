"""One input through OCR import, grading, validation, and storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_v2 import __version__
from harness_v2.core.contracts import FailureCase, GradingResult, RunRecord
from harness_v2.evaluation.independent import prepare_independent_input, scrub_score_text
from harness_v2.grading.rubric import rubric_from_data
from harness_v2.grading.scorer import ScoringParseError, build_prompt, items_from_completion
from harness_v2.grading.validator import validate
from harness_v2.improvement.proposals import build_proposal
from harness_v2.providers.gemma_ai_studio import QuotaExceeded
from harness_v2.recognition.candidates import readings_from_ocr
from harness_v2.recognition.ocr import XiaomiOCR
from harness_v2.recognition.structure import structure_text
from harness_v2.storage.sqlite_store import RunStore


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class Pipeline:
    def __init__(self, store: RunStore, provider: Any, *, independent: bool = True) -> None:
        self.store = store
        self.provider = provider
        self.independent = independent
        self.ocr = XiaomiOCR(store.artifacts)

    def run(self, case: dict[str, Any], *, experiment_id: str, commit: str) -> RunRecord:
        request_id = str(case.get("request_id") or uuid.uuid4().hex)
        public = prepare_independent_input(case)
        answer_text, ocr_name = self._answer_text(public, request_id)
        rubric = rubric_from_data(public.get("rubric") or [])
        prompt = build_prompt(str(public.get("problem_text") or ""), answer_text, rubric)
        if self.independent:
            prompt = scrub_score_text(prompt)
        self.store.artifacts.write_text(f"prompts/{request_id}.txt", prompt)
        exchange = self.provider.generate(prompt, request_id=request_id)
        raw_name = f"raw/{request_id}.txt"
        self.store.artifacts.write_text(raw_name, exchange.raw_response)
        result = self._grade(exchange, rubric, answer_text, request_id)
        proposal_link = self._maybe_propose(result, case, public, experiment_id, commit, request_id, raw_name)
        record = RunRecord(
            experiment_id=experiment_id,
            commit=commit,
            provider=type(self.provider).__name__,
            model_id=exchange.model_returned or exchange.model_requested,
            rubric_version=str(public.get("rubric_version") or "v0"),
            prompt_version=str(public.get("prompt_version") or "v0"),
            input_ids=[str(public.get("input_id") or request_id)],
            outputs={
                "grading": result.to_dict(),
                "ocr_artifact": ocr_name,
                "raw_response_artifact": raw_name,
                "regions": len(structure_text(answer_text)),
                "candidates": [
                    item.candidate_id
                    for item in readings_from_ocr(answer_text, public.get("alternatives"))
                ],
                "image_supplied": bool(public.get("image_path")),
            },
            checks=[check.to_dict() for check in result.checks],
            latency_ms=exchange.latency_ms,
            input_tokens=exchange.input_tokens,
            output_tokens=exchange.output_tokens,
            failure_reason=result.failure_reason,
            proposal_link=proposal_link,
            request_id=request_id,
            created_at=_now(),
        )
        return self.store.save_run(record)

    def record_quota(
        self,
        case: dict[str, Any],
        error: QuotaExceeded,
        *,
        experiment_id: str,
        commit: str,
    ) -> RunRecord:
        exchange = error.exchange
        request_id = exchange.request_id or str(case.get("request_id") or uuid.uuid4().hex)
        raw_name = f"raw/{request_id}.txt"
        self.store.artifacts.write_text(raw_name, exchange.raw_response)
        record = RunRecord(
            experiment_id=experiment_id,
            commit=commit,
            provider=type(self.provider).__name__,
            model_id=exchange.model_requested,
            rubric_version=str(case.get("rubric_version") or "v0"),
            prompt_version=str(case.get("prompt_version") or "v0"),
            input_ids=[str(case.get("input_id") or request_id)],
            outputs={"exchange": exchange.to_dict(), "raw_response_artifact": raw_name},
            checks=[],
            latency_ms=exchange.latency_ms,
            input_tokens=exchange.input_tokens,
            output_tokens=exchange.output_tokens,
            failure_reason="quota",
            proposal_link="",
            request_id=request_id,
            created_at=_now(),
        )
        return self.store.save_run(record)

    def _answer_text(self, case: dict[str, Any], request_id: str) -> tuple[str, str]:
        import_path = str(case.get("ocr_import_path") or "")
        if import_path:
            artifact = self.ocr.import_file(Path(import_path), ocr_id=request_id)
            return artifact.raw_text, artifact.artifact_name
        answer = str(case.get("answer_text") or "")
        if answer:
            name = f"ocr/{request_id}.txt"
            self.store.artifacts.write_text(name, answer)
            return answer, name
        image = str(case.get("image_path") or "")
        if image:
            self.ocr.run_cpu(Path(image), ocr_id=request_id)
        return "", ""

    def _grade(self, exchange: Any, rubric: list, answer_text: str, request_id: str) -> GradingResult:
        if exchange.status != "ok":
            return GradingResult(
                status="failed",
                items=[],
                total_score=None,
                max_points=sum(item.max_points for item in rubric) if rubric else None,
                failure_reason=exchange.error or exchange.status,
                request_id=request_id,
                provider=type(self.provider).__name__,
                model_id=exchange.model_returned or exchange.model_requested,
                raw_response=exchange.raw_response,
            )
        try:
            items, total, _fence = items_from_completion(exchange.raw_response, rubric)
        except ScoringParseError as exc:
            return GradingResult(
                status="failed",
                items=[],
                total_score=None,
                max_points=sum(item.max_points for item in rubric) if rubric else None,
                failure_reason=str(exc),
                request_id=request_id,
                provider=type(self.provider).__name__,
                model_id=exchange.model_returned or exchange.model_requested,
                raw_response=exchange.raw_response,
            )
        return validate(
            rubric,
            items,
            total,
            answer_text,
            request_id=request_id,
            provider=type(self.provider).__name__,
            model_id=exchange.model_returned or exchange.model_requested,
            raw_response=exchange.raw_response,
        )

    def _maybe_propose(
        self,
        result: GradingResult,
        case: dict[str, Any],
        public: dict[str, Any],
        experiment_id: str,
        commit: str,
        request_id: str,
        raw_name: str,
    ) -> str:
        if result.status == "validated":
            return ""
        errors = [check.name for check in result.checks if not check.ok]
        if result.failure_reason and result.failure_reason not in errors:
            errors.insert(0, result.failure_reason)
        uncertainties = [
            item.uncertainty for item in result.items if isinstance(item.uncertainty, (int, float))
        ]
        failure = FailureCase(
            case_id=uuid.uuid4().hex,
            problem_id=str(case.get("problem_id") or public.get("input_id") or request_id),
            input_version=str(case.get("input_version") or ""),
            model_version=result.model_id,
            harness_version=__version__,
            rubric_version=str(public.get("rubric_version") or "v0"),
            predicted_score=result.total_score,
            reference_score=_optional_score(case.get("reference_score")),
            rubric_results=[item.to_dict() for item in result.items],
            evidence=[span.to_dict() for item in result.items for span in item.evidence_spans],
            errors=errors,
            uncertainty=max(uncertainties) if uncertainties else None,
            execution_log=[
                {
                    "request_id": request_id,
                    "experiment_id": experiment_id,
                    "artifact": raw_name,
                    "status": result.status,
                }
            ],
        )
        proposal = build_proposal([failure], baseline_commit=commit)
        assert proposal is not None
        self.store.artifacts.write_json(f"failures/{failure.case_id}.json", failure.to_dict())
        self.store.artifacts.write_json(f"proposals/{proposal.proposal_id}.json", proposal.to_dict())
        return proposal.proposal_id
