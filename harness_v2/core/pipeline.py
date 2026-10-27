"""One input through OCR import, grading, validation, and storage."""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_v2.core.contracts import Check, GradingResult, RunRecord
from harness_v2.evaluation.independent import grading_allowlist, prepare_independent_input
from harness_v2.evaluation.review import review_disagreement
from harness_v2.evaluation.reference import failure_from_run
from harness_v2.grading.rubric import rubric_from_data
from harness_v2.grading.scorer import ScoringParseError, build_prompt, items_from_completion
from harness_v2.grading.validator import validate
from harness_v2.improvement.proposals import build_proposal
from harness_v2.providers.gemma_ai_studio import QuotaExceeded
from harness_v2.recognition.candidates import readings_from_ocr
from harness_v2.recognition.ocr import OCRArtifact, XiaomiOCR
from harness_v2.recognition.structure import structure_text
from harness_v2.storage.sqlite_store import RunStore
from harness_v2.storage.artifacts import ImmutableArtifactError


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Pipeline:
    def __init__(
        self,
        store: RunStore,
        provider: Any,
        *,
        independent: bool = True,
        codex_reviewer: Any | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.independent = independent
        self.codex_reviewer = codex_reviewer
        self.ocr = XiaomiOCR(store.artifacts)

    def _allocate_attempt(self, request_id: str) -> tuple[str, int]:
        key = hashlib.sha256(request_id.encode()).hexdigest()
        attempt = 1
        while True:
            artifact_id = request_id if attempt == 1 else f"{request_id}--attempt-{attempt}"
            try:
                self.store.artifacts.write_json(f"attempts/{key}/{attempt}.json", {
                    "request_id": request_id, "artifact_id": artifact_id, "run_attempt": attempt,
                })
            except ImmutableArtifactError:
                attempt += 1
                continue
            # Old/crashed runs may predate reservation markers.
            if any((self.store.artifacts.root / folder / f"{artifact_id}.txt").exists()
                   for folder in ("ocr_private", "ocr", "grading_private", "prompts", "raw")):
                attempt += 1
                continue
            return artifact_id, attempt

    def run(self, case: dict[str, Any], *, experiment_id: str, commit: str) -> RunRecord:
        case = dict(case)
        request_id = str(case.get("request_id") or uuid.uuid4().hex)
        case["request_id"] = request_id
        artifact_id, attempt = self._allocate_attempt(request_id)
        context = {"artifact_id": artifact_id, "run_attempt": attempt, "attempt_count": 0, "attempts": [],
                   "case_metadata": {"problem_id": str(case.get("problem_id") or case.get("input_id") or request_id),
                                     "input_version": str(case.get("input_version") or "")}}
        if "experiment_case" in case:
            context["experiment_case"] = dict(case["experiment_case"])
        started = time.perf_counter()
        try:
            return self._run(case, experiment_id=experiment_id, commit=commit, context=context)
        except QuotaExceeded as exc:
            exc.record = self.record_quota(case, exc, experiment_id=experiment_id, commit=commit,
                                           context=context)
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            result = GradingResult(status="failed", items=[], total_score=None, max_points=None,
                                   failure_reason=reason, request_id=request_id,
                                   provider=type(self.provider).__name__,
                                   model_id=str(getattr(self.provider, "model_id", "")))
            proposal_link = self._maybe_propose(result, case, {}, experiment_id, commit,
                                                request_id, context.get("raw_response_artifact", ""))
            return self.store.save_run(RunRecord(
                experiment_id=experiment_id, commit=commit, provider=type(self.provider).__name__,
                model_id=result.model_id, rubric_version=str(case.get("rubric_version") or "v0"),
                prompt_version=str(case.get("prompt_version") or "v0"),
                input_ids=[str(case.get("input_id") or request_id)],
                outputs={**context, "grading": result.to_dict(), "error_class": type(exc).__name__},
                checks=[], latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=None, output_tokens=None, failure_reason=reason,
                proposal_link=proposal_link, request_id=request_id, created_at=_now(),
            ))

    def _run(self, case: dict[str, Any], *, experiment_id: str, commit: str,
             context: dict[str, Any]) -> RunRecord:
        request_id = case["request_id"]
        artifact_id = context["artifact_id"]
        artifact = self._answer_text(case, artifact_id)
        answer_text = artifact.raw_text if artifact is not None else ""
        if artifact is not None:
            self._verify_ocr(artifact)
            context.update({"ocr_artifact": artifact.artifact_name, "ocr_sha256": artifact.sha256,
                            "ocr_redacted": artifact.redacted_name,
                            "ocr_metadata": {**artifact.metadata, "engine": artifact.engine,
                                "revision": artifact.revision, "dtype": artifact.dtype,
                                "elapsed_s": artifact.elapsed_s,
                                "source": artifact.metadata.get("source", artifact.source_name)}})
            if case.get("input_sha256") and artifact.sha256 != case["input_sha256"]:
                raise RuntimeError("grading input hash changed after dataset verification")
        raw_view = {
            "problem_text": case.get("problem_text") or case.get("problem") or "",
            "answer_text": answer_text,
            "rubric": case.get("rubric") or [],
        }
        # Validate before allowlisting so malformed rubric rows cannot disappear.
        rubric_from_data(raw_view["rubric"])
        # Reference labels are post-hoc data even when independent mode is off.
        view = prepare_independent_input(raw_view)
        rubric = rubric_from_data(view.get("rubric") or [])
        prompt_answer = str(view.get("answer_text") or "")
        grading_name = artifact.artifact_name if artifact is not None else ""
        grading_hash = hashlib.sha256(prompt_answer.encode("utf-8")).hexdigest()
        if prompt_answer != answer_text:
            grading_name = f"grading_private/{artifact_id}.txt"
            _, grading_hash = self.store.artifacts.write_private_text(grading_name, prompt_answer)
        prompt = build_prompt(str(view.get("problem_text") or ""), prompt_answer, rubric,
                              str(case.get("prompt_version") or "v0"))
        prompt_name = f"prompts/{artifact_id}.txt"
        context.update({"prompt_artifact": prompt_name,
                        "ocr_artifact": artifact.artifact_name if artifact else "",
                        "ocr_sha256": artifact.sha256 if artifact else "",
                        "grading_input_artifact": grading_name, "grading_input_sha256": grading_hash})
        self.store.artifacts.write_text(prompt_name, prompt)
        kwargs = {"settings": case["generation_settings"]} if "generation_settings" in case else {}
        exchange = self.provider.generate(prompt, request_id=request_id, **kwargs)
        raw_name = f"raw/{artifact_id}.txt"
        context.update({"raw_response_artifact": raw_name, "attempt_count": exchange.attempt_count,
                        "attempts": exchange.attempts, "settings": exchange.settings})
        self.store.artifacts.write_text(raw_name, exchange.raw_response)
        result = self._grade(exchange, rubric, prompt_answer, request_id)
        review_payload = self._maybe_review(result, view, rubric, request_id)
        proposal_link = self._maybe_propose(result, case, view, experiment_id, commit, request_id, raw_name)
        record = RunRecord(
            experiment_id=experiment_id,
            commit=commit,
            provider=type(self.provider).__name__,
            model_id=exchange.model_returned or exchange.model_requested,
            rubric_version=str(case.get("rubric_version") or "v0"),
            prompt_version=str(case.get("prompt_version") or "v0"),
            input_ids=[str(case.get("input_id") or request_id)],
            outputs={
                **context,
                "grading": result.to_dict(),
                "ocr_artifact": artifact.artifact_name if artifact is not None else "",
                "ocr_redacted": artifact.redacted_name if artifact is not None else "",
                "ocr_sha256": artifact.sha256 if artifact is not None else "",
                "grading_input_artifact": grading_name,
                "grading_input_sha256": grading_hash,
                "ocr_metadata": {
                    **artifact.metadata,
                    "source": artifact.metadata.get("source", artifact.source_name),
                    "engine": artifact.engine,
                    "revision": artifact.revision,
                    "dtype": artifact.dtype,
                    "elapsed_s": artifact.elapsed_s,
                } if artifact is not None else {},
                "raw_response_artifact": raw_name,
                "regions": len(structure_text(answer_text)),
                "candidates": [
                    item.candidate_id
                    for item in readings_from_ocr(answer_text, case.get("alternatives"))
                ],
                "image_supplied": bool(case.get("image_path")),
                "settings": exchange.settings,
                "provider_response_id": exchange.provider_response_id,
                "attempt_count": exchange.attempt_count,
                "attempts": exchange.attempts,
                "independent_review": review_payload,
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
        context: dict[str, Any] | None = None,
    ) -> RunRecord:
        if getattr(error, "record", None) is not None:
            return error.record
        exchange = error.exchange
        request_id = exchange.request_id or str(case.get("request_id") or uuid.uuid4().hex)
        if context is None:
            artifact_id, attempt = self._allocate_attempt(request_id)
            context = {"artifact_id": artifact_id, "run_attempt": attempt, "attempt_count": 0, "attempts": []}
        raw_name = f"raw/{context['artifact_id']}.txt"
        self.store.artifacts.write_text(raw_name, exchange.raw_response)
        record = RunRecord(
            experiment_id=experiment_id,
            commit=commit,
            provider=type(self.provider).__name__,
            model_id=exchange.model_requested,
            rubric_version=str(case.get("rubric_version") or "v0"),
            prompt_version=str(case.get("prompt_version") or "v0"),
            input_ids=[str(case.get("input_id") or request_id)],
            outputs={
                **context,
                "exchange": exchange.to_dict(),
                "raw_response_artifact": raw_name,
                "grading": {"status": "failed", "failure_reason": "quota"},
                "settings": exchange.settings,
                "provider_response_id": exchange.provider_response_id,
                "attempt_count": exchange.attempt_count,
                "attempts": exchange.attempts,
            },
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

    def _answer_text(self, case: dict[str, Any], request_id: str) -> OCRArtifact | None:
        import_path = str(case.get("ocr_import_path") or "")
        if import_path:
            return self.ocr.import_file(Path(import_path), ocr_id=request_id,
                                        metadata=case.get("ocr_metadata"),
                                        metadata_path=Path(case["ocr_metadata_path"]) if case.get("ocr_metadata_path") else None)
        answer = case.get("answer_text")
        if isinstance(answer, str) and answer:
            return self.ocr.store_text(answer, ocr_id=request_id, source_name="answer_text")
        image = str(case.get("image_path") or "")
        if image:
            return self.ocr.run_cpu(Path(image), ocr_id=request_id)
        return None

    def _verify_ocr(self, artifact: OCRArtifact) -> None:
        data = self.store.artifacts.read_bytes(artifact.artifact_name)
        digest = hashlib.sha256(data).hexdigest()
        if digest != artifact.sha256 or data.decode("utf-8") != artifact.raw_text:
            raise RuntimeError("grading input is not the stored OCR artifact")

    def _maybe_review(self, result: GradingResult, view: dict[str, Any], rubric: list, request_id: str) -> dict[str, Any]:
        if self.codex_reviewer is None or result.status != "validated":
            return {}
        payload = grading_allowlist(view)
        try:
            outcome = self.codex_reviewer.review_independent(payload, request_id=f"{request_id}-review")
            saved = outcome.to_dict() if hasattr(outcome, "to_dict") else {"text": str(outcome)}
            if not isinstance(saved, dict):
                saved = {"status": "failed", "error": "invalid reviewer outcome", "text": ""}
            reason = review_disagreement(saved, result, rubric, str(payload.get("answer_text") or ""))
        except Exception as exc:
            saved = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__,
                     "request_id": f"{request_id}-review", "text": ""}
            reason = "independent_review_exception"
        if reason:
            result.status = "review_required"
            result.failure_reason = reason
            result.checks.append(Check("independent_review", False, reason))
        else:
            result.checks.append(Check("independent_review", True, ""))
        return saved

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
            items, total, _fence, repaired, ambiguous, marker = items_from_completion(
                exchange.raw_response, rubric)
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
        result = validate(
            rubric,
            items,
            total,
            answer_text,
            request_id=request_id,
            provider=type(self.provider).__name__,
            model_id=exchange.model_returned or exchange.model_requested,
            raw_response=exchange.raw_response,
        )
        result.parse_repaired = repaired
        result.parse_ambiguous = ambiguous
        result.parse_marker = marker
        if ambiguous and result.status != "failed":
            result.status = "review_required"
            parts = [part for part in result.failure_reason.split(",") if part]
            if "parse_ambiguous" not in parts:
                parts.append("parse_ambiguous")
            result.failure_reason = ",".join(parts)
            result.checks.append(Check("parse_ambiguous", False, marker or "newline_or_latex"))
        return result

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
        failure = failure_from_run({
            "request_id": request_id, "experiment_id": experiment_id,
            "input_ids": [str(case.get("input_id") or request_id)], "model_id": result.model_id,
            "rubric_version": str(case.get("rubric_version") or "v0"),
            "failure_reason": result.failure_reason, "checks": [check.to_dict() for check in result.checks],
            "outputs": {"grading": result.to_dict(), "raw_response_artifact": raw_name,
                        "case_metadata": {"problem_id": str(case.get("problem_id") or case.get("input_id") or request_id),
                                          "input_version": str(case.get("input_version") or "")}},
        }, reference_score=case.get("reference_score"), reference_items=case.get("reference_items"))
        if failure is None:
            return ""
        proposal = build_proposal([failure], baseline_commit=commit)
        assert proposal is not None
        self.store.artifacts.write_json(f"failures/{failure.case_id}.json", failure.to_dict())
        self.store.artifacts.write_json(f"proposals/{proposal.proposal_id}.json", proposal.to_dict())
        return proposal.proposal_id
