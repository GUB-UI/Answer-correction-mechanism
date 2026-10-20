"""複数答案の一括処理。GPU を GLM-OCR → UniMERNet → 診断LLM の順に1回ずつ切り替える。

答案ごとに3モデルを載せ替えると、27B のロード（数分）が答案の数だけ発生する。
そこでフェーズ単位で全答案を処理し、各モデルのロードを1回にする。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from src.adapters import create_diagnosis_adapter
from src.adapters.base import DiagnosisRunContext
from src.adapters.mock_adapters import MockOCRAdapter
from src.adapters.ocr_adapter import (
    GLMOCRSdkAdapter,
    XiaomiOCRAdapter,
    refine_ocr_result_with_unimernet,
)
from src.config import AppConfig
from src.models import AnswerRecord, OCRResult, utc_now_iso
from src.ocr_runner import run_ocr_for_images
from src.runtime.services import GLM_OCR, LLM, UNIMERNET, XIAOMI_OCR, ServiceManager, get_service_manager
from src.sources import answer_image_paths, student_ocr_results
from src.storage import Storage

ProgressCallback = Callable[[str, int, int, str], None]


@dataclass
class BatchOptions:
    ocr_provider: str = "glmocr_sdk"
    diagnosis_provider: str = "colab_cuda"
    prompt_type: str = "step_by_step_check"
    reuse_existing_ocr: bool = True
    skip_existing_diagnosis: bool = True
    apply_formula: bool = True
    stop_services_after: bool = False


@dataclass
class BatchItem:
    answer_id: str
    student_anonymized_id: str
    problem_id: str
    ocr_id: str = ""
    diagnosis_id: str = ""
    score: float | None = None
    max_score: float | None = None
    status: str = "pending"
    errors: list[str] = field(default_factory=list)


@dataclass
class BatchReport:
    items: list[BatchItem]
    phase_seconds: dict[str, float] = field(default_factory=dict)
    service_events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [asdict(item) for item in self.items],
            "phase_seconds": self.phase_seconds,
            "service_events": self.service_events,
        }


def latest_student_ocr(storage: Storage, answer_id: str) -> OCRResult | None:
    """人の修正済みがあればそれを優先し、なければ最新の OCR を使う。"""
    results = student_ocr_results(storage.get_ocr_results_for_answer(answer_id))
    if not results:
        return None
    corrected = [r for r in results if r.human_corrected]
    pool = corrected or results
    return sorted(pool, key=lambda r: r.created_at)[-1]


def _noop_progress(phase: str, index: int, total: int, message: str) -> None:
    return None


def run_batch(
    storage: Storage,
    config: AppConfig,
    answers: list[AnswerRecord],
    options: BatchOptions | None = None,
    *,
    services: ServiceManager | None = None,
    progress: ProgressCallback | None = None,
    diagnosis_adapter: Any | None = None,
    ocr_adapter: Any | None = None,
    formula_refiner: Callable[[OCRResult], OCRResult] | None = None,
) -> BatchReport:
    opts = options or BatchOptions()
    manager = services or get_service_manager(config)
    notify = progress or _noop_progress
    problems = {p.problem_id: p for p in storage.list_problems()}
    items = {
        a.answer_id: BatchItem(a.answer_id, a.student_anonymized_id, a.problem_id)
        for a in answers
    }
    report = BatchReport(items=list(items.values()))
    use_glm = opts.ocr_provider in {"glmocr_sdk", "glm-ocr"}
    use_xiaomi = opts.ocr_provider in {"xiaomi_ocr", "xiaomi-ocr"}

    # フェーズ1: OCR（GLM-OCR のみ。数式置換は次のフェーズ）
    started = time.monotonic()
    new_ocr: dict[str, OCRResult] = {}
    to_ocr: list[AnswerRecord] = []
    for answer in answers:
        existing = latest_student_ocr(storage, answer.answer_id) if opts.reuse_existing_ocr else None
        if existing is not None:
            items[answer.answer_id].ocr_id = existing.ocr_id
        elif answer_image_paths(answer):
            to_ocr.append(answer)
        else:
            items[answer.answer_id].errors.append("OCR対象の画像も既存OCRもありません")
    if to_ocr:
        if use_xiaomi:
            manager.ensure(XIAOMI_OCR)
        elif use_glm:
            manager.ensure(GLM_OCR)
        if ocr_adapter is not None:
            adapter = ocr_adapter
        elif use_xiaomi:
            adapter = XiaomiOCRAdapter(config)
        elif use_glm:
            adapter = GLMOCRSdkAdapter(config, apply_formula=False)
        else:
            adapter = MockOCRAdapter()
        engine = config.ocr.model_name if (use_glm or use_xiaomi) else "mock"
        for index, answer in enumerate(to_ocr, start=1):
            notify("ocr", index, len(to_ocr), answer.answer_id)
            try:
                result = run_ocr_for_images(
                    adapter,
                    storage,
                    source_kind="student_answer",
                    source_id=answer.answer_id,
                    image_paths=answer_image_paths(answer),
                    engine=engine,
                )
                new_ocr[answer.answer_id] = result
                items[answer.answer_id].ocr_id = result.ocr_id
            except Exception as exc:
                items[answer.answer_id].errors.append(f"OCR: {exc}")
    report.phase_seconds["ocr"] = round(time.monotonic() - started, 1)

    # フェーズ2: UniMERNet で数式・数字領域を置換
    started = time.monotonic()
    if use_glm and opts.apply_formula and config.formula.enabled and new_ocr:
        manager.ensure(UNIMERNET)
        refine = formula_refiner or (lambda r: refine_ocr_result_with_unimernet(r, config))
        for index, (answer_id, result) in enumerate(new_ocr.items(), start=1):
            notify("formula", index, len(new_ocr), answer_id)
            try:
                storage.update_ocr_result(refine(result))
            except Exception as exc:
                items[answer_id].errors.append(f"UniMERNet: {exc}")
    report.phase_seconds["formula"] = round(time.monotonic() - started, 1)

    # フェーズ3: 診断LLM
    started = time.monotonic()
    targets: list[tuple[AnswerRecord, OCRResult]] = []
    for answer in answers:
        item = items[answer.answer_id]
        if not item.ocr_id:
            continue
        if opts.skip_existing_diagnosis:
            done = [
                d
                for d in storage.get_diagnosis_results_for_answer(answer.answer_id)
                if d.ocr_id == item.ocr_id and d.prompt_type == opts.prompt_type
            ]
            if done:
                latest = sorted(done, key=lambda d: d.created_at)[-1]
                item.diagnosis_id = latest.diagnosis_id
                item.score, item.max_score = latest.score, latest.max_score
                item.status = "skipped_existing"
                continue
        ocr = storage.get_ocr_result(item.ocr_id)
        if ocr is None:
            item.errors.append("OCR結果が見つかりません")
            continue
        if answer.problem_id not in problems:
            item.errors.append("問題が見つかりません")
            continue
        targets.append((answer, ocr))
    if targets:
        if opts.diagnosis_provider != "mock":
            manager.ensure(LLM)
        adapter = diagnosis_adapter or create_diagnosis_adapter(opts.diagnosis_provider, config)
        model_name = config.diagnosis.model_name if opts.diagnosis_provider != "mock" else "mock"
        for index, (answer, ocr) in enumerate(targets, start=1):
            notify("diagnosis", index, len(targets), answer.answer_id)
            item = items[answer.answer_id]
            context = DiagnosisRunContext(
                answer_id=answer.answer_id,
                problem_id=answer.problem_id,
                ocr_id=ocr.ocr_id,
                problem=problems[answer.problem_id],
                used_text=ocr.used_text,
                model_name=model_name,
                prompt_type=opts.prompt_type,
            )
            try:
                raw = adapter.run(context)
                diagnosis = adapter.to_diagnosis_result(
                    context,
                    raw,
                    diagnosis_id=storage.generate_id("diag"),
                    created_at=utc_now_iso(),
                )
                storage.save_diagnosis_result(diagnosis)
                item.diagnosis_id = diagnosis.diagnosis_id
                item.score, item.max_score = diagnosis.score, diagnosis.max_score
            except Exception as exc:
                item.errors.append(f"診断: {exc}")
    report.phase_seconds["diagnosis"] = round(time.monotonic() - started, 1)

    for item in report.items:
        if item.status == "pending":
            item.status = "error" if item.errors else ("done" if item.diagnosis_id else "no_result")
    if opts.stop_services_after:
        manager.stop_all()
    report.service_events = list(manager.events)
    return report
