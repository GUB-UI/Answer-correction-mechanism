from dataclasses import replace
import hashlib

import pytest

from src.adapters.mock_adapters import MockOCRAdapter
from src.adapters.ocr_adapter import ManualCorrectionOCRAdapter
from src.batch_pipeline import BatchOptions, run_batch
from src.config import load_config
from src.models import AnswerRecord, OCRResult, ProblemRecord, utc_now_iso
from src.ocr_runner import typed_text_as_ocr
from src.storage import Storage


class CountingOCR(MockOCRAdapter):
    def __init__(self):
        self.calls = 0

    def run(self, context):
        self.calls += 1
        return super().run(context)


@pytest.fixture
def case(tmp_path):
    config = load_config(tmp_path)
    storage = Storage(config)
    storage.save_problem(ProblemRecord(problem_id="p", title="test", created_at=utc_now_iso()))
    paths = [storage.save_image(content) for content in (b"page one", b"page two")]
    answer = AnswerRecord(answer_id="a", problem_id="p", image_paths=paths,
                          student_anonymized_id="student", created_at=utc_now_iso())
    storage.save_answer(answer)
    options = BatchOptions(ocr_provider="glmocr_sdk", diagnosis_provider="mock", apply_formula=False)
    adapter = CountingOCR()
    def run(config=config, answer=answer, options=options, **kwargs):
        return run_batch(storage, config, [answer], options, ocr_adapter=adapter, **kwargs)
    return storage, config, answer, options, adapter, run


def test_matching_provenance_reuses_and_persists_ordered_sha256(case):
    storage, config, answer, options, adapter, run = case
    first = run().items[0]
    saved = storage.get_ocr_result(first.ocr_id)
    assert saved.provenance.engine == "glmocr_sdk"
    assert saved.provenance.model_id == config.ocr.model_name
    assert saved.provenance.model_revision == config.ocr.revision
    assert saved.provenance.image_sha256 == [hashlib.sha256(p).hexdigest() for p in (b"page one", b"page two")]
    assert run().items[0].ocr_id == first.ocr_id
    assert adapter.calls == 2


@pytest.mark.parametrize("change", ["engine", "model_id", "revision", "settings", "image_bytes", "image_order", "image_count", "legacy"])
def test_each_provenance_mismatch_reruns_ocr(case, change):
    storage, config, answer, options, adapter, run = case
    first = run().items[0]
    if change == "engine":
        options = replace(options, ocr_provider="xiaomi_ocr")
    elif change == "model_id":
        config = replace(config, ocr=replace(config.ocr, model_name="other-model"))
    elif change == "revision":
        config = replace(config, ocr=replace(config.ocr, revision="other-revision"))
    elif change == "settings":
        config = replace(config, ocr_api=replace(config.ocr_api, min_edge=1200))
    elif change == "image_bytes":
        config.resolve_path(answer.image_paths[0]).write_bytes(b"changed in place")
    elif change == "image_order":
        answer = answer.model_copy(update={"image_paths": answer.image_paths[::-1]})
    elif change == "image_count":
        answer = answer.model_copy(update={"image_paths": answer.image_paths[:1]})
    elif change == "legacy":
        saved = storage.get_ocr_result(first.ocr_id)
        storage.update_ocr_result(saved.model_copy(update={"provenance": None}))
    second = run(config=config, answer=answer, options=options).items[0]
    assert second.ocr_id and second.ocr_id != first.ocr_id
    assert adapter.calls > 2


def test_correction_is_separate_and_requires_explicit_input_selection(case):
    storage, config, answer, options, adapter, run = case
    first = run().items[0]
    original = storage.get_ocr_result(first.ocr_id)
    corrected = ManualCorrectionOCRAdapter(storage).apply_correction(first.ocr_id, "corrected transcription")
    assert corrected.input_type == "manual_transcription"
    assert corrected.parent_ocr_id == first.ocr_id
    assert corrected.ocr_id != first.ocr_id
    assert corrected.provenance is None
    assert storage.get_ocr_result(first.ocr_id) == original
    assert run().items[0].ocr_id == first.ocr_id
    selected = run(options=replace(options, input_type="manual_transcription")).items[0]
    assert selected.ocr_id == corrected.ocr_id
    assert adapter.calls == 2


def test_legacy_human_corrected_record_is_never_reused_as_ocr(case):
    storage, config, answer, options, adapter, run = case
    first = run().items[0]
    saved = storage.get_ocr_result(first.ocr_id).model_dump()
    saved.pop("input_type")
    saved.update(human_corrected=True, used_text="manual text")
    legacy = OCRResult.model_validate(saved)
    assert legacy.input_type == "manual_transcription"
    storage.update_ocr_result(legacy)
    assert run().items[0].ocr_id != first.ocr_id
    assert adapter.calls == 4


def test_typed_text_is_an_explicit_input_and_not_an_ocr_cache_hit(case):
    storage, config, answer, options, adapter, run = case
    typed = typed_text_as_ocr(storage, answer_id=answer.answer_id, typed_text="typed answer")
    assert typed.input_type == "typed_text"
    assert run(options=replace(options, input_type="typed_text")).items[0].ocr_id == typed.ocr_id
    assert adapter.calls == 0
    assert run().items[0].ocr_id != typed.ocr_id
    assert adapter.calls == 2


def test_missing_image_prevents_unverifiable_reuse(case):
    storage, config, answer, options, adapter, run = case
    run()
    config.resolve_path(answer.image_paths[0]).unlink()
    item = run().items[0]
    assert not item.ocr_id
    assert item.errors
    assert adapter.calls == 2


def test_formula_setting_and_failed_refinement_do_not_reuse_raw_ocr(case):
    storage, config, answer, options, adapter, run = case
    first = run().items[0]
    options = replace(options, apply_formula=True)
    config = replace(config, formula=replace(config.formula, enabled=True))
    def fail(result):
        raise RuntimeError("refinement unavailable")
    failed = run(config=config, options=options, formula_refiner=fail).items[0]
    assert failed.ocr_id != first.ocr_id
    assert failed.errors
    completed = run(config=config, options=options, formula_refiner=lambda r: r.model_copy(update={
        "raw_output": {**r.raw_output, "formula_summary": {"completed": True}},
    })).items[0]
    assert completed.ocr_id != failed.ocr_id
    assert run(config=config, options=options).items[0].ocr_id == completed.ocr_id
    assert adapter.calls == 6


def test_model_revision_is_loaded_from_config_and_forwarded_to_managed_servers(tmp_path, monkeypatch):
    from src.runtime.services import build_service_specs
    monkeypatch.setenv("OCR_MODEL_REVISION", "pinned-revision")
    config = load_config(tmp_path)
    assert config.ocr.revision == "pinned-revision"
    for name in ("glm_ocr", "xiaomi_ocr"):
        command = build_service_specs(config)[name].command
        assert command[command.index("--revision") + 1] == "pinned-revision"


@pytest.mark.parametrize("outcome", ["unavailable", "prediction_error", "exception", "unchanged", "success", "no_formula"])
def test_real_formula_outcome_controls_reuse(case, monkeypatch, outcome):
    from io import BytesIO
    from PIL import Image
    from src.adapters import ocr_adapter as module

    storage, config, answer, options, adapter, run = case
    config = replace(config, formula=replace(config.formula, enabled=True))
    options = replace(options, apply_formula=True)
    image = BytesIO()
    Image.new("RGB", (40, 40), "white").save(image, format="PNG")
    for path in answer.image_paths:
        config.resolve_path(path).write_bytes(image.getvalue())
    original_run = adapter.run

    def layout(context):
        result = original_run(context)
        text = "a+b" if outcome != "no_formula" else "説明文"
        return result.model_copy(update={"raw_text": text, "used_text": text, "raw_output": {
            "image_path": str(context.image_path),
            "result": {"markdown_result": text, "json_result": [[{
                "label": "formula" if outcome != "no_formula" else "text",
                "content": text, "bbox_2d": [0, 0, 1000, 100],
            }]]},
        }})

    class Formula:
        def __init__(self, *args):
            pass

        def health(self):
            if outcome == "exception":
                raise RuntimeError("health error")
            return outcome != "unavailable"

        def predict_image(self, image):
            if outcome == "prediction_error":
                raise RuntimeError("prediction error")
            return "a+b"

    monkeypatch.setattr(adapter, "run", layout)
    monkeypatch.setattr(module, "UniMERNetClient", Formula)
    monkeypatch.setattr(module, "_ensure_service", lambda *args: None)
    kwargs = {"formula_refiner": lambda r: r} if outcome == "unchanged" else {}
    first = run(config=config, options=options, **kwargs).items[0]
    saved = storage.get_ocr_result(first.ocr_id)
    successful = outcome in {"success", "no_formula"}
    assert bool(saved.provenance and saved.provenance.settings["formula_enabled"]) is successful
    if outcome == "unavailable":
        assert "unimernet_unavailable" in saved.raw_output["formula_summary"]["skipped"]
    if outcome == "prediction_error":
        assert saved.raw_output["formula_replacements"][0]["error"] == "prediction error"
    if outcome == "success":
        assert saved.raw_output["formula_summary"]["replaced"] == 2
    second = run(config=config, options=options, **kwargs).items[0]
    assert (second.ocr_id == first.ocr_id) is successful
    assert adapter.calls == (2 if successful else 4)


def test_legacy_formula_provenance_without_success_is_not_reused(case):
    from src.ocr_provenance import ocr_provenance

    storage, config, answer, options, adapter, run = case
    first = run().items[0]
    config = replace(config, formula=replace(config.formula, enabled=True))
    options = replace(options, apply_formula=True)
    saved = storage.get_ocr_result(first.ocr_id)
    legacy = saved.model_copy(update={"provenance": ocr_provenance(
        config, answer.image_paths, provider=options.ocr_provider, adapter=adapter,
        apply_formula=True,
    )})
    storage.update_ocr_result(legacy)
    retried = run(config=config, options=options, formula_refiner=lambda r: r).items[0]
    assert retried.ocr_id != first.ocr_id
    assert adapter.calls == 4
