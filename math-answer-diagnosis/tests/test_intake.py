from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.config import load_config
from src.models import ProblemRecord, utc_now_iso
from src.storage import Storage


QUESTION = "不等式 2x < 8 を解け。"
STUDENT = "x < 3"
MODEL = "x < 4"


def intake_module():
    # A missing implementation is an ordinary failing test, not a collection error.
    from src import intake

    return intake


def response(*, existing_id=None, confidence=0.98, reasons=None):
    return json.dumps({"groups": [{
        "title": "一次不等式",
        "problem": {"text": QUESTION, "source_ids": ["doc_1"]},
        "student_answer": {"text": STUDENT, "source_ids": ["doc_2"]},
        "model_answer": {"text": MODEL, "source_ids": ["doc_3"]},
        "existing_problem_id": existing_id,
        "confidence": confidence,
        "review_reasons": reasons or [],
    }]}, ensure_ascii=False)


def documents(mode="separate"):
    api = intake_module()
    if mode == "problem_student":
        return [
            api.IntakeDocument(source_id="doc_1", text=QUESTION + "\n生徒答案: " + STUDENT,
                               hint="problem_student", image_paths=["data/images/shared.png"]),
            api.IntakeDocument(source_id="doc_3", text=MODEL, hint="model_answer"),
        ]
    if mode == "problem_model":
        return [
            api.IntakeDocument(source_id="doc_1", text=QUESTION + "\n模範解答: " + MODEL,
                               hint="problem_model"),
            api.IntakeDocument(source_id="doc_2", text=STUDENT, hint="student_answer"),
        ]
    return [api.IntakeDocument(source_id=f"doc_{i}", text=text, hint="auto")
            for i, text in enumerate([QUESTION, STUDENT, MODEL], 1)]


def parse(raw=None, docs=None, existing=None):
    return intake_module().parse_tagging_response(
        raw or response(), documents=docs or documents(),
        existing_problems=existing or [], model_name="local-qwen",
    )


@pytest.fixture
def storage(tmp_path):
    return Storage(load_config(tmp_path))


@pytest.mark.parametrize("mode", ["separate", "problem_student", "problem_model"])
def test_all_input_layouts_save_one_shared_problem_tag(storage, mode):
    payload = json.loads(response())
    if mode == "problem_student":
        payload["groups"][0]["student_answer"]["source_ids"] = ["doc_1"]
    if mode == "problem_model":
        payload["groups"][0]["model_answer"]["source_ids"] = ["doc_1"]
    batch = parse(json.dumps(payload), documents(mode))
    intake_module().save_intake_batch(storage, batch, student_id="student_001")
    problem = storage.list_problems()[0]
    answer = storage.list_answers()[0]
    assert answer.problem_id == problem.problem_id == batch.groups[0].problem_id
    assert problem.problem_text == QUESTION
    assert problem.correct_answer == MODEL
    assert answer.typed_text == STUDENT
    ocr = storage.get_ocr_results_for_answer(answer.answer_id)[0]
    assert ocr.used_text == STUDENT  # The shared image's question must not reach grading.
    assert ocr.raw_output["intake_batch_id"] == batch.batch_id
    assert problem.tagging_metadata["model_name"] == "local-qwen"
    assert problem.tagging_metadata["documents"][0]["text"].startswith(QUESTION)


def test_existing_problem_tag_is_reused_without_replacing_reference(storage):
    original = ProblemRecord(problem_id="prob_existing", title="既存問題",
                             problem_text=QUESTION, correct_answer=MODEL,
                             created_at=utc_now_iso())
    storage.save_problem(original)
    batch = parse(response(existing_id=original.problem_id), existing=[original])
    intake_module().save_intake_batch(storage, batch, student_id="student_002")
    assert len(storage.list_problems()) == 1
    assert storage.list_answers()[0].problem_id == original.problem_id
    assert storage.get_problem(original.problem_id).correct_answer == MODEL


def test_reference_only_input_can_be_saved_and_answer_added_later(storage):
    payload = json.loads(response())
    payload["groups"][0]["student_answer"] = {"text": "", "source_ids": []}
    batch = parse(json.dumps(payload), [documents()[0], documents()[2]])
    intake_module().save_intake_batch(storage, batch)
    assert len(storage.list_problems()) == 1
    assert storage.list_answers() == []
    assert storage.list_problems()[0].correct_answer == MODEL


def test_missing_reference_is_saved_as_pending_not_generated(storage):
    payload = json.loads(response())
    payload["groups"][0]["model_answer"] = {"text": "", "source_ids": []}
    batch = parse(json.dumps(payload), documents()[:2])
    intake_module().save_intake_batch(storage, batch, student_id="student_001")
    assert storage.list_problems()[0].correct_answer == ""
    assert len(storage.list_answers()) == 1


@pytest.mark.parametrize("change", ["unknown_problem", "unknown_source", "invented_answer", "out_of_range"])
def test_invalid_llm_output_is_rejected(change):
    payload = json.loads(response())
    group = payload["groups"][0]
    if change == "unknown_problem":
        group["existing_problem_id"] = "prob_invented"
    elif change == "unknown_source":
        group["problem"]["source_ids"] = ["doc_invented"]
    elif change == "invented_answer":
        group["student_answer"]["text"] = "x < 4 正解に訂正しました"
    else:
        group["confidence"] = 1.1
    with pytest.raises(ValueError):
        parse(json.dumps(payload))


def test_ambiguous_group_requires_review_before_any_records_are_written(storage):
    batch = parse(response(confidence=0.5, reasons=["設問番号が不明"] ))
    with pytest.raises(ValueError, match="確認"):
        intake_module().save_intake_batch(storage, batch, student_id="student_001")
    assert storage.list_problems() == storage.list_answers() == []
    intake_module().save_intake_batch(storage, batch, student_id="student_001", reviewed=True)
    assert len(storage.list_answers()) == 1


def test_repeated_save_is_idempotent(storage):
    batch = parse()
    for _ in range(2):
        intake_module().save_intake_batch(storage, batch, student_id="student_001")
    assert len(storage.list_problems()) == len(storage.list_answers()) == 1
    assert len(storage.get_ocr_results_for_answer(storage.list_answers()[0].answer_id)) == 1


def test_invalid_student_id_does_not_write_partial_problem(storage):
    with pytest.raises(ValueError):
        intake_module().save_intake_batch(storage, parse(), student_id="")
    assert storage.list_problems() == []


def test_conflicting_existing_reference_needs_review_and_keeps_original(storage):
    original = ProblemRecord(problem_id="prob_existing", title="既存",
                             problem_text=QUESTION, correct_answer="x < 5",
                             created_at=utc_now_iso())
    storage.save_problem(original)
    batch = parse(response(existing_id=original.problem_id), existing=[original])
    with pytest.raises(ValueError, match="確認"):
        intake_module().save_intake_batch(storage, batch, student_id="s1")
    intake_module().save_intake_batch(storage, batch, student_id="s1", reviewed=True)
    assert storage.get_problem(original.problem_id).correct_answer == "x < 5"


def test_tagging_uses_configured_local_llm_and_structured_documents(storage):
    api = intake_module()

    class LocalLLM:
        def chat_completion(self, **kwargs):
            prompt = json.loads(kwargs["messages"][1]["content"])
            # Drive the result from actual input to test transport-to-parser integration.
            assert prompt["documents"][0]["source_id"] == "doc_1"
            assert kwargs["model"] == "qwen-tagging"
            return response()

    config = replace(storage.config, tagging_model_name="qwen-tagging")
    batch = api.analyze_intake(config, documents(), [], client=LocalLLM())
    assert batch.groups[0].extraction.student_answer.text == STUDENT
    assert batch.model_name == "qwen-tagging"


def test_public_endpoint_is_not_used_for_local_tagging(storage):
    api = intake_module()
    config = replace(storage.config,
                     lmstudio=replace(storage.config.lmstudio, base_url="https://api.openai.com/v1"))
    with pytest.raises(ValueError, match="ローカル"):
        api.analyze_intake(config, documents(), [])


def test_multiple_problems_in_a_document_get_distinct_tags(storage):
    api = intake_module()
    payload = json.loads(response())
    payload["groups"][0]["student_answer"] = {"text": "", "source_ids": []}
    second = json.loads(response())["groups"][0]
    second["title"] = "二問目"
    second["problem"] = {"text": "3x = 9 を解け。", "source_ids": ["doc_4"]}
    second["model_answer"] = {"text": "x = 3", "source_ids": ["doc_4"]}
    second["student_answer"] = {"text": "", "source_ids": []}
    payload["groups"].append(second)
    docs = [documents()[0], documents()[2],
            api.IntakeDocument(source_id="doc_4", text="3x = 9 を解け。\n模範解答 x = 3")]
    batch = parse(json.dumps(payload), docs)
    api.save_intake_batch(storage, batch)
    assert len({p.problem_id for p in storage.list_problems()}) == 2


def test_exact_existing_question_reuses_tag_even_if_model_omits_match(storage):
    original = ProblemRecord(problem_id="prob_existing", title="既存",
                             problem_text=QUESTION, correct_answer=MODEL, created_at=utc_now_iso())
    storage.save_problem(original)
    batch = parse(existing=[original])
    intake_module().save_intake_batch(storage, batch, student_id="s1")
    assert storage.list_answers()[0].problem_id == original.problem_id
    assert len(storage.list_problems()) == 1


def test_explicit_student_document_cannot_be_treated_as_model_answer():
    docs = documents()
    docs[2].hint = "student_answer"
    with pytest.raises(ValueError, match="役割"):
        parse(docs=docs)


def test_reference_changed_after_preview_requires_review(storage):
    original = ProblemRecord(problem_id="prob_existing", title="既存",
                             problem_text=QUESTION, correct_answer=MODEL, created_at=utc_now_iso())
    storage.save_problem(original)
    batch = parse(response(existing_id=original.problem_id), existing=[original])
    storage.update_problem(original.model_copy(update={"correct_answer": "x < 5"}))
    with pytest.raises(ValueError, match="確認"):
        intake_module().save_intake_batch(storage, batch, student_id="s1")
    assert storage.list_answers() == []


def test_identical_question_groups_use_one_tag(storage):
    payload = json.loads(response())
    payload["groups"].append(json.loads(response())["groups"][0])
    batch = parse(json.dumps(payload))
    intake_module().save_intake_batch(storage, batch, student_id="s1", reviewed=True)
    assert len(storage.list_problems()) == 1
    assert len({answer.problem_id for answer in storage.list_answers()}) == 1


def test_changed_excerpt_is_marked_as_human_correction(storage):
    batch = parse()
    group = batch.groups[0]
    group.extraction.student_answer.text = "x < 2"
    group.human_corrected = True
    intake_module().save_intake_batch(storage, batch, student_id="s1", reviewed=True)
    result = storage.get_ocr_results_for_answer(group.answer_id)[0]
    assert result.human_corrected
    assert "x < 3" in result.raw_text
    assert result.used_text == "x < 2"


def test_answer_only_match_requires_review_even_when_llm_is_confident(storage):
    original = ProblemRecord(problem_id="prob_existing", title="割り算",
                             problem_text="100÷4は？", correct_answer="25", created_at=utc_now_iso())
    storage.save_problem(original)
    payload = json.loads(response(existing_id=original.problem_id))
    for role in ["problem", "model_answer"]:
        payload["groups"][0][role] = {"text": "", "source_ids": []}
    batch = parse(json.dumps(payload), [documents()[1]], [original])
    with pytest.raises(ValueError, match="確認"):
        intake_module().save_intake_batch(storage, batch, student_id="s1")
    assert storage.list_answers() == []


def test_swapped_role_correction_reconciles_image_provenance(storage):
    docs = documents()
    docs[1].image_paths = ["data/images/student.png"]
    docs[2].image_paths = ["data/images/model.png"]
    payload = json.loads(response())
    group = payload["groups"][0]
    group["student_answer"], group["model_answer"] = group["model_answer"], group["student_answer"]
    batch = parse(json.dumps(payload), docs)
    result = batch.groups[0]
    result.human_corrected = True
    result.extraction.student_answer.text = STUDENT
    result.extraction.model_answer.text = MODEL
    intake_module().save_intake_batch(storage, batch, student_id="s1", reviewed=True)
    assert storage.list_answers()[0].image_paths == ["data/images/student.png"]
    assert storage.list_problems()[0].model_answer_image_paths == ["data/images/model.png"]


def test_correcting_initially_identical_groups_can_split_them_into_distinct_problems(storage):
    api = intake_module()
    payload = json.loads(response())
    payload["groups"].append(json.loads(response())["groups"][0])
    docs = [*documents(), api.IntakeDocument(source_id="doc_4", text="3x = 9 を解け。")]
    batch = parse(json.dumps(payload), docs)
    second = batch.groups[1]
    second.extraction.problem.text = "3x = 9 を解け。"
    second.extraction.problem.source_ids = ["doc_4"]
    second.human_corrected = True
    api.save_intake_batch(storage, batch, student_id="s1", reviewed=True)
    assert len(storage.list_problems()) == 2
    assert len({answer.problem_id for answer in storage.list_answers()}) == 2


def test_analyze_intake_loads_managed_llm_before_chat(tmp_path, monkeypatch):
    """Colab keeps one GPU model. Tagging must start the LLM after OCR has run."""
    api = intake_module()
    config = replace(load_config(tmp_path).services, mode="managed")
    config = replace(load_config(tmp_path), services=config)
    calls = []

    class Manager:
        def ensure(self, name):
            calls.append(name)

    class Client:
        def chat_completion(self, **kwargs):
            assert calls == ["llm"]
            return response()

    monkeypatch.setattr("src.runtime.services.get_service_manager", lambda _config: Manager())
    monkeypatch.setattr(api, "LMStudioClient", lambda _config: Client())
    batch = api.analyze_intake(config, documents(), [])
    assert batch.groups and calls == ["llm"]


def test_injected_tagging_client_does_not_start_a_server(tmp_path, monkeypatch):
    api = intake_module()
    config = load_config(tmp_path)

    def fail_manager(_config):
        raise AssertionError("an injected client must not start services")

    monkeypatch.setattr("src.runtime.services.get_service_manager", fail_manager)

    class Client:
        def chat_completion(self, **kwargs):
            return response()

    batch = api.analyze_intake(config, documents(), [], client=Client())
    assert batch.groups
