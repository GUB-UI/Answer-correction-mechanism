import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.adapters.diagnosis_adapter import LMStudioDiagnosisAdapter
from src.batch_pipeline import BatchOptions, run_batch
from src.config import load_config
from src.models import AnswerRecord, ProblemRecord, utc_now_iso
from src.ocr_runner import typed_text_as_ocr
from src.prompts import DIAGNOSIS_PROMPTS
from src.storage import Storage


@pytest.fixture
def case(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        pytest.fail("Diagnosis reuse tests must not access the network")

    monkeypatch.setattr("socket.socket.connect", forbid_network)
    config = load_config(tmp_path)
    storage = Storage(config)
    storage.save_problem(ProblemRecord(
        problem_id="p", title="test", created_at=utc_now_iso(),
        problem_text="problem", correct_answer="answer", rubric="rubric version 1",
        score_allocation=[{"question": "(1)", "max_score": 5}],
    ))
    answer = AnswerRecord(answer_id="a", problem_id="p", student_anonymized_id="student",
                          created_at=utc_now_iso())
    storage.save_answer(answer)
    typed_text_as_ocr(storage, answer_id=answer.answer_id, typed_text="student work")
    options = BatchOptions(input_type="typed_text", diagnosis_provider="lmstudio")
    manager = SimpleNamespace(events=[], ensure=lambda service: None)
    monkeypatch.setattr("src.runtime.services.get_service_manager", lambda config: manager)
    client = MagicMock()
    client.backend_fingerprint.return_value = {
        "model_id": "loaded-weights-A", "settings": {"max_new_tokens_cap": 512},
    }
    client.chat_completion.return_value = json.dumps({
        "score": None, "max_score": 5, "question_scores": [],
    })

    def run(config=config, options=options, **kwargs):
        adapter = LMStudioDiagnosisAdapter(config, lmstudio_client=client, transport="lmstudio")
        return run_batch(storage, config, [answer], options, services=manager,
                         diagnosis_adapter=adapter, **kwargs).items[0]

    return storage, config, options, client, run


def test_exact_grading_configuration_is_stored_and_reused(case):
    storage, config, options, client, run = case
    first = run()
    # Reload from disk, rather than comparing an in-memory result only.
    saved = Storage(config).get_diagnosis_result(first.diagnosis_id)
    metadata = saved.grading_metadata
    assert metadata["model_id"] == config.diagnosis.model_name
    assert metadata["provider"] == "lmstudio"
    assert len(metadata["prompt_version"]) == len(metadata["rubric_identity"]) == 64
    settings = metadata["generation_settings"]["request"]
    assert settings == LMStudioDiagnosisAdapter.generation_settings
    assert all(client.chat_completion.call_args.kwargs[key] == value for key, value in settings.items())
    again = run()
    assert again.ocr_id == first.ocr_id
    assert again.diagnosis_id == first.diagnosis_id
    assert again.status == first.status == "review_required"
    assert again.score is first.score is None
    assert all(row["score"] is None for row in saved.question_scores)
    assert client.chat_completion.call_count == 1


@pytest.mark.parametrize("change", [
    "model", "prompt", "prompt_template", "prompt_type", "rubric", "allocation",
    "correct_answer", "problem_text", "temperature", "max_tokens", "disable_thinking",
    "response_format_json", "server_settings",
])
def test_grading_configuration_change_requires_regrading(case, monkeypatch, change):
    storage, config, options, client, run = case
    if change == "server_settings":
        config = replace(config, services=replace(config.services, mode="managed"))
    first = run(config=config)
    if change == "model":
        config = replace(config, diagnosis=replace(config.diagnosis, model_name="other-model"))
    elif change == "prompt":
        monkeypatch.setitem(DIAGNOSIS_PROMPTS, options.prompt_type,
                            DIAGNOSIS_PROMPTS[options.prompt_type] + "\nnew instructions")
    elif change == "prompt_template":
        original = LMStudioDiagnosisAdapter._build_user_prompt
        monkeypatch.setattr(LMStudioDiagnosisAdapter, "_build_user_prompt",
                            staticmethod(lambda context: original(context) + "\nnew template"))
    elif change == "prompt_type":
        options = replace(options, prompt_type="basic_scoring")
    elif change in {"rubric", "allocation", "correct_answer", "problem_text"}:
        problem = storage.list_problems()[0]
        field = "score_allocation" if change == "allocation" else change
        value = [{"question": "(1)", "max_score": 8}] if change == "allocation" else "changed"
        storage.save_problem(problem.model_copy(update={field: value}))
    elif change == "server_settings":
        config = replace(config, services=replace(
            config.services, llm=replace(config.services.llm, max_new_tokens=1024),
        ))
        client.backend_fingerprint.return_value["settings"]["max_new_tokens_cap"] = 1024
    else:
        value = {"temperature": 0.5, "max_tokens": 512, "disable_thinking": False,
                 "response_format_json": True}[change]
        monkeypatch.setattr(LMStudioDiagnosisAdapter, "generation_settings",
                            {**LMStudioDiagnosisAdapter.generation_settings, change: value})
    again = run(config=config, options=options)
    assert again.ocr_id == first.ocr_id
    assert again.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2
    assert storage.get_diagnosis_result(again.diagnosis_id).grading_metadata != (
        storage.get_diagnosis_result(first.diagnosis_id).grading_metadata
    )


@pytest.mark.parametrize("missing", [
    "all", "null", "empty", "model_id", "prompt_version", "rubric_identity",
    "generation_settings", "request_settings",
])
def test_legacy_or_incomplete_metadata_is_never_reused(case, missing):
    storage, config, options, client, run = case
    first = run()
    payload = storage.get_diagnosis_result(first.diagnosis_id).model_dump()
    if missing == "all":
        payload.pop("grading_metadata")
    elif missing == "null":
        payload["grading_metadata"] = None
    elif missing == "empty":
        payload["grading_metadata"] = {}
    elif missing == "request_settings":
        payload["grading_metadata"]["generation_settings"]["request"].pop("temperature")
    else:
        payload["grading_metadata"].pop(missing)
    (config.records_dir / "diagnosis_results.jsonl").write_text(json.dumps(payload) + "\n")
    again = run()
    assert again.ocr_id == first.ocr_id
    assert again.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2
    assert storage.get_diagnosis_result(again.diagnosis_id).grading_metadata


def test_skip_disabled_regrades_matching_configuration(case):
    storage, config, options, client, run = case
    first = run()
    again = run(options=replace(options, skip_existing_diagnosis=False))
    assert again.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2


def test_unknown_adapter_configuration_is_never_reused(case, monkeypatch):
    storage, config, options, client, run = case
    monkeypatch.setattr(LMStudioDiagnosisAdapter, "grading_metadata", lambda self, context: None)
    first = run()
    again = run()
    assert storage.get_diagnosis_result(first.diagnosis_id).grading_metadata is None
    assert again.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2


@pytest.mark.parametrize("mode", ["external", "managed"])
@pytest.mark.parametrize("change", ["weights", "cap", "unavailable"])
def test_running_backend_changes_under_same_alias_require_regrading(case, mode, change):
    storage, config, options, client, run = case
    config = replace(config, services=replace(config.services, mode=mode))
    first = run(config=config)
    if change == "weights":
        client.backend_fingerprint.return_value["model_id"] = "loaded-weights-B"
    elif change == "cap":
        client.backend_fingerprint.return_value["settings"]["max_new_tokens_cap"] = 1024
    else:
        client.backend_fingerprint.return_value = None
    again = run(config=config)
    assert again.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2


def test_unverified_backend_always_regrades(case):
    storage, config, options, client, run = case
    client.backend_fingerprint.return_value = None
    first = run()
    again = run()
    assert first.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2
    assert all(d.grading_metadata is None for d in storage.list_diagnosis_results())


def test_backend_change_during_generation_does_not_save_reusable_metadata(case):
    storage, config, options, client, run = case

    def switch(**kwargs):
        client.backend_fingerprint.return_value["model_id"] = "loaded-weights-B"
        return client.chat_completion.return_value

    client.chat_completion.side_effect = switch
    first = run()
    assert storage.get_diagnosis_result(first.diagnosis_id).grading_metadata is None
    again = run()
    assert again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2


def test_managed_result_records_running_weights_instead_of_desired_config(case):
    storage, config, options, client, run = case
    desired_b = replace(config, services=replace(
        config.services, mode="managed",
        llm=replace(config.services.llm, model_id="loaded-weights-B"),
    ))
    first = run(config=desired_b)
    metadata = storage.get_diagnosis_result(first.diagnosis_id).grading_metadata
    assert metadata["generation_settings"]["backend"]["model_id"] == "loaded-weights-A"
    client.backend_fingerprint.return_value["model_id"] = "loaded-weights-B"
    again = run(config=desired_b)
    assert again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2


def test_managed_startup_precedes_metadata_snapshot(case, monkeypatch):
    storage, config, options, client, run = case
    config = replace(config, services=replace(config.services, mode="managed"))
    client.backend_fingerprint.return_value = None

    def start(service):
        client.backend_fingerprint.return_value = {
            "model_id": "started-weights", "settings": {"max_new_tokens_cap": 512},
        }

    monkeypatch.setattr("src.runtime.services.get_service_manager",
                        lambda config: SimpleNamespace(ensure=start))
    first = run(config=config)
    saved = storage.get_diagnosis_result(first.diagnosis_id)
    assert saved.grading_metadata["generation_settings"]["backend"]["model_id"] == "started-weights"
    assert run(config=config).diagnosis_id == first.diagnosis_id
    assert client.chat_completion.call_count == 1
