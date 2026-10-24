import copy
import json
import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.adapters.diagnosis_adapter import LMStudioDiagnosisAdapter
from src.adapters.lmstudio_client import LMStudioClient
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
    client.prepare_chat_request.side_effect = LMStudioClient.prepare_chat_request
    client.backend_fingerprint.return_value = {
        "model_id": "loaded-weights-A", "settings": {"max_new_tokens_cap": 512, "model_revision": "revision-A"},
        "server_source_sha256": "a" * 64,
        "library_versions": dict.fromkeys(("torch", "transformers", "bitsandbytes", "tokenizers", "accelerate"), "test-version"),
    }
    client.chat_completion.return_value = json.dumps({
        "score": None, "max_score": 5, "question_scores": [],
    })

    def run(config=config, options=options, answers=None, **kwargs):
        adapter = LMStudioDiagnosisAdapter(config, lmstudio_client=client, transport="lmstudio")
        return run_batch(storage, config, answers or [answer], options, services=kwargs.pop("services", manager),
                         diagnosis_adapter=adapter, **kwargs).items[0]

    return storage, config, options, client, run


def test_exact_grading_configuration_is_stored_and_reused(case):
    storage, config, options, client, run = case
    first = run()
    # Reload from disk, rather than comparing an in-memory result only.
    saved = Storage(config).get_diagnosis_result(first.diagnosis_id)
    metadata = saved.grading_metadata
    assert metadata["model_id"] == "loaded-weights-A"
    assert metadata["model_alias"] == config.diagnosis.model_name
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
    "all", "null", "empty", "provider", "model_id", "prompt_version", "rubric_identity",
    "generation_settings", "request_settings", "backend", "model_alias",
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
    elif missing == "backend":
        payload["grading_metadata"]["generation_settings"].pop("backend")
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
    monkeypatch.setattr(LMStudioDiagnosisAdapter, "grading_metadata", lambda self, context, **kwargs: None)
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
            "model_id": "started-weights", "settings": {"max_new_tokens_cap": 512, "model_revision": "revision-A"},
            "server_source_sha256": "a" * 64,
            "library_versions": dict.fromkeys(("torch", "transformers", "bitsandbytes", "tokenizers", "accelerate"), "test-version"),
        }

    monkeypatch.setattr("src.runtime.services.get_service_manager",
                        lambda config: SimpleNamespace(ensure=start))
    first = run(config=config, services=SimpleNamespace(events=[], ensure=start))
    saved = storage.get_diagnosis_result(first.diagnosis_id)
    assert saved.grading_metadata["generation_settings"]["backend"]["model_id"] == "started-weights"
    assert run(config=config).diagnosis_id == first.diagnosis_id
    assert client.chat_completion.call_count == 1


@pytest.mark.parametrize("partial", [
    "empty", "provider_only", "model_id", "provider", "prompt_version", "rubric_identity",
    "generation_settings", "empty_settings", "request", "empty_request", "temperature",
    "max_tokens", "disable_thinking", "response_format_json", "backend", "backend_model",
    "backend_settings", "empty_backend_settings", "invalid_hash", "invalid_request_type",
    "invalid_temperature", "invalid_max_tokens", "invalid_bool", "empty_model",
    "mock_without_settings",
])
def test_equal_incomplete_metadata_on_both_sides_never_reuses(case, monkeypatch, partial):
    storage, config, options, client, run = case
    first = run()
    stored = storage.get_diagnosis_result(first.diagnosis_id)
    metadata = stored.grading_metadata
    settings = metadata["generation_settings"]
    if partial == "empty":
        metadata.clear()
    elif partial == "provider_only":
        metadata = {"provider": "mock"}
    elif partial in {"provider", "model_id", "prompt_version", "rubric_identity", "generation_settings"}:
        metadata.pop(partial)
    elif partial == "empty_settings":
        settings.clear()
    elif partial in {"request", "backend"}:
        settings.pop(partial)
    elif partial == "empty_request":
        settings["request"].clear()
    elif partial in {"temperature", "max_tokens", "disable_thinking", "response_format_json"}:
        settings["request"].pop(partial)
    elif partial == "backend_model":
        settings["backend"].pop("model_id")
    elif partial == "backend_settings":
        settings["backend"].pop("settings")
    elif partial == "empty_backend_settings":
        settings["backend"]["settings"].clear()
    elif partial == "invalid_hash":
        metadata["prompt_version"] = "not-a-sha256"
    elif partial == "invalid_request_type":
        settings["request"] = []
    elif partial == "invalid_temperature":
        settings["request"]["temperature"] = float("nan")
    elif partial == "invalid_max_tokens":
        settings["request"]["max_tokens"] = True
    elif partial == "invalid_bool":
        settings["request"]["disable_thinking"] = 1
    elif partial == "empty_model":
        metadata["model_id"] = " "
    else:
        metadata["provider"] = "mock"
        settings.clear()
    stored.grading_metadata = metadata
    # Reproduce equal partial metadata in memory, independently of JSON loading.
    monkeypatch.setattr(storage, "get_diagnosis_results_for_answer", lambda answer_id: [stored])
    monkeypatch.setattr(LMStudioDiagnosisAdapter, "grading_metadata", lambda self, context, **kwargs: metadata)
    again = run()
    assert again.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2
    assert storage.get_diagnosis_result(again.diagnosis_id).grading_metadata is None


@pytest.mark.parametrize("fingerprint", [{}, {"model_id": "weights"},
    {"model_id": "weights", "settings": {}}, {"settings": {"cap": 512}}])
def test_adapter_rejects_incomplete_fingerprint_from_injected_client(case, fingerprint):
    storage, config, options, client, run = case
    client.backend_fingerprint.return_value = fingerprint
    first = run()
    again = run()
    assert first.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2
    assert all(d.grading_metadata is None for d in storage.list_diagnosis_results())


def test_prompt_hash_tracks_transmitted_prefill_and_forces_regrading(case, monkeypatch):
    storage, config, options, client, run = case
    # Exercise the actual client request builder and send path with an in-memory SDK.
    client.chat_completion.side_effect = LMStudioClient.chat_completion.__get__(client, LMStudioClient)
    create = client._client.chat.completions.create
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=client.chat_completion.return_value),
    )])
    first = run()
    first_messages = create.call_args.kwargs["messages"]
    from src.adapters.lmstudio_client import _THINKING_OFF_PREFILL

    assert first_messages[-1] == {"role": "assistant", "content": _THINKING_OFF_PREFILL}
    monkeypatch.setattr("src.adapters.lmstudio_client._THINKING_OFF_PREFILL", "changed assistant prefill")
    again = run()
    second_messages = create.call_args.kwargs["messages"]
    assert second_messages[-1] == {"role": "assistant", "content": "changed assistant prefill"}
    assert first_messages != second_messages
    assert again.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert create.call_count == 2
    for diagnosis_id, messages in [(first.diagnosis_id, first_messages), (again.diagnosis_id, second_messages)]:
        expected = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert storage.get_diagnosis_result(diagnosis_id).grading_metadata["prompt_version"] == expected
    assert run().diagnosis_id == again.diagnosis_id
    assert create.call_count == 2


@pytest.mark.parametrize("builder", [None, lambda **kwargs: {}, lambda **kwargs: {"messages": "unknown"}])
def test_unknown_final_request_disables_reuse(case, builder):
    storage, config, options, client, run = case
    client.prepare_chat_request = builder
    first = run()
    again = run()
    assert first.diagnosis_id and again.diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2
    assert all(d.grading_metadata is None for d in storage.list_diagnosis_results())


def test_review_routing_change_requires_regrading(case, monkeypatch):
    storage, config, options, client, run = case
    client.chat_completion.return_value = json.dumps({
        "score": 1, "max_score": 5,
        "question_scores": [{"question": "(1)", "score": 1, "max_score": 5,
                             "reason": "基準を満たすので満点とする"}],
    })
    monkeypatch.delenv("REVIEW_FLAGS_ROUTE", raising=False)
    first = run()
    assert first.status == "validated"
    monkeypatch.setenv("REVIEW_FLAGS_ROUTE", "true")
    again = run()
    assert again.diagnosis_id != first.diagnosis_id
    assert again.status == "review_required"
    assert client.chat_completion.call_count == 2
    assert run().diagnosis_id == again.diagnosis_id


@pytest.mark.parametrize("post_processing", [None, {}, {"review_flags_route": 1}])
def test_equal_missing_or_invalid_post_processing_never_reuses(case, monkeypatch, post_processing):
    storage, config, options, client, run = case
    first = run()
    saved = storage.get_diagnosis_result(first.diagnosis_id)
    saved.grading_metadata["post_processing"] = post_processing
    monkeypatch.setattr(storage, "get_diagnosis_results_for_answer", lambda answer_id: [saved])
    monkeypatch.setattr(LMStudioDiagnosisAdapter, "grading_metadata",
                        lambda self, context, **kwargs: saved.grading_metadata)
    assert run().diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2


@pytest.mark.parametrize("change", ["revision", "weights_digest", "source", "library"])
def test_backend_provenance_change_requires_regrading(case, change):
    storage, config, options, client, run = case
    first = run()
    fingerprint = client.backend_fingerprint.return_value
    if change == "revision":
        fingerprint["settings"]["model_revision"] = "revision-B"
    elif change == "weights_digest":
        fingerprint["settings"]["weights_sha256"] = "b" * 64
    elif change == "source":
        fingerprint["server_source_sha256"] = "b" * 64
    else:
        fingerprint["library_versions"]["torch"] = "changed-version"
    assert run().diagnosis_id != first.diagnosis_id
    assert client.chat_completion.call_count == 2


def test_metadata_failure_is_isolated_to_one_answer(case, monkeypatch):
    storage, config, options, client, run = case
    first = storage.list_answers()[0]
    second = first.model_copy(update={"answer_id": "b"})
    storage.save_answer(second)
    typed_text_as_ocr(storage, answer_id="b", typed_text="another answer")
    original = LMStudioDiagnosisAdapter.grading_metadata

    def metadata(self, context, **kwargs):
        if context.answer_id == "a":
            raise RuntimeError("metadata unavailable")
        return original(self, context, **kwargs)

    monkeypatch.setattr(LMStudioDiagnosisAdapter, "grading_metadata", metadata)
    item = run(answers=[first, second])
    assert item.status == "failed"
    assert "metadata unavailable" in " ".join(item.errors)
    assert [d.answer_id for d in storage.list_diagnosis_results()] == ["b"]


def test_startup_failure_is_recorded_for_each_item(case, monkeypatch):
    storage, config, options, client, run = case
    config = replace(config, services=replace(config.services, mode="managed"))
    first = storage.list_answers()[0]
    second = first.model_copy(update={"answer_id": "b"})
    storage.save_answer(second)
    typed_text_as_ocr(storage, answer_id="b", typed_text="another answer")

    def fail(service):
        raise RuntimeError("startup failed")

    manager = SimpleNamespace(events=[], ensure=fail)
    adapter = LMStudioDiagnosisAdapter(config, lmstudio_client=client)
    report = run_batch(storage, config, [first, second], options, services=manager,
                       diagnosis_adapter=adapter)
    assert all(item.status == "failed" and "startup failed" in " ".join(item.errors)
               for item in report.items)
    assert not storage.list_diagnosis_results()


def test_batch_shares_startup_and_backend_snapshot(case, monkeypatch):
    storage, config, options, client, run = case
    config = replace(config, services=replace(config.services, mode="managed"))
    first = storage.list_answers()[0]
    second = first.model_copy(update={"answer_id": "b"})
    storage.save_answer(second)
    typed_text_as_ocr(storage, answer_id="b", typed_text="another answer")
    manager = SimpleNamespace(events=[], ensure=MagicMock())
    monkeypatch.setattr("src.runtime.services.get_service_manager", lambda config: manager)
    adapter = LMStudioDiagnosisAdapter(config, lmstudio_client=client)
    for skip in (False, True):
        client.backend_fingerprint.reset_mock()
        manager.ensure.reset_mock()
        report = run_batch(storage, config, [first, second], replace(options, skip_existing_diagnosis=skip),
                           services=manager, diagnosis_adapter=adapter)
        assert all(item.diagnosis_id for item in report.items)
        assert manager.ensure.call_count == 1
        # Startup stays once per batch. Identity is read before every answer and
        # again after each generation, including when every answer is reused.
        assert client.backend_fingerprint.call_count == (4 if not skip else 2)
    assert client.chat_completion.call_count == 2


@pytest.mark.parametrize("kind", ["model_id", "settings", "unavailable"])
def test_backend_change_between_cached_answers_is_not_reused(case, kind):
    """A fingerprint cached for one reused answer must not authorize the next."""
    storage, config, options, client, run = case
    config = replace(config, services=replace(config.services, mode="managed"))
    first = storage.list_answers()[0]
    second = first.model_copy(update={"answer_id": "b"})
    storage.save_answer(second)
    typed_text_as_ocr(storage, answer_id="b", typed_text="another answer")
    manager = SimpleNamespace(events=[], ensure=MagicMock())
    adapter = LMStudioDiagnosisAdapter(config, lmstudio_client=client)
    answers = [first, second]
    primed = run_batch(storage, config, answers, options, services=manager, diagnosis_adapter=adapter)
    saved_ids = {item.answer_id: item.diagnosis_id for item in primed.items}
    assert client.chat_completion.call_count == 2
    client.chat_completion.reset_mock()
    client.backend_fingerprint.reset_mock()
    manager.ensure.reset_mock()
    base = copy.deepcopy(client.backend_fingerprint.return_value)
    state = {"n": 0}

    def current_backend():
        state["n"] += 1
        if state["n"] == 1:
            client.fingerprint_unavailable_reason = None
            return copy.deepcopy(base)
        if kind == "unavailable":
            client.fingerprint_unavailable_reason = "backend_health_timeout"
            return None
        client.fingerprint_unavailable_reason = None
        changed = copy.deepcopy(base)
        if kind == "model_id":
            changed["model_id"] = "loaded-weights-B"
        else:
            changed["settings"]["model_revision"] = "revision-B"
        return changed

    client.backend_fingerprint.side_effect = current_backend
    report = run_batch(storage, config, answers, options, services=manager, diagnosis_adapter=adapter)
    assert manager.ensure.call_count == 1
    assert report.items[0].diagnosis_id == saved_ids["a"]
    assert report.items[1].diagnosis_id != saved_ids["b"]
    assert client.chat_completion.call_count == 1
    regenerated = storage.get_diagnosis_result(report.items[1].diagnosis_id)
    if kind == "unavailable":
        assert regenerated.grading_metadata is None
        assert regenerated.reuse_unavailable_reason == "backend_health_timeout"
        assert report.items[1].status == "review_required"
    else:
        assert regenerated.grading_metadata["generation_settings"]["backend"]["model_id"] == (
            "loaded-weights-B" if kind == "model_id" else "loaded-weights-A"
        )
        if kind == "settings":
            assert regenerated.grading_metadata["generation_settings"]["backend"]["settings"]["model_revision"] == "revision-B"
    assert report.items[0].status != "failed"


def test_unavailable_backend_reason_is_persisted_without_failing_grade(case):
    storage, config, options, client, run = case
    client.backend_fingerprint.return_value = None
    client.fingerprint_unavailable_reason = "backend_health_timeout"
    first = run()
    saved = Storage(config).get_diagnosis_result(first.diagnosis_id)
    assert saved.reuse_unavailable_reason == "backend_health_timeout"
    assert first.reuse_unavailable_reason == "backend_health_timeout"
    assert first.status == "review_required"


def test_changed_fingerprint_reason_is_persisted(case):
    storage, config, options, client, run = case

    def switch(**kwargs):
        client.backend_fingerprint.return_value["settings"]["model_revision"] = "revision-B"
        return client.chat_completion.return_value

    client.chat_completion.side_effect = switch
    first = run()
    saved = Storage(config).get_diagnosis_result(first.diagnosis_id)
    assert saved.grading_metadata is None
    assert saved.reuse_unavailable_reason == "grading_configuration_changed_during_generation"


def test_metadata_records_loaded_model_and_keeps_request_alias(case):
    storage, config, options, client, run = case
    first = run()
    saved = Storage(config).get_diagnosis_result(first.diagnosis_id)
    assert saved.grading_metadata["model_id"] == "loaded-weights-A"
    assert saved.grading_metadata["model_alias"] == config.diagnosis.model_name
    assert saved.model_name == config.diagnosis.model_name


def test_model_identity_mismatch_never_reuses_even_if_metadata_equal(case, monkeypatch):
    storage, config, options, client, run = case
    first = run()
    saved = storage.get_diagnosis_result(first.diagnosis_id)
    saved.grading_metadata["model_id"] = "different-loaded-model"
    monkeypatch.setattr(storage, "get_diagnosis_results_for_answer", lambda answer_id: [saved])
    monkeypatch.setattr(LMStudioDiagnosisAdapter, "grading_metadata",
                        lambda self, context, **kwargs: saved.grading_metadata)
    again = run()
    assert again.diagnosis_id != first.diagnosis_id
    assert storage.get_diagnosis_result(again.diagnosis_id).grading_metadata is None
