"""Phase 4: prompt registry. Default prompt bytes are pinned; new versions are explicit."""
import hashlib
import json

import pytest

from harness_v2.core.contracts import RubricItem
from harness_v2.grading.scorer import PROMPT_VERSIONS, build_prompt

RUBRIC = [RubricItem('R1', 4, 'desc')]
# sha256 of the pre-registry default prompt for ('problem', 'x+1', RUBRIC).
PINNED = '7a70eccff6c8e67b6da4dfa95d282a3b1bf0766375d2bc22b0a0f4e770f844f4'


def _default():
    return build_prompt('problem', 'x+1', RUBRIC)


def test_default_and_v0_are_identical_to_historical_prompt():
    assert build_prompt('problem', 'x+1', RUBRIC, 'harness_v2_default') == _default()
    assert build_prompt('problem', 'x+1', RUBRIC, 'v0') == _default()
    assert 'transcription errors' not in _default()
    assert hashlib.sha256(_default().encode()).hexdigest() == PINNED


def test_transcription_aware_adds_only_instruction_suffix():
    base = json.loads(_default())
    new = json.loads(build_prompt('problem', 'x+1', RUBRIC, 'harness_v2_ambiguity_aware_v1'))
    assert {k: v for k, v in new.items() if k != 'instructions'} == {k: v for k, v in base.items() if k != 'instructions'}
    assert new['instructions'].startswith(base['instructions'])
    assert 'transcription errors' in new['instructions'] and 'do not reconstruct' in new['instructions'] and 'at least 0.5' in new['instructions']


def test_ocr_uncertainty_v2_extends_v1_and_keeps_the_json_shape():
    from harness_v2.grading.scorer import AMBIGUITY_AWARE_V1

    v1 = AMBIGUITY_AWARE_V1
    v2 = PROMPT_VERSIONS["harness_v2_ocr_uncertainty_v2"]
    assert PROMPT_VERSIONS["harness_v2_ambiguity_aware_v1"] == v1
    assert v2.startswith(v1)
    extra = v2[len(v1):]
    assert "OCR transcript" in extra
    assert "misreads" in extra
    assert "do not treat it as a student error by default" in extra
    assert "Never invent or fill in a correct value" in extra
    assert "character misread" in extra
    for banned in ("1-A", "2-B", "3-A", "circ", "或", "pi", "reference"):
        assert banned not in extra
    assert not any(char.isdigit() for char in extra)
    base = json.loads(_default())
    prompted = json.loads(build_prompt("problem", "x+1", RUBRIC, "harness_v2_ocr_uncertainty_v2"))
    assert {key: value for key, value in prompted.items() if key != "instructions"} == {
        key: value for key, value in base.items() if key != "instructions"
    }
    assert prompted["instructions"].startswith(base["instructions"])
    assert v1 in prompted["instructions"]
    assert extra in prompted["instructions"]
    assert hashlib.sha256(_default().encode()).hexdigest() == PINNED


def test_ocr_uncertainty_v2_changes_reuse_identity(tmp_path):
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.run import identity, load_config, request_id_for, reusable_config
    from harness_v2.tests.test_p3_dataset import make_dataset
    from harness_v2.tests.test_p3_runner import config_at

    path = config_at(tmp_path)
    loaded = load_config(path)
    dataset = load_dataset(make_dataset(tmp_path))
    case = dataset.cases[0]
    current = request_id_for(dataset, case, loaded, 1)
    newer = {**loaded, "prompt_version": "harness_v2_ocr_uncertainty_v2"}
    assert request_id_for(dataset, case, newer, 1) != current
    assert identity({"config": reusable_config(loaded), "dataset": dataset.manifest, "commit": "c"}) != (
        identity({"config": reusable_config(newer), "dataset": dataset.manifest, "commit": "c"}))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prompt_version"] = "harness_v2_ocr_uncertainty_v2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_config(path)["prompt_version"] == "harness_v2_ocr_uncertainty_v2"


def test_partial_credit_v3_extends_v2_and_keeps_earlier_prompts():
    from harness_v2.grading.scorer import AMBIGUITY_AWARE_V1, OCR_UNCERTAINTY_V2

    v2 = OCR_UNCERTAINTY_V2
    v3 = PROMPT_VERSIONS["harness_v2_partial_credit_v3"]
    assert PROMPT_VERSIONS["harness_v2_ambiguity_aware_v1"] == AMBIGUITY_AWARE_V1
    assert PROMPT_VERSIONS["harness_v2_ocr_uncertainty_v2"] == v2
    assert v3.startswith(v2)
    extra = v3[len(v2):]
    assert "For each rubric criterion" in extra
    assert "quote the transcript evidence" in extra
    assert "even if the final answer is wrong" in extra
    assert "Never award points outside the rubric" in extra
    assert "The total is the sum of the awarded criterion points" in extra
    for banned in ("1-A", "2-B", "3-A", "circ", "或", "pi", "reference"):
        assert banned not in extra
    assert not any(char.isdigit() for char in extra)
    base = json.loads(_default())
    prompted = json.loads(build_prompt("problem", "x+1", RUBRIC, "harness_v2_partial_credit_v3"))
    assert {key: value for key, value in prompted.items() if key != "instructions"} == {
        key: value for key, value in base.items() if key != "instructions"
    }
    assert prompted["instructions"].startswith(base["instructions"])
    assert v2 in prompted["instructions"]
    assert extra in prompted["instructions"]
    assert hashlib.sha256(_default().encode()).hexdigest() == PINNED


def test_partial_credit_v3_changes_reuse_identity(tmp_path):
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.run import identity, load_config, request_id_for, reusable_config
    from harness_v2.tests.test_p3_dataset import make_dataset
    from harness_v2.tests.test_p3_runner import config_at

    path = config_at(tmp_path)
    loaded = load_config(path)
    dataset = load_dataset(make_dataset(tmp_path))
    case = dataset.cases[0]
    v2 = {**loaded, "prompt_version": "harness_v2_ocr_uncertainty_v2"}
    v3 = {**loaded, "prompt_version": "harness_v2_partial_credit_v3"}
    assert request_id_for(dataset, case, v3, 1) != request_id_for(dataset, case, v2, 1)
    assert request_id_for(dataset, case, v3, 1) != request_id_for(dataset, case, loaded, 1)
    assert identity({"config": reusable_config(v3), "dataset": dataset.manifest, "commit": "c"}) != (
        identity({"config": reusable_config(v2), "dataset": dataset.manifest, "commit": "c"}))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prompt_version"] = "harness_v2_partial_credit_v3"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_config(path)["prompt_version"] == "harness_v2_partial_credit_v3"


def test_suffix_names_no_problem_formula_or_score():
    text = PROMPT_VERSIONS['harness_v2_ambiguity_aware_v1']
    for banned in ('1-A', '2-B', '3-A', 'circ', '或', 'pi', 'reference'):
        assert banned not in text
    assert not any(ch.isdigit() and ch not in '015' for ch in text)


@pytest.mark.parametrize('label', ['p1', 'prompt-v0', 'harness_v2_ambiguity_aware_v2',
                                   'harnes_v2_ambiguity_aware_v1', ' harness_v2_ambiguity_aware_v1', ''])
def test_unregistered_version_rejected(label):
    with pytest.raises(ValueError):
        build_prompt('problem', 'x+1', RUBRIC, label)


def test_run_config_rejects_unregistered_prompt_version(tmp_path):
    from harness_v2.run import load_config
    cfg = {'name': 'x', 'provider': 'fixture', 'model': 'fixture', 'prompt_version': 'harness_v2_typo',
           'rubric_version': 'v1.1', 'generation_settings': {'temperature': 0}, 'fixture_completion': '{}'}
    path = tmp_path / 'c.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='prompt_version'):
        load_config(path)


def test_uncertainty_routes_to_review_and_null_score_fails_without_filling():
    from harness_v2.grading.scorer import items_from_completion
    from harness_v2.grading.validator import validate
    ok = json.dumps({'items': [{'item_id': 'R1', 'score': 4, 'evidence_spans': [{'text': 'x+1'}],
                                'math_reason': 'r', 'uncertainty': 0.6, 'review_reason': 'garbled'}],
                     'total_score': 4})
    null = json.dumps({'items': [{'item_id': 'R1', 'score': None, 'evidence_spans': [{'text': 'x+1'}],
                                  'math_reason': 'r', 'uncertainty': 0.9, 'review_reason': 'garbled'}],
                       'total_score': None})
    for raw, status in ((ok, 'review_required'), (null, 'failed')):
        items, total, *_ = items_from_completion(raw, RUBRIC)
        result = validate(RUBRIC, items, total, 'x+1')
        assert result.status == status
    assert result.items[0].score is None


def test_pipeline_sends_candidate_suffix_and_records_version(tmp_path):
    from harness_v2.core.pipeline import Pipeline
    from harness_v2.providers.fixture import FixtureProvider
    from harness_v2.storage.artifacts import ArtifactStore
    from harness_v2.storage.sqlite_store import RunStore
    completion = json.dumps({'items': [{'item_id': 'R1', 'score': 4, 'evidence_spans': [{'text': 'x+1'}],
                                        'math_reason': 'r', 'uncertainty': 0, 'review_reason': ''}],
                             'total_score': 4})
    seen = []
    provider = FixtureProvider(completion)
    original = provider.generate
    def spy(prompt, **kw):
        seen.append(prompt)
        return original(prompt, **kw)
    provider.generate = spy
    pipe = Pipeline(RunStore(tmp_path / 'r.sqlite', ArtifactStore(tmp_path / 'a')), provider)
    case = {'request_id': 'req-1', 'problem_text': 'p', 'answer_text': 'x+1',
            'rubric': [{'item_id': 'R1', 'max_points': 4, 'description': 'desc'}],
            'prompt_version': 'harness_v2_ambiguity_aware_v1', 'rubric_version': 'v1'}
    record = pipe.run(case, experiment_id='e', commit='c' * 40)
    assert 'do not reconstruct' in json.loads(seen[0])['instructions']
    assert record.prompt_version == 'harness_v2_ambiguity_aware_v1'
