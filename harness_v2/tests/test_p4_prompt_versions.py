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
        items, total, _ = items_from_completion(raw, RUBRIC)
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
