from harness_v2.core.pipeline import Pipeline
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.tests.test_p3_setup import store_at
from harness_v2.tests.test_pipeline import _case, _valid


def test_stored_validated_run_can_create_reference_failure_without_inference(tmp_path):
    from harness_v2.evaluation.reference import failure_from_run
    store = store_at(tmp_path)
    case = _case('r', _valid())
    case.pop('reference_score')
    case.update(problem_id='P', input_version='manual')
    Pipeline(store, FixtureProvider(_valid())).run(case, experiment_id='e', commit='c')
    saved = store.list_runs()[0]
    failure = failure_from_run(saved, reference_score=4, reference_items={'R1': 2})
    assert failure.problem_id == 'P'
    assert failure.input_version == 'manual'
    assert failure.predicted_score == 4
    assert failure.reference_score == 4
    assert failure.errors == ['reference_item_mismatch:R1']
    assert failure_from_run(saved, reference_score=4, reference_items={'R1': 4}) is None
    assert saved['outputs']['grading']['status'] == 'validated'


def test_shared_reference_evaluation_preserves_zero_and_missing_scores():
    from harness_v2.evaluation.reference import reference_errors
    assert reference_errors({'total_score': 1, 'items': []}, 0, {}) == ['reference_total_mismatch']
    assert reference_errors({'total_score': None, 'items': [{'item_id': 'x', 'score': None}]}, 0,
                            {'x': 0}) == ['reference_total_mismatch', 'reference_item_mismatch:x']


def test_plural_reference_maps_never_reach_provider_or_actual_codex_turn(tmp_path):
    import json
    from harness_v2.providers.codex_app_server import CodexAppServer
    from harness_v2.tests.test_codex import Script
    script = Script([{'id': 1, 'result': {}},
                     {'id': 2, 'result': {'thread': {'id': 'fresh'}}},
                     {'id': 3, 'result': {'turn': {'id': 'turn', 'status': 'completed', 'items': []}}}])
    reviewer = CodexAppServer(script)
    reviewer.initialize()
    prompts = []
    class Provider(FixtureProvider):
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return super().generate(prompt, **kwargs)
    case = _case('plural', _valid())
    case['answer_text'] = 'x+1; reference_scores: {"P":9713.25}'
    case['reference_scores'] = {'P': 9713.25}
    Pipeline(store_at(tmp_path), Provider(_valid()), codex_reviewer=reviewer).run(case, experiment_id='e', commit='c')
    codex_prompt = script.sent[-1]['params']['input'][0]['text']
    for prompt in [prompts[0], codex_prompt]:
        assert 'reference_scores' not in prompt
        assert '9713.25' not in prompt
