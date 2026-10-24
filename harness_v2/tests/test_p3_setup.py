import json

import pytest

from harness_v2.core.pipeline import Pipeline
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.providers.gemma_ai_studio import GemmaAIStudio, ProviderSetupError, QuotaExceeded
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.sqlite_store import RunStore
from harness_v2.tests.test_pipeline import _case, _valid


def store_at(tmp_path):
    return RunStore(tmp_path / 'runs.sqlite', ArtifactStore(tmp_path / 'artifacts'))


@pytest.mark.parametrize('error', [ProviderSetupError('SDK missing'), RuntimeError('before return')])
def test_provider_exceptions_are_recorded_and_same_id_can_retry(tmp_path, error):
    class Broken:
        def generate(self, *args, **kwargs):
            raise error
    store = store_at(tmp_path)
    case = _case('same', _valid())
    failed = Pipeline(store, Broken()).run(case, experiment_id='e', commit='c')
    assert failed.outputs['grading']['status'] == 'failed'
    assert type(error).__name__ in failed.failure_reason
    first_prompt = (store.artifacts.root / failed.outputs['prompt_artifact']).read_bytes()
    good = Pipeline(store, FixtureProvider(_valid())).run(case, experiment_id='e', commit='c')
    assert good.outputs['grading']['status'] == 'validated'
    assert failed.outputs['artifact_id'] != good.outputs['artifact_id']
    assert (store.artifacts.root / failed.outputs['prompt_artifact']).read_bytes() == first_prompt
    assert [r['request_id'] for r in store.list_runs()] == ['same', 'same']
    assert {r['outputs']['run_attempt'] for r in store.list_runs()} == {1, 2}


def test_missing_key_writes_failed_run_without_client(tmp_path, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    store = store_at(tmp_path)
    record = Pipeline(store, GemmaAIStudio()).run(_case('key', _valid()), experiment_id='e', commit='c')
    assert 'MissingAPIKey' in record.failure_reason
    assert store.list_runs()[0]['outputs']['grading']['status'] == 'failed'


def test_orphaned_write_once_artifact_does_not_block_retry(tmp_path):
    store = store_at(tmp_path)
    store.artifacts.write_private_text('ocr_private/same.txt', 'orphan')
    record = Pipeline(store, FixtureProvider(_valid())).run(_case('same', _valid()), experiment_id='e', commit='c')
    assert record.outputs['grading']['status'] == 'validated'
    assert record.outputs['run_attempt'] == 2
    assert store.artifacts.read_text('ocr_private/same.txt') == 'orphan'


def test_quota_is_recorded_once_by_pipeline_and_rerun_works(tmp_path):
    from harness_v2.core.contracts import ProviderExchange
    class Limited:
        def generate(self, prompt, *, request_id):
            raise QuotaExceeded(ProviderExchange(request_id, 'm', '', {}, None, None, 10, '429', 'quota'))
    store = store_at(tmp_path)
    with pytest.raises(QuotaExceeded) as caught:
        Pipeline(store, Limited()).run(_case('q', _valid()), experiment_id='e', commit='c')
    assert caught.value.record.failure_reason == 'quota'
    assert len(store.list_runs()) == 1
    good = Pipeline(store, FixtureProvider(_valid())).run(_case('q', _valid()), experiment_id='e', commit='c')
    assert good.outputs['grading']['status'] == 'validated'
