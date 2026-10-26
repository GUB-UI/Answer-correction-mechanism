from types import SimpleNamespace

import pytest

from harness_v2.providers.gemma_ai_studio import GemmaAIStudio, QuotaExceeded
from harness_v2.core.pipeline import Pipeline
from harness_v2.tests.test_p3_setup import store_at
from harness_v2.tests.test_pipeline import _case, _valid


@pytest.mark.parametrize('last', ['ok', 'error', 'quota'])
def test_total_latency_and_history_include_all_attempts(tmp_path, monkeypatch, last):
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-key')
    now = [0.0]
    calls = []
    class ServerError(Exception):
        code = 500
    class RateLimit(Exception):
        code = 429
    def generate_content(**kwargs):
        calls.append(kwargs)
        now[0] += 0.02 if len(calls) == 1 else 0.03
        if len(calls) == 1 or last == 'error':
            raise ServerError('server')
        if last == 'quota':
            raise RateLimit('quota')
        return SimpleNamespace(text=_valid())
    def sleep(seconds):
        now[0] += seconds
    provider = GemmaAIStudio(client_factory=lambda **kwargs: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)),
        monotonic=lambda: now[0], sleep=sleep, rng=lambda: 0, max_attempts=2)
    store = store_at(tmp_path)
    try:
        rec = Pipeline(store, provider).run(_case('r', _valid()), experiment_id='e', commit='c')
    except QuotaExceeded as exc:
        assert last == 'quota'
        rec = exc.record
    assert rec.latency_ms == pytest.approx(30050)
    attempts = rec.outputs['attempts']
    assert rec.outputs['attempt_count'] == len(attempts) == 2
    assert [a['latency_ms'] for a in attempts] == pytest.approx([20, 30])
    assert [a['status'] for a in attempts] == ['error', 'ok' if last == 'ok' else last]
    assert attempts[0]['error_class'] == 'ServerError'
    assert attempts[1]['error_class'] == ('' if last == 'ok' else 'RateLimit' if last == 'quota' else 'ServerError')
    saved = store.list_runs()[0]
    assert saved['latency_ms'] == rec.latency_ms
    assert saved['outputs']['attempts'] == attempts
