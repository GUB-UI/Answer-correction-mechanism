"""Finding 4: all tests stub SDK construction; no tier-verification request."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from harness_v2.providers.gemma_ai_studio import GemmaAIStudio, ProviderSetupError, QuotaExceeded


def fake_sdk(monkeypatch, *, quota=False):
    built, calls = [], []

    def generate(**kwargs):
        calls.append(kwargs)
        if quota:
            error = RuntimeError("429")
            error.code = 429
            raise error
        return SimpleNamespace(text='{}')

    def client(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate))

    google = ModuleType("google")
    google.genai = SimpleNamespace(Client=client)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return built, calls


@pytest.mark.parametrize("opt_in", [False, None, "true", 1])
def test_requires_explicit_boolean_free_tier_attestation(monkeypatch, opt_in):
    built, calls = fake_sdk(monkeypatch)
    with pytest.raises(ProviderSetupError, match="free_tier_project"):
        GemmaAIStudio(free_tier_project=opt_in).generate("grade", request_id="r")
    assert built == calls == []


def test_omitted_attestation_is_refused_before_sdk_construction(monkeypatch):
    built, calls = fake_sdk(monkeypatch)
    with pytest.raises(ProviderSetupError, match="free_tier_project"):
        GemmaAIStudio().generate("grade", request_id="r")
    assert built == calls == []


@pytest.mark.parametrize("quota", [False, True])
def test_endpoint_and_attestation_are_enforced_and_recorded(monkeypatch, quota):
    built, calls = fake_sdk(monkeypatch, quota=quota)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://paid.example")
    monkeypatch.setenv("GOOGLE_VERTEX_BASE_URL", "https://paid.example")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    provider = GemmaAIStudio(free_tier_project=True)
    if quota:
        with pytest.raises(QuotaExceeded) as caught:
            provider.generate("grade", request_id="r")
        exchange = caught.value.exchange
    else:
        exchange = provider.generate("grade", request_id="r")
    assert built == [{"vertexai": False, "api_key": "test-key",
                      "http_options": {"base_url": "https://generativelanguage.googleapis.com"}}]
    assert exchange.settings["provider_policy"] == {
        "free_tier_project": True, "vertexai": False,
        "base_url": "https://generativelanguage.googleapis.com",
        "attestation_source": "caller_config",
    }
    assert "provider_policy" not in calls[0]["config"]
    assert len(calls) == 1


def test_policy_attestation_survives_run_persistence(monkeypatch, tmp_path):
    from harness_v2.core.pipeline import Pipeline
    from harness_v2.storage.artifacts import ArtifactStore
    from harness_v2.storage.sqlite_store import RunStore
    fake_sdk(monkeypatch)
    store = RunStore(tmp_path / 'runs.sqlite', ArtifactStore(tmp_path / 'artifacts'))
    Pipeline(store, GemmaAIStudio(free_tier_project=True)).run({
        'request_id': 'r', 'answer_text': 'x',
        'rubric': [{'item_id': 'a', 'max_points': 3}],
    }, experiment_id='e', commit='c')
    policy = store.list_runs()[0]['outputs']['settings']['provider_policy']
    assert policy['free_tier_project'] is True
    assert policy['vertexai'] is False
    assert policy['base_url'] == 'https://generativelanguage.googleapis.com'
