"""Offline provider used by tests and the old/new compare script.

This is not a model. It returns a completion that was already saved on the case.
"""

from __future__ import annotations

from typing import Any

from harness_v2.core.contracts import ProviderExchange


class FixtureProvider:
    def __init__(self, completion: str, *, model_id: str = "fixture") -> None:
        self.completion = completion
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        settings: dict[str, Any] | None = None,
    ) -> ProviderExchange:
        del prompt, settings
        return ProviderExchange(
            request_id=request_id,
            model_requested=self.model_id,
            model_returned=self.model_id,
            settings={},
            input_tokens=None,
            output_tokens=None,
            latency_ms=0,
            raw_response=self.completion,
            status="ok",
        )
