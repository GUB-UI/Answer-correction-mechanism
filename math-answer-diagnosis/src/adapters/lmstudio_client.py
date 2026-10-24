from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from src.adapters.diagnosis_helpers import valid_backend_fingerprint
from src.config import AppConfig

# Qwen 3.5/3.8 の思考モードを REST API から確実に切るためのプレフィル。
# LM Studio は chat_template_kwargs を無視することがある。
_THINKING_OFF_PREFILL = "<think>\n\n</think>\n\n"


class LMStudioClient:
    """LM Studio の OpenAI 互換 API クライアント。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.fingerprint_unavailable_reason: str | None = None
        self._client = OpenAI(
            base_url=config.lmstudio.base_url,
            api_key=config.lmstudio.api_key,
            # Colab の 27B 4bit は1件数分かかることがあるため既定を長めにする。
            timeout=float(os.environ.get("LLM_REQUEST_TIMEOUT_SEC", "600")),
        )

    def backend_fingerprint(self) -> dict[str, Any] | None:
        """Require loaded identity and effective settings from the actual endpoint.

        An OpenAI model listing or response model alias cannot verify the weights
        or server-side generation overrides. Unsupported servers disable reuse.
        """
        from http.client import HTTPException
        from urllib.request import Request, urlopen

        self.fingerprint_unavailable_reason = None
        url = str(self._client.base_url).rstrip("/") + "/health"
        request = Request(url, headers={"Authorization": f"Bearer {self.config.lmstudio.api_key}"})
        try:
            with urlopen(request, timeout=2) as response:
                payload = json.load(response)
        except TimeoutError:
            self.fingerprint_unavailable_reason = "backend_health_timeout"
            return None
        except (OSError, ValueError, HTTPException) as exc:
            self.fingerprint_unavailable_reason = f"backend_health_unavailable:{type(exc).__name__}"
            return None
        if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("error"):
            self.fingerprint_unavailable_reason = "backend_health_invalid_response"
            return None
        fingerprint = payload.get("backend_fingerprint")
        if not valid_backend_fingerprint(fingerprint):
            self.fingerprint_unavailable_reason = "backend_fingerprint_unavailable"
            return None
        return fingerprint

    @staticmethod
    def prepare_chat_request(
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        response_format_json: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        disable_thinking: bool = True,
    ) -> dict[str, Any]:
        """Build the final JSON messages once for both hashing and transmission."""
        payload_messages = json.loads(json.dumps(messages, ensure_ascii=False))
        if disable_thinking:
            last = payload_messages[-1] if payload_messages else None
            if last is None or last.get("role") != "assistant":
                payload_messages.append(
                    {"role": "assistant", "content": _THINKING_OFF_PREFILL}
                )

        kwargs: dict[str, Any] = {
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        if model:
            kwargs["model"] = model
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        response_format_json: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        disable_thinking: bool = True,
    ) -> str:
        kwargs = self.prepare_chat_request(
            messages=messages, model=model, response_format_json=response_format_json,
            temperature=temperature, max_tokens=max_tokens, disable_thinking=disable_thinking,
        )

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception:
            if not response_format_json:
                raise
            kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        content = message.content
        if isinstance(content, str) and content.strip():
            return content
        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        raise RuntimeError("Local LLM response did not include text content")

    def vision_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            disable_thinking=True,
        )
