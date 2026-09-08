from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.config import AppConfig

# Qwen 3.5/3.8 の思考モードを REST API から確実に切るためのプレフィル。
# LM Studio は chat_template_kwargs を無視することがある。
_THINKING_OFF_PREFILL = "<think>\n\n</think>\n\n"


class LMStudioClient:
    """LM Studio の OpenAI 互換 API クライアント。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.lmstudio.base_url,
            api_key=config.lmstudio.api_key,
            timeout=180.0,
        )

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
        payload_messages = list(messages)
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
