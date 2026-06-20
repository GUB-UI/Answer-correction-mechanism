from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.config import AppConfig


class LMStudioClient:
    """LM Studio の OpenAI 互換 API クライアント。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.lmstudio.base_url,
            api_key=config.lmstudio.api_key,
        )

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        response_format_json: bool = False,
        temperature: float = 0.2,
    ) -> str:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
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

        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("LM Studio response did not include text content")
        return content

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
        )
