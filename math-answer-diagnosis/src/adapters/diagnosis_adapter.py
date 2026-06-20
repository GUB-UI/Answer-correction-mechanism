from __future__ import annotations

from src.adapters.base import DiagnosisAdapter, DiagnosisRunContext
from src.adapters.diagnosis_helpers import build_diagnosis_result, parse_diagnosis_json
from src.adapters.lmstudio_client import LMStudioClient
from src.adapters.mock_adapters import MockDiagnosisAdapter
from src.config import AppConfig, load_config
from src.models import DiagnosisResult
from src.prompts import DEFAULT_DIAGNOSIS_PROMPT_TYPE, DIAGNOSIS_PROMPTS


class LMStudioDiagnosisAdapter(DiagnosisAdapter):
    def __init__(
        self,
        config: AppConfig | None = None,
        lmstudio_client: LMStudioClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.lmstudio_client = lmstudio_client or LMStudioClient(self.config)

    def run(self, context: DiagnosisRunContext) -> dict:
        prompt_type = context.prompt_type or DEFAULT_DIAGNOSIS_PROMPT_TYPE
        system_prompt = DIAGNOSIS_PROMPTS.get(
            prompt_type,
            DIAGNOSIS_PROMPTS[DEFAULT_DIAGNOSIS_PROMPT_TYPE],
        )
        user_prompt = self._build_user_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw_text = self.lmstudio_client.chat_completion(
            messages=messages,
            model=context.model_name or self.config.diagnosis.model_name,
            response_format_json=True,
        )
        parsed = parse_diagnosis_json(raw_text)
        parsed["_transport"] = "lm_studio"
        parsed["_prompt_type"] = prompt_type
        return parsed

    def to_diagnosis_result(
        self,
        context: DiagnosisRunContext,
        raw_output: dict,
        diagnosis_id: str,
        created_at: str,
    ) -> DiagnosisResult:
        return build_diagnosis_result(
            context,
            raw_output,
            diagnosis_id,
            created_at=created_at,
        )

    @staticmethod
    def _build_user_prompt(context: DiagnosisRunContext) -> str:
        problem = context.problem
        return (
            "以下の情報をもとに診断してください。\n\n"
            f"【問題文】\n{problem.problem_text}\n\n"
            f"【模範解答】\n{problem.correct_answer}\n\n"
            f"【採点基準】\n{problem.rubric}\n\n"
            f"【OCR結果 used_text】\n{context.used_text}\n"
        )


def create_diagnosis_adapter(
    provider: str | None = None,
    config: AppConfig | None = None,
) -> DiagnosisAdapter:
    cfg = config or load_config()
    selected = (provider or cfg.diagnosis.provider).lower()
    if selected == "mock":
        return MockDiagnosisAdapter()
    if selected in {"lmstudio", "lm_studio"}:
        return LMStudioDiagnosisAdapter(config=cfg)
    raise ValueError(f"Unsupported diagnosis provider: {selected}")
