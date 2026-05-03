"""解答画像を LM Studio の API（OCR 用・採点用 LLM）で処理し JSON を返す。"""

from answer_correction.pipeline import process_answer_image, PipelineConfig

__all__ = ["process_answer_image", "PipelineConfig"]
