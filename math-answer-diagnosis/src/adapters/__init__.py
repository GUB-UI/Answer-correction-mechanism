from src.adapters.base import DiagnosisAdapter, OCRAdapter
from src.adapters.diagnosis_adapter import (
    LMStudioDiagnosisAdapter,
    create_diagnosis_adapter,
)
from src.adapters.lmstudio_client import LMStudioClient
from src.adapters.mock_adapters import MockDiagnosisAdapter, MockOCRAdapter
from src.adapters.ocr_adapter import (
    GLMOCRSdkAdapter,
    ManualCorrectionOCRAdapter,
    create_ocr_adapter,
)

__all__ = [
    "DiagnosisAdapter",
    "GLMOCRSdkAdapter",
    "LMStudioClient",
    "LMStudioDiagnosisAdapter",
    "ManualCorrectionOCRAdapter",
    "MockDiagnosisAdapter",
    "MockOCRAdapter",
    "OCRAdapter",
    "create_diagnosis_adapter",
    "create_ocr_adapter",
]
