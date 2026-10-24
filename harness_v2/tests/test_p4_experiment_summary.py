import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "docs/harness_v2/experiments/prompt_ambiguity_aware_v1_summary.md"


def test_summary_is_candidate_only_and_privacy_safe():
    text = DOC.read_text(encoding="utf-8")
    assert "Not for default adoption" in text
    assert "## Limits" in text and "Ambiguous-answer detection" in text
    assert not re.search(r"\.(png|jpe?g|gif|webp|heic)\b|!\[|data:image", text, re.I)
    assert "/workspace/" not in text and "reference_scores" not in text
