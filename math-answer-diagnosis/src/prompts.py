"""プロンプト定義"""

OCR_PROMPT = """この画像は高校数学の手書き答案です。
答案に書かれている日本語、数式、途中式を上から順に読み取ってください。
採点、解説、誤り分類はしないでください。
読めない箇所は [不明] と書いてください。
符号、指数、分数、ルート、添字はできるだけ正確に読み取ってください。
出力はMarkdownで、行ごとに整理してください。"""

DIAGNOSIS_COMMON_INSTRUCTION = """あなたは高校数学答案の診断補助AIです。
以下はOCRによって読み取られた答案です。
OCRに誤読が含まれる可能性があります。
答案内容に不自然な点がある場合は、OCR誤りの可能性として teacher_review_notes に記録してください。
問題文、模範解答、採点基準、OCR結果をもとに、採点・誤り分類・フィードバックを作成してください。
AI診断は教師確認前の下書きであり、最終判断ではありません。
出力は必ずJSON形式にしてください。"""

DIAGNOSIS_OUTPUT_SCHEMA = """{
  "score": 3,
  "max_score": 5,
  "error_categories": ["計算ミス"],
  "error_locations": [
    {
      "line": "2行目",
      "description": "符号処理に誤りがある可能性"
    }
  ],
  "reasoning_summary": "方針は正しいが、途中式に計算ミスがある可能性があります。",
  "student_feedback": "方針はよいので、符号と式変形を一行ずつ確認しましょう。",
  "teacher_review_notes": "2行目の符号が不自然なため、OCR誤りの可能性があります。",
  "possible_ocr_issue": true,
  "confidence": 0.68
}"""

DIAGNOSIS_PROMPTS: dict[str, str] = {
    "basic_scoring": (
        f"{DIAGNOSIS_COMMON_INSTRUCTION}\n"
        "採点基準に沿って総合的な得点を判定してください。\n"
        f"出力形式:\n{DIAGNOSIS_OUTPUT_SCHEMA}"
    ),
    "step_by_step_check": (
        f"{DIAGNOSIS_COMMON_INSTRUCTION}\n"
        "途中式を順に確認し、誤りの位置を error_locations に具体的に記録してください。\n"
        f"出力形式:\n{DIAGNOSIS_OUTPUT_SCHEMA}"
    ),
    "error_classification": (
        f"{DIAGNOSIS_COMMON_INSTRUCTION}\n"
        "誤り分類を重視し、error_categories を適切に選択してください。\n"
        f"出力形式:\n{DIAGNOSIS_OUTPUT_SCHEMA}"
    ),
    "student_feedback": (
        f"{DIAGNOSIS_COMMON_INSTRUCTION}\n"
        "生徒向けフィードバックを丁寧に作成してください。\n"
        f"出力形式:\n{DIAGNOSIS_OUTPUT_SCHEMA}"
    ),
    "teacher_review": (
        f"{DIAGNOSIS_COMMON_INSTRUCTION}\n"
        "教師確認が必要な点を teacher_review_notes に重点的に記録してください。\n"
        f"出力形式:\n{DIAGNOSIS_OUTPUT_SCHEMA}"
    ),
}

DEFAULT_DIAGNOSIS_PROMPT_TYPE = "basic_scoring"
