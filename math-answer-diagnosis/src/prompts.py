"""プロンプト定義"""

OCR_PROMPT = """この画像は高校数学の手書き答案です。
答案に書かれている日本語、数式、途中式を上から順に読み取ってください。

採点、解説、誤り分類はしないでください。
読めない箇所は [不明] と書いてください。
符号、指数、分数、ルート、添字はできるだけ正確に読み取ってください。

出力はMarkdownで、行ごとに整理してください。"""

# Xiaomi-OCR-0 の公式例は英語で、本文を Markdown、数式を LaTeX、表を OTSL にする。
# 採点側は Markdown を読むので、表は Markdown の表にし、日本語は翻訳しない。
XIAOMI_OCR_PROMPT = """この画像は高校数学の手書き答案です。ページ全体を、読み順に Markdown で書き起こしてください。
数式は LaTeX にしてください。行の中の式は $...$、独立した式は $$...$$ です。
表は Markdown の表にしてください。
日本語はそのまま書き、中国語に翻訳しないでください。
ヘッダーとフッターは無視してください。
採点、解説、誤りの指摘はしないでください。
読めない箇所は [不明] と書いてください。"""

DIAGNOSIS_COMMON_INSTRUCTION = """あなたは高校数学答案の診断補助AIです。
以下はOCRによって読み取られた答案です。
OCRに誤読が含まれる可能性があります。
答案内容に不自然な点がある場合は、OCR誤りの可能性として teacher_review_notes に記録してください。
問題文、模範解答、採点基準、OCR結果をもとに、採点・誤り分類・フィードバックを作成してください。
AI診断は教師確認前の下書きであり、最終判断ではありません。
思考過程や前置きは出さず、JSONオブジェクトのみを返してください。
reasoning_summary / student_feedback / teacher_review_notes は各1文にしてください。
設問が複数ある場合（(1)(2) など）は question_scores に設問ごとの得点を入れ、score はその合計にしてください。
question には設問番号だけを書いてください。(1) や 2 などです。
答案全体を question が「全体」「合計」「total」の要素にしてはいけません。全体の点は score と max_score です。
配点が与えられている場合は、その設問を question_scores にすべて書き、配点に無い設問は足さないでください。空の配列にしてはいけません。
possible_ocr_issue が true のときも、答案全体を 0 点にしてはいけません。
配点の設問ごとに、読める手順が正しければ部分点を付け、読めない行だけを減点してください。
そのときは review_status を needs_teacher_review にしてください。この得点は教師確認前の下書きです。"""

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
  "review_status": "needs_teacher_review",
  "confidence": 0.68,
  "question_scores": [
    {"question": "(1)", "score": 2, "max_score": 2, "reason": "正しく解けている"},
    {"question": "(2)", "score": 1, "max_score": 3, "reason": "2行目に計算ミス"}
  ]
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
    "recheck": (
        f"{DIAGNOSIS_COMMON_INSTRUCTION}\n"
        "これは再チェックです。前回のAI採点は教師採点と大きく食い違いました。\n"
        "前回の結果を鵜呑みにせず、途中式を1行ずつ検算し、採点基準と配点に照らして改めて採点してください。\n"
        "食い違いの原因が OCR の誤読らしい場合は possible_ocr_issue を true にし、"
        "どの行が疑わしいかを teacher_review_notes に書いてください。"
        "OCR結果を推測で書き換えて採点してはいけません。\n"
        f"出力形式:\n{DIAGNOSIS_OUTPUT_SCHEMA}"
    ),
}

RECHECK_PROMPT_TYPE = "recheck"
DEFAULT_DIAGNOSIS_PROMPT_TYPE = "basic_scoring"
