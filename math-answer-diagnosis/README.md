# 高校数学答案 OCR・診断LLMパイプライン

高校の探究研究向けに、手書き数学答案を OCR で読み取り、診断 LLM が採点・誤り分類・フィードバックを生成する最小システムです。

## 研究上の位置づけ

本システムでは、OCR は診断 LLM への前処理として位置づけます。OCR 結果は診断入力とは独立して保存し、診断結果に不自然さが見られた場合には、OCR 誤りか診断 LLM の誤りかを後から確認できる構造とします。OCR 精度そのものの詳細評価は本研究の主目的ではなく、必要に応じて追加検証します。

## 想定環境

- M3 MacBook Air / Apple Silicon（または Windows）
- Python 3.11 以上
- GLM-OCR 公式 SDK（OCR 用、任意）
- LM Studio（診断 LLM 用、OpenAI 互換 API）
- ローカル JSONL 保存

## OCR 方式

OCRは以下の2方式に対応しています。

1. **mock**
   外部モデルなしで動作確認するためのダミーOCRです。

2. **glmocr_sdk**
   GLM-OCR公式SDKをPythonから呼び出す方式です。
   SDKの導入方法は公式READMEに従ってください。
   SDK未導入でも、mock OCRでアプリ全体の動作確認は可能です。

GLM-OCR SDKを使用する場合は、公式READMEに従って別途インストールしてください。
SDK未導入の場合でも、MockOCRAdapterによりアプリ全体の動作確認は可能です。

LM StudioはOCRには使用しません。
LM Studioは診断LLMをローカル実行するために使用します。

## セットアップ

```bash
cd math-answer-diagnosis
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

`config.yaml` と `.env` で LM Studio や Adapter の設定を変更できます。URL や API キーはコードに直書きしません。

## Streamlit UI

```bash
cd math-answer-diagnosis
streamlit run app.py
```

画面フロー:

1. 問題登録
2. 答案登録
3. OCR実行（まずは mock）
4. OCR確認・修正（必要時のみ）
5. 診断実行（まずは mock）
6. 診断結果確認・CSVエクスポート

### Windows での動作確認

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pytest
streamlit run app.py
```

Windows ではまず Mock OCR で以下を確認してください。

- 問題登録できる
- 答案画像を登録できる
- Mock OCR を実行できる
- OCR `raw_text` / `used_text` を保存できる
- OCR 結果を修正できる
- Mock 診断を実行できる
- 診断結果を確認できる
- CSV エクスポートできる

GLM-OCR SDK の動作確認は必須条件ではありません。

## 動作確認（CLI）

Mock Adapter を使った end-to-end スモークテスト:

```bash
cd math-answer-diagnosis
python -m src smoke
```

## テスト

```bash
cd math-answer-diagnosis
pytest -q
```

## ディレクトリ構成

```text
math-answer-diagnosis/
  app.py
  pages/
  config.yaml
  .env.example
  requirements.txt
  src/
    models.py
    storage.py
    config.py
    prompts.py
    ui_common.py
    adapters/
  data/
    images/
    records/
  tests/
```

## データ保存

JSONL 形式で以下を保存します。

- `data/records/problems.jsonl`
- `data/records/answers.jsonl`
- `data/records/ocr_results.jsonl`
- `data/records/diagnosis_results.jsonl`

OCR では `raw_text`（OCR 生出力）と `used_text`（診断 LLM に渡すテキスト）を分けて保存します。診断 LLM には必ず `used_text` を渡します。

## 実装範囲

- Pydantic データモデル
- JSONL 保存層
- MockOCRAdapter / GLMOCRSdkAdapter / ManualCorrectionOCRAdapter
- MockDiagnosisAdapter / LMStudioDiagnosisAdapter
- Streamlit UI（問題登録〜診断結果確認）

## 設定例

`.env`:

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
OCR_PROVIDER=mock
OCR_MODEL_NAME=glm-ocr
DIAGNOSIS_PROVIDER=mock
```

実際の LM Studio 接続時は `DIAGNOSIS_PROVIDER=lmstudio` に切り替えます。
OCR を GLM-OCR SDK で実行する場合は `OCR_PROVIDER=glmocr_sdk` に切り替えます。
