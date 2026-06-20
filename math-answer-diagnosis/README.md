# 高校数学答案 OCR・診断LLMパイプライン

高校の探究研究向けに、手書き数学答案を OCR で読み取り、診断 LLM が採点・誤り分類・フィードバックを生成する最小システムです。

## 研究上の位置づけ

本システムでは、OCR は診断 LLM への前処理として位置づけます。OCR 結果は診断入力とは独立して保存し、診断結果に不自然さが見られた場合には、OCR 誤りか診断 LLM の誤りかを後から確認できる構造とします。OCR 精度そのものの詳細評価は本研究の主目的ではなく、必要に応じて追加検証します。

## 想定環境

- M3 MacBook Air / Apple Silicon
- Python 3.11 以上
- LM Studio（OpenAI 互換 API）
- Z.ai 系小型 OCR モデル（将来接続）
- ローカル JSONL 保存

## セットアップ

```bash
cd math-answer-diagnosis
python -m venv .venv
source .venv/bin/activate
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

## 動作確認（CLI）

Mock Adapter を使った end-to-end スモークテスト:

```bash
cd math-answer-diagnosis
python -m src smoke
```

このコマンドで以下を確認できます。

- 問題データの保存
- 答案データの保存
- Mock OCR 実行と `raw_text` / `used_text` の保存
- OCR 結果の手動修正
- Mock 診断実行と `possible_ocr_issue` の保存
- JSONL の再読み込み
- `config.yaml` / `.env` からの LM Studio 設定読み込み

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

OCR では `raw_text`（OCR 生出力）と `used_text`（診断 LLM に渡すテキスト）を分けて保存します。診断結果が不自然な場合に、OCR 確認ページで `used_text` を修正できます。

## 実装範囲

- Pydantic データモデル
- JSONL 保存層
- Adapter 基底クラス
- MockOCRAdapter / ZaiOCRAdapter（骨組み）/ ManualCorrectionOCRAdapter
- MockDiagnosisAdapter / LMStudioDiagnosisAdapter / LMStudioClient
- Streamlit UI（問題登録〜診断結果確認）

## LM Studio 設定例

`.env`:

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
OCR_PROVIDER=mock
DIAGNOSIS_PROVIDER=mock
```

実際の LM Studio 接続時は `DIAGNOSIS_PROVIDER=lmstudio` や `OCR_PROVIDER=zai` に切り替えます。
