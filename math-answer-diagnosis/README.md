# 高校数学答案 OCR・診断LLMパイプライン

高校の探究研究向けに、手書き数学答案を OCR で読み取り、診断 LLM が採点・誤り分類・フィードバックを生成する最小システムです。

## 研究上の位置づけ

本システムでは、OCR は診断 LLM への前処理として位置づけます。OCR 結果は診断入力とは独立して保存し、診断結果に不自然さが見られた場合には、OCR 誤りか診断 LLM の誤りかを後から確認できる構造とします。OCR 精度そのものの詳細評価は本研究の主目的ではなく、必要に応じて追加検証します。

## 想定環境

- M3 MacBook Air / Apple Silicon（または Windows）
- Python 3.11 以上
- GLM-OCR 公式 SDK（OCR 用、任意）
- oMLX（診断 LLM 用、Qwen3.8-27B-oQ4e-mtp + Lightning MTP）
- ローカル JSONL 保存

## OCR 方式

OCRは以下の2方式に対応しています。

1. **mock**
   外部モデルなしで動作確認するためのダミーOCRです。

2. **glmocr_sdk**
   GLM-OCR公式SDKで日本語・文書構造を読み、数字・数式は UniMERNet Small を優先して上書きします。
   UniMERNet は別プロセス（既定: `http://127.0.0.1:8091`）で動かします。

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

## 本番環境（Apple Silicon）

本番は OCR を GLM-OCR SDK、診断を oMLX（Lightning MTP）に接続します。

```bash
cd math-answer-diagnosis
bash scripts/setup_production.sh
bash scripts/start_omlx.sh
bash scripts/start_production.sh
```

- 診断モデル: `Qwen3.8-27B-oQ4e-mtp`（`.env` の `DIAGNOSIS_MODEL_NAME` で変更可）
- OCR 推論: `http://localhost:8080` の mlx-vlm（`mlx-community/GLM-OCR-bf16`）
- 数式OCR: `http://127.0.0.1:8091` の UniMERNet Small（`wanderkid/unimernet_small`）
- アプリ: Streamlit `http://localhost:8501`

## Streamlit UI

```bash
cd math-answer-diagnosis
streamlit run app.py
```

画面フロー:

1. セット登録（ローカルLLMで資料を分離・照合し、同じ問題に自動タグ付け）
2. または問題登録 → 答案登録を個別に行う
3. OCR実行（答案画像、および問題・模範画像）
4. OCR確認・修正（必要時。手書き精度の観察もここ）
5. 診断実行
6. 診断結果確認・CSVエクスポート

### セット登録・自動タグ付け

「セット登録」で資料の構成を選び、画像またはテキストを入力します。

- **自動判別**: 資料内の問題・生徒の解答・模範解答をローカルLLMが判別します。
- **問題＋生徒の解答がセット**: 一枚の答案用紙などから問題と生徒答案を分離します。
- **問題＋模範解答がセット**: 教材などから問題と模範を分離します。
- **それぞれ別の資料**: 役割を指定した各欄へ入力します。

「解析して自動タグ付け」でOCRとローカルLLMによる分類・紐付けを実行し、
問題ごとのプレビューを表示します。複数設問が含まれる場合は設問ごとにタグを発行します。
タグは既存の `problem_id` で、問題文・模範解答・生徒答案を同じIDで管理します。
既存問題と照合して同じ問題なら既存タグを再利用します。設問番号や解答だけが同じでも
同一問題と断定しないよう指示し、低確信度・既存内容との相違・問題文のない紐付けは
「要確認」にします。必要ならプレビューで紐付け先や引用元を修正できます。

問題＋模範だけ、問題＋生徒答案だけでも保存でき、不足資料は後でセット登録から
同じ問題に追加できます。生徒答案がある場合のみ匿名化生徒IDが必須です。
一回の登録では一人分の生徒答案を入力してください。模範がない場合は生成せず未登録とし、
問題文と模範が揃ってから診断へ進みます。既存の問題文・模範解答は保持して不足項目だけ補います。

元画像・OCR全文・抽出文・引用元・モデル出力・確認記録をローカルJSONLに保存します。
生徒答案の診断入力には抽出文だけを使います。自動タグ付け済み画像のOCR再実行は
「セット登録」で再解析してください。「OCR実行」で画像全体を上書きする操作は無効になります。
資料や入力形式を変えた場合は古いプレビューを破棄し、再解析が必要です。

#### ローカルLLMの設定

画像の読み取りには既存のGLM-OCRを、分類・紐付けにはoMLXまたはLM Studioの
OpenAI互換 `/v1/chat/completions` APIを使います。タグ付けではクラウドAPIへの
フォールバックはありません。接続先はループバック、LAN内IP、`.local` ホスト、
または `host.docker.internal` を指定してください。

```env
OMLX_BASE_URL=http://localhost:8000/v1
OMLX_API_KEY=omlx
DIAGNOSIS_MODEL_NAME=Qwen3.8-27B-oQ4e-mtp
# 別モデルを使う場合のみ指定（未指定なら診断モデルを使用）
TAGGING_MODEL_NAME=Qwen3.8-27B-oQ4e-mtp
```

LM Studioの場合は `OMLX_BASE_URL=http://localhost:1234/v1` と、ロード済みモデルの
名前を指定します。従来の `LMSTUDIO_BASE_URL` / `LMSTUDIO_API_KEY` も利用できます
（`OMLX_*` が優先）。診断providerが `mock` でも、自動タグ付けには実際のローカルLLMが必要です。
OCRの `mock` は動作確認用のダミーであり、実資料にはGLM-OCRを選択してください。

一回の解析は入力と既存問題候補のJSON合計60,000文字までです。多い場合は入力を分けるか、
「登録・読み取り設定」の検索範囲で既存問題を絞り込んでください。
テストではローカルHTTPサーバーとStreamlit AppTestで接続・保存を検証します。
実モデルの分割・紐付け精度は、利用環境のモデルと資料で別途確認してください。

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
OMLX_BASE_URL=http://localhost:8000/v1
OMLX_API_KEY=omlx
OCR_PROVIDER=glmocr_sdk
OCR_MODEL_NAME=mlx-community/GLM-OCR-bf16
UNIMERNET_ENABLED=true
UNIMERNET_MODEL_NAME=wanderkid/unimernet_small
UNIMERNET_BASE_URL=http://127.0.0.1:8091
DIAGNOSIS_PROVIDER=omlx
DIAGNOSIS_MODEL_NAME=Qwen3.8-27B-oQ4e-mtp
```
