# 再開メモ（2026-09-08）

高校数学答案 OCR・診断アプリ。作業ディレクトリは本フォルダ。親は `/Users/ssh56/Desktop/F2`。

## いまの状態（完了）

- 診断は **oMLX + Qwen3.8-27B-oQ4e-mtp + Lightning MTP**。LM Studio は使わない。
- OCR は **GLM-OCR（日本語・レイアウト）+ UniMERNet Small（数字・数式優先）**。
- 問題/答案はテキスト・画像・混在で登録できる（`pages/0_set_register.py`）。
- テスト **44 passed**。
- UniMERNet 重みは `data/models/unimernet_small/` に取得済み（git 対象外、約 773MB）。

## 起動

```bash
cd math-answer-diagnosis
bash scripts/setup_production.sh   # 初回のみ
bash scripts/start_production.sh   # oMLX :8000, GLM-OCR :8080, UniMERNet :8091, Streamlit :8501
```

UniMERNet だけ: `bash scripts/start_unimernet.sh` → `http://127.0.0.1:8091/health`

## 構成の約束

| 役割 | 場所 | 注意 |
|---|---|---|
| アプリ | `.venv` | `transformers` を 4.x に落とさない（PP-DocLayout が壊れる） |
| GLM-OCR | `.venv-mlx` :8080 | `mlx-community/GLM-OCR-bf16` |
| UniMERNet Small | `.venv-unimernet` :8091 | `unimernet` はここに隔離。`pyarrow<17` 必須 |
| 診断 | F2 の oMLX `:8000` | モデル `Qwen3.8-27B-oQ4e-mtp`。32GB なので GGUF 版 Qwen と同時ロードしない |

数式置換の判定: レイアウトが formula 系、または日本語がほぼ無い計算行だけ UniMERNet。日本語交じりブロックは GLM のまま。`bbox_2d` は 0–1000 正規化。

## 次にやること

1. Streamlit で答案 `ans_e1873ce1` を再 OCR し、OCR 確認画面の「GLM → UniMERNet」置換を見る。
2. 印刷の教科書画像（問題+模範）を `data/images/` に置いて OCR。手書きより formula 領域が付きやすい。
3. 手書きは PP-DocLayout が `text` にまとめがち。`8x<40` が `x<40` になる誤読は観察継続。最終答 `7 < a < 10` は生徒の実答（OCR 誤りではない）。
4. 必要ならこのブランチ `2026-05-03-aho1` を `origin`（GitHub `GUB-UI/Answer-correction-mechanism`）へ push。

## つまずきポイント

- UniMERNet 起動で `generate_cfg` / `evaluate` 欠け → `scripts/unimernet_server.py` の YAML に入れて直済み。
- `datasets` が `pyarrow` 25 で落ちる → UniMERNet venv は `pyarrow==16.1.0`（`<17`）。
- アプリ venv に `unimernet` を入れない。
