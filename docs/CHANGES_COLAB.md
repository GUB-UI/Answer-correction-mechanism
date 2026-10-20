# Colab / CUDA 対応の変更一覧

Mac の oMLX + mlx-vlm 経路は残し、Google Colab（CUDA）でも同じ Streamlit アプリが動くようにした。
GPU メモリは、モデルを別プロセスにして **GLM-OCR → UniMERNet → 診断LLM** の順に1つだけ載せることで空ける。
答案と採点は `APP_DATA_DIR`（Google Drive を想定）に書く。

診断モデルの実体は次のとおり。ID は `LLM_MODEL_ID` で変えられる。

| 呼び方 | Hugging Face | いつ使うか |
| --- | --- | --- |
| Qwen3.8-27B 4bit（推奨） | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`（約 22GB） | A100 40GB など 40GB 級 |
| 同じ 27B を読み込み時に NF4 | `Qwen/Qwen3.8-27B` | L4 など 24GB 級。Dynamic 4bit は載らない |
| 代替 | `Qwen/Qwen3.5-9B` を NF4 | T4 16GB。27B の 4bit は約 17GB 以上あり載らない |

## 追加したファイル

| ファイル | 内容と理由 |
| --- | --- |
| `math-answer-diagnosis/scripts/llm_server.py` | 診断 LLM の OpenAI 互換サーバー。CUDA では bitsandbytes の NF4。事前量子化済みチェックポイント（Unsloth）はその設定を使う。言語モデル部分だけ読み、採点の VRAM を抑える。 |
| `math-answer-diagnosis/scripts/glmocr_server.py` | `zai-org/GLM-OCR` を transformers で動かし、Mac の mlx-vlm と同じ `/chat/completions` を返す。公式 SDK がレイアウト後に叩く相手。 |
| `math-answer-diagnosis/scripts/_openai_compat.py` | 上の2つの共通 HTTP 層。思考オフ用の `<think>` プレフィルもここで扱う。 |
| `math-answer-diagnosis/scripts/colab_setup.sh` | アプリの依存関係と、transformers の版が衝突しない別環境の UniMERNet を用意する。UniMERNet の torch は既定で CPU（数式領域は小さい。GPU 版 torch の版ずれを避ける）。Colab は `ensurepip` が無いので `python3 -m venv` を使わず、`uv venv` と `uv pip install --python` で入れる。既存ディレクトリでは `--allow-existing`、壊れているときは `--clear --force` とし、置き換え確認で止まらない。 |
| `math-answer-diagnosis/scripts/start_streamlit_tunnel.sh` | Streamlit を起動し、cloudflared の一時 URL を出す。 |
| `math-answer-diagnosis/src/runtime/gpu.py` | `nvidia-smi` の VRAM から A100 / L4 / T4 / CPU の設定を選ぶ。 |
| `math-answer-diagnosis/src/runtime/services.py` | 3つのモデルサーバーを起動・停止する。`exclusive_gpu` では次を起動する前に前のプロセスを終了し、VRAM を返す。`external`（Mac の既定）では何もしない。 |
| `math-answer-diagnosis/src/runtime/envfile.py` | `.env` の必要なキーだけ書き換え、他の行は残す。 |
| `math-answer-diagnosis/src/runtime/__init__.py` | パッケージの印。 |
| `math-answer-diagnosis/src/teacher_compare.py` | 教師採点 CSV の取り込み、設問ごとの突合、MAE・一致率・相関・重み付きκ、不一致ログ、再チェック。 |
| `math-answer-diagnosis/src/batch_pipeline.py` | 答案をまとめて、OCR・数式・診断の各段階でモデルを1回だけ載せる。 |
| `math-answer-diagnosis/config.colab.yaml` | Colab 用の設定。Mac の `config.yaml` は変えていない。 |
| `math-answer-diagnosis/requirements-colab.txt` | transformers 5 系、bitsandbytes、GLM-OCR SDK。Colab の CUDA 版 torch は入れ直さない。 |
| `math-answer-diagnosis/notebooks/colab_answer_correction.ipynb` | Drive マウントからトンネル、一括採点、教師比較までの手順。clone は `BRANCH`（既定は PR ブランチ。`REPO_BRANCH` でも指定）を checkout する。既定ブランチには `scripts/colab_setup.sh` が無い。 |
| `math-answer-diagnosis/pages/7_teacher_compare.py` | 教師採点の取り込み・比較・再チェックの画面。 |
| `math-answer-diagnosis/data/examples/teacher_grades_template.csv` | 教師採点 CSV の見本（生徒5人分の実験でもこの形）。 |
| `math-answer-diagnosis/tests/test_teacher_compare.py` | 取り込み、突合、指標、再チェックのテスト。 |
| `math-answer-diagnosis/tests/test_runtime.py` | GPU 振り分け、サーバー管理、一括処理、設定の置き場所のテスト。 |
| `docs/CHANGES_COLAB.md` | この一覧。 |

## 変更したファイル

| ファイル | 内容と理由 |
| --- | --- |
| `math-answer-diagnosis/src/config.py` | データ置き場、OCR サーバーのホスト、サービス管理、比較の閾値、`APP_PASSWORD` を設定と環境変数から読めるようにした。 |
| `math-answer-diagnosis/src/models.py` | 診断に設問ごとの得点を追加。問題に配点、教師採点、不一致ログのレコードを追加。 |
| `math-answer-diagnosis/src/storage.py` | 教師採点と不一致ログの保存。JSONL の書き換えは一時ファイルから置き換え、Drive 書き込み中の切断でファイルが欠けないようにした。 |
| `math-answer-diagnosis/src/sources.py` | `data/...` を Drive 上のデータ置き場基準で解決する。Mac で保存した絶対パスも、同名ファイルが Drive にあればそちらを使う。 |
| `math-answer-diagnosis/src/ocr_runner.py` | 画像パスの解決を上のルールに合わせ、GLM-OCR のエラー文に Colab のサーバーを書いた。 |
| `math-answer-diagnosis/src/adapters/ocr_adapter.py` | SDK へ Colab のホストとレイアウトデバイスを渡す。数式の置換を後からかけられるようにし、一括処理で GLM-OCR を卸してから UniMERNet を載せられるようにした。 |
| `math-answer-diagnosis/src/adapters/diagnosis_adapter.py` | `colab_cuda` を追加（中身は OpenAI 互換の同じクライアント）。診断の前に LLM サーバーを起動する。プロンプトに配点を渡す。 |
| `math-answer-diagnosis/src/adapters/diagnosis_helpers.py` | `question_scores` をレコードに残す。 |
| `math-answer-diagnosis/src/adapters/lmstudio_client.py` | 待ち時間の既定を 600 秒にした。27B は1件に数分かかることがある。 |
| `math-answer-diagnosis/src/adapters/mock_adapters.py` | 配点があるときは mock も設問ごとの点を返す。画面とテストの確認用。 |
| `math-answer-diagnosis/src/adapters/base.py` | 再チェック用の追加指示を診断コンテキストに足した。 |
| `math-answer-diagnosis/src/prompts.py` | 設問ごとの得点を JSON に求める。`recheck` プロンプトを追加。OCR 結果を推測で書き換えないよう指示している。 |
| `math-answer-diagnosis/src/__main__.py` | `gpu-profile` / `services` / `batch` / `import-teacher` / `compare` を追加。 |
| `math-answer-diagnosis/src/ui_common.py` | トンネル公開用のパスワード確認。 |
| `math-answer-diagnosis/scripts/start_unimernet.sh` | Python と重みの場所を環境変数で指定できるようにした。Colab の別環境から起動するため。 |
| `math-answer-diagnosis/app.py` | データ置き場、実行モード、教師採点数、サーバー状態を表示。 |
| `math-answer-diagnosis/pages/0_set_register.py` から `pages/6_diagnosis_viewer.py` | パスワード確認を追加。OCR の説明、診断 provider に `colab_cuda`、診断結果に設問ごとの点を追加。 |
| `math-answer-diagnosis/.env.example` | Colab で使う環境変数の例をコメントで追加。Mac の既定値は変えていない。 |
| `math-answer-diagnosis/.gitignore` | サーバーの PID とログ（`.run/`）を除外。 |
| `math-answer-diagnosis/README.md` | Colab の使い方、モデルの対応表、教師比較のコマンド。 |
| `math-answer-diagnosis/CONTINUE.md` | 再開時に Colab 経路があることを1節追加。 |
| `math-answer-diagnosis/.cursor/rules/resume.mdc` | 同上の1行。 |
| `AGENTS.md` | クラウド上では GPU が無いことと、テストの場所を追記。 |

## 追記: Colab の A100 で UniMERNet 環境が作れなかった件

`bash scripts/colab_setup.sh` が `.venv-unimernet` の作成で止まった。Colab の Python には `ensurepip` が無く、`python3 -m venv` が `ensurepip --upgrade` で失敗する。失敗した環境には `pip` が無いので、続く `python -m pip install` も `No module named pip` になる。

`scripts/colab_setup.sh` は UniMERNet 環境を `python3 -m uv venv` で作り、パッケージは `python3 -m uv pip install --python .venv-unimernet/bin/python` で入れる。仮想環境の中の pip は使わない。前回の失敗で `.venv-unimernet/bin/python` だけ残っている場合も、その Python に対して `uv pip` が入れる。

## 追記: ノートが既定ブランチを取り、uv が確認で止まった件

A100 でノートの clone セルを実行すると、既定ブランチ `2026-05-03-aho1` が取られ、`scripts/colab_setup.sh` が無く `No such file or directory` になった。clone セルは `BRANCH` を checkout する。既定はマージ前の `cursor/colab-cuda-teacher-compare-77a2`。`REPO_BRANCH` でも変えられる。ディレクトリが既にあるときは `git fetch` してそのブランチへ切り替える。

`.venv-unimernet` が既にあると `uv venv` が置き換え確認で入力待ちになる。Python があるときは `--allow-existing`、無いときは `--clear --force` にして確認を出さない。

このあと A100-SXM4-40GB ではセットアップが通り、`gpu-profile` が `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` を選び、cloudflared のトンネルも出た。

## 追記: PR #8（資料登録と自動タグ付け）を取り込んだ件

ベース `2026-05-03-aho1` に PR #8 がマージされたので、このブランチへ merge した（rebase や force-push はしていない）。

衝突したファイルと、残した判断:

| ファイル | どう合わせたか |
| --- | --- |
| `.env.example` | Colab 用のコメントと、`TAGGING_MODEL_NAME` のコメントを両方残した。Colab ではタグ付け用の別名を空にし、載せる重みは診断モデルの1つだけにする。 |
| `.gitignore` | `.streamlit/config.toml` は #8 のテーマなので追跡する。それ以外の `.streamlit/` と、サーバーの `.run/` は無視する。 |
| `app.py` | #8 のサイドバーと「資料を登録」を起動画面にした。公開トンネル用のパスワード、データ置き場の表示、GPU を1つずつ載せる注記、教師採点のページを足した。以前の件数ダッシュボードは戻していない。 |
| `pages/0_set_register.py` | #8 の自動タグ付け画面を採用し、パスワード確認だけ足した。パスワード未設定なら画面はそのまま開く。 |
| `src/config.py` | データ置き場・サービス管理・比較・パスワードに、`tagging_model_name` を足した。 |
| `src/models.py` | 問題に配点（`score_allocation`）とタグ付け記録（`tagging_metadata`）の両方を持たせた。 |

`src/intake.py` は、呼び出し側がクライアントを渡さないとき、会話の前に診断 LLM を起動する。Colab の `exclusive_gpu` では、その前の GLM-OCR を卸してから Qwen を載せる。テストがクライアントを渡すとき、および Mac の `external` ではサーバーを起動しない。タグ付けのモデル名は診断モデルと同じにする。サーバーは要求された名前を無視して、起動時の `LLM_MODEL_ID` で答える。

## 実機で確認していないこと

A100 ではセットアップ、GPU 判定、トンネル表示まで確認した。次は未確認。

- A100 / L4 / T4 での 27B 4bit と GLM-OCR の実際の読み込み、VRAM 使用量、生成速度
- Google Drive への書き込み
- UniMERNet の数式認識そのもの
- bitsandbytes の NF4 が、その Colab の CUDA 版 torch で動くこと
