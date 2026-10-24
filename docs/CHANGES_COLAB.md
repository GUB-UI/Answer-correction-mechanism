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

## 追記: A100 で答案まで流したときに出た不具合

commit `656259d` の CLI 実行で分かったことと、このブランチでの直し。

| 現象 | 直し |
| --- | --- |
| `colab_setup.sh` が `.env.example` をコピーし、`OCR_MODEL_NAME=mlx-community/GLM-OCR-bf16` と `DIAGNOSIS_PROVIDER=omlx` が `config.colab.yaml` を上書きした。読み込まれた設定は `glmocr_sdk` / mlx / `omlx` だった。 | セットアップ、`gpu-profile --write-env`、ノートの `.env` 更新が `zai-org/GLM-OCR` と `colab_cuda` を書く。 |
| GLM-OCR の既定 8080 は Colab の node が使用中で `Errno 98`。8081 なら起動した。 | Colab の既定を 8081 にした。managed 起動時にそのポートが使用中なら、次の空きへずらして SDK も同じポートを見る。 |
| `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` を bfloat16 で読むと、チェックポイントの計算は float16、`lm_head` は非量子化で、全リクエストが HTTP 500（`Half != BFloat16`）。`--dtype float16` では動いた。 | 事前量子化モデルは、指定が auto のときチェックポイントの compute dtype を使う。無ければ float16。オンロードの NF4（公式 27B）は従来どおり GPU の dtype。 |
| 3件とも `raw_text == used_text` で UniMERNet の効果が見えなかった。 | 人手修正前は二つの欄は同じ文字列にする実装だった。効果は `formula_summary` と `glm_markdown` で見る。レイアウトに数式領域が無い、領域自体が無い、も `skipped` に残す。日本語交じりの行は文全体を置き換えず、`$...$` だけ読み直す。 |
| `question_scores` の question が設問番号ではなく「全体」になることがあった。 | 「全体」「合計」は捨てる。配点があるときはその設問 ID に揃うものだけ残す。プロンプトでも全体を設問にしないよう書いた。 |
| Qwen 起動時に flash-linear-attention のフォールバック警告。 | `colab_setup.sh` は wheel があるときだけ `flash-linear-attention` と `causal-conv1d` を入れる。wheel が無ければセットアップは止めない。この環境では未導入のまま進む。 |

手書きの数学ノート写真では、GLM-OCR が崩れた LaTeX や中国語（例: 最小值）、`x=-1` を `-17''` のように読むことがあった。原因を一つのプロンプト修正で直したとは言えない。拡大・縮小とプロンプトの追加は `GLMOCR_MIN_EDGE`、`GLMOCR_MAX_EDGE`、`GLMOCR_PROMPT_SUFFIX` で試せる。既定はオフ。

A100 40GB では Qwen 27B の 4bit が約 20GiB、ピーク 22.3GB、答案1件あたり約 45〜50 秒だった。

## 追記: 手書き OCR を Xiaomi-OCR-0 にし、A100 掃引の5件を直した

A100 掃引（commit `3618410`、UniMERNet オンは本文が壊れ、オフでも GLM の読みは採点に足りない）のあと。呼称「xiaomi ocr 0」に一致する Hugging Face のモデルは `SeerRay-Lab/Xiaomi-OCR-0`（Apache-2.0、Qwen3.5-0.8B ベース、2026-09 公開）。MiMo-VL は別の汎用 VLM なので使っていない。この環境に GPU は無く、手書き答案での精度は測っていない。

公式の文書プロンプトは英語で、本文を Markdown、数式を LaTeX、表を OTSL にする。採点は Markdown を読むので、プロンプトは日本語にし、表は Markdown、数式は LaTeX、日本語を中国語に翻訳しない、と指定した。手書きは公式どおりページ全体を渡し、領域では切らない。0.8B は Qwen 27B 4bit（約 20〜22GB）と同時には載せず、これまでどおりプロセスを1つずつ切り替える。

| ファイル | 何を、なぜ |
| --- | --- |
| `scripts/xiaomi_ocr_server.py` | `SeerRay-Lab/Xiaomi-OCR-0` の OpenAI 互換サーバー。ページ全体を greedy で読む。拡大は `XIAOMI_OCR_MIN_EDGE` が無い限りしない。 |
| `src/runtime/colab_env.py`、`config.colab.yaml` | Colab の既定を `OCR_PROVIDER=xiaomi_ocr`、`OCR_MODEL_NAME=SeerRay-Lab/Xiaomi-OCR-0`、`UNIMERNET_ENABLED=false` にした。GLM-OCR は `glmocr_sdk` のまま残す。 |
| `src/config.py`、`src/runtime/services.py` | `xiaomi_ocr` サービスを追加。`exclusive_gpu` では他の GPU プロセスを止めてから起動する。ポートは GLM と同じ 8081（使用中なら次の空き）。 |
| `src/adapters/ocr_adapter.py`、`src/prompts.py` | `XiaomiOCRAdapter`。日本語プロンプトで Markdown / LaTeX を受け取る。 |
| `src/batch_pipeline.py`、`pages/3_ocr_run.py`、`src/__main__.py` | 一括処理・画面・`services` から Xiaomi を選べるようにした。 |
| `requirements-colab.txt` | transformers の下限を 5.17 にした。Xiaomi は `Qwen3_5ForConditionalGeneration` が要る。torch は入れ直さない。 |
| `src/adapters/formula_merge.py` | 複数行や縦に長い領域は UniMERNet に渡さない。出力が元の式より長すぎる、短い、新しいギリシャ文字、複数行の array なら捨てて元の OCR を残す。 |
| `src/adapters/ocr_adapter.py` の `refine_ocr_result_with_unimernet` | 置換後に `formula_summary` を数え直す。以前は `replaced: 0` のままだった。 |
| `src/adapters/ocr_quality.py`、`src/ocr_runner.py` | 中国語の定型（最小值、当且仅当、解得 など）、同じ行の繰り返し、UniMERNet との大きな不一致で `ocr_suspect` を立てる。画像をまとめるときにフラグを消さない。 |
| `src/adapters/diagnosis_helpers.py`、`src/prompts.py`、`src/models.py` | `question_scores` が空なら配点から行を作る。満点なら各設問を満点、途中点なら配点比。`possible_ocr_issue` で合計が 0 かつ設問点も無いときは 0 点で確定せず、`review_status=needs_teacher_review`。設問に部分点があるのに合計が 0 のときは部分点の合計を使う。 |

GLM-OCR に戻す環境変数は `OCR_PROVIDER=glmocr_sdk` と `OCR_MODEL_NAME=zai-org/GLM-OCR`。UniMERNet を試すときは `UNIMERNET_ENABLED=true`。

## 追記: 採点 JSON の外側フェンスを1組だけ外す

v1.2 では Qwen3.8-27B-NVFP4（vLLM）の採点出力が、ときどき全体を ```json で囲んで厳密な `json.loads` に失敗した。P0 と P1 の文面は変えていない。vLLM の guided JSON は点数そのものが変わることがあったので、既定にはしていない。

| ファイル | 何を、なぜ |
| --- | --- |
| `src/grading/parse.py`、`src/grading/schema.json` | まずそのまま JSON として読む。失敗したときだけ、外側の ``` または ```json を1組外して読み直す。中の文字は変えない。囲みを外したことは `fence_stripped` に残し、それだけでは教師確認に回さない。読めない出力は `GradingOutputError` で、通常の採点結果にしない。読めたあとはスキーマ、配点があるときの項目数、点数の範囲、項目合計と `score` の一致を検証する。 |
| `src/adapters/diagnosis_helpers.py` | 採点の読み取りをこの手続きにした。`rubric_scores` があるとき、検証結果を `grading_validation` に残す。配点の ID が項目 ID と重ならないときは、設問番号とルーブリック ID を取り違えて欠落扱いにしない。 |
| `tests/test_grading_parse.py` | フェンス1組、中にフェンス文字があるだけの JSON、前後に文章がある場合、項目数と合計の検証。 |

## 追記: 要確認フラグは記録だけ（振り分けは既定でオフ）

v1.2 ではモデル自身の `needs_review` が 135 回とも false だった。試作の規則は 79/135 を立て、その多くは OCR の疑いが粗すぎた（Markdown の囲みや式の外の `\pi` まで拾っていた）。今回の規則は記録用で、`REVIEW_FLAGS_ROUTE` を付けない限り、どの答案を教師確認へ回すかは変えない。参照得点は使わない。

立つ条件は、JSON が読めない、項目が足りない・多い、点数と理由が矛盾する、数式として読めない印（式の中の「解得」「最小值」、`17°`、`''`、壊れた根号 `√‾`、`[不明]`）、根拠の式が入力に無い、である。根拠は LaTeX と空白と省略記号の差を吸収する。言い換えだけの文は根拠不一致にしない。フェンスを外したことだけでは立てない。

| ファイル | 何を、なぜ |
| --- | --- |
| `src/grading/review_flags.py` | `needs_review`、`grading_uncertain`、`ocr_uncertain` と、それぞれの理由を返す。答案テキストが無いときは OCR と根拠は判定しない。 |
| `src/adapters/diagnosis_helpers.py` | 診断結果の `raw_output.review_flags` に保存する。`REVIEW_FLAGS_ROUTE=true` のときだけ `review_status` を `needs_teacher_review` にする。点数は変えない。 |
| `tests/test_review_flags.py` | フェンスだけでは立たないこと、項目不足、点数と理由の矛盾、LaTeX のゆれ、言い換え、式の誤補正、振り分けが既定でオフであること。 |

## 追記: v1.2 の135件でフラグの偽陽性を数える

試作は 79/135 を立て、その多くは OCR の疑いだった。新しい規則を同じ135件にかけ、引き金ごとの件数を出す。答案本文は添付に無いので、OCR と根拠の再判定はしていない。参照得点は使わない。guided JSON は入れてない。

| ファイル | 何を、なぜ |
| --- | --- |
| `scripts/review_flag_study.py` | CSV と生の採点出力に、採点の読み取りと要確認フラグをかける。引き金の件数と例、CSV に残っていた試作の内訳を JSON で出す。 |
| `tests/test_review_flag_study.py` | フェンスを外した合計不一致を数え、答案が無いときは OCR 疑いを立てないこと。試作の列は別枠で数えること。 |

この環境で `results_v1.2.csv`（135行）と生出力93件（`result_from` で再利用した6件を含め、本文があるのは99件）にかけた結果:

- 新しい規則の `needs_review` は 0。引き金は無し。
- フェンスを外して読めたのは 4件。4件とも、それだけでは要確認にならない。
- 生の本文が無く CSV から組み立てたのは 36件。そのうちフェンス行で読み直せなかったのは 2件（`S3_C_P1_3-A_s23`、`S3_C_P1_3-A_s47`）。
- モデル自身の `needs_review` は 0、`grading_uncertain` は 0、`ocr_uncertain` は 39。モデルの `ocr_uncertain` だけでは要確認にしない。
- CSV に残っていた試作は `needs_review` 79。内訳は `key_formula_possibly_unreadable` 72、`evidence_not_in_input` 14、`json_fence`（コード囲み）6。OCR の印は markdown のコード囲み 42、`\pi` 27、偶数 15、解得 15、`17''` 12、`\in I` 12、`a_k` 12、或 12。

## 実機で確認していないこと

A100 ではセットアップ、GPU 判定、トンネル、27B 4bit の読み込みと採点時間まで確認した。次は未確認。

- 今回の dtype・ポート・`.env` 修正を入れたあとの再実行
- L4 / T4 での読み込み
- Google Drive への書き込み
- UniMERNet が手書きの数式を実際に良く読めるか
- `GLMOCR_MIN_EDGE` などを変えたときの読み取り品質
- `SeerRay-Lab/Xiaomi-OCR-0` を A100 に載せたときの VRAM と、この手書き答案での読み取り精度

## PR #9 修正1: OCR 再利用の条件と手動転記の分離

OCR 結果にエンジン、モデル ID・revision、OCR 設定、入力画像ごとの SHA-256（ページ順）を保存し、現在の実行とすべて一致するときだけ再利用する。情報のない旧レコード、画像の変更・欠落、設定の変更では再利用しない。数式置換が失敗した結果を、置換済みとして再利用しない。`OCR_MODEL_REVISION` は managed サーバーの processor・重みの読み込みにも渡す。

人の修正は元の OCR を残して別レコードの `manual_transcription` として保存する。旧 `human_corrected` レコードもこの入力型として読み込む。一括処理で手動転記を使うときは `--input-type manual_transcription`、テキスト答案は `--input-type typed_text` を明示する。OCR キャッシュとしては扱わない。

検証: `tests/test_ocr_reuse.py` に一致時の再利用、各条件の不一致、複数画像の順序・枚数、旧レコード、手動転記・テキスト入力の分離、数式置換失敗、revision の受け渡しを追加。全テストを指定の Python・pytest コマンドで実行し、147 件成功（既存の HTTP mock 用に localhost 接続を許可）。

## PR #9 修正2: 設問点の自動補完を廃止

合計点が満点・途中点・0点のどの場合も、配点や配点比から設問点を計算しない。モデルが省略した設問、null・無効な設問点は `None` のまま残し、設問と診断を `review_required` にする。この確認は `REVIEW_FLAGS_ROUTE` がオフでも必須。明示された 0 点や部分点は変えず、ルーブリックに実際に出力された同じ ID の点だけを正規化する。mock も配点から設問点を生成しない。合計点も設問点から書き換えず、不一致を確認理由に残す。

診断・一括処理の状態名を `validated` / `review_required` / `failed` に統一し、旧 `needs_teacher_review` は読み込み時に変換する。画面・プロンプトも同じ名前を使う。設問点が未確定のまま保存・教師比較・再チェックできるようにした。

検証: `tests/test_no_score_autofill.py` に合計点と OCR 疑いの組合せ、部分的な出力、明示的な 0 点、null・欠落・無効値、配点なし、ルーブリック項目不足、mock、保存後の null、教師比較、一括再利用時の確認状態を追加。既存の満点補完・合計書き換えの期待値を撤回し、旧状態名のテストを更新。

全テストを指定の Python・pytest コマンドで実行し、180 件成功（既存の HTTP mock 用に localhost 接続を許可）。`docs/harness_v2/CHANGES.md` はこのブランチには存在しないため、新規作成していない。

## 修正1: 数式補正の完了を明示的に確認

UniMERNet の利用不可・予測失敗・例外・結果未報告を補正完了として保存しない。全ページの補正成功（数式領域なしの確認を含む）のみ完了の provenance を付け、未完了なら次回も OCR と補正を実行する。実際の skipped 戻り値と予測例外、呼び出し例外、成功・数式なし・無処理の回帰テストを追加。

## 修正2: 外部 OCR サーバーの実体を確認して再利用

external モードでは実際の OCR 接続先の `/health` から `ok: true`、ロード済み `model`、`revision`、空でない実効 `settings` を取得し、provenance に保存する。モデル・revision・設定が変われば再実行し、不明・不正・未起動・取得失敗の場合も OCR を再実行する。処理前後で実体が一致しない結果は再利用しない。この情報を返さないサーバーでも OCR 自体は利用可能。バックエンドのモックで両 OCR provider と別名、設定変更、情報不足・通信失敗・処理中変更・旧キャッシュを検証。

## 修正3: 設問別得点のない旧レコードを要確認にする

合計点が数値でも `question_scores` が空または未保存なら、読み込み時に `review_required` とする。空文字・未指定・validated の旧ステータスにも適用し、failed は維持する。合計点や設問別得点は補完・変更しない。旧 JSON の読み込みと一括処理の診断キャッシュ再利用、明示的な設問別得点がそろう場合の検証を追加。

## PR #9 修正4: 採点設定が完全一致する診断だけを再利用する

一括処理で新しい診断に `grading_metadata` を保存する。モデル ID、provider、実際のプロンプト全文の SHA-256（内容に基づくバージョン）、ルーブリック・配点・問題文・模範解答・問題 ID の SHA-256、生成設定がすべて一致する場合だけ再利用する。生成設定は API 呼び出しと共通化し、managed モードではサーバーのモデル・量子化・生成上限なども記録する。メタデータ全体または一部がない旧診断と、設定を特定できないアダプターの診断は再採点する。得点の補完や状態名の変更は行わない。

検証: `tests/test_diagnosis_reuse.py` に完全一致時の保存・再利用、モデル・プロンプト本文・テンプレート・種別・ルーブリック・配点・問題文・模範解答・生成設定の変更、旧診断と部分的なメタデータ、再実行指定、設定不明時の回帰テストを追加。新規テストは通信を禁止し、クライアントとサービス管理をモック化した。関連テストを `/workspace/venv_clean/bin/python -m pytest -q -p no:cacheprovider` で実行し、161 件成功。ネットワーク・API 通信は実施していない。`docs/harness_v2/CHANGES.md` はこの worktree に存在しないため追加していない。

## PR #9 再修正1・2: 実際の診断バックエンドを確認し、managed 設定変更で再起動

診断 API の実接続先の `/v1/health`（base URL に `/health` を追加）から、`ok: true` と `backend_fingerprint` のロード済み `model_id`・空でない実効 `settings` を確認する。モデルの別名や希望設定を実体の証明として使わず、確認できないサーバーでは毎回再採点する。内蔵 LLM サーバーはロード後の生成設定、生成上限、量子化、dtype、device、チャットテンプレート、取得可能な revision を返す。採点直前と直後で情報が変わった結果にも、再利用可能なメタデータを付けない。

managed のキャッシュは設定変更時に管理対象プロセスを停止して作り直す。PID と起動仕様のハッシュを保存し、別 manager からも不一致・旧形式・情報欠落を検出して再起動する。メタデータ取得は起動確認後に行い、希望モデル B の設定で動作中の A を B と記録しない。

検証: 同じ別名のまま重み・生成上限を変更、実体不明、採点中変更、managed 起動順序、希望設定と実体の相違、設定変更時の manager 再構築、保存済み起動仕様の照合、health の不正・不足・通信失敗をモックで回帰確認。指定 Python・pytest コマンドで関連 59 件成功。ソケット接続を禁止し、ネットワーク・API 通信やモデルのロードは実施していない。`docs/harness_v2/CHANGES.md` は存在しない。

## PR #9 再修正3: 不完全な診断メタデータ同士の一致を拒否

再利用前に現在・保存済みの両方で provider、モデル ID、プロンプトとルーブリックの SHA-256、生成設定の必須項目・型を検証する。API バックエンドの実体にもロード済み ID と実効設定を必須とし、クライアントを差し替えた場合も検証する。mock は明示した実装バージョンで識別する。不完全な情報同士が一致しても再採点し、不完全な情報を新しい結果の再利用可能なメタデータとして保存しない。

検証: 空辞書、provider だけ、ルーブリック・生成設定・実体情報の欠落、空の設定、不正なハッシュ・型・非有限値など、現在値と保存値が同じ不完全なメタデータの回帰テストを追加。指定 Python・pytest コマンドで関連 192 件成功。実 HTTP 接続を使う既存 3 件は選択から除外し、ネットワーク・API 通信は実施していない。`docs/harness_v2/CHANGES.md` は存在しない。

## PR #9 再修正4: クライアント追加後の送信メッセージをハッシュ化

クライアントのリクエスト構築を共通化し、assistant プレフィルを追加した最終メッセージの JSON をプロンプトの SHA-256 に使う。プレフィル変更で再採点し、同じ内容なら再利用する。最終メッセージを取得できないクライアントでは再利用しない。入力メッセージは変更せず、既存 assistant の末尾や思考無効化を使わない場合も送信内容と一致させる。

検証: 実クライアントの送信経路を SDK のメモリ内モックで動かし、送信メッセージと保存ハッシュの一致、プレフィル変更時の再採点、その後の再利用を確認。最終リクエスト不明時、既存 assistant と思考無効化の組合せも回帰確認。指定 Python・pytest コマンドで関連 200 件成功。さらに全体のオフライン検証で 313 件成功。ソケット接続を禁止する一時 pytest プラグインを使い、通信が必要な既存 4 件と HTTP サーバーを使う `test_intake_ui.py` は除外した。ネットワーク・API 通信は実施していない。`docs/harness_v2/CHANGES.md` は存在しない。

## レビュー修正 #1: 後処理設定と診断再利用

- `REVIEW_FLAGS_ROUTE` の有効値を採点メタデータに保存し、不一致・欠落・不正な型では再利用しない。
- 回帰テストで振り分け切替時の再採点と `review_required` への遷移を確認。

## レビュー修正 #2: 重み・サーバー実装の固定

- リモートモデルは revision、ローカルモデルは重み・設定・tokenizer ファイルの内容 SHA-256 を必須化。ローカルハッシュは起動時に取得し、ロード中の変更も検出する。
- サーバーと共有処理のソースハッシュ、および torch / transformers / bitsandbytes / tokenizers / accelerate の版を保存。欠落した fingerprint は再利用しない。
- 重み・設定の上書き、実装・ライブラリ変更、必須情報欠落の回帰テストを追加。
