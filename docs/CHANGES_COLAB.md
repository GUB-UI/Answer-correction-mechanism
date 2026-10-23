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


## 追記: PR #10 再レビューの残件修正（2026-10-23）

部分修正だった指摘 2・4・5・6・9・10・12・13、新規 MAJOR 3件と MINOR 1件を修正。状態名は `validated` / `review_required` / `failed` のまま。欠落した項目点の補完や特定の数式・得点の分岐は追加していない。`math-answer-diagnosis/` は変更していない。

Gemma は `free_tier_project=True` という呼出側の明示的な確認を必須にした。無料枠のプロジェクトという申告を実行記録に残すもので、ネットワークで課金設定を照会・検証したという意味ではない。AI Studio のエンドポイントと `vertexai=False` を固定し、429 では即停止する。

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/evaluation/independent.py` | Scrubs generic/total/reference/AI score assignments, attributed prose, quoted ratios/units, and Japanese scoring annotations. Requires complete scalar values so symbolic expressions survive. Keeps unchanged embedded mathematical JSON verbatim and excludes non-scalar rubric text fields. Finding 2: independent reviewers must not receive earlier scores, while intervals, fractions, geometric point counts and symbolic expressions stay intact. |
| `harness_v2/storage/secrets.py` | Keeps unchanged embedded JSON text verbatim during public redaction. Finding 13: redaction must not gratuitously reformat mathematical text in prompt/handoff artifacts. |
| `harness_v2/grading/rubric.py` | Rejects malformed rows, blank/duplicate IDs, boolean/nonfinite/negative allocations and a nonfinite allocation sum. Finding 6: invalid rubric data must fail before allowlisting or inference. |
| `harness_v2/grading/scorer.py` | Raises on malformed response rows or missing/invalid IDs. Checks every supplied `max_points` and `max_score`, including null, boolean, string and nonfinite values; rubric allocations stay authoritative. Findings 5/6: bad model rows/maxima cannot disappear or validate. Missing item scores are never filled. |
| `harness_v2/grading/validator.py` | Independently validates direct rubric objects and avoids returning invalid maxima from failed rubric checks. Finding 6: callers bypassing the JSON loader must fail closed too. |
| `harness_v2/providers/gemma_ai_studio.py` | Requires the literal config flag `free_tier_project=True` before SDK construction, pins `https://generativelanguage.googleapis.com` and `vertexai=False`, records `provider_policy` on successful/error/quota exchanges, and keeps that metadata out of generation config. Replaces opaque client injection with a factory receiving the pinned options. Finding 4: the caller explicitly attests that the key is for a free-tier AI Studio project. This is a local config attestation, not a remotely verified billing claim. No tier-verification call, alternative endpoint, paid fallback or retry on 429. |
| `harness_v2/providers/codex_app_server.py` | Appends trusted grading instructions/schema after blinding; retains fresh threads. Filters fallback messages and completion notifications by thread and turn. Declines command/file approvals and returns JSON-RPC method-not-found for unsupported requests. Uses nonblocking byte framing/writes and a single absolute request/turn deadline, with bounded interruption and process cleanup on timeout. Findings 9/10 and new MAJOR 1: prevent foreign output contamination, invalid server replies, buffered-line false timeouts, partial-line/write hangs and instruction-free reviews. |
| `harness_v2/evaluation/review.py` | Defines the trusted review contract and accepts only completed, error-free, schema-valid reviews that pass grading validation and match each primary item score and total. Rejects nonfinite JSON, flags and unsupported response fields. New MAJOR 2: equal totals alone must not approve failed, flagged or item-disagreeing review. |
| `harness_v2/evaluation/schemas/grading_review.schema.json` | Closed JSON schema for independent items, evidence, reasons, uncertainty and nullable scores/total. New MAJOR 1/2: the prompt and acceptance check share an explicit response contract. |
| `harness_v2/core/pipeline.py` | Validates raw rubrics before blinding; records exact raw/derived grading artifact paths and SHA256 values; validates evidence against the text actually graded. Persists OCR engine/revision/dtype/elapsed time. Catches reviewer exceptions and saves their separate failed outcome with primary grading routed to `review_required`. Finding 13, new MAJOR 2/3 and new MINOR: preserve auditable inputs and completed grading even when review fails; retain CPU provenance. |
| `harness_v2/recognition/ocr.py` | Decodes imported and subprocess OCR from bytes without universal-newline translation. Finding 13: CRLF and other UTF-8 source bytes survive the private copy and grading path exactly. |
| `harness_v2/providers/grokbot_exchange.py` | Validates publication before redaction, rejects image-bearing content and unsupported fields, enforces finite rubric numbers, and blinds independent payload strings. Both file and GitHub handoffs use the same publication boundary. Finding 12: image keys, image data URIs (including non-base64), image markup/URLs and arbitrary nested payloads must not reach publishable files. |
| `harness_v2/providers/schemas/grokbot_request.schema.json` | Replaces unrestricted payload objects with declared text fields and a closed rubric-item schema. Finding 12: publication has a text-field allowlist instead of accepting arbitrary image-bearing metadata. |
| `harness_v2/requirements.txt` | Declares `jsonschema>=4`. Use one standard validator for the independent response contract; no legacy dependency changes. |
| `harness_v2/tests/test_rereview_independent.py` | Regressions for assignment/prose/Japanese/quoted score blinding, nested-field exclusion, untouched mathematical formatting, point counts and numeric-prefix symbolic math. Finding 2/13 and final-review counterexamples. |
| `harness_v2/tests/test_rereview_validation.py` | Parameterized regressions for both maximum aliases, malformed rows and invalid loaded/direct rubrics. Findings 5/6; no auto-filled scores. |
| `harness_v2/tests/test_rereview_gemma_policy.py` | Mock-only regressions for absent/non-boolean attestation, pinned SDK options despite environment overrides, 429 stopping and attestation persistence in SQLite. Finding 4, without network calls. |
| `harness_v2/tests/test_rereview_codex.py` | Regressions for actual blinded/trusted prompt, scoped output, request refusals, buffered/fragmented UTF-8/partial frames, absolute response/turn deadlines, blocked writes and timeout cleanup. Findings 9/10 and new MAJOR 1, including the final review's blocked-write counterexample. |
| `harness_v2/tests/test_rereview_pipeline.py` | Regressions for unsuccessful/malformed/flagged/item-disagreeing review, successful agreement, reviewer exceptions, exact CRLF grading, paired raw/derived hashes and persisted CPU metadata. Finding 13, all new MAJOR findings affecting acceptance/storage, and new MINOR. |
| `harness_v2/tests/test_rereview_grokbot.py` | Regressions reject image/unknown/nested fields in both delivery modes and verify independent publication blinding. Finding 12. |
| `harness_v2/tests/test_gemma.py` | Adapts SDK mocks to explicit attestation and factory construction, asserts pinned endpoint and preserves quota/retry tests. Finding 4's intentional setup API change remains covered offline. |
| `harness_v2/tests/test_codex.py` | Adds thread IDs to notification fixtures and expects an explicit decline response. Finding 9: exercise scoped protocol notifications and fail-closed approvals. |
| `harness_v2/tests/test_grokbot.py` | Uses embedded secret-bearing text within the publication allowlist and expects image-bearing requests to be rejected. Finding 12: update earlier tests to the stricter publication contract while retaining credential redaction checks. |
| `docs/superpowers/plans/2026-10-23-pr10-rereview.md` | Records the findings, constraints, implementation sequence and final validation. Keep the implementation/review rationale inspectable without git writes. |
| `commit_plan.json` | Groups every changed file exactly once into feature commits, with documentation last. Pre-existing untracked review/log files are excluded. User requested a reviewable commit plan, without committing or writing git state. |
| `docs/harness_v2/CHANGES.md`, `docs/CHANGES_COLAB.md` | Record each changed file, behavior, reason and validation result. Keep both existing change-log conventions complete. |

検証: `/tmp/pt2/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **311 passed**。新しい再レビューテスト6ファイルに **136 ケース**。最初の sandbox 実行は既存テストの localhost モックサーバー作成権限だけで失敗したため、外部接続・DNS を拒否する一時的な `PYTHONPATH=/tmp/pr10_offline` ガードを付け、localhost フィクスチャ用の権限で同じコマンドを実行した。外部 API・モデル取得・git 書き込みは実施していない。独立した read-only レビューで見つかった自然文の得点漏れ、数式の数値接頭辞破壊、stdin 書込みの停止も、失敗する回帰テストを先に追加して修正した。`git diff --check` も成功。

## harness v2: free-tier attestation optional (2026-10-23)

- `harness_v2/providers/gemma_ai_studio.py`: `free_tier_project` の明示オプトインを必須から任意に変更（ユーザー確認: AI Studioキーのプロジェクトは課金未設定）。`vertexai=False`、AI Studioエンドポイント固定、429で停止、有料フォールバックなしは維持。指定の有無を `provider_policy` に記録。
- `harness_v2/tests/test_rereview_gemma_policy.py`: 任意化と429停止の回帰テストに置換。
- 検証: `pytest -q harness_v2/tests math-answer-diagnosis/tests` → 308 passed。

## harness v2: Gemma リクエストのタイムアウト (2026-10-23)

- `harness_v2/providers/gemma_ai_studio.py`: AI Studio クライアントに `http_options.timeout=180000`（ms）を追加。Phase 2 実走で応答のない呼び出しがバッチを20分以上止めたため。エンドポイント固定・`vertexai=False`・429停止・フォールバックなしは維持。
- テスト: `test_rereview_gemma_policy.py`、`test_gemma.py` を更新・追加。`pytest -q harness_v2/tests math-answer-diagnosis/tests` → 309 passed。実機で1回呼び出し成功を確認。

## 追記: harness v2 の参照採点・比較・非公開データ保護の3修正（2026-10-23）

状態名は `validated` / `review_required` / `failed` を維持。参照得点は採点後の評価専用とし、参照との不一致だけでは状態や予測得点を変更しない。欠落得点の補完、特定の数式・得点の特例、`math-answer-diagnosis/` の編集、git 書き込みは行っていない。

| 修正 | ファイル | 何を、なぜ |
| --- | --- | --- |
| 1 | `harness_v2/core/pipeline.py` | 合計の `reference_score` と項目別の `reference_items` を採点・独立レビュー後に照合。`validated` でも不一致なら既存の FailureCase・提案・実行記録リンクを保存する。`independent=False` でもモデル入力の参照情報除去を必須にし、評価用ラベルによる採点への影響を防ぐ。 |
| 1 | `harness_v2/evaluation/independent.py` | 埋め込み JSON とラベル付きテキストから `reference_items` を除去。Python/YAML 風・不正形式の代入は残りのテキストを保守的に伏せ、値が漏れないようにする。元の OCR は非公開アーティファクトにそのまま残す。 |
| 1 | `harness_v2/tests/test_reference_evaluation.py` | 合計・項目別不一致、一致・参照なし、0点、未採点を検証。provider に渡るプロンプトと実際の Codex turn の両方に参照キー・値が無いこと、JSON/Python/YAML 形式でも漏れないことを確認する。 |
| 2 | `harness_v2/evaluation/comparison.py` | 結果読込・新旧双方の妥当性検証・ベンチマーク集計を共通化。旧形式の必須項目、正規フィールド、得点・根拠・合計と新形式の `validated` を確認し、項目別のレビュー理由・不確実性も保持する。不完全なケースも分母に残し、成功率の水増しを防ぐ。 |
| 2 | `scripts/compare_old_vs_new.py` | 終了コードだけでなく双方の保存結果を確認し、比較の成否・不完全な理由を記録。不完全・入力0件では CLI を非0終了にする。非公開テキストを含み得る summary も作成・再書込み時に権限を固定する。比較状態と採点状態は別に扱う。 |
| 2 | `harness_v2/tests/test_comparison_reliability.py` | 終了0でも結果なし・解析失敗・不正得点なら不成功になること、結果ファイル欠落・破損、別形式の別名による検証回避、旧形式必須項目欠落・項目別の要確認フラグ、空入力、集計・CLI 終了・summary 権限を検証する。 |
| 3 | `harness_v2/storage/permissions.py` | ディレクトリを `0o700`、ファイルを `0o600` に固定する共通処理を追加。既存ファイルも読込・切詰め・書込み前に fd で chmod し、呼出側の umask に依存しない。 |
| 3 | `harness_v2/storage/artifacts.py` | 全アーティファクトと親ディレクトリの作成・読込を保護。上書きを拒否した既存ファイルも chmod する。`ocr_private/`・`grading_private/`・raw 応答・任意名の非公開アーティファクトを、バイト列と不変性を保って保護する。 |
| 3 | `harness_v2/storage/sqlite_store.py` | DB を接続前に安全な権限で用意し、既存 DB・journal/WAL/SHM も修正。接続を明示的に閉じ、処理後も権限を確認する。新規 sidecar は DB の権限を継承させる。 |
| 3 | `harness_v2/recognition/ocr.py`, `harness_v2/recognition/run_ocr_cpu.py` | インポート元の生 OCR、生成 Markdown・統計・出力ディレクトリの権限を固定。アーティファクト保存前の OCR も公開しないため。 |
| 3 | `harness_v2/__main__.py`, `scripts/legacy_runner.py` | 新旧 CLI の結果と入力コピーを、既存ファイルへの再書込みも含めて保護。生の答案・モデル本文を含むため。 |
| 3 | `harness_v2/providers/grokbot_exchange.py` | request/result の作成・読込と handoff ファイル・ディレクトリの権限を固定。ファイル連携に残る答案も保護する。 |
| 3 | `harness_v2/tests/test_private_permissions.py` | `os.umask(0o000)` と `os.stat` で新規・既存アーティファクト、OCR、CLI・旧形式・連携ファイル、DB と稼働中の journal/WAL/SHM を検証。終了時に umask を復元する。CPU 出力はローカルのモデル代替で試し、重みを取得しない。 |
| 共通 | `docs/harness_v2/CHANGES.md`, `docs/CHANGES_COLAB.md`, `commit_plan.json` | 変更内容・理由・検証結果を記録し、修正1→2→3の3グループに整理。共有する両ドキュメントは最後のグループに置き、git 状態を変更せずレビュー可能にする。 |

検証: `/workspace/venv_h2/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **379 passed（12.27秒）**。既存アプリの localhost モックサーバー用に loopback ソケットの sandbox 権限を付けて実行した。最初の制限付き実行は345件成功、ソケット権限による失敗・エラー14件。外部接続・モデル取得は不要。read-only レビューで見つかった非 JSON の参照情報、空比較、形式の別名による検証回避も回帰テストを追加して修正した。`git diff --check` も成功。

## harness v2 Phase 3 / 01: 根拠照合と明示的な記載なし（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/core/contracts.py, harness_v2/grading/scorer.py, harness_v2/grading/validator.py` | 0点・真偽値true・理由付きの場合だけ空の根拠を許可。数式区切り・数式先頭の等号・空白を比較時だけ正規化し、生データを保持する。 |
| `harness_v2/evaluation/review.py, harness_v2/evaluation/schemas/grading_review.schema.json, harness_v2/tests/test_p3_evidence.py` | 独立レビューにも同じ契約を適用。型の偽装・非0点・捏造根拠の拒否とLaTeX差分の回帰テストを追加。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **394 passed in 12.69s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 02: ルート依存関係の不足を修正（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `requirements.txt` | google-genai・jsonschema・sympyを既存の範囲指定形式で追加。ルートからのインストールでもハーネスを動かせるようにする。 |
| `harness_v2/tests/test_p3_requirements.py` | harness_v2/requirements.txt の全依存と下限がルートに含まれることを検証し、Phase 2のセットアップ失敗を防ぐ。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **395 passed in 12.30s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 03: セットアップ失敗の記録と再実行（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/core/pipeline.py` | 例外発生時もfailed・failure_reason・例外クラス・成果物リンクを保存。同じrequest_idの再実行は試行番号付きパスを使い、不変成果物を上書きしない。429も停止前に1件だけ記録する。 |
| `harness_v2/tests/test_p3_setup.py` | キー未設定・準備失敗・返却前例外・孤立成果物・同一ID再試行・429の回帰テストを追加。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **400 passed in 12.58s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 04: 再試行を含む遅延と試行履歴（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/core/contracts.py, harness_v2/providers/gemma_ai_studio.py` | latency_msを再試行待機も含む合計時間に変更。各試行のlatency_ms・状態・例外クラス・既知のトークン数を保持する。 |
| `harness_v2/core/pipeline.py, harness_v2/tests/test_p3_latency.py` | 通常・失敗・429の記録に試行数と履歴を保存。偽時計で合計550msと各試行20/30msを検証し、最後の試行だけを記録する問題を防ぐ。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **403 passed in 12.15s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 05: 取り込みOCRの来歴（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/recognition/ocr.py` | <file>.meta.json・明示メタデータパス・ケースのocr_metadataを受付。engine・revision・dtype・source・timingを保持し、elapsed_sを正規化する。 |
| `harness_v2/core/pipeline.py, harness_v2/tests/test_p3_ocr_metadata.py` | 来歴をocr_metadataへ永続化。3通りの取り込み・優先順位・CRLFとハッシュの保持を回帰テストにし、Phase 2のCPU出力を監査可能にする。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **407 passed in 12.34s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 06: 保存済み実行からの参照不一致FailureCase（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/evaluation/reference.py, harness_v2/core/pipeline.py` | 既存の参照照合を共通化し、RunRecordとSQLite読込結果の両方からFailureCaseを生成。問題ID・入力版を記録し、予測得点と状態は変更しない。 |
| `harness_v2/tests/test_p3_reference.py` | validatedの保存済み実行・項目不一致・一致・0点・欠落点を検証。既存のGemma/Codex境界テストも実行し、参照の送信と得点補完を防ぐ。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **409 passed in 12.23s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 07: 版付き非公開データセット（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/evaluation/dataset.py` | dataset_id・versionと入力/基準/参照のSHA256を持つmanifestを追加。参照は別ファイルに分離し、パス逸脱・重複・不正参照を拒否。ファイル0600・ディレクトリ0700を保証する。 |
| `scripts/build_dev_dataset.py, scripts/dev_dataset_references.json` | 指定ローカル資料から手動C0/C1・手動・Xiaomi CPU fp32の7条件を作成。項目参照は指定された1-A/3-AのみJSONデータで保持し、実際の答案はgit外へ出力する。 |
| `harness_v2/tests/test_p3_dataset.py, docs/harness_v2/phase3_usage.md` | 一時fixtureでビルダー・ハッシュ改竄・CRLF・分離・権限を検証。実データでも/tmp/p3_dev_datasetへ7条件を構築し、APIは呼ばない。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **415 passed in 12.68s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 08: 繰返し実行CLIと再開（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/run.py` | config・dataset・repeats・outを受けるCLIを追加。IDを決定的に生成し、完了済みをスキップ、失敗試行を別成果物で再実行。429では状態・実行記録・FailureCaseを保存して即停止する。 |
| `harness_v2/providers/gemma_ai_studio.py, harness_v2/providers/fixture.py, harness_v2/core/pipeline.py, harness_v2/storage/sqlite_store.py` | モデル指定とseed等の生成設定を実呼出しへ渡し、試行来歴を保存。セットアップ例外もPipeline内で記録する。 |
| `harness_v2/tests/test_p3_runner.py, docs/harness_v2/phase3_usage.md` | fixtureでID・再開・429・構築例外・待機・CLI・設定伝播・参照分離を検証し、コマンドとJSON例を記載。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **420 passed in 13.13s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 09: BaselineとCandidateの比較条件（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/evaluation/comparison.py` | ケース/繰返しで対応付け、モデル・問題・入力ハッシュ・OCR来歴・基準版・生成設定等が異なる場合や結果欠落を理由付きで拒否。両側の成功率・遅延と差分を返す。 |
| `harness_v2/evaluation/compare.py, harness_v2/tests/test_p3_comparison.py, docs/harness_v2/phase3_usage.md` | 保存済みディレクトリから比較するCLIを追加。各拒否条件・同条件・CLIをテストし、manifest検証付きの実行例を記載。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **432 passed in 13.28s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 / 10: 評価指標と重大退行の判定（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/evaluation/metrics.py` | MAE・項目一致・根拠追跡率・誤加点/誤減点・要確認検出/誤警告・成功率・合計/試行別遅延・再試行を含む要求数/既知トークンを集計。欠落点は未採点、適用不能な率はnullとし、分母と不明使用量を表示する。 |
| `harness_v2/evaluation/comparison.py` | 両側の指標と差分を計算。新しいケース/繰返し/項目の誤採点、または根拠精度の低下はMAEが改善してもregressedとする。期待ケースと実行の来歴、生/採点入力ハッシュをmanifestと照合する。 |
| `harness_v2/core/pipeline.py, harness_v2/evaluation/independent.py` | 反復中の入力改変をモデル呼出し前に拒否し、失敗時も来歴を保持。最終確認で再現したreference_scores複数形の漏れも修正し、Gemma/Codex境界で参照を除去する。 |
| `harness_v2/tests/test_p3_metrics.py, harness_v2/tests/test_p3_comparison.py, harness_v2/tests/test_p3_runner.py, harness_v2/tests/test_p3_reference.py` | 数値fixtureで全指標・未採点・履歴使用量・誤警告・MAE改善に隠れた新規誤採点・入力改変・両側共通の不正来歴を検証。実際のCodex turnをローカル台本で検査し、外部呼出しは行わない。 |
| `docs/harness_v2/phase3_usage.md` | 分母・null・履歴・判定規則と07/08/09の再現コマンドを記載する。 |

検証: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **447 passed in 13.54s**。外部接続・DNSを拒否し、既存テストのlocalhostモックのみ許可。外部API呼出し・git書込みなし。

## harness v2 Phase 3 レビュー修正 / 01: 実装コミットの不変識別（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/run.py` | experiment/resultsにcommitを保存して実験identityに含める。保存済みcommitが異なる、または存在しない場合は更新・推論前に拒否し、`unknown`も完全一致で照合する。異なる実装の混在や既存記録の付替えを防ぐ。 |
| `harness_v2/tests/test_p3_runner.py` | 同一commitでの再開、異なるidentity、`unknown`との相互変更、拒否時のチェックポイント保持を検証する。異なる実装でのプロバイダ呼出しを防ぐ。 |

検証: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **452 passed in 13.48s**。PATHに`/workspace/venv_clean/bin`を使用。既存localhostモックのためサンドボックス権限を拡張し、git書込みなし。

## harness v2 Phase 3 レビュー修正 / 02: 中断後の実験記録の復元（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/core/pipeline.py, harness_v2/run.py` | 最初のSQLite保存時にexperiment_caseを含める。完了済み要求のスキップ前に欠落来歴を補修し、過去の試行も含めて不足するFailureCaseを復元。既存ファイルは保持し、中断による報告欠落や完了済み推論の再呼出しを防ぐ。 |
| `harness_v2/tests/test_p3_runner.py` | 行保存直後・失敗ファイル作成前の中断、旧形式の来歴欠落、完了済み得点の保持、過去の失敗試行、再開の冪等性と0600/0700を検証。SQLiteだけで復元できることを保証する。 |

検証: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **455 passed in 14.03s**。PATHに`/workspace/venv_clean/bin`を使用。既存localhostモックを許可し、git書込みなし。

## harness v2 Phase 3 レビュー修正 / 03: 実行済みOCR来歴の照合（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/recognition/ocr.py` | 自動・明示指定のメタデータファイルを取り込む際、元のバイト列からsidecar_sha256を算出し、答案のsource_sha256も記録。メタデータ結合後に確定し、申告値による上書きを防ぐ。答案が同じでもサイドカーの変更を追跡可能にする。 |
| `harness_v2/evaluation/comparison.py` | 各実行のengine・revision・dtype・source hashをmanifestと相手側の記録に照合。宣言済みsource/sidecar hashとrevision別名を扱い、来歴欠落・sidecar変更を拒否。経過時間は識別対象から除く。同じmanifestや両側の同一改竄に隠れる条件差を防ぐ。 |
| `harness_v2/tests/test_p3_comparison.py, harness_v2/tests/test_p3_ocr_metadata.py` | 片側/両側改変、項目欠落、revision別名、サイドカーのみの変更、CRLFの厳密ハッシュ、偽装値と非公開権限を回帰テストにする。 |

検証: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **473 passed in 14.33s**。PATHに`/workspace/venv_clean/bin`を使用。既存localhostモックを許可し、git書込みなし。

## harness v2 Phase 3 レビュー修正 / 04: 合計点だけが異なる場合の誤加点・誤減点（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/evaluation/metrics.py` | 既知の項目に誤りがなく合計不一致を__total__で記録する場合も、誤加点/誤減点の方向を集計。項目参照がない場合と同じ分岐にまとめ、合計誤りだけ方向の集計から漏れる問題を修正する。 |
| `harness_v2/tests/test_p3_metrics.py` | 全項目・部分項目の参照が一致する場合の両方向、既知の項目誤りがある場合の二重計上防止、入力得点の不変性を検証。未参照項目への誤り割当てや得点補完を行わず正しく集計する。 |

検証: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **479 passed in 14.09s**。PATHに`/workspace/venv_clean/bin`を使用。既存localhostモックを許可し、git書込みなし。

## harness v2 Phase 3 ライブ実行修正 / 05: 根拠照合の正規化を対称にする（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/grading/validator.py` | 数式区切り内の先頭`=`を削除する処理をやめ、区切り記号と空白の除去だけにする。答案側だけ`=`が消え、区切りのない引用（例 `= \sqrt{9}-\sqrt{2}`）が逐語でも照合失敗していた。Phase 3 ライブ実行で、正しく採点された1-A xiaomi_cpu_fp32の3件がこれだけでreview_requiredになっていた。比較用のみで保存テキストは変更しない。 |
| `harness_v2/tests/test_p3_evidence.py` | 先頭`=`の有無・区切りの有無の引用が一致する回帰テスト。既存の否定ケースは維持。 |

検証: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **482 passed in 14.16s**。

## harness v2 Phase 3 review fix / 06: 実装コミットの自動解決（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/run.py` | CLI・APIでcommitを省略すると、呼出し元の作業ディレクトリではなくharnessソースのHEADを取得し、追跡対象の変更には`+dirty`を付ける。解決不能・空文字・`unknown`は保存やプロバイダ呼出し前に拒否し、明示指定ならGitなしでも実行可能。既定値や旧`unknown`記録による異なる実装の混在を防ぐ。 |
| `harness_v2/tests/conftest.py, harness_v2/tests/test_p3_commit_identity.py, harness_v2/tests/test_p3_runner.py` | 解決処理をモックして実リポジトリ状態への依存を排除。クリーン/変更済みHEAD、Git失敗、CLI/API既定値、明示指定、旧記録の拒否と再開拒否時のファイル不変性を検証する。参照分離は維持する。 |

検証: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **506 passed in 14.31s**。PATHに`/workspace/venv_clean/bin`を使用。既存localhostモックのためサンドボックス権限を拡張し、git書込みなし。

## harness v2 Phase 3 review fix / 07: 失敗したモデル呼出しの生成設定も検証（2026-10-23）

| ファイル | 何を、なぜ |
| --- | --- |
| `harness_v2/evaluation/comparison.py` | 現在の実行と過去の試行の実効生成設定を、failedを含めて宣言値と照合。明示的なセットアップ例外で試行数0・応答記録なしの場合だけ除外し、設定の欠落・不正形式・呼出し有無の不明な記録は拒否する。provider_policyは引き続き生成パラメータから除外。失敗や再試行を含む指標の条件混在を防ぐ。 |
| `harness_v2/tests/test_p3_failed_settings.py` | JSON解析・HTTP・quota失敗、片側/両側の不一致、設定欠落/不正、再開成功後の失敗履歴、呼出し前の設定失敗、不明な試行数と応答記録の矛盾を検証。比較前後の記録と未採点を保持し、得点補完や参照送信を防ぐ。 |

検証: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` → **542 passed in 15.33s**。PATHに`/workspace/venv_clean/bin`を使用。既存localhostモックのためサンドボックス権限を拡張し、git書込みなし。
