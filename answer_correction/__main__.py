import argparse
import json
import sys

from answer_correction.pipeline import PipelineConfig, process_answer_image


def main() -> None:
    p = argparse.ArgumentParser(
        description="解答画像を LM Studio API で OCR → LLM し、JSON で標準出力に出す。"
    )
    p.add_argument("image", type=str, help="解答画像ファイル（JPEG / PNG / WebP）")
    p.add_argument(
        "--base-url",
        default=None,
        help="LM Studio の OpenAI 互換ベース URL（既定: LM_STUDIO_BASE_URL または http://localhost:1234/v1）",
    )
    p.add_argument("--ocr-model", default=None, help="OCR に使うモデル ID（環境 LM_STUDIO_OCR_MODEL と同義）")
    p.add_argument("--llm-model", default=None, help="構造化 JSON 用モデル ID（環境 LM_STUDIO_LLM_MODEL）")
    p.add_argument(
        "--image-url-format",
        choices=("raw_base64", "data_uri"),
        default=None,
        help="vision 入力の URL 形式。LM Studio によっては raw_base64 が安定（既定: 環境または raw_base64）",
    )
    p.add_argument(
        "--max-image-edge",
        type=int,
        default=None,
        help="長辺をこのピクセル以下に縮小（巨大画像時のワークアラウンド）。省略時は LM_STUDIO_MAX_IMAGE_EDGE を参照。",
    )
    args = p.parse_args()

    cfg = PipelineConfig(
        base_url=args.base_url,
        ocr_model=args.ocr_model,
        llm_model=args.llm_model,
        image_url_format=args.image_url_format,
        max_image_edge=args.max_image_edge,
    )

    try:
        out = process_answer_image(args.image, config=cfg)
    except Exception as e:
        err = {"ok": False, "error": {"type": type(e).__name__, "message": str(e)}}
        json.dump(err, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        raise SystemExit(1) from e

    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
