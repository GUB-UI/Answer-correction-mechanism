# AGENTS.md

## Cursor Cloud specific instructions

### What this is
`answer_correction` is a single Python CLI/library (no monorepo, no web server, no database). It sends an answer-sheet image to an OpenAI-compatible LLM backend (LM Studio) for OCR, then sends the OCR text to an LLM for structured grading, printing JSON to stdout. Source: `answer_correction/` (`__main__.py`, `pipeline.py`). Run with `python3 -m answer_correction <image>`.

### Environment
- Python 3.12 with deps from `requirements.txt` (`httpx`, `Pillow`) installed via the startup update script. The compiled `.pyc` files in the repo are tagged cpython-314, but the source runs fine on 3.12.
- There is **no test suite, no linter config, and no build step**. "Lint/build" is just a syntax check: `python3 -m py_compile answer_correction/*.py`.

### Running / testing end-to-end (important gotcha)
- The tool is a client that POSTs to `{base_url}/chat/completions` (default `http://localhost:1234/v1`). The cloud VM has **no GPU-backed LLM / LM Studio**, so the default run cannot reach a real backend.
- To exercise the pipeline end-to-end without LM Studio, run any OpenAI-compatible mock that handles `POST /v1/chat/completions` on port 1234 and point the CLI at it (`--base-url http://127.0.0.1:1234/v1`). The pipeline calls OCR first (request contains an `image_url` content part) and grading second (request asks for a JSON object); a mock can branch on the presence of an `image_url` part to return OCR text vs. a JSON grade object.
- All config is via CLI flags or env vars (`LM_STUDIO_BASE_URL`, `LM_STUDIO_OCR_MODEL`, `LM_STUDIO_LLM_MODEL`, `LM_STUDIO_IMAGE_URL_FORMAT`, `LM_STUDIO_MAX_IMAGE_EDGE`). There is no `.env` file. Request timeout is hardcoded to 600s.
