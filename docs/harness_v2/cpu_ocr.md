# Xiaomi-OCR-0 on a CPU-only box

Target machine: Linux, 8 vCPU, about 15 GB RAM, no GPU.

`SeerRay-Lab/Xiaomi-OCR-0` is a 0.8B model. fp16 weights are about 1.6 GB, so the weights themselves can fit in 15 GB. This harness did not download the model and did not time a page. A transformers runtime plus a full handwritten page on 8 CPU cores is not treated as a reasonable step (expect minutes per page; that latency is unmeasured here).

Phase 1 therefore does not launch the model.

- `XiaomiOCR.import_file` reads a UTF-8 text file produced by a separate OCR process and stores it once under `ocr/<id>.txt`. A second write of the same id fails.
- Image files are rejected. The harness does not copy student images into the artifact store.
- `XiaomiOCR.run_cpu` raises `OCRNotRunnable` with this requirement, so a later CPU runner can replace the method without changing callers.

To produce a file for import, run Xiaomi in its own process (the existing `scripts/xiaomi_ocr_server.py` path, or a CPU transformers process) and save the raw text. Do not rewrite that file after import.
