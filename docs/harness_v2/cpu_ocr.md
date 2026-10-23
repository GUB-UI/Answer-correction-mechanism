# Xiaomi-OCR-0 on CPU

`XiaomiOCR.import_file` remains the path that does not download a model. It stores the file bytes exactly once under `ocr_private/<id>.txt` (`O_CREAT|O_EXCL`) and writes a redacted sibling at `ocr/<id>.txt`. Grading reads the private file and checks its sha256.

`XiaomiOCR.run_cpu` launches `python -m harness_v2.recognition.run_ocr_cpu` when the caller asks for CPU OCR:

- Model `SeerRay-Lab/Xiaomi-OCR-0`, revision `ff990be318c4c68f7e621dd690231200f47e48c4`.
- dtype `float32` only. `bfloat16` is refused because it breaks the transcription.
- `enable_thinking` is false, `do_sample` is false, `max_new_tokens` is 4096, and `max_edge` is 0 (no resize).
- Free memory must be at least about 4.5 GiB before the process is started.
- The stored artifact records revision, dtype, and elapsed seconds.

Unit tests inject the memory check and a transcript runner. They do not import torch and do not download the model. A host without that free memory still gets `OCRNotRunnable`.
