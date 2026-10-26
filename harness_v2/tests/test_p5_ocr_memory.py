from harness_v2.recognition import run_ocr_cpu as ocr


def test_mem_available_is_used(tmp_path):
    info = tmp_path / "meminfo"
    info.write_text("MemTotal: 16000000 kB\nMemFree: 1000 kB\nMemAvailable: 7000000 kB\n")
    assert ocr.available_bytes(str(info)) == 7000000 * 1024


def test_falls_back_when_mem_available_missing(tmp_path, monkeypatch):
    info = tmp_path / "meminfo"
    info.write_text("MemTotal: 16000000 kB\n")
    monkeypatch.setattr(ocr.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 10)
    assert ocr.available_bytes(str(info)) == 40960


def test_missing_meminfo_file_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr.os, "sysconf", lambda name: 1)
    assert ocr.available_bytes(str(tmp_path / "absent")) == 1


def test_threshold_unchanged():
    assert ocr.MIN_FREE_BYTES == int(4.5 * 1024**3)
