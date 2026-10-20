from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import load_config
from src.storage import Storage
from src import ui_common


@pytest.fixture
def intake_app(tmp_path, monkeypatch):
    """An actual OpenAI-compatible local HTTP endpoint, with deterministic extraction."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.loads(request["messages"][1]["content"])
            docs = data["documents"]
            citations = {}
            for role, text in [("problem", "2x < 8 を解け。"),
                               ("student_answer", "x < 3"), ("model_answer", "x < 4")]:
                sid = next(doc["source_id"] for doc in docs if text in doc["text"])
                citations[role] = {"text": text, "source_ids": [sid]}
            group = {"title": "一次不等式", **citations, "existing_problem_id": None,
                     "confidence": 0.98, "review_reasons": []}
            content = json.dumps({"groups": [group]}, ensure_ascii=False)
            body = json.dumps({"id": "local-completion", "object": "chat.completion",
                               "created": 0, "model": "local-test",
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = load_config(tmp_path)
    config = replace(config, lmstudio=replace(config.lmstudio,
                      base_url=f"http://127.0.0.1:{server.server_port}/v1"))
    storage = Storage(config)
    monkeypatch.setattr(ui_common, "get_config", lambda: config)
    monkeypatch.setattr(ui_common, "get_storage", lambda: storage)
    app_path = Path(__file__).resolve().parents[1] / "pages" / "0_set_register.py"
    yield AppTest.from_file(str(app_path), default_timeout=15), storage
    server.shutdown()
    server.server_close()
    thread.join()


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_auto_input_goes_through_local_http_to_shared_tag_and_saved_answer(intake_app):
    app, storage = intake_app
    app.run()
    assert not app.exception
    app.text_area(key="intake_auto_text").set_value(
        "問題: 2x < 8 を解け。\n生徒答案: x < 3\n模範解答: x < 4")
    app.text_input(key="intake_student_id").set_value("student_001")
    button(app, "解析して自動タグ付け").click().run()
    assert not app.exception
    assert any("共通タグ" in c.value for c in app.caption)
    button(app, "確認した内容を保存").click().run()
    assert not app.exception
    problem = storage.list_problems()[0]
    answer = storage.list_answers()[0]
    assert problem.problem_id == answer.problem_id
    assert problem.correct_answer == "x < 4"
    assert storage.get_ocr_results_for_answer(answer.answer_id)[0].used_text == "x < 3"
    assert button(app, "確認した内容を保存").disabled


def test_changing_input_invalidates_previous_tagging_preview(intake_app):
    app, storage = intake_app
    app.run()
    app.text_area(key="intake_auto_text").set_value(
        "問題: 2x < 8 を解け。\n生徒答案: x < 3\n模範解答: x < 4")
    button(app, "解析して自動タグ付け").click().run()
    assert not app.exception
    app.text_area(key="intake_auto_text").set_value("別の資料に差し替えた").run()
    assert not app.exception
    assert not any(b.label == "確認した内容を保存" for b in app.button)
    assert storage.list_problems() == []


def test_save_without_student_id_has_no_partial_records(intake_app):
    app, storage = intake_app
    app.run()
    app.text_area(key="intake_auto_text").set_value(
        "問題: 2x < 8 を解け。\n生徒答案: x < 3\n模範解答: x < 4")
    button(app, "解析して自動タグ付け").click().run()
    button(app, "確認した内容を保存").click().run()
    assert not app.exception
    assert any("匿名化生徒ID" in e.value for e in app.error)
    assert storage.list_problems() == storage.list_answers() == []


@pytest.mark.parametrize("mode,values", [
    ("problem_student", {"pair_student": "問題: 2x < 8 を解け。\n生徒答案: x < 3", "model": "x < 4"}),
    ("problem_model", {"pair_model": "問題: 2x < 8 を解け。\n模範解答: x < 4", "student": "x < 3"}),
    ("separate", {"problem": "2x < 8 を解け。", "student": "x < 3", "model": "x < 4"}),
])
def test_each_explicit_input_layout_uses_local_tagging(intake_app, mode, values):
    app, storage = intake_app
    app.run()
    app.radio(key="intake_mode").set_value(mode).run()
    for key, value in values.items():
        app.text_area(key=f"intake_{key}_text").set_value(value)
    app.text_input(key="intake_student_id").set_value("s1")
    button(app, "解析して自動タグ付け").click().run()
    assert not app.exception
    assert not app.error
    button(app, "確認した内容を保存").click().run()
    assert not app.exception
    assert storage.list_answers()[0].problem_id == storage.list_problems()[0].problem_id


def test_incomplete_problem_cannot_proceed_to_diagnosis(intake_app):
    app, storage = intake_app
    from src.intake import parse_tagging_response, save_intake_batch, IntakeDocument
    payload = {"groups": [{"title": "未完了の問題", "problem": {"text": "問題", "source_ids": ["d1"]},
                          "student_answer": {"text": "答案", "source_ids": ["d1"]},
                          "model_answer": {"text": "", "source_ids": []},
                          "confidence": 0.99}]}
    batch = parse_tagging_response(json.dumps(payload), documents=[IntakeDocument(source_id="d1", text="問題と答案")],
                                    existing_problems=[], model_name="test")
    save_intake_batch(storage, batch, student_id="s1")
    path = Path(__file__).resolve().parents[1] / "pages" / "5_diagnosis_run.py"
    result = AppTest.from_file(str(path)).run()
    assert not result.exception
    assert any("模範解答" in warning.value for warning in result.warning)
    assert not any(b.label == "診断を実行" for b in result.button)


def test_whole_shared_image_cannot_replace_extracted_answer_on_ocr_page(intake_app):
    app, storage = intake_app
    import io
    from PIL import Image
    from src.intake import parse_tagging_response, save_intake_batch, IntakeDocument
    data = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(data, format="PNG")
    image_path = storage.save_image(data.getvalue())
    payload = {"groups": [{"title": "共有画像", "problem": {"text": "問題", "source_ids": ["d1"]},
                          "student_answer": {"text": "答案", "source_ids": ["d1"]},
                          "model_answer": {"text": "模範", "source_ids": ["d1"]},
                          "confidence": 0.99}]}
    batch = parse_tagging_response(json.dumps(payload),
                                    documents=[IntakeDocument(source_id="d1", text="問題と答案と模範",
                                                               image_paths=[image_path])],
                                    existing_problems=[], model_name="test")
    save_intake_batch(storage, batch, student_id="s1")
    path = Path(__file__).resolve().parents[1] / "pages" / "3_ocr_run.py"
    result = AppTest.from_file(str(path)).run()
    assert not result.exception
    assert button(result, "答案OCRを実行").disabled
    assert button(result, "問題・模範OCRを実行").disabled


def test_saved_batch_can_be_cleared_for_next_input(intake_app):
    app, storage = intake_app
    app.run()
    app.text_area(key="intake_auto_text").set_value(
        "問題: 2x < 8 を解け。\n生徒答案: x < 3\n模範解答: x < 4")
    app.text_input(key="intake_student_id").set_value("s1")
    button(app, "解析して自動タグ付け").click().run()
    button(app, "確認した内容を保存").click().run()
    button(app, "次の資料を登録").click().run()
    assert not app.exception
    assert app.text_area(key="intake_auto_text").value == ""
    assert not any(b.label == "確認した内容を保存" for b in app.button)
    assert len(storage.list_problems()) == 1
