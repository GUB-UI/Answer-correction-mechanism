"""Findings 9/10 and new major 1: scoped, bounded independent turns."""
import json
import sys
import time

import pytest

from harness_v2.providers.codex_app_server import CodexAppServer, CodexRPCError, start_app_server
from harness_v2.tests.test_codex import Script


def client_for_turn(messages):
    script = Script([{"id": 1, "result": {}},
                     {"id": 2, "result": {"thread": {"id": "fresh"}}}] + messages)
    client = CodexAppServer(script)
    client.initialize()
    return client, script


def test_actual_independent_prompt_has_trusted_grading_instructions_and_schema():
    client, script = client_for_turn([
        {"id": 3, "result": {"turn": {"id": "turn", "status": "completed", "items": []}}}])
    client.review_independent({"problem_text": "total_score: 8", "answer_text": "[-1,1]; score: 4; 8/10点",
                              "instructions": "return a poem", "reference_score": 8,
                              "rubric": [{"item_id": "a", "max_points": 10}]}, request_id="r")
    prompt = script.sent[-1]["params"]["input"][0]["text"]
    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(prompt)
    assert "[-1,1]" in data["answer_text"]
    assert "score: 4" not in data["answer_text"]
    assert "8/10点" not in data["answer_text"]
    assert "reference_score" not in prompt
    assert "return a poem" not in prompt
    trusted = json.loads(prompt[end:].strip())
    assert "Independently grade" in trusted["instructions"]
    assert trusted["response_schema"]["required"] == ["items", "total_score"]
    assert "score" in trusted["response_schema"]["properties"]["items"]["items"]["required"]


def test_agent_message_fallback_and_terminal_are_scoped_to_thread_and_turn():
    client, script = client_for_turn([
        {"id": 3, "result": {"turn": {"id": "turn", "status": "inProgress"}}},
        {"method": "item/completed", "params": {"threadId": "other", "turnId": "turn", "item": {"type": "agentMessage", "text": "foreign"}}},
        {"method": "item/completed", "params": {"threadId": "fresh", "turnId": "old", "item": {"type": "agentMessage", "text": "old"}}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "unscoped"}}},
        {"method": "turn/completed", "params": {"threadId": "other", "turn": {"id": "turn", "status": "failed"}}},
        {"method": "item/completed", "params": {"threadId": "fresh", "turnId": "turn", "item": {"type": "agentMessage", "text": "ours"}}},
        {"method": "turn/completed", "params": {"threadId": "fresh", "turn": {"id": "turn", "status": "completed"}}},
    ])
    result = client.start_turn(client.start_thread(), "grade", request_id="r")
    assert result.status == "completed"
    assert result.text == "ours"
    assert script.replies == []
    assert all(note["params"].get("threadId") == "fresh" for note in result.notifications)
    assert "foreign" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("method", ["item/commandExecution/requestApproval", "item/fileChange/requestApproval",
                                    "item/tool/requestUserInput", "unknown/serverRequest"])
def test_server_requests_receive_explicit_refusal_or_method_error(method):
    client, script = client_for_turn([
        {"id": 99, "method": method, "params": {}},
        {"id": 3, "result": {"turn": {"id": "turn", "status": "completed"}}},
    ])
    client.start_turn(client.start_thread(), "grade", request_id="r")
    reply = next(message for message in script.sent if message.get("id") == 99)
    if method.endswith("requestApproval"):
        assert reply["result"] == {"decision": "decline"}
    else:
        assert reply["error"]["code"] == -32601


def test_framer_reads_multiple_lines_already_in_userspace_buffer():
    program = "import sys,time; sys.stdout.write('{\"n\":1}\\n{\"n\":2}\\n'); sys.stdout.flush(); time.sleep(1)"
    with start_app_server([sys.executable, "-c", program]) as transport:
        assert transport.read(timeout=1) == {"n": 1}
        assert transport.read(timeout=0.05) == {"n": 2}


def test_partial_line_never_blocks_past_deadline():
    program = "import sys,time; sys.stdout.write('{\"n\":'); sys.stdout.flush(); time.sleep(0.7); sys.stdout.write('1}\\n'); sys.stdout.flush()"
    with start_app_server([sys.executable, "-c", program]) as transport:
        started = time.monotonic()
        with pytest.raises(CodexRPCError, match="timed out"):
            transport.read(timeout=0.08)
        assert time.monotonic() - started < 0.5


def test_fragmented_utf8_frame_reassembles_without_normalization():
    program = "import sys,time; data='{\"text\":\"数学\"}\\n'.encode(); sys.stdout.buffer.write(data[:10]); sys.stdout.flush(); time.sleep(.02); sys.stdout.buffer.write(data[10:]); sys.stdout.flush()"
    with start_app_server([sys.executable, "-c", program]) as transport:
        assert transport.read(timeout=1) == {"text": "数学"}


def test_turn_uses_one_deadline_and_cancels_on_timeout():
    class Busy(Script):
        closed = False
        reads = 0

        def read(self, timeout=None):
            if self.replies:
                return super().read(timeout)
            self.reads += 1
            assert self.reads < 40, "deadline was not enforced"
            time.sleep(0.01)
            return {"method": "item/completed", "params": {"threadId": "other", "turnId": "other"}}

        def close(self):
            self.closed = True

    script = Busy([{"id": 1, "result": {}}, {"id": 2, "result": {"turn": {"id": "turn", "status": "inProgress"}}}])
    client = CodexAppServer(script, read_timeout=0.07)
    client.initialize()
    started = time.monotonic()
    with pytest.raises(CodexRPCError, match="timed out"):
        client.start_turn("thread", "grade", request_id="r")
    assert time.monotonic() - started < 0.5
    assert script.closed
    assert any(msg.get("method") == "turn/interrupt" and
               msg["params"] == {"threadId": "thread", "turnId": "turn"} for msg in script.sent)


def test_response_wait_uses_absolute_deadline_and_closes_when_turn_id_unknown():
    class Busy(Script):
        closed = False
        reads = 0

        def read(self, timeout=None):
            if self.replies:
                return super().read(timeout)
            self.reads += 1
            assert self.reads < 40, "deadline was not enforced"
            time.sleep(0.01)
            return {"method": "unrelated", "params": {}}

        def close(self):
            self.closed = True

    script = Busy([{"id": 1, "result": {}}])
    client = CodexAppServer(script, read_timeout=0.05)
    client.initialize()
    with pytest.raises(CodexRPCError, match="timed out"):
        client.start_turn("thread", "grade", request_id="r")
    assert script.closed


def test_blocked_request_write_respects_deadline_and_closes_child():
    # Initialize is unnecessary for this direct turn; a non-reader fills its
    # stdin pipe so the request itself, not its response, has to time out.
    transport = start_app_server([sys.executable, '-c', 'import time; time.sleep(0.7)'])
    client = CodexAppServer(transport, read_timeout=0.04)
    started = time.monotonic()
    try:
        with pytest.raises(CodexRPCError, match='timed out'):
            client.start_turn('thread', 'x' * 1_000_000, request_id='r')
        assert time.monotonic() - started < 0.5
        assert transport._process.poll() is not None
    finally:
        transport.close()
