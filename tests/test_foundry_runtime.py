"""Exercises FoundryRuntime's run loop against the real azure-ai-agents model classes with a fake client
(no network): requires_action -> tool execution -> submit_tool_outputs -> completed, plus timeout."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("azure.ai.agents")
from azure.ai.agents.models import MessageTextContent, ThreadRun  # noqa: E402

from claims_triage.config import Settings  # noqa: E402
from claims_triage.runtime.base import AgentTimeoutError  # noqa: E402
from claims_triage.runtime.foundry import FoundryRuntime  # noqa: E402


def _run(status, **extra):
    return ThreadRun({"id": "run_1", "object": "thread.run", "status": status, "thread_id": "th_1",
                      "agent_id": "asst_1", **extra})


class FakeClient:
    def __init__(self, hang=False):
        self.hang = hang
        self.submitted, self.cancelled, self.messages_created = [], [], []
        req = {"type": "submit_tool_outputs", "submit_tool_outputs": {"tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "route_request", "arguments": json.dumps({"intent": "general_question", "rationale": "hi"})}}]}}
        self.runs = SimpleNamespace(
            create=lambda **kw: _run("in_progress") if hang else _run("requires_action", required_action=req),
            get=lambda **kw: _run("in_progress") if hang else _run("completed"),
            submit_tool_outputs=lambda **kw: self.submitted.append(kw) or _run("in_progress"),
            cancel=lambda **kw: self.cancelled.append(kw),
        )
        self.messages = SimpleNamespace(
            create=lambda **kw: self.messages_created.append(kw),
            get_last_message_text_by_role=lambda **kw: MessageTextContent(
                {"type": "text", "text": {"value": "Hello from Foundry", "annotations": []}}),
        )
        self.run_steps = SimpleNamespace(list=lambda **kw: [])


def _runtime(client, timeout=5.0):
    rt = object.__new__(FoundryRuntime)
    rt.settings = Settings(use_mock=False, run_timeout_s=timeout, poll_interval_s=0.0)
    rt.client = client
    rt.agent_ids = {"supervisor": "asst_1"}
    return rt


def test_function_calling_loop_submits_tool_outputs():
    client = FakeClient()
    calls = []
    res = _runtime(client).run("supervisor", "th_1", "hello",
                               lambda name, args: calls.append((name, args)) or '{"accepted": true}')
    assert res.text == "Hello from Foundry" and res.status == "completed"
    assert calls[0][0] == "route_request"
    out = client.submitted[0]["tool_outputs"][0]
    assert out.tool_call_id == "call_1" and out.output == '{"accepted": true}'


def test_run_timeout_cancels_and_raises():
    client = FakeClient(hang=True)
    with pytest.raises(AgentTimeoutError):
        _runtime(client, timeout=0.01).run("supervisor", "th_1", "hello", lambda n, a: "{}")
    assert client.cancelled
