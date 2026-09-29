"""Runtime abstraction: the ONLY way the orchestrator talks to agents.

WHY AN INTERFACE?
    The orchestrator, skills, guardrails and memory don't know or care whether agents run on
    Azure AI Foundry or on the local mock. They call these five methods:

        ensure_agents(specs, registry) -> create/update the agents (and knowledge store)
        new_thread(agent_key)          -> start a conversation thread for one agent
        post_note(thread_id, text)     -> add context to a thread without running the agent
        run(agent_key, thread_id, message, executor) -> send a message, let the agent work
                                          (calling tools via `executor`), return its final reply
        teardown()                     -> delete cloud resources

    Implementations:
        FoundryRuntime (runtime/foundry.py): Azure AI Foundry Agent Service (threads/runs,
                                             function calling, file_search).
        MockRuntime    (runtime/mock.py):    deterministic stand-in with the same contract, for demos
                                             without Azure quota (allowed by the brief).
    Moving to another backend (e.g. the new Foundry Responses-based agents) = one new class here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# The callback a runtime uses to execute a tool the agent asked for:
# (tool_name, arguments_json) -> output_json. In practice this is SkillRegistry.execute.
ToolExecutor = Callable[[str, str], str]


@dataclass
class RunResult:
    """What one agent run produced."""

    text: str                                                   # the agent's final message
    status: str = "completed"
    run_id: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)   # [{"tool": name, "arguments": json}, ...]
    usage: Optional[Dict[str, int]] = None                      # token usage, if the backend reports it


class AgentRunError(Exception):
    """The agent run failed (model error, content filter, rate limit exhausted...). The orchestrator retries."""


class AgentTimeoutError(AgentRunError):
    """The agent run didn't finish within the configured timeout. Subclass, so it's retried the same way."""


class AgentRuntime:
    """Base class / interface. See the module docstring for what each method must do."""

    mode = "abstract"

    def ensure_agents(self, specs, registry) -> Dict[str, str]:  # pragma: no cover - interface
        raise NotImplementedError

    def new_thread(self, agent_key: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def post_note(self, thread_id: str, text: str) -> None:
        """Add a message to a thread without running the agent (for orchestrator context notes)."""

    def run(self, agent_key: str, thread_id: str, message: str, executor: ToolExecutor) -> RunResult:  # pragma: no cover
        raise NotImplementedError

    def teardown(self) -> None:
        pass
