"""Azure AI Foundry Agent Service runtime (azure-ai-agents SDK, classic threads/runs API).

WHAT THIS CLASS DOES WITH AZURE
  Setup (`ensure_agents`, run by `python main.py setup` or at startup):
    1. Uploads knowledge/*.md and builds a Foundry VECTOR STORE from them (knowledge grounding).
    2. Creates, or updates in place, one Foundry AGENT per AgentSpec, with:
         * function tools declared with explicit JSON schemas (FunctionToolDefinition), and
         * FileSearchTool attached to the agents that need grounding (Coverage, Briefing).
       Idempotent: a fingerprint of prompt + tool schemas + model decides reuse vs update.
    3. Saves all resource IDs in .foundry_state.json, so `python main.py teardown` can delete them.

  Each agent turn (`run`):
    post user message to the agent's THREAD -> create a RUN -> poll its status:
        requires_action -> the model wants tools: execute them locally, submit_tool_outputs
        completed       -> read the agent's last message
        failed          -> retry on rate limits, otherwise raise AgentRunError
        (too long)      -> cancel and raise AgentTimeoutError (the orchestrator retries)

SDK NOTE
    azure-ai-projects 2.x moved to the new Foundry Agents (Responses) API and no longer exposes the
    classic AgentsClient. So we use azure-ai-agents' AgentsClient directly against the same Foundry
    *project endpoint*. azure-ai-projects is still used for telemetry (see observability.py).
    The classic API is deprecated (retiring 31 March 2027). Migrating means rewriting only this file.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ..agents import AgentSpec
from ..config import Settings
from ..observability import event, span
from .base import AgentRunError, AgentRuntime, AgentTimeoutError, RunResult, ToolExecutor

# Run statuses that mean "still working, keep polling".
_ACTIVE = {"queued", "in_progress", "requires_action", "cancelling"}


class FoundryRuntime(AgentRuntime):
    mode = "foundry"

    def __init__(self, settings: Settings):
        if not settings.project_endpoint:
            raise RuntimeError("PROJECT_ENDPOINT is not set. Use --mock or configure .env")
        # Imported here (not at module top) so mock mode doesn't need the Azure SDK.
        from azure.ai.agents import AgentsClient
        from azure.identity import DefaultAzureCredential

        self.settings = settings
        # Entra ID auth (az login / managed identity), no API keys in code.
        self.client = AgentsClient(endpoint=settings.project_endpoint, credential=DefaultAzureCredential())
        self.agent_ids: Dict[str, str] = {}            # agent key -> Foundry agent id
        self._state_path: Path = settings.foundry_state_path
        self._state: Dict[str, Any] = self._load_state()

    # --------------------------------------------------------------------------------- provisioning
    def _load_state(self) -> Dict[str, Any]:
        """IDs of resources created by earlier runs (agents, uploaded files, vector store)."""
        if self._state_path.exists():
            return json.loads(self._state_path.read_text())
        return {"agents": {}, "file_ids": [], "vector_store_id": None, "knowledge_hash": None}

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps(self._state, indent=2))

    def _knowledge_files(self) -> List[Path]:
        return sorted(self.settings.knowledge_dir.glob("*.md"))

    def _ensure_vector_store(self) -> str:
        """Upload the knowledge files and build a vector store, unless an identical one already exists."""
        from azure.ai.agents.models import FilePurpose

        files = self._knowledge_files()
        # Content hash of all knowledge files: if unchanged since last setup, reuse the vector store.
        digest = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:16]
        if self._state.get("vector_store_id") and self._state.get("knowledge_hash") == digest:
            try:
                self.client.vector_stores.get(self._state["vector_store_id"])
                return self._state["vector_store_id"]
            except Exception:
                event("setup.vector_store_missing", level="warning", id=self._state["vector_store_id"])
        # Knowledge changed (or store missing): delete the old store/files and rebuild.
        self._delete_knowledge()
        with span("setup.vector_store", files=len(files)):
            file_ids = []
            for p in files:
                info = self.client.files.upload_and_poll(file_path=str(p), purpose=FilePurpose.AGENTS)
                file_ids.append(info.id)
                event("setup.file_uploaded", file=p.name, file_id=info.id)
            # create_and_poll waits until the files are chunked + embedded and the store is ready.
            vs = self.client.vector_stores.create_and_poll(file_ids=file_ids, name="claims-triage-knowledge")
        self._state.update({"vector_store_id": vs.id, "file_ids": file_ids, "knowledge_hash": digest})
        self._save_state()
        event("setup.vector_store_ready", vector_store_id=vs.id, files=[p.name for p in files])
        return vs.id

    def _tools_for(self, spec: AgentSpec, registry, vector_store_id: str):
        """Build the Foundry tool list for one agent: its function tools (+ file_search if enabled)."""
        from azure.ai.agents.models import (FileSearchRankingOptions, FileSearchTool, FileSearchToolDefinition,
                                            FileSearchToolDefinitionDetails, FunctionDefinition,
                                            FunctionToolDefinition)

        # Function tools: name + description + JSON schema, taken straight from skills.py.
        tools = [
            FunctionToolDefinition(function=FunctionDefinition(
                name=d["name"], description=d["description"], parameters=d["parameters"]))
            for d in registry.definitions_for(spec.key)
        ]
        resources = None
        if spec.file_search:
            # file_search runs SERVER-SIDE in Foundry against the vector store; we never execute it.
            fs = FileSearchTool(vector_store_ids=[vector_store_id])
            # The knowledge base is tiny (a handful of chunks), so return every chunk instead of only
            # the top-scoring few: score_threshold=0 keeps low-scoring chunks, and the model then sees
            # all rules, not just the ones closest to its query.
            tools.append(FileSearchToolDefinition(file_search=FileSearchToolDefinitionDetails(
                max_num_results=20, ranking_options=FileSearchRankingOptions(ranker="auto", score_threshold=0.0))))
            resources = fs.resources
        return tools, resources

    def ensure_agents(self, specs: List[AgentSpec], registry) -> Dict[str, str]:
        """Create or update every agent. Safe to run repeatedly (idempotent)."""
        vs_id = self._ensure_vector_store() if any(s.file_search for s in specs) else None
        model = self.settings.model_deployment
        for spec in specs:
            tools, resources = self._tools_for(spec, registry, vs_id)
            # Fingerprint = prompt/schemas/model + the exact Foundry tool config (e.g. file_search
            # settings) + the vector store id, so any of those changing triggers an update.
            tool_sig = hashlib.sha256(json.dumps(
                [t.as_dict() if hasattr(t, "as_dict") else t for t in tools], sort_keys=True).encode()).hexdigest()[:6]
            fp = spec.fingerprint(registry, model) + tool_sig + (vs_id or "")[-6:]
            known = self._state["agents"].get(spec.key)
            kwargs = dict(model=model, name=spec.name, instructions=spec.instructions, tools=tools,
                          tool_resources=resources, temperature=spec.temperature,
                          metadata={"app": "claims-triage", "role": spec.key})
            with span("setup.agent", agent=spec.key):
                if known:
                    try:
                        self.client.get_agent(known["id"])            # still exists?
                        if known.get("fingerprint") != fp:
                            self.client.update_agent(known["id"], **kwargs)   # definition changed -> update
                            event("setup.agent_updated", agent=spec.key, agent_id=known["id"])
                        self.agent_ids[spec.key] = known["id"]
                        self._state["agents"][spec.key] = {"id": known["id"], "fingerprint": fp}
                        continue
                    except Exception:
                        # Deleted in the portal (or wrong project): fall through and recreate it.
                        event("setup.agent_missing", level="warning", agent=spec.key, agent_id=known["id"])
                agent = self.client.create_agent(**kwargs)
                event("setup.agent_created", agent=spec.key, agent_id=agent.id, model=model,
                      tools=[t["type"] if isinstance(t, dict) else getattr(t, "type", "?") for t in tools])
                self.agent_ids[spec.key] = agent.id
                self._state["agents"][spec.key] = {"id": agent.id, "fingerprint": fp}
        self._save_state()
        return dict(self.agent_ids)

    # ------------------------------------------------------------------------------------ execution
    def new_thread(self, agent_key: str) -> str:
        """A Foundry thread = server-side conversation history (short-term memory)."""
        thread = self.client.threads.create(metadata={"app": "claims-triage", "agent": agent_key})
        return thread.id

    def thread_exists(self, thread_id: str) -> bool:
        """Used when resuming a session: is the saved thread still there?"""
        try:
            self.client.threads.get(thread_id)
            return True
        except Exception:
            return False

    def post_note(self, thread_id: str, text: str) -> None:
        """Add orchestrator context to a thread. Assistant role if allowed, otherwise a labelled user message."""
        try:
            self.client.messages.create(thread_id=thread_id, role="assistant", content=text)
        except Exception:
            self.client.messages.create(thread_id=thread_id, role="user", content=f"[Orchestrator note]\n{text}")

    def run(self, agent_key: str, thread_id: str, message: str, executor: ToolExecutor) -> RunResult:
        """One agent turn: message -> run -> (tool calls)* -> final reply. See module docstring."""
        from azure.ai.agents.models import (MessageRole, RequiredFunctionToolCall, SubmitToolOutputsAction,
                                            ToolOutput)

        agent_id = self.agent_ids[agent_key]
        # 1. Add the user message to the agent's thread.
        self.client.messages.create(thread_id=thread_id, role="user", content=message)

        rate_limit_retries = 2
        for attempt in range(rate_limit_retries + 1):
            # 2. Start a run: the agent processes the thread.
            run = self.client.runs.create(thread_id=thread_id, agent_id=agent_id)
            calls: List[Dict[str, Any]] = []
            deadline = time.monotonic() + self.settings.run_timeout_s
            # 3. Poll until the run leaves the active states.
            while run.status in _ACTIVE:
                if time.monotonic() > deadline:
                    # Timeout: cancel the run server-side, then raise (orchestrator will retry).
                    try:
                        self.client.runs.cancel(thread_id=thread_id, run_id=run.id)
                    finally:
                        raise AgentTimeoutError(f"{agent_key} run {run.id} exceeded {self.settings.run_timeout_s}s")
                if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                    # 4. The model paused to call our function tools: execute each one locally...
                    outputs = []
                    for tc in run.required_action.submit_tool_outputs.tool_calls:
                        if isinstance(tc, RequiredFunctionToolCall):
                            out = executor(tc.function.name, tc.function.arguments)
                            outputs.append(ToolOutput(tool_call_id=tc.id, output=out))
                            calls.append({"tool": tc.function.name, "arguments": tc.function.arguments})
                    # ...and hand the results back so the run can continue.
                    run = self.client.runs.submit_tool_outputs(thread_id=thread_id, run_id=run.id, tool_outputs=outputs)
                    continue
                time.sleep(self.settings.poll_interval_s)
                run = self.client.runs.get(thread_id=thread_id, run_id=run.id)

            # 5. The run finished: success, retryable failure, or hard failure.
            if run.status == "failed":
                err = run.last_error or {}
                code = getattr(err, "code", None) or (err.get("code") if isinstance(err, dict) else None)
                msg = getattr(err, "message", None) or (err.get("message") if isinstance(err, dict) else str(err))
                if code == "rate_limit_exceeded" and attempt < rate_limit_retries:
                    wait = 5 * (attempt + 1)   # simple linear backoff: 5s, 10s
                    event("agent.rate_limited", level="warning", agent=agent_key, retry_in_s=wait)
                    time.sleep(wait)
                    continue
                raise AgentRunError(f"{agent_key} run {run.id} failed: {code}: {msg}")
            if run.status != "completed":
                raise AgentRunError(f"{agent_key} run {run.id} ended with status {run.status}")
            break

        # 6. Log server-side file_search calls, then read the agent's final message.
        self._log_run_steps(agent_key, thread_id, run.id)
        last = self.client.messages.get_last_message_text_by_role(thread_id=thread_id, role=MessageRole.AGENT)
        text = last.text.value if last is not None else ""
        usage = None
        if getattr(run, "usage", None):
            usage = {"prompt_tokens": run.usage.prompt_tokens, "completion_tokens": run.usage.completion_tokens,
                     "total_tokens": run.usage.total_tokens}
        return RunResult(text=text, status=getattr(run.status, "value", str(run.status)), run_id=run.id, tool_calls=calls, usage=usage)

    def _log_run_steps(self, agent_key: str, thread_id: str, run_id: str) -> None:
        """Surface server-side tool calls (file_search) in our trace, as they never hit the executor."""
        try:
            for step in self.client.run_steps.list(thread_id=thread_id, run_id=run_id):
                details = getattr(step, "step_details", None)
                for tc in getattr(details, "tool_calls", None) or []:
                    if getattr(tc, "type", "") == "file_search":
                        results = getattr(getattr(tc, "file_search", None), "results", None) or []
                        event("tool.file_search", agent=agent_key, run_id=run_id, results=len(results),
                              files=sorted({getattr(r, "file_name", "?") for r in results}))
        except Exception as exc:  # observability must never break the run
            event("tool.file_search", level="warning", agent=agent_key, message=f"could not list run steps: {exc}")

    # ------------------------------------------------------------------------------------- teardown
    def _delete_knowledge(self) -> None:
        """Delete the vector store and the uploaded knowledge files (errors ignored: best effort)."""
        if self._state.get("vector_store_id"):
            try:
                self.client.vector_stores.delete(self._state["vector_store_id"])
            except Exception:
                pass
        for fid in self._state.get("file_ids", []):
            try:
                self.client.files.delete(fid)
            except Exception:
                pass
        self._state.update({"vector_store_id": None, "file_ids": [], "knowledge_hash": None})

    def teardown(self) -> None:
        """Delete everything `setup` created (agents, vector store, files)."""
        for key, info in list(self._state.get("agents", {}).items()):
            try:
                self.client.delete_agent(info["id"])
                event("setup.agent_deleted", agent=key, agent_id=info["id"])
            except Exception as exc:
                event("setup.agent_delete_failed", level="warning", agent=key, error=str(exc))
        self._delete_knowledge()
        self._state = {"agents": {}, "file_ids": [], "vector_store_id": None, "knowledge_hash": None}
        self._save_state()
