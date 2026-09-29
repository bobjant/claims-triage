"""Supervisor orchestration: LLM routing on top, explicit routines underneath.

THE BIG PICTURE

    user message ──► Supervisor agent ──route_request(intent, entities)──► ROUTES[intent]
                                                                           │
    triage_batch:   INTAKE → VALIDATE → GATE_VALIDATION(HITL) → COVERAGE → BRIEF → GATE_ADJUSTER(HITL) → PERSIST → DONE
    explain_claim:  Briefing agent answers from the triage record + long-term memory
    policy_history: Briefing agent summarises long-term memory for a policy

    The LLM decides *what the user wants* (routing). Code decides *what happens next* (the routine).
    That split is the core design choice of this project: a regulated process must be auditable,
    so the sequence of steps is a fixed state machine, not something the model improvises.

HOW THE TRIAGE STATE MACHINE WORKS
    * STAGES lists every state; TRANSITIONS lists the legal next states of each one.
    * Every step ends by calling `_transition(report, NEXT_STAGE)`, which
        1. refuses illegal moves (raises RoutineError),
        2. logs a `routine.transition` event,
        3. appends an audit checkpoint to the session, and
        4. saves the session file (so progress survives a crash).
    * Two human-in-the-loop (HITL) checkpoints (`HumanGate`) pause the flow:
        GATE_VALIDATION: "some claims failed validation. Continue with the valid ones, or abort?"
        GATE_ADJUSTER:   "here are the recommendations. Accept, override (with a reason), or defer?"

ERROR HANDLING IN THIS FILE
    * Agent run fails or times out  -> call_agent retries (max_run_attempts), then RoutineError.
    * Agent returns bad JSON        -> _call_for_json sends one "repair" prompt, then falls back.
    * Agent skips a required tool   -> the stage checks its post-condition and calls the skill directly.
    * Coverage/Briefing agent down  -> degrade to verifier-only / placeholder briefing, flagged for a human.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import guardrails, knowledge
from .agents import ALL_AGENTS
from .config import ROOT
from .data_io import IngestError, load_claims_file, resolve_path
from .memory import LongTermMemory, SessionState, SessionStore
from .observability import event, span
from .runtime.base import AgentRunError, AgentRuntime, RunResult
from .skills import SkillRegistry, ToolContext, triage_record

# ================================================================================================
# The routine graph: the single place that defines what may follow what.
# ================================================================================================
STAGES = ["INTAKE", "VALIDATE", "GATE_VALIDATION", "COVERAGE", "BRIEF", "GATE_ADJUSTER", "PERSIST", "DONE", "ABORTED"]
TRANSITIONS: Dict[str, List[str]] = {
    "INTAKE": ["VALIDATE", "ABORTED"],                     # ABORTED = file unreadable / empty
    "VALIDATE": ["GATE_VALIDATION", "ABORTED"],
    "GATE_VALIDATION": ["COVERAGE", "BRIEF", "ABORTED"],   # BRIEF directly if no claim is valid
    "COVERAGE": ["BRIEF"],
    "BRIEF": ["GATE_ADJUSTER"],
    "GATE_ADJUSTER": ["PERSIST", "DONE"],                  # DONE = adjuster deferred; nothing persisted
    "PERSIST": ["DONE"],
    "DONE": [],                                            # terminal
    "ABORTED": [],                                         # terminal
}


def render_routine() -> str:
    """Human-readable dump of the state machine (`python main.py routine` or /routine in chat)."""
    lines = ["Triage routine (explicit state machine):"]
    for s in STAGES:
        if TRANSITIONS[s]:
            lines.append(f"  {s:<16} -> {' | '.join(TRANSITIONS[s])}")
    lines.append("  HITL gates: GATE_VALIDATION (proceed without invalid claims?), GATE_ADJUSTER (approve/override)")
    return "\n".join(lines)


# Where "triage C-1001" looks for a claim, in priority order (first file containing the ID wins).
CLAIM_SOURCES = ["data/claims.json", "data/claims_extended.json", "data/batches/*.json",
                 "data/samples/*.json", "data/samples/*.csv"]
CLAIM_ID_RE = re.compile(r"\b[cC]-?\d{4}\b")


class RoutineError(Exception):
    """Raised when the routine can't continue (illegal transition, agent failing repeatedly...)."""


# ================================================================================================
# Human-in-the-loop checkpoints
# ================================================================================================
class HumanGate:
    """Console human-in-the-loop. auto=True (--yes flag) accepts the defaults, for CI / scripted demos.
    In production this would be a work queue or UI; the routine's interface would stay the same."""

    def __init__(self, auto: bool = False, input_fn: Callable[[str], str] = input, print_fn=print):
        # input_fn / print_fn are injectable so tests can simulate a human.
        self.auto, self.input, self.print = auto, input_fn, print_fn

    def _ask(self, prompt: str, default: str, lower: bool = True) -> str:
        """Prompt the human; return the default in auto mode or on empty input. EOF (no terminal) = defer."""
        if self.auto:
            self.print(f"{prompt} [{default}] (auto)")
            return default
        try:
            ans = self.input(f"{prompt} [{default}]: ").strip()
            ans = ans.lower() if lower else ans
        except EOFError:
            return "defer"
        return ans or default

    def validation_gate(self, invalid: List[Dict[str, Any]], valid_count: int) -> str:
        """CHECKPOINT 1: show claims that failed validation; returns "continue" or "abort"."""
        if not invalid:
            return "continue"   # nothing to decide
        self.print("\n── CHECKPOINT 1 · Intake validation ─────────────────────────────")
        for r in invalid:
            codes = ", ".join(i["code"] for i in r["issues"] if i["severity"] == "error")
            self.print(f"  ✗ {r['claim_id']:<10} row {r['row']:<3} {codes}")
        self.print(f"  {valid_count} claim(s) passed; {len(invalid)} will get documentation requests.")
        ans = self._ask("  Continue with the valid claims? (c)ontinue / (a)bort", "c")
        return "abort" if ans.startswith("a") else "continue"

    def adjuster_gate(self, briefings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """CHECKPOINT 2: adjuster accepts all, overrides some (a reason is mandatory), or defers.
        Returns {"decision": "accept"|"defer", "overrides": {claim_id: {"final_action", "reason"}}}."""
        self.print("\n── CHECKPOINT 2 · Adjuster review ───────────────────────────────")
        ans = self._ask("  (a)ccept all / (o)verride some / (d)efer (nothing saved)", "a")
        if ans.startswith("d"):
            return {"decision": "defer", "overrides": {}}
        overrides: Dict[str, Dict[str, str]] = {}
        if ans.startswith("o"):
            ids = {b["claim_id"] for b in briefings}
            while True:
                cid = self._ask("    claim ID to override (blank to finish)", "").upper()
                if not cid or cid == "DEFER":
                    break
                if cid not in ids:
                    self.print(f"    unknown claim {cid}")
                    continue
                act = self._ask("    new action: auto_approve / request_documentation / route_to_investigator", "")
                if act not in knowledge.FINAL_ACTIONS:
                    self.print("    invalid action")
                    continue
                # Audit requirement: every override must say why (kept with its original case).
                reason = self._ask("    reason (required for audit)", "", lower=False)
                if not reason:
                    self.print("    a reason is required")
                    continue
                overrides[cid] = {"final_action": act, "reason": reason}
        return {"decision": "accept", "overrides": overrides}


# ================================================================================================
# Orchestrator
# ================================================================================================
@dataclass
class TriageReport:
    """Result of one triage run: what happened at each stage. Returned to the CLI and written to reports/."""

    batch_id: Optional[str] = None
    source: Optional[str] = None
    stage: str = "INTAKE"                                           # current / final stage
    validation: Dict[str, Any] = field(default_factory=dict)       # {"valid": [...], "invalid": [...], "unparseable": [...]}
    assessments: Dict[str, Any] = field(default_factory=dict)      # claim_id -> verified assessment
    briefings: List[Dict[str, Any]] = field(default_factory=list)  # one briefing per claim
    decisions: Dict[str, Any] = field(default_factory=dict)        # claim_id -> adjuster decision
    persisted: bool = False                                        # saved to long-term memory?
    message: str = ""                                              # one-line outcome for the user


class Orchestrator:
    """Runs routines and talks to agents only through the AgentRuntime interface (Foundry or mock)."""

    def __init__(self, runtime: AgentRuntime, registry: SkillRegistry, ctx: ToolContext,
                 store: SessionStore, gate: HumanGate, printer=print):
        self.rt, self.registry, self.ctx, self.store, self.gate = runtime, registry, ctx, store, gate
        self.state: SessionState = ctx.state
        self.memory: LongTermMemory = ctx.memory
        self.print = printer
        self.max_attempts = ctx.settings.max_run_attempts

    # --------------------------------------------------------------------------------- plumbing
    def setup(self) -> None:
        """Create or update all four agents (and the knowledge vector store) in the runtime."""
        with span("setup.agents"):
            self.rt.ensure_agents(ALL_AGENTS, self.registry)

    def _thread(self, agent_key: str) -> str:
        """Short-term memory: each agent has ONE thread per session. Reuse it if it still exists,
        otherwise create one and remember its ID in the session file."""
        tid = self.state.threads.get(agent_key)
        exists = getattr(self.rt, "thread_exists", lambda _t: True)
        if not tid or not exists(tid):
            tid = self.rt.new_thread(agent_key)
            self.state.threads[agent_key] = tid
            event("memory.thread_created", agent=agent_key, thread_id=tid)
        return tid

    def call_agent(self, agent_key: str, message: str) -> RunResult:
        """Run an agent on its session thread, retrying failed/timed-out runs.
        Tool calls made by the agent are executed through the skill registry, with the agent's key
        passed along so the registry can enforce which tools that agent may use."""
        thread_id = self._thread(agent_key)
        executor = lambda name, args: self.registry.execute(name, args, agent_key)  # noqa: E731
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            event("agent.run.start", agent=agent_key, thread_id=thread_id, attempt=attempt)
            with span(f"agent.{agent_key}", thread_id=thread_id, attempt=attempt):
                try:
                    res = self.rt.run(agent_key, thread_id, message, executor)
                    event("agent.run.end", agent=agent_key, run_id=res.run_id, status=res.status,
                          tool_calls=[c["tool"] for c in res.tool_calls], usage=res.usage)
                    return res
                except AgentRunError as exc:   # includes AgentTimeoutError
                    last_exc = exc
                    event("agent.run.failed", level="warning", agent=agent_key, attempt=attempt, error=str(exc))
        raise RoutineError(f"{agent_key} agent failed after {self.max_attempts} attempts: {last_exc}")

    def _transition(self, report: TriageReport, to: str, **info) -> None:
        """Move the state machine to `to`: validate, log, checkpoint, save."""
        if to not in TRANSITIONS[report.stage]:
            raise RoutineError(f"Illegal transition {report.stage} -> {to}")
        event("routine.transition", frm=report.stage, to=to, batch=report.batch_id, **info)
        self.state.routine_checkpoints.append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "batch_id": report.batch_id, "from": report.stage, "to": to, **info})
        report.stage = to
        self.store.save(self.state)  # checkpoint after every step

    @staticmethod
    def _task(**task) -> str:
        """Machine-readable task block appended to agent messages, e.g. ```task {"step": "ingest", ...}```.
        The real LLM reads it as clear instructions; the mock runtime parses it."""
        return "```task\n" + json.dumps(task) + "\n```"

    @staticmethod
    def parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from an agent reply (a ```json fenced block, or the outermost {...})."""
        # file_search adds citation markers like 【4:0†underwriting_rules.md】, which can land
        # inside the JSON and break parsing, so strip them first.
        text = re.sub(r"【[^】]*】", "", text)
        m = re.search(r"```json\s*(\{.*\})\s*```", text, re.S)
        candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None

    def _call_for_json(self, agent_key: str, message: str, key: str) -> Optional[Dict[str, Any]]:
        """Run an agent that must answer in JSON with top-level `key`.
        If the reply isn't valid, ask once more on the same thread ("repair prompt").
        Returns None if it still fails, and the caller then uses its deterministic fallback."""
        res = self.call_agent(agent_key, message)
        data = self.parse_json(res.text)
        if data is None or key not in data:
            event("agent.bad_output", level="warning", agent=agent_key, expected=key, preview=res.text[:120])
            res = self.call_agent(agent_key, f"Your last reply was not valid JSON with a top-level '{key}' key. "
                                             "Reply again with ONLY the JSON object, no prose.")
            data = self.parse_json(res.text)
            if data is None or key not in data:
                event("agent.bad_output", level="error", agent=agent_key, expected=key, giving_up=True)
                return None
        return data

    # ======================================================================= TRIAGE ROUTINE
    def triage(self, source_path: str) -> TriageReport:
        """Run the full triage routine on one claims file. Each _stage_* / _gate_* method does its
        work and then calls _transition(); this method just walks the stages in order and stops
        early at ABORTED."""
        report = TriageReport(source=source_path)
        with span("routine.triage", source=source_path):
            event("routine.start", routine="triage", source=source_path, as_of=self.state.as_of)
            try:
                self._stage_intake(report)               # INTAKE -> VALIDATE | ABORTED
                if report.stage == "ABORTED":
                    return report
                self._stage_validate(report)             # VALIDATE -> GATE_VALIDATION
                if report.stage == "ABORTED":
                    return report
                self._gate_validation(report)            # GATE 1 -> COVERAGE | BRIEF | ABORTED
                if report.stage == "ABORTED":
                    return report
                if report.stage == "COVERAGE":
                    self._stage_coverage(report)         # COVERAGE -> BRIEF
                self._stage_brief(report)                # BRIEF -> GATE_ADJUSTER
                self._gate_adjuster(report)              # GATE 2 -> PERSIST | DONE (deferred)
                if report.stage == "PERSIST":
                    self._stage_persist(report)          # PERSIST -> DONE
            except RoutineError as exc:
                # Unrecoverable problem: record it, and abort if the current stage allows that.
                event("routine.error", level="error", stage=report.stage, error=str(exc))
                report.message = f"Routine stopped at {report.stage}: {exc}"
                if report.stage in TRANSITIONS and "ABORTED" in TRANSITIONS[report.stage]:
                    self._transition(report, "ABORTED", reason=str(exc))
                self.store.save(self.state)
            # Whatever happened, tell the Supervisor's thread so follow-up questions have context.
            self._post_supervisor_note(report)
            return report

    # -- INTAKE ---------------------------------------------------------------------------------
    def _stage_intake(self, report: TriageReport) -> None:
        """Ask the Intake agent to ingest the file, then verify that it actually did."""
        before = set(self.state.batches)
        res = self.call_agent("intake", "Ingest the claims file below.\n" +
                              self._task(step="ingest", source_path=report.source))
        # Post-condition: a new batch must now exist in session memory.
        new = [b for b in self.state.batches if b not in before]
        if not new:
            # Post-condition failed: either the file is unreadable or the model skipped the tool.
            # Call the skill directly to find out which.
            out = json.loads(self.registry.execute("ingest_claims", {"source_path": report.source}, "intake"))
            if "error" in out:
                report.message = f"Could not ingest {report.source}: {out['message']}"
                self._transition(report, "ABORTED", reason=out["error"])
                return
            event("routine.fallback", level="warning", stage="INTAKE", reason="agent did not call ingest_claims")
            new = [out["batch_id"]]
        report.batch_id = new[-1]
        batch = self.state.batches[report.batch_id]
        if not batch["claims"]:
            report.message = f"No parseable claims in {report.source}."
            self._transition(report, "ABORTED", reason="empty_batch")
            return
        self.print(f"\n[Intake] {res.text}")
        self._transition(report, "VALIDATE", claims=len(batch["claims"]), unparseable=len(batch["parse_errors"]))

    # -- VALIDATE -------------------------------------------------------------------------------
    def _stage_validate(self, report: TriageReport) -> None:
        """Second turn on the SAME Intake thread. Deliberately no batch_id in the message: the agent
        must recall it from its thread (a live demo of short-term memory)."""
        res = self.call_agent("intake", "Now validate the batch you just ingested.\n" + self._task(step="validate"))
        batch = self.state.batches[report.batch_id]
        if not batch.get("validated"):
            # Post-condition failed: the agent didn't call validate_claims, so run it ourselves.
            event("routine.fallback", level="warning", stage="VALIDATE", reason="agent did not validate the batch")
            self.registry.execute("validate_claims", {"batch_id": report.batch_id}, "intake")
        results = batch["validation"]
        report.validation = {
            "valid": sorted({r["claim_id"] for r in results if r["status"] != "invalid"}),
            "invalid": [r for r in results if r["status"] == "invalid"],
            "unparseable": batch["parse_errors"],
        }
        self.print(f"[Validation] {res.text}")
        self._transition(report, "GATE_VALIDATION", valid=len(report.validation["valid"]),
                         invalid=len(report.validation["invalid"]))

    # -- GATE 1 ---------------------------------------------------------------------------------
    def _gate_validation(self, report: TriageReport) -> None:
        """Human checkpoint 1: proceed without the invalid claims, or abort the batch."""
        decision = self.gate.validation_gate(report.validation["invalid"], len(report.validation["valid"]))
        event("hitl.decision", gate="validation", decision=decision)
        if decision != "continue":
            report.message = "Aborted by handler at the validation checkpoint."
            self._transition(report, "ABORTED", reason="handler_abort")
        elif report.validation["valid"]:
            self._transition(report, "COVERAGE")
        else:
            # Nothing valid to assess: skip straight to briefings (documentation requests only).
            self._transition(report, "BRIEF", reason="no valid claims")

    # -- COVERAGE -------------------------------------------------------------------------------
    def _stage_coverage(self, report: TriageReport) -> None:
        """Coverage agent assesses valid claims (in chunks); every assessment goes through the verifier."""
        ids = report.validation["valid"]
        size = self.ctx.settings.coverage_chunk_size
        for i in range(0, len(ids), size):
            chunk = ids[i:i + size]
            with span("routine.coverage_chunk", claims=chunk):
                # 1. Ask the agent (facts via check_coverage + rules via file_search -> JSON).
                try:
                    data = self._call_for_json(
                        "coverage", "Assess coverage and anomalies for these claims.\n" +
                        self._task(step="assess", claim_ids=chunk, as_of=self.state.as_of), "assessments")
                except RoutineError as exc:  # agent down -> degrade to verifier-only, flag for human
                    event("routine.fallback", level="error", stage="COVERAGE", reason=str(exc))
                    data = None
                by_id = {normalize(a.get("claim_id")): a for a in (data or {}).get("assessments", [])}

                for cid in chunk:
                    # 2. The model's opinion for this claim (or an empty placeholder if it gave none).
                    llm = by_id.get(cid)
                    if llm is None:
                        event("routine.fallback", level="warning", stage="COVERAGE", claim_id=cid,
                              reason="no model assessment; verifier-only")
                        llm = {"claim_id": cid, "rules_fired": [], "proposed_action": "auto_approve",
                               "coverage_status": "undetermined", "rationale": "Model assessment unavailable"}
                    # 3. The facts the verifier needs (normally already computed by the agent's tool call).
                    facts = self.state.coverage_facts.get(cid)
                    if facts is None:  # tool never ran / failed for this claim -> compute directly
                        out = json.loads(self.registry.execute("check_coverage", {"claim_ids": [cid]}, "coverage"))
                        facts = next((f for f in out.get("claims", []) if "error" not in f), None)
                    if facts is None:
                        # Still no facts: can't assess safely, so ask for documents and flag for a human.
                        report.assessments[cid] = self.state.assessments[cid] = {
                            "claim_id": cid, "coverage_status": "undetermined", "rules_fired": [],
                            "final_action": "request_documentation", "proposed_action": llm.get("proposed_action"),
                            "rationale": "Coverage facts unavailable", "discrepancies": [], "needs_human_attention": True}
                        continue
                    # 4. Guardrail: reconcile the model's view with the deterministic rule check.
                    verified = guardrails.verify_assessment(llm, facts, self.ctx.rules)
                    report.assessments[cid] = self.state.assessments[cid] = verified
        self._transition(report, "BRIEF", assessed=len(report.assessments))

    # -- BRIEF ----------------------------------------------------------------------------------
    def _stage_brief(self, report: TriageReport) -> None:
        """Briefing agent writes one adjuster summary per claim (valid AND invalid ones)."""
        # Invalid claims are briefed too (they need documentation requests); rows without an ID are skipped.
        ids = report.validation["valid"] + [r["claim_id"] for r in report.validation["invalid"]
                                            if not r["claim_id"].startswith("<")]
        ids = list(dict.fromkeys(ids))   # de-duplicate, keep order
        data = None
        try:
            data = self._call_for_json("briefing", "Prepare adjuster briefings for these claims.\n" +
                                       self._task(step="brief", claim_ids=ids), "briefings")
        except RoutineError as exc:
            event("routine.fallback", level="error", stage="BRIEF", reason=str(exc))
        by_id = {normalize(b.get("claim_id")): b for b in (data or {}).get("briefings", [])}
        for cid in ids:
            # The verified minimum action for this claim (from the COVERAGE stage or validation).
            floor = triage_record(self.ctx, cid)["floor_action"] or "request_documentation"
            # Use the agent's briefing, or a safe placeholder if it didn't produce one.
            b = by_id.get(cid) or {
                "claim_id": cid, "headline": "Briefing unavailable - review manually",
                "summary": "The briefing agent did not return a summary for this claim; see the triage record.",
                "recommended_action": floor, "rules_cited": [], "memory_references": [], "documents_requested": []}
            # Guardrail: the briefing can't recommend less than the floor.
            b, notes = guardrails.enforce_briefing(b, floor)
            # Surface guardrail findings (from both stages) to the adjuster at checkpoint 2.
            b["guardrail_notes"] = notes + [f"{d['type']}:{d.get('rule_id')}" for d in
                                            (self.state.assessments.get(cid) or {}).get("discrepancies", [])]
            self.state.briefings[cid] = b
            report.briefings.append(b)
        self._print_briefings(report)
        self._transition(report, "GATE_ADJUSTER", briefings=len(report.briefings))

    # -- GATE 2 ---------------------------------------------------------------------------------
    def _gate_adjuster(self, report: TriageReport) -> None:
        """Human checkpoint 2: the adjuster approves, overrides or defers the recommendations."""
        result = self.gate.adjuster_gate(report.briefings)
        event("hitl.decision", gate="adjuster", decision=result["decision"], overrides=result["overrides"])
        if result["decision"] == "defer":
            report.message = "Adjuster deferred; recommendations are kept in the session but not saved to long-term memory."
            self._transition(report, "DONE", reason="deferred")
            return
        # Record one decision per claim: the recommendation, or the adjuster's override + reason.
        for b in report.briefings:
            cid = b["claim_id"]
            ov = result["overrides"].get(cid)
            decision = {"final_action": ov["final_action"] if ov else b["recommended_action"],
                        "recommended_action": b["recommended_action"],
                        "decided_by": "adjuster_override" if ov else ("auto_gate" if self.gate.auto else "adjuster"),
                        "override_reason": ov["reason"] if ov else None}
            if ov:
                event("hitl.override", claim_id=cid, frm=b["recommended_action"], to=ov["final_action"], reason=ov["reason"])
            self.state.decisions[cid] = report.decisions[cid] = decision
        self._transition(report, "PERSIST", decisions=len(report.decisions))

    # -- PERSIST --------------------------------------------------------------------------------
    def _stage_persist(self, report: TriageReport) -> None:
        """Write approved outcomes to LONG-TERM memory, so future runs can recall them."""
        for cid, decision in report.decisions.items():
            claim = self.ctx.find_claim(cid)
            if not claim or claim.get("policy_number") not in self.ctx.policies:
                continue  # nothing reliable to key the memory on (e.g. unknown policy number)
            val = self.state.validation.get(cid, {})
            assessment = self.state.assessments.get(cid, {})
            self.memory.record_outcome({
                "claim_id": cid,
                "policy_number": claim["policy_number"],
                "claim_type": claim.get("claim_type"),
                "loss_date": claim.get("loss_date"),
                "report_date": claim.get("report_date"),
                "claim_amount": claim.get("claim_amount"),
                "description": claim.get("description"),
                # 'incomplete' claims don't block a corrected resubmission (see ALREADY_PROCESSED check)
                "status": "incomplete" if val.get("status") == "invalid" else "final",
                "final_action": decision["final_action"],
                "rules_fired": [r["rule_id"] for r in assessment.get("rules_fired", [])],
                "decided_by": decision["decided_by"],
                "override_reason": decision["override_reason"],
                "triaged_on": self.state.as_of,
                "session_id": self.state.session_id,
            })
        report.persisted = True
        report.message = f"Triage complete; {len(report.decisions)} outcome(s) saved to long-term memory."
        self._transition(report, "DONE")

    def _post_supervisor_note(self, report: TriageReport) -> None:
        """Put the result on the Supervisor's thread so follow-up questions have context."""
        claims = [{"claim_id": b["claim_id"], "action": (report.decisions.get(b["claim_id"]) or {}).get(
            "final_action", b["recommended_action"])} for b in report.briefings]
        note = (f"Triage of {report.source} finished at stage {report.stage}. {report.message} "
                f"Results: {json.dumps(claims)}")
        self.rt.post_note(self._thread("supervisor"), note)
        if report.briefings:
            self.state.last_claim_ids = [report.briefings[0]["claim_id"]]
        self.store.save(self.state)

    def _print_briefings(self, report: TriageReport) -> None:
        """Console view of the briefings shown to the adjuster before checkpoint 2."""
        icon = {"auto_approve": "✅", "request_documentation": "📄", "route_to_investigator": "🔎"}
        self.print("\n══ ADJUSTER BRIEFINGS ═══════════════════════════════════════════")
        for b in report.briefings:
            a = self.state.assessments.get(b["claim_id"], {})
            self.print(f"\n{icon.get(b['recommended_action'], '•')} {b['claim_id']}  →  {b['recommended_action'].upper()}")
            self.print(f"   {b.get('headline', '')}")
            self.print(f"   {b.get('summary', '')}")
            if b.get("rules_cited") or a.get("rules_fired"):
                # Show each rule with its verification status, e.g. "FI-01[verified]".
                rules = a.get("rules_fired") or [{"rule_id": r, "verification": "?"} for r in b.get("rules_cited", [])]
                self.print("   Rules: " + ", ".join(f"{r['rule_id']}[{r.get('verification', '-')}]" for r in rules))
            if b.get("memory_references"):
                self.print("   Memory: " + "; ".join(b["memory_references"]))
            if b.get("documents_requested"):
                self.print("   Request: " + "; ".join(b["documents_requested"]))
            if b.get("guardrail_notes"):
                self.print("   ⚠ Guardrail: " + "; ".join(b["guardrail_notes"]))

    # ============================================================ SUPERVISOR (multi-turn entry point)
    def _session_context(self) -> str:
        """Compact summary of what this session knows, appended to every Supervisor message.
        Together with the thread history, this lets the Supervisor resolve "the theft one" or "it"."""
        triaged = []
        for cid, val in self.state.validation.items():
            claim = self.ctx.find_claim(cid) or {}
            triaged.append({"claim_id": cid, "claim_type": claim.get("claim_type"),
                            "policy_number": claim.get("policy_number"),
                            "action": (self.state.decisions.get(cid) or {}).get("final_action")
                            or (self.state.briefings.get(cid) or {}).get("recommended_action")})
        ctx = {"as_of": self.state.as_of, "triaged_claims": triaged, "last_claim_ids": self.state.last_claim_ids}
        return "SESSION CONTEXT:\n```context\n" + json.dumps(ctx) + "\n```"

    def handle(self, user_text: str) -> str:
        """Handle ONE user turn (used by `chat` and `ask`):
          1. the Supervisor agent reads the message (+ context) and calls route_request,
          2. we read the captured route (intent + entities),
          3. we run the matching handler from ROUTES and return its answer."""
        self.state.turns += 1
        with span("supervisor.turn", turn=self.state.turns):
            self.ctx.pending_route = None
            res = self.call_agent("supervisor", f"{user_text}\n\n{self._session_context()}")
            route = self.ctx.pending_route   # filled in by the route_request skill during the run
            if route is None:
                # The model answered without routing: just return its text.
                event("routine.route", level="warning", intent=None, reason="supervisor did not call route_request")
                self.store.save(self.state)
                return res.text
            event("routine.route", **route)
            handler = self.ROUTES[route["intent"]]
            answer = handler(self, route, user_text, res.text)
            if route.get("claim_ids"):
                self.state.last_claim_ids = route["claim_ids"]   # remembered for "it" / "that one" next turn
            self.store.save(self.state)
            return answer

    # Route handlers: one per intent. Signature: (route, original user text, supervisor's reply).
    def _route_triage(self, route, user_text, ack) -> str:
        """intent=triage_batch -> run the full state machine, on a whole file or on specific claims."""
        path = route.get("source_path")
        ids = list(route.get("claim_ids") or [])
        if path and not resolve_path(path).exists():
            # Guard against a model-invented path (e.g. "data/C-1001.json"): if the user named claim
            # IDs, triage those instead; otherwise let INTAKE abort with a clear file-not-found message.
            mentioned = [normalize(m) for m in CLAIM_ID_RE.findall(f"{user_text} {path}")]
            if ids or mentioned:
                event("routine.route_corrected", level="warning", invented_path=path, claim_ids=ids or mentioned)
                ids, path = ids or mentioned, None
        if ids:
            batch_path, missing = self._batch_for_claims(ids, path)
            if batch_path is None:
                where = path or "any claims file under data/"
                return f"I couldn't find {', '.join(missing)} in {where}. Check the claim ID or name the file."
            if missing:
                self.print(f"Not found, skipped: {', '.join(missing)}")
            path = str(batch_path)
        self.print(ack)
        report = self.triage(path or "data/claims.json")
        return report.message or f"Triage finished at stage {report.stage}."

    def _batch_for_claims(self, claim_ids: List[str], source: Optional[str] = None):
        """Build a small batch file holding just these claims, so "triage C-1001" works.
        Looks in `source` if given, otherwise in the known data files (first file that has the ID wins;
        within a file, the first occurrence). Returns (path of the new batch file or None, missing IDs)."""
        wanted = [normalize(c) for c in claim_ids]
        files = [resolve_path(source)] if source else [p for pattern in CLAIM_SOURCES
                                                       for p in sorted(ROOT.glob(pattern))]
        found: Dict[str, Dict[str, Any]] = {}
        for f in files:
            try:
                claims, _ = load_claims_file(str(f))
            except IngestError:
                continue
            for c in claims:
                if c["claim_id"] in wanted and c["claim_id"] not in found:
                    found[c["claim_id"]] = {k: v for k, v in c.items() if not k.startswith("_")}
                    event("routine.claim_located", claim_id=c["claim_id"], source=str(f))
        missing = [c for c in wanted if c not in found]
        if not found:
            return None, missing
        out_dir = self.ctx.settings.sessions_dir / "batches"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{self.state.session_id}_{'_'.join(c for c in wanted if c in found)}.json"
        out.write_text(json.dumps([found[c] for c in wanted if c in found], indent=2), encoding="utf-8")
        return out, missing

    def _route_explain(self, route, user_text, ack) -> str:
        """intent=explain_claim -> Briefing agent answers in prose from the triage record + memory."""
        ids = route.get("claim_ids") or []
        if not ids:
            return "Which claim do you mean? Please give a claim ID such as C-2031."
        res = self.call_agent("briefing", f"The adjuster asks: \"{user_text}\"\nAnswer in plain prose.\n" +
                              self._task(step="answer", claim_ids=ids))
        return res.text

    def _route_history(self, route, user_text, ack) -> str:
        """intent=policy_history -> Briefing agent summarises long-term memory for a policy.
        If only a claim ID was given, look up which policy it belongs to."""
        pol = route.get("policy_number")
        if not pol and route.get("claim_ids"):
            pol = (self.ctx.find_claim(route["claim_ids"][0]) or
                   self.memory.find_claim(route["claim_ids"][0]) or {}).get("policy_number")
        if not pol:
            return "Which policy? Please give a policy number such as POL-5521."
        res = self.call_agent("briefing", f"The adjuster asks: \"{user_text}\"\nSummarise the policy history in plain prose.\n" +
                              self._task(step="policy_history", policy_number=pol))
        return res.text

    def _route_general(self, route, user_text, ack) -> str:
        """intent=general_question -> the Supervisor's own reply is the answer."""
        return ack

    # Intent -> handler. Adding a new capability = add an intent in skills.INTENTS + a handler here.
    ROUTES: Dict[str, Callable[..., str]] = {
        "triage_batch": _route_triage,
        "explain_claim": _route_explain,
        "policy_history": _route_history,
        "general_question": _route_general,
    }


def normalize(cid: Any) -> str:
    """Normalise a claim ID coming back from an agent (e.g. 'c2031' -> 'C-2031')."""
    from .data_io import normalize_claim_id

    return normalize_claim_id(cid)
