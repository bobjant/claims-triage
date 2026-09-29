"""Deterministic mock of the Foundry Agent Service, for demos without Azure quota.

WHAT IS MOCKED (and what isn't)
    Mocked:   only the LLM's reasoning and the server-side file_search.
    Real:     everything else. The mock agents receive the SAME messages as the real ones, call the
              SAME function tools through the SAME executor (so schema checks, permissions, retries,
              logging and memory are all real), keep per-thread message history (short-term memory),
              and return the SAME JSON / prose formats. The orchestrator can't tell the difference.

HOW EACH MOCK AGENT "THINKS"
    Each agent is a method `_agent_<key>(turn, message)`:
      * supervisor: regex-based intent detection + reference resolution ("the theft one", "it")
      * intake:     calls ingest_claims / validate_claims; recalls batch_id from its own thread
      * coverage:   calls check_coverage + get_policy_history, "retrieves" rules via local keyword
                    search, applies rule conditions with the knowledge evaluator, returns JSON
      * briefing:   calls get_triage_record + get_policy_history and fills templated summaries

DEMO FAULTS (INJECT_FAULTS env var or --fault flag), each fires once:
    bad_json       coverage agent's first reply is malformed JSON     -> orchestrator repair/retry path
    hallucination  coverage agent cites a non-existent rule and drops a real one -> guardrail path
    agent_timeout  briefing agent's first run times out               -> orchestrator retry path
    tool_timeout   check_coverage raises a transient error once       -> skill executor retry path
                   (implemented in skills.check_coverage)
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from .. import knowledge
from ..config import Settings
from ..data_io import normalize_claim_id, normalize_policy_number
from ..memory import SessionState
from ..observability import event
from .base import AgentRuntime, AgentTimeoutError, RunResult, ToolExecutor

# The orchestrator embeds structured blocks in its messages; the mock reads them.
TASK_RE = re.compile(r"```task\s*(\{.*?\})\s*```", re.S)        # what the agent should do
CONTEXT_RE = re.compile(r"```context\s*(\{.*?\})\s*```", re.S)  # session context for the Supervisor

# Which documents to ask for when a given rule fires (used in briefings).
DOCS_FOR_RULE = {
    "COV-02": "Plumber's or cause-of-loss report showing where the water came from",
    "COV-03": "Purchase receipts or appraisals for the stolen jewelry/watches",
    "COV-04": "Police report reference number",
    "UW-05": "Written explanation of the reporting delay, plus mitigation receipts/photos",
    "UW-06": "Itemised contractor estimate or repair quote",
    "FI-03": "Repair invoices / closure evidence for the earlier flagged claim",
}

# Words a user might use to refer to a claim by its type ("what about the theft one?").
TYPE_WORDS = {"theft": "theft", "stolen": "theft", "water": "water_damage", "pipe": "water_damage",
              "flood": "water_damage", "fire": "fire", "smoke": "fire", "wind": "wind_hail",
              "hail": "wind_hail", "storm": "wind_hail", "vandal": "vandalism", "graffiti": "vandalism"}


class _Turn:
    """Helper handed to a mock agent for one run: lets it call tools and 'file_search' while
    recording everything on the thread, just like a real run would."""

    def __init__(self, runtime: "MockRuntime", agent_key: str, thread: List[Dict[str, Any]], executor: ToolExecutor):
        self.rt, self.agent_key, self.thread, self.executor = runtime, agent_key, thread, executor
        self.calls: List[Dict[str, Any]] = []

    def tool(self, name: str, **args) -> Dict[str, Any]:
        """Call a function tool through the real skill executor (same path as Foundry function calling)."""
        payload = json.dumps(args)
        out = json.loads(self.executor(name, payload))
        self.calls.append({"tool": name, "arguments": payload})
        self.thread.append({"role": "tool", "name": name, "arguments": args, "output": out})
        return out

    def file_search(self, query: str, k: int = 5) -> List[knowledge.Rule]:
        """Stand-in for Foundry file_search: keyword retrieval over the same knowledge files."""
        hits = self.rt.kb.search(query, k=k)
        event("tool.file_search", agent=self.agent_key, mode="mock-local-retrieval", query=query,
              results=[r.rule_id for r in hits])
        self.thread.append({"role": "tool", "name": "file_search", "arguments": {"query": query},
                            "output": [r.rule_id for r in hits]})
        return hits


class MockRuntime(AgentRuntime):
    mode = "mock"

    def __init__(self, settings: Settings, state: SessionState):
        self.settings = settings
        self.state = state                       # mock threads are stored in the session file
        self.rules = knowledge.load_rules(settings.knowledge_dir)
        self.rules_by_id = {r.rule_id: r for r in self.rules}
        self.kb = knowledge.LocalKnowledgeSearch(self.rules)
        self._faults_fired: set = set()
        self._last_json: Dict[str, str] = {}     # last good JSON per agent (for the bad_json repair path)

    def _fault(self, name: str) -> bool:
        """True the first time an injected fault is reached (each fault fires once)."""
        if name in self.settings.faults and name not in self._faults_fired:
            self._faults_fired.add(name)
            return True
        return False

    # ---------------------------------------------------------------------------- runtime contract
    def ensure_agents(self, specs, registry) -> Dict[str, str]:
        """No cloud resources: just log which tools each agent would get."""
        ids = {}
        for spec in specs:
            ids[spec.key] = f"mock-agent-{spec.key}"
            event("setup.agent_created", agent=spec.key, agent_id=ids[spec.key], mode="mock",
                  tools=[d["name"] for d in registry.definitions_for(spec.key)] + (["file_search"] if spec.file_search else []))
        return ids

    def new_thread(self, agent_key: str) -> str:
        tid = f"mock-thread-{agent_key}-{uuid.uuid4().hex[:8]}"
        self.state.mock_threads[tid] = []
        return tid

    def thread_exists(self, thread_id: str) -> bool:
        return thread_id in self.state.mock_threads

    def post_note(self, thread_id: str, text: str) -> None:
        self.state.mock_threads.setdefault(thread_id, []).append({"role": "assistant", "content": text})

    def run(self, agent_key: str, thread_id: str, message: str, executor: ToolExecutor) -> RunResult:
        """Append the message to the thread, dispatch to `_agent_<key>`, append the reply."""
        thread = self.state.mock_threads.setdefault(thread_id, [])
        thread.append({"role": "user", "content": message})
        turn = _Turn(self, agent_key, thread, executor)
        text = getattr(self, f"_agent_{agent_key}")(turn, message)
        thread.append({"role": "assistant", "content": text})
        return RunResult(text=text, run_id=f"mock-run-{uuid.uuid4().hex[:8]}", tool_calls=turn.calls,
                         usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    @staticmethod
    def _task(message: str) -> Dict[str, Any]:
        """Parse the ```task {...}``` block the orchestrator appended to the message."""
        m = TASK_RE.search(message)
        return json.loads(m.group(1)) if m else {}

    # ================================================================================ supervisor
    def _agent_supervisor(self, turn: _Turn, message: str) -> str:
        """Classify the user's intent, extract/resolve entities, call route_request."""
        ctx_match = CONTEXT_RE.search(message)
        ctx = json.loads(ctx_match.group(1)) if ctx_match else {}
        user_text = CONTEXT_RE.sub("", message).replace("SESSION CONTEXT:", "").strip()
        low = user_text.lower()

        # 1. Extract explicit entities: claim IDs, policy numbers, file paths.
        claim_ids = [normalize_claim_id(m) for m in re.findall(r"\b[cC]-?\d{4}\b", user_text)]
        policy = next((normalize_policy_number(m) for m in re.findall(r"\b[pP][oO][lL]-?\d{4}\b", user_text)), None)
        path = next(iter(re.findall(r"[\w./\-]+\.(?:json|csv)\b", user_text)), None)
        known = ctx.get("triaged_claims", [])

        # 2. No explicit ID? Resolve references from session context (short-term memory).
        if not claim_ids and known:  # resolve "the theft one", "it", "that claim" from thread context
            types = {t for w, t in TYPE_WORDS.items() if w in low}
            if types:
                claim_ids = [c["claim_id"] for c in known if c.get("claim_type") in types]
            elif re.search(r"\b(it|that|this|same|the claim|that one|this one)\b", low):
                claim_ids = list(ctx.get("last_claim_ids", []))[:1]

        # 3. Pick the intent (most specific first).
        if path or re.search(r"\b(tri?age|process|ingest|run the batch|new batch)\b", low):
            # Named claims without a file -> triage just those (the orchestrator finds them).
            route = dict(intent="triage_batch", claim_ids=claim_ids if not path else None,
                         source_path=path or (None if claim_ids else "data/claims.json"),
                         rationale="User asked to process claims")
        elif re.search(r"\b(history|previous|prior|past|before)\b", low) and (policy or claim_ids):
            route = dict(intent="policy_history", policy_number=policy, claim_ids=claim_ids,
                         rationale="User asked about earlier claims on a policy")
        elif claim_ids:
            route = dict(intent="explain_claim", claim_ids=claim_ids,
                         rationale="User asked about specific claim(s)")
        elif policy:
            route = dict(intent="policy_history", policy_number=policy, rationale="User referenced a policy")
        else:
            route = dict(intent="general_question", rationale="No claim, policy or batch referenced")
        # 4. Record the decision via the tool (the orchestrator reads it after the run).
        turn.tool("route_request", **{k: v for k, v in route.items() if v})

        if route["intent"] == "general_question":
            return ("I'm the Claims Triage Supervisor. Ask me to triage a batch (e.g. 'triage data/claims.json'), "
                    "explain a claim ('why was C-2031 flagged?'), or show a policy's history "
                    "('history for POL-5521'). A triage runs intake -> validation -> [checkpoint] -> "
                    "coverage & anomaly checks -> adjuster briefing -> [your approval].")
        target = route.get("source_path") or ", ".join(route.get("claim_ids") or []) or route.get("policy_number")
        return f"Routing to the {route['intent'].replace('_', ' ')} routine for {target}."

    # ==================================================================================== intake
    def _agent_intake(self, turn: _Turn, message: str) -> str:
        """Step 'ingest' -> ingest_claims. Step 'validate' -> validate_claims (batch_id from memory)."""
        task = self._task(message)
        if task.get("step") == "ingest":
            out = turn.tool("ingest_claims", source_path=task["source_path"])
            if "error" in out:
                return f"Could not ingest {task['source_path']}: {out['error']} - {out['message']}"
            bad = "; ".join(f"row {e['row']}: {e['code']}" for e in out["unparseable_rows"]) or "none"
            return (f"Ingested {out['claim_count']} claim(s) from {out['source_path']}. "
                    f"Unparseable rows: {bad}.\nbatch_id: {out['batch_id']}")
        if task.get("step") == "validate":
            # Thread memory: recover the batch_id from this agent's own earlier turn.
            batch_id = task.get("batch_id")
            if not batch_id:
                for msg in reversed(turn.thread):
                    m = re.search(r"batch_id:\s*(B-\d+)", str(msg.get("content", "")))
                    if msg["role"] == "assistant" and m:
                        batch_id = m.group(1)
                        break
            if not batch_id:
                return "I have no ingested batch in this conversation to validate."
            out = turn.tool("validate_claims", batch_id=batch_id)
            if "error" in out:
                return f"Validation failed: {out['message']}"
            s = out["summary"]
            lines = [f"Validated {s['total']} claim(s): {s['valid']} valid, {s['invalid']} invalid, "
                     f"{s['unparseable_rows']} unparseable row(s)."]
            for r in out["results"]:
                if r["status"] == "invalid":
                    codes = ", ".join(i["code"] for i in r["issues"] if i["severity"] == "error")
                    lines.append(f"- {r['claim_id']} (row {r['row']}): {codes}")
            lines.append(f"batch_id: {batch_id}")
            return "\n".join(lines)
        return "Please tell me which file to ingest or which batch to validate."

    # ================================================================================== coverage
    def _agent_coverage(self, turn: _Turn, message: str) -> str:
        """Facts (check_coverage) + history (get_policy_history) + rules (file_search) -> JSON assessments."""
        # Repair prompt after a bad_json fault: answer again with the valid JSON.
        if "not valid JSON" in message and "coverage" in self._last_json:
            return self._last_json["coverage"]
        task = self._task(message)
        claim_ids = task.get("claim_ids", [])
        out = turn.tool("check_coverage", claim_ids=claim_ids)
        if "error" in out:
            # Tool unavailable even after retries: report "undetermined" rather than guessing.
            result = {"assessments": [{"claim_id": c, "coverage_status": "undetermined", "rules_fired": [],
                                       "proposed_action": "request_documentation",
                                       "rationale": f"Coverage facts unavailable: {out['message']}"} for c in claim_ids]}
            return json.dumps(result)

        # Look up history for each policy involved (long-term memory).
        facts_list = [f for f in out["claims"] if "error" not in f]
        for pol in sorted({f["policy_number"] for f in facts_list}):
            turn.tool("get_policy_history", policy_number=pol)

        assessments = []
        for item in out["claims"]:
            if "error" in item:
                assessments.append({"claim_id": item["claim_id"], "coverage_status": "undetermined",
                                    "rules_fired": [], "proposed_action": "request_documentation",
                                    "rationale": item["message"]})
                continue
            f = item
            # "Retrieve" relevant rules (logged as file_search events, like the real run steps).
            retrieved = {}
            for q in (f"policy inception limit period reporting {f['claim_type']}",
                      f"{f['claim_type'].replace('_', ' ')} exclusion sub-limit covered perils {f['description']}",
                      "prior claims repeat frequency flagged fraud indicators"):
                for r in turn.file_search(q):
                    retrieved[r.rule_id] = r
            # "Reasoning": apply every rule's condition to the facts. Retrieval is logged above; a real
            # model only sees retrieved chunks, which is exactly why the verifier re-checks all rules.
            fired = knowledge.evaluate_rules(self.rules, f)
            action = knowledge.most_severe(r["action"] for r in fired)
            ids = {r["rule_id"] for r in fired}
            status = ("not_covered" if ids & {"COV-01", "UW-04", "UW-08"} else
                      "partially_covered" if ids & {"COV-02", "COV-03", "UW-03"} else "covered")
            assessments.append({
                "claim_id": f["claim_id"],
                "coverage_status": status,
                "rules_fired": [{"rule_id": r["rule_id"], "evidence": r["evidence"], "judgement": False} for r in fired],
                "proposed_action": action,
                "rationale": ("; ".join(f"{r['rule_id']} {r['title']}" for r in fired) or
                              "No rule fired; eligible for auto-approval under UW-07"),
            })

        # Demo fault: behave like a sloppy model (invent UW-99, forget a real rule).
        if self._fault("hallucination"):
            victim = next((a for a in assessments if a["rules_fired"]), None)
            if victim:
                dropped = victim["rules_fired"].pop(0)
                victim["rules_fired"].append({"rule_id": "UW-99", "evidence": "claimant seemed nervous", "judgement": False})
                event("mock.fault_injected", level="warning", fault="hallucination", claim=victim["claim_id"],
                      dropped=dropped["rule_id"], invented="UW-99")

        text = "```json\n" + json.dumps({"assessments": assessments}, indent=1) + "\n```"
        self._last_json["coverage"] = text
        # Demo fault: return broken JSON first; the orchestrator's repair prompt then gets the good one.
        if self._fault("bad_json"):
            event("mock.fault_injected", level="warning", fault="bad_json")
            return "Here are the assessments: {assessments: [{claim_id: C-20..."  # truncated / invalid
        return text

    # ================================================================================== briefing
    def _agent_briefing(self, turn: _Turn, message: str) -> str:
        """Step 'brief' -> JSON briefings. Steps 'answer' / 'policy_history' -> prose answers."""
        if self._fault("agent_timeout"):
            event("mock.fault_injected", level="warning", fault="agent_timeout")
            raise AgentTimeoutError("briefing run exceeded timeout (injected fault)")
        task = self._task(message)
        if task.get("step") == "brief":
            briefings = [self._brief_one(turn, cid) for cid in task.get("claim_ids", [])]
            return "```json\n" + json.dumps({"briefings": briefings}, indent=1) + "\n```"
        return self._answer(turn, task)

    def _history(self, turn: _Turn, policy_number: Optional[str], exclude: Optional[str] = None):
        """get_policy_history, minus the claim we're currently talking about."""
        if not policy_number:
            return {"prior_claims": [], "policy": None}
        out = turn.tool("get_policy_history", policy_number=policy_number)
        out["prior_claims"] = [p for p in out.get("prior_claims", []) if p["claim_id"] != exclude]
        return out

    @staticmethod
    def _months_between(a: str, b: str) -> int:
        """Rounded number of months between two ISO dates (for '6 months before this loss')."""
        from datetime import date

        d = abs((date.fromisoformat(b) - date.fromisoformat(a)).days)
        return max(1, round(d / 30.44))

    def _memory_refs(self, rec: Dict[str, Any], history: Dict[str, Any]):
        """Turn earlier claims into (short references, readable sentences) for the briefing.
        This is where 'similar water damage claim flagged 6 months ago' is written."""
        refs, sentences = [], []
        claim = rec.get("claim", {})
        # Earlier runs (long-term memory).
        for p in history.get("prior_claims", []):
            outcome = p.get("final_action") or "pending"
            refs.append(f"{p['claim_id']} ({p['loss_date']}, {p['claim_type']}, {outcome})")
            when = ""
            if claim.get("loss_date") and p.get("loss_date"):
                when = f" {self._months_between(p['loss_date'], claim['loss_date'])} months before this loss"
            similar = "a similar" if p["claim_type"] == claim.get("claim_type") else "a"
            verb = "flagged for " + outcome.replace("_", " ") if outcome != "auto_approve" else "auto-approved"
            sentences.append(f"This policy had {similar} {p['claim_type']} claim ({p['claim_id']}, loss "
                             f"{p['loss_date']}) that was {verb}{when}.")
        # Earlier claims in the same batch (current session).
        for f in (rec.get("coverage_facts") or {}).get("prior_claims", []):
            if f["source"] == "current_session" and f["claim_id"] not in " ".join(refs):
                refs.append(f"{f['claim_id']} ({f['loss_date']}, {f['claim_type']}, same batch)")
                sentences.append(f"The same batch also has an earlier {f['claim_type']} claim on this policy "
                                 f"({f['claim_id']}, loss {f['loss_date']}).")
        return refs, sentences

    def _brief_one(self, turn: _Turn, claim_id: str) -> Dict[str, Any]:
        """Build one adjuster briefing (JSON shape defined in the Briefing agent's instructions)."""
        rec = turn.tool("get_triage_record", claim_id=claim_id)
        if "error" in rec:
            return {"claim_id": claim_id, "headline": "No triage record", "summary": rec["message"],
                    "recommended_action": "request_documentation", "rules_cited": [], "memory_references": [],
                    "documents_requested": []}
        claim = rec["claim"]
        history = self._history(turn, claim.get("policy_number"), exclude=claim_id)
        refs, mem_sentences = self._memory_refs(rec, history)
        action = rec.get("floor_action") or "request_documentation"   # never go below the verified floor

        # Case A: the claim failed validation -> ask the claimant for corrected data.
        if rec["validation"]["status"] == "invalid":
            issues = [i for i in rec["validation"]["issues"] if i["severity"] == "error"]
            fields = sorted({i["field"] for i in issues})
            summary = (f"{(claim.get('claim_type') or 'unknown-type').replace('_', ' ').capitalize()} claim on {claim.get('policy_number') or 'no policy'} "
                       f"failed intake validation: " + "; ".join(i["message"] for i in issues) + ". "
                       "Coverage and anomaly checks were not run.")
            # Extra insight even for invalid claims: e.g. C-2034's policy had already expired.
            pol = history.get("policy")
            if pol and claim.get("report_date") and claim["report_date"] > pol["active_to"]:
                summary += (f" Note: policy {pol['policy_number']} expired on {pol['active_to']}, before this claim "
                            f"was reported on {claim['report_date']}, so a coverage issue is likely once the loss date is known.")
            if mem_sentences:
                summary += " " + " ".join(mem_sentences)
            return {"claim_id": claim_id, "headline": "Incomplete submission - return to claimant",
                    "summary": summary, "recommended_action": "request_documentation", "rules_cited": [],
                    "memory_references": refs,
                    "documents_requested": [f"Corrected claim form with valid {', '.join(fields)}"]}

        # Case B: valid claim -> summarise facts, fired rules and history.
        assessment = rec.get("assessment") or {}
        fired = assessment.get("rules_fired", [])
        escalating = [r for r in fired if knowledge.action_rank(r.get("action", "note")) > 0]
        amount = claim.get("claim_amount")
        facts = rec.get("coverage_facts") or {}
        parts = [f"{claim['claim_type'].replace('_', ' ').capitalize()} claim for {amount:,} on {claim['policy_number']} "
                 f"({claim.get('description')}); loss {claim['loss_date']}, reported {claim['report_date']}"
                 f" ({facts.get('report_lag_days')} day(s) later). Amount is {facts.get('pct_of_limit')}% of the "
                 f"{facts.get('policy_limit'):,} limit."]
        if fired:
            parts.append("Rules fired: " + "; ".join(
                f"{r['rule_id']} {r.get('title', '')} ({r.get('action')})" for r in fired) + ".")
        else:
            parts.append("No underwriting, coverage or fraud-indicator rule fired.")
        parts += mem_sentences
        # Headline names the most severe rule (rules are sorted most-severe-first by the verifier).
        headline = {
            "route_to_investigator": "Route to investigator - " + (escalating[0].get("title", "") if escalating else ""),
            "request_documentation": "Request documentation - " + (escalating[0].get("title", "") if escalating else ""),
            "auto_approve": "Clean claim - eligible for auto-approval (UW-07)",
        }[action]
        docs = [DOCS_FOR_RULE[r["rule_id"]] for r in fired if r["rule_id"] in DOCS_FOR_RULE]
        return {"claim_id": claim_id, "headline": headline.strip(" -"), "summary": " ".join(parts),
                "recommended_action": action, "rules_cited": [r["rule_id"] for r in fired],
                "memory_references": refs, "documents_requested": docs}

    def _answer(self, turn: _Turn, task: Dict[str, Any]) -> str:
        """Prose answers for follow-up questions ('why was C-2031 flagged?', 'history for POL-5521')."""
        # Policy history question.
        if task.get("step") == "policy_history":
            pol = task.get("policy_number")
            hist = self._history(turn, pol)
            if not hist.get("policy"):
                return f"{pol} is not in the policy register."
            p = hist["policy"]
            lines = [f"{pol}: {p['coverage_type']} policy, limit {p['policy_limit']:,}, active {p['active_from']} to {p['active_to']}."]
            if not hist["prior_claims"]:
                lines.append("No claims recorded in long-term memory for this policy.")
            for c in hist["prior_claims"]:
                lines.append(f"- {c['claim_id']}: {c['claim_type']} loss {c['loss_date']}, {c.get('claim_amount'):,} -> "
                             f"{c.get('final_action')} (rules: {', '.join(c.get('rules_fired', [])) or 'none'}; "
                             f"triaged {c.get('triaged_on')})")
            if hist.get("claims_in_current_session"):
                lines.append("In this session: " + ", ".join(hist["claims_in_current_session"]))
            return "\n".join(lines)

        # "Why was X flagged?" question, for one or more claims.
        answers = []
        for cid in task.get("claim_ids", []):
            rec = turn.tool("get_triage_record", claim_id=cid)
            if "error" in rec:
                answers.append(rec["message"])
                continue
            if rec.get("source") == "long_term_memory":   # triaged in an earlier run
                r = rec["record"]
                answers.append(f"{cid} was triaged in an earlier run on {r.get('triaged_on')}: {r.get('final_action')} "
                               f"(rules: {', '.join(r.get('rules_fired', [])) or 'none'}).")
                continue
            decision = rec.get("adjuster_decision") or {}
            final = decision.get("final_action") or rec.get("floor_action")
            if rec["validation"]["status"] == "invalid":
                issues = "; ".join(i["message"] for i in rec["validation"]["issues"] if i["severity"] == "error")
                answers.append(f"{cid} was stopped at intake validation ({issues}). Recommendation: {final}.")
                continue
            fired = (rec.get("assessment") or {}).get("rules_fired", [])
            if not fired:
                answers.append(f"{cid} was not flagged: no rule fired, so it is eligible for auto-approval (UW-07). "
                               f"Current status: {final}.")
                continue
            lines = [f"{cid} -> {final.replace('_', ' ')}. Reasons:"]
            for r in fired:
                rule = self.rules_by_id.get(r["rule_id"])
                if rule:
                    # Quote the rule's own wording from the knowledge base (grounded explanation).
                    self_hits = turn.file_search(f"{rule.rule_id} {rule.title}", k=1)
                    para = (self_hits[0] if self_hits else rule).text.split("\n\n")[-1].replace("\n", " ")
                    wording = para.split(". ")[0].rstrip(".")
                    lines.append(f"  - {r['rule_id']} {rule.title} [{r.get('verification', 'verified')}]: {wording}. "
                                 f"Evidence: {r.get('evidence')}")
                else:
                    lines.append(f"  - {r['rule_id']}: {r.get('evidence')}")
            history = self._history(turn, rec["claim"]["policy_number"], exclude=cid)
            _, mem = self._memory_refs(rec, history)
            if mem:
                lines.append("  Memory: " + " ".join(mem))
            if decision:
                lines.append(f"  Adjuster decision: {decision.get('final_action')} ({decision.get('decided_by')}).")
            answers.append("\n".join(lines))
        return "\n\n".join(answers) or "Which claim do you mean?"
