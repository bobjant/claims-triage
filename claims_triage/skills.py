"""Skills = the custom function tools the agents can call.

WHAT A SKILL IS
    A plain Python function plus: a name, a description (the model reads it to decide when to call
    the tool), an explicit JSON schema for its arguments, and the list of agents allowed to use it.
    The `@skill(...)` decorator records all of that. `SkillRegistry` then
      * hands the schemas to the runtime (which registers them on the Foundry agents), and
      * executes tool calls when an agent asks for one (`execute()`).

THE SIX SKILLS
    | skill               | agent(s)            | purpose                                               |
    |---------------------|---------------------|-------------------------------------------------------|
    | ingest_claims       | intake              | load a CSV/JSON batch, report unparseable rows        |
    | validate_claims     | intake              | required fields, dates, policy numbers, duplicates    |
    | check_coverage      | coverage            | compute coverage/anomaly FACTS per claim              |
    | get_policy_history  | coverage, briefing  | long-term memory recall of prior claims per policy    |
    | get_triage_record   | briefing            | everything the session knows about one claim          |
    | route_request       | supervisor          | structured routing decision (intent + entities)       |

DESIGN PRINCIPLE: skills compute FACTS, the knowledge base holds RULES.
    e.g. check_coverage returns "days_since_inception = 3". It does NOT decide "that's suspicious".
    The "is that suspicious?" threshold lives in knowledge/underwriting_rules.md, and the Anomaly &
    Coverage agent applies it (and guardrails.py double-checks it).

HOW TOOLS RUN
    All skills run client-side, in this Python process. With Foundry function calling, the agent
    run pauses with status `requires_action`, we execute the function here, and submit the output
    back (see runtime/foundry.py). The executor below never raises into that loop: every failure
    becomes a structured {"error": ...} JSON the model can read and react to.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional

from . import knowledge
from .config import Settings
from .data_io import (IngestError, load_claims_file, normalize_claim_id, normalize_policy_number,
                      parse_date)
from .memory import LongTermMemory, SessionState
from .observability import event, span

POLICY_NUMBER_PATTERN = re.compile(r"^POL-\d{4}$")
REQUIRED_FIELDS = ["claim_id", "policy_number", "claim_type", "loss_date", "report_date", "claim_amount"]
# The intents the Supervisor can route to (each one maps to a handler in orchestrator.ROUTES).
INTENTS = ["triage_batch", "explain_claim", "policy_history", "general_question"]


class TransientToolError(Exception):
    """Retryable failure (timeouts, throttling, flaky downstream). The executor retries these once."""


@dataclass
class ToolContext:
    """Everything a skill function may need, passed as its first argument (`ctx`).
    Keeps skills free of globals and easy to test."""

    settings: Settings
    state: SessionState                 # short-term / session memory (batches, validation, ...)
    memory: LongTermMemory              # long-term memory (approved outcomes from earlier runs)
    policies: Dict[str, Dict[str, Any]] # policy register, keyed by policy number
    rules: List[knowledge.Rule]         # parsed knowledge-base rules
    pending_route: Optional[Dict[str, Any]] = None  # set by route_request, read by the orchestrator
    _faults_fired: set = field(default_factory=set)

    @property
    def as_of(self) -> date:
        """The reference "today" for date checks (fixed via --as-of for reproducible demos)."""
        return date.fromisoformat(self.state.as_of)

    def fault_once(self, name: str) -> bool:
        """Demo chaos: returns True the first time an injected fault is hit."""
        if name in self.settings.faults and name not in self._faults_fired:
            self._faults_fired.add(name)
            return True
        return False

    def find_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Accepted (first) occurrence of a claim in the most recent batch that contains it.
        Rows marked as duplicates by validation are ignored."""
        for batch in reversed(list(self.state.batches.values())):
            for c in batch["claims"]:
                if c["claim_id"] == claim_id and not c.get("_duplicate_of_row"):
                    return c
        return None


@dataclass
class Skill:
    """Metadata for one tool: what the model sees (name/description/parameters) + the Python function."""

    name: str
    description: str
    parameters: Dict[str, Any]           # JSON schema of the arguments
    fn: Callable[..., Dict[str, Any]]    # fn(ctx, **args) -> JSON-serialisable dict
    agents: List[str]                    # which agent keys may call it

    def definition(self) -> Dict[str, Any]:
        """The function-tool definition registered on the agent (OpenAI/Foundry function format)."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


# All skills declared in this module, collected by the @skill decorator at import time.
_SKILLS: List[Skill] = []


def skill(name: str, description: str, parameters: Dict[str, Any], agents: List[str]):
    """Decorator: register a function as a tool with its schema and owning agents."""
    def deco(fn):
        _SKILLS.append(Skill(name, description, parameters, fn, agents))
        return fn

    return deco


# ==================================================================================================
# Intake & Validation skills
# ==================================================================================================
@skill(
    name="ingest_claims",
    description=("Load a batch of claim submissions from a JSON or CSV file. Returns a batch_id, the claim "
                 "IDs loaded and any rows that could not be parsed. Call this before validate_claims."),
    parameters={
        "type": "object",
        "properties": {
            "source_path": {"type": "string", "description": "Path to the .json or .csv claims file, e.g. data/claims.json"},
        },
        "required": ["source_path"],
        "additionalProperties": False,
    },
    agents=["intake"],
)
def ingest_claims(ctx: ToolContext, source_path: str) -> Dict[str, Any]:
    try:
        claims, parse_errors = load_claims_file(source_path)
    except IngestError as exc:
        # Whole file unusable -> tell the agent (and the routine) instead of crashing.
        return {"error": exc.code, "message": exc.message, "source_path": source_path}
    # Store the batch in session memory under a simple sequential ID (B-001, B-002, ...).
    batch_id = f"B-{len(ctx.state.batches) + 1:03d}"
    ctx.state.batches[batch_id] = {"source": source_path, "claims": claims, "parse_errors": parse_errors,
                                   "validated": False}
    ctx.state.last_batch_id = batch_id
    return {
        "batch_id": batch_id,
        "source_path": source_path,
        "claim_count": len(claims),
        "claim_ids": [c["claim_id"] or f"<row {c['_row']}: missing id>" for c in claims],
        "unparseable_rows": parse_errors,
    }


@skill(
    name="validate_claims",
    description=("Run data-quality validation on an ingested batch: required fields, date formats, "
                 "loss date after report date, future dates, invalid amounts, malformed or unknown policy "
                 "numbers, duplicate claim IDs in the batch, and claims already finalised in an earlier run."),
    parameters={
        "type": "object",
        "properties": {
            "batch_id": {"type": "string", "description": "batch_id returned by ingest_claims, e.g. B-001"},
        },
        "required": ["batch_id"],
        "additionalProperties": False,
    },
    agents=["intake"],
)
def validate_claims(ctx: ToolContext, batch_id: str) -> Dict[str, Any]:
    batch = ctx.state.batches.get(batch_id)
    if batch is None:
        return {"error": "unknown_batch", "message": f"No batch {batch_id}. Known: {list(ctx.state.batches)}"}

    first_row_for: Dict[str, int] = {}   # claim_id -> row of its first (accepted) occurrence
    results = []
    for claim in batch["claims"]:
        issues = _validate_one(ctx, claim)           # field-level checks
        cid = claim["claim_id"]
        if cid:
            # Duplicate detection: the first occurrence is kept, later ones are rejected.
            if cid in first_row_for:
                claim["_duplicate_of_row"] = first_row_for[cid]
                issues.append(_issue("DUPLICATE_CLAIM_ID", "claim_id", "error",
                                     f"Duplicate of row {first_row_for[cid]}; this submission is rejected"))
            else:
                first_row_for[cid] = claim["_row"]
        # Any "error"-severity issue makes the claim invalid; warnings alone don't.
        status = "invalid" if any(i["severity"] == "error" for i in issues) else (
            "valid_with_warnings" if issues else "valid")
        rec = {"claim_id": cid or f"<row {claim['_row']}>", "row": claim["_row"], "status": status, "issues": issues}
        results.append(rec)

    # Warn the accepted occurrence that a duplicate existed.
    dupes = {c["claim_id"] for c in batch["claims"] if c.get("_duplicate_of_row")}
    for rec in results:
        if rec["claim_id"] in dupes and rec["row"] == first_row_for.get(rec["claim_id"]):
            rec["issues"].append(_issue("DUPLICATE_SEEN", "claim_id", "warning",
                                        "Another submission with this ID was rejected as a duplicate"))
            if rec["status"] == "valid":
                rec["status"] = "valid_with_warnings"

    # Save per-claim results to session memory (only the accepted occurrence of each ID).
    for rec in results:
        if rec["row"] == first_row_for.get(rec["claim_id"]):
            ctx.state.validation[rec["claim_id"]] = {**rec, "batch_id": batch_id}
    batch["validated"] = True
    batch["validation"] = results

    valid = [r["claim_id"] for r in results if r["status"] != "invalid"]
    invalid = [r["claim_id"] for r in results if r["status"] == "invalid"]
    return {
        "batch_id": batch_id,
        "summary": {"total": len(results), "valid": len(valid), "invalid": len(invalid),
                    "unparseable_rows": len(batch["parse_errors"])},
        "valid_claim_ids": valid,
        "invalid_claim_ids": invalid,
        "results": results,
    }


def _issue(code: str, field_name: str, severity: str, message: str) -> Dict[str, str]:
    """One validation finding. severity "error" = claim rejected; "warning" = informational."""
    return {"code": code, "field": field_name, "severity": severity, "message": message}


def _validate_one(ctx: ToolContext, c: Dict[str, Any]) -> List[Dict[str, str]]:
    """All data-quality checks for a single claim. These are data rules (not underwriting rules),
    so they live in code rather than the knowledge base."""
    issues = []
    # 1. Required fields present?
    for f in REQUIRED_FIELDS:
        if c.get(f) in (None, ""):
            issues.append(_issue("MISSING_FIELD", f, "error", f"Required field '{f}' is missing or empty"))
    if not c.get("description"):
        issues.append(_issue("MISSING_DESCRIPTION", "description", "warning", "No loss description provided"))

    # 2. Dates: valid format, loss not after report, nothing in the future.
    loss, report = parse_date(c.get("loss_date")), parse_date(c.get("report_date"))
    for f, parsed in (("loss_date", loss), ("report_date", report)):
        if c.get(f) and parsed is None:
            issues.append(_issue("INVALID_DATE", f, "error", f"'{c.get(f)}' is not a valid ISO date (YYYY-MM-DD)"))
    if loss and report and loss > report:
        issues.append(_issue("LOSS_AFTER_REPORT", "loss_date", "error",
                             f"Loss date {loss} is after report date {report}"))
    for f, parsed in (("loss_date", loss), ("report_date", report)):
        if parsed and parsed > ctx.as_of:
            issues.append(_issue("FUTURE_DATE", f, "error", f"{f} {parsed} is after the as-of date {ctx.as_of}"))

    # 3. Amount: numeric and positive.
    amt = c.get("claim_amount")
    if amt not in (None, ""):
        if not isinstance(amt, (int, float)):
            issues.append(_issue("INVALID_AMOUNT", "claim_amount", "error", f"'{amt}' is not a number"))
        elif amt <= 0:
            issues.append(_issue("NON_POSITIVE_AMOUNT", "claim_amount", "error",
                                 f"Claim amount {amt} must be greater than zero"))

    # 4. Policy number: right format and present in the policy register.
    pn = c.get("policy_number")
    if pn:
        if not POLICY_NUMBER_PATTERN.match(pn):
            issues.append(_issue("INVALID_POLICY_FORMAT", "policy_number", "error",
                                 f"'{pn}' does not match the POL-#### format"))
        elif pn not in ctx.policies:
            issues.append(_issue("UNKNOWN_POLICY", "policy_number", "error",
                                 f"{pn} is not in the policy register"))

    # 5. Already finalised in an earlier run? (long-term memory lookup)
    #    Only "final" outcomes block; "incomplete" ones may be corrected and resubmitted.
    prior = ctx.memory.find_claim(c.get("claim_id") or "")
    if prior and prior.get("status") == "final":
        issues.append(_issue("ALREADY_PROCESSED", "claim_id", "error",
                             f"{c['claim_id']} was already finalised on {prior.get('triaged_on')} "
                             f"({prior.get('final_action')})"))
    return issues


# ==================================================================================================
# Anomaly & Coverage skills
# ==================================================================================================
@skill(
    name="check_coverage",
    description=("Cross-check validated claims against the policy register and claims history. Returns "
                 "FACTS only (policy period, limit, % of limit, days since inception, reporting lag, "
                 "prior claims in the last 12 months, scheduled items). It does NOT apply underwriting "
                 "rules; retrieve those from the knowledge base with file search and apply them to "
                 "these facts."),
    parameters={
        "type": "object",
        "properties": {
            "claim_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                          "description": "Claim IDs to check, e.g. [\"C-1234\"]"},
        },
        "required": ["claim_ids"],
        "additionalProperties": False,
    },
    agents=["coverage"],
)
def check_coverage(ctx: ToolContext, claim_ids: List[str]) -> Dict[str, Any]:
    # Demo fault: simulate the policy admin system timing out once (the executor retries it).
    if ctx.fault_once("tool_timeout"):
        raise TransientToolError("policy administration system timed out (injected fault)")
    out = []
    for raw in claim_ids:
        cid = normalize_claim_id(raw)
        val = ctx.state.validation.get(cid)
        claim = ctx.find_claim(cid)
        # Per-claim errors are returned inline, so one bad ID doesn't fail the whole call.
        if claim is None or val is None:
            out.append({"claim_id": cid, "error": "not_ingested", "message": "Claim not found in any validated batch"})
            continue
        if val["status"] == "invalid":
            out.append({"claim_id": cid, "error": "failed_validation",
                        "message": "Claim failed intake validation; coverage not assessed",
                        "issues": [i["code"] for i in val["issues"]]})
            continue
        facts = compute_facts(ctx, claim)
        ctx.state.coverage_facts[cid] = facts   # saved so the guardrail can re-check the same facts
        out.append(facts)
    return {"as_of": ctx.state.as_of, "claims": out,
            "note": "Apply rules from underwriting_rules.md, coverage_matrix.md and fraud_indicators.md to these facts."}


def compute_facts(ctx: ToolContext, claim: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the numeric/boolean facts that rule conditions refer to (e.g. pct_of_limit).
    The fact names here must match the names used in the knowledge-base conditions."""
    loss, report = parse_date(claim["loss_date"]), parse_date(claim["report_date"])
    pol = ctx.policies.get(claim["policy_number"])
    amount = claim["claim_amount"]
    facts: Dict[str, Any] = {
        "claim_id": claim["claim_id"],
        "policy_number": claim["policy_number"],
        "claim_type": claim["claim_type"],
        "claim_amount": amount,
        "loss_date": claim["loss_date"],
        "report_date": claim["report_date"],
        "description": (claim.get("description") or "").lower(),  # lower-case for "flood" in description
        "report_lag_days": (report - loss).days if loss and report else None,
        "policy_found": pol is not None,
        # Policy-dependent facts default to None ("unknown") and are filled in below if the policy exists.
        "coverage_type": None, "policy_limit": None, "active_from": None, "active_to": None,
        "scheduled_items": [], "loss_in_policy_period": None, "days_since_inception": None,
        "days_until_expiry": None, "pct_of_limit": None,
    }
    if pol:
        start, end = parse_date(pol["active_from"]), parse_date(pol["active_to"])
        facts.update({
            "coverage_type": pol["coverage_type"],
            "policy_limit": pol["policy_limit"],
            "active_from": pol["active_from"],
            "active_to": pol["active_to"],
            "scheduled_items": pol.get("scheduled_items", []),
            "loss_in_policy_period": bool(loss and start <= loss <= end),
            "days_since_inception": (loss - start).days if loss else None,
            "days_until_expiry": (end - loss).days if loss else None,
            "pct_of_limit": round(amount / pol["policy_limit"] * 100, 1) if pol["policy_limit"] else None,
        })

    # Claims history in the 12 months before this loss (feeds the FI-* fraud indicators).
    prior = _prior_claims(ctx, claim, loss)
    facts["prior_claims"] = prior
    facts["prior_claims_12m"] = len(prior)
    facts["prior_same_type_12m"] = sum(1 for p in prior if p["claim_type"] == claim["claim_type"])
    facts["prior_flagged_12m"] = sum(1 for p in prior if p.get("flagged"))
    return facts


def _prior_claims(ctx: ToolContext, claim: Dict[str, Any], loss: Optional[date]) -> List[Dict[str, Any]]:
    """Earlier claims on the same policy with a loss date in the 365 days before this one.
    Two sources: long-term memory (earlier runs) and other claims in the current session."""
    if loss is None:
        return []
    window_start = loss - timedelta(days=365)
    prior: Dict[str, Dict[str, Any]] = {}
    # 1. Long-term memory (earlier runs, adjuster-approved outcomes). This is the cross-run memory.
    for p in ctx.memory.policy_claims(claim["policy_number"]):
        pl = parse_date(p.get("loss_date"))
        if p["claim_id"] != claim["claim_id"] and p.get("status") == "final" and pl and window_start <= pl < loss:
            prior[p["claim_id"]] = {
                "claim_id": p["claim_id"], "claim_type": p["claim_type"], "loss_date": p["loss_date"],
                "claim_amount": p.get("claim_amount"), "final_action": p.get("final_action"),
                "flagged": p.get("final_action") != "auto_approve",   # anything not auto-approved counts as flagged
                "rules_fired": p.get("rules_fired", []), "triaged_on": p.get("triaged_on"),
                "source": "long_term_memory",
            }
    # 2. Earlier-dated claims on the same policy in the current session batches.
    for cid, val in ctx.state.validation.items():
        other = ctx.find_claim(cid)
        if not other or cid == claim["claim_id"] or cid in prior or val["status"] == "invalid":
            continue
        ol = parse_date(other.get("loss_date"))
        if other["policy_number"] == claim["policy_number"] and ol and window_start <= ol < loss:
            prior[cid] = {"claim_id": cid, "claim_type": other["claim_type"], "loss_date": other["loss_date"],
                          "claim_amount": other.get("claim_amount"), "final_action": None,
                          "flagged": False, "source": "current_session"}
    return sorted(prior.values(), key=lambda p: p["loss_date"])


@skill(
    name="get_policy_history",
    description=("Recall long-term memory for a policy: the policy record plus every earlier claim that an "
                 "adjuster finalised in a previous triage run (type, dates, amount, final action, rules "
                 "fired). Use it to spot repeat claims and to reference history in briefings."),
    parameters={
        "type": "object",
        "properties": {
            "policy_number": {"type": "string", "description": "Policy number, e.g. POL-5521"},
        },
        "required": ["policy_number"],
        "additionalProperties": False,
    },
    agents=["coverage", "briefing"],
)
def get_policy_history(ctx: ToolContext, policy_number: str) -> Dict[str, Any]:
    pn = normalize_policy_number(policy_number)
    history = ctx.memory.policy_claims(pn)
    event("memory.read", policy=pn, prior_claims=len(history))
    session_claims = [cid for cid in ctx.state.validation if (ctx.find_claim(cid) or {}).get("policy_number") == pn]
    return {
        "policy_number": pn,
        "policy": ctx.policies.get(pn),
        "prior_claims": history,
        "claims_in_current_session": session_claims,
        "memory_source": "long_term_memory (data/memory/policy_history.json)",
    }


# ==================================================================================================
# Adjuster Briefing skills
# ==================================================================================================
@skill(
    name="get_triage_record",
    description=("Return everything known about one claim: the submitted data, validation issues, "
                 "coverage facts, the rules that fired (with verification status), the minimum allowed "
                 "action ('floor_action'), any briefing produced and the adjuster decision. Falls back to "
                 "long-term memory for claims triaged in earlier runs."),
    parameters={
        "type": "object",
        "properties": {
            "claim_id": {"type": "string", "description": "Claim ID, e.g. C-1234 (C1234 is also accepted)"},
        },
        "required": ["claim_id"],
        "additionalProperties": False,
    },
    agents=["briefing"],
)
def get_triage_record(ctx: ToolContext, claim_id: str) -> Dict[str, Any]:
    cid = normalize_claim_id(claim_id)
    # Prefer this session's detailed record...
    claim = ctx.find_claim(cid)
    if claim is not None and cid in ctx.state.validation:
        return triage_record(ctx, cid)
    # ...otherwise fall back to what an earlier run saved in long-term memory.
    remembered = ctx.memory.find_claim(cid)
    if remembered:
        return {"claim_id": cid, "source": "long_term_memory", "record": remembered}
    return {"error": "unknown_claim", "claim_id": cid,
            "message": f"{cid} has not been triaged in this session or any earlier run."}


def triage_record(ctx: ToolContext, cid: str) -> Dict[str, Any]:
    """Assemble the full session record for one claim. Also used directly by the orchestrator.

    floor_action = the least severe recommendation the briefing is allowed to give:
      * the verified coverage assessment's final_action, or
      * request_documentation for claims that failed validation."""
    claim = ctx.find_claim(cid) or {}
    val = ctx.state.validation.get(cid, {})
    assessment = ctx.state.assessments.get(cid)
    floor = (assessment or {}).get("final_action") or (
        "request_documentation" if val.get("status") == "invalid" else None)
    return {
        "claim_id": cid,
        "source": "current_session",
        "claim": {k: v for k, v in claim.items() if not k.startswith("_")},   # hide internal "_" fields
        "validation": {"status": val.get("status"), "issues": val.get("issues", [])},
        "coverage_facts": ctx.state.coverage_facts.get(cid),
        "assessment": assessment,
        "floor_action": floor,
        "briefing": ctx.state.briefings.get(cid),
        "adjuster_decision": ctx.state.decisions.get(cid),
    }


# ==================================================================================================
# Supervisor skill
# ==================================================================================================
@skill(
    name="route_request",
    description=("Record the routing decision for the user's latest message. Call exactly once per user "
                 "message. The orchestrator runs the matching routine afterwards."),
    parameters={
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": INTENTS,
                       "description": ("triage_batch = run the full triage routine on a claims file; "
                                       "explain_claim = explain the status/flags of specific claim(s); "
                                       "policy_history = summarise past claims for a policy; "
                                       "general_question = anything else (you answer directly)")},
            "claim_ids": {"type": "array", "items": {"type": "string"},
                          "description": ("Claim IDs referred to, resolved from context if the user says 'it' / "
                                          "'the theft one'. For triage_batch: the specific claims to triage")},
            "policy_number": {"type": "string", "description": "Policy number referred to, if any"},
            "source_path": {"type": "string",
                            "description": ("Claims file for triage_batch, exactly as the user named it. Omit when "
                                            "the user named claim IDs but no file. Never invent a path")},
            "rationale": {"type": "string", "description": "One sentence on why this route was chosen"},
        },
        "required": ["intent", "rationale"],
        "additionalProperties": False,
    },
    agents=["supervisor"],
)
def route_request(ctx: ToolContext, intent: str, rationale: str, claim_ids: Optional[List[str]] = None,
                  policy_number: Optional[str] = None, source_path: Optional[str] = None) -> Dict[str, Any]:
    # This tool doesn't DO anything itself. It captures the Supervisor's decision in a structured
    # form. The orchestrator reads ctx.pending_route after the run and executes the matching routine.
    # This is how "LLM decides what the user wants, code decides what happens" is enforced.
    ctx.pending_route = {
        "intent": intent,
        "claim_ids": [normalize_claim_id(c) for c in (claim_ids or [])],
        "policy_number": normalize_policy_number(policy_number) if policy_number else None,
        "source_path": source_path,
        "rationale": rationale,
    }
    if intent == "general_question":
        # Nothing is handed off: the Supervisor's own reply IS the answer.
        return {"accepted": True, "note": "No routine needed. Now answer the user's message yourself in 2-4 "
                                          "sentences (what this assistant can do, or the answer to their question)."}
    return {"accepted": True, "note": "Routing recorded. The orchestrator will run the routine; reply with a one-line acknowledgement."}


# ==================================================================================================
# Registry / executor
# ==================================================================================================
class SkillRegistry:
    """Looks up skills per agent and executes tool calls safely."""

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.skills: Dict[str, Skill] = {s.name: s for s in _SKILLS}

    def for_agent(self, agent_key: str) -> List[Skill]:
        return [s for s in self.skills.values() if agent_key in s.agents]

    def definitions_for(self, agent_key: str) -> List[Dict[str, Any]]:
        """Tool schemas to register on a given agent (only the tools it owns)."""
        return [s.definition() for s in self.for_agent(agent_key)]

    def execute(self, name: str, arguments: Any, agent_key: str = "?") -> str:
        """Execute a tool call from an agent. ALWAYS returns a JSON string (errors included).
        Every call is logged as a `tool.call` event with its duration and outcome."""
        start = time.perf_counter()
        result = self._execute(name, arguments, agent_key)
        ok = "error" not in result
        event("tool.call", agent=agent_key, tool=name, ok=ok,
              ms=round((time.perf_counter() - start) * 1000, 1),
              args=arguments if isinstance(arguments, dict) else _safe_json(arguments),
              **({} if ok else {"error": result.get("error"), "message": result.get("message")}),
              level="info" if ok else "warning")
        return json.dumps(result, default=str)

    def _execute(self, name: str, arguments: Any, agent_key: str) -> Dict[str, Any]:
        # Guard 1: the tool must exist.
        skill_ = self.skills.get(name)
        if skill_ is None:
            return {"error": "unknown_tool", "message": f"No tool named {name}"}
        # Guard 2: the calling agent must own the tool (least privilege). "?" = internal/orchestrator call.
        if agent_key not in skill_.agents and agent_key != "?":
            return {"error": "tool_not_permitted", "message": f"{name} is not registered to agent {agent_key}"}
        # Guard 3: arguments must be valid JSON...
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        except json.JSONDecodeError as exc:
            return {"error": "invalid_arguments", "message": f"Arguments are not valid JSON: {exc}"}
        # ...and must match the tool's JSON schema.
        problems = validate_args(skill_.parameters, args)
        if problems:
            return {"error": "invalid_arguments", "message": "; ".join(problems)}

        # Run the tool: transient failures are retried once with a short backoff; any other
        # exception becomes a structured error so the agent loop never crashes.
        attempts = 2
        for attempt in range(1, attempts + 1):
            with span(f"tool.{name}", agent=agent_key, attempt=attempt):
                try:
                    return skill_.fn(self.ctx, **args)
                except TransientToolError as exc:
                    event("tool.retry", level="warning", tool=name, attempt=attempt, error=str(exc))
                    if attempt == attempts:
                        return {"error": "tool_unavailable", "message": f"{exc} (after {attempts} attempts)"}
                    time.sleep(0.5 * attempt)
                except Exception as exc:  # never let a tool bug crash the agent loop
                    event("tool.exception", level="error", tool=name, error=repr(exc))
                    return {"error": "tool_failed", "message": f"{type(exc).__name__}: {exc}"}
        return {"error": "tool_failed"}  # pragma: no cover


def _safe_json(text: Any) -> Any:
    """Parse a JSON string for logging; fall back to the raw value."""
    try:
        return json.loads(text)
    except Exception:
        return text


# JSON-schema type name -> Python type(s)
_JSON_TYPES = {"string": str, "array": list, "object": dict, "integer": int, "number": (int, float), "boolean": bool}


def validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    """Minimal JSON-schema check (types, required, enum, additionalProperties, minItems).
    Returns a list of human-readable problems. An empty list means the arguments are valid.
    (A full validator like `jsonschema` would also work; this keeps dependencies at zero.)"""
    problems = []
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args or args[req] in (None, ""):
            problems.append(f"missing required argument '{req}'")
    for key, val in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                problems.append(f"unexpected argument '{key}'")
            continue
        if val is None:
            continue
        expected = _JSON_TYPES.get(spec.get("type"))
        if expected and not isinstance(val, expected):
            problems.append(f"'{key}' must be of type {spec['type']}")
            continue
        if "enum" in spec and val not in spec["enum"]:
            problems.append(f"'{key}' must be one of {spec['enum']}")
        if spec.get("type") == "array":
            if len(val) < spec.get("minItems", 0):
                problems.append(f"'{key}' needs at least {spec['minItems']} item(s)")
            item_type = _JSON_TYPES.get(spec.get("items", {}).get("type"))
            if item_type and not all(isinstance(v, item_type) for v in val):
                problems.append(f"all items of '{key}' must be {spec['items']['type']}")
    return problems
