"""Guardrails between the LLM and the adjuster: "the model proposes, the rules verify".

WHY THIS EXISTS
    LLMs can skip a rule, cite a rule that doesn't exist, or talk themselves into a softer
    recommendation. In a regulated process like claims triage, those errors must not reach the
    adjuster silently. Every model output is therefore checked by deterministic code before it's used.

verify_assessment(): runs after the Anomaly & Coverage agent (COVERAGE stage)
    Re-evaluates every machine-checkable KB rule against the claim's facts and compares the result
    with what the model said:
      * hallucinated rule: model cites a rule ID not in the KB          -> dropped
      * missed rule:       condition is TRUE but the model didn't cite it -> added ("added_by_verifier")
      * condition not met: model cites a rule whose condition is FALSE    -> kept as "unverified", can't escalate
      * judgement rule:    rule has no condition (e.g. FI-06)             -> kept as "model_judgement", can't escalate
    final_action = the more severe of (verified rules, model's proposal). So the model can escalate
    but can never silently downgrade. Every discrepancy is logged and shown at the adjuster checkpoint.

enforce_briefing(): runs after the Adjuster Briefing agent (BRIEF stage)
    The briefing's recommended_action can never be less severe than the triage "floor".
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import knowledge
from .observability import event


def verify_assessment(llm: Dict[str, Any], facts: Dict[str, Any], rules: List[knowledge.Rule]) -> Dict[str, Any]:
    """Reconcile the model's assessment of one claim with the deterministic rule evaluation.

    llm   : one entry of the Coverage agent's {"assessments": [...]} JSON
    facts : the check_coverage facts for the same claim
    rules : all parsed knowledge-base rules
    Returns the verified assessment that the rest of the routine uses."""
    by_id = {r.rule_id: r for r in rules}
    # Ground truth: which rules SHOULD fire for these facts.
    expected = {r["rule_id"]: r for r in knowledge.evaluate_rules(rules, facts)}
    cid = facts["claim_id"]
    merged: List[Dict[str, Any]] = []          # the reconciled list of rules fired
    discrepancies: List[Dict[str, Any]] = []   # everything the model got wrong (for audit + HITL)

    # Pass 1: check every rule the MODEL cited.
    for cited in llm.get("rules_fired", []) or []:
        rid = str(cited.get("rule_id", "")).strip().upper()
        rule = by_id.get(rid)
        if rule is None:
            # The model invented a rule ID -> drop it.
            discrepancies.append({"type": "hallucinated_rule", "rule_id": rid, "detail": cited.get("evidence")})
            continue
        if rule.action == "auto_approve":
            continue  # e.g. UW-07 describes the default outcome; citing it isn't a finding, so drop it quietly
        entry = {"rule_id": rid, "title": rule.title, "action": rule.action, "severity": rule.severity,
                 "evidence": cited.get("evidence") or (expected.get(rid) or {}).get("evidence"), "source": rule.source}
        if not rule.machine_checkable:
            entry["verification"] = "model_judgement"
            entry["action"] = "note"  # judgement-only indicators inform, never escalate on their own
        elif rid in expected:
            entry["verification"] = "verified"          # model and code agree
        else:
            # Model says it fired, code says the condition is false -> keep for the human, don't escalate.
            entry["verification"] = "unverified"
            entry["action"] = "note"
            discrepancies.append({"type": "condition_not_met", "rule_id": rid, "detail": rule.condition})
        if rid not in {m["rule_id"] for m in merged}:
            merged.append(entry)

    # Pass 2: add any rule that SHOULD have fired but the model didn't cite.
    cited_ids = {m["rule_id"] for m in merged}
    for rid, exp in expected.items():
        if rid not in cited_ids:
            discrepancies.append({"type": "missed_rule", "rule_id": rid, "detail": exp["evidence"]})
            merged.append({**exp, "verification": "added_by_verifier"})

    # Most severe rules first (nicer to read in briefings and reports).
    merged.sort(key=lambda m: (-knowledge.action_rank(m["action"]), m["rule_id"]))

    # Decide the final action: never less severe than what the verified rules require.
    proposed = llm.get("proposed_action")
    if proposed not in knowledge.FINAL_ACTIONS:
        discrepancies.append({"type": "invalid_action", "rule_id": None, "detail": proposed})
        proposed = "auto_approve"   # treat as "no opinion"; the rules decide below
    rule_action = knowledge.most_severe(m["action"] for m in merged)
    final = knowledge.most_severe([rule_action, proposed])
    if final != proposed:
        discrepancies.append({"type": "action_escalated", "rule_id": None, "detail": f"{proposed} -> {final}"})

    # Observability: every discrepancy is its own warning event in the trace.
    for d in discrepancies:
        event("guardrail.discrepancy", level="warning", claim_id=cid, **d)
    event("guardrail.verified", claim_id=cid, rules=[m["rule_id"] for m in merged], final_action=final,
          discrepancies=len(discrepancies))

    return {
        "claim_id": cid,
        "coverage_status": llm.get("coverage_status", "undetermined"),
        "rules_fired": merged,
        "proposed_action": proposed,       # what the model suggested
        "final_action": final,             # what the system recommends after verification
        "rationale": llm.get("rationale", ""),
        "discrepancies": discrepancies,
        "needs_human_attention": bool(discrepancies),
    }


def enforce_briefing(briefing: Dict[str, Any], floor_action: str) -> Tuple[Dict[str, Any], List[str]]:
    """Make sure a briefing never recommends less than the verified floor_action.
    Returns (possibly corrected briefing, list of notes explaining any correction)."""
    notes = []
    rec = briefing.get("recommended_action")
    if rec not in knowledge.FINAL_ACTIONS:
        notes.append(f"invalid recommended_action {rec!r}; using floor {floor_action}")
        rec = floor_action
    if knowledge.action_rank(rec) < knowledge.action_rank(floor_action):
        notes.append(f"briefing downgraded {floor_action} -> {rec}; restored to {floor_action}")
        rec = floor_action
    # Belt and braces: auto-approval is only allowed if the floor itself is auto_approve.
    if rec == "auto_approve" and floor_action != "auto_approve":
        rec = floor_action
    for n in notes:
        event("guardrail.briefing", level="warning", claim_id=briefing.get("claim_id"), note=n)
    return {**briefing, "recommended_action": rec}, notes
