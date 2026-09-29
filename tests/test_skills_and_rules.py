"""Unit tests for the building blocks: KB rule parsing, the safe condition evaluator, ID
normalisation, validation issue codes, tolerant CSV ingestion, tool-argument schema checks and
the guardrail verifier."""
import json

import pytest

from claims_triage import guardrails, knowledge
from claims_triage.data_io import normalize_claim_id, normalize_policy_number
from claims_triage.skills import validate_args


def test_rule_files_parse(settings):
    rules = knowledge.load_rules(settings.knowledge_dir)
    ids = {r.rule_id for r in rules}
    assert {"UW-01", "UW-02", "COV-02", "FI-01", "FI-06"} <= ids
    assert all(r.action in knowledge.ACTION_RANK for r in rules)
    # FI-06 / UW-07 are judgement/policy statements without a machine condition
    assert not {r.rule_id: r for r in rules}["FI-06"].machine_checkable


def test_condition_evaluator_is_safe():
    with pytest.raises(knowledge.UnsafeExpression):
        knowledge.evaluate_condition("__import__('os').system('echo hi')", {})
    with pytest.raises(knowledge.UnsafeExpression):
        knowledge.evaluate_condition("unknown_fact > 1", {"x": 1})
    assert knowledge.evaluate_condition("days_since_inception <= 7", {"days_since_inception": None}) is None
    assert knowledge.evaluate_condition('"flood" in description', {"description": "basement flooding"}) is True


def test_id_normalisation():
    assert normalize_claim_id("c2031") == "C-2031"
    assert normalize_claim_id(" C-2031 ") == "C-2031"
    assert normalize_policy_number("pol-9981") == "POL-9981"


def test_validation_flags_expected_issues(make_orch):
    orch = make_orch()
    ex = orch.registry.execute
    batch = json.loads(ex("ingest_claims", {"source_path": "data/claims.json"}, "intake"))
    out = json.loads(ex("validate_claims", {"batch_id": batch["batch_id"]}, "intake"))
    codes = {r["claim_id"] + "#" + str(r["row"]): {i["code"] for i in r["issues"]} for r in out["results"]}
    assert {"MISSING_FIELD", "NON_POSITIVE_AMOUNT"} <= codes["C-2034#3"]
    assert "LOSS_AFTER_REPORT" in codes["C-2035#4"]
    assert "UNKNOWN_POLICY" in codes["C-2036#5"]
    assert "DUPLICATE_CLAIM_ID" in codes["C-2031#7"]
    assert "DUPLICATE_SEEN" in codes["C-2031#1"]
    assert set(out["valid_claim_ids"]) == {"C-2031", "C-2033", "C-2037", "C-2038"}


def test_messy_csv_is_tolerated(make_orch):
    orch = make_orch()
    batch = json.loads(orch.registry.execute("ingest_claims", {"source_path": "data/samples/claims_messy.csv"}, "intake"))
    assert batch["claim_count"] == 4
    assert {e["code"] for e in batch["unparseable_rows"]} == {"MISSING_COLUMNS", "EXTRA_COLUMNS"}
    assert "C-2044" in batch["claim_ids"]  # 'c2044' normalised


def test_parse_json_ignores_file_search_citations():
    from claims_triage.orchestrator import Orchestrator

    reply = '```json\n{"assessments": [{"claim_id": "C-2031"}]【4:0†underwriting_rules.md】}\n```【1:2†fraud_indicators.md】'
    assert Orchestrator.parse_json(reply) == {"assessments": [{"claim_id": "C-2031"}]}


def test_tool_argument_schema_enforced(make_orch):
    orch = make_orch()
    out = json.loads(orch.registry.execute("check_coverage", {"claim_ids": "C-2031"}, "coverage"))
    assert out["error"] == "invalid_arguments"
    out = json.loads(orch.registry.execute("check_coverage", {"claim_ids": ["C-2031"]}, "intake"))
    assert out["error"] == "tool_not_permitted"
    assert validate_args({"type": "object", "properties": {"intent": {"type": "string", "enum": ["a"]}},
                          "required": ["intent"]}, {"intent": "b"})


def test_guardrail_catches_hallucination_miss_and_downgrade(settings):
    rules = knowledge.load_rules(settings.knowledge_dir)
    facts = {"claim_id": "C-1", "days_since_inception": 3, "pct_of_limit": 10, "claim_amount": 1000,
             "policy_limit": 10000, "loss_in_policy_period": True, "report_lag_days": 1, "policy_found": True,
             "coverage_type": "homeowners", "claim_type": "fire", "description": "", "scheduled_items": [],
             "prior_same_type_12m": 0, "prior_claims_12m": 0, "prior_flagged_12m": 0}
    llm = {"rules_fired": [{"rule_id": "UW-99"}, {"rule_id": "UW-05"}], "proposed_action": "auto_approve"}
    v = guardrails.verify_assessment(llm, facts, rules)
    types = {(d["type"], d["rule_id"]) for d in v["discrepancies"]}
    assert ("hallucinated_rule", "UW-99") in types
    assert ("missed_rule", "UW-01") in types
    assert ("condition_not_met", "UW-05") in types
    assert v["final_action"] == "route_to_investigator"

    b, notes = guardrails.enforce_briefing({"claim_id": "C-1", "recommended_action": "auto_approve"},
                                           "request_documentation")
    assert b["recommended_action"] == "request_documentation" and notes
