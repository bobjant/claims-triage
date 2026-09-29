"""End-to-end tests of the orchestrator in mock mode: long-term memory changing an outcome across
runs, legal state-machine transitions, surviving injected faults, clean aborts, and multi-turn
Supervisor conversations (short-term memory across processes)."""
from claims_triage.orchestrator import STAGES, TRANSITIONS


def test_long_term_memory_changes_the_outcome(make_orch):
    # Without history, C-2031 (8,500 burst pipe) is a clean claim.
    orch = make_orch(session="no-memory")
    report = orch.triage("data/claims.json")
    assert report.stage == "DONE"
    assert orch.state.briefings["C-2031"]["recommended_action"] == "auto_approve"
    orch.memory.reset()

    # Run 1 (February): C-2032 "basement flooding" is flagged and persisted.
    feb = make_orch(session="feb", as_of="2026-02-20")
    feb.triage("data/batches/2026-02_batch.json")
    assert feb.memory.find_claim("C-2032")["final_action"] == "request_documentation"

    # Run 2 (September, new session/process): C-2031 now recalls C-2032 from long-term memory.
    sep = make_orch(session="sep")
    report = sep.triage("data/claims.json")
    b = sep.state.briefings["C-2031"]
    assert b["recommended_action"] == "route_to_investigator"
    assert any("C-2032" in m for m in b["memory_references"])
    assert "6 months" in b["summary"]
    assert {r["rule_id"] for r in sep.state.assessments["C-2031"]["rules_fired"]} >= {"FI-01", "FI-03"}

    expected = {"C-2033": "route_to_investigator", "C-2037": "auto_approve", "C-2038": "request_documentation",
                "C-2034": "request_documentation", "C-2035": "request_documentation"}
    for cid, action in expected.items():
        assert sep.state.decisions[cid]["final_action"] == action, cid


def test_routine_transitions_are_legal_and_logged(make_orch):
    orch = make_orch(session="trace")
    orch.triage("data/claims.json")
    path = [c["to"] for c in orch.state.routine_checkpoints]
    assert path == ["VALIDATE", "GATE_VALIDATION", "COVERAGE", "BRIEF", "GATE_ADJUSTER", "PERSIST", "DONE"]
    for c in orch.state.routine_checkpoints:
        assert c["to"] in TRANSITIONS[c["from"]]
    assert set(TRANSITIONS) == set(STAGES)


def test_faults_are_survived(make_orch):
    orch = make_orch(session="chaos", faults={"bad_json", "hallucination", "agent_timeout", "tool_timeout"})
    report = orch.triage("data/samples/claims_sample_original.json")
    assert report.stage == "DONE"
    notes = " ".join(" ".join(b.get("guardrail_notes", [])) for b in report.briefings)
    assert "hallucinated_rule:UW-99" in notes


def test_unreadable_file_aborts_cleanly(make_orch):
    orch = make_orch(session="bad")
    report = orch.triage("data/does_not_exist.json")
    assert report.stage == "ABORTED" and "Could not ingest" in report.message


def test_multi_turn_supervisor_resolves_references(make_orch):
    orch = make_orch(session="chat")
    orch.handle("triage data/claims.json")
    assert "C-2031" in orch.handle("why was C2031 flagged?")
    theft = orch.handle("what about the theft one?")
    assert "C-2033" in theft and "UW-01" in theft
    assert orch.state.last_claim_ids == ["C-2033"]
    # A new orchestrator on the same session id keeps the thread + context.
    again = make_orch(session="chat")
    assert "C-2033" in again.handle("and is it still pending?")
    hist = again.handle("show the history for POL-5521")
    assert "POL-5521" in hist


def test_extended_batch_outcomes(make_orch):
    """data/claims_extended.json (C-1001..C-1013): each record targets a specific rule or path."""
    orch = make_orch(session="extended")
    report = orch.triage("data/claims_extended.json")
    assert report.stage == "DONE"
    expected = {
        "C-1001": ("auto_approve", set()),
        "C-1002": ("auto_approve", set()),
        "C-1003": ("auto_approve", set()),
        "C-1004": ("route_to_investigator", {"FI-01", "FI-02"}),   # 4th claim in 12 months, repeat water
        "C-1005": ("route_to_investigator", {"UW-04"}),            # loss after the policy expired
        "C-1006": ("route_to_investigator", {"UW-03", "UW-06"}),   # 18,000 on a 15,000 limit
        "C-1007": ("request_documentation", {"COV-04"}),           # jewelry is scheduled -> no COV-03
        "C-1008": ("route_to_investigator", {"COV-01"}),           # earthquake not a covered peril
        "C-1009": ("auto_approve", {"FI-04"}),                     # round amount is only a note
        "C-1010": ("request_documentation", {"UW-05"}),            # reported 76 days late
        "C-1013": ("auto_approve", set()),                         # valid, but with a warning
    }
    for cid, (action, rules) in expected.items():
        assert orch.state.decisions[cid]["final_action"] == action, cid
        assert {r["rule_id"] for r in orch.state.assessments[cid]["rules_fired"]} == rules, cid
    assert orch.state.validation["C-1013"]["status"] == "valid_with_warnings"
    assert {i["code"] for i in orch.state.validation["C-1011"]["issues"]} == {"FUTURE_DATE"}
    assert {i["code"] for i in orch.state.validation["C-1012"]["issues"]} == {"INVALID_POLICY_FORMAT"}
    assert orch.state.decisions["C-1011"]["final_action"] == "request_documentation"


def test_triage_specific_claims_by_id(make_orch):
    """'triage C-1001' finds the claim in the data files and triages only that claim."""
    orch = make_orch(session="by-id")
    msg = orch.handle("triage C-1001 and c1009")
    assert "Triage complete" in msg
    assert set(orch.state.decisions) == {"C-1001", "C-1009"}
    assert orch.state.decisions["C-1001"]["final_action"] == "auto_approve"


def test_invented_path_falls_back_to_claim_ids(make_orch):
    """A model-invented path like data/C-1001.json is corrected to a claim lookup, not a failed run."""
    orch = make_orch(session="invented")
    route = {"intent": "triage_batch", "claim_ids": [], "source_path": "data/C-1001.json", "rationale": "x"}
    msg = orch._route_triage(route, "trige c1001", "ack")
    assert "Triage complete" in msg and set(orch.state.decisions) == {"C-1001"}
    route = {"intent": "triage_batch", "claim_ids": ["C-9999"], "source_path": None, "rationale": "x"}
    assert "couldn't find C-9999" in orch._route_triage(route, "triage C-9999", "ack")
