# Test plan: Claims Triage Assistant

This plan covers every requirement in the brief, plus the error handling, security, cost and
demo-readiness checks. Each test case gives the command to run and the expected result.

## 1. Scope and approach

| Level | What | How | Mode |
|---|---|---|---|
| L1 Automated | Unit and integration tests (18) | `pytest` | Mock + fake Foundry client |
| L2 Functional, mock | Full routines, all data files, fault injection, HITL paths | CLI | Mock (`--mock`), deterministic |
| L3 Functional, live | The same routines on Azure AI Foundry with gpt-4.1-mini | CLI + Foundry portal | Live |
| L4 Non-functional | Observability, security, performance, cost | Trace files, `scripts/trace_summary.py` | Both |

**How to judge live results:** the model's wording differs from run to run. So in live mode, check
the **final actions, the state-machine path, and which tools and rules were used**, not the exact
text. Final actions are always enforced by the rule verifier, so they are stable.

### Requirement traceability

| Requirement (brief) | Test cases |
|---|---|
| Foundry Agent Service + SDK, 4 agents created in code | SET-01…06 |
| ≥3 function tools with JSON schemas, registered to the right agents | AUTO (schema/permission tests), SET-01, SEC-02 |
| Intake & Validation agent | VAL-01…15 |
| Anomaly & Coverage agent + knowledge grounding | RUL-01…20, KB-01…04 |
| Adjuster Briefing agent + memory references | BRF-01…05, MEM-06 |
| Supervisor routing | SUP-01…09 |
| Explicit routine + HITL checkpoints | RTN-01…09 |
| Short-term (thread) memory | MEM-01…05 |
| Long-term (persisted) memory | MEM-06…11 |
| Error handling | ERR-01…12 |
| Observability / tracing | OBS-01…07 |

## 2. Environment and prerequisites

```bash
cd ~/Projects/claims-triage
source .venv/bin/activate          # afterwards `python` = the project's Python
python --version                   # 3.9+
```

| Item | Mock tests | Live tests |
|---|---|---|
| `.env` | not needed (add `--mock` to commands) | `PROJECT_ENDPOINT`, `MODEL_DEPLOYMENT_NAME=gpt-4.1-mini-1`, `USE_MOCK_LLM=false` |
| Azure sign-in | – | `az login` (account with the Foundry User / Azure AI User role) |
| Foundry resources | – | `python main.py setup --live` |

> **Important:** mock and live runs share the same long-term memory file
> (`data/memory/policy_history.json`). Start each scenario with `python main.py memory reset` unless
> the test says otherwise.

**Checking results:**
- The console shows the briefings and checkpoints.
- `reports/triage_<session>_<batch>.md` holds the adjuster report.
- `logs/trace_<session>_<time>.jsonl` holds the audit trail.
- `python scripts/trace_summary.py --session <id>` summarises the latest trace for a session: state path, agent runs, tokens and cost, tool errors, guardrail discrepancies, HITL decisions and memory writes.
- `.sessions/<id>.json` holds the session state and thread IDs.

### Test data

| File | Contents | Used by |
|---|---|---|
| `data/batches/2026-02_batch.json` | C-2029, C-2032 (February): seeds long-term memory | MEM, RUL |
| `data/claims.json` | C-2031…C-2038 (September): main demo, includes a duplicate | VAL, RUL, BRF |
| `data/claims_extended.json` | C-1001…C-1013: remaining rules and paths | VAL, RUL |
| `data/samples/claims_sample_original.json` | The brief's original 4 claims, unchanged | ERR (faults) |
| `data/samples/claims_messy.csv` | Truncated and extra-column rows, bad amount/date, lowercase IDs | VAL, ERR |
| `data/policy_coverage.json` | 9 synthetic policies | – |
| `knowledge/*.md` | 18 rules (UW, COV, FI) | KB, RUL |

---

## 3. L1: Automated tests

| ID | Command | Expected |
|---|---|---|
| AUTO-01 | `python -m pytest -q` | `18 passed`, no Azure calls |

What the 16 tests cover:
- KB parsing
- that the safe evaluator rejects code
- ID normalisation
- every validation code in `claims.json`
- tolerant CSV loading
- citation-marker stripping
- tool schema and permission enforcement
- guardrail hallucinated/missed/unverified rules and downgrade blocking
- long-term memory changing C-2031's outcome
- legal state transitions
- surviving all 4 faults
- clean abort on a missing file
- multi-turn reference resolution across processes
- expected outcomes for the extended batch
- the Foundry `requires_action → submit_tool_outputs` loop
- run timeout → cancel

## 4. Setup and provisioning (live)

| ID | Steps | Expected |
|---|---|---|
| SET-01 | `python main.py setup --live` on an empty project | 3 `setup.file_uploaded`, 1 `setup.vector_store_ready`, 4 `setup.agent_created`. Coverage and briefing list `file_search`. `.foundry_state.json` holds 4 agent IDs + vector store ID |
| SET-02 | Portal: Agents | 4 agents `claims-*`, model `gpt-4.1-mini-1`. Instructions and tools match `agents.py`/`skills.py` |
| SET-03 | Run `setup --live` again with no changes | No created/updated events: "ready" only (idempotent) |
| SET-04 | Edit one prompt in `agents.py` (e.g. add a sentence to BRIEFING) → `setup --live` | Exactly one `setup.agent_updated agent=briefing`. Same agent ID. Revert and rerun afterwards |
| SET-05 | Edit a rule file (see KB-03) → `setup --live` | Old vector store deleted, 3 files re-uploaded, new vector store; coverage + briefing updated |
| SET-06 | Delete one agent in the portal → `setup --live` | `setup.agent_missing` warning, then `setup.agent_created` for that agent |
| SET-07 | `python main.py teardown --live`, then check the portal | Agents, vector store and files gone. `.foundry_state.json` emptied. Run SET-01 again to restore |

## 5. Intake and validation

Run `python main.py memory reset`, then `python main.py triage data/claims.json --yes` (plus `--mock` for the deterministic variant).

| ID | Claim / row | Expected issue | Result |
|---|---|---|---|
| VAL-01 | C-2034 | MISSING_FIELD (loss_date) | invalid → documents |
| VAL-02 | C-2034 | NON_POSITIVE_AMOUNT (0) | invalid |
| VAL-03 | C-2035 | LOSS_AFTER_REPORT | invalid |
| VAL-04 | C-2036 | UNKNOWN_POLICY (POL-0000) | invalid |
| VAL-05 | C-2031 row 7 | DUPLICATE_CLAIM_ID; row 1 gets a DUPLICATE_SEEN warning and continues | row 7 rejected |
| VAL-06 | C-2031, C-2033, C-2037, C-2038 | none | valid → go to COVERAGE |

`python main.py memory reset`, then `python main.py triage data/claims_extended.json --yes`:

| ID | Claim | Expected |
|---|---|---|
| VAL-07 | C-1011 | FUTURE_DATE (both dates in 2027) → invalid |
| VAL-08 | C-1012 | INVALID_POLICY_FORMAT (`PL-1103`) → invalid |
| VAL-09 | C-1013 | MISSING_DESCRIPTION warning → status `valid_with_warnings`, still assessed |

`python main.py triage data/samples/claims_messy.csv --yes`:

| ID | Row | Expected |
|---|---|---|
| VAL-10 | row 4 (C-2043) | Unparseable: MISSING_COLUMNS, skipped, listed in the report |
| VAL-11 | row 6 (C-2045) | Unparseable: EXTRA_COLUMNS, skipped |
| VAL-12 | C-2041 | INVALID_AMOUNT (`12k`) |
| VAL-13 | C-2042 | INVALID_DATE (`2026/08/40`) |
| VAL-14 | `c2044`, `pol-9981` | Normalised to `C-2044` / `POL-9981`, valid |
| VAL-15 | Triage `data/claims.json` a 2nd time without resetting memory | Finalised claims rejected as ALREADY_PROCESSED. Claims saved as `incomplete` (e.g. C-2034) are **not** blocked |

## 6. Coverage and anomaly rules (one case per rule)

Preconditions: memory reset, then run the February batch with `--as-of 2026-02-20`, then the
September batch and the extended batch. Check the "Rules:" line of each briefing, or `reports/`.

| ID | Rule | Positive case | Negative case | Expected action |
|---|---|---|---|---|
| RUL-01 | UW-01 early inception (≤7 days) | C-2033 (3 days) | C-2031 | investigator |
| RUL-02 | UW-02 98–100% of limit | C-2033 (100%) | C-1006 (120% → UW-03 instead) | investigator |
| RUL-03 | UW-03 exceeds limit | C-1006 (18,000 / 15,000) | – | investigator |
| RUL-04 | UW-04 loss outside policy period | C-1005 (policy expired 2026-03-01) | – | investigator |
| RUL-05 | UW-05 reported >30 days late | C-2038 (53 d), C-1010 (76 d) | C-2031 (1 d) | documents |
| RUL-06 | UW-06 amount >10,000 | C-2038, C-1006, C-2033 | C-2037 | documents |
| RUL-07 | UW-07 auto-approval | C-2037, C-1001–1003, C-1013 | – | auto-approve |
| RUL-08 | UW-08 unknown policy | Not reachable end to end (UNKNOWN_POLICY rejects first). Defence in depth; verify with the evaluator: `python -c "from claims_triage.knowledge import *; print(evaluate_condition('policy_found == False', {'policy_found': False}))"` → `True` | – | – |
| RUL-09 | COV-01 peril not covered | C-1008 (earthquake) | C-2037 (wind_hail) | investigator |
| RUL-10 | COV-02 flood exclusion | C-2032 ("Basement flooding") | C-2031 ("Pipe burst") | documents |
| RUL-11 | COV-03 jewelry sub-limit | C-2033 (not scheduled) | **C-1007 (jewelry scheduled → must NOT fire)** | documents |
| RUL-12 | COV-04 theft >1,000 needs a police report | C-2033, C-1007 | C-2040 (950, CSV) | documents |
| RUL-13 | FI-01 repeat peril within 12 months | C-2031 (after Feb run), C-1004 | C-1001 | investigator |
| RUL-14 | FI-02 ≥3 prior claims | C-1004 (3 prior in batch) | C-1003 (2 prior) | investigator |
| RUL-15 | FI-03 previously flagged | C-2031 (C-2032 flagged) | C-2038 (prior C-2029 was auto-approved) | documents |
| RUL-16 | FI-04 round amount (note) | C-1009 (5,000) | – | **stays auto-approve** (a note never escalates) |
| RUL-17 | FI-05 same-day high-value theft (note) | C-2033 | C-1007 (4,000) | note only |
| RUL-18 | FI-06 judgement (no condition) | Live only; if cited, shown as `[model_judgement]` and never escalates | – | – |
| RUL-19 | Most-severe action wins | C-2033: investigator despite 3 document rules | – | investigator |
| RUL-20 | Invalid claims skip coverage | C-2034: no coverage facts, briefing notes that POL-4410 had expired | – | documents |

Deterministic check of all of the above in one command: `python -m pytest -q -k "extended or long_term"`.

## 7. Knowledge grounding

| ID | Steps | Expected |
|---|---|---|
| KB-01 | Live triage, then `python scripts/trace_summary.py --session <id>` | `file_search ≥ 1`. The console shows `tool.file_search ... files=["coverage_matrix.md","fraud_indicators.md","underwriting_rules.md"]` |
| KB-02 | `grep -nE "<= ?7\|10000\|1500\|C-20[0-9]{2}\|2026-" claims_triage/agents.py claims_triage/skills.py claims_triage/guardrails.py` | No matches: no rule thresholds and no demo answers in the prompts or tool code. Thresholds exist only in `knowledge/*.md`. (`knowledge.py` docstrings contain examples; that's expected) |
| KB-03 | **Rules live in the KB, not in code:** in `knowledge/underwriting_rules.md` change UW-06 to `claim_amount > 1500`. Mock: `memory reset` then `triage data/claims.json --mock --yes`. Live: `setup --live` first | C-2037 (1,800) changes from auto-approve to **request_documentation**, with UW-06 cited. **Revert** the file (and rerun `setup --live`) |
| KB-04 | Live: ask "why was C2031 flagged?" | The answer cites rule IDs (FI-01, FI-03) and paraphrases the rule text from the KB |

## 8. Guardrails (the model proposes, the rules verify)

| ID | Steps | Expected |
|---|---|---|
| GRD-01 | `python main.py triage data/samples/claims_sample_original.json --mock --yes --fault hallucination` | `guardrail.discrepancy type=hallucinated_rule rule_id=UW-99` and `type=missed_rule`. The briefing shows `⚠ Guardrail: hallucinated_rule:UW-99; missed_rule:…`. The final action is still correct |
| GRD-02 | Live September run → `trace_summary.py` | The discrepancy count is reported. Rules added by the verifier show as `[added_by_verifier]` in the briefing (e.g. on C-2033). Final actions match section 6 |
| GRD-03 | Automated: condition not met / downgrade | Covered by `test_guardrail_catches_hallucination_miss_and_downgrade` |
| GRD-04 | Inspect any briefing | Every rule shows its status: `verified`, `added_by_verifier`, `unverified` or `model_judgement` |

## 9. Adjuster briefing

| ID | Steps | Expected |
|---|---|---|
| BRF-01 | September run after the February run | C-2031 summary mentions C-2032 **"flagged … 6 months"** earlier |
| BRF-02 | Any `request_documentation` claim | `documents_requested` is concrete (e.g. police report, plumber report) |
| BRF-03 | Invalid claim C-2034 | Headline says the submission is incomplete; lists the fields to correct; notes the policy expired |
| BRF-04 | All briefings | `recommended_action` is never lower than the verified floor (the guardrail notes would say so) |
| BRF-05 | `reports/triage_<session>_<batch>.md` | Summary table + one section per claim + unparseable rows (for the CSV) |

## 10. Supervisor routing and multi-turn conversation

Run these on one session in live mode, then repeat with `--mock`. Start with `python main.py chat --session t-sup`.

| ID | Input | Expected |
|---|---|---|
| SUP-01 | `triage data/claims.json` | `routine.route intent=triage_batch source_path=data/claims.json`, and the full routine runs |
| SUP-02 | `why was C2031 flagged?` | intent `explain_claim`, claim ID normalised to `C-2031`; the answer cites FI-01/FI-03 + C-2032 |
| SUP-03 | `what about the theft one?` | Resolved to **C-2033** from context (no ID typed) |
| SUP-04 | `is it still pending?` | "it" = C-2033 (last claim discussed) |
| SUP-05 | `show the history for POL-5521` | intent `policy_history`; lists C-2032 (+ C-2031) from long-term memory |
| SUP-06 | `hi, what can you do?` | A short capability answer, not "handing off…" (routes `general_question` or answers directly) |
| SUP-07 | `why was C-9999 flagged?` | "C-9999 has not been triaged in this session or any earlier run." No crash |
| SUP-08 | In a **new** session: `why was it flagged?` | Asks which claim, or explains usage. No crash |
| SUP-09 | `trige c1001` (typo, no file named) | `routine.route` has `claim_ids=["C-1001"]` and no path; `routine.claim_located` finds it in `claims_extended.json`; only C-1001 is triaged. A model-invented path is corrected (`routine.route_corrected`); an unknown ID gets "I couldn't find …" |

## 11. Routine and human-in-the-loop checkpoints

| ID | Steps | Expected |
|---|---|---|
| RTN-01 | `python main.py routine` | Prints the state machine and both checkpoints |
| RTN-02 | Any complete triage → `trace_summary.py` | Path `INTAKE→VALIDATE→GATE_VALIDATION→COVERAGE→BRIEF→GATE_ADJUSTER→PERSIST→DONE` |
| RTN-03 | Same → `.sessions/<id>.json` → `routine_checkpoints` | One checkpoint per transition with a timestamp |
| RTN-04 | **Gate 1 abort:** `printf 'a\n' \| python main.py triage data/claims.json --session t-abort` | Stage `ABORTED`, "Aborted by handler at the validation checkpoint", no `memory.write`, exit code 1 |
| RTN-05 | **Gate 2 override:** `printf 'c\no\nC-2037\nrequest_documentation\nNeed contractor quote\n\n' \| python main.py triage data/claims.json --session t-ovr` | `hitl.override claim_id=C-2037`; memory entry has `decided_by: adjuster_override` and the reason, with its capitalisation kept |
| RTN-06 | Override with an invalid action, then with an empty reason | "invalid action" / "a reason is required", then asks again. Nothing is recorded until the input is valid |
| RTN-07 | **Gate 2 defer:** `printf 'c\nd\n' \| python main.py triage data/claims.json --session t-defer` | Stage `DONE` (deferred), no `memory.write`, `python main.py memory show` unchanged |
| RTN-08 | `--yes` | Both checkpoints print `(auto)`, decided_by `auto_gate` |
| RTN-09 | All claims invalid (e.g. a file with only C-2035 + C-2036) | Gate 1 → **BRIEF** directly (COVERAGE skipped: "no valid claims") |

## 12. Memory

**Short-term (thread level)**

| ID | Steps | Expected |
|---|---|---|
| MEM-01 | Any triage | VALIDATE message has no batch_id, yet Intake validates B-00x (recalled from its thread) |
| MEM-02 | `/state` in chat | Shows 4 thread IDs (`thread_…` live / `mock-thread-…` mock) |
| MEM-03 | SUP-03/SUP-04 | References resolved from the conversation |
| MEM-04 | Exit chat, then `python main.py ask --session t-sup "and the water one?"` | New process, same threads → C-2031 resolved |
| MEM-05 | Session created in mock, then reused with `--live` | `memory.session_mode_changed` warning; threads reset, no crash |

**Long-term (persisted across runs)**

| ID | Steps | Expected |
|---|---|---|
| MEM-06 | `memory reset` → September batch only | C-2031 = **auto-approve** (no history) |
| MEM-07 | `memory reset` → February (`--as-of 2026-02-20`) → September | C-2031 = **route_to_investigator** (FI-01, FI-03) |
| MEM-08 | `python main.py memory show` after MEM-07 | Entries grouped by policy, with `final_action`, `rules_fired`, `decided_by`, `triaged_on`, `status` |
| MEM-09 | After RTN-07 (defer) | Nothing written |
| MEM-10 | Invalid claims | Stored with `status: incomplete`, so a corrected resubmission isn't blocked (see VAL-15) |
| MEM-11 | Corrupt it: `echo '{bad' > data/memory/policy_history.json` → any triage | `memory.corrupt` warning, backup `policy_history.corrupt.json`, run continues with empty memory |

## 13. Error handling and resilience

| ID | Steps | Expected |
|---|---|---|
| ERR-01 | `triage data/nope.json` | ABORTED: "Could not ingest … No claims file" |
| ERR-02 | Broken JSON file (`echo '[{"claim_id": "C-1",}' > /tmp/bad.json`) | ABORTED with the line and column of the JSON error |
| ERR-03 | Unsupported type (`/tmp/claims.txt`) | ABORTED: unsupported_format |
| ERR-04 | Messy CSV | Bad rows skipped and reported; the rest processed (VAL-10…14) |
| ERR-05 | `--fault tool_timeout` (mock **and** live) | `tool.retry … attempt=1`, then success; the run completes |
| ERR-06 | `--mock --fault bad_json` | `agent.bad_output` warning → repair prompt → valid JSON → run completes |
| ERR-07 | `--mock --fault agent_timeout` | `agent.run.failed attempt=1` → attempt 2 succeeds |
| ERR-08 | `--mock --fault hallucination` | See GRD-01 |
| ERR-09 | All four faults together on `claims_sample_original.json --mock --yes` | Stage DONE; the report lists the guardrail notes |
| ERR-10 | Tool misuse (bad args / wrong agent) | Automated: `invalid_arguments`, `tool_not_permitted` returned as JSON; never crashes |
| ERR-11 | Live rate limit (optional): temporarily set the deployment capacity to 1K TPM | `agent.rate_limited` with backoff; or `agent.run.failed`, then retry. Restore the capacity afterwards |
| ERR-12 | Chat resilience: press Ctrl+C at the prompt | "Session saved: … (resume with --session …)" |

## 14. Observability

| ID | Steps | Expected |
|---|---|---|
| OBS-01 | Any run | Console events with icons (◆ routine, ● agent, ↳ tool, ⛨ guardrail, ✋ HITL, ▣ memory) |
| OBS-02 | `-v` | Full untruncated fields, including tool `args` |
| OBS-03 | `python scripts/trace_summary.py --session <id>` | Path, per-agent tokens, cost, tool errors, discrepancies, HITL, memory writes |
| OBS-04 | JSONL | Every line has `ts`, `session`, `level`, `event` |
| OBS-05 | `OTEL_CONSOLE_EXPORTER=true python main.py ask --session t-sup "why was C2031 flagged?"` | OpenTelemetry spans printed (`supervisor.turn`, `agent.*`, `tool.*`) |
| OBS-06 | Live + `ENABLE_TRACING=true` (App Insights connected to the project) | `setup.tracing exporter=azure_monitor` + `instrumentor=AIAgentsInstrumentor`; Foundry portal → Tracing shows the runs within a few minutes |
| OBS-07 | `ENABLE_TRACING=true` without App Insights | Warning event only; the run continues (tracing fails soft) |

## 15. Security and safety

| ID | Steps | Expected |
|---|---|---|
| SEC-01 | Evaluator injection | Automated: `__import__('os')…` → `UnsafeExpression` |
| SEC-02 | Tool least privilege | Automated: intake can't call `check_coverage` |
| SEC-03 | `grep -rniE "api[_-]?key\|secret\|password" --include=*.py .` (excluding `.venv`) | No credentials in code; auth is `DefaultAzureCredential` |
| SEC-04 | `git check-ignore .env .foundry_state.json .sessions logs` | All ignored |
| SEC-05 | Default `.env` | `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=false` (no claim text in traces) |
| SEC-06 | Data review | All data is synthetic, and the knowledge files are marked "SYNTHETIC" |

## 16. Performance and cost (live)

| ID | Steps | Target (measured on 2026-09-28) |
|---|---|---|
| PRF-01 | `time python main.py triage data/claims.json --yes` | ≤ 90 s (measured ~62–75 s) |
| PRF-02 | `trace_summary.py` after PRF-01 | ≤ $0.03 per triage (measured ~$0.014; ~24–30K input, 1–2.5K output tokens) |
| PRF-03 | Extended batch (13 claims, 3 coverage chunks at chunk size 5) | Completes with no rate limiting at 100K TPM |
| PRF-04 | Chat question | ≤ 20 s, ≤ $0.01 |

## 17. Pre-demo run-sheet (~10 minutes, live)

```bash
source .venv/bin/activate
python -m pytest -q                                                        # 18 passed
python main.py setup --live                                                # "ready", nothing created
python main.py memory reset
python main.py triage data/batches/2026-02_batch.json --as-of 2026-02-20 --session demo-feb --yes
python main.py triage data/claims.json --session demo --yes                # C-2031 → investigator
python main.py ask --session demo "why was C2031 flagged?"                 # mentions C-2032, 6 months
python scripts/trace_summary.py --session demo
python main.py memory reset                                                # clean slate for the live demo
```

## 18. Exit criteria and defect logging

**Pass criteria:**
- All AUTO tests pass.
- Every mock case passes exactly.
- Live cases match on final actions, state path and tool usage.
- Guardrail discrepancies are acceptable if the final actions are correct. Record the rate as a quality metric.

**Defect record:** ID · test case · mode (mock/live) · session ID · trace file · expected · actual · severity.

**Known limitations (not defects):**
- UW-08 can't be reached end to end.
- FI-06 only appears live, and not every time.
- Live wording varies between runs.
- `--fault bad_json/hallucination/agent_timeout` only work with `--mock`.
