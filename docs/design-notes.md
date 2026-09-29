# Design notes and likely review questions

## Why isn't the whole flow an LLM agent with sub-agents?
Claims triage is a regulated decision process. An auditor has to be able to answer "which steps ran,
in what order, and who approved it?" without re-running a model. So the routine is a
`TRANSITIONS` table, and each step is logged, check-pointed and traced. The LLM is used where it adds
value:
- understanding the handler's request (routing and resolving references),
- retrieving and applying rules to facts,
- writing an adjuster-ready narrative that brings in history.

## Why compute facts in code but keep rules in the knowledge base?
- Facts (date arithmetic, % of limit, prior-claim counts) are things LLMs get wrong and code gets
  right.
- Rules change often and belong to underwriting, not engineering. Keeping them in markdown lets them
  be versioned, reviewed and retrieved (grounding), and the same text is what the model reads.

## If the rules are machine-checkable, why use the LLM at all?
- **Coverage:** retrieval plus judgement for soft indicators (FI-06, narrative inconsistencies),
  explanations in plain language, and coverage status.
- **Verifier:** a safety net that makes the system robust to model errors. In production you'd track
  the discrepancy rate as a quality metric (it's already in the trace as `guardrail.discrepancy`).

## Memory design
| Scope | Mechanism | Lifetime | Example |
|---|---|---|---|
| Turn | tool outputs in the run | one run | check_coverage facts |
| Thread (short-term) | Foundry thread per agent per session; IDs in `.sessions/<id>.json` | session, across processes | Intake recalls its `batch_id`; Supervisor resolves "the theft one" |
| Working state | `SessionState` (validation, assessments, briefings, decisions, checkpoints) | session | `get_triage_record` |
| Long-term | `data/memory/policy_history.json`, written after HITL approval | across runs | "similar water damage claim flagged 6 months earlier" |

## Scaling to 10k claims per day
- Intake and validation are pure code, so they run in bulk without the LLM.
- Coverage facts are pure code. Send only the claims that need judgement to the Coverage agent, in
  chunks (`coverage_chunk_size`).
- Run chunks in parallel (async client), with a queue for HITL.
- Replace JSON memory with Azure AI Search and the session store with Cosmos DB.

## Security and privacy
- Tools are allow-listed per agent, and the executor rejects cross-agent calls.
- The rule evaluator is an AST whitelist interpreter with no `eval` and no attribute access or calls.
- Content recording in traces is off by default (`AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED`),
  because claim narratives can contain personal data.
- `DefaultAzureCredential` (Entra ID) is used; there are no API keys in code.

## Cost
Per claim in live mode: about 1 coverage call (chunked) and 1 briefing call (batched), plus 2 intake
calls per batch. You can use a smaller model (e.g. gpt-4o-mini) for Supervisor routing and Intake,
where the work is mostly tool calling.
