#!/usr/bin/env python3
"""Claims Triage Assistant: CLI entry point.

COMMANDS
  python main.py chat                               multi-turn conversation via the Supervisor
  python main.py ask "why was C2031 flagged?"       single turn (use --session to continue a conversation)
  python main.py triage data/claims.json            run the triage routine directly (skips the Supervisor)
  python main.py routine                            print the routine's state machine
  python main.py memory show|reset                  inspect / clear long-term memory
  python main.py setup | teardown                   create / delete Foundry agents + vector store

GLOBAL FLAGS
  --mock / --live        force mock mode or Azure AI Foundry (default: mock if PROJECT_ENDPOINT unset)
  --session ID           reuse a session (same agent threads = short-term memory across processes)
  --as-of YYYY-MM-DD     the date treated as "today" in date checks
  --yes                  auto-accept the two human-in-the-loop checkpoints
  --fault NAME           inject a failure: bad_json | hallucination | agent_timeout | tool_timeout
  -v                     verbose logging

SUGGESTED READING ORDER FOR THE CODE
  1. main.py                      how everything is wired together (build())
  2. claims_triage/orchestrator.py  the Supervisor flow + the explicit triage state machine
  3. claims_triage/agents.py      the four agents and their instructions
  4. claims_triage/skills.py      the function tools (with JSON schemas) and the safe executor
  5. claims_triage/guardrails.py  how model output is verified against the knowledge base
  6. claims_triage/knowledge.py   rule parsing + the safe condition evaluator
  7. claims_triage/memory.py      short-term (session/thread) and long-term (cross-run) memory
  8. claims_triage/runtime/       foundry.py (Azure) and mock.py (offline), behind one interface
  9. claims_triage/observability.py  console + JSONL + OpenTelemetry logging
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from claims_triage import observability
from claims_triage.config import Settings
from claims_triage.data_io import load_policies
from claims_triage.knowledge import load_rules
from claims_triage.memory import LongTermMemory, SessionStore
from claims_triage.orchestrator import HumanGate, Orchestrator, TriageReport, render_routine
from claims_triage.runtime import build_runtime
from claims_triage.skills import SkillRegistry, ToolContext

BANNER = """
╭──────────────────────────────────────────────────────────────╮
│  Claims Triage Assistant · Supervisor + 3 specialist agents  │
│  mode: {mode:<10} session: {session:<28}│
╰──────────────────────────────────────────────────────────────╯
Try: "triage data/claims.json" · "triage C-1001" · "why was C2031 flagged?" · "what about the theft one?"
     "show the history for POL-5521" · /routine · /memory · /quit
"""


def build(args) -> Orchestrator:
    """Wire up the whole system. This is the composition root: every component is created here.

        Settings -> logging -> session (short-term memory) -> ToolContext (session + long-term
        memory + policies + rules) -> SkillRegistry (tools) -> runtime (Foundry or mock)
        -> Orchestrator (+ HumanGate) -> agents provisioned.
    """
    # 1. Configuration: environment/.env first, then CLI flags override.
    settings = Settings.from_env()
    if args.mock:
        settings.use_mock = True
    if args.live:
        settings.use_mock = False
    settings.faults |= set(args.fault or [])
    if args.as_of:
        settings.as_of = date.fromisoformat(args.as_of)
    session_id = args.session or f"s-{datetime.now():%Y%m%d-%H%M%S}"
    mode = "mock" if settings.use_mock else "foundry"

    # 2. Logging/tracing, then the session (resumed if --session names an existing one).
    observability.setup(settings, session_id, verbose=args.verbose)
    store = SessionStore(settings.sessions_dir)
    state = store.load_or_create(session_id, settings.as_of.isoformat(), mode)
    if args.as_of:
        state.as_of = args.as_of

    # 3. Everything the tools need: session memory, long-term memory, policy register, KB rules.
    ctx = ToolContext(
        settings=settings,
        state=state,
        memory=LongTermMemory(settings.memory_path),
        policies=load_policies(settings.policies_path),
        rules=load_rules(settings.knowledge_dir),
    )
    registry = SkillRegistry(ctx)

    # 4. Agent backend: FoundryRuntime (live) or MockRuntime (offline).
    runtime = build_runtime(settings, state)
    observability.event("setup.start", mode=mode, session=session_id, as_of=state.as_of,
                        model=None if settings.use_mock else settings.model_deployment,
                        faults=sorted(settings.faults) or None)

    # 5. The orchestrator, with a console human-in-the-loop gate; then create/update the agents.
    orch = Orchestrator(runtime, registry, ctx, store, HumanGate(auto=args.yes))
    orch.setup()
    return orch


def write_report(orch: Orchestrator, report: TriageReport) -> Path:
    """Write a Markdown adjuster report for one triage run to reports/ and return its path."""
    s = orch.ctx.settings
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    path = s.reports_dir / f"triage_{orch.state.session_id}_{report.batch_id or 'none'}.md"
    # Header + summary table (one row per claim).
    lines = [f"# Triage report · {report.source}", "",
             f"- Session: `{orch.state.session_id}` · as-of {orch.state.as_of} · mode {orch.rt.mode}",
             f"- Final stage: **{report.stage}** · persisted to long-term memory: {report.persisted}",
             f"- {report.message}", "",
             "| Claim | Recommendation | Adjuster decision | Rules | Memory |", "|---|---|---|---|---|"]
    for b in report.briefings:
        d = report.decisions.get(b["claim_id"], {})
        rules = ", ".join(r["rule_id"] for r in (orch.state.assessments.get(b["claim_id"]) or {}).get("rules_fired", []))
        lines.append(f"| {b['claim_id']} | {b['recommended_action']} | {d.get('final_action', '-')} "
                     f"({d.get('decided_by', '-')}) | {rules or '-'} | {'; '.join(b.get('memory_references', [])) or '-'} |")
    lines.append("")
    # One section per claim with the full briefing.
    for b in report.briefings:
        lines += [f"## {b['claim_id']} · {b.get('headline', '')}", "", b.get("summary", ""), ""]
        if b.get("documents_requested"):
            lines += ["**Documents to request:** " + "; ".join(b["documents_requested"]), ""]
        if b.get("guardrail_notes"):
            lines += ["**Guardrail notes:** " + "; ".join(b["guardrail_notes"]), ""]
    # Rows that couldn't even be parsed.
    if report.validation.get("unparseable"):
        lines += ["## Unparseable rows", ""] + [f"- row {e['row']}: {e['code']} ({e['message']})"
                                                for e in report.validation["unparseable"]]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------
# Command handlers (one per sub-command). Each returns a process exit code.
# ---------------------------------------------------------------------------------------------
def cmd_triage(args) -> int:
    """Run the triage routine directly on a file (no Supervisor routing)."""
    orch = build(args)
    report = orch.triage(args.source)
    print(f"\n{report.message or 'Finished at ' + report.stage}")
    if report.briefings:
        print(f"Report: {write_report(orch, report)}")
    print(f"Trace:  {observability.trace_file()}")
    return 0 if report.stage == "DONE" else 1


def _handle(orch: Orchestrator, text: str) -> str:
    """One Supervisor turn (routing + the matching routine)."""
    answer = orch.handle(text)
    return answer


def cmd_ask(args) -> int:
    """Single question. With --session, it continues an earlier conversation (thread memory)."""
    orch = build(args)
    print("\n" + _handle(orch, " ".join(args.text)))
    return 0


def cmd_chat(args) -> int:
    """Interactive multi-turn REPL. Slash commands inspect the system without calling any agent."""
    orch = build(args)
    print(BANNER.format(mode=orch.rt.mode, session=orch.state.session_id))
    while True:
        try:
            text = input("\nhandler › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit", "quit", "exit"):
            break
        if text == "/routine":           # show the state machine
            print(render_routine())
            continue
        if text == "/memory":            # show long-term memory
            print(json.dumps(orch.memory.dump(), indent=2))
            continue
        if text == "/state":             # show short-term/session memory
            st = orch.state
            print(json.dumps({"session": st.session_id, "turns": st.turns, "threads": st.threads,
                              "claims": list(st.validation), "decisions": st.decisions,
                              "checkpoints": st.routine_checkpoints[-5:]}, indent=2, default=str))
            continue
        try:
            print("\nassistant › " + _handle(orch, text))
        except Exception as exc:  # keep the conversation alive
            observability.event("chat.error", level="error", error=repr(exc))
            print(f"\nassistant › Sorry, that request failed ({exc}). The session is intact; please try again.")
    print(f"Session saved: {orch.state.session_id} (resume with --session {orch.state.session_id})")
    return 0


def cmd_memory(args) -> int:
    """Show or reset long-term memory (data/memory/policy_history.json)."""
    settings = Settings.from_env()
    mem = LongTermMemory(settings.memory_path)
    if args.action == "reset":
        mem.reset()
        print(f"Long-term memory cleared ({settings.memory_path}).")
    else:
        print(json.dumps(mem.dump(), indent=2))
    return 0


def cmd_setup(args) -> int:
    """Provision agents + knowledge vector store (build() already does this; this just reports it)."""
    build(args)
    print("Agents and knowledge vector store are ready.")
    return 0


def cmd_teardown(args) -> int:
    """Delete the Foundry resources recorded in .foundry_state.json."""
    settings = Settings.from_env()
    if settings.use_mock and not args.live:
        print("Mock mode: nothing to tear down.")
        return 0
    from claims_triage.runtime.foundry import FoundryRuntime

    observability.setup(settings, "teardown")
    FoundryRuntime(settings).teardown()
    print("Deleted Foundry agents, vector store and uploaded files.")
    return 0


def main(argv=None) -> int:
    # Flush stdout line by line so user-facing output and stderr logs appear in the right order.
    sys.stdout.reconfigure(line_buffering=True)

    # Flags shared by most sub-commands.
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="use the deterministic mock LLM runtime")
    g.add_argument("--live", action="store_true", help="use Azure AI Foundry (requires PROJECT_ENDPOINT)")
    common.add_argument("--session", help="session id; reuse it to continue a conversation across runs")
    common.add_argument("--as-of", help="reference date for date checks (YYYY-MM-DD)")
    common.add_argument("--yes", "-y", action="store_true", help="auto-accept HITL checkpoints")
    common.add_argument("--fault", action="append",
                        choices=["bad_json", "hallucination", "agent_timeout", "tool_timeout"],
                        help="inject a failure to demo error handling (repeatable)")
    common.add_argument("-v", "--verbose", action="store_true")

    # Sub-commands: each sets `fn` to its handler.
    p = argparse.ArgumentParser(description="Multi-agent Claims Triage Assistant (Azure AI Foundry)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("triage", parents=[common], help="run the triage routine on a claims file")
    sp.add_argument("source", nargs="?", default="data/claims.json")
    sp.set_defaults(fn=cmd_triage)
    sp = sub.add_parser("chat", parents=[common], help="interactive multi-turn session")
    sp.set_defaults(fn=cmd_chat)
    sp = sub.add_parser("ask", parents=[common], help="one supervisor turn")
    sp.add_argument("text", nargs="+")
    sp.set_defaults(fn=cmd_ask)
    sp = sub.add_parser("routine", help="print the triage state machine")
    sp.set_defaults(fn=lambda a: print(render_routine()) or 0)
    sp = sub.add_parser("memory", help="show or reset long-term memory")
    sp.add_argument("action", choices=["show", "reset"])
    sp.set_defaults(fn=cmd_memory)
    sp = sub.add_parser("setup", parents=[common], help="create/update Foundry agents + vector store")
    sp.set_defaults(fn=cmd_setup)
    sp = sub.add_parser("teardown", parents=[common], help="delete Foundry resources created by setup")
    sp.set_defaults(fn=cmd_teardown)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
