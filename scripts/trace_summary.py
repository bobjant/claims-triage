#!/usr/bin/env python3
"""Summarise a JSONL trace file from logs/ (used by the test plan for performance, cost and
guardrail checks).

    python scripts/trace_summary.py                 # latest trace in logs/
    python scripts/trace_summary.py logs/trace_X.jsonl
    python scripts/trace_summary.py --session demo  # latest trace of a given session

Prices default to gpt-4.1-mini Global Standard (USD per 1M tokens); override with --in/--out.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pick_file(args) -> Path:
    if args.file:
        return Path(args.file)
    pattern = f"trace_{args.session}_*.jsonl" if args.session else "trace_*.jsonl"
    files = sorted((ROOT / "logs").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"No trace files matching logs/{pattern}")
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--session")
    ap.add_argument("--in", dest="price_in", type=float, default=0.40, help="USD per 1M input tokens")
    ap.add_argument("--out", dest="price_out", type=float, default=1.60, help="USD per 1M output tokens")
    args = ap.parse_args()

    path = pick_file(args)
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by = Counter(e["event"] for e in events)

    ts = [datetime.fromisoformat(e["ts"]) for e in events]
    print(f"Trace: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"Events: {len(events)}   wall time: {(max(ts) - min(ts)).total_seconds():.1f}s")

    # State machine path
    path_ = [f"{e['frm']}->{e['to']}" for e in events if e["event"] == "routine.transition"]
    if path_:
        print("Routine: " + "  ".join(path_))

    # Agent runs + tokens
    tok_in = tok_out = 0
    print("\nAgent runs:")
    for e in events:
        if e["event"] == "agent.run.end":
            u = e.get("usage") or {}
            tok_in += u.get("prompt_tokens", 0)
            tok_out += u.get("completion_tokens", 0)
            print(f"  {e['agent']:<11} in={u.get('prompt_tokens', 0):>6} out={u.get('completion_tokens', 0):>5}  tools={e.get('tool_calls')}")
    failed = [e for e in events if e["event"] == "agent.run.failed"]
    cost = tok_in * args.price_in / 1e6 + tok_out * args.price_out / 1e6
    print(f"  TOTAL      in={tok_in:>6} out={tok_out:>5}  cost≈${cost:.4f}   failed runs={len(failed)}")

    # Tools
    calls = [e for e in events if e["event"] == "tool.call"]
    errors = [e for e in calls if not e.get("ok")]
    print(f"\nTool calls: {len(calls)} (errors: {len(errors)})   retries: {by['tool.retry']}   "
          f"file_search: {by['tool.file_search']}")
    for e in errors:
        print(f"  ✗ {e['tool']}: {e.get('error')} - {e.get('message')}")

    # Guardrails
    disc = Counter(e["type"] for e in events if e["event"] == "guardrail.discrepancy")
    verified = [e for e in events if e["event"] == "guardrail.verified"]
    print(f"\nGuardrail: {len(verified)} claims verified, {sum(disc.values())} discrepancies {dict(disc)}")
    if verified:
        flagged = sum(1 for e in verified if e.get("discrepancies"))
        print(f"  claims with ≥1 discrepancy: {flagged}/{len(verified)}")

    # Fallbacks, HITL, memory
    for name in ("routine.fallback", "agent.bad_output", "hitl.decision", "hitl.override", "memory.write",
                 "routine.error", "mock.fault_injected"):
        if by[name]:
            print(f"{name}: {by[name]}")


if __name__ == "__main__":
    main()
