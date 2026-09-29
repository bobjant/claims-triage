"""Shared pytest fixtures.

`settings`   : a Settings object pointing all writable paths (memory, sessions, logs) at a temp dir.
`make_orch`  : builds a fully wired Orchestrator in MOCK mode (same wiring as main.build()), with the
               human checkpoints on auto-accept, so tests can run whole routines end to end.
"""
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claims_triage import observability  # noqa: E402
from claims_triage.config import Settings  # noqa: E402
from claims_triage.data_io import load_policies  # noqa: E402
from claims_triage.knowledge import load_rules  # noqa: E402
from claims_triage.memory import LongTermMemory, SessionStore  # noqa: E402
from claims_triage.orchestrator import HumanGate, Orchestrator  # noqa: E402
from claims_triage.runtime.mock import MockRuntime  # noqa: E402
from claims_triage.skills import SkillRegistry, ToolContext  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    s = Settings(use_mock=True, as_of=date(2026, 9, 28))
    s.memory_path = tmp_path / "memory.json"
    s.sessions_dir = tmp_path / "sessions"
    s.logs_dir = tmp_path / "logs"
    s.reports_dir = tmp_path / "reports"
    observability.setup(s, "test")
    return s


@pytest.fixture
def make_orch(settings):
    def _make(session="t", as_of="2026-09-28", faults=(), printer=lambda *a, **k: None):
        settings.faults = set(faults)
        store = SessionStore(settings.sessions_dir)
        state = store.load_or_create(session, as_of, "mock")
        state.as_of = as_of
        ctx = ToolContext(settings=settings, state=state, memory=LongTermMemory(settings.memory_path),
                          policies=load_policies(settings.policies_path), rules=load_rules(settings.knowledge_dir))
        registry = SkillRegistry(ctx)
        orch = Orchestrator(MockRuntime(settings, state), registry, ctx, store, HumanGate(auto=True, print_fn=printer),
                            printer=printer)
        orch.setup()
        return orch

    return _make
