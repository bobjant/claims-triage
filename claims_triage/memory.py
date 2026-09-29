"""Memory: short-term (per session) and long-term (across runs).

TWO KINDS OF MEMORY (both required by the brief)

  1. SHORT-TERM / THREAD-LEVEL: `SessionState` + `SessionStore`
     * Each agent gets ONE conversation thread per session (a Foundry thread in live mode). Thread
       IDs are saved in .sessions/<session_id>.json, so running `main.py ask --session demo ...`
       again, even as a brand-new process, re-attaches to the same threads and the agents "remember"
       the earlier conversation.
     * The session file also holds the routine's structured working data (validation results,
       assessments, briefings, decisions, checkpoints) so follow-up questions can be answered.

  2. LONG-TERM / CROSS-RUN: `LongTermMemory`
     * A JSON file of claim outcomes keyed by policy number (data/memory/policy_history.json).
     * Written ONLY after the adjuster approves at checkpoint 2, so it stores human-confirmed
       decisions, never raw model guesses.
     * Read by the `check_coverage` skill (to count prior claims) and the `get_policy_history` skill.
       That's where "this policyholder had a similar water damage claim flagged 6 months ago" comes from.
     * Swapping it for Azure AI Search / Cosmos DB only means re-implementing this small class.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .observability import event


def _atomic_write(path: Path, payload: Any) -> None:
    """Write JSON safely: write to a temp file, then rename over the target.
    A crash mid-write can never leave a half-written (corrupt) memory file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


# ================================================================================================
# Long-term memory
# ================================================================================================
class LongTermMemory:
    """File layout: {"version": 1, "policies": {"POL-5521": {"claims": [ {...outcome...}, ... ]}}}"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "policies": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            data.setdefault("policies", {})
            return data
        except json.JSONDecodeError:
            # Error handling: a corrupt memory file must not take the system down. Keep a backup
            # for investigation and start with empty memory.
            backup = self.path.with_suffix(".corrupt.json")
            os.replace(self.path, backup)
            event("memory.corrupt", level="warning", backup=str(backup))
            return {"version": 1, "policies": {}}

    def policy_claims(self, policy_number: str) -> List[Dict[str, Any]]:
        """All remembered claims for one policy, oldest loss first."""
        claims = self._data["policies"].get(policy_number, {}).get("claims", [])
        return sorted(claims, key=lambda c: c.get("loss_date") or "")

    def find_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Look up a remembered claim by ID across all policies (used for duplicate/resubmission checks)."""
        for pol in self._data["policies"].values():
            for c in pol.get("claims", []):
                if c.get("claim_id") == claim_id:
                    return c
        return None

    def record_outcome(self, entry: Dict[str, Any]) -> None:
        """Insert or replace (by claim_id) one adjuster-approved outcome and save immediately."""
        pol = self._data["policies"].setdefault(entry["policy_number"], {"claims": []})
        pol["claims"] = [c for c in pol["claims"] if c.get("claim_id") != entry["claim_id"]] + [entry]
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write(self.path, self._data)
        event("memory.write", claim_id=entry["claim_id"], policy=entry["policy_number"],
              final_action=entry.get("final_action"), status=entry.get("status"))

    def reset(self) -> None:
        """Forget everything (used by `main.py memory reset` before a demo)."""
        self._data = {"version": 1, "policies": {}}
        _atomic_write(self.path, self._data)

    def dump(self) -> Dict[str, Any]:
        return self._data


# ================================================================================================
# Short-term / session memory
# ================================================================================================
@dataclass
class SessionState:
    """Everything one conversation session needs to remember. Saved as JSON after every step."""

    session_id: str
    as_of: str                      # reference "today" for date checks, ISO string
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    mode: str = "mock"              # "mock" or "foundry": thread IDs are only valid in the mode that made them

    # Thread-level memory: agent key -> thread id (a Foundry thread id, or a mock thread id).
    threads: Dict[str, str] = field(default_factory=dict)
    # Mock mode only: the message history of each mock thread (Foundry keeps this server-side).
    mock_threads: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Structured working memory filled in by the skills and the routine, keyed by batch_id / claim_id.
    batches: Dict[str, Dict[str, Any]] = field(default_factory=dict)          # ingested files
    validation: Dict[str, Dict[str, Any]] = field(default_factory=dict)       # validate_claims results
    coverage_facts: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # check_coverage facts
    assessments: Dict[str, Dict[str, Any]] = field(default_factory=dict)      # verified coverage assessments
    briefings: Dict[str, Dict[str, Any]] = field(default_factory=dict)        # adjuster briefings
    decisions: Dict[str, Dict[str, Any]] = field(default_factory=dict)        # adjuster decisions (gate 2)
    last_batch_id: Optional[str] = None
    last_claim_ids: List[str] = field(default_factory=list)                   # for resolving "it" / "that claim"
    routine_checkpoints: List[Dict[str, Any]] = field(default_factory=list)   # audit trail of state transitions
    turns: int = 0                                                            # number of Supervisor turns


class SessionStore:
    """Loads/saves SessionState objects as .sessions/<session_id>.json."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def load_or_create(self, session_id: str, as_of: str, mode: str) -> SessionState:
        """Resume an existing session (same threads, same context) or start a new one."""
        p = self._path(session_id)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            state = SessionState(**raw)
            if state.mode != mode:
                # Switching between mock and live: the old thread IDs don't exist in the other world.
                event("memory.session_mode_changed", level="warning", was=state.mode, now=mode)
                state.threads, state.mock_threads, state.mode = {}, {}, mode
            event("memory.session_resumed", session=session_id, turns=state.turns,
                  threads=list(state.threads))
            return state
        event("memory.session_created", session=session_id)
        return SessionState(session_id=session_id, as_of=as_of, mode=mode)

    def save(self, state: SessionState) -> None:
        _atomic_write(self._path(state.session_id), asdict(state))
