"""Runtime configuration.

WHAT THIS FILE DOES
    Collects every tunable setting in one `Settings` object: the Azure endpoint, the model name,
    timeouts, file paths, tracing switches and demo faults. The rest of the code never reads
    environment variables directly; it receives a Settings instance. That keeps configuration in one
    place and makes tests easy (they just build a Settings with temp paths).

WHERE VALUES COME FROM (highest priority first)
    1. CLI flags in main.py (e.g. --mock, --live, --as-of, --fault) override...
    2. environment variables / the .env file (see .env.example), which override...
    3. the defaults declared below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Set

# Project root folder (the directory that contains main.py). All default paths hang off this.
ROOT = Path(__file__).resolve().parent.parent

# Load variables from a local .env file if python-dotenv is installed. Optional: plain environment
# variables work too.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _flag(name: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean ("1", "true", "yes", "on" count as True)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---- Azure AI Foundry -------------------------------------------------------------------
    project_endpoint: Optional[str] = None      # Foundry project endpoint URL
    model_deployment: str = "gpt-4o"            # name of the model deployment inside that project
    use_mock: bool = True                       # True = deterministic mock LLM, no Azure calls

    # ---- Agent run behaviour ------------------------------------------------------------------
    run_timeout_s: float = 120.0                # a single agent run is cancelled after this long
    poll_interval_s: float = 1.0                # how often we poll Foundry for run status
    max_run_attempts: int = 2                   # orchestrator retries a failed/timed-out run once
    coverage_chunk_size: int = 5                # claims sent to the Coverage agent per call

    # ---- File locations -------------------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    knowledge_dir: Path = ROOT / "knowledge"                                 # rule markdown files
    memory_path: Path = ROOT / "data" / "memory" / "policy_history.json"     # long-term memory
    policies_path: Path = ROOT / "data" / "policy_coverage.json"             # policy register
    sessions_dir: Path = ROOT / ".sessions"                                  # short-term/session memory
    logs_dir: Path = ROOT / "logs"                                           # JSONL traces
    reports_dir: Path = ROOT / "reports"                                     # adjuster reports
    foundry_state_path: Path = ROOT / ".foundry_state.json"                  # IDs of Azure resources we created

    # ---- Observability -------------------------------------------------------------------------
    enable_tracing: bool = False                # export OpenTelemetry spans to Azure Monitor
    content_recording: bool = False             # include prompt/response text in traces (PII risk)
    otel_console: bool = False                  # also print spans to the console
    appinsights_connection_string: Optional[str] = None

    # ---- Demo / chaos ----------------------------------------------------------------------------
    faults: Set[str] = field(default_factory=set)   # injected failures, e.g. {"bad_json"}
    as_of: date = field(default_factory=date.today)  # "today" for date checks (fixed for reproducible demos)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build Settings from environment variables (and .env)."""
        endpoint = os.getenv("PROJECT_ENDPOINT") or None
        # If no Foundry endpoint is configured we fall back to mock mode automatically, so the
        # project always runs out of the box.
        use_mock = _flag("USE_MOCK_LLM", default=endpoint is None)
        faults = {f.strip() for f in os.getenv("INJECT_FAULTS", "").split(",") if f.strip()}
        as_of_raw = os.getenv("TRIAGE_AS_OF")
        return cls(
            project_endpoint=endpoint,
            model_deployment=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
            use_mock=use_mock,
            run_timeout_s=float(os.getenv("AGENT_RUN_TIMEOUT_S", "120")),
            enable_tracing=_flag("ENABLE_TRACING"),
            content_recording=_flag("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"),
            otel_console=_flag("OTEL_CONSOLE_EXPORTER"),
            appinsights_connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING") or None,
            faults=faults,
            as_of=date.fromisoformat(as_of_raw) if as_of_raw else date.today(),
        )
