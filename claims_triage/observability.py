"""Logging and tracing (the "observability" requirement).

HOW TO USE IT (from any module)
    event("tool.call", tool="check_coverage", ok=True)      # record something that happened
    with span("agent.coverage"):                             # time a block of work
        ...

WHERE THE OUTPUT GOES
  1. Console (stderr): a human-readable one-liner per event. This is what you watch during the demo.
     The first word of the event name picks an icon, e.g. "routine.transition" -> ◆.
  2. JSONL trace file: logs/trace_<session>_<timestamp>.jsonl, one JSON object per event. Good for
     auditing ("which rules fired for C-2031 and who approved it?").
  3. OpenTelemetry spans (optional):
       * ENABLE_TRACING=true         -> exported to the Application Insights connected to the Foundry
                                        project (visible in Foundry portal -> Tracing).
       * OTEL_CONSOLE_EXPORTER=true  -> spans printed to the console.
     In live mode the Azure AI Agents SDK is also auto-instrumented (AIAgentsInstrumentor), so agent
     runs, tool calls and thread messages show up as spans without any extra code.

EVENT NAME PREFIXES USED ACROSS THE CODEBASE
    setup.*      provisioning (agents, vector store, tracing)
    routine.*    state-machine transitions, fallbacks, routing decisions
    agent.*      agent runs starting / ending / failing
    tool.*       function-tool calls, retries, file_search
    guardrail.*  verifier discrepancies (hallucinated or missed rules, downgrades)
    hitl.*       human decisions at the checkpoints
    memory.*     session / long-term memory reads and writes
"""
from __future__ import annotations

import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .config import Settings

# Module-level state, set once by setup().
_jsonl_path: Optional[Path] = None   # where JSONL events are appended
_tracer = None                       # OpenTelemetry tracer (None if OTel isn't installed)
_verbose = False                     # -v flag: show full, untruncated event fields
_session_id = "-"

# Icon per event-name prefix, so the console log is scannable at a glance.
_ICONS = {
    "routine": "◆",
    "agent": "●",
    "tool": "↳",
    "guardrail": "⛨",
    "hitl": "✋",
    "memory": "▣",
    "error": "✖",
    "setup": "⚙",
}

# Bulky fields hidden from the console unless -v is used (they're always in the JSONL file).
_QUIET_KEYS = {"args", "output_preview"}


def setup(settings: Settings, session_id: str, verbose: bool = False) -> None:
    """Initialise logging for this process. Call once, before anything else logs."""
    global _jsonl_path, _tracer, _verbose, _session_id
    _verbose = verbose
    _session_id = session_id
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _jsonl_path = settings.logs_dir / f"trace_{session_id}_{stamp}.jsonl"
    _tracer = _setup_otel(settings)


def _setup_otel(settings: Settings):
    """Configure OpenTelemetry exporters. Every step is optional and fails soft."""
    try:
        from opentelemetry import trace
    except ImportError:
        return None   # OTel not installed -> span() becomes a no-op

    # 1. Export to Azure Monitor / Application Insights (shows up in Foundry portal -> Tracing).
    want_azure = settings.enable_tracing
    if want_azure:
        conn = settings.appinsights_connection_string
        if not conn and not settings.use_mock and settings.project_endpoint:
            # No connection string given: ask the Foundry project which App Insights it's connected to.
            conn = _foundry_appinsights_connection_string(settings)
        if conn:
            try:
                from azure.monitor.opentelemetry import configure_azure_monitor

                configure_azure_monitor(connection_string=conn)
                event("setup.tracing", exporter="azure_monitor")
            except ImportError:
                event("setup.tracing", level="warning",
                      message="azure-monitor-opentelemetry not installed; Azure export disabled")
        else:
            event("setup.tracing", level="warning",
                  message="ENABLE_TRACING set but no Application Insights connection string found")

    # 2. Optionally print spans to the console (only if nothing else configured a provider already).
    if settings.otel_console and not isinstance(trace.get_tracer_provider(), _sdk_provider_cls()):
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

            provider = TracerProvider(resource=Resource.create({"service.name": "claims-triage"}))
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr)))
            trace.set_tracer_provider(provider)
        except ImportError:
            pass

    # 3. Live mode: auto-instrument the Azure AI Agents SDK (agent runs, tool calls, messages).
    if not settings.use_mock and (want_azure or settings.otel_console):
        try:
            from azure.ai.agents.telemetry import AIAgentsInstrumentor

            AIAgentsInstrumentor().instrument(enable_content_recording=settings.content_recording)
            event("setup.tracing", instrumentor="AIAgentsInstrumentor")
        except Exception as exc:  # pragma: no cover - best effort
            event("setup.tracing", level="warning", message=f"agent instrumentation failed: {exc}")

    return trace.get_tracer("claims_triage")


def _sdk_provider_cls():
    """The OTel SDK TracerProvider class, or a dummy type if the SDK isn't installed."""
    try:
        from opentelemetry.sdk.trace import TracerProvider

        return TracerProvider
    except ImportError:  # pragma: no cover
        return type(None)


def _foundry_appinsights_connection_string(settings: Settings) -> Optional[str]:
    """Ask the Foundry project for its connected Application Insights resource."""
    try:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        client = AIProjectClient(endpoint=settings.project_endpoint, credential=DefaultAzureCredential())
        return client.telemetry.get_application_insights_connection_string()
    except Exception as exc:  # pragma: no cover - depends on Azure
        event("setup.tracing", level="warning", message=f"could not fetch App Insights connection: {exc}")
        return None


def event(name: str, level: str = "info", **fields: Any) -> None:
    """Emit one structured event to console + JSONL (+ attach it to the current OTel span)."""
    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "session": _session_id,
        "level": level,
        "event": name,
        **fields,
    }
    # 1. Append to the JSONL audit trail.
    if _jsonl_path is not None:
        with _jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # 2. Attach to whatever span is currently open, so events show inside traces.
    if _tracer is not None:
        try:
            from opentelemetry import trace

            current = trace.get_current_span()
            if current is not None and current.is_recording():
                current.add_event(name, {k: _attr(v) for k, v in fields.items()})
        except Exception:  # pragma: no cover
            pass

    # 3. Print a readable line to the console.
    _console(name, level, fields)


def _console(name: str, level: str, fields: Dict[str, Any]) -> None:
    """Format one event as '  <icon> <name> key=value ...' (dim; yellow for warnings; red for errors)."""
    prefix = name.split(".", 1)[0]
    icon = _ICONS.get("error" if level == "error" else prefix, "·")
    shown = {k: v for k, v in fields.items() if _verbose or k not in _QUIET_KEYS}
    parts = []
    for k, v in shown.items():
        text = v if isinstance(v, str) else json.dumps(v, default=str)
        if not _verbose and len(text) > 140:
            text = text[:137] + "..."
        parts.append(f"{k}={text}")
    colour = {"warning": "\033[33m", "error": "\033[31m"}.get(level, "\033[2m")
    reset = "\033[0m"
    if not sys.stderr.isatty():  # no colour codes when output is piped to a file
        colour = reset = ""
    print(f"{colour}  {icon} {name} {' '.join(parts)}{reset}", file=sys.stderr)


def _attr(value: Any) -> Any:
    """OTel attributes must be primitives; serialise anything else to (truncated) JSON."""
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, default=str)[:2000]


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """OTel span (no-op if OpenTelemetry isn't installed) + a duration measurement.
    Exceptions inside the block are recorded on the span and then re-raised."""
    start = time.perf_counter()
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(f"claims_triage.{k}", _attr(v))
        try:
            yield
        except Exception as exc:
            s.record_exception(exc)
            try:
                from opentelemetry.trace import Status, StatusCode

                s.set_status(Status(StatusCode.ERROR, str(exc)))
            except ImportError:  # pragma: no cover
                pass
            raise
        finally:
            s.set_attribute("claims_triage.duration_ms", round((time.perf_counter() - start) * 1000, 1))


def trace_file() -> Optional[Path]:
    """Path of this run's JSONL trace (printed at the end of a triage)."""
    return _jsonl_path
