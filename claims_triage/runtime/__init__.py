"""Agent runtimes. build_runtime() picks Foundry (live) or the mock, based on Settings.use_mock."""
from .base import AgentRunError, AgentRuntime, AgentTimeoutError, RunResult  # noqa: F401


def build_runtime(settings, session_state):
    """Factory: return the runtime for the configured mode.
    Imports are lazy, so mock mode works even without the Azure SDK installed."""
    if settings.use_mock:
        from .mock import MockRuntime

        return MockRuntime(settings, session_state)
    from .foundry import FoundryRuntime

    return FoundryRuntime(settings)
