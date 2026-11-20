"""LangSmith tracing is configured for a command's run (entry 114): on with a key, sent to the agent address, never beside a project, and off in the tests."""
import os

from prometheus.common import langsmith_tracing

NAMES = ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_ENDPOINT", "LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")


def environment_only(monkeypatch):
    """`.env` is not read here: the test sees only what it sets."""
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(langsmith_tracing, "setting", lambda name, default: os.environ.get(name) or default)


def test_the_tests_never_trace():
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "off" in langsmith_tracing.configure()
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_no_key_is_off(monkeypatch):
    environment_only(monkeypatch)
    assert "off" in langsmith_tracing.configure()
    assert "LANGSMITH_TRACING" not in os.environ


def test_a_key_traces_to_the_agent_address_and_drops_a_project(monkeypatch):
    environment_only(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "old-project")
    line = langsmith_tracing.configure()
    assert "ON" in line and "not-a-real-key" not in line and "LANGSMITH_PROJECT ignored" in line
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert (os.environ["LANGSMITH_AGENT_ID"], os.environ["LANGSMITH_AGENT_ENVIRONMENT"]) == ("prometheus", "local")
    assert "LANGSMITH_PROJECT" not in os.environ
    from langsmith import address
    assert address.address_from_env() == address.agent_address("prometheus", "local")


def test_the_advisor_call_runs_in_the_callers_context():
    """`invoke_with_limit` runs the model in a worker thread; the caller's contextvars (LangChain's run, LangSmith's trace) must reach it."""
    import contextvars

    from langchain_core.runnables import RunnableLambda

    from prometheus.agents.advisor.calls import invoke_with_limit
    run = contextvars.ContextVar("run", default=None)
    run.set("desk")
    assert invoke_with_limit(RunnableLambda(lambda messages: run.get()), [], seconds=5) == "desk"


def test_a_command_says_nothing_when_tracing_is_off(capsys):
    langsmith_tracing.start()
    assert capsys.readouterr() == ("", "")
