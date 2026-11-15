"""Tests for prometheus.models.chat (Part 6 of the RAG guide).

Run:  pytest tests/test_chat_model.py -q

No test calls OpenRouter or reads the real .env: every test points the settings at an empty temporary folder
(REPO_ROOT) and sets only the environment variables it needs, with a fake key. Building a ChatOpenAI and binding
tools to it make no network call. A real call is `python -m prometheus.models.chat --live "hello"`.
"""
import pytest

from prometheus.models import chat
from prometheus.rag import config
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools

FAKE_KEY = "sk-or-test-NOT-A-REAL-KEY-12345"
SETTINGS = ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_HTTP_REFERER", "OPENROUTER_APP_TITLE",
            "DEFAULT_MODEL", "FAST_MODEL", "ADVISOR_MODEL", "CRITIC_MODEL", "ACCOUNT_MANAGER_MODEL", "TRADER_MODEL"]


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """No real .env and no inherited settings."""
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)


def test_an_explicit_model_id_overrides_the_role_and_an_unknown_role_is_still_refused(with_key):
    assert chat.get_chat_model("data_clerk", model="deepseek/deepseek-v4.1-flash-1017").model_name == "deepseek/deepseek-v4.1-flash-1017"
    assert chat.get_chat_model("critic", model="x/y").model_name == "x/y"
    assert chat.get_chat_model("critic").model_name == "z-ai/glm-5.3"  # without it the role decides
    with pytest.raises(ValueError, match="unknown role"):
        chat.get_chat_model("janitor", model="x/y")


def test_live_prints_the_reply_and_a_failure_names_the_kind_of_error_only(monkeypatch, with_key, capsys):
    class Reply:
        content = "hello back"

    class Working:
        def invoke(self, prompt):
            return Reply()

    class Broken:
        def invoke(self, prompt):
            raise ConnectionError(f"could not reach the server with Authorization: Bearer {FAKE_KEY}")

    monkeypatch.setattr(chat, "get_chat_model", lambda role: Working())
    assert chat.main(["--live", "hello"]) == 0 and "reply    : hello back" in capsys.readouterr().out
    monkeypatch.setattr(chat, "get_chat_model", lambda role: Broken())
    assert chat.main(["--live", "hello"]) == 1
    captured = capsys.readouterr()
    assert "ERROR: the call failed (ConnectionError)" in captured.err and FAKE_KEY not in captured.err + captured.out


