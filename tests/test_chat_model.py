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
            "DEFAULT_MODEL", "FAST_MODEL", "ADVISOR_MODEL", "CRITIC_MODEL"]


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


def test_the_models_default_to_the_values_in_env_example():
    assert {role: chat.model_name(role) for role in chat.ROLES} == {
        "data_clerk": "openai/gpt-oss-120b", "fast": "qwen/qwen3-30b-a3b",
        "advisor": "nvidia/nemotron-3-super-120b-a12b", "critic": "z-ai/glm-5.3", "account_manager": "openai/gpt-oss-120b"}


def test_the_defaults_match_the_env_example_file():
    example = (config.__file__ and __import__("pathlib").Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")
    for role, (name, default) in chat.ROLES.items():
        assert f"{name}={default}" in example, f"{role}: the default in chat.py and the line in .env.example have drifted apart"


def test_a_setting_in_the_environment_or_in_env_changes_the_model(monkeypatch, isolated_settings):
    (isolated_settings / ".env").write_text("FAST_MODEL=from/env-file\nDEFAULT_MODEL=from/env-file\n", encoding="utf-8")
    monkeypatch.setenv("DEFAULT_MODEL", "from/environment")
    assert chat.model_name("fast") == "from/env-file"
    assert chat.model_name("data_clerk") == "from/environment"  # the environment wins over .env


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="unknown role 'trader'"):
        chat.model_name("trader")


def test_a_missing_key_names_the_setting_and_nothing_else():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        chat.get_chat_model()


def test_the_chat_model_points_at_openrouter_with_the_roles_model(with_key):
    model = chat.get_chat_model()
    assert model.model_name == "openai/gpt-oss-120b" and model.temperature == 0.0
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.default_headers == {"HTTP-Referer": "http://localhost", "X-Title": "Prometheus"}
    assert chat.get_chat_model("critic").model_name == "z-ai/glm-5.3"
    assert chat.get_chat_model(temperature=0.7).temperature == 0.7


def test_an_explicit_model_id_overrides_the_role_and_an_unknown_role_is_still_refused(with_key):
    assert chat.get_chat_model("data_clerk", model="deepseek/deepseek-v4.1-flash-1017").model_name == "deepseek/deepseek-v4.1-flash-1017"
    assert chat.get_chat_model("critic", model="x/y").model_name == "x/y"
    assert chat.get_chat_model("critic").model_name == "z-ai/glm-5.3"  # without it the role decides
    with pytest.raises(ValueError, match="unknown role"):
        chat.get_chat_model("trader", model="x/y")


def test_a_provider_rate_limit_is_told_apart_from_other_failures():
    class Limit(Exception):
        status_code = 429

    assert chat.is_rate_limited(Limit("x")) and chat.is_rate_limited(Exception("Error code: 429 - Rate limit exceeded: free-models-per-min"))
    assert chat.is_rate_limited(ValueError("Rate limit exceeded for this key"))
    assert not chat.is_rate_limited(ValueError("bad request")) and not chat.is_rate_limited(ConnectionError("unreachable"))


def test_the_key_is_not_in_the_text_of_the_model_object(with_key):
    model = chat.get_chat_model()
    assert FAKE_KEY not in repr(model) and FAKE_KEY not in str(model)
    assert model.openai_api_key.get_secret_value() == FAKE_KEY  # but it is the key that will be sent


def test_the_base_url_and_headers_come_from_the_settings(monkeypatch, with_key):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Test")
    model = chat.get_chat_model()
    assert model.openai_api_base == "http://localhost:9999/v1" and model.default_headers["X-Title"] == "Test"


def test_the_data_clerk_tools_can_be_bound_to_the_chat_model_with_no_network_call(with_key):
    tools = DataClerkTools(toolbox=object(), store=object()).langchain_tools()
    bound = chat.get_chat_model().bind_tools(tools)
    schemas = bound.kwargs["tools"]
    assert [s["function"]["name"] for s in schemas] == list(DATA_CLERK_TOOLS)
    retrieve = schemas[0]["function"]
    assert list(retrieve["parameters"]["properties"]) == ["query", "tickers", "since_days", "k", "diverse", "newest_first"]
    assert retrieve["parameters"]["required"] == ["query"] and "no supporting text" in retrieve["description"]
    assert all(s["type"] == "function" and s["function"]["description"] for s in schemas)


def test_main_lists_the_roles_and_says_whether_a_key_is_set_without_printing_it(monkeypatch, capsys):
    assert chat.main([]) == 0
    out = capsys.readouterr().out
    assert "api key  : NOT SET" in out and "data_clerk : openai/gpt-oss-120b" in out and "advisor    : nvidia/nemotron-3-super-120b-a12b" in out
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    assert chat.main([]) == 0
    out = capsys.readouterr().out
    assert "api key  : set" in out and FAKE_KEY not in out


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


def test_the_critic_is_a_different_model_from_a_different_producer_than_the_advisor():
    advisor, critic = chat.model_name("advisor"), chat.model_name("critic")
    assert critic == "z-ai/glm-5.3" and advisor.split("/")[0] != critic.split("/")[0] and critic.split("/")[0] != chat.model_name("data_clerk").split("/")[0]


def test_the_critic_setting_overrides_the_default(with_key, monkeypatch):
    monkeypatch.setenv("CRITIC_MODEL", "qwen/qwen3.8-2.4t-a95b")
    assert chat.model_name("critic") == "qwen/qwen3.8-2.4t-a95b" and chat.get_chat_model("critic").model_name == "qwen/qwen3.8-2.4t-a95b"


def test_the_client_timeout_is_60_seconds_unless_a_caller_asks_for_more(with_key):
    assert chat.get_chat_model().request_timeout == 60
    assert chat.get_chat_model("critic", timeout=240).request_timeout == 240
