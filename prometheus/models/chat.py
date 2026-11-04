"""The chat model the agents talk to, through OpenRouter (design choices entry 22).

    python -m prometheus.models.chat                 which model each role uses (no network, the key is never printed)
    python -m prometheus.models.chat --live "hello"  one real call, to prove the key and the model work (costs a little)

`get_chat_model(role)` returns a LangChain `ChatOpenAI` pointed at OpenRouter, which speaks the OpenAI protocol.
The model for each role comes from the settings in .env (the names are in .env.example), so changing a model is
a settings change, not a code change:

    data_clerk -> DEFAULT_MODEL     fast -> FAST_MODEL     advisor -> ADVISOR_MODEL     critic -> CRITIC_MODEL     account_manager -> ACCOUNT_MANAGER_MODEL

`llm.py` (the local Hugging Face model from Lab 1.1) stays as it is. A local chat model for the agents is later work.
The temperature is 0 by default: the Data Clerk should pick the same tool for the same question.
"""
import argparse
import sys

from langchain_openai import ChatOpenAI

from prometheus.rag.config import setting  # the same lookup as the RAG settings: environment, then .env, then default

ROLES = {  # role -> (setting name, default). The defaults are the values in .env.example.
    "data_clerk": ("DEFAULT_MODEL", "openai/gpt-oss-120b"),
    "fast": ("FAST_MODEL", "qwen/qwen3-30b-a3b"),
    "advisor": ("ADVISOR_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
    "critic": ("CRITIC_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
    "account_manager": ("ACCOUNT_MANAGER_MODEL", "openai/gpt-oss-120b"),  # RaiseFunds, the Tree of Thought (entry 42): Jacob asked for gpt-oss-120b
}
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def is_rate_limited(error: Exception) -> bool:
    """A provider's rate or daily limit (HTTP 429), as opposed to the model or the request being wrong. OpenRouter's free
    models are limited to 20 requests a minute and, on a free-tier account, about 50 a day."""
    text = str(error).lower()
    return getattr(error, "status_code", None) == 429 or type(error).__name__ == "RateLimitError" or "rate limit" in text or "error code: 429" in text


def model_name(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}: choose one of {', '.join(ROLES)}")
    name, default = ROLES[role]
    return setting(name, default)


def get_chat_model(role: str = "data_clerk", *, model: str | None = None, temperature: float = 0.0, **options) -> ChatOpenAI:
    """A chat model for this role, or for the OpenRouter model id given in `model` (to compare models; the role then
    only checks that the role exists). Raises RuntimeError, naming the setting and never its value, if the key is missing."""
    role_model = model_name(role)  # also checks that the role exists, even when a model id is given
    model = model or role_model
    key = setting("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env; see .env.example)")
    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=setting("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        temperature=temperature,
        timeout=60,
        max_retries=2,
        default_headers={"HTTP-Referer": setting("OPENROUTER_HTTP_REFERER", "http://localhost"),
                         "X-Title": setting("OPENROUTER_APP_TITLE", "Prometheus")},
        **options,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", metavar="PROMPT", help="send this prompt to the data_clerk model and print the reply")
    args = parser.parse_args(argv)
    print(f"base url : {setting('OPENROUTER_BASE_URL', DEFAULT_BASE_URL)}")
    print(f"api key  : {'set' if setting('OPENROUTER_API_KEY', '') else 'NOT SET'}")
    for role in ROLES:
        print(f"{role:<11}: {model_name(role)}")
    if args.live:
        try:
            reply = get_chat_model("data_clerk").invoke(args.live)
        except Exception as error:  # a network or key problem: say what kind, never print the request headers
            print(f"ERROR: the call failed ({type(error).__name__})", file=sys.stderr)
            return 1
        print(f"\nreply    : {reply.content}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
