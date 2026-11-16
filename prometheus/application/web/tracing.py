"""Model activity and tracing (UI v1 spec section 7). Three layers, each optional above the first:

  1. the stepper and the feed: LangGraph's own `tasks` and `values` stream (`desk_run.py`): nothing to configure;
  2. MODEL ACTIVITY: `ActivityHandler`, a LangChain callback passed to `graph.stream(config={"callbacks": ...})`. On each chat-model end it adds the call and its tokens to the run's totals ("7 model calls . 12.4k tokens").
     THE FIRST LIVE RUN SHOWED THAT THE CALLBACKS REACH ONLY SOME OF THE AGENTS' MODEL CALLS (the Account Manager's, the Data Clerk's and the Trader's, not the Advisor's or the Critic's), so a live total would be a
     wrong total: as the spec says, the counter is left out of a live run (`desk_run.COUNT_MODEL_CALLS`) and the replay shows the exact figures its capture recorded. The handler is kept, tested live, for the day the calls reach it;
  3. LANGFUSE, off unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set (`LANGFUSE_HOST` for a server other than the cloud): `callbacks` then also returns Langfuse's LangChain handler (`langfuse.langchain.CallbackHandler`
     of the Langfuse Python SDK v3 and later), and the run's id is the trace's session with the tags `prometheus-ui`, the trade mode, `strict` and `neutral`. The keys are read with `rag.config.setting` (the environment, then the
     repo's settings file through that function), never by opening a file here. With the keys missing, or the package not installed, there is no handler and `status()` says "off". LangSmith needs no code: LANGSMITH_TRACING=true
     and LANGSMITH_API_KEY in the environment and LangGraph traces itself.

WITH TRACING ON, the prompts, the model replies and the account's figures of a run are sent to the Langfuse server.
"""
import threading
from typing import Callable

from langchain_core.callbacks import BaseCallbackHandler

from prometheus.rag.config import setting


class ActivityHandler(BaseCallbackHandler):
    """Counts the chat-model calls of a run and their tokens, and tells `on_change` (calls, tokens in, tokens out, model) after each. Never raises into the model call."""
    raise_error = False

    def __init__(self, on_change: Callable[[int, int, int, str | None], None]):
        self.on_change = on_change
        self.calls = self.tokens_in = self.tokens_out = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        model = (response.llm_output or {}).get("model_name") if getattr(response, "llm_output", None) else None
        usage = (response.llm_output or {}).get("token_usage") if getattr(response, "llm_output", None) else None
        if usage:
            tokens_in, tokens_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        else:                                           # the message carries it (`usage_metadata`) when the provider's reply has no `token_usage`
            for group in getattr(response, "generations", []) or []:
                for generation in group:
                    meta = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                    tokens_in += int(meta.get("input_tokens") or 0)
                    tokens_out += int(meta.get("output_tokens") or 0)
                    model = model or (getattr(getattr(generation, "message", None), "response_metadata", None) or {}).get("model_name")
        with self._lock:
            self.calls += 1
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            snapshot = (self.calls, self.tokens_in, self.tokens_out, model)
        try:
            self.on_change(*snapshot)
        except Exception:
            pass


def _keys() -> tuple[str, str, str] | None:
    public, secret = setting("LANGFUSE_PUBLIC_KEY", ""), setting("LANGFUSE_SECRET_KEY", "")
    return (public, secret, setting("LANGFUSE_HOST", "")) if public and secret else None


def _langfuse_handler():
    """Langfuse's LangChain handler, or None (no keys, or the package is not installed)."""
    keys = _keys()
    if keys is None:
        return None
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return None
    public, secret, host = keys
    Langfuse(public_key=public, secret_key=secret, **({"host": host} if host else {}))      # registers the client the handler finds by its public key
    return CallbackHandler(public_key=public)


def status() -> str:
    """"langfuse" when the keys are set and the package can be imported, else "off" (what the health route says)."""
    if _keys() is None:
        return "off"
    try:
        import langfuse.langchain  # noqa: F401
    except ImportError:
        return "off"
    return "langfuse"


def startup_line() -> str:
    keys, importable = _keys(), status() == "langfuse"
    if importable:
        return "tracing: Langfuse is ON (prompts, replies and the account's figures of a run are sent to the Langfuse server)"
    if keys is not None:
        return "tracing: Langfuse keys are set but the package is not installed (pip install langfuse): tracing is off"
    return "tracing: off (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to turn Langfuse on)"


def stream_config(run, *, on_activity: Callable[[int, int, int, str | None], None] | None, mode: str, strict: bool, neutral: bool) -> dict:
    """The `config` handed to `graph.stream`: the activity counter (when `on_activity` is given), Langfuse's handler when it is on, and the run's id as the trace's session with its tags."""
    handlers = [ActivityHandler(on_activity)] if on_activity is not None else []
    metadata = {}
    try:
        traced = _langfuse_handler()
    except Exception:
        traced = None
    if traced is not None:
        handlers.append(traced)
        metadata = {"langfuse_session_id": run.id, "langfuse_tags": ["prometheus-ui", mode, "strict" if strict else "not-strict", "neutral" if neutral else "no-neutral"]}
    return {"callbacks": handlers, "metadata": metadata}
