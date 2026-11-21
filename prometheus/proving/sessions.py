"""Saving a finished desk run for the Track Record (design/proving_ground_spec.md section 10, design choices entry 126).

    path = save_session(prompt, intent, final_state, langsmith_trace_id=None, account_id=None, screen=None)       # data/proving/sessions/<request_id>.json, or None
    payload = capture_of(prompt, intent, path, order, ...)                                                       # the dict that is written (the `--save` capture format plus a few fields of its own)

A live desk run of the page (`application/web/desk_run._stream`) and of the session (`application/session.run_session`) that reached the Advisor ends with ONE call to `save_session`. What is written is the `--save`
capture of `session.py`: `{request, intent, path, order}` with the work order WITHOUT its `data` (the article text stays out), through the one JSON encoder, plus `saved_utc`, `source: "run"`, `account_id` and
`langsmith_trace_id` (the trace of the run, when tracing is on: the Track Record posts the outcomes as feedback on it, section 12). The request text goes through the security layer's `screen` first when one is given
(a secret or another account's number is redacted, as on the page's own messages).

A FAILURE TO SAVE IS A LOG LINE, NEVER A FAILED RUN: `save_session` catches everything and returns None. A direct order has no recommendation to score and is not saved; neither is a run that never reached the Advisor.
Not in `--replay` mode (a replay does not call this) and not in tests: `tests/conftest.py` points `default_store` at a temporary folder. No model, no network, nothing is written outside `data/proving/sessions/`.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from prometheus.proving.store import ProvingStore

log = logging.getLogger(__name__)


def default_store() -> ProvingStore:
    """The store a run saves into (`data/proving`); a test replaces this function with one that gives a store in a temporary folder."""
    return ProvingStore()


def reached_the_advisor(order, plan) -> bool:
    """A run that has recommendations or candidates of the Advisor to keep, and is not a direct order."""
    return order is not None and isinstance(plan, dict) and not plan.get("direct") and bool(order.recommendations or order.candidates)


def capture_of(prompt: str, intent, path: list[str], order, *, account_id: int | None = None, langsmith_trace_id: str | None = None, screen=None) -> dict:
    """The saved session: the `--save` capture (request, intent, path, the work order without its `data`) and what the Track Record needs to know about it."""
    text = " ".join(str(prompt).split())
    if screen is not None:
        text = screen(text)
    return {"request": text, "intent": intent.model_dump(mode="json") if hasattr(intent, "model_dump") else intent, "path": list(path), "order": json.loads(order.model_dump_json(exclude={"data"})),
            "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "run", "account_id": account_id, "langsmith_trace_id": langsmith_trace_id}


def screen_of(guard, account_id=None):
    """The request text through the security layer's message screen (a secret and another account's number are redacted), or None when there is no guard (a test, a replay)."""
    return None if guard is None else (lambda text: guard.screen_message(text, account_id=account_id).text)


def current_trace_id() -> str | None:
    """The id of the LangSmith trace that is open in this thread, or None (tracing off, or no trace open)."""
    try:
        from langsmith.run_helpers import get_current_run_tree
        tree = get_current_run_tree()
        return str(tree.trace_id) if tree is not None else None
    except Exception:
        return None


def save_session(prompt: str, intent, final: dict | None, *, account_id: int | None = None, langsmith_trace_id: str | None = None, screen=None, store: ProvingStore | None = None) -> Path | None:
    """Keep a finished desk run (`final` is the desk's last state: `order`, `plan`, `path`) for the Track Record. Returns the file, or None when there was nothing to keep or the save failed (it never raises)."""
    try:
        order, plan = (final or {}).get("order"), (final or {}).get("plan")
        if not reached_the_advisor(order, plan):
            return None
        payload = capture_of(prompt, plan.get("intent", intent), (final or {}).get("path") or [], order, account_id=account_id, langsmith_trace_id=langsmith_trace_id or current_trace_id(), screen=screen)
        return (store or default_store()).save_session(order.request_id, payload)
    except Exception as error:      # a failure to save is a log line, never a failed run
        log.warning("the session could not be saved for the Track Record (%s)", type(error).__name__)
        return None


def save_capture(capture: dict, *, store: ProvingStore | None = None, trace_id: str | None = None) -> Path:
    """Keep a capture that is already made (the `--save` file of a real session) as a saved session: the way a run of before this feature, or a capture of a real run, joins the Track Record. Raises ValueError when it
    is not a capture."""
    order = capture.get("order") if isinstance(capture, dict) else None
    if not isinstance(order, dict) or not isinstance(order.get("request_id"), str):
        raise ValueError("not a session capture: it has no work order with a request_id")
    payload = {**capture, "saved_utc": capture.get("saved_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": capture.get("source", "run"),
               "langsmith_trace_id": capture.get("langsmith_trace_id") or trace_id}
    return (store or default_store()).save_session(order["request_id"], payload)
