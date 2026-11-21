"""One model call that must answer in JSON, with ONE retry when it does not (the loop of version 0's `advise`, shared by the plan call and the thesis call; design choices entry 51).

    run = new_run("plan", model)
    companies = ask_json(model, messages, run, "companies", RETRY_PLAN)       # the list under the key, or None; the run says what happened

`run` is one entry of `order.advisor_runs`: {"stage", "model", "tokens_in", "tokens_out", "calls", "dropped", "error"}. A model or network failure is NEVER an exception here: it is
`run["error"]` (the kind of the failure, never request details) and the caller carries on without that call's output. The reply is parsed leniently (a fenced block, text around
the JSON) but must contain an object whose `key` is a list; otherwise the model is told what was wrong, once.
"""
import contextvars
import json
import re
import threading

from langchain_core.messages import AIMessage, HumanMessage

from prometheus.common.trace import traced
from prometheus.models.chat import is_rate_limited


MODEL_TIMEOUT = 300   # seconds one model call may take. The client's own timeout is per read and a provider that keeps the connection open (a slow or stuck host) never trips it: one such call held a run for ten minutes.
CALL_LIMIT: contextvars.ContextVar[float | None] = contextvars.ContextVar("model_call_limit", default=None)      # None: MODEL_TIMEOUT, as every live run. The Proving Ground's fast Time Machine sets a shorter wait for the days it researches (`proving.speed.limit_of`)


def invoke_with_limit(model, messages: list, seconds: float | None = None):
    """`model.invoke(messages)`, but raises TimeoutError after `seconds` (the context's `CALL_LIMIT`, else MODEL_TIMEOUT, when left out). The call runs in a daemon thread, so a stuck one is abandoned, not killed (it ends with the process)."""
    seconds = (CALL_LIMIT.get() or MODEL_TIMEOUT) if seconds is None else seconds
    box: dict = {}

    def call():
        try:
            box["reply"] = model.invoke(messages)
        except BaseException as error:  # handed to the caller's thread
            box["error"] = error

    context = contextvars.copy_context()   # the caller's run (LangChain's callbacks and LangSmith's trace live in contextvars): without it the call is traced apart from its agent
    worker = threading.Thread(target=context.run, args=(call,), daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"the model call took more than {seconds:g} seconds")
    if "error" in box:
        raise box["error"]
    return box["reply"]


def new_run(stage: str, model) -> dict:
    return {"stage": stage, "model": getattr(model, "model_name", None), "tokens_in": 0, "tokens_out": 0, "calls": 0, "dropped": [], "error": None}


def parse_object(text: str, key: str) -> dict:
    """The JSON object in the reply, which must hold a list under `key`. Raises ValueError, with a short reason, if it does not."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError("not valid JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        raise ValueError(f"no {key} list")
    return data


def parse_list(text: str, key: str) -> list:
    """The list under `key` of the JSON object in the reply. Raises ValueError, with a short reason, if there is none."""
    return parse_object(text, key)[key]


def ask_json(model, messages: list, run: dict, key: str, retry: str) -> list | None:
    """Invoke the model; on a reply that is not the JSON asked for, say so once and ask again. Returns the list, or None (and `run["error"]` says why)."""
    found = ask_object(model, messages, run, key, retry)
    return None if found is None else found[key]


SPANS = {"plan": "advisor.plan", "thesis": "advisor.thesis", "review": "critic.evidence", "owner": "data_clerk.owner", "companies": "data_clerk.companies", "theme_members": "data_clerk.theme_members"}      # the span of a model step, by its stage (entry 115)


def step_name(model, messages: list, run: dict, key: str, retry: str) -> str:
    return SPANS.get(run["stage"], f"model.{run['stage']}")


@traced(step_name)
def ask_object(model, messages: list, run: dict, key: str, retry: str) -> dict | None:
    """The same, returning the WHOLE object (a reply that has other fields beside the list: the Data Clerk's company finder also gives a sector)."""
    for _ in range(2):
        try:
            reply = invoke_with_limit(model, messages)
        except Exception as error:  # a model or network failure: say what kind, never print request details
            run["error"], run["rate_limited"] = f"the {run['stage']} model call failed ({type(error).__name__})", is_rate_limited(error)
            return None
        run["calls"] += 1
        usage = getattr(reply, "usage_metadata", None) or {}
        run["tokens_in"] += usage.get("input_tokens", 0)
        run["tokens_out"] += usage.get("output_tokens", 0)
        content = reply.content if isinstance(reply.content, str) else ""
        try:
            found = parse_object(content, key)
            run["error"] = None
            return found
        except ValueError as problem:
            run["error"] = f"the {run['stage']} reply was not usable ({problem})"
            messages = [*messages, AIMessage(content=content), HumanMessage(content=retry.format(problem=problem))]
    return None
