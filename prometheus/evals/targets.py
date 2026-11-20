"""The functions `langsmith.evaluate` runs on each example (design choices entry 117; design/offline_evaluation_spec.md E5). NO PAID AGENT RERUN: nothing here runs the desk.

    redteam_target(guard)       runs the security guard on the example's text: a request goes through `Guard.screen_request` (code rules, then the Qwen guard of entry 116, cents), a document through `Guard.screen_document` (code
                                rules only, as in the desk). The rate limit is off and the audit log is off: an evaluation is not a user and must not write attack texts to the real log.
    recorded_redteam_target()   the same, FREE: the real guard (`Guard.with_scorer`) whose model is the REAL recorded replies of `tests/fixtures/security/guard_replies.json` (entries 116 and 121), so the join is the guard's own (the exact
                                rules and the model's risk, a hint only flags, the model is not asked when an exact rule already blocks) and not a copy of it. It is what the dry run uses.
    sessions_target(inputs)     REPLAYS the recorded output of the capture the example names (`example_key`): the cost of an experiment is the judge and nothing else.

Each returns what the evaluators read: {"action", "risk", "rules", "scorer_reason", "scored_by"} for the guard, the session's summary (`datasets.summary_of`) for a session.
"""
import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from prometheus.evals.datasets import GUARD_REPLIES, recorded_summary
from prometheus.security.audit import AuditLog
from prometheus.security.guard import Guard


def _silent_audit() -> AuditLog:
    return AuditLog(enabled=False)


def live_guard() -> Guard:
    """The real parts of a session: the code rules and the LLM Risk Scorer (the `guard` role's model, OpenRouter, a few cents for the whole red team)."""
    return Guard.with_scorer(audit=_silent_audit())


def code_guard() -> Guard:
    return Guard.code_only(audit=_silent_audit())


def _screen(guard: Guard, inputs: dict):
    if inputs.get("door", "request") == "document":
        return guard.screen_document(inputs["text"], doc_id="offline-eval", corpus="redteam")
    return guard.screen_request(inputs["text"], account_id=inputs.get("account_id"), rate=False)


def _model_reason(reason: str) -> str:
    """The model's sentence in a verdict's reason (the guard writes it last, as `model: <sentence>`); '' when the model was not asked."""
    return reason[len("model: "):] if reason.startswith("model: ") else reason.partition("; model: ")[2]


def _output(verdict) -> dict:
    return {"action": verdict.action, "risk": verdict.risk, "rules": list(verdict.rules), "scorer_reason": _model_reason(verdict.reason), "scored_by": verdict.scored_by}


def redteam_target(guard: Guard | None = None):
    """The target of the red-team dataset: `target(inputs) -> {action, risk, rules, scorer_reason, scored_by}`. With no `guard` it is the live one (the Qwen model is called for a request the code does not block)."""
    guard = guard or live_guard()

    def target(inputs: dict) -> dict:
        return _output(_screen(guard, inputs))
    return target


def recorded_replies() -> dict:
    """{request text: the recorded item} of the guard model's real replies (entry 116)."""
    return {item["text"]: item for item in json.loads(GUARD_REPLIES.read_text(encoding="utf-8"))["items"]}


def _replaying(item: dict) -> RunnableLambda:
    """A real LangChain model that answers with the recorded raw replies of one text, in order (a second one only when the first was not usable, as when they were captured)."""
    replies = iter(item["replies"])
    last = item["replies"][-1]
    return RunnableLambda(lambda messages: AIMessage(content=next(replies, last)))


def recorded_redteam_target():
    """The free target: the guard itself, with the guard model's real recorded reply to the text as its model. A request that has no recorded reply (none in the red-team file) is screened by the code rules alone, as before."""
    code, replies = code_guard(), recorded_replies()

    def target(inputs: dict) -> dict:
        item = replies.get(inputs["text"])
        if inputs.get("door", "request") != "request" or item is None:
            return _output(_screen(code, inputs))
        return _output(_screen(Guard.with_scorer(model=_replaying(item), audit=_silent_audit()), inputs))
    return target


def sessions_target(inputs: dict) -> dict:
    """The recorded output of the capture `inputs["example_key"]` names (no agent is rerun)."""
    return recorded_summary(inputs["example_key"])
