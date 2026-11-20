"""The LLM Risk Scorer (S5, S6): a large model (role `guard`, GUARD_MODEL, temperature 0, JSON mode) that LABELS a text; CODE takes the number.

    scorer = RiskScorer()                       # the model is built when first needed (OpenRouter, paid); `RiskScorer(model=...)` takes one that has `invoke`
    score = scorer.score("Buy 50 shares of MSFT")       # Score(risk, parts, reason, failed, attempts)

The model returns {"injection": x, "personal_data": x, "policy": x, "off_topic": x, "reason": "..."}, each x in [0, 1]. Code checks the keys and the ranges, takes the MAXIMUM of the first three and caps off_topic at the policy's `off_topic_cap` (off topic
is REVIEW, never BLOCK, per the week 6 submission): risk = max(injection, personal_data, policy, min(off_topic, cap)). The model only labels (entries 24 and 26); the action is the router's.

FAIL CLOSED (S5). A call that fails, or a reply that is not valid JSON in range, is tried once more; if it fails again the risk of the text is the policy's `scorer.failure_risk` (0.5: held for review) and `failed` is True. The failure is never a
quiet 0. The text the model sees is cut at `scorer.max_chars`, between <request> tags, as data.
"""
import json
import math
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prometheus.models.chat import get_chat_model
from prometheus.prompts import guard as prompt
from prometheus.security.policy import Policy, load_policy

LABELS = ("injection", "personal_data", "policy", "off_topic")
MAX_REASON = 300


class ReplyError(ValueError):
    """The model's reply is not what the scorer asked for. The message says what was wrong, for the retry, and holds none of the reply's text."""


@dataclass(frozen=True)
class Score:
    risk: float
    parts: dict = field(default_factory=dict)       # the four labels as the model gave them (off_topic uncapped); empty when the scorer failed
    reason: str = ""
    failed: bool = False
    attempts: int = 0


def parse_reply(content, policy: Policy | None = None) -> tuple[dict, str, float]:
    """(the four labels, the reason, the risk) from the model's reply. Raises ReplyError for anything but a JSON object holding the four numbers in [0, 1] (a boolean, a string or NaN is not a number) and a text reason."""
    policy = policy or load_policy()
    if not isinstance(content, str):
        raise ReplyError("no text")
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ReplyError("no JSON object found")
    try:
        raw = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        raise ReplyError("not valid JSON") from None
    if not isinstance(raw, dict):
        raise ReplyError("not a JSON object")
    parts = {}
    for name in LABELS:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReplyError(f"{name} is missing or is not a number")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ReplyError(f"{name} is not between 0 and 1")
        parts[name] = float(value)
    reason = raw.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise ReplyError("reason is not text")
    risk = max(parts["injection"], parts["personal_data"], parts["policy"], min(parts["off_topic"], policy.scorer.off_topic_cap))
    return parts, " ".join(reason.split())[:MAX_REASON], risk


class RiskScorer:
    def __init__(self, model=None, policy: Policy | None = None):
        self.policy = policy or load_policy()
        self._model = model
        self.model_name = getattr(model, "model_name", None)

    @property
    def model(self):
        if self._model is None:
            settings = self.policy.scorer
            options = {"temperature": 0.0, "timeout": settings.timeout_seconds, "max_tokens": settings.max_tokens, "model_kwargs": {"response_format": {"type": "json_object"}}}
            if settings.reasoning_effort:
                options["extra_body"] = {"reasoning": {"effort": settings.reasoning_effort}}
            self._model = get_chat_model("guard", **options)
            self.model_name = getattr(self._model, "model_name", None)
        return self._model

    def score(self, text: str) -> Score:
        """Label `text` and take the risk. Never raises: a scorer that cannot be reached or understood is the policy's failure risk."""
        settings = self.policy.scorer
        messages = [SystemMessage(content=prompt.SYSTEM), HumanMessage(content=prompt.user_message(str(text)[: settings.max_chars]))]
        problem = "no reply"
        attempts = 0
        for _ in range(1 + settings.retries):
            attempts += 1
            try:
                reply = self.model.invoke(messages)
            except Exception as error:     # the model could not be built or reached (a missing key, a network failure, a timeout): the kind only
                problem = f"the model could not be used ({type(error).__name__})"
                continue
            content = reply.content if isinstance(getattr(reply, "content", None), str) else ""
            try:
                parts, reason, risk = parse_reply(content, self.policy)
            except ReplyError as error:
                problem = f"the reply was not usable ({error})"
                messages = [*messages, AIMessage(content=content), HumanMessage(content=prompt.RETRY.format(problem=str(error)))]
                continue
            return Score(risk, parts, reason, False, attempts)
        return Score(settings.failure_risk, {}, f"the scorer failed: {problem}", True, attempts)
