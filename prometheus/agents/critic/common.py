"""Small helpers every part of the Critic shares: findings, clipping, quotes, times (Critic v1 spec section 8).

Nothing here knows a threshold; every threshold is in `configs/critic_v1.json` and arrives as an argument.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from prometheus.agents.advisor.advisor import normalize
from prometheus.agents.critic.record import Finding
from prometheus.rag.citations import _defuse

EASTERN = ZoneInfo("America/New_York")   # only for the words of a message; every decision about time goes through `market_calendar`


def clip(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def blocker(dimension: str, code: str, message: str, value=None, threshold=None, requires_human: bool = False) -> Finding:
    return Finding(code=code, dimension=dimension, severity="blocker", message=message, value=value, threshold=threshold, requires_human=requires_human)


def warning(dimension: str, code: str, message: str, value=None, threshold=None, requires_human: bool = False) -> Finding:
    return Finding(code=code, dimension=dimension, severity="warning", message=message, value=value, threshold=threshold, requires_human=requires_human)


def info(dimension: str, code: str, message: str, value=None, threshold=None) -> Finding:
    return Finding(code=code, dimension=dimension, severity="info", message=message, value=value, threshold=threshold)


def quote_in_text(quote: str | None, text: str | None) -> bool:
    """Whether a quote is in a text, by the rule the Advisor used to accept it (`verify_signal`): case, typography and white space ignored, and a "[3]" inside the article shown as "(3)" counts."""
    if not quote or not text:
        return False
    wanted = normalize(quote)
    return bool(wanted) and (wanted in normalize(text) or wanted in normalize(_defuse(text)))


def producer(model_id: str | None) -> str | None:
    """Who made a model: the text before the slash of an OpenRouter id ("z-ai/glm-5.3" -> "z-ai"). None when the id is unknown."""
    return model_id.split("/", 1)[0].strip().lower() if isinstance(model_id, str) and "/" in model_id else None


def utc_text(moment: datetime | None) -> str | None:
    """A moment as the desk writes them: UTC, seconds, a trailing Z."""
    return None if moment is None else moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def et_text(moment: datetime | None) -> str:
    """A moment for a message: "2026-11-05 10:30 ET"."""
    return "unknown" if moment is None else moment.astimezone(EASTERN).strftime("%Y-%m-%d %H:%M ET")


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def metric_value(metrics: dict | None, name: str):
    """The value of one metric result of the Data Clerk's, or None."""
    return ((metrics or {}).get(name) or {}).get("value")
