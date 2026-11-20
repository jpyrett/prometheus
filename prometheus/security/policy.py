"""The security policy: `configs/security_v1.yaml`, loaded and checked (S1).

    policy = load_policy()                  # the file at configs/security_v1.yaml, read once
    policy.router.block                     # 0.8
    policy.rules["request"]                 # the compiled rules of a user request

Every threshold, pattern list and message of the security layer is in that file and in no other place of the code. A file that does not hold what the layer needs, or holds a pattern that does not compile, is REFUSED with a message that names
the key (a policy that is half loaded is a rule that is skipped, and a skipped rule is what this layer exists to prevent). The risk of this file is SECURITY risk only: it holds no trade size limit, liquidity rule or trade-risk rule.
"""
import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from prometheus.rag.config import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "configs" / "security_v1.yaml"
DOORS = ("request", "document", "message")
MATCHES = ("patterns", "account_number", "near")      # HOW a rule reads a text (the YAML key `match`; `patterns` when it has none)
EXACT, HINT = "exact", "hint"      # WHAT a rule of the REQUEST door is worth (the YAML key `kind`, entry 121): an exact rule is a program's own check and blocks alone; a hint is a word pattern that only points at the text
MESSAGE_KEYS = ("block", "review", "rate", "redacted")
FLAGS = re.IGNORECASE | re.MULTILINE


class PolicyError(ValueError):
    """The policy file is missing, is not valid YAML, or does not hold what the security layer needs. The message names the key."""


@dataclass(frozen=True)
class Router:
    block: float
    review: float
    flag: float


@dataclass(frozen=True)
class ScorerSettings:
    failure_risk: float
    off_topic_cap: float
    retries: int
    max_chars: int
    reasoning_effort: str | None
    max_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class RateSettings:
    max_requests_per_minute: int
    max_requests_per_hour: int


@dataclass(frozen=True)
class AuditSettings:
    path: Path
    preview_chars: int


@dataclass(frozen=True)
class TrustSettings:
    grounded_weight: float
    other_weight: float
    default_history: float
    registered_sources: tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    """One rule: a score given to a text when any of its patterns matches (`match` patterns), when an account number is not the requester's (`account_number`) or when a word of `a` is within `window` characters of a word of `b` (`near`).
    `kind` (entry 121) is `exact` (what a program checks exactly: it blocks on its own) or `hint` (a word pattern: it points at the text and the guard model decides; only the REQUEST door has hints). Every rule of the other doors is exact."""
    id: str
    score: float
    match: str
    patterns: tuple[re.Pattern, ...]
    window: int = 0
    a: tuple[re.Pattern, ...] = ()
    b: tuple[re.Pattern, ...] = ()
    raw: bool = False      # matched on the text exactly as it came (`hidden_text` reads the characters the other rules remove)
    kind: str = EXACT


@dataclass(frozen=True)
class Policy:
    version: int
    router: Router
    scorer: ScorerSettings
    rate: RateSettings
    audit: AuditSettings
    trust: TrustSettings
    messages: dict
    rules: dict   # door -> tuple[Rule, ...]


def _number(value, name: str, low: float = 0.0, high: float | None = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < low or (high is not None and value > high):
        raise PolicyError(f"{name} must be a number from {low:g}" + (f" to {high:g}" if high is not None else " up") + f", not {value!r}")
    return float(value)


def _whole(value, name: str, low: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise PolicyError(f"{name} must be a whole number of at least {low}, not {value!r}")
    return value


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise PolicyError(f"{key} is missing from the policy (it must be a mapping)")
    return value


def _patterns(values, name: str) -> tuple[re.Pattern, ...]:
    if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
        raise PolicyError(f"{name} must be a list of patterns (text)")
    compiled = []
    for number, text in enumerate(values, 1):
        try:
            compiled.append(re.compile(text, FLAGS))
        except re.error as error:
            raise PolicyError(f"{name}, pattern {number} does not compile ({error})") from None
    return tuple(compiled)


def _rule(raw, door: str) -> Rule:
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"]:
        raise PolicyError(f"a rule of {door} has no id")
    name = f"rules.{door}.{raw['id']}"
    match = raw.get("match", "patterns")
    if match not in MATCHES:
        raise PolicyError(f"{name}.match must be one of {', '.join(MATCHES)}")
    if door == "request":
        kind = raw.get("kind")
        if kind not in (EXACT, HINT):
            raise PolicyError(f"{name}.kind must be {EXACT} or {HINT} (every request rule says which; `account_number` and `near` are now written `match:`), not {kind!r}")
    else:
        kind = raw.get("kind", EXACT)
        if kind != EXACT:
            raise PolicyError(f"{name}.kind: only the request door has hints (a document or a message has no model to decide); leave `kind` out or write {EXACT} (`account_number` and `near` are now written `match:`), not {kind!r}")
    score = _number(raw.get("score"), f"{name}.score", low=0.0)
    if score <= 0:
        raise PolicyError(f"{name}.score must be above 0")
    on_raw = raw.get("raw", False)
    if not isinstance(on_raw, bool):
        raise PolicyError(f"{name}.raw must be true or false")
    if match == "near":
        window = _whole(raw.get("window"), f"{name}.window")
        return Rule(raw["id"], score, match, (), window, _patterns(raw.get("a"), f"{name}.a"), _patterns(raw.get("b"), f"{name}.b"), on_raw, kind)
    patterns = _patterns(raw.get("patterns"), f"{name}.patterns")
    if match == "account_number" and any(p.groups < 1 for p in patterns):
        raise PolicyError(f"{name}: an account_number pattern needs a group that captures the number")
    return Rule(raw["id"], score, match, patterns, raw=on_raw, kind=kind)


def parse_policy(raw) -> Policy:
    """A policy from the YAML's mapping. Raises PolicyError."""
    if not isinstance(raw, dict):
        raise PolicyError("the policy is not a mapping")
    router = _section(raw, "router")
    block, review, flag = (_number(router.get(k), f"router.{k}") for k in ("block", "review", "flag"))
    if not 0 < flag < review < block <= 1:
        raise PolicyError("router thresholds must satisfy 0 < flag < review < block <= 1")
    scorer = _section(raw, "scorer")
    effort = scorer.get("reasoning_effort")
    if effort is not None and effort not in ("low", "medium", "high"):
        raise PolicyError("scorer.reasoning_effort must be low, medium, high or null")
    cap = _number(scorer.get("off_topic_cap"), "scorer.off_topic_cap")
    if cap >= block:
        raise PolicyError("scorer.off_topic_cap must be below router.block (off topic is review, never block)")
    scorer_settings = ScorerSettings(_number(scorer.get("failure_risk"), "scorer.failure_risk"), cap, _whole(scorer.get("retries"), "scorer.retries", 0), _whole(scorer.get("max_chars"), "scorer.max_chars"),
                                     effort, _whole(scorer.get("max_tokens"), "scorer.max_tokens"), _number(scorer.get("timeout_seconds"), "scorer.timeout_seconds", low=1.0, high=None))
    rate = _section(raw, "rate")
    audit = _section(raw, "audit")
    path = Path(str(audit.get("path") or "")).expanduser()
    if not str(audit.get("path") or "").strip():
        raise PolicyError("audit.path is missing")
    trust = _section(raw, "trust")
    sources = trust.get("registered_sources")
    if not isinstance(sources, list) or not sources or not all(isinstance(s, str) and s for s in sources):
        raise PolicyError("trust.registered_sources must be a list of id prefixes")
    messages = _section(raw, "messages")
    for key in MESSAGE_KEYS:
        if not isinstance(messages.get(key), str) or not messages[key].strip():
            raise PolicyError(f"messages.{key} is missing")
    rules_raw = _section(raw, "rules")
    rules = {}
    for door in DOORS:
        listed = rules_raw.get(door)
        if not isinstance(listed, list) or not listed:
            raise PolicyError(f"rules.{door} is missing")
        built = tuple(_rule(item, door) for item in listed)
        names = [(rule.id, rule.kind) for rule in built]      # the same id may be an exact rule and a hint (`sql_or_ddl`: the statements and the words): never twice the same
        if len(set(names)) != len(names):
            raise PolicyError(f"rules.{door} names a rule twice")
        rules[door] = built
    return Policy(version=_whole(raw.get("version"), "version"), router=Router(block, review, flag), scorer=scorer_settings,
                  rate=RateSettings(_whole(rate.get("max_requests_per_minute"), "rate.max_requests_per_minute"), _whole(rate.get("max_requests_per_hour"), "rate.max_requests_per_hour")),
                  audit=AuditSettings(path if path.is_absolute() else REPO_ROOT / path, _whole(audit.get("preview_chars"), "audit.preview_chars")),
                  trust=TrustSettings(_number(trust.get("grounded_weight"), "trust.grounded_weight"), _number(trust.get("other_weight"), "trust.other_weight"), _number(trust.get("default_history"), "trust.default_history"), tuple(sources)),
                  messages={key: messages[key] for key in MESSAGE_KEYS}, rules=rules)


@functools.lru_cache(maxsize=8)
def _load(path: str) -> Policy:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError:
        raise PolicyError(f"the security policy file is missing: {path}") from None
    except (OSError, yaml.YAMLError) as error:
        raise PolicyError(f"the security policy file could not be read ({type(error).__name__})") from None
    return parse_policy(raw)


def load_policy(path: str | Path | None = None) -> Policy:
    """The policy at `path` (default: `configs/security_v1.yaml`), read and checked once per file. Raises PolicyError."""
    return _load(str(Path(path) if path else DEFAULT_PATH))
