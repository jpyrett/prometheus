"""The router and the Trust Score: the Module 6 class example, with the thresholds and weights of `configs/security_v1.yaml` (S3, S7).

    act(0.55)                                   # "REDACT_OR_REVIEW"
    trust_score({"grounded": 1, "source": 1, "confidence": 0.69, "history": 0.5})      # 0.4 * grounded + 0.2 * (source + confidence + history)

`act` is a pure function: the same risk gives the same action and nothing else happens. THE TRUST SCORE IS RELIABILITY AND NEVER BLOCKS ANYTHING: it is recorded on the audit log and in the trace and changes no decision.
"""
import math

from prometheus.security.policy import Policy, Router, TrustSettings, load_policy

ALLOW, ALLOW_AND_FLAG, REDACT_OR_REVIEW, BLOCK = "ALLOW", "ALLOW_AND_FLAG", "REDACT_OR_REVIEW", "BLOCK"
ACTIONS = (ALLOW, ALLOW_AND_FLAG, REDACT_OR_REVIEW, BLOCK)
SIGNALS = ("grounded", "source", "confidence", "history")


def _unit(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number from 0 to 1, not {value!r}")
    return float(value)


def act(risk, router: Router | None = None) -> str:
    """The action for a risk in [0, 1]: >= block BLOCK, >= review REDACT_OR_REVIEW, >= flag ALLOW_AND_FLAG, else ALLOW. A risk that is not a number in range raises ValueError (it is never read as safe)."""
    router = router or load_policy().router
    risk = _unit(risk, "risk")
    if risk >= router.block:
        return BLOCK
    if risk >= router.review:
        return REDACT_OR_REVIEW
    if risk >= router.flag:
        return ALLOW_AND_FLAG
    return ALLOW


def trust_score(signals: dict, weights: TrustSettings | None = None) -> float:
    """trust = grounded_weight * grounded + other_weight * (source + confidence + history), each signal in [0, 1] (the class example is .4 and .2). Raises ValueError for a missing or out-of-range signal."""
    weights = weights or load_policy().trust
    values = {name: _unit(signals.get(name), name) for name in SIGNALS}
    return weights.grounded_weight * values["grounded"] + weights.other_weight * (values["source"] + values["confidence"] + values["history"])


def signals_of(record: dict, history, policy: Policy | None = None) -> dict:
    """The four signals of one recommendation, from what is already on the work order (S7), each in [0, 1]:

    grounded    the share of the recommendation's citations (its signals) that passed the citation check (`verified`); 0 with none
    source      the share of its citations whose chunk id starts with a registered source prefix (Finnhub, Massive, EDGAR); 0 with none
    confidence  the Advisor's calculated score / 100; 0 when it has none
    history     `history` (the ledger's past hit rate for the symbol, in [0, 1]) when there is one, else the policy's default
    """
    policy = policy or load_policy()
    cites = [s for s in (record.get("signals") or []) if isinstance(s, dict)]
    grounded = sum(1 for s in cites if s.get("verified") is True) / len(cites) if cites else 0.0
    registered = sum(1 for s in cites if str(s.get("chunk_id") or "").startswith(policy.trust.registered_sources)) / len(cites) if cites else 0.0
    score = record.get("score")
    confidence = min(max(score / 100, 0.0), 1.0) if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) else 0.0
    past = history if isinstance(history, (int, float)) and not isinstance(history, bool) and math.isfinite(history) and 0 <= history <= 1 else policy.trust.default_history
    return {"grounded": grounded, "source": registered, "confidence": confidence, "history": float(past)}
