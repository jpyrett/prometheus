"""Score, verdict and reasons (Critic v1 spec section 9, step 9). Plain arithmetic over the findings and the six dimensions' scores: no model, no network.

    score  = min(100, round(sum(weight x dimension score) / sum(weights of the applicable dimensions) + the great-deal bonus))
    score  = min(score, blocked_score_cap)   when any blocker exists
    approved = no blocker and score >= approve_min_score and no finding that asks for a person to look

The cap means one number tells the whole story: a recommendation with a blocker never shows 85. The score before the cap is kept (`score_before_blockers`), so the report can say "would have been 77".
A finding that asks for a person to look (`requires_human`) DENIES the recommendation (`needs_human_review`; design choices entry 103, Jacob 2026-11-15: *"if the Critic wants a person to review, then the Critic needs to Deny
the recommendation and return the reason"*): there is no third state between approved and denied. That denial is NOT a blocker and does not cap the score: it says the review could not be trusted without a person, not that
the recommendation is bad, so the score stays what the dimensions gave. `requires_human` and `requires_human_reasons` stay on the review: they name the finding.
The summary sentence is built from fixed wording and the findings' own messages: a model never writes a line the user reads. An APPROVE sentence also says the two findings that inform without denying (`INFORMS`).
"""
from dataclasses import dataclass, field

from prometheus.agents.critic.config import DIMENSIONS, CriticConfig
from prometheus.agents.critic.record import BLOCKER_ORDER, Dimension, Finding, Measure
from prometheus.agents.critic.report import scorecard  # noqa: F401  (the card the Account Manager shows; defined with the report so that scoring stays pure arithmetic)

# The two findings that INFORM an approval (Critic v2, rules C2 and C3): they stopped denying, so an APPROVE sentence says them, or the user would learn of a newer bad article or of a blend the quotes no longer support only from the card.
INFORMS = ("fresh_negative_news", "blend_not_supported")
LABEL = {"accuracy": "accuracy", "evidence": "evidence", "risk_fit": "risk fit", "timing": "timing", "value": "value", "chart": "chart"}


@dataclass
class Verdict:
    approved: bool
    score: int
    score_before_blockers: int
    grade: str
    denial_reasons: list[str] = field(default_factory=list)
    requires_human: bool = False                                        # a finding asked for a person to look: the recommendation is denied (`needs_human_review`)
    requires_human_reasons: list[str] = field(default_factory=list)     # the codes of those findings


def make_dimension(name: str, config: CriticConfig, score: float, applicable: bool = True, measures: list[Measure] | None = None) -> Dimension:
    """One dimension of the score: its weight, its 0 to 100 score, and the points it earns of its weight (`accuracy 27/30`)."""
    weight = config.weights[name]
    score = max(0.0, min(100.0, score))
    return Dimension(weight=weight, score=round(score, 2), points=round(score * weight / 100, 2), applicable=applicable, measures=measures or [])


def grade_of(score: int, config: CriticConfig) -> str:
    """A (>= 90), B (>= 80), C (>= 70), D (>= 60), else F: the marks are in the config."""
    for grade in ("A", "B", "C", "D"):
        if score >= config.verdict.grades[grade]:
            return grade
    return "F"


def weighted_score(dimensions: dict[str, Dimension], bonus: float) -> int:
    """`min(100, round(sum(weight x score) / sum(weights of the applicable dimensions) + bonus))`."""
    applicable = [d for d in dimensions.values() if d.applicable]
    total = sum(d.weight for d in applicable)
    return min(100, round(sum(d.weight * d.score for d in applicable) / total + bonus)) if total else 0


def order_blockers(findings: list[Finding]) -> list[Finding]:
    """The blockers, in the order of the closed list of the spec (9.3): accuracy first, then timing, risk fit, value, chart, evidence."""
    return sorted((f for f in findings if f.severity == "blocker"), key=lambda f: BLOCKER_ORDER.index(f.code))


def needs_a_person(findings: list[Finding]) -> list[Finding]:
    """The findings that ask for a person to look, one for each code (the first of a code), in the order they were raised."""
    first: dict[str, Finding] = {}
    for f in findings:
        if f.requires_human:
            first.setdefault(f.code, f)
    return list(first.values())


def decide(findings: list[Finding], dimensions: dict[str, Dimension], bonus: float, config: CriticConfig) -> Verdict:
    """The verdict of one review: approved or denied, nothing in between. A blocker denies whatever the score; with none, the score must reach `approve_min_score`; and a finding that asks for a person to
    look denies it too (`needs_human_review`, last in `denial_reasons`), with the score left as the dimensions gave it (only a blocker caps it)."""
    before = weighted_score(dimensions, bonus)
    blockers = order_blockers(findings)
    score = min(before, config.verdict.blocked_score_cap) if blockers else before
    reasons = [f.code for f in blockers]
    if not blockers and score < config.verdict.approve_min_score:
        reasons.append("score_below_minimum")
    human = [f.code for f in needs_a_person(findings)]
    if human:
        reasons.append("needs_human_review")
    return Verdict(approved=not reasons, score=score, score_before_blockers=before, grade=grade_of(score, config), denial_reasons=reasons, requires_human=bool(human), requires_human_reasons=human)


def great_deal_bonus_allowed(findings: list[Finding], quality_multiplier: float, red_flag_passed: bool, min_quality: float) -> bool:
    """The bonus is earned only by a recommendation that is otherwise sound: no blocker in accuracy, value or chart (a discount on a dead cat bounce is a trap), quality at least `min_quality`, and the Advisor's red-flag gate passed."""
    return not any(f.severity == "blocker" and f.dimension in ("accuracy", "value", "chart") for f in findings) and quality_multiplier >= min_quality and red_flag_passed


def summarize(symbol: str, verdict: Verdict, findings: list[Finding], dimensions: dict[str, Dimension], minimum: int) -> str:
    """One sentence, from fixed wording: the verdict, the score, and either the points of each dimension or the reasons it was denied: the two heaviest blockers, the score under the mark, and a person who must look
    (`DENY YUM: a person must look before this can trade: evidence_not_reviewed (The quotes could not be reviewed by the model: ...).`)."""
    if verdict.approved:
        parts = ", ".join(f"{LABEL[n]} {d.points:.0f}/{d.weight:g}" for n, d in dimensions.items() if d.applicable)
        told = list({f.code: f for f in findings if f.severity == "warning" and f.code in INFORMS}.values())
        note = (" Note: " + "; ".join(f"{f.message.rstrip('.')} ({f.code})" for f in told[:2]) + (f"; and {len(told) - 2} more" if len(told) > 2 else "") + ".") if told else ""
        return f"APPROVE {symbol}: score {verdict.score} ({verdict.grade}); {parts}.{note}"
    blockers = order_blockers(findings)
    reasons = []
    if blockers:
        heavy = "; ".join(f"{f.message.rstrip('.')} ({f.code})" for f in blockers[:2])
        reasons.append(heavy + (f"; and {len(blockers) - 2} more" if len(blockers) > 2 else ""))
    elif "score_below_minimum" in verdict.denial_reasons:
        reasons.append(f"score {verdict.score} ({verdict.grade}) is under the {minimum} needed")
    people = needs_a_person(findings)
    if people:
        reasons.append("a person must look before this can trade: " + ", ".join(f"{f.code} ({f.message.rstrip('.')})" for f in people[:2]) + (f", and {len(people) - 2} more" if len(people) > 2 else ""))
    return f"DENY {symbol}: " + "; ".join(reasons) + "."


assert set(LABEL) == set(DIMENSIONS)
