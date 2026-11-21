"""Score, verdict and reasons (Critic v1 spec section 9, step 9; the verdict is Critic v3's, design/critic_v3_two_reasons_spec.md rules V5 to V9). Plain arithmetic over the findings and the seven dimensions' scores: no model, no network.

    score  = min(100, round(sum(weight x dimension score) / sum(weights of the applicable dimensions) + the great-deal bonus + the chart-pattern bonus))
    score  = min(score, blocked_score_cap)   when the review is denied for a reason that is a blocker (a fault, a profile finding, or bad signs in two groups)

    DENIED   a fault or a profile finding fired (it denies alone), or at least TWO of the three groups news, performance and price are bad (`reasons.py`), or a finding asks for a person to look
    APPROVED otherwise: nothing between (design choices entry 103). ONE bad group does not deny: the approval's summary names it ("one reason against it").

THE SCORE NO LONGER DENIES (rule V8): there is no mark it must reach. It ranks and informs (the score and the grade stay on the card), and the cap means one number tells the whole story: a recommendation that was denied for a blocker never shows 85.
The score before the cap is kept (`score_before_blockers`), so the report can say "would have been 77".
A finding that asks for a person to look (`requires_human`) DENIES the recommendation (`needs_human_review`; design choices entry 103, Jacob 2026-11-15: *"if the Critic wants a person to review, then the Critic needs to Deny
the recommendation and return the reason"*): that denial is NOT a blocker and does not cap the score: it says the review could not be trusted without a person, not that the recommendation is bad. `requires_human` and
`requires_human_reasons` stay on the review: they name the finding.
The summary sentence is built from fixed wording and the findings' own messages: a model never writes a line the user reads. A DENY sentence names the two (or three) groups in words with the heaviest code of each; an APPROVE sentence
names a single bad group and the finding that still informs without denying (`INFORMS`).
"""
from dataclasses import dataclass, field

from prometheus.agents.critic.config import DIMENSIONS, CriticConfig
from prometheus.agents.critic.reasons import FAULT, GROUP_WORDS, GROUPS, PROFILE, class_of, codes_of, groups_of, heaviest
from prometheus.agents.critic.record import BLOCKER_ORDER, Dimension, Finding, Measure
from prometheus.agents.critic.report import scorecard  # noqa: F401  (the card the Account Manager shows; defined with the report so that scoring stays pure arithmetic)

# The finding that INFORMS an approval without being a sign of a group (Critic v2, rule C3): `blend_not_supported` stopped denying, so an APPROVE sentence says it, or the user would learn of a blend the quotes no longer support only from the card.
# (`fresh_negative_news` informed too in version 2; in version 3 it is a news sign and is named as the one reason against an approval.)
INFORMS = ("blend_not_supported",)
LABEL = {"accuracy": "accuracy", "evidence": "evidence", "risk_fit": "risk fit", "timing": "timing", "value": "value", "chart": "chart", "analysts": "analysts"}


@dataclass
class Verdict:
    approved: bool
    score: int
    score_before_blockers: int
    grade: str
    denial_reasons: list[str] = field(default_factory=list)
    requires_human: bool = False                                        # a finding asked for a person to look: the recommendation is denied (`needs_human_review`)
    requires_human_reasons: list[str] = field(default_factory=list)     # the codes of those findings
    groups: dict[str, list[str]] = field(default_factory=dict)          # {group: the codes of the signs that fired}: a group with a code is bad
    blocked: bool = False                                               # denied for a fault, a profile finding or bad signs in two groups: the score is capped and the review lists its blockers


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


def weighted_score(dimensions: dict[str, Dimension], bonus: float, pattern_bonus: float = 0.0) -> int:
    """`min(100, round(sum(weight x score) / sum(weights of the applicable dimensions) + bonus + pattern_bonus))`, never under 0 (a bearish chart lowers the score)."""
    applicable = [d for d in dimensions.values() if d.applicable]
    total = sum(d.weight for d in applicable)
    return max(0, min(100, round(sum(d.weight * d.score for d in applicable) / total + bonus + pattern_bonus))) if total else 0


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


def decide(findings: list[Finding], dimensions: dict[str, Dimension], bonus: float, config: CriticConfig, pattern_bonus: float = 0.0) -> Verdict:
    """The verdict of one review: approved or denied, nothing in between (rule V5). DENIED when a fault or a profile blocker fired (it denies alone; a blocker whose code is in no class denies alone too: what could not be
    classified is not approved), or when at least two of the three groups are bad, or when a finding asks for a person to look (`needs_human_review`, last in `denial_reasons`, the score left as the dimensions gave it). Otherwise
    APPROVED, whatever the score. `denial_reasons` (rule V9): for bad signs in two groups the group names first (`group:news`, `group:performance`), then the codes that denied in `reasons.REASON_ORDER`; the score is capped at
    `blocked_score_cap` when the denial has a blocker reason (anything but the person alone)."""
    before = weighted_score(dimensions, bonus, pattern_bonus)
    blockers = order_blockers(findings)
    alone = [f for f in blockers if class_of(f.code) in (FAULT, PROFILE, None)]
    grouped = groups_of(findings)
    bad = [g for g in GROUPS if grouped[g]]
    two = len(bad) >= 2
    blocked = bool(alone) or two
    reasons = [f"group:{g}" for g in bad] if two else []
    reasons += codes_of([*alone, *(f for g in bad for f in grouped[g])] if two else alone)
    human = [f.code for f in needs_a_person(findings)]
    if human:
        reasons.append("needs_human_review")
    score = min(before, config.verdict.blocked_score_cap) if blocked else before
    return Verdict(approved=not reasons, score=score, score_before_blockers=before, grade=grade_of(score, config), denial_reasons=reasons, requires_human=bool(human), requires_human_reasons=human,
                   groups={g: codes_of(found) for g, found in grouped.items()}, blocked=blocked)


def partition(findings: list[Finding], verdict: Verdict) -> tuple[list[Finding], list[Finding], list[Finding]]:
    """(blockers, warnings, notes) of a review. A blocker is what denied it: when the review was denied for a blocker reason (`verdict.blocked`) the blockers are listed as raised, in `BLOCKER_ORDER`; when it was not (approved, or
    denied only because a person must look) a finding that was raised as a blocker was a sign of ONE bad group that does not deny: it is shown as a warning, so an approved review has no blocker."""
    blockers = order_blockers(findings) if verdict.blocked else []
    warnings = [f if f.severity == "warning" else f.model_copy(update={"severity": "warning"}) for f in findings if f.severity == "warning" or (f.severity == "blocker" and not verdict.blocked)]
    return blockers, warnings, [f for f in findings if f.severity == "info"]


def great_deal_bonus_allowed(findings: list[Finding], quality_multiplier: float, red_flag_passed: bool, min_quality: float) -> bool:
    """The bonus is earned only by a recommendation that is otherwise sound: no blocker in accuracy, value or chart (a discount on a dead cat bounce is a trap), quality at least `min_quality`, and the Advisor's red-flag gate passed."""
    return not any(f.severity == "blocker" and f.dimension in ("accuracy", "value", "chart") for f in findings) and quality_multiplier >= min_quality and red_flag_passed


def _sign(group: str, found: list[Finding]) -> str:
    """One group in a sentence: `performance (trend_not_confirmed: The strategy is trend, ...)`, with the heaviest finding of the group."""
    f = heaviest(found)
    return f"{group} ({f.code}: {f.message.rstrip('.')})"


def _in_words(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def summarize(symbol: str, verdict: Verdict, findings: list[Finding], dimensions: dict[str, Dimension]) -> str:
    """One sentence, from fixed wording. APPROVE: the verdict, the score, the points of each dimension, and a note for the one bad group (rule V6: `one reason against it: performance (trend_not_confirmed: ...)`) and for the
    findings that inform (`INFORMS`). DENY: the blockers that denied alone (the two heaviest), the groups in words with the heaviest code of each (rule V9: `DENY XOM: poor news (fresh_negative_event: ...) and poor
    performance (trend_not_confirmed: ...).`), and a person who must look (`DENY YUM: a person must look before this can trade: evidence_not_reviewed (The quotes could not be reviewed by the model: ...).`)."""
    grouped = groups_of(findings)
    bad = [g for g in GROUPS if grouped[g]]
    if verdict.approved:
        parts = ", ".join(f"{LABEL[n]} {d.points:.0f}/{d.weight:g}" for n, d in dimensions.items() if d.applicable)
        told = list({f.code: f for f in findings if f.severity == "warning" and f.code in INFORMS}.values())
        notes = ([f"one reason against it: {_sign(bad[0], grouped[bad[0]])}"] if len(bad) == 1 else []) + [f"{f.message.rstrip('.')} ({f.code})" for f in told[:2]]
        note = (" Note: " + "; ".join(notes) + (f"; and {len(told) - 2} more" if len(told) > 2 else "") + ".") if notes else ""
        return f"APPROVE {symbol}: score {verdict.score} ({verdict.grade}); {parts}.{note}"
    reasons = []
    alone = [f for f in order_blockers(findings) if class_of(f.code) in (FAULT, PROFILE, None)]
    if alone:
        heavy = "; ".join(f"{f.message.rstrip('.')} ({f.code})" for f in alone[:2])
        reasons.append(heavy + (f"; and {len(alone) - 2} more" if len(alone) > 2 else ""))
    if len(bad) >= 2:
        reasons.append(_in_words([f"{GROUP_WORDS[g]} ({heaviest(grouped[g]).code}: {heaviest(grouped[g]).message.rstrip('.')})" for g in bad]))
    people = needs_a_person(findings)
    if people:
        reasons.append("a person must look before this can trade: " + ", ".join(f"{f.code} ({f.message.rstrip('.')})" for f in people[:2]) + (f", and {len(people) - 2} more" if len(people) > 2 else ""))
    return f"DENY {symbol}: " + "; ".join(reasons) + "."


assert set(LABEL) == set(DIMENSIONS)
