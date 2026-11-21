"""Which reason is which: the closed table of the Critic's version 3 (design/critic_v3_two_reasons_spec.md rules V1 to V4, design choices entry 137).

Jacob, 2026-11-20: *"Critic shouldn't nitpick about individual financial metrics. There should be more than one reason for a denial. A denial should be mostly due to poor news sentiment and poor stock performance or is
too risky for the user's published profile risk level or denied because the price is not a good buy any longer (too expensive)."*

Every finding that can deny belongs to exactly ONE class:

    fault        the recommendation cannot be traded safely or cannot be checked (a broken record, a price that does not match, a closed market, a stale quote, ...): denies ALONE
    profile      it is not for this user (the risk above the profile, an excluded symbol or sector, a price under the user's minimum): denies ALONE (it is about who the user is, not about the stock)
    news         a SIGN, one of three groups of judgment: bad news (a fresh negative event, fresh negative news, an event the source did not support, the analysts leaning bearish)
    performance  a SIGN: the stock is doing badly (a trend that is not confirmed, a dead cat bounce, a falling knife, the price under the entry zone, a bearish chart, poor returns)
    price        a SIGN: the price is not a good buy any longer (overpriced, an extreme valuation, above the entry zone)

A group is BAD when at least one of its signs fired. ONE bad group does not deny (the approval says it); bad signs in TWO groups deny. A finding that asks for a person to look (`needs_human_review`, entry 103) is a fault and
is handled by `scoring.decide`, not by this table. The table is closed: a test fails when a blocker code (`record.BLOCKER_CODES`) is in no class or in two, so a new reason cannot be added without saying what it is.

The classes decide the VERDICT only; what a finding says, and whether it is a blocker or a warning, is where it is raised (`recheck`, `timing`, `risk`, `value`, `chart`, `evidence`, `analysts`). A sign that did not deny is a warning on the review.
"""
from prometheus.agents.critic.record import BLOCKER_ORDER, Finding

FAULT, PROFILE = "fault", "profile"
GROUPS = ("news", "performance", "price")
GROUP_WORDS = {"news": "poor news", "performance": "poor performance", "price": "a price that is too high"}      # the groups as a DENY sentence says them (rule V9)

FAULTS = ("record_not_reviewable", "record_defect", "symbol_unknown", "price_mismatch", "source_missing", "source_altered", "data_unreliable",
          "market_closed", "quote_stale", "no_live_price", "bars_stale", "record_expired", "time_stop_passed", "stop_already_hit")
PROFILE_CODES = ("risk_above_profile", "sector_excluded", "symbol_excluded", "price_below_profile_min")
SIGNS: dict[str, tuple[str, ...]] = {
    "news": ("fresh_negative_event", "fresh_negative_news", "event_not_supported", "analysts_bearish"),
    "performance": ("trend_not_confirmed", "dead_cat_bounce", "falling_knife", "price_below_zone", "chart_bearish", "poor_performance"),
    "price": ("overpriced", "valuation_extreme", "price_left_zone"),
}

CLASS_OF: dict[str, str] = {**{code: FAULT for code in FAULTS}, **{code: PROFILE for code in PROFILE_CODES}, **{code: group for group, codes in SIGNS.items() for code in codes}}
assert len(CLASS_OF) == len(FAULTS) + len(PROFILE_CODES) + sum(len(codes) for codes in SIGNS.values()), "a code is in two classes"

# the order the reasons are named in: the blockers (`BLOCKER_ORDER`: accuracy, timing, risk fit, value, chart, evidence), then the signs that are only ever warnings, in the order of the groups
SIGN_CODES = tuple(code for codes in SIGNS.values() for code in codes)
REASON_ORDER = (*BLOCKER_ORDER, *(code for code in SIGN_CODES if code not in BLOCKER_ORDER))


def class_of(code: str) -> str | None:
    """The class of a finding code (`fault`, `profile`, `news`, `performance` or `price`), or None for a code that cannot deny (a plain warning or a note)."""
    return CLASS_OF.get(code)


def is_sign(code: str) -> bool:
    return CLASS_OF.get(code) in GROUPS


def groups_of(findings: list[Finding]) -> dict[str, list[Finding]]:
    """{group: the findings of that group's signs that fired}: always the three groups, in order, an empty list for a clean one. Any finding counts (a blocker or a warning), whatever its severity."""
    return {group: [f for f in findings if CLASS_OF.get(f.code) == group] for group in GROUPS}


def bad_groups(findings: list[Finding]) -> list[str]:
    """The names of the groups with at least one sign, in the order of `GROUPS`."""
    return [group for group, found in groups_of(findings).items() if found]


def heaviest(found: list[Finding]) -> Finding:
    """The finding to name for a group: the first by `REASON_ORDER` (a blocker before a warning), then the first raised."""
    return min(found, key=lambda f: REASON_ORDER.index(f.code) if f.code in REASON_ORDER else len(REASON_ORDER))


def codes_of(found: list[Finding]) -> list[str]:
    """The distinct codes of some findings, in `REASON_ORDER`."""
    return sorted({f.code for f in found}, key=lambda c: REASON_ORDER.index(c) if c in REASON_ORDER else len(REASON_ORDER))
