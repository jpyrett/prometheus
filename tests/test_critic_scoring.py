"""Score, verdict and reasons (Critic spec section 9; the verdict is Critic v3's, design/critic_v3_two_reasons_spec.md rules V1 to V9): plain arithmetic over findings and dimension scores."""
import pytest
from critic_helpers import CONFIG
from pydantic import ValidationError

from prometheus.agents.critic import reasons as rs
from prometheus.agents.critic import scoring as sc
from prometheus.agents.critic.common import blocker, info, warning
from prometheus.agents.critic.record import BLOCKER_CODES, BLOCKER_ORDER, Finding


def dims(accuracy=100, evidence=100, risk_fit=100, timing=100, value=100, chart=100, analysts=100, reduce=False):
    scores = dict(accuracy=accuracy, evidence=evidence, risk_fit=risk_fit, timing=timing, value=value, chart=chart, analysts=analysts)
    return {n: sc.make_dimension(n, CONFIG, s, applicable=not (reduce and n in ("risk_fit", "value", "chart"))) for n, s in scores.items()}


def test_the_score_never_denies_a_low_score_with_no_blocker_is_approved_and_the_grade_says_how_low():
    """AMENDED (Critic v3, rule V8): a score under 70 with no blocker was `score_below_minimum`, a denial. The score ranks and informs now; it has no mark."""
    v = sc.decide([], dims(accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60, analysts=60), 0, CONFIG)
    assert (v.approved, v.score, v.grade, v.denial_reasons, v.blocked) == (True, 60, "D", [], False)
    assert sc.decide([], dims(accuracy=0, evidence=0, risk_fit=0, timing=0, value=0, chart=0, analysts=0), 0, CONFIG).approved


def test_a_blocker_must_have_a_code_of_the_closed_list_of_its_dimension():
    with pytest.raises(ValidationError, match="not a blocker code of value"):
        Finding(code="market_closed", dimension="value", severity="blocker", message="x")
    with pytest.raises(ValidationError):
        Finding(code="made_up", dimension="value", severity="blocker", message="x")
    with pytest.raises(ValidationError):
        Finding(code="analysts_bearish", dimension="analysts", severity="blocker", message="x")             # the analysts' sign is a warning: the dimension has no blocker
    assert Finding(code="made_up", dimension="value", severity="warning", message="x").severity == "warning"       # a warning may have any code
    assert all(Finding(code=code, dimension=d, severity="blocker", message="x") for d, codes in BLOCKER_CODES.items() for code in codes) and len(BLOCKER_ORDER) == 26


# ---------------------------------------------------------------- V1: the class table is closed

def test_every_blocker_code_is_in_exactly_one_class_and_every_sign_of_a_group_is_in_one_group():
    classes = {}
    for code in rs.FAULTS:
        classes.setdefault(code, []).append(rs.FAULT)
    for code in rs.PROFILE_CODES:
        classes.setdefault(code, []).append(rs.PROFILE)
    for group, codes in rs.SIGNS.items():
        for code in codes:
            classes.setdefault(code, []).append(group)
    assert all(len(found) == 1 for found in classes.values()), {c: f for c, f in classes.items() if len(f) != 1}
    assert not [code for code in BLOCKER_ORDER if rs.class_of(code) is None], "a blocker code is in no class"
    assert set(rs.CLASS_OF) == set(classes) and set(rs.GROUPS) == {"news", "performance", "price"}
    assert set(rs.SIGNS["news"]) == {"fresh_negative_event", "fresh_negative_news", "event_not_supported", "analysts_bearish"}
    assert set(rs.SIGNS["performance"]) == {"trend_not_confirmed", "dead_cat_bounce", "falling_knife", "price_below_zone", "chart_bearish", "poor_performance"}
    assert set(rs.SIGNS["price"]) == {"overpriced", "valuation_extreme", "price_left_zone"}
    assert set(rs.PROFILE_CODES) == {"risk_above_profile", "sector_excluded", "symbol_excluded", "price_below_profile_min"}
    assert {"record_defect", "record_not_reviewable", "symbol_unknown", "price_mismatch", "source_missing", "source_altered", "market_closed", "quote_stale", "no_live_price", "bars_stale",
            "record_expired", "time_stop_passed", "stop_already_hit", "data_unreliable"} == set(rs.FAULTS)
    assert set(rs.REASON_ORDER) >= set(rs.CLASS_OF) - set(rs.FAULTS) - set(rs.PROFILE_CODES) and rs.REASON_ORDER[:len(BLOCKER_ORDER)] == BLOCKER_ORDER


# ---------------------------------------------------------------- V5, V6: the verdict

def test_a_fault_or_a_profile_finding_denies_alone_and_caps_the_score():
    for code, dim in (("market_closed", "timing"), ("price_mismatch", "accuracy"), ("data_unreliable", "accuracy"), ("risk_above_profile", "risk_fit"), ("symbol_excluded", "risk_fit")):
        v = sc.decide([blocker(dim, code, "x")], dims(), 0, CONFIG)
        assert (v.approved, v.denial_reasons, v.score, v.score_before_blockers, v.blocked) == (False, [code], 49, 100, True), code


def test_one_bad_group_approves_with_the_reason_named_and_two_deny():
    """ONE bad group does not deny (V6): the sentence says it; no blocker is left on an approval, the score is not capped. Bad signs in TWO groups deny (V5), with the group names first in the reasons (V9)."""
    trend = blocker("chart", "trend_not_confirmed", "The strategy is trend, but the return is -2.1%.")
    news = warning("evidence", "fresh_negative_news", "Newer stored text looks bad for PG.")
    price = blocker("value", "overpriced", "The price 167.10 is 50.0% above its six-month average.")
    one = sc.decide([trend], dims(), 0, CONFIG)
    assert (one.approved, one.denial_reasons, one.score, one.blocked) == (True, [], 100, False) and one.groups == {"news": [], "performance": ["trend_not_confirmed"], "price": []}
    assert sc.partition([trend], one) == ([], [trend.model_copy(update={"severity": "warning"})], [])
    assert sc.summarize("PG", one, [trend], dims()).endswith("Note: one reason against it: performance (trend_not_confirmed: The strategy is trend, but the return is -2.1%).")
    assert sc.decide([trend, blocker("chart", "dead_cat_bounce", "x"), warning("chart", "poor_performance", "x")], dims(), 0, CONFIG).approved       # three signs of ONE group are still one group
    two = sc.decide([trend, news], dims(), 0, CONFIG)
    assert (two.approved, two.denial_reasons, two.score, two.blocked) == (False, ["group:news", "group:performance", "trend_not_confirmed", "fresh_negative_news"], 49, True)
    assert sc.partition([trend, news], two)[0] == [trend] and sc.partition([trend, news], two)[1] == [news]
    three = sc.decide([trend, news, price], dims(), 0, CONFIG)
    assert three.denial_reasons == ["group:news", "group:performance", "group:price", "overpriced", "trend_not_confirmed", "fresh_negative_news"]       # the blockers in BLOCKER_ORDER, then the warnings in the order of the groups


def test_a_deny_sentence_names_the_groups_in_words_with_the_heaviest_code_of_each():
    trend = blocker("chart", "trend_not_confirmed", "The strategy is trend, but the return is -2.1%.")
    event = blocker("evidence", "fresh_negative_event", "Newer stored text reports earnings_miss for XOM.")
    other = warning("evidence", "analysts_bearish", "The analysts lean against XOM.")
    price = blocker("value", "overpriced", "The price is 50.0% above its average.")
    f = [event, trend, other]
    assert sc.summarize("XOM", sc.decide(f, dims(), 0, CONFIG), f, dims()) == ("DENY XOM: poor news (fresh_negative_event: Newer stored text reports earnings_miss for XOM) and "
                                                                          "poor performance (trend_not_confirmed: The strategy is trend, but the return is -2.1%).")
    f = [other, trend, price]       # the heaviest code of a group is a blocker before a warning, and of two blockers the first of the closed order
    assert sc.summarize("XOM", sc.decide(f, dims(), 0, CONFIG), f, dims()).startswith("DENY XOM: poor news (analysts_bearish: The analysts lean against XOM), poor performance (trend_not_confirmed: ")
    assert sc.summarize("XOM", sc.decide(f, dims(), 0, CONFIG), f, dims()).endswith("and a price that is too high (overpriced: The price is 50.0% above its average).")


def test_a_fault_and_two_bad_groups_are_all_named_and_a_lone_fault_keeps_the_old_sentence():
    closed = blocker("timing", "market_closed", "The market is closed.")
    trend = blocker("chart", "trend_not_confirmed", "x")
    news = warning("evidence", "fresh_negative_news", "y")
    f = [closed, trend, news]
    v = sc.decide(f, dims(), 0, CONFIG)
    assert v.denial_reasons == ["group:news", "group:performance", "market_closed", "trend_not_confirmed", "fresh_negative_news"]
    assert sc.summarize("PG", v, f, dims()) == "DENY PG: The market is closed (market_closed); poor news (fresh_negative_news: y) and poor performance (trend_not_confirmed: x)."
    assert sc.summarize("PG", sc.decide([closed], dims(), 0, CONFIG), [closed], dims()) == "DENY PG: The market is closed (market_closed)."


def test_the_pattern_bonus_moves_the_score_and_the_score_stays_between_0_and_100():
    assert sc.decide([], dims(accuracy=80, evidence=80, risk_fit=80, timing=80, value=80, chart=80, analysts=80), 0, CONFIG, pattern_bonus=8.0).score == 88
    assert sc.decide([], dims(accuracy=80, evidence=80, risk_fit=80, timing=80, value=80, chart=80, analysts=80), 0, CONFIG, pattern_bonus=-8.0).score == 72
    assert sc.weighted_score(dims(), 10, 8.0) == 100 and sc.weighted_score(dims(accuracy=0, evidence=0, risk_fit=0, timing=0, value=0, chart=0, analysts=0), 0, -8.0) == 0


# ---------------------------------------------------------------- the summary sentence

def summarize(findings, **kw):
    d = dims(**kw)
    v = sc.decide(findings, d, 0, CONFIG)
    return sc.summarize("PG", v, findings, d)


def test_an_approval_with_a_low_score_says_its_points_and_no_reason_to_deny():
    assert summarize([], accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60, analysts=60) == ("APPROVE PG: score 60 (D); accuracy 15/25, evidence 9/15, risk fit 9/15, timing 6/10, value 9/15, chart 6/10, analysts 6/10.")


# ---------------------------------------------------------------- a finding that asks for a person is a denial (entry 103)

def test_a_finding_that_asks_for_a_person_denies_last_among_the_reasons_and_never_caps_the_score():
    ask = warning("evidence", "evidence_not_reviewed", "The quotes could not be reviewed by the model: no key.", requires_human=True)
    v = sc.decide([ask], dims(), 0, CONFIG)
    assert (v.approved, v.score, v.score_before_blockers, v.denial_reasons, v.requires_human, v.requires_human_reasons) == (False, 100, 100, ["needs_human_review"], True, ["evidence_not_reviewed"])
    under = sc.decide([ask], dims(accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60, analysts=60), 0, CONFIG)
    assert under.denial_reasons == ["needs_human_review"] and under.score == 60            # AMENDED (V8): the score is no reason of its own
    stopped = sc.decide([ask, blocker("timing", "market_closed", "The market is closed.")], dims(), 0, CONFIG)
    assert stopped.denial_reasons == ["market_closed", "needs_human_review"] and stopped.score == 49 and stopped.score_before_blockers == 100
    assert sc.decide([warning("value", "x", "a warning that asks for nobody")], dims(), 0, CONFIG).approved
    assert summarize([ask]) == "DENY PG: a person must look before this can trade: evidence_not_reviewed (The quotes could not be reviewed by the model: no key)."
    assert summarize([ask, blocker("timing", "market_closed", "The market is closed.")]) == ("DENY PG: The market is closed (market_closed); a person must look before this can trade: "
                                                                                          "evidence_not_reviewed (The quotes could not be reviewed by the model: no key).")
    lone = blocker("chart", "trend_not_confirmed", "x")      # denied only because a person must look: a sign that does not deny is not a blocker, and nothing caps the score
    v = sc.decide([ask, lone], dims(), 0, CONFIG)
    assert v.score == 100 and not v.blocked and sc.partition([ask, lone], v)[0] == []
