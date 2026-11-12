"""Score, verdict and reasons (Critic spec section 9): plain arithmetic over findings and dimension scores."""
import pytest
from critic_helpers import CONFIG
from pydantic import ValidationError

from prometheus.agents.critic import scoring as sc
from prometheus.agents.critic.common import blocker, info, warning
from prometheus.agents.critic.record import BLOCKER_CODES, BLOCKER_ORDER, Finding


def dims(accuracy=100, evidence=100, risk_fit=100, timing=100, value=100, chart=100, reduce=False):
    scores = dict(accuracy=accuracy, evidence=evidence, risk_fit=risk_fit, timing=timing, value=value, chart=chart)
    return {n: sc.make_dimension(n, CONFIG, s, applicable=not (reduce and n in ("risk_fit", "value", "chart"))) for n, s in scores.items()}


def test_a_dimension_earns_its_share_of_its_weight():
    d = sc.make_dimension("accuracy", CONFIG, 90.0)
    assert (d.weight, d.score, d.points, d.applicable) == (30, 90.0, 27.0, True)
    assert sc.make_dimension("value", CONFIG, 140.0).score == 100.0 and sc.make_dimension("value", CONFIG, -5.0).score == 0.0       # a score stays between 0 and 100


@pytest.mark.parametrize("score, grade", [(100, "A"), (90, "A"), (89, "B"), (80, "B"), (79, "C"), (70, "C"), (69, "D"), (60, "D"), (59, "F"), (0, "F")])
def test_the_grades_at_their_marks(score, grade):
    assert sc.grade_of(score, CONFIG) == grade


def test_the_score_is_the_weighted_mean_plus_the_bonus_never_over_100():
    assert sc.weighted_score(dims(), 0) == 100 and sc.weighted_score(dims(accuracy=0), 0) == 70
    assert sc.weighted_score(dims(accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60), 0) == 60
    assert sc.weighted_score(dims(value=50), 2.5) == min(100, round(100 - 0.15 * 50 + 2.5)) == 95 and sc.weighted_score(dims(), 10) == 100
    assert sc.weighted_score(dims(chart=0), 0) == 90                                           # the chart is worth 10 of the 100 points


def test_a_reduction_is_scored_over_three_dimensions_only():
    d = dims(accuracy=100, evidence=50, timing=100, risk_fit=0, value=0, chart=0, reduce=True)
    assert sc.weighted_score(d, 0) == round((30 * 100 + 15 * 50 + 10 * 100) / 55) == 86


def test_a_review_with_no_applicable_dimension_scores_zero():
    assert sc.weighted_score({}, 0) == 0


def test_with_no_blocker_and_a_good_score_it_is_approved():
    v = sc.decide([warning("value", "thin_edge", "x")], dims(value=0), 0, CONFIG)
    assert (v.approved, v.score, v.score_before_blockers, v.grade, v.denial_reasons) == (True, 85, 85, "B", [])


def test_a_blocker_denies_whatever_the_score_and_caps_it():
    v = sc.decide([blocker("timing", "market_closed", "closed")], dims(), 0, CONFIG)
    assert (v.approved, v.score, v.score_before_blockers, v.grade, v.denial_reasons) == (False, 49, 100, "F", ["market_closed"])


def test_a_blocked_score_never_shows_more_than_the_cap_but_a_lower_one_is_kept():
    v = sc.decide([blocker("value", "overpriced", "x")], dims(accuracy=0, evidence=0, risk_fit=0), 0, CONFIG)
    assert v.score == v.score_before_blockers == 35 and not v.approved       # 35 is under the cap of 49: the cap only lowers


def test_a_score_under_the_mark_with_no_blocker_is_denied_for_that_reason_alone():
    v = sc.decide([], dims(accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60), 0, CONFIG)
    assert (v.approved, v.score, v.denial_reasons) == (False, 60, ["score_below_minimum"])
    assert sc.decide([], dims(accuracy=70, evidence=70, risk_fit=70, timing=70, value=70, chart=70), 0, CONFIG).approved           # exactly the mark


def test_the_denial_reasons_follow_the_order_of_the_closed_list_whatever_order_they_were_found_in():
    found = [blocker("evidence", "fresh_negative_event", "e"), blocker("value", "overpriced", "v"), blocker("accuracy", "data_mismatch", "a"), blocker("timing", "market_closed", "t"), blocker("risk_fit", "risk_above_profile", "r")]
    assert sc.decide(found, dims(), 0, CONFIG).denial_reasons == ["data_mismatch", "market_closed", "risk_above_profile", "overpriced", "fresh_negative_event"]
    assert [f.code for f in sc.order_blockers(found)] == sc.decide(found, dims(), 0, CONFIG).denial_reasons


def test_a_finding_that_asks_for_a_person_is_collected_once():
    found = [warning("accuracy", "identity_unverified", "x", requires_human=True), warning("evidence", "control_failed", "y", requires_human=True), warning("accuracy", "identity_unverified", "z", requires_human=True),
             warning("value", "thin_edge", "w")]
    v = sc.decide(found, dims(), 0, CONFIG)
    assert v.requires_human and v.requires_human_reasons == ["identity_unverified", "control_failed"]
    assert not sc.decide([warning("value", "thin_edge", "w")], dims(), 0, CONFIG).requires_human


def test_a_blocker_must_have_a_code_of_the_closed_list_of_its_dimension():
    with pytest.raises(ValidationError, match="not a blocker code of value"):
        Finding(code="market_closed", dimension="value", severity="blocker", message="x")
    with pytest.raises(ValidationError):
        Finding(code="made_up", dimension="value", severity="blocker", message="x")
    assert Finding(code="made_up", dimension="value", severity="warning", message="x").severity == "warning"       # a warning may have any code
    assert all(Finding(code=code, dimension=d, severity="blocker", message="x") for d, codes in BLOCKER_CODES.items() for code in codes) and len(BLOCKER_ORDER) == 29


# ---------------------------------------------------------------- the summary sentence

def summarize(findings, shares=44, suggested=75, action="open", **kw):
    d = dims(**kw)
    v = sc.decide(findings, d, 0, CONFIG)
    return sc.summarize("PG", action, v, findings, d, shares, suggested, 70)


def test_an_approval_names_the_size_the_score_and_every_dimensions_points():
    assert summarize([], accuracy=100, evidence=94, risk_fit=92, timing=100, value=37, chart=100) == "APPROVE 44 of 75 shares of PG: score 88 (B); accuracy 30/30, evidence 14/15, risk fit 18/20, timing 10/10, value 6/15, chart 10/10."
    assert summarize([], shares=75).startswith("APPROVE 75 shares of PG: score 100 (A)")
    assert summarize([], action="add", shares=10, suggested=10).startswith("APPROVE add 10 shares of PG")
    assert summarize([], action="add", shares=5, suggested=10).startswith("APPROVE add 5 of 10 shares of PG")


def test_a_reduction_has_no_share_count_and_only_its_three_dimensions():
    s = sc.summarize("PG", "reduce", sc.decide([], dims(reduce=True), 0, CONFIG), [], dims(reduce=True), 12, 12, 70)
    assert s == "APPROVE reduction of PG: score 100 (A); accuracy 30/30, evidence 15/15, timing 10/10."


def test_a_denial_names_the_two_heaviest_blockers_in_the_order_of_the_list_and_counts_the_rest():
    found = [blocker("value", "falling_knife", "It fell 52% in 21 sessions."), blocker("risk_fit", "risk_above_profile", "The risk is very_high (70.0)."), blocker("timing", "market_closed", "The market is closed.")]
    assert summarize(found) == "DENY PG: The market is closed (market_closed); The risk is very_high (70.0) (risk_above_profile); and 1 more."
    assert summarize(found[:2]) == "DENY PG: The risk is very_high (70.0) (risk_above_profile); It fell 52% in 21 sessions (falling_knife)."
    assert summarize(found[:1]) == "DENY PG: It fell 52% in 21 sessions (falling_knife)."


def test_a_denial_for_the_score_alone_says_so():
    assert summarize([], accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60) == "DENY PG: score 60 (D) is under the 70 needed."


def test_the_summary_is_built_by_code_from_the_findings_and_never_by_a_model():
    assert "model" not in summarize([blocker("value", "overpriced", "The price is high.")]).lower()


def test_info_findings_do_not_change_the_verdict():
    assert sc.decide([info("evidence", "counter_evidence", "x")], dims(), 0, CONFIG).approved
