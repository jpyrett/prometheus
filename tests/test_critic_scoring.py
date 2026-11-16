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


def test_a_score_under_the_mark_with_no_blocker_is_denied_for_that_reason_alone():
    v = sc.decide([], dims(accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60), 0, CONFIG)
    assert (v.approved, v.score, v.denial_reasons) == (False, 60, ["score_below_minimum"])
    assert sc.decide([], dims(accuracy=70, evidence=70, risk_fit=70, timing=70, value=70, chart=70), 0, CONFIG).approved           # exactly the mark


def test_a_blocker_must_have_a_code_of_the_closed_list_of_its_dimension():
    with pytest.raises(ValidationError, match="not a blocker code of value"):
        Finding(code="market_closed", dimension="value", severity="blocker", message="x")
    with pytest.raises(ValidationError):
        Finding(code="made_up", dimension="value", severity="blocker", message="x")
    assert Finding(code="made_up", dimension="value", severity="warning", message="x").severity == "warning"       # a warning may have any code
    assert all(Finding(code=code, dimension=d, severity="blocker", message="x") for d, codes in BLOCKER_CODES.items() for code in codes) and len(BLOCKER_ORDER) == 27


# ---------------------------------------------------------------- the summary sentence

def summarize(findings, **kw):
    d = dims(**kw)
    v = sc.decide(findings, d, 0, CONFIG)
    return sc.summarize("PG", v, findings, d, 70)


def test_a_denial_for_the_score_alone_says_so():
    assert summarize([], accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60) == "DENY PG: score 60 (D) is under the 70 needed."




# ---------------------------------------------------------------- a finding that asks for a person is a denial (entry 103)

def test_a_finding_that_asks_for_a_person_denies_last_among_the_reasons_and_never_caps_the_score():
    ask = warning("evidence", "evidence_not_reviewed", "The quotes could not be reviewed by the model: no key.", requires_human=True)
    v = sc.decide([ask], dims(), 0, CONFIG)
    assert (v.approved, v.score, v.score_before_blockers, v.denial_reasons, v.requires_human, v.requires_human_reasons) == (False, 100, 100, ["needs_human_review"], True, ["evidence_not_reviewed"])
    under = sc.decide([ask], dims(accuracy=60, evidence=60, risk_fit=60, timing=60, value=60, chart=60), 0, CONFIG)
    assert under.denial_reasons == ["score_below_minimum", "needs_human_review"] and under.score == 60
    stopped = sc.decide([ask, blocker("timing", "market_closed", "The market is closed.")], dims(), 0, CONFIG)
    assert stopped.denial_reasons == ["market_closed", "needs_human_review"] and stopped.score == 49 and stopped.score_before_blockers == 100
    assert sc.decide([warning("value", "x", "a warning that asks for nobody")], dims(), 0, CONFIG).approved
    assert summarize([ask]) == "DENY PG: a person must look before this can trade: evidence_not_reviewed (The quotes could not be reviewed by the model: no key)."
    assert summarize([ask, blocker("timing", "market_closed", "The market is closed.")]) == ("DENY PG: The market is closed (market_closed); a person must look before this can trade: "
                                                                                          "evidence_not_reviewed (The quotes could not be reviewed by the model: no key).")
