"""Score, verdict and reasons (Critic spec section 9; the verdict is Critic v3's, design/critic_v3_two_reasons_spec.md rules V1 to V9): plain arithmetic over findings and dimension scores."""
import pytest
from critic_helpers import CONFIG
from pydantic import ValidationError

from prometheus.agents.critic import scoring as sc
from prometheus.agents.critic.common import blocker, warning
from prometheus.agents.critic.record import BLOCKER_CODES, BLOCKER_ORDER, Finding


def dims(accuracy=100, evidence=100, risk_fit=100, timing=100, value=100, chart=100, analysts=100, reduce=False):
    scores = dict(accuracy=accuracy, evidence=evidence, risk_fit=risk_fit, timing=timing, value=value, chart=chart, analysts=analysts)
    return {n: sc.make_dimension(n, CONFIG, s, applicable=not (reduce and n in ("risk_fit", "value", "chart"))) for n, s in scores.items()}


def test_a_blocker_must_have_a_code_of_the_closed_list_of_its_dimension():
    with pytest.raises(ValidationError, match="not a blocker code of value"):
        Finding(code="market_closed", dimension="value", severity="blocker", message="x")
    with pytest.raises(ValidationError):
        Finding(code="made_up", dimension="value", severity="blocker", message="x")
    with pytest.raises(ValidationError):
        Finding(code="analysts_bearish", dimension="analysts", severity="blocker", message="x")             # the analysts' sign is a warning: the dimension has no blocker
    assert Finding(code="made_up", dimension="value", severity="warning", message="x").severity == "warning"       # a warning may have any code
    assert all(Finding(code=code, dimension=d, severity="blocker", message="x") for d, codes in BLOCKER_CODES.items() for code in codes) and len(BLOCKER_ORDER) == 26


# ---------------------------------------------------------------- the summary sentence

def summarize(findings, **kw):
    d = dims(**kw)
    v = sc.decide(findings, d, 0, CONFIG)
    return sc.summarize("PG", v, findings, d)
