"""The Critic's feedback as metadata of the recommendation (Critic v1 Amendment 2): the `critic` block a user reads and a screen draws, and the final gate spelled out in it."""
import copy
import json

import pytest
from critic_helpers import CONFIG, NOW, at, captured, make_critic, profile, review, tamper, view

from prometheus.agents.account_manager.decisions import decide_trades, record_user_decision
from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic import feedback as fb
from prometheus.agents.critic.config import DIMENSIONS
from prometheus.common.portfolio import Portfolio


def reviewed(name="PG_open", now=NOW, **options):
    order = captured(name)
    make_critic(order, **options).review_recommendations(order, view(), now)
    return order


def test_a_reduction_is_denied_as_a_defect_and_the_status_on_the_recommendation_says_why_every_row_applies():
    """AMENDED (entry 105): a `reduce` had three rows that did not apply (risk fit, value, chart: "not judged for a reduction") and a weight total of 55. The Critic does not review a record that is not a recommendation to
    buy: it is a `record_defect`, and the `critic` block written on it (the status) is a denial with that reason; no row is "not applicable" and no headline says "reduction"."""
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"action.action": "reduce"})
    make_critic(order).review_recommendations(order, view(), NOW)
    block = order.recommendations[0]["critic"]
    rows = {r["name"]: r for r in block["dimensions"]}
    assert not block["verdict"]["approved"] and block["verdict"]["decision"] == "deny" and "the Critic does not judge a holding or a sale" in block["verdict"]["summary"]
    assert block["gate"]["blocked_by"] == ["record_defect"] and all(r["applicable"] for r in rows.values()) and block["gate"]["weight_total"] == 100
    assert not any("reduction" in str(r["headline"]) for r in rows.values())




def test_the_gate_of_a_version_3_review_has_no_mark_and_carries_the_pattern_bonus_and_the_groups():
    """Critic v3: the score has no mark to reach, so the gate holds no `approve_min_score`, `passed_score` or `margin`; it holds the pattern bonus, the groups and the cap that a denial gets. The review's analysts row is a row of the card."""
    order = reviewed()
    block = order.recommendations[0]["critic"]
    gate = block["gate"]
    assert not {"approve_min_score", "passed_score", "margin"} & set(gate) and gate["blocked_score_cap"] == 49 and gate["pattern_bonus"] == order.reviews[0]["pattern_bonus"] == 6.25
    assert gate["groups"] == {"news": [], "performance": [], "price": []} and gate["blocked"] is False and gate["denial_reasons"] == []
    assert [r["name"] for r in block["dimensions"]] == list(DIMENSIONS) and DIMENSIONS[-1] == "analysts"
    analysts = next(r for r in block["dimensions"] if r["name"] == "analysts")
    assert analysts["applicable"] is False and analysts["title"] == "Analysts" and analysts["headline"] == "no analyst ratings the day could know" and gate["weight_total"] == 90
