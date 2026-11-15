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


def test_a_reduction_has_rows_that_do_not_apply():
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"action.action": "reduce"})
    make_critic(order).review_recommendations(order, view(), NOW)
    rows = {r["name"]: r for r in order.recommendations[0]["critic"]["dimensions"]}
    assert [n for n, r in rows.items() if not r["applicable"]] == ["risk_fit", "value", "chart"]
    assert rows["value"]["score"] is None and rows["value"]["points"] is None and rows["value"]["band"] is None and rows["value"]["headline"] == "not judged for a reduction"
    assert order.recommendations[0]["critic"]["gate"]["weight_total"] == 55


