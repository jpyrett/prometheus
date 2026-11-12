"""The Critic's feedback as metadata of the recommendation (Critic v1 Amendment 2): the `critic` block a user reads and a screen draws, and the final gate spelled out in it."""
import copy
import json

import pytest
from critic_helpers import CONFIG, NOW, at, captured, make_critic, portfolio, profile, review, tamper, view

from prometheus.agents.account_manager.decisions import decide_trades, record_user_decision
from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic import feedback as fb
from prometheus.agents.critic.config import DIMENSIONS


def reviewed(name="PG_open", now=NOW, **options):
    order = captured(name)
    make_critic(order, **options).review_recommendations(order, view(), portfolio(), now)
    return order


def test_every_reviewed_recommendation_carries_the_block_and_the_review_it_comes_from():
    order = reviewed()
    block, record = order.recommendations[0]["critic"], order.recommendations[0]
    review_ = order.reviews[0]
    assert block["schema"] == 1 and block["review_id"] == review_["review_id"] and block["reviewed_utc"] == review_["reviewed_utc"] and block["valid_until_utc"] == review_["valid_until_utc"]
    assert block["verdict"] == {"approved": True, "decision": "approve", "score": 88, "grade": "B", "band": "strong", "summary": review_["summary"]}
    assert block["decision"] is None                                                       # the Account Manager has not decided yet
    json.dumps(block)                                                                      # plain JSON: a screen or a person can have it as it is
    Recommendation.model_validate(record)                                                  # and the record still validates, with the block in it


def test_the_dimensions_are_in_a_fixed_order_and_their_points_add_up_to_the_blended_score():
    block = reviewed().recommendations[0]["critic"]
    rows = block["dimensions"]
    assert [r["name"] for r in rows] == list(DIMENSIONS) and [r["title"] for r in rows] == ["Accuracy", "Evidence", "Risk fit", "Timing", "Value", "Chart"]
    assert [r["weight"] for r in rows] == [30, 15, 20, 10, 15, 10] and sum(r["weight"] for r in rows) == 100
    assert all(r["applicable"] and r["headline"] and 0 <= r["share"] <= 1 for r in rows)
    assert sum(r["points"] for r in rows) == pytest.approx(87.5, abs=0.02)                # 30 + 15 + 18.42 + 10 + 6.58 + 7.5
    assert sum(r["weight_share"] for r in rows) == pytest.approx(1.0, abs=1e-3)
    assert {r["name"]: r["band"] for r in rows} == {"accuracy": "strong", "evidence": "strong", "risk_fit": "strong", "timing": "strong", "value": "weak", "chart": "ok"}


def test_the_final_gate_is_spelled_out():
    gate = reviewed().recommendations[0]["critic"]["gate"]
    assert gate["base_score"] == pytest.approx(87.5, abs=0.02) and gate["bonus"] == 0 and gate["weight_total"] == 100 and gate["score"] == 88 == gate["score_before_blockers"]
    assert (gate["approve_min_score"], gate["auto_min_score"], gate["blocked_score_cap"]) == (70, 80, 49)
    assert gate["passed_score"] is True and gate["blocked"] is False and gate["blocked_by"] == [] and gate["denial_reasons"] == [] and gate["margin"] == 18
    assert "sum(weight x dimension score)" in gate["formula"]


def test_a_blocker_shows_as_what_blocked_and_the_score_it_would_have_had():
    order = reviewed(now=at("2026-11-07 12:00"))                                            # a Saturday: the market is closed
    block = order.recommendations[0]["critic"]
    gate = block["gate"]
    assert block["verdict"]["decision"] == "deny" and block["verdict"]["score"] == 49 and block["verdict"]["band"] == "weak"
    assert gate["blocked"] and "market_closed" in gate["blocked_by"] and gate["passed_score"] is True and gate["score_before_blockers"] > 49 and gate["margin"] == 49 - 70
    codes = [b["code"] for b in block["findings"]["blockers"]]
    assert codes == gate["blocked_by"] and all(b["message"] and b["dimension"] for b in block["findings"]["blockers"])
    assert block["size"]["approved_shares"] == 0 and block["size"]["suggested_shares"] == 75


def test_the_warnings_and_the_need_for_a_person_are_in_the_block():
    block = reviewed(own={}).recommendations[0]["critic"]                                  # no six-month average: the value cannot be judged
    assert block["gate"]["passed_score"] is True and block["findings"]["warnings"] and all({"code", "dimension", "message", "requires_human"} <= set(w) for w in block["findings"]["warnings"])
    assert block["needs_human"] == {"required": True, "reasons": ["value_not_assessable"]}


def test_the_patterns_the_chart_found_are_in_the_block():
    block = reviewed().recommendations[0]["critic"]
    held = {p["name"] for p in block["patterns"]["holding"]}
    assert held == {"near_high", "macd_positive", "healthy_rsi", "obv_rising"} and block["patterns"]["points"] == 25 and "golden_cross" in block["patterns"]["unknown"]
    assert all({"name", "direction", "points", "evidence"} <= set(p) for p in block["patterns"]["holding"])


def test_a_reduction_has_rows_that_do_not_apply():
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"action.action": "reduce"})
    make_critic(order).review_recommendations(order, view(), portfolio(), NOW)
    rows = {r["name"]: r for r in order.recommendations[0]["critic"]["dimensions"]}
    assert [n for n, r in rows.items() if not r["applicable"]] == ["risk_fit", "value", "chart"]
    assert rows["value"]["score"] is None and rows["value"]["points"] is None and rows["value"]["band"] is None and rows["value"]["headline"] == "not judged for a reduction"
    assert order.recommendations[0]["critic"]["gate"]["weight_total"] == 55


def test_the_account_managers_decision_is_added_to_the_block_and_follows_the_users_answer():
    order = reviewed()
    decide_trades(order, portfolio(), profile(), 80, now=NOW)                               # Ask mode, nobody to ask: pending
    block = order.recommendations[0]["critic"]
    assert block["decision"]["will_trade"] == "pending_user" and block["decision"]["mode"] == "ask" and block["decision"]["shares"] == 44 and block["decision"]["asked_user"] is False
    record_user_decision(order, order.recommendations[0]["identity"]["record_id"], True, now=NOW)
    assert order.recommendations[0]["critic"]["decision"]["will_trade"] == "yes_user" and order.recommendations[0]["critic"]["decision"]["asked_user"] is True
    denied = reviewed(now=at("2026-11-07 12:00"))
    decide_trades(denied, portfolio(), profile(), 80, now=at("2026-11-07 12:00"))
    assert denied.recommendations[0]["critic"]["decision"]["will_trade"] == "no_system_denied"


def test_the_advisors_own_fields_are_untouched_and_a_record_that_was_not_reviewed_has_no_block():
    order = captured("PG_open")
    before = copy.deepcopy(order.recommendations)
    assert all(r.get("critic") is None for r in order.recommendations)
    make_critic(order).review_recommendations(order, view(), portfolio(), NOW)
    assert [{k: v for k, v in r.items() if k != "critic"} for r in order.recommendations] == [{k: v for k, v in r.items() if k != "critic"} for r in before]
    assert Recommendation.model_validate(before[0]).critic is None


def test_the_band_marks_come_from_the_grades_of_the_config():
    assert [fb.band(s, CONFIG) for s in (100, 80, 79.9, 60, 59.9, 0)] == ["strong", "strong", "ok", "ok", "weak", "weak"]


def test_one_review_per_recommendation_each_with_its_own_block():
    order = captured("SNDK")
    review(order)                                                                           # the helper reviews the order it is given
    assert order.recommendations[0]["critic"]["review_id"] == order.reviews[0]["review_id"] and order.recommendations[0]["critic"]["verdict"]["decision"] == "deny"
    assert order.recommendations[0]["critic"]["gate"]["blocked_by"][0] == "risk_above_profile"
