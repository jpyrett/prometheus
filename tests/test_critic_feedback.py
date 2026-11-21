"""The Critic's feedback as metadata of the recommendation (Critic v1 Amendment 2): the `critic` block a user reads and a screen draws, and the final gate spelled out in it."""

from critic_helpers import NOW, captured, make_critic, review, tamper, view


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


