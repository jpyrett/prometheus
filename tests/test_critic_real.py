"""The Critic on the three REAL captures (tests/fixtures/advisor_real), at Thursday 2026-11-05 10:30 Eastern, for a moderate profile (Critic spec 14.2, Appendix B).

The captures are the real work orders, metrics and records of the Advisor's run of that day; the Data Clerk's answers are those same numbers (the "everything matches" day), so what the Critic
decides here is the Critic's OWN judgement: the risk, the deal, the profit, the clock. The captures hold no six-month average: `own={}` gives the Critic none (the real case); the other tests give
it one as an INPUT for a rule.
"""
from critic_helpers import NOW, ReviewerModel, at, captured, review, view


def codes(r, kind="blockers"):
    return [f["code"] for f in r[kind]]


def test_sndk_is_denied_for_its_risk_and_because_a_collapse_is_not_a_discount():
    r = review(captured("SNDK"), own={})["SNDK"]
    assert not r["approved"] and {"risk_above_profile", "falling_knife"} <= set(codes(r)) and r["score"] <= 49 < r["score_before_blockers"]
    assert (r["risk"]["risk_score"], r["risk"]["risk_level"]) == (92.2, "very_high") and r["risk"]["allowed_level"] == "medium"     # 38.5 + 23.1 + 15.3 + 15.3: volatility, drawdown, beta and ATR all at the top; no stop, size or liquidity measure (entries 92 and 97)
    assert "fell 52% in 21 sessions" in next(b["message"] for b in r["blockers"] if b["code"] == "falling_knife")
    assert r["summary"].startswith("DENY SNDK: ") and "approved_shares" not in r and r["valid_until_utc"] is None


def test_pg_on_conviction_is_approved():
    """Before Amendment 5 this capture was not a recommendation (no strategy fired); with the `conviction` family its blend (78) makes it one, and since Amendment 3 the Critic no longer has a profit-potential
    rule (a profit floor and a thin-edge warning on the expected excess return after cost were removed: they misled about a company like Procter & Gamble).
    (Entry 103 first made this review a denial, because with `own={}` the capture has no six-month average and `value_not_assessable` asked for a person; Jacob then removed that rule: it is a plain note again, and the review
    is approved as it always was.)"""
    r = review(captured("PG"), own={})["PG"]
    assert r["approved"] and codes(r) == [] and "approved_shares" not in r and "thin_edge" not in codes(r, "warnings") and "excess_reviewed" not in r["value"]


def test_a_conviction_recommendation_whose_quotes_do_not_hold_is_approved_with_the_warning_and_fewer_evidence_points():
    """AMENDED (Critic v2, rule C3, design/critic_v2_evidence_spec.md). The conviction family stands on the blend. When the model says every quote says the opposite, the news part turns negative (-0.58) and the blend falls from
    0.57 to 0.11, under the 0.30 mark the strategy needs: `blend_not_supported`. It was a BLOCKER (the review was denied, its score capped at 49); in the four six-month tests of 2026-11-20 it fired 42 times and 37 of the 41 that closed were
    winners, so it is a WARNING now: the blend worked again is still calculated and shown, the evidence dimension still loses its points (4 of 15, against 15 of 15 when the quotes hold), and the review still says so."""
    held = review(captured("PG"))["PG"]
    r = review(captured("PG"), model=ReviewerModel(default="contradicts"))["PG"]
    c = r["evidence"]["counterfactual"]
    assert c["blend"] == 0.1111 and c["blend_recorded"] == 0.5725
    assert r["approved"] and codes(r) == [] and codes(r, "warnings") == ["blend_not_supported"] and r["denial_reasons"] == [] and r["requires_human"] is False
    assert (r["score"], r["grade"]) == (81, "B") and (held["score"], held["grade"]) == (93, "A")                       # Critic v3 (was 77 C and 88 B): the lower evidence points are the whole cost: 15 of 15 became 4.5 of 15
    assert (r["dimensions"]["evidence"]["points"], held["dimensions"]["evidence"]["points"]) == (4.5, 15.0)
    assert r["summary"].startswith("APPROVE PG: score 81 (B)") and "evidence 4/15" in r["summary"] and r["summary"].endswith("under the 0.3 the conviction strategy needs (blend_not_supported).")      # the approval says why it is not an easy one
    assert "blend_not_supported" not in held["summary"]


def test_pg_open_passes_every_check_and_is_approved_without_a_six_month_average_and_with_no_size():
    """AMENDED twice (entry 103). The warning `value_not_assessable` used to ask for a person (`requires_human`), and Part A of entry 103 made that a denial; Jacob then removed it ("it eliminates IPOs from being traded"):
    it is a plain note, it asks for nobody, and this review is approved at 89 again."""
    r = review(captured("PG_open"), own={})["PG"]
    assert r["approved"] and r["blockers"] == [] and not {"approved_shares", "suggested_shares"} & set(r) and r["value"]["deal"] == "unknown" and r["score"] == 94 and r["valid_until_utc"]      # Critic v3: 89 under the old weights (the analysts are out of the average, the net chart points are a bonus)
    assert "value_not_assessable" in codes(r, "warnings") and r["requires_human"] is False and r["requires_human_reasons"] == [] and r["denial_reasons"] == []
    assert r["risk"]["risk_score"] == 6.5 and r["risk"]["risk_level"] == "low" and r["dimensions"]["accuracy"]["score"] == 100


def test_the_same_three_with_a_six_month_average_four_percent_under_the_price():
    reviews = {name: next(iter(review(captured(name)).values())) for name in ("PG", "PG_open")}       # SNDK's denial is the first test of this file (its own real numbers)
    assert reviews["PG"]["approved"] and codes(reviews["PG"]) == []
    assert reviews["PG_open"]["approved"] and reviews["PG_open"]["value"]["deal"] == "fair" and reviews["PG_open"]["requires_human"] is False


def test_at_night_or_at_the_weekend_all_three_are_denied_and_say_what_the_score_would_have_been():
    for moment in ("2026-11-05 22:13", "2026-11-07 12:00"):
        for name, symbol in (("SNDK", "SNDK"), ("PG", "PG"), ("PG_open", "PG")):
            r = review(captured(name), now=at(moment))[symbol]
            assert not r["approved"] and "market_closed" in codes(r) and r["score"] <= 49


def test_an_aggressive_user_still_cannot_have_sndk_at_very_high_risk_and_a_conservative_one_may_have_pg_open():
    assert "risk_above_profile" in codes(review(captured("SNDK"), view_=view(risk_tolerance="aggressive"), own={})["SNDK"])
    r = review(captured("PG_open"), view_=view(risk_tolerance="conservative"), own={})["PG"]
    assert r["risk"]["allowed_level"] == "low" and r["risk"]["risk_level"] == "low" and r["approved"]                # PG_open's 6.5 is low risk: a conservative user may have it


def test_a_review_that_asks_for_a_person_is_denied_with_that_reason_on_the_review_the_recommendation_and_the_card_and_its_score_is_not_capped():
    """Entry 103 (Jacob, 2026-11-15: *"if the Critic wants a person to review, then the Critic needs to Deny the recommendation and return the reason in the WorkOrder and update the recommendation status to Denied"*). The real
    PG_open record reviewed while the symbol directory cannot be read (Finnhub's lookup failed: HTTP 429) has `identity_unverified`, a finding that asks for a person. The review is denied, `needs_human_review` is its
    reason, the score is what the six dimensions gave (88, not the blockers' cap of 49), and the same denial is on the work order's review, in the `critic` block of the recommendation (the status), in the summary and on
    the user's card."""
    from prometheus.agents.critic.report import scorecard

    order = captured("PG_open")
    r = review(order, listed={"PG": None})["PG"]
    assert (r["approved"], r["score"], r["score_before_blockers"], r["grade"], r["denial_reasons"], r["requires_human_reasons"]) == (False, 93, 93, "A", ["needs_human_review"], ["identity_unverified"])      # Critic v3: 88 B under the old weights
    assert r["blockers"] == [] and r["valid_until_utc"] is None and order.reviews[0]["review_id"] == r["review_id"]
    assert r["summary"] == "DENY PG: a person must look before this can trade: identity_unverified (PG could not be looked up in the symbol directory (the lookup failed (HTTP 429)))."
    block = next(x for x in order.recommendations if x["symbol"] == "PG")["critic"]                  # the recommendation's status
    assert (block["verdict"]["approved"], block["verdict"]["decision"], block["verdict"]["summary"]) == (False, "deny", r["summary"])
    assert block["gate"]["denial_reasons"] == ["needs_human_review"] and block["gate"]["blocked"] is False and block["needs_human"] == {"required": True, "reasons": ["identity_unverified"]}
    card = scorecard(r, next(x for x in order.recommendations if x["symbol"] == "PG"))["text"]
    assert "DENY " in card and "DENIED needs_human_review: a person must look before this can trade: identity_unverified" in card and "needs your eyes" not in card


def test_a_company_with_no_six_month_average_is_not_denied_the_value_note_asks_for_nobody():
    """Part F (Jacob, 2026-11-15: *"remove this rule about a company less than 126 bars. This does not matter because it eliminates IPOs from being traded and is a bad rule"*). The real PG_open review with no six-month average
    has the plain warning `value_not_assessable`, the value part counts the unknown credit, and the review is approved at 89: nobody is asked for."""
    r = review(captured("PG_open"), own={})["PG"]
    note = next(w for w in r["warnings"] if w["code"] == "value_not_assessable")
    assert note["requires_human"] is False and r["approved"] and r["denial_reasons"] == [] and r["value"]["deal"] == "unknown" and r["dimensions"]["value"]["score"] > 0
    assert not any(m["available"] for m in r["dimensions"]["value"]["measures"] if m["name"] == "discount_to_avg_6m")      # the measure that needs the average could not be made: it counted the unknown credit, as it always did
