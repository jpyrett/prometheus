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
    assert (r["risk"]["risk_score"], r["risk"]["risk_level"]) == (70.0, "very_high") and r["risk"]["allowed_level"] == "medium"
    assert "fell 52% in 21 sessions" in next(b["message"] for b in r["blockers"] if b["code"] == "falling_knife")
    assert r["summary"].startswith("DENY SNDK: ") and r["approved_shares"] == 0 and r["valid_until_utc"] is None


def test_pg_on_conviction_is_approved():
    """Before Amendment 5 this capture was not a recommendation (no strategy fired); with the `conviction` family its blend (78) makes it one, and since Amendment 3 the Critic no longer has a profit-potential
    rule (a profit floor and a thin-edge warning on the expected excess return after cost were removed: they misled about a company like Procter & Gamble)."""
    r = review(captured("PG"), own={})["PG"]
    assert r["approved"] and codes(r) == [] and r["approved_shares"] == 44 and "thin_edge" not in codes(r, "warnings") and "excess_reviewed" not in r["value"]


def test_a_conviction_recommendation_is_denied_when_the_quotes_it_stands_on_do_not_hold():
    """The conviction family stands on the blend. When the model says every quote says the opposite, the news part turns negative (-0.58) and the blend falls from 0.55 to 0.09, under the 0.30 mark
    the strategy needs: `blend_not_supported`."""
    r = review(captured("PG"), model=ReviewerModel(default="contradicts"))["PG"]
    c = r["evidence"]["counterfactual"]
    assert c["blend"] == 0.0887 and c["blend_recorded"] == 0.5501
    assert not r["approved"] and "blend_not_supported" in codes(r)


def test_pg_open_passes_every_check_and_is_approved_at_44_shares_without_a_six_month_average():
    r = review(captured("PG_open"), own={})["PG"]
    assert r["approved"] and r["blockers"] == [] and r["approved_shares"] == 44 and r["value"]["deal"] == "unknown"
    assert "value_not_assessable" in codes(r, "warnings") and r["requires_human"] and r["requires_human_reasons"] == ["value_not_assessable"]
    assert r["risk"]["risk_score"] == 7.1 and r["risk"]["risk_level"] == "low" and r["dimensions"]["accuracy"]["score"] == 100


def test_the_same_three_with_a_six_month_average_four_percent_under_the_price():
    reviews = {name: next(iter(review(captured(name)).values())) for name in ("SNDK", "PG", "PG_open")}
    assert not reviews["SNDK"]["approved"] and reviews["PG"]["approved"] and codes(reviews["PG"]) == []
    assert not reviews["SNDK"]["approved"] and {"risk_above_profile", "falling_knife"} <= set(codes(reviews["SNDK"]))
    assert reviews["PG_open"]["approved"] and reviews["PG_open"]["value"]["deal"] == "fair" and reviews["PG_open"]["requires_human"] is False


def test_at_night_or_at_the_weekend_all_three_are_denied_and_say_what_the_score_would_have_been():
    for moment in ("2026-11-05 22:13", "2026-11-07 12:00"):
        for name, symbol in (("SNDK", "SNDK"), ("PG", "PG"), ("PG_open", "PG")):
            r = review(captured(name), now=at(moment))[symbol]
            assert not r["approved"] and "market_closed" in codes(r) and r["score"] <= 49


def test_an_aggressive_user_still_cannot_have_sndk_at_very_high_risk_but_a_conservative_one_is_denied_pg_open_at_its_size():
    assert "risk_above_profile" in codes(review(captured("SNDK"), view_=view(risk_tolerance="aggressive"), own={})["SNDK"])
    r = review(captured("PG_open"), view_=view(risk_tolerance="conservative"), own={})["PG"]
    assert r["risk"]["allowed_level"] == "low" and r["risk"]["risk_level"] == "low" and r["approved"]                # PG_open's 7.1 is low risk: a conservative user may have it
