"""Risk fit: does the COMPANY'S risk match the user's profile? (Critic spec 8.4 and Appendix B.2, worked from the real captures.)

Design choices entry 80: the Critic sizes nothing and does not know the user's money, so there is no position measure, no cap and no number of shares here: how much to buy is the Account Manager's, and the
arithmetic of each cap is tested in tests/test_account_manager_sizing.py. The numbers below are worked by hand from the real captures' metrics and the risk measures' weights of configs/critic_v1.json.
"""
import math

import pytest
from critic_helpers import CONFIG, FakeClerk, captured, rec_of, tamper, view
from pydantic import ValidationError

from prometheus.agents.critic import risk as rk

R = CONFIG.risk


def case(name="PG_open", price=None, v=None, **changes):
    order = captured(name)
    if changes:
        order.recommendations[0] = tamper(order.recommendations[0], **changes)
    rec = rec_of(order)
    fetched = FakeClerk(order).metrics[rec.symbol]
    price = price if price is not None else rec.trade.entry_mid
    return rk.assess(rec, v or view(), fetched, price, R), rec


def codes(result, severity=None):
    return [f.code for f in result.findings if severity in (None, f.severity)]


# ---------------------------------------------------------------- Appendix B.2

def test_pg_scores_5_2_and_is_low_risk_by_hand():
    """PG's real numbers: vol_63d_ann 0.20697, atr_pct 0.019679, stop 9.8649 on 167.10 (5.9%), the rest of the measures are at their safe side. Each measure earns clip((x - safe) / (risky - safe)) of its weight."""
    r, rec = case()
    vol, atr, stop = (0.20697 - 0.15) / (0.60 - 0.15) * 27.8, (0.019679 - 0.015) / (0.06 - 0.015) * 11.1, (9.8649 / 167.1 - 0.05) / (0.25 - 0.05) * 11.1
    assert (vol, atr, stop) == pytest.approx((3.5195, 1.1541, 0.5015), abs=0.001)
    by = {m.name: m.value * m.weight for m in r.measures}
    assert by["vol_63d_ann"] == pytest.approx(vol, abs=0.01) and by["atr_pct"] == pytest.approx(atr, abs=0.01) and by["stop_loss_pct"] == pytest.approx(stop, abs=0.01)
    assert by["max_drawdown_63d"] == by["beta_63d"] == by["market_cap"] == by["adv_20d_dollars"] == by["flags"] == 0
    assert (r.detail.risk_score, r.detail.risk_level, r.detail.allowed_level, r.detail.tolerance_effective) == (round(vol + atr + stop, 1), "low", "medium", "moderate") and r.detail.risk_score == 5.2
    assert r.score == pytest.approx(round(100 * (1 - 0.5 * 5.2 / 45), 2)) == pytest.approx(94.22) and codes(r) == []
    assert "position_pct" not in {m.name for m in r.measures}


def test_sndk_is_very_high_risk_and_denied_for_a_moderate_profile():
    r, _ = case("SNDK")
    assert (r.detail.risk_score, r.detail.risk_level) == (77.8, "very_high") and codes(r, "blocker") == ["risk_above_profile"] and r.score == 0       # 27.8 + 16.7 + 11.1 + 11.1 + 11.1: five measures at their risky end
    assert {m.name: m.value for m in r.measures if m.weight >= 10 and m.name not in ("market_cap", "adv_20d_dollars")} == {"vol_63d_ann": 1.0, "max_drawdown_63d": 1.0, "beta_63d": 1.0, "atr_pct": 1.0, "stop_loss_pct": 1.0}


def test_very_high_risk_is_allowed_to_nobody_not_even_an_aggressive_user():
    r, _ = case("SNDK", v=view(risk_tolerance="aggressive"))
    assert codes(r, "blocker") == ["risk_above_profile"] and r.detail.allowed_level == "high"


# ---------------------------------------------------------------- tolerance

@pytest.mark.parametrize("stated, age, horizon, effective, why", [
    ("moderate", None, None, "moderate", None), ("aggressive", None, None, "aggressive", None),
    ("aggressive", 59, None, "aggressive", None), ("aggressive", 60, None, "moderate", "age 60"), ("aggressive", 70, None, "conservative", "age 70"), ("moderate", 70, None, "conservative", "age 70"),
    ("conservative", 30, None, "conservative", None), ("moderate", 65, None, "moderate", None),
    ("aggressive", None, 3, "aggressive", None), ("aggressive", None, 2.9, "moderate", "a 2.9-year horizon"), ("aggressive", None, 0.5, "conservative", "a 0.5-year horizon"),
    ("aggressive", 62, 0.5, "conservative", "a 0.5-year horizon"), ("aggressive", 72, 2, "conservative", "age 72")])
def test_the_effective_tolerance_is_the_lowest_of_the_stated_one_and_the_age_and_horizon_caps(stated, age, horizon, effective, why):
    assert rk.effective_tolerance(view(risk_tolerance=stated, age=age, investment_horizon_years=horizon), R) == (effective, why)


def test_a_lowered_tolerance_is_noted_and_changes_the_limit():
    r, _ = case(v=view(risk_tolerance="aggressive", age=72))
    note = next(f for f in r.findings if f.code == "tolerance_capped")
    assert note.severity == "info" and "conservative" in note.message and r.detail.tolerance_stated == "aggressive" and r.detail.allowed_level == "low"


# ---------------------------------------------------------------- levels and measures

@pytest.mark.parametrize("score, level", [(0, "low"), (19.9, "low"), (20, "medium"), (44.9, "medium"), (45, "high"), (69.9, "high"), (70, "very_high"), (100, "very_high")])
def test_the_levels_at_their_bounds(score, level):
    assert rk.level_of(score, R) == level


def measure(name):
    return next(m for m in R.measures if m.name == name)


@pytest.mark.parametrize("name, x, points", [
    ("vol_63d_ann", 0.15, 0.0), ("vol_63d_ann", 0.60, 1.0), ("vol_63d_ann", 0.375, 0.5), ("vol_63d_ann", 0.05, 0.0), ("vol_63d_ann", 2.0, 1.0),
    ("max_drawdown_63d", -0.10, 0.0), ("max_drawdown_63d", -0.40, 1.0), ("max_drawdown_63d", -0.25, 0.5),
    ("beta_63d", 0.8, 0.0), ("beta_63d", -2.0, 1.0), ("beta_63d", -0.1, 0.0), ("beta_63d", 1.4, 0.5),
    ("market_cap", 50e9, 0.0), ("market_cap", 2e9, 1.0), ("market_cap", math.sqrt(50e9 * 2e9), 0.5), ("market_cap", 1e12, 0.0),
    ("adv_20d_dollars", 100e6, 0.0), ("adv_20d_dollars", 5e6, 1.0), ("flags", 0.4, 0.4), ("flags", 5.0, 1.0)])
def test_each_measure_at_safe_at_risky_and_between(name, x, points):
    assert rk.measure_points(measure(name), x, R.unknown_points) == (pytest.approx(points), True)


def test_a_measure_that_could_not_be_made_counts_the_unknown_points_and_says_so():
    assert rk.measure_points(measure("vol_63d_ann"), None, 0.5) == (0.5, False)
    assert rk.measure_points(measure("market_cap"), 0.0, 0.5) == (0.5, False) and rk.measure_points(measure("market_cap"), -1.0, 0.5) == (0.5, False)
    r, _ = case(**{"identity.regime": "unknown"})
    order = captured("PG_open")
    fetched = FakeClerk(order).metrics["PG"]
    del fetched["vol_63d_ann"]
    r = rk.assess(rec_of(order), view(), fetched, 167.1, R)
    vol = next(m for m in r.measures if m.name == "vol_63d_ann")
    assert vol.available is False and vol.value == 0.5 and "unknown" in vol.note


def test_the_flags_add_to_the_score():
    r, _ = case(**{"identity.regime": "stress", "ranking.conflict": True, "trade.gap_risk": True, "quality.multiplier": 0.5})
    assert r.detail.flags == ["market regime stress", "strategies disagree", "a report inside the horizon", "weak quality (0.50)"]
    assert next(m for m in r.measures if m.name == "flags").value == pytest.approx(0.8) and r.detail.risk_score == pytest.approx(5.2 + 0.8 * 5.5, abs=0.1)      # the base 5.2, and four flags of five: 0.8 of the flags' weight of 5.5


def test_a_stockholders_deficit_is_a_flag_from_the_critics_own_numbers():
    order = captured("PG_open")
    fetched = FakeClerk(order).metrics["PG"]
    fetched["negative_equity"] = {**fetched.get("negative_equity", {"unit": "bool", "as_of": "2026-11-05", "source": "x", "window": "", "inputs_summary": {}, "note": None}), "value": True}
    assert "stockholders' deficit" in rk.assess(rec_of(order), view(), fetched, 167.1, R).detail.flags


# ---------------------------------------------------------------- no size, and no knowledge of the user's money

def test_the_assessment_holds_no_size_and_the_risk_detail_has_no_field_for_one():
    r, _ = case()
    assert set(rk.RiskDetail.model_fields) == {"tolerance_stated", "tolerance_effective", "allowed_level", "risk_score", "risk_level", "flags"}
    assert not hasattr(r, "approved_shares") and not hasattr(r.detail, "caps") and not hasattr(r.detail, "allowed_shares") and not hasattr(rk, "size_caps") and not hasattr(rk, "exposure_cap")
    assert not {"size_above_profile", "exposure_limit", "size_reduced"} & set(codes(r))


def test_the_critic_has_no_advisor_size_to_read_so_nothing_about_shares_changes_the_risk_assessment():
    """Entry 80: the Advisor's record has no size any more (before it, this test put a wild `suggested_shares` in and showed the assessment did not move). What is left of the claim: the shares the user
    holds change nothing, and a record that carries a size is not a record at all (the schema forbids the field), so a Critic could not read one."""
    plain, _ = case()
    held, _ = case(**{"action.held_shares": 7})
    assert held.detail == plain.detail and held.score == plain.score and [(m.name, m.value) for m in held.measures] == [(m.name, m.value) for m in plain.measures] and codes(held) == codes(plain) == []
    with pytest.raises(ValidationError):
        case(**{"trade.suggested_shares": 100000})


def test_a_size_limit_of_the_users_profile_cannot_change_a_review_because_the_critic_is_never_given_it():
    """The profile still carries the four size limits (the Account Manager's); `risk_view()` leaves them out, so the most restrictive profile gives the Critic exactly what the default one does."""
    plain, _ = case()
    strict, _ = case(v=view(max_position_pct=0.001, max_loss_per_trade_pct=0.0001, max_new_exposure_pct=0.001, max_order_dollars=1.0))
    assert strict.detail == plain.detail and strict.score == plain.score and codes(strict) == [] and codes(strict, "blocker") == []


# ---------------------------------------------------------------- the profile's own lists

def test_an_excluded_symbol_sector_or_price_is_a_blocker_each():
    assert "symbol_excluded" in codes(case(v=view(excluded_symbols=["PG"]))[0], "blocker")
    assert "sector_excluded" in codes(case(v=view(excluded_sectors=["XLP"]), **{"identity.sector_etf": "XLP"})[0], "blocker")
    assert "sector_excluded" not in codes(case(v=view(excluded_sectors=["XLP"]), **{"identity.sector_etf": None})[0])    # no sector known: nothing to exclude it by
    assert "sector_excluded" not in codes(case(v=view(excluded_sectors=["XLE"]), **{"identity.sector_etf": "XLP"})[0])
    r, _ = case(v=view(min_price=200.0))
    assert "price_below_profile_min" in codes(r, "blocker") and r.score == 0


# ---------------------------------------------------------------- a sale is not judged

def test_a_reduction_is_not_judged_and_proposes_no_size():
    r, rec = case(**{"action.action": "reduce", "action.held_shares": 12, "trade.family": None, "trade.stop_price": None})
    assert r.applicable is False and r.findings == [] and r.detail.risk_score is None and not hasattr(r, "approved_shares")
