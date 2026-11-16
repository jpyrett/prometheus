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


def measure(name):
    return next(m for m in R.measures if m.name == name)


def test_a_measure_that_could_not_be_made_counts_the_unknown_points_and_says_so():
    assert rk.measure_points(measure("vol_63d_ann"), None, 0.5) == (0.5, False)
    r, _ = case(**{"identity.regime": "unknown"})
    order = captured("PG_open")
    fetched = FakeClerk(order).metrics["PG"]
    del fetched["vol_63d_ann"]
    r = rk.assess(rec_of(order), view(), fetched, 167.1, R)
    vol = next(m for m in r.measures if m.name == "vol_63d_ann")
    assert vol.available is False and vol.value == 0.5 and "unknown" in vol.note


def test_the_risk_score_has_five_measures_that_add_up_to_100_and_a_wide_stop_changes_nothing():
    """Design choices entry 92 (Jacob, 2026-11-15): the stop's distance (`stop_loss_pct`) and the stock's own trading volume are not risk measures. The five that remain (no company size either, entry 97) add up to 100."""
    assert [m.name for m in R.measures] == ["vol_63d_ann", "max_drawdown_63d", "beta_63d", "atr_pct", "flags"] and sum(m.weight for m in R.measures) == pytest.approx(100.0)
    order = captured("PG_open")
    fetched = FakeClerk(order).metrics["PG"]
    base = rk.assess(rec_of(order), view(), fetched, 167.1, R)
    wide = rec_of(order).model_copy(deep=True)
    wide.trade.stop_distance = wide.trade.entry_mid * 0.5
    assert rk.assess(wide, view(), fetched, 167.1, R).detail.risk_score == base.detail.risk_score == 6.5


def test_the_flags_add_to_the_score():
    r, _ = case(**{"identity.regime": "stress", "ranking.conflict": True, "trade.gap_risk": True, "quality.multiplier": 0.5})
    assert r.detail.flags == ["market regime stress", "strategies disagree", "a report inside the horizon", "weak quality (0.50)"]
    assert next(m for m in r.measures if m.name == "flags").value == pytest.approx(0.8) and r.detail.risk_score == pytest.approx(6.5 + 0.8 * 7.8, abs=0.1)      # the base 6.5, and four flags of five: 0.8 of the flags' weight of 7.8


def test_a_stockholders_deficit_is_a_flag_from_the_critics_own_numbers():
    order = captured("PG_open")
    fetched = FakeClerk(order).metrics["PG"]
    fetched["negative_equity"] = {**fetched.get("negative_equity", {"unit": "bool", "as_of": "2026-11-05", "source": "x", "window": "", "inputs_summary": {}, "note": None}), "value": True}
    assert "stockholders' deficit" in rk.assess(rec_of(order), view(), fetched, 167.1, R).detail.flags


# ---------------------------------------------------------------- no size, and no knowledge of the user's money


# ---------------------------------------------------------------- the profile's own lists

def test_an_excluded_symbol_sector_or_price_is_a_blocker_each():
    assert "symbol_excluded" in codes(case(v=view(excluded_symbols=["PG"]))[0], "blocker")
    assert "sector_excluded" in codes(case(v=view(excluded_sectors=["XLP"]), **{"identity.sector_etf": "XLP"})[0], "blocker")
    assert "sector_excluded" not in codes(case(v=view(excluded_sectors=["XLP"]), **{"identity.sector_etf": None})[0])    # no sector known: nothing to exclude it by
    assert "sector_excluded" not in codes(case(v=view(excluded_sectors=["XLE"]), **{"identity.sector_etf": "XLP"})[0])
    r, _ = case(v=view(min_price=200.0))
    assert "price_below_profile_min" in codes(r, "blocker") and r.score == 0


# ---------------------------------------------------------------- a sale is not judged

def test_a_reduction_or_a_holding_is_not_a_record_so_the_risk_of_one_is_never_judged():
    """AMENDED (entry 105): a `reduce` with 12 shares held was "not judged" for risk and proposed no size. The record models refuse such a record (the action is not `open`, and a record that says something is held is
    refused too), and the risk step has no skip for it."""
    with pytest.raises(ValidationError, match="the action is 'reduce': a recommendation is to buy"):
        case(**{"action.action": "reduce", "action.held_shares": 12, "trade.family": None, "trade.stop_price": None})
    with pytest.raises(ValidationError, match="the record says something is held"):
        case(**{"action.held_shares": 12})
    r, _ = case()
    assert r.applicable is True and r.detail.risk_score is not None and not hasattr(r, "approved_shares")
