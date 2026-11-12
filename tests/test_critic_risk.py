"""Risk fit: does the risk match the user's profile? (Critic spec 8.4 and Appendix B.2, worked from the real captures.)"""
import math

import pytest
from critic_helpers import CONFIG, FakeClerk, captured, rec_of, tamper, view

from prometheus.agents.critic import risk as rk

R = CONFIG.risk
EQUITY = 74481.40       # cash and the holdings of configs/portfolios/jacob_pyrett.json at the capture's prices


def case(name="PG_open", price=None, equity=EQUITY, v=None, **changes):
    order = captured(name)
    if changes:
        order.recommendations[0] = tamper(order.recommendations[0], **changes)
    rec = rec_of(order)
    fetched = FakeClerk(order).metrics[rec.symbol]
    price = price if price is not None else rec.trade.entry_mid
    return rk.assess(rec, v or view(), fetched, price, equity, "test equity", R), rec


def codes(result, severity=None):
    return [f.code for f in result.findings if severity in (None, f.severity)]


# ---------------------------------------------------------------- Appendix B.2

def test_pg_at_the_moderate_cap_scores_7_1_and_is_low_risk():
    r, rec = case()
    assert r.approved_shares == 44 and r.detail.caps["position"] == 44 and r.detail.caps["loss"] == 75 and r.detail.allowed_shares == 44
    assert (r.detail.risk_score, r.detail.risk_level, r.detail.allowed_level, r.detail.tolerance_effective) == (7.1, "low", "medium", "moderate")
    by = {m.name: m.value * m.weight for m in r.measures}
    assert by["vol_63d_ann"] == pytest.approx(3.16, abs=0.01) and by["atr_pct"] == pytest.approx(1.04, abs=0.01) and by["stop_loss_pct"] == pytest.approx(0.45, abs=0.01) and by["position_pct"] == pytest.approx(2.44, abs=0.01)
    assert by["max_drawdown_63d"] == by["beta_63d"] == by["market_cap"] == by["adv_20d_dollars"] == by["flags"] == 0
    assert r.score == pytest.approx(round(100 * (1 - 0.5 * 7.1 / 45), 2)) and codes(r) == ["size_reduced"]


def test_pg_at_the_advisors_own_size_scores_10_6():
    r, _ = case(v=view(max_position_pct=0.20))
    assert r.approved_shares == 75 and r.detail.risk_score == 10.6 and codes(r) == []


def test_sndk_is_very_high_risk_and_denied_for_a_moderate_profile():
    r, _ = case("SNDK")
    assert (r.detail.risk_score, r.detail.risk_level) == (70.0, "very_high") and codes(r, "blocker") == ["risk_above_profile"] and r.score == 0
    assert {m.name: m.value for m in r.measures if m.weight >= 10 and m.name != "position_pct" and m.name not in ("market_cap", "adv_20d_dollars")} == {"vol_63d_ann": 1.0, "max_drawdown_63d": 1.0, "beta_63d": 1.0, "atr_pct": 1.0, "stop_loss_pct": 1.0}


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
    r = rk.assess(rec_of(order), view(), fetched, 167.1, EQUITY, "x", R)
    vol = next(m for m in r.measures if m.name == "vol_63d_ann")
    assert vol.available is False and vol.value == 0.5 and "unknown" in vol.note


def test_the_flags_add_to_the_score():
    r, _ = case(**{"identity.regime": "stress", "ranking.conflict": True, "trade.gap_risk": True, "quality.multiplier": 0.5})
    assert r.detail.flags == ["market regime stress", "strategies disagree", "a report inside the horizon", "weak quality (0.50)"]
    assert next(m for m in r.measures if m.name == "flags").value == pytest.approx(0.8) and r.detail.risk_score == pytest.approx(7.1 + 4.0)


def test_a_stockholders_deficit_is_a_flag_from_the_critics_own_numbers():
    order = captured("PG_open")
    fetched = FakeClerk(order).metrics["PG"]
    fetched["negative_equity"] = {**fetched.get("negative_equity", {"unit": "bool", "as_of": "2026-11-05", "source": "x", "window": "", "inputs_summary": {}, "note": None}), "value": True}
    assert "stockholders' deficit" in rk.assess(rec_of(order), view(), fetched, 167.1, EQUITY, "x", R).detail.flags


# ---------------------------------------------------------------- the size the profile allows

def test_the_loss_limit_can_be_the_one_that_binds():
    r, rec = case(v=view(max_position_pct=1.0, max_loss_per_trade_pct=0.002))
    assert r.detail.caps["loss"] == math.floor(0.002 * EQUITY / rec.trade.stop_distance) == 15 and r.approved_shares == 15
    assert "loss limit" in r.findings[0].message


def test_an_order_dollar_cap_is_a_third_limit():
    r, _ = case(v=view(max_order_dollars=2000))
    assert r.detail.caps["order"] == math.floor(2000 / 167.1) == 11 and r.approved_shares == 11 and "order limit" in r.findings[0].message


def test_an_addition_counts_what_is_already_held():
    r, rec = case(**{"action.action": "add", "action.held_shares": 30})
    assert r.detail.caps["position"] == math.floor((0.10 * EQUITY - 30 * 167.1) / 167.1) == 14 and r.approved_shares == 14


def test_the_critic_never_raises_a_size_only_cuts_it():
    r, rec = case(v=view(max_position_pct=1.0, max_loss_per_trade_pct=1.0), **{"trade.suggested_shares": 3})
    assert r.approved_shares == 3 and "size_reduced" not in codes(r)


def test_not_even_one_share_fitting_is_a_blocker_that_lists_the_limits():
    r, _ = case(v=view(max_loss_per_trade_pct=0.0001))
    [f] = [f for f in r.findings if f.code == "size_above_profile"]
    assert f.severity == "blocker" and r.approved_shares == 0 and "Not even one share" in f.message and "loss 0" in f.message and r.score == 0


@pytest.mark.parametrize("kwargs, why", [({"price": 0.0}, "there is no price"), ({"equity": None}, "the equity is unknown"), ({"stop_distance_gone": True}, "no stop distance")])
def test_a_position_that_cannot_be_sized_is_not_approved(kwargs, why):
    order = captured("PG_open")
    if kwargs.pop("stop_distance_gone", False):
        order.recommendations[0] = tamper(order.recommendations[0], **{"trade.stop_distance": None})
    rec = rec_of(order)
    price = kwargs.pop("price", 167.1) or None
    r = rk.assess(rec, view(), FakeClerk(order).metrics["PG"], price, kwargs.get("equity", EQUITY), "x", R)
    assert "size_above_profile" in codes(r, "blocker") and r.approved_shares == 0 and why in r.findings[-1].message if "stop" not in why else "size_above_profile" in codes(r, "blocker")


def test_the_exposure_cap_is_what_still_fits_under_the_reviews_limit():
    assert rk.exposure_cap(0.0, 100.0, 10000.0, 0.30) == 30 and rk.exposure_cap(2500.0, 100.0, 10000.0, 0.30) == 5 and rk.exposure_cap(3000.0, 100.0, 10000.0, 0.30) == 0
    assert rk.exposure_cap(9999.0, 100.0, 10000.0, 0.30) == 0 and rk.exposure_cap(0.0, None, 10000.0, 0.3) is None and rk.exposure_cap(0.0, 100.0, None, 0.3) is None


# ---------------------------------------------------------------- the profile's own lists

def test_an_excluded_symbol_sector_or_price_is_a_blocker_each():
    assert "symbol_excluded" in codes(case(v=view(excluded_symbols=["PG"]))[0], "blocker")
    assert "sector_excluded" in codes(case(v=view(excluded_sectors=["XLP"]), **{"identity.sector_etf": "XLP"})[0], "blocker")
    assert "sector_excluded" not in codes(case(v=view(excluded_sectors=["XLP"]), **{"identity.sector_etf": None})[0])    # no sector known: nothing to exclude it by
    assert "sector_excluded" not in codes(case(v=view(excluded_sectors=["XLE"]), **{"identity.sector_etf": "XLP"})[0])
    r, _ = case(v=view(min_price=200.0))
    assert "price_below_profile_min" in codes(r, "blocker") and r.score == 0


# ---------------------------------------------------------------- a sale is not judged

def test_a_reduction_is_not_judged_and_proposes_the_whole_holding_when_the_advisor_sized_none():
    r, rec = case(**{"action.action": "reduce", "action.held_shares": 12, "trade.suggested_shares": 0, "trade.family": None, "trade.stop_price": None})
    assert r.applicable is False and r.approved_shares == 12 and r.findings == [] and r.detail.risk_score is None
    r, _ = case(**{"action.action": "reduce", "action.held_shares": 12, "trade.suggested_shares": 5, "trade.family": None})
    assert r.approved_shares == 5
