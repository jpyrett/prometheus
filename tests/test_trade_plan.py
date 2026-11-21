"""Tests for prometheus.agents.advisor.trade_plan (Advisor v1 spec section 10, phase 5; design choices entry 80). The prices are literals that make the arithmetic checkable by hand.

The plan has NO SIZE since entry 80 (how much to buy or sell is the Account Manager's): the tests that were about the risk budget, the position and liquidity caps, the regime halving the budget,
the account's equity and the funds check were removed with the code they tested. The arithmetic of every cap, of the equity and of the funds check is tested on the Account Manager's code
(tests/test_account_manager_sizing.py); the prices of the plan against the real captures are tested in tests/test_trade_plan_real.py.
"""
import inspect
from datetime import date

import pytest
from advisor_helpers import CONFIG, TODAY, fr, healthy, metric
from pydantic import ValidationError

from prometheus.agents.advisor import trade_plan as tp
from prometheus.agents.advisor.record import TradePlan


def metrics(**kw):
    return healthy(**{"atr_14": 4.0, "last_close": 100.0, **kw})


def plan(family="capital_return", m=None, edge=0.004, horizon=126, quote=None, **kw):
    return tp.build(fr(family, edge=edge, horizon_bars=horizon), m if m is not None else metrics(), quote, TODAY, CONFIG, symbol="AAA", **kw)


def test_a_quote_is_the_entry_price_when_the_order_has_one():
    p = plan("reversion", horizon=10, quote={"price": 105.0})
    assert p.entry_basis == "quote" and p.entry_mid == 105.0 and p.entry_zone == [103.0, 107.0]


def test_the_order_style_by_family_and_trend_regime():
    assert plan("reversion", horizon=10).order_style == "limit" and plan("reversion", horizon=10).limit_price == 100.0
    assert plan("capital_return").order_style == "limit"
    above = plan("trend", m=metrics(above_sma_200=True), horizon=63)
    assert above.order_style == "marketable_limit" and above.limit_price == 101.0                  # last + 0.25 ATR
    assert plan("trend", m=metrics(above_sma_200=False), horizon=63).order_style == "limit"
    assert plan("event", horizon=40, reaction_atr=0.5).order_style == "limit" and any("one ATR or more" in n for n in plan("event", horizon=40, reaction_atr=2.0).notes)


def test_the_invalidation_rules_by_family():
    p = plan("trend", m=metrics(above_sma_200=True), horizon=63)
    kinds = [(r["kind"], r["value"]) for r in p.invalidation]
    assert kinds[0] == ("price_below", p.stop_price) and kinds[1] == ("days_elapsed", 63) and kinds[2][0] == "event_class" and "guidance_cut" in kinds[2][1] and kinds[3] == ("metric_crossed", "below_sma_200")
    assert plan("reversion", horizon=10).invalidation[-1] == {"kind": "metric_crossed", "value": "rsi_14_above_70"}
    assert len(plan("capital_return").invalidation) == 3 and plan("capital_return").invalidation[-1]["kind"] == "event_class"
    assert len(plan("event", horizon=40).invalidation) == 3


def test_a_target_pays_at_least_two_for_one_and_the_edges_own_target_stands_when_it_is_higher():
    """Advisor config version 3, rule E1 (design/advisor_exit_plan_spec.md): a conviction idea at 100 with an ATR of 4 has a stop 12 under the entry; the edge alone (0.6% a month for 40 sessions) put its target at 101.8, a reward of 0.15 for each unit risked. Now it is 124."""
    p = plan("conviction", horizon=40, edge=0.006)
    assert p.stop_distance == 12.0 and p.stop_price == 88.0 and p.target_price == 124.0 and p.trailing is False and p.trail_distance is None
    assert any("2 x the stop distance above the entry (124.00)" in n and "the family's edge alone gave 101.80" in n for n in p.notes)
    for family in ("event", "reversion", "theme"):
        q = plan(family, horizon=40 if family != "reversion" else 10, edge=0.004)
        assert q.target_price is not None and round((q.target_price - q.entry_mid) / q.stop_distance, 6) >= 2.0 and q.trailing is False, family
    big = plan("event", horizon=40, edge=0.5)           # the edge's target (250) is farther than 2 for 1 (116): it stands, and no note says the floor moved it
    assert big.target_price == 250.0 and not any(n.startswith("the target is ") for n in big.notes)


def test_the_trend_family_has_no_target_and_a_trailing_stop_and_nothing_else_trails():
    """Rule E2: the trend plan keeps its stop price as the STARTING stop, records the distance the stop trails the highest close by (the stop distance), and has no target; the time stop and the family's own invalidation stand. capital_return keeps no target and does not trail."""
    t = plan("trend", m=metrics(above_sma_200=True), horizon=63)
    assert t.target_price is None and t.trailing is True and t.trail_distance == t.stop_distance and t.stop_price == round(t.entry_mid - t.stop_distance, 4) and t.time_stop_date is not None
    assert any(n.startswith("trailing stop, no target") for n in t.notes) and [r["kind"] for r in t.invalidation] == ["price_below", "days_elapsed", "event_class", "metric_crossed"]
    for family in ("reversion", "event", "conviction", "theme", "capital_return"):
        q = plan(family, horizon=40)
        assert q.trailing is False and q.trail_distance is None, family
    assert plan("capital_return").target_price is None
    assert plan("trend", m=healthy(atr_14=None, last_close=100.0), horizon=63).trailing is False       # no prices, no plan: nothing trails


def test_a_record_made_before_the_exit_conventions_has_a_fixed_stop_and_loads():
    """A plan record of Advisor config version 2 (no `trailing`, no `trail_distance`) is read as a fixed-stop plan with its target: both new fields have their old-record defaults."""
    old = TradePlan(entry_basis="last_close", family="trend", entry_mid=100.0, stop_distance=12.0, stop_price=88.0, target_price=101.8)
    assert old.trailing is False and old.trail_distance is None and old.target_price == 101.8


def test_no_atr_or_no_price_means_no_plan_and_the_reason():
    p = plan("trend", m=healthy(atr_14=None, last_close=100.0), horizon=63)
    assert p.entry_zone is None and p.stop_price is None and "atr_14 is missing" in p.notes[0]
    assert "there is no price" in plan("trend", m=healthy(atr_14=4.0, last_close=None), horizon=63).notes[0]


