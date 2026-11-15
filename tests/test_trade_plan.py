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


def test_no_atr_or_no_price_means_no_plan_and_the_reason():
    p = plan("trend", m=healthy(atr_14=None, last_close=100.0), horizon=63)
    assert p.entry_zone is None and p.stop_price is None and "atr_14 is missing" in p.notes[0]
    assert "there is no price" in plan("trend", m=healthy(atr_14=4.0, last_close=None), horizon=63).notes[0]


