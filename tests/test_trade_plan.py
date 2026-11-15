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
    return healthy(**{"atr_14": 4.0, "last_close": 100.0, "adv_20d_dollars": 10e6, **kw})


def plan(family="capital_return", m=None, edge=0.004, horizon=126, quote=None, **kw):
    return tp.build(fr(family, edge=edge, horizon_bars=horizon), m if m is not None else metrics(), quote, TODAY, CONFIG, symbol="AAA", **kw)


def test_the_entry_zone_and_the_stop_follow_the_atr_by_family():
    p = plan("reversion", horizon=10, edge=0.01)
    assert p.entry_mid == 100.0 and p.entry_zone == [98.0, 102.0] and p.entry_basis == "last_close"
    assert p.stop_distance == 8.0 and p.stop_price == 92.0                                       # 2 ATR for reversion and event
    assert plan("event", horizon=40).stop_price == 92.0
    assert plan("trend", horizon=63).stop_distance == 12.0 and plan("capital_return").stop_price == 88.0     # 3 ATR for trend and capital_return


def test_the_stop_is_the_50_day_mean_when_it_is_nearer_and_below_for_trend_and_capital_return():
    p = plan("trend", m=metrics(sma_50=95.0), horizon=63)
    assert p.stop_distance == 5.0 and p.stop_price == 95.0 and any("50-day mean 95.00" in n for n in p.notes)
    assert plan("trend", m=metrics(sma_50=80.0), horizon=63).stop_distance == 12.0                # farther than 3 ATR: the ATR stop
    assert plan("trend", m=metrics(sma_50=105.0), horizon=63).stop_distance == 12.0               # above the price: not a stop
    assert plan("reversion", m=metrics(sma_50=95.0), horizon=10).stop_distance == 8.0              # the 50-day mean is only for trend and capital_return


def test_the_target_is_one_and_a_half_times_the_edge_at_the_horizon_and_there_is_none_for_capital_return():
    assert plan("reversion", horizon=10, edge=0.01).target_price == pytest.approx(100 * (1 + 1.5 * 0.01))            # a short horizon: the 20-day edge as it is
    assert plan("event", horizon=40, edge=0.008).target_price == pytest.approx(100 * (1 + 1.5 * 0.008 * 2))           # 40 bars: twice the 20-day edge
    assert plan("trend", horizon=63, edge=0.006).target_price == pytest.approx(100 * (1 + 1.5 * 0.006 * 3.15))
    assert plan("capital_return").target_price is None


def test_the_time_stop_counts_weekdays_from_today():
    assert tp.add_weekdays(date(2026, 10, 23), 1).isoformat() == "2026-10-26"                    # Friday + 1 = Monday
    assert tp.add_weekdays(date(2026, 10, 23), 10).isoformat() == "2026-11-06"
    assert plan("reversion", horizon=10).time_stop_date == "2026-11-06" and "holidays" in plan("reversion", horizon=10).time_stop_note


def test_a_quote_is_the_entry_price_when_the_order_has_one():
    p = plan("reversion", horizon=10, quote={"price": 105.0})
    assert p.entry_basis == "quote" and p.entry_mid == 105.0 and p.entry_zone == [103.0, 107.0]


def test_the_order_style_by_family_trend_regime_and_spread():
    assert plan("reversion", horizon=10).order_style == "limit" and plan("reversion", horizon=10).limit_price == 100.0
    assert plan("capital_return").order_style == "limit"
    above = plan("trend", m=metrics(above_sma_200=True), horizon=63)
    assert above.order_style == "marketable_limit" and above.limit_price == 101.0                  # last + 0.25 ATR
    assert plan("trend", m=metrics(above_sma_200=False), horizon=63).order_style == "limit"
    wide = metrics(above_sma_200=True)
    wide["median_spread_bps_1d"] = metric(10.5)
    assert plan("trend", m=wide, horizon=63).order_style == "limit"                                  # a spread over 10 bps: always a limit
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


def test_a_plan_has_no_size_and_build_is_given_no_money():
    """Entry 80: the plan is prices and dates. No field of it is a number of shares, a risk budget or a share of the account, and `build` takes neither the portfolio nor the equity."""
    p = plan("capital_return")
    assert p.stop_price == 88.0 and p.stop_distance == 12.0 and p.entry_zone == [98.0, 102.0]                      # the same numbers with or without anyone's money: there is none to give
    assert set(TradePlan.model_fields) == {"family", "entry_basis", "entry_zone", "entry_mid", "stop_distance", "stop_price", "target_price", "horizon_bars", "time_stop_date", "time_stop_note", "gap_risk",
                                           "order_style", "limit_price", "invalidation", "notes"}
    assert list(inspect.signature(tp.build).parameters) == ["family", "metrics", "quote", "today", "config", "symbol", "reaction_atr"]
    assert not hasattr(tp, "equity_of") and not hasattr(tp, "funds_check")                                        # moved to account_manager/sizing.py
    with pytest.raises(ValidationError):
        TradePlan(entry_basis="last_close", suggested_shares=5)                                                   # a size is not a field the record accepts


def test_a_reduction_has_an_entry_zone_and_no_amount():
    p = tp.build(None, metrics(), None, TODAY, CONFIG, symbol="AAA")
    assert p.entry_zone == [98.0, 102.0] and p.family is None and p.stop_price is None and p.invalidation == [] and p.stop_distance is None and p.target_price is None
