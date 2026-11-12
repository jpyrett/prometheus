"""Tests for prometheus.agents.advisor.trade_plan (Advisor v1 spec section 10, phase 5). The prices are literals that make the arithmetic checkable by hand; the portfolio is the real
made-up one of the SNDK/MU test (configs/portfolios/jacob_pyrett_sndk_mu.json: PG 15 @ 60, MCD 120 @ 40, GOOGL 30 @ 100, LLY 25 @ 155, $200 cash)."""
from datetime import date

import pytest
from advisor_helpers import CONFIG, TODAY, fr, healthy, metric

from prometheus.agents.advisor import trade_plan as tp
from prometheus.common.portfolio import Portfolio, load_portfolio
from prometheus.rag.config import REPO_ROOT

PORTFOLIO = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett_sndk_mu.json")
EMPTY = Portfolio(user="t", cash_avail=1000.0)
BIG = (100_000.0, "last_close", {})


def metrics(**kw):
    return healthy(**{"atr_14": 4.0, "last_close": 100.0, "adv_20d_dollars": 10e6, **kw})


def plan(family="capital_return", m=None, equity=(1000.0, "last_close", {}), portfolio=EMPTY, regime="trend", edge=0.004, horizon=126, quote=None, **kw):
    return tp.build(fr(family, edge=edge, horizon_bars=horizon), m if m is not None else metrics(), quote, regime, equity, portfolio, TODAY, CONFIG, symbol="AAA", **kw)


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


def test_a_thousand_dollar_account_gets_one_share_and_the_plan_says_it_is_oversized():
    p = plan("capital_return")                                                                   # risk budget $10, stop distance 12: floor(10 / 12) = 0 shares
    assert p.risk_budget == 10.0 and p.suggested_shares == 1 and p.oversized is True and p.position_pct == pytest.approx(0.10)
    assert p.size_note == "one share is 10% of equity; the 1.0% risk budget allows 0"


def test_the_size_is_the_risk_budget_over_the_stop_distance_capped_by_equity_and_liquidity():
    p = plan("capital_return", equity=BIG, portfolio=Portfolio(user="t", cash_avail=100_000.0))
    assert p.risk_budget == 1000.0 and p.suggested_shares == 83 and p.oversized is False                         # floor(1000 / 12) = 83
    thin = tp.build(fr("capital_return", horizon_bars=126), metrics(adv_20d_dollars=100_000.0), None, "trend", BIG, Portfolio(user="t", cash_avail=100_000.0), TODAY, CONFIG, symbol="AAA")
    assert thin.suggested_shares == 50                                                                           # 5% of $100,000 ADV / $100 = 50
    cheap = tp.build(fr("reversion", horizon_bars=10), metrics(atr_14=0.01, adv_20d_dollars=1e9), None, "trend", BIG, Portfolio(user="t", cash_avail=100_000.0), TODAY, CONFIG, symbol="AAA")
    assert cheap.suggested_shares == 200                                                                         # 20% of equity / $100: the risk budget alone would allow 50,000


def test_a_stress_or_unknown_regime_halves_the_risk_budget():
    for regime in ("stress", "unknown"):
        p = plan("capital_return", equity=BIG, portfolio=Portfolio(user="t", cash_avail=100_000.0), regime=regime)
        assert p.risk_budget == 500.0 and p.suggested_shares == 41
    assert plan("capital_return", equity=BIG, portfolio=Portfolio(user="t", cash_avail=100_000.0), regime="narrow").risk_budget == 1000.0


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
    assert p.entry_zone is None and p.stop_price is None and p.suggested_shares == 0 and "atr_14 is missing" in p.notes[0]
    assert "there is no price" in plan("trend", m=healthy(atr_14=4.0, last_close=None), horizon=63).notes[0]


def test_unknown_equity_means_no_size_not_a_guess():
    p = plan("capital_return", equity=(None, "no price for PG", {}))
    assert p.suggested_shares == 0 and p.stop_price == 88.0 and any("equity is unknown (no price for PG)" in n for n in p.notes)


def test_a_reduction_has_an_entry_zone_and_no_size():
    p = tp.build(None, metrics(), None, "trend", BIG, EMPTY, TODAY, CONFIG, symbol="AAA")
    assert p.entry_zone == [98.0, 102.0] and p.family is None and p.suggested_shares == 0 and p.stop_price is None and p.invalidation == []


def test_equity_is_cash_plus_the_holdings_at_their_price_and_never_a_guess():
    by_symbol = {"PG": healthy(last_close=150.0), "MCD": healthy(last_close=300.0), "GOOGL": healthy(last_close=200.0), "LLY": healthy(last_close=900.0)}
    equity, basis, prices = tp.equity_of(PORTFOLIO, by_symbol, {})
    assert equity == pytest.approx(200 + 15 * 150 + 120 * 300 + 30 * 200 + 25 * 900) and basis == "last_close" and prices["LLY"] == 900.0
    equity, basis, _ = tp.equity_of(PORTFOLIO, by_symbol, {"PG": {"price": 160.0}})
    assert equity == pytest.approx(200 + 15 * 160 + 120 * 300 + 30 * 200 + 25 * 900) and basis == "last_close, quote"
    without = dict(by_symbol)
    del without["MCD"]
    assert tp.equity_of(PORTFOLIO, without, {})[:2] == (None, "no price for MCD")
    assert tp.equity_of(EMPTY, {}, {}) == (1000.0, "cash only", {})


def test_funds_feasible_is_cash_or_what_the_sellable_holdings_in_profit_could_raise():
    prices = {"PG": 150.0, "MCD": 300.0, "GOOGL": 200.0, "LLY": 900.0}
    assert tp.funds_check(150.0, PORTFOLIO, prices, TODAY) == (True, "cash $200.00 covers $150.00")
    ok, note = tp.funds_check(900.0, PORTFOLIO, prices, TODAY)               # short by $700; four holdings in profit, one share of each kept: far more than that
    assert ok is True and note.startswith("short by $700.00; $") and "could be raised" in note
    loss = {"PG": 50.0, "MCD": 30.0, "GOOGL": 90.0, "LLY": 100.0}             # every holding is below its cost basis: nothing may be sold
    ok, note = tp.funds_check(900.0, PORTFOLIO, loss, TODAY)
    assert ok is False and "$0.00 could be raised" in note
    ok, _ = tp.funds_check(10**9, PORTFOLIO, prices, TODAY)
    assert ok is False
    today_buy = PORTFOLIO.model_copy(update={"positions": [p.model_copy(update={"acquired": TODAY}) for p in PORTFOLIO.positions]})
    assert tp.funds_check(900.0, today_buy, prices, TODAY)[0] is False        # bought today: locked until tomorrow (T+1)
