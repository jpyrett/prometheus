"""The paper broker (prometheus/agents/trader/broker.py, Trader spec sections 6 and 7): placing against one snapshot in paper and backtest mode, slippage, the entry-bar rule on both sides of 09:30, the cash cut, the
refusals, and a funded purchase that would not fill sells nothing. No network, no model: the prices are a dict."""
from datetime import date
from decimal import Decimal

import pytest
from trader_helpers import NOW, account, acct_now, at, bar, config, make_broker, manager, plan, row, synthetic, weekday_bars

from prometheus.agents.trader import tickets as tk
from prometheus.agents.trader.broker import PaperBroker, Prices, Refused, cut_note
from prometheus.common.portfolio import Portfolio


def one_buy(shares=10, price=100.0, **kw):
    return synthetic(row("PG", "open", shares, price, **kw))


# ---------------------------------------------------------------- paper: a sale

def sale(shares=3, price=400.0, **kw):
    return synthetic(row("MSFT", "reduce", shares, price, **kw))


# ---------------------------------------------------------------- sells before buys, and the refusals

def two_trades():
    return synthetic(row("PG", "open", 10, 100.0), row("MSFT", "reduce", 3, 400.0))


@pytest.mark.parametrize("kind, price, message", [
    ("stop", Decimal("0.0"), "order_type must be 'market' or 'limit'"),
    ("MARKET", Decimal("0.0"), "order_type must be 'market' or 'limit'"),
    ("", Decimal("100.0"), 'give an order_type of "limit" with a limit_price'),
    ("market", Decimal("100.0"), "a market order has no limit_price"),
    ("limit", Decimal("98.99"), r"limit_price 98.9900 is outside the range allowed for T1, 99.00 to 101.00"),
    ("limit", Decimal("101.01"), r"limit_price 101.0100 is outside the range allowed for T1, 99.00 to 101.00"),
    ("limit", Decimal("-1.0"), "is outside the range allowed"),
    ("limit", "100", "limit_price must be a number"),
    ("limit", True, "limit_price must be a number"),
    ("limit", None, "limit_price must be a number"),
], ids=str)
def test_an_order_the_model_may_not_give_is_refused_with_the_way_to_fix_it_and_nothing_is_placed(kind, price, message):
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0})
    with pytest.raises(Refused, match=message):
        broker.place("T1", kind, price)
    assert (t.status, t.placed_by, t.placed_utc, broker.placed, acct_now(broker).cash_avail) == ("new", None, None, [], Decimal("1500.0"))


def test_a_limit_order_with_no_price_takes_the_default_limit_and_market_orders_can_be_turned_off():
    broker, (t,) = make_broker(one_buy(limit=100.5), account(1500.0), quotes={"PG": 100.0})
    broker.place("T1", "limit")
    assert (t.order_type, t.limit_price) == ("limit", Decimal("100.5"))
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0})                                          # a plan with no limit price: the default is a market order
    broker.place("T1", "limit")
    assert (t.order_type, t.limit_price) == ("limit", Decimal("100.0"))                                                             # the reference price, inside the range
    off = config(orders={"allow_market": False})
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0}, cfg=off)
    assert (t.order_type, t.limit_price) == ("limit", Decimal("100.0"))
    with pytest.raises(Refused, match="market orders are turned off"):
        broker.place("T1", "market")
    broker.place("T1")
    assert t.status == "filled"


# ---------------------------------------------------------------- close out


def test_the_funding_sales_of_a_purchase_that_was_not_attempted_are_not_attempted_too():
    held = account(10.0, MSFT=(50, 150.0))
    broker, found = make_broker(one_buy(2, 100.0), held, quotes={"PG": 100.0, "MSFT": 420.0}, funding=plan("PG", ("MSFT", 5, 420.0)))
    assert broker.close_out() == ["T1", "T1-F1"]
    assert [(t.status, t.reject_code) for t in found] == [("rejected", "not_attempted"), ("rejected", "not_attempted")] and held.cash_avail == Decimal("10.0")


# ---------------------------------------------------------------- backtest


def test_in_a_backtest_the_fill_date_is_the_trade_date_of_the_portfolio_so_a_stock_bought_that_day_is_locked_for_a_sale():
    bars = {"MSFT": weekday_bars("2026-11-05", [(400.0, 401.0, 399.0, 400.0)] * 3), "SPY": weekday_bars("2026-11-05", [(400.0, 401.0, 399.0, 400.0)] * 3)}
    order = synthetic(row("MSFT", "reduce", 3, 400.0, valid_until="2026-11-05T21:00:00Z"))
    broker, (t,) = make_broker(order, account(0.0, MSFT=(5, 300.0)), bars=bars, mode="backtest", now=at("2026-11-05 10:30"))
    assert t.status == "new" and broker.prices.trade_date == date(2026, 11, 6)
    broker.account.trading_portfolio(Portfolio(user="Jacob Pyrett", cash_avail=0.0, positions=[{"symbol": "MSFT", "shares": 5, "cost_basis": 300.0, "acquired": "2026-11-06"}]))      # bought on the entry date, in this run
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "not_sellable_today")                                                   # the fill would be on the 6th, the day it was bought
    held = Portfolio(user="Jacob Pyrett", cash_avail=0.0, positions=[{"symbol": "MSFT", "shares": 5, "cost_basis": 300.0, "acquired": "2026-11-05"}])
    broker, (t,) = make_broker(order, held, bars=bars, mode="backtest", now=at("2026-11-05 10:30"))
    assert (t.status, t.reject_code) == ("rejected", "not_sellable_today")                                                   # bought on the decision's own day: locked when the ticket is built (the Eastern date of the run)


# ---------------------------------------------------------------- funding

def funded(cash=51.23, shares=44, quotes=None, funding=None, **kw):
    """PG bought with the help of two sales: MSFT x5 (420) and TSLA x2 (330)."""
    held = account(cash, MSFT=(50, 150.0), TSLA=(20, 300.0), GM=(60, 10.0))
    prices = quotes or {"PG": 100.0, "MSFT": 420.0, "TSLA": 330.0}
    order = synthetic(row("PG", "open", shares, 100.0))
    return make_broker(order, held, quotes=prices, funding=funding or plan("PG", ("MSFT", 5, 420.0), ("TSLA", 2, 330.0)), **kw) + (order,)


def test_cut_note_collapses_white_space_drops_control_characters_and_keeps_the_first_words():
    assert cut_note("  a   b\n c\t d ", 20) == "a b c d" and cut_note("one two three four", 2) == "one two" and cut_note("x\x00y\x1bz", 20) == "x y z"
    assert cut_note(None, 20) == "" and cut_note(7, 20) == "" and cut_note("", 20) == ""


def test_a_broker_refuses_two_trades_with_one_id_and_changes_no_ticket_it_is_not_asked_about():
    order = two_trades()
    found = tk.build_tickets(order, account(1500.0, MSFT=(5, 300.0)), NOW, config(), "paper")
    with pytest.raises(ValueError, match="same trade_id"):
        PaperBroker(config(), "paper", manager(), order, Prices("paper", date(2026, 11, 5)), [found[0], found[0]], lambda t: (None, ""), NOW)
