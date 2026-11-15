"""The paper broker (prometheus/agents/trader/broker.py, Trader spec sections 6 and 7): placing against one snapshot in paper and backtest mode, slippage, the entry-bar rule on both sides of 09:30, the cash cut, the
refusals, and a funded purchase that would not fill sells nothing. No network, no model: the prices are a dict."""
from datetime import date

import pytest
from trader_helpers import NOW, account, acct_now, at, bar, config, decided, make_broker, manager, plan, row, synthetic, weekday_bars

from prometheus.agents.trader import tickets as tk
from prometheus.agents.trader.broker import CASH_HINT, PaperBroker, Prices, Refused, cut_note
from prometheus.agents.trader.record import REJECT_CODES, Trade
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, apply_fill, trade_amount

NO_SLIP = config(costs={"slippage_bps": 0})


def one_buy(shares=10, price=100.0, **kw):
    return synthetic(row("PG", "open", shares, price, **kw))


def as_order(order, trades):
    """The order with the broker's trades written on it, the way the Trader writes them: for the output guard."""
    order.trades = [t.model_dump(mode="json") for t in trades]
    return order


# ---------------------------------------------------------------- paper: a purchase

def test_a_paper_market_buy_fills_at_the_quote_plus_slippage_and_moves_the_cash_and_the_position():
    order = one_buy()
    broker, (t,) = make_broker(order, account(1500.0), quotes={"PG": 100.0})
    placed = broker.place("T1", "market", placed_by="model", note="  take  it ")
    assert placed is t and (t.status, t.shares, t.order_type, t.limit_price, t.placed_by, t.note, t.reject_code) == ("filled", 10, "market", None, "model", "take it", None)
    assert (t.fill.price, t.fill.shares, t.fill.amount, t.fill.commission, t.fill.date, t.fill.price_source, t.fill.slippage_bps) == (100.05, 10, 1000.5, 0.0, "2026-11-05", "quote", 5.0)
    assert t.placed_utc == "2026-11-05T15:30:00Z" and "market order filled at the quote 100.0000" in t.reason
    assert acct_now(broker).cash_avail == 499.5 and acct_now(broker).position("PG").shares == 10 and acct_now(broker).position("PG").cost_basis == 100.05
    assert broker.filled == ["T1"] and broker.placed == [("T1", "model")] and [x.trade_id for x in broker.filled_trades()] == ["T1"]


def test_a_buy_placed_with_no_order_gets_the_default_order_the_ticket_carries():
    order = decided()
    broker, (t,) = make_broker(order, account(10000.0), quotes={"PG": 167.0})
    broker.place("T1", placed_by="code")
    assert (t.status, t.order_type, t.limit_price, t.placed_by) == ("filled", "limit", 167.1, "code")
    assert (t.fill.price, t.fill.shares, t.fill.amount) == (167.0835, 44, 7351.67)                                      # min(the limit 167.10, the quote plus slippage 167.0835)
    assert acct_now(broker).cash_avail == 2648.33


def test_the_account_after_a_fill_is_what_apply_fill_gives_the_trader_has_no_portfolio_to_compare_it_with():
    order = one_buy()
    start = account(1500.0, MSFT=(5, 400.0))
    broker, found = make_broker(order, start, quotes={"PG": 100.0})
    broker.place("T1", "market")
    again = apply_fill(start, "PG", "buy", 10, 100.05, date(2026, 11, 5))
    assert acct_now(broker) == again and start.cash_avail == 1500.0                                                     # the portfolio given is not changed


def test_a_limit_buy_the_quote_has_not_reached_rests_and_moves_no_money():
    broker, (t,) = make_broker(one_buy(limit=100.0), account(1500.0), quotes={"PG": 100.5})
    broker.place("T1")
    assert (t.status, t.fill, t.reject_code, t.order_type, t.limit_price) == ("active", None, None, "limit", 100.0) and "above the limit" in t.reason
    assert acct_now(broker).cash_avail == 1500.0 and broker.filled == [] and t.placed_utc is not None


def test_a_market_buy_over_the_price_range_is_rejected_a_limit_buy_is_filled_at_most_at_its_limit():
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 102.0})
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "price_outside_limits") and t.fill is None and acct_now(broker).cash_avail == 1500.0
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0})
    broker.place("T1", "limit", 100.5)
    assert (t.status, t.fill.price) == ("filled", 100.05)


def test_no_order_of_any_type_fills_outside_the_range_whatever_the_quote_does():
    for quote in (50.0, 99.0, 100.0, 100.9, 101.0, 105.0, 400.0):
        for kind, price in (("market", 0.0), ("limit", 99.0), ("limit", 100.5), ("limit", 101.0)):
            broker, (t,) = make_broker(one_buy(), account(100000.0), quotes={"PG": quote})
            broker.place("T1", kind, price)
            if t.status == "filled":
                assert t.fill.price <= 101.0, (quote, kind, price)                                                      # the top of the range is 101 (+1%)
                if kind == "limit":
                    assert t.fill.price <= price


def test_there_is_no_fill_without_a_quote():
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"MSFT": 400.0})
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "price_unavailable") and "no usable quote" in t.reason


# ---------------------------------------------------------------- the cash

def test_a_purchase_the_cash_does_not_cover_is_cut_to_the_shares_it_can_pay_for_and_the_reason_says_so():
    broker, (t,) = make_broker(one_buy(10), account(450.0), quotes={"PG": 100.0})
    broker.place("T1", "market")
    assert (t.status, t.shares, t.shares_approved, t.fill.shares, t.fill.amount) == ("filled", 4, 10, 4, 400.2)
    assert "cut from 10 to 4 shares for the cash" in t.reason and acct_now(broker).cash_avail == 49.8


def test_a_purchase_the_cash_does_not_pay_for_even_one_share_of_is_insufficient_cash_and_hints_at_raise_funds():
    broker, (t,) = make_broker(one_buy(10), account(51.23), quotes={"PG": 100.0}, cash_hint=CASH_HINT)
    broker.place("T1", "market")
    assert (t.status, t.reject_code, t.fill) == ("rejected", "insufficient_cash", None) and "cash $51.23 does not pay for one share at 100.0500" in t.reason and "--raise-funds" in t.reason
    broker, (t,) = make_broker(one_buy(10), account(51.23), quotes={"PG": 100.0})
    broker.place("T1", "market")
    assert "--raise-funds" not in t.reason


def test_the_cash_check_is_to_the_cent_the_cash_that_just_pays_for_the_shares_buys_them():
    broker, (t,) = make_broker(one_buy(10), account(1000.5), quotes={"PG": 100.0})
    broker.place("T1", "market")
    assert (t.shares, acct_now(broker).cash_avail) == (10, 0.0)
    broker, (t,) = make_broker(one_buy(10), account(1000.49), quotes={"PG": 100.0})
    broker.place("T1", "market")
    assert t.shares == 9


def test_an_amount_is_rounded_half_up_the_way_the_account_book_rounds_not_to_the_even_cent():
    order = synthetic(row("PG", "open", 1, 500.125, zone=(400.0, 600.0)))
    broker, (t,) = make_broker(order, account(1000.0), quotes={"PG": 500.125}, cfg=NO_SLIP)
    broker.place("T1", "market")
    assert round(500.125, 2) == 500.12 and trade_amount(1, 500.125) == 500.13                                           # Python's round goes to the even cent; the book goes up
    assert (t.fill.price, t.fill.amount, acct_now(broker).cash_avail) == (500.125, 500.13, 499.87)


def test_a_commission_is_counted_in_the_cash_a_purchase_needs_and_written_on_the_fill():
    cfg = config(costs={"commission_per_order": 1.0})
    broker, (t,) = make_broker(one_buy(10), account(101.04), quotes={"PG": 100.0}, cfg=cfg)
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "insufficient_cash")                                               # 100.05 and a commission of 1.00 is more than 101.04
    broker, (t,) = make_broker(one_buy(10), account(101.05), quotes={"PG": 100.0}, cfg=cfg)
    broker.place("T1", "market")
    assert (t.status, t.shares, t.fill.commission) == ("filled", 1, 1.0)


# ---------------------------------------------------------------- paper: a sale

def sale(shares=3, price=400.0, **kw):
    return synthetic(row("MSFT", "reduce", shares, price, **kw))


def test_a_paper_market_sale_fills_at_the_quote_less_slippage_and_the_proceeds_go_to_the_cash():
    broker, (t,) = make_broker(sale(), account(10.0, MSFT=(5, 300.0)), quotes={"MSFT": 400.0})
    broker.place("T1", "market")
    assert (t.status, t.side, t.shares, t.fill.price, t.fill.amount, t.fill.price_source) == ("filled", "sell", 3, 399.8, 1199.4, "quote")
    assert acct_now(broker).cash_avail == 1209.4 and acct_now(broker).position("MSFT").shares == 2


def test_a_sale_that_sells_the_whole_position_removes_it():
    broker, (t,) = make_broker(sale(5), account(0.0, MSFT=(5, 300.0)), quotes={"MSFT": 400.0})
    broker.place("T1", "market")
    assert t.shares == 5 and acct_now(broker).position("MSFT") is None


def test_a_market_sale_under_the_range_is_rejected_and_a_limit_sale_rests_below_its_limit():
    broker, (t,) = make_broker(sale(), account(0.0, MSFT=(5, 300.0)), quotes={"MSFT": 390.0})
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "price_outside_limits") and "below the lowest price allowed" in t.reason
    broker, (t,) = make_broker(sale(), account(0.0, MSFT=(5, 300.0)), quotes={"MSFT": 399.0})
    broker.place("T1", "limit", 400.0)
    assert (t.status, t.reject_code) == ("active", None) and acct_now(broker).position("MSFT").shares == 5


def test_a_sale_is_cut_to_the_shares_sellable_when_it_is_placed_not_when_it_was_built():
    order = sale(3)
    broker, (t,) = make_broker(order, account(0.0, MSFT=(5, 300.0)), quotes={"MSFT": 400.0})
    broker.account.trading_portfolio(Portfolio(user="Jacob Pyrett", cash_avail=0.0, positions=[{"symbol": "MSFT", "shares": 5, "cost_basis": 300.0, "acquired": "2026-11-05", "recent_shares": 4}]))     # 4 of the 5 were bought today, in this run
    broker.place("T1", "market")
    assert (t.status, t.shares, t.fill.shares) == ("filled", 1, 1) and "cut from 3 to 1 shares: only that many were sellable" in t.reason
    broker, (t,) = make_broker(order, account(0.0, MSFT=(5, 300.0)), quotes={"MSFT": 400.0})
    broker.account.trading_portfolio(Portfolio(user="Jacob Pyrett", cash_avail=0.0, positions=[{"symbol": "MSFT", "shares": 5, "cost_basis": 300.0, "acquired": "2026-11-05"}]))
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "not_sellable_today")
    broker, (t,) = make_broker(order, account(0.0, MSFT=(5, 300.0)), quotes={"MSFT": 400.0})
    broker.account.trading_portfolio(account(0.0))
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "not_held")


# ---------------------------------------------------------------- sells before buys, and the refusals

def two_trades():
    return synthetic(row("PG", "open", 10, 100.0), row("MSFT", "reduce", 3, 400.0))


def test_every_sale_is_placed_before_any_purchase_and_the_refusal_names_the_sale():
    broker, found = make_broker(two_trades(), account(1500.0, MSFT=(5, 300.0)), quotes={"PG": 100.0, "MSFT": 400.0})
    with pytest.raises(Refused, match=r"place the sale T2 first"):
        broker.place("T1", "market")
    assert found[0].status == "new" and acct_now(broker).cash_avail == 1500.0                                           # the refusal changed nothing
    broker.place("T2", "market")
    broker.place("T1", "market")
    assert broker.filled == ["T2", "T1"] and [t.status for t in found] == ["filled", "filled"]


def test_a_sale_that_was_rejected_does_not_hold_up_the_purchases():
    broker, found = make_broker(two_trades(), account(1500.0), quotes={"PG": 100.0, "MSFT": 400.0})     # MSFT is not held: T2 is rejected at once
    assert found[1].status == "rejected" and found[1].reject_code == "not_held"
    broker.place("T1", "market")
    assert found[0].status == "filled"


def test_a_trade_is_placed_once_a_filled_a_rejected_and_a_resting_trade_cannot_be_placed_again():
    broker, found = make_broker(synthetic(row("PG", "open", 10, 100.0), row("KO", "open", 1, 60.0, approved=False), row("GM", "open", 5, 50.0, limit=49.0)), account(5000.0),
                                quotes={"PG": 100.0, "GM": 52.0})
    broker.place("T1", "market")
    broker.place("T3", "limit", 49.0)
    for trade_id, state in (("T1", "filled"), ("T2", "rejected, not_permitted"), ("T3", "active")):
        with pytest.raises(Refused, match=rf"{trade_id} was already placed \(it is {state}\)"):
            broker.place(trade_id, "market")
    assert acct_now(broker).cash_avail == 5000.0 - 1000.5 and broker.filled == ["T1"]                                   # nothing was traded twice


def test_an_unknown_trade_id_is_refused_and_so_is_something_that_is_not_text():
    broker, _ = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0})
    for bad in ("T9", "", None, 7, "T1-F1"):
        with pytest.raises(Refused, match="there is no trade"):
            broker.place(bad)


def test_a_funding_sale_cannot_be_placed_by_name_it_goes_with_its_purchase():
    held = account(10.0, MSFT=(50, 150.0))
    broker, found = make_broker(one_buy(2, 100.0), held, quotes={"PG": 100.0, "MSFT": 420.0}, funding=plan("PG", ("MSFT", 5, 420.0)))
    assert [t.trade_id for t in found] == ["T1", "T1-F1"]
    with pytest.raises(Refused, match="T1-F1 is a funding sale: it is placed together with T1, never by you"):
        broker.place("T1-F1", "market")
    assert [t.trade_id for t in broker.tickets()] == ["T1"]                                                              # a funding sale is never listed as something to place


@pytest.mark.parametrize("kind, price, message", [
    ("stop", 0.0, "order_type must be 'market' or 'limit'"),
    ("MARKET", 0.0, "order_type must be 'market' or 'limit'"),
    ("", 100.0, 'give an order_type of "limit" with a limit_price'),
    ("market", 100.0, "a market order has no limit_price"),
    ("limit", 98.99, r"limit_price 98.9900 is outside the range allowed for T1, 99.00 to 101.00"),
    ("limit", 101.01, r"limit_price 101.0100 is outside the range allowed for T1, 99.00 to 101.00"),
    ("limit", -1.0, "is outside the range allowed"),
    ("limit", "100", "limit_price must be a number"),
    ("limit", True, "limit_price must be a number"),
    ("limit", None, "limit_price must be a number"),
])
def test_an_order_the_model_may_not_give_is_refused_with_the_way_to_fix_it_and_nothing_is_placed(kind, price, message):
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0})
    with pytest.raises(Refused, match=message):
        broker.place("T1", kind, price)
    assert (t.status, t.placed_by, t.placed_utc, broker.placed, acct_now(broker).cash_avail) == ("new", None, None, [], 1500.0)


def test_a_limit_price_on_the_edge_of_the_range_is_inside_and_one_with_more_places_is_rounded_to_four():
    for price, expected in ((99.0, 99.0), (101.0, 101.0), (100.00004, 100.0)):
        broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 90.0})
        broker.place("T1", "limit", price)
        assert t.limit_price == expected, price


def test_a_limit_order_with_no_price_takes_the_default_limit_and_market_orders_can_be_turned_off():
    broker, (t,) = make_broker(one_buy(limit=100.5), account(1500.0), quotes={"PG": 100.0})
    broker.place("T1", "limit")
    assert (t.order_type, t.limit_price) == ("limit", 100.5)
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0})                                          # a plan with no limit price: the default is a market order
    broker.place("T1", "limit")
    assert (t.order_type, t.limit_price) == ("limit", 100.0)                                                             # the reference price, inside the range
    off = config(orders={"allow_market": False})
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0}, cfg=off)
    assert (t.order_type, t.limit_price) == ("limit", 100.0)
    with pytest.raises(Refused, match="market orders are turned off"):
        broker.place("T1", "market")
    broker.place("T1")
    assert t.status == "filled"


def test_nothing_can_be_placed_after_the_run_was_closed():
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0})
    broker.closed = True
    with pytest.raises(Refused, match="the run was closed"):
        broker.place("T1", "market")
    assert t.status == "new"


# ---------------------------------------------------------------- the permission is checked again as a trade is placed (rule T1)

def test_a_trade_that_stopped_being_permitted_after_its_ticket_was_built_is_rejected_when_it_is_placed():
    order = one_buy()
    broker, (t,) = make_broker(order, account(1500.0), quotes={"PG": 100.0})
    order.trade_decisions[0]["will_trade"] = "no_user_denied"                                                           # the decision changes between the ticket and the placing
    broker.place("T1", "market")
    assert (t.status, t.reject_code, t.fill) == ("rejected", "not_permitted", None) and "no_user_denied" in t.reason and acct_now(broker).cash_avail == 1500.0
    order = one_buy()
    broker, (t,) = make_broker(order, account(1500.0), quotes={"PG": 100.0})
    order.reviews[0]["valid_until_utc"] = "2026-11-05T15:00:00Z"                                                        # the approval ran out before the Trader got to it
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "approval_expired")


def test_the_permission_closure_is_called_for_every_trade_placed():
    asked = []
    order = two_trades()
    broker, found = make_broker(order, account(1500.0, MSFT=(5, 300.0)), quotes={"PG": 100.0, "MSFT": 400.0})
    broker.permit = lambda t: (asked.append(t.trade_id), (None, ""))[1]
    broker.place("T2", "market")
    broker.place("T1", "market")
    assert asked == ["T2", "T1"]


# ---------------------------------------------------------------- an active trade judged again by a later run, by code

def test_a_resting_limit_buy_is_filled_by_a_later_run_when_the_quote_comes_down_to_its_limit():
    order = one_buy(limit=100.0)
    broker, (t,) = make_broker(order, account(1500.0), quotes={"PG": 100.5})
    broker.place("T1", placed_by="model", note="wait for it")
    assert t.status == "active"
    later, (t2,) = make_broker(order, account(1500.0), quotes={"PG": 99.0}, loaded=[Trade.model_validate(t.model_dump(mode="json"))])      # the next run: the trade is loaded from the order, the quote moved
    later.rejudge("T1")
    assert (t2.status, t2.fill.price, t2.fill.shares, t2.placed_by, t2.note) == ("filled", 99.0495, 10, "model", "wait for it") and later.filled == ["T1"]
    assert later.placed == []                                                                                             # judged again, not placed again


def test_a_judged_again_trade_whose_permission_ran_out_is_rejected_and_the_model_can_not_place_it():
    order = one_buy(limit=100.0)
    broker, (t,) = make_broker(order, account(1500.0), quotes={"PG": 100.5})
    broker.place("T1")
    order.reviews[0]["valid_until_utc"] = "2026-11-05T15:00:00Z"
    broker.rejudge("T1")
    assert (t.status, t.reject_code) == ("rejected", "approval_expired")
    broker, (t,) = make_broker(order := one_buy(limit=100.0), account(1500.0), quotes={"PG": 100.5})
    broker.place("T1")
    with pytest.raises(Refused, match="already placed"):
        broker.place("T1", "market")                                                                                      # an active trade is not `new`: only code judges it again


def test_rejudge_leaves_a_trade_that_is_not_active_alone():
    broker, (t,) = make_broker(one_buy(), account(1500.0), quotes={"PG": 100.0})
    broker.rejudge("T1")
    assert t.status == "new"                                                                                              # still a ticket
    broker.place("T1", "market")
    before = t.model_dump_json()
    broker.rejudge("T1")
    assert t.model_dump_json() == before


def test_a_resting_order_whose_price_cannot_be_had_in_a_later_run_stays_active_an_outage_does_not_make_it_final():
    order = one_buy(limit=100.0)
    broker, (t,) = make_broker(order, account(1500.0), quotes={"PG": 100.5})
    broker.place("T1", placed_by="model")
    later, (t2,) = make_broker(order, account(1500.0), quotes={}, loaded=[Trade.model_validate(t.model_dump(mode="json"))])                          # the quote is gone this time
    later.rejudge("T1")
    assert (t2.status, t2.reject_code, t2.placed_by) == ("active", None, "model") and "could not be judged again in this run: there is no usable quote" in t2.reason and later.filled == []
    later.prices.failed = "the quotes could not be fetched (RuntimeError)"
    later.rejudge("T1")
    assert t2.status == "active" and "the quotes could not be fetched" in t2.reason


# ---------------------------------------------------------------- close out

def test_what_nobody_placed_is_rejected_not_attempted_and_nothing_is_left_new():
    broker, found = make_broker(two_trades(), account(1500.0, MSFT=(5, 300.0)), quotes={"PG": 100.0, "MSFT": 400.0})
    broker.place("T2", "market")
    assert broker.close_out() == ["T1"]
    assert [(t.status, t.reject_code) for t in found] == [("rejected", "not_attempted"), ("filled", None)] and "did not place it" in found[0].reason
    assert not [t for t in found if t.status == "new"] and broker.close_out() == []


def test_the_funding_sales_of_a_purchase_that_was_not_attempted_are_not_attempted_too():
    held = account(10.0, MSFT=(50, 150.0))
    broker, found = make_broker(one_buy(2, 100.0), held, quotes={"PG": 100.0, "MSFT": 420.0}, funding=plan("PG", ("MSFT", 5, 420.0)))
    assert broker.close_out() == ["T1", "T1-F1"]
    assert [(t.status, t.reject_code) for t in found] == [("rejected", "not_attempted"), ("rejected", "not_attempted")] and held.cash_avail == 10.0


# ---------------------------------------------------------------- backtest

def pg_bars(*rows, start="2026-11-05"):
    return {"PG": weekday_bars(start, rows), "SPY": weekday_bars(start, [(400.0, 401.0, 399.0, 400.0)] * len(rows))}


def test_a_backtest_decision_after_the_open_enters_at_the_next_sessions_open_and_one_before_the_open_at_that_days_open():
    rows = ((100.0, 101.0, 99.0, 100.5), (102.0, 103.0, 100.0, 101.0), (101.0, 102.0, 100.0, 101.5))                          # Thursday the 5th, Friday the 6th, Monday the 9th
    for when, day, opened in (("2026-11-05 10:30", "2026-11-06", 102.0), ("2026-11-05 08:00", "2026-11-05", 100.0), ("2026-11-05 09:30", "2026-11-06", 102.0), ("2026-11-05 09:29", "2026-11-05", 100.0)):
        order = synthetic(row("PG", "open", 10, 102.0, valid_until="2026-11-05T21:00:00Z"))
        broker, (t,) = make_broker(order, account(5000.0), bars=pg_bars(*rows), mode="backtest", now=at(when), cfg=config(orders={"limit_band_pct": 0.1, "max_chase_pct": 0.1}))
        broker.place("T1", "market")
        assert (t.status, t.fill.date, t.fill.price_source) == ("filled", day, "bar_open"), when
        assert t.fill.price == round(opened * 1.0005, 4), when


def test_a_backtest_never_reads_a_bar_before_the_entry_date_the_future_stays_out_of_the_decision():
    rows = ((50.0, 51.0, 49.0, 50.5), (100.0, 101.0, 99.0, 100.5), (100.0, 101.0, 99.0, 100.5))
    bars = pg_bars(*rows, start="2026-11-04")                                                                                 # Wednesday the 4th is before the decision of Thursday 10:30
    broker, (t,) = make_broker(synthetic(row("PG", "open", 10, 100.0, valid_until="2026-11-05T21:00:00Z")), account(5000.0), bars=bars, mode="backtest", now=at("2026-11-05 10:30"))
    broker.place("T1", "market")
    assert t.fill.date == "2026-11-06" and t.fill.price == 100.05                                                          # Friday's open 100.0, not Wednesday's 50.0 nor Thursday's


def test_a_backtest_limit_buy_fills_at_the_limit_when_the_day_touches_it_and_is_rejected_when_it_does_not():
    order = synthetic(row("PG", "open", 10, 100.0, limit=99.0, valid_until="2026-11-05T21:00:00Z"))
    bars = pg_bars((100.0, 100.5, 99.0, 100.0), (100.0, 100.5, 99.0, 100.0))
    broker, (t,) = make_broker(order, account(5000.0), bars=bars, mode="backtest", now=at("2026-11-04 12:00"))                # the entry bar is Thursday's: open 100, low 99
    broker.place("T1")
    assert (t.status, t.fill.price, t.fill.price_source, t.fill.slippage_bps) == ("filled", 99.0, "bar_limit", 0.0)
    bars = pg_bars((100.0, 100.5, 99.01, 100.0), (100.0, 100.5, 99.01, 100.0))
    broker, (t,) = make_broker(order, account(5000.0), bars=bars, mode="backtest", now=at("2026-11-04 12:00"))
    broker.place("T1")
    assert (t.status, t.reject_code, t.fill) == ("rejected", "limit_not_reached", None)


def test_a_backtest_with_no_bar_stored_from_the_entry_day_on_waits_every_ticket_is_active():
    order = two_trades()
    bars = pg_bars((100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0), start="2026-11-02")                               # Monday and Tuesday: before the decision
    broker, found = make_broker(order, account(1500.0, MSFT=(5, 300.0)), bars={**bars, "MSFT": bars["PG"]}, mode="backtest")
    assert broker.prices.entry_date is None
    broker.place("T2", "market")
    broker.place("T1", "market")
    assert [(t.status, t.fill) for t in found] == [("active", None), ("active", None)] and "waiting for the next session" in found[0].reason
    assert acct_now(broker).cash_avail == 1500.0 and broker.filled == []


def test_a_symbol_with_no_bar_on_the_entry_date_while_others_have_one_is_price_unavailable():
    bars = {"SPY": weekday_bars("2026-11-05", [(400.0, 401.0, 399.0, 400.0)] * 3)}
    broker, (t,) = make_broker(one_buy(), account(1500.0), bars=bars, mode="backtest")
    assert broker.prices.entry_date == date(2026, 11, 6)
    broker.place("T1", "market")
    assert (t.status, t.reject_code) == ("rejected", "price_unavailable") and "PG has no stored bar on 2026-11-06" in t.reason


def test_backtest_slippage_comes_from_the_settings():
    rows = ((100.0, 101.0, 99.0, 100.0),) * 3
    for bps, expected in ((0, 100.0), (5, 100.05), (50, 100.5)):
        broker, (t,) = make_broker(synthetic(row("PG", "open", 10, 100.0, zone=(90.0, 110.0), valid_until="2026-11-05T21:00:00Z")), account(5000.0), bars=pg_bars(*rows), mode="backtest",
                                   cfg=config(costs={"slippage_bps": bps}, orders={"max_chase_pct": 0.1}), now=at("2026-11-04 12:00"))
        broker.place("T1", "market")
        assert t.fill.price == expected and t.fill.slippage_bps == bps


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


def test_a_funded_purchase_sells_first_then_buys_with_the_cash_now_there_and_everything_is_filled_in_that_order():
    broker, found, order = funded(shares=20)
    assert [t.trade_id for t in found] == ["T1", "T1-F1", "T1-F2"]
    broker.place("T1", "market", placed_by="model")
    buy, f1, f2 = found
    assert [t.status for t in found] == ["filled", "filled", "filled"] and broker.filled == ["T1-F1", "T1-F2", "T1"]                      # the sales, then the purchase
    assert (f1.fill.price, f1.fill.amount, f2.fill.price, f2.fill.amount) == (419.79, 2098.95, 329.835, 659.67)
    assert (buy.fill.shares, buy.fill.price, buy.fill.amount) == (20, 100.05, 2001.0) and buy.reason.endswith("market order filled at the quote 100.0000 with 5 bps of slippage")
    assert (f1.placed_by, f2.placed_by, f1.record_id, f1.funds_for) == ("model", "model", buy.record_id, "T1") and f1.placed_utc == buy.placed_utc
    assert acct_now(broker).cash_avail == round(51.23 + 2098.95 + 659.67 - 2001.0, 2) and acct_now(broker).position("MSFT").shares == 45 and acct_now(broker).position("TSLA").shares == 18
    assert wo.verify_trades(as_order(order, found)) == []                                                                              # the output guard has nothing to say


def test_a_funded_purchase_that_would_not_fill_sells_nothing_a_resting_limit_leaves_the_sales_waiting():
    held = account(51.23, MSFT=(50, 150.0))
    order = synthetic(row("PG", "open", 20, 100.0, limit=100.0))
    broker, found = make_broker(order, held, quotes={"PG": 100.5, "MSFT": 420.0}, funding=plan("PG", ("MSFT", 5, 420.0)))
    broker.place("T1")
    buy, sell = found
    assert (buy.status, sell.status, sell.fill, sell.reject_code) == ("active", "active", None, None) and "waiting for its purchase T1" in sell.reason
    assert acct_now(broker) == held and broker.filled == [] and (sell.placed_by, sell.placed_utc) == (buy.placed_by, buy.placed_utc)


def test_a_funded_purchase_the_price_rejects_sells_nothing_and_the_sales_are_funding_failed():
    broker, found, order = funded(quotes={"PG": 150.0, "MSFT": 420.0, "TSLA": 330.0}, shares=20)                                       # PG is far above its range
    held = acct_now(broker)
    broker.place("T1", "market")
    buy, f1, f2 = found
    assert buy.reject_code == "price_outside_limits" and (f1.status, f1.reject_code, f2.reject_code) == ("rejected", "funding_failed", "funding_failed")
    assert "T1 was rejected (price_outside_limits)" in f1.reason and acct_now(broker) == held and broker.filled == []


def test_a_funded_purchase_the_sales_could_not_pay_for_even_one_share_of_is_insufficient_cash_and_nothing_is_sold():
    held = account(10.0, MSFT=(50, 150.0))
    order = synthetic(row("PG", "open", 5, 100.0))
    broker, found = make_broker(order, held, quotes={"PG": 100.0, "MSFT": 10.0}, funding=plan("PG", ("MSFT", 5, 10.0)), cfg=config(orders={"max_chase_pct": 0.5, "limit_band_pct": 0.5}))
    broker.place("T1", "market")
    buy, sale_ = found
    assert (buy.status, buy.reject_code) == ("rejected", "insufficient_cash") and "nothing was sold" in buy.reason                         # 10.00 plus about 49.97 does not pay for a share at 100.05
    assert (sale_.status, sale_.reject_code) == ("rejected", "funding_failed") and acct_now(broker) == held and broker.filled == []
    assert wo.verify_trades(as_order(order, found)) == []


def test_a_sale_that_cannot_fill_is_funding_failed_and_the_others_go_on_the_purchase_is_cut_to_what_the_rest_pays_for():
    broker, found, order = funded(shares=20, quotes={"PG": 100.0, "MSFT": 380.0, "TSLA": 330.0})                                       # MSFT is far under its range: the market sale is rejected
    broker.place("T1", "market")
    buy, f1, f2 = found
    assert (f1.status, f1.reject_code) == ("rejected", "funding_failed") and "below the lowest price allowed" in f1.reason and f2.status == "filled"
    assert buy.status == "filled" and buy.shares == 7 and "cut from 20 to 7 shares for the cash" in buy.reason                              # 51.23 + 659.67 = 710.90 pays for 7 at 100.05 (700.35), not 8 (800.40)
    assert broker.filled == ["T1-F2", "T1"] and acct_now(broker).position("MSFT").shares == 50


def test_a_sale_with_no_price_counts_as_nothing_and_a_run_with_no_sale_that_fills_buys_what_the_cash_pays_for():
    broker, found, order = funded(cash=500.0, shares=20, quotes={"PG": 100.0})                                                           # no quote for MSFT or TSLA
    broker.place("T1", "market")
    buy, f1, f2 = found
    assert (f1.reject_code, f2.reject_code) == ("funding_failed", "funding_failed") and "no usable quote" in f1.reason
    assert buy.status == "filled" and buy.shares == 4 and broker.filled == ["T1"]


def test_the_permission_of_a_funded_purchase_is_checked_before_anything_is_sold():
    broker, found, order = funded(shares=20)
    order.trade_decisions[0]["will_trade"] = "no_user_denied"
    held = acct_now(broker)
    broker.place("T1", "market")
    assert [(t.status, t.reject_code) for t in found] == [("rejected", "not_permitted"), ("rejected", "funding_failed"), ("rejected", "funding_failed")] and acct_now(broker) == held


def test_a_funded_purchase_never_leaves_a_filled_sale_without_a_filled_purchase_over_a_range_of_prices_and_cash():
    for pg in (60.0, 99.0, 100.0, 101.0, 150.0):
        for cash in (0.0, 51.23, 400.0, 3000.0):
            for msft in (350.0, 380.0, 420.0, 450.0):
                broker, found, order = funded(cash=cash, shares=30, quotes={"PG": pg, "MSFT": msft, "TSLA": 330.0})
                broker.place("T1", "market")
                legs_filled = [t for t in found if t.kind == "funding" and t.status == "filled"]
                assert not legs_filled or found[0].status == "filled", (pg, cash, msft)
                assert wo.verify_trades(as_order(order, found)) == [], (pg, cash, msft)


def test_cut_note_collapses_white_space_drops_control_characters_and_keeps_the_first_words():
    assert cut_note("  a   b\n c\t d ", 20) == "a b c d" and cut_note("one two three four", 2) == "one two" and cut_note("x\x00y\x1bz", 20) == "x y z"
    assert cut_note(None, 20) == "" and cut_note(7, 20) == "" and cut_note("", 20) == ""


def test_every_reject_code_the_broker_uses_is_in_the_closed_list_and_a_trade_can_carry_it():
    broker, found = make_broker(two_trades(), account(0.0), quotes={"PG": 150.0})
    broker.place("T1", "market")
    broker.close_out()
    for t in found:
        assert t.reject_code in REJECT_CODES


def test_a_broker_refuses_two_trades_with_one_id_and_changes_no_ticket_it_is_not_asked_about():
    order = two_trades()
    found = tk.build_tickets(order, account(1500.0, MSFT=(5, 300.0)), NOW, config(), "paper")
    with pytest.raises(ValueError, match="same trade_id"):
        PaperBroker(config(), "paper", manager(), order, Prices("paper", date(2026, 11, 5)), [found[0], found[0]], lambda t: (None, ""), NOW)
