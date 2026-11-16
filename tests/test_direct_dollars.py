"""A DIRECT ORDER WITH MONEY: as many whole shares as the money pays for, and a fraction of a share with what is left (design choices entry 111, M3; design/dollar_orders_spec.md section 5).

Jacob (2026-11-15): *"When a direct order is given, buy as many whole shares as possible and a fractional share for the leftover funds available."* A stock the user named with dollars, or with no number, is sized by the
Account Manager (`AccountManager._size_direct`): the money is spread as ever (round robin in whole shares, one share of each stock in turn), what it leaves buys a fraction of a share of the stock whose turn is next, and
every such purchase is an ORDER IN DOLLARS, so that at the fill each buys its dollars' worth. The recommendation path is not changed (its leftover stays cash); a number of shares the user typed or answered is an order in shares.

REAL DATA ONLY, as `test_direct_order.py` (whose helpers these tests use): the real model's recorded forms for "Buy $5,000 of MSFT" and the others (`direct_order_forms.json`), its recorded answer to "how much?"
(`clarify_ask_mode.json`), the stored closes of Friday 2026-11-13 and the real next open (Friday's, with the run as of Thursday evening), the real Trader with no model.
"""
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from handshake_helpers import FRIDAY, MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_direct_order import BILL, BILL_PROFILE, account, direct_order, sized_order

from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, trade_amount

pytestmark = pytest.mark.usefixtures("friday_close")

EVENING = datetime(2026, 11, 12, 20, 30, tzinfo=timezone.utc)
MSFT_OPEN = Decimal("519.745")                                   # the real open of 2026-11-13 (the close it was sized at is 523.88)


def test_five_thousand_dollars_of_one_stock_is_whole_shares_and_a_fraction_and_exactly_five_thousand_dollars():
    order, am, *_ = sized_order("Buy $5,000 of MSFT", mode="auto")
    (d,), (entry,) = order.trade_decisions, order.direct_orders
    price = close_of("MSFT")
    assert d["shares"] == entry["shares"] == sh.buys(Decimal("5000"), price) == Decimal("9.54417") and int(d["shares"]) == 9 and (d["dollars"], d["est_cost"], d["size"]["pot"]) == (Decimal("5000"), Decimal("5000"), Decimal("5000"))
    assert (d["will_trade"], d["price"], entry["how_many"], d["size"]["how"]) == ("yes_user", price, "budget", "budget") and wo.verify_trade(order, d["record_id"], MOMENT) == []
    am.perform_trades(order, BILL, trader(), MOMENT)
    (t,) = order.trades
    assert (t["status"], t["dollars"], t["shares"], t["fill"]["amount"]) == ("filled", Decimal("5000"), Decimal("9.54417"), Decimal("5000")) and wo.verify_trades(order) == [] and order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("15000")


def test_the_same_order_at_the_next_sessions_real_open_buys_its_dollars_worth_there_more_shares_for_a_lower_price():
    order, am, *_ = sized_order("Buy $5,000 of MSFT", mode="auto")
    am.perform_trades(order, BILL, trader(mode="backtest"), EVENING)
    (t,) = order.trades
    assert t["fill"]["price"] == MSFT_OPEN < close_of("MSFT") and t["fill"]["price_source"] == "bar_open"
    assert (t["status"], t["shares"], t["fill"]["amount"]) == ("filled", sh.buys(Decimal("5000"), MSFT_OPEN), Decimal("5000")) and t["shares"] > order.trade_decisions[0]["shares"] and wo.verify_trades(order) == []


def test_several_stocks_with_a_budget_add_up_to_the_budget_and_the_fraction_goes_to_the_stock_with_the_fewest_shares():
    order, am, *_ = sized_order("I want to buy MSFT and TSLA and spend $6,000", mode="auto")
    msft, tsla = order.trade_decisions
    assert msft["est_cost"] + tsla["est_cost"] == Decimal("6000") and msft["dollars"] + tsla["dollars"] == Decimal("6000") and {d["size"]["pot"] for d in order.trade_decisions} == {Decimal("6000")} and {e["how_many"] for e in order.direct_orders} == {"budget"}
    whole = [d for d in order.trade_decisions if sh.whole(d["shares"])]
    fractions = [d for d in order.trade_decisions if not sh.whole(d["shares"])]
    assert len(fractions) == 1 and len(whole) == 1 and int(fractions[0]["shares"]) <= whole[0]["shares"]       # one of the two took the fraction: the one that had the fewest whole shares
    am.perform_trades(order, BILL, trader(), MOMENT)
    assert [t["status"] for t in order.trades] == ["filled", "filled"] and sum(t["fill"]["amount"] for t in order.trades) == Decimal("6000") and wo.verify_trades(order) == []


def test_with_no_number_and_no_dollars_the_whole_cash_is_the_order_and_ask_mode_asks_dollars_for_each_stock_in_full():
    order, *_ = sized_order("I want to buy MSFT", mode="auto")
    d = order.trade_decisions[0]
    assert (d["dollars"], d["size"]["pot"], d["shares"]) == (Decimal("20000"), Decimal("20000"), sh.buys(Decimal("20000"), close_of("MSFT"))) and order.direct_orders[0]["how_many"] == "cash"
    asked, _, channel, cards = sized_order("I want to buy MSFT", mode="ask", answers="percent_each_no_raise_needed")             # the real model reading "5% of my account in each company": $1,000 of each stock, in full
    e = asked.trade_decisions[0]
    assert len(channel.cards) == 1 and (e["dollars"], e["est_cost"], e["shares"]) == (Decimal("1000"), Decimal("1000"), sh.buys(Decimal("1000"), close_of("MSFT"))) and not sh.whole(e["shares"]) and asked.direct_orders[0]["how_many"] == "asked"
    assert e["size"]["pot"] == Decimal("20000") and e["size"]["binding"] == "asked" and cards[0]["dollars"] == Decimal("1000") and "order       buy $1,000.00: about 1.908" in cards[0]["text"]


def test_a_number_of_shares_the_user_typed_or_answered_is_an_order_in_shares_as_it_was():
    typed, am = direct_order("buy", "MSFT", 30)
    d = typed.trade_decisions[0]
    assert "dollars" not in d and d["size"] is None and d["shares"] == 30 and isinstance(d["shares"], Decimal) and wo.verify_trade(typed, d["record_id"], MOMENT) == []
    am.perform_trades(typed, BILL, trader(), MOMENT)
    assert typed.trades[0]["dollars"] is None and typed.trades[0]["shares"] == 30
    answered, *_ = sized_order("I want to buy MSFT", mode="ask", answers="shares_each_raise_yes")                                      # "10 shares of each": whole shares, an order in shares
    e = answered.trade_decisions[0]
    assert e["shares"] == 10 and "dollars" not in e and e["size"]["pot"] == Decimal("20000") and answered.direct_orders[0]["how_many"] == "asked"
    typed.trade_decisions[0]["dollars"] = Decimal("100")
    assert any("an order in shares: it carries no dollars" in r for r in wo.verify_trade(typed, d["record_id"], MOMENT))                # the word dollars on a typed order permits nothing


def test_with_fractions_off_a_direct_order_with_money_is_whole_shares_and_no_order_in_dollars():
    am = account(BILL_PROFILE, fractional_even_split=False)
    from prometheus.agents.account_manager import intent as it
    from test_direct_order import real_form
    typed = "Buy $5,000 of MSFT"
    intent = it.parse_intent(real_form(typed), typed)
    work = wo.new_work_order(typed, requester="desk", action_type="trade")
    assert am.name_the_stocks(work, intent.orders, typed, MOMENT) == []
    out = am.decide_direct(work, BILL, MOMENT, "auto", None, clarify=None, said_how_much=True, request_text=typed, balance=None, budget=Decimal("5000"))
    d = out.trade_decisions[0]
    assert (d["shares"], d["est_cost"]) == (9, Decimal("4714.92")) and "dollars" not in d and isinstance(d["shares"], Decimal) and d["size"]["pot"] == Decimal("5000")           # the whole-share round robin of entry 98: $285.08 unspent, held by the pot
