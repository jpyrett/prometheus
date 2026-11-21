"""THE DOLLARS THE USER WROTE FOR A STOCK ARE THAT STOCK'S LIMIT (design choices entry 122; design/direct_dollars_each_spec.md).

The bug (a live Auto session of 2026-11-20): "Buy $3,000 of XOM and $3,000 of MSFT" spent the whole $10,000. The form had one `budget` and no amount for each stock, so the model had nowhere to put the $3,000 of each and the
Account Manager spent all the cash round robin. Now an order has `dollars` (D1), code checks it as money the user wrote (D2), each such purchase is sized as an amount for EACH stock and is an order in dollars (D3), Ask mode asks
nothing that the user already answered (D4), a request that wrote an amount and whose reading has none is refused, never turned into all the cash (D5), and a purchase with no number next to a purchase with one is asked about or not
placed (D6).

REAL DATA ONLY, as `test_direct_order.py` and `test_direct_dollars.py` (whose helpers these tests use):
  - the real model's replies, recorded on 2026-11-20 with the new prompt (`tests/fixtures/session_real/direct_dollars_forms.json`, made by `scripts/capture_direct_dollars_forms.py`): they go through the real
    `AccountManager.understand_request` and `parse_intent`; the checks of D2 and D5 are made on a real reply edited by ONE field;
  - the stored closes of Friday 2026-11-13 read by the real Data Clerk (the `friday_close` fixture) and the real Trader without a model; the company names are the REAL symbol directory's answers for XOM and MSFT of the live
    session (the stored directory of the tests has no XOM);
  - the live session of 2026-11-20 itself (`tests/fixtures/session_real/auto_direct_dollars_each.json`: Auto mode, a throwaway account with $10,000, live quotes, the real Trader and the real Postgres book).
"""
import copy
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_account_manager_clarify import RECORDED, channel_of
from test_account_manager_clarify import Replay as AskReplay
from test_direct_order import BILL, BILL_PROFILE, SAMPLES, Replay, account, clerk, db, never, store  # noqa: F401  (db and store are fixtures)

from prometheus.agents.account_manager import direct
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.application.web import views
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import dumps, money
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.langgraph import desk
from prometheus.prompts import account_manager_request as prompt

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
FORMS = json.loads((FIXTURES / "direct_dollars_forms.json").read_text(encoding="utf-8"))["requests"]
LIVE = json.loads((FIXTURES / "auto_direct_dollars_each.json").read_text(encoding="utf-8"))
ROWS = LIVE["order"]["data"]["companies"]       # the real symbol directory's rows for XOM and MSFT (2026-11-20)
EACH = "Buy $3,000 of XOM and $3,000 of MSFT"
TEN = Portfolio(user="Bill Gates", cash_avail=Decimal("10000"))
EARLIER = ("direct_order_forms.json", "direct_order_decimal_forms.json", "theme_request_forms.json")


def reply(typed: str) -> dict:
    """The real reply to `typed` as a dict (for an edit of one field)."""
    return json.loads(FORMS[typed]["replies"][-1])


def edited(typed: str, number: int | None = None, **changes) -> str:
    """The real reply to `typed` with ONE field changed: `changes` for the form, or `number` for the order of that index (then `changes` are its fields)."""
    form = reply(typed)
    if number is None:
        form.update(changes)
    else:
        form["orders"][number].update(changes)
    return json.dumps(form)


def read(typed: str):
    """The real Account Manager reads the real model's recorded replies to `typed` (the second one answers code's refusal of the first, if there was one)."""
    am = AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk(), profile=BILL_PROFILE)
    return am.understand_request(typed), am


def fields(intent):
    return [(o.side, o.symbol or o.company, o.shares, o.dollars) for o in intent.orders]


def named(typed: str, intent, rows=ROWS) -> wo.WorkOrder:
    """A work order as the desk makes it: the user's orders and the Data Clerk's answer (a `lookup_symbol` order with the real rows of the live session) appended by the Account Manager's own `direct.name_the_stocks`."""
    order = wo.new_work_order(typed, requester="desk", action_type="trade")
    found = wo.new_work_order("Company names and symbols", requester="account_manager", action_type="lookup_symbol", subject=[o.named for o in intent.orders])
    found.data = wo.DataResult(companies=[row for row in rows if row["brand"] in {o.named for o in intent.orders}])
    assert direct.name_the_stocks(order, intent.orders, typed, MOMENT, found) == []
    return order


def decided(typed: str, intent, mode: str = "auto", cash: str = "10000", approver=None, clarify=None, model=None, **options):
    order = named(typed, intent)
    am = AccountManager(model=model or object(), model_id="direct", clerk=clerk(), profile=BILL_PROFILE, **options)
    out = am.decide_direct(order, Portfolio(user="Bill Gates", cash_avail=Decimal(cash)), MOMENT, mode, approver, clarify=clarify, said_how_much=intent.said_how_much(), request_text=typed,
                           balance="even" if intent.allocation == "even" else None, budget=intent.budget)
    return out, am


@pytest.fixture
def each_intent():
    return it.parse_intent(FORMS[EACH]["replies"][-1], EACH)


# ---------------------------------------------------------------- D1: the real model reads the dollars of each stock

def test_the_real_models_readings_of_the_six_requests_of_the_spec():
    got = {typed: read(typed)[0] for typed in ("Buy $3,000 of XOM and $3,000 of MSFT", "Buy $5,000 of MSFT", "Buy $2,000 of AAPL and 10 shares of MSFT", "Put $1,500 into NVDA and $500 into AMD", "Buy $5,000 of MU and AMD", "Buy $2,000 of AAPL and MSFT")}
    assert fields(got[EACH]) == [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)] and got[EACH].budget is None and got[EACH].action == "order" and got[EACH].said_how_much()
    assert got[EACH].describe() == "buy $3,000.00 of XOM, buy $3,000.00 of MSFT (your own order: no recommendation is made and the Critic is not asked)"
    assert fields(got["Buy $5,000 of MSFT"]) == [("buy", "MSFT", None, 5000)] and got["Buy $5,000 of MSFT"].budget is None
    assert fields(got["Buy $2,000 of AAPL and 10 shares of MSFT"]) == [("buy", "AAPL", None, 2000), ("buy", "MSFT", 10, None)] and got["Buy $2,000 of AAPL and 10 shares of MSFT"].said_how_much()
    assert fields(got["Put $1,500 into NVDA and $500 into AMD"]) == [("buy", "NVDA", None, 1500), ("buy", "AMD", None, 500)]
    for typed, total in (("Buy $5,000 of MU and AMD", 5000), ("Buy $2,000 of AAPL and MSFT", 2000)):      # an amount written ONCE for two stocks is ONE total: the budget, and no dollars on the orders
        assert got[typed].budget == total and [o.dollars for o in got[typed].orders] == [None, None] and all(o.shares is None for o in got[typed].orders) and got[typed].said_how_much()
    assert all(FORMS[typed]["outcome"].get("refused") is None for typed in got)


def test_the_requests_of_the_earlier_captures_are_read_as_they_were_with_the_new_prompt_except_three_readings_that_are_the_same_trade():
    """Every request of `direct_order_forms.json`, `direct_order_decimal_forms.json` and `theme_request_forms.json` was asked again with the new prompt. Their intents are the same as the earlier replies gave, apart from three
    that differ and that are listed: two where the model named the stock by its ticker (MSFT) where it had given the company's name (Microsoft Corp), the same stock; and the one request with a dollar amount for a stock, which is now
    that stock's dollars and was the budget."""
    differ = []
    for name in EARLIER:
        for typed, record in json.loads((FIXTURES / name).read_text(encoding="utf-8"))["requests"].items():
            before, after = (_first_reading(typed, record["replies"]), _first_reading(typed, FORMS[typed]["replies"]))
            if before != after:
                differ.append(typed)
    assert differ == ["please purchase 25 shares of Microsoft Corp (MSFT) for me", "Buy $5,000 of MSFT", "Buy 2.25 shares of Microsoft Corp (MSFT) and sell 0.222222 shares of TSLA"]


def _first_reading(typed: str, replies: list):
    for text in replies:
        try:
            return it.parse_intent(text, typed)
        except ValueError as error:
            problem = str(error)
    return problem


# ---------------------------------------------------------------- D2: code checks the number, on a real reply edited by one field

@pytest.mark.parametrize("typed, text, why", [
    (EACH, edited(EACH, 0, dollars=4000), "the amount 4000 is not in what you typed"),                                                    # dollars the user did not write
    (EACH, edited(EACH, 0, shares=5), "never both"),                                                                                     # shares and dollars
    (EACH, edited(EACH, 0, dollars="3000"), "orders must be a list of"),                                                                  # a string
    (EACH, edited(EACH, 0, dollars=True), "orders must be a list of"),                                                                    # a bool
    (EACH, edited(EACH, 0, dollars=3000.456), "dollars"),                                                                                # more than two decimals
    (EACH, edited(EACH, 0, dollars=0), "dollars"),                                                                                       # zero
    (EACH, edited(EACH, 0, dollars=-3000), "dollars"),
    (EACH, edited(EACH, budget=6000), "one total (budget) and the dollars of a stock"),                                                    # a total and the dollars of the orders
    (EACH, edited(EACH, allocation="even"), "an even split"),                                                                            # an even split of amounts that are already given
    # ("Buy 3,000 shares of XOM and 3,000 shares of MSFT" and "Buy 3000 of XOM and 3000 of MSFT" with the model's dollars 3000 were REFUSED here; since entry 138 a number written as shares or as a bare number before the stock is CORRECTED to shares, tests/test_request_numbers.py)
    ("Buy 20% of XOM and 20% of MSFT", edited(EACH, 0, dollars=20), "not as money"),                                                        # a percentage
    ("Buy $5,000 of MU and AMD", edited("Buy $5,000 of MU and AMD", action="order", symbols=[], budget=None, orders=[{"side": "buy", "symbol": "MU", "shares": None, "dollars": 5000}, {"side": "buy", "symbol": "AMD", "shares": None, "dollars": 5000}]),
     "the dollars of 2 orders but you wrote it 1 time"),                                                                                  # $5,000 written once is one total, never $5,000 for each: that would spend $10,000
])
def test_a_reading_whose_dollars_the_user_did_not_write_as_money_is_refused(typed, text, why):
    with pytest.raises(ValueError, match=re.escape(why)):
        it.parse_intent(text, typed)


def test_the_check_reads_numbers_and_never_the_words_around_the_stock():
    assert it.money_written("Buy $3,500 of XOM and 2k of MSFT, 4,000 shares, 20%, 1500 into NVDA, 7 dollars") == {Decimal("3500"), Decimal("7")} and it.money_written("2k of MSFT, 3 million", strict=True) == set()      # AMENDED (entry 138): 2k with no $ and no word dollars is a quantity, not money
    typed = "Put $1,500 into NVDA and $500 into AMD"
    assert fields(it.parse_intent(FORMS[typed]["replies"][-1], typed)) == [("buy", "NVDA", None, 1500), ("buy", "AMD", None, 500)]
    swapped = json.dumps({**reply(typed), "orders": [{"side": "buy", "symbol": "NVDA", "shares": None, "dollars": 500}, {"side": "buy", "symbol": "AMD", "shares": None, "dollars": 1500}]})
    assert fields(it.parse_intent(swapped, typed)) == [("buy", "NVDA", None, 500), ("buy", "AMD", None, 1500)]      # the amounts are in the words: the number is what is checked (no word of the request is read to pair them: the spec's rule)
    other = "Place $1,500 into NVDA, then $500 into AMD"      # the words around the amount and the stock are the model's reading: another way of saying it is read the same
    assert [o.dollars for o in it.parse_intent(FORMS[typed]["replies"][-1], other).orders] == [Decimal("1500"), Decimal("500")]


# ---------------------------------------------------------------- D5: an amount of money in the words and none in the reading is never all the cash

def test_a_reading_with_no_amount_for_a_request_that_wrote_one_is_refused_and_the_model_is_asked_once_more():
    nothing = edited(EACH, orders=[{"side": "buy", "symbol": "XOM", "shares": None, "dollars": None}, {"side": "buy", "symbol": "MSFT", "shares": None, "dollars": None}])      # the reading of the live session of 2026-11-20
    with pytest.raises(ValueError, match=r"you wrote an amount of money \(\$3,000\) but no amount was read"):
        it.parse_intent(nothing, EACH)
    am = AccountManager(model=Replay([nothing, FORMS[EACH]["replies"][-1]]), model_id="recorded", clerk=clerk(), profile=BILL_PROFILE)
    intent = am.understand_request(EACH)
    assert fields(intent) == [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)] and am.intent_run["retries"] == 1 and "no amount was read" in str(am.model.seen[1][-1].content)
    stubborn = AccountManager(model=Replay([nothing, nothing]), model_id="recorded", clerk=clerk(), profile=BILL_PROFILE)
    with pytest.raises(ValueError, match=r"I could not understand the request \(you wrote an amount of money"):
        stubborn.understand_request(EACH)


def test_a_request_with_no_money_words_keeps_the_older_ways_and_a_budget_or_shares_is_an_amount():
    typed = "I want to buy MSFT"
    assert fields(it.parse_intent(FORMS[typed]["replies"][-1], typed)) == [("buy", "MSFT", None, None)]      # nothing was written: Ask mode asks, Auto spends the cash round robin (entries 98 and 106)
    spend = "I want to buy MSFT and TSLA and spend $6,000"
    assert it.parse_intent(FORMS[spend]["replies"][-1], spend).budget == 6000
    shares = "Buy 5 shares of MSFT at under $600"
    one = json.dumps({"action": "order", "topic": None, "symbols": [], "companies": [], "count": None, "allocation": "default", "budget": None, "orders": [{"side": "buy", "symbol": "MSFT", "shares": 5, "dollars": None}]})
    assert fields(it.parse_intent(one, shares)) == [("buy", "MSFT", 5, None)]


def test_the_desk_stops_at_its_first_node_when_the_amount_was_never_read_and_nothing_is_traded(db):
    nothing = edited(EACH, orders=[{"side": "buy", "symbol": "XOM", "shares": None, "dollars": None}, {"side": "buy", "symbol": "MSFT", "shares": None, "dollars": None}])
    account_id = db.account("10000")
    state = desk.run_desk({"ask": {"text": EACH, "default_count": 3, "fetch": False}, "account": {"account": account_id}, "trade": {"mode": "paper", "book": True, "model": False}}, clerk_factory=never("Data Clerk's node"),
                          advisor_factory=never("Advisor"), critic_factory=never("Critic"), funds_factory=lambda: AccountManager(model=Replay([nothing, nothing]), model_id="recorded", clerk=clerk()),
                          with_critic=True, with_trader=True, trader_factory=lambda order: trader(), store_factory=lambda: db.store, confirm=lambda intent: True)
    assert state["order"] is None and state["path"] == ["account_manager"] and state["clarify"] is True and "you wrote an amount of money ($3,000) but no amount was read" in state["errors"][0]
    assert db.store.account_balance(account_id) == Decimal("10000") and db.store.list_portfolio(account_id).positions == []


# ---------------------------------------------------------------- D3 and D4: each stock is sized from its own dollars

def test_three_thousand_dollars_of_each_spends_exactly_three_thousand_dollars_of_each_and_six_thousand_in_all(each_intent, friday_close):
    order, am = decided(EACH, each_intent)
    xom, msft = order.trade_decisions
    prices = {"XOM": close_of("XOM"), "MSFT": close_of("MSFT")}
    for d, entry in zip(order.trade_decisions, order.direct_orders):
        s = d["symbol"]
        assert entry["dollars"] == Decimal("3000") and entry["how_many"] == "dollars" and entry["shares"] == d["shares"] == sh.buys(Decimal("3000"), prices[s])
        assert (d["will_trade"], d["dollars"], d["est_cost"], d["price"], d["size"]["pot"], d["size"]["how"]) == ("yes_user", Decimal("3000"), Decimal("3000.00"), prices[s], Decimal("6000.00"), "dollars")
        assert not sh.whole(d["shares"]) and wo.verify_trade(order, d["record_id"], MOMENT) == []
    am.perform_trades(order, BILL.model_copy(update={"cash_avail": Decimal("10000")}), trader(), MOMENT)
    assert [(t["symbol"], t["status"], t["fill"]["amount"]) for t in order.trades] == [("XOM", "filled", Decimal("3000.00")), ("MSFT", "filled", Decimal("3000.00"))]
    assert order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("4000.00") and wo.verify_trades(order) == [] and wo.pot_reasons(order) == []


def test_filled_at_the_next_real_open_each_stock_still_costs_its_dollars_and_no_more(each_intent, friday_close):
    from datetime import timedelta
    order, am = decided(EACH, each_intent)
    am.perform_trades(order, TEN, trader(mode="backtest"), MOMENT - timedelta(days=1, hours=0))
    assert [t["status"] for t in order.trades] == ["filled", "filled"] and all(t["fill"]["price_source"] == "bar_open" for t in order.trades)
    for t in order.trades:
        assert t["shares"] == sh.buys(Decimal("3000"), price_door(t["fill"]["price"])) and Decimal("2999.99") <= t["fill"]["amount"] <= Decimal("3000.00")
    assert sum((t["fill"]["amount"] for t in order.trades), Decimal(0)) <= Decimal("6000.00") and wo.verify_trades(order) == []


def test_with_fractions_off_each_stock_is_the_whole_shares_its_dollars_pay_for_and_the_pot_is_still_six_thousand(each_intent, friday_close):
    order, am = decided(EACH, each_intent, fractional_even_split=False)
    xom, msft = order.trade_decisions
    assert (xom["shares"], msft["shares"]) == (int(Decimal("3000") // close_of("XOM")), int(Decimal("3000") // close_of("MSFT"))) == (17, 5) and all("dollars" not in d for d in order.trade_decisions)
    assert [d["est_cost"] for d in order.trade_decisions] == [trade_amount(17, close_of("XOM")), trade_amount(5, close_of("MSFT"))] and all(d["est_cost"] <= Decimal("3000") for d in order.trade_decisions)
    assert {d["size"]["pot"] for d in order.trade_decisions} == {Decimal("6000.00")} and wo.verify_trade(order, xom["record_id"], MOMENT) == []
    am.perform_trades(order, TEN, trader(), MOMENT)
    assert sum((t["fill"]["amount"] for t in order.trades), Decimal(0)) <= Decimal("6000") and wo.verify_trades(order) == []


def test_with_less_cash_than_the_two_amounts_the_sizings_own_rule_cuts_and_nothing_passes_the_cash(each_intent, friday_close):
    order, am = decided(EACH, each_intent, cash="4000")
    assert {d["size"]["pot"] for d in order.trade_decisions} == {Decimal("4000.00")} and all(d["est_cost"] <= Decimal("3000") for d in order.trade_decisions)
    assert sum((d["est_cost"] for d in order.trade_decisions), Decimal(0)) <= Decimal("4000")
    am.perform_trades(order, Portfolio(user="Bill Gates", cash_avail=Decimal("4000")), trader(), MOMENT)
    assert sum((t["fill"]["amount"] for t in order.trades), Decimal(0)) <= Decimal("4000") and wo.verify_trades(order) == [] and order.trader_runs[-1]["portfolio_after"]["cash"] >= 0


def test_a_single_stock_with_dollars_is_sized_as_the_budget_was_and_ask_mode_asks_nothing_the_user_already_answered(friday_close):
    typed = "Buy $5,000 of MSFT"
    intent = it.parse_intent(FORMS[typed]["replies"][-1], typed)
    order = wo.new_work_order(typed, requester="desk", action_type="trade")
    am = account(BILL_PROFILE)
    assert am.name_the_stocks(order, intent.orders, typed, MOMENT) == []      # the real recorded directory has MSFT
    cards, channel = [], channel_of([])
    out = am.decide_direct(order, BILL, MOMENT, "ask", lambda card: cards.append(card) or True, clarify=channel, said_how_much=intent.said_how_much(), request_text=typed, budget=intent.budget)
    (d,), (entry,) = out.trade_decisions, out.direct_orders
    price = close_of("MSFT")
    assert channel.cards == [] and len(cards) == 1 and "order       buy $5,000.00: about 9.544" in cards[0]["text"]      # D4: no question about how much; the final proposal is shown (size first, ask last)
    assert (d["shares"], d["dollars"], d["est_cost"], d["size"]["pot"], d["size"]["how"], entry["how_many"], d["will_trade"]) == (sh.buys(Decimal("5000"), price), Decimal("5000"), Decimal("5000"), Decimal("5000"), "dollars", "dollars", "yes_user")
    assert wo.verify_trade(out, d["record_id"], MOMENT) == []


def test_a_sale_in_dollars_is_never_placed_and_says_how_many_shares_to_give(friday_close):
    typed = "Sell $500 of TSLA"
    order = wo.new_work_order(typed, requester="desk", action_type="trade")
    am = account(BILL_PROFILE)
    assert am.name_the_stocks(order, [it.DirectOrder(side="sell", symbol="TSLA", dollars=500)], typed, MOMENT) == []
    out = am.decide_direct(order, BILL, MOMENT, "auto", None, said_how_much=True, request_text=typed)
    (d,) = out.trade_decisions
    assert d["will_trade"] == "no_system_denied" and "say how many to sell" in d["reason"] and "$500.00 is not a number of shares to sell" in d["reason"]


# ---------------------------------------------------------------- D6: a purchase with no number next to a purchase with one

def one_without_a_number() -> tuple[str, "it.TradeIntent"]:
    typed = "Buy $3,000 of XOM and MSFT"
    return typed, it.parse_intent(edited(EACH, 1, dollars=None), typed)


def test_in_auto_mode_the_purchase_with_no_number_is_not_placed_and_the_others_go_ahead(friday_close):
    typed, intent = one_without_a_number()
    assert fields(intent) == [("buy", "XOM", None, 3000), ("buy", "MSFT", None, None)] and not intent.said_how_much()
    order, am = decided(typed, intent, mode="auto", clarify=lambda card: pytest.fail("Auto mode never asks"))
    xom, msft = order.trade_decisions
    assert (xom["will_trade"], xom["dollars"], xom["size"]["pot"]) == ("yes_user", Decimal("3000"), Decimal("3000.00"))
    assert (msft["will_trade"], msft["shares"], msft["dollars"] if "dollars" in msft else None) == ("no_system_denied", 0, None) and "say how much of MSFT" in msft["reason"] and "the cash is not spent on it by default" in msft["reason"]
    assert order.direct_orders[1]["shares"] is None and order.direct_orders[1]["how_many"] is None
    am.perform_trades(order, TEN, trader(), MOMENT)
    assert [(t["symbol"], t["status"], t["fill"]["amount"]) for t in order.trades] == [("XOM", "filled", Decimal("3000.00"))] and order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("7000.00") and wo.verify_trades(order) == []
    assert "NOT PLACED (no price)" in "\n".join(direct.direct_section(order)) and "say how much of MSFT" in "\n".join(direct.direct_section(order))


def test_in_ask_mode_the_account_manager_asks_how_much_of_the_one_with_no_number_and_both_carry_one_pot(friday_close):
    typed, intent = one_without_a_number()
    channel, cards = channel_of(RECORDED["percent_each_no_raise_needed"]["answers"]), []
    order, am = decided(typed, intent, mode="ask", approver=lambda card: cards.append(card) or True, clarify=channel, model=AskReplay(RECORDED["percent_each_no_raise_needed"]["replies"]))
    xom, msft = order.trade_decisions
    assert len(channel.cards) == 1 and "MSFT" in json.dumps(channel.cards[0], default=str) and "XOM" not in json.dumps(channel.cards[0], default=str)      # one set of questions, and only about the purchase with no number
    assert (xom["dollars"], msft["dollars"]) == (Decimal("3000"), Decimal("500")) and order.direct_orders[1]["how_many"] == "asked" and (xom["will_trade"], msft["will_trade"]) == ("yes_user", "yes_user")
    assert {d["size"]["pot"] for d in order.trade_decisions} == {Decimal("10000.00")} and wo.verify_trade(order, xom["record_id"], MOMENT) == [] and wo.verify_trade(order, msft["record_id"], MOMENT) == []      # a run has ONE pot (the sum of the two, never above the cash)
    am.perform_trades(order, TEN, trader(), MOMENT)
    assert [(t["symbol"], t["fill"]["amount"]) for t in order.trades] == [("XOM", Decimal("3000.00")), ("MSFT", Decimal("500.00"))] and wo.verify_trades(order) == [] and len(cards) == 2


def test_in_ask_mode_with_no_answer_or_nobody_to_ask_the_cash_is_not_spent_on_it(friday_close):
    typed, intent = one_without_a_number()
    for clarify, model in ((None, None), (channel_of([]), AskReplay(RECORDED["percent_each_no_raise_needed"]["replies"]))):
        order, _ = decided(typed, intent, mode="ask", approver=lambda card: True, clarify=clarify, model=model)
        xom, msft = order.trade_decisions
        assert xom["will_trade"] == "yes_user" and msft["will_trade"] == "no_system_denied" and "say how much of MSFT" in msft["reason"] and order.direct_orders[1]["shares"] is None


def test_a_number_of_shares_for_another_stock_is_a_number_too(friday_close):
    typed = "Buy 10 shares of XOM, then MSFT"
    intent = it.parse_intent(json.dumps({**reply(EACH), "orders": [{"side": "buy", "symbol": "XOM", "shares": 10, "dollars": None}, {"side": "buy", "symbol": "MSFT", "shares": None, "dollars": None}]}), typed)
    order, _ = decided(typed, intent, mode="auto")
    xom, msft = order.trade_decisions
    assert (xom["will_trade"], xom["shares"], "dollars" in xom) == ("yes_user", 10, False) and msft["will_trade"] == "no_system_denied" and "say how much of MSFT" in msft["reason"]


# ---------------------------------------------------------------- the user's own dollars are the limit of the decision (work_order)

def test_a_decision_that_costs_more_than_the_users_own_dollars_is_refused(each_intent, friday_close):
    order, _ = decided(EACH, each_intent)
    rid = order.trade_decisions[0]["record_id"]
    order.trade_decisions[0]["dollars"] = Decimal("3000.01")
    assert any("the user wrote $3,000.00 for this stock and the decision's dollars are $3,000.01" in r for r in wo.verify_trade(order, rid, MOMENT))
    order.trade_decisions[0]["dollars"] = Decimal("3000")
    off, _ = decided(EACH, each_intent, fractional_even_split=False)
    off.trade_decisions[0]["est_cost"] = Decimal("3000.01")
    assert any("estimated cost are $3,000.01" in r for r in wo.verify_trade(off, off.trade_decisions[0]["record_id"], MOMENT))


def test_the_users_record_holds_dollars_as_money_and_never_with_typed_shares(each_intent):
    order = named(EACH, each_intent)
    assert [e["dollars"] for e in order.direct_orders] == [Decimal("3000"), Decimal("3000")] and wo.verify_direct_orders(order.direct_orders) == []
    for bad, why in ((Decimal("-1"), "dollars is a number above zero"), ("3000", "dollars is a number above zero"), (True, "dollars is a number above zero")):
        broken = copy.deepcopy(order.direct_orders)
        broken[0]["dollars"] = bad
        assert any(why in r for r in wo.verify_direct_orders(broken)), bad
    typed_and_dollars = copy.deepcopy(order.direct_orders)
    typed_and_dollars[0].update(shares=Decimal("5"), how_many="typed")
    assert any("shares or dollars, never both" in r for r in wo.verify_direct_orders(typed_and_dollars))
    assert order.direct_orders[0]["how_many"] is None      # sized later: "dollars"


# ---------------------------------------------------------------- the prompt

def test_the_prompt_tells_the_model_about_dollars_with_made_up_names_only():
    for text in (prompt.SYSTEM, prompt.RETRY):
        assert "dollars" in text
    assert '"dollars": 3000' in prompt.SYSTEM and "ONE total" in prompt.SYSTEM and "never $5,000 for each" in prompt.SYSTEM
    assert not any(real in prompt.SYSTEM for real in ("XOM", "MSFT", "AAPL", "NVDA", "TSLA", "Microsoft", "Exxon", "oil"))


# ---------------------------------------------------------------- the live session of 2026-11-20

def test_the_live_auto_session_spent_six_thousand_dollars_of_ten_thousand_three_thousand_on_each_stock():
    assert LIVE["path"] == ["account_manager", "account_manager_direct", "trader"] and LIVE["request"] == EACH
    intent = it.TradeIntent.model_validate(LIVE["intent"])
    assert fields(intent) == [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)] and intent.budget is None
    order = wo.WorkOrder.model_validate(copy.deepcopy(LIVE["order"]))
    assert order.status == "traded" and order.errors == [] and wo.verify_trades(order) == [] and wo.pot_reasons(order) == [] and wo.verify_direct_orders(order.direct_orders) == []
    assert [money(e["dollars"]) for e in order.direct_orders] == [Decimal("3000.00")] * 2 and {e["how_many"] for e in order.direct_orders} == {"dollars"} and {e["typed_company"] for e in order.direct_orders} == {None}
    decided_at = datetime(2026, 11, 20, 16, 10, tzinfo=timezone.utc)
    for d, t in zip(order.trade_decisions, order.trades):
        assert wo.verify_trade(order, d["record_id"], decided_at) == []
        assert (d["will_trade"], d["origin"], money(d["dollars"]), money(d["est_cost"]), money(d["size"]["pot"]), d["size"]["how"]) == ("yes_user", "direct", Decimal("3000.00"), Decimal("3000.00"), Decimal("6000.00"), "dollars")
        assert sh.quantity(d["shares"]) == sh.buys(Decimal("3000"), price_door(d["price"])) and not sh.whole(d["shares"])      # a capture's numbers are floats after json: read through the doors
        assert (t["status"], t["booked"], money(t["dollars"]), money(t["fill"]["amount"]), t["fill"]["slippage_bps"]) == ("filled", True, Decimal("3000.00"), Decimal("3000.00"), 0)
        assert trade_amount(sh.quantity(t["fill"]["shares"]), price_door(t["fill"]["price"])) == Decimal("3000.00") and f"an order of $3,000.00: {sh.text(t['shares'])} shares" in t["reason"]
    run = order.trader_runs[-1]
    spent = sum((money(t["fill"]["amount"]) for t in order.trades), Decimal(0))
    assert spent == Decimal("6000.00") <= Decimal("6000.00") and money(run["portfolio_before"]["cash"]) == Decimal("10000.00") and money(run["portfolio_after"]["cash"]) == Decimal("4000.00")
    assert money(run["portfolio_before"]["cash"]) - money(run["portfolio_after"]["cash"]) == spent and run["booked"] is True


def test_the_page_shows_the_dollars_the_user_wrote_for_each_stock():
    view = views.work_order_view(LIVE["order"])
    assert [r["detail"]["your_order"].split(" (about")[0] for r in view["rows"]] == ["buy $3,000.00 of XOM", "buy $3,000.00 of MSFT"]
    assert "buy $3,000.00 of Exxonmobil Holdings Corp (XOM) (about 18.616196 shares) at $161.15" in "\n".join(direct.direct_section(wo.WorkOrder.model_validate(copy.deepcopy(LIVE["order"]))))
    assert dumps(wo.WorkOrder.model_validate(copy.deepcopy(LIVE["order"])).direct_orders)      # written through the one encoder: Decimals are JSON numbers
