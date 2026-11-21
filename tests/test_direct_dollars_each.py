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
from datetime import datetime
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
from prometheus.common import work_order as wo
from prometheus.common.money import dumps, money
from prometheus.common.portfolio import Portfolio
from prometheus.langgraph import desk
from prometheus.prompts import account_manager_request as prompt

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
FORMS = json.loads((FIXTURES / "direct_dollars_forms.json").read_text(encoding="utf-8"))["requests"]
LIVE = json.loads((FIXTURES / "auto_direct_dollars_each.json").read_text(encoding="utf-8"))
ROWS = LIVE["order"]["data"]["companies"]       # the real symbol directory's rows for XOM and MSFT (2026-11-20)
EACH = "Buy $3,000 of XOM and $3,000 of MSFT"
TEN = Portfolio(user="Bill Gates", cash_avail=Decimal("10000"))


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


# ---------------------------------------------------------------- D5: an amount of money in the words and none in the reading is never all the cash


def test_the_desk_stops_at_its_first_node_when_the_amount_was_never_read_and_nothing_is_traded(db):
    nothing = edited(EACH, orders=[{"side": "buy", "symbol": "XOM", "shares": None, "dollars": None}, {"side": "buy", "symbol": "MSFT", "shares": None, "dollars": None}])
    account_id = db.account("10000")
    state = desk.run_desk({"ask": {"text": EACH, "default_count": 3, "fetch": False}, "account": {"account": account_id}, "trade": {"mode": "paper", "book": True, "model": False}}, clerk_factory=never("Data Clerk's node"),
                          advisor_factory=never("Advisor"), critic_factory=never("Critic"), funds_factory=lambda: AccountManager(model=Replay([nothing, nothing]), model_id="recorded", clerk=clerk()),
                          with_critic=True, with_trader=True, trader_factory=lambda order: trader(), store_factory=lambda: db.store, confirm=lambda intent: True)
    assert state["order"] is None and state["path"] == ["account_manager"] and state["clarify"] is True and "you wrote an amount of money ($3,000) but no amount was read" in state["errors"][0]
    assert db.store.account_balance(account_id) == Decimal("10000") and db.store.list_portfolio(account_id).positions == []


# ---------------------------------------------------------------- D3 and D4: each stock is sized from its own dollars


def test_with_less_cash_than_the_two_amounts_the_sizings_own_rule_cuts_and_nothing_passes_the_cash(each_intent, friday_close):
    order, am = decided(EACH, each_intent, cash="4000")
    assert {d["size"]["pot"] for d in order.trade_decisions} == {Decimal("4000.00")} and all(d["est_cost"] <= Decimal("3000") for d in order.trade_decisions)
    assert sum((d["est_cost"] for d in order.trade_decisions), Decimal(0)) <= Decimal("4000")
    am.perform_trades(order, Portfolio(user="Bill Gates", cash_avail=Decimal("4000")), trader(), MOMENT)
    assert sum((t["fill"]["amount"] for t in order.trades), Decimal(0)) <= Decimal("4000") and wo.verify_trades(order) == [] and order.trader_runs[-1]["portfolio_after"]["cash"] >= 0


# ---------------------------------------------------------------- D6: a purchase with no number next to a purchase with one

def one_without_a_number() -> tuple[str, "it.TradeIntent"]:
    typed = "Buy $3,000 of XOM and MSFT"
    return typed, it.parse_intent(edited(EACH, 1, dollars=None), typed)


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
