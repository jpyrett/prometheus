"""A DIRECT BUY THE CASH DOES NOT COVER REACHES RAISEFUNDS (design choices entry 142; design/direct_raise_funds_spec.md, rules R1 to R8 and R12).

Jacob's real request on the page, 2026-11-21: "Buy 2 shares of ANT and raise funds if needed", account 6553, ended "NOT PLACED (no funds): the cash $60.14 does not pay for 2 shares of ANT at $88.91 ($177.82)". Nothing in the desk read the words,
asked, or ran RaiseFunds. Here the same request runs through the REAL LangGraph desk over a THROWAWAY Postgres account that is the same as 6553 (its cash and its twelve holdings, `tests/fixtures/raise_funds/account_6553.json`), with the REAL
Finnhub quotes of that day given back to the real Data Clerk, the REAL replies of the Account Manager's model (the request's form, and the thought generator of RaiseFunds, recorded by the content of the question), the real Trader with no model
and the real Postgres book. Only DPZ and MCD show a profit at those quotes, so a plan takes profits there (one share of DPZ, bought at $341.72 and now $348.61, raises the $117.68 that is short).

What the tests pin: the words (R2), the question and its answers in Ask mode (R3), the plan shown and approved (R4), Auto mode (R5), the desk (R6), the order of the trades and the cash (R7), what RaiseFunds cannot pay for (R8), the setting
ALWAYS_RAISE_FUNDS (R12), a basket of purchases paid for by ONE plan that never sells a stock being bought (R10), and that a purchase the cash covers never reaches RaiseFunds.
"""
from contextlib import suppress
from decimal import Decimal

import pytest
import raise_funds_helpers as H
from raise_funds_helpers import ANT, ASK, CASH, COST, PLAIN, SHORT, Cards, Model

from prometheus.agents.account_manager import funding
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.trader import venue
from prometheus.agents.trader.broker import PaperBroker
from prometheus.agents.trader.simulator import Execution
from prometheus.agents.trader.venue import PaperVenue
from prometheus.common import work_order as wo
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import trade_amount
from prometheus.persistence.accounts import SECRETS, open_account_store

SAMPLES = ("jacob_sample", "avery_chen", "blake_okonkwo")
DPZ = Decimal(str(H.QUOTES["DPZ"]["price"]))
SENTENCE = ("You have $60.14 available; 2 ANT at $88.91 cost $177.82, $117.68 short. Shall I raise the money by selling some of your holdings? I will show you what I would sell first.")


@pytest.fixture(scope="module")
def store():
    if not SECRETS.is_file():
        pytest.skip("secrets/postgres_local.txt is not on this machine (run scripts/setup_postgres.sh)")
    try:
        opened = open_account_store()
        opened.ping()
    except Exception as error:
        pytest.skip(f"Postgres is configured but not reachable ({type(error).__name__})")
    return opened


class Book:
    """THROWAWAY accounts the same as 6553, deleted at the end, and a check that the three sample accounts did not move."""

    def __init__(self, store):
        self.store, self.made, self.before = store, [], self.samples(store)

    @staticmethod
    def samples(store):
        return {seed: (store.account_balance(store.account_id_for_seed(seed)), store.list_portfolio(store.account_id_for_seed(seed)).model_dump_json()) for seed in SAMPLES}

    def account(self, mode: str = "ask", **options) -> int:
        return H.make_account(self.store, self.made, mode=mode, **options)

    def cash(self, account_id: int) -> Decimal:
        return self.store.account_balance(account_id)

    def held(self, account_id: int) -> dict[str, Decimal]:
        return {p.symbol: p.shares for p in self.store.list_portfolio(account_id).positions}


@pytest.fixture
def db(store):
    book = Book(store)
    try:
        yield book
    finally:
        for account_id in book.made:
            with suppress(ValueError):
                store.delete_account(account_id)
    assert book.samples(store) == book.before, "a sample account moved"


def bookings(monkeypatch, store) -> list[tuple[str, str, Decimal]]:
    """Every booking the REAL book is asked for from now on, in order: (side, symbol, shares); the call goes through."""
    log: list[tuple[str, str, Decimal]] = []
    real = store.book_trade

    def call(account_id, symbol, side, shares, *args, **kw):
        log.append((side, symbol, shares))
        return real(account_id, symbol, side, shares, *args, **kw)

    monkeypatch.setattr(store, "book_trade", call)
    return log


def calls(monkeypatch) -> list[tuple[str, str]]:
    """The handshake calls of every Account Manager and the Trader's placings, in order, as (kind, trade_id): the real methods are wrapped and call through."""
    log: list[tuple[str, str]] = []
    for name, kind in (("approve_trade", "approve"), ("preview_trade", "preview"), ("record_fill", "record")):
        real = getattr(AccountManager, name)

        def watched(self, trade, *args, _real=real, _kind=kind, **kw):
            log.append((_kind, trade["trade_id"]))
            return _real(self, trade, *args, **kw)

        monkeypatch.setattr(AccountManager, name, watched)
    placed = PaperBroker._place

    def place(broker, trade, *args, **kw):
        log.append(("place", trade.trade_id))
        return placed(broker, trade, *args, **kw)

    monkeypatch.setattr(PaperBroker, "_place", place)
    return log


# ---------------------------------------------------------------- R2 and R4: Ask mode, the words say it

def test_ask_mode_with_the_words_shows_the_plan_and_after_a_yes_sells_the_profit_and_then_buys_with_no_question_about_raising_funds(db, monkeypatch):
    account_id = db.account("ask")
    log, order_of_calls = bookings(monkeypatch, db.store), calls(monkeypatch)
    cards = Cards(funding=True)
    state = H.run_desk(db.store, account_id, ASK, approver=cards)
    order = state["order"]
    assert state["errors"] == [] and state["path"] == ["account_manager", "account_manager_direct", "account_manager_funds", "trader"] and order.status == "traded" and order.errors == []      # R6: the funds node is on the way
    assert cards.kinds == ["funding"]                                                                             # R2: the words are the permission, so the question of R3 is not asked; R4: the plan is shown and approved
    (leg,), (buy,) = cards.cards[0]["legs"], cards.cards[0]["buys"]
    assert (leg["symbol"], leg["shares"], leg["price"], leg["amount"], leg["gain"]) == ("DPZ", 1, DPZ, DPZ, DPZ - Decimal("341.72")) and (buy["symbol"], buy["shares"], buy["price"], buy["cost"]) == ("ANT", 2, ANT, COST)
    assert cards.cards[0]["cash"] == CASH and cards.cards[0]["cash_after"] == CASH + DPZ - COST == Decimal("230.93")      # the cash before and after, as the card shows it
    assert "sell 1 DPZ at $348.61: about $348.61, a gain of $6.89" in cards.cards[0]["text"] and "cash after" in cards.cards[0]["text"]
    decision = order.trade_decisions[0]
    assert (decision["will_trade"], decision["origin"], decision["shares"]) == ("yes_user", "direct", 2) and decision["funding"] == {"shortfall": SHORT, "plan_id": f"{order.request_id}-funds", "by": "words"}   # R7
    sell, buy_trade = order.trades[1], order.trades[0]
    assert (buy_trade["trade_id"], buy_trade["side"], buy_trade["status"], buy_trade["shares"]) == ("T1", "buy", "filled", 2) and (sell["trade_id"], sell["kind"], sell["funds_for"], sell["status"], sell["shares"]) == ("T1-F1", "funding", "T1", "filled", 1)
    assert [(side, symbol) for side, symbol, _ in log] == [("sell", "DPZ"), ("buy", "ANT")]                          # R7: the sale is booked FIRST, and then the purchase
    kinds = [call for call in order_of_calls if call[0] in ("approve", "record")]
    assert kinds == [("approve", "T1-F1"), ("record", "T1-F1"), ("approve", "T1"), ("record", "T1")]              # one trade at a time: the buy's approval meets the cash the sale left (entry 81)
    assert db.cash(account_id) == CASH + DPZ - COST and db.held(account_id)["ANT"] == 2 and db.held(account_id)["DPZ"] == Decimal("3.389558")      # never negative, to the cent
    assert wo.verify_trades(order) == [] and order.trader_runs[-1]["funding"]["approved"] is True and state["funds"]["plan"]["sells"][0]["symbol"] == "DPZ"


def test_a_no_to_the_plan_sells_nothing_and_the_purchase_is_not_placed_because_the_cash_does_not_cover_it_never_half(db):
    account_id = db.account("ask")
    cards = Cards(funding=False)
    state = H.run_desk(db.store, account_id, ASK, approver=cards)
    order = state["order"]
    buy, = order.trades
    assert cards.kinds == ["funding"] and (buy["trade_id"], buy["status"], buy["reject_code"]) == ("T1", "rejected", "insufficient_cash") and "does not pay for one share" in buy["reason"]      # nothing was sold, so the cash alone is asked, and it does not cover even one share
    assert db.cash(account_id) == CASH and "ANT" not in db.held(account_id) and db.held(account_id)["DPZ"] == Decimal("4.389558") and order.trader_runs[-1]["funding"]["approved"] is False


# ---------------------------------------------------------------- R3: Ask mode, no permission in the words

def test_ask_mode_without_the_words_asks_once_in_the_account_managers_own_sentence_and_a_yes_runs_the_plan(db):
    account_id = db.account("ask")
    cards = Cards(raise_funds=True, funding=True)
    state = H.run_desk(db.store, account_id, PLAIN, approver=cards)
    order = state["order"]
    assert cards.kinds == ["raise_funds", "funding"] and cards.cards[0]["text"] == SENTENCE and cards.cards[0]["shortfall"] == SHORT and cards.cards[0]["cash"] == CASH      # the question, then the plan to approve
    assert cards.cards[0]["buys"] == [{"symbol": "ANT", "shares": 2, "price": ANT, "cost": COST}]
    assert order.status == "traded" and order.trade_decisions[0]["funding"]["by"] == "asked" and db.held(account_id)["ANT"] == 2 and db.cash(account_id) == CASH + DPZ - COST


@pytest.mark.parametrize("answer, said", [(False, "you did not want holdings sold to pay for it"), (None, "no answer is a no")])
def test_a_no_or_no_answer_to_the_question_places_nothing_sells_nothing_and_the_reason_says_so(db, answer, said):
    account_id = db.account("ask")
    cards, model = Cards(raise_funds=answer), Model(PLAIN)
    state = H.run_desk(db.store, account_id, PLAIN, approver=cards, model=model)
    order = state["order"]
    decision = order.trade_decisions[0]
    assert cards.kinds == ["raise_funds"] and decision["will_trade"] == "no_funds" and said in decision["reason"] and "placed as you typed it or not at all" in decision["reason"]
    assert state["path"] == ["account_manager", "account_manager_direct"] and order.trades == [] and "funding" not in decision                  # no funds node, no Trader, no RaiseFunds
    assert db.cash(account_id) == CASH and "ANT" not in db.held(account_id) and model.asked == []                                          # RaiseFunds' model was never asked


# ---------------------------------------------------------------- R5: Auto mode asks nothing

def test_auto_mode_with_the_words_sells_and_buys_with_no_question_at_all(db):
    account_id = db.account("auto")
    cards = Cards()
    state = H.run_desk(db.store, account_id, ASK, approver=cards)
    order = state["order"]
    assert cards.cards == [] and order.status == "traded" and state["funds"]["ask_user"] is False                                              # R5: the user's own words are the approval of the sale
    assert [(t["trade_id"], t["status"]) for t in order.trades] == [("T1", "filled"), ("T1-F1", "filled")] and "without asking" in order.trader_runs[-1]["funding"]["note"] and order.trader_runs[-1]["funding"]["approved"] is True
    assert db.cash(account_id) == CASH + DPZ - COST and db.held(account_id)["ANT"] == 2 and wo.verify_trades(order) == []


def test_auto_mode_without_the_words_sells_nothing_places_nothing_and_says_what_to_add(db):
    account_id = db.account("auto")
    cards, model = Cards(), Model(PLAIN)
    state = H.run_desk(db.store, account_id, PLAIN, approver=cards, model=model)
    order = state["order"]
    decision = order.trade_decisions[0]
    assert cards.cards == [] and decision["will_trade"] == "no_funds" and funding.NOT_SAID in decision["reason"] and "say 'and raise funds if needed' to let me sell holdings to pay for it" in decision["reason"]
    assert state["path"] == ["account_manager", "account_manager_direct"] and order.trades == [] and model.asked == []
    assert db.cash(account_id) == CASH and "ANT" not in db.held(account_id) and db.held(account_id)["DPZ"] == Decimal("4.389558")


# ---------------------------------------------------------------- R8: what RaiseFunds cannot pay for

def test_when_no_holding_in_profit_could_pay_for_it_nothing_is_asked_or_sold_and_the_reason_names_the_shortfall_and_what_could_be_raised(db):
    account_id = db.account("ask", drop=("DPZ", "MCD"))                                                          # the only two holdings that show a profit at these quotes
    cards, model = Cards(raise_funds=True, funding=True), Model(ASK)
    state = H.run_desk(db.store, account_id, ASK, approver=cards, model=model)
    order = state["order"]
    decision = order.trade_decisions[0]
    assert cards.cards == [] and decision["will_trade"] == "no_funds" and "$117.68 is short" in decision["reason"] and "$0.00 could be raised from holdings in profit" in decision["reason"]
    assert state["path"] == ["account_manager", "account_manager_direct"] and order.trades == [] and model.asked == [] and db.cash(account_id) == CASH


def test_a_funded_purchase_the_sales_would_not_pay_for_at_the_traders_price_sells_nothing(db, monkeypatch):
    """The plan was made at the recorded quotes; the price the Trader meets is lower for DPZ, so the sale would not raise the $117.68: the Trader's own question to the Account Manager says so BEFORE anything is sold (R7: the
    cash is the only check, never overspent), the sale is `funding_failed` and the purchase is `insufficient_cash`."""
    account_id = db.account("ask")
    real = AccountManager.fund_decisions

    def lower_after_the_plan(self, *args, **kw):
        answer = real(self, *args, **kw)
        H.QUOTES["DPZ"] = {**H.QUOTES["DPZ"], "price": 100.0}
        return answer

    saved = dict(H.QUOTES["DPZ"])
    monkeypatch.setattr(AccountManager, "fund_decisions", lower_after_the_plan)      # a WRAP of the real method: it calls through, and then the price the Trader meets is lower
    try:
        state = H.run_desk(db.store, account_id, ASK, approver=Cards(funding=True))
    finally:
        H.QUOTES["DPZ"] = saved
    order = state["order"]
    assert {t["trade_id"]: (t["status"], t["reject_code"]) for t in order.trades} == {"T1": ("rejected", "insufficient_cash"), "T1-F1": ("rejected", "funding_failed")} and "nothing was sold" in order.trades[0]["reason"]
    assert db.cash(account_id) == CASH and db.held(account_id)["DPZ"] == Decimal("4.389558") and "ANT" not in db.held(account_id)


# ---------------------------------------------------------------- R12: the setting ALWAYS_RAISE_FUNDS

def test_the_setting_is_the_same_as_the_words_in_every_request_ask_mode_shows_the_plan_without_the_question_and_auto_mode_sells(db, monkeypatch):
    monkeypatch.setenv("ALWAYS_RAISE_FUNDS", "true")
    ask_id, auto_id = db.account("ask"), db.account("auto")
    asked = Cards(funding=True)
    state = H.run_desk(db.store, ask_id, PLAIN, approver=asked)
    assert asked.kinds == ["funding"] and state["order"].trade_decisions[0]["funding"]["by"] == "setting" and db.held(ask_id)["ANT"] == 2                              # no question of R3: the plan is shown and approved
    auto = Cards()
    state = H.run_desk(db.store, auto_id, PLAIN, approver=auto)
    assert auto.cards == [] and state["order"].status == "traded" and db.held(auto_id)["ANT"] == 2 and db.cash(auto_id) == CASH + DPZ - COST and state["funds"]["ask_user"] is False


@pytest.mark.parametrize("value, sells", [("true", True), ("TRUE", True), ("1", True), ("yes", True), ("false", False), ("", False), ("maybe", False)])
def test_only_a_true_setting_counts_and_unset_is_false(monkeypatch, value, sells):
    monkeypatch.setenv("ALWAYS_RAISE_FUNDS", value)
    assert funding.always_raise_funds() is sells and (funding.permission(False, PLAIN) == "setting") is sells


def test_a_dont_sell_anything_in_the_words_wins_over_the_setting_for_that_request(db, monkeypatch):
    monkeypatch.setenv("ALWAYS_RAISE_FUNDS", "true")
    typed = "Buy 2 shares of ANT but don't sell anything"
    account_id = db.account("auto")
    cards, model = Cards(), Model(typed)
    state = H.run_desk(db.store, account_id, typed, approver=cards, model=model)
    order = state["order"]
    assert state["plan"]["intent"].raise_funds is False and order.trade_decisions[0]["will_trade"] == "no_funds" and funding.NOT_SAID in order.trade_decisions[0]["reason"] and model.asked == []
    assert db.cash(account_id) == CASH and "ANT" not in db.held(account_id)


# ---------------------------------------------------------------- nothing else changes

def test_a_purchase_the_cash_covers_never_reaches_raisefunds_even_with_the_words(db):
    account_id = db.account("auto")
    db.store.add_funds(account_id, Decimal("500.00"))                                                             # $560.14 now: 2 x $88.91 is covered
    cards, model = Cards(), Model(ASK)
    state = H.run_desk(db.store, account_id, ASK, approver=cards, model=model)
    order = state["order"]
    assert state["path"] == ["account_manager", "account_manager_direct", "trader"] and cards.cards == [] and model.asked == [] and "funding" not in order.trade_decisions[0]
    assert [(t["symbol"], t["side"], t["status"]) for t in order.trades] == [("ANT", "buy", "filled")] and db.cash(account_id) == CASH + 500 - COST


def test_a_sale_is_never_made_to_pay_for_a_purchase_without_the_users_words_or_answer(db):
    """Entry 105: a sale comes only from the user's own order or from RaiseFunds, and RaiseFunds only on the user's word. Three ways to the same request, none of them permitted: no holding is sold."""
    for mode, answer in (("auto", None), ("ask", False), ("ask", None)):
        account_id = db.account(mode)
        H.run_desk(db.store, account_id, PLAIN, approver=Cards(raise_funds=answer), model=Model(PLAIN))
        assert db.held(account_id) == {p["symbol"]: Decimal(str(p["shares"])) for p in H.ACCOUNT["positions"]} and db.cash(account_id) == CASH


# ---------------------------------------------------------------- R10: a basket of purchases, ONE plan

def test_two_purchases_are_paid_for_by_one_plan_that_never_sells_either_and_the_sales_come_first(db, monkeypatch):
    typed = "Buy 2 shares of ANT and 1 share of GOOGL and raise funds if needed"
    account_id = db.account("auto")
    log = bookings(monkeypatch, db.store)
    state = H.run_desk(db.store, account_id, typed, approver=Cards())
    order, funds = state["order"], state["funds"]
    cost = COST + Decimal(str(H.QUOTES["GOOGL"]["price"]))
    assert funds["basket"] == ["ANT", "GOOGL"] and [x["symbol"] for x in funds["also"]] == ["GOOGL"] and funds["status"] == "plan" and funds["need"] == cost - CASH
    sold = {x["symbol"]: x["shares"] for x in funds["plan"]["sells"]}
    assert set(sold) == {"DPZ"} and "ANT" not in sold and "GOOGL" not in sold                                    # never a stock that is being bought in the run
    assert [(t["trade_id"], t["symbol"], t["side"], t["status"]) for t in order.trades] == [("T1", "ANT", "buy", "filled"), ("T1-F1", "DPZ", "sell", "filled"), ("T2", "GOOGL", "buy", "filled")]
    assert [(side, symbol) for side, symbol, _ in log] == [("sell", "DPZ"), ("buy", "ANT"), ("buy", "GOOGL")]     # sales first, then the purchases, one at a time
    assert db.cash(account_id) >= 0 and db.cash(account_id) == CASH + sold["DPZ"] * DPZ - cost and wo.verify_trades(order) == []


# ---------------------------------------------------------------- R1: the dollars of a direct order

def test_an_order_in_dollars_the_cash_does_not_cover_is_funded_and_never_costs_more_than_its_dollars(db):
    typed = "Buy $200 of ANT and raise funds if needed"
    account_id = db.account("auto")
    state = H.run_desk(db.store, account_id, typed, approver=Cards())
    order = state["order"]
    decision = order.trade_decisions[0]
    buy = next(t for t in order.trades if t["trade_id"] == "T1")
    assert decision["will_trade"] == "yes_user" and decision["dollars"] == Decimal("200.00") and decision["funding"]["shortfall"] == Decimal("200.00") - CASH and decision["size"]["pot"] == Decimal("200.00")
    assert buy["status"] == "filled" and buy["fill"]["amount"] <= Decimal("200.00") and buy["fill"]["shares"] > 2 and db.cash(account_id) >= 0 and wo.verify_trades(order) == []      # its dollars buy a fraction of a share too (entry 111, M3)


# ---------------------------------------------------------------- entry 143, S1 to S3: the user's OWN SALE in the same request

MCD = Decimal(str(H.QUOTES["MCD"]["price"]))
SELL_AND_BUY = "Sell 1 share of MCD and buy 3 shares of ANT"
SELL_AND_BUY_FUNDED = "Sell 1 share of MCD and buy 5 shares of ANT and raise funds if needed"
SELL_AND_BUY_ASK = "Sell 1 share of MCD and buy 5 shares of ANT"
COST5 = trade_amount(5, ANT)
SHORT5 = COST5 - CASH - MCD                                                                                         # what the purchase costs less the cash AND less what the user's own sale brings


def test_the_users_own_sale_pays_for_the_purchase_of_the_same_request_and_nothing_is_raised(db, monkeypatch):
    """"Sell 1 share of MCD and buy 3 shares of ANT": the cash ($60.14) alone does not pay for 3 x $88.91, the sale of one MCD ($273.24) placed first does. No RaiseFunds, no question, no funds node."""
    account_id = db.account("auto")
    log, model = bookings(monkeypatch, db.store), Model(SELL_AND_BUY)
    state = H.run_desk(db.store, account_id, SELL_AND_BUY, approver=Cards(), model=model)
    order = state["order"]
    sale, buy = order.trade_decisions
    assert state["path"] == ["account_manager", "account_manager_direct", "trader"] and order.status == "traded" and order.errors == [] and model.asked == []
    assert (sale["will_trade"], buy["will_trade"], buy["shares"]) == ("yes_user", "yes_user", 3) and "funding" not in buy and "from your own sale (placed first) pays for it" in buy["reason"]
    assert [(t["trade_id"], t["symbol"], t["side"], t["status"]) for t in order.trades] == [("T1", "MCD", "sell", "filled"), ("T2", "ANT", "buy", "filled")]
    assert [(side, symbol) for side, symbol, _ in log] == [("sell", "MCD"), ("buy", "ANT")] and db.cash(account_id) == CASH + MCD - trade_amount(3, ANT) and wo.verify_trades(order) == []


def test_the_shortfall_is_what_the_purchase_costs_less_the_cash_and_less_the_sale_and_raisefunds_never_sells_the_share_the_user_sold(db, monkeypatch):
    account_id = db.account("auto")
    log = bookings(monkeypatch, db.store)
    state = H.run_desk(db.store, account_id, SELL_AND_BUY_FUNDED, approver=Cards())
    order, funds = state["order"], state["funds"]
    _, buy = order.trade_decisions
    assert SHORT5 == Decimal("111.17") and buy["funding"]["shortfall"] == SHORT5 and funds["need"] == SHORT5 and funds["status"] == "plan"      # not $384.41: the sale is counted
    sold = {x["symbol"]: x["shares"] for x in funds["plan"]["sells"]}
    assert sold == {"DPZ": 1} and "MCD" not in sold                                                                # one profit taken; the plan works from the account once MCD's sale is filled (2 left, 1 is kept)
    assert [(t["trade_id"], t["symbol"], t["side"], t["status"]) for t in order.trades] == [("T1", "MCD", "sell", "filled"), ("T2", "ANT", "buy", "filled"), ("T2-F1", "DPZ", "sell", "filled")]
    assert [(side, symbol) for side, symbol, _ in log] == [("sell", "MCD"), ("sell", "DPZ"), ("buy", "ANT")]       # the user's sale first, then the funding sale, then the purchase
    assert db.cash(account_id) == CASH + MCD + DPZ - COST5 and db.held(account_id)["MCD"] == 2 and db.held(account_id)["ANT"] == 5 and wo.verify_trades(order) == []


def test_ask_mode_says_the_users_own_sale_in_the_question_and_in_the_plan_it_shows(db):
    account_id = db.account("ask")
    cards = Cards(raise_funds=True, funding=True)
    state = H.run_desk(db.store, account_id, SELL_AND_BUY_ASK, approver=cards)
    order = state["order"]
    ask, plan = cards.cards[0], cards.cards[1]
    assert cards.kinds == ["raise_funds", "funding"] and ask["cash"] == CASH and ask["shortfall"] == SHORT5 and ask["sales_proceeds"] == MCD
    assert [(s["symbol"], s["shares"], s["amount"]) for s in ask["sales"]] == [("MCD", 1, MCD)]
    assert ask["text"] == ("You have $60.14 available and your own sale of 1 MCD brings about $273.24; 5 ANT at $88.91 cost $444.55, $111.17 short. Shall I raise the money by selling some of your holdings? "
                           "I will show you what I would sell first.")
    assert [(o["symbol"], o["shares"], o["amount"]) for o in plan["own_sales"]] == [("MCD", 1, MCD)] and plan["own_sales_proceeds"] == MCD and "your own sale of 1 MCD" in plan["text"]
    assert plan["cash"] == CASH and plan["cash_after"] == CASH + MCD + DPZ - COST5                                  # the cash after counts the user's sale, the funding sale and the purchase
    assert order.status == "traded" and db.cash(account_id) == CASH + MCD + DPZ - COST5


def test_a_sale_the_account_cannot_make_brings_nothing_and_the_purchase_is_not_paid_for_by_it(db):
    """"Sell 5 shares of MCD and buy 3 shares of ANT": the account holds 3 MCD. The sale is refused, so its proceeds are not counted and the purchase meets the cash alone."""
    account_id = db.account("auto")
    state = H.run_desk(db.store, account_id, "Sell 5 shares of MCD and buy 3 shares of ANT", approver=Cards())
    order = state["order"]
    sale, buy = order.trade_decisions
    assert (sale["will_trade"], buy["will_trade"]) == ("no_funds", "no_funds") and "holds 3 shares of MCD" in sale["reason"] and "own sale" not in buy["reason"] and buy["reason"].startswith("the cash $60.14 does not pay")
    assert order.trades == [] and db.cash(account_id) == CASH and db.held(account_id)["MCD"] == 3


# ---------------------------------------------------------------- entry 143, B1: a sale fills LOWER than planned: the sales stay, the purchase is refused

class LowSaleVenue(PaperVenue):
    """The paper venue, except that every SALE is reported filled at the day's LOW of the recorded Finnhub quote (`low` of `tests/fixtures/raise_funds/account_6553.json`) instead of the quote it was estimated at: what a live broker may
    do and the paper broker cannot. A purchase is reported as estimated. The prices are the real recorded ones; nothing is invented."""

    def report_fill(self, trade, shares, estimate, prices):
        if trade.side != "sell":
            return estimate
        low = price_door(Decimal(str(H.QUOTES[trade.symbol]["low"])))
        return Execution("filled", f"sold at the day's low {low}", price=low, price_source="quote")


FOUR = "Buy 4 shares of ANT and raise funds if needed"
POOR = Decimal("8.00")                                                                                              # the account's own cash: $347.64 short of 4 ANT ($355.64), which one DPZ at the quote ($348.61) pays for and at the day's low ($346.955) does not
DPZ_LOW = price_door(Decimal(str(H.QUOTES["DPZ"]["low"])))


def test_a_sale_that_fills_lower_than_planned_stays_booked_and_the_purchase_it_was_to_pay_for_is_refused_never_overspent(db, monkeypatch):
    monkeypatch.setitem(venue.SIMULATED, "paper", LowSaleVenue)
    account_id = db.account("auto", cash=POOR)
    log = bookings(monkeypatch, db.store)
    state = H.run_desk(db.store, account_id, FOUR, approver=Cards())
    order = state["order"]
    buy, sale = order.trades
    assert (buy["trade_id"], buy["status"], buy["reject_code"]) == ("T1", "rejected", "sales_fell_short") and (sale["trade_id"], sale["kind"], sale["status"], sale["fill"]["price"]) == ("T1-F1", "funding", "filled", DPZ_LOW)
    assert "the sales brought $346.96, not the about $348.61 planned (1 DPZ at 346.9550 (planned 348.6100))" in buy["reason"] and "The sales stay: they were approved, filled and booked" in buy["reason"]
    assert [(side, symbol) for side, symbol, _ in log] == [("sell", "DPZ")]                                          # the sale is booked at what it REALLY brought; the purchase never reached the book
    assert db.cash(account_id) == POOR + trade_amount(1, DPZ_LOW) == Decimal("354.96") and "ANT" not in db.held(account_id) and db.held(account_id)["DPZ"] == Decimal("3.389558")      # never negative, never overspent
    assert order.status == "traded" and wo.verify_trades(order) == [] and any("The sales stay" in text for text in order.errors)                    # an outcome of the run, said to the user, and not an output-check failure
    assert "The sales stay (they were booked): T1-F1 sell 1 DPZ at 346.9550" in order.trader_runs[-1]["summary"]


def test_a_sale_that_fills_lower_but_still_pays_for_the_purchase_is_said_and_the_purchase_is_placed_in_full(db, monkeypatch):
    monkeypatch.setitem(venue.SIMULATED, "paper", LowSaleVenue)
    account_id = db.account("auto")
    state = H.run_desk(db.store, account_id, ASK, approver=Cards())                                                  # "Buy 2 shares of ANT and raise funds if needed": $117.68 short, one DPZ pays for it at either price
    order = state["order"]
    assert [(t["trade_id"], t["status"], t["fill"]["price"]) for t in order.trades if t["kind"] == "funding"] == [("T1-F1", "filled", DPZ_LOW)] and order.trades[0]["status"] == "filled" and order.trades[0]["shares"] == 2
    assert any("it was still enough to pay for the purchase" in text for text in order.errors) and db.cash(account_id) == CASH + trade_amount(1, DPZ_LOW) - COST and wo.verify_trades(order) == []


def test_the_output_check_of_the_whole_run_still_finds_sales_booked_for_a_purchase_that_failed_for_any_other_reason(db, monkeypatch):
    """Only `sales_fell_short` is the explained case: a filled funding sale whose purchase is `rejected` for another reason is still the worst case of spec section 7 and is still reported."""
    monkeypatch.setitem(venue.SIMULATED, "paper", LowSaleVenue)
    account_id = db.account("auto", cash=POOR)
    order = H.run_desk(db.store, account_id, FOUR, approver=Cards())["order"]
    assert wo.verify_trades(order) == []
    order.trades[0]["reject_code"] = "internal_error"
    assert any("is a filled funding sale but the purchase it pays for, T1, is rejected, not filled" in finding for finding in wo.verify_trades(order))
