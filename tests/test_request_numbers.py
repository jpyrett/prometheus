"""DOLLARS OR SHARES, A PRICE LIMIT AND "SKIP RECOMMENDATIONS" (design choices entry 138; design/request_numbers_spec.md; Jacob, 2026-11-20).

REAL DATA ONLY, as `test_direct_order.py`. The Account Manager's model is a REPLAY of its REAL replies, recorded on 2026-11-20 with the prompt of entry 138 (`tests/fixtures/session_real/request_numbers_forms.json`, made by
`scripts/capture_request_numbers_forms.py`): the first fifteen requests are the texts of the spec (Jacob's own "Buy $2500 worth of symbol KO and skip recommendations." among them), the last six are the same kinds of request for MSFT (and TSLA),
the stocks the recorded symbol directory and the stored closes of the end to end tests have. The real code checks every field of those replies against the words the user typed. The Data Clerk is the real one over the recorded Finnhub
directory, the prices are the stored closes of Friday 2026-11-13, the Trader is the real Trader with no model, the Advisor is the real Advisor on the real captures of `tests/fixtures/advisor_real/` (its model a replay of its real replies),
and the work order of the price-limit tests is the real session `tests/fixtures/session_real/bob_jones_oil.json`. A form that a test writes itself ("a model that said ...") has the shape of the real replies and is for a FAULT: a reading
the code must correct or refuse. Nothing else stands in for an agent, a model or a market number.
"""
import copy
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from advisor_helpers import MemoryLedger, Scripted, scored
from handshake_helpers import BOB, BUDGET, MOMENT, PROFILE, close_of, friday_close, manager, trader  # noqa: F401  (friday_close is a fixture)
from test_account_manager_clarify import channel_of
from test_advisor_sndk_pg import FIXTURES as ADVISOR_FIXTURES
from test_advisor_sndk_pg import RecordedClerk
from test_direct_order import BILL, BILL_PROFILE, Replay, clerk, never
from test_neutral_recommendation import neutral_only

from prometheus.agents.account_manager import decisions, direct, handshake, neutral
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.advisor.recommend import within_price_limit
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import dumps
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile
from prometheus.langgraph import desk
from prometheus.prompts import account_manager_request as prompt

pytestmark = pytest.mark.usefixtures("friday_close")

FIXTURES = Path(__file__).parent / "fixtures"
FORMS = json.loads((FIXTURES / "session_real" / "request_numbers_forms.json").read_text(encoding="utf-8"))["requests"]
CAPTURE = FIXTURES / "session_real" / "bob_jones_oil.json"
KO = "Buy $2500 worth of symbol KO and skip recommendations."
MU_10 = "I want to buy 10 of MU."
MU_1000 = "I want to buy $1000 of MU."
MU_LIMIT = "Buy $1000 of MU under $50 a share"
MSFT_SKIP = "Buy $2500 worth of symbol MSFT and skip recommendations."
MSFT_10 = "I want to buy 10 of MSFT."
MSFT_UNDER_50 = "Buy 10 of MSFT under $50 a share"
MSFT_UNDER_1000 = "Buy 10 of MSFT under $1000 a share"
MSFT_SPEND = "spend 1000 on MSFT"
ASK = UserProfile(user="Bill Gates", risk_tolerance="moderate", age=70, trade_mode="ask")
AUTO = UserProfile(user="Bill Gates", risk_tolerance="moderate", age=70, trade_mode="auto")


def real(typed: str, which: int = -1) -> str:
    return FORMS[typed]["replies"][which]


def read(typed: str) -> it.TradeIntent:
    """What the code makes of the model's real reply (its last one, when code refused the first)."""
    return it.parse_intent(real(typed), typed)


def form(**changes) -> str:
    """A form of the shape the real model answers in, with the defaults, changed: for a FAULT (a reading the code must correct or refuse)."""
    return json.dumps({"action": "order", "topic": None, "symbols": [], "companies": [], "count": None, "allocation": "default", "budget": None, "each": False, "max_price": None, "orders": [], **changes})


def order_of(symbol="MU", shares=None, dollars=None) -> dict:
    return {"side": "buy", "symbol": symbol, "shares": shares, "dollars": dollars}


def fields(intent: it.TradeIntent) -> list[tuple]:
    return [(o.side, o.symbol or o.company, o.shares, o.dollars) for o in intent.orders]


# ---------------------------------------------------------------- the recordings are what they say they are

def test_the_recordings_are_the_real_models_replies_to_the_spec_s_texts_and_the_msft_ones():
    assert len(FORMS) == 33 and all(r["replies"] and all(isinstance(x, str) for x in r["replies"]) for r in FORMS.values())      # AMENDED (entry 146): 21, the eight of entry 140 (see "the price floor" below) and the four of entry 146 (`test_one_company_shares.py`)
    assert {KO, MU_10, MU_1000, MU_LIMIT, MSFT_SKIP, MSFT_10, MSFT_UNDER_50, MSFT_UNDER_1000, MSFT_SPEND} <= set(FORMS)
    assert json.loads(real(KO))["orders"] == [order_of("KO", None, 2500)] and json.loads(real(MU_10))["orders"] == [order_of("MU", 10, None)]


# ---------------------------------------------------------------- S1: a recommend cue the user NEGATED is no cue

@pytest.mark.parametrize("typed, asks", [
    (KO, False),                                                                  # Jacob's own request: the live bug of 2026-11-20
    ("I want to buy MSFT, no research needed", False),
    ("Buy KO without the advisor", False),
    ("don't need a recommendation, buy $500 of KO", False),
    ("Skipping the analysis, buy 10 shares of KO", False),
    ("forget the recommendations and buy KO", False),
    ("Buy KO. Do not want any advice.", False),
    ("Buy 10 of MU if the recommendation passes", True),                          # the trade DEPENDS on it
    ("Buy KO if the recommendations pass", True),
    ("I have no cash but recommend three stocks", True),                          # the clause ends at "but": nothing is negated
    ("No, recommend three stocks", True),                                         # a comma ends it too
    ("Buy the 3 most recommended oil stocks", True),
    ("should I buy KO", True),
    ("skip recommendations and research AI stocks", True),                        # the second cue is not negated
    ("Buy $5,000 of KO", False),                                                  # no cue at all
])
def test_a_negated_recommend_cue_asks_for_nothing_and_every_other_cue_still_does(typed, asks):
    assert it.asks_for_recommendation(typed) is asks


def test_the_words_that_negate_and_the_words_that_are_skipped_are_closed_lists():
    assert it.skips_recommendations(KO) and it.skips_recommendations("buy AI stocks, skip recommendations") and not it.skips_recommendations("Buy 10 of MU if the recommendation passes") and not it.skips_recommendations("Buy 10 of MU")
    assert not it.NO_RECOMMEND.search("Buy a recommendation of mine") and not it.NO_RECOMMEND.search("I know research is hard")      # "no" is a whole word


# ---------------------------------------------------------------- S2: "skip recommendations" is a direct order

def test_the_real_reply_to_jacobs_request_is_a_direct_order_of_the_dollars_he_wrote_and_nothing_is_to_be_recommended():
    intent = read(KO)
    assert intent.action == "order" and fields(intent) == [("buy", "KO", None, Decimal("2500"))] and intent.budget is None and intent.said_how_much() and intent.corrected == []
    with pytest.raises(ValueError, match="not a search for companies"):
        intent.as_request()
    assert intent.describe() == "buy $2,500.00 of KO (your own order: no recommendation is made and the Critic is not asked)"


def test_a_model_that_read_it_as_buy_with_a_total_makes_the_same_direct_order_and_the_money_is_not_lost():
    """The reading of the live session: the model said "buy" (KO, no budget) and the desk went to the Advisor. Whatever the model says, the words ask for no recommendation and name the stock: a direct order, with the total."""
    intent = it.parse_intent(form(action="buy", symbols=["KO"], budget=2500), KO)
    assert intent.action == "order" and [o.symbol for o in intent.orders] == ["KO"] and intent.budget == Decimal("2500") and intent.said_how_much()


def test_a_request_that_does_ask_for_a_recommendation_keeps_the_dollars_of_its_order_as_the_budget_they_were_once_lost():
    typed = "Buy $2500 worth of KO if the recommendation passes"
    intent = it.parse_intent(form(orders=[order_of("KO", None, 2500)]), typed)
    assert intent.action == "buy" and intent.symbols == ["KO"] and intent.orders == [] and intent.budget == Decimal("2500")       # the old conversion dropped the dollars: the Account Manager would have spent ALL the cash
    two = "Buy $2500 of KO and $1000 of PEP if the recommendation passes"
    with pytest.raises(it.NeedsClarification, match="ONE total"):
        it.parse_intent(form(orders=[order_of("KO", None, 2500), order_of("PEP", None, 1000)]), two)
    shares = it.parse_intent(form(orders=[order_of("MU", 10, None)]), "Buy 10 of MU if the recommendation passes")
    assert shares.action == "buy" and shares.symbols == ["MU"] and shares.shares_of == {"MU": Decimal(10)} and shares.corrected == [] and "10 shares of MU" in shares.describe()      # AMENDED (entry 140, C1): the 10 shares are kept (they were dropped with a note: "cannot be combined with a recommendation yet")


def test_the_real_reply_to_a_request_that_made_the_trade_depend_on_the_recommendation_is_still_a_request_for_it():
    intent = read("Buy 10 of MU if the recommendation passes")
    assert intent.action == "buy" and intent.symbols == ["MU"] and intent.orders == []


def run_desk(typed, profile, portfolio=BILL, clarify=None, approver=None, confirm=lambda intent: True, factories=True):
    """The REAL desk over the real Account Manager (the replay of the model's real reply), the real Data Clerk over the recorded directory, the stored closes, the real Trader with no model and no database. The Data Clerk's node, the
    Advisor and the Critic fail the test if they are built: a direct order must not reach them."""
    request = {"ask": {"text": typed, "default_count": 3, "fetch": False}, "account": {"portfolio": portfolio, "profile": profile, "mode": profile.trade_mode}, "trade": {"mode": "paper", "book": False, "model": False}}
    return desk.run_desk(request, clerk_factory=never("Data Clerk's node"), advisor_factory=never("Advisor"), critic_factory=never("Critic"),
                         funds_factory=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()), with_critic=True, with_trader=True,
                         trader_factory=lambda order: trader(), confirm=confirm, approver=approver, clarify=clarify)


def test_the_desk_takes_skip_recommendations_straight_to_the_account_managers_decision_and_the_trader_and_buys_the_dollars():
    state = run_desk(MSFT_SKIP, AUTO)
    order = state["order"]
    assert state["errors"] == [] and state["path"] == ["account_manager", "account_manager_direct", "trader"] and order.status == "traded"
    assert order.recommendations == [] and order.reviews == [] and order.critic_runs == [] and order.advisor_runs == [] and state["plan"]["direct"] is True
    (decision,), (trade,) = order.trade_decisions, order.trades
    assert decision["origin"] == "direct" and decision["will_trade"] == "yes_user" and decision["dollars"] == Decimal("2500.00") and trade["status"] == "filled"
    assert 0 < trade["fill"]["amount"] <= Decimal("2500.00") and wo.verify_trades(order) == []        # an order in dollars never costs more than its dollars


# ---------------------------------------------------------------- S3: "skip recommendations" with no stock to buy

def test_a_request_that_skips_the_recommendations_and_names_no_stock_cannot_be_served_and_the_model_is_asked_only_once():
    typed = "buy AI stocks, skip recommendations"
    model = Replay(FORMS[typed]["replies"])                      # one reply: a second call fails the test
    am = AccountManager(model=model, model_id="recorded", clerk=clerk())
    with pytest.raises(it.NeedsClarification) as caught:
        am.understand_request(typed)
    assert str(caught.value) == it.SKIP_NEEDS_STOCKS and "Name the stocks to buy" in str(caught.value) and "let me recommend some" in str(caught.value) and len(model.seen) == 1


def test_the_desk_stops_at_its_first_node_with_the_message_and_asks_to_clarify():
    typed = "buy AI stocks, skip recommendations"
    request = {"ask": {"text": typed, "default_count": 3, "fetch": False}, "account": {"portfolio": BILL, "profile": AUTO, "mode": "auto"}}
    state = desk.run_desk(request, clerk_factory=never("Data Clerk's node"), advisor_factory=never("Advisor"), critic_factory=never("Critic"),
                          funds_factory=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()), with_critic=True)
    assert state["order"] is None and state["plan"] is None and state["clarify"] is True and state["errors"] == [it.SKIP_NEEDS_STOCKS] and state["path"] == ["account_manager"]


def test_skipping_the_recommendations_of_a_count_of_stocks_is_the_same_request_that_cannot_be_served():
    with pytest.raises(it.NeedsClarification):
        it.parse_intent(form(action="buy", symbols=["KO", "PEP", "MCD"], count=2), "Buy the best 2 of KO, PEP and MCD and skip recommendations")


# ---------------------------------------------------------------- D1: money is a number with a $ or the word dollars

@pytest.mark.parametrize("typed, amounts", [
    ("I want to buy $1000 of MU.", [1000]), ("buy 1000 bucks of AMD", [1000]), ("Buy 1000 dollars of MU", [1000]), ("buy 1 buck of MU", [1]), ("Buy $5,000 of MU", [5000]), ("buy 500 usd of MU", [500]),
    ("Buy $5k of MU", [5000]), ("Buy 5 thousand dollars of MU", [5000]), ("spend 2.5 million dollars", [2_500_000]), ("$3,000 of XOM and $2,000 of MSFT", [3000, 2000]),
    ("I have 10000 to spend", [10000]), ("I have 10000 in cash", [10000]),          # the money is said to be money: Jacob's own request of 2026-11-15
    ("I want to buy 10 of MU.", []), ("Buy 1000 MU", []), ("Buy 10 shares of MU", []), ("Buy 5k of MU", []), ("buy 5 thousand of MU", []), ("buy 2 million MU", []),
    ("Use 20% of my cash", []), ("spend 1000 on MU", []), ("Buy 3 AI stocks", []),
])
def test_a_number_is_money_only_with_a_dollar_sign_or_the_word_dollars(typed, amounts):
    assert it.money_amounts(typed) == [Decimal(a) for a in amounts]


def test_a_unit_alone_is_not_money_but_after_a_dollar_sign_or_a_word_dollars_it_is():
    assert it.money_written("2k of MSFT") == set() and it.money_written("$2k of MSFT") == {Decimal(2000)} and it.money_written("2k dollars of MSFT") == {Decimal(2000)}


# ---------------------------------------------------------------- D2: a bare number before a stock is shares

@pytest.mark.parametrize("typed, shares, dollars", [
    (MU_10, 10, None), ("Buy 1000 MU", 1000, None), (MU_1000, None, 1000), ("Buy 1000 dollars of MU", None, 1000), ("buy 500 bucks of AMD", None, 500), (KO, None, 2500),
    (MSFT_10, 10, None), (MSFT_SKIP, None, 2500),
])
def test_the_real_models_reading_of_a_number_is_the_kind_the_user_wrote_it_in(typed, shares, dollars):
    intent = read(typed)
    assert intent.action == "order" and [(o.shares, o.dollars) for o in intent.orders] == [(None if shares is None else Decimal(shares), None if dollars is None else Decimal(dollars))] and intent.budget is None and intent.count is None
    assert intent.said_how_much() and intent.corrected == []


def test_the_desk_buys_10_shares_for_i_want_to_buy_10_of_a_stock_end_to_end():
    """The request of Jacob's rule, end to end on the real replay, the real directory, the stored close and the real Trader: 10 SHARES (no dollars, no budget, no recommendation), placed as typed."""
    state = run_desk(MSFT_10, AUTO)
    order = state["order"]
    assert state["errors"] == [] and state["path"] == ["account_manager", "account_manager_direct", "trader"] and order.recommendations == [] and order.status == "traded"
    (decision,), (trade,) = order.trade_decisions, order.trades
    assert decision["shares"] == 10 and "dollars" not in decision and decision["will_trade"] == "yes_user" and trade["status"] == "filled" and trade["fill"]["shares"] == 10 and order.direct_orders[0]["how_many"] == "typed"


def test_a_number_of_shares_before_a_stock_is_never_read_with_the_words_of_money_in_between():
    assert [n for n, _ in it._order_phrases("buy 10 of MU", "MU")] == [Decimal(10)] and [n for n, _ in it._order_phrases("buy 10 shares of MU", "MU")] == [Decimal(10)]
    assert [n for n, _ in it._order_phrases("Buy 1000 MU", "MU")] == [Decimal(1000)] and [n for n, _ in it._order_phrases("Buy 50 shares of Microsoft Corp (MSFT)", "MSFT")] == [Decimal(50)]
    for money in ("buy 500 bucks of AMD", "Buy 1000 dollars of MU", "buy 500 usd of AMD", "buy 5 thousand of MU", "buy 2 million of MU"):
        assert it._order_phrases(money, "AMD" if "AMD" in money else "MU") == [], money


def test_the_real_reply_to_a_bare_number_with_a_k_is_refused_not_guessed():
    """"Buy 5k of MU": the model said 5000 shares, twice. 5k is no money (no $) and no number of shares (it was not written as a number of shares), so nothing is guessed and the user is told how to write it."""
    typed = "Buy 5k of MU"
    with pytest.raises(ValueError, match="5,000 shares of MU is not what you typed: say it as a number of shares"):
        it.parse_intent(real(typed), typed)
    am = AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk())
    with pytest.raises(ValueError, match="I could not understand the request"):
        am.understand_request(typed)


def test_a_percentage_stays_a_percentage():
    assert it.money_amounts("Buy MU with 20% of my cash") == [] and "20%" not in [str(a) for a in it.numbers_in("20 percent")]


# ---------------------------------------------------------------- D3: a bare number before stocks, companies, names, picks is the COUNT

def test_a_number_before_stocks_or_after_top_is_a_count_and_the_number_before_a_single_stock_is_not():
    assert it.count_written("Buy the top 3 AI stocks under $50 a share") == {Decimal(3)} and it.count_written("get me 2 oil companies") == {Decimal(2)} and it.count_written("the five best picks") == {Decimal(5)}
    assert it.count_written("I want to buy 10 of MU.") == set() and it.count_written("Buy 3 most recommended oil stocks") == {Decimal(3)}
    intent = read("Buy the top 3 AI stocks under $50 a share")
    assert (intent.action, intent.topic, intent.count, intent.budget) == ("buy", "AI", 3, None)


# ---------------------------------------------------------------- D4: the code's kind wins, and says so

@pytest.mark.parametrize("name, typed, text, expected, note", [
    ("dollars that were a bare number", MU_10, form(orders=[order_of("MU", None, 10)]), [("buy", "MU", Decimal(10), None)], "so it is 10 shares, not dollars"),
    ("a budget that was a bare number", MU_10, form(action="buy", symbols=["MU"], budget=10), [("buy", "MU", Decimal(10), None)], "so it is 10 shares of MU, not a budget"),
    ("a count that was a bare number", MU_10, form(action="buy", symbols=["MU"], count=10), [("buy", "MU", Decimal(10), None)], "so it is 10 shares of MU, not a number of companies"),
    ("a budget on an order with no number", MU_10, form(budget=10, orders=[order_of("MU")]), [("buy", "MU", Decimal(10), None)], "not a budget"),
    ("shares that were money", MU_1000, form(orders=[order_of("MU", 1000, None)]), [("buy", "MU", None, Decimal(1000))], "is money (it has a $ or the word dollars), so it is dollars, not shares of MU"),
    ("shares that were money with a word", "Buy 1000 dollars of MU", form(orders=[order_of("MU", 1000, None)]), [("buy", "MU", None, Decimal(1000))], "so it is dollars"),
])
def test_a_number_the_model_put_in_the_wrong_kind_is_corrected_in_code_and_logged_on_the_intent(name, typed, text, expected, note):
    intent = it.parse_intent(text, typed)
    assert intent.action == "order" and fields(intent) == expected and intent.budget is None and intent.count is None, name
    assert len(intent.corrected) == 1 and note in intent.corrected[0] and intent.describe().count("(note: ") == 1


@pytest.mark.parametrize("typed", ["Buy 3,000 shares of XOM and 3,000 shares of MSFT", "Buy 3000 of XOM and 3000 of MSFT"])
def test_the_dollars_the_model_read_for_numbers_written_as_shares_are_shares_not_a_refusal_any_more(typed):
    """These two were REFUSED ("not as money") before entry 138; Jacob's rule makes them what they say: 3000 shares of each."""
    text = form(symbols=["XOM", "MSFT"], orders=[order_of("XOM", None, 3000), order_of("MSFT", None, 3000)])
    intent = it.parse_intent(text, typed)
    assert fields(intent) == [("buy", "XOM", Decimal(3000), None), ("buy", "MSFT", Decimal(3000), None)] and len(intent.corrected) == 2


def test_dollars_that_were_written_as_dollars_are_never_turned_into_shares():
    typed = "Buy $3,000 of XOM and 3000 shares of MSFT"
    intent = it.parse_intent(form(orders=[order_of("XOM", None, 3000), order_of("MSFT", 3000, None)]), typed)
    assert fields(intent) == [("buy", "XOM", None, Decimal(3000)), ("buy", "MSFT", Decimal(3000), None)] and intent.corrected == []


def test_a_bare_number_is_never_a_budget_the_recommendation_path_refuses_it_with_the_way_to_write_it():
    with pytest.raises(ValueError, match="in what you typed, but not as money"):
        it.parse_intent(form(action="buy", topic="oil", count=3, budget=5000), "Buy the 3 best oil stocks with 5000")
    with pytest.raises(ValueError, match="in what you typed, but not as money"):
        it.parse_intent(form(action="buy", topic="oil", count=2, allocation="even", budget=5000), "Buy oil, two of them, 5k, divide it evenly")      # 5k alone is no money
    assert it.parse_intent(form(action="buy", topic="oil", count=2, allocation="even", budget=5000), "Buy oil, two of them, $5k, divide it evenly").budget == Decimal(5000)
    with pytest.raises(ValueError, match="you typed 10 shares of MU: put the number in shares, not null"):
        it.parse_intent(form(budget=10, orders=[order_of("MU"), order_of("AMD")]), "I want to buy 10 of MU and AMD")      # a number before one of two stocks is the shares of that one, never a budget of $10


def test_the_budget_of_an_order_is_checked_against_the_words_too():
    with pytest.raises(ValueError, match="is not in what you typed"):
        it.parse_intent(form(budget=777, orders=[order_of("MSFT")]), "I want to buy MSFT")      # a budget from nowhere was never refused for an order
    assert it.parse_intent(form(budget=6000, orders=[order_of("MSFT"), order_of("TSLA")]), "I want to buy MSFT and TSLA and spend $6,000").budget == Decimal(6000)


# ---------------------------------------------------------------- D5: a bare number after a money verb is asked about

@pytest.mark.parametrize("typed, found", [
    ("spend 1000 on MU", [1000]), ("Invest 1000 in MU", [1000]), ("put 1000 into MU", [1000]), ("I want to spend about 5k on MU", [5000]), ("allocate 2.5 million to MU", [2_500_000]),
    ("spend $1000 on MU", []), ("spend 1000 dollars on MU", []), ("spend 1000 bucks on MU", []), ("invest 10 shares in MU", []), ("spend 20% on MU", []), ("I have 10000 to spend", []),
    ("buy 10 of MU", []), ("invest in the top 3 AI stocks", []), ("put 3 stocks into my watchlist", []),
])
def test_the_numbers_after_a_money_verb_with_no_dollar_sign_are_the_ones_asked_about(typed, found):
    assert [v for v, _ in it.ambiguous_numbers(typed)] == [Decimal(f) for f in found]


def test_the_real_reply_to_spend_1000_on_a_stock_leaves_the_number_to_be_asked_about_and_is_never_1000_shares():
    typed = MSFT_SPEND
    intent = it.parse_intent(real(typed), typed)
    assert intent.action == "order" and fields(intent) == [("buy", "MSFT", None, None)] and intent.unclear_number == Decimal(1000) and not intent.said_how_much()
    assert "still to settle: is 1,000 shares or dollars?" in intent.describe()


@pytest.mark.parametrize("order", [order_of("MU", 1000, None), order_of("MU", None, 1000), order_of("MU")])
def test_whatever_the_model_read_for_that_number_it_is_taken_out_of_the_reading(order):
    intent = it.parse_intent(form(orders=[order]), "spend 1000 on MU")
    assert fields(intent) == [("buy", "MU", None, None)] and intent.unclear_number == Decimal(1000)


def test_a_bare_number_with_the_money_verb_after_it_is_not_asked_about_it_is_money_when_it_is_said_to_be():
    intent = it.parse_intent(form(action="buy", symbols=["MU"], budget=10000), "I have 10000 to spend. Buy as much MU as I can.")
    assert intent.unclear_number is None and intent.budget == Decimal(10000)


@pytest.mark.parametrize("answer, unit", [("shares", "shares"), ("1000 shares", "shares"), ("Shares please", "shares"), ("dollars", "dollars"), ("$1,000", "dollars"), ("1000 dollars", "dollars"), ("money", "dollars"),
                                          ("both", None), ("shares or dollars", None), ("500 shares", None), ("hm", None), ("", None), (None, None), ("$500", None)])
def test_the_users_answer_is_read_by_code(answer, unit):
    assert it.read_unit_answer(answer, Decimal(1000)) == unit


def test_the_number_becomes_the_dollars_of_the_one_stock_the_shares_of_it_or_a_total():
    intent = it.parse_intent(form(orders=[order_of("MU")]), "spend 1000 on MU")
    dollars, shares = it.settle_unit(intent, "dollars"), it.settle_unit(intent, "shares")
    assert fields(dollars) == [("buy", "MU", None, Decimal(1000))] and dollars.unclear_number is None and dollars.corrected == ["1,000 was settled with you as dollars"]
    assert fields(shares) == [("buy", "MU", Decimal(1000), None)] and shares.unclear_number is None
    two = it.parse_intent(form(orders=[order_of("MU"), order_of("AMD")]), "spend 1000 on MU and AMD")
    assert it.settle_unit(two, "dollars").budget == Decimal(1000) and [o.dollars for o in it.settle_unit(two, "dollars").orders] == [None, None]
    with pytest.raises(ValueError, match="goes with ONE stock you named"):
        it.settle_unit(two, "shares")
    theme = it.parse_intent(form(action="buy", topic="oil", count=2), "invest 1000 in 2 oil stocks")
    assert theme.unclear_number == Decimal(1000) and it.settle_unit(theme, "dollars").budget == Decimal(1000)
    with pytest.raises(ValueError, match="goes with ONE stock you named"):
        it.settle_unit(theme, "shares")


def settled(answers, profile=ASK, typed="spend 1000 on MU", channel=True):
    am = AccountManager(model=object(), model_id="none", clerk=clerk(), profile=profile, screen=lambda *lines: None)
    intent = it.parse_intent(form(orders=[order_of("MU")]), typed)
    ask = channel_of(answers) if channel else None
    return am, intent, ask


def test_in_ask_mode_the_user_is_asked_once_and_code_reads_the_answer():
    am, intent, ask = settled(["dollars"])
    out = am.settle_unclear_number(intent, "ask", ask)
    assert fields(out) == [("buy", "MU", None, Decimal(1000))] and out.unclear_number is None and len(ask.cards) == 1
    card = ask.cards[0]
    assert card["kind"] == "how_much" and "1,000" in card["question"] and "no $" in card["question"] and "$1,000.00" in card["question"] and "shares or" in card["question"]


def test_an_answer_that_names_neither_is_asked_once_more_and_then_the_run_stops():
    am, intent, ask = settled(["hm", "shares"])
    assert fields(am.settle_unclear_number(intent, "ask", ask)) == [("buy", "MU", Decimal(1000), None)] and len(ask.cards) == 2
    am, intent, ask = settled(["hm", "whatever"])
    with pytest.raises(it.NeedsClarification, match="the answer did not say which"):
        am.settle_unclear_number(intent, "ask", ask)
    am, intent, ask = settled([])
    with pytest.raises(it.NeedsClarification):
        am.settle_unclear_number(intent, "ask", ask)      # nobody answered


def test_in_auto_mode_nothing_is_asked_and_nothing_is_bought_on_a_guess():
    am, intent, ask = settled(["dollars"], profile=AUTO)
    with pytest.raises(it.NeedsClarification, match="Auto mode asks nothing") as caught:
        am.settle_unclear_number(intent, "auto", ask)
    assert ask.cards == [] and "$1,000.00" in str(caught.value) and "1,000 shares" in str(caught.value)
    am, intent, _ = settled([], channel=False)
    with pytest.raises(it.NeedsClarification):
        am.settle_unclear_number(intent, "ask", None)      # Ask mode with nobody to ask is a stop too
    nothing = it.parse_intent(form(orders=[order_of("MU", 10)]), "buy 10 of MU")
    assert am.settle_unclear_number(nothing, "auto", None) is nothing      # an intent with no such number is returned as it is


def test_the_desk_in_ask_mode_asks_which_and_buys_the_dollars_the_user_said_and_in_auto_mode_stops_with_the_message():
    cards, channel = [], channel_of(["dollars"])
    state = run_desk(MSFT_SPEND, ASK, clarify=channel, approver=lambda card: cards.append(card) or True)
    order = state["order"]
    assert state["errors"] == [] and state["path"] == ["account_manager", "account_manager_direct", "trader"] and len(channel.cards) == 1
    (decision,), (trade,) = order.trade_decisions, order.trades
    assert decision["dollars"] == Decimal("1000.00") and trade["status"] == "filled" and 0 < trade["fill"]["amount"] <= Decimal("1000.00") and wo.verify_trades(order) == []
    auto = run_desk(MSFT_SPEND, AUTO, clarify=lambda card: pytest.fail("Auto mode asks nothing"))
    assert auto["order"] is None and auto["plan"] is None and auto["clarify"] is True and "Auto mode asks nothing" in auto["errors"][0] and auto["path"] == ["account_manager"]


def test_the_desk_in_ask_mode_takes_the_answer_shares_as_the_shares_typed_and_places_it_as_typed_or_not_at_all():
    state = run_desk(MSFT_SPEND, ASK, clarify=channel_of(["shares"]), approver=lambda card: pytest.fail("a ticker the user typed is not asked about"))
    order = state["order"]
    (decision,) = order.trade_decisions
    assert order.direct_orders[0]["shares"] == 1000 and order.direct_orders[0]["how_many"] == "typed" and "dollars" not in order.direct_orders[0]      # 1000 shares of a $523.88 stock: the real cash ($20,000) does not pay for it
    assert decision["will_trade"] == "no_funds" and "does not pay for 1,000 shares of MSFT" in decision["reason"] and "as you typed it or not at all" in decision["reason"] and order.trades == [] and state["path"] == ["account_manager", "account_manager_direct"]


# ---------------------------------------------------------------- D6: the model's prompt

def test_the_prompt_says_it_with_jacobs_two_examples_and_names_no_real_stock():
    system = prompt.SYSTEM
    assert "I want to buy $1000 of ABCD." in system and "I want to buy 10 of ABCD." in system and "10 shares of the stock ABCD" in system and "$1000 worth of the stock ABCD" in system
    assert "dollar sign" in system and "dollars or bucks" in system and "WITHOUT these is a quantity" in system and '"max_price": null' in system and "SKIP the recommendations" in system and "ONE SHARE" in system
    assert not any(real_name in system for real_name in ("MU", "KO", "AMD", "MSFT", "Micron", "XOM")) and "max_price" in prompt.RETRY and "dollar sign" in prompt.RETRY


# ---------------------------------------------------------------- L1 and L2: the price limit

@pytest.mark.parametrize("typed, kinds", [
    ("Buy $1000 of MU under $50 a share", [("money", 1000), ("ceiling", 50)]),
    ("Buy the top 3 AI stocks under $50 a share", [("ceiling", 50)]),
    ("Buy stocks below $20", [("ceiling", 20)]), ("buy anything priced under $100", [("ceiling", 100)]), ("Recommend 3 oil stocks below $40", [("ceiling", 40)]), ("buy shares cheaper than $30", [("ceiling", 30)]),
    ("buy shares less than $30", [("ceiling", 30)]), ("Buy 10 shares of MU at $50 a share", [("ceiling", 50)]), ("Buy at $50 per share", [("ceiling", 50)]), ("each share $50", [("money", 50)]),
    ("Buy at $50/share", [("ceiling", 50)]), ("buy at $50 a pop", [("ceiling", 50)]), ("buy at 50 dollars a share", [("ceiling", 50)]), ("no more than $50 a share", [("ceiling", 50)]),
    ("Buy $5,000 of MU, at most $5000", [("money", 5000), ("money", 5000)]), ("spend no more than $5,000", [("money", 5000)]),                  # a cap on the money has no share beside it: it is an amount to spend
    ("Buy under $1000 of MU", [("money", 1000)]), ("buy less than $5000 worth of MU", [("money", 5000)]), ("buy MU under $1000 in total", [("money", 1000)]),   # followed by of, worth, in: an amount to spend
    ("Buy 10 shares of MU over $20 a share", [("floor", 20)]), ("stocks above $20 a share", [("floor", 20)]), ("at least $20 a share", [("floor", 20)]), ("buy more than $5000 of MU", [("money", 5000)]),
])
def test_a_dollar_amount_is_a_price_limit_when_it_is_tied_to_one_share_or_after_under_or_below(typed, kinds):
    assert [(w.kind, w.value) for w in it.written_amounts(typed)] == [(k, Decimal(v)) for k, v in kinds]


def test_a_price_limit_is_never_money_to_spend():
    typed = "Buy $1000 of MU under $50 a share"
    assert it.money_amounts(typed) == [Decimal(1000)] and it.price_limits(typed) == [Decimal(50)] and it.money_written("Buy the top 3 AI stocks under $50 a share") == set()
    assert it.parse_intent(real("Buy the top 3 AI stocks under $50 a share"), "Buy the top 3 AI stocks under $50 a share").said_how_much() is False      # the $50 is no answer to "how much?"
    assert it.TradeIntent(request="x", action="buy", topic="AI", max_price=Decimal(50)).said_how_much() is False


def test_the_real_reply_to_buy_1000_of_a_stock_under_50_a_share_is_dollars_on_the_order_and_a_limit_of_50():
    intent = read(MU_LIMIT)
    assert intent.action == "order" and fields(intent) == [("buy", "MU", None, Decimal(1000))] and intent.budget is None and intent.max_price == Decimal(50) and [o.max_price for o in intent.orders] == [Decimal(50)] and intent.corrected == []
    assert type(intent.max_price) is Decimal and intent.describe() == "buy $1,000.00 of MU, only while a share costs $50.00 or less (your own order: no recommendation is made and the Critic is not asked)"


def test_the_real_reply_to_a_theme_with_a_limit_carries_it_with_the_count_and_the_theme():
    typed = "Buy the top 3 AI stocks under $50 a share"
    intent = read(typed)
    assert (intent.action, intent.topic, intent.count, intent.max_price, intent.budget) == ("buy", "AI", 3, Decimal(50), None) and intent.describe().endswith("only while a share costs $50.00 or less")
    assert json.loads(dumps(intent.model_dump()))["max_price"] == 50 and it.TradeIntent.model_validate(intent.model_dump()).max_price == Decimal(50)      # a number in JSON, read back


def test_the_real_replies_to_recommend_3_oil_stocks_below_40_are_a_look_with_a_limit_not_a_purchase():
    """The model read "Recommend 3 oil stocks" as a purchase, twice, and code refused it ("buying was not asked for"). A recommend cue and no buying word is a request to LOOK: a program decides it from the words."""
    typed = "Recommend 3 oil stocks below $40"
    intent = it.parse_intent(real(typed, 0), typed)
    assert (intent.action, intent.topic, intent.count, intent.max_price) == ("research", "oil", 3, Decimal(40)) and "nothing will be bought" in intent.corrected[0] and intent.describe().startswith("look at (nothing will be bought)")
    assert read("Recommend 3 oil stocks below $40").action == "research"


def test_a_minimum_price_is_a_floor_it_is_read_from_the_words_and_kept_on_the_order():
    """AMENDED (entry 140, F1): this was "a minimum price is not built, it is ignored and the user is told in one line". The model's real reply (recorded before the floor existed: max_price null) is read by the words: $20 a share is the floor."""
    typed = "Buy 10 shares of MU over $20 a share"
    intent = read(typed)
    assert intent.max_price is None and intent.min_price == Decimal(20) and [o.min_price for o in intent.orders] == [Decimal(20)] and fields(intent) == [("buy", "MU", Decimal(10), None)]
    assert intent.corrected == ["$20 a share is your minimum price (the least one share may cost), not an amount to spend"] and "only while a share costs $20.00 or more" in intent.describe() and it.money_amounts(typed) == []


def test_a_limit_the_model_missed_is_taken_from_the_words_and_one_it_made_up_is_refused():
    missed = it.parse_intent(form(orders=[order_of("MU", None, 1000)]), MU_LIMIT)
    assert missed.max_price == Decimal(50) and "your price limit" in missed.corrected[0] and [o.max_price for o in missed.orders] == [Decimal(50)]
    wrong = it.parse_intent(form(orders=[order_of("MU", None, 1000)], max_price=40), MU_LIMIT)
    assert wrong.max_price == Decimal(50)      # the words win
    with pytest.raises(ValueError, match="is not a price limit that is in what you typed"):
        it.parse_intent(form(orders=[order_of("MU", 10)], max_price=50), "Buy 10 of MU")
    with pytest.raises(ValueError, match="max_price must be a number"):
        it.parse_intent(form(orders=[order_of("MU", 10)], max_price="50"), "Buy 10 of MU under $50 a share")
    with pytest.raises(ValueError, match="at most 4 decimals"):
        it.parse_intent(form(orders=[order_of("MU", 10)]), "Buy 10 of MU under $50.12345 a share")


def test_the_limit_the_model_read_as_the_money_to_spend_is_taken_back_and_two_limits_are_one_too_many():
    intent = it.parse_intent(form(orders=[order_of("MU", None, 50)]), "Buy MU under $50 a share")
    assert fields(intent) == [("buy", "MU", None, None)] and intent.max_price == Decimal(50) and any("price per share (your limit)" in c for c in intent.corrected)
    theme = it.parse_intent(form(action="buy", topic="oil", count=2, budget=50), "Buy 2 oil stocks under $50 a share")
    assert theme.budget is None and theme.max_price == Decimal(50)
    with pytest.raises(it.NeedsClarification, match="more than one price limit"):
        it.parse_intent(form(action="buy", topic="oil", count=2), "Buy 2 oil stocks under $50 a share and 2 gas ones under $30 a share")


# ---------------------------------------------------------------- L3: the Advisor drops what costs more, in code, before it ranks

def replay(name: str, limit):
    """The real Advisor step on the real captured inputs of `tests/fixtures/advisor_real/<name>.json` (its model a replay of its real replies), with the user's price limit on the order."""
    fx = json.loads((ADVISOR_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    order = wo.WorkOrder.model_validate(fx["order"])
    order.max_price = limit
    advisor = Advisor(model=Scripted(*fx["replies"]), clerk=RecordedClerk(fx["metrics"]), ledger=MemoryLedger())
    advisor.generate_recommendations(order, "funds", date.fromisoformat(fx["today"]))
    return order, fx


def test_a_candidate_priced_above_the_limit_is_left_out_with_its_reason_and_one_under_it_is_recommended_as_before():
    sndk, fx = replay("SNDK", None)
    assert [r["symbol"] for r in sndk.recommendations] == ["SNDK"] and fx["metrics"]["SNDK"]["last_close"]["value"] == 806.43
    over, _ = replay("SNDK", Decimal("500"))
    assert over.recommendations == [] and over.status == "recommended" and over.errors == []
    (left,) = [c for c in over.candidates if c["symbol"] == "SNDK"]
    assert left["reason"] == "price $806.43 is above your limit of $500.00 a share" and left["rank"] is not None
    under, _ = replay("SNDK", Decimal("806.43"))      # AT the limit is under it
    assert [r["symbol"] for r in under.recommendations] == ["SNDK"] and under.recommendations[0]["trade"]["entry_mid"] == 806.43
    pg, pg_fx = replay("PG_open", Decimal("500"))
    assert [r["symbol"] for r in pg.recommendations] == ["PG"] and pg_fx["metrics"]["PG"]["last_close"]["value"] < 500 < 806.43      # prices on both sides of one limit, two real captures


def test_the_number_asked_for_is_filled_from_those_under_the_limit():
    """The step is run on the real MSFT, AMD and MU of the real session of `auto_ai_quoted_theme.json` (the Advisor's real records): with a limit between their prices only the ones under it are eligible, in their own order, and `top_k` is filled from them."""
    capture = json.loads((FIXTURES / "session_real" / "auto_ai_quoted_theme.json").read_text(encoding="utf-8"))["order"]
    ranked = []
    for record in capture["recommendations"]:
        metrics = {name: {**value, "window": "test", "inputs_summary": {}, "note": None} for name, value in record["features"].items()}
        ranked.append(scored(symbol=record["symbol"], metrics=metrics, rank=False))
    assert {s.symbol: s.metrics["last_close"]["value"] for s in ranked} == {"MSFT": 523.88, "AMD": 534.06, "MU": 778.31}
    why: dict = {}
    kept = within_price_limit(ranked, {}, Decimal("600"), why)
    assert [s.symbol for s in kept] == ["MSFT", "AMD"] and why == {"MU": {"reason": "price $778.31 is above your limit of $600.00 a share"}}
    why = {}
    assert [s.symbol for s in within_price_limit(ranked, {}, Decimal("530"), why)] == ["MSFT"] and set(why) == {"AMD", "MU"}
    assert within_price_limit(ranked, {}, None, {}) is ranked      # no limit: the same list
    why = {}
    assert within_price_limit(ranked[:1], {"MSFT": {"price": 540.0}}, Decimal("530"), why) == [] and why["MSFT"]["reason"] == "price $540.00 is above your limit of $530.00 a share"      # the quote of the order is the price the trade plan uses, before the stored close
    assert [s.symbol for s in within_price_limit(ranked[:1], {"MSFT": {"price": 520.0}}, Decimal("530"), {})] == ["MSFT"]


def test_a_candidate_with_no_price_at_all_cannot_be_shown_to_be_under_the_limit_and_is_left_out():
    nothing = scored(symbol="ZZZ", metrics={"last_close": {"value": None, "unit": "usd", "as_of": "2026-10-23", "window": "test", "source": "test", "inputs_summary": {}, "note": "needs bars"}}, rank=False)
    why: dict = {}
    assert within_price_limit([nothing], {}, Decimal(50), why) == [] and why["ZZZ"]["reason"].startswith("no price to check against your limit of $50.00")


# ---------------------------------------------------------------- the desk: the limit goes on the work order, where the Advisor reads it

def test_the_desks_first_node_puts_the_limit_on_the_work_order_and_in_the_plan_for_the_accounts_check():
    typed = "Buy the top 3 AI stocks under $50 a share"
    state = None
    request = {"ask": {"text": typed, "default_count": 3, "fetch": False}, "account": {"portfolio": BILL, "profile": AUTO, "mode": "auto"}}
    graph = desk.build_desk(clerk_factory=never("Data Clerk's node"), advisor_factory=never("Advisor"), critic_factory=never("Critic"), with_critic=True,
                            funds_factory=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()))
    for state in graph.stream({"request": request, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}, "with_trader": False, "plan": None}, stream_mode="values"):
        if state["path"]:
            break
    assert state["path"] == ["account_manager"] and state["order"].max_price == Decimal(50) and state["plan"]["max_price"] == Decimal(50) and state["plan"]["find"] == 3
    assert json.loads(state["order"].model_dump_json())["max_price"] == 50 and wo.WorkOrder.model_validate_json(state["order"].model_dump_json()).max_price == Decimal(50)
    older = json.loads(wo.new_work_order("x").model_dump_json())
    older.pop("max_price")
    assert wo.new_work_order("x").max_price is None and wo.WorkOrder.model_validate(older).max_price is None      # an order with no limit, and an older saved order that has no such key


# ---------------------------------------------------------------- L4: the Account Manager checks it again, and the handshake refuses a fill above it

def reviewed() -> wo.WorkOrder:
    """The real session of `bob_jones_oil.json` as it was when the Critic had finished: XOM $170.59 and EOG $158.32 approved, APA denied."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(json.loads(CAPTURE.read_text(encoding="utf-8"))["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def limited(limit):
    """That session decided again with the user's price limit."""
    order = reviewed()
    return order, decisions.decide_trades(order, BOB, PROFILE, None, MOMENT, "auto", "even", BUDGET, max_price=limit)


def test_an_approved_purchase_priced_above_the_limit_gets_no_decision_and_the_others_carry_the_limit():
    unlimited = limited(None)[1]
    assert [d["symbol"] for d in unlimited.trade_decisions] == ["XOM", "APA", "EOG"] and all("max_price" not in d for d in unlimited.trade_decisions)
    order = limited(Decimal("160"))[1]
    assert [(d["symbol"], d["will_trade"]) for d in order.trade_decisions] == [("APA", "no_system_denied"), ("EOG", "yes_auto")]      # XOM ($170.59) has none; APA was denied by the Critic, not by the limit
    assert order.trade_decisions[1]["max_price"] == Decimal("160.0000") and "max_price" not in order.trade_decisions[0]
    assert order.trade_decisions[1]["size"]["pot"] == BUDGET and order.trade_decisions[1]["shares"] > 0
    assert [p.symbol for p in decisions.purchases_to_size(reviewed(), BOB, MOMENT, Decimal("160"))] == ["EOG"] and [p.symbol for p in decisions.purchases_to_size(reviewed(), BOB, MOMENT)] == ["XOM", "EOG"]
    assert decisions.has_purchase_to_size(reviewed(), MOMENT, Decimal("100")) is False and decisions.has_purchase_to_size(reviewed(), MOMENT, Decimal("160")) is True and decisions.has_purchase_to_size(reviewed(), MOMENT) is True
    none = limited(Decimal("100"))[1]
    assert [(d["symbol"], d["will_trade"]) for d in none.trade_decisions] == [("APA", "no_system_denied")] and none.status == "decided"      # nothing is left to size or to trade
    at = limited(Decimal("170.59"))[1]
    assert [d["symbol"] for d in at.trade_decisions] == ["XOM", "APA", "EOG"]      # AT the limit is under it


def test_the_account_manager_tells_the_user_which_purchase_has_no_decision_and_why():
    said = []
    order = reviewed()
    order.data = wo.DataResult()
    am = manager(screen=said.append)
    am.decide_trades(order, BOB, None, MOMENT, "auto", "even", BUDGET, max_price=Decimal("160"))
    assert any("XOM is at $170.59 a share, above your limit of $160.00: it gets no decision and is not bought" in line for line in said)
    assert order.data.notes == [decisions.OVER_LIMIT + "XOM is at $170.59 a share, above your limit of $160.00: it gets no decision and is not bought"] and [d["symbol"] for d in order.trade_decisions] == ["APA", "EOG"]


@pytest.fixture
def allowed(monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")


def test_the_neutral_companies_are_priced_first_so_the_number_asked_for_is_filled_from_those_under_the_limit(allowed):
    """The real neutral session of `pam_smith_neutral.json`: MU ($778.31) and AMD ($534.06) were scored neutral. With a limit of $600 MU is left out, with a line for the user, and AMD is the one decided."""
    said = []
    order = neutral_only()
    am = AccountManager(model=object(), model_id="neutral", clerk=clerk(), profile=UserProfile(user="Pam Smith", risk_tolerance="aggressive", age=30, trade_mode="auto"), fractional_even_split=False, screen=said.append)
    out = am.decide_trades(order, Portfolio(user="Pam Smith", cash_avail=Decimal("50000")), None, MOMENT, "auto", count=1, max_price=Decimal("600"))
    assert [(d["symbol"], d["will_trade"], d["origin"]) for d in out.trade_decisions] == [("AMD", "yes_auto", "neutral")] and out.trade_decisions[0]["max_price"] == Decimal("600")
    assert any(line.strip().startswith("MU is at $778.31 a share, above your limit of $600.00") for line in said)
    best = neutral.neutral_candidates(neutral_only())[0]["symbol"]
    again = AccountManager(model=object(), model_id="neutral", clerk=clerk(), profile=am.profile, fractional_even_split=False, screen=lambda *lines: None).decide_trades(neutral_only(), Portfolio(user="Pam Smith", cash_avail=Decimal("50000")), None, MOMENT, "auto", count=1)
    assert best == "MU" and [d["symbol"] for d in again.trade_decisions] == ["MU"]      # with no limit the one company asked for is the best scored of the two, MU: the limit is what puts AMD in its place


def trade(symbol="XOM", shares=8, side="buy"):
    return {"symbol": symbol, "side": side, "shares": shares, "shares_approved": shares}


def test_the_handshake_refuses_a_purchase_whose_fill_price_is_above_the_limit_and_never_a_sale_or_a_price_under_it():
    held = BOB.model_copy(update={"cash_avail": Decimal("15000.00")})
    price = close_of("XOM")
    refused = handshake.decide_approval(held, trade(), price, MOMENT.date(), max_price=Decimal("160"))
    assert not refused.approved and refused.code == "not_permitted" and refused.shares == 0 and "above your limit of $160.00 a share" in refused.reason and "nothing is left resting" in refused.reason
    assert handshake.decide_approval(held, trade(), price, MOMENT.date(), max_price=price).approved and handshake.decide_approval(held, trade(), price, MOMENT.date(), max_price=Decimal("200")).shares == 8
    assert handshake.decide_approval(held, trade(), price, MOMENT.date()).approved      # no limit
    assert handshake.decide_approval(held, trade(side="sell"), price, MOMENT.date(), max_price=Decimal("1")).code == "not_held"      # a sale is not about the limit: the portfolio holds no XOM here


def test_the_price_moved_since_the_decision_the_real_traders_purchase_is_rejected_at_the_handshake_and_nothing_is_bought_above_the_limit():
    """The user's limit is $200 when the Account Manager decides (EOG $158.32 and XOM $170.59 are under it); the decision of XOM then says $100, as if the price had moved above its limit since: the REAL Trader asks for the approval,
    the Account Manager refuses it (`not_permitted`), the trade is rejected and shown, EOG is bought, and the account holds no XOM."""
    order = limited(Decimal("200"))[1]
    xom = next(d for d in order.trade_decisions if d["symbol"] == "XOM")
    xom["max_price"] = Decimal("100")
    am = manager()
    am.perform_trades(order, BOB, trader(), MOMENT)
    by_symbol = {t["symbol"]: t for t in order.trades}
    assert by_symbol["XOM"]["status"] == "rejected" and by_symbol["XOM"]["reject_code"] == "not_permitted" and "above your limit of $100.00 a share" in by_symbol["XOM"]["reason"]
    assert by_symbol["EOG"]["status"] == "filled" and wo.verify_trades(order) == []
    assert [p["symbol"] for p in order.trader_runs[-1]["portfolio_after"]["positions"]] == ["EOG"]


def test_a_fill_above_the_limit_is_an_output_defect_the_traders_own_check_names():
    order = limited(Decimal("200"))[1]
    am = manager()
    am.perform_trades(order, BOB, trader(), MOMENT)
    assert wo.verify_trades(order) == []
    filled = next(t for t in order.trades if t["symbol"] == "EOG" and t["status"] == "filled")
    next(d for d in order.trade_decisions if d["symbol"] == "EOG")["max_price"] = Decimal(filled["fill"]["price"]) - Decimal("1")      # the limit was a dollar under what the fill cost
    assert any("above the user's price limit" in reason for reason in wo.verify_trades(order))
    bad = copy.deepcopy(order)
    next(d for d in bad.trade_decisions if d["symbol"] == "EOG")["max_price"] = "fifty"
    assert any("max_price must be a price above zero" in reason for reason in wo.verify_trade(bad, filled["record_id"], MOMENT))


# ---------------------------------------------------------------- L5: a direct order with a limit is placed only while the quote is at or under it

def test_the_real_reply_to_buy_10_of_a_stock_under_50_a_share_is_an_order_for_10_shares_with_a_limit():
    intent = read(MSFT_UNDER_50)
    assert intent.action == "order" and fields(intent) == [("buy", "MSFT", Decimal(10), None)] and intent.max_price == Decimal(50) and [o.max_price for o in intent.orders] == [Decimal(50)]


def test_a_direct_order_above_its_limit_is_not_placed_and_the_user_is_told_the_price_and_the_limit_and_nothing_rests():
    state = run_desk(MSFT_UNDER_50, AUTO)
    order = state["order"]
    assert state["errors"] == [] and state["path"] == ["account_manager", "account_manager_direct"] and order.status == "decided" and order.trades == []
    (decision,) = order.trade_decisions
    price = close_of("MSFT")
    assert decision["will_trade"] == "no_system_denied" and decision["shares"] == 0 and decision["max_price"] == Decimal("50.0000") and decision["price"] == price
    assert decision["reason"].startswith(f"MSFT is at ${price:,.2f} a share, above your limit of $50.00: not placed. An order that waits for its price needs the saved account book") and decision.get("open_order_id") is None      # entry 140 (W1): a limit that is not met WAITS as a day order when there is an account book to set the cash aside; this desk run has none, so it is not placed and says why (was: "Nothing is left resting in the market ... not built yet")
    text = "\n".join(direct.direct_section(order))
    assert "NOT PLACED (above your price limit)" in text and "no price" not in text and order.direct_orders[0]["max_price"] == Decimal("50.0000") and wo.verify_direct_orders(order.direct_orders) == []


def test_a_direct_order_at_or_under_its_limit_is_placed_as_typed_and_the_limit_is_on_its_decision():
    state = run_desk(MSFT_UNDER_1000, AUTO)
    order = state["order"]
    assert state["path"] == ["account_manager", "account_manager_direct", "trader"] and order.status == "traded" and order.errors == []
    (decision,), (trade,) = order.trade_decisions, order.trades
    assert decision["will_trade"] == "yes_user" and decision["max_price"] == Decimal("1000.0000") and decision["shares"] == 10 and trade["status"] == "filled" and trade["fill"]["shares"] == 10 and wo.verify_trades(order) == []


def test_the_dollars_of_an_order_above_its_limit_are_not_sized_so_the_other_stock_gets_the_money():
    """"The real reply to "Buy $5,000 of MSFT and TSLA under $400 a share": MSFT ($523.88) costs more than the limit and TSLA ($390.22) less, so MSFT takes no part of the money and TSLA is bought with the total."""
    typed = "Buy $5,000 of MSFT and TSLA under $400 a share"
    intent = read(typed)
    assert intent.budget == Decimal(5000) and [o.max_price for o in intent.orders] == [Decimal(400), Decimal(400)] and [o.symbol for o in intent.orders] == ["MSFT", "TSLA"]
    order = wo.new_work_order(typed, requester="desk", action_type="trade")
    am = AccountManager(model=object(), model_id="direct", clerk=clerk(), profile=AUTO)
    assert am.name_the_stocks(order, intent.orders, typed, MOMENT) == []
    out = am.decide_direct(order, BILL, MOMENT, "auto", None, said_how_much=True, request_text=typed, budget=intent.budget)
    by_symbol = {d["symbol"]: d for d in out.trade_decisions}
    assert close_of("MSFT") > 400 > close_of("TSLA")
    assert by_symbol["MSFT"]["will_trade"] == "no_system_denied" and "above your limit of $400.00" in by_symbol["MSFT"]["reason"] and by_symbol["MSFT"]["shares"] == 0
    assert by_symbol["TSLA"]["will_trade"] == "yes_user" and by_symbol["TSLA"]["dollars"] == Decimal("5000.00") and by_symbol["TSLA"]["max_price"] == Decimal("400")      # the whole $5,000 is TSLA's: nothing is held for a stock that is not placed


def test_a_sale_is_never_limited_by_a_price():
    entry = {"side": "sell", "max_price": Decimal("50")}
    assert direct.over_limit(entry, Decimal("500")) is None and direct.over_limit({"side": "buy", "max_price": Decimal("50")}, Decimal("500")) == Decimal("50.0000")
    assert direct.over_limit({"side": "buy", "max_price": 50.0}, Decimal("50")) is None and direct.over_limit({"side": "buy"}, Decimal("50")) is None      # a saved order's limit is a float: through the price door


# ---------------------------------------------------------------- the how-much ANSWER: Jacob's rule D1/D2 applies to what the user types back

from test_account_manager_clarify import CASH, CONFIG, EQUITY, PURCHASES, RECORDED  # noqa: E402
from test_account_manager_clarify import Replay as ClarifyReplay  # noqa: E402
from test_account_manager_clarify import ask_mode, facts, manager as clarify_manager, rows, scenario  # noqa: E402

from prometheus.agents.account_manager import clarify  # noqa: E402
from prometheus.prompts import account_manager_sizing as sizing_prompt  # noqa: E402


@pytest.mark.parametrize("said, bare", [
    ("10", [10]), ("10 of them", [10]), ("10 shares of each", [10]), ("ten", [10]), ("10 each", [10]), ("about 3,000 shares.", [3000]), ("10, split evenly", [10]), ("2.5 shares", ["2.5"]),
    ("$10", []), ("10 dollars", []), ("10 bucks", []), ("$3,000 in total", []), ("10%", []), ("10 percent", []), ("3k", []), ("5 thousand", []), ("as much as I can", []),
    ("$10 in each, 5 shares of the last", [5]), ("20% of my account, ten shares", [10]), ("", []),
])
def test_a_bare_number_in_an_answer_is_a_number_of_shares_and_money_and_percentages_are_not(said, bare):
    assert clarify.bare_numbers(said) == {Decimal(str(b)) for b in bare}


@pytest.mark.parametrize("choice, said", [
    (clarify.SizingChoice(unit="shares", amount=10), "10"), (clarify.SizingChoice(unit="shares", amount=10), "10 of them"), (clarify.SizingChoice(unit="shares", amount=10, scope="each"), "10 each"),
    (clarify.SizingChoice(unit="shares", amount=10), "ten"), (clarify.SizingChoice(unit="shares", amount=10, scope="each"), "10 shares of each"),
    (clarify.SizingChoice(unit="dollars", amount=3000), "$3,000"), (clarify.SizingChoice(unit="dollars", amount=3000), "3000 dollars"), (clarify.SizingChoice(unit="dollars", amount=500), "500 bucks"),
    (clarify.SizingChoice(unit="dollars", amount=3000), "3k dollars"), (clarify.SizingChoice(unit="dollars", amount=3000), "$3k"), (clarify.SizingChoice(unit="percent", amount=10), "10%"),
])
def test_the_answers_kind_is_the_one_it_was_written_in(choice, said):
    clarify.check_choice(choice, said, PURCHASES, CASH, EQUITY)


@pytest.mark.parametrize("choice, said, why", [
    (clarify.SizingChoice(unit="dollars", amount=10), "10", r"has no \$ and no word dollars"),                              # a bare number is never dollars
    (clarify.SizingChoice(unit="dollars", amount=10), "10 of them", r"has no \$ and no word dollars"),
    (clarify.SizingChoice(unit="dollars", amount=10, scope="each"), "10 each", r"written that way it is a number of SHARES"),
    (clarify.SizingChoice(unit="dollars", amount=3000), "3k", r"has no \$ and no word dollars"),                           # a unit alone is no money either
    (clarify.SizingChoice(unit="dollars", amount=10), "10%", "gave percent or shares, not dollars"),
    (clarify.SizingChoice(unit="shares", amount=10), "$10", "did not give a number of shares"),                          # money is not shares
    (clarify.SizingChoice(unit="shares", amount=10), "10 dollars", "did not give a number of shares"),
    (clarify.SizingChoice(unit="shares", amount=10), "10 bucks", "did not give a number of shares"),
    (clarify.SizingChoice(unit="shares", amount=10), "10%", "did not give a number of shares"),                          # a percentage is not shares
    (clarify.SizingChoice(unit="shares", amount=3000), "3k", "did not give a number of shares"),
    (clarify.SizingChoice(unit="percent", amount=10), "10", "did not give a percentage"),                                # a bare number is not a percentage either
    (clarify.SizingChoice(unit="shares", amount=2.5), "2.5", "shares are a whole number"),
])
def test_the_kind_a_model_gives_a_number_the_user_wrote_otherwise_is_refused_with_the_reason(choice, said, why):
    with pytest.raises(ValueError, match=why):
        clarify.check_choice(choice, said, PURCHASES, CASH, EQUITY)


def test_the_real_recorded_reading_of_ten_shares_of_each_stands_when_the_user_just_types_ten_each():
    """The model's REAL replies to "10 shares of each" (`clarify_ask_mode.json`, `shares_each_raise_yes`) given back for the answer "10 of each": the number has no $ and no %, so it is shares, and the purchases are 10 shares of each."""
    model, _ = scenario("shares_each_raise_yes")
    asked = ask_mode(clarify_manager(model), clarify=channel_of(["10 of each"]), request_text=RECORDED["shares_each_raise_yes"]["request"])
    assert [(d["symbol"], d["shares"]) for d in asked.trade_decisions if d["will_trade"] == "pending_user"] == [("CVX", 10), ("EOG", 10)]      # (the third decision is the Critic's denial)


def test_the_real_recorded_reading_of_dollars_does_not_stand_for_a_bare_number_and_the_default_size_is_shown_with_the_reason():
    """The model's REAL replies to "$3,000 in total, split evenly" (`dollars_total_even`) given back for the answer "3000 in total, split evenly": no $ and no word dollars, so code refuses dollars (the model, which here repeats itself, cannot make it money)
    and the default size is shown with the reason on the order. With the dollar sign, the same replies are accepted as before."""
    replies = RECORDED["dollars_total_even"]["replies"]
    outcome = clarify.run_loop(ClarifyReplay(replies, repeat=True), facts(), PURCHASES, CASH, EQUITY, channel_of(["3000 in total, split evenly"]), CONFIG)
    assert outcome.choice is None and outcome.problems and all("has no $ and no word dollars" in p for p in outcome.problems)
    order = ask_mode(clarify_manager(ClarifyReplay(replies, repeat=True)), clarify=channel_of(["3000 in total, split evenly"]), request_text=RECORDED["dollars_total_even"]["request"])
    assert any("how much to buy could not be worked out with you" in e and "has no $ and no word dollars" in e for e in order.errors) and "balance" not in rows(order)["CVX"]
    accepted = clarify.run_loop(ClarifyReplay(replies), facts(), PURCHASES, CASH, EQUITY, channel_of(["$3,000 in total, split evenly"]), CONFIG)
    assert accepted.choice.unit == "dollars" and accepted.choice.amount == 3000 and accepted.problems == []


@pytest.mark.parametrize("name", ["percent_total", "dollars_total_even", "shares_each_raise_yes", "percent_each_no_raise_needed", "as_much_as_i_can", "vague_then_percent"])
def test_every_real_recorded_answer_is_read_as_it_always_was(name):
    """The answers of the real captures were all written with a unit ("20% of my account", "$3,000", "10 shares of each", "5%", "as much as I can", "15 percent"): none changes."""
    model, answers = scenario(name)
    outcome = clarify.run_loop(model, facts(), PURCHASES, CASH, EQUITY, channel_of(answers), CONFIG)
    assert outcome.choice is not None and outcome.problems == []


def test_the_question_never_invites_a_bare_number_to_mean_dollars():
    system = sizing_prompt.SYSTEM
    assert "DOLLARS OR SHARES" in system and "Never invite a bare number to mean dollars" in system and "10 means 10 shares" in system and "a dollar amount written with a dollar sign or the word dollars ($3,000)" in system
    assert "A number typed WITHOUT these" in system and 'is a number of SHARES: unit "shares", never "dollars"' in system
    shown = []
    clarify.terminal_clarifier(ask=lambda prompt: "10", show=shown.append, interactive=True)({"question": "How much would you like to buy?", "n": 1, "of": 3, "facts": ["Cash $15,000.00"], "kind": "how_much"})
    assert "a number of shares (a number with no $ is shares: 10 is 10 shares)" in shown[0] and "dollars (with a $ or the word dollars: $3,000)" in shown[0] and "dollars, a number of shares" not in shown[0]
