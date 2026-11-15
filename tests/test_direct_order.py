"""A DIRECT ORDER: the user names the trade ("Buy 50 shares of MSFT", "sell 10 shares of Tesla"), so nothing is recommended or reviewed (design choices entries 87 and 88).

REAL DATA ONLY, as `handshake_helpers.py`: the price of MSFT is the stored market close of Friday 2026-11-13 read by the real Data Clerk (the simulated clock, no Finnhub quote); the symbol directory is Finnhub's REAL answers,
recorded on 2026-11-15 (`tests/fixtures/finnhub_company_search.json`) and given back to the real Data Clerk as its directory, as `tests/test_symbols.py` does; the Trader is the real Trader with no model (the sweep places the
order), the Account Manager the real one, the books the REAL local Postgres on THROWAWAY accounts that the tests create and delete (the three sample accounts are read before and after and must not move; skipped when
Postgres is not reachable). The Account Manager's model is a REPLAY of its REAL replies, recorded on 2026-11-15 (`tests/fixtures/session_real/direct_order_forms.json`): the real code checks every field of them against the
words the user typed. The only guards that are not the real thing are factories that FAIL THE TEST if the desk calls them (the Data Clerk's node, the Advisor and the Critic must never be run for a direct order): they check,
they replace nothing.

What the tests pin: the form (a trade is read only when the user's words give it, by ticker or by company), the Data Clerk's lookup and the Account Manager appending the company name and the symbol to the work order, the
Trader's rule T1 (a decision is `direct` only while the order holds the user's own record, and it runs out), the decision (priced by the Data Clerk, the user's own order is the approval, a company the user named is confirmed in
Ask mode and not in Auto mode, a purchase the cash does not pay for is refused with the shares it does pay for), that no profile limit ever applies, the handshake (placed as typed or not at all), and the graph and the session end
to end.
"""
import ast
import copy
import inspect
import json
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from handshake_helpers import FRIDAY, MOMENT, Recorder, close_of, friday_close, jacob_as_bob, trader  # noqa: F401  (friday_close is a fixture)
from test_account_manager_clarify import RECORDED
from test_account_manager_clarify import Replay as AskReplay
from test_account_manager_clarify import channel_of

from prometheus.agents.account_manager import direct
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.trader.report import trade_section
from prometheus.application import session
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, Position
from prometheus.common.user_profile import UserProfile
from prometheus.langgraph import desk
from prometheus.persistence.accounts import SECRETS, open_account_store
from prometheus.prompts import account_manager_request as prompt
from prometheus.tools.symbols import is_plain_ticker

pytestmark = pytest.mark.usefixtures("friday_close")

FIXTURES = Path(__file__).parent / "fixtures"
FORMS = json.loads((FIXTURES / "session_real" / "direct_order_forms.json").read_text(encoding="utf-8"))["requests"]
DIRECTORY = json.loads((FIXTURES / "finnhub_company_search.json").read_text(encoding="utf-8"))["queries"]
BUY_50 = "Buy 50 shares of MSFT"
BUY_COMPANY = "buy 50 shares of Microsoft"
BILL = Portfolio(user="Bill Gates", cash_avail=20000.0)
BILL_PROFILE = UserProfile(user="Bill Gates", risk_tolerance="moderate", age=70, trade_mode="ask")
BOB_PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto")
YEAR = (date(2026, 1, 1), date(2026, 12, 31))
SAMPLES = ("jacob_sample", "avery_chen", "blake_okonkwo")


def real_form(typed: str, which: int = 0) -> str:
    return FORMS[typed]["replies"][which]


def form(**changes) -> str:
    """A form of the shape the model answers in, with the defaults, changed (for the faults: one changed field of a real reply)."""
    return json.dumps({"action": "order", "topic": None, "symbols": [], "companies": [], "count": None, "allocation": "default", "budget": None, "orders": [{"side": "buy", "symbol": "MSFT", "shares": 50}], **changes})


class Replay:
    """The Account Manager's model: its REAL recorded replies, given back in order."""

    def __init__(self, replies):
        self.replies, self.seen, self.model_name = list(replies), [], "openai/gpt-oss-120b"

    def invoke(self, messages):
        self.seen.append(messages)
        if not self.replies:
            raise AssertionError("the model was asked more times than it has recorded replies")
        return SimpleNamespace(content=self.replies.pop(0), usage_metadata={"input_tokens": 10, "output_tokens": 10})


def directory(query: str):
    """The symbol directory: Finnhub's REAL recorded answers. A query nobody recorded is a failure of the test, not an empty answer."""
    if query not in DIRECTORY:
        raise AssertionError(f"the Data Clerk asked the symbol directory for {query!r}, which was not recorded")
    return DIRECTORY[query]


def clerk() -> DataClerk:
    real = DataClerk(model=object(), symbol_search=directory)
    real.lookup_interval = 0
    return real


def account(profile: UserProfile = BILL_PROFILE, **options) -> AccountManager:
    """The real Account Manager over the real Data Clerk (its symbols from the recorded directory, its prices from the stored closes)."""
    return AccountManager(model=object(), model_id="direct", clerk=clerk(), profile=profile, **options)


def reading_manager(typed: str, **options) -> AccountManager:
    """The real Account Manager whose model gives back what the real model said to `typed`."""
    return AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk(), **options)


def wanted(side: str, stock: str, shares: int) -> it.DirectOrder:
    """One order of the form: the stock is a ticker (in capitals) or a company's name."""
    return it.DirectOrder(side=side, shares=shares, **({"symbol": stock} if is_plain_ticker(stock) else {"company": stock}))


def typed_words(side: str, stock: str, shares: int) -> str:
    return f"{side.capitalize()} {shares} shares of {stock}"


def direct_order(side: str = "buy", stock: str = "MSFT", shares: int = 50, portfolio: Portfolio = BILL, profile: UserProfile = BILL_PROFILE, mode: str | None = None, approver=None, **options) -> tuple[wo.WorkOrder, AccountManager]:
    """A work order for one trade the user named, made as the desk makes it (the Account Manager asks the real Data Clerk for the company name and the symbol and appends them), then decided at the real moment of the real session."""
    order = wo.new_work_order(typed_words(side, stock, shares), requester="desk", action_type="trade")
    am = account(profile, **options)
    problems = am.name_the_stocks(order, [wanted(side, stock, shares)], typed_words(side, stock, shares), MOMENT)
    assert problems == [], problems
    return am.decide_direct(order, portfolio, MOMENT, mode, approver), am


# ---------------------------------------------------------------- the form: a trade is read only when the user's words give it

@pytest.mark.parametrize("typed, expected", [
    (BUY_50, [("buy", "MSFT", None, 50)]),
    ("please purchase 25 shares of Microsoft Corp (MSFT) for me", [("buy", None, "Microsoft Corp", 25)]),
    ("Sell 10 shares of TSLA and buy 5 shares of GM", [("sell", "TSLA", None, 10), ("buy", "GM", None, 5)]),
    (BUY_COMPANY, [("buy", None, "Microsoft", 50)]),
    ("Buy 20 shares of Procter & Gamble and sell 10 shares of Tesla", [("buy", None, "Procter & Gamble", 20), ("sell", None, "Tesla", 10)]),
])
def test_the_real_models_reply_to_a_request_that_names_the_trade_is_a_direct_order_the_words_support(typed, expected):
    intent = it.parse_intent(real_form(typed), typed)
    assert intent.action == "order" and [(o.side, o.symbol, o.company, o.shares) for o in intent.orders] == expected
    assert intent.said_how_much() and intent.describe().endswith("(your own order: no recommendation is made and the Critic is not asked)")
    assert intent.topic is None and intent.symbols == [] and intent.companies == [] and intent.count is None and intent.budget is None
    with pytest.raises(ValueError, match="not a search for companies"):
        intent.as_request()


def test_a_dollar_amount_is_not_a_number_of_shares_the_real_model_reads_it_as_a_direct_order_with_a_budget_and_no_shares():
    typed = "Buy $5,000 of MSFT"
    intent = it.parse_intent(real_form(typed), typed)
    assert intent.action == "order" and [(o.side, o.symbol, o.shares) for o in intent.orders] == [("buy", "MSFT", None)] and intent.budget == 5000 and intent.said_how_much()       # design choices entry 98: no recommendation
    with pytest.raises(ValueError, match="5,000 shares of MSFT is not what you typed"):
        it.parse_intent(form(orders=[{"side": "buy", "symbol": "MSFT", "shares": 5000}]), typed)                          # even a model that said "order" cannot make the dollars shares


def test_a_request_for_recommendations_is_still_a_request_for_recommendations():
    typed = "I want to buy oil stocks. Buy the 3 most recommended companies."
    intent = it.parse_intent(real_form(typed), typed)
    assert (intent.action, intent.topic, intent.count, intent.orders) == ("buy", "oil", 3, [])


def test_the_account_manager_reads_a_company_named_order_with_one_call_of_the_model():
    am = reading_manager(BUY_COMPANY)
    intent = am.understand_request(BUY_COMPANY)
    assert intent.orders[0].company == "Microsoft" and intent.orders[0].symbol is None and intent.describe().startswith("buy 50 shares of Microsoft") and len(am.model.seen) == 1


@pytest.mark.parametrize("typed, changes, message", [
    (BUY_50, {"orders": [{"side": "sell", "symbol": "MSFT", "shares": 50}]}, "sell MSFT is not what you typed"),                               # the side the user did not say
    (BUY_50, {"orders": [{"side": "buy", "symbol": "MSFT", "shares": 500}]}, "500 shares of MSFT is not what you typed"),                      # a number the user did not say
    (BUY_50, {"orders": [{"side": "buy", "symbol": "AAPL", "shares": 50}]}, "the ticker AAPL is not in what you typed"),                      # a ticker the user did not say
    (BUY_50, {"orders": [{"side": "buy", "symbol": "MSF", "shares": 50}]}, "the ticker MSF is not in what you typed"),                        # a prefix is not the ticker
    (BUY_COMPANY, {"orders": [{"side": "buy", "company": "Apple", "shares": 50}]}, "the company 'Apple' is not in what you typed"),            # a company the user did not name
    (BUY_COMPANY, {"orders": [{"side": "sell", "company": "Microsoft", "shares": 50}]}, "sell Microsoft is not what you typed"),
    (BUY_COMPANY, {"orders": [{"side": "buy", "company": "Microsoft", "shares": 5}]}, "5 shares of Microsoft is not what you typed"),
    (BUY_COMPANY, {"orders": [{"side": "buy", "symbol": "Microsoft", "shares": 50}]}, "not a ticker in capitals: a company's name goes in company"),     # a name in the ticker's place: the model is told where it goes
    (BUY_COMPANY, {"orders": [{"side": "buy", "company": "MSFT", "shares": 50}]}, "is a ticker: a ticker goes in symbol"),                    # a ticker in the name's place
    (BUY_COMPANY, {"orders": [{"side": "buy", "symbol": "MSFT", "company": "Microsoft", "shares": 50}]}, "orders must be a list"),           # both ways of naming one stock
    ("Buy $50 of Microsoft", {"orders": [{"side": "buy", "company": "Microsoft", "shares": 50}]}, "50 shares of Microsoft is not what you typed"),
    ("Buy 50.5 shares of MSFT", {"orders": [{"side": "buy", "symbol": "MSFT", "shares": 50}]}, "50 shares of MSFT is not what you typed"),     # a decimal is not a whole number of shares
    ("Please purchase twenty-five shares of MSFT", {"orders": [{"side": "buy", "symbol": "MSFT", "shares": 5}]}, "5 shares of MSFT is not what you typed"),    # "five" is the end of "twenty-five", not the number
    ("Buy 50 shares of MSFT and sell 10 shares of TSLA", {"orders": [{"side": "buy", "symbol": "MSFT", "shares": 10}, {"side": "sell", "symbol": "TSLA", "shares": 50}]},
     "10 shares of MSFT is not what you typed"),                                                                                              # the two numbers swapped between the tickers
    ("Sell 10 shares of TSLA and buy 5 shares of GM", {"orders": [{"side": "buy", "symbol": "TSLA", "shares": 10}, {"side": "sell", "symbol": "GM", "shares": 5}]},
     "buy TSLA is not what you typed"),                                                                                                       # the two sides swapped between the tickers
    (BUY_50, {"orders": [{"side": "buy", "symbol": "MSFT", "shares": 0}]}, "greater than or equal to 1"),
    (BUY_50, {"orders": [{"side": "buy", "symbol": "MSFT", "shares": True}]}, "orders must be a list"),
    (BUY_50, {"orders": [{"side": "buy", "symbol": "MSFT", "shares": 50}, {"side": "buy", "symbol": "MSFT", "shares": 50}]}, "one trade for each stock"),
    (BUY_50, {"orders": []}, "a direct order has the action 'order' and at least one trade"),
    (BUY_50, {"topic": "oil"}, "names the trades and nothing else"),
    (BUY_50, {"budget": 50}, "names the trades and nothing else"),
    (BUY_50, {"action": "buy"}, "no other action has any"),
    (BUY_50, {"orders": [{"side": "buy", "symbol": "MSFT", "shares": 50, "price": 400}]}, "orders must be a list"),
    (BUY_50, {"symbols": ["AAPL"]}, "names the trades and nothing else"),                                                                     # a ticker that is not one of the orders
])
def test_a_form_the_words_do_not_support_is_refused_with_the_reason(typed, changes, message):
    with pytest.raises(ValueError, match=message):
        it.parse_intent(form(**changes), typed)


def test_the_tickers_and_names_the_model_repeats_outside_the_orders_are_the_orders_own_and_nothing_is_lost():
    assert it.parse_intent(form(symbols=["MSFT"]), BUY_50).orders[0].symbol == "MSFT"
    repeated = form(orders=[{"side": "buy", "company": "Microsoft", "shares": 50}], companies=["microsoft"])
    assert it.parse_intent(repeated, BUY_COMPANY).orders[0].company == "Microsoft"


def test_the_prompt_teaches_the_orders_form_and_names_no_company():
    assert '"orders": []' in prompt.SYSTEM and 'action: "order"' in prompt.SYSTEM and "orders" in prompt.RETRY and '"company"' in prompt.SYSTEM
    assert "MSFT" not in prompt.SYSTEM and "Microsoft" not in prompt.SYSTEM


# ---------------------------------------------------------------- the Data Clerk finds the company name and the symbol; the Account Manager appends them to the work order

def test_the_data_clerk_finds_the_company_of_a_ticker_and_the_symbol_of_a_company_name_with_exact_matching():
    order = wo.new_work_order("lookup", action_type="lookup_symbol", subject=["MSFT", "Tesla", "Procter & Gamble"])
    found = clerk().request_symbols(order)
    assert found is order and found.status == "data_ready" and found.errors == []
    rows = {row["brand"]: (row["symbol"], row["company"], row["how"]) for row in found.data.companies}
    assert rows == {"MSFT": ("MSFT", "Microsoft Corp", "ticker"), "Tesla": ("TSLA", "Tesla Inc", "name"), "Procter & Gamble": ("PG", "The Procter & Gamble Company", "name")}
    assert [c["outcome"] for c in found.data.tool_calls] == ["ok", "ok", "ok"] and "Tesla: Tesla Inc (TSLA)" in found.data.answer


def test_the_data_clerk_appends_to_what_the_order_already_has_and_leaves_an_answered_order_alone():
    order = wo.new_work_order("lookup", action_type="lookup_symbol", subject=["Microsoft"])
    order.data = wo.DataResult(companies=[{"brand": "earlier", "symbol": "TSLA", "company": "Tesla Inc"}])
    found = clerk().request_symbols(order)
    assert [row["brand"] for row in found.data.companies] == ["earlier", "Microsoft"]
    again = copy.deepcopy(found)
    assert clerk().request_symbols(again).model_dump() == found.model_dump()          # data_ready already: unchanged


@pytest.mark.parametrize("subject, message", [(["Zzyzx Holdings"], "Zzyzx Holdings: Company Name not found."), (["ZZZZ"], "ZZZZ: Symbol: ZZZZ is not found")])
def test_a_stock_the_directory_does_not_have_fails_the_order_with_the_data_clerks_own_message_and_the_others_stay(subject, message):
    found = clerk().request_symbols(wo.new_work_order("lookup", action_type="lookup_symbol", subject=["Microsoft", *subject]))
    assert found.status == "data_failed" and message in found.errors and [row["symbol"] for row in found.data.companies] == ["MSFT"]


def test_a_symbol_lookup_order_is_checked():
    ok = wo.new_work_order("x", action_type="lookup_symbol", subject=["MSFT"])
    assert wo.verify_symbol_request(ok) == []
    for change, text in (({"action_type": "fetch_quote"}, "is not 'lookup_symbol'"), ({"subject": []}, "needs one or more tickers or company names"), ({"subject": ["a;b"]}, "needs one or more tickers or company names"),
                         ({"subject": ["x" * 61]}, "needs one or more tickers or company names"), ({"subject": [f"S{n}" for n in range(11)]}, "at most 10 stocks"), ({"status": "data_ready"}, "already has its data")):
        bad = copy.deepcopy(ok)
        for key, value in change.items():
            setattr(bad, key, value)
        assert text in "; ".join(wo.verify_symbol_request(bad)), change


def test_the_account_manager_asks_the_data_clerk_and_appends_the_company_name_and_the_symbol_to_the_work_order():
    order = wo.new_work_order(BUY_COMPANY, requester="desk", action_type="trade")
    am = account()
    assert am.name_the_stocks(order, [wanted("buy", "Microsoft", 50)], BUY_COMPANY, MOMENT) == []
    (entry,) = order.direct_orders
    assert (entry["symbol"], entry["company"], entry["typed_company"], entry["shares"], entry["side"], entry["request"]) == ("MSFT", "Microsoft Corporation", "Microsoft", 50, "buy", BUY_COMPANY)
    assert order.subject == ["MSFT"] and [(row["brand"], row["symbol"], row["company"]) for row in order.data.companies] == [("Microsoft", "MSFT", "Microsoft Corporation")]
    assert order.data.tool_calls[0]["name"] == "find_company" and order.status == "new" and wo.verify_direct_orders(order.direct_orders) == []


def test_a_ticker_the_user_typed_also_gets_its_company_name_on_the_work_order():
    order = wo.new_work_order(BUY_50, requester="desk", action_type="trade")
    assert account().name_the_stocks(order, [wanted("buy", "MSFT", 50)], BUY_50, MOMENT) == []
    (entry,) = order.direct_orders
    assert (entry["symbol"], entry["company"], entry["typed_company"]) == ("MSFT", "Microsoft Corp", None) and order.data.companies[0]["how"] == "ticker"


def test_a_stock_nobody_can_find_makes_no_order_and_says_why_in_the_data_clerks_words():
    for stock, message in (("Zzyzx Holdings", "Company Name not found."), ("ZZZZ", "Symbol: ZZZZ is not found")):
        order = wo.new_work_order("x", requester="desk", action_type="trade")
        problems = account().name_the_stocks(order, [wanted("buy", stock, 1)], typed_words("buy", stock, 1), MOMENT)
        assert any(message in p for p in problems) and order.direct_orders == [] and order.data is None, stock


def test_two_names_of_one_stock_are_refused_and_nothing_is_appended():
    order = wo.new_work_order("x", requester="desk", action_type="trade")
    typed = "Buy 5 shares of Microsoft and buy 5 shares of MSFT"
    problems = account().name_the_stocks(order, [wanted("buy", "Microsoft", 5), wanted("buy", "MSFT", 5)], typed, MOMENT)
    assert problems == ["two of your orders are for the same stock (a ticker and the company's name, for example): name each stock once"] and order.direct_orders == [] and order.data is None


def test_without_a_key_or_with_an_outage_the_lookup_is_a_message_and_not_a_guess():
    order = wo.new_work_order("x", requester="desk", action_type="trade")

    def down(query):
        raise ConnectionError("connection to finnhub failed: token abc")

    broken = DataClerk(model=object(), symbol_search=down)
    broken.lookup_interval = 0
    problems = AccountManager(model=object(), model_id="direct", clerk=broken, profile=BILL_PROFILE).name_the_stocks(order, [wanted("buy", "MSFT", 1)], "Buy 1 share of MSFT", MOMENT)
    assert problems and order.direct_orders == [] and "token" not in " ".join(problems)


# ---------------------------------------------------------------- the Trader's rule T1: the user's own order is a permission, and only while it stands

def test_a_decision_for_the_users_own_order_permits_the_trade_with_no_review_and_runs_out_with_its_price():
    order, _ = direct_order(shares=38)
    record = order.direct_orders[0]["record_id"]
    assert order.reviews == [] and order.recommendations == [] and wo.verify_trade(order, record, MOMENT) == []
    until = wo.parse_utc(order.trade_decisions[0]["valid_until_utc"])
    assert until == MOMENT + timedelta(minutes=30)
    assert wo.verify_trade(order, record, until) == [] and wo.verify_trade(order, record, until + timedelta(seconds=1))[0].startswith(wo.EXPIRY_PREFIX)


def test_the_word_direct_alone_permits_nothing_the_order_must_hold_the_users_own_record_and_a_decision_may_not_exceed_it():
    order, _ = direct_order(shares=38)
    record = order.direct_orders[0]["record_id"]
    stranger = copy.deepcopy(order)
    stranger.direct_orders = []
    assert "holds no such order of the user's" in wo.verify_trade(stranger, record, MOMENT)[0]
    more = copy.deepcopy(order)
    more.trade_decisions[0]["shares"] = 39
    assert "the decision gives 39 shares but the user typed 38" in "; ".join(wo.verify_trade(more, record, MOMENT))
    other = copy.deepcopy(order)
    other.trade_decisions[0]["symbol"] = "AAPL"
    assert "the user typed buy MSFT" in "; ".join(wo.verify_trade(other, record, MOMENT))
    sold = copy.deepcopy(order)
    sold.trade_decisions[0]["action"] = "reduce"
    assert "the user typed buy MSFT" in "; ".join(wo.verify_trade(sold, record, MOMENT))
    reviewed = copy.deepcopy(order)
    reviewed.reviews = [{"record_id": record, "approved": True}]
    assert "a direct order has no Critic review" in "; ".join(wo.verify_trade(reviewed, record, MOMENT))


def test_a_decision_that_is_not_direct_still_needs_the_critics_approval_the_old_rule_is_unchanged():
    order, _ = direct_order(shares=38)
    record = order.direct_orders[0]["record_id"]
    del order.trade_decisions[0]["origin"]
    assert wo.verify_trade(order, record, MOMENT) == [f"there is no Critic review of recommendation {record}"]


def test_the_records_of_the_users_orders_are_checked_and_each_has_its_symbol_and_its_company_name():
    good = {"record_id": "direct-1-1", "side": "buy", "symbol": "MSFT", "company": "Microsoft Corporation", "shares": 50, "request": BUY_50, "requested_utc": "2026-11-13T20:30:00Z", "typed_company": None}
    assert wo.verify_direct_orders([good]) == [] and wo.verify_direct_orders([]) and wo.verify_direct_orders("x")
    for change, text in (({"side": "short"}, "side must be one of"), ({"symbol": "Microsoft"}, "not a checked ticker"), ({"symbol": None}, "not a checked ticker"), ({"shares": 0}, "whole number"), ({"shares": True}, "whole number"),
                         ({"shares": 2_000_000}, "whole number"), ({"request": ""}, "no words of the user's"), ({"record_id": "x" * 81}, "record_id"), ({"company": None}, "no company name for it"),
                         ({"company": " "}, "no company name for it"), ({"typed_company": 5}, "typed_company is text")):
        assert text in "; ".join(wo.verify_direct_orders([{**good, **change}])), change
    assert "not used twice" in "; ".join(wo.verify_direct_orders([good, good])) and "keys" in "; ".join(wo.verify_direct_orders([{**good, "extra": 1}]))
    assert "keys" in "; ".join(wo.verify_direct_orders([{k: v for k, v in good.items() if k != "company"}]))


# ---------------------------------------------------------------- the decision

def test_the_account_manager_prices_the_order_with_the_data_clerk_and_the_users_own_order_is_the_approval_even_in_ask_mode():
    order, _ = direct_order(shares=38, mode="ask")
    price, (d,) = close_of("MSFT"), order.trade_decisions
    assert order.status == "decided" and order.errors == [] and order.recommendations == [] and order.reviews == [] and order.critic_runs == [] and order.advisor_runs == []
    assert (d["will_trade"], d["origin"], d["action"], d["shares"], d["price"], d["mode"], d["asked_user"], d["review_id"], d["company"]) == ("yes_user", "direct", "open", 38, price, "ask", False, None, "Microsoft Corp")
    assert d["est_cost"] == round(38 * price, 2) and d["funds_avail"] is True and d["size"] is None and "no recommendation was made and the Critic was not asked" in d["reason"]
    assert d["record_id"] == order.direct_orders[0]["record_id"] == f"direct-{order.request_id}-1"
    assert order.data.companies[0]["symbol"] == "MSFT" and order.data.sources == [] and order.data.references == []        # what the Data Clerk found, and nothing was searched


def test_fifty_shares_that_the_cash_does_not_pay_for_are_refused_with_the_shares_it_does_pay_for_and_never_made_smaller():
    order, _ = direct_order(shares=50)
    price = close_of("MSFT")
    fits = int(20000 // price)
    assert 50 * price > 20000 and fits < 50                                                  # the REAL stored close: Bill's $20,000 does not buy 50 shares of MSFT
    (d,) = order.trade_decisions
    assert (d["will_trade"], d["shares"], d["funds_avail"], d["est_cost"]) == ("no_funds", 50, False, round(50 * price, 2))
    assert f"the most it pays for is {fits} shares" in d["reason"] and "as you typed it or not at all" in d["reason"]
    assert order.status == "decided" and not any(t for t in order.trades)


def test_with_more_cash_the_same_order_is_approved_and_money_is_left():
    order, _ = direct_order(shares=50, portfolio=Portfolio(user="Bill Gates", cash_avail=30000.0))
    (d,) = order.trade_decisions
    assert d["will_trade"] == "yes_user" and d["est_cost"] == round(50 * close_of("MSFT"), 2) and 30000.0 - d["est_cost"] > 0


def test_a_sale_needs_the_shares_in_the_account_and_is_the_number_the_user_typed_not_the_whole_holding():
    sample = jacob_as_bob()                                                                   # the real sample portfolio: MSFT 50, TSLA 123, GM 60
    held = {p.symbol: p.shares for p in sample.positions}
    sell, _ = direct_order("sell", "MSFT", 10, sample, BOB_PROFILE)
    (d,) = sell.trade_decisions
    assert (d["will_trade"], d["action"], d["shares"], d["est_cost"]) == ("yes_user", "reduce", 10, 0.0) and held["MSFT"] == 50
    too_many, _ = direct_order("sell", "MSFT", 60, sample, BOB_PROFILE)
    assert too_many.trade_decisions[0]["will_trade"] == "no_funds" and "holds 50 shares of MSFT and you asked to sell 60" in too_many.trade_decisions[0]["reason"]
    none, _ = direct_order("sell", "TSLA", 1000, sample, BOB_PROFILE)
    assert none.trade_decisions[0]["will_trade"] == "no_funds" and "holds 123 shares of TSLA and you asked to sell 1,000" in none.trade_decisions[0]["reason"]


def test_an_order_that_cannot_be_priced_is_not_traded_and_says_so(monkeypatch):
    """The real Data Clerk answers a price from the stored bars of the day; on Saturday 2026-11-14 there is none (the real path of `request_quotes`: a symbol with no stored bar that day is a failed quote)."""
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.market_day", lambda *a, **k: date(2026, 11, 14))
    order, _ = direct_order(shares=1)
    (d,) = order.trade_decisions
    assert d["will_trade"] == "no_system_denied" and d["price"] is None and "no current price for MSFT" in d["reason"] and "no current price for MSFT" in "; ".join(order.errors)


def test_a_profile_that_is_not_the_portfolios_user_decides_nothing_and_a_decided_order_is_not_decided_again():
    order, am = direct_order(profile=UserProfile(user="Someone Else", risk_tolerance="moderate", age=40, trade_mode="ask"), shares=1)
    assert order.status == "new" and order.trade_decisions == [] and "the user profile is for 'Someone Else'" in order.errors[0]
    fine, am = direct_order(shares=1)
    again = copy.deepcopy(fine.trade_decisions)
    assert am.decide_direct(fine, BILL, MOMENT).trade_decisions == again


# ---------------------------------------------------------------- THE PROFILE'S LIMITS NEVER APPLY TO ANY DIRECT ORDER (Jacob, 2026-11-15)

def test_no_profile_limit_exists_so_none_can_cut_or_refuse_a_direct_order_whatever_its_kind():
    """Jacob, 2026-11-15: "profile limits never apply to any orders", "I do not want any of these limits ... remove them all": the profile has no size limit at all, so a direct order is cut by nothing but the account. A tight
    profile cannot even be written: the four fields are refused. What stays are the profile's other settings (a minimum price, excluded names), which belong to the Critic and are never read here."""
    from pydantic import ValidationError

    for field in ("max_position_pct", "max_new_exposure_pct", "max_loss_per_trade_pct", "max_order_dollars"):
        assert field not in UserProfile.model_fields
        with pytest.raises(ValidationError, match=field):
            BILL_PROFILE.model_copy(update={field: 0.01}).model_validate({**BILL_PROFILE.model_dump(), field: 0.01})
    odd = BILL_PROFILE.model_copy(update={"min_price": 1000.0, "excluded_symbols": ["MSFT"]})                       # the Critic's own preferences: a direct order skips the Critic and reads none of them
    for mode in ("ask", "auto"):
        for stock in ("MSFT", "Microsoft"):                                             # a ticker and a company's name
            order, _ = direct_order(stock=stock, shares=38, profile=odd, mode=mode, approver=lambda card: True)
            (d,) = order.trade_decisions
            assert (d["will_trade"], d["shares"], d["est_cost"]) == ("yes_user", 38, round(38 * close_of("MSFT"), 2)), (mode, stock)
    sample = jacob_as_bob()
    sale, _ = direct_order("sell", "MSFT", 50, sample, BOB_PROFILE)
    assert sale.trade_decisions[0]["will_trade"] == "yes_user" and sale.trade_decisions[0]["shares"] == 50                      # the whole holding too: nothing cuts a sale either


def test_the_direct_path_does_not_so_much_as_name_a_sizing_function():
    """The Advisor's no-size test reads the source; so does this one: the module of the direct order and the Account Manager's `decide_direct` use no `sizing` at all: the only check is that the cash covers the order."""
    import textwrap

    forbidden = {"sizing", "size_purchases", "equity_of", "funds_check", "decide_trades", "load_critic_config"}

    def names(source: str) -> set[str]:
        out = set()
        for node in ast.walk(ast.parse(textwrap.dedent(source))):
            if isinstance(node, ast.Name):
                out.add(node.id)
            elif isinstance(node, ast.Attribute):
                out.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                out |= {a.name.split(".")[-1] for a in node.names} | {a.asname for a in node.names if a.asname} | set((node.module or "").split("."))
        return out

    assert names(inspect.getsource(direct)) & forbidden == set()
    assert names(inspect.getsource(AccountManager.decide_direct)) & forbidden == set()
    assert names(inspect.getsource(AccountManager.name_the_stocks)) & forbidden == set()


# ---------------------------------------------------------------- a company the user NAMED: Ask mode confirms what the Data Clerk found, Auto mode does not

def test_in_ask_mode_the_user_confirms_the_symbol_the_company_the_shares_and_the_amount_of_a_company_they_named():
    cards = []

    def user(card):
        cards.append(card)
        return True

    order, _ = direct_order(stock="Microsoft", shares=38, mode="ask", approver=user)
    (d,), (card,) = order.trade_decisions, cards
    price = close_of("MSFT")
    assert d["will_trade"] == "yes_user" and d["asked_user"] is True and "the user confirmed the symbol the Data Clerk found for the company they named, 'Microsoft' -> Microsoft Corporation (MSFT)" in d["reason"]
    for shown in ("YOU NAMED A COMPANY, NOT A TICKER", "Microsoft", "Microsoft Corporation", "MSFT", f"buy 38 shares at ${price:,.2f}", f"${38 * price:,.2f}"):
        assert shown in card["text"], shown
    assert card["question"].startswith("Is this the order you meant: buy 38 shares of Microsoft Corporation (MSFT)") and (card["symbol"], card["shares"]) == ("MSFT", 38)
    assert wo.verify_trade(order, d["record_id"], MOMENT) == []


@pytest.mark.parametrize("answer, will_trade", [(False, "no_user_denied"), (None, "pending_user")])
def test_a_no_or_no_answer_to_that_question_is_not_a_trade(answer, will_trade):
    order, am = direct_order(stock="Microsoft", shares=38, mode="ask", approver=lambda card: answer)
    (d,) = order.trade_decisions
    assert d["will_trade"] == will_trade and order.status == "decided" and wo.verify_trade(order, d["record_id"], MOMENT)           # the Trader has no permission
    am.perform_trades(order, BILL, trader(), MOMENT)
    assert order.trades == []


def test_with_nobody_to_ask_in_ask_mode_a_named_company_waits_and_says_so():
    order, _ = direct_order(stock="Microsoft", shares=38, mode="ask", approver=None)
    (d,) = order.trade_decisions
    assert d["will_trade"] == "pending_user" and d["asked_user"] is False and "needs your yes in Ask mode and nobody was there to ask" in d["reason"]


def test_a_question_that_fails_is_not_a_yes():
    def broken(card):
        raise EOFError("closed")

    order, _ = direct_order(stock="Microsoft", shares=38, mode="ask", approver=broken)
    assert order.trade_decisions[0]["will_trade"] == "pending_user" and "the user could not be asked about MSFT (EOFError)" in order.errors


def test_in_auto_mode_the_symbol_the_data_clerk_returned_is_used_without_asking():
    asked = []
    order, _ = direct_order(stock="Microsoft", shares=38, mode="auto", approver=lambda card: asked.append(card) or False)
    (d,) = order.trade_decisions
    assert asked == [] and d["will_trade"] == "yes_user" and d["asked_user"] is False and "auto mode: the symbol the Data Clerk found for the company you named, 'Microsoft' -> Microsoft Corporation (MSFT), was used without asking" in d["reason"]


def test_a_ticker_the_user_typed_is_not_asked_about_in_either_mode_and_an_order_the_cash_cannot_pay_is_not_asked_about_either():
    asked = []
    for mode in ("ask", "auto"):
        direct_order(stock="MSFT", shares=38, mode=mode, approver=lambda card: asked.append(card) or True)
    refused, _ = direct_order(stock="Microsoft", shares=50, mode="ask", approver=lambda card: asked.append(card) or True)
    assert asked == [] and refused.trade_decisions[0]["will_trade"] == "no_funds"


# ---------------------------------------------------------------- the Trader and the handshake: as typed, or not at all

def test_the_trader_places_the_users_own_order_one_trade_with_the_handshake_and_the_portfolio_shows_it(monkeypatch):
    order, am = direct_order(shares=38)
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, BILL, trader(), MOMENT)
    price = close_of("MSFT")
    (t,) = order.trades
    assert rec.kinds() == ["place T1", "approve T1", "record T1"]
    assert (t["status"], t["side"], t["symbol"], t["shares"], t["record_id"], t["review_id"], t["fill"]["shares"]) == ("filled", "buy", "MSFT", 38, order.direct_orders[0]["record_id"], None, 38)
    assert abs(t["fill"]["price"] / price - 1) < 0.002                                         # the simulated broker fills a market order at the quote and a few basis points of slippage
    after = order.trader_runs[-1]["portfolio_after"]
    assert order.status == "traded" and order.errors == [] and wo.verify_trades(order) == [] and after["cash"] == round(20000.0 - t["fill"]["amount"], 2) and after["cash"] < 100
    assert any("FILLED 38" in line for line in trade_section(order))


def test_a_company_the_user_named_is_traded_under_the_symbol_the_data_clerk_returned():
    order, am = direct_order(stock="Microsoft", shares=38, mode="auto")
    am.perform_trades(order, BILL, trader(), MOMENT)
    (t,) = order.trades
    assert (t["status"], t["symbol"], t["fill"]["shares"]) == ("filled", "MSFT", 38) and order.direct_orders[0]["typed_company"] == "Microsoft"


def test_fifty_shares_for_an_account_that_can_pay_are_placed_and_money_is_left():
    rich = Portfolio(user="Bill Gates", cash_avail=30000.0)
    order, am = direct_order(shares=50, portfolio=rich)
    am.perform_trades(order, rich, trader(), MOMENT)
    price = close_of("MSFT")
    (t,) = order.trades
    assert t["status"] == "filled" and t["fill"]["shares"] == 50 and order.trader_runs[-1]["portfolio_after"]["cash"] == round(30000.0 - t["fill"]["amount"], 2) > 3000 and abs(t["fill"]["price"] / price - 1) < 0.002


def test_a_refused_order_has_no_ticket_and_the_trader_never_asks_the_account_for_anything(monkeypatch):
    order, am = direct_order(shares=50)
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, BILL, trader(), MOMENT)
    assert order.trades == [] and rec.kinds() == [] and order.status == "decided"           # nothing was permitted: no handshake at all (entry 81)


def test_when_the_cash_has_gone_since_the_decision_the_handshake_refuses_the_order_and_does_not_cut_it():
    """A decision for 38 shares was made against $20,000; the account now has $15,000. A recommended purchase would be cut to the shares the cash pays for; the user's own order is placed as typed or not at all."""
    order, am = direct_order(shares=38)
    poorer = Portfolio(user="Bill Gates", cash_avail=15000.0)
    am.perform_trades(order, poorer, trader(), MOMENT)
    (t,) = order.trades
    assert t["status"] == "rejected" and t["reject_code"] == "insufficient_cash" and "placed as you typed it or not at all" in t["reason"] and t["fill"] is None
    assert order.trader_runs[-1]["portfolio_after"]["cash"] == 15000.0 and wo.verify_trades(order) == []


def test_a_recommended_purchase_is_still_cut_to_the_cash_that_rule_is_for_the_user_named_trades_only():
    """The same handshake for a trade that is not direct: the real session's recommendation, decided for 8 shares, meets $15,000 less than it needs and is CUT (entry 81), as before."""
    from handshake_helpers import manager, real_order
    order, am = real_order(), manager()
    am.perform_trades(order, Portfolio(user="Bob Jones", cash_avail=close_of("XOM") * 3.5), trader(), MOMENT)
    first = order.trades[0]
    assert first["status"] == "filled" and first["fill"]["shares"] == 3 and first["shares_approved"] == 8


def test_a_sale_the_user_names_is_the_shares_typed_not_the_whole_holding():
    sample = jacob_as_bob()
    order, am = direct_order("sell", "MSFT", 10, sample, BOB_PROFILE)
    am.perform_trades(order, sample, trader(), MOMENT)
    price = close_of("MSFT")
    (t,) = order.trades
    after = order.trader_runs[-1]["portfolio_after"]
    assert (t["status"], t["side"], t["shares"], t["fill"]["shares"]) == ("filled", "sell", 10, 10) and after["cash"] == round(sample.cash_avail + t["fill"]["amount"], 2) and abs(t["fill"]["price"] / price - 1) < 0.002


def test_a_sale_of_shares_some_of_which_were_bought_today_is_refused_whole_and_not_cut_to_what_may_be_sold():
    """45 of Bob's 50 shares were bought on the day of the trade (T+1: not sellable). He asks to sell 10; 5 could be sold; the order is placed as typed or not at all."""
    bought_today = Portfolio(user="Bob Jones", cash_avail=100.0, positions=[Position(symbol="MSFT", shares=50, cost_basis=close_of("MSFT") * 0.9, acquired=FRIDAY, recent_shares=45)])
    order, am = direct_order("sell", "MSFT", 10, bought_today, BOB_PROFILE)
    assert order.trade_decisions[0]["will_trade"] == "yes_user"            # the account holds them: the T+1 rule is the Trader's, at placement
    am.perform_trades(order, bought_today, trader(), MOMENT)
    (t,) = order.trades
    assert (t["status"], t["reject_code"], t["fill"]) == ("rejected", "not_sellable_today", None) and "only 5 can be sold today" in t["reason"] and "as you typed it or not at all" in t["reason"]
    assert order.trader_runs[-1]["portfolio_after"]["cash"] == 100.0


# ---------------------------------------------------------------- the two REAL sessions of 2026-11-15 (`--save` captures): Bill Gates

def capture(name: str) -> dict:
    return json.loads((FIXTURES / "session_real" / f"{name}.json").read_text(encoding="utf-8"))


def captured_order(name: str) -> wo.WorkOrder:
    return wo.WorkOrder.model_validate(copy.deepcopy(capture(name)["order"]))


def test_the_real_session_with_twenty_thousand_dollars_was_refused_whole_at_the_real_price_and_nothing_was_bought():
    """`python -m prometheus.application.session --name "Bill Gates" --age 70 --risk moderate --cash 20000 --mode ask --prompt "Buy 50 shares of MSFT" --yes`, run with the real models and the real symbol directory."""
    c, order = capture("bill_gates_direct_order"), captured_order("bill_gates_direct_order")
    assert c["request"] == BUY_50 and c["intent"]["action"] == "order" and [(o["side"], o["symbol"], o["shares"]) for o in c["intent"]["orders"]] == [("buy", "MSFT", 50)]
    assert c["path"] == ["account_manager", "account_manager_direct"] and order.status == "decided" and order.trades == [] and order.trader_runs == []
    assert order.recommendations == [] and order.reviews == [] and order.critic_runs == [] and order.advisor_runs == [] and order.errors == []
    assert order.data.sources == [] and [(r["symbol"], r["brand"]) for r in order.data.companies] == [("MSFT", "MSFT")] and order.data.companies[0]["company"].startswith("Microsoft")      # the Data Clerk's company name and symbol, on the work order
    (d,), (entry,) = order.trade_decisions, order.direct_orders
    assert (d["will_trade"], d["origin"], d["shares"], d["price"], entry["symbol"], entry["company"]) == ("no_funds", "direct", 50, close_of("MSFT"), "MSFT", order.data.companies[0]["company"])
    assert "the most it pays for is 38 shares" in d["reason"]


def test_the_real_session_with_thirty_thousand_dollars_placed_the_order_with_the_traders_model_and_booked_it():
    c, order = capture("bill_gates_direct_order_funded"), captured_order("bill_gates_direct_order_funded")
    assert c["path"] == ["account_manager", "account_manager_direct", "trader"] and order.status == "traded" and order.recommendations == [] and order.reviews == [] and order.errors == []
    (t,), (d,) = order.trades, order.trade_decisions
    assert (d["will_trade"], d["origin"], d["shares"]) == ("yes_user", "direct", 50) and (t["status"], t["placed_by"], t["review_id"], t["booked"], t["fill"]["shares"]) == ("filled", "model", None, True, 50)
    assert wo.verify_trades(order) == [] and order.trader_runs[-1]["portfolio_after"]["cash"] == round(30000.0 - t["fill"]["amount"], 2) > 3000 and order.trader_runs[-1]["model"] == "openai/gpt-oss-120b"


def test_the_real_session_that_named_a_company_in_auto_mode_used_the_symbol_the_data_clerk_returned_without_asking():
    """`... --cash 30000 --mode auto --prompt "buy 50 shares of Microsoft"`, run with the real models and the real symbol directory: the Data Clerk found MSFT, nobody was asked, the Trader's model placed it."""
    c, order = capture("bill_gates_direct_order_company_auto"), captured_order("bill_gates_direct_order_company_auto")
    assert c["request"] == BUY_COMPANY and [(o["side"], o["symbol"], o["company"], o["shares"]) for o in c["intent"]["orders"]] == [("buy", None, "Microsoft", 50)]
    assert c["path"] == ["account_manager", "account_manager_direct", "trader"] and order.status == "traded" and order.recommendations == [] and order.reviews == [] and order.errors == []
    (entry,), (d,), (t,) = order.direct_orders, order.trade_decisions, order.trades
    assert (entry["typed_company"], entry["symbol"], entry["company"]) == ("Microsoft", "MSFT", "Microsoft Corporation") and order.data.companies[0]["brand"] == "Microsoft" and order.subject == ["MSFT"]
    assert (d["mode"], d["will_trade"], d["asked_user"]) == ("auto", "yes_user", False) and "auto mode: the symbol the Data Clerk found for the company you named" in d["reason"]
    assert (t["status"], t["symbol"], t["placed_by"], t["fill"]["shares"]) == ("filled", "MSFT", "model", 50) and wo.verify_trades(order) == []


@pytest.mark.parametrize("name, cash", [("bill_gates_direct_order", 20000.0), ("bill_gates_direct_order_funded", 30000.0)])
def test_the_account_managers_decision_of_each_real_run_is_reproduced_from_its_own_record(name, cash):
    """Decide the real run's order again from what it recorded (the user's order, with its company name and symbol) and the real stored close: the same decision, to the second."""
    recorded = captured_order(name)
    again = copy.deepcopy(recorded)
    again.status, again.trade_decisions, again.trades, again.trader_runs = "new", [], [], []
    again = direct.decide_direct(again, Portfolio(user="Bill Gates", cash_avail=cash), BILL_PROFILE, "ask", {"MSFT": close_of("MSFT")}, MOMENT, 30)
    assert again.trade_decisions == recorded.trade_decisions
    reasons = wo.verify_trade(recorded, recorded.direct_orders[0]["record_id"], MOMENT)
    assert reasons == [] if cash > 25000 else "not one of yes_auto, yes_user" in "; ".join(reasons)         # the funded order may trade; the refused one may not


def test_the_funded_real_order_would_have_run_out_half_an_hour_after_it_was_priced():
    order = captured_order("bill_gates_direct_order_funded")
    record = order.direct_orders[0]["record_id"]
    assert wo.verify_trade(order, record, MOMENT + timedelta(minutes=30)) == [] and wo.verify_trade(order, record, MOMENT + timedelta(minutes=31))[0].startswith(wo.EXPIRY_PREFIX)


# ---------------------------------------------------------------- the real Postgres: the desk and the session, end to end, on throwaway accounts

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
    """The real local Postgres for one test: throwaway accounts that are deleted at the end, and a check that the three sample accounts did not move."""

    def __init__(self, store):
        self.store, self.made, self.before = store, [], self.samples(store)

    @staticmethod
    def samples(store):
        out = {}
        for seed in SAMPLES:
            account_id = store.account_id_for_seed(seed)
            out[seed] = (store.account_balance(account_id), store.list_portfolio(account_id).model_dump_json(), store.number_trades(account_id, *YEAR),
                         store.number_approved_recommendations(account_id, *YEAR), store.number_denied_recommendations(account_id, *YEAR))
        return out

    def account(self, cash: str, name: str = "Direct Order Test") -> int:
        account_id = self.store.create_account(name, "moderate", age=70)
        self.made.append(account_id)
        self.store.add_funds(account_id, Decimal(cash), FRIDAY)
        return account_id


@pytest.fixture
def db(store):
    book = Book(store)
    try:
        yield book
    finally:
        for account_id in book.made:
            with suppress(ValueError):
                store.delete_account(account_id)
    left = [a for a in book.made if _exists(store, a)]
    assert left == [], f"throwaway accounts left behind: {left}"
    assert book.samples(store) == book.before, "a sample account moved"


def _exists(store, account_id) -> bool:
    try:
        store.get_user(account_id)
        return True
    except ValueError:
        return False


def never(name):
    """A factory that FAILS THE TEST if the desk builds the agent: a direct order must not reach it."""
    def factory(order):
        raise AssertionError(f"a direct order must not reach the {name}")
    return factory


def run_desk(db, account_id, typed=BUY_50, confirm=lambda intent: True, approver=None, mode=None):
    """The REAL LangGraph desk over the real Account Manager (its model's replies are the recorded real ones, its symbols the recorded real directory), the real Data Clerk's stored closes, the real Trader without a model and the real Postgres book."""
    request = {"ask": {"text": typed, "default_count": 3, "fetch": False}, "account": {"account": account_id, **({"mode": mode} if mode else {})}, "trade": {"mode": "paper", "book": True, "model": False}}
    return desk.run_desk(request, clerk_factory=never("Data Clerk's node"), advisor_factory=never("Advisor"), critic_factory=never("Critic"),
                         funds_factory=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()), with_critic=True, with_trader=True,
                         trader_factory=lambda order: trader(), store_factory=lambda: db.store, confirm=confirm, approver=approver)


def test_the_desk_takes_a_direct_order_from_the_account_manager_to_the_trader_and_skips_the_data_clerks_node_the_advisor_and_the_critic(db):
    account_id = db.account("30000")
    state = run_desk(db, account_id)
    order, price = state["order"], close_of("MSFT")
    assert state["errors"] == [] and state["path"] == ["account_manager", "account_manager_direct", "trader"] and order.status == "traded"
    assert order.recommendations == [] and order.reviews == [] and order.critic_runs == [] and order.advisor_runs == [] and order.errors == [] and order.data.sources == []
    assert [(r["symbol"], r["company"]) for r in order.data.companies] == [("MSFT", "Microsoft Corp")] and order.direct_orders[0]["company"] == "Microsoft Corp"      # the company name and the symbol are on the work order
    (t,) = order.trades
    assert (t["status"], t["fill"]["shares"], t["review_id"], t["booked"]) == ("filled", 50, None, True) and wo.verify_trades(order) == []
    held = {p.symbol: p.shares for p in db.store.list_portfolio(account_id).positions}
    assert held == {"MSFT": 50} and db.store.account_balance(account_id) == Decimal("30000") - Decimal(str(t["fill"]["amount"])) and abs(t["fill"]["price"] / price - 1) < 0.002
    assert float(db.store.account_balance(account_id)) > 3000                                                    # money is left
    assert db.store.number_trades(account_id, *YEAR) == 1 and db.store.number_approved_recommendations(account_id, *YEAR) == 1     # the decision is saved in the book as the user's yes
    assert desk.reached_its_end(state)


def test_a_company_the_user_names_is_found_by_the_data_clerk_confirmed_in_ask_mode_and_traded_under_its_symbol(db):
    account_id = db.account("30000")
    cards = []
    state = run_desk(db, account_id, BUY_COMPANY, approver=lambda card: cards.append(card) or True)
    order = state["order"]
    assert state["errors"] == [] and state["path"] == ["account_manager", "account_manager_direct", "trader"] and order.status == "traded"
    assert order.direct_orders[0]["typed_company"] == "Microsoft" and order.direct_orders[0]["symbol"] == "MSFT" and len(cards) == 1 and "Microsoft Corporation" in cards[0]["text"]
    assert {p.symbol: p.shares for p in db.store.list_portfolio(account_id).positions} == {"MSFT": 50}


def test_a_no_to_the_confirmation_of_a_named_company_trades_nothing_and_in_auto_mode_nobody_is_asked(db):
    account_id = db.account("30000")
    denied = run_desk(db, account_id, BUY_COMPANY, approver=lambda card: False)
    assert denied["path"] == ["account_manager", "account_manager_direct"] and denied["order"].trade_decisions[0]["will_trade"] == "no_user_denied" and denied["order"].trades == []
    assert db.store.account_balance(account_id) == Decimal("30000")
    auto = run_desk(db, account_id, BUY_COMPANY, approver=lambda card: pytest.fail("Auto mode asks nobody"), mode="auto")
    assert auto["order"].status == "traded" and auto["order"].trades[0]["fill"]["shares"] == 50 and {p.symbol: p.shares for p in db.store.list_portfolio(account_id).positions} == {"MSFT": 50}


def test_with_twenty_thousand_dollars_the_order_is_refused_whole_the_account_is_unchanged_and_the_report_says_why(db):
    account_id = db.account("20000")
    state = run_desk(db, account_id)
    order = state["order"]
    (d,) = order.trade_decisions
    assert state["path"] == ["account_manager", "account_manager_direct"] and order.status == "decided" and order.trades == [] and d["will_trade"] == "no_funds"
    assert db.store.account_balance(account_id) == Decimal("20000") and db.store.list_portfolio(account_id).positions == [] and db.store.number_trades(account_id, *YEAR) == 0
    text = desk.report(state)
    assert "DIRECT ORDER" in text and "buy 50 shares of Microsoft Corp (MSFT)" in text and "NOT PLACED (no funds)" in text and f"the most it pays for is {int(20000 // close_of('MSFT'))} shares" in text
    assert "ADVISOR" not in text and "CRITIC" not in text


def test_a_stock_the_data_clerk_cannot_find_stops_the_desk_at_its_first_node_with_the_data_clerks_message(db):
    account_id = db.account("30000")
    state = desk.run_desk({"ask": {"text": "Buy 5 shares of ZZZZ"}, "account": {"account": account_id}, "trade": {"mode": "paper", "book": True, "model": False}}, clerk_factory=never("Data Clerk's node"),
                          advisor_factory=never("Advisor"), critic_factory=never("Critic"), funds_factory=lambda: AccountManager(model=Replay([form(orders=[{"side": "buy", "symbol": "ZZZZ", "shares": 5}])]), model_id="recorded", clerk=clerk()),
                          with_critic=True, with_trader=True, trader_factory=lambda order: trader(), store_factory=lambda: db.store, confirm=lambda intent: True)
    assert state["path"] == ["account_manager"] and any("Symbol: ZZZZ is not found" in e for e in state["errors"]) and state["order"].direct_orders == [] and db.store.account_balance(account_id) == Decimal("30000")


def test_a_direct_order_needs_the_desk_that_decides_and_trades(db):
    account_id = db.account("1000")
    state = desk.run_desk({"ask": {"text": BUY_50}, "account": {"account": account_id}}, clerk_factory=never("Data Clerk"), advisor_factory=never("Advisor"),
                          funds_factory=lambda: AccountManager(model=Replay(FORMS[BUY_50]["replies"]), model_id="recorded", clerk=clerk()), store_factory=lambda: db.store, confirm=lambda intent: True)
    assert state["order"] is None and state["path"] == ["account_manager"] and "needs the desk that decides and trades" in state["errors"][0]


def test_a_user_who_does_not_confirm_what_was_understood_trades_nothing(db):
    account_id = db.account("30000")
    state = run_desk(db, account_id, confirm=lambda intent: False)
    assert state["order"] is None and "not confirmed" in state["errors"][0] and db.store.account_balance(account_id) == Decimal("30000")


class Forbidden:
    """An Advisor or a Critic the session must never use: anything asked of it fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"a direct order must not use the Advisor's or the Critic's {name}")


def session_parts(db, typed=BUY_50):
    return session.Parts(store=db.store, clerk=clerk(), advisor=Forbidden(), critic=Forbidden(), trader=trader(), new_manager=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()))


def keep_track(db, lines: list[str]) -> str:
    """The session keeps the account it created and says so ("account 41 is kept: ..."): the test deletes it at its end. Returns the whole output."""
    import re
    text = "\n".join(lines)
    db.made.extend(int(n) for n in re.findall(r"account (\d+) is kept", text))
    return text


def run_session(db, typed, cash, mode, ask):
    args = session.build_parser().parse_args(["--name", "Direct Order Test", "--age", "70", "--risk", "moderate", "--cash", cash, "--mode", mode, "--prompt", typed, "--yes"])
    lines: list[str] = []
    code = session.run_session(args, session.check_args(session.build_parser(), args), session_parts(db, typed), ask=ask, out=lines.append)
    return code, keep_track(db, lines)


def test_the_session_of_the_user_story_creates_the_user_takes_the_direct_order_and_shows_the_portfolio_after_the_trade(db):
    """Create a user with $30,000, who wants 50 shares of MSFT: there is money left, and the session shows the portfolio after the trade."""
    code, text = run_session(db, BUY_50, "30000", "ask", lambda prompt: pytest.fail(f"the user was asked {prompt!r}: a ticker the user typed is asked about by nobody"))
    price = close_of("MSFT")
    assert code == 0 and "buy 50 shares of MSFT (your own order" in text and "the Data Clerk found: you typed 'MSFT': Microsoft Corp (MSFT), appended to the work order" in text
    assert "the Account Manager priced the trades you named" in text and "APPROVED: your own order" in text
    assert "RECOMMENDATIONS" not in text and not any(skipped in text for skipped in ("the Data Clerk found the companies", "the Advisor ranked", "the Critic reviewed", "the Account Manager decided what may trade"))      # none of the steps a direct order skips ran
    left = float(db.store.account_balance(db.made[-1]))
    assert "FILLED 50" in text and "ACCOUNT  after the session" in text and "MSFT" in text and f"${left:,.2f}" in text and left > 3000 and abs(left - (30000 - 50 * price)) < 30000 * 0.002 and "SESSION  PASSED" in text


def test_the_session_asks_the_user_to_confirm_a_company_they_named_in_ask_mode_and_places_it_on_a_yes(db):
    questions = []

    def ask(prompt):
        questions.append(prompt)
        return "y"

    code, text = run_session(db, BUY_COMPANY, "30000", "ask", ask)
    assert code == 0 and len(questions) == 1 and questions[0].startswith("Is this the order you meant: buy 50 shares of Microsoft Corporation (MSFT)")
    assert "YOU NAMED A COMPANY, NOT A TICKER" in text and "Microsoft Corporation" in text and "FILLED 50" in text and "SESSION  PASSED" in text


def test_the_session_places_nothing_when_the_user_says_no_to_the_company_the_data_clerk_found(db):
    code, text = run_session(db, BUY_COMPANY, "30000", "ask", lambda prompt: "n")
    assert code == 1 and "NOT PLACED (you did not confirm it)" in text and "SESSION  the portfolio did NOT change" in text and "holdings  none" in text


def test_the_session_in_auto_mode_uses_the_symbol_the_data_clerk_returned_and_asks_nobody(db):
    code, text = run_session(db, BUY_COMPANY, "30000", "auto", lambda prompt: pytest.fail(f"Auto mode asks nobody, and it asked {prompt!r}"))
    assert code == 0 and "FILLED 50" in text and "YOU NAMED A COMPANY" not in text and "SESSION  PASSED" in text


def test_the_session_with_twenty_thousand_dollars_says_why_nothing_was_bought(db):
    code, text = run_session(db, BUY_50, "20000", "ask", lambda prompt: pytest.fail("a direct order asks nothing"))
    assert code == 1 and "NOT PLACED (no funds)" in text and "SESSION  the portfolio did NOT change" in text and "cash $20,000.00 does not pay for 50 shares of MSFT" in text and "holdings  none" in text


# ---------------------------------------------------------------- a stock named with NO number of shares is still a direct order (design choices entry 98; Jacob, 2026-11-15)

def sized_order(typed: str, portfolio: Portfolio = BILL, mode: str = "auto", approver=None, answers: str | None = None, profile: UserProfile = BILL_PROFILE):
    """The real flow for a request that names a stock and no number: the form of the real model's recorded reply, the Account Manager appending the Data Clerk's names, then deciding (and, in Ask mode, asking the user
    how many with the real model's recorded replies to `answers`' scenario). Returns (order, the Account Manager, the channel to the user, the cards the approver was shown)."""
    intent = it.parse_intent(real_form(typed), typed)
    order = wo.new_work_order(typed, requester="desk", action_type="trade")
    model = AskReplay(RECORDED[answers]["replies"]) if answers else object()
    channel = channel_of(RECORDED[answers]["answers"]) if answers else None
    am = AccountManager(model=model, model_id="direct", clerk=clerk(), profile=profile)
    assert am.name_the_stocks(order, intent.orders, typed, MOMENT) == []
    cards = []
    out = am.decide_direct(order, portfolio, MOMENT, mode, approver or (lambda card: cards.append(card) or True), clarify=channel, said_how_much=intent.said_how_much(), request_text=typed,
                           balance="even" if intent.allocation == "even" else None, budget=intent.budget)
    return out, am, channel, cards


@pytest.mark.parametrize("typed, expected, budget", [
    ("I want to buy MSFT", [("buy", "MSFT", None, None)], None),
    ("Buy Microsoft", [("buy", None, "Microsoft", None)], None),
    ("I want to buy MSFT and TSLA and spend $6,000", [("buy", "MSFT", None, None), ("buy", "TSLA", None, None)], 6000),
    ("sell MSFT", [("sell", "MSFT", None, None)], None),
    ("Buy $5,000 of MSFT", [("buy", "MSFT", None, None)], 5000),
])
def test_the_real_models_reply_to_a_stock_named_with_no_number_is_a_direct_order_with_no_shares(typed, expected, budget):
    intent = it.parse_intent(real_form(typed), typed)
    assert intent.action == "order" and [(o.side, o.symbol, o.company, o.shares) for o in intent.orders] == expected and intent.budget == budget
    assert "you did not say how many shares" in intent.describe() and intent.said_how_much() is (budget is not None)       # with dollars said, nothing is asked; with none, the Account Manager asks in Ask mode


def test_a_request_to_buy_named_stocks_is_a_direct_order_by_the_words_whatever_the_model_said():
    """Design choices entry 98: the program checks the user's own words, so the model's mood does not decide it. The real model read "I want to buy MSFT and TSLA and spend $6,000" as "buy" (symbols and a budget, the
    choice left to the desk); the words name two stocks, ask for no recommendation and have a buying word, so the Account Manager makes it a direct order with the budget."""
    typed = "I want to buy MSFT and TSLA and spend $6,000"
    assert json.loads(real_form(typed))["action"] == "buy"
    intent = it.parse_intent(real_form(typed), typed)
    assert intent.action == "order" and [(o.side, o.symbol, o.shares) for o in intent.orders] == [("buy", "MSFT", None), ("buy", "TSLA", None)] and intent.budget == 6000
    assert it.parse_intent(real_form("I want to buy MU, SNDK and AMD if the recommendations pass"), "I want to buy MU, SNDK and AMD if the recommendations pass").action == "buy"      # it asks for the recommendations


def test_a_request_for_recommendations_is_not_a_direct_order_even_when_a_model_says_so():
    for typed in ("I want to buy MU, SNDK and AMD if the recommendations pass", "I want to buy the three most recommended oil stocks"):
        assert it.parse_intent(real_form(typed), typed).action == "buy"                                                     # the real model's own reading
    typed = "I want to buy MSFT if the recommendations pass"
    wrong = it.parse_intent(form(orders=[{"side": "buy", "symbol": "MSFT", "shares": None}]), typed)                       # a form that makes it an order is made a request to buy and recommend by the words
    assert wrong.action == "buy" and wrong.orders == [] and wrong.symbols == ["MSFT"]
    with pytest.raises(ValueError, match="asked for recommendations or advice"):
        it.check_order(it.DirectOrder(side="buy", symbol="MSFT"), typed)
    with pytest.raises(ValueError, match="you typed 50 shares of MSFT: put the number in shares"):
        it.parse_intent(form(orders=[{"side": "buy", "symbol": "MSFT", "shares": None}]), BUY_50)                           # null shares when the words give a number is refused


def test_in_auto_mode_a_named_stock_with_no_number_gets_what_the_cash_allows_and_nobody_is_asked():
    order, _, _, cards = sized_order("I want to buy MSFT", mode="auto")
    (d,), (entry,) = order.trade_decisions, order.direct_orders
    price = close_of("MSFT")
    assert (d["will_trade"], d["origin"], d["action"], d["price"]) == ("yes_user", "direct", "open", price) and d["shares"] == int(20000.0 // price) == entry["shares"] and entry["how_many"] == "cash" and cards == []
    assert wo.verify_trade(order, d["record_id"], MOMENT) == [] and "as much as your cash allows" in d["reason"]                # the user's record holds the number the Account Manager worked out
    assert "no profile limit" not in d["reason"] and order.status == "decided" and order.errors == []


def test_in_auto_mode_the_dollars_the_user_said_are_the_budget_and_an_even_split_is_what_they_asked():
    order, _, _, _ = sized_order("I want to buy MSFT and TSLA and spend $6,000", mode="auto")
    msft, tsla = order.trade_decisions
    assert (msft["symbol"], tsla["symbol"]) == ("MSFT", "TSLA") and msft["shares"] >= 1 and tsla["shares"] >= 1       # a share of each in turn until the $6,000 is spent
    assert msft["est_cost"] + tsla["est_cost"] <= 6000.0 and 6000.0 - (msft["est_cost"] + tsla["est_cost"]) < max(close_of("MSFT"), close_of("TSLA")) and {e["how_many"] for e in order.direct_orders} == {"budget"}
    single, _, _, _ = sized_order("Buy $5,000 of MSFT", mode="auto")
    assert single.trade_decisions[0]["shares"] == int(5000.0 // close_of("MSFT")) and single.direct_orders[0]["how_many"] == "budget"


def test_in_ask_mode_the_user_is_asked_for_the_number_then_shown_the_order_and_says_yes():
    order, _, channel, cards = sized_order("I want to buy MSFT", mode="ask", answers="shares_each_raise_yes")               # the real model's recorded replies to "10 shares of each"
    (d,), (entry,) = order.trade_decisions, order.direct_orders
    assert len(channel.cards) == 1 and "How much" in channel.cards[0]["question"]                                              # asked once, before anything was proposed
    assert d["shares"] == 10 == entry["shares"] and entry["how_many"] == "asked" and d["will_trade"] == "yes_user" and d["asked_user"] is True and wo.verify_trade(order, d["record_id"], MOMENT) == []
    assert len(cards) == 1 and "YOU DID NOT SAY HOW MANY SHARES" in cards[0]["text"] and cards[0]["shares"] == 10 and cards[0]["question"].startswith("Is this the order you meant: buy 10 shares")


def test_in_ask_mode_a_no_to_the_order_places_nothing_and_no_answer_is_pending():
    order, *_ = sized_order("I want to buy MSFT", mode="ask", answers="shares_each_raise_yes", approver=lambda card: False)
    assert order.trade_decisions[0]["will_trade"] == "no_user_denied" and not any(d["will_trade"] in wo.TRADING for d in order.trade_decisions)
    order, *_ = sized_order("I want to buy MSFT", mode="ask", answers="shares_each_raise_yes", approver=lambda card: None)
    assert order.trade_decisions[0]["will_trade"] == "pending_user"


def test_a_request_with_no_channel_to_the_user_in_ask_mode_proposes_what_the_cash_allows_and_waits_for_a_yes():
    intent = it.parse_intent(real_form("I want to buy MSFT"), "I want to buy MSFT")
    order = wo.new_work_order("I want to buy MSFT", requester="desk", action_type="trade")
    am = account(BILL_PROFILE)
    assert am.name_the_stocks(order, intent.orders, "I want to buy MSFT", MOMENT) == []
    out = am.decide_direct(order, BILL, MOMENT, "ask", None, clarify=None, said_how_much=False, request_text="I want to buy MSFT")
    assert out.trade_decisions[0]["will_trade"] == "pending_user" and out.trade_decisions[0]["shares"] == int(20000.0 // close_of("MSFT"))


def test_a_sale_with_no_number_is_not_placed_and_says_to_name_how_many():
    portfolio = Portfolio(user="Bill Gates", cash_avail=1000.0, positions=[Position(symbol="MSFT", shares=40, cost_basis=300.0, acquired=date(2025, 1, 2))])
    order, *_ = sized_order("sell MSFT", portfolio=portfolio)
    d = order.trade_decisions[0]
    assert d["will_trade"] == "no_system_denied" and d["shares"] == 0 and "say how many to sell" in d["reason"] and "holds 40 shares of MSFT" in d["reason"] and order.direct_orders[0]["shares"] is None
    assert wo.verify_trade(order, d["record_id"], MOMENT) != []


def test_a_decision_for_an_order_whose_record_has_no_number_is_refused_by_the_trader_so_the_account_managers_sizing_is_what_permits_it():
    order, *_ = sized_order("I want to buy MSFT", mode="auto")
    record_id = order.trade_decisions[0]["record_id"]
    order.direct_orders[0]["shares"] = None
    assert any("holds no number of shares" in r for r in wo.verify_trade(order, record_id, MOMENT))


def test_the_trader_buys_what_the_account_manager_worked_out_for_a_stock_named_with_no_number():
    order, am, *_ = sized_order("I want to buy MSFT", mode="auto")
    am.perform_trades(order, BILL, trader(), MOMENT)
    (t,) = order.trades
    assert order.status == "traded" and t["status"] == "filled" and t["side"] == "buy" and t["shares"] == order.direct_orders[0]["shares"] == int(20000.0 // close_of("MSFT")) and wo.verify_trades(order) == []
