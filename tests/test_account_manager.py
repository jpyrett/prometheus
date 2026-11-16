"""Tests for prometheus.agents.account_manager.account_manager, DataClerk.request_quotes and the desk's funds node (design choices entry 42).

Run:  pytest tests/test_account_manager.py -q

A scripted model stands in for the chat model and a fake toolbox for Finnhub, so each test decides what the "LLM" proposes and what the quotes are. The Data Clerk, its `get_quote`
tool, the work order, the tree search and the report are the REAL code. No network, no key, no cost.
"""
import json
from datetime import date
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage

from prometheus.agents.account_manager import account_manager as am
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import portfolio as pf
from prometheus.common import work_order as wo
from prometheus.langgraph import desk
from prometheus.prompts import account_manager as prompts
from prometheus.reasoning import raise_funds as rf
from test_desk_graph import FakeAdvisor, FakeClerk, FakeResolver

PRICES = {"MSFT": Decimal("420.0"), "TSLA": Decimal("250.0"), "GM": Decimal("55.0"), "MU": Decimal("410.0")}
TODAY = date(2026, 11, 3)


class QuoteToolbox:
    """Finnhub's /quote, as a table. Records every symbol it is asked for; a symbol not in the table fails like an unknown one does."""

    def __init__(self, prices=None):
        self.prices, self.asked = dict(PRICES if prices is None else prices), []

    def get_quote(self, symbol):
        self.asked.append(symbol)
        if symbol not in self.prices:
            raise RuntimeError(f"No quote returned for {symbol} (Finnhub answers an unknown symbol with zeros)")
        p = self.prices[symbol]
        return {"symbol": symbol, "price": p, "change": 1.0, "change_percent": 0.25, "high": p, "low": p, "open": p, "previous_close": p,
                "as_of_utc": "2026-11-03T21:00:00Z", "source": "Finnhub /quote"}


class Scripted:
    """Anything with invoke. Returns the next scripted reply (text, a message, or an exception to raise) and records every request."""
    model_name = "scripted/account-manager"

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply, usage_metadata={"input_tokens": 400, "output_tokens": 60, "total_tokens": 460})


def thoughts(*items) -> str:
    return json.dumps({"thoughts": [{"symbol": s, "shares": n, "idea": f"idea for {s}"} for s, n in items]})


def manager(*replies, prices=None):
    toolbox = QuoteToolbox(prices)
    clerk = DataClerk(toolbox=toolbox, model=object())  # the real Data Clerk; only the toolbox under it is fake, and its own model is never called for a quote
    model = Scripted(*replies)
    agent = am.AccountManager.classic(model=model, clerk=clerk)
    agent.toolbox, agent.scripted = toolbox, model
    return agent


@pytest.fixture
def portfolio():
    """The sample portfolio with no free cash, so that the round numbers below hold; the tests that use its real cash_avail (51.23) load the file themselves."""
    return pf.load_portfolio(am.DEFAULT_PORTFOLIO).model_copy(update={"cash_avail": Decimal("0")})


# ---------------------------------------------------------------- the Data Clerk answers a fetch_quote order

def quote_order(symbols):
    return wo.new_work_order("prices", requester="account_manager", action_type="fetch_quote", subject=symbols, live=True)


def clerk_with(toolbox):
    return DataClerk(toolbox=toolbox, model=object())


@pytest.mark.parametrize("order, message", [
    (wo.new_work_order("x", action_type="search_news", subject=["MU"]), "is not 'fetch_quote'"),
    (wo.new_work_order("x", action_type="fetch_quote", subject=None), "needs one or more ticker symbols"),
    (wo.new_work_order("x", action_type="fetch_quote", subject=[]), "needs one or more ticker symbols"),
    (wo.new_work_order("x", action_type="fetch_quote", subject="MU"), None),  # one symbol as text is fine
    (wo.new_work_order("x", action_type="fetch_quote", subject=[f"S{i}" for i in range(31)]), "at most 30 symbols"),
    (wo.new_work_order("x", action_type="fetch_quote", subject=[f"S{i}" for i in range(30)]), None),  # 30 are fine: the complex scenario asks for 11
])
def test_a_quote_order_that_is_not_one_is_refused_with_the_reason(order, message):
    reasons = wo.verify_quote_request(order)
    assert (message is None and reasons == []) or any(message in r for r in reasons)


def test_an_order_that_already_has_its_data_comes_back_unchanged():
    toolbox = QuoteToolbox()
    clerk = clerk_with(toolbox)
    order = clerk.request_quotes(quote_order(["MU"]))
    again = clerk.request_quotes(order)
    assert again is order and toolbox.asked == ["MU"]  # a repeated request costs nothing


def test_the_search_path_still_refuses_fetch_quote_it_is_a_different_request():
    assert any("not handled by the Data Clerk's request_data yet" in r for r in wo.verify_request(quote_order(["MU"])))


# ---------------------------------------------------------------- the thought generator's reply


@pytest.mark.parametrize("text, reason", [("", "no JSON object found"), ("no json here", "no JSON object found"), ("{not json}", "not valid JSON"),
                                          ('{"recommendations": []}', "no thoughts list"), ('{"thoughts": "GM"}', "no thoughts list"), ("[1, 2]", "no JSON object found")])
def test_a_reply_that_is_not_the_shape_asked_for_is_refused_with_a_reason(text, reason):
    with pytest.raises(ValueError, match=reason):
        am.parse_thoughts(text)


# ---------------------------------------------------------------- RaiseFunds, end to end with a scripted model


def test_a_reply_that_is_not_json_is_retried_once_and_counted(portfolio):
    agent = manager("I would sell some Tesla.", thoughts(("TSLA", 4)))
    result = agent.raise_funds(portfolio, "MU", 2, today=TODAY)
    assert result["status"] == "plan" and result["run"]["calls"] == 2 and result["run"]["retries"] == 1
    retry = agent.scripted.seen[1]
    assert "not valid JSON in the required shape (no JSON object found)" in retry[-1].content and retry[-2].content == "I would sell some Tesla."


def test_a_model_that_never_gives_json_ends_in_infeasible_with_the_error_and_no_traceback(portfolio):
    agent = manager("nope", "still nope")
    result = agent.raise_funds(portfolio, "MU", 2, today=TODAY)
    assert result["status"] == "infeasible" and result["plan"] is None and result["errors"] == ["expanding n0 failed (ValueError)"] and result["run"]["retries"] == 2


def test_a_symbol_the_data_clerk_cannot_quote_stops_the_request_before_any_model_call(portfolio):
    agent = manager(prices={"MU": 410.0, "MSFT": 420.0, "TSLA": 250.0})  # GM has no quote
    result = agent.raise_funds(portfolio, "MU", 2, today=TODAY)
    assert result["status"] == "no_prices" and any(e.startswith("GM: no quote") for e in result["errors"]) and agent.scripted.seen == []
    assert set(result["quotes"]) == {"MU", "MSFT", "TSLA"} and "quotes that did come back: MU 410.00" in am.report(result)


def test_a_bad_request_is_refused_without_asking_the_data_clerk_or_the_model(portfolio):
    for target, shares in (("../etc", 2), ("MU", 0), ("MU", True), ("MU", 1.5), (None, 2)):
        agent = manager()
        result = agent.raise_funds(portfolio, target, shares, today=TODAY)
        assert result["status"] == "refused" and result["errors"] and agent.toolbox.asked == [] and agent.scripted.seen == []


def test_buying_more_than_the_portfolio_can_pay_for_is_infeasible_and_says_how_many_could_be_bought(portfolio):
    agent = manager(thoughts(("MSFT", 50), ("TSLA", 123), ("GM", 60)))
    result = agent.raise_funds(portfolio, "MU", 500, today=TODAY)
    assert result["status"] == "infeasible" and result["max_affordable"] == 134 and "INFEASIBLE" in am.report(result)


def test_the_default_account_manager_uses_the_account_manager_role_and_names_a_missing_key(monkeypatch):
    from prometheus.models import chat
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(chat, "setting", lambda name, default="": default if name != "OPENROUTER_API_KEY" else "")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        am.AccountManager()
    assert chat.model_name("account_manager") == "openai/gpt-oss-120b"


# ---------------------------------------------------------------- the best recommendation

def order_with(*recommendations):
    order = wo.new_work_order("x")
    order.recommendations = list(recommendations)
    return order


def rec(symbol, score, supported=True):
    return {"symbol": symbol, "score": score, "supported": supported}


# ---------------------------------------------------------------- the desk: the funds node

def funds_desk(model_replies, request=None, **options):
    clerk = DataClerk(toolbox=QuoteToolbox(), model=object())
    return desk.run_desk({"symbol": "MSFT", **(request or {})}, clerk_factory=lambda order: FakeClerk(), advisor_factory=lambda order: FakeAdvisor(), resolver=FakeResolver(),
                         with_funds=True, funds_factory=lambda: am.AccountManager.classic(model=Scripted(*model_replies), clerk=clerk), **options)


def test_the_desk_raises_funds_for_the_best_recommendation_and_the_work_order_is_not_changed():
    funds_request = {"portfolio": pf.load_portfolio(am.DEFAULT_PORTFOLIO), "shares": 2}
    state = funds_desk([thoughts(("MSFT", 2), ("TSLA", 2)), thoughts(("GM", 8))], request={"funds": funds_request, "symbol": "MSFT"})  # MSFT is the target: selling it is rejected
    assert state["path"] == ["account_manager", "data_clerk", "advisor", "account_manager_funds"] and state["order"].status == "recommended"
    funds = state["funds"]
    assert funds["status"] == "plan" and funds["target"] == "MSFT" and "account_manager_funds" in state["seconds"]  # MSFT: FakeAdvisor's one recommendation (score 67)
    assert funds["trace"][0]["children"][0]["status"] == "rejected" and funds["after"]["shares"]["MSFT"] == 52  # the 50 held plus the 2 bought
    assert "ACCOUNT MANAGER: RAISE FUNDS" in desk.report(state) and "FINAL ANSWER" in desk.report(state)


def test_the_desk_reads_the_portfolio_from_a_file_when_given_a_path():
    state = funds_desk([thoughts(("TSLA", 4))], request={"funds": {"portfolio": str(am.DEFAULT_PORTFOLIO), "shares": 2, "buy": "MU"}})
    assert state["funds"]["status"] == "plan"


def test_a_missing_portfolio_or_a_bad_file_is_a_refusal_not_a_crash(tmp_path):
    for spec in ({"shares": 2}, {"portfolio": str(tmp_path / "nope.json"), "shares": 2}):
        funds = funds_desk([], request={"funds": spec})["funds"]
        assert funds["status"] == "refused" and "could not start" in funds["errors"][0]


def test_no_bullish_recommendation_and_no_named_symbol_is_a_refusal():
    class Bearish(FakeAdvisor):
        def advise(self, order):
            order = super().advise(order)
            order.recommendations[0].update(score=33)
            return order

    state = desk.run_desk({"symbol": "MSFT", "funds": {"portfolio": str(am.DEFAULT_PORTFOLIO), "shares": 2}}, clerk_factory=lambda order: FakeClerk(), advisor_factory=lambda order: Bearish(),
                          resolver=FakeResolver(), with_funds=True, funds_factory=lambda: pytest.fail("the Account Manager must not be built"))
    assert state["funds"]["status"] == "refused" and "no recommendation with a verified bullish score" in state["funds"]["errors"][0]


# ---------------------------------------------------------------- a fill stock, explore, and the command line (entry 44)

def test_a_fill_stock_that_is_the_stock_to_buy_is_refused(portfolio):
    agent = manager()
    result = agent.raise_funds(portfolio, "MU", 2, today=TODAY, fill="mu")
    assert result["status"] == "refused" and "another symbol" in result["errors"][0] and agent.toolbox.asked == [] and agent.scripted.seen == []


def test_a_fill_stock_with_no_quote_stops_before_any_model_call(portfolio):
    agent = manager()
    result = agent.raise_funds(portfolio, "MU", 2, today=TODAY, fill="NVDA")  # the fake toolbox has no NVDA
    assert result["status"] == "no_prices" and agent.scripted.seen == []


@pytest.mark.parametrize("argv", [["--buy", "MU", "--fill", "NVDA", "--buy", "SNDK"], ["--buy", "MU:two"], ["--buy", "MU", "--portfolio", "/nonexistent.json"]])
def test_the_command_line_refuses_what_it_cannot_run_before_any_call(argv, capsys):
    assert am.main(argv) == 1
    assert capsys.readouterr().err.startswith("ERROR")


# ---------------------------------------------------------------- whole plans in the reply and in the prompt (entry 50)

def test_the_reply_may_hold_whole_plans_under_plans():
    assert am.parse_thoughts('{"plans": [{"sells": [{"symbol": "GM", "shares": 3}], "idea": "x"}]}') == [{"sells": [{"symbol": "GM", "shares": 3}], "idea": "x"}]


def test_the_prompt_for_whole_plans_asks_for_plans_lists_what_each_holding_can_raise_and_names_the_rules():
    p = rf.build_problem(pf.load_portfolio(am.DEFAULT_PORTFOLIO).model_copy(update={"cash_avail": Decimal("0")}), "MU", 2, PRICES, TODAY, only_green=True, never_liquidate=True, lean=True, whole_plans=True, green_first=True)
    system, user = prompts.system_prompt(3, True, None, True, True, 3, True, True, True), prompts.user_message(p, rf.root(p), [])
    assert 'Reply with up to 3 DIFFERENT plans' in system and '"plans"' in system and "Only holdings that show a profit" in system and "Never sell a position out" in system
    assert "The LAST sell of a plan must not sell more shares than the money still needed requires" in system and "sell the MOST PROFITABLE holdings first" in system
    assert "What each holding can raise if all its sellable shares are sold: " in user and "Give up to 3 different plans that each raise the money still needed." in user


def test_a_method_argument_overrides_the_default_and_the_default_deadline_and_parallel_reach_the_search():
    manager = am.AccountManager(model=Scripted(thoughts(("TSLA", 4))), clerk=clerk_with(QuoteToolbox()), only_green=False, never_liquidate=False)
    result = manager.raise_funds(pf.load_portfolio(am.DEFAULT_PORTFOLIO).model_copy(update={"cash_avail": Decimal("0")}), "MU", 2, today=TODAY, explore=False, green_first=False, lean=False, up_to=False)
    assert result["limits"]["deadline"] == 90.0 and result["status"] == "plan" and result["green_first"] is False
    assert am.AccountManager(model=object(), clerk=object(), deadline=None).deadline is None


def captured_options(monkeypatch, **arguments):
    """The options the Account Manager gives `get_chat_model` (no key, no network: the model is never built)."""
    seen = {}
    monkeypatch.setattr(am, "get_chat_model", lambda role, **options: seen.update(options) or object())
    am.AccountManager(clerk=object(), **arguments)
    return seen


