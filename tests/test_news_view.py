"""A REQUEST THAT ONLY ASKS FOR THE NEWS (design choices entry 149; design/news_view_spec.md, rules N1 to N11; Jacob, 2026-11-23: *"Show the news for Tesla"*, *"Whats Tesla doing the last week?"*).

REAL DATA ONLY. The Account Manager's model is a REPLAY of its REAL replies (`tests/fixtures/session_real/news_request_forms.json`, made by `scripts/capture_news_request_forms.py`): Jacob's four forms, a window in the words, a window over the limit, a theme, six
stocks, and three requests that look like news and are not. The Data Clerk's answer is the REAL one of the live session `tests/fixtures/news_real/news_tsla_week.json` ("What's Tesla doing the last week?", Ask mode, a throwaway account, the fetch REFUSED by
the test-only switch because a fetch writes `news/TSLA.txt`): 12 articles of the stored news, 9 cited. The Data Clerk is the real class: only its model (`ask`) gives back that recorded answer, so `request_news` (the check, the question, the news-only sweep, the
citation check and the status) runs for real. A form that a test writes itself ("a model that said ...") has the shape of the real replies and is for a FAULT. The only stand-in is the paid news call (`tools.fetch_news`), and only in the test that follows the fetch.
Nothing is fetched, no model is called, no quote is read: `nothing_is_fetched` (in `tests/conftest.py`, for every test) makes the toolbox's doors to the paid APIs raise, so that a news run, which ALWAYS asks its Data Clerk to fetch
(entry 149, N7), reaches a refusal or a failure that is a line in the order, never the network. (A first version of this module had no such guard and its route tests fetched TSLA's news for real: it wrote news/TSLA.txt and the vector store.)
"""
import copy
import json
from datetime import date
from pathlib import Path

import pytest
from handshake_helpers import MOMENT, PROFILE
from test_direct_order import YEAR, Replay, clerk, db, never, store  # noqa: F401  (db and store are fixtures: throwaway Postgres accounts)

from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.account_manager.config import load_config
from prometheus.agents.data_clerk.data_clerk_agent import NEWS_ONLY, DataClerk, Result
from prometheus.application import session
from prometheus.application.web import desk_run, views
from prometheus.application.web.app import create_app
from prometheus.application.web.emitter import Emitter
from prometheus.application.web.replay import Replay as Replayed
from prometheus.application.web.replay import load_capture
from prometheus.application.web.runs import Run
from prometheus.common import trading_clock
from prometheus.common import work_order as wo
from prometheus.langgraph import desk
from prometheus.rag.citations import SourceBook

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
FORMS = json.loads((FIXTURES / "news_request_forms.json").read_text(encoding="utf-8"))["requests"]
NEWS_REAL = Path(__file__).parent / "fixtures" / "news_real"      # not in session_real: the offline evaluation's sessions dataset (and its test) takes EVERY capture there that holds a work order, and a news run is no trading session to evaluate
CAPTURE = json.loads((NEWS_REAL / "news_tsla_week.json").read_text(encoding="utf-8"))
JACOBS = "What's Tesla doing the last week?"


def understood(typed: str) -> it.TradeIntent:
    """What the real Account Manager makes of the model's real replies to `typed` (it asks once more when code refuses the first, as it does live)."""
    return AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk(), profile=PROFILE).understand_request(typed)


def a_form(**changes) -> str:
    """A form of the shape the real model answers in, for a FAULT: the real model never wrote these."""
    return json.dumps({"action": "news", "topic": None, "symbols": ["TSLA"], "companies": [], "count": None, "allocation": "default", "budget": None, "each": False, "max_price": None, "min_price": None, "raise_funds": False, "orders": [], "news_days": None, **changes})


# ---------------------------------------------------------------- N1 to N3: the model reads it, code checks it, a program reads the window

@pytest.mark.parametrize("typed, named, days", [
    ("Show the news for Tesla", {"companies": ["Tesla"]}, 7),                               # Jacob's forms: no window, the default week
    ("Show the news for TSLA", {"symbols": ["TSLA"]}, 7),
    ("Whats Tesla's news?", {"companies": ["Tesla"]}, 7),
    (JACOBS, {"companies": ["Tesla"]}, 7),
    ("TSLA headlines for the past 3 days", {"symbols": ["TSLA"]}, 3),
    ("news on Apple and Microsoft this month", {"companies": ["Apple", "Microsoft"]}, 30),
    ("Show me NVDA news from the last 2 weeks", {"symbols": ["NVDA"]}, 14),
    ("What's happening with Tesla lately?", {"companies": ["Tesla"]}, 7),                    # the model gave no window: the default
])
def test_the_real_replies_to_the_news_requests_are_news_with_the_stocks_and_the_window_the_words_give(typed, named, days):
    intent = understood(typed)
    assert intent.action == "news" and {"symbols": intent.symbols, "companies": intent.companies} | {k: [] for k in ("symbols", "companies") if k not in named} == {"symbols": [], "companies": []} | named
    assert intent.news_days == days and intent.topic is None and intent.orders == [] and intent.budget is None and intent.count is None and intent.corrected == []
    assert intent.describe() == f"show the news of {', '.join([*intent.symbols, *intent.companies])} for the last {days} days (nothing will be bought)"


def test_a_theme_is_searched_as_a_term_and_a_window_over_the_limit_is_cut_and_said():
    theme = understood("news about quantum computing")
    assert (theme.action, theme.topic, theme.symbols, theme.companies, theme.news_days) == ("news", "quantum computing", [], [], 7) and theme.as_request() == ("quantum computing", "phrase")
    assert theme.describe() == "show the news about 'quantum computing' for the last 7 days (nothing will be bought)"
    cut = understood("Show the news for TSLA for the last 200 days")      # the real model wrote 200; the program reads 200 from the words and the limit is 90
    assert cut.news_days == load_config().news.max_days == 90 and cut.corrected == ["the news is kept for at most 90 days: showing 90"] and "(note: the news is kept for at most 90 days: showing 90)" in cut.describe()


def test_the_window_the_words_write_is_the_programs_and_wins_over_the_model_and_the_dump_has_no_window_for_any_other_request():
    typed = "TSLA headlines for the past 3 days"
    real = json.loads(FORMS[typed]["replies"][-1])
    assert real["news_days"] == 3
    assert it.parse_intent(a_form(news_days=14), typed).news_days == 3      # a model that said 14: the words say 3
    assert it.parse_intent(a_form(news_days=14), "Show the news for TSLA").news_days == 14 and it.parse_intent(a_form(news_days=None), "Show the news for TSLA").news_days == 7      # no window written: the model's, else the default
    for words, days in {"last 3 days": 3, "past 2 weeks": 14, "last week": 7, "past month": 30, "this week": 7, "this month": 30, "today": 1, "yesterday": 2, "past 5 months": 150, "last 1 day": 1, "this year": None, "last year": None, "last 0 days": None, "the last night": None}.items():
        assert it.news_window(f"news for TSLA {words}") == days
    assert it.news_window("today and the last week") == 7      # more than one span: the longest
    buy = json.loads(FORMS["buy 5 shares of TSLA"]["replies"][-1])
    assert "news_days" not in it.parse_intent(json.dumps({**buy, "news_days": 9}), "buy 5 shares of TSLA").model_dump()      # a window means nothing to any other request, and no dump has the key (the Proving Ground's cache keys do not change)


def test_the_three_requests_that_look_like_news_and_are_not_keep_the_old_actions_and_six_stocks_is_a_request_to_clarify():
    assert understood("what's Tesla's news, buy it if it's good").action == "buy"
    assert understood("buy 5 shares of TSLA").action == "order"
    with pytest.raises(ValueError, match="could not understand"):      # "should I buy Tesla?": the model's reply names nothing; it is not news and it is refused as it always was
        understood("should I buy Tesla?")
    with pytest.raises(it.NeedsClarification, match="at most 5 stocks and 6 were named"):      # the model is not asked again: it would drop one without telling the user (the first real capture did)
        understood("Show the news for TSLA, AAPL, MSFT, NVDA, AMD and META")
    assert len(FORMS["Show the news for TSLA, AAPL, MSFT, NVDA, AMD and META"]["replies"]) == 1


# ---------------------------------------------------------------- N2: what a news intent may not hold, each refusal once

@pytest.mark.parametrize("change, said, words", [
    ({"orders": [{"side": "buy", "symbol": "TSLA", "shares": 5, "dollars": None}]}, "orders", "buy 5 TSLA and show the news for TSLA"),
    ({"budget": 500}, "budget", "show the news for TSLA, spend $500"),
    ({"each": True}, "each", "show the news for TSLA, each $500"),
    ({"count": 3}, "count", "show the news for the top 3 TSLA"),
    ({"allocation": "even"}, "allocation", "show the news for TSLA split evenly"),
    ({"raise_funds": True}, "raise funds", "show the news for TSLA and raise funds"),
    ({"max_price": 50}, "max price", "show the news for TSLA under $50 a share"),
    ({"min_price": 20}, "min price", "show the news for TSLA over $20 a share"),
    ({"topic": "electric cars"}, "not both", "show the news for TSLA about electric cars"),
])
def test_a_news_form_that_holds_a_trade_a_number_or_both_a_theme_and_stocks_is_refused_with_its_reason(change, said, words):
    with pytest.raises(ValueError) as refused:
        it.parse_intent(a_form(**change), words)
    assert said in str(refused.value) and "nothing is bought" in str(refused.value) or "not both" in str(refused.value)


def test_no_stock_and_no_theme_a_stock_that_is_not_in_the_words_and_a_window_that_is_not_a_number_are_refused():
    with pytest.raises(ValueError, match="no theme, ticker or company"):
        it.parse_intent(a_form(symbols=[]), "show me the news")
    with pytest.raises(ValueError, match="the ticker TSLA is not in what you typed"):
        it.parse_intent(a_form(), "show the news for Apple")
    with pytest.raises(ValueError, match="whole number of days"):
        it.parse_intent(a_form(news_days="a week"), "show the news for TSLA")
    assert it.TradeIntent(request="x", action="buy", symbols=["TSLA"]).model_dump().keys().isdisjoint({"news_days"})
    with pytest.raises(ValueError, match="news_days"):
        it.TradeIntent(request="x", action="buy", symbols=["TSLA"], news_days=7)      # the window goes with the news only


def test_the_program_that_turns_a_buy_into_an_order_never_touches_a_news_intent():
    intent = it.parse_intent(a_form(), "Show the news for TSLA")
    assert it.as_direct_order(intent, "Show the news for TSLA") is None


# ---------------------------------------------------------------- N6 and N7: the Data Clerk's news request, and the fetch

def recorded_result() -> Result:
    """The REAL Result of the live session (what the Data Clerk's model and its search made): given back by `ask`, so that everything the Data Clerk does around its model runs for real."""
    data = CAPTURE["order"]["data"]
    fields = {k: data[k] for k in ("answer", "references", "sources", "problems", "notes", "no_supporting_text", "tool_calls", "turns", "tokens_in", "tokens_out", "sweep", "filing_evidence", "security_dropped")}
    return Result(question="", **fields)


class RecordedAnswer(DataClerk):
    """The real Data Clerk whose MODEL gives back the live session's answer: `ask` records how it was called and returns the recorded `Result`. `verify_request`, the question, the sweep spec, `attach_result` and the status are the real ones."""

    def __init__(self, **options):
        super().__init__(model=object(), **options)
        self.asked = []

    def ask(self, question, today=None, sweep=None, only=None):
        self.asked.append({"question": question, "today": today, "sweep": sweep, "only": only})
        return recorded_result()


def news_order(symbols=("TSLA",), days=7, subject=None, valid=True) -> wo.WorkOrder:
    """The work order the desk makes for the news (`desk.news_plan`): a search of the stored news in the window, `top_k` from the config, live (the latest news is fetched)."""
    cap = load_config().news
    order = wo.new_work_order(f"Summarize the news of {', '.join(symbols)} for the last {days} days", requester="desk", action_type="search_news", source="news", subject=list(symbols) if subject is None else subject, company="Tesla Inc", time_period=f"last {days} days",
                              since_days=days, top_k=cap.top_k, live=True)
    assert not valid or wo.verify_request(order) == []
    return order


def test_request_news_asks_for_a_summary_of_the_news_offers_the_news_search_only_and_sweeps_the_news_only():
    agent = RecordedAnswer()
    order = agent.request_news(news_order(), date(2026, 11, 23))
    (asked,) = agent.asked
    assert asked["only"] == NEWS_ONLY == ("retrieve_news",) and asked["today"] == date(2026, 11, 23)
    assert asked["sweep"]["corpora"] == ("news",) and asked["sweep"]["evidence"] is False and asked["sweep"]["tickers"] == ["TSLA"] and asked["sweep"]["since_days"] == 7 and asked["sweep"]["k"] == 12
    assert "Summarize the news of TSLA for the last 7 days" in asked["question"] and "subject: TSLA (Tesla Inc)" in asked["question"] and "use news from the last 7 days" in asked["question"] and "use at most 12 sources" in asked["question"]
    assert "call retrieve_news" in asked["question"] and "cite each source you use as [n]" in asked["question"] and "fetch" not in asked["question"] and "press release" not in asked["question"].lower()
    assert order.status == "data_ready" and order.errors == [] and order.data.answer == CAPTURE["order"]["data"]["answer"] and len(order.data.sources) == 12 and len(order.data.references) == 9      # the answer, its references and its sources are on the order as ever
    assert request_news_twice(agent, order) is order and len(agent.asked) == 1      # an order that has its data is not asked again


def request_news_twice(agent, order):
    return agent.request_news(order, date(2026, 11, 23))


def test_a_term_is_asked_across_every_company_and_a_request_the_data_clerk_may_not_take_comes_back_failed_with_the_reasons():
    agent = RecordedAnswer()
    theme = news_order(subject="quantum computing")
    agent.request_news(theme, date(2026, 11, 23))
    assert agent.asked[0]["sweep"]["tickers"] is None and "searching for the term across every company" in agent.asked[0]["question"] and "subject: quantum computing" in agent.asked[0]["question"]
    broken = news_order(days=500, valid=False)      # a window the news does not keep
    failed = agent.request_news(broken, date(2026, 11, 23))
    assert failed.status == "data_failed" and any("since_days" in e for e in failed.errors) and len(agent.asked) == 1


def test_the_news_sweep_searches_the_news_and_nothing_else_and_reads_no_filing_evidence_while_every_other_sweep_is_as_it_was():
    agent = RecordedAnswer()
    result = Result(question="q")
    agent._sweep(result, SourceBook(), agent._news_sweep_spec(news_order()))      # the model searched nothing: the code does, over the (empty) stored news of the tests
    assert list(result.sweep) == ["news"] and result.sweep["news"]["by"] == "code" and result.filing_evidence == {}
    every = Result(question="q")
    agent._sweep(every, SourceBook(), agent._sweep_spec(news_order()))      # an ordinary order: the three corpora and the filing evidence, as always
    assert list(every.sweep) == ["news", "press_releases", "filings", "filing_evidence"]


def test_the_latest_news_is_fetched_for_the_stocks_named_only_with_the_news_tool_and_nothing_else():
    class Paid:
        """The paid news call and nothing else: the one stand-in of these tests. Everything else the toolbox could fetch fails the test if it is touched."""
        def __init__(self):
            self.calls = []

        def fetch_news(self, symbol, from_date=7, **options):
            self.calls.append((symbol, from_date, options))
            return {"symbol": symbol, "fetched": 40, "written": 3, "duplicates": 37, "indexed": {}}

        def __getattr__(self, name):
            raise AssertionError(f"a news request fetched more than the news: {name}")

    agent = RecordedAnswer(allow_fetch=True)
    agent.tools.toolbox = Paid.__new__(Paid)      # fetch_daily_bars, fetch_press_releases and fetch_filings are reached through the tools' toolbox: any of them fails the test
    paid = agent.tools = Paid()
    assert agent.refresh_news(["TSLA", "$AAPL", "TSLA"], 7) == [{"step": "news", "symbol": "TSLA", "outcome": "ok", "detail": {"fetched": 40, "written": 3, "duplicates": 37}},
                                                                {"step": "news", "symbol": "AAPL", "outcome": "ok", "detail": {"fetched": 40, "written": 3, "duplicates": 37}}] and paid.calls == [("TSLA", 7, {}), ("AAPL", 7, {})]
    with pytest.raises(ValueError, match="refresh 1 to 15 symbols"):
        agent.refresh_news([], 7)


def test_a_fetch_that_fails_or_is_refused_is_a_line_and_never_an_exception_and_a_past_day_that_is_bound_refuses_it():
    class Failing:
        def fetch_news(self, symbol, from_date=7):
            raise RuntimeError("the news API said no")

    wanted = RecordedAnswer(allow_fetch=True)
    wanted.tools = Failing()
    (failed,) = wanted.refresh_news(["TSLA"], 7)
    assert (failed["outcome"], failed["symbol"]) == ("error", "TSLA") and failed["detail"] == "RuntimeError: the news API said no"
    with trading_clock.as_of(date(2026, 11, 13)):      # the Proving Ground: nothing is fetched while a past day is bound, whatever the Data Clerk may do
        (refused,) = wanted.refresh_news(["TSLA"], 7)
        assert refused["outcome"] == "refused" and trading_clock.NEVER_FETCHES in refused["detail"] and "2026-11-13" in refused["detail"]
    (none,) = RecordedAnswer().refresh_news(["TSLA"], 7)      # built without the right to fetch
    assert none["outcome"] == "refused" and "without allow_fetch" in none["detail"]
    plain = RecordedAnswer()
    fetching = plain.fetching(True)
    assert fetching is not plain and fetching.allow_fetch is True and plain.allow_fetch is False and fetching.tools is plain.tools and plain.fetching(False).allow_fetch is False      # a copy: the session's own Data Clerk is not changed


# ---------------------------------------------------------------- N8: the one shape the page and the session read

def test_the_view_of_the_real_session_is_one_item_per_article_the_cited_first_and_every_item_a_source_on_the_order():
    view = views.news_view(CAPTURE["order"])
    data = CAPTURE["order"]["data"]
    assert (view["stocks"], view["not_found"], view["topic"], view["days"], view["status"]) == ([{"symbol": "TSLA", "company": "Tesla Inc"}], [], None, 7, "data_ready")
    assert view["fetched"] is False and view["fetch_note"].startswith("could not fetch the latest news (the Data Clerk was built without allow_fetch") and view["fetch_note"].endswith("showing the stored news")
    numbers = [i["number"] for i in view["items"]]
    assert len(numbers) == 12 == len(set(numbers)) and numbers == [1, 2, 3, 5, 6, 7, 8, 9, 10, 4, 11, 12]      # the nine cited first, then the three that are not
    assert [i["cited"] for i in view["items"]] == [True] * 9 + [False] * 3 and set(numbers) == {s["number"] for s in data["sources"]}
    by_number = {s["number"]: s for s in data["sources"]}
    assert all(i["headline"] == by_number[i["number"]]["headline"] and i["url"] == by_number[i["number"]]["url"] and i["publisher"] == by_number[i["number"]]["publisher"] and i["date"] == by_number[i["number"]]["published_utc"][:10] for i in view["items"])
    assert all(set(i) == {"number", "numbers", "headline", "publisher", "date", "published_utc", "url", "symbols", "cited"} for i in view["items"]) and all(wo.corpus_of(s["chunk_id"]) == "news" for s in data["sources"])
    assert "【" not in view["summary"] and "[1]" in view["summary"] and "[9]" in view["summary"] and view["summary"].startswith("**Tesla (TSLA) news summary")      # the model's own citation marks, written as [n]
    assert view["problems"] == [] and view["no_supporting_text"] is False and "judging word" in view["notes"][0]
    assert views.news_headline(view) == "The news of TSLA for the last 7 days"
    assert views.news_view(wo.WorkOrder.model_validate(CAPTURE["order"])) == view      # the order as an object reads the same as its dict


def test_the_chunks_of_one_article_are_joined_and_its_number_is_the_first_chunks_and_every_chunks_number_finds_it():
    order = copy.deepcopy(CAPTURE["order"])
    first = order["data"]["sources"][0]
    second = {**first, "chunk_id": first["chunk_id"].replace("#0", "#1"), "chunk": 2, "number": 13}      # another chunk of the same real article, numbered after the twelve
    order["data"]["sources"].append(second)
    order["data"]["references"].append({"number": 13, "chunk_id": second["chunk_id"], "article_id": first["article_id"], "headline": first["headline"], "publisher": first["publisher"], "published_utc": first["published_utc"], "url": first["url"], "score": 0.5})
    items = views.news_view(order)["items"]
    assert len(items) == 12 and (items[0]["number"], items[0]["numbers"]) == (1, [1, 13])      # still 12 articles: the thirteenth source is another chunk of the first
    assert views.news_view(order)["items"][-1]["cited"] is False


def test_a_theme_has_no_stocks_and_no_fetch_note_and_a_failed_fetch_is_said_in_the_pages_words_and_the_summary_is_redacted():
    order = copy.deepcopy(CAPTURE["order"])
    order["subject"], order["data"]["companies"], order["data"]["refresh"] = "quantum computing", [], []
    theme = views.news_view(order)
    assert theme["topic"] == "quantum computing" and theme["stocks"] == [] and theme["fetch_note"] is None and views.news_headline(theme) == "The news about 'quantum computing' for the last 7 days"
    order["subject"], order["data"]["companies"] = ["TSLA"], [{"brand": "TSLA", "symbol": "TSLA", "company": "Tesla Inc"}, {"brand": "Chick-fil-a", "symbol": None, "company": None, "note": "privately held, or not known to be listed"}]
    order["data"]["refresh"] = [{"step": "news", "symbol": "TSLA", "outcome": "error", "detail": "RuntimeError: the news API said no"}]
    failed = views.news_view(order)
    assert failed["fetched"] is False and failed["fetch_note"] == "could not fetch the latest news (RuntimeError: the news API said no): showing the stored news" and failed["not_found"] == [{"typed": "Chick-fil-a", "reason": "privately held, or not known to be listed"}]
    order["data"]["refresh"] = [{"step": "news", "symbol": "TSLA", "outcome": "ok", "detail": {"fetched": 40}}]
    assert views.news_view(order)["fetched"] is True and views.news_view(order)["fetch_note"] is None
    from prometheus.security.guard import Guard
    order["data"]["answer"] = "Summary [1]. The key is sk-or-v1-abcdefabcdefabcdefabcdefabcdefabcdef1234"
    assert "sk-or-v1" not in views.news_view(order, guard=Guard.code_only(), account_id=1)["summary"] and "[redacted]" in views.news_view(order, guard=Guard.code_only(), account_id=1)["summary"]


# ---------------------------------------------------------------- N4 and N5: the route, and nothing touches the account

def run_the_desk(db, typed, mode, clerk_agent=None, **request):
    """The whole desk (the Critic and the Trader built in) on a throwaway account, given a request in the user's own words: (the account, the Data Clerk, the nodes that started, the final state). The Advisor, the Critic and the Trader are tripwires, and
    nothing may be asked of the user."""
    account_id = db.account("10000")
    agent = clerk_agent or RecordedAnswer()
    graph = desk.build_desk(clerk_factory=lambda order: agent, advisor_factory=never("Advisor"), critic_factory=never("Critic"), trader_factory=never("Trader"), with_critic=True, with_trader=True,
                            funds_factory=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()), store_factory=lambda: db.store, clock=lambda: MOMENT,
                            confirm=lambda intent: pytest.fail("a news request asks nothing: it was shown for a yes"), approver=lambda card: pytest.fail("a news request asks nothing"), clarify=lambda card: pytest.fail("a news request asks nothing"))
    started, final = [], None
    state0 = {"request": {"ask": {"text": typed, "default_count": 3}, "account": {"account": account_id, "mode": mode}, "trade": {"mode": "paper", "book": True, "model": False}, **request}, "order": None, "errors": [], "resolution": None, "funds": None,
              "path": [], "seconds": {}, "with_trader": True, "plan": None}
    for kind, chunk in graph.stream(state0, stream_mode=["tasks", "values"]):
        if kind == "tasks" and "result" not in chunk and "error" not in chunk:
            started.append(chunk["name"])
        elif kind == "values":
            final = chunk
    return account_id, agent, started, final


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_a_news_request_runs_the_account_manager_and_the_data_clerk_only_on_the_desk_with_the_critic_and_the_trader_and_the_account_is_unchanged(db, mode):
    account_id, agent, started, final = run_the_desk(db, "Show the news for TSLA", mode)
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] == final["path"] and final["errors"] == [] and final.get("clarify") is None      # the Advisor, the Critic, the decision, the funds node and the Trader were never built (their factories fail the test)
    assert order.status == "data_ready" and order.subject == ["TSLA"] and order.since_days == 7 and order.top_k == 12 and order.live is True and order.action_type == "search_news" and order.source == "news" and desk.reached_its_end(final)
    assert order.trade_decisions == [] and order.trades == [] and order.recommendations == [] and order.reviews == [] and order.direct_orders == [] and order.max_price is None and order.case is None
    assert final["plan"]["action"] == "news" and final["plan"]["intent"].news_days == 7 and final["plan"]["refuse_fetch"] is False
    assert db.store.account_balance(account_id) == 10000 and db.store.list_portfolio(account_id).positions == [] and db.store.number_trades(account_id, *YEAR) == 0      # nothing read for money, nothing written
    assert order.data.refresh == [{"step": "news", "symbol": "TSLA", "outcome": "error", "detail": "RuntimeError: no fetch in tests"}]      # the news run asked its Data Clerk to fetch (a copy with the right) and the fetch failed: the run went on with the stored news and says so
    assert [c["symbol"] for c in order.data.companies] == ["TSLA"] and views.news_view(order)["fetched"] is False and views.news_view(order)["fetch_note"] == "could not fetch the latest news (RuntimeError: no fetch in tests): showing the stored news"


def test_the_news_run_asks_its_data_clerk_for_the_right_to_fetch_whatever_the_run_says_and_the_test_only_switch_takes_it_away(db):
    seen = []

    class Watching(RecordedAnswer):
        def refresh_news(self, symbols_, days=7):
            seen.append((self.allow_fetch, list(symbols_), days))
            return super().refresh_news(symbols_, days)

    agent = Watching()      # built WITHOUT allow_fetch, as the session's own Data Clerk is when `--fetch` is not given
    run_the_desk(db, "Show the news for TSLA", "ask", clerk_agent=agent)
    run_the_desk(db, "Show the news for TSLA", "ask", clerk_agent=agent, refuse_news_fetch=True)
    run_the_desk(db, "Show the news for TSLA", "ask", clerk_agent=Watching(allow_fetch=True), refuse_news_fetch=True)      # the switch wins over a Data Clerk that may fetch
    assert seen == [(True, ["TSLA"], 7), (False, ["TSLA"], 7), (False, ["TSLA"], 7)]
    web = "".join(path.read_text(encoding="utf-8") for path in (Path(desk.__file__).parents[1] / "application" / "web").glob("*.py"))
    assert "refuse_news_fetch" not in web and "refuse-news-fetch" not in web      # the page never sets it and never offers it


def test_a_theme_is_searched_as_a_term_and_nothing_is_fetched_for_it(db):
    account_id, agent, started, final = run_the_desk(db, "news about quantum computing", "ask")
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] and order.status == "data_ready" and order.subject == "quantum computing" and agent.asked[0]["sweep"]["tickers"] is None and order.data.refresh == [] and views.news_view(order)["fetch_note"] is None
    assert db.store.account_balance(account_id) == 10000


def test_no_stock_found_and_no_theme_is_data_failed_with_the_reason_and_the_run_asks_to_clarify(db, monkeypatch):
    monkeypatch.setattr("prometheus.langgraph.desk.find_companies_of", lambda *args, **kw: [{"brand": "Nowhere Inc", "parent": None, "symbol": None, "company": None, "public": False, "how": None, "note": "no listed company found for this name: not scored"}])
    account_id, agent, started, final = run_the_desk(db, JACOBS, "ask")
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] and order.status == "data_failed" and order.errors == ["no listed company was found for the news you asked for"] and final["clarify"] is True and agent.asked == [] and not desk.reached_its_end(final)
    assert views.news_view(order)["not_found"] == [{"typed": "Nowhere Inc", "reason": "no listed company found for this name: not scored"}]


# ---------------------------------------------------------------- N10: the session

def session_parts(db, typed=JACOBS):
    return session.Parts(store=db.store, clerk=RecordedAnswer(), advisor=never("Advisor"), critic=never("Critic"), trader=never("Trader"), new_manager=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()))


def test_the_session_prints_the_news_and_nothing_about_decisions_or_money_saves_the_order_with_its_data_and_keeps_nothing_for_the_track_record(db, tmp_path, monkeypatch):
    kept = []
    monkeypatch.setattr("prometheus.proving.sessions.save_session", lambda *a, **kw: kept.append(a))
    out = tmp_path / "saved.json"
    args = session.build_parser().parse_args(["--name", "News Test", "--age", "40", "--risk", "moderate", "--cash", "10000", "--mode", "ask", "--prompt", "Show the news for TSLA", "--refuse-news-fetch", "--save", str(out)])
    lines: list[str] = []
    code = session.run_session(args, session.check_args(session.build_parser(), args), session_parts(db, "Show the news for TSLA"), ask=lambda prompt: pytest.fail(f"the user was asked {prompt!r}"), out=lines.append)
    import re
    text = "\n".join(lines)
    db.made.extend(int(n) for n in re.findall(r"account (\d+) is kept", text))
    assert code == 0 and "NEWS  The news of TSLA for the last 7 days" in text and "SUMMARY" in text and "ARTICLES  (12; the ones the summary cites first)" in text and "understood: show the news of TSLA for the last 7 days (nothing will be bought)" in text
    assert "[2] Tesla Falls 4% Despite HSBC Upgrade" in text and "https://finnhub.io/api/news?id=dae7e5a60bcff3b16866c80aa940d96a61687168634b2854127af6bc031a16d7" in text and "  cited" in text
    assert not any(word in text for word in ("HOW MUCH WAS BOUGHT", "TRADER  results", "RECOMMENDATIONS  asked for", "decided what may trade", "SESSION  PASSED", "Go ahead with this?")) and kept == []
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert set(saved) == {"request", "intent", "path", "order"} and saved["path"] == ["account_manager", "data_clerk"] and saved["intent"]["action"] == "news" and saved["intent"]["news_days"] == 7 and len(saved["order"]["data"]["sources"]) == 12


def test_the_real_session_that_made_the_capture_is_what_the_issue_says_it_is():
    """The N-capture: Ask mode, a throwaway account, "What's Tesla doing the last week?", the fetch refused (`--refuse-news-fetch`, because a fetch writes news/TSLA.txt)."""
    order, data = CAPTURE["order"], CAPTURE["order"]["data"]
    assert CAPTURE["request"] == JACOBS and CAPTURE["path"] == ["account_manager", "data_clerk"] and CAPTURE["intent"]["action"] == "news" and CAPTURE["intent"]["news_days"] == 7 and CAPTURE["intent"]["companies"] == ["Tesla"]
    assert (order["status"], order["subject"], order["company"], order["since_days"], order["top_k"], order["source"]) == ("data_ready", ["TSLA"], "Tesla Inc", 7, 12, "news")
    assert [(c["name"], c["outcome"]) for c in data["tool_calls"]] == [("retrieve_news", "ok")] and set(data["sweep"]) == {"news"} and data["results"]["press_releases"] == {"shown": [], "cited": [], "swept": []} and data["results"]["filings"]["shown"] == []
    assert order["trade_decisions"] == order["trades"] == order["recommendations"] == order["reviews"] == [] and data["filing_evidence"] == {}
    assert data["refresh"] and data["refresh"][0]["outcome"] == "refused" and (len(data["sources"]), len(data["references"])) == (12, 9)
    assert wo.verify_action(wo.WorkOrder.model_validate(order)) == []      # the Data Clerk's own check of the saved answer: a search, no invalid citation



def test_the_real_session_with_the_fetch_fetched_the_latest_news_first_and_searched_the_news_only():
    """The same request with the REAL fetch (Jacob's yes, 2026-11-23; `news_tsla_week_fetched.json`): the Data Clerk fetched TSLA's news for the window (271 articles, all already stored, so 0 written), then searched the news alone."""
    fetched = json.loads((NEWS_REAL / "news_tsla_week_fetched.json").read_text(encoding="utf-8"))
    order, data = fetched["order"], fetched["order"]["data"]
    assert fetched["request"] == JACOBS and fetched["path"] == ["account_manager", "data_clerk"] and fetched["intent"]["action"] == "news" and fetched["intent"]["news_days"] == 7
    assert (order["status"], order["subject"], order["since_days"], order["source"]) == ("data_ready", ["TSLA"], 7, "news")
    assert data["refresh"] == [{"step": "news", "symbol": "TSLA", "outcome": "ok", "detail": {"fetched": 271, "written": 0, "duplicates": 271}}]
    assert [c["name"] for c in data["tool_calls"]] == ["retrieve_news"] and set(data["sweep"]) == {"news"} and len(data["sources"]) == 12
    assert order["trade_decisions"] == order["trades"] == order["recommendations"] == order["reviews"] == []
    assert wo.verify_action(wo.WorkOrder.model_validate(order)) == []

# ---------------------------------------------------------------- N9: the web bridge

def events_of(http, run_id):
    found, kind = [], None
    with http.stream("GET", f"/api/runs/{run_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("event:"):
                kind = line[7:]
            elif line.startswith("data:"):
                found.append((kind, json.loads(line[6:])))
    return found


def test_a_replay_of_the_real_capture_emits_the_news_asks_nothing_shows_no_work_order_and_ends_with_the_status_news():
    from fastapi.testclient import TestClient

    http = TestClient(create_app(replay=Replayed(load_capture(NEWS_REAL / "news_tsla_week.json"), speed=0)))
    run_id = http.post("/api/runs", json={"prompt": "ignored", "account_id": 1}).json()["run_id"]
    events = events_of(http, run_id)
    kinds = [kind for kind, _ in events]
    assert kinds.count("news") == 1 and "question" not in kinds and "work_order" not in kinds      # Ask mode in the capture: and still no question
    news = next(data for kind, data in events if kind == "news")
    assert news["stocks"] == [{"symbol": "TSLA", "company": "Tesla Inc"}] and news["days"] == 7 and len(news["items"]) == 12 and news["fetched"] is False
    finished = events[-1][1]
    assert events[-1][0] == "run_finished" and finished["status"] == "news" and finished["headline"] == "The news of TSLA for the last 7 days" and finished["reasons"] == [news["fetch_note"]] and finished["path"] == ["account_manager", "data_clerk"]
    assert kinds.index("news") < kinds.index("run_finished") and all(kind != "portfolio" or data["changes"] == {} and data["cash_change"] == 0 for kind, data in events)      # the portfolio is as it was


def test_the_live_runs_last_line_for_a_news_run_is_the_headline_and_a_news_run_that_failed_says_why_and_asks_to_clarify():
    order = wo.WorkOrder.model_validate(CAPTURE["order"])
    intent = it.TradeIntent.model_validate(CAPTURE["intent"])
    run = Run("news", "live")
    out = Emitter(run)
    live = desk_run.Live(out)
    live.path, live.seconds = ["account_manager", "data_clerk"], {"account_manager": 6.2, "data_clerk": 5.3}
    desk_run._after_node(out, "data_clerk", {"order": order, "plan": {"action": "news", "intent": intent}, "seconds": live.seconds}, "ask")
    desk_run._finish(run, {"account_id": 1}, None, out, live, {"order": order, "plan": {"action": "news", "intent": intent}, "errors": [], "path": live.path}, 0.0)
    types = [e["type"] for e in run.events]
    assert types.count("news") == 1 and types[-1] == "run_finished" and "work_order" not in types and run.events[-1]["data"]["status"] == "news"
    failed = copy.deepcopy(CAPTURE["order"])
    failed["status"], failed["errors"] = "data_failed", ["no listed company was found for the news you asked for"]
    run = Run("failed", "live")
    out = Emitter(run)
    desk_run._finish(run, {"account_id": 1}, None, out, desk_run.Live(out), {"order": wo.WorkOrder.model_validate(failed), "plan": {"action": "news", "intent": intent}, "errors": [], "clarify": True, "path": ["account_manager", "data_clerk"]}, 0.0)
    last = run.events[-1]["data"]
    assert last["status"] == "failed" and last["headline"] == "The news could not be shown: no listed company was found for the news you asked for" and last["reasons"][-1].startswith("Please clarify your request")
