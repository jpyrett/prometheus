"""A REQUEST THAT ONLY ASKS FOR THE CURRENT PRICE (design choices entry 150; design/quote_request_spec.md, rules Q1 to Q9; Jacob, 2026-11-23: *"Whats the current price of: <SYMBOL>? it has a bug where it finds the 7-day news instead and never just returns the stock quote from the API to show the latest price"*).

REAL DATA ONLY. The Account Manager's model is a REPLAY of its REAL replies (`tests/fixtures/session_real/quote_request_forms.json`, made by `scripts/capture_quote_request_forms.py`): Jacob's own request, the other ways to ask for a price, a dollar amount in a
question about the price, a theme, eleven stocks, and the requests that look like a price and are not (the news, the price and the news, a trade that depends on the price, a trade). The prices of the route and session tests are the stored market closes of
the pinned Friday (the real Data Clerk over the stored daily bars, `friday_close`), the symbols are plain tickers (a company's name asks the recorded directory, `tests/test_direct_order.py`), and the real run is the live session of 2026-11-23 that asked
Jacob's own request (`tests/fixtures/quote_real/quote_tsla.json`: ONE Finnhub quote, nothing else fetched, no file written). Nothing is fetched, no model is called: `nothing_is_fetched` (in `tests/conftest.py`, for every test) cuts the toolbox's network.
A form that a test writes itself ("a model that said ...") has the shape of the real replies and is for a FAULT.
"""
import json
import re
from pathlib import Path

import pytest
from handshake_helpers import MOMENT, PROFILE, friday_close  # noqa: F401  (friday_close is a fixture)
from test_direct_order import YEAR, Replay, clerk, db, never, store  # noqa: F401  (db and store are fixtures: throwaway Postgres accounts)

from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.account_manager.config import load_config
from prometheus.application import session
from prometheus.application.web import desk_run, views
from prometheus.application.web.app import create_app
from prometheus.application.web.emitter import Emitter
from prometheus.application.web.replay import Replay as Replayed
from prometheus.application.web.replay import load_capture
from prometheus.application.web.runs import Run
from prometheus.common import work_order as wo
from prometheus.langgraph import desk
from prometheus.prompts import account_manager_request as prompt

FIXTURES = Path(__file__).parent / "fixtures"
FORMS = json.loads((FIXTURES / "session_real" / "quote_request_forms.json").read_text(encoding="utf-8"))["requests"]
QUOTE_REAL = FIXTURES / "quote_real"      # not in session_real: the offline evaluation's sessions dataset (and its test) takes EVERY capture there that holds a work order, and a price run is no trading session to evaluate
CAPTURE = json.loads((QUOTE_REAL / "quote_tsla.json").read_text(encoding="utf-8"))
JACOBS = "Whats the current price of: TSLA?"
TSLA = "What's the current price of TSLA?"
TWO = "Quote for AAPL and NVDA"


def understood(typed: str) -> it.TradeIntent:
    """What the real Account Manager makes of the model's real replies to `typed` (it asks once more when code refuses the first, as it does live)."""
    return AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk(), profile=PROFILE).understand_request(typed)


def a_form(**changes) -> str:
    """A form of the shape the real model answers in, for a FAULT: the real model never wrote these."""
    return json.dumps({"action": "quote", "topic": None, "symbols": ["TSLA"], "companies": [], "count": None, "allocation": "default", "budget": None, "each": False, "max_price": None, "min_price": None, "raise_funds": False, "orders": [], "news_days": None, **changes})


# ---------------------------------------------------------------- Q1 and Q2: the model reads it, code checks it, nothing of the words is read as a number

@pytest.mark.parametrize("typed, symbols, companies", [
    (JACOBS, ["TSLA"], []),                                                  # Jacob's own request, with its colon: it was read as news
    (TSLA, ["TSLA"], []),
    ("Price of Tesla", [], ["Tesla"]),
    ("How much is MSFT trading at?", ["MSFT"], []),
    (TWO, ["AAPL", "NVDA"], []),
    ("What's a share of Microsoft worth right now?", [], ["Microsoft"]),
    ("is TSLA over $400 right now?", ["TSLA"], []),                          # a dollar amount in a question about the price is not a limit: it is still a quote
])
def test_the_real_replies_to_the_requests_for_a_price_are_a_quote_of_the_stocks_named_and_nothing_else(typed, symbols, companies):
    intent = understood(typed)
    assert intent.action == "quote" and (intent.symbols, intent.companies) == (symbols, companies)
    assert intent.topic is None and intent.orders == [] and intent.budget is None and intent.count is None and intent.max_price is None and intent.min_price is None and intent.raise_funds is False and intent.corrected == [] and intent.news_days is None
    assert intent.describe() == f"show the current price of {', '.join([*symbols, *companies])} (nothing will be bought)"
    assert "news_days" not in intent.model_dump() and "max_price" not in intent.model_dump()      # a dump of a price request has nothing of a trade or a window


def test_the_requests_that_look_like_a_price_and_are_not_keep_the_old_actions():
    assert understood("What's TSLA doing the last week?").action == "news"            # stays news
    assert understood("Show the news for Tesla").action == "news"
    assert understood("price and news of TSLA").action == "news"                      # the price AND the news is news, as today (open item 1 of the spec)
    assert understood("what's TSLA's price, buy 5 if it's under $400").action == "buy"
    assert understood("buy 5 shares of TSLA").action == "order"


def test_a_theme_has_no_price_and_more_stocks_than_a_quote_may_name_are_a_request_to_clarify_and_the_model_is_not_asked_again():
    with pytest.raises(it.NeedsClarification, match="a theme has no price: name the stocks"):
        understood("price of oil stocks")
    with pytest.raises(it.NeedsClarification, match="at most 10 stocks and 11 were named"):
        understood("How much are TSLA, AAPL, MSFT, NVDA, AMD, META, GOOGL, AMZN, NFLX, INTC and ORCL trading at?")
    assert [len(FORMS[t]["replies"]) for t in ("price of oil stocks", "How much are TSLA, AAPL, MSFT, NVDA, AMD, META, GOOGL, AMZN, NFLX, INTC and ORCL trading at?")] == [1, 1]
    assert load_config().quote.max_stocks == 10


@pytest.mark.parametrize("change, said, words", [
    ({"orders": [{"side": "buy", "symbol": "TSLA", "shares": 5, "dollars": None}]}, "orders", "buy 5 TSLA and show the price of TSLA"),
    ({"budget": 500}, "budget", "show the price of TSLA, spend $500"),
    ({"each": True}, "each", "show the price of TSLA, each $500"),
    ({"count": 3}, "count", "show the price of the top 3 TSLA"),
    ({"allocation": "even"}, "allocation", "show the price of TSLA split evenly"),
    ({"raise_funds": True}, "raise funds", "show the price of TSLA and raise funds"),
    ({"max_price": 400}, "max price", "show the price of TSLA under $400 a share"),
    ({"min_price": 400}, "min price", "show the price of TSLA over $400 a share"),
    ({"news_days": 7}, "news_days", "show the price of TSLA for the last week"),
])
def test_a_quote_form_that_holds_a_trade_a_number_or_a_window_is_refused_with_its_reason_and_the_model_is_asked_again(change, said, words):
    with pytest.raises(ValueError) as refused:
        it.parse_intent(a_form(**change), words)
    assert said in str(refused.value) and not isinstance(refused.value, it.NeedsClarification)      # the model can fix it: a second try is made (`understand_request`)
    assert "nothing is bought" in str(refused.value) or "news" in str(refused.value)


def test_no_stock_a_stock_that_is_not_in_the_words_and_a_quote_that_names_a_company_and_a_ticker_together_are_handled():
    with pytest.raises(it.NeedsClarification, match="name the stocks you want the price of"):
        it.parse_intent(a_form(symbols=[]), "what is the price?")
    with pytest.raises(ValueError, match="the ticker TSLA is not in what you typed"):
        it.parse_intent(a_form(), "price of Apple")
    both = it.parse_intent(a_form(symbols=["AAPL"], companies=["Tesla"]), "price of AAPL and Tesla")
    assert both.describe() == "show the current price of AAPL, Tesla (nothing will be bought)" and both.as_request() == ("AAPL, Tesla", "list")


def test_the_program_that_turns_a_buy_into_an_order_never_touches_a_quote_intent_and_the_prompt_says_a_price_is_a_quote_and_never_news():
    typed = "is TSLA over $400 right now?"
    intent = it.parse_intent(a_form(), typed)
    assert it.as_direct_order(intent, typed) is None
    assert 'action "quote"' in prompt.SYSTEM and 'is "quote", never "news"' in prompt.SYSTEM      # one bullet, and the news bullet says that a price is a quote
    assert all("TSLA" not in line and "Tesla" not in line for line in prompt.SYSTEM.splitlines() if 'action "quote"' in line)      # made-up tickers only


# ---------------------------------------------------------------- Q3 to Q5: the route, and nothing touches the account

def run_the_desk(db, typed, mode, clerk_agent=None, fetch=False, **request):
    """The whole desk (the Critic and the Trader built in) on a throwaway account, given a request in the user's own words: (the account, the Data Clerk, the nodes that started, the final state). The Advisor, the Critic and the Trader are tripwires, and
    nothing may be asked of the user. The Data Clerk is the real one over the stored closes."""
    account_id = db.account("10000")
    agent = clerk_agent or clerk()
    graph = desk.build_desk(clerk_factory=lambda order: (seen.append(order.live), agent)[1], advisor_factory=never("Advisor"), critic_factory=never("Critic"), trader_factory=never("Trader"), with_critic=True, with_trader=True,
                            funds_factory=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()), store_factory=lambda: db.store, clock=lambda: MOMENT,
                            confirm=lambda intent: pytest.fail("a price request asks nothing: it was shown for a yes"), approver=lambda card: pytest.fail("a price request asks nothing"), clarify=lambda card: pytest.fail("a price request asks nothing"))
    seen, started, final = [], [], None
    state0 = {"request": {"ask": {"text": typed, "default_count": 3, "fetch": fetch}, "account": {"account": account_id, "mode": mode}, "trade": {"mode": "paper", "book": True, "model": False}, **request}, "order": None, "errors": [], "resolution": None, "funds": None,
              "path": [], "seconds": {}, "with_trader": True, "plan": None}
    for kind, chunk in graph.stream(state0, stream_mode=["tasks", "values"]):
        if kind == "tasks" and "result" not in chunk and "error" not in chunk:
            started.append(chunk["name"])
        elif kind == "values":
            final = chunk
    assert seen == [False] * len(seen)      # the Data Clerk is built NOT live: it could not fetch anything
    return account_id, agent, started, final


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_a_price_request_runs_the_account_manager_and_the_data_clerk_only_on_the_desk_with_the_critic_and_the_trader_and_the_account_is_unchanged(db, friday_close, mode):
    account_id, agent, started, final = run_the_desk(db, TSLA, mode)
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] == final["path"] and final["errors"] == [] and final.get("clarify") is None      # the Advisor, the Critic, the decision, the funds node and the Trader were never built (their factories fail the test)
    assert order.status == "data_ready" and order.subject == ["TSLA"] and order.action_type == "fetch_quote" and order.live is False and desk.reached_its_end(final) and order.errors == []
    assert order.trade_decisions == [] and order.trades == [] and order.recommendations == [] and order.reviews == [] and order.direct_orders == [] and order.max_price is None and order.case is None
    quote = order.data.quotes["TSLA"]
    assert (quote["price"], quote["simulated"]) == (390.22, True) and order.data.sources == [] and order.data.sweep == {} and order.data.refresh == [] and [c["name"] for c in order.data.tool_calls] == ["get_quote"] and order.data.tokens_in == 0      # one quote, no model, no search, no news fetch
    assert final["plan"]["action"] == "quote" and final["plan"]["intent"].symbols == ["TSLA"]
    assert db.store.account_balance(account_id) == 10000 and db.store.list_portfolio(account_id).positions == [] and db.store.number_trades(account_id, *YEAR) == 0      # nothing read for money, nothing written


def test_two_stocks_are_quoted_once_each_and_the_run_never_asks_the_clerk_for_news_a_search_or_bars(db, friday_close):
    calls = []

    class Watching(type(clerk())):
        def request_quotes(self, order):
            calls.append(list(order.subject))
            return super().request_quotes(order)

        def __getattribute__(self, name):
            if name in ("request_news", "request_data", "refresh_news", "refresh_data", "ask", "fetching"):
                raise AssertionError(f"a price request used the Data Clerk's {name}")
            return super().__getattribute__(name)

    agent = Watching(model=object())
    _, _, started, final = run_the_desk(db, TWO, "auto", clerk_agent=agent)
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] and calls == [["AAPL", "NVDA"]] and list(order.data.quotes) == ["AAPL", "NVDA"] and order.status == "data_ready"
    assert [s["symbol"] for s in views.quote_view(order)["stocks"]] == ["AAPL", "NVDA"]


def test_a_symbol_whose_quote_fails_is_named_and_the_others_are_still_shown_and_with_none_the_order_is_data_failed(db, friday_close, monkeypatch):
    rows = lambda *symbols: [{"brand": s, "parent": None, "symbol": s, "company": None, "public": True, "how": "symbol as typed", "note": None} for s in symbols]      # tickers as typed, as the Data Clerk's own step makes them
    monkeypatch.setattr("prometheus.langgraph.desk.find_companies_of", lambda *args, **kw: rows("TSLA", "ZZZZ"))      # ZZZZ has no stored bar: its quote is a real failure
    _, _, _, final = run_the_desk(db, TWO, "ask")
    order = final["order"]
    assert order.status == "data_ready" and list(order.data.quotes) == ["TSLA"] and len(order.data.problems) == 1 and order.data.problems[0].startswith("ZZZZ: no quote (") and order.errors == order.data.problems and desk.reached_its_end(final)
    view = views.quote_view(order)
    assert [s["symbol"] for s in view["stocks"]] == ["TSLA"] and view["problems"] == order.data.problems and views.run_verdict(None, None, action="quote", order=order) == {"status": "quote", "headline": "TSLA $390.22 (-1.82%)", "reasons": order.data.problems}
    monkeypatch.setattr("prometheus.langgraph.desk.find_companies_of", lambda *args, **kw: rows("ZZZZ"))
    _, _, _, failed = run_the_desk(db, TWO, "ask")
    assert failed["order"].status == "data_failed" and failed["order"].data.quotes == {} and failed["order"].errors[0].startswith("ZZZZ: no quote (") and not desk.reached_its_end(failed)


def test_no_stock_found_is_data_failed_with_the_reason_and_the_run_asks_to_clarify(db, friday_close, monkeypatch):
    monkeypatch.setattr("prometheus.langgraph.desk.find_companies_of", lambda *args, **kw: [{"brand": "Nowhere Inc", "parent": None, "symbol": None, "company": None, "public": False, "how": None, "note": "no listed company found for this name: not scored"}])
    _, _, started, final = run_the_desk(db, TSLA, "ask")
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] and order.status == "data_failed" and order.errors == ["no listed company was found for the price you asked for"] and final["clarify"] is True and not desk.reached_its_end(final)
    assert views.quote_view(order)["not_found"] == [{"typed": "Nowhere Inc", "reason": "no listed company found for this name: not scored"}] and views.quote_view(order)["stocks"] == []


def test_a_company_the_user_named_is_found_in_the_directory_and_its_name_is_on_the_line_of_its_price(db, friday_close, monkeypatch):
    """"Price of Tesla": the real `classify` over Finnhub's REAL recorded directory (the desk's own call has no directory to give, and in a test the network is cut): Tesla Inc is TSLA, and the line says so."""
    from functools import partial

    from prometheus.application import recommendations as rec
    from test_direct_order import directory

    monkeypatch.setattr(desk, "classify", partial(rec.classify, search=directory))
    _, _, started, final = run_the_desk(db, "Price of Tesla", "auto")
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] and order.status == "data_ready" and order.subject == ["TSLA"] and order.company == "Tesla Inc" and list(order.data.quotes) == ["TSLA"]
    assert views.quote_lines(views.quote_view(order, final["plan"])) == ["TSLA  Tesla Inc  $390.22  -7.23 (-1.82%)  market close of 2026-11-13 (simulated clock)"]


def test_the_fetch_box_means_nothing_to_a_price_run(db, friday_close):
    """The page's "refresh market data first" box is `request["ask"]["fetch"]` (and `--fetch` of the session): a price run is the same with it on, built not live, with nothing refreshed."""
    _, _, started, final = run_the_desk(db, TSLA, "ask", fetch=True)
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] and order.live is False and order.data.refresh == [] and order.data.sources == [] and list(order.data.quotes) == ["TSLA"] and final["plan"].get("fetch") is None


# ---------------------------------------------------------------- Q6: the one shape the page and the session read

def test_the_view_of_the_real_session_is_the_quote_copied_with_the_time_in_words_and_the_one_line_for_the_stock():
    view = views.quote_view(CAPTURE["order"])
    quote = CAPTURE["order"]["data"]["quotes"]["TSLA"]
    assert (view["status"], view["not_found"], view["problems"], view["notes"]) == ("data_ready", [], [], [])
    (stock,) = view["stocks"]
    assert {k: stock[k] for k in ("symbol", "price", "change", "change_percent", "open", "high", "low", "previous_close", "as_of_utc", "source")} == {"symbol": "TSLA", **{k: quote[k] for k in ("price", "change", "change_percent", "open", "high", "low", "previous_close", "as_of_utc", "source")}}
    assert stock["company"] is None and stock["simulated"] is False and stock["when"] == "live quote, last trade 2026-11-23 15:26 ET"      # 20:26:31 UTC, in Eastern time
    assert set(stock) == {"symbol", "company", "price", "change", "change_percent", "open", "high", "low", "previous_close", "as_of_utc", "source", "simulated", "when"}
    assert views.quote_lines(view) == ["TSLA  $388.42  +42.60 (+12.32%)  live quote, last trade 2026-11-23 15:26 ET"] and views.quote_headline(view) == "TSLA $388.42 (+12.32%)"
    assert views.quote_view(wo.WorkOrder.model_validate(CAPTURE["order"])) == view      # the order as an object reads the same as its dict


def test_the_stored_close_is_said_as_the_market_close_of_its_day_a_company_is_named_when_the_clerk_found_it_and_a_loss_and_a_cheap_stock_are_written_as_they_are():
    order = json.loads(json.dumps(CAPTURE["order"]))
    quote = order["data"]["quotes"]["TSLA"]
    order["data"]["quotes"] = {"TSLA": {**quote, "price": 390.22, "change": -7.23, "change_percent": -1.8191, "as_of_utc": "2026-11-13T20:29:00Z", "simulated": True},
                               "PENNY": {**quote, "symbol": "PENNY", "price": 0.4512, "change": 0.0, "change_percent": 0.0, "as_of_utc": "2026-11-13T20:29:00Z", "simulated": True}}
    order["data"]["companies"] = [{"brand": "Tesla", "symbol": "TSLA", "company": "Tesla Inc"}, {"brand": "Penny", "symbol": "PENNY", "company": None},
                                  {"brand": "Chick-fil-a", "symbol": None, "company": None, "note": "privately held, or not known to be listed"}]
    order["data"]["problems"] = ["XYZ: no quote (RuntimeError: nothing)"]
    view = views.quote_view(order)
    assert [s["company"] for s in view["stocks"]] == ["Tesla Inc", None] and all(s["when"] == "market close of 2026-11-13 (simulated clock)" for s in view["stocks"])
    assert views.quote_lines(view) == ["TSLA  Tesla Inc  $390.22  -7.23 (-1.82%)  market close of 2026-11-13 (simulated clock)", "PENNY  $0.4512  +0.00 (+0.00%)  market close of 2026-11-13 (simulated clock)",
                                       "not found: Chick-fil-a: privately held, or not known to be listed", "problem: XYZ: no quote (RuntimeError: nothing)"]
    assert views.quote_lines(view, problems=False)[-1].startswith("not found:") and views.quote_headline(view) == "TSLA $390.22 (-1.82%) · PENNY $0.4512 (+0.00%)"
    from prometheus.security.guard import Guard
    order["data"]["problems"] = ["The key is sk-or-v1-abcdefabcdefabcdefabcdefabcdefabcdef1234"]
    redacted = views.quote_view(order, guard=Guard.code_only(), account_id=1)["problems"][0]
    assert "sk-or-v1" not in redacted and "[redacted]" in redacted      # the same door as the run's last message


def test_the_view_of_the_route_reads_the_stored_close_of_the_pinned_friday(db, friday_close):
    _, _, _, final = run_the_desk(db, TSLA, "ask")
    (stock,) = views.quote_view(final["order"], final["plan"])["stocks"]
    assert (stock["price"], stock["simulated"], stock["when"]) == (390.22, True, "market close of 2026-11-13 (simulated clock)")


# ---------------------------------------------------------------- Q8: the session

def session_parts(db, typed=JACOBS):
    return session.Parts(store=db.store, clerk=clerk(), advisor=never("Advisor"), critic=never("Critic"), trader=never("Trader"), new_manager=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()))


def test_the_session_prints_a_line_for_the_stock_and_nothing_about_decisions_or_money_saves_the_order_with_its_data_and_keeps_nothing_for_the_track_record(db, friday_close, tmp_path, monkeypatch):
    kept = []
    monkeypatch.setattr("prometheus.proving.sessions.save_session", lambda *a, **kw: kept.append(a))
    out = tmp_path / "saved.json"
    args = session.build_parser().parse_args(["--name", "Quote Test", "--age", "40", "--risk", "moderate", "--cash", "10000", "--mode", "ask", "--prompt", JACOBS, "--save", str(out)])
    lines: list[str] = []
    code = session.run_session(args, session.check_args(session.build_parser(), args), session_parts(db), ask=lambda prompt_: pytest.fail(f"the user was asked {prompt_!r}"), out=lines.append)
    text = "\n".join(lines)
    db.made.extend(int(n) for n in re.findall(r"account (\d+) is kept", text))
    assert code == 0 and "QUOTE  TSLA $390.22 (-1.82%)" in text and "  TSLA  $390.22  -7.23 (-1.82%)  market close of 2026-11-13 (simulated clock)" in text and "understood: show the current price of TSLA (nothing will be bought)" in text
    assert not any(word in text for word in ("HOW MUCH WAS BOUGHT", "TRADER  results", "RECOMMENDATIONS  asked for", "decided what may trade", "SESSION  PASSED", "Go ahead with this?", "NEWS  ")) and kept == []
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert set(saved) == {"request", "intent", "path", "order"} and saved["path"] == ["account_manager", "data_clerk"] and saved["intent"]["action"] == "quote" and saved["order"]["data"]["quotes"]["TSLA"]["price"] == 390.22
    assert saved["order"]["trade_decisions"] == saved["order"]["trades"] == saved["order"]["recommendations"] == []


def test_the_real_session_that_made_the_capture_is_what_the_issue_says_it_is():
    """The Q-capture: Ask mode, a throwaway account, Jacob's own request, `--cleanup --yes`: ONE live Finnhub quote and nothing else fetched."""
    order, data = CAPTURE["order"], CAPTURE["order"]["data"]
    assert CAPTURE["request"] == JACOBS and CAPTURE["path"] == ["account_manager", "data_clerk"] and CAPTURE["intent"]["action"] == "quote" and CAPTURE["intent"]["symbols"] == ["TSLA"]
    assert (order["status"], order["subject"], order["action_type"], order["live"], order["errors"]) == ("data_ready", ["TSLA"], "fetch_quote", False, [])
    assert [(c["name"], c["args"], c["outcome"]) for c in data["tool_calls"]] == [("get_quote", {"symbol": "TSLA"}, "ok")] and data["quotes"]["TSLA"]["source"] == "Finnhub /quote" and "simulated" not in data["quotes"]["TSLA"]
    assert data["sources"] == [] and data["references"] == [] and data["sweep"] == {} and data["refresh"] == [] and data["bars"] == {} and data["tokens_in"] == data["tokens_out"] == 0      # no news source on the order, no search, no model
    assert order["trade_decisions"] == order["trades"] == order["recommendations"] == order["reviews"] == [] and data["answer"] == "TSLA 388.42"
    assert wo.verify_quote_request(wo.WorkOrder.model_validate({**order, "status": "new"})) == []      # the Data Clerk's own check of the order it was asked


# ---------------------------------------------------------------- Q7: the web bridge, the price in the main output

def events_of(http, run_id):
    found, kind = [], None
    with http.stream("GET", f"/api/runs/{run_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("event:"):
                kind = line[7:]
            elif line.startswith("data:"):
                found.append((kind, json.loads(line[6:])))
    return found


def test_a_replay_of_the_real_capture_emits_the_quote_says_it_in_the_output_asks_nothing_shows_no_work_order_and_ends_with_the_status_quote():
    from fastapi.testclient import TestClient

    http = TestClient(create_app(replay=Replayed(load_capture(QUOTE_REAL / "quote_tsla.json"), speed=0)))
    run_id = http.post("/api/runs", json={"prompt": "ignored", "account_id": 1}).json()["run_id"]
    events = events_of(http, run_id)
    kinds = [kind for kind, _ in events]
    assert kinds.count("quote") == 1 and "question" not in kinds and "work_order" not in kinds and "news" not in kinds      # Ask mode in the capture: and still no question
    quote = next(data for kind, data in events if kind == "quote")
    assert [s["symbol"] for s in quote["stocks"]] == ["TSLA"] and quote["stocks"][0]["price"] == 388.42 and quote["stocks"][0]["when"] == "live quote, last trade 2026-11-23 15:26 ET" and quote["problems"] == []
    said = [data for kind, data in events if kind == "log" and data["agent"] == "data_clerk" and data["block"]]      # the price is written in the run's OUTPUT: one block, a line for the stock
    assert [d["text"] for d in said] == ["TSLA  $388.42  +42.60 (+12.32%)  live quote, last trade 2026-11-23 15:26 ET"]
    finished = events[-1][1]
    assert events[-1][0] == "run_finished" and finished["status"] == "quote" and finished["headline"] == "TSLA $388.42 (+12.32%)" and finished["reasons"] == [] and finished["path"] == ["account_manager", "data_clerk"]
    assert kinds.index("quote") < kinds.index("run_finished") and all(kind != "portfolio" or data["changes"] == {} and data["cash_change"] == 0 for kind, data in events)      # the portfolio is as it was
    assert events[-2][0] == "log" and events[-2][1]["level"] == "good" and events[-2][1]["text"] == "TSLA $388.42 (+12.32%)"      # the last line of the output is the price, in the colour of a good end


def test_the_live_runs_output_for_a_price_run_is_the_stock_lines_then_the_headline_and_a_price_run_that_failed_says_why_and_asks_to_clarify():
    order = wo.WorkOrder.model_validate(CAPTURE["order"])
    intent = it.TradeIntent.model_validate(CAPTURE["intent"])
    run = Run("quote", "live")
    out = Emitter(run)
    live = desk_run.Live(out)
    live.path, live.seconds = ["account_manager", "data_clerk"], {"account_manager": 7.4, "data_clerk": 0.1}
    plan = {"action": "quote", "intent": intent}
    desk_run._after_node(out, "data_clerk", {"order": order, "plan": plan, "seconds": live.seconds}, "ask")
    desk_run._finish(run, {"account_id": 1}, None, out, live, {"order": order, "plan": plan, "errors": [], "path": live.path}, 0.0)
    types = [e["type"] for e in run.events]
    assert types.count("quote") == 1 and types[-1] == "run_finished" and "work_order" not in types and run.events[-1]["data"]["status"] == "quote" and run.events[-1]["data"]["headline"] == "TSLA $388.42 (+12.32%)"
    assert [e["data"]["text"] for e in run.events if e["type"] == "log" and e["data"]["block"]] == ["TSLA  $388.42  +42.60 (+12.32%)  live quote, last trade 2026-11-23 15:26 ET"]
    failed = json.loads(json.dumps(CAPTURE["order"]))
    failed["status"], failed["errors"], failed["data"]["quotes"], failed["data"]["problems"] = "data_failed", ["TSLA: no quote (RuntimeError: nope)"], {}, ["TSLA: no quote (RuntimeError: nope)"]
    run = Run("failed", "live")
    out = Emitter(run)
    desk_run._finish(run, {"account_id": 1}, None, out, desk_run.Live(out), {"order": wo.WorkOrder.model_validate(failed), "plan": plan, "errors": [], "clarify": True, "path": ["account_manager", "data_clerk"]}, 0.0)
    last = run.events[-1]["data"]
    assert last["status"] == "failed" and last["headline"] == "The price could not be shown: TSLA: no quote (RuntimeError: nope)" and last["reasons"][-1].startswith("Please clarify your request")
    assert not [e for e in run.events if e["type"] == "log" and e["data"]["agent"] == "data_clerk"]      # nothing to show as a stock line: the last line says why
