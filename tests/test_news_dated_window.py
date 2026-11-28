"""A DATED WINDOW FOR A NEWS REQUEST (design choices entry 151; design/news_dated_window_spec.md, rules D1 to D9; Jacob, 2026-11-28: *"We cannot fetch a single day news in the past ... it is currently finding the last most recent day instead of the specific day"*).

REAL DATA ONLY. The Account Manager's model is a REPLAY of its REAL replies (`tests/fixtures/session_real/news_dated_forms.json`, made by `scripts/capture_news_dated_forms.py` with TODAY FIXED at Saturday 2026-11-28: "news on TSLA on November 20", the same
with an ISO date, "NVDA news from Nov 3 to Nov 5", "what happened with Apple last Tuesday?", "TSLA news the day before Thanksgiving" and the rolling control "TSLA news last week"); the real Account Manager's code checks every field against the words. The
searches run on a small real Chroma store (the retriever tests' own, with its keyword embedder) and, for the desk, on a COPY of the real stored news index (`tests/test_as_of_clock.py` does the same: the real one is never opened by a test). The Data Clerk is the real
class; its model is the SCRIPTED model of `tests/test_data_clerk_agent.py` (the repository's own double for the chat model), and the fetch is a recorder: nothing is fetched, no paid API is called and no data file is written (`nothing_is_fetched`, in
`tests/conftest.py`, for every test). A form that a test writes itself ("a model that said ...") has the shape of the real replies and is for a FAULT.
"""
import json
import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from handshake_helpers import MOMENT, PROFILE
from test_data_clerk_agent import ScriptedModel, call, say
from test_direct_order import Replay, clerk, db, never, store  # noqa: F401  (db and store are fixtures: throwaway Postgres accounts)
from test_news_view import FORMS as NEWS_FORMS
from test_news_view import RecordedAnswer
from test_rag_retriever import KeywordEmbeddings, chunk, fill, make_store

from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.account_manager.config import load_config
from prometheus.agents.data_clerk.data_toolbox import DataToolbox
from prometheus.agents.data_clerk.data_clerk_agent import NEWS_ONLY, DataClerk, Result
from prometheus.application import session
from prometheus.application.web import views
from prometheus.common import trading_clock
from prometheus.common import work_order as wo
from prometheus.langgraph import desk
from prometheus.prompts import account_manager_request as prompt
from prometheus.rag.citations import SourceBook
from prometheus.rag.config import REPO_ROOT
from prometheus.rag.retriever import search_news
from prometheus.rag.store import get_embeddings
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolLimitError

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
CAPTURE = json.loads((FIXTURES / "news_dated_forms.json").read_text(encoding="utf-8"))
FORMS = CAPTURE["requests"]
TODAY = date.fromisoformat(CAPTURE["today"])      # Saturday 2026-11-28, the day the model was told
EASTERN = ZoneInfo("America/New_York")
STORE = REPO_ROOT / "data" / "processed" / "vectorstore"
DAY_OF_THE_ASK = "What was the news on TSLA on 2026-11-20?"


def understood(typed: str) -> it.TradeIntent:
    """What the real Account Manager makes of the model's real replies to `typed`, told that today is `TODAY` (it asks once more when code refuses the first reply, as it does live)."""
    return AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk(), profile=PROFILE).understand_request(typed, today=TODAY)


def a_form(**changes) -> str:
    """A form of the shape the real model answers in, for a FAULT: the real model never wrote these."""
    return json.dumps({"action": "news", "topic": None, "symbols": ["TSLA"], "companies": [], "count": None, "allocation": "default", "budget": None, "each": False, "max_price": None, "min_price": None, "raise_funds": False, "orders": [], "news_days": None,
                       "news_from": None, "news_to": None, **changes})


def read(typed: str, **changes) -> it.TradeIntent:
    return it.parse_intent(a_form(**changes), typed, TODAY)


def days(first: str, last: str) -> tuple[date, date]:
    return date.fromisoformat(first), date.fromisoformat(last)


# ---------------------------------------------------------------- D2: a program reads an explicit date (a closed list of forms)

@pytest.mark.parametrize("words", ["news on TSLA on 2026-11-03", "TSLA news 11/3", "TSLA news 11/3/2026", "TSLA news 11/3/26", "TSLA news Nov 3", "TSLA news on November 3rd", "TSLA news Nov. 3, 2026", "TSLA news on 3 November 2026"])
def test_each_form_of_the_closed_list_is_read_as_that_day(words):
    assert it.news_dates(words, TODAY) == days("2026-11-03", "2026-11-03")


@pytest.mark.parametrize("words", ["NVDA news Nov 3 to Nov 5", "NVDA news Nov 3 through Nov 5", "NVDA news Nov 3 thru Nov 5", "NVDA news Nov 3 until Nov 5", "NVDA news Nov 3 - Nov 5", "NVDA news Nov 3 – 5", "NVDA news between Nov 3 and Nov 5",
                                   "NVDA news from 2026-11-03 to 2026-11-05", "NVDA news 11/3-11/5", "NVDA news Nov 3 to 5", "NVDA news between 3 November and 5 November"])
def test_a_range_is_two_dates_joined_by_each_joiner_and_a_second_date_may_be_only_a_day_number(words):
    assert it.news_dates(words, TODAY) == days("2026-11-03", "2026-11-05")


def test_a_date_with_no_year_is_the_most_recent_one_on_or_before_today_and_a_range_that_crosses_the_year_keeps_its_order():
    assert it.news_dates("news Nov 20", TODAY) == days("2026-11-20", "2026-11-20")      # before today: this year
    assert it.news_dates("news Nov 30", TODAY) == days("2025-11-30", "2025-11-30")      # after today: last year
    assert it.news_dates("news Dec 28 to Jan 3", TODAY) == days("2025-12-28", "2026-01-03")
    assert it.news_dates("news Nov 3 and Nov 10", TODAY) == days("2026-11-03", "2026-11-10")      # two dates that are not joined: the days from the earliest to the latest


def test_a_date_that_does_not_exist_and_a_day_the_program_does_not_read_are_not_read():
    for words in ("news Feb 30", "news 13/1", "news 2026-02-30", "news last Tuesday", "news the day before Thanksgiving", "news Black Friday", "news for the last 3 days", "TSLA headlines"):
        assert it.news_dates(words, TODAY) is None, words
    assert it.news_dates("news Nov 5 to Nov 3", TODAY) == days("2026-11-05", "2026-11-03")      # backwards as written: `window_problem` refuses it


def test_a_date_wins_over_the_model_and_over_a_rolling_span_in_the_same_words_and_corrected_says_so():
    intent = read("TSLA news Nov 3 and the last week", news_days=7)      # the model said 7 days
    assert (intent.news_from, intent.news_to, intent.news_days) == ("2026-11-03", "2026-11-03", None)
    assert intent.corrected == ["showing the news of 2026-11-03; the 'last week' in the same request is not used"] and "on Tuesday 2026-11-03" in intent.describe() and "(note: showing the news of 2026-11-03;" in intent.describe()
    wrong = read("TSLA news from Nov 3 to Nov 5", news_from="2026-11-10", news_to="2026-11-12")      # the model's days are not the words'
    assert (wrong.news_from, wrong.news_to, wrong.corrected) == ("2026-11-03", "2026-11-05", []) and wrong.describe() == "show the news of TSLA from 2026-11-03 to 2026-11-05 (nothing will be bought)"


# ---------------------------------------------------------------- D1: the validator, each refusal once

@pytest.mark.parametrize("change, words, said", [
    ({"news_from": "2026-11-24"}, "TSLA news last Tuesday", "go together"),
    ({"news_from": "2026-11-25", "news_to": "2026-11-24"}, "TSLA news last Tuesday", "is after the last day"),
    ({"news_from": "2026-11-27", "news_to": "2026-12-01"}, "TSLA news last Friday", "is after today (2026-11-28)"),
    ({"news_from": "2026-06-01", "news_to": "2026-10-01"}, "TSLA news since June", "123 days"),
    ({"news_from": "2025-10-01", "news_to": "2025-10-02"}, "TSLA news a year ago", "more than 365 days before today"),
    ({"news_from": "November 20", "news_to": "November 20"}, "TSLA news that day", "YYYY-MM-DD"),
])
def test_a_news_form_whose_dates_are_not_a_window_that_can_be_shown_is_refused_with_its_reason(change, words, said):
    with pytest.raises(ValueError) as refused:
        it.parse_intent(a_form(**change), words, TODAY)
    assert said in str(refused.value)


def test_dates_the_words_wrote_that_cannot_be_shown_are_a_request_to_clarify_at_once_and_a_model_is_not_asked_again():
    for words, said in (("TSLA news on Dec 3, 2026", "is after today (2026-11-28)"), ("TSLA news from Jan 1 to Jun 30", "181 days"), ("TSLA news on Jan 5, 2025", "more than 365 days before today"), ("TSLA news from Nov 5 to Nov 3", "is after the last day")):
        with pytest.raises(it.NeedsClarification) as refused:
            read(words)
        assert said in str(refused.value) and "Ask for a day or a span of days that has passed" in str(refused.value)


def test_a_dated_window_goes_with_the_news_only_and_a_dump_has_neither_key_unless_there_is_one():
    with pytest.raises(ValueError, match="go with a request for the news"):
        it.TradeIntent(request="x", action="buy", symbols=["TSLA"], news_from="2026-11-20", news_to="2026-11-20")
    with pytest.raises(ValueError, match="leave news_days null"):      # a dated window and a rolling one are two ways to say the same thing
        it.TradeIntent.model_validate({"request": "x", "action": "news", "symbols": ["TSLA"], "news_from": "2026-11-20", "news_to": "2026-11-20", "news_days": 7}, context={"today": TODAY})
    buy = json.loads(NEWS_FORMS["buy 5 shares of TSLA"]["replies"][-1])
    assert {"news_from", "news_to"}.isdisjoint(it.parse_intent(json.dumps({**buy, "news_from": "2026-11-20", "news_to": "2026-11-20"}), "buy 5 shares of TSLA", TODAY).model_dump())      # a window means nothing to a request that is not for the news: dropped, as news_days is
    quote = a_form(action="quote", news_from="2026-11-20", news_to="2026-11-20")
    with pytest.raises(ValueError, match="go with a request for the news"):
        it.parse_intent(quote, "what's the price of TSLA", TODAY)
    rolling = read("TSLA news for the last 3 days", news_days=3)
    assert rolling.news_days == 3 and {"news_from", "news_to"}.isdisjoint(rolling.model_dump()) and (rolling.news_from, rolling.news_to) == (None, None) and rolling.describe() == "show the news of TSLA for the last 3 days (nothing will be bought)"
    dated = read("TSLA news on Nov 20")
    assert (dated.model_dump()["news_from"], dated.model_dump()["news_to"], dated.news_days) == ("2026-11-20", "2026-11-20", None) and "news_days" not in dated.model_dump()


def test_the_models_own_weekday_arithmetic_is_checked_against_the_weekday_the_words_name():
    saturday = a_form(companies=["Apple"], symbols=[], news_from="2026-11-21", news_to="2026-11-21")      # what the real model's first reply to "last Tuesday" was (a Saturday)
    with pytest.raises(ValueError, match=r"the person wrote Tuesday, but 2026-11-21 is a Saturday: work the day out again from today's date \(Saturday 2026-11-28\)"):
        it.parse_intent(saturday, "what happened with Apple last Tuesday?", TODAY)
    assert it.parse_intent(a_form(companies=["Apple"], symbols=[], news_from="2026-11-24", news_to="2026-11-24"), "what happened with Apple last Tuesday?", TODAY).news_from == "2026-11-24"
    assert read("TSLA news since Monday", news_from="2026-11-23", news_to="2026-11-28").news_from == "2026-11-23"      # a span with the named day in it
    assert read("what did TSLA do on Nov 3, a Tuesday", news_from="2026-11-21", news_to="2026-11-21").news_from == "2026-11-03"      # a date the program read is not the model's: not checked


# ---------------------------------------------------------------- D2 and D3: the real model's replies, with today in its message

@pytest.mark.parametrize("typed, named, window", [
    ("news on TSLA on November 20", {"symbols": ["TSLA"]}, ("2026-11-20", "2026-11-20")),
    (DAY_OF_THE_ASK, {"symbols": ["TSLA"]}, ("2026-11-20", "2026-11-20")),
    ("NVDA news from Nov 3 to Nov 5", {"symbols": ["NVDA"]}, ("2026-11-03", "2026-11-05")),
    ("what happened with Apple last Tuesday?", {"companies": ["Apple"]}, ("2026-11-24", "2026-11-24")),      # the model's: the first reply was a Saturday, code refused it, the second is the Tuesday
    ("TSLA news the day before Thanksgiving", {"symbols": ["TSLA"]}, ("2026-11-25", "2026-11-25")),
])
def test_the_real_replies_to_the_dated_requests_are_news_of_that_day_with_no_rolling_window(typed, named, window):
    intent = understood(typed)
    assert intent.action == "news" and {"symbols": intent.symbols, "companies": intent.companies} == {"symbols": [], "companies": []} | named and (intent.news_from, intent.news_to) == window and intent.news_days is None
    assert intent.topic is None and intent.orders == [] and intent.budget is None and intent.corrected == []
    assert len(FORMS[typed]["replies"]) == (2 if "Tuesday" in typed else 1)


def test_the_rolling_control_is_what_it_was_and_the_model_was_told_todays_date_in_its_message_and_not_in_its_prompt():
    control = understood("TSLA news last week")
    assert (control.news_days, control.news_from, control.news_to) == (7, None, None) and control.describe() == "show the news of TSLA for the last 7 days (nothing will be bought)"
    model = Replay(FORMS["NVDA news from Nov 3 to Nov 5"]["replies"])
    AccountManager(model=model, model_id="recorded", clerk=clerk(), profile=PROFILE).understand_request("NVDA news from Nov 3 to Nov 5", today=TODAY)
    system, user = model.seen[0]
    assert system.content == prompt.SYSTEM and "2026-11-28" not in prompt.SYSTEM and "Saturday" not in prompt.SYSTEM      # the system prompt is the same text every day (a cached prefix)
    assert user.content == "Today is Saturday 2026-11-28.\n<request>\nNVDA news from Nov 3 to Nov 5\n</request>"
    assert prompt.user_message("x") == "<request>\nx\n</request>" and "news_from" in prompt.RETRY and "news_to" in prompt.RETRY and '"news_from": null, "news_to": null}' in prompt.SYSTEM
    bullet = next(line for line in prompt.SYSTEM.splitlines() if line.startswith("- news_from and news_to"))
    assert "YYYY-MM-DD" in bullet and "leave news_days null" in bullet and not any(real in bullet for real in ("TSLA", "Tesla", "AAPL", "Apple", "NVDA"))      # made-up names only
    retry = Replay(FORMS["what happened with Apple last Tuesday?"]["replies"])
    AccountManager(model=retry, model_id="recorded", clerk=clerk(), profile=PROFILE).understand_request("what happened with Apple last Tuesday?", today=TODAY)
    assert "is a Sunday" in retry.seen[1][-1].content      # the model was told why its first reply (2026-11-22) was refused


# ---------------------------------------------------------------- D4: the work order

def dated_order(first="2026-11-20", last="2026-11-20", **options) -> wo.WorkOrder:
    return wo.new_work_order(f"Summarize the news of TSLA {wo.window_words(first, last)}", requester="desk", action_type="search_news", source="news", subject=["TSLA"], company="Tesla Inc", time_period=first if first == last else f"{first} to {last}", news_from=first, news_to=last,
                             top_k=12, live=True, **options)


def test_a_work_order_holds_a_dated_window_or_a_rolling_one_and_verify_request_refuses_each_fault_once():
    order = dated_order()
    assert (order.news_from, order.news_to, order.since_days, order.time_period) == ("2026-11-20", "2026-11-20", None, "2026-11-20") and wo.verify_request(order) == []
    assert dated_order("2026-11-03", "2026-11-05").time_period == "2026-11-03 to 2026-11-05" and wo.window_words("2026-11-03", "2026-11-05") == "from 2026-11-03 to 2026-11-05" and wo.window_words("2026-11-03", "2026-11-03") == "on 2026-11-03"
    rolling = wo.new_work_order("q", action_type="search_news", since_days=7)
    assert (rolling.news_from, rolling.news_to) == (None, None) and wo.verify_request(rolling) == []
    for faults, said in (({"news_to": None}, "go together"), ({"news_from": "2026-11-25", "news_to": "2026-11-24"}, "is after news_to"), ({"news_from": "2026-06-01", "news_to": "2026-10-01"}, "at most 90 days"), ({"news_from": "Nov 20", "news_to": "2026-11-20"}, "real dates written YYYY-MM-DD"),
                         ({"since_days": 7}, "since_days is not used with a dated window"), ({"source": "filings", "action_type": "search_filing"}, "for the news only")):
        broken = dated_order()
        for name, value in faults.items():
            setattr(broken, name, value)
        reasons = wo.verify_request(broken)
        assert any(said in reason for reason in reasons), (faults, reasons)


# ---------------------------------------------------------------- D5: the search is held to the window, in code

def at(day: date, hour: int, minute: int, *, plus_days: int = 0) -> datetime:
    """A moment on the Eastern clock of `day` (plus some days), as UTC: the daylight saving of that day is the zone's, never a 24-hour sum."""
    return datetime.combine(day + timedelta(days=plus_days), time(hour, minute), tzinfo=EASTERN).astimezone(timezone.utc)


def a_week(tmp_path, day: date):
    """One TSLA article at each edge of the Eastern day `day` (and a day before and after it, and one in the middle of the day before), all of them 'tesla car'."""
    moments = {"first-minute": at(day, 0, 5), "noon": at(day, 12, 0), "last-minute": at(day, 23, 30), "before-by-five-minutes": at(day, 23, 55, plus_days=-1), "after-by-five-minutes": at(day, 0, 5, plus_days=1), "day-before-noon": at(day, 12, 0, plus_days=-1)}
    store = fill(make_store(tmp_path / f"chroma-{day}", KeywordEmbeddings()), [chunk(name, 0, 1, "tesla car", ["TSLA"], when) for name, when in moments.items()])
    return store, {"first-minute", "noon", "last-minute"}


@pytest.mark.parametrize("day", [date(2026, 11, 20), date(2026, 3, 8)])      # an ordinary day, and the day the clocks go forward (23 hours long: no 24-hour arithmetic gets its edges right)
def test_search_news_held_to_an_eastern_day_by_since_and_until_returns_exactly_that_days_articles_and_a_models_since_days_can_only_narrow(tmp_path, day):
    store, inside = a_week(tmp_path, day)
    window = dict(since=trading_clock.start_of_day(day), until=trading_clock.end_of_day(day), now=trading_clock.end_of_day(day), store=store, k=20)
    assert {h.article_id for h in search_news("tesla car", tickers="TSLA", **window)} == inside      # 23:30 ET of the day is in; 00:05 ET of the next day, 23:55 ET of the day before and the middle of the day before are out
    assert {h.article_id for h in search_news("tesla car", tickers="TSLA", since_days=30, **window)} == inside      # a since_days of the model's cannot widen it
    three_days = {**window, "since": trading_clock.start_of_day(day - timedelta(days=2))}      # a window of three days: the day before is in it ...
    assert {"day-before-noon", "before-by-five-minutes"} <= {h.article_id for h in search_news("tesla car", tickers="TSLA", **three_days)}
    narrowed = {h.article_id for h in search_news("tesla car", tickers="TSLA", since_days=1, **three_days)}      # ... and a since_days of 1 counts back one day (24 hours) from the window's end and narrows it
    assert inside <= narrowed and "day-before-noon" not in narrowed
    with pytest.raises(ValueError, match="since must be a datetime with a timezone"):
        search_news("tesla car", since=datetime(2026, 11, 20), store=store)


def test_a_search_with_no_since_is_what_it_was(tmp_path):
    store, _ = a_week(tmp_path, date(2026, 11, 20))
    assert {h.article_id for h in search_news("tesla car", tickers="TSLA", store=store, k=20)} == {"first-minute", "noon", "last-minute", "before-by-five-minutes", "after-by-five-minutes", "day-before-noon"}


def test_the_tools_within_a_window_hold_every_search_to_it_the_model_cannot_move_it_and_the_message_names_it(tmp_path):
    day = date(2026, 11, 20)
    store, inside = a_week(tmp_path, day)
    tools = DataClerkTools(toolbox=object(), store=store)
    held = tools.within(day, day)
    assert held is not tools and tools._window is None and held._window == (day, day) and held.store is store      # a copy: the tools it was called on are not changed
    assert {h["article_id"] for h in held.retrieve_news("tesla car", tickers="TSLA", since_days=30, k=20)["hits"]} == inside      # the model asked for 30 days: still the one day
    assert {h["article_id"] for h in held.retrieve_news("tesla car", tickers="TSLA", k=20, newest_first=True)["hits"]} == inside
    assert len(tools.retrieve_news("tesla car", tickers="TSLA", k=20)["hits"]) == 6      # the tools it was copied from still see everything
    nothing = held.retrieve_news("tesla car", tickers="META")
    assert nothing["found"] is False and nothing["reason"] == "no_article_matches_filters" and "(tickers META, from 2026-11-20 to 2026-11-20)" in nothing["message"] and "the last" not in nothing["message"]
    assert "from 2026-11-20 to 2026-11-22" in tools.within(day, date(2026, 11, 22)).retrieve_news("tesla car", tickers="META")["message"]
    for bad in (("2026-11-20", day), (day, datetime(2026, 11, 20)), (date(2026, 11, 21), day)):
        with pytest.raises(ValueError):
            tools.within(*bad)
    assert "within" not in DATA_CLERK_TOOLS and "fetch_news_window" not in DATA_CLERK_TOOLS      # the model's tool list is what it was


def test_a_past_day_that_is_bound_still_caps_the_window(tmp_path):
    day = date(2026, 11, 20)
    store, _ = a_week(tmp_path, day)
    held = DataClerkTools(toolbox=object(), store=store).within(day, day)
    with trading_clock.as_of(date(2026, 11, 19)):      # the Proving Ground is at 2026-11-19: nothing of the 20th is known yet
        assert held.retrieve_news("tesla car", tickers="TSLA", k=20)["hits"] == []


def scripted_clerk(script, store, **options) -> tuple[DataClerk, ScriptedModel]:
    model = ScriptedModel(script)
    agent = DataClerk(toolbox=object(), store=store, model=model, **options)
    return agent, model


def test_request_news_on_a_dated_order_asks_in_words_holds_the_models_search_and_the_codes_sweep_to_the_day_and_changes_nothing_else(tmp_path):
    day = date(2026, 11, 20)
    store, inside = a_week(tmp_path, day)
    agent, model = scripted_clerk([call("retrieve_news", query="tesla car", tickers=["TSLA"], since_days=30, k=20), say("Tesla news [1] [2] [3].")], store)
    order = agent.request_news(dated_order(), TODAY)
    asked = model.seen[0][1][-1].content      # the one human message of the question
    assert "Summarize the news of TSLA on 2026-11-20" in asked and "use news from 2026-11-20 (that day only, US Eastern time)" in asked and "the search is already limited to that window: do not set since_days" in asked and "from the last" not in asked
    assert order.status == "data_ready" and order.errors == [] and {s["article_id"] for s in order.data.sources} == inside and len(order.data.references) == 3      # the model asked for 30 days; it got the one day
    assert [c["name"] for c in order.data.tool_calls] == ["retrieve_news"] and order.data.sweep["news"]["by"] == "model" and agent.tools._window is None      # the Data Clerk's own tools are not windowed
    quiet, _ = scripted_clerk([say("Nothing to add.")], store)      # a model that searches nothing: the code's sweep searches the news, and it is held to the day too
    swept = quiet.request_news(dated_order(), TODAY)
    assert swept.data.sweep["news"]["by"] == "code" and list(swept.data.sweep) == ["news"] and {s["article_id"] for s in swept.data.sources} == inside and all(s["swept"] for s in swept.data.sources)
    span, _ = scripted_clerk([say("Nothing to add.")], store)
    assert {s["article_id"] for s in span.request_news(dated_order("2026-11-19", "2026-11-21"), TODAY).data.sources} == inside | {"before-by-five-minutes", "after-by-five-minutes", "day-before-noon"}      # a span is its days, both included
    broken = dated_order()
    broken.since_days = 7      # both kinds of window on one order: the Data Clerk refuses it before any model call
    refused = scripted_clerk([], store)[0].request_news(broken, TODAY)
    assert refused.status == "data_failed" and any("since_days is not used with a dated window" in e for e in refused.errors)
    assert agent._within(wo.new_work_order("q", action_type="search_news", since_days=7)) is agent and agent._within(dated_order()) is not agent and NEWS_ONLY == ("retrieve_news",)


def test_the_question_of_a_rolling_order_is_what_it_was():
    rolling = wo.new_work_order("Summarize the news of TSLA for the last 7 days", requester="desk", action_type="search_news", source="news", subject=["TSLA"], company="Tesla Inc", since_days=7, top_k=12, live=True)
    asked = DataClerk._news_question(rolling)
    assert "use news from the last 7 days" in asked and "do not set since_days" not in asked and "US Eastern" not in asked


# ---------------------------------------------------------------- D7: the fetch

class Paid:
    """The paid news calls and nothing else: a recorder (the one stand-in of the fetch tests). A window and a rolling fetch are told apart by the method called."""

    def __init__(self):
        self.calls = []

    def fetch_news(self, symbol, from_date=7, **options):
        self.calls.append(("fetch_news", symbol, from_date, options))
        return {"symbol": symbol, "fetched": 40, "written": 3, "duplicates": 37, "indexed": {}}

    def fetch_news_window(self, symbol, start, end):
        self.calls.append(("fetch_news_window", symbol, start, end))
        return {"symbol": symbol, "fetched": 11, "written": 2, "duplicates": 9, "indexed": {}}


def test_refresh_news_with_a_window_fetches_its_days_and_the_day_after_the_rolling_fetch_is_as_it_was_and_the_guards_hold():
    agent = RecordedAnswer(allow_fetch=True)
    paid = agent.tools = Paid()
    ok = [{"step": "news", "symbol": s, "outcome": "ok", "detail": {"fetched": 11, "written": 2, "duplicates": 9}} for s in ("TSLA", "AAPL")]
    assert agent.refresh_news(["TSLA", "$AAPL"], start=date(2026, 11, 20), end=date(2026, 11, 20)) == ok
    assert paid.calls == [("fetch_news_window", "TSLA", date(2026, 11, 20), date(2026, 11, 21)), ("fetch_news_window", "AAPL", date(2026, 11, 20), date(2026, 11, 21))]      # the UTC day after: an evening article of an Eastern day is the next UTC day
    paid.calls.clear()
    agent.refresh_news(["TSLA"], 7)
    assert paid.calls == [("fetch_news", "TSLA", 7, {})]
    with pytest.raises(ValueError, match="needs both start and end"):
        agent.refresh_news(["TSLA"], 7, start=date(2026, 11, 20))
    with trading_clock.as_of(date(2026, 11, 13)):
        (refused,) = agent.refresh_news(["TSLA"], start=date(2026, 11, 3), end=date(2026, 11, 3))
        assert refused["outcome"] == "refused" and trading_clock.NEVER_FETCHES in refused["detail"]
    (none,) = RecordedAnswer().refresh_news(["TSLA"], start=date(2026, 11, 3), end=date(2026, 11, 3))
    assert none["outcome"] == "refused" and "without allow_fetch" in none["detail"] and paid.calls == [("fetch_news", "TSLA", 7, {})]

    class Failing(Paid):
        def fetch_news_window(self, symbol, start, end):
            raise RuntimeError("the news API said no")

    broken = RecordedAnswer(allow_fetch=True)
    broken.tools = Failing()
    (failed,) = broken.refresh_news(["TSLA"], start=date(2026, 11, 20), end=date(2026, 11, 20))
    assert failed["outcome"] == "error" and failed["detail"] == "RuntimeError: the news API said no"


def test_the_tools_fetch_a_window_through_the_toolbox_with_a_start_and_an_end_and_index_what_is_new(monkeypatch):
    class Recording:
        def __init__(self):
            self.calls = []

        def fetch_news(self, **kwargs):
            self.calls.append(kwargs)
            return {"symbol": kwargs["symbol"], "fetched": 11, "matched": 5, "written": 2, "duplicates": 3}

    indexed = []
    monkeypatch.setattr("prometheus.tools.data_clerk.ingest_symbol", lambda symbol, store=None: indexed.append(symbol) or dict.fromkeys(("articles", "chunks", "added", "updated", "unchanged", "deleted"), 0))      # nothing is embedded
    toolbox = Recording()
    result = DataClerkTools(toolbox=toolbox).fetch_news_window("$tsla", date(2026, 11, 20), date(2026, 11, 21))
    assert toolbox.calls == [{"symbol": "TSLA", "start": date(2026, 11, 20), "end": date(2026, 11, 21)}] and indexed == ["TSLA"] and (result["fetched"], result["written"], result["symbol"]) == (11, 2, "TSLA")
    with trading_clock.as_of(date(2026, 11, 13)), pytest.raises(ToolLimitError, match="never fetches"):
        DataClerkTools(toolbox=toolbox).fetch_news_window("TSLA", date(2026, 11, 3), date(2026, 11, 4))
    assert len(toolbox.calls) == 1


# ---------------------------------------------------------------- D5, D6, D7, D8: the desk, on the stored news

@pytest.fixture(scope="module")
def real_embeddings():
    try:
        return get_embeddings()
    except Exception as error:   # no local copy of the model and no network
        pytest.skip(f"the embedding model is not available: {type(error).__name__}")


@pytest.fixture(scope="module")
def real_index(real_embeddings, tmp_path_factory):
    """The real stored index, COPIED to a temporary folder and opened there (Chroma writes to the folder it opens: the real one is never opened by a test)."""
    if not (STORE / "chroma.sqlite3").exists():
        pytest.skip("the real vector store is not on this machine")
    folder = tmp_path_factory.mktemp("vectorstore")
    shutil.copytree(STORE, folder / "index")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VECTOR_STORE_PATH", str(folder / "index"))
        yield folder / "index"
    shutil.rmtree(folder, ignore_errors=True)


def run_the_dated_desk(db, typed, agent, **request):
    """The whole desk (the Critic and the Trader built in) on a throwaway account, given a request in the user's own words, the Account Manager's model a replay of its real replies and today `TODAY`. As `test_news_view.run_the_desk`, over these forms."""
    account_id = db.account("10000")
    graph = desk.build_desk(clerk_factory=lambda order: agent, advisor_factory=never("Advisor"), critic_factory=never("Critic"), trader_factory=never("Trader"), with_critic=True, with_trader=True,
                            funds_factory=lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk()), store_factory=lambda: db.store, clock=lambda: MOMENT,
                            confirm=lambda intent: pytest.fail("a news request asks nothing: it was shown for a yes"), approver=lambda card: pytest.fail("a news request asks nothing"), clarify=lambda card: pytest.fail("a news request asks nothing"))
    started, final = [], None
    state0 = {"request": {"ask": {"text": typed, "default_count": 3}, "account": {"account": account_id, "mode": "ask"}, "trade": {"mode": "paper", "book": True, "model": False}, **request}, "order": None, "errors": [], "resolution": None, "funds": None,
              "path": [], "seconds": {}, "with_trader": True, "plan": None}
    for kind, chunk_ in graph.stream(state0, stream_mode=["tasks", "values"]):
        if kind == "tasks" and "result" not in chunk_ and "error" not in chunk_:
            started.append(chunk_["name"])
        elif kind == "values":
            final = chunk_
    return account_id, started, final


def test_the_desk_answers_a_request_for_a_past_day_with_that_days_stored_news_only(db, real_index, monkeypatch):
    """The bug, on the real stored news: "What was the news on TSLA on 2026-11-20?" used to be answered with the latest news (the window was counted back from now)."""
    monkeypatch.setattr(trading_clock, "calendar_today", lambda real_now=None, config=None: TODAY)      # the Account Manager's model and its checks work on the same day as the capture
    searching = [call("retrieve_news", query="Tesla news of the day", tickers=["TSLA"], since_days=30, k=12), say("The day's news is in the sources [1].")]      # the model searches and asks for 30 days: it gets the one day
    agent = DataClerk(model=ScriptedModel(searching), sweep=True)      # no store of its own: the real index (a copy)
    agent.lookup_interval = 0
    account_id, started, final = run_the_dated_desk(db, DAY_OF_THE_ASK, agent, refuse_news_fetch=True)
    order, plan = final["order"], final["plan"]
    assert started == ["account_manager", "data_clerk"] == final["path"] and final["errors"] == [] and order.status == "data_ready"      # nothing but the two agents ran
    assert (order.news_from, order.news_to, order.since_days, order.time_period, order.source, order.live) == ("2026-11-20", "2026-11-20", None, "2026-11-20", "news", True) and order.query == "Summarize the news of TSLA on 2026-11-20"
    assert (plan["news_from"], plan["news_to"], plan["news_days"]) == ("2026-11-20", "2026-11-20", None) and plan["intent"].describe() == "show the news of TSLA on Friday 2026-11-20 (nothing will be bought)"
    sources = order.data.sources
    assert len(sources) >= 3 and any(not s.get("swept") for s in sources) and order.data.sweep["news"]["by"] == "model"      # the day has real stored TSLA news, and the model's own search found it
    assert all(datetime.fromisoformat(s["published_utc"].replace("Z", "+00:00")).astimezone(EASTERN).date() == date(2026, 11, 20) for s in sources), [s["published_utc"] for s in sources]      # every source is inside the Eastern day
    assert order.data.refresh == [{"step": "news", "symbol": None, "outcome": "refused", "detail": "the Data Clerk was built without allow_fetch: nothing is fetched (fetching uses paid APIs and writes files)"}]      # the test-only switch: no fetch
    assert db.store.account_balance(account_id) == 10000 and db.store.list_portfolio(account_id).positions == []
    view = views.news_view(order, plan)
    assert (view["window"], view["days"]) == ({"from": "2026-11-20", "to": "2026-11-20"}, None) and views.news_headline(view) == "The news of TSLA on 2026-11-20" and views.run_verdict(None, None, action="news", order=order)["headline"] == "The news of TSLA on 2026-11-20"
    assert {i["date"] for i in view["items"]} <= {"2026-11-20", "2026-11-21"}      # an item's date is its UTC date: an evening article of the Eastern day is the next UTC day


def test_the_desk_builds_the_order_of_a_span_and_fetches_its_days(db, monkeypatch):
    monkeypatch.setattr(trading_clock, "calendar_today", lambda real_now=None, config=None: TODAY)
    seen, fetched = [], []

    def refused(self, **kwargs):      # the toolbox's own fetch, replaced by a recorder that fails as the cut network does: no file under news/ is opened
        fetched.append(kwargs)
        raise RuntimeError("no fetch in tests")

    monkeypatch.setattr(DataToolbox, "fetch_news", refused)

    class Watching(RecordedAnswer):
        def refresh_news(self, symbols_, days=7, start=None, end=None):
            seen.append((self.allow_fetch, list(symbols_), days, start, end))
            return super().refresh_news(symbols_, days, start, end)

    agent = Watching()
    _, started, final = run_the_dated_desk(db, "NVDA news from Nov 3 to Nov 5", agent)      # the news run asks its Data Clerk for the right to fetch, whatever the run says
    order = final["order"]
    assert started == ["account_manager", "data_clerk"] and (order.news_from, order.news_to, order.since_days, order.time_period) == ("2026-11-03", "2026-11-05", None, "2026-11-03 to 2026-11-05") and order.query == "Summarize the news of NVDA from 2026-11-03 to 2026-11-05"
    assert seen == [(True, ["NVDA"], 7, date(2026, 11, 3), date(2026, 11, 5))] and "from 2026-11-03 to 2026-11-05" in agent.asked[0]["question"] and "do not set since_days" in agent.asked[0]["question"]
    assert fetched == [{"symbol": "NVDA", "start": date(2026, 11, 3), "end": date(2026, 11, 6)}]      # the window and the day after it, through the real tools
    assert order.data.refresh == [{"step": "news", "symbol": "NVDA", "outcome": "error", "detail": "RuntimeError: no fetch in tests"}]      # the fetch failed: the run went on with the stored news and said so
    assert views.news_headline(views.news_view(order, final["plan"])) == "The news of NVDA from 2026-11-03 to 2026-11-05"
    rolling = run_the_dated_desk(db, "TSLA news last week", Watching())[2]
    assert (rolling["order"].since_days, rolling["order"].news_from, rolling["order"].news_to, rolling["order"].time_period) == (7, None, None, "last 7 days") and seen[-1] == (True, ["TSLA"], 7, None, None) and rolling["plan"]["news_from"] is None
    assert fetched[-1]["from_date"] == 7 and "start" not in fetched[-1]      # the rolling fetch is what it was


# ---------------------------------------------------------------- D8: what is shown

def test_a_dated_run_is_shown_by_its_days_in_the_headline_the_session_and_the_empty_day_says_so_plainly(db, tmp_path, monkeypatch):
    monkeypatch.setattr(trading_clock, "calendar_today", lambda real_now=None, config=None: TODAY)
    agent = RecordedAnswer()
    _, _, final = run_the_dated_desk(db, DAY_OF_THE_ASK, agent, refuse_news_fetch=True)
    order = final["order"]
    view = views.news_view(order, final["plan"])
    assert view["window"] == {"from": "2026-11-20", "to": "2026-11-20"} and view["days"] is None and views.news_window_text(view) == "on 2026-11-20"
    span = views.news_view(dated_order("2026-11-03", "2026-11-05"))
    assert span["window"] == {"from": "2026-11-03", "to": "2026-11-05"} and views.news_headline(span) == "The news of TSLA from 2026-11-03 to 2026-11-05"
    rolling = views.news_view(wo.new_work_order("q", subject=["TSLA"], since_days=7))
    assert rolling["window"] is None and rolling["days"] == 7 and views.news_headline(rolling) == "The news of TSLA for the last 7 days"
    empty = {**view, "items": [], "status": "data_ready"}
    lines = session.news_lines(empty, final["plan"]["intent"])
    assert lines[1] == "NEWS  The news of TSLA on 2026-11-20" and "  understood: show the news of TSLA on Friday 2026-11-20 (nothing will be bought)" in lines and "ARTICLES  none: the stored news has nothing about this on 2026-11-20" in lines
    assert "ARTICLES  none: the stored news has nothing about this in the window" in session.news_lines({**empty, "window": None, "days": 7})      # a rolling window says what it said


def test_the_session_prints_a_past_days_news_and_saves_the_intent_with_its_days(db, tmp_path, monkeypatch):
    monkeypatch.setattr(trading_clock, "calendar_today", lambda real_now=None, config=None: TODAY)
    forms = FORMS[DAY_OF_THE_ASK]["replies"]
    parts = session.Parts(store=db.store, clerk=RecordedAnswer(), advisor=never("Advisor"), critic=never("Critic"), trader=never("Trader"), new_manager=lambda: AccountManager(model=Replay(forms), model_id="recorded", clerk=clerk()))
    out = tmp_path / "saved.json"
    args = session.build_parser().parse_args(["--name", "News Test", "--age", "40", "--risk", "moderate", "--cash", "10000", "--mode", "ask", "--prompt", DAY_OF_THE_ASK, "--refuse-news-fetch", "--save", str(out)])
    lines: list[str] = []
    code = session.run_session(args, session.check_args(session.build_parser(), args), parts, ask=lambda prompt_: pytest.fail(f"the user was asked {prompt_!r}"), out=lines.append)
    import re
    text = "\n".join(lines)
    db.made.extend(int(n) for n in re.findall(r"account (\d+) is kept", text))
    assert code == 0 and "NEWS  The news of TSLA on 2026-11-20" in text and "understood: show the news of TSLA on Friday 2026-11-20 (nothing will be bought)" in text and "for the last" not in text
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["intent"]["news_from"] == saved["intent"]["news_to"] == "2026-11-20" and "news_days" not in saved["intent"] and saved["order"]["news_from"] == "2026-11-20" and saved["order"]["since_days"] is None


def test_the_config_has_the_age_limit_of_a_dated_window():
    assert load_config().news.max_age_days == 365 and load_config().news.max_days == 90
    assert trading_clock.start_of_day(date(2026, 11, 20)) == datetime(2026, 11, 20, 5, 0, tzinfo=timezone.utc) and trading_clock.end_of_day(date(2026, 11, 20)) == datetime(2026, 11, 21, 4, 59, 59, 999999, tzinfo=timezone.utc)
    assert trading_clock.start_of_day(date(2026, 3, 8)) == datetime(2026, 3, 8, 5, 0, tzinfo=timezone.utc) and trading_clock.end_of_day(date(2026, 3, 8)) == datetime(2026, 3, 9, 3, 59, 59, 999999, tzinfo=timezone.utc)      # the day the clocks go forward is 23 hours long
    assert trading_clock.calendar_today(datetime(2026, 11, 29, 3, 0, tzinfo=timezone.utc)) == date(2026, 11, 28)      # 22:00 ET: still Saturday in New York
    with trading_clock.as_of(date(2026, 11, 13)):
        assert trading_clock.calendar_today() == date(2026, 11, 13)      # the bound day is the Proving Ground's today
