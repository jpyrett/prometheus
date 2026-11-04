"""Tests for prometheus.common.work_order and DataClerk.request_data (Part 8 of the RAG guide).

Run:  pytest tests/test_work_order.py -q

No network, no key, no cost: a scripted model stands in for the chat model (see test_data_clerk_agent.py) and the stores are in
temporary folders.
"""
import json
import shutil
from datetime import date
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings
from langchain_chroma import Chroma

from prometheus.agents.data_clerk.data_clerk_agent import DataClerk, Result
from prometheus.common import work_order as wo
from prometheus.rag import store as rag_store
from test_data_clerk_agent import ScriptedModel, call, say  # the same scripted model and turns as the agent tests
from test_data_clerk_tools import FakeToolbox
from test_symbols import META, PG_NAME, PG_TICKER, AnyQuery, FakeSearch  # the REAL Finnhub rows the symbol tests already use
from test_filings import DistinctEmbeddings
from test_rag_retriever import DAY, NOW, KeywordEmbeddings, chunk, fill, make_store


@pytest.fixture
def index(tmp_path):
    return fill(make_store(tmp_path / "chroma", KeywordEmbeddings()), [
        chunk("a1", 0, 1, "apple iphone phone", ["AAPL"], NOW - 5 * DAY),
        chunk("a2", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 4 * DAY),
        chunk("n1", 0, 1, "nvidia chip chip chip", ["NVDA"], NOW - 3 * DAY)])


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    news, press, filings = tmp_path / "news", tmp_path / "press_releases", tmp_path / "filings"
    news.mkdir()
    press.mkdir()
    filings.mkdir()
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(news))
    monkeypatch.setenv("PRESS_RELEASES_OUTPUT_DIR", str(press))
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(filings))
    return FakeToolbox(news, press, filings)


def clerk(script, index, toolbox, **options):
    model = ScriptedModel(script)
    return DataClerk(toolbox=toolbox, store=index, model=model, **options), model


def order(**fields):
    return wo.new_work_order(fields.pop("query", "What is new with Apple's phone?"), **fields)


# ---------------------------------------------------------------- the order itself

def test_a_new_order_has_an_id_a_timestamp_and_the_designs_fields():
    o = order(requester="jacob", subject=["AAPL"], since_days=7, top_k=3, criteria={"sentiment": "bullish"})
    assert len(o.request_id) == 12 and o.created_utc.endswith("Z") and o.requester == "jacob"
    assert (o.action_type, o.source, o.status, o.top_k, o.since_days, o.live) == ("search_news", "news", "new", 3, 7, False)
    assert o.data is None and o.recommendations == [] and o.errors == [] and o.criteria == {"sentiment": "bullish"}
    assert order().request_id != order().request_id


def test_an_order_is_plain_json_that_comes_back_the_same():
    o = order(subject="memory stocks", time_period="last 7 days")
    again = wo.WorkOrder.model_validate_json(o.model_dump_json())
    assert again == o and json.loads(o.model_dump_json())["schema_version"] == 1


# ---------------------------------------------------------------- verify_request

def test_a_good_request_has_no_reasons_against_it():
    assert wo.verify_request(order()) == []
    assert wo.verify_request(order(since_days=90, top_k=20)) == []


@pytest.mark.parametrize("fields, reason", [
    ({"query": "   "}, "the query is empty"),
    ({"query": "x" * 1001}, "longer than 1000"),
    ({"action_type": "launch_rocket"}, "is not one of"),
    ({"action_type": "trade"}, "not handled by the Data Clerk"),
    ({"top_k": 0}, "top_k must be"), ({"top_k": 21}, "top_k must be"), ({"top_k": True}, "top_k must be"),
    ({"since_days": 0}, "since_days must be"), ({"since_days": 91}, "since_days must be"), ({"since_days": "7"}, "since_days must be"),
])
def test_each_bad_request_is_refused_with_its_reason(fields, reason):
    reasons = wo.verify_request(order(**fields))
    assert any(reason in r for r in reasons), reasons


@pytest.mark.parametrize("source", ["social", "Press_Releases", "News", "Filings", ""])  # press_releases is wired since entry 34 and filings since entry 35 (it was in this list until then)
def test_a_source_that_is_not_wired_is_refused_and_news_is_the_default(source):
    assert wo.verify_request(order()) == [] and order().source == "news"
    reasons = wo.verify_request(order(source=source))
    assert any(f"source {source!r} is not wired yet (only: news, press_releases, filings)" in r for r in reasons), reasons


def test_an_order_that_already_has_its_data_is_not_worked_twice():
    o = order()
    o.status = "data_ready"
    assert any("already has its data" in r for r in wo.verify_request(o))


# ---------------------------------------------------------------- request_data: the agent call

def test_request_data_puts_the_answer_the_cited_chunks_and_the_sources_on_the_same_order(index, toolbox):
    c, model = clerk([call("retrieve_news", query="apple iphone phone", tickers="AAPL", k=2), say("Apple has a phone out [1].")], index, toolbox)
    o = order(subject=["AAPL"], since_days=7, top_k=2)
    same = c.request_data(o)
    assert same is o and o.status == "data_ready" and o.errors == []
    assert o.data.answer == "Apple has a phone out [1]." and [r["chunk_id"] for r in o.data.references] == ["a1#0"]
    assert [s["chunk_id"] for s in o.data.sources] == ["a1#0", "a2#0"] and o.data.sources[0]["text"] == "apple iphone phone"
    assert o.data.completed_utc.endswith("Z") and o.data.turns == 2 and o.data.tokens_in > 0
    asked = model.seen[0][1][1].content  # the question the model received
    assert asked.startswith("What is new with Apple's phone?") and "subject: AAPL" in asked and "last 7 days" in asked and "at most 2 sources" in asked


def test_a_live_order_asks_for_fresh_news_and_a_normal_one_does_not(index, toolbox):
    c, model = clerk([say("ok"), say("ok")], index, toolbox)
    c.request_data(order(subject=["SNDK"], live=True))
    c.request_data(order(subject=["SNDK"]))
    live, normal = model.seen[0][1][1].content, model.seen[1][1][1].content
    assert "fresh news is required: call fetch_news once for each ticker in the subject first, then search" in live  # no window on this order, so no from_date
    assert "fresh news" not in normal


def test_the_window_of_a_live_order_becomes_the_from_date_of_the_fetch(index, toolbox):
    c, model = clerk([say("ok"), say("ok")], index, toolbox)
    c.request_data(order(subject=["MSFT"], live=True, since_days=14))
    c.request_data(order(subject=["MSFT"], live=True))  # no window: no from_date is asked for
    windowed, open_ended = model.seen[0][1][1].content, model.seen[1][1][1].content
    assert "call fetch_news once for each ticker in the subject first, with from_date=14, then search" in windowed
    assert "from_date" not in open_ended and "call fetch_news once for each ticker in the subject first, then search" in open_ended


def test_the_company_name_on_the_order_is_given_to_the_data_clerk_next_to_the_symbol(index, toolbox):
    c, model = clerk([say("ok"), say("ok")], index, toolbox)
    c.request_data(order(subject=["PG"], company="Procter & Gamble Co"))
    c.request_data(order(subject=["PG"]))
    assert "subject: PG (Procter & Gamble Co)" in model.seen[0][1][1].content
    assert "subject: PG;" in model.seen[1][1][1].content  # no company: just the symbol, followed by the next hint
    assert wo.WorkOrder.model_validate_json(order(subject=["PG"], company="Procter & Gamble Co").model_dump_json()).company == "Procter & Gamble Co"


def test_the_order_survives_json_with_its_data(index, toolbox):
    c, _ = clerk([call("retrieve_news", query="apple iphone phone", tickers="AAPL"), say("Yes [1].")], index, toolbox)
    o = c.request_data(order())
    assert wo.WorkOrder.model_validate_json(o.model_dump_json()) == o


def test_a_search_that_finds_nothing_is_a_result_not_a_failure(index, toolbox):
    c, _ = clerk([call("retrieve_news", query="cake"), say("The stored news does not cover that.")], index, toolbox)
    o = c.request_data(order(query="cake?"))
    assert o.status == "data_ready" and o.data.no_supporting_text and o.data.references == [] and o.errors == []


def test_a_wrong_citation_that_is_not_corrected_makes_the_order_fail_with_the_reason(index, toolbox):
    c, _ = clerk([call("retrieve_news", query="apple iphone phone", tickers="AAPL"), say("Apple [7]."), say("Apple [8].")], index, toolbox)
    o = c.request_data(order())
    assert o.status == "data_failed" and any("cites [8]" in e for e in o.errors) and o.data.answer == "Apple [8]."


def test_a_model_failure_fails_the_order_and_keeps_the_reason(index, toolbox):
    class Broken(ScriptedModel):
        def bind_tools(self, tools):
            class Bound:
                def invoke(self, messages):
                    raise ConnectionError("down")

            return Bound()

    o = DataClerk(toolbox=toolbox, store=index, model=Broken([])).request_data(order())
    assert o.status == "data_failed" and o.errors == ["the model call failed (ConnectionError)"] and o.data.answer == ""


def test_a_request_the_data_clerk_should_not_take_is_refused_without_calling_the_model(index, toolbox):
    c, model = clerk([], index, toolbox)  # an empty script: any model call would raise
    o = c.request_data(order(action_type="trade"))
    assert o.status == "data_failed" and o.data is None and any("not handled by the Data Clerk" in e for e in o.errors) and model.seen == []


def test_an_order_that_is_already_done_is_not_asked_again(index, toolbox):
    c, model = clerk([call("retrieve_news", query="apple iphone phone", tickers="AAPL"), say("Yes [1].")], index, toolbox)
    o = c.request_data(order())
    again = c.request_data(o)  # the second call would need more script
    assert again is o and again.status == "data_ready" and again.errors == [] and len(model.seen) == 2  # no new model call, nothing changed


# ---------------------------------------------------------------- verify_action: the checks on the data

def result(**fields) -> Result:
    base = dict(question="q", answer="Apple [1].", references=[{"number": 1, "chunk_id": "a#0"}], sources=[{"number": 1, "chunk_id": "a#0"}],
                tool_calls=[{"name": "retrieve_news", "args": {}, "outcome": "ok"}])  # a search_news order must have searched
    return Result(**{**base, **fields})


def test_data_that_is_consistent_passes():
    o = wo.attach_result(order(), result())
    assert o.status == "data_ready" and wo.verify_action(o) == []


def test_there_is_nothing_to_verify_before_the_clerk_has_worked():
    assert wo.verify_action(order()) == ["the order has no data yet"]


def test_a_reference_to_a_chunk_that_is_not_among_the_sources_fails():
    o = wo.attach_result(order(), result(references=[{"number": 1, "chunk_id": "ghost#0"}]))
    assert o.status == "data_failed" and any("not among the sources: ghost#0" in e for e in o.errors)


def test_more_references_than_sources_fails():
    refs = [{"number": 1, "chunk_id": "a#0"}, {"number": 2, "chunk_id": "a#0"}]
    o = wo.attach_result(order(), result(references=refs))
    assert o.status == "data_failed" and any("more references than sources" in e for e in o.errors)


def test_an_empty_answer_fails_even_if_the_clerk_did_not_say_so():
    o = wo.attach_result(order(), result(answer="   ", references=[], sources=[]))
    assert o.status == "data_failed" and o.errors.count("the answer is empty") == 1


def test_what_the_clerk_itself_flagged_carries_over():
    o = wo.attach_result(order(), result(problems=["cites [8], which are not among the sources"]))
    assert o.status == "data_failed" and "cites [8], which are not among the sources" in o.errors


def test_a_news_search_order_answered_without_a_search_fails_with_the_reason():
    """Found in the "flying car" run: the Data Clerk answered a bare term from memory or with a refusal, the order came back data_ready,
    and nothing said that no search had been made."""
    reason = "no search of the stored news succeeded (no retrieve_news call), so the answer rests on no source"
    for calls in ([], [{"name": "fetch_news", "args": {}, "outcome": "ok"}],                       # no search at all, or a fetch only
                  [{"name": "retrieve_news", "args": {}, "outcome": "limit"}],                       # a search the budget refused
                  [{"name": "retrieve_news", "args": {}, "outcome": "error"}]):                      # a search that failed
        o = wo.attach_result(order(), result(answer="Flying cars are an emerging technology.", references=[], sources=[], tool_calls=calls))
        assert o.status == "data_failed" and o.errors == [reason], calls


def test_a_model_failure_is_the_only_reason_and_does_not_add_a_no_search_reason():
    o = wo.attach_result(order(), result(answer="", references=[], sources=[], tool_calls=[], problems=["the model call failed (ConnectionError)"]))
    assert o.status == "data_failed" and o.errors == ["the model call failed (ConnectionError)"]


def test_an_order_that_is_not_a_news_search_is_not_held_to_the_no_search_rule():
    o = order(action_type="fetch_quote")
    wo.attach_result(o, result(references=[], sources=[], tool_calls=[]))
    assert "no search of the stored news succeeded" not in " ".join(o.errors)


def test_an_unsearched_answer_is_caught_through_request_data_and_the_advisor_then_refuses_it(index, toolbox):
    from prometheus.agents.advisor.advisor import Advisor
    c, _ = clerk([say("I can only provide information about companies.")], index, toolbox)
    o = c.request_data(order(query="flying car", subject="flying car"))
    assert o.status == "data_failed" and any("no search of the stored news succeeded" in e for e in o.errors)
    refusing = ScriptedModel([])  # any call would raise
    Advisor(model=refusing).advise(o)
    assert o.recommendations == [] and refusing.seen == [] and any("needs an order whose data is ready" in e for e in o.errors)


def test_every_order_tells_the_data_clerk_that_it_is_a_news_search():
    c, model = clerk([say("ok")], None, None)
    c.request_data(order(query="flying car", subject="flying car"))
    assert "this is a request to search the stored news: call retrieve_news before you answer, whatever the subject is" in model.seen[0][1][1].content


def test_no_supporting_text_together_with_sources_contradicts_itself():
    o = wo.attach_result(order(), result(no_supporting_text=True))
    assert o.status == "data_failed" and any("no supporting text but also lists sources" in e for e in o.errors)


# ================================================================ press releases (design choices entry 34)
#
# `source="press_releases"`; the symbol is the order's subject (argument #1) and the company name its `company` (argument #2). The Data Clerk resolves them in CODE
# before any model runs: the symbol if it is found, else the company name (exact matches only); nothing found is a refused order and costs nothing. The lookups
# below answer from the REAL Finnhub rows of test_symbols.py (PG, META). The chunks are toy chunks for the fake word embedder, as in the news tests.

PG_CO = "Procter & Gamble Co"  # PG's name as Finnhub's name search describes it


@pytest.fixture
def press_index(tmp_path):
    return fill(make_store(tmp_path / "chroma_pr", KeywordEmbeddings()), [
        chunk("pr:p1", 0, 1, "cake tax", ["PG"], NOW - 3 * DAY, publisher="Press release"),
        chunk("pr:m1", 0, 1, "meta tax", ["META"], NOW - 2 * DAY, publisher="Press release")])


def directory(**extra):
    """The symbol directory: real rows for PG (by symbol and by name) and META."""
    return FakeSearch(PG=PG_TICKER, META=META, **{PG_CO: PG_NAME}, **extra)


def press_clerk(script, index, toolbox, press_index, search=None, **options):
    model = ScriptedModel(script)
    return DataClerk(toolbox=toolbox, store=index, press_store=press_index, model=model, symbol_search=search or directory(), **options), model


def press_order(**fields):
    fields = {"source": "press_releases", "since_days": 365, "top_k": 3, "query": "What do the press releases say?", **fields}
    return order(**fields)


# ---------------------------------------------------------------- verify_request

def test_a_press_release_request_is_wired_and_needs_a_symbol_a_company_name_or_both():
    assert wo.WIRED_SOURCES == ("news", "press_releases", "filings") and wo.LOOKUP_SOURCES == ("press_releases", "filings")
    assert wo.verify_request(press_order(subject=["PG"])) == []
    assert wo.verify_request(press_order(subject="PG")) == []                     # the symbol as plain text
    assert wo.verify_request(press_order(company=PG_CO)) == []                    # the name alone
    assert wo.verify_request(press_order(subject=["PG"], company=PG_CO)) == []    # both
    for nothing in ({}, {"subject": None}, {"subject": []}, {"subject": ["  "]}, {"company": "   "}, {"subject": "", "company": None}):
        assert "a press release request needs a symbol, a company name, or both" in wo.verify_request(press_order(**nothing)), nothing


def test_a_press_release_request_is_for_one_company_at_a_time():
    assert any("one symbol at a time" in r for r in wo.verify_request(press_order(subject=["PG", "META"])))


def test_a_press_release_window_may_reach_two_years_and_news_still_stops_at_ninety_days():
    assert wo.verify_request(press_order(subject=["PG"], since_days=730)) == []
    assert any("since_days must be a whole number of days from 1 to 730" in r for r in wo.verify_request(press_order(subject=["PG"], since_days=731)))
    assert any("from 1 to 730" in r for r in wo.verify_request(press_order(subject=["PG"], since_days=0)))
    assert wo.verify_request(order(since_days=90)) == [] and any("from 1 to 90" in r for r in wo.verify_request(order(since_days=91)))  # news: unchanged


def test_a_news_order_needs_no_symbol_and_is_not_asked_for_one():
    assert wo.verify_request(order(subject=None)) == [] and wo.verify_request(order(subject="memory stocks")) == []


def test_the_order_has_a_resolved_field_that_is_empty_until_the_lookup_and_survives_json():
    o = press_order(subject=["PG"])
    assert o.resolved is None
    o.resolved = {"typed_symbol": "pg", "typed_company": None, "symbol": "PG", "company": "Procter & Gamble Co/The", "how": "ticker"}
    assert wo.WorkOrder.model_validate_json(o.model_dump_json()).resolved == o.resolved


# ---------------------------------------------------------------- verify_action: a press release order must have searched the press releases

def result_with(*tools, outcome="ok") -> Result:
    return Result(question="q", answer="PG [1].", references=[{"number": 1, "chunk_id": "a#0"}], sources=[{"number": 1, "chunk_id": "a#0"}],
                  tool_calls=[{"name": name, "args": {}, "outcome": outcome} for name in tools])


def test_a_press_release_order_is_satisfied_by_a_press_release_search_and_not_by_a_news_search():
    done = wo.attach_result(press_order(subject=["PG"]), result_with("retrieve_press_releases"))
    assert done.status == "data_ready" and done.errors == []
    wrong = wo.attach_result(press_order(subject=["PG"]), result_with("retrieve_news"))
    assert wrong.status == "data_failed" and wrong.errors == ["no search of the stored press releases succeeded (no retrieve_press_releases call), so the answer rests on no source"]


def test_a_news_order_is_satisfied_by_a_news_search_and_not_by_a_press_release_search():
    wrong = wo.attach_result(order(), result_with("retrieve_press_releases"))
    assert wrong.status == "data_failed" and wrong.errors == ["no search of the stored news succeeded (no retrieve_news call), so the answer rests on no source"]


def test_a_press_release_search_that_failed_does_not_count_as_a_search():
    o = wo.attach_result(press_order(subject=["PG"]), result_with("retrieve_press_releases", outcome="error"))
    assert o.status == "data_failed" and any("no retrieve_press_releases call" in e for e in o.errors)


# ---------------------------------------------------------------- request_data for a press release order: the symbol first, then the company name

def test_a_symbol_is_resolved_searched_and_the_order_says_what_was_decided(index, toolbox, press_index):
    search = directory()
    c, model = press_clerk([call("retrieve_press_releases", query="cake tax", tickers="PG", k=3), say("Procter & Gamble announced a tax cake [1].")],
                           index, toolbox, press_index, search)
    o = press_order(subject=["pg"])  # lower case, as typed
    assert c.request_data(o) is o
    assert o.status == "data_ready" and o.errors == [] and search.asked == ["PG"]
    assert o.resolved == {"typed_symbol": "pg", "typed_company": None, "symbol": "PG", "company": "Procter & Gamble Co/The", "how": "ticker"}
    assert o.subject == ["PG"] and o.company == "Procter & Gamble Co/The"  # the order now carries what the lookup found, not what was typed
    assert [r["chunk_id"] for r in o.data.references] == ["pr:p1#0"] and o.data.sources[0]["publisher"] == "Press release"
    asked = model.seen[0][1][1].content
    assert "subject: PG (Procter & Gamble Co/The)" in asked and "press releases from the last 365 days" in asked and "at most 3 sources" in asked
    assert "PRESS RELEASES: call retrieve_press_releases (not retrieve_news)" in asked and 'tickers=["PG"]' in asked
    assert "fresh press releases" not in asked and "retrieve_news before" not in asked  # not live, and not the news instruction


def test_a_company_name_alone_is_resolved_to_its_symbol(index, toolbox, press_index):
    search = directory()
    c, model = press_clerk([call("retrieve_press_releases", query="cake tax", tickers="PG"), say("Yes [1].")], index, toolbox, press_index, search)
    o = press_order(company=PG_CO)
    c.request_data(o)
    assert o.status == "data_ready" and search.asked == [PG_CO]  # no symbol was given, so no symbol lookup
    assert o.resolved == {"typed_symbol": None, "typed_company": PG_CO, "symbol": "PG", "company": PG_CO, "how": "name"}
    assert o.subject == ["PG"] and "subject: PG (Procter & Gamble Co)" in model.seen[0][1][1].content


def test_when_both_are_given_and_the_symbol_is_found_the_symbol_is_used_even_if_the_name_is_another_companys(index, toolbox, press_index):
    search = directory()
    c, _ = press_clerk([call("retrieve_press_releases", query="meta tax", tickers="META"), say("Yes [1].")], index, toolbox, press_index, search)
    o = press_order(subject=["META"], company=PG_CO)
    c.request_data(o)
    assert o.status == "data_ready" and search.asked == ["META"]  # the company name was never looked up
    assert o.resolved["symbol"] == "META" and o.resolved["how"] == "ticker" and o.resolved["typed_company"] == PG_CO  # what was typed is kept
    assert o.subject == ["META"] and o.company == "Meta Platforms Inc-Class A"


def test_when_the_symbol_is_not_found_the_company_name_is_used(index, toolbox, press_index):
    search = directory()
    c, _ = press_clerk([call("retrieve_press_releases", query="cake tax", tickers="PG"), say("Yes [1].")], index, toolbox, press_index, search)
    o = press_order(subject=["ZZZZ"], company=PG_CO)
    c.request_data(o)
    assert o.status == "data_ready" and search.asked == ["ZZZZ", PG_CO]  # the symbol first, then the name
    assert o.resolved == {"typed_symbol": "ZZZZ", "typed_company": PG_CO, "symbol": "PG", "company": PG_CO, "how": "name"} and o.subject == ["PG"]


@pytest.mark.parametrize("fields, errors", [
    ({"subject": ["ZZZZ"]}, ["Symbol: ZZZZ is not found"]),
    ({"company": "Nonexistent Holdings"}, ["Company Name not found."]),
    ({"subject": ["ZZZZ"], "company": "Nonexistent Holdings"}, ["Company Name not found."]),
    ({"company": "Proctor & Gamble"}, ["Company Name not found."]),  # a typo is not matched
])
def test_nothing_found_refuses_the_order_with_the_message_and_costs_nothing(index, toolbox, press_index, fields, errors):
    c, model = press_clerk([], index, toolbox, press_index)  # an empty script: any model call would raise
    o = press_order(**fields)
    assert c.request_data(o) is o
    assert o.status == "data_failed" and o.errors == errors and o.data is None and o.resolved is None
    assert model.seen == [] and toolbox.calls == []  # no model call, and nothing fetched (nothing paid, nothing written)


def test_an_exact_name_that_matches_several_listings_is_refused_with_the_detail_and_the_same_message(index, toolbox, press_index):
    abroad = [r for r in PG_TICKER if r["symbol"] in ("PG.MX", "PG.VI")]
    c, model = press_clerk([], index, toolbox, press_index, AnyQuery(abroad))
    o = press_order(company="Procter & Gamble Co/The")
    c.request_data(o)
    assert o.status == "data_failed" and o.errors[0] == "Company Name not found." and "PG.MX" in o.errors[1] and "Give the symbol" in o.errors[1] and model.seen == []


def test_a_failed_lookup_is_reported_as_a_failure_and_not_as_not_found(index, toolbox, press_index):
    from prometheus.tools.symbols import SymbolLookupError

    def rate_limited(query):
        raise SymbolLookupError("the symbol lookup failed (HTTP 429); type the ticker in capitals to skip it", reason="the symbol lookup failed (HTTP 429)")

    c, model = press_clerk([], index, toolbox, press_index, rate_limited)
    o = press_order(subject=["PG"])
    c.request_data(o)
    assert o.status == "data_failed" and o.errors == ["the request could not be checked: the symbol lookup failed (HTTP 429)"]  # no "type the ticker" advice: no way out here
    assert not any("not found" in e for e in o.errors) and model.seen == [] and toolbox.calls == []


def test_a_symbol_with_text_that_cannot_be_a_symbol_falls_back_to_the_company_name(index, toolbox, press_index):
    search = directory()
    c, _ = press_clerk([call("retrieve_press_releases", query="cake tax", tickers="PG"), say("Yes [1].")], index, toolbox, press_index, search)
    o = press_order(subject=["Procter and Gamble!"], company=PG_CO)  # a person typed the name in the symbol field
    c.request_data(o)
    assert o.status == "data_ready" and search.asked == [PG_CO] and o.resolved["symbol"] == "PG"  # not a symbol: no lookup for it, then the name


def test_an_order_with_a_request_problem_is_refused_before_the_lookup(index, toolbox, press_index):
    search = directory()
    c, model = press_clerk([], index, toolbox, press_index, search)
    o = press_order(subject=["PG", "META"])
    c.request_data(o)
    assert o.status == "data_failed" and any("one symbol at a time" in e for e in o.errors) and search.asked == [] and model.seen == []


def test_a_news_order_never_uses_the_symbol_directory(index, toolbox, press_index):
    c, _ = press_clerk([call("retrieve_news", query="apple iphone phone", tickers="AAPL"), say("Yes [1].")], index, toolbox, press_index,
                       lambda query: pytest.fail("a news order must not look anything up"))
    o = order(subject=["AAPL"], company="Apple")
    c.request_data(o)
    assert o.status == "data_ready" and o.resolved is None and o.subject == ["AAPL"] and o.company == "Apple"  # untouched


def test_a_live_press_release_order_asks_for_a_fetch_before_the_search_and_the_fetch_indexes_the_new_releases(index, toolbox, press_index):
    c, model = press_clerk([call("fetch_press_releases", symbol="PG", from_date=365), call("retrieve_press_releases", query="cake tax", tickers="PG"),
                            say("Yes [1].")], index, toolbox, press_index, allow_fetch=True)
    o = press_order(subject=["PG"], live=True)
    c.request_data(o)
    assert "fresh press releases are required: call fetch_press_releases once for PG first, with from_date=365, then search" in model.seen[0][1][1].content
    assert [(t["name"], t["outcome"]) for t in o.data.tool_calls] == [("fetch_press_releases", "ok"), ("retrieve_press_releases", "ok")]
    assert toolbox.calls == [("fetch_press_releases", ("PG",), dict(from_date=365))]
    # the fake toolbox wrote the three real fixture releases to PG_PRs.txt, and the tool indexed them into the press collection
    assert {"pr:44107d69ebe0e861#0", "pr:5a31fe63040096fc#0", "pr:519ab77bd1631dd7#0"} <= set(press_index.get()["ids"])
    assert o.status == "data_ready"


def test_a_press_release_order_that_never_searched_the_press_releases_is_a_failed_order(index, toolbox, press_index):
    c, _ = press_clerk([call("retrieve_news", query="apple iphone phone", tickers="PG"), say("The stored news says nothing about it.")], index, toolbox, press_index)
    o = press_order(subject=["PG"])  # the model searched the NEWS (and found nothing, and cited nothing): for a press release order that is no search
    c.request_data(o)
    assert o.status == "data_failed" and o.errors == ["no search of the stored press releases succeeded (no retrieve_press_releases call), so the answer rests on no source"]


def test_the_question_tells_the_model_to_put_the_company_name_in_its_search_query_when_the_order_has_one(index, toolbox, press_index):
    """Found by the first real run: 'executive changes' scored 0.23 (under the 0.40 bar) and 'Apple executive changes' 0.51, so a query without the name made the
    Data Clerk say, wrongly, that the releases did not cover it. Measured with the real model on AAPL (6 runs each): the query named the company 0 of 6 times
    for 'executive changes' and for 'dividend' before this sentence, and 6 of 6 after."""
    c, model = press_clerk([say("ok")], index, toolbox, press_index)
    o = press_order(subject=["PG"])
    c.request_data(o)
    asked = model.seen[0][1][1].content
    assert 'write the search query with the company\'s name "Procter & Gamble Co/The" and the topic' in asked
    assert "a query without the name scores under the relevance bar" in asked


def test_without_a_company_name_there_is_no_such_instruction():
    o = press_order(subject=["PG"])  # not resolved: no company on the order
    assert "company's name" not in DataClerk._press_release_question(o)


def test_a_company_name_without_its_generic_words_resolves_through_the_whole_data_clerk(index, toolbox, press_index):
    """The flexible search end to end: 'Procter & Gamble' (no 'Co') finds the real row 'Procter & Gamble Co', and the order records the directory's name."""
    c, _ = press_clerk([call("retrieve_press_releases", query="cake tax", tickers="PG"), say("Yes [1].")], index, toolbox, press_index, AnyQuery(PG_NAME))
    o = press_order(company="Procter & Gamble")
    c.request_data(o)
    assert o.status == "data_ready" and o.resolved["symbol"] == "PG" and o.resolved["how"] == "name"
    assert o.resolved["typed_company"] == "Procter & Gamble" and o.resolved["company"] == "Procter & Gamble Co" and o.company == "Procter & Gamble Co"


def test_a_typo_in_the_company_name_is_still_refused_through_the_whole_data_clerk(index, toolbox, press_index):
    c, model = press_clerk([], index, toolbox, press_index, AnyQuery(PG_NAME))  # the directory returns the right rows; the typo must still be refused
    o = press_order(company="Proctor & Gamble")
    c.request_data(o)
    assert o.status == "data_failed" and o.errors == ["Company Name not found."] and model.seen == [] and toolbox.calls == []


# ================================================================ filings (design choices entry 35)
#
# `source="filings"` is asked for in exactly the way press releases are, and resolved by the same code (`symbols.resolve_target`) with the same messages. The tests below
# are the press release ones again with the source changed, because that is the claim: one set of rules, two sources. The chunks are toy chunks for the fake word embedder.


@pytest.fixture
def filings_index(tmp_path):
    """The filings collection, with the real store's search settings and the filings tests' distinct-vector embedder: the live-order test below indexes the 168 chunks of the real
    fixture into it and then searches for one toy chunk with a ticker filter. With Chroma's defaults and the plain fake (most of those chunks have the SAME vector) that search found
    nothing about one run in two, and the scripted model ran out of turns ("the model call failed (AssertionError)"): found by repeating the full suite (design/week4.md section 17)."""
    store = Chroma(client=chromadb.PersistentClient(path=str(tmp_path / "chroma_f"), settings=Settings(anonymized_telemetry=False)), collection_name="test",
                   embedding_function=DistinctEmbeddings(), collection_configuration={"hnsw": rag_store.HNSW})
    return fill(store, [
        chunk("sec:0000000001-25-000001:items", 0, 1, "cake tax", ["PG"], NOW - 3 * DAY, publisher="SEC filing"),
        chunk("sec:0000000002-25-000002:items", 0, 1, "meta tax", ["META"], NOW - 2 * DAY, publisher="SEC filing")])


def filings_clerk(script, index, toolbox, filings_index, search=None, **options):
    model = ScriptedModel(script)
    return DataClerk(toolbox=toolbox, store=index, filings_store=filings_index, model=model, symbol_search=search or directory(), **options), model


def filings_order(**fields):
    fields = {"source": "filings", "action_type": "search_filing", "since_days": 365, "top_k": 3, "query": "What do the filings say?", **fields}
    return order(**fields)


def test_a_filings_request_needs_a_symbol_a_company_name_or_both_and_is_for_one_company_at_a_time():
    assert wo.verify_request(filings_order(subject=["PG"])) == [] and wo.verify_request(filings_order(subject="PG")) == []
    assert wo.verify_request(filings_order(company=PG_CO)) == [] and wo.verify_request(filings_order(subject=["PG"], company=PG_CO)) == []
    for nothing in ({}, {"subject": None}, {"subject": []}, {"subject": ["  "]}, {"company": "   "}, {"subject": "", "company": None}):
        assert "a filings request needs a symbol, a company name, or both" in wo.verify_request(filings_order(**nothing)), nothing
    assert "a filings request is for one symbol at a time" in wo.verify_request(filings_order(subject=["PG", "META"]))
    assert "a press release request needs a symbol, a company name, or both" in wo.verify_request(press_order())  # the press release wording is unchanged


def test_a_filings_window_may_reach_two_years_like_press_releases_and_news_still_stops_at_ninety_days():
    assert wo.MAX_DAYS == {"news": 90, "press_releases": 730, "filings": 730}
    assert wo.verify_request(filings_order(subject=["PG"], since_days=730)) == []
    assert any("from 1 to 730" in r for r in wo.verify_request(filings_order(subject=["PG"], since_days=731)))
    assert any("from 1 to 730" in r for r in wo.verify_request(filings_order(subject=["PG"], since_days=0)))
    assert any("from 1 to 90" in r for r in wo.verify_request(order(since_days=91)))  # news: unchanged


def test_a_filings_order_is_satisfied_by_a_filings_search_and_by_no_other():
    assert wo.SEARCH_TOOL["filings"] == ("retrieve_filings", "filings")
    done = wo.attach_result(filings_order(subject=["PG"]), result_with("retrieve_filings"))
    assert done.status == "data_ready" and done.errors == []
    for other in ("retrieve_news", "retrieve_press_releases"):
        wrong = wo.attach_result(filings_order(subject=["PG"]), result_with(other))
        assert wrong.status == "data_failed" and wrong.errors == ["no search of the stored filings succeeded (no retrieve_filings call), so the answer rests on no source"]
    for source, tool in (("news", "retrieve_filings"), ("press_releases", "retrieve_filings")):  # and a filings search satisfies neither of the others
        assert wo.attach_result(order(source=source, subject=["PG"]), result_with(tool)).status == "data_failed"
    failed = wo.attach_result(filings_order(subject=["PG"]), result_with("retrieve_filings", outcome="error"))
    assert failed.status == "data_failed" and any("no retrieve_filings call" in e for e in failed.errors)


def test_a_symbol_is_resolved_for_filings_searched_and_the_order_says_what_was_decided(index, toolbox, filings_index):
    search = directory()
    c, model = filings_clerk([call("retrieve_filings", query="cake tax", tickers="PG", k=3), say("Procter & Gamble told the SEC about a tax cake [1].")],
                             index, toolbox, filings_index, search)
    o = filings_order(subject=["pg"])  # lower case, as typed
    assert c.request_data(o) is o
    assert o.status == "data_ready" and o.errors == [] and search.asked == ["PG"]
    assert o.resolved == {"typed_symbol": "pg", "typed_company": None, "symbol": "PG", "company": "Procter & Gamble Co/The", "how": "ticker"}
    assert o.subject == ["PG"] and o.company == "Procter & Gamble Co/The"
    assert [r["chunk_id"] for r in o.data.references] == ["sec:0000000001-25-000001:items#0"] and o.data.sources[0]["publisher"] == "SEC filing"
    asked = model.seen[0][1][1].content
    assert "subject: PG (Procter & Gamble Co/The)" in asked and "filings from the last 365 days" in asked and "at most 3 sources" in asked
    assert "SEC FILINGS: call retrieve_filings (not retrieve_news)" in asked and 'tickers=["PG"]' in asked
    assert "fresh filings" not in asked and "retrieve_press_releases" not in asked and "retrieve_news before" not in asked  # not live, and not another source's instruction


def test_a_company_name_alone_is_resolved_for_filings_to_its_symbol(index, toolbox, filings_index):
    search = directory()
    c, model = filings_clerk([call("retrieve_filings", query="cake tax", tickers="PG"), say("Yes [1].")], index, toolbox, filings_index, search)
    o = filings_order(company=PG_CO)
    c.request_data(o)
    assert o.status == "data_ready" and search.asked == [PG_CO]  # no symbol was given, so no symbol lookup
    assert o.resolved == {"typed_symbol": None, "typed_company": PG_CO, "symbol": "PG", "company": PG_CO, "how": "name"}
    assert o.subject == ["PG"] and "subject: PG (Procter & Gamble Co)" in model.seen[0][1][1].content


def test_for_filings_too_the_symbol_wins_when_both_are_given_even_if_the_name_is_another_companys(index, toolbox, filings_index):
    search = directory()
    c, _ = filings_clerk([call("retrieve_filings", query="meta tax", tickers="META"), say("Yes [1].")], index, toolbox, filings_index, search)
    o = filings_order(subject=["META"], company=PG_CO)
    c.request_data(o)
    assert o.status == "data_ready" and search.asked == ["META"]  # the company name was never looked up
    assert o.resolved["symbol"] == "META" and o.resolved["how"] == "ticker" and o.resolved["typed_company"] == PG_CO and o.subject == ["META"]


def test_for_filings_too_a_symbol_that_is_not_found_falls_back_to_the_company_name(index, toolbox, filings_index):
    search = directory()
    c, _ = filings_clerk([call("retrieve_filings", query="cake tax", tickers="PG"), say("Yes [1].")], index, toolbox, filings_index, search)
    o = filings_order(subject=["ZZZZ"], company=PG_CO)
    c.request_data(o)
    assert o.status == "data_ready" and search.asked == ["ZZZZ", PG_CO]  # the symbol first, then the name
    assert o.resolved == {"typed_symbol": "ZZZZ", "typed_company": PG_CO, "symbol": "PG", "company": PG_CO, "how": "name"} and o.subject == ["PG"]


@pytest.mark.parametrize("fields, errors", [
    ({"subject": ["ZZZZ"]}, ["Symbol: ZZZZ is not found"]),
    ({"company": "Nonexistent Holdings"}, ["Company Name not found."]),
    ({"subject": ["ZZZZ"], "company": "Nonexistent Holdings"}, ["Company Name not found."]),  # both not found: the error goes back to the user
    ({"company": "Proctor & Gamble"}, ["Company Name not found."]),  # a typo is not matched
])
def test_for_filings_nothing_found_refuses_the_order_with_the_message_and_costs_nothing(index, toolbox, filings_index, fields, errors):
    c, model = filings_clerk([], index, toolbox, filings_index)  # an empty script: any model call would raise
    o = filings_order(**fields)
    assert c.request_data(o) is o
    assert o.status == "data_failed" and o.errors == errors and o.data is None and o.resolved is None
    assert model.seen == [] and toolbox.calls == []  # no model call, and nothing fetched (nothing paid, nothing written)


def test_for_filings_a_name_that_matches_several_listings_is_refused_with_the_detail_and_the_same_message(index, toolbox, filings_index):
    abroad = [r for r in PG_TICKER if r["symbol"] in ("PG.MX", "PG.VI")]
    c, model = filings_clerk([], index, toolbox, filings_index, AnyQuery(abroad))
    o = filings_order(company="Procter & Gamble Co/The")
    c.request_data(o)
    assert o.status == "data_failed" and o.errors[0] == "Company Name not found." and "PG.MX" in o.errors[1] and model.seen == []


def test_for_filings_a_failed_lookup_is_reported_as_a_failure_and_not_as_not_found(index, toolbox, filings_index):
    from prometheus.tools.symbols import SymbolLookupError

    def rate_limited(query):
        raise SymbolLookupError("the symbol lookup failed (HTTP 429); type the ticker in capitals to skip it", reason="the symbol lookup failed (HTTP 429)")

    c, model = filings_clerk([], index, toolbox, filings_index, rate_limited)
    o = filings_order(subject=["PG"])
    c.request_data(o)
    assert o.status == "data_failed" and o.errors == ["the request could not be checked: the symbol lookup failed (HTTP 429)"]
    assert not any("not found" in e for e in o.errors) and model.seen == [] and toolbox.calls == []


def test_a_filings_order_with_a_request_problem_is_refused_before_the_lookup(index, toolbox, filings_index):
    search = directory()
    c, model = filings_clerk([], index, toolbox, filings_index, search)
    for fields, message in (({"subject": ["PG", "META"]}, "one symbol at a time"), ({}, "needs a symbol, a company name, or both"), ({"subject": ["PG"], "since_days": 731}, "from 1 to 730")):
        o = filings_order(**fields)
        c.request_data(o)
        assert o.status == "data_failed" and any(message in e for e in o.errors)
    assert search.asked == [] and model.seen == []


def test_a_live_filings_order_asks_for_a_fetch_before_the_search_and_the_fetch_indexes_the_new_filings(index, toolbox, filings_index):
    c, model = filings_clerk([call("fetch_filings", symbol="PG", from_date=365), call("retrieve_filings", query="cake tax", tickers="PG"), say("Yes [1].")],
                             index, toolbox, filings_index, allow_fetch=True)
    o = filings_order(subject=["PG"], live=True)
    c.request_data(o)
    assert "fresh filings are required: call fetch_filings once for PG first, with from_date=365, then search" in model.seen[0][1][1].content
    assert [(t["name"], t["outcome"]) for t in o.data.tool_calls] == [("fetch_filings", "ok"), ("retrieve_filings", "ok")]
    assert toolbox.calls == [("fetch_filings", ("PG",), dict(from_date=365)), ("fetch_filing_text", ("PG",), dict(from_date=365))]  # one fetch call of the model: both steps
    # the fake toolbox wrote the six real fixture filings to PG.txt, and the tool indexed them into the filings collection
    assert {"sec:0000320193-25-000077:items#0", "sec:0000320193-26-000006#0"} <= set(filings_index.get()["ids"])
    assert o.status == "data_ready"


def test_a_filings_order_that_never_searched_the_filings_is_a_failed_order(index, toolbox, filings_index):
    c, _ = filings_clerk([call("retrieve_news", query="apple iphone phone", tickers="PG"), say("The stored news says nothing about it.")], index, toolbox, filings_index)
    o = filings_order(subject=["PG"])  # the model searched the NEWS: for a filings order that is no search
    c.request_data(o)
    assert o.status == "data_failed" and o.errors == ["no search of the stored filings succeeded (no retrieve_filings call), so the answer rests on no source"]


def test_the_filings_question_has_the_company_name_hint_and_the_press_release_question_is_word_for_word_what_it_was():
    """The two questions are built by one function with the tool names as arguments (`_lookup_question`). The press release wording was written, measured on a real run
    (entry 34, D14) and pinned before filings existed: it must not change by a word."""
    o = press_order(subject=["PG"], company="Procter & Gamble Co/The", live=True)
    assert DataClerk._question(o) == (
        "What do the press releases say?\n\n(Work order: subject: PG (Procter & Gamble Co/The); use press releases from the last 365 days; use at most 3 sources; "
        "this is a request about the company's PRESS RELEASES: call retrieve_press_releases (not retrieve_news) with tickers=[\"PG\"] before you answer, whatever the question is; "
        "write the search query with the company's name \"Procter & Gamble Co/The\" and the topic: a query without the name scores under the relevance bar and finds nothing, "
        "and then \"the releases do not cover it\" would be a wrong answer; "
        "fresh press releases are required: call fetch_press_releases once for PG first, with from_date=365, then search.)")
    f = filings_order(subject=["PG"], company="Procter & Gamble Co/The", live=True)
    assert DataClerk._question(f) == (
        "What do the filings say?\n\n(Work order: subject: PG (Procter & Gamble Co/The); use filings from the last 365 days; use at most 3 sources; "
        "this is a request about the company's SEC FILINGS: call retrieve_filings (not retrieve_news) with tickers=[\"PG\"] before you answer, whatever the question is; "
        "write the search query with the company's name \"Procter & Gamble Co/The\" and the topic: a query without the name scores under the relevance bar and finds nothing, "
        "and then \"the filings do not cover it\" would be a wrong answer; "
        "fresh filings are required: call fetch_filings once for PG first, with from_date=365, then search.)")
    assert "company's name" not in DataClerk._question(filings_order(subject=["PG"]))  # not resolved: no company on the order, no such instruction


def test_a_company_name_without_its_generic_words_resolves_for_filings_through_the_whole_data_clerk_and_a_typo_does_not(index, toolbox, filings_index):
    c, _ = filings_clerk([call("retrieve_filings", query="cake tax", tickers="PG"), say("Yes [1].")], index, toolbox, filings_index, AnyQuery(PG_NAME))
    o = filings_order(company="Procter & Gamble")
    c.request_data(o)
    assert o.status == "data_ready" and o.resolved["symbol"] == "PG" and o.resolved["company"] == "Procter & Gamble Co" and o.company == "Procter & Gamble Co"
    c, model = filings_clerk([], index, toolbox, filings_index, AnyQuery(PG_NAME))  # the directory returns the right rows; the typo must still be refused
    typo = filings_order(company="Proctor & Gamble")
    c.request_data(typo)
    assert typo.status == "data_failed" and typo.errors == ["Company Name not found."] and model.seen == [] and toolbox.calls == []


# ================================================================ the action type search_filing (design choices entry 37)
#
# Filings have their own action, as the design PDF names it ("search_news, search_filing, fetch_quote ..."). It is handled by request_data like search_news, and the work order
# checks it against the source in BOTH directions: search_filing is for filings and for nothing else, and a filings order has no other action. Press releases and news keep
# search_news (entry 34).

def test_search_filing_is_an_action_type_that_the_data_clerk_handles_and_the_order_keeps_through_json():
    assert "search_filing" in wo.ACTION_TYPES and wo.SEARCH_ACTIONS == ("search_news", "search_filing") and wo.ACTION_SOURCE == {"search_filing": "filings"}
    o = filings_order(subject=["PG"])
    assert o.action_type == "search_filing" and wo.verify_request(o) == []
    assert wo.WorkOrder.model_validate_json(o.model_dump_json()).action_type == "search_filing"
    assert order().action_type == "search_news" and press_order(subject=["PG"]).action_type == "search_news"  # the others are unchanged


@pytest.mark.parametrize("source", ["news", "press_releases", "social"])
def test_search_filing_with_a_source_that_is_not_filings_is_refused_with_the_reason(source):
    reasons = wo.verify_request(order(action_type="search_filing", source=source, subject=["PG"], company="Procter & Gamble Co"))
    assert f"action_type 'search_filing' is for source 'filings', but the order's source is {source!r}" in reasons


def test_a_filings_order_with_the_news_action_is_refused_with_the_reason():
    assert wo.verify_request(filings_order(subject=["PG"], action_type="search_news")) == ["a filings order's action_type is 'search_filing', not 'search_news'"]


def test_the_other_actions_are_still_refused_as_not_handled_and_the_message_names_both_that_are():
    for action in ("trade", "fetch_quote", "search_socials", "calculate_financials"):
        reasons = wo.verify_request(order(action_type=action))
        assert reasons == [f"action_type {action!r} is not handled by the Data Clerk's request_data yet (only search_news, search_filing)"], reasons
    assert any("is not one of" in r and "search_filing" in r for r in wo.verify_request(order(action_type="launch_rocket")))  # the list of known actions has it
    assert len(wo.verify_request(filings_order(subject=["PG"], action_type="trade"))) == 1  # one reason, not two: the mismatch is only checked for a search action


def test_a_search_filing_order_must_have_searched_the_filings_before_it_is_ready():
    wrong = wo.attach_result(filings_order(subject=["PG"]), result_with("retrieve_news"))
    assert wrong.status == "data_failed" and wrong.errors == ["no search of the stored filings succeeded (no retrieve_filings call), so the answer rests on no source"]
    assert wo.attach_result(filings_order(subject=["PG"]), result_with("retrieve_filings")).status == "data_ready"


def test_the_data_clerk_refuses_a_mismatched_filings_order_before_any_lookup_and_before_any_model(index, toolbox, filings_index):
    search = directory()
    c, model = filings_clerk([], index, toolbox, filings_index, search)  # an empty script: any model call would raise
    for o in (filings_order(subject=["PG"], action_type="search_news"), order(action_type="search_filing", source="news", subject=["PG"])):
        assert c.request_data(o) is o
        assert o.status == "data_failed" and len(o.errors) == 1 and o.data is None and o.resolved is None
    assert search.asked == [] and model.seen == [] and toolbox.calls == []


# ================================================================ the results of each source on the order (design choices entry 39)
#
# The order's `data` carries, for news, press releases and filings, which numbered sources came from each corpus and which the answer cites, and the whole filing evidence per symbol,
# so that the next agent reads them from the order. The groups are computed by code from the chunk ids; they point at `sources`, nothing is copied.

def test_a_chunk_id_says_which_corpus_it_is_from_and_the_prefixes_are_the_loaders():
    from prometheus.rag.loaders import FILING_ID_PREFIX, PRESS_RELEASE_ID_PREFIX, load_filings, load_news, load_press_releases
    fixtures = Path(__file__).parent / "fixtures"
    assert wo.CORPORA == ("news", "press_releases", "filings") and wo.CORPUS_PREFIX == {PRESS_RELEASE_ID_PREFIX: "press_releases", FILING_ID_PREFIX: "filings"}
    for doc in load_news(fixtures / "news_sample.txt"):
        assert wo.corpus_of(doc.id + "#0") == "news", doc.id  # the real news ids (finnhub:..., polygon:...)
    assert {wo.corpus_of(d.id + "#0") for d in load_press_releases(fixtures / "press_releases_sample.txt")} == {"press_releases"}
    assert {wo.corpus_of(d.id + "#0") for d in load_filings(fixtures / "filings_sample.txt")} == {"filings"}


def src(number, chunk_id):
    return {"number": number, "chunk_id": chunk_id, "headline": "h", "publisher": "p", "text": f"text {number}"}


def test_group_results_lists_the_numbers_by_corpus_and_which_of_them_are_cited():
    sources = [src(1, "polygon:abc#0"), src(2, "pr:111#0"), src(3, "sec:0000000001-25-000001:items#0"), src(4, "finnhub:77#1"), src(5, "sec:0000000001-25-000001:risk_factors#4")]
    references = [{"chunk_id": "finnhub:77#1"}, {"chunk_id": "sec:0000000001-25-000001:risk_factors#4"}]
    assert wo.group_results(sources, references) == {"news": {"shown": [1, 4], "cited": [4], "swept": []}, "press_releases": {"shown": [2], "cited": [], "swept": []},
                                                              "filings": {"shown": [3, 5], "cited": [5], "swept": []}}
    assert wo.group_results([], []) == {c: {"shown": [], "cited": [], "swept": []} for c in wo.CORPORA}  # all three keys are always there
    assert wo.group_results([{"chunk_id": "pr:x#0"}], []) == {c: {"shown": [], "cited": [], "swept": []} for c in wo.CORPORA}  # a source with no number cannot be pointed at


def data_result_with_three_corpora() -> Result:
    sources = [src(1, "polygon:abc#0"), src(2, "pr:111#0"), src(3, "sec:0000000001-25-000001:items#0")]
    return Result(question="q", answer="News [1], the release [2], the filing [3].", references=[{"number": n, "chunk_id": s["chunk_id"]} for n, s in ((1, sources[0]), (3, sources[2]))],
                  sources=sources, tool_calls=[{"name": "retrieve_news", "args": {}, "outcome": "ok"}])


def test_attach_result_puts_the_results_of_each_source_on_the_order_and_sources_of_gives_the_items():
    o = wo.attach_result(order(), data_result_with_three_corpora())
    d = o.data
    assert o.status == "data_ready" and d.results == {"news": {"shown": [1], "cited": [1], "swept": []}, "press_releases": {"shown": [2], "cited": [], "swept": []},
                                                      "filings": {"shown": [3], "cited": [3], "swept": []}}
    assert [s["number"] for s in d.sources_of("news")] == [1] and [s["chunk_id"] for s in d.sources_of("press_releases")] == ["pr:111#0"]
    assert d.sources_of("filings")[0]["text"] == "text 3" and d.sources_of("press_releases", cited_only=True) == [] and [s["number"] for s in d.sources_of("filings", cited_only=True)] == [3]
    assert len(d.sources) == 3  # nothing was copied: the groups are numbers that point at `sources`
    with pytest.raises(ValueError, match="corpus must be one of news, press_releases, filings"):
        d.sources_of("social")


def test_an_order_from_before_this_change_still_loads_and_has_no_results():
    old = json.loads(wo.attach_result(order(), data_result_with_three_corpora()).model_dump_json())
    for key in ("results", "filing_evidence", "sweep"):
        del old["data"][key]  # an order saved before these fields existed
    again = wo.WorkOrder.model_validate(old)
    assert again.data.results == {} and again.data.filing_evidence == {} and again.data.sources_of("news") == []


def test_the_results_and_the_filing_evidence_survive_json():
    r = data_result_with_three_corpora()
    r.filing_evidence = {"META": {"symbol": "META", "digest": "FILING EVIDENCE for META", "evidence": [], "balance_sheet": None, "gaps": ["g"]}}
    o = wo.attach_result(order(), r)
    again = wo.WorkOrder.model_validate_json(o.model_dump_json())
    assert again == o and again.data.filing_evidence["META"]["digest"] == "FILING EVIDENCE for META" and again.data.results["filings"]["cited"] == [3]


def three_corpus_clerk(script, tmp_path, toolbox, filings_index, press_index, files=("META",)):
    """The REAL Data Clerk over three stores (news, press releases, filings) and, for the evidence tool, the real stored filing files of the fixtures."""
    news = fill(make_store(tmp_path / "chroma_news_meta", KeywordEmbeddings()), [chunk("m1", 0, 1, "meta tax", ["META"], NOW - DAY)])
    fixtures = {"META": "filings_meta_sample.txt", "AAPL": "filings_sample.txt"}
    for symbol in files:
        shutil.copy(Path(__file__).parent / "fixtures" / fixtures[symbol], toolbox.filings_dir / f"{symbol}.txt")
    model = ScriptedModel(script)
    clerk = DataClerk(toolbox=toolbox, store=news, press_store=press_index, filings_store=filings_index, model=model, symbol_search=directory())
    clerk.tools.today = date(2026, 11, 1)
    return clerk, model


def test_the_data_clerk_puts_the_results_of_news_press_releases_and_filings_and_the_filing_evidence_on_the_order(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), call("retrieve_press_releases", query="meta tax", tickers="META"),
                               call("retrieve_filings", query="meta tax", tickers="META"), call("fetch_filings_recommendation", symbol="META"),
                               say("Meta: the news [1], its own release [2] and its filing [3] agree.")], tmp_path, toolbox, filings_index, press_index)
    o = filings_order(subject=["META"])
    c.request_data(o)
    d = o.data
    assert o.status == "data_ready" and o.errors == [] and [(t["name"], t["outcome"]) for t in d.tool_calls] == [
        ("retrieve_news", "ok"), ("retrieve_press_releases", "ok"), ("retrieve_filings", "ok"), ("fetch_filings_recommendation", "ok")]
    assert d.results == {"news": {"shown": [1], "cited": [1], "swept": []}, "press_releases": {"shown": [2], "cited": [2], "swept": []},
                         "filings": {"shown": [3], "cited": [3], "swept": []}}
    assert [s["publisher"] for s in d.sources] == ["Yahoo", "Press release", "SEC filing"]
    assert d.sources_of("news")[0]["chunk_id"] == "m1#0" and d.sources_of("press_releases")[0]["chunk_id"] == "pr:m1#0" and d.sources_of("filings")[0]["chunk_id"].startswith("sec:")
    ev = d.filing_evidence["META"]
    assert list(d.filing_evidence) == ["META"] and ev["symbol"] == "META" and ev["digest"].startswith("FILING EVIDENCE for META") and ev["evidence"]
    assert ev["balance_sheet"] is None and any(g.startswith("no balance sheet") for g in ev["gaps"])  # the real META records of the fixture: the 10-Q has no text here
    from prometheus.agents.data_clerk.filing_signals import FORBIDDEN_KEYS, keys_of
    assert not [k for k in keys_of(d.filing_evidence) if any(w in k.lower() for w in FORBIDDEN_KEYS)]  # it scores nothing, and the order carries none of it either


def test_evidence_that_failed_or_was_refused_is_not_on_the_order(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_filings", query="meta tax", tickers="META"), call("fetch_filings_recommendation", symbol="NVDA"),  # no NVDA file: refused with a reason
                               call("fetch_filings_recommendation", symbol="META"), call("fetch_filings_recommendation", symbol="AAPL"),  # the third attempt is past the budget of two
                               call("fetch_filings_recommendation", symbol="META"), say("Done [1].")], tmp_path, toolbox, filings_index, press_index, files=("META", "AAPL"))
    o = filings_order(subject=["META"])
    c.request_data(o)
    assert [t["outcome"] for t in o.data.tool_calls] == ["ok", "invalid", "ok", "limit", "limit"]
    assert list(o.data.filing_evidence) == ["META"]  # the failed NVDA call and the two refused calls left nothing on the order


def test_the_evidence_of_two_symbols_is_kept_apart_by_symbol(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_filings", query="meta tax", tickers="META"), call("fetch_filings_recommendation", symbol="META"),
                               call("fetch_filings_recommendation", symbol="AAPL"), say("Done [1].")], tmp_path, toolbox, filings_index, press_index, files=("META", "AAPL"))
    o = filings_order(subject=["META"])
    c.request_data(o)
    assert sorted(o.data.filing_evidence) == ["AAPL", "META"]
    assert o.data.filing_evidence["AAPL"]["symbol"] == "AAPL" and o.data.filing_evidence["META"]["symbol"] == "META"
    assert o.data.filing_evidence["AAPL"]["digest"].startswith("FILING EVIDENCE for AAPL") and o.data.filing_evidence["META"]["digest"].startswith("FILING EVIDENCE for META")


def test_a_filings_order_with_no_evidence_call_has_none_and_the_results_still_say_what_was_searched(index, toolbox, filings_index):
    c, _ = filings_clerk([call("retrieve_filings", query="cake tax", tickers="PG"), say("Yes [1].")], index, toolbox, filings_index)
    o = filings_order(subject=["PG"])
    c.request_data(o)
    assert o.data.filing_evidence == {} and o.data.results["filings"] == {"shown": [1], "cited": [1], "swept": []} and o.data.results["news"] == {"shown": [], "cited": [], "swept": []}


def test_the_advisor_reads_the_same_numbered_sources_the_results_point_at(tmp_path, toolbox, filings_index, press_index):
    """The next agent: the Advisor reads `data.sources` by number. The numbers of `data.results` are exactly the numbers it is shown, for every corpus."""
    from prometheus.agents.advisor.advisor import Advisor
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), call("retrieve_press_releases", query="meta tax", tickers="META"),
                               call("retrieve_filings", query="meta tax", tickers="META"), say("Meta [1] [2] [3].")], tmp_path, toolbox, filings_index, press_index)
    o = filings_order(subject=["META"])
    c.request_data(o)
    by_number, shown_text = Advisor._sources(o)
    assert sorted(by_number) == sorted(n for group in o.data.results.values() for n in group["shown"]) == [1, 2, 3]
    assert "SEC filing" in shown_text and "Press release" in shown_text and "Yahoo" in shown_text  # all three corpora are in what the Advisor reads


# ================================================================ the sweep: an order carries all three sources (design choices entry 40)
#
# After the model's loop the Data Clerk's CODE searches every corpus the model did not search and reads the stored filing evidence, so that the Advisor always finds the three
# sources on the order. The model never sees what the sweep finds, so its answer does not cite it; it runs after the model, so it cannot switch off the fetch tools.

def meta_order(**fields):
    """A NEWS order: the model's own search is the news one, so it passes verify_action whatever else the sweep adds."""
    return order(**{"subject": ["META"], "query": "meta tax", "since_days": 30, "top_k": 3, **fields})


def test_the_sweep_adds_the_corpora_the_model_did_not_search_after_the_models_own_sources(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), say("Meta's news [1].")], tmp_path, toolbox, filings_index, press_index)
    o = meta_order()
    c.request_data(o)
    d = o.data
    assert o.status == "data_ready" and o.errors == []
    assert [(s["number"], s["publisher"], s.get("swept", False)) for s in d.sources] == [(1, "Yahoo", False), (2, "Press release", True), (3, "SEC filing", True)]
    assert d.results == {"news": {"shown": [1], "cited": [1], "swept": []}, "press_releases": {"shown": [], "cited": [], "swept": [2]}, "filings": {"shown": [], "cited": [], "swept": [3]}}
    assert [r["number"] for r in d.references] == [1]  # the model never saw 2 and 3, so its answer does not cite them
    assert [t["name"] for t in d.tool_calls] == ["retrieve_news"]  # the sweep is not a tool call of the model, and it does not make a search order pass verify_action
    assert d.sweep["news"] == {"by": "model", "sources": 1, "note": ""} and d.sweep["press_releases"] == {"by": "code", "sources": 1, "note": ""}
    assert d.sweep["filings"] == {"by": "code", "sources": 1, "note": ""} and d.sweep["filing_evidence"]["by"] == "code" and d.sweep["filing_evidence"]["sources"] > 0
    assert [s["chunk_id"] for s in d.sources_of("press_releases")] == ["pr:m1#0"] and [s["number"] for s in d.sources_of("filings")] == [3] and d.sources_of("filings", cited_only=True) == []
    assert list(d.filing_evidence) == ["META"] and d.filing_evidence["META"]["evidence"]  # read by code: the model never called the evidence tool


def test_what_the_model_searched_itself_is_not_searched_again(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), call("retrieve_press_releases", query="meta tax", tickers="META"),
                               call("retrieve_filings", query="meta tax", tickers="META"), call("fetch_filings_recommendation", symbol="META"), say("Meta [1] [2] [3].")],
                              tmp_path, toolbox, filings_index, press_index)
    o = meta_order()
    c.request_data(o)
    assert not any(s.get("swept") for s in o.data.sources) and [s["number"] for s in o.data.sources] == [1, 2, 3]
    assert {k: v["by"] for k, v in o.data.sweep.items()} == {"news": "model", "press_releases": "model", "filings": "model", "filing_evidence": "model"}


def test_a_search_that_found_nothing_counts_as_searched_so_the_models_nothing_stands(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="zebra unicorn", tickers="META"), say("The stored news does not cover it.")], tmp_path, toolbox, filings_index, press_index)
    o = meta_order()
    c.request_data(o)
    d = o.data
    assert d.no_supporting_text and d.sweep["news"]["by"] == "model" and d.sweep["news"]["sources"] == 0
    assert o.status == "data_ready" and o.errors == []  # the swept press release and filing do not contradict "the news has nothing"
    assert [s["publisher"] for s in d.sources] == ["Press release", "SEC filing"] and all(s["swept"] for s in d.sources)


def test_the_sweep_can_be_turned_off(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), say("Meta [1].")], tmp_path, toolbox, filings_index, press_index)
    c.sweep = False
    o = meta_order()
    c.request_data(o)
    assert [s["number"] for s in o.data.sources] == [1] and o.data.sweep == {} and o.data.filing_evidence == {} and o.data.results["press_releases"]["swept"] == []


def test_ask_alone_never_sweeps(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), say("Meta [1].")], tmp_path, toolbox, filings_index, press_index)
    r = c.ask("meta tax")
    assert [s["number"] for s in r.sources] == [1] and r.sweep == {} and r.filing_evidence == {}


def test_the_sweep_fetches_nothing_and_a_live_order_can_still_fetch_first(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("fetch_press_releases", symbol="META"), call("retrieve_press_releases", query="meta tax", tickers="META"), say("Meta [1].")],
                              tmp_path, toolbox, filings_index, press_index)
    c.allow_fetch = True
    o = press_order(subject=["META"], query="meta tax", live=True)
    c.request_data(o)
    assert [call_[0] for call_ in toolbox.calls] == ["fetch_press_releases"]  # the model's own fetch happened; the sweep that followed added no call to the toolbox
    assert o.data.sweep["press_releases"]["by"] == "model" and o.data.sweep["news"]["by"] == "code" and o.data.sweep["filings"]["by"] == "code"


def test_with_no_stored_filings_the_evidence_is_missing_with_the_reason_and_the_order_is_still_ready(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), say("Meta [1].")], tmp_path, toolbox, filings_index, press_index, files=())
    o = meta_order()
    c.request_data(o)
    assert o.status == "data_ready" and o.data.filing_evidence == {} and o.data.sweep["filing_evidence"]["sources"] == 0
    assert "no filings of META are stored" in o.data.sweep["filing_evidence"]["note"]


def test_a_term_order_is_swept_across_every_company_and_reads_no_evidence(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax"), say("Meta [1].")], tmp_path, toolbox, filings_index, press_index)
    o = order(query="meta tax", subject="meta tax", since_days=7, top_k=3)
    c.request_data(o)
    assert o.status == "data_ready" and [s.get("swept") for s in o.data.sources] == [None, True, True]  # no ticker filter: the term finds the press release and the filing of any company
    assert o.data.filing_evidence == {} and "exactly one" in o.data.sweep["filing_evidence"]["note"]


def test_the_sweep_query_names_the_company_when_the_order_has_one_and_the_query_does_not():
    spec = DataClerk._sweep_spec(order(query="What is new?", company="Meta Platforms", subject=["META"], since_days=30, top_k=4))
    assert spec == {"query": "Meta Platforms: What is new?", "tickers": ["META"], "since_days": 30, "k": 4}
    assert DataClerk._sweep_spec(order(query="What is new with meta platforms?", company="Meta Platforms"))["query"] == "What is new with meta platforms?"
    assert DataClerk._sweep_spec(order(query="flying car", subject="flying car"))["tickers"] is None  # a term filters nothing


def test_a_search_that_fails_in_the_sweep_is_a_note_and_never_a_problem(tmp_path, toolbox, filings_index, press_index):
    class Broken:
        def similarity_search_with_score(self, *args, **kwargs):
            raise RuntimeError("secret path /home/x/.env")
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), say("Meta [1].")], tmp_path, toolbox, filings_index, press_index)
    c.tools.press_store = Broken()
    o = meta_order()
    c.request_data(o)
    assert o.status == "data_ready" and o.data.problems == [] and o.data.sweep["press_releases"] == {"by": "code", "sources": 0, "note": "the search failed (RuntimeError)"}
    assert "secret" not in o.model_dump_json()  # the kind of the failure, never its details


def test_a_swept_order_is_still_plain_json(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_news", query="meta tax", tickers="META"), say("Meta [1].")], tmp_path, toolbox, filings_index, press_index)
    o = meta_order()
    c.request_data(o)
    assert wo.WorkOrder.model_validate_json(o.model_dump_json()) == o
