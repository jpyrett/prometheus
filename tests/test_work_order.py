"""Tests for prometheus.common.work_order and DataClerk.request_data (Part 8 of the RAG guide).

Run:  pytest tests/test_work_order.py -q

No network, no key, no cost: a scripted model stands in for the chat model (see test_data_clerk_agent.py) and the stores are in
temporary folders.
"""
import json

import pytest

from prometheus.agents.data_clerk.agent import DataClerk, Result
from prometheus.common import work_order as wo
from test_data_clerk_agent import ScriptedModel, call, say  # the same scripted model and turns as the agent tests
from test_data_clerk_tools import FakeToolbox
from test_rag_retriever import DAY, NOW, KeywordEmbeddings, chunk, fill, make_store


@pytest.fixture
def index(tmp_path):
    return fill(make_store(tmp_path / "chroma", KeywordEmbeddings()), [
        chunk("a1", 0, 1, "apple iphone phone", ["AAPL"], NOW - 5 * DAY),
        chunk("a2", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 4 * DAY),
        chunk("n1", 0, 1, "nvidia chip chip chip", ["NVDA"], NOW - 3 * DAY)])


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    news = tmp_path / "news"
    news.mkdir()
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(news))
    return FakeToolbox(news)


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


@pytest.mark.parametrize("source", ["filings", "press_releases", "News", "", "social"])
def test_a_source_that_is_not_wired_is_refused_and_news_is_the_default(source):
    assert wo.verify_request(order()) == [] and order().source == "news"
    reasons = wo.verify_request(order(source=source))
    assert any(f"source {source!r} is not wired yet (only: news)" in r for r in reasons), reasons


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
