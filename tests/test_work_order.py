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


# ---------------------------------------------------------------- verify_request


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


# ---------------------------------------------------------------- request_data: the agent call


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


# ---------------------------------------------------------------- verify_action: a press release order must have searched the press releases

def result_with(*tools, outcome="ok") -> Result:
    return Result(question="q", answer="PG [1].", references=[{"number": 1, "chunk_id": "a#0"}], sources=[{"number": 1, "chunk_id": "a#0"}],
                  tool_calls=[{"name": name, "args": {}, "outcome": outcome} for name in tools])


# ---------------------------------------------------------------- request_data for a press release order: the symbol first, then the company name


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


# ================================================================ the action type search_filing (design choices entry 37)
#
# Filings have their own action, as the design PDF names it ("search_news, search_filing, fetch_quote ..."). It is handled by request_data like search_news, and the work order
# checks it against the source in BOTH directions: search_filing is for filings and for nothing else, and a filings order has no other action. Press releases and news keep
# search_news (entry 34).


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


def test_evidence_that_failed_or_was_refused_is_not_on_the_order(tmp_path, toolbox, filings_index, press_index):
    c, _ = three_corpus_clerk([call("retrieve_filings", query="meta tax", tickers="META"), call("fetch_filings_recommendation", symbol="NVDA"),  # no NVDA file: refused with a reason
                               call("fetch_filings_recommendation", symbol="META"), call("fetch_filings_recommendation", symbol="AAPL"),  # the third attempt is past the budget of two
                               call("fetch_filings_recommendation", symbol="META"), say("Done [1].")], tmp_path, toolbox, filings_index, press_index, files=("META", "AAPL"))
    o = filings_order(subject=["META"])
    c.request_data(o)
    assert [t["outcome"] for t in o.data.tool_calls] == ["ok", "invalid", "ok", "limit", "limit"]
    assert list(o.data.filing_evidence) == ["META"]  # the failed NVDA call and the two refused calls left nothing on the order


# ================================================================ the sweep: an order carries all three sources (design choices entry 40)
#
# After the model's loop the Data Clerk's CODE searches every corpus the model did not search and reads the stored filing evidence, so that the Advisor always finds the three
# sources on the order. The model never sees what the sweep finds, so its answer does not cite it; it runs after the model, so it cannot switch off the fetch tools.

def meta_order(**fields):
    """A NEWS order: the model's own search is the news one, so it passes verify_action whatever else the sweep adds."""
    return order(**{"subject": ["META"], "query": "meta tax", "since_days": 30, "top_k": 3, **fields})


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


# ---------------------------------------------------------------- the metrics order of Advisor v1 (design choices entry 51)

def metric_order(symbols=("AAPL",), metrics=({"name": "rsi_14"},), **kw):
    return wo.new_work_order("metrics", requester="advisor", action_type="calculate_financials", subject=list(symbols), metrics=list(metrics), **kw)


@pytest.mark.parametrize("kwargs, message", [
    (dict(symbols=("AAPL",), metrics=({"name": "rsi_14"},)), None),
    (dict(symbols=("AAPL",), metrics=({"name": "nope"}, {"name": "rsi_14"}, {"name": "nope"})), "unknown metrics: nope; known metrics: ret_1d"),
    (dict(symbols=[f"S{i}" for i in range(31)]), "at most 30 symbols"),
    (dict(symbols=[f"S{i}" for i in range(30)]), None),
    (dict(symbols=()), "needs one or more ticker symbols"),
    (dict(metrics=()), "non-empty list of metrics"),
    (dict(metrics=[{"name": "rsi_14"}] * 101), "at most 100 metrics"),
    (dict(metrics=("rsi_14",)), 'each metric is {"name"'),
    (dict(sector="XLZ"), "unknown sector ETF XLZ; known sectors: XLK"),
    (dict(sector="XLK"), None),
    (dict(sector={"AAPL": "XLK"}), None),
    (dict(sector=5), "sector must be one ETF ticker"),
])
def test_verify_metric_request(kwargs, message):
    reasons = wo.verify_metric_request(metric_order(**kwargs))
    assert (message is None and reasons == []) or any(message in r for r in reasons), reasons


def test_a_metrics_request_must_be_a_metrics_order_and_must_not_already_have_its_data():
    assert any("is not 'calculate_financials'" in r for r in wo.verify_metric_request(wo.new_work_order("x", subject=["AAPL"], metrics=[{"name": "rsi_14"}])))
    done = metric_order()
    done.status = "data_ready"
    assert any("already has its data" in r for r in wo.verify_metric_request(done))


