"""The Data Clerk's verification call (Critic v1 spec section 5.2): `DataClerk.request_verification` and the three read methods of `DataClerkTools` behind it.

No network, no key: stores live in temporary folders (the fake one-word-per-dimension embedder of test_rag_retriever), the symbol directory is a fake, and the Data Clerk's model is an object that
raises when it is invoked (this call must never reach it).
"""
from datetime import datetime, timedelta

import pytest
from test_filing_signals import APPLE_TODAY, store as filing_store
from test_rag_retriever import DAY, NOW, KeywordEmbeddings, chunk, fill, make_store
from test_symbols import AnyQuery

from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import work_order as wo
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolInputError
from prometheus.tools.symbols import SymbolLookupError


class NoModel:
    """The Data Clerk's model: this call is code, so asking it anything is a bug."""

    def invoke(self, *args, **kwargs):
        raise AssertionError("request_verification must not call a model")

    bind_tools = invoke


@pytest.fixture
def stores(tmp_path):
    news = fill(make_store(tmp_path / "news", KeywordEmbeddings()), [
        chunk("a1", 0, 1, "apple iphone phone", ["AAPL"], NOW - 5 * DAY, publisher="Yahoo"),
        chunk("a2", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 1 * DAY, publisher="Reuters"),
        chunk("n1", 0, 1, "nvidia chip chip chip", ["NVDA"], NOW - 3 * DAY)])
    press = fill(make_store(tmp_path / "press", KeywordEmbeddings()), [
        chunk("pr:p1", 0, 1, "apple tesla announces", ["AAPL"], NOW - 2 * DAY, publisher="Press release")])
    filings = fill(make_store(tmp_path / "filings", KeywordEmbeddings()), [
        chunk("sec:0000320193-26-000006:risk_factors", 0, 1, "apple tax risk", ["AAPL"], NOW - 9 * DAY, publisher="SEC filing")])
    return news, press, filings


def clerk(stores, search=None):
    news, press, filings = stores
    c = DataClerk(toolbox=object(), model=NoModel(), store=news, press_store=press, filings_store=filings, symbol_search=search or AnyQuery([]))
    c.lookup_interval = 0
    return c


def order(**spec):
    return wo.new_work_order("verify", requester="critic", action_type="verify_data", subject=["AAPL"], verify=spec)


# ---------------------------------------------------------------- read_sources

def test_a_chunk_is_read_again_from_the_collection_its_id_names(stores):
    got = clerk(stores).tools.read_sources(["a1#0", "pr:p1#0", "sec:0000320193-26-000006:risk_factors#0"])
    assert [got[k]["corpus"] for k in got] == ["news", "press_releases", "filings"] and all(v["found"] for v in got.values())
    a1 = got["a1#0"]
    assert (a1["article_id"], a1["headline"], a1["publisher"], a1["tickers"], a1["text"]) == ("a1", "Headline of a1", "Yahoo", ["AAPL"], "apple iphone phone")
    assert a1["published_utc"] == (NOW - 5 * DAY).strftime("%Y-%m-%dT%H:%M:%SZ") and a1["url"] == "https://example.com/a1"
    assert got["pr:p1#0"]["publisher"] == "Press release"


def test_an_unknown_chunk_an_evidence_id_and_a_wrong_collection_are_each_not_found_with_a_reason(stores):
    got = clerk(stores).tools.read_sources(["nope#0", "evidence:0000320193-26-000006:E3", "pr:a1#0"])
    assert got["nope#0"] == {"found": False, "note": "not in the stored news"}
    assert got["pr:a1#0"] == {"found": False, "note": "not in the stored press releases"}
    assert got["evidence:0000320193-26-000006:E3"]["found"] is False and "evidence id" in got["evidence:0000320193-26-000006:E3"]["note"]


def test_the_answer_keeps_the_order_asked_and_drops_repeats(stores):
    assert list(clerk(stores).tools.read_sources(["n1#0", "a1#0", "n1#0"])) == ["n1#0", "a1#0"]


def test_a_store_that_cannot_be_read_is_reported_per_chunk_never_raised(stores):
    class Broken:
        def get(self, **kw):
            raise OSError("disk /secret/path is gone")

    c = clerk(stores)
    c.tools.store = Broken()
    got = c.tools.read_sources(["a1#0", "pr:p1#0"])
    assert got["a1#0"] == {"found": False, "note": "the store could not be read (OSError)"} and "/secret" not in str(got) and got["pr:p1#0"]["found"] is True


# ---------------------------------------------------------------- read_evidence

@pytest.fixture
def apple_filings(tmp_path, monkeypatch, capsys):
    """The real Apple filings, stored where `fetch_filings_recommendation` reads them."""
    filing_store(tmp_path, monkeypatch, "AAPL", "filings_sample.txt", "edgar_aapl_10q_0000320193-26-000006")
    capsys.readouterr()


def test_evidence_items_are_read_again_by_id_from_the_stored_filings(stores, apple_filings):
    tools = clerk(stores).tools
    tools.today = APPLE_TODAY
    items = {i["id"]: i for i in tools.fetch_filings_recommendation("AAPL")["evidence"]}
    first = next(iter(items))
    got = tools.read_evidence("AAPL", [first, "E999"])
    assert got[first]["found"] is True and got[first]["quote"] == items[first]["quote"] and got[first]["what"] == items[first]["what"]
    assert set(got[first]["source"]) == {"accession", "form", "filed"} and got[first]["source"]["accession"] == items[first]["source"]["accession"]
    assert got["E999"] == {"found": False, "note": "AAPL has no evidence item E999 in its stored filings"}


def test_no_stored_filings_means_every_evidence_id_is_not_found_with_the_tools_message(stores, tmp_path, monkeypatch):
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(tmp_path / "empty"))
    got = clerk(stores).tools.read_evidence("AAPL", ["E1", "E2"])
    assert set(got) == {"E1", "E2"} and all(v["found"] is False and "no filings of AAPL are stored" in v["note"] for v in got.values())


# ---------------------------------------------------------------- recent_sources

def test_the_newest_stored_news_and_press_releases_come_back_newest_first_with_their_corpus(stores):
    got = clerk(stores).tools.recent_sources("AAPL", "Apple", since_days=14, k=8)
    assert [h["article_id"] for h in got] == ["a2", "pr:p1", "a1"] and [h["corpus"] for h in got] == ["news", "press_releases", "news"]
    assert got[0]["text"] == "apple iphone phone phone" and {"chunk_id", "headline", "publisher", "published_utc", "tickers"} <= set(got[0])


def test_recent_sources_respects_the_window_the_symbol_and_k(stores):
    tools = clerk(stores).tools
    assert [h["article_id"] for h in tools.recent_sources("AAPL", None, since_days=3, k=8)] == ["a2", "pr:p1"]      # a1 is 5 days old
    assert [h["article_id"] for h in tools.recent_sources("AAPL", None, since_days=14, k=1)] == ["a2"]
    assert tools.recent_sources("TSLA", None, since_days=14, k=8) == []


@pytest.mark.parametrize("kwargs, message", [({"since_days": 0}, "since_days"), ({"since_days": 31}, "since_days"), ({"k": 0}, "k must"), ({"k": 21}, "k must"), ({"k": True}, "k must")])
def test_recent_sources_refuses_bad_arguments(stores, kwargs, message):
    with pytest.raises(ToolInputError, match=message):
        clerk(stores).tools.recent_sources("AAPL", None, **{"since_days": 14, "k": 8, **kwargs})


def test_the_critics_methods_are_not_tools_of_the_model():
    assert not {"read_sources", "read_evidence", "recent_sources"} & set(DATA_CLERK_TOOLS) and len(DATA_CLERK_TOOLS) == 13


# ---------------------------------------------------------------- request_verification

def test_a_full_request_is_answered_without_a_model_and_the_order_is_data_ready(stores, apple_filings):
    search = AnyQuery([{"symbol": "AAPL", "description": "APPLE INC", "type": "Common Stock"}])
    c = clerk(stores, search)
    c.tools.today = APPLE_TODAY
    first = c.tools.fetch_filings_recommendation("AAPL")["evidence"][0]["id"]
    result = c.request_verification(order(chunks=["a1#0", "zzz#0"], evidence={"AAPL": [first]}, symbols=["AAPL"], company={"AAPL": "Apple"}, fresh={"since_days": 14, "k": 8}))
    v = result.data.verification
    assert result.status == "data_ready" and result.errors == []
    assert v["chunks"]["a1#0"]["found"] is True and v["chunks"]["zzz#0"]["found"] is False
    assert v["evidence"]["AAPL"][first]["found"] is True
    assert v["symbols"] == {"AAPL": {"listed": True, "company": "Apple Inc", "note": None}} or v["symbols"]["AAPL"]["listed"] is True
    assert [h["article_id"] for h in v["fresh"]["AAPL"]] == ["a2", "pr:p1", "a1"]
    names = [(c["name"], c["outcome"]) for c in result.data.tool_calls]
    assert names == [("read_sources", "ok"), ("read_evidence", "ok"), ("find_symbol", "ok"), ("recent_sources", "ok")]
    assert result.data.tool_calls[0]["args"] == {"chunks": 2}


def test_an_unlisted_symbol_is_listed_false_and_a_failed_lookup_is_null_not_false(stores):
    class Down:
        def __call__(self, query):
            raise SymbolLookupError("Finnhub answered HTTP 429")

    listed = clerk(stores, AnyQuery([])).request_verification(order(symbols=["ZZZZ"]))
    assert listed.data.verification["symbols"]["ZZZZ"] == {"listed": False, "company": None, "note": "Finnhub's directory has no such symbol"} and listed.status == "data_ready"
    failed = clerk(stores, Down()).request_verification(order(symbols=["AAPL"]))
    got = failed.data.verification["symbols"]["AAPL"]
    assert got["listed"] is None and "the lookup failed" in got["note"] and "429" in got["note"]
    assert failed.status == "data_failed" and any("find_symbol AAPL" in e for e in failed.errors)      # the only part failed


def test_one_failed_part_does_not_stop_the_others(stores, monkeypatch):
    c = clerk(stores)
    monkeypatch.setattr(c.tools, "recent_sources", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index is rebuilding")))
    result = c.request_verification(order(chunks=["a1#0"], symbols=["AAPL"], fresh={"since_days": 7}))
    assert result.status == "data_ready" and result.data.verification["chunks"]["a1#0"]["found"] is True
    assert any("recent_sources AAPL: RuntimeError: index is rebuilding" in e for e in result.errors)
    assert [c["outcome"] for c in result.data.tool_calls if c["name"] == "recent_sources"] == ["error"]


def test_every_part_failing_makes_the_order_data_failed(stores, monkeypatch):
    c = clerk(stores)
    monkeypatch.setattr(c.tools, "read_sources", lambda ids: (_ for _ in ()).throw(RuntimeError("boom")))
    result = c.request_verification(order(chunks=["a1#0"]))
    assert result.status == "data_failed" and result.errors and result.data.verification["chunks"] == {}


def test_a_refused_request_comes_back_data_failed_with_the_reasons_and_no_data(stores):
    result = clerk(stores).request_verification(order(chunks=["has space"]))
    assert result.status == "data_failed" and result.data is None and any("chunks must be a list" in e for e in result.errors)


def test_an_order_that_already_has_its_data_is_returned_unchanged(stores):
    done = order(chunks=["a1#0"])
    done.status = "reviewed"
    assert clerk(stores).request_verification(done) is done and done.data is None and done.errors == []


def test_the_symbol_lookup_is_paced_and_nothing_is_fetched(stores):
    c = clerk(stores, AnyQuery([]))
    c.lookup_interval = 0.05
    started = datetime.now()
    c.request_verification(order(symbols=["AAPL", "MSFT", "TSLA"]))
    assert datetime.now() - started >= timedelta(seconds=0.1)       # three lookups, no faster than one every 50 ms
    assert c.allow_fetch is False


def test_a_chunk_asked_twice_is_read_once_and_the_clock_is_passed_to_the_search(stores):
    later = NOW + 30 * DAY             # a month on, nothing in these stores is inside the last 7 days
    result = clerk(stores).request_verification(order(chunks=["a1#0", "a1#0"], symbols=["AAPL"], fresh={"since_days": 7, "k": 3}), now=later)
    assert result.data.verification["chunks"].keys() == {"a1#0"} and result.data.verification["fresh"]["AAPL"] == []
    assert len(clerk(stores).request_verification(order(symbols=["AAPL"], fresh={"since_days": 7, "k": 3}), now=NOW).data.verification["fresh"]["AAPL"]) == 3
