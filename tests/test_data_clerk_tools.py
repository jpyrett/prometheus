"""Tests for prometheus.tools.data_clerk (Part 6 of the RAG guide).

Run:  pytest tests/test_data_clerk_tools.py -q

A FAKE toolbox stands in for the DataToolbox, so no test uses the network, a key or a paid API, and no test writes
to the real news files or the real index: every store and news folder lives in a temporary folder. The error
handling is tested through a small LangGraph graph, because ToolNode cannot be called by hand in LangGraph 1.x.
"""
import shutil
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.agents.data_clerk.edgar_text import EdgarNotConfigured
from prometheus.rag import citations
from prometheus.rag.citations import SourceBook
from prometheus.rag.config import REPO_ROOT
from prometheus.tools import data_clerk
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolInputError, ToolLimitError, describe_tool_error
from test_rag_retriever import DAY, NOW, KeywordEmbeddings, chunk, fill, make_store  # the same fake embedder and helpers

FIXTURE = Path(__file__).parent / "fixtures" / "news_sample.txt"
PRESS_FIXTURE = Path(__file__).parent / "fixtures" / "press_releases_sample.txt"  # the first 3 REAL records of data/press_releases/AAPL.txt, copied byte for byte
FILINGS_FIXTURE = Path(__file__).parent / "fixtures" / "filings_sample.txt"  # 6 REAL records of data/filings/AAPL.txt (7 Documents: the 10-K is two), copied byte for byte


class FakeFinancials:
    def __init__(self, parent):
        self.parent = parent

    def get(self, symbol):
        self.parent.calls.append(("financials.get", (symbol,), {}))
        return {"symbol": symbol, "as_of": "2026-10-25", "metrics": {"pe_ratio": {"value": 31.2, "unit": "x"}}}


class FakeToolbox:
    """Records every call, returns canned results, and can be told to fail."""

    def __init__(self, news_dir: Path | None = None, press_dir: Path | None = None, filings_dir: Path | None = None):
        self.calls: list[tuple] = []
        self.fail: Exception | None = None
        self.news_dir = news_dir
        self.press_dir = press_dir
        self.filings_dir = filings_dir
        self.edgar_missing = False            # fetch_filing_text raises EdgarNotConfigured, as the real one does without SEC_USER_AGENT
        self.edgar_fail: Exception | None = None
        self.financials = FakeFinancials(self)

    def _call(self, name, args, kwargs):
        self.calls.append((name, args, kwargs))
        if self.fail:
            raise self.fail

    def fetch_news(self, company=None, symbol=None, keywords=None, from_date=30):
        self._call("fetch_news", (), dict(company=company, symbol=symbol, keywords=keywords, from_date=from_date))
        if self.news_dir:  # what the real one does: append the articles to news/<SYMBOL>.txt
            shutil.copy(FIXTURE, self.news_dir / f"{symbol}.txt")
        return {"symbol": symbol, "path": "/home/someone/secret/news/AAPL.txt", "fetched": 9, "matched": 4, "written": 3, "duplicates": 1}

    def fetch_press_releases(self, symbol, from_date=365):
        self._call("fetch_press_releases", (symbol,), dict(from_date=from_date))
        if self.press_dir:  # what the real one does: append the releases to <SYMBOL>_PRs.txt
            shutil.copy(PRESS_FIXTURE, self.press_dir / f"{symbol}_PRs.txt")
        return {"symbol": symbol, "path": "/home/someone/secret/pr.txt", "fetched": 5, "written": 2, "duplicates": 3}

    def fetch_filings(self, symbol, from_date=365, forms=()):
        self._call("fetch_filings", (symbol,), dict(from_date=from_date))
        if self.filings_dir:  # what the real one does: append the filings to <SYMBOL>.txt in the filings folder
            shutil.copy(FILINGS_FIXTURE, self.filings_dir / f"{symbol}.txt")
        return {"symbol": symbol, "path": "/home/someone/secret/f.txt", "fetched": 7, "written": 1, "duplicates": 6}

    def fetch_filing_text(self, symbol, forms=("10-Q",), from_date=365):
        self._call("fetch_filing_text", (symbol,), dict(from_date=from_date))
        if self.edgar_missing:  # the real message holds the path of the .env file: it must never reach the model
            raise EdgarNotConfigured("SEC_USER_AGENT is not set in /home/someone/secret/.env: EDGAR needs a User-Agent that names you and gives a contact")
        if self.edgar_fail:
            raise self.edgar_fail
        return {"symbol": symbol, "path": "/home/someone/secret/f.txt", "candidates": 3, "written": 1}

    def search(self, symbol, terms, from_date=30, refresh=True):
        self._call("search", (symbol, terms), dict(from_date=from_date, refresh=refresh))
        news = [{"published": "2026-10-2%d" % (i % 10), "publisher": "Yahoo", "headline": f"News {i}", "url": f"https://x/{i}"} for i in range(30)]
        filings = [{"filed": "2026-10-01", "form": "8-K", "url": f"https://sec/{i}", "text_status": "ok"} for i in range(15)]
        return {"symbol": symbol, "terms": terms, "from_date": from_date, "news": news, "filings": filings, "filings_in_window": 15}

    def discover(self, phrase, from_date=7, top=10, refresh=True):
        self._call("discover", (phrase,), dict(from_date=from_date, top=top, refresh=refresh))
        return {"phrase": phrase, "from_date": from_date, "scanned": 100, "matched": 4, "tickers": [{"symbol": "MU", "articles": 3}]}

    def get_quote(self, symbol):
        self._call("get_quote", (symbol,), {})
        return {"symbol": symbol, "price": 410.25, "change": 2.5, "change_percent": 0.61, "high": 412.0, "low": 405.0, "open": 406.0, "previous_close": 407.75,
                "as_of_utc": "2026-11-03T21:00:00Z", "source": "Finnhub /quote"}

    def performance_summary(self, symbol, points=61):
        self._call("performance_summary", (symbol,), {})
        return {"symbol": symbol, "volume": 1000, "average_price": 339.8,
                "distribution": {"type": "normal", "mean": 339.8, "std_dev": 7.1, "curve": [{"price": 1.0, "density": 0.1}] * 61}}


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


@pytest.fixture
def press_index(tmp_path):
    """Press releases already stored, searched with the same fake word embedder: two of Apple, one of Nvidia, the Nvidia one the newest."""
    return fill(make_store(tmp_path / "chroma_pr", KeywordEmbeddings()), [
        chunk("pr:a1", 0, 1, "apple iphone phone", ["AAPL"], NOW - 20 * DAY, publisher="Press release"),
        chunk("pr:a2", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 10 * DAY, publisher="Press release"),
        chunk("pr:n1", 0, 1, "nvidia chip chip chip", ["NVDA"], NOW - 2 * DAY, publisher="Press release")])


@pytest.fixture
def filings_index(tmp_path):
    """Filings already stored, searched with the same fake word embedder: two sections of an Apple filing, one of Nvidia (the newest), the way a 10-K is two articles."""
    return fill(make_store(tmp_path / "chroma_f", KeywordEmbeddings()), [
        chunk("sec:0000000001-25-000001:business", 0, 1, "apple iphone phone", ["AAPL"], NOW - 20 * DAY, publisher="SEC filing"),
        chunk("sec:0000000001-25-000001:risk_factors", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 20 * DAY, publisher="SEC filing"),
        chunk("sec:0000000002-25-000002:items", 0, 1, "nvidia chip chip chip", ["NVDA"], NOW - 2 * DAY, publisher="SEC filing")])


@pytest.fixture
def index(tmp_path):
    """Eight stored chunks of seven articles, searched with the fake word embedder."""
    return fill(make_store(tmp_path / "chroma", KeywordEmbeddings()), [
        chunk("a1", 0, 1, "apple iphone phone", ["AAPL"], NOW - 5 * DAY),
        chunk("a2", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 4 * DAY),
        chunk("a3", 0, 1, "apple nvidia chip", ["AAPL", "NVDA"], NOW - 3 * DAY),
        chunk("n1", 0, 1, "nvidia chip chip chip", ["NVDA"], NOW - 3 * DAY),
        chunk("t1", 0, 1, "tesla car", ["TSLA"], NOW - 1 * DAY),
    ])


@pytest.fixture
def tools(toolbox, index, press_index, filings_index) -> DataClerkTools:
    return DataClerkTools(toolbox, store=index, press_store=press_index, filings_store=filings_index)


# ---------------------------------------------------------------- the allow-list and what the model sees

def test_the_allow_list_is_exactly_what_the_model_is_shown(tools):
    assert DATA_CLERK_TOOLS == ("retrieve_news", "retrieve_press_releases", "retrieve_filings", "fetch_news", "fetch_press_releases", "fetch_filings", "fetch_filings_recommendation",
                                "keyword_search", "discover_tickers", "get_financials", "get_price_summary", "get_quote", "calculate_metrics")
    assert [t.name for t in tools.langchain_tools()] == list(DATA_CLERK_TOOLS)


def test_nothing_dangerous_is_on_the_allow_list_or_reachable_through_the_tools(tools):
    never = {"init_aws_cli", "fetch_historical_data", "convert_to_parquet", "export_project", "sync_market_news", "sync_cnbc_news"}
    assert never.isdisjoint(DATA_CLERK_TOOLS)
    assert never.isdisjoint({t.name for t in tools.langchain_tools()})
    assert not [name for name in never if hasattr(tools, name)]  # the class does not even have such a method


def test_every_tool_has_a_description_and_every_argument_is_documented(tools):
    for t in tools.langchain_tools():
        assert len(t.description) > 40, t.name
        assert "self" not in t.args, t.name
        for arg, spec in t.args.items():
            assert spec.get("description"), f"{t.name}.{arg} has no description for the model"


def test_retrieve_news_passes_newest_first_to_the_search_and_the_default_is_off(index, toolbox):
    tools = DataClerkTools(toolbox, index)
    plain = [h["chunk_id"] for h in tools.retrieve_news("apple iphone", tickers="AAPL", k=5)["hits"]]
    newest = [h["published_utc"] for h in tools.retrieve_news("apple iphone", tickers="AAPL", k=5, newest_first=True)["hits"]]
    assert newest == sorted(newest, reverse=True) and plain  # in date order only when asked
    with pytest.raises(ToolInputError, match="newest_first must be true or false"):
        tools.retrieve_news("apple iphone", newest_first="yes")


def test_the_model_sees_the_arguments_of_retrieve_news_and_not_the_relevance_bar(tools):
    retrieve = tools.langchain_tools()[0]
    assert list(retrieve.args) == ["query", "tickers", "since_days", "k", "diverse", "newest_first"]  # min_score stays ours
    assert retrieve.get_input_schema().model_json_schema()["required"] == ["query"]
    assert "diverse" in retrieve.description and "newest_first" in retrieve.description and "no supporting text" in retrieve.description


def test_the_toolbox_of_the_real_default_is_rooted_where_rag_reads_its_files():
    # no network: building a DataToolbox only sets paths
    assert Path(DataClerkTools().toolbox.base_dir) == REPO_ROOT


# ---------------------------------------------------------------- argument checks

@pytest.mark.parametrize("value,expected", [("aapl", "AAPL"), ("$nvda", "NVDA"), (" brk.b ", "BRK.B"), ("$$tsla", "TSLA")])
def test_symbols_are_cleaned(value, expected):
    assert data_clerk._symbol(value) == expected


@pytest.mark.parametrize("bad", ["../../.env", "AAPL/../x", "", "A B", "TOOLONGSYMBOL1", 5, None, ["AAPL"]])
def test_a_bad_symbol_is_a_tool_input_error(bad):
    with pytest.raises(ToolInputError):
        data_clerk._symbol(bad)


def test_tickers_may_be_one_symbol_or_a_list_without_repeats():
    assert data_clerk._symbols(None) == [] and data_clerk._symbols("aapl") == ["AAPL"]
    assert data_clerk._symbols(["aapl", "$AAPL", "nvda"]) == ["AAPL", "NVDA"]
    for bad in (5, {"a": 1}, ["AAPL", 5], ["AAPL", "../x"], ["A"] * 11):
        with pytest.raises(ToolInputError):
            data_clerk._symbols(bad)


@pytest.mark.parametrize("bad", [0, -1, 91, 10_000, True, "7", 2.5, None])
def test_days_outside_the_allowed_range_never_reach_the_toolbox(tools, toolbox, bad):
    with pytest.raises(ToolInputError, match="from_date"):
        tools.fetch_news("AAPL", from_date=bad)
    assert toolbox.calls == []


def test_the_longer_limits_for_filings_and_the_shorter_one_for_discover(tools, toolbox):
    tools.fetch_filings("AAPL", from_date=730)
    with pytest.raises(ToolInputError):
        tools.fetch_filings("AAPL", from_date=731)
    tools.discover_tickers("GLP-1", from_date=30, top=20)
    for kwargs in (dict(from_date=31), dict(top=21), dict(top=0)):
        with pytest.raises(ToolInputError):
            tools.discover_tickers("GLP-1", **kwargs)
    assert [c[0] for c in toolbox.calls] == ["fetch_filings", "fetch_filing_text", "discover"]  # the EDGAR text step runs inside fetch_filings


def test_keywords_company_terms_and_phrase_are_limited(tools, toolbox):
    for call in (lambda: tools.fetch_news("AAPL", keywords=["x"] * 11), lambda: tools.fetch_news("AAPL", keywords=["y" * 61]),
                 lambda: tools.fetch_news("AAPL", keywords="iphone"), lambda: tools.fetch_news("AAPL", company="c" * 61),
                 lambda: tools.keyword_search("AAPL", []), lambda: tools.keyword_search("AAPL", ["  "]),
                 lambda: tools.keyword_search("AAPL", "iphone"), lambda: tools.discover_tickers("   ")):
        with pytest.raises(ToolInputError):
            call()
    assert toolbox.calls == []


# ---------------------------------------------------------------- retrieve_news (rag/)

def test_retrieve_news_returns_hits_as_plain_data(tools):
    result = tools.retrieve_news("apple iphone phone", tickers="aapl", k=2)
    assert result["found"] and result["reason"] is None and result["message"] is None
    assert [h["article_id"] for h in result["hits"]] == ["a1", "a2"]
    assert {"chunk_id", "headline", "publisher", "published_utc", "url", "tickers", "text", "score"} <= set(result["hits"][0])


def test_retrieve_news_filters_by_ticker_and_window(tools):
    assert {h["article_id"] for h in tools.retrieve_news("nvidia chip", tickers=["NVDA"], k=5)["hits"]} == {"a3", "n1"}
    assert tools.retrieve_news("tesla car", tickers="TSLA")["hits"][0]["article_id"] == "t1"


def test_nothing_relevant_says_no_supporting_text_with_the_closest_score(tools):
    result = tools.retrieve_news("cake")
    assert not result["found"] and result["hits"] == [] and result["reason"] == "nothing_relevant"
    assert result["message"].startswith("No supporting text: no stored article scored 0.40 or more")
    assert "no filters" in result["message"] and "closest scored" in result["message"] and "Do not answer from memory" in result["message"]


def test_a_near_miss_tells_the_model_to_reword_once_and_a_clear_miss_does_not(tools):
    near = tools.retrieve_news("apple tesla car", tickers="AAPL")  # the best AAPL chunk scores 0.33, within 0.10 of the 0.40 bar
    assert near["reason"] == "nothing_relevant" and "closest scored 0.33" in near["message"]
    assert "close to the bar" in near["message"] and "search once more with a clearer question" in near["message"]
    far = tools.retrieve_news("cake")
    assert "close to the bar" not in far["message"] and "search once more" not in far["message"]


def test_a_ticker_with_no_stored_article_tells_the_model_to_fetch(tools):
    result = tools.retrieve_news("anything about meta", tickers="META")
    assert result["reason"] == "no_article_matches_filters" and not result["found"]
    assert "tickers META" in result["message"] and "call fetch_news" in result["message"]


def test_an_empty_index_says_so_without_asking_for_a_fetch(tmp_path):
    empty = DataClerkTools(FakeToolbox(), store=make_store(tmp_path / "empty", KeywordEmbeddings()))
    result = empty.retrieve_news("apple")
    assert result["reason"] == "no_article_matches_filters" and "may be empty" in result["message"] and "fetch_news" not in result["message"]


@pytest.mark.parametrize("kwargs", [dict(query=""), dict(query="apple", k=0), dict(query="apple", k=21), dict(query="apple", since_days=0),
                                    dict(query="apple", tickers="../../.env"), dict(query="apple", tickers=[1])])
def test_bad_retrieve_arguments_are_tool_input_errors(tools, kwargs):
    with pytest.raises(ToolInputError):
        tools.retrieve_news(**kwargs)


# ---------------------------------------------------------------- the fetch tools (the toolbox, then the index)

def test_fetch_news_calls_the_toolbox_and_then_makes_the_new_articles_searchable(tools, toolbox, tmp_path):
    result = tools.fetch_news("$aapl", company="Apple", keywords=["iPhone"], from_date=14)
    assert toolbox.calls == [("fetch_news", (), dict(company="Apple", symbol="AAPL", keywords=["iPhone"], from_date=14))]
    assert result == {"symbol": "AAPL", "fetched": 9, "matched": 4, "written": 3, "duplicates": 1,
                      "indexed": {"articles": 3, "chunks": 4, "added": 4, "updated": 0, "unchanged": 0, "deleted": 0}}
    assert "secret" not in str(result) and "path" not in result  # the local file path stays out of the prompt
    again = tools.fetch_news("AAPL")  # same file again: everything is unchanged
    assert again["indexed"]["added"] == 0 and again["indexed"]["unchanged"] == 4


def test_fetch_news_defaults_are_a_week_and_no_filters(tools, toolbox):
    tools.fetch_news("AAPL")
    assert toolbox.calls[0][2] == dict(company=None, symbol="AAPL", keywords=[], from_date=7)


def test_when_the_fetch_fails_nothing_is_indexed_and_the_error_is_not_swallowed(tools, toolbox, index):
    toolbox.fail = RuntimeError("Finnhub HTTP 429")
    before = index.get()["ids"]
    with pytest.raises(RuntimeError, match="429"):
        tools.fetch_news("AAPL")
    assert index.get()["ids"] == before


def test_fetch_press_releases_calls_the_toolbox_and_then_makes_the_new_releases_searchable(tools, toolbox, press_index, index):
    result = tools.fetch_press_releases("aapl")
    assert toolbox.calls == [("fetch_press_releases", ("AAPL",), dict(from_date=365))]
    assert result == {"symbol": "AAPL", "fetched": 5, "written": 2, "duplicates": 3,
                      "indexed": {"articles": 3, "chunks": 3, "added": 3, "updated": 0, "unchanged": 0, "deleted": 0}}
    assert "secret" not in str(result) and "path" not in result  # the local file path stays out of the prompt
    stored = set(press_index.get()["ids"])
    assert {"pr:44107d69ebe0e861#0", "pr:5a31fe63040096fc#0", "pr:519ab77bd1631dd7#0"} <= stored  # the three REAL releases of the fixture, in the PRESS collection
    assert not any(i.startswith("pr:4") or i.startswith("pr:5") for i in index.get()["ids"])      # and not in the news collection
    again = tools.fetch_press_releases("AAPL")  # the same file again: everything is unchanged
    assert again["indexed"]["added"] == 0 and again["indexed"]["unchanged"] == 3


def test_fetch_press_releases_defaults_to_a_year_and_checks_its_arguments(tools, toolbox):
    tools.fetch_press_releases("AAPL")
    assert toolbox.calls[0][2] == dict(from_date=365)
    for bad in (0, 731, True, "7", None):
        with pytest.raises(ToolInputError):
            tools.fetch_press_releases("AAPL", from_date=bad)
    with pytest.raises(ToolInputError):
        tools.fetch_press_releases("../../.env")
    assert len(toolbox.calls) == 1  # nothing was fetched for the bad ones


def test_when_the_press_release_fetch_fails_nothing_is_indexed_and_the_error_is_not_swallowed(tools, toolbox, press_index):
    toolbox.fail = RuntimeError("Finnhub press releases HTTP 429")
    before = press_index.get()["ids"]
    with pytest.raises(RuntimeError, match="429"):
        tools.fetch_press_releases("AAPL")
    assert press_index.get()["ids"] == before


def test_fetch_filings_calls_the_toolbox_and_then_makes_the_new_filings_searchable(tools, toolbox, filings_index, press_index, index):
    result = tools.fetch_filings("$aapl", from_date=100)
    assert toolbox.calls == [("fetch_filings", ("AAPL",), dict(from_date=100)), ("fetch_filing_text", ("AAPL",), dict(from_date=100))]  # the Massive fetch, then the EDGAR text, then one index
    indexed = result.pop("indexed")
    assert result == {"symbol": "AAPL", "fetched": 7, "written": 1, "duplicates": 6, "edgar_text": {"candidates": 3, "written": 1}}  # counts and no path
    assert "secret" not in str(result) and "path" not in result  # the local file path stays out of the prompt
    assert indexed["articles"] == 7 and indexed["chunks"] > 150 and indexed["added"] == indexed["chunks"] and indexed["unchanged"] == 0
    stored = set(filings_index.get()["ids"])
    assert {"sec:0000320193-25-000077:items#0", "sec:0000320193-26-000006#0"} <= stored  # two of the REAL filings of the fixture, in the FILINGS collection
    assert not any(i.startswith("sec:") for i in index.get()["ids"] + press_index.get()["ids"])  # and in neither of the other two
    again = tools.fetch_filings("AAPL")  # the same file again: everything is unchanged
    assert again["indexed"]["added"] == 0 and again["indexed"]["unchanged"] == indexed["chunks"]


def test_fetch_filings_defaults_to_a_year_and_checks_its_arguments(tools, toolbox):
    tools.fetch_filings("AAPL")
    assert toolbox.calls[0][2] == toolbox.calls[1][2] == dict(from_date=365)  # both steps, a year
    for bad in (0, 731, True, "7", None):
        with pytest.raises(ToolInputError):
            tools.fetch_filings("AAPL", from_date=bad)
    with pytest.raises(ToolInputError):
        tools.fetch_filings("../../.env")
    assert [c[0] for c in toolbox.calls] == ["fetch_filings", "fetch_filing_text"]  # nothing was fetched for the bad ones


def test_when_the_filings_fetch_fails_nothing_is_indexed_and_the_error_is_not_swallowed(tools, toolbox, filings_index):
    toolbox.fail = RuntimeError("Massive filings HTTP 429")
    before = filings_index.get()["ids"]
    with pytest.raises(RuntimeError, match="429"):
        tools.fetch_filings("AAPL")
    assert filings_index.get()["ids"] == before


# ---------------------------------------------------------------- retrieve_press_releases (rag/, the press release collection)

def test_retrieve_press_releases_returns_hits_as_plain_data(tools):
    result = tools.retrieve_press_releases("apple iphone phone", tickers="aapl", k=2)
    assert result["found"] and result["reason"] is None and result["message"] is None
    assert [h["article_id"] for h in result["hits"]] == ["pr:a1", "pr:a2"]
    assert {"chunk_id", "headline", "publisher", "published_utc", "url", "tickers", "text", "score"} <= set(result["hits"][0])
    assert result["hits"][0]["publisher"] == "Press release"


def test_the_two_searches_each_see_only_their_own_collection(tools):
    assert {h["article_id"] for h in tools.retrieve_news("nvidia chip", tickers="NVDA", k=5)["hits"]} == {"a3", "n1"}
    assert {h["article_id"] for h in tools.retrieve_press_releases("nvidia chip", tickers="NVDA", k=5)["hits"]} == {"pr:n1"}


def test_retrieve_press_releases_filters_by_ticker_and_window_and_can_put_the_newest_first(tools):
    assert tools.retrieve_press_releases("apple iphone phone", tickers="NVDA")["found"] is False  # a ticker with no apple release
    recent = tools.retrieve_press_releases("apple iphone phone", tickers="AAPL", since_days=15)
    assert [h["article_id"] for h in recent["hits"]] == ["pr:a2"]  # pr:a1 is 20 days old
    newest = [h["published_utc"] for h in tools.retrieve_press_releases("apple iphone phone", tickers="AAPL", newest_first=True)["hits"]]
    assert newest == sorted(newest, reverse=True) and len(newest) == 2
    with pytest.raises(ToolInputError, match="newest_first must be true or false"):
        tools.retrieve_press_releases("apple iphone", newest_first="yes")


def test_a_ticker_with_no_stored_press_release_tells_the_model_to_fetch_press_releases_not_news(tools):
    result = tools.retrieve_press_releases("anything about meta", tickers="META")
    assert result["reason"] == "no_article_matches_filters" and not result["found"]
    assert "No stored press release matches" in result["message"] and "tickers META" in result["message"]
    assert "call fetch_press_releases" in result["message"] and "fetch_news" not in result["message"]


def test_press_releases_that_do_not_answer_say_no_supporting_text_in_their_own_words(tools):
    far = tools.retrieve_press_releases("cake")
    assert far["reason"] == "nothing_relevant" and far["message"].startswith("No supporting text: no stored press release scored 0.40 or more")
    assert "the stored press releases do not cover it" in far["message"] and "Do not answer from memory" in far["message"]
    near = tools.retrieve_press_releases("apple tesla car", tickers="AAPL")  # the best chunk scores 0.33, inside the near-miss margin
    assert "if you believe the stored press releases should cover this" in near["message"] and "search once more" in near["message"]


def test_an_empty_press_release_index_says_so_without_asking_for_a_fetch(tmp_path):
    empty = DataClerkTools(FakeToolbox(), press_store=make_store(tmp_path / "empty_pr", KeywordEmbeddings()))
    result = empty.retrieve_press_releases("apple")
    assert result["reason"] == "no_article_matches_filters" and "may be empty" in result["message"] and "fetch_press_releases" not in result["message"]


@pytest.mark.parametrize("kwargs", [dict(query=""), dict(query="apple", k=0), dict(query="apple", k=21), dict(query="apple", since_days=0),
                                    dict(query="apple", tickers="../../.env"), dict(query="apple", tickers=[1])])
def test_bad_press_release_search_arguments_are_tool_input_errors(tools, kwargs):
    with pytest.raises(ToolInputError):
        tools.retrieve_press_releases(**kwargs)


def test_the_model_sees_the_press_release_search_without_the_news_only_options(tools):
    retrieve = {t.name: t for t in tools.langchain_tools()}["retrieve_press_releases"]
    assert list(retrieve.args) == ["query", "tickers", "since_days", "k", "newest_first"]  # no diverse (one story per release), and min_score stays ours
    assert retrieve.get_input_schema().model_json_schema()["required"] == ["query"]
    assert "press release" in retrieve.description and "company's own statement" in retrieve.description and "no supporting text" in retrieve.description


def test_both_search_tools_number_their_sources_in_one_book_so_a_number_means_one_chunk(tools):
    """The reason the two tools share a SourceBook: [3] must never be a news article for one citation and a press release for another."""
    book = SourceBook()
    by_name = {t.name: t for t in tools.langchain_tools(book)}
    news = call(by_name["retrieve_news"], {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}, "c1")
    press = call(by_name["retrieve_press_releases"], {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}, "c2")
    assert [h["number"] for h in news.artifact["hits"]] == [1, 2]
    assert [(h["number"], h["article_id"]) for h in press.artifact["hits"]] == [(3, "pr:a1"), (4, "pr:a2")]  # new numbers, continuing
    assert "[3] Press release," in press.content and len(book) == 4 and book.check("[1] and [4]").ok and not book.check("[5]").ok


def test_without_sec_user_agent_the_edgar_step_is_reported_as_skipped_and_everything_else_still_happens(tools, toolbox, filings_index):
    """The 10-Q text is an extra step inside fetch_filings (design choices entry 36). Not being set up for EDGAR is not a failure: the Massive filings are fetched and
    indexed, and the result says what was not done and why, in words that hold no local path (the real error message names the .env file)."""
    toolbox.edgar_missing = True
    result = tools.fetch_filings("AAPL")
    assert [c[0] for c in toolbox.calls] == ["fetch_filings", "fetch_filing_text"]
    assert result["written"] == 1 and result["indexed"]["articles"] == 7 and result["indexed"]["added"] > 150  # the rest happened: the filings are stored and searchable
    assert set(result["edgar_text"]) == {"skipped"} and "SEC_USER_AGENT is not set" in result["edgar_text"]["skipped"] and "not fetched from EDGAR" in result["edgar_text"]["skipped"]
    assert "secret" not in str(result) and ".env" not in str(result) and "/home" not in str(result)
    assert "sec:0000320193-26-000006#0" in set(filings_index.get()["ids"])


def test_any_other_edgar_failure_stops_the_call_and_nothing_is_indexed(tools, toolbox, filings_index):
    """Only 'not set up' is reported and skipped. An HTTP error, or a 10-Q whose layout nobody has seen, stops the operation (as every error does)."""
    before = filings_index.get()["ids"]
    toolbox.edgar_fail = RuntimeError("EDGAR 0000320193-26-000006 HTTP 503")
    with pytest.raises(RuntimeError, match="HTTP 503"):
        tools.fetch_filings("AAPL")
    toolbox.edgar_fail = ValueError("0000320193-26-000006 (10-Q): no Item heading (like 'Item 1. Financial Statements') was found in the text")
    with pytest.raises(ValueError, match="no Item heading"):
        tools.fetch_filings("AAPL")
    assert filings_index.get()["ids"] == before


def test_when_the_massive_fetch_fails_edgar_is_not_asked(tools, toolbox):
    toolbox.fail = RuntimeError("Massive filings HTTP 429")
    with pytest.raises(RuntimeError, match="429"):
        tools.fetch_filings("AAPL")
    assert [c[0] for c in toolbox.calls] == ["fetch_filings"]


def test_the_model_is_told_about_the_edgar_step_and_what_to_do_when_it_was_skipped(tools):
    fetch = {t.name: t for t in tools.langchain_tools()}["fetch_filings"]
    text = " ".join(fetch.description.split())
    assert "fetches the text of the stored 10-Q filings from SEC EDGAR" in text and "edgar_text" in text and "skipped" in text
    assert "do not call this again for it" in text and list(fetch.args) == ["symbol", "from_date"]  # no new argument: the tool the model sees keeps its shape


# ---------------------------------------------------------------- retrieve_filings (rag/, the filings collection)

def test_retrieve_filings_returns_hits_as_plain_data(tools):
    result = tools.retrieve_filings("apple iphone phone", tickers="aapl", k=2)
    assert result["found"] and result["reason"] is None and result["message"] is None
    assert [h["article_id"] for h in result["hits"]] == ["sec:0000000001-25-000001:business", "sec:0000000001-25-000001:risk_factors"]  # one hit per SECTION, the exact match first
    assert {"chunk_id", "headline", "publisher", "published_utc", "url", "tickers", "text", "score"} <= set(result["hits"][0])
    assert result["hits"][0]["publisher"] == "SEC filing"


def test_the_three_searches_each_see_only_their_own_collection(tools):
    assert {h["article_id"] for h in tools.retrieve_news("nvidia chip", tickers="NVDA", k=5)["hits"]} == {"a3", "n1"}
    assert {h["article_id"] for h in tools.retrieve_press_releases("nvidia chip", tickers="NVDA", k=5)["hits"]} == {"pr:n1"}
    assert {h["article_id"] for h in tools.retrieve_filings("nvidia chip", tickers="NVDA", k=5)["hits"]} == {"sec:0000000002-25-000002:items"}


def test_retrieve_filings_filters_by_ticker_and_window_and_can_put_the_newest_first(tools):
    assert tools.retrieve_filings("apple iphone phone", tickers="NVDA")["found"] is False  # a ticker with no apple filing
    assert tools.retrieve_filings("apple iphone phone", tickers="AAPL", since_days=15)["found"] is False  # both Apple sections are 20 days old
    assert len(tools.retrieve_filings("apple iphone phone", tickers="AAPL", since_days=25)["hits"]) == 2
    newest = [h["published_utc"] for h in tools.retrieve_filings("apple iphone phone nvidia chip", newest_first=True, k=5, tickers=["AAPL", "NVDA"])["hits"]]
    assert newest == sorted(newest, reverse=True) and len(newest) == 3 and newest[0] > newest[1]  # the Nvidia filing (2 days old) before the two Apple sections (20)
    with pytest.raises(ToolInputError, match="newest_first must be true or false"):
        tools.retrieve_filings("apple iphone", newest_first="yes")


def test_a_ticker_with_no_stored_filing_tells_the_model_to_fetch_filings_not_news(tools):
    result = tools.retrieve_filings("anything about meta", tickers="META")
    assert result["reason"] == "no_article_matches_filters" and not result["found"]
    assert "No stored filing matches" in result["message"] and "tickers META" in result["message"]
    assert "call fetch_filings" in result["message"] and "fetch_news" not in result["message"] and "fetch_press_releases" not in result["message"]


def test_filings_that_do_not_answer_say_no_supporting_text_in_their_own_words(tools):
    far = tools.retrieve_filings("cake")
    assert far["reason"] == "nothing_relevant" and far["message"].startswith("No supporting text: no stored filing scored 0.40 or more")
    assert "the stored filings do not cover it" in far["message"] and "Do not answer from memory" in far["message"]
    near = tools.retrieve_filings("apple tesla car", tickers="AAPL")  # the best chunk scores 0.33, inside the near-miss margin
    assert "if you believe the stored filings should cover this" in near["message"] and "search once more" in near["message"]


def test_an_empty_filings_index_says_so_without_asking_for_a_fetch(tmp_path):
    empty = DataClerkTools(FakeToolbox(), filings_store=make_store(tmp_path / "empty_f", KeywordEmbeddings()))
    result = empty.retrieve_filings("apple")
    assert result["reason"] == "no_article_matches_filters" and "may be empty" in result["message"] and "fetch_filings" not in result["message"]


@pytest.mark.parametrize("kwargs", [dict(query=""), dict(query="apple", k=0), dict(query="apple", k=21), dict(query="apple", since_days=0),
                                    dict(query="apple", tickers="../../.env"), dict(query="apple", tickers=[1])])
def test_bad_filings_search_arguments_are_tool_input_errors(tools, kwargs):
    with pytest.raises(ToolInputError):
        tools.retrieve_filings(**kwargs)


def test_the_model_sees_the_filings_search_without_the_news_only_options_and_is_told_what_has_no_text(tools):
    retrieve = {t.name: t for t in tools.langchain_tools()}["retrieve_filings"]
    assert list(retrieve.args) == ["query", "tickers", "since_days", "k", "newest_first"]  # the same arguments as the press release search
    assert retrieve.get_input_schema().model_json_schema()["required"] == ["query"]
    text = " ".join(retrieve.description.split())  # the docstring wraps lines
    assert "company's own statement" in text and "no supporting text" in text and "10-K and 8-K filings are stored WITH text" in text
    assert "a 10-Q whose text was fetched from EDGAR" in text and "A 10-Q without it" in text  # entry 36: a 10-Q can have text now
    assert "never describe what such a filing contains" in text and "Form 144" in text and "call fetch_filings" in text


def test_a_filing_without_text_is_stored_as_a_source_that_says_so(tools, filings_index):
    """The real record of a 10-Q, through the real loader and ingest: what the model would read for it."""
    tools.fetch_filings("AAPL")  # indexes the six real filings
    stored = filings_index.get(ids=["sec:0000320193-26-000006#0"], include=["documents"])["documents"][0]
    assert "10-Q (quarterly report) filed 2026-01-30. The text of this filing is not stored" in stored


def test_all_three_search_tools_number_their_sources_in_one_book(tools):
    """The same rule with filings: [3] must never be a news article for one citation and a filing for another."""
    book = SourceBook()
    by_name = {t.name: t for t in tools.langchain_tools(book)}
    call(by_name["retrieve_news"], {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}, "c1")
    call(by_name["retrieve_press_releases"], {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}, "c2")
    filings = call(by_name["retrieve_filings"], {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}, "c3")
    assert [h["number"] for h in filings.artifact["hits"]] == [5, 6] and "[5] SEC filing," in filings.content and len(book) == 6
    assert book.check("[1] and [6]").ok and not book.check("[7]").ok


# ---------------------------------------------------------------- fetch_filings_recommendation (the stored filings read as evidence: design choices entry 38)

@pytest.fixture
def stored_apple(tools, toolbox):
    """The six real filings of Apple, stored, as `fetch_filings` leaves them (no 10-Q text, so no balance sheet: what data/filings/AAPL.txt holds today)."""
    shutil.copy(FILINGS_FIXTURE, toolbox.filings_dir / "AAPL.txt")
    tools.today = date(2026, 5, 1)  # the window is counted back from this day: a fixed one, not the clock
    return tools


def test_the_evidence_method_is_one_of_the_tools_the_model_sees_with_two_arguments_and_no_test_hook(tools):
    by_name = {t.name: t for t in tools.langchain_tools()}
    tool = by_name["fetch_filings_recommendation"]
    assert list(tool.args) == ["symbol", "since_days"] and tool.get_input_schema().model_json_schema()["required"] == ["symbol"]  # `today` is an attribute of the class, not an argument
    assert tool.response_format == "content_and_artifact" and len(by_name) == len(DATA_CLERK_TOOLS) == 13
    text = " ".join(tool.description.split())
    assert "only reads filings that are already stored" in text and "does NOT rate anything" in text and "Do not call a stock good or bad and do not give a score" in text
    assert "negative-type topics" in text and "positive-type topics" in text and "exact quote" in text and "call fetch_filings first" in text


def test_despite_its_name_the_evidence_tool_is_not_a_fetch_tool(tools):
    """It reads the stored files and costs nothing, so the budget does not turn it off with fetching and the MCP server offers it without --allow-fetch."""
    from prometheus.agents.data_clerk.data_clerk_agent import FETCH_TOOLS as agent_fetch
    from prometheus.mcp.data_clerk_server import FETCH_TOOLS as server_fetch
    assert "fetch_filings_recommendation" not in agent_fetch and "fetch_filings_recommendation" not in server_fetch
    assert "fetch_filings_recommendation" in DATA_CLERK_TOOLS and "fetch_filings_recommendation" not in data_clerk.RETRIEVE_TOOLS


def test_the_model_reads_the_digest_and_the_whole_evidence_travels_as_the_artifact(stored_apple, toolbox):
    tool = {t.name: t for t in stored_apple.langchain_tools()}["fetch_filings_recommendation"]
    message = call(tool, {"symbol": "aapl"})
    assert message.content.startswith("FILING EVIDENCE for AAPL (stored filings, last 365 days, as of 2026-04-20). This is evidence, not a score: nothing here rates the stock.")
    assert message.artifact["digest"] == message.content and message.artifact["symbol"] == "AAPL" and message.artifact["balance_sheet"] is None
    assert [e["id"] for e in message.artifact["evidence"]][:1] == ["E1"] and any("no balance sheet" in g for g in message.artifact["gaps"])
    assert "secret" not in message.content and toolbox.calls == []  # no toolbox call at all: it reads the stored files only (no network, no key, no cost)


def test_the_method_returns_the_same_result_the_tool_carries_and_takes_a_window(stored_apple):
    full = stored_apple.fetch_filings_recommendation("AAPL")
    assert full["window_days"] == 365 and full["filings"]["in_window"] == 6
    assert stored_apple.fetch_filings_recommendation("AAPL", since_days=60)["filings"]["in_window"] == 2  # the 8-K of 2026-04-20 and the Form 144 of 2026-04-02
    assert stored_apple.fetch_filings_recommendation("$aapl")["symbol"] == "AAPL"


@pytest.mark.parametrize("kwargs", [dict(symbol=""), dict(symbol="../../.env"), dict(symbol="AAPL", since_days=0), dict(symbol="AAPL", since_days=731), dict(symbol="AAPL", since_days=True),
                                    dict(symbol="AAPL", since_days="30"), dict(symbol=["AAPL"])])
def test_bad_arguments_of_the_evidence_tool_are_tool_input_errors(stored_apple, kwargs):
    with pytest.raises(ToolInputError):
        stored_apple.fetch_filings_recommendation(**kwargs)


def test_a_symbol_with_no_stored_filings_says_to_run_fetch_filings_first(stored_apple):
    with pytest.raises(ToolInputError, match="no filings of NVDA are stored: run fetch_filings for it first"):
        stored_apple.fetch_filings_recommendation("NVDA")


def test_a_broken_filing_file_stops_with_its_line_and_is_not_swallowed(tools, toolbox):
    (toolbox.filings_dir / "AAPL.txt").write_text("stray text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AAPL.txt:1: text outside a record"):
        tools.fetch_filings_recommendation("AAPL")


def test_the_evidence_result_scores_nothing_through_the_tool_too(stored_apple):
    from prometheus.agents.data_clerk.filing_signals import FORBIDDEN_KEYS, keys_of
    assert not [k for k in keys_of(stored_apple.fetch_filings_recommendation("AAPL")) if any(w in k.lower() for w in FORBIDDEN_KEYS)]


# ---------------------------------------------------------------- keyword search and the other data

def test_keyword_search_never_refreshes_and_shows_a_bounded_list(tools, toolbox):
    result = tools.keyword_search("nvda", ["  Huang ", "GPU"], from_date=10)
    assert toolbox.calls == [("search", ("NVDA", ["Huang", "GPU"]), dict(from_date=10, refresh=False))]
    assert result["news_total"] == 30 and len(result["news"]) == 20 and result["filings_total"] == 15 and len(result["filings"]) == 10
    assert result["filings_in_window"] == 15 and result["terms"] == ["Huang", "GPU"]


def test_discover_never_refreshes(tools, toolbox):
    result = tools.discover_tickers("  GLP-1 ")
    assert toolbox.calls == [("discover", ("GLP-1",), dict(from_date=7, top=10, refresh=False))]
    assert result["tickers"][0]["symbol"] == "MU"


def test_get_quote_passes_through_with_a_cleaned_symbol(tools, toolbox):
    assert tools.get_quote("$mu")["price"] == 410.25
    assert toolbox.calls == [("get_quote", ("MU",), {})]


@pytest.mark.parametrize("bad", ["../etc", "", "MU; DROP", 5])
def test_get_quote_refuses_a_symbol_that_is_not_a_ticker_before_anything_is_called(tools, toolbox, bad):
    with pytest.raises(ToolInputError):
        tools.get_quote(bad)
    assert toolbox.calls == []


def test_financials_pass_through_with_a_cleaned_symbol(tools, toolbox):
    assert tools.get_financials("$aapl")["metrics"]["pe_ratio"]["value"] == 31.2
    assert toolbox.calls == [("financials.get", ("AAPL",), {})]


def test_the_price_summary_drops_the_chart_curve_and_leaves_the_toolbox_result_alone(tools, toolbox):
    summary = tools.get_price_summary("aapl")
    assert summary["distribution"] == {"type": "normal", "mean": 339.8, "std_dev": 7.1} and summary["volume"] == 1000
    assert len(toolbox.performance_summary("AAPL")["distribution"]["curve"]) == 61  # a fresh call still has its curve


# ---------------------------------------------------------------- the LangChain edge

def call(tool, args, call_id="c1"):
    return tool.invoke({"name": tool.name, "args": args, "id": call_id, "type": "tool_call"})


def test_retrieve_news_as_a_langchain_tool_gives_numbered_sources_and_the_hits_as_the_artifact(tools):
    book = SourceBook()
    retrieve = tools.langchain_tools(book)[0]
    message = call(retrieve, {"query": "apple iphone phone", "tickers": "AAPL", "k": 2})
    assert message.content.startswith(citations.HEADER)
    assert "[1] Yahoo," in message.content and "[2] Yahoo," in message.content and "apple iphone phone" in message.content
    assert [h["number"] for h in message.artifact["hits"]] == [1, 2] and message.artifact["found"] is True
    assert [h["chunk_id"] for h in message.artifact["hits"]] == ["a1#0", "a2#0"]
    assert len(book) == 2 and book.check("yes [1]").ok


def test_numbers_continue_across_two_searches_in_one_run(tools):
    book = SourceBook()
    retrieve = tools.langchain_tools(book)[0]
    first = call(retrieve, {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}, "c1")
    second = call(retrieve, {"query": "nvidia chip", "tickers": "NVDA", "k": 2}, "c2")
    assert [h["number"] for h in first.artifact["hits"]] == [1, 2]
    numbers = {h["article_id"]: h["number"] for h in second.artifact["hits"]}
    assert numbers == {"n1": 3, "a3": 4}  # new chunks get new numbers
    third = call(retrieve, {"query": "apple iphone", "tickers": "AAPL", "k": 1}, "c3")
    assert third.artifact["hits"][0]["number"] == 1  # a chunk seen before keeps its number
    assert book.check("[1] and [3] and [4]").ok and not book.check("[5]").ok


def test_no_supporting_text_reaches_the_model_as_the_message_and_the_artifact_says_why(tools):
    message = call(tools.langchain_tools()[0], {"query": "cake"})
    assert message.content.startswith("No supporting text:") and message.status != "error"
    assert message.artifact == {"found": False, "reason": "nothing_relevant", "hits": []}


# ---------------------------------------------------------------- errors, through a small graph

def run_graph(tools_list, steps, handler=describe_tool_error):
    """A scripted 'model' asks for the tool calls in `steps` (one list per turn), a ToolNode runs them, and the
    run ends with the text 'done'. Returns every message."""
    script = iter(steps)
    counter = iter(range(1000))

    def agent(state: MessagesState):
        try:
            step = next(script)
        except StopIteration:
            return {"messages": [AIMessage(content="done")]}
        return {"messages": [AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{next(counter)}"} for n, a in step])]}

    node = ToolNode(tools_list, handle_tool_errors=handler) if handler else ToolNode(tools_list)
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile().invoke({"messages": [HumanMessage(content="hi")]})["messages"]


def test_a_failing_tool_becomes_a_message_and_the_run_goes_on(tools, toolbox):
    toolbox.fail = RuntimeError("Finnhub HTTP 429")
    messages = run_graph(tools.langchain_tools(), [[("fetch_news", {"symbol": "AAPL"})]])
    tool_message = next(m for m in messages if m.type == "tool")
    assert tool_message.status == "error"
    assert tool_message.content == ("The tool failed (RuntimeError): Finnhub HTTP 429. Do not repeat the same call. "
                                    "Tell the user what failed, or use a different tool.")
    assert messages[-1].content == "done"  # the run reached its last step instead of crashing


def test_without_the_handler_the_same_failure_ends_the_whole_run(tools, toolbox):
    toolbox.fail = RuntimeError("Finnhub HTTP 429")
    with pytest.raises(RuntimeError, match="429"):
        run_graph(tools.langchain_tools(), [[("fetch_news", {"symbol": "AAPL"})]], handler=None)


def test_bad_arguments_come_back_as_a_message_that_says_to_fix_them(tools, toolbox):
    messages = run_graph(tools.langchain_tools(), [[("fetch_news", {"symbol": "AAPL", "from_date": 5000})]])
    tool_message = next(m for m in messages if m.type == "tool")
    assert tool_message.status == "error"
    assert tool_message.content == "Invalid arguments: from_date must be a whole number of days from 1 to 90. Fix the arguments and call the tool again."
    assert toolbox.calls == []  # and the toolbox was never called


def test_a_missing_argument_and_an_unknown_tool_are_answered_without_a_crash(tools):
    messages = run_graph(tools.langchain_tools(), [[("retrieve_news", {}), ("export_project", {"dest": "/tmp/x"})]])
    by_name = {m.name: m for m in messages if m.type == "tool"}
    assert by_name["retrieve_news"].status == "error" and "query" in by_name["retrieve_news"].content
    assert by_name["export_project"].status == "error" and "retrieve_news" in by_name["export_project"].content  # lists the valid names
    assert messages[-1].content == "done"


def test_retrieve_news_works_through_the_graph_and_keeps_its_artifact(tools):
    messages = run_graph(tools.langchain_tools(SourceBook()), [[("retrieve_news", {"query": "apple iphone phone", "tickers": ["AAPL"]})]])
    tool_message = next(m for m in messages if m.type == "tool")
    assert tool_message.status != "error" and tool_message.content.startswith(citations.HEADER)
    assert tool_message.artifact["hits"][0]["number"] == 1


def test_error_messages_are_short_one_line_and_leak_no_arguments():
    long_text = "boom " * 200 + "\n" + "second line"
    message = describe_tool_error(RuntimeError(long_text))
    assert "\n" not in message and len(message) < 420
    assert describe_tool_error(ValueError("not ours")).startswith("The tool failed (ValueError)")  # only ToolInputError means "fix and retry"
    assert describe_tool_error(ToolInputError("k must be 1 to 20")).startswith("Invalid arguments: k must be 1 to 20.")
    limit = describe_tool_error(ToolLimitError("retrieve_news was already used 3 times for this question"))
    assert limit == ("Limit reached: retrieve_news was already used 3 times for this question. Do not ask for this tool again. "
                     "Answer now with the sources you already have, or say what they do not cover.")
