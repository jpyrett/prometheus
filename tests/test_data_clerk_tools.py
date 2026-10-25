"""Tests for prometheus.tools.data_clerk (Part 6 of the RAG guide).

Run:  pytest tests/test_data_clerk_tools.py -q

A FAKE toolbox stands in for the DataAgent, so no test uses the network, a key or a paid API, and no test writes
to the real news files or the real index: every store and news folder lives in a temporary folder. The error
handling is tested through a small LangGraph graph, because ToolNode cannot be called by hand in LangGraph 1.x.
"""
import shutil
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.rag import citations
from prometheus.rag.citations import SourceBook
from prometheus.rag.config import REPO_ROOT
from prometheus.tools import data_clerk
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolInputError, ToolLimitError, describe_tool_error
from test_rag_retriever import DAY, NOW, KeywordEmbeddings, chunk, fill, make_store  # the same fake embedder and helpers

FIXTURE = Path(__file__).parent / "fixtures" / "news_sample.txt"


class FakeFinancials:
    def __init__(self, parent):
        self.parent = parent

    def get(self, symbol):
        self.parent.calls.append(("financials.get", (symbol,), {}))
        return {"symbol": symbol, "as_of": "2026-10-25", "metrics": {"pe_ratio": {"value": 31.2, "unit": "x"}}}


class FakeToolbox:
    """Records every call, returns canned results, and can be told to fail."""

    def __init__(self, news_dir: Path | None = None):
        self.calls: list[tuple] = []
        self.fail: Exception | None = None
        self.news_dir = news_dir
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
        return {"symbol": symbol, "path": "/home/someone/secret/pr.txt", "fetched": 5, "written": 2, "duplicates": 3}

    def fetch_filings(self, symbol, from_date=365, forms=()):
        self._call("fetch_filings", (symbol,), dict(from_date=from_date))
        return {"symbol": symbol, "path": "/home/someone/secret/f.txt", "fetched": 7, "written": 1, "duplicates": 6}

    def search(self, symbol, terms, from_date=30, refresh=True):
        self._call("search", (symbol, terms), dict(from_date=from_date, refresh=refresh))
        news = [{"published": "2026-10-2%d" % (i % 10), "publisher": "Yahoo", "headline": f"News {i}", "url": f"https://x/{i}"} for i in range(30)]
        filings = [{"filed": "2026-10-01", "form": "8-K", "url": f"https://sec/{i}", "text_status": "ok"} for i in range(15)]
        return {"symbol": symbol, "terms": terms, "from_date": from_date, "news": news, "filings": filings, "filings_in_window": 15}

    def discover(self, phrase, from_date=7, top=10, refresh=True):
        self._call("discover", (phrase,), dict(from_date=from_date, top=top, refresh=refresh))
        return {"phrase": phrase, "from_date": from_date, "scanned": 100, "matched": 4, "tickers": [{"symbol": "MU", "articles": 3}]}

    def performance_summary(self, symbol, points=61):
        self._call("performance_summary", (symbol,), {})
        return {"symbol": symbol, "volume": 1000, "average_price": 339.8,
                "distribution": {"type": "normal", "mean": 339.8, "std_dev": 7.1, "curve": [{"price": 1.0, "density": 0.1}] * 61}}


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    news = tmp_path / "news"
    news.mkdir()
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(news))
    return FakeToolbox(news)


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
def tools(toolbox, index) -> DataClerkTools:
    return DataClerkTools(toolbox, store=index)


# ---------------------------------------------------------------- the allow-list and what the model sees

def test_the_allow_list_is_exactly_what_the_model_is_shown(tools):
    assert DATA_CLERK_TOOLS == ("retrieve_news", "fetch_news", "fetch_press_releases", "fetch_filings",
                                "keyword_search", "discover_tickers", "get_financials", "get_price_summary")
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


def test_the_model_sees_the_arguments_of_retrieve_news_and_not_the_relevance_bar(tools):
    retrieve = tools.langchain_tools()[0]
    assert list(retrieve.args) == ["query", "tickers", "since_days", "k", "diverse"]  # min_score stays ours
    assert retrieve.get_input_schema().model_json_schema()["required"] == ["query"]
    assert "diverse" in retrieve.description and "no supporting text" in retrieve.description


def test_the_toolbox_of_the_real_default_is_rooted_where_rag_reads_its_files():
    # no network: building a DataAgent only sets paths
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
    assert [c[0] for c in toolbox.calls] == ["fetch_filings", "discover"]


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


def test_press_releases_and_filings_report_counts_not_paths(tools, toolbox):
    assert tools.fetch_press_releases("aapl") == {"symbol": "AAPL", "fetched": 5, "written": 2, "duplicates": 3}
    assert tools.fetch_filings("$meta", from_date=100) == {"symbol": "META", "fetched": 7, "written": 1, "duplicates": 6}
    assert toolbox.calls == [("fetch_press_releases", ("AAPL",), dict(from_date=365)), ("fetch_filings", ("META",), dict(from_date=100))]


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
