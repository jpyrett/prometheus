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


def test_retrieve_news_passes_newest_first_to_the_search_and_the_default_is_off(index, toolbox):
    tools = DataClerkTools(toolbox, index)
    plain = [h["chunk_id"] for h in tools.retrieve_news("apple iphone", tickers="AAPL", k=5)["hits"]]
    newest = [h["published_utc"] for h in tools.retrieve_news("apple iphone", tickers="AAPL", k=5, newest_first=True)["hits"]]
    assert newest == sorted(newest, reverse=True) and plain  # in date order only when asked
    with pytest.raises(ToolInputError, match="newest_first must be true or false"):
        tools.retrieve_news("apple iphone", newest_first="yes")


# ---------------------------------------------------------------- argument checks


def test_tickers_may_be_one_symbol_or_a_list_without_repeats():
    assert data_clerk._symbols(None) == [] and data_clerk._symbols("aapl") == ["AAPL"]
    assert data_clerk._symbols(["aapl", "$AAPL", "nvda"]) == ["AAPL", "NVDA"]
    for bad in (5, {"a": 1}, ["AAPL", 5], ["AAPL", "../x"], ["A"] * 11):
        with pytest.raises(ToolInputError):
            data_clerk._symbols(bad)


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


def test_a_near_miss_tells_the_model_to_reword_once_and_a_clear_miss_does_not(tools):
    near = tools.retrieve_news("apple tesla car", tickers="AAPL")  # the best AAPL chunk scores 0.33, within 0.10 of the 0.40 bar
    assert near["reason"] == "nothing_relevant" and "closest scored 0.33" in near["message"]
    assert "close to the bar" in near["message"] and "search once more with a clearer question" in near["message"]
    far = tools.retrieve_news("cake")
    assert "close to the bar" not in far["message"] and "search once more" not in far["message"]


# ---------------------------------------------------------------- the fetch tools (the toolbox, then the index)

def test_fetch_news_calls_the_toolbox_and_then_makes_the_new_articles_searchable(tools, toolbox, tmp_path):
    result = tools.fetch_news("$aapl", company="Apple", keywords=["iPhone"], from_date=14)
    assert toolbox.calls == [("fetch_news", (), dict(company="Apple", symbol="AAPL", keywords=["iPhone"], from_date=14))]
    assert result == {"symbol": "AAPL", "fetched": 9, "matched": 4, "written": 3, "duplicates": 1,
                      "indexed": {"articles": 3, "chunks": 4, "added": 4, "updated": 0, "unchanged": 0, "deleted": 0}}
    assert "secret" not in str(result) and "path" not in result  # the local file path stays out of the prompt
    again = tools.fetch_news("AAPL")  # same file again: everything is unchanged
    assert again["indexed"]["added"] == 0 and again["indexed"]["unchanged"] == 4


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


def test_fetch_filings_defaults_to_a_year_and_checks_its_arguments(tools, toolbox):
    tools.fetch_filings("AAPL")
    assert toolbox.calls[0][2] == toolbox.calls[1][2] == dict(from_date=365)  # both steps, a year
    for bad in (0, 731, True, "7", None):
        with pytest.raises(ToolInputError):
            tools.fetch_filings("AAPL", from_date=bad)
    with pytest.raises(ToolInputError):
        tools.fetch_filings("../../.env")
    assert [c[0] for c in toolbox.calls] == ["fetch_filings", "fetch_filing_text"]  # nothing was fetched for the bad ones


# ---------------------------------------------------------------- retrieve_press_releases (rag/, the press release collection)


def test_retrieve_press_releases_filters_by_ticker_and_window_and_can_put_the_newest_first(tools):
    assert tools.retrieve_press_releases("apple iphone phone", tickers="NVDA")["found"] is False  # a ticker with no apple release
    recent = tools.retrieve_press_releases("apple iphone phone", tickers="AAPL", since_days=15)
    assert [h["article_id"] for h in recent["hits"]] == ["pr:a2"]  # pr:a1 is 20 days old
    newest = [h["published_utc"] for h in tools.retrieve_press_releases("apple iphone phone", tickers="AAPL", newest_first=True)["hits"]]
    assert newest == sorted(newest, reverse=True) and len(newest) == 2
    with pytest.raises(ToolInputError, match="newest_first must be true or false"):
        tools.retrieve_press_releases("apple iphone", newest_first="yes")


def test_press_releases_that_do_not_answer_say_no_supporting_text_in_their_own_words(tools):
    far = tools.retrieve_press_releases("cake")
    assert far["reason"] == "nothing_relevant" and far["message"].startswith("No supporting text: no stored press release scored 0.40 or more")
    assert "the stored press releases do not cover it" in far["message"] and "Do not answer from memory" in far["message"]
    near = tools.retrieve_press_releases("apple tesla car", tickers="AAPL")  # the best chunk scores 0.33, inside the near-miss margin
    assert "if you believe the stored press releases should cover this" in near["message"] and "search once more" in near["message"]


@pytest.mark.parametrize("kwargs", [dict(query=""), dict(query="apple", k=0), dict(query="apple", k=21), dict(query="apple", since_days=0),
                                    dict(query="apple", tickers="../../.env"), dict(query="apple", tickers=[1])])
def test_bad_press_release_search_arguments_are_tool_input_errors(tools, kwargs):
    with pytest.raises(ToolInputError):
        tools.retrieve_press_releases(**kwargs)


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


# ---------------------------------------------------------------- retrieve_filings (rag/, the filings collection)


def test_retrieve_filings_filters_by_ticker_and_window_and_can_put_the_newest_first(tools):
    assert tools.retrieve_filings("apple iphone phone", tickers="NVDA")["found"] is False  # a ticker with no apple filing
    assert tools.retrieve_filings("apple iphone phone", tickers="AAPL", since_days=15)["found"] is False  # both Apple sections are 20 days old
    assert len(tools.retrieve_filings("apple iphone phone", tickers="AAPL", since_days=25)["hits"]) == 2
    newest = [h["published_utc"] for h in tools.retrieve_filings("apple iphone phone nvidia chip", newest_first=True, k=5, tickers=["AAPL", "NVDA"])["hits"]]
    assert newest == sorted(newest, reverse=True) and len(newest) == 3 and newest[0] > newest[1]  # the Nvidia filing (2 days old) before the two Apple sections (20)
    with pytest.raises(ToolInputError, match="newest_first must be true or false"):
        tools.retrieve_filings("apple iphone", newest_first="yes")


def test_filings_that_do_not_answer_say_no_supporting_text_in_their_own_words(tools):
    far = tools.retrieve_filings("cake")
    assert far["reason"] == "nothing_relevant" and far["message"].startswith("No supporting text: no stored filing scored 0.40 or more")
    assert "the stored filings do not cover it" in far["message"] and "Do not answer from memory" in far["message"]
    near = tools.retrieve_filings("apple tesla car", tickers="AAPL")  # the best chunk scores 0.33, inside the near-miss margin
    assert "if you believe the stored filings should cover this" in near["message"] and "search once more" in near["message"]


@pytest.mark.parametrize("kwargs", [dict(query=""), dict(query="apple", k=0), dict(query="apple", k=21), dict(query="apple", since_days=0),
                                    dict(query="apple", tickers="../../.env"), dict(query="apple", tickers=[1])])
def test_bad_filings_search_arguments_are_tool_input_errors(tools, kwargs):
    with pytest.raises(ToolInputError):
        tools.retrieve_filings(**kwargs)


# ---------------------------------------------------------------- fetch_filings_recommendation (the stored filings read as evidence: design choices entry 38)

@pytest.fixture
def stored_apple(tools, toolbox):
    """The six real filings of Apple, stored, as `fetch_filings` leaves them (no 10-Q text, so no balance sheet: what data/filings/AAPL.txt holds today)."""
    shutil.copy(FILINGS_FIXTURE, toolbox.filings_dir / "AAPL.txt")
    tools.today = date(2026, 5, 1)  # the window is counted back from this day: a fixed one, not the clock
    return tools


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


def test_error_messages_are_short_one_line_and_leak_no_arguments():
    long_text = "boom " * 200 + "\n" + "second line"
    message = describe_tool_error(RuntimeError(long_text))
    assert "\n" not in message and len(message) < 420
    assert describe_tool_error(ValueError("not ours")).startswith("The tool failed (ValueError)")  # only ToolInputError means "fix and retry"
    assert describe_tool_error(ToolInputError("k must be 1 to 20")).startswith("Invalid arguments: k must be 1 to 20.")
    limit = describe_tool_error(ToolLimitError("retrieve_news was already used 3 times for this question"))
    assert limit == ("Limit reached: retrieve_news was already used 3 times for this question. Do not ask for this tool again. "
                     "Answer now with the sources you already have, or say what they do not cover.")
