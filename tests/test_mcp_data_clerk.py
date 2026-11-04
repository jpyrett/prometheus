"""Tests for prometheus.mcp.data_clerk_server (Part 8 of the RAG guide).

Run:  pytest tests/test_mcp_data_clerk.py -q

The server is exercised through FastMCP's in-process Client, the same way the existing MCP tests do it, with a fake toolbox and a small
temporary index, so there is no network, no key and no cost. One test starts the real server as a subprocess over stdio and only
lists its tools (it proves the protocol works and that nothing but the protocol is written to stdout).
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from prometheus.mcp import data_clerk_server as dcs
from prometheus.rag.config import REPO_ROOT
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools
from test_data_clerk_tools import FakeToolbox
from test_rag_retriever import DAY, NOW, KeywordEmbeddings, chunk, fill, make_store

FETCH = ("fetch_news", "fetch_press_releases", "fetch_filings")
NEVER_SERVED = ("init_aws_cli", "fetch_historical_data", "convert_to_parquet", "export_project", "sync_market_news", "sync_cnbc_news")


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


@pytest.fixture
def tools(index, toolbox):
    return DataClerkTools(toolbox, index)


def served(tools, allow_fetch=False):
    return Client(dcs.build_server(tools, allow_fetch=allow_fetch))


def as_json(value):
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------- what is offered

async def test_by_default_only_the_search_and_read_tools_are_offered(tools):
    async with served(tools) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == set(DATA_CLERK_TOOLS) - set(FETCH) == {"retrieve_news", "retrieve_press_releases", "retrieve_filings", "fetch_filings_recommendation", "keyword_search",
                                                           "discover_tickers", "get_financials", "get_price_summary", "get_quote"}


async def test_with_fetching_allowed_the_whole_allow_list_is_offered_and_nothing_else(tools):
    async with served(tools, allow_fetch=True) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == set(DATA_CLERK_TOOLS) and not names & set(NEVER_SERVED)


async def test_the_dangerous_tools_are_never_offered_even_when_fetching_is_allowed(tools):
    async with served(tools, allow_fetch=True) as c:
        for name in NEVER_SERVED:
            with pytest.raises(ToolError):
                await c.call_tool(name, {})


def test_served_tools_names_the_same_set_as_the_server_offers():
    assert dcs.served_tools() == tuple(n for n in DATA_CLERK_TOOLS if n not in FETCH) and dcs.served_tools(True) == DATA_CLERK_TOOLS


async def test_the_hints_to_the_host_say_which_tools_read_and_which_write(tools):
    async with served(tools, allow_fetch=True) as c:
        hints = {t.name: t.annotations for t in await c.list_tools()}
    for name in ("retrieve_news", "retrieve_press_releases", "retrieve_filings", "fetch_filings_recommendation", "keyword_search", "discover_tickers", "get_price_summary", "get_financials", "get_quote"):
        assert hints[name].read_only_hint is True, name
    assert hints["fetch_filings_recommendation"].open_world_hint is False  # it reads the stored filings only, whatever its name says
    assert hints["retrieve_press_releases"].open_world_hint is False and hints["retrieve_filings"].open_world_hint is False  # they read the stored text only, like retrieve_news
    assert hints["retrieve_news"].open_world_hint is False and hints["get_financials"].open_world_hint is True
    assert hints["get_quote"].open_world_hint is True  # it calls Finnhub, and writes nothing
    for name in FETCH:
        assert hints[name].read_only_hint is False and hints[name].destructive_hint is False and hints[name].open_world_hint is True, name


# ---------------------------------------------------------------- the same tools as the agent's

async def test_each_served_tool_has_the_same_arguments_and_description_as_the_langchain_tool(tools):
    langchain = {t.name: t for t in tools.langchain_tools()}
    async with served(tools, allow_fetch=True) as c:
        for t in await c.list_tools():
            schema, lc = t.input_schema, langchain[t.name]
            assert list(schema["properties"]) == list(lc.args), t.name
            assert set(schema.get("required", [])) == set(lc.get_input_schema().model_json_schema().get("required", [])), t.name
            flat = lambda text: " ".join(text.split())  # LangChain joins the first paragraph onto one line and drops the Args block
            assert flat(t.description).startswith(flat(lc.description)), t.name
            for arg, spec in schema["properties"].items():  # the guidance for each argument is the same text in both
                assert flat(spec.get("description") or "") == flat(lc.args[arg].get("description") or ""), (t.name, arg)
                assert spec.get("description"), (t.name, arg)  # and none is missing


async def test_retrieve_news_over_mcp_returns_exactly_what_the_method_returns(tools):
    args = {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}
    direct = as_json(tools.retrieve_news(**args))
    async with served(tools) as c:
        over_mcp = (await c.call_tool("retrieve_news", args)).structured_content
    assert over_mcp == direct and [h["chunk_id"] for h in over_mcp["hits"]] == ["a1#0", "a2#0"] and over_mcp["found"] is True


async def test_a_search_with_nothing_relevant_comes_back_as_a_message_not_an_error(tools):
    async with served(tools) as c:
        result = (await c.call_tool("retrieve_news", {"query": "cake"})).structured_content
    assert result["found"] is False and result["hits"] == [] and "No supporting text" in result["message"]


async def test_the_server_returns_hits_without_numbers_because_the_caller_owns_the_numbering(tools):
    async with served(tools) as c:
        hits = (await c.call_tool("retrieve_news", {"query": "apple iphone phone", "tickers": "AAPL"})).structured_content["hits"]
    assert hits and all("number" not in h for h in hits)
    assert set(hits[0]) >= {"chunk_id", "article_id", "headline", "publisher", "published_utc", "url", "tickers", "text", "score"}


async def test_another_tool_over_mcp_returns_exactly_what_the_method_returns(tools):
    args = {"symbol": "aapl", "terms": ["iphone"], "from_date": 7}
    direct = as_json(tools.keyword_search(**args))
    async with served(tools) as c:
        over_mcp = (await c.call_tool("keyword_search", args)).structured_content
    assert over_mcp == direct and over_mcp["symbol"] == "AAPL" and over_mcp["news_total"] == 30


# ---------------------------------------------------------------- the same argument checks protect this way in too

@pytest.mark.parametrize("name, args, message", [
    ("retrieve_news", {"query": "apple", "k": 99}, "k must be"),
    ("retrieve_news", {"query": "apple", "tickers": "../etc"}, "not a valid ticker"),
    ("keyword_search", {"symbol": "../etc", "terms": ["x"]}, "not a valid ticker"),
    ("keyword_search", {"symbol": "AAPL", "terms": []}, "terms needs at least one word"),
    ("discover_tickers", {"phrase": "x", "from_date": 999}, "from_date must be"),
])
async def test_arguments_that_are_not_allowed_are_refused_with_the_reason(tools, name, args, message):
    async with served(tools) as c:
        with pytest.raises(ToolError) as caught:
            await c.call_tool(name, args)
    assert message in str(caught.value) and "Traceback" not in str(caught.value) and "site-packages" not in str(caught.value)


async def test_a_toolbox_failure_reaches_the_host_as_an_error_without_a_path_or_a_key(tools, toolbox):
    toolbox.fail = RuntimeError("Finnhub HTTP 429")
    async with served(tools) as c:
        with pytest.raises(ToolError) as caught:
            await c.call_tool("keyword_search", {"symbol": "AAPL", "terms": ["x"]})
    assert "429" in str(caught.value) and "/home/" not in str(caught.value)


# ---------------------------------------------------------------- fetching

async def test_without_the_flag_a_fetch_cannot_be_called_and_the_toolbox_is_never_touched(tools, toolbox):
    async with served(tools) as c:
        with pytest.raises(ToolError):
            await c.call_tool("fetch_news", {"symbol": "AAPL"})
    assert toolbox.calls == []


async def test_with_the_flag_a_fetch_runs_and_the_new_articles_become_searchable(tools, toolbox, index):
    assert "finnhub:143136906#0" not in index.get()["ids"]
    async with served(tools, allow_fetch=True) as c:
        result = (await c.call_tool("fetch_news", {"symbol": "AAPL"})).structured_content
    assert [x[0] for x in toolbox.calls] == ["fetch_news"] and result["symbol"] == "AAPL" and result["indexed"]["added"] > 0
    assert "path" not in result and "finnhub:143136906#0" in index.get()["ids"]


# ---------------------------------------------------------------- the command line and the real stdio server

def test_the_command_line_warns_when_fetching_is_on_and_stays_quiet_when_it_is_off(monkeypatch, capsys):
    seen = {}

    class Stub:
        def run(self, **options):
            seen.update(ran=True, run_options=options)

    monkeypatch.setattr(dcs, "build_server", lambda allow_fetch=False: seen.update(allow_fetch=allow_fetch) or Stub())
    assert dcs.main([]) == 0 and seen == {"allow_fetch": False, "ran": True, "run_options": {"show_banner": False}} and capsys.readouterr().err == ""
    assert dcs.main(["--allow-fetch"]) == 0 and seen["allow_fetch"] is True
    assert "WARNING: the fetch tools are on" in capsys.readouterr().err


async def test_the_real_server_speaks_mcp_over_stdio_and_writes_nothing_else_to_stdout():
    transport = StdioTransport(command=sys.executable, args=["-m", "prometheus.mcp.data_clerk_server"], cwd=str(REPO_ROOT))
    async with Client(transport, timeout=90) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == set(DATA_CLERK_TOOLS) - set(FETCH)


# ================================================================ press releases (design choices entry 34)

@pytest.fixture
def press_index(tmp_path):
    return fill(make_store(tmp_path / "chroma_pr", KeywordEmbeddings()), [
        chunk("pr:a1", 0, 1, "apple iphone phone", ["AAPL"], NOW - 20 * DAY, publisher="Press release"),
        chunk("pr:a2", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 10 * DAY, publisher="Press release")])


@pytest.fixture
def press_tools(index, press_index, toolbox):
    return DataClerkTools(toolbox, index, press_index)


async def test_retrieve_press_releases_is_offered_by_default_and_fetch_press_releases_only_with_the_flag(press_tools):
    async with served(press_tools) as c:
        offered = {t.name for t in await c.list_tools()}
    assert "retrieve_press_releases" in offered and "fetch_press_releases" not in offered
    async with served(press_tools, allow_fetch=True) as c:
        assert {"retrieve_press_releases", "fetch_press_releases"} <= {t.name for t in await c.list_tools()}


async def test_retrieve_press_releases_over_mcp_returns_exactly_what_the_method_returns_without_numbers(press_tools):
    args = {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}
    direct = as_json(press_tools.retrieve_press_releases(**args))
    async with served(press_tools) as c:
        over_mcp = (await c.call_tool("retrieve_press_releases", args)).structured_content
    assert over_mcp == direct and [h["chunk_id"] for h in over_mcp["hits"]] == ["pr:a1#0", "pr:a2#0"] and all("number" not in h for h in over_mcp["hits"])
    assert over_mcp["hits"][0]["publisher"] == "Press release"


async def test_the_two_searches_over_mcp_see_only_their_own_collection(press_tools):
    async with served(press_tools) as c:
        news = (await c.call_tool("retrieve_news", {"query": "apple iphone phone", "tickers": "AAPL"})).structured_content["hits"]
        press = (await c.call_tool("retrieve_press_releases", {"query": "apple iphone phone", "tickers": "AAPL"})).structured_content["hits"]
    assert not {h["article_id"] for h in news} & {h["article_id"] for h in press} and all(h["article_id"].startswith("pr:") for h in press)


async def test_a_bad_press_release_search_is_refused_with_the_reason_over_mcp(press_tools):
    async with served(press_tools) as c:
        with pytest.raises(ToolError, match="k must be"):
            await c.call_tool("retrieve_press_releases", {"query": "apple", "k": 99})
        with pytest.raises(ToolError, match="not a valid ticker"):
            await c.call_tool("retrieve_press_releases", {"query": "apple", "tickers": "../etc"})


async def test_without_the_flag_a_press_release_fetch_cannot_be_called_and_the_toolbox_is_never_touched(press_tools, toolbox):
    async with served(press_tools) as c:
        with pytest.raises(ToolError):
            await c.call_tool("fetch_press_releases", {"symbol": "AAPL"})
    assert toolbox.calls == []


async def test_with_the_flag_a_press_release_fetch_runs_and_the_new_releases_become_searchable(press_tools, toolbox, press_index):
    async with served(press_tools, allow_fetch=True) as c:
        result = (await c.call_tool("fetch_press_releases", {"symbol": "aapl", "from_date": 30})).structured_content
    assert toolbox.calls == [("fetch_press_releases", ("AAPL",), dict(from_date=30))] and result["indexed"]["added"] == 3
    assert "pr:44107d69ebe0e861#0" in press_index.get()["ids"] and "secret" not in str(result)  # the three real releases of the fixture, and no local path


# ================================================================ filings (design choices entry 35)

@pytest.fixture
def filings_index(tmp_path):
    return fill(make_store(tmp_path / "chroma_f", KeywordEmbeddings()), [
        chunk("sec:0000000001-25-000001:business", 0, 1, "apple iphone phone", ["AAPL"], NOW - 20 * DAY, publisher="SEC filing"),
        chunk("sec:0000000001-25-000001:risk_factors", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 20 * DAY, publisher="SEC filing")])


@pytest.fixture
def filings_tools(index, press_index, filings_index, toolbox):
    return DataClerkTools(toolbox, index, press_index, filings_index)


async def test_retrieve_filings_is_offered_by_default_and_fetch_filings_only_with_the_flag(filings_tools):
    async with served(filings_tools) as c:
        offered = {t.name for t in await c.list_tools()}
    assert "retrieve_filings" in offered and "fetch_filings" not in offered and len(offered) == 9  # six before filings were searchable, seven before the evidence tool, eight before get_quote (entry 42)
    async with served(filings_tools, allow_fetch=True) as c:
        assert {"retrieve_filings", "fetch_filings"} <= {t.name for t in await c.list_tools()}


async def test_retrieve_filings_over_mcp_returns_exactly_what_the_method_returns_without_numbers(filings_tools):
    args = {"query": "apple iphone phone", "tickers": "AAPL", "k": 2}
    direct = as_json(filings_tools.retrieve_filings(**args))
    async with served(filings_tools) as c:
        over_mcp = (await c.call_tool("retrieve_filings", args)).structured_content
    assert over_mcp == direct and [h["chunk_id"] for h in over_mcp["hits"]] == ["sec:0000000001-25-000001:business#0", "sec:0000000001-25-000001:risk_factors#0"]
    assert all("number" not in h for h in over_mcp["hits"]) and over_mcp["hits"][0]["publisher"] == "SEC filing"


async def test_the_three_searches_over_mcp_see_only_their_own_collection(filings_tools):
    async with served(filings_tools) as c:
        hits = {name: (await c.call_tool(name, {"query": "apple iphone phone", "tickers": "AAPL"})).structured_content["hits"]
                for name in ("retrieve_news", "retrieve_press_releases", "retrieve_filings")}
    ids = {name: {h["article_id"] for h in found} for name, found in hits.items()}
    assert all(i.startswith("sec:") for i in ids["retrieve_filings"]) and all(i.startswith("pr:") for i in ids["retrieve_press_releases"])
    assert not any(i.startswith(("sec:", "pr:")) for i in ids["retrieve_news"]) and not ids["retrieve_filings"] & ids["retrieve_press_releases"]


async def test_a_bad_filings_search_is_refused_with_the_reason_over_mcp(filings_tools):
    async with served(filings_tools) as c:
        with pytest.raises(ToolError, match="k must be"):
            await c.call_tool("retrieve_filings", {"query": "apple", "k": 99})
        with pytest.raises(ToolError, match="not a valid ticker"):
            await c.call_tool("retrieve_filings", {"query": "apple", "tickers": "../etc"})


async def test_without_the_flag_a_filings_fetch_cannot_be_called_and_the_toolbox_is_never_touched(filings_tools, toolbox):
    async with served(filings_tools) as c:
        with pytest.raises(ToolError):
            await c.call_tool("fetch_filings", {"symbol": "AAPL"})
    assert toolbox.calls == []


async def test_with_the_flag_a_filings_fetch_runs_and_the_new_filings_become_searchable(filings_tools, toolbox, filings_index):
    async with served(filings_tools, allow_fetch=True) as c:
        result = (await c.call_tool("fetch_filings", {"symbol": "aapl", "from_date": 30})).structured_content
    assert toolbox.calls == [("fetch_filings", ("AAPL",), dict(from_date=30)), ("fetch_filing_text", ("AAPL",), dict(from_date=30))]  # the EDGAR text step runs inside it
    assert result["indexed"]["articles"] == 7 and result["indexed"]["added"] > 150 and result["edgar_text"] == {"candidates": 3, "written": 1}
    assert "sec:0000320193-25-000077:items#0" in filings_index.get()["ids"] and "secret" not in str(result)  # the real filings of the fixture, and no local path


async def test_the_evidence_tool_is_offered_without_the_fetch_flag_and_over_mcp_returns_the_whole_result(filings_tools, toolbox):
    shutil.copy(Path(__file__).parent / "fixtures" / "filings_sample.txt", toolbox.filings_dir / "AAPL.txt")
    filings_tools.today = date(2026, 5, 1)
    direct = as_json(filings_tools.fetch_filings_recommendation("AAPL"))
    async with served(filings_tools) as c:  # no allow_fetch: it is not a fetch tool
        assert "fetch_filings_recommendation" in {t.name for t in await c.list_tools()}
        over_mcp = (await c.call_tool("fetch_filings_recommendation", {"symbol": "aapl"})).structured_content
        with pytest.raises(ToolError, match="no filings of NVDA are stored"):
            await c.call_tool("fetch_filings_recommendation", {"symbol": "NVDA"})
        with pytest.raises(ToolError, match="not a valid ticker"):
            await c.call_tool("fetch_filings_recommendation", {"symbol": "../etc"})
    assert over_mcp == direct and over_mcp["digest"].startswith("FILING EVIDENCE for AAPL") and over_mcp["evidence"] and toolbox.calls == []
