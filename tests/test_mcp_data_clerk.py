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


# ================================================================ filings (design choices entry 35)

@pytest.fixture
def filings_index(tmp_path):
    return fill(make_store(tmp_path / "chroma_f", KeywordEmbeddings()), [
        chunk("sec:0000000001-25-000001:business", 0, 1, "apple iphone phone", ["AAPL"], NOW - 20 * DAY, publisher="SEC filing"),
        chunk("sec:0000000001-25-000001:risk_factors", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 20 * DAY, publisher="SEC filing")])


@pytest.fixture
def filings_tools(index, press_index, filings_index, toolbox):
    return DataClerkTools(toolbox, index, press_index, filings_index)


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
