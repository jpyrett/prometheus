"""Tests for market quote resource + get_market_data prompt (stub now; Redis later)."""

import pytest

from prometheus.mcp.client import connect_client
from prometheus.mcp.server import get_market_data, quote


# ── Stub behavior (current placeholder in server.py) ────────────────────────


def test_quote_stub_normalizes_symbol():
    result = quote("aapl")
    assert "symbol=AAPL" in result
    assert "price=stub" in result
    assert "source=placeholder" in result


def test_quote_stub_trims_symbol():
    result = quote("  spy  ")
    assert "symbol=SPY" in result


@pytest.fixture
async def mcp_client():
    async with connect_client() as client:
        yield client


async def test_quote_via_mcp_stub(mcp_client):
    contents = await mcp_client.read_resource("market://AAPL/quote")
    text = contents[0].text
    assert "symbol=AAPL" in text
    assert "price=stub" in text


# ── get_market_data prompt (user intent; no URI/tool names in text) ─────────


def test_get_market_data_prompt_direct():
    text = get_market_data("  aapl  ")
    assert text.startswith("What's happening with AAPL?")
    assert "trading-desk briefing" in text
    # Prompt = intent only — must not name MCP URIs or tools
    assert "market://" not in text
    assert "read_resource" not in text
    assert "fetch_quote" not in text


async def test_get_market_data_prompt_via_mcp(mcp_client):
    prompts = await mcp_client.list_prompts()
    names = {p.name for p in prompts}
    assert "get_market_data" in names

    got = await mcp_client.get_prompt("get_market_data", {"symbol": "aapl"})
    text = got.messages[0].content.text
    assert "What's happening with AAPL?" in text
    assert "market://AAPL/quote" not in text


# ── Redis-backed quote (TODO) ───────────────────────────────────────────────


@pytest.mark.skip(reason="TODO: wire quote() to Redis, then assert real price payload")
def test_quote_from_redis_stub():
    """Stub: fetch market:{symbol}:quote from Redis and return live fields."""
    assert False, "not implemented"


@pytest.mark.skip(reason="TODO: MCP read_resource after Redis is wired")
async def test_quote_via_mcp_from_redis_stub(mcp_client):
    """Stub: read_resource('market://AAPL/quote') returns Redis-backed quote."""
    assert False, "not implemented"
