"""Tests for market://{symbol}/quote resource (stub now; Redis later)."""

import pytest
from fastmcp import Client

from prometheus.mcp.server import quote, server


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
    async with Client(server) as client:
        yield client


async def test_quote_via_mcp_stub(mcp_client):
    contents = await mcp_client.read_resource("market://AAPL/quote")
    text = contents[0].text
    assert "symbol=AAPL" in text
    assert "price=stub" in text


# ── Redis-backed quote (TODO) ───────────────────────────────────────────────


@pytest.mark.skip(reason="TODO: wire quote() to Redis, then assert real price payload")
def test_quote_from_redis_stub():
    """Stub: fetch market:{symbol}:quote from Redis and return live fields."""
    assert False, "not implemented"


@pytest.mark.skip(reason="TODO: MCP read_resource after Redis is wired")
async def test_quote_via_mcp_from_redis_stub(mcp_client):
    """Stub: read_resource('market://AAPL/quote') returns Redis-backed quote."""
    assert False, "not implemented"
