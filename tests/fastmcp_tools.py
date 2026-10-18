"""Tests for Prometheus FastMCP tools in prometheus/mcp/server.py."""

import pytest

from prometheus.mcp.client import connect_client
from prometheus.mcp.server import (
    fetch_quote,
    verify_fetch_quote,
    _quote_payload,
)


# ── Pure unit tests (call functions directly) ───────────────────────────────


def test_fetch_quote_normalizes_symbol():
    assert fetch_quote("aapl") == "symbol=AAPL\nprice=stub\nsource=placeholder"
    assert fetch_quote("  spy  ") == _quote_payload("SPY")


def test_fetch_quote_strips_leading_dollar():
    assert fetch_quote("$AAPL") == "symbol=AAPL\nprice=stub\nsource=placeholder"
    assert fetch_quote("  $msft  ") == "symbol=MSFT\nprice=stub\nsource=placeholder"
    assert fetch_quote("$TEST") == "symbol=TEST\nprice=999.99\nsource=placeholder"


def test_fetch_quote_test_symbol_hardcoded_price():
    """Intentional DEMO fixture — remove with Redis. Accepts test / TEST."""
    expected = "symbol=TEST\nprice=999.99\nsource=placeholder"
    assert fetch_quote("test") == expected
    assert fetch_quote("TEST") == expected


def test_fetch_quote_rejects_blank():
    with pytest.raises(ValueError):
        fetch_quote("   ")


def test_verify_fetch_quote_true():
    payload = fetch_quote("AAPL").replace("\n", "\\n")
    expr = f'fetch_quote("AAPL") = "{payload}"'
    assert verify_fetch_quote(expr) is True


def test_verify_fetch_quote_true_lowercase_symbol():
    payload = fetch_quote("msft").replace("\n", "\\n")
    expr = f'fetch_quote("msft") = "{payload}"'
    assert verify_fetch_quote(expr) is True


def test_verify_fetch_quote_true_with_dollar():
    payload = fetch_quote("$AAPL").replace("\n", "\\n")
    expr = f'fetch_quote("$AAPL") = "{payload}"'
    assert verify_fetch_quote(expr) is True


def test_verify_fetch_quote_false_or_bad_format():
    assert verify_fetch_quote('fetch_quote("AAPL") = "WRONG"') is False
    assert verify_fetch_quote("not a valid expression") is False
    assert verify_fetch_quote('fetch_quote("") = "x"') is False


# ── MCP path (in-process Client ↔ server) ───────────────────────────────────


@pytest.fixture
async def mcp_client():
    async with connect_client() as client:
        yield client


async def test_list_tools_includes_quote_tools(mcp_client):
    tools = await mcp_client.list_tools()
    names = {t.name for t in tools}
    assert "fetch_quote" in names
    assert "verify_fetch_quote" in names
    assert "enter_symbol" in names
    assert "noisy_fetch_quote" in names
    assert "report_progress" in names
    assert "ping_catalog" in names


async def test_fetch_quote_via_mcp(mcp_client):
    result = await mcp_client.call_tool("fetch_quote", {"symbol": "aapl"})
    assert result.data == "symbol=AAPL\nprice=stub\nsource=placeholder"


async def test_verify_fetch_quote_via_mcp(mcp_client):
    payload = "symbol=AAPL\\nprice=stub\\nsource=placeholder"
    result = await mcp_client.call_tool(
        "verify_fetch_quote",
        {"expression": f'fetch_quote("AAPL") = "{payload}"'},
    )
    assert result.data is True


async def test_enter_symbol_elicitation_via_mcp(mcp_client):
    """Elicitation tool; client auto-answers symbol=AAPL."""
    result = await mcp_client.call_tool("enter_symbol", {})
    assert result.data == "AAPL"


async def test_noisy_fetch_quote_logging_via_mcp(mcp_client, capsys):
    """Logging tool; server ctx.info/warning reach log_handler."""
    result = await mcp_client.call_tool("noisy_fetch_quote", {"symbol": "aapl"})
    assert result.data == "symbol=AAPL\nprice=stub\nsource=placeholder"
    captured = capsys.readouterr().out
    assert "fetching quote for symbol=" in captured
    assert "stub" in captured


async def test_report_progress_via_mcp(mcp_client, capsys):
    """Progress tool; ctx.report_progress reaches progress_handler."""
    result = await mcp_client.call_tool("report_progress", {"steps": 2})
    assert result.data == "progress complete (2 steps)"
    captured = capsys.readouterr().out
    assert "[progress]" in captured


async def test_ping_catalog_list_changed_via_mcp(mcp_client, capsys):
    """List-changed for tools/prompts/static resources (not quote content)."""
    result = await mcp_client.call_tool("ping_catalog", {})
    assert result.data == "catalog list-changed notifications sent"
    captured = capsys.readouterr().out
    assert "[list-changed] tools" in captured
    assert "[list-changed] prompts" in captured
    assert "[list-changed] resources" in captured


def test_forced_failure_known_demo():
    """Intentionally fails so you can see a red pytest result."""
    print("KNOWN FAILURE: test_forced_failure_known_demo is a deliberate demo fail — ignore this one.")
    assert False, "KNOWN FAILURE: deliberate forced_failure demo (not a real regression)"
