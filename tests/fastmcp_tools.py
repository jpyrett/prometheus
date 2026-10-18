"""Tests for Prometheus FastMCP tools in prometheus/mcp/server.py."""

import pytest
from fastmcp import Client

from prometheus.mcp.server import (
    add_initials_between,
    server,
    verify_add_initials_between,
)


# ── Pure unit tests (call functions directly) ───────────────────────────────


def test_add_initials_between_with_initials():
    assert add_initials_between("Alice", "Bob", "JP") == "Alice_JP_Bob"


def test_add_initials_between_without_initials():
    assert add_initials_between("Alice", "Bob") == "Alice_Bob"
    assert add_initials_between("Alice", "Bob", None) == "Alice_Bob"
    assert add_initials_between("Alice", "Bob", "") == "Alice_Bob"
    assert add_initials_between("Alice", "Bob", "   ") == "Alice_Bob"


def test_add_initials_between_trims():
    assert add_initials_between("  Alice  ", "  Bob  ", "  JP  ") == "Alice_JP_Bob"


def test_verify_add_initials_between_true():
    expr = '"Alice" add_initials_between "Bob" = "Alice_JP_Bob"'
    assert verify_add_initials_between(expr, "JP") is True
    assert verify_add_initials_between(expr, '"JP"') is True


def test_verify_add_initials_between_optional_initials():
    expr = '"Alice" add_initials_between "Bob" = "Alice_Bob"'
    assert verify_add_initials_between(expr) is True
    assert verify_add_initials_between(expr, None) is True
    assert verify_add_initials_between(expr, "") is True


def test_verify_add_initials_between_false_or_bad_format():
    expr = '"Alice" add_initials_between "Bob" = "WRONG"'
    assert verify_add_initials_between(expr, "JP") is False
    assert verify_add_initials_between("not a valid expression", "JP") is False
    assert verify_add_initials_between(
        '"Alice" add_initials_between "Bob" = "Alice_JP_Bob"',
        "J1",  # digits not allowed by initials regex
    ) is False


# ── MCP path (in-process Client ↔ server) ───────────────────────────────────


@pytest.fixture
async def mcp_client():
    async with Client(server) as client:
        yield client


async def test_list_tools_includes_initials_tools(mcp_client):
    tools = await mcp_client.list_tools()
    names = {t.name for t in tools}
    assert "add_initials_between" in names
    assert "verify_add_initials_between" in names


async def test_add_initials_between_via_mcp(mcp_client):
    result = await mcp_client.call_tool(
        "add_initials_between",
        {"a": "Alice", "b": "Bob", "initials": "JP"},
    )
    assert result.data == "Alice_JP_Bob"


async def test_add_initials_between_via_mcp_optional_initials(mcp_client):
    result = await mcp_client.call_tool(
        "add_initials_between",
        {"a": "Alice", "b": "Bob"},
    )
    assert result.data == "Alice_Bob"


async def test_verify_add_initials_between_via_mcp(mcp_client):
    result = await mcp_client.call_tool(
        "verify_add_initials_between",
        {
            "expression": '"Alice" add_initials_between "Bob" = "Alice_JP_Bob"',
            "initials": "JP",
        },
    )
    assert result.data is True
