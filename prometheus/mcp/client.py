"""Part 4 — Connect a Client (host-side). Does not run the MCP server.

server.py must not import this module (avoids mixing stdio server with host code).

    from prometheus.mcp.client import connect_client
    async with connect_client() as c:
        ...

Defaults:
    mode=\"legacy\" — needed for elicitation demos (pushed form)
    cache=False — Never set cache=True or will break market://{symbol}/quote

Handlers configured:
    elicitation — HITL forms (enter_symbol)
    logging — server ctx.info / ctx.warning → log_handler
    progress — server ctx.report_progress → progress_handler
    message_handler — tools/prompts/resources list-changed + cancellation
                      (NOT resources/updated — quotes always change)

Not configured (deprecated / wrong layer for Prometheus):
    sampling — use LangGraph / OpenRouter / local LLM on the host instead
    roots — use tool args, resource URIs (e.g. market://{{symbol}}/quote), or config
"""

from __future__ import annotations

import sys

import mcp.types as mcp_types
from fastmcp import Client
from fastmcp.client.messages import MessageHandler

from prometheus.mcp.server import server

CLIENT_MODE = "legacy"
CLIENT_CACHE = False  # Never set cache=True or will break market://{symbol}/quote


async def elicitation_handler(message, response_type, params, context):
    """ELICITATION auto-answer for pytest / non-interactive demos (always AAPL)."""
    return {"symbol": "AAPL"}


async def interactive_elicitation_handler(message, response_type, params, context):
    """ELICITATION HITL: print server message and read a symbol from the terminal.

    Used by main.py. Matches server enter_symbol SymbolChoice.
    """
    print(f"\n[elicitation] {message}")
    if not sys.stdin.isatty():
        print("   (stdin is not a TTY — defaulting to AAPL)")
        return {"symbol": "AAPL"}
    raw = input("symbol> ").strip()
    return {"symbol": raw or "AAPL"}


async def log_handler(params):
    """LOGGING: receive log records the server emits (ctx.info / ctx.warning / …)."""
    data = params.data
    msg = data.get("msg", data) if isinstance(data, dict) else data
    print(f"   [server log · {params.level}] {msg}")


async def progress_handler(
    progress: float,
    total: float | None,
    message: str | None,
) -> None:
    """PROGRESS: long-running tools call ctx.report_progress; host can render a bar."""
    if total is not None and total > 0:
        pct = 100.0 * progress / total
        print(f"   [progress] {pct:.0f}% ({progress}/{total}) {message or ''}".rstrip())
    else:
        print(f"   [progress] {progress} {message or ''}".rstrip())


class PrometheusMessageHandler(MessageHandler):
    """LIST-CHANGED + CANCELLATION notifications.

    Refresh tools / prompts / static resource catalogs when the server says they
    changed. Ignore resources/updated — live quotes always move; we poll/read
    market://{symbol}/quote instead of subscribing to content updates.
    """

    async def on_tool_list_changed(
        self, notification: mcp_types.ToolListChangedNotification
    ) -> None:
        print("   [list-changed] tools — re-fetch tools/list")

    async def on_prompt_list_changed(
        self, notification: mcp_types.PromptListChangedNotification
    ) -> None:
        print("   [list-changed] prompts — re-fetch prompts/list")

    async def on_resource_list_changed(
        self, notification: mcp_types.ResourceListChangedNotification
    ) -> None:
        # Catalog of URIs (config://…, new static docs, etc.) — not quote ticks.
        print("   [list-changed] resources — re-fetch resources/list (static catalog)")

    async def on_resource_updated(
        self, notification: mcp_types.ResourceUpdatedNotification
    ) -> None:
        # Intentionally no-op: market://{symbol}/quote content always changes.
        return

    async def on_cancelled(
        self, notification: mcp_types.CancelledNotification
    ) -> None:
        params = notification.params
        reason = getattr(params, "reason", None) if params else None
        req_id = getattr(params, "requestId", None) if params else None
        print(f"   [cancelled] request_id={req_id} reason={reason}")


def connect_client(**kwargs):
    """In-process Client: mode=legacy, cache=False, elicitation + notifications."""
    kwargs.setdefault("mode", CLIENT_MODE)
    kwargs.setdefault("cache", CLIENT_CACHE)
    kwargs.setdefault("elicitation_handler", elicitation_handler)
    kwargs.setdefault("log_handler", log_handler)
    kwargs.setdefault("progress_handler", progress_handler)
    kwargs.setdefault("message_handler", PrometheusMessageHandler())
    return Client(server, **kwargs)


# Notebook-style alias
client = connect_client
