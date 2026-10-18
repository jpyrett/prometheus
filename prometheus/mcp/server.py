"""Run prometheus/mcp/server.py when you want to serve over stdio.

This module is the MCP *server* only (tools / resources / prompts).
Host/client connection lives in client.py — do not import client.py here.

Stdio hosts must connect with mode=\"legacy\" and cache=False
(see prometheus.mcp.client). Never set cache=True or will break the FastMCP
market://{symbol}/quote resource. Server emits no cache hints (cache_ttl=None).
"""

import logging

import mcp.types as mcp_types
from pydantic import BaseModel
from fastmcp import FastMCP, Context

# Quiet noisy SDK / framework logs.
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

# cache_ttl=None = no server-side cache hints (FastMCP has no cache= kwarg).
server = FastMCP("prometheus", cache_ttl=None)

# ── TOOLS ──  model-controlled actions; the type hints become the input schema.
#
# Notebook Part 5 mapping (what we port vs skip):
#   elicitation (enter_symbol) — YES: HITL still current; see enter_symbol below
#   logging / progress / list-changed / cancellation — YES: see noisy_fetch_quote, report_progress, ping_catalog, client handlers
#   sampling (summarize via create_message) — NO: deprecated; summarize in LangGraph/LLM host
#   roots (list_workspace via list_roots) — NO: deprecated; use tool args / resource URIs / config
#   resource *content* updates (quotes) — NO: prices always move; no resources/updated for quotes
#
# Quote data: tool fetch_quote = model-triggered action; resource market://{symbol}/quote =
# host-loaded context. Both share _quote_payload (Redis later).


def _normalize_symbol(symbol: str) -> str:
    """Trim, drop a leading ``$`` (e.g. ``$AAPL``), and uppercase."""
    symbol = symbol.strip()
    if symbol.startswith("$"):
        symbol = symbol[1:].strip()
    symbol = symbol.upper()
    if not symbol:
        raise ValueError("symbol must be non-empty")
    return symbol


def _quote_payload(symbol: str) -> str:
    """Normalize symbol and return quote text (stub now; Redis later).

    Leading ``$`` is allowed (``$AAPL`` → ``AAPL``). Intentional demo fixture:
    symbol ``TEST`` returns a hardcoded price of ``999.99`` so LocalLLM smoke
    runs have a real-looking number (``price=stub`` makes the model claim “no
    quote data”). TODO: remove this TEST special-case once Redis / live quotes
    are wired.
    """
    symbol = _normalize_symbol(symbol)
    # TODO: remove TEST hardcode when Redis / live quotes land
    if symbol == "TEST":
        return f"symbol={symbol}\nprice=999.99\nsource=placeholder"
    # TODO: redis.get(f"market:{symbol}:quote")
    return f"symbol={symbol}\nprice=stub\nsource=placeholder"


@server.tool()
def fetch_quote(symbol: str) -> str:
    """Fetch the latest quote for a market symbol (trading-desk action).

    Model-controlled: call when the agent needs a price now. Same payload as
    the market://{symbol}/quote resource (shared stub / Redis later).
    """
    return _quote_payload(symbol)


@server.tool()
def verify_fetch_quote(expression: str) -> bool:
    """Check whether fetch_quote(symbol) equals the claimed result.

    Expected expression (quoted symbol + quoted payload; use \\n for newlines):
        fetch_quote("AAPL") = "symbol=AAPL\\nprice=stub\\nsource=placeholder"
        fetch_quote("$AAPL") = "symbol=AAPL\\nprice=stub\\nsource=placeholder"
    """
    import re

    expr_m = re.match(
        r"""\s*fetch_quote\s*\(\s*(["'])(\$?[A-Za-z0-9.\-]+)\1\s*\)\s*=\s*(["'])(.*?)\3\s*$""",
        expression,
        flags=re.DOTALL,
    )
    if not expr_m:
        return False

    symbol, claimed = expr_m[2].strip(), expr_m[4]
    # Allow escaped newlines in the claimed quoted string.
    claimed = claimed.replace("\\n", "\n")
    try:
        return _quote_payload(symbol) == claimed
    except ValueError:
        return False


class SymbolChoice(BaseModel):
    symbol: str


@server.tool()
async def enter_symbol(ctx: Context) -> str:
    """ELICITATION: ask the user for a market symbol mid-tool, then use the answer.

    Host must supply elicitation_handler (see prometheus.mcp.client).
    """
    result = await ctx.elicit(
        message="Which market symbol do you want?",
        response_type=SymbolChoice,
    )
    if result.action == "accept":
        return _normalize_symbol(result.data.symbol)
    raise RuntimeError(f"enter_symbol elicitation not accepted (action={result.action})")


@server.tool()
async def noisy_fetch_quote(symbol: str, ctx: Context) -> str:
    """LOGGING: fetch_quote with ctx.info / ctx.warning along the way.

    Same payload as fetch_quote. Host must supply log_handler (see prometheus.mcp.client).
    """
    await ctx.info(f"fetching quote for symbol={symbol!r}")
    payload = _quote_payload(symbol)
    await ctx.warning("quote source is still a stub (Redis later)")
    return payload


@server.tool()
async def report_progress(ctx: Context, steps: int = 3) -> str:
    """PROGRESS demo: multi-step tool that reports progress to the host.

    Host must supply progress_handler (see prometheus.mcp.client).
    """
    n = max(1, int(steps))
    for i in range(1, n + 1):
        await ctx.report_progress(progress=i, total=n, message=f"step {i}/{n}")
        await ctx.info(f"report_progress step {i}/{n}")
    return f"progress complete ({n} steps)"


@server.tool()
async def ping_catalog(ctx: Context) -> str:
    """LIST-CHANGED: notify host that tools / prompts / static resources may have changed.

    Does not cover quote *content* updates (market://…/quote always moves — no
    resources/updated). Use when the catalog of tools, prompts, or static URIs changes.
    """
    await ctx.send_notification(mcp_types.ToolListChangedNotification())
    await ctx.send_notification(mcp_types.PromptListChangedNotification())
    await ctx.send_notification(mcp_types.ResourceListChangedNotification())
    return "catalog list-changed notifications sent"


# ── RESOURCES ──  application-controlled, read-only context addressed by URI.
# Prefer resource URIs / tool args / config over deprecated client "roots".
# Resource = where the host loads data into context (e.g. market://AAPL/quote).
# tools = “do something”; resources = “look this up” — not user-intent templates.

@server.resource("config://server")
def server_config() -> str:
    """A STATIC resource — always the same URI, returns fixed text."""
    return "name=prometheus\nversion=1.0\nmax_results=10"


@server.resource("market://{symbol}/quote")
def quote(symbol: str) -> str:
    """Read-only quote for a market symbol (e.g. market://AAPL/quote).

    Resource role: host loads this URI into context (Redis-backed later).
    Pair with the get_market_data *prompt* for user intent; this handler only
    supplies data — it is not the briefing template. Same payload as fetch_quote.

    Never set cache=True or will break the FastMCP market://{symbol}/quote resource.
    Hosts: use prometheus.mcp.client.connect_client() (cache=False).
    """
    return _quote_payload(symbol)


# ── PROMPTS ──  user-controlled reusable templates.
# Prompt = user intent (e.g. “What’s happening with AAPL?”).
# Prompt text must not name URIs or tools — that’s app/agent wiring, not the
# template’s job. Host loads market://AAPL/quote (resource) into context separately.

@server.prompt()
def get_market_data(symbol: str) -> str:
    """Prompt = user intent (“What’s happening with AAPL?”), slash-command style.

    Returns briefing task text only. Does not name URIs or tools — app/agent
    wiring loads market://{symbol}/quote into context via the quote resource.
    """
    symbol = _normalize_symbol(symbol)
    return (
        f"What's happening with {symbol}?\n\n"
        "Give a concise trading-desk briefing for this symbol: recent price "
        "context, notable moves, and what to watch next. Use only facts present "
        "in the conversation context; if quote data is missing, say so clearly "
        "instead of inventing numbers."
    )


if __name__ == "__main__":
    # stdio — host must connect with mode="legacy" and cache=False (see client.py).
    server.run()
