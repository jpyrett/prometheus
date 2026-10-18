"""Run prometheus/mcp/server.py when you want to serve over stdio.

This module is the MCP *server* only (tools / resources / prompts).
Host/client connection lives in client.py — do not import client.py here.

Stdio hosts must connect with mode=\"legacy\" and cache=False
(see prometheus.mcp.client). Never set cache=True or will break the FastMCP
market://{symbol}/quote resource. Server emits no cache hints (cache_ttl=None).
"""

import logging

from fastmcp import FastMCP

# Quiet noisy SDK / framework logs.
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

# cache_ttl=None = no server-side cache hints (FastMCP has no cache= kwarg).
server = FastMCP("prometheus", cache_ttl=None)

# ── TOOLS ──  model-controlled actions; the type hints become the input schema.
@server.tool()
def add_initials_between(a: str, b: str, initials: str | None = None) -> str:
    """Insert trimmed initials between two trimmed strings, joined with underscores.

    initials is optional: None or blank -> "Alice_Bob"; otherwise "Alice_JP_Bob".
    """
    a, b = a.strip(), b.strip()
    init = (initials or "").strip()
    return f"{a}_{init}_{b}" if init else f"{a}_{b}"


@server.tool()
def verify_add_initials_between(expression: str, initials: str | None = None) -> bool:
    """Check whether add_initials_between(a, b, initials) equals the claimed result.

    Expected expression (quoted strings):
        "Alice" add_initials_between "Bob" = "Alice_JP_Bob"
        "Alice" add_initials_between "Bob" = "Alice_Bob"      # when initials omitted/blank
    Expected initials (optional; letters, optionally quoted):
        "JP"  |  JP  |  ""  |  omitted
    """
    import re

    # expression: two quoted inputs + claimed quoted result
    expr_m = re.match(
        r"""\s*(["'])(.*?)\1\s+add_initials_between\s+(["'])(.*?)\3\s*=\s*(["'])(.*?)\5\s*$""",
        expression,
    )
    if not expr_m:
        return False

    a, b, claimed = expr_m[2].strip(), expr_m[4].strip(), expr_m[6].strip()
    if not a or not b:
        return False

    # initials optional: None / blank are allowed; otherwise letters (optionally quoted)
    if initials is None or not str(initials).strip():
        init = ""
    else:
        init_m = re.match(r"""^\s*(?:(["'])([A-Za-z.]+)\1|([A-Za-z.]+))\s*$""", initials)
        if not init_m:
            return False
        init = (init_m[2] if init_m[1] else init_m[3]).strip()

    expected = f"{a}_{init}_{b}" if init else f"{a}_{b}"
    return expected == claimed

# ── RESOURCES ──  application-controlled, read-only context addressed by URI.
# Resource = where the host loads data into context (e.g. market://AAPL/quote).
# tools = “do something”; resources = “look this up” — not user-intent templates.

@server.resource("config://server")
def server_config() -> str:
    """A STATIC resource — always the same URI, returns fixed text."""
    return "name=prometheus\nversion=1.0\nmax_results=10"

@server.resource("note://{topic}")
def note(topic: str) -> str:
    """A resource TEMPLATE — the {topic} segment is filled in per request."""
    return f"This is the note about '{topic}'."


@server.resource("market://{symbol}/quote")
def quote(symbol: str) -> str:
    """Read-only quote for a market symbol (e.g. market://AAPL/quote).

    Resource role: host loads this URI into context (Redis-backed later).
    Pair with the get_market_data *prompt* for user intent; this handler only
    supplies data — it is not the briefing template.

    Never set cache=True or will break the FastMCP market://{symbol}/quote resource.
    Hosts: use prometheus.mcp.client.connect_client() (cache=False).
    """
    symbol = symbol.strip().upper()
    # TODO: redis.get(f"market:{symbol}:quote")
    return f"symbol={symbol}\nprice=stub\nsource=placeholder"


# ── PROMPTS ──  user-controlled reusable templates.
# Prompt = user intent (e.g. “What’s happening with AAPL?”).
# Prompt text must not name URIs or tools — that’s app/agent wiring, not the
# template’s job. Host loads market://AAPL/quote (resource) into context separately.

@server.prompt()
def review_code(code: str) -> str:
    """A prompt template the user can invoke (e.g. as a slash-command)."""
    return f"Please review the following code and suggest improvements:\n\n{code}"


@server.prompt()
def get_market_data(symbol: str) -> str:
    """Prompt = user intent (“What’s happening with AAPL?”), slash-command style.

    Returns briefing task text only. Does not name URIs or tools — app/agent
    wiring loads market://{symbol}/quote into context via the quote resource.
    """
    symbol = symbol.strip().upper()
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
