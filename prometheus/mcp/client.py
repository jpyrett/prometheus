"""Part 4 — Connect a Client (host-side). Does not run the MCP server.

server.py must not import this module (avoids mixing stdio server with host code).

    from prometheus.mcp.client import connect_client
    async with connect_client() as c:
        ...

Defaults:
    mode=\"legacy\" — needed for elicitation demos (pushed form)
    cache=False — Never set cache=True or will break market://{symbol}/quote

Not configured (deprecated / wrong layer for Prometheus):
    sampling — use LangGraph / OpenRouter / local LLM on the host instead
    roots — use tool args, resource URIs (e.g. market://{{symbol}}/quote), or config
"""

from fastmcp import Client

from prometheus.mcp.server import server

CLIENT_MODE = "legacy"
CLIENT_CACHE = False  # Never set cache=True or will break market://{symbol}/quote


async def elicitation_handler(message, response_type, params, context):
    """ELICITATION: server asked the user a question. Auto-answer for demos/tests.

    Real host: show `message` in UI and collect structured input matching response_type.
    """
    # Matches server pick_number NumberChoice (notebook Part 5 demo).
    return {"favorite_number": 24}


async def log_handler(params):
    """LOGGING: receive log records the server emits."""
    data = params.data
    msg = data.get("msg", data) if isinstance(data, dict) else data
    print(f"   [server log · {params.level}] {msg}")


def connect_client(**kwargs):
    """In-process Client: mode=legacy, cache=False, elicitation + logging handlers."""
    kwargs.setdefault("mode", CLIENT_MODE)
    kwargs.setdefault("cache", CLIENT_CACHE)
    kwargs.setdefault("elicitation_handler", elicitation_handler)
    kwargs.setdefault("log_handler", log_handler)
    return Client(server, **kwargs)


# Notebook-style alias
client = connect_client
