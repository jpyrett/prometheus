"""Part 4 — Connect a Client (host-side). Does not run the MCP server.

server.py must not import this module (avoids mixing stdio server with host code).

    from prometheus.mcp.client import connect_client
    async with connect_client() as c:
        ...

Defaults:
    mode=\"legacy\" — sampling / roots / elicitation (in-process or stdio host)
    cache=False — Never set cache=True or will break market://{symbol}/quote
"""

from pathlib import Path

import mcp.types as types
from fastmcp import Client

from prometheus.mcp.server import server

CLIENT_MODE = "legacy"
CLIENT_CACHE = False  # Never set cache=True or will break market://{symbol}/quote

_CAPSTONE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = [
    types.Root(uri=_CAPSTONE_ROOT.as_uri(), name="capstone"),
]


async def sampling_handler(messages, params, context):
    """SAMPLING: the server asked our 'model' to generate. Return a canned reply."""
    last = messages[-1].content.text
    return f"[mock completion for: {last[:50]}]"


async def elicitation_handler(message, response_type, params, context):
    """ELICITATION: the server asked the user a question. Auto-answer it."""
    return {"favorite_number": 24}


async def log_handler(params):
    """LOGGING: receive log records the server emits."""
    data = params.data
    msg = data.get("msg", data) if isinstance(data, dict) else data
    print(f"   [server log · {params.level}] {msg}")


def connect_client(**kwargs):
    """In-process Client to prometheus server: mode=legacy, cache=False, mock handlers."""
    kwargs.setdefault("mode", CLIENT_MODE)
    kwargs.setdefault("cache", CLIENT_CACHE)
    kwargs.setdefault("sampling_handler", sampling_handler)
    kwargs.setdefault("elicitation_handler", elicitation_handler)
    kwargs.setdefault("log_handler", log_handler)
    kwargs.setdefault("roots", DEFAULT_ROOTS)
    return Client(server, **kwargs)


# Notebook-style alias
client = connect_client
