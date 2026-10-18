"""Run prometheus/mcp/server.py when you want to serve over stdio."""

import logging
from pydantic import BaseModel
import mcp.types as types
from fastmcp import FastMCP, Context, Client

# Quiet noisy SDK / framework logs for a cleaner notebook.
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

server = FastMCP("prometheus")

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




if __name__ == "__main__":
    server.run()  # stdio — client/host starts this process

