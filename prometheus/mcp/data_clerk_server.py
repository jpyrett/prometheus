"""The Data Clerk's MCP server: the same tools the agent uses, served over MCP (design choices entry 25).

    python -m prometheus.mcp.data_clerk_server                  # stdio; search and read tools only
    python -m prometheus.mcp.data_clerk_server --allow-fetch    # also the three fetch tools (PAID APIs, they WRITE files)
    python labs/module3_rag/other/mcp_check.py                        # connect to it, list the tools, call retrieve_news

The week 2 design says the Data Clerk "calls its own MCP (quotes, news, filings, RAG)". This is that server for the text tools. It does
NOT contain new logic. Each MCP tool is the bound method of `DataClerkTools` (prometheus/tools/data_clerk.py): the same function the
LangChain agent calls, with the same type hints, the same docstring and the same argument checks. FastMCP reads the name, the typed
arguments and the docstring and builds the MCP tool; LangChain builds its tool from the same three things. So there is one
implementation and two ways to reach it, and `tests/test_mcp_data_clerk.py` checks that they agree.

WHAT THE SERVER DOES NOT DO, ON PURPOSE
* It does not number sources. The agent that calls it owns the numbering: a number means "the nth source in THIS answer", and a
  stateless server cannot know that. `retrieve_news` returns the hits (chunk id, headline, publisher, date, url, tickers, text, score),
  and the caller wraps them in a `SourceBook` to cite them.
* It does not enforce the agent's per-question budget (3 searches, one fetch per symbol, no fetching after web text was read). The
  budget belongs to a run of an agent, and the server has no runs. A host that connects to this server needs its own.
* It does not offer the fetch tools unless it is started with --allow-fetch. A tool that is not offered cannot be called: any host
  that connects (a stray one, a compromised one) can search the stored news and nothing else. This is the same rule as the agent's
  `allow_fetch`, and for the same reason: a fetch uses paid APIs and writes the real news files.

The allow-list is `DATA_CLERK_TOOLS`. Never served: init_aws_cli, fetch_historical_data, convert_to_parquet, export_project, and the two
scheduled refresh jobs (design choices entry 14).
"""
import argparse
import logging
import sys

from fastmcp import FastMCP

from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools

logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

FETCH_TOOLS = ("fetch_news", "fetch_press_releases", "fetch_filings")

# MCP "annotations" are hints to the host: does the tool change anything, does it reach outside this machine.
READ_LOCAL = {"readOnlyHint": True, "openWorldHint": False}       # reads the stored text only
READ_REMOTE = {"readOnlyHint": True, "openWorldHint": True}       # calls an outside API, changes nothing here
WRITES = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True}  # downloads and adds files, deletes nothing
ANNOTATIONS = {"retrieve_news": READ_LOCAL, "retrieve_press_releases": READ_LOCAL, "retrieve_filings": READ_LOCAL, "fetch_filings_recommendation": READ_LOCAL, "keyword_search": READ_LOCAL, "discover_tickers": READ_LOCAL,
               "get_price_summary": READ_LOCAL,
               "get_financials": READ_REMOTE, "get_quote": READ_REMOTE, "fetch_news": WRITES, "fetch_press_releases": WRITES, "fetch_filings": WRITES}


def served_tools(allow_fetch: bool = False) -> tuple[str, ...]:
    """The names this server offers: the allow-list, minus the fetch tools unless they are asked for."""
    return tuple(n for n in DATA_CLERK_TOOLS if allow_fetch or n not in FETCH_TOOLS)


def build_server(tools: DataClerkTools | None = None, allow_fetch: bool = False) -> FastMCP:
    """A FastMCP server for the Data Clerk's tools. `tools` is the `DataClerkTools` to serve (tests pass one with a fake toolbox
    and a temporary store; by default it is the real one)."""
    tools = tools if tools is not None else DataClerkTools()
    server = FastMCP("prometheus-data-clerk", cache_ttl=None)
    for name in served_tools(allow_fetch):
        server.tool(getattr(tools, name), name=name, annotations=ANNOTATIONS[name])
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Data Clerk's tools over MCP (stdio).")
    parser.add_argument("--allow-fetch", action="store_true", help="also offer fetch_news, fetch_press_releases and fetch_filings "
                        "(they use the PAID APIs and WRITE the news files)")
    args = parser.parse_args(argv)
    if args.allow_fetch:
        print("WARNING: the fetch tools are on: connected hosts can use the paid APIs and write the news files.", file=sys.stderr)
    build_server(allow_fetch=args.allow_fetch).run(show_banner=False)  # stdio: nothing may be printed to stdout, the protocol uses it
    return 0


if __name__ == "__main__":
    sys.exit(main())
