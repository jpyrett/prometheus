"""Talk to the Data Clerk's MCP server the way another host would (Part 8 of the RAG guide).

Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/other/mcp_check.py                       # starts the real server as a subprocess and talks to it over stdio
    python labs/module3_rag/other/mcp_check.py "what did MKBHD say about the foldable iPhone" --tickers AAPL

What it does, in order:
  1. starts `python -m prometheus.mcp.data_clerk_server` (stdio, fetching OFF) and lists the tools it offers, with their hints
  2. calls `retrieve_news` over MCP on the real index and prints the hits
  3. wraps the hits in a SourceBook, the way an agent does, to show the numbered text a model would read
  4. calls the same method directly and checks that the MCP answer is identical
  5. tries `fetch_news`, which a server without --allow-fetch must not offer: it is refused and nothing is fetched

Nothing is fetched and no file is written. No model is called, so it costs nothing and needs no key.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # so "import prometheus" works from any folder

from fastmcp import Client  # noqa: E402
from fastmcp.client.transports import StdioTransport  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

from prometheus.rag.config import REPO_ROOT  # noqa: E402
from prometheus.rag.citations import SourceBook  # noqa: E402
from prometheus.rag.retriever import Hit  # noqa: E402
from prometheus.tools.data_clerk import DataClerkTools  # noqa: E402


async def run(question: str, tickers: str | None) -> int:
    transport = StdioTransport(command=sys.executable, args=["-m", "prometheus.mcp.data_clerk_server"], cwd=str(REPO_ROOT))
    args = {"query": question, **({"tickers": tickers} if tickers else {}), "k": 3}
    async with Client(transport, timeout=120) as client:
        tools = await client.list_tools()
        print(f"1. the server offers {len(tools)} tools (fetching is off, so no fetch tool):")
        for t in tools:
            a = t.annotations
            kind = "reads stored text" if a.read_only_hint and not a.open_world_hint else "reads an outside API" if a.read_only_hint else "WRITES files"
            print(f"     {t.name:<18} {kind}")
        result = (await client.call_tool("retrieve_news", args)).structured_content
        print(f"\n2. retrieve_news({json.dumps(args)}) over MCP: found={result['found']}, {len(result['hits'])} hits")
        for h in result["hits"]:
            print(f"     {h['score']:.2f}  {h['publisher']}  {h['published_utc'][:10]}  {h['headline'][:70]}  [{', '.join(h['tickers'][:3])}]")
        if result["hits"]:
            numbered = SourceBook().add([Hit(**h) for h in result["hits"]])
            print("\n3. what a model would read once an agent wraps the hits in a SourceBook (first 420 characters):\n")
            print("     " + numbered[:420].replace("\n", "\n     ") + " ...")
        else:
            print(f"\n   {result['message']}")
        direct = json.loads(json.dumps(DataClerkTools().retrieve_news(**args), default=str))
        same = direct == result
        print(f"\n4. the same call made directly on the method: identical to the MCP answer: {same}")
        try:
            await client.call_tool("fetch_news", {"symbol": "AAPL"})
            print("\n5. fetch_news: it ran (UNEXPECTED: the server was started without --allow-fetch)")
            return 1
        except ToolError as error:
            print(f"\n5. fetch_news over MCP: refused ({str(error)[:90]}). Nothing was fetched.")
        return 0 if same else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", default="what did MKBHD say about the foldable iPhone")
    parser.add_argument("--tickers", default="AAPL", help="a ticker symbol to filter on (default AAPL; '' for none)")
    args = parser.parse_args()
    return asyncio.run(run(args.question, args.tickers or None))


if __name__ == "__main__":
    sys.exit(main())
