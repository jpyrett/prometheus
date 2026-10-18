"""Prometheus application host — local LLM + in-process FastMCP Client.

Owns the conversation loop (lab pattern). MCP Sampling is not used; the host
calls LocalLLM, then FastMCP tools / resources / prompts as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prometheus.mcp.client import connect_client
from prometheus.models.llm import LocalLLM


@dataclass
class SmokeReport:
    symbol: str
    tool_name: str
    tool_result: object
    prompt_name: str
    prompt_text: str
    quote_text: str
    llm_answer: str | None
    elicited: bool = False


class PrometheusHost:
    """Minimal LLM application that exercises FastMCP tools and prompts."""

    def __init__(
        self,
        llm: LocalLLM | None = None,
        *,
        client_kwargs: dict[str, Any] | None = None,
    ):
        self.llm = llm
        self.client_kwargs = client_kwargs or {}

    def _client(self):
        return connect_client(**self.client_kwargs)

    async def call_tool(self, name: str, arguments: dict | None = None):
        async with self._client() as client:
            result = await client.call_tool(name, arguments or {})
            return result.data

    async def get_prompt(self, name: str, arguments: dict | None = None) -> str:
        async with self._client() as client:
            got = await client.get_prompt(name, arguments or {})
            return got.messages[0].content.text

    async def read_resource(self, uri: str) -> str:
        async with self._client() as client:
            contents = await client.read_resource(uri)
            return contents[0].text

    async def list_catalog(self) -> dict[str, list[str]]:
        async with self._client() as client:
            tools = await client.list_tools()
            prompts = await client.list_prompts()
            resources = await client.list_resources()
            templates = await client.list_resource_templates()
            return {
                "tools": [t.name for t in tools],
                "prompts": [p.name for p in prompts],
                "resources": [r.uri for r in resources],
                "templates": [t.uri_template for t in templates],
            }

    async def run_smoke(
        self,
        *,
        symbol: str | None = None,
        elicit_symbol: bool = True,
        enable_thinking: bool = False,
        max_new_tokens: int = 256,
        skip_llm: bool = False,
    ) -> SmokeReport:
        """End-to-end: optional enter_symbol HITL → fetch_quote → prompt → LLM."""
        tool_name = "fetch_quote"
        prompt_name = "get_market_data"
        elicited = False

        async with self._client() as client:
            tools = await client.list_tools()
            prompts = await client.list_prompts()
            resources = await client.list_resources()
            templates = await client.list_resource_templates()
            print("MCP catalog:")
            print(f"  tools: {[t.name for t in tools]}")
            print(f"  prompts: {[p.name for p in prompts]}")
            print(f"  resources: {[r.uri for r in resources]}")
            print(f"  templates: {[t.uri_template for t in templates]}")

            if elicit_symbol or not symbol:
                print("\nCalling enter_symbol (MCP elicitation) ...")
                entered = (await client.call_tool("enter_symbol", {})).data
                symbol = str(entered).strip().upper()
                elicited = True
                print(f"enter_symbol -> {symbol!r}")
            else:
                symbol = symbol.strip().upper()
                print(f"\nUsing --symbol {symbol!r} (elicitation skipped)")

            quote_uri = f"market://{symbol}/quote"

            tool_result = (
                await client.call_tool(tool_name, {"symbol": symbol})
            ).data
            print(f"\nTool {tool_name}({symbol!r}) ->\n{tool_result}")

            claimed = str(tool_result).replace("\n", "\\n")
            verified = (
                await client.call_tool(
                    "verify_fetch_quote",
                    {"expression": f'fetch_quote("{symbol}") = "{claimed}"'},
                )
            ).data
            print(f"verify_fetch_quote -> {verified!r}")

            got = await client.get_prompt(prompt_name, {"symbol": symbol})
            prompt_text = got.messages[0].content.text
            print(f"\nPrompt {prompt_name}:\n{prompt_text}\n")

            quote_text = (await client.read_resource(quote_uri))[0].text
            print(f"Resource {quote_uri}:\n{quote_text}\n")
            assert quote_text == tool_result, "tool and resource quote payloads must match"

        llm_answer: str | None = None
        if not skip_llm:
            if self.llm is None:
                raise RuntimeError("LocalLLM required unless skip_llm=True")
            full_prompt = (
                f"{prompt_text}\n\n"
                f"Quote data currently in context:\n{quote_text}\n"
            )
            print("Calling LocalLLM with prompt + quote context ...")
            result = self.llm.chat(
                full_prompt,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
            LocalLLM.show(result)
            llm_answer = result.answer

        return SmokeReport(
            symbol=symbol,
            tool_name=tool_name,
            tool_result=tool_result,
            prompt_name=prompt_name,
            prompt_text=prompt_text,
            quote_text=quote_text,
            llm_answer=llm_answer,
            elicited=elicited,
        )
