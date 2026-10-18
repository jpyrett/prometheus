#!/usr/bin/env python3
"""Prometheus entrypoint — local LLM + FastMCP smoke run.

Usage (from repo root, cmu env):

    python main.py                  # elicit symbol (HITL), then LLM briefing
    python main.py --skip-llm       # elicit + MCP only (no HF weights)
    python main.py --symbol MSFT    # skip elicitation; use MSFT
    python main.py --symbol AAPL --thinking

Cursor / harness: run this file as the application process.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prometheus LLM + FastMCP host")
    p.add_argument(
        "--symbol",
        default=os.getenv("SMOKE_SYMBOL") or None,
        help="Skip elicitation and use this symbol (default: prompt via enter_symbol)",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip loading LocalLLM; exercise MCP only",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Pass enable_thinking=True to LocalLLM.chat",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation budget for the smoke briefing",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override LOCAL_MODEL / HF_MODEL (e.g. Qwen/Qwen3-4B)",
    )
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    from prometheus.application.host import PrometheusHost
    from prometheus.mcp.client import interactive_elicitation_handler

    llm = None
    if not args.skip_llm:
        from prometheus.models.llm import LocalLLM

        llm = LocalLLM(model_name=args.model)

    # Interactive HITL when no --symbol; tests keep auto elicitation_handler default.
    elicit = not bool(args.symbol)
    client_kwargs = {}
    if elicit:
        client_kwargs["elicitation_handler"] = interactive_elicitation_handler

    host = PrometheusHost(llm=llm, client_kwargs=client_kwargs)
    report = await host.run_smoke(
        symbol=args.symbol,
        elicit_symbol=elicit,
        enable_thinking=args.thinking,
        max_new_tokens=args.max_new_tokens,
        skip_llm=args.skip_llm,
    )

    print("\n--- smoke summary ---")
    print(f"symbol: {report.symbol} (elicited={report.elicited})")
    print(f"tool:   {report.tool_name} -> {report.tool_result!r}")
    print(f"prompt: {report.prompt_name} ({len(report.prompt_text)} chars)")
    print(f"quote:  {report.quote_text!r}")
    if report.llm_answer is not None:
        print(f"llm:\n{report.llm_answer}")
    else:
        print("llm:    (skipped)")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
