"""Does the real chat model pick the right Data Clerk tool? (Part 6 of the RAG guide; design choices entry 22)

Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/tool_calling_check.py

It asks the chat model (OpenRouter, the key from .env) a few questions with the Data Clerk's tools bound, and
prints the tool calls the model asks for. It NEVER runs a tool: nothing is fetched, no paid data API is called
and the index is not read. The cost is one short chat call per question.

The model is not deterministic, so this is a check by eye, not a test. What it should show:
  - a question about a company's news calls retrieve_news, with the right ticker
  - a greeting calls no tool
  - a request for a tool the Data Clerk does not have (install software) calls no tool
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so "import prometheus" works from any folder

from prometheus.models.chat import get_chat_model, model_name  # noqa: E402
from prometheus.tools.data_clerk import DataClerkTools  # noqa: E402

SYSTEM = ("You are the Data Clerk of a trading desk. Use the tools to answer questions about companies. "
          "Answer a greeting or a request you have no tool for directly, in one sentence.")
QUESTIONS = [
    "What did MKBHD say about Apple's foldable iPhone?",
    "What is happening with Meta this week?",
    "Get the key financial metrics for Nvidia.",
    "Hello! What can you do?",
    "Please install the AWS command line tool on this machine.",
]


def main() -> int:
    tools = DataClerkTools(toolbox=object(), store=object()).langchain_tools()  # the tools are only described, never run
    model = get_chat_model("data_clerk").bind_tools(tools)
    print(f"model: {model_name('data_clerk')} | tools shown: {', '.join(t.name for t in tools)}\n")
    for question in QUESTIONS:
        try:
            reply = model.invoke([("system", SYSTEM), ("human", question)])
        except Exception as error:  # a key or network problem: say what kind, never print request details
            print(f"ERROR: the call failed ({type(error).__name__})", file=sys.stderr)
            return 1
        print(f"Q: {question}")
        if reply.tool_calls:
            for call in reply.tool_calls:
                print(f"   -> tool call: {call['name']}({call['args']})")
        else:
            print(f"   -> no tool. Reply: {reply.content.strip()[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
