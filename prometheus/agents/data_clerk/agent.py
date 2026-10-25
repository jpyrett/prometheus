"""The Data Clerk: an LLM with tools and a loop (design choices entries 13, 15, 22, 23 and 24).

    from prometheus.agents.data_clerk.agent import DataClerk
    result = DataClerk().ask("What did MKBHD say about Apple's foldable iPhone?")
    result.answer, result.references, result.problems

    python -m prometheus.agents.data_clerk.agent "What did MKBHD say about Apple's foldable iPhone?"

    DataClerk().request_data(work_order)     # the agent call the other agents make: a work order in, the same order out, with its data

NOT to be confused with `data_agent.py` in this folder, which is the Data Clerk's TOOLBOX (the DataAgent: plain Python,
no LLM, no loop). This file is the agent. It receives its toolbox, so a test can hand it a fake one.

THE LOOP (a LangGraph graph, built fresh for every question so that each has its own SourceBook and budget):

    agent  --tool calls-->  tools  --> agent  ...  --answer-->  verify  --valid-->  end
                                                                   |
                                                       invalid citation, first time
                                                                   +--> one correction --> agent

* `agent` asks the chat model, with the system prompt (prompts/data_clerk.py). On the LAST allowed turn it asks the
  model WITHOUT tools and tells it to answer with what it has, so a loop can never end in an empty answer.
* `tools` is a ToolNode whose errors become short messages (`describe_tool_error`), so a failing tool does not end the run.
* every tool is wrapped by a BUDGET: so many searches, one fetch per symbol, and fetching is OFF unless you ask for it
  (`allow_fetch=True`), because a fetch uses the paid APIs and writes the real news files. And once news text from the
  web has been read in a question, fetching is no longer allowed in it (least privilege): a hidden instruction in an
  article must never be able to unlock a paid call, whatever the model does. Fetch before you read, not after.
* `verify` runs the citation check. A citation to a source that does not exist gets one correction; after that it is
  reported as a problem. An answer with no citation at all is a NOTE, not a problem (an honest "the sources do not say"
  cites nothing).

The result carries what the next agent needs: the answer, the cited chunks (`references`), every source the model was
shown, what went wrong, and what was used.
"""
import argparse
import json
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from prometheus.common.work_order import WorkOrder, attach_result, verify_request
from prometheus.models.chat import get_chat_model, is_rate_limited
from prometheus.prompts.data_clerk import LAST_TURN, repair_message, system_prompt
from prometheus.rag.citations import SourceBook
from prometheus.tools.data_clerk import DataClerkTools, ToolLimitError, describe_tool_error

# one search at a time across all questions in this process: the embedding model and the Chroma index are shared
_RETRIEVE_LOCK = threading.Lock()

FETCH_TOOLS = ("fetch_news", "fetch_press_releases", "fetch_filings")
DEFAULT_CALLS = {"retrieve_news": 3, "keyword_search": 2, "discover_tickers": 2, "get_financials": 2, "get_price_summary": 2,
                 "fetch_news": 1, "fetch_press_releases": 1, "fetch_filings": 1}
MAX_TURNS = 6  # model turns per question, including a correction; the last one is without tools


class Budget:
    """What one question may use. `check` raises ToolLimitError when a call is not allowed; it counts the attempt."""

    def __init__(self, calls: dict[str, int] | None = None, allow_fetch: bool = False):
        self.limits = {**DEFAULT_CALLS, **(calls or {})}
        self.allow_fetch = allow_fetch
        self.used: dict[str, int] = {}
        self.fetched: set[tuple[str, str]] = set()
        self.stopped: list[str] = []  # every call a limit stopped, for the result
        self.read_web_text = False    # news text from the web is now in the conversation
        self._lock = threading.Lock()

    def check(self, name: str, args: dict) -> None:
        with self._lock:
            try:
                if name in FETCH_TOOLS:
                    if not self.allow_fetch:
                        raise ToolLimitError("fetching is turned off for this question, so only the stored news can be searched")
                    if self.read_web_text:
                        raise ToolLimitError("fetching is not allowed after news text has been read in this question, because text from the web "
                                             "could be steering the request. Fetch before searching, or ask the user to request it directly")
                    key = (name, str(args.get("symbol", "")).strip().lstrip("$").upper())
                    if key in self.fetched:
                        raise ToolLimitError(f"{name} was already called for {key[1]} in this question")
                    self.fetched.add(key)
                self.used[name] = self.used.get(name, 0) + 1
                if self.used[name] > self.limits.get(name, 0):
                    raise ToolLimitError(f"{name} may be used at most {self.limits.get(name, 0)} times for a question and it already was")
            except ToolLimitError as error:
                self.stopped.append(f"{name}: {error}")
                raise


    def observe(self, name: str, output) -> None:
        """Note that web text reached the conversation: a search that returned hits, a keyword search or a trend search that found something."""
        with self._lock:
            if name == "retrieve_news":
                artifact = output[1] if isinstance(output, tuple) and len(output) == 2 else {}
                self.read_web_text |= bool(isinstance(artifact, dict) and artifact.get("hits"))
            elif name == "keyword_search":
                self.read_web_text |= bool(isinstance(output, dict) and (output.get("news") or output.get("filings")))
            elif name == "discover_tickers":
                self.read_web_text |= bool(isinstance(output, dict) and output.get("tickers"))


def budgeted(tool: StructuredTool, budget: Budget) -> StructuredTool:
    """The same tool (name, description, arguments, return format) behind the budget. Searches take the shared lock."""
    def run(**kwargs):
        budget.check(tool.name, kwargs)
        if tool.name == "retrieve_news":
            with _RETRIEVE_LOCK:
                output = tool.func(**kwargs)
        else:
            output = tool.func(**kwargs)
        budget.observe(tool.name, output)
        return output

    return StructuredTool(name=tool.name, description=tool.description, args_schema=tool.args_schema, func=run,
                          response_format=tool.response_format)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int
    repaired: bool
    forced: bool  # the last turn was asked without tools


@dataclass
class Result:
    """What one question produced."""
    question: str
    answer: str = ""
    references: list[dict] = field(default_factory=list)   # the chunks the answer cites: number, ids, headline, publisher, date, url, score
    sources: list[dict] = field(default_factory=list)      # every chunk the model was shown, with its number and text
    problems: list[str] = field(default_factory=list)      # things that make the answer unreliable: invalid citations, no answer, a failed call
    notes: list[str] = field(default_factory=list)         # things worth knowing: cited nothing, a limit stopped a call, the last turn was forced
    no_supporting_text: bool = False                       # it searched and no source cleared the relevance bar
    tool_calls: list[dict] = field(default_factory=list)   # name, args and outcome (ok, error, invalid, limit) of every call
    turns: int = 0
    repaired: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    invalid_citations: list[int] = field(default_factory=list)  # cited numbers that match no source
    has_citations: bool = False
    error: str = ""             # the kind of model or network failure, if the call failed ("RateLimitError", ...)
    rate_limited: bool = False  # the failure was a provider's rate limit, not the model's doing

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


class DataClerk:
    def __init__(self, toolbox=None, store=None, model=None, model_id: str | None = None, allow_fetch: bool = False,
                 max_turns: int = MAX_TURNS, calls: dict[str, int] | None = None):
        """toolbox: the DataAgent (a fake in tests). store: the Chroma store (the real one when left out). model: a chat
        model (anything with `bind_tools` and `invoke`; a fake in tests); otherwise OpenRouter through `get_chat_model`
        with the model of the `data_clerk` role, or `model_id`. allow_fetch: let the model use the fetch tools, which
        call paid APIs and write files; off unless you say so."""
        if max_turns < 2:
            raise ValueError("max_turns must be at least 2 (one turn to ask for a tool, one to answer)")
        self.tools = DataClerkTools(toolbox, store)
        self.model = model or get_chat_model("data_clerk", model=model_id)
        self.allow_fetch, self.max_turns, self.calls = allow_fetch, max_turns, calls

    def _graph(self, book: SourceBook, budget: Budget, today: date | None):
        tools = [budgeted(t, budget) for t in self.tools.langchain_tools(book)]
        bound = self.model.bind_tools(tools)
        prompt = system_prompt(today)
        max_turns = self.max_turns

        def agent(state: State):
            last = state["turns"] >= max_turns - 1
            messages = [SystemMessage(content=prompt)] + state["messages"]
            if last:  # no tools on the last turn: it has to answer
                reply = self.model.invoke(messages + [HumanMessage(content=LAST_TURN)])
            else:
                reply = bound.invoke(messages)
            return {"messages": [reply], "turns": state["turns"] + 1, "forced": last or state["forced"]}

        def after_agent(state: State):
            reply = state["messages"][-1]
            if getattr(reply, "tool_calls", None) and state["turns"] < max_turns:
                return "tools"
            return "verify"

        def verify(state: State):
            check = book.check(state["messages"][-1].content or "")
            if check.invalid and not state["repaired"] and state["turns"] < max_turns:
                return {"messages": [HumanMessage(content=repair_message(check.invalid, len(book)), additional_kwargs={"repair": True})],
                        "repaired": True}
            return {}

        def after_verify(state: State):
            last = state["messages"][-1]
            return "agent" if last.type == "human" and last.additional_kwargs.get("repair") else END

        graph = StateGraph(State)
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode(tools, handle_tool_errors=describe_tool_error))
        graph.add_node("verify", verify)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "verify": "verify"})
        graph.add_edge("tools", "agent")
        graph.add_conditional_edges("verify", after_verify, {"agent": "agent", END: END})
        return graph.compile()

    def ask(self, question: str, today: date | None = None) -> Result:
        """Answer one question. Never raises for a model or API failure: it comes back in `problems`."""
        book, budget = SourceBook(), Budget(self.calls, self.allow_fetch)
        result = Result(question=question)
        messages: list = []
        try:
            final = self._graph(book, budget, today).invoke(
                {"messages": [HumanMessage(content=question)], "turns": 0, "repaired": False, "forced": False},
                config={"recursion_limit": 6 * self.max_turns + 10})
            messages = final["messages"]
            result.turns, result.repaired, forced = final["turns"], final["repaired"], final["forced"]
        except Exception as error:  # a model or network failure: report the kind, never request details
            result.problems.append(f"the model call failed ({type(error).__name__})")
            result.error, result.rate_limited = type(error).__name__, is_rate_limited(error)
            return result
        self._fill(result, messages, book, budget, forced)
        return result

    def request_data(self, order: WorkOrder, today: date | None = None) -> WorkOrder:
        """The Data Clerk's agent call (the design's `DataClerk.request_data`): check the work order, answer its query, put the
        result on the order, check that, and return the SAME order. An order that already has its data comes back unchanged. A request the Data Clerk should not
        take is not an exception: the order comes back `data_failed` with the reasons in `errors`."""
        if order.status in ("data_ready", "recommended"):
            return order  # it already has its data: a repeated request changes nothing (and never damages a good order)
        reasons = verify_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        return attach_result(order, self.ask(self._question(order), today))

    @staticmethod
    def _question(order: WorkOrder) -> str:
        """The work order's query, with the order's own limits added in words: the subject, the time window and how many sources."""
        hints = []
        if order.subject:
            hints.append("subject: " + (order.subject if isinstance(order.subject, str) else ", ".join(order.subject)) + (f" ({order.company})" if order.company else ""))
        if order.since_days:
            hints.append(f"use news from the last {order.since_days} days")
        elif order.time_period:
            hints.append(f"time period: {order.time_period}")
        hints.append(f"use at most {order.top_k} sources")
        hints.append("this is a request to search the stored news: call retrieve_news before you answer, whatever the subject is")
        if order.live:
            window = f", with from_date={order.since_days}" if order.since_days else ""
            hints.append(f"fresh news is required: call fetch_news once for each ticker in the subject first{window}, then search")
        return f"{order.query.strip()}\n\n(Work order: {'; '.join(hints)}.)"

    @staticmethod
    def _fill(result: Result, messages: list, book: SourceBook, budget: Budget, forced: bool) -> None:
        replies = [m for m in messages if isinstance(m, AIMessage)]
        for m in replies:
            usage = m.usage_metadata or {}
            result.tokens_in += usage.get("input_tokens", 0)
            result.tokens_out += usage.get("output_tokens", 0)
        answers = [m for m in replies if not m.tool_calls]
        result.answer = (answers[-1].content or "").strip() if answers else ""
        outcome = {}
        for m in messages:
            if isinstance(m, ToolMessage):
                text = m.content if isinstance(m.content, str) else ""
                outcome[m.tool_call_id] = ("limit" if text.startswith("Limit reached") else "invalid" if text.startswith("Invalid arguments")
                                           else "error" if m.status == "error" else "ok")
                hits = (m.artifact or {}).get("hits", []) if isinstance(m.artifact, dict) else []
                for h in hits:
                    if not any(s["chunk_id"] == h["chunk_id"] for s in result.sources):
                        result.sources.append(h)
        for m in replies:
            for call in m.tool_calls or []:
                result.tool_calls.append({"name": call["name"], "args": call["args"], "outcome": outcome.get(call["id"], "ok")})
        check = book.check(result.answer)
        result.references = check.references()
        result.invalid_citations, result.has_citations = list(check.invalid), check.has_citations
        searched = any(c["name"] == "retrieve_news" for c in result.tool_calls)
        result.no_supporting_text = searched and len(book) == 0
        if not result.answer:
            result.problems.append("the answer is empty")
        if check.invalid:
            result.problems.append("cites " + ", ".join(f"[{n}]" for n in check.invalid) + ", which are not among the sources" +
                                   (" (corrected once, still wrong)" if result.repaired else ""))
        if len(book) and result.answer and not check.has_citations:
            result.notes.append(f"cites nothing although {len(book)} sources were retrieved (an honest 'the sources do not say' does)")
        if result.repaired and not check.invalid:
            result.notes.append("a citation to a source that does not exist was corrected once")
        if budget.stopped:
            result.notes.append(f"a limit stopped {len(budget.stopped)} tool call(s): " + "; ".join(sorted(set(budget.stopped)))[:240])
        if forced:
            result.notes.append("the last turn was without tools")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the Data Clerk a question.")
    parser.add_argument("question", help="the question, in words")
    parser.add_argument("--allow-fetch", action="store_true",
                        help="let the model fetch news, press releases and filings: this USES THE PAID APIs and WRITES the news files")
    parser.add_argument("--model", help="an OpenRouter model id instead of the data_clerk role's model")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS, help=f"model turns (default {MAX_TURNS}; the last is without tools)")
    parser.add_argument("--json", action="store_true", help="print the whole result as JSON")
    args = parser.parse_args(argv)
    try:
        clerk = DataClerk(allow_fetch=args.allow_fetch, model_id=args.model, max_turns=args.max_turns)
    except (RuntimeError, ValueError) as error:  # for example a missing key: the message names the setting, never a value
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    result = clerk.ask(args.question)
    if args.json:
        print(json.dumps(result.as_dict(), indent=1, default=str))
        return 0 if result.ok else 1
    print(f"Q: {result.question}\n")
    print(result.answer or "(no answer)")
    if result.references:
        print("\nCited sources:")
        for r in result.references:
            print(f"  [{r['number']}] {r['publisher']}, {r['published_utc'][:10]}: {r['headline'][:70]}  (score {r['score']:.2f})\n      {r['url']}")
    print(f"\nTools: " + (", ".join(f"{c['name']}({c['outcome']})" for c in result.tool_calls) or "none") +
          f" | turns {result.turns} | tokens {result.tokens_in} in, {result.tokens_out} out | sources shown {len(result.sources)}")
    for note in result.notes:
        print(f"note: {note}")
    for problem in result.problems:
        print(f"PROBLEM: {problem}")
    if result.no_supporting_text:
        print("no supporting text: the search found nothing that clears the relevance bar")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
