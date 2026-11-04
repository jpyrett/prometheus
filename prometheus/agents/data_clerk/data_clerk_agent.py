"""The Data Clerk: an LLM with tools and a loop (design choices entries 13, 15, 22, 23 and 24).

    from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
    result = DataClerk().ask("What did MKBHD say about Apple's foldable iPhone?")
    result.answer, result.references, result.problems

    python -m prometheus.agents.data_clerk.data_clerk_agent "What did MKBHD say about Apple's foldable iPhone?"

    DataClerk().request_data(work_order)     # the agent call the other agents make: a work order in, the same order out, with its data

NOT to be confused with `data_toolbox.py` in this folder, which is the Data Clerk's TOOLBOX (the DataToolbox: plain Python,
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

THE SWEEP (design choices entry 40). An order carries ALL THREE corpora (news, press releases, filings) and the filing evidence, whatever its `source` and whatever the model
chose to search. After the loop, CODE searches every corpus the model did not search (the same filters as the order, the stored text only, never a fetch) and numbers what it
finds after the model's sources in the same SourceBook. The model never sees those sources, so its answer does not cite them; they are flagged `swept` on the order and are for
the next agent. Code, not the model, because a model that is told to search three places searches one or two, and because it runs AFTER the model: a search before it would
count as "web text was read" and switch off the fetch tools (the budget's least-privilege rule). `DataClerk(sweep=False)` turns it off.

The result carries what the next agent needs: the answer, the cited chunks (`references`), every source the model was
shown, what went wrong, and what was used.
"""
import argparse
import json
import re
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from prometheus.common.work_order import DataResult, WorkOrder, attach_result, corpus_of, verify_quote_request, verify_request
from prometheus.models.chat import get_chat_model, is_rate_limited
from prometheus.prompts.data_clerk import LAST_TURN, repair_message, system_prompt
from prometheus.rag.citations import SourceBook
from prometheus.rag.retriever import Hit
from prometheus.tools import symbols
from prometheus.tools.data_clerk import EVIDENCE_TOOLS, RETRIEVE_TOOLS, DataClerkTools, ToolInputError, ToolLimitError, describe_tool_error

# one search at a time across all questions in this process: the embedding model and the Chroma index are shared
_RETRIEVE_LOCK = threading.Lock()

FETCH_TOOLS = ("fetch_news", "fetch_press_releases", "fetch_filings")
LOOKUP_SOURCES = ("press_releases", "filings")  # the sources whose symbol or company name is looked up in code, before any model runs (entries 34 and 35)
DEFAULT_CALLS = {"retrieve_news": 3, "retrieve_press_releases": 3, "retrieve_filings": 3, "fetch_filings_recommendation": 2, "keyword_search": 2, "discover_tickers": 2, "get_financials": 2, "get_price_summary": 2, "get_quote": 5,
                 "fetch_news": 1, "fetch_press_releases": 1, "fetch_filings": 1}
JUDGES = re.compile(r"\b(?:bullish|bearish)\b", re.I)  # the words of the Advisor's job; the Data Clerk's answer should not carry them (entry 41)
SWEEP_TOOLS = (("news", "retrieve_news"), ("press_releases", "retrieve_press_releases"), ("filings", "retrieve_filings"))  # the corpora every order carries (entry 40), and the search of each
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
        """Note that web text reached the conversation: a search that returned hits, a keyword search or a trend search that found something.
        A press release is text from the web like an article (anyone can get a line into one), so its hits count too, and so do a filing's (an 8-K can quote an
        exhibit or a release)."""
        with self._lock:
            if name in RETRIEVE_TOOLS:
                artifact = output[1] if isinstance(output, tuple) and len(output) == 2 else {}
                self.read_web_text |= bool(isinstance(artifact, dict) and artifact.get("hits"))
            elif name in EVIDENCE_TOOLS:  # quotes of filings: web text like any filing text (entry 38). The artifact holds the evidence
                artifact = output[1] if isinstance(output, tuple) and len(output) == 2 else {}
                self.read_web_text |= bool(isinstance(artifact, dict) and artifact.get("evidence"))
            elif name == "keyword_search":
                self.read_web_text |= bool(isinstance(output, dict) and (output.get("news") or output.get("filings")))
            elif name == "discover_tickers":
                self.read_web_text |= bool(isinstance(output, dict) and output.get("tickers"))


def budgeted(tool: StructuredTool, budget: Budget) -> StructuredTool:
    """The same tool (name, description, arguments, return format) behind the budget. Searches take the shared lock."""
    def run(**kwargs):
        budget.check(tool.name, kwargs)
        if tool.name in RETRIEVE_TOOLS:
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
    filing_evidence: dict = field(default_factory=dict)    # per symbol, the whole result of fetch_filings_recommendation when it was called and succeeded (entry 39), by the model or by the sweep
    sweep: dict = field(default_factory=dict)              # per corpus (news, press_releases, filings, filing_evidence): who searched it ("model" or "code"), how many sources it gave, and why none (entry 40)
    error: str = ""             # the kind of model or network failure, if the call failed ("RateLimitError", ...)
    rate_limited: bool = False  # the failure was a provider's rate limit, not the model's doing

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


class DataClerk:
    def __init__(self, toolbox=None, store=None, model=None, model_id: str | None = None, allow_fetch: bool = False,
                 max_turns: int = MAX_TURNS, calls: dict[str, int] | None = None, press_store=None, symbol_search=None, filings_store=None, sweep: bool = True):
        """toolbox: the DataToolbox (a fake in tests). store: the Chroma store of the news, press_store the one of the press releases and filings_store the one of
        the SEC filings (the real ones when left out). model: a chat model (anything with `bind_tools` and `invoke`; a fake in tests); otherwise OpenRouter
        through `get_chat_model` with the model of the `data_clerk` role, or `model_id`. allow_fetch: let the model use the fetch tools, which
        call paid APIs and write files; off unless you say so. symbol_search: the symbol directory a press release or filings order is checked against
        (Finnhub's search; a fake in tests, see prometheus/tools/symbols.py). sweep: after the model's loop, search the corpora it did not search and read the stored
        filing evidence, so that an order carries all three sources (entry 40); `request_data` only, `ask` never sweeps."""
        if max_turns < 2:
            raise ValueError("max_turns must be at least 2 (one turn to ask for a tool, one to answer)")
        self.tools = DataClerkTools(toolbox, store, press_store, filings_store)
        self.model = model or get_chat_model("data_clerk", model=model_id)
        self.allow_fetch, self.max_turns, self.calls, self.symbol_search, self.sweep = allow_fetch, max_turns, calls, symbol_search, sweep

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

    def ask(self, question: str, today: date | None = None, sweep: dict | None = None) -> Result:
        """Answer one question. Never raises for a model or API failure: it comes back in `problems`. `sweep` (`_sweep_spec` of an order) makes code search the corpora the model
        did not search, once the answer is done (entry 40); without it nothing is added."""
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
        if sweep is not None:
            self._sweep(result, book, sweep)
        return result

    @staticmethod
    def _sweep_spec(order: WorkOrder) -> dict:
        """What the sweep searches with: the order's own query (with the company's name in front when the order has one and the query does not say it: a query without the name scores
        under the relevance bar, found on the first press release run), its window and its `top_k`. The symbols are the order's when it has a list of them; a subject that is a TERM
        ("flying car") filters nothing: it is searched across every company."""
        query = order.query.strip()
        if order.company and order.company.lower() not in query.lower():
            query = f"{order.company}: {query}"
        symbols = [s for s in order.subject if isinstance(s, str) and s.strip()] if isinstance(order.subject, list) else []
        return {"query": query, "tickers": symbols or None, "since_days": order.since_days, "k": order.top_k}

    def _sweep(self, result: Result, book: SourceBook, spec: dict) -> None:
        """Search every corpus the model did not search (a search that came back `ok`, even with nothing found, counts as searched) and read the stored filing evidence of the order's
        symbol if the model did not. What is found is numbered after the model's sources in the same book and flagged `swept`. A failing search is a note in `result.sweep`, never
        an exception and never a problem: the order's answer does not depend on it. Stored text only: nothing is fetched and nothing is paid for."""
        done = {c["name"] for c in result.tool_calls if c["outcome"] == "ok"}
        for corpus, tool in SWEEP_TOOLS:
            if tool in done:
                result.sweep[corpus] = {"by": "model", "sources": sum(corpus_of(s["chunk_id"]) == corpus for s in result.sources), "note": ""}
                continue
            try:
                with _RETRIEVE_LOCK:
                    found = getattr(self.tools, tool)(spec["query"], tickers=spec["tickers"], since_days=spec["since_days"], k=spec["k"])
            except Exception as error:  # a bad filter or a store that cannot be read: say the kind, never the details
                result.sweep[corpus] = {"by": "code", "sources": 0, "note": f"the search failed ({type(error).__name__})"}
                continue
            hits = [Hit(**h) for h in found["hits"]]
            if hits:
                book.add(hits)
                for hit in hits:
                    result.sources.append({"number": book.number_of(hit), **hit.as_dict(), "swept": True})
            result.sweep[corpus] = {"by": "code", "sources": len(hits), "note": found.get("message") or ""}
        symbols = spec["tickers"] or []
        if result.filing_evidence:
            result.sweep["filing_evidence"] = {"by": "model", "sources": sum(len(e.get("evidence", [])) for e in result.filing_evidence.values()), "note": ""}
        elif len(symbols) != 1:
            result.sweep["filing_evidence"] = {"by": "code", "sources": 0, "note": "the filing evidence is read for one symbol: the order does not name exactly one"}
        else:
            try:
                evidence = self.tools.fetch_filings_recommendation(symbols[0])
            except ToolInputError as error:  # for example "no filings of X are stored": the reason is the message
                result.sweep["filing_evidence"] = {"by": "code", "sources": 0, "note": str(error)[:200]}
            except Exception as error:
                result.sweep["filing_evidence"] = {"by": "code", "sources": 0, "note": f"reading the filings failed ({type(error).__name__})"}
            else:
                result.filing_evidence[evidence["symbol"]] = evidence
                result.sweep["filing_evidence"] = {"by": "code", "sources": len(evidence["evidence"]), "note": ""}

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
        if order.source in LOOKUP_SOURCES and not self._resolve_target(order):
            return order  # the symbol or company name was not found: refused in code, before any model call and before anything is fetched
        return attach_result(order, self.ask(self._question(order), today, sweep=self._sweep_spec(order) if self.sweep else None))

    def request_quotes(self, order: WorkOrder) -> WorkOrder:
        """The Data Clerk's agent call for a `fetch_quote` order (entry 42): the current price of every symbol in `order.subject`, put on `order.data.quotes`, and the SAME order returned.

        There is no model call: a price is one lookup, so the Data Clerk calls its own `get_quote` tool (`DataClerkTools.get_quote`, the tool the model is also offered and the MCP
        server serves) once per symbol, in code. A symbol that fails (not a ticker, unknown to Finnhub, the API down) is named in `order.errors` and the order is `data_failed`;
        the quotes that did come back stay on the order, so a caller can see what it has. An order that already has its data comes back unchanged."""
        if order.status in ("data_ready", "recommended"):
            return order
        reasons = verify_quote_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        wanted = [order.subject] if isinstance(order.subject, str) else list(order.subject)
        quotes, calls, problems = {}, [], []
        for raw in dict.fromkeys(s.strip().lstrip("$").upper() for s in wanted):
            try:
                quotes[raw] = self.tools.get_quote(raw)
                calls.append({"name": "get_quote", "args": {"symbol": raw}, "outcome": "ok"})
            except Exception as error:  # a bad ticker, an unknown symbol, a missing key or an API failure: say which symbol and what kind, never request details
                calls.append({"name": "get_quote", "args": {"symbol": raw}, "outcome": "error"})
                problems.append(f"{raw}: no quote ({type(error).__name__}: {' '.join(str(error).split())[:160]})")
        order.data = DataResult(answer=", ".join(f"{s} {q['price']:.2f}" for s, q in quotes.items()), quotes=quotes, problems=problems, tool_calls=calls,
                                completed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
        order.errors = [*order.errors, *problems]
        order.status = "data_failed" if problems else "data_ready"
        return order

    def _resolve_target(self, order: WorkOrder) -> bool:
        """Press releases and filings (the same rules, entries 34 and 35): turn the order's two arguments (subject = the symbol, company = the company name) into ONE symbol, by the rules of
        `symbols.resolve_target` (the symbol first, else the company name, exact matches only). On success the order says what was decided
        (`resolved`) and carries the resolved symbol and company name, so that the question, the fetch and the report all use them. When nothing is
        found, the order is `data_failed` and `errors` holds the message ("Symbol: X is not found" or "Company Name not found."). Returns whether
        the order may go on. A lookup that FAILS (rate limit, no key) is reported as a failure, never as "not found"."""
        symbol = order.subject if isinstance(order.subject, str) else next(iter(order.subject or []), None)
        try:
            found = symbols.resolve_target(symbol, order.company, self.symbol_search)
        except symbols.TargetNotFound as error:
            order.status, order.errors = "data_failed", [*order.errors, str(error), *([error.detail] if error.detail else [])]
            return False
        except (symbols.SymbolLookupError, ValueError) as error:
            order.status, order.errors = "data_failed", [*order.errors, f"the request could not be checked: {getattr(error, 'reason', error)}"]
            return False
        order.resolved = {"typed_symbol": symbol, "typed_company": order.company, "symbol": found.symbol, "company": found.company, "how": found.how}
        order.subject, order.company = [found.symbol], found.company
        return True

    @staticmethod
    def _question(order: WorkOrder) -> str:
        """The work order's query, with the order's own limits added in words: the subject, the time window and how many sources."""
        if order.source == "press_releases":
            return DataClerk._press_release_question(order)
        if order.source == "filings":
            return DataClerk._filings_question(order)
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
    def _press_release_question(order: WorkOrder) -> str:
        """The same, for a press release order: the resolved symbol and company, the window, and which tools to use. The symbol is the one the
        lookup found (`order.subject`), never what the user typed."""
        return DataClerk._lookup_question(order, "press releases", "PRESS RELEASES", "retrieve_press_releases", "fetch_press_releases")

    @staticmethod
    def _filings_question(order: WorkOrder) -> str:
        """The same, for a filings order (entry 35): the press release question with the filing tools and words."""
        return DataClerk._lookup_question(order, "filings", "SEC FILINGS", "retrieve_filings", "fetch_filings")

    @staticmethod
    def _lookup_question(order: WorkOrder, noun: str, shouted: str, search_tool: str, fetch_tool: str) -> str:
        """The question for an order whose symbol was resolved in code (press releases, filings): the resolved symbol and company, the window, which tools to use."""
        symbol = order.subject[0] if isinstance(order.subject, list) else order.subject
        hints = [f"subject: {symbol}" + (f" ({order.company})" if order.company else "")]
        if order.since_days:
            hints.append(f"use {noun} from the last {order.since_days} days")
        hints.append(f"use at most {order.top_k} sources")
        hints.append(f"this is a request about the company's {shouted}: call {search_tool} (not retrieve_news) with "
                     f"tickers=[\"{symbol}\"] before you answer, whatever the question is")
        if order.company:  # found by the first real run: a query like "executive changes" scored 0.23 (under the 0.40 bar) and "Apple executive changes" 0.51
            hints.append(f"write the search query with the company's name \"{order.company}\" and the topic: a query without the name scores under the relevance bar "
                         f"and finds nothing, and then \"the {'releases' if noun == 'press releases' else noun} do not cover it\" would be a wrong answer")
        if order.live:
            window = f", with from_date={order.since_days}" if order.since_days else ""
            hints.append(f"fresh {noun} are required: call {fetch_tool} once for {symbol} first{window}, then search")
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
                if m.name in EVIDENCE_TOOLS and m.status != "error" and isinstance(m.artifact, dict) and m.artifact.get("symbol"):
                    result.filing_evidence[m.artifact["symbol"]] = m.artifact  # the whole evidence, for the next agent (a failed or refused call has no artifact)
        for m in replies:
            for call in m.tool_calls or []:
                result.tool_calls.append({"name": call["name"], "args": call["args"], "outcome": outcome.get(call["id"], "ok")})
        check = book.check(result.answer)
        result.references = check.references()
        result.invalid_citations, result.has_citations = list(check.invalid), check.has_citations
        searched = any(c["name"] in RETRIEVE_TOOLS for c in result.tool_calls)
        result.no_supporting_text = searched and len(book) == 0
        if not result.answer:
            result.problems.append("the answer is empty")
        if check.invalid:
            result.problems.append("cites " + ", ".join(f"[{n}]" for n in check.invalid) + ", which are not among the sources" +
                                   (" (corrected once, still wrong)" if result.repaired else ""))
        if len(book) and result.answer and not check.has_citations:
            result.notes.append(f"cites nothing although {len(book)} sources were retrieved (an honest 'the sources do not say' does)")
        if any(JUDGES.search(sentence) and "advisor" not in sentence.lower() for sentence in re.split(r"(?<=[.!?])\s+|\n", result.answer)):  # a sentence that sends the judging to the Advisor is what the prompt asks for  # the Data Clerk does not judge (entry 41): a note for the reader, not a problem (a source may itself say "bullish")
            result.notes.append("the answer uses a judging word (bullish or bearish): the Data Clerk should only report what the sources say")
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
