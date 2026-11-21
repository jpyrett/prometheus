"""The Data Clerk: an LLM with tools and a loop (design choices entries 13, 15, 22, 23 and 24).

    from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
    result = DataClerk().ask("What did MKBHD say about Apple's foldable iPhone?")
    result.answer, result.references, result.problems

    python -m prometheus.agents.data_clerk.data_clerk_agent "What did MKBHD say about Apple's foldable iPhone?"

    DataClerk().request_data(work_order)     # the agent call the other agents make: a work order in, the same order out, with its data
    DataClerk().request_bars(work_order)     # a `fetch_bars` order (Trader v1, asked by the Account Manager): the stored daily bars of a date range, no model call

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
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from prometheus.common.trading_clock import NEVER_FETCHES, bound_day, market_day, use_live_trading
from prometheus.common.trace import traced
from prometheus.common.work_order import HAS_DATA, DataResult, WorkOrder, attach_result, corpus_of, iso_date, sector_map, verify_bar_request, verify_metric_request, verify_quote_request, verify_request, verify_symbol_request, verify_verification_request
from prometheus.models.chat import get_chat_model, is_rate_limited
from prometheus.prompts.data_clerk import COMPANIES_RETRY, LAST_TURN, THEME_MEMBERS_RETRY, companies_prompt, repair_message, system_prompt, theme_members_prompt, theme_members_question
from prometheus.rag.citations import SourceBook
from prometheus.rag.loaders import check_symbol
from prometheus.rag.retriever import Hit
from prometheus.tools import symbols
from prometheus.tools.data_clerk import EVIDENCE_TOOLS, RETRIEVE_TOOLS, DataClerkTools, ToolInputError, ToolLimitError, describe_tool_error

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# one search at a time across all questions in this process: the embedding model and the Chroma index are shared
_RETRIEVE_LOCK = threading.Lock()

FETCH_TOOLS = ("fetch_news", "fetch_press_releases", "fetch_filings")
LOOKUP_SOURCES = ("press_releases", "filings")  # the sources whose symbol or company name is looked up in code, before any model runs (entries 34 and 35)
DEFAULT_CALLS = {"retrieve_news": 3, "retrieve_press_releases": 3, "retrieve_filings": 3, "fetch_filings_recommendation": 2, "keyword_search": 2, "discover_tickers": 2, "get_financials": 2, "get_price_summary": 2, "get_quote": 5, "calculate_metrics": 2,
                 "fetch_news": 1, "fetch_press_releases": 1, "fetch_filings": 1}
JUDGES = re.compile(r"\b(?:bullish|bearish)\b", re.I)  # the words of the Advisor's job; the Data Clerk's answer should not carry them (entry 41)
SWEEP_TOOLS = (("news", "retrieve_news"), ("press_releases", "retrieve_press_releases"), ("filings", "retrieve_filings"))  # the corpora every order carries (entry 40), and the search of each
MAX_REFRESH_SYMBOLS = 15  # `refresh_data` fetches for at most this many symbols at a time
MAX_EVIDENCE_SYMBOLS = 10  # the sweep reads the stored filing evidence of an order's symbols when it names at most this many
NEWS_ADDED = "stored market news, confirmed by the Data Clerk's model"   # the `how` of a company the stored market news tied to a theme and the model confirmed belongs to it (entry 120)
NOT_LISTED = "privately held, or not known to be listed"   # the Data Clerk's words for a name whose listed owner its model could not give (entry 101)
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
    security_dropped: list[dict] = field(default_factory=list)   # the documents the security layer dropped before anything read them ({id, corpus, action, risk, rules}; entry 116)
    error: str = ""             # the kind of model or network failure, if the call failed ("RateLimitError", ...)
    rate_limited: bool = False  # the failure was a provider's rate limit, not the model's doing

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


class DataClerk:
    def __init__(self, toolbox=None, store=None, model=None, model_id: str | None = None, allow_fetch: bool = False,
                 max_turns: int = MAX_TURNS, calls: dict[str, int] | None = None, press_store=None, symbol_search=None, filings_store=None, sweep: bool = True, guard=None):
        """toolbox: the DataToolbox (a fake in tests). store: the Chroma store of the news, press_store the one of the press releases and filings_store the one of
        the SEC filings (the real ones when left out). model: a chat model (anything with `bind_tools` and `invoke`; a fake in tests); otherwise OpenRouter
        through `get_chat_model` with the model of the `data_clerk` role, or `model_id`. allow_fetch: let the model use the fetch tools, which
        call paid APIs and write files; off unless you say so. symbol_search: the symbol directory a press release or filings order is checked against
        (Finnhub's search; a fake in tests, see prometheus/tools/symbols.py). sweep: after the model's loop, search the corpora it did not search and read the stored
        filing evidence, so that an order carries all three sources (entry 40); `request_data` only, `ask` never sweeps.

        guard (entry 116): the security layer's door for documents. With one, each document a search finds is screened by the code rules and a poisoned one is dropped before the model or the Advisor sees it; `Result.security_dropped`
        (and `order.data.security_dropped`) lists them. None screens nothing."""
        if max_turns < 2:
            raise ValueError("max_turns must be at least 2 (one turn to ask for a tool, one to answer)")
        self.tools = DataClerkTools(toolbox, store, press_store, filings_store, guard=guard)
        self.model = model or get_chat_model("data_clerk", model=model_id)
        self.allow_fetch, self.max_turns, self.calls, self.symbol_search, self.sweep = allow_fetch, max_turns, calls, symbol_search, sweep
        self.lookup_interval = 0.3   # seconds between two symbol lookups of `find_companies` (Finnhub answered HTTP 429 to ten searches in two seconds); a test sets 0

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
            result.security_dropped.extend(d for d in found.get("security_dropped", []) if d not in result.security_dropped)
            if hits:
                book.add(hits)
                for hit in hits:
                    result.sources.append({"number": book.number_of(hit), **hit.as_dict(), "swept": True})
            result.sweep[corpus] = {"by": "code", "sources": len(hits), "note": found.get("message") or ""}
        symbols = spec["tickers"] or []
        if result.filing_evidence:
            result.sweep["filing_evidence"] = {"by": "model", "sources": sum(len(e.get("evidence", [])) for e in result.filing_evidence.values()), "note": ""}
        elif not 1 <= len(symbols) <= MAX_EVIDENCE_SYMBOLS:
            result.sweep["filing_evidence"] = {"by": "code", "sources": 0, "note": f"the filing evidence is read for the symbols an order names (1 to {MAX_EVIDENCE_SYMBOLS}): this order names {len(symbols)}"}
        else:  # one symbol (entry 40) or a short list (entry 53: a ranking of several companies needs each one's evidence); the stored filings only, nothing is fetched
            notes = []
            for symbol in symbols:
                try:
                    evidence = self.tools.fetch_filings_recommendation(symbol)
                except ToolInputError as error:  # for example "no filings of X are stored": the reason is the message
                    notes.append(f"{symbol}: {str(error)[:120]}")
                except Exception as error:
                    notes.append(f"{symbol}: reading the filings failed ({type(error).__name__})")
                else:
                    result.filing_evidence[evidence["symbol"]] = evidence
            result.sweep["filing_evidence"] = {"by": "code", "sources": sum(len(e["evidence"]) for e in result.filing_evidence.values()), "note": "; ".join(notes)}

    @traced("data_clerk.request_data")
    def request_data(self, order: WorkOrder, today: date | None = None) -> WorkOrder:
        """The Data Clerk's agent call (the design's `DataClerk.request_data`): check the work order, answer its query, put the
        result on the order, check that, and return the SAME order. An order that already has its data comes back unchanged. A request the Data Clerk should not
        take is not an exception: the order comes back `data_failed` with the reasons in `errors`."""
        if order.status in HAS_DATA:
            return order  # it already has its data: a repeated request changes nothing (and never damages a good order)
        reasons = verify_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        if order.source in LOOKUP_SOURCES and not self._resolve_target(order):
            return order  # the symbol or company name was not found: refused in code, before any model call and before anything is fetched
        return attach_result(order, self.ask(self._question(order), today, sweep=self._sweep_spec(order) if self.sweep else None))

    @traced("data_clerk.request_quotes", run_type="tool")
    def request_quotes(self, order: WorkOrder) -> WorkOrder:
        """The Data Clerk's agent call for a `fetch_quote` order (entry 42): the current price of every symbol in `order.subject`, put on `order.data.quotes`, and the SAME order returned.

        There is no model call: a price is one lookup, so the Data Clerk calls its own `get_quote` tool (`DataClerkTools.get_quote`, the tool the model is also offered and the MCP
        server serves) once per symbol, in code. A symbol that fails (not a ticker, unknown to Finnhub, the API down) is named in `order.errors` and the order is `data_failed`;
        the quotes that did come back stay on the order, so a caller can see what it has. An order that already has its data comes back unchanged.

        With `USE_LIVE_TRADING=false` (design choices entry 76) the price is the market close of the simulated day (`common/trading_clock.market_day`: the previous business day) read from the stored daily bars
        (`DataClerkTools.close_quote`), marked `simulated`, and Finnhub is not called; a symbol with no stored bar on that day is a failed quote like any other."""
        if order.status in HAS_DATA:
            return order
        reasons = verify_quote_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        wanted = [order.subject] if isinstance(order.subject, str) else list(order.subject)
        quotes, calls, problems = {}, [], []
        day = None if use_live_trading() else market_day()   # USE_LIVE_TRADING=false (entry 76): every price is the market close of the simulated day, from the stored bars, and Finnhub is not asked
        for raw in dict.fromkeys(s.strip().lstrip("$").upper() for s in wanted):
            try:
                quotes[raw] = self.tools.get_quote(raw) if day is None else self.tools.close_quote(raw, day)
                calls.append({"name": "get_quote", "args": {"symbol": raw}, "outcome": "ok"})
            except Exception as error:  # a bad ticker, an unknown symbol, a missing key or an API failure: say which symbol and what kind, never request details
                calls.append({"name": "get_quote", "args": {"symbol": raw}, "outcome": "error"})
                problems.append(f"{raw}: no quote ({type(error).__name__}: {' '.join(str(error).split())[:160]})")
        order.data = DataResult(answer=", ".join(f"{s} {q['price']:.2f}" for s, q in quotes.items()), quotes=quotes, problems=problems, tool_calls=calls,
                                notes=[] if day is None else [f"USE_LIVE_TRADING=false: every price is the market close of {day}, from the stored daily bars (simulated); no live quote was asked for"],
                                completed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
        order.errors = [*order.errors, *problems]
        order.status = "data_failed" if problems else "data_ready"
        return order

    @traced("data_clerk.request_symbols", run_type="tool")
    def request_symbols(self, order: WorkOrder, today: date | None = None) -> WorkOrder:
        """The Data Clerk's agent call for a `lookup_symbol` order (design choices entries 88 and 101): for each stock in `order.subject` (a ticker in capitals, or a company name as the user typed it) it finds the COMPANY'S NAME and the
        SYMBOL and APPENDS them to the work order's `data.companies` (one row each, the shape of `find_companies`: {brand: what was typed, parent, symbol, company, public, how, note}), and returns the SAME order.

        1. THE EXACT LOOKUP, no model (the symbol directory, Finnhub's search, with the EXACT matching of press releases and filings, entries 34 and 35, `symbols.resolve_target`): a ticker is the row whose symbol it is; a company
        name is the row whose name has the same words, else the same words without the generic company words (Inc, Corp, Co, The, and ...), never a spelling that is only similar, US listings preferred. Most requests end here.
        2. A NAME THE DIRECTORY DOES NOT HAVE (entry 101: "Google", which Finnhub lists as Alphabet Inc.): the Data Clerk asks its model for the LISTED OWNER of that name (the proposal step of `find_companies`, one name) and then
        looks the OWNER'S NAME up with the same exact lookup (`symbols.find_company`). The model supplies a company name and nothing else that is used: the symbol is always the directory's, its ticker hint is not read, and
        a name that only starts like a listing is not accepted. A ticker never reaches the model, and neither does a name that fits several listings (the directory already answered that one). Found: a row with `how: "brand"`.

        Every stock gets a row when the directory ANSWERED, found or not: a stock that is not listed has `symbol: None` and the reason in `note` ("privately held, or not known to be listed", "no listed company named ... in Finnhub's
        directory", the candidates of a name that fits several listings, "no such symbol in Finnhub's directory" for a ticker). A lookup that FAILED (a rate limit, no key, the model did not answer) is never "not found": it is named in
        the order's errors and has NO row, which is how the caller tells a company that is not listed from a question that was not answered. The order is `data_failed` when any stock has no symbol, with the rows that were found on
        it; `data_ready` when every stock has one. An order that already has its data comes back unchanged."""
        if order.status in HAS_DATA:
            return order
        reasons = verify_symbol_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        search = self._paced(self.symbol_search or symbols.search_finnhub)
        wanted = [order.subject] if isinstance(order.subject, str) else list(order.subject)
        rows, calls, problems, tokens = [], [], [], {"in": 0, "out": 0}
        for typed in dict.fromkeys(" ".join(n.split()) for n in wanted):
            is_ticker = symbols.is_plain_ticker(typed)
            call = {"name": "find_symbol" if is_ticker else "find_company", "args": {"symbol" if is_ticker else "company": typed}}
            try:
                resolution = symbols.resolve_target(typed if is_ticker else None, None if is_ticker else typed, search)
                rows.append({"brand": typed, "parent": None, "symbol": resolution.symbol, "company": resolution.company, "public": True, "how": resolution.how, "note": ""})
                calls.append({**call, "outcome": "ok"})
            except symbols.TargetNotFound as error:   # the directory ANSWERED: not in it, or several listings fit
                calls.append({**call, "outcome": "error"})
                if is_ticker or error.detail:         # a ticker is the row whose symbol it is, and several listings are the directory's own answer: no model, never a guess
                    rows.append(self._not_listed(typed, error.detail or "no such symbol in Finnhub's directory"))
                    problems.append(f"{typed}: {error}" + (f" ({error.detail})" if error.detail else ""))
                    continue
                row, failure = self._listed_owner(typed, search, today, tokens, calls)
                if failure:
                    problems.append(f"{typed}: the symbol lookup failed ({failure})")
                    continue
                rows.append(row)
                if row["symbol"] is None:
                    problems.append(f"{typed}: {row['note']}")
            except (symbols.SymbolLookupError, ValueError) as error:   # the lookup itself failed (a rate limit, no key): never "not found", and no row
                calls.append({**call, "outcome": "error"})
                problems.append(f"{typed}: the symbol lookup failed ({getattr(error, 'reason', error)})")
        data = order.data or DataResult()
        data.companies = [*data.companies, *rows]
        data.problems, data.tool_calls = [*data.problems, *problems], [*data.tool_calls, *calls]
        data.tokens_in, data.tokens_out = data.tokens_in + tokens["in"], data.tokens_out + tokens["out"]
        data.answer = "; ".join(f"{c['brand']}: {c['company']} ({c['symbol']})" for c in rows if c["symbol"])
        data.completed_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        order.data = data
        order.errors = [*order.errors, *problems]
        order.status = "data_failed" if problems else "data_ready"
        return order

    @staticmethod
    def _not_listed(typed: str, note: str, parent: str | None = None) -> dict:
        """The row of a stock the directory ANSWERED for and does not list: no symbol, and the reason in the Data Clerk's own words."""
        return {"brand": typed, "parent": parent, "symbol": None, "company": None, "public": False, "how": None, "note": note}

    def _listed_owner(self, typed: str, search, today: date | None, tokens: dict, calls: list) -> tuple[dict | None, str | None]:
        """The row of a NAME the exact lookup did not find (entry 101): the model names the listed owner of what was typed, and the OWNER'S NAME is looked up by the exact words (`symbols.find_company`). Returns (the row, None), or
        (None, why the question was not answered: the model did not reply, or the directory could not be asked). The row has a symbol only when the directory has a company of that name; its `note` says why not otherwise. The model's
        ticker hint and the start of a listed name are NOT used here: they are fine for choosing companies to rank (`_resolve_proposal`), and a trade needs the exact words."""
        from prometheus.agents.advisor.calls import new_run
        run = new_run("owner", self.model)
        reply = self._propose(typed, 1, today, run)
        tokens["in"] += run["tokens_in"]
        tokens["out"] += run["tokens_out"]
        calls.append({"name": "find_listed_owner", "args": {"company": typed}, "outcome": "ok" if reply is not None else "error"})
        if reply is None:
            return None, run["error"] or "the model did not answer"
        proposal = next((p for p in reply["companies"] if isinstance(p, dict)), {})
        owner = " ".join(str(proposal.get("parent")).split()) if isinstance(proposal.get("parent"), str) and proposal["parent"].strip() else None
        if not owner:
            return self._not_listed(typed, NOT_LISTED), None
        if len(owner) > 60 or not symbols.NAME_CHARS.fullmatch(owner):   # not a company's name: it is not echoed, and nothing is looked up
            return self._not_listed(typed, "the Data Clerk's model gave no company name that can be looked up"), None
        try:
            found = symbols.find_company(owner, search)
        except symbols.TargetNotFound as error:
            return self._not_listed(typed, f"its owner {owner!r} matches several listings, so none is chosen ({error.detail[:120]})", owner), None
        except (symbols.SymbolLookupError, ValueError) as error:
            return None, str(getattr(error, "reason", error))
        if found is None:
            return self._not_listed(typed, f"no listed company named {owner!r} in Finnhub's directory", owner), None
        return {"brand": typed, "parent": owner, "symbol": found.symbol, "company": found.company, "public": True, "how": "brand",
                "note": "the listed owner of what you typed, named by the Data Clerk's model; the symbol is the directory's"}, None

    @traced("data_clerk.request_bars", run_type="tool")
    def request_bars(self, order: WorkOrder) -> WorkOrder:
        """The Data Clerk's agent call for a `fetch_bars` order (Trader v1 spec section 12): the stored daily bars of every symbol of `order.subject` from `order.bars["start"]` to `order.bars["end"]`,
        put on `order.data.bars` as {symbol: [bar, ...]} (oldest first, each {"date", "open", "high", "low", "close", "volume"}), and the SAME order returned. The Account Manager asks, for the Trader's backtest;
        the Trader never calls the Data Clerk.

        There is no model call: a bar is a stored number, so the Data Clerk calls its own tool code (`DataClerkTools.daily_bars`) in code. Nothing is fetched and nothing is written under `data/historical`. A symbol
        with no stored bar in the range is an empty list and a note in `data.notes`: that is a RESULT (the future may not be stored yet), not a failure. The order is `data_failed` only when the request is refused
        (`verify_bar_request`, or a symbol that is not a ticker) or the store could not be read; the reason is in `order.errors`. An order that already has its data comes back unchanged."""
        if order.status in HAS_DATA:
            return order
        reasons = verify_bar_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        wanted = [order.subject] if isinstance(order.subject, str) else list(order.subject)
        try:
            symbols = list(dict.fromkeys(check_symbol(s.strip().lstrip("$")) for s in wanted))
        except ValueError as error:  # not a ticker symbol: refused before anything is read
            order.status, order.errors = "data_failed", [*order.errors, str(error)]
            return order
        start, end = iso_date(order.bars["start"]), iso_date(order.bars["end"])
        args = {"symbols": len(symbols), "start": order.bars["start"], "end": order.bars["end"]}
        try:
            found = self.tools.daily_bars(symbols, start, end)
        except Exception as error:  # the bar folder is missing or unreadable, or a bug: say the kind, never details
            order.status, order.errors = "data_failed", [*order.errors, f"the stored bars could not be read ({type(error).__name__}: {' '.join(str(error).split())[:160]})"]
            order.data = DataResult(tool_calls=[{"name": "daily_bars", "args": args, "outcome": "error"}], completed_utc=_utc_now())
            return order
        bars = {s: found["bars"].get(s, []) for s in symbols}
        notes = [f"{s}: no stored bar from {order.bars['start']} to {order.bars['end']}" for s in symbols if not bars[s]] + [f"bars: {gap}" for gap in found["gaps"]]
        order.data = DataResult(answer=f"daily bars of {len(symbols)} symbols, {order.bars['start']} to {order.bars['end']}", bars=bars, notes=notes,
                                tool_calls=[{"name": "daily_bars", "args": args, "outcome": "ok"}], completed_utc=_utc_now())
        order.status = "data_ready"
        return order

    @traced("data_clerk.request_metrics")
    def request_metrics(self, order: WorkOrder, today: date | None = None) -> WorkOrder:
        """The Data Clerk's agent call for a `calculate_financials` order (Advisor v1 spec section 4.1, entry 51): the metrics asked for, for every symbol of `order.subject`, put on
        `order.data.metrics` as {symbol: {metric: result}} (each result: value, unit, as_of, window, source, inputs_summary, note), and the SAME order returned.

        There is no model call: a number is a calculation, so the Data Clerk calls its own tool code (`DataClerkTools.metrics_for`, the code behind the `calculate_metrics` tool the model is
        also offered) in code. A metric that cannot be computed is `value: None` with a note: that is a result, not a failure. The order is `data_failed` only when a symbol could not be read at
        ALL (no stored bars, no Finnhub payload and no stored filings), and that symbol is named in `order.errors`; every other symbol keeps its results. What could not be read for a symbol
        that has other data (a Finnhub call that failed, no stored filings) is in `data.notes`. An order that already has its data comes back unchanged."""
        if order.status in HAS_DATA:
            return order
        reasons = verify_metric_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        wanted = [order.subject] if isinstance(order.subject, str) else list(order.subject)
        try:
            symbols = list(dict.fromkeys(check_symbol(s.strip().lstrip("$")) for s in wanted))
        except ValueError as error:  # not a ticker symbol: refused before anything is read
            order.status, order.errors = "data_failed", [*order.errors, str(error)]
            return order
        try:
            found = self.tools.metrics_for(symbols, order.metrics, today, sector_map(order))
            call = {"name": "calculate_metrics", "args": {"symbols": len(symbols), "metrics": len(order.metrics)}, "outcome": "ok"}
        except Exception as error:  # the bars could not be read at all, or a bug in a metric: say the kind, never details
            order.status, order.errors = "data_failed", [*order.errors, f"the metrics could not be calculated ({type(error).__name__}: {' '.join(str(error).split())[:160]})"]
            order.data = DataResult(tool_calls=[{"name": "calculate_metrics", "args": {"symbols": len(symbols), "metrics": len(order.metrics)}, "outcome": "error"}], completed_utc=_utc_now())
            return order
        unreadable = [s for s in symbols if not found["bars"].get(s) and all(r["value"] is None for r in found["results"][s].values())]  # no bars and not one number from anything else
        failed = [f"{s}: nothing could be read ({found['problems'].get(s, 'no stored bars for it')})" for s in unreadable]
        notes = [f"{s}: {reason}" for s, reason in found["problems"].items() if s not in unreadable] + [f"bars: {gap}" for gap in found["gaps"]]
        order.data = DataResult(answer=f"{len(order.metrics)} metrics for {len(symbols)} symbols", metrics=found["results"], problems=failed, notes=notes, tool_calls=[call], completed_utc=_utc_now())
        order.errors = [*order.errors, *failed]
        order.status = "data_failed" if failed else "data_ready"
        return order

    @traced("data_clerk.request_verification", run_type="tool")
    def request_verification(self, order: WorkOrder, now: datetime | None = None) -> WorkOrder:
        """The Data Clerk's agent call for a `verify_data` order (Critic v1 spec section 5.2, design choices entry 58): read again, from the stores, what a recommendation rests on, and put the answer on
        `order.data.verification`; the SAME order is returned. `order.verify` says what (`verify_verification_request` is the check):

            chunks    stored chunk ids, read again with their text                                  (`DataClerkTools.read_sources`)
            evidence  {symbol: [E ids]}, the filing-evidence items, read again from the stored filings  (`read_evidence`)
            symbols   tickers, looked up in Finnhub's directory: listed, and under which name        (`symbols.find_symbol`)
            fresh     the newest stored news and press releases of each symbol                        (`recent_sources`, no relevance bar)

        There is no model call: a lookup is a lookup, and this Data Clerk answers a Critic that wants to know whether the Advisor's evidence is real, so nothing here may be "reasoned". Nothing is
        fetched and nothing is written. A chunk that is not found is a RESULT (`found: False` with a note), not a failure; a symbol whose lookup failed is `listed: None`, never `False` (a rate limit is not
        "not listed"). Each part that raises is named in `order.errors`; the order is `data_failed` only when the request is refused or EVERY part failed. An order that already has its data comes back unchanged."""
        if order.status in HAS_DATA:
            return order
        reasons = verify_verification_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        spec, answer, calls, problems = order.verify, {"chunks": {}, "evidence": {}, "symbols": {}, "fresh": {}}, [], []
        attempted = failed = 0

        def part(name: str, args: dict, call) -> bool:
            """Run one part: record the tool call and, when it raises, the problem. Returns whether it worked."""
            nonlocal attempted, failed
            attempted += 1
            try:
                call()
            except Exception as error:  # the kind and a short message, never request details
                failed += 1
                calls.append({"name": name, "args": args, "outcome": "error"})
                problems.append(f"{name}{' ' + args['symbol'] if 'symbol' in args else ''}: {type(error).__name__}: {' '.join(str(error).split())[:160]}")
                return False
            calls.append({"name": name, "args": args, "outcome": "ok"})
            return True

        chunks = list(dict.fromkeys(spec.get("chunks", [])))
        if chunks:
            with _RETRIEVE_LOCK:
                part("read_sources", {"chunks": len(chunks)}, lambda: answer["chunks"].update(self.tools.read_sources(chunks)))
        for symbol, ids in spec.get("evidence", {}).items():
            part("read_evidence", {"symbol": symbol, "ids": len(ids)}, lambda s=symbol, i=ids: answer["evidence"].__setitem__(s, self.tools.read_evidence(s, i)))
        wanted = list(dict.fromkeys(spec.get("symbols", [])))
        search = self._paced(self.symbol_search or symbols.search_finnhub) if wanted else None
        for symbol in wanted:
            attempted += 1
            try:
                found = symbols.find_symbol(symbol, search)
                answer["symbols"][symbol] = {"listed": found is not None, "company": found.company if found else None, "note": None if found else "Finnhub's directory has no such symbol"}
                calls.append({"name": "find_symbol", "args": {"symbol": symbol}, "outcome": "ok"})
            except Exception as error:  # a lookup that FAILED (rate limit, no key): unknown, not "not listed"
                answer["symbols"][symbol] = {"listed": None, "company": None, "note": f"the lookup failed ({getattr(error, 'reason', type(error).__name__)})"[:160]}
                calls.append({"name": "find_symbol", "args": {"symbol": symbol}, "outcome": "error"})
                problems.append(f"find_symbol {symbol}: {answer['symbols'][symbol]['note']}")
                failed += 1
        fresh = spec.get("fresh")
        for symbol in wanted if fresh else []:
            with _RETRIEVE_LOCK:
                part("recent_sources", {"symbol": symbol}, lambda s=symbol: answer["fresh"].__setitem__(
                    s, self.tools.recent_sources(s, spec.get("company", {}).get(s), fresh.get("since_days", 14), fresh.get("k", 8), now)))
        order.data = DataResult(answer=f"read again: {len(answer['chunks'])} chunks, {sum(len(v) for v in answer['evidence'].values())} evidence items, {len(answer['symbols'])} symbols, "
                                       f"{sum(len(v) for v in answer['fresh'].values())} newer sources", verification=answer, problems=problems, tool_calls=calls, completed_utc=_utc_now())
        order.errors = [*order.errors, *problems]
        order.status = "data_failed" if attempted and failed == attempted else "data_ready"
        return order

    # ---------------------------------------------------------------- a phrase to companies, and the data for them (design choices entry 53)

    @traced("data_clerk.find_companies")
    def find_companies(self, phrase: str, count: int = 10, today: date | None = None) -> dict:
        """Which listed companies belong to an industry or a theme ("Pizza"): the Data Clerk's model PROPOSES brands and the companies that own them, and CODE turns every proposal into a real
        ticker or says it cannot. The model's memory is only the list of names; it supplies no ticker that is used, no number and no evidence (every score still comes from bars, filings and news).

        Each proposal ({brand, parent, ticker}) is resolved in this order, the first that works winning: the parent's name by `symbols.find_company` (the same words as a Finnhub listing, generic
        company words ignored: "Domino's Pizza" is "Domino's Pizza Inc"); the brand's name the same way; the start of a listed name when exactly one listing starts with it
        (`resolve_company`: "Papa John's" is "Papa John's International"); the model's ticker hint, only if Finnhub has that ticker AND every distinctive word of one of the two names is in the other. A proposal with no parent is private (or unsure): listed as not scored. Tickers the stored market news ties to the phrase (`discover_tickers`, free) with at least 3 articles are CANDIDATES, and a candidate the model did not propose is added only when the Data Clerk's model
        confirms that the company belongs to the phrase (`_confirmed_by_news`, entry 120: one extra model call, fail closed; a company only mentioned next to the theme is left out, with its reason in `notes`).
        Returns {phrase, companies: [{brand, parent, symbol, company, public, how, note}], sector_etf (one of the 11, or None), notes, run, membership (the run of the confirming call, None when there was none)}; never raises for a model or lookup failure: it is in `notes`."""
        from prometheus.agents.advisor.calls import new_run
        from prometheus.agents.data_clerk.metrics_catalog import sector_etfs
        text = " ".join(phrase.split()) if isinstance(phrase, str) else ""
        if not text or len(text) > 60 or not re.fullmatch(r"[A-Za-z0-9 &.,'\u2019\-]+", text):
            raise ValueError("the phrase must be 1 to 60 characters of letters, digits, spaces and & . , ' -")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 20:
            raise ValueError("count must be a whole number from 1 to 20")
        self.symbol_search = self._paced(self.symbol_search or symbols.search_finnhub)
        run = new_run("companies", self.model)
        out = {"phrase": text, "companies": [], "sector_etf": None, "notes": [], "run": run, "membership": None}
        reply = self._propose(text, count, today, run)
        if reply is None:
            out["notes"].append(run["error"])
        else:
            sector = reply.get("sector_etf")
            out["sector_etf"] = sector.strip().upper() if isinstance(sector, str) and sector.strip().upper() in sector_etfs() else None
            for proposal in reply["companies"][:count]:
                if isinstance(proposal, dict):
                    out["companies"].append(self._resolve_proposal(proposal, out["notes"]))
        try:
            proposed = {c["symbol"] for c in out["companies"]}
            news = list({row["symbol"]: row for row in self.tools.discover_tickers(text, from_date=30, top=10)["tickers"] if row.get("articles", 0) >= 3 and row.get("symbol") not in proposed}.values())
        except Exception as error:  # the stored market news cannot be read: the model's list stands alone
            out["notes"].append(f"stored-news discovery failed ({type(error).__name__})")
            news = []
        if news:
            out["companies"].extend(self._confirmed_by_news(text, news, today, out))
        seen = set()
        out["companies"] = [c for c in out["companies"] if not (c["symbol"] and (c["symbol"] in seen or seen.add(c["symbol"])))]
        return out

    def _confirmed_by_news(self, text: str, news: list[dict], today: date | None, out: dict) -> list[dict]:
        """The companies of the stored market news (`news`: the rows of `discover_tickers`, none of them proposed) that BELONG to the phrase (entry 120, design/theme_membership_spec.md). A news article names a company for many reasons
        (a headline about oil prices names Procter & Gamble), so the Data Clerk's model is asked ONCE, for all of them: for each ticker, the directory's own name for the company and the top headline the discovery found; its answer is
        {"members": [{symbol, belongs, reason}]} and CODE checks the form. FAIL CLOSED: a ticker the directory does not name is never sent; a symbol that was not sent is ignored; a ticker the answer leaves out, a `belongs` that is
        not true or false, or no usable answer at all adds nothing (the model's own proposals stand alone), and `out["notes"]` says why. A company the model refuses is a note with its reason, never a candidate.
        The call is `out["membership"]` (the run: model, tokens, error). Returns the rows to add: {brand, parent, symbol, company, public, how, note}."""
        from prometheus.agents.advisor.calls import ask_object, new_run
        notes = out["notes"]
        asked = []
        for row in news:
            symbol = row["symbol"]
            try:
                listed = symbols.find_symbol(symbol, self.symbol_search)
            except symbols.SymbolLookupError as error:
                notes.append(f"{symbol}: in the news for {text!r} but left out: the symbol lookup failed ({getattr(error, 'reason', error)})")
                continue
            if listed is None:
                notes.append(f"{symbol}: in the news for {text!r} but left out: no company of that symbol in Finnhub's directory")
                continue
            headlines = [h for h in row.get("headlines") or [] if isinstance(h, str) and h.strip()]
            asked.append({"symbol": symbol, "company": " ".join(listed.company.split())[:80], "headline": " ".join(headlines[0].split())[:200] if headlines else ""})
        if not asked:
            return []
        run = new_run("theme_members", self.model)
        out["membership"] = run
        reply = ask_object(self.model, [SystemMessage(content=theme_members_prompt(today)), HumanMessage(content=theme_members_question(text, asked))], run, "members", THEME_MEMBERS_RETRY)
        if reply is None:
            notes.append(f"the stored news names {', '.join(a['symbol'] for a in asked)} for {text!r}, but none was added: the model could not confirm that they belong ({run['error']})")
            return []
        verdicts = {}
        for item in reply["members"]:
            if isinstance(item, dict) and isinstance(item.get("symbol"), str) and item["symbol"].strip().upper() in {a["symbol"] for a in asked} and item["symbol"].strip().upper() not in verdicts:
                verdicts[item["symbol"].strip().upper()] = item
        added, count = [], {row["symbol"]: row["articles"] for row in news}
        for entry in asked:
            symbol, item = entry["symbol"], verdicts.get(entry["symbol"])
            name = f"{symbol} ({entry['company']})"
            reason = " ".join(item["reason"].split())[:200] if item is not None and isinstance(item.get("reason"), str) and item["reason"].strip() else "no reason given"
            if item is None or not isinstance(item.get("belongs"), bool):
                notes.append(f"{name}: in the news for {text!r} but left out: the model did not say whether it belongs")
            elif not item["belongs"]:
                notes.append(f"{name}: in the news for {text!r} but not in the theme: {reason}")
            else:
                added.append({"brand": symbol, "parent": None, "symbol": symbol, "company": entry["company"], "public": True, "how": NEWS_ADDED, "note": f"{count[symbol]} stored articles; the model says it belongs: {reason}"})
        return added

    def _propose(self, text: str, count: int, today: date | None, run: dict) -> dict | None:
        """The proposal step: the Data Clerk's model names up to `count` companies for a phrase ({brand, parent, ticker} each, and a sector fund), in the one JSON-with-a-retry loop of the Advisor's calls
        (`calls.ask_object`). Used by `find_companies` for an industry or a theme and by `_listed_owner` for one name. None when the model did not answer (`run["error"]` says why)."""
        from prometheus.agents.advisor.calls import ask_object
        return ask_object(self.model, [SystemMessage(content=companies_prompt(text, count, today)), HumanMessage(content=f"Industry or theme: {text}")], run, "companies", COMPANIES_RETRY)

    def _paced(self, search):
        """The symbol search, no faster than `lookup_interval` and with a pause and a second try when Finnhub answers 429 (a rate limit is not "not found"). Wrapped once."""
        if getattr(search, "_paced", False) or not self.lookup_interval:
            return search
        last = [0.0]

        def paced(query: str):
            for attempt in range(3):
                time.sleep(max(0.0, last[0] + self.lookup_interval - time.monotonic()))
                last[0] = time.monotonic()
                try:
                    return search(query)
                except symbols.SymbolLookupError as error:
                    if "429" not in str(error) or attempt == 2:
                        raise
                    time.sleep(3.0 * (attempt + 1))
        paced._paced = True
        return paced

    def _resolve_proposal(self, proposal: dict, notes: list[str]) -> dict:
        """One proposal -> {brand, parent, symbol, company, public, how, note}; see `find_companies` for the order of the attempts."""
        clean = lambda v: " ".join(v.split())[:60] if isinstance(v, str) and v.strip() else None
        brand, parent, hint = clean(proposal.get("brand") or proposal.get("name")), clean(proposal.get("parent")), clean(proposal.get("ticker"))
        row = {"brand": brand or parent or "?", "parent": parent, "symbol": None, "company": None, "public": False, "how": None, "note": None}
        if not parent:
            row["note"] = "privately held, or not known to be listed: not scored"
            return row
        found, how = None, None
        try:
            for name in dict.fromkeys(n for n in (parent, brand) if n):
                found = symbols.find_company(name, self.symbol_search)
                if found:
                    how = f'company name "{name}": the same words as the listing'
                    break
            if not found:
                try:
                    start = symbols.resolve_company(parent, self.symbol_search)
                    found, how = (start, "the start of the listed name") if start.how == "name" else (None, None)
                except symbols.SymbolLookupError:
                    pass
            if not found and hint:
                guess = symbols.find_symbol(hint, self.symbol_search)
                words = lambda t: {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) >= 3 and w not in symbols.GENERIC_WORDS}
                listed, named = words(guess.company) if guess else set(), words(parent)
                if listed and named and (named <= listed or listed <= named):  # every distinctive word of one name is in the other: "Yum Brands" and "YUM! BRANDS INC", not "Zorblax Pizza Holdings" and "Domino's Pizza" (they share "pizza")
                    found, how = guess, f"the model's ticker {hint}, checked: Finnhub has it and its name has the same distinctive words as {parent!r}"
        except symbols.TargetNotFound as error:  # several listings carry the name: not guessed
            row["note"] = f"{parent!r} matches several listings ({error.detail[:100]}): not scored"
            return row
        except symbols.SymbolLookupError as error:
            notes.append(f"{parent}: the lookup failed ({getattr(error, 'reason', error)})")
            row["note"] = "the symbol lookup failed: not scored"
            return row
        if not found:
            row["note"] = f"no listed company named {parent!r} in Finnhub's directory: not scored"
            return row
        row.update(symbol=found.symbol, company=found.company, public=True, how=how)
        if hint and hint.upper() != found.symbol:
            row["note"] = f"the model guessed {hint}; the lookup says {found.symbol}"
        return row

    @traced("data_clerk.refresh_data")
    def refresh_data(self, symbols_: list[str], news_days: int = 14, filings_days: int = 365, bars_days: int = 400, end: date | None = None, bars_for: list[str] | None = None) -> list[dict]:
        """Fetch what the Advisor needs for these symbols, with the PAID APIs, and say what happened to each step (only when the Data Clerk was built with `allow_fetch`: otherwise nothing is
        fetched and the one entry says so). For SPY, the 11 sector ETFs and every symbol: the daily bars (`DataToolbox.fetch_daily_bars`, one call each, Massive REST). For each symbol: its news
        (`news_days`), its press releases and its SEC filings with the 10-Q text from EDGAR (`filings_days`), indexed for search. A step that fails is an entry with `outcome: "error"` and its
        kind, and the next step still runs: a company whose filings cannot be read still gets its bars and news. `bars_for` are more symbols that need only their bars (the user's holdings: the
        Advisor sizes a position from their prices). Returns [{step, symbol, outcome, detail}]."""
        from prometheus.agents.data_clerk.metrics_catalog import SPY, sector_etfs
        if bound_day() is not None:   # the Proving Ground (`trading_clock.as_of`): a past day calls no paid API and writes no stored text
            return [{"step": "all", "symbol": None, "outcome": "refused", "detail": f"{NEVER_FETCHES}: nothing is fetched while the desk works as of {bound_day()}"}]
        if not self.allow_fetch:
            return [{"step": "all", "symbol": None, "outcome": "refused", "detail": "the Data Clerk was built without allow_fetch: nothing is fetched (fetching uses paid APIs and writes files)"}]
        wanted = list(dict.fromkeys(check_symbol(s.strip().lstrip("$")) for s in symbols_))
        if not wanted or len(wanted) > MAX_REFRESH_SYMBOLS:
            raise ValueError(f"refresh 1 to {MAX_REFRESH_SYMBOLS} symbols at a time")
        report = []

        def step(name: str, symbol: str | None, call) -> None:
            try:
                detail = call()
                report.append({"step": name, "symbol": symbol, "outcome": "ok", "detail": detail})
            except Exception as error:  # the kind and a short message, never request details or a key
                report.append({"step": name, "symbol": symbol, "outcome": "error", "detail": f"{type(error).__name__}: {' '.join(str(error).split())[:160]}"})

        step("daily_bars", None, lambda: {s: i["stored"] for s, i in self.tools.toolbox.fetch_daily_bars(list(dict.fromkeys([*wanted, *(check_symbol(b.strip().lstrip("$")) for b in bars_for or []), SPY, *sector_etfs()])), bars_days, end).items()})
        for symbol in wanted:
            step("news", symbol, lambda s=symbol: {k: v for k, v in self.tools.fetch_news(s, from_date=news_days).items() if k in ("fetched", "written", "duplicates")})
            step("press_releases", symbol, lambda s=symbol: {k: v for k, v in self.tools.fetch_press_releases(s, from_date=filings_days).items() if k in ("fetched", "written", "duplicates")})
            step("filings", symbol, lambda s=symbol: {k: v for k, v in self.tools.fetch_filings(s, from_date=filings_days).items() if k in ("fetched", "written", "edgar_text")})
        return report

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
                if isinstance(m.artifact, dict):
                    result.security_dropped.extend(d for d in m.artifact.get("security_dropped", []) if d not in result.security_dropped)
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
        if result.security_dropped:
            result.notes.append(f"{len(result.security_dropped)} retrieved document(s) were held back by the security screen before anything read them")


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
