"""The desk: a LangGraph that passes ONE work order from agent to agent (design choices entry 29).

    python -m prometheus.langgraph.desk --symbol MSFT --from-days-ago 14        # a symbol: fetch fresh news, then answer and recommend
    python -m prometheus.langgraph.desk --term "flying car"                     # a term, no symbol: search the stored news
    python -m prometheus.langgraph.desk --symbol SNDK --term "long-term contracts"   # a term inside one symbol's news
    ./test_prometheus.sh                                                        # the same, with prompts for the four parameters

The graph:

    START -> account_manager -> data_clerk -> advisor -> END                    (with_funds=True: ... -> advisor -> account_manager_funds -> END, entry 42)
                   |                |
                   |                +-- the data is not ready (failed, or no search was made) --> END
                   +-- the request is not valid --> END

Each node is one agent of the week 2 design, reduced to what exists today, and each one reads and updates the SAME work order, which is the
state of the graph (the design: "the saved WORK_ORDER is the same object every agent reads"):

  account_manager  builds the work order from the four parameters and checks it (`verify_request`). It is the Account Manager's `parse_input`
                   and `create_work_order`, without the trading half.
  data_clerk       `DataClerk.request_data(order)`: fetches if the order is live, searches, answers, then (entry 40) searches the stored sources the model did not search and reads the filing
                   evidence, so the order carries all three; puts the data and the sources on the order.
  advisor          `Advisor.advise(order)`: reads the chunks, `data.results` and `data.filing_evidence` on the order and appends recommendations with a calculated score.
  account_manager_funds  (only with `build_desk(with_funds=True)`, entry 42) the Account Manager's RaiseFunds: it takes the best recommendation (or the stock the user named), asks the Data Clerk for the
                   current prices, and runs a Tree of Thought for what to sell from the user's portfolio to buy it. Its answer is `state["funds"]`; the work order is not changed.

THE FOUR PARAMETERS (what the user types, and what each becomes):

  symbol         a ticker, for example MSFT, OR A COMPANY NAME, for example "Procter & Gamble". A ticker in capitals is used as it is. Anything else
                 (a name, or a ticker in lower case) is LOOKED UP (Finnhub's symbol search, prometheus/tools/symbols.py), never guessed: the answer is
                 shown ("Procter&Gamble" -> PG, Procter & Gamble Co), carried on the work order as `company`, and an ambiguous name is refused.
                 With a symbol the order is LIVE: fresh news is fetched for it with the PAID APIs (and written to news/<SYMBOL>.txt) unless
                 fetching is turned off.
  term           a topic, for example "flying car". With no symbol it is searched in the stored news and nothing is fetched. With a symbol the
                 term narrows the question inside that symbol's news (fetching is still for the whole symbol, never filtered by the term, so the
                 file stays complete).
  source         what is searched: `news` (the default), `press_releases` (design choices entry 34) or `filings` (the SEC's filings, entry 35).
  from_days_ago  how many days back (the `from_date` of the fetch and the window of the search). News: default 7, from 1 to 90. Press releases and filings: default 365, from 1 to 730.

At least one of symbol and term is required. The Advisor's query is the work order built from these, not a free sentence.

A PRESS RELEASE request (`--source press_releases`) takes a symbol and/or a COMPANY NAME (`--company`, a fifth parameter that only this source uses). Both are
passed on the work order; the Data Clerk looks them up in code: the symbol if it is found, else the company name (matched on its words, ignoring case, punctuation and generic
words such as Inc and Corp, but never a typo); if neither is found the
order is `data_failed` with "Symbol: X is not found" or "Company Name not found." and the Advisor never runs. Fuzzy company-name matching is a TODO.
    python -m prometheus.langgraph.desk --source press_releases --symbol AAPL
    python -m prometheus.langgraph.desk --source press_releases --company "Apple Inc"

FILINGS (`--source filings`, entry 35) are asked for in exactly the same way (a symbol and/or `--company`), looked up by the same code with the same messages, and the order
is built by the same branch of `order_from_request`: only the corpus the Data Clerk searches is different.
    python -m prometheus.langgraph.desk --source filings --symbol AAPL
    python -m prometheus.langgraph.desk --source filings --company "Apple Inc" --no-fetch
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO, AccountManager, report as funds_report, top_recommendation
from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common.work_order import LOOKUP_SOURCES, REQUEST_LABEL, WIRED_SOURCES, WorkOrder, new_work_order, verify_request
from prometheus.common.portfolio import load_portfolio
from prometheus.rag.loaders import check_symbol
from prometheus.tools import symbols

DEFAULT_SOURCE = WIRED_SOURCES[0]
DEFAULT_DAYS = 7
DEFAULT_DAYS_BY_SOURCE = {"news": 7, "press_releases": 365, "filings": 365}  # a company issues a few press releases or filings a month: a week would usually find none (entries 34, 35)
TOP_K = 10
MAX_TERM = 200
REQUESTER = "desk"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"  # where a finished order is saved (tests point it elsewhere)


class DeskState(TypedDict, total=False):
    request: dict                                              # the four parameters as given
    order: WorkOrder | None                                    # THE work order: every node reads and updates this one object
    errors: list[str]                                          # why the request was refused (no order could be built, or it is not valid)
    resolution: dict | None                                    # a company name that was looked up: what was typed, the ticker, the company, how it matched
    funds: dict | None                                         # the Account Manager's RaiseFunds answer (only when the desk is built with funds, entry 42)
    path: Annotated[list[str], lambda old, new: old + new]     # the nodes that ran, in order
    seconds: Annotated[dict, lambda old, new: {**old, **new}]  # how long each node took


def order_from_request(request: dict) -> WorkOrder:
    """The four parameters -> a work order. Raises ValueError, with a message for the user, if there is no symbol and no term or a value is wrong.

    symbol only    "What is the latest news on MSFT ...?"        subject [MSFT], live
    term only      the term itself                                 subject the term, not live (there is no ticker to fetch)
    both           "<term> news for <symbol>"                     subject [MSFT], live: the term narrows the question inside that symbol

    SOURCES press_releases (entry 34) and filings (entry 35) take a symbol and/or a COMPANY NAME (request["company"]), and the term only narrows the question:
    symbol and/or company   subject [symbol] or None, company the typed name, live (a fresh fetch) unless request["fetch"] is false
    The symbol is NOT checked as a ticker here: if it is not found the Data Clerk falls back to the company name, so a bad symbol with a good name must get that far.
    """
    source = request.get("source") or DEFAULT_SOURCE
    by_name = source in LOOKUP_SOURCES  # press releases and filings: asked for by symbol and/or company name, looked up by the Data Clerk
    symbol, term = request.get("symbol"), request.get("term")
    if isinstance(symbol, str) and symbol.strip():
        symbol = symbol.strip().lstrip("$").strip()
        if by_name:  # whether it is a symbol that exists is the Data Clerk's lookup, not a pattern check
            if len(symbol) > symbols.MAX_TEXT:
                raise ValueError(f"the symbol is longer than {symbols.MAX_TEXT} characters")
        else:
            symbol = check_symbol(symbol)
    else:
        symbol = None
    term = term.strip() if isinstance(term, str) and term.strip() else None
    company = request.get("company").strip() if isinstance(request.get("company"), str) and request.get("company").strip() else None
    if by_name and not symbol and not company:
        raise ValueError(f"a {REQUEST_LABEL[source]} request needs a symbol, a company name, or both")
    if not by_name and not symbol and not term:
        raise ValueError("give a symbol, a term, or both")
    if term and len(term) > MAX_TERM:
        raise ValueError(f"the term is longer than {MAX_TERM} characters")
    if company and len(company) > symbols.MAX_TEXT:
        raise ValueError(f"the company name is longer than {symbols.MAX_TEXT} characters")
    days = request.get("from_days_ago")
    days = DEFAULT_DAYS_BY_SOURCE.get(source, DEFAULT_DAYS) if days in (None, "") else days
    if isinstance(days, str) and days.strip().isdigit():
        days = int(days)
    if not isinstance(days, int) or isinstance(days, bool):
        raise ValueError("from_days_ago must be a whole number of days")
    if by_name:
        who = symbol or company
        things = "press releases" if source == "press_releases" else "SEC filings"
        query = (f"{term} {things} for {who}" if term else
                 f"What do {who}'s latest {things} say, and what do they mean for the stock? Is it a bullish or bearish opportunity?")
        return new_work_order(query, requester=request.get("requester", REQUESTER), subject=[symbol] if symbol else None, company=company, source=source,
                              action_type="search_filing" if source == "filings" else "search_news",  # entry 37: filings have their own action; press releases keep search_news (entry 34)
                              time_period=f"last {days} days", since_days=days, top_k=TOP_K, live=bool(request.get("fetch", True)),
                              criteria={"sentiment": "bullish or bearish"})
    live = bool(symbol) and bool(request.get("fetch", True))
    if symbol and term:
        query = f"{term} news for {symbol}"
    elif symbol:
        query = f"What is the latest news on {symbol}, and what does it mean for the stock? Is it a bullish or bearish opportunity?"
    else:
        query = term
    return new_work_order(query, requester=request.get("requester", REQUESTER), subject=[symbol] if symbol else term, company=company if symbol else None, source=source,
                          time_period=f"last {days} days", since_days=days, top_k=TOP_K, live=live, criteria={"sentiment": "bullish or bearish"})


def _timed(name: str, step: Callable[[DeskState], dict]) -> Callable[[DeskState], dict]:
    """A node that also records its name in `path` and how long it took in `seconds`."""
    def node(state: DeskState) -> dict:
        started = time.perf_counter()
        update = step(state)
        return {**update, "path": [name], "seconds": {name: round(time.perf_counter() - started, 1)}}
    return node


def build_desk(clerk_factory: Callable[[WorkOrder], DataClerk] | None = None, advisor_factory: Callable[[WorkOrder], Advisor] | None = None,
               with_advisor: bool = True, resolver: Callable[[str], symbols.Resolution] | None = None,
               with_funds: bool = False, funds_factory: Callable[[], AccountManager] | None = None):
    """The compiled graph. The factories make the agents for an order (fakes in tests); by default the Data Clerk may fetch only if the order is
    live, and the Advisor uses the advisor role's model. `resolver` turns a company name into a ticker (a fake in tests; the Finnhub lookup of
    prometheus/tools/symbols.py by default, found when it is called). `with_funds` adds the `account_manager_funds` node after the Advisor (entry 42); `funds_factory` makes its
    AccountManager (a fake model in tests). The request then carries `funds`: {"portfolio": a file path or a Portfolio, "shares": how many to buy, "buy": the symbol, optional}."""
    clerk_factory = clerk_factory or (lambda order: DataClerk(allow_fetch=order.live))
    advisor_factory = advisor_factory or (lambda order: Advisor())
    funds_factory = funds_factory or (lambda: AccountManager())

    def account_manager(state: DeskState) -> dict:
        request, resolution = state["request"], None
        typed = request.get("symbol")
        if (request.get("source") not in LOOKUP_SOURCES and isinstance(typed, str) and typed.strip()
                and not symbols.is_plain_ticker(typed)):  # a name, or a ticker in lower case: look it up, never guess. (Press releases and filings have their own exact lookup, in the Data Clerk)
            try:
                found = (resolver or symbols.resolve_company)(typed)
            except symbols.SymbolLookupError as error:
                return {"order": None, "errors": [str(error)], "resolution": None}
            resolution = {"typed": found.typed, "symbol": found.symbol, "company": found.company, "how": found.how}
            request = {**request, "symbol": found.symbol, "company": found.company}
        try:
            order = order_from_request(request)
        except ValueError as error:
            return {"order": None, "errors": [str(error)], "resolution": resolution}
        reasons = verify_request(order)
        order.errors = [*order.errors, *reasons]
        return {"order": order, "errors": reasons, "resolution": resolution}

    def data_clerk(state: DeskState) -> dict:
        order = state["order"]
        try:
            clerk = clerk_factory(order)
        except (RuntimeError, ValueError) as error:  # for example a missing key: the message names the setting, never a value
            order.status, order.errors = "data_failed", [*order.errors, f"the Data Clerk could not start: {error}"]
            return {"order": order}
        return {"order": clerk.request_data(order)}

    def advisor(state: DeskState) -> dict:
        order = state["order"]
        try:
            agent = advisor_factory(order)
        except (RuntimeError, ValueError) as error:
            order.errors = [*order.errors, f"the Advisor could not start: {error}"]
            return {"order": order}
        return {"order": agent.advise(order)}

    def account_manager_funds(state: DeskState) -> dict:
        """RaiseFunds for the best recommendation: never raises, and never changes the work order."""
        spec = state["request"].get("funds") or {}
        order = state["order"]
        target = spec.get("buy") or top_recommendation(order)
        if not target:
            return {"funds": {"status": "refused", "errors": ["the Advisor made no recommendation with a verified bullish score to buy, and no symbol was given"], "quotes": {}, "run": {}}}
        try:
            portfolio = spec["portfolio"] if not isinstance(spec.get("portfolio"), (str, Path)) else load_portfolio(spec["portfolio"])
            manager = funds_factory()
        except (KeyError, ValueError, OSError, RuntimeError) as error:  # no portfolio, a bad portfolio file, or a missing key: the message names the setting, never a value
            return {"funds": {"status": "refused", "errors": [f"the Account Manager could not start: {error}"], "quotes": {}, "run": {}}}
        return {"funds": manager.raise_funds(portfolio, target, spec.get("shares", 1))}

    def after_advisor(state: DeskState) -> str:
        return "account_manager_funds" if with_funds and state["order"].status == "recommended" else END

    def after_account_manager(state: DeskState) -> str:
        return "data_clerk" if state.get("order") is not None and not state.get("errors") else END

    def after_data_clerk(state: DeskState) -> str:
        return "advisor" if with_advisor and state["order"].status == "data_ready" else END

    graph = StateGraph(DeskState)
    graph.add_node("account_manager", _timed("account_manager", account_manager))
    graph.add_node("data_clerk", _timed("data_clerk", data_clerk))
    graph.add_node("advisor", _timed("advisor", advisor))
    if with_funds:
        graph.add_node("account_manager_funds", _timed("account_manager_funds", account_manager_funds))
    graph.add_edge(START, "account_manager")
    graph.add_conditional_edges("account_manager", after_account_manager, {"data_clerk": "data_clerk", END: END})
    graph.add_conditional_edges("data_clerk", after_data_clerk, {"advisor": "advisor", END: END})
    if with_funds:
        graph.add_conditional_edges("advisor", after_advisor, {"account_manager_funds": "account_manager_funds", END: END})
        graph.add_edge("account_manager_funds", END)
    else:
        graph.add_edge("advisor", END)
    return graph.compile()


def run_desk(request: dict, **options) -> DeskState:
    """Run one request through the desk. `options` are those of `build_desk`. Returns the final state: the order, why a request was refused, the
    path through the graph and the seconds each node took."""
    return build_desk(**options).invoke({"request": request, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}})


def report(state: DeskState, total_seconds: float | None = None) -> str:
    """What the user reads: the request, the path, the answer with its sources, the recommendations with every signal, and the timings."""
    lines = []
    request, order = state["request"], state.get("order")
    source = request.get("source") or DEFAULT_SOURCE
    lines.append(f"REQUEST  symbol {request.get('symbol') or '-'} | term {request.get('term') or '-'} | source {source} "
                 f"| from {request.get('from_days_ago') or DEFAULT_DAYS_BY_SOURCE.get(source, DEFAULT_DAYS)} days ago"
                 + (f" | company {request['company']}" if request.get("company") else ""))
    if state.get("resolution"):
        r = state["resolution"]
        lines.append(f"RESOLVED {r['typed']!r} -> {r['symbol']} ({r['company']}), looked up and matched as a {r['how']}")
    lines.append("PATH     " + " -> ".join(state.get("path", [])) + ("" if order is not None and order.status == "recommended" else "  (stopped here)"))
    if order is None or state.get("errors"):
        lines.append("\nREFUSED  " + "; ".join(state.get("errors") or ["no work order could be built"]))
        return "\n".join(lines)
    d = order.data
    lines.append(f"ORDER    {order.request_id} | status {order.status} | {'LIVE (fetching on)' if order.live else 'not live (fetching off)'} | window last {order.since_days} days "
                 f"| subject {order.subject}" + (f" ({order.company})" if order.company else ""))
    lines.append(f"QUERY    {order.query}")
    if order.resolved:  # a press release or filings order: what the exact lookup decided (the symbol first, else the company name)
        r = order.resolved
        lines.append(f"RESOLVED {'press releases' if order.source == 'press_releases' else order.source} for {r['symbol']} ({r['company']}), found by its {'symbol' if r['how'] == 'ticker' else 'company name'} "
                     f"(typed: symbol {r['typed_symbol'] or '-'}, company {r['typed_company'] or '-'})")
    if order.errors:
        lines.append("ERRORS   " + "; ".join(order.errors))
    if d:
        lines.append("TOOLS    " + (", ".join(f"{c['name']}({c['outcome']})" for c in d.tool_calls) or "none"))
        for c in d.tool_calls:
            lines.append(f"           {c['name']}: " + ", ".join(f"{k}={v!r}" for k, v in c["args"].items()))
        lines.append(f"DATA     {len(d.sources)} sources shown, {len(d.references)} cited" + (" | NO SUPPORTING TEXT" if d.no_supporting_text else ""))
        if d.results:  # what the next agent finds on the order: the results of each source (entry 39)
            lines.append("RESULTS  " + " | ".join(f"{corpus.replace('_', ' ')}: {len(group['shown'])} shown, {len(group['cited'])} cited" + (f", {len(group['swept'])} swept" if group.get("swept") else "")
                                                    for corpus, group in d.results.items()))
        if d.sweep:  # who searched each of the three sources and the evidence: the model, or the Data Clerk's code afterwards (entry 40)
            lines.append("SWEEP    " + " | ".join(f"{corpus.replace('_', ' ')}: {how['by']}, {how['sources']}" + (f" ({how['note'][:60]})" if how["note"] and how["sources"] == 0 else "") for corpus, how in d.sweep.items()))
        for symbol, evidence in d.filing_evidence.items():
            lines.append(f"EVIDENCE filings of {symbol}: {len(evidence['evidence'])} passages, balance sheet {'read' if evidence['balance_sheet'] else 'not available'}, "
                         f"{len(evidence['gaps'])} gap(s) (scores nothing: the Advisor does)")
        lines.append("\nDATA CLERK ANSWER\n" + d.answer)
        for r in d.references:
            lines.append(f"  [{r['number']}] {r['publisher']}, {r['published_utc'][:10]}: {r['headline'][:80]}  (score {r['score']:.2f})")
        for note in d.notes:
            lines.append(f"  note: {note}")
    if order.recommendations:
        lines.append("\nADVISOR RECOMMENDATIONS")
        for r in order.recommendations:
            score = r["score"] if r["supported"] else "none (no verified signal)"
            lines.append(f"  {r['symbol']:<7} {r['company'] or '':<22} stance {r['stance'] or '?':<8} calculated score {score!s:<26} (model's own score {r['model_score']})")
            for s in r["signals"]:
                mark = "FAIL" if not s["verified"] else "OK  " if s.get("counted") else "SAME"
                lines.append(f"            {mark} [{s['source']}] {s['direction']}: {s['quote'][:96]!r}" + ("" if s["verified"] else f"   <- {s['reason']}"))
    elif order.status == "recommended":
        lines.append("\nADVISOR  no recommendation: nothing in the sources gave a signal for a company")
    if state.get("funds"):
        lines.append("\nACCOUNT MANAGER: RAISE FUNDS (a Tree of Thought, entry 42)\n" + funds_report(state["funds"]))
    seconds = state.get("seconds", {})
    parts = "  ".join(f"{name} {seconds[name]:.1f}" for name in ("account_manager", "data_clerk", "advisor", "account_manager_funds") if name in seconds)
    lines.append(f"\nTIMING (seconds)  {parts}" + (f"  | total {total_seconds:.1f}" if total_seconds is not None else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run one request through the desk: Account Manager, Data Clerk, Advisor, on one work order.")
    parser.add_argument("--symbol", default="", help='a ticker such as MSFT, or a company name such as "Procter & Gamble" (looked up, never guessed); fetches fresh news with the PAID APIs unless --no-fetch')
    parser.add_argument("--term", default="", help='a topic, for example "flying car" (no symbol: searches the stored news, fetches nothing)')
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"what to search: {', '.join(WIRED_SOURCES)} (default {DEFAULT_SOURCE!r})")
    parser.add_argument("--company", default="", help='press_releases and filings only: a company name such as "Apple Inc", the argument that is used when the symbol is not given or not found. '
                        "matched on its words against Finnhub's name: case, punctuation and Inc/Corp/Co/The are ignored, a typo is NOT matched; the symbol wins when both are given")
    parser.add_argument("--from-days-ago", default=None, help=f"how many days back (default {DEFAULT_DAYS} for news, 1 to 90; "
                        f"{DEFAULT_DAYS_BY_SOURCE['press_releases']} for press_releases and filings, 1 to 730)")
    parser.add_argument("--no-fetch", action="store_true", help="never fetch: search the stored news only, even with a symbol")
    parser.add_argument("--no-advisor", action="store_true", help="stop after the Data Clerk")
    parser.add_argument("--model", help="an OpenRouter model id for the Advisor instead of ADVISOR_MODEL")
    parser.add_argument("--raise-funds", action="store_true", help="after the Advisor, the Account Manager raises the money to buy the best recommendation from the portfolio (a Tree of Thought, "
                        "entry 42; it calls Finnhub for quotes and OpenRouter for the model)")
    parser.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO), help="--raise-funds: the portfolio JSON file (default: the sample portfolio of Jacob Pyrett)")
    parser.add_argument("--buy-shares", type=int, default=1, help="--raise-funds: how many shares of the recommended stock to buy (default 1)")
    parser.add_argument("--buy", default="", help="--raise-funds: buy this symbol instead of the Advisor's best recommendation")
    parser.add_argument("--json", action="store_true", help="print the finished work order as JSON instead of the report")
    parser.add_argument("--show-graph", action="store_true", help="print the graph as a Mermaid diagram and stop")
    args = parser.parse_args(argv)
    if args.show_graph:
        print(build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None, with_funds=args.raise_funds).get_graph().draw_mermaid())
        return 0
    days = args.from_days_ago if args.from_days_ago is not None else str(DEFAULT_DAYS_BY_SOURCE.get(args.source, DEFAULT_DAYS))
    request = {"symbol": args.symbol, "term": args.term, "source": args.source, "from_days_ago": days, "fetch": not args.no_fetch}
    if args.company:
        request["company"] = args.company  # only when given: a news request is exactly the four parameters it always was
    options = {"with_advisor": not args.no_advisor}
    if args.raise_funds:
        options["with_funds"] = True
        request["funds"] = {"portfolio": args.portfolio, "shares": args.buy_shares, **({"buy": args.buy} if args.buy else {})}
    if args.model:
        options["advisor_factory"] = lambda order: Advisor(model_id=args.model)
    state = run_desk(request, **options)
    total = time.perf_counter() - started
    order = state.get("order")
    if args.json:
        print(json.dumps({"request": request, "resolution": state.get("resolution"), "errors": state.get("errors"), "path": state.get("path"), "seconds": state.get("seconds"),
                          "funds": state.get("funds"), "order": order.model_dump() if order else None}, indent=1, default=str))
    else:
        print(report(state, total))
    if order is not None and not state.get("errors"):
        # the symbol and the term both go in the name, so that a run with a symbol and a term does not overwrite a run with the symbol alone
        named = (state.get("resolution") or {}).get("symbol") or (order.resolved or {}).get("symbol") or request["symbol"]  # a looked-up name is saved under its ticker
        parts = (named, request["term"], "" if request["source"] == DEFAULT_SOURCE else request["source"])  # press releases are saved apart from news
        label = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", part.strip().lstrip("$")).strip("_") for part in parts if part.strip()) or "order"
        out = OUT_DIR / f"desk_{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"request": request, "resolution": state.get("resolution"), "path": state.get("path"), "seconds": state.get("seconds"), "funds": state.get("funds"), "order": order.model_dump()}, indent=1, default=str),
                       encoding="utf-8")
        if not args.json:
            print(f"saved: {out.relative_to(OUT_DIR.parents[1]) if out.is_relative_to(OUT_DIR.parents[1]) else out}")
    done = order is not None and not state.get("errors") and order.status == ("data_ready" if args.no_advisor else "recommended")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
