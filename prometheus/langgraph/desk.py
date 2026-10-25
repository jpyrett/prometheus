"""The desk: a LangGraph that passes ONE work order from agent to agent (design choices entry 29).

    python -m prometheus.langgraph.desk --symbol MSFT --from-days-ago 14        # a symbol: fetch fresh news, then answer and recommend
    python -m prometheus.langgraph.desk --term "flying car"                     # a term, no symbol: search the stored news
    python -m prometheus.langgraph.desk --symbol SNDK --term "long-term contracts"   # a term inside one symbol's news
    ./test_prometheus.sh                                                        # the same, with prompts for the four parameters

The graph:

    START -> account_manager -> data_clerk -> advisor -> END
                   |                |
                   |                +-- the data is not ready (failed, or no search was made) --> END
                   +-- the request is not valid --> END

Each node is one agent of the week 2 design, reduced to what exists today, and each one reads and updates the SAME work order, which is the
state of the graph (the design: "the saved WORK_ORDER is the same object every agent reads"):

  account_manager  builds the work order from the four parameters and checks it (`verify_request`). It is the Account Manager's `parse_input`
                   and `create_work_order`, without the trading half.
  data_clerk       `DataClerk.request_data(order)`: fetches if the order is live, searches, answers, puts the data and the sources on the order.
  advisor          `Advisor.advise(order)`: reads the chunks on the order and appends recommendations with a calculated score.

THE FOUR PARAMETERS (what the user types, and what each becomes):

  symbol         a ticker, for example MSFT, OR A COMPANY NAME, for example "Procter & Gamble". A ticker in capitals is used as it is. Anything else
                 (a name, or a ticker in lower case) is LOOKED UP (Finnhub's symbol search, prometheus/tools/symbols.py), never guessed: the answer is
                 shown ("Procter&Gamble" -> PG, Procter & Gamble Co), carried on the work order as `company`, and an ambiguous name is refused.
                 With a symbol the order is LIVE: fresh news is fetched for it with the PAID APIs (and written to news/<SYMBOL>.txt) unless
                 fetching is turned off.
  term           a topic, for example "flying car". With no symbol it is searched in the stored news and nothing is fetched. With a symbol the
                 term narrows the question inside that symbol's news (fetching is still for the whole symbol, never filtered by the term, so the
                 file stays complete).
  source         where the news comes from. Only `news` is wired, so it is the default and the only value accepted.
  from_days_ago  how many days back (the `from_date` of the fetch and the window of the search). Default 7, from 1 to 90.

At least one of symbol and term is required. The Advisor's query is the work order built from these, not a free sentence.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.data_clerk.agent import DataClerk
from prometheus.common.work_order import WIRED_SOURCES, WorkOrder, new_work_order, verify_request
from prometheus.rag.loaders import check_symbol
from prometheus.tools import symbols

DEFAULT_SOURCE = WIRED_SOURCES[0]
DEFAULT_DAYS = 7
TOP_K = 10
MAX_TERM = 200
REQUESTER = "desk"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"  # where a finished order is saved (tests point it elsewhere)


class DeskState(TypedDict, total=False):
    request: dict                                              # the four parameters as given
    order: WorkOrder | None                                    # THE work order: every node reads and updates this one object
    errors: list[str]                                          # why the request was refused (no order could be built, or it is not valid)
    resolution: dict | None                                    # a company name that was looked up: what was typed, the ticker, the company, how it matched
    path: Annotated[list[str], lambda old, new: old + new]     # the nodes that ran, in order
    seconds: Annotated[dict, lambda old, new: {**old, **new}]  # how long each node took


def order_from_request(request: dict) -> WorkOrder:
    """The four parameters -> a work order. Raises ValueError, with a message for the user, if there is no symbol and no term or a value is wrong.

    symbol only    "What is the latest news on MSFT ...?"        subject [MSFT], live
    term only      the term itself                                 subject the term, not live (there is no ticker to fetch)
    both           "<term> news for <symbol>"                     subject [MSFT], live: the term narrows the question inside that symbol
    """
    symbol, term = request.get("symbol"), request.get("term")
    symbol = check_symbol(symbol.strip().lstrip("$")) if isinstance(symbol, str) and symbol.strip() else None
    term = term.strip() if isinstance(term, str) and term.strip() else None
    if not symbol and not term:
        raise ValueError("give a symbol, a term, or both")
    if term and len(term) > MAX_TERM:
        raise ValueError(f"the term is longer than {MAX_TERM} characters")
    source = request.get("source") or DEFAULT_SOURCE
    days = request.get("from_days_ago")
    days = DEFAULT_DAYS if days in (None, "") else days
    if isinstance(days, str) and days.strip().isdigit():
        days = int(days)
    if not isinstance(days, int) or isinstance(days, bool):
        raise ValueError("from_days_ago must be a whole number of days")
    company = request.get("company") if isinstance(request.get("company"), str) and request.get("company").strip() else None
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
               with_advisor: bool = True, resolver: Callable[[str], symbols.Resolution] | None = None):
    """The compiled graph. The factories make the agents for an order (fakes in tests); by default the Data Clerk may fetch only if the order is
    live, and the Advisor uses the advisor role's model. `resolver` turns a company name into a ticker (a fake in tests; the Finnhub lookup of
    prometheus/tools/symbols.py by default, found when it is called)."""
    clerk_factory = clerk_factory or (lambda order: DataClerk(allow_fetch=order.live))
    advisor_factory = advisor_factory or (lambda order: Advisor())

    def account_manager(state: DeskState) -> dict:
        request, resolution = state["request"], None
        typed = request.get("symbol")
        if isinstance(typed, str) and typed.strip() and not symbols.is_plain_ticker(typed):  # a name, or a ticker in lower case: look it up, never guess
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

    def after_account_manager(state: DeskState) -> str:
        return "data_clerk" if state.get("order") is not None and not state.get("errors") else END

    def after_data_clerk(state: DeskState) -> str:
        return "advisor" if with_advisor and state["order"].status == "data_ready" else END

    graph = StateGraph(DeskState)
    graph.add_node("account_manager", _timed("account_manager", account_manager))
    graph.add_node("data_clerk", _timed("data_clerk", data_clerk))
    graph.add_node("advisor", _timed("advisor", advisor))
    graph.add_edge(START, "account_manager")
    graph.add_conditional_edges("account_manager", after_account_manager, {"data_clerk": "data_clerk", END: END})
    graph.add_conditional_edges("data_clerk", after_data_clerk, {"advisor": "advisor", END: END})
    graph.add_edge("advisor", END)
    return graph.compile()


def run_desk(request: dict, **options) -> DeskState:
    """Run one request through the desk. `options` are those of `build_desk`. Returns the final state: the order, why a request was refused, the
    path through the graph and the seconds each node took."""
    return build_desk(**options).invoke({"request": request, "order": None, "errors": [], "resolution": None, "path": [], "seconds": {}})


def report(state: DeskState, total_seconds: float | None = None) -> str:
    """What the user reads: the request, the path, the answer with its sources, the recommendations with every signal, and the timings."""
    lines = []
    request, order = state["request"], state.get("order")
    lines.append(f"REQUEST  symbol {request.get('symbol') or '-'} | term {request.get('term') or '-'} | source {request.get('source') or DEFAULT_SOURCE} "
                 f"| from {request.get('from_days_ago') or DEFAULT_DAYS} days ago")
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
    if order.errors:
        lines.append("ERRORS   " + "; ".join(order.errors))
    if d:
        lines.append("TOOLS    " + (", ".join(f"{c['name']}({c['outcome']})" for c in d.tool_calls) or "none"))
        for c in d.tool_calls:
            lines.append(f"           {c['name']}: " + ", ".join(f"{k}={v!r}" for k, v in c["args"].items()))
        lines.append(f"DATA     {len(d.sources)} sources shown, {len(d.references)} cited" + (" | NO SUPPORTING TEXT" if d.no_supporting_text else ""))
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
    seconds = state.get("seconds", {})
    parts = "  ".join(f"{name} {seconds[name]:.1f}" for name in ("account_manager", "data_clerk", "advisor") if name in seconds)
    lines.append(f"\nTIMING (seconds)  {parts}" + (f"  | total {total_seconds:.1f}" if total_seconds is not None else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run one request through the desk: Account Manager, Data Clerk, Advisor, on one work order.")
    parser.add_argument("--symbol", default="", help='a ticker such as MSFT, or a company name such as "Procter & Gamble" (looked up, never guessed); fetches fresh news with the PAID APIs unless --no-fetch')
    parser.add_argument("--term", default="", help='a topic, for example "flying car" (no symbol: searches the stored news, fetches nothing)')
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"only {DEFAULT_SOURCE!r} is wired")
    parser.add_argument("--from-days-ago", default=str(DEFAULT_DAYS), help=f"how many days back (default {DEFAULT_DAYS}, 1 to 90)")
    parser.add_argument("--no-fetch", action="store_true", help="never fetch: search the stored news only, even with a symbol")
    parser.add_argument("--no-advisor", action="store_true", help="stop after the Data Clerk")
    parser.add_argument("--model", help="an OpenRouter model id for the Advisor instead of ADVISOR_MODEL")
    parser.add_argument("--json", action="store_true", help="print the finished work order as JSON instead of the report")
    parser.add_argument("--show-graph", action="store_true", help="print the graph as a Mermaid diagram and stop")
    args = parser.parse_args(argv)
    if args.show_graph:
        print(build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None).get_graph().draw_mermaid())
        return 0
    request = {"symbol": args.symbol, "term": args.term, "source": args.source, "from_days_ago": args.from_days_ago, "fetch": not args.no_fetch}
    options = {"with_advisor": not args.no_advisor}
    if args.model:
        options["advisor_factory"] = lambda order: Advisor(model_id=args.model)
    state = run_desk(request, **options)
    total = time.perf_counter() - started
    order = state.get("order")
    if args.json:
        print(json.dumps({"request": request, "resolution": state.get("resolution"), "errors": state.get("errors"), "path": state.get("path"), "seconds": state.get("seconds"),
                          "order": order.model_dump() if order else None}, indent=1, default=str))
    else:
        print(report(state, total))
    if order is not None and not state.get("errors"):
        # the symbol and the term both go in the name, so that a run with a symbol and a term does not overwrite a run with the symbol alone
        named = (state.get("resolution") or {}).get("symbol") or request["symbol"]  # a looked-up name is saved under its ticker
        label = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", part.strip().lstrip("$")).strip("_") for part in (named, request["term"]) if part.strip()) or "order"
        out = OUT_DIR / f"desk_{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"request": request, "resolution": state.get("resolution"), "path": state.get("path"), "seconds": state.get("seconds"), "order": order.model_dump()}, indent=1, default=str),
                       encoding="utf-8")
        if not args.json:
            print(f"saved: {out.relative_to(OUT_DIR.parents[1]) if out.is_relative_to(OUT_DIR.parents[1]) else out}")
    done = order is not None and not state.get("errors") and order.status == ("data_ready" if args.no_advisor else "recommended")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
