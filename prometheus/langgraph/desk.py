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

    A NEUTRAL RECOMMENDATION IS AN APPROVAL (entries 91 and 107; `ALLOW_NEUTRAL_RECOMMENDATIONS=true`): when the Advisor recommended nothing for the Critic to review but scored a company NEUTRAL (whoever named it), the graph goes from the
    advisor straight to account_manager_decision (advisor -> account_manager_decision -> trader), where the Account Manager treats the neutral company as approved and decides it; the node passes the number of companies the request asked for
    (`plan["count"]`, None when it gave none) and the Account Manager buys at most that many, the Critic's approved ones first.

    with_critic=True (Critic v1, entries 56 to 59; `--critic`):
    START -> account_manager -> data_clerk -> advisor -> critic -> account_manager_decision -> END     (or -> account_manager_funds -> END with with_funds)
                                                  |
                                                  +-- no recommendations --> END

    with_trader=True (Trader v1, entries 73 to 75; `--trade`; needs the Critic):
    ... -> critic -> account_manager_decision -> trader -> END            (with_funds: ... -> account_manager_decision -> account_manager_funds -> trader -> END)
                                  |                                       The trader node runs only when at least one decision says yes_auto or yes_user; otherwise the graph ends before it.
                                  +-- nothing is permitted --> END

    A DIRECT ORDER (entries 87 and 88: the user names the trade, "Buy 50 shares of MSFT" or "of Microsoft"; only with the Critic's desk, `with_critic=True`, which is the one that decides and trades) skips the research, the Advisor and the Critic:
    START -> account_manager -> account_manager_direct -> [account_manager_funds ->] trader -> END                (the trader node runs only when the decision says yes_user; the funds node only when a direct buy the cash does not cover is permitted to be paid for by selling holdings, entry 142; the Data Clerk's node, the Advisor and the Critic are never run: the Account Manager asks the Data Clerk for each
                                                                                       company name and symbol itself and appends them to the work order when it makes it)
    A STOCK THE DATA CLERK DID NOT FIND (entries 101 and 102): "Google" is found as its listed owner (Alphabet Inc., GOOGL); a name that is not listed ("Chick-fil-a": privately held) is, when others were found, DROPPED from the order
    without a question in AUTO mode and asked about one by one in ASK mode (the `approver` carries the question; the Account Manager's model words it, code checks it and reads the yes or no); a blocking issue (nothing found, a lookup that
    failed, two orders for one stock), a no, or no answer ends the run at the first node with the reasons, and its last message asks the user to clarify the request (`direct_order.CLARIFY`).

Each node is one agent of the week 2 design, reduced to what exists today, and each one reads and updates the SAME work order, which is the
state of the graph (the design: "the saved WORK_ORDER is the same object every agent reads"):

  account_manager  builds the work order from the four parameters and checks it (`verify_request`). It is the Account Manager's `parse_input`
                   and `create_work_order`, without the trading half.
  data_clerk       `DataClerk.request_data(order)`: fetches if the order is live, searches, answers, then (entry 40) searches the stored sources the model did not search and reads the filing
                   evidence, so the order carries all three; puts the data and the sources on the order.
  advisor          `Advisor.advise(order)`: reads the chunks, `data.results` and `data.filing_evidence` on the order and appends recommendations with a calculated score.
  critic           (only with `build_desk(with_critic=True)`, entries 56 to 59) the Advisor then runs `generate_recommendations` (given NO portfolio: it recommends purchases only, entry 105) and the Critic approves or denies each one (a review that
                   wants a person to look is DENIED, entry 103; it judges recommendations to buy and nothing about the holdings): it asks the
                   Data Clerk for the numbers, a quote and the sources again, judges the company's risk against the user's risk tolerance, timing and value, and puts a review beside each recommendation (`order.reviews`).
                   It is given the profile's `RiskView` and no portfolio and no money: it sizes nothing (entry 80).
  account_manager_decision  (only with the Critic) the Account Manager turns each verdict into a trade decision (`order.trade_decisions`): an APPROVED recommendation trades on its own in Auto mode (nothing is asked) or is asked about in
                   Ask mode, a DENIED one is not traded and is returned to the user with the Critic's reason, and it decides HOW MUCH to buy (`decisions.decide_trades`, `sizing.py`: from the user's money; Auto spends the funds round-robin; entry 104).
  account_manager_funds  (only with `build_desk(with_funds=True)`, entry 42) the Account Manager's RaiseFunds: it takes the best recommendation (or the stock the user named), asks the Data Clerk for the
                   current prices, and runs a Tree of Thought for what to sell from the user's portfolio to buy it. Its answer is `state["funds"]`; the work order is not changed.
  account_manager_direct  (only with the Critic's desk, entries 87 and 88) a DIRECT ORDER: the user named the trade, so no recommendation is made and the Critic is not asked, and the Data Clerk's NODE is skipped too:
                   the `account_manager` node itself asked the Data Clerk for the company name and the symbol of each stock the user named (a ticker or a company's name) and APPENDED them to the work order
                   (`AccountManager.name_the_stocks`: `data.companies` and the user's own orders in `direct_orders`). This node asks the Data Clerk for the current price of each symbol, decides each order
                   (`AccountManager.decide_direct`: the user's own order is the approval, `yes_user`, `origin: "direct"`; for a company the user NAMED, Ask mode asks them to confirm the symbol, the company, the shares and the amount, and Auto mode
                   does not ask; it is placed as typed or not at all; the profile's limits never apply) and the order goes on to the Trader.
  account_manager_funds  RAISING FUNDS FOR A PURCHASE THE CASH DOES NOT COVER (entry 142; `funding.py`, design/direct_raise_funds_spec.md). On every desk that has the Critic AND the Trader, a decision that carries `funding` (the user permitted selling holdings
                   to pay for it: "raise funds if needed" in their words, the setting ALWAYS_RAISE_FUNDS, or their yes in Ask mode; a direct order's typed shares, the dollars of a direct order, or the shares typed with a recommendation) sends the run
                   through this node on its way to the Trader: `AccountManager.fund_decisions` runs RaiseFunds ONCE for the whole shortfall and its answer is `state["funds"]`, the Trader's funding plan; when it cannot pay for the purchase the decision is `no_funds`
                   and the Trader is not run. Without such a decision the run never touches it. (With `with_funds`, the older flag of entry 42, the node also plans the best recommendation as it always did.)
  trader           (only with `build_desk(with_trader=True)`, Trader v1, entries 73 to 75) the Account Manager hands the decided order to the Trader (`AccountManager.perform_trades`): it executes the permitted trades against a
                   SIMULATED broker, on paper (the current quote) or as a backtest (the stored bars), and the Account Manager approves each trade against the funds it holds and records each fill as it happens, one trade at a time (entry 81: the Trader holds no cash). With the funds node the RaiseFunds plan arrives as the funding
                   plan (its sells need the user's yes). It never raises, and it never edits the Advisor's records, the Critic's reviews or the decisions; it writes `order.trades` and `order.trader_runs` and sets `traded`.

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

MONEY IS `decimal.Decimal` (design choices entry 112): the plan's budget is the `Money` the user's words gave (`intent.budget`), the funds node hands RaiseFunds a whole `Decimal` number of shares (`shares.ceil` of a fractional purchase), and what `--json` and the saved
order write is written by the one encoder (`money.dumps`): a Decimal is a JSON number, never a string.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from prometheus.agents.account_manager import direct as direct_order
from prometheus.agents.account_manager import funding as funding_module
from prometheus.agents.account_manager import neutral as neutral_names
from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO, DEFAULT_PROFILE, AccountManager, report as funds_report, top_recommendation
from prometheus.agents.account_manager.decisions import terminal_approver
from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.critic.critic import Critic
from prometheus.agents.critic.report import review_section
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.trader.report import trade_section
from prometheus.agents.trader.trader import Trader
from prometheus.application.recommendations import Request, Result, classify, find_companies_of, scored_symbols
from prometheus.common import shares as sh
from prometheus.common.market_calendar import eastern_date
from prometheus.common.money import dumps
from prometheus.common.trading_clock import clock_note, trading_now
from prometheus.common.work_order import LOOKUP_SOURCES, REQUEST_LABEL, TRADING, WIRED_SOURCES, WorkOrder, new_work_order, verify_request
from prometheus.common.portfolio import Portfolio, load_portfolio
from prometheus.rag.loaders import check_symbol
from prometheus.tools import symbols
from prometheus.common import langsmith_tracing
from prometheus.security.guard import Guard

DEFAULT_SOURCE = WIRED_SOURCES[0]
DEFAULT_DAYS = 7
DEFAULT_DAYS_BY_SOURCE = {"news": 7, "press_releases": 365, "filings": 365}  # a company issues a few press releases or filings a month: a week would usually find none (entries 34, 35)
TOP_K = 10
ASK_NEWS_DAYS = 14   # a request in the user's own words looks at two weeks of news unless it says otherwise (the runner's own default)
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
    with_trader: bool                                          # the desk was built with the Trader: its path ends at the trader node, and not at the decision (`run_desk` sets it)
    clarify: bool                                              # the run stopped because the request could not be read or its stocks could not be made (entry 102): its last message asks the user to clarify it (`direct.CLARIFY`)
    plan: dict | None                                          # a request in the user's own words (`request["ask"]`): what the Account Manager understood and the companies the Data Clerk found (entry 78)
    security: dict | None                                      # the security layer's verdict on the user's request when it was refused or flagged (entry 116): the action, the risk, the rules; an ALLOW leaves it out


CRITIC_ENDS = ("reviewed", "decided")   # where a run with the Critic and without the Trader ends: `decided`, or `reviewed` while a decision waits for the user's answer


def reached_its_end(state: DeskState) -> bool:
    """Whether the run went as far as the desk it was built as goes: what the report calls a finished PATH and what the exit code of a run without `--trade` means (the exit code of a run with `--trade` is its own, in `main`).

    Without the Critic the path ends at the Advisor, `recommended`. With the Critic and no Trader it ends at the Account Manager's decision: `decided`, or `reviewed` while a decision waits for the user (it is the Critic path's
    normal end, not a stop). With the Trader it ends at the Trader (`traded`; a run of the Trader that left a trade resting or a decision pending also ran to its node); a decided order the Trader had nothing to do with did
    not reach it. When the Advisor recommended nothing the Critic is never asked, the order stays `recommended` and the path is finished. An order that stays `recommended` AFTER the Critic ran (it could not start, say) is
    not finished: the verdicts and the decisions are missing."""
    order, path = state.get("order"), state.get("path", [])
    if order is None:
        return False
    if order.status == "traded" or "trader" in path:
        return True
    trader_built = bool(state.get("with_trader")) or (state.get("request") or {}).get("trade") is not None   # `trade` is in the request when `--trade` was typed
    if "account_manager_direct" in path:   # a direct order (entry 87) ends at the Account Manager's decision, as the Critic's path does; a decided order the Trader had nothing to do with did not reach it
        return not trader_built and order.status == "decided"
    if "critic" in path:
        return not trader_built and order.status in CRITIC_ENDS
    return order.status == "recommended"


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


def open_book():
    """The Postgres account book of the local container (`secrets/postgres_local.txt`). Imported here and not at the top of the file, so that a desk that never uses a saved account never loads the Postgres driver."""
    from prometheus.persistence.accounts import open_account_store
    return open_account_store()


def build_desk(clerk_factory: Callable[[WorkOrder], DataClerk] | None = None, advisor_factory: Callable[[WorkOrder], Advisor] | None = None,
               with_advisor: bool = True, resolver: Callable[[str], symbols.Resolution] | None = None,
               with_funds: bool = False, funds_factory: Callable[[], AccountManager] | None = None,
               with_critic: bool = False, critic_factory: Callable[[WorkOrder], Critic] | None = None, approver: Callable[[dict], bool | None] | None = None,
               clock: Callable[[], datetime] | None = None,
               with_trader: bool = False, trader_factory: Callable[[WorkOrder], Trader] | None = None, store_factory: Callable[[], object] | None = None,
               confirm: Callable[[object], bool | None] | None = None, clarify: Callable[[dict], str | None] | None = None, guard: Guard | None = None):
    """The compiled graph. The factories make the agents for an order (fakes in tests); by default the Data Clerk may fetch only if the order is
    live, and the Advisor uses the advisor role's model. `resolver` turns a company name into a ticker (a fake in tests; the Finnhub lookup of
    prometheus/tools/symbols.py by default, found when it is called). `with_funds` adds the `account_manager_funds` node after the Advisor (entry 42); `funds_factory` makes its
    AccountManager (a fake model in tests). The request then carries `funds`: {"portfolio": a file path or a Portfolio, "shares": how many to buy, "buy": the symbol, optional}.

    `clarify` (design choices entry 85) is the channel to the user for the Account Manager's questions about how much to buy: a callable that takes a card (the question, which one it is, the facts for the first)
    and returns what the user typed, or None. In Ask mode, when the request did not already say how much, the decision node asks (`AccountManager.decide_trades`) BEFORE it sizes; Auto mode never asks, and with no
    `clarify` nothing is asked.

    `with_critic` (Critic v1, entries 56 to 59) puts two nodes after the Advisor, `critic` and `account_manager_decision`, and makes the Advisor run `generate_recommendations` instead of `advise` (the Critic
    reads schema 2 records). The request then carries `account`: {"portfolio": a path or a Portfolio, "profile": a path or a UserProfile, "mode": "ask" or "auto", all optional: the sample files and the
    profile's own mode}. `critic_factory` makes the Critic for an order (fakes in tests), `approver` is asked about each recommendation that needs the user (Ask mode; None: it stays pending), and `clock`
    returns the aware moment everything is dated and judged at (a fake in tests; the real clock by default). The date a recommendation is made on is the US EASTERN date of that moment, never the UTC one.

    `guard` (entry 116, design/agentic_security_spec.md) is the security layer's door for the user's own words: the `account_manager` node screens `request["ask"]["text"]` BEFORE the Account Manager's model reads it (the rate limit first,
    then the code rules and, for the real parts, the LLM Risk Scorer). A BLOCK or a REDACT_OR_REVIEW (held for review: it fails closed in both modes) ends the run at the first node with the policy's message in `errors`, no model called and
    `state["security"]` set; a request that screens ALLOW or ALLOW_AND_FLAG runs exactly as before. None is `Guard.code_only()` (rules and router, no model call); the session and the web page pass `Guard.with_scorer()`.
    The Trust Score of each recommendation (`guard.trust`) is worked out after the Critic's reviews: recorded and traced, it changes nothing.

    `with_trader` (Trader v1, entries 73 to 75) adds the `trader` node after the decisions (after the funds node with `with_funds`): the Trader executes what is permitted against a simulated broker. It needs `with_critic`
    (it trades what the Critic reviewed and the Account Manager decided on) and raises ValueError without it. The node runs only when at least one decision says `yes_auto` or `yes_user`. The request then carries
    `trade`: {"mode": "paper" or "backtest", "book": True to save the fills in the account, "model": False for no model (the sweep places the default orders), all optional}. `trader_factory` makes the Trader for an
    order (a fake or a Trader over a memory ledger in tests; used as it is); without one the node builds `Trader(mode=..., use_model=...)` from the request.

    The request's `account` may also carry `account`: a sample key or an account number of the Postgres account book (`--account`). The profile, the portfolio and the booking then come from that account, not from the JSON
    files: the manager is connected to it first (`AccountManager.connect`), a profile that was given is set after, and the portfolio is the saved one. `store_factory` makes the store (`open_book`, the local
    container, by default; a fake in tests). A database problem is a RuntimeError that names the kind of failure and never a connection detail, so each node reports it as it reports a bad profile file.

    A REQUEST IN THE USER'S OWN WORDS (design choices entry 78). `request["ask"]` = {"text": what the user typed, "default_count": how many companies when the text does not say, "fetch": refresh the data with the PAID
    APIs, "news_days": 14} starts the SAME graph with the Account Manager reading the text: its model fills in a form (`AccountManager.understand_request`) that code checks against the words, `confirm` (a callable
    given the form: True to go on, anything else stops the desk with no work done) shows it to the user, and the work order is made. The Data Clerk node then finds the companies of a theme, a list of names or tickers
    (`recommendations.find_companies_of`, the runner's own step), refreshes their data when asked, and answers the work order; the Advisor node ranks them and keeps the top N (`generate_recommendations`, case industry
    for a theme, funds for tickers); a request only to LOOK ends there; otherwise the Critic reviews each recommendation against the profile, the decision node applies the user's "split it evenly" and "spend this much"
    (`decide_trades(balance, budget)`) and the Trader executes. A request that NAMES THE TRADE ("Buy 50 shares of MSFT", entry 87) is a DIRECT ORDER: `state["plan"]["direct"]` is set, the graph goes from the
    `account_manager` node to `account_manager_direct` and then to the Trader, and the Data Clerk, the Advisor and the Critic are skipped; it needs `with_critic` (the desk that decides and trades). `state["plan"]` carries the form, the companies and a `Result` (the runner's own record, so its reports read it). Nothing here names a sector or a company."""
    if with_trader and not with_critic:
        raise ValueError("with_trader needs with_critic: the Trader executes the trades the Critic reviewed and the Account Manager decided on")
    clerk_factory = clerk_factory or (lambda order: DataClerk(allow_fetch=order.live))
    advisor_factory = advisor_factory or (lambda order: Advisor())
    funds_factory = funds_factory or (lambda: AccountManager())
    critic_factory = critic_factory or (lambda order: Critic())
    guard = guard or Guard.code_only()
    clock = clock or trading_now   # the real clock, or the simulated one (the close of the last trading day that has closed: entry 129) when USE_LIVE_TRADING=false: entry 76

    def saved_account(manager: AccountManager, who, profile) -> Portfolio:
        """Connect `manager` to the saved account `who` (a sample key such as jacob_sample, or an account number), set the profile that was given (AFTER `connect`, which forgets a profile read from the default file), and
        return the saved portfolio. A ValueError (no such account, a bad profile file) or RuntimeError keeps its message, which names the thing and never a value; any other failure (the database is down, the password
        is refused) becomes a RuntimeError that names the KIND of failure only: a connection error can carry the host, the user and more."""
        try:
            store = (store_factory or open_book)()
            who = str(who).strip()
            manager.connect(store, int(who) if who.isdigit() else store.account_id_for_seed(who))
            if profile:
                manager.profile = profile
            _ = manager.profile                # the account's own row when no profile was given: read now, so a failure is reported here
            return manager.list_portfolio()
        except (ValueError, RuntimeError):
            raise
        except FileNotFoundError:
            raise RuntimeError("the account book is not set up: the file with its password is missing (run scripts/setup_postgres.sh)") from None
        except Exception as error:
            raise RuntimeError(f"the account book could not be read ({type(error).__name__})") from None

    def account(state: DeskState) -> tuple[AccountManager, Portfolio]:
        """The Account Manager with the user's profile, and the user's portfolio, from `request["account"]` (the sample files when it says nothing; the saved account when it names one: `saved_account`).
        Raises ValueError or RuntimeError, with a message that names the setting and never a value."""
        spec = state["request"].get("account") or {}
        manager = funds_factory()
        if spec.get("account") is not None:
            return manager, saved_account(manager, spec["account"], spec.get("profile"))
        portfolio = spec.get("portfolio", DEFAULT_PORTFOLIO)
        manager.profile = spec.get("profile") or DEFAULT_PROFILE
        _ = manager.profile                    # read it now: a profile that is not valid is an error HERE, where a node can report it, and not later inside the Critic's call
        return manager, portfolio if isinstance(portfolio, Portfolio) else load_portfolio(portfolio)

    def read_request(state: DeskState) -> dict:
        """The security layer screens what the user typed (entry 116), then the Account Manager reads it (`read_text`). A request that is BLOCKed or held for REVIEW stops the desk here: the policy's message is the error, no model is called
        and nothing is done. ALLOW and ALLOW_AND_FLAG go on exactly as before (a flag is also put in `state["security"]`)."""
        ask = state["request"]["ask"]
        verdict = guard.screen_request(ask.get("text"), account_id=(state["request"].get("account") or {}).get("account"))
        if verdict.refused:
            return {"order": None, "errors": [verdict.message], "plan": None, "clarify": False, "security": verdict.as_dict()}
        update = read_text(state)
        return {**update, "security": verdict.as_dict()} if verdict.action != "ALLOW" else update

    def read_text(state: DeskState) -> dict:
        """The Account Manager reads what the user typed (`request["ask"]`), shows it for a yes (`confirm`), and makes the work order. Never raises: a request it cannot read, a profile or account that cannot be opened,
        or a user who says no is a message in `errors`, and the desk stops here."""
        ask = state["request"]["ask"]
        try:
            manager, _ = account(state)
        except (ValueError, RuntimeError, OSError) as error:
            return {"order": None, "errors": [str(error)], "plan": None}
        try:
            intent = manager.understand_request(ask.get("text"))
            asked = None if intent.action == "order" else classify(*intent.as_request())
            mode = (state["request"].get("account") or {}).get("mode") or manager.profile.trade_mode      # the run's own mode
            intent = manager.settle_unclear_number(intent, mode, clarify)      # "spend 1000 on MU": shares or dollars? asked in Ask mode, a stop in Auto mode (entry 138, D5): never bought on a guess
        except (ValueError, RuntimeError, OSError) as error:        # the request could not be read: the run's last message asks the user to clarify it (entry 102)
            return {"order": None, "errors": [str(error)], "plan": None, "clarify": True}
        if intent.action == "order" and not with_critic:
            return {"order": None, "errors": ["a direct order needs the desk that decides and trades (with the Critic's desk, build_desk(with_critic=True))"], "plan": None}
        if confirm is not None and confirm(intent) is not True:
            return {"order": None, "errors": ["stopped: what was understood was not confirmed, so nothing was done"], "plan": None}
        if intent.action == "order":      # the user named the trade (entries 87 and 88): the work order holds the user's own orders, and there is nothing to search, recommend or review
            order = new_work_order(intent.request, requester=state["request"].get("requester", REQUESTER), action_type="trade", source=DEFAULT_SOURCE)
            mode = (state["request"].get("account") or {}).get("mode") or manager.profile.trade_mode      # the run's own mode, the one `decide_direct` uses
            named = manager.name_orders(order, intent.orders, intent.request, clock(), approver, mode)    # this path skips the Data Clerk's node: the Account Manager asks the Data Clerk for each company name and symbol and appends them to the work order
            order.errors = [*order.errors, *named.problems]       # entry 102: a stock that is not listed is dropped (Auto mode) or asked about (Ask mode); a blocking issue stops the run here and its last message asks the user to clarify
            kept = intent.model_copy(update={"orders": named.orders}) if named.dropped else intent      # the plan's intent holds only the stocks that stay
            return {"order": order, "errors": named.problems, "plan": {"intent": kept, "direct": True, "action": "order", "request_text": intent.request, "dropped": named.dropped}, "clarify": bool(named.problems)}
        find, days, fetch = intent.count or ask.get("default_count", 3), ask.get("news_days", ASK_NEWS_DAYS), bool(ask.get("fetch"))
        case = ask.get("case") or ("industry" if asked.kind in ("phrase", "list") else "funds")
        order = new_work_order(f"What do the stored news, press releases and filings say about {intent.what()}?", requester=state["request"].get("requester", REQUESTER), subject=asked.symbols or asked.phrase or asked.text, source=DEFAULT_SOURCE,
                               time_period=f"last {days} days", since_days=days, top_k=TOP_K, live=fetch, case=case, criteria={"sentiment": "bullish or bearish"}, max_price=intent.max_price, min_price=intent.min_price)      # a price limit ("under $50 a share") and a price floor ("over $20 a share") go on the order, where the Advisor reads them (entries 138 and 140)
        reasons = verify_request(order)
        order.errors = [*order.errors, *reasons]
        plan = {"intent": intent, "request": asked, "find": find, "case": case, "action": intent.action, "balance": "even" if intent.allocation == "even" else None, "budget": intent.total_budget,
                "each": intent.each_dollars,      # "the top 3 with $3,000 each" (entry 125): the amount for EACH purchase, never a total
                "max_price": intent.max_price,      # "under $50 a share" (entry 138): the Account Manager checks it again against the price it sizes at
                "min_price": intent.min_price,      # "over $20 a share" (entry 140): the same from the other side
                "raise_funds": intent.raise_funds,      # the user's words let the Account Manager sell holdings to pay for a purchase the cash does not cover (entry 142, R2)
                "shares_of": dict(intent.shares_of) or None, "shares_each": intent.shares_each,      # "buy 10 of MU if the recommendation passes", "5 shares of each of the top 3" (entry 140, C1): the shares are each approved purchase's limit
                "count": intent.count,      # the number of companies the request asked for, or None: `find` falls back to a default of 3 for the Advisor and that default must not cut anything (entry 107)
                "said_how_much": intent.said_how_much(), "request_text": intent.request, "fetch": fetch,
                "news_days": days, "result": Result(asked, case, eastern_date(clock()), find)}
        return {"order": order, "errors": reasons, "plan": plan}

    def account_manager(state: DeskState) -> dict:
        request, resolution = state["request"], None
        if request.get("ask") is not None:
            return read_request(state)
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
        plan = state.get("plan")
        if plan is None:
            return {"order": clerk.request_data(order)}
        # a request in the user's own words: the companies first (a theme or a list of names is not a symbol yet), then their data, then the work order is answered
        result, today = plan["result"], eastern_date(clock())
        try:
            result.companies = find_companies_of(plan["request"], clerk, today, plan["find"], result)
            listed = scored_symbols(result)
            if not listed:
                result.notes.append("no listed company was found to score")
                order.status, order.errors = "data_failed", [*order.errors, "no listed company was found to score"]
                return {"order": order}
            if plan["fetch"]:
                held = [p.symbol for p in account(state)[1].positions]      # NOT for the Advisor (it is given no portfolio): the Account Manager prices what the user holds (`DataClerk.request_quotes`; the simulated clock prices a holding from its stored bar), so those bars are refreshed too
                result.refresh = clerk.refresh_data(listed, news_days=plan["news_days"], filings_days=365, bars_for=held)
        except (RuntimeError, ValueError, OSError) as error:
            order.status, order.errors = "data_failed", [*order.errors, f"the Data Clerk could not prepare the data: {error}"]
            return {"order": order}
        order.subject, order.sector = listed, result.sector
        order.query = f"What do the stored news, press releases and filings say about {', '.join(listed)}?"
        return {"order": clerk.request_data(order, today)}

    def advisor(state: DeskState) -> dict:
        order = state["order"]
        try:
            agent = advisor_factory(order)      # the Advisor is given NO portfolio (entry 105): it recommends purchases and does not know what the user holds
        except (RuntimeError, ValueError, OSError) as error:
            order.errors = [*order.errors, f"the Advisor could not start: {error}"]
            return {"order": order}
        if state.get("plan"):   # a request in the user's own words: the top N of the companies the Data Clerk found
            plan = state["plan"]
            return {"order": agent.generate_recommendations(order, plan["case"], eastern_date(clock()), top_k=plan["find"], only_subject=True)}
        if with_critic:   # version 1: the Critic reads the records of recommendations to buy. The case is the order's own, else inferred from its subject
            return {"order": agent.generate_recommendations(order, today=eastern_date(clock()))}
        return {"order": agent.advise(order)}

    def critic(state: DeskState) -> dict:
        """The Critic approves or denies each recommendation: never raises, and never changes a recommendation. It is given the profile's risk view and the clock: no portfolio, no money, no size."""
        order = state["order"]
        try:
            manager, _ = account(state)      # the Account Manager holds the profile; the portfolio is the Account Manager's business, not the Critic's
            agent = critic_factory(order)
        except (RuntimeError, ValueError, OSError) as error:
            order.errors = [*order.errors, f"the Critic could not start: {error}"]
            return {"order": order}
        reviewed = agent.review_recommendations(order, manager.risk_view(), clock())
        guard.trust(reviewed)      # the Trust Score of each recommendation (entry 116): reliability, recorded and traced, changes no decision
        return {"order": reviewed}

    def account_manager_decision(state: DeskState) -> dict:
        """The Account Manager turns each review into a trade decision and sizes it: Auto trades (round-robin, unless the user asked for another split), Ask asks the user, a denial is final. It asks the Data Clerk for the
        holdings' current prices itself, when it needs them (`AccountManager.decide_trades`)."""
        order = state["order"]
        try:
            manager, portfolio = account(state)
        except (RuntimeError, ValueError, OSError) as error:
            order.errors = [*order.errors, f"the Account Manager could not decide: {error}"]
            return {"order": order}
        plan = state.get("plan") or {}
        return {"order": manager.decide_trades(order, portfolio, approver, clock(), (state["request"].get("account") or {}).get("mode"), plan.get("balance"), plan.get("budget"),
                                               clarify=clarify, said_how_much=bool(plan.get("said_how_much")), request_text=plan.get("request_text") or "", count=plan.get("count"), each=plan.get("each"), max_price=plan.get("max_price"),
                                               min_price=plan.get("min_price"), shares_of=plan.get("shares_of"), shares_each=plan.get("shares_each"), raise_funds=bool(plan.get("raise_funds")))}

    def account_manager_direct(state: DeskState) -> dict:
        """A DIRECT ORDER (entry 87): the Account Manager prices the trades the user named (it asks the Data Clerk for the quotes) and decides them: the user's own order is the approval, so there is no recommendation, no
        Critic review and no second question, in Ask mode too. Never raises: an account that cannot be opened is a message on the order."""
        order = state["order"]
        try:
            manager, portfolio = account(state)
        except (RuntimeError, ValueError, OSError) as error:
            order.errors = [*order.errors, f"the Account Manager could not decide: {error}"]
            return {"order": order}
        plan = state.get("plan") or {}
        intent = plan.get("intent")
        return {"order": manager.decide_direct(order, portfolio, clock(), (state["request"].get("account") or {}).get("mode"), approver, clarify=clarify, said_how_much=intent.said_how_much() if intent else True,
                                               request_text=plan.get("request_text") or "", balance="even" if intent and intent.allocation == "even" else None, budget=intent.total_budget if intent else None,
                                               raise_funds=bool(intent and intent.raise_funds))}

    def trader(state: DeskState) -> dict:
        """The Account Manager hands the decided order to the Trader, which executes what is permitted against the simulated broker (paper or backtest): never raises, and never edits what the other agents wrote. The
        RaiseFunds answer of the funds node (`state["funds"]`) is the funding plan; `approver` is asked whether the funding sells may be made (a sale is never automatic)."""
        order = state["order"]
        spec = state["request"].get("trade") or {}
        try:
            manager, portfolio = account(state)
            agent = trader_factory(order) if trader_factory else Trader(mode=spec.get("mode"), use_model=spec.get("model", True) is not False)
        except (RuntimeError, ValueError, OSError) as error:
            order.errors = [*order.errors, f"the Trader could not start: {error}"]
            return {"order": order}
        return {"order": manager.perform_trades(order, portfolio, agent, clock(), funding=state.get("funds"), approver=approver, book=bool(spec.get("book")))}

    def account_manager_funds(state: DeskState) -> dict:
        """RaiseFunds, never raising. When a decision carries `funding` (entry 142: the user permitted selling holdings to pay for the purchase the cash does not cover) it is RaiseFunds for those purchases, ONE plan for the whole shortfall
        (`AccountManager.fund_decisions`), the Trader's funding plan; the decisions change only when it cannot pay for them (`no_funds`). Otherwise (the older `--raise-funds` of entry 42) it is RaiseFunds for the best recommendation and
        never changes the work order."""
        spec = state["request"].get("funds") or {}
        order = state["order"]
        if funding_module.needs_funding(order):
            try:
                manager, portfolio = account(state)
            except (RuntimeError, ValueError, OSError) as error:
                order.errors = [*order.errors, f"the Account Manager could not raise the funds: {error}"]
                return {"order": order}
            return {"order": order, "funds": manager.fund_decisions(order, portfolio, clock())}
        trades = [d for d in order.trade_decisions if d["will_trade"] in ("yes_auto", "yes_user") and d["action"] != "reduce"]   # with the Critic: what it approved, the Account Manager sized (the shares it asks RaiseFunds for are the decision's) and the user or Auto allowed
        target = spec.get("buy") or (trades[0]["symbol"] if trades else None if with_critic else top_recommendation(order))
        if not spec.get("shares") and trades and trades[0]["symbol"] == target:
            spec = {**spec, "shares": sh.ceil(trades[0]["shares"])}      # RaiseFunds plans WHOLE shares (entry 110): a purchase of a fraction is raised for the next whole share, so that the plan raises at least enough
        if not target:
            return {"funds": {"status": "refused", "errors": ["the Critic approved no purchase that was allowed to trade, and no symbol was given" if with_critic else
                                                              "the Advisor made no recommendation with a verified bullish score to buy, and no symbol was given"], "quotes": {}, "run": {}}}
        try:
            if with_critic and (state["request"].get("account") or {}).get("account") is not None:   # a saved account: the plan is made for ITS holdings, the ones the Trader will trade (the funds' own portfolio is not used)
                manager, portfolio = account(state)
            else:
                chosen = spec["portfolio"] if "portfolio" in spec or not with_critic else (state["request"].get("account") or {}).get("portfolio", DEFAULT_PORTFOLIO)   # with the Critic: the user's own portfolio
                portfolio = chosen if not isinstance(chosen, (str, Path)) else load_portfolio(chosen)
                manager = funds_factory()
        except (KeyError, ValueError, OSError, RuntimeError) as error:  # no portfolio, a bad portfolio file, or a missing key: the message names the setting, never a value
            return {"funds": {"status": "refused", "errors": [f"the Account Manager could not start: {error}"], "quotes": {}, "run": {}}}
        return {"funds": manager.raise_funds(portfolio, target, spec.get("shares", 1))}

    def after_advisor(state: DeskState) -> str:
        if with_critic:
            if (state.get("plan") or {}).get("action") == "research":   # the user asked only to look: nothing is reviewed, decided or traded
                return END
            order = state["order"]
            if order.status != "recommended":
                return END
            if order.recommendations:
                return "critic"
            return "account_manager_decision" if neutral_names.neutral_candidates(order) else END       # nothing for the Critic to review, but a company the user named that the Advisor scored neutral counts as approved (entry 91)
        return "account_manager_funds" if with_funds and state["order"].status == "recommended" else END

    def permitted(order: WorkOrder) -> bool:
        """Whether the Trader has anything to do: the order is decided or waiting for a user's answer, and at least one decision says yes_auto or yes_user."""
        return order.status in ("decided", "reviewed") and any(d.get("will_trade") in TRADING for d in order.trade_decisions)

    def after_decision(state: DeskState) -> str:
        if with_funds:
            return "account_manager_funds" if state["order"].status in ("decided", "reviewed") else END
        return after_permission(state["order"])

    def after_funds(state: DeskState) -> str:
        return "trader" if with_trader and permitted(state["order"]) else END

    def after_account_manager(state: DeskState) -> str:
        if state.get("order") is None or state.get("errors"):
            return END
        return "account_manager_direct" if (state.get("plan") or {}).get("direct") and with_critic else "data_clerk"      # a direct order skips the research, the Advisor and the Critic (entry 87)

    def after_direct(state: DeskState) -> str:
        return after_permission(state["order"])

    def after_permission(order: WorkOrder) -> str:
        """Where a decided order goes: nowhere when nothing is permitted; to RaiseFunds first when a permitted purchase is to be paid for by selling holdings the user allowed to be sold (entry 142); else straight to the Trader."""
        if not (with_trader and permitted(order)):
            return END
        return "account_manager_funds" if funds_node and funding_module.needs_funding(order) else "trader"

    def after_data_clerk(state: DeskState) -> str:
        return "advisor" if with_advisor and state["order"].status == "data_ready" else END

    funds_node = with_funds or (with_critic and with_trader)      # entry 142: every desk that decides AND trades can raise funds for a purchase the cash does not cover
    graph = StateGraph(DeskState)
    graph.add_node("account_manager", _timed("account_manager", account_manager))
    graph.add_node("data_clerk", _timed("data_clerk", data_clerk))
    graph.add_node("advisor", _timed("advisor", advisor))
    if with_critic:
        graph.add_node("critic", _timed("critic", critic))
        graph.add_node("account_manager_decision", _timed("account_manager_decision", account_manager_decision))
    if funds_node:
        graph.add_node("account_manager_funds", _timed("account_manager_funds", account_manager_funds))
    if with_trader:
        graph.add_node("trader", _timed("trader", trader))
    graph.add_edge(START, "account_manager")
    if with_critic:   # the desk that decides and trades can take a direct order: account_manager -> account_manager_direct -> trader
        graph.add_node("account_manager_direct", _timed("account_manager_direct", account_manager_direct))
        graph.add_conditional_edges("account_manager", after_account_manager, {"data_clerk": "data_clerk", "account_manager_direct": "account_manager_direct", END: END})
        graph.add_conditional_edges("account_manager_direct", after_direct, {"trader": "trader", **({"account_manager_funds": "account_manager_funds"} if funds_node else {}), END: END} if with_trader else {END: END})
    else:
        graph.add_conditional_edges("account_manager", after_account_manager, {"data_clerk": "data_clerk", END: END})
    graph.add_conditional_edges("data_clerk", after_data_clerk, {"advisor": "advisor", END: END})
    if with_critic:
        graph.add_conditional_edges("advisor", after_advisor, {"critic": "critic", "account_manager_decision": "account_manager_decision", END: END})
        graph.add_edge("critic", "account_manager_decision")
        if with_funds:
            graph.add_conditional_edges("account_manager_decision", after_decision, {"account_manager_funds": "account_manager_funds", END: END})
            if with_trader:   # the funds node runs first and its answer is the Trader's funding plan
                graph.add_conditional_edges("account_manager_funds", after_funds, {"trader": "trader", END: END})
            else:
                graph.add_edge("account_manager_funds", END)
        elif with_trader:
            graph.add_conditional_edges("account_manager_decision", after_decision, {"trader": "trader", "account_manager_funds": "account_manager_funds", END: END})
        else:
            graph.add_edge("account_manager_decision", END)
        if with_trader:
            if not with_funds:      # the funds node of entry 142: its answer is the Trader's funding plan (with `with_funds` the edge above is the same one)
                graph.add_conditional_edges("account_manager_funds", after_funds, {"trader": "trader", END: END})
            graph.add_edge("trader", END)
    elif with_funds:
        graph.add_conditional_edges("advisor", after_advisor, {"account_manager_funds": "account_manager_funds", END: END})
        graph.add_edge("account_manager_funds", END)
    else:
        graph.add_edge("advisor", END)
    return graph.compile()


def run_desk(request: dict, **options) -> DeskState:
    """Run one request through the desk. `options` are those of `build_desk`. Returns the final state: the order, why a request was refused, the
    path through the graph and the seconds each node took."""
    return build_desk(**options).invoke({"request": request, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}, "with_trader": bool(options.get("with_trader")), "plan": None})


def report(state: DeskState, total_seconds: float | None = None) -> str:
    """What the user reads: the request, the path, the answer with its sources, the recommendations with every signal, and the timings."""
    lines = []
    request, order = state["request"], state.get("order")
    source = request.get("source") or DEFAULT_SOURCE
    lines.append(f"REQUEST  symbol {request.get('symbol') or '-'} | term {request.get('term') or '-'} | source {source} "
                 f"| from {request.get('from_days_ago') or DEFAULT_DAYS_BY_SOURCE.get(source, DEFAULT_DAYS)} days ago"
                 + (f" | company {request['company']}" if request.get("company") else ""))
    if (request.get("ask") or {}).get("text"):   # a request in the user's own words: what the Account Manager understood of it
        plan = state.get("plan")
        lines.append("ASK      " + " ".join(str(request["ask"]["text"]).split())[:200] + (f"\nUNDERSTOOD  {plan['intent'].describe()}" if plan else ""))
    if clock_note():   # USE_LIVE_TRADING=false: the desk works at a simulated moment, and the report says so
        lines.append(clock_note())
    if state.get("resolution"):
        r = state["resolution"]
        lines.append(f"RESOLVED {r['typed']!r} -> {r['symbol']} ({r['company']}), looked up and matched as a {r['how']}")
    saved = (request.get("account") or {}).get("account")
    if saved is not None:   # a saved account WRITES: the decisions always (entry 72), the fills only with --book. Said in the report so that nobody is surprised
        lines.append(f"ACCOUNT  saved account {str(saved)[:40]} (the Postgres account book): the Account Manager's decisions are saved in it; "
                     + ("the Trader's fills are booked in it" if (request.get("trade") or {}).get("book") else "no fill is booked in it"))
    path = state.get("path", [])
    lines.append("PATH     " + " -> ".join(path) + ("" if reached_its_end(state) else "  (stopped here)"))   # see `reached_its_end`: a Critic run ends at the decision, a Trader run at the trader node
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
    review = review_section(order)
    if review:
        lines.append("\n" + "\n".join(review))
    neutral = neutral_names.neutral_lines(order)
    if neutral:
        lines.append("\n" + "\n".join(neutral))
    if state.get("funds"):
        lines.append("\nACCOUNT MANAGER: RAISE FUNDS (a Tree of Thought, entry 42)\n" + funds_report(state["funds"]))
    direct = direct_order.direct_section(order)
    if direct:
        lines.append("\n" + "\n".join(direct))
    trades = trade_section(order)
    if trades:
        lines.append("\n" + "\n".join(trades))
    seconds = state.get("seconds", {})
    parts = "  ".join(f"{name} {seconds[name]:.1f}" for name in ("account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "account_manager_direct", "account_manager_funds", "trader") if name in seconds)
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
    parser.add_argument("--critic", action="store_true", help="after the Advisor (version 1: ranked, audited recommendations), the Critic approves or denies each one and the Account Manager decides what to trade "
                        "(Critic v1, entries 56 to 59; it calls Finnhub for numbers and quotes and OpenRouter for the critic's model: PAID)")
    parser.add_argument("--profile", default=None, help="--critic: the user's profile JSON file (risk tolerance, Ask or Auto, the size limits the Account Manager sizes by; default: the sample profile of Jacob Pyrett, or the saved account's own row with --account)")
    parser.add_argument("--mode", choices=("ask", "auto"), help="--critic: override the profile's trade mode for this run (ask: you approve each trade; auto: what the Critic approves with a high score trades, and the funds are spent round-robin)")
    parser.add_argument("--trade", nargs="?", const="paper", choices=("paper", "backtest"), metavar="{paper,backtest}",
                        help="--critic: after the decisions, the TRADER executes what is permitted against a SIMULATED broker (Trader v1, entries 73 to 75; there is no live trading). paper (the default when only --trade is given): at the "
                        "current quote, which COSTS (Finnhub); backtest: at the next session's open from the stored daily bars. The model (gpt-oss-120b, OpenRouter) is PAID. Fills stay in memory unless --book")
    parser.add_argument("--no-trader-model", action="store_true", help="--trade: no model for the Trader: the program places the default orders (the free, repeatable way; a paper run still asks Finnhub for quotes)")
    parser.add_argument("--account", help="--critic: use a saved account of the Postgres account book (a sample key such as jacob_sample, or an account number) for the profile and the portfolio, instead of the JSON files. "
                        "It also saves the Account Manager's decisions in that account (entry 72)")
    parser.add_argument("--book", action="store_true", help="--trade paper and --account: save the fills in that account (the cash and the holdings change in the book)")
    parser.add_argument("--json", action="store_true", help="print the finished work order as JSON instead of the report")
    parser.add_argument("--show-graph", action="store_true", help="print the graph as a Mermaid diagram and stop")
    args = parser.parse_args(argv)
    langsmith_tracing.start()   # LangSmith traces the run when LANGSMITH_API_KEY is set (entry 114)
    if args.trade and not args.critic:
        parser.error("--trade needs --critic: the Trader executes the trades the Critic reviewed and the Account Manager decided on")
    if (args.no_trader_model or args.book) and not args.trade:
        parser.error("--no-trader-model and --book go with --trade")
    if args.book and not args.account:
        parser.error("--book needs --account: the fills are saved in a saved account")
    if args.book and args.trade != "paper":
        parser.error("--book goes with --trade paper: a backtest is a replay and never writes the account book")
    if args.account and not args.critic:
        parser.error("--account goes with --critic: the profile and the portfolio of the account are read for the Critic's review and the decisions")
    if args.show_graph:
        print(build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None, with_funds=args.raise_funds, with_critic=args.critic, with_trader=bool(args.trade)).get_graph().draw_mermaid())
        return 0
    days = args.from_days_ago if args.from_days_ago is not None else str(DEFAULT_DAYS_BY_SOURCE.get(args.source, DEFAULT_DAYS))
    request = {"symbol": args.symbol, "term": args.term, "source": args.source, "from_days_ago": days, "fetch": not args.no_fetch}
    if args.company:
        request["company"] = args.company  # only when given: a news request is exactly the four parameters it always was
    options = {"with_advisor": not args.no_advisor}
    if args.raise_funds:
        options["with_funds"] = True
        request["funds"] = {"shares": args.buy_shares, **({"buy": args.buy} if args.buy else {})} if args.account else {"portfolio": args.portfolio, "shares": args.buy_shares, **({"buy": args.buy} if args.buy else {})}   # a saved account's plan is made for ITS holdings
    if args.critic:
        options["with_critic"] = True
        if args.account:   # the account's own row is the profile unless --profile gives one; its holdings are the portfolio (--portfolio is for the JSON files)
            request["account"] = {"account": args.account, **({"profile": args.profile} if args.profile else {}), **({"mode": args.mode} if args.mode else {})}
        else:
            request["account"] = {"portfolio": args.portfolio, "profile": args.profile or str(DEFAULT_PROFILE), **({"mode": args.mode} if args.mode else {})}
        options["approver"] = terminal_approver()   # Ask mode at a terminal: you are asked; with nobody there the decision stays pending, never a yes
    if args.trade:
        options["with_trader"] = True
        request["trade"] = {"mode": args.trade, "book": args.book, "model": not args.no_trader_model}
    if args.model:
        options["advisor_factory"] = lambda order: Advisor(model_id=args.model)
    state = run_desk(request, **options)
    total = time.perf_counter() - started
    order = state.get("order")
    if args.json:
        print(dumps({"request": request, "resolution": state.get("resolution"), "errors": state.get("errors"), "path": state.get("path"), "seconds": state.get("seconds"),
                     "funds": state.get("funds"), "decisions": order.trade_decisions if order else None, "order": order.model_dump() if order else None}, indent=1))      # the one encoder: a Decimal is a number (entry 112)
    else:
        print(report(state, total))
    if order is not None and not state.get("errors"):
        # the symbol and the term both go in the name, so that a run with a symbol and a term does not overwrite a run with the symbol alone
        named = (state.get("resolution") or {}).get("symbol") or (order.resolved or {}).get("symbol") or request["symbol"]  # a looked-up name is saved under its ticker
        parts = (named, request["term"], "" if request["source"] == DEFAULT_SOURCE else request["source"])  # press releases are saved apart from news
        label = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", part.strip().lstrip("$")).strip("_") for part in parts if part.strip()) or "order"
        out = OUT_DIR / f"desk_{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dumps({"request": request, "resolution": state.get("resolution"), "path": state.get("path"), "seconds": state.get("seconds"), "funds": state.get("funds"), "order": order.model_dump()}, indent=1),
                       encoding="utf-8")
        if not args.json:
            print(f"saved: {out.relative_to(OUT_DIR.parents[1]) if out.is_relative_to(OUT_DIR.parents[1]) else out}")
    if args.trade:   # the Trader's own end: a trade that was made, a decision still waiting for you, or an order that was only decided (nothing to trade)
        done = order is not None and not state.get("errors") and order.status in ("traded", "decided", "reviewed")
    elif args.no_advisor:
        done = order is not None and not state.get("errors") and order.status == "data_ready"
    else:   # the Critic path ends decided (or reviewed, a decision waiting for you), not recommended: `reached_its_end` says which end counts
        done = not state.get("errors") and reached_its_end({**state, "request": request})
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
