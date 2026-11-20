"""The work order: the one JSON object the agents pass to each other (design choices entry 25).

The week 2 design says every agent reads and updates the same `<WORK_ORDER>`: the Account Manager creates it, the Data Clerk
fills in the data, the Advisor appends a recommendation, the Critic judges it, the Trader executes what is permitted. This is that object, with the fields of the Data Clerk, the Advisor, the Critic, the Account
Manager's decisions and the Trader (Trader v1, design/trader_v1_final_spec.md; `trades` and `trader_runs` are what the Trader writes).

    wo = new_work_order("AI memory shortage. Which companies are affected?", since_days=7)
    verify_request(wo)                    # [] when the Data Clerk may work on it, else the reasons it may not
    DataClerk().request_data(wo)          # fills wo.data (answer, cited chunks, sources, problems) and sets the status
    verify_action(wo)                     # [] when the data can be trusted: the checks of the design's verify_action
    Advisor().advise(wo)                  # appends wo.recommendations
    wo.model_dump_json()                  # it is plain JSON: store it, log it, hand it to the next agent

PRESS RELEASES (design choices entry 34): `source="press_releases"`, `subject` the symbol (argument #1), `company` the company name (argument #2). The
Data Clerk resolves them in code before any model runs: the symbol if it is found, else the company name's symbol; nothing found is `data_failed` with
"Symbol: X is not found" or "Company Name not found." in `errors`. What it decided is on the order as `resolved`.

FILINGS (design choices entry 35): `source="filings"` (the SEC's filings: 10-K, 8-K, 10-Q ...) is asked for in exactly the same way and resolved by the same code, with the
same messages. Its `action_type` is `search_filing` (design choices entry 37; entry 35 had kept `search_news`): the work order checks that `search_filing` goes with `source="filings"` and
that a filings order has that action. Press releases keep `search_news` (entry 34): `source` says which corpus is searched.

STATUS moves one way:  new -> data_ready | data_failed -> recommended -> reviewed -> decided -> traded   (the Critic sets `reviewed`, the Account Manager `decided`: Critic v1, design choices entry 59; the Trader sets `traded`: Trader v1).
`data_ready` means the Data Clerk did its job. That includes an honest "the stored news does not cover it" (`data.no_supporting_text`):
nothing to find is a result, not a failure. `data_failed` means something is wrong: an invalid citation, an empty answer, a model
failure, or a request the Data Clerk should not have taken.

THE RESULTS OF EACH SOURCE (design choices entry 39): `data.results` says, for news, press releases and filings, which numbered sources of `data.sources` came from that corpus,
which of them the answer cites and which were added by the Data Clerk's code sweep (entry 40: every order carries all three corpora, `data.sweep` says how each was searched;
`data.sources_of("filings")` gives the items); `data.filing_evidence` holds, per symbol, the whole result of the Data Clerk's
`fetch_filings_recommendation` tool (the balance sheet for both periods, the capital actions, the exact quotes on dilution, executive compensation, buybacks and dividends,
the gaps; it scores nothing) when the Data Clerk called it. The Advisor reads these from the order; nothing here is a score.

ADVISOR V1 (design choices entry 51): an order may also carry `metrics` (a `calculate_financials` request: the metrics asked for), `case` (funds / theme / market), `sector` (the sector ETF the user gives) and `candidates`
(what the Advisor looked at and did not return); `DataResult.metrics` holds the numbers the Data Clerk calculated (`DataClerk.request_metrics`, `verify_metric_request`). An order with none of them is a version 1
order and reads exactly as before; one that carries any is `schema_version` 2.

CRITIC V1 (design choices entries 56 to 59): the Critic reads `recommendations` and writes `reviews` (one per recommendation, joined by `record_id`) and `critic_runs`; the Account Manager then writes
`trade_decisions` (one per review: `will_trade`, `funds_avail`). The Critic asks the Data Clerk, directly, for the numbers and the sources again with a `verify_data` order (`verify`, answered on
`data.verification`). `verify_trade` is the one check the Trader must pass before any trade: no review, no approval or no permission means no trade. An order that carries reviews is `schema_version` 3.

TRADER V1 (design/trader_v1_final_spec.md): the Trader reads `trade_decisions` and writes `trades` (one `Trade` record per ticket, joined to a decision by `record_id`; `TRADE_STATUSES`) and `trader_runs` (one record per run:
prometheus/agents/trader/record.py), and the order becomes `traded`. `verify_trades` is its output guard: the trades must agree with the decisions or nothing is booked. The Trader never asks the Data Clerk for a price itself:
the Account Manager does, with a `fetch_quote` order (the current quote) or a `fetch_bars` order (the stored daily bars, `bars` = {"start", "end"}, answered on `data.bars`: `verify_bar_request`, `DataClerk.request_bars`),
in groups of at most 30 symbols. An order that carries `bars` is `schema_version` 2 (a request field, like `metrics` and `verify`); one that carries trades is 4. The Trader may be called again on the same order (a `reviewed` one
whose pending decision the user has answered since, a `decided` one with a resting order): a trade that is `filled` or `rejected` is final, the numbering of the tickets goes on, and `review_order` and `decide_trades` hand
a `traded` order back unchanged, as they do a `reviewed` or a `decided` one. The Trader never edits `recommendations`, `reviews` or `trade_decisions`: they are the same bytes afterwards.

THE POT AND DOLLARS (design choices entry 111; `design/dollar_orders_spec.md`): a decision the Account Manager sized from money carries `size.pot` (`decision_pot`), the money of the run, and may carry `dollars` (`decision_dollars`), an order in dollars whose `shares`
are only the estimate. `verify_trade` checks the dollars (a number above zero to the cent, not above the pot, none for a direct order whose shares the user typed or for a sale), `verify_fill` checks the fill's AMOUNT against them (and the old check, more shares than the
decision gave, for an order in shares), and `verify_trades` fails a work order whose purchases carry two pots or cost more than their pot (`pot_reasons`).

DECIMAL MONEY (design choices entry 112; design/decimal_money_spec.md): the plain dicts of `trade_decisions`, `trades`, `trader_runs` and `direct_orders` hold the ACCOUNT's money, prices and share counts: a `Decimal` after the code that wrote them,
a float or an int after `json` read them (a capture, a saved order). Every reader here goes through a door (`money.money`, `money.price`, `shares.quantity`) and compares as `Decimal`; the four fields are written as JSON NUMBERS by the one encoder
(`money.Jsonable`: a whole value as an integer, any other by its own digits), never as strings. The Advisor's and the Critic's numbers on the same order (`recommendations`, `reviews`, `candidates`) are measurements and stay floats; a whole work
order is never read with `parse_float=Decimal`.

The chunks travel with the order (`data.sources` holds every chunk the model was shown, with its text) so the Critic and the audit
log can check what a score was based on.
"""
import re
import uuid
from datetime import date, datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from prometheus.common import shares as sh
from prometheus.common.money import CENT, Jsonable, cents, digits, money, price
from prometheus.common.trace import traced

ACTION_TYPES = ("trade", "search_news", "search_filing", "fetch_quote", "search_socials", "calculate_financials", "verify_data", "fetch_bars", "lookup_symbol")
SEARCH_ACTIONS = ("search_news", "search_filing")   # what the Data Clerk's request_data handles today: a search of stored text (the other actions are refused by verify_request)
ACTION_SOURCE = {"search_filing": "filings"}       # an action that belongs to one source, and the source that has only that action: both directions are checked (entry 37)
SOURCE_ACTION = {source: action for action, source in ACTION_SOURCE.items()}
STATUSES = ("new", "data_ready", "data_failed", "recommended", "reviewed", "decided", "traded")
HAS_DATA = ("data_ready", "recommended", "reviewed", "decided", "traded")  # the statuses of an order that already has its data: asking the Data Clerk again changes nothing
WIRED_SOURCES = ("news", "press_releases", "filings")  # the sources the Data Clerk can search today; social media and the rest are not wired into the search yet
LOOKUP_SOURCES = ("press_releases", "filings")         # the sources asked for by symbol (argument #1) and/or company name (argument #2), resolved in code (entries 34 and 35)
REQUEST_LABEL = {"press_releases": "press release", "filings": "filings"}  # how a message names the request: "a press release request ...", "a filings request ..."
MAX_QUERY = 1000
MAX_TOP_K = 20
MAX_SINCE_DAYS = 90                                   # news. A source with another limit is in MAX_DAYS
MAX_DAYS = {"news": 90, "press_releases": 730, "filings": 730}  # the window each source allows: the fetch tools' own limits (a company issues a few releases and filings a month)
SEARCH_TOOL = {"news": ("retrieve_news", "news"),     # per source: the tool a search must have used, and what the stored text is called in a message
               "press_releases": ("retrieve_press_releases", "press releases"),
               "filings": ("retrieve_filings", "filings")}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


CORPORA = ("news", "press_releases", "filings")      # the three sources whose results the Data Clerk puts on the order
CORPUS_PREFIX = {"pr:": "press_releases", "sec:": "filings"}  # chunk ids: prometheus/rag/loaders.py (PRESS_RELEASE_ID_PREFIX, FILING_ID_PREFIX); a news id has neither


def corpus_of(chunk_id: str) -> str:
    """Which source a chunk id belongs to: `pr:...#0` press releases, `sec:...#3` filings, anything else (`finnhub:...`, `polygon:...`) news."""
    return next((corpus for prefix, corpus in CORPUS_PREFIX.items() if str(chunk_id).startswith(prefix)), "news")


def group_results(sources: list[dict], references: list[dict]) -> dict[str, dict]:
    """The numbers of the sources on the order, by corpus: {"news": {"shown": [1, 2], "cited": [1], "swept": [3]}, "press_releases": ..., "filings": ...}.
    `shown` are the sources the Data Clerk's model was shown, `cited` those its answer cites, `swept` those its code added afterwards from a corpus the model had not searched (entry 40:
    the model never saw them and cannot have cited them). All three corpora are always there. Computed by code from the chunk ids: the numbers point at `sources`, nothing is copied."""
    cited = {r.get("chunk_id") for r in references}
    out = {corpus: {"shown": [], "cited": [], "swept": []} for corpus in CORPORA}
    for source in sources:
        if source.get("number") is None:
            continue
        group = out[corpus_of(source.get("chunk_id", ""))]
        group["swept" if source.get("swept") else "shown"].append(source["number"])
        if source.get("chunk_id") in cited:
            group["cited"].append(source["number"])
    return out


class DataResult(BaseModel):
    """What the Data Clerk put on the order: the `Result` of its agent, as plain data."""
    answer: str = ""
    references: list[dict] = Field(default_factory=list)  # the chunks the answer cites
    sources: list[dict] = Field(default_factory=list)     # every chunk the model was shown, with its number and text, then the chunks the code sweep added (`swept: true`, entry 40)
    sweep: dict[str, dict] = Field(default_factory=dict)  # per corpus (news, press_releases, filings, filing_evidence): who searched it ("model" or "code"), how many sources it gave and why none (entry 40)
    results: dict[str, dict] = Field(default_factory=dict)  # per corpus (news, press_releases, filings): the numbers of `sources` shown and cited (`group_results`); empty on an order that has no data yet
    filing_evidence: dict[str, dict] = Field(default_factory=dict)  # per symbol: the whole result of the fetch_filings_recommendation tool, if the Data Clerk called it (it scores nothing)
    metrics: dict[str, dict] = Field(default_factory=dict)  # per symbol, per metric name: the metric result of a `calculate_financials` order (value, unit, as_of, window, source, inputs_summary, note); entry 51
    companies: list[dict] = Field(default_factory=list)  # a phrase's companies, found by the Data Clerk (`find_companies`): {brand, parent, symbol, company, public, how, note}; empty on any other order
    market_brief: dict = Field(default_factory=dict)  # Case 3 only (the "what is happening" brief of Advisor v1 spec section 13): not built yet, always empty
    verification: dict = Field(default_factory=dict)  # the answer to a `verify_data` order (Critic v1): the chunks and filing-evidence items read again from the store, the symbols checked against the directory, the newest stored text; empty on any other order
    quotes: dict[str, dict] = Field(default_factory=dict)  # per symbol: the current quote (price, change, high, low, open, previous close, as_of_utc, source) of a `fetch_quote` order (entry 42); empty on any other order
    bars: dict[str, list[dict]] = Field(default_factory=dict)  # per symbol: the stored daily bars of a `fetch_bars` order, oldest first, each {"date": "YYYY-MM-DD", "open", "high", "low", "close", "volume"}; a symbol with none in the range is an empty list. Empty on any other order
    security_dropped: list[dict] = Field(default_factory=list)  # the documents the security layer dropped from the evidence before anything read them: {id, corpus, action, risk, rules} (entry 116); empty when none was
    problems: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    no_supporting_text: bool = False
    tool_calls: list[dict] = Field(default_factory=list)
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    completed_utc: str = ""

    def sources_of(self, corpus: str, cited_only: bool = False) -> list[dict]:
        """The items of `sources` that came from one corpus ("news", "press_releases" or "filings"), in the order they were numbered (the ones the model was shown, then the swept ones);
        with `cited_only` only those the answer cites."""
        if corpus not in CORPORA:
            raise ValueError(f"corpus must be one of {', '.join(CORPORA)}")
        group = self.results.get(corpus, {})
        wanted = set(group.get("cited", []) if cited_only else [*group.get("shown", []), *group.get("swept", [])])
        return [s for s in self.sources if s.get("number") in wanted]


class WorkOrder(BaseModel):
    schema_version: int = 1
    request_id: str
    requester: str = "user"
    created_utc: str
    query: str                                  # the request in the user's own words: the Data Clerk's question
    action_type: Literal["trade", "search_news", "search_filing", "fetch_quote", "search_socials", "calculate_financials", "verify_data", "fetch_bars", "lookup_symbol"] = "search_news"
    subject: str | list[str] | None = None      # a symbol, a list of symbols, or a term such as "memory stocks". Press releases and filings: ARGUMENT #1, the symbol (one)
    company: str | None = None                  # the company's name: for news, when the symbol was looked up from a name ("Procter & Gamble Co"), so that the Data Clerk can use it.
                                                # Press releases and filings: ARGUMENT #2, the name to look up when the symbol is not given or not found
    source: str = "news"                        # news | press_releases | filings. For press_releases and filings the Data Clerk resolves symbol/company first (`resolved`)
    time_period: str | None = None              # in words ("last 7 days")
    since_days: int | None = None               # the same, as a number of days
    live: bool = False                          # needs fresh, uncached data
    top_k: int = 5
    criteria: dict | None = None                # for the Advisor, for example {"sentiment": "bullish"}
    status: Literal["new", "data_ready", "data_failed", "recommended", "reviewed", "decided", "traded"] = "new"
    resolved: dict | None = None                # press releases and filings: what the lookup decided: typed_symbol, typed_company, symbol, company, how ("ticker" or "name")
    data: DataResult | None = None
    recommendations: Annotated[list[dict], Jsonable] = Field(default_factory=list)      # the Advisor's records (floats: measurements); the Critic's block of each carries a copy of the Account Manager's decision (`feedback.attach_decisions`: shares, est_cost), which may hold Decimals: written as numbers
    advisor_runs: list[dict] = Field(default_factory=list)  # one record per Advisor call: model, whether it had evidence, tokens, what it dropped
    errors: list[str] = Field(default_factory=list)
    # Advisor v1 (entry 51). All optional: an order without them is a version 1 order and reads exactly as before; one that carries any of them is `schema_version` 2.
    metrics: list[dict] | None = None           # a `calculate_financials` order: the metrics asked for, each {"name": str, "params": dict} (params optional)
    case: Literal["funds", "theme", "industry", "market"] | None = None  # which use case the order serves. None: the Advisor infers it (symbols as the subject -> funds, a term -> theme). industry: rank the companies of an industry, leaders included (entry 53)
    sector: str | dict[str, str] | None = None  # the sector ETF (a key of configs/sector_etfs.json), given by the user until the profile endpoint exists: one ticker (the subject symbols' sector)
                                                # or, on a metrics order that covers several companies, {symbol: ETF ticker} (spec 3 says a string; a map is the one widening, entry 51)
    candidates: list[dict] = Field(default_factory=list)  # the symbols that reached scoring and were not returned as recommendations, with the gate or rank that stopped them
    # Critic v1 (entries 56 to 59). All optional, as the Advisor's: an order that carries reviews is `schema_version` 3.
    verify: dict | None = None                  # a `verify_data` order: what to read again (`verify_verification_request` says what is allowed)
    reviews: list[dict] = Field(default_factory=list)          # the Critic: one review per recommendation, in the same order, joined by `record_id` (prometheus/agents/critic/record.py)
    critic_runs: list[dict] = Field(default_factory=list)      # one record per Critic step: the model, the tokens, the Data Clerk orders it made, what it dropped, errors
    trade_decisions: Annotated[list[dict], Jsonable] = Field(default_factory=list)  # the Account Manager: one decision per review (will_trade, funds_avail, ...): the Trader's permission
    # Trader v1 (design/trader_v1_final_spec.md). An order that carries trades is `schema_version` 4; a `fetch_bars` order carries `bars`.
    bars: dict | None = None                    # a `fetch_bars` order: the range to read, {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} (`verify_bar_request` says what is allowed)
    trades: Annotated[list[dict], Jsonable] = Field(default_factory=list)       # the Trader: one trade per ticket (prometheus/agents/trader/record.py `Trade`), each joined to a decision by `record_id`; `TRADE_STATUSES`
    trader_runs: Annotated[list[dict], Jsonable] = Field(default_factory=list)  # one record per Trader run: the model, the mode, the tool calls, the summary, the account before and after, what was dropped, errors
    # A DIRECT ORDER (design choices entry 87): the user named the trade ("Buy 50 shares of MSFT"), so there is no recommendation and no review. The Account Manager writes one record per trade here (`DIRECT_KEYS`) and
    # its decision for it carries `origin: "direct"`; `verify_trade` accepts such a decision only while the order holds the user's matching record.
    direct_orders: Annotated[list[dict], Jsonable] = Field(default_factory=list)


def new_work_order(query: str, *, requester: str = "user", action_type: str = "search_news", subject=None, source: str = "news",
                   time_period: str | None = None, since_days: int | None = None, top_k: int = 5, live: bool = False,
                   criteria: dict | None = None, company: str | None = None, metrics: list[dict] | None = None, case: str | None = None,
                   sector: str | dict[str, str] | None = None, verify: dict | None = None, bars: dict | None = None) -> WorkOrder:
    """A new order, with an id and a timestamp. It is not checked here: `verify_request` is the check, so that a bad request becomes
    a reported reason and not an exception halfway through the desk."""
    return WorkOrder.model_construct(
        schema_version=2 if (metrics is not None or case is not None or sector is not None or verify is not None or bars is not None) else 1, request_id=uuid.uuid4().hex[:12], requester=requester, created_utc=_now(), query=query, action_type=action_type,
        subject=subject, company=company, source=source, time_period=time_period, since_days=since_days, live=live, top_k=top_k, criteria=criteria,
        status="new", resolved=None, data=None, recommendations=[], advisor_runs=[], errors=[], metrics=metrics, case=case, sector=sector, candidates=[], verify=verify,
        reviews=[], critic_runs=[], trade_decisions=[], bars=bars, trades=[], trader_runs=[], direct_orders=[])


def verify_request(order: WorkOrder) -> list[str]:
    """Why the Data Clerk may NOT work on this order; an empty list means it may (the design's verify_request)."""
    reasons = []
    if not isinstance(order.query, str) or not order.query.strip():
        reasons.append("the query is empty")
    elif len(order.query) > MAX_QUERY:
        reasons.append(f"the query is longer than {MAX_QUERY} characters")
    if order.action_type not in ACTION_TYPES:
        reasons.append(f"action_type {order.action_type!r} is not one of {', '.join(ACTION_TYPES)}")
    elif order.action_type not in SEARCH_ACTIONS:
        reasons.append(f"action_type {order.action_type!r} is not handled by the Data Clerk's request_data yet (only {', '.join(SEARCH_ACTIONS)})")
    else:  # a search action must match the source it searches: search_filing is for filings and for nothing else, and a filings order has no other action (entry 37)
        wanted = ACTION_SOURCE.get(order.action_type)
        if wanted and order.source != wanted:
            reasons.append(f"action_type {order.action_type!r} is for source {wanted!r}, but the order's source is {order.source!r}")
        elif not wanted and order.source in SOURCE_ACTION:
            reasons.append(f"a {order.source} order's action_type is {SOURCE_ACTION[order.source]!r}, not {order.action_type!r}")
    if not isinstance(order.top_k, int) or isinstance(order.top_k, bool) or not 1 <= order.top_k <= MAX_TOP_K:
        reasons.append(f"top_k must be a whole number from 1 to {MAX_TOP_K}")
    maximum = MAX_DAYS.get(order.source, MAX_SINCE_DAYS)
    if order.since_days is not None and (not isinstance(order.since_days, int) or isinstance(order.since_days, bool)
                                         or not 1 <= order.since_days <= maximum):
        reasons.append(f"since_days must be a whole number of days from 1 to {maximum}")
    if order.source not in WIRED_SOURCES:
        reasons.append(f"source {order.source!r} is not wired yet (only: {', '.join(WIRED_SOURCES)})")
    if order.source in LOOKUP_SOURCES:  # argument #1 is the symbol (subject), argument #2 the company name: at least one, and one company
        symbols = [order.subject] if isinstance(order.subject, str) else list(order.subject or [])
        symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
        if len(symbols) > 1:
            reasons.append(f"a {REQUEST_LABEL[order.source]} request is for one symbol at a time")
        elif not symbols and not (isinstance(order.company, str) and order.company.strip()):
            reasons.append(f"a {REQUEST_LABEL[order.source]} request needs a symbol, a company name, or both")
    if order.status in HAS_DATA:
        reasons.append(f"the order already has its data (status {order.status})")
    return reasons


MAX_QUOTE_SYMBOLS = 30   # one call per symbol, paced at 4 a second (Finnhub's limit is 300 a minute). It was 10, the bound of the LLM's tools (MAX_SYMBOLS); the complex RaiseFunds scenario (entry 45) needs 11
                         # (eight holdings and three purchases), and a quote order is answered by code, not by a model that could ask for too much


def verify_quote_request(order: WorkOrder) -> list[str]:
    """Why the Data Clerk may NOT answer this `fetch_quote` order (`DataClerk.request_quotes`); an empty list means it may. A quote order has its own check because
    `verify_request` is the check of a SEARCH (it still refuses `fetch_quote`, entry 25): here the subject is a list of 1 to 30 symbols and nothing else matters."""
    reasons = []
    if order.action_type != "fetch_quote":
        reasons.append(f"action_type {order.action_type!r} is not 'fetch_quote'")
    symbols = [order.subject] if isinstance(order.subject, str) else list(order.subject or [])
    if not symbols or not all(isinstance(s, str) and s.strip() for s in symbols):
        reasons.append("a quote request needs one or more ticker symbols as its subject")
    elif len(symbols) > MAX_QUOTE_SYMBOLS:
        reasons.append(f"a quote request is for at most {MAX_QUOTE_SYMBOLS} symbols at a time")
    if order.status in HAS_DATA:
        reasons.append(f"the order already has its data (status {order.status})")
    return reasons


SYMBOL_ACTION = "lookup_symbol"
MAX_LOOKUP_NAMES = 10     # the company names of one direct request (`MAX_COUNT` of the intent form): one Finnhub search each, paced


def verify_symbol_request(order: WorkOrder) -> list[str]:
    """Why the Data Clerk may NOT answer this `lookup_symbol` order (`DataClerk.request_symbols`, design choices entry 88); an empty list means it may. The subject is a list of 1 to `MAX_LOOKUP_NAMES` stocks as
    the user typed them, each a ticker (in capitals) or a company name: text of letters, digits, spaces and & . , ' - / ( ) !, at most 60 characters. Its own check, as the quote order's: `verify_request` is the check of a SEARCH."""
    reasons = []
    if order.action_type != SYMBOL_ACTION:
        reasons.append(f"action_type {order.action_type!r} is not {SYMBOL_ACTION!r}")
    names = [order.subject] if isinstance(order.subject, str) else list(order.subject or [])
    if not names or not all(isinstance(n, str) and n.strip() and len(n.strip()) <= 60 and re.fullmatch(r"[A-Za-z0-9 &.,'\u2019\-/()!]+", n.strip()) for n in names):
        reasons.append("a symbol lookup needs one or more tickers or company names as its subject (letters, digits, spaces and & . , ' - / ( ) !, at most 60 characters each)")
    elif len(names) > MAX_LOOKUP_NAMES:
        reasons.append(f"a symbol lookup is for at most {MAX_LOOKUP_NAMES} stocks at a time")
    if order.status in HAS_DATA:
        reasons.append(f"the order already has its data (status {order.status})")
    return reasons


METRIC_ACTION = "calculate_financials"
MAX_METRIC_SYMBOLS = 30   # as the quote order: the Advisor's code path asks for its candidates (at most 15 new ideas and the holdings), not the model
MAX_METRICS = 100         # the catalogue has 80 metrics (Advisor v1 spec 4.1 says 40, but the union of what the five families, the gates and the plan read is more than that: entry 51)


def sector_map(order: WorkOrder) -> dict[str, str] | None:
    """{symbol: sector ETF ticker} from the order's `sector`: a single ticker applies to every symbol of the subject, a map is taken as it is, nothing is {}. None when `sector` is not
    one of those (the check reports it)."""
    symbols = [order.subject] if isinstance(order.subject, str) else [s for s in (order.subject or []) if isinstance(s, str)]
    sector = order.sector
    if sector is None:
        return {}
    if isinstance(sector, str):
        return {s.strip().lstrip("$").upper(): sector.strip().upper() for s in symbols}
    if isinstance(sector, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in sector.items()):
        return {k.strip().lstrip("$").upper(): v.strip().upper() for k, v in sector.items()}
    return None


def verify_metric_request(order: WorkOrder) -> list[str]:
    """Why the Data Clerk may NOT answer this `calculate_financials` order (`DataClerk.request_metrics`); an empty list means it may. Like `verify_quote_request`, it has its own check:
    `verify_request` is the check of a SEARCH and still refuses `calculate_financials` (entry 25). The subject is 1 to 30 symbols, `metrics` a non-empty list of at most 100 requests
    whose names are in the catalogue (the message lists the unknown names and the known ones)."""
    reasons = []
    if order.action_type != METRIC_ACTION:
        reasons.append(f"action_type {order.action_type!r} is not {METRIC_ACTION!r}")
    symbols = [order.subject] if isinstance(order.subject, str) else list(order.subject or [])
    if not symbols or not all(isinstance(s, str) and s.strip() for s in symbols):
        reasons.append("a metrics request needs one or more ticker symbols as its subject")
    elif len(symbols) > MAX_METRIC_SYMBOLS:
        reasons.append(f"a metrics request is for at most {MAX_METRIC_SYMBOLS} symbols at a time")
    requests = order.metrics
    if not isinstance(requests, list) or not requests:
        reasons.append("a metrics request needs a non-empty list of metrics")
    elif len(requests) > MAX_METRICS:
        reasons.append(f"a metrics request asks for at most {MAX_METRICS} metrics at a time")
    elif not all(isinstance(r, dict) and isinstance(r.get("name"), str) and isinstance(r.get("params", {}), dict) for r in requests):
        reasons.append('each metric is {"name": text, "params": {...}} (params optional)')
    else:
        from prometheus.agents.data_clerk.metrics_catalog import METRIC_NAMES, sector_etfs  # imported here: the catalogue needs pyarrow and httpx, and the order is plain data
        unknown = [r["name"] for r in requests if r["name"] not in METRIC_NAMES]
        if unknown:
            reasons.append(f"unknown metrics: {', '.join(dict.fromkeys(unknown))}; known metrics: {', '.join(METRIC_NAMES)}")
        sectors = sector_map(order)
        bad = sorted({etf for etf in sectors.values() if etf not in sector_etfs()}) if sectors is not None else []
        if sectors is None and order.sector is not None:
            reasons.append("sector must be one ETF ticker or a {symbol: ETF ticker} map")
        elif bad:
            reasons.append(f"unknown sector ETF {', '.join(bad)}; known sectors: {', '.join(sector_etfs())}")
    if order.status in HAS_DATA:
        reasons.append(f"the order already has its data (status {order.status})")
    return reasons


BARS_ACTION = "fetch_bars"
MAX_BAR_SYMBOLS = 30   # as the quote and metric orders: the Account Manager asks for the symbols of the tickets and of the funding sells (at most 20 trades), plus the benchmark
MAX_BAR_DAYS = 400     # calendar days from `start` to `end`: a backtest follows a trade for at most `backtest.max_follow_days` (90) after the decision, and the store reads one file per trading day
BAR_KEYS = ("start", "end")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def iso_date(value) -> date | None:
    """"2026-11-05" -> a date; None for anything else. Only the plain YYYY-MM-DD form: Python's own `fromisoformat` also reads "20261105" and "2026-W45-4", and a date in a request is written one way."""
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def verify_bar_request(order: WorkOrder) -> list[str]:
    """Why the Data Clerk may NOT answer this `fetch_bars` order (`DataClerk.request_bars`); an empty list means it may. Like the quote, metric and verification orders it has its own check:
    `verify_request` is the check of a SEARCH and still refuses `fetch_bars`. The subject is 1 to 30 symbols; `bars` is exactly {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}, both real dates, `start`
    on or before `end`, at most 400 days apart; the order has no data yet. The Account Manager is the one that asks (`AccountManager.request_bars`), for the Trader."""
    reasons = []
    if order.action_type != BARS_ACTION:
        reasons.append(f"action_type {order.action_type!r} is not {BARS_ACTION!r}")
    symbols = [order.subject] if isinstance(order.subject, str) else list(order.subject or [])
    if not symbols or not all(isinstance(s, str) and s.strip() for s in symbols):
        reasons.append("a bars request needs one or more ticker symbols as its subject")
    elif len(symbols) > MAX_BAR_SYMBOLS:
        reasons.append(f"a bars request is for at most {MAX_BAR_SYMBOLS} symbols at a time")
    spec = order.bars
    if not isinstance(spec, dict) or set(spec) != set(BAR_KEYS):
        reasons.append('a bars request needs `bars` with exactly "start" and "end": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}')
    else:
        start, end = iso_date(spec["start"]), iso_date(spec["end"])
        if start is None or end is None:
            reasons.append("bars start and end must each be a date written YYYY-MM-DD")
        elif start > end:
            reasons.append(f"bars start {spec['start']} is after end {spec['end']}")
        elif (end - start).days > MAX_BAR_DAYS:
            reasons.append(f"a bars request covers at most {MAX_BAR_DAYS} days (from {spec['start']} to {spec['end']} is {(end - start).days})")
    if order.status in HAS_DATA:
        reasons.append(f"the order already has its data (status {order.status})")
    return reasons


def verify_action(order: WorkOrder) -> list[str]:
    """Whether the data on the order can be trusted; an empty list means it can (the design's verify_action, which checks row counts).
    "Nothing found" passes: it is a result, but only after a search: a search order (search_news or search_filing) whose answer involved no successful search fails. What fails is data that contradicts itself or that the Data Clerk itself flagged."""
    data = order.data
    if data is None:
        return ["the order has no data yet"]
    reasons = list(data.problems)
    tool, label = SEARCH_TOOL.get(order.source, SEARCH_TOOL["news"])  # each source has its own search tool: a press release order is not satisfied by a news search
    searched = any(c.get("name") == tool and c.get("outcome") == "ok" for c in data.tool_calls)
    if order.action_type in SEARCH_ACTIONS and not searched and not data.problems:
        # a search order must search. An answer without a search rests on no source: it is the model's memory or a refusal
        reasons.append(f"no search of the stored {label} succeeded (no {tool} call), so the answer rests on no source")
    if not data.answer.strip() and not reasons:  # when the Data Clerk already says why (a failed model call), that is the reason
        reasons.append("the answer is empty")
    known = {s.get("chunk_id") for s in data.sources}
    missing = [r.get("chunk_id") for r in data.references if r.get("chunk_id") not in known]
    if missing:
        reasons.append(f"{len(missing)} cited chunk(s) are not among the sources: {', '.join(map(str, missing))[:120]}")
    if len(data.references) > len(data.sources):
        reasons.append("more references than sources")
    if data.no_supporting_text and any(not s.get("swept") for s in data.sources):  # the sweep's sources (entry 40) were found after the model's search: they do not contradict "nothing found" there
        reasons.append("the data says no supporting text but also lists sources")
    return reasons


def attach_result(order: WorkOrder, result) -> WorkOrder:
    """Put a Data Clerk `Result` on the order and set the status from `verify_action`. Returns the same order."""
    fields = {k: v for k, v in result.as_dict().items() if k in DataResult.model_fields}
    order.data = DataResult(**fields, results=group_results(fields.get("sources", []), fields.get("references", [])), completed_utc=_now())
    reasons = verify_action(order)
    order.errors = [*order.errors, *reasons]
    order.status = "data_failed" if reasons else "data_ready"
    return order


# ---------------------------------------------------------------- Critic v1 (design choices entries 56 to 59)

VERIFY_ACTION = "verify_data"
MAX_VERIFY_CHUNKS = 60    # chunk ids a `verify_data` order may ask to read again (the Critic sends a second order for more)
MAX_VERIFY_SYMBOLS = 30   # as the quote and metric orders
MAX_CHUNK_ID = 200
MAX_FRESH_DAYS = 30       # the stored news and press releases the Critic asks for as "newer": the Data Clerk's own discover limit
MAX_FRESH_K = 20          # the retriever's MAX_K
VERIFY_KEYS = ("chunks", "evidence", "symbols", "company", "fresh")
EVIDENCE_ID = re.compile(r"E\d+")
WILL_TRADE = ("yes_auto", "yes_user", "pending_user", "no_system_denied", "no_expired", "no_funds", "no_user_denied")  # the week 5 design's WillTrade: Yes (user approved), No (system denied), No (no funds), No (user denied)
TRADING = ("yes_auto", "yes_user")                  # the only two decisions that let the Trader trade
NEUTRAL = "neutral"                                 # the `origin` of a decision for a company the Advisor scored NEUTRAL while ALLOW_NEUTRAL_RECOMMENDATIONS is on (entry 91): no review and no plan; the Advisor's own `order.candidates` record is its permission
DIRECT = "direct"                                   # the `origin` of a decision made for a trade the USER named (entry 87): no recommendation, no Critic review; the order's `direct_orders` is its permission
DIRECT_SIDES = ("buy", "sell")
DIRECT_KEYS = ("record_id", "side", "symbol", "company", "shares", "request", "requested_utc")   # what a record of `order.direct_orders` holds: `company` is the company's name as the data source gives it (entry 88)
DIRECT_OPTIONAL = ("typed_company", "how_many", "found_by", "dollars")  # `dollars` (entry 122): the money the user wrote for THIS stock ("Buy $3,000 of XOM"), a number above zero to the cent; the order then has no shares until the Account Manager sizes it (`how_many` "dollars"). entry 88: what the user typed when they named a COMPANY and not a ticker (the `symbol` is then the Data Clerk's); None when they typed the ticker. `found_by` (entry 101): how the Data Clerk found it, "ticker", "name" or "brand" (the symbol is the LISTED OWNER's of what was typed: "Google" is Alphabet Inc.)
MAX_DIRECT_SHARES = 1_000_000                       # a typing mistake, not a limit of the account: the cash and the holdings are what limit a trade. The number of shares may have a decimal part (entry 110)


def verify_verification_request(order: WorkOrder) -> list[str]:
    """Why the Data Clerk may NOT answer this `verify_data` order (`DataClerk.request_verification`); an empty list means it may. Like the quote and metric orders it has its own check
    (`verify_request` is the check of a SEARCH). `order.verify` is {"chunks": [chunk ids], "evidence": {symbol: [E ids]}, "symbols": [tickers], "company": {symbol: name}, "fresh": {"since_days": 1 to 30,
    "k": 1 to 20}}; every key is optional, at least one of chunks, evidence and symbols is needed, and `fresh` needs symbols (it is read for each of them)."""
    from prometheus.rag.loaders import check_symbol  # imported here: the order is plain data

    reasons = []
    if order.action_type != VERIFY_ACTION:
        reasons.append(f"action_type {order.action_type!r} is not {VERIFY_ACTION!r}")
    spec = order.verify
    if not isinstance(spec, dict):
        return [*reasons, "a verification request needs `verify`: what to read again"]
    unknown = [k for k in spec if k not in VERIFY_KEYS]
    if unknown:
        reasons.append(f"unknown verify keys: {', '.join(map(str, unknown))}; known keys: {', '.join(VERIFY_KEYS)}")

    def symbol_ok(value) -> bool:
        try:
            return isinstance(value, str) and check_symbol(value.strip().lstrip("$")) == value
        except ValueError:
            return False

    chunks, evidence, symbols = spec.get("chunks", []), spec.get("evidence", {}), spec.get("symbols", [])
    if not isinstance(chunks, list) or not all(isinstance(c, str) and 0 < len(c) <= MAX_CHUNK_ID and not any(ch.isspace() for ch in c) for c in chunks):
        reasons.append(f"chunks must be a list of chunk ids (text of 1 to {MAX_CHUNK_ID} characters with no whitespace)")
    elif len(chunks) > MAX_VERIFY_CHUNKS:
        reasons.append(f"a verification request reads at most {MAX_VERIFY_CHUNKS} chunks at a time")
    if not isinstance(evidence, dict) or not all(symbol_ok(s) and isinstance(ids, list) and all(isinstance(i, str) and EVIDENCE_ID.fullmatch(i) for i in ids) for s, ids in evidence.items()):
        reasons.append('evidence must be {"SYMBOL": ["E3", ...]} with checked symbols and evidence ids like E3')
    if not isinstance(symbols, list) or not all(symbol_ok(s) for s in symbols):
        reasons.append("symbols must be a list of checked ticker symbols")
    elif len(symbols) > MAX_VERIFY_SYMBOLS:
        reasons.append(f"a verification request is for at most {MAX_VERIFY_SYMBOLS} symbols at a time")
    company = spec.get("company", {})
    if not isinstance(company, dict) or not all(symbol_ok(s) and isinstance(name, str) and len(name) <= 120 for s, name in company.items()):
        reasons.append('company must be {"SYMBOL": "name"}')
    fresh = spec.get("fresh")
    if fresh is not None:
        days, k = (fresh.get("since_days", 14), fresh.get("k", 8)) if isinstance(fresh, dict) else (None, None)
        if not isinstance(fresh, dict) or set(fresh) - {"since_days", "k"}:
            reasons.append('fresh must be {"since_days": 1 to 30, "k": 1 to 20}')
        elif not (isinstance(days, int) and not isinstance(days, bool) and 1 <= days <= MAX_FRESH_DAYS and isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= MAX_FRESH_K):
            reasons.append(f"fresh needs since_days from 1 to {MAX_FRESH_DAYS} and k from 1 to {MAX_FRESH_K}")
        elif not symbols:
            reasons.append("fresh is read for each of `symbols`: give at least one")
    if not any(spec.get(k) for k in ("chunks", "evidence", "symbols")) and not reasons:
        reasons.append("there is nothing to verify: give chunks, evidence or symbols")
    if order.status in HAS_DATA:
        reasons.append(f"the order already has its data (status {order.status})")
    return reasons


def review_of(order: WorkOrder, record_id: str) -> dict | None:
    """The Critic's review of one recommendation (by its `identity.record_id`), or None."""
    return next((r for r in order.reviews if r.get("record_id") == record_id), None)


def decision_of(order: WorkOrder, record_id: str) -> dict | None:
    """The Account Manager's decision about one recommendation, or None."""
    return next((d for d in order.trade_decisions if d.get("record_id") == record_id), None)


def parse_utc(text) -> datetime | None:
    """"2026-11-05T15:30:00Z" (the form the desk writes) or any ISO 8601 text with an offset -> an aware datetime; None when it cannot be read or has no timezone."""
    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


EXPIRY_PREFIX = "the approval ran out"      # how `verify_trade` starts the sentence about an approval whose time is over: the Trader's `tickets.permission` and the Account Manager's `approve_trade` both turn a verdict whose ONLY reason is this into `approval_expired`


def is_neutral_candidate(candidate) -> bool:
    """Whether this record of `order.candidates` is a company the Advisor scored NEUTRAL (no strategy passed, or no edge) and that no gate stopped (entries 91 and 107): the only kind a neutral decision may be for. WHOEVER
    NAMED IT: the user's own list, a theme's companies found by the Data Clerk, a company the sources name: a neutral stance and no failed gate is all it needs (entry 107; it used to need `origin == "subject"`). A gate that
    failed (stale data, a red flag, an earnings report days away) is a refusal that has nothing to do with the stance, so it is never turned into an approval, and a bearish stance trades nothing. Whether the user already holds
    some of it does not matter and is not asked here (entry 105: the Advisor knows no holding; the Account Manager sizes the purchase and knows what is held)."""
    return isinstance(candidate, dict) and candidate.get("stance") == "neutral" and candidate.get("gate") is None and isinstance(candidate.get("symbol"), str)


def neutral_candidate_of(order: WorkOrder, symbol: str) -> dict | None:
    """The Advisor's neutral candidate record for this symbol (`is_neutral_candidate`), or None."""
    return next((c for c in order.candidates if is_neutral_candidate(c) and c["symbol"] == symbol), None)


def direct_order_of(order: WorkOrder, record_id: str) -> dict | None:
    """The user's own direct order (entry 87) that a decision answers, by `record_id`, or None."""
    return next((d for d in order.direct_orders if isinstance(d, dict) and d.get("record_id") == record_id), None)


def verify_direct_orders(entries: list) -> list[str]:
    """Why these records of `order.direct_orders` are not the user's orders as the Account Manager writes them; empty means they are: each is a record with the keys of `DIRECT_KEYS` (and optionally `DIRECT_OPTIONAL`), a
    `record_id` that is text of at most 80 characters and appears once, a side of `DIRECT_SIDES`, a checked ticker AND the company's name (the two the Data Clerk found and the Account Manager appended to the work order when
    it made it: entry 88), a number of shares above zero and at most `MAX_DIRECT_SHARES` (a whole number, or one with up to six decimals the user typed: entry 110), and the user's own words."""
    from prometheus.rag.loaders import check_symbol  # imported here: the order is plain data

    if not isinstance(entries, list) or not entries:
        return ["a direct order needs at least one trade the user named"]
    reasons, seen = [], set()
    for index, entry in enumerate(entries, 1):
        label = f"direct order #{index}"
        if not isinstance(entry, dict) or not set(DIRECT_KEYS) <= set(entry) <= set(DIRECT_KEYS) | set(DIRECT_OPTIONAL):
            reasons.append(f"{label} is not a record with the keys {', '.join(DIRECT_KEYS)} (and, optionally, {', '.join(DIRECT_OPTIONAL)})")
            continue
        if not isinstance(entry["record_id"], str) or not 0 < len(entry["record_id"].strip()) <= 80 or entry["record_id"] in seen:
            reasons.append(f"{label} needs a record_id that is text of at most 80 characters and is not used twice")
        seen.add(entry["record_id"])
        if entry["side"] not in DIRECT_SIDES:
            reasons.append(f"{label}: the side must be one of {', '.join(DIRECT_SIDES)}")
        try:
            if not isinstance(entry["symbol"], str) or check_symbol(entry["symbol"]) != entry["symbol"]:
                raise ValueError
        except ValueError:
            reasons.append(f"{label}: the symbol is not a checked ticker")
        if not (isinstance(entry.get("company"), str) and entry["company"].strip()):
            reasons.append(f"{label}: the work order has no company name for it (the Data Clerk finds it and the Account Manager appends it)")
        if entry.get("typed_company") is not None and (not isinstance(entry["typed_company"], str) or not entry["typed_company"].strip()):
            reasons.append(f"{label}: typed_company is text, or None when the user typed the ticker")
        shares = entry["shares"]
        if shares is not None and not (0 < (_quantity(shares) or 0) <= MAX_DIRECT_SHARES):
            reasons.append(f"{label}: the shares must be a number above zero and at most {MAX_DIRECT_SHARES:,} (a whole number, or one with up to {sh.PLACES} decimals), or None while the user has given no number and the Account Manager has not sized the order (design choices entry 98)")
        if entry.get("found_by") not in (None, "ticker", "name", "brand"):
            reasons.append(f"{label}: found_by is ticker, name or brand (how the Data Clerk found the symbol), or None")
        if entry.get("how_many") not in (None, "typed", "asked", "cash", "budget", "dollars"):
            reasons.append(f"{label}: how_many is typed, asked, cash, budget or dollars (where the number of shares came from), or None")
        if entry.get("dollars") is not None:
            try:
                if money(entry["dollars"]) <= 0:
                    raise ValueError
            except ValueError:
                reasons.append(f"{label}: dollars is a number above zero to the cent (what the user wrote for this stock), or left out")
            else:
                if entry["shares"] is not None and entry.get("how_many") in (None, "typed"):
                    reasons.append(f"{label}: an order has shares or dollars, never both: the user typed both")
        if not isinstance(entry["request"], str) or not entry["request"].strip():
            reasons.append(f"{label} carries no words of the user's")
    return reasons


def verify_trade(order: WorkOrder, record_id: str, now: datetime) -> list[str]:
    """Why the Trader may NOT trade this recommendation; an empty list means it may. This is the Trader's precondition (rule T1 of design/trader_v1_final_spec.md): `tickets.permission` asks it when a ticket is built and
    `PaperBroker.place` asks it again for every trade it places (the model's `place_order`, the sweep, and a resting order judged again), so that no code path can trade without a verdict and a permission. A trade needs ALL of: a review of the recommendation that approved it; a decision that says `yes_auto` or `yes_user`; an approval that has not run out (`valid_until_utc`:
    the price check is old after it); a number of shares above zero (a whole number, or one with a fraction of at most six decimals: design choices entry 110). The number of shares is the Account Manager's (the decision's): the Critic approves a recommendation and no size, so there is no review number to compare it with.

    A DIRECT ORDER (design choices entry 87) is the one other way to a trade: the user named it, so there is no recommendation and no review, and the user's own order is the approval. A decision with `origin` "direct"
    needs, instead of the review, the user's record on `order.direct_orders` (the same `record_id`, symbol and side: a buy or a sell) and not more shares than the user typed; its approval runs out at the decision's own
    `valid_until_utc`. A decision that says it is direct and finds no such record on the order is refused, so the word alone permits nothing.

    A NEUTRAL RECOMMENDATION (design choices entries 91 and 107) is the third way: while the Account Manager's setting ALLOW_NEUTRAL_RECOMMENDATIONS is on, a company the Advisor scored neutral (whoever named it) counts as approved.
    A decision with `origin` "neutral" needs, instead of the review, the Advisor's own neutral record of that symbol on `order.candidates` (`is_neutral_candidate`), is only a purchase, has no review beside it, and runs out at
    its own `valid_until_utc`. The word alone permits nothing here either, and a Critic denial is never turned into this: a denied recommendation has a review, and a neutral candidate never had one."""
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be a datetime with a timezone")
    review, decision = review_of(order, record_id), decision_of(order, record_id)
    direct = direct_order_of(order, record_id) if decision is not None and decision.get("origin") == DIRECT else None
    neutral = decision is not None and decision.get("origin") == NEUTRAL
    reasons = []
    if neutral:
        # a company the Advisor scored neutral, while the Account Manager's setting treats that as approved (entry 91): the Advisor's own record on the order is the permission, and no review stands in for it
        if neutral_candidate_of(order, decision.get("symbol")) is None:
            reasons.append(f"the decision for {record_id} says it is a neutral recommendation, but the work order holds no neutral score of the Advisor's for {decision.get('symbol')!r}")
        if decision.get("action") != "open":
            reasons.append("a neutral recommendation is only a purchase")
        if review is not None:
            reasons.append("a neutral recommendation has no Critic review: the work order holds one for it")
    elif decision is not None and decision.get("origin") == DIRECT:
        if direct is None:
            reasons.append(f"the decision for {record_id} says it is the user's direct order, but the work order holds no such order of the user's")
        else:
            if decision.get("symbol") != direct["symbol"] or ("sell" if decision.get("action") == "reduce" else "buy") != direct["side"]:
                reasons.append(f"the decision is not the user's order: the user typed {direct['side']} {direct['symbol']}")
            if direct["shares"] is None:
                reasons.append("the user's order holds no number of shares: the Account Manager has not sized it (the user wrote none)")
            elif (given := _quantity(decision.get("shares"))) is not None and (typed := _quantity(direct["shares"])) is not None and given > typed:       # for every quantity, a fraction too (entry 110: this guard once ran for an int only)
                reasons.append(f"the decision gives {sh.text(given)} shares but the user typed {sh.text(typed)}" if direct.get("how_many") in (None, "typed")
                               else f"the decision gives {sh.text(given)} shares but the user's order holds {sh.text(typed)} ({direct['how_many']})")
        if review is not None:
            reasons.append("a direct order has no Critic review: the work order holds one for it")
    elif review is None:
        reasons.append(f"there is no Critic review of recommendation {record_id}")
    elif not review.get("approved"):
        reasons.append("the Critic did not approve it")
    if decision is not None and decision.get("origin") not in (DIRECT, NEUTRAL) and decision.get("action") != "open":
        reasons.append(f"the decision's action is {decision.get('action')!r}: a recommendation is a purchase to open, and a sale never comes from a recommendation (it is the user's own direct order or RaiseFunds; entry 105)")
    if decision is None:
        reasons.append(f"there is no trade decision for recommendation {record_id}")
    elif decision.get("will_trade") not in TRADING:
        reasons.append(f"the trade decision is {decision.get('will_trade')!r}, not one of {', '.join(TRADING)}")
    approval = decision if direct is not None or neutral else review if review is not None and review.get("approved") else None
    if approval is not None:
        until = parse_utc(approval.get("valid_until_utc"))
        if until is None:
            reasons.append("the approval has no valid_until_utc")
        elif now > until:
            reasons.append(f"{EXPIRY_PREFIX} at {approval['valid_until_utc']}: the price must be checked again")
    if decision is not None and decision.get("will_trade") in TRADING:
        if not (_quantity(decision.get("shares")) or 0) > 0:
            reasons.append("the decision needs a number of shares above zero to trade")
        try:
            dollars = decision_dollars(decision)
        except ValueError as error:
            reasons.append(str(error))
        else:
            if dollars is not None and (pot := decision_pot(decision)) is not None and dollars > pot:      # an order in dollars never passes the money of the request (M1 and M2)
                reasons.append(f"the decision's dollars, ${dollars:,.2f}, are more than the money of the request, ${pot:,.2f}")
            if dollars is not None and direct is not None and direct["shares"] is not None and direct.get("how_many") in (None, "typed"):
                reasons.append("a direct order whose shares the user typed is an order in shares: it carries no dollars")
            if direct is not None and direct.get("dollars") is not None and decision.get("action") != "reduce":      # the user's own dollars for THIS stock are its limit (entry 122): the decision's dollars, or its estimated cost, never pass them
                try:
                    own = money(direct["dollars"])
                except ValueError:
                    own = None
                spend = dollars if dollars is not None else _money_or_none(decision.get("est_cost"))
                if own is None or spend is None or spend > own:
                    reasons.append(f"the user wrote ${own:,.2f} for this stock" + (f" and the decision's {'dollars' if dollars is not None else 'estimated cost'} are ${spend:,.2f}" if own is not None and spend is not None else " and the decision does not say what it costs") + ": the user's dollars are the most it may cost")
            if dollars is not None and decision.get("action") == "reduce":
                reasons.append("a sale is an order in shares: it carries no dollars")
    return reasons


# ---------------------------------------------------------------- Trader v1 (design/trader_v1_final_spec.md)

TRADE_STATUSES = ("new", "active", "filled", "rejected")   # `new`: a ticket not placed yet; `active`: a resting order; `filled` and `rejected` are final. On a finished order a trade is never `new`
TRADE_KINDS = ("recommendation", "funding")                # a purchase or sale of a recommendation, or a sale that pays for a purchase (RaiseFunds)
TRADE_SIDES = ("buy", "sell")


def trade_of(order: WorkOrder, trade_id: str) -> dict | None:
    """One trade of the order (by its `trade_id`: "T1", or "T1-F1" for a sale that funds T1), or None."""
    return next((t for t in order.trades if isinstance(t, dict) and t.get("trade_id") == trade_id), None)


def trades_of(order: WorkOrder, record_id: str) -> list[dict]:
    """Every trade of the order that belongs to one recommendation (by `record_id`), in the order they are on the order: its own purchase or sale and any funding sells that were made for it. Empty when none."""
    return [t for t in order.trades if isinstance(t, dict) and t.get("record_id") == record_id]


def _positive(value) -> bool:
    """Whether `value` is a number above zero (a Decimal, or the int or float a JSON file gave: read by the digits it is written with); a bool and a string are not."""
    try:
        return digits(value) > 0
    except ValueError:
        return False


def _is_number(value) -> bool:
    """Whether `value` is a number (a Decimal, an int or a float; a bool and a string are not)."""
    try:
        digits(value)
        return True
    except ValueError:
        return False


def _quantity(value) -> Decimal | None:
    """`value` as a share quantity (`shares.quantity`: a Decimal, a fraction has at most six decimals), or None when it is not one: a bool, a string, a negative number, seven decimals. The guards on the
    number of shares use it, so that each runs for a fraction as it does for a whole number (entry 110)."""
    try:
        return sh.quantity(value)
    except ValueError:
        return None


def _money_or_none(value) -> Decimal | None:
    """`value` as dollars to the cent (the number a decision's `est_cost` holds), or None when it is not one."""
    try:
        return money(value)
    except ValueError:
        return None


def decision_dollars(decision) -> Decimal | None:
    """The DOLLARS of a decision that is an order in dollars (design choices entry 111, M2): the most the purchase may cost, a number above zero to the cent; None for an order in shares (a decision with no `dollars`).
    Raises ValueError when the decision carries a `dollars` that is not a number above zero to the cent: a bool, a string, nothing, a fraction of a cent."""
    if not isinstance(decision, dict) or decision.get("dollars") is None:
        return None
    value = decision["dollars"]
    try:
        dollars = money(value)      # a Decimal, or the int or float a JSON file gave, by its digits; a bool, a string and a fraction of a cent are refused
    except ValueError:
        raise ValueError(f"the decision's dollars must be a number above zero to the cent, not {value!r}") from None
    if dollars <= 0:
        raise ValueError(f"the decision's dollars must be a number above zero to the cent, not {value!r}")
    return dollars


def decision_pot(decision) -> Decimal | None:
    """The POT a decision was sized under (`decision["size"]["pot"]`: the budget the user gave, else the cash, never more than the cash, in dollars to the cent; design choices entry 111, M1), or None: a decision that was not sized
    from money (a direct order whose shares the user typed, a sale, a RaiseFunds purchase) has none and its limit is the cash."""
    pot = ((decision or {}).get("size") or {}).get("pot") if isinstance(decision, dict) else None
    try:
        value = digits(pot)
    except ValueError:
        return None
    return value.quantize(CENT, rounding=ROUND_DOWN) if value >= 0 else None      # the pot is a CEILING: a number with a fraction of a cent (a defect: the sizing writes cents) counts rounded down


def pot_reasons(order: WorkOrder) -> list[str]:
    """Why the fills of this work order break its POT (M1: a run never spends more than its money); empty means they do not, or that no decision carries a pot. The decisions that permit a purchase and carry a pot must all carry the SAME
    one (two different pots are a defect), and what the filled purchases of those decisions cost in all (the fills' amounts) is not above it. Checked by a program on the finished work order, whatever price each trade filled at."""
    pots = {}
    for decision in order.trade_decisions:
        if isinstance(decision, dict) and decision.get("will_trade") in TRADING and decision.get("action") != "reduce" and decision_pot(decision) is not None:
            pots[decision.get("record_id")] = decision_pot(decision)
    if not pots:
        return []
    if len(set(pots.values())) > 1:
        return [f"the purchases of this work order carry different pots: {', '.join(f'${p:,.2f}' for p in sorted(set(pots.values())))}: a run has one"]
    pot = next(iter(pots.values()))
    spent = sum((digits((t.get("fill") or {}).get("amount") or 0) for t in order.trades
                 if isinstance(t, dict) and t.get("status") == "filled" and t.get("side") == "buy" and t.get("kind") != "funding" and t.get("record_id") in pots and isinstance(t.get("fill"), dict)), Decimal(0))
    return [f"the fills of the purchases cost ${spent:,.2f}, more than the money of the request, ${pot:,.2f} (the pot): a run never spends more than its money"] if spent > pot else []


@traced("trader.verify_trades")
def verify_trades(order: WorkOrder) -> list[str]:
    """The Trader's OUTPUT guard over the WHOLE run (rule T10 and step 8 of the spec): why these trades do NOT agree with the Account Manager's decisions; an empty list means they do. Since the per-trade handshake
    (design choices entry 81) each fill is checked one at a time before it is reported (`verify_fill`) and booked as it happens, so this is a REPORT: what was booked stays booked, it is the account book's.
    A reason for each of: a trade that is not a record at all, one whose `trade_id` is missing or appears twice, or whose status is not one of `TRADE_STATUSES`; a `rejected` trade with no code; a `filled`
    trade with no fill, or a fill whose price or share count is not positive; a funding trade that is not a sale, or whose `funds_for` names no purchase on the order; a `filled` funding sale whose purchase is not `filled` (holdings sold and nothing bought: the Trader's order of work prevents it, and this is the check behind that); and, for a `filled` trade of a
    recommendation: a `record_id` with no decision that says `yes_auto` or `yes_user`, a symbol or a side that is not the decision's (a sale only for a `reduce`), a SALE that is not the user's own direct order (a
    sale tied to a recommendation: it never comes from one, entry 105), more shares than the decision gave, and a second filled trade for the same recommendation. A trade that is only `active` or `rejected` cannot move the account, so it is held to the shape above and not to the decision.
    AND THE POT (design choices entry 111, M1; `pot_reasons`): the decisions of the purchases that carry a pot all carry the same one, and the fills of those purchases cost no more than it in all."""
    reasons: list[str] = []
    seen: set[str] = set()
    by_id = {t.get("trade_id"): t for t in order.trades if isinstance(t, dict)}
    filled_for: dict[str, str] = {}   # record_id -> the trade_id of the first filled recommendation trade
    for index, trade in enumerate(order.trades):
        reasons += _trade_reasons(order, trade, index, by_id, seen, filled_for, whole_run=True)
    return reasons + pot_reasons(order)      # and the pot (M1)


@traced("trader.verify_fill")
def verify_fill(order: WorkOrder, trade: dict, others: list[dict]) -> list[str]:
    """The single-trade part of the output guard (design choices entry 81, amendment 2 of design/trader_handshake_spec.md): why ONE trade, as the Trader is about to report it to the Account Manager as filled, does not agree
    with the decisions; an empty list means it may be reported and booked. `others` are the other trades of the order (this one is left out if it is among them). It checks what `verify_trades` checks of one trade: its shape
    (a trade_id that no other trade has, a status of the list, a fill with a positive price and a number of shares above zero), the decision (it says `yes_auto` or `yes_user`, the symbol and the side are its, the shares are
    not above its shares), that no other trade of the order is already filled for the same recommendation, and a funding sale's `funds_for` (a purchase on the order, and it is a sale). What it does NOT check is
    the one thing that cannot be known yet: that the purchase a funding sale pays for is filled (it is placed after its sales): the whole-run `verify_trades` reports that at the end."""
    me = trade.get("trade_id") if isinstance(trade, dict) else None
    rest = [t for t in others if isinstance(t, dict) and t.get("trade_id") != me]
    by_id = {t.get("trade_id"): t for t in rest}
    if isinstance(trade, dict):
        by_id[me] = trade
    seen = {t.get("trade_id") for t in rest if isinstance(t.get("trade_id"), str) and t.get("trade_id")}
    filled_for = {t["record_id"]: str(t.get("trade_id")) for t in rest if t.get("status") == "filled" and t.get("kind") != "funding" and isinstance(t.get("record_id"), str)}
    return _trade_reasons(order, trade, 0, by_id, seen, filled_for, whole_run=False)


def _trade_reasons(order: WorkOrder, trade, index: int, by_id: dict, seen: set, filled_for: dict, whole_run: bool) -> list[str]:
    """The reasons one trade disagrees with the order (the body of `verify_trades`); `seen` and `filled_for` are changed in place as the loop of `verify_trades` goes on. `whole_run` False leaves out the check that
    needs the whole run (a filled funding sale whose purchase is not yet filled)."""
    reasons: list[str] = []
    if not isinstance(trade, dict):
        return [f"trade #{index + 1} is not a record"]
    trade_id = trade.get("trade_id")
    label = f"trade {trade_id}" if isinstance(trade_id, str) and trade_id else f"trade #{index + 1}"
    if not isinstance(trade_id, str) or not trade_id:
        reasons.append(f"{label} has no trade_id")
    elif trade_id in seen:
        reasons.append(f"{label}: the trade_id appears more than once")
    else:
        seen.add(trade_id)
    status = trade.get("status")
    if status not in TRADE_STATUSES:
        reasons.append(f"{label}: status {status!r} is not one of {', '.join(TRADE_STATUSES)}")
    if status == "rejected" and not (isinstance(trade.get("reject_code"), str) and trade["reject_code"].strip()):
        reasons.append(f"{label} is rejected but has no reject_code")
    fill = trade.get("fill")
    if status == "filled":
        if not isinstance(fill, dict):
            reasons.append(f"{label} is filled but has no fill")
        elif not _positive(fill.get("price")) or not (_quantity(fill.get("shares")) or 0) > 0:
            reasons.append(f"{label}: the fill needs a price and a number of shares above zero")
    if trade.get("kind") == "funding":
        parent = by_id.get(trade.get("funds_for"))
        if trade.get("side") != "sell":
            reasons.append(f"{label} is a funding trade but is not a sale")
        if parent is None or parent.get("side") != "buy" or parent.get("kind") == "funding":
            reasons.append(f"{label} funds {trade.get('funds_for')!r}, which is not a purchase on this order")
        elif whole_run and status == "filled" and parent.get("status") != "filled":   # holdings sold and nothing bought is the worst case (spec section 7): the Trader's order of work prevents it, this is the check behind that
            reasons.append(f"{label} is a filled funding sale but the purchase it pays for, {trade.get('funds_for')}, is {parent.get('status')}, not filled")
    elif status == "filled":
        record_id = trade.get("record_id")
        decision = decision_of(order, record_id) if isinstance(record_id, str) else None
        if decision is None or decision.get("will_trade") not in TRADING:
            said = "no trade decision" if decision is None else f"the decision {decision.get('will_trade')!r}"
            reasons.append(f"{label} is filled but recommendation {record_id} has {said}, not one of {', '.join(TRADING)}")
        else:
            if trade.get("symbol") != decision.get("symbol"):
                reasons.append(f"{label} is for {trade.get('symbol')!r} but the decision is for {decision.get('symbol')!r}")
            if trade.get("side") != ("sell" if decision.get("action") == "reduce" else "buy"):
                reasons.append(f"{label} is a {trade.get('side')} but the decision's action is {decision.get('action')!r}")
            if trade.get("side") == "sell" and decision.get("origin") != DIRECT:
                reasons.append(f"{label} is a sale tied to recommendation {record_id}: a sale comes only from the user's own direct order or from RaiseFunds (a funding sale), never from a recommendation")
            shares = _quantity(fill.get("shares")) if isinstance(fill, dict) else None
            try:
                dollars = decision_dollars(decision)
            except ValueError:
                dollars = None       # `verify_trade` says what is wrong with the decision's dollars
            if dollars is not None and trade.get("side") == "buy":
                # an ORDER IN DOLLARS (entry 111, M2): its shares are an estimate, so what is checked is what the fill COST, worked out here from its shares and its price (and what the fill says it cost): never above the dollars
                from prometheus.common.portfolio import trade_amount

                cost = trade_amount(shares, price(fill["price"])) if shares is not None and isinstance(fill, dict) and _positive(fill.get("price")) else None
                said = digits(fill["amount"]) if isinstance(fill, dict) and _is_number(fill.get("amount")) else None
                for what, amount in (("cost", cost), ("says it cost", said)):
                    if amount is not None and cents(amount) > dollars:
                        reasons.append(f"{label} {what} ${amount:,.2f} but the decision's dollars are ${dollars:,.2f}")
            else:
                gave = _quantity(decision.get("shares"))
                if shares is not None and gave is not None and shares > gave:                      # for every quantity, a fraction too (entry 110: this guard once ran when both were ints)
                    reasons.append(f"{label} filled {sh.text(shares)} shares but the decision gave {sh.text(gave)}")
        if isinstance(record_id, str):
            if record_id in filled_for:
                reasons.append(f"{label} is a second filled trade for recommendation {record_id} (the first is {filled_for[record_id]})")
            else:
                filled_for[record_id] = str(trade_id)
    return reasons
