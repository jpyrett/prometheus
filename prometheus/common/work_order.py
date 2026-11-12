"""The work order: the one JSON object the agents pass to each other (design choices entry 25).

The week 2 design says every agent reads and updates the same `<WORK_ORDER>`: the Account Manager creates it, the Data Clerk
fills in the data, the Advisor appends a recommendation, the Critic judges it. This is that object, as far as the Data Clerk and
the Advisor need it today (the Account Manager, Critic and Trader fields come with those agents).

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

STATUS moves one way:  new -> data_ready | data_failed -> recommended -> reviewed -> decided   (the Critic sets `reviewed`, the Account Manager `decided`: Critic v1, design choices entry 59).
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

The chunks travel with the order (`data.sources` holds every chunk the model was shown, with its text) so the Critic and the audit
log can check what a score was based on.
"""
import re
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

ACTION_TYPES = ("trade", "search_news", "search_filing", "fetch_quote", "search_socials", "calculate_financials", "verify_data")
SEARCH_ACTIONS = ("search_news", "search_filing")   # what the Data Clerk's request_data handles today: a search of stored text (the other actions are refused by verify_request)
ACTION_SOURCE = {"search_filing": "filings"}       # an action that belongs to one source, and the source that has only that action: both directions are checked (entry 37)
SOURCE_ACTION = {source: action for action, source in ACTION_SOURCE.items()}
STATUSES = ("new", "data_ready", "data_failed", "recommended", "reviewed", "decided")
HAS_DATA = ("data_ready", "recommended", "reviewed", "decided")  # the statuses of an order that already has its data: asking the Data Clerk again changes nothing
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
    action_type: Literal["trade", "search_news", "search_filing", "fetch_quote", "search_socials", "calculate_financials", "verify_data"] = "search_news"
    subject: str | list[str] | None = None      # a symbol, a list of symbols, or a term such as "memory stocks". Press releases and filings: ARGUMENT #1, the symbol (one)
    company: str | None = None                  # the company's name: for news, when the symbol was looked up from a name ("Procter & Gamble Co"), so that the Data Clerk can use it.
                                                # Press releases and filings: ARGUMENT #2, the name to look up when the symbol is not given or not found
    source: str = "news"                        # news | press_releases | filings. For press_releases and filings the Data Clerk resolves symbol/company first (`resolved`)
    time_period: str | None = None              # in words ("last 7 days")
    since_days: int | None = None               # the same, as a number of days
    live: bool = False                          # needs fresh, uncached data
    top_k: int = 5
    criteria: dict | None = None                # for the Advisor, for example {"sentiment": "bullish"}
    status: Literal["new", "data_ready", "data_failed", "recommended", "reviewed", "decided"] = "new"
    resolved: dict | None = None                # press releases and filings: what the lookup decided: typed_symbol, typed_company, symbol, company, how ("ticker" or "name")
    data: DataResult | None = None
    recommendations: list[dict] = Field(default_factory=list)
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
    trade_decisions: list[dict] = Field(default_factory=list)  # the Account Manager: one decision per review (will_trade, funds_avail, ...): the Trader's permission


def new_work_order(query: str, *, requester: str = "user", action_type: str = "search_news", subject=None, source: str = "news",
                   time_period: str | None = None, since_days: int | None = None, top_k: int = 5, live: bool = False,
                   criteria: dict | None = None, company: str | None = None, metrics: list[dict] | None = None, case: str | None = None,
                   sector: str | dict[str, str] | None = None, verify: dict | None = None) -> WorkOrder:
    """A new order, with an id and a timestamp. It is not checked here: `verify_request` is the check, so that a bad request becomes
    a reported reason and not an exception halfway through the desk."""
    return WorkOrder.model_construct(
        schema_version=2 if (metrics is not None or case is not None or sector is not None or verify is not None) else 1, request_id=uuid.uuid4().hex[:12], requester=requester, created_utc=_now(), query=query, action_type=action_type,
        subject=subject, company=company, source=source, time_period=time_period, since_days=since_days, live=live, top_k=top_k, criteria=criteria,
        status="new", resolved=None, data=None, recommendations=[], advisor_runs=[], errors=[], metrics=metrics, case=case, sector=sector, candidates=[], verify=verify,
        reviews=[], critic_runs=[], trade_decisions=[])


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


def verify_trade(order: WorkOrder, record_id: str, now: datetime) -> list[str]:
    """Why the Trader may NOT trade this recommendation; an empty list means it may. The Trader is not built: this is its precondition, tested now, so that no later code path can trade without a
    verdict and a permission. A trade needs ALL of: a review of the recommendation that approved it; a decision that says `yes_auto` or `yes_user`; an approval that has not run out (`valid_until_utc`:
    the price check is old after it); at least one share. A review and a decision that disagree about the recommendation or the shares are refused."""
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be a datetime with a timezone")
    review, decision = review_of(order, record_id), decision_of(order, record_id)
    reasons = []
    if review is None:
        reasons.append(f"there is no Critic review of recommendation {record_id}")
    elif not review.get("approved"):
        reasons.append("the Critic did not approve it")
    if decision is None:
        reasons.append(f"there is no trade decision for recommendation {record_id}")
    elif decision.get("will_trade") not in TRADING:
        reasons.append(f"the trade decision is {decision.get('will_trade')!r}, not one of {', '.join(TRADING)}")
    if review is not None and review.get("approved"):
        until = parse_utc(review.get("valid_until_utc"))
        if until is None:
            reasons.append("the approval has no valid_until_utc")
        elif now > until:
            reasons.append(f"the approval ran out at {review['valid_until_utc']}: the price must be checked again")
    if decision is not None and decision.get("will_trade") in TRADING:
        shares = decision.get("shares")
        if not isinstance(shares, int) or isinstance(shares, bool) or shares < 1:
            reasons.append("the decision has no whole number of shares to trade")
        elif review is not None and isinstance(review.get("approved_shares"), int) and shares > review["approved_shares"]:
            reasons.append(f"the decision is for {shares} shares but the Critic approved {review['approved_shares']}")
    return reasons
