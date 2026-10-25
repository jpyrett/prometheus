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

STATUS moves one way:  new -> data_ready | data_failed -> recommended.
`data_ready` means the Data Clerk did its job. That includes an honest "the stored news does not cover it" (`data.no_supporting_text`):
nothing to find is a result, not a failure. `data_failed` means something is wrong: an invalid citation, an empty answer, a model
failure, or a request the Data Clerk should not have taken.

The chunks travel with the order (`data.sources` holds every chunk the model was shown, with its text) so the Critic and the audit
log can check what a score was based on.
"""
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

ACTION_TYPES = ("trade", "search_news", "fetch_quote", "search_socials", "calculate_financials")
STATUSES = ("new", "data_ready", "data_failed", "recommended")
WIRED_SOURCES = ("news",)  # the sources the Data Clerk can search today; filings, press releases and the rest are not wired into the search yet
MAX_QUERY = 1000
MAX_TOP_K = 20
MAX_SINCE_DAYS = 90


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class DataResult(BaseModel):
    """What the Data Clerk put on the order: the `Result` of its agent, as plain data."""
    answer: str = ""
    references: list[dict] = Field(default_factory=list)  # the chunks the answer cites
    sources: list[dict] = Field(default_factory=list)     # every chunk the model was shown, with its number and text
    problems: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    no_supporting_text: bool = False
    tool_calls: list[dict] = Field(default_factory=list)
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    completed_utc: str = ""


class WorkOrder(BaseModel):
    schema_version: int = 1
    request_id: str
    requester: str = "user"
    created_utc: str
    query: str                                  # the request in the user's own words: the Data Clerk's question
    action_type: Literal["trade", "search_news", "fetch_quote", "search_socials", "calculate_financials"] = "search_news"
    subject: str | list[str] | None = None      # a symbol, a list of symbols, or a term such as "memory stocks"
    company: str | None = None                  # the company's name when the symbol was looked up from a name ("Procter & Gamble Co"), so that the Data Clerk can use it
    source: str = "news"
    time_period: str | None = None              # in words ("last 7 days")
    since_days: int | None = None               # the same, as a number of days
    live: bool = False                          # needs fresh, uncached data
    top_k: int = 5
    criteria: dict | None = None                # for the Advisor, for example {"sentiment": "bullish"}
    status: Literal["new", "data_ready", "data_failed", "recommended"] = "new"
    data: DataResult | None = None
    recommendations: list[dict] = Field(default_factory=list)
    advisor_runs: list[dict] = Field(default_factory=list)  # one record per Advisor call: model, whether it had evidence, tokens, what it dropped
    errors: list[str] = Field(default_factory=list)


def new_work_order(query: str, *, requester: str = "user", action_type: str = "search_news", subject=None, source: str = "news",
                   time_period: str | None = None, since_days: int | None = None, top_k: int = 5, live: bool = False,
                   criteria: dict | None = None, company: str | None = None) -> WorkOrder:
    """A new order, with an id and a timestamp. It is not checked here: `verify_request` is the check, so that a bad request becomes
    a reported reason and not an exception halfway through the desk."""
    return WorkOrder.model_construct(
        schema_version=1, request_id=uuid.uuid4().hex[:12], requester=requester, created_utc=_now(), query=query, action_type=action_type,
        subject=subject, company=company, source=source, time_period=time_period, since_days=since_days, live=live, top_k=top_k, criteria=criteria,
        status="new", data=None, recommendations=[], advisor_runs=[], errors=[])


def verify_request(order: WorkOrder) -> list[str]:
    """Why the Data Clerk may NOT work on this order; an empty list means it may (the design's verify_request)."""
    reasons = []
    if not isinstance(order.query, str) or not order.query.strip():
        reasons.append("the query is empty")
    elif len(order.query) > MAX_QUERY:
        reasons.append(f"the query is longer than {MAX_QUERY} characters")
    if order.action_type not in ACTION_TYPES:
        reasons.append(f"action_type {order.action_type!r} is not one of {', '.join(ACTION_TYPES)}")
    elif order.action_type != "search_news":
        reasons.append(f"action_type {order.action_type!r} is not handled by the Data Clerk's request_data yet (only search_news)")
    if not isinstance(order.top_k, int) or isinstance(order.top_k, bool) or not 1 <= order.top_k <= MAX_TOP_K:
        reasons.append(f"top_k must be a whole number from 1 to {MAX_TOP_K}")
    if order.since_days is not None and (not isinstance(order.since_days, int) or isinstance(order.since_days, bool)
                                         or not 1 <= order.since_days <= MAX_SINCE_DAYS):
        reasons.append(f"since_days must be a whole number of days from 1 to {MAX_SINCE_DAYS}")
    if order.source not in WIRED_SOURCES:
        reasons.append(f"source {order.source!r} is not wired yet (only: {', '.join(WIRED_SOURCES)})")
    if order.status in ("data_ready", "recommended"):
        reasons.append(f"the order already has its data (status {order.status})")
    return reasons


def verify_action(order: WorkOrder) -> list[str]:
    """Whether the data on the order can be trusted; an empty list means it can (the design's verify_action, which checks row counts).
    "Nothing found" passes: it is a result, but only after a search: a search_news order whose answer involved no successful search fails. What fails is data that contradicts itself or that the Data Clerk itself flagged."""
    data = order.data
    if data is None:
        return ["the order has no data yet"]
    reasons = list(data.problems)
    searched = any(c.get("name") == "retrieve_news" and c.get("outcome") == "ok" for c in data.tool_calls)
    if order.action_type == "search_news" and not searched and not data.problems:
        # a news-search order must search. An answer without a search rests on no source: it is the model's memory or a refusal
        reasons.append("no search of the stored news succeeded (no retrieve_news call), so the answer rests on no source")
    if not data.answer.strip() and not reasons:  # when the Data Clerk already says why (a failed model call), that is the reason
        reasons.append("the answer is empty")
    known = {s.get("chunk_id") for s in data.sources}
    missing = [r.get("chunk_id") for r in data.references if r.get("chunk_id") not in known]
    if missing:
        reasons.append(f"{len(missing)} cited chunk(s) are not among the sources: {', '.join(map(str, missing))[:120]}")
    if len(data.references) > len(data.sources):
        reasons.append("more references than sources")
    if data.no_supporting_text and data.sources:
        reasons.append("the data says no supporting text but also lists sources")
    return reasons


def attach_result(order: WorkOrder, result) -> WorkOrder:
    """Put a Data Clerk `Result` on the order and set the status from `verify_action`. Returns the same order."""
    fields = {k: v for k, v in result.as_dict().items() if k in DataResult.model_fields}
    order.data = DataResult(**fields, completed_utc=_now())
    reasons = verify_action(order)
    order.errors = [*order.errors, *reasons]
    order.status = "data_failed" if reasons else "data_ready"
    return order
