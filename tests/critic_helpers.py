"""Helpers the Critic tests share: a Data Clerk that replays the data an Advisor run was made from, a model that reads the prompt it is given, a profile, and a clock.

The three real captures of `tests/fixtures/advisor_real` (SNDK, PG, PG_open) are the inputs of the Critic's end-to-end tests: a REAL work order (real sources, real filing evidence), the REAL
metrics the Data Clerk calculated, and the REAL records the Advisor wrote from them. `FakeClerk` answers the Critic's three requests from that same data ("everything matches", the case a
healthy day gives), and each test changes ONE thing (a number, a quote, the clock) to prove a check can fail.

    order, fx = replay("PG_open")                       # the Advisor's step on the captured data (offline, free)
    critic = make_critic(order)                         # a Critic with a FakeClerk and a ReviewerModel
    critic.review_recommendations(order, view(), NOW)               # the Critic is given no portfolio and no money: sizing is the Account Manager's
"""
import copy
import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from advisor_helpers import MemoryLedger
from langchain_core.messages import AIMessage

from prometheus.agents.critic.critic import Critic
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, load_portfolio
from prometheus.common.user_profile import RiskView, UserProfile
from prometheus.rag.config import REPO_ROOT

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 5, 10, 30, tzinfo=ET)       # Thursday: the day the captures were made, in the regular session
PORTFOLIO = REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json"
PLACEHOLDER = "the captures hold no six-month average, EPS or P/S: these are INPUTS for the rules (a price 4% above its average), not market data"


def at(text: str) -> datetime:
    """An Eastern moment from "YYYY-MM-DD HH:MM"."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def profile(**kw) -> UserProfile:
    return UserProfile(**{"user": "Jacob Pyrett", "risk_tolerance": "moderate", **kw})


def view(**kw) -> RiskView:
    return profile(**kw).risk_view()


def portfolio() -> Portfolio:
    return load_portfolio(PORTFOLIO)


def metric_result(value, source="day_aggs:2025-10-02..2026-11-05", as_of="2026-11-05", unit="x", note=None):
    return {"value": value, "unit": unit, "as_of": as_of, "window": "test", "source": source, "inputs_summary": {}, "note": note}


class FakeClerk:
    """The Data Clerk as the Critic sees it. By default it answers from the data the Advisor was given: the same metrics, a quote at the last close, the order's own sources, every symbol listed.

    metrics      {symbol: {metric: result}}, default the order's own `data.metrics` plus `OWN` for every symbol
    quotes       {symbol: quote}, default the last close of each symbol, as of one minute before `now`
    fresh        {symbol: [hit, ...]}: newer stored text
    missing      chunk ids to answer as not found
    listed       {symbol: True | False | None}
    fail         names of requests that raise ("metrics", "quotes", "verification")
    """

    def __init__(self, order, now=NOW, metrics=None, quotes=None, fresh=None, missing=(), listed=None, fail=(), own=None, changes=None):
        self.order, self.now, self.fresh, self.missing, self.fail = order, now, fresh or {}, set(missing), set(fail)
        self.listed = listed or {}
        self.metrics = copy.deepcopy(metrics if metrics is not None else order.data.metrics)
        for symbol, results in self.metrics.items():
            last = (results.get("last_close") or {}).get("value")
            pct = (results.get("pct_of_52w_high") or {}).get("value")
            made = {"sma_126": last / 1.044, "high_126d": last * 1.02, "low_126d": last * 0.85, "eps": 5.0, "ps_ratio": 4.0, **({"high_52w": last / pct} if last and pct else {})} if last else {}
            for name, value in (own(symbol, last) if callable(own) else own if own is not None else made).items():
                results.setdefault(name, metric_result(value, source="day_aggs:2025-10-02..2026-11-05" if name != "high_52w" else "finnhub_metric:2026-11-06"))
        for symbol, updates in (changes or {}).items():     # {symbol: {metric: value}}: what the Critic's calculation says today, for a number to differ
            for name, value in updates.items():
                self.metrics[symbol][name] = {**self.metrics[symbol].get(name, metric_result(value)), "value": value}
        self.quotes = quotes
        self.asked = []

    # the three requests
    def request_metrics(self, order, today=None):
        self.asked.append(("metrics", list(order.subject), [m["name"] for m in order.metrics], today))
        if "metrics" in self.fail:
            raise RuntimeError("the metrics are down")
        order.data = wo.DataResult(metrics={s: {n: self.metrics[s][n] for n in (m["name"] for m in order.metrics) if n in self.metrics[s]} for s in order.subject if s in self.metrics})
        order.status = "data_ready"
        return order

    def request_quotes(self, order):
        self.asked.append(("quotes", list(order.subject)))
        if "quotes" in self.fail:
            raise RuntimeError("the quotes are down")
        stamp = (self.now - timedelta(minutes=1)).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        quotes = {}
        for symbol in order.subject:
            if self.quotes is not None:
                if symbol in self.quotes:
                    quotes[symbol] = self.quotes[symbol]
                continue
            last = (self.metrics.get(symbol) or {}).get("last_close", {}).get("value")
            if last:
                quotes[symbol] = {"symbol": symbol, "price": last, "previous_close": last, "as_of_utc": stamp, "source": "fake"}
        order.data = wo.DataResult(quotes=quotes)
        order.status = "data_ready"
        return order

    def request_verification(self, order, now=None):
        self.asked.append(("verification", dict(order.verify)))
        if "verification" in self.fail:
            raise RuntimeError("the store is down")
        spec, sources = order.verify, {s["chunk_id"]: s for s in self.order.data.sources}
        chunks = {}
        for chunk_id in spec.get("chunks", []):
            s = sources.get(chunk_id)
            chunks[chunk_id] = ({"found": False, "note": "not in the stored news"} if s is None or chunk_id in self.missing else
                                {"found": True, "corpus": wo.corpus_of(chunk_id), "article_id": s["article_id"], "headline": s["headline"], "publisher": s["publisher"],
                                 "published_utc": s["published_utc"], "url": s["url"], "tickers": list(s["tickers"]), "text": s["text"]})
        evidence = {}
        for symbol, ids in spec.get("evidence", {}).items():
            items = {i["id"]: i for i in (self.order.data.filing_evidence.get(symbol) or {}).get("evidence", [])}
            evidence[symbol] = {i: ({"found": True, "topic": items[i]["topic"], "filter": items[i]["filter"], "quote": items[i]["quote"], "what": items[i]["what"], "derived": items[i]["derived"],
                                     "source": {k: items[i]["source"].get(k) for k in ("accession", "form", "filed")}} if i in items else {"found": False, "note": "no such item"}) for i in ids}
        symbols = {s: {"listed": self.listed.get(s, True), "company": f"{s} INC" if self.listed.get(s, True) else None, "note": None if self.listed.get(s, True) is not False else "Finnhub's directory has no such symbol"}
                   for s in spec.get("symbols", [])}
        for s, flag in self.listed.items():
            if flag is None and s in symbols:
                symbols[s] = {"listed": None, "company": None, "note": "the lookup failed (HTTP 429)"}
        order.data = wo.DataResult(verification={"chunks": chunks, "evidence": evidence, "symbols": symbols, "fresh": {s: self.fresh.get(s, []) for s in spec.get("symbols", [])} if spec.get("fresh") else {}})
        order.status = "data_ready"
        return order


CLAIM_LINE = re.compile(r'^(\[\d+\]|E\d+|[a-z_0-9]+(?:#\d+)?) "', re.M)
SOURCE_HEAD = re.compile(r"^(\[\d+\]|E\d+|F\d+) (.*)$", re.M)


class ReviewerModel:
    """A chat model that READS the prompt it is given, as the real one must: it finds the claims and the fresh sources in the user message and answers each.

    labels   {ref: label} for chosen claims; every other claim gets `default`
    control  what it says about the planted control claim ("unrelated" is the right answer); the control is the claim whose source is headed "Community notes"
    fresh    {id: (label, quote)} for fresh sources; the others are "none"
    raw      a reply to give as it is (a string or an Exception), for a broken model
    """
    model_name = "z-ai/glm-5.3"

    def __init__(self, labels=None, default="supports", control="unrelated", fresh=None, raw=None):
        self.labels, self.default, self.control, self.fresh, self.raw = labels or {}, default, control, fresh or {}, list(raw) if isinstance(raw, list) else ([raw] if raw is not None else [])
        self.seen = []

    def invoke(self, messages):
        self.seen.append(messages)
        if self.raw:
            reply = self.raw.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return AIMessage(content=reply, usage_metadata={"input_tokens": 400, "output_tokens": 80, "total_tokens": 480})
        text = messages[-1].content
        claims_part = text.split("SOURCES of the claims")[0]
        refs = CLAIM_LINE.findall(claims_part.split("CLAIMS:", 1)[1]) if "CLAIMS:" in claims_part else []
        notice = next((m.group(1) for m in SOURCE_HEAD.finditer(text) if "Community notes" in m.group(2)), None)
        claims = [{"ref": r, "label": self.control if r == notice else self.labels.get(r, self.default), "reason": "read it in the source"} for r in refs]
        fresh_ids = re.findall(r"^(F\d+) ", text.split("FRESH SOURCES", 1)[-1], re.M) if "FRESH SOURCES (" in text else []
        fresh = [{"id": i, "label": self.fresh.get(i, ("none", None))[0], "quote": self.fresh.get(i, ("none", None))[1], "reason": "checked"} for i in fresh_ids]
        return AIMessage(content=json.dumps({"claims": claims, "fresh": fresh}), usage_metadata={"input_tokens": 400, "output_tokens": 80, "total_tokens": 480})


def make_critic(order, model=None, clerk=None, ledger=None, history=None, **clerk_options):
    """A Critic over fakes: a ReviewerModel (the right answers) and a FakeClerk over `order`'s own data, the reviews going to a memory ledger."""
    return Critic(model=model or ReviewerModel(), clerk=clerk or FakeClerk(order, **clerk_options), ledger=ledger if ledger is not None else MemoryLedger(), history=history)


def review(order, critic=None, view_=None, now=NOW, **options):
    """Review an order with a default Critic; returns the reviews as dicts keyed by symbol."""
    critic = critic or make_critic(order, **options)
    critic.review_recommendations(order, view_ or view(), now)
    return {r["symbol"]: r for r in order.reviews}


# ---------------------------------------------------------------- the captured Advisor runs, built once

import functools  # noqa: E402

from advisor_helpers import MemoryLedger as _MemoryLedger  # noqa: E402,F401
from test_advisor_sndk_pg import replay  # noqa: E402

from prometheus.agents.advisor.config import load_config as load_advisor_config  # noqa: E402
from prometheus.agents.advisor.record import Recommendation  # noqa: E402
from prometheus.agents.critic.config import load_config as load_critic_config  # noqa: E402

ADVISOR_CONFIG = load_advisor_config()
CONFIG = load_critic_config()


@functools.lru_cache(maxsize=None)
def _captured(name: str):
    return replay(name)


def captured(name: str):
    """A copy of the work order the Advisor made from a real capture ("SNDK", "PG", "PG_open"): a test may change it."""
    return _captured(name)[0].model_copy(deep=True)


def rec_of(order, index: int = 0) -> Recommendation:
    return Recommendation.model_validate(order.recommendations[index])


def tamper(record: dict, **changes) -> dict:
    """A copy of a record with nested changes: `tamper(r, **{"trade.stop_price": 1.0, "evidence.signals.0.quote": "x"})`. A path step that is a number indexes a list."""
    out = json.loads(json.dumps(record))
    for path, value in changes.items():
        node = out
        *parents, last = path.split(".")
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return out


def tampered_order(name: str, **changes):
    """A captured order whose first recommendation has been changed."""
    order = captured(name)
    order.recommendations[0] = tamper(order.recommendations[0], **changes)
    return order
