"""The recommendations run, end to end: a request in words, a ranked report out (design choices entry 53; the shell front is `./test_recommendations.sh`).

    python -m prometheus.application.recommendations "Pizza" --find 5 --fetch
    python -m prometheus.application.recommendations DPZ                    # one symbol
    python -m prometheus.application.recommendations "DPZ PZZA YUM"         # a list of symbols (spaces or commas)
    python -m prometheus.application.recommendations "Domino's Pizza"       # a company name (looked up to its ticker)
    python -m prometheus.application.recommendations "weight loss drugs"    # a search phrase: the Data Clerk finds the companies
    python -m prometheus.application.recommendations 'MCD, YUM, "Burger King"'   # a LIST to compare: tickers and names or brands; the Data Clerk finds the listed owner of a brand (QSR)

WHAT IT DOES, in order (each step is one call the LangGraph will make later):

  1. CLASSIFY the request (`classify`): a ticker in capitals, or several, is symbols; a comma separated list that has a ticker and a name or brand ("MCD, YUM, Burger King") is a list, each item looked up;
     a name that is exactly a listed company's name is a company; anything else is a phrase.
  2. FIND the companies (phrase only): `DataClerk.find_companies`. The Data Clerk's model proposes brands and the companies that own them; code looks every name up in Finnhub's directory. A company that
     is privately held (Little Caesars, Donatos) or has no listing is reported as not scored.
  3. REFRESH the data (only with `--fetch`: paid APIs, writes files): `DataClerk.refresh_data` fetches the daily bars, news, press releases and SEC filings of each company (and the bars of the holdings).
  4. ORDER: one work order for all the symbols; `DataClerk.request_data` fills it (the Data Clerk's model answers, then code searches all three corpora and reads each company's filing evidence).
  5. RECOMMEND: `Advisor.generate_recommendations(order, portfolio, case, today, top_k=find)`: a ranked list of the best `find` companies, and a score and rank for every company evaluated.
  6. CHECK RELEVANCE (phrase only): the best similarity between the phrase and each company's stored news and press releases (ChromaDB, the RAG index): is its text about the phrase? Shown, never
     used to rank: the ranking is about trades, the relevance is about the question.
  7. REVIEW (Critic v1, with a `critic`; `--no-critic` leaves it out): the Critic asks the Data Clerk again for the numbers, a quote and the sources, and approves or denies each recommendation against the
     user's profile, the market's trading window and the price (`Critic.review_recommendations`); then the Account Manager decides what may trade (`decide_trades`: Ask or Auto, a denial is final).
  8. REPORT (`report`): the companies found, the data refresh, the ranking best to worst with each company's reason, and for each recommendation its strategy, plan and thesis, then the CRITIC REVIEW.

The case is `industry` for a phrase (leaders ranked best to worst: a large leader is not excluded) and `funds` for symbols and companies (the "here is money" case); `--case` overrides it. Without `--fetch` nothing
is fetched: the stored data is used, and a company with no stored bars fails the Advisor's freshness and liquidity gates with that said.

Every run appends its recommendations to the ledger (data/ledger/recommendations.jsonl), and the ledger's cooldown then hides the same strategy's recommendation for the same company on a later run.
`--no-ledger` is for exploring: a throwaway ledger, so nothing is read from or written to the real one.
"""
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from prometheus.agents.advisor.gates import WAIVED_NOTE
from prometheus.agents.critic.report import review_section
from prometheus.common import work_order as wo
from prometheus.common.market_calendar import eastern_date
from prometheus.common.portfolio import Portfolio, load_portfolio
from prometheus.rag.config import REPO_ROOT
from prometheus.rag.loaders import check_symbol
from prometheus.tools import symbols as symbol_tools

TICKER = re.compile(r"\$?[A-Z0-9][A-Z0-9.\-]{0,9}")
MAX_SYMBOLS = 15
DEFAULT_PORTFOLIO = "configs/portfolios/jacob_pyrett.json"
DEFAULT_PROFILE = str(REPO_ROOT / "configs" / "profiles" / "jacob_pyrett.json")
CASES = ("funds", "theme", "industry")
RELEVANT = 0.40   # the retriever's relevance bar (rag/retriever.MIN_SCORE): a hit under it is "not about this"


@dataclass
class Request:
    kind: str                        # "symbols" | "company" | "phrase" | "list"
    text: str
    symbols: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)   # kind "list": every item as typed, in order (a ticker, a company name or a brand)
    company: str | None = None       # the listed name, for kind "company"
    phrase: str | None = None
    note: str | None = None


@dataclass
class Result:
    request: Request
    case: str
    today: date
    find: int
    companies: list[dict] = field(default_factory=list)      # what was found (phrase), or the symbols as typed
    sector: str | None = None
    refresh: list[dict] = field(default_factory=list)
    order: wo.WorkOrder | None = None
    relevance: dict[str, float | None] = field(default_factory=dict)
    now: datetime | None = None                              # the moment the Critic reviewed at (None: no Critic)
    notes: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- 1. classify

def classify(text: str, kind: str | None = None, search=None) -> Request:
    """What the user asked for. `kind` forces it ("symbols", "company", "phrase": the flags --symbols, --company, --phrase); otherwise: every word a ticker in capitals (DPZ, "DPZ PZZA YUM",
    "DPZ,PZZA") is symbols; text that is exactly a listed company's name (`symbol_tools.find_company`: the same words, generic company words ignored) is a company; anything else, a phrase.
    Raises ValueError for empty input, a bad symbol, or more than 15 symbols."""
    clean = " ".join(text.split()) if isinstance(text, str) else ""
    if not clean:
        raise ValueError("give a symbol, a list of symbols, a company name or a search phrase")
    if len(clean) > 120:
        raise ValueError("the request is longer than 120 characters")
    quoted = any(q in clean for q in '"\u201c\u201d')
    clean = re.sub(r'["\u201c\u201d]', "", clean).strip()
    items = [i.strip() for i in re.split(r"\s*,\s*|\s+and\s+|\s+&\s+", clean, flags=re.I) if i.strip()]
    tickers = [i for i in items if TICKER.fullmatch(i)]
    if kind == "list" or (kind is None and len(items) >= 2 and tickers and len(tickers) < len(items) and ("," in clean or quoted)):
        if len(items) < 2 or len(items) > MAX_SYMBOLS:
            raise ValueError(f"a list has 2 to {MAX_SYMBOLS} items, separated by commas")
        if any(len(i) > 60 for i in items):
            raise ValueError("an item of the list is longer than 60 characters")
        return Request("list", clean, symbols=[check_symbol(t.lstrip("$")) for t in tickers], items=[check_symbol(i.lstrip("$")) if TICKER.fullmatch(i) else i for i in items])
    words = [w for w in re.split(r"[,\s]+", clean) if w]
    if kind == "symbols" or (kind is None and all(TICKER.fullmatch(w) for w in words)):
        picked = list(dict.fromkeys(check_symbol(w.lstrip("$")) for w in words))
        if len(picked) > MAX_SYMBOLS:
            raise ValueError(f"at most {MAX_SYMBOLS} symbols")
        return Request("symbols", clean, symbols=picked)
    if kind == "phrase":
        return Request("phrase", clean, phrase=clean)
    try:
        found = symbol_tools.find_company(clean, search)
    except symbol_tools.TargetNotFound as error:
        if kind == "company":
            raise ValueError(f"{clean!r}: {error.detail or error}") from None
        found = None
    except symbol_tools.SymbolLookupError as error:
        if kind == "company":
            raise ValueError(f"the company lookup failed: {getattr(error, 'reason', error)}") from None
        return Request("phrase", clean, phrase=clean, note=f"the company lookup failed ({getattr(error, 'reason', error)}): treated as a phrase")
    if found:
        return Request("company", clean, symbols=[found.symbol], company=found.company)
    if kind == "company":
        raise ValueError(f"no listed company is named {clean!r}")
    return Request("phrase", clean, phrase=clean)


# ---------------------------------------------------------------- 6. relevance

def phrase_relevance(phrase: str, symbols: list[str], searches=None) -> dict[str, float | None]:
    """For each symbol, the best cosine similarity (0 to 1) between the phrase and its stored news and press releases (ChromaDB, through `rag.retriever`); None when it has no stored text or the index
    cannot be read. `searches` are the search functions (the real two when left out; fakes in tests)."""
    if searches is None:
        from prometheus.rag.retriever import search_news, search_press_releases
        searches = (search_news, search_press_releases)
    out: dict[str, float | None] = {}
    for symbol in symbols:
        best = None
        for search in searches:
            try:
                hits = search(phrase, tickers=[symbol], k=1, min_score=None)
            except Exception:  # an index that cannot be read, a model that cannot load: no number, never a guess
                continue
            if hits:
                best = max(best if best is not None else 0.0, hits[0].score)
        out[symbol] = best
    return out


# ---------------------------------------------------------------- 2 to 5: the run

def _list_companies(request: Request, clerk, today: date, result: Result) -> list[dict]:
    """The companies of a list, in the order typed. A ticker is itself. Any other item (a company name or a brand: "Burger King") goes to `DataClerk.find_companies` for ONE company (its model names the
    owner, code looks the owner up in Finnhub); the tickers that the stored market news adds to a phrase are left out, because an item is one company, not an industry. An item with no listed company
    is kept, not scored, with the reason. The same company twice (MCD and "McDonald's") is kept once."""
    out, seen = [], set()
    for item in request.items:
        if item in request.symbols:
            found = [{"brand": item, "parent": None, "symbol": item, "company": None, "public": True, "how": "symbol as typed", "note": None}]
        else:
            reply = clerk.find_companies(item, count=1, today=today)
            result.notes.extend(reply["notes"])
            found = [c for c in reply["companies"] if c.get("how") != "stored market news"][:1]
            if not found:
                found = [{"brand": item, "parent": None, "symbol": None, "company": None, "public": False, "how": None, "note": "no listed company found for this name: not scored"}]
            if result.sector is None:
                result.sector = reply.get("sector_etf")
        for company in found:
            if company.get("symbol") in seen:
                continue
            seen.add(company.get("symbol"))
            out.append(company)
    return out


def run(request: Request, *, clerk, advisor, portfolio: Portfolio, today: date, find: int = 5, case: str | None = None, sector: str | None = None, fetch: bool = False,
        news_days: int = 14, filings_days: int = 365, relevance_searches=None, log=None, critic=None, manager=None, now: datetime | None = None, approver=None, mode: str | None = None) -> Result:
    """Steps 2 to 7 for one request. `clerk` is the Data Clerk and `advisor` the Advisor (fakes in tests). Nothing is raised for a failed step: it is in `result.notes`, and the run goes as far as it can.

    `critic` (Critic v1): with one, the recommendations are reviewed and the Account Manager decides what may trade; without it the run is exactly what it was. `manager` is the Account Manager that holds
    the user's profile (`AccountManager(model=object())` over the sample profile when left out: its decisions need no model), `now` the aware moment the review is made at (the real clock by default),
    `approver` is asked about each approved recommendation that needs the user (Ask mode) and `mode` overrides the profile's. `today` is the date of the run: when it is not the Eastern date of `now`
    every recommendation is expired (a replay, not a trade), and the run says so."""
    log = log or (lambda text: None)
    case = case or ("industry" if request.kind in ("phrase", "list") else "funds")
    result = Result(request, case, today, find, sector=sector)

    def timed(name, call):
        started = time.monotonic()
        try:
            return call()
        finally:
            result.seconds[name] = round(time.monotonic() - started, 1)

    # 2. the companies
    if request.kind == "phrase":
        log(f"Finding the companies of {request.phrase!r} (the Data Clerk's model proposes, Finnhub's directory confirms)...")
        found = timed("find", lambda: clerk.find_companies(request.phrase, count=max(10, find * 2), today=today))
        result.companies, result.sector = found["companies"], sector or found.get("sector_etf")
        result.notes.extend(found["notes"])
    elif request.kind == "list":
        log(f"Looking up the {len(request.items)} items of the list (the Data Clerk finds the listed owner of a brand)...")
        result.companies = timed("find", lambda: _list_companies(request, clerk, today, result))
    else:
        result.companies = [{"brand": s, "parent": None, "symbol": s, "company": request.company if request.kind == "company" else None, "public": True,
                             "how": "company name" if request.kind == "company" else "symbol as typed", "note": None} for s in request.symbols]
    symbols = [c["symbol"] for c in result.companies if c.get("symbol")]
    if not symbols:
        result.notes.append("no listed company was found to score")
        return result
    held = [p.symbol for p in portfolio.positions]
    # 3. the data
    if fetch:
        log(f"Refreshing the data of {', '.join(symbols)} (paid APIs: daily bars, news, press releases, filings)...")
        result.refresh = timed("refresh", lambda: clerk.refresh_data(symbols, news_days=news_days, filings_days=filings_days, bars_for=held))
    # 4. the work order
    query = (f"What do the stored news, press releases and filings say about the {request.phrase} companies: {', '.join(symbols)}?" if request.kind == "phrase"
             else f"What do the stored news, press releases and filings say about {', '.join(symbols)}?")
    order = wo.new_work_order(query, subject=symbols, since_days=news_days, top_k=10, sector=result.sector, case=case)
    log("The Data Clerk answers the work order (searches the stored news, press releases and filings)...")
    timed("data_clerk", lambda: clerk.request_data(order, today))
    result.order = order
    if order.status != "data_ready":
        result.notes.append(f"the Data Clerk's order is not ready: {'; '.join(order.errors) or order.status}")
        return result
    # 5. the Advisor
    log(f"The Advisor scores and ranks {len(symbols)} companies (case {case}, top {find})...")
    timed("advisor", lambda: advisor.generate_recommendations(order, portfolio, case, today, top_k=find, only_subject=True))
    # 5b. the Critic and the Account Manager's decisions
    if critic is not None and order.status == "recommended" and order.recommendations:
        from prometheus.agents.account_manager.account_manager import AccountManager
        manager = manager or AccountManager(model=object())
        result.now = now or datetime.now(timezone.utc)
        if today != eastern_date(result.now):
            result.notes.append(f"the run date {today} is not the Eastern date of the review ({eastern_date(result.now)}): every recommendation is expired. This is a replay, not a trade")
        log(f"The Critic reviews {len(order.recommendations)} recommendation(s) (it asks the Data Clerk again for the numbers, a quote and the sources)...")
        timed("critic", lambda: critic.review_recommendations(order, manager.risk_view(), portfolio, result.now))
        if order.status == "reviewed":
            log("The Account Manager decides what may trade...")
            timed("decide", lambda: manager.decide_trades(order, portfolio, approver, result.now, mode))
    # 6. relevance
    if request.kind == "phrase":
        result.relevance = timed("relevance", lambda: phrase_relevance(request.phrase, symbols, relevance_searches))
    return result


# ---------------------------------------------------------------- 7. the report

def _pct(value, digits=2) -> str:
    return "n/a" if value is None else f"{value * 100:+.{digits}f}%"


def _money(value) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _cut(text: str, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[:width - 1].rstrip() + "…"


def ranking_rows(result: Result) -> list[dict]:
    """Every NEW idea the Advisor evaluated, best to worst: {rank, symbol, company, score, stance, status, reason, record}. A recommendation is `RECOMMENDED`; the rest say why not. A symbol
    the user holds is not here (it has its own section)."""
    order = result.order
    names = {c["symbol"]: c.get("company") for c in result.companies if c.get("symbol")}
    rows = []
    for r in order.recommendations if order else []:
        if r["ranking"].get("universe_rank") is not None:
            rows.append({"rank": r["ranking"]["universe_rank"], "symbol": r["symbol"], "company": r.get("company") or names.get(r["symbol"]), "score": r["score"], "stance": r["stance"],
                         "status": "RECOMMENDED", "reason": r["action"]["reason"], "record": r, "negative_equity": r.get("negative_equity", False), "blend": r.get("blend")})
    for c in order.candidates if order else []:
        if c.get("rank") is not None and not c.get("held"):
            rows.append({"rank": c["rank"], "symbol": c["symbol"], "company": c.get("company") or names.get(c["symbol"]), "score": c.get("score"), "stance": c.get("stance"),
                         "status": "not recommended", "reason": c["reason"], "record": None, "negative_equity": c.get("negative_equity", False), "blend": c.get("blend")})
    return sorted(rows, key=lambda row: row["rank"])


KEY_NUMBERS = [  # (label, metric, kind): the numbers a comparison of companies looks at, all calculated by code (bars, Finnhub metrics, the filings)
    ("Last close", "last_close", "usd"), ("Market cap", "market_cap", "big"), ("Avg daily $ volume", "adv_20d_dollars", "big"), ("12-1 month momentum", "mom_12_1", "frac"), ("1-month return", "ret_21d", "frac"),
    ("Of 52-week high", "pct_of_52w_high", "frac"), ("RSI 14", "rsi_14", "num1"), ("Volatility (annual)", "vol_ann", "frac"), ("Earnings yield", "earnings_yield", "frac"),
    ("Net payout yield (TTM)", "net_payout_yield_ttm", "frac"), ("Return on equity %", "roe", "num1"), ("Revenue growth %", "revenue_growth", "num1"), ("Debt / equity", "debt_to_equity", "num2"),
    ("Current ratio", "current_ratio", "num2")]


def _number(value, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "frac":
        return f"{value * 100:.1f}%"
    if kind == "usd":
        return f"${value:,.2f}"
    if kind == "big":
        return f"${value / 1e9:,.1f}B" if abs(value) >= 1e9 else f"${value / 1e6:,.0f}M"
    return f"{value:.{1 if kind == 'num1' else 2}f}"


def key_numbers(result: Result, symbols: list[str]) -> list[str]:
    """The table of KEY_NUMBERS, one column per symbol (in the order given); a row is left out when no company has the number. No model, no network."""
    metrics = (result.order.data.metrics if result.order and result.order.data else None) or {}
    columns = [s for s in symbols if metrics.get(s)]
    rows = []
    for label, key, kind in KEY_NUMBERS:
        cells = [("waived" if key == "debt_to_equity" and (metrics[s].get("negative_equity") or {}).get("value") is True else _number((metrics[s].get(key) or {}).get("value"), kind)) for s in columns]
        if any(c != "n/a" for c in cells):
            rows.append(f"  {label:<24}" + "".join(f"{c:>13}" for c in cells))
    equity = [("DEFICIT" if (metrics[s].get("negative_equity") or {}).get("value") is True else "positive" if (metrics[s].get("negative_equity") or {}).get("value") is False else "n/a") for s in columns]
    if rows and any(e != "n/a" for e in equity):
        rows.append(f"  {'Stockholders' + chr(39) + ' equity':<24}" + "".join(f"{e:>13}" for e in equity))
    return [f"  {'':<24}" + "".join(f"{s:>13}" for s in columns), *rows] if rows else []


def why_not(result: Result, symbol: str) -> list[str]:
    """For a company that was not recommended: what each strategy family found, one line each (the first reason a family did not pass, or that it could not be computed and what it lacks)."""
    record = next((c for c in (result.order.candidates if result.order else []) if c["symbol"] == symbol), None)
    out = []
    for f in (record or {}).get("families", []):
        if f["passed"]:
            what = f"PASSED ({f['direction']}, edge {_pct(f['edge_20d'])})"
        elif f["computable"]:
            reasons = [n for n in f["notes"] if n != WAIVED_NOTE]
            what = reasons[0] if reasons else "did not pass"
        else:
            what = "not computable: " + (f["missing"][0] if f["missing"] else "missing input") + (f" ({f['notes'][0]})" if f["notes"] and "no value" in f["notes"][0] else "")
        out.append(f"      {f['family']:<15}: {_cut(what, 110)}" + ("  [debt/equity waived]" if f.get("waived") else ""))
    return out


def _part(blend: dict | None, group: str) -> str:
    """One part of a blend for the ranking table: its value (-1 to +1), or n/a when it had nothing to say (it then counts as 0)."""
    component = ((blend or {}).get("components") or {}).get(group)
    return "n/a" if not component or not component["available"] else f"{component['value']:+.2f}"


def blend_lines(blend: dict | None) -> list[str]:
    """The blend of one recommendation, part by part: the weight, the value, what it adds to the blend, and for the news the items with their age and weight."""
    if not blend:
        return []
    labels = {"news": "news + press releases", "performance": "stock performance", "historical": "historical metrics", "filings": "filings"}
    out = [f"      blend   : {blend['score']} (0-100; {blend['value']:+.2f}) = " + " ".join(
        f"{labels[g]} {c['value']:+.2f} x {c['weight']:.0%}" + ("" if c["available"] else " (n/a: counted as 0)") for g, c in blend["components"].items())]
    items = blend["components"]["news"]["parts"]
    if items:
        shown = sorted(items, key=lambda i: -i["weight"])[:3]
        out.append(f"      news    : {len(items)} item(s), the newer the heavier (half-life {blend['half_life_days']:g} days): " + "; ".join(
            f"{i['ref']} {i['direction']} {'?' if i['age_days'] is None else i['age_days']}d old, weight {i['weight']:.2f}" for i in shown))
    return out


def report(result: Result) -> str:
    """The whole report as text (no model, no network): the request, the companies found, the data refresh, the ranking best to worst, the recommendations with their plans and theses, what was
    not scored, the user's holdings, the model calls and every note and error."""
    out, order, req = [], result.order, result.request
    line = lambda text="": out.append(text)
    bar = "=" * 118
    line(bar)
    what = (f"symbols {', '.join(req.symbols)}" if req.kind == "symbols" else f"company {req.company} ({req.symbols[0]})" if req.kind == "company" else f"comparison of {', '.join(req.items)}" if req.kind == "list" else f"search phrase {req.phrase!r}")
    line(f"PROMETHEUS RECOMMENDATIONS   {what}   |   case: {result.case}   |   top {result.find}   |   as of {result.today}")
    line(bar)
    line()
    line("COMPANIES" + (" FOUND by the Data Clerk (the model proposed them; Finnhub's directory confirmed each ticker)" if req.kind == "phrase" else
                        " (tickers as typed; a name or brand was looked up by the Data Clerk: the model names the owner, Finnhub's directory confirms the ticker)" if req.kind == "list" else ""))
    for c in result.companies:
        shown = f"{c['symbol']:<7} {_cut(c.get('company') or '', 34):<34}" if c.get("symbol") else f"{'-':<7} {'NOT SCORED':<34}"
        extra = c.get("note") or (f"{c['how']}" if c.get("how") else "")
        owner = f" (owned by {c['parent']})" if c.get("parent") and c.get("brand") and c["brand"].lower() not in (c.get("company") or "").lower() else ""
        line(f"  {_cut(str(c.get('brand') or ''), 22):<22} {shown}{owner}  {_cut(extra, 60)}")
    if result.sector:
        line(f"  sector fund for residual returns: {result.sector}")
    if req.note:
        line(f"  note: {req.note}")
    if result.refresh:
        ok = sum(r["outcome"] == "ok" for r in result.refresh)
        line()
        line(f"DATA REFRESH: {ok} of {len(result.refresh)} steps ok" + ("" if ok == len(result.refresh) else " -- failed: " + "; ".join(f"{r['step']} {r['symbol'] or ''}: {_cut(str(r['detail']), 70)}" for r in result.refresh if r["outcome"] != "ok")))
    if order is None or order.status not in ("recommended", "reviewed", "decided"):
        line()
        line("NO RECOMMENDATIONS: " + ("; ".join(result.notes) or "the Advisor did not run") + (f" | order errors: {'; '.join(order.errors)}" if order and order.errors else ""))
        return "\n".join(out)
    regime = next((r["identity"] for r in order.recommendations if r.get("identity")), None)
    rows = ranking_rows(result)
    line()
    line(f"RANKING, best to worst  ({len(rows)} companies scored; ranked by the blended score, the highest first; score 50 is neutral)")
    head = f"  {'#':>2}  {'Symbol':<7} {'Company':<30} {'Score':>5}  {'Stance':<8} {'Blend':>6} {'News':>6} {'Price':>6} {'Hist':>6} {'File':>6}  " + ("Relevance  " if result.relevance else "") + "Result"
    line(head)
    line("  " + "-" * (len(head) - 2))
    for row in rows:
        rel = result.relevance.get(row["symbol"]) if result.relevance else None
        rel_text = (f"{rel:>5.2f} {'on topic' if rel >= RELEVANT else 'weak':<8}" if rel is not None else f"{'none':>5} {'':<8}") + " " if result.relevance else ""
        score = "n/a" if row["score"] is None else str(row["score"])
        line(f"  {row['rank']:>2}  {row['symbol'] + ('*' if row['negative_equity'] else ''):<7} {_cut(row['company'] or '', 30):<30} {score:>5}  {row['stance'] or 'n/a':<8} {(row['blend'] or {}).get('score', 'n/a'):>6} {_part(row['blend'], 'news'):>6} {_part(row['blend'], 'performance'):>6} {_part(row['blend'], 'historical'):>6} {_part(row['blend'], 'filings'):>6}  {rel_text}"
             + (f"RECOMMENDED #{row['record']['ranking']['rank']}" if row["record"] else f"no: {_cut(row['reason'], 58)}"))
    line("  Blend (0-100) = 40% news and press releases (the newer the heavier) + 30% stock performance + 20% historical metrics + 10% filings; the four columns are those parts, -1 to +1 (n/a = nothing to")
    line("  say, counted as 0). The Score is the blend on a 0-100 scale and it ranks the companies; a company is recommended when a strategy passed.")
    flagged = [row["symbol"] for row in rows if row["negative_equity"]]
    if flagged:
        line(f"  * NEGATIVE EQUITY ({', '.join(flagged)}): a stockholders' deficit, so debt/equity has no meaning: the debt/equity rules were WAIVED for them (red-flag gate, reversion and capital-return rules,")
        line("    quality haircut); every other rule applied. Their scores are computed, and ranked, on the other rules alone.")
    unscored = [c for c in result.companies if not c.get("symbol")]
    if unscored:
        line(f"  not scored (no listed company): {', '.join(str(c.get('brand')) for c in unscored)}")
    if regime:
        line(f"  market regime: {regime['regime']}" + (f" ({regime['regime_note']})" if regime.get("regime_note") else ""))
    table = key_numbers(result, [row["symbol"] for row in rows])
    if table:
        line()
        line("KEY NUMBERS (calculated by code from the daily bars, Finnhub's metrics and the filings; n/a = no source gives it)")
        out.extend(table)
    unlisted = [row for row in rows if not row["record"]]
    if unlisted:
        line()
        line("WHY NOT RECOMMENDED (what each strategy found)")
        for row in unlisted:
            line(f"  {row['symbol']:<6} {_cut(row['company'] or '', 40)}   -> {_cut(row['reason'], 100)}")
            out.extend(why_not(result, row["symbol"]))
    picked = [row for row in rows if row["record"]]
    line()
    line(f"RECOMMENDATIONS  ({len(picked)} of the top {result.find} asked for" + ("" if len(picked) >= result.find else f"; the others did not pass: see the ranking above") + ")")
    if not picked:
        line("  none: no strategy passed for any company.")
    for row in picked:
        r = row["record"]
        t, rk = r["trade"], r["ranking"]
        line()
        line(f"  #{rk['rank']}  {r['symbol']}  {row['company'] or ''}   score {r['score']}   action: {r['action']['action']}")
        passed = [f"{f['family']} ({f['direction']}, edge {_pct(f['edge_20d'])})" for f in r["families"] if f["passed"]]
        line(f"      why     : strategies that passed: {', '.join(passed) or 'none'}; opens on {t['family']}; confidence {r['agreement']['confidence']:.2f} from {r['agreement']['evidence_n']} pieces of text evidence")
        if t.get("entry_zone"):
            line(f"      plan    : entry {t['entry_zone'][0]:.2f} to {t['entry_zone'][1]:.2f} ({t['order_style']}" + (f" at {t['limit_price']:.2f}" if t.get("limit_price") else "") + f"), stop {_money(t['stop_price'])}, "
                 f"target {_money(t['target_price']) if t.get('target_price') else 'none (held for the payout)'}, time stop {t['time_stop_date']}")
            line(f"      size    : {t['suggested_shares']} share(s) = {'n/a' if t['position_pct'] is None else format(t['position_pct'] * 100, '.1f') + '%'} of equity {_money(t['equity'])}" + (f" ({t['size_note']})" if t.get("oversized") else "") + f"; funds feasible: {t['funds_feasible']}")
        out.extend(blend_lines(r.get("blend")))
        for e in r["evidence"]["events"][:2]:
            line(f"      event   : {e['class']} ({e['direction']}, {e.get('age_bars')} bars ago): {_cut(e['quote'], 90)}")
        for s in r["evidence"]["signals"][:2]:
            line(f"      evidence: {s['direction']} {s.get('evidence_id') or '[' + str(s['source']) + ']'}: {_cut(s['quote'], 90)}")
        counter = [c["ref"] for c in r["evidence"]["counter_evidence"]]
        if counter:
            line(f"      against : counter-evidence on file: {', '.join(counter[:4])}")
        failed = [g["gate"] for g in r["gates"] if not g["passed"]]
        line(f"      gates   : " + ("all passed" if not failed else "FAILED " + ", ".join(failed)))
        if r.get("negative_equity"):
            line("      equity  : NEGATIVE (stockholders' deficit): the debt/equity rules were waived for this company")
        th = r["thesis"]
        if th.get("thesis"):
            line(f"      thesis  : {th['thesis']}")
            for side, label in (("bull_case", "for"), ("bear_case", "against")):
                for p in th[side][:3]:
                    line(f"        {label:<7} {_cut(p['text'], 100)}  [{p['ref']}]")
        else:
            line(f"      thesis  : not written ({_cut(th.get('thesis_error') or '', 100)})")
    review = review_section(order)
    if review:
        line()
        out.extend(review)
    held = [r for r in order.recommendations if r["action"]["action"] in ("reduce", "add")] + [c for c in order.candidates if c.get("held")]
    if held:
        line()
        line("YOUR HOLDINGS (reviewed before the new ideas)")
        for h in held:
            if "action" in h:
                line(f"  {h['symbol']:<7} {h['action']['action'].upper():<7} {_cut(h['action']['reason'], 90)}")
            else:
                line(f"  {h['symbol']:<7} {'HOLD':<7} {_cut(h['reason'].replace('hold: ', ''), 90)}")
    line()
    runs = "; ".join(f"{r['stage']}: {r['calls']} call(s), {r['tokens_in']} in / {r['tokens_out']} out" + (f", ERROR {r['error']}" if r.get("error") else "") for r in order.advisor_runs)
    line(f"MODEL CALLS (Advisor): {runs or 'none'}")
    for r in order.critic_runs:
        line(f"MODEL CALLS (Critic): {r['model']}: {r['calls']} call(s) for {r['records_called']} recommendation(s), {r['tokens_in']} in / {r['tokens_out']} out" + (f", {r['controls_failed']} reading check(s) FAILED" if r["controls_failed"] else ""))
    line("TIMES (seconds): " + ", ".join(f"{k} {v}" for k, v in result.seconds.items()))
    problems = [*result.notes, *order.errors, *(f"{r['stage']}: dropped {d}" for r in order.advisor_runs for d in r["dropped"][:3])]
    if problems:
        line("NOTES AND ERRORS:")
        for p in problems:
            line(f"  - {_cut(str(p), 150)}")
    line("Nothing here is bought or sold. The numbers are calculated by code; the model only proposed the companies, found the quotes and wrote the thesis, and every quote, number and reference in it was checked."
         + (" The Critic's verdict is arithmetic over what the Data Clerk returned again; its model only labelled quotes." if order.reviews else ""))
    return "\n".join(out)


# ---------------------------------------------------------------- the command

def throwaway_ledger():
    """A ledger in a new temporary folder: empty (no cooldown from earlier runs) and gone with the temporary folder (nothing reaches `data/ledger/`). For exploring a comparison."""
    import tempfile

    from prometheus.memory.ledger import Ledger
    return Ledger(Path(tempfile.mkdtemp(prefix="prometheus_ledger_")) / "recommendations.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", nargs="?", default="", help='a symbol, a list of symbols, a company name or a search phrase, for example "Pizza"')
    parser.add_argument("--symbols", help="force: this is symbols (a list, spaces or commas)")
    parser.add_argument("--company", help="force: this is a company name")
    parser.add_argument("--phrase", help="force: this is a search phrase")
    parser.add_argument("--list", dest="items", help='force: this is a list to compare, separated by commas: tickers, company names or brands, for example "MCD, YUM, Burger King"')
    parser.add_argument("--find", type=int, default=5, help="how many top recommendations to return (1 to 20; default 5)")
    parser.add_argument("--case", choices=CASES, help="funds (the default for symbols and companies), industry (the default for a phrase: leaders ranked), theme (laggards of a positive theme)")
    parser.add_argument("--sector", help="the sector ETF (XLK, XLP, XLY ...); for a phrase the Data Clerk's model suggests one")
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO, help=f"a portfolio file (default {DEFAULT_PORTFOLIO})")
    parser.add_argument("--cash", type=float, help="instead of a portfolio: an empty one with this much cash (the 'here is $1,000' case)")
    parser.add_argument("--from-days-ago", type=int, default=14, help="days of news (1 to 90; default 14)")
    parser.add_argument("--today", help="the run date, YYYY-MM-DD (default: the US Eastern date of the moment of the review: a recommendation is for the trading day it is made on)")
    parser.add_argument("--fetch", action="store_true", help="REFRESH the data with the PAID APIs first (bars, news, press releases, filings: writes files); without it the stored data is used")
    parser.add_argument("--model", help="the Advisor's OpenRouter model id (default: ADVISOR_MODEL of .env)")
    parser.add_argument("--effort", choices=("low", "medium", "high"), help="the Advisor model's reasoning effort (default: low for gpt-oss, the model's own for any other)")
    parser.add_argument("--providers", help="the OpenRouter providers to prefer, comma separated (default for gpt-oss: cerebras,groq; 'none': OpenRouter's own choice)")
    parser.add_argument("--no-ledger", action="store_true", help="explore: use a throwaway ledger, so the real cooldowns (an earlier recommendation of the same strategy hides a company for a while) are not read and nothing is written to data/ledger/")
    parser.add_argument("--save", help="write the finished work order (JSON) to this file")
    critic_group = parser.add_argument_group("the Critic (Critic v1, entries 56 to 59): it reviews each recommendation, then the Account Manager decides what may trade. It costs Finnhub calls and the critic's model")
    critic_group.add_argument("--no-critic", action="store_true", help="leave the Critic out: the run ends at the Advisor's report, as before")
    critic_group.add_argument("--profile", default=str(DEFAULT_PROFILE), help=f"the user's profile (risk tolerance, Ask or Auto, the size limits; default {DEFAULT_PROFILE})")
    critic_group.add_argument("--mode", choices=("ask", "auto"), help="override the profile's trade mode: ask (you approve each trade) or auto (what the Critic approves with a high score is decided 'trade' without asking)")
    critic_group.add_argument("--now", help='review as if it were this US Eastern moment, "YYYY-MM-DD HH:MM" (default: now). A recommendation is only valid in the trading window of the day it was made')
    critic_group.add_argument("--critic-model", help="the critic's OpenRouter model id (default: CRITIC_MODEL of .env, z-ai/glm-5.3)")
    critic_group.add_argument("--critic-effort", choices=("low", "high", "max"), help="the critic's reasoning effort (default: the model's own, max for GLM 5.3: do not lower it to save time)")
    critic_group.add_argument("--no-ask", action="store_true", help="never prompt: an Ask-mode decision stays 'pending' (the default when nobody is at the terminal)")
    args = parser.parse_args(argv)
    given = [(k, v) for k, v in (("symbols", args.symbols), ("company", args.company), ("phrase", args.phrase), ("list", args.items)) if v]
    if len(given) > 1 or (given and args.what):
        parser.error("give the request once: the text, or one of --symbols, --company, --phrase, --list")
    kind, text = (given[0] if given else (None, args.what))
    if not 1 <= args.find <= 20:
        parser.error("--find must be from 1 to 20")
    if not 1 <= args.from_days_ago <= 90:
        parser.error("--from-days-ago must be from 1 to 90")
    try:
        request = classify(text, kind)
        now = datetime.strptime(args.now, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York")) if args.now else datetime.now(timezone.utc)
        today = date.fromisoformat(args.today) if args.today else eastern_date(now)
        portfolio = Portfolio(user="cash", cash_avail=args.cash) if args.cash is not None else load_portfolio(REPO_ROOT / args.portfolio)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.no_critic and (args.mode or args.now or args.critic_model or args.critic_effort or args.no_ask):
        parser.error("the Critic options (--mode, --now, --critic-model, --critic-effort, --no-ask) go with the Critic: drop --no-critic")
    from prometheus.agents.advisor.advisor import Advisor
    from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
    try:
        clerk = DataClerk(allow_fetch=args.fetch)
        ledger = throwaway_ledger() if args.no_ledger else None
        advisor = Advisor(model_id=args.model, clerk=clerk, ledger=ledger, effort=args.effort or "auto",
                          providers="auto" if not args.providers else None if args.providers == "none" else tuple(args.providers.split(",")))
    except RuntimeError as error:  # a missing key: the message names the setting, never its value
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    critic = manager = approver = None
    if not args.no_critic:
        from prometheus.agents.account_manager.account_manager import AccountManager
        from prometheus.agents.account_manager.decisions import terminal_approver
        from prometheus.agents.critic.critic import Critic
        try:
            manager = AccountManager(model=object(), profile=args.profile)   # the profile and the decisions need no model
            _ = manager.profile  # read now: a profile that is not valid stops the run before anything is paid for
            if args.cash is not None:   # "here is $1,000": the cash belongs to the profile's user, or the Account Manager would refuse to decide for a stranger's portfolio
                portfolio = portfolio.model_copy(update={"user": manager.profile.user})
            critic = Critic(clerk=clerk, ledger=throwaway_ledger() if args.no_ledger else None, history=advisor.ledger, model_id=args.critic_model, effort=args.critic_effort)
        except (ValueError, OSError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        approver = terminal_approver(interactive=False if args.no_ask else None)
    started = time.monotonic()
    result = run(request, clerk=clerk, advisor=advisor, portfolio=portfolio, today=today, find=args.find, case=args.case, sector=args.sector, fetch=args.fetch,
                 news_days=args.from_days_ago, log=lambda text: print(f"  .. {text}", flush=True), critic=critic, manager=manager, now=now, approver=approver, mode=args.mode)
    print()
    print(report(result))
    if args.save and result.order is not None:
        Path(args.save).write_text(result.order.model_dump_json(indent=1), encoding="utf-8")
        print(f"(work order saved to {args.save})")
    print(f"\nfinished in {time.monotonic() - started:.0f} s")
    ok = result.order is not None and result.order.status in ("recommended", "reviewed", "decided")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
