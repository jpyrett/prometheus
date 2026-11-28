"""The Data Clerk's tools: what the LLM may ask for, and nothing else (design choices entries 14 and 22).

Each tool is a plain method of `DataClerkTools` with type hints and a docstring written for the model. LangChain
reads the name, the docstring and the typed arguments to build the tool the model sees, and FastMCP can read
the same three things (Part 8). The class is handed its toolbox (the `DataToolbox`), so a test can hand it a fake
one and run every tool with no network, no keys and no cost.

    DATA_CLERK_TOOLS       the allow-list. Only these names are shown to the LLM.
    DataClerkTools         the tools themselves
    .langchain_tools(book) the LangChain tools; `retrieve_news`, `retrieve_press_releases` and `retrieve_filings` number their sources in `book` (a SourceBook);
                           `fetch_filings_recommendation` returns its digest as the text and the whole evidence as the artifact (entry 38)
    describe_tool_error    the message the model gets when a tool fails
    ToolInputError         raised for arguments the model should fix and send again
    ToolLimitError         raised by the agent's budget when a limit on this question's tool use is reached

Code paths that are NOT tools of the model, so not on the allow-list either: the Critic's `read_sources`, `read_evidence` and the like, and `daily_bars` (the stored bars of a date range, for the Account Manager's
backtest through `DataClerk.request_bars`). A model has no use for them, and the 13-tool pin of the allow-list holds.

Never on the allow-list, on purpose: init_aws_cli (installs software), fetch_historical_data (over 100 GB),
convert_to_parquet (a long batch job), export_project (rsync to any destination), and the two scheduled
refresh jobs sync_market_news and sync_cnbc_news.

AS OF A PAST DAY (the Proving Ground, entry 127: `trading_clock.as_of`). While a day is bound every tool here reads only what was known on it: the searches are closed by the retriever itself (`rag/retriever.py`), the stored-text
tools of the toolbox take `until`, the filing evidence stops at the day, the metrics are computed as of it, and `get_quote` answers with the stored close of that day (marked `simulated`). The fetch tools refuse (`trading_clock.NEVER_FETCHES`): a test
of a past day calls no paid API and writes no stored text. `get_financials` is the one thing that is today's snapshot (Finnhub's metrics have no history, Jacob: keep and disclose); its payload says `point_in_time: False`.

A DATED WINDOW (entry 151, design/news_dated_window_spec.md): a news request for a past day or span ("What was the news on TSLA on November 20?") runs on `tools.within(start, end)`, a COPY of these tools whose every search (`_retrieve`: the model's
and the code sweep's) is held to that window by the code below the model, `since` the first instant of the first Eastern day, `until` the last instant of the last one: the model cannot be trusted to pass dates, so its own `since_days` can only narrow
the window (it counts back from the window's end) and never widen it or move it. `fetch_news_window` is the fetch of such a window; it is not a tool of the model.

See what the model sees:   python -m prometheus.tools.data_clerk
"""
import copy
import functools
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool, tool

from prometheus.agents.data_clerk import filing_signals
from prometheus.common import trading_clock
from prometheus.agents.data_clerk.edgar_text import EdgarNotConfigured
from prometheus.rag.citations import SourceBook
from prometheus.rag.config import REPO_ROOT
from prometheus.rag.ingest import ingest_filings, ingest_press_releases, ingest_symbol
from prometheus.rag.loaders import check_symbol, filing_file, load_filings
from prometheus.common.work_order import corpus_of
from prometheus.rag import retriever
from prometheus.rag.retriever import K, MAX_K, MIN_SCORE, Hit, search_filings, search_news, search_press_releases

DATA_CLERK_TOOLS = ("retrieve_news", "retrieve_press_releases", "retrieve_filings", "fetch_news", "fetch_press_releases", "fetch_filings", "fetch_filings_recommendation",
                    "keyword_search", "discover_tickers", "get_financials", "get_price_summary", "get_quote", "calculate_metrics")
EVIDENCE_TOOLS = ("fetch_filings_recommendation",)  # tools that read stored filings and return evidence: the digest is the text the model reads, the full result is the artifact (entry 38)
RETRIEVE_TOOLS = ("retrieve_news", "retrieve_press_releases", "retrieve_filings")  # the tools that search a collection and number their hits in the run's SourceBook

# What differs between the searches, in the words of the messages the model reads when nothing is found.
NEWS_WORDS = dict(item="article", label="news", fetch="fetch_news",
                  not_covered="the stored news does not cover it", should_cover="the stored news should cover this")
PRESS_WORDS = dict(item="press release", label="press releases", fetch="fetch_press_releases",
                   not_covered="the stored press releases do not cover it", should_cover="the stored press releases should cover this")
FILING_WORDS = dict(item="filing", label="filings", fetch="fetch_filings",
                    not_covered="the stored filings do not cover it", should_cover="the stored filings should cover this")

# Limits on what the model may ask for. The paid APIs are paced and slow, and a mistyped 36500 days would be a
# very long download, so the bounds are checked before anything is called.
MAX_NEWS_DAYS = 90
MAX_FILING_DAYS = 730          # press releases and filings
MAX_SEARCH_DAYS = 90
MAX_DISCOVER_DAYS = 30
MAX_DISCOVER_TOP = 20
MAX_SYMBOLS = 10
MAX_TERMS = 10
MAX_TEXT = 60                  # longest company name, keyword or search term
SHOWN_NEWS = 20                # keyword_search shows at most this many articles and filings, and says how many there were
SHOWN_FILINGS = 10
NEAR_MISS = 0.10               # a closest score this near the relevance bar earns the model a hint to reword and retry once
EVIDENCE_PREFIX = "evidence:"  # a signal's `chunk_id` for a filing-evidence item (not a stored chunk: `read_evidence` reads those)
CORPUS_LABELS = {"news": "news", "press_releases": "press releases", "filings": "filings"}   # what a corpus is called in a message


class ToolInputError(ValueError):
    """The model sent arguments that are not allowed. It can fix them and call again."""


class ToolLimitError(Exception):
    """A limit on this question's tool use was reached (too many searches, a fetch that is turned off). Raised by the
    agent's budget, not by a tool; the message tells the model to answer with what it has."""


# ---------------------------------------------------------------- argument checks

def _symbol(value) -> str:
    """One ticker symbol: a leading '$' is dropped, then the symbol is checked like a file name would be."""
    if not isinstance(value, str):
        raise ToolInputError("a ticker symbol must be text, like 'AAPL'")
    try:
        return check_symbol(value.strip().lstrip("$"))
    except ValueError as error:
        raise ToolInputError(str(error)) from None


def _symbols(value) -> list[str]:
    """None, one symbol or a list of symbols -> a list of checked symbols without repeats."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list):
        raise ToolInputError("tickers must be one symbol like 'AAPL' or a list of symbols")
    if len(items) > MAX_SYMBOLS:
        raise ToolInputError(f"at most {MAX_SYMBOLS} tickers at a time")
    return list(dict.fromkeys(_symbol(item) for item in items))


def _days(name: str, value, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ToolInputError(f"{name} must be a whole number of days from 1 to {maximum}")
    return value


def _words(name: str, values, required: bool = False) -> list[str]:
    """A list of short non-empty texts (keywords or search terms)."""
    if values is None:
        values = []
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ToolInputError(f"{name} must be a list of texts")
    values = [v.strip() for v in values if v.strip()]
    if required and not values:
        raise ToolInputError(f"{name} needs at least one word")
    if len(values) > MAX_TERMS or any(len(v) > MAX_TEXT for v in values):
        raise ToolInputError(f"{name}: at most {MAX_TERMS} items of at most {MAX_TEXT} characters")
    return values


def _text(name: str, value, required: bool = False) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ToolInputError(f"{name} is required")
        return None
    if not isinstance(value, str) or len(value.strip()) > MAX_TEXT:
        raise ToolInputError(f"{name} must be text of at most {MAX_TEXT} characters")
    return value.strip()


# ---------------------------------------------------------------- the tools

class DataClerkTools:
    """The Data Clerk's tools. `toolbox` is the DataToolbox (a fake in tests), `store` the Chroma store of the news, `press_store`
    the one of the press releases and `filings_store` the one of the SEC filings (the real ones when left out)."""

    def __init__(self, toolbox=None, store=None, press_store=None, filings_store=None, bars_root=None, guard=None):
        """`guard` (entry 116, design/agentic_security_spec.md): the security layer's door for documents. With one, every document a retrieval tool finds is screened by the code rules (no model call per document) and one that BLOCKs or is held for
        review is dropped before anything reads it; the tool says which in `security_dropped`. None screens nothing (the tests of the tools, the MCP server); the session and the web page pass their guard."""
        if toolbox is None:
            from prometheus.agents.data_clerk.data_toolbox import DataToolbox  # imported here: it needs pyarrow and httpx
            toolbox = DataToolbox(base_dir=REPO_ROOT)  # the same root as rag/: news/ and .env are the ones rag/ reads
        self.toolbox = toolbox
        self._today: date | None = None
        self.store = store
        self.press_store = press_store
        self.filings_store = filings_store
        self.guard = guard
        self._window: tuple[date, date] | None = None      # a DATED window of the news (entry 151): set only on the copy `within` makes; every search is held to it
        self.bars_root = bars_root  # where the day_aggs_v1 files are (a folder holding YYYY/MM/YYYY-MM-DD.csv.gz): the toolbox's own data/historical when left out; a test points it at a temporary folder

    def within(self, start: date, end: date) -> "DataClerkTools":
        """These tools with every search held to the calendar days `start` to `end` (US Eastern, `end` included), a COPY like `DataClerk.fetching` (the same toolbox, stores and guard; the tools it was called on are not changed). The window is applied
        by `_retrieve` to EVERY search, the model's and the sweep's, with `since` the first instant of `start`, `until` and `now` the last instant of `end` (`trading_clock.start_of_day`, `end_of_day`), so a `since_days` the model passes counts back from the window's
        end and the narrower bound wins: it can narrow the window, never widen it and never move it. A past day bound by the Proving Ground still caps `until` and `now`. ValueError for a `start` or `end` that is not a date or a `start` after the `end`."""
        for name, day in (("start", start), ("end", end)):
            if type(day) is not date:
                raise ValueError(f"{name} must be a date (got {type(day).__name__})")
        if start > end:
            raise ValueError(f"the window starts {start} and ends {end}: the first day comes first")
        other = copy.copy(self)
        other._window = (start, end)
        return other

    @property
    def today(self) -> date | None:
        """The date `fetch_filings_recommendation` and the metrics count back from: the one set here (a test sets it; the model never sees it), else the day bound by `trading_clock.as_of`, else None (the callers then use
        the real date, as before). A day that is bound is also the LAST day any of them may read: see `_through`."""
        return self._today if self._today is not None else trading_clock.bound_day()

    @today.setter
    def today(self, value: date | None) -> None:
        self._today = value

    @staticmethod
    def _through(day: date | None) -> date | None:
        """`day` cut to the bound day: while a day is bound nothing may be read after it, whatever date a caller passes. Unbound, or with no `day`, the day as it was."""
        bound = trading_clock.bound_day()
        return day if bound is None or day is None or day <= bound else bound

    @staticmethod
    def _no_fetch(tool_name: str) -> None:
        """Refuse a fetch tool while a day is bound: it would call a paid API and write the stored text (`trading_clock.NEVER_FETCHES`). The model is told to answer with what is stored."""
        day = trading_clock.bound_day()
        if day is not None:
            raise ToolLimitError(f"{trading_clock.NEVER_FETCHES}: {tool_name} is turned off while the desk works as of {day}, so only the text stored on or before that day can be searched")

    # ---- search by meaning (rag/)

    def retrieve_news(self, query: str, tickers: str | list[str] | None = None, since_days: int | None = None,
                      k: int = K, diverse: bool = False, newest_first: bool = False) -> dict:
        """Search the stored news by meaning and return numbered sources to cite.

        Use this for any question about what a company's news says. It only searches news that is already
        stored. It returns the best matching articles, one per article, each with a number: cite them as [1],
        [2] in your answer. If it says there is no supporting text, say so and do not answer from memory. If
        no stored article matches the ticker at all, call fetch_news for that ticker and search again.

        Put the company's name in the question, and set tickers when the question is about one company. Set
        diverse only for a broad question such as "what is happening with Apple?", where you want different
        stories. Leave it off for a question with one specific answer. Set newest_first for a question about
        the latest, newest or most recent news: a search ranks by meaning, not by date, and without it the
        results can be older articles that match the words better.

        Args:
            query: The question or topic, in words. Use the company's name.
            tickers: One ticker symbol such as "AAPL", or a list of symbols. Only articles that name one of them are searched.
            since_days: Only articles from the last N days (a whole number, 1 or more).
            k: How many articles to return, 1 to 20. The default is 5.
            diverse: Spread the results over different stories. For broad questions only.
            newest_first: Of the relevant articles, return the newest first. For questions about the latest news.
        """
        return self._retrieve(search_news, self.store, NEWS_WORDS, query, tickers, since_days, k, diverse=diverse, newest_first=newest_first)

    def retrieve_press_releases(self, query: str, tickers: str | list[str] | None = None, since_days: int | None = None,
                                k: int = K, newest_first: bool = False) -> dict:
        """Search the stored press releases by meaning and return numbered sources to cite.

        Use this for any question about what a company itself announced: earnings results, product launches, executive
        changes, dividends, partnerships. A press release is the company's own statement, not independent reporting, so
        say that it is a press release when you rely on it. It only searches releases that are already stored. It returns
        the best matching releases, one per release, each with a number: cite them as [1], [2] in your answer. If it says
        there is no supporting text, say so and do not answer from memory. If no stored release matches the ticker at all,
        call fetch_press_releases for that ticker and search again.

        Put the company's name in the question, and set tickers when the question is about one company. Set newest_first
        for a question about the latest, newest or most recent announcement: a search ranks by meaning, not by date.

        Args:
            query: The question or topic, in words. Use the company's name.
            tickers: One ticker symbol such as "AAPL", or a list of symbols. Only releases of these companies are searched.
            since_days: Only releases from the last N days (a whole number, 1 or more).
            k: How many releases to return, 1 to 20. The default is 5.
            newest_first: Of the relevant releases, return the newest first. For questions about the latest announcement.
        """
        return self._retrieve(search_press_releases, self.press_store, PRESS_WORDS, query, tickers, since_days, k, newest_first=newest_first)

    def retrieve_filings(self, query: str, tickers: str | list[str] | None = None, since_days: int | None = None,
                         k: int = K, newest_first: bool = False) -> dict:
        """Search the stored SEC filings (10-K annual reports, 8-K current reports, 10-Q, proxy statements, insider sale notices) by meaning and return numbered sources to cite.

        Use this for any question about what a company told the SEC: its business and risk factors (10-K), a material event such
        as results, an acquisition or an executive change (8-K), or whether a form was filed at all. A filing is the company's own
        statement to the regulator, not independent reporting, so say that it is a filing when you rely on it. 10-K and 8-K
        filings are stored WITH text, and so is a 10-Q whose text was fetched from EDGAR. A 10-Q without it, a proxy statement (DEF 14A), a
        Schedule 13G and a Form 144 (an insider plans to sell shares) are stored without text: such a source says only the form, the date and
        that its text is not stored. Say exactly that; never describe what such a filing contains. It only searches filings that are already stored. It returns the
        best matching sections, one per section, each with a number: cite them as [1], [2] in your answer. If it says there is no
        supporting text, say so and do not answer from memory. If no stored filing matches the ticker at all, call fetch_filings
        for that ticker and search again.

        Put the company's name in the question, and set tickers when the question is about one company. Set newest_first for a
        question about the latest or most recent filing: a search ranks by meaning, not by date.

        Args:
            query: The question or topic, in words. Use the company's name.
            tickers: One ticker symbol such as "AAPL", or a list of symbols. Only filings of these companies are searched.
            since_days: Only filings made in the last N days (a whole number, 1 or more).
            k: How many filing sections to return, 1 to 20. The default is 5.
            newest_first: Of the relevant filings, return the newest first. For questions about the latest filing.
        """
        return self._retrieve(search_filings, self.filings_store, FILING_WORDS, query, tickers, since_days, k, newest_first=newest_first)

    def _retrieve(self, search, store, words: dict, query, tickers, since_days, k, **flags) -> dict:
        """The one search-and-explain routine behind both retrieval tools: search `store` with `search`, and when nothing comes back say WHY in words
        the model can act on (no stored text for that ticker, so fetch; or stored text that scores under the bar, so say the sources do not cover it)."""
        symbols = _symbols(tickers)
        options = dict(tickers=symbols or None, since_days=since_days, k=k, store=store)
        if self._window is not None:      # a dated window (entry 151): held to the Eastern days in code, whatever the model passed; `since_days` counts back from the window's end, so it can only narrow it
            first, last = self._window
            options.update(since=trading_clock.start_of_day(first), until=trading_clock.end_of_day(last), now=trading_clock.end_of_day(last))
        try:
            hits = search(query, **flags, **options)
            if hits:
                kept, dropped = self._screened(hits)
                extra = {"security_dropped": dropped} if dropped else {}
                if kept:
                    return {"found": True, "reason": None, "message": None, "hits": [h.as_dict() for h in kept], **extra}
                return {"found": False, "reason": "nothing_relevant", "hits": [], **extra,
                        "message": f"No supporting text: every matching stored {words['item']} was held back by the security screen. Say that {words['not_covered']}. Do not answer from memory."}
            closest = search(query, k=1, min_score=None, **{key: v for key, v in options.items() if key != "k"})
        except ValueError as error:  # every ValueError from the search is a bad argument
            raise ToolInputError(str(error)) from None
        when = f"from {self._window[0]} to {self._window[1]}" if self._window is not None else f"the last {since_days} days" if since_days else ""      # a dated window is named by its days (entry 151)
        where = ", ".join(filter(None, [f"tickers {', '.join(symbols)}" if symbols else "", when])) or "no filters"
        if not closest:
            reason = "no_article_matches_filters"
            message = (f"No stored {words['item']} matches the filters ({where}). " +
                       (f"The {words['label']} for {', '.join(symbols)} may not be fetched yet: call {words['fetch']} for it, then search again."
                        if symbols else f"The {words['label']} index may be empty or the time window too short."))
        else:
            reason = "nothing_relevant"
            message = (f"No supporting text: no stored {words['item']} scored {MIN_SCORE:.2f} or more for this question ({where}); "
                       f"the closest scored {closest[0].score:.2f}. Say that {words['not_covered']}. "
                       f"Do not answer from memory.")
            if closest[0].score >= MIN_SCORE - NEAR_MISS:  # a short question often just scores a little low
                message += (f" That is close to the bar, so if you believe {words['should_cover']}, search "
                            "once more with a clearer question that names the company and the topic.")
        return {"found": False, "reason": reason, "message": message, "hits": []}

    def _screened(self, hits: list[Hit]) -> tuple[list[Hit], list[dict]]:
        """The hits the security layer lets through, and a record of each one it dropped ({id, corpus, action, risk, rules}). With no guard nothing is dropped."""
        if self.guard is None:
            return hits, []
        kept, dropped = [], []
        for hit in hits:
            corpus = corpus_of(hit.chunk_id)
            verdict = self.guard.screen_document(hit.text, doc_id=hit.chunk_id, corpus=corpus)
            if verdict.refused:
                dropped.append({"id": hit.chunk_id, "corpus": corpus, "action": verdict.action, "risk": verdict.risk, "rules": list(verdict.rules)})
            else:
                kept.append(hit)
        return kept, dropped

    # ---- fetch (the toolbox, which uses the paid APIs and writes the text files)

    def fetch_news(self, symbol: str, company: str | None = None, keywords: list[str] | None = None,
                   from_date: int = 7) -> dict:
        """Download the latest news for one ticker and add the new articles to the stored news, so that retrieve_news can find them.

        This is slow (seconds to a minute) and uses paid APIs. Call it when retrieve_news says nothing is
        stored for a ticker, or when the user asks for fresh news. It skips articles that are already stored.
        Leave company and keywords empty to keep every article about the ticker.

        Args:
            symbol: The ticker symbol, such as "AAPL".
            company: Optional company name. Only articles that mention it are kept.
            keywords: Optional list of words. Only articles that mention any of them are kept.
            from_date: How many days back to fetch, 1 to 90. The default is 7.
        """
        self._no_fetch("fetch_news")
        symbol = _symbol(symbol)
        company = _text("company", company)
        keywords = _words("keywords", keywords)
        from_date = _days("from_date", from_date, MAX_NEWS_DAYS)
        stats = self.toolbox.fetch_news(company=company, symbol=symbol, keywords=keywords, from_date=from_date)
        return self._news_fetched(symbol, stats)

    def fetch_news_window(self, symbol: str, start: date, end: date) -> dict:
        """NOT a tool of the model (it is not on `DATA_CLERK_TOOLS`, so the model's tool list is what it was): fetch the news of one ticker for the days `start` to `end` (`DataToolbox.fetch_news(start=, end=)`, the providers' own days, which are UTC
        days) and index what is new, as `fetch_news` does. It is the fetch of a DATED window (entry 151): `DataClerk.refresh_news` asks for the days of the window and the day after, because an evening article of an Eastern day is published the next UTC day;
        the search, not the fetch, decides what is shown (`within`). Refused while a past day is bound, like every fetch. Same result as `fetch_news`."""
        self._no_fetch("fetch_news")
        symbol = _symbol(symbol)
        stats = self.toolbox.fetch_news(symbol=symbol, start=start, end=end)
        return self._news_fetched(symbol, stats)

    def _news_fetched(self, symbol: str, stats: dict) -> dict:
        """What a news fetch answers: the toolbox's counts, and the articles indexed so that `retrieve_news` can find them now."""
        indexed = ingest_symbol(symbol, store=self.store)  # the new articles become searchable now
        return {"symbol": symbol, **{key: stats.get(key) for key in ("fetched", "matched", "written", "duplicates")},
                "indexed": {key: indexed[key] for key in ("articles", "chunks", "added", "updated", "unchanged", "deleted")}}

    def fetch_press_releases(self, symbol: str, from_date: int = 365) -> dict:
        """Download a company's official press releases and add the new ones to the stored press releases, so that retrieve_press_releases can find them.

        This is slow (seconds) and uses a paid API. Call it when retrieve_press_releases says nothing is stored for a ticker, or when
        the work order asks for fresh press releases. It skips releases that are already stored. Fetch before you search, not after.

        Args:
            symbol: The ticker symbol, such as "AAPL".
            from_date: How many days back to fetch, 1 to 730. The default is 365.
        """
        self._no_fetch("fetch_press_releases")
        symbol = _symbol(symbol)
        from_date = _days("from_date", from_date, MAX_FILING_DAYS)
        stats = self.toolbox.fetch_press_releases(symbol, from_date=from_date)
        indexed = ingest_press_releases(symbol, store=self.press_store)  # the new releases become searchable now
        return {"symbol": symbol, **{key: stats.get(key) for key in ("fetched", "written", "duplicates")},
                "indexed": {key: indexed[key] for key in ("articles", "chunks", "added", "updated", "unchanged", "deleted")}}

    def fetch_filings(self, symbol: str, from_date: int = 365) -> dict:
        """Download a company's SEC filings (10-K, 10-Q, 8-K and others) and add the new ones to the stored filings, so that retrieve_filings can find them.

        This is slow (many seconds) and uses a paid API. Call it when retrieve_filings says nothing is stored for a ticker, or when
        the work order asks for fresh filings. It skips filings that are already stored. Fetch before you search, not after.
        The paid API gives no text for a 10-Q, so this also fetches the text of the stored 10-Q filings from SEC EDGAR when it can. The result
        says what happened in edgar_text: how many 10-Qs it wrote the text of, or "skipped" with the reason (for example SEC_USER_AGENT is not
        set). If it says skipped, the 10-Qs stay without text: tell the user that, and do not call this again for it.

        Args:
            symbol: The ticker symbol, such as "AAPL".
            from_date: How many days back to fetch, 1 to 730. The default is 365.
        """
        self._no_fetch("fetch_filings")
        symbol = _symbol(symbol)
        from_date = _days("from_date", from_date, MAX_FILING_DAYS)
        stats = self.toolbox.fetch_filings(symbol, from_date=from_date)
        try:  # the 10-Q's text comes from EDGAR (design choices entry 36). Not being set up for EDGAR is reported, and the rest still happens; any other failure stops here
            text = self.toolbox.fetch_filing_text(symbol, from_date=from_date)
            edgar = {key: text.get(key) for key in ("candidates", "written")}
        except EdgarNotConfigured:  # a fixed reason: the error's own message holds the path of the .env file, and a tool result goes into the model's prompt
            edgar = {"skipped": "SEC_USER_AGENT is not set (EDGAR needs a User-Agent that names you and gives a contact), so the text of the 10-Qs was not fetched from EDGAR"}
        indexed = ingest_filings(symbol, store=self.filings_store)  # the new filings, and the new 10-Q text, become searchable now
        return {"symbol": symbol, **{key: stats.get(key) for key in ("fetched", "written", "duplicates")}, "edgar_text": edgar,
                "indexed": {key: indexed[key] for key in ("articles", "chunks", "added", "updated", "unchanged", "deleted")}}

    def fetch_filings_recommendation(self, symbol: str, since_days: int = filing_signals.DEFAULT_WINDOW) -> dict:
        """Get the evidence in a company's STORED filings that a recommendation needs: the latest balance sheet, the buybacks, dividends and share issuance, and the passages about stock dilution and executive compensation.

        It only reads filings that are already stored (it costs nothing and fetches nothing: call fetch_filings first when none are stored). It returns evidence and does NOT
        rate anything: the latest balance sheet for both periods with the change (cash, debt, equity, shares issued and outstanding), the capital actions of the cash-flow
        statement (repurchases, dividends paid, share-based compensation, debt raised), the headline income figures, and the passages about stock dilution, executive compensation
        and employee equity compensation (negative-type topics) and about buybacks and dividends (positive-type topics), each as an exact quote with its filing, section and date,
        with how many were found and how many are shown. It also lists what could not be read (for example that no stored 10-Q has its financial statements yet). Report what
        it says, with the [E] numbers, and the gaps it lists. Do not call a stock good or bad and do not give a score: the Advisor does that.

        Args:
            symbol: The ticker symbol, such as "AAPL".
            since_days: Only filings filed in the last N days, 1 to 730. The default is 365.
        """
        symbol = _symbol(symbol)
        since_days = _days("since_days", since_days, MAX_FILING_DAYS)
        try:
            docs = load_filings(filing_file(symbol))
        except FileNotFoundError:
            raise ToolInputError(f"no filings of {symbol} are stored: run fetch_filings for it first") from None
        return filing_signals.recommendation_evidence(symbol, filing_signals.filings_from_documents(docs), today=self.today, since_days=since_days)   # a bound day also closes the window at the day (`until` defaults to it)

    # ---- keyword search over the stored text, and the other data

    @staticmethod
    def _stored_until() -> dict:
        """The `until` the toolbox's stored-text searches take, only while a day is bound (the toolbox defaults to it too: this makes the door visible here, and an unbound call passes exactly what it always passed)."""
        day = trading_clock.bound_day()
        return {} if day is None else {"until": day}

    def keyword_search(self, symbol: str, terms: list[str], from_date: int = 30) -> dict:
        """Find stored news and filings for one ticker that contain any of the exact words you give. This matches words, not meaning: for a question, use retrieve_news.

        It searches only what is already stored and returns a list of headlines with their dates and links.

        Args:
            symbol: The ticker symbol, such as "AAPL".
            terms: One or more words or phrases that must appear, such as ["Zuckerberg", "billionaire tax"].
            from_date: How many days back to search, 1 to 90. The default is 30.
        """
        symbol = _symbol(symbol)
        terms = _words("terms", terms, required=True)
        from_date = _days("from_date", from_date, MAX_SEARCH_DAYS)
        found = self.toolbox.search(symbol, terms, from_date=from_date, refresh=False, **self._stored_until())
        news, filings = found.get("news", []), found.get("filings", [])
        return {"symbol": symbol, "terms": terms, "from_date": from_date,
                "news_total": len(news), "news": news[:SHOWN_NEWS],
                "filings_total": len(filings), "filings": filings[:SHOWN_FILINGS],
                "filings_in_window": found.get("filings_in_window")}

    def discover_tickers(self, phrase: str, from_date: int = 7, top: int = 10) -> dict:
        """Find which companies a trend or topic is about, such as "GLP-1" or "weight loss", when you do not know the tickers.

        It counts the tickers named in stored market-wide news that contains the exact phrase, and ranks them.
        It does not fetch anything new.

        Args:
            phrase: The topic to look for, in words.
            from_date: How many days back to look, 1 to 30. The default is 7.
            top: How many tickers to return, 1 to 20. The default is 10.
        """
        phrase = _text("phrase", phrase, required=True)
        from_date = _days("from_date", from_date, MAX_DISCOVER_DAYS)
        top = _days("top", top, MAX_DISCOVER_TOP)
        return self.toolbox.discover(phrase, from_date=from_date, top=top, refresh=False, **self._stored_until())

    def get_financials(self, symbol: str) -> dict:
        """Get ten common financial metrics for one ticker (P/E, EPS, margins, return on equity, debt to equity and others) from Finnhub.

        Args:
            symbol: The ticker symbol, such as "AAPL".
        """
        return self.toolbox.financials.get(_symbol(symbol))

    def get_price_summary(self, symbol: str) -> dict:
        """Summarize how one ticker has traded this month: date range, volume, number of trades, lowest, average and highest price, and the spread of the daily closes.

        Args:
            symbol: The ticker symbol, such as "AAPL".
        """
        summary = dict(self.toolbox.performance_summary(_symbol(symbol)))
        if isinstance(summary.get("distribution"), dict):  # the 61-point bell curve is for a chart, not for a model
            summary["distribution"] = {k: v for k, v in summary["distribution"].items() if k != "curve"}
        return summary

    def get_quote(self, symbol: str) -> dict:
        """Get the current price of one ticker from Finnhub: the last price, the change since the previous close, today's high, low and open, and when it last traded.

        Outside market hours the time is the last trade of the session, not now. Use this for "what is it trading at"; use get_price_summary for how it traded this month.

        Args:
            symbol: The ticker symbol, such as "AAPL".
        """
        symbol = _symbol(symbol)
        day = trading_clock.bound_day()
        if day is not None:   # a past day has no live price: the stored close of that day, marked `simulated` (the same answer `DataClerk.request_quotes` gives)
            return self.close_quote(symbol, day)
        return self.toolbox.get_quote(symbol)

    def calculate_metrics(self, symbols: list[str], metrics: list[str]) -> dict:
        """Calculate technical and fundamental numbers for up to 10 tickers: returns over 5 to 252 sessions, RSI, MACD, Bollinger %B, ADX, ATR, volatility, beta, volume ratios, P/E, margins, debt, share count change and more.

        It only reads the stored daily bars and the stored filings and asks Finnhub for the company's financial metrics, and it rates NOTHING: it returns numbers, each with its unit, the date it is
        as of, the window it covers, where it came from, the inputs behind it and, when it cannot be computed, why (for example "needs 30 bars, 17 stored"). A number that is None is a
        gap: say so, never estimate it. An unknown metric name is refused with the list of known names.

        Args:
            symbols: One to 10 ticker symbols, such as ["AAPL", "PG"].
            metrics: The names of the metrics wanted, such as ["rsi_14", "ret_21d", "roe"].
        """
        from prometheus.agents.data_clerk.metrics_catalog import METRIC_NAMES  # imported here: the catalogue needs pyarrow and httpx
        names = _words("metrics", metrics, required=True)
        unknown = [n for n in names if n not in METRIC_NAMES]
        if unknown:
            raise ToolInputError(f"unknown metrics: {', '.join(unknown)}; known metrics: {', '.join(METRIC_NAMES)}")
        picked = _symbols(symbols)
        if not picked:
            raise ToolInputError("symbols needs at least one ticker")
        return self.metrics_for(picked, [{"name": n} for n in dict.fromkeys(names)], self.today)

    def metrics_for(self, symbols: list[str], requests: list[dict], today: date | None = None, sectors: dict[str, str] | None = None) -> dict:
        """The code path behind `calculate_metrics` (and `DataClerk.request_metrics`, Advisor v1 spec section 4): read the bars ONCE for every symbol and for SPY and the sector ETFs, ask Finnhub
        for each symbol's financial metrics (only when a requested metric needs them), read each symbol's STORED filing evidence (free, never fetched), and compute. Returns
        {as_of, results: {symbol: {metric: result}}, problems: {symbol: reason}, gaps: [...bars store gaps]}. A problem is something that could not be read (a Finnhub call that failed, no stored filings): the
        metrics that needed it say so in their own `note`, and everything else is still computed. No symbol check here: the callers check them."""
        from prometheus.agents.data_clerk import metrics_catalog as catalog
        from prometheus.agents.data_clerk.bars import BarStore
        today = self._through(today or self.today or date.today())   # while a day is bound, never later than it: the bars and the evidence are read as of that day
        root = self.bars_root or getattr(self.toolbox, "root_data_dir", REPO_ROOT / "data" / "historical") / "day_aggs_v1"
        store = BarStore(root).load([*symbols, catalog.SPY, *catalog.sector_etfs()], end=today)
        needs = {kind for r in requests for kind in catalog.METRICS[r["name"]].inputs}
        financials, evidence, problems = {}, {}, {}
        for symbol in symbols:
            if "financials" in needs:
                try:
                    financials[symbol] = self.toolbox.financials.get(symbol)
                except Exception as error:  # a bad ticker, no key, an API failure: say which kind, never request details
                    problems[symbol] = f"Finnhub metrics failed ({type(error).__name__}: {' '.join(str(error).split())[:120]})"
            if "evidence" in needs:
                try:
                    evidence[symbol] = self.fetch_filings_recommendation(symbol)
                except ToolInputError:
                    problems.setdefault(symbol, "no filings are stored for it")
                except Exception as error:
                    problems.setdefault(symbol, f"reading the stored filings failed ({type(error).__name__})")
        tables = {}
        if "splits" in needs:     # the stored split tables (read only, `data/historical/splits_v1/`), by the Proving Ground's own reader: a table is only ever read for what the metric needs (`close_history`), and it adjusts nothing after `today`
            from prometheus.agents.data_clerk.data_toolbox import SPLITS_DIR
            from prometheus.proving import splits as splits_mod
            for symbol in symbols:
                try:
                    table, _ = splits_mod.read(symbol, Path(root).parent / SPLITS_DIR)
                except ValueError:      # not a ticker: no table (the callers check the symbols)
                    table = None
                if table is not None:
                    tables[symbol] = table
        results = catalog.compute(symbols, requests, store, financials, evidence, today, sectors, tables)
        return {"as_of": today.isoformat(), "results": results, "problems": problems, "gaps": store.gaps[:20],
                "bars": {s: len(store.bars(s, end=today)) for s in symbols}}

    def daily_bars(self, symbols: list[str], start: date, end: date) -> dict:
        """The stored daily bars of each symbol from `start` to `end`, both included, oldest first: {"bars": {symbol: [{"date": "YYYY-MM-DD", "open", "high", "low", "close", "volume"}, ...]}, "gaps": [...]}.
        The code path behind `DataClerk.request_bars` (Trader v1 spec section 12, the backtest's prices): NOT a tool of the model, so not in DATA_CLERK_TOOLS (a model has no use for it, and the 13-tool pin holds).
        It reads the same `day_aggs_v1` files as `metrics_for` (the same root rule: `bars_root`, else the toolbox's own `data/historical`) and only as many day files as the range holds. Every symbol is a key; one with
        no stored bar in the range is an empty list (the future may not be stored yet: a result, not an error). `gaps` says what the store could not read or is missing inside the range. Nothing is fetched and
        nothing is written. Raises ToolInputError for symbols or dates that make no sense, FileNotFoundError when the bar folder does not exist at all (a store that cannot be read is not "no bars")."""
        if not isinstance(symbols, list) or not symbols:
            raise ToolInputError("symbols needs at least one ticker")
        picked = list(dict.fromkeys(_symbol(s) for s in symbols))
        if type(start) is not date or type(end) is not date:
            raise ToolInputError("start and end must be dates")
        if start > end:
            raise ToolInputError("start must be on or before end")
        from prometheus.agents.data_clerk.bars import BarStore
        root = self.bars_root or getattr(self.toolbox, "root_data_dir", REPO_ROOT / "data" / "historical") / "day_aggs_v1"
        if not Path(root).is_dir():
            raise FileNotFoundError(f"the bar folder {root} does not exist")
        store = BarStore(root).load(picked, end=end, start=start, max_bars=(end - start).days + 1)  # at most one file a day, so the range is never cut short
        return {"bars": {s: [{"date": b.date.isoformat(), "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
                            for b in store.bars(s, end=end, max_bars=(end - start).days + 1) if b.date >= start] for s in picked},
                "gaps": store.gaps[:20]}

    def analyst_ratings(self, symbols: list[str], store: str, today: date | None = None) -> dict:
        """The code path behind `DataClerk.request_analyst_ratings` (Critic v3, design choices entry 137): NOT a tool of the model, so not in DATA_CLERK_TOOLS. For each symbol: ask Finnhub for its analyst recommendation trends
        and keep the answer in the store (`DataToolbox.fetch_analyst_ratings`: one paid call each, never while a day is bound), then read the STORED periods the day could know (`read_analyst_ratings`: only periods on or before
        `today`, cut to the bound day), newest first. A symbol whose call failed (no key, HTTP error) is a problem in words and is still answered with what is stored. Returns {"ratings": {symbol: {"periods": [...], "fetched":
        bool, "note": str | None}}, "problems": [...]}: a symbol with nothing stored has an empty list and a note (a result, not an error)."""
        picked = list(dict.fromkeys(_symbol(s) for s in symbols))
        until = self._through(today)
        bound = trading_clock.bound_day() is not None
        ratings, problems = {}, []
        for symbol in picked:
            fetched, note = False, None
            if not bound:
                try:
                    self.toolbox.fetch_analyst_ratings([symbol], store)
                    fetched = True
                except Exception as error:  # a missing key or an API failure: the kind and a short message, never request details; the stored periods are still read
                    problems.append(f"{symbol}: the analyst ratings could not be fetched ({type(error).__name__}: {' '.join(str(error).split())[:160]})")
                    note = "Finnhub could not be asked; the stored counts are used"
            periods = self.toolbox.read_analyst_ratings(symbol, store, until)
            if not periods:
                note = "no stored analyst ratings the day could know" if note is None else note + ", and there are none stored"
            ratings[symbol] = {"periods": periods, "fetched": fetched, "note": note}
        return {"ratings": ratings, "problems": problems}

    def close_quote(self, symbol: str, day: date) -> dict:
        """A quote made from the STORED daily bar of `day`, for the simulated clock (`USE_LIVE_TRADING=false`, `common/trading_clock.py`, design choices entry 76): the market close of that trading day instead of
        Finnhub's live price. Same shape as `get_quote` ({symbol, price, change, change_percent, high, low, open, previous_close, as_of_utc, source}) plus `simulated: True`, so that a reader can tell it from a live
        quote (the Critic's second-provider price check skips it: the same bar would be compared with itself). `price` is the day's close; `as_of_utc` is a minute before the simulated moment, so the quote is fresh at it.
        Not a tool of the model (not in DATA_CLERK_TOOLS). Reads the same `day_aggs_v1` files as `daily_bars`; calls nothing and writes nothing. Raises ToolInputError for a symbol or a day that makes no sense or when
        the symbol has no stored bar ON that day (the nearest earlier bar is NOT used: that would be another day's price), FileNotFoundError when the bar folder does not exist."""
        from prometheus.agents.data_clerk.bars import BarStore
        from prometheus.common.market_calendar import simulated_moment
        if type(day) is not date:
            raise ToolInputError("day must be a date")
        bound = trading_clock.bound_day()
        if bound is not None and day > bound:   # the close of a later day is the future of the desk's day
            raise ToolInputError(f"no quote for {day}: the desk works as of {bound}, and a later close is not known yet")
        symbol = _symbol(symbol)
        root = self.bars_root or getattr(self.toolbox, "root_data_dir", REPO_ROOT / "data" / "historical") / "day_aggs_v1"
        if not Path(root).is_dir():
            raise FileNotFoundError(f"the bar folder {root} does not exist")
        series = BarStore(root).load([symbol], end=day, max_bars=10).bars(symbol, end=day, max_bars=2)   # the day and the one before it: a few day files are enough
        if not series or series[-1].date != day:
            raise ToolInputError(f"no stored daily bar for {symbol} on {day}")
        bar, before = series[-1], (series[-2] if len(series) > 1 else None)
        previous = before.close if before else bar.close
        moment = (simulated_moment(day) - timedelta(minutes=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
        return {"symbol": symbol, "price": round(float(bar.close), 4), "change": round(float(bar.close - previous), 4), "change_percent": round(float((bar.close / previous - 1) * 100), 4) if previous else 0.0,
                "high": round(float(bar.high), 4), "low": round(float(bar.low), 4), "open": round(float(bar.open), 4), "previous_close": round(float(previous), 4), "as_of_utc": moment,
                "source": f"daily bar of {day} (USE_LIVE_TRADING=false: the market close of the last trading day that has closed)", "simulated": True}

    # ---- the Critic's code path (Critic v1 spec 5.2): NOT tools of the model, so not in DATA_CLERK_TOOLS (a model has no use for them)

    def read_sources(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Read stored chunks again by their ids: {chunk_id: {found, corpus, article_id, headline, publisher, published_utc, url, tickers, text} | {found: False, note}}. Each id goes to the collection its
        prefix names (`corpus_of`: `pr:` press releases, `sec:` filings, anything else news). The Critic uses it to prove that a source the Advisor quoted exists in the store today and says what it said.
        An id that is not in the store, an `evidence:` id (not a chunk) and a collection that cannot be read are each `found: False` with the reason: never an exception, never a guess."""
        wanted = list(dict.fromkeys(chunk_ids))
        out: dict[str, dict] = {}
        by_corpus: dict[str, list[str]] = {}
        for chunk_id in wanted:
            if chunk_id.startswith(EVIDENCE_PREFIX):
                out[chunk_id] = {"found": False, "note": "a filing-evidence item is not a chunk: ask for it by its evidence id"}
            else:
                by_corpus.setdefault(corpus_of(chunk_id), []).append(chunk_id)
        stores = {"news": self.store, "press_releases": self.press_store, "filings": self.filings_store}
        for corpus, ids in by_corpus.items():
            try:
                got = (stores[corpus] or retriever.get_vector_store(corpus)).get(ids=ids, include=["documents", "metadatas"])
            except Exception as error:  # a store that cannot be read: the kind, never the details
                out.update({i: {"found": False, "note": f"the store could not be read ({type(error).__name__})"} for i in ids})
                continue
            stored = {i: (doc, meta or {}) for i, doc, meta in zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", []))}
            cutoff = self._cutoff_ts(corpus)
            for chunk_id in ids:
                if chunk_id not in stored:
                    out[chunk_id] = {"found": False, "note": f"not in the stored {CORPUS_LABELS[corpus]}"}
                    continue
                doc, meta = stored[chunk_id]
                if cutoff is not None and not (isinstance(meta.get("published_ts"), (int, float)) and meta["published_ts"] <= cutoff):   # a day is bound: a chunk dated after it is not read (and one with no date cannot be placed)
                    out[chunk_id] = {"found": False, "note": f"not in the stored {CORPUS_LABELS[corpus]} as of {trading_clock.bound_day()}"}
                    continue
                out[chunk_id] = {"found": True, "corpus": corpus, "article_id": meta.get("article_id"), "headline": meta.get("headline"), "publisher": meta.get("publisher"),
                                 "published_utc": meta.get("published_utc"), "url": meta.get("url"), "tickers": list(meta.get("tickers") or []), "text": doc}
        return {chunk_id: out[chunk_id] for chunk_id in wanted}

    @staticmethod
    def _cutoff_ts(corpus: str) -> int | None:
        """While a day is bound: the last `published_ts` a chunk of this collection may have (the end of that Eastern day; for the filings, which have a date and no time, 23:59:59 UTC of the day); None when nothing is bound."""
        day, horizon = trading_clock.bound_day(), trading_clock.horizon()
        if day is None:
            return None
        return int(datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc).timestamp()) if corpus == "filings" else int(horizon.timestamp())

    def read_evidence(self, symbol: str, ids: list[str]) -> dict[str, dict]:
        """The filing-evidence items of one symbol with these ids (E3 ...), read again from the STORED filings (free, nothing fetched): {id: {found, topic, filter, quote, what, derived, source: {accession,
        form, filed}} | {found: False, note}}. With no stored filings every id is `found: False` with the tool's own message."""
        symbol = _symbol(symbol)
        try:
            items = {item["id"]: item for item in self.fetch_filings_recommendation(symbol).get("evidence", []) if item.get("id")}
        except ToolInputError as error:
            return {i: {"found": False, "note": str(error)} for i in ids}
        out = {}
        for evidence_id in dict.fromkeys(ids):
            item = items.get(evidence_id)
            out[evidence_id] = ({"found": False, "note": f"{symbol} has no evidence item {evidence_id} in its stored filings"} if item is None else
                                {"found": True, "topic": item.get("topic"), "filter": item.get("filter"), "quote": item.get("quote"), "what": item.get("what"), "derived": bool(item.get("derived")),
                                 "source": {k: (item.get("source") or {}).get(k) for k in ("accession", "form", "filed")}})
        return out

    def recent_sources(self, symbol: str, company: str | None = None, since_days: int = 14, k: int = 8, now=None) -> list[dict]:
        """The newest stored news and press releases of one symbol, newest first, at most `k`: the Hit fields plus `corpus`. No relevance bar (the Critic wants what is NEWER than the Advisor's evidence, on
        topic or not: a person decides what matters). Stored text only: nothing is fetched."""
        symbol = _symbol(symbol)
        since_days = _days("since_days", since_days, MAX_DISCOVER_DAYS)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
            raise ToolInputError(f"k must be a whole number from 1 to {MAX_K}")
        query = f"{company or symbol}: latest news, results, guidance, risks"
        options = dict(tickers=[symbol], since_days=since_days, k=k, min_score=None, newest_first=True, now=now)
        found = [(h, "news") for h in search_news(query, store=self.store, **options)] + [(h, "press_releases") for h in search_press_releases(query, store=self.press_store, **options)]
        found.sort(key=lambda pair: pair[0].published_utc, reverse=True)
        if self.guard is not None:      # the Critic's fresh sources are evidence too: a document the security layer drops is not shown to it (it is logged, entry 116)
            passed = {h.chunk_id for h in self._screened([h for h, _ in found])[0]}
            found = [pair for pair in found if pair[0].chunk_id in passed]
        return [{**hit.as_dict(), "corpus": corpus} for hit, corpus in found[:k]]

    # ---- the LangChain edge

    def langchain_tools(self, book: SourceBook | None = None) -> list[BaseTool]:
        """The allow-listed tools as LangChain tools. The three retrieval tools return the numbered sources as their text and
        the full hits (with their numbers) as the artifact. They share ONE `book`, one book per run, so a [3] means the same
        chunk whichever of them found it, and a news article, a press release and a filing can never share a number."""
        book = book if book is not None else SourceBook()
        tools = []
        for name in DATA_CLERK_TOOLS:
            method = getattr(self, name)
            if name in RETRIEVE_TOOLS:
                tools.append(self._retrieve_tool(method, book))
            elif name in EVIDENCE_TOOLS:
                tools.append(self._evidence_tool(method))
            else:
                tools.append(tool(method, parse_docstring=True))
        return tools

    @staticmethod
    def _evidence_tool(method) -> StructuredTool:
        """The model reads the digest (compact, fixed wording); the whole result (the figures, every quote with its source, the gaps) goes along as the artifact for the code that
        wants it. The same shape as the retrieve tools, with no source numbers: the evidence carries its own [E] numbers."""
        @functools.wraps(method)  # same name, docstring and arguments as the method
        def evidence(*args, **kwargs):
            result = method(*args, **kwargs)
            return result["digest"], result

        return StructuredTool.from_function(evidence, name=method.__name__, parse_docstring=True, response_format="content_and_artifact")

    @staticmethod
    def _retrieve_tool(method, book: SourceBook) -> StructuredTool:
        @functools.wraps(method)  # same name, docstring and arguments as the method
        def retrieve(*args, **kwargs):
            result = method(*args, **kwargs)
            hits = [Hit(**h) for h in result["hits"]]
            extra = {"security_dropped": result["security_dropped"]} if result.get("security_dropped") else {}      # the documents the security layer dropped (entry 116), for the Data Clerk's result
            if not hits:
                return result["message"], {"found": False, "reason": result["reason"], "hits": [], **extra}
            text = book.add(hits)
            return text, {"found": True, "reason": None,
                          "hits": [{"number": book.number_of(h), **h.as_dict()} for h in hits], **extra}

        return StructuredTool.from_function(retrieve, name=method.__name__, parse_docstring=True,
                                            response_format="content_and_artifact")


def describe_tool_error(error: Exception) -> str:
    """The message the model gets when a tool raises (the `handle_tool_errors` of the ToolNode, entry 14).

    The toolbox raises on every error, and without this the whole run would stop. The message is short and
    factual, and it says what to do next, so that the Data Clerk can report the failure or try something else."""
    text = " ".join(str(error).split())[:300]
    if isinstance(error, ToolLimitError):
        return f"Limit reached: {text}. Do not ask for this tool again. Answer now with the sources you already have, or say what they do not cover."
    if isinstance(error, ToolInputError):
        return f"Invalid arguments: {text}. Fix the arguments and call the tool again."
    return (f"The tool failed ({type(error).__name__}): {text}. Do not repeat the same call. "
            f"Tell the user what failed, or use a different tool.")


def main() -> int:
    """Print the tools the way the model sees them: name, description and typed arguments."""
    tools = DataClerkTools().langchain_tools()
    print(f"{len(tools)} tools are shown to the model (DATA_CLERK_TOOLS):\n")
    for t in tools:
        schema = t.get_input_schema().model_json_schema()
        required = set(schema.get("required", []))
        print(f"{t.name}\n  {t.description.replace(chr(10), chr(10) + '  ')}")
        for arg, spec in t.args.items():
            kind = spec.get("type") or " or ".join(x.get("type", "?") for x in spec.get("anyOf", []))
            print(f"    {arg}: {kind}{'' if arg in required else ' (optional)'}  {spec.get('description', '')}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
