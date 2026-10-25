"""The Data Clerk's tools: what the LLM may ask for, and nothing else (design choices entries 14 and 22).

Each tool is a plain method of `DataClerkTools` with type hints and a docstring written for the model. LangChain
reads the name, the docstring and the typed arguments to build the tool the model sees, and FastMCP can read
the same three things (Part 8). The class is handed its toolbox (the `DataAgent`), so a test can hand it a fake
one and run every tool with no network, no keys and no cost.

    DATA_CLERK_TOOLS       the allow-list. Only these names are shown to the LLM.
    DataClerkTools         the tools themselves
    .langchain_tools(book) the LangChain tools; `retrieve_news` numbers its sources in `book` (a SourceBook)
    describe_tool_error    the message the model gets when a tool fails
    ToolInputError         raised for arguments the model should fix and send again
    ToolLimitError         raised by the agent's budget when a limit on this question's tool use is reached

Never on the allow-list, on purpose: init_aws_cli (installs software), fetch_historical_data (over 100 GB),
convert_to_parquet (a long batch job), export_project (rsync to any destination), and the two scheduled
refresh jobs sync_market_news and sync_cnbc_news.

See what the model sees:   python -m prometheus.tools.data_clerk
"""
import functools
import sys

from langchain_core.tools import BaseTool, StructuredTool, tool

from prometheus.rag.citations import SourceBook
from prometheus.rag.config import REPO_ROOT
from prometheus.rag.ingest import ingest_symbol
from prometheus.rag.loaders import check_symbol
from prometheus.rag.retriever import K, MAX_K, MIN_SCORE, Hit, search_news

DATA_CLERK_TOOLS = ("retrieve_news", "fetch_news", "fetch_press_releases", "fetch_filings",
                    "keyword_search", "discover_tickers", "get_financials", "get_price_summary")

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
    """The Data Clerk's tools. `toolbox` is the DataAgent (a fake in tests), `store` the Chroma store (the real
    one when left out)."""

    def __init__(self, toolbox=None, store=None):
        if toolbox is None:
            from prometheus.agents.data_clerk.data_agent import DataAgent  # imported here: it needs pyarrow and httpx
            toolbox = DataAgent(base_dir=REPO_ROOT)  # the same root as rag/: news/ and .env are the ones rag/ reads
        self.toolbox = toolbox
        self.store = store

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
        symbols = _symbols(tickers)
        options = dict(tickers=symbols or None, since_days=since_days, k=k, store=self.store)
        try:
            hits = search_news(query, diverse=diverse, newest_first=newest_first, **options)
            if hits:
                return {"found": True, "reason": None, "message": None, "hits": [h.as_dict() for h in hits]}
            closest = search_news(query, k=1, min_score=None, **{key: v for key, v in options.items() if key != "k"})
        except ValueError as error:  # every ValueError from the search is a bad argument
            raise ToolInputError(str(error)) from None
        where = ", ".join(filter(None, [f"tickers {', '.join(symbols)}" if symbols else "",
                                         f"the last {since_days} days" if since_days else ""])) or "no filters"
        if not closest:
            reason = "no_article_matches_filters"
            message = (f"No stored article matches the filters ({where}). " +
                       (f"The news for {', '.join(symbols)} may not be fetched yet: call fetch_news for it, then search again."
                        if symbols else "The news index may be empty or the time window too short."))
        else:
            reason = "nothing_relevant"
            message = (f"No supporting text: no stored article scored {MIN_SCORE:.2f} or more for this question ({where}); "
                       f"the closest scored {closest[0].score:.2f}. Say that the stored news does not cover it. "
                       f"Do not answer from memory.")
            if closest[0].score >= MIN_SCORE - NEAR_MISS:  # a short question often just scores a little low
                message += (" That is close to the bar, so if you believe the stored news should cover this, search "
                            "once more with a clearer question that names the company and the topic.")
        return {"found": False, "reason": reason, "message": message, "hits": []}

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
        symbol = _symbol(symbol)
        company = _text("company", company)
        keywords = _words("keywords", keywords)
        from_date = _days("from_date", from_date, MAX_NEWS_DAYS)
        stats = self.toolbox.fetch_news(company=company, symbol=symbol, keywords=keywords, from_date=from_date)
        indexed = ingest_symbol(symbol, store=self.store)  # the new articles become searchable now
        return {"symbol": symbol, **{key: stats.get(key) for key in ("fetched", "matched", "written", "duplicates")},
                "indexed": {key: indexed[key] for key in ("articles", "chunks", "added", "updated", "unchanged", "deleted")}}

    def fetch_press_releases(self, symbol: str, from_date: int = 365) -> dict:
        """Download a company's official press releases and store them. They are stored as text only: retrieve_news does not search them yet.

        Slow and uses a paid API. Call it only when the user asks for press releases.

        Args:
            symbol: The ticker symbol, such as "AAPL".
            from_date: How many days back to fetch, 1 to 730. The default is 365.
        """
        symbol = _symbol(symbol)
        from_date = _days("from_date", from_date, MAX_FILING_DAYS)
        stats = self.toolbox.fetch_press_releases(symbol, from_date=from_date)
        return {"symbol": symbol, **{key: stats.get(key) for key in ("fetched", "written", "duplicates")}}

    def fetch_filings(self, symbol: str, from_date: int = 365) -> dict:
        """Download a company's SEC filings (10-K, 10-Q, 8-K and others) and store them. They are stored as text only: retrieve_news does not search them yet, keyword_search does.

        Slow and uses a paid API. Call it only when the user asks for filings.

        Args:
            symbol: The ticker symbol, such as "AAPL".
            from_date: How many days back to fetch, 1 to 730. The default is 365.
        """
        symbol = _symbol(symbol)
        from_date = _days("from_date", from_date, MAX_FILING_DAYS)
        stats = self.toolbox.fetch_filings(symbol, from_date=from_date)
        return {"symbol": symbol, **{key: stats.get(key) for key in ("fetched", "written", "duplicates")}}

    # ---- keyword search over the stored text, and the other data

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
        found = self.toolbox.search(symbol, terms, from_date=from_date, refresh=False)
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
        return self.toolbox.discover(phrase, from_date=from_date, top=top, refresh=False)

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

    # ---- the LangChain edge

    def langchain_tools(self, book: SourceBook | None = None) -> list[BaseTool]:
        """The allow-listed tools as LangChain tools. `retrieve_news` returns the numbered sources as its text and
        the full hits (with their numbers) as the artifact; its numbers continue in `book`, one book per run."""
        book = book if book is not None else SourceBook()
        tools = []
        for name in DATA_CLERK_TOOLS:
            method = getattr(self, name)
            if name == "retrieve_news":
                tools.append(self._retrieve_news_tool(method, book))
            else:
                tools.append(tool(method, parse_docstring=True))
        return tools

    @staticmethod
    def _retrieve_news_tool(method, book: SourceBook) -> StructuredTool:
        @functools.wraps(method)  # same name, docstring and arguments as the method
        def retrieve_news(*args, **kwargs):
            result = method(*args, **kwargs)
            hits = [Hit(**h) for h in result["hits"]]
            if not hits:
                return result["message"], {"found": False, "reason": result["reason"], "hits": []}
            text = book.add(hits)
            return text, {"found": True, "reason": None,
                          "hits": [{"number": book.number_of(h), **h.as_dict()} for h in hits]}

        return StructuredTool.from_function(retrieve_news, name="retrieve_news", parse_docstring=True,
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
