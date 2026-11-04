"""Search the stored news by meaning: stage 2 of the slide (Encode query, Search vectors, Fetch text by ID).

    python -m prometheus.rag.retriever "what did MKBHD say about the foldable iPhone" --tickers AAPL
    python -m prometheus.rag.retriever "what is happening with Meta" --tickers META --since-days 7 --diverse
    python -m prometheus.rag.retriever "what did Apple announce about AI" --tickers AAPL --press-releases   # the press releases (entry 34)
    python -m prometheus.rag.retriever "what risks does Apple name" --tickers AAPL --filings                 # the SEC filings (entry 35)

`search_news` is the one function the rest of the system calls (the Part 6 tool wraps it). It answers the four
questions the week 3 design asks of retrieval (design choices entry 20):

* Which chunks may be searched? Only those that pass the filters: a list of tickers and a rolling window of
  the last N days. The filters are part of the search itself, so they never leave fewer results than asked for.
* How good is a hit? Its score is the cosine similarity between the question and the chunk (1.0 is identical).
* Is there anything? A hit must score at least MIN_SCORE. If none does, the result is an empty list, and the
  caller has to say "no supporting text" instead of guessing.
* No repeats. One hit per article (its best chunk). With diverse=True the hits are also spread over different
  stories (maximal marginal relevance), for a broad question over news that repeats itself.
* Newest first, on request. A search ranks by meaning, not by date, so a question about "the latest news" can be
  answered with older articles that match the words better (found in the SNDK run: all five hits were from Oct 19
  while newer articles were stored). With newest_first=True the articles that pass the relevance bar are ordered
  by date, newest first, and the newest k are returned.
"""
import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from prometheus.rag.loaders import check_symbol
from prometheus.rag.store import count_chunks, get_vector_store

# Every number below was set from real questions (guide Part 5, design choices entry 20).
K = 5               # articles returned when the caller does not say: the answer was in the top 5 for 23 of 23 questions
MAX_K = 20          # the most a caller may ask for: the Advisor needs a few chunks, not a flood
MIN_SCORE = 0.40    # a hit must score at least this. Every answerable question scored 0.47 or more, every off-topic one
                    # under 0.40. It cannot tell "right company, topic not in the news" (scores 0.40 to 0.64) from an answer
FETCH_FACTOR = 10   # candidate chunks fetched per wanted article, so that grouping by article still finds enough
MMR_LAMBDA = 0.5    # for diverse=True: 1.0 is all relevance, 0.0 is all variety; 0.3 lost answers, 0.7 changed little


@dataclass(frozen=True)
class Hit:
    """One search result: what a citation needs, plus the text and the score."""
    chunk_id: str        # "polygon:9f6e...#1": the article id, then the chunk number counted from 0
    article_id: str
    chunk: int           # which chunk of the article this is, counted from 1
    chunks: int          # how many chunks the article has
    headline: str
    publisher: str
    published_utc: str   # "2026-10-23T19:03:00Z"
    url: str
    tickers: list[str]
    text: str
    score: float         # cosine similarity to the question; higher is better

    def as_dict(self) -> dict:
        return asdict(self)


def _symbols(tickers: str | list[str] | None) -> list[str]:
    """None, one symbol or a list of symbols -> a list of checked, upper-case symbols."""
    if tickers is None:
        return []
    items = [tickers] if isinstance(tickers, str) else list(tickers)
    if not all(isinstance(t, str) for t in items):
        raise ValueError("tickers must be text, like 'AAPL' or ['AAPL', 'NVDA']")
    return [check_symbol(t) for t in items]


def _where(symbols: list[str], since_ts: int | None) -> dict | None:
    """The Chroma filter. `tickers` is a list in the metadata, so a symbol is matched with $contains; several
    symbols are OR-ed. Chroma wants at least two clauses inside $or and $and, hence the special cases."""
    clauses = []
    if symbols:
        has = [{"tickers": {"$contains": s}} for s in symbols]
        clauses.append(has[0] if len(has) == 1 else {"$or": has})
    if since_ts is not None:
        clauses.append({"published_ts": {"$gte": since_ts}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _best_chunk_per_article(scored: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """The input is best first, so the first chunk seen for an article is its best one."""
    best: dict[str, tuple[Document, float]] = {}
    for doc, score in scored:
        best.setdefault(doc.metadata["article_id"], (doc, score))
    return list(best.values())


def _spread(store: Chroma, query: str, candidates: list[tuple[Document, float]], k: int,
            lambda_mult: float) -> list[tuple[Document, float]]:
    """Maximal marginal relevance over the candidates: each pick is the one that is most like the question and
    least like what is already picked. The first pick is always the best match."""
    ids = [doc.id for doc, _ in candidates]
    got = store.get(ids=ids, include=["embeddings"])
    vector_of = dict(zip(got["ids"], got["embeddings"]))
    query_vector = np.array(store.embeddings.embed_query(query), dtype=np.float32)
    picked = maximal_marginal_relevance(query_vector, [vector_of[i] for i in ids], lambda_mult=lambda_mult, k=k)
    return [candidates[i] for i in picked]


def _hit(doc: Document, score: float) -> Hit:
    m = doc.metadata
    return Hit(chunk_id=doc.id, article_id=m["article_id"], chunk=m["chunk"] + 1, chunks=m["chunks"],
               headline=m["headline"], publisher=m["publisher"], published_utc=m["published_utc"], url=m["url"],
               tickers=list(m["tickers"]), text=doc.page_content, score=score)


def search_news(query: str, *, tickers: str | list[str] | None = None, since_days: int | None = None,
                k: int = K, min_score: float | None = MIN_SCORE, diverse: bool = False,
                lambda_mult: float = MMR_LAMBDA, newest_first: bool = False, store: Chroma | None = None,
                now: datetime | None = None) -> list[Hit]:
    """Search the news index by meaning. Returns at most k hits, one per article, best first (or, with
    diverse=True, in the order maximal marginal relevance picks them). An empty list means "no supporting text".

    query:      the question, in words.
    tickers:    one symbol or a list: only articles that name any of them in their `tickers` (for a Finnhub
                article that list is only the symbol it was fetched under, so the query text still matters).
    since_days: only articles published in the last N days, counted back from `now` (rolling, N * 24 hours).
    k:          how many articles, 1 to MAX_K.
    min_score:  the relevance bar, 0 to 1; None turns it off (for looking at scores).
    diverse:    spread the hits over different stories; use it for a broad question, not a narrow one.
    newest_first: of the articles that pass the relevance bar (among the closest k * FETCH_FACTOR chunks), return the
                newest k, newest first, instead of the k that match best. For "latest", "newest", "most recent". The bar
                still applies, so a new article that is not about the question is not returned. It wins over diverse.
    store, now: for tests; the real index and the real clock when left out."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("the question must be a non-empty text")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be a whole number from 1 to {MAX_K}")
    if since_days is not None and (not isinstance(since_days, int) or isinstance(since_days, bool) or since_days < 1):
        raise ValueError("since_days must be a whole number of days, 1 or more")
    if min_score is not None and not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1 (or None for no bar)")
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult must be between 0 and 1")
    if not isinstance(newest_first, bool):
        raise ValueError("newest_first must be true or false")
    symbols = _symbols(tickers)

    since_ts = None
    if since_days is not None:
        since_ts = int(((now or datetime.now(timezone.utc)) - timedelta(days=since_days)).timestamp())
    store = store or get_vector_store()
    # The store returns cosine DISTANCES (0 is identical) and the collection is cosine (entry 10), so 1 - distance
    # is the cosine similarity. LangChain's similarity_search_with_relevance_scores adds up the same way, but it
    # warns, with a message that holds every document, as soon as one score is below 0. That happens when a large
    # k reaches far-away chunks, and the experiments hit it (a test checks that the two agree).
    scored = [(doc, 1.0 - distance) for doc, distance in store.similarity_search_with_score(
        query.strip(), k=k * FETCH_FACTOR, filter=_where(symbols, since_ts))]

    candidates = _best_chunk_per_article(scored)
    if min_score is not None:
        candidates = [(doc, score) for doc, score in candidates if score >= min_score]
    if newest_first:
        candidates = sorted(candidates, key=lambda c: (-c[0].metadata["published_ts"], -c[1]))  # newest first, the better match first on a tie
    elif diverse and len(candidates) > k:
        candidates = _spread(store, query.strip(), candidates, k, lambda_mult)
    return [_hit(doc, score) for doc, score in candidates[:k]]


def search_press_releases(query: str, **options) -> list[Hit]:
    """`search_news` over the PRESS RELEASE collection (design choices entry 34): the same filters, relevance bar, one hit per release, `diverse`
    and `newest_first`, and the same `Hit`. `options` are those of `search_news`; `store` is the press release store (the real one when left out)."""
    store = options.pop("store", None) or get_vector_store("press_releases")
    return search_news(query, store=store, **options)


def search_filings(query: str, **options) -> list[Hit]:
    """`search_news` over the FILINGS collection (design choices entry 35): the same filters, relevance bar, one hit per article (here: per filing SECTION, so a 10-K
    gives at most two hits, its Business and its Risk Factors), `diverse` and `newest_first`, and the same `Hit`. `options` are those of `search_news`; `store` is
    the filings store (the real one when left out)."""
    store = options.pop("store", None) or get_vector_store("filings")
    return search_news(query, store=store, **options)


def _describe_filters(tickers: list[str] | None, since_days: int | None) -> str:
    parts = []
    if tickers:
        parts.append("tickers " + ", ".join(tickers))
    if since_days:
        parts.append(f"last {since_days} days")
    return " | ".join(parts) or "none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the stored news by meaning.")
    parser.add_argument("query", help="the question, in words")
    parser.add_argument("--tickers", nargs="+", metavar="SYMBOL", help="only articles that name any of these tickers")
    parser.add_argument("--since-days", type=int, help="only articles from the last N days")
    parser.add_argument("-k", type=int, default=K, help=f"how many articles (default {K}, at most {MAX_K})")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help=f"relevance bar from 0 to 1 (default {MIN_SCORE})")
    parser.add_argument("--diverse", action="store_true", help="spread the hits over different stories (MMR)")
    parser.add_argument("--newest-first", action="store_true", help="of the relevant articles, the newest first (for 'latest' questions)")
    parser.add_argument("--text", action="store_true", help="also print the text of each hit")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--press-releases", action="store_true", help="search the press releases instead of the news")
    which.add_argument("--filings", action="store_true", help="search the SEC filings instead of the news")
    args = parser.parse_args(argv)
    kind, search = (("press_releases", search_press_releases) if args.press_releases else ("filings", search_filings) if args.filings else ("news", search_news))
    try:
        if not count_chunks(kind):
            print("nothing is stored yet. Ingest some " + {"press_releases": "press releases first:  python -m prometheus.rag.ingest --press-releases --all",
                                                           "filings": "filings first:  python -m prometheus.rag.ingest --filings --all",
                                                           "news": "news first:  python -m prometheus.rag.ingest --all"}[kind], file=sys.stderr)
            return 1
        asked = dict(tickers=args.tickers, since_days=args.since_days, k=args.k, diverse=args.diverse, newest_first=args.newest_first)
        hits = search(args.query, min_score=args.min_score, **asked)
        print(f"question : {args.query!r}")
        print(f"filters  : {_describe_filters(args.tickers, args.since_days)}")
        print(f"settings : k={args.k}, relevance bar {args.min_score}, diverse {'yes' if args.diverse else 'no'}, newest first {'yes' if args.newest_first else 'no'}")
        if not hits:
            closest = search(args.query, **{**asked, "k": 1, "diverse": False, "newest_first": False}, min_score=None)
            if closest:
                c = closest[0]
                print(f"\nno supporting text: nothing scored {args.min_score} or more.")
                print(f"closest was {c.score:.2f}: {c.publisher}, {c.published_utc[:10]}: {c.headline[:80]}")
            else:
                print("\nno supporting text: no article matches the filters at all.")
            return 0
        print()
        for rank, h in enumerate(hits, 1):
            print(f"{rank:>2}  {h.score:.2f}  {h.publisher[:13]:<13} {h.published_utc[:10]}  {h.headline[:72]}  "  # 13: "Press release" fits whole
                  f"[{', '.join(h.tickers[:4])}{', ...' if len(h.tickers) > 4 else ''}]  chunk {h.chunk}/{h.chunks}")
            print(f"            {h.url}")
            if args.text:
                print("            " + h.text.replace("\n", "\n            "))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
