"""Ingest news into the vector store: stage 1 of the slide (Chunk text, Encode chunks, Store vectors + text).

    python -m prometheus.rag.ingest AAPL NVDA        ingest news/AAPL.txt and news/NVDA.txt
    python -m prometheus.rag.ingest --all            ingest every file in the news folder
    python -m prometheus.rag.ingest --rebuild --all  delete the collection first (the index is derived data)
    python -m prometheus.rag.ingest --press-releases AAPL   the same for <SYMBOL>_PRs.txt, into the press release collection (entry 34)
    python -m prometheus.rag.ingest --filings AAPL          the same for the SEC filings file <SYMBOL>.txt in the filings folder, into the filings collection (entry 35)

It is safe to run again. A chunk that has not changed is skipped (it is not encoded again), a changed
chunk is overwritten, a chunk that no longer exists is deleted, and an article that is stored a second
time (the same article sits in more than one symbol's file) gets its `tickers` lists merged.
"""
import argparse
import sys
import time
from collections import defaultdict
from typing import Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prometheus.rag.config import filings_dir, news_dir, press_releases_dir
from prometheus.rag.loaders import (PRESS_RELEASES_SUFFIX, filing_file, load_filings, load_news, load_press_releases, news_file,
                                    press_release_file)
from prometheus.rag.store import count_chunks, count_too_long, get_vector_store, reset_collection

# Characters, not tokens. With these numbers no chunk of the 288 news articles passes the embedding model's
# 256-token limit, even with the headline put back in front (design choices entry 17).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
HEADLINE_PREFIX_MAX = 250  # the longest real headline is 208 characters (design choices entry 19)
LOOKUP_BATCH = 100  # article ids per lookup in the store


def _prefix(headline: str) -> str:
    """The headline as it is put in front of a later chunk. An absurdly long one is cut short: a 3,000
    character headline in front of every chunk would fill the embedder's whole window and leave the
    article unfindable (found by breaking the input on purpose, entry 19)."""
    if len(headline) <= HEADLINE_PREFIX_MAX:
        return headline
    return headline[:HEADLINE_PREFIX_MAX].rsplit(" ", 1)[0] + " ..."


def to_chunks(docs: list[Document]) -> list[Document]:
    """Cut each article into chunks. A chunk's id is <article id>#<number>, and it keeps the article's
    metadata plus start_index (where it begins in the article), chunk and chunks (its number and how many).

    A chunk that does not begin with the headline gets the headline put back in front, so that a chunk
    like "CRWV | negative | Stock declined 11.91% ..." still says which article it belongs to."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
    chunks = []
    for doc in docs:
        headline = doc.metadata["headline"]
        parts = splitter.split_documents([doc])
        for number, part in enumerate(parts):
            text = part.page_content if part.page_content.startswith(headline) else f"{_prefix(headline)}\n\n{part.page_content}"
            chunks.append(Document(
                id=f"{doc.id}#{number}",
                page_content=text,
                metadata={**part.metadata, "chunk": number, "chunks": len(parts)},
            ))
    return chunks


def merge_tickers(stored: list[str], new: list[str]) -> list[str]:
    """The union of two ticker lists, the stored ones first, without repeats."""
    return stored + [ticker for ticker in new if ticker not in stored]


def _stored(store: Chroma, article_ids: list[str]) -> dict[str, dict[str, tuple[str, dict]]]:
    """What the store already holds for these articles: {article id: {chunk id: (text, metadata)}}."""
    found: dict[str, dict[str, tuple[str, dict]]] = defaultdict(dict)
    for start in range(0, len(article_ids), LOOKUP_BATCH):
        batch = article_ids[start:start + LOOKUP_BATCH]
        got = store.get(where={"article_id": {"$in": batch}}, include=["documents", "metadatas"])
        for chunk_id, text, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            found[meta["article_id"]][chunk_id] = (text, meta)
    return found


def ingest_documents(docs: list[Document], store: Chroma,
                     too_long: Callable[[list[str]], int] | None = None) -> dict:
    """Chunk, encode and store these articles, reconciling with what the store already holds.

    Returns {articles, chunks, added, updated, unchanged, deleted, over_limit}. over_limit is how many
    chunks the embedding model would cut off (None when no `too_long` check is given)."""
    articles: dict[str, Document] = {}
    for doc in docs:  # the same article twice in one call: keep the first text, merge the tickers
        if doc.id in articles:
            known = articles[doc.id].metadata
            known["tickers"] = merge_tickers(known["tickers"], doc.metadata["tickers"])
        else:
            articles[doc.id] = Document(id=doc.id, page_content=doc.page_content, metadata=dict(doc.metadata))

    chunks = to_chunks(list(articles.values()))
    by_article: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        by_article[chunk.metadata["article_id"]].append(chunk)
    stored = _stored(store, list(articles))

    stats = {"articles": len(articles), "chunks": len(chunks), "added": 0, "updated": 0, "unchanged": 0, "deleted": 0}
    to_write: list[Document] = []
    stale: list[str] = []
    for article_id, new_chunks in by_article.items():
        old = stored.get(article_id, {})
        tickers: list[str] = []
        for _, meta in old.values():
            tickers = merge_tickers(tickers, meta["tickers"])
        tickers = merge_tickers(tickers, new_chunks[0].metadata["tickers"])
        for chunk in new_chunks:
            chunk.metadata["tickers"] = tickers
            if chunk.id not in old:
                stats["added"] += 1
                to_write.append(chunk)
            elif old[chunk.id] == (chunk.page_content, chunk.metadata):
                stats["unchanged"] += 1
            else:
                stats["updated"] += 1
                to_write.append(chunk)
        stale += [chunk_id for chunk_id in old if chunk_id not in {c.id for c in new_chunks}]

    if stale:
        store.delete(ids=stale)
    if to_write:
        store.add_documents(to_write, ids=[chunk.id for chunk in to_write])  # encodes the text, then upserts
    stats["deleted"] = len(stale)
    stats["over_limit"] = too_long([c.page_content for c in chunks]) if too_long else None
    return stats


def ingest_symbol(symbol: str, store: Chroma | None = None) -> dict:
    """Ingest news/<SYMBOL>.txt. With no store given it uses the real one (and checks chunk lengths
    against the real embedding model); a test passes its own store."""
    path = news_file(symbol)
    stats = ingest_documents(load_news(path), store or get_vector_store(),
                             too_long=count_too_long if store is None else None)
    return {"symbol": symbol.strip().upper(), "path": str(path), **stats}


def ingest_press_releases(symbol: str, store: Chroma | None = None) -> dict:
    """Ingest <press releases dir>/<SYMBOL>_PRs.txt into the PRESS RELEASE collection (its own collection, entry 34). Exactly the news path
    (`ingest_documents`: the same chunk size, the same headline in front of a later chunk, the same safe re-runs) with another loader and another
    store. With no store given it uses the real one; a test passes its own."""
    path = press_release_file(symbol)
    stats = ingest_documents(load_press_releases(path), store or get_vector_store("press_releases"),
                             too_long=count_too_long if store is None else None)
    return {"symbol": symbol.strip().upper(), "path": str(path), **stats}


def ingest_filings(symbol: str, store: Chroma | None = None) -> dict:
    """Ingest <filings dir>/<SYMBOL>.txt into the FILINGS collection (its own collection, entry 35). The news path again (`ingest_documents`: the same chunk size,
    the same headline in front of a later chunk, the same safe re-runs) with another loader and another store. A 10-K section is 16,000 to 200,000 characters, so it
    becomes 20 to 250 chunks that all share one article id (the section's); the headline in front of each says form, section and date. With no store given it uses
    the real one; a test passes its own."""
    path = filing_file(symbol)
    stats = ingest_documents(load_filings(path), store or get_vector_store("filings"),
                             too_long=count_too_long if store is None else None)
    return {"symbol": symbol.strip().upper(), "path": str(path), **stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="*", help="ticker symbols to ingest, for example AAPL NVDA")
    parser.add_argument("--all", action="store_true", help="ingest every .txt file in the news folder (with --press-releases: every <SYMBOL>_PRs.txt; with --filings: every .txt in the filings folder)")
    parser.add_argument("--rebuild", action="store_true", help="delete the collection before ingesting (the news one, or with --press-releases or --filings that one)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--press-releases", action="store_true", help="ingest <SYMBOL>_PRs.txt (press releases) into the press release collection, not news")
    which.add_argument("--filings", action="store_true", help="ingest <SYMBOL>.txt of the filings folder (SEC filings) into the filings collection, not news")
    args = parser.parse_args(argv)
    if bool(args.symbols) == args.all:
        parser.error("give ticker symbols, or --all (not both)")
    kind, ingest = (("press_releases", ingest_press_releases) if args.press_releases else ("filings", ingest_filings) if args.filings else ("news", ingest_symbol))
    noun = "press releases" if args.press_releases else "filing documents" if args.filings else "articles"
    try:
        if args.press_releases:
            folder = press_releases_dir()
            found = [path.name[:-len(PRESS_RELEASES_SUFFIX)] for path in sorted(folder.glob(f"*{PRESS_RELEASES_SUFFIX}"))]
        elif args.filings:
            folder = filings_dir()
            found = [path.stem for path in sorted(folder.glob("*.txt"))]
        else:
            folder = news_dir()
            found = [path.stem for path in sorted(folder.glob("*.txt"))]
        symbols = found if args.all else args.symbols
        if not symbols:
            print(f"ERROR: no {'press release' if args.press_releases else 'filing' if args.filings else 'news'} files in {folder}", file=sys.stderr)
            return 1
        if args.rebuild:
            print("rebuild: " + ("deleted the collection" if reset_collection(kind) else "there was no collection to delete"))
        for symbol in symbols:
            started = time.monotonic()
            s = ingest(symbol)
            print(f"{s['symbol']}: {s['articles']} {noun} -> {s['chunks']} chunks | added {s['added']}, "
                  f"updated {s['updated']}, unchanged {s['unchanged']}, deleted {s['deleted']} | "
                  f"chunks over the embedder's limit: {s['over_limit']} | {time.monotonic() - started:.1f} s")
            if s["over_limit"]:  # silent but harmful: the model ignores the end of such a chunk
                print(f"WARNING: {s['symbol']}: {s['over_limit']} chunks are longer than the embedder reads, so the end "
                      f"of their text cannot be found by search (design choices entry 19)", file=sys.stderr)
        print(f"the collection now holds {count_chunks(kind)} chunks")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
