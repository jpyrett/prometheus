"""Ingest news into the vector store: stage 1 of the slide (Chunk text, Encode chunks, Store vectors + text).

    python -m prometheus.rag.ingest AAPL NVDA        ingest news/AAPL.txt and news/NVDA.txt
    python -m prometheus.rag.ingest --all            ingest every file in the news folder
    python -m prometheus.rag.ingest --rebuild --all  delete the collection first (the index is derived data)
    python -m prometheus.rag.ingest --press-releases AAPL   the same for <SYMBOL>_PRs.txt, into the press release collection (entry 34)
    python -m prometheus.rag.ingest --filings AAPL          the same for the SEC filings file <SYMBOL>.txt in the filings folder, into the filings collection (entry 35)

A record that fails to load (a broken article or press release) is DISCARDED, not stored and not an error (design choices entry 83): the stats say how many (`skipped`, a list of `Skipped`) and the
command prints them; the others are ingested as usual. A chunk that was stored before the record broke is left as it was: `--rebuild` clears it.

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
from prometheus.rag.loaders import (PRESS_RELEASES_SUFFIX, Skipped, filing_file, load_filings, load_news, load_press_releases, news_file,
                                    press_release_file)
from prometheus.rag.store import count_chunks, count_too_long, get_vector_store, reset_collection, token_lengths, token_limit

# Characters, not tokens. With these numbers no chunk of the stored news passes the embedding model's
# 256-token limit, even with the headline put back in front (design choices entry 17: 800, measured on 288 articles).
# 680 since entry 130 (2026-11-20): the 6 months of backfilled news (132,847 articles) had 8 chunks over the limit at 800,
# 4 at 760, 1 at 720 and none at 680 (the longest is 247 tokens). Press releases: none at any of them (174 tokens).
# Filings share the size and are NOT inside the limit at either (dense tables: 2,701 chunks over at 800, 1,298 at 680).
CHUNK_SIZE = 680
CHUNK_OVERLAP = 100
HEADLINE_PREFIX_MAX = 250  # the longest real headline is 208 characters (design choices entry 19)
LOOKUP_BATCH = 100  # article ids per lookup in the store
WRITE_BATCH = 2000  # chunks per upsert, under Chroma's max batch size
FIT_SHRINK = 0.85  # a chunk over the token limit is cut again at its own characters-per-token rate, less this margin
FIT_ROUNDS = 12     # the most times one chunk is cut again (each round shrinks it; a dense table needs two or three)
FIT_MIN_CHARS = 40  # under this a piece is cut in two by position: the splitter has no separator left to cut at


def _prefix(headline: str) -> str:
    """The headline as it is put in front of a later chunk. An absurdly long one is cut short: a 3,000
    character headline in front of every chunk would fill the embedder's whole window and leave the
    article unfindable (found by breaking the input on purpose, entry 19)."""
    if len(headline) <= HEADLINE_PREFIX_MAX:
        return headline
    return headline[:HEADLINE_PREFIX_MAX].rsplit(" ", 1)[0] + " ..."


def _with_headline(headline: str, body: str) -> str:
    """The text of a chunk: its body, with the headline put back in front when the body does not begin with it."""
    return body if body.startswith(headline) else f"{_prefix(headline)}\n\n{body}"


def _cut_again(start: int, body: str, tokens: int, limit: int) -> list[tuple[int, str]]:
    """One piece that is too long for the embedder, cut into smaller pieces: [(where it begins in the article, its text)]. The new size is the piece's own
    characters-per-token rate times the limit, less a margin, so a dense table (many tokens per character) is cut finer than prose. A piece the splitter
    cannot cut (no separator left) is cut in two by position."""
    size = max(FIT_MIN_CHARS, int(len(body) * limit / tokens * FIT_SHRINK))
    pieces = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=min(CHUNK_OVERLAP, size // 4), add_start_index=True).create_documents([body])
    if len(pieces) < 2:
        half = len(body) // 2
        return [(start, body[:half]), (start + half, body[half:])] if half else [(start, body)]
    return [(start + piece.metadata["start_index"], piece.page_content) for piece in pieces]


def _fit(headline: str, parts: list[tuple[int, str]], lengths: Callable[[list[str]], list[int]], limit: int) -> list[tuple[int, str]]:
    """The pieces of one article, each cut again until its chunk text (the headline in front included) is within the embedder's token limit. The pieces that
    already fit are not touched, so prose is chunked exactly as by characters alone; only what is too dense is cut finer."""
    for _ in range(FIT_ROUNDS):
        counts = lengths([_with_headline(headline, body) for _, body in parts])
        if all(count <= limit for count in counts):
            break
        fitted: list[tuple[int, str]] = []
        for (start, body), count in zip(parts, counts):
            fitted.extend([(start, body)] if count <= limit else _cut_again(start, body, count, limit))
        if len(fitted) == len(parts):   # nothing could be cut (a headline that alone is over the limit): said by the length check, not looped on
            break
        parts = fitted
    return parts


def to_chunks(docs: list[Document], lengths: Callable[[list[str]], list[int]] | None = None, limit: int | None = None) -> list[Document]:
    """Cut each article into chunks. A chunk's id is <article id>#<number>, and it keeps the article's
    metadata plus start_index (where it begins in the article), chunk and chunks (its number and how many).

    A chunk that does not begin with the headline gets the headline put back in front, so that a chunk
    like "CRWV | negative | Stock declined 11.91% ..." still says which article it belongs to.

    BY CHARACTERS, THEN BY TOKENS (design choices entry 131). The first cut is by characters (`CHUNK_SIZE`). With `lengths` (the embedding model's own token
    count of each text, `store.token_lengths`) and `limit` (`store.token_limit`), a chunk whose text is over the limit is cut again until it fits (`_fit`):
    the embedder ignores everything past its limit, and a filing's table has two or three times the tokens per character of prose, so no character count
    keeps it inside. Without them (a test with its own store) the cut is by characters alone, as before."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
    chunks = []
    for doc in docs:
        headline = doc.metadata["headline"]
        parts = [(part.metadata["start_index"], part.page_content) for part in splitter.split_documents([doc])]
        if lengths is not None and limit is not None:
            parts = _fit(headline, parts, lengths, limit)
        for number, (start, body) in enumerate(parts):
            chunks.append(Document(
                id=f"{doc.id}#{number}",
                page_content=_with_headline(headline, body),
                metadata={**doc.metadata, "start_index": start, "chunk": number, "chunks": len(parts)},
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
                     too_long: Callable[[list[str]], int] | None = None, fit: bool = False) -> dict:
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

    chunks = to_chunks(list(articles.values()), token_lengths, token_limit()) if fit else to_chunks(list(articles.values()))   # `fit`: the real store, the real embedder (entry 131)
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
        for start in range(0, len(to_write), WRITE_BATCH):  # Chroma refuses one upsert over its own batch limit (5461 here): a symbol with months of news has more chunks than that
            batch = to_write[start:start + WRITE_BATCH]
            store.add_documents(batch, ids=[chunk.id for chunk in batch])  # encodes the text, then upserts
    stats["deleted"] = len(stale)
    stats["over_limit"] = too_long([c.page_content for c in chunks]) if too_long else None
    return stats


def ingest_symbol(symbol: str, store: Chroma | None = None) -> dict:
    """Ingest news/<SYMBOL>.txt. With no store given it uses the real one (and checks chunk lengths
    against the real embedding model); a test passes its own store."""
    path = news_file(symbol)
    skipped: list[Skipped] = []
    stats = ingest_documents(load_news(path, skipped), store or get_vector_store(),
                             too_long=count_too_long if store is None else None, fit=store is None)
    return {"symbol": symbol.strip().upper(), "path": str(path), **stats, "skipped": skipped}


def ingest_press_releases(symbol: str, store: Chroma | None = None) -> dict:
    """Ingest <press releases dir>/<SYMBOL>_PRs.txt into the PRESS RELEASE collection (its own collection, entry 34). Exactly the news path
    (`ingest_documents`: the same chunk size, the same headline in front of a later chunk, the same safe re-runs) with another loader and another
    store. With no store given it uses the real one; a test passes its own."""
    path = press_release_file(symbol)
    skipped: list[Skipped] = []
    stats = ingest_documents(load_press_releases(path, skipped), store or get_vector_store("press_releases"),
                             too_long=count_too_long if store is None else None, fit=store is None)
    return {"symbol": symbol.strip().upper(), "path": str(path), **stats, "skipped": skipped}


def ingest_filings(symbol: str, store: Chroma | None = None) -> dict:
    """Ingest <filings dir>/<SYMBOL>.txt into the FILINGS collection (its own collection, entry 35). The news path again (`ingest_documents`: the same chunk size,
    the same headline in front of a later chunk, the same safe re-runs) with another loader and another store. A 10-K section is 16,000 to 200,000 characters, so it
    becomes 20 to 250 chunks that all share one article id (the section's); the headline in front of each says form, section and date. With no store given it uses
    the real one; a test passes its own."""
    path = filing_file(symbol)
    stats = ingest_documents(load_filings(path), store or get_vector_store("filings"),
                             too_long=count_too_long if store is None else None, fit=store is None)
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
            if s.get("skipped"):  # a failed record is discarded: never stored, never found, never scored (entry 83); a person looks at the file
                print(f"  discarded {len(s['skipped'])} failed record{'s' if len(s['skipped']) != 1 else ''} (not stored): "
                      + "; ".join(f"{item.where}: {item.reason}" for item in s["skipped"][:5]) + (" ..." if len(s["skipped"]) > 5 else ""))
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
