"""Check that the RAG stack works end to end (Part 1 of hpw_to_do_langchain_rag.md).

Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/check_install.py

It indexes five real headlines from news/*.txt into a throw-away Chroma folder (deleted at the end),
so it cannot touch your real index. The first run downloads the embedding model (~90 MB) into
~/.cache/huggingface; after that it works offline. Any failed check stops the script with an error.
"""
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# (id, symbol, published_utc, publisher, text). Headlines and summaries are copied from news/*.txt.
RAW = [
    ("aapl-1", "AAPL", "2026-10-23T19:03:00Z", "Yahoo",
     "The iPhone Duo Launch Kicks Off a Crucial Few Weeks for Apple. Apple's foldable iPhone goes on "
     "sale ahead of an expected touchscreen MacBook launch and fiscal fourth-quarter earnings."),
    ("aapl-2", "AAPL", "2026-10-23T18:35:23Z", "Yahoo",
     "Apple Named Qnity as a New Domestic Manufacturing Partner. Qnity is partnering with Apple to "
     "produce technology for the iPhone 18 Pro camera's new variable aperture."),
    ("aapl-3", "AAPL", "2026-10-23T17:22:03Z", "CNBC",
     "Apple's iPhone Duo shows encouraging early demand signals, with third-party sales data still to "
     "come. CNBC's MacKenzie Sigalos reports from Apple's Fifth Avenue store as the company launches "
     "its first foldable in more than 70 countries, with limited inventory making the early demand "
     "picture difficult to gauge."),
    ("tsla-1", "TSLA", "2026-10-23T11:40:46Z", "Yahoo",
     "Tesla Gives Up On Cybertruck In Australia. Company Says It Will Refund All Deposits It Has "
     "Collected Since 2019 For The Troubled Truck"),
    ("nvda-1", "NVDA", "2026-10-17T21:30:00Z", "The Motley Fool",
     "CoreWeave Is Projected to Have a 6.3x Gross Leverage Ratio by Year End. The Market Is Worried "
     "About the Neocloud's Balance Sheet."),
]


def ts(iso: str) -> int:
    """ISO time -> epoch seconds. Chroma range filters ($gte, $lt ...) only compare NUMBERS."""
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' -> ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(f"CHECK FAILED: {label} {detail}")


def show(hits) -> None:
    for doc, score in hits:
        print(f"         {score:6.3f}  {doc.metadata['doc_id']:7} {doc.page_content[:64]}...")


def main() -> int:
    # 1) Embedding model: text -> a list of 384 numbers (a "vector"). Runs locally on the CPU.
    print("1) embeddings")
    t0 = time.time()
    emb = HuggingFaceEmbeddings(
        model_name=MODEL,
        encode_kwargs={"normalize_embeddings": True},  # unit-length vectors: cosine behaves well
    )
    vec = emb.embed_query("hello")
    check("vector length is 384", len(vec) == 384, f"model load + first embed {time.time() - t0:.1f}s, "
          f"device={emb._client.device}")

    docs = [
        Document(page_content=text, metadata={
            "doc_id": doc_id, "symbol": symbol, "published_utc": when,
            "published_ts": ts(when), "publisher": publisher})
        for doc_id, symbol, when, publisher, text in RAW
    ]
    ids = [r[0] for r in RAW]

    with tempfile.TemporaryDirectory() as tmp:
        # 2) A persistent Chroma store (a folder on disk). Telemetry is switched off here.
        print("2) Chroma store")
        client = chromadb.PersistentClient(path=tmp, settings=Settings(anonymized_telemetry=False))
        check("telemetry is off", client.get_settings().anonymized_telemetry is False)
        store = Chroma(
            client=client,
            collection_name="smoke_test",
            embedding_function=emb,
            collection_configuration={"hnsw": {"space": "cosine"}},  # relevance = 1 - cosine distance
        )

        # 3) Index: embed each document and store it. Same ids twice -> upsert, no duplicates.
        print("3) index")
        store.add_documents(docs, ids=ids)
        store.add_documents(docs, ids=ids)
        n = store._collection.count()
        check("5 documents added twice are stored once each (re-ingest is safe)", n == 5, f"count={n}")

        # 4) Semantic search: the query shares almost no words with the documents it should find.
        print("4) semantic search")
        q1 = "new folding smartphone sales"
        hits = store.similarity_search_with_relevance_scores(q1, k=3)
        print(f"     query: {q1!r}  (score = relevance, higher is better)")
        show(hits)
        check("top 2 hits are the Apple foldable-iPhone articles",
              {h[0].metadata["symbol"] for h in hits[:2]} == {"AAPL"} and
              all("foldable" in h[0].page_content for h in hits[:2]))
        check("relevance scores are sorted best-first", hits[0][1] >= hits[1][1] >= hits[2][1])

        q2 = "AI cloud company worried about its debt"
        hits = store.similarity_search_with_relevance_scores(q2, k=2)
        print(f"     query: {q2!r}")
        show(hits)
        check("top hit is the CoreWeave article", hits[0][0].metadata["doc_id"] == "nvda-1")

        # Two score flavours exist and point in opposite directions. Know which one you are holding:
        #   similarity_search_with_score(...)            -> DISTANCE  (lower  = closer)
        #   similarity_search_with_relevance_scores(...) -> RELEVANCE (higher = closer), = 1 - cosine distance
        distance = store.similarity_search_with_score(q1, k=1)[0][1]
        relevance = store.similarity_search_with_relevance_scores(q1, k=1)[0][1]
        check("relevance == 1 - distance (cosine)", abs((1 - distance) - relevance) < 1e-6,
              f"distance={distance:.3f}, relevance={relevance:.3f}")

        # 5) Metadata filters: the work order's subject and time period become filters on the search.
        print("5) metadata filters")
        only_tsla = store.similarity_search("earnings", k=5, filter={"symbol": "TSLA"})
        check("filter symbol == TSLA returns only TSLA", [d.metadata["doc_id"] for d in only_tsla] == ["tsla-1"])
        since = ts("2026-10-23T18:00:00Z")
        recent = store.similarity_search("iPhone", k=5, filter={"published_ts": {"$gte": since}})
        check("filter published_ts >= 18:00 drops the 17:22 article",
              sorted(d.metadata["doc_id"] for d in recent) == ["aapl-1", "aapl-2"])
        both = store.similarity_search(
            "iPhone", k=5, filter={"$and": [{"symbol": "AAPL"}, {"published_ts": {"$gte": since}}]})
        check("symbol AND date filters combine", sorted(d.metadata["doc_id"] for d in both) == ["aapl-1", "aapl-2"])

        # 6) A retriever is what the LangGraph tool will call (this is as_retriever from the guide).
        print("6) retriever")
        retriever = store.as_retriever(search_kwargs={"k": 2, "filter": {"symbol": "AAPL"}})
        got = retriever.invoke("how is iPhone demand?")
        check("retriever honours k and filter", len(got) == 2 and all(d.metadata["symbol"] == "AAPL" for d in got))

        # 7) A Chroma database is just a folder; a brand-new process can read it back.
        print("7) persistence")
        files = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file())
        print(f"     files on disk: {files}")
        code = ("import chromadb; from chromadb.config import Settings; "
                f"c = chromadb.PersistentClient(path={tmp!r}, settings=Settings(anonymized_telemetry=False)); "
                "print(c.get_collection('smoke_test').count())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        check("a new process sees the 5 stored documents", out.stdout.strip().splitlines()[-1] == "5",
              out.stdout.strip().splitlines()[-1])

    print("\nALL CHECKS PASSED: LangChain + Chroma + local embeddings work on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
