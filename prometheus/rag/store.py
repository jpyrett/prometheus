"""The embedding model and the Chroma collection ("Encode chunks" and "Store vectors + text" in the slide).

This is the only module that names the embedding model class or the vector database. Everything else
asks it for `get_vector_store()`, so changing either one is a change here and in config.py.

Look at what is stored:   python -m prometheus.rag.store
"""
import argparse
import re
import statistics
import sys
from collections import Counter
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from prometheus.rag.config import embedding_device, embedding_model, vector_store_path


@lru_cache(maxsize=2)
def _load_embeddings(model: str, device: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # unit-length vectors: cosine behaves well
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """The embedding model from the settings. Loading takes a few seconds, so it happens once per process."""
    return _load_embeddings(embedding_model(), embedding_device())


# What can be stored. Each kind is its own collection, so a search of one never sees the others and the news collection is the same
# as it was before press releases and filings existed (design choices entries 34 and 35).
KINDS = ("news", "press_releases", "filings")


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}: one of {', '.join(KINDS)}")
    return kind


def collection_name(model: str | None = None, kind: str = "news") -> str:
    """<kind>-<model>, for example news-sentence-transformers-all-MiniLM-L6-v2. The model's name is part of the collection's name,
    so vectors from two different models never share a collection (they cannot be compared with each other).
    Chroma allows only letters, digits, '.', '_' and '-', so anything else (the '/') becomes '-'."""
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", model or embedding_model()).strip("-._")
    return f"{_kind(kind)}-{clean}"


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(vector_store_path()), settings=Settings(anonymized_telemetry=False))


# How Chroma's approximate search (HNSW) is set up. The space is cosine (design choices entry 10). The other three
# make the search thorough. With Chroma's defaults (100, 100, 16) the search found only 96% to 98% of the true
# nearest chunks on this corpus, and after many in-place updates some questions got a different top 10. These
# values found 99.9% to 100% and cost nothing measurable at this size (entry 20). A collection keeps the settings
# it was created with, so a change here takes effect after `python -m prometheus.rag.ingest --rebuild --all`.
HNSW = {"space": "cosine", "ef_construction": 400, "ef_search": 300, "max_neighbors": 32}


def get_vector_store(kind: str = "news") -> Chroma:
    """The Chroma collection for `kind` ("news", the default, "press_releases" or "filings"): a folder on disk, telemetry off, cosine distance (fixed when the
    collection is created, design choices entry 10) and a thorough search (HNSW above). All kinds live in the same folder, each in its own collection."""
    return Chroma(
        client=_client(),
        collection_name=collection_name(kind=kind),
        embedding_function=get_embeddings(),
        collection_configuration={"hnsw": HNSW},
    )


def reset_collection(kind: str = "news") -> bool:
    """Delete the collection of `kind` (news by default). The index is derived data: ingest again to rebuild it.
    Returns whether there was a collection to delete."""
    try:
        _client().delete_collection(collection_name(kind=kind))
        return True
    except NotFoundError:
        return False


def count_chunks(kind: str = "news") -> int:
    """How many chunks the collection of `kind` holds (0 if there is none yet)."""
    try:
        return _client().get_collection(collection_name(kind=kind)).count()
    except NotFoundError:
        return 0


def count_too_long(texts: list[str], embeddings: HuggingFaceEmbeddings | None = None) -> int:
    """How many of these texts are longer than the embedding model reads. The model silently ignores
    everything past its limit (256 tokens for MiniLM), so a text over the limit is only partly searchable.

    LangChain keeps the real model in `_client`. This function is the only place that touches it."""
    limit = token_limit(embeddings)
    return sum(length > limit for length in token_lengths(texts, embeddings))


def token_limit(embeddings: HuggingFaceEmbeddings | None = None) -> int:
    """How many tokens the embedding model reads of one text (256 for MiniLM), its own special tokens included."""
    return (embeddings or get_embeddings())._client.max_seq_length


def token_lengths(texts: list[str], embeddings: HuggingFaceEmbeddings | None = None) -> list[int]:
    """The length of each text in the embedding model's own tokens, special tokens included: what `count_too_long` compares with `token_limit`, and what the
    chunker uses to cut a chunk that is too long for the model (design choices entry 131). No text: an empty list (the tokenizer raises on one)."""
    if not texts:   # a file with no article (a symbol whose every fetch came back empty): the tokenizer raises IndexError on an empty list
        return []
    model = (embeddings or get_embeddings())._client
    return [len(ids) for ids in model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]


def main(argv: list[str] | None = None) -> int:
    """Print what the vector store holds: python -m prometheus.rag.store [--press-releases | --filings]"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--press-releases", action="store_true", help="show the press release collection instead of the news one")
    which.add_argument("--filings", action="store_true", help="show the filings collection instead of the news one")
    # No argv means "no options" (as main() always meant): the command line below passes sys.argv itself. Parsing sys.argv here would read pytest's arguments.
    args = parser.parse_args([] if argv is None else argv)
    kind = "press_releases" if args.press_releases else "filings" if args.filings else "news"
    path, name = vector_store_path(), collection_name(kind=kind)
    print(f"folder     : {path}")
    print(f"collection : {name}")
    if not path.exists() or not count_chunks(kind):
        print("nothing is stored yet. Ingest some " + {"press_releases": "press releases first:  python -m prometheus.rag.ingest --press-releases AAPL",
                                                       "filings": "filings first:  python -m prometheus.rag.ingest --filings AAPL",
                                                       "news": "news first:  python -m prometheus.rag.ingest AAPL"}[kind])
        return 0
    data = _client().get_collection(name).get(include=["documents", "metadatas"])
    texts, metas = data["documents"], data["metadatas"]
    embeddings = get_embeddings()
    model = embeddings._client
    print(f"model      : {embedding_model()} on {embedding_device()}, {len(embeddings.embed_query('x'))} numbers per vector, "
          f"reads at most {model.max_seq_length} tokens")

    per_article = Counter(m["article_id"] for m in metas)
    sizes = Counter(per_article.values())
    print(f"stored     : {len(texts)} chunks of {len(per_article)} articles | chunks per article: "
          + ", ".join(f"{n} chunk{'s' if n > 1 else ''} x {sizes[n]}" for n in sorted(sizes)))
    lengths = [len(t) for t in texts]
    print(f"text length: median {int(statistics.median(lengths))}, longest {max(lengths)} characters | "
          f"over the embedder's limit: {count_too_long(texts, embeddings)} of {len(texts)}")
    per_ticker = Counter(t for m in metas for t in m["tickers"])
    print("tickers    : " + ", ".join(f"{t} {n}" for t, n in per_ticker.most_common(8)) + "  (chunks per ticker)")

    def show(label: str, index: int) -> None:
        meta = metas[index]
        print(f"\n--- {label} ---")
        print(f"id: {ids[index]}  |  chunk {meta['chunk'] + 1} of {meta['chunks']}  |  tickers: {', '.join(meta['tickers'])}")
        preview = texts[index] if len(texts[index]) <= 260 else texts[index][:260] + " ..."
        print("  " + preview.replace("\n", "\n  "))

    ids = data["ids"]
    first = next(i for i, m in enumerate(metas) if m["chunk"] == 0)
    later = next((i for i, m in enumerate(metas) if m["chunk"] > 0), None)
    several = next((i for i, m in enumerate(metas) if len(m["tickers"]) > 1), None)
    show("a first chunk", first)
    if later is not None:
        show("a later chunk (the headline is put back in front)", later)
    if several is not None:
        show("a chunk of an article with several tickers", several)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
