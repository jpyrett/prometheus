"""The embedding model and the Chroma collection ("Encode chunks" and "Store vectors + text" in the slide).

This is the only module that names the embedding model class or the vector database. Everything else
asks it for `get_vector_store()`, so changing either one is a change here and in config.py.

Look at what is stored:   python -m prometheus.rag.store
"""
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


def collection_name(model: str | None = None) -> str:
    """news-<model>. The model's name is part of the collection's name, so vectors from two different
    models never share a collection (they cannot be compared with each other).
    Chroma allows only letters, digits, '.', '_' and '-', so anything else (the '/') becomes '-'."""
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", model or embedding_model()).strip("-._")
    return f"news-{clean}"


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(vector_store_path()), settings=Settings(anonymized_telemetry=False))


def get_vector_store() -> Chroma:
    """The Chroma collection for news: a folder on disk, telemetry off, cosine distance (fixed when the
    collection is created, design choices entry 10)."""
    return Chroma(
        client=_client(),
        collection_name=collection_name(),
        embedding_function=get_embeddings(),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def reset_collection() -> bool:
    """Delete the news collection. The index is derived data: ingest again to rebuild it.
    Returns whether there was a collection to delete."""
    try:
        _client().delete_collection(collection_name())
        return True
    except NotFoundError:
        return False


def count_chunks() -> int:
    """How many chunks the news collection holds (0 if there is none yet)."""
    try:
        return _client().get_collection(collection_name()).count()
    except NotFoundError:
        return 0


def count_too_long(texts: list[str], embeddings: HuggingFaceEmbeddings | None = None) -> int:
    """How many of these texts are longer than the embedding model reads. The model silently ignores
    everything past its limit (256 tokens for MiniLM), so a text over the limit is only partly searchable.

    LangChain keeps the real model in `_client`. This function is the only place that touches it."""
    model = (embeddings or get_embeddings())._client
    lengths = model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    return sum(len(ids) > model.max_seq_length for ids in lengths)


def main() -> int:
    """Print what the vector store holds: python -m prometheus.rag.store"""
    path, name = vector_store_path(), collection_name()
    print(f"folder     : {path}")
    print(f"collection : {name}")
    if not path.exists() or not count_chunks():
        print("nothing is stored yet. Ingest some news first:  python -m prometheus.rag.ingest AAPL")
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
    sys.exit(main())
