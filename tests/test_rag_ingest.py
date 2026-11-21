"""Tests for prometheus.rag.store and prometheus.rag.ingest (Part 4 of the RAG guide).

Run:  pytest tests/test_rag_ingest.py -q

Most tests use a FAKE embedder (fast, no model): they check the bookkeeping, which does not depend on
what the vectors mean. The tests at the end use the real MiniLM model to prove that the stored chunks can
really be found by meaning; they skip themselves when the model is not available. No test touches the real
index in data/processed/vectorstore: every store lives in a temporary folder.
"""
import shutil
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from prometheus.rag import config, ingest, store as rag_store
from prometheus.rag.ingest import ingest_documents, to_chunks
from prometheus.rag.loaders import load_news

FIXTURE = Path(__file__).parent / "fixtures" / "news_sample.txt"
HEADLINE = "The headline"


def make_store(path: Path, embeddings=None) -> Chroma:
    return Chroma(
        client=chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False)),
        collection_name="test",
        embedding_function=embeddings or DeterministicFakeEmbedding(size=16),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def article(article_id="a:1", body="A short summary.", tickers=("AAPL",)) -> Document:
    return Document(id=article_id, page_content=f"{HEADLINE}\n\n{body}", metadata={
        "article_id": article_id, "headline": HEADLINE, "tickers": list(tickers),
        "published_utc": "2026-10-23T19:03:00Z", "published_ts": 1792782180,
        "publisher": "Yahoo", "api": "finnhub", "url": f"https://example.com/{article_id}"})


LONG_BODY = " ".join(f"Sentence number {i} says something about the company." for i in range(60))  # about 3,400 characters


def held(store: Chroma) -> dict[str, tuple[str, dict]]:
    got = store.get(include=["documents", "metadatas"])
    return dict(zip(got["ids"], zip(got["documents"], got["metadatas"])))


@pytest.fixture
def store(tmp_path) -> Chroma:
    return make_store(tmp_path / "chroma")


# ---------------------------------------------------------------- running again


def test_another_article_is_left_alone(store):
    ingest_documents([article("a:1", LONG_BODY), article("b:2", LONG_BODY)], store)
    before = {k: v for k, v in held(store).items() if k.startswith("b:2")}
    stats = ingest_documents([article("a:1", "changed")], store)
    assert stats["unchanged"] == 0 and stats["deleted"] > 0
    assert {k: v for k, v in held(store).items() if k.startswith("b:2")} == before


# ---------------------------------------------------------------- the same article in two files


def test_the_same_article_twice_in_one_call_is_stored_once_with_merged_tickers(store):
    stats = ingest_documents([article(tickers=["AAPL"]), article(tickers=["NVDA"])], store)
    assert stats["articles"] == 1 and stats["added"] == 1
    assert held(store)["a:1#0"][1]["tickers"] == ["AAPL", "NVDA"]


# ---------------------------------------------------------------- the command line (a fake store, no model)

@pytest.fixture
def cli(monkeypatch, tmp_path, store):
    news = tmp_path / "news"
    news.mkdir()
    shutil.copy(FIXTURE, news / "AAPL.txt")
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(news))
    monkeypatch.setattr(ingest, "get_vector_store", lambda kind="news": store)  # `kind` since press releases got their own collection (entry 34)
    monkeypatch.setattr(ingest, "count_too_long", lambda texts: 0)
    monkeypatch.setattr(ingest, "count_chunks", lambda kind="news": len(store.get()["ids"]))
    return news


def test_cli_warns_loudly_when_chunks_are_over_the_embedder_limit(cli, monkeypatch, capsys):
    assert ingest.main(["AAPL"]) == 0  # nothing over the limit: no warning
    assert "WARNING" not in capsys.readouterr().err
    monkeypatch.setattr(ingest, "count_too_long", lambda texts: 2)
    assert ingest.main(["AAPL"]) == 0
    captured = capsys.readouterr()
    assert "chunks over the embedder's limit: 2" in captured.out
    assert "WARNING: AAPL: 2 chunks are longer than the embedder reads" in captured.err


def test_cli_reports_problems_instead_of_a_traceback(cli, capsys):
    assert ingest.main(["NOT A SYMBOL"]) == 1
    assert ingest.main(["MSFT"]) == 1  # no news file for it
    (cli / "AAPL.txt").unlink()
    assert ingest.main(["--all"]) == 1  # no news files at all
    err = capsys.readouterr().err
    assert err.count("ERROR") == 3 and "not a valid ticker symbol" in err and "no news files" in err


def test_cli_needs_symbols_or_all():
    with pytest.raises(SystemExit):
        ingest.main([])
    with pytest.raises(SystemExit):
        ingest.main(["AAPL", "--all"])


# ---------------------------------------------------------------- the real embedding model (skipped when unavailable)

@pytest.fixture(scope="module")
def real_embeddings():
    try:
        return rag_store.get_embeddings()
    except Exception as error:  # no local copy of the model and no network
        pytest.skip(f"the embedding model is not available: {type(error).__name__}")


def test_real_chunks_of_every_news_file_fit_the_embedder_limit(real_embeddings):
    files = sorted(config.news_dir().glob("*.txt"))
    if not files:
        pytest.skip(f"no news files in {config.news_dir()}")
    chunks = to_chunks([doc for file in files for doc in load_news(file)])
    assert rag_store.count_too_long([c.page_content for c in chunks], real_embeddings) == 0
    assert rag_store.count_too_long(["word " * 400], real_embeddings) == 1  # and the check can see a text that is too long


def test_real_chunks_of_every_filing_fit_the_embedder_limit_once_cut_by_tokens_and_lose_no_text(real_embeddings):
    """Design choices entry 131. A filing's tables have two or three times the tokens per character of prose: by characters alone thousands of the real
    filing chunks are over the embedder's limit (the part past it is not searchable). Cut again by the model's own token count, none is, every article's
    text is still all there (each chunk's body is the article's text from its `start_index`), and a chunk that already fitted is the same chunk."""
    from prometheus.rag.loaders import load_filings
    files = sorted(config.filings_dir().glob("*.txt"))
    if not files:
        pytest.skip(f"no filings in {config.filings_dir()}")
    docs = list({doc.id: doc for file in reversed(files) for doc in reversed(load_filings(file))}.values())   # one text per article id, as `ingest_documents` keeps
    lengths = lambda texts: rag_store.token_lengths(texts, real_embeddings)   # noqa: E731
    limit = rag_store.token_limit(real_embeddings)
    by_characters = to_chunks(docs)
    fitted = to_chunks(docs, lengths, limit)
    assert rag_store.count_too_long([c.page_content for c in by_characters], real_embeddings) > 1000     # the problem is real
    assert rag_store.count_too_long([c.page_content for c in fitted], real_embeddings) == 0
    text_of = {doc.id: doc.page_content for doc in docs}

    def left_out(chunks) -> set[tuple[str, str]]:
        """(article id, text) of everything that is in an article and in none of its chunks; each chunk must be the article's own text, where it says it is."""
        covered: dict[str, int] = {}
        missing = set()
        for chunk in chunks:
            article, start = text_of[chunk.metadata["article_id"]], chunk.metadata["start_index"]
            body = chunk.page_content if article.startswith(chunk.page_content, start) else chunk.page_content.split("\n\n", 1)[1]    # with the headline put in front, or without
            assert article[start:start + len(body)] == body
            done = covered.get(chunk.metadata["article_id"], 0)
            if start > done and article[done:start].strip():
                missing.add((chunk.metadata["article_id"], article[done:start].strip()))
            covered[chunk.metadata["article_id"]] = max(done, start + len(body))
        assert all(len(text_of[article].rstrip()) <= end for article, end in covered.items())       # each article is covered to its end
        return missing

    # The character splitter itself leaves out one lone page number ("26", between two tables of one 10-Q): found by this test, older than it. The cut by tokens loses nothing more.
    assert left_out(fitted) == left_out(by_characters) and len(left_out(by_characters)) <= 1
    kept = {c.page_content for c in fitted}
    fitting = [c.page_content for c, n in zip(by_characters, lengths([c.page_content for c in by_characters])) if n <= limit]
    assert all(text in kept for text in fitting)                                    # what already fitted is untouched


def test_the_real_entry_points_end_to_end(monkeypatch, tmp_path, real_embeddings, capsys):
    news = tmp_path / "news"
    news.mkdir()
    shutil.copy(FIXTURE, news / "AAPL.txt")
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(news))
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "index"))
    assert rag_store.count_chunks() == 0 and rag_store.reset_collection() is False
    assert ingest.main(["AAPL"]) == 0
    assert "added 5" in capsys.readouterr().out and rag_store.count_chunks() == 5
    client = chromadb.PersistentClient(path=str(tmp_path / "index"), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(rag_store.collection_name())
    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert rag_store._client().get_settings().anonymized_telemetry is False  # the client the code really uses
    assert ingest.main(["--rebuild", "AAPL"]) == 0
    assert "deleted the collection" in capsys.readouterr().out and rag_store.count_chunks() == 5
    assert rag_store.main() == 0
    out = capsys.readouterr().out
    assert "stored     : 5 chunks of 3 articles" in out and "over the embedder's limit: 0 of 5" in out
    assert "reads at most 256 tokens" in out  # also proves the one private attribute we rely on still exists
