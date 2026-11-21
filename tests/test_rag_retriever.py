"""Tests for prometheus.rag.retriever (Part 5 of the RAG guide).

Run:  pytest tests/test_rag_retriever.py -q

Most tests use a tiny FAKE embedder with one dimension per word, so we know exactly which chunks are alike and
what every score is. They check the behaviour that does not depend on a real model: filters that act before the
ranking, one hit per article, the relevance bar, the date window, and the MMR option. The tests at the end use
the real MiniLM model on the three real articles in tests/fixtures; they skip themselves when the model is not
available. No test touches the real index in data/processed/vectorstore: every store lives in a temporary folder.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from prometheus.rag import retriever, store as rag_store
from prometheus.rag.loaders import load_news
from prometheus.rag.retriever import Hit, search_news

FIXTURE = Path(__file__).parent / "fixtures" / "news_sample.txt"
DAY = timedelta(days=1)
# The test clock follows the real one: yesterday at noon UTC. The tools count windows (since_days) back from the real clock, so a clock pinned to a past day made those tests fail as the days went by.
# The synthetic chunks are dated NOW minus some days; a test that needs a fixed date (the real stored sample above) builds its own.
NOW = (datetime.now(timezone.utc) - DAY).replace(hour=12, minute=0, second=0, microsecond=0)


class KeywordEmbeddings(Embeddings):
    """A stand-in for the real model: one dimension per word, so the similarity of two texts is easy to work out."""
    WORDS = ["apple", "iphone", "phone", "nvidia", "chip", "tesla", "car", "meta", "tax", "cake"]

    def _vector(self, text: str) -> list[float]:
        words = text.lower().split()
        counts = [float(words.count(w)) for w in self.WORDS] + [0.01]  # the last number keeps the vector from being all zero
        length = sum(c * c for c in counts) ** 0.5
        return [c / length for c in counts]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def make_store(path: Path, embeddings) -> Chroma:
    return Chroma(
        client=chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False)),
        collection_name="test",
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def chunk(article_id, number, chunks, text, tickers, when, publisher="Yahoo") -> Document:
    """One stored chunk, with the metadata that ingest gives it."""
    return Document(id=f"{article_id}#{number}", page_content=text, metadata={
        "article_id": article_id, "chunk": number, "chunks": chunks, "headline": f"Headline of {article_id}",
        "publisher": publisher, "published_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "published_ts": int(when.timestamp()),
        "url": f"https://example.com/{article_id}", "tickers": list(tickers), "api": "test", "start_index": 0})


def fill(store: Chroma, docs: list[Document]) -> Chroma:
    store.add_documents(docs, ids=[d.id for d in docs])
    return store


@pytest.fixture
def news(tmp_path) -> Chroma:
    """Eight chunks of seven articles. n2 has two chunks, and a2 is a near-duplicate of a1."""
    return fill(make_store(tmp_path / "chroma", KeywordEmbeddings()), [
        chunk("a1", 0, 1, "apple iphone phone", ["AAPL"], NOW - 5 * DAY),
        chunk("a2", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - 4 * DAY),
        chunk("a3", 0, 1, "apple nvidia chip", ["AAPL", "NVDA"], NOW - 3 * DAY),
        chunk("n1", 0, 1, "nvidia chip chip chip", ["NVDA"], NOW - 3 * DAY),
        chunk("n2", 0, 2, "nvidia chip chip", ["NVDA"], NOW - 2 * DAY),
        chunk("n2", 1, 2, "nvidia chip", ["NVDA"], NOW - 2 * DAY),
        chunk("t1", 0, 1, "tesla car", ["TSLA"], NOW - 1 * DAY),
        chunk("m1", 0, 1, "meta tax", ["META"], NOW - 1 * DAY),
    ])


def ids(hits: list[Hit]) -> list[str]:
    return [h.article_id for h in hits]


# ---------------------------------------------------------------- newest first (found in the SNDK run: "latest" returned old articles)

@pytest.fixture
def dated(tmp_path):
    """Four apple articles of different ages and different similarity to "apple iphone", and one old non-apple article."""
    return fill(make_store(tmp_path / "dated", KeywordEmbeddings()), [
        chunk("old-best", 0, 1, "apple iphone phone", ["AAPL"], NOW - 6 * DAY),            # the closest match, the oldest
        chunk("mid-good", 0, 1, "apple iphone phone phone chip", ["AAPL"], NOW - 3 * DAY),
        chunk("new-ok", 0, 1, "apple iphone chip chip", ["AAPL"], NOW - 1 * DAY),
        chunk("newest-ok", 0, 1, "apple iphone chip chip chip", ["AAPL"], NOW - 1 * DAY + timedelta(hours=5)),
        chunk("new-offtopic", 0, 1, "tesla car", ["AAPL"], NOW - 1 * DAY + timedelta(hours=9))])  # the newest of all, but not about apple


# ---------------------------------------------------------------- diverse (MMR)


def test_the_bar_applies_before_the_spreading_so_variety_cannot_pull_in_noise(news):
    # Everything except a1 and a2 is below 0.5. With the most variety (lambda 0) an MMR that ran first would
    # reach for the unrelated articles; the bar is applied first, so only a1 and a2 are candidates.
    assert ids(search_news("apple iphone phone", min_score=0.5, k=2, diverse=True, lambda_mult=0.0, store=news)) == ["a1", "a2"]
    noisy = search_news("apple iphone phone", min_score=None, k=2, diverse=True, lambda_mult=0.0, store=news)
    assert ids(noisy)[0] == "a1" and ids(noisy)[1] not in {"a2"}  # without the bar it does reach for something else


# ---------------------------------------------------------------- bad input


@pytest.mark.parametrize("bar", [1.5, -0.1, 35])
def test_the_bar_must_be_between_0_and_1(news, bar):
    with pytest.raises(ValueError, match="min_score"):
        search_news("apple", min_score=bar, store=news)


def test_lambda_must_be_between_0_and_1(news):
    with pytest.raises(ValueError, match="lambda_mult"):
        search_news("apple", diverse=True, lambda_mult=2, store=news)


@pytest.mark.parametrize("tickers", [["AAPL", "../../.env"], "$AAPL", "", "WAYTOOLONGSYMBOL", [1], [None]])
def test_a_bad_ticker_is_refused_before_anything_is_searched(news, tickers):
    with pytest.raises(ValueError, match="ticker"):
        search_news("apple", tickers=tickers, store=news)


# ---------------------------------------------------------------- the command line

@pytest.fixture
def cli(news, monkeypatch, capsys):
    monkeypatch.setattr(retriever, "get_vector_store", lambda kind="news": news)  # `kind` since press releases got their own collection (entry 34)
    monkeypatch.setattr(retriever, "count_chunks", lambda kind="news": 8)
    return capsys


def test_cli_can_print_the_text_and_use_the_options(cli):
    assert retriever.main(["nvidia chip", "--text", "--diverse", "--since-days", "30", "--min-score", "0.1"]) == 0
    out = cli.readouterr().out
    assert "last 30 days" in out and "diverse yes" in out and "nvidia chip chip chip" in out


def test_cli_says_no_supporting_text_and_shows_how_close_the_best_chunk_was(cli):
    assert retriever.main(["cake"]) == 0
    out = cli.readouterr().out
    assert "no supporting text: nothing scored 0.4 or more." in out
    assert "closest was 0.0" in out and "Headline of" in out


def test_cli_says_when_the_filters_leave_nothing_to_search(cli):
    assert retriever.main(["apple", "--tickers", "ZZZ"]) == 0
    assert "no supporting text: no article matches the filters at all." in cli.readouterr().out


def test_cli_reports_a_bad_ticker_without_a_traceback(cli):
    assert retriever.main(["apple", "--tickers", "../../.env"]) == 1
    captured = cli.readouterr()
    assert "ERROR: not a valid ticker symbol" in captured.err and "Traceback" not in captured.err


def test_cli_tells_you_to_ingest_when_the_index_is_empty(news, monkeypatch, capsys):
    monkeypatch.setattr(retriever, "get_vector_store", lambda kind="news": news)
    monkeypatch.setattr(retriever, "count_chunks", lambda kind="news": 0)
    assert retriever.main(["apple"]) == 1
    assert "nothing is stored yet" in capsys.readouterr().err
