"""Tests for prometheus.rag.citations (Part 6 of the RAG guide).

Run:  pytest tests/test_rag_citations.py -q

No model, no index, no network: a Hit is plain data, so every case is built by hand.
"""
import pytest

from prometheus.rag.citations import SourceBook
from prometheus.rag.retriever import Hit


def hit(n: int, text: str = "Some text.", tickers=("AAPL",), publisher="Yahoo", score=0.73) -> Hit:
    return Hit(chunk_id=f"a{n}#0", article_id=f"a{n}", chunk=1, chunks=1, headline=f"Headline {n}", publisher=publisher,
               published_utc="2026-10-23T19:03:00Z", url=f"https://example.com/{n}", tickers=list(tickers), text=text, score=score)


# ---------------------------------------------------------------- the check

@pytest.fixture
def book() -> SourceBook:
    b = SourceBook()
    b.add([hit(1), hit(2), hit(3)])
    return b


def test_a_number_with_no_source_is_reported_once(book):
    result = book.check("a [0] b [4] c [4] d [99, 1]")
    assert not result.ok and result.invalid == [0, 4, 99] and list(result.cited) == [1]
    assert result.problems() == ["cites [0], [4], [99], which are not among the sources"]


def test_problems_lists_both_kinds(book):
    assert book.check("only [7]").problems(expect_citations=True) == ["cites [7], which are not among the sources"]  # it did cite something
    assert SourceBook().check("nothing").problems(expect_citations=True) == ["cites no source although sources were retrieved"]


