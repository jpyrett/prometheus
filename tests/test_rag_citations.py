"""Tests for prometheus.rag.citations (Part 6 of the RAG guide).

Run:  pytest tests/test_rag_citations.py -q

No model, no index, no network: a Hit is plain data, so every case is built by hand.
"""
import pytest

from prometheus.rag.citations import HEADER, SourceBook, format_source
from prometheus.rag.retriever import Hit


def hit(n: int, text: str = "Some text.", tickers=("AAPL",), publisher="Yahoo", score=0.73) -> Hit:
    return Hit(chunk_id=f"a{n}#0", article_id=f"a{n}", chunk=1, chunks=1, headline=f"Headline {n}", publisher=publisher,
               published_utc="2026-10-23T19:03:00Z", url=f"https://example.com/{n}", tickers=list(tickers), text=text, score=score)


# ---------------------------------------------------------------- numbering


# ---------------------------------------------------------------- the check

@pytest.fixture
def book() -> SourceBook:
    b = SourceBook()
    b.add([hit(1), hit(2), hit(3)])
    return b


def test_a_valid_citation_points_at_its_chunk(book):
    result = book.check("Apple launched a phone [2].")
    assert result.ok and result.has_citations and result.invalid == []
    assert list(result.cited) == [2] and result.cited[2].chunk_id == "a2#0"
    assert [h.chunk_id for h in result.uncited] == ["a1#0", "a3#0"]


def test_a_number_with_no_source_is_reported_once(book):
    result = book.check("a [0] b [4] c [4] d [99, 1]")
    assert not result.ok and result.invalid == [0, 4, 99] and list(result.cited) == [1]
    assert result.problems() == ["cites [0], [4], [99], which are not among the sources"]


def test_no_citations_is_ok_unless_sources_were_expected(book):
    result = book.check("No supporting text was found for that question.")
    assert result.ok and not result.has_citations and result.problems() == []
    assert result.problems(expect_citations=True) == ["cites no source although sources were retrieved"]
    assert len(result.uncited) == 3


def test_problems_lists_both_kinds(book):
    assert book.check("only [7]").problems(expect_citations=True) == ["cites [7], which are not among the sources"]  # it did cite something
    assert SourceBook().check("nothing").problems(expect_citations=True) == ["cites no source although sources were retrieved"]


def test_numbers_cited_from_a_second_search_resolve_to_the_right_chunks():
    book = SourceBook()
    book.add([hit(1), hit(2)])
    book.add([hit(3), hit(2)])
    result = book.check("a [1] b [2] c [3]")
    assert [h.chunk_id for h in result.cited.values()] == ["a1#0", "a2#0", "a3#0"] and result.ok
