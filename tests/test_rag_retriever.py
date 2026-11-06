"""Tests for prometheus.rag.retriever (Part 5 of the RAG guide).

Run:  pytest tests/test_rag_retriever.py -q

Most tests use a tiny FAKE embedder with one dimension per word, so we know exactly which chunks are alike and
what every score is. They check the behaviour that does not depend on a real model: filters that act before the
ranking, one hit per article, the relevance bar, the date window, and the MMR option. The tests at the end use
the real MiniLM model on the three real articles in tests/fixtures; they skip themselves when the model is not
available. No test touches the real index in data/processed/vectorstore: every store lives in a temporary folder.
"""
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from prometheus.rag import retriever, store as rag_store
from prometheus.rag.ingest import ingest_documents
from prometheus.rag.loaders import load_news
from prometheus.rag.retriever import MAX_K, MIN_SCORE, Hit, search_news

FIXTURE = Path(__file__).parent / "fixtures" / "news_sample.txt"
COREWEAVE = "polygon:287f3b5c2f0c59c2b903cd391bb97c0c090f48a25c64cdf5e7df1dab0e85382a"
DAY = timedelta(days=1)
# The test clock follows the real one: yesterday at noon UTC. The tools count windows (since_days) back from the real clock, so a clock pinned to a past day made those tests fail as the days went by.
# The synthetic chunks are dated NOW minus some days; a test that needs a fixed date (the real stored sample above) builds its own.
NOW = (datetime.now(timezone.utc) - DAY).replace(hour=12, minute=0, second=0, microsecond=0)
STAMP = "%Y-%m-%dT%H:%M:%SZ"


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


class HotCold(Embeddings):
    """Three fixed vectors, so that one pair of texts points in exactly opposite directions (cosine similarity -1)."""
    VECTORS = {"hot": [1.0, 0.0], "warm": [0.6, 0.8], "cold": [-1.0, 0.0]}

    def embed_documents(self, texts):
        return [self.VECTORS[t.split()[0]] for t in texts]

    def embed_query(self, text):
        return self.VECTORS[text.split()[0]]


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


# ---------------------------------------------------------------- what a hit is


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


def test_by_default_the_best_match_comes_first_even_if_it_is_old(dated):
    hits = search_news("apple iphone", tickers="AAPL", k=2, store=dated, now=NOW)
    assert [h.article_id for h in hits][0] == "old-best"


def test_newest_first_returns_the_newest_relevant_articles_and_not_the_best_matching_ones(dated):
    hits = search_news("apple iphone", tickers="AAPL", k=2, newest_first=True, min_score=0.3, store=dated, now=NOW)
    assert [h.article_id for h in hits] == ["newest-ok", "new-ok"]  # newest first, and the older, closer matches are cut by k


def test_newest_first_still_applies_the_relevance_bar_so_a_new_article_about_something_else_is_not_returned(dated):
    hits = search_news("apple iphone", tickers="AAPL", k=5, newest_first=True, store=dated, now=NOW)
    assert "new-offtopic" not in [h.article_id for h in hits]
    dates = [h.published_utc for h in hits]
    assert dates == sorted(dates, reverse=True) and len(hits) >= 2  # the whole list is in date order


def test_newest_first_combines_with_the_date_window(dated):
    hits = search_news("apple iphone", tickers="AAPL", k=5, since_days=2, newest_first=True, min_score=0.3, store=dated, now=NOW)
    assert [h.article_id for h in hits] == ["newest-ok", "new-ok"]


def test_a_tie_in_date_goes_to_the_better_match(tmp_path):
    same_day = fill(make_store(tmp_path / "tie", KeywordEmbeddings()), [
        chunk("weaker", 0, 1, "apple iphone chip chip", ["AAPL"], NOW - DAY),
        chunk("closer", 0, 1, "apple iphone phone", ["AAPL"], NOW - DAY)])
    assert [h.article_id for h in search_news("apple iphone", tickers="AAPL", newest_first=True, min_score=0.3, store=same_day, now=NOW)] == ["closer", "weaker"]


def test_newest_first_wins_over_diverse(dated):
    hits = search_news("apple iphone", tickers="AAPL", k=2, newest_first=True, diverse=True, min_score=0.3, store=dated, now=NOW)
    assert [h.article_id for h in hits] == ["newest-ok", "new-ok"]


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_newest_first_must_be_true_or_false(dated, value):
    with pytest.raises(ValueError, match="newest_first must be true or false"):
        search_news("apple iphone", newest_first=value, store=dated, now=NOW)

def test_hits_come_best_first_and_carry_what_a_citation_needs(news):
    hits = search_news("apple iphone phone", min_score=None, k=3, store=news)
    assert ids(hits) == ["a1", "a2", "a3"]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    best = hits[0]
    assert best.score == pytest.approx(1.0, abs=1e-3)  # the same words: cosine similarity 1
    assert (best.chunk_id, best.article_id, best.chunk, best.chunks) == ("a1#0", "a1", 1, 1)
    assert (best.headline, best.publisher, best.url) == ("Headline of a1", "Yahoo", "https://example.com/a1")
    assert (best.published_utc, best.tickers, best.text) == ((NOW - 5 * DAY).strftime(STAMP), ["AAPL"], "apple iphone phone")
    assert best.as_dict() == {
        "chunk_id": "a1#0", "article_id": "a1", "chunk": 1, "chunks": 1, "headline": "Headline of a1",
        "publisher": "Yahoo", "published_utc": (NOW - 5 * DAY).strftime(STAMP), "url": "https://example.com/a1",
        "tickers": ["AAPL"], "text": "apple iphone phone", "score": best.score}
    with pytest.raises(AttributeError):
        best.score = 0.0  # a Hit is frozen: the evidence cannot be edited after the search


def test_one_hit_per_article_and_k_counts_articles(news):
    hits = search_news("nvidia chip", min_score=None, k=5, store=news)
    assert ids(hits).count("n2") == 1  # n2 has two chunks that both match, but it appears once
    assert len(set(ids(hits))) == len(hits)
    n2 = next(h for h in hits if h.article_id == "n2")
    assert (n2.chunk_id, n2.chunk, n2.chunks, n2.text) == ("n2#1", 2, 2, "nvidia chip")  # its BEST chunk, not its first
    assert ids(hits)[:3] == ["n2", "n1", "a3"]
    assert ids(search_news("nvidia chip", min_score=None, k=2, store=news)) == ["n2", "n1"]


def test_a_filter_that_matches_nothing_gives_no_hits(news):
    assert search_news("apple", tickers="ZZZ", min_score=None, store=news) == []


# ---------------------------------------------------------------- filters

def test_the_ticker_filter_takes_one_symbol_or_a_list_in_any_case(news):
    assert set(ids(search_news("apple", tickers="aapl", min_score=None, store=news))) == {"a1", "a2", "a3"}
    assert set(ids(search_news("apple", tickers=["NVDA"], min_score=None, store=news))) == {"a3", "n1", "n2"}  # a3 names both
    assert set(ids(search_news("apple", tickers=[" nvda ", "tsla"], min_score=None, store=news))) == {"a3", "n1", "n2", "t1"}


def test_the_filter_acts_before_the_ranking_so_it_never_leaves_fewer_hits_than_exist(news):
    # "tesla car" is closest to t1, which the filter excludes. Filtering AFTER the search would drop it and lose
    # the rest of the top results; filtering inside the search still returns every article that passes.
    hits = search_news("tesla car", tickers="AAPL", min_score=None, k=5, store=news)
    assert set(ids(hits)) == {"a1", "a2", "a3"}


def test_the_filters_combine(news):
    # the NVDA articles are n1 and a3 (3 days old), n2 (2 days old). Only n2 is newer than 2.5 days.
    assert ids(search_news("nvidia chip", tickers="NVDA", since_days=3, min_score=None, store=news, now=NOW)) == ["n2", "n1", "a3"]
    assert ids(search_news("nvidia chip", tickers="NVDA", since_days=2, min_score=None, store=news, now=NOW)) == ["n2"]


def test_the_date_window_counts_back_from_the_clock_it_is_given_and_includes_the_edge(tmp_path):
    edge, just_outside = NOW - 2 * DAY, NOW - 2 * DAY - timedelta(seconds=1)
    store = fill(make_store(tmp_path / "window", KeywordEmbeddings()), [
        chunk("edge", 0, 1, "apple iphone", ["AAPL"], edge),
        chunk("late", 0, 1, "apple iphone phone", ["AAPL"], just_outside),
        chunk("new", 0, 1, "apple iphone phone phone", ["AAPL"], NOW - timedelta(hours=1))])
    assert set(ids(search_news("apple iphone", since_days=2, min_score=None, store=store, now=NOW))) == {"edge", "new"}
    assert set(ids(search_news("apple iphone", since_days=1, min_score=None, store=store, now=NOW))) == {"new"}
    # a clock a day later moves the window a day later: "the last 2 days" is relative, not a fixed date
    assert set(ids(search_news("apple iphone", since_days=2, min_score=None, store=store, now=NOW + DAY))) == {"new"}
    assert set(ids(search_news("apple iphone", since_days=2, min_score=None, store=store, now=NOW + 4 * DAY))) == set()
    # and without a clock it uses the real one
    assert len(search_news("apple iphone", since_days=36500, min_score=None, store=store)) == 3


# ---------------------------------------------------------------- scores and "no supporting text"

def test_nothing_relevant_means_an_empty_list(news):
    assert search_news("cake", store=news) == []  # no stored text has anything in common with it
    weak = search_news("cake", min_score=None, store=news)
    assert weak and max(h.score for h in weak) < MIN_SCORE  # the hits exist, but none clears the bar


def test_the_bar_keeps_a_hit_that_scores_exactly_the_bar(news):
    score = {h.article_id: h.score for h in search_news("apple iphone phone", min_score=None, k=3, store=news)}
    assert 0.5 < score["a2"] < 0.99  # a hit whose score is neither 0 nor 1, so a hair either way matters
    assert "a2" in ids(search_news("apple iphone phone", min_score=score["a2"], k=3, store=news))  # exactly the bar: kept
    assert "a2" not in ids(search_news("apple iphone phone", min_score=score["a2"] + 1e-6, k=3, store=news))  # a hair above: dropped
    assert "a1" in ids(search_news("apple iphone phone", min_score=score["a2"] + 1e-6, k=3, store=news))  # the better hit stays


def test_the_default_bar_drops_weak_hits_and_keeps_strong_ones(news):
    hits = search_news("apple iphone", store=news)  # a1 0.82, a2 0.58, a3 0.41 (just above the bar), the rest below
    assert ids(hits) == ["a1", "a2", "a3"]
    assert all(h.score >= MIN_SCORE for h in hits)
    assert ids(search_news("apple iphone", min_score=0.5, store=news)) == ["a1", "a2"]


def test_the_score_is_the_cosine_similarity_that_langchain_reports(news):
    hits = search_news("nvidia chip", min_score=None, k=5, store=news)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = {d.id: s for d, s in news.similarity_search_with_relevance_scores("nvidia chip", k=50)}
    for hit in hits:
        assert hit.score == pytest.approx(theirs[hit.chunk_id], abs=1e-6)


def test_a_negative_score_raises_no_warning(tmp_path):
    # LangChain's own scored search warns, with a message that holds every document, when a score is below 0.
    store = fill(make_store(tmp_path / "hotcold", HotCold()), [
        chunk("w", 0, 1, "warm", ["AAPL"], NOW), chunk("c", 0, 1, "cold", ["AAPL"], NOW)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hits = search_news("hot", min_score=None, k=2, store=store)
    assert ids(hits) == ["w", "c"]
    assert hits[1].score == pytest.approx(-1.0, abs=1e-3)
    assert not [w for w in caught if "Relevance scores" in str(w.message)]
    assert search_news("hot", k=2, store=store)[0].article_id == "w"  # and the default bar cuts the opposite one


# ---------------------------------------------------------------- diverse (MMR)

def test_diverse_picks_another_story_instead_of_the_near_duplicate(news):
    plain = search_news("apple iphone phone", min_score=0.2, k=2, store=news)
    spread = search_news("apple iphone phone", min_score=0.2, k=2, diverse=True, lambda_mult=0.3, store=news)
    assert ids(plain) == ["a1", "a2"]  # a2 is almost the same story as a1
    assert ids(spread) == ["a1", "a3"]  # the best match first, then the one least like it that is still relevant
    assert ids(search_news("apple iphone phone", min_score=0.2, k=2, diverse=True, lambda_mult=1.0, store=news)) == ["a1", "a2"]


def test_the_bar_applies_before_the_spreading_so_variety_cannot_pull_in_noise(news):
    # Everything except a1 and a2 is below 0.5. With the most variety (lambda 0) an MMR that ran first would
    # reach for the unrelated articles; the bar is applied first, so only a1 and a2 are candidates.
    assert ids(search_news("apple iphone phone", min_score=0.5, k=2, diverse=True, lambda_mult=0.0, store=news)) == ["a1", "a2"]
    noisy = search_news("apple iphone phone", min_score=None, k=2, diverse=True, lambda_mult=0.0, store=news)
    assert ids(noisy)[0] == "a1" and ids(noisy)[1] not in {"a2"}  # without the bar it does reach for something else


# ---------------------------------------------------------------- bad input

@pytest.mark.parametrize("query", ["", "   ", None, 5])
def test_the_question_must_be_text(news, query):
    with pytest.raises(ValueError, match="question"):
        search_news(query, store=news)


@pytest.mark.parametrize("k", [0, -1, MAX_K + 1, True, "5", 2.5, None])
def test_k_must_be_a_whole_number_in_range(news, k):
    with pytest.raises(ValueError, match="k must be"):
        search_news("apple", k=k, store=news)


@pytest.mark.parametrize("days", [0, -3, True, 1.5, "7"])
def test_since_days_must_be_a_whole_number_of_days(news, days):
    with pytest.raises(ValueError, match="since_days"):
        search_news("apple", since_days=days, store=news)


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


def test_cli_prints_the_hits_with_score_source_and_filters(cli):
    assert retriever.main(["apple iphone phone", "--tickers", "AAPL", "-k", "2"]) == 0
    out = cli.readouterr().out
    assert "question : 'apple iphone phone'" in out and "filters  : tickers AAPL" in out
    assert "k=2, relevance bar 0.4, diverse no" in out
    assert " 1  1.00  Yahoo" in out and "Headline of a1" in out and "chunk 1/1" in out and "https://example.com/a1" in out
    assert "Headline of a2" in out and "Headline of a3" not in out


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


# ---------------------------------------------------------------- the real embedding model (skipped when unavailable)

@pytest.fixture(scope="module")
def real_embeddings():
    try:
        return rag_store.get_embeddings()
    except Exception as error:  # no local copy of the model and no network
        pytest.skip(f"the embedding model is not available: {type(error).__name__}")


@pytest.fixture
def real_news(tmp_path, real_embeddings) -> Chroma:
    """The three real articles in tests/fixtures (four chunks), embedded by the real model."""
    store = make_store(tmp_path / "real", real_embeddings)
    ingest_documents(load_news(FIXTURE), store)
    return store


def test_with_the_real_model_questions_find_their_article(real_news):
    # no word in common with the article: found by meaning
    assert search_news("new folding smartphone sales", store=real_news)[0].article_id == "finnhub:143136906"
    first = search_news("an AI cloud company worried about its debt", min_score=None, store=real_news)[0]
    assert first.article_id == COREWEAVE
    assert search_news("what do Meta and Apple plan for AI inference spending", store=real_news)[0].article_id == "finnhub:143103074"


def test_with_the_real_model_an_off_topic_question_has_no_supporting_text(real_news):
    assert search_news("chocolate cake recipe", store=real_news) == []
    assert search_news("chocolate cake recipe", min_score=None, store=real_news)  # the hits exist, none clears the bar


def test_with_the_real_model_filters_and_the_bar_work_together(real_news):
    only_coreweave = search_news("debt", tickers=["MSFT"], min_score=None, store=real_news)
    assert ids(only_coreweave) == [COREWEAVE]  # the list filter works on real chunks
    # the CoreWeave article names META, but a question about foldable phones has nothing to find there
    assert ids(search_news("foldable iPhone sales", tickers="META", min_score=None, store=real_news)) == [COREWEAVE]
    assert search_news("foldable iPhone sales", tickers="META", store=real_news) == []
    # the window: the iPhone article is from 2026-10-23, the CoreWeave one from 2026-10-17
    now = datetime(2026, 10, 24, 0, 0, tzinfo=timezone.utc)
    recent = search_news("news", since_days=3, min_score=None, k=5, store=real_news, now=now)
    assert COREWEAVE not in ids(recent) and "finnhub:143136906" in ids(recent)


def test_the_real_entry_point_end_to_end(monkeypatch, tmp_path, real_embeddings, capsys):
    from prometheus.rag import ingest
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    (news_dir / "AAPL.txt").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(news_dir))
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "index"))
    assert ingest.main(["AAPL"]) == 0
    capsys.readouterr()
    assert retriever.main(["new folding smartphone sales", "--tickers", "AAPL", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "The iPhone Duo Launch Kicks Off a Crucial Few Weeks for Apple" in out
    assert [h.article_id for h in search_news("new folding smartphone sales", k=1)] == ["finnhub:143136906"]  # the real store, found through the settings
