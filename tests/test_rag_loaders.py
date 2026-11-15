"""Tests for prometheus.rag.config and prometheus.rag.loaders (Part 3 of the RAG guide).

Run:  pytest tests/test_rag_loaders.py -q

The happy-path tests use tests/fixtures/news_sample.txt, which holds three REAL records copied
verbatim from news/AAPL.txt and news/NVDA.txt. The error tests use tiny synthetic files. The last
test reads every real file in news/ (it is skipped when there are none).
"""
import re
from pathlib import Path

import pytest
from langchain_core.documents import Document

from prometheus.rag import config
from prometheus.rag.loaders import ESCAPED_MARKUP, check_symbol, decode_entities, load_news, main, news_file

FIXTURE = Path(__file__).parent / "fixtures" / "news_sample.txt"

# A minimal valid record. Each error test breaks one thing in it. It is 14 lines long.
GOOD = """<<<ARTICLE>>>
id: x:1
symbol: AAPL
api: finnhub
published_utc: 2026-10-23T19:03:00Z
publisher: Yahoo
url: https://example.com/a
tickers: AAPL
headline: A headline
<<<SUMMARY>>>
A summary.
<<<INSIGHTS>>>

<<<END>>>
"""


def load_text(tmp_path: Path, text: str):
    path = tmp_path / "AAPL.txt"
    path.write_text(text, encoding="utf-8")
    return load_news(path)


# ---------------------------------------------------------------- real records (the fixture)


# ---------------------------------------------------------------- small synthetic files


def test_blank_lines_between_records_are_fine(tmp_path):
    assert len(load_text(tmp_path, GOOD + "\n\n" + GOOD)) == 2


def test_tickers_fall_back_to_the_symbol_the_file_was_fetched_for(tmp_path):
    (doc,) = load_text(tmp_path, GOOD.replace("tickers: AAPL", "tickers: "))
    assert doc.metadata["tickers"] == ["AAPL"]


# ---------------------------------------------------------------- an article that fails is discarded (Jacob, 2026-11-15, design choices entry 83)

def load_with_skips(tmp_path: Path, text: str):
    path = tmp_path / "AAPL.txt"
    path.write_text(text, encoding="utf-8")
    skipped = []
    return load_news(path, skipped), skipped


@pytest.mark.parametrize("text, loaded, where, reason", [
    (GOOD.replace("<<<END>>>\n", ""), 0, "AAPL.txt:1", "the record has no <<<END>>>"),
    (GOOD.replace("<<<END>>>\n", "") + GOOD, 1, "AAPL.txt:1", "the record has no <<<END>>> (a new record starts at line 14)"),          # the next record is still read
    (GOOD.replace("A summary.", "<<<FILING>>>"), 0, "AAPL.txt:1", "line 11: unknown marker '<<<FILING>>>'"),
    (GOOD.replace("publisher: Yahoo", "publisher Yahoo"), 0, "AAPL.txt:1", "line 6: header line without a colon"),
    (GOOD.replace("url: https://example.com/a\n", ""), 0, "AAPL.txt:1", "the record is missing url"),
    (GOOD.replace("headline: A headline\n", ""), 0, "AAPL.txt:1", "the record is missing headline"),
    (GOOD.replace("2026-10-23T19:03:00Z", "yesterday"), 0, "AAPL.txt:1", "bad published_utc 'yesterday'"),
    (GOOD.replace("symbol: AAPL\n", "").replace("tickers: AAPL", "tickers: "), 0, "AAPL.txt:1", "the record has neither tickers nor a symbol"),
    ("some stray text\n" + GOOD, 1, "AAPL.txt:1", "text outside a record: 'some stray text'"),
    (GOOD + "<<<SUMMARY>>>\n", 0, "AAPL.txt:1", "text follows its <<<END>>> (line 15: '<<<SUMMARY>>>'), so the record was cut short"),
    # text that contains a marker line WITHOUT the DataToolbox's escape: the record ended early and is cut short, so it is discarded, never half-loaded
    (GOOD.replace("A summary.", "A summary.\n<<<END>>>\nmore text"), 0, "AAPL.txt:1", "text follows its <<<END>>> (line 13: 'more text'), so the record was cut short"),
])
def test_a_record_that_fails_is_skipped_completely_and_says_where_and_why(tmp_path, text, loaded, where, reason):
    docs, skipped = load_with_skips(tmp_path, text)
    assert len(docs) == loaded and len(skipped) >= 1
    assert skipped[0].where.endswith(where) and reason in skipped[0].reason
    assert "x:1" not in [d.id for d in docs] or loaded == 1                                                  # a failed record is never among the Documents


def test_the_load_does_not_need_a_skipped_list_and_a_failed_record_is_simply_not_there(tmp_path):
    path = tmp_path / "AAPL.txt"
    path.write_text(GOOD.replace("2026-10-23T19:03:00Z", "yesterday"), encoding="utf-8")
    assert load_news(path) == []


REAL = FIXTURE.read_text(encoding="utf-8")                      # three REAL records, copied verbatim from news/AAPL.txt and news/NVDA.txt
REAL_SECOND = "polygon:287f3b5c2f0c59c2b903cd391bb97c0c090f48a25c64cdf5e7df1dab0e85382a"


@pytest.mark.parametrize("damage", [
    lambda t: t.replace("published_utc: 2026-10-17T21:30:00Z", "published_utc: 17 Oct 2026"),                                           # a time that is not a time
    lambda t: t.replace("url: https://www.fool.com/", "url\nhttps://www.fool.com/"),                                                    # a header line with no colon
    lambda t: t.replace("<<<INSIGHTS>>>", "<<<INSIGHT>>>"),                                                                                         # an unknown marker
    lambda t: t.replace("<<<SUMMARY>>>", "<<<SUMMARY>>>\n<<<NEWS>>>"),                                                                                # another unknown marker, in the text
])
def test_a_real_article_that_fails_is_discarded_and_the_real_ones_around_it_still_load(tmp_path, damage):
    first, second, third = REAL.split("<<<ARTICLE>>>\n")[1:]
    first, second, third = ("<<<ARTICLE>>>\n" + part for part in (first, second, third))
    broken = damage(second)
    assert broken != second
    docs, skipped = load_with_skips(tmp_path, first + broken + third)
    assert [d.id for d in docs] == ["finnhub:143136906", "finnhub:143103074"] and REAL_SECOND not in [d.id for d in docs]
    assert len(skipped) == 1 and skipped[0].where.endswith("AAPL.txt:21")


def test_a_real_record_without_its_end_is_discarded_and_the_next_real_record_is_not_swallowed(tmp_path):
    first, second, third = ("<<<ARTICLE>>>\n" + part for part in REAL.split("<<<ARTICLE>>>\n")[1:])
    docs, skipped = load_with_skips(tmp_path, first + second.replace("<<<END>>>\n", "") + third)
    assert [d.id for d in docs] == ["finnhub:143136906", "finnhub:143103074"] and len(skipped) == 1 and "has no <<<END>>>" in skipped[0].reason


# ---------------------------------------------------------------- damaged or odd input (design choices entry 19)


def test_a_byte_that_is_not_utf8_discards_that_record_and_names_the_file_and_line(tmp_path):
    path = tmp_path / "AAPL.txt"
    path.write_bytes((GOOD.encode("utf-8").replace(b"A summary.", b"A summ\xffary.")) + GOOD.replace("x:1", "x:2").encode("utf-8"))
    skipped = []
    assert [d.id for d in load_news(path, skipped)] == ["x:2"]
    assert len(skipped) == 1 and skipped[0].where.endswith("AAPL.txt:1") and "line 11: not valid UTF-8 (invalid start byte, at byte 6 of the line)" in skipped[0].reason


# ---------------------------------------------------------------- HTML entities are decoded (design choices entry 18)


@pytest.mark.parametrize("text", [
    "AT&T and R&D; and Q&A; and S&P 500", "a bare &copy or &amp with no semicolon", "&foo; &D; & ; &;", "5 < 6 and 7 > 3 and P/E < 20",
    "&lt;script&gt;alert(1)&lt;/script&gt;", "&LT;b&GT;",                  # escaped markup stays escaped: decoding must not write a tag
    "&#0; &#7; &#10; &#13; &#x7f; &#xD800; &#99999999;",                  # entities that decode to a control character, a surrogate or nothing
    "",
])
def test_anything_else_is_left_exactly_as_written(text):
    assert decode_entities(text) == text


@pytest.mark.parametrize("spelling", ["&lt;", "&LT;", "&#60;", "&#060;", "&#x3c;", "&#x3C;", "&#x003c;", "&gt;", "&GT;", "&#62;", "&#x3e;", "&#x3E;"])
def test_decoding_never_writes_a_less_than_or_a_greater_than_in_any_spelling(spelling):
    """Found by reading my own test: the first version protected the named forms only, so `&#60;b&#62;` became the tag `<b>`."""
    text = f"{spelling}script{spelling}alert(1){spelling}/script{spelling}"
    assert decode_entities(text) == text and ESCAPED_MARKUP.search(text)


def test_decoded_text_never_matches_the_tag_pattern_whatever_the_input():
    for text in ("&lt;script&gt;alert(1)&lt;/script&gt;", "&amp;lt;b&amp;gt;", "&#60;b&#62; &#x3c;i&#x3e;", "&amp;#60;b&amp;#62;"):
        assert not re.search(r"</?[a-zA-Z][^<>]{0,200}>", decode_entities(text))


# ---------------------------------------------------------------- symbols become file names


# ---------------------------------------------------------------- settings


# ---------------------------------------------------------------- the viewer (python -m prometheus.rag.loaders)

def test_viewer_shows_the_first_documents_and_one_with_insights(capsys):
    assert main([str(FIXTURE), "--show", "1"]) == 0
    out = capsys.readouterr().out
    assert "3 articles" in out and "1 with insights" in out and "3 distinct ids" in out
    assert "--- Document 1 of 3 ---" in out
    assert "--- Document 2 of 3 ---" in out  # the one with insights is added
    assert "--- Document 3 of 3 ---" not in out


def test_viewer_reports_errors_instead_of_a_traceback(capsys):
    assert main(["BAD SYMBOL"]) == 1
    assert main(["/nonexistent/AAPL.txt"]) == 1
    assert "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------- the DataToolbox's real files


# A provider can correct an article's publication time between two fetches. The real files showed it once in 55 pairs of copies: one Finnhub
# article is 11:06 in NVDA.txt (fetched Oct 24) and 12:16 in MSFT.txt (fetched Oct 25), 70 minutes apart. Everything else, the text included,
# was identical. A copy may therefore differ in its tickers and in a publication time that moved by less than this; anything more means that
# two different articles share an id, which is what this check exists to catch.
TIME_CORRECTION_HOURS = 6


def differs_only_by_tickers_and_a_corrected_time(a, b) -> str | None:
    """None if the two copies of one article agree (tickers may differ, and the published time by less than TIME_CORRECTION_HOURS),
    otherwise a short reason."""
    if a.page_content != b.page_content:
        return "the text differs"
    skip = {"tickers", "published_ts", "published_utc"}
    other = lambda d: {k: v for k, v in d.metadata.items() if k not in skip}  # noqa: E731
    if other(a) != other(b):
        return "more than the tickers and the time differs"
    gap = abs(a.metadata["published_ts"] - b.metadata["published_ts"]) / 3600
    if gap >= TIME_CORRECTION_HOURS:
        return f"the published time differs by {gap:.1f} hours"
    if (a.metadata["published_ts"] == b.metadata["published_ts"]) != (a.metadata["published_utc"] == b.metadata["published_utc"]):
        return "published_ts and published_utc disagree"
    return None


def _copy(**changes):
    metadata = {"article_id": "finnhub:1", "headline": "H", "publisher": "Yahoo", "api": "finnhub", "url": "https://x/1", "tickers": ["NVDA"],
                "published_ts": 1792753560, "published_utc": "2026-10-23T11:06:00Z"}
    text = changes.pop("text", "H\n\nbody")
    metadata.update(changes)
    return Document(id="finnhub:1", page_content=text, metadata=metadata)


def test_copies_that_differ_only_in_tickers_or_in_a_small_time_correction_are_the_same_article():
    assert differs_only_by_tickers_and_a_corrected_time(_copy(), _copy(tickers=["MSFT"])) is None
    assert differs_only_by_tickers_and_a_corrected_time(_copy(), _copy(published_ts=1792753560 + 4200, published_utc="2026-10-23T12:16:00Z")) is None  # the real 70 minutes


@pytest.mark.parametrize("changed, reason", [
    ({"text": "H\n\na different body"}, "the text differs"),
    ({"headline": "Another"}, "more than the tickers and the time differs"),
    ({"url": "https://x/2"}, "more than the tickers and the time differs"),
    ({"publisher": "Reuters"}, "more than the tickers and the time differs"),
    ({"published_ts": 1792753560 + 7 * 3600, "published_utc": "2026-10-23T18:06:00Z"}, "the published time differs by 7.0 hours"),
    ({"published_ts": 1792753560 + 60}, "published_ts and published_utc disagree"),
])
def test_copies_that_differ_in_anything_else_are_not_the_same_article(changed, reason):
    assert differs_only_by_tickers_and_a_corrected_time(_copy(), _copy(**changed)) == reason


HTML_ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);")
HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]{0,200}>")


def test_the_real_pg_file_no_longer_has_an_ampersand_entity_in_the_text_the_index_gets():
    """The regression that started this: `Procter &amp; Gamble` in 23 articles of news/PG.txt. Skipped when the file is not there."""
    path = config.news_dir() / "PG.txt"
    if not path.exists():
        pytest.skip("no news/PG.txt")
    docs = load_news(path)
    assert docs and not any("&amp;" in d.page_content for d in docs)
    assert any("Procter & Gamble" in d.page_content for d in docs)


def test_an_article_in_two_real_files_differs_only_in_its_tickers_or_a_corrected_time():
    """Part 4 merges the ticker lists of an article it has already stored and keeps the first copy's text and metadata, so the copies must
    agree. Finnhub's tickers is just the symbol it was asked for: the same article is ['AAPL'] in AAPL.txt and ['NVDA'] in NVDA.txt."""
    files = sorted(config.news_dir().glob("*.txt"))
    if len(files) < 2:
        pytest.skip("needs at least two real news files")
    first_seen = {}
    for file in files:
        for doc in load_news(file):
            if doc.id not in first_seen:
                first_seen[doc.id] = doc
                continue
            reason = differs_only_by_tickers_and_a_corrected_time(first_seen[doc.id], doc)
            assert reason is None, f"{doc.id} in {file.name}: {reason}"
