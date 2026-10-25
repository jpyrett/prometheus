"""Tests for prometheus.rag.config and prometheus.rag.loaders (Part 3 of the RAG guide).

Run:  pytest tests/test_rag_loaders.py -q

The happy-path tests use tests/fixtures/news_sample.txt, which holds three REAL records copied
verbatim from news/AAPL.txt and news/NVDA.txt. The error tests use tiny synthetic files. The last
test reads every real file in news/ (it is skipped when there are none).
"""
from pathlib import Path

import pytest

from prometheus.rag import config
from prometheus.rag.loaders import load_news, main, news_file

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

def test_fixture_loads_three_documents_with_unique_ids():
    docs = load_news(FIXTURE)
    assert len(docs) == 3
    assert [d.id for d in docs] == [d.metadata["article_id"] for d in docs]
    assert len({d.id for d in docs}) == 3


def test_finnhub_record_becomes_headline_plus_summary():
    doc = load_news(FIXTURE)[0]
    assert doc.id == "finnhub:143136906"
    assert doc.page_content == (
        "The iPhone Duo Launch Kicks Off a Crucial Few Weeks for Apple\n\n"
        "Apple’s foldable iPhone goes on sale ahead of an expected touchscreen MacBook launch "
        "and fiscal fourth-quarter earnings.")
    assert doc.metadata == {
        "article_id": "finnhub:143136906",
        "headline": "The iPhone Duo Launch Kicks Off a Crucial Few Weeks for Apple",
        "tickers": ["AAPL"],
        "published_utc": "2026-10-23T19:03:00Z",
        "published_ts": 1792782180,  # 2026-10-23 19:03:00 UTC, worked out by hand
        "publisher": "Yahoo",
        "api": "finnhub",
        "url": "https://finnhub.io/api/news?id=d785010198f95cde7609b110185876b490a53c6cc083289353e226aeb037f1b1",
    }


def test_polygon_record_keeps_the_insight_lines_and_every_ticker():
    doc = load_news(FIXTURE)[1]
    assert doc.id.startswith("polygon:287f3b5c")
    headline, summary, insights = doc.page_content.split("\n\n")  # three paragraphs
    assert headline.startswith("CoreWeave Is Projected to Have a 6.3x Gross Leverage Ratio")
    assert summary.startswith("CoreWeave's stock fell 15.5%")
    lines = insights.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("CRWV | negative | Stock declined 11.91%")
    assert [line.split(" | ")[0] for line in lines] == ["CRWV", "MSFT", "META", "NVDA"]
    assert doc.metadata["tickers"] == ["CRWV", "MSFT", "META", "NVDA"]  # the file is NVDA's, the list has all four
    assert doc.metadata["published_ts"] == 1792272600  # 2026-10-17 21:30:00 UTC, worked out by hand
    assert doc.metadata["publisher"] == "The Motley Fool"
    assert doc.metadata["api"] == "polygon"


def test_only_the_decided_fields_become_metadata():
    for doc in load_news(FIXTURE):
        assert set(doc.metadata) == {"article_id", "headline", "tickers", "published_utc",
                                     "published_ts", "publisher", "api", "url"}


def test_a_colon_inside_a_headline_is_kept():
    doc = load_news(FIXTURE)[2]
    assert doc.metadata["headline"] == "Meta Vs. Apple: Divergent Edge AI Inference Strategies and Spending"
    assert doc.page_content.startswith("Meta Vs. Apple: Divergent Edge AI")


# ---------------------------------------------------------------- small synthetic files

def test_minimal_record(tmp_path):
    (doc,) = load_text(tmp_path, GOOD)
    assert doc.page_content == "A headline\n\nA summary."
    assert doc.metadata["tickers"] == ["AAPL"]


def test_empty_file_gives_no_documents(tmp_path):
    assert load_text(tmp_path, "") == []


def test_blank_lines_between_records_are_fine(tmp_path):
    assert len(load_text(tmp_path, GOOD + "\n\n" + GOOD)) == 2


def test_empty_summary_and_insights_leave_just_the_headline(tmp_path):
    (doc,) = load_text(tmp_path, GOOD.replace("A summary.\n", ""))
    assert doc.page_content == "A headline"


def test_a_line_that_only_looks_like_a_marker_is_text(tmp_path):
    # The DataAgent escapes a text line that starts with <<< by putting a space in front of it.
    (doc,) = load_text(tmp_path, GOOD.replace("A summary.", " <<<SUMMARY>>> is only mentioned here."))
    assert "<<<SUMMARY>>> is only mentioned here." in doc.page_content


def test_tickers_fall_back_to_the_symbol_the_file_was_fetched_for(tmp_path):
    (doc,) = load_text(tmp_path, GOOD.replace("tickers: AAPL", "tickers: "))
    assert doc.metadata["tickers"] == ["AAPL"]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_news(Path("/nonexistent/AAPL.txt"))


# ---------------------------------------------------------------- errors name the file and line

@pytest.mark.parametrize("text, expected", [
    (GOOD.replace("<<<END>>>\n", ""), "AAPL.txt:1: the record has no <<<END>>>"),
    (GOOD.replace("<<<END>>>\n", "") + GOOD, "AAPL.txt:14: a new record starts before the one at line 1 ended"),
    (GOOD.replace("A summary.", "<<<FILING>>>"), "AAPL.txt:11: unknown marker '<<<FILING>>>'"),
    (GOOD.replace("publisher: Yahoo", "publisher Yahoo"), "AAPL.txt:6: header line without a colon"),
    (GOOD.replace("url: https://example.com/a\n", ""), "AAPL.txt:1: the record is missing url"),
    (GOOD.replace("headline: A headline\n", ""), "AAPL.txt:1: the record is missing headline"),
    (GOOD.replace("2026-10-23T19:03:00Z", "yesterday"), "AAPL.txt:1: bad published_utc 'yesterday'"),
    (GOOD.replace("symbol: AAPL\n", "").replace("tickers: AAPL", "tickers: "),
     "AAPL.txt:1: the record has neither tickers nor a symbol"),
    ("some stray text\n" + GOOD, "AAPL.txt:1: text outside a record: 'some stray text'"),
    (GOOD + "<<<SUMMARY>>>\n", "AAPL.txt:15: text outside a record"),
])
def test_bad_records_raise_with_file_and_line(tmp_path, text, expected):
    with pytest.raises(ValueError) as error:
        load_text(tmp_path, text)
    assert expected in str(error.value)


# ---------------------------------------------------------------- symbols become file names

def test_news_file_uppercases_the_symbol_and_uses_the_news_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(tmp_path))
    assert news_file("aapl") == tmp_path / "AAPL.txt"
    assert news_file(" brk.b ") == tmp_path / "BRK.B.txt"
    assert news_file("BF-B") == tmp_path / "BF-B.txt"


@pytest.mark.parametrize("bad", ["../../.env", "AAPL/../../x", "/etc/passwd", "", "A B", "AAPL.txt/", "$AAPL",
                                 "TOOLONGSYMBOL1", "-AAPL"])
def test_news_file_refuses_anything_that_is_not_a_ticker(bad):
    with pytest.raises(ValueError, match="not a valid ticker symbol"):
        news_file(bad)


# ---------------------------------------------------------------- settings

def test_setting_order_is_environment_then_env_file_then_default(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("ZZ_TEST_SETTING=from_env_file\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("ZZ_TEST_SETTING", raising=False)
    assert config.setting("ZZ_TEST_SETTING", "default") == "from_env_file"
    monkeypatch.setenv("ZZ_TEST_SETTING", "from_environment")
    assert config.setting("ZZ_TEST_SETTING", "default") == "from_environment"
    assert config.setting("ZZ_OTHER_SETTING", "default") == "default"


def test_news_dir_defaults_to_news_at_the_repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)  # a root with no .env
    monkeypatch.delenv("NEWS_OUTPUT_DIR", raising=False)
    assert config.news_dir() == tmp_path / "news"


def test_a_relative_path_setting_is_resolved_against_the_repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("NEWS_OUTPUT_DIR", "somewhere/else")
    assert config.news_dir() == tmp_path / "somewhere" / "else"
    monkeypatch.setenv("NEWS_OUTPUT_DIR", "/absolute/place")
    assert config.news_dir() == Path("/absolute/place")


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


# ---------------------------------------------------------------- the DataAgent's real files

def test_every_real_news_file_loads_and_fits_the_design():
    files = sorted(config.news_dir().glob("*.txt"))
    if not files:
        pytest.skip(f"no news files in {config.news_dir()}")
    for file in files:
        docs = load_news(file)
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids)), f"{file.name}: an article id appears twice"
        for doc in docs:
            assert doc.page_content.startswith(doc.metadata["headline"])
            assert isinstance(doc.metadata["published_ts"], int)
            assert doc.metadata["tickers"] and all(isinstance(t, str) for t in doc.metadata["tickers"])
            # The design filters by `tickers`, so a file's own symbol must be in every article's list.
            assert file.stem in doc.metadata["tickers"], f"{file.name}: {doc.id} does not list {file.stem}"


def test_an_article_in_two_real_files_differs_only_in_its_tickers():
    """Part 4 merges the ticker lists of an article it has already stored and keeps everything else,
    so everything else must match. Finnhub's tickers is just the symbol it was asked for: the same
    article is ['AAPL'] in AAPL.txt and ['NVDA'] in NVDA.txt."""
    files = sorted(config.news_dir().glob("*.txt"))
    if len(files) < 2:
        pytest.skip("needs at least two real news files")
    first_seen = {}
    for file in files:
        for doc in load_news(file):
            if doc.id not in first_seen:
                first_seen[doc.id] = doc
                continue
            first = first_seen[doc.id]
            assert doc.page_content == first.page_content, f"{doc.id}: the text differs between files"
            without_tickers = lambda d: {k: v for k, v in d.metadata.items() if k != "tickers"}  # noqa: E731
            assert without_tickers(doc) == without_tickers(first), f"{doc.id}: more than the tickers differs"
