"""Tests for the press release path of the RAG layer (design choices entry 34): the file <SYMBOL>_PRs.txt, its loader, its own Chroma collection,
ingest, search, and the DataToolbox writing the file.

Run:  pytest tests/test_press_releases.py -q

No network, no key, no cost. The records are the first three REAL press releases of data/press_releases/AAPL.txt, copied byte for byte into
tests/fixtures/press_releases_sample.txt. The loader tests that break a record use tiny synthetic files, as the news loader tests do. The toolbox test
does not invent an API answer: it builds the shape Finnhub returns from those same three real records (and so also proves their ids are reproducible).
Every store and folder lives in a temporary folder; the real index and the real files are never touched.
"""
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prometheus.agents.data_clerk import data_toolbox
from prometheus.agents.data_clerk.data_toolbox import DataToolbox
from prometheus.rag import config, ingest, retriever, store as rag_store
from prometheus.rag.ingest import ingest_press_releases
from prometheus.rag.loaders import (PRESS_RELEASE_PUBLISHER, load_news, load_press_releases, main as loaders_main, press_release_file)
from prometheus.rag.retriever import search_news, search_press_releases
from test_rag_retriever import KeywordEmbeddings, make_store

FIXTURE = Path(__file__).parent / "fixtures" / "press_releases_sample.txt"
GIANNANDREA = "pr:5a31fe63040096fc"

# A minimal valid record. Each error test breaks one thing in it.
GOOD = """<<<PRESS_RELEASE>>>
id: abc123
symbol: AAPL
published_utc: 2026-10-23T19:03:00Z
url: https://example.com/a
headline: A headline
<<<TEXT>>>
Some text.
<<<END>>>
"""


def load_text(tmp_path: Path, text: str):
    path = tmp_path / "AAPL_PRs.txt"
    path.write_text(text, encoding="utf-8")
    return load_press_releases(path)


# ---------------------------------------------------------------- the loader, on real records


# ---------------------------------------------------------------- the loader, on broken input: a release that fails is discarded (design choices entry 83)

def load_with_skips(tmp_path, text):
    path = tmp_path / "AAPL_PRs.txt"
    path.write_text(text, encoding="utf-8")
    skipped = []
    return load_press_releases(path, skipped), skipped


@pytest.mark.parametrize("text, loaded, message", [
    (GOOD.replace("<<<END>>>\n", ""), 0, "the record has no <<<END>>>"),
    (GOOD + GOOD.replace("id: abc123", "id: abc124").replace("<<<END>>>\n", "") + GOOD, 2, "the record has no <<<END>>> (a new record starts at line"),
    ("stray text\n" + GOOD, 1, "text outside a record"),
    (GOOD + "stray text\n", 0, "text follows its <<<END>>>"),
    (GOOD.replace("<<<TEXT>>>", "<<<BODY>>>"), 0, "unknown marker '<<<BODY>>>'"),
    (GOOD.replace("headline: A headline", "headline without a colon"), 0, "header line without a colon"),
    (GOOD.replace("id: abc123\n", ""), 0, "the record is missing id"),
    (GOOD.replace("symbol: AAPL\n", ""), 0, "the record is missing symbol"),
    (GOOD.replace("headline: A headline\n", ""), 0, "the record is missing headline"),
    (GOOD.replace("2026-10-23T19:03:00Z", "yesterday"), 0, "bad published_utc 'yesterday'"),
    (GOOD.replace("symbol: AAPL", "symbol: ../../.env"), 0, "not a valid ticker symbol"),
])
def test_a_release_the_toolbox_would_not_have_written_is_skipped_completely_and_says_where_and_why(tmp_path, text, loaded, message):
    docs, skipped = load_with_skips(tmp_path, text)
    assert len(docs) == loaded and skipped and message in skipped[0].reason and "AAPL_PRs.txt:" in skipped[0].where


def test_a_real_release_that_fails_is_discarded_and_the_real_ones_around_it_still_load(tmp_path):
    first, second, third = ("<<<PRESS_RELEASE>>>\n" + part for part in FIXTURE.read_text(encoding="utf-8").split("<<<PRESS_RELEASE>>>\n")[1:])
    broken = second.replace("published_utc: 2025-12-01T17:10:00Z", "published_utc: Dec 1")
    assert broken != second
    docs, skipped = load_with_skips(tmp_path, first + broken + third)
    assert [d.id for d in docs] == ["pr:44107d69ebe0e861", "pr:519ab77bd1631dd7"] and len(skipped) == 1 and skipped[0].where.endswith("AAPL_PRs.txt:10")


def test_bytes_that_are_not_utf8_discard_that_release_and_name_the_line(tmp_path):
    path = tmp_path / "AAPL_PRs.txt"
    path.write_bytes(GOOD.encode().replace(b"Some text.", b"Some \xff text.") + GOOD.replace("abc123", "abc124").encode())
    skipped = []
    assert [d.id for d in load_press_releases(path, skipped)] == ["pr:abc124"] and len(skipped) == 1 and "line 8: not valid UTF-8" in skipped[0].reason


# ---------------------------------------------------------------- where the files are


def test_the_loader_command_line_shows_what_a_press_release_file_becomes(capsys):
    assert loaders_main([str(FIXTURE), "--press-releases", "--show", "1"]) == 0
    out = capsys.readouterr().out
    assert "3 press releases | 2025-10-30T16:30:00Z to 2025-12-04T09:00:00Z | 0 with insights | 3 distinct ids" in out
    assert "Apple reports fourth quarter results" in out and "pr:44107d69ebe0e861" in out


def test_the_loader_command_line_says_a_broken_release_was_discarded_without_a_traceback_and_a_filing_file_still_stops_it(tmp_path, capsys):
    bad = tmp_path / "AAPL_PRs.txt"
    bad.write_text(GOOD.replace("<<<END>>>\n", ""), encoding="utf-8")
    assert loaders_main([str(bad), "--press-releases"]) == 0                                                  # entry 83: a failed release is discarded, not an error
    captured = capsys.readouterr()
    assert "1 record failed and was discarded (never loaded)" in captured.out and "the record has no <<<END>>>" in captured.out and "Traceback" not in captured.err
    filings = tmp_path / "AAPL.txt"
    filings.write_text("<<<FILING>>>\naccession: 0000320193-25-000079\n", encoding="utf-8")                      # filings keep the strict rule: the file stops the load
    assert loaders_main([str(filings), "--filings"]) == 1
    assert "ERROR:" in capsys.readouterr().err


# ---------------------------------------------------------------- two collections in one folder


def test_an_unknown_kind_is_refused_before_anything_is_created():
    for call in (rag_store.collection_name, rag_store.get_vector_store, rag_store.reset_collection, rag_store.count_chunks):
        with pytest.raises(ValueError, match="unknown kind 'social'"):  # was 'filings' until filings became a kind (entry 35); 'social' is not one yet
            call(kind="social") if call is rag_store.collection_name else call("social")


@pytest.fixture
def folder_store(tmp_path, monkeypatch):
    """The real store functions over a temporary folder and the fake word embedder (no model download)."""
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "vectors"))
    monkeypatch.setattr(rag_store, "get_embeddings", lambda: KeywordEmbeddings())
    return rag_store


def test_each_kind_has_its_own_collection_so_a_count_or_a_reset_of_one_never_touches_the_other(folder_store):
    news, press = folder_store.get_vector_store("news"), folder_store.get_vector_store("press_releases")
    news.add_texts(["apple iphone"], ids=["n#0"], metadatas=[{"article_id": "n"}])
    press.add_texts(["apple iphone", "nvidia chip"], ids=["p#0", "q#0"], metadatas=[{"article_id": "p"}, {"article_id": "q"}])
    assert (folder_store.count_chunks(), folder_store.count_chunks("press_releases")) == (1, 2)  # "news" is the default
    assert folder_store.reset_collection("press_releases") is True
    assert (folder_store.count_chunks("news"), folder_store.count_chunks("press_releases")) == (1, 0)
    assert folder_store.reset_collection("press_releases") is False  # nothing left to delete


# ---------------------------------------------------------------- ingest

@pytest.fixture
def press_store(tmp_path):
    return make_store(tmp_path / "chroma_pr", KeywordEmbeddings())


@pytest.fixture
def ingest_cli(monkeypatch, tmp_path, press_store):
    prs = tmp_path / "prs"
    prs.mkdir()
    (prs / "AAPL_PRs.txt").write_bytes(FIXTURE.read_bytes())
    (prs / "META_PRs.txt").write_bytes(FIXTURE.read_bytes().replace(b"symbol: AAPL", b"symbol: META"))
    (prs / "AAPL.txt").write_text("not a press release file: the old name", encoding="utf-8")  # the pre-entry-34 name is not picked up by --all
    monkeypatch.setenv("PRESS_RELEASES_OUTPUT_DIR", str(prs))
    monkeypatch.setattr(ingest, "get_vector_store", lambda kind="news": press_store if kind == "press_releases" else pytest.fail("the news store must not be used"))
    monkeypatch.setattr(ingest, "count_too_long", lambda texts: 0)
    monkeypatch.setattr(ingest, "count_chunks", lambda kind="news": len(press_store.get()["ids"]))
    return prs


def test_the_ingest_command_line_reads_press_release_files_into_the_press_collection(ingest_cli, capsys):
    assert ingest.main(["--press-releases", "AAPL"]) == 0
    out = capsys.readouterr().out
    assert "AAPL: 3 press releases -> 3 chunks | added 3, updated 0, unchanged 0, deleted 0" in out and "the collection now holds 3 chunks" in out


def test_all_with_press_releases_takes_every_PRs_file_and_only_those_and_rebuild_clears_only_the_press_collection(ingest_cli, monkeypatch, capsys):
    resets = []
    monkeypatch.setattr(ingest, "reset_collection", lambda kind="news": resets.append(kind) or True)
    assert ingest.main(["--press-releases", "--rebuild", "--all"]) == 0
    out = capsys.readouterr().out
    assert resets == ["press_releases"] and "AAPL: 3 press releases" in out and "META: 3 press releases" in out and "the old name" not in out
    assert out.count("press releases ->") == 2  # AAPL and META, and not the file with the old name


def test_the_ingest_command_line_says_when_there_are_no_press_release_files(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRESS_RELEASES_OUTPUT_DIR", str(tmp_path))
    assert ingest.main(["--press-releases", "--all"]) == 1
    assert "ERROR: no press release files in" in capsys.readouterr().err


# ---------------------------------------------------------------- search

@pytest.fixture
def stored(press_store, monkeypatch, tmp_path):
    monkeypatch.setenv("PRESS_RELEASES_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "AAPL_PRs.txt").write_bytes(FIXTURE.read_bytes())
    ingest_press_releases("AAPL", store=press_store)
    return press_store


def test_bad_search_arguments_are_refused_for_press_releases_as_for_news(stored):
    for bad in (dict(k=0), dict(k=21), dict(since_days=0), dict(tickers="../../.env")):
        with pytest.raises(ValueError):
            search_press_releases("Apple", store=stored, **bad)
    with pytest.raises(ValueError):
        search_press_releases("   ", store=stored)


def test_the_retriever_command_line_can_search_the_press_releases(stored, monkeypatch, capsys):
    monkeypatch.setattr(retriever, "get_vector_store", lambda kind="news": stored if kind == "press_releases" else pytest.fail("the news store must not be used"))
    monkeypatch.setattr(retriever, "count_chunks", lambda kind="news": 3 if kind == "press_releases" else 0)
    assert retriever.main(["Apple", "--tickers", "AAPL", "--press-releases", "--min-score", "0"]) == 0
    out = capsys.readouterr().out
    assert "Press release" in out and "Apple reports fourth quarter results" in out and "chunk 1/1" in out


def test_the_retriever_command_line_tells_you_how_to_ingest_press_releases_when_there_are_none(monkeypatch, capsys):
    monkeypatch.setattr(retriever, "count_chunks", lambda kind="news": 0)
    assert retriever.main(["Apple", "--press-releases"]) == 1
    assert "ingest --press-releases --all" in capsys.readouterr().err


# ---------------------------------------------------------------- the toolbox writes <SYMBOL>_PRs.txt

def finnhub_answer_from_the_real_records():
    """The shape Finnhub's /press-releases returns ({"majorDevelopment": [{symbol, datetime, headline, description, url}]}), filled from the
    three real records of the fixture. Nothing is made up: each value is one that was in the file."""
    items = []
    for record in FIXTURE.read_text(encoding="utf-8").split("<<<PRESS_RELEASE>>>\n")[1:]:
        head, _, rest = record.partition("<<<TEXT>>>\n")
        fields = dict(line.split(": ", 1) for line in head.strip().splitlines())
        when = datetime.strptime(fields["published_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        items.append({"symbol": fields["symbol"], "datetime": int(when.timestamp()), "headline": fields["headline"],
                      "description": rest.rsplit("\n<<<END>>>", 1)[0], "url": fields["url"]})
    return {"symbol": "AAPL", "majorDevelopment": items}


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    asked = []

    def fake_get_json(client, limiter, name, url, params, headers=None):
        asked.append((name, url, params, sorted(headers or {})))  # the real code never puts the key in the url: it is a header
        return finnhub_answer_from_the_real_records()

    monkeypatch.setenv("PRESS_RELEASES_OUTPUT_DIR", str(tmp_path / "prs"))
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key-not-real")
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(fake_get_json))
    box = DataToolbox(base_dir=tmp_path)
    box.asked = asked
    return box


def test_the_toolbox_appends_to_symbol_underscore_PRs_in_the_press_release_folder(toolbox, tmp_path, capsys):
    stats = toolbox.fetch_press_releases("aapl", from_date=400)
    capsys.readouterr()
    path = tmp_path / "prs" / "AAPL_PRs.txt"
    assert stats == {"symbol": "AAPL", "path": str(path), "fetched": 3, "written": 3, "duplicates": 0, "failed": 0} and path.exists()
    assert not (tmp_path / "prs" / "AAPL.txt").exists() and not (tmp_path / "data" / "press_releases").exists()  # neither the old name nor the old folder
    name, url, params, headers = toolbox.asked[0]
    assert url == data_toolbox.FINNHUB_PRESS_URL and params["symbol"] == "AAPL" and headers == ["X-Finnhub-Token"] and "token" not in str(params)


def test_what_the_toolbox_writes_is_what_the_loader_reads_and_the_ids_are_reproducible_from_the_content(toolbox, tmp_path, capsys):
    toolbox.fetch_press_releases("AAPL")
    capsys.readouterr()
    written = load_press_releases(tmp_path / "prs" / "AAPL_PRs.txt")
    real = load_press_releases(FIXTURE)
    assert [d.id for d in written] == [d.id for d in real]  # the toolbox recomputed the real ids from symbol + time + headline
    assert [d.page_content for d in written] == [d.page_content for d in real] and [d.metadata for d in written] == [d.metadata for d in real]
    first = real[0]
    published = datetime.fromtimestamp(first.metadata["published_ts"], tz=timezone.utc).isoformat()
    assert first.id == "pr:" + hashlib.sha1(f"AAPL|{published}|{first.metadata['headline']}".encode()).hexdigest()[:16]


def test_fetching_again_appends_nothing_so_the_file_never_gets_a_duplicate(toolbox, tmp_path, capsys):
    toolbox.fetch_press_releases("AAPL")
    before = (tmp_path / "prs" / "AAPL_PRs.txt").read_bytes()
    again = toolbox.fetch_press_releases("AAPL")
    capsys.readouterr()
    assert (again["fetched"], again["written"], again["duplicates"]) == (3, 0, 3)
    assert (tmp_path / "prs" / "AAPL_PRs.txt").read_bytes() == before


def test_the_toolbox_still_checks_its_arguments_and_reaches_the_api_for_nothing_that_is_refused(toolbox):
    for bad in (dict(symbol=""), dict(symbol="AAPL", from_date=0), dict(symbol="AAPL", from_date=True)):
        with pytest.raises(ValueError):
            toolbox.fetch_press_releases(**bad)
    assert toolbox.asked == []


# ---------------------------------------------------------------- the store's command line

def test_store_main_with_no_arguments_means_no_options_and_never_reads_the_callers_command_line(monkeypatch, capsys):
    """Found by the first full run with the network on: main() once parsed sys.argv, which under pytest is pytest's own arguments (SystemExit 2). main() with no
    arguments has always meant 'the news collection'; the command line passes its own arguments."""
    monkeypatch.setattr("sys.argv", ["pytest", "-q", "tests/", "--some-option-of-the-caller"])
    monkeypatch.setattr(rag_store, "count_chunks", lambda kind="news": 0)
    assert rag_store.main() == 0 and "collection : news-" in capsys.readouterr().out


def test_store_main_can_show_the_press_release_collection(monkeypatch, capsys):
    monkeypatch.setattr(rag_store, "count_chunks", lambda kind="news": 0)
    assert rag_store.main(["--press-releases"]) == 0
    out = capsys.readouterr().out
    assert "collection : press_releases-" in out and "ingest --press-releases AAPL" in out


