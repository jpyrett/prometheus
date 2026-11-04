"""Tests for the filings path of the RAG layer (design choices entry 35): the file <SYMBOL>.txt of the filings folder, its loader, its own Chroma collection,
ingest, search, and the DataToolbox writing the file.

Run:  pytest tests/test_filings.py -q

No network, no key, no cost. The records are six REAL filings of data/filings/AAPL.txt (two 8-Ks, the 10-K, a 10-Q, a proxy statement and a Form 144), copied
byte for byte into tests/fixtures/filings_sample.txt by a script that checked each is an exact slice of the source. The loader tests that break a record use tiny
synthetic files, as the news and press release loader tests do. The toolbox test does not invent an API answer: it builds the shape the Massive filings endpoints
return from those same six real records, and checks that what the toolbox writes loads to the same Documents as the real fixture. Every store and folder lives in a
temporary folder; the real index and the real files are never touched.
"""
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings
from langchain_chroma import Chroma

from prometheus.agents.data_clerk import data_toolbox
from prometheus.agents.data_clerk import edgar_text
from prometheus.agents.data_clerk.data_toolbox import DataToolbox
from prometheus.rag import config, ingest, retriever, store as rag_store
from prometheus.rag.ingest import ingest_filings, to_chunks
from prometheus.rag.loaders import (FORM_NAMES, SECTION_LABELS, decode_entities, filing_file, form_name, load_filings, load_news, load_press_releases,
                                    main as loaders_main)
from prometheus.rag.retriever import search_filings, search_news, search_press_releases
from test_rag_retriever import KeywordEmbeddings

FIXTURE = Path(__file__).parent / "fixtures" / "filings_sample.txt"
EIGHT_K = "sec:0000320193-25-000077:items"
BUSINESS = "sec:0000320193-25-000079:business"
RISKS = "sec:0000320193-25-000079:risk_factors"
TEN_Q = "sec:0000320193-26-000006"
CEO_8K = "sec:0001140361-26-015711:items"
PROXY = "sec:0001308179-26-000008"
FORM_144 = "sec:0001959173-26-002757"
FIXTURE_IDS = [EIGHT_K, BUSINESS, RISKS, TEN_Q, CEO_8K, PROXY, FORM_144]  # the order of the records in the file; the 10-K is two Documents

# A minimal valid record. Each error test breaks one thing in it.
GOOD = """<<<FILING>>>
accession: 0000320193-25-000077
symbol: AAPL
form: 8-K
filed: 2025-10-30
period_end: 
url: https://www.sec.gov/Archives/edgar/data/320193/0000320193-25-000077.txt
text_status: ok
<<<TEXT>>>
== items ==
Item 2.02    Results.
<<<END>>>
"""
NO_TEXT = GOOD.replace("text_status: ok", "text_status: not_available").replace("== items ==\nItem 2.02    Results.\n", "")


class DistinctEmbeddings(KeywordEmbeddings):
    """The fake word embedder with a tiny tail taken from a hash of the text, so that no two chunks have the SAME vector. About 160 chunks of the real 10-K contain none
    of the embedder's ten words, so with the plain fake they would all be one identical vector, and HNSW search is known to lose track of some of many identical points:
    14 runs of the search tests failed once (an article missing from a search that asked for more chunks than the store holds). Real text vectors are never identical."""

    def _vector(self, text: str) -> list[float]:
        vector = super()._vector(text) + [b / 255 * 0.01 for b in hashlib.sha1(text.encode()).digest()[:4]]
        length = sum(x * x for x in vector) ** 0.5
        return [x / length for x in vector]


def load_text(tmp_path: Path, text: str):
    path = tmp_path / "AAPL.txt"
    path.write_text(text, encoding="utf-8")
    return load_filings(path)


def real_records() -> dict[str, tuple[dict, str]]:
    """The fixture read by hand, without the loader: {accession: (header fields, text between <<<TEXT>>> and <<<END>>>)}."""
    out = {}
    for record in FIXTURE.read_text(encoding="utf-8").split("<<<FILING>>>\n")[1:]:
        head, _, rest = record.partition("<<<TEXT>>>\n")
        fields = dict(line.partition(": ")[::2] for line in head.strip().splitlines())
        out[fields["accession"]] = (fields, rest.rsplit("<<<END>>>", 1)[0].strip())
    return out


# ---------------------------------------------------------------- the loader, on real records

def test_the_fixture_has_the_six_real_filings_and_loads_seven_documents_with_unique_prefixed_ids():
    assert len(real_records()) == 6
    docs = load_filings(FIXTURE)
    assert [d.id for d in docs] == FIXTURE_IDS and [d.id for d in docs] == [d.metadata["article_id"] for d in docs]
    assert len(set(d.id for d in docs)) == 7 and all(d.id.startswith("sec:") for d in docs)  # "sec:" keeps them apart from news and press release ids in one SourceBook


def test_a_filing_with_text_becomes_one_document_per_section_with_the_filing_in_its_headline():
    by_id = {d.id: d for d in load_filings(FIXTURE)}
    assert by_id[EIGHT_K].metadata["headline"] == "AAPL 8-K, filed 2025-10-30"
    assert by_id[BUSINESS].metadata["headline"] == "AAPL 10-K, Business, filed 2025-10-31, period ending 2025-09-27"
    assert by_id[RISKS].metadata["headline"] == "AAPL 10-K, Risk Factors, filed 2025-10-31, period ending 2025-09-27"
    assert by_id[EIGHT_K].page_content.startswith("AAPL 8-K, filed 2025-10-30\n\nItem 2.02    Results of Operations and Financial Condition.\nOn October 30, 2025, Apple Inc.")


def test_the_two_sections_of_the_10_K_are_the_text_of_the_file_split_at_its_section_lines_and_nothing_else():
    """Worked out from the raw file, not from the loader: nothing is dropped, added or reordered when a 10-K becomes two Documents."""
    _, text = real_records()["0000320193-25-000079"]
    business, risks = text.split("\n\n== risk_factors ==\n")
    business = business.removeprefix("== business ==\n")
    by_id = {d.id: d for d in load_filings(FIXTURE)}
    assert by_id[BUSINESS].page_content == by_id[BUSINESS].metadata["headline"] + "\n\n" + decode_entities(business.strip())
    assert by_id[RISKS].page_content == by_id[RISKS].metadata["headline"] + "\n\n" + decode_entities(risks.strip())
    assert len(business) > 15000 and len(risks) > 60000  # the real sizes: this is why a filing is not one chunk


def test_the_metadata_of_a_filing_has_exactly_the_keys_of_a_news_article_so_everything_downstream_treats_them_alike():
    news_keys = set(load_news(Path(__file__).parent / "fixtures" / "news_sample.txt")[0].metadata)
    press_keys = set(load_press_releases(Path(__file__).parent / "fixtures" / "press_releases_sample.txt")[0].metadata)
    for doc in load_filings(FIXTURE):
        assert set(doc.metadata) == news_keys == press_keys


def test_the_values_of_the_metadata_of_a_real_filing():
    doc = {d.id: d for d in load_filings(FIXTURE)}[CEO_8K]
    assert doc.metadata == {
        "article_id": CEO_8K,
        "headline": "AAPL 8-K, filed 2026-04-20",
        "tickers": ["AAPL"],
        "published_utc": "2026-04-20T00:00:00Z",  # the file has a date and no time of day: midnight UTC
        "published_ts": int(datetime(2026, 4, 20, tzinfo=timezone.utc).timestamp()),
        "publisher": "SEC filing",
        "api": "polygon",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/0001140361-26-015711.txt",
    }
    assert "Tim Cook will transition from his role as Chief Executive Officer to Executive Chair" in doc.page_content


def test_a_filing_without_text_is_one_short_document_that_says_the_text_is_not_stored_and_names_the_form_in_plain_words():
    by_id = {d.id: d for d in load_filings(FIXTURE)}
    assert by_id[TEN_Q].page_content == ("AAPL 10-Q, filed 2026-01-30\n\nForm 10-Q (quarterly report) filed 2026-01-30. "
                                         "The text of this filing is not stored: only its form, date and link are known.")
    assert "proxy statement" in by_id[PROXY].page_content and "notice of a planned sale of restricted shares by an insider" in by_id[FORM_144].page_content
    for no_text in (TEN_Q, PROXY, FORM_144):
        assert "The text of this filing is not stored" in by_id[no_text].page_content and by_id[no_text].metadata["publisher"] == "SEC filing"


@pytest.mark.parametrize("form, name", [("10-K", "annual report"), ("10-Q", "quarterly report"), ("8-K", "current report of a material event"),
                                        ("144/A", "notice of a planned sale of restricted shares by an insider (insider selling), amended"),
                                        ("SCHEDULE 13G/A", "report by an investor who holds more than 5% of the shares, as a passive holder, amended"),
                                        ("S-1", ""), ("", ""), ("/A", "")])
def test_form_names(form, name):
    assert form_name(form) == name


def test_every_form_the_toolbox_fetches_has_a_plain_name_so_no_filing_without_text_is_unsearchable():
    """The loader never imports the toolbox, so this is what ties its table of form names to the toolbox's list of forms."""
    assert [f for f in data_toolbox.FILING_FORMS if not form_name(f)] == []
    assert all(base in FORM_NAMES for base in {f.removesuffix("/A") for f in data_toolbox.FILING_FORMS})


def test_a_form_that_is_not_in_the_table_still_loads_and_just_has_no_plain_name(tmp_path):
    docs = load_text(tmp_path, NO_TEXT.replace("form: 8-K", "form: S-1"))
    assert docs[0].page_content.endswith("Form S-1 filed 2025-10-30. The text of this filing is not stored: only its form, date and link are known.")


def test_the_source_text_is_kept_as_the_filing_gave_it():
    doc = {d.id: d for d in load_filings(FIXTURE)}[CEO_8K]
    assert "Item 5.02Departure of Directors" in doc.page_content  # the filing's own missing space after the item number is not repaired


def test_html_entities_are_decoded_like_in_news(tmp_path):
    doc = load_text(tmp_path, GOOD.replace("Item 2.02    Results.", "Procter &amp; Gamble&#39;s results."))[0]
    assert doc.page_content.endswith("Procter & Gamble's results.")


def test_a_section_that_repeats_gets_a_number_so_two_blocks_never_share_an_id(tmp_path):
    """An 8-K can have two blocks of items (the toolbox appends one section per matching API row)."""
    docs = load_text(tmp_path, GOOD.replace("Item 2.02    Results.\n", "Item 2.02    Results.\n\n== items ==\nItem 9.01    Exhibits.\n"))
    assert [d.id for d in docs] == ["sec:0000320193-25-000077:items", "sec:0000320193-25-000077:items_2"]
    assert docs[1].page_content.endswith("Item 9.01    Exhibits.") and docs[1].metadata["headline"] == "AAPL 8-K, items 2, filed 2025-10-30"


def test_an_unknown_section_name_is_kept_and_its_underscores_become_spaces_in_the_headline(tmp_path):
    doc = load_text(tmp_path, GOOD.replace("== items ==", "== management_discussion =="))[0]
    assert doc.id == "sec:0000320193-25-000077:management_discussion" and doc.metadata["headline"] == "AAPL 8-K, management discussion, filed 2025-10-30"


def test_a_record_whose_sections_are_all_empty_is_treated_as_having_no_text(tmp_path):
    docs = load_text(tmp_path, GOOD.replace("Item 2.02    Results.\n", ""))
    assert [d.id for d in docs] == ["sec:0000320193-25-000077"] and "The text of this filing is not stored" in docs[0].page_content


def test_two_filings_in_a_file_load_in_file_order(tmp_path):
    docs = load_text(tmp_path, GOOD + NO_TEXT.replace("0000320193-25-000077", "0000320193-26-000006").replace("form: 8-K", "form: 10-Q"))
    assert [d.id for d in docs] == ["sec:0000320193-25-000077:items", "sec:0000320193-26-000006"]


# ---------------------------------------------------------------- one filing twice: the text fetched from EDGAR later (design choices entry 36)

WITH_TEXT = GOOD.replace("form: 8-K", "form: 10-Q").replace("== items ==\nItem 2.02    Results.", "== mdna ==\nItem 2. Management's Discussion\nthe discussion.\n\n== controls ==\nItem 4. Controls\nthe controls.")
TEN_Q_NO_TEXT = NO_TEXT.replace("form: 8-K", "form: 10-Q")


def accession(n: int, record: str) -> str:
    return record.replace("0000320193-25-000077", f"0000320193-25-{n:06d}")


def test_a_record_with_text_that_comes_after_the_same_filing_without_text_replaces_it(tmp_path):
    docs = load_text(tmp_path, TEN_Q_NO_TEXT + WITH_TEXT)
    assert [d.id for d in docs] == ["sec:0000320193-25-000077:mdna", "sec:0000320193-25-000077:controls"]
    assert docs[0].metadata["headline"] == "AAPL 10-Q, Management's Discussion and Analysis, filed 2025-10-30"
    assert "The text of this filing is not stored" not in "".join(d.page_content for d in docs)


def test_a_record_without_text_never_replaces_one_with_text(tmp_path):
    docs = load_text(tmp_path, WITH_TEXT + TEN_Q_NO_TEXT)
    assert [d.id for d in docs] == ["sec:0000320193-25-000077:mdna", "sec:0000320193-25-000077:controls"]


def test_of_two_records_with_text_the_last_wins(tmp_path):
    later = WITH_TEXT.replace("the discussion.", "the NEWER discussion.")
    docs = load_text(tmp_path, WITH_TEXT + later)
    assert [d.id for d in docs] == ["sec:0000320193-25-000077:mdna", "sec:0000320193-25-000077:controls"] and "NEWER" in docs[0].page_content


def test_the_documents_come_in_the_order_the_filings_first_appear_in_the_file(tmp_path):
    text = accession(1, TEN_Q_NO_TEXT) + accession(2, GOOD) + accession(1, WITH_TEXT)  # filing 1 (no text), filing 2, then the text of filing 1
    assert [d.id for d in load_text(tmp_path, text)] == ["sec:0000320193-25-000001:mdna", "sec:0000320193-25-000001:controls", "sec:0000320193-25-000002:items"]


def test_two_different_filings_are_never_merged(tmp_path):
    assert len(load_text(tmp_path, accession(1, TEN_Q_NO_TEXT) + accession(2, TEN_Q_NO_TEXT))) == 2


def test_a_broken_second_record_still_names_its_line_after_the_first_was_read(tmp_path):
    second = len(TEN_Q_NO_TEXT.splitlines()) + 1
    with pytest.raises(ValueError, match=rf"AAPL\.txt:{second}: bad filed"):
        load_text(tmp_path, TEN_Q_NO_TEXT + WITH_TEXT.replace("filed: 2025-10-30", "filed: nonsense"))


def test_every_section_name_the_edgar_module_can_produce_has_a_label_for_its_headline():
    """The loader never imports the EDGAR module, so this is what ties its labels to its names."""
    assert [name for name in edgar_text.ITEM_NAMES.values() if name not in SECTION_LABELS] == []
    assert SECTION_LABELS["risk_factors"] == "Risk Factors"  # the 10-K's section and Part II Item 1A of a 10-Q share the name and the label


# ---------------------------------------------------------------- the loader, on broken input (it names the file and line)

@pytest.mark.parametrize("text, message", [
    (GOOD.replace("<<<END>>>\n", ""), "the record has no <<<END>>>"),
    (GOOD + GOOD.replace("0000320193-25-000077", "0000320193-25-000078").replace("<<<END>>>\n", "") + GOOD, "a new record starts before the one at line"),
    ("stray text\n" + GOOD, "text outside a record"),
    (GOOD.replace("<<<TEXT>>>", "<<<BODY>>>"), "unknown marker '<<<BODY>>>'"),
    (GOOD.replace("<<<FILING>>>", "<<<PRESS_RELEASE>>>"), "text outside a record: '<<<PRESS_RELEASE>>>'"),  # a press release record is not read as a filing
    (GOOD.replace("form: 8-K", "form 8-K"), "header line without a colon"),
    (GOOD.replace("accession: 0000320193-25-000077\n", ""), "the record is missing accession"),
    (GOOD.replace("symbol: AAPL\n", ""), "the record is missing symbol"),
    (GOOD.replace("form: 8-K\n", ""), "the record is missing form"),
    (GOOD.replace("filed: 2025-10-30\n", ""), "the record is missing filed"),
    (GOOD.replace("url: https://www.sec.gov/Archives/edgar/data/320193/0000320193-25-000077.txt\n", ""), "the record is missing url"),
    (GOOD.replace("text_status: ok\n", ""), "the record is missing text_status"),
    (GOOD.replace("0000320193-25-000077\nsymbol", "../../.env\nsymbol"), "bad accession '../../.env'"),
    (GOOD.replace("0000320193-25-000077\nsymbol", "0000320193-25-00007\nsymbol"), "bad accession '0000320193-25-00007'"),
    (GOOD.replace("filed: 2025-10-30", "filed: yesterday"), "bad filed 'yesterday'"),
    (GOOD.replace("filed: 2025-10-30", "filed: 2025-10-30T10:00:00Z"), "bad filed '2025-10-30T10:00:00Z'"),
    (GOOD.replace("period_end: \n", "period_end: soon\n"), "bad period_end 'soon'"),
    (GOOD.replace("text_status: ok", "text_status: maybe"), "bad text_status 'maybe'"),
    (GOOD.replace("text_status: ok", "text_status: not_available"), "text_status is not_available but the record has text"),
    (NO_TEXT.replace("text_status: not_available", "text_status: ok"), "text_status is ok but the record has no text"),
    (GOOD.replace("== items ==\n", ""), "the text does not start with a section line"),
    (GOOD.replace("symbol: AAPL", "symbol: ../../.env"), "not a valid ticker symbol"),
])
def test_a_record_the_toolbox_would_not_have_written_stops_the_load_with_the_file_and_line(tmp_path, text, message):
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        load_text(tmp_path, text)
    assert "AAPL.txt:" in str(error.value)


def test_the_line_in_the_message_is_the_line_where_the_bad_record_starts(tmp_path):
    second = len(GOOD.splitlines()) + 1  # the second record starts on the line after the first one's <<<END>>>
    with pytest.raises(ValueError, match=rf"AAPL\.txt:{second}: bad filed"):
        load_text(tmp_path, GOOD + GOOD.replace("filed: 2025-10-30", "filed: nonsense"))


def test_bytes_that_are_not_utf8_name_the_line(tmp_path):
    path = tmp_path / "AAPL.txt"
    path.write_bytes(GOOD.encode().replace(b"Results.", b"Res\xffults."))
    line = GOOD.splitlines().index("Item 2.02    Results.") + 1
    with pytest.raises(ValueError, match=f"AAPL.txt:{line}: not valid UTF-8"):
        load_filings(path)


def test_an_empty_file_is_no_filings_and_a_byte_order_mark_is_ignored(tmp_path):
    assert load_text(tmp_path, "") == []
    assert len(load_text(tmp_path, "\ufeff" + GOOD)) == 1


def test_a_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_filings(tmp_path / "nothing.txt")


def test_the_press_release_loader_still_refuses_a_filing_record_and_still_reads_its_own(tmp_path):
    """The shared record parser takes the start marker as an argument: one format is never read as the other, in either direction."""
    path = tmp_path / "x.txt"
    path.write_text(GOOD, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown marker '<<<FILING>>>'|text outside a record"):
        load_press_releases(path)
    assert len(load_press_releases(Path(__file__).parent / "fixtures" / "press_releases_sample.txt")) == 3


# ---------------------------------------------------------------- where the files are

def test_the_filings_folder_defaults_to_where_the_toolbox_always_wrote(monkeypatch):
    monkeypatch.delenv("FILINGS_OUTPUT_DIR", raising=False)
    monkeypatch.setattr(config, "dotenv_values", lambda path: {})  # whatever .env says, the default is the toolbox's folder
    assert config.filings_dir() == config.REPO_ROOT / "data" / "filings"


def test_the_filings_folder_can_be_set_by_the_environment_and_a_relative_path_is_from_the_repo_root(monkeypatch, tmp_path):
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(tmp_path / "f"))
    assert config.filings_dir() == tmp_path / "f"
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", "somewhere/f")
    assert config.filings_dir() == config.REPO_ROOT / "somewhere" / "f"


def test_the_file_of_a_symbol_is_the_symbol_and_the_symbol_is_checked_first(monkeypatch, tmp_path):
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(tmp_path))
    assert filing_file("aapl") == tmp_path / "AAPL.txt" and filing_file(" brk.b ") == tmp_path / "BRK.B.txt"
    for bad in ("../../.env", "A B", "", "TOOLONGSYMBOL1"):
        with pytest.raises(ValueError):
            filing_file(bad)


def test_the_loader_command_line_shows_what_a_filing_file_becomes(capsys):
    assert loaders_main([str(FIXTURE), "--filings", "--show", "1"]) == 0
    out = capsys.readouterr().out
    assert "7 filing documents | 2025-10-30T00:00:00Z to 2026-04-20T00:00:00Z | 0 with insights | 7 distinct ids" in out
    assert "AAPL 8-K, filed 2025-10-30" in out and EIGHT_K in out


def test_the_loader_command_line_reports_a_broken_file_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "AAPL.txt"
    bad.write_text(GOOD.replace("<<<END>>>\n", ""), encoding="utf-8")
    assert loaders_main([str(bad), "--filings"]) == 1
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err and "Traceback" not in captured.err


def test_the_loader_command_line_says_so_for_a_file_with_no_filings(tmp_path, capsys):
    empty = tmp_path / "ZZZ.txt"
    empty.write_text("", encoding="utf-8")
    assert loaders_main([str(empty), "--filings"]) == 0 and "no filings" in capsys.readouterr().out


def test_the_loader_command_line_takes_one_kind_at_a_time():
    with pytest.raises(SystemExit):
        loaders_main(["AAPL", "--filings", "--press-releases"])


# ---------------------------------------------------------------- three collections in one folder

def test_the_collection_of_filings_has_its_own_name_and_the_others_keep_theirs():
    model = "sentence-transformers/all-MiniLM-L6-v2"
    assert rag_store.collection_name(model, kind="filings") == "filings-sentence-transformers-all-MiniLM-L6-v2"
    assert rag_store.collection_name(model) == "news-sentence-transformers-all-MiniLM-L6-v2"  # unchanged: no re-ingest
    assert rag_store.collection_name(model, kind="press_releases") == "press_releases-sentence-transformers-all-MiniLM-L6-v2"
    assert rag_store.KINDS == ("news", "press_releases", "filings")


@pytest.fixture
def folder_store(tmp_path, monkeypatch):
    """The real store functions over a temporary folder and the fake word embedder (no model download)."""
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "vectors"))
    monkeypatch.setattr(rag_store, "get_embeddings", lambda: KeywordEmbeddings())
    return rag_store


def test_each_kind_has_its_own_collection_so_a_count_or_a_reset_of_one_never_touches_the_others(folder_store):
    stores = {kind: folder_store.get_vector_store(kind) for kind in ("news", "press_releases", "filings")}
    stores["news"].add_texts(["apple iphone"], ids=["n#0"], metadatas=[{"article_id": "n"}])
    stores["press_releases"].add_texts(["apple iphone", "nvidia chip"], ids=["p#0", "q#0"], metadatas=[{"article_id": "p"}, {"article_id": "q"}])
    stores["filings"].add_texts(["apple iphone", "nvidia chip", "tesla car"], ids=["f#0", "g#0", "h#0"], metadatas=[{"article_id": c} for c in "fgh"])
    assert [folder_store.count_chunks(k) for k in ("news", "press_releases", "filings")] == [1, 2, 3]
    assert folder_store.reset_collection("filings") is True
    assert [folder_store.count_chunks(k) for k in ("news", "press_releases", "filings")] == [1, 2, 0]
    assert folder_store.reset_collection("filings") is False  # nothing left to delete


def test_the_store_command_line_can_show_the_filings_collection(monkeypatch, capsys):
    monkeypatch.setattr(rag_store, "count_chunks", lambda kind="news": 0)
    assert rag_store.main(["--filings"]) == 0
    out = capsys.readouterr().out
    assert "collection : filings-" in out and "ingest --filings AAPL" in out
    with pytest.raises(SystemExit):
        rag_store.main(["--filings", "--press-releases"])


def test_the_store_module_run_as_a_command_passes_its_filings_flag_on(tmp_path):
    """Run for real (no model: an empty folder shows 'nothing is stored yet'): `python -m prometheus.rag.store --filings` must show the filings collection."""
    import os, subprocess, sys
    env = {**os.environ, "VECTOR_STORE_PATH": str(tmp_path / "empty"), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-m", "prometheus.rag.store", "--filings"], capture_output=True, text=True, env=env, cwd=config.REPO_ROOT)
    assert done.returncode == 0 and "collection : filings-" in done.stdout, done.stdout + done.stderr


# ---------------------------------------------------------------- ingest

@pytest.fixture
def filings_store(tmp_path):
    """The filings collection over the fake word embedder, with the SAME search settings as the real store (`rag_store.HNSW`: thorough). With Chroma's defaults the
    approximate search returned 6 or 7 of the 7 articles for the same question on different fresh stores (found by running these tests repeatedly, entries 10 and 20 say
    why): the fixture holds a 10-K of 160 near-identical chunks, and a test that sometimes fails is worth nothing."""
    return Chroma(client=chromadb.PersistentClient(path=str(tmp_path / "chroma_f"), settings=Settings(anonymized_telemetry=False)), collection_name="test",
                  embedding_function=DistinctEmbeddings(), collection_configuration={"hnsw": rag_store.HNSW})


@pytest.fixture
def folder(monkeypatch, tmp_path):
    folder = tmp_path / "filings"
    folder.mkdir()
    (folder / "AAPL.txt").write_bytes(FIXTURE.read_bytes())
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(folder))
    return folder


def test_ingest_filings_chunks_and_stores_them_and_is_safe_to_repeat(filings_store, folder):
    first = ingest_filings("aapl", store=filings_store)
    assert (first["symbol"], first["articles"]) == ("AAPL", 7) and first["path"] == str(folder / "AAPL.txt")
    assert first["chunks"] > 150 and first["added"] == first["chunks"]  # the 10-K alone is about 150 chunks of 800 characters
    got = filings_store.get(include=["metadatas", "documents"])
    assert len(got["ids"]) == first["chunks"] and all(i.startswith("sec:") for i in got["ids"])
    assert all(m["publisher"] == "SEC filing" and m["tickers"] == ["AAPL"] for m in got["metadatas"])
    again = ingest_filings("AAPL", store=filings_store)
    assert (again["added"], again["updated"], again["unchanged"], again["deleted"]) == (0, 0, first["chunks"], 0)


def test_a_10_K_section_is_one_article_with_many_chunks_numbered_from_zero_and_each_later_chunk_says_what_it_belongs_to(filings_store, folder):
    ingest_filings("AAPL", store=filings_store)
    got = filings_store.get(where={"article_id": RISKS}, include=["metadatas", "documents"])
    count = len(got["ids"])
    assert count > 100 and sorted(got["ids"]) == sorted(f"{RISKS}#{n}" for n in range(count))
    assert {m["chunks"] for m in got["metadatas"]} == {count}
    by_number = {m["chunk"]: text for m, text in zip(got["metadatas"], got["documents"])}
    headline = "AAPL 10-K, Risk Factors, filed 2025-10-31, period ending 2025-09-27"
    assert by_number[0].startswith(headline) and all(by_number[n].startswith(headline) for n in (1, 50, count - 1))  # the headline is put back in front


def test_a_filing_without_text_is_one_chunk_and_an_8_K_of_a_thousand_characters_is_more_than_one(filings_store, folder):
    """Found by the first run of this test: a real 8-K of 924 characters is 3 chunks (the chunker is 800 characters with an overlap), not 1."""
    ingest_filings("AAPL", store=filings_store)
    for article in (TEN_Q, PROXY, FORM_144):
        assert len(filings_store.get(where={"article_id": article})["ids"]) == 1, article
    assert len(filings_store.get(where={"article_id": EIGHT_K})["ids"]) > 1


def test_ingesting_filings_does_not_touch_the_news_or_press_release_collections(folder_store, folder, monkeypatch):
    monkeypatch.setattr(ingest, "count_too_long", lambda texts, embeddings=None: 0)  # the real check needs the real model; this test uses the word embedder
    news, press = folder_store.get_vector_store("news"), folder_store.get_vector_store("press_releases")
    news.add_texts(["apple iphone"], ids=["n#0"], metadatas=[{"article_id": "n"}])
    press.add_texts(["apple iphone"], ids=["p#0"], metadatas=[{"article_id": "p"}])
    stats = ingest_filings("AAPL")  # the real function, with the real store functions over the temporary folder
    assert folder_store.count_chunks("filings") == stats["chunks"] > 100
    assert (folder_store.count_chunks("news"), folder_store.count_chunks("press_releases")) == (1, 1)


def test_a_symbol_with_an_empty_file_ingests_nothing_and_does_not_fail(filings_store, folder):
    """The real toolbox creates an empty file when the API returns no filings."""
    (folder / "ZZZ.txt").write_text("", encoding="utf-8")
    stats = ingest_filings("ZZZ", store=filings_store)
    assert (stats["articles"], stats["chunks"], stats["added"]) == (0, 0, 0)


def test_a_symbol_with_no_file_is_a_file_not_found_error(filings_store, folder):
    with pytest.raises(FileNotFoundError):
        ingest_filings("NVDA", store=filings_store)


def test_to_chunks_gives_the_filings_the_same_chunker_as_news():
    chunks = to_chunks(load_filings(FIXTURE))
    assert max(len(c.page_content) for c in chunks) < 900  # 800 characters, plus the headline put in front of a later chunk (the news rule)


@pytest.fixture
def ingest_cli(monkeypatch, folder, filings_store):
    (folder / "META.txt").write_bytes(FIXTURE.read_bytes().replace(b"symbol: AAPL", b"symbol: META"))
    monkeypatch.setattr(ingest, "get_vector_store", lambda kind="news": filings_store if kind == "filings" else pytest.fail("another store must not be used"))
    monkeypatch.setattr(ingest, "count_too_long", lambda texts: 0)
    monkeypatch.setattr(ingest, "count_chunks", lambda kind="news": len(filings_store.get()["ids"]))
    return folder


def test_the_ingest_command_line_reads_filing_files_into_the_filings_collection(ingest_cli, capsys):
    assert ingest.main(["--filings", "AAPL"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"AAPL: 7 filing documents -> \d+ chunks \| added \d+, updated 0, unchanged 0, deleted 0", out) and "the collection now holds" in out


def test_all_with_filings_takes_every_file_of_the_filings_folder_and_rebuild_clears_only_the_filings_collection(ingest_cli, monkeypatch, capsys):
    resets = []
    monkeypatch.setattr(ingest, "reset_collection", lambda kind="news": resets.append(kind) or True)
    assert ingest.main(["--filings", "--rebuild", "--all"]) == 0
    out = capsys.readouterr().out
    assert resets == ["filings"] and "AAPL: 7 filing documents" in out and "META: 7 filing documents" in out and out.count("filing documents ->") == 2


def test_the_ingest_command_line_says_when_there_are_no_filing_files(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(tmp_path))
    assert ingest.main(["--filings", "--all"]) == 1
    assert "ERROR: no filing files in" in capsys.readouterr().err


def test_the_ingest_command_line_takes_one_kind_at_a_time():
    with pytest.raises(SystemExit):
        ingest.main(["--filings", "--press-releases", "AAPL"])


# ---------------------------------------------------------------- search

@pytest.fixture
def stored(filings_store, folder):
    ingest_filings("AAPL", store=filings_store)
    return filings_store


def test_search_filings_is_search_news_on_the_filings_collection(stored):
    options = dict(tickers="AAPL", k=20, min_score=None, store=stored)
    same = [(h.chunk_id, h.score) for h in search_news("Apple", **options)]
    assert [(h.chunk_id, h.score) for h in search_filings("Apple", **options)] == same and len(same) == 7  # one hit per article: the 10-K gives two


def test_a_hit_is_a_news_shaped_hit_with_the_filing_publisher_and_the_symbol(stored):
    hits = {h.article_id: h for h in search_filings("Apple", tickers="AAPL", k=20, min_score=None, store=stored)}
    hit = hits[CEO_8K]
    assert hit.publisher == "SEC filing" and hit.tickers == ["AAPL"] and hit.chunk_id == f"{CEO_8K}#{hit.chunk - 1}" and 1 <= hit.chunk <= hit.chunks
    assert hit.url == "https://www.sec.gov/Archives/edgar/data/320193/0001140361-26-015711.txt" and hit.published_utc == "2026-04-20T00:00:00Z"
    assert hit.headline == "AAPL 8-K, filed 2026-04-20"


def test_a_10_K_gives_one_hit_per_section_not_one_hit_per_chunk(stored):
    hits = search_filings("Apple", tickers="AAPL", k=20, min_score=None, store=stored)
    assert sorted(h.article_id for h in hits if h.article_id.startswith("sec:0000320193-25-000079")) == [BUSINESS, RISKS]
    risks = next(h for h in hits if h.article_id == RISKS)
    assert risks.chunks > 100 and 1 <= risks.chunk <= risks.chunks  # the best of its chunks, and the hit says which of how many


def test_the_ticker_and_window_filters_and_newest_first_work_on_filings(stored):
    assert search_filings("Apple", tickers="NVDA", min_score=None, store=stored) == []
    dates = [h.published_utc[:10] for h in search_filings("Apple", tickers="AAPL", k=20, min_score=None, newest_first=True, store=stored)]
    assert dates == ["2026-04-20", "2026-04-02", "2026-01-30", "2026-01-08", "2025-10-31", "2025-10-31", "2025-10-30"]  # the real filed dates, newest first
    now = datetime(2026, 4, 25, tzinfo=timezone.utc)
    recent = search_filings("Apple", tickers="AAPL", since_days=10, k=20, min_score=None, store=stored, now=now)
    assert {h.published_utc[:10] for h in recent} == {"2026-04-20"}  # the Apr 2 Form 144 is outside ten days of Apr 25


def test_bad_search_arguments_are_refused_for_filings_as_for_news(stored):
    for bad in (dict(k=0), dict(k=21), dict(since_days=0), dict(tickers="../../.env")):
        with pytest.raises(ValueError):
            search_filings("Apple", store=stored, **bad)
    with pytest.raises(ValueError):
        search_filings("   ", store=stored)


def test_the_press_release_search_does_not_see_filings_and_the_filings_search_does_not_see_press_releases(stored, tmp_path):
    other = Chroma(client=chromadb.PersistentClient(path=str(tmp_path / "chroma_other"), settings=Settings(anonymized_telemetry=False)), collection_name="other",
                   embedding_function=DistinctEmbeddings(), collection_configuration={"hnsw": rag_store.HNSW})  # an empty press release collection
    assert search_press_releases("Apple", tickers="AAPL", min_score=None, store=other) == []
    assert len(search_filings("Apple", tickers="AAPL", k=20, min_score=None, store=stored)) == 7


def test_the_retriever_command_line_can_search_the_filings(stored, monkeypatch, capsys):
    monkeypatch.setattr(retriever, "get_vector_store", lambda kind="news": stored if kind == "filings" else pytest.fail("the news store must not be used"))
    monkeypatch.setattr(retriever, "count_chunks", lambda kind="news": 100 if kind == "filings" else 0)
    assert retriever.main(["Apple", "--tickers", "AAPL", "--filings", "--min-score", "0", "-k", "20", "--newest-first"]) == 0
    out = capsys.readouterr().out
    assert "SEC filing" in out and "AAPL 8-K, filed 2026-04-20" in out and "https://www.sec.gov/Archives/edgar/data/320193/" in out


def test_the_retriever_command_line_tells_you_how_to_ingest_filings_when_there_are_none(monkeypatch, capsys):
    monkeypatch.setattr(retriever, "count_chunks", lambda kind="news": 0)
    assert retriever.main(["Apple", "--filings"]) == 1
    assert "ingest --filings --all" in capsys.readouterr().err


# ---------------------------------------------------------------- the toolbox writes <SYMBOL>.txt in the filings folder

def massive_answers_from_the_real_records():
    """What the three Massive endpoints return, filled from the six real records of the fixture. Nothing is made up: each value is one that was in the file.
    /vX/index lists every filing of a form; /10-K/vX/sections has one row per section; /8-K/vX/text has the items text of an 8-K."""
    index, sections, items = [], [], []
    for accession, (head, text) in real_records().items():
        index.append({"accession_number": accession, "form_type": head["form"], "filing_date": head["filed"], "filing_url": head["url"]})
        if head["form"] == "10-K":
            parts = re.split(r"(?m)^== (\w+) ==\n", text)
            for name, body in zip(parts[1::2], parts[2::2]):
                sections.append({"filing_url": head["url"], "period_end": head["period_end"], "section": name, "text": body.strip()})
        if head["form"] == "8-K":
            items.append({"accession_number": accession, "items_text": text.removeprefix("== items ==\n")})
    return index, sections, items


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    asked = []
    index, sections, items = massive_answers_from_the_real_records()

    def fake_get_json(client, limiter, name, url, params, headers=None):
        asked.append((name, url, params))
        if url.endswith("/vX/index"):
            return {"results": [r for r in index if r["form_type"] == params["form_type"]]}
        if url.endswith("/10-K/vX/sections"):
            return {"results": sections}
        if url.endswith("/8-K/vX/text"):
            return {"results": items}
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(tmp_path / "fl"))
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key-not-real")
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(fake_get_json))
    box = DataToolbox(base_dir=tmp_path)
    box.asked = asked
    return box


def test_the_toolbox_writes_symbol_dot_txt_in_the_filings_folder_setting(toolbox, tmp_path, capsys):
    stats = toolbox.fetch_filings("aapl", from_date=400)
    capsys.readouterr()
    path = tmp_path / "fl" / "AAPL.txt"
    assert stats == {"symbol": "AAPL", "path": str(path), "fetched": 6, "written": 6, "duplicates": 0} and path.exists()
    assert not (tmp_path / "data" / "filings").exists()  # not the default folder: the setting was honoured
    assert all(url.startswith(data_toolbox.FILINGS_URL) and "apiKey" in params for _, url, params in toolbox.asked)


def test_the_toolbox_folder_defaults_to_data_filings_under_its_base_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("FILINGS_OUTPUT_DIR", raising=False)
    assert DataToolbox(base_dir=tmp_path)._filings_dir(env={}) == tmp_path / "data" / "filings"
    assert DataToolbox(base_dir=tmp_path)._filings_dir(env={"FILINGS_OUTPUT_DIR": "elsewhere"}) == tmp_path / "elsewhere"


def test_what_the_toolbox_writes_is_what_the_loader_reads_and_it_reads_the_same_documents_as_the_real_fixture(toolbox, tmp_path, capsys):
    toolbox.fetch_filings("AAPL")
    capsys.readouterr()
    written = {d.id: d for d in load_filings(tmp_path / "fl" / "AAPL.txt")}
    real = {d.id: d for d in load_filings(FIXTURE)}
    assert sorted(written) == sorted(real) == sorted(FIXTURE_IDS)  # the file order differs (the toolbox sorts by date), the Documents do not
    for article_id, doc in real.items():
        assert written[article_id].page_content == doc.page_content and written[article_id].metadata == doc.metadata, article_id


def test_fetching_again_appends_nothing_so_the_file_never_gets_a_duplicate(toolbox, tmp_path, capsys):
    toolbox.fetch_filings("AAPL")
    before = (tmp_path / "fl" / "AAPL.txt").read_bytes()
    again = toolbox.fetch_filings("AAPL")
    capsys.readouterr()
    assert (again["fetched"], again["written"], again["duplicates"]) == (6, 0, 6)
    assert (tmp_path / "fl" / "AAPL.txt").read_bytes() == before


def test_the_toolbox_still_checks_its_arguments_and_reaches_the_api_for_nothing_that_is_refused(toolbox):
    for bad in (dict(symbol=""), dict(symbol="AAPL", from_date=0), dict(symbol="AAPL", from_date=True)):
        with pytest.raises(ValueError):
            toolbox.fetch_filings(**bad)
    assert toolbox.asked == []


def test_keyword_search_still_finds_the_filings_in_the_folder_the_fetch_wrote(toolbox, tmp_path, capsys, monkeypatch):
    toolbox.fetch_filings("AAPL")
    capsys.readouterr()
    monkeypatch.setattr(data_toolbox, "datetime", type("D", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 5, 1, tzinfo=timezone.utc))}))
    found = toolbox.search("AAPL", ["Ternus"], from_date=60, refresh=False)
    capsys.readouterr()
    assert [(f["form"], f["filed"]) for f in found["filings"]] == [("8-K", "2026-04-20")]  # the same reader, now on the setting's folder
