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


def test_html_entities_are_decoded_like_in_news(tmp_path):
    doc = load_text(tmp_path, GOOD.replace("Item 2.02    Results.", "Procter &amp; Gamble&#39;s results."))[0]
    assert doc.page_content.endswith("Procter & Gamble's results.")


# ---------------------------------------------------------------- one filing twice: the text fetched from EDGAR later (design choices entry 36)

WITH_TEXT = GOOD.replace("form: 8-K", "form: 10-Q").replace("== items ==\nItem 2.02    Results.", "== mdna ==\nItem 2. Management's Discussion\nthe discussion.\n\n== controls ==\nItem 4. Controls\nthe controls.")
TEN_Q_NO_TEXT = NO_TEXT.replace("form: 8-K", "form: 10-Q")


def accession(n: int, record: str) -> str:
    return record.replace("0000320193-25-000077", f"0000320193-25-{n:06d}")


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


def test_the_press_release_loader_still_refuses_a_filing_record_and_still_reads_its_own(tmp_path):
    """The shared record parser takes the start marker as an argument: one format is never read as the other, in either direction (the filing loader still refuses a release with an error; the press release loader discards a filing record, entry 83)."""
    path = tmp_path / "x.txt"
    path.write_text(GOOD, encoding="utf-8")
    skipped = []
    assert load_press_releases(path, skipped) == [] and skipped and "text outside a record" in skipped[0].reason      # entry 83: not read as a release, and discarded rather than an error
    assert len(load_press_releases(Path(__file__).parent / "fixtures" / "press_releases_sample.txt")) == 3


# ---------------------------------------------------------------- where the files are


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


# ---------------------------------------------------------------- three collections in one folder


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


# ---------------------------------------------------------------- search

@pytest.fixture
def stored(filings_store, folder):
    ingest_filings("AAPL", store=filings_store)
    return filings_store


def test_bad_search_arguments_are_refused_for_filings_as_for_news(stored):
    for bad in (dict(k=0), dict(k=21), dict(since_days=0), dict(tickers="../../.env")):
        with pytest.raises(ValueError):
            search_filings("Apple", store=stored, **bad)
    with pytest.raises(ValueError):
        search_filings("   ", store=stored)


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
