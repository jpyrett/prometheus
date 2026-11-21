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
from prometheus.rag.ingest import ingest_filings
from prometheus.rag.loaders import SECTION_LABELS, load_filings
from prometheus.rag.retriever import search_filings, search_news
from test_rag_retriever import KeywordEmbeddings

FIXTURE = Path(__file__).parent / "fixtures" / "filings_sample.txt"

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


# ---------------------------------------------------------------- one filing twice: the text fetched from EDGAR later (design choices entry 36)


def accession(n: int, record: str) -> str:
    return record.replace("0000320193-25-000077", f"0000320193-25-{n:06d}")


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


def test_bytes_that_are_not_utf8_name_the_line(tmp_path):
    path = tmp_path / "AAPL.txt"
    path.write_bytes(GOOD.encode().replace(b"Results.", b"Res\xffults."))
    line = GOOD.splitlines().index("Item 2.02    Results.") + 1
    with pytest.raises(ValueError, match=f"AAPL.txt:{line}: not valid UTF-8"):
        load_filings(path)


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


def test_all_with_filings_takes_every_file_of_the_filings_folder_and_rebuild_clears_only_the_filings_collection(ingest_cli, monkeypatch, capsys):
    resets = []
    monkeypatch.setattr(ingest, "reset_collection", lambda kind="news": resets.append(kind) or True)
    assert ingest.main(["--filings", "--rebuild", "--all"]) == 0
    out = capsys.readouterr().out
    assert resets == ["filings"] and "AAPL: 7 filing documents" in out and "META: 7 filing documents" in out and out.count("filing documents ->") == 2


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


def test_the_toolbox_still_checks_its_arguments_and_reaches_the_api_for_nothing_that_is_refused(toolbox):
    for bad in (dict(symbol=""), dict(symbol="AAPL", from_date=0), dict(symbol="AAPL", from_date=True)):
        with pytest.raises(ValueError):
            toolbox.fetch_filings(**bad)
    assert toolbox.asked == []
