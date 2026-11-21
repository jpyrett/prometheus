"""Tests for the text of a 10-Q taken from EDGAR (design choices entry 36): prometheus/agents/data_clerk/edgar_text.py and DataToolbox.fetch_filing_text.

Run:  pytest tests/test_edgar_text.py -q

No network, no key, no cost. The submission is a REAL one: tests/fixtures/edgar_aapl_10q_0000320193-26-000006.txt.gz is a byte-for-byte prefix of the complete submission
that EDGAR returned for AAPL's 10-Q filed 2026-01-30 (the SGML header, the 10-Q itself as an inline-XBRL page of 765,989 characters, and the three certifications; the other 56
documents of the 5.2 MB file are not in it), plus the file's real closing tag. The filing records come from tests/fixtures/filings_sample.txt (real records of data/filings/AAPL.txt,
one of them this 10-Q with `text_status: not_available`). The tests that break a page use tiny synthetic ones, as the loader tests do. Every folder is a temporary one.
"""
import gzip
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import httpx
import pytest
from chromadb.config import Settings
from langchain_chroma import Chroma

from prometheus.agents.data_clerk import data_toolbox, edgar_text
from prometheus.agents.data_clerk.data_toolbox import DataToolbox, _RateLimiter
from prometheus.agents.data_clerk.edgar_text import EdgarLayoutError, EdgarNotConfigured
from prometheus.rag import store as rag_store
from prometheus.tools.data_clerk import DataClerkTools
from test_filings import DistinctEmbeddings, massive_answers_from_the_real_records

SUBMISSION = gzip.decompress((Path(__file__).parent / "fixtures" / "edgar_aapl_10q_0000320193-26-000006.txt.gz").read_bytes()).decode("utf-8")
FILINGS = Path(__file__).parent / "fixtures" / "filings_sample.txt"
ACCESSION = "0000320193-26-000006"
URL = f"https://www.sec.gov/Archives/edgar/data/320193/{ACCESSION}.txt"
AGENT = "Jacob Pyrett jacob_pyrett@yahoo.com"


# ---------------------------------------------------------------- the real submission


def test_a_document_type_that_is_not_there_is_an_error_that_names_it():
    with pytest.raises(EdgarLayoutError, match="no document of type 10-K"):
        edgar_text.main_document(SUBMISSION, "10-K")


def test_only_the_forms_the_module_knows_are_taken():
    assert edgar_text.TEXT_FORMS == ("10-Q",)
    with pytest.raises(EdgarLayoutError, match="cannot be taken from EDGAR yet"):
        edgar_text.filing_sections(SUBMISSION, "EX-31.1")


# ---------------------------------------------------------------- the setting

@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(tmp_path / "filings"))
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    (tmp_path / "filings").mkdir()
    (tmp_path / "filings" / "AAPL.txt").write_bytes(FILINGS.read_bytes())
    monkeypatch.setattr(data_toolbox, "datetime", type("D", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 5, 1, tzinfo=timezone.utc))}))  # a fixed "today"
    return DataToolbox(base_dir=tmp_path)


@pytest.mark.parametrize("value", ["Prometheus", "no contact here", "Jacob Pyrett j\u00e4cob@example.com", "name@example.com\u0000", "   "])
def test_a_user_agent_without_a_contact_or_with_odd_characters_is_refused(box, value):
    with pytest.raises(EdgarNotConfigured):
        box._sec_user_agent(env={"SEC_USER_AGENT": value})


# ---------------------------------------------------------------- fetch_filing_text

@pytest.fixture
def downloads(box, monkeypatch):
    """EDGAR answered by the real fixture: every download is recorded and returns the submission's bytes."""
    asked = []

    def fake(client, limiter, name, url):
        asked.append((name, url, client.headers.get("user-agent")))
        return SUBMISSION.encode("utf-8")

    monkeypatch.setattr(DataToolbox, "_get_bytes", staticmethod(fake))
    monkeypatch.setenv("SEC_USER_AGENT", AGENT)
    box.asked = asked
    return box


def test_only_filings_of_the_asked_forms_are_candidates_and_other_forms_are_refused(downloads, capsys):
    assert downloads.fetch_filing_text("AAPL", forms=("10-Q",), from_date=400)["candidates"] == 1  # the proxy statement and the Form 144 of the fixture have no text either: not fetched
    capsys.readouterr()
    for forms in (("10-K",), ("DEF 14A",), (), ("10-Q", "144")):
        with pytest.raises(ValueError, match="forms must be some of 10-Q"):
            downloads.fetch_filing_text("AAPL", forms=forms)


def test_bad_arguments_are_refused_before_anything_is_read_or_sent(downloads):
    for kwargs in (dict(symbol=""), dict(symbol="  "), dict(symbol="AAPL", from_date=0), dict(symbol="AAPL", from_date=True), dict(symbol="AAPL", from_date="30")):
        with pytest.raises(ValueError):
            downloads.fetch_filing_text(**kwargs)
    assert downloads.asked == []


def test_a_symbol_whose_filings_were_never_fetched_is_a_file_not_found_error_that_says_what_to_do(downloads):
    with pytest.raises(FileNotFoundError, match="fetch the filings of NVDA first"):
        downloads.fetch_filing_text("NVDA")


def test_a_url_that_is_not_an_edgar_archive_url_is_never_sent_the_identity(downloads, tmp_path):
    path = tmp_path / "filings" / "AAPL.txt"
    path.write_bytes(path.read_bytes().replace(URL.encode(), b"https://evil.example.com/" + ACCESSION.encode() + b".txt"))
    with pytest.raises(ValueError, match="not an EDGAR archive url"):
        downloads.fetch_filing_text("AAPL", from_date=400)
    assert downloads.asked == []


def test_a_submission_for_another_accession_is_refused_and_nothing_is_written(downloads, tmp_path):
    path = tmp_path / "filings" / "AAPL.txt"
    path.write_bytes(path.read_bytes().replace(f"accession: {ACCESSION}".encode(), b"accession: 0000320193-26-000099"))  # the record says 99, the download is for 06
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match=f"downloaded submission is for accession {ACCESSION}"):
        downloads.fetch_filing_text("AAPL", from_date=400)
    assert path.read_bytes() == before


def test_a_page_with_no_items_stops_the_operation_with_the_accession_and_writes_nothing(downloads, tmp_path, monkeypatch):
    page = SUBMISSION.replace(edgar_text.main_document(SUBMISSION, "10-Q"), "<html><body><p>hello</p></body></html>")
    monkeypatch.setattr(DataToolbox, "_get_bytes", staticmethod(lambda client, limiter, name, url: page.encode()))
    before = (tmp_path / "filings" / "AAPL.txt").read_bytes()
    with pytest.raises(ValueError, match=f"{ACCESSION} \\(10-Q\\): no Item heading"):
        downloads.fetch_filing_text("AAPL", from_date=400)
    assert (tmp_path / "filings" / "AAPL.txt").read_bytes() == before


def test_the_keyword_search_lists_the_10_Q_once_and_with_its_text_after_the_text_was_fetched(downloads, capsys):
    def search(term):
        found = downloads.search("AAPL", [term], from_date=400, refresh=False)
        return [(f["form"], f["filed"], f["text_status"]) for f in found["filings"]], found["filings_in_window"]

    matches_before, in_window_before = search("Parekh")
    assert matches_before == []  # before: the 10-Q has no text, so a word from its text finds nothing
    downloads.fetch_filing_text("AAPL", from_date=400)
    matches_after, in_window_after = search("Parekh")
    capsys.readouterr()
    assert matches_after == [("10-Q", "2026-01-30", "ok")]  # once, with its text
    assert in_window_after == in_window_before == 6  # and the filing is not counted twice: the six real filings of the fixture are in the window


# ---------------------------------------------------------------- the download (a mock transport, so the real client code runs)

def mock_client(handler, **kwargs):
    return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)


def get(handler, url=URL):
    with mock_client(handler) as client:
        return DataToolbox._get_bytes(client, _RateLimiter(0), "EDGAR test", url)


def test_a_429_or_a_5xx_is_retried_with_a_pause_and_then_succeeds_or_gives_up(monkeypatch):
    pauses = []
    monkeypatch.setattr(data_toolbox.time, "sleep", pauses.append)
    answers = iter([429, 503, 200])
    assert get(lambda r: httpx.Response(next(answers), content=b"ok")) == b"ok" and pauses == [5, 10]
    with pytest.raises(RuntimeError, match="HTTP 503$"):
        get(lambda r: httpx.Response(503))
    assert len(pauses) == 2 + data_toolbox.API_MAX_ATTEMPTS - 1


def test_a_network_error_is_retried_and_then_reported_by_its_kind_only(monkeypatch):
    monkeypatch.setattr(data_toolbox.time, "sleep", lambda s: None)

    def broken(request):
        raise httpx.ConnectError("connection to sec.gov refused with a secret in it")

    with pytest.raises(RuntimeError, match="^EDGAR test request failed: ConnectError$"):
        get(broken)


def test_a_body_over_the_size_limit_is_refused(monkeypatch):
    monkeypatch.setattr(data_toolbox, "EDGAR_MAX_BYTES", 10)
    with pytest.raises(RuntimeError, match="over the limit of 10"):
        get(lambda r: httpx.Response(200, content=b"x" * 11))
    assert get(lambda r: httpx.Response(200, content=b"x" * 10)) == b"x" * 10


# ---------------------------------------------------------------- the whole path: the tool, Massive, EDGAR, the index (only the network is faked)

def test_fetch_filings_end_to_end_stores_the_filings_adds_the_10_Q_text_from_edgar_and_indexes_all_of_it(box, tmp_path, monkeypatch, capsys):
    """The real DataClerkTools, the real DataToolbox, the real loader and ingest and store code. Faked: the Massive answers (built from the six real records) and the EDGAR
    download (the real fixture). It proves the order of the steps: the Massive fetch writes the 10-Q with no text, EDGAR adds its text, and ONE index afterwards holds the text."""
    (tmp_path / "filings" / "AAPL.txt").unlink()  # the temporary folder starts empty: the fetch writes the file
    index, sections, items = massive_answers_from_the_real_records()

    def massive(client, limiter, name, url, params, headers=None):
        if url.endswith("/vX/index"):
            return {"results": [r for r in index if r["form_type"] == params["form_type"]]}
        return {"results": sections if url.endswith("/10-K/vX/sections") else items}

    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(massive))
    monkeypatch.setattr(DataToolbox, "_get_bytes", staticmethod(lambda client, limiter, name, url: SUBMISSION.encode("utf-8")))
    monkeypatch.setenv("SEC_USER_AGENT", AGENT)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key-not-real")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")  # the toolbox checks both keys before any fetch
    store = Chroma(client=chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False)), collection_name="test",
                   embedding_function=DistinctEmbeddings(), collection_configuration={"hnsw": rag_store.HNSW})
    result = DataClerkTools(box, filings_store=store).fetch_filings("aapl", from_date=400)
    capsys.readouterr()
    assert result["written"] == 6 and result["edgar_text"] == {"candidates": 1, "written": 1}
    assert result["indexed"]["articles"] == 7 - 1 + 11  # the 10-Q went from one short Document to eleven Items; the other six Documents are as before
    assert store.get(where={"article_id": f"sec:{ACCESSION}"})["ids"] == []  # the short "text not stored" Document is not in the index
    stored = store.get(where={"article_id": f"sec:{ACCESSION}:other_information"}, include=["documents", "metadatas"])
    assert stored["ids"] and "Kevan Parekh" in " ".join(stored["documents"]) and stored["metadatas"][0]["publisher"] == "SEC filing"
    again = DataClerkTools(box, filings_store=store).fetch_filings("AAPL", from_date=400)
    capsys.readouterr()
    assert again["written"] == 0 and again["edgar_text"] == {"candidates": 0, "written": 0} and again["indexed"]["added"] == 0 and again["indexed"]["updated"] == 0


# ---------------------------------------------------------------- Procter & Gamble's real 10-Q: the headings are two-cell table rows (found by the first real run of the Advisor)

PG_LINES = ["PART I | Item 1. | Financial Statements | 1", "Item 2. | Management's Discussion and Analysis of Financial Condition and Results of Operations | 14", "PART II | Item 1. | Legal Proceedings | 30",
            "PART I. FINANCIAL INFORMATION", "Item 1. | Financial Statements", "Consolidated Statements of Earnings", "Net sales | $ | 22,386 | $ | 21,740",
            "Item 2. | Management's Discussion and Analysis of Financial Condition and Results of Operations", "Overview", "Item 3. | Quantitative and Qualitative Disclosures About Market Risk",
            "Market risk is described in the 10-K.", "PART II. OTHER INFORMATION", "Item 1A. | Risk Factors", "No changes to the risk factors."]


def test_a_heading_written_as_a_two_cell_row_is_a_heading_and_the_table_of_contents_is_not():
    from prometheus.agents.data_clerk.edgar_text import _two_cell_heading, split_items
    assert _two_cell_heading("Item 1. | Financial Statements") == "Item 1. Financial Statements" and _two_cell_heading("Item 1A. | Risk Factors") == "Item 1A. Risk Factors"
    assert _two_cell_heading("Item 1. | Financial Statements | 1") == "Item 1. | Financial Statements | 1"                            # a page number in a third cell: the table of contents
    assert _two_cell_heading("PART I | Item 1. | Financial Statements | 1") == "PART I | Item 1. | Financial Statements | 1"
    assert _two_cell_heading("Item 1. | 12") == "Item 1. | 12" and _two_cell_heading("Net sales | $") == "Net sales | $"
    sections = dict(split_items(PG_LINES))
    assert list(sections) == ["financial_statements", "mdna", "market_risk", "risk_factors"] and sections["financial_statements"].splitlines()[1] == "Consolidated Statements of Earnings"
    assert "Net sales | $ | 22,386 | $ | 21,740" in sections["financial_statements"] and sections["risk_factors"].endswith("No changes to the risk factors.")
