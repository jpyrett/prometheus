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
from prometheus.rag.loaders import load_filings
from prometheus.tools.data_clerk import DataClerkTools
from test_filings import DistinctEmbeddings, massive_answers_from_the_real_records

SUBMISSION = gzip.decompress((Path(__file__).parent / "fixtures" / "edgar_aapl_10q_0000320193-26-000006.txt.gz").read_bytes()).decode("utf-8")
FILINGS = Path(__file__).parent / "fixtures" / "filings_sample.txt"
ACCESSION = "0000320193-26-000006"
URL = f"https://www.sec.gov/Archives/edgar/data/320193/{ACCESSION}.txt"
AGENT = "Jacob Pyrett jacob_pyrett@yahoo.com"
NAMES = ["financial_statements", "mdna", "market_risk", "controls", "legal_proceedings", "risk_factors", "unregistered_sales", "defaults", "mine_safety", "other_information", "exhibits"]


def lines_of(html: str) -> list[str]:
    return edgar_text.html_to_lines(html)


# ---------------------------------------------------------------- the real submission

def test_the_header_of_the_real_submission_gives_the_accession_the_form_and_the_period():
    assert edgar_text.submission_facts(SUBMISSION) == {"accession": ACCESSION, "form": "10-Q", "period_end": "2025-12-27"}  # CONFORMED PERIOD OF REPORT: 20251227


def test_the_10_Q_document_is_taken_and_not_one_of_its_exhibits():
    assert SUBMISSION.count("<DOCUMENT>") == 4  # the fixture: the 10-Q and three certifications
    document = edgar_text.main_document(SUBMISSION, "10-Q")
    assert len(document) == 765_989 and document.lstrip().startswith("<XBRL>") and "Quarterly Report on Form 10-Q" in document
    certification = edgar_text.main_document(SUBMISSION, "EX-31.1")
    assert len(certification) == 10_407 and "CERTIFICATION" in certification.upper() and len(certification) < len(document) / 50


def test_a_document_type_that_is_not_there_is_an_error_that_names_it():
    with pytest.raises(EdgarLayoutError, match="no document of type 10-K"):
        edgar_text.main_document(SUBMISSION, "10-K")


def test_most_of_the_page_is_xbrl_markup_and_the_text_is_what_is_left_after_it_is_dropped():
    document = edgar_text.main_document(SUBMISSION, "10-Q")
    lines = lines_of(document)
    text = "\n".join(lines)
    assert len(lines) == 570 and len(text) < len(document) / 10  # 766,000 characters of page, about 58,000 of text
    for hidden in ("us-gaap", "dei:", "iso4217", "xbrli", "EntityCentralIndexKey", "0000320193"):  # the facts of the hidden ix:header (56,191 characters) never reach the text
        assert hidden not in text, hidden


def test_a_table_row_is_one_line_with_its_cells_between_bars_and_the_real_numbers_are_kept_as_they_are():
    text = "\n".join(lines_of(edgar_text.main_document(SUBMISSION, "10-Q")))
    assert "Total net sales | 143,756 | 124,300" in text and "Products | $ | 113,743 | $ | 97,960" in text  # the dollar sign is a cell of its own in the real page
    assert "Apple Inc. | Q1 2026 Form 10-Q | 20" in text  # a page footer: real text, kept
    assert "Item 1. | Financial Statements | 1" in text  # the table of contents, a table with page numbers


def test_the_real_10_Q_is_split_into_its_eleven_items_in_order_and_named():
    sections = edgar_text.filing_sections(SUBMISSION, "10-Q")
    assert [name for name, _ in sections] == NAMES  # Part I Items 1 to 4, then Part II Items 1, 1A, 2, 3, 4, 5, 6: the part is tracked, so Part II's Item 1 is not Part I's
    assert {name: len(body) for name, body in sections} == {
        "financial_statements": 24654, "mdna": 18051, "market_risk": 391, "controls": 1386, "legal_proceedings": 5227, "risk_factors": 410,
        "unregistered_sales": 1401, "defaults": 45, "mine_safety": 83, "other_information": 1996, "exhibits": 1117}
    assert sum(len(body) for _, body in sections) == 54761


def test_each_section_starts_with_its_own_heading_and_has_text_after_it():
    heads = {"financial_statements": "Item 1. Financial Statements", "market_risk": "Item 3. Quantitative and Qualitative Disclosures About Market Risk",
             "legal_proceedings": "Item 1. Legal Proceedings", "risk_factors": "Item 1A. Risk Factors", "exhibits": "Item 6. Exhibits"}
    by_name = dict(edgar_text.filing_sections(SUBMISSION, "10-Q"))
    for name, heading in heads.items():
        assert by_name[name].startswith(heading + "\n") and len(by_name[name]) > len(heading) + 1, name
    assert by_name["defaults"] == "Item 3. Defaults Upon Senior Securities\nNone."  # the shortest real Item


def test_the_cover_page_and_the_table_of_contents_are_left_out():
    text = "\n".join(body for _, body in edgar_text.filing_sections(SUBMISSION, "10-Q"))
    for left_out in ("Commission File Number", "Indicate by check mark", "Item 1. | Financial Statements | 1", "UNITED STATES"):
        assert left_out not in text, left_out
    assert text.count("Item 1. Financial Statements") == 1  # the heading once: the table of contents row is not a second one


def test_real_text_from_the_filing_reaches_the_sections():
    by_name = dict(edgar_text.filing_sections(SUBMISSION, "10-Q"))
    assert "Total net sales | 143,756 | 124,300" in by_name["financial_statements"]
    assert "Kevan Parekh, the Company’s Senior Vice President and Chief Financial Officer, entered into a trading plan" in by_name["other_information"]
    assert "Digital Markets Act Investigations" in by_name["legal_proceedings"] and "business seasonality".lower() in by_name["mdna"].lower()


def test_only_the_forms_the_module_knows_are_taken():
    assert edgar_text.TEXT_FORMS == ("10-Q",)
    with pytest.raises(EdgarLayoutError, match="cannot be taken from EDGAR yet"):
        edgar_text.filing_sections(SUBMISSION, "EX-31.1")


# ---------------------------------------------------------------- html to lines (small pages)

def test_hidden_parts_scripts_and_styles_are_dropped_and_the_rest_is_kept():
    html = ('<html><head><title>T</title><style>p{color:red}</style></head><body><ix:header><ix:hidden>a fact</ix:hidden></ix:header>'
            '<script>var x = 1;</script><div style="display: none">hidden text</div><p>Visible <b>text</b></p><span style="DISPLAY:NONE">also hidden</span>after</body></html>')
    assert lines_of(html) == ["Visible text", "after"]


def test_block_elements_start_new_lines_and_whitespace_and_no_break_spaces_collapse():
    assert lines_of("<div>one</div><p>two&nbsp;&nbsp; words</p>three<br/>four<br>five<ul><li>six</li></ul>\u200b") == ["one", "two words", "three", "four", "five", "six"]


def test_a_table_row_has_its_cells_between_bars_and_empty_cells_and_empty_rows_are_left_out():
    html = "<table><tr><td>Net sales</td><td></td><td> $ </td><td>1,000</td></tr><tr><td> </td><td></td></tr><tr><th>Total</th><td>2,000</td></tr></table>"
    assert lines_of(html) == ["Net sales | $ | 1,000", "Total | 2,000"]


def test_a_table_inside_a_cell_does_not_lose_the_outer_row():
    html = "<table><tr><td>outer</td><td><table><tr><td>inner</td><td>cell</td></tr></table></td><td>end</td></tr></table>"
    assert "inner | cell" in lines_of(html) and any(line.startswith("outer") and line.endswith("end") for line in lines_of(html))


def test_html_that_is_not_well_formed_cannot_leave_the_text_hidden():
    assert lines_of('<div style="display:none">hidden<p>still hidden</div><p>shown</p>') == ["shown"]  # the end tag of the hidden div closes it
    assert lines_of("<p>one<p>two<br><br>three<img src='x'>four</p>") == ["one", "two", "threefour"]  # tags that are never closed; an image is inline, not a line break
    assert lines_of("<td>cell outside a table</td><p>x</p>") == ["cell outside a table", "x"]


# ---------------------------------------------------------------- the Items of a 10-Q (small texts)

def test_an_item_heading_in_the_table_of_contents_as_plain_lines_gives_way_to_the_real_one():
    lines = ["Part I", "Item 1. Financial Statements 1", "Item 2. Management's Discussion 5", "PART I - FINANCIAL INFORMATION", "Item 1. Financial Statements", "the statements", "more statements",
             "Item 2. Management's Discussion", "the discussion"]
    assert edgar_text.split_items(lines) == [("financial_statements", "Item 1. Financial Statements\nthe statements\nmore statements"),
                                             ("mdna", "Item 2. Management's Discussion\nthe discussion")]  # the longer one of each name wins: the table of contents lines are one line long


def test_a_cross_reference_at_the_start_of_a_line_is_not_a_heading():
    lines = ["PART I", "Item 2. Management's Discussion", "text", "Item 1A of the 2025 Form 10-K under the heading Risk Factors", "Part I, Item 1A of the Form 10-K", "more text"]
    assert edgar_text.split_items(lines) == [("mdna", "Item 2. Management's Discussion\ntext\nItem 1A of the 2025 Form 10-K under the heading Risk Factors\nPart I, Item 1A of the Form 10-K\nmore text")]


def test_the_part_is_tracked_so_the_same_item_number_in_two_parts_is_two_sections():
    lines = ["PART I", "Item 1. Financial Statements", "a", "PART II - OTHER INFORMATION", "Item 1. Legal Proceedings", "b"]
    assert [name for name, _ in edgar_text.split_items(lines)] == ["financial_statements", "legal_proceedings"]


def test_headings_are_found_in_capitals_with_other_dashes_and_without_a_title():
    lines = ["PART II \u2014 OTHER INFORMATION", "ITEM 5. OTHER INFORMATION", "x", "Item 6 \u2013 Exhibits", "y", "Item 1A.", "z"]
    assert [name for name, _ in edgar_text.split_items(lines)] == ["other_information", "exhibits", "risk_factors"]


def test_an_item_that_has_no_known_name_is_named_by_its_part_and_number_and_without_a_part_by_its_number():
    assert edgar_text.split_items(["PART II", "Item 7. Something New", "x"]) == [("partii_item7", "Item 7. Something New\nx")]
    assert edgar_text.split_items(["Item 2. Management's Discussion", "x"]) == [("item2", "Item 2. Management's Discussion\nx")]


def test_an_item_with_nothing_after_its_heading_is_dropped_and_text_before_the_first_heading_is_left_out():
    lines = ["cover page", "PART I", "Item 1. Financial Statements", "the statements", "Item 2. Management's Discussion"]
    assert edgar_text.split_items(lines) == [("financial_statements", "Item 1. Financial Statements\nthe statements")]


def test_a_table_row_that_looks_like_a_heading_is_not_one():
    """Two shapes of a table-of-contents row. In the real 10-Q the item number and the title are cells of their own (`Item 2. | Management's ... | 13`), which the capital-letter rule
    already refuses; a row whose first cell holds both (`Item 2. Management's Discussion | 13`) starts like a heading, and only the rule about the bar refuses it (found by the mutation check)."""
    for row in ("Item 2. | Management's Discussion | 13", "Item 2. Management's Discussion | 13"):
        assert edgar_text.split_items(["PART I", "Item 1. Financial Statements", "a", row, "b"]) == [("financial_statements", f"Item 1. Financial Statements\na\n{row}\nb")], row


def test_text_with_no_item_heading_is_an_error_and_not_one_big_section():
    with pytest.raises(EdgarLayoutError, match="no Item heading"):
        edgar_text.split_items(["just", "some", "text"])
    page = f"<SEC-HEADER>x</SEC-HEADER><DOCUMENT>\n<TYPE>10-Q\n<TEXT>\n<html><body><p>hello</p></body></html>\n</TEXT>\n</DOCUMENT>\n"
    with pytest.raises(EdgarLayoutError, match="no Item heading"):
        edgar_text.filing_sections(page, "10-Q")


# ---------------------------------------------------------------- the setting

@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(tmp_path / "filings"))
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    (tmp_path / "filings").mkdir()
    (tmp_path / "filings" / "AAPL.txt").write_bytes(FILINGS.read_bytes())
    monkeypatch.setattr(data_toolbox, "datetime", type("D", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 5, 1, tzinfo=timezone.utc))}))  # a fixed "today"
    return DataToolbox(base_dir=tmp_path)


def test_the_user_agent_comes_from_the_environment_then_the_env_file_and_has_no_default(box, monkeypatch):
    with pytest.raises(EdgarNotConfigured, match="SEC_USER_AGENT is not set") as error:
        box._sec_user_agent(env={})
    assert "Your Name you@example.com" in str(error.value)  # the message shows the shape, never a value that was invented
    assert box._sec_user_agent(env={"SEC_USER_AGENT": "  Jacob   Pyrett jacob_pyrett@yahoo.com "}) == AGENT  # spaces collapsed
    monkeypatch.setenv("SEC_USER_AGENT", "From Environment env@example.com")
    assert box._sec_user_agent(env={"SEC_USER_AGENT": AGENT}) == "From Environment env@example.com"  # the environment first


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


def test_the_10_Q_with_no_text_is_downloaded_and_its_text_is_appended_as_a_second_record(downloads, tmp_path, capsys):
    path = tmp_path / "filings" / "AAPL.txt"
    before = path.read_bytes()
    stats = downloads.fetch_filing_text("aapl", from_date=400)
    capsys.readouterr()
    assert stats == {"symbol": "AAPL", "path": str(path), "candidates": 1, "written": 1}
    assert downloads.asked == [(f"EDGAR {ACCESSION}", URL, AGENT)]  # the url of the record, with the identity in the header
    after = path.read_bytes()
    assert after.startswith(before) and after.count(f"accession: {ACCESSION}\n".encode()) == 2  # append-only: nothing that was there changed
    record = after[len(before):].decode()
    assert record.startswith(f"<<<FILING>>>\naccession: {ACCESSION}\nsymbol: AAPL\nform: 10-Q\nfiled: 2026-01-30\nperiod_end: 2025-12-27\nurl: {URL}\ntext_status: ok\n<<<TEXT>>>\n== financial_statements ==\n")
    assert record.count("\n== ") == 11 and record.endswith("<<<END>>>\n")


def test_what_the_toolbox_appends_loads_as_eleven_documents_that_replace_the_one_without_text(downloads, tmp_path, capsys):
    path = tmp_path / "filings" / "AAPL.txt"
    assert [d.id for d in load_filings(path) if d.id.startswith(f"sec:{ACCESSION}")] == [f"sec:{ACCESSION}"]  # before: the short "text not stored" Document
    downloads.fetch_filing_text("AAPL", from_date=400)
    capsys.readouterr()
    docs = [d for d in load_filings(path) if d.id.startswith(f"sec:{ACCESSION}")]
    assert [d.id for d in docs] == [f"sec:{ACCESSION}:{name}" for name in NAMES]
    assert docs[0].metadata["headline"] == "AAPL 10-Q, Financial Statements, filed 2026-01-30, period ending 2025-12-27"
    assert docs[1].metadata["headline"] == "AAPL 10-Q, Management's Discussion and Analysis, filed 2026-01-30, period ending 2025-12-27"
    assert docs[5].metadata["headline"] == "AAPL 10-Q, Risk Factors, filed 2026-01-30, period ending 2025-12-27"
    assert docs[0].metadata["published_utc"] == "2026-01-30T00:00:00Z" and docs[0].metadata["publisher"] == "SEC filing" and docs[0].metadata["url"] == URL
    assert docs[0].page_content.startswith("AAPL 10-Q, Financial Statements, filed 2026-01-30, period ending 2025-12-27\n\nItem 1. Financial Statements\n")
    assert len(load_filings(path)) == 7 - 1 + 11  # the other six Documents of the fixture are unchanged: the 10-Q went from 1 Document to 11


def test_the_second_run_finds_nothing_to_fetch_and_makes_no_request(downloads, tmp_path, capsys):
    downloads.fetch_filing_text("AAPL", from_date=400)
    size = (tmp_path / "filings" / "AAPL.txt").stat().st_size
    again = downloads.fetch_filing_text("AAPL", from_date=400)
    capsys.readouterr()
    assert again["candidates"] == 0 and again["written"] == 0 and len(downloads.asked) == 1 and (tmp_path / "filings" / "AAPL.txt").stat().st_size == size


def test_nothing_to_fetch_needs_no_user_agent_and_something_to_fetch_does(box, capsys):
    assert box.fetch_filing_text("AAPL", from_date=30)["candidates"] == 0  # the 10-Q was filed 91 days before the fixed "today": outside 30 days, nothing to do, no setting needed
    with pytest.raises(EdgarNotConfigured, match="SEC_USER_AGENT is not set"):
        box.fetch_filing_text("AAPL", from_date=400)
    capsys.readouterr()


def test_a_filing_filed_before_the_window_is_not_fetched(downloads, capsys):
    assert downloads.fetch_filing_text("AAPL", from_date=60)["candidates"] == 0 and downloads.asked == []
    capsys.readouterr()


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


def test_a_failure_on_the_second_10_Q_keeps_the_first_that_was_written(downloads, tmp_path, monkeypatch, capsys):
    path = tmp_path / "filings" / "AAPL.txt"
    second = "0000320193-26-000013"
    record = next(r for r in path.read_text(encoding="utf-8").split("<<<FILING>>>\n")[1:] if f"accession: {ACCESSION}" in r)
    path.write_text(path.read_text(encoding="utf-8") + "<<<FILING>>>\n" + record.replace(ACCESSION, second).replace("filed: 2026-01-30", "filed: 2026-03-01"), encoding="utf-8")

    def fake(client, limiter, name, url):
        if second in url:
            raise RuntimeError(f"EDGAR {second} HTTP 503")
        return SUBMISSION.encode()

    monkeypatch.setattr(DataToolbox, "_get_bytes", staticmethod(fake))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        downloads.fetch_filing_text("AAPL", from_date=400)
    capsys.readouterr()
    assert len([d for d in load_filings(path) if d.id.startswith(f"sec:{ACCESSION}:")]) == 11  # the first one is stored with its text
    assert [d.id for d in load_filings(path) if second in d.id] == [f"sec:{second}"]            # the second is still the short one: a rerun fetches it


def test_fetch_filings_still_skips_a_filing_whose_text_was_fetched_later(downloads, tmp_path, monkeypatch, capsys):
    """The Massive fetch skips by accession, and the accession of the 10-Q is in the file twice now: it must still count as stored."""
    downloads.fetch_filing_text("AAPL", from_date=400)
    seen = {line[11:].strip() for line in (tmp_path / "filings" / "AAPL.txt").read_text(encoding="utf-8").splitlines() if line.startswith("accession: ")}
    assert ACCESSION in seen and len(seen) == 6
    capsys.readouterr()


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


def test_the_real_client_sends_the_identity_in_the_user_agent_header_and_asks_for_the_stored_url(box, monkeypatch, capsys):
    """No fake of `_get_bytes` here: the real download code runs, over a mock transport that answers with the real fixture."""
    requests = []
    real = httpx.Client

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=SUBMISSION.encode())

    monkeypatch.setenv("SEC_USER_AGENT", AGENT)
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert box.fetch_filing_text("AAPL", from_date=400)["written"] == 1
    capsys.readouterr()
    assert [str(r.url) for r in requests] == [URL] and requests[0].headers["user-agent"] == AGENT


def test_a_200_returns_the_bytes_and_a_404_or_a_redirect_stops_without_the_identity_in_the_message():
    assert get(lambda r: httpx.Response(200, content=b"abc")) == b"abc"
    for status in (404, 403, 302):
        with pytest.raises(RuntimeError, match=f"EDGAR test HTTP {status}$"):
            get(lambda r, s=status: httpx.Response(s, headers={"location": "https://elsewhere.example.com/"}))


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
