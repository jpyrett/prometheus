"""Tests for the filtered filing evidence (design choices entry 38): prometheus/agents/data_clerk/filing_signals.py and `loaders.filing_parts`.

Run:  pytest tests/test_filing_signals.py -q

No network, no key, no cost. Two REAL companies, two real layouts:
  * Apple: the six real filings of tests/fixtures/filings_sample.txt (two 8-Ks, the 10-K, a 10-Q, a proxy statement, a Form 144) and, as the 10-Q's text, the real submission
    tests/fixtures/edgar_aapl_10q_0000320193-26-000006.txt.gz (a byte-exact prefix of what EDGAR returned);
  * META: the six real filings of tests/fixtures/filings_meta_sample.txt (three 8-Ks of which one is an officer's pay terms, a 10-Q, a proxy statement, a Form 144) and the real
    submission tests/fixtures/edgar_meta_10q_0001628280-26-066773.txt.gz.
The 10-Q text goes into the filing file the way the product puts it there (DataToolbox.fetch_filing_text appends a second record), with the download faked. Every expected number is
one read off those real pages. The tests that break a page use small synthetic ones. Nothing is scored by this module, and a test walks whole results to prove it.
"""
import gzip
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from prometheus.agents.data_clerk import data_toolbox, filing_signals as fs
from prometheus.agents.data_clerk.data_toolbox import DataToolbox
from prometheus.rag.loaders import filing_parts, load_filings

FIXTURES = Path(__file__).parent / "fixtures"
APPLE_TODAY, META_TODAY = date(2026, 5, 1), date(2026, 11, 1)


def real(name: str) -> str:
    return gzip.decompress((FIXTURES / f"{name}.txt.gz").read_bytes()).decode("utf-8", errors="replace")


def store(tmp_path, monkeypatch, symbol, filings_fixture, submission=None, today=APPLE_TODAY, folder_name=None):
    """The stored filing file of one company: its real records and, when `submission` is given, the 10-Q text taken out of the real EDGAR submission by the real fetch_filing_text
    (only the download is faked). Returns the filings as `filings_from_documents` groups them, and the path."""
    folder = tmp_path / (folder_name or symbol)
    folder.mkdir()
    (folder / f"{symbol}.txt").write_bytes((FIXTURES / filings_fixture).read_bytes())
    monkeypatch.setenv("FILINGS_OUTPUT_DIR", str(folder))
    if submission:
        monkeypatch.setenv("SEC_USER_AGENT", "Jacob Pyrett jacob_pyrett@yahoo.com")
        monkeypatch.setattr(DataToolbox, "_get_bytes", staticmethod(lambda client, limiter, name, url: real(submission).encode("utf-8")))
        monkeypatch.setattr(data_toolbox, "datetime", type("D", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(today.year, today.month, today.day, tzinfo=timezone.utc))}))
        DataToolbox(base_dir=tmp_path).fetch_filing_text(symbol, from_date=400)
    return fs.filings_from_documents(load_filings(folder / f"{symbol}.txt")), folder / f"{symbol}.txt"


@pytest.fixture
def apple(tmp_path, monkeypatch, capsys):
    filings, _ = store(tmp_path, monkeypatch, "AAPL", "filings_sample.txt", "edgar_aapl_10q_0000320193-26-000006")
    capsys.readouterr()
    return fs.recommendation_evidence("AAPL", filings, today=APPLE_TODAY)


@pytest.fixture
def meta(tmp_path, monkeypatch, capsys):
    filings, _ = store(tmp_path, monkeypatch, "META", "filings_meta_sample.txt", "edgar_meta_10q_0001628280-26-066773", META_TODAY)
    capsys.readouterr()
    return fs.recommendation_evidence("META", filings, today=META_TODAY)


@pytest.fixture
def meta_no_10q_text(tmp_path, monkeypatch):
    """META exactly as `data/filings/META.txt` is today: the 10-Q is stored without text."""
    filings, _ = store(tmp_path, monkeypatch, "META", "filings_meta_sample.txt", None, META_TODAY, folder_name="META_without_10q_text")
    return fs.recommendation_evidence("META", filings, today=META_TODAY)


def row(items: list[dict], label: str, **where) -> dict:
    return next(r for r in items if r["label"] == label and all(r.get(k) == v for k, v in where.items()))


# ---------------------------------------------------------------- reading a filing Document back (the loader owns the format)

def test_filing_parts_reads_back_what_the_loader_wrote_for_every_kind_of_document():
    docs = load_filings(FIXTURES / "filings_sample.txt")
    parts = [filing_parts(d) for d in docs]
    assert [p["section"] for p in parts] == ["items", "business", "risk_factors", None, "items", None, None]
    tenk = next(p for p in parts if p["section"] == "business")
    assert tenk == {"accession": "0000320193-25-000079", "section": "business", "symbol": "AAPL", "form": "10-K", "label": "Business", "filed": "2025-10-31",
                    "period_end": "2025-09-27", "url": "https://www.sec.gov/Archives/edgar/data/320193/0000320193-25-000079.txt", "has_text": True}
    eight_k = parts[0]
    assert (eight_k["form"], eight_k["label"], eight_k["period_end"], eight_k["has_text"]) == ("8-K", "", "", True)
    no_text = next(p for p in parts if p["form"] == "DEF 14A")
    assert (no_text["section"], no_text["has_text"], no_text["filed"]) == (None, False, "2026-01-08")


def test_filing_parts_agrees_with_the_loader_on_every_document_of_both_real_files():
    for name in ("filings_sample.txt", "filings_meta_sample.txt"):
        for doc in load_filings(FIXTURES / name):
            parts = filing_parts(doc)
            assert doc.metadata["headline"].startswith(f"{parts['symbol']} {parts['form']}") and parts["filed"] == doc.metadata["published_utc"][:10]
            assert doc.id == f"sec:{parts['accession']}" + (f":{parts['section']}" if parts["section"] else "")


@pytest.mark.parametrize("bad", [dict(id="pr:abc123", headline="AAPL 8-K, filed 2025-10-30"), dict(id="sec:0000320193-25-000077:items", headline="Apple reports results"),
                                 dict(id="news:1", headline="AAPL 8-K, filed 2025-10-30"), dict(id="sec:a:b:c", headline="AAPL 8-K, filed 2025-10-30")])
def test_a_document_that_is_not_a_filings_is_refused(bad):
    from langchain_core.documents import Document
    with pytest.raises(ValueError, match="not a filing Document"):
        filing_parts(Document(id=bad["id"], page_content="x", metadata={"headline": bad["headline"], "url": "u"}))


# ---------------------------------------------------------------- the pieces: numbers, statements, rows

@pytest.mark.parametrize("cell, number", [("45,317", 45317.0), ("(2,058)", -2058.0), ("—", 0.0), ("$ 0.26", 0.26), ("0.525", 0.525), ("12.5%", 12.5), ("(1.5)%", -1.5),
                                           ("December 27,2025", None), ("Net sales", None), ("", None), ("$", None)])
def test_to_number(cell, number):
    assert fs.to_number(cell) == number


@pytest.mark.parametrize("line, statement", [("CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)", "balance_sheet"), ("CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS (Unaudited)", "cash_flow"),
                                              ("CONDENSED CONSOLIDATED STATEMENTS OF SHAREHOLDERS’ EQUITY (Unaudited)", "equity"), ("CONDENSED CONSOLIDATED STATEMENTS OF STOCKHOLDERS' EQUITY", "equity"),
                                              ("CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)", "income"), ("CONDENSED CONSOLIDATED STATEMENTS OF INCOME", "income"),
                                              ("CONDENSED CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME (Unaudited)", "comprehensive"),
                                              ("Notes to Condensed Consolidated Financial Statements (Unaudited)", "notes"), ("Note 1 – Summary of Significant Accounting Policies", "notes"),
                                              ("LIABILITIES AND SHAREHOLDERS’ EQUITY:", None),  # a line inside the balance sheet that holds the words of another statement
                                              ("The balance sheets of the Company are presented in millions of dollars and are unaudited.", None),
                                              ("Total shareholders’ equity | 88,190 | 73,733", None)])
def test_a_statement_starts_only_at_a_short_all_capitals_heading(line, statement):
    assert fs.statement_of(line) == statement


def test_a_row_has_a_label_and_numbers_and_the_dollar_cells_are_dropped_and_the_header_rows_are_kept_as_columns():
    text = "\n".join(["CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)", "(In millions, except number of shares)", "December 27,2025 | September 27,2025", "Current liabilities:",
               "Accounts payable | $ | 70,587 | $ | 69,860", "Accumulated deficit | (2,177) | (14,264)", "a line without a bar", "Commercial paper | — | 7,979"])
    rows = fs.parse_rows(text)
    assert [(r["label"], r["values"], r["group"]) for r in rows] == [("Accounts payable", [70587.0, 69860.0], "Current liabilities"), ("Accumulated deficit", [-2177.0, -14264.0], "Current liabilities"),
                                                                       ("Commercial paper", [0.0, 7979.0], "Current liabilities")]
    assert rows[0]["columns"] == ["December 27,2025 | September 27,2025"] and rows[0]["units"] == "(In millions, except number of shares)" and rows[0]["statement"] == "balance_sheet"
    assert rows[0]["cells"] == ["70,587", "69,860"]


def test_year_headers_and_two_line_headers_become_the_columns_of_the_rows_after_them():
    text = "\n".join(["CONDENSED CONSOLIDATED STATEMENTS OF INCOME", "Three Months Ended September 30, | Nine Months Ended September 30,", "2026 | 2025 | 2026 | 2025", "Revenue | $ | 64,601 | $ | 51,242 | $ | 181,712 | $ | 141,073"])
    only = fs.parse_rows(text)[0]
    assert only["columns"] == ["Three Months Ended September 30, | Nine Months Ended September 30,", "2026 | 2025 | 2026 | 2025"] and only["values"] == [64601.0, 51242.0, 181712.0, 141073.0]


@pytest.mark.parametrize("current, prior, expected", [(45317, 35934, (9383, 26.11)), (-2177, -14264, (12087, 84.74)), (0, -26248, (26248, 100.0)), (5, 0, (5, None)), (None, 3, (None, None))])
def test_change_is_the_difference_and_the_percent_of_the_prior_in_its_size(current, prior, expected):
    assert tuple(fs.change(current, prior).values()) == expected


def test_share_classes_of_the_two_real_labels():
    apple = "Common stock and additional paid-in capital, $0.00001 par value: 50,400,000 shares authorized; 14,702,703 and 14,773,260 shares issued and outstanding, respectively"
    one = fs.share_classes(apple, "(In millions, except number of shares, which are reflected in thousands, and par value)")
    assert one == [{"class": "common stock", "current": 14702703.0, "prior": 14773260.0, "change": -70557.0, "change_pct": -0.48, "authorized": 50400000.0, "unit": "thousands"}]
    meta = ("Common stock and additional paid-in capital, $0.000006 par value; 5,000 million Class A shares authorized, 2,214 million and 2,187 million shares issued and outstanding, "
            "as of September 30, 2026 and December 31, 2025, respectively; 4,141 million Class B shares authorized, 342 million and 343 million shares issued and outstanding, "
            "as of September 30, 2026 and December 31, 2025, respectively")
    two = fs.share_classes(meta, "(In millions, except number of shares and par value)")
    assert [(c["class"], c["current"], c["prior"], c["authorized"], c["unit"]) for c in two] == [("Class A", 2214.0, 2187.0, 5000.0, "million"), ("Class B", 342.0, 343.0, 4141.0, "million")]
    assert fs.share_classes("Common stock, no shares stated", "") == []


# ---------------------------------------------------------------- the real balance sheets

def test_apples_balance_sheet_for_both_periods_with_the_change(apple):
    sheet = apple["balance_sheet"]
    assert sheet["columns"] == ["December 27,2025", "September 27,2025"] and sheet["units"].startswith("(In millions, except number of shares, which are reflected in thousands")
    items = sheet["items"]
    assert (items["cash"]["current"], items["cash"]["prior"], items["cash"]["change"], items["cash"]["change_pct"]) == (45317, 35934, 9383, 26.11)
    assert (items["total_assets"]["current"], items["total_assets"]["prior"]) == (379297, 359241)
    assert (items["total_liabilities"]["current"], items["total_liabilities"]["prior"]) == (291107, 285508)
    assert (items["total_equity"]["label"], items["total_equity"]["current"], items["total_equity"]["prior"]) == ("Total shareholders’ equity", 88190, 73733)
    assert (items["retained_earnings"]["label"], items["retained_earnings"]["current"], items["retained_earnings"]["prior"]) == ("Accumulated deficit", -2177, -14264)
    assert [(r["group"], r["current"]) for r in items["marketable_securities"]] == [("Current assets", 21590), ("Non-current assets", 77888)]
    assert sheet["source"] == {"accession": "0000320193-26-000006", "form": "10-Q", "filed": "2026-01-30", "period_end": "2025-12-27", "section": "financial_statements",
                               "url": "https://www.sec.gov/Archives/edgar/data/320193/0000320193-26-000006.txt"}


def test_apples_debt_rows_are_current_or_non_current_by_their_place_and_the_same_label_twice_is_two_rows(apple):
    debt = [(r["label"], r["classification"], r["current"], r["prior"]) for r in apple["balance_sheet"]["debt"]]
    assert debt == [("Commercial paper", "current", 1997, 7979), ("Term debt", "current", 11827, 12350), ("Term debt", "non-current", 76685, 78328)]


def test_apples_shares_issued_and_outstanding_fell_and_are_in_thousands(apple):
    shares = apple["balance_sheet"]["shares"]
    assert shares["unit"] == "thousands" and len(shares["classes"]) == 1
    assert (shares["current"], shares["prior"], shares["change"], shares["change_pct"]) == (14702703, 14773260, -70557, -0.48)
    assert shares["classes"][0]["authorized"] == 50400000 and shares["label"].endswith("shares issued and outstanding, respectively")


def test_metas_balance_sheet_has_other_labels_other_columns_and_another_unit(meta):
    sheet = meta["balance_sheet"]
    assert sheet["columns"] == ["September 30, 2026", "December 31, 2025"] and sheet["units"] == "(In millions, except number of shares and par value)"
    items = sheet["items"]
    assert (items["cash"]["current"], items["cash"]["prior"], items["cash"]["change"]) == (11222, 35873, -24651)  # the cash fell by two thirds
    assert (items["total_equity"]["label"], items["total_equity"]["current"], items["total_equity"]["prior"]) == ("Total stockholders' equity", 272386, 217243)
    assert (items["retained_earnings"]["label"], items["retained_earnings"]["current"]) == ("Retained earnings", 165342)
    assert [(r["label"], r["classification"], r["current"], r["prior"]) for r in sheet["debt"]] == [("Long-term debt", "non-current", 80922, 58744)]  # META has no "Non-current liabilities:" line
    assert (items["total_assets"]["current"], items["total_liabilities"]["current"]) == (478836, 206450)


def test_metas_shares_are_two_classes_and_the_total_is_their_sum_in_millions(meta):
    shares = meta["balance_sheet"]["shares"]
    assert shares["unit"] == "million" and [c["class"] for c in shares["classes"]] == ["Class A", "Class B"]
    assert [(c["current"], c["prior"], c["change"], c["authorized"]) for c in shares["classes"]] == [(2214, 2187, 27, 5000), (342, 343, -1, 4141)]
    assert (shares["current"], shares["prior"], shares["change"], shares["change_pct"]) == (2556, 2530, 26, 1.03)  # Class A alone would say +1.23%


def test_without_the_10_Q_text_there_is_no_balance_sheet_and_the_result_says_why_and_what_to_run(meta_no_10q_text):
    out = meta_no_10q_text
    assert out["balance_sheet"] is None and out["capital_actions"] == [] and out["income"] == []
    gap = next(g for g in out["gaps"] if g.startswith("no balance sheet"))
    assert "1 10-Q filing(s) in the window are stored without text" in gap and "fetch_filings" in gap and "SEC_USER_AGENT" in gap and "Business and Risk Factors only" in gap
    assert out["evidence"] and all(e["source"]["form"] == "8-K" for e in out["evidence"])  # what the stored 8-Ks say is still there


# ---------------------------------------------------------------- capital actions and the income statement

def test_apples_capital_actions_are_the_cash_flow_rows_with_both_periods(apple):
    actions = {(a["item"], a["label"]): a for a in apple["capital_actions"]}
    assert (actions["repurchases of common stock", "Repurchases of common stock"]["current"], actions["repurchases of common stock", "Repurchases of common stock"]["prior"]) == (-24701, -23606)
    assert (actions["dividends paid", "Payments for dividends and dividend equivalents"]["current"], actions["dividends paid", "Payments for dividends and dividend equivalents"]["prior"]) == (-3921, -3856)
    assert actions["share-based compensation expense", "Share-based compensation expense"]["current"] == 3594
    assert actions["taxes paid for net share settlement", "Payments for taxes related to net share settlement of equity awards"]["current"] == -2922
    assert [a["label"] for a in apple["capital_actions"] if a["item"] == "repayments of debt"] == ["Repayments of term debt", "Repayments of commercial paper, net"]
    assert apple["capital_actions"][0]["columns"] == ["December 27,2025", "December 28,2024"]
    assert {a["item"]: a["topic"] for a in apple["capital_actions"]} == {"share-based compensation expense": "equity_compensation", "taxes paid for net share settlement": "equity_compensation",
                                                                         "dividends paid": "dividend", "repurchases of common stock": "buyback", "repayments of debt": None}


def test_metas_buyback_stopped_and_its_debt_and_share_based_pay_grew(meta):
    actions = {a["item"]: a for a in meta["capital_actions"]}
    assert (actions["repurchases of common stock"]["current"], actions["repurchases of common stock"]["prior"], actions["repurchases of common stock"]["label"]) == (0, -26248, "Repurchases of Class A common stock")
    assert (actions["dividends paid"]["current"], actions["dividends paid"]["prior"]) == (-4060, -3986)
    assert (actions["share-based compensation expense"]["current"], actions["share-based compensation expense"]["prior"]) == (21053, 14537)
    assert (actions["proceeds from issuing debt"]["current"], actions["proceeds from issuing debt"]["prior"], actions["proceeds from issuing debt"]["topic"]) == (24910, 0, None)
    assert actions["taxes paid for net share settlement"]["current"] == -12768 and actions["repurchases of common stock"]["columns"] == ["2026", "2025"]


def test_the_income_headlines_keep_every_column_the_filing_has(apple, meta):
    a = {i["item"]: i for i in apple["income"]}
    assert (a["revenue"]["values"], a["net income"]["values"], a["diluted earnings per share"]["values"]) == ([143756, 124300], [42097, 36330], [2.84, 2.40])
    assert a["basic weighted shares"]["values"] == [14748158, 15081724] and a["diluted weighted shares"]["values"] == [14810356, 15150865]
    m = {i["item"]: i for i in meta["income"]}
    assert m["revenue"]["values"] == [64601, 51242, 181712, 141073] and m["diluted earnings per share"]["values"] == [3.83, 1.05, 20.44, 14.62]
    assert m["revenue"]["columns"] == ["Three Months Ended September 30, | Nine Months Ended September 30,", "2026 | 2025 | 2026 | 2025"]
    assert m["basic weighted shares"]["values"] == [2552, 2517, 2543, 2520] and m["diluted weighted shares"]["values"] == [2573, 2572, 2568, 2578]


# ---------------------------------------------------------------- the passages

def test_the_passages_are_exact_pieces_of_the_stored_text_and_come_with_their_filing(apple, meta, tmp_path, monkeypatch):
    texts = {}
    for symbol, fixture, submission, today in (("AAPL", "filings_sample.txt", "edgar_aapl_10q_0000320193-26-000006", APPLE_TODAY), ("META", "filings_meta_sample.txt", "edgar_meta_10q_0001628280-26-066773", META_TODAY)):
        (tmp_path / "again").mkdir(exist_ok=True)
        filings, _ = store(tmp_path / "again", monkeypatch, symbol, fixture, submission, today)
        texts[symbol] = {f["accession"]: f["sections"] for f in filings}
    checked = 0
    for symbol, out in (("AAPL", apple), ("META", meta)):
        for e in out["evidence"]:
            s = e["source"]
            assert e["quote"] in texts[symbol][s["accession"]][s["section"]], e["id"]  # a substring, never reworded: the Advisor can verify it against the source
            assert len(e["quote"]) <= fs.MAX_QUOTE and e["source"]["url"].startswith("https://www.sec.gov/Archives/") and re.fullmatch(r"E\d+", e["id"])
            checked += 1
    assert checked > 40


def test_an_officers_pay_terms_in_an_8_K_are_the_first_executive_compensation_evidence_and_the_risk_factors_come_last(meta):
    pay = [e for e in meta["evidence"] if e["topic"] == "executive_compensation"]
    assert pay[0]["source"]["form"] == "8-K" and pay[0]["source"]["filed"] == "2026-01-16" and "Compensatory Arrangements" in pay[0]["quote"]
    assert "will receive an annual base salary of $1,000,000 and a one-time, non-recurring cash sign-on bonus of $2,000,000" in pay[1]["quote"] and pay[1]["matched"] == "base salary"
    ranks = [fs.SECTION_RANK.get(e["source"]["section"], 5) for e in pay]
    assert ranks == sorted(ranks)  # source type before recency: the 8-K items, then the statements, then the risk factors


def test_metas_dilution_passages_are_the_plan_increase_the_anti_dilutive_shares_and_the_share_issuance_rows(meta):
    quotes = [e["quote"] for e in meta["evidence"] if e["topic"] == "dilution"]
    assert any("pursuant to the automatic increase provision under our 2025 Equity Incentive Plan" in q and "an increase of 55 million shares of Class A common stock" in q for q in quotes)
    assert any("approximately 55 million and 48 million shares, respectively, were excluded from the diluted EPS calculation as including them would have an anti-dilutive effect" in q for q in quotes)
    assert any(q.startswith("Issuance of common stock | 47 |") for q in quotes) and any(q == "Effect of dilutive shares | 21 | 55 | 25 | 58" for q in quotes)
    rows = [e for e in meta["evidence"] if e["quote"].startswith("Issuance of common stock | 47")]
    assert rows[0]["statement"] == "equity" and "equity statement row" in rows[0]["what"]  # a table row says which statement it is in


def test_apples_executive_compensation_evidence_is_the_item_5_02_heading_of_the_ceo_8_K_and_apples_dilution_is_the_dilutive_awards_row(apple):
    pay = [e for e in apple["evidence"] if e["topic"] == "executive_compensation"]
    assert [(e["source"]["form"], e["source"]["filed"]) for e in pay] == [("8-K", "2026-04-20")] and pay[0]["quote"].startswith("Item 5.02Departure of Directors or Certain Officers")
    dilution = [e for e in apple["evidence"] if e["topic"] == "dilution" and not e["derived"]]
    assert [e["quote"] for e in dilution] == ["Effect of dilutive share-based awards | 62,198 | 69,141"]
    assert dilution[0]["statement"] == "notes" and "table in the notes" in dilution[0]["what"]  # the earnings-per-share note, not the income statement: found by the first run of this test


def test_share_count_and_cash_flow_facts_are_derived_evidence_that_says_which_way_the_number_moved(apple, meta):
    a = [e for e in apple["evidence"] if e["derived"]]
    shares = next(e for e in a if "shares issued and outstanding" in e["what"])
    assert shares["topic"] == "buyback" and shares["filter"] == "positive" and "fell: 14,702,703 vs 14,773,260 (-70,557, -0.48%) thousands" in shares["what"]
    m = [e for e in meta["evidence"] if e["derived"]]
    grew = next(e for e in m if "shares issued and outstanding" in e["what"])
    assert grew["topic"] == "dilution" and grew["filter"] == "negative" and "rose: 2,556 vs 2,530 (+26, +1.03%) million" in grew["what"] and "Class A 2,214 vs 2,187 (+27, +1.23%)" in grew["what"]
    stopped = next(e for e in m if e["topic"] == "buyback" and "cash flow" in e["what"])
    assert "repurchases of common stock: 0 vs -26,248" in stopped["what"] and stopped["quote"] == "Repurchases of Class A common stock"  # the buyback is in the positive group with both numbers beside it
    assert all(e["source"]["accession"] in ("0000320193-26-000006", "0001628280-26-066773") for e in a + m)


def test_each_topic_is_in_its_fixed_group_and_the_groups_are_listed_negative_first(apple, meta):
    assert fs.TOPIC_FILTER == {"dilution": "negative", "executive_compensation": "negative", "equity_compensation": "negative", "buyback": "positive", "dividend": "positive"}
    for out in (apple, meta):
        assert all(e["filter"] == fs.TOPIC_FILTER[e["topic"]] for e in out["evidence"])
        groups = [e["filter"] for e in out["evidence"]]
        assert groups == sorted(groups, key=lambda g: g != "negative")  # all the negative-type topics, then the positive-type ones
        assert [e["id"] for e in out["evidence"]] == [f"E{n}" for n in range(1, len(out["evidence"]) + 1)]


def test_how_many_passages_were_found_and_how_many_are_shown_is_reported_per_topic(meta):
    for topic, c in meta["found"].items():
        assert c["filter"] == fs.TOPIC_FILTER[topic] and c["shown"] == min(c["found"], fs.EVIDENCE_PER_TOPIC) and c["shown"] <= fs.EVIDENCE_PER_TOPIC
        assert sum(1 for e in meta["evidence"] if e["topic"] == topic and not e["derived"]) == c["shown"]
        assert c["derived"] == sum(1 for e in meta["evidence"] if e["topic"] == topic and e["derived"])
    assert meta["found"]["equity_compensation"]["found"] > fs.EVIDENCE_PER_TOPIC  # the cap hid some, and the result says how many


def test_a_heading_with_nothing_to_quote_and_a_repeated_line_are_not_evidence():
    filings = [{"accession": "0000000001-26-000001", "form": "10-Q", "filed": "2026-01-02", "period_end": "", "url": "u", "has_text": True,
                "sections": {"financial_statements": "Restricted Stock Units\nRestricted stock units vest over four years and are settled in shares of the Company.\nRestricted stock units vest over four years and are settled in shares of the Company."}}]
    found = fs.passages(filings)
    assert [p["quote"] for p in found["equity_compensation"]] == ["Restricted stock units vest over four years and are settled in shares of the Company."]


def test_a_long_line_is_cut_to_the_sentence_with_the_match_and_marked_cut():
    sentence = "The board approved an increase of 55 million shares reserved for issuance under the plan."
    line = "Intro sentence that comes first. " + sentence + " " + "Then a long tail of other sentences follows, " * 30 + "The end."
    quote, cut = fs.quote_of(line, fs.TOPICS["dilution"].search(line))
    assert quote == sentence and not cut and quote in line  # the sentence is a substring, and short enough
    huge = "The plan will see an increase of 55 million shares " + "and further words " * 60
    quote, cut = fs.quote_of(huge, fs.TOPICS["dilution"].search(huge))
    assert cut and len(quote) == fs.MAX_QUOTE and huge.startswith(quote)


# ---------------------------------------------------------------- it scores nothing

def test_no_result_has_a_score_a_stance_a_rating_or_a_recommendation_as_a_key(apple, meta, meta_no_10q_text):
    for out in (apple, meta, meta_no_10q_text):
        assert not [k for k in fs.keys_of(out) if any(word in k.lower() for word in fs.FORBIDDEN_KEYS)], fs.keys_of(out)
    assert fs.keys_of({"a": [{"score": 1}]}) == {"a", "score"}  # the walker finds a key that is deep in a list


def test_the_digest_says_it_is_evidence_and_not_a_score_and_carries_the_real_numbers(apple, meta):
    d = apple["digest"]
    assert d.startswith("FILING EVIDENCE for AAPL") and "This is evidence, not a score: nothing here rates the stock." in d
    for fact in ("Cash and cash equivalents: 45,317 vs 35,934 (+9,383, +26.11%)", "debt, Term debt (non-current): 76,685 vs 78,328 (-1,643, -2.10%)",
                 "shares issued and outstanding, common stock: 14,702,703 vs 14,773,260 (-70,557, -0.48%) thousands; authorized 50,400,000", "repurchases of common stock: -24,701 vs -23,606",
                 "so a cash outflow is negative", "NEGATIVE-TYPE TOPICS (dilution, executive compensation, employee equity compensation)", "POSITIVE-TYPE TOPICS (buybacks, dividends)",
                 "NOT AVAILABLE: executive compensation plans are described in the proxy statement"):
        assert fact in d, fact
    m = meta["digest"]
    assert "shares issued and outstanding, all classes: 2,556 vs 2,530 (+26, +1.03%) million" in m and "repurchases of common stock [row: Repurchases of Class A common stock]: 0 vs -26,248" in m
    assert "Powell McCormick will receive an annual base salary of $1,000,000" in m


def test_the_digest_is_small_enough_for_a_prompt_and_every_evidence_item_is_in_it(apple, meta):
    for out in (apple, meta):
        assert len(out["digest"]) < 14000  # about 3,500 tokens for the richest real company so far; the structured result is for code, the digest for a model
        assert all(f"[{e['id']}]" in out["digest"] for e in out["evidence"])


# ---------------------------------------------------------------- the window, the empty cases, the gaps

def test_filings_older_than_the_window_are_not_read_and_the_result_says_so(tmp_path, monkeypatch, capsys):
    filings, _ = store(tmp_path, monkeypatch, "AAPL", "filings_sample.txt", "edgar_aapl_10q_0000320193-26-000006")
    capsys.readouterr()
    out = fs.recommendation_evidence("AAPL", filings, today=APPLE_TODAY, since_days=60)  # the 10-Q was filed 91 days before; the 8-K of 2026-04-20 is inside
    assert out["balance_sheet"] is None and out["as_of"] == "2026-04-20" and [f["form"] for f in out["filings"]["with_text"]] == ["8-K"]
    assert any(g.startswith("no balance sheet") for g in out["gaps"])
    nothing = fs.recommendation_evidence("AAPL", filings, today=date(2028, 1, 1), since_days=30)
    assert nothing["evidence"] == [] and nothing["as_of"] == "" and "no filing of AAPL was filed in the last 30 days (the latest stored was filed 2026-04-20)" in nothing["gaps"][0]


def test_no_stored_filings_at_all_is_a_gap_that_says_what_to_run():
    out = fs.recommendation_evidence("AAPL", [], today=APPLE_TODAY)
    assert out["gaps"] == ["no filing of AAPL is stored: run fetch_filings first"] and out["evidence"] == [] and out["balance_sheet"] is None
    assert "NOT AVAILABLE: no filing of AAPL is stored" in out["digest"]


def test_the_filings_of_the_window_are_counted_by_form_with_how_many_have_text(meta_no_10q_text, apple):
    forms = {e["form"]: (e["count"], e["with_text"], e["latest"]) for e in meta_no_10q_text["filings"]["by_form"]}
    assert forms == {"8-K": (3, 3, "2026-10-21"), "10-Q": (1, 0, "2026-10-22"), "144": (1, 0, "2026-10-19"), "DEF 14A": (1, 0, "2026-04-16")}
    assert [(f["form"], f["filed"], f["sections"]) for f in apple["filings"]["with_text"]][:2] == [("8-K", "2026-04-20", ["items"]), ("10-Q", "2026-01-30", [
        "financial_statements", "mdna", "market_risk", "controls", "legal_proceedings", "risk_factors", "unregistered_sales", "defaults", "mine_safety", "other_information", "exhibits"])]
    assert apple["filings"]["in_window"] == 6 and apple["window_days"] == 365  # the six real filings of the fixture


def test_a_proxy_statement_stored_without_text_is_named_as_a_gap_for_executive_compensation(meta_no_10q_text):
    assert any("executive compensation plans are described in the proxy statement, and 1 proxy/information statement(s)" in g for g in meta_no_10q_text["gaps"])


def test_a_10_Q_with_only_some_statements_still_gives_what_it_has():
    text = "\n".join(["CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS (Unaudited)", "2026 | 2025", "Repurchases of common stock | (500) | (400)", "Payments for dividends | (10) | (9)"])
    filings = [{"accession": "0000000001-26-000001", "form": "10-Q", "filed": "2026-01-02", "period_end": "2025-12-31", "url": "u", "has_text": True, "sections": {"financial_statements": text}}]
    out = fs.recommendation_evidence("ZZZ", filings, today=date(2026, 2, 1))
    assert out["balance_sheet"] is None and [a["item"] for a in out["capital_actions"]] == ["repurchases of common stock", "dividends paid"]
    assert any(e["derived"] and e["topic"] == "buyback" for e in out["evidence"])  # the facts of the statements that exist are still evidence
    assert out["gaps"][0] == "the financial statements of the 10-Q filed 2026-01-02 have no balance sheet that could be read (the rows found are those of: cash_flow)"


def test_the_result_is_plain_json(apple):
    import json
    assert json.loads(json.dumps(apple)) == apple
