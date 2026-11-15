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


@pytest.mark.parametrize("bad", [dict(id="pr:abc123", headline="AAPL 8-K, filed 2025-10-30"), dict(id="sec:0000320193-25-000077:items", headline="Apple reports results"),
                                 dict(id="news:1", headline="AAPL 8-K, filed 2025-10-30"), dict(id="sec:a:b:c", headline="AAPL 8-K, filed 2025-10-30")])
def test_a_document_that_is_not_a_filings_is_refused(bad):
    from langchain_core.documents import Document
    with pytest.raises(ValueError, match="not a filing Document"):
        filing_parts(Document(id=bad["id"], page_content="x", metadata={"headline": bad["headline"], "url": "u"}))


# ---------------------------------------------------------------- the pieces: numbers, statements, rows


@pytest.mark.parametrize("current, prior, expected", [(45317, 35934, (9383, 26.11)), (-2177, -14264, (12087, 84.74)), (0, -26248, (26248, 100.0)), (5, 0, (5, None)), (None, 3, (None, None))])
def test_change_is_the_difference_and_the_percent_of_the_prior_in_its_size(current, prior, expected):
    assert tuple(fs.change(current, prior).values()) == expected


# ---------------------------------------------------------------- the real balance sheets


# ---------------------------------------------------------------- capital actions and the income statement


# ---------------------------------------------------------------- the passages


# ---------------------------------------------------------------- it scores nothing

def test_no_result_has_a_score_a_stance_a_rating_or_a_recommendation_as_a_key(apple, meta, meta_no_10q_text):
    for out in (apple, meta, meta_no_10q_text):
        assert not [k for k in fs.keys_of(out) if any(word in k.lower() for word in fs.FORBIDDEN_KEYS)], fs.keys_of(out)
    assert fs.keys_of({"a": [{"score": 1}]}) == {"a", "score"}  # the walker finds a key that is deep in a list


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


def test_a_10_Q_with_only_some_statements_still_gives_what_it_has():
    text = "\n".join(["CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS (Unaudited)", "2026 | 2025", "Repurchases of common stock | (500) | (400)", "Payments for dividends | (10) | (9)"])
    filings = [{"accession": "0000000001-26-000001", "form": "10-Q", "filed": "2026-01-02", "period_end": "2025-12-31", "url": "u", "has_text": True, "sections": {"financial_statements": text}}]
    out = fs.recommendation_evidence("ZZZ", filings, today=date(2026, 2, 1))
    assert out["balance_sheet"] is None and [a["item"] for a in out["capital_actions"]] == ["repurchases of common stock", "dividends paid"]
    assert any(e["derived"] and e["topic"] == "buyback" for e in out["evidence"])  # the facts of the statements that exist are still evidence
    assert out["gaps"][0] == "the financial statements of the 10-Q filed 2026-01-02 have no balance sheet that could be read (the rows found are those of: cash_flow)"


# ---------------------------------------------------------------- the Sandisk 10-Q (real lines of the stored page, found by the first real run of the Advisor)

SNDK_SHARES_LABEL = ("Common stock, $0.01 par value; authorized \u2014 450 shares; issued \u2014 150 shares and outstanding \u2014 144 shares "
                     "(as of July 3, 2026 issued \u2014 149 shares and outstanding \u2014 146 shares)")


def test_the_share_count_of_the_dash_layout_is_the_outstanding_shares_now_and_at_the_earlier_date():
    classes = fs.share_classes(SNDK_SHARES_LABEL, "(in millions, except par value)")
    assert classes == [{"class": "common stock", "current": 144.0, "prior": 146.0, "change": -2.0, "change_pct": -1.37, "authorized": 450.0, "unit": "million"}]
    text = f"CONDENSED CONSOLIDATED BALANCE SHEETS\n(in millions, except par value)\nDecember 31, 2026 | July 3, 2026\nTotal assets | $ | 25,793 | $ | 22,507\n{SNDK_SHARES_LABEL} | $ | 2 | $ | 1"
    sheet = fs.balance_sheet(fs.parse_rows(text))
    assert sheet["units"] == "(in millions, except par value)" and sheet["shares"]["current"] == 144.0 and sheet["shares"]["change_pct"] == -1.37


# ---------------------------------------------------------------- Procter & Gamble's 10-Q (real lines of the stored page: a third layout, found by the first real run of the Advisor)

PG_STATEMENTS = """CONSOLIDATED STATEMENTS OF EARNINGS
Three Months Ended December 31 | Six Months Ended December 31
Amounts in millions except per share amounts | 2025 | 2024 | 2025 | 2024
NET SALES | $ | 22,208 | $ | 21,882 | $ | 44,594 | $ | 43,619
NET EARNINGS | 4,331 | 4,659 | 9,112 | 8,646
CONSOLIDATED BALANCE SHEETS
Amounts in millions | December 31, 2025 | June 30, 2025
Cash and cash equivalents | $ | 10,825 | $ | 9,556
Accounts receivable | 6,279 | 6,185
Total inventories | 7,817 | 7,551
TOTAL ASSETS | $ | 127,286 | $ | 125,231
CONSOLIDATED STATEMENTS OF CASH FLOWS
Six Months Ended December 31
Amounts in millions | 2025 | 2024
OPERATING ACTIVITIES (1)
Net earnings | 9,112 | 8,646
Share-based compensation expense | 262 | 241
TOTAL OPERATING ACTIVITIES | 10,380 | 9,127
Dividends to shareholders | (5,093) | (4,886)
Treasury stock purchases | (3,528) | (4,449)"""


def test_procter_and_gambles_cash_balance_and_payout_coverage_are_worked_from_those_figures():
    from prometheus.agents.data_clerk import fundamentals as f
    rows = fs.parse_rows(PG_STATEMENTS)
    evidence = {"period_months": fs.period_months(rows, "10-Q"), "capital_actions": fs.capital_actions(rows), "income": fs.income_headlines(rows), "balance_sheet": fs.balance_sheet(rows)}
    assert f.cash_balance(evidence)[0] == 10825e6 and f.payout_coverage(evidence)[0] == pytest.approx(5093 / 9112)


# ---------------------------------------------------------------- Domino's, Papa John's and Yum's 10-Qs (real lines of the stored pages; the pizza run was the fourth, fifth and sixth layout)

DPZ_STATEMENTS = """Condensed Consolidated Balance Sheets
(Unaudited)
(In thousands) | September 6, 2026 | December 28, 2025
Cash and cash equivalents | $ | 229,035 | $ | 125,675
Accounts receivable, net | 286,489 | 315,958
Inventories | 89,535 | 79,189
Total assets | $ | 1,766,373 | $ | 1,716,459
Total stockholders\u2019 deficit | (3,955,098 | ) | (3,901,142 | )
Condensed Consolidated Statements of Income
(Unaudited)
Fiscal Quarter Ended | Three Fiscal Quarters Ended
(In thousands, except per-share data) | 2026 | 2025 | 2026 | 2025
Total revenues | 1,185,598 | 1,147,054 | 3,530,619 | 3,404,254
Net income | $ | 152,179 | $ | 139,319 | $ | 427,740 | $ | 420,061
Condensed Consolidated Statements of Cash Flows
(Unaudited)
Three Fiscal Quarters Ended
(In thousands) | 2026 | 2025
Net cash provided by operating activities | 520,601 | 552,256
Purchases of common stock | (308,239 | ) | (277,698 | )
Payments of common stock dividends and equivalents | (134,331 | ) | (119,503 | )"""
PZZA_STATEMENTS = """Condensed Consolidated Balance Sheets
(Unaudited)
(In thousands, except per share amounts) | June 28,2026 | December 28,2025
Accounts receivable, net | 94,512 | 103,068
Inventories | 32,546 | 34,336
Total assets | $ | 805,031 | $ | 837,508
Condensed Consolidated Statements of Operations
(Unaudited)
Three Months Ended | Six Months Ended
(In thousands, except per share amounts) | June 28,2026 | June 29,2025 | June 28,2026 | June 29,2025
Total revenues | 482,397 | 529,166 | 961,006 | 1,047,475
Net income | 8,700 | 9,671 | 15,638 | 19,014
Condensed Consolidated Statements of Cash Flows
(In thousands) | June 28,2026 | June 29,2025
Net cash provided by operating activities | 35,828 | 66,843
Dividends paid to common stockholders | (30,864) | (30,493)"""
YUM_STATEMENTS = """CONDENSED CONSOLIDATED STATEMENTS OF INCOME (Unaudited)
(in millions, except per share data)
Quarter ended | Year to date
Revenues | 6/30/2026 | 6/30/2025 | 6/30/2026 | 6/30/2025
Total revenues | 2,169 | 1,933 | 4,228 | 3,720
Net Income | $ | 853 | $ | 374 | $ | 1,285 | $ | 628
CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS (Unaudited)
(in millions)
Year to date
Net Cash Provided by Operating Activities | 923 | 850
Repurchase shares of Common Stock | (674) | (338)
Dividends paid on Common Stock | (413) | (395)
CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)
(in millions)
6/30/2026 | 12/31/2025
Cash and cash equivalents | $ | 674 | $ | 709
Total Assets | $ | 8,682 | $ | 8,197
Common Stock, no par value, 750 shares authorized; 273 shares issued in 2026 and 277 shares issued in 2025 | — | —
Total Shareholders\u2019 Deficit | (7,107) | (7,325)"""


def pizza(text):
    rows = fs.parse_rows(text)
    return rows, {a["item"]: a for a in fs.capital_actions(rows)}, {i["item"]: i["values"] for i in fs.income_headlines(rows)}, fs.balance_sheet(rows)


def test_dominos_negative_numbers_split_over_two_cells_title_case_headings_and_a_units_row_are_read():
    rows, actions, income, sheet = pizza(DPZ_STATEMENTS)
    assert {r["units"] for r in rows if r["statement"] == "cash_flow"} == {"(In thousands)"} and sheet["units"] == "(In thousands)"
    assert (actions["repurchases of common stock"]["current"], actions["repurchases of common stock"]["prior"]) == (-308239.0, -277698.0)          # "(308,239 | )" is one negative number
    assert actions["dividends paid"]["current"] == -134331.0 and actions["cash from operations"]["current"] == 520601.0
    assert income["revenue"] == [1185598.0, 1147054.0, 3530619.0, 3404254.0] and income["net income"][2] == 427740.0
    assert sheet["items"]["inventories"]["current"] == 89535.0 and sheet["items"]["accounts_receivable"]["current"] == 286489.0
    assert sheet["items"]["total_equity"]["current"] == -3955098.0                                                           # a stockholders' DEFICIT is the total equity, negative
    assert fs.period_months(rows, "10-Q", sheet) == 9                                                                      # "Three Fiscal Quarters": December 28, 2025 to September 6, 2026


def test_yums_year_to_date_months_labels_and_share_count_are_read():
    rows, actions, income, sheet = pizza(YUM_STATEMENTS)
    assert actions["repurchases of common stock"]["current"] == -674.0 and actions["dividends paid"]["current"] == -413.0 and income["revenue"][2] == 4228.0
    assert fs.period_months(rows, "10-Q", sheet) == 6 and sheet["items"]["total_equity"]["current"] == -7107.0              # the balance sheet's own columns are 6/30/2026 and 12/31/2025
    assert sheet["shares"] and (sheet["shares"]["current"], sheet["shares"]["prior"], sheet["shares"]["unit"]) == (273.0, 277.0, "million")


def test_months_from_the_balance_sheet_dates_to_the_nearest_quarter():
    cols = lambda a, b: {"columns": [a, b]}
    assert fs.months_from_balance_sheet(cols("September 6, 2026", "December 28, 2025")) == 9
    assert fs.months_from_balance_sheet(cols("December 31, 2025", "June 30, 2025")) == 6 and fs.months_from_balance_sheet(cols("6/30/2026", "12/31/2025")) == 6
    assert fs.months_from_balance_sheet({"columns": ["(In thousands)", "June 28,2026", "December 28,2025"]}) == 6          # the units cell is skipped
    assert fs.months_from_balance_sheet(cols("June 28, 2026", "June 28, 2026")) is None and fs.months_from_balance_sheet(cols("x", "y")) is None and fs.months_from_balance_sheet(None) is None


# ---------------------------------------------------------------- McDonald's and Restaurant Brands' 10-Qs (real lines): bare units, four cash-flow columns, other labels, a Part I heading in title case

MCD_STATEMENTS = """CONDENSED CONSOLIDATED STATEMENT OF CASH FLOWS (UNAUDITED)
In millions | 2026 | 2025 | 2026 | 2025
Cash provided by operations | 2,807 | 1,998 | 5,222 | 4,426
Purchases of restaurant businesses | (74) | (56) | (153) | (131)
Treasury stock purchases | (856) | (505) | (1,251) | (982)
Common stock dividends | (1,317) | (1,265) | (2,640) | (2,530)
CONDENSED CONSOLIDATED BALANCE SHEET (UNAUDITED)
In millions, except per share data | June 30,2026 | December 31,2025
Total assets | $ | 56,000 | $ | 55,000"""
QSR_STATEMENTS = """CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS
(In millions of U.S. dollars, except share data, Unaudited) | 2026 | 2025
Net cash provided by operating activities from continuing operations | 1,308 | 1,159
Payment of common share dividends and Partnership exchangeable unit distributions | (875) | (826)
Repurchases of common shares | (444) | (306)"""


def test_mcdonalds_bare_units_four_cash_flow_columns_and_its_labels_are_read():
    _, actions, _, sheet = pizza(MCD_STATEMENTS)
    assert actions["cash from operations"]["current"] == 5222.0 and actions["cash from operations"]["prior"] == 4426.0                       # the year to date, the last two of four columns
    assert actions["repurchases of common stock"]["current"] == -1251.0 and actions["dividends paid"]["current"] == -2640.0 and "purchases of restaurant businesses" not in actions
    assert sheet["units"] == "In millions, except per share data" and fs.period_months(fs.parse_rows(MCD_STATEMENTS), "10-Q", sheet) == 6


def test_the_title_case_part_line_is_found_and_a_10q_stored_without_a_part_line_is_read_as_part_one():
    from langchain_core.documents import Document
    from prometheus.agents.data_clerk import edgar_text
    assert edgar_text.PART_LINE.fullmatch("Part I. Financial Information") and edgar_text.PART_LINE.fullmatch("PART II - OTHER INFORMATION") and not edgar_text.PART_LINE.fullmatch("Part I, Item 1A of the Form 10-K")

    def doc(section, body):  # the Document the loader makes for a section of a 10-Q: id sec:<accession>:<section>, headline "<SYMBOL> <form>, filed <date>"
        return Document(id=f"sec:0001618921-26-000099:{section}", page_content=f"QSR 10-Q, filed 2026-10-29\n\n{body}", metadata={"headline": "QSR 10-Q, filed 2026-10-29", "url": "u"})
    stored = fs.filings_from_documents([doc("item1", "statements"), doc("item2", "management discussion"), doc("item3", "market risk"), doc("item4", "controls"), doc("item5", "other")])
    assert set(stored[0]["sections"]) == {"financial_statements", "mdna", "market_risk", "controls", "item5"} and stored[0]["sections"]["financial_statements"] == "statements"
    named = fs.filings_from_documents([doc("financial_statements", "statements"), doc("mdna", "management discussion")])                       # a filing that was named properly is left alone
    assert set(named[0]["sections"]) == {"financial_statements", "mdna"}


def test_negative_equity_is_true_for_a_deficit_false_for_positive_equity_and_missing_without_a_balance_sheet():
    from prometheus.agents.data_clerk import fundamentals as fund
    deficit = fund.negative_equity({"balance_sheet": {"items": {"total_equity": {"current": -1023.0, "label": "Total shareholders\u2019 equity (deficit)"}}, "units": "In millions"}})
    assert deficit[0] is True and deficit[1]["total_equity_as_reported"] == -1023.0
    assert fund.negative_equity({"balance_sheet": {"items": {"total_equity": {"current": 5.0}}, "units": None}})[0] is False
    with pytest.raises(fund.Missing):
        fund.negative_equity({"balance_sheet": {"items": {}}})
    with pytest.raises(fund.Missing):
        fund.negative_equity({})
