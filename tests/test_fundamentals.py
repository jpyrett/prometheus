"""Tests for prometheus.agents.data_clerk.fundamentals (Advisor v1 spec section 5.4 and 5.5, phase 2).

The evidence is the REAL one: Apple's first-quarter 10-Q (three months, the cash flow covers three) and META's third-quarter 10-Q (nine months, four-column income statement), both
read from the real stored pages in tests/fixtures by the real `filing_signals`. Every expected number is worked from the figures on those pages (listed in test_filing_signals.py).
The tests that remove a row use small hand-made evidence dictionaries: they test what happens when a figure is absent.
"""
import pytest
from test_filing_signals import APPLE_TODAY, META_TODAY, store

from prometheus.agents.data_clerk import filing_signals as fs
from prometheus.agents.data_clerk import fundamentals as f


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


def test_units_and_the_year_to_date_pair():
    assert f.multiplier("(In millions, except number of shares)") == 1e6
    assert f.multiplier("(In thousands)") == 1e3 and f.multiplier("(In billions)") == 1e9 and f.multiplier("(in dollars)") == 1.0
    assert f.ytd([1, 2, 3, 4]) == (3, 4)      # three months and year to date, this year and last: the year to date is the second pair
    assert f.ytd([1, 2]) == (1, 2) and f.ytd([1, 2, 3]) is None


def test_earnings_yield():
    assert f.earnings_yield(25.0) == (0.04, {"pe_ratio": 25.0})
    with pytest.raises(f.Missing, match="not positive"):
        f.earnings_yield(-3.0)
    with pytest.raises(f.Missing, match="no P/E"):
        f.earnings_yield(None)


def test_share_count_change_is_the_balance_sheets_change_as_a_fraction(apple, meta):
    value, summary = f.share_count_change(apple)
    assert value == pytest.approx(apple["balance_sheet"]["shares"]["change_pct"] / 100) and value < 0   # Apple's count fell (it buys back stock)
    assert summary["current"] == apple["balance_sheet"]["shares"]["current"]
    assert f.share_count_change(meta)[0] > 0                                                              # META's rose
    with pytest.raises(f.Missing, match="no shares"):
        f.share_count_change({"balance_sheet": {"shares": None}})
    with pytest.raises(f.Missing, match="no shares"):
        f.share_count_change({})


def test_every_ratio_is_missing_not_guessed_when_its_inputs_are_not_there():
    for fn in (f.payout_coverage, f.sbc_to_revenue, f.accrual_ratio, f.cfo_negative_with_profit, f.receivables_vs_sales, f.inventory_vs_sales):
        with pytest.raises(f.Missing):
            fn({})


def test_a_negative_net_income_has_no_payout_coverage():
    evidence = {"capital_actions": [{"item": "dividends paid", "current": -5.0, "units": "(In dollars)"}], "income": [{"item": "net income", "values": [-10.0, 4.0], "units": "(In dollars)"}]}
    with pytest.raises(f.Missing, match="not positive"):
        f.payout_coverage(evidence)


def test_every_statement_is_put_in_dollars_by_its_own_units_line_and_an_unknown_scale_is_never_assumed():
    """Companies are compared in the same unit: millions with millions. A figure of unknown scale is Missing, not dollars (it would be wrong by a thousand or a million)."""
    for line, factor in (("Amounts in millions", 1e6), ("(in millions, except per share amounts)", 1e6), ("($ in thousands)", 1e3), ("(Thousands of dollars)", 1e3), ("(In 000's)", 1e3), ("(000s omitted)", 1e3),
                         ("Dollars in Billions", 1e9), ("(In millions, except number of shares, which are reflected in thousands, and per-share amounts)", 1e6), ("(In dollars)", 1.0), ("Whole dollars", 1.0),
                         ("(in U.S. dollars)", 1.0)):
        assert f.multiplier(line) == factor, line
    for line in (None, "", "(except per share amounts)", "Assets", "(unaudited)"):
        with pytest.raises(f.Missing, match="units of the statement"):
            f.multiplier(line)
    # the same payout in two companies' own units gives the same dollars: 500 million and 500,000 thousand
    a = {"capital_actions": [{"item": "repurchases", "current": -500.0, "units": "(In millions)", "label": "x"}]}
    b = {"capital_actions": [{"item": "repurchases", "current": -500000.0, "units": "(In thousands)", "label": "x"}]}
    assert abs(f._dollars(a["capital_actions"][0])) == abs(f._dollars(b["capital_actions"][0])) == 500e6
    with pytest.raises(f.Missing):
        f._dollars({"item": "repurchases", "current": -500.0, "units": None})
