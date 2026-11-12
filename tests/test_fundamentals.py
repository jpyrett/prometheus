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


def test_apples_net_payout_yield_is_worked_from_the_cash_flow_and_annualized(apple):
    """Three months: 24,701 million of buybacks and 3,921 of dividends (the 10-Q's figures), no row of stock issued: (24,701 + 3,921) x 4 over a market cap of 4 trillion."""
    value, summary = f.net_payout_yield_ttm(apple, 4.0e12)
    assert value == pytest.approx((24701e6 + 3921e6) * 4 / 4.0e12)
    assert summary["months"] == 3 and summary["absent"] == ["proceeds from issuing common stock"] and summary["repurchases"] == 24701e6


def test_metas_net_payout_yield_uses_nine_months_and_a_buyback_that_stopped(meta):
    value, summary = f.net_payout_yield_ttm(meta, 1.6e12)
    assert value == pytest.approx(4060e6 * 12 / 9 / 1.6e12)         # repurchases 0 (a row that says zero), dividends 4,060 million, nine months annualized by 12 / 9
    assert summary["months"] == 9 and summary["repurchases"] == 0.0 and "repurchases" not in summary["absent"]


@pytest.mark.parametrize("change, reason", [({"period_months": None}, "months"), ({"capital_actions": []}, "no repurchases or dividends"), ({}, None)])
def test_net_payout_yield_says_what_is_missing(apple, change, reason):
    evidence = {**apple, **change}
    if reason is None:
        with pytest.raises(f.Missing, match="market capitalization"):
            f.net_payout_yield_ttm(evidence, None)
        with pytest.raises(f.Missing, match="market capitalization"):
            f.net_payout_yield_ttm(evidence, 0)
    else:
        with pytest.raises(f.Missing, match=reason):
            f.net_payout_yield_ttm(evidence, 4.0e12)


def test_proceeds_from_issuing_stock_are_taken_off():
    evidence = {"period_months": 12, "capital_actions": [{"item": "repurchases of common stock", "current": -100.0, "units": "(In millions)"},
                                                        {"item": "dividends paid", "current": -50.0, "units": "(In millions)"},
                                                        {"item": "proceeds from issuing common stock", "current": 30.0, "units": "(In millions)"}]}
    value, summary = f.net_payout_yield_ttm(evidence, 10_000e6)
    assert value == pytest.approx((100e6 + 50e6 - 30e6) / 10_000e6) and summary["absent"] == []


def test_share_count_change_is_the_balance_sheets_change_as_a_fraction(apple, meta):
    value, summary = f.share_count_change(apple)
    assert value == pytest.approx(apple["balance_sheet"]["shares"]["change_pct"] / 100) and value < 0   # Apple's count fell (it buys back stock)
    assert summary["current"] == apple["balance_sheet"]["shares"]["current"]
    assert f.share_count_change(meta)[0] > 0                                                              # META's rose
    with pytest.raises(f.Missing, match="no shares"):
        f.share_count_change({"balance_sheet": {"shares": None}})
    with pytest.raises(f.Missing, match="no shares"):
        f.share_count_change({})


def test_payout_coverage_sbc_and_the_accrual_ratios_use_the_same_period(apple, meta):
    assert f.payout_coverage(apple)[0] == pytest.approx(3921 / 42097)               # dividends over net income, the quarter
    assert f.payout_coverage(meta)[0] == pytest.approx(4060 / 52481)                # net income of the NINE months (the third of four columns), not the quarter's 9,860
    assert f.sbc_to_revenue(apple)[0] == pytest.approx(3594 / 143756)
    assert f.sbc_to_revenue(meta)[0] == pytest.approx(21053 / 181712)
    cfo = next(a["current"] for a in apple["capital_actions"] if a["item"] == "cash from operations")
    assets = apple["balance_sheet"]["items"]["total_assets"]["current"]
    value, summary = f.accrual_ratio(apple)
    assert value == pytest.approx((42097 - cfo) / assets) and summary["total_assets"] == assets * 1e6
    assert f.cfo_negative_with_profit(apple)[0] is False and f.cfo_negative_with_profit(meta)[0] is False


def test_cash_balance_is_the_balance_sheets_cash_in_dollars(apple):
    cash = apple["balance_sheet"]["items"]["cash"]["current"]
    value, summary = f.cash_balance(apple)
    assert value == cash * 1e6 and summary["as_reported"] == cash
    with pytest.raises(f.Missing, match="no cash"):
        f.cash_balance({})


def test_receivables_and_inventory_against_sales_are_percentage_points(apple):
    receivables = apple["balance_sheet"]["items"]["accounts_receivable"]["change_pct"]
    value, summary = f.receivables_vs_sales(apple)
    assert value == pytest.approx(receivables - (143756 - 124300) / 124300 * 100) and summary["revenue_change_pct"] == pytest.approx(15.65, abs=0.01)
    inventory = apple["balance_sheet"]["items"]["inventories"]["change_pct"]
    assert f.inventory_vs_sales(apple)[0] == pytest.approx(inventory - (143756 - 124300) / 124300 * 100)


def test_every_ratio_is_missing_not_guessed_when_its_inputs_are_not_there():
    for fn in (f.payout_coverage, f.sbc_to_revenue, f.accrual_ratio, f.cfo_negative_with_profit, f.receivables_vs_sales, f.inventory_vs_sales):
        with pytest.raises(f.Missing):
            fn({})


def test_a_negative_net_income_has_no_payout_coverage():
    evidence = {"capital_actions": [{"item": "dividends paid", "current": -5.0, "units": "(In dollars)"}], "income": [{"item": "net income", "values": [-10.0, 4.0], "units": "(In dollars)"}]}
    with pytest.raises(f.Missing, match="not positive"):
        f.payout_coverage(evidence)


def test_nothing_here_scores_anything(apple, meta):
    """The module's ratios are numbers and booleans; their inputs summaries hold no judgement words."""
    for evidence in (apple, meta):
        for fn in (f.share_count_change, f.payout_coverage, f.sbc_to_revenue, f.accrual_ratio, f.cfo_negative_with_profit, f.receivables_vs_sales):
            try:
                _, summary = fn(evidence)
            except f.Missing:
                continue
            assert not fs.keys_of(summary) & set(fs.FORBIDDEN_KEYS)


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
