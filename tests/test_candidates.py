"""Tests for prometheus.agents.advisor.candidates (Advisor v1 spec section 7, step 2, phase 6)."""
from advisor_helpers import NEWS, ready_order

from prometheus.agents.advisor import candidates as cd


def collect(order, case="funds", theme=None, cap=15):
    return cd.collect_candidates(order, case, theme, cap, 3)


def test_a_theme_order_has_no_symbols_in_its_subject_and_takes_the_discover_tickers_with_enough_articles():
    order = ready_order(subject="AI memory shortage", sources=NEWS)
    assert cd.subject_symbols(order) == [] and cd.infer_case(order) == "theme" and cd.theme_term(order) == "AI memory shortage"
    rows = [{"symbol": "MU", "articles": 9}, {"symbol": "SNDK", "articles": 3}, {"symbol": "WDC", "articles": 2}, {"symbol": "bad symbol!", "articles": 9}]
    found, _ = collect(order, case="theme", theme=rows)
    assert [(c["symbol"], c["origin"]) for c in found if c["origin"] != "sources"] == [("MU", "discover"), ("SNDK", "discover")]
    funds, _ = collect(order, case="funds", theme=rows)
    assert all(c["origin"] != "discover" for c in funds)                                                                    # discovery is for the theme case only


def test_symbols_are_cleaned_and_a_subject_can_be_one_symbol_as_text_or_a_dollar_symbol():
    assert cd.subject_symbols(ready_order(subject="$pg")) == ["PG"] and cd.subject_symbols(ready_order(subject=("$aapl", "msft", "AAPL"))) == ["AAPL", "MSFT"]
    assert cd.infer_case(ready_order(subject=("PG",))) == "funds" and cd.theme_term(ready_order(subject=("PG",))) == "Which of these should we buy?"


