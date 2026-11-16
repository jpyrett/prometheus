"""Tests for prometheus.agents.advisor.candidates (Advisor v1 spec section 7, step 2, phase 6)."""
import pytest
from advisor_helpers import NEWS, ready_order

from prometheus.agents.advisor import candidates as cd
from prometheus.common import work_order as wo


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


def test_a_company_the_user_holds_is_an_ordinary_candidate_the_advisor_knows_no_holding():
    """AMENDED (entry 105). This file gave `collect_candidates` a portfolio holding MCD and PG: a held name was a candidate with `origin: "held"` and `held: true`, took its company name from the holding, and took room from the
    new ideas (up to 30 symbols with the holdings). The Advisor is given no portfolio: a candidate is `{symbol, company, origin}`, and a symbol the user holds and names is a candidate like any other."""
    import inspect
    assert "portfolio" not in inspect.signature(cd.collect_candidates).parameters
    order = ready_order(subject=("PG", "MCD"), sources=NEWS)
    found, dropped = collect(order)
    assert [(c["symbol"], c["origin"]) for c in found] == [("PG", "subject"), ("MCD", "subject")] and all(set(c) == {"symbol", "company", "origin"} for c in found) and dropped == []
    many = ready_order(subject=tuple(f"A{n:02d}" for n in range(20)), sources=NEWS)
    kept, cut = collect(many, cap=15)
    assert len(kept) == 15 and "new idea(s) beyond the first 15: A15" in cut[0] and "A19" in cut[0]
