"""Tests for prometheus.agents.advisor.candidates (Advisor v1 spec section 7, step 2, phase 6)."""
import pytest
from advisor_helpers import NEWS, ready_order

from prometheus.agents.advisor import candidates as cd
from prometheus.common.portfolio import Portfolio, Position
from prometheus.common import work_order as wo

PORTFOLIO = Portfolio(user="t", cash_avail=100.0, positions=[Position(symbol="MCD", name="McDonald's Corp", shares=10, cost_basis=40.0), Position(symbol="PG", name="Procter & Gamble Co", shares=5, cost_basis=60.0)])


def collect(order, case="funds", portfolio=PORTFOLIO, theme=None, cap=15):
    return cd.collect_candidates(order, portfolio, case, theme, cap, 3)


def test_the_subject_comes_first_then_the_symbols_the_sources_name_then_the_holdings_tagged_held():
    sources = [(1, "PG", "Procter & Gamble (PG) lifted its outlook.", "2026-10-22T00:00:00Z"), (2, "MU", "Micron (MU) rallied on Tuesday after the report.", "2026-10-21T00:00:00Z")]
    order = ready_order(subject=("SNDK",), sources=sources)
    found, dropped = collect(order)
    assert [(c["symbol"], c["origin"], c["held"]) for c in found] == [("SNDK", "subject", False), ("MU", "sources", False), ("PG", "sources", True), ("MCD", "held", True)] and dropped == []


def test_a_ticker_in_a_list_that_the_text_does_not_name_does_not_qualify():
    """An article lists every company it touches (entry 26): only a ticker the text itself names counts."""
    order = ready_order(subject=("SNDK",), sources=[(1, "MU", "Memory chip stocks have declined roughly one-third from their July peak.", "2026-10-22T00:00:00Z")])
    assert [c["symbol"] for c in collect(order, portfolio=Portfolio(user="t"))[0]] == ["SNDK"]


def test_a_held_symbol_that_is_also_the_subject_is_a_subject_and_held_and_has_the_company_name():
    order = ready_order(subject=("PG",), company="Procter & Gamble Co")
    found, _ = collect(order)
    assert found[0] == {"symbol": "PG", "company": "Procter & Gamble Co", "origin": "subject", "held": True} and [c["symbol"] for c in found] == ["PG", "MCD"]


def test_a_theme_order_has_no_symbols_in_its_subject_and_takes_the_discover_tickers_with_enough_articles():
    order = ready_order(subject="AI memory shortage", sources=NEWS)
    assert cd.subject_symbols(order) == [] and cd.infer_case(order) == "theme" and cd.theme_term(order) == "AI memory shortage"
    rows = [{"symbol": "MU", "articles": 9}, {"symbol": "SNDK", "articles": 3}, {"symbol": "WDC", "articles": 2}, {"symbol": "bad symbol!", "articles": 9}]
    found, _ = collect(order, case="theme", theme=rows, portfolio=Portfolio(user="t"))
    assert [(c["symbol"], c["origin"]) for c in found if c["origin"] != "sources"] == [("MU", "discover"), ("SNDK", "discover")]
    funds, _ = collect(order, case="funds", theme=rows, portfolio=Portfolio(user="t"))
    assert all(c["origin"] != "discover" for c in funds)                                                                    # discovery is for the theme case only


def test_symbols_are_cleaned_and_a_subject_can_be_one_symbol_as_text_or_a_dollar_symbol():
    assert cd.subject_symbols(ready_order(subject="$pg")) == ["PG"] and cd.subject_symbols(ready_order(subject=("$aapl", "msft", "AAPL"))) == ["AAPL", "MSFT"]
    assert cd.infer_case(ready_order(subject=("PG",))) == "funds" and cd.theme_term(ready_order(subject=("PG",))) == "Which of these should we buy?"


def test_new_ideas_are_capped_with_the_subject_first_then_by_sources_then_alphabetically_and_the_cut_is_said():
    sources = [(i, t, f"{t} stock rose on {t} news today.", "2026-10-22T00:00:00Z") for i, t in enumerate(["CCC", "BBB", "BBB", "AAA", "DDD"], 1)]
    order = ready_order(subject=("ZZZ",), sources=sources)
    found, dropped = collect(order, portfolio=Portfolio(user="t"), cap=3)
    assert [c["symbol"] for c in found] == ["ZZZ", "BBB", "AAA"] and dropped == ["2 new idea(s) beyond the first 3: CCC, DDD"]


def test_the_holdings_are_never_cut_and_the_metrics_order_stays_within_30_symbols():
    positions = [Position(symbol=f"H{i}", shares=1, cost_basis=1.0) for i in range(20)]
    big = Portfolio(user="t", positions=positions)
    sources = [(i, f"N{i:02d}", f"N{i:02d} stock rose.", "2026-10-22T00:00:00Z") for i in range(1, 16)]
    found, dropped = collect(ready_order(subject=("PG",), sources=sources), portfolio=big)
    assert len(found) == 30 and sum(c["held"] for c in found) == 20 and len(found) <= wo.MAX_METRIC_SYMBOLS and "new idea(s) beyond the first 10" in dropped[0]
    cut_held = Portfolio(user="t", positions=[*positions, Position(symbol="N15", shares=1, cost_basis=1.0)])
    found, _ = collect(ready_order(subject=("PG",), sources=sources), portfolio=cut_held)
    assert "N15" in [c["symbol"] for c in found]                                                                            # a cut idea that is held is still reviewed as a holding
