"""Tests for `Advisor.generate_recommendations` (Advisor v1 spec section 7, phase 6, design choices entry 51): the whole step on fakes.

A scripted model replies, a fake Data Clerk gives the numbers, a ledger lives in memory: no network, no key, no cost. What is checked is the step's behaviour: the status, the records
validate, the rejects are in `candidates` with their reasons, `advisor_runs` has the plan and the thesis, the ledger got one line per record, version 0's `advise` still runs, and
`top_recommendation` picks the right symbol from the v1 score.
"""
import json

import pytest
from advisor_helpers import PG_EVENT, PG_SIGNAL, TODAY, FakeClerk, advisor_with, company, full_metrics, plan_reply, ready_order, thesis_reply

from prometheus.agents.account_manager.account_manager import top_recommendation
from prometheus.agents.advisor.advisor import Advisor
from prometheus.common import work_order as wo

GOOD = (plan_reply(company(signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply())


def run(*replies, order=None, case="funds", **kw):
    advisor, model = advisor_with(*replies, **kw)
    out = advisor.generate_recommendations(order or ready_order(), case, TODAY)
    return out, advisor, model


# ---------------------------------------------------------------- preconditions

def test_an_order_without_its_data_is_refused_in_v0s_words_and_nothing_is_called():
    order = wo.new_work_order("x", subject=["PG"])
    advisor, model = advisor_with()
    out = advisor.generate_recommendations(order, "funds", TODAY)
    assert out is order and out.errors == ["the Advisor needs an order whose data is ready (status new)"] and out.status == "new" and model.seen == [] and out.advisor_runs == []


def test_the_market_case_is_refused_because_the_brief_is_not_built():
    out, _, model = run(case="market")
    assert out.status == "data_ready" and "case 'market' is not supported" in out.errors[0] and model.seen == []


def test_nothing_to_look_at_is_a_result_not_a_failure():
    order = ready_order(subject="a term with no companies", sources=[(1, "MU", "Memory chip stocks have declined roughly one-third from their July peak.", "2026-10-22T00:00:00Z")])
    out, _, model = run(order=order, case="theme", metrics={})
    assert out.status == "recommended" and out.recommendations == [] and model.seen == []


# ---------------------------------------------------------------- the data clerk


def test_a_failed_metrics_order_does_not_stop_the_step_the_symbols_become_candidates_with_no_metrics():
    out, _, _ = run(*GOOD, clerk=FakeClerk({}, fail=True))
    assert out.status == "recommended" and out.recommendations == [] and "metrics: boom" in out.errors
    assert out.candidates[0]["symbol"] == "PG" and out.candidates[0]["gate"] == "no_metrics" and out.candidates[0]["reason"].startswith("gate no_metrics failed")


# ---------------------------------------------------------------- the model calls


def test_a_symbol_the_model_names_that_is_not_a_candidate_is_dropped_unless_it_has_a_verified_signal():
    reply = plan_reply(company(signals=[PG_SIGNAL]), company("ZZZZ", "Zed Corp", signals=[{"source": 1, "direction": "bullish", "quote": "Zed Corp rose"}]))
    out, _, _ = run(reply, thesis_reply())
    assert any("ZZZZ: not a candidate and no verified signal" in d for d in out.advisor_runs[0]["dropped"])


# ---------------------------------------------------------------- v0 and the Account Manager

def test_version_0_still_advises_on_the_same_kind_of_order_and_top_recommendation_reads_the_v1_score():
    advisor, _ = advisor_with(json.dumps({"recommendations": [{"symbol": "PG", "company": "Procter & Gamble", "stance": "bullish", "model_score": 70, "signals": [PG_SIGNAL]}]}))
    v0 = advisor.advise(ready_order(), today=TODAY)
    assert v0.status == "recommended" and v0.recommendations[0]["symbol"] == "PG" and v0.recommendations[0]["score"] == 67        # (1 + 1) / (1 + 2): v0's formula, untouched
    out, _, _ = run(*GOOD)
    assert top_recommendation(out) == "PG"                                                                                           # supported, and its v1 score is above 50
    both = ready_order(subject=("PG", "MU"), sources=[(1, "PG", "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition, while the company reported Q1 earnings.", "2026-10-22T13:00:43Z")])
    out2, _, _ = run(plan_reply(company(signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply(), thesis_reply(symbol="MU"), order=both, metrics={"PG": full_metrics(), "MU": full_metrics(roe=30.0, share_count_change=-0.05)})
    assert {r["symbol"] for r in out2.recommendations} == {"PG", "MU"} and top_recommendation(out2) == max(out2.recommendations, key=lambda r: r["score"])["symbol"]


# ---------------------------------------------------------------- the theme case

def test_the_theme_case_discovers_the_basket_and_recommends_the_laggard():
    rows = [{"symbol": "PG", "articles": 12, "positive": 9, "neutral": 2, "negative": 1}, {"symbol": "MU", "articles": 6, "positive": 5, "neutral": 1, "negative": 0}]
    metrics = {"PG": full_metrics(ret_21d=0.01), "MU": full_metrics(ret_21d=0.40)}
    micron = "Micron (MU) shares rallied on Tuesday after the quarterly report."
    order = ready_order(subject="AI memory shortage", sources=[(1, "PG", "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition, while the company reported Q1 earnings.", "2026-10-22T13:00:43Z"),
                                                               (2, "MU", micron, "2026-10-22T10:00:00Z")])
    mu = company("MU", "Micron", signals=[{"source": 2, "direction": "bullish", "quote": micron}])
    out, advisor, _ = run(plan_reply(company(signals=[PG_SIGNAL]), mu), thesis_reply(), order=order, case="theme", metrics=metrics, discover=rows)
    assert advisor.clerk.asked[0] == ("discover", "AI memory shortage")
    assert [r["symbol"] for r in out.recommendations] == ["PG"] and out.recommendations[0]["identity"]["sleeve"] == "theme" and out.recommendations[0]["trade"]["family"] in ("theme", "capital_return")
    leader = next(c for c in out.candidates if c["symbol"] == "MU")
    assert leader["theme_role"] == "leader" and "theme family did not pass" in leader["reason"]


# ---------------------------------------------------------------- the industry case: leaders ranked best to worst, top_k, and a rank for every company evaluated (entry 53)


@pytest.mark.parametrize("top_k", [0, 21, True, "5"])
def test_a_bad_top_k_is_refused_and_nothing_is_called(top_k):
    advisor, model = advisor_with(metrics={})
    out = advisor.generate_recommendations(ready_order(), "industry", TODAY, top_k)
    assert out.status == "data_ready" and out.errors == ["top_k must be a whole number from 1 to 20"] and model.seen == []
