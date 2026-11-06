"""Tests for `Advisor.generate_recommendations` (Advisor v1 spec section 7, phase 6, design choices entry 51): the whole step on fakes.

A scripted model replies, a fake Data Clerk gives the numbers, a ledger lives in memory: no network, no key, no cost. What is checked is the step's behaviour: the status, the records
validate, the rejects are in `candidates` with their reasons, `advisor_runs` has the plan and the thesis, the ledger got one line per record, version 0's `advise` still runs, and
`top_recommendation` picks the right symbol from the v1 score.
"""
import json

import pytest
from advisor_helpers import (EMPTY_PORTFOLIO, PG_EVENT, PG_QUOTE, PG_SIGNAL, TODAY, FakeClerk, MemoryLedger, Scripted, advisor_with, company, full_metrics, plan_reply, ready_order,
                             thesis_reply)

from prometheus.agents.account_manager.account_manager import top_recommendation
from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.advisor.record import MODEL_FIELDS, Candidate, Recommendation
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, Position

GOOD = (plan_reply(company(signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply())


def run(*replies, order=None, portfolio=EMPTY_PORTFOLIO, case="funds", **kw):
    advisor, model = advisor_with(*replies, **kw)
    out = advisor.generate_recommendations(order or ready_order(), portfolio, case, TODAY)
    return out, advisor, model


def test_the_whole_step_on_the_fixture_universe(tmp_path):
    out, advisor, model = run(*GOOD)
    assert out.status == "recommended" and out.errors == [] and out.schema_version == 2
    assert [r["symbol"] for r in out.recommendations] == ["PG"]
    record = Recommendation.model_validate(out.recommendations[0])                          # the record validates against schema 2
    assert record.action.action == "open" and record.identity.case == "funds" and record.identity.sleeve == "preserve" and record.identity.regime == "trend" and record.identity.as_of_utc == "2026-10-23T00:00:00Z"
    assert record.ranking.rank == 1 and record.ranking.concentrated is True and record.stance == "bullish" and record.supported is True and 50 < record.score <= 100
    assert [f.family for f in record.families if f.passed] == ["event", "capital_return"] and record.trade.family == "capital_return" and record.trade.suggested_shares >= 1
    assert record.thesis.thesis.startswith("Open Procter") and record.thesis.thesis_error is None
    assert [g.gate for g in record.gates] == ["no_metrics", "liquidity", "freshness", "red_flag", "event_risk"] and all(g.passed for g in record.gates)
    assert record.audit.config_version == "1" and record.audit.metric_order == out.advisor_runs[0]["metric_order"] and record.audit.plan_run["calls"] == 1 and record.audit.thesis_run["calls"] == 1
    assert [(r["stage"], r["calls"], r["error"]) for r in out.advisor_runs] == [("plan", 1, None), ("thesis", 1, None)] and len(model.seen) == 2     # the model is called exactly twice
    assert len(advisor.ledger.records) == 1 and advisor.ledger.records[0]["identity"]["record_id"] == record.identity.record_id and out.candidates == []


def test_the_record_carries_everything_the_critic_needs():
    record = run(*GOOD)[0].recommendations[0]
    assert set(record) >= {"identity", "action", "families", "quality", "evidence", "features", "agreement", "ranking", "trade", "risk", "portfolio_fit", "gates", "thesis", "audit", "signals", "advisor"}
    assert record["evidence"]["events"][0]["class"] == "guidance_raise" and record["evidence"]["events"][0]["age_bars"] == 1 and record["evidence"]["signals"][0]["quote"].startswith("Procter")
    assert record["features"]["roe"] == {"value": 20.0, "unit": "x", "as_of": "2026-10-23", "source": "test"} and "recent_bars" in record["features"]
    assert record["trade"]["stop_price"] is not None and record["trade"]["invalidation"][0]["kind"] == "price_below" and record["trade"]["time_stop_date"] == "2027-04-19"
    assert record["agreement"]["evidence_n"] == 1 and record["agreement"]["confidence"] == pytest.approx((1 + record["blend"]["value"]) / 2, abs=1e-4)   # entry 55: the blend, not the agreement of the text alone
    assert record["blend"]["score"] == round(50 + 50 * record["blend"]["value"]) and set(record["blend"]["components"]) == {"news", "performance", "historical", "filings"}
    capital = next(f for f in record["families"] if f["family"] == "capital_return")
    assert capital["features_used"]["net_payout_yield_ttm"] == 0.05 and capital["basis"] == "prior" and capital["basis_ref"] == "boudoukh_et_al_2007"
    trend = next(f for f in record["families"] if f["family"] == "trend")
    assert trend["computable"] is False and "mom_12_1" in trend["missing"]                  # a family that could not run says what it needed
    json.dumps(record)                                                                      # plain JSON: store it, log it, hand it to the Critic


def test_no_field_of_the_record_comes_from_the_model_except_the_ones_listed():
    """Run twice with a model that says different things in every field the model owns: only those fields differ."""
    def result(score, company_name, thesis_text, quote_signals):
        reply = plan_reply(company(name=company_name, signals=quote_signals, events=[PG_EVENT], model_score=score))
        out = run(reply, thesis_reply(thesis=thesis_text))[0]
        return out.recommendations[0]
    one = result(10, "Procter & Gamble", "Open it: the expected return is an unmeasured prior.", [PG_SIGNAL])
    two = result(99, "Procter and Gamble Co", "Hold your nose, but the return is an unmeasured prior.", [PG_SIGNAL])
    assert MODEL_FIELDS == ("model_score", "thesis", "bull_case", "bear_case", "signals[*].quote", "events[*].quote", "company")

    def scrub(rec):
        rec = json.loads(json.dumps(rec))
        for key in ("model_score", "company"):
            rec.pop(key)
        rec["thesis"] = None
        rec["identity"]["record_id"] = rec["identity"]["request_id"] = rec["audit"]["metric_order"] = None       # ids drawn at random for each run
        return rec
    assert one["model_score"] == 10 and two["model_score"] == 99 and scrub(one) == scrub(two)       # the model's numbers change nothing a program computed


def test_a_model_score_of_100_does_not_make_a_symbol_pass_a_gate_or_a_family():
    out = run(plan_reply(company(model_score=100, signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply(), metrics={"PG": full_metrics(adv_20d_dollars=1.0)})[0]
    assert out.recommendations == [] and out.candidates[0]["symbol"] == "PG" and out.candidates[0]["gate"] == "liquidity"


# ---------------------------------------------------------------- preconditions

def test_an_order_without_its_data_is_refused_in_v0s_words_and_nothing_is_called():
    order = wo.new_work_order("x", subject=["PG"])
    advisor, model = advisor_with()
    out = advisor.generate_recommendations(order, EMPTY_PORTFOLIO, "funds", TODAY)
    assert out is order and out.errors == ["the Advisor needs an order whose data is ready (status new)"] and out.status == "new" and model.seen == [] and out.advisor_runs == []


def test_the_market_case_is_refused_because_the_brief_is_not_built():
    out, _, model = run(case="market")
    assert out.status == "data_ready" and "case 'market' is not supported" in out.errors[0] and model.seen == []


def test_the_case_comes_from_the_argument_then_the_order_then_the_subject():
    assert run(*GOOD)[0].recommendations[0]["identity"]["case"] == "funds"
    assert run(*GOOD, order=ready_order(case="funds"), case=None)[0].recommendations[0]["identity"]["case"] == "funds"
    order = ready_order(subject=("PG",))
    out, advisor, _ = run(*GOOD, order=order, case=None)
    assert out.recommendations[0]["identity"]["case"] == "funds" and advisor.clerk.asked[0][0] == "metrics"


def test_nothing_to_look_at_is_a_result_not_a_failure():
    order = ready_order(subject="a term with no companies", sources=[(1, "MU", "Memory chip stocks have declined roughly one-third from their July peak.", "2026-10-22T00:00:00Z")])
    out, _, model = run(order=order, case="theme", portfolio=Portfolio(user="t"), metrics={})
    assert out.status == "recommended" and out.recommendations == [] and model.seen == []


# ---------------------------------------------------------------- the data clerk

def test_one_metrics_order_asks_for_every_number_for_every_candidate_with_the_sector_of_the_subject():
    order = ready_order(sector="XLP")
    out, advisor, _ = run(*GOOD, order=order)
    kind, symbols, names, sector = advisor.clerk.asked[0]
    assert kind == "metrics" and symbols == ["PG"] and sector == {"PG": "XLP"} and len([a for a in advisor.clerk.asked if a[0] == "metrics"]) == 1
    assert {"rsi_14", "mom_12_1", "net_payout_yield_ttm", "recent_bars", "adv_20d_dollars", "atr_14", "spy_above_sma_200", "sector_etf", "roe", "risk_factor_change"} <= set(names) and len(names) == len(set(names))
    assert "metrics" in out.data.model_fields_set or out.data.metrics                  # the numbers are on the order for the next agent


def test_a_failed_metrics_order_does_not_stop_the_step_the_symbols_become_candidates_with_no_metrics():
    out, _, _ = run(*GOOD, clerk=FakeClerk({}, fail=True))
    assert out.status == "recommended" and out.recommendations == [] and "metrics: boom" in out.errors
    assert out.candidates[0]["symbol"] == "PG" and out.candidates[0]["gate"] == "no_metrics" and out.candidates[0]["reason"].startswith("gate no_metrics failed")


# ---------------------------------------------------------------- the model calls

def test_a_plan_reply_that_is_not_json_is_retried_once():
    out, _, model = run("sorry, here is some prose", *GOOD)
    assert out.status == "recommended" and [r["calls"] for r in out.advisor_runs] == [2, 1] and len(model.seen) == 3 and "was not valid JSON" in model.seen[1][-1].content
    assert out.recommendations[0]["signals_verified"] == 1


def test_without_the_plan_call_there_are_no_text_signals_but_the_families_that_need_none_still_run():
    out, _, _ = run(RuntimeError("provider down"), thesis_reply(), order=ready_order())
    assert out.status == "recommended" and out.advisor_runs[0]["error"] == "the plan model call failed (RuntimeError)" and out.errors == ["the plan model call failed (RuntimeError)"]
    record = out.recommendations[0]
    assert record["symbol"] == "PG" and record["signals"] == [] and record["agreement"]["evidence_n"] == 0 and [f["family"] for f in record["families"] if f["passed"]] == ["capital_return"]


def test_an_invented_quote_an_invented_event_and_an_event_the_text_does_not_show_are_dropped_with_reasons():
    reply = plan_reply(company(signals=[PG_SIGNAL, {"source": 1, "direction": "bullish", "quote": "Procter & Gamble will double its profit next year"}],
                               events=[PG_EVENT, {"class": "going_concern", "source": 1, "quote": PG_QUOTE}, {"class": "guidance_cut", "source": 9, "quote": PG_QUOTE}]))
    out, _, _ = run(reply, thesis_reply())
    record = out.recommendations[0]
    assert record["signals_claimed"] == 2 and record["signals_verified"] == 1 and [e["class"] for e in record["evidence"]["events"]] == ["guidance_raise"]
    dropped = out.advisor_runs[0]["dropped"]
    assert any("going_concern" in d and "does not show going_concern" in d for d in dropped) and any("guidance_cut" in d and "source 9" in d for d in dropped)


def test_a_symbol_the_model_names_that_is_not_a_candidate_is_dropped_unless_it_has_a_verified_signal():
    reply = plan_reply(company(signals=[PG_SIGNAL]), company("ZZZZ", "Zed Corp", signals=[{"source": 1, "direction": "bullish", "quote": "Zed Corp rose"}]))
    out, _, _ = run(reply, thesis_reply())
    assert any("ZZZZ: not a candidate and no verified signal" in d for d in out.advisor_runs[0]["dropped"])


def test_a_thesis_that_fails_twice_leaves_null_and_the_reason_and_the_record_is_still_returned():
    bad = thesis_reply(thesis="It will triple to 37 dollars: an unmeasured prior.")
    out, _, _ = run(GOOD[0], bad, bad)
    record = out.recommendations[0]
    assert record["symbol"] == "PG" and record["thesis"]["thesis"] is None and "numbers that are not in the record: 37" in record["thesis"]["thesis_error"]
    assert out.advisor_runs[1]["stage"] == "thesis" and out.advisor_runs[1]["calls"] == 2 and out.status == "recommended"


# ---------------------------------------------------------------- candidates, gates, ledger, cooldown, holdings

def test_the_rejects_are_candidates_with_the_gate_or_the_rank_that_stopped_them():
    sources = [(1, "PG", "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition, while the company reported Q1 earnings.", "2026-10-22T13:00:43Z"),
               (2, "MU", "Micron (MU) shares rallied on Tuesday after the quarterly report.", "2026-10-22T10:00:00Z")]
    metrics = {"PG": full_metrics(), "MU": full_metrics(adv_20d_dollars=1000.0)}
    out, _, _ = run(plan_reply(company(signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply(), order=ready_order(subject=("PG", "MU"), sources=sources), metrics=metrics)
    assert [r["symbol"] for r in out.recommendations] == ["PG"]
    reject = Candidate.model_validate(out.candidates[0])
    assert reject.symbol == "MU" and reject.gate == "liquidity" and reject.origin == "subject" and "gate liquidity failed" in reject.reason and reject.gate_value["adv_20d_dollars"] == 1000.0


def test_the_stale_bars_of_today_fail_freshness_and_say_how_stale(monkeypatch):
    advisor, _ = advisor_with(*GOOD)
    from datetime import date
    out = advisor.generate_recommendations(ready_order(), EMPTY_PORTFOLIO, "funds", date(2026, 11, 5))     # the newest stored bar is 2026-10-23: nine sessions earlier
    assert out.recommendations == [] and out.candidates[0]["gate"] == "freshness" and "9 sessions before 2026-11-05" in out.candidates[0]["reason"]


def test_the_ledger_gets_one_line_per_record_and_a_cooldown_keeps_the_same_family_from_opening_again():
    first, advisor, _ = run(*GOOD)
    earlier = advisor.ledger.records[0]
    again, _, _ = run(*GOOD, ledger=MemoryLedger([earlier]))
    assert again.recommendations == [] and again.candidates[0]["symbol"] == "PG" and again.candidates[0]["cooldown_of"] == earlier["identity"]["record_id"]
    assert again.candidates[0]["reason"].startswith("cooldown: an earlier capital_return recommendation")
    invalidated = json.loads(json.dumps(earlier))
    invalidated["trade"]["invalidation"] = [{"kind": "price_below", "value": 150.0}]                    # the price (100) is under its stop: the idea was invalidated
    free, _, _ = run(*GOOD, ledger=MemoryLedger([invalidated]))
    assert [r["symbol"] for r in free.recommendations] == ["PG"]


def test_a_holding_with_a_red_flag_is_a_reduce_returned_above_every_open():
    portfolio = Portfolio(user="t", cash_avail=500.0, positions=[Position(symbol="MCD", name="McDonald's Corp", shares=10, cost_basis=40.0)])
    metrics = {"PG": full_metrics(), "MCD": full_metrics(dilution_current=True, last_close=300.0)}
    out, _, _ = run(plan_reply(company(signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply(), thesis_reply(), order=ready_order(subject=("PG",)), portfolio=portfolio, metrics=metrics)
    assert [(r["symbol"], r["action"]["action"]) for r in out.recommendations] == [("MCD", "reduce"), ("PG", "open")]
    reduce = out.recommendations[0]
    assert reduce["action"]["held_shares"] == 10 and reduce["action"]["unrealized_pct"] == pytest.approx(300 / 40 - 1, abs=1e-4) and "red_flag gate failed" in reduce["action"]["reason"]
    assert reduce["trade"]["suggested_shares"] == 0 and reduce["ranking"]["rank"] == 1 and out.recommendations[1]["ranking"]["rank"] == 2          # a reduction has no invented size


def test_a_holding_with_nothing_to_say_is_a_hold_candidate_not_a_recommendation():
    portfolio = Portfolio(user="t", cash_avail=500.0, positions=[Position(symbol="MCD", name="McDonald's Corp", shares=10, cost_basis=40.0)])
    out, _, _ = run(*GOOD, order=ready_order(subject=("PG",)), portfolio=portfolio, metrics={"PG": full_metrics(), "MCD": full_metrics()})
    held = next(c for c in out.candidates if c["symbol"] == "MCD")
    assert held["held"] is True and held["origin"] == "held" and held["reason"].startswith("hold:")


# ---------------------------------------------------------------- v0 and the Account Manager

def test_version_0_still_advises_on_the_same_kind_of_order_and_top_recommendation_reads_the_v1_score():
    advisor, _ = advisor_with(json.dumps({"recommendations": [{"symbol": "PG", "company": "Procter & Gamble", "stance": "bullish", "model_score": 70, "signals": [PG_SIGNAL]}]}))
    v0 = advisor.advise(ready_order(), today=TODAY)
    assert v0.status == "recommended" and v0.recommendations[0]["symbol"] == "PG" and v0.recommendations[0]["score"] == 67        # (1 + 1) / (1 + 2): v0's formula, untouched
    out, _, _ = run(*GOOD)
    assert top_recommendation(out) == "PG"                                                                                           # supported, and its v1 score is above 50
    both = ready_order(subject=("PG", "MU"), sources=[(1, "PG", "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition, while the company reported Q1 earnings.", "2026-10-22T13:00:43Z")])
    out2, _, _ = run(plan_reply(company(signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply(), thesis_reply(symbol="MU"), order=both, metrics={"PG": full_metrics(), "MU": full_metrics(roe=30.0, net_payout_yield_ttm=0.08, share_count_change=-0.05)})
    assert {r["symbol"] for r in out2.recommendations} == {"PG", "MU"} and top_recommendation(out2) == max(out2.recommendations, key=lambda r: r["score"])["symbol"]


def test_the_orders_json_round_trips_with_the_new_fields():
    out = run(*GOOD)[0]
    back = wo.WorkOrder.model_validate_json(out.model_dump_json())
    assert back.recommendations == out.recommendations and back.candidates == out.candidates and back.schema_version == 2


# ---------------------------------------------------------------- the theme case

def test_the_theme_case_discovers_the_basket_and_recommends_the_laggard():
    rows = [{"symbol": "PG", "articles": 12, "positive": 9, "neutral": 2, "negative": 1}, {"symbol": "MU", "articles": 6, "positive": 5, "neutral": 1, "negative": 0}]
    metrics = {"PG": full_metrics(ret_21d=0.01), "MU": full_metrics(ret_21d=0.40)}
    micron = "Micron (MU) shares rallied on Tuesday after the quarterly report."
    order = ready_order(subject="AI memory shortage", sources=[(1, "PG", "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition, while the company reported Q1 earnings.", "2026-10-22T13:00:43Z"),
                                                               (2, "MU", micron, "2026-10-22T10:00:00Z")])
    mu = company("MU", "Micron", signals=[{"source": 2, "direction": "bullish", "quote": micron}])
    out, advisor, _ = run(plan_reply(company(signals=[PG_SIGNAL]), mu), thesis_reply(), order=order, case="theme", portfolio=Portfolio(user="t", cash_avail=1000.0), metrics=metrics, discover=rows)
    assert advisor.clerk.asked[0] == ("discover", "AI memory shortage")
    assert [r["symbol"] for r in out.recommendations] == ["PG"] and out.recommendations[0]["identity"]["sleeve"] == "theme" and out.recommendations[0]["trade"]["family"] in ("theme", "capital_return")
    leader = next(c for c in out.candidates if c["symbol"] == "MU")
    assert leader["theme_role"] == "leader" and "theme family did not pass" in leader["reason"]


# ---------------------------------------------------------------- the industry case: leaders ranked best to worst, top_k, and a rank for every company evaluated (entry 53)

def theses_reply(*symbols):
    return json.dumps({"theses": [{"symbol": s, "thesis": "Open it on its capital return: the expected excess return is an unmeasured prior.", "bull_case": [{"text": "Return on equity is 20.", "ref": "roe"}], "bear_case": []} for s in symbols]})


def industry_run(top_k=None, **kw):
    texts = [(1, "AAA", "AAA (AAA) is the largest chain in the industry.", "2026-10-22T10:00:00Z"), (2, "BBB", "BBB (BBB) is the second largest chain.", "2026-10-22T10:00:00Z"),
             (3, "CCC", "CCC (CCC) is a smaller chain.", "2026-10-22T10:00:00Z"), (4, "DDD", "DDD (DDD) is a chain with thin trading.", "2026-10-22T10:00:00Z")]
    metrics = {"AAA": full_metrics(net_payout_yield_ttm=0.07, share_count_change=-0.03, ret_21d=0.30), "BBB": full_metrics(net_payout_yield_ttm=0.05), "CCC": full_metrics(net_payout_yield_ttm=0.06, share_count_change=-0.006),
               "DDD": full_metrics(net_payout_yield_ttm=0.08, adv_20d_dollars=1000.0)}          # DDD has the best numbers and fails the liquidity gate
    order = ready_order(subject=("AAA", "BBB", "CCC", "DDD"), sources=texts, case="industry")
    advisor, model = advisor_with(plan_reply(), theses_reply("AAA", "BBB", "CCC"), metrics=metrics)
    out = advisor.generate_recommendations(order, EMPTY_PORTFOLIO, "industry", TODAY, top_k)
    return out, advisor, model


def test_the_industry_case_ranks_the_leaders_best_to_worst_and_a_large_leader_is_not_excluded():
    out, _, _ = industry_run()
    assert out.status == "recommended" and out.errors == [] and [r["symbol"] for r in out.recommendations] == ["AAA", "BBB", "CCC"]       # AAA already rose 30% in 21 bars: a theme case would call it a leader and not recommend it
    assert [r["identity"]["case"] for r in out.recommendations] == ["industry"] * 3 and out.recommendations[0]["identity"]["sleeve"] == "industry"
    scores = [r["score"] for r in out.recommendations]
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[-1] and [r["ranking"]["universe_rank"] for r in out.recommendations] == [1, 2, 3] and [r["ranking"]["rank"] for r in out.recommendations] == [1, 2, 3]
    assert all(r["ranking"]["concentrated"] is False for r in out.recommendations)                                                       # the one-bet flag is for the funds sleeve


def test_every_company_evaluated_has_a_score_and_a_rank_and_one_that_failed_a_gate_is_ranked_after_those_that_did_not():
    out, _, _ = industry_run()
    candidate = out.candidates[0]
    assert candidate["symbol"] == "DDD" and candidate["rank"] == 4 and candidate["gate"] == "liquidity" and candidate["stance"] == "bullish" and candidate["score"] > 50     # the best numbers, ranked last: it cannot be bought
    assert "gate liquidity failed" in candidate["reason"]


def test_top_k_replaces_the_number_of_new_ideas_and_the_rest_are_ranked_candidates():
    out, _, _ = industry_run(top_k=2)
    assert [r["symbol"] for r in out.recommendations] == ["AAA", "BBB"]
    assert [(c["symbol"], c["rank"], c["reason"]) for c in out.candidates if c["symbol"] == "CCC"] == [("CCC", 3, "rank 3 is beyond the top 2")]
    one, _, _ = industry_run(top_k=1)
    assert [r["symbol"] for r in one.recommendations] == ["AAA"] and Advisor.__init__ and one.advisor_runs[1]["stage"] == "thesis"


@pytest.mark.parametrize("top_k", [0, 21, True, "5"])
def test_a_bad_top_k_is_refused_and_nothing_is_called(top_k):
    advisor, model = advisor_with(metrics={})
    out = advisor.generate_recommendations(ready_order(), EMPTY_PORTFOLIO, "industry", TODAY, top_k)
    assert out.status == "data_ready" and out.errors == ["top_k must be a whole number from 1 to 20"] and model.seen == []


def test_the_default_number_of_new_ideas_is_three_for_funds_and_five_for_industry_and_the_advisors_settings_are_not_changed():
    advisor, _ = advisor_with()
    industry_cfg = advisor.config.sleeves["industry"].top_k
    industry_run(top_k=2)
    assert industry_cfg == 5 and advisor.config.sleeves["funds"].top_k == 3 and advisor.config.sleeves["theme"].top_k == 5
