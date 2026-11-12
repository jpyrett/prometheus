"""Tests for prometheus.agents.advisor.thesis (Advisor v1 spec section 11.2, phase 6): the digest, each check on its own, the retry once, and `thesis: null` after two failures.

The record under test is the one the whole step writes for PG with the fakes of advisor_helpers (the same record the Critic will read)."""
import json

import pytest
from advisor_helpers import EMPTY_PORTFOLIO, PG_EVENT, PG_SIGNAL, TODAY, Scripted, advisor_with, company, plan_reply, ready_order, thesis_reply

from prometheus.agents.advisor import thesis as th
from prometheus.agents.advisor.calls import new_run


@pytest.fixture(scope="module")
def record():
    advisor, _ = advisor_with(plan_reply(company(signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply())
    order = advisor.generate_recommendations(ready_order(), EMPTY_PORTFOLIO, "funds", TODAY)
    return order.recommendations[0]


def item(**kw):
    base = {"symbol": "PG", "thesis": "Open Procter & Gamble on its capital return: the expected excess return is an unmeasured prior.", "bull_case": [{"text": "It lifted its outlook.", "ref": "[1]"}], "bear_case": []}
    base.update(kw)
    return base


def test_the_digest_is_fixed_wording_from_the_record_and_never_shows_the_models_own_score(record):
    text = th.digest(record)
    assert text.startswith("RECORD PG (Procter & Gamble)\nDecision: open (") and "model_score" not in text and "model" not in text.split("Expected:")[0].split("Decision:")[0]
    assert "Family capital_return: passed, bullish, score 0.7875" in text and "Family trend: not computable, missing mom_12_1" in text
    assert "Score: 71 of 100 (the blend of news, performance, historical metrics and filings), stance bullish. No return is promised" in text and "ALLOWED REFS: [1], guidance_raise, " in text and 'Evidence [1] bullish: "Procter & Gamble (PG) lifted' in text
    assert "Gates: no_metrics ok; liquidity ok; freshness ok; red_flag ok; event_risk ok (no earnings date known" in text


def test_the_refs_a_bullet_may_cite_are_signals_evidence_events_and_metrics(record):
    refs = th.refs_of(record)
    assert refs["[1]"] == "source" and refs["guidance_raise"] == "event" and refs["roe"] == "metric" and "[2]" not in refs


def test_a_good_thesis_passes(record):
    assert th.check(item(), record) == []
    assert th.check(item(bull_case=[{"text": "Return on equity is 20.", "ref": "roe"}, {"text": "A guidance raise.", "ref": "guidance_raise"}]), record) == []


def test_at_most_three_sentences(record):
    four = "One. Two. Three. Four is an unmeasured prior."
    assert th.check(item(thesis=four), record) == ["the thesis has 4 sentences (at most 3)"]
    assert th.check(item(thesis="ROE is 20.0 and that is an unmeasured prior. Two. Three."), record) == []                  # a decimal point is not a sentence end
    assert len(th.sentences("Price is 163.45 per share. It is up 4.2% today! Fine?")) == 3


@pytest.mark.parametrize("ref", ["[9]", "E7", "made_up_metric", "capital_return_strategy"])
def test_a_ref_that_is_not_in_the_record_fails(record, ref):
    assert any("is not one of the record's allowed refs" in r for r in th.check(item(bull_case=[{"text": "x", "ref": ref}]), record))


def test_a_ref_may_be_written_with_or_without_its_brackets(record):
    assert th.check(item(bull_case=[{"text": "It lifted its outlook.", "ref": "1"}]), record) == []


def test_a_number_the_record_does_not_have_fails_and_a_rounded_one_passes(record):
    assert any("numbers that are not in the record: 37" in r for r in th.check(item(thesis="Buy it at 37 dollars: an unmeasured prior."), record))
    assert th.check(item(thesis="ROE is 20 and the price is $100.00, an unmeasured prior."), record) == []                      # 20 and 100.0 are in the record
    assert th.check(item(thesis="The beta is 0.7 and the cost is 0.20%: an unmeasured prior."), record) == []                    # 0.7 is beta_63d; 0.002 is the cost, shown as a percent
    assert th.check(item(thesis="The market cap is 300 billion: an unmeasured prior."), record) == []                              # 3e11 shown in billions
    assert any("0.9137" in r for r in th.check(item(thesis="The beta is 0.9137: an unmeasured prior."), record))
    assert any("numbers that are not in the record: 7919" in r for r in th.check(item(bull_case=[{"text": "Up 7919 percent", "ref": "[1]"}]), record))      # the bullets are checked too


def test_a_number_inside_a_ref_a_metric_name_or_a_date_is_not_a_number(record):
    assert th.check(item(bull_case=[{"text": "See [1] and rsi_14 and E2 on 2026-10-23.", "ref": "[1]"}]), record) == []
    assert any("2026-12-25" in r for r in th.check(item(thesis="Nothing happens before 2026-12-25: an unmeasured prior."), record))


def test_the_digits_of_an_identifier_are_not_numbers_of_the_record(record):
    """Found as a flaky test: a random record id that happened to contain 37 made 37 an allowed number."""
    tweaked = json.loads(json.dumps(record))
    tweaked["identity"]["record_id"], tweaked["identity"]["request_id"] = "123456789012", "8675309"
    tweaked["signals"][0]["chunk_id"], tweaked["audit"]["metric_order"] = "finnhub:3737#0", "3737"
    assert 123456789012.0 not in th.allowed_numbers(tweaked) and 3737.0 not in th.allowed_numbers(tweaked) and 8675309.0 not in th.allowed_numbers(tweaked)
    assert any("3737" in r for r in th.check(item(thesis="The id is 3737: an unmeasured prior."), tweaked))


def test_the_thesis_does_not_have_to_use_the_word_unmeasured_any_more(record):
    assert not any("unmeasured" in r for r in th.check(item(thesis="Open Procter & Gamble on its capital return."), record))


def test_counter_evidence_requires_a_bear_case_that_cites_it(record):
    with_counter = json.loads(json.dumps(record))
    with_counter["evidence"]["counter_evidence"] = [{"ref": "[2]", "direction": "bearish", "quote": "x"}]
    with_counter["evidence"]["signals"].append({"source": 2, "direction": "bearish", "quote": "x"})
    reasons = th.check(item(), with_counter)
    assert any("bear_case must cite at least one of them ([2])" in r for r in reasons)
    assert th.check(item(bear=None, bear_case=[{"text": "A bearish item exists.", "ref": "[2]"}]), with_counter) == []
    assert any("bear_case must cite" in r for r in th.check(item(bear_case=[{"text": "Unrelated.", "ref": "roe"}]), with_counter))


def test_a_malformed_thesis_is_refused(record):
    assert th.check({"symbol": "PG"}, record) == ["the thesis is empty"]
    assert th.check(item(bull_case="none"), record) == ["bull_case and bear_case must be lists of {text, ref}"]
    assert th.check(item(bull_case=[{"text": "x"}]), record) == ["bull_case and bear_case must be lists of {text, ref}"]


def write(replies, records):
    model = Scripted(*replies)
    run = new_run("thesis", model)
    return th.write_theses(model, records, run), run, model


def test_write_theses_takes_a_good_reply_in_one_call(record):
    theses, run, model = write([thesis_reply()], [record])
    assert theses["PG"].thesis.startswith("Open Procter") and theses["PG"].thesis_error is None and run["calls"] == 1 and len(model.seen) == 1
    assert "RECORD PG" in model.seen[0][1].content


def test_a_failed_thesis_gets_one_retry_with_the_reasons_and_a_good_second_reply_is_taken(record):
    bad = thesis_reply(thesis="Buy it at 37 dollars, an unmeasured prior.")
    theses, run, model = write([bad, thesis_reply()], [record])
    assert theses["PG"].thesis is not None and run["calls"] == 2
    retry = model.seen[1][-1].content
    assert "These theses failed the program's checks" in retry and "PG: the thesis has numbers that are not in the record: 37" in retry


def test_two_failures_leave_null_and_the_reasons_and_the_record_is_not_lost(record):
    bad = thesis_reply(thesis="Buy it at 37 dollars, an unmeasured prior.")
    theses, run, _ = write([bad, bad], [record])
    assert theses["PG"].thesis is None and "numbers that are not in the record: 37" in theses["PG"].thesis_error and run["calls"] == 2


def test_bad_json_is_retried_once_and_a_model_failure_is_recorded_not_raised(record):
    theses, run, _ = write(["not json at all", thesis_reply()], [record])
    assert theses["PG"].thesis is not None and run["calls"] == 2
    theses, run, _ = write([RuntimeError("down")], [record])
    assert theses["PG"].thesis is None and theses["PG"].thesis_error == "the thesis model call failed (RuntimeError)" and run["error"] == theses["PG"].thesis_error
    theses, run, _ = write(["junk", "junk"], [record])
    assert theses["PG"].thesis is None and "not usable" in theses["PG"].thesis_error


def test_a_symbol_the_model_leaves_out_is_reported(record):
    theses, _, _ = write([thesis_reply(symbol="XXXX"), thesis_reply(symbol="XXXX")], [record])
    assert theses["PG"].thesis is None and theses["PG"].thesis_error == "the model gave no thesis for it"


def test_no_records_means_no_call():
    theses, run, model = write([], [])
    assert theses == {} and model.seen == [] and run["calls"] == 0
