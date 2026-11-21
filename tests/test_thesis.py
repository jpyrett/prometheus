"""Tests for prometheus.agents.advisor.thesis (Advisor v1 spec section 11.2, phase 6): the digest, each check on its own, the retry once, and `thesis: null` after two failures.

The record under test is the one the whole step writes for PG with the fakes of advisor_helpers (the same record the Critic will read)."""
import json

import pytest
from advisor_helpers import PG_EVENT, PG_SIGNAL, TODAY, Scripted, advisor_with, company, plan_reply, ready_order, thesis_reply

from prometheus.agents.advisor import thesis as th
from prometheus.agents.advisor.calls import new_run


@pytest.fixture(scope="module")
def record():
    advisor, _ = advisor_with(plan_reply(company(signals=[PG_SIGNAL], events=[PG_EVENT])), thesis_reply())
    order = advisor.generate_recommendations(ready_order(), "funds", TODAY)
    return order.recommendations[0]


def item(**kw):
    base = {"symbol": "PG", "thesis": "Open Procter & Gamble on its capital return: the expected excess return is an unmeasured prior.", "bull_case": [{"text": "It lifted its outlook.", "ref": "[1]"}], "bear_case": []}
    base.update(kw)
    return base


def test_at_most_three_sentences(record):
    four = "One. Two. Three. Four is an unmeasured prior."
    assert th.check(item(thesis=four), record) == ["the thesis has 4 sentences (at most 3)"]
    assert th.check(item(thesis="ROE is 20.0 and that is an unmeasured prior. Two. Three."), record) == []                  # a decimal point is not a sentence end
    assert len(th.sentences("Price is 163.45 per share. It is up 4.2% today! Fine?")) == 3


def test_a_number_the_record_does_not_have_fails_and_a_rounded_one_passes(record):
    assert any("numbers that are not in the record: 37" in r for r in th.check(item(thesis="Buy it at 37 dollars: an unmeasured prior."), record))
    assert th.check(item(thesis="ROE is 20 and the price is $100.00, an unmeasured prior."), record) == []                      # 20 and 100.0 are in the record
    assert th.check(item(thesis="The beta is 0.7 and the cost is 0.20%: an unmeasured prior."), record) == []                    # 0.7 is beta_63d; 0.002 is the cost, shown as a percent
    assert th.check(item(thesis="The cash balance is 10 billion: an unmeasured prior."), record) == []                             # 1e10 shown in billions
    assert any("0.9137" in r for r in th.check(item(thesis="The beta is 0.9137: an unmeasured prior."), record))
    assert any("numbers that are not in the record: 7919" in r for r in th.check(item(bull_case=[{"text": "Up 7919 percent", "ref": "[1]"}]), record))      # the bullets are checked too


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


def test_no_records_means_no_call():
    theses, run, model = write([], [])
    assert theses == {} and model.seen == [] and run["calls"] == 0
