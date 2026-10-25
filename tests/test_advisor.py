"""Tests for prometheus.agents.advisor.advisor and prometheus.prompts.advisor (Part 8 of the RAG guide).

Run:  pytest tests/test_advisor.py -q

A scripted model stands in for the chat model, so each test decides exactly what the "Advisor model" replies (a good signal, an invented
quote, a quote that does not name the company, broken JSON). No test uses the network, a key or a paid API.
"""
import json
from datetime import date

import pytest
from langchain_core.messages import AIMessage

from prometheus.agents.advisor import advisor as adv
from prometheus.agents.data_clerk.agent import Result
from prometheus.common import work_order as wo
from prometheus.prompts import advisor as prompts
from prometheus.rag.citations import HEADER

HUANG = ("Memory Stocks Are Plummetting. Jensen Huang Says the Bears Have It Wrong.\n\nMemory chip stocks have declined roughly one-third "
         "from their July peak amid investor concerns about cyclical oversupply and rising debt yields. However, Nvidia CEO Jensen Huang "
         "argues the shortage won't ease until after 2029, and memory stocks trade at attractive valuations with extreme profit margins.")
CONTRACTS = ("Micron Has Locked In Its Revenue\n\nMicron has secured 35% of its revenue through long-term contracts, which reduces the "
             "cyclical risk of its memory business. Sandisk has secured 50%.")


def source(number, text, tickers=("MU",), headline="H"):
    return {"number": number, "chunk_id": f"art{number}#0", "article_id": f"art{number}", "chunk": 1, "chunks": 1, "headline": headline,
            "publisher": "The Motley Fool", "published_utc": "2026-10-23T10:00:00Z", "url": f"https://example.com/{number}",
            "tickers": list(tickers), "text": text, "score": 0.6}


def ready_order(*sources):
    order = wo.new_work_order("AI memory shortage. Which companies are affected, and what does the recent news say?", since_days=7, top_k=5)
    cited = [{"number": 1, "chunk_id": sources[0]["chunk_id"]}] if sources else []
    return wo.attach_result(order, Result(question="q", answer="Yes [1]." if sources else "Nothing found.", references=cited, sources=list(sources),
                                          no_supporting_text=not sources))


class Scripted:
    """Anything with invoke. Records each request and returns the next scripted reply."""
    model_name = "scripted/advisor"

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply, usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600})


def reply(*recommendations) -> str:
    return json.dumps({"recommendations": list(recommendations)})


def rec(symbol="MU", company="Micron", stance="bullish", model_score=70, signals=()):
    return {"symbol": symbol, "company": company, "stance": stance, "model_score": model_score, "signals": list(signals)}


def sig(source=1, direction="bullish", quote="Micron has secured 35% of its revenue through long-term contracts"):
    return {"source": source, "direction": direction, "quote": quote}


# ---------------------------------------------------------------- the pieces

def test_normalize_makes_the_models_typography_irrelevant():
    assert adv.normalize("Won\u2019t  EASE\u00a0until\u2011after \u201c2029\u201d") == 'won\'t ease until-after "2029"'


@pytest.mark.parametrize("quote, symbol, company, expected", [
    ("Micron has secured 35% of its revenue", "MU", "Micron Technology", True),
    ("MU stock rose", "MU", None, True),
    ("MU", "MU", "Micron", True),
    ("Memory chip stocks have declined roughly one-third", "MU", "Micron", False),
    ("a multiple of earnings", "MU", "Micron", False),                       # "mu" inside a word is not the ticker
    ("EMU Group shares fell on Monday", "MU", "Micron", False),              # nor is "MU" inside "EMU", even in capitals
    ("Shares of MU-listed funds rose", "MU", "Micron", True),                # but MU beside a hyphen is the ticker
    ("Bloom shipped fuel cells", "BE", "Bloom Energy", True),
    ("Energy prices rose", "BE", "Bloom Energy", False),                     # "Energy" is a generic word of a name
    ("SK Hynix leads in HBM", "SKHY", "SK Hynix", True),
    ("Nvidia CEO Jensen Huang argues", "MU", "Micron", False),               # a different company is not this one
    ("technology stocks fell", "MU", "Micron Technology", False),
])
def test_a_quote_names_a_company_by_its_ticker_or_a_distinctive_word_of_its_name(quote, symbol, company, expected):
    assert adv.names_company(quote, symbol, company) is expected


@pytest.mark.parametrize("bullish, bearish, score", [(0, 0, None), (1, 0, 67), (2, 0, 75), (3, 0, 80), (0, 1, 33), (0, 2, 25), (1, 1, 50), (2, 1, 60)])
def test_the_score_is_calculated_from_the_verified_signals(bullish, bearish, score):
    assert adv.calculate_score(bullish, bearish) == score


SOURCES = {1: source(1, CONTRACTS), 2: source(2, HUANG, ("MU", "NVDA"))}


@pytest.mark.parametrize("signal, sources, reason", [
    (sig(), {}, "no sources were provided"),
    (sig(source=9), SOURCES, "source 9 is not one of the sources shown"),
    (sig(source="x"), SOURCES, "source 'x' is not one of the sources shown"),
    (sig(source=True), SOURCES, "source True is not one of the sources shown"),
    (sig(direction="sideways"), SOURCES, "the direction must be bullish or bearish"),
    (sig(quote=""), SOURCES, "the quote is missing or too short to check"),
    (sig(quote="Micron"), SOURCES, "the quote is missing or too short to check"),
    (sig(quote="Micron has secured 90% of its revenue through spot sales"), SOURCES, "the quote is not in source 1"),
    (sig(source=2, quote="Memory chip stocks have declined roughly one-third from their July peak"), SOURCES, "the quote does not name the company"),
])
def test_each_way_a_signal_can_fail_gets_its_reason(signal, sources, reason):
    assert adv.verify_signal(signal, "MU", "Micron", sources) == (False, reason)


def test_a_real_quote_that_names_the_company_is_verified_even_with_different_typography_or_a_string_source_number():
    assert adv.verify_signal(sig(), "MU", "Micron", SOURCES) == (True, None)
    assert adv.verify_signal(sig(source="1", quote="MICRON  has secured 35% of its revenue\u00a0through long\u2011term contracts"), "MU", "Micron", SOURCES) == (True, None)


def test_a_quote_with_a_defused_bracket_is_found_in_the_source_the_model_read():
    text = "Micron said it expects [2] new fabs to open in 2027, the company said."  # the model read "(2)", the chunk holds "[2]"
    assert adv.verify_signal(sig(quote="Micron said it expects (2) new fabs to open in 2027"), "MU", "Micron", {1: source(1, text)}) == (True, None)
    assert adv.verify_signal(sig(quote="Micron said it expects [2] new fabs to open in 2027"), "MU", "Micron", {1: source(1, text)}) == (True, None)


@pytest.mark.parametrize("text", ['{"recommendations": []}', '```json\n{"recommendations": []}\n```', 'Here you go: {"recommendations": []} Done.'])
def test_the_reply_is_found_in_plain_json_a_code_fence_or_chatter(text):
    assert adv.parse_reply(text) == []


@pytest.mark.parametrize("text, problem", [("", "no JSON object found"), ("no json here", "no JSON object found"), ("{not json}", "not valid JSON"),
                                            ('{"other": 1}', "no recommendations list"), ('{"recommendations": "x"}', "no recommendations list")])
def test_a_reply_without_a_recommendations_list_is_refused_with_the_reason(text, problem):
    with pytest.raises(ValueError, match=problem):
        adv.parse_reply(text)


# ---------------------------------------------------------------- one recommendation

def test_the_score_comes_from_verified_signals_only_and_the_models_own_number_is_recorded_but_never_used():
    raw = rec(model_score=95, signals=[sig(), sig(direction="bullish", quote="Micron invented a quote that is not there at all"),
                                       sig(source=2, direction="bearish", quote="Memory chip stocks have declined roughly one-third")])
    r = adv.build_recommendation(raw, SOURCES, [])
    assert r["signals_claimed"] == 3 and r["signals_verified"] == 1 and (r["bullish_signals"], r["bearish_signals"]) == (1, 0)
    assert r["score"] == 67 and r["model_score"] == 95 and r["supported"] and r["consistent"] is True  # stance bullish, score on the bullish side
    assert [s["verified"] for s in r["signals"]] == [True, False, False]
    assert [s["reason"] for s in r["signals"]] == [None, "the quote is not in source 1", "the quote does not name the company"]
    assert r["signals"][0]["chunk_id"] == "art1#0" and r["signals"][1]["chunk_id"] is None


def test_the_models_own_score_does_not_change_the_calculated_one():
    low = adv.build_recommendation(rec(model_score=5, signals=[sig()]), SOURCES, [])
    high = adv.build_recommendation(rec(model_score=99, signals=[sig()]), SOURCES, [])
    assert low["score"] == high["score"] == 67 and (low["model_score"], high["model_score"]) == (5, 99)


def test_a_recommendation_with_no_verified_signal_has_no_score_however_confident_the_model_was():
    r = adv.build_recommendation(rec(model_score=90, stance="bullish", signals=[sig(quote="Micron will double its revenue next year for sure")]), SOURCES, [])
    assert r["score"] is None and not r["supported"] and r["consistent"] is None and r["model_score"] == 90 and r["signals_verified"] == 0
    assert adv.build_recommendation(rec(signals=[]), SOURCES, [])["score"] is None


def test_the_stretched_sentence_does_not_count_a_quote_about_memory_stocks_for_micron():
    """The failure measured in the week 3 re-run: the chunk says 'memory chip stocks' and names no company."""
    r = adv.build_recommendation(rec(signals=[{"source": 2, "direction": "bearish", "quote": "Memory chip stocks have declined roughly one-third from their July peak"}]),
                                 SOURCES, [])
    assert r["score"] is None and r["signals"][0]["reason"] == "the quote does not name the company"


def test_the_stance_is_checked_against_the_calculated_score():
    assert adv.build_recommendation(rec(stance="bearish", signals=[sig()]), SOURCES, [])["consistent"] is False
    assert adv.build_recommendation(rec(stance="bullish", signals=[sig()]), SOURCES, [])["consistent"] is True
    sources = {**SOURCES, 3: source(3, "Micron faces rising debt yields and the risk of oversupply as new capacity arrives.")}
    both = [sig(), sig(source=3, direction="bearish", quote="Micron faces rising debt yields and the risk of oversupply")]
    balanced = adv.build_recommendation(rec(stance="neutral", signals=both), sources, [])
    assert (balanced["bullish_signals"], balanced["bearish_signals"], balanced["score"], balanced["consistent"]) == (1, 1, 50, True)
    assert adv.build_recommendation(rec(stance="bullish", signals=both), sources, [])["consistent"] is False  # 50 is neutral, not bullish


def test_the_same_signal_twice_counts_once_and_no_more_than_three_are_read():
    r = adv.build_recommendation(rec(signals=[sig(), sig(), sig(direction="bearish")]), SOURCES, [])
    assert r["signals_claimed"] == 2  # the repeat is dropped; the bearish copy of the same words is a different signal
    many = [sig(quote=f"Micron has secured 35% of its revenue through long-term contracts{'!' * i}") for i in range(6)]
    assert adv.build_recommendation(rec(signals=many), SOURCES, [])["signals_claimed"] <= 3


@pytest.mark.parametrize("raw, note", [("not a dict", "without a symbol"), ({"company": "x"}, "without a symbol"), ({"symbol": "../etc"}, "invalid symbol"),
                                       ({"symbol": 7}, "without a symbol")])
def test_a_recommendation_that_cannot_be_used_is_dropped_with_a_note(raw, note):
    dropped = []
    assert adv.build_recommendation(raw, SOURCES, dropped) is None and note in dropped[0]


def test_a_symbol_is_cleaned_and_odd_fields_become_none_not_errors():
    r = adv.build_recommendation({"symbol": " $mu ", "company": 7, "stance": "maybe", "model_score": 250, "signals": "x"}, SOURCES, [])
    assert r["symbol"] == "MU" and r["company"] is None and r["stance"] is None and r["model_score"] is None and r["signals_claimed"] == 0
    assert adv.build_recommendation({"symbol": "MU", "model_score": True}, SOURCES, [])["model_score"] is None


# ---------------------------------------------------------------- the Advisor

def test_with_evidence_the_advisor_reads_the_chunks_and_writes_a_recommendation_with_a_calculated_score():
    order = ready_order(source(1, CONTRACTS), source(2, HUANG, ("MU", "NVDA")))
    model = Scripted(reply(rec(signals=[sig()])))
    same = adv.Advisor(model=model).advise(order, today=date(2026, 10, 25))
    assert same is order and order.status == "recommended" and order.errors == []
    r = order.recommendations[0]
    assert (r["symbol"], r["score"], r["supported"], r["signals_verified"]) == ("MU", 67, True, 1) and r["advisor"] == {"model": "scripted/advisor", "evidence": True}
    system, human = model.seen[0]
    assert "Today is 2026-10-25" in system.content and HEADER in human.content
    assert "[1] The Motley Fool, 2026-10-23, MU, relevance 0.60\nMicron Has Locked In Its Revenue" in human.content and "[2] The Motley Fool" in human.content
    run = order.advisor_runs[0]
    assert (run["evidence"], run["sources"], run["calls"], run["tokens_in"], run["tokens_out"], run["error"]) == (True, 2, 1, 500, 100, None)


def test_the_advisor_reads_the_chunks_and_not_the_data_clerks_summary():
    order = ready_order(source(1, CONTRACTS))
    order.data.answer = "A SUMMARY THE DATA CLERK WROTE THAT MUST NOT REACH THE ADVISOR [1]."
    model = Scripted(reply())
    adv.Advisor(model=model).advise(order)
    assert "A SUMMARY THE DATA CLERK WROTE" not in model.seen[0][1].content


def test_without_evidence_the_advisor_is_told_there_are_no_sources_and_no_signal_can_count():
    order = wo.new_work_order("AI memory shortage. Which companies are affected?")
    model = Scripted(reply(rec(model_score=88, signals=[sig(quote="Micron said it will raise memory prices by ten percent next quarter")])))
    adv.Advisor(model=model).advise(order, use_evidence=False)
    assert "Sources: none were retrieved for this work order." in model.seen[0][1].content and HEADER not in model.seen[0][1].content
    r = order.recommendations[0]
    assert order.status == "recommended" and r["score"] is None and not r["supported"] and r["model_score"] == 88 and r["advisor"]["evidence"] is False
    assert r["signals"][0]["reason"] == "no sources were provided" and order.advisor_runs[0]["sources"] == 0


def test_use_evidence_false_ignores_data_that_is_on_the_order():
    order = ready_order(source(1, CONTRACTS))
    model = Scripted(reply(rec(signals=[sig()])))
    adv.Advisor(model=model).advise(order, use_evidence=False)
    assert "Sources: none were retrieved" in model.seen[0][1].content and order.recommendations[0]["score"] is None


@pytest.mark.parametrize("status", ["new", "data_failed"])
def test_with_evidence_the_advisor_refuses_an_order_whose_data_is_not_ready_and_does_not_call_the_model(status):
    order = wo.new_work_order("q")
    order.status = status
    model = Scripted()  # any call would raise
    adv.Advisor(model=model).advise(order)
    assert order.errors == [f"the Advisor needs an order whose data is ready (status {status})"] and model.seen == [] and order.recommendations == []
    assert order.status == status


def test_a_reply_that_is_not_json_gets_one_retry_and_the_retry_is_used():
    order = ready_order(source(1, CONTRACTS))
    model = Scripted("I think Micron looks great!", reply(rec(signals=[sig()])))
    adv.Advisor(model=model).advise(order)
    assert order.status == "recommended" and order.recommendations[0]["score"] == 67 and order.advisor_runs[0]["calls"] == 2
    assert "not valid JSON in the required shape (no JSON object found)" in model.seen[1][-1].content
    assert order.advisor_runs[0]["tokens_in"] == 1000 and order.advisor_runs[0]["error"] is None


def test_two_unusable_replies_leave_the_order_as_it_was_with_the_reason_on_it():
    order = ready_order(source(1, CONTRACTS))
    adv.Advisor(model=Scripted("nope", "still nope")).advise(order)
    assert order.status == "data_ready" and order.recommendations == [] and order.errors == ["the advisor's reply was not usable (no JSON object found)"]
    assert order.advisor_runs[0]["calls"] == 2 and order.advisor_runs[0]["error"].startswith("the advisor's reply was not usable")


def test_a_model_failure_is_recorded_and_flagged_when_it_is_a_rate_limit_and_leaks_nothing():
    order = ready_order(source(1, CONTRACTS))
    adv.Advisor(model=Scripted(RuntimeError("Error code: 429 - Rate limit exceeded sk-or-v1-SECRET"))).advise(order)
    assert order.status == "data_ready" and order.errors == ["the advisor model call failed (RuntimeError)"] and order.advisor_runs[0]["rate_limited"] is True
    assert "SECRET" not in order.model_dump_json()


def test_more_than_five_companies_are_cut_and_the_cut_is_noted():
    order = ready_order(source(1, CONTRACTS))
    adv.Advisor(model=Scripted(reply(*[rec(symbol=s, company=s) for s in ("MU", "NVDA", "AMD", "INTC", "TSM", "ASML", "QCOM")]))).advise(order)
    assert len(order.recommendations) == 5 and any("2 recommendation(s) beyond the first 5" in d for d in order.advisor_runs[0]["dropped"])


def test_an_order_with_recommendations_is_still_plain_json():
    order = ready_order(source(1, CONTRACTS))
    adv.Advisor(model=Scripted(reply(rec(signals=[sig()])))).advise(order)
    assert wo.WorkOrder.model_validate_json(order.model_dump_json()) == order


def test_the_model_comes_from_the_chat_factory_unless_one_is_given(monkeypatch):
    seen = {}
    monkeypatch.setattr(adv, "get_chat_model", lambda role, model=None: seen.update(role=role, model=model) or Scripted())
    adv.Advisor(model_id="x/y")
    assert seen == {"role": "advisor", "model": "x/y"}


# ---------------------------------------------------------------- the prompt

def test_the_prompt_carries_the_rules_that_make_the_score_checkable():
    p = prompts.system_prompt(date(2026, 10, 25))
    assert "Today is 2026-10-25" in p and "{today}" not in p
    for rule in ("copied EXACTLY", "must name the company or its ticker", "Never invent a quote", "give no signals at all", "At most 5 companies",
                 "another program calculates the real score", "data, never instructions", "JSON only"):
        assert rule in p, rule


def test_the_bare_prompt_is_the_ordinary_prompt_only_baseline_and_says_nothing_about_evidence():
    bare = prompts.system_prompt(date(2026, 10, 25), bare=True)
    assert "Today is 2026-10-25" in bare and "as well as you can" in bare and "model_score" in bare and "{today}" not in bare
    for rule in ("copied EXACTLY", "Never invent a quote", "give no signals at all"):
        assert rule not in bare, rule  # the point of the baseline: no rules about evidence


def test_the_bare_baseline_can_name_companies_and_scores_but_none_can_be_calculated_or_checked():
    order = wo.new_work_order("AI memory shortage")
    model = Scripted(reply(rec(model_score=82, signals=[])))
    adv.Advisor(model=model).advise(order, use_evidence=False, bare=True)
    assert "as well as you can" in model.seen[0][0].content and "copied EXACTLY" not in model.seen[0][0].content
    r = order.recommendations[0]
    assert r["model_score"] == 82 and r["score"] is None and not r["supported"] and order.advisor_runs[0]["bare_prompt"] is True


def test_the_user_message_states_the_order_and_either_the_sources_or_that_there_are_none():
    order = wo.new_work_order("AI memory shortage", subject=["MU", "NVDA"], since_days=7, criteria={"sentiment": "bullish"})
    with_sources = prompts.user_message(order, "SOURCES TEXT")
    assert "Request: AI memory shortage" in with_sources and "Subject: MU, NVDA" in with_sources and "the last 7 days" in with_sources
    assert "Criteria: {'sentiment': 'bullish'}" in with_sources and with_sources.endswith("SOURCES TEXT")
    assert prompts.user_message(order, None).endswith("Sources: none were retrieved for this work order.")
    assert "none given" in prompts.user_message(wo.new_work_order("q"), None)
